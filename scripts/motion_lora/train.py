"""Motion-LoRA fine-tune on Wan2.1-T2V-1.3B base.

v1 design (synthesized from MotionMatcher / rCM / "Distilling Diversity & Control"):

  - Frozen: Wan2.1-T2V-1.3B base (no LongLive LoRA loaded)
  - Trainable: LoRA on blocks 10/11/12, self_attn + cross_attn, q/k/v/o
                rank=32, alpha=32 (24 modules total)
  - Loss: 1.0 * L_attention + 0.01 * L_anchor + 0.1 * L_pixel
      L_attention: L2 distance on **attention module outputs** at blocks 10/11/12.
                   Substitutes MotionMatcher's attention-map L2 because Wan uses
                   fused flash_attention with no map materialization.
      L_anchor:    rCM analog. On unrelated prompts, force LoRA-on output to stay
                   close to LoRA-off (prevent drift on out-of-domain prompts).
      L_pixel:     vanilla flow MSE on reference video (small weight, anchor).
  - Timesteps: sample t in [500, 999] (Wan FlowMatch uses 1000-step training schedule;
               the upper half is "structural" / high-noise — MotionMatcher's w'_t mask).
  - Augmentation: pixel-level random crop + color jitter on reference video, re-encoded
                  via VAE; pre-compute K variants at startup, sample each step.
  - Output: peft state dict at output_dir/motion_lora.pt — pluggable into base Wan
            via configure_lora_for_model + set_peft_model_state_dict.

Two modes — single reference or multi-reference (same motion class):

  Single ref (instance-level — locks this exact video's motion):
    python scripts/motion_lora/train.py \\
        --config configs/motion_lora.yaml \\
        --reference_video celebv_X.mp4 --reference_caption "a man walking ..." \\
        --anchor_prompts scripts/motion_lora/anchor_prompts_default.txt \\
        --output_dir logs/motion_lora_walking_v1

  Multi-ref (class-level — learns generic motion class with stronger appearance
  disentanglement; recommended after single-ref smoke validates pipeline):
    python scripts/motion_lora/pick_reference.py --keyword walking ... \\
        --output refs.jsonl
    python scripts/motion_lora/train.py \\
        --config configs/motion_lora.yaml \\
        --references_jsonl refs.jsonl \\
        --anchor_prompts scripts/motion_lora/anchor_prompts_default.txt \\
        --output_dir logs/motion_lora_walking_v2

  references_jsonl format: one JSON object per line, with keys 'path' (or 'video',
  in which case it's resolved against $LL_DATA/motion_refs) and 'caption'.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

# Repo root on sys.path so `from utils...` and `from wan...` resolve.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import peft  # noqa: E402

from utils.wan_wrapper import (  # noqa: E402
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
# Single-GPU and multi-GPU (DDP) modes share this code. torchrun sets
# LOCAL_RANK/RANK/WORLD_SIZE env vars; their absence == single-GPU.
# Each rank loads its own copy of the model, VAE, text_encoder, and variant
# cache. Per step, each rank samples its own variant + anchor prompt; DDP
# all-reduces gradients on LoRA params. Effective batch size = world_size.
# ---------------------------------------------------------------------------

def init_distributed():
    """Returns (rank, world_size, device, is_main). Idempotent."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ.get("RANK", str(local_rank)))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda")
    return rank, world_size, device, rank == 0


def log(msg: str, is_main: bool = True):
    if is_main:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# motion-LoRA target_modules: blocks 10/11/12, self_attn + cross_attn, q/k/v/o.
# Wan2.1-T2V-1.3B has 32 transformer blocks; middle 1/3 = blocks 10-12. Rationale:
# MotionMatcher targets UNet down_block.2 (the bottleneck); on a flat 32-layer DiT
# the analogous depth is ~1/3 in. We don't include `ffn` because MotionMatcher
# doesn't and we want to keep the LoRA scope narrow (architectural appearance
# disentanglement — see "Separate Motion from Appearance" arXiv 2501.16714).
# ---------------------------------------------------------------------------
MOTION_LORA_BLOCKS = [10, 11, 12]
MOTION_LORA_SUBMODULES = ["self_attn", "cross_attn"]
MOTION_LORA_PROJS = ["q", "k", "v", "o"]


def build_target_modules() -> list[str]:
    """Return the 24 module names peft should attach LoRA to."""
    targets = []
    for b in MOTION_LORA_BLOCKS:
        for sub in MOTION_LORA_SUBMODULES:
            for proj in MOTION_LORA_PROJS:
                targets.append(f"blocks.{b}.{sub}.{proj}")
    return targets


def inject_motion_lora(transformer: torch.nn.Module, rank: int, alpha: int):
    """Wrap `transformer` (a WanModel) with a peft LoRA adapter restricted to our
    target_modules. The base weights stay frozen; only LoRA A/B matrices train."""
    target_modules = build_target_modules()
    cfg = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        # Don't bias / don't init lora_B random — peft default is good (A=Kaiming, B=zero).
    )
    return peft.get_peft_model(transformer, cfg)


# ---------------------------------------------------------------------------
# Attention-output hooks
# ---------------------------------------------------------------------------
# Wan uses fused flash_attention (wan/modules/attention.py) which produces a
# fused softmax(QK^T) V output without materialising the QK^T attention map.
# MotionMatcher's loss is L2 on the attention map; we substitute L2 on the
# **attention module output** (post softmax(QK^T)V, before the final `o` projection).
# This carries equivalent motion-binding information and is computable.
# We hook the OUTPUT of `self_attn` and `cross_attn` modules in target blocks.
# ---------------------------------------------------------------------------

class AttentionCapture:
    """Forward-hook helper. Stores attn outputs in a dict keyed by module path."""

    def __init__(self):
        self.outputs: dict[str, torch.Tensor] = {}
        self._handles = []
        self._enabled = True

    def install(self, model: torch.nn.Module, blocks: list[int],
                submodules: list[str]) -> None:
        for b in blocks:
            for sub in submodules:
                # peft wraps modules with LoRA — the actual `self_attn` module
                # under PeftModel is at base_model.model.blocks.B.self_attn.
                # We attach the hook at the underlying module so it fires
                # regardless of LoRA on/off state.
                key = f"blocks.{b}.{sub}"
                module = self._resolve(model, key)
                handle = module.register_forward_hook(self._make_hook(key))
                self._handles.append(handle)

    def _resolve(self, model: torch.nn.Module, dotted_path: str) -> torch.nn.Module:
        cur = model
        # peft wraps as PeftModel(base_model=LoraModel(model=WanModel))
        # walk through if needed.
        for prefix in ("base_model.model", "base_model", ""):
            try:
                root = cur
                if prefix:
                    for part in prefix.split("."):
                        root = getattr(root, part)
                obj = root
                for part in dotted_path.split("."):
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise AttributeError(f"Could not resolve {dotted_path} on {type(model).__name__}")

    def _make_hook(self, key: str):
        def hook(module, inputs, output):
            if self._enabled:
                self.outputs[key] = output
        return hook

    def clear(self):
        self.outputs.clear()

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Reference video loader + augmentation
# ---------------------------------------------------------------------------

def load_video_pixels(path: str, num_frames: int, height: int, width: int,
                      device: torch.device) -> torch.Tensor:
    """Load `num_frames` frames from `path`, resized to (height, width).
    Returns tensor [C, F, H, W] in [-1, 1] bfloat16 on `device`.
    """
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(path)
    total = len(vr)
    if total < num_frames:
        raise ValueError(f"Video {path} has {total} frames, need {num_frames}")
    # Uniformly sample `num_frames` frames across the whole clip.
    indices = torch.linspace(0, total - 1, num_frames).long().tolist()
    frames = vr.get_batch(indices)  # [F, H, W, C], uint8, torch.Tensor
    frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [F, C, H, W], [0,1]
    frames = transforms.functional.resize(
        frames, [height, width], antialias=True
    )
    frames = frames * 2.0 - 1.0  # [-1, 1]
    frames = frames.permute(1, 0, 2, 3).contiguous()  # [C, F, H, W]
    return frames.to(device=device, dtype=torch.bfloat16)


def make_augmented_variants(reference: torch.Tensor, n_variants: int,
                            crop_min: float = 0.85,
                            jitter_brightness: float = 0.15,
                            jitter_contrast: float = 0.15) -> list[torch.Tensor]:
    """Produce `n_variants` augmented copies of `reference` (shape [C, F, H, W]).

    Augmentations: random spatial crop (resized back), color jitter. NO temporal
    crop or temporal flip — would corrupt the dynamics signal which is exactly
    what we want LoRA to learn.
    """
    C, F, H, W = reference.shape
    out = []
    for _ in range(n_variants):
        # Random crop scale in [crop_min, 1.0]
        scale = random.uniform(crop_min, 1.0)
        ch, cw = int(H * scale), int(W * scale)
        ty = random.randint(0, H - ch)
        tx = random.randint(0, W - cw)
        crop = reference[:, :, ty:ty + ch, tx:tx + cw]
        # Resize each frame back to (H, W). Treat F as batch.
        crop_bf = rearrange(crop, "c f h w -> f c h w")
        crop_resized = transforms.functional.resize(
            crop_bf.float(), [H, W], antialias=True
        ).to(reference.dtype)
        crop_resized = rearrange(crop_resized, "f c h w -> c f h w")

        # Color jitter (apply same shift to all frames so motion isn't disturbed)
        b = 1.0 + random.uniform(-jitter_brightness, jitter_brightness)
        c = 1.0 + random.uniform(-jitter_contrast, jitter_contrast)
        # bring [-1, 1] → [0, 1] for jitter math, then back
        x = (crop_resized + 1.0) * 0.5
        x = x * b
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * c + mean
        x = x.clamp(0.0, 1.0)
        x = x * 2.0 - 1.0
        out.append(x.to(reference.dtype))
    return out


# ---------------------------------------------------------------------------
# Timestep sampling
# ---------------------------------------------------------------------------

def sample_structural_timestep(batch_size: int, t_min: int, t_max: int,
                               device: torch.device) -> torch.Tensor:
    """Uniform int in [t_min, t_max] (training-time int timesteps in Wan's 0..999
    convention). MotionMatcher's w'_t mask = "first 500 timesteps in DDPM order"
    = high-noise / structural in flow-match: t in [500, 999]. We restrict to
    that range.
    """
    return torch.randint(t_min, t_max + 1, [batch_size], device=device, dtype=torch.long)


# ---------------------------------------------------------------------------
# Forward helper
# ---------------------------------------------------------------------------

def diffusion_forward(diffusion: WanDiffusionWrapper,
                      latent: torch.Tensor,
                      noise: torch.Tensor,
                      timestep_int: torch.Tensor,
                      text_emb: torch.Tensor):
    """Run one rectified-flow forward through the (LoRA-wrapped or raw) model.

    Returns (flow_pred, flow_target). flow_pred = predicted velocity.
    flow_target = noise - x0 (the rectified-flow training target — see
    utils/loss.py:FlowPredLoss).

    `latent` and `noise` are [B, F, C, H, W]. `timestep_int` is [B] long in
    0..999; the wrapper expects [B, F]. xt is built via scheduler.add_noise,
    which handles the FlowMatch shift correctly (don't manually do
    (1-sigma)*x + sigma*noise — sigmas are shifted, not linear in t).
    """
    B, num_frames, C, H, W = latent.shape
    sched = diffusion.scheduler
    timestep_per_frame = timestep_int.view(B, 1).expand(B, num_frames)  # [B, F]

    # scheduler.add_noise expects flattened [N, C, H, W] + [N] timesteps
    latent_flat = latent.reshape(B * num_frames, C, H, W)
    noise_flat = noise.reshape(B * num_frames, C, H, W)
    timestep_flat = timestep_per_frame.reshape(B * num_frames).float()
    xt_flat = sched.add_noise(latent_flat, noise_flat, timestep_flat)
    xt = xt_flat.reshape(B, num_frames, C, H, W).to(latent.dtype)

    cond = {"prompt_embeds": text_emb}
    flow_pred, _ = diffusion(xt, cond, timestep_per_frame)
    flow_target = noise - latent  # velocity = noise - x0 (rectified flow)
    return flow_pred, flow_target


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_references(args) -> list[dict]:
    """Returns a list of {'path': abs_path, 'caption': str}. Supports two modes:

    Single ref: --reference_video + --reference_caption (legacy / smoke test mode).
    Multi-ref:  --references_jsonl (each line has 'path' or 'video' + 'caption').

    For 'video' (no 'path'), the caller's pick_reference output convention is
    used: resolve against $LL_DATA/motion_refs first, else $LL_DATA, else cwd.
    """
    refs = []
    if args.references_jsonl:
        ll_data = os.environ.get("LL_DATA", "")
        with open(args.references_jsonl) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = __import__("json").loads(line)
                cap = row.get("caption")
                if not cap:
                    raise ValueError(
                        f"{args.references_jsonl}:{line_no} missing 'caption'")
                p = row.get("path")
                if not p:
                    name = row.get("video")
                    if not name:
                        raise ValueError(
                            f"{args.references_jsonl}:{line_no} needs 'path' or 'video'")
                    cands = [
                        Path(ll_data) / "motion_refs" / name,
                        Path(ll_data) / name,
                        Path(name),
                    ] if ll_data else [Path(name)]
                    p = next((c for c in cands if c.exists()), None)
                    if p is None:
                        raise FileNotFoundError(
                            f"{name} not found in any of {cands}")
                refs.append({"path": str(p), "caption": cap})
    elif args.reference_video and args.reference_caption:
        refs.append({"path": args.reference_video,
                     "caption": args.reference_caption})
    else:
        raise ValueError(
            "must give either --references_jsonl OR (--reference_video + --reference_caption)"
        )
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reference_video", default=None,
                    help="single-reference mode: path to one mp4")
    ap.add_argument("--reference_caption", default=None,
                    help="single-reference mode: caption for --reference_video")
    ap.add_argument("--references_jsonl", default=None,
                    help="multi-reference mode: jsonl with {path|video, caption} per line. "
                         "Mutually exclusive with --reference_video. "
                         "pick_reference.py output is directly compatible.")
    ap.add_argument("--anchor_prompts", required=True,
                    help="txt file: one unrelated prompt per line")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(args.output_dir)

    rank, world_size, device, is_main = init_distributed()

    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out_dir / "config.yaml")
    if world_size > 1:
        dist.barrier()

    # Per-rank seed: same base seed + rank offset → different sample paths but
    # reproducible. Necessary so DDP ranks visit different variants per step
    # (otherwise effective batch = 1).
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    log(f"[motion-lora] world_size={world_size} rank={rank} device={device}", is_main)

    # ---- 1. Load Wan2.1-T2V-1.3B base ------------------------------------
    log("[motion-lora] loading Wan2.1-T2V-1.3B base ...", is_main)
    diffusion = WanDiffusionWrapper(
        model_name="Wan2.1-T2V-1.3B", is_causal=False,
        timestep_shift=cfg.get("timestep_shift", 8.0),
    )
    diffusion.model.requires_grad_(False)
    diffusion.to(device=device, dtype=torch.bfloat16)

    text_encoder = WanTextEncoder().to(device)
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)

    # ---- 2. Inject motion LoRA -------------------------------------------
    log(f"[motion-lora] injecting LoRA: blocks={MOTION_LORA_BLOCKS} "
        f"submodules={MOTION_LORA_SUBMODULES} projs={MOTION_LORA_PROJS} "
        f"rank={cfg.lora.rank} alpha={cfg.lora.alpha}", is_main)
    diffusion.model = inject_motion_lora(
        diffusion.model, rank=cfg.lora.rank, alpha=cfg.lora.alpha
    )
    if is_main:
        diffusion.model.print_trainable_parameters()

    # Keep an unwrapped reference to the PeftModel so `disable_adapter()` and
    # `get_peft_model_state_dict()` work uniformly with or without DDP.
    peft_model = diffusion.model

    # ---- 3. Install attention hooks (BEFORE DDP wrap) -------------------
    # Hooks live on the underlying nn.Module and fire regardless of DDP.
    capture = AttentionCapture()
    capture.install(diffusion.model, MOTION_LORA_BLOCKS, MOTION_LORA_SUBMODULES)
    log(f"[motion-lora] attention hooks installed at "
        f"{len(MOTION_LORA_BLOCKS) * len(MOTION_LORA_SUBMODULES)} sites",
        is_main)

    # Wrap with DDP after hooks + LoRA are in place. find_unused_parameters=False
    # is fine because every LoRA module sees grad each forward (q/k/v/o all
    # visited in self-attn + cross-attn paths).
    if world_size > 1:
        diffusion.model = DDP(
            diffusion.model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )

    # ---- 4. Resolve and encode all references ---------------------------
    refs = _resolve_references(args)
    log(f"[motion-lora] {len(refs)} reference(s):", is_main)
    if is_main:
        for r in refs:
            log(f"  - {Path(r['path']).name}  '{r['caption'][:80]}...'", True)

    H = cfg.data.height
    W = cfg.data.width
    F_pixels = cfg.data.num_frames
    n_variants = cfg.data.n_augment_variants

    log(f"[motion-lora] loading + augmenting + encoding via VAE "
        f"({len(refs)} refs × {n_variants} variants) ...", is_main)
    variants_latent: list[torch.Tensor] = []
    variants_ref_idx: list[int] = []
    text_embs: list[torch.Tensor] = []
    for ref_idx, ref in enumerate(refs):
        pixels = load_video_pixels(ref["path"], F_pixels, H, W, device)
        variants_pixel = make_augmented_variants(pixels, n_variants)
        for v in variants_pixel:
            with torch.no_grad():
                lat = vae.encode_to_latent(v.unsqueeze(0))
            variants_latent.append(lat)
            variants_ref_idx.append(ref_idx)
        with torch.no_grad():
            text_embs.append(
                text_encoder([ref["caption"]])["prompt_embeds"]
            )
    log(f"[motion-lora] total {len(variants_latent)} variants, "
        f"latent shape {variants_latent[0].shape}", is_main)

    # ---- 5. Encode anchor prompts ---------------------------------------

    anchor_texts = [
        line.strip() for line in open(args.anchor_prompts) if line.strip()
    ]
    if not anchor_texts:
        raise ValueError(f"no anchor prompts in {args.anchor_prompts}")
    log(f"[motion-lora] caching {len(anchor_texts)} anchor prompt embeddings ...",
        is_main)
    anchor_embs = []
    with torch.no_grad():
        for ap_text in anchor_texts:
            anchor_embs.append(
                text_encoder([ap_text])["prompt_embeds"].cpu()
            )

    # ---- 6. Optimizer ----------------------------------------------------
    # Optimize the LoRA params on the unwrapped peft_model (DDP wrapper exposes
    # the same parameters but routing through .module is more explicit).
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        weight_decay=cfg.optim.weight_decay,
    )

    # ---- 7. Training loop -----------------------------------------------
    L_ATT_W = cfg.loss.attention_weight  # 1.0
    L_ANC_W = cfg.loss.anchor_weight     # 0.01
    L_PIX_W = cfg.loss.pixel_weight      # 0.1
    T_MIN = cfg.loss.timestep_min        # 500
    T_MAX = cfg.loss.timestep_max        # 999
    ANCHOR_EVERY = cfg.loss.anchor_every  # 2 (anchor every Nth step)
    LOG_EVERY = cfg.train.log_every

    peft_model.train()
    log(f"[motion-lora] beginning training: {cfg.train.steps} steps "
        f"(world_size={world_size} → effective batch={world_size})", is_main)

    for step in range(cfg.train.steps):
        # ---- on-reference branch ----
        # Each rank picks its own variant (different RNG seed) → DDP all-reduce
        # gives effective batch = world_size.
        var_idx = random.randint(0, len(variants_latent) - 1)
        x0 = variants_latent[var_idx].to(device=device, dtype=torch.bfloat16)
        cur_text_emb = text_embs[variants_ref_idx[var_idx]]
        B = x0.shape[0]
        eps = torch.randn_like(x0)
        t = sample_structural_timestep(B, T_MIN, T_MAX, device)

        # LoRA OFF (frozen base) — capture reference attention outputs.
        # disable_adapter() lives on PeftModel; works whether DDP-wrapped or not
        # because forward dispatch goes ddp_model.forward → module.forward → peft.
        capture.clear()
        capture.enable()
        with peft_model.disable_adapter():
            with torch.no_grad():
                _ = diffusion_forward(diffusion, x0, eps, t, cur_text_emb)
        attn_ref = {k: v.detach() for k, v in capture.outputs.items()}

        # LoRA ON — gradients flow
        capture.clear()
        flow_pred, flow_target = diffusion_forward(diffusion, x0, eps, t, cur_text_emb)
        attn_lora = capture.outputs

        # L_attention: L2 distance on attn module outputs across blocks 10/11/12 × {self,cross}
        L_attention = sum(
            F.mse_loss(attn_lora[k].float(), attn_ref[k].float())
            for k in attn_lora
        )
        # L_pixel: vanilla flow MSE
        L_pixel = F.mse_loss(flow_pred.float(), flow_target.float())

        # ---- anchor branch (rCM analog) ----
        if step % ANCHOR_EVERY == 0:
            anc_idx = random.randint(0, len(anchor_embs) - 1)
            anc_emb = anchor_embs[anc_idx].to(device, dtype=torch.bfloat16)
            z0_anchor = torch.zeros_like(x0)  # "clean" = 0 → pure noise sample at scale sigma
            z_noise = torch.randn_like(x0)
            t_anc = sample_structural_timestep(B, T_MIN, T_MAX, device)

            capture.disable()  # don't need attn maps for anchor branch
            with peft_model.disable_adapter():
                with torch.no_grad():
                    flow_base, _ = diffusion_forward(
                        diffusion, z0_anchor, z_noise, t_anc, anc_emb
                    )
            flow_lora_anc, _ = diffusion_forward(
                diffusion, z0_anchor, z_noise, t_anc, anc_emb
            )
            capture.enable()
            L_anchor = F.mse_loss(flow_lora_anc.float(), flow_base.float())
        else:
            L_anchor = torch.tensor(0.0, device=device)

        loss = L_ATT_W * L_attention + L_ANC_W * L_anchor + L_PIX_W * L_pixel
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (step % LOG_EVERY == 0 or step == cfg.train.steps - 1) and is_main:
            log(
                f"[motion-lora] step={step:4d}/{cfg.train.steps} "
                f"L_att={L_attention.item():.4f} "
                f"L_anc={L_anchor.item():.4f} "
                f"L_pix={L_pixel.item():.4f} "
                f"total={loss.item():.4f}",
                True,
            )

    # ---- 8. Save LoRA (rank 0 only) -------------------------------------
    if is_main:
        save_path = out_dir / "motion_lora.pt"
        state = peft.get_peft_model_state_dict(peft_model)
        torch.save(state, save_path)
        log(f"[motion-lora] saved {len(state)} keys to {save_path}", True)

    capture.remove()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
