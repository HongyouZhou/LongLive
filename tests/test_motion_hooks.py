"""Synthetic shape unit-test for MotionAttnInjector. CPU-only, no Wan deps."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Load motion_hooks directly to avoid importing model/__init__.py (which pulls
# in the full Wan stack incl. PIL/diffusers).
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "motion_hooks", str(REPO_ROOT / "longlive" / "model" / "motion_hooks.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["motion_hooks"] = _mod  # required for dataclasses to resolve cls.__module__
_spec.loader.exec_module(_mod)
MotionAttnInjector = _mod.MotionAttnInjector
MotionConfig = _mod.MotionConfig


class FakeSelfAttn(torch.nn.Module):
    """Stand-in for WanSelfAttention: takes [B, L, D], returns [B, L, D]."""
    def __init__(self, dim: int):
        super().__init__()
        self.o = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.o(x)


class FakeAttentionBlock(torch.nn.Module):
    """Stand-in for WanAttentionBlock; only `self_attn` is exposed."""
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = FakeSelfAttn(dim)

    def forward(self, x):
        # Mimic WanAttentionBlock: residual + modulated attn output.
        y = self.self_attn(x)
        return x + y


class FakeWanModel(torch.nn.Module):
    """Stand-in for WanModel.blocks[*]: list of FakeAttentionBlock."""
    def __init__(self, dim: int, num_blocks: int):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [FakeAttentionBlock(dim) for _ in range(num_blocks)]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def test_capture_then_inject_round_trip():
    torch.manual_seed(0)
    B, L, D = 1, 64, 32
    num_blocks = 8
    block_idxs = [3, 4, 5]
    alpha = 0.5

    model = FakeWanModel(D, num_blocks)

    x_ref = torch.randn(B, L, D)
    x_student = torch.randn(B, L, D)

    inj = MotionAttnInjector(model, block_idxs, alpha=alpha)

    with inj:
        # Phase 1: capture on x_ref.
        inj.set_mode("capture")
        _ = model(x_ref)
        # All requested blocks should be cached, with the correct shape.
        assert set(inj.cache.keys()) == set(block_idxs)
        for i in block_idxs:
            assert inj.cache[i].shape == (B, L, D), \
                f"block {i} cache shape={inj.cache[i].shape}"

        # Phase 2: inject — student forward should produce something different
        # from the unhooked baseline (because attn outputs at 3 blocks are blended).
        inj.set_mode("inject")
        out_injected = model(x_student)

    # After exit, cache cleared and hooks removed.
    assert inj.cache == {}
    assert inj._handles == []
    assert inj.mode is None

    # Compare against an unhooked re-run.
    out_baseline = model(x_student)
    diff = (out_injected - out_baseline).abs().mean().item()
    assert diff > 1e-6, f"injected output should differ from baseline; diff={diff}"
    print(f"  ✓ capture/inject round-trip: diff={diff:.4f}")


def test_inject_without_capture_raises():
    model = FakeWanModel(16, 4)
    inj = MotionAttnInjector(model, [1], alpha=0.5)
    with inj:
        try:
            inj.set_mode("inject")
        except RuntimeError as e:
            assert "before 'capture'" in str(e)
            print("  ✓ inject-before-capture raises RuntimeError")
            return
    raise AssertionError("expected RuntimeError")


def test_alpha_zero_is_passthrough():
    """alpha=0 means inject mode == identity on student output."""
    torch.manual_seed(1)
    B, L, D = 1, 32, 16
    model = FakeWanModel(D, 4)
    x_ref = torch.randn(B, L, D)
    x_student = torch.randn(B, L, D)

    inj = MotionAttnInjector(model, [1, 2], alpha=0.0)
    with inj:
        inj.set_mode("capture")
        _ = model(x_ref)
        inj.set_mode("inject")
        out = model(x_student)
    out_baseline = model(x_student)
    assert torch.allclose(out, out_baseline, atol=1e-6), \
        "alpha=0 should be identical to baseline"
    print("  ✓ alpha=0 is identity (passthrough)")


def test_alpha_one_replaces_completely():
    """alpha=1 means student attn output is fully replaced by V_ref's."""
    torch.manual_seed(2)
    B, L, D = 1, 24, 16
    model = FakeWanModel(D, 4)
    x_ref = torch.randn(B, L, D)
    x_student = torch.randn(B, L, D)

    inj = MotionAttnInjector(model, [2], alpha=1.0)
    with inj:
        inj.set_mode("capture")
        # Capture forward: hook stores ref attn at block 2.
        # Run model: every block sees the OUTPUT of the previous block.
        # We need to record the attn output that block 2 produces during the
        # capture pass — which the hook does. Then inject pass at block 2
        # replaces with that recorded value (alpha=1).
        _ = model(x_ref)
        ref_attn = inj.cache[2].clone()
        inj.set_mode("inject")
        # Manually compute what block 2 output should be on x_student with
        # full replacement: x_student_after_blk2 = x_student_before_blk2 + ref_attn
        # Run pre-block-2 chain on x_student manually.
        h = x_student
        for i in range(2):
            h = model.blocks[i](h)
        # block 2 with alpha=1: returns x + ref_attn (regardless of what self_attn
        # produces on this specific h).
        expected_after_blk2 = h + ref_attn
        out_full = model(x_student)
    # Build the reference output post-hook by running blocks 3..N on expected_after_blk2.
    h2 = expected_after_blk2
    for i in range(3, len(model.blocks)):
        h2 = model.blocks[i](h2)
    assert torch.allclose(out_full, h2, atol=1e-5), \
        "alpha=1 should replace block-2 attn entirely"
    print("  ✓ alpha=1 fully replaces attn")


def test_realistic_shapes():
    """The shapes the actual 14B teacher will see: [B=1, L=32760, D=5120]
    is too big for CPU; sanity-check at smaller realistic ratios."""
    B, L, D = 1, 1024, 256
    model = FakeWanModel(D, 40)  # 40 blocks like 14B teacher
    x_ref = torch.randn(B, L, D)
    x_student = torch.randn(B, L, D)
    inj = MotionAttnInjector(model, [18, 19, 20], alpha=0.5)
    with inj:
        inj.set_mode("capture")
        _ = model(x_ref)
        inj.set_mode("inject")
        out = model(x_student)
    assert out.shape == (B, L, D)
    print(f"  ✓ realistic 40-block × {L}-token shape OK")


def test_beta_schedule_constant():
    cfg = MotionConfig(beta_max=0.15, beta_schedule="constant")
    for s in [0, 50, 200, 500, 2999]:
        assert abs(cfg.beta_at(s) - 0.15) < 1e-9, f"step {s}: got {cfg.beta_at(s)}"
    print("  ✓ constant β=0.15 at every step")


def test_beta_schedule_linear_warmup():
    cfg = MotionConfig(
        beta_max=0.15, beta_warmup_start=0.0, beta_warmup_steps=200,
        beta_schedule="linear_warmup",
    )
    assert abs(cfg.beta_at(0) - 0.0) < 1e-9
    assert abs(cfg.beta_at(100) - 0.075) < 1e-9
    assert abs(cfg.beta_at(199) - 0.149250) < 1e-4
    assert abs(cfg.beta_at(200) - 0.15) < 1e-9
    assert abs(cfg.beta_at(500) - 0.15) < 1e-9  # plateau
    print("  ✓ linear_warmup ramps 0 → 0.15 over 200 then plateau")


def test_beta_schedule_warmup_cyclic():
    cfg = MotionConfig(
        beta_max=0.2, beta_warmup_start=0.0, beta_warmup_steps=200,
        beta_schedule="warmup_cyclic", cyclic_period=10, cyclic_high_ratio=0.7,
    )
    # warmup region behaves linearly
    assert abs(cfg.beta_at(100) - 0.1) < 1e-9
    # post-warmup: 7-on, 3-off cycle
    expected_post = [0.2] * 7 + [0.0] * 3  # cycle of 10
    for i, expected in enumerate(expected_post):
        got = cfg.beta_at(200 + i)
        assert abs(got - expected) < 1e-9, f"step {200+i}: got {got}, expected {expected}"
    # cycle wraps: step 210 should be 0.2 again
    assert abs(cfg.beta_at(210) - 0.2) < 1e-9
    # ensure overall fraction at β_max equals cyclic_high_ratio
    n_high = sum(1 for s in range(200, 1000) if cfg.beta_at(s) > 0)
    frac = n_high / 800
    assert abs(frac - 0.7) < 0.01, f"high fraction = {frac}, expected 0.7"
    print(f"  ✓ warmup_cyclic 7-on/3-off; high fraction = {frac:.3f}")


def test_beta_schedule_zero_warmup():
    """beta_warmup_steps=0 should plateau at beta_max from step 0."""
    cfg = MotionConfig(beta_max=0.2, beta_warmup_steps=0, beta_schedule="linear_warmup")
    assert abs(cfg.beta_at(0) - 0.2) < 1e-9
    assert abs(cfg.beta_at(1000) - 0.2) < 1e-9
    print("  ✓ beta_warmup_steps=0 → constant β_max from step 0")


def test_beta_schedule_invalid_raises():
    """attach_motion_config rejects unknown beta_schedule string."""
    class A:
        seed = 0
        motion = {"enabled": True, "beta_schedule": "not_a_schedule"}
    class H: pass
    h = H()
    try:
        _mod.attach_motion_config(h, A())
    except ValueError as e:
        assert "beta_schedule" in str(e)
        print("  ✓ invalid beta_schedule raises ValueError")
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    print("running motion_hooks tests:")
    test_capture_then_inject_round_trip()
    test_inject_without_capture_raises()
    test_alpha_zero_is_passthrough()
    test_alpha_one_replaces_completely()
    test_realistic_shapes()
    test_beta_schedule_constant()
    test_beta_schedule_linear_warmup()
    test_beta_schedule_warmup_cyclic()
    test_beta_schedule_zero_warmup()
    test_beta_schedule_invalid_raises()
    print("all passed")
