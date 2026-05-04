"""CPU-only adapter dispatch + OFT registration tests. No GPU / Wan / FSDP /
14B teacher. Self-running like tests/test_motion_hooks.py since the env
ships without pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---- Fake Wan attention block: a class named exactly the strings that
# configure_adapter_for_model walks for (CausalWanAttentionBlock for
# generator, WanAttentionBlock for fake_score). The walk matches by
# class __name__, so the fake module class only needs to share that name.

class CausalWanAttentionBlock(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        # Mirror real Wan attn: q/k/v/o + ffn(0)/ffn(2). dim small to keep
        # block-Cayley OFT init quick (oft_block_size=8 below ⇒ 8×8 blocks).
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=False),
        )

    def forward(self, x):
        h = self.o(self.q(x))
        return h + self.ffn(h)


class FakeTransformer(nn.Module):
    def __init__(self, dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([CausalWanAttentionBlock(dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _fresh_import():
    """Force a fresh import of lora_utils so its module-level _autoload runs
    cleanly, populating _ADAPTER_REGISTRY with both 'lora' and 'oft'.
    """
    for mod in list(sys.modules):
        if mod.startswith("longlive."):
            del sys.modules[mod]
    from longlive.utils import lora_utils
    return lora_utils


# ---- Tests --------------------------------------------------------------


def test_registry_has_lora_and_oft_after_autoload():
    lu = _fresh_import()
    keys = sorted(lu._ADAPTER_REGISTRY)
    assert "lora" in keys, f"lora missing from registry: {keys}"
    assert "oft" in keys, f"oft missing from registry: {keys}"


def test_double_registration_raises():
    lu = _fresh_import()
    try:
        lu.register_adapter("oft", lambda cfg, t: None)
    except KeyError as e:
        assert "already registered" in str(e)
        return
    raise AssertionError("expected KeyError on duplicate register_adapter")


def test_unknown_adapter_raises_with_suggestion():
    lu = _fresh_import()
    model = FakeTransformer()
    cfg = {"type": "banana"}
    try:
        lu.configure_adapter_for_model(model, "generator", cfg, is_main_process=False)
    except ValueError as e:
        msg = str(e)
        assert "banana" in msg and "lora" in msg and "oft" in msg, msg
        return
    raise AssertionError("expected ValueError for unknown adapter type")


def test_oft_dispatch_creates_peft_oft_model():
    lu = _fresh_import()
    import peft

    model = FakeTransformer(dim=64)
    cfg = {
        "type": "oft",
        "oft_block_size": 8,
        "use_cayley_neumann": True,
        "num_cayley_neumann_terms": 5,
        "module_dropout": 0.0,
        "verbose": False,
    }
    wrapped = lu.configure_adapter_for_model(model, "generator", cfg, is_main_process=False)
    pcfg = wrapped.peft_config["default"]
    assert isinstance(pcfg, peft.OFTConfig), f"got {type(pcfg).__name__}"
    assert pcfg.oft_block_size == 8
    assert pcfg.use_cayley_neumann is True

    n_trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in wrapped.parameters())
    assert 0 < n_trainable < n_total, f"trainable={n_trainable}, total={n_total}"


def test_oft_init_is_identity():
    """init_weights=True (PEFT default) ⇒ A=0 ⇒ R=I ⇒ wrapped(x) == base(x)
    before any training step. Pre-training the student must equal the base
    model exactly (or to fp32 numerical tolerance) — required for stable
    distillation startup.
    """
    lu = _fresh_import()
    torch.manual_seed(0)

    base = FakeTransformer(dim=64).eval()
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        y_base = base(x).clone()

    cfg = {
        "type": "oft",
        "oft_block_size": 8,
        "use_cayley_neumann": True,
        "verbose": False,
    }
    wrapped = lu.configure_adapter_for_model(base, "generator", cfg, is_main_process=False)
    wrapped.eval()
    with torch.no_grad():
        y_oft = wrapped(x)

    assert torch.allclose(y_base, y_oft, atol=1e-4), (
        f"OFT init not identity: max abs diff={(y_base - y_oft).abs().max().item():.3e}"
    )


def test_lora_dispatch_unchanged():
    """Regression: LoRA path still produces peft.LoraConfig with the same
    rank / alpha translation as before."""
    lu = _fresh_import()
    import peft

    model = FakeTransformer()
    cfg = {"type": "lora", "rank": 4, "alpha": 8, "dropout": 0.0, "verbose": False}
    wrapped = lu.configure_adapter_for_model(model, "generator", cfg, is_main_process=False)
    pcfg = wrapped.peft_config["default"]
    assert isinstance(pcfg, peft.LoraConfig)
    assert pcfg.r == 4
    assert pcfg.lora_alpha == 8


# ---- Self-runner --------------------------------------------------------


def main():
    tests = [
        test_registry_has_lora_and_oft_after_autoload,
        test_double_registration_raises,
        test_unknown_adapter_raises_with_suggestion,
        test_oft_dispatch_creates_peft_oft_model,
        test_oft_init_is_identity,
        test_lora_dispatch_unchanged,
    ]
    print("running oft_adapter tests:")
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"  ✓ {t.__name__}")
    if failed:
        print(f"{failed} / {len(tests)} failed")
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
