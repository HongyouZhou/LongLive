# OFT — Orthogonal Fine-Tuning adapter

**Touches L1**(参数化层)。`R = (I+A)(I-A)⁻¹` 块 Cayley,A 反对称,块大小默认 64。R init = I(`init_weights=True`),student 起始等于 base model。

实现走 PEFT 0.19+ 的 `peft.OFTConfig`,跟 LoRA 共享 `get_peft_model` / `get_peft_model_state_dict` / `merge_and_unload()` 路径。`__init__.py` 只做一次 `register_adapter("oft", ...)` 把 PEFT OFTConfig factory 注册到 `longlive.utils.lora_utils._ADAPTER_REGISTRY`。

## 启用

YAML adapter 段切到 `type: "oft"`(见 `configs/cat_dunk_oft_v1.yaml` 完整示例)。其他 layer (L0/L2-L5) 都不动 —— 与 motion-DMD / Bridge / NFT 等正交,可同时启用。

## 关键超参

- `oft_block_size`:64 起步;b=128/256 表达力更强但参数更多(参数量 ≈ `Σ_layer (d/b) · b·(b-1)/2`)。
- `use_cayley_neumann: true`:用 Neumann 级数近似 `(I-A)⁻¹`,避免每 step 块求逆。
- `num_cayley_neumann_terms: 5`:Neumann 项数,默认即可。
- `module_dropout: 0.0`:类比 LoRA dropout;先关。

## 与 LoRA 对比的实操

`configs/cat_dunk_oft_v1.yaml` 是 `configs/longlive_train_motion.yaml` 的完整 copy + 仅 `adapter:` 段切到 `type: "oft"`。同 base ckpt + 同 data + 同 iters → 大致公平对比。LR sweep `{1e-5, 3e-5, 1e-4}` 通过 CLI override 或多 config 实现。
