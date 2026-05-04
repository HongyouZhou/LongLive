# DMD Training Framework

## 1. 三角色

```
generator    1.3B causal Wan        trainable     ← 被蒸馏目标
fake_score   1.3B Wan (critic)      trainable     ← 估计 student 分布
real_score   14B Wan  (teacher)     frozen        ← 提供 score
```

## 2. Generator step（vanilla DMD）

```
            x̂ = generator(noise)
                    │
                    ▼
            x_t = add_noise(x̂, t)
              │            │
              ▼            ▼
       fake_score(x_t)   real_score(x_t)  cond+uncond CFG
              │            │
          pred_fake     pred_real
                    │
                    ▼
        grad = (pred_fake − pred_real) / |x̂ − pred_real|
                    │
                    ▼
        loss = 0.5·MSE(x̂, (x̂ − grad).detach())
```

## 3. Critic step

```
        x̂ = generator(noise)  [no_grad]
                    │
                    ▼
        fake_score(add_noise(x̂, t))  →  flow loss  →  backward
```

generator/critic = 1 / `dfake_gen_update_ratio` = 1/5。

## 4. Motion-DMD

**思路**：让 teacher 的 score 偏向一段参考视频 V_ref 的 motion，generator 蒸馏后获得对应运动模式。**只改 §2 里的 `pred_real`，critic / 网络 / rollout 全不变。**

### 4.1 Motion 编码（offline）

`scripts/motion_dmd/precache_motion_refs.py` 把若干参考 mp4 一次性编成 latent cache，训练时 mmap 加载，不在 step 里付 VAE-encode 代价。

```
ref.mp4
   │  read_video → 取 81 帧 (linspace 均匀采样, 不足则重复)
   ▼
[F=81, H, W, 3] uint8
   │  /255 → resize 480×832 → ×2−1   (规格化到 [-1, 1])
   ▼
[C=3, F=81, H=480, W=832] float32           ← Wan VAE 编码器是 fp32
   │  WanVAEWrapper.encode_to_latent
   │  时间 (1+4k) 压缩: 81 → 21；空间 8× 下采样: 480/8=60, 832/8=104
   ▼
[F=21, C=16, H=60, W=104] bf16              ← 单条 V_ref latent
```

stack 后保存：

```
walking_v1.latents.pt = {
    latents : [N, 21, 16, 60, 104] bf16,   # 全部 ref
    captions: [N] str,                      # 与 latents 同序
    paths   : [N] str,
}
```

训练启动时 `_maybe_load_motion_refs` 把 `latents` pin-memory 加载到 CPU，`captions` 用 T5 提前 encode 成 N 份 `motion_caption_dicts`（保证多 rank 一致）；每个 sequence 由 rank-0 抽一个 `ref_idx`、broadcast，把对应 latent 搬 GPU 进入 §4.2 的 capture pass。

### 4.2 注入：两次 teacher forward + 一个钩子

`MotionAttnInjector` 用 `register_forward_hook` 挂在 14B teacher 选定的若干 self_attn 模块上（默认 `blocks=[18,19,20]`，40 block 中段），分两阶段：

```
① capture:  teacher( v_ref_t  | caption_V_ref )
            hook 把每个目标 block 的 self_attn 输出
            (post-o-projection, shape=[B, L, D]) detach 后存进 cache[i]

② inject:   teacher(   x_t    | cond_student )
            hook 拦截同一批 self_attn 输出，返回:
              out_inject = (1 − α) · out_student + α · cache[i]
            后续 cross-attn / FFN / 下游 block 全部基于 out_inject 继续算
            最终输出 → pred_real_motion
```

要点：
- **同一 timestep `t`、同一噪声调度**：`v_ref_t = add_noise(V_ref, t)`、`x_t = add_noise(x̂, t)`，保证 cache 的 attn 与 inject pass 形状/分布对齐。
- **caption 不一样**：capture 用 V_ref 自带 caption（让 motion 被正确激活），inject 用 student 当前 prompt（hook 已经把 motion 信号烧进 attn，prompt 不需要换）。
- **只动 self_attn 输出**：cross-attn / FFN / 其它 block 不挂钩子，因此文本对齐不被破坏，只有运动相关的空间-时间结构被偏置。
- **hook 退出即清空**：`__enter__/__exit__` 自动注册/反注册，cache 在每个 generator step 内一新一弃，不影响 FSDP all-gather。

### 4.3 Score blend

```
  pred_real_blended = (1−β)·pred_real + β·pred_real_motion
                       └ vanilla teacher       └ motion-biased teacher
```

把 `pred_real_blended` 代入 §2 走 DMD grad。motion-on 的 step 共 4 次 teacher forward（cond + uncond + capture + inject）。

### 4.4 两个混合系数

- `α` ∈ [0,1]：钩子内 attn 替换强度（空间层面，决定单次 inject pass 注入多少 V_ref attn）。固定值，不随 step 变。
- `β` ∈ [0,1]：score 层面混合强度（决定 motion teacher 在 DMD grad 里的权重）。**按 step 调度**；`β=0` 时该 step 自动退回 vanilla `_compute_kl_grad`，dispatch 在 `compute_distribution_matching_loss` 里。

### 4.5 β 调度（`MotionConfig.beta_at(step)`）

```
constant            β
                    ┤█████████████████████████  beta_max
                    └──────────────────────►step

linear_warmup       β
                    ┤        ╱─────────────  beta_max
                    ┤      ╱
                    ┤    ╱
                    ┤──╯                       beta_warmup_start
                    └──┬───────────────────►step
                     beta_warmup_steps

warmup_cyclic       β            ┌─┐ ┌─┐ ┌─┐
                    ┤        ╱──┘ └─┘ └─┘ └─  beta_max
                    ┤      ╱
                    ┤    ╱
                    ┤──╯       └─┘ └─┘ └─┘ 0
                    └──┬───┬───┬───┬───┬──►step
                     warmup  cyclic_period
                            (high_ratio at β_max, rest at 0;
                             β=0 段自动走 vanilla DMD,
                             = v1 Bernoulli inject_prob 的确定性版)
```

trainer 每步 `self.model.global_step = self.step` 同步进 base model（`distillation.py`），`beta_at` 据此读 warmup / cycle 进度。

## 5. 关键组件

| 组件 | 作用 | 位置 |
|---|---|---|
| `_compute_kl_grad` | vanilla DMD 梯度 | `longlive/model/dmd.py` |
| `_compute_kl_grad_with_motion` | motion 替换分支 | `longlive/model/dmd.py` |
| `MotionAttnInjector` | teacher self_attn 钩子，capture / inject 两阶段 | `longlive/model/motion_hooks.py` |
| `MotionConfig.beta_at(step)` | β schedule（constant / linear_warmup / warmup_cyclic） | `longlive/model/motion_hooks.py` |
| `_maybe_load_motion_refs` | 启动时加载 V_ref latent + 预 encode caption | `longlive/trainer/distillation.py` |
| `_maybe_pick_motion_ref` | 每 sequence rank-0 抽 V_ref 并 broadcast | `longlive/trainer/distillation.py` |
| `precache_motion_refs.py` | V_ref 离线缓存生成 | `scripts/motion_dmd/` |

## 6. v3 调参

```
α = 0.3                  # 钩子混合强度
β_max = 0.10             # score blend 稳态
beta_schedule = linear_warmup
beta_warmup_steps = 600  # 20% of max_iters=3000
block_idxs = [18,19,20]  # 14B 40 block 中段
```



### 讨论
- 实验高效性
- 输入video, finetune
- 快速finetune
  - 一个视频
  - 从效率角度考虑
  - lora变体
   - 根据模态调整lora
  - 稀疏注意力上也能做工作
- 视频生成双向注意力
- 多看看算法
  - 3D VGGT, flash VGGT
  - 多参考

### 目标, 小猫扣篮. 
- 用尽可能少的时间, 达到finetune的效果, 比如扣篮视频, finetune出来猫扣篮
  - motion editing
  - sihan xu, 密歇根大学phd, cvpr 2024, 如果在few step 做image editing
  - yue ma, hongkong ust, fast video motion transfer
  - chengfeng xu, uc berkly, streaming diffusion v2, few step video editing
  - iccv 2025 best paper finallist
  - jun zhu, diffusion nft, 如何在few step做finetune
- ICL, video 作为context, test time training.
- chi zhang, west lake uni, diffusion sampling. 
- zeke xie, diffusion sampling. 

整理一个list. 