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

## 4. 关键组件

| 组件 | 作用 | 位置 |
|---|---|---|
| `_compute_kl_grad` | DMD 梯度 | `longlive/model/dmd.py` |
| `compute_distribution_matching_loss` | generator 端 DMD loss 入口 | `longlive/model/dmd.py` |
| `ScoreDistillationTrainer` | FSDP / LoRA / OFT 挂载 / checkpoint / WandB | `longlive/trainer/distillation.py` |

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
