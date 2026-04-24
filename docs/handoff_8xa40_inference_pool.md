# Handoff: 8×A40 推理/评测 Pool 设计

> 讨论如何搭建独立的 8×A40 推理池,给 2×PRO 6000 的训练任务做实时/批量评测。
> 训练本身另一个 session 在跟,不讨论训练细节。

## 1. 项目背景(最简版)

- **工作**:在 LongLive(NVIDIA ICLR 2026,arXiv:2509.22622)上叠加 motion conditioning,方案命名 **Motion-Recache**——把 LongLive 的 `KV-recache` 从 prompt switching 推广到 motion conditioning。
- **核心技术**:
  - Motion encoder(270M,Wan VAE frozen + 4 层 transformer)把参考视频编码为 256 个 4096-dim tokens。
  - Tokens 直接 concat 到 `conditional_dict["prompt_embeds"]` 后面,伪装成 T5 text 延续。
  - `WanT2VCrossAttention` 做 split-cache:text K/V 跨步缓存(detached),motion K/V 每步重算(fresh 梯度)。
- **训练方式**:复用 LongLive 原生 DMD 蒸馏流水线,只改 3 个文件 + 2 个新文件。Config `configs/longlive_finetune_motion.yaml`。
- **数据**:OpenVid-1M part0 抽 1000 clip(motion_score≥3, aesthetic≥4.5),位置 `/home/hongyou/dev/data/wm/`,自配对(`motion_a==motion_b`,`switch_frame=-1`)。
- **训练状态**(截至 handoff):lab(2×RTX PRO 6000 Blackwell)从 step 2000 断点续训,目标 3000 步,约 13h 后收敛。

## 2. 为什么要独立推理池

训练端瓶颈:
- 2×PRO 6000 训练速度 ~25s/iter,3000 步 ~20h。
- 推理/评测若占用训练 GPU,每次 vis 阻塞 ~3 分钟(已观察到 step 501 耗时 137s)。
- 更重要:想要**跨 checkpoint 的系统化评测**(e.g. 对每 50 步的 ckpt 跑同一组 prompt + motion_ref 组合),训练 GPU 没有余力。

独立 8×A40:
- A40 48GB ×8 = 384GB 聚合显存。
- **A40 用于推理足够**:只需要 1.3B generator + motion_encoder + VAE,不需要 14B teacher。
- 8 卡并行可覆盖 50-step checkpoint 节奏内的全部评测。

## 3. 预期架构

```
[Lab]   2×RTX PRO 6000 Blackwell  训练 LongLive + motion
        每 50 步 save:
        ~/longlive_work/logs/checkpoint_model_XXXXXX/model.pt  (~2.4GB)

              │ (待定:rsync daemon / inotify / sshfs)
              ▼

[某节点] 8×A40  推理 pool
        - 监听 ckpt 目录,新 ckpt 到达 → 触发评测
        - 8 worker 并行,每 worker 占 1 A40
        - 固定评测集:N prompts × M motion_refs
        - 输出:mp4 + 指标(光流 L2 / CLIP score / VBench)
        - wandb 独立 run,step 对齐训练 step
```

## 4. 吞吐估算

| 项 | 数值 |
|---|---|
| A40 单次推理(240 帧,1.3B causal streaming) | ~30-60s(待实测) |
| 8 A40 并行吞吐 | 8-16 videos/min |
| 单 ckpt 评测(80 对 prompt×motion) | ~5-10 min |
| 训练 ckpt 间隔(50 步) | ~20 min |
| **余量** | **2-3 倍** ✓ |

## 5. 主要未决问题

### A. 硬件拓扑
- **8×A40 物理位置**:arp 本机?独立节点?与 lab 是否同机房 / 同 LAN?
- **A40 是否 NVLink**:标准 PCIe A40 无 NVLink,多卡独立并行(每 worker 1 卡)最合适。
- **A40 环境**:conda env 是否已 ready?Wan VAE 权重是否已就位?OpenVid motion_refs 能否访问?

### B. Checkpoint 同步机制
四选一:
1. **Rsync daemon**(推荐):lab 挂 rsync 服务,推理节点定时拉。LAN ~1Gbps → 2.4GB ckpt ~24s,不影响训练。
2. **SSHFS 反向挂载**:推理节点挂 `lab:~/longlive_work/logs/`。访问透明但写入/读取慢。
3. **主动推送**:训练 save 时直接 scp 到推理节点(需训练代码改动)。
4. **NFS/共享存储**:最理想,但需要基建。

### C. 评测集设计
- **Prompt 源**:
  - OpenVid val 100 条?(已有)
  - VBench 标准 prompt 集?
  - 手工精选 5-10 个 diverse prompt 用于定性 vis?
- **Motion ref 源**:
  - 从 OpenVid motion_refs/ 精选 8 个(覆盖人动作、动物动作、相机运动、慢/快动作)
  - 还是用外部 reference(Pexels / 测试用专门素材)?
- **评测组合**:
  - **主轴**:同 prompt × 不同 motion → 看 motion 控制力
  - **副轴**:不同 prompt × 同 motion → 看 motion 泛化
  - **对照**:无 motion(原 LongLive baseline)

### D. 指标自动化
- **纯定性**(mp4 肉眼看):最快,但不能 track progress
- **定量**:
  - 光流 L2:生成视频的光流 vs motion_ref 的光流(motion fidelity)
  - CLIP score:prompt-video 对齐
  - VBench motion dimensions:dynamic_degree, motion_smoothness, temporal_flickering
  - LPIPS/FID:视觉质量
- **推荐 MVP**:先出 mp4(定性)+ 光流 L2(定量 motion control),VBench 后续加。

### E. Wandb 对接
- 独立 project `longlive-eval`?或同 project 下 `run_name="eval"` 分开?
- 日志对齐方式:`wandb.log({...}, step=ckpt_step)`,dashboard 上可并列显示 training loss + eval video。

## 6. 相关文件(已写好,可直接用)

| 文件 | 作用 |
|---|---|
| `inference_motion.py` | 单进程推理脚本。接受 `--prompt`, `--motion_ref`, `--lora_ckpt`,输出 mp4。 |
| `configs/longlive_inference_motion.yaml` | 推理 config,字段 `motion_encoder`, `adapter`, `lora_ckpt` 等 |
| `model/motion_encoder.py` | MotionEncoder 定义(推理和训练共用) |
| `utils/dataset.py:_load_motion_video` | 加载参考视频的工具函数 |

## 7. 推荐 MVP 路线图(预估 1-2 天)

### Day 1 上午:打通 end-to-end
1. 确认 8×A40 所在节点、环境、Wan VAE 权重
2. 打通一次单 A40 推理:`python inference_motion.py --lora_ckpt logs/checkpoint_model_002000/model.pt ...`(注意:ckpt 在 lab 本地,需先 scp 过来)
3. 记录单次推理耗时,验证 fit A40 48GB

### Day 1 下午:8 卡并行
4. 写 `scripts/eval_daemon.py` 的 **poll + subprocess.Popen 版本**(最简):
   ```python
   while True:
       new_ckpts = find_new_checkpoints()
       for ckpt in new_ckpts:
           subprocess.Popen([..., "--lora_ckpt", ckpt], env={"CUDA_VISIBLE_DEVICES": str(gpu_id)})
   ```
5. 设计固定 80 对 (prompt, motion) 评测集,产出 mp4

### Day 2 上午:数据回流
6. Rsync daemon / sshfs 配置,让 ckpt 自动传到推理节点
7. Wandb eval run 初始化,推送 mp4 + 基础指标

### Day 2 下午:指标 & 自动化
8. 加光流 L2 计算(RAFT or OpenCV Farneback)
9. 第一次完整 run:回放 step 500 / 1000 / 1500 / 2000 / 2500 的 ckpt,全部跑评测,看 loss 曲线之外的真实进步

### 后续(day 3+)
10. VBench 评测 pipeline 集成
11. Baseline 对比:冻结 motion_encoder 的版本 + 原 LongLive 无 motion 版本

## 8. 风险清单

| 风险 | 缓解 |
|---|---|
| A40 Ampere 加载 Blackwell 写的 bf16 权重 | 理论 OK(state_dict arch-agnostic),但需首次验证 |
| 推理节点 conda env 与训练不同步 | 使用同一 `longlive-blackwell` env 目录(走 sshfs)或复制一份 |
| Ckpt 2.4GB × 60(全部 ckpt)= 150GB 存储 | 只保留最近 N 个,或评测完立即删 |
| 8 个并行 Python 进程共享 `motion_ref` 视频文件 I/O | 预加载到内存 / ramdisk |
| A40 推理速度比预估慢 | 用量化/bf16 推理,或降到 120 帧 |
| wandb 上传大量视频超出 quota | 只上传每个 ckpt 1-2 个代表性视频,其余存本地 |

## 9. 辅助信息

- 训练 run 在跑:`hongyou/longlive/runs/6bkbu39n`(断点续训后的 ID,原 ID `jn7g6a7f`)
- 训练 config:`configs/longlive_finetune_motion.yaml`
- 数据位置:`/home/hongyou/dev/data/wm/{motion_refs, prompts}/`
- Motion pair jsonl:`/home/hongyou/dev/data/wm/prompts/motion_pairs_{train,val}.jsonl`
- OpenVid 过滤参数:`motion_score >= 3, aesthetic >= 4.5, 2s <= seconds <= 20s`

## 10. 开始新 session 时的第一个问题应该问

> "8×A40 在哪台机器?能访问 lab 的 /home/hongyou/longlive_work/logs/ 吗?如果不能,打算用什么机制同步 checkpoint?"

这是所有后续设计的依赖点。
