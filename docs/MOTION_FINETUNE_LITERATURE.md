# Motion Finetune 文献调研

目标场景:用一段参考视频(如扣篮),few-step finetune 出新主体(如猫)做同样 motion。
约束:`dmd_architecture_frozen`(不改 distillation loop / 网络),`no_motion_encoder`(不加 encoder),改进只走数据/采样/超参/loss 层面。

★ = 与"扣篮→猫"few-step finetune 直接相关、必读
✦ = 值得读的 baseline 或参考

---

## 1. VGGT / 3D 几何基础模型(motion 先验候选)

| Tag | 论文 | 作者 / 机构 | 会议/年份 | 链接 | 创新点 (2-3 句) |
|---|---|---|---|---|---|
| ✦ | **VGGT: Visual Geometry Grounded Transformer** | Wang, Chen, Karaev, Vedaldi, Rupprecht, Novotny / Oxford VGG + Meta AI | CVPR 2025 (Best Paper) | [arXiv 2503.11651](https://arxiv.org/abs/2503.11651) | 单一前馈 transformer 在一次 forward 中联合预测相机位姿、深度、point map、3D track,支持 1–100 张视图,推理 < 1s。把过去需要级联 SfM/MVS pipeline 的多任务 3D 重建压成 end-to-end。可作为 motion finetune 时几何先验的特征提取器。 |
| | **FastVGGT** | HKU / Tsinghua | arXiv 2509.02560 | [arXiv](https://arxiv.org/abs/2509.02560) | 训练-free token merging(init-frame + salient + region-random)作用于 global attention,VGGT 在 1000 张图上 4× 加速。是 VGGT 推理瓶颈下的 drop-in 替代,不需要重新训。 |
| | **π³ (Pi3)** | Wang, Zhou, He / Shanghai AI Lab | ICLR 2026, arXiv 2507.13347 | [arXiv](https://arxiv.org/abs/2507.13347) | Reference-frame-free、permutation-equivariant 设计,KITTI 上 57.4 FPS vs VGGT 43.2,模型更小、更快、更好。适合在线训练时对乱序视频 chunk 抽几何特征。 |
| | **CUT3R** | Wang, Zhang, Holynski, Efros, Kanazawa / UC Berkeley | CVPR 2025, arXiv 2501.12387 | [arXiv](https://arxiv.org/abs/2501.12387) | Stateful recurrent transformer,把 3D 重建做成 streaming/online 更新,**显式支持 dynamic scene**(VGGT/DUSt3R 假设静态)。最适合"猫扣篮"这种动态主体。 |
| | **Fast3R** | Yang, Sax, Liang, Henaff, Feiszli / Meta | CVPR 2025, arXiv 2501.13927 | [arXiv](https://arxiv.org/abs/2501.13927) | N-image transformer 一次过,绕过 DUSt3R 系列的 pairwise 迭代对齐。VGGT 的并发竞品。 |
| | **DUSt3R** | Wang, Leroy, Cabon, Chidlovskii, Revaud / Naver Labs Europe | CVPR 2024, arXiv 2312.14132 | [arXiv](https://arxiv.org/abs/2312.14132) | 3R 系列鼻祖,直接做 pairwise pointmap 回归,跳过相机标定与 SfM。今天主要作 baseline 参考,被 VGGT/π³ 在视频场景超越。 |
| ★ | **Geometry Forcing** | Wu, Wu, He, Guo, Duan, Bian / MSRA + Tsinghua | arXiv 2507.07982 | [arXiv](https://arxiv.org/abs/2507.07982) | REPA-style 损失:把 video diffusion 中间特征对齐到 frozen VGGT 特征上,在长时序视频一致性上击败 DINOv2 对齐。**纯 auxiliary loss、不动架构**,完全适配 `dmd_architecture_frozen` 约束,可直接加在 motion-DMD 上。 |

## 2. Sihan Xu(UMich SLED, Joyce Chai 组)— few-step image editing

| Tag | 论文 | 作者 | 会议/年份 | 链接 | 创新点 (2-3 句) |
|---|---|---|---|---|---|
| ★ | **InfEdit (Inversion-Free Image Editing)** | Xu, Huang, Pan, Ma, Chai | CVPR 2024 | [arXiv 2312.04965](https://arxiv.org/abs/2312.04965) | 推导出 DDCM(Denoising Diffusion Consistent Model):当起始样本已知时,特定方差调度让 denoising 退化为 multi-step consistency sampling,**完全消除 explicit inversion**。基于 LCM + prompt2prompt,3 步 consistency sampling 同时支持刚性与非刚性编辑。few-step + consistency-model + attention-swap 配方的母版,可直接迁移到 motion-DMD 做"虚拟 inversion"。 |
| ✦ | **CycleNet** | Xu, Ma, Huang, Lee, Chai | NeurIPS 2023, arXiv 2310.13165 | [arXiv](https://arxiv.org/abs/2310.13165) | 把 cycle-consistency 正则当成 ControlNet-style 条件,只用 ~1 个 image pair 即可适配未见 domain 的扩散图像翻译。低数据 motion adaptation 思路的先声。 |
| | **UniCtrl** | Xia, Chen, Xu et al. | TMLR 2024, arXiv 2403.02332 | [arXiv](https://arxiv.org/abs/2403.02332) | 训练-free 的统一 attention 控制,用于 T2V 时空一致性。可作为"是否需要 LoRA vs inference-time attention 操控"的对照基线。 |

## 3. Yue Ma(HKUST,Follow-Your-* 系列)

| Tag | 论文 | 角色 | 会议/年份 | 链接 | 创新点 (2-3 句) |
|---|---|---|---|---|---|
| ★ | **FastVMT** | 1st | ICLR 2026, arXiv 2602.05551 | [project](https://fastvmt.github.io/) | 识别 DiT motion transfer 中两类冗余:motion redundancy(用 local attention masking 削)+ gradient redundancy(跨 diffusion step 复用 + skip)。**training-free 3.43× 加速**,正面回应"motion finetune 怎么变快"问题。 |
| ★ | **Follow-Your-Motion** | 1st | arXiv 2506.05207 | [arXiv](https://arxiv.org/abs/2506.05207) | Spatial-temporal decoupled LoRA(在 3D attention 中拆 spatial appearance vs temporal motion)+ sparse motion sampling + adaptive RoPE 加速。引入 MotionBench 评测。**架构上最贴近我们 v1 路线的工作**。 |
| ✦ | **Follow-Your-Pose** | 1st | AAAI 2024, arXiv 2304.01186 | [project](https://follow-your-pose.github.io/) | 两阶段训练利用 pose-free videos,把骨架 motion transfer 到 T2V。Follow-Your-* 系列起点。 |
| | **Follow-Your-Click** | 1st | AAAI 2025, arXiv 2403.08268 | [project](https://follow-your-click.github.io/) | 一次点击 + 短 prompt 驱动静态图局部区域动画。one-shot 区域 motion control 的代表。 |
| | **Follow-Your-Emoji** | 1st | SIGGRAPH Asia 2024 / IJCV 2025, arXiv 2406.01900 | [project](https://follow-your-emoji.github.io/) | 表情 landmark 驱动 freestyle 人像动画,保身份。Pose-driven motion transfer 在人脸的极致版。 |
| | **Follow-Your-Handle / MagicStick** | 1st | WACV 2025, arXiv 2312.03047 | [project](https://magic-stick-edit.github.io/) | 通过对 control handle(edge map / pose)的形变在关键帧上编辑形状/大小/位置/动作,然后传播到全片。 |
| | **Follow-Your-Canvas** | co-1st | AAAI 2025, arXiv 2409.01055 | [arXiv](https://arxiv.org/abs/2409.01055) | 高分辨率视频 outpainting,把 canvas 作为驱动条件。 |
| | **Follow-Your-Creation** | 1st | arXiv 2506.04590 | [project](https://follow-your-creation.github.io/) | 把单 monocular 视频生成 4D 结构,formulate 为 video inpainting 问题。 |
| | **Follow-Your-Shape** | senior | arXiv 2508.08134 | [project](https://follow-your-shape.github.io/) | Trajectory-guided 区域控制做 shape-aware image editing,training-free / mask-free。 |
| | **Follow-Your-Pose v2** | co-author | ICLR 2025, arXiv 2406.03035 | [arXiv](https://arxiv.org/abs/2406.03035) | 多条件人像动画。 |
| | **Follow-Your-Instruction** | co-1st | arXiv 2508.05580 | [arXiv](https://arxiv.org/abs/2508.05580) | MLLM agent 用作世界数据合成。 |
| | **Controllable Video Generation Survey** | 1st | arXiv 2507.16869 | [Awesome list](https://github.com/mayuelala/Awesome-Controllable-Video-Generation) | 自家 survey,可作为 controllable video generation 整体入口。 |

## 4. Chenfeng Xu(Berkeley → UT Austin)— inference systems 路线

| Tag | 论文 | 角色 | 会议/年份 | 链接 | 创新点 (2-3 句) |
|---|---|---|---|---|---|
| ✦ | **StreamDiffusion** | co-1st | ICCV 2025, arXiv 2312.12491 | [code](https://github.com/cumulo-autumn/StreamDiffusion) | Pipeline 级实时方案:Stream Batch + RCFG + stochastic similarity filter,4090 上 91 FPS image stream。把"实时交互式扩散"做成系统问题而非模型问题。 |
| ✦ | **StreamV2V** | co-author | ICLR 2025, arXiv 2405.15757 | [project](https://jeff-liangf.github.io/projects/streamv2v/) | "Feature bank"存过去帧的 K/V,融合到当前帧 self-attention 实现 training-free 时序一致 streaming V2V,1×A100 上 20 FPS。**是"few-step inference 时换 motion / 保 appearance"的现成蓝本**。 |
| | **StreamDiffusionV2** | senior | arXiv 2511.07399 | [project](https://streamdiffusionv2.github.io/) | Training-free 视频 diffusion 流式系统:SLO-aware batching/block scheduler、sink-token-guided rolling KV cache、motion-aware noise controller,跨 step 与 layer 并行去噪,Wan 14B 在 4×H100 上 58 FPS,无 TRT/quant。 |
| | **Sparse VideoGen / Quant VideoGen** | co-author | ICML 2025 / arXiv 2026 | 见 §10 | 部署侧加速;详见稀疏注意力一节。 |

> 注:Chenfeng Xu 这条线**完全没有 motion finetune**,只做 training-free inference 系统加速,主要价值是 StreamV2V 的 feature-bank 思想可以借给"inference-time 改 motion"。

## 5. Jun Zhu(TSAIL)/ Diffusion NFT

| Tag | 论文 | 角色 | 会议/年份 | 链接 | 创新点 (2-3 句) |
|---|---|---|---|---|---|
| ★ | **DiffusionNFT** | co-senior | ICLR 2026 Oral, arXiv 2509.16117 | [code](https://github.com/NVlabs/DiffusionNFT) | NFT = Negative-aware Fine-Tuning。在**前向 noising 过程**用 flow-matching 对正/负样本做 contrast,无需 likelihood 评估、无需轨迹存储,任意黑盒 solver 通用,只需 clean image。**比 DPO/PPO 类 reward finetune 快约 25×**,适合做 motion reward shaping。 |
| ★ | **rCM (Score-Regularized Continuous-Time CM)** | co-senior | ICLR 2026, arXiv 2510.08431 | [code](https://github.com/NVlabs/rcm) | 首次把 continuous-time consistency(sCM/MeanFlow)scale 到 14B 参数 + 5s 视频,自研 FlashAttention-2 JVP kernel,score distillation 作长跳跃正则,1–4 step 出图视频,15–50× 加速。**"video few-step finetune"目前最强的成熟配方**。 |
| ★ | **TurboDiffusion** | co-author | arXiv 2512.16093 | [arXiv](https://arxiv.org/abs/2512.16093) | rCM step 蒸馏 + SLA/SageAttention + W8A8,组合达 100–200× 视频端到端加速。是 rCM 的工程落地参考。 |
| | **Analytic-Precond** | senior | ICLR 2025, arXiv 2502.02922 | [arXiv](https://arxiv.org/abs/2502.02922) | 解析最优化 consistency distillation 的 preconditioning 系数,闭合 teacher–student gap,2–3× 训练加速。CD 路线的超参一站式解决。 |
| | **NFT (LLM 版)** | co-author | arXiv 2505.18116 | [arXiv](https://arxiv.org/abs/2505.18116) | 在数学推理上把"错误答案"建成 implicit negative policy,介于 SL 与 RL 之间。是 DiffusionNFT 概念的 LLM 起源。 |
| | **DBIM / CDBM** | senior | ICLR 2025 / NeurIPS 2024 | [arXiv 2405.15885](https://arxiv.org/abs/2405.15885) / [arXiv 2410.22637](https://arxiv.org/abs/2410.22637) | 在 diffusion bridge 上分别做 implicit-model few-step 采样、consistency 训练。如果把 motion finetune 看成"从 base policy 到 motion policy 的 bridge",可作为另一种 framing。 |
| | **Aligning Diffusion Behaviors with Q-functions** | senior | NeurIPS 2024 | [OpenReview](https://openreview.net/pdf?id=Wd1DFLUp1M) | 用 Q-function reward 对 diffusion 行为策略做 finetune(RL/控制场景)。是 DiffusionNFT 在 reward 侧的概念前作。 |

## 6. Chi Zhang(西湖大学,Westlake-AGI-Lab)— sampler 蒸馏路线

| Tag | 论文 | 一作 / 角色 | 会议/年份 | 链接 | 创新点 (2-3 句) |
|---|---|---|---|---|---|
| | **EPD-Solver** | Beier Zhu / Chi Zhang corr. | ICCV 2025, arXiv 2507.14797 | [project](https://epd-solver.github.io/) | 在每个 ODE step 上加并行梯度评估,用蒸馏学这些可学权重,5 NFE 在 CIFAR-10 上 FID 4.47。**"freeze 模型 + 学 sampler 参数"的代表**——和我们"freeze sampler + 蒸馏模型"正好对偶。 |
| | **AdaSDE** | Ruoyu Wang / Chi Zhang corr. | NeurIPS 2025, arXiv 2510.23285 | [code](https://github.com/Westlake-AGI-Lab/AdaSDE) | 单步 SDE solver,每步一个轻量蒸馏估计的可学 coefficient,融合 ODE 效率与 SDE 误差容忍。可解释 LongLive teacher SDE 与 student ODE 间的 motion gap。 |
| | **EPD-Solver V2 / RDPO** | Beier Zhu / Chi Zhang corr. | arXiv 2512.22796 | [arXiv](https://arxiv.org/abs/2512.22796) | 用 Residual Dirichlet Policy Optimization(RL)代替纯蒸馏来优化 parallel-gradient 权重。RL-on-sampler 思路与 rCM / v1 LoRA RL finetune 同源。 |
| ★ | **DyWeight** | Tong Zhao / Chi Zhang corr. | arXiv 2603.11607 | [code](https://github.com/Westlake-AGI-Lab/DyWeight) | 学一个动态聚合历史梯度的权重,等价于隐式调整有效步长;NFE=3 在 CIFAR-10 FID 8.16(EPD 10.40,−21.5%),SD/FLUX-dev 上同样有效。**完全不动 model**,可作为 Wan student 上的 sampler 升级,符合 `dmd_architecture_frozen`。 |

> 这条线整体是**"freeze model + learn lightweight solver via distillation/RL"**,与我们 motion-DMD 正交,可叠加使用。

## 7. Zeke Xie(HKUST-GZ, xLeaF Lab)

| Tag | 论文 | 角色 | 会议/年份 | 链接 | 创新点 (2-3 句) |
|---|---|---|---|---|---|
| ★ | **AMD (Adaptive Matching Distillation)** | senior | arXiv 2602.07345 | [arXiv](https://arxiv.org/abs/2602.07345) | 诊断 DMD few-step 训练的"Forbidden Zones":真实 teacher 在某些区域不可靠 + fake teacher 排斥不足。提出 reward-proxy gradient prioritization + repulsive landscape sharpening。**直接对应我们 motion-DMD 实际遇到的失败模式**,是最优先要复现的诊断脚本。 |
| ★ | **MagicDistillation** | senior | arXiv 2503.13319 | [arXiv](https://arxiv.org/abs/2503.13319) | LoRA fake-DiT + Weak-to-Strong distribution matching 在 portrait video 上 4-step 匹配 Wan-I2V/HunyuanVideo 50-step。**配方与我们 pipeline 同构**(LoRA fake + DMD on video DiT),paper 本身就在 cite Wan baseline。 |
| | **W2SD (Weak-to-Strong with Reflection)** | senior | ICLR 2026, arXiv 2502.00473 | [arXiv](https://arxiv.org/abs/2502.00473) | 采样阶段用 (strong − weak) 模型差异在 reflection 中把 latent 推向真实流形,适用 UNet/DiT/MoE/video。是 MagicDistillation 的 sampling-time 变体。 |
| | **PISA (Piecewise Sparse Attention)** | senior | arXiv 2602.01077 | [arXiv](https://arxiv.org/abs/2602.01077) | Training-free 分块稀疏 attention,Wan2.1-14B 上 1.91×、Hunyuan-Video 上 2.57×。可与 step distillation 正交叠加。 |
| | **Z-Sampling (Zigzag Diffusion Sampling)** | senior | ICLR 2025, arXiv 2412.10891 | [arXiv](https://arxiv.org/abs/2412.10891) | 交替 denoise↔invert 利用 CFG guidance gap 累积语义,HPSv2 win-rate ↑94%。Plug-and-play,可直接跑在我们 val prompts 上做对照。 |
| | **Golden Noise** | senior | ICCV 2025, arXiv 2411.09502 | [arXiv](https://arxiv.org/abs/2411.09502) | 学一个 prompt-conditioned noise perturbation("noise prompt"),把任意 Gaussian 转成"golden"噪声。零成本提升 val prompt 出图质量。 |
| | **Not All Noises Are Created Equally** | senior | ICME 2026, arXiv 2407.14041 | [arXiv](https://arxiv.org/abs/2407.14041) | 用 inversion stability 排序噪声,对 noise 而非模型做无 finetune 优化。同样属于 noise-side 干预。 |
| | **CoRe² (Collect, Reflect, Refine)** | senior | arXiv 2503.09662 | [arXiv](https://arxiv.org/abs/2503.09662) | 训一个"easy components"的 weak surrogate,然后用 W2S 引导 high-frequency 细节;在 SDXL/SD3.5/FLUX/LlamaGen 上验证。把 W2S 思路推广到 T2I。 |
| | **RF-Sampling (Reflective Flow Sampling)** | senior | arXiv 2026 | [OpenReview](https://openreview.net/forum?id=vc3P7CcBLe) | Training-free reflection sampler,**专为 CFG-distilled flow models(FLUX-style)设计**——最贴近我们 distilled-flow 设置。 |
| | **FastLightGen** | senior | arXiv 2603.01685 | [arXiv](https://arxiv.org/abs/2603.01685) | 视频 DiT 联合压缩 step 数与参数量。 |
| | **Alignment of Diffusion Models (Survey)** | senior | ACM CSUR 2026, arXiv 2409.07253 | [arXiv](https://arxiv.org/abs/2409.07253) | 扩散模型 alignment 方法的综述与 taxonomy。可作为入口文献。 |

## 8. ICL / TTT / Reference-injection for video

| Tag | 论文 | 类别 | 会议/年份 | 链接 | 创新点 (2-3 句) |
|---|---|---|---|---|---|
| ★ | **Video Creation by Demonstration (δ-Diffusion)** | ICL / reference-injection | arXiv 2412.09551, DeepMind/Cornell | [arXiv](https://arxiv.org/abs/2412.09551) | Demo video + context image → 让 context image 中的对象去做 demo 的 action。设计了 appearance-bottleneck encoder,从 demo 抽出"action latent"且最小化外观泄露。**最贴近"猫扣篮"用例**。 |
| ★ | **Video DiTs are In-Context Learners** | ICL | arXiv 2412.10783, Kuaishou Kling | [arXiv](https://arxiv.org/abs/2412.10783) | 把 reference 与 target 沿 spatial/time 轴拼起来 + joint caption,只做轻量任务级 tuning,DiT 自动学到 in-context 模式。架构改动极小,适合在 Wan 上做对比。 |
| ★ | **DiTFlow** | reference-injection | CVPR 2025, arXiv 2412.07776 | [arXiv](https://arxiv.org/abs/2412.07776) | 从参考视频的 cross-frame attention 中提 patch-wise Attention Motion Flow(AMF),在推理时通过 latent optimization 注入到目标。**training-free 的高质量 motion transfer**,可作零 finetune baseline。 |
| | **Vid-ICL** | ICL | arXiv 2407.07356, Tsinghua/Bytedance | [arXiv](https://arxiv.org/abs/2407.07356) | 自回归 transformer 把 demonstration video + query clip 当 in-context 样本,零样本生成模仿 demo motion 的续片。pure next-token,无权重更新。 |
| | **AICL (Action ICL)** | ICL | arXiv 2403.11535 | [arXiv](https://arxiv.org/abs/2403.11535) | 用一段 reference video 教模型一个 action 概念,通过 in-context 条件把动作泛化到未见主体。 |
| | **VideoPoet** | ICL | ICML 2024, arXiv 2312.14125, Google | [arXiv](https://arxiv.org/abs/2312.14125) | Decoder-only LLM 把 video/image/text/audio token 化混合,串多种任务训练后零样本支持改 motion 等编辑。 |
| | **VACE** | reference-injection / ICL | ICCV 2025, arXiv 2503.07598, Ali-vilab | [arXiv](https://arxiv.org/abs/2503.07598) | 统一 Video Condition Unit 把 reference videos / masks / images 一次性打包成 DiT 输入,不需要每任务 LoRA。 |
| | **TTT-DiT (One-Minute Video)** | TTT | arXiv 2504.05298, Stanford/NVIDIA | [arXiv](https://arxiv.org/abs/2504.05298) | 在预训练 DiT 中插入 TTT 层,这些层的隐藏状态在推理时根据输入 storyboard 自训,长片段一致性 +34 Elo。 |
| | **CustomTTT** | TTT / reference-injection | arXiv 2412.15646 | [arXiv](https://arxiv.org/abs/2412.15646) | 在单 reference video 上做 per-sample TTT,用 layer-wise selective update 同时定制 motion 与 appearance,避免 catastrophic forgetting。 |
| ★ | **MotionDirector** | TTT / reference-tune | ECCV 2024 Oral, arXiv 2310.08465 | [code](https://github.com/showlab/MotionDirector) | 双路 LoRA(spatial-LoRA 吸 appearance,temporal-LoRA 抓 motion)+ appearance-debiased temporal loss,从单参考视频做 motion fitting。最经典的 motion-vs-appearance 解耦工作。 |
| | **Customize-A-Video** | TTT | arXiv 2402.14780 | [arXiv](https://arxiv.org/abs/2402.14780) | 单参考 clip one-shot TTT,用 appearance absorber 模块吸收外观,使 LoRA 只承载 motion。 |
| | **MotionShop** | reference-injection | arXiv 2412.05355 | [arXiv](https://arxiv.org/abs/2412.05355) | Sampling 时用 Mixture-of-Score-Guidance 从参考视频拉 motion,无 finetune。 |
| | **MotionFlow** | reference-injection | arXiv 2412.05275 | [arXiv](https://arxiv.org/abs/2412.05275) | 推理时操控 attention 把参考视频 motion 改导向新主体,零训练。 |
| | **Motion Inversion** | reference-injection | SIGGRAPH 2025, arXiv 2403.20193 | [code](https://github.com/EnVision-Research/MotionInversion) | 把 reference video 反演成两个 1D motion embedding(分别注入 temporal-attn 的 Q-K 和 V),参数量极小且支持 motion arithmetic。 |
| | **Reenact Anything** | reference-injection | arXiv 2408.00458 | [arXiv](https://arxiv.org/abs/2408.00458) | 在参考视频上优化一个"motion text token",然后用该 token prompt 模型,实现语义级 motion transfer。 |
| | **DreamVideo-2** | reference-injection | arXiv 2410.13830, Ali/Fudan | [arXiv](https://arxiv.org/abs/2410.13830) | 单参考图 + bbox 轨迹直接驱动 masked reference attention,同时定 subject 与 motion;**zero-shot,无 test-time tuning**。 |

## 9. LoRA 变体(模态/结构特化)

| Tag | 论文 | 类别 | 会议/年份 | 链接 | 创新点 (2-3 句) |
|---|---|---|---|---|---|
| | **DoRA** | generic | ICML 2024 Oral, arXiv 2402.09353 | [arXiv](https://arxiv.org/abs/2402.09353) | 把预训练权重分解为 magnitude × direction,LoRA 只更新 direction。无推理开销,几乎闭合 LoRA–full FT 差距,可作所有 LoRA 的 init drop-in。 |
| | **AdaLoRA** | generic / structure-aware | ICLR 2023, arXiv 2303.10512 | [arXiv](https://arxiv.org/abs/2303.10512) | 把 LoRA 写成 SVD 形式,根据 importance 修剪不重要的 singular triplet,从而把 rank 预算重新分配到关键 matrix 上。 |
| | **VeRA** | generic | ICLR 2024, arXiv 2310.11454 | [arXiv](https://arxiv.org/abs/2310.11454) | 所有层共享一对 frozen random A、B,只学每层小 scaling 向量,参数量再降 ~10×。 |
| | **LoRA-FA** | generic | arXiv 2308.03303 | [arXiv](https://arxiv.org/abs/2308.03303) | 冻 A 只训 B,激活内存 1.4× 下降而精度不降。 |
| | **LoRA+** | generic | ICML 2024, arXiv 2402.12354 | [arXiv](https://arxiv.org/abs/2402.12354) | 理论证明 A、B 应该使用不同学习率(B 远高于 A),只调 LR 配比就能 1–2% 提升、~2× 收敛加速,零额外算力。 |
| | **PiSSA** | generic | NeurIPS 2024 Spotlight, arXiv 2404.02948 | [arXiv](https://arxiv.org/abs/2404.02948) | 用 W 的 top SVD 成分初始化 A、B(剩余冻结),收敛更快、终点更优。LoRA 起点的现成升级。 |
| | **Tied-LoRA** | generic | arXiv 2311.09578 | [arXiv](https://arxiv.org/abs/2311.09578) | 跨层共享 A、B 并选择性训练,以一小部分 LoRA 参数维持相近性能。 |
| | **HydraLoRA** | generic / MoE-like | NeurIPS 2024 Oral, arXiv 2404.19245 | [arXiv](https://arxiv.org/abs/2404.19245) | 1 个共享 A + 多个任务专属 B 头 + 学习路由,把通用结构与任务特异结构分开。 |
| | **MoLE (Mixture of LoRA Experts)** | generic / diffusion | ICLR 2024, arXiv 2404.13628 | [arXiv](https://arxiv.org/abs/2404.13628) | 多 LoRA 当 expert,层级 gating 融合,解决"算术合并 LoRA 会掉性能"的问题。 |
| | **Custom Diffusion** | diffusion / structure-aware | CVPR 2023, arXiv 2212.04488 | [arXiv](https://arxiv.org/abs/2212.04488) | 实证发现只需更新 cross-attention 的 K、V(~3% 权重)即可学新概念。layer-selection 这个洞察是后续 LoRA 设计的基础。 |
| | **SVDiff** | diffusion / structure-aware | ICCV 2023, arXiv 2303.11305 | [arXiv](https://arxiv.org/abs/2303.11305) | 只 fine-tune 权重的 singular values(spectral shifts),checkpoint 只 1.7 MB,比 DreamBooth 小 2200×。 |
| | **OFT / BOFT** | diffusion | NeurIPS 2023 / ICLR 2024 | [OFT](https://arxiv.org/abs/2306.07280) / [BOFT](https://arxiv.org/abs/2311.06243) | 用乘法式正交更新代替加法式 low-rank,保持 hyperspherical energy 从而保留 base model 语义;BOFT 加 butterfly factorization 进一步省参数。 |
| | **Mix-of-Show (ED-LoRA)** | diffusion | NeurIPS 2023, arXiv 2305.18292 | [arXiv](https://arxiv.org/abs/2305.18292) | Embedding-decomposed LoRA 单概念调,然后中心节点做 gradient fusion 支持任意多概念组合。 |
| ★ | **AnimateDiff / MotionLoRA** | video / structure-aware | ICLR 2024 Spotlight, arXiv 2307.04725 | [arXiv](https://arxiv.org/abs/2307.04725) | LoRA 只加在 temporal motion module 上,~77 MB 学一种镜头/动作类型。"LoRA 只在时间层"的母版。 |
| ★ | **MotionDirector** | video / structure-aware | ECCV 2024 Oral, arXiv 2310.08465 | (见 §8) | Spatial-LoRA 吸 appearance + Temporal-LoRA 学 motion + appearance-debiased temporal loss。motion-vs-appearance 解耦的标准做法。 |
| ★ | **VMC (Video Motion Customization)** | video | CVPR 2024, arXiv 2312.00845 | [arXiv](https://arxiv.org/abs/2312.00845) | 单参考视频 one-shot tune,**只改 temporal-attention 层**,用相邻帧 noise residual 作 motion 蒸馏目标。再次证明 motion 主要住在时间注意力上。 |
| ★ | **Customize-A-Video** | video | ECCV 2024, arXiv 2402.14780 | [arXiv](https://arxiv.org/abs/2402.14780) | Temporal-attention LoRA + appearance absorber,前者承载 motion 后者吸外观,两者训练目标分离。 |
| | **Motion Inversion** | video / structure-aware | (见 §8) | (见 §8) | 不用 LoRA,而是把 motion 编码成两个 1D embedding 分别注入 temporal-attn 的 Q-K 与 V——证明 attn map 与 attn value 是两个不同 motion 通道。 |
| | **LiON-LoRA** | video / structure-aware | ICCV 2025, arXiv 2507.05678 | [arXiv](https://arxiv.org/abs/2507.05678) | DiT 视频模型 LoRA 融合三原则:Linear scalability、shallow-layer Orthogonality、跨层 Norm consistency。给"LoRA 加在哪一层效果如何"做了系统研究。 |

## 10. 视频生成稀疏注意力

| Tag | 论文 | 类别 | 会议/年份 | 链接 | 创新点 (2-3 句) |
|---|---|---|---|---|---|
| ★ | **PAB (Pyramid Attention Broadcast)** | training-free | ICLR 2025, arXiv 2408.12588 | [arXiv](https://arxiv.org/abs/2408.12588) | 观察到 attention 在 step 间呈 U 形差异,据此跨 step 以"金字塔"形式广播 attention 输出。OpenSora/OpenSora-Plan/Latte 上 10.6× / 21.6 FPS,完全免训练。 |
| | **DiTFastAttn** | training-free | NeurIPS 2024, arXiv 2406.08552 | [arXiv](https://arxiv.org/abs/2406.08552) | Post-training 组合 window-attn + residual sharing + cross-step sharing + CFG sharing,DiT/PixArt-Σ/OpenSora 上 1.8× 端到端。 |
| | **Sparse VideoGen (SVG)** | training-free | ICML 2025, arXiv 2502.01776 | [arXiv](https://arxiv.org/abs/2502.01776) | 在线 profile 把每个 head 标为 spatial vs temporal sparse pattern,CogVideoX-1.5 / HunyuanVideo 上 2.28× / 2.33×。 |
| | **Sparse VideoGen 2 (SVG2)** | training-free | NeurIPS 2025 Spotlight, arXiv 2505.18875 | [arXiv](https://arxiv.org/abs/2505.18875) | 用 semantic-aware k-means 重排 token 把"关键 token"打包成 dense block,HunyuanVideo / Wan2.1 上 2.30× / 1.89×。 |
| ★ | **STA (Sliding Tile Attention)** | trainable | ICML 2025, arXiv 2502.04507 | [arXiv](https://arxiv.org/abs/2502.04507) | 3D tile-by-tile windowed attention,硬件感知。HunyuanVideo 训练免费下 945→685s,finetune 后 268s 且 VBench 仅 −0.09%。FA2 之上 2.8–17×。 |
| ★ | **Radial Attention** | trainable | NeurIPS 2025, arXiv 2506.19852 | [arXiv](https://arxiv.org/abs/2506.19852) | 基于"Spatiotemporal Energy Decay"现象给出**静态 O(n log n) mask**(window 随时序距离衰减),Wan2.1-14B / HunyuanVideo / Mochi-1 上推理 1.9× / 训练 4.4×,允许 4× 更长视频。 |
| ★ | **VSA (Video Sparse Attention)** | trainable | NeurIPS 2025, arXiv 2505.13389 | [arXiv](https://arxiv.org/abs/2505.13389) | **同时 train+infer 都稀疏**,coarse-tile selection + fine block compute 单 differentiable kernel(85% FA3 MFU);训练 FLOPs 2.53×,Wan-2.1 retrofit 后 attention 6× / e2e 31→18s。 |
| | **Sparse-vDiT** | training-free | arXiv 2506.03065 | [arXiv](https://arxiv.org/abs/2506.03065) | 离线对每层每头搜 diagonal / multi-diagonal / vertical-stripe 中最优 pattern,CogVideoX-1.5 / HunyuanVideo / Wan2.1 上 1.76 / 1.85 / 1.58×。 |
| | **AdaSpa** | training-free | ICCV 2025, arXiv 2502.21079 | [arXiv](https://arxiv.org/abs/2502.21079) | Training-free 分块 hierarchical sparsity + Fused LSE-cached online search,HunyuanVideo 等上 1.59–2.04×,完全 plug-and-play。 |
| | **LiteAttention** | training-free | arXiv 2511.11062 | [arXiv](https://arxiv.org/abs/2511.11062) | 利用稀疏模式在 denoising step 间的时间相干性,标记后跳过+传播,免去逐步 profiling。Wan2.1/2.2 上 ~1.9× / 54% sparsity 零质量损失。 |
| | **TurboDiffusion / SLA** | trainable | arXiv 2026, [code](https://github.com/thu-ml/TurboDiffusion) | (见 §5) | 可训 Sparse-Linear Attention 叠在 SageAttention 上,attention 17–20× / e2e 100–200×。 |
| | **SpargeAttention** | training-free | ICML 2025, arXiv 2502.18137 | [arXiv](https://arxiv.org/abs/2502.18137) | 通用 training-free 两阶段在线过滤(预测 block-mask + softmax-aware filter),Mochi 1.83×、Wan/Hunyuan 也适用。 |
| | **Importance-Based Token Merging** | token reduction | arXiv 2411.16720 | [arXiv](https://arxiv.org/abs/2411.16720) | 用 CFG 派生的 importance 引导 token merging 偏向,T2I/multi-view/video DiT 通用。 |
| | **VidToMe** | token reduction | CVPR 2024 | [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_VidToMe_Video_Token_Merging_for_Zero-Shot_Video_Editing_CVPR_2024_paper.pdf) | 跨帧 token merging 做零样本视频编辑(SD-style UNet 时代),开创 video token 合并思路。 |

## 11. Motion editing / transfer + ICCV 2025 awards

| Tag | 论文 | 类别 | 会议/年份 | 链接 | 创新点 (2-3 句) |
|---|---|---|---|---|---|
| ★ | **MotionDirector** | reference-tune | ECCV 2024 Oral | (见 §8/§9) | 双路 LoRA 解耦 appearance/motion + appearance-debiased temporal loss。开源完整,**最贴近用户任务的可复现 baseline**。 |
| ★ | **VMC** | reference-tune | CVPR 2024, arXiv 2312.00845 | (见 §9) | 用相邻 latent 帧 noise residual 作 motion 信号,只 fine-tune temporal attention。residual-based motion 表达的代表作。 |
| ★ | **MotionMatcher** | reference-tune | arXiv 2502.13234 | [arXiv](https://arxiv.org/abs/2502.13234) | 不在像素而是在高层 spatio-temporal motion features 上做 matching。**与我们 v1 路线 attn-output L2 思路同源**——是 MotionDirector 的 feature-level 替代。 |
| | **MotionInversion** | reference-tune | SIGGRAPH 2025 | (见 §8) | 把 motion 反演成 1D embedding 注入 temporal self-attention,backbone 完全冻结。 |
| | **DreamVideo-2** | reference-tune | arXiv 2410.13830 | (见 §8) | Zero-shot subject-driven + bbox 引导 motion control,一次 inference 同时定 subject + motion。 |
| ★ | **Customize-A-Video** | one-shot / reference-tune | ECCV 2024 | (见 §9) | One-shot 单参考视频学 motion,Appearance Absorber LoRA 解耦外观。 |
| ★ | **Still-Moving** | reference-tune / one-shot | NeurIPS 2024, arXiv 2407.08674 | [arXiv](https://arxiv.org/abs/2407.08674) | **不需要任何 customized video data**:用 customized image + frozen video model 桥接,把 T2I LoRA 直搬到 T2V。降低数据门槛。 |
| ★ | **MotionClone** | tuning-free | ICLR 2025, arXiv 2406.05338 | [arXiv](https://arxiv.org/abs/2406.05338) | 从参考视频提 temporal-attention map,推理时直接 plug-in 引导。零成本 baseline,任何 tuning 方法的下界。 |
| | **TokenFlow** | tuning-free | ICLR 2024, arXiv 2307.10373 | [code](https://github.com/omerbt/TokenFlow) | 用 nearest-neighbor token propagation 在帧间扩散特征,实现零样本视频编辑;后续帧间一致性工作的基石。 |
| | **FreeNoise** | tuning-free | ICLR 2024, arXiv 2310.15169 | [arXiv](https://arxiv.org/abs/2310.15169) | Local noise shuffling + window-based attention fusion,无训练扩长度并支持多 prompt motion 注入,仅 +17% 时间。 |
| | **Tune-A-Video** | one-shot | ICCV 2023, arXiv 2212.11565 | [arXiv](https://arxiv.org/abs/2212.11565) | One-shot tuning 鼻祖,在 spatio-temporal attention 上单 (text, video) 对训练。所有后续 one-shot 论文的基线。 |
| | **LAMP** | one-shot | CVPR 2024, arXiv 2310.10769 | [arXiv](https://arxiv.org/abs/2310.10769) | 8–16 视频单 GPU 学 motion pattern,first-frame-conditioned pipeline 让 T2I 出内容、video model 专注 motion。 |
| ★ | **MotionBooth** | one-shot / reference-tune | NeurIPS 2024, arXiv 2406.17758 | [arXiv](https://arxiv.org/abs/2406.17758) | 少量 subject 图 + subject region loss + cross-attention loss,object motion 与 camera motion 分别可控。**最贴近"猫(subject)+ 扣篮(motion)"双重定制**。 |
| | **Tora** | trajectory | CVPR 2025, arXiv 2407.21705 | [code](https://github.com/alibaba/Tora) | **首个 DiT-based trajectory control**(基于 OpenSora),长视频高分辨率不再受 16 帧限制。CogVideoX/Wan 类 DiT 时代的代表方法。 |
| | **DragAnything** | trajectory | ECCV 2024, arXiv 2403.07420 | [arXiv](https://arxiv.org/abs/2403.07420) | 用 entity representation(SAM mask 中心特征)替代像素 drag,实现任意物体 + 背景的轨迹控制,语义比 DragNUWA 更强。 |
| | **MOFA-Video** | trajectory | ECCV 2024, arXiv 2405.20222 | [arXiv](https://arxiv.org/abs/2405.20222) | 把稀疏 trajectory 转 dense motion field 再适配 frozen SVD,介于 trajectory 与 reference-video 之间的中间路线。 |
| ★ | **FlowEdit (ICCV 2025 Best Student Paper)** | finalist | ICCV 2025, arXiv 2412.08629 | [arXiv](https://arxiv.org/abs/2412.08629) | **Inversion-free** text-based editing on pre-trained flow models(FLUX/SD3),不做反演直接在流空间内编辑。直接对 video editing 与 motion transfer 思路可迁移。 |
| | BrickGPT (ICCV 2025 Marr Prize) | finalist (off-topic) | ICCV 2025 | — | text → 稳定 LEGO 结构,与视频/motion 无关。仅作 ICCV 2025 awards 完整列表参考。 |

> 其他 ICCV 2025 honorable mentions(Spatially-Varying Autofocus、RayZer)同样与 video/motion 无直接关联,不展开。

---

# 可推进方向(讨论候选)

**A. DMD 病灶诊断 + W2S 修复**(Zeke Xie 系)
- 关键论文:AMD(诊断 Forbidden Zones)+ MagicDistillation(LoRA fake DiT + W2S distribution matching)。
- 与我们 motion-DMD 镜像得吓人。最低成本动作:把 AMD 的诊断脚本套到 motion-DMD 上,看 fake teacher 排斥力是否同样不足。
- 风险:论文很新,代码可能不全。

**B. NFT/rCM 替换 DMD 这条线**(Jun Zhu 系)
- 关键论文:DiffusionNFT(reward-based finetune)+ rCM(continuous-time consistency)+ TurboDiffusion(工程落地)。
- 整体替换违反 `dmd_architecture_frozen`,但 NFT 的 negative-aware loss 可单独加在 DMD 之外做 reward shaping。
- 风险:边界要重新确认。

**C. Motion-locus + structural LoRA + 几何/feature loss 升级**(MotionDirector / VMC / MotionMatcher / Geometry Forcing)
- 共识:motion 住在 temporal attention,LoRA 加那一层 + dual-LoRA 解耦 appearance,我们 v1 路线已采用。
- 下一步:把 attn-output L2 换成 MotionMatcher 的 feature-level loss,再叠加 Geometry Forcing 的 VGGT REPA loss。
- 风险:VGGT forward 的显存,可能要 offline 抽特征。
- **个人最看好,符合所有硬约束(DMD 不动、不加 encoder)**。

**D. Reference-as-context 不 finetune 路线**(MotionClone / DiTFlow / δ-Diffusion)
- 三条不动权重的捷径,zero-shot 出"猫扣篮"。
- 风险:跟 LongLive 主线偏离,只能作为 baseline 对照,不构成研究故事。

**推荐**:**A**(诊断驱动)+ **C**(loss 升级)的组合,贴在 `dmd_architecture_frozen` 硬约束内,工作量可控,与 v1 路线进度连贯。
