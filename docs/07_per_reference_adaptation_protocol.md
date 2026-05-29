# 07 — Per-Reference Adaptation Protocol

**Date**: 2026-05-29

**Status**: protocol and method framing decision before implementing the next
per-reference fast-adaptation experiments.

**Supersedes for new experiments**: the old one-LoRA full-sweep protocol where
a Skateboarding-adapted LoRA is evaluated on all UCF, LOVEU, and VBench prompts.
That old protocol remains useful only as a base-preservation / contamination
stress test.

---

## 1. Research Target

The product setting is:

```text
given one reference video -> quickly distill one LoRA -> evaluate that LoRA on
the prompts attached to the same reference motion
```

Therefore the primary unit of training and evaluation is not a dataset-wide
LoRA. It is a **reference unit**:

```text
unit = one reference video + one train caption + one prompt set
```

Each selected unit owns an independent adaptation run and checkpoint. Aggregate
dataset scores are averages over the selected units' own prompts.

---

## 2. Default Evaluation Protocol

### 2.1 UCF Sports

UCF has 10 action categories and 6 cross-appearance prompts per category.

Default UCF protocol:

```text
for each selected UCF category:
  choose one reference video for that category
  train one LoRA on that reference video
  evaluate only that category's 6 subject-swap prompts

default full UCF = 10 LoRAs x 6 prompts = 60 prompt evaluations
```

The default selected reference is deterministic: use the first sorted clip from
the category manifest unless the config provides explicit unit ids or paths.
Future variants can support `refs_per_category > 1`, but the default should
stay one reference per category to keep cost and interpretation controlled.

### 2.2 LOVEU-TGVE

LOVEU has one source/reference video per CSV row and four edit prompts in the
current motion-eval manifest:

- `Style Change Caption`
- `Object Change Caption`
- `Background Change Caption`
- `Multiple Changes Caption`

Default LOVEU protocol:

```text
for each selected LOVEU source video:
  train one LoRA on that source/reference video
  evaluate only that video's 4 edit prompts

default full LOVEU = 76 LoRAs x 4 prompts = 304 prompt evaluations
```

Do not include `Our GT caption` by default. If a future experiment enables it,
that should be an explicit config switch because it changes LOVEU from 4 to 5
prompts per video.

### 2.3 VBench

VBench has no paired reference video for each prompt, so it is not part of the
default per-reference adaptation metric.

Use VBench only as an optional base-preservation stress test on a small set of
representative adapted LoRAs. It answers:

```text
did this adaptation contaminate general prompts / semantic quality?
```

It does not answer whether the reference motion was distilled correctly.

---

## 3. Configurable Scale

Every experiment should expose protocol scale in config, so smoke runs and full
runs share the same code path.

Recommended schema:

```yaml
protocol:
  name: per_reference_adaptation
  seed: 0

  datasets:
    ucf:
      enabled: true
      categories: all        # or [Skateboarding, Golf-Swing]
      max_categories: null   # smoke: 1 or 2
      refs_per_category: 1
      prompts_per_unit: all  # or integer

    loveu:
      enabled: true
      max_units: null        # smoke: 4 or 8
      unit_ids: null         # optional explicit Video name list
      prompt_types:
        - Style Change Caption
        - Object Change Caption
        - Background Change Caption
        - Multiple Changes Caption

  vbench:
    enabled: false
    adapted_unit_ids: []     # run VBench only for listed adapted LoRAs
```

The protocol runner should materialize a resolved unit manifest before starting
training. That manifest should be written to the run directory and should record:

```text
unit_id
dataset
reference_video_path
train_caption
eval_prompts
teacher
student_lora_out_dir
motion_eval_run_id
```

---

## 4. Method Framing

The next method family should unify EM-RAM and on-policy distillation under one
constraint:

```text
teacher: fixed LongLive few-step product
reference: video-level motion signal / preference / target-distribution tilt
student: current on-policy LoRA rollout
```

The reference video must enter the optimization directly. It is not represented
by a trained reference LoRA. The method should therefore be understood as
reference-conditioned adaptation around a fixed few-step teacher, not as
teacher-LoRA-to-student-LoRA distillation.

### 4.1 EM-RAM View

EM-RAM provides the current strongest concrete mechanism:

```text
E-step:
  sample current-student on-policy rollouts
  score each rollout against the reference video
  build a reward-tilted empirical target distribution under a KL budget

M-step:
  update only the allowed motion subspace around the LongLive teacher
  keep static / appearance / pixel-heavy directions anchored to LongLive
```

This already matches the target invariance:

```text
motion may move toward the reference;
appearance and pixel details do not receive reward credit by default.
```

### 4.2 On-Policy Distillation View

On-policy distillation is still valuable, but not as "match a separate
reference teacher velocity." In this project it should mean:

```text
optimize on states visited by the current student, while the reference video
defines a tilted target distribution or update weights, and the fixed LongLive
teacher supplies the base score/velocity anchor
```

In other words, the teacher term is LongLive preservation; the reference term is
the distribution tilt. A pure velocity loss to LongLive is only anti-drift.

### 4.3 Candidate Objectives

The initial implementation should start from the most grounded objective:

```text
per-reference MP-EM-RAM
```

Then the more theoretical extension is:

```text
EM-weighted on-policy forward-KL / mirror-descent variant
```

A useful abstract form is:

```text
q_ref(x) ∝ p_student(x) exp(r_ref(x) / eta)
minimize a projected / trust-region update from p_student toward q_ref
around the fixed LongLive teacher
```

The implementation can remain close to the current MP-EM-RAM trainer:

```text
target = v_longlive + alpha_ref * P_motion(reference_weighted_residual)
loss   = ||v_student - stopgrad(target)||^2
```

where `alpha_ref` comes from EM/RAM and `P_motion` encodes "motion consistent,
pixel may differ."

---

## 5. Teacher Policy

The teacher in this project is the released few-step LongLive product:

```text
teacher = longlive_base.pt + lora.pt
```

It is not a per-reference MotionDirector LoRA, not a cached reference-specific
adapter, and not an oracle teacher. Training such an adapter would change the
problem definition and is not part of the default protocol.

The intended framework is:

```text
frozen few-step LongLive teacher + one reference video/signal
  -> distill/adapt into a few-step student LoRA/model
```

Suggested config shape:

```yaml
teacher:
  type: longlive_fewstep
  base_ckpt: ${oc.env:LL_DATA}/longlive_models/models/longlive_base.pt
  baseline_lora_ckpt: ${oc.env:LL_DATA}/longlive_models/models/lora.pt
```

The reference video is not a teacher checkpoint. It enters the optimization via
reward, motion features, update weighting, or another explicitly defined
reference signal. A loss that only matches the student velocity to the fixed
LongLive teacher velocity is just base preservation; by itself it cannot inject
reference motion.

For UCF/LOVEU unit construction, implementation still needs a generic
`reference_video_path` dataset, because the existing `SkateboardingLatentDataset`
is category/manifest based and does not accept an arbitrary source video path.

---

## 6. Student Adaptation Policy

Each unit then runs the chosen fast-adaptation method against the fixed LongLive
few-step teacher and the unit's reference video.

Pure on-policy teacher velocity matching is not a reference adaptation method in
this protocol:

```text
x0 ~ current student rollout for this unit's train caption
xt = noise(x0, t_anchor)
loss = ||v_student(xt, t, c_train) - stopgrad(v_longlive(xt, t, c_train))||^2
```

Because `v_longlive` has no reference-video conditioning, this loss is an
anti-drift/base-preservation control only. It should not be treated as the raw
baseline for motion distillation.

The baseline valid reference-adaptation objective is per-reference MP-EM-RAM:

```text
E-step:
  score on-policy rollouts against the unit reference video
  compute EM_alpha and component_feasibility

M-step:
  target = v_longlive + reward/reference-weighted motion residual
  keep static/appearance directions anchored to v_longlive
```

For future on-policy forward-KL variants, the fixed LongLive teacher should
still provide the base score/velocity, while the reference video defines the
tilted target distribution or update weights:

```text
weight_i = EM_alpha_i * component_feasibility_i
loss_i = weight_i * reference-motion update around v_longlive
```

This keeps the method aligned with the earlier MP-EM-RAM finding: useful
motion updates are selected by on-policy reference evidence, while the fixed
few-step teacher preserves the LongLive distribution.

---

## 7. Orchestration Requirements

One submitted sbatch should own the whole selected experiment:

```text
resolve units
for unit in units:
  load the fixed LongLive few-step teacher
  train student LoRA for this unit
  generate only this unit's eval prompts
  score motion_eval against this unit's reference video
aggregate selected-unit scores
optionally run VBench stress tests for selected adapted LoRAs
```

Do not submit one sbatch per unit by default. For smoke/full comparability,
`max_categories`, `max_units`, and `prompts_per_unit` should reduce the unit
manifest while preserving the same train -> eval path.

Every unit should get stable output paths:

```text
$LL_DATA/per_reference_adaptation_runs/{experiment}/{dataset}/{unit_id}/
$LL_DATA/motion_eval_runs/{experiment}_{slurm_job_id}/{dataset}/{unit_id}/
```

The aggregate result should keep the existing reporting convention:

1. UCF Sports table, if UCF enabled.
2. LOVEU-TGVE table, if LOVEU enabled.
3. VBench table, only if explicitly enabled.

---

## 8. Interpretation

The per-reference protocol changes the meaning of metrics.

Old one-LoRA full-sweep protocol:

```text
does a single adapted LoRA preserve broad prompt behavior?
```

New per-reference protocol:

```text
given the correct reference video, how well does fast adaptation transfer that
motion to the corresponding prompts without collapsing quality?
```

For new per-reference MP-EM-RAM and on-policy forward-KL variants, this protocol
is the primary evidence. The old full-sweep protocol can still be run after the
fact as a contamination check, but it should not be used as the main success
criterion.
