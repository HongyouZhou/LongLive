"""Run per-reference adaptation: resolve units, train, evaluate, aggregate.

This is the executable form of docs/07_per_reference_adaptation_protocol.md.
One process owns the whole selected experiment inside one allocation.  It
spawns method trainers with torchrun, then evaluates only the prompts attached
to the same reference video used during training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
UCF_PROMPTS_YAML = REPO_ROOT / "scripts" / "motion_eval" / "prompts" / "ucf_sports.yaml"
LOVEU_EDIT_COLUMNS = [
    "Style Change Caption",
    "Object Change Caption",
    "Background Change Caption",
    "Multiple Changes Caption",
]
LOVEU_VIDEO_COLUMN = "Video name"
LOVEU_CAPTION_COLUMN = "Our GT caption"
METRIC_COLUMNS = (
    "app_div",
    "temp_consist",
    "pick_score",
    "motion_fidelity",
    "dynamic_score",
)

METHOD_PRESETS: dict[str, dict[str, Any]] = {
    "base": {
        "kind": "baseline",
        "ckpt": "longlive_models/models/lora.pt",
        "eval_config": "configs/motion_eval_inference.yaml",
    },
    "ram_v1": {
        "kind": "train",
        "train_module": "longlive.methods.diffusion_ram.train",
        "train_template": "longlive/methods/diffusion_ram/configs/per_reference_ram.yaml",
        "eval_config": "configs/motion_eval_inference_diffusion_ram.yaml",
        "wandb_project": "longlive_per_reference_ram",
    },
    "em_ram": {
        "kind": "train",
        "train_module": "longlive.methods.em_ram.train",
        "train_template": "longlive/methods/em_ram/configs/per_reference_em_ram.yaml",
        "eval_config": "configs/motion_eval_inference_em_ram.yaml",
        "wandb_project": "longlive_per_reference_em_ram",
    },
    "mp_em_ram": {
        "kind": "train",
        "train_module": "longlive.methods.motion_projected_em_ram.train",
        "train_template": "longlive/methods/motion_projected_em_ram/configs/per_reference_mp_em_ram.yaml",
        "eval_config": "configs/motion_eval_inference_motion_projected_em_ram.yaml",
        "wandb_project": "longlive_per_reference_mpem",
    },
    "mp_em_ram_feature_weight": {
        "kind": "train",
        "train_module": "longlive.methods.motion_projected_em_ram.train",
        "train_template": "longlive/methods/motion_projected_em_ram/configs/per_reference_mp_em_ram_feature_weight.yaml",
        "eval_config": "configs/motion_eval_inference_motion_projected_em_ram.yaml",
        "wandb_project": "longlive_per_reference_mpem",
    },
}


def _select(cfg: DictConfig, key: str, default: Any = None) -> Any:
    value = OmegaConf.select(cfg, key)
    return default if value is None else value


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = safe.strip("._-")
    return safe or "unit"


def _hash_id(*parts: str, n: int = 10) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return h[:n]


def _as_data_rel(data_root: Path, path: str | Path) -> str:
    p = Path(os.path.expandvars(str(path))).expanduser()
    if not p.is_absolute():
        p = data_root / p
    if not p.exists():
        raise FileNotFoundError(f"reference video not found: {p}")
    return str(p.resolve().relative_to(data_root.resolve()))


def _prompt_limit(prompts: list[dict], value: Any) -> list[dict]:
    if value is None or str(value).lower() == "all":
        return prompts
    return prompts[: int(value)]


def _normalize_eval_prompts(raw_prompts: Any) -> list[dict]:
    rows = []
    for idx, item in enumerate(raw_prompts or []):
        if isinstance(item, str):
            prompt = item
            prompt_type = f"prompt_{idx}"
        else:
            prompt = str(item["prompt"])
            prompt_type = item.get("prompt_type", f"prompt_{idx}")
        if not prompt.strip():
            raise ValueError("eval prompt must be non-empty")
        rows.append({"prompt": prompt, "prompt_type": prompt_type, "prompt_idx": idx})
    if not rows:
        raise ValueError("explicit protocol unit requires at least one eval prompt")
    return rows


def _load_explicit_units(cfg: DictConfig, data_root: Path) -> list[dict]:
    raw_units = _select(cfg, "protocol.units", None)
    if not raw_units:
        return []
    units = []
    for idx, raw in enumerate(raw_units):
        reference = raw.get("reference_video", raw.get("reference_video_path", None))
        if not reference:
            raise ValueError(f"protocol.units[{idx}] missing reference_video")
        train_caption = str(raw.get("train_caption", "")).strip()
        if not train_caption:
            raise ValueError(f"protocol.units[{idx}] missing train_caption")
        dataset = str(raw.get("dataset", "custom"))
        unit_id = str(raw.get("unit_id", "")) or (
            f"{_safe_id(dataset)}_{Path(str(reference)).stem}_{_hash_id(str(reference), train_caption, n=6)}"
        )
        units.append({
            "unit_id": _safe_id(unit_id),
            "dataset": dataset,
            "reference_video": _as_data_rel(data_root, reference),
            "train_caption": train_caption,
            "eval_prompts": _normalize_eval_prompts(raw.get("eval_prompts", [])),
        })
    return units


def _load_ucf_units(cfg: DictConfig, data_root: Path) -> list[dict]:
    if not bool(_select(cfg, "protocol.datasets.ucf.enabled", False)):
        return []

    import yaml

    manifest = data_root / "ucf_sports" / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} missing; run scripts/prepare_motion_eval.py --datasets ucf")

    with open(UCF_PROMPTS_YAML) as f:
        prompt_spec = yaml.safe_load(f)

    by_cat: dict[str, list[str]] = {}
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row["path"]
            if not rel.startswith("ucf_sports/"):
                rel = f"ucf_sports/{rel}"
            by_cat.setdefault(row["category"], []).append(rel)
    for rels in by_cat.values():
        rels.sort()

    requested = _select(cfg, "protocol.datasets.ucf.categories", "all")
    if requested is None or str(requested).lower() == "all":
        categories = list(prompt_spec["categories"].keys())
    else:
        categories = list(requested)

    max_categories = _select(cfg, "protocol.datasets.ucf.max_categories", None)
    if max_categories is not None:
        categories = categories[: int(max_categories)]

    refs_per_category = int(_select(cfg, "protocol.datasets.ucf.refs_per_category", 1))
    prompts_per_unit = _select(cfg, "protocol.datasets.ucf.prompts_per_unit", "all")

    units = []
    for category in categories:
        if category not in prompt_spec["categories"]:
            raise KeyError(f"unknown UCF category in prompts yaml: {category!r}")
        if category not in by_cat:
            raise KeyError(f"UCF category has no prepared videos: {category!r}")
        entry = prompt_spec["categories"][category]
        eval_prompts = [
            {"prompt": prompt, "prompt_type": f"prompt_{idx}", "prompt_idx": idx}
            for idx, prompt in enumerate(entry["inference_prompts"])
        ]
        eval_prompts = _prompt_limit(eval_prompts, prompts_per_unit)
        for ref_idx, rel in enumerate(by_cat[category][:refs_per_category]):
            data_rel = _as_data_rel(data_root, rel)
            units.append({
                "unit_id": f"ucf_{_safe_id(category)}_r{ref_idx:02d}",
                "dataset": "ucf",
                "category": category,
                "reference_video": data_rel,
                "train_caption": entry["train_caption"],
                "eval_prompts": eval_prompts,
            })
    return units


def _is_loveu_data_row(row: dict[str, str]) -> bool:
    name = (row.get(LOVEU_VIDEO_COLUMN) or "").strip()
    if not name or name.endswith(":"):
        return False
    return any((row.get(c) or "").strip() for c in LOVEU_EDIT_COLUMNS)


def _load_loveu_units(cfg: DictConfig, data_root: Path) -> list[dict]:
    if not bool(_select(cfg, "protocol.datasets.loveu.enabled", False)):
        return []

    loveu_root = data_root / "loveu_tgve"
    prompts_csv = loveu_root / "prompts.csv"
    videos_dir = loveu_root / "videos"
    if not prompts_csv.exists():
        raise FileNotFoundError(
            f"{prompts_csv} missing; run scripts/prepare_motion_eval.py --datasets loveu"
        )

    requested_ids = _select(cfg, "protocol.datasets.loveu.unit_ids", None)
    requested_set = {str(x) for x in requested_ids} if requested_ids else None
    prompt_types = list(_select(cfg, "protocol.datasets.loveu.prompt_types", LOVEU_EDIT_COLUMNS))
    prompts_per_unit = _select(cfg, "protocol.datasets.loveu.prompts_per_unit", "all")
    max_units = _select(cfg, "protocol.datasets.loveu.max_units", None)

    units = []
    with open(prompts_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _is_loveu_data_row(row):
                continue
            video_id = row[LOVEU_VIDEO_COLUMN].strip()
            unit_id = f"loveu_{_safe_id(video_id)}"
            if requested_set and video_id not in requested_set and unit_id not in requested_set:
                continue

            candidates = sorted(videos_dir.glob(f"{video_id}.*"))
            if not candidates:
                raise FileNotFoundError(f"LOVEU video {video_id!r} missing under {videos_dir}")
            train_caption = (row.get(LOVEU_CAPTION_COLUMN) or "").strip()
            if not train_caption:
                raise ValueError(f"LOVEU row {video_id!r} has empty {LOVEU_CAPTION_COLUMN!r}")

            eval_prompts = []
            for prompt_idx, col in enumerate(prompt_types):
                prompt = (row.get(col) or "").strip()
                if prompt:
                    eval_prompts.append({
                        "prompt": prompt,
                        "prompt_type": col,
                        "prompt_idx": prompt_idx,
                    })
            eval_prompts = _prompt_limit(eval_prompts, prompts_per_unit)
            if not eval_prompts:
                continue

            units.append({
                "unit_id": unit_id,
                "dataset": "loveu",
                "video_id": video_id,
                "reference_video": _as_data_rel(data_root, candidates[0]),
                "train_caption": train_caption,
                "eval_prompts": eval_prompts,
            })
            if max_units is not None and len(units) >= int(max_units):
                break
    return units


def resolve_units(cfg: DictConfig) -> list[dict]:
    data_root = Path(str(cfg.data_root))
    units = _load_explicit_units(cfg, data_root)
    units.extend(_load_ucf_units(cfg, data_root))
    units.extend(_load_loveu_units(cfg, data_root))
    if not units:
        raise RuntimeError("per-reference protocol resolved zero units")
    return units


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_unit_prompt_manifest(path: Path, unit: dict) -> list[dict]:
    rows = []
    for prompt_idx, prompt_row in enumerate(unit["eval_prompts"]):
        prompt = prompt_row["prompt"]
        prompt_id = (
            f"{_safe_id(unit['unit_id'])}_{prompt_idx:02d}_"
            f"{_hash_id(unit['unit_id'], str(prompt_idx), prompt, n=8)}"
        )
        rows.append({
            "prompt_id": prompt_id,
            "dataset": unit["dataset"],
            "unit_id": unit["unit_id"],
            "key": {
                "unit_id": unit["unit_id"],
                "prompt_idx": prompt_idx,
                "prompt_type": prompt_row.get("prompt_type"),
            },
            "prompt": prompt,
            "ref_videos": [unit["reference_video"]],
            "paper_verbatim": unit["dataset"] == "loveu",
        })
    write_jsonl(path, rows)
    return rows


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    print("[per_ref] $ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _method_attr(method: DictConfig, name: str, default: Any = None) -> Any:
    value = OmegaConf.select(method, name)
    return default if value is None else value


def _resolve_method(method: DictConfig) -> DictConfig:
    name = str(method.name)
    preset_name = str(_method_attr(method, "preset", name))
    if preset_name in METHOD_PRESETS:
        preset = OmegaConf.create({"name": name, **METHOD_PRESETS[preset_name]})
        return OmegaConf.merge(preset, method)
    if _method_attr(method, "kind", None) is not None:
        return method
    raise KeyError(
        f"unknown method preset {preset_name!r}; available={sorted(METHOD_PRESETS)}"
    )


def _render_train_config(
    *,
    cfg: DictConfig,
    method: DictConfig,
    unit: dict,
    method_dir: Path,
    unit_index: int,
    dry_run: bool,
) -> Path:
    train_template = Path(str(method.train_template))
    train_cfg = OmegaConf.load(train_template)
    overrides = _method_attr(method, "overrides", None)
    if overrides is not None:
        train_cfg = OmegaConf.merge(train_cfg, overrides)

    unit_train_dir = method_dir / unit["dataset"] / unit["unit_id"] / "train"
    unit_scratch_dir = method_dir / unit["dataset"] / unit["unit_id"] / "scratch"
    train_cfg.data_root = str(cfg.data_root)
    train_cfg.reference_video_path = unit["reference_video"]
    train_cfg.train_caption = unit["train_caption"]
    train_cfg.cover_prompts = [row["prompt"] for row in unit["eval_prompts"]]
    train_cfg.unit_id = unit["unit_id"]
    train_cfg.out_dir = str(unit_train_dir)
    train_cfg.cache_dir = str(method_dir / "_tracklet_cache")
    train_cfg.scratch_dir = str(unit_scratch_dir)
    train_cfg.seed = int(_select(cfg, "protocol.seed", 0)) + int(unit_index)
    train_cfg.wandb_project = str(_method_attr(method, "wandb_project", getattr(train_cfg, "wandb_project", "longlive_per_reference")))

    rendered = unit_train_dir / f"{_safe_id(str(method.name))}_{_safe_id(unit['unit_id'])}.resolved.yaml"
    unit_train_dir.mkdir(parents=True, exist_ok=True)
    unit_scratch_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(train_cfg, rendered)
    return rendered


def _train_method(
    *,
    cfg: DictConfig,
    method: DictConfig,
    unit: dict,
    method_dir: Path,
    unit_index: int,
    env: dict[str, str],
    dry_run: bool,
) -> Path:
    rendered = _render_train_config(
        cfg=cfg,
        method=method,
        unit=unit,
        method_dir=method_dir,
        unit_index=unit_index,
        dry_run=dry_run,
    )
    ckpt = method_dir / unit["dataset"] / unit["unit_id"] / "train" / "lora_final.pt"
    if ckpt.exists() and bool(_select(cfg, "runtime.skip_completed_train", True)):
        print(f"[per_ref] train exists, skipping: {ckpt}", flush=True)
        return ckpt

    nproc = int(_select(cfg, "runtime.nproc_per_node", 8))
    master_port = int(_select(cfg, "runtime.master_port", 29500)) + int(unit_index)
    cmd = [
        "torchrun",
        "--nproc_per_node", str(nproc),
        "--master_port", str(master_port),
        "-m", str(method.train_module),
        "--config", str(rendered),
    ]
    if bool(_select(cfg, "runtime.disable_wandb", False)):
        cmd.append("--disable-wandb")
    _run(cmd, cwd=REPO_ROOT, env=env, dry_run=dry_run)
    if not dry_run and not ckpt.exists():
        raise FileNotFoundError(f"training finished but final ckpt is missing: {ckpt}")
    return ckpt


def _eval_ckpt(
    *,
    cfg: DictConfig,
    method: DictConfig,
    unit: dict,
    ckpt: Path,
    method_dir: Path,
    env: dict[str, str],
    dry_run: bool,
) -> Path:
    eval_dir = method_dir / unit["dataset"] / unit["unit_id"] / "motion_eval"
    prompts_manifest = eval_dir / "prompts_manifest.jsonl"
    if not dry_run:
        eval_dir.mkdir(parents=True, exist_ok=True)
    write_unit_prompt_manifest(prompts_manifest, unit)

    gpus = str(_select(cfg, "runtime.eval_gpus", "0,1,2,3,4,5,6,7"))
    python_bin = str(_select(cfg, "runtime.python_bin", sys.executable))
    eval_config = str(method.eval_config)
    dispatch_cmd = [
        python_bin,
        str(REPO_ROOT / "scripts" / "motion_eval" / "eval_dispatch.py"),
        "--config", eval_config,
        "--ckpt", str(ckpt),
        "--manifest", str(prompts_manifest),
        "--output_dir", str(eval_dir),
        "--gpu_ids", gpus,
        "--python_bin", python_bin,
        "--seed", str(int(_select(cfg, "protocol.seed", 0))),
    ]
    _run(dispatch_cmd, cwd=REPO_ROOT, env=env, dry_run=dry_run)

    scores_csv = eval_dir / "scores.csv"
    run_eval_cmd = [
        python_bin,
        str(REPO_ROOT / "scripts" / "motion_eval" / "run_eval.py"),
        "--prompts_manifest", str(prompts_manifest),
        "--gen_dir", str(eval_dir),
        "--ref_root", str(cfg.data_root),
        "--output", str(scores_csv),
        "--wandb_run_name", f"{cfg.experiment_name}_{method.name}_{unit['unit_id']}",
        "--ckpt_tag", str(ckpt),
    ]
    _run(run_eval_cmd, cwd=REPO_ROOT, env=env, dry_run=dry_run)
    return scores_csv


def _read_scores(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def aggregate(run_root: Path, results: list[dict]) -> dict:
    summary: dict[str, Any] = {
        "results": results,
        "by_method_dataset": {},
    }
    grouped: dict[tuple[str, str], list[dict]] = {}
    for result in results:
        for row in _read_scores(Path(result["scores_csv"])):
            if row.get("ok") != "True":
                continue
            grouped.setdefault((result["method"], result["dataset"]), []).append(row)

    for (method, dataset), rows in sorted(grouped.items()):
        out: dict[str, float | int] = {"n": len(rows)}
        for metric in METRIC_COLUMNS:
            vals = [float(r[metric]) for r in rows if r.get(metric) not in ("", None)]
            if vals:
                out[metric] = sum(vals) / len(vals)
        dyn_vals = [
            1.0 if str(r.get("is_dynamic", "")).lower() == "true" else 0.0
            for r in rows
            if r.get("is_dynamic") not in ("", None)
        ]
        if dyn_vals:
            out["dynamic_rate"] = sum(dyn_vals) / len(dyn_vals)
        summary["by_method_dataset"].setdefault(method, {})[dataset] = out

    with open(run_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(run_root / "summary_by_method_dataset.csv", "w", newline="") as f:
        fieldnames = ["method", "dataset", "n", *METRIC_COLUMNS, "dynamic_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, by_dataset in summary["by_method_dataset"].items():
            for dataset, metrics in by_dataset.items():
                writer.writerow({"method": method, "dataset": dataset, **metrics})
    return summary


def run_protocol(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    data_root = Path(str(cfg.data_root))
    output_root = Path(str(_select(cfg, "output_root", data_root / "per_reference_adaptation_runs")))
    run_root = output_root / str(cfg.experiment_name)
    run_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_root / "protocol_config.resolved.yaml")

    units = resolve_units(cfg)
    if args.limit_units is not None:
        units = units[: args.limit_units]
    write_jsonl(run_root / "units.jsonl", units)
    print(f"[per_ref] resolved {len(units)} units -> {run_root / 'units.jsonl'}", flush=True)

    method_filter = (
        {x.strip() for x in args.methods.split(",") if x.strip()}
        if args.methods
        else None
    )
    methods = [
        _resolve_method(m)
        for m in cfg.methods
        if method_filter is None or str(m.name) in method_filter
    ]
    if not methods:
        raise RuntimeError("no methods selected")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("LL_DATA", str(data_root))
    env.setdefault("HF_HOME", str(data_root / "hf_cache"))
    env.setdefault("TRANSFORMERS_CACHE", str(data_root / "hf_cache"))
    env.setdefault("TORCH_HOME", str(data_root / "hf_cache" / "torch_hub"))

    results = []
    t0 = time.time()
    for method in methods:
        method_name = str(method.name)
        method_dir = run_root / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        kind = str(_method_attr(method, "kind", "train"))
        print(f"[per_ref] === method {method_name} ({kind}) ===", flush=True)
        for unit_index, unit in enumerate(units):
            print(
                f"[per_ref] unit {unit_index + 1}/{len(units)} "
                f"{unit['unit_id']} ref={unit['reference_video']}",
                flush=True,
            )
            if kind == "baseline":
                ckpt = Path(str(method.ckpt))
                if not ckpt.is_absolute():
                    ckpt = data_root / ckpt
                if not args.dry_run and not ckpt.exists():
                    raise FileNotFoundError(f"baseline ckpt not found: {ckpt}")
            elif kind == "train":
                ckpt = _train_method(
                    cfg=cfg,
                    method=method,
                    unit=unit,
                    method_dir=method_dir,
                    unit_index=unit_index,
                    env=env,
                    dry_run=args.dry_run,
                )
            else:
                raise ValueError(f"unknown method kind for {method_name}: {kind!r}")
            scores_csv = _eval_ckpt(
                cfg=cfg,
                method=method,
                unit=unit,
                ckpt=ckpt,
                method_dir=method_dir,
                env=env,
                dry_run=args.dry_run,
            )
            results.append({
                "method": method_name,
                "dataset": unit["dataset"],
                "unit_id": unit["unit_id"],
                "ckpt": str(ckpt),
                "scores_csv": str(scores_csv),
            })
            aggregate(run_root, results)

    summary = aggregate(run_root, results)
    print(f"[per_ref] DONE in {(time.time() - t0) / 60.0:.1f} min", flush=True)
    print(f"[per_ref] summary: {run_root / 'summary.json'}", flush=True)
    print(json.dumps(summary["by_method_dataset"], indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--methods", default="", help="Comma-separated method names to run")
    p.add_argument("--limit-units", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run_protocol(parse_args())
