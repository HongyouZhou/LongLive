"""Build a unified eval manifest from UCF Sports prompts + LOVEU-TGVE prompts.

Joins:
  - UCF: scripts/motion_eval/prompts/ucf_sports.yaml × $LL_DATA/ucf_sports/manifest.csv
    -> 10 categories × 6 inference prompts = 60 rows.
    Each row's ref_videos is the list of all category clips (motion fidelity
    will mean over them; see scripts/motion_eval/metrics/motion_fidelity.py).

  - LOVEU: $LL_DATA/loveu_tgve/prompts.csv (LOVEU-TGVE-2023_Dataset.csv verbatim)
    -> 76 videos × 4 edit prompt columns = 304 rows.
    Each row's ref_videos is a single source video (the one being edited).
    Optionally include the original Caption column as a 5th prompt per video
    (--include_loveu_caption); off by default since the Caption describes
    the source, not a target generation.

Output: JSONL, one row per prompt. Row schema::

    {
      "prompt_id": "ab12cd34",
      "dataset": "ucf" | "loveu",
      "key": {"category": "..."} or {"video_id": "...", "prompt_type": "..."},
      "prompt": "<text>",
      "ref_videos": ["<path relative to data_root>", ...],
      "paper_verbatim": true | false | "partial"   # ucf only; loveu always true
    }

prompt_id = first 8 hex chars of sha256(f"{dataset}|{key_str}|{prompt}").
Asserts no collisions across all emitted rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
UCF_PROMPTS_YAML = Path(__file__).parent / "prompts" / "ucf_sports.yaml"


def _prompt_id(dataset: str, key_str: str, prompt: str) -> str:
    h = hashlib.sha256(f"{dataset}|{key_str}|{prompt}".encode("utf-8")).hexdigest()
    return h[:8]


def _load_ucf_video_manifest(manifest_csv: Path) -> dict[str, list[str]]:
    """Return {category: [relative_video_path, ...]} from prepare_motion_eval.py output."""
    by_cat: dict[str, list[str]] = {}
    with open(manifest_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_cat.setdefault(row["category"], []).append(row["path"])
    for cat in by_cat:
        by_cat[cat].sort()
    return by_cat


def _build_ucf_rows(data_root: Path) -> list[dict]:
    ucf_manifest = data_root / "ucf_sports" / "manifest.csv"
    if not ucf_manifest.exists():
        raise FileNotFoundError(
            f"{ucf_manifest} not found. Run scripts/prepare_motion_eval.py --datasets ucf first."
        )
    by_cat = _load_ucf_video_manifest(ucf_manifest)

    with open(UCF_PROMPTS_YAML) as f:
        spec = yaml.safe_load(f)

    rows = []
    for category, entry in spec["categories"].items():
        if category not in by_cat:
            print(
                f"[manifest] WARNING: ucf category {category!r} has no clips under "
                f"{ucf_manifest.parent / 'videos'} — skipping its prompts",
                file=sys.stderr,
            )
            continue
        ref_videos = by_cat[category]
        for prompt_idx, prompt in enumerate(entry["inference_prompts"]):
            key = {"category": category, "prompt_idx": prompt_idx}
            key_str = json.dumps(key, sort_keys=True, separators=(",", ":"))
            rows.append({
                "prompt_id": _prompt_id("ucf", key_str, prompt),
                "dataset": "ucf",
                "key": key,
                "prompt": prompt,
                "ref_videos": ref_videos,
                "paper_verbatim": entry.get("paper_verbatim", False),
            })
    return rows


# LOVEU's CSV has 5 prompt columns. The 4 "edit" columns are the test
# prompts; "Caption" describes the source video and is normally not used
# as a generation target.
LOVEU_EDIT_COLUMNS = ["Style Change", "Object Change", "Background Change", "Multiple Changes"]
LOVEU_VIDEO_COLUMN = "Video name"
LOVEU_CAPTION_COLUMN = "Caption"


def _build_loveu_rows(data_root: Path, include_caption: bool) -> list[dict]:
    loveu_root = data_root / "loveu_tgve"
    prompts_csv = loveu_root / "prompts.csv"
    videos_dir = loveu_root / "videos"
    if not prompts_csv.exists():
        raise FileNotFoundError(
            f"{prompts_csv} not found. Run scripts/prepare_motion_eval.py --datasets loveu first."
        )

    rows = []
    with open(prompts_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            video_id = r[LOVEU_VIDEO_COLUMN].strip()
            # Match a video file by stem (LOVEU mp4s typically named <video_id>.mp4).
            candidates = list(videos_dir.glob(f"{video_id}.*"))
            if not candidates:
                print(
                    f"[manifest] WARNING: loveu video {video_id!r} has no file under "
                    f"{videos_dir} — skipping",
                    file=sys.stderr,
                )
                continue
            ref_rel = str(candidates[0].relative_to(data_root))

            prompt_columns = list(LOVEU_EDIT_COLUMNS)
            if include_caption:
                prompt_columns = [LOVEU_CAPTION_COLUMN] + prompt_columns

            for col in prompt_columns:
                prompt = r.get(col, "").strip()
                if not prompt:
                    continue
                key = {"video_id": video_id, "prompt_type": col}
                key_str = json.dumps(key, sort_keys=True, separators=(",", ":"))
                rows.append({
                    "prompt_id": _prompt_id("loveu", key_str, prompt),
                    "dataset": "loveu",
                    "key": key,
                    "prompt": prompt,
                    "ref_videos": [ref_rel],
                    "paper_verbatim": True,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=os.environ.get("LL_DATA"))
    ap.add_argument("--datasets", type=str, default="ucf,loveu")
    ap.add_argument("--output", type=str, required=True,
                    help="JSONL output path (one prompt per line)")
    ap.add_argument("--include_loveu_caption", action="store_true",
                    help="Include the LOVEU 'Caption' column as a 5th prompt per video "
                         "(off by default; Caption describes the source clip)")
    args = ap.parse_args()

    if not args.data_root:
        raise SystemExit("--data_root unset and $LL_DATA not in env")

    data_root = Path(args.data_root)
    requested = {s.strip().lower() for s in args.datasets.split(",") if s.strip()}
    valid = {"ucf", "loveu"}
    if not requested.issubset(valid):
        raise SystemExit(f"Unknown datasets {requested - valid}, valid={valid}")

    all_rows: list[dict] = []
    if "ucf" in requested:
        ucf_rows = _build_ucf_rows(data_root)
        print(f"[manifest] ucf: {len(ucf_rows)} prompts")
        all_rows.extend(ucf_rows)
    if "loveu" in requested:
        loveu_rows = _build_loveu_rows(data_root, args.include_loveu_caption)
        print(f"[manifest] loveu: {len(loveu_rows)} prompts")
        all_rows.extend(loveu_rows)

    # Collision check: 8 hex chars * 4 bits = 32 bits. For ~600 rows, ~4e-5
    # collision probability. Defensive assert.
    seen = {}
    for r in all_rows:
        if r["prompt_id"] in seen:
            other = seen[r["prompt_id"]]
            raise RuntimeError(
                f"prompt_id collision {r['prompt_id']}: "
                f"({r['dataset']}, {r['key']}, {r['prompt']!r}) vs "
                f"({other['dataset']}, {other['key']}, {other['prompt']!r})"
            )
        seen[r["prompt_id"]] = r

    # Deterministic ordering by prompt_id.
    all_rows.sort(key=lambda r: r["prompt_id"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[manifest] wrote {out_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
