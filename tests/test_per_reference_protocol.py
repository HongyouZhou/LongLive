import json

from omegaconf import OmegaConf

from longlive.data.motion_refs import ReferenceVideoDataset
from scripts.per_reference.run_protocol import (
    _resolve_method,
    resolve_units,
    write_unit_prompt_manifest,
)


def _write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_resolve_units_uses_one_reference_per_unit(tmp_path):
    data_root = tmp_path / "wm"
    _write(
        data_root / "ucf_sports" / "manifest.csv",
        "category,path\nSkateboarding,videos/Skateboarding/ref0.mp4\n",
    )
    _write(data_root / "ucf_sports" / "videos" / "Skateboarding" / "ref0.mp4")
    _write(
        data_root / "loveu_tgve" / "prompts.csv",
        "Video name,Our GT caption,Style Change Caption,Object Change Caption,"
        "Background Change Caption,Multiple Changes Caption,Source\n"
        "loveu_ref,A person walks,A marble person walks,A robot walks,"
        "A person walks on the moon,A robot walks on the moon,src\n",
    )
    _write(data_root / "loveu_tgve" / "videos" / "loveu_ref.mp4")

    cfg = OmegaConf.create({
        "data_root": str(data_root),
        "protocol": {
            "seed": 0,
            "datasets": {
                "ucf": {
                    "enabled": True,
                    "categories": ["Skateboarding"],
                    "max_categories": None,
                    "refs_per_category": 1,
                    "prompts_per_unit": 2,
                },
                "loveu": {
                    "enabled": True,
                    "max_units": 1,
                    "unit_ids": None,
                    "prompts_per_unit": 2,
                    "prompt_types": [
                        "Style Change Caption",
                        "Object Change Caption",
                        "Background Change Caption",
                        "Multiple Changes Caption",
                    ],
                },
            },
        },
    })

    units = resolve_units(cfg)
    assert [u["dataset"] for u in units] == ["ucf", "loveu"]
    assert units[0]["reference_video"] == "ucf_sports/videos/Skateboarding/ref0.mp4"
    assert units[1]["reference_video"] == "loveu_tgve/videos/loveu_ref.mp4"
    assert len(units[0]["eval_prompts"]) == 2
    assert len(units[1]["eval_prompts"]) == 2

    manifest_path = tmp_path / "prompts.jsonl"
    rows = write_unit_prompt_manifest(manifest_path, units[0])
    assert len(rows) == 2
    assert rows[0]["ref_videos"] == [units[0]["reference_video"]]
    parsed = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert parsed[0]["unit_id"] == units[0]["unit_id"]


def test_reference_video_dataset_resolves_data_relative_path(tmp_path):
    data_root = tmp_path / "wm"
    _write(data_root / "refs" / "motion.mp4")
    dataset = ReferenceVideoDataset(
        data_root=data_root,
        vae=object(),
        reference_video_path="refs/motion.mp4",
        train_caption="a person moves",
        device="cpu",
        unit_id="unit0",
    )
    assert dataset.train_clip_path == data_root / "refs" / "motion.mp4"
    assert dataset.train_caption == "a person moves"


def test_explicit_units_are_the_source_of_distillation_count(tmp_path):
    data_root = tmp_path / "wm"
    _write(data_root / "refs" / "a.mp4")
    _write(data_root / "refs" / "b.mp4")

    cfg = OmegaConf.create({
        "data_root": str(data_root),
        "protocol": {
            "seed": 0,
            "units": [
                {
                    "unit_id": "ref_a",
                    "dataset": "custom",
                    "reference_video": "refs/a.mp4",
                    "train_caption": "a person jumps",
                    "eval_prompts": ["a robot jumps"],
                },
                {
                    "unit_id": "ref_b",
                    "dataset": "custom",
                    "reference_video": "refs/b.mp4",
                    "train_caption": "a person runs",
                    "eval_prompts": [
                        {"prompt_type": "style", "prompt": "a marble person runs"},
                    ],
                },
            ],
            "datasets": {
                "ucf": {"enabled": False},
                "loveu": {"enabled": False},
            },
        },
    })

    units = resolve_units(cfg)
    assert [u["unit_id"] for u in units] == ["ref_a", "ref_b"]
    assert [u["reference_video"] for u in units] == ["refs/a.mp4", "refs/b.mp4"]
    assert units[0]["eval_prompts"][0]["prompt"] == "a robot jumps"
    assert units[1]["eval_prompts"][0]["prompt_type"] == "style"


def test_method_entry_resolves_from_registry():
    method = _resolve_method(OmegaConf.create({
        "name": "em_ram",
        "overrides": {"outer_epochs": 2},
    }))
    assert method.kind == "train"
    assert method.train_module == "longlive.methods.em_ram.train"
    assert method.train_template.endswith("per_reference_em_ram.yaml")
    assert method.overrides.outer_epochs == 2
