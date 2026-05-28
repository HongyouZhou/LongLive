from types import SimpleNamespace
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from longlive.data.motion_refs import GeneralPromptDataset, SkateboardingLatentDataset
from longlive.methods.motiondirector import data as motiondirector_data
from longlive.specs.wan import get_wan_model_spec, get_wan_model_spec_from_config


def test_motiondirector_data_path_reexports_shared_data() -> None:
    assert motiondirector_data.SkateboardingLatentDataset is SkateboardingLatentDataset
    assert motiondirector_data.GeneralPromptDataset is GeneralPromptDataset


def test_wan21_cache_spec_preserves_legacy_dimensions() -> None:
    spec = get_wan_model_spec("Wan2.1-T2V-1.3B")
    cache = spec.cache

    assert cache.transformer_blocks == 30
    assert cache.frame_seq_length == 1560
    assert cache.kv_shape(batch_size=2, tokens=18720) == (2, 18720, 12, 128)
    assert cache.crossattn_shape(batch_size=2) == (2, 512, 12, 128)

    assert cache.kv_tokens_for_frames(num_output_frames=21, local_attn_size=-1) == 32760
    assert cache.kv_tokens_for_frames(num_output_frames=21, local_attn_size=12) == 18720
    assert cache.attention_tokens(local_attn_size=-1) == 32760
    assert cache.attention_tokens(local_attn_size=12) == 18720
    assert cache.wrapper_seq_len(local_attn_size=12) == 32760


def test_wan_spec_from_attr_config_defaults_to_wan21() -> None:
    cfg = SimpleNamespace(local_attn_size=12)
    assert get_wan_model_spec_from_config(cfg).name == "Wan2.1-T2V-1.3B"


def main() -> None:
    test_motiondirector_data_path_reexports_shared_data()
    test_wan21_cache_spec_preserves_legacy_dimensions()
    test_wan_spec_from_attr_config_defaults_to_wan21()


if __name__ == "__main__":
    main()
