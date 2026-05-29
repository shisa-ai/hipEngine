from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hipengine.llm import LLM
from hipengine.models import StepFunUnsupportedCapabilityError, resolve_model
from hipengine.quant import resolve_quant


DEFAULT_STEPFUN_GGUF_DIR = Path("/data/models/gguf")


def _stepfun_gguf_paths() -> tuple[Path, ...]:
    root = Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", DEFAULT_STEPFUN_GGUF_DIR))
    paths = tuple(sorted(root.glob("Step-3.7-flash-Q3_K_L-*.gguf")))
    if len(paths) != 3:
        pytest.skip(
            "StepFun GGUF Q3_K_L shards not found; set HIPENGINE_STEPFUN_GGUF_DIR "
            "to a directory containing Step-3.7-flash-Q3_K_L-00001..00003.gguf"
        )
    return paths


def test_stepfun_model_plugin_resolves_architecture_aliases() -> None:
    for alias in (
        "step35",
        "step3p5",
        "step3p7",
        "Step3p5ForCausalLM",
        "Step3p7ForConditionalGeneration",
        "stepfun_step3_7_gguf",
    ):
        plugin = resolve_model(alias)
        assert plugin.name == "stepfun_step3_7_gguf"
        assert plugin.default_quant == "gguf_q3_k_l"
        assert plugin.default_backend == "hip_gfx1151"

    plugin = resolve_model("step35")
    assert plugin.capability_status("text_decode") == "supported"
    assert plugin.capability_status("vision") == "deferred"
    with pytest.raises(StepFunUnsupportedCapabilityError, match="deferred"):
        plugin.require_capability("vision")
    with pytest.raises(StepFunUnsupportedCapabilityError, match="not declared"):
        plugin.require_capability("audio")
    assert "step_full_attention_qkv_proj" in plugin.decode_layer_sequence(
        attention_kind="full_attention", mlp_kind="dense_mlp"
    )
    assert "sliding_attention_decode" in plugin.decode_layer_sequence(
        attention_kind="sliding_attention", mlp_kind="moe"
    )


def test_stepfun_gguf_q3_k_l_quant_plugin_is_registered() -> None:
    plugin = resolve_quant("gguf_q3_k_l")

    assert plugin.weight_storage == "gguf_mixed_q3_k_l"
    assert plugin.activation_preprocess == "none"
    assert plugin.compute_dtype == "fp32_accum"
    assert plugin.calibration_artifact == "gguf"
    assert plugin.kernel_family == "gguf_q3_k_l_mixed_gemv"


def test_llm_resolves_stepfun_split_gguf_metadata_without_torch(tmp_path: Path) -> None:
    had_torch = "torch" in sys.modules
    for source in _stepfun_gguf_paths():
        (tmp_path / source.name).symlink_to(source)

    llm = LLM(str(tmp_path), backend="hip_gfx1151", quant="gguf_q3_k_l")
    weight_index, model_plugin = llm._load_model_metadata()

    assert model_plugin.name == "stepfun_step3_7_gguf"
    assert weight_index.architecture == "step35"
    assert weight_index.tensor_count == 754
    assert llm.model == str(tmp_path)
    if not had_torch:
        assert "torch" not in sys.modules
