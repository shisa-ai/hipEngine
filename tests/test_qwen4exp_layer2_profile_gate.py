from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_layer2_profile_gate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen4exp_layer2_profile_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_json_value_thaws_nested_immutable_profile_manifest() -> None:
    module = _load_script()
    value = MappingProxyType({"rows": (1, MappingProxyType({"variant": "x"}))})

    assert module._json_value(value) == {"rows": [1, {"variant": "x"}]}


def test_state_summary_hashes_owned_state_and_metadata() -> None:
    module = _load_script()
    snapshot = SimpleNamespace(
        position=7,
        decode_state=SimpleNamespace(
            buffers={
                "gdn_matrix": np.asarray([1.0, 2.0], dtype=np.float32).view(np.uint8),
                "residual": np.asarray([0x3F80, 0x4000], dtype=np.uint16).view(np.uint8),
            }
        ),
        ple_hash_states={0: SimpleNamespace(tokens=(1, 2), next_position=7)},
    )
    runner = SimpleNamespace(
        snapshot=lambda: snapshot,
        attention_states=(
            SimpleNamespace(position_host=np.asarray([6]), context_host=np.asarray([7])),
        ),
        index_states=(SimpleNamespace(count=7, pooled_count=1),),
    )

    first = module._state_summary(runner)
    second = module._state_summary(runner)

    assert first == second
    assert first["finite"] is True
    assert first["position"] == 7
    assert first["buffer_bytes"] == {"gdn_matrix": 8, "residual": 4}
    assert first["attention_positions"] == [[6, 7]]
    assert first["index_counts"] == [[7, 1]]


def test_state_repeat_gate_requires_candidate_repeatability_and_layout() -> None:
    module = _load_script()
    metadata = {"position": 7, "attention_positions": [[6, 7]], "index_counts": [[7, 1]]}
    strict = [{
        "prompt_id": "p", "state_sha256": "s", "layout_sha256": "l",
        "finite": True, **metadata,
    }]
    candidate = [[
        {"prompt_id": "p", "state_sha256": "c", "layout_sha256": "l", "finite": True, **metadata},
        {"prompt_id": "p", "state_sha256": "c", "layout_sha256": "l", "finite": True, **metadata},
        {"prompt_id": "p", "state_sha256": "c", "layout_sha256": "l", "finite": True, **metadata},
    ]]

    passed = module._state_repeat_gate(strict, candidate)
    assert passed["passed"] is True
    assert passed["prompts"][0]["strict_candidate_state_exact"] is False

    candidate[0][2] = {**candidate[0][2], "state_sha256": "different"}
    failed = module._state_repeat_gate(strict, candidate)
    assert failed["passed"] is False
    assert failed["mismatches"][0]["repeat_exact"] is False


def test_qsa_dense_fixed256_candidate_is_t0_and_fail_closed() -> None:
    module = _load_script()
    candidate = module.CANDIDATES["qsa_dense_fixed256"]

    assert candidate.environment == {
        "HIPENGINE_QWEN4_EXP_QSA_DENSE_FIXED256": "1"
    }
    assert candidate.classification == "T0"
    assert candidate.base_profile == "strict"
    assert candidate.candidate_key[-1] == "bf16_context_batch_paged_c1_exact_spans"
    assert candidate.fallback_key[-1] == "bf16_context_batch_spans"


def test_q8_mmq_attn_gate_candidate_is_explicit_and_fail_closed() -> None:
    module = _load_script()
    candidate = module.CANDIDATES["q8_mmq_attn_gate"]

    assert candidate.environment == {"HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE": "1"}
    assert candidate.classification == "T2"
    assert candidate.base_profile == "production"
    assert candidate.candidate_key[-1] == (
        "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out"
    )
    assert candidate.fallback_key[-1] == "coltile8_rowbatch4_f32_f32_out"


def test_gfx1151_registry_contains_layer2_candidate_and_strict_fallback() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve

    register_gfx1151_kernels(replace=True)

    assert resolve(
        backend="hip_gfx1151",
        layer="moe_linear",
        quant="gguf_q5_k",
        variant="selected_wmma_prefill_compact_bf16_bf16_out",
    ) is not None
    assert resolve(
        backend="hip_gfx1151",
        layer="linear",
        quant="gguf_q5_k",
        variant="selected_gemv_bf16_bf16_out",
    ) is not None
    assert resolve(
        backend="hip_gfx1151",
        layer="linear",
        quant="gguf_q8_0",
        variant="mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out",
    ) is not None


def test_task_gate_requires_repeatability_and_flags_cross_route_divergence() -> None:
    module = _load_script()
    strict = {"a": {"ids": [1, 2], "ids_sha256": "s", "text": "strict"}}
    candidate = {
        "a": [
            {"ids": [1, 3], "ids_sha256": "c", "text": "candidate"},
            {"ids": [1, 3], "ids_sha256": "c", "text": "candidate"},
        ]
    }

    result = module._task_gate(strict, candidate, categories={"a": "general_en"})

    assert result["candidate_repeat_exact"] is True
    assert result["strict_exact_count"] == 0
    assert result["status"] == "requires_review"
    assert result["divergences"][0]["id"] == "a"
