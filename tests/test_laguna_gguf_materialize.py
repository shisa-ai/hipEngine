from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFReader
from hipengine.loading.laguna_gguf import build_laguna_gguf_tensor_map
from hipengine.loading.laguna_gguf_materialize import (
    DEFAULT_LAGUNA_SAFETY_RESERVE_BYTES,
    DEFAULT_LAGUNA_SCRATCH_BYTES,
    LAYOUT_DENSE_F16,
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    LagunaMemoryAdmissionError,
    plan_laguna_gguf_materialization,
    plan_laguna_memory_admission,
)
from tests._laguna_synthetic import laguna_tensors, make_laguna_info

MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")


def _plan():
    model_map = build_laguna_gguf_tensor_map(
        make_laguna_info(tensors=laguna_tensors())
    )
    return plan_laguna_gguf_materialization(model_map)


def test_laguna_materialization_plan_covers_all_tensors_without_f16_contraction() -> None:
    plan = _plan()

    assert len(plan.tensor_names) == 814
    assert set(plan.tensor_names) == {tensor.name for tensor in laguna_tensors()}
    assert plan.source_nbytes == sum(tensor.nbytes for tensor in laguna_tensors())
    assert plan.root_specs["token_embedding"].layout == LAYOUT_RAW_GGUF
    assert plan.root_specs["lm_head"].layout == LAYOUT_GGUF_Q6_K_T16
    assert plan.root_specs["output_norm"].layout == LAYOUT_DENSE_F32

    dense = plan.layer_specs[0]
    assert dense["attn_q"].layout == LAYOUT_DENSE_F16
    assert dense["attn_q"].resident_dtype == "fp16"
    assert dense["attn_q"].resident_nbytes == dense["attn_q"].source.nbytes
    assert dense["attn_gate"].layout == LAYOUT_DENSE_F16
    assert dense["ffn_gate"].layout == LAYOUT_Q4_K_PACK8
    assert dense["ffn_down"].layout == LAYOUT_RAW_GGUF

    sparse = plan.layer_specs[1]
    assert sparse["ffn_gate_inp"].layout == LAYOUT_DENSE_F32
    assert sparse["exp_probs_b"].layout == LAYOUT_DENSE_F32
    assert sparse["ffn_gate_exps"].layout == LAYOUT_GGUF_Q4_K_T16
    assert sparse["ffn_down_exps"].layout == LAYOUT_GGUF_Q6_K_T16
    assert sparse["ffn_gate_shexp"].layout == LAYOUT_Q4_K_PACK8
    assert sparse["ffn_down_shexp"].layout == LAYOUT_RAW_GGUF
    assert plan.precision_contractions == ()


def test_laguna_materialization_plan_preserves_rank3_expert_contract() -> None:
    spec = _plan().layer_specs[1]["ffn_gate_exps"]

    assert spec.source.shape == (256, 1_024, 3_072)
    assert spec.source.byte_shape == (256, 1_024, 1_728)
    assert spec.layout == LAYOUT_GGUF_Q4_K_T16
    assert spec.allocation_names == ("tiles",)
    assert spec.allocation_nbytes["tiles"] == 256 * (1_024 // 16) * (3_072 // 256) * 2_368
    assert spec.loader_transient_nbytes == spec.source.nbytes + spec.resident_nbytes


def test_laguna_4k_kv_plan_uses_swa_ring_not_full_context() -> None:
    admission = plan_laguna_memory_admission(
        _plan(),
        context_length=4_096,
        available_bytes=120 * 2**30,
    )

    assert admission.kv.global_layer_count == 12
    assert admission.kv.sliding_layer_count == 36
    assert admission.kv.global_tokens_per_layer == 4_096
    assert admission.kv.sliding_tokens_per_layer == 512
    assert admission.kv.bytes_per_layer_token == 4_096
    assert admission.kv.resident_nbytes == 276_824_064
    assert admission.scratch_nbytes == DEFAULT_LAGUNA_SCRATCH_BYTES
    assert admission.safety_reserve_nbytes == DEFAULT_LAGUNA_SAFETY_RESERVE_BYTES
    assert admission.passed
    assert admission.headroom_bytes > 0


def test_laguna_admission_rejects_naive_all_layers_full_kv() -> None:
    with pytest.raises(ValueError, match="all-layers-full-KV"):
        plan_laguna_memory_admission(
            _plan(),
            context_length=4_096,
            available_bytes=120 * 2**30,
            honor_sliding_window=False,
        )


def test_laguna_admission_rejects_peak_over_budget_before_allocation() -> None:
    accepted = plan_laguna_memory_admission(
        _plan(),
        context_length=4_096,
        available_bytes=120 * 2**30,
    )

    with pytest.raises(LagunaMemoryAdmissionError, match="exceeds available"):
        plan_laguna_memory_admission(
            _plan(),
            context_length=4_096,
            available_bytes=accepted.peak_required_nbytes - 1,
        )


def test_completed_laguna_artifact_dry_plan_covers_814_tensors() -> None:
    if not MODEL.exists():
        pytest.skip(f"local Laguna GGUF not found: {MODEL}")
    reader = GGUFReader(MODEL)
    model_map = build_laguna_gguf_tensor_map(reader.info)

    plan = plan_laguna_gguf_materialization(model_map)
    admission = plan_laguna_memory_admission(
        plan,
        context_length=4_096,
        available_bytes=120 * 2**30,
    )

    assert len(plan.tensor_names) == 814
    assert set(plan.tensor_names) == {tensor.name for tensor in reader.info.tensors}
    assert admission.passed
