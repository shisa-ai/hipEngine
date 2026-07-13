from __future__ import annotations

import inspect

import numpy as np
import pytest

from scripts.qwen35_gguf_moe_replay import (
    ReplayRecorder,
    _correlate_with_reference,
    _install_replay_helper,
    _restore_replay_helper,
    _selected_down_tile_shape,
    aggregate_records,
    summarize_counts,
)
import hipengine.runtime.qwen35_gguf_runner as qgr


def test_summarize_counts_records_hot_thresholds() -> None:
    counts = np.array([0, 1, 3, 16, 31, 32, 63, 64, 128], dtype=np.int64)
    summary = summarize_counts(counts)
    assert summary["experts"] == 9
    assert summary["compact_rows"] == int(counts.sum())
    assert summary["nonzero_experts"] == 8
    assert summary["experts_ge_16"] == 6
    assert summary["experts_ge_32"] == 4
    assert summary["experts_ge_64"] == 2
    assert summary["experts_ge_128"] == 1
    assert summary["rows_in_experts_ge_64"] == 64 + 128
    assert summary["max_rows_per_expert"] == 128


def test_aggregate_records_splits_q4_and_down_quant_totals() -> None:
    records = [
        {"down_quant": "gguf_q5_k", "timings_ms": {"gate_up_avg": 1.5, "down_avg": 0.7}},
        {"down_quant": "gguf_q5_k", "timings_ms": {"gate_up_avg": 1.6, "down_avg": 0.8}},
        {"down_quant": "gguf_q6_k", "timings_ms": {"gate_up_avg": 1.4, "down_avg": 0.2}},
    ]
    agg = aggregate_records(records)
    assert agg["selected_moe_total_ms"] == 6.2
    assert agg["by_component"]["q4_dual_gate_up"]["layers"] == 3
    assert agg["by_component"]["q4_dual_gate_up"]["total_ms"] == 4.5
    assert agg["by_component"]["gguf_q5_k_down"]["total_ms"] == 1.5
    assert agg["by_component"]["gguf_q6_k_down"]["total_ms"] == 0.2


def test_correlate_with_reference_computes_component_and_total_delta() -> None:
    agg = {
        "by_component": {
            "q4_dual_gate_up": {"total_ms": 40.0},
            "gguf_q5_k_down": {"total_ms": 20.0},
            "gguf_q6_k_down": {"total_ms": 2.0},
        }
    }
    ref = {
        "moe_q4_k_selected_dual_wmma_prefill": 50.0,
        "moe_q5_k_selected_wmma_prefill": 25.0,
        "moe_q6_k_selected_wmma_prefill": 2.0,
    }
    corr = _correlate_with_reference(agg, ref)
    assert corr["selected_moe_replay_ms"] == 62.0
    assert corr["selected_moe_reference_rocprof_ms"] == 77.0
    assert corr["components"]["q4_dual_gate_up"]["delta_pct"] == pytest.approx(-20.0)


def test_replay_helper_tracks_runtime_compact_wmma_keyword_interface() -> None:
    recorder = ReplayRecorder(warmup_iters=0, replay_iters=1, sample_groups=1)
    original = _install_replay_helper(recorder)
    try:
        signature = inspect.signature(qgr._try_run_post_attention_moe_rows_compact_wmma)
        assert "gpu_stage_recorder" in signature.parameters
        assert "stage_prefix" in signature.parameters
    finally:
        _restore_replay_helper(original)


def test_selected_down_tile_shape_covers_resident_t16_layouts() -> None:
    assert _selected_down_tile_shape("gguf_q5_k_t16_v1") == (16, 16)
    assert _selected_down_tile_shape("gguf_q6_k_t16_v1") == (16, 16)
    assert _selected_down_tile_shape("gguf_q5_k") == (16, 16)
