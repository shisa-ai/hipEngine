from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from scripts.laguna_long_context_profile import (
    ATTACK_LENGTHS,
    DECODE_OUTPUT_TOKENS,
    EAGER_DECODE_CONTEXT_LIMIT,
    FINAL_SWEEP_LENGTHS,
    LAP0_LENGTHS,
    LC0_TRACE_LENGTHS,
    PROFILE_LENGTH_SETS,
    SHORT_FOCUS_LENGTHS,
    STANDARD_DECODE_LENGTHS,
    WPF_SHORT_LENGTHS,
    _parse_args,
    _parse_chunk_size,
    _parse_decode_output_tokens,
    _parse_lengths,
    _summarize_samples,
    _timing_order,
)
from scripts.laguna_long_context_trace_summary import (
    _aggregate_segments,
    _kernel_family,
    _segment_requests,
    _summarize_segment,
    attach_summary,
)


def test_lpf5_length_parser_and_order_are_strict_and_balanced() -> None:
    assert SHORT_FOCUS_LENGTHS == (512, 4096)
    assert SHORT_FOCUS_LENGTHS in PROFILE_LENGTH_SETS
    assert WPF_SHORT_LENGTHS == (512, 1024)
    assert WPF_SHORT_LENGTHS in PROFILE_LENGTH_SETS
    assert LAP0_LENGTHS == (128, 512, 1024, 4096)
    assert LAP0_LENGTHS in PROFILE_LENGTH_SETS
    assert ATTACK_LENGTHS == (4096, 16384, 65536, 131072)
    assert ATTACK_LENGTHS in PROFILE_LENGTH_SETS
    assert LC0_TRACE_LENGTHS == (16384, 65536)
    assert LC0_TRACE_LENGTHS in PROFILE_LENGTH_SETS
    assert FINAL_SWEEP_LENGTHS == (512, 1024, 4096, 32768, 65536, 131072)
    assert FINAL_SWEEP_LENGTHS in PROFILE_LENGTH_SETS
    assert STANDARD_DECODE_LENGTHS == (512,)
    assert STANDARD_DECODE_LENGTHS in PROFILE_LENGTH_SETS
    assert DECODE_OUTPUT_TOKENS == (1, 128)
    assert EAGER_DECODE_CONTEXT_LIMIT == 4096
    assert _parse_decode_output_tokens("1") == 1
    assert _parse_decode_output_tokens("128") == 128
    assert _parse_lengths("512,1024,4096") == (512, 1024, 4096)
    assert [
        _parse_chunk_size(value)
        for value in ("128", "256", "512", "1024", "2048")
    ] == [
        128,
        256,
        512,
        1024,
        2048,
    ]
    assert _timing_order((512, 1024, 4096), 0) == (512, 1024, 4096)
    assert _timing_order((512, 1024, 4096), 1) == (4096, 1024, 512)
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _parse_lengths("512,0")
    with pytest.raises(argparse.ArgumentTypeError, match="distinct"):
        _parse_lengths("512,512")
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="128, 256, 512, 1024, or 2048",
    ):
        _parse_chunk_size("64")
    with pytest.raises(argparse.ArgumentTypeError, match="1 or 128"):
        _parse_decode_output_tokens("32")


def test_lpf5_cli_supports_direct_gguf_profile_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Path("/models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "laguna_long_context_profile.py",
            str(model),
            "--direct-gguf",
            "--safety-reserve-gib",
            "0",
            "--quant-label",
            "UD-Q2_K_XL",
            "--package-matrix-rows",
            "--compare-raw-k-prefill-rowbatch",
            "--raw-k-prefill-rowbatch",
            "32",
            "--raw-k-prefill-rowbatch-control",
            "8",
        ],
    )

    args = _parse_args()

    assert args.model == model
    assert args.direct_gguf is True
    assert args.safety_reserve_gib == 0.0
    assert args.quant_label == "UD-Q2_K_XL"
    assert args.package_matrix_rows is True
    assert args.compare_raw_k_prefill_rowbatch is True
    assert args.raw_k_prefill_rowbatch == 32
    assert args.raw_k_prefill_rowbatch_control == 8


def test_lpf5_cli_supports_grouped_exact_iq_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Path("/models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "laguna_long_context_profile.py",
            str(model),
            "--direct-gguf",
            "--compare-grouped-exact-iq",
        ],
    )

    args = _parse_args()

    assert args.compare_grouped_exact_iq is True


def test_lpf5_cli_supports_pair16_grouped_gate_up_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Path("/models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "laguna_long_context_profile.py",
            str(model),
            "--direct-gguf",
            "--compare-pair16-grouped-gate-up",
        ],
    )

    args = _parse_args()

    assert args.compare_pair16_grouped_gate_up is True


def test_lpf5_timing_summary_preserves_rates_and_repeat_ids() -> None:
    summary = _summarize_samples(
        [
            {"length": 512, "prefill_seconds": 10.0, "next_token_id": 7},
            {"length": 512, "prefill_seconds": 12.0, "next_token_id": 7},
        ]
    )
    assert summary["median_seconds"] == 11.0
    assert summary["median_tok_s"] == pytest.approx(512 / 11)
    assert summary["repeat_deterministic"] is True

    with pytest.raises(ValueError, match="one length"):
        _summarize_samples(
            [
                {"length": 512, "prefill_seconds": 1.0, "next_token_id": 1},
                {"length": 1024, "prefill_seconds": 1.0, "next_token_id": 1},
            ]
        )


def test_lpf5_timing_summary_preserves_fixed_horizon_decode() -> None:
    summary = _summarize_samples(
        [
            {
                "length": 512,
                "prefill_seconds": 1.0,
                "next_token_id": 7,
                "output_tokens": 128,
                "decode_seconds": 8.0,
                "final_token_id": 9,
                "generated_ids_sha256": "abc",
            },
            {
                "length": 512,
                "prefill_seconds": 1.1,
                "next_token_id": 7,
                "output_tokens": 128,
                "decode_seconds": 10.0,
                "final_token_id": 9,
                "generated_ids_sha256": "abc",
            },
        ]
    )

    assert summary["output_tokens"] == 128
    assert summary["decode_forward_calls"] == 127
    assert summary["decode_median_seconds"] == 9.0
    assert summary["decode_median_tok_s"] == pytest.approx(127 / 9)
    assert summary["final_token_ids"] == [9, 9]
    assert summary["repeat_generated_ids_deterministic"] is True


def _row(
    dispatch: int,
    name: str,
    start: int,
    duration: int,
    *,
    grid_y: int = 1,
) -> dict[str, str]:
    return {
        "Dispatch_Id": str(dispatch),
        "Kernel_Name": name,
        "Start_Timestamp": str(start),
        "End_Timestamp": str(start + duration),
        "Grid_Size_Y": str(grid_y),
        "Workgroup_Size_X": "128",
        "Grid_Size_X": "9216",
        "VGPR_Count": "16",
        "SGPR_Count": "128",
        "LDS_Block_Size": "1024",
        "Scratch_Size": "0",
    }


def _request_rows(start: int, chunks: int) -> list[dict[str, str]]:
    rows = []
    timestamp = start
    dispatch = start
    for _ in range(chunks):
        rows.append(
            _row(
                dispatch,
                "gguf_q4_k_embedding_bf16_out_kernel",
                timestamp,
                10,
                grid_y=128,
            )
        )
        rows.append(
            _row(
                dispatch + 1,
                "laguna_global_attention_prefill_bf16_kernel",
                timestamp + 10,
                30,
                grid_y=128,
            )
        )
        rows.append(
            _row(
                dispatch + 2,
                "laguna_swa_attention_prefill_bf16_kernel",
                timestamp + 40,
                20,
                grid_y=128,
            )
        )
        timestamp += 60
        dispatch += 3
    rows.append(_row(dispatch, "argmax_stage2_kernel", timestamp, 10))
    return rows


def test_lpf5_trace_segments_requests_and_attributes_all_families() -> None:
    rows = [
        _row(0, "__amd_rocclr_copyBuffer", 0, 5),
        *_request_rows(100, 1),
        *_request_rows(1000, 4),
    ]
    segments = _segment_requests(rows)
    assert [(item["length"], item["chunks"]) for item in segments] == [
        (128, 1),
        (512, 4),
    ]
    summary = _summarize_segment(segments[1])
    assert summary["dispatches"] == 13
    assert summary["attention_duration_ns"] == 200
    assert summary["families"]["embedding"]["calls"] == 4
    assert summary["families"]["global_attention"]["calls"] == 4
    assert summary["families"]["swa_attention"]["calls"] == 4
    assert summary["families"]["lm_head_argmax"]["calls"] == 1
    assert sum(item["duration_ns"] for item in summary["families"].values()) == 250
    assert summary["attention_share_of_kernel_sum"] == pytest.approx(200 / 250)
    aggregate = _aggregate_segments([summary])
    assert aggregate["512"]["median_attention_duration_ns"] == 200
    assert aggregate["512"]["families"]["embedding"]["calls_per_pass"] == [4]


def test_lpf5_trace_segments_non_q4_gguf_embedding_requests() -> None:
    rows = _request_rows(100, 1)
    rows[0]["Kernel_Name"] = "gguf_q5_k_embedding_bf16_out_kernel"

    segments = _segment_requests(rows)

    assert [(item["length"], item["chunks"]) for item in segments] == [(128, 1)]
    assert _summarize_segment(segments[0])["families"]["embedding"]["calls"] == 1


def test_lpf5_trace_attributes_structural_blas_attention_composite() -> None:
    rows = [
        _row(0, "gguf_q4_k_embedding_bf16_out_kernel", 0, 10, grid_y=128),
        _row(
            1,
            "void laguna_dense_initial_cache_bf16_to_f32_kernel<true>",
            10,
            2,
        ),
    ]
    timestamp = 12
    for dispatch in range(2, 10):
        rows.append(_row(dispatch, "Cijk_qk", timestamp, 3))
        timestamp += 3
    rows.append(
        _row(
            10,
            "laguna_dense_initial_causal_softmax_f32_kernel",
            timestamp,
            5,
        )
    )
    timestamp += 5
    for dispatch in range(11, 19):
        rows.append(_row(dispatch, "Cijk_pv", timestamp, 4))
        timestamp += 4
    rows.append(_row(19, "argmax_stage2_kernel", timestamp, 1))

    summary = _summarize_segment(
        {"length": 128, "chunks": 1, "rows": rows}
    )
    assert summary["families"]["global_attention"]["calls"] == 18
    assert summary["families"]["global_attention"]["duration_ns"] == 63
    assert summary["families"]["source_f16_projection"]["calls"] == 0
    assert summary["attention_duration_ns"] == 63


def test_lpf5_trace_attributes_packed_query_blas_attention_composite() -> None:
    rows = [
        _row(0, "gguf_q4_k_embedding_bf16_out_kernel", 0, 10, grid_y=128),
        _row(
            1,
            "void laguna_dense_initial_cache_bf16_to_f32_kernel<false>",
            10,
            2,
        ),
        _row(
            2,
            "void laguna_dense_initial_query_head_transpose_f32_kernel<true>",
            12,
            3,
        ),
        _row(3, "Cijk_qk", 15, 4),
        _row(
            4,
            "laguna_dense_initial_causal_softmax_f32_kernel",
            19,
            5,
        ),
        _row(5, "Cijk_pv", 24, 6),
        _row(
            6,
            "void laguna_dense_initial_query_head_transpose_f32_kernel<false>",
            30,
            7,
        ),
        _row(7, "argmax_stage2_kernel", 37, 1),
    ]

    summary = _summarize_segment(
        {"length": 128, "chunks": 1, "rows": rows}
    )
    assert summary["families"]["swa_attention"]["calls"] == 6
    assert summary["families"]["swa_attention"]["duration_ns"] == 27
    assert summary["families"]["source_f16_projection"]["calls"] == 0
    assert summary["attention_duration_ns"] == 27


def test_lpf5_trace_attributes_packed_query_wave_softmax_composite() -> None:
    rows = [
        _row(0, "gguf_q4_k_embedding_bf16_out_kernel", 0, 10, grid_y=128),
        _row(
            1,
            "void laguna_dense_initial_cache_bf16_to_f32_kernel<false>",
            10,
            2,
        ),
        _row(
            2,
            "void laguna_dense_initial_query_head_transpose_f32_kernel<true>",
            12,
            3,
        ),
        _row(3, "Cijk_qk", 15, 4),
        _row(
            4,
            "laguna_dense_initial_causal_softmax_wave_rows_f32_kernel",
            19,
            5,
        ),
        _row(5, "Cijk_pv", 24, 6),
        _row(
            6,
            "void laguna_dense_initial_query_head_transpose_f32_kernel<false>",
            30,
            7,
        ),
        _row(7, "argmax_stage2_kernel", 37, 1),
    ]

    summary = _summarize_segment(
        {"length": 128, "chunks": 1, "rows": rows}
    )
    assert summary["families"]["swa_attention"]["calls"] == 6
    assert summary["families"]["swa_attention"]["duration_ns"] == 27
    assert summary["attention_duration_ns"] == 27


def test_lpf5_trace_attributes_packed_output_blas_attention_composite() -> None:
    rows = [
        _row(0, "gguf_q4_k_embedding_bf16_out_kernel", 0, 10, grid_y=128),
        _row(
            1,
            "void laguna_dense_initial_cache_bf16_to_f32_kernel<false>",
            10,
            2,
        ),
        _row(
            2,
            "void laguna_dense_initial_query_head_transpose_f32_kernel<true>",
            12,
            3,
        ),
        _row(3, "Cijk_qk", 15, 4),
        _row(
            4,
            "laguna_dense_initial_causal_softmax_wave_rows_f32_kernel",
            19,
            5,
        ),
        _row(5, "Cijk_pv", 24, 6),
        _row(
            6,
            "laguna_softplus_head_gate_packed_tiles_kernel<_Float16, true>",
            30,
            7,
        ),
        _row(7, "argmax_stage2_kernel", 37, 1),
    ]

    summary = _summarize_segment(
        {"length": 128, "chunks": 1, "rows": rows}
    )
    assert summary["families"]["swa_attention"]["calls"] == 5
    assert summary["families"]["swa_attention"]["duration_ns"] == 20
    assert summary["families"]["norm_rope_gate"]["calls"] == 1
    assert summary["families"]["norm_rope_gate"]["duration_ns"] == 7
    assert summary["attention_duration_ns"] == 20


def test_lpf5_trace_attributes_direct_packed_query_blas_attention_composite() -> None:
    rows = [
        _row(0, "gguf_q4_k_embedding_bf16_out_kernel", 0, 10, grid_y=128),
        _row(
            1,
            "void laguna_dense_initial_cache_bf16_to_f32_kernel<false>",
            10,
            2,
        ),
        _row(2, "Cijk_qk", 12, 4),
        _row(
            3,
            "laguna_dense_initial_causal_softmax_wave_rows_f32_kernel",
            16,
            5,
        ),
        _row(4, "Cijk_pv", 21, 6),
        _row(
            5,
            "laguna_softplus_head_gate_packed_tiles_kernel<_Float16, true>",
            27,
            7,
        ),
        _row(6, "argmax_stage2_kernel", 34, 1),
    ]

    summary = _summarize_segment(
        {"length": 128, "chunks": 1, "rows": rows}
    )
    assert summary["families"]["swa_attention"]["calls"] == 4
    assert summary["families"]["swa_attention"]["duration_ns"] == 17
    assert summary["families"]["norm_rope_gate"]["calls"] == 1
    assert summary["families"]["norm_rope_gate"]["duration_ns"] == 7
    assert summary["attention_duration_ns"] == 17


@pytest.mark.parametrize(
    ("kernel_name", "family"),
    [
        ("laguna_f16w_tiled_exact_kernel<unsigned short, 16>", "source_f16_projection"),
        ("laguna_f16w_wmma_kernel<unsigned short, 4, true>", "source_f16_projection"),
        (
            "Cijk_Alik_Bljk_HSS_BH_Bias_HA_S_SAV_UserArgs_MT96x96x32",
            "source_f16_projection",
        ),
        (
            "bf16_to_fp16_scaled_rows_kernel(unsigned short const*, _Float16*)",
            "source_f16_projection",
        ),
        ("q4_k_t16_selected_dual_direct_gemv_kernel<unsigned short>", "selected_q4_gate_up"),
        ("gguf_q8_1_mmq_ds8_f32_pack_bf16_kernel", "selected_q4_gate_up"),
        (
            "gguf_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32_"
            "prefill_compact32_kernel<1, false, true, 128>",
            "selected_q4_gate_up",
        ),
        (
            "gguf_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32_"
            "prefill_compact32_kernel<1, false, true, 128, true>",
            "selected_q4_gate_up",
        ),
        ("gguf_q8_1_mmq_ds4_f32_pack_bf16_kernel<1, false>", "selected_q4_q6_down"),
        (
            "gguf_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32_"
            "prefill_compact32_kernel<1, true, false, 64>",
            "selected_q4_q6_down",
        ),
        (
            "gguf_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32_"
            "prefill_compact32_kernel<1, true, false, 64, false>",
            "selected_q4_q6_down",
        ),
        (
            "gguf_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32_"
            "prefill_compact32_kernel<1>",
            "selected_q4_q6_down",
        ),
        ("qk_t16_selected_direct_gemv_kernel<unsigned short, 6>", "selected_q4_q6_down"),
        ("qk_t16_selected_grouped_smallm_kernel<unsigned short, 6>", "selected_q4_q6_down"),
        (
            "gguf_iq2_xs_selected_dual_silu_gemv_tile2_kernel",
            "selected_iq_gate_up",
        ),
        (
            "gguf_iq3_xxs_selected_dual_silu_gemv_kernel",
            "selected_iq_gate_up",
        ),
        ("gguf_iq3_xxs_selected_gemv_kernel", "selected_iq_down"),
        (
            "gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch_kernel<8>",
            "selected_iq_down",
        ),
        ("gguf_iq4_xs_selected_gemv_kernel", "selected_iq_down"),
        (
            "gguf_iq4_xs_selected_grouped_prefill_compact_kernel",
            "selected_iq_down",
        ),
        ("laguna_global_write_kv_rows_bf16_kernel", "prefill_kv_write"),
        ("laguna_sigmoid_correction_topk_f32_kernel", "router"),
        ("q4_k_pack8_gemv_kernel<unsigned short>", "dense_shared_quant_projection"),
        (
            "gguf_k_prefill_out_kernel<unsigned short, unsigned short, 6>",
            "dense_shared_quant_projection",
        ),
        (
            "gguf_k_prefill_out_rowbatch_kernel<unsigned short, float, 5, 8>",
            "dense_shared_quant_projection",
        ),
        (
            "gguf_k_prefill_out_coltile_rowbatch_kernel<unsigned short, float, 5, 4, 8>",
            "dense_shared_quant_projection",
        ),
        (
            "gguf_q6_k_prefill_wmma_kernel<unsigned short, unsigned short, 64, 16>",
            "dense_shared_quant_projection",
        ),
        ("silu_mul_separate_out_kernel<unsigned short>", "activation_reduce_residual"),
        ("weighted_lanes_sum_shared_add_out_kernel<unsigned short>", "activation_reduce_residual"),
        ("gguf_rmsnorm_bf16_f32_weight_kernel", "norm_rope_gate"),
        ("unrecognized_kernel", "other"),
    ],
)
def test_lpf5_trace_family_classifier(kernel_name: str, family: str) -> None:
    assert _kernel_family(kernel_name) == family


def test_lpf5_trace_attachment_fails_closed_on_segment_order(tmp_path: Path) -> None:
    child = {
        "pass": True,
        "performance_claim": False,
        "protocol": {"warmup_rows": 128, "lengths": [512, 1024, 4096]},
        "rows": [
            {"length": 512},
            {"length": 1024},
            {"length": 4096},
        ],
    }
    rows = [
        *_request_rows(100, 1),
        *_request_rows(1000, 4),
        *_request_rows(2000, 8),
        *_request_rows(3000, 32),
    ]
    attached = attach_summary(
        child,
        rows,
        trace_path=tmp_path / "trace.csv",
        trace_sha256="abc",
    )
    assert set(attached["profiler"]["lengths"]) == {"512", "1024", "4096"}
    assert len(attached["profiler"]["attention_resources"]) == 2
    assert attached["profiler"]["family_resources"]

    bad_rows = [
        *_request_rows(100, 1),
        *_request_rows(1000, 8),
        *_request_rows(2000, 4),
        *_request_rows(3000, 32),
    ]
    with pytest.raises(ValueError, match="do not match child order"):
        attach_summary(
            child,
            bad_rows,
            trace_path=tmp_path / "trace.csv",
            trace_sha256="abc",
        )
