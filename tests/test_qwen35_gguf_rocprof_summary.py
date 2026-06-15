"""Tests for ``scripts/qwen35_gguf_rocprof_summary.py`` (P9.E1).

Synthetic CSV inputs exercise the bucket classifier, per-phase aggregation,
optional footprint-driven effective GB/s, and the paired prefill+decode mode
(both with and without the leading prefill-prefix strip).
"""

from __future__ import annotations

import csv as _csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence

import pytest


def _load_script_module():
    """Load the script-as-module so dataclasses inside it can find their module.

    ``dataclasses`` resolves the host module by name via ``sys.modules`` when
    deciding whether default values are ``KW_ONLY`` markers etc. So we have
    to register the loaded module under its ``__name__`` before executing it.
    """

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "qwen35_gguf_rocprof_summary.py"
    module_name = "_qwen35_gguf_rocprof_summary_test_module"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module()


_CSV_COLUMNS = (
    "Kernel_Name",
    "Start_Timestamp",
    "End_Timestamp",
    "VGPR_Count",
    "Accum_VGPR_Count",
    "SGPR_Count",
    "Scratch_Size",
    "LDS_Block_Size",
    "Workgroup_Size_X",
    "Workgroup_Size_Y",
    "Workgroup_Size_Z",
    "Grid_Size_X",
    "Grid_Size_Y",
    "Grid_Size_Z",
)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write a rocprofv3-shaped CSV with proper quoting for templated kernel names."""

    with path.open("w", newline="") as fh:
        writer = _csv.DictWriter(
            fh,
            fieldnames=list(_CSV_COLUMNS),
            quoting=_csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({col: r.get(col, "") for col in _CSV_COLUMNS})


# ---------------------------------------------------------------------------
# Bucket classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        # MoE compact selected -- P8 WMMA prefill and P9.B GEMV decode + legacy
        ("gguf_q4_k_selected_dual_wmma_prefill_compact_kernel<unsigned short>", "moe_q4_k_selected_dual_wmma_prefill"),
        ("gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_kernel<unsigned short>", "moe_q4_k_selected_dual_wmma_prefill"),
        ("gguf_q4_k_selected_dual_pack8_gemv_decode_compact_kernel<unsigned short>", "moe_q4_k_selected_dual_pack8_gemv_decode_p9"),
        ("q4_k_t16_selected_dual_direct_gemv_kernel<unsigned short>", "moe_q4_k_selected_dual_t16_gemv_decode_p9"),
        ("q4_k_t16_selected_dual_gemv_kernel<unsigned short>", "moe_q4_k_selected_dual_t16_gemv_decode_p9"),
        ("gguf_q4_k_selected_dual_prefill_out_kernel<unsigned short, unsigned short>", "moe_q4_k_selected_legacy_decode"),
        ("gguf_k_selected_wmma_prefill_compact_kernel<unsigned short, 5>", "moe_q5_k_selected_wmma_prefill"),
        ("gguf_k_selected_wmma_prefill_compact_kernel<unsigned short, 6>", "moe_q6_k_selected_wmma_prefill"),
        ("gguf_k_t16_selected_wmma_prefill_compact_kernel<unsigned short, 5>", "moe_q5_k_selected_wmma_prefill"),
        ("gguf_k_t16_selected_wmma_prefill_compact_kernel<unsigned short, 6>", "moe_q6_k_selected_wmma_prefill"),
        ("gguf_k_selected_pack8_gemv_decode_compact_kernel<unsigned short, 5>", "moe_q5_k_selected_pack8_gemv_decode_p9"),
        ("gguf_k_selected_pack8_gemv_decode_compact_kernel<unsigned short, 6>", "moe_q6_k_selected_pack8_gemv_decode_p9"),
        ("qk_t16_selected_direct_gemv_kernel<unsigned short, 5>", "moe_q5_k_selected_t16_gemv_decode_p9"),
        ("qk_t16_selected_direct_gemv_kernel<unsigned short, 6>", "moe_q6_k_selected_t16_gemv_decode_p9"),
        ("qk_t16_selected_gemv_kernel<unsigned short, 5>", "moe_q5_k_selected_t16_gemv_decode_p9"),
        ("gguf_k_selected_pack8_prefill_out_kernel<unsigned short, unsigned short, 5>", "moe_q5_k_selected_legacy_decode"),
        ("gguf_k_selected_pack8_prefill_out_kernel<unsigned short, unsigned short, 6>", "moe_q6_k_selected_legacy_decode"),
        # Dense Q8_0
        ("gguf_q8_0_prefill_wmma_kernel<unsigned short, unsigned short, 32, 32>", "dense_q8_0_wmma_prefill"),
        ("gguf_q8_0_prefill_dual_wmma_kernel<unsigned short, unsigned short, 16, 32>", "dense_q8_0_wmma_prefill"),
        ("gguf_q8_0_t16_prefill_wmma_kernel<unsigned short, unsigned short, 64, 32>", "dense_q8_0_wmma_prefill"),
        ("gguf_q8_0_pack8_gemv_kernel<unsigned short>", "other"),  # legacy non-decode pack8
        ("gguf_q8_0_pack8_gemv_decode_kernel<unsigned short>", "dense_q8_0_pack8_gemv_decode_p9"),
        ("gguf_q8_0_pack8_dual_gate_up_gemv_decode_kernel<unsigned short>", "dense_q8_0_pack8_gemv_decode_p9"),
        ("q8_0_t16_gemv_kernel<unsigned short, unsigned short>", "dense_q8_0_t16_gemv_decode_p9"),
        ("q8_0_t16_dual_gemv_kernel<unsigned short, unsigned short>", "dense_q8_0_t16_gemv_decode_p9"),
        ("q8_0_t16_dual_split_gemv_kernel<unsigned short, unsigned short>", "dense_q8_0_t16_gemv_decode_p9"),
        ("gguf_k_pack8_prefill_out_kernel<unsigned short, unsigned short, 8>", "dense_q8_0_legacy_decode"),
        ("gguf_k_pack8_prefill_out_kernel<unsigned short, float, 6>", "dense_q6_k_legacy_decode"),
        # Dense Q4_K / Q6_K
        ("gguf_q4_k_pack8_gemv_decode_kernel<unsigned short, unsigned short>", "dense_q4_k_pack8_gemv_decode_p9"),
        ("gguf_q4_k_pack8_gemv_decode_kernel<unsigned short, float>", "dense_q4_k_pack8_gemv_decode_p9"),
        ("gguf_q6_k_pack8_gemv_decode_kernel<unsigned short, float>", "dense_q6_k_pack8_gemv_decode_p9"),
        ("q6_k_t16_gemv_kernel<unsigned short, float>", "dense_q6_k_t16_gemv_decode_p9"),
        ("gguf_q4_k_prefill_wmma_kernel<unsigned short, unsigned short>", "dense_q4_k_prefill"),
        ("gguf_q4_k_prefill_dual_wmma_kernel<unsigned short>", "dense_q4_k_prefill"),
        ("gguf_q4_k_prefill_out_kernel<unsigned short, unsigned short>", "dense_q4_k_prefill"),
        # GDN
        ("qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_kernel<unsigned short>", "gdn_prefill_recurrent"),
        ("qwen35_gdn_prefill_recurrent_k2_kernel", "gdn_prefill_recurrent"),
        ("qwen35_gdn_prefill_recurrent_segments_k2_kernel", "gdn_prefill_recurrent"),
        ("qwen35_gdn_prefill_rmsnorm_gate_bf16_kernel<unsigned short>", "gdn_prefill_rmsnorm_gate"),
        ("qwen35_linear_attn_prefill_prepare_kernel<unsigned short>", "gdn_prepare"),
        ("qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<unsigned short>", "gdn_decode"),
        ("qwen35_linear_attn_conv_prefill_kernel", "linear_attn_conv"),
        # Full attention
        ("qwen35_paged_full_attn_prefill_gqa_gate_bf16_kernel<true>", "full_attention_prefill"),
        ("qwen35_paged_full_attn_decode_context_tensor_kernel", "full_attention_decode"),
        # Router + scheduler + combine + silu
        ("qwen35_router_logits_token_tile_kernel<unsigned short, 4>", "router"),
        ("qwen35_router_select_kernel", "router"),
        ("qwen35_moe_group_count_kernel", "moe_scheduler"),
        ("qwen35_moe_group_prefix_kernel", "moe_scheduler"),
        ("qwen35_moe_group_scatter_gather_kernel", "moe_scheduler"),
        ("qwen35_moe_wmma_tile_map_kernel", "moe_scheduler"),
        ("silu_mul_dual_out_kernel<unsigned short>", "silu_mul"),
        ("weighted_lanes_sum_out_kernel<unsigned short, float>", "moe_combine"),
        ("shared_gate_combine_residual_batch_out_kernel<unsigned short>", "moe_combine"),
        ("gguf_rmsnorm_bf16_f32_weight_kernel<unsigned short>", "rmsnorm"),
        # KV + runtime
        ("qwen35_write_paged_kv_cache_kernel", "kv_write"),
        ("hipMemcpy", "copy"),
        ("__amd_rocclr_copyBuffer", "copy"),
        ("__amd_rocclr_fillBuffer", "copy"),
        ("__amd_rocclr_fillBufferAligned", "copy"),
        # Unknown
        ("some_unknown_kernel_name", "other"),
        ("some_unclassified_prefill_kernel", "other"),
    ],
)
def test_bucket_classifier_covers_expected_kernels(name: str, expected: str) -> None:
    assert SCRIPT.classify_kernel(name) == expected


# ---------------------------------------------------------------------------
# CSV parsing + aggregation
# ---------------------------------------------------------------------------


def test_read_kernel_trace_skips_malformed_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "trace.csv"
    csv_path.write_text(
        "Kernel_Name,Start_Timestamp,End_Timestamp,VGPR_Count,Scratch_Size,LDS_Block_Size\n"
        "good_kernel,100,200,16,0,256\n"
        ",100,200,16,0,256\n"  # empty name
        "bad_ts,not_a_number,200,,,\n"  # bad timestamp
        "negative_dur,200,100,16,0,256\n"  # end < start
        "another_good,300,500,8,0,128\n"
    )
    kernels = SCRIPT.read_kernel_trace(csv_path)
    assert [k.name for k in kernels] == ["good_kernel", "another_good"]
    assert [k.duration_ns for k in kernels] == [100, 200]


def test_summary_includes_rocprof_resource_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "trace.csv"
    _write_csv(
        csv_path,
        [
            {
                "Kernel_Name": "q8_0_t16_gemv_kernel<unsigned short, unsigned short>",
                "Start_Timestamp": 0,
                "End_Timestamp": 1_000_000,
                "VGPR_Count": 64,
                "Accum_VGPR_Count": 8,
                "SGPR_Count": 48,
                "Scratch_Size": 0,
                "LDS_Block_Size": 128,
                "Workgroup_Size_X": 128,
                "Workgroup_Size_Y": 1,
                "Workgroup_Size_Z": 1,
                "Grid_Size_X": 10,
                "Grid_Size_Y": 2,
                "Grid_Size_Z": 1,
            },
            {
                "Kernel_Name": "q8_0_t16_gemv_kernel<unsigned short, unsigned short>",
                "Start_Timestamp": 1_000_000,
                "End_Timestamp": 2_000_000,
                "VGPR_Count": 64,
                "Accum_VGPR_Count": 8,
                "SGPR_Count": 48,
                "Scratch_Size": 0,
                "LDS_Block_Size": 128,
                "Workgroup_Size_X": 128,
                "Workgroup_Size_Y": 1,
                "Workgroup_Size_Z": 1,
                "Grid_Size_X": 11,
                "Grid_Size_Y": 2,
                "Grid_Size_Z": 1,
            },
        ],
    )
    summary = SCRIPT.build_summary(
        prefill_csv=None,
        decode_csv=None,
        single_csv=csv_path,
        tokens_prefill=2,
        tokens_decode=None,
        footprints=SCRIPT._QWEN36_35B_A3B_DEFAULT_FOOTPRINTS_PER_DISPATCH,
        top=10,
        prefill_dispatches_from_single=False,
    )
    phase = summary["phases"]["prefill"]
    bucket = phase["buckets"][0]
    resources = bucket["resource_summary"]
    assert resources["vgpr_count"] == {"min": 64, "max": 64, "values": [64]}
    assert resources["accum_vgpr_count"] == {"min": 8, "max": 8, "values": [8]}
    assert resources["sgpr_count"] == {"min": 48, "max": 48, "values": [48]}
    assert resources["scratch_bytes"] == {"min": 0, "max": 0, "values": [0]}
    assert resources["lds_bytes"] == {"min": 128, "max": 128, "values": [128]}
    assert resources["workgroup_size"] == {"min": 128, "max": 128, "values": [128]}
    assert resources["grid_size"] == {"min": 20, "max": 22, "values": [20, 22]}
    assert phase["top_kernels"][0]["resource_summary"]["scratch_bytes"]["max"] == 0


def test_summary_single_csv_prefill_phase(tmp_path: Path) -> None:
    csv_path = tmp_path / "prefill.csv"
    # Synthetic prefill: 2x Q8_0 WMMA (1ms each), 1x Q4_K selected dual WMMA
    # (3ms), 1x GDN k2 (10ms), 1x router (0.5ms).
    _write_csv(
        csv_path,
        [
            {"Kernel_Name": "gguf_q8_0_prefill_wmma_kernel<unsigned short, unsigned short, 32, 32>", "Start_Timestamp": 0, "End_Timestamp": 1_000_000},
            {"Kernel_Name": "gguf_q8_0_prefill_wmma_kernel<unsigned short, unsigned short, 32, 32>", "Start_Timestamp": 1_000_000, "End_Timestamp": 2_000_000},
            {"Kernel_Name": "gguf_q4_k_selected_dual_wmma_prefill_compact_kernel<unsigned short>", "Start_Timestamp": 2_000_000, "End_Timestamp": 5_000_000},
            {"Kernel_Name": "qwen35_gdn_prefill_recurrent_k2_kernel", "Start_Timestamp": 5_000_000, "End_Timestamp": 15_000_000},
            {"Kernel_Name": "qwen35_router_logits_token_tile_kernel<unsigned short, 4>", "Start_Timestamp": 15_000_000, "End_Timestamp": 15_500_000},
        ],
    )
    summary = SCRIPT.build_summary(
        prefill_csv=None,
        decode_csv=None,
        single_csv=csv_path,
        tokens_prefill=512,
        tokens_decode=None,
        footprints=SCRIPT._QWEN36_35B_A3B_DEFAULT_FOOTPRINTS_PER_DISPATCH,
        top=10,
        prefill_dispatches_from_single=False,
    )
    assert summary["schema"] == SCRIPT.SCHEMA
    assert set(summary["phases"]) == {"prefill"}
    phase = summary["phases"]["prefill"]
    assert phase["total_kernel_ms"] == pytest.approx(15.5, abs=1e-6)
    assert phase["total_dispatches"] == 5
    assert phase["ms_per_token"] == pytest.approx(15.5 / 512, abs=1e-6)

    buckets = {b["bucket"]: b for b in phase["buckets"]}
    # Bucket totals match the synthetic data.
    assert buckets["gdn_prefill_recurrent"]["total_ms"] == pytest.approx(10.0)
    assert buckets["gdn_prefill_recurrent"]["dispatches"] == 1
    assert buckets["dense_q8_0_wmma_prefill"]["total_ms"] == pytest.approx(2.0)
    assert buckets["dense_q8_0_wmma_prefill"]["dispatches"] == 2
    assert buckets["moe_q4_k_selected_dual_wmma_prefill"]["total_ms"] == pytest.approx(3.0)
    assert buckets["router"]["total_ms"] == pytest.approx(0.5)
    # ms/token populated.
    for b in phase["buckets"]:
        assert b["ms_per_token"] == pytest.approx(b["total_ms"] / 512, abs=1e-6)


def test_summary_effective_gb_s_uses_footprint_overrides(tmp_path: Path) -> None:
    csv_path = tmp_path / "decode.csv"
    # One Q8_0 P9 decode dispatch in 1 ms. With a footprint of 1 GB (1e9 bytes)
    # the back-calculated GB/s is exactly 1e3 GB/s, regardless of model defaults.
    _write_csv(
        csv_path,
        [
            {"Kernel_Name": "gguf_q8_0_pack8_gemv_decode_kernel<unsigned short>", "Start_Timestamp": 0, "End_Timestamp": 1_000_000},
        ],
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"dense_q8_0_pack8_gemv_decode_p9": 10**9}))
    rc = SCRIPT.main(
        [
            "--csv",
            str(csv_path),
            "--tokens-prefill",
            "1",
            "--config-json",
            str(config),
            "--json",
            str(tmp_path / "out.json"),
            "--quiet",
        ]
    )
    assert rc == 0
    out = json.loads((tmp_path / "out.json").read_text())
    bucket = next(
        b for b in out["phases"]["prefill"]["buckets"]
        if b["bucket"] == "dense_q8_0_pack8_gemv_decode_p9"
    )
    assert bucket["footprint_bytes_per_dispatch"] == 10**9
    assert bucket["effective_gb_s"] == pytest.approx(1000.0, rel=1e-6)
    assert bucket["total_ms"] == pytest.approx(1.0)
    assert bucket["dispatches"] == 1


def test_summary_no_footprint_emits_none_gb_s(tmp_path: Path) -> None:
    csv_path = tmp_path / "trace.csv"
    _write_csv(
        csv_path,
        [
            {"Kernel_Name": "qwen35_router_logits_token_tile_kernel<unsigned short, 4>", "Start_Timestamp": 0, "End_Timestamp": 1_000_000},
        ],
    )
    summary = SCRIPT.build_summary(
        prefill_csv=None,
        decode_csv=None,
        single_csv=csv_path,
        tokens_prefill=None,
        tokens_decode=None,
        footprints=SCRIPT._QWEN36_35B_A3B_DEFAULT_FOOTPRINTS_PER_DISPATCH,
        top=10,
        prefill_dispatches_from_single=False,
    )
    bucket = next(b for b in summary["phases"]["prefill"]["buckets"] if b["bucket"] == "router")
    assert bucket["effective_gb_s"] is None
    assert bucket["footprint_bytes_per_dispatch"] is None


# ---------------------------------------------------------------------------
# Paired prefill+decode mode + prefix strip
# ---------------------------------------------------------------------------


def test_paired_mode_strips_prefill_prefix(tmp_path: Path) -> None:
    prefill_csv = tmp_path / "prefill.csv"
    paired_csv = tmp_path / "paired.csv"
    # Prefill prefix: 2 kernels (1ms each). Decode tail: 3 kernels (2ms each).
    prefix_rows = [
        {"Kernel_Name": "qwen35_router_logits_token_tile_kernel<unsigned short, 4>", "Start_Timestamp": 0, "End_Timestamp": 1_000_000},
        {"Kernel_Name": "gguf_q8_0_prefill_wmma_kernel<unsigned short, unsigned short, 32, 32>", "Start_Timestamp": 1_000_000, "End_Timestamp": 2_000_000},
    ]
    decode_rows = [
        {"Kernel_Name": "gguf_q8_0_pack8_gemv_decode_kernel<unsigned short>", "Start_Timestamp": 2_000_000, "End_Timestamp": 4_000_000},
        {"Kernel_Name": "gguf_q4_k_selected_dual_pack8_gemv_decode_compact_kernel<unsigned short>", "Start_Timestamp": 4_000_000, "End_Timestamp": 6_000_000},
        {"Kernel_Name": "qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<unsigned short>", "Start_Timestamp": 6_000_000, "End_Timestamp": 8_000_000},
    ]
    _write_csv(prefill_csv, prefix_rows)
    _write_csv(paired_csv, prefix_rows + decode_rows)
    out_path = tmp_path / "out.json"
    rc = SCRIPT.main(
        [
            "--prefill-csv",
            str(prefill_csv),
            "--decode-csv",
            str(paired_csv),
            "--tokens-prefill",
            "512",
            "--tokens-decode",
            "128",
            "--strip-prefill-prefix",
            "--json",
            str(out_path),
            "--quiet",
        ]
    )
    assert rc == 0
    summary = json.loads(out_path.read_text())
    prefill = summary["phases"]["prefill"]
    decode = summary["phases"]["decode"]
    assert prefill["total_dispatches"] == 2
    assert decode["total_dispatches"] == 3  # prefill prefix stripped
    assert decode["total_kernel_ms"] == pytest.approx(6.0)
    assert summary["inputs"]["decode_prefill_prefix_dispatches"] == 2
    # Decode-side ms/token reflects only the 3 decode dispatches.
    assert decode["ms_per_token"] == pytest.approx(6.0 / 128, abs=1e-6)


def test_paired_mode_without_strip_keeps_prefix(tmp_path: Path) -> None:
    prefill_csv = tmp_path / "prefill.csv"
    paired_csv = tmp_path / "paired.csv"
    prefix_rows = [
        {"Kernel_Name": "qwen35_router_logits_token_tile_kernel<unsigned short, 4>", "Start_Timestamp": 0, "End_Timestamp": 1_000_000},
    ]
    decode_rows = [
        {"Kernel_Name": "qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<unsigned short>", "Start_Timestamp": 1_000_000, "End_Timestamp": 3_000_000},
    ]
    _write_csv(prefill_csv, prefix_rows)
    _write_csv(paired_csv, prefix_rows + decode_rows)
    out_path = tmp_path / "out.json"
    rc = SCRIPT.main(
        [
            "--prefill-csv",
            str(prefill_csv),
            "--decode-csv",
            str(paired_csv),
            "--json",
            str(out_path),
            "--quiet",
        ]
    )
    assert rc == 0
    summary = json.loads(out_path.read_text())
    decode = summary["phases"]["decode"]
    # Without the strip flag, the decode phase reports the full CSV.
    assert decode["total_dispatches"] == 2
    assert any(
        "prefill prefix not subtracted" in note for note in summary["notes"]
    )


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------


def test_cli_requires_one_of_csv_or_pair(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        SCRIPT.main([])


def test_cli_rejects_mixed_csv_and_pair(tmp_path: Path) -> None:
    csv_path = tmp_path / "trace.csv"
    _write_csv(csv_path, [])
    with pytest.raises(SystemExit):
        SCRIPT.main(
            [
                "--csv",
                str(csv_path),
                "--prefill-csv",
                str(csv_path),
                "--decode-csv",
                str(csv_path),
            ]
        )


def test_cli_rejects_missing_csv(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        SCRIPT.main(["--csv", str(tmp_path / "does_not_exist.csv")])
