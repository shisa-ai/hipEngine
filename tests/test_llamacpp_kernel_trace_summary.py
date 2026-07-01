"""Tests for scripts/llamacpp_kernel_trace_summary.py."""

from __future__ import annotations

import csv as csv_lib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "llamacpp_kernel_trace_summary.py"
    module_name = "_llamacpp_kernel_trace_summary_test_module"
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
    with path.open("w", newline="") as fh:
        writer = csv_lib.DictWriter(fh, fieldnames=list(_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in _CSV_COLUMNS})


@pytest.mark.parametrize(
    "name,expected",
    [
        ("void mul_mat_vec_q_moe<(ggml_type)10, 2>(void const*, void const*, int const*, float*)", "llama_mmvq_moe"),
        ("void mul_mat_vec_q<(ggml_type)14, 1, false, false>(void const*, void const*, int const*)", "llama_mmvq"),
        ("void mul_mat_vec_f<float, float, 4, 256, false, false>(float const*, float const*)", "llama_mmvf"),
        ("quantize_q8_1(float const*, void*, long, long, long, long, long, unsigned int)", "llama_quantize_q8_1"),
        ("void k_argsort_f32_i32<(ggml_sort_order)1>(float const*, int*, int, int)", "llama_topk_argsort"),
        ("void gated_delta_net_cuda<128, false, false>(float const*, float const*)", "llama_gdn"),
        ("void flash_attn_tile<256, 256, 4, 8, false>(char const*, char const*)", "llama_flash_attn"),
        ("rope_f32_kernel", "llama_rope"),
        ("void l2_norm_f32<32>(float const*, float*, int, long, long, long, float)", "llama_norm"),
        ("__amd_rocclr_copyBuffer", "llama_copy_layout"),
        ("void concat_non_cont<unsigned int, 0>(char const*, char const*, char*)", "llama_copy_layout"),
        ("void k_bin_bcast<&(op_mul(float, float)), float, float, float, float const*>(float const*)", "llama_elementwise"),
        ("unknown_kernel", "other"),
    ],
)
def test_classify_kernel(name: str, expected: str) -> None:
    assert SCRIPT.classify_kernel(name) == expected


def test_read_kernel_trace_skips_bad_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "trace.csv"
    csv_path.write_text(
        "Kernel_Name,Start_Timestamp,End_Timestamp,VGPR_Count,Scratch_Size,LDS_Block_Size\n"
        "good,10,20,32,0,256\n"
        ",10,20,32,0,256\n"
        "bad,xx,20,32,0,256\n"
        "negative,30,20,32,0,256\n"
    )
    rows = SCRIPT.read_kernel_trace(csv_path)
    assert len(rows) == 1
    assert rows[0].name == "good"
    assert rows[0].duration_ns == 10
    assert rows[0].vgpr == 32


def test_build_summary_groups_buckets_and_top_kernels(tmp_path: Path) -> None:
    csv_path = tmp_path / "trace.csv"
    _write_csv(
        csv_path,
        [
            {
                "Kernel_Name": "void mul_mat_vec_q_moe<(ggml_type)10, 2>(void const*)",
                "Start_Timestamp": 0,
                "End_Timestamp": 2_000_000,
                "VGPR_Count": 64,
            },
            {
                "Kernel_Name": "void mul_mat_vec_q<(ggml_type)14, 1, false, false>(void const*)",
                "Start_Timestamp": 2_000_000,
                "End_Timestamp": 3_000_000,
                "VGPR_Count": 60,
            },
            {
                "Kernel_Name": "quantize_q8_1(float const*, void*)",
                "Start_Timestamp": 3_000_000,
                "End_Timestamp": 3_500_000,
                "VGPR_Count": 24,
            },
            {
                "Kernel_Name": "__amd_rocclr_copyBuffer",
                "Start_Timestamp": 3_500_000,
                "End_Timestamp": 3_750_000,
                "VGPR_Count": 16,
            },
        ],
    )
    out_json = tmp_path / "summary.json"
    rc = SCRIPT.main(["--csv", str(csv_path), "--json", str(out_json), "--label", "unit", "--top", "2"])
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["schema"] == SCRIPT.SCHEMA
    assert payload["label"] == "unit"
    assert payload["total_kernel_ms"] == pytest.approx(3.75)
    assert payload["total_dispatches"] == 4
    buckets = {row["bucket"]: row for row in payload["buckets"]}
    assert buckets["llama_mmvq_moe"]["total_ms"] == pytest.approx(2.0)
    assert buckets["llama_mmvq"]["total_ms"] == pytest.approx(1.0)
    assert buckets["llama_quantize_q8_1"]["total_ms"] == pytest.approx(0.5)
    assert buckets["llama_copy_layout"]["total_ms"] == pytest.approx(0.25)
    assert payload["top_kernels"][0]["kernel"].startswith("void mul_mat_vec_q_moe")
    assert len(payload["top_kernels"]) == 2
