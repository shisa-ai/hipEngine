"""Unit tests for the Surya profile aggregator.

The classifier and the per-call arithmetic decide what the profile *says*, so
they are tested against the real kernel names observed in the committed trace
rather than against invented ones.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "surya_profile_report", _ROOT / "scripts" / "surya_profile_report.py"
)
assert _SPEC is not None and _SPEC.loader is not None
R = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = R
_SPEC.loader.exec_module(R)

TRACE = _ROOT / "benchmarks" / "results" / "2026-09-11-gfx1151-surya-kernel-trace.json"


# --- classifier -------------------------------------------------------------


@pytest.mark.parametrize(
    "kernel,expected",
    [
        ("rocblas_gemm_plain", "gemm_library"),
        ("rocblas_gemm_strided_batched", "gemm_library"),
        ("qwen35_gdn_prefill_recurrent_kernel", "gdn_recurrent"),
        ("evie_gdn_l2norm_scale_kernel", "gdn_recurrent"),
        ("evie_gdn_gates_kernel", "gdn_recurrent"),
        ("qwen35_linear_attn_conv_decode_kernel", "gdn_conv"),
        ("qwen35_linear_attn_conv_prefill_segments_kernel", "gdn_conv"),
        ("surya_scatter_kv_kernel", "surya_kv_write"),
        ("surya_gdn_l2norm_kernel", "gdn_recurrent"),
        ("surya_split_qgate_kernel", "surya_other"),
        ("surya_causal_mask_scale_kernel", "surya_other"),
        ("evie_rmsnorm_kernel", "evie_norm"),
        ("evie_layernorm_kernel", "evie_norm"),
        ("evie_softmax_rows_kernel", "evie_softmax"),
        ("evie_rope_kernel", "evie_rope"),
        ("evie_add_kernel", "evie_elementwise"),
        ("evie_add_bias_kernel", "evie_elementwise"),
        ("evie_silu_mul_kernel", "evie_activation"),
        ("evie_gelu_tanh_kernel", "evie_activation"),
        ("evie_gelu_erf_kernel", "evie_activation"),
        ("rocclr_copybuffer", "copy_fill"),
        # rocBLAS dispatches to Tensile assembly kernels, which carry no "gemm"
        # substring; missing these hid 77% of the vision profile.
        ("Cijk_Alik_Bljk_SB_MT32x32x8_SN_1LDSB0_AMAS0_BL1_BS1_EPS0", "gemm_library"),
        ("Cijk_Ailk_Bljk_SB_GB_MT128x64x12_SN_1LDSB0_AMAS0_BL0_BS1", "gemm_library"),
    ],
)
def test_family_classification(kernel: str, expected: str) -> None:
    assert R._family(kernel) == expected


def test_gdn_is_classified_before_evie() -> None:
    """GDN kernels ship in the EVIE library but are linear attention.

    Bucketing them as evie_other would hide the second-largest family in the
    profile behind the vision elementwise noise.
    """

    assert R._family("evie_gdn_gates_kernel") == "gdn_recurrent"
    assert R._family("evie_gdn_l2norm_scale_kernel") == "gdn_recurrent"


def test_every_kernel_in_the_committed_trace_is_classified() -> None:
    """No committed kernel may fall through to the catch-all ``other``."""

    if not TRACE.exists():
        pytest.skip("committed trace artifact not present")
    data = json.loads(TRACE.read_text())
    names = list(data["kernels"]) if isinstance(data["kernels"], dict) else [
        k["name"] for k in data["kernels"]
    ]
    assert names
    unclassified = [name for name in names if R._family(name) == "other"]
    assert not unclassified, f"unclassified kernels: {unclassified}"


def test_every_kernel_in_the_profile_artifact_is_classified() -> None:
    """The profile must have no catch-all bucket hiding real work."""

    path = (_ROOT / "benchmarks" / "results"
            / "2026-09-11-gfx1151-surya-post-sgemv-profile.json")
    if not path.exists():
        pytest.skip("profile artifact not present")
    data = json.loads(path.read_text())
    offenders: list[str] = []
    for phase, summary in data["phases"].items():
        for entry in summary.get("top_kernels", []):
            if R._family(entry["name"]) == "other":
                offenders.append(f"{phase}:{entry['name']}")
    assert not offenders, f"unclassified in profile: {offenders}"


# --- aggregation ------------------------------------------------------------


def _write_kernel_csv(directory: Path, rows: list[tuple[str, int, int]],
                      *, nested: bool = False) -> None:
    """Write a kernel trace; ``nested`` mimics rocprofv3's hostname subdir."""

    target = directory / "zbook" if nested else directory
    target.mkdir(parents=True, exist_ok=True)
    path = target / "trace_kernel_trace.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Kernel_Name", "Start_Timestamp", "End_Timestamp",
                         "Grid_Size_X", "Workgroup_Size_X"])
        for name, start, end in rows:
            writer.writerow([name, start, end, 1, 64])


def test_reader_finds_csvs_in_rocprofv3s_hostname_subdirectory(tmp_path: Path) -> None:
    """rocprofv3 nests its CSVs under a hostname directory, not the output root."""

    _write_kernel_csv(tmp_path, [("rocblas_gemm_plain", 0, 1_000_000)], nested=True)
    assert len(R._read_kernels(tmp_path)) == 1

    copies = tmp_path / "zbook"
    copies.mkdir(parents=True, exist_ok=True)
    with (copies / "trace_memory_copy_trace.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Kind", "Direction", "Start_Timestamp", "End_Timestamp"])
        writer.writerow(["MEMORY_COPY", "MEMORY_COPY_HOST_TO_DEVICE", 0, 500])
    assert len(R._read_copies(tmp_path)) == 1


def test_per_call_cost_divides_by_the_calls_the_driver_made(tmp_path: Path) -> None:
    """Warmup is included in the driver's call count, so the division is exact."""

    # 4 calls of 1 ms each
    _write_kernel_csv(
        tmp_path,
        [("rocblas_gemm_plain", i * 2_000_000, i * 2_000_000 + 1_000_000)
         for i in range(4)],
    )
    kernels = R._read_kernels(tmp_path)
    summary = R._summarize(kernels, [], calls=4)
    assert summary["kernel_total_ms"] == pytest.approx(4.0)
    assert summary["kernel_per_call_ms"] == pytest.approx(1.0)
    assert summary["kernel_calls_per_call"] == pytest.approx(1.0)
    assert summary["kernel_families"][0]["name"] == "gemm_library"
    assert summary["kernel_families"][0]["share_pct"] == pytest.approx(100.0)


def test_zero_calls_does_not_divide_by_zero(tmp_path: Path) -> None:
    _write_kernel_csv(tmp_path, [("evie_add_kernel", 0, 500)])
    summary = R._summarize(R._read_kernels(tmp_path), [], calls=0)
    assert summary["kernel_per_call_ms"] is None
    assert summary["kernel_calls_per_call"] is None
    assert summary["kernel_total_ms"] == pytest.approx(0.0005)


def test_malformed_and_reversed_rows_are_dropped(tmp_path: Path) -> None:
    directory = tmp_path / "t"
    _write_kernel_csv(directory, [("evie_add_kernel", 0, 1000)])
    path = directory / "trace_kernel_trace.csv"
    with path.open("a", newline="") as handle:
        handle.write("evie_add_kernel,not_a_number,5,1,64\n")
        handle.write("evie_add_kernel,900,100,1,64\n")  # end before start
    kernels = R._read_kernels(directory)
    assert len(kernels) == 1, "malformed and reversed rows must be dropped"


def test_transfers_are_aggregated_by_direction(tmp_path: Path) -> None:
    """The trace records direction and duration but no byte count."""

    directory = tmp_path / "t"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "trace_memory_copy_trace.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Kind", "Direction", "Start_Timestamp", "End_Timestamp"])
        writer.writerow(["MEMORY_COPY", "MEMORY_COPY_HOST_TO_DEVICE", 0, 500_000])
        writer.writerow(["MEMORY_COPY", "MEMORY_COPY_DEVICE_TO_HOST", 1_000_000, 1_200_000])
    summary = R._summarize([], R._read_copies(directory), calls=2)
    transfers = summary["transfers"]
    assert transfers["bytes_available"] is False
    assert transfers["total_mb"] is None, "must not invent a byte count"
    assert transfers["total_ms"] == pytest.approx(0.7)
    assert transfers["ms_per_call"] == pytest.approx(0.35)
    assert set(transfers["by_kind"]) == {"host_to_device", "device_to_host"}
    assert transfers["by_kind"]["host_to_device"]["ms"] == pytest.approx(0.5)


def test_transfer_bytes_are_reported_when_the_trace_has_them(tmp_path: Path) -> None:
    directory = tmp_path / "t"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "trace_memory_copy_trace.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Kind", "Direction", "Bytes",
                         "Start_Timestamp", "End_Timestamp"])
        writer.writerow(["MEMORY_COPY", "MEMORY_COPY_HOST_TO_DEVICE", 4_000_000, 0, 500])
    transfers = R._summarize([], R._read_copies(directory), calls=1)["transfers"]
    assert transfers["bytes_available"] is True
    assert transfers["total_mb"] == pytest.approx(4.0)
    assert transfers["by_kind"]["host_to_device"]["mb"] == pytest.approx(4.0)


def test_summary_is_ranked_by_total_time(tmp_path: Path) -> None:
    _write_kernel_csv(
        tmp_path,
        [
            ("evie_add_kernel", 0, 1_000),                 # 1 us, elementwise
            ("rocblas_gemm_plain", 0, 9_000_000),          # 9 ms, gemm
            ("evie_rmsnorm_kernel", 0, 2_000),             # 2 us, norm
        ],
    )
    summary = R._summarize(R._read_kernels(tmp_path), [], calls=1)
    names = [entry["name"] for entry in summary["kernel_families"]]
    assert names[0] == "gemm_library", names
    totals = [entry["total_ms"] for entry in summary["kernel_families"]]
    assert totals == sorted(totals, reverse=True)
