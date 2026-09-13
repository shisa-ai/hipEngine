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
            / "2026-09-12-gfx1151-surya-phase-attributed-profile.json")
    if not path.exists():
        pytest.skip("profile artifact not present")
    data = json.loads(path.read_text())
    offenders: list[str] = []
    for phase, summary in data["phases"].items():
        for entry in summary.get("top_kernels", []):
            if R._family(entry["name"]) == "other":
                offenders.append(f"{phase}:{entry['name']}")
    assert not offenders, f"unclassified in profile: {offenders}"


def test_profile_artifact_headline_is_range_attributed() -> None:
    """A committed profile must not present whole-process totals as a phase.

    The 2026-09-11 profile did exactly that for decode: the vision tower and the
    prefill that seed the KV state were billed to decode, which made a 1066-token
    context look 1.58x costlier per token than a 106-token one when it is ~1.05x.
    """

    path = (_ROOT / "benchmarks" / "results"
            / "2026-09-12-gfx1151-surya-phase-attributed-profile.json")
    if not path.exists():
        pytest.skip("profile artifact not present")
    data = json.loads(path.read_text())
    assert "range" in data["protocol"]["attribution"]
    for phase, summary in data["phases"].items():
        assert summary["attribution"] == "range", phase
        phase_range = next(r for r in summary["ranges"] if r["is_phase"])
        assert phase_range["name"] == summary["phase_range"]
        # The headline is the phase range, not the whole process.
        assert summary["kernel_per_call_ms"] == pytest.approx(
            phase_range["kernel_per_call_ms"])
        assert summary["kernel_total_ms"] == pytest.approx(
            phase_range["kernel_total_ms"])
        assert summary["whole_process"]["kernel_total_ms"] >= summary["kernel_total_ms"]
    decode = data["phases"]["decode-full"]
    # Setup is the vision tower plus a prefill; on the full page it is larger
    # than the whole 200-step decode, so the distinction is not cosmetic.
    setup = next(r for r in decode["ranges"] if not r["is_phase"])
    assert setup["kernel_total_ms"] > decode["kernel_total_ms"] / 2
    assert decode["whole_process"]["kernel_per_call_ms"] > 1.4 * decode["kernel_per_call_ms"]


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
    kernels = R._read_kernels(tmp_path)
    assert len(kernels) == 1
    # start_ns is what range attribution needs; without it a row cannot be
    # assigned to a phase and every kernel would land in "outside".
    assert kernels[0]["start_ns"] == 0

    copies = tmp_path / "zbook"
    copies.mkdir(parents=True, exist_ok=True)
    with (copies / "trace_memory_copy_trace.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Kind", "Direction", "Start_Timestamp", "End_Timestamp"])
        writer.writerow(["MEMORY_COPY", "MEMORY_COPY_HOST_TO_DEVICE", 0, 500])
    read_copies = R._read_copies(tmp_path)
    assert len(read_copies) == 1
    assert read_copies[0]["start_ns"] == 0


# --- range attribution ------------------------------------------------------


def _ranges() -> list[dict]:
    """A decode trace: setup at 0-100 ms, decode at 100-300 ms."""

    return [
        {"name": "surya-decode-setup", "calls": 1,
         "start_ns": 0, "end_ns": 100_000_000},
        {"name": "surya-decode", "calls": 4,
         "start_ns": 100_000_000, "end_ns": 300_000_000},
    ]


def test_attribute_buckets_rows_by_start_timestamp() -> None:
    rows = [
        {"start_ns": 50_000_000, "duration_ns": 1_000},      # setup
        {"start_ns": 100_000_000, "duration_ns": 1_000},     # range start is inclusive
        {"start_ns": 299_999_999, "duration_ns": 1_000},     # last ns of decode
        {"start_ns": 300_000_000, "duration_ns": 1_000},     # end is exclusive
        {"start_ns": 0, "duration_ns": 1_000},               # setup, first ns
    ]
    buckets, outside = R._attribute(rows, _ranges())
    assert len(buckets["surya-decode-setup"]) == 2
    assert len(buckets["surya-decode"]) == 2
    assert [row["start_ns"] for row in outside] == [300_000_000]


def test_attribute_keeps_rows_with_no_timestamp_out_of_every_range() -> None:
    """A row the reader could not timestamp must not be guessed into a phase."""

    buckets, outside = R._attribute([{"start_ns": None, "duration_ns": 1}], _ranges())
    assert all(not rows for rows in buckets.values())
    assert len(outside) == 1


def test_phase_cost_excludes_setup_kernels() -> None:
    """The error this exists to prevent: billing vision+prefill to decode.

    Setup kernels are far larger than one decode step, so a whole-process
    division reports a per-token cost that no decode step actually takes.
    """

    kernels = [
        {"kernel": "rocblas_gemm_plain", "start_ns": 10_000_000,
         "duration_ns": 90_000_000},          # 90 ms of setup
        {"kernel": "rocblas_gemm_plain", "start_ns": 120_000_000,
         "duration_ns": 40_000_000},         # 4 decode steps, 10 ms each
        {"kernel": "evie_softmax_rows_kernel", "start_ns": 350_000_000,
         "duration_ns": 5_000_000},          # outside every range
    ]
    summary = R._attribute_phases(kernels, [], 4, _ranges(), "surya-decode")
    assert summary["attribution"] == "range"
    # 40 ms / 4 calls, not 135 ms / 4 calls
    assert summary["kernel_per_call_ms"] == pytest.approx(10.0)
    assert summary["kernel_total_ms"] == pytest.approx(40.0)
    assert summary["calls_executed"] == 4
    assert summary["whole_process"]["kernel_per_call_ms"] == pytest.approx(33.75)
    setup = next(r for r in summary["ranges"] if r["name"] == "surya-decode-setup")
    assert setup["kernel_per_call_ms"] == pytest.approx(90.0)
    assert setup["calls_in_range"] == 1
    assert setup["is_phase"] is False
    decode = next(r for r in summary["ranges"] if r["name"] == "surya-decode")
    assert decode["is_phase"] is True
    assert decode["wall_ms"] == pytest.approx(200.0)
    assert decode["host_gap_ms"] == pytest.approx(160.0)
    assert summary["outside"]["kernel_total_ms"] == pytest.approx(5.0)


def test_range_uses_its_own_call_count_not_calls_executed() -> None:
    """Vision does one warmup outside its range, so calls_executed overshoots."""

    ranges = [{"name": "surya-vision", "calls": 20,
               "start_ns": 0, "end_ns": 100_000_000}]
    kernels = [{"kernel": "rocblas_gemm_plain", "start_ns": i * 4_000_000,
                "duration_ns": 1_000_000} for i in range(20)]
    summary = R._attribute_phases(kernels, [], 21, ranges, "surya-vision")
    assert summary["calls_executed"] == 21
    assert summary["calls_in_range"] == 20
    assert summary["kernel_per_call_ms"] == pytest.approx(1.0)


def test_range_without_a_call_count_falls_back_to_calls_executed() -> None:
    """Driver output predating the per-range count still attributes correctly."""

    ranges = [{"name": "surya-decode", "start_ns": 0, "end_ns": 100_000_000},
              {"name": "surya-decode-setup", "start_ns": 100_000_000,
               "end_ns": 200_000_000}]
    kernels = [{"kernel": "rocblas_gemm_plain", "start_ns": i * 1_000_000,
                "duration_ns": 1_000_000} for i in range(4)]
    summary = R._attribute_phases(kernels, [], 4, ranges, "surya-decode")
    assert summary["calls_in_range"] == 4
    assert summary["kernel_per_call_ms"] == pytest.approx(1.0)
    setup = next(r for r in summary["ranges"] if r["name"] == "surya-decode-setup")
    assert setup["calls_in_range"] == 1


def test_trace_without_ranges_is_labelled_whole_process() -> None:
    """No ranges must not be silently presented as a phase cost."""

    kernels = [{"kernel": "rocblas_gemm_plain", "start_ns": 0, "duration_ns": 1_000_000}]
    summary = R._attribute_phases(kernels, [], 1, [], "surya-decode")
    assert summary["attribution"] == "whole_process"
    assert "ranges" not in summary
    assert summary["whole_process"]["kernel_total_ms"] == pytest.approx(1.0)


# --- context span -----------------------------------------------------------


def test_context_span_reports_the_range_not_one_endpoint() -> None:
    """A per-token cost must not be read as measured at a context it never saw.

    The driver decodes 200 steps, so the KV length runs from prompt_tokens to
    prompt_tokens + steps - 1 and the mean is neither endpoint.
    """

    driver = {"phase": "decode", "decode_steps": 200, "context_start": 106,
              "context_end": 305, "pos_start": 50, "pos_end": 249}
    span = R._context_span(driver)
    assert span["context_start"] == 106
    assert span["context_end"] == 305
    assert span["mean_context"] == pytest.approx(205.5)
    # mRoPE positions are not KV lengths: image spans advance (t, h, w) over the
    # merged grid, so the position is far below the token count.
    assert span["pos_start"] == 50
    assert span["pos_end"] == 249


def test_context_span_is_absent_for_phases_without_kv_state() -> None:
    assert R._context_span({"phase": "vision"}) is None
    assert R._context_span({"phase": "prefill", "context_start": None}) is None


def test_committed_decode_phases_carry_a_context_span() -> None:
    """The artifact must say which contexts its decode rows were measured at."""

    path = (_ROOT / "benchmarks" / "results"
            / "2026-09-12-gfx1151-surya-phase-attributed-profile.json")
    if not path.exists():
        pytest.skip("profile artifact not present")
    data = json.loads(path.read_text())
    for phase in ("decode-small", "decode-full"):
        span = data["phases"][phase]["context"]
        assert span is not None, phase
        assert span["context_end"] > span["context_start"]
        assert span["mean_context"] == pytest.approx(
            (span["context_start"] + span["context_end"]) / 2)
        assert span["pos_end"] < span["context_end"], "mRoPE positions are not KV lengths"
    assert data["phases"]["vision-small"]["context"] is None


def test_copies_are_attributed_to_ranges_too(tmp_path: Path) -> None:
    """A transfer during setup is not a decode transfer."""

    copies = [
        {"kind": "host_to_device", "start_ns": 10_000_000, "bytes": None,
         "duration_ns": 2_000_000},
        {"kind": "host_to_device", "start_ns": 150_000_000, "bytes": None,
         "duration_ns": 8_000_000},
    ]
    summary = R._attribute_phases([], copies, 4, _ranges(), "surya-decode")
    assert summary["transfers"]["total_ms"] == pytest.approx(8.0)
    assert summary["transfers"]["ms_per_call"] == pytest.approx(2.0)
    setup = next(r for r in summary["ranges"] if r["name"] == "surya-decode-setup")
    assert setup["transfers"]["total_ms"] == pytest.approx(2.0)


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
