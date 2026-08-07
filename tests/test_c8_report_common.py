"""CPU tests for C8 retained-report schema and provenance helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.c8_report_common import (
    REPORT_SCHEMA,
    dependency_adjusted_bytes,
    implementation_files,
    route_timing_result,
)


_LIBRARIES = {
    "libcudart_so13": 10,
    "libcublasLt_so13": 20,
    "libcublas_so13": 30,
    "cudnn_payload_total": 40,
}


def test_route_timing_schema_uses_explicit_consistent_units() -> None:
    row = route_timing_result(
        [0.003, 0.001, 0.002],
        batch=2,
        c1_seq_total_s=0.004,
    )

    assert REPORT_SCHEMA == 2
    assert row == {
        "c1_seq_total_s": 0.004,
        "route_wall_median_s": 0.002,
        "batch_median_ms": 2.0,
        "batch_req_per_s": 1000.0,
        "vs_c1_seq_req_per_s": 2.0,
        "route_wall_s_raw": [0.003, 0.001, 0.002],
        "p50_ms": 2.0,
        "p95_ms": 3.0,
        "sample_count": 3,
    }


@pytest.mark.parametrize(
    ("route", "route_only", "route_total", "total"),
    (
        ("custom_cuda_runtime_subset", {}, 0, 10),
        (
            "c8_batch_encoder_cublaslt_route",
            {"libcublasLt_so13": 20, "libcublas_so13": 30},
            50,
            60,
        ),
        (
            "c8_batch_encoder_cudnn_route",
            {"cudnn_payload_total": 40},
            40,
            50,
        ),
    ),
)
def test_dependency_totals_are_route_specific(
    route: str,
    route_only: dict[str, int],
    route_total: int,
    total: int,
) -> None:
    report = dependency_adjusted_bytes(route, libraries=_LIBRARIES)

    assert report["scope"] == "system_cuda_runtime_libraries_only"
    assert report["base"] == {"libcudart_so13": 10}
    assert report["base_total_bytes"] == 10
    assert report["route_only"] == route_only
    assert report["route_only_total_bytes"] == route_total
    assert report["total_bytes"] == total
    assert report["host_library_inventory_bytes"] == _LIBRARIES


def test_dependency_totals_reject_unknown_route() -> None:
    with pytest.raises(ValueError, match="unsupported C8 dependency route"):
        dependency_adjusted_bytes("all-installed-libraries", libraries=_LIBRARIES)


def test_implementation_manifest_includes_benchmark_and_report_sources() -> None:
    repo = Path(__file__).resolve().parents[1]
    files = implementation_files(
        repo,
        report_sources=("scripts/benchmark_c8_batch_throughput.py",),
    )

    assert "scripts/c8_report_common.py" in files
    assert "scripts/benchmark_c8_batch_throughput.py" in files
    assert "hipengine/runtime/moonshine_cuda_batch.py" in files


def test_implementation_manifest_rejects_escape() -> None:
    repo = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="escapes repository"):
        implementation_files(repo, report_sources=("../outside.py",))
