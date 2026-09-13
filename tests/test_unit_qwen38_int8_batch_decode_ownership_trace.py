"""Deterministic checks for the IKV-C2 ownership-trace generator.

``scripts/qwen38_int8_batch_decode_ownership_trace.py`` is the committed
generator for the IKV-C2 primitive-gate ``rocprofv3`` evidence. Its whole
purpose is to prove the packed consumer is genuinely row-batched: the batch
producer and reducer launch once for all rows while the c1 leaf launches once
per row. If the reduction silently accepted a per-row batch path, the artifact
would certify the opposite of what it claims, so the failure modes are pinned
here rather than left to the GPU run.

The GPU trace itself is not exercised (no ROCm needed); these tests cover
classification, CSV reduction, the summary arithmetic against the retained
measured durations, and the case table's agreement with the primitive gate.
"""

from __future__ import annotations

import csv
import ast
import json
import re
from pathlib import Path

import pytest

from scripts.qwen38_int8_batch_decode_ownership_trace import (
    COMPILER_VERSION_FILE_ENV,
    GATE_CASES,
    GATE_TEST,
    REQUIRE_CACHED_BUILD_ENV,
    _trace_environment,
    classify_kernel,
    read_launches,
    summarize_ownership,
)

from hipengine.core import build as _hipengine_build

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_TEST_PATH = REPO_ROOT / "tests" / "test_gpu_qwen38_int8_batch_attention_gpu.py"
RETAINED_ARTIFACT = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "2026-09-11-w7900-ikv-c2-batch-decode-ownership-trace.json"
)

_BATCH_PRODUCER = (
    "void (anonymous namespace)::"
    "qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_batch_kernel"
    "<float, 6l, 24l, 4l>(float const*, signed char const*)"
)
_BATCH_REDUCER = (
    "void (anonymous namespace)::"
    "qwen35_paged_full_attn_decode_split_k_reduce_gate_batch_strided_kernel"
    "<hip_bfloat16>(float const*, float const*)"
)
_C1_PRODUCER = (
    "void (anonymous namespace)::"
    "qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_kernel"
    "<float, 6l, 24l, 4l>(float const*, signed char const*)"
)
_C1_REDUCER = (
    "void (anonymous namespace)::"
    "qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel"
    "<hip_bfloat16>(float const*, float const*)"
)

_CSV_COLUMNS = (
    "Kind",
    "Kernel_Name",
    "Start_Timestamp",
    "End_Timestamp",
    "Grid_Size_X",
    "Grid_Size_Y",
    "Grid_Size_Z",
    "Workgroup_Size_X",
    "Workgroup_Size_Y",
    "Workgroup_Size_Z",
    "VGPR_Count",
)


def _launch(kind: str, duration_ns: int, *, grid_z: int, vgpr: int = 64) -> dict:
    return {
        "kind": kind,
        "kernel_name": f"kernel::{kind}",
        "duration_ns": int(duration_ns),
        "grid": ["1024", "5", str(grid_z)],
        "workgroup": ["256", "1", "1"],
        "vgpr_count": int(vgpr),
    }


def _retained_launches() -> list[dict]:
    """The measured dispatch set from the retained W7900 trace."""

    return [
        _launch("batch_producer", 73175, grid_z=4),
        _launch("batch_reducer", 2387, grid_z=4, vgpr=16),
        _launch("c1_producer", 12694, grid_z=1),
        _launch("c1_reducer", 6200, grid_z=1, vgpr=16),
        _launch("c1_producer", 71986, grid_z=1),
        _launch("c1_reducer", 4880, grid_z=1, vgpr=16),
        _launch("c1_producer", 72627, grid_z=1),
        _launch("c1_reducer", 4867, grid_z=1, vgpr=16),
        _launch("c1_producer", 67694, grid_z=1),
        _launch("c1_reducer", 4973, grid_z=1, vgpr=16),
    ]


@pytest.mark.parametrize(
    ("kernel_name", "expected"),
    (
        (_BATCH_PRODUCER, "batch_producer"),
        (_BATCH_REDUCER, "batch_reducer"),
        (_C1_PRODUCER, "c1_producer"),
        (_C1_REDUCER, "c1_reducer"),
    ),
)
def test_classify_kernel_recognizes_each_ownership_kind(
    kernel_name: str, expected: str
) -> None:
    assert classify_kernel(kernel_name) == expected


def test_batch_producer_is_not_misclassified_as_the_c1_leaf() -> None:
    """The batch name embeds ``_int8_batch_kernel``; ordering must win."""

    assert classify_kernel(_BATCH_PRODUCER) == "batch_producer"
    assert classify_kernel(_BATCH_REDUCER) == "batch_reducer"


@pytest.mark.parametrize(
    "kernel_name",
    (
        "__amd_rocclr_copyBuffer",
        "void at::native::elementwise_kernel<128, 4>",
        "",
    ),
)
def test_classify_kernel_ignores_unrelated_dispatches(kernel_name: str) -> None:
    assert classify_kernel(kernel_name) is None


def test_read_launches_reduces_a_kernel_trace_csv(tmp_path: Path) -> None:
    path = tmp_path / "trace_kernel_trace.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "Kind": "KERNEL_DISPATCH",
                "Kernel_Name": "__amd_rocclr_copyBuffer",
                "Start_Timestamp": "100",
                "End_Timestamp": "160",
                "Grid_Size_X": "512",
                "Grid_Size_Y": "1",
                "Grid_Size_Z": "1",
                "Workgroup_Size_X": "512",
                "Workgroup_Size_Y": "1",
                "Workgroup_Size_Z": "1",
                "VGPR_Count": "16",
            }
        )
        writer.writerow(
            {
                "Kind": "KERNEL_DISPATCH",
                "Kernel_Name": _BATCH_PRODUCER,
                "Start_Timestamp": "1000",
                "End_Timestamp": "1735",
                "Grid_Size_X": "1024",
                "Grid_Size_Y": "5",
                "Grid_Size_Z": "4",
                "Workgroup_Size_X": "256",
                "Workgroup_Size_Y": "1",
                "Workgroup_Size_Z": "1",
                "VGPR_Count": "64",
            }
        )

    launches = read_launches(path)

    assert len(launches) == 1
    launch = launches[0]
    assert launch["kind"] == "batch_producer"
    assert launch["duration_ns"] == 735
    assert launch["grid"] == ["1024", "5", "4"]
    assert launch["workgroup"] == ["256", "1", "1"]
    assert launch["vgpr_count"] == 64


def test_read_launches_rejects_inverted_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "trace_kernel_trace.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "Kind": "KERNEL_DISPATCH",
                "Kernel_Name": _BATCH_PRODUCER,
                "Start_Timestamp": "2000",
                "End_Timestamp": "1000",
                "Grid_Size_X": "1024",
                "Grid_Size_Y": "5",
                "Grid_Size_Z": "4",
                "Workgroup_Size_X": "256",
                "Workgroup_Size_Y": "1",
                "Workgroup_Size_Z": "1",
                "VGPR_Count": "64",
            }
        )

    with pytest.raises(ValueError, match="inverted timestamp"):
        read_launches(path)


def test_read_launches_rejects_a_trace_without_ikv_dispatches(tmp_path: Path) -> None:
    path = tmp_path / "trace_kernel_trace.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "Kind": "KERNEL_DISPATCH",
                "Kernel_Name": "__amd_rocclr_copyBuffer",
                "Start_Timestamp": "100",
                "End_Timestamp": "160",
                "Grid_Size_X": "512",
                "Grid_Size_Y": "1",
                "Grid_Size_Z": "1",
                "Workgroup_Size_X": "512",
                "Workgroup_Size_Y": "1",
                "Workgroup_Size_Z": "1",
                "VGPR_Count": "16",
            }
        )

    with pytest.raises(ValueError, match="no IKV-C2 dispatches"):
        read_launches(path)


def test_summary_reproduces_the_retained_measured_blocks() -> None:
    ownership, kernel_time_ns = summarize_ownership(_retained_launches(), 4)

    assert ownership == {
        "batch_launch_count": 2,
        "c1_leaf_launch_count": 8,
        "launches_per_kind": {
            "batch_producer": 1,
            "batch_reducer": 1,
            "c1_producer": 4,
            "c1_reducer": 4,
        },
        "single_batch_launch_covers_all_rows": True,
        "per_row_serial_c1_launches": 4,
    }
    assert kernel_time_ns == {
        "batch_path_total": 75562,
        "serial_c1_path_total": 245921,
        "ratio_serial_over_batch": 3.255,
        "batch_producer": 73175,
        "batch_reducer": 2387,
        "c1_producer": 225001,
        "c1_reducer": 20920,
    }


def test_summary_rejects_a_per_row_batch_path() -> None:
    """A batch producer that launched per row is not row-batched evidence."""

    launches = _retained_launches()
    launches.append(_launch("batch_producer", 18000, grid_z=4))

    with pytest.raises(ValueError, match="must each launch exactly once"):
        summarize_ownership(launches, 4)


def test_summary_rejects_a_missing_c1_leaf_row() -> None:
    launches = [item for item in _retained_launches() if item["duration_ns"] != 4973]

    with pytest.raises(ValueError, match="once per row"):
        summarize_ownership(launches, 4)


def test_summary_rejects_a_trace_missing_the_packed_path() -> None:
    launches = [
        item for item in _retained_launches() if not item["kind"].startswith("batch_")
    ]

    with pytest.raises(ValueError, match="both the packed path and the independent c1"):
        summarize_ownership(launches, 4)


def test_summary_rejects_c1_only_widths() -> None:
    with pytest.raises(ValueError, match="only meaningful above c1"):
        summarize_ownership(_retained_launches(), 1)


def test_generator_cases_match_the_primitive_gate_parametrization() -> None:
    """The artifact's rows/live_counts must not drift from the gate it traces."""

    tree = ast.parse(GATE_TEST_PATH.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_qwen38_int8_batch_attention_matches_cpu_and_independent_c1"
    )
    parametrize = [
        mark for mark in function.decorator_list
        if isinstance(mark, ast.Call)
        and isinstance(mark.func, ast.Attribute)
        and mark.func.attr == "parametrize"
    ]
    assert len(parametrize) == 1
    argnames, argvalues = map(ast.literal_eval, parametrize[0].args)
    assert tuple(argnames) == ("rows", "live_counts")
    ids = tuple(ast.literal_eval(
        next(keyword.value for keyword in parametrize[0].keywords if keyword.arg == "ids")
    ))
    assert len(ids) == len(argvalues)

    gate_cases = {
        case_id: (int(rows), tuple(int(count) for count in counts))
        for case_id, (rows, counts) in zip(ids, argvalues, strict=True)
    }
    assert gate_cases == dict(GATE_CASES)


def test_gate_node_id_names_the_traced_case() -> None:
    assert GATE_TEST.endswith(
        "::test_qwen38_int8_batch_attention_matches_cpu_and_independent_c1"
    )
    assert "c4-ragged" in GATE_CASES


def test_retained_artifact_is_consistent_with_its_own_summary() -> None:
    """Re-deriving the blocks from the committed launches must agree."""

    artifact = json.loads(RETAINED_ARTIFACT.read_text(encoding="utf-8"))
    rows = int(artifact["workload"]["rows"])
    ownership, kernel_time_ns = summarize_ownership(artifact["launches"], rows)

    assert ownership == artifact["ownership"]
    assert kernel_time_ns == artifact["kernel_time_ns"]
    assert artifact["kind"] == "qwen38_int8_row_batched_decode_ownership_trace"
    assert artifact["performance_claim"] is False


def test_retained_artifact_preserves_trace_provenance() -> None:
    """A regeneration must not erase how the trace was produced."""

    artifact = json.loads(RETAINED_ARTIFACT.read_text(encoding="utf-8"))

    assert "rocprofv3 --kernel-trace" in artifact["command"]
    assert "c4-ragged" in artifact["command"]
    assert artifact["reduction"] in {
        "profiled rocprofv3 run",
    } or artifact["reduction"].startswith("parse-only")


def test_trace_environment_uses_the_names_the_build_module_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lookalike env name silently disables the guard and the artifact lies.

    ``HIPENGINE_HIP_REQUIRE_CACHED_BUILD`` reads like the real switch, does
    nothing, and leaves the profiled process free to spawn hipcc while the
    artifact still claims it was cache-only.
    """

    version_file = tmp_path / "hipcc-version.txt"
    version_file.write_text("AMD clang version 22.0.0git\n", encoding="utf-8")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")

    environment = _trace_environment(version_file, "hip_gfx1100")

    assert REQUIRE_CACHED_BUILD_ENV == _hipengine_build._ENV_REQUIRE_CACHED_BUILD
    assert environment[REQUIRE_CACHED_BUILD_ENV] == "1"
    assert environment[COMPILER_VERSION_FILE_ENV] == str(version_file)
    assert "HIPENGINE_HIP_REQUIRE_CACHED_BUILD" not in environment
    assert "ROCR_VISIBLE_DEVICES" not in environment


@pytest.mark.parametrize("flag", ["1", "true", "yes", "on"])
def test_require_cached_guard_is_actually_activated(
    flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the real predicate, not just the name."""

    monkeypatch.setenv(_hipengine_build._ENV_REQUIRE_CACHED_BUILD, flag)
    assert _hipengine_build._environment_requires_cached_build() is True


def test_compiler_version_file_env_name_is_read_by_the_build_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The version-file name must be one the cache key actually consults."""

    version_file = tmp_path / "hipcc-version.txt"
    version_file.write_text("AMD clang version 22.0.0git\n", encoding="utf-8")
    monkeypatch.setenv("HIPENGINE_COMPILER_VERSION_TEXT", "")
    monkeypatch.setenv(COMPILER_VERSION_FILE_ENV, str(version_file))

    identity = _hipengine_build._environment_version_identity("hipcc")

    # The identity carries the raw override values, so the name is proven by
    # finding this path in it; the resolved value then proves the file is read.
    assert str(version_file) in identity
    assert (
        _hipengine_build._compiler_version_from_environment("hipcc")
        == "AMD clang version 22.0.0git"
    )


def test_trace_environment_rejects_an_empty_compiler_version_file(
    tmp_path: Path,
) -> None:
    version_file = tmp_path / "hipcc-version.txt"
    version_file.write_text("   \n", encoding="utf-8")

    with pytest.raises(ValueError, match="compiler version file is empty"):
        _trace_environment(version_file, "hip_gfx1100")


def test_retained_artifact_makes_its_cache_only_claim_checkable() -> None:
    """The guard must be recorded, and recorded under the name that works.

    The artifact this replaces asserted a cache-only run while its recorded
    command set ``HIPENGINE_HIP_REQUIRE_CACHED_BUILD``, a name no module reads.
    """

    artifact = json.loads(RETAINED_ARTIFACT.read_text(encoding="utf-8"))
    environment = artifact["trace_environment"]

    assert environment[REQUIRE_CACHED_BUILD_ENV] == "1"
    assert COMPILER_VERSION_FILE_ENV in environment
    assert "HIPENGINE_HIP_REQUIRE_CACHED_BUILD" not in environment
    assert artifact["reduction"] == "profiled rocprofv3 run"
    assert any(REQUIRE_CACHED_BUILD_ENV in note for note in artifact["notes"])
    assert all(
        "HIPENGINE_HIP_REQUIRE_CACHED_BUILD" not in note for note in artifact["notes"]
    )


def test_no_script_uses_a_lookalike_cache_only_switch() -> None:
    """A near-miss name reads as correct and silently does nothing.

    The lookalike switch is the shape of mistake that matters: it reads as the
    real guard, so a profiled run looks cache-only while remaining free to spawn
    hipcc and corrupt the trace. Only quoted use is flagged, so prose that names
    the trap to explain it stays allowed.
    """

    pattern = re.compile(r"""["']HIPENGINE_HIP_REQUIRE_CACHED_BUILD["']""")
    offenders = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "scripts").rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    )

    assert offenders == []


def test_retained_artifact_records_the_promoted_admitted_width() -> None:
    """A stale capability snapshot is what the generator exists to prevent."""

    artifact = json.loads(RETAINED_ARTIFACT.read_text(encoding="utf-8"))
    capability = artifact["capability"]

    assert capability["admitted_max_direct_rows"] == 4
    assert (
        capability["decode_batch_variant"]
        == "per_token_head_gqa_splitk_gate_bf16_batch_strided_spans"
    )
