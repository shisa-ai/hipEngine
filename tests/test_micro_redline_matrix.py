from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO_ROOT / "benchmarks" / "micro" / "redline_matrix.py"
    spec = importlib.util.spec_from_file_location("micro_redline_matrix", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _checkout(tmp_path: Path) -> tuple[Path, str]:
    checkout = tmp_path / "redline"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    (checkout / "README.md").write_text("redline fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "fixture"], check=True)
    return checkout, _git(checkout, "rev-parse", "HEAD")


def _stats(value: float) -> dict:
    return {
        "samples": 3,
        "n": 3,
        "median": value,
        "p05": value,
        "p95": value,
        "min": value,
        "max": value,
        "stdev": 0.0,
    }


def _metric(clock: str, value: float, iterations: int) -> dict:
    return {
        "status": "ok",
        "clock": clock,
        "sequence_us": _stats(value * iterations),
        "per_iteration_us": _stats(value),
    }


def _row(backend: str, value: float, *, mode: str = "independent_throughput") -> dict:
    ordering = {
        "hip": "none",
        "vulkan": "none",
        "redline": "none",
    }[backend]
    clock = {
        "hip": "hip_event",
        "vulkan": "vulkan_timestamp",
        "redline": "redline_pm4_timestamp",
    }[backend]
    strategy = {
        "hip": "multi_stream",
        "vulkan": "vulkan_command_buffer",
        "redline": "retained_pm4_ib",
    }[backend]
    return {
        "k": 512,
        "rows": 1,
        "workgroup_size": 64,
        "body_repeats": 32,
        "correctness_pass": True,
        "timing_mode": mode,
        "dependency_contract": {
            "work_dependency": "independent",
            "inter_dispatch_ordering": ordering,
            "output_partitioning": "disjoint",
            "validation_status": "pass",
        },
        "submission": {
            "strategy": strategy,
            "recording_in_timed_region": False,
            "submit_in_host_wall": True,
            "completion_in_host_wall": True,
            "queue_or_stream_count": 2 if backend != "vulkan" else 1,
        },
        "timing": {
            "single": {
                "logical_iterations": 1,
                "dispatches_per_iteration": 1,
                "gpu_elapsed": _metric(clock, value + 1.0, 1),
                "host_wall": _metric("steady_clock", value + 3.0, 1),
            },
            "burst": {
                "logical_iterations": 8,
                "dispatches_per_iteration": 1,
                "gpu_elapsed": _metric(clock, value, 8),
                "host_wall": _metric("steady_clock", value + 2.0, 8),
            },
        },
        "correctness": {
            "single_dispatch": {"status": "pass", "oracle": "CPU"},
            "timed_sequence": {
                "status": "pass",
                "oracle": "CPU all outputs",
                "logical_iterations": 8,
                "coverage": "all_dispatches",
            },
            "synchronization": {
                "status": "pass",
                "method": "disjoint outputs",
                "barrier_count": 0,
            },
        },
    }


def _result(backend: str, value: float) -> dict:
    return {
        "schema_version": 2,
        "kind": "hipengine_micro_result",
        "bench": "f32_gemv_geometry_sweep",
        "backend": backend,
        "hardware": {"gpu_name": "AMD Radeon Pro W7900", "gfx_arch": "gfx1100"},
        "source": {
            "repo": str(REPO_ROOT),
            "branch": "redline-integration-spike",
            "commit": "a" * 40,
            "dirty": False,
            "source_hash": f"sha256:{backend}",
        },
        "command": ["python3", "geometry_sweep.py"],
        "parameters": {"timing_mode": "independent_throughput"},
        "correctness": {"status": "pass"},
        "measurements": {"rows": [_row(backend, value)]},
        "classification": "geometry",
        "environment_ref": "/tmp/environment.json",
    }


def test_validate_redline_checkout_requires_exact_clean_commit(tmp_path: Path) -> None:
    module = _load_module()
    checkout, commit = _checkout(tmp_path)

    evidence = module.validate_redline_checkout(checkout, expected_commit=commit)
    assert evidence["commit"] == commit
    assert evidence["dirty"] is False

    (checkout / "README.md").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        module.validate_redline_checkout(checkout, expected_commit=commit)
    with pytest.raises(ValueError, match="commit"):
        module.validate_redline_checkout(
            checkout, expected_commit="0" * 40, allow_dirty=True
        )


def test_canonicalize_joint_runner_result_preserves_backend_rows() -> None:
    module = _load_module()
    hip_row = _row("hip", 10.0)
    hip_row["backend"] = "hip"
    vulkan_row = _row("vulkan", 8.0)
    vulkan_row["backend"] = "vulkan"
    joint = {
        "schema_version": 2,
        "kind": "hipengine_micro_comparison",
        "bench": "reduction_sweep",
        "classification": "diagnostic_unclassified",
        "hardware": {
            "hip": {"gpu_name": "AMD Radeon Pro W7900", "gfx_arch": "gfx1100"},
            "vulkan": {"gpu_name": "AMD Radeon Pro W7900", "gfx_arch": "gfx1100"},
        },
        "source": {
            "repo": str(REPO_ROOT),
            "branch": "redline-integration-spike",
            "commit": "a" * 40,
            "dirty": False,
            "source_hash": "sha256:joint",
        },
        "command": ["python3", "reduction_sweep.py"],
        "config": {"timing_mode": "independent_throughput", "reps": 8},
        "correctness": {
            "hip": {"status": "pass", "row_count": 1},
            "vulkan": {"status": "pass", "row_count": 1},
        },
        "environment": {"ref": "/tmp/environment.json", "captured": None},
        "rows": [hip_row, vulkan_row],
        "comparisons": [],
        "inputs": {},
    }

    canonical = module.canonicalize_backend_result(joint, backend="hip")

    assert canonical["kind"] == "hipengine_micro_result"
    assert canonical["backend"] == "hip"
    assert canonical["measurements"]["rows"] == [hip_row]
    assert canonical["correctness"]["status"] == "pass"
    assert canonical["environment_ref"] == "/tmp/environment.json"
    assert canonical["runner_view"] == {
        "kind": "hipengine_micro_comparison",
        "selected_backend": "hip",
        "selected_rows": 1,
        "total_rows": 2,
        "timing_samples_recomputed": False,
    }


def test_normalize_redline_result_records_direct_pm4_proof(tmp_path: Path) -> None:
    module = _load_module()
    library = tmp_path / "libredline_dispatch.so"
    adapter = tmp_path / "redline_hip_timing.py"
    sidecar = tmp_path / "geometry.redline.co"
    manifest = tmp_path / "geometry.redline.radiowave.json"
    library.write_bytes(b"library")
    adapter.write_text("# adapter\n", encoding="utf-8")
    sidecar.write_bytes(b"hsaco")
    manifest.write_text('{"schema_version": 3}\n', encoding="utf-8")
    result = _result("hip", 7.0)

    normalized = module.normalize_redline_result(
        result,
        redline_evidence={
            "root": "/external/redline",
            "commit": module.PINNED_REDLINE_COMMIT,
            "dirty": False,
        },
        library_path=library,
        adapter_paths=[adapter],
        sidecar_paths=[sidecar, manifest],
    )

    assert normalized["backend"] == "redline"
    assert normalized["measurements"]["rows"][0]["submission"]["strategy"] == "retained_pm4_ib"
    row = normalized["measurements"]["rows"][0]
    assert row["timing"]["burst"]["gpu_elapsed"]["clock"] == "redline_pm4_timestamp"
    assert row["timing"]["single"]["retained_lane_count"] == 1
    assert row["timing"]["burst"]["retained_lane_count"] == 2
    proof = normalized["redline_provenance"]["execution_proof"]
    assert proof == {
        "api": "redline-capi",
        "native_hip_fallback_available": False,
        "profiled_retained_pm4_required": True,
        "radiowave_manifest_verified": True,
    }
    assert normalized["redline_provenance"]["library_sha256"].startswith("sha256:")
    assert normalized["redline_provenance"]["sidecars"][0]["sha256"].startswith("sha256:")
    json.dumps(normalized, allow_nan=False)


def test_three_backend_comparison_reports_all_pairs_and_transport_scope() -> None:
    module = _load_module()
    results = {
        "hip": _result("hip", 10.0),
        "vulkan": _result("vulkan", 8.0),
        "redline": _result("redline", 6.0),
    }
    results["redline"]["redline_provenance"] = {
        "execution_proof": {
            "api": "redline-capi",
            "native_hip_fallback_available": False,
            "profiled_retained_pm4_required": True,
            "radiowave_manifest_verified": True,
        },
        "checkout": {"commit": module.PINNED_REDLINE_COMMIT, "dirty": False},
    }

    comparison = module.build_three_backend_comparison(
        results,
        family="geometry",
        command=["python3", "redline_matrix.py"],
        input_refs={backend: f"/tmp/{backend}.json" for backend in results},
        same_hsaco_control=None,
    )

    assert comparison["kind"] == "hipengine_micro_comparison"
    assert comparison["performance_claim"] is True
    assert comparison["transport_attribution"]["status"] == "blocked_no_same_hsaco_control"
    assert comparison["summary"]["redline_first_rows"] == 1
    burst_gpu = [
        item
        for item in comparison["comparisons"]
        if item["control"] == "burst" and item["gpu_elapsed"]["status"] == "ok"
    ]
    assert {item["pair"] for item in burst_gpu} == {
        "redline_vs_hip",
        "redline_vs_vulkan",
        "vulkan_vs_hip",
    }
    redline_hip = next(item for item in burst_gpu if item["pair"] == "redline_vs_hip")
    assert redline_hip["gpu_elapsed"]["lhs_us_per_iteration"] == 6.0
    assert redline_hip["gpu_elapsed"]["rhs_us_per_iteration"] == 10.0
    assert redline_hip["gpu_elapsed"]["lhs_vs_rhs_speedup"] == pytest.approx(10.0 / 6.0)


def test_three_backend_comparison_rejects_mismatched_independent_hip_redline_lanes() -> None:
    module = _load_module()
    results = {
        "hip": _result("hip", 10.0),
        "vulkan": _result("vulkan", 8.0),
        "redline": _result("redline", 6.0),
    }
    results["redline"]["measurements"]["rows"][0]["submission"][
        "queue_or_stream_count"
    ] = 1
    results["redline"]["redline_provenance"] = {
        "execution_proof": {
            "api": "redline-capi",
            "native_hip_fallback_available": False,
            "profiled_retained_pm4_required": True,
            "radiowave_manifest_verified": True,
        },
        "checkout": {"commit": module.PINNED_REDLINE_COMMIT, "dirty": False},
    }

    with pytest.raises(ValueError, match="lane counts"):
        module.build_three_backend_comparison(
            results,
            family="geometry",
            command=["python3", "redline_matrix.py"],
            input_refs={backend: f"/tmp/{backend}.json" for backend in results},
        )


def test_tri_comparator_matches_native_dispatch_count_aliases() -> None:
    module = _load_module()

    hip = {
        "sweep": "grid",
        "node_count": 941,
        "grid_blocks": 8192,
    }
    redline = {
        "sweep": "grid",
        "dispatch_count": 941,
        "grid_blocks": 8192,
    }

    assert module._row_key("dispatch", hip) == module._row_key("dispatch", redline)


def test_family_command_uses_matched_independent_lane_count(tmp_path: Path) -> None:
    module = _load_module()
    command = module.build_family_command(
        runner=REPO_ROOT / "benchmarks/micro/runners/geometry_sweep.py",
        backend="hip",
        mode="independent_throughput",
        output=tmp_path / "hip.json",
        environment=tmp_path / "environment.json",
        build_dir=tmp_path / "build",
        gfx_arch="gfx1100",
        gpu_name="AMD Radeon Pro W7900",
        device_index=0,
        independent_lanes=2,
        reps=20,
        warmup=5,
        samples=7,
        family_args=["--k-list", "512"],
    )

    assert command[command.index("--independent-streams") + 1] == "2"
    assert command[command.index("--backend") + 1] == "hip"
