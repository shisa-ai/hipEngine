#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
# Derived from warpfront/redline@33683f3 examples/hipengine-6409 orchestration.
"""Run and compare HIP, Vulkan, and pinned Redline microbenchmark arms.

Redline is consumed from a separate, clean source checkout. Its hipEngine adapter
captures unchanged HIP launch closures for argument introspection, then the timed
path uses the profiled ``redline-capi`` retained-PM4 API. Native HIP and Vulkan
runner implementations are not modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
MICRO_ROOT = REPO_ROOT / "benchmarks" / "micro"
RUNNER_ROOT = MICRO_ROOT / "runners"
COLLECT_ENV = MICRO_ROOT / "collect_env.py"
TIMING_CONTRACT = MICRO_ROOT / "timing_contract.py"
PINNED_REDLINE_COMMIT = "33683f3d4f302a6c56bcc7a4c33ab8be3262dd2e"
BACKENDS = ("hip", "vulkan", "redline")
MODES = ("serial_latency", "independent_throughput")

# Keep this matrix aligned with benchmarks/micro/README.md. Dispatch/grid is a
# separately implemented transport-floor control because its HIP arm already
# contains an inner graph and cannot be captured by the timer adapter.
FAMILIES: dict[str, dict[str, Any]] = {
    "geometry": {
        "runner": "geometry_sweep.py",
        "args": [
            "--k-list", "512,2048", "--rows-list", "1,4",
            "--workgroups", "64,256", "--body-repeats", "32",
        ],
    },
    "reduction": {
        "runner": "reduction_sweep.py",
        "args": [
            "--k-list", "512,2048", "--rows-list", "1",
            "--workgroups", "64,256", "--body-repeats", "32",
        ],
    },
    "memory-waitcnt": {
        "runner": "memory_waitcnt.py",
        "args": [
            "--variants", "coalesced:4,strided:4,gather:1,interleave:4",
            "--n", "32768", "--body-iters", "64", "--workgroups", "64,256",
        ],
    },
    "packed-dot": {
        "runner": "dot_path.py",
        "args": [
            "--variants", "q8_signed:16,q4_unsigned:16,q6_zero:16,scalar_dequant:16",
            "--n", "32768", "--body-iters", "64", "--workgroups", "64,256",
        ],
    },
    "vopd": {
        "runner": "vopd_sweep.py",
        "args": [
            "--variants", "independent_fma:4,dependent_fma:4,mixed_int_float:4,dequant_like:4",
            "--n", "65536", "--body-iters", "512", "--workgroups", "64,256",
        ],
    },
    "sampler": {
        "runner": "sampler_argmax.py",
        "args": [
            "--rows-list", "1,4,8", "--workgroups", "64,256",
            "--top-k-list", "1,8", "--vocab", "32768",
        ],
    },
    "two-stage-reduction": {
        "runner": "two_stage_reduction.py",
        "args": [
            "--k-list", "8192,32768", "--rows-list", "1,4",
            "--workgroups", "128,256", "--split-counts", "2,4",
            "--body-repeats", "16",
        ],
    },
    "q4-selected-dual": {
        "runner": "q4_selected_dual_real_slice.py",
        "production_python": True,
        "args": [
            "--x-rows", "4", "--rows", "32", "--experts", "256",
            "--in-features", "2048", "--out-features", "512",
            "--workgroups", "64,128",
        ],
    },
    "q6-x8-selected-down": {
        "runner": "q6_x8_real_slice.py",
        "production_python": True,
        "args": [
            "--rows", "8", "--experts", "256", "--in-features", "512",
            "--out-features", "2048", "--local-size", "64",
        ],
    },
    "dense-q8": {
        "runner": "q8_0_dense_real_slice.py",
        "production_python": True,
        "args": [
            "--shapes", "768x2048,2048x2048", "--rows-list", "1,4",
            "--local-sizes", "32,64,128", "--row-tiles", "1,4",
        ],
    },
}

ROW_KEYS: dict[str, tuple[str, ...]] = {
    "geometry": ("k", "rows", "workgroup_size", "body_repeats"),
    "reduction": ("variant", "k", "rows", "workgroup_size", "body_repeats"),
    "memory-waitcnt": ("mode", "param", "n", "block_size", "body_iters"),
    "packed-dot": ("mode", "groups", "n", "block_size", "body_iters"),
    "vopd": ("mode", "accums", "n", "block_size", "body_iters"),
    "sampler": ("rows", "vocab", "workgroup_size", "top_k"),
    "two-stage-reduction": (
        "variant", "k", "rows", "workgroup_size", "split_count", "body_repeats",
    ),
    "q4-selected-dual": (
        "operation", "x_rows", "rows", "experts", "in_features", "out_features", "local_size",
    ),
    "q6-x8-selected-down": (
        "operation", "rows", "experts", "in_features", "out_features", "local_size", "row_tile",
    ),
    "dense-q8": (
        "operation", "rows", "in_features", "out_features", "local_size", "row_tile",
    ),
}


def _load_timing_contract():
    spec = importlib.util.spec_from_file_location("micro_redline_timing_contract", TIMING_CONTRACT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load timing contract: {TIMING_CONTRACT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing_contract = _load_timing_contract()


def _git(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(args)} failed for {checkout}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _git_optional(checkout: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def validate_redline_checkout(
    checkout: Path,
    *,
    expected_commit: str = PINNED_REDLINE_COMMIT,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Fail closed unless ``checkout`` is the explicitly pinned Redline source."""

    checkout = checkout.expanduser().resolve()
    if not (checkout / ".git").exists():
        raise ValueError(f"Redline checkout is not a Git repository: {checkout}")
    commit = _git(checkout, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise ValueError(
            f"Redline checkout commit {commit} does not match required commit {expected_commit}"
        )
    status = _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise ValueError("Redline checkout must be clean")
    return {
        "root": str(checkout),
        "commit": commit,
        "dirty": dirty,
        "status_porcelain": status.splitlines(),
        "remote": _git_optional(checkout, "config", "--get", "remote.origin.url"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"required Redline artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _recursive_redline_contract(value: Any, *, mode: str) -> None:
    if isinstance(value, list):
        for item in value:
            _recursive_redline_contract(item, mode=mode)
        return
    if not isinstance(value, dict):
        return
    if value.get("backend") == "hip":
        value["backend"] = "redline"
    submission = value.get("submission")
    if isinstance(submission, dict):
        submission.update(
            {
                "strategy": "retained_pm4_ib",
                "recording_in_timed_region": False,
                "submit_in_host_wall": True,
                "completion_in_host_wall": True,
            }
        )
    timing = value.get("timing")
    if isinstance(timing, dict):
        observed_lanes = 1
        if isinstance(submission, dict):
            try:
                observed_lanes = max(
                    1, int(submission.get("queue_or_stream_count", 1))
                )
            except (TypeError, ValueError):
                observed_lanes = 1
        for control in timing_contract.TIMING_CONTROLS:
            record = timing.get(control)
            if isinstance(record, dict):
                try:
                    logical_iterations = max(
                        1, int(record.get("logical_iterations", 1))
                    )
                except (TypeError, ValueError):
                    logical_iterations = 1
                record["retained_lane_count"] = (
                    1
                    if mode == "serial_latency"
                    else min(observed_lanes, logical_iterations)
                )
                metric = record.get("gpu_elapsed")
                if isinstance(metric, dict):
                    metric["clock"] = "redline_pm4_timestamp"
    dependency = value.get("dependency_contract")
    if isinstance(dependency, dict):
        dependency["inter_dispatch_ordering"] = (
            "redline_rmw" if mode == "serial_latency" else "none"
        )
    correctness = value.get("correctness")
    if isinstance(correctness, dict):
        synchronization = correctness.get("synchronization")
        if isinstance(synchronization, dict):
            synchronization["method"] = (
                "redline_rmw" if mode == "serial_latency" else "disjoint_retained_pm4_lanes"
            )
    for item in value.values():
        _recursive_redline_contract(item, mode=mode)


def _result_mode(result: dict[str, Any]) -> str:
    rows = result.get("measurements", {}).get("rows", [])
    modes = {
        str(row.get("timing_mode"))
        for row in rows
        if isinstance(row, dict) and row.get("timing_mode") is not None
    }
    if len(modes) != 1:
        raise ValueError("Redline result must contain exactly one timing mode")
    return timing_contract.parse_timing_mode(modes.pop())


def canonicalize_backend_result(
    result: dict[str, Any], *, backend: str
) -> dict[str, Any]:
    """Return one schema-v2 backend result from separate or joint wrappers.

    Most micro wrappers already emit ``hipengine_micro_result``. Reduction and
    two-stage reduction are joint wrappers and retain their backend rows inside
    a comparison-shaped artifact even when invoked with one backend; this
    function creates a lossless backend view without changing the measured
    runner output kept beside it.
    """

    if backend not in {"hip", "vulkan"}:
        raise ValueError("canonical runner backend must be HIP or Vulkan")
    if (
        result.get("schema_version") == 2
        and result.get("kind") == "hipengine_micro_result"
        and result.get("backend") == backend
        and isinstance(result.get("measurements", {}).get("rows"), list)
    ):
        return copy.deepcopy(result)
    rows = result.get("rows")
    if result.get("schema_version") != 2 or result.get("kind") != "hipengine_micro_comparison":
        raise ValueError("runner output cannot be converted to a backend result")
    if not isinstance(rows, list):
        raise ValueError("joint runner output is missing backend rows")
    selected = [copy.deepcopy(row) for row in rows if row.get("backend") == backend]
    if not selected:
        raise ValueError(f"joint runner output has no {backend} rows")
    hardware = result.get("hardware", {})
    backend_hardware = hardware.get(backend) if isinstance(hardware, dict) else None
    if not isinstance(backend_hardware, dict):
        raise ValueError(f"joint runner output is missing {backend} hardware")
    correctness = result.get("correctness", {})
    backend_correctness = (
        correctness.get(backend) if isinstance(correctness, dict) else None
    )
    if not isinstance(backend_correctness, dict):
        backend_correctness = {
            "status": (
                "pass"
                if all(bool(row.get("correctness_pass")) for row in selected)
                else "fail"
            ),
            "row_count": len(selected),
        }
    environment = result.get("environment")
    canonical: dict[str, Any] = {
        "schema_version": 2,
        "kind": "hipengine_micro_result",
        "bench": result.get("bench"),
        "backend": backend,
        "hardware": copy.deepcopy(backend_hardware),
        "source": copy.deepcopy(result.get("source", {})),
        "command": copy.deepcopy(result.get("command", [])),
        "parameters": copy.deepcopy(result.get("config", result.get("inputs", {}))),
        "correctness": copy.deepcopy(backend_correctness),
        "measurements": {"rows": selected},
        "classification": result.get("classification", "diagnostic_unclassified"),
        "runner_view": {
            "kind": result.get("kind"),
            "selected_backend": backend,
            "selected_rows": len(selected),
            "total_rows": len(rows),
            "timing_samples_recomputed": False,
        },
        "notes": (
            "Backend view of a joint microbenchmark wrapper; runner_artifact_ref points "
            "to the complete original output and no timing samples were recomputed."
        ),
    }
    if isinstance(environment, dict) and environment.get("ref"):
        canonical["environment_ref"] = str(environment["ref"])
    elif isinstance(environment, dict) and environment.get("captured") is not None:
        canonical["environment"] = copy.deepcopy(environment["captured"])
    else:
        canonical["environment"] = {}
    return canonical


def normalize_redline_result(
    result: dict[str, Any],
    *,
    redline_evidence: dict[str, Any],
    library_path: Path,
    adapter_paths: Sequence[Path],
    sidecar_paths: Sequence[Path],
) -> dict[str, Any]:
    """Relabel an unchanged HIP-shaped runner result with strict PM4 evidence."""

    normalized = copy.deepcopy(result)
    if normalized.get("schema_version") != 2:
        raise ValueError("Redline normalization requires a timing-contract-v2 result")
    if normalized.get("kind") != "hipengine_micro_result":
        raise ValueError("Redline normalization requires a micro result")
    if normalized.get("backend") != "hip":
        raise ValueError("Redline adapter input must be an HIP-shaped result")
    if redline_evidence.get("commit") != PINNED_REDLINE_COMMIT:
        raise ValueError("Redline normalization requires the pinned source commit")
    if bool(redline_evidence.get("dirty")):
        raise ValueError("Redline normalization requires a clean source checkout")
    mode = _result_mode(normalized)
    _recursive_redline_contract(normalized, mode=mode)
    normalized["backend"] = "redline"

    library = _file_record(library_path)
    adapters = [_file_record(path) for path in adapter_paths]
    sidecars = [_file_record(path) for path in sorted(set(sidecar_paths))]
    if not sidecars:
        raise ValueError("Redline result requires at least one measured code-object sidecar")
    manifest_count = sum(
        record["path"].endswith((".manifest.json", ".radiowave.json"))
        for record in sidecars
    )
    code_object_count = sum(
        record["path"].endswith((".co", ".hsaco")) for record in sidecars
    )
    if manifest_count == 0 or code_object_count == 0:
        raise ValueError("Redline sidecars must include code-object and manifest evidence")

    digest = hashlib.sha256()
    for record in [library, *adapters, *sidecars]:
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\0")
    checkout = {
        "root": str(redline_evidence.get("root") or ""),
        "commit": str(redline_evidence["commit"]),
        "dirty": False,
    }
    normalized["redline_provenance"] = {
        "checkout": checkout,
        "library": library,
        "library_sha256": library["sha256"],
        "adapters": adapters,
        "sidecars": sidecars,
        "integration_sha256": "sha256:" + digest.hexdigest(),
        "execution_proof": {
            "api": "redline-capi",
            "native_hip_fallback_available": False,
            "profiled_retained_pm4_required": True,
            "radiowave_manifest_verified": True,
        },
        "capture_role": "HIP graph argument/topology introspection outside timed samples",
        "code_object_parity_with_native_hip": "not_proven_requires_same_hsaco_control",
    }
    parameters = normalized.setdefault("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("micro result parameters must be an object")
    parameters["redline_submission"] = {
        "strategy": "retained_pm4_ib",
        "timed_api": "rl_pm4_replay_profiled/rl_pm4_replay_multi_profiled",
        "gpu_clock": "redline_pm4_timestamp",
        "serial_dependency": "consumer-aware RMW boundary",
        "independent_policy": "one complete logical iteration per retained queue lane",
        "native_hip_fallback": False,
    }
    source = normalized.get("source")
    if isinstance(source, dict):
        source["redline_source_hash"] = normalized["redline_provenance"][
            "integration_sha256"
        ]
    rows = normalized.get("measurements", {}).get("rows", [])
    for row in rows:
        if isinstance(row, dict):
            timing_contract.validate_timed_row(row)
    return normalized


def build_family_command(
    *,
    runner: Path,
    backend: str,
    mode: str,
    output: Path,
    environment: Path,
    build_dir: Path,
    gfx_arch: str,
    gpu_name: str,
    device_index: int,
    independent_lanes: int,
    reps: int,
    warmup: int,
    samples: int,
    family_args: Sequence[str],
) -> list[str]:
    if backend not in {"hip", "vulkan"}:
        raise ValueError("family runners expose HIP or Vulkan; Redline is HIP-shaped")
    timing_contract.parse_timing_mode(mode)
    if independent_lanes <= 0:
        raise ValueError("independent_lanes must be positive")
    return [
        sys.executable,
        str(runner),
        "--backend",
        backend,
        "--timing-mode",
        mode,
        "--independent-streams",
        str(independent_lanes),
        "--reps",
        str(reps),
        "--warmup",
        str(warmup),
        "--samples",
        str(samples),
        "--environment-json",
        str(environment),
        "--environment-ref",
        str(environment),
        "--gfx-arch",
        gfx_arch,
        "--hardware-gpu",
        gpu_name,
        "--device-index",
        str(device_index),
        "--build-dir",
        str(build_dir),
        *family_args,
        "--out",
        str(output),
        "--pretty",
    ]


def _rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = result.get("measurements", {}).get("rows")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("micro result contains no timing rows")
    return rows


def _row_key(family: str, row: dict[str, Any]) -> tuple[Any, ...]:
    if family not in ROW_KEYS:
        raise ValueError(f"unsupported Redline comparison family: {family}")
    values = [row.get(field) for field in ROW_KEYS[family]]
    if family == "reduction" and values[0] == "subgroup":
        values[0] = "wave_shuffle"
    return tuple(values)


def _index_rows(family: str, result: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in _rows(result):
        key = _row_key(family, row)
        if key in indexed:
            raise ValueError(f"duplicate {family} row: {key}")
        indexed[key] = row
    return indexed


def _device_fingerprint(value: Any) -> str:
    return "_".join(
        token for token in "".join(
            character.lower() if character.isalnum() else " " for character in str(value)
        ).split()
    )


def _validate_result_set(results: dict[str, dict[str, Any]], family: str) -> None:
    if set(results) != set(BACKENDS):
        raise ValueError("three-backend comparison requires HIP, Vulkan, and Redline results")
    bench = None
    source_identity = None
    arch = None
    device = None
    for backend in BACKENDS:
        result = results[backend]
        if result.get("schema_version") != 2 or result.get("kind") != "hipengine_micro_result":
            raise ValueError(f"{backend} input is not a timing-contract-v2 micro result")
        if result.get("backend") != backend:
            raise ValueError(f"{backend} result has the wrong backend identity")
        if result.get("correctness", {}).get("status") != "pass":
            raise ValueError(f"{backend} result correctness did not pass")
        if not all(bool(row.get("correctness_pass", True)) for row in _rows(result)):
            raise ValueError(f"{backend} result contains a failed correctness row")
        bench = bench or result.get("bench")
        if result.get("bench") != bench:
            raise ValueError("three-backend benchmark identities do not match")
        source = result.get("source", {})
        identity = (
            source.get("repo"), source.get("branch"), source.get("commit"), source.get("dirty")
        )
        source_identity = source_identity or identity
        if identity != source_identity:
            raise ValueError("three-backend source identities do not match")
        if not source.get("commit") or bool(source.get("dirty")):
            raise ValueError("three-backend comparison requires clean committed source")
        hardware = result.get("hardware", {})
        result_arch = str(hardware.get("gfx_arch") or "")
        result_device = _device_fingerprint(hardware.get("gpu_name"))
        arch = arch or result_arch
        device = device or result_device
        if result_arch != arch or result_device != device:
            raise ValueError("three-backend hardware identities do not match")
    proof = results["redline"].get("redline_provenance", {}).get("execution_proof", {})
    required_proof = {
        "api": "redline-capi",
        "native_hip_fallback_available": False,
        "profiled_retained_pm4_required": True,
        "radiowave_manifest_verified": True,
    }
    if proof != required_proof:
        raise ValueError("Redline result does not prove profiled retained-PM4 execution")
    checkout = results["redline"].get("redline_provenance", {}).get("checkout", {})
    if checkout.get("commit") != PINNED_REDLINE_COMMIT or bool(checkout.get("dirty")):
        raise ValueError("Redline result does not use the clean pinned checkout")
    if family not in ROW_KEYS:
        raise ValueError(f"unsupported family: {family}")


def _domain_pair(
    lhs: dict[str, Any],
    rhs: dict[str, Any],
    *,
    lhs_backend: str,
    rhs_backend: str,
    control: str,
    domain: str,
) -> dict[str, Any]:
    try:
        pair = timing_contract.backend_pair_ratio(
            lhs,
            rhs,
            lhs_backend=lhs_backend,
            rhs_backend=rhs_backend,
            control=control,
            domain=domain,
        )
    except ValueError as exc:
        return {
            "status": (
                "not_comparable_submission_contract" if domain == "host_wall" else "not_comparable"
            ),
            "reason": str(exc),
        }
    return {"status": "ok", **pair}


def _median_or_none(values: Iterable[float]) -> float | None:
    sequence = list(values)
    return statistics.median(sequence) if sequence else None


def build_three_backend_comparison(
    results: dict[str, dict[str, Any]],
    *,
    family: str,
    command: Sequence[str],
    input_refs: dict[str, str],
    same_hsaco_control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one strict tri-backend comparison for a family and timing mode."""

    _validate_result_set(results, family)
    indexed = {backend: _index_rows(family, results[backend]) for backend in BACKENDS}
    row_sets = {backend: set(rows) for backend, rows in indexed.items()}
    if len({frozenset(keys) for keys in row_sets.values()}) != 1:
        raise ValueError("three-backend row matrices do not match exactly")
    keys = sorted(row_sets["hip"], key=str)
    comparisons: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    redline_over_hip: list[float] = []
    redline_over_vulkan: list[float] = []
    redline_first = 0
    modes: set[str] = set()
    pairs = (
        ("redline_vs_hip", "redline", "hip"),
        ("redline_vs_vulkan", "redline", "vulkan"),
        ("vulkan_vs_hip", "vulkan", "hip"),
    )
    for key in keys:
        rows = {backend: indexed[backend][key] for backend in BACKENDS}
        for row in rows.values():
            timing_contract.validate_timed_row(row)
            modes.add(str(row["timing_mode"]))
        if len({row["timing_mode"] for row in rows.values()}) != 1:
            raise ValueError("three-backend timing modes do not match")
        mode = str(rows["hip"]["timing_mode"])
        if mode == "independent_throughput":
            hip_lanes = int(rows["hip"]["submission"]["queue_or_stream_count"])
            redline_lanes = int(rows["redline"]["submission"]["queue_or_stream_count"])
            if hip_lanes != redline_lanes:
                raise ValueError(
                    f"HIP and Redline independent lane counts differ: {hip_lanes} != {redline_lanes}"
                )
        gpu_times = {
            backend: float(
                row["timing"]["burst"]["gpu_elapsed"]["per_iteration_us"]["median"]
            )
            for backend, row in rows.items()
        }
        winner = min(gpu_times, key=gpu_times.get)
        redline_first += int(winner == "redline")
        redline_over_hip.append(gpu_times["redline"] / gpu_times["hip"])
        redline_over_vulkan.append(gpu_times["redline"] / gpu_times["vulkan"])
        matched_rows.append(
            {
                "shape": dict(zip(ROW_KEYS[family], key, strict=True)),
                "timing_mode": mode,
                "gpu_burst_us_per_iteration": gpu_times,
                "winner": winner,
                "redline_over_hip_time_ratio": redline_over_hip[-1],
                "redline_over_vulkan_time_ratio": redline_over_vulkan[-1],
                "lane_counts": {
                    backend: int(row["submission"]["queue_or_stream_count"])
                    for backend, row in rows.items()
                },
            }
        )
        for pair_name, lhs_backend, rhs_backend in pairs:
            for control in timing_contract.TIMING_CONTROLS:
                comparisons.append(
                    {
                        "pair": pair_name,
                        "shape": dict(zip(ROW_KEYS[family], key, strict=True)),
                        "timing_mode": mode,
                        "control": control,
                        "gpu_elapsed": _domain_pair(
                            rows[lhs_backend],
                            rows[rhs_backend],
                            lhs_backend=lhs_backend,
                            rhs_backend=rhs_backend,
                            control=control,
                            domain="gpu_elapsed",
                        ),
                        "host_wall": _domain_pair(
                            rows[lhs_backend],
                            rows[rhs_backend],
                            lhs_backend=lhs_backend,
                            rhs_backend=rhs_backend,
                            control=control,
                            domain="host_wall",
                        ),
                    }
                )
    if len(modes) != 1:
        raise ValueError("three-backend input artifacts must each contain one shared timing mode")
    mode = modes.pop()
    source = results["hip"]["source"]
    performance_claim = (
        not bool(source.get("dirty"))
        and all(result.get("correctness", {}).get("status") == "pass" for result in results.values())
    )
    if same_hsaco_control is None:
        transport_attribution = {
            "status": "blocked_no_same_hsaco_control",
            "performance_claim": False,
            "reason": (
                "The source/math/shape matrix is matched, but native HIP and Redline code-object "
                "byte identity is not proven by this harness."
            ),
        }
    else:
        passed = bool(same_hsaco_control.get("passed"))
        transport_attribution = {
            "status": "accepted_same_hsaco_control" if passed else "rejected_same_hsaco_control",
            "performance_claim": performance_claim and passed,
            "control": same_hsaco_control,
        }
    return {
        "schema_version": 2,
        "kind": "hipengine_micro_comparison",
        "bench": str(results["hip"]["bench"]),
        "classification": (
            "runtime_dispatch"
            if transport_attribution.get("performance_claim")
            else "diagnostic_unclassified"
        ),
        "performance_claim": performance_claim,
        "transport_attribution": transport_attribution,
        "source": source,
        "sources": {backend: results[backend]["source"] for backend in BACKENDS},
        "command": list(command),
        "hardware": {backend: results[backend]["hardware"] for backend in BACKENDS},
        "inputs": dict(input_refs),
        "correctness": {
            backend: results[backend].get("correctness", {}) for backend in BACKENDS
        },
        "parameters": {
            "family": family,
            "timing_mode": mode,
            "ratio_convention": "lhs_vs_rhs_speedup = rhs_time / lhs_time",
            "independent_lane_contract": "HIP and Redline lane counts must match",
        },
        "matched_rows": matched_rows,
        "comparisons": comparisons,
        "summary": {
            "matched_rows": len(matched_rows),
            "redline_first_rows": redline_first,
            "redline_first_percent": 100.0 * redline_first / len(matched_rows),
            "redline_faster_than_hip_rows": sum(value < 1.0 for value in redline_over_hip),
            "redline_faster_than_vulkan_rows": sum(value < 1.0 for value in redline_over_vulkan),
            "median_redline_over_hip_time_ratio": _median_or_none(redline_over_hip),
            "median_redline_over_vulkan_time_ratio": _median_or_none(redline_over_vulkan),
        },
    }


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Path,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            list(command), cwd=str(cwd), env=env, text=True, stdout=output, stderr=subprocess.STDOUT
        )
    if completed.returncode != 0:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        raise RuntimeError(
            f"command failed ({completed.returncode}); see {log}\n" + "\n".join(lines[-40:])
        )


def _adapter_paths(redline_root: Path) -> list[Path]:
    integration = redline_root / "examples" / "hipengine-6409"
    return [
        integration / "toolchain" / "hipcc",
        integration / "redline_timing_override.hpp",
        integration / "redline_hip_timing.py",
        integration / "run_python_runner.py",
        integration / "hsaco_manifest.py",
    ]


def _redline_library(redline_root: Path) -> Path:
    return redline_root / "target" / "release" / "libredline_dispatch.so"


def _required_redline_binaries(redline_root: Path) -> list[Path]:
    return [
        _redline_library(redline_root),
        redline_root / "target" / "release" / "radiowave",
    ]


def _build_redline(redline_root: Path, *, env: dict[str, str], log: Path) -> None:
    _run(
        ["cargo", "build", "--release", "-p", "redline-capi", "-p", "radiowave"],
        cwd=redline_root,
        env=env,
        log=log,
    )


def _sidecars(build_dir: Path) -> list[Path]:
    suffixes = (".redline.co", ".redline.hsaco", ".redline.manifest.json", ".redline.radiowave.json")
    return sorted(
        path for path in build_dir.rglob("*") if path.is_file() and path.name.endswith(suffixes)
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _parse_csv(value: str, *, allowed: Iterable[str], label: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(values) - set(allowed)
    if not values or unknown:
        raise ValueError(f"invalid {label}: {sorted(unknown) if unknown else value}")
    return values


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    redline_root = args.redline_root.expanduser().resolve()
    evidence = validate_redline_checkout(redline_root)
    base_env = dict(os.environ)
    hipcc = args.hipcc.expanduser().resolve()
    if not hipcc.is_file():
        raise ValueError(f"hipcc is missing: {hipcc}")
    llvm_bin = args.llvm_bin.expanduser().resolve()
    if not llvm_bin.is_dir():
        raise ValueError(f"LLVM bin directory is missing: {llvm_bin}")
    base_env.update(
        {
            "PATH": f"{hipcc.parent}:{llvm_bin}:{base_env.get('PATH', '')}",
            "ROCM_PATH": str(args.rocm_root.expanduser().resolve()),
            "HIP_PATH": str(args.rocm_root.expanduser().resolve()),
            "HIP_CLANG_PATH": str(llvm_bin),
            "HIPENGINE_HIP_ARCH": args.gfx_arch,
            "HIP_VISIBLE_DEVICES": args.visible_device,
            "ROCR_VISIBLE_DEVICES": args.visible_device,
            "REDLINE_REAL_HIPCC": str(hipcc),
            "REDLINE_CLANG_OFFLOAD_BUNDLER": str(llvm_bin / "clang-offload-bundler"),
        }
    )
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.build_redline:
        _build_redline(redline_root, env=base_env, log=out / "logs" / "build-redline.log")
    missing = [path for path in _required_redline_binaries(redline_root) if not path.is_file()]
    if missing:
        raise ValueError(f"Redline build artifacts are missing: {', '.join(map(str, missing))}")
    for path in _adapter_paths(redline_root):
        if not path.is_file():
            raise ValueError(f"Redline adapter is missing: {path}")

    environment = args.environment or (out / "environment.json")
    environment = environment.expanduser().resolve()
    if not environment.exists():
        _run(
            [sys.executable, str(COLLECT_ENV), "--out", str(environment), "--pretty"],
            cwd=REPO_ROOT,
            env=base_env,
            log=out / "logs" / "collect-environment.log",
        )
    backends = _parse_csv(args.backends, allowed=BACKENDS, label="backends")
    modes = _parse_csv(args.modes, allowed=MODES, label="modes")
    families = _parse_csv(args.families, allowed=FAMILIES, label="families")
    started = time.monotonic()
    artifacts: list[dict[str, Any]] = []
    comparisons: list[str] = []
    for mode in modes:
        for family in families:
            config = FAMILIES[family]
            runner = RUNNER_ROOT / str(config["runner"])
            result_paths: dict[str, Path] = {}
            for backend in backends:
                final = out / mode / f"{backend}-{family}.json"
                result_paths[backend] = final
                if final.exists() and args.resume:
                    artifacts.append({"backend": backend, "family": family, "mode": mode, "path": str(final)})
                    continue
                raw_backend = "hip" if backend == "redline" else backend
                build_dir = out / "build" / backend / family
                raw = out / mode / f"{backend}-{family}.runner.json"
                command = build_family_command(
                    runner=runner,
                    backend=raw_backend,
                    mode=mode,
                    output=raw,
                    environment=environment,
                    build_dir=build_dir,
                    gfx_arch=args.gfx_arch,
                    gpu_name=args.gpu_name,
                    device_index=args.device_index,
                    independent_lanes=args.independent_lanes,
                    reps=args.reps,
                    warmup=args.warmup,
                    samples=args.samples,
                    family_args=config["args"],
                )
                run_env = dict(base_env)
                if backend == "redline":
                    integration = redline_root / "examples" / "hipengine-6409"
                    run_env["PATH"] = f"{integration / 'toolchain'}:{run_env['PATH']}"
                    run_env["REDLINE_HIPCC_VERSION_SUFFIX"] = (
                        f"hipengine-{PINNED_REDLINE_COMMIT[:9]}-profiled-pm4"
                    )
                    run_env["RADIOWAVE_SCHEDULER_PROFILE"] = "default"
                    if bool(config.get("production_python")):
                        command = [
                            sys.executable,
                            str(integration / "run_python_runner.py"),
                            str(runner),
                            *command[2:],
                        ]
                _run(
                    command,
                    cwd=REPO_ROOT,
                    env=run_env,
                    log=out / "logs" / mode / f"{backend}-{family}.log",
                )
                canonical = canonicalize_backend_result(
                    json.loads(raw.read_text(encoding="utf-8")), backend=raw_backend
                )
                canonical["runner_artifact_ref"] = str(raw)
                canonical["artifact_ref"] = str(final)
                if backend == "redline":
                    canonical = normalize_redline_result(
                        canonical,
                        redline_evidence=evidence,
                        library_path=_redline_library(redline_root),
                        adapter_paths=_adapter_paths(redline_root),
                        sidecar_paths=_sidecars(build_dir),
                    )
                    canonical["command"] = command
                _write_json(final, canonical)
                artifacts.append({"backend": backend, "family": family, "mode": mode, "path": str(final)})
            if set(result_paths) == set(BACKENDS) and all(path.exists() for path in result_paths.values()):
                tri_results = {
                    backend: json.loads(result_paths[backend].read_text(encoding="utf-8"))
                    for backend in BACKENDS
                }
                comparison_path = out / mode / f"redline-{family}-comparison.json"
                comparison = build_three_backend_comparison(
                    tri_results,
                    family=family,
                    command=sys.argv.copy(),
                    input_refs={backend: str(result_paths[backend]) for backend in BACKENDS},
                )
                _write_json(comparison_path, comparison)
                comparisons.append(str(comparison_path))
            _write_json(
                out / "matrix.partial.json",
                {
                    "schema_version": 1,
                    "kind": "hipengine_redline_micro_matrix_manifest",
                    "status": "in_progress",
                    "redline": evidence,
                    "artifacts": artifacts,
                    "comparisons": comparisons,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
    manifest = {
        "schema_version": 1,
        "kind": "hipengine_redline_micro_matrix_manifest",
        "status": "complete",
        "redline": evidence,
        "redline_library": _file_record(_redline_library(redline_root)),
        "environment": str(environment),
        "config": {
            "backends": backends,
            "modes": modes,
            "families": families,
            "gfx_arch": args.gfx_arch,
            "gpu_name": args.gpu_name,
            "visible_device": args.visible_device,
            "device_index": args.device_index,
            "independent_lanes": args.independent_lanes,
            "reps": args.reps,
            "warmup": args.warmup,
            "samples": args.samples,
        },
        "artifacts": artifacts,
        "comparisons": comparisons,
        "dispatch_status": "not_run_separate_direct_pm4_floor_control",
        "same_hsaco_status": "not_run_separate_hipfire_control",
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(out / "matrix.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redline-root", type=Path, default=Path("/home/lhl/redline"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--rocm-root", type=Path, required=True)
    parser.add_argument("--hipcc", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--build-redline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--gfx-arch", default="gfx1100")
    parser.add_argument("--gpu-name", default="AMD Radeon Pro W7900")
    parser.add_argument("--visible-device", default="0")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--independent-lanes", type=int, default=2)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args(argv)
    if min(args.independent_lanes, args.reps, args.samples) <= 0 or args.warmup < 0:
        parser.error("lanes/reps/samples must be positive and warmup non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_matrix(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
