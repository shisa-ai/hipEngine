#!/usr/bin/env python3
"""Run, normalize, or compare the v2 HIP dispatch/grid-floor benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
GRAPH_BENCH = REPO_ROOT / "scripts" / "graph_node_microbench.py"
GRAPH_SOURCE = REPO_ROOT / "scripts" / "microbench" / "graph_node_microbench.hip"
TIMING_CONTRACT = REPO_ROOT / "benchmarks" / "micro" / "timing_contract.py"
HIP_TIMING_HEADER = REPO_ROOT / "benchmarks" / "micro" / "runners" / "micro_timing_hip.hpp"
COLLECT_ENV = Path(__file__).resolve().parents[1] / "collect_env.py"
BENCH_NAME = "dispatch_grid_floor"


def _load_timing_contract():
    spec = importlib.util.spec_from_file_location(
        "micro_hip_dispatch_timing_contract", TIMING_CONTRACT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load timing contract: {TIMING_CONTRACT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing_contract = _load_timing_contract()


def _load_collect_env_module():
    spec = importlib.util.spec_from_file_location("micro_collect_env", COLLECT_ENV)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load environment collector: {COLLECT_ENV}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _find_gfx_arch(text: str) -> str | None:
    match = re.search(r"\bgfx[0-9a-fA-F]+\b", text)
    return match.group(0) if match else None


def _infer_gfx_arch(
    legacy: dict[str, Any],
    environment: dict[str, Any],
    override: str | None,
) -> str:
    if override:
        return override
    env_arch = os.environ.get("HIPENGINE_HIP_ARCH")
    if env_arch:
        return env_arch
    devices = environment.get("devices")
    if isinstance(devices, dict):
        for key in ("rocminfo_name_gfx_lines", "vulkan_summary_lines", "lspci_display_lines"):
            lines = devices.get(key)
            if isinstance(lines, list):
                arch = _find_gfx_arch("\n".join(str(line) for line in lines))
                if arch:
                    return arch
    hardware = legacy.get("hardware")
    if isinstance(hardware, dict):
        arch = hardware.get("gfx_arch") or hardware.get("arch")
        if arch:
            return str(arch)
    return "unknown"


def _infer_gpu_name(
    legacy: dict[str, Any],
    environment: dict[str, Any],
    override: str | None,
) -> str:
    if override:
        return override
    hardware = legacy.get("hardware")
    if isinstance(hardware, dict) and hardware.get("gpu"):
        return str(hardware["gpu"])
    devices = environment.get("devices")
    if isinstance(devices, dict):
        for key in ("rocm_smi_lines", "lspci_display_lines", "vulkan_summary_lines"):
            lines = devices.get(key)
            if not isinstance(lines, list):
                continue
            for line in lines:
                text = str(line)
                if any(marker in text for marker in ("AMD", "ATI", "Radeon", "Instinct")):
                    return text
    return "unknown"


def _collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    if args.environment_json:
        return json.loads(args.environment_json.read_text(encoding="utf-8"))
    collector = _load_collect_env_module()
    return collector.collect_environment(
        repo_root=REPO_ROOT,
        include_device_probes=not args.skip_device_probes,
        include_privileged=False,
        timeout_s=args.env_timeout_s,
        max_output_chars=args.env_max_output_chars,
    )


def _source_record(environment: dict[str, Any], source_hash: str) -> dict[str, Any]:
    repo = environment.get("repo")
    if not isinstance(repo, dict):
        repo = {}
    return {
        "repo": str(repo.get("root") or REPO_ROOT),
        "branch": str(repo.get("branch") or ""),
        "commit": str(repo.get("commit") or ""),
        "dirty": bool(repo.get("dirty")),
        "source_hash": source_hash,
    }


def _samples(raw: dict[str, Any], name: str, expected: int) -> list[float]:
    values = raw.get(name)
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError(f"{name} must contain exactly {expected} timing samples")
    return [float(value) for value in values]


def _contract_row(
    raw: dict[str, Any],
    config: dict[str, Any],
    *,
    sweep: str,
) -> dict[str, Any]:
    config_mode = timing_contract.parse_timing_mode(str(config.get("timing_mode")))
    timing_mode = timing_contract.parse_timing_mode(str(raw.get("timing_mode")))
    if timing_mode != config_mode:
        raise ValueError("HIP row timing mode does not match the raw artifact config")
    node_count = int(raw["node_count"])
    sample_count = int(config["reps"])
    if node_count <= 0 or sample_count <= 0:
        raise ValueError("node_count and reps must be positive")
    expected_strategy = (
        "hip_graph" if timing_mode == "serial_latency" else "multi_stream"
    )
    if raw.get("submission_strategy") != expected_strategy:
        raise ValueError(
            f"HIP {timing_mode} rows must use {expected_strategy} submission"
        )
    burst_iterations = node_count
    burst_dispatches = 1
    correctness_pass = bool(raw.get("single_correctness_pass")) and bool(
        raw.get("burst_correctness_pass")
    )
    correctness = timing_contract.make_correctness(
        status="pass" if correctness_pass else "fail",
        oracle="exact narrow-kernel output after the measured dispatch sequence",
        logical_iterations=burst_iterations,
        coverage=(
            "all_dispatches"
            if timing_mode == "independent_throughput"
            else "chained_final_state"
        ),
        synchronization_method=(
            "disjoint_output_slices"
            if timing_mode == "independent_throughput"
            else "hip_stream_order"
        ),
        barrier_count=0,
    )
    contract = timing_contract.make_timed_row_contract(
        timing_mode=timing_mode,
        backend="hip",
        repetitions=burst_iterations,
        dispatches_per_iteration=burst_dispatches,
        dependency_validation_status="pass" if correctness_pass else "fail",
        submission=timing_contract.make_submission(
            strategy=expected_strategy,
            queue_or_stream_count=int(raw.get("stream_count", 1)),
            recording_in_timed_region=False,
        ),
        single_timing=timing_contract.make_timing_control(
            logical_iterations=1,
            dispatches_per_iteration=1,
            gpu_samples_us=_samples(raw, "single_gpu_samples_us", sample_count),
            host_samples_us=_samples(raw, "single_host_samples_us", sample_count),
            gpu_clock="hip_event",
        ),
        burst_timing=timing_contract.make_timing_control(
            logical_iterations=burst_iterations,
            dispatches_per_iteration=burst_dispatches,
            gpu_samples_us=_samples(raw, "burst_gpu_samples_us", sample_count),
            host_samples_us=_samples(raw, "burst_host_samples_us", sample_count),
            gpu_clock="hip_event",
        ),
        correctness=correctness,
    )
    base = {
        key: value
        for key, value in raw.items()
        if key
        not in {
            "single_gpu_samples_us",
            "single_host_samples_us",
            "burst_gpu_samples_us",
            "burst_host_samples_us",
        }
    }
    sequence_median = float(
        contract["timing"]["burst"]["gpu_elapsed"]["sequence_us"]["median"]
    )
    return {
        **base,
        **contract,
        "sweep": sweep,
        "burst_us_median": sequence_median,
        "us_per_dispatch": sequence_median / node_count,
    }


def normalize_legacy_dispatch_result(
    legacy: dict[str, Any],
    *,
    environment: dict[str, Any],
    wrapper_command: list[str],
    legacy_command: list[str] | None,
    source_hash: str,
    hardware_gpu: str | None = None,
    gfx_arch: str | None = None,
    environment_ref: str | None = None,
) -> dict[str, Any]:
    """Convert a raw graph-node artifact into the v2 timing contract."""

    config = legacy.get("config") if isinstance(legacy.get("config"), dict) else {}
    raw_rows = legacy.get("rows") if isinstance(legacy.get("rows"), list) else []
    raw_grid_rows = (
        legacy.get("grid_sweep") if isinstance(legacy.get("grid_sweep"), list) else []
    )
    rows = [
        *(_contract_row(row, config, sweep="count") for row in raw_rows),
        *(_contract_row(row, config, sweep="grid") for row in raw_grid_rows),
    ]
    if not rows:
        raise ValueError("raw HIP dispatch artifact contains no timing rows")
    correctness_pass = all(
        row["correctness"]["timed_sequence"]["status"] == "pass" for row in rows
    )
    result: dict[str, Any] = {
        "schema_version": 2,
        "kind": "hipengine_micro_result",
        "bench": BENCH_NAME,
        "backend": "hip",
        "hardware": {
            "gpu_name": _infer_gpu_name(legacy, environment, hardware_gpu),
            "gfx_arch": _infer_gfx_arch(legacy, environment, gfx_arch),
        },
        "source": _source_record(environment, source_hash),
        "command": wrapper_command,
        "cwd": str(REPO_ROOT),
        "parameters": _json_safe(
            {
                "benchmark_family": "dispatch_and_grid_floor",
                "legacy_runner": str(GRAPH_BENCH.relative_to(REPO_ROOT)),
                "legacy_run_tag": legacy.get("run_tag"),
                "legacy_status": legacy.get("status"),
                "legacy_command": legacy_command,
                "counts": config.get("counts"),
                "kernels": config.get("kernels"),
                "grid_sweep": config.get("grid_sweep"),
                "grid_sweep_count": config.get("grid_sweep_count"),
                "n_elements": config.get("n_elements"),
                "local_size_x": config.get("local_size_x"),
                "reps": config.get("reps"),
                "warmup": config.get("warmup"),
                "timing_mode": config.get("timing_mode"),
                "independent_streams": config.get("independent_streams"),
                "target_nodes_per_batch": config.get("target_nodes_per_batch"),
                "method": config.get("method"),
            }
        ),
        "correctness": {
            "status": "pass" if correctness_pass else "fail",
            "oracle": "exact single and measured-burst narrow-kernel outputs",
            "mismatches": sum(int(row.get("correctness_mismatches", 0)) for row in rows),
        },
        "classification": "runtime_dispatch",
        "measurements": _json_safe(
            {
                "rows": rows,
                "count_rows": rows[: len(raw_rows)],
                "grid_sweep_rows": rows[len(raw_rows) :],
                "hip_only_wide_diagnostics": legacy.get("hip_only_wide_diagnostics"),
            }
        ),
        "notes": (
            "Serial rows replay a pre-recorded HIP graph; independent rows use multiple "
            "nonblocking HIP streams and disjoint outputs. Each row contains HIP-event "
            "and submit-to-completion host-wall single and burst controls. Wide-argument "
            "HIP diagnostics are deliberately excluded from cross-backend timing rows."
        ),
    }
    if environment_ref:
        result["environment_ref"] = environment_ref
    else:
        result["environment"] = environment
    return _json_safe(result)


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row.get("sweep")),
        int(row.get("node_count", row.get("dispatch_count", 0))),
        int(row.get("grid_blocks", 0)),
        str(row.get("timing_mode")),
    )


def _domain_comparison(
    hip_row: dict[str, Any],
    vulkan_row: dict[str, Any],
    *,
    control: str,
    domain: str,
) -> dict[str, Any]:
    hip_metric = hip_row["timing"][control][domain]
    vulkan_metric = vulkan_row["timing"][control][domain]
    if hip_metric.get("status") != "ok" or vulkan_metric.get("status") != "ok":
        return {
            "status": "unavailable",
            "hip_status": hip_metric.get("status"),
            "vulkan_status": vulkan_metric.get("status"),
        }
    try:
        ratio = timing_contract.comparison_ratio(
            hip_row,
            vulkan_row,
            control=control,
            domain=domain,
        )
    except ValueError as exc:
        if domain != "host_wall":
            raise
        return {
            "status": "not_comparable_submission_contract",
            "reason": str(exc),
        }
    return {"status": "ok", **ratio}


def build_comparison(
    hip_result: dict[str, Any],
    vulkan_result: dict[str, Any],
    *,
    command: list[str],
    hip_ref: str | None = None,
    vulkan_ref: str | None = None,
    out_ref: str | None = None,
) -> dict[str, Any]:
    if hip_result.get("schema_version") != 2 or vulkan_result.get("schema_version") != 2:
        raise ValueError("dispatch comparison requires v2 timing-contract results")
    if hip_result.get("backend") != "hip" or vulkan_result.get("backend") != "vulkan":
        raise ValueError("dispatch comparison inputs must be HIP then Vulkan")
    hip_arch = str(hip_result.get("hardware", {}).get("gfx_arch", ""))
    vulkan_arch = str(vulkan_result.get("hardware", {}).get("gfx_arch", ""))
    if not hip_arch or hip_arch == "unknown" or hip_arch != vulkan_arch:
        raise ValueError("HIP and Vulkan dispatch gfx architectures do not match")
    hip_parameters = hip_result.get("parameters", {})
    vulkan_parameters = vulkan_result.get("parameters", {})
    for field in ("n_elements", "local_size_x", "reps", "warmup"):
        if hip_parameters.get(field) != vulkan_parameters.get(field):
            raise ValueError(f"HIP and Vulkan dispatch {field} values do not match")
    hip_rows = {
        _row_key(row): row
        for row in hip_result.get("measurements", {}).get("rows", [])
        if isinstance(row, dict)
    }
    vulkan_rows = {
        _row_key(row): row
        for row in vulkan_result.get("measurements", {}).get("rows", [])
        if isinstance(row, dict)
    }
    if not hip_rows or not vulkan_rows:
        raise ValueError("dispatch comparison inputs must contain timing rows")
    hip_modes = {key[3] for key in hip_rows}
    vulkan_modes = {key[3] for key in vulkan_rows}
    if hip_modes != vulkan_modes:
        raise ValueError("HIP and Vulkan dispatch timing modes do not match")
    if set(hip_rows) != set(vulkan_rows):
        raise ValueError("HIP and Vulkan dispatch row shapes do not match")
    comparisons: list[dict[str, Any]] = []
    for key in sorted(set(hip_rows) & set(vulkan_rows)):
        hip_row = hip_rows[key]
        vulkan_row = vulkan_rows[key]
        timing_contract.dependency_signature(hip_row)
        timing_contract.dependency_signature(vulkan_row)
        for control in timing_contract.TIMING_CONTROLS:
            comparisons.append(
                {
                    "timing_mode": key[3],
                    "control": control,
                    "sweep": key[0],
                    "dispatch_count": key[1],
                    "grid_blocks": key[2],
                    "gpu_elapsed": _domain_comparison(
                        hip_row, vulkan_row, control=control, domain="gpu_elapsed"
                    ),
                    "host_wall": _domain_comparison(
                        hip_row, vulkan_row, control=control, domain="host_wall"
                    ),
                }
            )
    if not comparisons:
        raise ValueError("HIP and Vulkan dispatch artifacts have no matched rows")
    return _json_safe(
        {
            "schema_version": 2,
            "kind": "hipengine_micro_comparison",
            "bench": BENCH_NAME,
            "classification": "runtime_dispatch",
            "source": hip_result.get("source", {}),
            "command": command,
            "hardware": {
                "hip": hip_result.get("hardware", {}),
                "vulkan": vulkan_result.get("hardware", {}),
            },
            "inputs": {
                "hip_result": hip_ref,
                "vulkan_result": vulkan_ref,
                "out": out_ref,
            },
            "correctness": {
                "hip": hip_result.get("correctness", {}),
                "vulkan": vulkan_result.get("correctness", {}),
            },
            "comparisons": comparisons,
            "interpretation": (
                "GPU elapsed ratios compare matched dependency contracts. Serial host-wall "
                "ratios compare pre-recorded HIP graph replay with pre-recorded Vulkan command "
                "buffer submit/fence. Independent HIP multi-stream host wall is intentionally "
                "not ratioed against a single Vulkan command-buffer submission."
            ),
        }
    )


def _legacy_command(args: argparse.Namespace, legacy_json: Path) -> list[str]:
    command = [
        sys.executable,
        str(GRAPH_BENCH.relative_to(REPO_ROOT)),
        "--counts",
        args.counts,
        "--kernels",
        args.kernels,
        "--n",
        str(args.n),
        "--reps",
        str(args.reps),
        "--warmup",
        str(args.warmup),
        "--timing-mode",
        args.timing_mode,
        "--independent-streams",
        str(args.independent_streams),
        "--grid-sweep-count",
        str(args.grid_sweep_count),
        "--json",
        str(legacy_json),
    ]
    if args.grid_sweep:
        command.extend(["--grid-sweep", args.grid_sweep])
    if args.compiler_version_file:
        command.extend(["--compiler-version-file", str(args.compiler_version_file)])
    if args.require_cached_build:
        command.append("--require-cached-build")
    if args.hardware_gpu:
        command.extend(["--hardware-gpu", args.hardware_gpu])
    return command


def _run_legacy(command: list[str], *, gfx_arch: str | None) -> int:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_pythonpath
        else os.pathsep.join([str(REPO_ROOT), existing_pythonpath])
    )
    if gfx_arch:
        env["HIPENGINE_HIP_ARCH"] = gfx_arch
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return completed.returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("HIP_RESULT", "VULKAN_RESULT"),
        type=Path,
        help="Build a matched v2 HIP/Vulkan comparison artifact",
    )
    parser.add_argument("--legacy-input", type=Path, help="Normalize an existing graph_node_microbench JSON artifact")
    parser.add_argument("--legacy-json", type=Path, help="Keep the raw graph_node_microbench JSON at this path")
    parser.add_argument("--out", type=Path, help="Write normalized JSON to this path instead of stdout")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print normalized JSON")
    parser.add_argument("--environment-json", type=Path, help="Use a pre-collected environment JSON")
    parser.add_argument(
        "--environment-ref",
        help="Record this external environment artifact path instead of embedding it",
    )
    parser.add_argument(
        "--skip-device-probes",
        action="store_true",
        help="Skip rocminfo/rocm-smi/vulkaninfo/lspci collection",
    )
    parser.add_argument("--env-timeout-s", type=float, default=8.0, help="Per-command environment probe timeout")
    parser.add_argument("--env-max-output-chars", type=int, default=20000, help="Per-command environment output limit")
    parser.add_argument(
        "--gfx-arch",
        default=os.environ.get("HIPENGINE_HIP_ARCH"),
        help="HIP arch to record and pass via HIPENGINE_HIP_ARCH",
    )
    parser.add_argument("--hardware-gpu", default=None, help="Human-readable GPU name for the normalized artifact")

    parser.add_argument("--counts", default="1,50,200,941", help="Comma-separated node counts for the HIP benchmark")
    parser.add_argument("--kernels", default="tiny", help="Comma-separated HIP profiles: tiny and optional HIP-only wide")
    parser.add_argument("--n", type=int, default=256, help="Trivial-kernel element count")
    parser.add_argument("--grid-sweep", default="", help="Optional comma-separated grid block sweep")
    parser.add_argument("--grid-sweep-count", type=int, default=941, help="Node count used for grid sweep")
    parser.add_argument("--reps", type=int, default=50, help="Timed repetitions")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warmup iterations")
    parser.add_argument(
        "--timing-mode",
        choices=timing_contract.TIMING_MODES,
        default="serial_latency",
    )
    parser.add_argument("--independent-streams", type=int, default=4)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    wrapper_command = [
        sys.executable,
        str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        *(argv if argv is not None else sys.argv[1:]),
    ]
    if args.compare:
        hip_path, vulkan_path = args.compare
        result = build_comparison(
            json.loads(hip_path.read_text(encoding="utf-8")),
            json.loads(vulkan_path.read_text(encoding="utf-8")),
            command=wrapper_command,
            hip_ref=str(hip_path),
            vulkan_ref=str(vulkan_path),
            out_ref=str(args.out) if args.out else None,
        )
        text = json.dumps(
            result,
            indent=2 if args.pretty else None,
            sort_keys=args.pretty,
            allow_nan=False,
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0

    environment = _collect_environment(args)
    source_hash = _hash_files(
        [Path(__file__).resolve(), GRAPH_BENCH, GRAPH_SOURCE, HIP_TIMING_HEADER, TIMING_CONTRACT]
    )

    legacy_command: list[str] | None = None
    temp_path: Path | None = None
    if args.legacy_input:
        legacy_path = args.legacy_input
    else:
        if args.legacy_json:
            legacy_path = args.legacy_json
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            tmp = tempfile.NamedTemporaryFile(prefix="hipengine-dispatch-floor-", suffix=".json", delete=False)
            tmp.close()
            temp_path = Path(tmp.name)
            legacy_path = temp_path
        legacy_command = _legacy_command(args, legacy_path)
        rc = _run_legacy(legacy_command, gfx_arch=args.gfx_arch)
        if rc != 0:
            return rc

    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    result = normalize_legacy_dispatch_result(
        legacy,
        environment=environment,
        wrapper_command=wrapper_command,
        legacy_command=legacy_command,
        source_hash=source_hash,
        hardware_gpu=args.hardware_gpu,
        gfx_arch=args.gfx_arch,
        environment_ref=args.environment_ref,
    )
    text = json.dumps(result, indent=2 if args.pretty else None, sort_keys=args.pretty, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if temp_path:
        temp_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
