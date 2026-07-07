#!/usr/bin/env python3
"""Run or normalize the HIP dispatch/grid-floor microbenchmark.

This runner wraps ``scripts/graph_node_microbench.py`` so the existing HIP
graph/direct launch diagnostic can emit the canonical
``hipengine_micro_result`` artifact used by ``benchmarks/micro``.
"""

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
COLLECT_ENV = Path(__file__).resolve().parents[1] / "collect_env.py"
BENCH_NAME = "dispatch_grid_floor"


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


def _reference_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    for row in rows:
        if row.get("node_count") == 941:
            return row
    return rows[-1]


def _primary_kernel(legacy: dict[str, Any]) -> str:
    verdict = legacy.get("verdict")
    if isinstance(verdict, dict) and verdict.get("kernel"):
        return str(verdict["kernel"])
    config = legacy.get("config")
    if isinstance(config, dict):
        kernels = config.get("kernels")
        if isinstance(kernels, list) and kernels:
            return str(kernels[0])
    rows_by_kernel = legacy.get("rows_by_kernel")
    if isinstance(rows_by_kernel, dict) and rows_by_kernel:
        return str(next(iter(rows_by_kernel)))
    return "unknown"


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


def _timing_summary(legacy: dict[str, Any], primary_kernel: str) -> dict[str, Any]:
    config = legacy.get("config") if isinstance(legacy.get("config"), dict) else {}
    rows_by_kernel = legacy.get("rows_by_kernel")
    if not isinstance(rows_by_kernel, dict):
        rows_by_kernel = {}
    primary_rows = rows_by_kernel.get(primary_kernel)
    if not isinstance(primary_rows, list):
        primary_rows = legacy.get("rows") if isinstance(legacy.get("rows"), list) else []
    reference = _reference_row(primary_rows)
    direct = reference.get("direct") if isinstance(reference.get("direct"), dict) else {}
    graph = reference.get("graph") if isinstance(reference.get("graph"), dict) else {}
    direct_us = direct.get("us_per_node")
    graph_us = graph.get("steady_us_per_node")
    speedup = reference.get("graph_speedup_vs_direct")
    return _json_safe(
        {
            "unit": "us_per_launch",
            "median": direct_us if isinstance(direct_us, (int, float)) else None,
            "warmup_iters": config.get("warmup"),
            "measured_iters": config.get("reps"),
            "primary": {
                "kernel": primary_kernel,
                "node_count": reference.get("node_count"),
                "direct_us_per_node_median": direct_us,
                "direct_burst_us_median": direct.get("burst_us_median"),
                "direct_burst_us_min": direct.get("burst_us_min"),
                "graph_steady_us_per_node_median": graph_us,
                "graph_batch_us_median": graph.get("batch_us_median"),
                "graph_replay_latency_us_median": graph.get("replay_latency_us_median"),
                "graph_speedup_vs_direct": speedup,
            },
        }
    )


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
    """Convert a ``graph_node_microbench.py`` artifact into micro-result schema."""

    primary_kernel = _primary_kernel(legacy)
    config = legacy.get("config") if isinstance(legacy.get("config"), dict) else {}
    result: dict[str, Any] = {
        "schema_version": 1,
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
                "n_elements": config.get("n_elements"),
                "reps": config.get("reps"),
                "warmup": config.get("warmup"),
                "target_nodes_per_batch": config.get("target_nodes_per_batch"),
                "method": config.get("method"),
            }
        ),
        "correctness": {
            "status": "not_applicable",
            "oracle": "dispatch-floor diagnostic; trivial kernels measure launch/runtime overhead",
        },
        "timing": _timing_summary(legacy, primary_kernel),
        "classification": "runtime_dispatch",
        "measurements": _json_safe(
            {
                "rows": legacy.get("rows"),
                "rows_by_kernel": legacy.get("rows_by_kernel"),
                "grid_sweep": legacy.get("grid_sweep"),
                "verdict": legacy.get("verdict"),
                "arg_scaling_verdict": legacy.get("arg_scaling_verdict"),
            }
        ),
        "notes": (
            "Normalized from scripts/graph_node_microbench.py. This row attributes "
            "HIP direct-launch and graph-replay overhead; it does not test shader math "
            "or LLVM codegen quality. The legacy benchmark reports medians/minima, not "
            "full p05/p95 distributions."
        ),
    }
    if environment_ref:
        result["environment_ref"] = environment_ref
    else:
        result["environment"] = environment
    return _json_safe(result)


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
        "--grid-sweep-count",
        str(args.grid_sweep_count),
        "--observed-residual-us-per-op",
        str(args.observed_residual_us_per_op),
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
    parser.add_argument("--kernels", default="tiny,wide", help="Comma-separated kernel profiles: tiny and/or wide")
    parser.add_argument("--n", type=int, default=256, help="Trivial-kernel element count")
    parser.add_argument("--grid-sweep", default="", help="Optional comma-separated grid block sweep")
    parser.add_argument("--grid-sweep-count", type=int, default=941, help="Node count used for grid sweep")
    parser.add_argument("--reps", type=int, default=50, help="Timed repetitions")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warmup iterations")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--observed-residual-us-per-op", type=float, default=20.6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    environment = _collect_environment(args)
    source_hash = _hash_files([Path(__file__).resolve(), GRAPH_BENCH, GRAPH_SOURCE])
    wrapper_command = [
        sys.executable,
        str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        *(argv if argv is not None else sys.argv[1:]),
    ]

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
