#!/usr/bin/env python3
"""Capture the independent ZBook Qwen3.6 GGUF c1 ownership baseline.

This adapter records one same-host current-package wall baseline and one marked
rocprofv3 decode-only profile. It compares no historical or cross-host rate and
makes no optimization/profile-qualification claim; its output selects the first
mechanism for the independent ZBook production-numerics lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.gguf_decode_rocprof import (
    MARKER_PREFIX,
    ROLE_MARKER_PREFIX,
    _annotate_kernel_roles,
    _build_child_command,
    _default_roctx_sdk,
    _filter_kernels_by_windows,
    _load_child,
    _prepare_roctx_override,
    _read_hip_launches,
    _read_kernels,
    _read_marker_windows,
    _read_role_windows,
    _run_command,
    _single_file,
    _summarize_role_rows,
    _summarize_rows,
    _summarize_wall_runs,
)

KIND = "zbook_qwen36_gguf_production_numerics_c1_profile"
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_GDN_MODE = "chain_lds32_direct_nonvolatile"


def rank_candidate_roles(
    roles: Sequence[Mapping[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return the largest attributed decode roles for mechanism selection."""

    if limit <= 0:
        raise ValueError("candidate role limit must be positive")
    attributed = [dict(row) for row in roles if str(row.get("name")) != "unattributed"]
    attributed.sort(key=lambda row: float(row.get("gpu_us_per_token", 0.0)), reverse=True)
    return attributed[:limit]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compiler_version_file(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        result = subprocess.run(
            ["hipcc", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(result.stdout, encoding="utf-8")
    if not resolved.read_text(encoding="utf-8").strip():
        raise ValueError(f"compiler-version file is empty: {resolved}")
    return resolved


def _child_command(
    args: argparse.Namespace,
    *,
    mode: str,
    steps: int,
    warmup_steps: int,
    benchmark_warmups: int,
    repetitions: int,
    output: Path,
    require_cached: bool,
    compiler_file: Path,
) -> list[str]:
    return _build_child_command(
        python=sys.executable,
        script=REPO_ROOT / "scripts" / "gguf_decode_rocprof.py",
        child_mode=mode,
        source_root=REPO_ROOT,
        model=Path(args.model),
        backend=str(args.backend),
        prompt_token_id=int(args.prompt_token_id),
        prompt_length=int(args.prompt_length),
        expected_token_id=int(args.expected_token_id),
        steps=int(steps),
        warmup_steps=int(warmup_steps),
        benchmark_warmups=int(benchmark_warmups),
        repetitions=int(repetitions),
        compiler_version_file=compiler_file,
        child_json=output,
        require_cached=require_cached,
    )


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if socket.gethostname() != str(args.expected_host):
        raise ValueError(
            f"this lane requires host {args.expected_host!r}, got {socket.gethostname()!r}"
        )
    if str(args.backend) != "hip_gfx1151":
        raise ValueError("the ZBook lane requires backend hip_gfx1151")
    if not Path(args.model).is_file():
        raise ValueError(f"model does not exist: {args.model}")
    if int(args.prompt_length) != 512 or int(args.baseline_steps) != 128:
        raise ValueError("binding ZBook c1 baseline requires p512/d128")
    if int(args.baseline_repetitions) < 5 or int(args.profile_steps) < 16:
        raise ValueError("binding ZBook profile requires >=5 wall repeats and >=16 profile steps")

    raw_root = Path(args.raw_root).resolve()
    if raw_root.exists():
        raise ValueError(f"raw root already exists; choose a fresh path: {raw_root}")
    raw_root.mkdir(parents=True)
    trace_dir = raw_root / "trace"
    trace_dir.mkdir()
    compiler_file = _compiler_version_file(Path(args.compiler_version_file))

    env = os.environ.copy()
    env.update(
        {
            "HIPENGINE_HIP_ARCH": "gfx1151",
            "HIPENGINE_GGUF_DECODE_REPACK": "1",
            "HIPENGINE_GGUF_GDN_PREFILL_MODE": str(args.gdn_mode),
            "HIPENGINE_COMPILER_VERSION_FILE": str(compiler_file),
        }
    )

    if not args.skip_warmbuild:
        warm_json = raw_root / "warmbuild.json"
        warm_command = _child_command(
            args,
            mode="warmbuild",
            steps=1,
            warmup_steps=1,
            benchmark_warmups=0,
            repetitions=1,
            output=warm_json,
            require_cached=False,
            compiler_file=compiler_file,
        )
        _run_command(warm_command, cwd=REPO_ROOT, env=env)

    baseline_json = raw_root / "baseline.json"
    baseline_command = _child_command(
        args,
        mode="baseline",
        steps=int(args.baseline_steps),
        warmup_steps=int(args.baseline_warmup_steps),
        benchmark_warmups=1,
        repetitions=int(args.baseline_repetitions),
        output=baseline_json,
        require_cached=True,
        compiler_file=compiler_file,
    )
    _run_command(baseline_command, cwd=REPO_ROOT, env=env)

    profile_json = raw_root / "profile-child.json"
    profile_command = _child_command(
        args,
        mode="profile",
        steps=int(args.profile_steps),
        warmup_steps=int(args.profile_warmup_steps),
        benchmark_warmups=0,
        repetitions=1,
        output=profile_json,
        require_cached=True,
        compiler_file=compiler_file,
    )
    roctx_override, roctx_dependencies = _prepare_roctx_override(Path(args.roctx_sdk))
    profile_env = env.copy()
    ld_prefix = os.pathsep.join(
        [str(roctx_override), *(str(path) for path in roctx_dependencies)]
    )
    profile_env["LD_LIBRARY_PATH"] = (
        f"{ld_prefix}:{profile_env.get('LD_LIBRARY_PATH', '')}"
    )
    rocprof_command = [
        str(args.rocprofv3),
        "--kernel-trace",
        "--marker-trace",
        "--hip-trace",
        "--output-format",
        "csv",
        "-d",
        str(trace_dir),
        "-o",
        "zbook-production-numerics-c1",
        "--",
        *profile_command,
    ]
    _run_command(rocprof_command, cwd=REPO_ROOT, env=profile_env)

    baseline = _load_child(baseline_json, expected_token_id=int(args.expected_token_id))
    profile_child = _load_child(profile_json, expected_token_id=int(args.expected_token_id))
    baseline_summary = _summarize_wall_runs(
        baseline["measured_runs"], expected_token_id=int(args.expected_token_id)
    )
    profile_wall = _summarize_wall_runs(
        profile_child["measured_runs"], expected_token_id=int(args.expected_token_id)
    )

    kernel_csv = _single_file(trace_dir, "*_kernel_trace.csv")
    marker_csv = _single_file(trace_dir, "*_marker_api_trace.csv")
    hip_api_csv = _single_file(trace_dir, "*_hip_api_trace.csv")
    windows = _read_marker_windows(marker_csv, MARKER_PREFIX)
    expected_indices = list(range(int(args.profile_steps)))
    observed_indices = [index for index, _start, _end in windows]
    if observed_indices != expected_indices:
        raise ValueError(
            f"profile marker windows differ: expected {expected_indices}, got {observed_indices}"
        )
    all_kernels = _read_kernels(kernel_csv)
    selected_kernels = _filter_kernels_by_windows(
        all_kernels,
        [(start, end) for _index, start, end in windows],
    )
    role_windows = _read_role_windows(marker_csv, ROLE_MARKER_PREFIX)
    hip_launches = _read_hip_launches(hip_api_csv)
    role_kernels = _annotate_kernel_roles(
        selected_kernels,
        launches=hip_launches,
        windows=role_windows,
    )
    profile_summary = _summarize_rows(
        role_kernels,
        steps=int(args.profile_steps),
        top=int(args.top),
    )
    roles = _summarize_role_rows(role_kernels, steps=int(args.profile_steps))
    attributed = sum(row["role"] != "unattributed" for row in role_kernels)
    profile_summary.update(
        {
            "roles": roles,
            "candidate_roles": rank_candidate_roles(roles, limit=int(args.role_limit)),
            "marker_windows": len(windows),
            "role_marker_windows": len(role_windows),
            "hip_kernel_launch_apis": len(hip_launches),
            "whole_trace_kernels": len(all_kernels),
            "selected_decode_kernels": len(selected_kernels),
            "role_attributed_kernels": attributed,
            "host_profiled_wall": profile_wall,
        }
    )

    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=str(baseline["route"]["resolved_backend"]),
        target_arch=str(baseline["route"]["target_arch"]),
        model_path=Path(args.model),
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_HIP_ARCH": "gfx1151",
            "HIPENGINE_GGUF_DECODE_REPACK": "1",
            "HIPENGINE_GGUF_GDN_PREFILL_MODE": str(args.gdn_mode),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        },
        build_profile="zbook_qwen36_gguf_current_c1_ownership",
        timing_protocol="same_host_p512_d128_1x5_plus_marked_decode_windows_v1",
        warmups=1,
        repetitions=int(args.baseline_repetitions),
        profiler={
            "enabled": True,
            "kind": "rocprofv3_kernel_marker_hip_trace",
            "command": rocprof_command,
            "profile_steps": int(args.profile_steps),
            "kernel_trace_sha256": _sha256(kernel_csv),
            "marker_trace_sha256": _sha256(marker_csv),
            "hip_api_trace_sha256": _sha256(hip_api_csv),
        },
    )
    measurement_valid = bool(
        provenance["host_name"] == str(args.expected_host)
        and not provenance["dirty"]
        and baseline_summary["all_tokens_exact"]
        and profile_wall["all_tokens_exact"]
        and len(windows) == int(args.profile_steps)
        and attributed > 0
    )
    return {
        "schema_version": 1,
        "kind": KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if measurement_valid else "invalid",
        "measurement_valid": measurement_valid,
        "performance_claim": False,
        "optimization_claim": False,
        "profile_qualification_claim": False,
        "measurement_host": {
            "host_name": str(args.expected_host),
            "cross_host_comparison": "prohibited",
        },
        "workload": {
            "model": str(Path(args.model).resolve()),
            "quant": "gguf_q4_k_m",
            "kv_dtype": "bf16",
            "prompt_source": "repeated_token_id",
            "prompt_token_id": int(args.prompt_token_id),
            "prompt_length": int(args.prompt_length),
            "decode_steps": int(args.baseline_steps),
            "sampling": "greedy expected-token fixture",
            "route": "current-package c1 eager decode",
        },
        "current_c1_baseline": {
            "summary": baseline_summary,
            "child_route": baseline["route"],
            "command": baseline_command,
        },
        "decode_ownership": profile_summary,
        "raw_trace": {
            "root": str(raw_root),
            "kernel_csv_sha256": _sha256(kernel_csv),
            "marker_csv_sha256": _sha256(marker_csv),
            "hip_api_csv_sha256": _sha256(hip_api_csv),
            "profile_child_sha256": _sha256(profile_json),
        },
        "provenance": provenance,
        "limitations": [
            "Current ownership/baseline packet only; no candidate was measured.",
            "Repeated-token exactness is a profiling guard, not the complete production task gate.",
            "Absolute rates are ZBook-local and cannot be compared with another gfx1151 host.",
            "Public execution-profile model plans remain unregistered for this model/backend/quant.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--expected-host", default="zbook")
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--expected-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--baseline-steps", type=int, default=128)
    parser.add_argument("--baseline-warmup-steps", type=int, default=1)
    parser.add_argument("--baseline-repetitions", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=24)
    parser.add_argument("--profile-warmup-steps", type=int, default=4)
    parser.add_argument("--gdn-mode", default=DEFAULT_GDN_MODE)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--skip-warmbuild", action="store_true")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--rocprofv3", default=shutil.which("rocprofv3") or "rocprofv3")
    parser.add_argument("--roctx-sdk", type=Path, default=_default_roctx_sdk())
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--role-limit", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    command = [sys.executable, "scripts/zbook_production_numerics_c1_profile.py", *raw_argv]
    try:
        artifact = run(args, command=command)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(json.dumps(artifact["decode_ownership"]["candidate_roles"], indent=2))
    return 0 if artifact["measurement_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
