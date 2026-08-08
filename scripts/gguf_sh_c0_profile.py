#!/usr/bin/env python3
"""Run the SH-C0 role-resolved GGUF decode and whole-GTT baseline matrix.

Each workload gets two independent processes:

* a non-profiled resident eager baseline (one discarded full run plus repeated
  measured runs), wrapped by a 10-ms whole-card GTT sampler; and
* a cached-only ``rocprofv3`` child with kernel, HIP API, and ROCTX traces.

The profiler child emits nested host role ranges. HIP launch correlation IDs,
not timestamp containment of asynchronous kernels, join those ranges to kernel
rows. Profiled rates are diagnostic; only the non-profiled wall rows are
eligible as campaign baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.util.amdgpu_vram import VramSampler, select_card
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
    _sha256,
    _single_file,
    _summarize_role_rows,
    _summarize_rows,
    _summarize_wall_runs,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_WORKLOADS = (512, 4096, 32768, 65536)
KIND = "hipengine_gguf_sh_c0_attribution"
SCHEMA_VERSION = 1


def _parse_length(text: str) -> int:
    value = str(text).strip().lower()
    multiplier = 1
    if value.endswith("k"):
        multiplier = 1024
        value = value[:-1]
    parsed = int(value) * multiplier
    if parsed <= 0:
        raise argparse.ArgumentTypeError("workload lengths must be positive")
    return parsed


def _run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    print(f"[sh-c0] {' '.join(command)}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )


def _run_with_gtt_sampler(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    card_name: str | None,
    interval_ms: float,
) -> dict[str, Any]:
    card = select_card(card_name=card_name) if card_name else select_card()
    sampler = VramSampler(
        card,
        interval_ms=interval_ms,
        memory_domain="gtt",
        keep_samples=False,
    )
    sampler.start()
    try:
        _run_logged(
            command,
            cwd=cwd,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    finally:
        sampler.stop()
    return sampler.result().to_dict(include_samples=False)


def _trace_summary(
    trace_dir: Path,
    *,
    profile_steps: int,
    top: int,
) -> dict[str, Any]:
    kernel_csv = _single_file(trace_dir, "*_kernel_trace.csv")
    marker_csv = _single_file(trace_dir, "*_marker_api_trace.csv")
    hip_api_csv = _single_file(trace_dir, "*_hip_api_trace.csv")
    step_windows = _read_marker_windows(marker_csv, MARKER_PREFIX)
    expected_indices = list(range(profile_steps))
    observed_indices = [index for index, _start, _end in step_windows]
    if observed_indices != expected_indices:
        raise ValueError(
            f"expected exact decode step markers {expected_indices}, observed {observed_indices}"
        )
    kernels = _read_kernels(kernel_csv)
    selected = _filter_kernels_by_windows(
        kernels,
        [(start, end) for _index, start, end in step_windows],
    )
    role_windows = _read_role_windows(marker_csv, ROLE_MARKER_PREFIX)
    launches = _read_hip_launches(hip_api_csv)
    annotated = _annotate_kernel_roles(
        selected,
        launches=launches,
        windows=role_windows,
    )
    summary = _summarize_rows(annotated, steps=profile_steps, top=top)
    summary.update(
        {
            "roles": _summarize_role_rows(annotated, steps=profile_steps),
            "step_marker_windows": len(step_windows),
            "role_marker_windows": len(role_windows),
            "whole_trace_kernels": len(kernels),
            "selected_decode_kernels": len(selected),
            "role_attributed_kernels": sum(
                row.get("role") != "unattributed" for row in annotated
            ),
            "raw_trace": {
                "kernel_csv": str(kernel_csv),
                "kernel_csv_sha256": _sha256(kernel_csv),
                "marker_csv": str(marker_csv),
                "marker_csv_sha256": _sha256(marker_csv),
                "hip_api_csv": str(hip_api_csv),
                "hip_api_csv_sha256": _sha256(hip_api_csv),
            },
        }
    )
    return summary


def _tracked_source_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    changed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return {
        "root": str(REPO_ROOT),
        "commit": commit,
        "tracked_dirty": bool(changed),
        "tracked_status_entries": changed,
    }


def _host_device_summary(profile_child: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    wall = _summarize_wall_runs(
        profile_child["measured_runs"],
        expected_token_id=int(profile_child["workload"]["expected_token_id"]),
    )
    host_us = float(wall["median_ms_per_token"]) * 1000.0
    device_us = float(trace["gpu_us_per_token"])
    return {
        "profiled_host": wall,
        "gpu_kernel_us_per_token": device_us,
        "profiled_host_minus_gpu_us_per_token": host_us - device_us,
        "gpu_kernel_share_of_profiled_host_wall_pct": (
            100.0 * device_us / host_us if host_us else None
        ),
        "qualification": (
            "Profiler overhead and trace collection perturb host wall; use the "
            "non-profiled baseline for throughput."
        ),
    }


def _child_command(
    *,
    mode: str,
    model: Path,
    backend: str,
    prompt_length: int,
    expected_token_id: int,
    steps: int,
    warmup_steps: int,
    benchmark_warmups: int,
    repetitions: int,
    compiler_file: Path,
    output: Path,
    require_cached: bool,
) -> list[str]:
    return _build_child_command(
        python=sys.executable,
        script=REPO_ROOT / "scripts" / "gguf_decode_rocprof.py",
        child_mode=mode,
        source_root=REPO_ROOT,
        model=model,
        backend=backend,
        prompt_token_id=9707,
        prompt_length=prompt_length,
        expected_token_id=expected_token_id,
        steps=steps,
        warmup_steps=warmup_steps,
        benchmark_warmups=benchmark_warmups,
        repetitions=repetitions,
        compiler_version_file=compiler_file,
        child_json=output,
        require_cached=require_cached,
    )


def _run_workload(
    args: argparse.Namespace,
    *,
    prompt_length: int,
    env: dict[str, str],
    roctx_prefix: str,
) -> dict[str, Any]:
    label = str(prompt_length)
    root = Path(args.raw_root) / label
    root.mkdir(parents=True, exist_ok=True)

    baseline_json = root / "baseline.json"
    baseline_command = _child_command(
        mode="baseline",
        model=args.model,
        backend=args.backend,
        prompt_length=prompt_length,
        expected_token_id=args.expected_token_id,
        steps=args.baseline_steps,
        warmup_steps=args.warmup_steps,
        benchmark_warmups=args.baseline_warmups,
        repetitions=args.baseline_repetitions,
        compiler_file=args.compiler_version_file,
        output=baseline_json,
        require_cached=True,
    )
    gtt = _run_with_gtt_sampler(
        baseline_command,
        cwd=REPO_ROOT,
        env=env,
        stdout_path=root / "baseline.stdout.log",
        stderr_path=root / "baseline.stderr.log",
        card_name=args.card_name,
        interval_ms=args.gtt_interval_ms,
    )
    baseline = _load_child(
        baseline_json,
        expected_token_id=args.expected_token_id,
    )
    baseline_wall = _summarize_wall_runs(
        baseline["measured_runs"],
        expected_token_id=args.expected_token_id,
    )

    trace_dir = root / "trace"
    trace_dir.mkdir()
    profile_json = root / "profile.json"
    profile_command = _child_command(
        mode="profile",
        model=args.model,
        backend=args.backend,
        prompt_length=prompt_length,
        expected_token_id=args.expected_token_id,
        steps=args.profile_steps,
        warmup_steps=args.warmup_steps,
        benchmark_warmups=0,
        repetitions=1,
        compiler_file=args.compiler_version_file,
        output=profile_json,
        require_cached=True,
    )
    profile_env = dict(env)
    profile_env["LD_LIBRARY_PATH"] = (
        f"{roctx_prefix}:{profile_env.get('LD_LIBRARY_PATH', '')}"
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
        f"sh-c0-{prompt_length}",
        "--",
        *profile_command,
    ]
    _run_logged(
        rocprof_command,
        cwd=REPO_ROOT,
        env=profile_env,
        stdout_path=root / "profile.stdout.log",
        stderr_path=root / "profile.stderr.log",
    )
    profile = _load_child(profile_json, expected_token_id=args.expected_token_id)
    trace = _trace_summary(
        trace_dir,
        profile_steps=args.profile_steps,
        top=args.top,
    )
    if trace["role_attributed_kernels"] <= 0:
        raise RuntimeError(f"{prompt_length}: no decode kernels received role attribution")

    return {
        "prompt_length": prompt_length,
        "decode_tokens": args.baseline_steps,
        "non_profiled": {
            "wall": baseline_wall,
            "whole_gtt_10ms": gtt,
            "owned_memory": baseline["memory"],
            "child_json": str(baseline_json),
            "child_json_sha256": _sha256(baseline_json),
            "command": baseline_command,
        },
        "profiled": {
            "trace": trace,
            "host_device": _host_device_summary(profile, trace),
            "child_memory": profile["memory"],
            "child_json": str(profile_json),
            "child_json_sha256": _sha256(profile_json),
            "command": rocprof_command,
        },
        "route": baseline["route"],
        "source": baseline["source"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--workloads",
        nargs="+",
        type=_parse_length,
        default=list(DEFAULT_WORKLOADS),
    )
    parser.add_argument("--backend", choices=("hip_gfx1100", "hip_gfx1151"), default="hip_gfx1151")
    parser.add_argument("--expected-token-id", type=int, default=9707)
    parser.add_argument("--baseline-steps", type=int, default=128)
    parser.add_argument("--baseline-warmups", type=int, default=1)
    parser.add_argument("--baseline-repetitions", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--gtt-interval-ms", type=float, default=10.0)
    parser.add_argument("--card-name")
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=Path("/tmp/hipengine-hipcc-version.txt"),
    )
    parser.add_argument("--rocprofv3", default=shutil.which("rocprofv3") or "rocprofv3")
    parser.add_argument("--roctx-sdk", type=Path, default=_default_roctx_sdk())
    parser.add_argument("--raw-root", type=Path, default=Path("/tmp/hipengine-sh-c0"))
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    for name in ("baseline_steps", "baseline_repetitions", "profile_steps"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("baseline_warmups", "warmup_steps"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.gtt_interval_ms <= 0:
        raise ValueError("--gtt-interval-ms must be positive")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    if args.raw_root.exists():
        shutil.rmtree(args.raw_root)
    args.raw_root.mkdir(parents=True)
    if not args.compiler_version_file.exists():
        compiler = subprocess.run(
            ["hipcc", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        args.compiler_version_file.write_text(compiler.stdout, encoding="utf-8")

    roctx_override, roctx_dependencies = _prepare_roctx_override(args.roctx_sdk)
    roctx_prefix = os.pathsep.join(
        [str(roctx_override), *(str(path) for path in roctx_dependencies)]
    )
    env = os.environ.copy()
    env["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    env["HIPENGINE_HIP_ARCH"] = "gfx1151" if args.backend == "hip_gfx1151" else "gfx1100"
    env["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    env["GPU_MAX_HW_QUEUES"] = "1"

    rows = [
        _run_workload(args, prompt_length=length, env=env, roctx_prefix=roctx_prefix)
        for length in args.workloads
    ]
    tracked_source = _tracked_source_state()
    all_exact = all(
        row["non_profiled"]["wall"]["all_tokens_exact"]
        and row["profiled"]["host_device"]["profiled_host"]["all_tokens_exact"]
        for row in rows
    )
    artifact = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted_diagnostic" if all_exact else "rejected_correctness",
        "performance_claim": False,
        "correctness_claim": bool(all_exact),
        "source": tracked_source,
        "workload": {
            "model": str(args.model.resolve()),
            "quant": "gguf_q4_k_m",
            "kv_dtype": "bf16",
            "backend": args.backend,
            "prompt_source": "repeated_token_id",
            "prompt_token_id": 9707,
            "prompt_lengths": list(args.workloads),
            "baseline_decode_tokens": args.baseline_steps,
            "profile_decode_tokens": args.profile_steps,
            "baseline_protocol": "one discarded full run plus repeated non-profiled eager runs",
            "profile_protocol": "cached-only rocprof kernel+HIP API+ROCTX role trace",
            "gtt_sampling_interval_ms": args.gtt_interval_ms,
        },
        "rows": rows,
        "summary": {
            "non_profiled_decode_tok_s": {
                str(row["prompt_length"]): row["non_profiled"]["wall"]["median_tok_s"]
                for row in rows
            },
            "whole_gtt_peak_gib": {
                str(row["prompt_length"]): row["non_profiled"]["whole_gtt_10ms"]["peak_gib"]
                for row in rows
            },
            "tracked_peak_gib": {
                str(row["prompt_length"]): row["non_profiled"]["owned_memory"]["summary"]["tracked_peak_allocated_gib"]
                for row in rows
            },
            "all_tokens_exact": all_exact,
        },
        "notes": [
            "Profiled host rates are diagnostic and are not compared with the non-profiled fork topline.",
            "Whole-GTT is a same-scope 10-ms sysfs sample; tracked and owned components remain process-local hipEngine accounting.",
            "Role attribution maps ROCTX-enclosed HIP launch APIs to asynchronous kernel rows by Correlation_Id.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"[sh-c0] wrote {args.out}", flush=True)
    for row in rows:
        print(
            f"[sh-c0] {row['prompt_length']}: "
            f"{row['non_profiled']['wall']['median_tok_s']:.3f} tok/s, "
            f"GTT {row['non_profiled']['whole_gtt_10ms']['peak_gib']:.3f} GiB, "
            f"GPU {row['profiled']['trace']['gpu_us_per_token']:.1f} us/token",
            flush=True,
        )
    return 0 if all_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
