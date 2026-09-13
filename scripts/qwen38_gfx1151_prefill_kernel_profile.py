#!/usr/bin/env python3
"""Profile the gfx1151 GGUF prefill kernel mix at fixed prompt lengths.

Three explicit process boundaries, so compiler activity can never be mistaken for
measured device work:

1. an unprofiled build child populates a scoped JIT cache for the exact workload;
2. an unprofiled cache-only warm child proves the same workload starts with
   ``HIPENGINE_REQUIRE_CACHED_BUILD=1``, an empty compiler guard and no observed
   compiler subprocess, and that the cache tree hash does not move;
3. rocprofv3 wraps only the final child, one invocation per prompt length, with
   ``--selected-regions`` so the trace contains the measured prefill and not the
   model load, the discarded warmup prefill, or session teardown.

The child opens one ROCTx region around each measured prefill and records the
wall time, the returned token, and (bulk path) the per-stage GPU timings.

Usage:
  qwen38_gfx1151_prefill_kernel_profile.py \\
      --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \\
      --prompt-lengths 512 1024 4096 \\
      --compiler-version-file /tmp/hipcc-version-gfx1151.txt \\
      --cache-root /tmp/prefill-profile/cache --run-root /tmp/prefill-profile/run \\
      --run-tag qwen38-prefill --out /tmp/prefill-profile/profile.json

  # re-summarize an existing run without touching the GPU
  qwen38_gfx1151_prefill_kernel_profile.py --summarize-only \\
      --run-root /tmp/prefill-profile/run --out /tmp/prefill-profile/families.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gguf_continuous_owner_rocprof import (  # noqa: E402
    _default_roctx_sdk,
    _prepare_roctx_override,
    _run_monitored,
    _write_json,
    prepare_compiler_guard,
    snapshot_cache_tree,
    validate_cache_only_stage,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")

# Family classification over hipEngine kernel symbol names. Every dispatch is
# classified; anything that lands in "other" is reported by name so a naming
# surprise cannot hide the dominant kernel. Generic kernels whose name does not
# carry a quant (for example a BF16 weight GEMV) stay in "other" rather than
# being guessed into a quant family.
_FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("gdn", r"gdn|chain_conv|linear_attn|recurrent|conv1d|ssm"),
    ("attention", r"paged_attn|flash_attn|attention|attn_"),
    ("q4", r"q4_t16|q4_k|q4k|q4_|mmq|q8_1|dp4a|dense_wmma|dense_dual_wmma|"
           r"dense_rowtile|dual_rowtile|col8|col4"),
    ("q5", r"q5_t16|q5_k|q5k|q5_"),
    ("q6", r"q6_k|q6k|qmicro|planar|t16_wmma_prefill_shared"),
    ("norm_rope", r"rmsnorm|rope|rotary"),
    ("blas", r"cijk_|rocblas|hipblas|tensile"),
    ("elementwise", r"silu|gelu|add|mul|cast|convert|copy|quant|repack|dequant|"
                    r"fill|zero"),
)


def _classify(name: str) -> str:
    lowered = name.lower()
    for family, pattern in _FAMILY_PATTERNS:
        if re.search(pattern, lowered):
            return family
    return "other"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_lengths(values: Sequence[str]) -> tuple[int, ...]:
    lengths: list[int] = []
    for value in values:
        for item in str(value).replace(",", " ").split():
            lengths.append(int(item))
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("prompt lengths must be positive integers")
    if len(set(lengths)) != len(lengths):
        raise ValueError("prompt lengths must be distinct")
    return tuple(lengths)


# --------------------------------------------------------------------------- #
# child: run the workload, optionally inside one ROCTx region per measured pass
# --------------------------------------------------------------------------- #


def _child_run(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.qwen35_gguf_bench import _RoctxProfilerControl, _reset_existing_session
    from scripts.qwen38_production_ar_gate import profile_session

    control = _RoctxProfilerControl(enabled=bool(args.profile))
    if args.profile and control._resume is None:  # noqa: SLF001 - explicit guard
        raise RuntimeError(
            "profiled child requested --profile but the ROCTx profiler controls "
            "are unavailable; rocprofv3 --selected-regions would emit no rows"
        )

    class _SessionArgs:
        model = args.model
        max_sequence_length = int(args.max_sequence_length)

    lengths = _parse_lengths(args.prompt_lengths)
    results: list[dict[str, Any]] = []
    with profile_session(_SessionArgs(), None) as (session, profile):
        runtime = session.runtime
        for length in lengths:
            tokens = [int(args.prompt_token_id)] * length
            _reset_existing_session(session, runtime)
            warmup_start = time.perf_counter()
            session.prefill(tokens, use_bulk=True, bulk_attention_mode="bulk",
                            return_logits=False)
            warmup_seconds = time.perf_counter() - warmup_start
            _reset_existing_session(session, runtime)
            started = time.perf_counter()
            with control.region("prefill", selected="prefill" if args.profile else ""):
                result = session.prefill(
                    tokens, use_bulk=True, bulk_attention_mode="bulk",
                    return_logits=False, record_gpu_stage_timings=True)
            measured_seconds = time.perf_counter() - started
            row = {
                "prompt_length": length,
                "warmup_prefill_seconds": warmup_seconds,
                "measured_prefill_seconds": measured_seconds,
                "measured_prefill_tok_s": length / measured_seconds if measured_seconds else None,
                "first_token_id": int(result.token_id),
                "logits_finite": bool(result.logits is not None and result.logits.all()),
                "gpu_stage_timings_ms": dict(session.last_prefill_gpu_stage_timings_ms),
            }
            results.append(row)
            print(f"prefill {length}: {measured_seconds:.4f}s "
                  f"({row['measured_prefill_tok_s']:.1f} tok/s) token={row['first_token_id']}",
                  flush=True)
    return {
        "kind": "qwen38_gfx1151_prefill_profile_child",
        "profile": profile,
        "prompt_token_id": int(args.prompt_token_id),
        "bulk_attention_mode": "bulk",
        "use_bulk_prefill": True,
        "roctx_selected_region": bool(args.profile),
        "lengths": results,
        "environment": {k: v for k, v in os.environ.items()
                        if k.startswith(("HIPENGINE_", "GPU_MAX_HW_QUEUES"))},
    }


# --------------------------------------------------------------------------- #
# trace summarization
# --------------------------------------------------------------------------- #


def _load_dispatches(trace_dir: Path) -> list[dict[str, Any]]:
    """Read every kernel-trace CSV under one trace directory."""

    rows: list[dict[str, Any]] = []
    for path in sorted(Path(trace_dir).rglob("*.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            name_key = next((f for f in fields if "Kernel_Name" in f), None)
            if name_key is None:
                continue
            duration_key = next(
                (f for f in fields if f.strip().lower() in ("duration", "durationns")), None)
            start_key = next((f for f in fields if "Start_Timestamp" in f), None)
            end_key = next((f for f in fields if "End_Timestamp" in f), None)
            if duration_key is None and not (start_key and end_key):
                continue
            geometry = {
                "grid_x": next((f for f in fields if "Grid_Size_X" in f), None),
                "grid_y": next((f for f in fields if "Grid_Size_Y" in f), None),
                "grid_z": next((f for f in fields if "Grid_Size_Z" in f), None),
                "workgroup_x": next((f for f in fields if "Workgroup_Size_X" in f), None),
                "workgroup_y": next((f for f in fields if "Workgroup_Size_Y" in f), None),
                "workgroup_z": next((f for f in fields if "Workgroup_Size_Z" in f), None),
                "vgpr": next((f for f in fields if "VGPR_Count" in f or f == "VGPR"), None),
                "sgpr": next((f for f in fields if "SGPR_Count" in f or f == "SGPR"), None),
                "lds": next((f for f in fields if "LDS_Block_Size" in f), None),
                "scratch": next((f for f in fields if "Scratch_Size" in f), None),
            }
            for row in reader:
                name = (row.get(name_key) or "").strip()
                if not name:
                    continue
                try:
                    if duration_key is not None:
                        ns = float(row[duration_key])
                    else:
                        ns = float(row[end_key]) - float(row[start_key])
                except (TypeError, ValueError):
                    continue
                if ns <= 0:
                    continue
                entry: dict[str, Any] = {"name": name, "ns": ns, "csv": path.name}
                for key, column in geometry.items():
                    if column is None:
                        continue
                    raw = (row.get(column) or "").strip()
                    if raw and raw not in ("-", "N/A"):
                        entry[key] = raw
                rows.append(entry)
    return rows


def _blocks(entry: Mapping[str, Any], axis: str) -> str | None:
    """rocprofv3 reports Grid_Size_* in work-items; convert to workgroups."""

    total = entry.get(f"grid_{axis}")
    workgroup = entry.get(f"workgroup_{axis}")
    if total is None or workgroup is None:
        return None
    try:
        divisor = int(workgroup)
        if divisor <= 0:
            return None
        return str(int(total) // divisor)
    except (TypeError, ValueError):
        return None


def _summarize_trace(trace_dir: Path, *, top: int = 25) -> dict[str, Any]:
    dispatches = _load_dispatches(trace_dir)
    total_ns = sum(row["ns"] for row in dispatches)
    families: dict[str, dict[str, Any]] = {}
    kernels: dict[str, dict[str, Any]] = {}
    for row in dispatches:
        family = _classify(row["name"])
        bucket = families.setdefault(family, {"ns": 0.0, "launches": 0, "kernels": set()})
        bucket["ns"] += row["ns"]
        bucket["launches"] += 1
        bucket["kernels"].add(row["name"])
        kernel = kernels.setdefault(row["name"], {"ns": 0.0, "launches": 0, "geometry": {}})
        kernel["ns"] += row["ns"]
        kernel["launches"] += 1
        for key in ("grid_x", "grid_y", "grid_z", "workgroup_x", "workgroup_y",
                    "workgroup_z", "vgpr", "sgpr", "lds", "scratch"):
            if key in row:
                kernel["geometry"].setdefault(key, set()).add(row[key])
    family_rows = [
        {
            "family": family,
            "total_ms": round(bucket["ns"] / 1e6, 3),
            "share_pct": round(100.0 * bucket["ns"] / total_ns, 2) if total_ns else 0.0,
            "launches": bucket["launches"],
            "distinct_kernels": len(bucket["kernels"]),
        }
        for family, bucket in families.items()
    ]
    family_rows.sort(key=lambda row: -row["total_ms"])
    kernel_rows = []
    for name, kernel in sorted(kernels.items(), key=lambda kv: -kv[1]["ns"])[:top]:
        geometry = {key: sorted(values) for key, values in kernel["geometry"].items()}
        blocks = {}
        for key in ("grid_x", "grid_y", "grid_z"):
            values = geometry.get(key, [])
            converted = {_blocks({key: value, f"workgroup_{key[-1]}": geometry.get(
                f"workgroup_{key[-1]}", [None])[0]}, key[-1]) for value in values}
            if all(value is not None for value in converted):
                blocks[key] = sorted(converted, key=lambda item: int(item))
        kernel_rows.append({
            "kernel": name,
            "family": _classify(name),
            "total_ms": round(kernel["ns"] / 1e6, 3),
            "share_pct": round(100.0 * kernel["ns"] / total_ns, 2) if total_ns else 0.0,
            "launches": kernel["launches"],
            "mean_us": round(kernel["ns"] / kernel["launches"] / 1e3, 3),
            "grid_size_work_items": geometry,
            "grid_workgroups": blocks,
        })
    return {
        "trace_dir": str(trace_dir),
        "total_kernel_ms": round(total_ns / 1e6, 3),
        "dispatch_count": len(dispatches),
        "families": family_rows,
        "top_kernels": kernel_rows,
    }


def _summarize_run(run_root: Path, lengths: Sequence[int]) -> dict[str, Any]:
    traces: dict[str, Any] = {}
    for length in lengths:
        trace_dir = Path(run_root) / "trace" / str(length)
        if not trace_dir.is_dir() or not any(trace_dir.rglob("*.csv")):
            continue
        traces[str(length)] = _summarize_trace(trace_dir)
    return {"traces": traces}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def _queue_environment(base: Mapping[str, str], queues: str) -> dict[str, str]:
    environment = {str(key): str(value) for key, value in base.items()}
    environment["GPU_MAX_HW_QUEUES"] = str(int(queues))
    return environment


def _child_command(
    args: argparse.Namespace,
    *,
    lengths: Sequence[int],
    output: Path,
    profile: bool,
) -> list[str]:
    command = [
        str(args.python),
        str(Path(__file__).resolve()),
        "--child-mode", "run",
        "--model", str(args.model),
        "--backend", str(args.backend),
        "--quant", str(args.quant),
        "--prompt-lengths", *[str(length) for length in lengths],
        "--max-sequence-length", str(args.max_sequence_length),
        "--prompt-token-id", str(args.prompt_token_id),
        "--compiler-version-file", str(args.compiler_version_file),
        "--cache-root", str(args.cache_root),
        "--out", str(output),
    ]
    if profile:
        command.append("--profile")
    return command


def _profile_command(rocprofv3: str, trace_dir: Path, child: Sequence[str]) -> list[str]:
    return [
        str(rocprofv3),
        "--kernel-trace",
        "--selected-regions",
        "--output-format", "csv",
        "-d", str(trace_dir),
        "--",
        *[str(value) for value in child],
    ]


def _stage(
    *,
    name: str,
    command: Sequence[str],
    environment: Mapping[str, str],
    run_root: Path,
    timeout_seconds: float,
    cwd: Path,
) -> dict[str, Any]:
    log_dir = Path(run_root) / "logs"
    observation = _run_monitored(
        command,
        cwd=cwd,
        environment=environment,
        stdout_path=log_dir / f"{name}.out",
        stderr_path=log_dir / f"{name}.err",
        timeout_seconds=timeout_seconds,
    )
    observation["name"] = name
    observation["command"] = [str(value) for value in command]
    if observation["returncode"] != 0:
        tail = ""
        for path_key in ("stdout", "stderr"):
            path = Path(observation[path_key])
            if path.exists():
                tail += path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"{name} stage failed (exit {observation['returncode']}):\n{tail}")
    return observation


def run(args: argparse.Namespace) -> dict[str, Any]:
    lengths = _parse_lengths(args.prompt_lengths)
    cache_root = Path(args.cache_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    compiler_file = Path(args.compiler_version_file).expanduser().resolve()
    if not compiler_file.is_file():
        raise FileNotFoundError(f"compiler version file not found: {compiler_file}")
    if cache_root.exists() and any(cache_root.iterdir()) and not args.rebuild:
        pass  # reuse: the cache-only stage still proves it is complete
    else:
        if cache_root.exists() and args.rebuild:
            shutil.rmtree(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    roctx_override, roctx_dependencies = _prepare_roctx_override(
        Path(args.roctx_sdk).expanduser().resolve(), run_root)
    guard = prepare_compiler_guard(run_root / "compiler-guard")

    base_environment = os.environ.copy()
    base_environment["HIPENGINE_BUILD_CACHE_ROOT"] = str(cache_root)
    base_environment["HIPENGINE_HIP_ARCH"] = str(args.backend).removeprefix("hip_")
    base_environment["HIPENGINE_COMPILER_VERSION_FILE"] = str(compiler_file)
    base_environment = _queue_environment(base_environment, args.gpu_max_hw_queues)
    library_path = os.pathsep.join(
        [str(roctx_override), *[str(path) for path in roctx_dependencies],
         base_environment.get("LD_LIBRARY_PATH", "")]
    ).strip(os.pathsep)
    base_environment["LD_LIBRARY_PATH"] = library_path

    build_environment = dict(base_environment)
    build_environment.pop("HIPENGINE_REQUIRE_CACHED_BUILD", None)
    cached_environment = dict(base_environment)
    cached_environment["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    cached_environment["PATH"] = os.pathsep.join(
        [str(guard["directory"]), base_environment.get("PATH", "")])

    before_build = snapshot_cache_tree(cache_root)
    build = _stage(
        name="build",
        command=_child_command(args, lengths=lengths,
                               output=run_root / "child-build.json", profile=False),
        environment=build_environment, run_root=run_root,
        timeout_seconds=args.child_timeout_seconds, cwd=REPO_ROOT)
    after_build = snapshot_cache_tree(cache_root)

    warm = _stage(
        name="cache-only-warm",
        command=_child_command(args, lengths=lengths,
                               output=run_root / "child-warm.json", profile=False),
        environment=cached_environment, run_root=run_root,
        timeout_seconds=args.child_timeout_seconds, cwd=REPO_ROOT)
    after_warm = snapshot_cache_tree(cache_root)
    validate_cache_only_stage(
        before=after_build, after=after_warm,
        compiler_guard_marker=guard["marker"],
        observed_compiler_processes=warm["observed_compiler_processes"])

    profiles: list[dict[str, Any]] = []
    if not args.skip_profile:
        for length in lengths:
            trace_dir = run_root / "trace" / str(length)
            trace_dir.mkdir(parents=True, exist_ok=True)
            child = _child_command(args, lengths=(length,),
                                   output=run_root / f"child-profile-{length}.json",
                                   profile=True)
            profiles.append(_stage(
                name=f"profile-{length}",
                command=_profile_command(args.rocprofv3, trace_dir, child),
                environment=cached_environment, run_root=run_root,
                timeout_seconds=args.child_timeout_seconds, cwd=REPO_ROOT))

    after_profile = snapshot_cache_tree(cache_root)
    validate_cache_only_stage(
        before=after_warm, after=after_profile,
        compiler_guard_marker=guard["marker"],
        observed_compiler_processes=[
            line for stage in profiles for line in stage["observed_compiler_processes"]])

    child_payloads = {}
    for path in sorted(run_root.glob("child-*.json")):
        child_payloads[path.stem] = json.loads(path.read_text(encoding="utf-8"))

    from hipengine.benchmark.provenance import collect_artifact_provenance

    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT, configured_backend=args.backend,
        resolved_backend=args.backend, target_arch=str(args.backend).removeprefix("hip_"),
        model_path=args.model, quant=args.quant, kv_dtype="bf16",
        command=[sys.executable, *sys.argv],
        environment={k: v for k, v in base_environment.items()
                     if k.startswith(("HIPENGINE_", "GPU_MAX_HW_QUEUES"))},
        build_profile="qwen38_gfx1151_prefill_kernel_profile",
        timing_protocol="one_discarded_warmup_prefill_then_one_measured_prefill_per_length",
        warmups=1, repetitions=1, profiler={"kind": "rocprofv3", "selected_regions": True})

    payload: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen38_gfx1151_prefill_kernel_profile",
        "status": "accepted",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_tag": args.run_tag,
        "model": {"path": str(args.model), "sha256": _sha256_file(args.model),
                  "quant": args.quant, "backend": args.backend,
                  "kv_storage": "bf16"},
        "prompt_lengths": list(lengths),
        "prompt_token_id": int(args.prompt_token_id),
        "provenance": provenance,
        "stages": {
            "build": build,
            "cache_only_warm": warm,
            "profiles": profiles,
        },
        "cache": {
            "root": str(cache_root),
            "before_build": before_build,
            "after_build": after_build,
            "after_warm": after_warm,
            "after_profile": after_profile,
            "compiler_guard_marker": str(guard["marker"]),
            "compiler_guard_invoked": guard["marker"].exists(),
        },
        "children": child_payloads,
        "summaries": _summarize_run(run_root, lengths)["traces"],
        "limitations": [
            "Kernel time is rocprofv3 device duration inside the ROCTx-selected "
            "prefill region; it excludes model load, the discarded warmup "
            "prefill, and session teardown.",
            "rocprofv3 reports Grid_Size_* in work-items, not workgroups; the "
            "grid_workgroups field divides by the workgroup size. Raw values are "
            "kept in grid_size_work_items.",
            "The prompt is a repeated single token id, matching the published "
            "resident-sweep protocol; kernel selection is shape-driven, so this "
            "does not change which owners run.",
            "VGPR/SGPR and LDS are the dispatch-recorded resource counts for the "
            "launch, not a compiler report for the whole module.",
        ],
    }
    _write_json(Path(args.out), payload)
    return payload


def _print_summary(payload: Mapping[str, Any]) -> None:
    for length, summary in payload.get("summaries", {}).items():
        print(f"\n=== prompt {length}: {summary['total_kernel_ms']} ms of kernel time "
              f"across {summary['dispatch_count']} dispatches")
        print(f"{'share':>7} {'total_ms':>10} {'launches':>9} {'kernels':>8}  family")
        for row in summary["families"]:
            print(f"{row['share_pct']:6.2f}% {row['total_ms']:10.2f} "
                  f"{row['launches']:9d} {row['distinct_kernels']:8d}  {row['family']}")
        print(f"{'':7} {'':10} {'':9} {'':8}  top kernels")
        for row in summary["top_kernels"][:10]:
            raw = row["grid_size_work_items"]
            blocks = row["grid_workgroups"]
            workgroup = "x".join(raw.get(f"workgroup_{axis}", ["?"])[0]
                                 for axis in ("x", "y", "z"))
            grid = "x".join(blocks.get(f"grid_{axis}", ["?"])[0]
                            for axis in ("x", "y", "z"))
            print(f"{row['share_pct']:6.2f}% {row['total_ms']:10.2f} "
                  f"{row['launches']:9d} {'':8}  {row['kernel'][:64]}")
            print(f"{'':7} {'':10} {'':9} {'':8}    grid={grid} wg={workgroup} "
                  f"vgpr={raw.get('vgpr')} sgpr={raw.get('sgpr')} "
                  f"lds={raw.get('lds')}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--child-mode", choices=("run",), default=None,
                        help="internal: run the workload in this process")
    parser.add_argument("--summarize-only", action="store_true",
                        help="re-summarize an existing run root without a GPU run")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--prompt-lengths", nargs="+", default=["512", "1024", "4096"])
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None,
                        help="driver/summarize mode: run root holding logs, child JSON and traces")
    parser.add_argument("--run-tag", default="qwen38-gfx1151-prefill-kernel-profile")
    parser.add_argument("--gpu-max-hw-queues", choices=("1", "2", "4", "8"), default="2")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--rocprofv3", default="rocprofv3")
    parser.add_argument("--roctx-sdk", type=Path, default=_default_roctx_sdk())
    parser.add_argument("--child-timeout-seconds", type=float, default=2700.0)
    parser.add_argument("--profile", action="store_true",
                        help="internal: open the ROCTx selected region")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.child_mode == "run":
        payload = _child_run(args)
        _write_json(Path(args.out), payload)
        return 0
    if args.summarize_only:
        if args.run_root is None:
            parser.error("--run-root is required with --summarize-only")
        lengths = _parse_lengths(args.prompt_lengths)
        payload = {"schema": 1, "kind": "qwen38_gfx1151_prefill_kernel_profile_summary",
                   "run_root": str(args.run_root),
                   "summaries": _summarize_run(Path(args.run_root), lengths)["traces"]}
        _write_json(Path(args.out), payload)
        _print_summary(payload)
        return 0
    if args.compiler_version_file is None or args.cache_root is None or args.run_root is None:
        parser.error("--compiler-version-file, --cache-root and --run-root are required")
    payload = run(args)
    _print_summary(payload)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
