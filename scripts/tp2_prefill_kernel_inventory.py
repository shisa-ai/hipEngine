#!/usr/bin/env python3
"""Ranked bulk-prefill kernel inventory for the TP2 resident route, per rank.

Why this exists
---------------
The TP2 prefill route is the last large gap in the route: 459.7 tok/s against
the single-card resident bulk route's 875.8 and llama.cpp's tensor split at
1474.6 (512-token prompt, 128 decode, c=1, same host and model). Prefill is one
bulk pass per layer, so the per-layer budget is a sum over a handful of large
kernels plus whatever the two-rank exchange costs - and nothing in that sum is
attributed today. This script names the kernels inside one bulk prefill, counts
their launches, and splits them by rank, so the next attack comes from a ranked
list instead of a hypothesis.

How it measures
---------------
One child process builds the production TP2 session with ``bulk_prefill=True``,
warms it (workspace allocation, decode-graph capture and JIT loads all happen
there), synchronizes, launches a labeled probe on each device, synchronizes
again, and then runs N bulk prefills inside a single ROCTX region whose start
and end are both bounded by a full device synchronize. Because of those two
synchronizations every kernel that starts inside the region belongs to the
measured bulk passes, so the rollup can attribute by start timestamp instead of
hoping that asynchronously executed work fits inside a host marker window.

The labeled probe is what makes the per-rank split trustworthy: the same cast
kernel is launched on rank 0 with 1,024 elements and on rank 1 with 65,536, so
the two rows differ in grid size and the rollup can map each ``Agent_Id`` to a
rank from the trace itself. It refuses to report a per-rank split it cannot
verify, and falls back to a combined inventory instead.

Run:

    python3 scripts/tp2_prefill_kernel_inventory.py --prompt-tokens 512 \
        --prefills 3 --json benchmarks/results/<date>-w7900-tp2-prefill-inventory.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402 - sys.path is set above

from scripts.gguf_decode_rocprof import _default_roctx_sdk  # noqa: E402
from scripts.mtp_verifier_rocprof import (  # noqa: E402
    _prepare_roctx_override,
    _roctx_sdk_dep_paths,
    _single_file,
)
from scripts.tp2_decode_kernel_inventory import (  # noqa: E402
    SMALL_KERNEL_US,
    _agent_rank_map,
    _device_rows,
    _float_or_none,
    _git_revision,
    _git_status,
    _launch_probe,
    _read_kernel_rows,
    _read_region_window,
    _run,
    _sha256,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_ROCTX_SDK = _default_roctx_sdk()
PREFILL_MARKER = "tp2prefill:"


def _summarize(
    rows: list[dict[str, Any]], *, prefills: int, layers: int, prompt_tokens: int
) -> dict[str, Any]:
    """Rank kernels by cost per bulk prefill.

    ``per_prefill`` is the unit that matters: the route runs one bulk pass per
    layer for the whole prompt, so a kernel's per-prefill total is its share of
    the wall. ``us_per_layer`` divides that by the layer count, which is what a
    per-layer fusion or dispatch decision needs.
    """

    copies = [row for row in rows if row["kernel"].startswith("__amd_rocclr")]
    rows = [row for row in rows if not row["kernel"].startswith("__amd_rocclr")]
    by_kernel: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = by_kernel.setdefault(
            row["kernel"],
            {"kernel": row["kernel"], "calls": 0, "total_ns": 0, "durations": []},
        )
        entry["calls"] += 1
        entry["total_ns"] += row["duration_ns"]
        entry["durations"].append(row["duration_ns"])

    entries: list[dict[str, Any]] = []
    for entry in by_kernel.values():
        calls = entry["calls"]
        total_ns = entry["total_ns"]
        durations = np.array(sorted(entry["durations"]), dtype=np.float64) / 1e3
        entries.append(
            {
                "kernel": entry["kernel"],
                "calls": calls,
                "calls_per_prefill": calls / prefills,
                "calls_per_layer": calls / prefills / layers,
                "ms_per_prefill": total_ns / 1e6 / prefills,
                "us_per_call": total_ns / 1e3 / calls,
                "us_per_layer": total_ns / 1e3 / prefills / layers,
                "us_per_prompt_token": total_ns / 1e3 / prefills / prompt_tokens,
                "min_us": float(durations[0]),
                "p50_us": float(np.percentile(durations, 50)),
                "p90_us": float(np.percentile(durations, 90)),
                "max_us": float(durations[-1]),
            }
        )
    entries.sort(key=lambda item: -item["ms_per_prefill"])
    small = [entry for entry in entries if entry["us_per_call"] < SMALL_KERNEL_US]
    return {
        "kernel_calls": len(rows),
        "kernel_calls_per_prefill": len(rows) / prefills,
        "copy_calls_per_prefill": len(copies) / prefills,
        "copy_ms_per_prefill": sum(row["duration_ns"] for row in copies) / 1e6 / prefills,
        "kernel_ms_per_prefill": sum(row["duration_ns"] for row in rows) / 1e6 / prefills,
        "kernels": entries,
        "small_kernel_count": len(small),
        "small_kernel_calls_per_prefill": sum(e["calls_per_prefill"] for e in small),
        "small_kernel_ms_per_prefill": sum(e["ms_per_prefill"] for e in small),
    }


def _child(args: argparse.Namespace) -> int:
    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

    session = MlpTP2GenerationSession(
        str(args.model),
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=int(args.capacity),
        reduce_mode=str(args.reduce_mode),
        bulk_prefill=True,
        bulk_prefill_rows=(
            int(args.bulk_prefill_rows) if args.bulk_prefill_rows is not None else None
        ),
    )
    rng = np.random.default_rng(11)
    prompt = [int(i) for i in rng.integers(1000, 50000, size=int(args.prompt_tokens))]
    layer_count = len(session._config.layer_types)

    # Warmup: the bulk workspace is sized and allocated on first use, the decode
    # graph is captured, and every JIT load happens here, so the measured region
    # contains steady-state bulk prefill only. Synchronize so no warmup kernel
    # can be attributed to the region.
    session.generate(prompt, max_new_tokens=2, eos_token_id=None)
    session.runtime.device_synchronize()

    record: dict[str, Any] = {
        "kind": "tp2_prefill_kernel_inventory_child",
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "model": str(args.model),
        "model_sha256": _sha256(Path(args.model)) if Path(args.model).exists() else None,
        "route": {
            "mode": session.mode,
            "schedule": session.schedule,
            "prefill_schedule": session.prefill_schedule,
            "driver": session.driver,
            "head_shard": bool(session.head_shard),
            "reduce_mode": session.reduce_mode,
        },
        "capacity": int(args.capacity),
        "prompt_tokens": int(args.prompt_tokens),
        "prefills": int(args.prefills),
        "logits_rows": int(args.logits_rows),
        "layer_count": layer_count,
        "bulk_prefill_workspace_rows": int(getattr(session, "_bulk_rows", 0)),
        "devices": _device_rows(session),
        "marker_prefix": PREFILL_MARKER,
        "warm_only": bool(args.warm_only),
    }

    if args.warm_only:
        record["status"] = "warm-only"
        if args.json:
            Path(args.json).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        session.close()
        print(
            f"warm-only complete: {layer_count} layers, capacity {args.capacity}, "
            f"bulk workspace {record['bulk_prefill_workspace_rows']} rows",
            flush=True,
        )
        return 0

    from scripts.gguf_decode_rocprof import _Roctx

    marker = _Roctx()
    # The probe runs after a synchronize and before the region, so it is visible
    # in the raw trace but outside the measured window.
    record["probe"] = _launch_probe(session)
    session.runtime.device_synchronize()

    marker.push(PREFILL_MARKER + "start")
    started = time.perf_counter()
    digests: list[str] = []
    for _ in range(int(args.prefills)):
        # ``logits_rows`` defaults to the product path: ``generate`` projects the
        # head for the last prompt row only, so an inventory that asks for every
        # row would profile a path the engine does not ship and would rank the
        # head projection at the top of a list that no longer contains it.
        want_rows = int(args.logits_rows)
        logits = session.bulk_prefill(
            prompt, logits_rows=(None if want_rows == 0 else want_rows)
        )
        import hashlib

        digests.append(hashlib.sha256(np.asarray(logits, dtype=np.float32).tobytes()).hexdigest())
    session.runtime.device_synchronize()
    region_wall = time.perf_counter() - started
    marker.pop()

    record.update(
        {
            "status": "complete",
            "logits_sha256": digests,
            "region_wall_ms": region_wall * 1e3,
            "region_wall_ms_per_prefill": region_wall * 1e3 / max(1, int(args.prefills)),
            "prefill_tokens_per_s": int(args.prompt_tokens)
            * max(1, int(args.prefills))
            / region_wall,
            "command": " ".join([sys.executable, *sys.argv]),
        }
    )
    session.close()
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.json}", flush=True)
    print(
        f"child: {args.prefills} bulk prefills of {args.prompt_tokens} tokens, region "
        f"{region_wall * 1e3:.3f} ms ({record['prefill_tokens_per_s']:.1f} tok/s)",
        flush=True,
    )
    return 0


def _rollup(raw_root: Path, child: dict[str, Any], *, top: int) -> dict[str, Any]:
    kernel_csv = _single_file(raw_root, "*_kernel_trace.csv")
    marker_csv = _single_file(raw_root, "*_marker_api_trace.csv")
    kernels = _read_kernel_rows(kernel_csv)
    region_start, region_end = _read_region_window(marker_csv, PREFILL_MARKER)
    in_region = [row for row in kernels if region_start <= row["start_ns"] <= region_end]
    if not in_region:
        raise ValueError("prefill region contains no kernels")

    prefills = int(child["prefills"])
    layers = int(child["layer_count"])
    prompt_tokens = int(child["prompt_tokens"])
    mapping, mapping_note = _agent_rank_map(kernels, child.get("probe") or {})

    per_rank: dict[str, Any] = {}
    for rank, agents in sorted(mapping.items()):
        rows = [row for row in in_region if row.get("agent_id") in agents]
        per_rank[f"rank{rank}"] = {
            "agent_ids": sorted(agents),
            "kernels": _summarize(
                rows, prefills=prefills, layers=layers, prompt_tokens=prompt_tokens
            ),
        }

    report: dict[str, Any] = {
        "kind": "tp2_prefill_kernel_inventory",
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": child.get("host"),
        "model": child.get("model"),
        "model_sha256": child.get("model_sha256"),
        "route": child.get("route"),
        "capacity": child.get("capacity"),
        "prompt_tokens": prompt_tokens,
        "prefills": prefills,
        "layer_count": layers,
        "bulk_prefill_workspace_rows": child.get("bulk_prefill_workspace_rows"),
        "devices": child.get("devices"),
        "region_wall_ms_per_prefill": child.get("region_wall_ms_per_prefill"),
        "prefill_tokens_per_s": child.get("prefill_tokens_per_s"),
        "logits_sha256": child.get("logits_sha256"),
        "logits_repeatable": len(set(child.get("logits_sha256") or [])) == 1,
        "agent_rank_map": mapping_note,
        "source_revision": _git_revision(),
        "git_status_porcelain": _git_status(),
        "combined": _summarize(
            in_region, prefills=prefills, layers=layers, prompt_tokens=prompt_tokens
        ),
    }
    if per_rank and mapping:
        report["per_rank"] = per_rank
    return report


def _print_report(report: dict[str, Any], *, top: int) -> None:
    print(
        f"\nbulk prefill inventory: {report['prompt_tokens']} prompt tokens, "
        f"{report['prefills']} prefills, {report['layer_count']} layers, "
        f"workspace {report['bulk_prefill_workspace_rows']} rows"
    )
    print(
        f"region wall {report['region_wall_ms_per_prefill']:.3f} ms/prefill "
        f"= {report['prefill_tokens_per_s']:.1f} tok/s "
        f"(repeatable logits: {report['logits_repeatable']})"
    )
    for label, summary in [
        ("combined", report["combined"]),
        *[
            (name, entry["kernels"])
            for name, entry in sorted((report.get("per_rank") or {}).items())
        ],
    ]:
        print(
            f"\n=== {label}: {summary['kernel_ms_per_prefill']:.3f} ms/prefill of kernels, "
            f"{summary['kernel_calls_per_prefill']:.0f} calls/prefill, "
            f"{summary['small_kernel_ms_per_prefill']:.3f} ms/prefill in "
            f"{summary['small_kernel_calls_per_prefill']:.0f} small kernels"
        )
        print(
            f"{'kernel':<52} {'calls':>7} {'c/layer':>8} {'ms/pf':>9} "
            f"{'us/layer':>9} {'p50_us':>9}"
        )
        for entry in summary["kernels"][:top]:
            print(
                f"{entry['kernel'][:52]:<52} {entry['calls_per_prefill']:>7.1f} "
                f"{entry['calls_per_layer']:>8.2f} {entry['ms_per_prefill']:>9.4f} "
                f"{entry['us_per_layer']:>9.2f} {entry['p50_us']:>9.2f}"
            )


def _parent(args: argparse.Namespace) -> int:
    root = str(REPO_ROOT)
    out = Path(args.json) if args.json else Path("/tmp/tp2-prefill-kernel-inventory.json")
    child_json = Path(args.child_json)
    raw_root = Path(args.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    roctx_override = _prepare_roctx_override(Path(args.roctx_sdk))
    env = os.environ.copy()
    ld_prefix = os.pathsep.join(
        [str(roctx_override), *(str(p) for p in _roctx_sdk_dep_paths(Path(args.roctx_sdk)))]
    )
    env["LD_LIBRARY_PATH"] = f"{ld_prefix}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONPATH"] = f"{root}:{env.get('PYTHONPATH', '')}"
    if args.compiler_version_file:
        env["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    child = [
        sys.executable,
        "scripts/tp2_prefill_kernel_inventory.py",
        "--child",
        "--model",
        str(args.model),
        "--capacity",
        str(int(args.capacity)),
        "--prefills",
        str(int(args.prefills)),
        "--prompt-tokens",
        str(int(args.prompt_tokens)),
        "--reduce-mode",
        str(args.reduce_mode),
        "--json",
        str(child_json),
    ]
    if args.bulk_prefill_rows is not None:
        child += ["--bulk-prefill-rows", str(int(args.bulk_prefill_rows))]
    child += ["--logits-rows", str(int(args.logits_rows))]

    # Prebuild outside the profiler, then require the cache for the traced run:
    # a profiler-injected child must never spawn hipcc.
    print("=== warm build (outside rocprofv3) ===", flush=True)
    _run([*child, "--warm-only"], env=env, cwd=root)
    env["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    print("=== profiled child ===", flush=True)
    _run(
        [
            str(args.rocprofv3),
            "--kernel-trace",
            "--marker-trace",
            "--output-format",
            "csv",
            "-d",
            str(raw_root),
            "-o",
            "tp2-prefill-inventory",
            "--",
            *child,
        ],
        env=env,
        cwd=root,
    )

    child_record = json.loads(child_json.read_text())
    report = _rollup(raw_root, child_record, top=int(args.top))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    _print_report(report, top=int(args.top))
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--prefills", type=int, default=3, help="bulk prefills inside the measured region")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument(
        "--logits-rows",
        type=int,
        default=1,
        help="head rows to project; 1 matches generate(), 0 means every prompt row",
    )
    parser.add_argument("--bulk-prefill-rows", type=int, default=None)
    parser.add_argument("--reduce-mode", choices=("device", "host"), default="device")
    parser.add_argument("--child", action="store_true", help="run the profiled leaf only")
    parser.add_argument("--warm-only", action="store_true", help="build and warm the session, then exit")
    parser.add_argument("--rollup-only", type=Path, default=None, help="re-roll an existing trace directory")
    parser.add_argument("--child-json", type=Path, default=Path("/tmp/tp2-prefill-inventory-child.json"))
    parser.add_argument("--raw-root", type=Path, default=Path("/tmp/tp2-prefill-inventory-raw"))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--rocprofv3", default=shutil.which("rocprofv3") or "rocprofv3")
    parser.add_argument("--roctx-sdk", type=Path, default=DEFAULT_ROCTX_SDK)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    args = parser.parse_args()

    if args.child or args.warm_only:
        return _child(args)
    if args.rollup_only is not None:
        child_record = json.loads(Path(args.child_json).read_text())
        report = _rollup(Path(args.rollup_only), child_record, top=int(args.top))
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        _print_report(report, top=int(args.top))
        return 0
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
