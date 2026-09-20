#!/usr/bin/env python3
"""Ranked decode-kernel inventory for the TP2 resident route, per rank.

Why this exists
---------------
The TP2 default route moves 61.8% of the resident TP1 control's bytes but takes
76.1% of its wall, and the layer audit split the deficit into two parts: the
projections already run at or above the resident blended bandwidth, while the
attention/GDN half carries 89 us of small-kernel time per full-attention layer
and 50 us per GDN layer - 3.82 ms/token, 15.8% of the wall, paid on both ranks
because those weights are replicated. That budget is an aggregate. This script
names the kernels inside it, counts launches per layer, and splits them by rank,
so the next fusion target comes from a ranked list instead of a hypothesis.

How it measures
---------------
One child process builds the production TP2 session (``MlpTP2GenerationSession``
on both ranks), warms it, synchronizes, launches a labeled probe on each device,
synchronizes again, and then runs N decode steps inside a single ROCTX region
whose start and end are both bounded by a full device synchronize. Because of
those two synchronizations every kernel that starts inside the region belongs to
the measured decode steps, so the rollup can attribute by start timestamp
instead of hoping that asynchronously executed work fits inside a host marker
window.

The labeled probe is what makes the per-rank split trustworthy: the same cast
kernel is launched on rank 0 with 1,024 elements and on rank 1 with 65,536, so
the two rows differ in grid size and the rollup can map each ``Agent_Id`` to a
rank from the trace itself. It refuses to report a per-rank split it cannot
verify, and falls back to a combined inventory instead.

Usage
-----
    # profile + roll up (prebuilds the JIT cache first, then traces cache-only)
    python3 scripts/tp2_decode_kernel_inventory.py \
        --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf --steps 8 \
        --json benchmarks/results/<artifact>.json

    # re-roll an existing trace
    python3 scripts/tp2_decode_kernel_inventory.py --rollup-only <raw dir> \
        --json /tmp/report.json

    # the leaf alone (needs the profiler's ROCTX library on LD_LIBRARY_PATH)
    python3 scripts/tp2_decode_kernel_inventory.py --child --model M.gguf --steps 8
"""

from __future__ import annotations
import pathlib

import argparse
import collections
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.gguf_decode_rocprof import _default_roctx_sdk  # noqa: E402 - sys.path is set above
from scripts.mtp_verifier_rocprof import (  # noqa: E402 - sys.path is set above
    _prepare_roctx_override,
    _roctx_sdk_dep_paths,
    _single_file,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
# The profiler-sdk ROCTX shim ships with the ROCm SDK packages, not with
# /opt/rocm on this image; reuse the resolver the decode profiler already uses.
DEFAULT_ROCTX_SDK = _default_roctx_sdk()
DECODE_MARKER = "tp2decode:"
# One grid-size-distinct launch per rank, so Agent_Id -> rank comes from the
# trace rather than from an assumed ordering.
PROBE_ELEMENTS = (1024, 65536)
# A kernel below this mean duration is launch/latency-bound rather than
# bandwidth-bound; the fusion question is entirely about this set.
SMALL_KERNEL_US = 10.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_rows(session: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rank, device in enumerate(session.devices):
        info = session.runtime.device_info(int(device))
        out[str(rank)] = {
            "device_index": int(device),
            "name": info.name,
            "uuid": info.uuid,
            "pci_bus_id": info.pci_bus_id,
        }
    return out


def _launch_probe(session: Any) -> dict[str, Any]:
    """One small cast launch per rank with a distinct element count.

    The rollup joins these back to the trace by kernel name and grid size to map
    ``Agent_Id`` to a rank. Counts differ by 64x, so the two grids differ for any
    block size this kernel could use.
    """

    from hipengine.core.device import scoped_current_device
    from hipengine.core.memory import free, malloc
    from hipengine.kernels.hip_gfx1100.convert.cast import f32_to_bf16

    launches: list[dict[str, Any]] = []
    for rank, device in enumerate(session.devices):
        count = PROBE_ELEMENTS[min(rank, len(PROBE_ELEMENTS) - 1)]
        src = malloc(count * 4, runtime=session.runtime, device=int(device))
        dst = malloc(count * 2, runtime=session.runtime, device=int(device))
        with scoped_current_device(session.runtime, int(device)):
            f32_to_bf16(
                src.ptr,
                dst.ptr,
                count,
                stream=session._rank_stream(int(device)),
                runtime=session.runtime,
            )
        session.runtime.device_synchronize()
        launches.append({"rank": rank, "device_index": int(device), "elements": count})
        free(src, runtime=session.runtime)
        free(dst, runtime=session.runtime)
    # The trace reports the mangled kernel symbol, so the rollup matches on a
    # fragment rather than this label.
    return {"kernel_fragment": "f32_to_bf16_kernel", "launches": launches}


def _child(args: argparse.Namespace) -> int:
    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession
    from scripts.gguf_decode_rocprof import _Roctx

    session = MlpTP2GenerationSession(
        str(args.model),
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=int(args.capacity),
        reduce_mode=str(args.reduce_mode),
    )
    rng = np.random.default_rng(11)
    prompt = [int(i) for i in rng.integers(1000, 50000, size=int(args.prompt_tokens))]
    layer_count = len(session._config.layer_types)

    # Warmup: graph capture, first-touch pages and JIT loads all happen here, so
    # the measured region contains steady-state decode only. Synchronize so no
    # warmup kernel can be attributed to the region.
    session.generate(prompt, max_new_tokens=2, eos_token_id=None)
    session.runtime.device_synchronize()

    record: dict[str, Any] = {
        "kind": "tp2_decode_kernel_inventory_child",
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "model": str(args.model),
        "model_sha256": _sha256(Path(args.model)) if Path(args.model).exists() else None,
        "route": {
            "mode": session.mode,
            "schedule": session.schedule,
            "driver": session.driver,
            "head_shard": bool(session.head_shard),
            "reduce_mode": session.reduce_mode,
        },
        "capacity": int(args.capacity),
        "prompt_tokens": int(args.prompt_tokens),
        "decode_steps": int(args.steps),
        "layer_count": layer_count,
        "devices": _device_rows(session),
        "marker_prefix": DECODE_MARKER,
        "warm_only": bool(args.warm_only),
    }

    if args.warm_only:
        record["status"] = "warm-only"
        if args.json:
            Path(args.json).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        session.close()
        print(f"warm-only complete: {layer_count} layers, capacity {args.capacity}", flush=True)
        return 0

    marker = _Roctx()
    # The probe runs after a synchronize and before the region, so it is visible
    # in the raw trace but outside the measured decode window.
    record["probe"] = _launch_probe(session)

    # Prefill (and the first decode step) happen outside the measured region:
    # ``generate`` prefills token by token inside its own call, which would put
    # 64 prefill steps in the region and inflate every count by that factor.
    warm = session.generate(prompt, max_new_tokens=1, eos_token_id=None)
    token = int(warm.token_ids[-1])
    position = len(prompt) + 1
    session.runtime.device_synchronize()

    session.runtime.device_synchronize()
    marker.push(DECODE_MARKER + "start")
    started = time.perf_counter()
    tokens: list[int] = []
    for _ in range(int(args.steps)):
        step_logits, _trace = session._forward_token(token, position, kind="decode")
        token = int(np.argmax(step_logits))
        tokens.append(token)
        position += 1
    session.runtime.device_synchronize()
    region_wall = time.perf_counter() - started
    marker.pop()

    record.update(
        {
            "status": "complete",
            "tokens": tokens,
            "region_wall_ms": region_wall * 1e3,
            "region_wall_ms_per_step": region_wall * 1e3 / max(1, int(args.steps)),
            "command": " ".join([sys.executable, *sys.argv]),
        }
    )
    if len(tokens) != int(args.steps):
        raise ValueError(f"expected {args.steps} decode tokens, got {len(tokens)}")
    session.close()
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.json}", flush=True)
    print(
        f"child: {args.steps} decode steps, region {region_wall * 1e3:.3f} ms "
        f"({region_wall * 1e3 / max(1, int(args.steps)):.3f} ms/step)",
        flush=True,
    )
    return 0


def _agent_rank_map(
    kernels: list[dict[str, Any]], probe: dict[str, Any]
) -> tuple[dict[str, int], str]:
    """Map trace ``Agent_Id`` to a rank from the labeled probe launches."""

    by_agent: dict[str, dict[int, int]] = collections.defaultdict(dict)
    for row in kernels:
        fragment = str(probe.get("kernel_fragment") or "")
        if not fragment or fragment not in str(row.get("kernel")):
            continue
        grid = row.get("grid_x")
        agent = row.get("agent_id")
        if agent is None or not grid:
            continue
        by_agent[str(agent)][int(grid)] = by_agent[str(agent)].get(int(grid), 0) + 1
    if len(by_agent) != len(probe.get("launches", [])):
        return {}, (
            f"probe kernel appears on {len(by_agent)} agents, expected "
            f"{len(probe.get('launches', []))}; per-rank split not reported"
        )
    # The two launches differ in element count, so the agent holding the smaller
    # grid is the rank that launched the smaller probe.
    grids = sorted({grid for entry in by_agent.values() for grid in entry})
    if len(grids) < 2:
        return {}, (
            f"probe grids {grids} do not separate the ranks; per-rank split not "
            "reported"
        )
    mapping: dict[str, int] = {}
    for rank, launch in enumerate(probe["launches"]):
        # The probe with more elements produces the larger grid.
        wanted = grids[-1] if rank == len(probe["launches"]) - 1 else grids[0]
        holders = [agent for agent, entry in by_agent.items() if wanted in entry]
        if len(holders) != 1:
            return {}, (
                f"grid {wanted} is ambiguous across agents {holders}; per-rank "
                "split not reported"
            )
        mapping[str(holders[0])] = rank
    return mapping, "labeled probe: distinct grid size per rank"


def _summarize(rows: list[dict[str, Any]], *, steps: int, layers: int) -> dict[str, Any]:
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
    entries = []
    for entry in by_kernel.values():
        calls = entry["calls"]
        total_ns = entry["total_ns"]
        durations = np.array(sorted(entry["durations"]), dtype=np.float64) / 1e3
        # min versus median separates a kernel that does work from one that waits:
        # a spin kernel's median carries the peer's arrival latency, its minimum
        # is close to the cost of the operation itself.
        entries.append(
            {
                "kernel": entry["kernel"],
                "calls": calls,
                "calls_per_step": calls / steps,
                "calls_per_layer": calls / steps / layers,
                "ms_per_step": total_ns / 1e6 / steps,
                "us_per_call": total_ns / 1e3 / calls,
                "us_per_layer": total_ns / 1e3 / steps / layers,
                "min_us": float(durations[0]),
                "p50_us": float(np.percentile(durations, 50)),
                "p90_us": float(np.percentile(durations, 90)),
                "max_us": float(durations[-1]),
            }
        )
    entries.sort(key=lambda item: -item["ms_per_step"])
    small = [entry for entry in entries if entry["us_per_call"] < SMALL_KERNEL_US]
    return {
        "kernel_calls": len(rows),
        "kernel_calls_per_step": len(rows) / steps,
        "copy_calls_per_step": len(copies) / steps,
        "copy_ms_per_step": sum(row["duration_ns"] for row in copies) / 1e6 / steps,
        "kernel_ms_per_step": sum(row["duration_ns"] for row in rows) / 1e6 / steps,
        "kernels": entries,
        "small_kernel_count": len(small),
        "small_kernel_calls_per_step": sum(e["calls_per_step"] for e in small),
        "small_kernel_ms_per_step": sum(e["ms_per_step"] for e in small),
    }


def _read_region_window(path: Path, prefix: str) -> tuple[int, int]:
    """Start/end timestamp of the single ROCTX region named ``<prefix>...``.

    The shared marker reader requires a numeric suffix after its prefix, and this
    region is deliberately named for a human reader, so it is parsed here.
    """

    import csv

    windows: list[tuple[int, int]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = (
                row.get("Function")
                or row.get("Marker_Name")
                or row.get("Marker_Text")
                or row.get("Name")
                or ""
            ).strip()
            if not name.startswith(prefix):
                continue
            try:
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end >= start:
                windows.append((start, end))
    if not windows:
        raise ValueError(f"no {prefix} marker region in {path}")
    if len(windows) > 1:
        raise ValueError(f"expected one {prefix} region in {path}, found {len(windows)}")
    return windows[0]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_kernel_rows(path: Path) -> list[dict[str, Any]]:
    """Kernel rows plus the agent and grid columns the rank map needs.

    Read in one pass: joining a filtered helper's output against a second pass
    over the same CSV would misalign as soon as the helper drops a row.
    """

    import csv

    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end < start:
                continue
            rows.append(
                {
                    "kernel": (
                        row.get("Kernel_Name")
                        or row.get("KernelName")
                        or row.get("Name")
                        or ""
                    ).strip(),
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                    "agent_id": (row.get("Agent_Id") or "").strip() or None,
                    "grid_x": _float_or_none(row.get("Grid_Size_X")),
                    "vgpr": _float_or_none(row.get("VGPR_Count")),
                    "scratch": _float_or_none(row.get("Scratch_Size")),
                }
            )
    return rows


def _rollup(raw_root: Path, child: dict[str, Any], *, top: int) -> dict[str, Any]:
    kernel_csv = _single_file(raw_root, "*_kernel_trace.csv")
    marker_csv = _single_file(raw_root, "*_marker_api_trace.csv")
    kernels = _read_kernel_rows(kernel_csv)
    region_start, region_end = _read_region_window(marker_csv, DECODE_MARKER)
    # Attribute by start timestamp: the region opens after a synchronize, so any
    # kernel starting inside it belongs to the measured decode steps.
    in_region = [
        row for row in kernels if region_start <= row["start_ns"] <= region_end
    ]
    if not in_region:
        raise ValueError("decode region contains no kernels")

    steps = int(child["decode_steps"])
    layers = int(child["layer_count"])
    mapping, mapping_note = _agent_rank_map(kernels, child.get("probe") or {})

    report: dict[str, Any] = {
        "kind": "tp2_decode_kernel_inventory",
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": child.get("host"),
        "model": child.get("model"),
        "model_sha256": child.get("model_sha256"),
        "route": child.get("route"),
        "capacity": child.get("capacity"),
        "prompt_tokens": child.get("prompt_tokens"),
        "decode_steps": steps,
        "layer_count": layers,
        "devices": child.get("devices"),
        "region_wall_ms": child.get("region_wall_ms"),
        "region_wall_ms_per_step": child.get("region_wall_ms_per_step"),
        "probe": child.get("probe"),
        "agent_rank_map": {agent: rank for agent, rank in mapping.items()},
        "agent_rank_source": mapping_note,
        "marker_window_ns": [region_start, region_end],
        "small_kernel_us_threshold": SMALL_KERNEL_US,
        "combined": _summarize(in_region, steps=steps, layers=layers),
        "per_rank": {},
        "commands": {
            "child": child.get("command"),
            "rollup": " ".join([sys.executable, *sys.argv]),
        },
        "traces": {
            "kernel_csv": str(kernel_csv),
            "marker_csv": str(marker_csv),
            "kernel_rows": len(kernels),
            "kernel_rows_in_region": len(in_region),
        },
        "notes": [
            "Launch counts and per-kernel durations come from a rocprofv3 kernel "
            "trace; the region wall is a profiled wall and is not a performance "
            "measurement (quote the unprofiled production cell instead).",
            "Start timestamps of graphed kernels are dispatch-clustered, so "
            "timestamp gaps between kernels are not interpreted here.",
            "The per-rank split is verified, not assumed: one probe launch per "
            "rank with a distinct element count maps Agent_Id to a rank from the "
            "trace, and an unverifiable mapping is reported as unavailable.",
            "A spin/exchange kernel's duration includes waiting for its peer, so "
            "its tail is rank skew being absorbed rather than work; min and p50 "
            "bound the operation itself.",
            "calls_per_layer averages over all layers; a kernel that runs once "
            "per full-attention layer shows a quarter of that value.",
        ],
    }
    for agent, rank in sorted(mapping.items(), key=lambda item: item[1]):
        rows = [row for row in in_region if row.get("agent_id") == agent]
        report["per_rank"][str(rank)] = _summarize(rows, steps=steps, layers=layers)
        report["per_rank"][str(rank)]["agent_id"] = agent
        report["per_rank"][str(rank)]["name"] = (
            (child.get("devices") or {}).get(str(rank), {}).get("name")
        )
    if report["region_wall_ms_per_step"]:
        report["kernel_share_of_wall"] = (
            report["combined"]["kernel_ms_per_step"]
            / report["region_wall_ms_per_step"]
        )
    else:
        report["kernel_share_of_wall"] = None
    report["top"] = report["combined"]["kernels"][: int(top)]
    return report


def _print_report(report: dict[str, Any], *, top: int) -> None:
    combined = report["combined"]
    print(
        f"decode region: {report['decode_steps']} steps, "
        f"{report['region_wall_ms_per_step']:.3f} ms/step wall, "
        f"{combined['kernel_calls_per_step']:.1f} kernel calls/step, "
        f"{combined['kernel_ms_per_step']:.3f} ms/step kernel time "
        f"({(report['kernel_share_of_wall'] or 0) * 100:.1f}% of wall)"
    )
    print(
        f"small kernels (<{SMALL_KERNEL_US:.0f} us mean): "
        f"{combined['small_kernel_count']} distinct, "
        f"{combined['small_kernel_calls_per_step']:.1f} calls/step, "
        f"{combined['small_kernel_ms_per_step']:.3f} ms/step"
    )
    if report["per_rank"]:
        print(f"per-rank split: {report['agent_rank_source']}")
        for rank, entry in sorted(report["per_rank"].items()):
            print(
                f"  rank {rank} (agent {entry['agent_id']}, {entry['name']}): "
                f"{entry['kernel_calls_per_step']:.1f} calls/step, "
                f"{entry['kernel_ms_per_step']:.3f} ms/step kernel, "
                f"small {entry['small_kernel_ms_per_step']:.3f} ms/step"
            )
    else:
        print(f"per-rank split unavailable: {report['agent_rank_source']}")
    print()
    header = (
        f"{'kernel':52s} {'/step':>7s} {'us/layer':>9s} {'min':>7s} {'p50':>7s} "
        f"{'p90':>7s} {'max':>8s}"
    )
    print(header)
    for entry in report["combined"]["kernels"][: int(top)]:
        name = entry["kernel"]
        if len(name) > 50:
            name = name[:47] + "..."
        print(
            f"{name:52s} {entry['calls_per_step']:7.1f} {entry['us_per_layer']:9.2f} "
            f"{entry['min_us']:7.2f} {entry['p50_us']:7.2f} {entry['p90_us']:7.2f} "
            f"{entry['max_us']:8.2f}"
        )


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "-C", "/home/lhl/hipEngine-main", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def _git_status() -> str | None:
    result = subprocess.run(
        ["git", "-C", "/home/lhl/hipEngine-main", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def _run(command: list[str], *, env: dict[str, str], cwd: str) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=cwd, env=env)
    if result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(command)}")


def _parent(args: argparse.Namespace) -> int:
    root = "/home/lhl/hipEngine-main"
    out = Path(args.json) if args.json else Path("/tmp/tp2-decode-kernel-inventory.json")
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
        "scripts/tp2_decode_kernel_inventory.py",
        "--child",
        "--model",
        str(args.model),
        "--capacity",
        str(int(args.capacity)),
        "--steps",
        str(int(args.steps)),
        "--prompt-tokens",
        str(int(args.prompt_tokens)),
        "--reduce-mode",
        str(args.reduce_mode),
        "--json",
        str(child_json),
    ]

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
            "tp2-decode-inventory",
            "--",
            *child,
        ],
        env=env,
        cwd=root,
    )

    child_record = json.loads(child_json.read_text())
    report = _rollup(raw_root, child_record, top=int(args.top))
    report["source_revision"] = _git_revision()
    report["git_status_porcelain"] = _git_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    _print_report(report, top=int(args.top))
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--steps", type=int, default=8, help="decode steps inside the measured region")
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--reduce-mode", choices=("device", "host"), default="device")
    parser.add_argument("--child", action="store_true", help="run the profiled leaf only")
    parser.add_argument("--warm-only", action="store_true", help="build and warm the session, then exit")
    parser.add_argument("--rollup-only", type=Path, default=None, help="re-roll an existing trace directory")
    parser.add_argument("--child-json", type=Path, default=Path("/tmp/tp2-decode-inventory-child.json"))
    parser.add_argument("--raw-root", type=Path, default=Path("/tmp/tp2-decode-inventory-raw"))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--rocprofv3", default=shutil.which("rocprofv3") or "rocprofv3")
    parser.add_argument("--roctx-sdk", type=Path, default=DEFAULT_ROCTX_SDK)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    args = parser.parse_args()

    if args.child or args.warm_only:
        return _child(args)
    if args.rollup_only is not None:
        child_record = json.loads(args.child_json.read_text())
        report = _rollup(args.rollup_only, child_record, top=int(args.top))
        report["source_revision"] = _git_revision()
        report["git_status_porcelain"] = _git_status()
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        _print_report(report, top=int(args.top))
        if args.json:
            print(f"\nwrote {args.json}")
        return 0
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
