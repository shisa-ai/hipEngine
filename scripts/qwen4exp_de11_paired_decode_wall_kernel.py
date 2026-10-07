#!/usr/bin/env python3
"""Join a decode wall and a decode kernel sum captured in one interleaved session.

`DE-11` in `docs/QWEN4EXP-OPTIMIZATION-LEDGER.md` reports the decode non-kernel
remainder as `wall - kernel_sum`, where the wall came from one session and the
kernel sum from another. A sequential `code-p4096` comparison on this host drifts
up to 8%, so that subtraction is not reliable. This tool re-derives the pair from
artifacts captured in one session.

Two producers, both the same canonical harness and the same model/quant/profile:

* the **wall** is the unprofiled
  `scripts/qwen4exp_canonical_ar_bench.py hipengine` artifact; the per-step wall is
  `samples[].decode_ms / decode_transitions` (the synchronized 128-step loop), and
  the reported wall is the median over the measured repetitions;
* the **kernel sum** is the same command run under
  `rocprofv3 --kernel-trace -f csv`, whose kernel-dispatch timestamps are on
  `CLOCK_MONOTONIC` and therefore directly comparable to the harness's
  `phase_windows_ns["decode"]`. The kernel sum is the total `DurationNs` of the
  dispatches contained in that decode window, divided by `decode_transitions`.

The profiled capture serializes launches, so its absolute wall is inflated and is
never reported as the engine wall. Only the window-bounded kernel sum is taken
from it.

Subcommands:

* ``wall`` - read a canonical bench artifact and emit per-step wall statistics.
* ``kernel`` - read a canonical bench artifact plus its rocprofv3 kernel trace and
  emit the decode-window kernel sum per step.
* ``pair`` - join one wall and one kernel artifact for the same case into a
  remainder row.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gguf_decode_rocprof import (  # noqa: E402
    _filter_kernels_by_windows,
    _read_kernels,
)

DEFAULT_STEPS = 128


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def decode_steps(artifact: Mapping[str, Any]) -> int:
    protocol = artifact.get("protocol")
    if isinstance(protocol, Mapping) and protocol.get("decode_transitions"):
        return int(protocol["decode_transitions"])
    return DEFAULT_STEPS


def wall_rows(artifact: Mapping[str, Any]) -> list[dict[str, float]]:
    """Return one per-step wall row per measured sample, in capture order."""

    steps = decode_steps(artifact)
    rows: list[dict[str, float]] = []
    for sample in artifact.get("samples") or []:
        decode_ms = float(sample["decode_ms"])
        rows.append(
            {
                "repetition": int(sample["repetition"]),
                "decode_ms": decode_ms,
                "ms_per_step": decode_ms / steps,
                "decode_tok_s": 1000.0 * steps / decode_ms,
                "prefill_tok_s": float(sample["prefill_tok_s"]),
            }
        )
    if not rows:
        raise ValueError("artifact contains no measured samples")
    return rows


def summarize_wall(artifact: Mapping[str, Any]) -> dict[str, Any]:
    rows = wall_rows(artifact)
    per_step = [row["ms_per_step"] for row in rows]
    median = statistics.median(per_step)
    spread = (max(per_step) - min(per_step)) / median if median else 0.0
    case_ids = sorted({str(row) for row in (artifact.get("protocol") or {}).get("case_ids", [])})
    return {
        "kind": "qwen4exp_de11_decode_wall",
        "case_id": case_ids[0] if len(case_ids) == 1 else case_ids,
        "decode_transitions": decode_steps(artifact),
        "measured_repetitions": len(rows),
        "wall_ms_per_step_median": median,
        "wall_ms_per_step_samples": per_step,
        "within_arm_spread": spread,
        "decode_tok_s_median": statistics.median([row["decode_tok_s"] for row in rows]),
        "source_head": (artifact.get("source") or {}).get("head"),
        "profile_manifest_sha256": (artifact.get("profile") or {}).get("manifest_sha256"),
        "host": (artifact.get("host") or {}).get("machine_id"),
    }


def _last_measured_decode_window(artifact: Mapping[str, Any]) -> tuple[int, int]:
    samples = artifact.get("samples") or []
    if not samples:
        raise ValueError("artifact contains no measured samples")
    sample = samples[-1]
    window = (sample.get("phase_windows_ns") or {}).get("decode")
    if not window or len(window) != 2:
        raise ValueError("measured sample is missing phase_windows_ns.decode")
    return int(window[0]), int(window[1])


def summarize_kernel(
    artifact: Mapping[str, Any],
    trace_path: Path,
) -> dict[str, Any]:
    steps = decode_steps(artifact)
    start_ns, end_ns = _last_measured_decode_window(artifact)
    kernels = _read_kernels(trace_path)
    inside = _filter_kernels_by_windows(kernels, [(start_ns, end_ns)])
    if not inside:
        raise ValueError(
            f"no kernel dispatches fall inside the decode window "
            f"[{start_ns}, {end_ns}]; is the trace from the same process as the artifact?"
        )
    total_ns = sum(int(row["duration_ns"]) for row in inside)
    families: dict[str, dict[str, float]] = {}
    for row in inside:
        name = str(row["kernel"])
        entry = families.setdefault(name, {"ms": 0.0, "n": 0.0})
        entry["ms"] += int(row["duration_ns"]) / 1_000_000.0
        entry["n"] += 1.0
    case_ids = sorted({str(row) for row in (artifact.get("protocol") or {}).get("case_ids", [])})
    return {
        "kind": "qwen4exp_de11_decode_kernel_sum",
        "case_id": case_ids[0] if len(case_ids) == 1 else case_ids,
        "decode_transitions": steps,
        "decode_window_ns": [start_ns, end_ns],
        "span_s": (end_ns - start_ns) / 1_000_000_000.0,
        "launches": len(inside),
        "launches_per_step": len(inside) / steps,
        "kernel_ms": total_ns / 1_000_000.0,
        "kernel_ms_per_step": total_ns / 1_000_000.0 / steps,
        "trace": str(trace_path),
        "source_head": (artifact.get("source") or {}).get("head"),
        "profile_manifest_sha256": (artifact.get("profile") or {}).get("manifest_sha256"),
        "families": {name: {"ms": value["ms"], "n": value["n"]} for name, value in families.items()},
    }


def pair_row(
    wall: Mapping[str, Any],
    kernel: Mapping[str, Any],
) -> dict[str, Any]:
    if wall.get("case_id") != kernel.get("case_id"):
        raise ValueError(f"case mismatch: {wall.get('case_id')!r} vs {kernel.get('case_id')!r}")
    wall_step = float(wall["wall_ms_per_step_median"])
    kernel_step = float(kernel["kernel_ms_per_step"])
    return {
        "case_id": wall.get("case_id"),
        "wall_ms_per_step": wall_step,
        "kernel_ms_per_step": kernel_step,
        "remainder_ms_per_step": wall_step - kernel_step,
        "kernel_share": kernel_step / wall_step if wall_step else None,
        "launches_per_step": kernel.get("launches_per_step"),
        "source_head": wall.get("source_head"),
        "profile_manifest_sha256": wall.get("profile_manifest_sha256"),
        "host": wall.get("host"),
    }


def session_summary(session_dir: Path) -> dict[str, Any]:
    """Join every wall and kernel artifact in one interleaved session directory."""

    wall_runs: dict[str, list[dict[str, Any]]] = {}
    kernel_runs: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((session_dir / "wall").glob("*.json")):
        row = json.loads(path.read_text())
        wall_runs.setdefault(str(row["case_id"]), []).append(row)
    for path in sorted((session_dir / "kernel").glob("*.json")):
        row = json.loads(path.read_text())
        kernel_runs.setdefault(str(row["case_id"]), []).append(row)
    cases: dict[str, Any] = {}
    for case_id in sorted(set(wall_runs) | set(kernel_runs)):
        walls = [float(row["wall_ms_per_step_median"]) for row in wall_runs.get(case_id, [])]
        kernels = [float(row["kernel_ms_per_step"]) for row in kernel_runs.get(case_id, [])]
        if not walls or not kernels:
            raise ValueError(f"{case_id}: need at least one wall and one kernel run")
        wall_median = statistics.median(walls)
        kernel_median = statistics.median(kernels)
        cases[case_id] = {
            "wall_ms_per_step_median": wall_median,
            "wall_ms_per_step_runs": walls,
            "kernel_ms_per_step_median": kernel_median,
            "kernel_ms_per_step_runs": kernels,
            "remainder_ms_per_step": wall_median - kernel_median,
            "kernel_share": kernel_median / wall_median if wall_median else None,
            "launches_per_step": [float(row["launches_per_step"]) for row in kernel_runs[case_id]],
        }
    return {
        "kind": "qwen4exp_de11_paired_decode_session",
        "session_dir": str(session_dir),
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    wall_parser = sub.add_parser("wall", help="Summarize the decode wall of a bench artifact")
    wall_parser.add_argument("artifact", type=Path)
    wall_parser.add_argument("--output", type=Path)

    kernel_parser = sub.add_parser("kernel", help="Summarize the decode-window kernel sum")
    kernel_parser.add_argument("artifact", type=Path)
    kernel_parser.add_argument("--trace", type=Path, required=True)
    kernel_parser.add_argument("--output", type=Path)

    pair_parser = sub.add_parser("pair", help="Join a wall and a kernel-sum artifact")
    pair_parser.add_argument("wall", type=Path)
    pair_parser.add_argument("kernel", type=Path)
    pair_parser.add_argument("--output", type=Path)

    session_parser = sub.add_parser("session", help="Join a whole interleaved session directory")
    session_parser.add_argument("session_dir", type=Path)
    session_parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "wall":
        payload = summarize_wall(json.loads(args.artifact.read_text()))
    elif args.mode == "kernel":
        payload = summarize_kernel(json.loads(args.artifact.read_text()), args.trace)
    elif args.mode == "pair":
        payload = pair_row(
            json.loads(args.wall.read_text()),
            json.loads(args.kernel.read_text()),
        )
    else:
        payload = session_summary(args.session_dir)
    if getattr(args, "output", None) is not None:
        _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
