#!/usr/bin/env python3
"""X-4's named next step: analyze marker-scoped decode kernel/API/role owners.

`scripts/qwen4exp_context_decode_profile.py` captures the decode transition and its own
`decision.next` reads *"analyze marker-scoped kernel/API/role owners"*. That analysis was
never done, and six `DE-*` rows are parked "after X-4" waiting on it.

This reads a rocprofv3 capture produced by that harness and attributes every decode kernel
to its semantic role, using the same join the established decoder profiler uses:

  1. step windows (`qwen4exp_decode_live4097_rep<N>`) select the measured transitions;
  2. each HIP launch call is joined to the **innermost** role range containing it on the
     same host thread;
  3. kernels are mapped to roles by `correlation_id`.

Attributing via the host-side launch timestamp rather than the device timestamp is what the
existing pipeline does, and it is why the HIP API trace is required.

Diagnostic only. No runtime change, no promotion, and the profiled wall is not a
performance number.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gguf_decode_rocprof import (  # noqa: E402
    _annotate_kernel_roles,
    _filter_kernels_by_windows,
    _read_hip_launches,
    _read_kernels,
    _read_marker_windows,
    _read_role_windows,
)

STEP_PREFIX = "qwen4exp_decode_live4097_rep"
ROLE_PREFIX = "qwen4exp_role:"


def _family(role: str) -> str:
    """Collapse `linear:layers.7.attn_qkv` to `linear:attn_qkv`, keeping root roles whole."""
    if role == "unattributed" or ":" not in role:
        return role
    head, tail = role.split(":", 1)
    parts = tail.split(".")
    if "layers" in parts:
        index = parts.index("layers")
        tail = ".".join(parts[index + 2:]) or tail
    return f"{head}:{tail}"


def _role_from_kernel_name(name: object) -> str | None:
    """Recover a role from a `--kernel-rename`d kernel name.

    Kernels replayed from a captured HIP graph have **no HIP launch API row**, so the
    correlation-id join cannot reach them - measured here as 0 of 1,797 unattributed
    correlation ids appearing in the launch trace. `rocprofv3 --kernel-rename` renames
    each kernel with the enclosing roctx region, so the role is already in the name. This
    fallback is therefore not a heuristic: it recovers an identity the profiler assigned.
    """
    text = str(name)
    prefix = "qwen4exp_role:"
    if not text.startswith(prefix):
        return None
    role = text[len(prefix):].split()[0].strip()
    return role or None


def _resolve_roles(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Apply the kernel-name fallback where the correlation-id join found nothing."""
    output: list[dict[str, Any]] = []
    recovered = 0
    for row in rows:
        annotated = dict(row)
        if annotated.get("role") == "unattributed":
            role = _role_from_kernel_name(annotated.get("kernel"))
            if role is not None:
                annotated["role"] = role
                annotated["role_source"] = "kernel_rename"
                recovered += 1
            else:
                annotated["role_source"] = "none"
        else:
            annotated["role_source"] = "correlation"
        output.append(annotated)
    return output, recovered


def _summarize(rows: Sequence[Mapping[str, Any]], *, steps: int) -> dict[str, Any]:
    by_role: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: {"ms": 0.0, "kernels": 0})
    by_family: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: {"ms": 0.0, "kernels": 0})
    total_ms = 0.0
    for row in rows:
        duration_ms = (int(row["end_ns"]) - int(row["start_ns"])) / 1e6
        role = str(row.get("role", "unattributed"))
        total_ms += duration_ms
        for table, key in ((by_role, role), (by_family, _family(role))):
            table[key]["ms"] += duration_ms
            table[key]["kernels"] += 1
    def ranked(table):
        return [
            {"name": name, "ms_per_step": value["ms"] / steps,
             "kernels_per_step": value["kernels"] / steps,
             "share_pct": (value["ms"] / total_ms * 100.0) if total_ms else 0.0}
            for name, value in sorted(table.items(), key=lambda kv: -kv[1]["ms"])
        ]
    return {
        "total_kernel_ms_per_step": total_ms / steps,
        "roles": ranked(by_role),
        "families": ranked(by_family),
        "unattributed_kernels": sum(
            1 for row in rows if row.get("role") == "unattributed"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--stem", default="x4full")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    def one(suffix: str) -> Path:
        path = args.trace_dir / f"{args.stem}_{suffix}.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        return path

    windows = _read_marker_windows(one("marker_api_trace"), STEP_PREFIX)
    if not windows:
        raise SystemExit(f"no step windows with prefix {STEP_PREFIX!r}")
    all_kernels = _read_kernels(one("kernel_trace"))
    selected = _filter_kernels_by_windows(
        all_kernels, [(start, end) for _index, start, end in windows])
    role_windows = _read_role_windows(one("marker_api_trace"), ROLE_PREFIX)
    launches = _read_hip_launches(one("hip_api_trace"))
    annotated = _annotate_kernel_roles(
        selected, launches=launches, windows=role_windows)
    annotated, recovered = _resolve_roles(annotated)
    steps = len(windows)
    summary = _summarize(annotated, steps=steps)

    report: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_x4_decode_role_attribution",
        "performance_claim": False,
        "promotion_claim": False,
        "purpose": (
            "X-4's named next step - analyze marker-scoped kernel/API/role owners - so the "
            "decode rows parked 'after X-4' have a device split to rank against."
        ),
        "command": sys.argv,
        "source_trace": {
            "stem": args.stem,
            "step_windows": len(windows),
            "role_windows": len(role_windows),
            "hip_launches": len(launches),
            "whole_trace_kernels": len(all_kernels),
            "selected_decode_kernels": len(selected),
        },
        "attribution": {
            "roles_observed": len(summary["roles"]),
            "kernels_unattributed": summary["unattributed_kernels"],
            "kernels_attributed": len(selected) - summary["unattributed_kernels"],
            "recovered_by_kernel_rename": recovered,
            "by_source": {
                source: sum(1 for row in annotated if row["role_source"] == source)
                for source in ("correlation", "kernel_rename", "none")
            },
        },
        "total_kernel_ms_per_step": summary["total_kernel_ms_per_step"],
        "roles_top": summary["roles"][: args.top],
        "families_top": summary["families"][: args.top],
        "limits": (
            "Profiled capture: the wall and per-kernel times are diagnostic, not a "
            "performance measurement. One case (code-p4096), one live count (4097), one "
            "model, one host, one execution profile. Role attribution depends on the "
            "innermost-window join; kernels launched outside any role range are reported "
            "as unattributed rather than assigned."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({
        "step_windows": len(windows),
        "selected_kernels": len(selected),
        "attributed": report["attribution"]["kernels_attributed"],
        "unattributed": report["attribution"]["kernels_unattributed"],
        "total_kernel_ms_per_step": round(summary["total_kernel_ms_per_step"], 4),
    }, indent=1))
    for row in summary["families"][:12]:
        print(f"  {row['name']:34s} {row['ms_per_step']:8.3f} ms/step "
              f"{row['share_pct']:5.2f}%  {row['kernels_per_step']:6.1f} kern/step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
