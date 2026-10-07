#!/usr/bin/env python3
"""DE-2 shared-expert decode attribution from the committed decode capture.

The DE-2 ledger row ends: *"a scoped marker or a name-to-kernel join that this
capture cannot supply"*, and a later note concludes the join is *"structurally
impossible"* because the published X-4 capture renamed every in-window kernel to
a ``qwen4exp_role:`` name. That conclusion holds for the renamed export only. A
**real-kernel-name twin of the same decode protocol is committed**
(``benchmarks/results/2026-09-29-qwen4exp-de1-dp4a-e2e/rocprof/e2ebase_kernel_trace.csv``,
``renamed = 0``), so the join is available.

What the name alone cannot do is separate the shared expert from the rest of the
layer, because the whole layer MoE runs as **one replayed HIP graph** and every
node inherits the enclosing ``moe:layers.N.expert_gate`` role. This script does
the separation by **dispatch signature** instead:

  1. select the ``qwen4exp_decode_live4097_rep`` step windows from the renamed
     capture and keep the rows whose kernel name is ``qwen4exp_role:moe:...``
     (the replayed MoE graph nodes);
  2. read the real-kernel-name twin and attach the real name to each distinct
     dispatch signature ``(Grid_Size_X, Workgroup_Size_X, VGPR, LDS)``;
  3. classify each signature to a tensor owner by that name plus the exact GGUF
     tensor shape (``ffn_gate_shep``/``up``/``down`` Q8_0, ``ffn_gate_inp`` F32
     router, ``ffn_gate_inp_shexp`` F32 shared gate).

The classification is a read of exact shipped payload shapes, not a heuristic
bits-per-weight estimate, and it reproduces two independently published numbers
(the routed gate+up 14.013 and routed down 5.444 ms/step, and the shared combine
0.0671 ms/step) as a self-check.

Read-only. No GPU, no model load.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gguf_decode_rocprof import (  # noqa: E402
    _read_hip_launches,
    _read_marker_windows,
)
from scripts.qwen4exp_de2_shared_expert_pricing import read_tensors  # noqa: E402

DEFAULT_RENAMED_TRACE = (
    ROOT / "benchmarks/results/2026-09-29-qwen4exp-x4-decode-split/rocprof"
)
DEFAULT_NAME_TRACE = (
    ROOT / "benchmarks/results/2026-09-29-qwen4exp-de1-dp4a-e2e/rocprof"
)
DEFAULT_MODEL_ROOT = Path(
    "/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
STEP_PREFIX = "qwen4exp_decode_live4097_rep"
MOE_PREFIX = "qwen4exp_role:moe:"

# A dispatch signature is the profiler's full launch geometry for one kernel.
_SHAPE_FIELDS = (
    "Grid_Size_X",
    "Grid_Size_Y",
    "Grid_Size_Z",
    "Workgroup_Size_X",
    "Workgroup_Size_Y",
    "Workgroup_Size_Z",
    "VGPR_Count",
    "LDS_Block_Size",
)

# Roles that make up the shared expert block (weight-bearing + elementwise).
SHARED_ROLES = (
    "shared_gate_up",
    "shared_down",
    "shared_gate_weight",
    "shared_silu_mul",
    "shared_combine",
)
ROUTED_ROLES = ("routed_gate_up", "routed_down", "routed_selected_misc")
ROUTER_ROLES = ("router_gemv", "router_select")
# f32_to_bf16 / bf16_to_f32 / weighted_sum_out share one dispatch signature, so
# the shared-down conversion cannot be split from the routed weighted sum here.
MIXED_ROLES = ("moe_convert_mixed",)


def _optional_int(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def read_dispatch_rows(path: Path) -> list[dict[str, Any]]:
    """Read kernel rows with their launch geometry."""
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            start = _optional_int(raw.get("Start_Timestamp"))
            end = _optional_int(raw.get("End_Timestamp"))
            if start is None or end is None or end < start:
                continue
            kernel = (
                raw.get("Kernel_Name") or raw.get("KernelName") or raw.get("Name") or ""
            ).strip()
            rows.append(
                {
                    "kernel": kernel,
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                    "correlation_id": _optional_int(raw.get("Correlation_Id")),
                    "signature": tuple(str(raw.get(field, "")) for field in _SHAPE_FIELDS),
                }
            )
    return rows


def select_windows(
    rows: Iterable[Mapping[str, Any]], windows: Sequence[tuple[int, int]]
) -> list[dict[str, Any]]:
    """Keep rows whose device interval lies inside any step window."""
    ordered = sorted(windows)
    return [
        dict(row)
        for row in rows
        if any(
            int(row["start_ns"]) >= start and int(row["end_ns"]) <= end
            for start, end in ordered
        )
    ]


def _grid_x(signature: Sequence[str]) -> int | None:
    return _optional_int(signature[0]) if signature else None


def _wg_x(signature: Sequence[str]) -> int | None:
    return _optional_int(signature[3]) if len(signature) > 3 else None


def classify_signature(real_name: str, signature: Sequence[str]) -> str:
    """Map a (real kernel name, dispatch signature) pair to a tensor owner.

    ``real_name`` comes from the real-kernel-name twin; ``signature`` supplies the
    launch geometry. The GGUF shape cross-check lives in :func:`classify_with_gguf`.
    """
    name = real_name
    gx = _grid_x(signature)
    wx = _wg_x(signature)
    if "gguf_q4_k_selected_dual_silu" in name:
        return "routed_gate_up"
    if "q5_1_selected_weighted_sum" in name:
        return "routed_down"
    if "gguf_k_selected_prefill_out" in name:
        return "routed_selected_misc"
    if "qwen35_router_select" in name:
        return "router_select"
    if "gguf_k_pack8_prefill_out_kernel" in name:
        if gx is not None and wx:
            out_features = (gx // wx) * 8
            if out_features == 640:
                return "shared_gate_up"
            if out_features == 2560:
                return "shared_down"
            return f"pack8_out{out_features}"
        return "pack8_unknown"
    if "dense_gemv" in name:
        if gx is not None and wx:
            out_features = gx // wx
            if out_features == 1:
                return "shared_gate_weight"
            if out_features == 512:
                return "router_gemv"
            return f"dense_gemv_out{out_features}"
        return "dense_gemv_unknown"
    if "silu_mul_f32" in name:
        return "shared_silu_mul"
    if "shared_gate_combine_out" in name:
        return "shared_combine"
    if (
        "f32_to_bf16" in name
        or "bf16_to_f32" in name
        or "weighted_sum_out" in name
    ):
        return "moe_convert_mixed"
    if "silu_mul_separate_out" in name:
        return "misc"
    return f"unclassified:{name}"


def classify_with_gguf(
    real_name: str, signature: Sequence[str], gguf_shapes: Mapping[str, Any]
) -> str:
    """As :func:`classify_signature`, with the shared-expert role asserted against
    the shipped GGUF payload shapes."""
    role = classify_signature(real_name, signature)
    gx = _grid_x(signature)
    wx = _wg_x(signature)
    if role in ("shared_gate_up", "shared_down") and gx is not None and wx:
        out_features = (gx // wx) * 8
        gate = gguf_shapes.get("gate_shexp_out")
        down = gguf_shapes.get("down_shexp_out")
        if role == "shared_gate_up" and out_features != gate:
            raise ValueError(
                f"shared_gate_up expected out={gate}, signature gives {out_features}"
            )
        if role == "shared_down" and out_features != down:
            raise ValueError(
                f"shared_down expected out={down}, signature gives {out_features}"
            )
    if role == "shared_gate_weight":
        if gguf_shapes.get("gate_inp_shexp_out") != 1:
            raise ValueError("shared gate weight must be a 1-wide F32 GEMV")
    if role == "router_gemv":
        if gguf_shapes.get("router_out") != 512:
            raise ValueError("router must project to 512 experts")
    return role


def gguf_role_shapes(model_root: Path) -> dict[str, Any]:
    """Exact out-features of the shared expert and router from the GGUF header."""
    tensors = read_tensors(model_root)
    shapes: dict[str, Any] = {}
    for name, tensor in tensors.items():
        shape = tensor.get("shape") or []
        if len(shape) != 2:
            continue
        if name.endswith("ffn_gate_shexp.weight"):
            shapes["gate_shexp_out"] = int(shape[1])
        elif name.endswith("ffn_down_shexp.weight"):
            shapes["down_shexp_out"] = int(shape[1])
        elif name.endswith("ffn_gate_inp.weight"):
            shapes["router_out"] = int(shape[1])
    for name, tensor in tensors.items():
        if name.endswith("ffn_gate_inp_shexp.weight"):
            shapes["gate_inp_shexp_out"] = 1
    missing = {
        "gate_shexp_out",
        "down_shexp_out",
        "router_out",
        "gate_inp_shexp_out",
    } - set(shapes)
    if missing:
        raise SystemExit(f"GGUF missing expected tensors: {sorted(missing)}")
    return shapes


def summarize(
    moe_rows: Sequence[Mapping[str, Any]],
    name_by_signature: Mapping[tuple[str, ...], str],
    gguf_shapes: Mapping[str, Any],
    *,
    steps: int,
    total_decode_ms_per_step: float,
) -> dict[str, Any]:
    """Aggregate replayed MoE-graph rows into owner buckets."""
    by_signature: dict[tuple[str, ...], dict[str, float]] = collections.defaultdict(
        lambda: {"kernels": 0.0, "ms": 0.0}
    )
    for row in moe_rows:
        sig = tuple(row["signature"])
        by_signature[sig]["kernels"] += 1
        by_signature[sig]["ms"] += int(row["duration_ns"]) / 1e6
    by_role: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: {"kernels": 0.0, "ms": 0.0}
    )
    signature_rows: list[dict[str, Any]] = []
    for sig, value in sorted(by_signature.items(), key=lambda kv: -kv[1]["ms"]):
        real_name = name_by_signature.get(sig, "")
        role = (
            classify_with_gguf(real_name, sig, gguf_shapes)
            if real_name
            else "no_name_match"
        )
        ms_per_step = value["ms"] / steps
        by_role[role]["kernels"] += value["kernels"]
        by_role[role]["ms"] += value["ms"]
        signature_rows.append(
            {
                "role": role,
                "real_kernel": real_name,
                "grid_x": _grid_x(sig),
                "workgroup_x": _wg_x(sig),
                "vgpr": _optional_int(sig[6]) if len(sig) > 6 else None,
                "lds": _optional_int(sig[7]) if len(sig) > 7 else None,
                "kernels_per_step": value["kernels"] / steps,
                "ms_per_step": ms_per_step,
            }
        )
    moe_ms = sum(value["ms"] for value in by_signature.values()) / steps
    roles = {
        role: {
            "ms_per_step": value["ms"] / steps,
            "kernels_per_step": value["kernels"] / steps,
            "pct_of_moe": 100.0 * value["ms"] / sum(v["ms"] for v in by_signature.values()),
            "pct_of_decode_kernel_sum": 100.0 * (value["ms"] / steps) / total_decode_ms_per_step,
        }
        for role, value in sorted(by_role.items(), key=lambda kv: -kv[1]["ms"])
    }

    def group_total(names: Sequence[str]) -> float:
        return sum(roles.get(n, {}).get("ms_per_step", 0.0) for n in names)

    shared_ms = group_total(SHARED_ROLES)
    return {
        "moe_graph_ms_per_step": moe_ms,
        "moe_graph_kernels_per_step": sum(v["kernels"] for v in by_signature.values())
        / steps,
        "roles": roles,
        "signatures": signature_rows,
        "shared_expert_ms_per_step": shared_ms,
        "shared_expert_pct_of_decode_kernel_sum": 100.0
        * shared_ms
        / total_decode_ms_per_step,
        "routed_ms_per_step": group_total(ROUTED_ROLES),
        "router_ms_per_step": group_total(ROUTER_ROLES),
        "mixed_ms_per_step": group_total(MIXED_ROLES),
        "unclassified_ms_per_step": sum(
            value["ms_per_step"]
            for role, value in roles.items()
            if role.startswith("unclassified") or role.endswith("unknown")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renamed-trace-dir", type=Path, default=DEFAULT_RENAMED_TRACE)
    parser.add_argument("--renamed-stem", default="x4full")
    parser.add_argument("--name-trace-dir", type=Path, default=DEFAULT_NAME_TRACE)
    parser.add_argument("--name-stem", default="e2ebase")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    renamed_kernel = args.renamed_trace_dir / f"{args.renamed_stem}_kernel_trace.csv"
    renamed_marker = args.renamed_trace_dir / f"{args.renamed_stem}_marker_api_trace.csv"
    name_kernel = args.name_trace_dir / f"{args.name_stem}_kernel_trace.csv"
    name_marker = args.name_trace_dir / f"{args.name_stem}_marker_api_trace.csv"

    windows = _read_marker_windows(renamed_marker, STEP_PREFIX)
    if not windows:
        raise SystemExit(f"no step windows with prefix {STEP_PREFIX!r}")
    step_pairs = [(start, end) for _index, start, end in windows]
    steps = len(step_pairs)

    renamed_rows = read_dispatch_rows(renamed_kernel)
    selected = select_windows(renamed_rows, step_pairs)
    moe_rows = [row for row in selected if str(row["kernel"]).startswith(MOE_PREFIX)]
    if not moe_rows:
        raise SystemExit("no replayed MoE-graph rows found in the step windows")
    total_decode_ms_per_step = (
        sum(int(row["duration_ns"]) for row in selected) / 1e6 / steps
    )

    # The real-kernel-name twin gives one name per dispatch signature. Restrict it
    # to graph-replayed rows (those with no HIP launch API row) so that a generic
    # signature shared with an eager kernel - e.g. shared_gate_combine_out_kernel
    # vs gated_mean_sigmoid_f32_kernel - resolves to the MoE-graph owner.
    name_windows = _read_marker_windows(name_marker, STEP_PREFIX)
    name_selected = select_windows(
        read_dispatch_rows(name_kernel),
        [(start, end) for _index, start, end in name_windows],
    )
    launched = {
        launch["correlation_id"]
        for launch in _read_hip_launches(args.name_trace_dir / f"{args.name_stem}_hip_api_trace.csv")
    }
    graph_replayed = [
        row for row in name_selected if row.get("correlation_id") not in launched
    ]
    names_by_signature: dict[tuple[str, ...], set[str]] = collections.defaultdict(set)
    for row in graph_replayed:
        names_by_signature[tuple(row["signature"])].add(str(row["kernel"]))
    name_by_signature = {
        sig: sorted(names)[0] for sig, names in names_by_signature.items()
    }
    ambiguous = {
        sig: sorted(names)
        for sig, names in names_by_signature.items()
        if len(names) > 1
    }

    gguf_shapes = gguf_role_shapes(args.model_root)
    summary = summarize(
        moe_rows,
        name_by_signature,
        gguf_shapes,
        steps=steps,
        total_decode_ms_per_step=total_decode_ms_per_step,
    )

    payload = {
        "schema": 1,
        "kind": "qwen4exp-de2-shared-expert-attribution",
        "performance_claim": False,
        "promotion_claim": False,
        "command": sys.argv,
        "model_root": str(args.model_root),
        "renamed_trace": str(renamed_kernel),
        "name_trace": str(name_kernel),
        "name_trace_graph_replayed_kernels": len(graph_replayed),
        "step_windows": steps,
        "selected_decode_kernels": len(selected),
        "moe_graph_kernels": len(moe_rows),
        "total_decode_kernel_ms_per_step": total_decode_ms_per_step,
        "gguf_role_shapes": gguf_shapes,
        "ambiguous_signatures": {str(sig): names for sig, names in ambiguous.items()},
        "result": summary,
        "limits": (
            "Profiled capture: per-kernel times are diagnostic, not a performance "
            "measurement. One case (code-p4096), one live count (4097), one model, one "
            "host, one execution profile. The shared expert runs inside the replayed "
            "per-layer MoE HIP graph, so its kernels carry generic names; the split is "
            "by dispatch signature cross-checked against exact GGUF payload shapes. "
            "f32_to_bf16 / bf16_to_f32 / weighted_sum_out share one signature and are "
            "reported together as moe_convert_mixed."
        ),
    }
    text = json.dumps(payload, indent=1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
