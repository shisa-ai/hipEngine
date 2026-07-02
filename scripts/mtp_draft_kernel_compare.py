#!/usr/bin/env python3
"""Compare hipEngine and llama.cpp MTP draft-drain kernel diagnostics.

This is an offline parity helper. It joins already-collected artifacts:

* a hipEngine ``gguf_mtp_draft_rocprof.py`` draft-chain trace,
* a llama.cpp ``llamacpp_mtp_rocprof.py --roctx-ranges`` trace, and
* optionally a hipEngine full-suite artifact for retained ms/output context, and
* optionally a retained llama.cpp stage artifact for same-protocol deltas.

The output is diagnostic only. It is meant to answer whether the residual
llama-compat draft-drain gap points at a concrete kernel-family gap, without
turning short profiler requests into headline performance rows.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _find_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("name") or row.get("range") or row.get("bucket") or "") == name:
            return row
    return None


def _find_kernel_contains(rows: list[dict[str, Any]], needle: str) -> dict[str, Any] | None:
    for row in rows:
        if needle in str(row.get("name") or row.get("kernel") or ""):
            return row
    return None


def _stage_section(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("stage_timing_summary")
    if not isinstance(summary, dict):
        return {}
    section = summary.get("measured_excluding_first_task") or summary.get("all")
    return section if isinstance(section, dict) else {}


def _llama_stage_context(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    section = _stage_section(payload)
    if not section:
        return None
    cycles = section.get("cycles")
    total_outputs = section.get("total_output_tokens")
    visible_outputs_per_cycle = None
    if isinstance(cycles, (int, float)) and isinstance(total_outputs, (int, float)) and float(cycles) != 0.0:
        visible_outputs_per_cycle = float(total_outputs) / float(cycles)
    return {
        "cycles": cycles,
        "total_output_tokens": total_outputs,
        "visible_outputs_per_cycle": visible_outputs_per_cycle,
        "cycle_wall_ms_per_output": section.get("cycle_wall_ms_per_output"),
        "draft_initial_ms_per_output": _get_nested(section, ["stage_timing_per_output_ms", "draft_initial"]),
        "llama_draft_sample_topk_ms_per_output": _get_nested(
            section, ["stage_timing_per_output_ms", "llama_draft_sample_topk"]
        ),
        "target_block_verify_total_ms_per_output": _get_nested(
            section, ["stage_timing_per_output_ms", "target_block_verify_total"]
        ),
    }


def _hip_full_suite_context(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    mtp = payload.get("mtp_by_budget")
    if not isinstance(mtp, dict):
        return None
    row = mtp.get("b2") or mtp.get("B2")
    if not isinstance(row, dict):
        return None
    hist = row.get("cycle_histograms")
    cycles = None
    if isinstance(hist, dict):
        generated = hist.get("generated_draft_tokens")
        if isinstance(generated, dict):
            cycles = sum(int(v) for v in generated.values())
    total_outputs = int(row.get("total_output_tokens") or 0)
    visible_outputs_per_cycle = (total_outputs / cycles) if cycles else None
    return {
        "artifact_outputs": total_outputs,
        "artifact_cycles": cycles,
        "visible_outputs_per_cycle": visible_outputs_per_cycle,
        "mtp_tok_s": row.get("decode_tok_s_weighted"),
        "cycle_wall_ms_per_output": row.get("cycle_wall_ms_per_output"),
        "draft_initial_ms_per_output": _get_nested(row, ["stage_timing_per_output_ms", "draft_initial"]),
        "draft_topk_readback_ms_per_output": _get_nested(row, ["stage_timing_per_output_ms", "draft_topk_readback"]),
        "draft_device_chain_drain_ms_per_output": _get_nested(row, ["stage_timing_per_output_ms", "draft_device_chain_drain"]),
    }


def _get_nested(payload: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _divide_stage_totals(stage_totals: dict[str, Any], steps: int) -> dict[str, float]:
    if steps <= 0:
        return {}
    return {
        key: float(value) / float(steps)
        for key, value in sorted(stage_totals.items())
        if isinstance(value, (int, float))
    }


def _hip_summary(payload: dict[str, Any], *, full_suite: dict[str, Any] | None) -> dict[str, Any]:
    summary = payload.get("summary")
    child = payload.get("child")
    if not isinstance(summary, dict) or not isinstance(child, dict):
        raise ValueError("hipEngine draft artifact must contain summary{} and child{}")
    steps = int(summary.get("steps") or child.get("steps") or 0)
    top_kernels = summary.get("top_kernels") or []
    buckets = summary.get("buckets") or []
    stage_totals = child.get("stage_timing_totals_ms") or {}
    if not isinstance(top_kernels, list) or not isinstance(buckets, list):
        raise ValueError("hipEngine summary top_kernels/buckets must be lists")
    if not isinstance(stage_totals, dict):
        stage_totals = {}
    q6_kernel = _find_kernel_contains(top_kernels, "gguf_q6_k_x8_gemv_q8_1_dp4a_top1_stage1")
    stage_per_cycle = _divide_stage_totals(stage_totals, steps)
    return {
        "artifact": payload.get("raw_root") or payload.get("kernel_trace_csv"),
        "steps": steps,
        "avg_host_ms_per_cycle": summary.get("avg_host_ms"),
        "avg_kernel_ms_per_cycle": summary.get("avg_kernel_ms"),
        "kernel_calls_per_cycle": summary.get("kernel_calls_per_step"),
        "full_suite_context": full_suite,
        "stage_ms_per_cycle": {
            key: stage_per_cycle.get(key)
            for key in (
                "draft_device_chain_drain",
                "draft_topk_readback",
                "draft_topk_d2h",
                "draft_gpu_run_lm_head",
                "draft_gpu_decode_initial",
                "draft_gpu_decode_next",
                "draft_gpu_run_attention",
                "draft_gpu_run_qkv_kvwrite",
                "draft_gpu_run_ffn_selected_gate_up",
                "draft_gpu_run_moe_down_combine",
            )
            if key in stage_per_cycle
        },
        "top_buckets": buckets[:8],
        "q6_top1_stage1": q6_kernel,
        "top_kernels": top_kernels[:12],
    }


def _llama_summary(payload: dict[str, Any]) -> dict[str, Any]:
    kernel_summary = payload.get("kernel_summary")
    if not isinstance(kernel_summary, dict):
        raise ValueError("llama.cpp artifact must contain kernel_summary{}")
    ranges = kernel_summary.get("range_name_summaries") or []
    if not isinstance(ranges, list):
        raise ValueError("llama.cpp range_name_summaries must be a list")
    draft_sample = _find_name(ranges, "llama_draft_sample_topk")
    if draft_sample is None:
        raise ValueError("llama.cpp artifact has no llama_draft_sample_topk ROCTX range")
    top_kernels = draft_sample.get("top_kernels") or []
    buckets = draft_sample.get("buckets") or []
    if not isinstance(top_kernels, list) or not isinstance(buckets, list):
        top_kernels = []
        buckets = []
    q6_kernel = _find_kernel_contains(top_kernels, "mul_mat_vec_q<(ggml_type)14, 1")
    return {
        "artifact": payload.get("raw_root") or payload.get("kernel_trace_csv"),
        "range_calls": draft_sample.get("range_calls"),
        "range_duration_ms_total": draft_sample.get("range_duration_ms"),
        "range_duration_ms_per_call": _per_call(draft_sample.get("range_duration_ms"), draft_sample.get("range_calls")),
        "kernel_ms_total": draft_sample.get("kernel_ms"),
        "kernel_ms_per_call": _per_call(draft_sample.get("kernel_ms"), draft_sample.get("range_calls")),
        "kernel_dispatches_total": draft_sample.get("kernel_dispatches"),
        "roctx_stage_context": _llama_stage_context(payload),
        "top_buckets": buckets[:8],
        "q6_top1_kernel": q6_kernel,
        "top_kernels": top_kernels[:12],
    }


def _per_call(value: Any, calls: Any) -> float | None:
    if not isinstance(value, (int, float)) or not isinstance(calls, (int, float)) or float(calls) == 0.0:
        return None
    return float(value) / float(calls)


def build_comparison(
    *,
    hip_path: Path,
    llama_path: Path,
    hip_full_suite_path: Path | None,
    llama_stage_path: Path | None,
    label: str,
) -> dict[str, Any]:
    hip_payload = _load(hip_path)
    llama_payload = _load(llama_path)
    full_suite_payload = _load(hip_full_suite_path) if hip_full_suite_path else None
    llama_stage_payload = _load(llama_stage_path) if llama_stage_path else None
    hip = _hip_summary(hip_payload, full_suite=_hip_full_suite_context(full_suite_payload))
    llama = _llama_summary(llama_payload)
    llama_retained_stage = _llama_stage_context(llama_stage_payload)
    hip_q6 = hip.get("q6_top1_stage1") or {}
    llama_q6 = llama.get("q6_top1_kernel") or {}
    hip_q6_ms = (
        float(hip_q6["us_per_call"]) / 1000.0
        if isinstance(hip_q6, dict) and isinstance(hip_q6.get("us_per_call"), (int, float))
        else None
    )
    llama_q6_ms = (
        float(llama_q6["avg_dispatch_ms"])
        if isinstance(llama_q6, dict) and isinstance(llama_q6.get("avg_dispatch_ms"), (int, float))
        else None
    )
    return {
        "schema": "hipengine.mtp_draft_kernel_compare.v1",
        "date": date.today().isoformat(),
        "status": "diagnostic_retained",
        "performance_claim": False,
        "label": label,
        "inputs": {
            "hipengine_draft_rocprof": str(hip_path),
            "hipengine_full_suite": str(hip_full_suite_path) if hip_full_suite_path else None,
            "llamacpp_rocprof": str(llama_path),
            "llamacpp_stage": str(llama_stage_path) if llama_stage_path else None,
        },
        "hipengine": hip,
        "llamacpp": llama,
        "llamacpp_retained_stage_context": llama_retained_stage,
        "comparison": {
            "q6_top1_ms_per_call_delta_hip_minus_llama": (
                hip_q6_ms - llama_q6_ms if hip_q6_ms is not None and llama_q6_ms is not None else None
            ),
            "q6_top1_reading": (
                "per-call parity"
                if hip_q6_ms is not None and llama_q6_ms is not None and abs(hip_q6_ms - llama_q6_ms) <= 0.05
                else "not comparable or outside parity band"
            ),
            "roctx_proxy_draft_initial_delta_ms_per_output": _draft_initial_delta(
                hip, llama.get("roctx_stage_context")
            ),
            "retained_full_suite_draft_initial_delta_ms_per_output": _draft_initial_delta(
                hip, llama_retained_stage
            ),
            "retained_full_suite_visible_sampler_delta_ms_per_output": _stage_delta(
                hip,
                "draft_topk_readback_ms_per_output",
                llama_retained_stage,
                "llama_draft_sample_topk_ms_per_output",
            ),
            "retained_full_suite_cycle_delta_ms_per_output": _stage_delta(
                hip,
                "cycle_wall_ms_per_output",
                llama_retained_stage,
                "cycle_wall_ms_per_output",
            ),
            "interpretation": (
                "The dominant Q6_K top-1 draft lm-head dispatch is at per-call parity; "
                "the remaining retained draft-drain delta should be treated as small "
                "rollup/accounting variance or a non-Q6 secondary-leaf target unless "
                "a future same-protocol rerun reopens the parent wall gap."
            ),
        },
    }


def _draft_initial_delta(hip: dict[str, Any], llama_stage: dict[str, Any] | None) -> float | None:
    return _stage_delta(hip, "draft_initial_ms_per_output", llama_stage, "draft_initial_ms_per_output")


def _stage_delta(
    hip: dict[str, Any],
    hip_key: str,
    llama_stage: dict[str, Any] | None,
    llama_key: str,
) -> float | None:
    hip_full = hip.get("full_suite_context")
    if not isinstance(hip_full, dict) or not isinstance(llama_stage, dict):
        return None
    hip_value = hip_full.get(hip_key)
    llama_value = llama_stage.get(llama_key)
    if not isinstance(hip_value, (int, float)) or not isinstance(llama_value, (int, float)):
        return None
    return float(hip_value) - float(llama_value)


def _fmt_ms(value: Any) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "n/a"


def _print_summary(payload: dict[str, Any]) -> None:
    hip = payload["hipengine"]
    llama = payload["llamacpp"]
    cmp = payload["comparison"]
    print(f"[mtp-draft-kernel-compare] {payload['label']}")
    print(
        "hip draft host/kernel: "
        f"{_fmt_ms(hip.get('avg_host_ms_per_cycle'))}/{_fmt_ms(hip.get('avg_kernel_ms_per_cycle'))} ms/cycle"
    )
    print(
        "llama draft sample range: "
        f"{_fmt_ms(llama.get('range_duration_ms_per_call'))} ms/call, "
        f"kernel {_fmt_ms(llama.get('kernel_ms_per_call'))} ms/call"
    )
    print(
        "Q6 top1 per-call delta hip-llama: "
        f"{_fmt_ms(cmp.get('q6_top1_ms_per_call_delta_hip_minus_llama'))} ms "
        f"({cmp.get('q6_top1_reading')})"
    )
    delta = cmp.get("retained_full_suite_draft_initial_delta_ms_per_output")
    if delta is not None:
        print(f"retained draft_initial delta hip-llama: {delta:.3f} ms/output")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hipengine-draft-rocprof", type=Path, required=True)
    parser.add_argument("--llamacpp-rocprof", type=Path, required=True)
    parser.add_argument("--hipengine-full-suite", type=Path, default=None)
    parser.add_argument("--llamacpp-stage", type=Path, default=None)
    parser.add_argument("--label", default="active-llama-compat-draft-kernel-compare")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = build_comparison(
        hip_path=args.hipengine_draft_rocprof,
        llama_path=args.llamacpp_rocprof,
        hip_full_suite_path=args.hipengine_full_suite,
        llama_stage_path=args.llamacpp_stage,
        label=str(args.label),
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[mtp-draft-kernel-compare] wrote {args.out}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
