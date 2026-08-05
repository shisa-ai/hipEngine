#!/usr/bin/env python3
"""Run the SH-M2 dedicated-versus-liveness-owned scratch screen.

Both legs retain the frozen 4,096-row full-attention query shape and every
other SH-C0 chunk surface.  The only changed input is the disable-only
``HIPENGINE_GGUF_PREFILL_SCRATCH_LIVENESS_ALIAS`` control.  Each timed child is
wrapped in a 10-ms whole-GTT sampler.  Separate correctness-only children
compare full logits plus hidden/layer/Conv/GDN/live-KV state at
512/4K/32K/64K and a fixed-input four-transition trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402
from scripts.gguf_sh_m1_screen import (  # noqa: E402
    BASELINE_QUERY_ROWS,
    DEFAULT_MODEL,
    FIXED_LAYER_ROWS,
    ScreenError,
    _context_map,
    _load_json,
    _median,
    _parse_length,
    _run_logged,
    _run_with_gtt_sampler,
    _session_buffers_from_run,
    _sha256,
    _state_command,
    _tracked_reclamation,
    _validate_state_child,
    build_benchmark_command,
    chunk_sizes,
    validate_benchmark_leg,
)

KIND = "hipengine_gguf_sh_m2_scratch_liveness_screen"
SCHEMA_VERSION = 1
DEFAULT_WORKLOADS = (512, 4096, 32768, 65536)
MODES = ("dedicated", "liveness")
ENV_NAME = "HIPENGINE_GGUF_PREFILL_SCRATCH_LIVENESS_ALIAS"
_GIB = 1 << 30


def _mode_environment(base: Mapping[str, str], mode: str) -> dict[str, str]:
    if mode not in MODES:
        raise ScreenError(f"unsupported SH-M2 mode: {mode}")
    env = dict(base)
    env[ENV_NAME] = "0" if mode == "dedicated" else "1"
    return env


def _scratch_census(payload: Mapping[str, Any]) -> dict[str, Any]:
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ScreenError("benchmark leg has no runs for scratch census")
    buffers = _session_buffers_from_run(runs[0])
    census = buffers.get("bulk_prefill_scratch_census")
    if not isinstance(census, Mapping):
        raise ScreenError("benchmark leg has no bulk-prefill scratch census")
    try:
        physical = int(census["physical_owner_bytes"])
        logical = int(census["logical_field_bytes"])
        allocation_mode = str(census["allocation_mode"])
        rows = int(census["rows"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScreenError("benchmark leg has malformed scratch census") from exc
    return {
        "allocation_mode": allocation_mode,
        "rows": rows,
        "physical_owner_bytes": physical,
        "physical_owner_gib": physical / _GIB,
        "logical_field_bytes": logical,
        "logical_field_gib": logical / _GIB,
        "arena_bytes": int(census.get("arena_bytes", 0)),
        "dedicated_field_bytes": int(census.get("dedicated_field_bytes", 0)),
        "largest_fields_bytes": dict(census.get("largest_fields_bytes", {})),
        "allocation_offsets": dict(census.get("allocation_offsets", {})),
        "allocation_lifetimes": dict(census.get("allocation_lifetimes", {})),
        "allocation_groups": dict(census.get("allocation_groups", {})),
    }


def validate_benchmark_mode(
    payload: Mapping[str, Any],
    *,
    mode: str,
    prompt_length: int,
    decode_tokens: int,
    warmup_decode_tokens: int,
    warmup_runs: int,
    measured_runs: int,
    expected_token_id: int,
) -> dict[str, Any]:
    validate_benchmark_leg(
        payload,
        query_rows=BASELINE_QUERY_ROWS,
        prompt_length=prompt_length,
        decode_tokens=decode_tokens,
        warmup_decode_tokens=warmup_decode_tokens,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        expected_token_id=expected_token_id,
    )
    census = _scratch_census(payload)
    expected_rows = min(
        ((int(prompt_length) + int(decode_tokens) + int(warmup_decode_tokens) + 1 + 255) // 256) * 256,
        BASELINE_QUERY_ROWS,
    )
    expected_mode = (
        "liveness_aliased"
        if mode == "liveness" and expected_rows >= BASELINE_QUERY_ROWS
        else "dedicated"
    )
    if census["allocation_mode"] != expected_mode:
        raise ScreenError(
            f"{mode} scratch mode {census['allocation_mode']!r} != {expected_mode!r}"
        )
    if int(census["rows"]) != expected_rows:
        raise ScreenError(f"{mode} scratch census changed the frozen row shape")
    return census


def compare_state_children(
    dedicated: Mapping[str, Any],
    liveness: Mapping[str, Any],
    *,
    expected_contexts: Sequence[int],
) -> dict[str, Any]:
    dedicated_rows = _validate_state_child(
        dedicated,
        query_rows=BASELINE_QUERY_ROWS,
        expected_contexts=expected_contexts,
    )
    liveness_rows = _validate_state_child(
        liveness,
        query_rows=BASELINE_QUERY_ROWS,
        expected_contexts=expected_contexts,
    )
    rows: list[dict[str, Any]] = []
    for context in expected_contexts:
        control = dedicated_rows[int(context)]
        candidate = liveness_rows[int(context)]
        comparison = {
            "prompt_length": int(context),
            "prefill_logits_exact": control.get("prefill_logits")
            == candidate.get("prefill_logits"),
            "prefill_state_exact": control.get("prefill_state")
            == candidate.get("prefill_state"),
            "trajectory_exact": control.get("trajectory")
            == candidate.get("trajectory"),
            "final_state_exact": control.get("final_state")
            == candidate.get("final_state"),
        }
        comparison["passed"] = all(
            bool(comparison[name])
            for name in (
                "prefill_logits_exact",
                "prefill_state_exact",
                "trajectory_exact",
                "final_state_exact",
            )
        )
        rows.append(comparison)
    return {
        "passed": all(bool(row["passed"]) for row in rows),
        "contexts": rows,
        "dedicated_scratch_rows": int(dedicated["bulk_prefill_scratch_rows"]),
        "liveness_scratch_rows": int(liveness["bulk_prefill_scratch_rows"]),
        "protocol": (
            "byte-exact FP32 logits plus hidden/layer/Conv/GDN/live-BF16-KV "
            "fingerprints after prefill and the fixed-input decode trajectory"
        ),
    }


def summarize_context(
    *,
    prompt_length: int,
    dedicated: Mapping[str, Any],
    liveness: Mapping[str, Any],
    dedicated_gtt: Mapping[str, Any],
    liveness_gtt: Mapping[str, Any],
) -> dict[str, Any]:
    control_summary = dedicated.get("summary")
    candidate_summary = liveness.get("summary")
    if not isinstance(control_summary, Mapping) or not isinstance(candidate_summary, Mapping):
        raise ScreenError("benchmark leg has no summary")
    control_prefill = _median(control_summary, "prefill_tok_s")
    candidate_prefill = _median(candidate_summary, "prefill_tok_s")
    control_decode = _median(control_summary, "decode_tok_s")
    candidate_decode = _median(candidate_summary, "decode_tok_s")
    control_tracked = _median(control_summary, "tracked_peak_allocated_gib")
    candidate_tracked = _median(candidate_summary, "tracked_peak_allocated_gib")
    control_owned = _median(control_summary, "owned_session_peak_gib")
    candidate_owned = _median(candidate_summary, "owned_session_peak_gib")
    control_gtt = float(dedicated_gtt["peak_gib"])
    candidate_gtt = float(liveness_gtt["peak_gib"])
    values = (
        control_prefill,
        candidate_prefill,
        control_decode,
        candidate_decode,
        control_tracked,
        candidate_tracked,
        control_owned,
        candidate_owned,
        control_gtt,
        candidate_gtt,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ScreenError("benchmark context contains an invalid metric")
    control_scratch = _scratch_census(dedicated)
    candidate_scratch = _scratch_census(liveness)
    return {
        "prompt_length": int(prompt_length),
        "dedicated": {
            "prefill_tok_s": control_prefill,
            "decode_tok_s": control_decode,
            "tracked_peak_gib": control_tracked,
            "owned_session_peak_gib": control_owned,
            "whole_gtt_peak_gib": control_gtt,
            "scratch": control_scratch,
            "reclamation": _tracked_reclamation(dedicated),
        },
        "liveness": {
            "prefill_tok_s": candidate_prefill,
            "decode_tok_s": candidate_decode,
            "tracked_peak_gib": candidate_tracked,
            "owned_session_peak_gib": candidate_owned,
            "whole_gtt_peak_gib": candidate_gtt,
            "scratch": candidate_scratch,
            "reclamation": _tracked_reclamation(liveness),
        },
        "comparison": {
            "prefill_speedup": candidate_prefill / control_prefill,
            "prefill_loss_pct": 100.0 * (control_prefill - candidate_prefill) / control_prefill,
            "decode_speedup": candidate_decode / control_decode,
            "decode_loss_pct": 100.0 * (control_decode - candidate_decode) / control_decode,
            "tracked_savings_gib": control_tracked - candidate_tracked,
            "owned_session_savings_gib": control_owned - candidate_owned,
            "whole_gtt_savings_gib": control_gtt - candidate_gtt,
            "bulk_scratch_savings_gib": (
                int(control_scratch["physical_owner_bytes"])
                - int(candidate_scratch["physical_owner_bytes"])
            )
            / _GIB,
        },
    }


def classify_screen(
    rows: Sequence[Mapping[str, Any]],
    *,
    state_comparison: Mapping[str, Any],
    long_context_min: int = 4096,
    min_tracked_savings_gib: float = 1.0,
    max_prefill_loss_pct: float = 1.0,
    max_decode_loss_pct: float = 1.0,
) -> dict[str, Any]:
    if not rows:
        raise ScreenError("classification requires context rows")
    long_rows = [row for row in rows if int(row["prompt_length"]) >= int(long_context_min)]
    if not long_rows:
        raise ScreenError("classification requires at least one 4K+ row")
    state_exact = state_comparison.get("passed") is True
    lifecycle_exact = all(
        int(row[mode]["reclamation"]["delta_bytes"]) == 0
        for row in rows
        for mode in MODES
    )
    prefill_ok = all(
        float(row["comparison"]["prefill_loss_pct"]) <= float(max_prefill_loss_pct)
        for row in rows
    )
    decode_ok = all(
        float(row["comparison"]["decode_loss_pct"]) <= float(max_decode_loss_pct)
        for row in rows
    )
    tracked_memory_ok = all(
        float(row["comparison"]["tracked_savings_gib"])
        >= float(min_tracked_savings_gib)
        for row in long_rows
    )
    whole_gtt_ok = all(
        float(row["comparison"]["whole_gtt_savings_gib"]) > 0.0
        for row in long_rows
    )
    shape_ok = all(
        int(row["dedicated"]["scratch"]["rows"])
        == int(row["liveness"]["scratch"]["rows"])
        and row["dedicated"]["scratch"]["allocation_mode"] == "dedicated"
        and row["liveness"]["scratch"]["allocation_mode"]
        == (
            "liveness_aliased"
            if int(row["liveness"]["scratch"]["rows"]) >= BASELINE_QUERY_ROWS
            else "dedicated"
        )
        for row in rows
    )
    checks = {
        "state_exact": state_exact,
        "lifecycle_exact": lifecycle_exact,
        "prefill_within_threshold": prefill_ok,
        "decode_within_threshold": decode_ok,
        "tracked_memory_gate_passed": tracked_memory_ok,
        "whole_gtt_direction_passed": whole_gtt_ok,
        "execution_shape_fixed": shape_ok,
    }
    passed = all(checks.values())
    if not state_exact:
        status = "reject_correctness"
        conclusion = "The liveness owners change exact logits or resident state."
    elif not lifecycle_exact:
        status = "reject_lifecycle"
        conclusion = "A benchmark leg does not return tracked bytes exactly to baseline."
    elif not shape_ok:
        status = "reject_shape"
        conclusion = "The candidate changes scratch rows or did not select the intended arena."
    elif not prefill_ok or not decode_ok:
        status = "reject_performance"
        conclusion = "The liveness owners lose more than 1% prefill or decode throughput."
    elif not tracked_memory_ok:
        status = "reject_tracked_memory"
        conclusion = "The liveness owners do not save at least 1 GiB tracked peak at every 4K+ context."
    elif not whole_gtt_ok:
        status = "reject_whole_gtt_direction"
        conclusion = "The tracked saving does not reproduce in the same direction for whole GTT at every 4K+ context."
    else:
        status = "promote_liveness_alias"
        conclusion = (
            "The exact fixed-shape liveness owners pass state, lifecycle, wall, "
            "tracked-memory, and same-direction whole-GTT gates."
        )
    return {
        "status": status,
        "selected_default": "liveness" if passed else "dedicated",
        "promotion_passed": passed,
        "conclusion": conclusion,
        **checks,
        "thresholds": {
            "long_context_min": int(long_context_min),
            "min_tracked_savings_gib": float(min_tracked_savings_gib),
            "max_prefill_loss_pct": float(max_prefill_loss_pct),
            "max_decode_loss_pct": float(max_decode_loss_pct),
            "tracked_close_delta_bytes": 0,
            "whole_gtt_savings_direction": "positive at every 4K+ context",
        },
    }


def run_screen(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    workloads = tuple(int(value) for value in args.workloads)
    if not workloads or len(set(workloads)) != len(workloads):
        raise ScreenError("workloads must be unique and non-empty")
    for name in ("decode_tokens", "measured_runs"):
        if int(getattr(args, name)) <= 0:
            raise ScreenError(f"--{name.replace('_', '-')} must be positive")
    for name in ("warmup_decode_tokens", "warmup_runs", "state_decode_steps"):
        if int(getattr(args, name)) < 0:
            raise ScreenError(f"--{name.replace('_', '-')} must be non-negative")
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise ScreenError(f"model does not exist: {model}")
    if args.raw_root.exists():
        shutil.rmtree(args.raw_root)
    args.raw_root.mkdir(parents=True)
    if not args.compiler_version_file.is_file():
        compiler = subprocess.run(["hipcc", "--version"], capture_output=True, text=True, check=True)
        args.compiler_version_file.write_text(compiler.stdout, encoding="utf-8")

    base_env = os.environ.copy()
    base_env["HIPENGINE_BACKEND"] = "hip_gfx1151"
    base_env["HIPENGINE_HIP_ARCH"] = "gfx1151"
    base_env["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    base_env["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    base_env["GPU_MAX_HW_QUEUES"] = "1"

    legs: dict[int, dict[str, dict[str, Any]]] = {}
    for index, context in enumerate(workloads):
        context_root = args.raw_root / str(context)
        context_root.mkdir()
        order = MODES if index % 2 == 0 else tuple(reversed(MODES))
        legs[context] = {}
        for mode in order:
            output = context_root / f"{mode}.json"
            child_command = build_benchmark_command(
                python=sys.executable,
                model=model,
                prompt_length=context,
                query_rows=BASELINE_QUERY_ROWS,
                decode_tokens=int(args.decode_tokens),
                warmup_decode_tokens=int(args.warmup_decode_tokens),
                warmup_runs=int(args.warmup_runs),
                measured_runs=int(args.measured_runs),
                compiler_version_file=args.compiler_version_file,
                output=output,
            )
            env = _mode_environment(base_env, mode)
            gtt = _run_with_gtt_sampler(
                child_command,
                env=env,
                stdout_path=context_root / f"{mode}.stdout.log",
                stderr_path=context_root / f"{mode}.stderr.log",
                card_name=args.card_name,
                interval_ms=float(args.gtt_interval_ms),
            )
            payload = _load_json(output, label=f"{context}/{mode} benchmark")
            census = validate_benchmark_mode(
                payload,
                mode=mode,
                prompt_length=context,
                decode_tokens=int(args.decode_tokens),
                warmup_decode_tokens=int(args.warmup_decode_tokens),
                warmup_runs=int(args.warmup_runs),
                measured_runs=int(args.measured_runs),
                expected_token_id=int(args.expected_token_id),
            )
            legs[context][mode] = {
                "payload": payload,
                "gtt": gtt,
                "census": census,
                "command": child_command,
                "environment": {ENV_NAME: env[ENV_NAME]},
                "json": str(output),
                "json_sha256": _sha256(output),
            }

    state_payloads: dict[str, dict[str, Any]] = {}
    state_root = args.raw_root / "state"
    state_root.mkdir()
    state_capacity = max(workloads) + int(args.decode_tokens) + int(args.warmup_decode_tokens) + 1
    for mode in MODES:
        output = state_root / f"{mode}.json"
        state_command = _state_command(
            python=sys.executable,
            model=model,
            contexts=workloads,
            query_rows=BASELINE_QUERY_ROWS,
            decode_steps=int(args.state_decode_steps),
            max_sequence_length=state_capacity,
            compiler_version_file=args.compiler_version_file,
            output=output,
        )
        env = _mode_environment(base_env, mode)
        _run_logged(
            state_command,
            env=env,
            stdout_path=state_root / f"{mode}.stdout.log",
            stderr_path=state_root / f"{mode}.stderr.log",
        )
        payload = _load_json(output, label=f"{mode} state child")
        _validate_state_child(
            payload,
            query_rows=BASELINE_QUERY_ROWS,
            expected_contexts=workloads,
        )
        state_payloads[mode] = payload

    state_comparison = compare_state_children(
        state_payloads["dedicated"],
        state_payloads["liveness"],
        expected_contexts=workloads,
    )
    rows = [
        summarize_context(
            prompt_length=context,
            dedicated=legs[context]["dedicated"]["payload"],
            liveness=legs[context]["liveness"]["payload"],
            dedicated_gtt=legs[context]["dedicated"]["gtt"],
            liveness_gtt=legs[context]["liveness"]["gtt"],
        )
        for context in workloads
    ]
    decision = classify_screen(
        rows,
        state_comparison=state_comparison,
        long_context_min=int(args.long_context_min),
        min_tracked_savings_gib=float(args.min_tracked_savings_gib),
        max_prefill_loss_pct=float(args.max_prefill_loss_pct),
        max_decode_loss_pct=float(args.max_decode_loss_pct),
    )
    first_payload = legs[workloads[0]]["liveness"]["payload"]
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend="hip_gfx1151",
        resolved_backend=str(first_payload["resolved_backend"]),
        target_arch=str(first_payload["target_arch"]),
        model_path=model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_BACKEND": base_env.get("HIPENGINE_BACKEND"),
            "HIPENGINE_HIP_ARCH": base_env.get("HIPENGINE_HIP_ARCH"),
            "HIPENGINE_GGUF_DECODE_REPACK": base_env.get("HIPENGINE_GGUF_DECODE_REPACK"),
            "GPU_MAX_HW_QUEUES": base_env.get("GPU_MAX_HW_QUEUES"),
            ENV_NAME: "0 control / 1 candidate",
        },
        build_profile="gguf_sh_m2_scratch_liveness_screen",
        timing_protocol="independent right-sized process per context/mode; one warmup plus three measurements",
        warmups=int(args.warmup_runs),
        repetitions=int(args.measured_runs),
        profiler={"enabled": False, "reason": "host wall plus whole-GTT memory screen"},
    )
    raw_legs = {
        str(context): {
            mode: {
                "command": legs[context][mode]["command"],
                "environment": legs[context][mode]["environment"],
                "child_json": legs[context][mode]["json"],
                "child_json_sha256": legs[context][mode]["json_sha256"],
                "whole_gtt_10ms": legs[context][mode]["gtt"],
            }
            for mode in MODES
        }
        for context in workloads
    }
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": decision["status"],
        "performance_claim": bool(decision["promotion_passed"]),
        "correctness_claim": bool(state_comparison["passed"]),
        "workload": {
            "model": str(model),
            "quant": "gguf_q4_k_m",
            "kv_dtype": "bf16",
            "backend": "hip_gfx1151",
            "prompt_source": "repeated_token_id",
            "prompt_token_id": int(args.expected_token_id),
            "prompt_lengths": list(workloads),
            "decode_tokens": int(args.decode_tokens),
            "warmup_decode_tokens": int(args.warmup_decode_tokens),
            "fixed_chunk_sizes": chunk_sizes(BASELINE_QUERY_ROWS),
            "modes": {
                "dedicated": {ENV_NAME: "0"},
                "liveness": {ENV_NAME: "1"},
            },
        },
        "protocol": {
            "benchmark": "independent persistent session per context/mode; one discarded run plus three measured runs",
            "whole_gtt": f"whole-device amdgpu GTT sampled every {float(args.gtt_interval_ms):g} ms",
            "correctness": state_comparison["protocol"],
            "promotion_rule": (
                "fixed 4,096 query rows, byte-exact state/logits/trajectory, exact tracked close, "
                "<=1% prefill/decode loss, >=1.0 GiB tracked-peak saving and positive whole-GTT "
                "saving at every 4K+ context"
            ),
        },
        "rows": rows,
        "state_comparison": state_comparison,
        "decision": decision,
        "provenance": provenance,
        "raw": {
            "root": str(args.raw_root),
            "legs": raw_legs,
            "state_children": {
                mode: {
                    "environment": {ENV_NAME: "0" if mode == "dedicated" else "1"},
                    "json": str(state_root / f"{mode}.json"),
                    "json_sha256": _sha256(state_root / f"{mode}.json"),
                }
                for mode in MODES
            },
        },
        "notes": [
            "Both legs keep the frozen q4096 execution shape and explicit five-surface chunk policy.",
            "Right-sized scratch below 4,096 physical rows stays dedicated; owner-slot coloring targets only the 4,096-row class required by SH-M2.",
            "The architecture capability and exact compact-GDN route own admission; the environment only provides the same-revision disable control.",
            "State-child timings are excluded from performance evidence.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--workloads", nargs="+", type=_parse_length, default=list(DEFAULT_WORKLOADS))
    parser.add_argument("--expected-token-id", type=int, default=9707)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-decode-tokens", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--state-decode-steps", type=int, default=4)
    parser.add_argument("--gtt-interval-ms", type=float, default=10.0)
    parser.add_argument("--card-name")
    parser.add_argument("--compiler-version-file", type=Path, default=Path("/tmp/hipengine-hipcc-version.txt"))
    parser.add_argument("--raw-root", type=Path, default=Path("/tmp/hipengine-sh-m2"))
    parser.add_argument("--long-context-min", type=int, default=4096)
    parser.add_argument("--min-tracked-savings-gib", type=float, default=1.0)
    parser.add_argument("--max-prefill-loss-pct", type=float, default=1.0)
    parser.add_argument("--max-decode-loss-pct", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    payload = run_screen(args, command=command)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if payload["decision"]["promotion_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
