#!/usr/bin/env python3
"""Complete-model gate for the withheld dense Q4T16 rowtile variants on gfx1151.

Two keys are withheld from the gfx1151 alias table:

* ``("linear", "gguf_q4_k_t16_v1", "dense_rowtile_col4_bf16_bf16_out")``
* ``("linear+residual", "gguf_q4_k_t16_v1",
  "dense_rowtile_bf16_residual_bf16_out")``

The leaf screen (``scripts/qwen38_q4_dense_rowtile_gfx1151_screen.py``) shows
both are faster and bit-exact on actual Qwen3.8-27B Q4_K weights at the shapes
and row bands where dispatch can select them. This harness runs the required
complete-model gate: c>N generated-token equality against independent c1
generations, in both the shipped (excluded) and the candidate (admitted)
configuration, with per-variant call counts so the displaced work is measured
rather than assumed.

The diagnostic derives the batch width from the number of prompts it is given,
so the default mode exercises the suite's first ``--rows`` prompts. Pass
``--windows`` to cover the *whole* suite with sliding full-width windows: each
window keeps the row band under test instead of shrinking the batch to fit the
tail, and the resulting payload carries a per-prompt coverage table.

The registry is mutated in-process. The ``excluded`` arm is the pre-admission
control: the two keys were admitted on 2026-09-12, so the shipped gfx1151
alias table now registers them, and ``--arms admitted`` reproduces the shipped
configuration. Nothing here changes the shipped default.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import (
    KernelKey,
    is_registered,
    register,
    resolve,
    unregister,
)
from scripts.qwen35_batch_gguf_diagnostic import build_parser, run

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_PROMPTS = ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
DEFAULT_OUTPUT = Path("/tmp/hip1151-t6/q4-dense-rowtile-gate.json")

BACKEND = "hip_gfx1151"
SOURCE_BACKEND = "hip_gfx1100"

# The two withheld keys this gate decides.
CANDIDATES = (
    ("linear", "gguf_q4_k_t16_v1", "dense_rowtile_col4_bf16_bf16_out"),
    (
        "linear+residual",
        "gguf_q4_k_t16_v1",
        "dense_rowtile_bf16_residual_bf16_out",
    ),
)

# Every dense Q4T16 variant whose call count explains the candidate's effect.
TRACKED = (
    ("linear", "gguf_q4_k_t16_v1", "dense_rowtile_col4_bf16_bf16_out"),
    ("linear", "gguf_q4_k_t16_v1", "dense_rowtile_bf16_bf16_out"),
    ("linear", "gguf_q4_k_t16_v1", "dense_rowtile16_w2_bf16_bf16_out"),
    ("linear", "gguf_q4_k_t16_v1", "dense_single_local32_bf16_bf16_out"),
    (
        "linear+residual",
        "gguf_q4_k_t16_v1",
        "dense_rowtile_bf16_residual_bf16_out",
    ),
    (
        "linear+residual",
        "gguf_q4_k_t16_v1",
        "dense_single_local32_bf16_residual_bf16_out",
    ),
)

COUNTS: dict[tuple[str, str, str, int], int] = {}
ORIGINALS: dict[tuple[str, str, str], object] = {}

# Positional index of the ``rows`` argument per layer signature.
ROWS_ARG_INDEX = {"linear": 3, "linear+residual": 4}


def _key(layer: str, quant: str, variant: str) -> KernelKey:
    return KernelKey(BACKEND, layer, quant, variant)


def _install_counters() -> None:
    """Wrap every tracked variant that is currently registered.

    Idempotent, so it can run again after the candidate keys are admitted.
    """

    for layer, quant, variant in TRACKED:
        key_tuple = (layer, quant, variant)
        if key_tuple in ORIGINALS:
            continue
        key = _key(layer, quant, variant)
        if not is_registered(key):
            continue
        fn = resolve(backend=BACKEND, layer=layer, quant=quant, variant=variant)
        ORIGINALS[key_tuple] = fn
        COUNTS[key_tuple] = 0

        def make_wrapper(inner, counter_key, rows_index):
            def wrapper(*args, **kwargs):
                rows = int(args[rows_index]) if len(args) > rows_index else 0
                key = (*counter_key, rows)
                COUNTS[key] = COUNTS.get(key, 0) + 1
                return inner(*args, **kwargs)

            return wrapper

        register(
            key,
            make_wrapper(fn, key_tuple, ROWS_ARG_INDEX[layer]),
            replace=True,
        )


def _restore_counters() -> None:
    for key_tuple, fn in ORIGINALS.items():
        layer, quant, variant = key_tuple
        register(_key(layer, quant, variant), fn, replace=True)
    ORIGINALS.clear()


def _set_admitted(admitted: bool) -> dict[str, str]:
    """Register or drop the two withheld keys; return the resulting state."""

    state: dict[str, str] = {}
    for layer, quant, variant in CANDIDATES:
        key = _key(layer, quant, variant)
        if admitted:
            fn = resolve(
                backend=SOURCE_BACKEND, layer=layer, quant=quant, variant=variant
            )
            register(key, fn, replace=True)
            state[f"{layer}/{variant}"] = "admitted"
        else:
            unregister(key)
            state[f"{layer}/{variant}"] = "excluded"
        # Any other excluded key must stay excluded: this gate admits exactly
        # the two keys under review and nothing else.
        assert is_registered(key) is admitted, key
    return state


def _calls_by_rows() -> dict[str, dict[str, int]]:
    """Aggregate the counters into per-variant row-width histograms."""

    summary: dict[str, dict[str, int]] = {}
    for (layer, _quant, variant, rows), count in sorted(COUNTS.items()):
        label = f"{layer}/{variant}"
        summary.setdefault(label, {})[str(rows)] = count
    return summary


def _configure(admitted: bool) -> dict[str, str]:
    """Set registry state, then wrap whatever is registered for counting."""

    _restore_counters()
    state = _set_admitted(admitted)
    _install_counters()
    COUNTS.clear()
    return state


def _load_suite_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _window_bounds(count: int, width: int) -> list[tuple[int, int]]:
    """Cover ``count`` prompts with full-width windows of ``width`` prompts.

    The diagnostic derives the batch width from the number of prompts in the
    suite, so a window must never be shorter than ``width``: a short tail
    window would silently run at a narrower row band than the one under test.
    The final window slides back and overlaps instead.
    """

    if width < 2:
        raise ValueError("window width must be at least 2")
    if count < width:
        raise ValueError(f"prompt suite has {count} row(s), need at least {width}")
    if count == width:
        return [(0, count)]
    bounds: list[tuple[int, int]] = []
    start = 0
    while start + width < count:
        bounds.append((start, start + width))
        start += width
    bounds.append((count - width, count))
    deduped: list[tuple[int, int]] = []
    for bound in bounds:
        if bound not in deduped:
            deduped.append(bound)
    return deduped


def _mem_available_bytes() -> int | None:
    """Host MemAvailable, so a multi-arm process can refuse to OOM the box."""

    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _arm_args(
    args: argparse.Namespace, *, rows: int, suite: Path
) -> argparse.Namespace:
    parser = build_parser()
    argv = [
        "--fixture",
        str(args.fixture),
        "--model",
        str(args.model),
        "--rows",
        str(rows),
        "--backend",
        BACKEND,
        "--quant",
        args.quant,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--repeat-runs",
        "1",
        "--prompt-suite",
        str(suite),
        "--execute",
    ]
    return parser.parse_args(argv)


def _run_summary(payload: dict) -> dict:
    """Extract the gate-relevant fields from a diagnostic payload."""

    runs = []
    for entry in payload.get("runs", []):
        runs.append(
            {
                "generated_token_ids": entry.get("generated_token_ids"),
                "row_equal": entry.get("row_equal"),
                "all_rows_equal": entry.get("all_rows_equal"),
                "native_caware_decode": entry.get("native_caware_decode"),
                "serial_decode_fallback": entry.get("serial_decode_fallback"),
                "execution_path": entry.get("execution_path"),
            }
        )
    return {
        "status": payload.get("status"),
        "passed": payload.get("passed"),
        "prompt_suite": payload.get("prompt_suite"),
        "prompt_rows": payload.get("prompt_rows"),
        "prompt_token_count": payload.get("prompt_token_count"),
        "prepared_context_tokens": payload.get("prepared_context_tokens"),
        "independent_c1_token_ids": payload.get("independent_c1_token_ids"),
        "blockers": payload.get("blockers"),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "tests/fixtures/gguf/qwen35_0_8b_q4_k_m_e2e.json",
    )
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--rows", type=str, default="2,4")
    parser.add_argument(
        "--arms",
        type=str,
        default="excluded,admitted",
        help="Comma list of arms to run: excluded, admitted",
    )
    parser.add_argument(
        "--min-mem-available-gib",
        type=float,
        default=40.0,
        help=(
            "Refuse to start an arm when host MemAvailable is below this; the "
            "2026-09-12 leak-free rerun still holds one model per arm, and a "
            "host-wide OOM takes unrelated session services down with it"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--windows",
        action="store_true",
        help=(
            "Cover the whole prompt suite with sliding windows of --rows width "
            "instead of the suite's first --rows prompts"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    row_widths = tuple(int(part) for part in args.rows.split(","))
    arms = tuple(part.strip() for part in args.arms.split(",") if part.strip())
    for arm_name in arms:
        if arm_name not in {"excluded", "admitted"}:
            raise ValueError(f"unknown arm {arm_name!r}")
    suite_rows = _load_suite_rows(args.prompts)
    suite_ids = [str(row.get("id")) for row in suite_rows]

    scratch: Path | None = None
    cases: list[tuple[int, int, int, Path]] = []
    if args.windows:
        scratch = Path(tempfile.mkdtemp(prefix="q4-rowtile-gate-suite-"))
        for rows in row_widths:
            for start, end in _window_bounds(len(suite_rows), rows):
                window_path = scratch / f"suite-rows{rows}-{start}-{end}.jsonl"
                window_path.write_text(
                    "".join(
                        json.dumps(row, sort_keys=True) + "\n"
                        for row in suite_rows[start:end]
                    )
                )
                cases.append((rows, start, end, window_path))
    else:
        for rows in row_widths:
            cases.append((rows, 0, min(rows, len(suite_rows)), args.prompts))

    register_gfx1151_kernels()
    results: list[dict] = []
    try:
        for rows, window_start, window_end, suite_path in cases:
            for arm_name in arms:
                admitted = arm_name == "admitted"
                label = arm_name
                available = _mem_available_bytes()
                if (
                    available is not None
                    and available < args.min_mem_available_gib * 1024**3
                ):
                    raise RuntimeError(
                        f"host MemAvailable {available / 1024**3:.1f} GiB is below "
                        f"--min-mem-available-gib {args.min_mem_available_gib:.1f} "
                        f"before rows={rows} arm={label}"
                    )
                state = _configure(admitted)
                arm = _arm_args(args, rows=rows, suite=suite_path)
                print(
                    f"=== rows={rows} window={window_start}:{window_end} "
                    f"arm={label} {state} ===",
                    flush=True,
                )
                payload = run(arm)
                summary = _run_summary(payload)
                calls = _calls_by_rows()
                after = _mem_available_bytes()
                prompt_rows = summary["prompt_rows"] or []
                first_run = summary["runs"][0] if summary["runs"] else {}
                results.append(
                    {
                        "rows": rows,
                        "window_start": window_start,
                        "window_end": window_end,
                        "suite": str(suite_path),
                        "arm": label,
                        "registry_state": state,
                        "status": summary["status"],
                        "passed": summary["passed"],
                        "prompt_suite": summary["prompt_suite"],
                        "prompt_rows": prompt_rows,
                        "prompt_ids": [row.get("id") for row in prompt_rows],
                        "prompt_categories": [
                            row.get("category") for row in prompt_rows
                        ],
                        "prompt_token_count": summary["prompt_token_count"],
                        "native_token_ids": first_run.get("generated_token_ids") or [],
                        "row_equal": first_run.get("row_equal") or [],
                        "native_caware_decode": first_run.get("native_caware_decode"),
                        "serial_decode_fallback": first_run.get(
                            "serial_decode_fallback"
                        ),
                        "independent_c1_token_ids": summary[
                            "independent_c1_token_ids"
                        ],
                        "prepared_context_tokens": summary[
                            "prepared_context_tokens"
                        ],
                        "calls": calls,
                        "blockers": summary["blockers"],
                        "mem_available_gib_before": (
                            None if available is None else round(available / 1024**3, 2)
                        ),
                        "mem_available_gib_after": (
                            None if after is None else round(after / 1024**3, 2)
                        ),
                        "raw": payload,
                    }
                )
                print(
                    f"    status={summary['status']} "
                    f"prompts={[row.get('id') for row in prompt_rows]} "
                    f"native={first_run.get('native_caware_decode')} "
                    f"fallback={first_run.get('serial_decode_fallback')}",
                    flush=True,
                )
                print(f"    calls={json.dumps(calls)}", flush=True)
    finally:
        _restore_counters()
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)

    verdicts: list[dict] = []
    for rows, window_start, window_end, _suite in cases:
        pair = {
            entry["arm"]: entry
            for entry in results
            if entry["rows"] == rows
            and entry["window_start"] == window_start
            and entry["window_end"] == window_end
        }
        if set(pair) != set(arms):
            # Partial arm runs are diagnostics for the caller, not verdicts.
            continue
        excluded = pair["excluded"]
        admitted = pair["admitted"]
        equal_arms = admitted["native_token_ids"] == excluded["native_token_ids"]
        candidate_exact = admitted["passed"] is True
        control_exact = excluded["passed"] is True
        verdicts.append(
            {
                "rows": rows,
                "window_start": window_start,
                "window_end": window_end,
                "prompt_ids": admitted["prompt_ids"],
                "control_eq_ok": control_exact,
                "candidate_eq_ok": candidate_exact,
                "arms_identical": equal_arms,
                "candidate_exact": bool(candidate_exact and equal_arms),
                "candidate_all_rows_equal": all(
                    bool(value) for value in (admitted["row_equal"] or [])
                ),
                "candidate_native_caware_decode": admitted[
                    "native_caware_decode"
                ],
                "candidate_serial_decode_fallback": admitted[
                    "serial_decode_fallback"
                ],
                "distinct_c1_rows": len(
                    {tuple(row) for row in (admitted["independent_c1_token_ids"] or [])}
                ),
                "displaced_calls": {
                    "col4": sum(
                        admitted["calls"]
                        .get("linear/dense_rowtile_col4_bf16_bf16_out", {})
                        .values()
                    ),
                    "fused_residual": sum(
                        admitted["calls"]
                        .get(
                            "linear+residual/dense_rowtile_bf16_residual_bf16_out",
                            {},
                        )
                        .values()
                    ),
                },
                "displaced_calls_by_rows": {
                    "col4": admitted["calls"].get(
                        "linear/dense_rowtile_col4_bf16_bf16_out", {}
                    ),
                    "fused_residual": admitted["calls"].get(
                        "linear+residual/dense_rowtile_bf16_residual_bf16_out", {}
                    ),
                },
                "residual_primitive_add_calls_control": sum(
                    excluded["calls"]
                    .get(
                        "linear+residual/dense_single_local32_bf16_residual_bf16_out",
                        {},
                    )
                    .values()
                ),
            }
        )

    prompt_coverage: dict[str, dict] = {}
    for result in results:
        if result["arm"] != "admitted":
            continue
        for index, prompt_id in enumerate(result["prompt_ids"]):
            if prompt_id is None:
                continue
            entry = prompt_coverage.setdefault(
                str(prompt_id),
                {
                    "category": (
                        result["prompt_categories"][index]
                        if index < len(result["prompt_categories"])
                        else None
                    ),
                    "cases": [],
                    "equal": True,
                },
            )
            equal = bool((result["row_equal"] or [False])[index])
            entry["cases"].append(
                {
                    "rows": result["rows"],
                    "window": [result["window_start"], result["window_end"]],
                    "equal": equal,
                }
            )
            entry["equal"] = bool(entry["equal"] and equal)

    covered = sorted(prompt_coverage)
    payload = {
        "kind": "qwen38_gfx1151_q4_dense_rowtile_complete_model_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "git_head": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ("git", "status", "--short", "--untracked-files=no"),
            cwd=ROOT,
            text=True,
        ).splitlines(),
        "model": str(args.model),
        "quant": args.quant,
        "backend": BACKEND,
        "max_new_tokens": args.max_new_tokens,
        "prompts": str(args.prompts),
        "windowed": bool(args.windows),
        "arms": list(arms),
        "suite_prompt_ids": suite_ids,
        "candidates": [list(item) for item in CANDIDATES],
        "verdicts": verdicts,
        "prompt_coverage": prompt_coverage,
        "runs": results,
        "all_rows_exact": all(v["candidate_exact"] for v in verdicts),
        "all_prompts_covered": (
            covered == sorted(suite_ids) if args.windows else None
        ),
        "all_prompts_equal": all(
            bool(entry["equal"]) for entry in prompt_coverage.values()
        ),
    }
    payload["gate_passed"] = bool(
        payload["all_rows_exact"]
        and payload["all_prompts_equal"]
        and (payload["all_prompts_covered"] is not False)
        and set(arms) == {"excluded", "admitted"}
        and len(verdicts) == len(cases)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.output}")
    for verdict in verdicts:
        print(
            f"rows={verdict['rows']} window={verdict['window_start']}:"
            f"{verdict['window_end']}: "
            f"candidate_exact={verdict['candidate_exact']} "
            f"displaced={verdict['displaced_calls']}"
        )
    print(f"gate_passed={payload['gate_passed']}")
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
