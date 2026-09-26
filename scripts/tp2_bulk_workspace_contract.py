#!/usr/bin/env python3
"""Bulk prefill's input/resource contract across prompt lengths and reuse.

The bulk prefill candidate is being considered as the default TP2 prefill
schedule. Two things stand between it and that: its workspace contract, and the
claim - recorded on 2026-09-18 and never reproduced since - that a workspace
allocated lazily at first prefill is *erratic* (25-137 tok/s) where one built at
construction is not (386 tok/s).

This harness answers both on the revision it runs against:

* **Length ladder.** One session, prompts from short to long, each one warmed
  then measured, recording per call: workspace rows actually allocated, wall
  time, and the head logits' finiteness. The first call at a new length is kept
  separate from its repeats, because a workspace that has to grow pays a rebuild
  there and steady state does not.
* **Reuse in both directions.** The ladder revisits its shortest length at the
  end, so a prompt shorter than the current workspace is measured for reuse
  rather than for a rebuild.
* **Allocation timing variants.** The same ladder under three workspace
  policies: prompt-sized (``bulk_prefill_rows=None``, what the shipping route
  uses), pinned at 512 rows (allocated eagerly at construction), and pinned at
  the session capacity. If the deferred-allocation slowdown is real it shows up
  as a steady-state gap between the first and the others.
* **Prefill followed by decode.** Each length also runs ``generate`` for a
  couple of tokens, so the prompt is consumed by the bulk route and the decode
  loop continues from it in the same call.

What this does *not* claim: it does not score KL against a teacher (the
production comparison in ``scripts/tp2_teacher_coverage_broad.py`` does that),
and it does not certify warm-state continuation.

Usage::

    uv run python scripts/tp2_bulk_workspace_contract.py \\
        --lengths 52,128,511,512,513,1024 \\
        --json benchmarks/results/2026-09-25-w7900-tp2-bulk-workspace-contract.json
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_TOKEN_ID = 9707
VARIANTS = ("prompt-sized", "pinned-512", "pinned-capacity")

# The over-capacity refusal is a plain ``ValueError`` raised before any
# allocation or launch. Anything else is a harness failure, not a contract
# answer, so it must propagate instead of being recorded as a valid refusal.
EXPECTED_REFUSAL_TYPE = "ValueError"
CAPACITY_REFUSAL_FRAGMENT = "exceeds the bulk prefill capacity"

# This harness has no numerical oracle: it proves shape/finiteness and
# completion counts, not KL, top-1, or argmax identity. Any argmax recorded
# below is a diagnostic label, never a correctness verdict.
ORACLE_SCOPE = (
    "finiteness and completion counts only; no numerical/KL/top-1/argmax oracle"
)


def parse_lengths(text: str) -> list[int]:
    lengths: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value < 1:
            raise argparse.ArgumentTypeError(f"prompt length must be positive: {value}")
        lengths.append(value)
    if not lengths:
        raise argparse.ArgumentTypeError("--lengths must name at least one length")
    return lengths


def _vram_used_gib(runtime: Any, device: int, scoped: Any) -> float:
    with scoped(runtime, device):
        free_bytes, total_bytes = runtime.mem_get_info()
    return round((total_bytes - free_bytes) / 2**30, 6)


def _finite(logits: Any) -> bool:
    import numpy as np

    return bool(np.isfinite(np.asarray(logits, dtype=np.float32)).all())


def _logits_facts(logits: Any) -> dict[str, Any]:
    """Describe a returned logits block precisely enough to judge finiteness.

    ``bulk_prefill(..., logits_rows=1)`` returns only the projected rows, so a
    non-finite element anywhere is a non-finite element in a row the product
    path reads. Recording the shape and the non-finite count separates "the
    head wrote nothing" from "the buffer is partly unwritten".
    """

    import numpy as np

    array = np.asarray(logits, dtype=np.float32)
    finite = np.isfinite(array)
    last_row = np.isfinite(array[-1]) if array.ndim >= 1 and array.shape[0] else finite
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "elements": int(array.size),
        "nonfinite_elements": int(array.size - int(finite.sum())),
        "all_finite": bool(finite.all()),
        "last_row_finite": bool(last_row.all()),
        "last_row_nonfinite": int(last_row.size - int(last_row.sum())),
        # Diagnostic only: without a teacher/oracle this number cannot be
        # called correct, so it never feeds a pass/fail check.
        "last_row_argmax_diagnostic": (
            int(np.argmax(np.nan_to_num(array[-1], nan=-np.inf))) if array.size else None
        ),
    }


def _is_capacity_refusal(error: BaseException, rows: int, expected_capacity: int) -> bool:
    """True only for the exact over-capacity ``ValueError`` this route raises."""

    message = str(error)
    return (
        type(error).__name__ == EXPECTED_REFUSAL_TYPE
        and CAPACITY_REFUSAL_FRAGMENT in message
        and f"prompt of {int(rows)} tokens" in message
        and f"capacity {int(expected_capacity)}" in message
    )


def _probe_capacity_refusal(
    session: Any, prompt: tuple[int, ...], rows: int, expected_capacity: int
) -> dict[str, Any]:
    """Ask a pinned session to overflow and classify the answer.

    Only ``ValueError`` is treated as a refusal; any other exception is a
    harness failure and propagates. A ``ValueError`` whose type or message does
    not match the expected capacity refusal is recorded as a non-refusal so the
    verdict fails closed rather than accepting an unrelated error.
    """

    try:
        session.bulk_prefill(prompt, logits_rows=1)
    except ValueError as error:
        return {
            "raised": True,
            "type": type(error).__name__,
            "message": str(error)[:300],
            "expected_type": EXPECTED_REFUSAL_TYPE,
            "expected_capacity": int(expected_capacity),
            "capacity_refusal": _is_capacity_refusal(error, rows, expected_capacity),
        }
    return {
        "raised": False,
        "type": None,
        "message": "",
        "expected_type": EXPECTED_REFUSAL_TYPE,
        "expected_capacity": int(expected_capacity),
        "capacity_refusal": False,
    }


def _workspace_allocation_signature(session: Any, devices: tuple[int, ...]) -> dict[str, Any]:
    """Device-buffer pointers identifying the live bulk workspace allocation.

    Row counts alone do not prove the allocation is unchanged: a release and
    rebuild at the same size would keep the count and swap the pointers. This
    signature lets the revisit check compare the actual buffers.
    """

    hidden = getattr(session, "_bulk_hidden", {}) or {}
    logits_buf = getattr(session, "_bulk_logits_buf", {}) or {}
    signature: dict[str, Any] = {}
    for device in devices:
        entry: dict[str, Any] = {}
        if device in hidden:
            entry["hidden"] = [int(ptr) for ptr in hidden[device]]
        buffer = logits_buf.get(device)
        if buffer is not None:
            entry["logits"] = int(getattr(buffer, "ptr", -1))
        signature[str(device)] = entry
    return signature


def run_variant(
    *,
    model: Path,
    devices: tuple[int, ...],
    lengths: list[int],
    decode_tokens: int,
    repeats: int,
    token_id: int,
    variant: str,
    capacity: int,
    scoped: Any,
    runtime: Any,
) -> dict[str, Any]:
    """Walk the ladder in one session under one workspace policy."""

    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

    pinned = {
        "prompt-sized": None,
        "pinned-512": 512,
        "pinned-capacity": int(capacity),
    }[variant]

    record: dict[str, Any] = {
        "variant": variant,
        "bulk_prefill_rows_requested": pinned,
        "lengths": [],
    }
    session = None
    build_started = time.perf_counter()
    try:
        session = MlpTP2GenerationSession(
            model,
            devices=devices,
            mode="tp2",
            max_sequence_length=int(capacity),
            bulk_prefill=True,
            bulk_prefill_rows=pinned,
        )
        record["session_build_s"] = round(time.perf_counter() - build_started, 3)
        record["workspace_rows_after_build"] = int(getattr(session, "_bulk_rows", -1))
        record["vram_after_build_gib"] = {
            str(device): _vram_used_gib(runtime, device, scoped) for device in devices
        }

        # Warm once on the shortest prompt so the ladder's first measurement is
        # a workspace question and not a decode-schedule capture.
        warm_prompt = (int(token_id),) * min(lengths)
        session.bulk_prefill(warm_prompt, logits_rows=1)
        record["workspace_rows_after_warmup"] = int(getattr(session, "_bulk_rows", -1))

        order = list(lengths) + [min(lengths)]
        for index, length in enumerate(order):
            prompt = (int(token_id),) * int(length)
            entry: dict[str, Any] = {
                "length": int(length),
                "revisit": bool(index >= len(lengths)),
            }
            # A pinned workspace refuses a longer prompt by design. Record that
            # refusal where it happens instead of aborting the ladder: it is one
            # of the contract's answers, not a harness failure.
            pinned_ceiling = pinned if pinned is not None else capacity
            if int(length) > int(pinned_ceiling):
                # A refusal must be a pure rejection: the workspace allocation
                # and the session's health must be identical before and after.
                entry["workspace_rows_before"] = int(getattr(session, "_bulk_rows", -1))
                entry["workspace_allocation_before"] = _workspace_allocation_signature(
                    session, devices
                )
                entry["session_poisoned_before"] = bool(
                    getattr(session, "_poisoned", False)
                )
                refusal = _probe_capacity_refusal(
                    session, prompt, int(length), int(pinned_ceiling)
                )
                entry["refused_over_pinned_capacity"] = refusal
                entry["workspace_rows_after_first"] = int(getattr(session, "_bulk_rows", -1))
                entry["workspace_allocation_after"] = _workspace_allocation_signature(
                    session, devices
                )
                entry["session_poisoned_after"] = bool(
                    getattr(session, "_poisoned", False)
                )
                record["lengths"].append(entry)
                print(
                    f"  {variant}: len={length} refused={refusal['raised']} "
                    f"capacity_refusal={refusal['capacity_refusal']} "
                    f"msg={refusal.get('message', '')[:80]}",
                    flush=True,
                )
                continue
            entry["workspace_rows_before"] = int(getattr(session, "_bulk_rows", -1))
            entry["workspace_allocation_before"] = _workspace_allocation_signature(
                session, devices
            )
            started = time.perf_counter()
            first = session.bulk_prefill(prompt, logits_rows=1)
            entry["first_call_s"] = round(time.perf_counter() - started, 6)
            entry["workspace_rows_after_first"] = int(getattr(session, "_bulk_rows", -1))
            entry["workspace_allocation_after"] = _workspace_allocation_signature(
                session, devices
            )
            facts = _logits_facts(first)
            entry["finite"] = facts["all_finite"]
            entry["logits"] = facts
            entry["last_row_argmax_diagnostic"] = facts["last_row_argmax_diagnostic"]
            steady: list[float] = []
            repeat_outputs: list[dict[str, Any]] = []
            # ``repeats`` counts the total measured calls: the first call plus
            # ``repeats - 1`` steady-state repeats. ``--repeats 1`` therefore
            # measures only the first call and records no repeats, matching
            # ``evaluate_variant_checks``.
            for _ in range(max(0, repeats - 1)):
                started = time.perf_counter()
                logits = session.bulk_prefill(prompt, logits_rows=1)
                steady.append(time.perf_counter() - started)
                # Every measured call is checked, not just the first: a later
                # repeat that writes non-finite or a different-shaped block is
                # exactly the failure the finiteness gate exists to catch.
                repeat_outputs.append(_logits_facts(logits))
            entry["repeat_outputs"] = repeat_outputs
            samples = [entry["first_call_s"], *steady]
            entry["steady_samples_s"] = [round(value, 6) for value in steady]
            entry["steady_median_s"] = round(statistics.median(steady), 6) if steady else None
            entry["min_s"] = round(min(samples), 6)
            entry["median_s"] = round(statistics.median(samples), 6)
            entry["prefill_tok_per_s_median"] = round(int(length) / statistics.median(samples), 2)
            entry["prefill_tok_per_s_best"] = round(int(length) / min(samples), 2)
            entry["vram_gib"] = {
                str(device): _vram_used_gib(runtime, device, scoped) for device in devices
            }

            # Prefill followed by decode in one call, from the same session.
            started = time.perf_counter()
            generation = session.generate(prompt, max_new_tokens=int(decode_tokens))
            entry["generate_s"] = round(time.perf_counter() - started, 6)
            entry["generate_prefill_step_s"] = round(
                sum(t.total_s for t in generation.step_traces if t.kind == "prefill"), 6
            )
            entry["generate_decode_steps"] = int(
                sum(1 for t in generation.step_traces if t.kind != "prefill")
            )
            entry["generate_prefill_steps"] = int(
                sum(1 for t in generation.step_traces if t.kind == "prefill")
            )
            entry["generated_token_count"] = int(len(generation.token_ids))
            entry["generated_tokens"] = [int(t) for t in generation.token_ids[:4]]
            record["lengths"].append(entry)
            print(
                f"  {variant}: len={length} rows={entry['workspace_rows_after_first']} "
                f"first={entry['first_call_s']:.4f}s steady={entry['steady_median_s']} "
                f"{entry['prefill_tok_per_s_median']} tok/s finite={entry['finite']}",
                flush=True,
            )
    finally:
        if session is not None:
            session.close()
        record["session_closed_s"] = round(time.perf_counter() - build_started, 3)
    return record


def evaluate_variant_checks(
    record: dict[str, Any], *, decode_tokens: int, repeats: int, capacity: int
) -> dict[str, Any]:
    """Pure fail-closed verdicts for one variant's ladder record.

    Every check requires real evidence: an empty ladder fails, a later repeat
    that is non-finite or differently shaped fails, a refusal that is not the
    exact expected over-capacity ``ValueError`` fails, and the revisit must show
    both unchanged rows and unchanged device-buffer pointers. ``capacity`` is
    kept in the signature so callers cannot accidentally evaluate a record
    against a different session envelope.
    """

    lengths = list(record.get("lengths", []))
    entries = [e for e in lengths if "refused_over_pinned_capacity" not in e]
    refusals = [e for e in lengths if "refused_over_pinned_capacity" in e]
    pinned = record.get("bulk_prefill_rows_requested")
    expected_refusal_lengths = [
        int(e["length"])
        for e in lengths
        if pinned is not None and int(e["length"]) > int(pinned)
    ]
    refused_lengths = [int(e["length"]) for e in refusals]

    def _shape(entry: dict[str, Any]) -> Any:
        return (entry.get("logits") or {}).get("shape")

    def _shape_ok(entry: dict[str, Any]) -> bool:
        shape = _shape(entry)
        return (
            isinstance(shape, list)
            and len(shape) == 2
            and int(shape[0]) == 1
            and int(shape[1]) >= 1
            and int((entry.get("logits") or {}).get("elements", 0)) > 0
        )

    def _repeat_ok(entry: dict[str, Any]) -> bool:
        outputs = entry.get("repeat_outputs")
        expected = max(0, int(repeats) - 1)
        if not isinstance(outputs, list) or len(outputs) != expected:
            return False
        first_shape = _shape(entry)
        return all(
            bool(out.get("all_finite"))
            and out.get("shape") == first_shape
            and int(out.get("elements", 0)) > 0
            for out in outputs
        )

    return {
        "has_lengths": bool(entries),
        "every_length_finite": bool(entries)
        and all(bool(e.get("finite")) for e in entries),
        "every_length_shape_expected": bool(entries) and all(_shape_ok(e) for e in entries),
        "every_repeat_output_finite_and_shaped": bool(entries)
        and all(_repeat_ok(e) for e in entries),
        "every_length_generate_completed": bool(entries)
        and all(
            int(e.get("generate_decode_steps", -1)) == int(decode_tokens)
            and int(e.get("generated_token_count", -1)) == int(decode_tokens)
            for e in entries
        ),
        "workspace_never_below_prompt": bool(entries)
        and all(
            int(e["workspace_rows_after_first"]) >= int(e["length"]) for e in entries
        ),
        "revisit_rows_unchanged": bool(entries)
        and bool(entries[-1].get("revisit"))
        and int(entries[-1]["workspace_rows_before"])
        == int(entries[-1]["workspace_rows_after_first"]),
        "revisit_allocation_unchanged": bool(entries)
        and bool(entries[-1].get("revisit"))
        and bool(entries[-1].get("workspace_allocation_after"))
        and entries[-1].get("workspace_allocation_before")
        == entries[-1].get("workspace_allocation_after"),
        "over_pinned_lengths_refused": all(
            bool(r["refused_over_pinned_capacity"].get("raised"))
            and bool(r["refused_over_pinned_capacity"].get("capacity_refusal"))
            for r in refusals
        ),
        "refusals_preserve_workspace_rows": all(
            r.get("workspace_rows_before") is not None
            and int(r["workspace_rows_before"]) == int(r["workspace_rows_after_first"])
            for r in refusals
        ),
        "refusals_preserve_workspace_allocation": all(
            bool(r.get("workspace_allocation_before"))
            and r.get("workspace_allocation_before")
            == r.get("workspace_allocation_after")
            for r in refusals
        ),
        "refusals_leave_session_healthy": all(
            r.get("session_poisoned_before") is False
            and r.get("session_poisoned_after") is False
            for r in refusals
        ),
        "refused_lengths_match_expected": refused_lengths == expected_refusal_lengths,
        "refused_lengths": refused_lengths,
        "lengths_run": [int(e["length"]) for e in entries],
        "growth_rebuilds_recorded": sorted(
            {int(e["workspace_rows_after_first"]) for e in entries}
        ),
    }


def summarize_checks(
    result: dict[str, Any], *, decode_tokens: int, repeats: int, capacity: int
) -> dict[str, Any]:
    """Attach per-variant verdicts and the aggregate pass/fail to ``result``."""

    checks = {
        variant: evaluate_variant_checks(
            record,
            decode_tokens=int(decode_tokens),
            repeats=int(repeats),
            capacity=int(capacity),
        )
        for variant, record in result["variants"].items()
    }
    result["checks"] = checks
    result["all_checks_passed"] = bool(checks) and all(
        all(value for value in verdict.values() if isinstance(value, bool))
        for verdict in checks.values()
    )
    return result


def _git_source_state() -> dict[str, Any]:
    """Revision and dirty flag for the tree that produced an artifact."""

    def run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args], capture_output=True, text=True, check=False
            )
        except OSError:
            return ""
        return completed.stdout.strip()

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "source_revision": revision or None,
        "source_dirty": bool(status),
        "source_status_porcelain": status or None,
    }


def _exact_command(argv: list[str] | None) -> str:
    """Shell-safe invocation, honoring an explicit ``main(argv)``.

    ``main`` can be called with a supplied argument list (tests do), so the
    recorded command must come from that list rather than ``sys.argv``.
    """

    raw = list(argv) if argv is not None else sys.argv[1:]
    return shlex.join([sys.executable, str(Path(__file__).resolve()), *raw])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths("52,128,511,512,513,1024"),
        help="comma-separated prompt lengths, walked in order and revisited at the end",
    )
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--decode-tokens", type=int, default=2)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="total measured calls per length: first call plus repeats-1 steady-state repeats",
    )
    parser.add_argument("--token-id", type=int, default=DEFAULT_TOKEN_ID)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help=f"comma-separated subset of {','.join(VARIANTS)}",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not variants:
        raise SystemExit(
            f"--variants must name at least one of {list(VARIANTS)}"
        )
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; expected a subset of {list(VARIANTS)}")
    if int(args.decode_tokens) < 1:
        raise SystemExit(f"--decode-tokens must be positive: {args.decode_tokens}")
    if int(args.repeats) < 1:
        raise SystemExit(f"--repeats must be positive: {args.repeats}")
    devices = tuple(int(part) for part in args.devices.split(","))
    capacity = max(args.lengths) + int(args.decode_tokens) + 1

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    result: dict[str, Any] = {
        "schema": 1,
        "kind": "tp2_bulk_workspace_contract",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "model": str(args.model),
        "exact_command": _exact_command(argv),
        "oracle_scope": ORACLE_SCOPE,
        "devices": list(devices),
        "lengths": list(args.lengths),
        "revisit_length": min(args.lengths),
        "decode_tokens": int(args.decode_tokens),
        "repeats": int(args.repeats),
        "session_capacity": int(capacity),
        "route": {
            "mode": "tp2",
            "bulk_prefill": True,
            "attention_shard": "session default",
            "reduce_mode": "session default",
        },
        "variants": {},
    }
    result.update(_git_source_state())
    print(
        f"ladder={args.lengths} revisit={min(args.lengths)} capacity={capacity} "
        f"variants={variants}",
        flush=True,
    )
    for variant in variants:
        print(f"variant {variant}", flush=True)
        result["variants"][variant] = run_variant(
            model=args.model,
            devices=devices,
            lengths=list(args.lengths),
            decode_tokens=int(args.decode_tokens),
            repeats=int(args.repeats),
            token_id=int(args.token_id),
            variant=variant,
            capacity=capacity,
            scoped=scoped_current_device,
            runtime=runtime,
        )

    # Verdicts the caller does not have to recompute.
    summarize_checks(
        result,
        decode_tokens=int(args.decode_tokens),
        repeats=int(args.repeats),
        capacity=int(capacity),
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1) + "\n")
        print(f"artifact: {args.json}", flush=True)
    print(f"all_checks_passed={result['all_checks_passed']}", flush=True)
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
