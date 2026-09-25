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
import statistics
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_TOKEN_ID = 9707
VARIANTS = ("prompt-sized", "pinned-512", "pinned-capacity")


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
        "last_row_argmax": int(np.argmax(np.nan_to_num(array[-1], nan=-np.inf))) if array.size else None,
    }


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
                refusal: dict[str, Any] = {"raised": False}
                try:
                    session.bulk_prefill(prompt, logits_rows=1)
                except Exception as error:  # noqa: BLE001 - the message is the evidence
                    refusal = {
                        "raised": True,
                        "type": type(error).__name__,
                        "message": str(error)[:300],
                    }
                entry["refused_over_pinned_capacity"] = refusal
                entry["workspace_rows_after_first"] = int(getattr(session, "_bulk_rows", -1))
                record["lengths"].append(entry)
                print(
                    f"  {variant}: len={length} refused={refusal['raised']} "
                    f"msg={refusal.get('message', '')[:80]}",
                    flush=True,
                )
                continue
            entry["workspace_rows_before"] = int(getattr(session, "_bulk_rows", -1))
            started = time.perf_counter()
            first = session.bulk_prefill(prompt, logits_rows=1)
            entry["first_call_s"] = round(time.perf_counter() - started, 6)
            entry["workspace_rows_after_first"] = int(getattr(session, "_bulk_rows", -1))
            facts = _logits_facts(first)
            entry["finite"] = facts["all_finite"]
            entry["logits"] = facts
            entry["last_row_argmax"] = facts["last_row_argmax"]
            steady: list[float] = []
            for _ in range(max(1, repeats - 1)):
                started = time.perf_counter()
                logits = session.bulk_prefill(prompt, logits_rows=1)
                steady.append(time.perf_counter() - started)
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
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--token-id", type=int, default=DEFAULT_TOKEN_ID)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help=f"comma-separated subset of {','.join(VARIANTS)}",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; expected a subset of {list(VARIANTS)}")
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
    checks: dict[str, Any] = {}
    for variant, record in result["variants"].items():
        entries = [e for e in record["lengths"] if "refused_over_pinned_capacity" not in e]
        refusals = [e for e in record["lengths"] if "refused_over_pinned_capacity" in e]
        checks[variant] = {
            "every_length_finite": all(e["finite"] for e in entries),
            "every_length_generate_completed": all(e["generate_decode_steps"] >= 1 for e in entries),
            "workspace_never_below_prompt": all(
                e["workspace_rows_after_first"] >= e["length"] for e in entries
            ),
            "revisit_reused_workspace": (
                bool(entries)
                and entries[-1]["revisit"]
                and entries[-1]["workspace_rows_after_first"] >= entries[-1]["length"]
            ),
            "growth_rebuilds_recorded": sorted(
                {e["workspace_rows_after_first"] for e in entries}
            ),
            "over_pinned_lengths_refused": all(
                e["refused_over_pinned_capacity"].get("raised") for e in refusals
            ),
            "refused_lengths": [e["length"] for e in refusals],
            "lengths_run": [e["length"] for e in entries],
        }
    result["checks"] = checks
    result["all_checks_passed"] = all(
        all(value for key, value in verdict.items() if isinstance(value, bool))
        for verdict in checks.values()
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1) + "\n")
        print(f"artifact: {args.json}", flush=True)
    print(f"all_checks_passed={result['all_checks_passed']}", flush=True)
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
