#!/usr/bin/env python3
"""Report the packed-AR prefill row slab and where a wave stops fitting in it.

Why this exists. The published C1-C8 AR rows lose at C2 and C6-C8, and the committed
decomposition says our admission wall grows ~7.7x across C1->C8 while llama.cpp grows 3.8x, i.e.
prompt processing does not amortize across a wave. Grouping was promoted and helped a lot
(C7 +176%, C8 +157% in `2026-08-30-w7900-q4km-c1c8-hipengine-prefill-row-grouped.json`) but the
post-grouping row is *non-monotone*: C7 = 426.692 tok/s beats C8 = 397.655 tok/s. `_
plan_packed_ar_prefill_chunks` explains the shape exactly: while `total_rows <= row_capacity` every
slot lands in one chunk, and past that boundary the planner switches to slot-fair rounds, which
serializes. So the width where a wave starts splitting is a physical constant of the resident
session, not a scheduler choice.

This script prints that constant and shows the plan for wave sizes 1..8 at a stated prompt length,
so the cliff is measured rather than inferred from a tok/s dip. It loads weights and reads buffer
geometry; it does not benchmark, and it never mutates engine behaviour.

Usage:
    python3 scripts/gguf_packed_prefill_slab_probe.py \
        --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf --prompt-tokens 36
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance is best effort
        return "unknown"


def _plan_shape(
    planner: Any,
    *,
    lanes: int,
    prompt_tokens: int,
    token_pool: Sequence[int],
    row_capacity: int,
) -> dict[str, Any]:
    """Plan one wave and describe its chunk shape (pure CPU, no device work)."""

    prompts = tuple(
        tuple(int(token_pool[(lane + index) % len(token_pool)]) for index in range(prompt_tokens))
        for lane in range(lanes)
    )
    chunks = planner(prompts, row_capacity=row_capacity)
    rows = [int(chunk.rows) for chunk in chunks]
    slots = [len(tuple(chunk.slot_indices)) for chunk in chunks]
    return {
        "lanes": lanes,
        "total_rows": lanes * prompt_tokens,
        "chunk_count": len(chunks),
        "chunk_rows": rows,
        "slots_per_chunk": slots,
        # One chunk means the wave prefills together; more means serial slot-fair rounds.
        "single_chunk": len(chunks) == 1,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=36,
        help="prompt length per lane; 36 matches the canonical mtp-bench suite encoding",
    )
    parser.add_argument("--max-sequence-length", type=int, default=0, help="0 = engine default")
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, help="write the report here")
    args = parser.parse_args(argv)

    if not _hip_available():
        print("SKIP: libamdhip64.so is unavailable; this probe needs the ROCm runtime.")
        return 2
    if not args.model.is_file():
        print(f"FAIL: model not found: {args.model}")
        return 2

    from hipengine.runtime.qwen35_gguf_runner import (
        Qwen35GGUFResidentSession,
        _plan_packed_ar_prefill_chunks,
    )

    # Valid ids only; the shape is what matters, and the ids are stated in the artifact.
    token_pool = (760, 4087, 369, 220, 16, 17, 13, 2424, 11, 5346, 264, 26864)
    kwargs: dict[str, Any] = {"require_cached_build": bool(args.require_cached_build)}
    if args.max_sequence_length:
        kwargs["max_sequence_length"] = int(args.max_sequence_length)

    started = time.perf_counter()
    with Qwen35GGUFResidentSession(args.model, **kwargs) as session:
        runner = session.runner
        scratch = getattr(session, "scratch", None)
        capacity = int(getattr(scratch, "max_positions", 0) or 0)
        slab_rows = int(session._bulk_prefill_scratch.rows)
        planned_capacity = int(session._prefill_scratch_rows(capacity))
        plans = [
            _plan_shape(
                _plan_packed_ar_prefill_chunks,
                lanes=width,
                prompt_tokens=int(args.prompt_tokens),
                token_pool=token_pool,
                row_capacity=slab_rows,
            )
            for width in WIDTHS
        ]
        first_split = next((p["lanes"] for p in plans if not p["single_chunk"]), None)

    report = {
        "schema": "hipengine.gguf.packed-prefill.slab-probe.2026-08-30",
        "date": time.strftime("%Y-%m-%d"),
        "kind": "geometry-probe",
        "question": (
            "at which wave size does packed AR prefill stop fitting in one chunk, and does that "
            "boundary line up with the non-monotone grouped admission row (C7 > C8)?"
        ),
        "model": {"path": str(args.model), "size_bytes": args.model.stat().st_size},
        "host": {
            "hostname": platform.node(),
            "gpu": "AMD Radeon Pro W7900 (gfx1100)" if _hip_available() else "unavailable",
            "cpu": platform.processor() or "unknown",
        },
        "software": {"head": _git_rev(), "python": platform.python_version()},
        "protocol": {
            "command": " ".join([sys.executable, Path(__file__).name, *list(argv or ())]),
            "prompt_tokens_per_lane": int(args.prompt_tokens),
            "token_ids": "cycled valid ids from the canonical verifier prompt defaults",
            "note": "geometry only: no timed arm, no engine flag, no state mutation",
        },
        "measurements": {
            "scratch_max_positions": capacity,
            "prefill_scratch_rows_planned": planned_capacity,
            "bulk_prefill_scratch_rows": slab_rows,
            "linear_layer_chunk_size": int(session._linear_prefill_layer_chunk_size(capacity)),
            "full_attention_layer_chunk_size": int(
                session._full_attention_prefill_layer_chunk_size(capacity)
            ),
            "plans": plans,
            "first_splitting_wave": first_split,
            "wall_seconds": round(time.perf_counter() - started, 3),
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
