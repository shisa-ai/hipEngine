"""Measure the MTP prefill catch-up cost (ledger ``MT-5``).

The question MT-5 leaves open: what does our MTP prefill catch-up cost, and how
does it scale with prompt length?

Our catch-up (``llama_cpp_mtp_catchup_rows``) keeps the **full prompt's** hidden
rows and builds a right-shifted copy so that draft row ``i`` sees the target
hidden state at ``i - 1``. That is ``O(prompt_tokens x hidden_size)`` host memory
retained, plus a second array of the same size for the shifted result. The
reference approach cited in the ledger keeps a single lagging row (``O(1)``) and
replays the final chunk instead.

This script measures the cost of what we actually do. It is a CPU-only
measurement of the host-side catch-up; it does not include the prefill itself,
the draft, or any device work. Run it as::

    python3 scripts/mtp_prefill_catchup_cost.py --output <path.json>

The measurement is deterministic in its *sizing* terms (bytes are exact) and
noisy only in its timing terms, which are reported as medians over repeats.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hipengine.generation.qwen35_gguf import _llama_cpp_mtp_catchup_rows as catchup_rows

# The bench harness carries its own copy of this function. Measuring the engine's
# implementation is what matters, so the two are compared once here: if they ever
# diverge, this measurement is describing a path the engine does not run.
from scripts.gguf_mtp_bench import (
    llama_cpp_mtp_catchup_rows as harness_catchup_rows,
)

# Qwen3.6-35B-A3B text config: text_config.hidden_size.
DEFAULT_HIDDEN_SIZE = 2048
DEFAULT_TOKENS = (1024, 4096, 16384)


def _measure(
    *, tokens: int, hidden_size: int, repeats: int, warmups: int
) -> dict[str, object]:
    rng = np.random.default_rng(20260929)
    prompt_ids = [int(x) for x in rng.integers(0, 100000, size=tokens)]
    seeds = rng.standard_normal((tokens, hidden_size), dtype=np.float32)

    # The caller already holds `seeds`; the catch-up adds the shifted result.
    retained_bytes = int(seeds.nbytes)
    # Peak extra allocation the catch-up itself creates: the zero row plus the
    # concatenated copy, both transient except for the returned array.
    catchup_peak_bytes = int((tokens * hidden_size * 4) + (hidden_size * 4))

    for _ in range(warmups):
        catchup_rows(prompt_ids, seeds)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out_tokens, out_hidden = catchup_rows(prompt_ids, seeds)
        samples.append((time.perf_counter() - start) * 1e3)

    # Correctness guard: the shift must still be what the unit test pins, and the
    # engine implementation must still agree with the harness copy.
    assert out_tokens == prompt_ids
    assert out_hidden.shape == (tokens, hidden_size)
    assert np.all(out_hidden[0] == 0.0)
    if tokens > 1:
        assert np.array_equal(out_hidden[1], seeds[0])
    harness_tokens, harness_hidden = harness_catchup_rows(prompt_ids, seeds)
    assert harness_tokens == out_tokens
    assert np.array_equal(harness_hidden, out_hidden)

    return {
        "prompt_tokens": tokens,
        "hidden_size": hidden_size,
        "retained_hidden_bytes": retained_bytes,
        "catchup_peak_bytes": catchup_peak_bytes,
        "total_bytes": retained_bytes + catchup_peak_bytes,
        "catchup_ms_samples": samples,
        "catchup_ms_median": statistics.median(samples),
        "catchup_ms_min": min(samples),
        "bytes_per_token": hidden_size * 4,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOKENS),
        help="prompt lengths to measure (default: 1024 4096 16384)",
    )
    args = parser.parse_args()

    cases = [
        _measure(
            tokens=int(t),
            hidden_size=int(args.hidden_size),
            repeats=int(args.repeats),
            warmups=int(args.warmups),
        )
        for t in args.tokens
    ]

    reference_bytes = int(args.hidden_size * 4)
    artifact = {
        "schema": 1,
        "kind": "mtp_prefill_catchup_cost",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "boundary": (
            "host-side MTP prefill catch-up only (llama_cpp_mtp_catchup_rows); "
            "excludes the prefill itself, the draft model, and all device work"
        ),
        "host_hidden_size": int(args.hidden_size),
        "repeats": int(args.repeats),
        "warmups": int(args.warmups),
        "cases": cases,
        "reference_single_row_bytes": reference_bytes,
        "notes": [
            "retained_hidden_bytes is the caller-held prompt hidden state the "
            "catch-up requires; it is O(prompt_tokens x hidden_size).",
            "catchup_peak_bytes is the transient copy the catch-up builds.",
            "The reference approach in the ledger keeps one lagging row and "
            "replays the final chunk, i.e. O(1) hidden state.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")

    for case in cases:
        mib = case["total_bytes"] / (1024 * 1024)
        print(
            f"tokens={case['prompt_tokens']:>6d} "
            f"total={mib:>8.2f} MiB "
            f"catchup_median={case['catchup_ms_median']:>7.3f} ms"
        )
    print(f"reference single row: {reference_bytes / 1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
