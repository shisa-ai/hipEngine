"""Attribute the capacity-sensitive KV reset cost (ledger ``RT-4``).

``Qwen4ExpGGUFResidentModelRunner.reset`` zeroes the **full** key and value
caches for every QSA attention state, regardless of how many tokens the request
actually wrote::

    self.runtime.memset(attention.key_cache.ptr, 0, attention.key_cache.nbytes)
    self.runtime.memset(attention.value_cache.ptr, 0, attention.value_cache.nbytes)

The caches are sized by capacity, so the reset cost is ``capacity x
bf16_kv_bytes_per_token`` and is paid in full even by a request that used a
fraction of it. RT-4 asks for that cost to be attributed separately at short and
native capacities before anyone changes the clearing contract.

This measures the memset itself - the dominant term in ``reset()`` - over a
capacity ladder, and reports what a request that wrote ``W`` tokens wastes.

It allocates only the KV-sized buffers, not the model, so it does not need a
resident model and does not disturb other work. Run::

    python3 scripts/qwen4exp_kv_reset_cost.py --output <path.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free, malloc

# Qwen3.8-Flash-Next UD-Q4_K_XL: 12 QSA layers x 2 kv heads x 256 key length x
# 2 (key+value) x 2 bytes (bf16) = 24576 bytes per token. Read from the GGUF
# metadata by the caller and passed in, so a model change is visible here.
DEFAULT_BYTES_PER_TOKEN = 24576
DEFAULT_CAPACITIES = (1024, 4096, 16384, 65536, 131072)


def _measure(
    *, capacity: int, bytes_per_token: int, repeats: int, warmups: int, runtime
) -> dict[str, object]:
    nbytes = capacity * bytes_per_token
    key = malloc(nbytes, runtime=runtime)
    value = malloc(nbytes, runtime=runtime)
    try:
        def one_reset() -> None:
            # Exactly what reset() does per attention state.
            runtime.memset(key.ptr, 0, key.nbytes)
            runtime.memset(value.ptr, 0, value.nbytes)

        for _ in range(warmups):
            one_reset()
        runtime.device_synchronize()

        samples = []
        for _ in range(repeats):
            runtime.device_synchronize()
            start = time.perf_counter()
            one_reset()
            runtime.device_synchronize()
            samples.append((time.perf_counter() - start) * 1e3)

        total_cleared = 2 * nbytes
        median = statistics.median(samples)
        return {
            "capacity_tokens": capacity,
            "bytes_per_cache": nbytes,
            "bytes_cleared_per_reset": total_cleared,
            "reset_ms_samples": samples,
            "reset_ms_median": median,
            "reset_ms_min": min(samples),
            "effective_gb_per_s": (total_cleared / (median / 1e3)) / 1e9,
        }
    finally:
        free(value, runtime=runtime)
        free(key, runtime=runtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bytes-per-token", type=int, default=DEFAULT_BYTES_PER_TOKEN)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument(
        "--capacities", type=int, nargs="+", default=list(DEFAULT_CAPACITIES)
    )
    parser.add_argument(
        "--written-tokens",
        type=int,
        default=512,
        help="tokens a 'short' request wrote, for the waste fraction",
    )
    args = parser.parse_args()

    runtime = get_hip_runtime()
    cases = [
        _measure(
            capacity=int(c),
            bytes_per_token=int(args.bytes_per_token),
            repeats=int(args.repeats),
            warmups=int(args.warmups),
            runtime=runtime,
        )
        for c in args.capacities
    ]

    written = int(args.written_tokens)
    for case in cases:
        capacity = int(case["capacity_tokens"])
        used = min(written, capacity)
        case["written_tokens"] = used
        case["wasted_bytes"] = (capacity - used) * int(args.bytes_per_token) * 2
        case["wasted_fraction"] = (capacity - used) / capacity
        case["wasted_ms_estimate"] = case["reset_ms_median"] * case["wasted_fraction"]

    artifact = {
        "schema": 1,
        "kind": "qwen4exp_kv_reset_cost",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "boundary": (
            "the K/V memset that dominates Qwen4ExpGGUFResidentModelRunner.reset; "
            "excludes the recurrent state zero, index resets and position writes"
        ),
        "bytes_per_token": int(args.bytes_per_token),
        "repeats": int(args.repeats),
        "warmups": int(args.warmups),
        "written_tokens": written,
        "cases": cases,
        "notes": [
            "The caches are capacity-sized, so the reset is O(capacity) even for "
            "a request that wrote a handful of tokens.",
            "wasted_bytes/wasted_ms_estimate are what a request that wrote "
            "--written-tokens pays for storage it never touched.",
            "This measures the memset only; it does not claim a saving from any "
            "alternative clearing contract.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")

    for case in cases:
        print(
            f"capacity={case['capacity_tokens']:>7d} "
            f"cleared={case['bytes_cleared_per_reset'] / 1048576:>8.1f} MiB "
            f"median={case['reset_ms_median']:>8.3f} ms "
            f"{case['effective_gb_per_s']:>7.1f} GB/s "
            f"wasted={case['wasted_fraction'] * 100:>5.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
