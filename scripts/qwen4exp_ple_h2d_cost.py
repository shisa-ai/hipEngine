"""Measure the PLE staging + H2D copy cost at its real shapes (ledger ``PL-2``).

PL-2 asks to "first measure exposed copy/wait time" before attempting an
asynchronous double-buffer PLE staging path. The per-request PLE H2D payload is
small, and the in-engine telemetry that reports it is instrumented (the artifact
carrying it explicitly excludes its wall from performance evidence). This
measures the copy itself, at the two real shapes, with no model loaded:

* prefill: one chunk of ``prompt_tokens`` x ``hidden_size`` fp32 rows. The
  engine's prefill path (``Qwen4ExpGGUFResidentModelRunner``) copies this but
  does **not** record it in PLE telemetry, so the telemetry understates the
  request total.
* decode: one ``hidden_size`` fp32 row per transition, which *is* recorded.

Both are host->device copies out of a pinned staging buffer, on unified memory.

Run::

    python3 scripts/qwen4exp_ple_h2d_cost.py --output <path.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc

# Qwen3.8-Flash-Next UD-Q4_K_XL.
DEFAULT_HIDDEN = 2560
DEFAULT_PROMPT_TOKENS = 512
DEFAULT_DECODE_STEPS = 128


def _measure_copy(*, rows: int, hidden: int, repeats: int, warmups: int, runtime) -> dict:
    """Time a pinned-host -> device copy of ``rows x hidden`` fp32."""
    host = np.zeros((rows, hidden), dtype=np.float32)
    nbytes = host.nbytes
    device = malloc(nbytes, runtime=runtime)
    try:
        def one() -> None:
            copy_host_to_device(
                device, host_array_ptr(host), nbytes, runtime=runtime
            )

        for _ in range(warmups):
            one()
        runtime.device_synchronize()

        samples = []
        for _ in range(repeats):
            runtime.device_synchronize()
            start = time.perf_counter_ns()
            one()
            runtime.device_synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e3)  # us

        median_us = statistics.median(samples)
        return {
            "rows": rows,
            "nbytes": nbytes,
            "median_us": median_us,
            "min_us": min(samples),
            "samples_us": samples,
            "effective_gb_per_s": (nbytes / (median_us / 1e6)) / 1e9,
        }
    finally:
        free(device, runtime=runtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_DECODE_STEPS)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()

    runtime = get_hip_runtime()
    hidden = int(args.hidden)

    prefill = _measure_copy(
        rows=int(args.prompt_tokens),
        hidden=hidden,
        repeats=int(args.repeats),
        warmups=int(args.warmups),
        runtime=runtime,
    )
    decode = _measure_copy(
        rows=1,
        hidden=hidden,
        repeats=int(args.repeats),
        warmups=int(args.warmups),
        runtime=runtime,
    )

    steps = int(args.decode_steps)
    decode_total_us = decode["median_us"] * steps
    request_total_us = prefill["median_us"] + decode_total_us

    artifact = {
        "schema": 1,
        "kind": "qwen4exp_ple_h2d_cost",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "boundary": (
            "pinned host -> device copy at the PLE staging shapes; excludes the "
            "PLE row gather/dequantize (CPU, mmap table) and the staging memcpy "
            "into the pinned ring"
        ),
        "hidden": hidden,
        "prompt_tokens": int(args.prompt_tokens),
        "decode_steps": steps,
        "repeats": int(args.repeats),
        "warmups": int(args.warmups),
        "prefill_copy": prefill,
        "decode_copy": decode,
        "per_request": {
            "prefill_copy_us": prefill["median_us"],
            "decode_copies_us": decode_total_us,
            "total_us": request_total_us,
            "total_bytes": prefill["nbytes"] + decode["nbytes"] * steps,
        },
        "notes": [
            "The engine's prefill PLE copy does not call record_h2d, so in-engine "
            "PLE telemetry counts only the decode copies and understates the "
            "per-request H2D payload.",
            "This is the copy only; it is not a claim about an asynchronous path.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")

    print(
        f"prefill copy : {prefill['rows']:>4d} rows x {hidden} fp32 = "
        f"{prefill['nbytes'] / 1048576:>6.2f} MiB  "
        f"{prefill['median_us']:>9.2f} us  {prefill['effective_gb_per_s']:>7.1f} GB/s"
    )
    print(
        f"decode copy  :    1 row  x {hidden} fp32 = "
        f"{decode['nbytes'] / 1024:>6.2f} KiB  "
        f"{decode['median_us']:>9.2f} us  {decode['effective_gb_per_s']:>7.1f} GB/s"
    )
    print(
        f"per request  : {steps} decode copies + 1 prefill copy = "
        f"{request_total_us / 1e3:>7.3f} ms, "
        f"{(prefill['nbytes'] + decode['nbytes'] * steps) / 1048576:.2f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
