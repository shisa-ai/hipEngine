"""Attribute the host-side cost of one Gemma 4 decode step (punchlist X7).

X7's Check names the work: ``Gemma4Runner._forward_block`` copies vocabulary
logits to the host and applies NumPy softcap, and ``next_token`` selects on the
host -- "Attribute those costs directly." X5 records its own status as
"unknown until host/timeline attribution", so this measurement is the stated
prerequisite for both rows.

Three host operations sit on the decode path, all at the real vocabulary width:

1. ``copy_device_to_host`` of ``vocab * float32`` (262144 x 4 = 1 MiB),
   ``gemma4.py:907``
2. ``np.tanh(logits / cap) * cap`` when ``final_logit_softcapping`` is set,
   ``gemma4.py:915``
3. ``np.argmax`` over the row in ``next_token``, ``gemma4.py:935``

Each is timed directly against its real shape rather than inferred from a
before/after difference, so the numbers do not absorb unrelated work. The
device buffer is real: the copy is measured against GPU memory, not a
simulation of it.

Run: ``.venv/bin/python scripts/gemma4_host_logits_cost.py [--steps N]``
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

# The production vocabulary, from the same artifact the campaign gates use.
VOCAB = 262144
BYTES = VOCAB * 4
# final_logit_softcapping on the 26B-A4B artifact; the arithmetic shape does not
# depend on the exact value, but use a realistic one so tanh's argument range
# matches production rather than saturating immediately.
SOFTCAP = 30.0


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[index]


def _time(fn, repeats: int) -> list[float]:
    """Microseconds per call, first call discarded as warm-up."""
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e6)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200, help="timed repeats")
    parser.add_argument(
        "--decode-ms",
        type=float,
        default=None,
        help="decode step time to express the host share against, if known",
    )
    args = parser.parse_args()
    repeats = max(20, args.steps)

    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )

    rng = np.random.default_rng(20260929)
    source = rng.standard_normal(VOCAB).astype(np.float32)
    logits = np.empty(VOCAB, dtype=np.float32)
    cap = np.float32(SOFTCAP)

    buffer = malloc(source.nbytes)
    try:
        copy_host_to_device(buffer, host_array_ptr(source), source.nbytes)

        copy_us = _time(
            lambda: copy_device_to_host(host_array_ptr(logits), buffer, source.nbytes),
            repeats,
        )
        # The copy must actually be observed before the buffer is reused.
        copy_us_sync = _time(
            lambda: (
                copy_device_to_host(host_array_ptr(logits), buffer, source.nbytes),
                float(logits[0]),
            )[1],
            repeats,
        )
        softcap_us = _time(
            lambda: (np.tanh(logits / cap) * cap).astype(np.float32), repeats
        )
        argmax_us = _time(lambda: int(np.argmax(logits)), repeats)
        # The greedy path in next_token also pays a reshape/asarray; measure
        # the whole expression a caller actually executes.
        greedy_us = _time(
            lambda: int(np.argmax(np.asarray(logits, dtype=np.float32).reshape(-1))),
            repeats,
        )
    finally:
        free(buffer)

    def report(name: str, samples: list[float]) -> float:
        mean = statistics.fmean(samples)
        print(
            f"  {name:<34} mean {mean:8.2f} us   "
            f"p50 {_percentile(samples, 0.5):8.2f}   "
            f"p95 {_percentile(samples, 0.95):8.2f}   "
            f"max {max(samples):8.2f}"
        )
        return mean

    print(f"host cost of one decode step, vocab={VOCAB} f32 ({BYTES / 1024:.0f} KiB)")
    print(f"  {repeats} timed repeats each\n")

    d2h = report("device->host logits copy", copy_us)
    d2h_sync = report("copy + observe (synced)", copy_us_sync)
    soft = report("softcap  tanh(x/c)*c", softcap_us)
    argm = report("argmax (next_token)", argmax_us)
    greedy = report("greedy path as written", greedy_us)

    host_total = d2h_sync + soft + greedy
    print(f"\n  host total per decode step (synced copy + softcap + greedy):")
    print(f"    {host_total:.2f} us = {host_total / 1000:.3f} ms")

    if args.decode_ms:
        step_us = args.decode_ms * 1000
        print(f"\n  against a {args.decode_ms:.2f} ms decode step:")
        print(f"    host share = {100 * host_total / step_us:.2f}%")
        print(
            f"    of which the transfer is {100 * d2h_sync / step_us:.2f}% "
            f"and NumPy softcap+argmax {100 * (soft + greedy) / step_us:.2f}%"
        )

    print(
        "\n  note: the unsynced copy returns before the DMA completes; the\n"
        "  synced figure is the one that bounds what a host-resident sampler\n"
        "  must wait for. Softcap and argmax cannot start until it lands."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())