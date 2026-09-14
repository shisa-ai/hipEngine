#!/usr/bin/env python3
"""VibeVoice-ASR batched prefill scratch A/B, allocation accounting, output hash.

Compares the shipped per-runner prefill scratch arena against the legacy
behaviour, in which ``q4_prefill`` called ``malloc`` once per scratch buffer and
freed the batch in a ``finally`` block.

Where the legacy cost enters the timing
---------------------------------------
The legacy path freed its batch at the *end* of each prefill, inside the same
timed region. This harness cannot call the removed code, so it substitutes a
shim (``PerCallScratch``) whose ``take()`` frees the *previous* call's batch and
then allocates the new one. A timed call therefore contains exactly one
malloc-batch and one free-batch, which is the same work the legacy path did
inside its own timed region; only the phase within the call differs. The first
timed call in each lane has no preceding batch to free, so it slightly
*underestimates* the legacy cost, making the measured delta conservative.

Input restoration
-----------------
``prefill_rows`` overwrites its input buffer with its post-layer-stack result.
Every pass here re-uploads the pristine prompt rows before calling, or each pass
would consume the previous pass's output and the output hash would depend on the
iteration count.

Timing boundaries
-----------------
Only the ``prefill_rows`` call is timed; the input re-upload happens before the
timer starts, and the output hash after it stops. Lanes alternate within one
process, so host drift cannot favour a lane.

Usage:
    python3 scripts/vibevoice_q4_prefill_scratch_ab.py [--trials N] [--json PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

GGUF = Path("/tmp/vibevoice-asr-q4km.gguf")
LM_FIXTURE = (Path(__file__).resolve().parent.parent
              / "tests" / "fixtures" / "vibevoice_asr" / "vibevoice_asr_lm.npz")


class PerCallScratch:
    """Legacy behaviour: one malloc per buffer per call, freed on the next call."""

    def __init__(self) -> None:
        self._previous: list = []

    def take(self, sizes):
        from hipengine.core.memory import free, malloc

        for buffer in self._previous:
            free(buffer)
        self._previous = [malloc(int(size)) for size in sizes]
        return self._previous

    def close(self) -> None:
        from hipengine.core.memory import free

        for buffer in self._previous:
            free(buffer)
        self._previous = []


def _prompt_rows(runner, lm) -> list[np.ndarray]:
    input_ids = np.asarray(lm["input_ids"])[0]
    positions = np.asarray(lm["audio_placeholder_positions"])
    audio = lm["audio_embeds"].astype(np.float32)
    rows = [runner.embed_row(int(t)) for t in input_ids]
    for p in positions:
        rows[p] = audio[p - positions[0]]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--lane", choices=("both", "legacy", "reused"), default="both",
                        help="restrict to one lane, so a separate-process "
                             "before/after run is reproducible too")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from hipengine.core.hip import HipRuntime
    from hipengine.core.memory import copy_host_array_to_device, free, malloc
    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits
    from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime

    if not GGUF.is_file():
        raise SystemExit(f"missing Q4 GGUF: {GGUF}")
    if not LM_FIXTURE.is_file():
        raise SystemExit(f"missing LM fixture: {LM_FIXTURE}")

    with np.load(LM_FIXTURE) as data:
        lm = {k: data[k] for k in data.files}

    weights = load_vibevoice_qwen2_q4(GGUF)
    runner = VibevoiceQwen2Q4Runtime(weights, max_context=512)
    shipped = runner._prefill_scratch
    if shipped is None:
        raise SystemExit("Q4 runner exposes no prefill scratch arena; this A/B is stale")

    payload: dict[str, object] = {}
    try:
        hidden = runner.spec.hidden_size
        rows = _prompt_rows(runner, lm)
        total = len(rows)
        pristine = f32_to_bf16_bits(np.asarray(rows, dtype=np.float32))
        prompt = malloc(total * hidden * 2)

        def run(scratch) -> str:
            runner._prefill_scratch = scratch
            copy_host_array_to_device(prompt, pristine)   # outside the timer
            runner.reset()
            t0 = time.perf_counter()
            runner.prefill_rows(prompt, total, 0)
            elapsed = (time.perf_counter() - t0) * 1000.0
            host = np.empty(total * hidden, dtype=np.uint16)
            from hipengine.core.memory import copy_device_to_host, host_array_ptr

            copy_device_to_host(host_array_ptr(host), prompt, total * hidden * 2)
            digest = hashlib.sha256(host.tobytes()).hexdigest()
            return elapsed, digest

        # Allocation accounting: how many device allocations does one prefill make?
        allocations: list[int] = []
        original_malloc = HipRuntime.malloc

        def counted_malloc(self, nbytes):
            allocations.append(int(nbytes))
            return original_malloc(self, nbytes)

        run(shipped)                       # ensure the arena exists
        HipRuntime.malloc = counted_malloc
        try:
            allocations.clear()
            run(shipped)
            reused_allocations = list(allocations)
        finally:
            HipRuntime.malloc = original_malloc

        legacy_allocations: list[int] = []
        shim_probe = PerCallScratch()
        HipRuntime.malloc = counted_malloc
        try:
            allocations.clear()
            run(shim_probe)
            legacy_allocations = list(allocations)
        finally:
            HipRuntime.malloc = original_malloc
        shim_probe.close()

        for _ in range(args.warmup):
            if args.lane in ("both", "legacy"):
                run(PerCallScratch())
            if args.lane in ("both", "reused"):
                run(shipped)

        legacy: list[float] = []
        reused: list[float] = []
        legacy_digests: list[str] = []
        reused_digests: list[str] = []
        for _ in range(args.trials):
            plan = []
            if args.lane in ("both", "legacy"):
                plan.append((PerCallScratch(), legacy, legacy_digests))
            if args.lane in ("both", "reused"):
                plan.append((shipped, reused, reused_digests))
            for scratch, bucket, digests in plan:
                ms, digest = run(scratch)
                bucket.append(ms)
                digests.append(digest)

        lm_ = statistics.median(legacy) if legacy else None
        rm_ = statistics.median(reused) if reused else None
        both = lm_ is not None and rm_ is not None
        identical = True
        if both:
            identical = (set(legacy_digests) == set(reused_digests)
                         and len(set(reused_digests)) == 1)
        else:
            only = reused_digests or legacy_digests
            identical = len(set(only)) == 1

        print(f"prompt rows                     : {total}")
        print(f"device allocations, legacy      : {len(legacy_allocations)} "
              f"({sum(legacy_allocations) / 1e6:.1f} MB)")
        print(f"device allocations, reused arena: {len(reused_allocations)} "
              f"({sum(reused_allocations) / 1e6:.1f} MB)")
        if lm_ is not None:
            print(f"legacy per-call malloc/free     : {lm_:7.1f} ms  "
                  f"{[round(v, 1) for v in legacy]}")
        if rm_ is not None:
            print(f"reused arena                    : {rm_:7.1f} ms  "
                  f"{[round(v, 1) for v in reused]}")
        if both:
            print(f"delta                           : {lm_ - rm_:+7.1f} ms "
                  f"({(lm_ / rm_ - 1) * 100:+.2f}%)")
        print(f"output bit-identical            : {identical} "
              f"sha={(reused_digests or legacy_digests)[0][:32]}")

        payload = {
            "prompt_rows": total,
            "trials": args.trials,
            "lane": args.lane,
            "legacy_device_allocations": len(legacy_allocations),
            "legacy_allocation_bytes": legacy_allocations,
            "reused_device_allocations": len(reused_allocations),
            "reused_allocation_bytes": reused_allocations,
            "legacy_ms": round(lm_, 1) if lm_ is not None else None,
            "legacy_trials": [round(v, 1) for v in legacy],
            "reused_ms": round(rm_, 1) if rm_ is not None else None,
            "reused_trials": [round(v, 1) for v in reused],
            "delta_ms": round(lm_ - rm_, 1) if both else None,
            "delta_pct": round((lm_ / rm_ - 1) * 100, 2) if both else None,
            "output_bit_identical": identical,
            "output_sha256": (reused_digests or legacy_digests)[0],
            "arena_capacity_bytes": shipped._arena.capacity_bytes if shipped._arena else None,
        }
        free(prompt)
    finally:
        runner._prefill_scratch = shipped
        runner.close()
        weights.close()

    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.json}")

    if not payload.get("output_bit_identical"):
        raise SystemExit("FAIL: the two lanes produced different prefill output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
