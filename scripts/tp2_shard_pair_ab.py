"""Paired same-session A/B for a TP2 prefill change.

The reason this exists as a script rather than a one-off: two earlier attempts to
measure the same change disagreed in sign, and both were wrong in a way that a
better protocol fixes.

- One measured the change across *different sessions* (a fresh build per arm), so
  session-build variance sat on top of the effect.
- One timed ``bulk_prefill`` without ``logits_rows``, which projects the head for
  every prompt row instead of the last one. That is not the product path, and the
  ~560 ms of extra head work dominated the denominator, diluting a real ~30 ms
  effect into a plausible-looking +3% that meant nothing.

So this driver pins the two things that made those measurements lie: **one
session**, and an explicit ``--logits-rows`` that defaults to the product path.

It also interleaves the arms (A/B/A/B) rather than running them in blocks. Blocked
arms let any warm-up or clock drift land entirely on one side; interleaving puts
it on both, and the reported drift is the check.

Usage::

    # toggle a backend capability set
    python scripts/tp2_shard_pair_ab.py \\
        --capability GGUF_Q4_DUAL_SILU_PREFILL_OUT_FEATURES=17408,8704 \\
        --prompt-tokens 512 --rounds 4

    # control: the same arms with no capability change, to measure the noise floor
    python scripts/tp2_shard_pair_ab.py --prompt-tokens 512 --rounds 4
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

sys.path.insert(0, "/home/lhl/hipEngine-main")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument(
        "--logits-rows",
        type=int,
        default=1,
        help="1 is the product path; 512 is the full-row control, which is a different workload",
    )
    parser.add_argument("--rounds", type=int, default=4, help="A/B pairs; the total is 2x this")
    parser.add_argument(
        "--capability",
        default=None,
        help="NAME=v1,v2 backend capability for the B arm; omit for a noise-floor control",
    )
    parser.add_argument("--reduce-mode", default="device")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession
    import hipengine.kernels.hip_gfx1100 as backend

    runtime = get_hip_runtime()
    devices = tuple(int(part) for part in str(args.devices).split(","))

    arm_a: Any = None
    arm_b: Any = None
    if args.capability:
        name, _, value = args.capability.partition("=")
        widths = frozenset(int(part) for part in value.split(",") if part.strip())
        if not hasattr(backend, name):
            raise SystemExit(f"the gfx1100 backend has no capability named {name!r}")
        arm_a = getattr(backend, name)
        arm_b = widths
        print(f"capability {name}: A={sorted(arm_a)}  B={sorted(arm_b)}")
    else:
        print("no --capability given: both arms are identical, so this measures the noise floor")

    session = MlpTP2GenerationSession(
        args.model,
        devices=devices,
        mode="tp2",
        max_sequence_length=int(args.max_sequence_length),
        bulk_prefill=True,
        bulk_prefill_rows=int(args.prompt_tokens),
        reduce_mode=str(args.reduce_mode),
    )
    prompt = [9707] * int(args.prompt_tokens)

    def run() -> float:
        runtime.device_synchronize()
        started = time.perf_counter()
        session.bulk_prefill(prompt, logits_rows=int(args.logits_rows))
        runtime.device_synchronize()
        return (time.perf_counter() - started) * 1000.0

    try:
        run()  # warm-up, unmeasured
        results: dict[str, list[float]] = {"A": [], "B": []}
        order = ["A", "B"] * int(args.rounds)
        for label in order:
            if arm_b is not None:
                setattr(backend, args.capability.partition("=")[0],
                        arm_b if label == "B" else arm_a)
            results[label].append(run())
    finally:
        if arm_a is not None:
            setattr(backend, args.capability.partition("=")[0], arm_a)
        session.close()

    a, b = results["A"], results["B"]
    median_a, median_b = statistics.median(a), statistics.median(b)
    # Per-round paired deltas: the arms are adjacent in time, so pairing removes
    # any slow drift that a block design would attribute to the change.
    paired = [100.0 * (x / y - 1.0) for x, y in zip(a, b)]
    report = {
        "schema": 1,
        "kind": "tp2_prefill_paired_ab",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "model": args.model,
        "devices": list(devices),
        "prompt_tokens": int(args.prompt_tokens),
        "logits_rows": int(args.logits_rows),
        "reduce_mode": str(args.reduce_mode),
        "capability": args.capability,
        "rounds": int(args.rounds),
        "arm_a_wall_ms": [round(v, 3) for v in a],
        "arm_b_wall_ms": [round(v, 3) for v in b],
        "arm_a_median_ms": round(median_a, 3),
        "arm_b_median_ms": round(median_b, 3),
        "throughput_delta_percent": round(100.0 * (median_a / median_b - 1.0), 3),
        "paired_deltas_percent": [round(v, 3) for v in paired],
        "paired_all_same_sign": all(v > 0 for v in paired) or all(v < 0 for v in paired),
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")

    print(f"  A {[round(v, 1) for v in a]}  median {median_a:.1f} ms")
    print(f"  B {[round(v, 1) for v in b]}  median {median_b:.1f} ms")
    print(f"  B vs A: {report['throughput_delta_percent']:+.2f}% throughput")
    print(f"  paired per-round deltas: {report['paired_deltas_percent']}  "
          f"same sign: {report['paired_all_same_sign']}")
    if args.capability is None:
        print("  (noise floor: a same-sign result here means the protocol is not resolving "
              "the change, not that there is one)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
