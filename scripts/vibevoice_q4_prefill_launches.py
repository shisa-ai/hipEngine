#!/usr/bin/env python3
"""VibeVoice-ASR batched-prefill launch count and output hash.

Counts kernel-wrapper calls made by one batched prefill, by name, and prints a
SHA-256 of the prefill output rows. Run it on two revisions to show that a
change removed launches while leaving the output byte-identical.

The wrapper-call count is a launch count, not a wall-time claim: it is the
quantity a launch-removal change is supposed to move. Whether that moves wall
time is a separate measurement.

Usage:
    python3 scripts/vibevoice_q4_prefill_launches.py [--route q4|bf16] [--repeat]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

GGUF = Path("/tmp/vibevoice-asr-q4km.gguf")
LM_FIXTURE = (Path(__file__).resolve().parent.parent
              / "tests" / "fixtures" / "vibevoice_asr" / "vibevoice_asr_lm.npz")
PINNED_HF_MODEL_ID = "microsoft/VibeVoice-ASR-HF"


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
    parser.add_argument("--route", choices=("q4", "bf16"), default="q4")
    parser.add_argument("--repeat", action="store_true",
                        help="also count device allocations for a repeated prefill")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from hipengine.core.hip import HipRuntime
    from hipengine.core.memory import copy_host_array_to_device, free, malloc
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits

    if not LM_FIXTURE.is_file():
        raise SystemExit(f"missing LM fixture: {LM_FIXTURE}")

    with np.load(LM_FIXTURE) as data:
        lm = {k: data[k] for k in data.files}

    weights = None
    runner = None
    try:
        if args.route == "q4":
            from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
            from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime

            if not GGUF.is_file():
                raise SystemExit(f"missing Q4 GGUF: {GGUF}")
            weights = load_vibevoice_qwen2_q4(GGUF)
            runner = VibevoiceQwen2Q4Runtime(weights, max_context=512)
            kernels = runner.kernels
        else:
            from hipengine.loading.hf_cache import resolve_model_path
            from hipengine.loading.vibevoice_asr import load_vibevoice_qwen2
            from hipengine.runtime.vibevoice_qwen2 import VibevoiceQwen2Runtime

            weights = load_vibevoice_qwen2(str(resolve_model_path(PINNED_HF_MODEL_ID)))
            runner = VibevoiceQwen2Runtime(weights, max_context=512)
            kernels = runner.kernels

        hidden = runner.spec.hidden_size
        rows = _prompt_rows(runner, lm)
        total = len(rows)
        pristine = f32_to_bf16_bits(np.asarray(rows, dtype=np.float32))
        prompt = malloc(total * hidden * 2)
        try:
            def prefill() -> None:
                copy_host_array_to_device(prompt, pristine)
                runner.reset()
                runner.prefill_rows(prompt, total, 0)

            counts: Counter[str] = Counter()
            saved = []

            # bf16_to_f32 lives in the q4 kernel module, not on the runner, so
            # wrap it separately. The artifact's "conversion launches" figure is
            # bf16_to_f32 + f32_to_bf16, which is what a Q6_K ffn_down
            # round-trip removes.
            conversion_targets = [kernels]
            if args.route == "q4":
                import hipengine.kernels.hip_gfx1100.vibevoice.q4 as q4_module

                conversion_targets.append(q4_module)

            for target in conversion_targets:
                for name in dir(target):
                    if name.startswith("_"):
                        continue
                    fn = getattr(target, name)
                    if not callable(fn):
                        continue

                    def wrap(fn=fn, name=name, target=target):
                        def inner(*a, **kw):
                            counts[name] += 1
                            return fn(*a, **kw)
                        return inner

                    saved.append((target, name, fn))
                    setattr(target, name, wrap())
            try:
                prefill()                    # warm any lazy state
                counts.clear()
                t0 = time.perf_counter()
                prefill()
                prefill_ms = (time.perf_counter() - t0) * 1000.0
            finally:
                for target, name, fn in saved:
                    setattr(target, name, fn)

            conversions = counts["bf16_to_f32"] + counts["f32_to_bf16"]

            host = np.empty(total * hidden, dtype=np.uint16)
            from hipengine.core.memory import copy_device_to_host, host_array_ptr

            copy_device_to_host(host_array_ptr(host), prompt, total * hidden * 2)
            digest = hashlib.sha256(host.tobytes()).hexdigest()

            payload: dict[str, object] = {
                "route": args.route,
                "prompt_rows": total,
                "wrapper_calls_per_prefill": sum(counts.values()),
                "bf16_to_f32_launches": counts["bf16_to_f32"],
                "f32_to_bf16_launches": counts["f32_to_bf16"],
                "conversion_launches": conversions,
                "call_breakdown": counts.most_common(),
                "prefill_ms": round(prefill_ms, 1),
                "hidden_rows_sha256": digest,
            }
            print(f"route                        : {args.route}")
            print(f"prompt rows                  : {total}")
            print(f"kernel-wrapper calls/prefill : {sum(counts.values())}")
            print(f"conversion launches (2 conv) : {conversions} "
                  f"(bf16_to_f32 {counts['bf16_to_f32']}, "
                  f"f32_to_bf16 {counts['f32_to_bf16']})")
            for name, count in counts.most_common(8):
                print(f"    {name:32s} {count}")
            print(f"prefill ms (2 calls)         : {prefill_ms:.1f}")
            print(f"hidden_rows sha256           : {digest}")

            if args.repeat:
                allocations: list[int] = []
                original = HipRuntime.malloc

                def counted(self, nbytes):
                    allocations.append(int(nbytes))
                    return original(self, nbytes)

                prefill()                    # ensure the arena exists
                HipRuntime.malloc = counted
                try:
                    allocations.clear()
                    prefill()
                    repeated = list(allocations)
                finally:
                    HipRuntime.malloc = original
                payload["device_allocations_on_repeat"] = len(repeated)
                payload["device_allocation_bytes_on_repeat"] = repeated
                print(f"device allocations, 2nd call : {len(repeated)} "
                      f"({sum(repeated) / 1e6:.1f} MB)")
        finally:
            free(prompt)
    finally:
        if runner is not None:
            runner.close()
        if weights is not None:
            closer = getattr(weights, "close", None)
            if closer is not None:
                closer()

    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
