#!/usr/bin/env python3
"""A/B the pre-change and current decode plans on one fixed KV prefix per context.

The decode-attribution harness reports timings but not which route ran, and the
route is exactly what this change moves: the production plan's decode scope went
from the staged singleton to key-sliced flash. This probe replays the same
teacher token and KV prefix under each plan's own decode-variant set, so the
route in each row is the one the layer actually resolved -- reported by the
layer's route reporter and by the launcher's decode-selection counter -- and the
timings are attributable to it.

Rows:

* ``plain``  -- strict native chain, the arithmetic oracle for the other two;
* ``staged`` -- the pre-change production plan for this scope;
* ``flash``  -- the current production plan.

Protocol matches ``scripts/gemma4_decode_attribution.py``: fixed teacher token
and KV prefix, rewind before every replay, one untimed warmup, median of
``--repeats`` forward/sample/sync phases. The rewind also makes the replay a
pure decode measurement: the prefill cost is outside every row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from scripts.gemma4_campaign_bench import (
        DEFAULT_ARTIFACT,
        _resolve_generator,
        exact_prompt_ids,
    )
    from hipengine.core.hip import get_hip_runtime
    from hipengine.generation.gemma4_profiles import register_gemma4_gguf_profiles
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import (
        last_prefill_attention_route,
    )

    register_gemma4_gguf_profiles()
    artifact = args.artifact or DEFAULT_ARTIFACT
    llm, runner, loading = _resolve_generator(artifact, max(args.contexts) + 16)
    runtime = get_hip_runtime()
    generator = llm._get_text_generator()

    original = runner.decode_attention_variants
    report = {
        "kind": "gemma4_flash_plan_ab",
        "artifact": str(artifact),
        "protocol": (
            "fixed teacher token and KV prefix, rewind each replay, one untimed "
            "decode warmup, median repeated forward/sample/sync phase wall"
        ),
        "profile_env": os.environ.get("HIPENGINE_EXECUTION_PROFILE"),
        "plan_decode_variants": list(original or ()),
        "repeats": args.repeats,
        "loading": {k: v for k, v in loading.items() if k != "resolution"},
        "cases": [],
    }
    plans = (
        ("plain", ("gemma4_plain",)),
        ("staged", ("gemma4_plain", "gemma4_staged")),
        ("flash", tuple(original or ())),
    )
    try:
        for context in args.contexts:
            ids = exact_prompt_ids(generator.tokenize, context)
            runner.reset()
            prefill = runner.forward(ids)
            runtime.device_synchronize()
            teacher = int(runner.next_token(prefill))
            case = {
                "context_tokens": context,
                "keys": context + 1,
                "teacher_token": teacher,
                "prompt_ids_sha256": hashlib.sha256(
                    np.array(ids, dtype=np.int32).tobytes()).hexdigest(),
                "rows": [],
            }
            for label, variants in plans:
                runner.decode_attention_variants = variants
                runner.rewind(context)
                runner.forward([teacher])
                runtime.device_synchronize()
                samples = []
                for _ in range(args.repeats):
                    runner.rewind(context)
                    runtime.device_synchronize()
                    start = time.perf_counter()
                    logits = runner.forward([teacher])
                    forward_end = time.perf_counter()
                    sampled = int(runner.next_token(logits))
                    sample_end = time.perf_counter()
                    runtime.device_synchronize()
                    end = time.perf_counter()
                    samples.append({
                        "forward_s": forward_end - start,
                        "sample_s": sample_end - forward_end,
                        "synchronize_s": end - sample_end,
                        "wall_s": end - start,
                        "sampled_token": sampled,
                        "logits_sha256": hashlib.sha256(
                            np.asarray(logits, dtype=np.float32).tobytes()
                        ).hexdigest(),
                    })
                row = {
                    "plan": label,
                    "variants": list(variants),
                    "reported_route": last_prefill_attention_route(),
                    "decode_selection": attention.decode_selection(),
                    "argmax": samples[0]["sampled_token"],
                    "logits_sha256": samples[0]["logits_sha256"],
                    "replay_logits_bitwise_equal": len(
                        {s["logits_sha256"] for s in samples}) == 1,
                    "median": {
                        key: statistics.median(s[key] for s in samples)
                        for key in ("forward_s", "sample_s", "synchronize_s", "wall_s")
                    },
                    "samples": samples,
                }
                case["rows"].append(row)
                print(json.dumps({
                    "context": context, "plan": label,
                    "route": row["reported_route"],
                    "selection": row["decode_selection"],
                    "argmax": row["argmax"],
                    "forward_ms": round(row["median"]["forward_s"] * 1e3, 3),
                }), flush=True)
            staged = next(r for r in case["rows"] if r["plan"] == "staged")
            flash = next(r for r in case["rows"] if r["plan"] == "flash")
            plain = next(r for r in case["rows"] if r["plan"] == "plain")
            case["staged_bitwise_equal_plain"] = (
                staged["logits_sha256"] == plain["logits_sha256"])
            case["flash_matches_plain_token"] = flash["argmax"] == plain["argmax"]
            case["staged_over_flash"] = (
                staged["median"]["forward_s"] / flash["median"]["forward_s"])
            report["cases"].append(case)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        runner.decode_attention_variants = original
        llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
