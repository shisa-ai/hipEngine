#!/usr/bin/env python3
"""Reproduce the pre-change decode route on the same KV prefix as the parity run.

The three-arm logits parity script forces each route by re-registering the flash
variant's key. That answers "does flash agree with the strict chain", but it does
not answer "what did the production plan run before flash existed", because the
plan's own selection is what changed. This probe replays one fixed context with
the plan's decode variant set replaced, so each row is exactly what that plan
would have launched:

* ``plain``  -- no scoped candidate, the strict native chain;
* ``staged`` -- the pre-change production plan (staged registered, no flash);
* ``flash``  -- the current production plan.

Each row records the launcher's own decode-selection counter, so the route that
ran is reported rather than assumed, and the logits hash, so a row can be
matched against a stored artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--context", type=int, default=32768)
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
    llm, runner, loading = _resolve_generator(artifact, args.context + 16)
    runtime = get_hip_runtime()
    generator = llm._get_text_generator()
    ids = exact_prompt_ids(generator.tokenize, args.context)
    runner.reset()
    prefill = runner.forward(ids)
    runtime.device_synchronize()
    teacher = int(runner.next_token(prefill))

    original = runner.decode_attention_variants
    report = {
        "kind": "gemma4_flash_route_baseline_probe",
        "artifact": str(artifact),
        "context_tokens": args.context,
        "keys": args.context + 1,
        "teacher_token": teacher,
        "prompt_ids_sha256": hashlib.sha256(
            np.array(ids, dtype=np.int32).tobytes()).hexdigest(),
        "profile_env": os.environ.get("HIPENGINE_EXECUTION_PROFILE"),
        "plan_decode_variants": list(original or ()),
        "loading": {k: v for k, v in loading.items() if k != "resolution"},
        "rows": [],
    }
    try:
        for label, variants in (
            ("plain", ("gemma4_plain",)),
            ("staged", ("gemma4_plain", "gemma4_staged")),
            ("flash", tuple(original or ())),
        ):
            runner.decode_attention_variants = variants
            runner.rewind(args.context)
            runtime.device_synchronize()
            logits = np.asarray(runner.forward([teacher]), dtype=np.float32).reshape(-1)
            runtime.device_synchronize()
            row = {
                "plan": label,
                "variants": list(variants),
                "decode_selection": attention.decode_selection(),
                "reported_route": last_prefill_attention_route(),
                "argmax": int(np.argmax(logits)),
                "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
            }
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
    finally:
        runner.decode_attention_variants = original
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
