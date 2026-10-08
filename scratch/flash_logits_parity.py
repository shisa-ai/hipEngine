#!/usr/bin/env python3
"""Compare the three long-global decode attention routes on real logits.

One model residency, one fixed teacher token and KV prefix per context, three
arms: the native ``global_class`` kernel, the staged singleton, and the
key-sliced flash kernel the production plan now selects. The arm is chosen by
replacing the registered callable under the flash variant's registry key, which
is the mechanism the layer resolves through, so this exercises the production
call site rather than a direct kernel call.

The question this answers is whether the flash route's reassociation changes the
sampled token because it is wrong or because the choice was a near-tie. So each
arm reports not only its argmax but the top-1/top-2 margin and, for the tokens
the arms disagree on, the gap between them in the reference ordering. A flip
inside that gap is a tie broken differently; a flip outside it is a defect.

Each arm also confirms the path that actually ran: the native arm through the
launcher's own decode-selection counter, the flash arm through it, and the
staged arm through the plan its launcher returns.
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


def logits_bits(logits) -> np.ndarray:
    values = np.asarray(logits)
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise RuntimeError("non-finite or non-FP32 decode logits")
    return values.reshape(-1).view(np.uint32).copy()


def arm_report(reference: np.ndarray, logits: np.ndarray, route: str) -> dict:
    delta = np.abs(logits.astype(np.float64) - reference.astype(np.float64))
    order = np.argsort(reference)[::-1]
    argmax = int(np.argmax(logits))
    top = np.argsort(logits)[::-1]
    reference_top = int(order[0])
    return {
        "route": route,
        "argmax": argmax,
        "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
        "bitwise_equal_to_reference": bool(np.array_equal(logits, reference)),
        "max_abs_delta": float(delta.max()),
        "mean_abs_delta": float(delta.mean()),
        "delta_at_reference_top1": float(delta[reference_top]),
        "delta_at_this_choice": float(delta[argmax]),
        "top1_minus_top2": float(logits[top[0]] - logits[top[1]]),
        "reference_top1_minus_this_choice": float(reference[reference_top] - reference[argmax]),
        "rank_of_this_choice_in_reference": int(np.where(order == argmax)[0][0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from scripts.gemma4_campaign_bench import DEFAULT_ARTIFACT, _resolve_generator, exact_prompt_ids
    from hipengine.core.hip import get_hip_runtime
    from hipengine.generation.gemma4_profiles import (
        DECODE_ATTENTION_FLASH,
        DECODE_ATTENTION_LAYER,
        DECODE_ATTENTION_QUANT,
        GEMMA4_GGUF_BACKEND,
        register_gemma4_gguf_profiles,
    )
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_staged as staged_module
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import last_prefill_attention_route
    from hipengine.kernels.registry import KernelKey, register

    register_gemma4_gguf_profiles()
    key = KernelKey(
        GEMMA4_GGUF_BACKEND, DECODE_ATTENTION_LAYER, DECODE_ATTENTION_QUANT,
        DECODE_ATTENTION_FLASH,
    )
    artifact = args.artifact or DEFAULT_ARTIFACT
    llm, runner, loading = _resolve_generator(artifact, max(args.contexts) + 16)
    runtime = get_hip_runtime()
    generator = llm._get_text_generator()
    report = {
        "kind": "gemma4_flash_logits_parity",
        "artifact": str(artifact),
        "loading": {k: v for k, v in loading.items() if k != "resolution"},
        "profile_env": os.environ.get("HIPENGINE_EXECUTION_PROFILE"),
        "cases": [],
    }
    staged_plans: list = []

    def staged_arm(*a, **kw):
        plan = staged_module.gemma4_attention_staged_bf16(*a, **kw)
        staged_plans.append(plan)
        return plan

    arms = (
        ("native", attention.gemma4_attention_prefill_bf16),
        ("staged", staged_arm),
        ("flash", None),
    )
    try:
        for context in args.contexts:
            ids = exact_prompt_ids(generator.tokenize, context)
            runner.reset()
            prefill = runner.forward(ids)
            runtime.device_synchronize()
            teacher = int(runner.next_token(prefill))
            reference_logits = None
            rows = []
            for arm, callable_ in arms:
                if callable_ is None:
                    register(key, attention.gemma4_attention_decode_flash_bf16, replace=True)
                else:
                    register(key, callable_, replace=True)
                staged_plans.clear()
                runner.rewind(context)
                runtime.device_synchronize()
                logits = np.asarray(runner.forward([teacher]), dtype=np.float32).reshape(-1)
                runtime.device_synchronize()
                selection = attention.decode_selection()
                route = last_prefill_attention_route()
                if arm == "native" and selection != 4:
                    raise RuntimeError(f"native arm ran decode selection {selection}, not 4")
                if arm == "flash" and selection != 3:
                    raise RuntimeError(f"flash arm ran decode selection {selection}, not 3")
                if arm == "staged" and not staged_plans:
                    raise RuntimeError("staged arm did not reach the staged launcher")
                if reference_logits is None:
                    reference_logits = logits
                row = {
                    "arm": arm,
                    "reported_route": route,
                    "decode_selection": selection,
                    "staged_plan_rows": staged_plans[-1].rows if staged_plans else None,
                    **arm_report(reference_logits, logits, arm),
                }
                rows.append(row)
                print(json.dumps({"context": context, **row}), flush=True)
            # Every arm re-registers, so the registered default is restored at
            # the end of the loop by the flash entry.
            flash_row = next(r for r in rows if r["arm"] == "flash")
            native_row = next(r for r in rows if r["arm"] == "native")
            report["cases"].append({
                "context_tokens": context,
                "keys": context + 1,
                "teacher_token": teacher,
                "prompt_ids_sha256": hashlib.sha256(
                    np.array(ids, dtype=np.int32).tobytes()
                ).hexdigest(),
                "arms": rows,
                "flash_matches_native_token": flash_row["argmax"] == native_row["argmax"],
                # The decisive comparison: a token the flash arm chose that the
                # native arm ranked below its top-1 is a tie broken differently
                # only if the gap is within the numerical difference at those
                # two logits -- not merely within the global maximum.
                "flash_flip_within_numerical_noise": (
                    flash_row["reference_top1_minus_this_choice"]
                    <= max(flash_row["delta_at_reference_top1"],
                           flash_row["delta_at_this_choice"])
                ),
                "staged_bitwise_equal_native": next(
                    r["bitwise_equal_to_reference"]
                    for r in rows if r["arm"] == "staged"),
            })
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        register(key, attention.gemma4_attention_decode_flash_bf16, replace=True)
        llm.close()

    print(json.dumps({
        "contexts": args.contexts,
        "summary": [
            {
                "context": case["context_tokens"],
                "native_token": next(r["argmax"] for r in case["arms"] if r["arm"] == "native"),
                "flash_token": next(r["argmax"] for r in case["arms"] if r["arm"] == "flash"),
                "staged_token": next(r["argmax"] for r in case["arms"] if r["arm"] == "staged"),
                "flash_max_abs_delta": next(
                    r["max_abs_delta"] for r in case["arms"] if r["arm"] == "flash"),
                "flash_delta_at_native_top1": next(
                    r["delta_at_reference_top1"]
                    for r in case["arms"] if r["arm"] == "flash"),
                "native_top1_minus_flash_choice": next(
                    r["reference_top1_minus_this_choice"]
                    for r in case["arms"] if r["arm"] == "flash"),
                "flash_top1_minus_top2": next(
                    r["top1_minus_top2"] for r in case["arms"] if r["arm"] == "flash"),
                "staged_bitwise_equal_native": next(
                    r["bitwise_equal_to_reference"] for r in case["arms"] if r["arm"] == "staged"),
            }
            for case in report["cases"]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
