#!/usr/bin/env python3
"""D11 expert-decode chain differential: arm-off (incumbent) vs arm-on.

One process, the real artifact, greedy 128-token decode on the public
``LLM`` surface, twice:

  arm-off  ``gemma4_project_experts_selected`` forced to the pre-D11
           single-resolve incumbent (the committed HEAD behavior before
           this change) -- a faithful shim of the old function body.
  arm-on   the committed preference chain (tiles sibling -> pack8 ->
           incumbent).

Counters make the run non-vacuous: every selected-variant resolution is
attributed to the candidate that resolved it, so an identical token
trajectory with zero arm-on pack8/tiles hits would fail the harness rather
than pass as coincidence.

D9's dense rows==1 t16 rewrite is HEAD behavior and runs in BOTH arms.

Writes /tmp/d11_chain_differential.json.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hipengine
from hipengine.llm import SamplingParams

ARTIFACT = (
    "/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/"
    "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)
PROMPT = "What is the capital of France?"
STEPS = 128
OUT = Path("/tmp/d11_chain_differential.json")

COUNTED_VARIANTS = {
    "selected_gemv_bf16_bf16_out",
    "selected_pack8_gemv_bf16_bf16_out",
}


def _reset_caches() -> None:
    from hipengine.kernels import registry
    from hipengine.runtime.gguf_linear import clear_gguf_linear_dispatch_cache

    registry._RESOLVE_CACHE.clear()
    clear_gguf_linear_dispatch_cache()


def _arm_off_shim(counters: dict):
    """The pre-D11 body: one key, the registered incumbent, raw primary."""

    from hipengine.kernels.registry import (
        KernelKey,
        MissingKernelError,
        resolve,
    )
    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as ge

    def shim(weight, x_ptr, selected_ptr, out_ptr, x_rows, rows,
             num_experts, in_features, out_features, *, stream=0):
        if isinstance(weight, int):
            return False
        key = KernelKey(
            weight.backend, "linear", weight.spec.quant_key, ge._SELECTED_VARIANT
        )
        _ensure_linear_kernel_registered(key)
        try:
            fn = resolve(
                backend=key.backend, layer=key.layer,
                quant=key.quant, variant=key.variant,
            )
        except MissingKernelError:
            return False
        label = f"off:{weight.spec.quant_key}:incumbent"
        counters[label] = counters.get(label, 0) + 1
        fn(
            x_ptr, selected_ptr, weight.allocation().buffer.ptr, out_ptr,
            x_rows, rows, num_experts, in_features, out_features,
            stream=stream,
        )
        return True

    return shim


_ACTIVE_COUNTERS: dict = {}


def _run_arm() -> tuple[list[int], list[int], dict]:
    """One arm: load, tokenize, greedy decode, close. Returns ids+tokens+counts."""

    counters = _ACTIVE_COUNTERS
    llm = hipengine.LLM(model=ARTIFACT)
    try:
        generator = llm._get_text_generator()
        inner = getattr(generator, "_inner", generator)
        prompt_ids = inner.tokenize_chat(PROMPT, enable_thinking=True)
        outputs = llm.generate_detailed(prompt_ids, SamplingParams(max_tokens=STEPS))
        tokens = list(outputs[0].generated_token_ids)
    finally:
        llm.close()
        gc.collect()
    return tokens, prompt_ids, dict(counters)


def main() -> int:
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as ge

    original = ge.gemma4_project_experts_selected
    report: dict = {
        "kind": "gemma4-d11-expert-chain-differential",
        "artifact": ARTIFACT,
        "prompt": PROMPT,
        "steps": STEPS,
        "arms": {},
    }

    # Arm-off: pre-D11 incumbent-only chain.
    _reset_caches()
    _ACTIVE_COUNTERS.clear()
    ge.gemma4_project_experts_selected = _arm_off_shim(_ACTIVE_COUNTERS)
    try:
        off_tokens, prompt_ids, off_counters = _run_arm()
    finally:
        ge.gemma4_project_experts_selected = original
    report["arms"]["off"] = {"counters": off_counters, "tokens": off_tokens}

    # Arm-on: committed preference chain behind a counting route. The count
    # wraps ``_selected_route`` rather than registry.resolve: the route memo
    # answers warm calls without touching the registry, and this harness
    # counts *routing decisions* so non-vacuity holds either way.
    _reset_caches()
    on_counters = _ACTIVE_COUNTERS
    on_counters.clear()
    real_route = ge._selected_route

    def counting_route(weight, out_features):
        fn, allocation = real_route(weight, out_features)
        if not isinstance(weight, int) and fn is not None:
            # Tag by candidate identity, not by function name: the pack8
            # leaf and the raw incumbent are both decorated ``wrapper``.
            from hipengine.kernels.registry import resolve as reg_resolve

            pack8_fn = reg_resolve(
                backend=weight.backend, layer="linear",
                quant=weight.spec.quant_key,
                variant=ge._SELECTED_PACK8_VARIANT, missing="missing",
            )
            if allocation == "tiles":
                tag = "tiles"
            elif fn is pack8_fn:
                tag = "pack8"
            else:
                tag = "incumbent"
            key = f"on:{weight.spec.quant_key}:{tag}"
            on_counters[key] = on_counters.get(key, 0) + 1
        return fn, allocation

    ge._selected_route = counting_route
    try:
        on_tokens, prompt_ids_on, _ = _run_arm()
    finally:
        ge._selected_route = real_route
    assert prompt_ids_on == prompt_ids, "tokenizer must be arm-independent"
    report["arms"]["on"] = {"counters": on_counters, "tokens": on_tokens}

    on_total = sum(on_counters.values())
    # off keys are "off:{quant}:{tag}"; on keys are "on:{quant}:{tag}".
    off_names = {k.split(":", 2)[2] for k in off_counters}
    on_tiles = sum(v for k, v in on_counters.items() if k.endswith(":tiles"))
    # Replacement routes: an on-arm kernel identity the off arm never ran
    # (the pack8 selected leaf lands as a decorated "wrapper"; matching on
    # identity rather than name keeps that robust).
    on_replacements = sum(
        v for k, v in on_counters.items()
        if k.split(":", 2)[2] not in off_names
    )
    on_pack8 = on_replacements - on_tiles
    on_incumbent = on_total - on_replacements
    identical = off_tokens == on_tokens
    report["summary"] = {
        "identical": identical,
        "mismatch_count": sum(
            1 for a, b in zip(off_tokens, on_tokens) if a != b
        )
        + abs(len(off_tokens) - len(on_tokens)),
        "arm_off_total_selected_calls": sum(off_counters.values()),
        "arm_on_total_selected_calls": on_total,
        "arm_on_pack8_hits": on_pack8,
        "arm_on_q5k_tiles_hits": on_tiles,
        "arm_on_incumbent_hits": on_incumbent,
        "non_vacuous": bool(on_pack8 >= 35 and on_tiles >= 35),
    }
    OUT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=1, sort_keys=True))
    print(f"wrote {OUT}")
    return 0 if identical and report["summary"]["non_vacuous"] else 1


if __name__ == "__main__":
    sys.exit(main())