#!/usr/bin/env python3
"""Qualify checked/full-row dense Q8 against complete model state and logits.

This is a diagnostic registry substitution. Parent arms publish the existing
DPP leaf; candidate arms replace only that key for the six admitted Q8 dense
geometries and rows 512..4096. The candidate is never made a profile default.
Each case runs parent/candidate/candidate/candidate/parent, compares full
prefill and decode logits, recurrent/layout metadata and full KV bytes, and
requires zero allocations after close.
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
sys.path.insert(0, str(ROOT))

from hipengine.core.memory import copy_device_to_host, host_array_ptr, memory_stats
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.qwen4_exp_profiles import (
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    QWEN4_EXP_QUANTS,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.models import resolve_model
from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE,
    _git_metadata,
    _host_metadata,
    load_fixture,
)
from scripts.qwen4exp_layer2_profile_gate import _state_summary
from scripts.qwen4exp_dense_dpp_checked_full_rows_natural_owner import (
    DECLARED_SHAPES,
    gate_q8_admissible,
)

PARENT = "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
CANDIDATE = "coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out"
BACKEND = "hip_gfx1151"
QUANT = "gguf_q8_0"
CHUNK_SIZE = 4096


def admitted_shape(rows: int, k: int, n: int) -> bool:
    rows, k, n = map(int, (rows, k, n))
    return 512 <= rows <= 4096 and rows % 4 == 0 and (k, n) in DECLARED_SHAPES and gate_q8_admissible(rows, k, n)


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8).tobytes()).hexdigest()


def full_kv_digest(runner) -> str:
    digest = hashlib.sha256()
    for attention in runner.attention_states:
        for buffer in (attention.key_cache, attention.value_cache):
            raw = np.empty(int(buffer.nbytes), dtype=np.uint8)
            copy_device_to_host(host_array_ptr(raw), buffer, runtime=runner.runtime)
            digest.update(raw)
    return digest.hexdigest()


def capture(runner, decode_steps: int, full_kv: bool) -> dict:
    first = runner.prefill(runner._qualification_prompt_tokens)
    prefill_logits = first.logits.copy()
    prefill_state = _state_summary(runner)
    token = int(first.token_id)
    step_logits = []
    for _ in range(decode_steps):
        row = runner.step(token)
        runner.runtime.device_synchronize()
        token = int(row.token_id)
        step_logits.append(row.logits.copy())
    decode_logits = np.stack(step_logits)
    state = _state_summary(runner)
    if full_kv:
        state["full_kv_sha256"] = full_kv_digest(runner)
    return {
        "prefill_logits": prefill_logits,
        "decode_logits": decode_logits,
        "prefill_state": prefill_state,
        "state": state,
        "prefill_logits_sha256": array_digest(prefill_logits),
        "decode_logits_sha256": array_digest(decode_logits),
        "prefill_state_sha256": prefill_state["state_sha256"],
        "state_sha256": state["state_sha256"],
        "full_kv_sha256": state.get("full_kv_sha256"),
        "token_id": token,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--full-kv", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.decode_steps <= 128:
        parser.error("decode-steps must be in 1..128")

    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    os.environ["HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION"] = "1"
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    profile = __import__("hipengine.execution_profiles", fromlist=["resolve_runtime_profile"]).resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION,
    )
    fixture, fixture_sha = load_fixture(DEFAULT_FIXTURE)
    cases = fixture["cases"]
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["id"] in wanted]
        if {case["id"] for case in cases} != wanted:
            parser.error("unknown case-id")
    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    generator = profile.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root,
        weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend=BACKEND,
        max_sequence_length=4352,
        prefill_chunk_size=CHUNK_SIZE,
    ))
    runner = generator.runner
    key = KernelKey(BACKEND, "linear", QUANT, PARENT)
    parent = resolve(backend=BACKEND, layer="linear", quant=QUANT, variant=PARENT)
    candidate = resolve(backend=BACKEND, layer="linear", quant=QUANT, variant=CANDIDATE)
    report = {
        "schema": 1,
        "kind": "qwen4exp_dense_dpp_checked_full_rows_state_gate",
        "status": "running",
        "performance_claim": False,
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model": "Qwen3.8-Flash-Next UD-Q4_K_XL",
        "model_root": str(args.model_root),
        "fixture_sha256": fixture_sha,
        "manifest_sha256": profile.manifest_sha256,
        "strict_manifest_sha256": profile.strict_manifest_sha256,
        "parent_variant": PARENT,
        "candidate_variant": CANDIDATE,
        "scope": {
            "backend": BACKEND,
            "quant": QUANT,
            "rows": [512, 1024, 4096],
            "shapes": sorted([list(shape) for shape in DECLARED_SHAPES]),
            "prefill_chunk_size": CHUNK_SIZE,
            "decode_steps": args.decode_steps,
            "full_kv": bool(args.full_kv),
        },
        "cases": [],
    }
    try:
        # The hook substitutes only admitted raw Q8 dense launches. All other
        # launches and all decode launches retain their registered parent.
        state = {"enabled": False, "candidate_calls": 0, "parent_calls": 0}
        def hooked(*call_args, **kwargs):
            rows, k, n = map(int, call_args[3:6])
            if state["enabled"] and admitted_shape(rows, k, n):
                state["candidate_calls"] += 1
                return candidate(*call_args, **kwargs)
            state["parent_calls"] += 1
            return parent(*call_args, **kwargs)
        register(key, hooked, replace=True)
        for case in cases:
            runner._qualification_prompt_tokens = list(case["prompt_token_ids"])
            case_result = {"id": case["id"], "prompt_tokens": case["prompt_tokens"], "arms": []}
            baseline = None
            start_candidate = state["candidate_calls"]
            for arm_index, enabled in enumerate((False, True, True, True, False)):
                state["enabled"] = enabled
                before_candidate = state["candidate_calls"]
                arm = capture(runner, args.decode_steps, args.full_kv)
                if baseline is None:
                    baseline = arm
                else:
                    np.testing.assert_array_equal(arm["prefill_logits"], baseline["prefill_logits"])
                    np.testing.assert_array_equal(arm["decode_logits"], baseline["decode_logits"])
                    assert arm["prefill_state"] == baseline["prefill_state"], case["id"]
                    assert arm["state"] == baseline["state"], case["id"]
                    assert arm["full_kv_sha256"] == baseline["full_kv_sha256"], case["id"]
                case_result["arms"].append({
                    "arm": arm_index,
                    "enabled": enabled,
                    "candidate_calls": state["candidate_calls"] - before_candidate,
                    "prefill_logits_sha256": arm["prefill_logits_sha256"],
                    "decode_logits_sha256": arm["decode_logits_sha256"],
                    "prefill_state_sha256": arm["prefill_state_sha256"],
                    "state_sha256": arm["state_sha256"],
                    "full_kv_sha256": arm["full_kv_sha256"],
                    "token_id": arm["token_id"],
                })
            if sum(arm["candidate_calls"] for arm in case_result["arms"] if arm["enabled"]) <= 0:
                raise AssertionError((case["id"], "candidate did not engage"))
            case_result["candidate_calls_enabled"] = sum(
                arm["candidate_calls"] for arm in case_result["arms"] if arm["enabled"]
            )
            case_result["candidate_calls_total"] = state["candidate_calls"] - start_candidate
            case_result["exact"] = True
            report["cases"].append(case_result)
            print(case["id"], "full logits/state/KV exact", case_result["candidate_calls_enabled"], flush=True)
        report["candidate_calls"] = state["candidate_calls"]
        report["parent_calls"] = state["parent_calls"]
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        register(key, parent, replace=True)
        generator.close()
        report["memory_after_close"] = memory_stats()
        report["lifecycle_passed"] = (
            report["memory_after_close"]["active_allocations"] == 0
            and report["memory_after_close"]["current_allocated_bytes"] == 0
        )
        if not report["lifecycle_passed"]:
            report["status"] = "failed"
            report["error"] = "allocation leak after generator close"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
