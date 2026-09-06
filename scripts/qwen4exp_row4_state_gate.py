#!/usr/bin/env python3
"""Verify promoted row4 against its opt-out on full logits and recurrent state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.memory import memory_stats, copy_device_to_host, host_array_ptr
from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.qwen4_exp_profiles import (
    register_qwen4_exp_gfx1151_profiles, QWEN4_EXP_MODEL,
    QWEN4_EXP_BACKEND, QWEN4_EXP_QUANTS,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.models import resolve_model
from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture, _git_metadata, _host_metadata
from scripts.qwen4exp_layer2_profile_gate import _state_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--route-package", choices=("q5k-row4", "qsa-h256-wave", "qsa-h256-page256", "q4-bundle", "prefill-bundle", "q51-pair", "gdn-register", "q4-pair", "q8-wave-scale", "gr-wave-scale", "q8-mmq-prepack"), default="q5k-row4")
    p.add_argument("--case-id", action="append")
    p.add_argument("--all-cases", action="store_true")
    p.add_argument("--decode-steps", type=int, default=1)
    p.add_argument("--full-kv", action="store_true")
    args = p.parse_args()
    if not 1 <= args.decode_steps <= 128:
        p.error("--decode-steps must be in 1..128")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION)
    fixture, digest = load_fixture(DEFAULT_FIXTURE)
    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root, weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend="hip_gfx1151", max_sequence_length=4352, prefill_chunk_size=512))
    flag = ("HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL"
            if args.route_package == "q5k-row4"
            else "HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL")
    if args.route_package in {"q4-bundle", "prefill-bundle"}:
        flag = "HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL"
    if args.route_package == "q51-pair":
        flag = "HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL"
    if args.route_package == "gdn-register":
        flag = "HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL"
    if args.route_package == "q4-pair":
        flag = "HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL"
    if args.route_package == "q8-wave-scale":
        flag = "HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE"
    if args.route_package == "gr-wave-scale":
        flag = "HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE"
    if args.route_package == "q8-mmq-prepack":
        flag = "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK"
    from hipengine.kernels.registry import KernelKey, register, resolve
    key = (KernelKey("hip_gfx1151", "linear", "gguf_q5_k",
                     "selected_grouped_row4_gemv_bf16_bf16_out")
           if args.route_package == "q5k-row4" else
           KernelKey("hip_gfx1151", "qsa_sparse_attention", "bf16_kv",
                     "strict_h256_page256_wave_rows_spans"
                     if args.route_package == "qsa-h256-page256" else "strict_h256_wave_rows_spans"))
    if args.route_package in {"q4-bundle", "prefill-bundle"}:
        key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q4_k",
                        "selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out")
    if args.route_package == "q51-pair":
        key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1",
                        "selected_grouped_prefill_pair2_bf16_bf16_out")
    if args.route_package == "gdn-register":
        key = KernelKey("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                        "qwen4exp_sigmoid_register_prefill")
    if args.route_package == "q4-pair":
        key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q4_k",
                        "selected_dual_grouped_pair2_bf16_bf16_out")
    if args.route_package == "q8-wave-scale":
        key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                        "coltile8_rowbatch4_wave_scale_f32_f32_out")
    if args.route_package == "gr-wave-scale":
        key = KernelKey("hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0",
                        "coltile2_branch4_rowbatch4_wave_scale_f32_exact")
    if args.route_package == "q8-mmq-prepack":
        key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                        "mmq128_prepacked_q8_1_d4x3_guarded_f32_f32_out")
    original = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
    calls = [0]

    def counted(*a, **kw):
        calls[0] += 1
        return original(*a, **kw)

    register(key, counted, replace=True)
    qsa_calls = [0]
    qsa_original = None
    if args.route_package == "prefill-bundle":
        qsa_key = KernelKey(
            "hip_gfx1151", "qsa_sparse_attention", "bf16_kv",
            "strict_h256_page256_wave_rows_spans")
        qsa_original = resolve(
            backend=qsa_key.backend, layer=qsa_key.layer, quant=qsa_key.quant,
            variant=qsa_key.variant)

        def counted_qsa(*a, **kw):
            qsa_calls[0] += 1
            return qsa_original(*a, **kw)

        register(qsa_key, counted_qsa, replace=True)
    report = {
        "source": _git_metadata(ROOT), "host": _host_metadata(), "command": sys.argv,
        "manifest_sha256": resolved.manifest_sha256,
        "strict_manifest_sha256": resolved.strict_manifest_sha256,
        "fixture_sha256": digest, "cases": [],
        "route_package": args.route_package,
        "decode_steps": args.decode_steps,
        "full_kv": args.full_kv,
        "scope": (
            "full logits and snapshot decode buffers/PLE/attention positions/index counts; "
            + ("full KV payload included" if args.full_kv else "not full KV payload")
        ),
        "timing_scope": "diagnostic per-step wall; host logit copies between steps; first arm not warmed",
    }
    try:
        if args.route_package == "q8-mmq-prepack":
            assert os.environ.get(flag) == "1", "production must bind prepacked MMQ by default"
            generator.runner.configure_mmq_prefill_resources()
            report["sidecar_bytes"] = generator.runner._q8_mmq_weight_sidecars.nbytes
            report["sidecar_count"] = len(generator.runner._q8_mmq_weight_sidecars.mapping)
            os.environ[flag] = "0"
        if args.route_package in {"q5k-row4", "q51-pair", "gdn-register", "q4-pair", "q8-wave-scale", "gr-wave-scale"}:
            assert os.environ[flag] == "1", "production must select the retained route without an override"
        if args.route_package == "prefill-bundle":
            assert os.environ[flag] == "1"
            assert os.environ["HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL"] == "page256"
        if args.case_id and not set(args.case_id) <= {c["id"] for c in fixture["cases"]}:
            raise ValueError("unknown case id")
        for case in fixture["cases"]:
            if args.case_id:
                if case["id"] not in args.case_id:
                    continue
            elif not args.all_cases and case["prompt_tokens"] != 512 and case["id"] != "code-p4096":
                continue
            baseline = None
            summaries = []
            for enabled in ("0", "1", "0"):
                os.environ[flag] = (
                    "page256" if args.route_package == "qsa-h256-page256" and enabled == "1"
                    else enabled)
                if args.route_package == "prefill-bundle":
                    os.environ["HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL"] = (
                        "page256" if enabled == "1" else "0")
                start_calls = calls[0]
                start_qsa_calls = qsa_calls[0]
                first = generator.runner.prefill(case["prompt_token_ids"])
                logits = first.logits.copy()
                token = int(first.token_id)
                step_logits = []
                step_seconds = []
                for _ in range(args.decode_steps):
                    start = time.perf_counter()
                    next_row = generator.runner.step(token)
                    generator.runner.runtime.device_synchronize()
                    step_seconds.append(time.perf_counter() - start)
                    token = int(next_row.token_id)
                    step_logits.append(next_row.logits.copy())
                next_logits = np.stack(step_logits)
                state = _state_summary(generator.runner)
                if args.full_kv:
                    kv_digest = hashlib.sha256()
                    for attention in generator.runner.attention_states:
                        for buffer in (attention.key_cache, attention.value_cache):
                            raw = np.empty(buffer.nbytes, dtype=np.uint8)
                            copy_device_to_host(
                                host_array_ptr(raw), buffer, runtime=generator.runner.runtime)
                            kv_digest.update(raw)
                    state["full_kv_sha256"] = kv_digest.hexdigest()
                invoked = calls[0] - start_calls
                expected = enabled == "1" and (
                    args.route_package in {"q5k-row4","q4-bundle","prefill-bundle","q51-pair","gdn-register","q4-pair","q8-wave-scale","gr-wave-scale","q8-mmq-prepack"} or case["prompt_tokens"] > 2051)
                assert (invoked > 0) == expected, f"route not engaged correctly: {case['id']}"
                invoked_qsa = qsa_calls[0] - start_qsa_calls
                if args.route_package == "prefill-bundle":
                    assert (invoked_qsa > 0) == (enabled == "1" and case["prompt_tokens"] > 2051)
                actual = (logits, next_logits, state)
                if baseline is None:
                    baseline = actual
                else:
                    np.testing.assert_array_equal(actual[0], baseline[0])
                    np.testing.assert_array_equal(actual[1], baseline[1])
                    assert state == baseline[2], case["id"]
                assert state["finite"]
                summaries.append({
                    "enabled": enabled, "state_sha256": state["state_sha256"],
                    "layout_sha256": state["layout_sha256"],
                    "prefill_logits_sha256": hashlib.sha256(logits).hexdigest(),
                    "step_logits_sha256": hashlib.sha256(next_logits).hexdigest(),
                    "candidate_calls": invoked,
                    "qsa_candidate_calls": invoked_qsa,
                    "step_seconds": step_seconds,
                    "full_kv_sha256": state.get("full_kv_sha256"),
                })
            report["cases"].append({"id": case["id"], "exact": True, "captures": summaries})
            print(case["id"], "full logits/state exact", flush=True)
    finally:
        os.environ[flag] = "1"
        register(key, original, replace=True)
        if qsa_original is not None:
            register(qsa_key, qsa_original, replace=True)
            os.environ["HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL"] = "page256"
        generator.close()
        report["memory_after_close"] = memory_stats()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
