#!/usr/bin/env python3
"""Verify promoted row4 against its opt-out on full logits and recurrent state."""
import argparse
import hashlib
import json
import os
from contextlib import ExitStack, contextmanager
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
from scripts.qwen4exp_halo_box_campaign_ab import (
    q8_down_row4_expected_calls, q51_fold128_expected_calls, q51_fold_pair_expected_calls,
    q8_down_bundle_expected_calls,q51_register_cache_expected_calls,
    q51_row_publish_expected_calls,require_engaged_expectation,
    q8_mmq_vec4_expected_calls,q8_mmq_raw_vector_expected_calls,q8_mapped_down_expected_calls,
    q8_bundle_call_in_scope,apply_chunk_mode,validate_chunk_coverage,q8_down_register_expected_calls,
    gdn_wave_norm_expected_calls,mmq_token64_expected_calls,gr_dpp_expected_calls,
    router_dpp_expected_calls,route_scope,GDN_COLWARPS_ENV,
    Q51_IU8_EXACT_ENV,Q51_PAIR2_PREEMPTING_ENV,Q51_PAIR2_GATE_FIELD,
    Q51_PAIR_FIELD,Q51_FOLD128_FIELD,Q51_FOLD_PAIR_FIELD,
    q51_pair2_control_record,q51_register_cache_control_record,
    q8_down_control_record,Q8_MAPPED_DOWN_MAP_GATE,
    Q8_DOWN_FORKB_FIELD,Q8_DOWN_ROW4_FIELD,Q8_DOWN_BUNDLE_FIELD,
    q8_mapped_down_control_record,
    q51_fold_control_record,q8_down_ladder_control_record,
    MMQ_VECTOR_ROUTE_PACKAGES,mmq_token64_control_record,mmq_vector_control_record)


def apply_state_gate_mode(runner, package, enabled, flag, *, environment=os.environ):
    if package in {"q5k-row4", "qsa-h256-wave", "qsa-h256-page256",
                   "q4-bundle", "prefill-bundle", "q51-pair", "q4-pair"}:
        return  # route_scope binds the runner-local choice.
    if package == "mmq-token64":
        # Request-local on the runner; the prefill call site binds the arm.
        return
    if package in {"q8-mmq-raw-vector", "q8-mmq-vec4"}:
        # Request-local on the runner; ``route_scope`` admits the MMQ parent
        # prerequisite and binds the arm around the measured prefill.
        return
    if package == "q5k-iu8-exact":
        # The q5k opt-out arm de-groups layer 2 entirely (its only grouped
        # entry is the q5k flag), which also changes that layer's Q8_0 down
        # route selection. Pin the #19 q8-wmma-down T1 route off in BOTH arms
        # so this gate keeps isolating the q5k gate/up arithmetic alone.
        environment["HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN"] = "0"
    if package == "norm-f16-publish":
        return  # route_scope binds the runner-local publication policy.
    if package == "dense-wide-padded":
        # route_scope binds the runner's parent for both registry arms.
        return
    if package in {"gdn-narrow", "gdn-wave-norm"}:
        # Request-local on the runner; ``route_scope`` binds the arm around the
        # measured prefill and restores the profile's decision on exit.
        return
    if package == "q51-register-cache":
        # Request-local on the runner; ``route_scope`` binds the arm around the
        # measured prefill, closes the owners that preempt the pair2 chain and
        # holds the downstream row-publish sibling off.
        return
    if package == "q51-row-publish":
        # Request-local on the runner, like the GDN serial owner; the arm also
        # closes the exact iu8 Q5_1 down that would preempt the pair2 chain.
        return
    if package in {"q51-fold128", "q51-fold-pair", "q8-down-row4",
                   "q8-down-bundle"}:
        # Request-local on the runner; ``route_scope`` binds the rung around
        # the measured prefill and restores the plan's published value.
        return
    if package == "q8-down-register":
        # Request-local on the runner; ``route_scope`` binds the arm around the
        # measured prefill and closes the owners that preempt both Q8_0
        # expert-down chains.
        return
    if package == "q8-mapped-down":
        # Request-local on the runner; ``route_scope`` binds the arm around the
        # measured prefill and closes the owners that preempt the token-major
        # mapped path.
        return
    if package == "router-dpp":
        environment[flag] = (
            "f32_hidden_token_tile4_dpp_exact" if enabled == "1"
            else "f32_hidden_token_tile4_dense_exact"
        )
        return
    if package in {"qsa-head-pair", "qsa-head-quad"}:
        # Request-local on the runner; ``route_scope`` binds the arm around the
        # measured prefill and restores the profile's decision on exit.
        return
    if package == "qsa-ordered-v2":
        # Request-local on the runner; ``route_scope`` binds the leaf and the
        # case keeps the binding open through the decode steps that count it.
        return
    if package == "chunk1024":
        apply_chunk_mode(runner,"after" if enabled=="1" else "before",
                         allocated_chunk_size=1024)
    else:
        # Every package that reaches this branch owns a binder-set process key;
        # a request-local package returns above, so an empty name here would be
        # a wiring bug rather than a missing key.
        assert flag, f"{package} has no process key to bind"
        environment[flag] = enabled


@contextmanager
def arm_control(package, enabled, runner, *, environment=os.environ):
    """The live control for one measured arm, released on every exit path.

    The caller keeps the control open for exactly the work it counts: every
    prefill arm closes it as soon as the prefill returns, and the decode-only
    ordered-v2 arm keeps it bound through the steps that count it. The stack
    also closes when the block exits, including on a raise, so a failed prefill
    or decode cannot leave the arm's env pins or runner bindings in place for
    the next arm. ``ExitStack.close`` is idempotent, so the caller's early close
    and this exit close agree.
    """

    with ExitStack() as scope:
        scope.enter_context(route_scope(
            package, "after" if enabled == "1" else "before", runner,
            environment=environment,
        ))
        yield scope


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--route-package", choices=("q5k-row4", "qsa-h256-wave", "qsa-h256-page256", "q4-bundle", "prefill-bundle", "q51-pair", "gdn-register", "q4-pair", "q8-wave-scale", "gr-wave-scale", "q8-mmq-prepack", "q8-down-row4", "q51-fold128", "q8-down-bundle", "q51-fold-pair", "q8-mmq-vec4", "q51-register-cache", "q8-mmq-raw-vector", "q8-mapped-down", "chunk1024", "q8-down-register", "q51-row-publish", "gdn-wave-norm", "mmq-token64", "qsa-head-pair", "qsa-head-quad", "q4-iu8-exact", "q51-iu8-exact", "q5k-iu8-exact", "qsa-ordered-v2", "gr-iu8", "gr-iu8-down", "q8-wmma-down", "q8-dpp-reduction", "gr-dpp-reduction", "router-dpp", "gdn-narrow", "dense-wide-padded"), default="q5k-row4")
    p.add_argument("--prefill-chunk-size", type=int, choices=(512, 1024, 2048, 4096))
    p.add_argument("--case-id", action="append")
    p.add_argument("--all-cases", action="store_true")
    p.add_argument("--decode-steps", type=int, default=1)
    p.add_argument("--full-kv", action="store_true")
    p.add_argument("--include-heldouts", action="store_true",
                   help="Append all category-heldout prompts to the exact state gate")
    args = p.parse_args()
    if args.include_heldouts and args.route_package not in {"gdn-narrow", "dense-wide-padded"}:
        p.error("--include-heldouts is scoped to gdn-narrow/dense-wide-padded")
    if not 1 <= args.decode_steps <= 128:
        p.error("--decode-steps must be in 1..128")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    from hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_q8_1_selected_prefill import (
        register_gguf_q5_k_q8_1_selected_prefill_kernels,
    )
    register_gguf_q5_k_q8_1_selected_prefill_kernels(replace=True)
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
        backend="hip_gfx1151", max_sequence_length=4352,
        prefill_chunk_size=args.prefill_chunk_size or (1024 if args.route_package in {"chunk1024","q8-down-register", "q51-row-publish", "gdn-wave-norm", "mmq-token64", "qsa-head-pair", "qsa-head-quad", "q4-iu8-exact", "q51-iu8-exact", "q5k-iu8-exact", "qsa-ordered-v2", "gr-iu8", "gr-iu8-down", "q8-wmma-down"} else 512)))
    gate_cases = list(fixture["cases"])
    if args.include_heldouts:
        from scripts.gguf_gdn_semantic_gate import _load_suites
        from scripts.gguf_mtp_bench import build_chat_prompt
        heldout_path = ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl"
        heldouts = _load_suites([heldout_path])
        for row in heldouts:
            tokens = list(build_chat_prompt(generator.tokenizer, str(row["prompt"])))
            if args.route_package == "dense-wide-padded":
                from scripts.qwen4exp_dense_wide_padded_gate_support import heldout_tokens
                tokens = heldout_tokens(tokens)
            if len(tokens) + args.decode_steps > 4352:
                raise ValueError("heldout exceeds runner capacity")
            gate_cases.append({"id": row["id"], "category": row["category"],
                               "prompt_tokens": len(tokens), "prompt_token_ids": tokens})
    flag = ""
    if args.route_package in {"q4-bundle", "prefill-bundle"}:
        flag = ""  # Runner-local selection is bound by arm_control.
    if args.route_package == "q51-pair":
        flag = ""  # route_scope binds the runner.
    if args.route_package == "gdn-register":
        flag = ""  # route_scope binds the owning runner.
    if args.route_package in {"gdn-wave-norm", "gdn-narrow"}:
        # Request-local on the runner; no env key names this leaf.
        flag = ""
    if args.route_package in {"qsa-head-pair","qsa-head-quad"}:
        # Request-local on the runner; no env key names this leaf.
        flag=""
    if args.route_package == "q4-pair":
        flag = ""  # route_scope binds the owning runner.
    if args.route_package == "q4-iu8-exact":
        flag = "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT"
    if args.route_package == "q51-iu8-exact":
        flag = "HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT"
    if args.route_package == "q5k-iu8-exact":
        flag = "HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT"
    if args.route_package == "qsa-ordered-v2":
        # Request-local on the runner; no env key names this selection.
        flag = ""
    if args.route_package == "gr-iu8":
        flag = "HIPENGINE_QWEN4_EXP_GR_IU8"
    if args.route_package == "gr-iu8-down":
        flag = "HIPENGINE_QWEN4_EXP_GR_IU8_DOWN"
    if args.route_package == "q8-wmma-down":
        flag = "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN"
    if args.route_package == "q8-wave-scale":
        flag = ""  # route_scope binds the owning runner.
    if args.route_package == "q8-dpp-reduction":
        flag = "HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION"
    if args.route_package == "router-dpp":
        flag = "HIPENGINE_QWEN4_EXP_ROUTER_VARIANT"
    if args.route_package == "gr-dpp-reduction":
        flag = "HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION"
    if args.route_package == "gr-wave-scale":
        flag = ""  # route_scope binds the owning runner.
    if args.route_package == "q8-mmq-prepack":
        flag = "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK"
    if args.route_package == "q8-mmq-vec4":
        # Request-local on the runner; no env key names this selection.
        flag = ""
    if args.route_package == "q8-mmq-raw-vector":
        # Request-local on the runner; no env key names this selection.
        flag = ""
    if args.route_package == "mmq-token64":
        # Request-local on the runner; no env key names this selection. The
        # arm admits the MMQ parent it rides through ``route_scope``.
        flag = ""
    if args.route_package in {"q8-down-row4", "q51-fold128", "q8-down-bundle",
                             "q51-fold-pair"}:
        # Request-local on the runner; no env key names this selection.
        # ``route_scope`` binds the rung around the measured prefill.
        flag = ""
    if args.route_package == "q8-mapped-down":
        # Request-local on the runner; no env key names this selection.
        flag = ""
    if args.route_package=="q8-down-register":
        # Request-local on the runner; no env key names this selection.
        flag=""
    if args.route_package == "q51-register-cache":
        # Request-local on the runner; no env key names this selection.
        flag = ""
    if args.route_package == "q51-row-publish":
        # Request-local on the runner; no env key names this selection.
        flag = ""
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
    if args.route_package == "q4-iu8-exact":
        key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q4_k",
                        "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out")
    if args.route_package == "q51-iu8-exact":
        key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1",
                        "selected_wmma_iu8_risk_prefill_bf16_bf16_out")
    if args.route_package == "q5k-iu8-exact":
        key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_k",
                        "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out")
    if args.route_package == "qsa-ordered-v2":
        key = KernelKey("hip_gfx1151", "qsa_sparse_attention", "bf16_kv",
                        "strict_ordered_three_pass_v2_spans")
    if args.route_package == "q8-wave-scale":
        key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                        "coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out")
    if args.route_package == "q8-dpp-reduction":
        key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                        "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out")
    if args.route_package == "router-dpp":
        key = KernelKey("hip_gfx1151", "router_logits", "f32",
                        "f32_hidden_token_tile4_dpp_exact")
    if args.route_package == "gr-dpp-reduction":
        key = KernelKey("hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0",
                        "coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact")
    if args.route_package == "gr-wave-scale":
        key = KernelKey("hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0",
                        "coltile2_branch4_rowbatch4_wave_scale_f32_exact")
    if args.route_package == "q8-mmq-prepack":
        key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                        "mmq128_prepacked_q8_1_d4x3_guarded_f32_f32_out")
    if args.route_package == "q8-mmq-vec4":
        key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                        "mmq128_prepacked_vec4_q8_1_d4x3_guarded_f32_f32_out")
    if args.route_package == "q8-mmq-raw-vector":
        key = KernelKey("hip_gfx1151","linear","gguf_q8_0",
                        "mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out")
    if args.route_package == "q8-down-row4":
        key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                        "selected_grouped_row4_gemv_bf16_bf16_out")
    if args.route_package == "q8-wmma-down":
        key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                        "selected_grouped_wmma_prefill_bf16_bf16_out")
    if args.route_package == "q51-fold128":
        key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1",
                        "selected_grouped_prefill_pair2_fold128_bf16_bf16_out")
    if args.route_package in {"q8-down-bundle","q8-mapped-down"}:
        key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                        "selected_grouped_row4_bundle_gemv_bf16_bf16_out")
    if args.route_package == "q51-fold-pair":
        key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1",
                        "selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out")
    if args.route_package == "q51-register-cache":
        key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1",
                        "selected_grouped_prefill_pair2_register_cache_bf16_bf16_out")
    if args.route_package == "q51-row-publish":
        key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1",
                        "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out")
    if args.route_package=="q8-down-register":
        key=KernelKey("hip_gfx1151","linear","gguf_q8_0",
                      "selected_grouped_row4_register_gemv_bf16_bf16_out")
    if args.route_package in {"gdn-wave-norm", "gdn-narrow"}:
        key=KernelKey("hip_gfx1151","gdn_recurrence_norm_gate",
                      "f32_state","qwen4exp_sigmoid_wave_norm_prefill")
    if args.route_package=="mmq-token64":
        key=KernelKey("hip_gfx1151","linear","gguf_q8_0",
                      "mmq128_token64_q8_1_d4x3_guarded_f32_f32_out")
    if args.route_package=="qsa-head-pair":
        key=KernelKey("hip_gfx1151","qsa_sparse_attention","bf16_kv",
                      "strict_h256_head_pair_rows_spans")
    if args.route_package=="qsa-head-quad":
        key=KernelKey("hip_gfx1151","qsa_sparse_attention","bf16_kv",
                      "strict_h256_head_quad_rows_spans")
    if args.route_package == "dense-wide-padded":
        from scripts.qwen4exp_dense_wide_padded_gate_support import PARENT
        key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0", PARENT)
    original = (None if args.route_package=="chunk1024" else
                resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant))
    if args.route_package == "q8-mapped-down":
        # The mapped path admits the bundle key and the published register
        # decision may rewrite it, so both owners are instrumented.
        mapped_register_key = KernelKey(
            key.backend, key.layer, key.quant,
            "selected_grouped_row4_register_gemv_bf16_bf16_out")
        mapped_register_original = resolve(
            backend=mapped_register_key.backend,
            layer=mapped_register_key.layer,
            quant=mapped_register_key.quant,
            variant=mapped_register_key.variant)
    calls = [0]
    mapped_calls=[0]
    mapped_owner_calls = {"bundle": 0, "register": 0}
    mapped_register_key = None
    mapped_register_original = None
    narrow_enabled = [False]
    narrow_candidate = None
    gdn_narrow_key = None
    gdn_narrow_original = None
    if args.route_package == "gdn-narrow":
        from scripts.qwen4exp_gdn_narrow_gate_support import expected_calls, validate_call
        gdn_narrow_key = KernelKey(key.backend, key.layer, key.quant,
                                   "qwen4exp_sigmoid_wave_norm_narrow_prefill")
        gdn_narrow_original = resolve(backend=gdn_narrow_key.backend,
                                      layer=gdn_narrow_key.layer,
                                      quant=gdn_narrow_key.quant,
                                      variant=gdn_narrow_key.variant)

        def counted_narrow(*a, **kw):
            validate_call(a, max_rows=generator.runner.prefill_chunk_size)
            calls[0] += 1
            return gdn_narrow_original(*a, **kw)

        # The arm selects the leaf through the runner's published variant, so
        # engagement is counted on the kernel that actually ran.
        register(gdn_narrow_key, counted_narrow, replace=True)
    if args.route_package == "dense-wide-padded":
        from scripts.qwen4exp_dense_wide_padded_gate_support import (
            CANDIDATE, call_in_scope, expected_calls as tail_expected_calls,
        )
        narrow_candidate = resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                                   variant=CANDIDATE)

    natural_router = None
    if args.route_package == "router-dpp":
        from scripts.qwen4exp_router_dpp_gate_support import NaturalRouterCheck
        natural_router = NaturalRouterCheck()

    def counted(*a, **kw):
        if args.route_package == "dense-wide-padded":
            if narrow_enabled[0] and call_in_scope(a):
                calls[0] += 1
                return narrow_candidate(*a, **kw)
            return original(*a, **kw)
        if args.route_package == "gdn-narrow":
            # The wide arm is the parent control; validate its admission shape
            # and let the narrow key's wrapper own the engagement count.
            validate_call(a, max_rows=generator.runner.prefill_chunk_size)
            return original(*a, **kw)
        if narrow_candidate is not None:
            validate_call(a, max_rows=generator.runner.prefill_chunk_size)
            if narrow_enabled[0]:
                calls[0] += 1
                return narrow_candidate(*a, **kw)
            return original(*a, **kw)
        if args.route_package=="q8-down-register" and a[2]:
            mapped_calls[0]+=1
        if q8_bundle_call_in_scope(args.route_package,a):
            calls[0] += 1
        result = original(*a, **kw)
        if natural_router is not None:
            natural_router.check(*a, **kw)
        return result

    observed_chunks = []
    original_chunk = None
    if args.route_package=="chunk1024":
        original_chunk = generator.runner._prefill_chunk
        def counted_chunk(token_ids, **kwargs):
            observed_chunks.append(len(token_ids))
            calls[0] += 1
            return original_chunk(token_ids,**kwargs)
        generator.runner._prefill_chunk = counted_chunk
    else:
        if args.route_package == "q8-mapped-down":
            # The mapped path admits the bundle key and the published register
            # decision may rewrite it, so both owners are instrumented and only
            # the mapped path's map-pointer calls count.
            def counted_mapped_bundle(*a, **kw):
                if a[2]:
                    calls[0] += 1
                    mapped_owner_calls["bundle"] += 1
                return original(*a, **kw)

            def counted_mapped_register(*a, **kw):
                if a[2]:
                    calls[0] += 1
                    mapped_owner_calls["register"] += 1
                return mapped_register_original(*a, **kw)

            register(key, counted_mapped_bundle, replace=True)
            register(mapped_register_key, counted_mapped_register, replace=True)
        else:
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
        "status": "running",
        "allocated_chunk_size": generator.runner.prefill_chunk_size,
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
    if narrow_candidate is not None:
        report["diagnostic_registry_substitution"] = {
            "parent": key.variant, "candidate": (
                CANDIDATE if args.route_package == "dense-wide-padded"
                else "qwen4exp_sigmoid_wave_norm_narrow_prefill"),
            "manifest_scope": "named profile; diagnostic arms override only the named leaf",
            "runtime_default_changed": False,
        }
    if gdn_narrow_key is not None:
        report["diagnostic_runner_variant"] = {
            "parent": key.variant, "candidate": gdn_narrow_key.variant,
            "manifest_scope": "named profile; diagnostic arms override only the named leaf",
            "runtime_default_changed": False,
        }
    if args.route_package=="chunk1024":
        report["chunk_protocol"] = {
            "allocated_chunk_size":1024,"active_sequence":[512,1024,512],
            "kernel_flags_changed":False,
            "memory_scope":"Shared larger allocation;not an allocation-size A/B",
        }
    try:
        if args.route_package == "q8-mmq-prepack":
            assert os.environ.get(flag) == "1", "production must bind prepacked MMQ by default"
            generator.runner.configure_mmq_prefill_resources()
            report["sidecar_bytes"] = generator.runner._q8_mmq_weight_sidecars.nbytes
            report["sidecar_count"] = len(generator.runner._q8_mmq_weight_sidecars.mapping)
            os.environ[flag] = "0"
        if args.route_package in {"qsa-h256-wave", "qsa-h256-page256"}:
            assert generator.runner.qsa_h256_prefill_mode == "page256"
            report["selection_control"] = {
                "runner_field": "qsa_h256_prefill_mode", "before": "parent",
                "after": "wave" if args.route_package == "qsa-h256-wave" else "page256",
                "scope": "per measured run; restored on exit",
            }
        if args.route_package == "q5k-row4":
            assert generator.runner.q5k_row4_prefill is True
            report["selection_control"] = {
                "runner_field": "q5k_row4_prefill", "before": False, "after": True,
                "scope": "per measured run; restored on exit",
            }
        if args.route_package == "q51-pair":
            assert generator.runner.q51_pair_prefill is True
            report["selection_control"] = {
                "runner_field": Q51_PAIR_FIELD, "before": False, "after": True,
                "published_value": generator.runner.q51_pair_prefill,
                "scope": "per measured run; restored on exit",
            }
        if args.route_package == "gdn-register":
            assert generator.runner.gdn_register_prefill is True
            report["selection_control"] = {
                "runner_field": "gdn_register_prefill", "before": False, "after": True,
                "published_value": generator.runner.gdn_register_prefill,
                "scope": "per measured run; restored on exit",
                "isolation": "tiled and peer prefill off; wave-norm variant unset",
            }
        if args.route_package == "q4-pair":
            from scripts.qwen4exp_halo_box_campaign_ab import q4_pair_control_record
            assert generator.runner.q4_pair_prefill is True
            report["selection_control"] = q4_pair_control_record(generator.runner)
        if args.route_package == "q8-wave-scale":
            from scripts.qwen4exp_halo_box_campaign_ab import q8_wave_control_record
            assert generator.runner.q8_wave_scale_prefill is True
            report["selection_control"] = q8_wave_control_record(generator.runner)
        if args.route_package == "gr-wave-scale":
            from scripts.qwen4exp_halo_box_campaign_ab import gr_wave_control_record
            assert generator.runner.gr_wave_scale is True
            report["selection_control"] = gr_wave_control_record(generator.runner)
        if args.route_package in {"q4-iu8-exact", "q51-iu8-exact", "q5k-iu8-exact", "gr-iu8", "gr-iu8-down", "q8-wmma-down"}:
            # These packages still own a binder-set process key. The Q5_1 fold
            # and Q8_0 row4/bundle rungs are runner state, so their premise is
            # the plan's own publication, asserted in their blocks below.
            assert flag, "a process-keyed route package must name its key"
            assert os.environ[flag] == "1", "production must select the retained route without an override"
        if args.route_package in {"qsa-head-pair", "qsa-head-quad"}:
            # The leaf is runner state, so its production default is asserted
            # through the runner rather than through a binder-set env key.
            assert generator.runner.qsa_prefill_head_mode == "quad", (
                "production must publish the four-head fallback grouping")
        if args.route_package in {"gdn-wave-norm", "gdn-narrow"}:
            # Same for the GDN serial owner: production publishes the plan's own
            # narrow selection on the runner.
            assert generator.runner.gdn_prefill_variant == (
                "qwen4exp_sigmoid_wave_norm_narrow_prefill"), (
                "production must publish the narrow GDN serial owner")
            # The counted serial leaf is the tiled branch's fallback, so the
            # premise of every serial-arm count is that production binds the
            # tiled branch open. ``route_scope`` closes it per measured run.
            assert os.environ.get(GDN_COLWARPS_ENV, "") not in {"", "0"}, (
                "production must bind the tiled GDN branch before a serial arm")
            report["serial_route_control"] = {
                "environment": GDN_COLWARPS_ENV,
                "value": "0",
                "arms": "both",
                "scope": "per measured run; restored on exit",
                "production_value": os.environ[GDN_COLWARPS_ENV],
                "reason": (
                    "production binds the row-split tiled recurrence on every "
                    "GDN layer, so the counted serial leaf only launches with "
                    "the tiled branch closed"),
            }
        if args.route_package == "q51-register-cache":
            # The plan publishes the register-cache fallback on the runner, but
            # the same preemptors hold the pair2 chain closed and the published
            # row-publish sibling replaces the leaf this arm counts, so both
            # arms close them, hold the sibling off and record the pair2 gate.
            assert generator.runner.q51_register_cache is True, (
                "production must publish the register-cache fallback")
            assert getattr(generator.runner, Q51_PAIR2_GATE_FIELD, False), (
                "the pair2 chain's own gate must be published on the runner")
            report["selection_control"] = q51_register_cache_control_record(
                generator.runner)
        if args.route_package == "q51-row-publish":
            # The plan publishes the row-publish selection on the runner, but
            # the routed F16 WMMA down (and, behind it, the exact iu8 down and
            # the grouped-WMMA/MoE/MMQ/T16 owners) preempts the whole pair2
            # chain this arm counts, so both arms close them and the record
            # separates the forced value from what it replaced.
            assert generator.runner.q51_row_publish is True, (
                "production must publish the row-publish selection")
            assert os.environ.get(Q51_IU8_EXACT_ENV, "") not in {"", "0"}, (
                "production must bind the preempting exact Q5_1 down")
            assert generator.runner.routed_f16_enabled is True, (
                "production must bind the preempting routed F16 down")
            assert getattr(generator.runner, Q51_PAIR2_GATE_FIELD, False), (
                "the pair2 chain's own gate must be published on the runner")
            report["selection_control"] = {
                "runner_field": "q51_row_publish",
                "before": False, "after": True,
                "scope": "per measured run; restored on exit",
                **q51_pair2_control_record(generator.runner),
                "reason": (
                    "the routed F16 WMMA down is the production owner of this "
                    "scope and, behind it, the exact iu8, grouped-WMMA/"
                    "grouped-MoE, MMQ and T16 owners also preempt the pair2 "
                    "chain the arm counts; both arms close all of them"),
            }
        if args.route_package == "q8-down-register":
            # The plan publishes the down-register selection on the runner, but
            # the routed F16 WMMA down (and, in front of the sorted chain, the
            # Q8_0 WMMA down) preempts both chains this arm counts, so both
            # arms close them and the record separates the forced value from
            # what it replaced.
            assert generator.runner.q8_down_register is True, (
                "production must publish the down-register selection")
            assert generator.runner.routed_f16_enabled is True, (
                "production must bind the preempting routed F16 down")
            assert os.environ.get(
                "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN", "") not in {
                "", "0"}, (
                "production must bind the preempting Q8_0 WMMA down")
            assert getattr(generator.runner, Q8_DOWN_FORKB_FIELD, False), (
                "the Q8_0 expert-down chain gate must be on")
            report["selection_control"] = {
                "runner_field": "q8_down_register",
                "before": False, "after": True,
                "scope": "per measured run; restored on exit",
                **q8_down_control_record(runner=generator.runner),
                "reason": (
                    "the routed F16 WMMA down owns the grouped chain and the "
                    "Q8_0 WMMA down sits in front of the sorted chain, so "
                    "neither counted path reaches the register leaf in "
                    "production; both arms close those owners and the chain "
                    "gates are recorded because they must stay on"),
                "published_value": generator.runner.q8_down_register,
                "mapped_owner_runner_field": generator.runner.q8_mapped_down,
            }
        if args.route_package in {"q51-fold128", "q51-fold-pair"}:
            # The rung is request-local runner state; ``route_scope`` closes
            # the pair2 preemptors, binds the ladder rung this arm does not
            # count off (or requires the parent it rides) and binds the
            # published register-cache/row-publish siblings off. The record
            # separates every bound value from the plan's published one.
            assert getattr(generator.runner, Q51_PAIR_FIELD, False), (
                "production must bind the grouped pair2 Q5_1 selection")
            assert getattr(generator.runner, Q51_PAIR2_GATE_FIELD, False), (
                "the pair2 chain's own gate must be published on the runner")
            if args.route_package == "q51-fold-pair":
                assert getattr(generator.runner, Q51_FOLD128_FIELD, False), (
                    "production must publish the fold128 parent this arm rides")
            report["selection_control"] = q51_fold_control_record(
                args.route_package, generator.runner)
        if args.route_package in {"q8-down-row4", "q8-down-bundle"}:
            # The rung is request-local runner state; ``route_scope`` closes
            # the owners that preempt the sorted chain, binds the rung this arm
            # does not count off (or requires the parent it rides) and binds
            # the published register decision off.
            assert getattr(generator.runner, Q8_DOWN_FORKB_FIELD, False), (
                "production must bind the grouped Q8_0 down chain")
            if args.route_package == "q8-down-bundle":
                assert getattr(generator.runner, Q8_DOWN_ROW4_FIELD, False), (
                    "production must publish the row4 parent this arm rides")
            report["selection_control"] = q8_down_ladder_control_record(
                args.route_package, generator.runner)
        if args.route_package in MMQ_VECTOR_ROUTE_PACKAGES:
            # The plan publishes the child on the runner and ``route_scope``
            # admits the parent prerequisite per arm, so the premise asserted
            # here is the published decision; the scope fails loudly if the
            # parent or the prepacked mapping cannot be provisioned.
            field = ("dense_mmq_raw_vector_enabled"
                     if args.route_package == "q8-mmq-raw-vector"
                     else "dense_mmq_vec4_enabled")
            assert getattr(generator.runner, field) is True, (
                "production must publish the MMQ vector child")
            report["selection_control"] = mmq_vector_control_record(
                args.route_package, generator.runner)
        if args.route_package == "mmq-token64":
            # ``route_scope`` admits the raw parent the leg rides and provisions
            # it per arm; the published leg decision is what this gate measures.
            report["selection_control"] = mmq_token64_control_record(
                generator.runner)
        if args.route_package == "q8-mapped-down":
            # The plan publishes the token-major mapped admission on the runner
            # and the arm binds it per measured run. The current call's row4
            # map is the ownership proof, so the map gate is recorded beside
            # the preemptors the scope closes.
            assert generator.runner.q8_mapped_down is True, (
                "production must publish the token-major mapped down owner")
            assert getattr(generator.runner, Q8_MAPPED_DOWN_MAP_GATE, False), (
                "production must build the grouped row4 gate/up map")
            assert getattr(generator.runner, Q8_DOWN_FORKB_FIELD, False), (
                "the Q8_0 expert-down chain gate must be on")
            assert os.environ.get(
                "HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT", "") not in {"", "0"}, (
                "production must bind the grouped Q5_K IU8 owner the mapped "
                "arm closes")
            report["selection_control"] = {
                "runner_field": "q8_mapped_down",
                "before": False, "after": True,
                "scope": "per measured run; restored on exit",
                **q8_mapped_down_control_record(runner=generator.runner),
                "counted_owners": ["bundle", "register"],
                "counted_owner_rule": (
                    "the mapped path admits the bundle key and the published "
                    "register decision may rewrite it; the arm counts the "
                    "owner that actually launched, filtered to the map pointer"),
                "reason": (
                    "the token-major mapped path admits the bundle leaf only "
                    "from the grouped row4 gate/up map this invocation "
                    "builds; the routed F16 WMMA down, the Q8_0 WMMA down "
                    "and the grouped Q5_K IU8 gate/up owner all preempt that "
                    "map producer, so both arms close them and the chain "
                    "gates and map gate are recorded because they must stay "
                    "on"),
                "published_value": generator.runner.q8_mapped_down,
            }
        if args.route_package == "q8-dpp-reduction":
            # The DPP leaf's flag is read with a "1" default rather than bound by
            # the profile, so its production default is asserted through the
            # default instead of through a binder-set value.
            os.environ.setdefault(flag, "1")
            assert os.environ[flag] == "1", "production must select the retained route without an override"
        if args.route_package in {"q4-bundle", "prefill-bundle"}:
            assert generator.runner.q4_bundle_prefill is True
            report["selection_control"] = {
                "runner_field": "q4_bundle_prefill", "before": False, "after": True,
                "scope": "per measured run; restored on exit",
                "published_value": generator.runner.q4_bundle_prefill,
            }
        if args.route_package == "prefill-bundle":
            assert generator.runner.qsa_h256_prefill_mode == "page256"
            report["selection_control"]["additional_control"] = {
                "runner_field": "qsa_h256_prefill_mode",
                "before": "parent", "after": "page256",
            }
        if args.include_heldouts:
            report["heldout_suite"] = {
                "path": str(heldout_path),
                "sha256": hashlib.sha256(heldout_path.read_bytes()).hexdigest(),
                "case_ids": [row["id"] for row in heldouts],
                "token_policy": ("short prompts repeated/truncated to512 to exercise HC tail"
                                 if args.route_package == "dense-wide-padded"
                                 else "original tokenized prompts"),
            }
        if args.case_id and not set(args.case_id) <= {c["id"] for c in gate_cases}:
            raise ValueError("unknown case id")
        for case in gate_cases:
            if args.case_id:
                if case["id"] not in args.case_id:
                    continue
            elif (not args.all_cases and not str(case["id"]).startswith("heldout_")
                  and case["prompt_tokens"] != 512 and case["id"] != "code-p4096"):
                continue
            baseline = None
            summaries = []
            enabled_sequence = (("0", "1", "1", "1", "0")
                                if narrow_candidate is not None or gdn_narrow_key is not None
                                else ("0", "1", "0"))
            for enabled in enabled_sequence:
                narrow_enabled[0] = enabled == "1"
                apply_state_gate_mode(generator.runner,args.route_package,enabled,flag)
                observed_chunks.clear()
                start_calls = calls[0]
                start_mapped=mapped_calls[0]
                start_owners = dict(mapped_owner_calls)
                start_qsa_calls = qsa_calls[0]
                natural_start = len(natural_router.captures) if natural_router is not None else 0
                # The arm's live control must not survive a failed run, so
                # the stack is entered under ``with``: a raise in the prefill
                # or the decode steps releases every env pin and runner
                # binding before the next arm starts.
                with arm_control(
                        args.route_package, enabled,
                        generator.runner) as case_scope:
                    first = generator.runner.prefill(case["prompt_token_ids"])
                    prefill_invoked = calls[0] - start_calls
                    # Every other arm's control only covers its measured
                    # prefill. The ordered-v2 leaf is decode-only, so its
                    # binding stays in force through the steps that count it
                    # and closes after them; a closed stack keeps this
                    # branch's restore on the same path.
                    if args.route_package != "qsa-ordered-v2":
                        case_scope.close()
                    logits = first.logits.copy()
                    token = int(first.token_id)
                    prefill_state = (_state_summary(generator.runner)
                                     if narrow_candidate is not None or gdn_narrow_key is not None
                                     else None)
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
                    case_scope.close()
                if args.route_package in {"qsa-head-pair","qsa-head-quad"}:
                    assert invoked==prefill_invoked,"head pair ran in decode"
                    expected_pair=24 if enabled=="1" and case["prompt_tokens"]==4096 else 0
                    assert prefill_invoked==expected_pair,(prefill_invoked,expected_pair)
                if args.route_package=="qsa-ordered-v2":
                    assert prefill_invoked==0,"ordered v2 ran in prefill"
                    expected_decode=(args.decode_steps*12
                                    if enabled=="1" and case["prompt_tokens"]==4096 else 0)
                    assert invoked-prefill_invoked==expected_decode,(
                        invoked-prefill_invoked,expected_decode)
                if args.route_package=="mmq-token64":
                    expected_token64=mmq_token64_expected_calls(
                        case["prompt_tokens"],1024) if enabled=="1" else 0
                    assert prefill_invoked==expected_token64,(prefill_invoked,expected_token64)
                    assert invoked==prefill_invoked,"token64 ran in decode"
                if args.route_package == "dense-wide-padded":
                    expected_tail = (tail_expected_calls(case["prompt_tokens"],
                                                        generator.runner.prefill_chunk_size)
                                     if enabled == "1" else 0)
                    assert prefill_invoked == expected_tail, (prefill_invoked, expected_tail)
                    assert invoked == prefill_invoked, "padded tail ran during decode"
                if args.route_package == "gdn-narrow":
                    expected_gdn = (expected_calls(case["prompt_tokens"],
                                                   generator.runner.prefill_chunk_size)
                                    if enabled == "1" else 0)
                    assert prefill_invoked == expected_gdn, (prefill_invoked, expected_gdn)
                    assert invoked == prefill_invoked, "narrow GDN ran in decode"
                if args.route_package == "gdn-wave-norm":
                    expected_gdn=gdn_wave_norm_expected_calls(
                        case["prompt_tokens"],1024) if enabled=="1" else 0
                    assert prefill_invoked==expected_gdn,(prefill_invoked,expected_gdn)
                    assert invoked==prefill_invoked,"wave norm ran in decode"
                if args.route_package=="q8-down-register":
                    expected_register=q8_down_register_expected_calls(
                        generator.runner,case["prompt_tokens"],1024) if enabled=="1" else 0
                    require_engaged_expectation(
                        args.route_package, case, expected_register, 1024,
                        arm_enabled=enabled == "1")
                    assert prefill_invoked==expected_register
                    assert invoked==prefill_invoked,"Q8 register ran during decode"
                    assert mapped_calls[0]-start_mapped==(q8_mapped_down_expected_calls(case["prompt_tokens"],1024) if enabled=="1" else 0)
                if args.route_package == "q8-mapped-down":
                    expected_mapped_calls = q8_mapped_down_expected_calls(
                        case["prompt_tokens"],512) if enabled=="1" else 0
                    assert prefill_invoked == expected_mapped_calls,(case["id"],prefill_invoked)
                    assert invoked == prefill_invoked,"mapped Q8 down ran during decode"
                    owners = {name: mapped_owner_calls[name] - start_owners[name]
                              for name in mapped_owner_calls}
                    assert sum(owners.values()) == prefill_invoked, (
                        case["id"], owners)
                if args.route_package == "q8-mmq-raw-vector":
                    expected_raw_calls = q8_mmq_raw_vector_expected_calls(
                        case["prompt_tokens"],512) if enabled=="1" else 0
                    assert prefill_invoked == expected_raw_calls,(case["id"],prefill_invoked)
                    assert invoked == prefill_invoked,"raw MMQ vector ran during decode"
                if args.route_package == "q8-mmq-vec4":
                    expected_vec4_calls = q8_mmq_vec4_expected_calls(
                        case["prompt_tokens"], 512) if enabled == "1" else 0
                    assert prefill_invoked == expected_vec4_calls, (case["id"], prefill_invoked)
                    assert invoked == prefill_invoked, "MMQ vec4 ran during decode"
                if args.route_package in {"q51-fold128", "q51-fold-pair", "q51-register-cache", "q51-row-publish"}:
                    count_fn = {
                        "q51-fold128": q51_fold128_expected_calls,
                        "q51-fold-pair": q51_fold_pair_expected_calls,
                        "q51-register-cache": q51_register_cache_expected_calls,
                        "q51-row-publish": q51_row_publish_expected_calls,
                    }[args.route_package]
                    expected_fold_calls = count_fn(
                        generator.runner, case["prompt_tokens"],
                        1024 if args.route_package == "q51-row-publish" else 512) if enabled == "1" else 0
                    require_engaged_expectation(
                        args.route_package, case, expected_fold_calls,
                        1024 if args.route_package == "q51-row-publish" else 512,
                        arm_enabled=enabled == "1")
                    assert prefill_invoked == expected_fold_calls, (case["id"], prefill_invoked)
                    assert invoked == prefill_invoked, "fold128 ran during decode"
                if args.route_package in {"q8-down-row4", "q8-down-bundle"}:
                    count_fn = (q8_down_row4_expected_calls
                                if args.route_package == "q8-down-row4"
                                else q8_down_bundle_expected_calls)
                    expected_calls = count_fn(
                        generator.runner, case["prompt_tokens"],
                        512) if enabled == "1" else 0
                    require_engaged_expectation(
                        args.route_package, case, expected_calls, 512,
                        arm_enabled=enabled == "1")
                    assert prefill_invoked == expected_calls, (case["id"], prefill_invoked)
                    assert invoked == prefill_invoked, "prefill candidate ran during decode"
                expected = enabled == "1" and (
                    args.route_package in {"q5k-row4","q4-bundle","prefill-bundle","q51-pair","gdn-register","q4-pair","q8-wave-scale","q8-dpp-reduction","gr-wave-scale","q8-mmq-prepack","q8-down-row4","q51-fold128","q4-iu8-exact","q51-iu8-exact","q5k-iu8-exact","q8-wmma-down"} or case["prompt_tokens"] > 2051)
                if args.route_package == "router-dpp":
                    expected_router = (router_dpp_expected_calls(case["prompt_tokens"],
                                                                generator.runner.prefill_chunk_size)
                                       if enabled == "1" else 0)
                    assert prefill_invoked == expected_router, (prefill_invoked, expected_router)
                    assert invoked == prefill_invoked, "router DPP ran in decode"
                    expected = expected_router > 0
                if args.route_package == "gr-dpp-reduction":
                    expected_calls = (gr_dpp_expected_calls(case["prompt_tokens"],
                                                          generator.runner.prefill_chunk_size)
                                      if enabled == "1" else 0)
                    assert prefill_invoked == expected_calls, (prefill_invoked, expected_calls)
                    assert invoked == prefill_invoked, "GR DPP ran in decode"
                    expected = expected_calls > 0
                if args.route_package in {"q8-down-row4", "q8-down-bundle"}:
                    expected = expected_calls > 0
                if args.route_package in {"q51-fold128", "q51-fold-pair", "q51-register-cache", "q51-row-publish"}:
                    expected = expected_fold_calls > 0
                if args.route_package == "q8-mmq-vec4":
                    expected = expected_vec4_calls > 0
                if args.route_package == "q8-mmq-raw-vector":
                    expected = expected_raw_calls > 0
                if args.route_package == "q8-mapped-down":
                    expected = expected_mapped_calls > 0
                if args.route_package=="q8-down-register":
                    expected=expected_register>0
                if args.route_package == "dense-wide-padded":
                    expected = expected_tail > 0
                if args.route_package in {"gdn-wave-norm", "gdn-narrow"}:
                    expected=expected_gdn>0
                if args.route_package=="mmq-token64":
                    expected=expected_token64>0
                if args.route_package == "chunk1024":
                    validate_chunk_coverage(observed_chunks,case["prompt_tokens"],
                                            generator.runner.prefill_chunk_size)
                    assert invoked == prefill_invoked,"prefill chunks executed during decode"
                    expected = True
                assert (invoked > 0) == expected, f"route not engaged correctly: {case['id']}"
                invoked_qsa = qsa_calls[0] - start_qsa_calls
                if args.route_package == "prefill-bundle":
                    assert (invoked_qsa > 0) == (enabled == "1" and case["prompt_tokens"] > 2051)
                actual = (logits, next_logits, state, prefill_state)
                if baseline is None:
                    baseline = actual
                else:
                    np.testing.assert_array_equal(actual[0], baseline[0])
                    np.testing.assert_array_equal(actual[1], baseline[1])
                    assert state == baseline[2], case["id"]
                    assert prefill_state == baseline[3], case["id"]
                assert state["finite"]
                summaries.append({
                    "enabled": enabled, "state_sha256": state["state_sha256"],
                    "layout_sha256": state["layout_sha256"],
                    "prefill_logits_sha256": hashlib.sha256(logits).hexdigest(),
                    "step_logits_sha256": hashlib.sha256(next_logits).hexdigest(),
                    "candidate_calls": invoked,
                    "candidate_prefill_calls": prefill_invoked,
                    "candidate_decode_calls": invoked - prefill_invoked,
                    "qsa_candidate_calls": invoked_qsa,
                    "step_seconds": step_seconds,
                    "full_kv_sha256": state.get("full_kv_sha256"),
                    "prefill_state_sha256": (prefill_state["state_sha256"]
                                             if prefill_state is not None else None),
                })
                if natural_router is not None:
                    captures = natural_router.captures[natural_start:]
                    assert len(captures) == invoked
                    summaries[-1]["natural_router_captures"] = captures
                    summaries[-1]["router_variant_override"] = os.environ[flag]
                if args.route_package=="chunk1024":
                    summaries[-1].update(active_chunk_size=generator.runner.prefill_chunk_size,
                                         executed_prefill_chunks=list(observed_chunks))
                if args.route_package=="q8-down-register":
                    summaries[-1].update(mapped_candidate_calls=mapped_calls[0]-start_mapped,
                                         compact_candidate_calls=invoked-(mapped_calls[0]-start_mapped))
                if args.route_package=="q8-mapped-down":
                    summaries[-1]["mapped_owner_calls"] = owners
            report["cases"].append({"id": case["id"], "prompt_tokens": case["prompt_tokens"],
                                    "exact": True, "captures": summaries})
            print(case["id"], "full logits/state exact", flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if original_chunk is not None:
            generator.runner._prefill_chunk = original_chunk
            generator.runner.prefill_chunk_size = 1024
        else:
            # A request-local package names no process key, so there is nothing
            # to restore; ``flag`` is checked before the set so an empty name can
            # never reach the environment and fail the cleanup of a run that
            # otherwise passed.
            if flag and args.route_package not in {
                "dense-wide-padded", "gdn-wave-norm", "gdn-narrow",
                "q51-row-publish", "q51-register-cache", "q8-down-register",
                "q8-mapped-down", "q8-mmq-raw-vector", "q8-mmq-vec4",
                "mmq-token64", "q51-fold128", "q51-fold-pair",
                "q8-down-row4", "q8-down-bundle",
            }:
                os.environ[flag] = ("f32_hidden_token_tile4_dense_exact"
                                    if args.route_package == "router-dpp" else "1")
            register(key, original, replace=True)
        if mapped_register_original is not None:
            register(mapped_register_key, mapped_register_original, replace=True)
        if gdn_narrow_original is not None:
            register(gdn_narrow_key, gdn_narrow_original, replace=True)
        if qsa_original is not None:
            register(qsa_key, qsa_original, replace=True)
        generator.close()
        report["memory_after_close"] = memory_stats()
        if narrow_candidate is not None or gdn_narrow_key is not None:
            stats = report["memory_after_close"]
            report["lifecycle_passed"] = stats["active_allocations"] == 0 and stats["current_allocated_bytes"] == 0
            if not report["lifecycle_passed"]:
                report["status"] = "failed"
                report["error"] = "allocation leak after generator close"
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
