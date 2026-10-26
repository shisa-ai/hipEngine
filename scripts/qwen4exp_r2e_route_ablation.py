"""R2e route-drift ablation: leave-one-route-out vs the frozen strict teacher.

Replicates the control comparison (production incumbent, R11 flags off,
teacher-forced chains) with ONE promoted prefill route disabled via its
launch-time env override (applied post-construction, after the profile
binder has set the production matrix). Output is directly comparable to
the recorded control's incumbent_envelope_vs_strict.
"""
import argparse
import contextlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
os.environ.setdefault("HIPENGINE_COMPILER_VERSION_FILE", "/tmp/hipengine-hipcc-version.txt")
os.environ.setdefault("HIPENGINE_REQUIRE_CACHED_BUILD", "1")

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.qwen4_exp_profiles import (
    register_qwen4_exp_gfx1151_profiles, QWEN4_EXP_MODEL, QWEN4_EXP_BACKEND,
    QWEN4_EXP_QUANTS)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime import qwen4_exp_runner as runner_module
from hipengine.runtime.qwen4_exp_runner import qsa_prefill_head_mode
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.models import resolve_model
from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE, load_fixture, _forced_logits, _kl)
from scripts.qwen4exp_halo_box_campaign_ab import (
    head_route_configuration, restore_route_configuration)

# route -> (env name(s), disabled value)
ROUTES = {
    "moe_wmma_27_47": (("HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL",), "0"),
    "q8_mmq": (("HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL",), "0"),
    "row4_gemv": (("HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL",), "0"),
    "q4_iu8": (("HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT",
                "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL"), "0"),
    "decode_dp4a": (("HIPENGINE_QWEN4_EXP_Q4_DP4A64",), "0"),
}

# route -> runner-scoped H256 prefill head grouping. These are request-local
# state on the runner, so the ablation binds the mode instead of an env key.
RUNNER_ROUTE_MODES = {
    "qsa_h256_quad": "parent",
}

# route -> runner-scoped leaf the ablation binds off. The PF-1 fork-b grouped
# down and Q5_1 down M1 selectors are request-local state on the runner, so the
# ablation binds the field instead of writing a process key nothing reads.
RUNNER_ROUTE_OFF = {
    "q51_down_m1": "q51_down_m1",
    "forkb_grouped_down": "forkb_grouped_down",
}

# The scalar sparse-prefill rows owner and its head-grouped leaves. Launches of
# these are counted so the ablation reports engagement instead of a field value.
QSA_SCALAR_PARENT_VARIANT = "strict_h256_page256_wave_rows_spans"
QSA_HEAD_VARIANTS = frozenset({
    "strict_h256_head_pair_rows_spans", "strict_h256_head_quad_rows_spans"})


def count_scalar_launches(counts, *, module=runner_module):
    """Count launches of the scalar rows owner through the runner's resolver.

    The mixer resolves the rows leaf on every sparse chunk and launches it only
    when WMMA attention does not own the chunk, so a resolution count would
    claim engagement the kernel never had. Wrapping the resolved launcher counts
    the calls that actually happen.
    """

    original = module.resolve

    def counting_resolve(*args, **kwargs):
        resolved = original(*args, **kwargs)
        variant = kwargs.get("variant")
        if (kwargs.get("layer") != "qsa_sparse_attention"
                or variant not in (QSA_SCALAR_PARENT_VARIANT, *QSA_HEAD_VARIANTS)):
            return resolved
        bucket = "head" if variant in QSA_HEAD_VARIANTS else "parent"

        def launch(*call_args, **call_kwargs):
            counts[bucket] += 1
            return resolved(*call_args, **call_kwargs)

        return launch

    module.resolve = counting_resolve
    return original


def restore_scalar_launches(original, *, module=runner_module):
    module.resolve = original


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--route", required=True,
                   choices=(sorted(ROUTES) + sorted(RUNNER_ROUTE_MODES)
                            + sorted(RUNNER_ROUTE_OFF)
                            + ["strict_baseline"]))
    p.add_argument("--teacher", required=True)
    p.add_argument("--decode-steps", type=int, default=82)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    profile = (ExecutionProfile.STRICT if a.route == "strict_baseline"
               else ExecutionProfile.PRODUCTION)
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=profile)
    fixture, digest = load_fixture(DEFAULT_FIXTURE)
    root = "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
    index = load_gguf_index(discover_gguf_files(root)[0])
    generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=root, weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend=QWEN4_EXP_BACKEND, max_sequence_length=4352,
        prefill_chunk_size=1024))

    # incumbent arm + route ablation, applied AFTER construction
    runner_mode = RUNNER_ROUTE_MODES.get(a.route)
    runner_off = RUNNER_ROUTE_OFF.get(a.route)
    if (a.route == "strict_baseline" or runner_mode is not None
            or runner_off is not None):
        envs, off = (), "0"
    else:
        envs, off = ROUTES[a.route]
    for name in envs:
        os.environ[name] = off
    engagement_asserts = {name: os.environ[name] for name in envs}
    if runner_mode is not None:
        scope = qsa_prefill_head_mode(generator.runner, runner_mode)
    elif runner_off is not None:
        scope = getattr(runner_module, runner_off)(generator.runner, False)
    else:
        scope = contextlib.nullcontext()
    # The head leaf is the scalar fallback, so the ablation needs WMMA off for
    # its launches to happen at all. Both the override and the launch counter
    # are registered on the stack that unwinds on any failure below.
    counts = {"head": 0, "parent": 0}

    with contextlib.ExitStack() as stack:
        stack.enter_context(scope)
        head_configuration = head_route_configuration(a.route)
        stack.callback(restore_route_configuration, head_configuration)
        stack.callback(
            restore_scalar_launches, count_scalar_launches(counts))
        teacher = np.load(a.teacher)
        all_kls, top1_hits, top1_total = [], 0, 0
        per_case = {}
        for case in fixture["cases"]:
            cid = case["id"]
            forced_chain = teacher[f"{cid}__chain"].tolist()
            ref = teacher[f"{cid}__logits"]
            rows, chain = _forced_logits(
                generator, case["prompt_token_ids"], a.decode_steps,
                forced=forced_chain)
            kls = [_kl(ref[i], rows[i]) for i in range(ref.shape[0])]
            t1 = [bool(ref[i].argmax() == rows[i].argmax()) for i in range(ref.shape[0])]
            all_kls.extend(kls)
            top1_hits += sum(t1)
            top1_total += len(t1)
            per_case[cid] = {
                "kl_mean": float(np.mean(kls)), "kl_max": float(np.max(kls)),
                "top1": float(np.mean(t1)),
                "chain_matches_teacher": bool(list(chain) == list(forced_chain)),
            }
            print(f"  {cid}: mean {per_case[cid]['kl_mean']:.6f} max "
                  f"{per_case[cid]['kl_max']:.6f} top1 {per_case[cid]['top1']:.4f}",
                  flush=True)
        kls = np.array(all_kls)
        # engagement guard: the ablation envs must still hold the disabled value
        for name in envs:
            assert os.environ.get(name) == off, f"engagement lost: {name}"
        if runner_off is not None:
            # Configuration: the ablated leaf stayed bound off for the run.
            assert getattr(generator.runner, runner_off) is False, (
                f"configuration lost: {runner_off}")
        if runner_mode is not None:
            # Configuration: the arm's own decision stayed bound.
            assert generator.runner.qsa_prefill_head_mode == runner_mode, (
                "configuration lost: qsa_prefill_head_mode")
            # Engagement: the scalar rows owner ran, and the ablation removed
            # the head-grouped launches. Counts, not the field value.
            assert counts["parent"] > 0, (
                "scalar rows owner never launched; WMMA still owns the route")
            assert counts["head"] == 0, (
                f"head-grouped leaf launched {counts['head']} times in the "
                "ablated arm")
        report = {
            "schema": 1, "kind": "r2e_route_drift_ablation",
            "route": a.route, "ablated_envs": list(envs),
            "ablated_runner_fields": (
                {runner_off: False} if runner_off is not None else {}),
            "env_values_at_end": engagement_asserts,
            "head_engagement": (
                {
                    "mode": runner_mode,
                    "scalar_parent_launches": counts["parent"],
                    "head_leaf_launches": counts["head"],
                    "wmma": head_configuration,
                }
                if runner_mode is not None else None),
            "decode_steps": a.decode_steps,
            "fixture_sha256": digest,
            "manifest_sha256": resolved.manifest_sha256,
            "envelope": {
                "rows": int(kls.size),
                "kl_mean": float(kls.mean()),
                "kl_p95": float(np.percentile(kls, 95)),
                "kl_p99": float(np.percentile(kls, 99)),
                "kl_max": float(kls.max()),
                "top1_overall": top1_hits / top1_total,
            },
            "per_case": per_case,
        }
    with open(a.output, "w") as h:
        json.dump(report, h, indent=1)
        h.write("\n")
    e = report["envelope"]
    print("ENVELOPE %s: mean %.6f p95 %.6f p99 %.6f max %.6f top1 %.6f" % (
        a.route, e["kl_mean"], e["kl_p95"], e["kl_p99"], e["kl_max"],
        e["top1_overall"]), flush=True)


if __name__ == "__main__":
    main()
