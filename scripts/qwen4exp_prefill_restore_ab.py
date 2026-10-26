"""Counterbalanced same-residency candidate versus current production.

Chunk comparisons time the production arm through one additional prefill-only
owner, so no donor decoder or donor KV cache is allocated. `--workspace-check-only`
still builds and executes an independent full donor as the numerical reference.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from contextlib import nullcontext

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE, load_fixture, _hipengine_case_sample, _host_metadata, _git_metadata,
)
from scripts.qwen4exp_conservative_cost import summarize_cost
from scripts.qwen4exp_layer2_profile_gate import (
    _make_generator, _state_summary, _strict_trajectory, _candidate_trajectory, CANDIDATES,
)
from hipengine.runtime.qwen4_exp_runner import qsa_prefill_head_mode
from scripts.qwen4exp_journey_localize import set_flags
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
from scripts.qwen4exp_chunk_workspace import capture_workspace, use_workspace, workspace_description
from scripts.qwen4exp_q8_repair_depth_gate import (
    observe_prefill_chunks, resolve_allocation_profile, validate_chunk_allocation,
)


SHARED_DECODE_SAFE_FLAGS = frozenset(
    "HIPENGINE_QWEN4_EXP_" + flag for flag in (
        "Q8_0_SELECTED_WMMA_DOWN", "Q8_DOWN_VARIANT",
        "QSA_H256_WAVE_PREFILL",
        "QSA_ORDERED_DECODE",
    )
)

# Chunk size the compared arms are timed at. The runner's base owner keeps the
# larger production chunk; the comparison arm borrows this one.
COMPARISON_CHUNK = 1024

# Prefill-only owners whose extra allocation must be admitted before preparing
# another prefill owner. Same tuple the owned-workspace check uses.
PREFILL_OWNERS = (
    "gdn_prefill_scratch", "qsa_prefill_scratch", "ple_prefill_scratch",
    "qsa_prefill_metadata", "_prefill_buffers",
)

# Total device components admitted for an independent full decoder/KV owner.
DECODER_OWNERS = ("scratch_bytes", "kv_bytes", "index_bytes", "runtime_state_bytes")

PREFILL_ONLY_OWNER = "prefill_only_owner"
FULL_DONOR_REFERENCE = "full_donor_reference"

OWNERSHIP = {
    PREFILL_ONLY_OWNER: (
        "Separate correctly sized prefill owners; same active KV/recurrent/decode "
        "owners and graphs. Shared maximum-size PLE staging. No donor decoder or "
        "donor KV cache is allocated: only prefill scratch and prefill inputs."
    ),
    FULL_DONOR_REFERENCE: (
        "Validation-only: an independent full donor runner is executed as the "
        "numerical reference; the active runner then borrows that donor's prefill "
        "buffers with its own decode state."
    ),
}


def validate_shared_graphs(environments):
    # These switches affect prefill or uncaptured QSA attention, not the
    # captured GDN and MoE decode units or their graph keys.
    changed = {key for env in environments for key in env}
    if not changed <= SHARED_DECODE_SAFE_FLAGS:
        raise ValueError("candidate can change captured decode units")


def round_order(case_index, arms, repetition):
    order = list(arms)
    if case_index % 2:
        order.reverse()
    offset = repetition % len(order)
    return order[offset:] + order[:offset]


def owner_mode(reference):
    """A full decoder/KV owner is only needed where the owner is executed."""
    return FULL_DONOR_REFERENCE if reference else PREFILL_ONLY_OWNER


def prefill_owner_bytes(config, *, context_tokens, chunk_size):
    """Device bytes one additional prefill-only owner must be admitted for."""
    from hipengine.loading.qwen4_exp_scratch import qwen4_exp_scratch_breakdown

    plan = qwen4_exp_scratch_breakdown(
        config, context_tokens=context_tokens, prefill_chunk_size=chunk_size)
    return sum(plan[name] for name in PREFILL_OWNERS)


def donor_owner_bytes(admission_plan):
    """Device bytes the independent full decoder/KV reference owner adds."""
    return sum(admission_plan[key] for key in DECODER_OWNERS)


def admitted_owner_bytes(config, *, reference, context_tokens, chunk_size, admission_plan):
    """Admission bound for the owner this run prepares.

    Timing comparisons admit the prefill-only set. The validation reference is
    executed, so it admits the full decoder/KV owner it actually allocates.
    """
    if reference:
        return donor_owner_bytes(admission_plan)
    return prefill_owner_bytes(
        config, context_tokens=context_tokens, chunk_size=chunk_size)


def require_reserve(free_bytes, admitted_bytes, reserve_bytes, *, mode):
    if free_bytes < admitted_bytes + reserve_bytes:
        raise MemoryError(f"{mode} would consume the declared reserve")


def prepare_prefill_owner(runner, *, chunk_size, context_tokens, reserve_bytes):
    """Prepare one prefill-only owner under an exact device budget.

    The admitted bound and the observed allocation are the same prefill-only
    set, so the extra owner cannot hide a decoder/KV allocation.
    """
    from hipengine.core.memory import memory_stats
    from scripts.qwen4exp_chunk_memory_probe import prepare_lazy_group_risk

    bound = admitted_owner_bytes(
        runner.config, reference=False, context_tokens=context_tokens,
        chunk_size=chunk_size, admission_plan=None)
    free_before, _ = runner.runtime.mem_get_info()
    require_reserve(free_before, bound, reserve_bytes, mode=PREFILL_ONLY_OWNER)
    baseline = memory_stats()["current_allocated_bytes"]
    owner = runner._allocate_extra_prefill_workspace(chunk_size)
    try:
        prepare_lazy_group_risk(owner)
        observed = memory_stats()["current_allocated_bytes"] - baseline
        if observed != bound:
            raise ValueError(
                f"prefill-only owner allocated {observed} bytes, admitted {bound}")
        free_after, _ = runner.runtime.mem_get_info()
        if free_after < reserve_bytes:
            raise MemoryError(
                "prefill-only owner preparation consumed the declared reserve")
    except BaseException:
        try:
            if owner in runner._prefill_workspaces:
                runner._prefill_workspaces.remove(owner)
        finally:
            owner.close()
        raise
    return owner, dict(
        owner_mode=PREFILL_ONLY_OWNER, free_before_owner=free_before,
        admitted_owner_bytes=bound, observed_owner_bytes=observed,
        free_after_setup=free_after, reserve_bytes=reserve_bytes)


def prepare_reference_owner(runner, *, chunk_size, allocation_plan, reserve_bytes):
    """Prepare the independent full decoder/KV owner used as the reference."""
    from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
    from scripts.qwen4exp_chunk_memory_probe import prepare_lazy_group_risk

    bound = admitted_owner_bytes(
        runner.config, reference=True, context_tokens=runner.max_sequence_length,
        chunk_size=chunk_size, admission_plan=allocation_plan)
    free_before, _ = runner.runtime.mem_get_info()
    require_reserve(free_before, bound, reserve_bytes, mode=FULL_DONOR_REFERENCE)
    donor = Qwen4ExpGGUFResidentModelRunner(
        runner.resident, max_sequence_length=runner.max_sequence_length,
        prefill_chunk_size=chunk_size, backend=runner.backend, runtime=runner.runtime)
    try:
        if not np.array_equal(runner.attention_states[0].block_host,
                              donor.attention_states[0].block_host):
            raise ValueError("donor block indices differ from active KV layout")
        prepare_lazy_group_risk(donor)
        free_after, _ = runner.runtime.mem_get_info()
        if free_after < reserve_bytes:
            raise MemoryError("donor workspace preparation consumed the declared reserve")
    except BaseException:
        donor.close()
        raise
    return donor, dict(
        owner_mode=FULL_DONOR_REFERENCE, free_before_owner=free_before,
        admitted_owner_bytes=bound, observed_owner_bytes=None,
        free_after_setup=free_after, reserve_bytes=reserve_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True, action="append")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--shared-decode-graphs", action="store_true")
    parser.add_argument("--candidate-chunk-size", type=int)
    parser.add_argument("--allocation-evidence", type=Path)
    parser.add_argument("--workspace-check-only", action="store_true")
    args = parser.parse_args()
    check_host()
    if not _git_metadata(ROOT)["tracked_clean"]:
        parser.error("performance/workspace validation requires a clean tracked tree")
    if args.workspace_check_only and args.candidate_chunk_size is None:
        parser.error("--workspace-check-only requires a chunk comparison")
    if args.pairs < 3:
        parser.error("at least three pairs required")
    if len(set(args.candidate)) != len(args.candidate):
        parser.error("duplicate candidates")
    candidate_envs = {name: dict(CANDIDATES[name].environment) for name in args.candidate}
    if args.candidate_chunk_size is not None:
        if (args.candidate_chunk_size <= 1024 or args.allocation_evidence is None
                or len(candidate_envs) != 1 or any(candidate_envs.values())
                or not args.shared_decode_graphs):
            parser.error("chunk comparison requires one unchanged production arm, larger chunk, "
                         "allocation evidence and shared decode graphs")
    if args.shared_decode_graphs:
        validate_shared_graphs(candidate_envs.values())
    args.prefill_chunk_size = args.candidate_chunk_size or 1024
    args.max_sequence_length = 4096 + 128 + 8
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
    from hipengine.runtime.gguf_linear import clear_gguf_linear_dispatch_cache
    from hipengine.runtime.moe_graph import MoeGraphCache

    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        register_gfx1151_kernels(replace=True)
        register_qwen4_exp_gfx1151_profiles()
        reset_memory_stats()
        host, model = _host_metadata(), model_identity(args.model_root)
        allocation = None
        if args.candidate_chunk_size is not None:
            allocation = json.loads(args.allocation_evidence.read_bytes())
            allocation_profile = resolve_allocation_profile()
            validate_chunk_allocation(
                allocation, chunk=args.candidate_chunk_size, context=args.max_sequence_length,
                manifest=allocation_profile.manifest_sha256, host=host, model=model)
        fixture, digest = load_fixture(DEFAULT_FIXTURE)
        report = dict(status="running", command=sys.argv, source=_git_metadata(ROOT),
                      host=host, model=model,
                      fixture_sha256=digest, samples=[], performance_claim=False,
                      candidates=args.candidate,
                      protocol=dict(pairs=args.pairs, warmups=1, chunk=COMPARISON_CHUNK,
                                    kv="BF16", shared_decode_graphs=args.shared_decode_graphs))
        generator, profile, _ = _make_generator(args, "production")
        runner = generator.runner
        donor = None
        workspaces = {}
        controlled = {key for env in candidate_envs.values() for key in env}
        previous = {key: os.environ.get(key) for key in controlled}
        arm_envs = {"before": previous}
        arm_envs.update({name: {**previous, **env} for name, env in candidate_envs.items()})
        report["arm_overrides"] = arm_envs
        # The QSA prefill head grouping is request-local runner state, so an arm
        # that declares one is bound by scope rather than by an env key.
        arm_head_modes = {name: CANDIDATES[name].qsa_prefill_head_mode
                          for name in args.candidate}
        report["arm_head_modes"] = {
            name: mode for name, mode in arm_head_modes.items() if mode is not None}
        original_step = runner.step
        last = {}
        names = ("moe_graph_cache", "layer_graph_cache")
        caches = {"before": {name: getattr(runner, name) for name in names}}
        for candidate in args.candidate:
            caches[candidate] = caches["before"] if args.shared_decode_graphs else {
                name: MoeGraphCache(runner.runtime, enabled=cache.enabled)
                for name, cache in caches["before"].items()
            }

        def step(*a, **kw):
            result = original_step(*a, **kw)
            last["result"] = result
            return result

        runner.step = step

        def sample(arm, case, rep):
            runner.runtime.device_synchronize()
            for name, cache in caches[arm].items():
                setattr(runner, name, cache)
            clear_gguf_linear_dispatch_cache()
            set_flags(arm_envs[arm])
            if os.environ.get("HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL") == "1":
                runner.configure_mmq_prefill_resources()
            workspace = workspaces.get(arm)
            with (
                qsa_prefill_head_mode(runner, arm_head_modes.get(arm)),
                use_workspace(runner, workspace)
                if workspace is not None else nullcontext(),
            ):
                with observe_prefill_chunks(
                    runner, tokens=case["prompt_tokens"], size=runner.prefill_chunk_size,
                ) as chunks:
                    row = _hipengine_case_sample(
                        runner, case=case, repetition=rep,
                        transitions=fixture["decode_transitions"])
                row["prefill_chunks"] = chunks
                row["active_chunk_size"] = runner.prefill_chunk_size
            logits = np.asarray(last["result"].logits)
            state = _state_summary(runner)
            if not logits.size or not np.isfinite(logits).all() or not state["finite"]:
                raise ValueError("nonfinite logits/state")
            row.update(arm=arm, mode="before" if arm == "before" else "after", finite=True,
                       logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest(),
                       state_sha256=state["state_sha256"])
            return row

        try:
            if args.candidate_chunk_size is not None:
                from scripts.qwen4exp_chunk_memory_probe import prepare_lazy_group_risk

                plan = allocation["admission"]["plan"]
                prepare_lazy_group_risk(runner)
                mode = owner_mode(args.workspace_check_only)
                if args.workspace_check_only:
                    owner, guard = prepare_reference_owner(
                        runner, chunk_size=COMPARISON_CHUNK,
                        allocation_plan=plan, reserve_bytes=plan["reserve_bytes"])
                    donor = owner
                else:
                    owner, guard = prepare_prefill_owner(
                        runner, chunk_size=COMPARISON_CHUNK,
                        context_tokens=args.max_sequence_length,
                        reserve_bytes=plan["reserve_bytes"])
                workspaces = {"before": capture_workspace(owner),
                              args.candidate[0]: capture_workspace(runner)}
                report["allocation_evidence"] = allocation
                report["workspaces"] = {arm: workspace_description(workspace)
                                        for arm, workspace in workspaces.items()}
                report["prepared_memory"] = memory_stats()
                report["workspace_memory_guard"] = guard
                report["protocol"].update(
                    chunk_by_arm={arm: value["chunk_size"] for arm, value in report["workspaces"].items()},
                    workspace_ownership=OWNERSHIP[mode],
                )
                if args.workspace_check_only:
                    case = next(case for case in fixture["cases"] if case["id"] == "code-p4096")
                    with observe_prefill_chunks(
                        donor, tokens=case["prompt_tokens"], size=COMPARISON_CHUNK,
                    ) as native_chunks:
                        reference = _strict_trajectory(donor, case["prompt_token_ids"], 4)
                    reference_state = _state_summary(donor)
                    with use_workspace(runner, workspaces["before"]):
                        with observe_prefill_chunks(
                            runner, tokens=case["prompt_tokens"], size=COMPARISON_CHUNK,
                        ) as borrowed_chunks:
                            actual = _candidate_trajectory(
                                runner, case["prompt_token_ids"],
                                [row["token_id"] for row in reference[:-1]])
                        actual_state = _state_summary(runner)
                    for before, after in zip(reference, actual, strict=True):
                        np.testing.assert_array_equal(before["logits"], after["logits"])
                        if before["token_id"] != after["token_id"]:
                            raise ValueError("borrowed workspace changed token")
                    if actual_state != reference_state:
                        raise ValueError("borrowed workspace changed state")
                    report["workspace_check"] = dict(
                        case_id=case["id"], rows=len(reference), logits_exact=True,
                        reference_profile="production",
                        state_exact=True, native_chunks=native_chunks,
                        borrowed_chunks=borrowed_chunks)
                    report["status"] = "workspace_check_passed"
                    return
            for index, case in enumerate(fixture["cases"]):
                arms = tuple(arm_envs)
                for arm in arms:
                    sample(arm, case, -1)
                for repetition in range(args.pairs):
                    for slot, arm in enumerate(round_order(index, arms, repetition)):
                        row = sample(arm, case, repetition)
                        row["sequence_slot"] = repetition * len(arms) + slot
                        report["samples"].append(row)
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(case["id"], arm, row["prefill_tok_s"], row["decode_tok_s"], flush=True)
            report["comparisons"] = {
                candidate: summarize_cost(
                    [row for row in report["samples"] if row["arm"] in ("before", candidate)],
                    args.pairs)
                for candidate in args.candidate
            }
            for case in fixture["cases"]:
                for arm in arm_envs:
                    rows = [r for r in report["samples"] if
                            r["case_id"] == case["id"] and r["arm"] == arm]
                    for field in ("logits_sha256", "state_sha256"):
                        if len({row[field] for row in rows}) != 1:
                            raise ValueError(f"nonrepeatable {field}: {case['id']} {arm}")
            report["status"] = "completed"
        finally:
            set_flags(previous)
            runner.step = original_step
            runner.runtime.device_synchronize()
            report["graph_stats"] = {
                arm: {name: cache.stats for name, cache in arm_caches.items()}
                for arm, arm_caches in caches.items()
            }
            for mode_caches in caches.values():
                for name, cache in mode_caches.items():
                    if cache is not getattr(runner, name):
                        cache.close()
            try:
                if donor is not None:
                    report["donor_graph_stats"] = dict(
                        moe=donor.moe_graph_cache.stats, layer=donor.layer_graph_cache.stats)
                    donor.close()
            finally:
                generator.close()
            report["after_close"] = memory_stats()
            report["base_manifest"] = profile.manifest_sha256
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
