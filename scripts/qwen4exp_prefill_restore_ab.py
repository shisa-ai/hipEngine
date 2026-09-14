"""Counterbalanced same-residency candidate versus current production."""

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
from scripts.qwen4exp_journey_localize import set_flags
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
from scripts.qwen4exp_chunk_workspace import capture_workspace, use_workspace, workspace_description
from scripts.qwen4exp_q8_repair_depth_gate import (
    observe_prefill_chunks, resolve_allocation_profile, validate_chunk_allocation,
)


SHARED_DECODE_SAFE_FLAGS = frozenset(
    "HIPENGINE_QWEN4_EXP_" + flag for flag in (
        "Q8_0_SELECTED_WMMA_DOWN", "Q8_DOWN_VARIANT",
        "QSA_H256_WAVE_PREFILL", "QSA_HEAD_PAIR",
        "QSA_ORDERED_DECODE", "QSA_ORDERED_DECODE_V2",
    )
)


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
                      protocol=dict(pairs=args.pairs, warmups=1, chunk=1024, kv="BF16",
                                    shared_decode_graphs=args.shared_decode_graphs))
        generator, profile, _ = _make_generator(args, "production")
        runner = generator.runner
        donor = None
        workspaces = {}
        controlled = {key for env in candidate_envs.values() for key in env}
        previous = {key: os.environ.get(key) for key in controlled}
        arm_envs = {"before": previous}
        arm_envs.update({name: {**previous, **env} for name, env in candidate_envs.items()})
        report["arm_overrides"] = arm_envs
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
            with use_workspace(runner, workspace) if workspace is not None else nullcontext():
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
                from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
                from scripts.qwen4exp_chunk_memory_probe import prepare_lazy_group_risk

                plan = allocation["admission"]["plan"]
                prepare_lazy_group_risk(runner)
                extra_bound = sum(plan[key] for key in (
                    "scratch_bytes", "kv_bytes", "index_bytes", "runtime_state_bytes"))
                free_before_donor, _ = runner.runtime.mem_get_info()
                if free_before_donor < extra_bound + plan["reserve_bytes"]:
                    raise MemoryError("donor workspace would consume the declared reserve")
                donor = Qwen4ExpGGUFResidentModelRunner(
                    runner.resident, max_sequence_length=runner.max_sequence_length,
                    prefill_chunk_size=1024, backend=runner.backend, runtime=runner.runtime)
                if not np.array_equal(runner.attention_states[0].block_host,
                                      donor.attention_states[0].block_host):
                    raise ValueError("donor block indices differ from active KV layout")
                prepare_lazy_group_risk(donor)
                workspaces = {"before": capture_workspace(donor),
                              args.candidate[0]: capture_workspace(runner)}
                report["allocation_evidence"] = allocation
                report["workspaces"] = {arm: workspace_description(workspace)
                                        for arm, workspace in workspaces.items()}
                report["prepared_memory"] = memory_stats()
                free_after_setup, _ = runner.runtime.mem_get_info()
                if free_after_setup < plan["reserve_bytes"]:
                    raise MemoryError("workspace setup consumed the declared reserve")
                report["workspace_memory_guard"] = dict(
                    free_before_donor=free_before_donor, extra_runner_bound=extra_bound,
                    free_after_setup=free_after_setup, reserve_bytes=plan["reserve_bytes"])
                report["protocol"].update(
                    chunk_by_arm={arm: value["chunk_size"] for arm, value in report["workspaces"].items()},
                    workspace_ownership="Separate correctly sized prefill owners; same active "
                    "KV/recurrent/decode owners and graphs. Shared maximum-size PLE staging. "
                    "Donor non-prefill allocations stay live but are not executed.",
                )
                if args.workspace_check_only:
                    case = next(case for case in fixture["cases"] if case["id"] == "code-p4096")
                    with observe_prefill_chunks(
                        donor, tokens=case["prompt_tokens"], size=1024,
                    ) as native_chunks:
                        reference = _strict_trajectory(donor, case["prompt_token_ids"], 4)
                    reference_state = _state_summary(donor)
                    with use_workspace(runner, workspaces["before"]):
                        with observe_prefill_chunks(
                            runner, tokens=case["prompt_tokens"], size=1024,
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
                    report["protocol"]["workspace_ownership"] = (
                        "Validation-only: donor executed as independent reference; "
                        "active runner then borrowed donor prefill buffers with its own decode state.")
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
