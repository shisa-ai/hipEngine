"""Qualify Q8 fallback numerics across canonical 512/1K/4K prefills."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.benchmark.execution_profiles import RowDescriptor, compare_profile_logits
from scripts.qwen4exp_layer2_profile_gate import (
    _make_generator, _strict_trajectory, _candidate_trajectory, _state_summary,
    _state_repeat_gate, CANDIDATES,
)
from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE, load_fixture, _host_metadata, _git_metadata,
)
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
from scripts.qwen4exp_candidate_dispatch import count_candidate_dispatch, shape_records


def resolve_allocation_profile():
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    # Import the same kernel families as generator construction before resolving.
    from hipengine.generation import qwen4_exp_gguf
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_MODEL, QWEN4_EXP_BACKEND, QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels()
    register_qwen4_exp_gfx1151_profiles()
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION)


def validate_chunk_allocation(packet, *, chunk, context, manifest, host, model):
    if (packet.get("schema", 0) < 2 or packet["status"] != "passed"
            or not packet["source"]["tracked_clean"]
            or packet["chunk_size"] < chunk or packet["prepared_context"] < context
            or packet["prepared_runners"] < 1 or not packet["lazy_group_risk"]
            or packet["manifest_sha256"] != manifest
            or packet["host"]["machine_id"] != host["machine_id"]
            or packet["model_identity"]["fingerprint"] != model["fingerprint"]
            or packet["model_identity"]["revision"] != model["revision"]
            or packet["allocation_margins"]["scratch_margin_bytes"] < 0
            or packet["memory_after_close"]["current_allocated_bytes"]):
        raise ValueError("chunk requires matching passed allocation and lazy-queue evidence")
    queues = packet["lazy_group_risk"][0]["queues"]
    if (len(queues) != 2
            or {row["owner"] for row in queues} != {"gdn_prefill_scratch", "qsa_prefill_scratch"}
            or any(row["rows"] < chunk or row["compact_rows"] < row["rows"]
                   or row["output_width"] <= 0
                   or row["nbytes"] < 4 + row["compact_rows"] * row["output_width"] * 4
                   for row in queues)):
        raise ValueError("incomplete chunk repair queue preparation")


@contextmanager
def observe_prefill_chunks(runner, *, tokens, size):
    from scripts.qwen4exp_halo_box_campaign_ab import validate_chunk_coverage

    original = runner._prefill_chunk
    had_override = "_prefill_chunk" in vars(runner)
    observed = []

    def counted(values, **kwargs):
        observed.append(len(values))
        return original(values, **kwargs)

    runner._prefill_chunk = counted
    try:
        yield observed
        validate_chunk_coverage(observed, tokens, size)
    finally:
        if had_override:
            runner._prefill_chunk = original
        else:
            del runner._prefill_chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES),
                        default="production_conservative")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--allocation-evidence", type=Path)
    args = parser.parse_args()
    check_host()
    if args.decode_steps < 1 or args.repeats < 3 or args.prefill_chunk_size < 1:
        parser.error("positive decode-steps and at least three repeats required")
    if args.prefill_chunk_size > 1024 and args.allocation_evidence is None:
        parser.error("larger chunks require --allocation-evidence")
    args.max_sequence_length = 4096 + args.decode_steps + 8
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles

    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        register_gfx1151_kernels(replace=True)
        register_qwen4_exp_gfx1151_profiles()
        reset_memory_stats()
        host, model = _host_metadata(), model_identity(args.model_root)
        allocation = None
        if args.allocation_evidence is not None:
            resolved = resolve_allocation_profile()
            allocation = json.loads(args.allocation_evidence.read_bytes())
            validate_chunk_allocation(
                allocation, chunk=args.prefill_chunk_size, context=args.max_sequence_length,
                manifest=resolved.manifest_sha256, host=host, model=model)
        fixture, fixture_hash = load_fixture(DEFAULT_FIXTURE)
        cases = fixture["cases"]
        if args.case_id:
            requested = set(args.case_id)
            if not requested <= {case["id"] for case in cases}:
                raise ValueError("unknown canonical case")
            cases = [case for case in cases if case["id"] in requested]
        descriptors, teachers, strict_states = [], {}, {}
        report = dict(status="running", command=sys.argv, source=_git_metadata(ROOT),
                      host=host, model=model, candidate=args.candidate,
                      arithmetic_class=("T2" if args.prefill_chunk_size != 1024
                                        else CANDIDATES[args.candidate].classification),
                      fixture_sha256=fixture_hash, lifecycle={},
                      performance_claim=False, promotion_claim=False,
                      protocol=dict(chunk=args.prefill_chunk_size, strict_chunk=1024,
                                    kv="BF16", decode_steps=args.decode_steps,
                                    repeats=args.repeats, task_gate="separate",
                                    complete_fixture=not bool(args.case_id),
                                    case_ids=[case["id"] for case in cases]),
                      allocation_evidence=allocation, chunk_dispatches=[])
        strict_args = SimpleNamespace(**vars(args))
        strict_args.prefill_chunk_size = 1024
        generator, profile, _ = _make_generator(strict_args, "strict")
        report["strict_manifest"] = profile.manifest_sha256
        try:
            for case in cases:
                with observe_prefill_chunks(
                    generator.runner, tokens=case["prompt_tokens"], size=1024,
                ) as chunks:
                    trajectory = _strict_trajectory(
                        generator.runner, case["prompt_token_ids"], args.decode_steps)
                report["chunk_dispatches"].append(dict(
                    arm="strict", case_id=case["id"], repeat=0, chunks=chunks))
                teachers[case["id"]] = trajectory
                strict_states[case["id"]] = {
                    **_state_summary(generator.runner), "prompt_id": case["id"]}
                for step, sample in enumerate(trajectory):
                    descriptors.append(RowDescriptor(
                        scenario_id="q8-repair-canonical-depth", scenario_step=len(descriptors),
                        request_id=case["id"], teacher_step=step, category=case["category"],
                        shape=f"p{case['prompt_tokens']}_prefill_last" if step == 0
                              else f"p{case['prompt_tokens']}_c1",
                        transition="prefill_to_c1" if step == 0 else "steady",
                        teacher_token_id=sample["token_id"],
                    ))
                print("strict", case["id"], flush=True)
        finally:
            generator.close()
            report["lifecycle"]["strict"] = memory_stats()
        generator, profile, _ = _make_generator(args, "production")
        report["production_base_manifest"] = profile.manifest_sha256
        overrides = dict(CANDIDATES[args.candidate].environment)
        report["overrides"] = overrides
        previous = {key: os.environ.get(key) for key in overrides}
        os.environ.update(overrides)
        outputs, states = [], []
        deterministic = True
        dispatch_stack = ExitStack()
        counter = {"calls": 0, "shapes": {}}
        try:
            if overrides.get("HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL") == "1":
                generator.runner.configure_mmq_prefill_resources()
            spec = CANDIDATES[args.candidate]
            if spec.requires_dispatch_count:
                from hipengine.kernels.registry import KernelKey
                counter = dispatch_stack.enter_context(count_candidate_dispatch(
                    key=KernelKey(*spec.candidate_key),
                    direct_target=spec.direct_dispatch_target,
                    direct_reference=spec.direct_dispatch_reference,
                    shape_positions=spec.dispatch_shape_positions))
            for case in cases:
                teacher = teachers[case["id"]]
                reference = None
                state_reference = None
                case_states = []
                for repeat in range(args.repeats):
                    with observe_prefill_chunks(
                        generator.runner, tokens=case["prompt_tokens"],
                        size=args.prefill_chunk_size,
                    ) as chunks:
                        trajectory = _candidate_trajectory(
                            generator.runner, case["prompt_token_ids"],
                            [sample["token_id"] for sample in teacher[:-1]])
                    report["chunk_dispatches"].append(dict(
                        arm="candidate", case_id=case["id"], repeat=repeat, chunks=chunks))
                    logits = np.stack([sample["logits"] for sample in trajectory])
                    state = _state_summary(generator.runner)
                    case_states.append(state)
                    if reference is None:
                        reference, state_reference = logits, state
                    else:
                        deterministic &= bool(np.array_equal(reference, logits)
                                              and state == state_reference)
                    print("candidate", case["id"], repeat, flush=True)
                outputs.extend(reference)
                states.append(case_states)
            strict_logits = np.stack([sample["logits"] for case in cases
                                      for sample in teachers[case["id"]]])
            report["quality"] = compare_profile_logits(
                strict_logits, np.stack(outputs), descriptors)
            report["deterministic"] = deterministic
            report["state_gate"] = _state_repeat_gate(
                [strict_states[case["id"]] for case in cases], states)
            report["candidate_dispatch_calls"] = counter["calls"]
            report["candidate_dispatch_shapes"] = shape_records(counter)
            report["candidate_dispatch_shape_positions"] = spec.dispatch_shape_positions
            report["candidate_direct_dispatch_target"] = spec.direct_dispatch_target
            report["candidate_direct_dispatch_reference"] = spec.direct_dispatch_reference
            report["candidate_registered_key"] = spec.candidate_key
            report["candidate_dispatch_mode"] = (
                "direct_alias" if spec.direct_dispatch_target
                else "registry" if spec.count_registered_dispatch else "not_counted")
            if spec.requires_dispatch_count and counter["calls"] == 0:
                raise ValueError("candidate never dispatched")
            report["status"] = "completed"
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            try:
                generator.close()
            finally:
                dispatch_stack.close()
            report["lifecycle"]["candidate"] = memory_stats()
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
