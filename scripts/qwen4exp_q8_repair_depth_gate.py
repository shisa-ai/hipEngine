"""Qualify Q8 fallback numerics across canonical 512/1K/4K prefills."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from functools import wraps

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
    args = parser.parse_args()
    check_host()
    if args.decode_steps < 1 or args.repeats < 3:
        parser.error("positive decode-steps and at least three repeats required")
    args.prefill_chunk_size = 1024
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
        fixture, fixture_hash = load_fixture(DEFAULT_FIXTURE)
        cases = fixture["cases"]
        if args.case_id:
            requested = set(args.case_id)
            if not requested <= {case["id"] for case in cases}:
                raise ValueError("unknown canonical case")
            cases = [case for case in cases if case["id"] in requested]
        descriptors, teachers, strict_states = [], {}, {}
        report = dict(status="running", command=sys.argv, source=_git_metadata(ROOT),
                      host=_host_metadata(), model=model_identity(args.model_root),
                      fixture_sha256=fixture_hash, lifecycle={},
                      performance_claim=False, promotion_claim=False,
                      protocol=dict(chunk=1024, kv="BF16", decode_steps=args.decode_steps,
                                    repeats=args.repeats, task_gate="separate",
                                    complete_fixture=not bool(args.case_id),
                                    case_ids=[case["id"] for case in cases]))
        generator, profile, _ = _make_generator(args, "strict")
        report["strict_manifest"] = profile.manifest_sha256
        try:
            for case in cases:
                trajectory = _strict_trajectory(generator.runner, case["prompt_token_ids"],
                                                args.decode_steps)
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
        dispatch_count = 0
        dispatch_original = None
        try:
            if overrides.get("HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL") == "1":
                generator.runner.configure_mmq_prefill_resources()
            spec = CANDIDATES[args.candidate]
            if spec.count_registered_dispatch:
                from hipengine.kernels.registry import KernelKey, register, resolve
                key = KernelKey(*spec.candidate_key)
                dispatch_original = resolve(backend=key.backend, layer=key.layer,
                                            quant=key.quant, variant=key.variant)

                @wraps(dispatch_original)
                def counted(*a, **kw):
                    nonlocal dispatch_count
                    dispatch_count += 1
                    return dispatch_original(*a, **kw)

                register(key, counted, replace=True)
            for case in cases:
                teacher = teachers[case["id"]]
                reference = None
                state_reference = None
                case_states = []
                for repeat in range(args.repeats):
                    trajectory = _candidate_trajectory(
                        generator.runner, case["prompt_token_ids"],
                        [sample["token_id"] for sample in teacher[:-1]])
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
            report["candidate_dispatch_calls"] = dispatch_count
            if spec.count_registered_dispatch and dispatch_count == 0:
                raise ValueError("candidate never dispatched")
            report["status"] = "completed"
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            generator.close()
            if dispatch_original is not None:
                register(key, dispatch_original, replace=True)
            report["lifecycle"]["candidate"] = memory_stats()
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
