"""Capture complete paired task outputs for manual non-inferiority review.

Predeclared review: each prompt must preserve requested facts, constraints,
code behavior and language. No per-prompt degradation versus strict allowed.
Truncated responses require further review, never an automatic pass.
"""

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.qwen4exp_layer2_profile_gate import _make_generator, CANDIDATES
from scripts.qwen4exp_canonical_ar_bench import _host_metadata, _git_metadata
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
from scripts.qwen4exp_candidate_dispatch import count_candidate_dispatch, shape_records


def task_candidate(name):
    spec = CANDIDATES[name]
    return ("q8_fallback" if name == "production_q8_fallback" else name), spec


def task_dispatch_context(candidate):
    from hipengine.kernels.registry import KernelKey

    counted = candidate.requires_dispatch_count
    return count_candidate_dispatch(
        key=KernelKey(*candidate.candidate_key) if counted else None,
        direct_target=candidate.direct_dispatch_target if counted else None,
        direct_reference=candidate.direct_dispatch_reference if counted else None,
        shape_positions=candidate.dispatch_shape_positions if counted else None,
    )


def select_task_prompts(prompts, requested):
    if requested is None:
        return prompts, True
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate task prompt")
    if not set(requested) <= {row["id"] for row in prompts}:
        raise ValueError("unknown task prompt")
    selected = [row for row in prompts if row["id"] in requested]
    if not selected:
        raise ValueError("empty task prompt selection")
    return selected, len(selected) == len(prompts)


def completion(runner, tokenizer, ids, max_tokens):
    runner.reset()
    result = runner.prefill(ids)
    output = []
    reason = "length"
    for step in range(max_tokens):
        token = int(result.token_id)
        output.append(token)
        if token == tokenizer.eos_token_id:
            reason = "eos"
            break
        if step + 1 < max_tokens:
            result = runner.step(token)
    return dict(ids=output, finish_reason=reason,
                text=tokenizer.decode(output, skip_special=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES),
                        default="production_q8_fallback")
    parser.add_argument("--prompt-id", action="append")
    args = parser.parse_args()
    check_host()
    source = _git_metadata(ROOT)
    if not source["tracked_clean"]:
        parser.error("task capture requires a clean committed tracked tree")
    if args.max_tokens <= 0:
        parser.error("max-tokens must be positive")
    args.prefill_chunk_size = 1024
    args.max_sequence_length = args.max_tokens + 1024
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
        prompts, complete_suite = select_task_prompts(
            _load_suites(DEFAULT_PROMPTS), args.prompt_id)
        candidate_name, candidate = task_candidate(args.candidate)
        report = dict(status="running", command=sys.argv, source=source,
                      host=_host_metadata(), model=model_identity(args.model_root),
                      protocol=__doc__, prompts=prompts, arms={}, lifecycle={},
                      candidate=args.candidate, complete_suite=complete_suite,
                      max_tokens=args.max_tokens, repeats=2,
                      performance_claim=False, task_passed=False)
        for name, profile in (("strict", "strict"), (candidate_name, candidate.base_profile)):
            generator, resolved, _ = _make_generator(args, profile)
            overrides = candidate.environment if name != "strict" else {}
            previous = {key: os.environ.get(key) for key in overrides}
            os.environ.update(overrides)
            report["arms"][name] = dict(manifest=resolved.manifest_sha256,
                                       overrides=dict(overrides), cases=[])
            dispatch_stack = ExitStack()
            try:
                counter = (dispatch_stack.enter_context(task_dispatch_context(candidate))
                           if name != "strict" else None)
                if overrides.get("HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL") == "1":
                    generator.runner.configure_mmq_prefill_resources()
                for prompt in prompts:
                    ids = build_chat_prompt(generator.tokenizer, prompt["prompt"])
                    repeats = []
                    for repeat in range(2):
                        repeats.append(completion(generator.runner, generator.tokenizer,
                                                  ids, args.max_tokens))
                    report["arms"][name]["cases"].append(dict(
                        id=prompt["id"], category=prompt["category"], repeats=repeats,
                        deterministic=repeats[0] == repeats[1]))
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(name, prompt["id"], len(repeats[0]["ids"]),
                          repeats[0]["finish_reason"], flush=True)
                if counter is not None:
                    report["arms"][name]["dispatch"] = {
                        "calls": counter["calls"], "shapes": shape_records(counter),
                        "key": candidate.candidate_key,
                        "direct_target": candidate.direct_dispatch_target,
                        "required": candidate.requires_dispatch_count,
                    }
                    if candidate.requires_dispatch_count and counter["calls"] == 0:
                        raise ValueError("task candidate never dispatched")
            except Exception as error:
                report["status"] = "failed"
                report["error"] = f"{type(error).__name__}: {error}"
                raise
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
                report["lifecycle"][name] = memory_stats()
                args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "captured_requires_manual_review"
        if (_git_metadata(ROOT) != source
                or any(value["current_allocated_bytes"] for value in report["lifecycle"].values())):
            report["status"] = "invalid_source_or_lifecycle"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if report["status"] == "invalid_source_or_lifecycle":
            raise RuntimeError(report["status"])


if __name__ == "__main__":
    main()
