"""Supplementary active-shape tasks; not a replacement for the full profile gate.

Predeclared criterion: complete, finite, repeatable outputs; no loss on any
reference-correct task. The prompt requires only an option letter, so merely
mentioning the expected answer is insufficient. Reference failures remain visible.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_candidate_dispatch import count_candidate_dispatch, shape_records
from scripts.qwen4exp_layer2_profile_gate import _make_generator, _state_summary, CANDIDATES
from scripts.qwen4exp_q8_repair_depth_gate import (
    observe_prefill_chunks, resolve_allocation_profile, validate_chunk_allocation,
)
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity


def score_choice(text, expected):
    return text.strip().upper() == expected


def task_verdict(rows):
    if not rows or not all(row["valid"] for row in rows):
        return "invalid_capture"
    if any(row["strict_correct"] and not row["candidate_correct"] for row in rows):
        return "task_regression"
    if not all(row["candidate_correct"] for row in rows):
        return "reference_unscorable"
    return "passed_supplemental"


def completion(runner, tokenizer, prompt, limit):
    runner.reset()
    result = runner.prefill(prompt)
    tokens = []
    finite = True
    finish = "length"
    for index in range(limit):
        logits = np.asarray(result.logits)
        finite &= bool(logits.size and np.isfinite(logits).all())
        token = int(result.token_id)
        tokens.append(token)
        if token == tokenizer.eos_token_id:
            finish = "eos"
            break
        if index + 1 < limit:
            result = runner.step(token)
    visible = tokens[:-1] if finish == "eos" else tokens
    state = _state_summary(runner)
    return dict(ids=tokens, text=tokenizer.decode(visible, skip_special=False),
                finish=finish, finite=finite and state["finite"],
                state_sha256=state["state_sha256"])


def traced_completion(runner, tokenizer, prompt, limit, chunk):
    with observe_prefill_chunks(runner, tokens=len(prompt), size=chunk) as chunks:
        result = completion(runner, tokenizer, prompt, limit)
    return {**result, "prefill_chunks": chunks}


def build_active_prompt(generator, task, *, context_tokens=4096):
    from scripts.gguf_mtp_long_context_task_gate import _TokenizerAdapter
    from scripts.qwen35_paro_kv_quality_smoke import _build_prompt_tokens

    marker = "__HIPENGINE_ACTIVE_TASK_CONTENT__"
    rendered = generator.render_chat_prompt(
        [{"role": "user", "content": marker}], enable_thinking=False)
    if rendered.count(marker) != 1:
        raise ValueError("embedded template must preserve the task content exactly once")
    before, _, after = rendered.partition(marker)
    raw_task = {
        **task, "prompt_format": "raw",
        "prefix": before + task["prefix"], "suffix": task["suffix"] + after,
    }
    prompt, metadata = _build_prompt_tokens(
        _TokenizerAdapter(generator.tokenizer), raw_task, context_tokens=context_tokens)
    metadata.update(
        prompt_format="qwen4exp_embedded", enable_thinking=False,
        chat_template_sha256=hashlib.sha256(generator.tokenizer.chat_template.encode()).hexdigest())
    return prompt, metadata


def main():
    from scripts.gguf_mtp_long_context_task_gate import load_tasks, DEFAULT_SUITE
    from hipengine.core.memory import memory_stats
    from hipengine.kernels.registry import KernelKey

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--allocation-evidence", type=Path)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.prefill_chunk_size < 1 or args.max_tokens < 1:
        parser.error("positive chunk and output limits required")
    if args.prefill_chunk_size > 1024 and args.allocation_evidence is None:
        parser.error("larger chunks require allocation evidence")
    check_host()
    source = _git_metadata(ROOT)
    if not source["tracked_clean"]:
        parser.error("task capture requires clean committed source")
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    args.max_sequence_length = 4096 + args.max_tokens + 8
    spec = CANDIDATES[args.candidate]
    host, model = _host_metadata(), model_identity(args.model_root)
    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        profile = resolve_allocation_profile()
        if args.allocation_evidence is not None:
            validate_chunk_allocation(
                json.loads(args.allocation_evidence.read_bytes()), chunk=args.prefill_chunk_size,
                context=args.max_sequence_length, manifest=profile.manifest_sha256,
                host=host, model=model)
        tasks = load_tasks(DEFAULT_SUITE)
        packet = dict(
            status="running", source=source, host=host, model=model, command=sys.argv,
            performance_claim=False, promotion_claim=False, criterion=__doc__,
            fixture_sha256=hashlib.sha256(DEFAULT_SUITE.read_bytes()).hexdigest(),
            protocol=dict(context=4096, repeats=3, max_tokens=args.max_tokens,
                          strict_chunk=1024, candidate_chunk=args.prefill_chunk_size),
            candidate=args.candidate, overrides=dict(spec.environment),
            cases={}, lifecycle={}, comparisons=[])
        for arm in ("strict", "candidate"):
            arm_args = SimpleNamespace(**vars(args))
            if arm == "strict":
                arm_args.prefill_chunk_size = 1024
            generator, resolved, _ = _make_generator(
                arm_args, "strict" if arm == "strict" else spec.base_profile)
            overrides = {} if arm == "strict" else spec.environment
            previous = {key: os.environ.get(key) for key in overrides}
            os.environ.update(overrides)
            packet["cases"][arm] = []
            try:
                if overrides.get("HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL") == "1":
                    generator.runner.configure_mmq_prefill_resources()
                counted = arm == "candidate" and spec.requires_dispatch_count
                with count_candidate_dispatch(
                    key=KernelKey(*spec.candidate_key) if counted else None,
                    direct_target=spec.direct_dispatch_target if counted else None,
                    direct_reference=spec.direct_dispatch_reference if counted else None,
                    shape_positions=spec.dispatch_shape_positions if counted else None,
                ) as counter:
                    for task in tasks:
                        prompt, prompt_info = build_active_prompt(generator, task)
                        runs = [traced_completion(
                            generator.runner, generator.tokenizer, prompt,
                            args.max_tokens, arm_args.prefill_chunk_size)
                                for _ in range(3)]
                        packet["cases"][arm].append(dict(
                            id=task["id"], category=task["category"], prompt=prompt_info,
                            expected=task["expected_choice"], runs=runs,
                            repeated=all(run == runs[0] for run in runs),
                            correct=score_choice(runs[0]["text"], task["expected_choice"]),
                            manifest=resolved.manifest_sha256))
                        args.output.write_text(json.dumps(packet, indent=2) + "\n")
                        print(arm, task["id"], runs[0]["text"], runs[0]["finish"], flush=True)
                    if counted and counter["calls"] == 0:
                        raise ValueError("candidate never dispatched")
                    if arm == "candidate":
                        packet["candidate_dispatch_calls"] = counter["calls"]
                        packet["candidate_dispatch_shapes"] = shape_records(counter)
            except Exception as error:
                packet["status"] = "invalid_capture"
                packet["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                generator.close()
                packet["lifecycle"][arm] = memory_stats()
                args.output.write_text(json.dumps(packet, indent=2) + "\n")
        for before, after in zip(packet["cases"]["strict"], packet["cases"]["candidate"], strict=True):
            valid = bool(
                before["id"] == after["id"] and before["prompt"] == after["prompt"]
                and before["repeated"] and after["repeated"]
                and all(run["finish"] == "eos" and run["finite"]
                        for case in (before, after) for run in case["runs"]))
            packet["comparisons"].append(dict(
                id=before["id"], valid=valid, strict_correct=before["correct"],
                candidate_correct=after["correct"],
                output_ids_exact=before["runs"][0]["ids"] == after["runs"][0]["ids"]))
        packet["status"] = task_verdict(packet["comparisons"])
        if (_git_metadata(ROOT) != source
                or any(row["current_allocated_bytes"] for row in packet["lifecycle"].values())):
            packet["status"] = "invalid_capture"
        args.output.write_text(json.dumps(packet, indent=2) + "\n")
        return 0 if packet["status"] == "passed_supplemental" else 1


if __name__ == "__main__":
    raise SystemExit(main())
