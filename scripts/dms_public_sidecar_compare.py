#!/usr/bin/env python3
"""Matched public-library DMS diagnostics, not an arithmetic promotion gate.

One arm per process; use identical arguments except metadata/output. Reports
backend prefill and request-minus-prefill (inclusive) decode, not kernel time.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def select_prompts(path):
    """Freeze the first prompt in each category, preserving manifest order."""
    selected = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            selected.setdefault(row["category"], row)
    return list(selected.values())


def make_prompt(llm, row, length):
    content = "\n".join(message["content"] for message in row["messages"])
    start = llm.tokenize("<|im_start|>user\nBackground notes:\n")
    end = llm.tokenize("\nIgnore the background notes and answer this request:\n" + content +
                       "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    fill = llm.tokenize("This is background text about software documentation, experiments, and careful testing. ")
    count = length - len(start) - len(end)
    if count < 0:
        raise ValueError(f"prompt {row['id']} needs more than {length} tokens")
    return tuple(start) + (tuple(fill) * (count // len(fill) + 1))[:count] + tuple(end)


def timing_row(timing, prompt_tokens, output_tokens):
    prefill = float(timing["prefill_ms"])
    total = float(timing["request_total_ms"])
    remaining = total - prefill
    if not all(math.isfinite(x) and x > 0 for x in (prefill, total, remaining)):
        raise ValueError("invalid timing window")
    return {"prefill_ms": prefill, "request_total_ms": total,
            "post_prefill_inclusive_ms": remaining,
            "prefill_tokens_per_second": prompt_tokens * 1000 / prefill,
            "post_prefill_tokens_per_second": (output_tokens - 1) * 1000 / remaining}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--prompts", type=Path, default=ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl")
    parser.add_argument("--lengths", default="128,2048,12288")
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    if min(lengths) <= 0 or args.output_tokens < 2 or args.repetitions < 2:
        parser.error("positive lengths, >=2 output tokens and >=2 repetitions required")
    from hipengine import LLM, SamplingParams
    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.core.memory import memory_stats, reset_memory_stats

    result = {"schema": 1, "kind": "dms_public_sidecar_comparison", "status": "diagnostic",
              "arm": args.arm, "command": sys.argv, "rows": [],
              "protocol": {"surface": "LLM.generate_detailed", "profile": "production",
                           "codec": "int8_per_token_head", "scale_dtype": "fp32",
                           "prefix_cache": "off", "concurrency": 1, "speculative_mtp": False,
                           "warmups_per_prompt_shape": 1, "repetitions": args.repetitions,
                           "selection": "first prompt per category in manifest order",
                           "lengths": lengths, "output_tokens": args.output_tokens,
                           "timing": "backend request-minus-prefill includes sampling/scheduling; n-1 decode tokens",
                           "memory": "tracked allocator snapshots, not whole-device peak or dense-equivalent footprint",
                           "correctness": "same-arm exact greedy repeatability and route/length checks; not dense equivalence or task-quality certification"},
              "prompt_manifest_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest()}
    if args.metadata:
        result["sidecar_metadata"] = json.loads(args.metadata.read_text())
        result["sidecar_metadata_path"] = str(args.metadata.resolve())
    result["provenance"] = collect_artifact_provenance(
        repo_root=ROOT, configured_backend=args.backend, model_path=args.model,
        quant="gguf_q4_k_m", kv_dtype="int8_per_token_head", command=[sys.executable, *sys.argv],
        timing_protocol=result["protocol"]["timing"], warmups=1, repetitions=args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")

    llm = LLM(args.model, backend=args.backend, execution_profile="production",
              max_sequence_length=max(lengths) + args.output_tokens + 16,
              max_active_requests=1, prefix_cache="off", kv_storage="int8_per_token_head",
              kv_scale_dtype="fp32", dms_metadata_path=None if args.metadata is None else str(args.metadata),
              engine_command_timeout_seconds=1800)
    params = SamplingParams(max_tokens=args.output_tokens, temperature=0, ignore_eos=True)
    try:
        # Public tokenizer initialization registers the selected generator's plans.
        llm.tokenize("initialization outside measured requests")
        result["execution_profile"] = {
            "resolved": llm.resolved_execution_profile,
            "manifest": llm.execution_profile_manifest,
            "manifest_sha256": llm.execution_profile_manifest_sha256,
            "strict_manifest_sha256": llm.execution_profile_strict_manifest_sha256,
            "fell_back_to_strict": llm.execution_profile_fell_back_to_strict,
        }
        prompts = select_prompts(args.prompts)
        prompt_manifest = {"sequences": []}
        for length in lengths:
            for prompt in prompts:
                ids = make_prompt(llm, prompt, length)
                prompt_manifest["sequences"].append({
                    "sequence_id": f"{prompt['id']}-{length}", "category": prompt["category"],
                    "split": str(length), "token_ids": list(ids)})
                args.output.with_suffix(".prompts.json").write_text(json.dumps(prompt_manifest) + "\n")
                previous = None
                for iteration in range(-1, args.repetitions):
                    reset_memory_stats()
                    begin = time.perf_counter()
                    output = llm.generate_detailed([ids], params)[0]
                    elapsed = time.perf_counter() - begin
                    telemetry = dataclasses.asdict(output.telemetry)
                    generated = list(output.generated_token_ids)
                    observed_prompt_tokens = telemetry["decode_state"]["prompt_tokens"]
                    assert observed_prompt_tokens == length, (observed_prompt_tokens, length)
                    assert len(generated) == args.output_tokens, generated
                    diagnostics = telemetry["diagnostics"]
                    assert diagnostics["prefix_cache"]["reused_tokens"] == 0
                    if args.metadata:
                        dms = diagnostics["dms"]
                        assert dms["backend"]["device_payloads"]
                        assert dms["backend"]["window_size"] == result["sidecar_metadata"]["window_size"]
                        assert dms["backend"]["codec"] == "int8_per_token_head"
                    else:
                        assert not diagnostics.get("dms")
                    repeat_equal = previous is None or previous == generated
                    # Preserve a failing row rather than losing evidence to an assertion.
                    row = {"prompt_id": prompt["id"], "category": prompt["category"],
                           "prompt_tokens": length, "prompt_ids_sha256": digest(ids),
                           "iteration": iteration, "warmup": iteration == -1,
                           "wall_seconds": elapsed, "generated_token_ids": generated,
                           "text": output.text, "repeat_equal": repeat_equal,
                           "timing": timing_row(telemetry["timing"], length, len(generated)),
                           "telemetry": telemetry, "tracked_memory_after": memory_stats()}
                    result["rows"].append(row)
                    previous = generated
                    save()
                    print(json.dumps({"arm": args.arm, "prompt": prompt["id"], "length": length,
                                      "iteration": iteration, "repeat_equal": repeat_equal,
                                      "wall_seconds": elapsed}), flush=True)
        result["completed"] = True
        result["repeatability_passed"] = all(row["repeat_equal"] for row in result["rows"])
    except BaseException as error:
        result["completed"] = False
        result["error"] = repr(error)
        raise
    finally:
        llm.close()
        result["tracked_memory_after_close"] = memory_stats()
        save()
    return 0 if result["repeatability_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
