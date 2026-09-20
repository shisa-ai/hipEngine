#!/usr/bin/env python3
"""Attribute GGUF AR decode costs without changing the sampling distribution.

Runs the resident service (no HTTP/MTP/cache) on the full category suite and
heldouts by default. Timings are instrumented diagnostics, not a promotion gate.
Method timings are inclusive; a blocking D2H includes outstanding GPU work.
The first two decode transitions are excluded from steady-state summaries.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

from hipengine import LLM, SamplingParams
from hipengine.generation import qwen35_gguf as generation
from hipengine.generation import sampling
from hipengine.runtime import native_sampler, qwen35_gguf_runner as runtime
from hipengine.runtime.gguf_decode_graph import Qwen35GGUFDecodeGraph


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("greedy_default", "greedy_eager", "host_sampled", "native_sampled")


class Trace:
    def __init__(self):
        self.current = None
        self.transitions = []

    def install(self, stack):
        original = generation.Qwen35GGUFResidentModelRunner._step_native_rows

        def transition(owner, rows, **kwargs):
            if self.current is not None:
                raise RuntimeError("nested decode transition")
            self.current = {"seconds": defaultdict(float), "calls": defaultdict(int),
                            "d2h_bytes": defaultdict(int)}
            start = time.perf_counter()
            try:
                return original(owner, rows, **kwargs)
            finally:
                self.current["wall_s"] = time.perf_counter() - start
                self.transitions.append(self.current)
                self.current = None

        stack.enter_context(patch.object(
            generation.Qwen35GGUFResidentModelRunner, "_step_native_rows", transition))
        for obj, method, label in (
            (runtime.Qwen35GGUFResidentSession, "step", "model_step_including_readback"),
            (runtime.Qwen35GGUFResidentSession, "_read_sample", "readback_including_sync"),
            (generation, "_select_from_gguf_logits", "host_selection"),
            (sampling, "_top_k_candidate_ids", "host_vocab_sort"),
            (native_sampler.NativeSamplerWorkspace, "sample", "native_selection"),
            (native_sampler.NativeSamplerWorkspace, "_synchronize", "native_kernel_wait"),
            (Qwen35GGUFDecodeGraph, "replay", "graph_replay"),
            (runtime.Qwen35GGUFResidentSession, "capture_decode_graph", "graph_capture"),
        ):
            self.wrap(stack, obj, method, label)
        for module, label in ((runtime, "model_d2h"), (native_sampler, "sampler_d2h")):
            self.wrap(stack, module, "copy_device_to_host", label, copy=True)

    def wrap(self, stack, obj, method, label, *, copy=False):
        original = getattr(obj, method)

        def wrapped(*args, **kwargs):
            active = self.current
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if active is not None:
                    active["seconds"][label] += time.perf_counter() - start
                    active["calls"][label] += 1
                    if copy:
                        count = kwargs.get("nbytes", args[2] if len(args) > 2 else None)
                        count = int(args[1].nbytes if count is None else count)
                        active["d2h_bytes"][label] += count
                        size_label = label + ("_large" if count > 64 else "_scalar")
                        active["seconds"][size_label] += time.perf_counter() - start

        stack.enter_context(patch.object(obj, method, wrapped))


def summarize(transitions):
    steady = transitions[2:]
    if not steady:
        raise ValueError("need at least three measured decode transitions")
    totals = {kind: defaultdict(float) for kind in ("seconds", "calls", "d2h_bytes")}
    for step in steady:
        for kind, values in totals.items():
            for key, value in step[kind].items():
                values[key] += value
    return {
        "transitions": len(transitions), "steady_transitions": len(steady),
        "steady_wall_s": sum(step["wall_s"] for step in steady),
        "median_transition_ms": statistics.median(step["wall_s"] for step in steady) * 1000,
        "steady_totals": totals,
        "all_calls": {key: sum(step["calls"].get(key, 0) for step in transitions)
                      for key in {key for step in transitions for key in step["calls"]}},
    }


def trajectory_checks(rows):
    groups = defaultdict(list)
    greedy = defaultdict(list)
    for row in rows:
        key = (row["prompt_id"], row["prompt_tokens"])
        groups[(*key, row["arm"])].append(row["generated_token_ids"])
        if row["arm"].startswith("greedy"):
            greedy[key].append(row["generated_token_ids"])
    return {
        "fixed_seed_repeats_exact": all(all(ids == group[0] for ids in group) for group in groups.values()),
        "repeated_groups": sum(len(group) > 1 for group in groups.values()),
        "greedy_default_eager_exact": all(all(ids == group[0] for ids in group) for group in greedy.values()),
        "note": "control/repeatability only; no host/native RNG equality or task-quality claim",
    }


def run(args):
    suite = []
    for source in args.prompts:
        for line in source.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["source"] = str(source)
                suite.append(row)
    if args.prompt_id:
        suite = [row for row in suite if row["id"] in args.prompt_id]
    if not suite:
        raise ValueError("no prompts selected")
    if any(context < 0 for context in args.context_tokens):
        raise ValueError("context tokens must be non-negative")
    if args.max_tokens < 5 or args.repeats < 1:
        raise ValueError("max_tokens must be >=5 and repeats >=1")
    env_keys = ("HIPENGINE_QWEN35_NATIVE_SAMPLER", "HIPENGINE_GGUF_DECODE_GRAPH")
    original_env = {key: os.environ.get(key) for key in env_keys}
    trace = Trace()
    payload = {
        "schema": "hipengine.gguf_sampling_overhead.v1", "performance_claim": False,
        "purpose": "AR attribution; not a quality or default-promotion gate",
        "host": platform.node(), "model": str(args.model.resolve()),
        "backend": args.backend, "quant": args.quant, "kv_storage": "bf16",
        "execution_profile": "strict", "capacity": 4,
        "max_sequence_length": args.max_sequence_length,
        "mtp": False, "prefix_cache": "off", "seed": 17,
        "command": shlex.join([sys.executable, *sys.argv]),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {k: v for k, v in os.environ.items() if k.startswith("HIPENGINE_")},
        "timing_note": "inclusive method walls, no extra synchronization; first two transitions excluded",
        "suite_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.prompts},
        "rows": [],
    }
    llm = LLM(str(args.model), backend=args.backend, quant=args.quant,
              max_active_requests=4, max_sequence_length=args.max_sequence_length,
              kv_storage="bf16", prefix_cache="off", execution_profile="strict")
    try:
        llm.prepare(max_sequence_length=args.max_sequence_length)
        payload["variant_manifest_sha256"] = llm.execution_profile_manifest_sha256
        payload["variant_manifest"] = llm.execution_profile_manifest
        payload["hipcc_version"] = subprocess.check_output(["hipcc", "--version"], text=True)
        payload["hardware"] = [line.strip() for line in subprocess.check_output(
            ["rocminfo"], text=True).splitlines() if "Name:" in line]
        print("prepared", flush=True)
        # Warm the ordinary AR route; per-arm first transitions remain excluded.
        llm.generate_detailed([[9707] * 32], SamplingParams(max_tokens=5, ignore_eos=True))
        with ExitStack() as stack:
            trace.install(stack)
            for prompt in suite:
                text = "\n".join(message["content"] for message in prompt["messages"])
                base_ids = llm.tokenize(text)
                for context in args.context_tokens:
                    ids = base_ids if context == 0 else (base_ids * ((context + len(base_ids) - 1) // len(base_ids)))[:context]
                    if len(ids) + args.max_tokens > args.max_sequence_length:
                        raise ValueError("context plus output exceeds max_sequence_length")
                    for repeat in range(args.repeats):
                        arms = args.arms if repeat % 2 == 0 else list(reversed(args.arms))
                        for arm in arms:
                            os.environ[env_keys[0]] = "1" if arm == "native_sampled" else "0"
                            os.environ[env_keys[1]] = "0" if arm == "greedy_eager" else "1"
                            sampled = arm.endswith("sampled")
                            params = SamplingParams(max_tokens=args.max_tokens,
                                                    temperature=0.7 if sampled else 0.0,
                                                    top_p=0.95 if sampled else 1.0,
                                                    seed=17, ignore_eos=True)
                            trace.transitions = []
                            start = time.perf_counter()
                            output = llm.generate_detailed([ids], params)[0]
                            wall = time.perf_counter() - start
                            generated = list(output.generated_token_ids or ())
                            if len(generated) != args.max_tokens:
                                raise RuntimeError("incomplete generation")
                            if len(trace.transitions) != len(generated) - 1:
                                raise RuntimeError("unexpected AR transition count")
                            telemetry = output.telemetry.to_json_dict()
                            row = {
                                "prompt_id": prompt["id"], "category": prompt["category"],
                                "source": prompt["source"], "prompt_tokens": len(ids),
                                "prompt_ids_sha256": hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest(),
                                "arm": arm, "repeat": repeat, "request_wall_s": wall,
                                "generated_token_ids": generated, "telemetry": telemetry,
                                "profile": summarize(trace.transitions),
                            }
                            decode = telemetry["decode_state"]
                            expected_mode = "gpu_sample" if arm == "native_sampled" else "host_logits_sample"
                            if sampled and decode.get("sampler_mode") != expected_mode:
                                raise RuntimeError(f"unexpected sampler route: {decode}")
                            if sampled and decode.get("full_vocab_logits_d2h") is not (arm == "host_sampled"):
                                raise RuntimeError("unexpected logits transfer route")
                            payload["rows"].append(row)
                            args.json.write_text(json.dumps(payload, indent=2) + "\n")
                            print(json.dumps({"prompt": prompt["id"], "context": len(ids),
                                              "arm": arm, "repeat": repeat,
                                              "profile": row["profile"]}), flush=True)
        payload["checks"] = trajectory_checks(payload["rows"])
        payload["complete"] = True
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        if not payload["checks"]["fixed_seed_repeats_exact"] or not payload["checks"]["greedy_default_eager_exact"]:
            raise RuntimeError("trajectory repeatability failed; see artifact")
    finally:
        llm.close()
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--prompts", type=Path, nargs="+", default=[
        ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl"])
    parser.add_argument("--prompt-id", nargs="+")
    parser.add_argument("--context-tokens", type=int, nargs="+", default=[0], help="0: natural length; otherwise repeat source tokens to shape")
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--json", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
