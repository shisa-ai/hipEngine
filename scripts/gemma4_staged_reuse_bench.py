#!/usr/bin/env python3
"""Compare staged attention libraries on identical BF16 buffers and HIP events.

Build and save the baseline before editing its source. Pass that shared object
with --baseline-library; the candidate defaults to the current staged build.
This measures all three attention stages, not whole-model generation.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import Gemma4AttentionScratch
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_staged as staged


def _bf16(values: np.ndarray) -> np.ndarray:
    bits = values.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _model_bench(args, paths, libraries, runtime):
    """Paired public generation; only the staged library changes between arms."""
    import time
    from scripts.gemma4_campaign_bench import (
        _provenance, _resolve_generator, exact_prompt_ids, _public_wall,
    )
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer
    prompts = [int(p) for p in args.prompts.split(",")]
    context = -(-(max(prompts) + 128) // 256) * 256
    llm, runner, info = _resolve_generator(args.model, context)
    print(f"loaded: {info}", flush=True)
    build = staged.build_gemma4_attention_staged
    rows = []
    arm_calls = {name: 0 for name in libraries}
    def run(name, prompt_ids, outputs=1):
        def select_library(**kwargs):
            arm_calls[name] += 1
            return libraries[name]
        staged.build_gemma4_attention_staged = select_library
        runtime.device_synchronize()
        started = time.perf_counter()
        result = _public_wall(llm, prompt_ids, outputs)
        runtime.device_synchronize()
        result["synchronized_wall_s"] = time.perf_counter() - started
        result["route"] = gemma4_layer.last_prefill_attention_route()
        assert result["route"] == "gemma4_staged", result
        assert result["generated_tokens"] == outputs, result
        return result
    try:
        for prompt in prompts:
            ids = exact_prompt_ids(llm._get_text_generator().tokenize, prompt)
            reference = None
            for name in libraries:
                for _ in range(args.warmups):
                    result = run(name, ids)
                    reference = result["generated_token_ids"] if reference is None else reference
                    assert result["generated_token_ids"] == reference
            samples = {name: [] for name in libraries}
            for repeat in range(args.repeats):
                order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
                for name in order:
                    result = run(name, ids)
                    reference = result["generated_token_ids"] if reference is None else reference
                    assert result["generated_token_ids"] == reference
                    samples[name].append(result)
                    print(f"prompt={prompt} repeat={repeat} arm={name} {result}", flush=True)
            medians = {name: statistics.median(r["synchronized_wall_s"] for r in records)
                       for name, records in samples.items()}
            rows.append(dict(prompt_tokens=prompt, output_tokens=1, context=context,
                             prompt_ids_sha256=hashlib.sha256(np.array(ids, dtype="<u4").tobytes()).hexdigest(),
                             samples=samples, median_s=medians,
                             speedup=medians["baseline"]/medians["candidate"], exact_ids=True))
        # No library override, no variant override: the shipped public path.
        staged.build_gemma4_attention_staged = build
        shipping_ids = exact_prompt_ids(llm._get_text_generator().tokenize, 512)
        shipping = _public_wall(llm, shipping_ids, 8)
        shipping["route"] = gemma4_layer.last_prefill_attention_route()
        assert shipping["route"] == "gemma4_staged"
        assert shipping["generated_tokens"] == 8
        assert all(arm_calls.values()), arm_calls
    finally:
        staged.build_gemma4_attention_staged = build
    return dict(kind="gemma4_staged_score_reuse_public_ab", performance_claim=False,
                command=" ".join(sys.argv), model=str(args.model), quant="UD-Q4_K_XL",
                physical_host="zbook", hardware="Radeon 8060S (gfx1151)",
                kv_dtype="bf16", execution_profile="production", loading=info,
                provenance=_provenance(args.model), warmups=args.warmups,
                repeats=args.repeats, timing_scope="public generation synchronized wall, one output token",
                libraries={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                           for name, path in paths.items()}, arm_library_calls=arm_calls,
                rows=rows, unmodified_default_request=shipping)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-library", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path)
    parser.add_argument("--mode", choices=("primitive", "model"), default="primitive")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prompts", default="512,8192,32768")
    parser.add_argument("--keys", default="1024,8192,32768")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 1 or args.repeats < 2 or args.repeats % 2 or args.warmups < 0:
        parser.error("tokens must be positive; repeats must be positive and even; warmups cannot be negative")
    keys_list = [int(k) for k in args.keys.split(",")]
    if not keys_list or min(keys_list) < args.tokens:
        parser.error("every key count must be at least tokens")
    if args.candidate_library is None:
        artifact = staged.build_gemma4_attention_staged(load=False)
        args.candidate_library = Path(artifact.output_path)
    paths = {"baseline": args.baseline_library, "candidate": args.candidate_library}
    libraries = {name: ctypes.CDLL(str(path.resolve())) for name, path in paths.items()}
    runtime = get_hip_runtime()
    if args.mode == "model":
        if args.model is None:
            parser.error("--model is required in model mode")
        payload = _model_bench(args, paths, libraries, runtime)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        return 0
    start, stop = runtime.event_create(), runtime.event_create()
    rows = []
    try:
        for head_dim, kv_heads, window in ((512, 2, 0), (256, 8, 1024)):
            for keys in keys_list:
                shape = (args.tokens, keys, 16, kv_heads, head_dim)
                rng = np.random.default_rng(71)
                query = _bf16(rng.standard_normal((args.tokens, 16, head_dim), dtype=np.float32) * .7)
                key = _bf16(rng.standard_normal((keys, kv_heads, head_dim), dtype=np.float32) * .7)
                value = _bf16(rng.standard_normal((keys, kv_heads, head_dim), dtype=np.float32) * .7)
                own = keys - args.tokens + np.arange(args.tokens)[:, None]
                indices = np.arange(keys)[None, :]
                mask = indices <= own
                if window:
                    mask &= indices > own - window
                mask = mask.astype(np.uint8)
                buffers = []
                scratch = Gemma4AttentionScratch()
                try:
                    for array in (query, key, value, mask, query):
                        buffer = malloc(array.nbytes)
                        buffers.append(buffer)
                        copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
                    def launch(name):
                        staged.gemma4_attention_staged_bf16(
                            *(buffer.ptr for buffer in buffers), tokens=args.tokens,
                            keys=keys, num_heads=16, num_kv_heads=kv_heads,
                            head_dim=head_dim, scale=.05, window=window,
                            row_offset=keys - args.tokens, library=libraries[name],
                            runtime=runtime, scratch=scratch,
                        )
                    outputs = {}
                    for name in libraries:
                        for _ in range(args.warmups):
                            launch(name)
                        launch(name)
                        output = np.empty_like(query)
                        copy_device_to_host(host_array_ptr(output), buffers[-1], output.nbytes)
                        outputs[name] = output
                    np.testing.assert_array_equal(outputs["baseline"], outputs["candidate"])
                    samples = {name: [] for name in libraries}
                    for repeat in range(args.repeats):
                        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
                        for name in order:
                            runtime.event_record(start)
                            launch(name)
                            runtime.event_record(stop)
                            runtime.event_synchronize(stop)
                            samples[name].append(runtime.event_elapsed_time_ms(start, stop))
                    medians = {name: statistics.median(values) for name, values in samples.items()}
                    row = dict(tokens=args.tokens, keys=keys, head_dim=head_dim,
                               num_heads=16, num_kv_heads=kv_heads, window=window,
                               row_offset=keys-args.tokens, scale=.05, samples_ms=samples,
                               median_ms=medians, speedup=medians["baseline"]/medians["candidate"],
                               bitwise_equal=True)
                    rows.append(row)
                    print(f"shape={shape} {medians} speedup={row['speedup']:.4f}", flush=True)
                finally:
                    scratch.close()
                    for buffer in buffers:
                        free(buffer)
    finally:
        runtime.event_destroy(start)
        runtime.event_destroy(stop)
    payload = dict(kind="gemma4_staged_attention_library_ab", performance_claim=False,
                   command=" ".join(sys.argv), dtype="bf16", warmups=args.warmups,
                   repeats=args.repeats, timing_scope="three attention stages, HIP events",
                   libraries={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                              for name, path in paths.items()}, rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
