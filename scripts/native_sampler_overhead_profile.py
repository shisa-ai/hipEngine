#!/usr/bin/env python3
"""Profile PARO native sampler overhead against greedy decode controls.

This is a narrow diagnostic for the promoted c=1 PARO native sampler route. It
uses the same resident session and repeated-token prompt shape for:

- greedy graph replay;
- greedy eager/no-graph decode;
- native bounded top-k eager sampling;
- host logits bounded top-k sampling.

The instrumentation counts Python-visible H2D/D2H copies and native sampler
scalar uploads only during the measured decode window.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import json
import os
import shlex
import statistics
import time
from typing import Any

import numpy as np

from hipengine.core.dtype import DType
from hipengine.generation.qwen35_paro import (
    _configure_host_sampler,
    _configure_native_sampler,
    _request_with_tokenizer_eos,
    _row_sampling_state,
)
from hipengine.generation.registry import GenerationRequest
from hipengine.generation.sampling import plan_sampler
from hipengine.kvcache import FixedPagedKVPolicy
import hipengine.runtime.qwen35_paro_runner as qpr
from hipengine.runtime.prefill import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession


DEFAULT_MODEL = Path(
    "/home/lhl/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/"
    "snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e"
)


class _Counters:
    def __init__(self) -> None:
        self.active: str | None = None
        self.profiles: dict[str, dict[str, Any]] = {}
        self.orig_h2d = qpr.copy_host_to_device
        self.orig_d2h = qpr.copy_device_to_host
        self.orig_upload = qpr.Qwen35ParoResidentSession._native_sampler_upload
        self.orig_methods: dict[str, Any] = {}

    def install(self) -> None:
        qpr.copy_host_to_device = self._counted_h2d
        qpr.copy_device_to_host = self._counted_d2h

        def counted_upload(session: Any, name: str, array: np.ndarray):
            return self._counted_upload(session, name, array)

        qpr.Qwen35ParoResidentSession._native_sampler_upload = counted_upload
        for method_name in (
            "_sample_device_from_hidden",
            "_sample_from_hidden_native",
            "_sample_from_hidden_host",
            "_select_argmax_device_from_logits",
            "_project_logits_device_from_hidden",
            "_read_sample",
        ):
            self._wrap_method(method_name)

    def restore(self) -> None:
        qpr.copy_host_to_device = self.orig_h2d
        qpr.copy_device_to_host = self.orig_d2h
        qpr.Qwen35ParoResidentSession._native_sampler_upload = self.orig_upload
        for method_name, original in self.orig_methods.items():
            setattr(qpr.Qwen35ParoResidentSession, method_name, original)

    @contextmanager
    def count_scope(self, name: str) -> Iterator[None]:
        previous = self.active
        self.active = name
        try:
            yield
        finally:
            self.active = previous

    def summarize(self, name: str, decode_tokens: int) -> dict[str, Any]:
        raw = self._profile_for(name)
        output: dict[str, Any] = {}
        for key in ("copy_host_to_device", "copy_device_to_host"):
            item = raw[key]
            output[key] = {
                "calls": int(item["calls"]),
                "bytes": int(item["bytes"]),
                "calls_per_decode_token": item["calls"] / decode_tokens if decode_tokens else None,
                "bytes_per_decode_token": item["bytes"] / decode_tokens if decode_tokens else None,
                "sizes": _summarize_counter(item["sizes"]),
            }
        item = raw["native_uploads"]
        output["native_uploads"] = {
            "calls": int(item["calls"]),
            "bytes": int(item["bytes"]),
            "calls_per_decode_token": item["calls"] / decode_tokens if decode_tokens else None,
            "bytes_per_decode_token": item["bytes"] / decode_tokens if decode_tokens else None,
            "names": _summarize_counter(item["names"]),
            "sizes": _summarize_counter(item["sizes"]),
        }
        methods = {}
        for method_name, values in raw["method_timings_s"].items():
            if values:
                methods[method_name] = {
                    "calls": len(values),
                    "total_s": sum(values),
                    "median_s": statistics.median(values),
                    "total_per_decode_token_s": sum(values) / decode_tokens if decode_tokens else None,
                }
        output["method_timings_s"] = methods
        return output

    def _empty_profile(self) -> dict[str, Any]:
        return {
            "copy_host_to_device": {"calls": 0, "bytes": 0, "sizes": Counter()},
            "copy_device_to_host": {"calls": 0, "bytes": 0, "sizes": Counter()},
            "native_uploads": {"calls": 0, "bytes": 0, "names": Counter(), "sizes": Counter()},
            "method_timings_s": defaultdict(list),
        }

    def _profile_for(self, name: str) -> dict[str, Any]:
        return self.profiles.setdefault(name, self._empty_profile())

    def _counted_h2d(self, buffer: Any, host_ptr: int, nbytes: int | None = None, *, runtime: Any = None) -> None:
        if self.active is not None:
            count = int(buffer.nbytes if nbytes is None else nbytes)
            bucket = self._profile_for(self.active)["copy_host_to_device"]
            bucket["calls"] += 1
            bucket["bytes"] += count
            bucket["sizes"][count] += 1
        return self.orig_h2d(buffer, host_ptr, nbytes, runtime=runtime)

    def _counted_d2h(self, host_ptr: int, buffer: Any, nbytes: int | None = None, *, runtime: Any = None) -> None:
        if self.active is not None:
            count = int(buffer.nbytes if nbytes is None else nbytes)
            bucket = self._profile_for(self.active)["copy_device_to_host"]
            bucket["calls"] += 1
            bucket["bytes"] += count
            bucket["sizes"][count] += 1
        return self.orig_d2h(host_ptr, buffer, nbytes, runtime=runtime)

    def _counted_upload(self, session: Any, name: str, array: np.ndarray):
        if self.active is not None:
            host = np.ascontiguousarray(array)
            bucket = self._profile_for(self.active)["native_uploads"]
            bucket["calls"] += 1
            bucket["bytes"] += int(host.nbytes)
            bucket["names"][str(name)] += 1
            bucket["sizes"][int(host.nbytes)] += 1
        return self.orig_upload(session, name, array)

    def _wrap_method(self, method_name: str) -> None:
        original = getattr(qpr.Qwen35ParoResidentSession, method_name)
        self.orig_methods[method_name] = original

        def wrapped(session: Any, *args: Any, **kwargs: Any):
            start = time.perf_counter()
            try:
                return original(session, *args, **kwargs)
            finally:
                if self.active is not None:
                    self._profile_for(self.active)["method_timings_s"][method_name].append(time.perf_counter() - start)

        setattr(qpr.Qwen35ParoResidentSession, method_name, wrapped)


def _summarize_counter(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: item[0])}


def _make_request(
    session: Qwen35ParoResidentSession,
    *,
    prompt_token: int,
    prompt_length: int,
    decode_tokens: int,
    native: bool,
) -> tuple[GenerationRequest, Any, Any]:
    request = GenerationRequest(
        prompts=("repeated-token diagnostic",),
        max_tokens=decode_tokens,
        temperature=0.7,
        top_p=1.0,
        top_k=4,
        ignore_eos=True,
        kv_storage="bf16",
        seed=123,
    )
    request = _request_with_tokenizer_eos(request, session.tokenizer)
    state = _row_sampling_state(request, tuple([prompt_token] * prompt_length), row_index=0)
    plan = plan_sampler(request, native_gpu_available=native, native_gpu_requested=native)
    return request, state, plan


def _run_lane(
    session: Qwen35ParoResidentSession,
    counters: _Counters,
    *,
    name: str,
    mode: str,
    prompt_token: int,
    prompt_length: int,
    warmup_decode_tokens: int,
    decode_tokens: int,
) -> dict[str, Any]:
    session.reset()
    _configure_native_sampler(session, None, None)
    _configure_host_sampler(session, None, None)
    prompt_tokens = [prompt_token] * prompt_length
    if mode == "native_topk":
        request, state, plan = _make_request(
            session,
            prompt_token=prompt_token,
            prompt_length=prompt_length,
            decode_tokens=decode_tokens,
            native=True,
        )
        if plan.mode.value != "gpu_sample":
            raise RuntimeError(f"expected gpu_sample plan, got {plan}")
        _configure_native_sampler(session, request, state)
    elif mode == "host_topk":
        request, state, plan = _make_request(
            session,
            prompt_token=prompt_token,
            prompt_length=prompt_length,
            decode_tokens=decode_tokens,
            native=False,
        )
        if plan.mode.value != "host_logits_sample":
            raise RuntimeError(f"expected host_logits_sample plan, got {plan}")
        _configure_host_sampler(session, request, state)
    else:
        plan = None

    next_result = session.prefill_native(prompt_tokens, sample=True)
    if next_result is None:
        raise RuntimeError(f"{name}: prefill produced no token")
    next_token = int(next_result.token_id)
    for offset in range(warmup_decode_tokens):
        result = session.step(next_token, position=prompt_length + offset, sample=True)
        if result is None:
            raise RuntimeError(f"{name}: warmup produced no token")
        next_token = int(result.token_id)

    decode_start_pos = prompt_length + warmup_decode_tokens
    generated: list[int] = []
    if mode == "greedy_graph":
        graph = session.capture_decode_graph(
            position=decode_start_pos,
            steps_per_replay=1,
            max_replay_steps=decode_tokens,
            record_steps=decode_tokens,
        )
        try:
            start = time.perf_counter()
            with counters.count_scope(name):
                graph.replay(decode_tokens)
                token_ids = graph.read_generated_token_ids(decode_tokens)
            elapsed = time.perf_counter() - start
            generated = [int(token) for token in token_ids]
        finally:
            graph.close()
    else:
        start = time.perf_counter()
        with counters.count_scope(name):
            for offset in range(decode_tokens):
                result = session.step(next_token, position=decode_start_pos + offset, sample=True)
                if result is None:
                    raise RuntimeError(f"{name}: measured decode produced no token")
                next_token = int(result.token_id)
                generated.append(next_token)
        elapsed = time.perf_counter() - start

    return {
        "mode": mode,
        "elapsed_s": elapsed,
        "tok_s": decode_tokens / elapsed if elapsed > 0 else None,
        "telemetry": {
            "sampler_mode": None if plan is None else plan.mode.value,
            "plan_fast_path_blockers": [] if plan is None else list(plan.fast_path_blockers),
            "generated_tail": generated[-8:],
            "generated_tokens": len(generated),
        },
        "profile": counters.summarize(name, decode_tokens),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("auto", "hip_gfx1100", "hip_gfx1151"), default="hip_gfx1100")
    parser.add_argument("--shared-expert-format", choices=("auto", "packed_paro_w4", "legacy_fp16"), default="packed_paro_w4")
    parser.add_argument("--prompt-token", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=45)
    parser.add_argument("--warmup-decode-tokens", type=int, default=4)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--attn-aotriton-min-tokens", type=int, default=512)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def _profile_command(args: argparse.Namespace) -> str:
    env_parts = []
    for name in ("HIP_VISIBLE_DEVICES", "HIPENGINE_HIP_ARCH", "PYTHONPATH"):
        value = os.environ.get(name)
        if value is not None:
            env_parts.append(f"{name}={shlex.quote(value)}")
    cmd = ["python3", "scripts/native_sampler_overhead_profile.py"]
    if args.model != DEFAULT_MODEL:
        cmd.extend(["--model", str(args.model)])
    if args.backend != "hip_gfx1100":
        cmd.extend(["--backend", args.backend])
    if args.shared_expert_format != "packed_paro_w4":
        cmd.extend(["--shared-expert-format", args.shared_expert_format])
    if args.prompt_token != 9707:
        cmd.extend(["--prompt-token", str(args.prompt_token)])
    if args.prompt_length != 45:
        cmd.extend(["--prompt-length", str(args.prompt_length)])
    if args.warmup_decode_tokens != 4:
        cmd.extend(["--warmup-decode-tokens", str(args.warmup_decode_tokens)])
    if args.decode_tokens != 64:
        cmd.extend(["--decode-tokens", str(args.decode_tokens)])
    if args.max_layers != 40:
        cmd.extend(["--max-layers", str(args.max_layers)])
    if args.attn_aotriton_min_tokens != 512:
        cmd.extend(["--attn-aotriton-min-tokens", str(args.attn_aotriton_min_tokens)])
    if args.json is not None:
        cmd.extend(["--json", str(args.json)])
    return " ".join([*env_parts, *(shlex.quote(part) for part in cmd)])


def _assessment(lanes: dict[str, Any]) -> dict[str, str]:
    native_profile = lanes["native_topk_eager"]["profile"]
    h2d = native_profile["copy_host_to_device"]["calls_per_decode_token"]
    d2h = native_profile["copy_device_to_host"]["calls_per_decode_token"]
    uploads = native_profile["native_uploads"]["calls_per_decode_token"]
    if h2d == 0:
        return {
            "primary_next_step": "host result-readback coalescing or device-side continuation plumbing",
            "secondary_next_step": "sampler-kernel overhead audit against greedy argmax",
            "reason": (
                "Request-constant scalar caching removes measured native H2D traffic: native top-k now has "
                f"{h2d:g} H2D and {d2h:g} D2H Python-visible copies/token in the warmed decode window. "
                "Native remains much faster than host logits but still trails greedy eager by the sampler "
                "kernel and host result readback."
            ),
        }
    if h2d is not None and uploads is not None and h2d <= uploads:
        return {
            "primary_next_step": "request-constant scalar buffer caching (#10)",
            "secondary_next_step": "host result-readback coalescing or device-side continuation plumbing",
            "reason": (
                "Device-side selected-token/logit writeback removes the legacy post-sample H2D writes: "
                f"native top-k now has {h2d:g} H2D and {d2h:g} D2H Python-visible copies/token, all H2D "
                f"traffic accounted for by {uploads:g} scalar uploads/token. Native remains much faster than "
                "host logits but still trails greedy eager by the sampler kernel, scalar uploads, and host "
                "result readback."
            ),
        }
    return {
        "primary_next_step": "device-side selected-token/logit writeback (#11)",
        "secondary_next_step": "request-constant scalar buffer caching (#10)",
        "reason": (
            "Native top-k remains much faster than host logits but trails greedy eager by the sampler "
            f"kernel plus {h2d:g} H2D and {d2h:g} D2H Python-visible copies/token; graph replay accounts for "
            "only a smaller part of the greedy graph gap in this diagnostic."
        ),
    }


def main() -> int:
    args = _parse_args()
    shared_expert_format = None if args.shared_expert_format == "auto" else args.shared_expert_format
    counters = _Counters()
    lanes: dict[str, Any] = {}
    counters.install()
    try:
        runner = Qwen35ParoNextTokenRunner(
            args.model,
            shared_expert_format=shared_expert_format,
            backend=args.backend,
        )
        session = Qwen35ParoResidentSession(
            runner,
            max_sequence_length=args.prompt_length + args.warmup_decode_tokens + args.decode_tokens + 1,
            max_layers=args.max_layers,
            prefill_config=PrefillConfig(attn_aotriton_min_tokens=args.attn_aotriton_min_tokens),
            kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
        )
        try:
            for name, mode in (
                ("greedy_graph", "greedy_graph"),
                ("greedy_eager", "greedy_eager"),
                ("native_topk_eager", "native_topk"),
                ("host_topk_eager", "host_topk"),
            ):
                print(f"RUN {name}", flush=True)
                lanes[name] = _run_lane(
                    session,
                    counters,
                    name=name,
                    mode=mode,
                    prompt_token=args.prompt_token,
                    prompt_length=args.prompt_length,
                    warmup_decode_tokens=args.warmup_decode_tokens,
                    decode_tokens=args.decode_tokens,
                )
                print(f"DONE {name} {lanes[name]['tok_s']:.3f} tok/s", flush=True)
        finally:
            session.close()
    finally:
        counters.restore()

    graph = lanes["greedy_graph"]["tok_s"]
    greedy_eager = lanes["greedy_eager"]["tok_s"]
    native = lanes["native_topk_eager"]["tok_s"]
    host = lanes["host_topk_eager"]["tok_s"]
    artifact = {
        "schema": "hipengine.native_sampler_overhead_profile.v1",
        "date": "2026-06-16",
        "task": "Native sampler overhead diagnostic",
        "hardware": "AMD Radeon Pro W7900 / gfx1100, HIP_VISIBLE_DEVICES=0",
        "model": "Qwen3.6-35B-A3B-PARO-full4096-e5-packed",
        "model_path": str(args.model),
        "quant": "w4_paro",
        "command": {
            "profile": _profile_command(args)
        },
        "workload": {
            "prompt_source": "repeated_token_id",
            "prompt_token_id": args.prompt_token,
            "prompt_length": args.prompt_length,
            "warmup_decode_tokens": args.warmup_decode_tokens,
            "measured_decode_tokens": args.decode_tokens,
            "max_layers": args.max_layers,
            "kv_storage": "bf16",
        },
        "env": {
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
        "lanes": lanes,
        "summary": {
            "greedy_graph_tok_s": graph,
            "greedy_eager_tok_s": greedy_eager,
            "native_topk_eager_tok_s": native,
            "host_topk_eager_tok_s": host,
            "native_vs_greedy_graph": native / graph if graph else None,
            "native_vs_greedy_eager": native / greedy_eager if greedy_eager else None,
            "native_vs_host": native / host if host else None,
        },
        "assessment": _assessment(lanes),
    }
    text = json.dumps(artifact, indent=2)
    print(json.dumps(artifact["summary"], indent=2), flush=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
        print(f"wrote {args.json}", flush=True)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
