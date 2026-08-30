#!/usr/bin/env python3
"""Collect Qwen4Exp wall time, ROCTX ranges, role markers, and runtime census.

This is the durable version of the fresh-profile harness used for the
2026-08-30 gfx1151 llama.cpp comparison. It deliberately keeps profiler-only
instrumentation out of dispatch: ROCTX ranges wrap existing runner entry points
and runtime calls are restored before the generator is closed.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROUTE_ENV_KEYS = (
    "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL",
    "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL",
    "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL",
    "HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS",
    "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL",
    "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS",
    "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS",
    "HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL",
    "HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS",
    "HIPENGINE_QWEN4_EXP_Q4_DP4A64",
    "HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS",
    "HIPENGINE_QWEN4_EXP_MOE_GRAPH",
)


class Roctx:
    """Minimal ROCTX range wrapper for rocprofiler-sdk's traced shim."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("librocprofiler-sdk-roctx.so.1")
        self.lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
        self.lib.roctxRangePushA.restype = ctypes.c_int
        self.lib.roctxRangePop.argtypes = []
        self.lib.roctxRangePop.restype = ctypes.c_int

    def push(self, text: str) -> None:
        self.lib.roctxRangePushA(text.encode("utf-8"))

    def pop(self) -> None:
        self.lib.roctxRangePop()


class RoleMarkers:
    """Profiler-only owner ranges for correlation-ID role attribution."""

    def __init__(self, module: Any, marker: Roctx) -> None:
        self.module = module
        self.marker = marker
        self.originals: dict[str, Any] = {}

    @staticmethod
    def _layer(weight: Any) -> str:
        return str(getattr(getattr(weight, "spec", None), "slot_path", "unknown"))

    def install(self) -> None:
        specs: dict[str, Callable[[tuple[Any, ...], dict[str, Any]], str]] = {
            "launch_gguf_linear": lambda a, _k: "linear:" + self._layer(a[0]),
            "run_qwen4_exp_gr_read": lambda a, _k: "gr_read:" + self._layer(a[2]),
            "run_qwen4_exp_moe": lambda a, _k: "moe:" + self._layer(a[1]["expert_gate"]),
            "run_qwen4_exp_gdn_token_mixer": lambda a, _k: "gdn:" + self._layer(a[1]["attn_qkv"]),
            "run_qwen4_exp_qsa_prefill_token_mixer": lambda a, _k: "qsa_prefill:" + self._layer(a[1].projections["attn_q"]),
            "run_qwen4_exp_dense_qsa_token_mixer": lambda a, _k: "qsa_decode:" + self._layer(a[1].projections["attn_q"]),
            "run_qwen4_exp_ple": lambda a, _k: "ple:" + self._layer(a[2]["ple_key"]),
        }
        for name, role_fn in specs.items():
            original = getattr(self.module, name)
            self.originals[name] = original

            def wrapper(
                *args: Any,
                _original: Any = original,
                _role_fn: Callable[[tuple[Any, ...], dict[str, Any]], str] = role_fn,
                **kwargs: Any,
            ) -> Any:
                self.marker.push("qwen4exp_role:" + _role_fn(args, kwargs))
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.marker.pop()

            setattr(self.module, name, wrapper)

    def close(self) -> None:
        for name, original in self.originals.items():
            setattr(self.module, name, original)
        self.originals.clear()


class RuntimeCensus:
    """Count Python-visible runtime operations in a measured window.

    Compiled launch wrappers call HIP directly and therefore appear only in the
    rocprof HIP API census. This census intentionally exposes that boundary.
    """

    METHODS = (
        "memcpy",
        "memcpy_async",
        "memset",
        "memset_async",
        "device_synchronize",
        "stream_synchronize",
        "graph_launch",
    )

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.counts: Counter[str] = Counter()
        self.bytes: Counter[str] = Counter()
        self.sizes: dict[str, Counter[int]] = defaultdict(Counter)
        self.kinds: dict[str, Counter[str]] = defaultdict(Counter)
        self.size_kinds: dict[str, Counter[tuple[int, str]]] = defaultdict(Counter)
        self.originals: dict[str, Any] = {}

    def install(self) -> None:
        for name in self.METHODS:
            original = getattr(self.runtime, name)
            self.originals[name] = original

            def wrapper(*args: Any, _name: str = name, _original: Any = original, **kwargs: Any) -> Any:
                self.counts[_name] += 1
                if _name.startswith("memcpy") and len(args) >= 3:
                    nbytes = int(args[2])
                    self.bytes[_name] += nbytes
                    self.sizes[_name][nbytes] += 1
                    if len(args) >= 4:
                        kind = str(int(args[3]))
                        self.kinds[_name][kind] += 1
                        self.size_kinds[_name][(nbytes, kind)] += 1
                elif _name.startswith("memset") and len(args) >= 3:
                    nbytes = int(args[2])
                    self.bytes[_name] += nbytes
                    self.sizes[_name][nbytes] += 1
                return _original(*args, **kwargs)

            setattr(self.runtime, name, wrapper)

    def close(self) -> None:
        for name, original in self.originals.items():
            setattr(self.runtime, name, original)
        self.originals.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": dict(self.counts),
            "bytes": dict(self.bytes),
            "sizes": {
                name: {str(size): count for size, count in counts.most_common()}
                for name, counts in self.sizes.items()
            },
            "memcpy_kinds": {name: dict(counts) for name, counts in self.kinds.items()},
            "memcpy_size_kinds": {
                name: {
                    f"{size}:{kind}": count
                    for (size, kind), count in counts.most_common()
                }
                for name, counts in self.size_kinds.items()
            },
        }


def _graph_snapshot(cache: Any, runtime: Any) -> dict[str, Any]:
    if cache is None:
        return {"enabled": False, "stats": {}, "graphs": 0, "nodes": 0, "node_types": {}}
    node_types: Counter[str] = Counter()
    nodes = 0
    graphs = getattr(cache, "_graphs", {})
    for graph in graphs.values():
        for node in runtime.graph_nodes(int(graph)):
            nodes += 1
            node_types[str(runtime.graph_node_type(int(node)))] += 1
    return {
        "enabled": bool(cache.enabled),
        "stats": cache.stats,
        "graphs": len(graphs),
        "nodes": nodes,
        "node_types": dict(node_types),
    }


def _wall_summary(mode: str, prompt_tokens: int, walls: list[float]) -> dict[str, float | int]:
    total = sum(walls)
    return {
        "count": len(walls),
        "sum": total,
        "mean": statistics.mean(walls),
        "median": statistics.median(walls),
        "min": min(walls),
        "max": max(walls),
        "tok_s": (prompt_tokens * len(walls) / total) if mode == "prefill" else (len(walls) / total),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True, help="Directory containing the split GGUF parts")
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--prompt-file", type=Path, help="Prompt text for prefill mode")
    parser.add_argument("--expected-prompt-tokens", type=int, help="Optional token-count assertion for the prompt")
    parser.add_argument("--profile", action="store_true", help="Emit ROCTX measurement ranges")
    parser.add_argument("--role-markers", action="store_true", help="Emit profiler-only qwen4exp_role:* ranges")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--warm-decode-steps", type=int, default=8)
    parser.add_argument("--warm-trajectory-repetitions", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, help="Defaults to 768 for prefill and 128 for decode")
    parser.add_argument("--prefill-chunk-size", type=int, help="Defaults to 512 for prefill and 256 for decode")
    parser.add_argument("--hip-arch", default="gfx1151")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "prefill" and args.prompt_file is None:
        raise SystemExit("--prompt-file is required in prefill mode")
    max_sequence_length = args.max_sequence_length or (768 if args.mode == "prefill" else 128)
    prefill_chunk_size = args.prefill_chunk_size or (512 if args.mode == "prefill" else 256)

    os.environ.setdefault("HIPENGINE_HIP_ARCH", args.hip_arch)
    if args.compiler_version_file is not None:
        os.environ.setdefault("HIPENGINE_COMPILER_VERSION_FILE", str(args.compiler_version_file))
    if args.require_cached_build:
        os.environ.setdefault("HIPENGINE_REQUIRE_CACHED_BUILD", "1")

    from hipengine.core.memory import memory_stats
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model
    import hipengine.runtime.qwen4_exp_runner as runner_module

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile.PRODUCTION,
    )

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=args.model_root,
            weight_index=index,
            model_plugin=plugin,
            backend="hip_gfx1151",
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=prefill_chunk_size,
        )

    roctx = Roctx() if args.profile else None
    report: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_profile_gap_window",
        "mode": args.mode,
        "profile": bool(args.profile),
        "model_root": str(args.model_root),
        "prompt_file": str(args.prompt_file) if args.prompt_file else None,
        "manifest_sha256": resolved.manifest_sha256,
        "strict_manifest_sha256": resolved.strict_manifest_sha256,
        "fell_back_to_strict": resolved.fell_back_to_strict,
        "route_env": {},
        "wall_seconds": [],
    }
    generator = resolved.construct_generator(factory)
    try:
        runner = generator.runner
        report["route_env"] = {key: os.environ.get(key) for key in ROUTE_ENV_KEYS}
        if args.mode == "prefill":
            ids = generator.tokenizer.encode(args.prompt_file.read_text())
            if args.expected_prompt_tokens is not None and len(ids) != args.expected_prompt_tokens:
                raise RuntimeError(
                    f"expected {args.expected_prompt_tokens} prompt tokens, got {len(ids)}"
                )
            runner.prefill(ids)
            runner.runtime.device_synchronize()
            census = RuntimeCensus(runner.runtime)
            census.install()
            roles = RoleMarkers(runner_module, roctx) if args.role_markers and roctx else None
            if roles is not None:
                roles.install()
            try:
                for rep in range(args.repetitions):
                    if roctx:
                        roctx.push(f"qwen4exp_prefill_p{len(ids)}_{rep}")
                    started = time.perf_counter()
                    result = runner.prefill(ids)
                    runner.runtime.device_synchronize()
                    report["wall_seconds"].append(time.perf_counter() - started)
                    if roctx:
                        roctx.pop()
                report["token_id"] = int(result.token_id)
                report["logits_sha256"] = hashlib.sha256(result.logits.tobytes()).hexdigest()
                report["runtime_census"] = census.snapshot()
            finally:
                if roles is not None:
                    roles.close()
                census.close()
            prompt_tokens = len(ids)
        else:
            for _ in range(args.warm_trajectory_repetitions):
                result = runner.prefill([9707])
                for _ in range(args.warm_decode_steps + args.decode_steps):
                    result = runner.step(int(result.token_id))
                runner.runtime.device_synchronize()
            result = runner.prefill([9707])
            for _ in range(args.warm_decode_steps):
                result = runner.step(int(result.token_id))
            runner.runtime.device_synchronize()
            report["graph_before"] = _graph_snapshot(runner.moe_graph_cache, runner.runtime)
            census = RuntimeCensus(runner.runtime)
            census.install()
            roles = RoleMarkers(runner_module, roctx) if args.role_markers and roctx else None
            if roles is not None:
                roles.install()
            try:
                if roctx:
                    roctx.push("qwen4exp_decode_window")
                for step in range(args.decode_steps):
                    if roctx:
                        roctx.push(f"qwen4exp_decode_step_{step}")
                    started = time.perf_counter()
                    result = runner.step(int(result.token_id))
                    report["wall_seconds"].append(time.perf_counter() - started)
                    if roctx:
                        roctx.pop()
                runner.runtime.device_synchronize()
                if roctx:
                    roctx.pop()
                report["token_id"] = int(result.token_id)
                report["logits_sha256"] = hashlib.sha256(result.logits.tobytes()).hexdigest()
                report["runtime_census"] = census.snapshot()
            finally:
                if roles is not None:
                    roles.close()
                census.close()
            report["graph_after"] = _graph_snapshot(runner.moe_graph_cache, runner.runtime)
            prompt_tokens = 0
        report["wall_summary"] = _wall_summary(args.mode, prompt_tokens, report["wall_seconds"])
    finally:
        generator.close()
    report["memory_after_close"] = memory_stats()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
