#!/usr/bin/env python3
"""Decode HIP-graph replay correctness gate for Qwen3.5/PARO.

This gate runs the same native prefill twice from a fixture prompt, then compares
normal eager token-by-token decode against the reusable HIP graph replay path.
The graph records generated token ids on device so the full decode sequence can
be checked without host interaction inside the replayed step.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.kvcache import ResolvedKVPolicy
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_kv_policy_args import add_kv_policy_args, append_kv_policy_flags, kv_policy_json, resolve_args_kv_policy

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)
DEFAULT_FIXTURE = "fixtures/qwen35_paro/parent_512_32_seed1234.json"


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text(encoding="utf-8")


def _load_fixture(path: Path) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    required = {"prompt_ids", "expected_generated_token_ids", "decode_len"}
    missing = required.difference(fixture)
    if missing:
        raise ValueError(f"fixture missing required keys: {sorted(missing)}")
    return fixture


def _read_logits(session: Qwen35ParoResidentSession) -> np.ndarray:
    logits = np.empty((session.vocab_size,), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(logits),
        DeviceBuffer(session.lm_logits.ptr, logits.nbytes),
        runtime=session.runtime,
    )
    return logits


def _result_dict(result) -> dict[str, Any]:
    return {
        "token_id": int(result.token_id),
        "token_text": result.token_text,
        "logit": float(result.logit),
    }


def _run_once(
    runner: Qwen35ParoNextTokenRunner,
    prompt_tokens: list[int],
    *,
    decode_tokens: int,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
    prefill_config: PrefillConfig,
    decode_mode: str,
    graph_steps_per_replay: int,
    kv_policy: ResolvedKVPolicy,
) -> dict[str, Any]:
    max_sequence = len(prompt_tokens) + decode_tokens + 2
    generated_ids: list[int]
    generated: list[dict[str, Any]]
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_layers=max_layers,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        prefill_config=prefill_config,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    ) as session:
        prefill_start = time.perf_counter()
        seed = session.prefill_native(prompt_tokens, sample=True)
        prefill_seconds = time.perf_counter() - prefill_start
        if seed is None:
            raise RuntimeError("native prefill did not produce a seed token")
        decode_start = time.perf_counter()
        if decode_mode == "eager":
            current = seed
            generated = []
            for offset in range(decode_tokens):
                current = session.step(current.token_id, position=len(prompt_tokens) + offset)
                if current is None:
                    raise RuntimeError(f"eager decode did not produce token {offset}")
                generated.append(_result_dict(current))
            generated_ids = [int(item["token_id"]) for item in generated]
            final = current
        elif decode_mode == "graph":
            graph = session.capture_decode_graph(
                position=len(prompt_tokens),
                steps_per_replay=graph_steps_per_replay,
                max_replay_steps=decode_tokens,
                record_steps=decode_tokens,
            )
            try:
                graph.replay(decode_tokens)
                generated_ids = graph.read_generated_token_ids(decode_tokens)
                final = graph.read_sample()
                generated = [
                    {"token_id": int(token_id), "token_text": "", "logit": None}
                    for token_id in generated_ids
                ]
                if generated:
                    generated[-1] = _result_dict(final)
            finally:
                graph.close()
        else:
            raise ValueError(f"unsupported decode mode {decode_mode!r}")
        decode_seconds = time.perf_counter() - decode_start
        final_logits = _read_logits(session)
        detail = getattr(session, "last_prefill_execution", None)
        resolved_chunk_sizes = {
            "linear": int(session.prefill_config.linear_chunk_size),
            "moe": int(session.prefill_config.moe_chunk_size),
            "full_attn_query": int(session.prefill_config.full_attn_query_chunk_size),
            "full_attn_post": int(session.prefill_config.full_attn_post_chunk_size),
            "full_attn_rope": int(session.prefill_config.full_attn_rope_chunk_size),
        }
        chunk_tuning = getattr(session, "prefill_chunk_tuning", None)
    return {
        "decode_mode": decode_mode,
        "seed": _result_dict(seed),
        "generated": generated,
        "generated_token_ids": generated_ids,
        "final": _result_dict(final),
        "final_logits": final_logits,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "prefill_execution_detail": detail,
        "prefill_chunk_sizes": resolved_chunk_sizes,
        "prefill_chunk_tuning": chunk_tuning,
    }


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    max_v = float(np.max(x))
    shifted = x - max_v
    log_denom = max_v + math.log(float(np.sum(np.exp(shifted))))
    return x - log_denom


def _kl_divergence(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    log_p = _log_softmax(reference_logits)
    log_q = _log_softmax(candidate_logits)
    p = np.exp(log_p)
    return float(np.sum(p * (log_p - log_q)))


def _strip_logits(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key != "final_logits"}


def _first_mismatch(a: list[int], b: list[int]) -> dict[str, int] | None:
    for idx, (left, right) in enumerate(zip(a, b, strict=False)):
        if left != right:
            return {"index": idx, "left": int(left), "right": int(right)}
    if len(a) != len(b):
        return {"index": min(len(a), len(b)), "left": len(a), "right": len(b)}
    return None


def _command(args: argparse.Namespace) -> str:
    command = f"python3 scripts/qwen35_decode_graph_fixture_gate.py --model {args.model} --fixture {args.fixture}"
    command += f" --max-layers {args.max_layers} --graph-steps-per-replay {args.graph_steps_per_replay}"
    if args.max_new_tokens is not None:
        command += f" --max-new-tokens {args.max_new_tokens}"
    if args.compiler_version_file is not None:
        command += f" --compiler-version-file {args.compiler_version_file}"
    if args.require_cached_build:
        command += " --require-cached-build"
    if args.attn_aotriton_min_tokens:
        command += f" --attn-aotriton-min-tokens {args.attn_aotriton_min_tokens}"
    for flag, attr in (
        ("--prefill-linear-chunk-size", "prefill_linear_chunk_size"),
        ("--prefill-moe-chunk-size", "prefill_moe_chunk_size"),
        ("--prefill-full-attn-query-chunk-size", "prefill_full_attn_query_chunk_size"),
        ("--prefill-full-attn-post-chunk-size", "prefill_full_attn_post_chunk_size"),
        ("--prefill-full-attn-rope-chunk-size", "prefill_full_attn_rope_chunk_size"),
    ):
        value = int(getattr(args, attr, 0))
        if value:
            command += f" {flag} {value}"
    if not getattr(args, "prefill_chunk_autotune", True):
        command += " --no-prefill-chunk-autotune"
    if getattr(args, "prefill_chunk_memory_budget_gib", 0.0):
        command += f" --prefill-chunk-memory-budget-gib {args.prefill_chunk_memory_budget_gib}"
    if args.kl_threshold != 0.05:
        command += f" --kl-threshold {args.kl_threshold}"
    command = append_kv_policy_flags(command, args)
    if args.json is not None:
        command += f" --json {args.json}"
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture = _load_fixture(args.fixture)
    prompt_tokens = [int(item) for item in fixture["prompt_ids"]]
    decode_tokens = int(fixture["decode_len"] if args.max_new_tokens is None else args.max_new_tokens)
    expected = [int(item) for item in fixture["expected_generated_token_ids"][:decode_tokens]]
    if decode_tokens <= 0:
        raise ValueError("decode_tokens must be positive")
    if decode_tokens % args.graph_steps_per_replay != 0:
        raise ValueError("decode_tokens must be divisible by graph_steps_per_replay")
    compiler_version = _read_compiler_version(args.compiler_version_file)
    runner = Qwen35ParoNextTokenRunner(args.model)
    kv_policy = resolve_args_kv_policy(args, block_size=256)
    prefill_config = PrefillConfig(
        linear_chunk_size=args.prefill_linear_chunk_size,
        moe_chunk_size=args.prefill_moe_chunk_size,
        full_attn_query_chunk_size=args.prefill_full_attn_query_chunk_size,
        full_attn_post_chunk_size=args.prefill_full_attn_post_chunk_size,
        full_attn_rope_chunk_size=args.prefill_full_attn_rope_chunk_size,
        attn_aotriton_min_tokens=args.attn_aotriton_min_tokens,
        auto_tune_chunk_sizes=args.prefill_chunk_autotune,
        chunk_tune_memory_budget_gib=args.prefill_chunk_memory_budget_gib,
    )
    eager = _run_once(
        runner,
        prompt_tokens,
        decode_tokens=decode_tokens,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        prefill_config=prefill_config,
        decode_mode="eager",
        graph_steps_per_replay=args.graph_steps_per_replay,
        kv_policy=kv_policy,
    )
    graph = _run_once(
        runner,
        prompt_tokens,
        decode_tokens=decode_tokens,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        prefill_config=prefill_config,
        decode_mode="graph",
        graph_steps_per_replay=args.graph_steps_per_replay,
        kv_policy=kv_policy,
    )
    eager_ids = [int(item) for item in eager["generated_token_ids"]]
    graph_ids = [int(item) for item in graph["generated_token_ids"]]
    seed_match = int(eager["seed"]["token_id"]) == int(graph["seed"]["token_id"])
    generated_match = eager_ids == graph_ids
    expected_match = graph_ids == expected
    final_sample_match = int(eager["final"]["token_id"]) == int(graph["final"]["token_id"])
    final_kl = _kl_divergence(eager["final_logits"], graph["final_logits"])
    final_top1_match = int(np.argmax(eager["final_logits"])) == int(np.argmax(graph["final_logits"]))
    finite_logits = bool(np.isfinite(eager["final_logits"]).all() and np.isfinite(graph["final_logits"]).all())
    kl_pass = final_kl <= float(args.kl_threshold)
    passed = bool(seed_match and generated_match and expected_match and final_sample_match and final_top1_match and finite_logits and kl_pass)
    return {
        "schema": 1,
        "status": "accepted" if passed else "rejected_correctness",
        "blocked_reason": None if passed else "decode graph replay fixture gate failed",
        "model": str(Path(args.model)),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": "qwen35_paro_decode_graph_fixture_gate",
        "command": _command(args),
        "performance_claim": False,
        "fixture": args.fixture.as_posix(),
        "prompt_length": len(prompt_tokens),
        "decode_tokens": decode_tokens,
        "max_layers": int(args.max_layers),
        "attn_aotriton_min_tokens": int(args.attn_aotriton_min_tokens),
        "kv_storage_dtype": kv_policy.storage_dtype.value,
        "kv_policy": kv_policy_json(kv_policy),
        "requested_prefill_chunk_sizes": {
            "linear": int(args.prefill_linear_chunk_size),
            "moe": int(args.prefill_moe_chunk_size),
            "full_attn_query": int(args.prefill_full_attn_query_chunk_size),
            "full_attn_post": int(args.prefill_full_attn_post_chunk_size),
            "full_attn_rope": int(args.prefill_full_attn_rope_chunk_size),
        },
        "prefill_chunk_autotune": bool(args.prefill_chunk_autotune),
        "prefill_chunk_memory_budget_gib": float(args.prefill_chunk_memory_budget_gib),
        "prefill_chunk_sizes": graph["prefill_chunk_sizes"],
        "prefill_chunk_tuning": graph["prefill_chunk_tuning"],
        "graph_steps_per_replay": int(args.graph_steps_per_replay),
        "thresholds": {"final_kl_max": float(args.kl_threshold)},
        "eager": _strip_logits(eager),
        "graph": _strip_logits(graph),
        "eager_generated_token_ids": eager_ids,
        "graph_generated_token_ids": graph_ids,
        "expected_generated_token_ids": expected,
        "seed_match": seed_match,
        "generated_match": generated_match,
        "generated_first_mismatch": _first_mismatch(eager_ids, graph_ids),
        "expected_match": expected_match,
        "expected_first_mismatch": _first_mismatch(graph_ids, expected),
        "final_sample_match": final_sample_match,
        "final_top1_match": final_top1_match,
        "finite_logits": finite_logits,
        "final_kl": final_kl,
        "kl_pass": kl_pass,
        "passed": passed,
        "parent_metrics": fixture.get("parent_metrics"),
        "notes": [
            "Correctness gate only; timings are diagnostic and no throughput row is retained here.",
            "Graph replay records generated ids on device and compares the full decode sequence against eager decode.",
            "Final lm-head logits are compared with KL/top-1 after replay; per-step logits are covered by the existing native-prefill fixture gate.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", type=Path, default=Path(DEFAULT_FIXTURE))
    parser.add_argument("--max-layers", type=int, default=40, help="0 means all configured layers")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override fixture decode_len")
    parser.add_argument("--graph-steps-per-replay", type=int, default=1)
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument(
        "--attn-aotriton-min-tokens",
        type=int,
        default=512,
        help="Run native prefill with AOTriton full-attention when prompt length is at least this threshold (0 disables for diagnostics).",
    )
    parser.add_argument("--prefill-linear-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-moe-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-full-attn-query-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-full-attn-post-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-full-attn-rope-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-chunk-autotune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefill-chunk-memory-budget-gib", type=float, default=0.0)
    add_kv_policy_args(parser, help_prefix="Resident KV storage for eager and graph decode")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.graph_steps_per_replay <= 0:
        raise ValueError("--graph-steps-per-replay must be positive")
    if args.attn_aotriton_min_tokens < 0:
        raise ValueError("--attn-aotriton-min-tokens must be non-negative")
    if args.prefill_chunk_memory_budget_gib < 0.0:
        raise ValueError("--prefill-chunk-memory-budget-gib must be non-negative")
    for name in (
        "prefill_linear_chunk_size",
        "prefill_moe_chunk_size",
        "prefill_full_attn_query_chunk_size",
        "prefill_full_attn_post_chunk_size",
        "prefill_full_attn_rope_chunk_size",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    payload = run(args)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
