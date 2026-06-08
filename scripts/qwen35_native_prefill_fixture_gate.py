#!/usr/bin/env python3
"""Native-prefill fixture gate against serial resident logits.

This correctness gate is intentionally separate from throughput benchmarks. It
runs the parent fixture once through serial token-by-token resident prefill and
once through ``prefill_native(...)``, compares generated IDs, computes KL over
full lm-head logits at the prefill seed and sampled decode positions, and checks
that the native run still matches the parent fixture's expected generated IDs.
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
from hipengine.kvcache import ResolvedKVPolicy, resolve_kv_policy
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_kv_policy_args import add_kv_policy_args, append_kv_policy_flags, kv_policy_json, resolve_args_kv_policy

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)
DEFAULT_FIXTURE = "fixtures/qwen35_paro/parent_512_32_seed1234.json"


def _load_fixture(path: Path) -> dict[str, Any]:
    fixture = json.loads(path.read_text())
    required = {"prompt_ids", "expected_generated_token_ids", "decode_len"}
    missing = required.difference(fixture)
    if missing:
        raise ValueError(f"fixture missing required keys: {sorted(missing)}")
    return fixture


def _owned_device_bytes(session: Qwen35ParoResidentSession) -> int:
    allocation_bytes = sum(int(allocation.buffer.nbytes) for allocation in session.allocations)
    buffer_bytes = sum(int(buffer.nbytes) for buffer in session.buffers)
    state_bytes = sum(
        int(state.workspace.allocation(name).buffer.nbytes)
        for state in session.states
        for name in state.workspace.names
    )
    return allocation_bytes + buffer_bytes + state_bytes


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
    prefill_mode: str,
    prefill_config: PrefillConfig | None = None,
    kv_policy: ResolvedKVPolicy | None = None,
) -> dict[str, Any]:
    max_sequence = len(prompt_tokens) + decode_tokens + 2
    logits: list[np.ndarray] = []
    generated: list[dict[str, Any]] = []
    resolved_kv_policy = kv_policy or resolve_kv_policy("bf16")
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_layers=max_layers,
        prefill_config=prefill_config,
        kv_policy=resolved_kv_policy.create_policy(),
        kv_scale_dtype=resolved_kv_policy.scale_dtype,
        kv_scale_granularity=resolved_kv_policy.scale_granularity,
    ) as session:
        owned_device_bytes = _owned_device_bytes(session)
        prefill_start = time.perf_counter()
        if prefill_mode == "native":
            seed = session.prefill_native(prompt_tokens, sample=True)
        elif prefill_mode == "serial":
            seed = None
            for position, token_id in enumerate(prompt_tokens):
                seed = session.step(token_id, position=position, sample=(position == len(prompt_tokens) - 1))
        else:
            raise ValueError(f"unsupported prefill_mode {prefill_mode!r}")
        prefill_seconds = time.perf_counter() - prefill_start
        if seed is None:
            raise RuntimeError(f"{prefill_mode} prefill did not produce a seed token")
        owned_buffer_summary_after_prefill = session.owned_buffer_summary()
        logits.append(_read_logits(session))
        current = seed
        decode_start = time.perf_counter()
        for offset in range(decode_tokens):
            current = session.step(current.token_id, position=len(prompt_tokens) + offset)
            if current is None:
                raise RuntimeError(f"{prefill_mode} decode did not produce token {offset}")
            generated.append(_result_dict(current))
            logits.append(_read_logits(session))
        decode_seconds = time.perf_counter() - decode_start
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
        "prefill_mode": prefill_mode,
        "kv_policy": kv_policy_json(resolved_kv_policy),
        "seed": _result_dict(seed),
        "generated": generated,
        "logits": logits,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "owned_device_bytes": owned_device_bytes,
        "owned_buffer_summary_after_prefill": owned_buffer_summary_after_prefill,
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


def _compare_logits(serial_logits: list[np.ndarray], native_logits: list[np.ndarray]) -> dict[str, Any]:
    if len(serial_logits) != len(native_logits):
        raise ValueError("serial/native logits length mismatch")
    kls = [_kl_divergence(serial, native) for serial, native in zip(serial_logits, native_logits, strict=True)]
    serial_top1 = [int(np.argmax(item)) for item in serial_logits]
    native_top1 = [int(np.argmax(item)) for item in native_logits]
    top1_matches = [a == b for a, b in zip(serial_top1, native_top1, strict=True)]
    max_abs_argmax_logit_delta = [
        abs(float(native[token]) - float(serial[token]))
        for serial, native, token in zip(serial_logits, native_logits, serial_top1, strict=True)
    ]
    return {
        "positions": len(kls),
        "kl": kls,
        "max_kl": max(kls) if kls else 0.0,
        "mean_kl": float(np.mean(kls)) if kls else 0.0,
        "serial_top1": serial_top1,
        "native_top1": native_top1,
        "top1_matches": top1_matches,
        "top1_agreement": (sum(top1_matches) / len(top1_matches)) if top1_matches else 1.0,
        "max_abs_argmax_logit_delta": max(max_abs_argmax_logit_delta) if max_abs_argmax_logit_delta else 0.0,
    }


def _strip_logits(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key != "logits"}


def _native_kv_memory_audit(summary: dict[str, Any], storage_dtype: str) -> dict[str, Any]:
    full_layers = list(summary.get("full_attention_layers", ()))
    if storage_dtype != "int8_per_token_head":
        return {"required": False, "passed": True, "persistent_bf16_kv_layers": []}
    persistent_bf16 = [
        int(layer.get("layer_id", -1))
        for layer in full_layers
        if layer.get("storage_dtype") == "bf16" or layer.get("payload_dtype") == "bf16"
    ]
    missing_scales = [
        int(layer.get("layer_id", -1))
        for layer in full_layers
        if not layer.get("scale_metadata") or int(layer.get("scale_metadata", {}).get("scale_bytes", 0)) <= 0
    ]
    return {
        "required": True,
        "passed": not persistent_bf16 and not missing_scales,
        "persistent_bf16_kv_layers": persistent_bf16,
        "missing_int8_scale_layers": missing_scales,
        "full_attention_kv_payload_bytes": int(summary.get("full_attention_kv_payload_bytes", 0)),
        "full_attention_kv_scale_bytes": int(summary.get("full_attention_kv_scale_bytes", 0)),
    }


def _command(args: argparse.Namespace) -> str:
    command = f"python3 scripts/qwen35_native_prefill_fixture_gate.py --model {args.model} --fixture {args.fixture}"
    if args.max_layers:
        command += f" --max-layers {args.max_layers}"
    if args.max_new_tokens is not None:
        command += f" --max-new-tokens {args.max_new_tokens}"
    if args.kl_threshold != 0.05:
        command += f" --kl-threshold {args.kl_threshold}"
    if args.top1_threshold != 0.90:
        command += f" --top1-threshold {args.top1_threshold}"
    if getattr(args, "attn_aotriton_min_tokens", 0):
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
    command = append_kv_policy_flags(command, args)
    if args.json is not None:
        command += f" --json {args.json}"
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture = _load_fixture(args.fixture)
    prompt_tokens = [int(item) for item in fixture["prompt_ids"]]
    decode_tokens = int(fixture["decode_len"] if args.max_new_tokens is None else args.max_new_tokens)
    expected = [int(item) for item in fixture["expected_generated_token_ids"][:decode_tokens]]
    runner = Qwen35ParoNextTokenRunner(args.model)
    serial_kv_policy = resolve_kv_policy("bf16")
    native_kv_policy = resolve_args_kv_policy(args, block_size=256)
    serial = _run_once(runner, prompt_tokens, decode_tokens=decode_tokens, max_layers=args.max_layers, prefill_mode="serial", kv_policy=serial_kv_policy)
    native = _run_once(
        runner,
        prompt_tokens,
        decode_tokens=decode_tokens,
        max_layers=args.max_layers,
        prefill_mode="native",
        prefill_config=PrefillConfig(
            linear_chunk_size=args.prefill_linear_chunk_size,
            moe_chunk_size=args.prefill_moe_chunk_size,
            full_attn_query_chunk_size=args.prefill_full_attn_query_chunk_size,
            full_attn_post_chunk_size=args.prefill_full_attn_post_chunk_size,
            full_attn_rope_chunk_size=args.prefill_full_attn_rope_chunk_size,
            attn_aotriton_min_tokens=args.attn_aotriton_min_tokens,
            auto_tune_chunk_sizes=args.prefill_chunk_autotune,
            chunk_tune_memory_budget_gib=args.prefill_chunk_memory_budget_gib,
        ),
        kv_policy=native_kv_policy,
    )
    comparison = _compare_logits(serial["logits"], native["logits"])
    serial_generated_ids = [int(item["token_id"]) for item in serial["generated"]]
    native_generated_ids = [int(item["token_id"]) for item in native["generated"]]
    seed_match = int(serial["seed"]["token_id"]) == int(native["seed"]["token_id"])
    generated_match = serial_generated_ids == native_generated_ids
    expected_match = native_generated_ids == expected
    finite_logits = all(np.isfinite(item).all() for item in serial["logits"] + native["logits"])
    kl_pass = comparison["max_kl"] <= float(args.kl_threshold)
    top1_pass = comparison["top1_agreement"] >= float(args.top1_threshold)
    memory_audit = _native_kv_memory_audit(native["owned_buffer_summary_after_prefill"], native_kv_policy.storage_dtype.value)
    memory_audit_pass = bool(memory_audit["passed"])
    passed = bool(seed_match and generated_match and expected_match and finite_logits and kl_pass and top1_pass and memory_audit_pass)
    return {
        "schema": 1,
        "status": "accepted" if passed else "rejected_correctness",
        "blocked_reason": None if passed else "native prefill fixture gate failed",
        "model": str(Path(args.model)),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": "qwen35_paro_native_prefill_fixture_gate",
        "command": _command(args),
        "performance_claim": False,
        "fixture": args.fixture.as_posix(),
        "prompt_length": len(prompt_tokens),
        "decode_tokens": decode_tokens,
        "max_layers": int(args.max_layers),
        "attn_aotriton_min_tokens": int(args.attn_aotriton_min_tokens),
        "native_kv_storage_dtype": native_kv_policy.storage_dtype.value,
        "kv_storage_dtype": native_kv_policy.storage_dtype.value,
        "kv_policy": kv_policy_json(native_kv_policy),
        "serial_reference_kv_policy": kv_policy_json(serial_kv_policy),
        "requested_prefill_chunk_sizes": {
            "linear": int(args.prefill_linear_chunk_size),
            "moe": int(args.prefill_moe_chunk_size),
            "full_attn_query": int(args.prefill_full_attn_query_chunk_size),
            "full_attn_post": int(args.prefill_full_attn_post_chunk_size),
            "full_attn_rope": int(args.prefill_full_attn_rope_chunk_size),
        },
        "prefill_chunk_autotune": bool(args.prefill_chunk_autotune),
        "prefill_chunk_memory_budget_gib": float(args.prefill_chunk_memory_budget_gib),
        "prefill_chunk_sizes": native["prefill_chunk_sizes"],
        "prefill_chunk_tuning": native["prefill_chunk_tuning"],
        "thresholds": {
            "kl_max": float(args.kl_threshold),
            "top1_agreement_min": float(args.top1_threshold),
        },
        "serial": _strip_logits(serial),
        "native": _strip_logits(native),
        "serial_generated_token_ids": serial_generated_ids,
        "native_generated_token_ids": native_generated_ids,
        "expected_generated_token_ids": expected,
        "seed_match": seed_match,
        "generated_match": generated_match,
        "expected_match": expected_match,
        "finite_logits": finite_logits,
        "logit_gate": comparison,
        "kl_pass": kl_pass,
        "top1_pass": top1_pass,
        "native_kv_memory_audit": memory_audit,
        "memory_audit_pass": memory_audit_pass,
        "passed": passed,
        "parent_metrics": fixture.get("parent_metrics"),
        "notes": [
            "Correctness gate only; timings are diagnostic and no throughput row is retained here.",
            "KL/top-1 compare native prefill against the validated serial resident path over full lm-head logits.",
            "Expected generated IDs come from the parent fixture after consuming the prefill seed token.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", type=Path, default=Path(DEFAULT_FIXTURE))
    parser.add_argument("--max-layers", type=int, default=40, help="0 means all configured layers")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override fixture decode_len")
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
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
    add_kv_policy_args(
        parser,
        legacy_storage_flags=("--native-kv-storage-dtype",),
        help_prefix="KV storage policy for the native prefill candidate; serial reference remains BF16",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.max_new_tokens is not None and args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
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
        args.json.write_text(text + "\n")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
