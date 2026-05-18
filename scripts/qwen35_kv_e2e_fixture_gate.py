#!/usr/bin/env python3
"""End-to-end Qwen3.5/PARO KV-storage fixture gate.

This gate compares a BF16-KV native resident run against a candidate KV policy
(``bf16`` or ``int8_per_token_head``) on the fixed parent prompt/decode fixture.
It collects full lm-head logits at the prefill seed and each decode position,
then applies KL/top-1 gates in addition to deterministic generated-token checks.
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
    fixture = json.loads(path.read_text(encoding="utf-8"))
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


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"compiler version file {path} is empty")
    return text


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
    kv_policy: ResolvedKVPolicy,
) -> dict[str, Any]:
    max_sequence = len(prompt_tokens) + decode_tokens + 2
    logits: list[np.ndarray] = []
    generated: list[dict[str, Any]] = []
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
        owned_device_bytes = _owned_device_bytes(session)
        prefill_start = time.perf_counter()
        seed = session.prefill_native(prompt_tokens, sample=True)
        prefill_seconds = time.perf_counter() - prefill_start
        if seed is None:
            raise RuntimeError("native prefill did not produce a seed token")
        logits.append(_read_logits(session))
        summary_after_prefill = session.owned_buffer_summary()
        current = seed
        decode_start = time.perf_counter()
        for offset in range(decode_tokens):
            current = session.step(current.token_id, position=len(prompt_tokens) + offset, sample=True)
            if current is None:
                raise RuntimeError(f"decode did not produce token {offset}")
            generated.append(_result_dict(current))
            logits.append(_read_logits(session))
        decode_seconds = time.perf_counter() - decode_start
        summary_after_decode = session.owned_buffer_summary()
        detail = getattr(session, "last_prefill_execution", None)
        chunk_sizes = {
            "linear": int(session.prefill_config.linear_chunk_size),
            "moe": int(session.prefill_config.moe_chunk_size),
            "full_attn_query": int(session.prefill_config.full_attn_query_chunk_size),
            "full_attn_post": int(session.prefill_config.full_attn_post_chunk_size),
            "full_attn_rope": int(session.prefill_config.full_attn_rope_chunk_size),
        }
        chunk_tuning = getattr(session, "prefill_chunk_tuning", None)
    return {
        "kv_policy": kv_policy_json(kv_policy),
        "seed": _result_dict(seed),
        "generated": generated,
        "generated_token_ids": [int(item["token_id"]) for item in generated],
        "logits": logits,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "owned_device_bytes": owned_device_bytes,
        "owned_buffer_summary_after_prefill": summary_after_prefill,
        "owned_buffer_summary_after_decode": summary_after_decode,
        "prefill_execution_detail": detail,
        "prefill_chunk_sizes": chunk_sizes,
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


def _compare_logits(reference_logits: list[np.ndarray], candidate_logits: list[np.ndarray]) -> dict[str, Any]:
    if len(reference_logits) != len(candidate_logits):
        raise ValueError("reference/candidate logits length mismatch")
    kls = [
        _kl_divergence(reference, candidate)
        for reference, candidate in zip(reference_logits, candidate_logits, strict=True)
    ]
    reference_top1 = [int(np.argmax(item)) for item in reference_logits]
    candidate_top1 = [int(np.argmax(item)) for item in candidate_logits]
    top1_matches = [a == b for a, b in zip(reference_top1, candidate_top1, strict=True)]
    max_abs_argmax_logit_delta = [
        abs(float(candidate[token]) - float(reference[token]))
        for reference, candidate, token in zip(reference_logits, candidate_logits, reference_top1, strict=True)
    ]
    return {
        "positions": len(kls),
        "kl": kls,
        "max_kl": max(kls) if kls else 0.0,
        "mean_kl": float(np.mean(kls)) if kls else 0.0,
        "reference_top1": reference_top1,
        "candidate_top1": candidate_top1,
        "top1_matches": top1_matches,
        "top1_agreement": (sum(top1_matches) / len(top1_matches)) if top1_matches else 1.0,
        "max_abs_argmax_logit_delta": max(max_abs_argmax_logit_delta) if max_abs_argmax_logit_delta else 0.0,
    }


def _strip_logits(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key != "logits"}


def _first_mismatch(a: list[int], b: list[int]) -> dict[str, int] | None:
    for idx, (left, right) in enumerate(zip(a, b, strict=False)):
        if left != right:
            return {"index": idx, "left": int(left), "right": int(right)}
    if len(a) != len(b):
        return {"index": min(len(a), len(b)), "left": len(a), "right": len(b)}
    return None


def _kv_memory_audit(summary: dict[str, Any], storage_dtype: str) -> dict[str, Any]:
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
    command = f"python3 scripts/qwen35_kv_e2e_fixture_gate.py --model {args.model} --fixture {args.fixture}"
    command += f" --max-layers {args.max_layers}"
    if args.max_new_tokens is not None:
        command += f" --max-new-tokens {args.max_new_tokens}"
    if args.compiler_version_file is not None:
        command += f" --compiler-version-file {args.compiler_version_file}"
    if args.require_cached_build:
        command += " --require-cached-build"
    if args.kl_threshold != 0.05:
        command += f" --kl-threshold {args.kl_threshold}"
    if args.top1_threshold != 0.90:
        command += f" --top1-threshold {args.top1_threshold}"
    if args.attn_aotriton_min_tokens != 512:
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
    if decode_tokens < 0:
        raise ValueError("decode_tokens must be non-negative")
    expected = [int(item) for item in fixture["expected_generated_token_ids"][:decode_tokens]]
    compiler_version = _read_compiler_version(args.compiler_version_file)
    runner = Qwen35ParoNextTokenRunner(args.model)
    reference_kv_policy = resolve_kv_policy("bf16")
    candidate_kv_policy = resolve_args_kv_policy(args, block_size=256)
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
    reference = _run_once(
        runner,
        prompt_tokens,
        decode_tokens=decode_tokens,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        prefill_config=prefill_config,
        kv_policy=reference_kv_policy,
    )
    candidate = _run_once(
        runner,
        prompt_tokens,
        decode_tokens=decode_tokens,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        prefill_config=prefill_config,
        kv_policy=candidate_kv_policy,
    )

    comparison = _compare_logits(reference["logits"], candidate["logits"])
    reference_ids = [int(item) for item in reference["generated_token_ids"]]
    candidate_ids = [int(item) for item in candidate["generated_token_ids"]]
    seed_match = int(reference["seed"]["token_id"]) == int(candidate["seed"]["token_id"])
    generated_match = reference_ids == candidate_ids
    expected_match = candidate_ids == expected
    finite_logits = all(np.isfinite(item).all() for item in reference["logits"] + candidate["logits"])
    kl_pass = comparison["max_kl"] <= float(args.kl_threshold)
    top1_pass = comparison["top1_agreement"] >= float(args.top1_threshold)
    audit_after_prefill = _kv_memory_audit(
        candidate["owned_buffer_summary_after_prefill"],
        candidate_kv_policy.storage_dtype.value,
    )
    audit_after_decode = _kv_memory_audit(
        candidate["owned_buffer_summary_after_decode"],
        candidate_kv_policy.storage_dtype.value,
    )
    memory_audit_pass = bool(audit_after_prefill["passed"] and audit_after_decode["passed"])
    passed = bool(
        seed_match
        and generated_match
        and expected_match
        and finite_logits
        and kl_pass
        and top1_pass
        and memory_audit_pass
    )
    return {
        "schema": 1,
        "status": "accepted" if passed else "rejected_correctness",
        "blocked_reason": None if passed else "KV end-to-end fixture gate failed",
        "model": str(Path(args.model)),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": "qwen35_paro_kv_e2e_fixture_gate",
        "command": _command(args),
        "performance_claim": False,
        "fixture": args.fixture.as_posix(),
        "prompt_length": len(prompt_tokens),
        "decode_tokens": decode_tokens,
        "max_layers": int(args.max_layers),
        "kv_storage_dtype": candidate_kv_policy.storage_dtype.value,
        "reference_kv_storage_dtype": reference_kv_policy.storage_dtype.value,
        "candidate_kv_storage_dtype": candidate_kv_policy.storage_dtype.value,
        "reference_kv_policy": kv_policy_json(reference_kv_policy),
        "candidate_kv_policy": kv_policy_json(candidate_kv_policy),
        "kv_policy": kv_policy_json(candidate_kv_policy),
        "attn_aotriton_min_tokens": int(args.attn_aotriton_min_tokens),
        "requested_prefill_chunk_sizes": {
            "linear": int(args.prefill_linear_chunk_size),
            "moe": int(args.prefill_moe_chunk_size),
            "full_attn_query": int(args.prefill_full_attn_query_chunk_size),
            "full_attn_post": int(args.prefill_full_attn_post_chunk_size),
            "full_attn_rope": int(args.prefill_full_attn_rope_chunk_size),
        },
        "prefill_chunk_autotune": bool(args.prefill_chunk_autotune),
        "prefill_chunk_memory_budget_gib": float(args.prefill_chunk_memory_budget_gib),
        "prefill_chunk_sizes": candidate["prefill_chunk_sizes"],
        "prefill_chunk_tuning": candidate["prefill_chunk_tuning"],
        "thresholds": {
            "kl_max": float(args.kl_threshold),
            "top1_agreement_min": float(args.top1_threshold),
        },
        "reference": _strip_logits(reference),
        "candidate": _strip_logits(candidate),
        "reference_seed_token_id": int(reference["seed"]["token_id"]),
        "candidate_seed_token_id": int(candidate["seed"]["token_id"]),
        "reference_generated_token_ids": reference_ids,
        "candidate_generated_token_ids": candidate_ids,
        "expected_generated_token_ids": expected,
        "seed_match": seed_match,
        "generated_match": generated_match,
        "generated_first_mismatch": _first_mismatch(reference_ids, candidate_ids),
        "expected_match": expected_match,
        "expected_first_mismatch": _first_mismatch(candidate_ids, expected),
        "finite_logits": finite_logits,
        "logit_position_labels": ["prefill_seed"] + [f"decode_{idx}" for idx in range(decode_tokens)],
        "logit_gate": comparison,
        "kl_pass": kl_pass,
        "top1_pass": top1_pass,
        "candidate_kv_memory_audit_after_prefill": audit_after_prefill,
        "candidate_kv_memory_audit_after_decode": audit_after_decode,
        "memory_audit_pass": memory_audit_pass,
        "passed": passed,
        "parent_metrics": fixture.get("parent_metrics"),
        "notes": [
            "Correctness gate only; timings are diagnostic and no throughput row is retained here.",
            "BF16 KV native prefill/decode is the reference; candidate KV storage is selected by --kv-storage.",
            "KL/top-1 compare full lm-head logits at the prefill seed and every sampled decode position.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", type=Path, default=Path(DEFAULT_FIXTURE))
    parser.add_argument("--max-layers", type=int, default=40, help="0 means all configured layers")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override fixture decode_len; 0 gates prefill seed only")
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
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
    add_kv_policy_args(
        parser,
        default_storage="int8_per_token_head",
        help_prefix="Candidate KV storage for the BF16-vs-candidate E2E fixture gate",
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
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
