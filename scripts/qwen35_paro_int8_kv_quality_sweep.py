#!/usr/bin/env python3
# ruff: noqa: E402
"""Long-context Qwen3.5/PARO BF16-vs-INT8 KV quality sweep.

This is the PARO analogue of the GGUF INT8/Q8 KV quality gates.  It runs a
resident BF16-KV reference and a candidate KV policy over one or more fixed
synthetic prompt lengths, records full lm-head logits at the prefill seed and at
each greedy decode step, and applies the usual KL/top-1 quality guard.

The script is intentionally a correctness/diagnostic harness.  Timings are
reported to make long runs auditable, but no performance claim should be made
from this output without the normal benchmark protocol.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.kvcache import ResolvedKVPolicy, resolve_kv_policy
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_kv_policy_args import (
    add_kv_policy_args,
    append_kv_policy_flags,
    kv_policy_json,
    resolve_args_kv_policy,
)
from scripts.qwen35_paro_bench import _prompt_tokens

DEFAULT_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")


def _parse_count(text: str) -> int:
    value = text.strip().lower()
    if not value:
        raise ValueError("empty count")
    if value.endswith("k"):
        return int(float(value[:-1]) * 1024)
    if value.endswith("m"):
        return int(float(value[:-1]) * 1024 * 1024)
    return int(value)


def _parse_count_list(text: str) -> list[int]:
    values = [_parse_count(item) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one prompt length")
    return sorted(set(values))


def _format_count(value: int) -> str:
    if value % 1024 == 0:
        return f"{value // 1024}K"
    return str(value)


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"compiler version file {path} is empty")
    return text


def _read_logits(session: Qwen35ParoResidentSession) -> np.ndarray:
    logits = np.empty((session.vocab_size,), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(logits),
        DeviceBuffer(session.lm_logits.ptr, logits.nbytes),
        runtime=session.runtime,
    )
    return logits


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


def _compare_logits(reference_logits: Sequence[np.ndarray], candidate_logits: Sequence[np.ndarray]) -> dict[str, Any]:
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
        for reference, candidate, token in zip(
            reference_logits,
            candidate_logits,
            reference_top1,
            strict=True,
        )
    ]
    first_top1_mismatch = None
    for idx, (ref_top, cand_top) in enumerate(zip(reference_top1, candidate_top1, strict=True)):
        if ref_top != cand_top:
            first_top1_mismatch = {"index": int(idx), "reference": int(ref_top), "candidate": int(cand_top)}
            break
    return {
        "positions": int(len(kls)),
        "kl": [float(item) for item in kls],
        "max_kl": float(max(kls)) if kls else 0.0,
        "mean_kl": float(np.mean(kls)) if kls else 0.0,
        "reference_top1": reference_top1,
        "candidate_top1": candidate_top1,
        "top1_matches": top1_matches,
        "top1_agreement": float(sum(top1_matches) / len(top1_matches)) if top1_matches else 1.0,
        "first_top1_mismatch": first_top1_mismatch,
        "max_abs_argmax_logit_delta": float(max(max_abs_argmax_logit_delta))
        if max_abs_argmax_logit_delta
        else 0.0,
    }


def _result_dict(result: Any) -> dict[str, Any]:
    return {
        "token_id": int(result.token_id),
        "token_text": result.token_text,
        "logit": float(result.logit),
    }


def _compact_owned_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "allocation_bytes": int(summary.get("allocation_bytes", 0)),
        "buffer_bytes": int(summary.get("buffer_bytes", 0)),
        "owned_direct_bytes": int(summary.get("owned_direct_bytes", 0)),
        "full_attention_layer_count": int(summary.get("full_attention_layer_count", 0)),
        "full_attention_kv_payload_bytes": int(summary.get("full_attention_kv_payload_bytes", 0)),
        "full_attention_kv_scale_bytes": int(summary.get("full_attention_kv_scale_bytes", 0)),
        "full_attention_kv_total_bytes": int(summary.get("full_attention_kv_total_bytes", 0)),
        "full_attention_kv_payload_bytes_per_element": float(
            summary.get("full_attention_kv_payload_bytes_per_element", 0.0)
        ),
        "kv_storage_dtype": summary.get("kv_storage_dtype"),
        "kv_scale_dtype": summary.get("kv_scale_dtype"),
        "kv_scale_granularity": summary.get("kv_scale_granularity"),
    }


def _strip_logits(run: dict[str, Any]) -> dict[str, Any]:
    stripped = {
        key: value
        for key, value in run.items()
        if key
        not in {
            "logits",
            "owned_buffer_summary_after_prefill",
            "owned_buffer_summary_after_decode",
        }
    }
    stripped["owned_buffer_summary_after_prefill"] = _compact_owned_summary(
        run.get("owned_buffer_summary_after_prefill", {})
    )
    stripped["owned_buffer_summary_after_decode"] = _compact_owned_summary(
        run.get("owned_buffer_summary_after_decode", {})
    )
    return stripped


def _first_mismatch(a: list[int], b: list[int]) -> dict[str, int] | None:
    for idx, (left, right) in enumerate(zip(a, b, strict=False)):
        if left != right:
            return {"index": int(idx), "left": int(left), "right": int(right)}
    if len(a) != len(b):
        return {"index": min(len(a), len(b)), "left": int(len(a)), "right": int(len(b))}
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


def _run_case(
    *,
    session: Qwen35ParoResidentSession,
    model: Path,
    prompt: str,
    token_id: int | None,
    prompt_length: int,
    decode_steps: int,
) -> dict[str, Any]:
    session.reset()
    # Keep parity with scripts/qwen35_readme_sweep.py: the same resident allocation
    # can still auto-resolve chunk sizes per prompt length.
    session._resolve_prefill_config_for_length(int(prompt_length))
    prompt_tokens = _prompt_tokens(model, prompt, token_id, prompt_length)
    logits: list[np.ndarray] = []
    generated: list[dict[str, Any]] = []
    prefill_start = time.perf_counter()
    seed = session.prefill_native(prompt_tokens, sample=True)
    prefill_seconds = time.perf_counter() - prefill_start
    if seed is None:
        raise RuntimeError("native prefill did not produce a seed token")
    logits.append(_read_logits(session))
    summary_after_prefill = session.owned_buffer_summary()
    current = seed
    decode_start = time.perf_counter()
    for offset in range(decode_steps):
        current = session.step(current.token_id, position=prompt_length + offset, sample=True)
        if current is None:
            raise RuntimeError(f"decode did not produce token {offset}")
        generated.append(_result_dict(current))
        logits.append(_read_logits(session))
    decode_seconds = time.perf_counter() - decode_start
    summary_after_decode = session.owned_buffer_summary()
    return {
        "prompt_length": int(prompt_length),
        "decode_steps": int(decode_steps),
        "seed": _result_dict(seed),
        "generated": generated,
        "generated_token_ids": [int(item["token_id"]) for item in generated],
        "logits": logits,
        "finite_logits": bool(all(np.isfinite(item).all() for item in logits)),
        "prefill_seconds": float(prefill_seconds),
        "decode_seconds": float(decode_seconds),
        "owned_buffer_summary_after_prefill": summary_after_prefill,
        "owned_buffer_summary_after_decode": summary_after_decode,
        "prefill_chunk_sizes": {
            "linear": int(session.prefill_config.linear_chunk_size),
            "moe": int(session.prefill_config.moe_chunk_size),
            "full_attn_query": int(session.prefill_config.full_attn_query_chunk_size),
            "full_attn_post": int(session.prefill_config.full_attn_post_chunk_size),
            "full_attn_rope": int(session.prefill_config.full_attn_rope_chunk_size),
        },
        "prefill_chunk_tuning": getattr(session, "prefill_chunk_tuning", None),
    }


def _run_policy_sweep(
    *,
    runner: Qwen35ParoNextTokenRunner,
    model: Path,
    prompt: str,
    token_id: int | None,
    prompt_lengths: Sequence[int],
    decode_steps: int,
    max_sequence_length: int,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
    prefill_config: PrefillConfig,
    kv_policy: ResolvedKVPolicy,
) -> dict[int, dict[str, Any]]:
    runs: dict[int, dict[str, Any]] = {}
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=max_layers,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        prefill_config=prefill_config,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    ) as session:
        for prompt_length in prompt_lengths:
            runs[int(prompt_length)] = _run_case(
                session=session,
                model=model,
                prompt=prompt,
                token_id=token_id,
                prompt_length=int(prompt_length),
                decode_steps=int(decode_steps),
            )
    gc.collect()
    return runs


def _command(args: argparse.Namespace) -> str:
    command = (
        "python3 scripts/qwen35_paro_int8_kv_quality_sweep.py"
        f" --model {args.model}"
        f" --prompt-lengths {args.prompt_lengths}"
        f" --decode-steps {args.decode_steps}"
        f" --token-id {args.token_id}"
        f" --max-layers {args.max_layers}"
        f" --kl-threshold {args.kl_threshold}"
        f" --top1-threshold {args.top1_threshold}"
    )
    if args.prompt != "Hello":
        command += f" --prompt {json.dumps(args.prompt)}"
    if args.max_sequence_length:
        command += f" --max-sequence-length {args.max_sequence_length}"
    if args.compiler_version_file is not None:
        command += f" --compiler-version-file {args.compiler_version_file}"
    if args.require_cached_build:
        command += " --require-cached-build"
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
    if args.backend != "hip_gfx1100":
        command += f" --backend {args.backend}"
    if args.shared_expert_format != "packed_paro_w4":
        command += f" --shared-expert-format {args.shared_expert_format}"
    command = append_kv_policy_flags(command, args)
    if args.json is not None:
        command += f" --json {args.json}"
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompt_lengths = _parse_count_list(args.prompt_lengths)
    decode_steps = int(args.decode_steps)
    max_sequence_length = int(args.max_sequence_length or max(prompt_lengths) + decode_steps + 2)
    if max_sequence_length < max(prompt_lengths) + decode_steps + 1:
        raise ValueError("max-sequence-length must cover the largest prompt plus decode")
    compiler_version = _read_compiler_version(args.compiler_version_file)
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
    runner = Qwen35ParoNextTokenRunner(
        args.model,
        shared_expert_format=None if args.shared_expert_format == "auto" else args.shared_expert_format,
        backend=args.backend,
    )
    reference_policy = resolve_kv_policy("bf16")
    candidate_policy = resolve_args_kv_policy(args, block_size=256)
    started = time.perf_counter()
    reference_runs = _run_policy_sweep(
        runner=runner,
        model=args.model,
        prompt=args.prompt,
        token_id=args.token_id,
        prompt_lengths=prompt_lengths,
        decode_steps=decode_steps,
        max_sequence_length=max_sequence_length,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        prefill_config=prefill_config,
        kv_policy=reference_policy,
    )
    candidate_runs = _run_policy_sweep(
        runner=runner,
        model=args.model,
        prompt=args.prompt,
        token_id=args.token_id,
        prompt_lengths=prompt_lengths,
        decode_steps=decode_steps,
        max_sequence_length=max_sequence_length,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        prefill_config=prefill_config,
        kv_policy=candidate_policy,
    )
    rows: list[dict[str, Any]] = []
    for prompt_length in prompt_lengths:
        reference = reference_runs[int(prompt_length)]
        candidate = candidate_runs[int(prompt_length)]
        comparison = _compare_logits(reference["logits"], candidate["logits"])
        reference_ids = [int(item) for item in reference["generated_token_ids"]]
        candidate_ids = [int(item) for item in candidate["generated_token_ids"]]
        seed_match = int(reference["seed"]["token_id"]) == int(candidate["seed"]["token_id"])
        generated_match = reference_ids == candidate_ids
        kl_pass = comparison["mean_kl"] <= float(args.kl_threshold)
        top1_pass = comparison["top1_agreement"] >= float(args.top1_threshold)
        finite_logits = bool(reference["finite_logits"] and candidate["finite_logits"])
        audit_after_prefill = _kv_memory_audit(
            candidate["owned_buffer_summary_after_prefill"],
            candidate_policy.storage_dtype.value,
        )
        audit_after_decode = _kv_memory_audit(
            candidate["owned_buffer_summary_after_decode"],
            candidate_policy.storage_dtype.value,
        )
        memory_audit_pass = bool(audit_after_prefill["passed"] and audit_after_decode["passed"])
        passed = bool(finite_logits and kl_pass and top1_pass and memory_audit_pass)
        rows.append(
            {
                "workload": f"{_format_count(int(prompt_length))}/{decode_steps}",
                "prompt_length": int(prompt_length),
                "decode_steps": int(decode_steps),
                "passed": passed,
                "seed_match": bool(seed_match),
                "generated_match": bool(generated_match),
                "generated_first_mismatch": _first_mismatch(reference_ids, candidate_ids),
                "finite_logits": finite_logits,
                "kl_pass": bool(kl_pass),
                "top1_pass": bool(top1_pass),
                "memory_audit_pass": memory_audit_pass,
                "logit_gate": comparison,
                "reference_seed_token_id": int(reference["seed"]["token_id"]),
                "candidate_seed_token_id": int(candidate["seed"]["token_id"]),
                "reference_generated_token_ids": reference_ids,
                "candidate_generated_token_ids": candidate_ids,
                "reference": _strip_logits(reference),
                "candidate": _strip_logits(candidate),
                "candidate_kv_memory_audit_after_prefill": audit_after_prefill,
                "candidate_kv_memory_audit_after_decode": audit_after_decode,
            }
        )
    overall_passed = all(bool(row["passed"]) for row in rows)
    elapsed = time.perf_counter() - started
    return {
        "schema": 1,
        "status": "accepted" if overall_passed else "rejected_correctness",
        "blocked_reason": None if overall_passed else "one or more PARO KV quality rows failed",
        "performance_claim": False,
        "mode": "qwen35_paro_int8_kv_quality_sweep",
        "command": _command(args),
        "model": str(args.model),
        "quant": "w4_paro",
        "backend": runner.backend,
        "requested_backend": args.backend,
        "target_arch": runner.target_arch,
        "prompt_lengths": [int(item) for item in prompt_lengths],
        "decode_steps": int(decode_steps),
        "max_sequence_length": int(max_sequence_length),
        "max_layers": int(args.max_layers),
        "token_id": None if args.token_id is None else int(args.token_id),
        "prompt": args.prompt if args.token_id is None else None,
        "elapsed_seconds": float(elapsed),
        "reference_kv_policy": kv_policy_json(reference_policy),
        "candidate_kv_policy": kv_policy_json(candidate_policy),
        "quality_thresholds": {
            "kl_mean_max": float(args.kl_threshold),
            "top1_agreement_min": float(args.top1_threshold),
        },
        "prefill_config_request": {
            "linear_chunk_size": int(args.prefill_linear_chunk_size),
            "moe_chunk_size": int(args.prefill_moe_chunk_size),
            "full_attn_query_chunk_size": int(args.prefill_full_attn_query_chunk_size),
            "full_attn_post_chunk_size": int(args.prefill_full_attn_post_chunk_size),
            "full_attn_rope_chunk_size": int(args.prefill_full_attn_rope_chunk_size),
            "attn_aotriton_min_tokens": int(args.attn_aotriton_min_tokens),
            "prefill_chunk_autotune": bool(args.prefill_chunk_autotune),
            "prefill_chunk_memory_budget_gib": float(args.prefill_chunk_memory_budget_gib),
        },
        "rows": rows,
        "summary": {
            "passed": bool(overall_passed),
            "first_failing_row": next((row["workload"] for row in rows if not row["passed"]), None),
            "max_kl_by_workload": {
                row["workload"]: float(row["logit_gate"]["max_kl"]) for row in rows
            },
            "top1_by_workload": {
                row["workload"]: float(row["logit_gate"]["top1_agreement"]) for row in rows
            },
        },
        "notes": [
            "Correctness/quality diagnostic only; no throughput row is retained from this script.",
            "BF16 KV resident PARO is the reference; candidate KV storage is selected by --kv-storage.",
            "Quality gate uses mean KL <= threshold and top-1 agreement >= threshold over prefill seed plus decode logits.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-lengths", default="4K", help="Comma-separated lengths: 4K,32K,64K,128K")
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--prompt", default="Hello", help="Used only if --token-id is omitted via direct API/tests")
    parser.add_argument("--max-sequence-length", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=40, help="0 means all configured layers")
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument(
        "--attn-aotriton-min-tokens",
        type=int,
        default=512,
        help="Run native prefill with AOTriton full-attention when prompt length is at least this threshold.",
    )
    parser.add_argument("--prefill-linear-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-moe-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-full-attn-query-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-full-attn-post-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-full-attn-rope-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-chunk-autotune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefill-chunk-memory-budget-gib", type=float, default=0.0)
    parser.add_argument("--backend", choices=("auto", "hip_gfx1100", "hip_gfx1151"), default="hip_gfx1100")
    parser.add_argument(
        "--shared-expert-format",
        choices=("auto", "legacy_fp16", "packed_paro_w4"),
        default="packed_paro_w4",
    )
    add_kv_policy_args(
        parser,
        default_storage="int8_per_token_head",
        help_prefix="Candidate KV storage for the BF16-vs-candidate PARO quality sweep",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.decode_steps < 0:
        raise ValueError("--decode-steps must be non-negative")
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
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0 if payload["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
