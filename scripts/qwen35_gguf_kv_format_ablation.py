#!/usr/bin/env python3
# ruff: noqa: E402
"""Screen Hadamard, KIVI, and Q8_0-shaped KV formats on hipEngine GGUF.

The harness loads one Q4_K_M GGUF resident-weight session with BF16 K/V,
records a BF16-reference-token trajectory, and resets/reuses that session for
host-emulated candidate cache formats. Candidate K/V are quantized and
reconstructed from the resident BF16 cache before teacher-forced decode.

This is a representation-fidelity diagnostic, not native-kernel or performance
evidence. The current decode row is consumed as BF16 during its own attention
step and round-tripped immediately afterward.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from scripts.qwen35_paro_int8_kv_quality_sweep import _compare_logits
from scripts.qwen35_paro_kv_format_ablation import (
    FormatSpec,
    _aggregate_reconstruction,
    _bf16_bits_to_float32,
    _candidate_catalog,
    _compact_run,
    _distribution_summary,
    _format_memory_bytes,
    _git_provenance,
    _parse_candidates,
    _roundtrip_pair,
    _select_recommendation,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
FAST_GGUF_CANDIDATES = "baseline_max,group32,hadamard_group32,kivi_int8"
# Exact tokenization of qwen35_paro_kv_format_ablation.FAST_SCREEN_PROMPT with
# the Qwen3.6 tokenizer. Repeating this 45-token unit reproduces the PARO S1
# mixed_v1 512-token prompt hash and avoids requiring a tokenizer sidecar next
# to a standalone GGUF file.
_MIXED_V1_TOKEN_UNIT = (
    79852,
    6297,
    79659,
    21059,
    13,
    30523,
    279,
    4711,
    2029,
    11,
    11290,
    220,
    16,
    22,
    348,
    220,
    17,
    18,
    11,
    10033,
    264,
    12654,
    14650,
    11,
    5707,
    25899,
    303,
    6163,
    321,
    220,
    247359,
    11,
    321,
    20480,
    279,
    4566,
    5224,
    5515,
    5046,
    9226,
    763,
    220,
    23,
    7563,
    220,
)


@dataclass(frozen=True)
class GGUFCacheLayout:
    full_layer_ids: tuple[int, ...]
    num_kv_heads: int
    head_dim: int


def _cache_layout(session: object) -> GGUFCacheLayout:
    runner = getattr(session, "runner", None)
    weights = getattr(runner, "weights", None)
    cfg = getattr(weights, "config", None)
    scratch = getattr(session, "scratch", None)
    if cfg is None or scratch is None:
        raise RuntimeError("GGUF session is missing resident weights or scratch")
    key_caches = tuple(getattr(scratch, "full_key_caches", ()))
    value_caches = tuple(getattr(scratch, "full_value_caches", ()))
    if not key_caches or len(key_caches) != len(value_caches):
        raise RuntimeError("GGUF session has an invalid full-attention cache table")
    full_layer_ids = tuple(
        index
        for index, (key_cache, value_cache) in enumerate(zip(key_caches, value_caches, strict=True))
        if key_cache is not None and value_cache is not None
    )
    if not full_layer_ids:
        raise RuntimeError("GGUF session has no full-attention cache layers")
    return GGUFCacheLayout(
        full_layer_ids=full_layer_ids,
        num_kv_heads=int(cfg.head_count_kv),
        head_dim=int(cfg.key_length),
    )


def _fixed_mixed_prompt_tokens(prompt_length: int) -> list[int]:
    if int(prompt_length) <= 0:
        raise ValueError("prompt_length must be positive")
    repeats = (int(prompt_length) + len(_MIXED_V1_TOKEN_UNIT) - 1) // len(_MIXED_V1_TOKEN_UNIT)
    return list((_MIXED_V1_TOKEN_UNIT * repeats)[: int(prompt_length)])


def _prompt_sha256(prompt_tokens: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(prompt_tokens, dtype=np.int32).tobytes()).hexdigest()


def _logit_row(result: object) -> np.ndarray:
    logits = np.asarray(getattr(result, "logits"), dtype=np.float32)
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 1:
        raise ValueError(f"expected one GGUF logit row, got shape {logits.shape}")
    return np.ascontiguousarray(logits, dtype=np.float32)


def _capture_cache_sample(session: object, *, tokens: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    layout = _cache_layout(session)
    scratch = getattr(session, "scratch")
    runtime = getattr(session, "runtime")
    max_positions = int(getattr(scratch, "max_positions"))
    rows = min(int(tokens), max_positions)
    shape = (rows, layout.num_kv_heads, layout.head_dim)
    nbytes = int(np.prod(shape)) * np.dtype(np.uint16).itemsize
    keys: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for layer_id in layout.full_layer_ids:
        key_buf, value_buf = scratch.full_cache(layer_id)
        key_bits = np.empty(shape, dtype=np.uint16)
        value_bits = np.empty(shape, dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(key_bits),
            DeviceBuffer(key_buf.ptr, nbytes),
            nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(value_bits),
            DeviceBuffer(value_buf.ptr, nbytes),
            nbytes,
            runtime=runtime,
        )
        keys.append(_bf16_bits_to_float32(key_bits))
        values.append(_bf16_bits_to_float32(value_bits))
    return keys, values


def _roundtrip_session_cache(
    session: object,
    spec: FormatSpec,
    *,
    start: int,
    rows: int,
    scale_dtype: str,
) -> None:
    if int(rows) <= 0:
        return
    storage = getattr(getattr(session, "kv_storage_dtype", None), "value", None)
    if storage is not None and storage != "bf16":
        raise ValueError("GGUF format emulation requires a BF16 resident cache")
    layout = _cache_layout(session)
    scratch = getattr(session, "scratch")
    runtime = getattr(session, "runtime")
    width = layout.num_kv_heads * layout.head_dim
    row_bytes = width * np.dtype(np.uint16).itemsize
    shape = (int(rows), layout.num_kv_heads, layout.head_dim)
    offset_bytes = int(start) * row_bytes
    nbytes = int(rows) * row_bytes
    for layer_id in layout.full_layer_ids:
        key_buf, value_buf = scratch.full_cache(layer_id)
        key_bits = np.empty(shape, dtype=np.uint16)
        value_bits = np.empty(shape, dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(key_bits),
            DeviceBuffer(key_buf.ptr + offset_bytes, nbytes),
            nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(value_bits),
            DeviceBuffer(value_buf.ptr + offset_bytes, nbytes),
            nbytes,
            runtime=runtime,
        )
        key, value = _roundtrip_pair(
            _bf16_bits_to_float32(key_bits),
            _bf16_bits_to_float32(value_bits),
            spec,
            scale_dtype=scale_dtype,
        )
        key_out = float_array_to_bf16_bits(key)
        value_out = float_array_to_bf16_bits(value)
        copy_host_to_device(
            DeviceBuffer(key_buf.ptr + offset_bytes, nbytes),
            host_array_ptr(key_out),
            nbytes,
            runtime=runtime,
        )
        copy_host_to_device(
            DeviceBuffer(value_buf.ptr + offset_bytes, nbytes),
            host_array_ptr(value_out),
            nbytes,
            runtime=runtime,
        )


def _run_loaded_session(
    session: object,
    *,
    prompt_tokens: Sequence[int],
    prompt_length: int,
    decode_steps: int,
    forced_input_ids: Sequence[int] | None,
    scale_dtype: str,
    emulated_spec: FormatSpec | None = None,
    sample_tokens: int = 0,
) -> tuple[dict[str, Any], tuple[list[np.ndarray], list[np.ndarray]] | None, int]:
    started = time.perf_counter()
    resolved_prompt = [int(token) for token in prompt_tokens]
    if len(resolved_prompt) != int(prompt_length):
        raise ValueError("prompt_tokens length must equal prompt_length")
    if forced_input_ids is not None and len(forced_input_ids) != int(decode_steps):
        raise ValueError("forced_input_ids must contain exactly decode_steps tokens")
    session.reset()
    seed = session.prefill(
        resolved_prompt,
        use_bulk=True,
        bulk_attention_mode="bulk",
        return_logits=True,
    )
    logits = [_logit_row(seed)]
    captured = _capture_cache_sample(session, tokens=min(prompt_length, sample_tokens)) if sample_tokens > 0 else None
    if emulated_spec is not None:
        _roundtrip_session_cache(
            session,
            emulated_spec,
            start=0,
            rows=prompt_length,
            scale_dtype=scale_dtype,
        )
    generated_ids: list[int] = []
    current = seed
    for offset in range(int(decode_steps)):
        input_id = int(current.token_id) if forced_input_ids is None else int(forced_input_ids[offset])
        current = session.step(
            input_id,
            position=int(prompt_length) + offset,
            return_logits=True,
        )
        generated_ids.append(int(current.token_id))
        logits.append(_logit_row(current))
        if emulated_spec is not None:
            _roundtrip_session_cache(
                session,
                emulated_spec,
                start=int(prompt_length) + offset,
                rows=1,
                scale_dtype=scale_dtype,
            )
    full_layers = len(_cache_layout(session).full_layer_ids)
    return (
        {
            "seed_token_id": int(seed.token_id),
            "generated_token_ids": generated_ids,
            "logits": logits,
            "finite_logits": bool(all(np.isfinite(row).all() for row in logits)),
            "elapsed_seconds": float(time.perf_counter() - started),
        },
        captured,
        full_layers,
    )


def _run_candidate_screen(
    session: object,
    *,
    prompt_tokens: Sequence[int],
    prompt_length: int,
    decode_steps: int,
    sample_tokens: int,
    scale_dtype: str,
    target_context_tokens: int,
    candidates: Sequence[FormatSpec],
    kl_threshold: float,
    top1_threshold: float,
    extra_budget_bytes: int,
) -> dict[str, Any]:
    layout = _cache_layout(session)
    reference, captured, full_layers = _run_loaded_session(
        session,
        prompt_tokens=prompt_tokens,
        prompt_length=prompt_length,
        decode_steps=decode_steps,
        forced_input_ids=None,
        scale_dtype=scale_dtype,
        sample_tokens=sample_tokens,
    )
    if captured is None or not captured[0]:
        raise RuntimeError("GGUF reference cache capture was empty")
    keys, values = captured
    forced_ids = (
        []
        if int(decode_steps) == 0
        else [
            int(reference["seed_token_id"]),
            *[int(token) for token in reference["generated_token_ids"][: int(decode_steps) - 1]],
        ]
    )
    baseline_memory = _format_memory_bytes(
        _candidate_catalog(layout.head_dim)["baseline_max"],
        tokens=target_context_tokens,
        full_layers=full_layers,
        num_kv_heads=layout.num_kv_heads,
        head_dim=layout.head_dim,
        scale_dtype=scale_dtype,
    )
    rows: list[dict[str, Any]] = []
    for spec in candidates:
        candidate_run, _, _ = _run_loaded_session(
            session,
            prompt_tokens=prompt_tokens,
            prompt_length=prompt_length,
            decode_steps=decode_steps,
            forced_input_ids=forced_ids,
            scale_dtype=scale_dtype,
            emulated_spec=spec,
        )
        gate = _compare_logits(reference["logits"], candidate_run["logits"])
        memory = _format_memory_bytes(
            spec,
            tokens=target_context_tokens,
            full_layers=full_layers,
            num_kv_heads=layout.num_kv_heads,
            head_dim=layout.head_dim,
            scale_dtype=scale_dtype,
        )
        rows.append(
            {
                **spec.to_json(),
                "emulation_semantics": (
                    "BF16 cache replaced by format reconstruction before decode; "
                    "current row round-tripped after its own attention"
                ),
                "logit_gate": gate,
                "quality_gate_passed": bool(
                    gate["mean_kl"] <= float(kl_threshold)
                    and gate["top1_agreement"] >= float(top1_threshold)
                ),
                "reconstruction": _aggregate_reconstruction(
                    keys,
                    values,
                    spec,
                    scale_dtype=scale_dtype,
                ),
                "target_context_memory": memory,
                "extra_bytes_over_baseline": int(memory["total_bytes"] - baseline_memory["total_bytes"]),
                "candidate": _compact_run(candidate_run),
            }
        )
    return {
        "reference": reference,
        "keys": keys,
        "values": values,
        "full_layers": full_layers,
        "baseline_memory": baseline_memory,
        "rows": rows,
        "forced_ids": forced_ids,
        "recommendation": _select_recommendation(rows, extra_budget_bytes=extra_budget_bytes),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.core.dtype import DType
    from hipengine.kvcache.policy import FixedPagedKVPolicy
    from hipengine.runtime import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    started = time.perf_counter()
    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8")
        if not compiler_version.strip():
            raise ValueError("compiler version file is empty")
    prompt_tokens = (
        _fixed_mixed_prompt_tokens(args.prompt_length)
        if args.prompt_profile == "mixed_v1"
        else [int(args.token_id)] * int(args.prompt_length)
    )
    prompt_digest = _prompt_sha256(prompt_tokens)
    candidates = _parse_candidates(args.candidates, head_dim=256)
    max_sequence_length = int(args.prompt_length) + int(args.decode_steps) + 2
    extra_budget_bytes = int(float(args.extra_budget_gib) * 1024**3)
    prefill_config = PrefillConfig(attn_aotriton_min_tokens=int(args.attn_aotriton_min_tokens))
    with Qwen35GGUFResidentSession(
        args.model,
        backend=args.backend,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=True,
        use_gemv_decode=True,
        prefill_config=prefill_config,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        layout = _cache_layout(session)
        if layout.head_dim != 256:
            candidates = _parse_candidates(args.candidates, head_dim=layout.head_dim)
        screen = _run_candidate_screen(
            session,
            prompt_tokens=prompt_tokens,
            prompt_length=args.prompt_length,
            decode_steps=args.decode_steps,
            sample_tokens=args.sample_tokens,
            scale_dtype=args.scale_dtype,
            target_context_tokens=args.target_context_tokens,
            candidates=candidates,
            kl_threshold=args.kl_threshold,
            top1_threshold=args.top1_threshold,
            extra_budget_bytes=extra_budget_bytes,
        )
        target_arch = session.runner.target_arch
        resolved_backend = session.backend
        layout = _cache_layout(session)
    gc.collect()
    elapsed_seconds = float(time.perf_counter() - started)
    keys = screen["keys"]
    values = screen["values"]
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=args.backend,
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16_reference_host_emulation",
        command=["python3", "scripts/qwen35_gguf_kv_format_ablation.py", *sys.argv[1:]],
        timing_protocol="setup-inclusive diagnostic wall; quality claim only",
        warmups=0,
        repetitions=1,
    )
    return {
        "schema": 1,
        "status": "diagnostic_complete",
        "performance_claim": False,
        "mode": "qwen35_gguf_kv_format_ablation",
        "provenance": provenance,
        "git": _git_provenance(),
        "model": str(args.model.resolve()),
        "backend": resolved_backend,
        "target_arch": target_arch,
        "workload": {
            "prompt_length": int(args.prompt_length),
            "decode_steps": int(args.decode_steps),
            "prompt_profile": args.prompt_profile,
            "token_id": None if args.prompt_profile == "mixed_v1" else int(args.token_id),
            "prompt_token_sha256": prompt_digest,
            "distinct_prompt_tokens": int(len(set(prompt_tokens))),
        },
        "screen_budget": {
            "profile": "gguf_fast_accuracy_v1" if args.fast_accuracy_screen else "custom",
            "wall_time_budget_seconds": float(args.wall_time_budget_seconds),
            "elapsed_seconds": elapsed_seconds,
            "met": bool(elapsed_seconds <= float(args.wall_time_budget_seconds)),
            "candidate_count": len(screen["rows"]),
            "model_setup": "one BF16 GGUF resident-weight session reset and reused across reference and candidates",
        },
        "shape": {
            "full_attention_layers": int(screen["full_layers"]),
            "full_attention_layer_ids": list(layout.full_layer_ids),
            "num_kv_heads": int(layout.num_kv_heads),
            "head_dim": int(layout.head_dim),
            "scale_dtype": args.scale_dtype,
            "sample_tokens": min(int(args.sample_tokens), int(args.prompt_length)),
        },
        "quality_thresholds": {
            "kl_mean_max": float(args.kl_threshold),
            "top1_agreement_min": float(args.top1_threshold),
        },
        "target_memory": {
            "context_tokens": int(args.target_context_tokens),
            "extra_budget_bytes": extra_budget_bytes,
            "baseline": screen["baseline_memory"],
        },
        "reference": _compact_run(screen["reference"]),
        "captured_distribution": {
            "key": _distribution_summary(
                np.concatenate([item.reshape(-1, layout.head_dim) for item in keys], axis=0)
            ),
            "value": _distribution_summary(
                np.concatenate([item.reshape(-1, layout.head_dim) for item in values], axis=0)
            ),
        },
        "candidates": screen["rows"],
        "recommendation": screen["recommendation"],
        "elapsed_seconds": elapsed_seconds,
        "notes": [
            "Diagnostic only: host emulation ranks formats without native candidate kernels.",
            "All candidates use the identical Q4_K_M weights, BF16 reference, prompt, and teacher-forced token history.",
            "The current decode row is BF16 during its own attention and round-tripped immediately afterward.",
            "The group32 row matches llama.cpp Q8_0 storage granularity but not its direct integer-dot attention arithmetic.",
            "A candidate must pass a native long-context gate before support or performance claims.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--fast-accuracy-screen", action="store_true")
    parser.add_argument("--prompt-profile", choices=("repeated_token", "mixed_v1"), default="mixed_v1")
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--sample-tokens", type=int, default=512)
    parser.add_argument("--candidates", default=FAST_GGUF_CANDIDATES)
    parser.add_argument("--scale-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--target-context-tokens", type=int, default=262144)
    parser.add_argument("--extra-budget-gib", type=float, default=1.0)
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--wall-time-budget-seconds", type=float, default=600.0)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--attn-aotriton-min-tokens", type=int, default=512)
    parser.add_argument("--backend", choices=("auto", "hip_gfx1100", "hip_gfx1151"), default="hip_gfx1100")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.fast_accuracy_screen:
        args.prompt_length = 512
        args.decode_steps = 8
        args.sample_tokens = 512
        args.candidates = FAST_GGUF_CANDIDATES
        args.prompt_profile = "mixed_v1"
        args.wall_time_budget_seconds = 600.0
    for name in ("prompt_length", "sample_tokens", "target_context_tokens"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.decode_steps) < 0 or float(args.extra_budget_gib) < 0.0:
        raise ValueError("decode steps and extra budget must be non-negative")
    if float(args.wall_time_budget_seconds) <= 0.0:
        raise ValueError("wall-time budget must be positive")
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
