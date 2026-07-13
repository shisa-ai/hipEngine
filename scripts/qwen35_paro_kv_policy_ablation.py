#!/usr/bin/env python3
# ruff: noqa: E402
"""Screen mixed-precision and residual-cache PARO KV policies.

This builds on ``qwen35_paro_kv_format_ablation.py``.  Candidate policies are
emulated in a BF16 resident cache so layer/head sensitivity and bounded BF16
sink/recent-token residuals can be ranked before runtime plumbing.  It is a
correctness diagnostic, not a performance claim or a native mixed-cache gate.
"""

from __future__ import annotations

import argparse
import gc
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

from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, host_array_ptr
from hipengine.kvcache import resolve_kv_policy
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_paro_bench import _prompt_tokens
from scripts.qwen35_paro_int8_kv_quality_sweep import _compare_logits, _read_logits
from scripts.qwen35_paro_kv_format_ablation import (
    DEFAULT_MODEL,
    FormatSpec,
    _bf16_bits_to_float32,
    _format_memory_bytes,
    _git_provenance,
    _read_compiler_version,
    _roundtrip_pair,
    _run_session,
)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    format_spec: FormatSpec
    bf16_layer_indices: tuple[int, ...] = ()
    bf16_head_indices: tuple[int, ...] = ()
    sink_tokens: int = 0
    recent_tokens: int = 0

    def __post_init__(self) -> None:
        if any(item < 0 for item in self.bf16_layer_indices):
            raise ValueError("BF16 layer indices must be non-negative")
        if any(item < 0 for item in self.bf16_head_indices):
            raise ValueError("BF16 head indices must be non-negative")
        if self.sink_tokens < 0 or self.recent_tokens < 0:
            raise ValueError("sink/recent windows must be non-negative")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "format": self.format_spec.to_json(),
            "bf16_layer_indices": list(self.bf16_layer_indices),
            "bf16_head_indices": list(self.bf16_head_indices),
            "sink_tokens": int(self.sink_tokens),
            "recent_tokens": int(self.recent_tokens),
        }


def _parse_index_list(text: str) -> tuple[int, ...]:
    value = str(text).strip().lower()
    if value in {"", "none", "empty", "-"}:
        return ()
    indices = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if any(item < 0 for item in indices):
        raise ValueError("policy indices must be non-negative")
    return indices


def _apply_policy_arrays(
    key: np.ndarray,
    value: np.ndarray,
    policy: PolicySpec,
    *,
    layer_index: int,
    active_tokens: int,
    scale_dtype: str,
    start_token: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    source_k = np.asarray(key, dtype=np.float32)
    source_v = np.asarray(value, dtype=np.float32)
    if source_k.shape != source_v.shape or source_k.ndim != 3:
        raise ValueError("K/V policy arrays must share [tokens, heads, dim] shape")
    if int(layer_index) in policy.bf16_layer_indices:
        return source_k.copy(), source_v.copy()
    out_k, out_v = _roundtrip_pair(source_k, source_v, policy.format_spec, scale_dtype=scale_dtype)
    heads = [item for item in policy.bf16_head_indices if item < source_k.shape[1]]
    if len(heads) != len(policy.bf16_head_indices):
        raise ValueError("BF16 head index exceeds K/V head count")
    if heads:
        out_k[:, heads, :] = source_k[:, heads, :]
        out_v[:, heads, :] = source_v[:, heads, :]
    positions = np.arange(int(start_token), int(start_token) + source_k.shape[0], dtype=np.int64)
    preserve = positions < int(policy.sink_tokens)
    if policy.recent_tokens > 0:
        preserve |= positions >= max(0, int(active_tokens) - int(policy.recent_tokens))
    if np.any(preserve):
        out_k[preserve] = source_k[preserve]
        out_v[preserve] = source_v[preserve]
    return np.ascontiguousarray(out_k), np.ascontiguousarray(out_v)


def _policy_memory_bytes(
    policy: PolicySpec,
    *,
    tokens: int,
    full_layers: int,
    num_kv_heads: int,
    head_dim: int,
    scale_dtype: str,
) -> dict[str, int]:
    layers = int(full_layers)
    heads = int(num_kv_heads)
    bf16_layers = len(set(policy.bf16_layer_indices))
    bf16_heads = len(set(policy.bf16_head_indices))
    if bf16_layers > layers or any(item >= layers for item in policy.bf16_layer_indices):
        raise ValueError("BF16 layer index exceeds full-attention layer count")
    if bf16_heads > heads or any(item >= heads for item in policy.bf16_head_indices):
        raise ValueError("BF16 head index exceeds K/V head count")
    base = _format_memory_bytes(
        policy.format_spec,
        tokens=tokens,
        full_layers=layers,
        num_kv_heads=heads,
        head_dim=head_dim,
        scale_dtype=scale_dtype,
    )
    format_one_layer = _format_memory_bytes(
        policy.format_spec,
        tokens=tokens,
        full_layers=1,
        num_kv_heads=heads,
        head_dim=head_dim,
        scale_dtype=scale_dtype,
    )["total_bytes"]
    bf16_one_layer = 2 * int(tokens) * heads * int(head_dim) * 2
    full_layer_delta = bf16_layers * (bf16_one_layer - format_one_layer)
    quantized_layers = layers - bf16_layers
    format_one_head = _format_memory_bytes(
        policy.format_spec,
        tokens=tokens,
        full_layers=1,
        num_kv_heads=1,
        head_dim=head_dim,
        scale_dtype=scale_dtype,
    )["total_bytes"]
    bf16_one_head = 2 * int(tokens) * int(head_dim) * 2
    head_delta = quantized_layers * bf16_heads * (bf16_one_head - format_one_head)
    quantized_heads = heads - bf16_heads
    sink = min(int(tokens), int(policy.sink_tokens))
    recent = min(max(0, int(tokens) - sink), int(policy.recent_tokens))
    residual_rows = sink + recent
    residual_bytes = residual_rows * quantized_layers * quantized_heads * int(head_dim) * 2 * 2
    total = int(base["total_bytes"] + full_layer_delta + head_delta + residual_bytes)
    return {
        "base_format_bytes": int(base["total_bytes"]),
        "full_layer_replacement_bytes": int(full_layer_delta),
        "head_replacement_bytes": int(head_delta),
        "residual_rows": int(residual_rows),
        "residual_bytes": int(residual_bytes),
        "total_bytes": total,
        "bf16_full_layers": bf16_layers,
        "bf16_heads_per_quantized_layer": bf16_heads,
    }


def _roundtrip_session_policy_range(
    session: Qwen35ParoResidentSession,
    policy: PolicySpec,
    *,
    start: int,
    rows: int,
    active_tokens: int,
    scale_dtype: str,
) -> None:
    if rows <= 0:
        return
    width = int(session.config.num_key_value_heads) * int(session.config.head_dim)
    row_bytes = width * np.dtype(np.uint16).itemsize
    shape = (int(rows), int(session.config.num_key_value_heads), int(session.config.head_dim))
    offset_bytes = int(start) * row_bytes
    nbytes = int(rows) * row_bytes
    for layer_index, layer_id in enumerate(sorted(session.full_caches)):
        key_tensor, value_tensor, key_buf, value_buf = session.full_caches[layer_id]
        if key_tensor.dtype.value != "bf16" or value_tensor.dtype.value != "bf16":
            raise ValueError("policy emulation requires BF16 resident caches")
        key_bits = np.empty(shape, dtype=np.uint16)
        value_bits = np.empty(shape, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(key_bits), DeviceBuffer(key_buf.ptr + offset_bytes, nbytes), nbytes, runtime=session.runtime)
        copy_device_to_host(host_array_ptr(value_bits), DeviceBuffer(value_buf.ptr + offset_bytes, nbytes), nbytes, runtime=session.runtime)
        key, value = _apply_policy_arrays(
            _bf16_bits_to_float32(key_bits),
            _bf16_bits_to_float32(value_bits),
            policy,
            layer_index=layer_index,
            active_tokens=active_tokens,
            scale_dtype=scale_dtype,
            start_token=start,
        )
        key_out = float_array_to_bf16_bits(key)
        value_out = float_array_to_bf16_bits(value)
        copy_host_to_device(DeviceBuffer(key_buf.ptr + offset_bytes, nbytes), host_array_ptr(key_out), nbytes, runtime=session.runtime)
        copy_host_to_device(DeviceBuffer(value_buf.ptr + offset_bytes, nbytes), host_array_ptr(value_out), nbytes, runtime=session.runtime)


def _run_policy_session(
    *,
    runner: Qwen35ParoNextTokenRunner,
    model: Path,
    prompt_length: int,
    decode_steps: int,
    token_id: int,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
    prefill_config: PrefillConfig,
    scale_dtype: str,
    forced_input_ids: Sequence[int],
    policy: PolicySpec,
) -> dict[str, Any]:
    resolved = resolve_kv_policy("bf16", block_size=256)
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=int(prompt_length) + int(decode_steps) + 2,
        max_layers=max_layers,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        prefill_config=prefill_config,
        kv_policy=resolved.create_policy(),
    ) as session:
        prompt_tokens = _prompt_tokens(model, "Hello", token_id, prompt_length)
        seed = session.prefill_native(prompt_tokens, sample=True)
        if seed is None:
            raise RuntimeError("native prefill did not produce a seed")
        logits = [_read_logits(session)]
        _roundtrip_session_policy_range(
            session,
            policy,
            start=0,
            rows=prompt_length,
            active_tokens=prompt_length,
            scale_dtype=scale_dtype,
        )
        generated: list[int] = []
        for offset in range(decode_steps):
            current = session.step(int(forced_input_ids[offset]), position=prompt_length + offset, sample=True)
            if current is None:
                raise RuntimeError(f"decode did not produce token {offset}")
            generated.append(int(current.token_id))
            logits.append(_read_logits(session))
            active = prompt_length + offset + 1
            if policy.recent_tokens > 0:
                falling = active - int(policy.recent_tokens) - 1
                if falling >= int(policy.sink_tokens) and falling >= 0:
                    _roundtrip_session_policy_range(
                        session,
                        policy,
                        start=falling,
                        rows=1,
                        active_tokens=active,
                        scale_dtype=scale_dtype,
                    )
            else:
                _roundtrip_session_policy_range(
                    session,
                    policy,
                    start=prompt_length + offset,
                    rows=1,
                    active_tokens=active,
                    scale_dtype=scale_dtype,
                )
        result = {
            "seed_token_id": int(seed.token_id),
            "generated_token_ids": generated,
            "finite_logits": bool(all(np.isfinite(item).all() for item in logits)),
            "logits": logits,
        }
    gc.collect()
    return result


def _base_policy_catalog(*, full_layers: int, num_kv_heads: int, head_dim: int, small_window: int, large_window: int) -> list[PolicySpec]:
    baseline = FormatSpec("baseline_max", k_group_size=head_dim, v_group_size=head_dim)
    group64 = FormatSpec("group64", k_group_size=64, v_group_size=64)
    policies = [PolicySpec("baseline_max", baseline), PolicySpec("group64", group64)]
    policies.extend(PolicySpec(f"bf16_layer_{idx}", baseline, bf16_layer_indices=(idx,)) for idx in range(full_layers))
    for count in range(1, min(4, full_layers) + 1):
        policies.append(PolicySpec(f"bf16_prefix_{count}", baseline, bf16_layer_indices=tuple(range(count))))
        policies.append(PolicySpec(f"bf16_suffix_{count}", baseline, bf16_layer_indices=tuple(range(full_layers - count, full_layers))))
    policies.extend(PolicySpec(f"bf16_head_{idx}", baseline, bf16_head_indices=(idx,)) for idx in range(num_kv_heads))
    policies.extend(
        (
            PolicySpec(f"sink_{small_window}", baseline, sink_tokens=small_window),
            PolicySpec(f"sink_{large_window}", baseline, sink_tokens=large_window),
            PolicySpec(f"recent_{small_window}", baseline, recent_tokens=small_window),
            PolicySpec(f"recent_{large_window}", baseline, recent_tokens=large_window),
            PolicySpec(f"sink_recent_{small_window}", baseline, sink_tokens=small_window, recent_tokens=small_window),
            PolicySpec(f"group64_recent_{small_window}", group64, recent_tokens=small_window),
            PolicySpec(f"group64_sink_recent_{small_window}", group64, sink_tokens=small_window, recent_tokens=small_window),
        )
    )
    return policies


def _select_policy_recommendation(rows: Sequence[dict[str, Any]], *, extra_budget_bytes: int) -> dict[str, Any]:
    fit = [row for row in rows if int(row["extra_bytes_over_baseline"]) <= int(extra_budget_bytes)]
    if not fit:
        return {"name": None, "fit_candidates": [], "reason": "no policy fits extra-byte budget"}
    best = min(
        fit,
        key=lambda row: (
            not bool(row["quality_gate_passed"]),
            -float(row["logit_gate"]["top1_agreement"]),
            float(row["logit_gate"]["mean_kl"]),
            int(row["extra_bytes_over_baseline"]),
        ),
    )
    return {
        "name": str(best["name"]),
        "fit_candidates": [str(row["name"]) for row in fit],
        "quality_gate_passed": bool(best["quality_gate_passed"]),
        "mean_kl": float(best["logit_gate"]["mean_kl"]),
        "top1_agreement": float(best["logit_gate"]["top1_agreement"]),
        "extra_bytes_over_baseline": int(best["extra_bytes_over_baseline"]),
        "reason": "pass both gates first, then highest top-1 and lowest mean KL among policies fitting budget",
    }


def _run_policy(
    policy: PolicySpec,
    *,
    runner: Qwen35ParoNextTokenRunner,
    args: argparse.Namespace,
    compiler_version: str | None,
    prefill_config: PrefillConfig,
    forced_ids: Sequence[int],
    reference_logits: Sequence[np.ndarray],
    baseline_bytes: int,
    full_layers: int,
) -> dict[str, Any]:
    candidate = _run_policy_session(
        runner=runner,
        model=args.model,
        prompt_length=args.prompt_length,
        decode_steps=args.decode_steps,
        token_id=args.token_id,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        prefill_config=prefill_config,
        scale_dtype=args.scale_dtype,
        forced_input_ids=forced_ids,
        policy=policy,
    )
    gate = _compare_logits(reference_logits, candidate["logits"])
    memory = _policy_memory_bytes(
        policy,
        tokens=args.target_context_tokens,
        full_layers=full_layers,
        num_kv_heads=int(runner.config.num_key_value_heads),
        head_dim=int(runner.config.head_dim),
        scale_dtype=args.scale_dtype,
    )
    return {
        **policy.to_json(),
        "logit_gate": gate,
        "quality_gate_passed": bool(gate["mean_kl"] <= args.kl_threshold and gate["top1_agreement"] >= args.top1_threshold),
        "target_context_memory": memory,
        "extra_bytes_over_baseline": int(memory["total_bytes"] - baseline_bytes),
        "candidate": {key: value for key, value in candidate.items() if key != "logits"},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    compiler_version = _read_compiler_version(args.compiler_version_file)
    prefill_config = PrefillConfig(attn_aotriton_min_tokens=int(args.attn_aotriton_min_tokens))
    runner = Qwen35ParoNextTokenRunner(
        args.model,
        shared_expert_format=None if args.shared_expert_format == "auto" else args.shared_expert_format,
        backend=args.backend,
    )
    reference, _, full_layers = _run_session(
        runner=runner,
        model=args.model,
        prompt_length=args.prompt_length,
        decode_steps=args.decode_steps,
        token_id=args.token_id,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        prefill_config=prefill_config,
        storage="bf16",
        scale_dtype=args.scale_dtype,
        forced_input_ids=None,
    )
    forced_ids = [int(reference["seed_token_id"]), *[int(item) for item in reference["generated_token_ids"][: max(0, args.decode_steps - 1)]]]
    baseline_format = FormatSpec("baseline_max", k_group_size=int(runner.config.head_dim), v_group_size=int(runner.config.head_dim))
    baseline_bytes = _format_memory_bytes(
        baseline_format,
        tokens=args.target_context_tokens,
        full_layers=full_layers,
        num_kv_heads=int(runner.config.num_key_value_heads),
        head_dim=int(runner.config.head_dim),
        scale_dtype=args.scale_dtype,
    )["total_bytes"]
    policies = _base_policy_catalog(
        full_layers=full_layers,
        num_kv_heads=int(runner.config.num_key_value_heads),
        head_dim=int(runner.config.head_dim),
        small_window=args.small_window,
        large_window=args.large_window,
    )
    if args.policies == "custom":
        if args.custom_format == "baseline_max":
            custom_format = baseline_format
        else:
            group_size = int(args.custom_format.removeprefix("group"))
            custom_format = FormatSpec(
                args.custom_format,
                k_group_size=group_size,
                v_group_size=group_size,
            )
        policies = [
            PolicySpec(
                "custom",
                custom_format,
                bf16_layer_indices=_parse_index_list(args.custom_bf16_layers),
                bf16_head_indices=_parse_index_list(args.custom_bf16_heads),
                sink_tokens=int(args.custom_sink_tokens),
                recent_tokens=int(args.custom_recent_tokens),
            )
        ]
    elif args.policies != "default":
        requested = {item.strip() for item in args.policies.split(",") if item.strip()}
        known = {item.name for item in policies}
        if not requested or requested - known:
            raise ValueError(f"unknown/empty policies: {sorted(requested - known)}")
        policies = [item for item in policies if item.name in requested]
    rows = [
        _run_policy(
            policy,
            runner=runner,
            args=args,
            compiler_version=compiler_version,
            prefill_config=prefill_config,
            forced_ids=forced_ids,
            reference_logits=reference["logits"],
            baseline_bytes=baseline_bytes,
            full_layers=full_layers,
        )
        for policy in policies
    ]

    if args.policies == "default":
        sensitivity = [row for row in rows if row["name"].startswith("bf16_layer_")]
        ranked_layers = [
            int(row["bf16_layer_indices"][0])
            for row in sorted(
                sensitivity,
                key=lambda row: (-float(row["logit_gate"]["top1_agreement"]), float(row["logit_gate"]["mean_kl"])),
            )
        ]
        extras: list[PolicySpec] = []
        for count in range(2, min(4, full_layers) + 1):
            selected = tuple(sorted(ranked_layers[:count]))
            extras.append(PolicySpec(f"bf16_sensitive_{count}", baseline_format, bf16_layer_indices=selected))
        group64 = FormatSpec("group64", k_group_size=64, v_group_size=64)
        for count in range(1, min(2, full_layers) + 1):
            selected = tuple(sorted(ranked_layers[:count]))
            extras.append(PolicySpec(f"group64_bf16_sensitive_{count}", group64, bf16_layer_indices=selected))
        for policy in extras:
            rows.append(
                _run_policy(
                    policy,
                    runner=runner,
                    args=args,
                    compiler_version=compiler_version,
                    prefill_config=prefill_config,
                    forced_ids=forced_ids,
                    reference_logits=reference["logits"],
                    baseline_bytes=baseline_bytes,
                    full_layers=full_layers,
                )
            )
    extra_budget_bytes = int(float(args.extra_budget_gib) * 1024**3)
    recommendation = _select_policy_recommendation(rows, extra_budget_bytes=extra_budget_bytes)
    return {
        "schema": 1,
        "status": "diagnostic_complete",
        "performance_claim": False,
        "mode": "qwen35_paro_kv_policy_ablation",
        "provenance": _git_provenance(),
        "model": str(args.model),
        "backend": runner.backend,
        "target_arch": runner.target_arch,
        "workload": {"prompt_length": int(args.prompt_length), "decode_steps": int(args.decode_steps), "token_id": int(args.token_id)},
        "shape": {
            "max_layers": int(args.max_layers),
            "full_attention_layers": int(full_layers),
            "num_kv_heads": int(runner.config.num_key_value_heads),
            "head_dim": int(runner.config.head_dim),
            "scale_dtype": args.scale_dtype,
        },
        "quality_thresholds": {"kl_mean_max": float(args.kl_threshold), "top1_agreement_min": float(args.top1_threshold)},
        "target_memory": {
            "context_tokens": int(args.target_context_tokens),
            "extra_budget_bytes": extra_budget_bytes,
            "baseline_per_token_head_bytes": int(baseline_bytes),
        },
        "reference": {key: value for key, value in reference.items() if key != "logits"},
        "policies": rows,
        "recommendation": recommendation,
        "elapsed_seconds": float(time.perf_counter() - started),
        "notes": [
            "Diagnostic only: policies are emulated in BF16 caches and require native validation.",
            "Layer/head BF16 entries model primary mixed storage; sink/recent entries include an extra BF16 residual side-cache over the INT8 primary.",
            "The current-token row remains BF16 during its own emulated attention, matching the format-screen limitation.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--policies", default="default", help="default, custom, or comma-separated built-in names")
    parser.add_argument(
        "--custom-format",
        choices=("baseline_max", "group32", "group64"),
        default="baseline_max",
    )
    parser.add_argument("--custom-bf16-layers", default="none", help="Full-attention layer ordinals for --policies custom")
    parser.add_argument("--custom-bf16-heads", default="none", help="KV head indices for --policies custom")
    parser.add_argument("--custom-sink-tokens", type=int, default=0)
    parser.add_argument("--custom-recent-tokens", type=int, default=0)
    parser.add_argument("--small-window", type=int, default=64)
    parser.add_argument("--large-window", type=int, default=128)
    parser.add_argument("--scale-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--target-context-tokens", type=int, default=262144)
    parser.add_argument("--extra-budget-gib", type=float, default=1.0)
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--attn-aotriton-min-tokens", type=int, default=512)
    parser.add_argument("--backend", choices=("auto", "hip_gfx1100", "hip_gfx1151"), default="hip_gfx1100")
    parser.add_argument("--shared-expert-format", choices=("auto", "legacy_fp16", "packed_paro_w4"), default="packed_paro_w4")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    for name in ("prompt_length", "small_window", "large_window", "target_context_tokens"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (
        args.decode_steps < 0
        or args.extra_budget_gib < 0.0
        or args.custom_sink_tokens < 0
        or args.custom_recent_tokens < 0
    ):
        raise ValueError("decode steps, extra budget, and custom windows must be non-negative")
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
