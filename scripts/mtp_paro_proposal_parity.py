#!/usr/bin/env python3
"""Compare PARO MTP native proposal output with independent torch/W8 references.

This is a correctness-only B1 diagnostic. It captures the target's final
output-normalized BF16 prompt rows, runs the target-contract native proposer,
then replays the same shifted prompt through the torch sidecar reference. The
borrowed target W8A16 head is checked against an independent CPU quantize/dot
reference over the full vocabulary.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import w8a16_linear_bf16_f32_out
from hipengine.loading import load_weight_index, qwen35_paro_config_from_hf
from hipengine.loading.qwen35_paro import normalize_qwen35_weight_name
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from hipengine.speculative.mtp_native import NativeMtpChainProposer, NativeMtpW8A16Head
from scripts.mtp_prompt_suite_economics import _load_prompt_encoder, _load_prompt_suite


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def _stable_topk(values: np.ndarray, k: int) -> list[int]:
    ids = np.arange(values.size, dtype=np.int64)
    return [int(x) for x in np.lexsort((ids, -np.asarray(values, dtype=np.float32)))[: int(k)]]


def _cpu_w8_logits(head_f16: np.ndarray, hidden_f32: np.ndarray, *, chunk_rows: int = 8192) -> np.ndarray:
    hidden = np.asarray(hidden_f32, dtype=np.float32).reshape(-1)
    logits = np.empty((head_f16.shape[0],), dtype=np.float32)
    for start in range(0, int(head_f16.shape[0]), int(chunk_rows)):
        end = min(start + int(chunk_rows), int(head_f16.shape[0]))
        weight = np.asarray(head_f16[start:end], dtype=np.float32)
        scale = np.maximum(np.max(np.abs(weight), axis=1), np.float32(1.0e-8)) / np.float32(127.0)
        quantized = np.clip(np.rint(weight / scale[:, None]), -127, 127).astype(np.int8)
        logits[start:end] = (quantized.astype(np.float32) @ hidden) * scale
    return logits


_TORCH_REFERENCE_CACHE: dict[tuple[str, str], tuple[Any, Any, Any, dict[str, Any]]] = {}


def _load_torch_reference(model: Path, device: Any):
    from safetensors.torch import load_file

    cache_key = (str(model.resolve()), str(device))
    cached = _TORCH_REFERENCE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    index = load_weight_index(model)
    cfg = qwen35_paro_config_from_hf(index.config)
    infos = {normalize_qwen35_weight_name(name): info for name, info in index.tensors.items()}
    embed_info = infos["embed_tokens.weight"]
    head_info = infos["lm_head.weight"]
    with safe_open(str(embed_info.shard_path), framework="pt", device=str(device)) as handle:
        embed = handle.get_tensor(embed_info.name).contiguous()
    with safe_open(str(head_info.shard_path), framework="pt", device=str(device)) as handle:
        lm_head = handle.get_tensor(head_info.name).contiguous()
    weights = {
        name: tensor.to(device=device).contiguous()
        for name, tensor in load_file(str(model / "mtp-bf16.safetensors"), device=str(device)).items()
    }
    loaded = (cfg, embed, lm_head, weights)
    _TORCH_REFERENCE_CACHE[cache_key] = loaded
    return loaded


def _torch_w8_logits(head: Any, hidden: Any, *, chunk_rows: int = 8192) -> Any:
    """Independent chunked torch reference for target per-row symmetric W8."""

    import torch

    hidden_f32 = hidden.float().reshape(-1)
    output = torch.empty((int(head.shape[0]),), dtype=torch.float32, device=head.device)
    for start in range(0, int(head.shape[0]), int(chunk_rows)):
        end = min(start + int(chunk_rows), int(head.shape[0]))
        weight = head[start:end].float()
        scale = torch.clamp(weight.abs().amax(dim=1), min=1.0e-8) / 127.0
        quantized = torch.clamp(torch.round(weight / scale[:, None]), -127, 127).to(torch.int8)
        output[start:end] = torch.mv(quantized.float(), hidden_f32) * scale
    return output


def run(
    *,
    model: Path,
    prompts_file: Path,
    prompt_name: str,
    prompt_render: str,
    backend: str,
    tail_tokens: int = 4,
    reference_device: str = "cpu",
    root_token_override: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    suite = _load_prompt_suite(prompts_file)
    prompt = next((row for row in suite["prompts"] if row["name"] == prompt_name), None)
    if prompt is None:
        raise ValueError(f"unknown prompt {prompt_name!r}")
    encoder = _load_prompt_encoder(model, prompt_render)
    encoded = encoder.encode(prompt["prompt"])
    prompt_tokens = [int(x) for x in encoded.token_ids]
    if int(tail_tokens) <= 0:
        raise ValueError("tail_tokens must be positive")
    tail_start = max(0, len(prompt_tokens) - int(tail_tokens))

    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    target_hidden_bits: np.ndarray
    native_hidden_bits: np.ndarray
    native_logits: np.ndarray
    native_token: int
    capture_buf = None
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=len(prompt_tokens) + 8,
        max_batch_size=2,
    ) as session:
        hidden = int(session.config.hidden_size)
        capture_host = np.empty((len(prompt_tokens), hidden), dtype=np.uint16)
        capture_buf = malloc(capture_host.nbytes, runtime=session.runtime)
        capture = Tensor.from_handle(capture_buf.ptr, capture_host.shape, DType.BF16, Device("hip", 0))
        try:
            result = None
            capture_layer_id = int(session.layer_limit) - 1
            for position, token in enumerate(prompt_tokens):
                result = session.step_with_hidden_taps(
                    token,
                    position=position,
                    capture_layer_ids=(capture_layer_id,),
                    capture_hidden_concat=capture,
                    capture_row=position,
                    sample=position == len(prompt_tokens) - 1,
                    capture_final_hidden_bf16=capture,
                )
            if result is None:
                raise RuntimeError("target prompt did not produce a root token")
            sampled_root_token = int(result.token_id)
            root_token = (
                sampled_root_token
                if root_token_override is None
                else int(root_token_override)
            )
            if root_token < 0 or root_token >= int(session.vocab_size):
                raise RuntimeError(
                    f"target root token {root_token} outside vocab; sampled={sampled_root_token}"
                )
            scoring_head = NativeMtpW8A16Head(
                weight_int8_ptr=int(session.lm_head_weight.tensor.ptr),
                scale_f32_ptr=int(session.lm_head_scale.tensor.ptr),
                vocab_size=int(session.vocab_size),
                threads=int(session.lm_head_threads),
                owner=session,
            )
            with NativeMtpChainProposer(
                model,
                max_positions=len(prompt_tokens) + 8,
                max_mtp_tokens=len(prompt_tokens) + 8,
                runtime=session.runtime,
                compiler_version=session.compiler_version,
                scoring_head=scoring_head,
            ) as proposer:
                proposer.reset()
                for idx in range(tail_start, len(prompt_tokens)):
                    input_token = (
                        prompt_tokens[idx + 1]
                        if idx + 1 < len(prompt_tokens)
                        else root_token
                    )
                    proposer.advance(
                        input_token=int(input_token),
                        target_hidden_ptr=(
                            int(capture_buf.ptr) + idx * hidden * DType.BF16.itemsize
                        ),
                        position=idx + 1,
                        read_expert_topk=True,
                        read_lm_head_value=True,
                    )
                native_hidden_bits = proposer.read_final_hidden_bf16()
                native_token = int(proposer.current.token)
                w8a16_linear_bf16_f32_out(
                    proposer.final_hidden_buf.ptr,
                    int(session.lm_head_weight.tensor.ptr),
                    int(session.lm_head_scale.tensor.ptr),
                    session.lm_logits.ptr,
                    1,
                    hidden,
                    int(session.vocab_size),
                    threads=int(session.lm_head_threads),
                    library=session.libraries["w8a16"],
                    runtime=session.runtime,
                )
                session.runtime.device_synchronize()
                native_logits = np.empty((int(session.vocab_size),), dtype=np.float32)
                copy_device_to_host(
                    host_array_ptr(native_logits),
                    DeviceBuffer(session.lm_logits.ptr, native_logits.nbytes),
                    native_logits.nbytes,
                    runtime=session.runtime,
                )
            copy_device_to_host(
                host_array_ptr(capture_host),
                DeviceBuffer(capture_buf.ptr, capture_host.nbytes),
                capture_host.nbytes,
                runtime=session.runtime,
            )
            target_hidden_bits = capture_host.copy()
        finally:
            if capture_buf is not None:
                free(capture_buf, runtime=session.runtime)

    print(
        f"[parity] target/native capture complete root={root_token} native={native_token}",
        flush=True,
    )
    import torch
    from scripts.mtp_torch_proposal_smoke import _advance, _rope_tables

    device = torch.device(reference_device)
    cfg, embed, lm_head, weights = _load_torch_reference(model, device)
    cos, sin = _rope_tables(len(prompt_tokens) + 8, int(cfg.rotary_dim or cfg.head_dim), float(cfg.rope_theta), device=device)
    state = None
    for idx in range(tail_start, len(prompt_tokens)):
        input_token = prompt_tokens[idx + 1] if idx + 1 < len(prompt_tokens) else root_token
        hidden_row = torch.from_numpy(_bf16_bits_to_f32(target_hidden_bits[idx : idx + 1]).copy()).to(
            device=device,
            dtype=torch.bfloat16,
        )
        state = _advance(
            token=int(input_token),
            target_hidden=hidden_row,
            state=state,
            embed_tokens=embed,
            lm_head=lm_head[:1],
            weights=weights,
            position=idx + 1,
            cfg=cfg,
            cos=cos,
            sin=sin,
        )
    assert state is not None
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("[parity] torch sidecar tail replay complete", flush=True)
    reference_hidden_f32 = state.hidden.float().cpu().numpy()
    native_hidden_f32 = _bf16_bits_to_f32(native_hidden_bits)

    reference_w8_logits = _torch_w8_logits(lm_head, state.hidden)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("[parity] torch W8 full-vocab scoring complete", flush=True)
    reference_w8_logits_host = reference_w8_logits.cpu().numpy()

    native_top8 = _stable_topk(native_logits, 8)
    reference_top8 = _stable_topk(reference_w8_logits_host, 8)
    hidden_delta = np.abs(native_hidden_f32 - reference_hidden_f32)
    return {
        "schema": "hipengine.paro_mtp_proposal_parity.v1",
        "status": "passed" if native_token == native_top8[0] and native_top8 == reference_top8 else "mismatch",
        "performance_claim": False,
        "model": str(model),
        "prompt": {
            "name": prompt_name,
            "category": prompt.get("category"),
            "split": prompt.get("split"),
            "render": prompt_render,
            "tokens": len(prompt_tokens),
            "tail_replay_tokens": len(prompt_tokens) - tail_start,
            "tail_start_position": tail_start,
        },
        "root_token": root_token,
        "sampled_root_token": sampled_root_token,
        "root_token_override": root_token_override,
        "native_token": native_token,
        "native_top8": native_top8,
        "torch_w8_reference_top8": reference_top8,
        "fused_native_top1_matches_materialized_native_w8": native_token == native_top8[0],
        "native_reference_top8_equal": native_top8 == reference_top8,
        "native_reference_top1_equal": native_token == reference_top8[0],
        "hidden": {
            "max_abs_native_vs_torch": float(hidden_delta.max(initial=0.0)),
            "mean_abs_native_vs_torch": float(hidden_delta.mean()),
            "native_finite": bool(np.isfinite(native_hidden_f32).all()),
            "torch_finite": bool(np.isfinite(reference_hidden_f32).all()),
        },
        "w8_logits": {
            "native_finite": bool(np.isfinite(native_logits).all()),
            "torch_reference_finite": bool(np.isfinite(reference_w8_logits_host).all()),
        },
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, default=Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"))
    parser.add_argument("--prompt-name")
    parser.add_argument("--prompt-names", help="comma-separated prompt names; reuses one torch reference load")
    parser.add_argument("--prompt-render", choices=("raw", "qwen_chat_thinking_off", "qwen_chat_thinking_on"), default="raw")
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--tail-tokens", type=int, default=4)
    parser.add_argument("--reference-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--root-token", type=int, help="explicit fixture root when sampler diagnostics are unavailable")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    names = [
        name.strip()
        for name in str(args.prompt_names or args.prompt_name or "").split(",")
        if name.strip()
    ]
    if not names:
        parser.error("one of --prompt-name or --prompt-names is required")
    results = [
        run(
            model=args.model,
            prompts_file=args.prompts_file,
            prompt_name=name,
            prompt_render=args.prompt_render,
            backend=args.backend,
            tail_tokens=int(args.tail_tokens),
            reference_device=args.reference_device,
            root_token_override=args.root_token,
        )
        for name in names
    ]
    result: dict[str, Any]
    if len(results) == 1:
        result = results[0]
    else:
        result = {
            "schema": "hipengine.paro_mtp_proposal_parity_suite.v1",
            "status": "passed" if all(row["status"] == "passed" for row in results) else "mismatch",
            "performance_claim": False,
            "model": str(args.model),
            "prompt_render": args.prompt_render,
            "tail_tokens": int(args.tail_tokens),
            "results": results,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "prompts": names}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
