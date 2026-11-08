"""DSpine GGUF cycle helpers for Qwen3.8-27B (campaign measurement surface).

Mirrors dflash2_gguf_cycle: a target-corrected greedy cycle over the resident
session public surface (prefill tap capture + verify_target_block). The native
GPU DSpine drafter proposes a block; the target verifies it; the accepted prefix
plus the corrected token are committed, so the output stream is bit-identical to
true greedy AR (the lossless gate). Acceptance only changes how many tokens each
target forward commits.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.kernels.cpu_reference.dspine import DSpineNumpyDrafter
from hipengine.loading.safetensors import load_weight_index, read_tensor_storage_bytes
from hipengine.runtime.dspine import DSpineNativeDrafter
from hipengine.runtime.qwen35_gguf_runner import DFlash2HiddenCaptureTargets
from hipengine.speculative.dflash2_native import _to_bf16_bits


def load_dspine_weights(drafter_path: str | Path) -> tuple[dict, dict]:
    """Load the DSpine safetensors artifact as f32 numpy weights + config."""
    root = Path(drafter_path)
    config = json.loads((root / "config.json").read_text())
    index = load_weight_index(root)
    weights = {}
    for name, info in index.tensors.items():
        if name.endswith("_version"):
            continue
        payload = read_tensor_storage_bytes(info)
        bits = np.frombuffer(payload, dtype="<u2").astype(np.uint32) << np.uint32(16)
        weights[name] = bits.view(np.float32).reshape(info.shape)
    return config, weights


def _capture_prompt_taps(session, prompt_ids, tap_layers, hidden_size, runtime):
    row_nbytes = hidden_size * DType.BF16.itemsize
    tap_depths = [l + 1 for l in tap_layers]
    buffers = {d: DeviceBuffer(ptr=runtime.malloc(len(prompt_ids) * row_nbytes),
                               nbytes=len(prompt_ids) * row_nbytes) for d in tap_depths}
    targets = DFlash2HiddenCaptureTargets(hidden_size=hidden_size, rows=len(prompt_ids), buffers=buffers)
    probe = session.prefill(prompt_ids, use_bulk=True, dflash2_capture=targets, return_logits=False)
    runtime.device_synchronize()
    ctx_taps = np.zeros((len(prompt_ids), len(tap_layers) * hidden_size), dtype=np.float32)
    for i, d in enumerate(tap_depths):
        raw = np.empty(len(prompt_ids) * hidden_size, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(raw), buffers[d], runtime=runtime)
        ctx_taps[:, i * hidden_size:(i + 1) * hidden_size] = (
            (raw.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(len(prompt_ids), hidden_size))
    for b in buffers.values():
        from hipengine.core.memory import free
        free(b, runtime=runtime)
    return ctx_taps, int(probe.token_id)


def run_dspine_cycle(session, native, nd, token_embd, head, *, prompt_ids,
                     max_new_tokens, runtime) -> dict:
    """One DSpine target-corrected greedy cycle; returns ids + speed + acceptance."""
    tap_layers = list(nd.config["dflash_config"]["target_layer_ids"])
    block_size = int(nd.config["block_size"])
    mask = nd.mask_token_id
    n_drafts = block_size - 1
    hidden_size = int(session.runner.hidden_size)
    session.reset()
    ctx_taps, bonus = _capture_prompt_taps(session, prompt_ids, tap_layers, hidden_size, runtime)
    native.reset_projected_context(_to_bf16_bits(nd.project_target_hidden(ctx_taps)))
    output_ids: list[int] = []
    produced_total = 0
    accepted_sum = 0
    draft_count = 0
    t0 = time.perf_counter()
    while produced_total < max_new_tokens:
        ctx_len = ctx_taps.shape[0]
        block_input = np.asarray([bonus] + [mask] * n_drafts, dtype=np.int64)
        noise = token_embd[block_input].astype(np.float32)
        positions = np.arange(0, ctx_len + block_size, dtype=np.int64)
        native.forward(noise, positions, anchor_id=bonus)
        drafts = [int(t) for t in native.refine(head, bonus)[:n_drafts]]
        start_pos = int(session.position)
        bres = session.verify_target_block(
            [bonus] + drafts, bulk_attention_mode="native",
            capture_layer_output_hidden=tap_layers,
            capture_linear_state_rows=True, defer_linear_state_commit=True)
        target_rows = [int(t) for t in bres.token_ids]
        max_accept = max_new_tokens - produced_total
        k = 0
        for j in range(n_drafts):
            if k + 1 >= max_accept:
                break
            if target_rows[j] != drafts[j]:
                break
            k += 1
        session._commit_verify_linear_state_row(k, position=start_pos + k + 1)
        bonus = int(target_rows[k])
        produced = k + 1
        accept = [int(block_input[0])] + drafts[:k]
        layh = bres.layer_output_hidden
        new_taps = np.stack([np.asarray(layh[l], dtype=np.float32)[:produced] for l in tap_layers], axis=0)
        new_concat = new_taps.transpose(1, 0, 2).reshape(produced, len(tap_layers) * hidden_size)
        native.append_projected_rows(
            _to_bf16_bits(nd.project_target_hidden(new_concat)),
            np.arange(ctx_len, ctx_len + produced, dtype=np.int64))
        ctx_taps = np.concatenate([ctx_taps, new_concat], axis=0)
        output_ids.extend(accept)
        produced_total += produced
        accepted_sum += k
        draft_count += 1
    elapsed = time.perf_counter() - t0
    return {
        "output_ids": output_ids,
        "tokens_per_s": len(output_ids) / elapsed if elapsed else 0.0,
        "elapsed_s": elapsed,
        "mean_acceptance": accepted_sum / (draft_count * n_drafts) if draft_count else 0.0,
        "accepted_per_draft": accepted_sum / draft_count if draft_count else 0.0,
        "draft_forwards": draft_count,
    }