"""EAGLE-4 target-corrected greedy cycle over the Qwen3.5 GGUF resident session.

Shared by the campaign baseline harness and the serving provider. The native GPU
EAGLE-4 drafter autoregressively proposes a token chain from the target's aux
hidden states; the target verifies it via verify_target_block; the accepted prefix
plus the corrected token are committed, so the stream is bit-identical to true
greedy AR (the lossless gate). A dynamic tree (vs this chain) is a speed follow-up.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr
from hipengine.kernels.cpu_reference.eagle4 import EAGLE4NumpyDrafter
from hipengine.loading.safetensors import load_weight_index, read_tensor_storage_bytes
from hipengine.runtime.eagle4 import EAGLE4NativeDrafter
from hipengine.runtime.qwen35_gguf_runner import DFlash2HiddenCaptureTargets


def load_eagle4_weights(drafter_path: str | Path) -> tuple[dict, dict]:
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


def _capture_prompt_aux(session, prompt_ids, aux_layers, hidden_size, runtime):
    row_nbytes = hidden_size * DType.BF16.itemsize
    tap_depths = [l + 1 for l in aux_layers]
    buffers = {d: DeviceBuffer(ptr=runtime.malloc(len(prompt_ids) * row_nbytes),
                               nbytes=len(prompt_ids) * row_nbytes) for d in tap_depths}
    targets = DFlash2HiddenCaptureTargets(hidden_size=hidden_size, rows=len(prompt_ids), buffers=buffers)
    probe = session.prefill(prompt_ids, use_bulk=True, dflash2_capture=targets, return_logits=False)
    runtime.device_synchronize()
    taps = []
    for d in tap_depths:
        raw = np.empty(len(prompt_ids) * hidden_size, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(raw), buffers[d], runtime=runtime)
        taps.append((raw.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(len(prompt_ids), hidden_size))
    for b in buffers.values():
        free(b, runtime=runtime)
    return np.concatenate(taps, axis=-1), int(probe.token_id)


def run_eagle4_cycle(session, native, nd, *, prompt_ids, max_new_tokens, runtime,
                     n_draft=6, stop_ids=(), eos_token_id=None, ignore_eos=False) -> dict:
    aux_layers = nd.aux_layers
    hidden_size = int(session.runner.hidden_size)
    session.reset()
    ctx_aux, bonus = _capture_prompt_aux(session, prompt_ids, aux_layers, hidden_size, runtime)
    output_ids: list[int] = []
    produced_total = 0
    accepted_sum = 0
    draft_count = 0
    finish_reason = "length"
    stopped_eos = None
    t0 = time.perf_counter()
    while produced_total < max_new_tokens:
        root_bonus = bonus
        root_pos = ctx_aux.shape[0] - 1
        drafts, _ = native.draft_greedy_chain(ctx_aux[-1], bonus, root_pos, n_draft)
        drafts = [int(t) for t in drafts]
        start_pos = int(session.position)
        bres = session.verify_target_block(
            [bonus] + drafts, bulk_attention_mode="native",
            capture_layer_output_hidden=aux_layers,
            capture_linear_state_rows=True, defer_linear_state_commit=True)
        target_rows = [int(t) for t in bres.token_ids]
        max_accept = max_new_tokens - produced_total
        k = 0
        for j in range(n_draft):
            if k + 1 >= max_accept:
                break
            if target_rows[j] != drafts[j]:
                break
            k += 1
        session._commit_verify_linear_state_row(k, position=start_pos + k + 1)
        bonus = int(target_rows[k])
        produced = k + 1
        accept = [root_bonus] + drafts[:k]
        layh = bres.layer_output_hidden
        new_aux = np.concatenate([np.asarray(layh[l], dtype=np.float32)[:produced] for l in aux_layers], axis=-1)
        ctx_aux = np.concatenate([ctx_aux, new_aux], axis=0)
        for tok in accept:
            if produced_total >= max_new_tokens:
                break
            output_ids.append(tok)
            produced_total += 1
            if tok in stop_ids or (not ignore_eos and eos_token_id is not None and tok == eos_token_id):
                finish_reason = "stop"
                stopped_eos = tok
                break
        accepted_sum += k
        draft_count += 1
        if finish_reason == "stop":
            break
    elapsed = time.perf_counter() - t0
    return {
        "output_ids": output_ids,
        "tokens_per_s": len(output_ids) / elapsed if elapsed else 0.0,
        "elapsed_s": elapsed,
        "mean_acceptance": accepted_sum / (draft_count * n_draft) if draft_count else 0.0,
        "accepted_per_draft": accepted_sum / draft_count if draft_count else 0.0,
        "draft_forwards": draft_count,
        "finish_reason": finish_reason,
        "eos_token_id": stopped_eos,
    }