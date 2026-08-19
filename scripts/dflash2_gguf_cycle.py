#!/usr/bin/env python3
"""D1: end-to-end DFlash2 speculative cycle on the Qwen3.8-27B GGUF target.

Single-prompt greedy driver that ties the exact host-side DFlash2 numpy
drafter to the resident GGUF target session:

  * Prefill captures the full-prompt post-layer hidden at the DFlash2 tap
    depths (5, 19, 33, 47, 61) via ``dflash2_capture``.
  * Each cycle proposes a 7-token DFlash2 block (mask-token noise embeddings,
    exact reference semantics), then verifies it against the target with
    sequential greedy ``session.step`` calls, committing each accepted row and
    never running a rejected row (cache stays clean, no rollback needed).
  * Accepted rows' tap hidden is captured during their commit step and appended
    to the drafter's projected context (projected once, cached per row).

Correctness target (D1): the DFlash2 greedy output agrees with a pure
autoregressive greedy run on the same target, and the measured acceptance is
reported for D4 comparison against exact MTP B3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.kernels.cpu_reference.ops import rmsnorm
from hipengine.loading.dflash import validate_dflash_drafter_metadata
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.safetensors import load_weight_index
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from hipengine.runtime.qwen35_gguf_runner import (
    DFLASH2_TAP_LAYER_IDS,
    DFLASH2_TAP_DEPTHS,
    DFlash2HiddenCaptureTargets,
    Qwen35GGUFResidentSession,
)
from hipengine.speculative.dflash2_drafter import (
    DFlash2NumpyDrafter,
    load_dflash2_numpy_weights,
)
from hipengine.speculative.dflash2_native import (
    DFlash2NativeDrafter,
    _to_bf16_bits,
)
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer


def _load_target_arrays(model: str) -> tuple[Qwen35GGUFTokenizer, np.ndarray, np.ndarray]:
    reader = GGUFReader(model)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(reader.info)
    weights: dict[str, tuple[object, GGMLQuantizationType, tuple[int, ...]]] = {}
    for t in reader.info.tensors:
        if t.name == "token_embd.weight" or t.name == "output.weight":
            weights[t.name] = (reader.tensor_data(t.name), t.ggml_type, t.shape)

    def dq(name: str) -> np.ndarray:
        data, qtype, _ = weights[name]
        return dequantize_gguf_data(data, qtype).astype(np.float32)

    token_embd = dq("token_embd.weight")
    head = dq("output.weight")
    print(f"[target] token_embd {token_embd.shape} ({token_embd.nbytes/1e6:.0f}MB f32) "
          f"output_head {head.shape} ({head.nbytes/1e6:.0f}MB f32)")
    return tokenizer, token_embd, head


def _load_drafter(path: str) -> tuple[DFlash2NumpyDrafter, dict[str, np.ndarray]]:
    index = load_weight_index(path)
    drafter = validate_dflash_drafter_metadata(index)
    weights = load_dflash2_numpy_weights(index)
    print(f"[drafter] {drafter.config.architecture} block_size={drafter.config.block_size} "
          f"mask={drafter.config.mask_token_id} layers={drafter.config.num_hidden_layers} "
          f"taps={drafter.config.target_layer_ids}")
    return DFlash2NumpyDrafter(drafter.config, weights), weights


def _capture_taps_host(
    session: Qwen35GGUFResidentSession,
    targets: DFlash2HiddenCaptureTargets,
    *,
    runtime,
) -> np.ndarray:
    """Copy the 5 tap buffers to a host (rows, n_taps*hidden) f32 array."""
    hidden_size = int(targets.hidden_size)
    rows = int(targets.rows)
    row_nbytes = hidden_size * DType.BF16.itemsize
    taps = np.zeros((rows, len(DFLASH2_TAP_DEPTHS) * hidden_size), dtype=np.float32)
    for depth_index, depth in enumerate(DFLASH2_TAP_DEPTHS):
        buf = targets.buffers[depth]
        raw = np.empty(rows * hidden_size, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(raw), buf, runtime=runtime)
        taps[:, depth_index * hidden_size : (depth_index + 1) * hidden_size] = (
            raw.view(np.float16).astype(np.float32).reshape(rows, hidden_size)
        )
    return taps


def _step_and_taps(
    session: Qwen35GGUFResidentSession,
    token: int,
    *,
    runtime,
    capture: bool,
) -> tuple[int, np.ndarray | None]:
    """Run one greedy decode step, optionally capturing the 5 tap rows."""
    if capture:
        result = session.step(
            int(token),
            return_logits=False,
            capture_layer_output_hidden=list(DFLASH2_TAP_LAYER_IDS),
        )
        taps = np.zeros((len(DFLASH2_TAP_LAYER_IDS), session.runner.hidden_size), dtype=np.float32)
        for i, layer_id in enumerate(DFLASH2_TAP_LAYER_IDS):
            taps[i] = session._last_layer_output_hidden[int(layer_id)]
        return int(result.token_id), taps
    result = session.step(int(token), return_logits=False)
    return int(result.token_id), None


def _run_dflash2_cycle(
    session: Qwen35GGUFResidentSession,
    drafter: DFlash2NumpyDrafter,
    token_embd: np.ndarray,
    head: np.ndarray,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    block_size: int,
    runtime,
) -> dict[str, Any]:
    cfg = drafter.config
    mask = int(cfg.mask_token_id)
    hidden_size = int(cfg.hidden_size)
    n_drafts = block_size - 1

    # --- prefill taps -----------------------------------------------------
    hidden_size_sess = int(session.runner.hidden_size)
    row_nbytes = hidden_size_sess * DType.BF16.itemsize
    prompt_len = len(prompt_ids)
    buffers = {
        depth: DeviceBuffer(ptr=runtime.malloc(prompt_len * row_nbytes), nbytes=prompt_len * row_nbytes)
        for depth in DFLASH2_TAP_DEPTHS
    }
    try:
        targets = DFlash2HiddenCaptureTargets(hidden_size=hidden_size_sess, rows=prompt_len, buffers=buffers)
        t_prefill = time.perf_counter()
        probe = session.prefill(
            prompt_ids,
            use_bulk=True,
            dflash2_capture=targets,
            return_logits=False,
        )
        runtime.device_synchronize()
        prefill_s = time.perf_counter() - t_prefill
        tap_rows = _capture_taps_host(session, targets, runtime=runtime)
        if not np.isfinite(tap_rows).all():
            raise FloatingPointError("prefill tap capture contains NaN/Inf")
        print(f"[prefill] {prompt_len} tokens in {prefill_s:.2f}s first_token={probe.token_id}")

        # Projected-context cache: project each accepted row's taps once.
        projected = drafter.project_target_hidden(tap_rows[None])[0]  # (prompt_len, hidden)
        output_ids: list[int] = []
        bonus = int(probe.token_id)
        produced_total = 0
        acceptance_lengths: list[int] = []
        cycle_times: list[float] = []
        t_decode = time.perf_counter()
        while produced_total < max_new_tokens:
            t_cycle = time.perf_counter()
            ctx_len = int(projected.shape[0])
            # --- draft proposal (reference semantics) ---------------------
            block_input = np.asarray([bonus] + [mask] * n_drafts, dtype=np.int64)
            noise = token_embd[block_input].astype(np.float32)  # (block_size, hidden)
            positions = np.arange(0, ctx_len + block_size, dtype=np.int64)
            hidden = noise[None]
            for layer in range(cfg.num_hidden_layers):
                hidden = drafter.forward_layer(
                    hidden, projected[None], positions[None], layer
                )
            hidden = rmsnorm(hidden[0], drafter._w("norm.weight"), eps=drafter.eps).astype(np.float32)
            draft_hidden = hidden[1 - block_size :]  # (block_size-1, hidden)
            proposal = drafter.propose(
                draft_hidden[None], head, anchor_ids=np.asarray([bonus], dtype=np.int64)
            )
            drafts = [int(token) for token in proposal.path[0]]
            # --- sequential greedy verify (commit-only-accepted) ----------
            accept: list[int] = []
            new_taps: list[np.ndarray] = []
            posterior, anchor_taps = _step_and_taps(session, bonus, runtime=runtime, capture=True)
            accept.append(bonus)
            new_taps.append(anchor_taps)
            accepted = 0
            for i, draft in enumerate(drafts):
                if produced_total + len(accept) >= max_new_tokens:
                    break
                if posterior != draft:
                    break
                posterior, taps_i = _step_and_taps(session, draft, runtime=runtime, capture=True)
                accept.append(draft)
                new_taps.append(taps_i)
                accepted += 1
            # bonus for next cycle = target prediction after the accepted prefix.
            bonus = posterior
            produced = len(accept)
            # --- extend projected context with the accepted rows' taps ----
            if produced:
                new_tap_rows = np.stack(new_taps[:produced], axis=1)  # (n_taps, produced, hidden)
                new_concat = new_tap_rows.transpose(1, 0, 2).reshape(produced, len(DFLASH2_TAP_DEPTHS) * hidden_size)
                projected = np.concatenate([projected, drafter.project_target_hidden(new_concat)], axis=0)
            output_ids.extend(accept)
            produced_total += produced
            acceptance_lengths.append(produced)
            cycle_times.append(time.perf_counter() - t_cycle)
        decode_s = time.perf_counter() - t_decode

        # --- stats --------------------------------------------------------
        n_cycles = len(acceptance_lengths)
        accepted_draft = sum(max(0, length - 1) for length in acceptance_lengths)
        n_drafts_total = n_cycles * n_drafts
        acc_per_draft = accepted_draft / n_drafts_total if n_drafts_total else 0.0
        acc_per_output = accepted_draft / produced_total if produced_total else 0.0
        mean_acc = float(np.mean(acceptance_lengths)) if acceptance_lengths else 0.0
        print(f"[dflash2] cycles={n_cycles} produced={produced_total} "
              f"mean_acceptance={mean_acc:.2f} accepted/draft={acc_per_draft:.3f} "
              f"tokens/s={produced_total/decode_s:.2f}")
        return {
            "output_ids": output_ids,
            "acceptance_lengths": acceptance_lengths,
            "cycles": n_cycles,
            "accepted_drafts": accepted_draft,
            "drafts_total": n_drafts_total,
            "accepted_per_draft": acc_per_draft,
            "accepted_per_output": acc_per_output,
            "mean_acceptance": mean_acc,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "tokens_per_s": produced_total / decode_s if decode_s else 0.0,
            "mean_cycle_s": float(np.mean(cycle_times)) if cycle_times else 0.0,
        }
    finally:
        for buf in buffers.values():
            runtime.free(buf.ptr)


def _run_dflash2_cycle_batch(
    session: Qwen35GGUFResidentSession,
    drafter: DFlash2NativeDrafter,
    numpy_weights: dict[str, np.ndarray],
    token_embd: np.ndarray,
    head: np.ndarray,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    block_size: int,
    runtime,
    verify_mode: str = "native",
) -> dict[str, Any]:
    """End-to-end DFlash2 cycle with a B7 batched chain verifier.

    Replaces the per-draft sequential ``session.step`` verify with one
    ``verify_target_block`` bulk pass over ``[bonus] + drafts`` (B+1 rows).  The
    bulk pass returns the target greedy row per block row, and direct linear-
    state commit (``_commit_verify_linear_state_row``) makes the partial-accept
    target state exact (validated: bulk target rows and post-commit next
    prediction match the token-serial path).  Accepted rows' per-layer taps come
    from the bulk ``capture_layer_output_hidden`` and extend the drafter's
    projected context exactly like the sequential cycle.
    """
    cfg = drafter.config
    mask = int(cfg.mask_token_id)
    hidden_size = int(cfg.hidden_size)
    fwd_bs = int(drafter.block_size)  # drafter forward always runs at config block size
    n_drafts = block_size - 1          # verify chain length (CLI block size)

    hidden_size_sess = int(session.runner.hidden_size)
    row_nbytes = hidden_size_sess * DType.BF16.itemsize
    prompt_len = len(prompt_ids)
    buffers = {
        depth: DeviceBuffer(ptr=runtime.malloc(prompt_len * row_nbytes), nbytes=prompt_len * row_nbytes)
        for depth in DFLASH2_TAP_DEPTHS
    }
    try:
        targets = DFlash2HiddenCaptureTargets(hidden_size=hidden_size_sess, rows=prompt_len, buffers=buffers)
        t_prefill = time.perf_counter()
        probe = session.prefill(
            prompt_ids,
            use_bulk=True,
            dflash2_capture=targets,
            return_logits=False,
        )
        runtime.device_synchronize()
        prefill_s = time.perf_counter() - t_prefill
        tap_rows = _capture_taps_host(session, targets, runtime=runtime)
        if not np.isfinite(tap_rows).all():
            raise FloatingPointError("prefill tap capture contains NaN/Inf")
        print(f"[prefill] {prompt_len} tokens in {prefill_s:.2f}s first_token={probe.token_id}")

        npd = DFlash2NumpyDrafter(cfg, numpy_weights)
        projected = npd.project_target_hidden(tap_rows[None])[0]
        drafter.reset_projected_context(_to_bf16_bits(projected))
        head_ptr = drafter.upload_weight("output_head.weight", head)

        output_ids: list[int] = []
        bonus = int(probe.token_id)
        produced_total = 0
        acceptance_lengths: list[int] = []
        cycle_times: list[float] = []
        recall_at1: list[int] = []
        recall_at16: list[int] = []
        recall_unary: list[int] = []
        t_decode = time.perf_counter()
        while produced_total < max_new_tokens:
            t_cycle = time.perf_counter()
            ctx_len = int(drafter.ctx_len)
            # --- draft proposal (native kernels, fixed config block size) -
            block_input = np.asarray([bonus] + [mask] * (fwd_bs - 1), dtype=np.int64)
            noise = token_embd[block_input].astype(np.float32)  # (fwd_bs, hidden)
            positions = np.arange(0, ctx_len + fwd_bs, dtype=np.int64)
            draft_ptr = drafter.forward(_to_bf16_bits(noise), positions)
            drafter.runtime.device_synchronize()
            path, _ = drafter.select(
                draft_ptr, head_ptr, None, np.asarray([bonus], dtype=np.int64)
            )
            cands = drafter.last_candidates()
            unary = cands[:, 0]  # top-1 of on-device top-16 == global argmax (no full-logit host copy)
            drafts = [int(token) for token in path[:n_drafts]]
            # --- batched chain verify (B+1 rows in one bulk pass) --------
            start_pos = int(session.position)
            block_inputs = [bonus] + drafts
            if os.environ.get("DF2_CYCLE_DEBUG"):
                t_v0 = time.perf_counter()
            bres = session.verify_target_block(
                block_inputs,
                bulk_attention_mode=verify_mode,
                capture_layer_output_hidden=list(DFLASH2_TAP_LAYER_IDS),
                capture_linear_state_rows=True,
                defer_linear_state_commit=True,
            )
            target_rows = [int(t) for t in bres.token_ids]
            if os.environ.get("DF2_CYCLE_DEBUG"):
                print(f"  [dbg] block_inputs={len(block_inputs)} rows target={len(target_rows)} verify_wall={(time.perf_counter()-t_v0)*1000:.0f}ms", flush=True)
            # acceptance: draft j accepted iff draft_j == target_rows[j]
            max_accept = max_new_tokens - produced_total  # total rows this cycle
            k = 0
            for j in range(n_drafts):
                if k + 1 >= max_accept:
                    break
                recall_at1.append(1 if target_rows[j] == drafts[j] else 0)
                recall_at16.append(1 if int(target_rows[j]) in {int(c) for c in cands[j]} else 0)
                recall_unary.append(1 if int(target_rows[j]) == int(unary[j]) else 0)
                if target_rows[j] != drafts[j]:
                    break
                k += 1
            # commit accepted rows (0..k) via the last accepted row's state
            session._commit_verify_linear_state_row(k, position=start_pos + k + 1)
            bonus = int(target_rows[k])
            produced = k + 1
            # accepted token list: bonus + accepted drafts
            accept = [block_inputs[0]] + drafts[:k]
            # --- extend projected context with accepted rows' bulk taps ---
            lay_hidden = bres.layer_output_hidden
            if produced:
                # lay_hidden[layer] is (B+1, hidden); accepted rows are 0..k
                new_tap_rows = np.stack(
                    [lay_hidden[int(layer_id)][: produced] for layer_id in DFLASH2_TAP_LAYER_IDS],
                    axis=0,
                )  # (n_taps, produced, hidden)
                new_concat = new_tap_rows.transpose(1, 0, 2).reshape(
                    produced, len(DFLASH2_TAP_DEPTHS) * hidden_size
                )
                proj_new = npd.project_target_hidden(new_concat)
                append_pos = np.arange(ctx_len, ctx_len + produced, dtype=np.int32)
                drafter.append_projected_rows(_to_bf16_bits(proj_new), append_pos)
            output_ids.extend(accept)
            produced_total += produced
            acceptance_lengths.append(produced)
            cycle_times.append(time.perf_counter() - t_cycle)
        decode_s = time.perf_counter() - t_decode

        n_cycles = len(acceptance_lengths)
        accepted_draft = sum(max(0, length - 1) for length in acceptance_lengths)
        n_drafts_total = n_cycles * n_drafts
        acc_per_draft = accepted_draft / n_drafts_total if n_drafts_total else 0.0
        acc_per_output = accepted_draft / produced_total if produced_total else 0.0
        mean_acc = float(np.mean(acceptance_lengths)) if acceptance_lengths else 0.0
        print(f"[dflash2-batch] cycles={n_cycles} produced={produced_total} "
              f"mean_acceptance={mean_acc:.2f} accepted/draft={acc_per_draft:.3f} "
              f"tokens/s={produced_total/decode_s:.2f}")
        if recall_at1:
            r1 = sum(recall_at1) / len(recall_at1)
            r16 = sum(recall_at16) / len(recall_at16)
            ru = sum(recall_unary) / len(recall_unary)
            print(f"[dflash2-batch] recall@1={r1:.3f} recall@16={r16:.3f} unary-argmax={ru:.3f} "
                  f"(n={len(recall_at1)} draft positions)")
        return {
            "output_ids": output_ids,
            "acceptance_lengths": acceptance_lengths,
            "cycles": n_cycles,
            "accepted_drafts": accepted_draft,
            "drafts_total": n_drafts_total,
            "accepted_per_draft": acc_per_draft,
            "accepted_per_output": acc_per_output,
            "mean_acceptance": mean_acc,
            "recall_at1": float(sum(recall_at1) / len(recall_at1)) if recall_at1 else None,
            "recall_at16": float(sum(recall_at16) / len(recall_at16)) if recall_at16 else None,
            "recall_unary_argmax": float(sum(recall_unary) / len(recall_unary)) if recall_unary else None,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "tokens_per_s": produced_total / decode_s if decode_s else 0.0,
            "mean_cycle_s": float(np.mean(cycle_times)) if cycle_times else 0.0,
        }
    finally:
        for buf in buffers.values():
            runtime.free(buf.ptr)


def _run_dflash2_cycle_native(
    session: Qwen35GGUFResidentSession,
    drafter: DFlash2NativeDrafter,
    numpy_weights: dict[str, np.ndarray],
    token_embd: np.ndarray,
    head: np.ndarray,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    block_size: int,
    runtime,
) -> dict[str, Any]:
    """End-to-end DFlash2 cycle using the native drafter forward + selector.

    Mirrors ``_run_dflash2_cycle`` (same sequential greedy verify and accepted-
    row tap append) but the draft proposal runs on the GPU (native kernels)
    instead of the numpy drafter.  ``numpy_weights`` is used only for the
    host-side fc projection of new tap rows (kept on CPU like the numpy path).
    """
    cfg = drafter.config
    mask = int(cfg.mask_token_id)
    hidden_size = int(cfg.hidden_size)
    n_drafts = block_size - 1

    hidden_size_sess = int(session.runner.hidden_size)
    row_nbytes = hidden_size_sess * DType.BF16.itemsize
    prompt_len = len(prompt_ids)
    buffers = {
        depth: DeviceBuffer(ptr=runtime.malloc(prompt_len * row_nbytes), nbytes=prompt_len * row_nbytes)
        for depth in DFLASH2_TAP_DEPTHS
    }
    try:
        targets = DFlash2HiddenCaptureTargets(hidden_size=hidden_size_sess, rows=prompt_len, buffers=buffers)
        t_prefill = time.perf_counter()
        probe = session.prefill(
            prompt_ids,
            use_bulk=True,
            dflash2_capture=targets,
            return_logits=False,
        )
        runtime.device_synchronize()
        prefill_s = time.perf_counter() - t_prefill
        tap_rows = _capture_taps_host(session, targets, runtime=runtime)
        if not np.isfinite(tap_rows).all():
            raise FloatingPointError("prefill tap capture contains NaN/Inf")
        print(f"[prefill] {prompt_len} tokens in {prefill_s:.2f}s first_token={probe.token_id}")

        # Seed the native projected-context cache from the prefill taps.
        npd = DFlash2NumpyDrafter(cfg, numpy_weights)
        projected = npd.project_target_hidden(tap_rows[None])[0]
        drafter.reset_projected_context(_to_bf16_bits(projected))
        head_ptr = drafter.upload_weight("output_head.weight", head)

        output_ids: list[int] = []
        bonus = int(probe.token_id)
        produced_total = 0
        acceptance_lengths: list[int] = []
        cycle_times: list[float] = []
        recall_at1: list[int] = []  # draft token == target argmax for that position
        recall_at16: list[int] = []  # target argmax appears in the drafter's top-16
        recall_unary: list[int] = []  # raw draft-logit argmax == target argmax
        t_decode = time.perf_counter()
        while produced_total < max_new_tokens:
            t_cycle = time.perf_counter()
            ctx_len = int(drafter.ctx_len)
            # --- draft proposal (native kernels) -------------------------
            block_input = np.asarray([bonus] + [mask] * n_drafts, dtype=np.int64)
            noise = token_embd[block_input].astype(np.float32)  # (block_size, hidden)
            positions = np.arange(0, ctx_len + block_size, dtype=np.int64)
            draft_ptr = drafter.forward(_to_bf16_bits(noise), positions)
            drafter.runtime.device_synchronize()
            path, _ = drafter.select(
                draft_ptr, head_ptr, None, np.asarray([bonus], dtype=np.int64)
            )
            cands = drafter.last_candidates()  # (n_drafts, top_k)
            unary = cands[:, 0]  # top-1 of on-device top-16 == global argmax (no full-logit host copy)
            drafts = [int(token) for token in path[:n_drafts]]
            # --- sequential greedy verify (commit-only-accepted) ----------
            accept: list[int] = []
            new_taps: list[np.ndarray] = []
            posterior, anchor_taps = _step_and_taps(session, bonus, runtime=runtime, capture=True)
            accept.append(bonus)
            new_taps.append(anchor_taps)
            for i, draft in enumerate(drafts):
                if produced_total + len(accept) >= max_new_tokens:
                    break
                recall_at1.append(1 if posterior == draft else 0)
                recall_at16.append(1 if int(posterior) in {int(c) for c in cands[i]} else 0)
                recall_unary.append(1 if posterior == int(unary[i]) else 0)
                if posterior != draft:
                    break
                posterior, taps_i = _step_and_taps(session, draft, runtime=runtime, capture=True)
                accept.append(draft)
                new_taps.append(taps_i)
            bonus = posterior
            produced = len(accept)
            # --- extend the native projected context with accepted rows ----
            if produced:
                new_tap_rows = np.stack(new_taps[:produced], axis=1)  # (n_taps, produced, hidden)
                new_concat = new_tap_rows.transpose(1, 0, 2).reshape(produced, len(DFLASH2_TAP_DEPTHS) * hidden_size)
                proj_new = npd.project_target_hidden(new_concat)
                append_pos = np.arange(ctx_len, ctx_len + produced, dtype=np.int32)
                drafter.append_projected_rows(_to_bf16_bits(proj_new), append_pos)
            output_ids.extend(accept)
            produced_total += produced
            acceptance_lengths.append(produced)
            cycle_times.append(time.perf_counter() - t_cycle)
        decode_s = time.perf_counter() - t_decode

        n_cycles = len(acceptance_lengths)
        accepted_draft = sum(max(0, length - 1) for length in acceptance_lengths)
        n_drafts_total = n_cycles * n_drafts
        acc_per_draft = accepted_draft / n_drafts_total if n_drafts_total else 0.0
        acc_per_output = accepted_draft / produced_total if produced_total else 0.0
        mean_acc = float(np.mean(acceptance_lengths)) if acceptance_lengths else 0.0
        print(f"[dflash2-native] cycles={n_cycles} produced={produced_total} "
              f"mean_acceptance={mean_acc:.2f} accepted/draft={acc_per_draft:.3f} "
              f"tokens/s={produced_total/decode_s:.2f}")
        if recall_at1:
            r1 = sum(recall_at1) / len(recall_at1)
            r16 = sum(recall_at16) / len(recall_at16)
            ru = sum(recall_unary) / len(recall_unary)
            print(f"[dflash2-native] recall@1={r1:.3f} recall@16={r16:.3f} unary-argmax={ru:.3f} (n={len(recall_at1)} draft positions)")
        return {
            "output_ids": output_ids,
            "acceptance_lengths": acceptance_lengths,
            "cycles": n_cycles,
            "accepted_drafts": accepted_draft,
            "drafts_total": n_drafts_total,
            "accepted_per_draft": acc_per_draft,
            "accepted_per_output": acc_per_output,
            "mean_acceptance": mean_acc,
            "recall_at1": float(sum(recall_at1) / len(recall_at1)) if recall_at1 else None,
            "recall_at16": float(sum(recall_at16) / len(recall_at16)) if recall_at16 else None,
            "recall_unary_argmax": float(sum(recall_unary) / len(recall_unary)) if recall_unary else None,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "tokens_per_s": produced_total / decode_s if decode_s else 0.0,
            "mean_cycle_s": float(np.mean(cycle_times)) if cycle_times else 0.0,
        }
    finally:
        for buf in buffers.values():
            runtime.free(buf.ptr)


def _run_ar(
    session: Qwen35GGUFResidentSession,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    runtime,
) -> tuple[list[int], float]:
    session.reset()
    probe = session.prefill(prompt_ids, use_bulk=True, return_logits=False)
    out = [int(probe.token_id)]
    t0 = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        res = session.step(out[-1], return_logits=False)
        out.append(int(res.token_id))
    return out, time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument(
        "--drafter",
        default="/home/lhl/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
    )
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. Continue:")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--native", action="store_true", help="use the native GPU drafter forward+selector instead of the numpy drafter")
    parser.add_argument("--batch-verify", action="store_true", help="verify the B7 draft chain with one bulk verify_target_block pass instead of sequential session.step calls")
    parser.add_argument("--compare-ar", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    tokenizer, token_embd, head = _load_target_arrays(args.model)
    drafter, numpy_weights = load_and_build_drafter(args.drafter)
    block_size = args.block_size
    if block_size < 2 or block_size > int(drafter.config.block_size):
        print(f"warning: requested block_size {block_size} outside [2, {drafter.config.block_size}]; clamping")
        block_size = min(max(2, block_size), int(drafter.config.block_size))
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"[prompt] tokens={len(prompt_ids)} decoded={tokenizer.decode(prompt_ids)!r}")

    runtime = get_hip_runtime()
    max_seq = len(prompt_ids) + args.max_new_tokens + block_size + 2

    results: dict[str, Any] = {"prompt_ids": prompt_ids, "prompt_text": args.prompt}

    # Pure AR reference (fresh session).
    if args.compare_ar:
        with Qwen35GGUFResidentSession(
            args.model, backend=args.backend, compiler_version=None,
            require_cached_build=False, max_sequence_length=max_seq, max_batch_size=1,
        ) as session:
            ar_out, ar_s = _run_ar(session, prompt_ids=prompt_ids, max_new_tokens=args.max_new_tokens, runtime=runtime)
        results["ar_output_ids"] = ar_out
        results["ar_tokens_per_s"] = args.max_new_tokens / ar_s
        results["ar_decode_s"] = ar_s
        print(f"[ar] produced={len(ar_out)} in {ar_s:.2f}s tokens/s={args.max_new_tokens/ar_s:.2f}")

    with Qwen35GGUFResidentSession(
        args.model, backend=args.backend, compiler_version=None,
        require_cached_build=False, max_sequence_length=max_seq, max_batch_size=1,
    ) as session:
        if args.native and args.batch_verify:
            native_drafter = DFlash2NativeDrafter(
                drafter.config, numpy_weights, max_context_len=max_seq,
            )
            try:
                df2 = _run_dflash2_cycle_batch(
                    session,
                    native_drafter,
                    numpy_weights,
                    token_embd,
                    head,
                    prompt_ids=prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    block_size=block_size,
                    runtime=runtime,
                )
            finally:
                native_drafter.close()
        elif args.native:
            native_drafter = DFlash2NativeDrafter(
                drafter.config, numpy_weights, max_context_len=max_seq,
            )
            try:
                df2 = _run_dflash2_cycle_native(
                    session,
                    native_drafter,
                    numpy_weights,
                    token_embd,
                    head,
                    prompt_ids=prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    block_size=block_size,
                    runtime=runtime,
                )
            finally:
                native_drafter.close()
        else:
            df2 = _run_dflash2_cycle(
                session,
                drafter,
                token_embd,
                head,
                prompt_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                block_size=block_size,
                runtime=runtime,
            )
    results.update(df2)
    results["output_text"] = tokenizer.decode(df2["output_ids"])

    if args.compare_ar and "ar_output_ids" in results:
        df2_ids = results["output_ids"][: len(results["ar_output_ids"])]
        ar_ids = results["ar_output_ids"]
        common = min(len(df2_ids), len(ar_ids))
        matches = sum(1 for a, b in zip(df2_ids[:common], ar_ids[:common]) if a == b)
        results["ar_agreement_common"] = common
        results["ar_agreement_matches"] = matches
        results["ar_agreement"] = matches / common if common else 0.0
        print(f"[correctness] DFlash2 vs AR greedy: {matches}/{common} tokens agree ({results['ar_agreement']:.3f})")
        print(f"  df2: {results['output_text']!r}")
        print(f"  ar : {tokenizer.decode(ar_ids)!r}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(results)
        payload["schema"] = 1
        payload["kind"] = "hipengine_dflash2_gguf_cycle"
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def load_and_build_drafter(path: str):
    index = load_weight_index(path)
    meta = validate_dflash_drafter_metadata(index)
    weights = load_dflash2_numpy_weights(index)
    print(f"[drafter] {meta.config.architecture} block_size={meta.config.block_size} "
          f"mask={meta.config.mask_token_id} layers={meta.config.num_hidden_layers} "
          f"taps={meta.config.target_layer_ids}")
    return DFlash2NumpyDrafter(meta.config, weights), weights


if __name__ == "__main__":
    raise SystemExit(main())
