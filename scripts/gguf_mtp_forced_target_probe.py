#!/usr/bin/env python3
"""Forced-prefix GGUF MTP target verifier probe.

This diagnostic replays the target session to the start of one recorded MTP
cycle, runs the target verifier on ``[prev] + draft_tokens``, and emits
row-level top-k/candidate logits.  It is meant for semantic parity debugging,
not performance measurement.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gguf_mtp_bench import GGUF_PATH, build_chat_prompt, hidden_state_summary


class ProbeError(RuntimeError):
    pass


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _set_env_bool(name: str, value: bool) -> None:
    os.environ[name] = "1" if bool(value) else "0"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ProbeError(f"expected JSON object in {path}")
    return data


def _find_cycle(cycles: list[dict[str, Any]], cycle_number: int) -> tuple[int, dict[str, Any]]:
    for index, row in enumerate(cycles):
        if int(row.get("cycle", index)) == int(cycle_number):
            return index, row
    raise ProbeError(f"cycle {cycle_number} not found in trace")


def _flatten_output_tokens(cycles: Iterable[dict[str, Any]]) -> list[int]:
    tokens: list[int] = []
    for row in cycles:
        output = row.get("output_tokens")
        if not isinstance(output, list):
            raise ProbeError("trace cycle is missing output_tokens[]")
        tokens.extend(int(token) for token in output)
    return tokens


def _parse_token_csv(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        return []
    values: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            token = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--candidate-token must be an int or comma-separated ints") from exc
        if token < 0:
            raise argparse.ArgumentTypeError("--candidate-token values must be non-negative")
        values.append(token)
    return values


def _parse_layer_row(raw: str | None) -> tuple[int, int]:
    if raw is None or raw.strip() == "":
        raise argparse.ArgumentTypeError("expected LAYER:ROW")
    parts = raw.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected LAYER:ROW")
    try:
        layer = int(parts[0].strip())
        row = int(parts[1].strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer LAYER:ROW") from exc
    if layer < 0:
        raise argparse.ArgumentTypeError("layer must be non-negative")
    if row < 0:
        raise argparse.ArgumentTypeError("row must be non-negative")
    return layer, row


def _unique_ints(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        token = int(value)
        if token not in seen:
            result.append(token)
            seen.add(token)
    return result


def _apply_trace_route_env(workload: dict[str, Any], *, decode_repack: bool | None) -> dict[str, str | None]:
    if decode_repack is None:
        decode_repack = bool(workload.get("decode_repack", workload.get("decode_repack_env") == "1"))
    _set_env_bool("HIPENGINE_GGUF_DECODE_REPACK", bool(decode_repack))

    for name in (
        "HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A",
        "HIPENGINE_GGUF_T16_SELECTED_DP4A",
        "HIPENGINE_GGUF_RAW_SELECTED_DP4A",
    ):
        _set_env_bool(name, bool(workload.get("verify_dp4a", False)))

    dense_pair = bool(workload.get("verify_dense_q8_dp4a", False))
    dense_all = bool(workload.get("verify_dense_q8_dp4a_all", False))
    dense_shared = bool(workload.get("verify_dense_q8_dp4a_shared", False))
    dense_f32 = bool(workload.get("verify_dense_q8_dp4a_f32", False))
    _set_env_bool("HIPENGINE_GGUF_Q8_0_RAW_SIDECAR", dense_pair or dense_all or dense_shared or dense_f32)
    _set_env_bool("HIPENGINE_GGUF_DENSE_Q8_DP4A", dense_pair)
    _set_env_bool("HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL", dense_all or dense_shared or dense_f32)
    _set_env_bool("HIPENGINE_GGUF_DENSE_Q8_DP4A_SHARED", dense_shared)
    _set_env_bool("HIPENGINE_GGUF_DENSE_Q8_DP4A_F32", dense_f32)

    selected_down = str(workload.get("selected_down_x8_repack", workload.get("selected_down_x8_repack_env") or "off"))
    os.environ["HIPENGINE_GGUF_SELECTED_X8_REPACK"] = selected_down
    _set_env_bool("HIPENGINE_GGUF_SELECTED_GATE_UP_X8", bool(workload.get("selected_gate_up_x8", False)))
    _set_env_bool("HIPENGINE_GGUF_SELECTED_GATE_UP_RAW", bool(workload.get("selected_gate_up_raw", False)))

    # Full row logits are the point of this probe; force the direct top-1
    # verifier lm-head route off even if the shell inherited it from a prior run.
    _set_env_bool("HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A", False)
    _set_env_bool("HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR", False)

    return {name: os.environ.get(name) for name in sorted(os.environ) if name.startswith("HIPENGINE_GGUF_")}


def _repo_provenance() -> dict[str, Any]:
    def git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).strip()
        except Exception:
            return None

    status = git(["status", "--short"])
    return {
        "root": str(REPO_ROOT),
        "commit": git(["rev-parse", "HEAD"]),
        "branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
    }


def _top_k_rows(logits: np.ndarray, *, top_k: int) -> list[dict[str, Any]]:
    row = np.ascontiguousarray(np.asarray(logits, dtype=np.float32).reshape(-1))
    if row.size == 0:
        raise ProbeError("cannot rank an empty logits row")
    if not np.all(np.isfinite(row)):
        raise FloatingPointError("logits row contains NaN or Inf")
    k = min(max(1, int(top_k)), int(row.size))
    if k == 1:
        top_idx = np.asarray([int(np.argmax(row))], dtype=np.int64)
    else:
        top_idx = np.argpartition(row, -k)[-k:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
    top_logit = float(row[int(top_idx[0])])
    max_logit = float(np.max(row))
    exp_sum = float(np.exp(row.astype(np.float64) - max_logit).sum())
    result: list[dict[str, Any]] = []
    for rank, token in enumerate(top_idx.tolist(), start=1):
        logit = float(row[int(token)])
        result.append(
            {
                "rank": int(rank),
                "token_id": int(token),
                "logit": logit,
                "margin_from_top": float(top_logit - logit),
                "prob": float(np.exp(float(logit) - max_logit) / exp_sum),
            }
        )
    return result


def _candidate_rows(logits: np.ndarray, token_ids: Sequence[int]) -> list[dict[str, Any]]:
    row = np.ascontiguousarray(np.asarray(logits, dtype=np.float32).reshape(-1))
    if row.size == 0:
        raise ProbeError("cannot score candidates from an empty logits row")
    if not np.all(np.isfinite(row)):
        raise FloatingPointError("logits row contains NaN or Inf")
    top_logit = float(np.max(row))
    max_logit = top_logit
    exp_sum = float(np.exp(row.astype(np.float64) - max_logit).sum())
    result: list[dict[str, Any]] = []
    for token in _unique_ints(token_ids):
        if token < 0 or token >= int(row.size):
            result.append({"token_id": int(token), "in_vocab": False})
            continue
        logit = float(row[token])
        rank = int(np.count_nonzero(row > logit) + 1)
        result.append(
            {
                "token_id": int(token),
                "in_vocab": True,
                "rank": rank,
                "logit": logit,
                "margin_from_top": float(top_logit - logit),
                "prob": float(np.exp(float(logit) - max_logit) / exp_sum),
            }
        )
    return result


def _llama_accept_count(draft_tokens: Sequence[int], target_tokens: Sequence[int]) -> int:
    accepted = 0
    for draft, target in zip(draft_tokens, target_tokens, strict=False):
        if int(draft) != int(target):
            break
        accepted += 1
    return accepted


def _copy_verify_logits(session: Any, rows: int) -> np.ndarray:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    if session.runner is None:
        raise ProbeError("session runner is closed")
    logits_buf = getattr(session, "_verify_logits_buf", None)
    if logits_buf is None:
        raise ProbeError(
            "target block did not populate _verify_logits_buf; "
            "disable HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A for this probe"
        )
    vocab_size = int(session.runner.vocab_size)
    logits = np.empty((int(rows), vocab_size), dtype=np.float32)
    runtime = session.runtime
    copy_device_to_host(
        host_array_ptr(logits),
        DeviceBuffer(int(logits_buf.ptr), int(logits.nbytes)),
        int(logits.nbytes),
        runtime=runtime,
    )
    return np.ascontiguousarray(logits, dtype=np.float32)


def _copy_current_hidden_seed(session: Any) -> np.ndarray:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
    from hipengine.runtime.qwen35_gguf_runner import DType

    if session.runner is None or session.scratch is None:
        raise ProbeError("session is closed")
    hidden = np.empty((1, int(session.runner.hidden_size)), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(hidden),
        DeviceBuffer(
            int(session.scratch.hidden_seed_fp32.ptr),
            int(session.runner.hidden_size) * DType.FP32.itemsize,
        ),
        hidden.nbytes,
        runtime=session.runtime,
    )
    return np.ascontiguousarray(hidden, dtype=np.float32)


def _bf16_roundtrip_f32(values: np.ndarray) -> np.ndarray:
    row = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    bits = row.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return np.ascontiguousarray((rounded & np.uint32(0xFFFF0000)).view(np.float32), dtype=np.float32)


def _capture_moe_component_arrays(capture: Any) -> dict[str, np.ndarray]:
    if (
        not bool(getattr(capture, "is_moe", False))
        or capture.moe_routing_weights_f32 is None
        or capture.moe_shared_out_f32 is None
        or capture.moe_shared_gate_f32 is None
    ):
        return {}
    hidden_size = int(capture.hidden_size)
    top_k = int(capture.top_k)
    down = np.ascontiguousarray(capture.ffn_or_moe_down_f32, dtype=np.float32).reshape(top_k, hidden_size)
    weights = np.ascontiguousarray(capture.moe_routing_weights_f32, dtype=np.float32).reshape(top_k)
    weighted_rows = np.ascontiguousarray(down * weights.reshape(top_k, 1), dtype=np.float32)
    selected_acc = np.zeros((hidden_size,), dtype=np.float32)
    for index in range(top_k):
        selected_acc = np.float32(selected_acc + weighted_rows[index])
    # The fused combine kernel preserves the old two-kernel contract: selected
    # expert sum rounds to BF16 before adding the shared expert and residual.
    selected_bf16 = _bf16_roundtrip_f32(selected_acc)
    shared = np.ascontiguousarray(capture.moe_shared_out_f32, dtype=np.float32).reshape(hidden_size)
    gate_logit = float(np.asarray(capture.moe_shared_gate_f32, dtype=np.float32).reshape(-1)[0])
    gate = np.float32(1.0 / (1.0 + np.exp(np.float32(-gate_logit))))
    shared_gated = np.ascontiguousarray(gate * shared, dtype=np.float32)
    ffn_out = np.ascontiguousarray(selected_bf16 + gate * shared, dtype=np.float32)
    residual = np.ascontiguousarray(capture.residual_f32, dtype=np.float32).reshape(hidden_size)
    return {
        "moe_selected_down_weighted": weighted_rows,
        "moe_selected_weighted_sum_f32": selected_acc,
        "moe_selected_weighted_bf16": selected_bf16,
        "moe_shared_gated": shared_gated,
        "ffn_out_combined_from_components": ffn_out,
        "post_moe_rounded_from_components": _bf16_roundtrip_f32(residual + ffn_out),
    }


def _probe_bulk_or_native(
    session: Any,
    verifier_inputs: list[int],
    *,
    mode: str,
    use_wmma_prefill: bool,
    capture_pre_output_norm_hidden: bool,
    capture_layer_output_hidden: list[int],
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray | None, dict[int, np.ndarray] | None]:
    result = session.verify_target_block(
        verifier_inputs,
        bulk_attention_mode=mode,
        use_wmma_prefill=bool(use_wmma_prefill),
        capture_pre_output_norm_hidden=bool(capture_pre_output_norm_hidden),
        capture_layer_output_hidden=capture_layer_output_hidden,
        record_stage_timings=False,
    )
    return (
        [int(token) for token in result.token_ids],
        _copy_verify_logits(session, len(verifier_inputs)),
        np.ascontiguousarray(result.hidden_seeds, dtype=np.float32),
        (
            None
            if result.pre_output_norm_hidden is None
            else np.ascontiguousarray(result.pre_output_norm_hidden, dtype=np.float32)
        ),
        (
            None
            if result.layer_output_hidden is None
            else {
                int(layer_id): np.ascontiguousarray(hidden, dtype=np.float32)
                for layer_id, hidden in result.layer_output_hidden.items()
            }
        ),
    )


def _probe_serial(
    session: Any,
    verifier_inputs: list[int],
    *,
    capture_pre_output_norm_hidden: bool,
    capture_layer_output_hidden: list[int],
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray | None, dict[int, np.ndarray] | None]:
    token_ids: list[int] = []
    logits_rows: list[np.ndarray] = []
    hidden_rows: list[np.ndarray] = []
    pre_output_norm_rows: list[np.ndarray] = []
    layer_output_rows: dict[int, list[np.ndarray]] = {int(layer_id): [] for layer_id in capture_layer_output_hidden}
    for token in verifier_inputs:
        row = session.step(
            int(token),
            return_logits=True,
            capture_hidden_seed_fp32=True,
            capture_pre_output_norm_hidden=bool(capture_pre_output_norm_hidden),
            capture_layer_output_hidden=capture_layer_output_hidden,
        )
        token_ids.append(int(row.token_id))
        logits_rows.append(np.ascontiguousarray(row.logits.reshape(-1), dtype=np.float32))
        hidden_rows.append(_copy_current_hidden_seed(session).reshape(-1))
        if capture_pre_output_norm_hidden:
            pre_hidden = session.last_pre_output_norm_hidden
            if pre_hidden is None:
                raise ProbeError("pre-output_norm hidden was requested but not captured")
            pre_output_norm_rows.append(np.ascontiguousarray(pre_hidden.reshape(-1), dtype=np.float32))
        if capture_layer_output_hidden:
            last_layer_hidden = session.last_layer_output_hidden
            for layer_id in capture_layer_output_hidden:
                layer_hidden = last_layer_hidden.get(int(layer_id))
                if layer_hidden is None:
                    raise ProbeError(f"layer {layer_id} output hidden was requested but not captured")
                layer_output_rows[int(layer_id)].append(np.ascontiguousarray(layer_hidden.reshape(-1), dtype=np.float32))
    return (
        token_ids,
        np.ascontiguousarray(np.stack(logits_rows, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.stack(hidden_rows, axis=0), dtype=np.float32),
        (
            np.ascontiguousarray(np.stack(pre_output_norm_rows, axis=0), dtype=np.float32)
            if capture_pre_output_norm_hidden
            else None
        ),
        (
            {
                int(layer_id): np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)
                for layer_id, rows in sorted(layer_output_rows.items())
            }
            if capture_layer_output_hidden
            else None
        ),
    )


def _boundary_array_summaries(
    capture: Any,
    *,
    row_index: int,
    input_token: int,
    position: int,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    arrays: dict[str, np.ndarray | None] = {
        "hidden_in": capture.hidden_in_f32,
        "attn_norm": capture.attn_norm_f32,
        "linear_qkv": getattr(capture, "linear_qkv_f32", None),
        "linear_z": getattr(capture, "linear_z_f32", None),
        "ssm_alpha": getattr(capture, "ssm_alpha_f32", None),
        "ssm_beta": getattr(capture, "ssm_beta_f32", None),
        "conv_out": getattr(capture, "conv_out_f32", None),
        "recurrent_out": getattr(capture, "recurrent_out_f32", None),
        "recurrent_bf16": getattr(capture, "recurrent_bf16_f32", None),
        "attn_out": capture.attn_out_f32,
        "attn_residual": capture.residual_f32,
        "attn_post_norm": capture.post_norm_f32,
        "moe_router_logits": getattr(capture, "moe_router_logits_f32", None),
        "moe_selected_swiglu": getattr(capture, "moe_selected_intermediate_f32", None),
        "ffn_or_moe_down": capture.ffn_or_moe_down_f32,
        "moe_shared_intermediate": getattr(capture, "moe_shared_intermediate_f32", None),
        "moe_shared_out": capture.moe_shared_out_f32,
        "post_moe_delta_from_residual": (
            np.asarray(capture.layer_out_f32, dtype=np.float32).reshape(-1)
            - np.asarray(capture.residual_f32, dtype=np.float32).reshape(-1)
        ),
        "layer_out": capture.layer_out_f32,
    }
    arrays.update(_capture_moe_component_arrays(capture))
    summaries: dict[str, Any] = {}
    values: dict[str, list[float]] = {}
    for name, array in arrays.items():
        if array is None:
            continue
        row = np.ascontiguousarray(np.asarray(array, dtype=np.float32).reshape(-1), dtype=np.float32)
        summaries[name] = hidden_state_summary(
            row,
            label=f"target_verify_layer_{int(capture.layer_id)}_{name}",
            depth=int(row_index),
            token_id=int(input_token),
            position=int(position),
        )
        values[name] = [float(value) for value in row]
    return summaries, values


def _capture_single_layer_boundary(
    *,
    args: argparse.Namespace,
    compiler_version: str | None,
    prompt_tokens: list[int],
    cycles_to_replay: list[dict[str, Any]],
    initial_prev_token: int,
    initial_prev_position: int,
    cycle_prev_token: int,
    cycle_start_position: int,
    verifier_inputs: list[int],
    target_tokens: list[int],
    layer_id: int,
    row_index: int,
    max_sequence_length: int,
    include_raw: bool,
) -> dict[str, Any]:
    if row_index < 0 or row_index >= len(verifier_inputs):
        raise ProbeError(
            f"layer boundary row {row_index} outside verifier input length {len(verifier_inputs)}"
        )

    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    session = Qwen35GGUFResidentSession(
        args.model,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=bool(args.use_wmma_prefill),
        use_gemv_decode=bool(args.use_gemv_decode),
        prefill_config=PrefillConfig(),
    )
    try:
        prefill = session.prefill(
            prompt_tokens,
            use_bulk=True,
            bulk_attention_mode=str(args.prefill_attention_mode),
            return_logits=False,
            capture_hidden_seed_fp32=True,
        )
        current_prev = int(prefill.token_id)
        if current_prev != initial_prev_token:
            raise ProbeError(
                f"boundary prefill token {current_prev} does not match trace initial_prev_token {initial_prev_token}"
            )
        if int(session.position) != initial_prev_position:
            raise ProbeError(
                f"boundary prefill position {session.position} does not match trace {initial_prev_position}"
            )
        current_prev, replay_rows = _replay_prior_cycles(
            session,
            cycles_to_replay,
            current_prev=current_prev,
            mode=str(args.replay_target_block_verify_mode or args.target_block_verify_mode),
            use_wmma_prefill=bool(args.target_block_wmma_prefill),
        )
        if current_prev != cycle_prev_token:
            raise ProbeError(
                f"boundary replay prev token {current_prev} does not match cycle prev {cycle_prev_token}"
            )
        if int(session.position) != cycle_start_position:
            raise ProbeError(
                f"boundary replay position {session.position} does not match cycle start {cycle_start_position}"
            )

        prior_row_replay: list[dict[str, Any]] = []
        for prior_row in range(row_index):
            prior_input = int(verifier_inputs[prior_row])
            result = session.step(prior_input, return_logits=False)
            sampled = int(result.token_id)
            expected = int(target_tokens[prior_row]) if prior_row < len(target_tokens) else None
            if expected is not None and sampled != expected:
                raise ProbeError(
                    f"boundary row replay mismatch at row {prior_row}: sampled {sampled}, trace {expected}"
                )
            prior_row_replay.append(
                {
                    "row": int(prior_row),
                    "position": int(cycle_start_position + prior_row),
                    "input_token": int(prior_input),
                    "sampled_token": int(sampled),
                    "trace_target_token": expected,
                }
            )
        if int(session.position) != int(cycle_start_position + row_index):
            raise ProbeError(
                f"boundary capture position {session.position} does not match row position "
                f"{cycle_start_position + row_index}"
            )

        input_token = int(verifier_inputs[row_index])
        position = int(cycle_start_position + row_index)
        capture = session.capture_attention_layer(
            input_token,
            position=position,
            layer_id=int(layer_id),
            run_preceding_layers=True,
        )
        summaries, values = _boundary_array_summaries(
            capture,
            row_index=int(row_index),
            input_token=input_token,
            position=position,
        )
        record: dict[str, Any] = {
            "layer": int(layer_id),
            "row": int(row_index),
            "position": int(position),
            "input_token": int(input_token),
            "trace_target_token": int(target_tokens[row_index]) if row_index < len(target_tokens) else None,
            "capture": capture.as_summary_dict(),
            "prior_cycle_replay": replay_rows,
            "prior_row_replay": prior_row_replay,
            "summaries": summaries,
            "moe_selected_experts": (
                None
                if capture.moe_selected_experts_i64 is None
                else [int(value) for value in np.asarray(capture.moe_selected_experts_i64).reshape(-1)]
            ),
            "moe_routing_weights": (
                None
                if capture.moe_routing_weights_f32 is None
                else [float(value) for value in np.asarray(capture.moe_routing_weights_f32, dtype=np.float32).reshape(-1)]
            ),
            "moe_shared_gate": (
                None
                if capture.moe_shared_gate_f32 is None
                else [float(value) for value in np.asarray(capture.moe_shared_gate_f32, dtype=np.float32).reshape(-1)]
            ),
        }
        if include_raw:
            record["values"] = values
        return record
    finally:
        session.close()


def _capture_single_row_router_trace(
    *,
    args: argparse.Namespace,
    compiler_version: str | None,
    prompt_tokens: list[int],
    cycles_to_replay: list[dict[str, Any]],
    initial_prev_token: int,
    initial_prev_position: int,
    cycle_prev_token: int,
    cycle_start_position: int,
    verifier_inputs: list[int],
    target_tokens: list[int],
    row_index: int,
    max_sequence_length: int,
) -> dict[str, Any]:
    if row_index < 0 or row_index >= len(verifier_inputs):
        raise ProbeError(
            f"router trace row {row_index} outside verifier input length {len(verifier_inputs)}"
        )

    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    session = Qwen35GGUFResidentSession(
        args.model,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=bool(args.use_wmma_prefill),
        use_gemv_decode=bool(args.use_gemv_decode),
        prefill_config=PrefillConfig(),
    )
    try:
        prefill = session.prefill(
            prompt_tokens,
            use_bulk=True,
            bulk_attention_mode=str(args.prefill_attention_mode),
            return_logits=False,
            capture_hidden_seed_fp32=True,
        )
        current_prev = int(prefill.token_id)
        if current_prev != initial_prev_token:
            raise ProbeError(
                f"router trace prefill token {current_prev} does not match trace initial_prev_token {initial_prev_token}"
            )
        if int(session.position) != initial_prev_position:
            raise ProbeError(
                f"router trace prefill position {session.position} does not match trace {initial_prev_position}"
            )
        current_prev, replay_rows = _replay_prior_cycles(
            session,
            cycles_to_replay,
            current_prev=current_prev,
            mode=str(args.replay_target_block_verify_mode or args.target_block_verify_mode),
            use_wmma_prefill=bool(args.target_block_wmma_prefill),
        )
        if current_prev != cycle_prev_token:
            raise ProbeError(
                f"router trace replay prev token {current_prev} does not match cycle prev {cycle_prev_token}"
            )
        if int(session.position) != cycle_start_position:
            raise ProbeError(
                f"router trace replay position {session.position} does not match cycle start {cycle_start_position}"
            )

        prior_row_replay: list[dict[str, Any]] = []
        for prior_row in range(row_index):
            prior_input = int(verifier_inputs[prior_row])
            result = session.step(prior_input, return_logits=False)
            sampled = int(result.token_id)
            expected = int(target_tokens[prior_row]) if prior_row < len(target_tokens) else None
            if expected is not None and sampled != expected:
                raise ProbeError(
                    f"router trace row replay mismatch at row {prior_row}: sampled {sampled}, trace {expected}"
                )
            prior_row_replay.append(
                {
                    "row": int(prior_row),
                    "position": int(cycle_start_position + prior_row),
                    "input_token": int(prior_input),
                    "sampled_token": int(sampled),
                    "trace_target_token": expected,
                }
            )
        if int(session.position) != int(cycle_start_position + row_index):
            raise ProbeError(
                f"router trace position {session.position} does not match row position "
                f"{cycle_start_position + row_index}"
            )

        input_token = int(verifier_inputs[row_index])
        position = int(cycle_start_position + row_index)
        captures = session.capture_attention_router_trace(
            input_token,
            position=position,
        )
        layers: list[dict[str, Any]] = []
        for capture in captures:
            router_logits = np.ascontiguousarray(
                capture.moe_router_logits_f32, dtype=np.float32
            ).reshape(-1)
            layers.append(
                {
                    "layer": int(capture.layer_id),
                    "layer_type": str(capture.layer_type),
                    "capture": capture.as_summary_dict(),
                    "hidden_in_summary": hidden_state_summary(
                        np.ascontiguousarray(capture.hidden_in_f32, dtype=np.float32).reshape(-1),
                        label=f"target_verify_layer_{int(capture.layer_id)}_router_hidden_in",
                        depth=int(row_index),
                        token_id=input_token,
                        position=position,
                    ),
                    "layer_out_summary": hidden_state_summary(
                        np.ascontiguousarray(capture.layer_out_f32, dtype=np.float32).reshape(-1),
                        label=f"target_verify_layer_{int(capture.layer_id)}_router_layer_out",
                        depth=int(row_index),
                        token_id=input_token,
                        position=position,
                    ),
                    "moe_selected_experts": [
                        int(value)
                        for value in np.asarray(capture.moe_selected_experts_i64).reshape(-1)
                    ],
                    "moe_routing_weights": [
                        float(value)
                        for value in np.asarray(
                            capture.moe_routing_weights_f32, dtype=np.float32
                        ).reshape(-1)
                    ],
                    "moe_shared_gate": [
                        float(value)
                        for value in np.asarray(
                            capture.moe_shared_gate_f32, dtype=np.float32
                        ).reshape(-1)
                    ],
                    "router_top_k": _top_k_rows(router_logits, top_k=int(capture.top_k)),
                    "values": {
                        "moe_router_logits": [float(value) for value in router_logits],
                    },
                }
            )
        return {
            "row": int(row_index),
            "position": int(position),
            "input_token": int(input_token),
            "trace_target_token": int(target_tokens[row_index])
            if row_index < len(target_tokens)
            else None,
            "prior_cycle_replay": replay_rows,
            "prior_row_replay": prior_row_replay,
            "layers": layers,
        }
    finally:
        session.close()


def _run_trace_block(
    session: Any,
    verifier_inputs: list[int],
    *,
    mode: str,
    use_wmma_prefill: bool,
    capture_linear_state_rows: bool,
    defer_linear_state_commit: bool,
) -> list[int]:
    if mode == "serial-exact":
        result = session.verify_target_block_serial_exact(
            verifier_inputs,
            capture_linear_state_rows=bool(capture_linear_state_rows),
        )
    else:
        result = session.verify_target_block(
            verifier_inputs,
            bulk_attention_mode=mode,
            use_wmma_prefill=bool(use_wmma_prefill),
            capture_linear_state_rows=bool(capture_linear_state_rows),
            defer_linear_state_commit=bool(defer_linear_state_commit),
            record_stage_timings=False,
        )
    if capture_linear_state_rows and not result.linear_state_rows_captured:
        raise ProbeError("trace replay requested linear-state rows, but verifier did not capture them")
    return [int(token) for token in result.token_ids]


def _replay_prior_cycles(
    session: Any,
    cycles: list[dict[str, Any]],
    *,
    current_prev: int,
    mode: str,
    use_wmma_prefill: bool,
) -> tuple[int, list[dict[str, Any]]]:
    replay_rows: list[dict[str, Any]] = []
    for row in cycles:
        cycle_number = int(row.get("cycle", len(replay_rows)))
        start_position = int(row["cycle_start_seq_position"])
        if int(session.position) != start_position:
            raise ProbeError(
                f"cycle {cycle_number} replay position {session.position} does not match trace {start_position}"
            )
        cycle_prev = int(row["cycle_prev_token"])
        if int(current_prev) != cycle_prev:
            raise ProbeError(
                f"cycle {cycle_number} replay prev {current_prev} does not match trace {cycle_prev}"
            )
        draft_tokens = [int(token) for token in row.get("draft_tokens", [])]
        trace_target_tokens = [int(token) for token in row.get("target_tokens", [])]
        trace_output_tokens = [int(token) for token in row.get("output_tokens", [])]
        accepted = int(row.get("accepted_draft_tokens", _llama_accept_count(draft_tokens, trace_target_tokens)))
        consumed_rows = accepted + 1
        verifier_inputs = [cycle_prev] + draft_tokens
        if consumed_rows <= 0 or consumed_rows > len(verifier_inputs):
            raise ProbeError(
                f"cycle {cycle_number} consumed rows {consumed_rows} outside verifier input length {len(verifier_inputs)}"
            )
        sampled_tokens = _run_trace_block(
            session,
            verifier_inputs,
            mode=mode,
            use_wmma_prefill=bool(use_wmma_prefill),
            capture_linear_state_rows=True,
            defer_linear_state_commit=(mode != "serial-exact"),
        )
        if sampled_tokens[: len(trace_target_tokens)] != trace_target_tokens:
            raise ProbeError(
                f"cycle {cycle_number} target replay mismatch: sampled {sampled_tokens}, trace {trace_target_tokens}"
            )
        if trace_output_tokens != sampled_tokens[:consumed_rows]:
            raise ProbeError(
                f"cycle {cycle_number} output replay mismatch: sampled {sampled_tokens[:consumed_rows]}, "
                f"trace {trace_output_tokens}"
            )
        session._commit_verify_linear_state_row(consumed_rows - 1, position=start_position + consumed_rows)
        current_prev = int(trace_output_tokens[-1])
        replay_rows.append(
            {
                "cycle": int(cycle_number),
                "start_position": int(start_position),
                "verifier_input_tokens": verifier_inputs,
                "sampled_tokens": sampled_tokens,
                "accepted_draft_tokens": int(accepted),
                "consumed_rows": int(consumed_rows),
                "committed_position": int(start_position + consumed_rows),
                "next_prev_token": int(current_prev),
            }
        )
    return int(current_prev), replay_rows


def exact_command_payload(argv: Sequence[object]) -> dict[str, Any]:
    argv_strings = [str(item) for item in argv]
    return {"argv": argv_strings, "command": shlex.join(argv_strings)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(GGUF_PATH))
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-token", action="append", type=_parse_token_csv, default=[])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--target-block-verify-mode", choices=("bulk", "native", "serial-exact"), default="bulk")
    parser.add_argument("--replay-target-block-verify-mode", choices=("bulk", "native", "serial-exact"), default=None)
    parser.add_argument("--target-block-wmma-prefill", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decode-repack", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-wmma-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gemv-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefill-attention-mode", choices=("bulk", "native"), default="bulk")
    parser.add_argument(
        "--raw-hidden-row",
        action="append",
        type=int,
        default=[],
        help="Diagnostic only: include full FP32 hidden_seed values for the given verifier row index.",
    )
    parser.add_argument(
        "--pre-output-norm-row",
        action="append",
        type=int,
        default=[],
        help="Diagnostic only: include a summary of the pre-output_norm BF16 residual row at the given verifier row index.",
    )
    parser.add_argument(
        "--raw-pre-output-norm-row",
        action="append",
        type=int,
        default=[],
        help="Diagnostic only: include full FP32 values for the pre-output_norm BF16 residual row at the given verifier row index.",
    )
    parser.add_argument(
        "--layer-output-row",
        action="append",
        type=_parse_layer_row,
        default=[],
        metavar="LAYER:ROW",
        help="Diagnostic only: include a summary of the post-layer BF16 residual row for LAYER:ROW.",
    )
    parser.add_argument(
        "--raw-layer-output-row",
        action="append",
        type=_parse_layer_row,
        default=[],
        metavar="LAYER:ROW",
        help="Diagnostic only: include full FP32 post-layer BF16 residual values for LAYER:ROW.",
    )
    parser.add_argument(
        "--layer-boundary-row",
        action="append",
        type=_parse_layer_row,
        default=[],
        metavar="LAYER:ROW",
        help="Diagnostic only: include a summary of sub-boundary buffers inside one target layer for LAYER:ROW.",
    )
    parser.add_argument(
        "--raw-layer-boundary-row",
        action="append",
        type=_parse_layer_row,
        default=[],
        metavar="LAYER:ROW",
        help="Diagnostic only: include full FP32 sub-boundary buffer values inside one target layer for LAYER:ROW.",
    )
    parser.add_argument(
        "--router-trace-row",
        action="append",
        type=int,
        default=[],
        help="Diagnostic only: include per-layer MoE router logits/top-k for one verifier row.",
    )
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if not _hip_available():
        raise ProbeError("ROCm/HIP is not available")
    if not args.model.exists():
        raise ProbeError(f"model not found: {args.model}")
    trace = _read_json(args.trace)
    workload = trace.get("workload")
    if not isinstance(workload, dict):
        raise ProbeError("trace is missing workload{}")
    cycles = trace.get("cycles")
    if not isinstance(cycles, list) or not all(isinstance(row, dict) for row in cycles):
        raise ProbeError("trace is missing cycles[]")
    cycle_index, cycle = _find_cycle(cycles, int(args.cycle))
    prompt = str(workload.get("prompt", ""))
    if not prompt:
        raise ProbeError("trace workload is missing prompt")
    prompt_reasoning = str(workload.get("prompt_reasoning", "off"))
    initial_prev_token = int(workload["initial_prev_token"])
    initial_prev_position = int(workload["initial_prev_position"])
    cycle_prev_token = int(cycle["cycle_prev_token"])
    cycle_start_position = int(cycle["cycle_start_seq_position"])
    draft_tokens = [int(token) for token in cycle.get("draft_tokens", [])]
    target_tokens = [int(token) for token in cycle.get("target_tokens", [])]
    output_tokens = [int(token) for token in cycle.get("output_tokens", [])]
    prefix_visible_outputs = _flatten_output_tokens(cycles[:cycle_index])
    if not prefix_visible_outputs:
        raise ProbeError("forced cycle must have at least one prior visible output token")
    if int(prefix_visible_outputs[-1]) != cycle_prev_token:
        raise ProbeError(
            f"prefix final output {prefix_visible_outputs[-1]} does not match cycle_prev_token {cycle_prev_token}"
        )
    consumed_prefix_tokens = [initial_prev_token] + prefix_visible_outputs[:-1]
    expected_start_position = initial_prev_position + len(consumed_prefix_tokens)
    if expected_start_position != cycle_start_position:
        raise ProbeError(
            f"derived start position {expected_start_position} does not match trace {cycle_start_position}"
        )
    verifier_inputs = [cycle_prev_token] + draft_tokens
    extra_candidates = [token for values in args.candidate_token for token in values]
    candidate_tokens = _unique_ints([*draft_tokens, *target_tokens, *output_tokens, *extra_candidates])
    raw_hidden_rows = {int(row) for row in args.raw_hidden_row}
    pre_output_norm_rows = {int(row) for row in args.pre_output_norm_row}
    raw_pre_output_norm_rows = {int(row) for row in args.raw_pre_output_norm_row}
    pre_output_norm_capture_rows = pre_output_norm_rows | raw_pre_output_norm_rows
    layer_output_rows = {(int(layer), int(row)) for layer, row in args.layer_output_row}
    raw_layer_output_rows = {(int(layer), int(row)) for layer, row in args.raw_layer_output_row}
    layer_output_capture_rows = layer_output_rows | raw_layer_output_rows
    capture_layer_output_hidden = sorted({int(layer) for layer, _row in layer_output_capture_rows})
    layer_boundary_rows = {(int(layer), int(row)) for layer, row in args.layer_boundary_row}
    raw_layer_boundary_rows = {(int(layer), int(row)) for layer, row in args.raw_layer_boundary_row}
    layer_boundary_capture_rows = sorted(layer_boundary_rows | raw_layer_boundary_rows)
    for layer_id, row_index in layer_boundary_capture_rows:
        if int(row_index) >= len(verifier_inputs):
            raise ProbeError(
                f"layer boundary row {row_index} outside verifier input length {len(verifier_inputs)}"
            )
    router_trace_rows = sorted({int(row) for row in args.router_trace_row})
    for row_index in router_trace_rows:
        if int(row_index) < 0 or int(row_index) >= len(verifier_inputs):
            raise ProbeError(
                f"router trace row {row_index} outside verifier input length {len(verifier_inputs)}"
            )
    route_env = _apply_trace_route_env(workload, decode_repack=args.decode_repack)
    if _env_truthy("HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A"):
        raise ProbeError("direct verifier top-1 path is enabled; full row logits would be unavailable")

    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8")
    gguf_info = scan_gguf(args.model)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(gguf_info)
    prompt_tokens = build_chat_prompt(tokenizer, prompt, reasoning=prompt_reasoning)
    if int(workload.get("prompt_tokens", len(prompt_tokens))) != len(prompt_tokens):
        raise ProbeError(
            f"rebuilt prompt has {len(prompt_tokens)} tokens; trace has {workload.get('prompt_tokens')}"
        )

    max_sequence_length = max(
        int(cycle_start_position) + len(verifier_inputs) + 4,
        len(prompt_tokens) + len(consumed_prefix_tokens) + len(verifier_inputs) + 4,
    )
    session = Qwen35GGUFResidentSession(
        args.model,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=bool(args.use_wmma_prefill),
        use_gemv_decode=bool(args.use_gemv_decode),
        prefill_config=PrefillConfig(),
    )
    try:
        prefill = session.prefill(
            prompt_tokens,
            use_bulk=True,
            bulk_attention_mode=str(args.prefill_attention_mode),
            return_logits=False,
            capture_hidden_seed_fp32=True,
        )
        current_prev = int(prefill.token_id)
        if current_prev != initial_prev_token:
            raise ProbeError(f"prefill token {current_prev} does not match trace initial_prev_token {initial_prev_token}")
        if int(session.position) != initial_prev_position:
            raise ProbeError(f"prefill position {session.position} does not match trace {initial_prev_position}")
        current_prev, replay_rows = _replay_prior_cycles(
            session,
            cycles[:cycle_index],
            current_prev=current_prev,
            mode=str(args.replay_target_block_verify_mode or args.target_block_verify_mode),
            use_wmma_prefill=bool(args.target_block_wmma_prefill),
        )
        if current_prev != cycle_prev_token:
            raise ProbeError(f"replayed prev token {current_prev} does not match cycle prev {cycle_prev_token}")
        if int(session.position) != cycle_start_position:
            raise ProbeError(f"replayed position {session.position} does not match cycle start {cycle_start_position}")
        cycle_pending_hidden_seed = _copy_current_hidden_seed(session)

        if args.target_block_verify_mode == "serial-exact":
            sampled_tokens, logits, hidden_seeds, pre_output_norm_hidden, layer_output_hidden = _probe_serial(
                session,
                verifier_inputs,
                capture_pre_output_norm_hidden=bool(pre_output_norm_capture_rows),
                capture_layer_output_hidden=capture_layer_output_hidden,
            )
        else:
            sampled_tokens, logits, hidden_seeds, pre_output_norm_hidden, layer_output_hidden = _probe_bulk_or_native(
                session,
                verifier_inputs,
                mode=str(args.target_block_verify_mode),
                use_wmma_prefill=bool(args.target_block_wmma_prefill),
                capture_pre_output_norm_hidden=bool(pre_output_norm_capture_rows),
                capture_layer_output_hidden=capture_layer_output_hidden,
            )
        accepted_draft_tokens = _llama_accept_count(draft_tokens, sampled_tokens[: len(draft_tokens)])
        rows: list[dict[str, Any]] = []
        for row_index, (input_token, sampled_token) in enumerate(zip(verifier_inputs, sampled_tokens, strict=True)):
            row_logits = logits[row_index]
            row_record = {
                "row": int(row_index),
                "position": int(cycle_start_position + row_index),
                "input_token": int(input_token),
                "sampled_token": int(sampled_token),
                "trace_target_token": int(target_tokens[row_index]) if row_index < len(target_tokens) else None,
                "trace_output_token": int(output_tokens[row_index]) if row_index < len(output_tokens) else None,
                "draft_token_at_depth": int(draft_tokens[row_index]) if row_index < len(draft_tokens) else None,
                "hidden_seed_summary": hidden_state_summary(
                    hidden_seeds[row_index],
                    label="target_verify_hidden_seed",
                    depth=int(row_index),
                    token_id=int(input_token),
                    position=int(cycle_start_position + row_index),
                ),
                "top_k": _top_k_rows(row_logits, top_k=int(args.top_k)),
                "candidate_scores": _candidate_rows(row_logits, candidate_tokens),
            }
            if int(row_index) in raw_hidden_rows:
                row_record["hidden_seed_values"] = [
                    float(value)
                    for value in np.ascontiguousarray(hidden_seeds[row_index], dtype=np.float32).reshape(-1)
                ]
            if int(row_index) in pre_output_norm_capture_rows:
                if pre_output_norm_hidden is None:
                    raise ProbeError("pre-output_norm rows were requested but not captured")
                pre_row = np.ascontiguousarray(pre_output_norm_hidden[row_index], dtype=np.float32).reshape(-1)
                row_record["pre_output_norm_hidden_summary"] = hidden_state_summary(
                    pre_row,
                    label="target_verify_pre_output_norm_hidden",
                    depth=int(row_index),
                    token_id=int(input_token),
                    position=int(cycle_start_position + row_index),
                )
                if int(row_index) in raw_pre_output_norm_rows:
                    row_record["pre_output_norm_hidden_values"] = [float(value) for value in pre_row]
            row_layer_ids = sorted(
                layer_id
                for layer_id, requested_row in layer_output_capture_rows
                if int(requested_row) == int(row_index)
            )
            if row_layer_ids:
                if layer_output_hidden is None:
                    raise ProbeError("layer output rows were requested but not captured")
                layer_summaries: dict[str, Any] = {}
                layer_values: dict[str, list[float]] = {}
                for layer_id in row_layer_ids:
                    layer_rows = layer_output_hidden.get(int(layer_id))
                    if layer_rows is None:
                        raise ProbeError(f"layer {layer_id} output rows were requested but not captured")
                    layer_row = np.ascontiguousarray(layer_rows[row_index], dtype=np.float32).reshape(-1)
                    layer_summaries[str(layer_id)] = hidden_state_summary(
                        layer_row,
                        label=f"target_verify_layer_output_{layer_id}",
                        depth=int(row_index),
                        token_id=int(input_token),
                        position=int(cycle_start_position + row_index),
                    )
                    if (int(layer_id), int(row_index)) in raw_layer_output_rows:
                        layer_values[str(layer_id)] = [float(value) for value in layer_row]
                row_record["layer_output_hidden_summaries"] = layer_summaries
                if layer_values:
                    row_record["layer_output_hidden_values"] = layer_values
            rows.append(row_record)
    finally:
        session.close()

    layer_boundary_captures: list[dict[str, Any]] = []
    for layer_id, row_index in layer_boundary_capture_rows:
        layer_boundary_captures.append(
            _capture_single_layer_boundary(
                args=args,
                compiler_version=compiler_version,
                prompt_tokens=prompt_tokens,
                cycles_to_replay=cycles[:cycle_index],
                initial_prev_token=int(initial_prev_token),
                initial_prev_position=int(initial_prev_position),
                cycle_prev_token=int(cycle_prev_token),
                cycle_start_position=int(cycle_start_position),
                verifier_inputs=verifier_inputs,
                target_tokens=target_tokens,
                layer_id=int(layer_id),
                row_index=int(row_index),
                max_sequence_length=int(max_sequence_length),
                include_raw=(int(layer_id), int(row_index)) in raw_layer_boundary_rows,
            )
        )
    router_trace_captures: list[dict[str, Any]] = []
    for row_index in router_trace_rows:
        router_trace_captures.append(
            _capture_single_row_router_trace(
                args=args,
                compiler_version=compiler_version,
                prompt_tokens=prompt_tokens,
                cycles_to_replay=cycles[:cycle_index],
                initial_prev_token=int(initial_prev_token),
                initial_prev_position=int(initial_prev_position),
                cycle_prev_token=int(cycle_prev_token),
                cycle_start_position=int(cycle_start_position),
                verifier_inputs=verifier_inputs,
                target_tokens=target_tokens,
                row_index=int(row_index),
                max_sequence_length=int(max_sequence_length),
            )
        )

    artifact = {
        "schema": 1,
        "kind": "hipengine_gguf_mtp_forced_target_probe",
        "status": "complete",
        "performance_claim": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repo": _repo_provenance(),
        "model": str(args.model),
        "source_trace": str(args.trace),
        "command": exact_command_payload(sys.argv),
        "route_env": route_env,
        "probe": {
            "cycle": int(args.cycle),
            "cycle_index": int(cycle_index),
            "prompt": prompt,
            "prompt_reasoning": prompt_reasoning,
            "prompt_tokens": int(len(prompt_tokens)),
            "initial_prev_token": int(initial_prev_token),
            "initial_prev_position": int(initial_prev_position),
            "prefix_visible_output_tokens": prefix_visible_outputs,
            "consumed_prefix_tokens": consumed_prefix_tokens,
            "cycle_prev_token": int(cycle_prev_token),
            "cycle_start_seq_position": int(cycle_start_position),
            "verifier_input_tokens": verifier_inputs,
            "trace_draft_tokens": draft_tokens,
            "trace_target_tokens": target_tokens,
            "trace_output_tokens": output_tokens,
            "candidate_tokens": candidate_tokens,
        },
        "result": {
            "target_block_verify_mode": str(args.target_block_verify_mode),
            "replay_target_block_verify_mode": str(args.replay_target_block_verify_mode or args.target_block_verify_mode),
            "target_block_wmma_prefill": bool(args.target_block_wmma_prefill),
            "prior_cycle_replay": replay_rows,
            "cycle_pending_hidden_seed_summary": hidden_state_summary(
                cycle_pending_hidden_seed,
                label="cycle_pending_hidden_seed",
                depth=-1,
                token_id=int(cycle_prev_token),
                position=int(cycle_start_position),
            ),
            "sampled_tokens": sampled_tokens,
            "accepted_draft_tokens": int(accepted_draft_tokens),
            "rows": rows,
            "layer_boundary_captures": layer_boundary_captures,
            "router_trace_captures": router_trace_captures,
        },
        "notes": [
            "Diagnostic only: forced-prefix target verifier score capture, not a performance run.",
            "Prefix replay consumes initial_prev_token plus prior visible outputs except the final cycle_prev_token, matching scripts/gguf_mtp_bench.py block verifier state.",
            "The verifier direct Q6 top-1 path is forced off so full row logits are available.",
            "cycle_pending_hidden_seed_summary is the seed row that starts the MTP draft for this cycle.",
            "hidden_seed_summary is the FP32 post-output_norm verifier row used as the MTP draft seed.",
            "hidden_seed_values is emitted only for --raw-hidden-row diagnostics.",
            "pre_output_norm_hidden_summary is emitted only for requested --pre-output-norm-row or --raw-pre-output-norm-row diagnostics.",
            "pre_output_norm_hidden_values is emitted only for --raw-pre-output-norm-row diagnostics.",
            "layer_output_hidden_summaries is emitted only for requested --layer-output-row or --raw-layer-output-row LAYER:ROW diagnostics.",
            "layer_output_hidden_values is emitted only for --raw-layer-output-row LAYER:ROW diagnostics.",
            "layer_boundary_captures is emitted only for requested --layer-boundary-row or --raw-layer-boundary-row LAYER:ROW diagnostics.",
            "Layer boundary captures run in isolated replay sessions so diagnostic sub-layer taps do not perturb the scored verifier probe.",
            "Layer boundary captures include attn_norm and, for MoE layers, router logits, selected SwigLU/down/weighted sums, shared intermediate/out/gated contribution, host-reconstructed ffn_out_combined_from_components, and post_moe_rounded_from_components.",
            "router_trace_captures is emitted only for requested --router-trace-row ROW diagnostics and runs one isolated row replay through all layers, capturing per-layer router logits/top-k without perturbing the scored verifier probe.",
            "post_moe_delta_from_residual is derived as layer_out - attn_residual and is an approximate bridge to llama.cpp post_moe/ffn residual comparisons after BF16 output rounding.",
        ],
    }
    text = json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
