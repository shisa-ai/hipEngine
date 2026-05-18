#!/usr/bin/env python3
"""Probe layer-3 full-attention stages after accepted native linear prefill.

This diagnostic compares the first full-attention layer's c=1 stages for the
last prompt token when the first three linear-attention layers are run via:

1. fully serial token-by-token resident prefill, and
2. accepted native linear-prefix prefill followed by serial c=1 layer-3
   full-attention execution.

It is correctness/blocker evidence only, not a benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.runtime import embedding_lookup_batch_fp16_i64
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)

LINEAR_PREFIX_LAYERS = 3
FULL_ATTENTION_LAYER = 3

STAGE_ORDER = (
    "prefix_hidden",
    "attn_input",
    "q_rot",
    "k_rot",
    "v_rot",
    "q_proj",
    "key_bf16",
    "value",
    "query",
    "key",
    "gate",
    "attn_out",
    "gated_attn",
    "o_proj",
    "mlp_input",
    "residual",
    "moe_out",
)


def _prompt_tokens(token_id: int, prompt_length: int) -> list[int]:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    return [int(token_id)] * int(prompt_length)


def _host_dtype(dtype: DType) -> type[np.generic]:
    if dtype is DType.FP16:
        return np.float16
    if dtype is DType.FP32:
        return np.float32
    if dtype is DType.BF16:
        return np.uint16
    raise ValueError(f"unsupported probe dtype {dtype}")


def _read_tensor(session: Qwen35ParoResidentSession, tensor: Tensor, *, row: int | None = None) -> np.ndarray:
    shape = tuple(int(dim) for dim in tensor.shape)
    dtype = _host_dtype(tensor.dtype)
    if row is None:
        out_shape = shape
        ptr = tensor.ptr
    else:
        if len(shape) < 2:
            raise ValueError(f"cannot read row from tensor shape {shape}")
        if row < 0 or row >= shape[0]:
            raise ValueError(f"row {row} outside tensor shape {shape}")
        row_elems = int(np.prod(shape[1:]))
        out_shape = shape[1:]
        ptr = tensor.ptr + row * row_elems * tensor.dtype.itemsize
    out = np.empty(out_shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(out), DeviceBuffer(ptr, out.nbytes), runtime=session.runtime)
    # BF16 tensors in this path are only cache buffers; stage tensors are FP16/FP32.
    # Returning raw uint16 as float32 is still useful if a future stage adds them.
    return out.astype(np.float32).reshape(-1)


def _hidden_row(session: Qwen35ParoResidentSession, tensor: Tensor, row: int) -> Tensor:
    return Tensor.from_handle(
        tensor.ptr + int(row) * session.hidden_nbytes,
        (1, session.config.hidden_size),
        tensor.dtype,
        tensor.device,
    )


def _run_serial_linear_prefix_for_one_token(
    session: Qwen35ParoResidentSession,
    *,
    token_id: int,
    position: int,
) -> Tensor:
    session._set_token_embedding(token_id)
    session._set_position(position)
    hidden = session.hidden
    next_hidden = session.next_hidden
    for layer_id in range(LINEAR_PREFIX_LAYERS):
        state = session.states[layer_id]
        conv_state, recurrent_state = session._slot_linear_state(layer_id, 0)
        out = state.run_linear_attention_moe_c1_layer_fp16(
            hidden,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            linear_scratch=session.linear_scratch[layer_id],
            moe_scratch=session.moe_scratch[layer_id],
            library=session.libraries,
        )
        session.runtime.memcpy_async(next_hidden.ptr, out.ptr, session.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, 0)
        hidden, next_hidden = next_hidden, hidden
    session.runtime.stream_synchronize(0)
    return hidden


def _run_native_linear_prefix(session: Qwen35ParoResidentSession, prompt_tokens: list[int]) -> Tensor:
    tokens = len(prompt_tokens)
    token_arr = np.asarray(prompt_tokens, dtype=np.int64)
    token_buf = session._dev(token_arr)
    hidden = session._prefill_hidden_view_for_rows(tokens)
    embedding_lookup_batch_fp16_i64(
        session.embedding.tensor.ptr,
        token_buf.ptr,
        hidden.ptr,
        tokens,
        session.config.hidden_size,
        session.vocab_size,
        library=session.libraries["runtime_state"],
        runtime=session.runtime,
    )
    for layer_id in range(LINEAR_PREFIX_LAYERS):
        state = session.states[layer_id]
        conv_state, recurrent_state = session._slot_linear_state(layer_id, 0)
        linear_scratch = session.linear_scratch[layer_id]
        if linear_scratch.attn_input.shape[0] < tokens:
            linear_scratch = state.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
            session.linear_scratch[layer_id] = linear_scratch
        moe_scratch = session.moe_scratch[layer_id]
        if moe_scratch.normed.shape[0] < tokens:
            moe_scratch = state.reserve_moe_c1_scratch(tokens=tokens, activation_dtype=DType.FP16)
            session.moe_scratch[layer_id] = moe_scratch
        out = state.run_linear_attention_moe_c1_layer_fp16(
            hidden,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            linear_scratch=linear_scratch,
            moe_scratch=moe_scratch,
            tokens=tokens,
            library=session.libraries,
        )
        session.runtime.memcpy_async(hidden.ptr, out.ptr, tokens * session.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, 0)
    session.runtime.stream_synchronize(0)
    return hidden


def _run_full_attention_layer3(
    session: Qwen35ParoResidentSession,
    hidden: Tensor,
    *,
    position: int,
    capture: bool,
) -> dict[str, np.ndarray]:
    session._set_position(position)
    position_tensor, append_spans, decode_spans = session._slot_spans(0)
    state = session.states[FULL_ATTENTION_LAYER]
    key_cache, value_cache = session._slot_full_cache(FULL_ATTENTION_LAYER, 0)
    scratch = session.full_scratch[FULL_ATTENTION_LAYER]
    moe_scratch = session.moe_scratch[FULL_ATTENTION_LAYER]
    if not capture:
        state.run_full_attention_moe_c1_layer_fp16(
            hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            append_spans=append_spans,
            decode_spans=decode_spans,
            cos_table=session.cos,
            sin_table=session.sin,
            position=position_tensor,
            max_positions=session.max_sequence_length,
            attention_scratch=scratch,
            moe_scratch=moe_scratch,
            chunk_size=session.chunk_size,
            num_splits=1,
            library=session.libraries,
        )
        return {}

    stages: dict[str, np.ndarray] = {"prefix_hidden": _read_tensor(session, hidden)}
    state.input_rmsnorm_fp16(hidden, scratch.attn_input, tokens=1, library=session.libraries)
    stages["attn_input"] = _read_tensor(session, scratch.attn_input)
    state.rotate_full_attention_inputs_fp16(scratch.attn_input, scratch, tokens=1, library=session.libraries)
    stages["q_rot"] = _read_tensor(session, scratch.q_rot)
    stages["k_rot"] = _read_tensor(session, scratch.k_rot)
    stages["v_rot"] = _read_tensor(session, scratch.v_rot)
    state.project_full_attention_qkv_fp16(scratch, tokens=1, library=session.libraries)
    stages["q_proj"] = _read_tensor(session, scratch.q_proj)
    stages["key_bf16"] = _read_tensor(session, scratch.key_bf16)
    stages["value"] = _read_tensor(session, scratch.value)
    _query, _key, _value, gate = state.prepare_full_attention_qkv_fp16(
        scratch,
        cos_table=session.cos,
        sin_table=session.sin,
        position=position_tensor,
        max_positions=session.max_sequence_length,
        tokens=1,
        library=session.libraries,
    )
    stages["query"] = _read_tensor(session, scratch.query)
    stages["key"] = _read_tensor(session, scratch.key)
    stages["gate"] = _read_tensor(session, gate)
    state.append_full_attention_kv_fp16(
        scratch,
        key_cache=key_cache,
        value_cache=value_cache,
        spans=append_spans,
        library=session.libraries,
    )
    gated = state.decode_full_attention_context_gate_fp16(
        scratch,
        key_cache=key_cache,
        value_cache=value_cache,
        spans=decode_spans,
        gate=gate,
        library=session.libraries,
    )
    stages["attn_out"] = _read_tensor(session, scratch.attn_out)
    stages["gated_attn"] = _read_tensor(session, gated)
    attn_out = state.project_full_attention_o_fp16(gated, scratch, tokens=1, library=session.libraries)
    stages["o_proj"] = _read_tensor(session, attn_out)
    mlp_input, residual = state.post_attention_add_rmsnorm_fp16(
        hidden,
        attn_out,
        moe_scratch,
        tokens=1,
        library=session.libraries,
    )
    stages["mlp_input"] = _read_tensor(session, mlp_input)
    stages["residual"] = _read_tensor(session, residual)
    moe_out = state.run_moe_c1_fp16(
        mlp_input,
        residual,
        scratch=moe_scratch,
        tokens=1,
        library=session.libraries,
    )
    stages["moe_out"] = _read_tensor(session, moe_out)
    session.runtime.device_synchronize()
    return stages


def _run_serial_stages(runner: Qwen35ParoNextTokenRunner, prompt_tokens: list[int]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with Qwen35ParoResidentSession(runner, max_sequence_length=len(prompt_tokens) + 2, max_layers=FULL_ATTENTION_LAYER + 1) as session:
        plan = session.native_prefill_plan().to_json_dict()
        if session.config.layer_types[FULL_ATTENTION_LAYER] != "full_attention":
            raise RuntimeError(f"layer {FULL_ATTENTION_LAYER} is not full_attention")
        stages: dict[str, np.ndarray] = {}
        for position, token_id in enumerate(prompt_tokens):
            hidden = _run_serial_linear_prefix_for_one_token(session, token_id=token_id, position=position)
            stages = _run_full_attention_layer3(
                session,
                hidden,
                position=position,
                capture=(position == len(prompt_tokens) - 1),
            )
        return stages, plan


def _run_native_prefix_then_serial_full_stages(
    runner: Qwen35ParoNextTokenRunner,
    prompt_tokens: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    tokens = len(prompt_tokens)
    with Qwen35ParoResidentSession(runner, max_sequence_length=tokens + 2, max_layers=FULL_ATTENTION_LAYER + 1) as session:
        plan = session.native_prefill_plan().to_json_dict()
        hidden_rows = _run_native_linear_prefix(session, prompt_tokens)
        stages: dict[str, np.ndarray] = {}
        for position in range(tokens):
            row_hidden = _hidden_row(session, hidden_rows, position)
            stages = _run_full_attention_layer3(
                session,
                row_hidden,
                position=position,
                capture=(position == tokens - 1),
            )
        return stages, plan


def _diff_payload(serial: np.ndarray, native: np.ndarray) -> dict[str, Any]:
    if serial.shape != native.shape:
        raise ValueError(f"shape mismatch {serial.shape} vs {native.shape}")
    diff = native - serial
    abs_diff = np.abs(diff)
    serial_norm = float(np.linalg.norm(serial))
    native_norm = float(np.linalg.norm(native))
    cosine = None
    if serial_norm > 0.0 and native_norm > 0.0:
        cosine = float(np.dot(serial, native) / (serial_norm * native_norm))
    top = np.argsort(abs_diff)[-8:][::-1]
    return {
        "elements": int(serial.size),
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "rms_abs": float(math.sqrt(float(np.mean(diff * diff)))),
        "cosine": cosine,
        "serial_norm": serial_norm,
        "native_norm": native_norm,
        "top_abs_indices": [
            {
                "index": int(index),
                "serial": float(serial[index]),
                "native": float(native[index]),
                "abs_delta": float(abs_diff[index]),
            }
            for index in top
        ],
    }


def _command(args: argparse.Namespace) -> str:
    command = (
        "python3 scripts/qwen35_native_prefill_fullattn_stage_probe.py "
        f"--model {args.model} --token-id {args.token_id} --prompt-length {args.prompt_length} "
        f"--atol {args.atol}"
    )
    if args.json is not None:
        command += f" --json {args.json}"
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=4)
    parser.add_argument("--atol", type=float, default=2.0e-2)
    parser.add_argument("--json", type=Path, help="Optional path to write JSON output")
    args = parser.parse_args(argv)

    prompt_tokens = _prompt_tokens(args.token_id, args.prompt_length)
    runner = Qwen35ParoNextTokenRunner(Path(args.model))
    serial_stages, serial_plan = _run_serial_stages(runner, prompt_tokens)
    native_stages, native_plan = _run_native_prefix_then_serial_full_stages(runner, prompt_tokens)
    stage_diffs = {stage: _diff_payload(serial_stages[stage], native_stages[stage]) for stage in STAGE_ORDER}
    first_divergent_stage = next((stage for stage in STAGE_ORDER if stage_diffs[stage]["max_abs"] > args.atol), None)
    passed = first_divergent_stage is None
    payload = {
        "schema": 1,
        "status": "accepted" if passed else "rejected_correctness",
        "blocked_reason": None if passed else "layer-3 full-attention c=1 fallback stages diverge after native linear-prefix prefill",
        "model": str(Path(args.model)),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": "qwen35_paro_native_prefix_layer3_full_attention_stage_probe",
        "command": _command(args),
        "performance_claim": False,
        "prompt_source": "repeated_token_id",
        "token_id": int(args.token_id),
        "prompt_length": len(prompt_tokens),
        "linear_prefix_layers": LINEAR_PREFIX_LAYERS,
        "full_attention_layer": FULL_ATTENTION_LAYER,
        "stage_order": list(STAGE_ORDER),
        "first_divergent_stage": first_divergent_stage,
        "atol": float(args.atol),
        "serial_native_prefill_plan": serial_plan,
        "native_prefix_plan": native_plan,
        "stage_diffs": stage_diffs,
        "passed": passed,
        "notes": [
            "Correctness diagnostic only; timings are intentionally omitted and no throughput claim is made.",
            "This probes whether a serial c=1 full-attention layer-3 bridge can consume accepted native linear-prefix rows.",
        ],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
