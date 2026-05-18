#!/usr/bin/env python3
"""Probe where Qwen3.5/PARO native linear-prefix prefill diverges.

This diagnostic compares the first layer's linear-attention stages for serial
c=1 prompt prefill against the current native linear-prefix prefill helper.  It
is correctness/blocker evidence only, not a benchmark.
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
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.runtime import embedding_lookup_batch_fp16_i64
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)

STAGE_ORDER = (
    "input_norm",
    "qkv_rot",
    "z_rot",
    "qkv",
    "z",
    "ab",
    "conv_out",
    "gated_recurrent",
    "attention_out",
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
    raise ValueError(f"unsupported probe dtype {dtype}")


def _read_tensor(
    session: Qwen35ParoResidentSession,
    tensor: Tensor,
    *,
    row: int | None = None,
) -> np.ndarray:
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
    return out.astype(np.float32).reshape(-1)


def _run_serial_layer0_stages(
    runner: Qwen35ParoNextTokenRunner,
    prompt_tokens: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=len(prompt_tokens) + 2,
        max_layers=1,
    ) as session:
        plan = session.native_prefill_plan().to_json_dict()
        state = session.states[0]
        scratch = session.linear_scratch[0]
        conv_state, recurrent_state = session._slot_linear_state(0, 0)
        stages: dict[str, np.ndarray] = {}
        for position, token_id in enumerate(prompt_tokens):
            capture = position == len(prompt_tokens) - 1
            session._set_token_embedding(token_id)
            session._set_position(position)
            state.input_rmsnorm_fp16(
                session.hidden,
                scratch.attn_input,
                tokens=1,
                library=session.libraries,
            )
            if capture:
                stages["input_norm"] = _read_tensor(session, scratch.attn_input)
            state.rotate_linear_attention_inputs_fp16(
                scratch.attn_input,
                scratch,
                tokens=1,
                library=session.libraries,
            )
            if capture:
                stages["qkv_rot"] = _read_tensor(session, scratch.qkv_rot)
                stages["z_rot"] = _read_tensor(session, scratch.z_rot)
            state.project_linear_attention_qkv_z_fp16(scratch, tokens=1, library=session.libraries)
            if capture:
                stages["qkv"] = _read_tensor(session, scratch.qkv)
                stages["z"] = _read_tensor(session, scratch.z)
            state.project_linear_attention_ab_fp16(scratch.attn_input, scratch, tokens=1, library=session.libraries)
            if capture:
                stages["ab"] = np.concatenate((_read_tensor(session, scratch.a), _read_tensor(session, scratch.b)))
            state.run_linear_attention_conv_gdn_fp16(
                scratch,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                library=session.libraries,
            )
            if capture:
                stages["conv_out"] = _read_tensor(session, scratch.conv_out)
            attn_out = state.project_linear_attention_out_fp16(scratch, tokens=1, library=session.libraries)
            session.runtime.device_synchronize()
            if capture:
                stages["gated_recurrent"] = _read_tensor(session, scratch.recurrent_bf16)
                stages["attention_out"] = _read_tensor(session, attn_out)
        if not stages:
            raise RuntimeError("serial stage probe produced no final row")
    return stages, plan


def _run_native_layer0_stages(
    runner: Qwen35ParoNextTokenRunner,
    prompt_tokens: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    tokens = len(prompt_tokens)
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=tokens + 2,
        max_layers=1,
    ) as session:
        plan = session.native_prefill_plan().to_json_dict()
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
        state = session.states[0]
        scratch = state.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        conv_state, recurrent_state = session._slot_linear_state(0, 0)
        last = tokens - 1
        stages: dict[str, np.ndarray] = {}
        state.input_rmsnorm_fp16(hidden, scratch.attn_input, tokens=tokens, library=session.libraries)
        stages["input_norm"] = _read_tensor(session, scratch.attn_input, row=last)
        state.rotate_linear_attention_inputs_fp16(scratch.attn_input, scratch, tokens=tokens, library=session.libraries)
        stages["qkv_rot"] = _read_tensor(session, scratch.qkv_rot, row=last)
        stages["z_rot"] = _read_tensor(session, scratch.z_rot, row=last)
        state.project_linear_attention_qkv_z_fp16(scratch, tokens=tokens, library=session.libraries)
        stages["qkv"] = _read_tensor(session, scratch.qkv, row=last)
        stages["z"] = _read_tensor(session, scratch.z, row=last)
        state.project_linear_attention_ab_fp16(scratch.attn_input, scratch, tokens=tokens, library=session.libraries)
        stages["ab"] = np.concatenate((_read_tensor(session, scratch.a, row=last), _read_tensor(session, scratch.b, row=last)))
        state.run_linear_attention_prefill_conv_gdn_fp16(
            scratch,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            tokens=tokens,
            library=session.libraries,
        )
        stages["conv_out"] = _read_tensor(session, scratch.conv_out, row=last)
        stages["gated_recurrent"] = _read_tensor(session, scratch.recurrent_bf16, row=last)
        attn_out = state.project_linear_attention_prefill_out_fp16(scratch, tokens=tokens, library=session.libraries)
        session.runtime.device_synchronize()
        stages["attention_out"] = _read_tensor(session, attn_out, row=last)
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
        "python3 scripts/qwen35_native_prefill_stage_probe.py "
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
    parser.add_argument("--atol", type=float, default=1.0e-3)
    parser.add_argument("--json", type=Path, help="Optional path to write JSON output")
    args = parser.parse_args(argv)

    prompt_tokens = _prompt_tokens(args.token_id, args.prompt_length)
    runner = Qwen35ParoNextTokenRunner(Path(args.model))
    serial_stages, serial_plan = _run_serial_layer0_stages(runner, prompt_tokens)
    native_stages, native_plan = _run_native_layer0_stages(runner, prompt_tokens)
    stage_diffs = {
        stage: _diff_payload(serial_stages[stage], native_stages[stage])
        for stage in STAGE_ORDER
    }
    first_divergent_stage = next(
        (stage for stage in STAGE_ORDER if stage_diffs[stage]["max_abs"] > args.atol),
        None,
    )
    passed = first_divergent_stage is None
    payload = {
        "schema": 1,
        "status": "accepted" if passed else "rejected_correctness",
        "blocked_reason": None if passed else "native prefill diverges from serial c=1 inside layer 0 linear-attention stages after conv layout parity",
        "model": str(Path(args.model)),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": "qwen35_paro_native_prefill_layer0_stage_bisect",
        "command": _command(args),
        "performance_claim": False,
        "prompt_source": "repeated_token_id",
        "token_id": int(args.token_id),
        "prompt_length": len(prompt_tokens),
        "stage_order": list(STAGE_ORDER),
        "first_divergent_stage": first_divergent_stage,
        "atol": float(args.atol),
        "serial_native_prefill_plan": serial_plan,
        "native_prefill_plan": native_plan,
        "stage_diffs": stage_diffs,
        "passed": passed,
        "notes": [
            "Correctness diagnostic only; timings are intentionally omitted and no throughput claim is made.",
            "This bisects the native-prefix mismatch within layer0 linear attention before MoE, comparing the lowp gated recurrent tensor that feeds out_proj.",
        ],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
