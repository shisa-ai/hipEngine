#!/usr/bin/env python3
"""Probe where Qwen3.5/PARO native linear-prefix prefill diverges.

This diagnostic compares the first layer's linear-attention out-projection for
serial c=1 prompt prefill against the current native linear-prefix prefill helper.
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
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.runtime import embedding_lookup_batch_fp16_i64
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)


def _prompt_tokens(token_id: int, prompt_length: int) -> list[int]:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    return [int(token_id)] * int(prompt_length)


def _read_fp16(session: Qwen35ParoResidentSession, ptr: int, shape: tuple[int, ...]) -> np.ndarray:
    out = np.empty(shape, dtype=np.float16)
    copy_device_to_host(host_array_ptr(out), DeviceBuffer(ptr, out.nbytes), runtime=session.runtime)
    return out.astype(np.float32)


def _run_serial_layer0_attention(
    runner: Qwen35ParoNextTokenRunner,
    prompt_tokens: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=len(prompt_tokens) + 2,
        max_layers=1,
    ) as session:
        plan = session.native_prefill_plan().to_json_dict()
        state = session.states[0]
        scratch = session.linear_scratch[0]
        conv_state, recurrent_state = session._slot_linear_state(0, 0)
        final = None
        for position, token_id in enumerate(prompt_tokens):
            session._set_token_embedding(token_id)
            session._set_position(position)
            state.input_rmsnorm_fp16(
                session.hidden,
                scratch.attn_input,
                tokens=1,
                library=session.libraries,
            )
            attn_out = state.run_linear_attention_out_proj_fp16(
                scratch.attn_input,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                scratch=scratch,
                tokens=1,
                library=session.libraries,
            )
            session.runtime.device_synchronize()
            if position == len(prompt_tokens) - 1:
                final = _read_fp16(session, attn_out.ptr, attn_out.shape)
        if final is None:
            raise RuntimeError("serial attention probe produced no final row")
    return final.reshape(-1), plan


def _run_native_layer0_attention(
    runner: Qwen35ParoNextTokenRunner,
    prompt_tokens: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    tokens = len(prompt_tokens)
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=tokens + 2,
        max_layers=1,
    ) as session:
        plan = session.native_prefill_plan().to_json_dict()
        token_arr = np.asarray(prompt_tokens, dtype=np.int64)
        token_buf = session._dev(token_arr)
        embedding_lookup_batch_fp16_i64(
            session.embedding.tensor.ptr,
            token_buf.ptr,
            session.prefill_hidden.ptr,
            tokens,
            session.config.hidden_size,
            session.vocab_size,
            library=session.libraries["runtime_state"],
            runtime=session.runtime,
        )
        hidden = Tensor.from_handle(
            session.prefill_hidden.ptr,
            (tokens, session.config.hidden_size),
            DType.FP16,
            session.device,
        )
        state = session.states[0]
        scratch = state.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        conv_state, recurrent_state = session._slot_linear_state(0, 0)
        state.input_rmsnorm_fp16(hidden, scratch.attn_input, tokens=tokens, library=session.libraries)
        attn_out = state.run_linear_attention_prefill_out_proj_fp16(
            scratch.attn_input,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            scratch=scratch,
            tokens=tokens,
            library=session.libraries,
        )
        session.runtime.device_synchronize()
        last_ptr = attn_out.ptr + (tokens - 1) * session.hidden_nbytes
        final = _read_fp16(session, last_ptr, (1, session.config.hidden_size))
    return final.reshape(-1), plan


def _diff_payload(serial: np.ndarray, native: np.ndarray) -> dict[str, Any]:
    diff = native - serial
    abs_diff = np.abs(diff)
    serial_norm = float(np.linalg.norm(serial))
    native_norm = float(np.linalg.norm(native))
    cosine = None
    if serial_norm > 0.0 and native_norm > 0.0:
        cosine = float(np.dot(serial, native) / (serial_norm * native_norm))
    top = np.argsort(abs_diff)[-8:][::-1]
    return {
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
    serial, serial_plan = _run_serial_layer0_attention(runner, prompt_tokens)
    native, native_plan = _run_native_layer0_attention(runner, prompt_tokens)
    diff = _diff_payload(serial, native)
    passed = bool(diff["max_abs"] <= args.atol)
    payload = {
        "schema": 1,
        "status": "accepted" if passed else "rejected_correctness",
        "blocked_reason": None if passed else "native prefill diverges from serial c=1 inside layer 0 linear-attention out projection",
        "model": str(Path(args.model)),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": "qwen35_paro_native_prefill_layer0_attention_probe",
        "command": _command(args),
        "performance_claim": False,
        "prompt_source": "repeated_token_id",
        "token_id": int(args.token_id),
        "prompt_length": len(prompt_tokens),
        "stage": "layer0_linear_attention_out_proj_last_token",
        "atol": float(args.atol),
        "serial_native_prefill_plan": serial_plan,
        "native_prefill_plan": native_plan,
        "diff": diff,
        "passed": passed,
        "notes": [
            "Correctness diagnostic only; timings are intentionally omitted and no throughput claim is made.",
            "This isolates the native-prefix mismatch before MoE and before later layer-prefix interactions.",
        ],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
