#!/usr/bin/env python3
"""CPU-replay GGUF Qwen3.5 GDN recurrence from captured boundary arrays."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader  # noqa: E402
from hipengine.loading.materialize import float_array_to_bf16_bits  # noqa: E402
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map  # noqa: E402
from hipengine.loading.qwen35_gguf_materialize import _gguf_ssm_a_to_kernel_a_log  # noqa: E402
from hipengine.quant.gguf import bf16_to_float32  # noqa: E402

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_CAPTURE = Path("benchmarks/results/mtp-gguf-iter271-gdn-replay-window-capture.json")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter272-gdn-replay-cpu-compare.json")


@dataclass(frozen=True)
class GDNShape:
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int

    @property
    def repeat(self) -> int:
        return self.num_v_heads // self.num_k_heads

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def qkv_width(self) -> int:
        return 2 * self.key_dim + self.num_v_heads * self.head_v_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=272)
    args = parser.parse_args()

    artifact = build_gdn_replay_artifact(
        model=args.model,
        capture_path=args.capture,
        layer_id=args.layer,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "recurrent_out_max_abs": artifact["recurrent_out_vs_cpu"]["max_abs_diff"],
                "recurrent_out_rms": artifact["recurrent_out_vs_cpu"]["rms_abs_diff"],
                "recurrent_bf16_max_abs": artifact["recurrent_bf16_vs_cpu_bf16"]["max_abs_diff"],
                "within_tolerance": artifact["within_tolerance"],
            },
            indent=2,
        )
    )


def build_gdn_replay_artifact(
    *,
    model: Path,
    capture_path: Path,
    layer_id: int = 0,
    iteration: int = 272,
) -> dict[str, Any]:
    capture = json.loads(capture_path.read_text())
    captures = capture.get("captures")
    if not isinstance(captures, list):
        raise ValueError("capture artifact must be a multi-position batch with captures")

    reader = GGUFReader(model)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    cfg = model_map.config
    layer = model_map.layers[int(layer_id)]
    shape = GDNShape(
        num_k_heads=int(cfg.ssm_group_count),
        num_v_heads=int(cfg.ssm_time_step_rank),
        head_k_dim=int(cfg.ssm_state_size),
        head_v_dim=int(cfg.ssm_inner_size) // int(cfg.ssm_time_step_rank),
    )
    norm_weight = reader.dequantize_tensor(layer.tensor("ssm_norm").name).astype(np.float32)
    dt_bias = reader.dequantize_tensor(layer.tensor("ssm_dt_bias").name).astype(np.float32)
    a_raw = reader.dequantize_tensor(layer.tensor("ssm_a").name).astype(np.float32)
    a_log = _gguf_ssm_a_to_kernel_a_log(a_raw).astype(np.float32)

    conv_out, gate, alpha, beta, recurrent_out, recurrent_bf16 = _read_capture_arrays(captures)
    comparison = compare_gdn_replay(
        conv_out=conv_out,
        gate=gate,
        alpha=alpha,
        beta=beta,
        norm_weight=norm_weight,
        dt_bias=dt_bias,
        a_log=a_log,
        device_recurrent_out=recurrent_out,
        device_recurrent_bf16=recurrent_bf16,
        shape=shape,
        eps=float(cfg.rms_norm_eps),
    )
    within = (
        comparison["recurrent_out_vs_cpu"]["max_abs_diff"] <= 1.0e-5
        and comparison["recurrent_bf16_vs_cpu_bf16"]["max_abs_diff"] <= 5.0e-5
    )
    return {
        "schema": 1,
        "kind": "mtp_gguf_gdn_replay_cpu_compare",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "model": str(model),
        "source_capture": str(capture_path),
        "source_iteration": capture.get("iteration"),
        "layer_id": int(layer_id),
        "positions": capture.get("positions"),
        "token_ids": capture.get("token_ids"),
        "shape": {
            "num_k_heads": shape.num_k_heads,
            "num_v_heads": shape.num_v_heads,
            "head_k_dim": shape.head_k_dim,
            "head_v_dim": shape.head_v_dim,
            "qkv_width": shape.qkv_width,
            "value_dim": shape.value_dim,
        },
        "recurrent_out_vs_cpu": comparison["recurrent_out_vs_cpu"],
        "recurrent_bf16_vs_cpu_bf16": comparison["recurrent_bf16_vs_cpu_bf16"],
        "cpu_f32_vs_cpu_bf16": comparison["cpu_f32_vs_cpu_bf16"],
        "samples": comparison["samples"],
        "within_tolerance": bool(within),
        "conclusion": _conclusion(comparison, within),
    }


def compare_gdn_replay(
    *,
    conv_out: np.ndarray,
    gate: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    norm_weight: np.ndarray,
    dt_bias: np.ndarray,
    a_log: np.ndarray,
    device_recurrent_out: np.ndarray,
    device_recurrent_bf16: np.ndarray,
    shape: GDNShape,
    eps: float,
) -> dict[str, Any]:
    cpu_recurrent = replay_gdn(
        conv_out=conv_out,
        gate=gate,
        alpha=alpha,
        beta=beta,
        norm_weight=norm_weight,
        dt_bias=dt_bias,
        a_log=a_log,
        shape=shape,
        eps=eps,
    )
    device_recurrent_out = np.asarray(device_recurrent_out, dtype=np.float32)
    device_recurrent_bf16 = np.asarray(device_recurrent_bf16, dtype=np.float32)
    if device_recurrent_out.shape != cpu_recurrent.shape:
        raise ValueError("device_recurrent_out shape must match replayed GDN output")
    if device_recurrent_bf16.shape != cpu_recurrent.shape:
        raise ValueError("device_recurrent_bf16 shape must match replayed GDN output")
    cpu_bf16 = bf16_to_float32(float_array_to_bf16_bits(cpu_recurrent)).astype(np.float32)
    return {
        "recurrent_out_vs_cpu": _diff_metrics(cpu_recurrent, device_recurrent_out),
        "recurrent_bf16_vs_cpu_bf16": _diff_metrics(cpu_bf16, device_recurrent_bf16),
        "cpu_f32_vs_cpu_bf16": _diff_metrics(cpu_recurrent, cpu_bf16),
        "samples": {
            "cpu_recurrent_out_final": [float(x) for x in cpu_recurrent[-1, :8]],
            "device_recurrent_out_final": [float(x) for x in device_recurrent_out[-1, :8]],
            "cpu_recurrent_bf16_final": [float(x) for x in cpu_bf16[-1, :8]],
            "device_recurrent_bf16_final": [float(x) for x in device_recurrent_bf16[-1, :8]],
        },
    }


def replay_gdn(
    *,
    conv_out: np.ndarray,
    gate: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    norm_weight: np.ndarray,
    dt_bias: np.ndarray,
    a_log: np.ndarray,
    shape: GDNShape,
    eps: float,
) -> np.ndarray:
    conv_out = np.asarray(conv_out, dtype=np.float32)
    gate = np.asarray(gate, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    norm_weight = np.asarray(norm_weight, dtype=np.float32).reshape(-1)
    dt_bias = np.asarray(dt_bias, dtype=np.float32).reshape(-1)
    a_log = np.asarray(a_log, dtype=np.float32).reshape(-1)
    tokens = conv_out.shape[0]
    _validate_shapes(conv_out, gate, alpha, beta, norm_weight, dt_bias, a_log, shape)

    state = np.zeros((shape.num_v_heads, shape.head_k_dim, shape.head_v_dim), dtype=np.float32)
    outputs = np.empty((tokens, shape.value_dim), dtype=np.float32)
    sqrt_head_k = np.sqrt(np.float32(shape.head_k_dim))
    eps32 = np.float32(eps)
    for token_idx in range(tokens):
        conv = conv_out[token_idx]
        gate_token = gate[token_idx]
        out_token = np.empty((shape.num_v_heads, shape.head_v_dim), dtype=np.float32)
        for v_head in range(shape.num_v_heads):
            k_head = v_head // shape.repeat
            q_base = k_head * shape.head_k_dim
            k_base = shape.key_dim + k_head * shape.head_k_dim
            v_base = 2 * shape.key_dim + v_head * shape.head_v_dim
            q = conv[q_base : q_base + shape.head_k_dim]
            k = conv[k_base : k_base + shape.head_k_dim]
            value = conv[v_base : v_base + shape.head_v_dim]
            q_scale = (np.float32(1.0) / max(float(np.sqrt(np.sum(q * q))), 1.0e-6)) / sqrt_head_k
            k_scale = np.float32(1.0) / max(float(np.sqrt(np.sum(k * k))), 1.0e-6)
            q_norm = q * np.float32(q_scale)
            k_norm = k * np.float32(k_scale)
            beta_v = _sigmoid(beta[token_idx, v_head])
            decay = np.exp(
                -np.exp(a_log[v_head]) * _softplus(alpha[token_idx, v_head] + dt_bias[v_head])
            ).astype(np.float32)
            state_v = state[v_head]
            kv_mem = k_norm @ (state_v * decay)
            delta = (value - kv_mem) * beta_v
            new_state = state_v * decay + np.outer(k_norm, delta).astype(np.float32)
            state[v_head] = new_state.astype(np.float32)
            out_acc = (q_norm @ state[v_head]).astype(np.float32)
            inv_rms = np.float32(1.0) / np.sqrt(np.mean(out_acc * out_acc) + eps32)
            gate_v = gate_token[v_head * shape.head_v_dim : (v_head + 1) * shape.head_v_dim]
            out_token[v_head] = out_acc * inv_rms * norm_weight * _silu(gate_v)
        outputs[token_idx] = out_token.reshape(-1)
    return outputs


def _read_capture_arrays(captures: list[object]) -> tuple[np.ndarray, ...]:
    return (
        _stack_capture_array(captures, "conv_out_f32"),
        _stack_capture_array(captures, "linear_z_f32"),
        _stack_capture_array(captures, "ssm_alpha_f32"),
        _stack_capture_array(captures, "ssm_beta_f32"),
        _stack_capture_array(captures, "recurrent_out_f32"),
        _stack_capture_array(captures, "recurrent_bf16_f32"),
    )


def _stack_capture_array(captures: list[object], key: str) -> np.ndarray:
    arrays = []
    for capture in captures:
        if not isinstance(capture, dict) or not isinstance(capture.get("arrays"), dict):
            raise ValueError("capture records must include arrays; rerun with --include-arrays")
        if key not in capture["arrays"]:
            raise ValueError(f"capture artifact missing arrays.{key}")
        arrays.append(np.asarray(capture["arrays"][key], dtype=np.float32))
    return np.stack(arrays, axis=0)


def _validate_shapes(
    conv_out: np.ndarray,
    gate: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    norm_weight: np.ndarray,
    dt_bias: np.ndarray,
    a_log: np.ndarray,
    shape: GDNShape,
) -> None:
    tokens = conv_out.shape[0]
    expected = {
        "conv_out": (tokens, shape.qkv_width),
        "gate": (tokens, shape.value_dim),
        "alpha": (tokens, shape.num_v_heads),
        "beta": (tokens, shape.num_v_heads),
        "norm_weight": (shape.head_v_dim,),
        "dt_bias": (shape.num_v_heads,),
        "a_log": (shape.num_v_heads,),
    }
    actual = {
        "conv_out": conv_out.shape,
        "gate": gate.shape,
        "alpha": alpha.shape,
        "beta": beta.shape,
        "norm_weight": norm_weight.shape,
        "dt_bias": dt_bias.shape,
        "a_log": a_log.shape,
    }
    for name, expected_shape in expected.items():
        if actual[name] != expected_shape:
            raise ValueError(f"{name} shape {actual[name]} != expected {expected_shape}")


def _sigmoid(values: np.ndarray | np.float32 | float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-values))


def _softplus(values: np.ndarray | np.float32 | float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.where(values > np.float32(20.0), values, np.log1p(np.exp(values)))


def _silu(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / (np.float32(1.0) + np.exp(-values))


def _diff_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape")
    diff = candidate - reference
    return {
        "count": int(reference.size),
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "rms_abs_diff": float(np.sqrt(np.mean(diff * diff, dtype=np.float32)))
        if diff.size
        else 0.0,
        "mean_abs_diff": float(np.mean(np.abs(diff), dtype=np.float32)) if diff.size else 0.0,
        "reference_rms": float(np.sqrt(np.mean(reference * reference, dtype=np.float32)))
        if reference.size
        else 0.0,
        "candidate_rms": float(np.sqrt(np.mean(candidate * candidate, dtype=np.float32)))
        if candidate.size
        else 0.0,
    }


def _conclusion(comparison: dict[str, Any], within: bool) -> str:
    recurrent = comparison["recurrent_out_vs_cpu"]
    recurrent_bf16 = comparison["recurrent_bf16_vs_cpu_bf16"]
    if within:
        return (
            "CPU GDN recurrent replay matches device recurrent_out/recurrent_bf16 over the "
            f"captured prompt; recurrent max_abs={recurrent['max_abs_diff']:.6g}, "
            f"bf16 max_abs={recurrent_bf16['max_abs_diff']:.6g}. Layer-0 attention branch "
            "is now explained through attn_out, so mismatch search should move outside this branch."
        )
    return (
        "CPU GDN recurrent replay diverges from device output; inspect recurrence state/order. "
        f"recurrent max_abs={recurrent['max_abs_diff']:.6g}, "
        f"bf16 max_abs={recurrent_bf16['max_abs_diff']:.6g}."
    )


if __name__ == "__main__":
    main()
