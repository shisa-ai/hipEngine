#!/usr/bin/env python3
"""Smoke-test DFlash tree Conv/GDN t-loop kernels against NumPy oracles.

The fixture is intentionally tiny and deterministic: candidate rows are already
in topological order and ``parent_ids[t]`` either points to the root state
(``-1``) or to a prior candidate row.  It validates both BF16 and FP16 wrapper
variants for tree sizes N={2,4,8} without importing torch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.linear_attn import (
    build_qwen35_linear_attn_conv,
    build_qwen35_linear_attn_gdn,
    qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_bf16,
    qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_fp16,
    qwen35_linear_attn_tree_conv_decode_bf16_tloop,
    qwen35_linear_attn_tree_conv_decode_fp16_tloop,
)

PARENTS_BY_N: dict[int, tuple[int, ...]] = {
    2: (-1, 0),
    4: (-1, 0, 1, 0),
    8: (-1, 0, 1, 1, 3, 2, 5, 0),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=None,
        help="Read precomputed hipcc --version text before loading cached HIP libraries.",
    )
    parser.add_argument(
        "--require-cached-build",
        action="store_true",
        help="Fail instead of invoking hipcc if the expected HIP cache artifact is absent.",
    )
    args = parser.parse_args()
    compiler_version = _read_text(args.compiler_version_file) if args.compiler_version_file else None

    runtime = get_hip_runtime()
    conv_lib = build_qwen35_linear_attn_conv(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    gdn_lib = build_qwen35_linear_attn_gdn(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )

    max_error = 0.0
    for max_nodes, parents_tuple in PARENTS_BY_N.items():
        parents = np.asarray(parents_tuple, dtype=np.int64)
        for variant in ("bf16", "fp16"):
            conv_error = _run_conv_tree_case(runtime, conv_lib, variant, max_nodes, parents)
            gdn_error = _run_gdn_tree_case(runtime, gdn_lib, variant, max_nodes, parents)
            max_error = max(max_error, conv_error, gdn_error)
            print(
                f"variant={variant} max_nodes={max_nodes} "
                f"conv_max_abs={conv_error:.3g} gdn_max_abs={gdn_error:.3g}"
            )
    print(f"tree_tloop_smoke max_abs={max_error:.3g}")
    return 0 if max_error <= 2.0e-6 else 1


def _run_conv_tree_case(runtime, library, variant: str, max_nodes: int, parent_ids: np.ndarray) -> float:
    channels = 13
    kernel_size = 4
    hidden_f32 = _pattern((max_nodes, channels), scale=1.0 / 17.0, offset=-0.5)
    hidden_device, hidden_ref = _lowp_pair(hidden_f32, variant)
    base_state = _pattern((channels, kernel_size), scale=1.0 / 13.0, offset=-0.5, mul=5, add=2)
    conv_weight = _pattern((channels, kernel_size), scale=1.0 / 11.0, offset=-0.5, mul=3, add=4)
    expected_out, expected_tree = _conv_tree_ref(hidden_ref, base_state, conv_weight, parent_ids)
    out = np.empty_like(expected_out)
    tree = np.empty_like(expected_tree)
    buffers = []

    def dev(array: np.ndarray):
        contiguous = np.ascontiguousarray(array)
        buffer = malloc(contiguous.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(contiguous), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        contiguous = np.ascontiguousarray(array)
        buffer = malloc(contiguous.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        hidden_dev = dev(hidden_device)
        base_dev = dev(base_state)
        tree_dev = out_dev(tree)
        weight_dev = dev(conv_weight)
        parent_dev = dev(parent_ids)
        out_dev_buf = out_dev(out)
        wrapper = (
            qwen35_linear_attn_tree_conv_decode_bf16_tloop
            if variant == "bf16"
            else qwen35_linear_attn_tree_conv_decode_fp16_tloop
        )
        wrapper(
            hidden_dev.ptr,
            base_dev.ptr,
            tree_dev.ptr,
            weight_dev.ptr,
            parent_dev.ptr,
            out_dev_buf.ptr,
            max_nodes,
            channels,
            kernel_size,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev_buf, runtime=runtime)
        copy_device_to_host(host_array_ptr(tree), tree_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    return max(float(np.max(np.abs(out - expected_out))), float(np.max(np.abs(tree - expected_tree))))


def _run_gdn_tree_case(runtime, library, variant: str, max_nodes: int, parent_ids: np.ndarray) -> float:
    num_k_heads = 1
    num_v_heads = 2
    head_k_dim = 128
    head_v_dim = 5  # Non-power-of-two fixture catches finalize reduction masking.
    eps = 1.0e-6
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    width = 2 * key_dim + value_dim
    conv_out = _pattern((max_nodes, width), scale=1.0 / 29.0, offset=-0.5, mul=19, add=5)
    gate_f32 = _pattern((max_nodes, value_dim), scale=1.0 / 17.0, offset=-0.5, mul=7, add=3)
    a_f32 = _pattern((max_nodes, num_v_heads), scale=1.0 / 11.0, offset=-0.5, mul=3, add=5)
    b_f32 = _pattern((max_nodes, num_v_heads), scale=1.0 / 13.0, offset=-0.5, mul=5, add=7)
    gate_device, gate_ref = _lowp_pair(gate_f32, variant)
    a_device, a_ref = _lowp_pair(a_f32, variant)
    b_device, b_ref = _lowp_pair(b_f32, variant)
    dt_bias = np.asarray([0.05, -0.02], dtype=np.float32)
    a_log = np.asarray([-1.0, -0.5], dtype=np.float32)
    norm_weight = np.asarray([1.0, 0.5, 1.5, 0.75, -0.25], dtype=np.float32)
    base_state = _pattern((num_v_heads, head_k_dim, head_v_dim), scale=1.0 / 23.0, offset=-0.5, mul=7, add=3)
    expected_out, expected_tree, expected_acc = _gdn_tree_ref(
        conv_out,
        gate_ref,
        a_ref,
        b_ref,
        dt_bias,
        a_log,
        norm_weight,
        base_state,
        parent_ids,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        eps,
    )
    out = np.empty_like(expected_out)
    tree = np.empty_like(expected_tree)
    acc = np.empty_like(expected_acc)
    buffers = []

    def dev(array: np.ndarray):
        contiguous = np.ascontiguousarray(array)
        buffer = malloc(contiguous.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(contiguous), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        contiguous = np.ascontiguousarray(array)
        buffer = malloc(contiguous.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        conv_dev = dev(conv_out)
        gate_dev = dev(gate_device)
        a_dev = dev(a_device)
        b_dev = dev(b_device)
        dt_dev = dev(dt_bias)
        a_log_dev = dev(a_log)
        norm_dev = dev(norm_weight)
        base_dev = dev(base_state)
        tree_dev = out_dev(tree)
        parent_dev = dev(parent_ids)
        acc_dev = out_dev(acc)
        out_dev_buf = out_dev(out)
        wrapper = (
            qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_bf16
            if variant == "bf16"
            else qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_fp16
        )
        wrapper(
            conv_dev.ptr,
            gate_dev.ptr,
            a_dev.ptr,
            b_dev.ptr,
            dt_dev.ptr,
            a_log_dev.ptr,
            norm_dev.ptr,
            base_dev.ptr,
            tree_dev.ptr,
            parent_dev.ptr,
            acc_dev.ptr,
            out_dev_buf.ptr,
            eps,
            max_nodes,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev_buf, runtime=runtime)
        copy_device_to_host(host_array_ptr(tree), tree_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(acc), acc_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    return max(
        float(np.max(np.abs(out - expected_out))),
        float(np.max(np.abs(tree - expected_tree))),
        float(np.max(np.abs(acc - expected_acc))),
    )


def _conv_tree_ref(
    hidden: np.ndarray,
    base_state: np.ndarray,
    conv_weight: np.ndarray,
    parent_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    max_nodes, channels = hidden.shape
    kernel_size = base_state.shape[1]
    tree = np.zeros((max_nodes, channels, kernel_size), dtype=np.float32)
    out = np.zeros((max_nodes, channels), dtype=np.float32)
    for t in range(max_nodes):
        parent = int(parent_ids[t])
        parent_state = base_state if parent < 0 else tree[parent]
        for channel in range(channels):
            node = np.empty((kernel_size,), dtype=np.float32)
            node[:-1] = parent_state[channel, 1:]
            node[-1] = np.float32(hidden[t, channel])
            tree[t, channel] = node
            out[t, channel] = _silu(np.sum(node * conv_weight[channel], dtype=np.float32))
    return out, tree


def _gdn_tree_ref(
    conv_out: np.ndarray,
    gate: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    dt_bias: np.ndarray,
    a_log: np.ndarray,
    norm_weight: np.ndarray,
    base_state: np.ndarray,
    parent_ids: np.ndarray,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_nodes = conv_out.shape[0]
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    repeat = num_v_heads // num_k_heads
    tree = np.zeros((max_nodes, num_v_heads, head_k_dim, head_v_dim), dtype=np.float32)
    acc = np.zeros((max_nodes, num_v_heads, head_v_dim), dtype=np.float32)
    out = np.zeros((max_nodes, value_dim), dtype=np.float32)
    for t in range(max_nodes):
        parent = int(parent_ids[t])
        for v_head in range(num_v_heads):
            k_head = v_head // repeat
            q = conv_out[t, k_head * head_k_dim : (k_head + 1) * head_k_dim].astype(np.float32)
            k = conv_out[t, key_dim + k_head * head_k_dim : key_dim + (k_head + 1) * head_k_dim].astype(np.float32)
            q_scale = np.float32(1.0) / np.sqrt(np.sum(q * q, dtype=np.float32) + np.float32(1.0e-6))
            q_scale *= np.float32(1.0) / np.sqrt(np.float32(head_k_dim))
            k_scale = np.float32(1.0) / np.sqrt(np.sum(k * k, dtype=np.float32) + np.float32(1.0e-6))
            beta = _sigmoid(np.float32(b[t, v_head]))
            decay = np.exp(-np.exp(a_log[v_head]) * _softplus(np.float32(a[t, v_head]) + dt_bias[v_head]))
            parent_state = base_state[v_head] if parent < 0 else tree[parent, v_head]
            for dv in range(head_v_dim):
                value = conv_out[t, 2 * key_dim + v_head * head_v_dim + dv]
                k_norm = k * k_scale
                decayed = parent_state[:, dv] * decay
                kv_mem = np.sum(k_norm * decayed, dtype=np.float32)
                delta = (value - kv_mem) * beta
                new_state = decayed + k_norm * delta
                tree[t, v_head, :, dv] = new_state
                acc[t, v_head, dv] = np.sum((q * q_scale) * new_state, dtype=np.float32)
            row = acc[t, v_head]
            inv_rms = np.float32(1.0) / np.sqrt(np.mean(row * row, dtype=np.float32) + np.float32(eps))
            for dv in range(head_v_dim):
                pos = v_head * head_v_dim + dv
                out[t, pos] = row[dv] * inv_rms * norm_weight[dv] * _silu(np.float32(gate[t, pos]))
    return out, tree, acc


def _pattern(
    shape: tuple[int, ...],
    *,
    scale: float,
    offset: float,
    mul: int = 7,
    add: int = 3,
) -> np.ndarray:
    indices = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    values = ((indices * mul + add) % 31).astype(np.float32) * np.float32(scale) + np.float32(offset)
    return values.astype(np.float32)


def _lowp_pair(values: np.ndarray, variant: str) -> tuple[np.ndarray, np.ndarray]:
    if variant == "bf16":
        bits = _float32_to_bf16_bits(values)
        return bits, _bf16_bits_to_float32(bits)
    if variant == "fp16":
        fp16 = values.astype(np.float16)
        return fp16, fp16.astype(np.float32)
    raise ValueError(f"unsupported variant: {variant}")


def _float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    arr = np.asarray(bits, dtype=np.uint16)
    widened = arr.astype(np.uint32) << np.uint32(16)
    return widened.view(np.float32)


def _sigmoid(value: np.float32) -> np.float32:
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-value, dtype=np.float32))


def _softplus(value: np.float32) -> np.float32:
    return value if value > np.float32(20.0) else np.log1p(np.exp(value, dtype=np.float32), dtype=np.float32)


def _silu(value: np.float32 | np.ndarray) -> np.float32 | np.ndarray:
    return value / (np.float32(1.0) + np.exp(-value, dtype=np.float32))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
