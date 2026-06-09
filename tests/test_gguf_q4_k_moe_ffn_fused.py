"""B1: fused selected-expert GGUF Q4_K MoE FFN megakernel correctness.

Gates the fused ``gate_up -> silu*mul -> down`` megakernel (one block per
selected (token, expert) row, intermediate on-chip) against the B0 CPU oracle:

  * f32 path: per-selected-row down output matches a CPU per-row recompute, and
    the routing-weighted combine matches B0 ``gguf_moe_selected_ffn`` tightly;
  * bf16 path: the combine clears the project KL<=0.05 / top-1>=90% gate vs B0;
  * row-invariance: rows=1 alone is bit-identical to the same row inside a batch
    (the T1 self-consistency requirement);

The synthetic Q4_K weights/inputs are reused from the B0 fixture builder.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.cpu_reference import gguf_moe_selected_ffn, gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_moe_ffn_fused import (
    build_gguf_q4_k_moe_ffn_fused,
    gguf_q4_k_selected_ffn_fused_bf16_bf16_out,
    gguf_q4_k_selected_ffn_fused_f32_f32_out,
)
from hipengine.quant.gguf import GGMLQuantizationType
from tests.test_cpu_reference_moe_ffn import (
    FFN_LEN,
    HIDDEN,
    NUM_EXPERTS,
    TOKENS,
    TOP_K,
    build_b0_fixture,
)

Q4_K = GGMLQuantizationType.Q4_K
ROWS = TOKENS * TOP_K


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def fused_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q4_k_moe_ffn_fused(load=True)


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    nan_mask = np.isnan(f32)
    lsb = (u32 >> 16) & 1
    rounded = ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)
    rounded[nan_mask] = 0x7FC0
    return rounded.reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _selected_flat(f: dict) -> np.ndarray:
    # row r -> token r//TOP_K, lane r%TOP_K; matches kernel x_row = row // (rows/x_rows).
    return np.ascontiguousarray(f["selected_experts"].reshape(-1), dtype=np.int64)


def _silu(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / (np.float32(1.0) + np.exp(-v).astype(np.float32))


def _cpu_per_row_down(f: dict) -> np.ndarray:
    """CPU reference for the per-selected-row down output (no routing combine)."""

    x = f["x"].astype(np.float32)
    sel = f["selected_experts"]
    out = np.zeros((ROWS, HIDDEN), dtype=np.float32)
    for r in range(ROWS):
        t, k = divmod(r, TOP_K)
        e = int(sel[t, k])
        xt = x[t : t + 1]
        gate = gguf_quant_gemv(xt, f["gate_q"][e], Q4_K)
        up = gguf_quant_gemv(xt, f["up_q"][e], Q4_K)
        inter = (_silu(gate) * up).astype(np.float32)
        down = gguf_quant_gemv(inter, f["down_q"][e], Q4_K)
        out[r] = down[0]
    return out


def _run_fused(library, *, dtype: str, x: np.ndarray, selected: np.ndarray, gate, up, down, x_rows: int, rows: int) -> np.ndarray:
    """Launch the fused kernel and return out[rows, hidden] as float32."""

    if dtype == "f32":
        x_dev = np.ascontiguousarray(x, dtype=np.float32)
        out_host = np.zeros((rows, HIDDEN), dtype=np.float32)
        launch = gguf_q4_k_selected_ffn_fused_f32_f32_out
    elif dtype == "bf16":
        x_dev = _f32_to_bf16_u16(x)
        out_host = np.zeros((rows, HIDDEN), dtype=np.uint16)
        launch = gguf_q4_k_selected_ffn_fused_bf16_bf16_out
    else:
        raise ValueError(dtype)

    gate_b = np.ascontiguousarray(gate, dtype=np.uint8)
    up_b = np.ascontiguousarray(up, dtype=np.uint8)
    down_b = np.ascontiguousarray(down, dtype=np.uint8)
    sel_b = np.ascontiguousarray(selected, dtype=np.int64)

    bufs = []
    try:
        x_buf = malloc(x_dev.nbytes); copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes); bufs.append(x_buf)
        sel_buf = malloc(sel_b.nbytes); copy_host_to_device(sel_buf, host_array_ptr(sel_b), sel_b.nbytes); bufs.append(sel_buf)
        gate_buf = malloc(gate_b.nbytes); copy_host_to_device(gate_buf, host_array_ptr(gate_b), gate_b.nbytes); bufs.append(gate_buf)
        up_buf = malloc(up_b.nbytes); copy_host_to_device(up_buf, host_array_ptr(up_b), up_b.nbytes); bufs.append(up_buf)
        down_buf = malloc(down_b.nbytes); copy_host_to_device(down_buf, host_array_ptr(down_b), down_b.nbytes); bufs.append(down_buf)
        out_buf = malloc(out_host.nbytes); bufs.append(out_buf)
        launch(
            x_buf.ptr, sel_buf.ptr, gate_buf.ptr, up_buf.ptr, down_buf.ptr, out_buf.ptr,
            x_rows, rows, NUM_EXPERTS, HIDDEN, FFN_LEN, threads=256, library=library,
        )
        copy_device_to_host(host_array_ptr(out_host), out_buf, out_host.nbytes)
    finally:
        for buf in bufs:
            free(buf)
    return out_host if dtype == "f32" else _bf16_u16_to_f32(out_host)


def _combine(per_row: np.ndarray, routing_weights: np.ndarray) -> np.ndarray:
    out = np.zeros((TOKENS, HIDDEN), dtype=np.float32)
    for t in range(TOKENS):
        for k in range(TOP_K):
            out[t] += np.float32(routing_weights[t, k]) * per_row[t * TOP_K + k]
    return out


def test_fused_f32_matches_cpu_per_row_down(fused_library):
    f = build_b0_fixture()
    gpu = _run_fused(
        fused_library, dtype="f32", x=f["x"], selected=_selected_flat(f),
        gate=f["gate_q"], up=f["up_q"], down=f["down_q"], x_rows=TOKENS, rows=ROWS,
    )
    cpu = _cpu_per_row_down(f)
    max_abs = float(np.max(np.abs(gpu - cpu)))
    max_rel = float(np.max(np.abs(gpu - cpu) / np.maximum(np.abs(cpu), 1.0)))
    assert max_rel < 5e-3, f"max_abs={max_abs} max_rel={max_rel}"


def test_fused_f32_combine_matches_b0_oracle(fused_library):
    f = build_b0_fixture()
    gpu = _run_fused(
        fused_library, dtype="f32", x=f["x"], selected=_selected_flat(f),
        gate=f["gate_q"], up=f["up_q"], down=f["down_q"], x_rows=TOKENS, rows=ROWS,
    )
    combined = _combine(gpu, f["routing_weights"])
    oracle = gguf_moe_selected_ffn(
        f["x"], f["selected_experts"], f["routing_weights"], f["gate_q"], f["up_q"], f["down_q"],
        Q4_K, Q4_K, Q4_K,
    )
    max_rel = float(np.max(np.abs(combined - oracle) / np.maximum(np.abs(oracle), 1.0)))
    assert max_rel < 5e-3, f"max_rel={max_rel}"


def test_fused_bf16_combine_passes_kl_top1_gate(fused_library):
    f = build_b0_fixture()
    gpu = _run_fused(
        fused_library, dtype="bf16", x=f["x"], selected=_selected_flat(f),
        gate=f["gate_q"], up=f["up_q"], down=f["down_q"], x_rows=TOKENS, rows=ROWS,
    )
    combined = _combine(gpu, f["routing_weights"])
    oracle = gguf_moe_selected_ffn(
        f["x"], f["selected_experts"], f["routing_weights"], f["gate_q"], f["up_q"], f["down_q"],
        Q4_K, Q4_K, Q4_K,
    )
    result = evaluate_logits(oracle, combined)
    assert result.passed, f"kl_mean={result.kl_mean} kl_max={result.kl_max} top1={result.top1_agreement}"


def test_fused_is_row_invariant(fused_library):
    """rows=1 alone is bit-identical to the same row inside the full batch."""

    f = build_b0_fixture()
    batched = _run_fused(
        fused_library, dtype="f32", x=f["x"], selected=_selected_flat(f),
        gate=f["gate_q"], up=f["up_q"], down=f["down_q"], x_rows=TOKENS, rows=ROWS,
    )
    sel_flat = _selected_flat(f)
    for r in range(ROWS):
        t = r // TOP_K
        single = _run_fused(
            fused_library, dtype="f32", x=f["x"][t : t + 1], selected=sel_flat[r : r + 1],
            gate=f["gate_q"], up=f["up_q"], down=f["down_q"], x_rows=1, rows=1,
        )
        np.testing.assert_array_equal(single[0], batched[r])


def test_fused_inactive_expert_emits_zeros(fused_library):
    f = build_b0_fixture()
    sel = _selected_flat(f).copy()
    sel[0] = -1  # inactive lane
    gpu = _run_fused(
        fused_library, dtype="f32", x=f["x"], selected=sel,
        gate=f["gate_q"], up=f["up_q"], down=f["down_q"], x_rows=TOKENS, rows=ROWS,
    )
    np.testing.assert_array_equal(gpu[0], np.zeros((HIDDEN,), dtype=np.float32))
