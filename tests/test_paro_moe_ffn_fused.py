"""B3: fused selected-expert PARO MoE FFN megakernel correctness.

Gates the fused PARO megakernel (one block per selected (token, expert) row;
``rotate1 -> AWQ gate_up -> silu*mul -> down_rotate -> AWQ down`` with both
incoherence rotations and the ffn_len-wide intermediate kept on-chip) against
the B0 PARO oracle (``paro_moe_selected_ffn`` / ``build_paro_fixture``):

  * f32 path: per-selected-row down output matches a CPU per-row recompute
    composed from the GPU-validated PARO primitives, and the routing-weighted
    combine matches the B0 oracle tightly;
  * bf16 path: the combine clears the project KL<=0.05 / top-1>=90% gate vs B0;
  * row-invariance: rows=1 alone is bit-identical to the same row inside a batch
    (the T1 verifier self-consistency requirement);
  * inactive expert lanes emit zeros.

The synthetic AWQ weights / rotations / inputs are reused from the B0 fixture.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.cpu_reference import awq_pack8_gemv_transposed, paro_rotate1
from hipengine.kernels.cpu_reference.ops import _silu
from hipengine.kernels.hip_gfx1100.quant.paro_moe_ffn_fused import (
    build_paro_moe_ffn_fused,
    paro_selected_ffn_fused_bf16_bf16_out,
    paro_selected_ffn_fused_f32_f32_out,
    paro_selected_ffn_fused_fp16_fp16_out,
)
from tests.test_cpu_reference_paro_moe_ffn import (
    FFN_LEN,
    GROUP_SIZE,
    HIDDEN,
    KROT,
    NUM_EXPERTS,
    TOP_K,
    TOKENS,
    _oracle,
    build_paro_fixture,
)

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
    return build_paro_moe_ffn_fused(load=True)


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
    return np.ascontiguousarray(f["selected"].reshape(-1), dtype=np.int64)


def _cpu_per_row_down(f: dict) -> np.ndarray:
    """CPU reference for the per-selected-row down output (no routing combine)."""

    x = f["x"].astype(np.float32)
    sel = f["selected"]
    gqw, gqz, gsc = f["gate"]
    uqw, uqz, usc = f["up"]
    dqw, dqz, dsc = f["down"]
    r1p, r1t, r1s, r1k = f["r1"]
    drp, drt, drs, drk = f["dr"]
    x_rot = paro_rotate1(x, r1p, r1t, r1s, GROUP_SIZE, int(r1k))
    out = np.zeros((ROWS, HIDDEN), dtype=np.float32)
    for r in range(ROWS):
        t, k = divmod(r, TOP_K)
        e = int(sel[t, k])
        xt = x_rot[t : t + 1]
        gate = awq_pack8_gemv_transposed(xt, gqw[e], gqz[e], gsc[e], HIDDEN, FFN_LEN, GROUP_SIZE)
        up = awq_pack8_gemv_transposed(xt, uqw[e], uqz[e], usc[e], HIDDEN, FFN_LEN, GROUP_SIZE)
        act = (_silu(gate) * up).astype(np.float32)
        act_rot = paro_rotate1(act, drp, drt, drs, GROUP_SIZE, int(drk))
        down = awq_pack8_gemv_transposed(act_rot, dqw[e], dqz[e], dsc[e], FFN_LEN, HIDDEN, GROUP_SIZE)
        out[r] = down[0]
    return out


def _combine(per_row: np.ndarray, routing: np.ndarray) -> np.ndarray:
    out = np.zeros((TOKENS, HIDDEN), dtype=np.float32)
    for t in range(TOKENS):
        for k in range(TOP_K):
            out[t] += np.float32(routing[t, k]) * per_row[t * TOP_K + k]
    return out


def _upload(host: np.ndarray, bufs: list) -> int:
    arr = np.ascontiguousarray(host)
    buf = malloc(arr.nbytes)
    copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
    bufs.append(buf)
    bufs.append(arr)  # keep the host array alive until after the launch
    return buf.ptr


def _run_fused(library, *, dtype: str, x: np.ndarray, selected: np.ndarray, f: dict, x_rows: int, rows: int) -> np.ndarray:
    """Launch the fused PARO kernel and return out[rows, hidden] as float32."""

    if dtype == "f32":
        sca = np.float32
        x_dev = np.ascontiguousarray(x, dtype=np.float32)
        out_host = np.zeros((rows, HIDDEN), dtype=np.float32)
        launch = paro_selected_ffn_fused_f32_f32_out
        cast = lambda a: np.ascontiguousarray(a, dtype=np.float32)
    elif dtype == "bf16":
        x_dev = _f32_to_bf16_u16(x)
        out_host = np.zeros((rows, HIDDEN), dtype=np.uint16)
        launch = paro_selected_ffn_fused_bf16_bf16_out
        cast = _f32_to_bf16_u16
    elif dtype == "fp16":
        x_dev = np.ascontiguousarray(x, dtype=np.float16)
        out_host = np.zeros((rows, HIDDEN), dtype=np.float16)
        launch = paro_selected_ffn_fused_fp16_fp16_out
        cast = lambda a: np.ascontiguousarray(a, dtype=np.float16)
    else:
        raise ValueError(dtype)

    gqw, gqz, gsc = f["gate"]
    uqw, uqz, usc = f["up"]
    dqw, dqz, dsc = f["down"]
    r1p, r1t, r1s, _ = f["r1"]
    drp, drt, drs, _ = f["dr"]

    bufs: list = []
    held: list = []
    try:
        x_ptr = _upload(x_dev, bufs)
        sel_ptr = _upload(np.ascontiguousarray(selected, dtype=np.int64), bufs)
        gqw_ptr = _upload(np.ascontiguousarray(gqw, dtype=np.int32), bufs)
        gqz_ptr = _upload(np.ascontiguousarray(gqz, dtype=np.int32), bufs)
        gsc_ptr = _upload(cast(gsc), bufs)
        uqw_ptr = _upload(np.ascontiguousarray(uqw, dtype=np.int32), bufs)
        uqz_ptr = _upload(np.ascontiguousarray(uqz, dtype=np.int32), bufs)
        usc_ptr = _upload(cast(usc), bufs)
        dqw_ptr = _upload(np.ascontiguousarray(dqw, dtype=np.int32), bufs)
        dqz_ptr = _upload(np.ascontiguousarray(dqz, dtype=np.int32), bufs)
        dsc_ptr = _upload(cast(dsc), bufs)
        r1p_ptr = _upload(np.ascontiguousarray(r1p, dtype=np.int16), bufs)
        r1t_ptr = _upload(cast(r1t), bufs)
        r1s_ptr = _upload(cast(r1s), bufs)
        drp_ptr = _upload(np.ascontiguousarray(drp, dtype=np.int16), bufs)
        drt_ptr = _upload(cast(drt), bufs)
        drs_ptr = _upload(cast(drs), bufs)

        out_buf = malloc(out_host.nbytes)
        bufs.append(out_buf)
        launch(
            x_ptr, sel_ptr,
            gqw_ptr, gqz_ptr, gsc_ptr,
            uqw_ptr, uqz_ptr, usc_ptr,
            dqw_ptr, dqz_ptr, dsc_ptr,
            r1p_ptr, r1t_ptr, r1s_ptr,
            drp_ptr, drt_ptr, drs_ptr,
            out_buf.ptr,
            x_rows, rows, NUM_EXPERTS, HIDDEN, FFN_LEN, GROUP_SIZE, KROT,
            threads=256, library=library,
        )
        copy_device_to_host(host_array_ptr(out_host), out_buf, out_host.nbytes)
    finally:
        for buf in bufs:
            if hasattr(buf, "ptr"):
                free(buf)
    if dtype == "bf16":
        return _bf16_u16_to_f32(out_host)
    return out_host.astype(np.float32)


def test_fused_f32_matches_cpu_per_row_down(fused_library):
    f = build_paro_fixture()
    gpu = _run_fused(fused_library, dtype="f32", x=f["x"], selected=_selected_flat(f), f=f, x_rows=TOKENS, rows=ROWS)
    cpu = _cpu_per_row_down(f)
    max_abs = float(np.max(np.abs(gpu - cpu)))
    max_rel = float(np.max(np.abs(gpu - cpu) / np.maximum(np.abs(cpu), 1.0)))
    assert max_rel < 5e-3, f"max_abs={max_abs} max_rel={max_rel}"


def test_fused_f32_combine_matches_b0_oracle(fused_library):
    f = build_paro_fixture()
    gpu = _run_fused(fused_library, dtype="f32", x=f["x"], selected=_selected_flat(f), f=f, x_rows=TOKENS, rows=ROWS)
    combined = _combine(gpu, f["routing"])
    oracle = _oracle(f)
    max_rel = float(np.max(np.abs(combined - oracle) / np.maximum(np.abs(oracle), 1.0)))
    assert max_rel < 5e-3, f"max_rel={max_rel}"


def test_fused_bf16_combine_passes_kl_top1_gate(fused_library):
    f = build_paro_fixture()
    gpu = _run_fused(fused_library, dtype="bf16", x=f["x"], selected=_selected_flat(f), f=f, x_rows=TOKENS, rows=ROWS)
    combined = _combine(gpu, f["routing"])
    oracle = _oracle(f)
    result = evaluate_logits(oracle, combined)
    assert result.passed, f"kl_mean={result.kl_mean} kl_max={result.kl_max} top1={result.top1_agreement}"


def test_fused_fp16_combine_passes_kl_top1_gate(fused_library):
    # The deployed runtime path is fp16 (theta/scales/activations are F16). This
    # exercises the _Float16 instantiation used by run_moe_c1_fp16.
    f = build_paro_fixture()
    gpu = _run_fused(fused_library, dtype="fp16", x=f["x"], selected=_selected_flat(f), f=f, x_rows=TOKENS, rows=ROWS)
    combined = _combine(gpu, f["routing"])
    oracle = _oracle(f)
    result = evaluate_logits(oracle, combined)
    assert result.passed, f"kl_mean={result.kl_mean} kl_max={result.kl_max} top1={result.top1_agreement}"


def test_fused_is_row_invariant(fused_library):
    """rows=1 alone is bit-identical to the same row inside the full batch."""

    f = build_paro_fixture()
    batched = _run_fused(fused_library, dtype="f32", x=f["x"], selected=_selected_flat(f), f=f, x_rows=TOKENS, rows=ROWS)
    sel_flat = _selected_flat(f)
    for r in range(ROWS):
        t = r // TOP_K
        single = _run_fused(
            fused_library, dtype="f32", x=f["x"][t : t + 1], selected=sel_flat[r : r + 1],
            f=f, x_rows=1, rows=1,
        )
        np.testing.assert_array_equal(single[0], batched[r])


def test_fused_inactive_expert_emits_zeros(fused_library):
    f = build_paro_fixture()
    sel = _selected_flat(f).copy()
    sel[0] = -1  # inactive lane
    gpu = _run_fused(fused_library, dtype="f32", x=f["x"], selected=sel, f=f, x_rows=TOKENS, rows=ROWS)
    np.testing.assert_array_equal(gpu[0], np.zeros((HIDDEN,), dtype=np.float32))
