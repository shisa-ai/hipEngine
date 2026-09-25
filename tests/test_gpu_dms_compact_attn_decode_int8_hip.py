"""Numerical gate for the HIP compact-DMS INT8 decode attention kernel.

The DMS retention topology gives every (row, kv head) its own dense live
extent, so ``live_counts`` is ``[rows, kv_heads]`` and each head's K/V is a
different number of tokens. The verifier has to read that same law to serve a
DMS row, and this is the kernel that already implements it: the AR route
resolves ``("dms_compact_attn_decode", "int8_per_token_head",
"grouped_gqa_splitk")`` to it. It had no numerical gate of its own.

Contract: the split-K kernel dequantizes ``int8 * scale`` per slot and attends
each head's dense extent at ``base_offsets`` for ``live_counts`` tokens. The
oracle is ``compact_attention_reference`` -- the registered CPU reference for
this layer -- fed the dequantized payload, so a passing case means the kernel
read exactly the tokens the AR route would read.

Cases pin the structural edges as well as the numbers: live == 0 writes zeros,
live == 1 is bit-exact (a single-row softmax is the V row), and both dispatch
branches are covered (the wave kernel is selected only at 24q/4kv/256d/rows=1).
INT8 payload past each head's live extent carries an extreme value with a large
scale, so any out-of-extent read corrupts the output instead of passing quietly.

GPU cases skip cleanly on no-ROCm runners.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve
from tests.test_gpu_dms_streaming_pack_hip import _hip_available

pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is unavailable"
)

_CANARY_I8 = np.int8(127)
_CANARY_SCALE = np.float32(64.0)


def _dequant_reference(
    q: np.ndarray,
    k_i8: np.ndarray,
    v_i8: np.ndarray,
    k_scale: np.ndarray,
    v_scale: np.ndarray,
    base: np.ndarray,
    live: np.ndarray,
    dim: int,
    scale: float,
) -> np.ndarray:
    """Dequantize the slot planes, then run the registered CPU reference."""

    from hipengine.kvcache.dms import compact_attention_reference

    rows, kv_heads = live.shape
    capacity = int(live.max(initial=0))
    keys = np.zeros((rows, kv_heads, capacity, dim), dtype=np.float32)
    values = np.zeros_like(keys)
    for r in range(rows):
        for h in range(kv_heads):
            n = int(live[r, h])
            if n <= 0:
                continue
            start = int(base[r, h])
            keys[r, h, :n] = k_i8[start:start + n].astype(np.float32) * k_scale[
                start:start + n, None
            ]
            values[r, h, :n] = v_i8[start:start + n].astype(np.float32) * v_scale[
                start:start + n, None
            ]
    return compact_attention_reference(q, keys, values, live, scale=scale)


def _device_run(
    q: np.ndarray,
    k_i8: np.ndarray,
    v_i8: np.ndarray,
    k_scale: np.ndarray,
    v_scale: np.ndarray,
    base: np.ndarray,
    live: np.ndarray,
    dim: int,
    scale: float,
    *,
    chunk: int = 256,
    splits: int | None = None,
) -> np.ndarray:
    from hipengine.kernels.hip_gfx1100.attention.dms_compact_int8 import (
        build_dms_compact_int8,
        dms_compact_attn_decode_splitk_int8,
    )

    rows, q_heads, _ = q.shape
    kv_heads = live.shape[1]
    if splits is None:
        capacity = int(live.max(initial=0))
        splits = max(1, (capacity + chunk - 1) // chunk)
    out = np.zeros((rows, q_heads, dim), dtype=np.float32)
    partial_out = np.zeros((rows * q_heads * splits, dim), dtype=np.float32)
    partial_m = np.zeros((rows * q_heads * splits,), dtype=np.float32)
    partial_l = np.zeros_like(partial_m)
    buffers: dict[str, object] = {}

    def upload(name: str, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        buf = malloc(array.nbytes)
        buffers[name] = buf
        copy_host_to_device(buf, host_array_ptr(array), array.nbytes)

    try:
        for name, array in (
            ("q", q),
            ("k", k_i8),
            ("v", v_i8),
            ("k_scale", k_scale),
            ("v_scale", v_scale),
            ("base", base),
            ("live", live),
            ("po", partial_out),
            ("pm", partial_m),
            ("pl", partial_l),
            ("out", out),
        ):
            upload(name, array)
        dms_compact_attn_decode_splitk_int8(
            buffers["q"].ptr,
            buffers["k"].ptr,
            buffers["v"].ptr,
            buffers["base"].ptr,
            buffers["live"].ptr,
            buffers["po"].ptr,
            buffers["pm"].ptr,
            buffers["pl"].ptr,
            buffers["out"].ptr,
            rows,
            q_heads,
            kv_heads,
            dim,
            scale,
            chunk,
            splits,
            k_scale_ptr=buffers["k_scale"].ptr,
            v_scale_ptr=buffers["v_scale"].ptr,
            library=build_dms_compact_int8(load=True),
        )
        copy_device_to_host(host_array_ptr(out), buffers["out"], out.nbytes)
    finally:
        for buf in buffers.values():
            free(buf)
    return out


def _case_buffers(
    rows: int,
    kv_heads: int,
    dim: int,
    capacity: int,
    live: np.ndarray,
    rng: np.random.Generator,
    *,
    q_heads: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slot-major int8 planes with extreme canaries past each head's live extent."""

    total = rows * kv_heads * capacity
    k_i8 = np.full((total, dim), _CANARY_I8, dtype=np.int8)
    v_i8 = np.full((total, dim), _CANARY_I8, dtype=np.int8)
    k_scale = np.full((total,), _CANARY_SCALE, dtype=np.float32)
    v_scale = np.full((total,), _CANARY_SCALE, dtype=np.float32)
    base = np.zeros((rows, kv_heads), dtype=np.int32)
    for r in range(rows):
        for h in range(kv_heads):
            slot = (r * kv_heads + h) * capacity
            base[r, h] = slot
            n = int(live[r, h])
            if n:
                k_i8[slot:slot + n] = rng.integers(-127, 128, size=(n, dim), dtype=np.int8)
                v_i8[slot:slot + n] = rng.integers(-127, 128, size=(n, dim), dtype=np.int8)
                k_scale[slot:slot + n] = rng.uniform(0.002, 0.02, size=n).astype(np.float32)
                v_scale[slot:slot + n] = rng.uniform(0.002, 0.02, size=n).astype(np.float32)
    heads = int(q_heads) if q_heads is not None else 2 * kv_heads
    assert heads % kv_heads == 0
    q = rng.normal(0.0, 0.6, size=(rows, heads, dim)).astype(np.float32)
    return q, k_i8, v_i8, k_scale, v_scale, base, live


def test_dms_compact_attn_decode_int8_registers_under_its_own_key() -> None:
    clear_registry_for_tests()
    from hipengine.kernels.hip_gfx1100.attention.dms_compact_int8 import (
        dms_compact_attn_decode_splitk_int8,
        register_dms_compact_int8_kernels,
    )

    register_dms_compact_int8_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="dms_compact_attn_decode",
            quant="int8_per_token_head",
            variant="grouped_gqa_splitk",
        )
        is dms_compact_attn_decode_splitk_int8
    )
    # The BF16 sibling's keys must not be shadowed by the INT8 registration.
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="dms_compact_attn_decode",
            quant="bf16",
            variant="grouped_gqa_splitk",
            missing="none",
        )
        is None
    ) or resolve(
        backend="hip_gfx1100",
        layer="dms_compact_attn_decode",
        quant="bf16",
        variant="grouped_gqa_splitk",
    ) is not dms_compact_attn_decode_splitk_int8


def test_dms_compact_attn_decode_int8_wrapper_validates_before_gpu_load() -> None:
    from hipengine.kernels.hip_gfx1100.attention.dms_compact_int8 import (
        dms_compact_attn_decode_splitk_int8,
    )

    pointers = (1,) * 9
    scales = {"k_scale_ptr": 1, "v_scale_ptr": 1}
    with pytest.raises(ValueError, match="GQA"):
        dms_compact_attn_decode_splitk_int8(
            *(pointers + (2, 5, 4, 256, 0.25, 256, 1)), **scales
        )
    with pytest.raises(ValueError, match="chunk"):
        dms_compact_attn_decode_splitk_int8(
            *(pointers + (2, 4, 4, 256, 0.25, 512, 1)), **scales
        )
    with pytest.raises(ValueError, match="GQA geometry"):
        dms_compact_attn_decode_splitk_int8(
            *(pointers + (2, 4, 0, 256, 0.25, 256, 1)), **scales
        )


def test_dms_compact_attn_decode_int8_matches_reference_on_per_head_counts() -> None:
    """Variable live counts per head, read against the registered CPU oracle."""

    rng = np.random.default_rng(20260923)
    rows, kv_heads, dim, capacity = 3, 2, 128, 24
    # The registered CPU reference rejects live <= 0 (its range check is
    # (0, capacity]); the zero-live edge is pinned structurally below instead.
    live = np.array([[3, 5], [7, 1], [24, 2]], dtype=np.int32)
    q, k_i8, v_i8, k_scale, v_scale, base, live = _case_buffers(
        rows, kv_heads, dim, capacity, live, rng
    )
    scale = float(dim) ** -0.5
    expected = _dequant_reference(
        q, k_i8, v_i8, k_scale, v_scale, base, live, dim, scale
    )
    got = _device_run(q, k_i8, v_i8, k_scale, v_scale, base, live, dim, scale, chunk=8)

    assert np.all(np.isfinite(got)), "canary read produced non-finite output"
    assert np.allclose(got, expected, atol=2e-4, rtol=2e-3), {
        "per_head_variable_counts_mismatch": {
            "live": live.tolist(),
            "max_abs_diff": float(np.max(np.abs(got - expected))),
        }
    }


def test_dms_compact_attn_decode_int8_wave_branch_matches_reference() -> None:
    """The 24q/4kv/256d/rows=1 shape takes the dedicated wave kernel."""

    rng = np.random.default_rng(20260924)
    rows, kv_heads, dim, capacity = 1, 4, 256, 32
    live = np.array([[9, 2, 32, 1]], dtype=np.int32)
    q, k_i8, v_i8, k_scale, v_scale, base, live = _case_buffers(
        rows, kv_heads, dim, capacity, live, rng, q_heads=24
    )
    scale = float(dim) ** -0.5
    expected = _dequant_reference(
        q, k_i8, v_i8, k_scale, v_scale, base, live, dim, scale
    )
    got = _device_run(q, k_i8, v_i8, k_scale, v_scale, base, live, dim, scale, chunk=16)

    assert np.all(np.isfinite(got)), "canary read produced non-finite output"
    assert np.allclose(got, expected, atol=2e-4, rtol=2e-3), {
        "wave_branch_mismatch": {
            "live": live.tolist(),
            "max_abs_diff": float(np.max(np.abs(got - expected))),
        }
    }


def test_dms_compact_attn_decode_int8_structural_edges_are_exact() -> None:
    """live == 0 writes zeros; live == 1 is the single V row."""

    rng = np.random.default_rng(20260925)
    rows, kv_heads, dim, capacity = 2, 2, 64, 8
    live = np.array([[0, 1], [1, 0]], dtype=np.int32)
    q, k_i8, v_i8, k_scale, v_scale, base, live = _case_buffers(
        rows, kv_heads, dim, capacity, live, rng
    )
    scale = float(dim) ** -0.5
    got = _device_run(q, k_i8, v_i8, k_scale, v_scale, base, live, dim, scale, chunk=4)

    zero_rows = {(0, 0), (1, 1)}
    for r in range(rows):
        for h in range(kv_heads):
            block = got[r, h * (q.shape[1] // kv_heads):(h + 1) * (q.shape[1] // kv_heads)]
            if (r, h) in zero_rows:
                assert np.all(block == 0.0), f"live==0 must write zeros at ({r},{h})"
                continue
            slot = int(base[r, h])
            expected_v = v_i8[slot].astype(np.float32) * v_scale[slot]
            assert np.allclose(block, expected_v, atol=1e-5, rtol=1e-5), (
                f"live==1 must return the single V row at ({r},{h})"
            )
