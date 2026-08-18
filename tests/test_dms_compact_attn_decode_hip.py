"""Numerical RED gate for the HIP DMS compact decode attention kernel
(C2-7 U4).

Gate per the campaign plan: KL <= 0.05 and top-1 >= 90% vs
``compact_attention_reference`` (the same contract the paged decode
kernels carry), plus bit-exact structural cases (live=1 -> the single V
row, live=0 -> zeros) that pin GQA mapping and dense-extent indexing.
The slot buffers carry NaN canary bits past each head's live count, so
any out-of-extent read corrupts the output. GPU cases skip cleanly on
no-ROCm runners.
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
from tests.test_dms_streaming_pack_hip import _bf16_bits, _hip_available

_NAN_BF16 = np.uint16(0x7FC0)  # canary: any pad read yields NaN output


def test_dms_compact_attn_decode_registers_and_build_plan() -> None:
    clear_registry_for_tests()
    from hipengine.kernels.hip_gfx1100.attention import (
        dms_compact_attn_decode_bf16,
        plan_dms_compact_build,
        register_dms_compact_kernels,
    )

    register_dms_compact_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="dms_compact_attn_decode",
            quant="bf16",
            variant="grouped_gqa",
        )
        is dms_compact_attn_decode_bf16
    )
    artifact = plan_dms_compact_build(compiler_version="dms-test-version")
    assert artifact.family == "dms_compact"


def test_dms_compact_attn_decode_wrapper_validates_before_gpu_load() -> None:
    from hipengine.kernels.hip_gfx1100.attention import (
        dms_compact_attn_decode_bf16,
    )

    args = (0, 0, 0, 0, 0, 0, 2, 4, 2, 16, 0.25, 8)
    with pytest.raises(ValueError, match="rows"):
        dms_compact_attn_decode_bf16(*(0, 0, 0, 0, 0, 0, 0, 4, 2, 16, 0.25, 8))
    with pytest.raises(ValueError, match="GQA"):
        dms_compact_attn_decode_bf16(*(args[:6] + (2, 5, 2, 16, 0.25, 8)))
    with pytest.raises(ValueError, match="dim"):
        dms_compact_attn_decode_bf16(*(args[:9] + (0, 0.25, 8)))
    with pytest.raises(ValueError, match="scale"):
        dms_compact_attn_decode_bf16(*(args[:10] + (0.0, 8)))


def _extent_buffers(
    rows: int,
    kv_heads: int,
    dim: int,
    cap: int,
    live: np.ndarray,
    k_rows: np.ndarray,  # [rows, kv_heads, cap, dim] fp32 (real rows in [:live])
    v_rows: np.ndarray,
):
    """Slot-major bf16 buffers with NaN canary bits past each head's live."""
    total = rows * kv_heads * cap
    k_bits = np.full((total, dim), _NAN_BF16, dtype=np.uint16)
    v_bits = np.full((total, dim), _NAN_BF16, dtype=np.uint16)
    base = np.zeros((rows, kv_heads), dtype=np.int32)
    for r in range(rows):
        for h in range(kv_heads):
            s = (r * kv_heads + h) * cap
            base[r, h] = s
            n = int(live[r, h])
            if n:
                k_bits[s:s + n] = _bf16_bits(k_rows[r, h, :n])
                v_bits[s:s + n] = _bf16_bits(v_rows[r, h, :n])
    return k_bits, v_bits, base


def _device_run(
    q: np.ndarray,
    k_bits: np.ndarray,
    v_bits: np.ndarray,
    base: np.ndarray,
    live: np.ndarray,
    dim: int,
    scale: float,
    score_capacity: int,
) -> np.ndarray:
    from hipengine.kernels.hip_gfx1100.attention import (
        build_dms_compact,
        dms_compact_attn_decode_bf16,
    )

    rows, q_heads, _ = q.shape
    kv_heads = base.shape[1]
    out = np.zeros((rows, q_heads, dim), dtype=np.float32)
    buffers: dict[str, object] = {}

    def upload(name: str, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        buf = malloc(array.nbytes)
        buffers[name] = buf
        copy_host_to_device(buf, host_array_ptr(array), array.nbytes)

    try:
        upload("q", q)
        upload("k", k_bits)
        upload("v", v_bits)
        upload("base", base)
        upload("live", live)
        upload("out", out)
        library = build_dms_compact(load=True)
        dms_compact_attn_decode_bf16(
            buffers["q"].ptr,
            buffers["k"].ptr,
            buffers["v"].ptr,
            buffers["base"].ptr,
            buffers["live"].ptr,
            buffers["out"].ptr,
            rows,
            q_heads,
            kv_heads,
            dim,
            scale,
            score_capacity,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), buffers["out"], out.nbytes)
    finally:
        for buf in buffers.values():
            free(buf)
    return out


def _oracle(
    q: np.ndarray,
    k_bits: np.ndarray,
    v_bits: np.ndarray,
    base: np.ndarray,
    live: np.ndarray,
    dim: int,
    scale: float,
) -> np.ndarray:
    from hipengine.kvcache.dms import compact_attention_reference

    rows, kv_heads = live.shape
    cap = int(k_bits.shape[0] // (rows * kv_heads))
    k_f32 = (k_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    v_f32 = (v_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    k_rows = k_f32.reshape(rows, kv_heads, cap, dim)
    v_rows = v_f32.reshape(rows, kv_heads, cap, dim)
    return compact_attention_reference(q, k_rows, v_rows, live, scale=scale)


def _kl_top1(ref: np.ndarray, cand: np.ndarray) -> tuple[float, float]:
    """Max softmax-KL and top-1 agreement, rows flattened to (q_heads*dim).

    Top-1 is tie-tolerant: the candidate's chosen value must be at least
    the reference top value minus a fp32-rounding epsilon, so an exact
    tie in the reference (common with bf16-quantized data) cannot flip
    the count on a 1-ulp rounding difference.
    """
    ref64 = ref.reshape(ref.shape[0], -1).astype(np.float64)
    cand64 = cand.reshape(cand.shape[0], -1).astype(np.float64)

    def logsm(x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    log_ref = logsm(ref64)
    log_cand = logsm(cand64)
    kl = float(np.max(np.sum(np.exp(log_ref) * (log_ref - log_cand), axis=-1)))
    top = ref64.max(axis=-1)
    eps = 1e-5 * np.maximum(1.0, np.abs(top))
    top1 = float(np.mean(cand64.max(axis=-1) >= top - eps))
    return kl, top1


def _gate_case(
    rows: int,
    kv_heads: int,
    group: int,
    dim: int,
    live: list[list[int]],
    seed: int,
) -> None:
    q_heads = kv_heads * group
    cap = int(max(max(row) for row in live)) + 2
    live_arr = np.asarray(live, dtype=np.int32)
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((rows, q_heads, dim)).astype(np.float32)
    k_f32 = rng.normal(0.0, 0.2, size=(rows, kv_heads, cap, dim)).astype(np.float32)
    v_f32 = rng.normal(0.0, 0.2, size=(rows, kv_heads, cap, dim)).astype(np.float32)
    scale = float(dim**-0.5)
    k_bits, v_bits, base = _extent_buffers(
        rows, kv_heads, dim, cap, live_arr, k_f32, v_f32
    )
    score_capacity = int(live_arr.max())
    dev1 = _device_run(q, k_bits, v_bits, base, live_arr, dim, scale, score_capacity)
    assert np.isfinite(dev1).all(), f"NaN/Inf output (pad read?): {dev1}"
    ref = _oracle(q, k_bits, v_bits, base, live_arr, dim, scale)
    kl, top1 = _kl_top1(ref, dev1)
    assert kl <= 0.05, f"max softmax KL {kl:.6f} > 0.05"
    assert top1 >= 0.9, f"top-1 agreement {top1:.3f} < 0.9"
    # Determinism: a second launch is bit-identical.
    dev2 = _device_run(q, k_bits, v_bits, base, live_arr, dim, scale, score_capacity)
    np.testing.assert_array_equal(dev1, dev2, err_msg="non-deterministic output")


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dms_compact_attn_decode_live_zero_one_bit_exact() -> None:
    # Structural pin: live=1 -> the single V row (softmax of one element
    # is exactly 1); live=0 -> zeros (the host oracle rejects live=0, so
    # the kernel defines zero output as the safe neutral). Canary bits in
    # the pad region prove no out-of-extent reads.
    rows, kv_heads, group, dim = 2, 2, 2, 16
    live = np.asarray([[1, 0], [1, 1]], dtype=np.int32)
    cap = 3
    rng = np.random.default_rng(1234)
    v_f32 = rng.standard_normal((rows, kv_heads, cap, dim)).astype(np.float32)
    k_f32 = rng.standard_normal((rows, kv_heads, cap, dim)).astype(np.float32)
    q = rng.standard_normal((rows, kv_heads * group, dim)).astype(np.float32)
    k_bits, v_bits, base = _extent_buffers(
        rows, kv_heads, dim, cap, live, k_f32, v_f32
    )
    out = _device_run(
        q, k_bits, v_bits, base, live, dim, float(dim**-0.5), int(live.max())
    )
    for r in range(rows):
        for h in range(kv_heads):
            n = int(live[r, h])
            # The kernel reads the bf16-quantized V row; expect exactly that.
            expect = (
                _bf16_roundtrip(v_bits[int(base[r, h])])
                if n == 1
                else np.zeros(dim, dtype=np.float32)
            )
            got = out[r, h * group:(h + 1) * group]
            assert np.all(got == expect[None, :]), (
                f"row {r} head {h} live {n}: {got[0]} vs {expect}"
            )


def _bf16_roundtrip(bits: np.ndarray) -> np.ndarray:
    """BF16 bits (uint16) -> FP32 with the same BF16 mantissa (oracle view)."""
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32).copy()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dms_compact_attn_decode_kl_top1_gate_small() -> None:
    _gate_case(
        rows=4,
        kv_heads=2,
        group=2,
        dim=16,
        live=[[5, 3], [1, 7], [4, 4], [2, 6]],
        seed=777,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dms_compact_attn_decode_kl_top1_gate_production_shape() -> None:
    # Geometry near the production Qwen3.5-0.8B decode shape (4 KV heads,
    # group 2, D128) with variable extents up to the window+2 capacity.
    _gate_case(
        rows=2,
        kv_heads=4,
        group=2,
        dim=128,
        live=[[33, 17, 5, 29], [12, 40, 8, 3]],
        seed=778,
    )
