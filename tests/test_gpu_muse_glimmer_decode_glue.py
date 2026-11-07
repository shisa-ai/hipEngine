"""GPU tier: fused Glimmer decode glue is bit-identical to the launch chain it
replaces.

Strict parent-parity contract: ``muse_glimmer_qk_norm_rope_kv_append`` equals
bf16_to_f32 x2 -> gguf head-norm partial rotary (256 threads, rotary_dim ==
head_dim) -> f32_to_bf16 x2 -> K/V append, bit for bit, for RoPE tables and
identity (NoPE) tables at several positions.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
    _HIP = True
except OSError:
    _HIP = False

pytestmark = pytest.mark.skipif(not _HIP, reason="ROCm runtime not available")

N_Q, N_KV, D, MAX_POS = 32, 2, 128, 64


def _bf16(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


@pytest.mark.parametrize("position", [0, 7, 63])
@pytest.mark.parametrize("nope", [False, True])
def test_fused_qk_rope_append_matches_chain(position: int, nope: bool) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (
        muse_glimmer_kv_append,
        muse_glimmer_qk_norm_rope_kv_append,
    )
    from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32, f32_to_bf16
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
        gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight,
    )

    rt = get_hip_runtime()
    rng = np.random.default_rng(position * 2 + int(nope))
    q_dim, kv_dim = N_Q * D, N_KV * D
    q = _bf16(rng.standard_normal(q_dim) * 3)
    k = _bf16(rng.standard_normal(kv_dim) * 3)
    v = _bf16(rng.standard_normal(kv_dim))
    qw = (rng.random(D) + 0.5).astype(np.float32)
    kw = np.ones(D, dtype=np.float32)
    if nope:
        cos = np.ones((MAX_POS, D), dtype=np.float32)
        sin = np.zeros((MAX_POS, D), dtype=np.float32)
    else:
        ang = rng.random((MAX_POS, D)).astype(np.float32) * 6.0
        cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    bufs = []

    def up(a):
        b = malloc(a.nbytes, runtime=rt)
        copy_host_array_to_device(b, np.ascontiguousarray(a), runtime=rt)
        bufs.append(b)
        return b

    def zeros(nbytes):
        return up(np.zeros(nbytes, dtype=np.uint8))

    def read(b, n, dtype=np.uint16):
        out = np.empty(n, dtype=dtype)
        rt.device_synchronize()
        copy_device_to_host(out.ctypes.data, b, runtime=rt)
        return out

    try:
        dq, dk, dv = up(q), up(k), up(v)
        dqw, dkw, dcos, dsin = up(qw), up(kw), up(cos), up(sin)
        dpos = up(np.asarray([position], dtype=np.int64))
        # Reference chain.
        qf, kf, qr, kr = zeros(q_dim * 4), zeros(kv_dim * 4), zeros(q_dim * 4), zeros(kv_dim * 4)
        q_ref, k_ref = zeros(q_dim * 2), zeros(kv_dim * 2)
        kc_ref, vc_ref = zeros(MAX_POS * kv_dim * 2), zeros(MAX_POS * kv_dim * 2)
        bf16_to_f32(dq.ptr, qf.ptr, q_dim, runtime=rt)
        bf16_to_f32(dk.ptr, kf.ptr, kv_dim, runtime=rt)
        gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight(
            qf.ptr, kf.ptr, dqw.ptr, dkw.ptr, dcos.ptr, dsin.ptr, dpos.ptr,
            qr.ptr, kr.ptr, 1e-5, 1, N_Q, N_KV, D, D, MAX_POS, runtime=rt,
        )
        f32_to_bf16(qr.ptr, q_ref.ptr, q_dim, runtime=rt)
        f32_to_bf16(kr.ptr, k_ref.ptr, kv_dim, runtime=rt)
        muse_glimmer_kv_append(kc_ref.ptr, k_ref.ptr, dpos.ptr, row_elems=kv_dim, n=kv_dim, runtime=rt)
        muse_glimmer_kv_append(vc_ref.ptr, dv.ptr, dpos.ptr, row_elems=kv_dim, n=kv_dim, runtime=rt)
        # Fused.
        q_out = zeros(q_dim * 2)
        kc, vc = zeros(MAX_POS * kv_dim * 2), zeros(MAX_POS * kv_dim * 2)
        muse_glimmer_qk_norm_rope_kv_append(
            dq.ptr, dk.ptr, dv.ptr, dqw.ptr, dkw.ptr, dcos.ptr, dsin.ptr, dpos.ptr,
            q_out.ptr, kc.ptr, vc.ptr, eps=1e-5, n_q_heads=N_Q, n_kv_heads=N_KV,
            head_dim=D, max_positions=MAX_POS, runtime=rt,
        )
        np.testing.assert_array_equal(read(q_out, q_dim), read(q_ref, q_dim))
        np.testing.assert_array_equal(read(kc, MAX_POS * kv_dim), read(kc_ref, MAX_POS * kv_dim))
        np.testing.assert_array_equal(read(vc, MAX_POS * kv_dim), read(vc_ref, MAX_POS * kv_dim))
    finally:
        for b in bufs:
            free(b, runtime=rt)


@pytest.mark.parametrize("tokens", [2, 5, 8])
@pytest.mark.parametrize("nope", [False, True])
def test_fused_qk_rope_append_multirow_matches_per_row_chain(tokens: int, nope: bool) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (
        muse_glimmer_kv_append,
        muse_glimmer_qk_norm_rope_kv_append,
    )
    from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32, f32_to_bf16
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
        gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight,
    )

    rt = get_hip_runtime()
    rng = np.random.default_rng(tokens * 3 + int(nope))
    q_dim, kv_dim = N_Q * D, N_KV * D
    q = _bf16(rng.standard_normal(tokens * q_dim) * 3)
    k = _bf16(rng.standard_normal(tokens * kv_dim) * 3)
    v = _bf16(rng.standard_normal(tokens * kv_dim))
    qw = (rng.random(D) + 0.5).astype(np.float32)
    kw = np.ones(D, dtype=np.float32)
    positions = np.asarray([3, 11, 0, 47, 29, 8, 63, 1][:tokens], dtype=np.int64)
    if nope:
        cos = np.ones((MAX_POS, D), dtype=np.float32)
        sin = np.zeros((MAX_POS, D), dtype=np.float32)
    else:
        ang = rng.random((MAX_POS, D)).astype(np.float32) * 6.0
        cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    bufs = []

    def up(a):
        b = malloc(a.nbytes, runtime=rt)
        copy_host_array_to_device(b, np.ascontiguousarray(a), runtime=rt)
        bufs.append(b)
        return b

    def zeros(nbytes):
        return up(np.zeros(nbytes, dtype=np.uint8))

    def read(b, n, dtype=np.uint16):
        out = np.empty(n, dtype=dtype)
        rt.device_synchronize()
        copy_device_to_host(out.ctypes.data, b, runtime=rt)
        return out

    try:
        dq, dk, dv = up(q), up(k), up(v)
        dqw, dkw, dcos, dsin = up(qw), up(kw), up(cos), up(sin)
        dpos = up(positions)
        # Per-row reference chain, appended into shared caches.
        kc_ref, vc_ref = zeros(MAX_POS * kv_dim * 2), zeros(MAX_POS * kv_dim * 2)
        q_ref_rows = []
        for row in range(tokens):
            pos1 = up(np.asarray([positions[row]], dtype=np.int64))
            qf, kf = zeros(q_dim * 4), zeros(kv_dim * 4)
            qr, kr = zeros(q_dim * 4), zeros(kv_dim * 4)
            q_ref, k_ref = zeros(q_dim * 2), zeros(kv_dim * 2)
            bf16_to_f32(dq.ptr + row * q_dim * 2, qf.ptr, q_dim, runtime=rt)
            bf16_to_f32(dk.ptr + row * kv_dim * 2, kf.ptr, kv_dim, runtime=rt)
            gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight(
                qf.ptr, kf.ptr, dqw.ptr, dkw.ptr, dcos.ptr, dsin.ptr, pos1.ptr,
                qr.ptr, kr.ptr, 1e-5, 1, N_Q, N_KV, D, D, MAX_POS, runtime=rt,
            )
            f32_to_bf16(qr.ptr, q_ref.ptr, q_dim, runtime=rt)
            f32_to_bf16(kr.ptr, k_ref.ptr, kv_dim, runtime=rt)
            muse_glimmer_kv_append(kc_ref.ptr, k_ref.ptr, pos1.ptr, row_elems=kv_dim, n=kv_dim, runtime=rt)
            muse_glimmer_kv_append(vc_ref.ptr, dv.ptr + row * kv_dim * 2, pos1.ptr,
                                   row_elems=kv_dim, n=kv_dim, runtime=rt)
            q_ref_rows.append(read(q_ref, q_dim))
        # Fused multi-row.
        q_out = zeros(tokens * q_dim * 2)
        kc, vc = zeros(MAX_POS * kv_dim * 2), zeros(MAX_POS * kv_dim * 2)
        muse_glimmer_qk_norm_rope_kv_append(
            dq.ptr, dk.ptr, dv.ptr, dqw.ptr, dkw.ptr, dcos.ptr, dsin.ptr, dpos.ptr,
            q_out.ptr, kc.ptr, vc.ptr, eps=1e-5, n_q_heads=N_Q, n_kv_heads=N_KV,
            head_dim=D, max_positions=MAX_POS, tokens=tokens, runtime=rt,
        )
        np.testing.assert_array_equal(read(q_out, tokens * q_dim), np.concatenate(q_ref_rows))
        np.testing.assert_array_equal(read(kc, MAX_POS * kv_dim), read(kc_ref, MAX_POS * kv_dim))
        np.testing.assert_array_equal(read(vc, MAX_POS * kv_dim), read(vc_ref, MAX_POS * kv_dim))
    finally:
        for b in bufs:
            free(b, runtime=rt)


@pytest.mark.parametrize("hidden", [6656, 4096, 300, 9000])
def test_fused_add_rmsnorm_rmsnorm_matches_chain(hidden: int) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_array_to_device, free, malloc
    from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (
        muse_glimmer_add_rmsnorm_rmsnorm,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
        gemma4_add_rmsnorm_scale_bf16,
        gemma4_rmsnorm_f32w_bf16,
    )

    rt = get_hip_runtime()
    rng = np.random.default_rng(hidden)
    branch = _bf16(rng.standard_normal(hidden) * 4)
    resid = _bf16(rng.standard_normal(hidden) * 20)
    w1 = (rng.random(hidden) * 2).astype(np.float32)
    w2 = (rng.random(hidden) + 0.5).astype(np.float32)
    bufs = []

    def up(a):
        b = malloc(a.nbytes, runtime=rt)
        copy_host_array_to_device(b, np.ascontiguousarray(a), runtime=rt)
        bufs.append(b)
        return b

    def read(b):
        out = np.empty(hidden, dtype=np.uint16)
        rt.device_synchronize()
        copy_device_to_host(out.ctypes.data, b, runtime=rt)
        return out

    try:
        db, dr, dw1, dw2 = up(branch), up(resid), up(w1), up(w2)
        z = np.zeros(hidden, dtype=np.uint16)
        r_ref, n_ref, r_out, n_out = up(z), up(z), up(z), up(z)
        gemma4_add_rmsnorm_scale_bf16(db.ptr, dr.ptr, dw1.ptr, 0, r_ref.ptr, 1, hidden, 1e-8, runtime=rt)
        gemma4_rmsnorm_f32w_bf16(r_ref.ptr, dw2.ptr, n_ref.ptr, 1, hidden, 1e-5, runtime=rt)
        muse_glimmer_add_rmsnorm_rmsnorm(
            db.ptr, dr.ptr, dw1.ptr, dw2.ptr, r_out.ptr, n_out.ptr,
            rows=1, hidden_size=hidden, post_eps=1e-8, pre_eps=1e-5, runtime=rt,
        )
        np.testing.assert_array_equal(read(r_out), read(r_ref))
        np.testing.assert_array_equal(read(n_out), read(n_ref))
    finally:
        for b in bufs:
            free(b, runtime=rt)


@pytest.mark.parametrize("keys,window", [(37, 0), (512, 0), (3000, 2048)])
def test_gated_decode_attention_matches_combine_then_gate(keys: int, window: int) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_array_to_device, free, malloc
    from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (
        muse_glimmer_decode_attention,
    )
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import gguf_gate_mul_bf16

    rt = get_hip_runtime()
    rng = np.random.default_rng(keys)
    q_dim, kv_dim = N_Q * D, N_KV * D
    q = _bf16(rng.standard_normal(q_dim))
    kc = _bf16(rng.standard_normal(keys * kv_dim))
    vc = _bf16(rng.standard_normal(keys * kv_dim))
    gate = _bf16(rng.standard_normal(q_dim) * 4)
    bufs = []

    def up(a):
        b = malloc(a.nbytes, runtime=rt)
        copy_host_array_to_device(b, np.ascontiguousarray(a), runtime=rt)
        bufs.append(b)
        return b

    def read(b):
        out = np.empty(q_dim, dtype=np.uint16)
        rt.device_synchronize()
        copy_device_to_host(out.ctypes.data, b, runtime=rt)
        return out

    try:
        dq, dk, dv, dg = up(q), up(kc), up(vc), up(gate)
        parts = up(np.zeros(64 * N_Q * 130, dtype=np.float32))
        ref, out = up(np.zeros(q_dim, np.uint16)), up(np.zeros(q_dim, np.uint16))
        common = dict(n_q_heads=N_Q, n_kv_heads=N_KV, key_begin=0, key_end=keys,
                      n_splits=64, tokens=1, window=window, scale=D ** -0.5, runtime=rt)
        muse_glimmer_decode_attention(dq.ptr, dk.ptr, dv.ptr, ref.ptr, parts.ptr, **common)
        gguf_gate_mul_bf16(ref.ptr, dg.ptr, ref.ptr, q_dim, runtime=rt)
        muse_glimmer_decode_attention(dq.ptr, dk.ptr, dv.ptr, out.ptr, parts.ptr,
                                      gate_ptr=dg.ptr, **common)
        np.testing.assert_array_equal(read(out), read(ref))
    finally:
        for b in bufs:
            free(b, runtime=rt)
