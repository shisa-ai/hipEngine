"""GPU tier: Muse Glimmer shared-row WMMA decode/verify attention.

The kernel replaces per-row decode attention for both AR decode and the exact
DFlash block verify, so it carries two contracts:

* Outer floor vs the f32 CPU reference over the same BF16 operands
  (max |err| <= 0.02 on BF16 outputs), for decode and multi-row causal blocks,
  with and without the sliding window.
* Verify exactness: row i of a block equals a tokens=1 call at that row's
  position bit-for-bit, on both sides of split boundaries and for SWA windows
  that start mid-tile; the device ``key_end`` (decode-graph) path equals the
  host one.
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

N_Q, N_KV, D = 32, 2, 128
SPLIT_KEYS, N_BLOCKS = 64, 16


def _bf16(x: np.ndarray) -> np.ndarray:
    from hipengine.loading.materialize import float_array_to_bf16_bits
    return float_array_to_bf16_bits(np.asarray(x, dtype=np.float32))


def _f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


class _Bufs:
    def __init__(self):
        from hipengine.core.hip import get_hip_runtime
        self.rt = get_hip_runtime()
        self.items = []

    def up(self, a):
        from hipengine.core.memory import copy_host_array_to_device, malloc
        b = malloc(max(a.nbytes, 2), runtime=self.rt)
        copy_host_array_to_device(b, np.ascontiguousarray(a), runtime=self.rt)
        self.items.append(b)
        return b

    def read(self, b, shape, dtype=np.uint16):
        from hipengine.core.memory import copy_device_to_host
        out = np.empty(shape, dtype=dtype)
        self.rt.device_synchronize()
        copy_device_to_host(out.ctypes.data, b, out.nbytes, runtime=self.rt)
        return out

    def close(self):
        from hipengine.core.memory import free
        for b in reversed(self.items):
            free(b, runtime=self.rt)


def _attend(bufs, q, kc, vc, *, key_end, tokens, window, gate=None, key_end_dev=0):
    from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (
        muse_glimmer_decode_attention_shared,
    )

    dq = bufs.up(q)
    out = bufs.up(np.zeros(tokens * N_Q * D, np.uint16))
    parts = bufs.up(np.zeros(N_BLOCKS * tokens * N_Q * 130, np.float32))
    muse_glimmer_decode_attention_shared(
        dq.ptr, kc.ptr, vc.ptr, out.ptr, parts.ptr, n_q_heads=N_Q, n_kv_heads=N_KV,
        key_begin=0, key_end=key_end, split_keys=SPLIT_KEYS, n_blocks=N_BLOCKS,
        tokens=tokens, window=window, scale=D ** -0.5,
        gate_ptr=0 if gate is None else gate.ptr, key_end_dev=key_end_dev, runtime=bufs.rt,
    )
    return bufs.read(out, (tokens, N_Q * D))


def _cache(rng, keys):
    return _bf16(rng.standard_normal(keys * N_KV * D)), _bf16(rng.standard_normal(keys * N_KV * D))


@pytest.mark.parametrize("keys,tokens,window", [
    (37, 1, 0), (512, 1, 0), (3000, 1, 2048), (1100, 5, 0), (2500, 5, 1000), (700, 12, 0),
])
def test_shared_attention_matches_cpu_reference(keys, tokens, window) -> None:
    from hipengine.kernels.cpu_reference.muse_glimmer import gqa_attention

    rng = np.random.default_rng(keys * 7 + tokens)
    q = _bf16(rng.standard_normal(tokens * N_Q * D) * 0.5)
    kb, vb = _cache(rng, keys)
    bufs = _Bufs()
    try:
        got = _f32(_attend(bufs, q, bufs.up(kb), bufs.up(vb), key_end=keys, tokens=tokens,
                           window=window)).reshape(tokens, N_Q, D)
    finally:
        bufs.close()
    pos = np.arange(keys - tokens, keys)[:, None]
    key = np.arange(keys)[None, :]
    mask = key <= pos
    if window:
        mask &= key > pos - window
    ref = gqa_attention(_f32(q).reshape(tokens, N_Q, D), _f32(kb).reshape(keys, N_KV, D),
                        _f32(vb).reshape(keys, N_KV, D), mask, n_heads=N_Q,
                        n_kv_heads=N_KV, scale=D ** -0.5)
    assert np.isfinite(got).all()
    err = float(np.abs(got - ref).max())
    assert err <= 0.02, f"keys={keys} tokens={tokens} window={window}: max|err| {err}"


@pytest.mark.parametrize("key_end,tokens,window", [
    (64 * 7 + 3, 5, 0),       # block straddles a split boundary
    (64 * 40, 5, 0),          # block ends exactly on a split boundary
    (64 * 50 + 9, 8, 300),    # SWA window starting mid-tile, full wave group
    (64 * 33 + 1, 13, 2048),  # two token groups, window
    (5, 5, 0),                # whole context is the block
])
def test_verify_rows_equal_decode_steps(key_end, tokens, window) -> None:
    rng = np.random.default_rng(key_end + tokens)
    q = _bf16(rng.standard_normal(tokens * N_Q * D) * 0.5)
    gate = _bf16(rng.standard_normal(tokens * N_Q * D) * 4)
    kb, vb = _cache(rng, key_end)
    bufs = _Bufs()
    try:
        kc, vc = bufs.up(kb), bufs.up(vb)
        dg = bufs.up(gate)
        block = _attend(bufs, q, kc, vc, key_end=key_end, tokens=tokens, window=window, gate=dg)
        row_q = q.reshape(tokens, N_Q * D)
        row_g = gate.reshape(tokens, N_Q * D)
        for i in range(tokens):
            gi = bufs.up(row_g[i])
            single = _attend(bufs, row_q[i], kc, vc, key_end=key_end - tokens + i + 1,
                             tokens=1, window=window, gate=gi)
            np.testing.assert_array_equal(single[0], block[i], err_msg=f"row {i}")
    finally:
        bufs.close()


def test_device_key_end_matches_host_key_end() -> None:
    rng = np.random.default_rng(3)
    keys = 64 * 9 + 5
    q = _bf16(rng.standard_normal(N_Q * D) * 0.5)
    kb, vb = _cache(rng, keys + 100)
    bufs = _Bufs()
    try:
        kc, vc = bufs.up(kb), bufs.up(vb)
        host = _attend(bufs, q, kc, vc, key_end=keys, tokens=1, window=2048)
        dev_end = bufs.up(np.asarray([keys], dtype=np.int64))
        # The host extent is a capacity bound when the device extent is given.
        dev = _attend(bufs, q, kc, vc, key_end=keys + 100, tokens=1, window=2048,
                      key_end_dev=dev_end.ptr)
        np.testing.assert_array_equal(dev, host)
    finally:
        bufs.close()


def test_masked_keys_cannot_move_a_verify_row() -> None:
    """Real-model case (layer call 13 of a 5-row verify at key_end 35, captured
    from Muse Glimmer greedy decode). Before each wave zeroed the K/V of keys
    its row cannot see, row 2 differed from its decode step by one BF16 ulp
    (0xb6e3 vs 0xb6e2): the f16 WMMA is not exactly invariant to V entries
    multiplied by a zero probability."""

    from pathlib import Path

    case = np.load(Path(__file__).parent.parent / "fixtures" / "muse_glimmer"
                   / "shared_attention_masked_v_case.npz")
    key_end, tokens, window = int(case["key_end"]), int(case["tokens"]), int(case["window"])
    q = case["Q"]
    bufs = _Bufs()
    try:
        kc, vc = bufs.up(case["K"]), bufs.up(case["V"])
        block = _attend(bufs, q, kc, vc, key_end=key_end, tokens=tokens, window=window)
        rows = q.reshape(tokens, N_Q * D)
        for i in range(tokens):
            single = _attend(bufs, rows[i], kc, vc, key_end=key_end - tokens + i + 1,
                             tokens=1, window=window)
            np.testing.assert_array_equal(single[0], block[i], err_msg=f"row {i}")
    finally:
        bufs.close()
