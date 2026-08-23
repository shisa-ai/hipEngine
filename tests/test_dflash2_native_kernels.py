"""GPU RED tests for the native DFlash2 kernels (grouped dynamic conv, top-16,
candidate selector) against the D0 CPU-reference oracle.

Strict exact/parent-parity RED contract: kernels read BF16 inputs and
accumulate FP32; the CPU oracle runs on the BF16-rounded inputs (same
discipline as ``test_dflash_dense_wmma.py``) and the two must agree to a small
FP32 tolerance.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.dflash2 import (
    candidate_selector_select,
    grouped_dynamic_convolve,
)


def _has_gpu() -> bool:
    try:  # noqa: SIM105
        from hipengine.core.hip import get_hip_runtime
    except Exception:
        return False
    try:
        get_hip_runtime()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="HIP runtime not available")


def _to_bf16_bits(x_f32: np.ndarray) -> np.ndarray:
    bits = x_f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _from_bf16_bits(x_u16: np.ndarray) -> np.ndarray:
    return (x_u16.astype(np.uint32) << 16).view(np.float32)


@pytest.fixture(scope="module")
def _dflash2_lib(hip_test_target_arch):
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.speculative.dflash2 import build_dflash2

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment(hip_test_target_arch):
        return build_dflash2(load=True, compiler_version=compiler_version)


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()


def _upload(runtime, bufs, array, dtype=None):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    arr = np.ascontiguousarray(array, dtype=dtype if dtype is not None else array.dtype)
    buf = malloc(arr.nbytes, runtime=runtime)
    bufs.append(buf)
    copy_host_to_device(buf, host_array_ptr(arr), runtime=runtime)
    return buf


def _download(runtime, buf, shape, dtype):
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    arr = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(arr), buf, runtime=runtime)
    return arr


def _free_all(runtime, bufs):
    from hipengine.core.memory import free

    for buf in reversed(bufs):
        free(buf, runtime=runtime)


# ---------------------------------------------------------------------------
# Grouped dynamic conv
# ---------------------------------------------------------------------------

def test_dflash2_grouped_conv_red(_dflash2_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.speculative.dflash2 import dflash2_grouped_conv

    rng = np.random.default_rng(0xDF2A1)
    rows, hidden_size, group_size = 8, 5120, 16
    groups = hidden_size // group_size
    x_f32 = (rng.standard_normal((rows, hidden_size), dtype=np.float32) * 0.5)
    # base_kernel (2, hidden); dyn (rows, 2*groups)
    base_f32 = rng.standard_normal((2, hidden_size), dtype=np.float32) * 0.2
    dyn_f32 = rng.standard_normal((rows, 2 * groups), dtype=np.float32) * 0.1
    x_bf = _to_bf16_bits(x_f32)
    base_bf = _to_bf16_bits(base_f32)
    dyn_bf = _to_bf16_bits(dyn_f32)

    # CPU oracle on the BF16-rounded inputs (oracle is rank-3, batch=1).
    oracle = grouped_dynamic_convolve(
        _from_bf16_bits(x_bf)[None],
        _from_bf16_bits(dyn_bf).reshape(1, rows, 2, groups),
        _from_bf16_bits(base_bf),
        group_size,
    )[0]

    bufs = []
    try:
        x_dev = _upload(_runtime, bufs, x_bf)
        dyn_dev = _upload(_runtime, bufs, dyn_bf)
        base_dev = _upload(_runtime, bufs, base_bf)
        out_dev = _upload(_runtime, bufs, np.zeros((rows, hidden_size), np.uint16))
        dflash2_grouped_conv(
            x_dev.ptr, dyn_dev.ptr, base_dev.ptr, out_dev.ptr,
            rows, hidden_size, group_size, library=_dflash2_lib, runtime=_runtime,
        )
        _runtime.device_synchronize()
        got = _from_bf16_bits(_download(_runtime, out_dev, (rows, hidden_size), np.uint16))
        oracle_round = _from_bf16_bits(_to_bf16_bits(oracle))
        scale = max(float(np.max(np.abs(oracle_round))), 1.0)
        np.testing.assert_allclose(got, oracle_round, atol=1e-3 * scale, rtol=1e-3)
        # Zero-padded first row: tap-1 must not leak.
        assert np.abs(got[0] - oracle_round[0]).max() <= 1e-3 * scale
    finally:
        _free_all(_runtime, bufs)


@pytest.mark.parametrize("rows", [1, 7, 64])
def test_dflash2_grouped_conv_prepare_finish_red(
    _dflash2_lib,
    _runtime,
    rows,
    monkeypatch: pytest.MonkeyPatch,
):
    """Full prepare+finish against the oracle path (kernel_projection included).

    The native path keeps BF16 between ops: the projection output is BF16 and
    the conv output is BF16.  The oracle models those round-trips so the RED
    contract isolates the conv kernel math (projection reduction order is
    covered separately by test_dflash_dense_wmma).
    """
    from hipengine.kernels.cpu_reference.ops import linear
    from hipengine.kernels.hip_gfx1100.speculative.dflash2 import dflash2_grouped_conv
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
        dflash_dense_bf16_to_bf16,
    )

    # The WMMA dense path has a pre-existing rows<16 gfx1151 precision quirk
    # (~0.3% max on a few elements); the naive path is exact.  The conv
    # projection is tiny (5120->1280) and uses the naive path for exact RED.
    monkeypatch.setenv("HIPENGINE_DFLASH_DRAFTER_DENSE", "naive")

    rng = np.random.default_rng(0xDF2A2 + rows)
    hidden_size, group_size = 5120, 16
    groups = hidden_size // group_size
    kernel_size = 2
    # kernel_projection (2*kernel_size*groups, hidden) = (1280, 5120)
    proj_f32 = rng.standard_normal((2 * kernel_size * groups, hidden_size), dtype=np.float32) * 0.05
    base_f32 = rng.standard_normal((kernel_size, kernel_size, hidden_size), dtype=np.float32) * 0.2
    h_f32 = rng.standard_normal((rows, hidden_size), dtype=np.float32) * 0.5
    h_bf = _to_bf16_bits(h_f32)
    proj_bf = _to_bf16_bits(proj_f32)
    base_bf = _to_bf16_bits(base_f32)

    # Oracle with native BF16 round-trips.
    hidden_f32 = _from_bf16_bits(h_bf)
    proj_f32 = _from_bf16_bits(proj_bf)
    proj_oracle = linear(hidden_f32, proj_f32)  # (rows, 1280) f32
    proj_oracle_bf = _from_bf16_bits(_to_bf16_bits(proj_oracle))  # bf16 round-trip
    dyn_in = proj_oracle_bf[:, : 2 * groups].reshape(1, rows, 2, groups)
    dyn_out = proj_oracle_bf[:, 2 * groups :].reshape(1, rows, 2, groups)
    oracle_h = grouped_dynamic_convolve(
        hidden_f32[None], dyn_in, _from_bf16_bits(base_bf)[0], group_size
    )[0]
    oracle_h_bf = _from_bf16_bits(_to_bf16_bits(oracle_h))
    oracle_finish = grouped_dynamic_convolve(
        oracle_h_bf[None], dyn_out, _from_bf16_bits(base_bf)[1], group_size
    )[0]

    bufs = []
    try:
        h_dev = _upload(_runtime, bufs, h_bf)
        proj_dev = _upload(_runtime, bufs, proj_bf)
        proj_out = _upload(_runtime, bufs, np.zeros((rows, 2 * kernel_size * groups), np.uint16))
        dflash_dense_bf16_to_bf16(
            h_dev.ptr, proj_dev.ptr, proj_out.ptr, rows, hidden_size,
            2 * kernel_size * groups, runtime=_runtime,
        )
        # input-side conv: base[0], dyn = proj[:, :2*groups] (stride 1280)
        convolved_dev = _upload(_runtime, bufs, np.zeros((rows, hidden_size), np.uint16))
        base_side0 = _upload(_runtime, bufs, base_bf[0].copy())
        dflash2_grouped_conv(
            h_dev.ptr, proj_out.ptr, base_side0.ptr, convolved_dev.ptr,
            rows, hidden_size, group_size,
            dyn_offset=0, dyn_stride=2 * kernel_size * groups,
            library=_dflash2_lib, runtime=_runtime,
        )
        # output-side conv: base[1], dyn = proj[:, 2*groups:] (stride 1280)
        finish_dev = _upload(_runtime, bufs, np.zeros((rows, hidden_size), np.uint16))
        base_side1 = _upload(_runtime, bufs, base_bf[1].copy())
        dflash2_grouped_conv(
            convolved_dev.ptr, proj_out.ptr, base_side1.ptr, finish_dev.ptr,
            rows, hidden_size, group_size,
            dyn_offset=2 * groups, dyn_stride=2 * kernel_size * groups,
            library=_dflash2_lib, runtime=_runtime,
        )
        _runtime.device_synchronize()
        got_h = _from_bf16_bits(_download(_runtime, convolved_dev, (rows, hidden_size), np.uint16))
        got_finish = _from_bf16_bits(_download(_runtime, finish_dev, (rows, hidden_size), np.uint16))
        # Kernel outputs round to BF16; round the oracle outputs to match.
        oracle_h_round = _from_bf16_bits(_to_bf16_bits(oracle_h))
        oracle_finish_round = _from_bf16_bits(_to_bf16_bits(oracle_finish))
        scale_h = max(float(np.max(np.abs(oracle_h_round))), 1.0)
        scale_f = max(float(np.max(np.abs(oracle_finish_round))), 1.0)
        np.testing.assert_allclose(got_h, oracle_h_round, atol=1e-3 * scale_h, rtol=1e-3)
        np.testing.assert_allclose(got_finish, oracle_finish_round, atol=1e-3 * scale_f, rtol=1e-3)
    finally:
        _free_all(_runtime, bufs)


# ---------------------------------------------------------------------------
# Top-16 rows
# ---------------------------------------------------------------------------

def test_dflash2_top16_rows_red(_dflash2_lib, _runtime):
    from hipengine.kernels.cpu_reference.dflash2 import dflash2_topk
    from hipengine.kernels.hip_gfx1100.speculative.dflash2 import dflash2_top16_rows

    rng = np.random.default_rng(0xDF2A3)
    rows, vocab, top_k = 7, 4096, 16
    logits = rng.standard_normal((rows, vocab), dtype=np.float32)
    oracle_vals, oracle_ids = dflash2_topk(logits, top_k)

    bufs = []
    try:
        logits_dev = _upload(_runtime, bufs, np.ascontiguousarray(logits, dtype=np.float32))
        ids_dev = _upload(_runtime, bufs, np.zeros((rows, top_k), np.int32))
        vals_dev = _upload(_runtime, bufs, np.zeros((rows, top_k), np.float32))
        dflash2_top16_rows(
            logits_dev.ptr, ids_dev.ptr, vals_dev.ptr, rows, vocab, top_k,
            library=_dflash2_lib, runtime=_runtime,
        )
        _runtime.device_synchronize()
        got_ids = _download(_runtime, ids_dev, (rows, top_k), np.int32)
        got_vals = _download(_runtime, vals_dev, (rows, top_k), np.float32)
        for r in range(rows):
            assert set(got_ids[r].tolist()) == set(int(i) for i in oracle_ids[r].tolist())
            # values must match descending (order-independent ids, value alignment)
            val_by_id = {int(i): float(v) for i, v in zip(got_ids[r].tolist(), got_vals[r].tolist())}
            for i, v in zip(oracle_ids[r].tolist(), oracle_vals[r].tolist()):
                assert val_by_id[int(i)] == pytest.approx(float(v), rel=1e-6)
    finally:
        _free_all(_runtime, bufs)


# ---------------------------------------------------------------------------
# Sliding-window attention
# ---------------------------------------------------------------------------

def _cpu_sliding_attention(q, k, v, qpos, kpos, *, window, is_causal):
    """Masked GQA attention mirroring dflash2_sliding_attention kernel."""
    b, ql, qh, hd = q.shape
    kv = k.shape[2]
    scale = hd ** -0.5
    groups = qh // kv
    out = np.zeros((b, ql, qh, hd), dtype=np.float32)
    for bi in range(b):
        for qi in range(ql):
            for h in range(qh):
                kvh = h // groups
                scores = np.empty((k.shape[1],), dtype=np.float32)
                for ki in range(k.shape[1]):
                    dist = abs(int(qpos[bi * ql + qi]) - int(kpos[bi * k.shape[1] + ki]))
                    masked = (window > 0 and dist >= window) or (is_causal and int(kpos[bi * k.shape[1] + ki]) > int(qpos[bi * ql + qi]))
                    if masked:
                        scores[ki] = -np.inf
                    else:
                        scores[ki] = float(np.dot(q[bi, qi, h], k[bi, ki, kvh])) * scale
                m = scores.max()
                p = np.exp(scores - m)
                p /= p.sum()
                out[bi, qi, h] = np.sum(p[:, None] * v[bi, :, kvh, :].astype(np.float32), axis=0)
    return out


def test_dflash2_sliding_attention_red(_dflash2_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.speculative.dflash2 import (
        dflash2_sliding_attention_f32_bf16,
    )

    rng = np.random.default_rng(0xDF2A5)
    b, ql, kv_len, qh, kvh, hd = 1, 7, 40, 32, 8, 128
    window = 16
    q = (rng.standard_normal((b, ql, qh, hd), dtype=np.float32) * 0.5)
    k = (rng.standard_normal((b, kv_len, kvh, hd), dtype=np.float32) * 0.5)
    v_f32 = rng.standard_normal((b, kv_len, kvh, hd), dtype=np.float32) * 0.5
    v_bf = _to_bf16_bits(v_f32)
    qpos = np.asarray([20 + i for i in range(ql)], dtype=np.int32)
    kpos = np.arange(kv_len, dtype=np.int32)

    oracle = _cpu_sliding_attention(q, k, v_f32, qpos, kpos, window=window, is_causal=False)
    oracle_round = _from_bf16_bits(_to_bf16_bits(oracle))

    bufs = []
    try:
        q_dev = _upload(_runtime, bufs, np.ascontiguousarray(q, dtype=np.float32))
        k_dev = _upload(_runtime, bufs, np.ascontiguousarray(k, dtype=np.float32))
        v_dev = _upload(_runtime, bufs, v_bf)
        qp_dev = _upload(_runtime, bufs, qpos)
        kp_dev = _upload(_runtime, bufs, kpos)
        out_dev = _upload(_runtime, bufs, np.zeros((b, ql, qh, hd), np.uint16))
        dflash2_sliding_attention_f32_bf16(
            q_dev.ptr, k_dev.ptr, v_dev.ptr, qp_dev.ptr, kp_dev.ptr, out_dev.ptr,
            b, ql, kv_len, qh, kvh, hd,
            sliding_window=window, is_causal=False, library=_dflash2_lib, runtime=_runtime,
        )
        _runtime.device_synchronize()
        got = _from_bf16_bits(_download(_runtime, out_dev, (b, ql, qh, hd), np.uint16))
        scale = max(float(np.max(np.abs(oracle_round))), 1.0)
        np.testing.assert_allclose(got, oracle_round, atol=2e-3 * scale, rtol=2e-3)
    finally:
        _free_all(_runtime, bufs)


# ---------------------------------------------------------------------------
# Candidate selector
# ---------------------------------------------------------------------------

def test_dflash2_selector_red(_dflash2_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.speculative.dflash2 import (
        dflash2_selector,
        dflash2_top16_rows,
    )

    rng = np.random.default_rng(0xDF2A4)
    rows, hidden_size, rank, top_k, vocab = 7, 5120, 256, 16, 4096
    hidden_f32 = (rng.standard_normal((rows, hidden_size), dtype=np.float32) * 0.5)
    hp_f32 = rng.standard_normal((rank, hidden_size), dtype=np.float32) * 0.05
    ca_f32 = rng.standard_normal((vocab, rank), dtype=np.float32) * 0.1
    cb_f32 = rng.standard_normal((vocab, rank), dtype=np.float32) * 0.1
    logits = rng.standard_normal((rows, vocab), dtype=np.float32)
    anchor = np.asarray([rng.integers(0, vocab)], dtype=np.int32)

    hidden_bf = _to_bf16_bits(hidden_f32)
    hp_bf = _to_bf16_bits(hp_f32)
    ca_bf = _to_bf16_bits(ca_f32)
    cb_bf = _to_bf16_bits(cb_f32)

    # Oracle (on bf16-rounded inputs for exact parity; rank-3 batch=1).
    oracle = candidate_selector_select(
        _from_bf16_bits(hidden_bf)[None],
        logits[None],
        anchor.astype(np.int64),
        _from_bf16_bits(ca_bf),
        _from_bf16_bits(cb_bf),
        _from_bf16_bits(hp_bf),
        top_k=top_k,
    )

    bufs = []
    try:
        hidden_dev = _upload(_runtime, bufs, hidden_bf)
        hp_dev = _upload(_runtime, bufs, hp_bf)
        ca_dev = _upload(_runtime, bufs, ca_bf)
        cb_dev = _upload(_runtime, bufs, cb_bf)
        anchor_dev = _upload(_runtime, bufs, anchor)
        logits_dev = _upload(_runtime, bufs, np.ascontiguousarray(logits, dtype=np.float32))
        ids_dev = _upload(_runtime, bufs, np.zeros((rows, top_k), np.int32))
        vals_dev = _upload(_runtime, bufs, np.zeros((rows, top_k), np.float32))
        dflash2_top16_rows(
            logits_dev.ptr, ids_dev.ptr, vals_dev.ptr, rows, vocab, top_k,
            library=_dflash2_lib, runtime=_runtime,
        )
        h_scratch = _upload(_runtime, bufs, np.zeros((rows, rank), np.float32))
        path_dev = _upload(_runtime, bufs, np.zeros((rows,), np.int32))
        scores_dev = _upload(_runtime, bufs, np.zeros((rows, top_k), np.float32))
        dflash2_selector(
            hidden_dev.ptr, hp_dev.ptr, ids_dev.ptr, vals_dev.ptr, anchor_dev.ptr,
            ca_dev.ptr, cb_dev.ptr, h_scratch.ptr, path_dev.ptr, scores_dev.ptr,
            rows, hidden_size, rank, top_k, vocab,
            library=_dflash2_lib, runtime=_runtime,
        )
        _runtime.device_synchronize()
        got_path = _download(_runtime, path_dev, (rows,), np.int32)
        got_scores = _download(_runtime, scores_dev, (rows, top_k), np.float32)
        # Greedy path must match exactly (deterministic argmax chain).
        np.testing.assert_array_equal(got_path, oracle.path[0])
        # Compare each position's best score (order-independent of the
        # candidate table ordering).
        for i in range(rows):
            oracle_best = float(np.max(oracle.scores[0, i]))
            got_best = float(np.max(got_scores[i]))
            assert got_best == pytest.approx(oracle_best, rel=1e-4, abs=1e-3)
    finally:
        _free_all(_runtime, bufs)
