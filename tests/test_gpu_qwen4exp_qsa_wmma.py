"""The WMMA QSA prefill attention matches a float64 reference.

Covers the three launches together the way the runner uses them: the paged
BF16 K/V gather to contiguous F16 rows, the selected-position lists folded
into four-token-block bitmasks, and the attention kernel in its dense causal
window and its masked sparse window (selected blocks plus each query's own
incomplete tail block).
"""

import ctypes

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa_wmma import (
    build_qwen4_exp_qsa_wmma,
    qsa_wmma_mask_words,
    qwen4_exp_qsa_wmma_attention_f32,
    qwen4_exp_qsa_wmma_gather_kv_f16,
    qwen4_exp_qsa_wmma_selection_mask,
)

HEADS, KV_HEADS, DIM, RATIO = 24, 2, 256, 4
Q_WIDTH = HEADS * DIM
KV_WIDTH = KV_HEADS * DIM
BLOCK_SIZE = 16


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)


def _to_bf16(arr: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _from_bf16(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _paged(linear: np.ndarray, table: np.ndarray) -> np.ndarray:
    """[positions, KV_WIDTH] -> paged cache rows ordered by physical block."""

    paged = np.zeros((table.size * BLOCK_SIZE, KV_WIDTH), dtype=np.uint16)
    for position in range(linear.shape[0]):
        physical = int(table[position // BLOCK_SIZE])
        paged[physical * BLOCK_SIZE + position % BLOCK_SIZE] = linear[position]
    return paged


def _reference(q, k, v, start, visible):
    """visible(row_position) -> boolean mask over key positions."""

    out = np.zeros((q.shape[0], HEADS, DIM), dtype=np.float64)
    q64 = q.reshape(q.shape[0], HEADS, DIM).astype(np.float64) / 16.0
    k64 = k.reshape(-1, KV_HEADS, DIM).astype(np.float64)
    v64 = v.reshape(-1, KV_HEADS, DIM).astype(np.float64)
    for row in range(q.shape[0]):
        keys = np.flatnonzero(visible(start + row))
        for head in range(HEADS):
            kv_head = head // (HEADS // KV_HEADS)
            scores = k64[keys, kv_head] @ q64[row, head]
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            out[row, head] = weights @ v64[keys, kv_head]
    return out.reshape(q.shape[0], Q_WIDTH)


def _run(q, k_bits, v_bits, start, rows, selection=None):
    runtime = get_hip_runtime()
    library = build_qwen4_exp_qsa_wmma(load=True)
    context = start + rows
    logical_blocks = (context + BLOCK_SIZE - 1) // BLOCK_SIZE
    table = np.random.default_rng(context).permutation(logical_blocks + 3)[
        :logical_blocks
    ].astype(np.int32)
    table_full = np.zeros(logical_blocks + 3, dtype=np.int32)
    table_full[:logical_blocks] = table
    paged_len = logical_blocks + 3
    k_paged = _paged(k_bits, np.concatenate([table, np.arange(3, dtype=np.int32)]))[
        : paged_len * BLOCK_SIZE
    ]
    v_paged = _paged(v_bits, np.concatenate([table, np.arange(3, dtype=np.int32)]))[
        : paged_len * BLOCK_SIZE
    ]
    sentinel = np.float32(-321.0)
    out_host = np.full((rows + 1, Q_WIDTH), sentinel, dtype=np.float32)
    buffers = []

    def upload(arr):
        host = np.ascontiguousarray(arr)
        dev = malloc(host.nbytes, runtime=runtime)
        buffers.append(dev)
        copy_host_to_device(dev, host_array_ptr(host), runtime=runtime)
        return dev

    try:
        q_dev = upload(q)
        k_dev, v_dev, table_dev = upload(k_paged), upload(v_paged), upload(table_full)
        k16 = upload(np.zeros((context, KV_WIDTH), dtype=np.uint16))
        v16 = upload(np.zeros((context, KV_WIDTH), dtype=np.uint16))
        out_dev = upload(out_host)
        device = Device("hip", 0)
        live = upload(np.asarray([context], dtype=np.int64))
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(
                table_dev.ptr, (1, table_full.size), DType.INT32, device
            ),
            live_counts=Tensor.from_handle(live.ptr, (1,), DType.INT64, device),
            max_live_count=context,
            storage_dtype=DType.BF16,
        )
        qwen4_exp_qsa_wmma_gather_kv_f16(
            k_dev.ptr, v_dev.ptr, spans, k16.ptr, v16.ptr,
            block_size=BLOCK_SIZE, first_position=0, context_len=context,
            library=library, runtime=runtime,
        )
        mask_ptr, words = 0, 0
        if selection is not None:
            positions, counts = selection
            words = qsa_wmma_mask_words(context)
            mask_dev = upload(np.zeros((rows, words), dtype=np.uint32))
            qwen4_exp_qsa_wmma_selection_mask(
                upload(positions).ptr, upload(counts).ptr, mask_dev.ptr,
                rows=rows, selected_stride=positions.shape[1], mask_words=words,
                library=library, runtime=runtime,
            )
            mask_ptr = mask_dev.ptr
        qwen4_exp_qsa_wmma_attention_f32(
            q_dev.ptr, k16.ptr, v16.ptr, out_dev.ptr,
            rows=rows, start_position=start, mask_ptr=mask_ptr, mask_words=words,
            library=library, runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_host), out_dev, runtime=runtime)
        k16_host = np.zeros((context, KV_WIDTH), dtype=np.float16)
        copy_device_to_host(host_array_ptr(k16_host), k16, runtime=runtime)
    finally:
        for dev in reversed(buffers):
            free(dev, runtime=runtime)
    assert (out_host[rows:] == sentinel).all()
    return out_host[:rows], k16_host


def _inputs(start, rows, seed):
    rng = np.random.default_rng(seed)
    context = start + rows
    q = (rng.standard_normal((rows, Q_WIDTH)) * 1.5).astype(np.float32)
    k_bits = _to_bf16(rng.standard_normal((context, KV_WIDTH)).astype(np.float32))
    v_bits = _to_bf16(rng.standard_normal((context, KV_WIDTH)).astype(np.float32))
    return q, k_bits, v_bits


def _check(actual, reference):
    assert np.isfinite(actual).all()
    error = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
    assert error <= 2e-2
    assert np.abs(actual - reference).max() <= 0.08 * np.abs(reference).max()


@pytest.mark.parametrize("start,rows", [(0, 50), (30, 20), (0, 16), (7, 1)])
def test_dense_window_matches_reference(start, rows) -> None:
    q, k_bits, v_bits = _inputs(start, rows, seed=start * 31 + rows)
    actual, k16 = _run(q, k_bits, v_bits, start, rows)
    k, v = _from_bf16(k_bits), _from_bf16(v_bits)
    # The gather is exact: every BF16 value here is representable in F16.
    assert np.array_equal(k16.astype(np.float32), k)
    context = start + rows
    reference = _reference(
        q, k, v, start, lambda position: np.arange(context) <= position
    )
    _check(actual, reference)


@pytest.mark.parametrize("start,rows,keep", [(100, 37, 0.3), (64, 8, 0.6), (257, 5, 0.1)])
def test_sparse_window_matches_reference(start, rows, keep) -> None:
    q, k_bits, v_bits = _inputs(start, rows, seed=start + rows)
    context = start + rows
    rng = np.random.default_rng(context)
    stride = context + 8
    positions = np.full((rows, stride), -1, dtype=np.int64)
    counts = np.zeros(rows, dtype=np.int32)
    visible_rows = []
    for row in range(rows):
        position = start + row
        tail_start = ((position + 1) // RATIO) * RATIO
        blocks = np.flatnonzero(rng.random(tail_start // RATIO) < keep)
        selected = np.concatenate(
            [(blocks[:, None] * RATIO + np.arange(RATIO)).ravel(),
             np.arange(tail_start, position + 1)]
        ).astype(np.int64)
        rng.shuffle(selected)
        positions[row, : selected.size] = selected
        counts[row] = selected.size
        mask = np.zeros(context, dtype=bool)
        mask[selected] = True
        visible_rows.append(mask)
    actual, _ = _run(q, k_bits, v_bits, start, rows, selection=(positions, counts))
    reference = _reference(
        q, _from_bf16(k_bits), _from_bf16(v_bits), start,
        lambda position: visible_rows[position - start],
    )
    _check(actual, reference)
