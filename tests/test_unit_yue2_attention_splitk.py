"""Split-K span attention gate: agreement with its strict parent kernel.

The production decode route reassociates the softmax (per-split maxima and sums,
combined afterwards), so it cannot be bit-identical and the exact span kernel stays
its parent. This gate pins the reassociation down: on the AR geometry the two kernels
must agree to fp32 rounding across contexts that exercise one split, several splits,
and a context that divides unevenly, plus the masked-token path, the empty-split path,
and an independent NumPy reference that neither kernel shares code with.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_array_to_device,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.vibevoice import attention_spans_splitk, encoder
from hipengine.kvcache.spans import KVLiveSpans

Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128


def _has_hip() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


requires_hip = pytest.mark.skipif(not _has_hip(), reason="ROCm/HIP runtime is not available")


def _upload(array: np.ndarray):
    host = np.ascontiguousarray(array)
    buffer = malloc(max(host.nbytes, 8))
    copy_host_array_to_device(buffer, host)
    return buffer


def _download(buffer, shape, dtype):
    out = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(out), buffer)
    return out


def _bf16_bits(values) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((raw + 0x7FFF + ((raw >> 16) & 1)) >> 16).astype(np.uint16)


def _widen(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _tensor(buffer, shape, dtype) -> Tensor:
    return Tensor.from_handle(buffer.ptr, shape, dtype, Device("hip"))


def _run(context: int, splits: int, *, evict=None, seed: int = 0):
    """One decode row through both kernels, plus the NumPy reference."""

    rng = np.random.default_rng(seed)
    capacity = context + 8
    query = rng.standard_normal((1, Q_HEADS, HEAD_DIM)).astype(np.float32)
    keys = _bf16_bits(rng.standard_normal((capacity, KV_HEADS, HEAD_DIM)))
    values = _bf16_bits(rng.standard_normal((capacity, KV_HEADS, HEAD_DIM)))
    scale = 1.0 / np.sqrt(HEAD_DIM)

    query_buf = _upload(query)
    key_buf = _upload(keys)
    value_buf = _upload(values)
    strict_buf = _upload(np.zeros((1, Q_HEADS, HEAD_DIM), np.float32))
    split_buf = _upload(np.zeros((1, Q_HEADS, HEAD_DIM), np.float32))
    workspace = malloc(attention_spans_splitk.workspace_floats(1, Q_HEADS, HEAD_DIM, splits) * 4)

    evicted = np.zeros(capacity, dtype=np.bool_)
    if evict:
        evicted[list(evict)] = True
    base = _upload(np.arange(capacity, dtype=np.int32))
    counts = _upload(np.array([context], dtype=np.int64))
    positions = _upload(np.arange(capacity, dtype=np.int64))
    evicted_buf = _upload(evicted)
    row_position = _upload(np.array([context - 1], dtype=np.int64))
    spans = KVLiveSpans(
        _tensor(base, (capacity,), "int32"),
        _tensor(counts, (1,), "int64"),
        capacity,
        _tensor(positions, (capacity,), "int64"),
        _tensor(evicted_buf, (capacity,), "bool"),
        "bf16",
        row_positions=_tensor(row_position, (1,), "int64"),
    )

    encoder.vv_attention_spans(
        query_buf.ptr, key_buf.ptr, value_buf.ptr, strict_buf.ptr, spans,
        1, Q_HEADS, KV_HEADS, HEAD_DIM, scale,
    )
    attention_spans_splitk.vv_attention_spans_splitk(
        query_buf.ptr, key_buf.ptr, value_buf.ptr, split_buf.ptr, workspace.ptr, spans,
        1, Q_HEADS, KV_HEADS, HEAD_DIM, scale, splits,
    )
    strict = _download(strict_buf, (1, Q_HEADS, HEAD_DIM), np.float32)
    split = _download(split_buf, (1, Q_HEADS, HEAD_DIM), np.float32)

    # Independent reference: dense causal softmax per query head over the live slots.
    live = [slot for slot in range(context) if not evicted[slot]]
    key_rows = _widen(keys)[live]
    value_rows = _widen(values)[live]
    reference = np.zeros_like(strict)
    group = Q_HEADS // KV_HEADS
    for head in range(Q_HEADS):
        kv_head = head // group
        scores = np.einsum("d,td->t", query[0, head], key_rows[:, kv_head, :]) * scale
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        reference[0, head] = weights @ value_rows[:, kv_head, :]
    return strict, split, reference


@pytest.mark.parametrize(
    "context,splits",
    [
        (1, 1),      # a single live token
        (16, 1),     # one split covers the whole context
        (32, 2),     # two splits, each a full chunk
        (100, 4),    # a context that divides unevenly (25 tokens a split)
        (128, 8),
        (256, 8),
        (997, 32),   # product depth with the split count the runtime picks
    ],
)
@requires_hip
def test_splitk_matches_the_strict_span_kernel(context, splits):
    strict, split, reference = _run(context, splits)
    assert np.max(np.abs(strict - reference)) < 1e-5, "the strict kernel moved"
    assert np.max(np.abs(split - reference)) < 1e-5, "the split-K kernel disagrees with the reference"
    assert np.max(np.abs(split - strict)) < 1e-5, (
        f"max abs difference {np.max(np.abs(split - strict))} on context={context} splits={splits}"
    )


@requires_hip
def test_splitk_honours_the_evict_mask():
    # Masking an interior slot must remove it from both routes' softmax, and both must
    # still match the reference that drops the same slot.
    strict, split, reference = _run(64, 4, evict=(3, 17, 40), seed=7)
    assert np.max(np.abs(split - reference)) < 1e-5
    assert np.max(np.abs(split - strict)) < 1e-5


@requires_hip
def test_splitk_is_deterministic():
    first = _run(256, 8, seed=11)[1]
    second = _run(256, 8, seed=11)[1]
    assert np.array_equal(first, second)


@requires_hip
def test_splitk_rejects_geometry_it_does_not_cover():
    assert attention_spans_splitk.supports(Q_HEADS, KV_HEADS, HEAD_DIM)
    assert not attention_spans_splitk.supports(Q_HEADS, KV_HEADS, 256)
    assert not attention_spans_splitk.supports(12, 8, HEAD_DIM)
    query = _upload(np.zeros((1, Q_HEADS, HEAD_DIM), np.float32))
    cache = _upload(np.zeros((8, KV_HEADS, HEAD_DIM), np.uint16))
    out = _upload(np.zeros((1, Q_HEADS, HEAD_DIM), np.float32))
    workspace = malloc(attention_spans_splitk.workspace_floats(1, Q_HEADS, HEAD_DIM, 4) * 4)
    base = _upload(np.arange(8, dtype=np.int32))
    counts = _upload(np.array([4], dtype=np.int64))
    positions = _upload(np.arange(8, dtype=np.int64))
    evicted = _upload(np.zeros(8, dtype=np.bool_))
    row_position = _upload(np.array([3], dtype=np.int64))
    spans = KVLiveSpans(
        _tensor(base, (8,), "int32"),
        _tensor(counts, (1,), "int64"),
        8,
        _tensor(positions, (8,), "int64"),
        _tensor(evicted, (8,), "bool"),
        "bf16",
        row_positions=_tensor(row_position, (1,), "int64"),
    )
    with pytest.raises(ValueError):
        attention_spans_splitk.vv_attention_spans_splitk(
            query.ptr, cache.ptr, cache.ptr, out.ptr, workspace.ptr, spans,
            1, Q_HEADS, KV_HEADS, 256, 0.1, 4,
        )
    with pytest.raises(ValueError):
        attention_spans_splitk.vv_attention_spans_splitk(
            query.ptr, cache.ptr, cache.ptr, out.ptr, workspace.ptr, spans,
            1, Q_HEADS, KV_HEADS, HEAD_DIM, 0.1, 0,
        )


def test_split_count_stays_inside_the_kernel_bound():
    assert attention_spans_splitk.split_count(0) == 1
    assert attention_spans_splitk.split_count(32) == 1
    assert attention_spans_splitk.split_count(1000) == 32
    assert attention_spans_splitk.split_count(16000) == attention_spans_splitk.MAX_SPLITS
