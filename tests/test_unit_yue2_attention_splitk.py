"""Split-K span attention gate: agreement with its strict parent kernel.

The production decode route reassociates the softmax (per-split maxima and sums,
combined afterwards), so it cannot be bit-identical and the exact span kernel stays
its parent. This gate pins the reassociation down: on the AR geometry the two kernels
must agree to fp32 rounding across contexts that exercise one split, several splits,
and a context that divides unevenly, plus the masked-token path, the empty-split path,
and an independent NumPy reference that neither kernel shares code with.

``plan_split`` is also checked directly, because the split count now depends on the
device's compute-unit count: the same kernel has to fill 40 CUs on gfx1151, 96 on
gfx1100, and stay inside its bounds on parts nobody has measured.
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


def _run(context: int, plan, *, evict=None, seed: int = 0):
    """One decode row through both kernels, plus the NumPy reference."""

    rng = np.random.default_rng(seed)
    # The span ABI caps max_live_count at 16000, so the longest case gets no slack.
    capacity = min(context + 8, 16000)
    query = rng.standard_normal((1, Q_HEADS, HEAD_DIM)).astype(np.float32)
    keys = _bf16_bits(rng.standard_normal((capacity, KV_HEADS, HEAD_DIM)))
    values = _bf16_bits(rng.standard_normal((capacity, KV_HEADS, HEAD_DIM)))
    scale = 1.0 / np.sqrt(HEAD_DIM)

    query_buf = _upload(query)
    key_buf = _upload(keys)
    value_buf = _upload(values)
    strict_buf = _upload(np.zeros((1, Q_HEADS, HEAD_DIM), np.float32))
    split_buf = _upload(np.zeros((1, Q_HEADS, HEAD_DIM), np.float32))
    workspace = malloc(
        attention_spans_splitk.workspace_floats(1, Q_HEADS, HEAD_DIM, plan.splits) * 4
    )

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
        1, Q_HEADS, KV_HEADS, HEAD_DIM, scale, context, plan=plan,
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


def _plan(context, splits, chunk=None):
    """An explicit plan, so the sweep covers split counts the policy would not pick."""

    chunk = chunk if chunk is not None else -(-context // splits)
    return attention_spans_splitk.SplitPlan(splits=splits, chunk=chunk)


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
        (997, 64),   # ...and the widest split count a 96-CU part plans for
        (16000, 64), # the longest context this route can serve
    ],
)
@requires_hip
def test_splitk_matches_the_strict_span_kernel(context, splits):
    strict, split, reference = _run(context, _plan(context, splits))
    assert np.max(np.abs(strict - reference)) < 1e-5, "the strict kernel moved"
    assert np.max(np.abs(split - reference)) < 1e-5, "the split-K kernel disagrees with the reference"
    assert np.max(np.abs(split - strict)) < 1e-5, (
        f"max abs difference {np.max(np.abs(split - strict))} on context={context} splits={splits}"
    )


@requires_hip
def test_splitk_honours_the_evict_mask():
    # Masking an interior slot must remove it from both routes' softmax, and both must
    # still match the reference that drops the same slot.
    strict, split, reference = _run(64, _plan(64, 4), evict=(3, 17, 40), seed=7)
    assert np.max(np.abs(split - reference)) < 1e-5
    assert np.max(np.abs(split - strict)) < 1e-5


@requires_hip
def test_splitk_is_deterministic():
    first = _run(256, _plan(256, 8), seed=11)[1]
    second = _run(256, _plan(256, 8), seed=11)[1]
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
    # A plan that does not cover the live context would leave its tail unattended, so
    # the wrapper refuses it instead of letting the kernel drop tokens.
    with pytest.raises(ValueError):
        attention_spans_splitk.vv_attention_spans_splitk(
            query.ptr, cache.ptr, cache.ptr, out.ptr, workspace.ptr, spans,
            1, Q_HEADS, KV_HEADS, HEAD_DIM, 0.1, 4, plan=_plan(4, 1, chunk=2),
        )
    # A chunk wider than the staged scores would overrun shared memory.
    with pytest.raises(ValueError):
        attention_spans_splitk.vv_attention_spans_splitk(
            query.ptr, cache.ptr, cache.ptr, out.ptr, workspace.ptr, spans,
            1, Q_HEADS, KV_HEADS, HEAD_DIM, 0.1, 4,
            plan=attention_spans_splitk.SplitPlan(
                splits=1, chunk=attention_spans_splitk.MAX_CHUNK + 1
            ),
        )


@pytest.mark.parametrize("compute_units", [40, 96, 128, 2, None])
def test_plan_split_stays_inside_the_kernel_bounds(compute_units):
    for context in (1, 2, 8, 31, 32, 97, 512, 998, 4096, 16000):
        for kv_heads in (1, 2, 4, 8):
            plan = attention_spans_splitk.plan_split(
                context, kv_heads, compute_units=compute_units
            )
            assert 1 <= plan.splits <= attention_spans_splitk.MAX_SPLITS
            assert 1 <= plan.chunk <= attention_spans_splitk.MAX_CHUNK
            assert plan.covers >= min(context, attention_spans_splitk.MAX_SPLITS
                                      * attention_spans_splitk.MAX_CHUNK)


def test_plan_split_widens_with_the_device_where_the_context_does_not_fill_it():
    """Device width decides the grid when chunking alone would under-fill the part."""

    # A context short enough that the chunk target plans fewer blocks than the device
    # holds: 200 tokens is 7 splits, 56 blocks, which under-fills 96 compute units.
    context, kv_heads = 200, KV_HEADS
    small = attention_spans_splitk.plan_split(context, kv_heads, compute_units=40)
    large = attention_spans_splitk.plan_split(context, kv_heads, compute_units=96)
    wider = attention_spans_splitk.plan_split(context, kv_heads, compute_units=256)
    fill = attention_spans_splitk.BLOCKS_PER_UNIT
    assert small.splits * kv_heads >= fill * 40
    assert large.splits > small.splits, "a 96-CU part must not plan a 40-CU grid"
    assert wider.splits >= large.splits
    for plan in (small, large, wider):
        assert plan.covers >= context
        assert plan.chunk <= attention_spans_splitk.MAX_CHUNK


def test_plan_split_lands_on_the_measured_best_at_product_depth():
    """At product depth both target parts plan the split count measured on the W7900.

    The chunk target already fills 40 and 96 compute units at 998 tokens, so the device
    bound does not move the answer: a kernel-only sweep on the 96-CU gfx1100 put the
    floor at 24-32 splits (0.025 ms per call) and 48 splits at 0.031 ms.
    """

    for compute_units in (40, 96):
        plan = attention_spans_splitk.plan_split(998, KV_HEADS, compute_units=compute_units)
        assert plan.splits == 32
        assert plan.chunk == 32
        assert plan.splits * KV_HEADS >= 256  # the grid the sweep measured as the floor


def test_plan_split_keeps_the_product_context_wide_enough():
    """At product depth the chunk bound, not the device, sets the split count."""

    for compute_units in (40, 96):
        plan = attention_spans_splitk.plan_split(16000, KV_HEADS, compute_units=compute_units)
        assert plan.splits == attention_spans_splitk.MAX_SPLITS
        assert plan.covers >= 16000


def test_plan_split_does_not_split_a_short_context_into_empty_blocks():
    """A tiny context on a wide part stays near one token per split, not below."""

    plan = attention_spans_splitk.plan_split(16, KV_HEADS, compute_units=96)
    assert plan.chunk >= attention_spans_splitk.MIN_CHUNK
    assert plan.splits * KV_HEADS >= 1
