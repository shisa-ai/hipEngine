"""Gemma 4 direct per-token/head INT8 decode consumer vs the CPU representation oracle.

The GPU consumer must match the independent CPU reference on the *same*
quantized representation: INT8 payload plus per-token/per-KV-head FP16 or FP32
scale, reconstructed as ``float32(int8) * float32(scale)``. The original
unquantized FP32 fixture rows are a diagnostic, never the binding gate.

Coverage: both Gemma 4 GQA geometries, FP16 and FP32 scales, zero and negative
payloads, distinct K/V, shuffled physical pages with sentinel slots, a context
ladder around the 1024 page boundaries plus an unrelated value and 4096, causal
and sliding-window masks, eviction and position semantics, all-masked rows, and
a repeated run into a poisoned output buffer.
"""

from __future__ import annotations

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
from hipengine.kernels.cpu_reference import (
    gemma4_attention_decode_int8_per_token_head,
    quantize_kv_int8_per_token_head,
)
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_int8
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata

_DEVICE = Device("hip", 0)
_GEOMETRIES = ((16, 8, 256), (16, 2, 512))
_SCALE_DTYPES = ("fp32", "fp16")
_NP_SCALE = {"fp32": np.float32, "fp16": np.float16}
_DTYPE_ENUM = {"fp32": DType.FP32, "fp16": DType.FP16}
_SENTINEL_I8 = np.int8(-128)
_SENTINEL_SCALE = 4096.0
_POISON_BITS = 0x7FC00000  # a quiet-NaN float32 bit pattern for output poisoning


def _poison(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, _POISON_BITS, dtype=np.uint32).view(np.float32)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _tensor(ptr: int, shape: tuple[int, ...], dtype: DType) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, _DEVICE)


def _upload(buffers: list, array: np.ndarray, *, runtime):
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes, runtime=runtime)
    buffers.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes, runtime=runtime)
    return buffer


def _download(buffer, array: np.ndarray, *, runtime) -> np.ndarray:
    copy_device_to_host(host_array_ptr(array), buffer, array.nbytes, runtime=runtime)
    return array


def _rows(num_kv_heads: int, head_dim: int, context: int, seed: int):
    """Distinct K/V rows with zero, negative, and wide-range values.

    Head 0 slot 0 is an all-zero row (zero scale, zero payload). Head 1 holds
    extrema and a round-half-to-even tie so a K/V swap changes payload and scale.
    """

    rng = np.random.default_rng(seed)
    key = rng.uniform(-6.0, 6.0, size=(context, num_kv_heads, head_dim)).astype(np.float32)
    value = rng.uniform(-6.0, 6.0, size=(context, num_kv_heads, head_dim)).astype(np.float32)
    key[0, 0] = 0.0
    value[0, 0] = 0.0
    key[-1, min(1, num_kv_heads - 1), 0] = 127.0
    key[-1, min(1, num_kv_heads - 1), 1] = -127.0
    key[-1, min(1, num_kv_heads - 1), 2] = 63.5
    value[-1, min(1, num_kv_heads - 1), 0] = -254.0
    value[-1, min(1, num_kv_heads - 1), 1] = 254.0
    value[-1, min(1, num_kv_heads - 1), 2] = 125.0
    return key, value


def _scatter(key, value, block_table, *, block_size, scale_dtype, cache_blocks):
    context, num_kv_heads, head_dim = key.shape
    qk, qv, ks, vs = quantize_kv_int8_per_token_head(
        key, value, scale_dtype=_NP_SCALE[scale_dtype]
    )
    key_cache = np.full((cache_blocks, block_size, num_kv_heads, head_dim), _SENTINEL_I8, np.int8)
    value_cache = np.full_like(key_cache, _SENTINEL_I8)
    np_dtype = _NP_SCALE[scale_dtype]
    k_scale = np.full(
        (cache_blocks, block_size, num_kv_heads), np.asarray(_SENTINEL_SCALE, np_dtype), np_dtype
    )
    v_scale = np.full_like(k_scale, np.asarray(_SENTINEL_SCALE, np_dtype))
    for slot in range(context):
        logical_block, offset = divmod(slot, block_size)
        physical = int(block_table[logical_block])
        key_cache[physical, offset] = qk[slot]
        value_cache[physical, offset] = qv[slot]
        k_scale[physical, offset] = ks[slot].astype(np_dtype)
        v_scale[physical, offset] = vs[slot].astype(np_dtype)
    return key_cache, value_cache, k_scale, v_scale, (qk, qv, ks, vs)


def _float_attention(query, key, value, context, *, token_positions=None, evict_mask=None,
                     row_position=None, sliding_window=None, scale=1.0):
    """Unquantized diagnostic attention over the original float rows."""

    q = query.astype(np.float64)
    num_q_heads, head_dim = q.shape
    num_kv_heads = key.shape[1]
    kv_group = num_q_heads // num_kv_heads
    query_position = context - 1 if row_position is None else int(row_position)
    out = np.empty_like(q)
    for head in range(num_q_heads):
        kv_head = head // kv_group
        logits = []
        vectors = []
        for slot in range(context):
            pos = slot if token_positions is None else int(token_positions[slot])
            visible = 0 <= pos <= query_position
            if sliding_window is not None:
                visible = visible and pos > query_position - sliding_window
            if evict_mask is not None:
                visible = visible and not bool(evict_mask[slot])
            logits.append(float(key[slot, kv_head] @ q[head]) * scale if visible else -np.inf)
            vectors.append(value[slot, kv_head].astype(np.float64))
        logits = np.asarray(logits)
        if not np.isfinite(logits).any():
            out[head] = 0.0
            continue
        m = np.max(logits)
        w = np.exp(logits - m)
        w = w / w.sum()
        out[head] = sum(w[i] * vectors[i] for i in range(context))
    return out


def _run_case(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    context: int,
    block_size: int,
    scale_dtype: str,
    seed: int,
    token_positions: np.ndarray | None = None,
    evict_mask: np.ndarray | None = None,
    row_position: int | None = None,
    sliding_window: int | None = None,
    max_context_len: int | None = None,
    runs: int = 1,
    poison_masked_scales: bool = False,
    library,
):
    runtime = get_hip_runtime()
    key, value = _rows(num_kv_heads, head_dim, context, seed)
    blocks_needed = (context + block_size - 1) // block_size
    # Shuffled physical pages with a couple of untouched sentinel blocks.
    physical = list(range(blocks_needed + 2))[::-1]
    block_table = np.asarray(physical, dtype=np.int32)
    key_cache, value_cache, k_scale, v_scale, _ = _scatter(
        key, value, block_table, block_size=block_size,
        scale_dtype=scale_dtype, cache_blocks=len(physical),
    )
    assert not np.array_equal(key_cache, value_cache), "fixture K/V payloads must differ"
    assert not np.array_equal(k_scale, v_scale), "fixture K/V scales must differ"
    if poison_masked_scales:
        assert evict_mask is not None, "poison_masked_scales needs an eviction mask"
        for slot in range(context):
            if bool(evict_mask[slot]):
                logical_block, offset = divmod(slot, block_size)
                phys = int(block_table[logical_block])
                k_scale[phys, offset, :] = np.inf
                v_scale[phys, offset, :] = np.nan
    rng = np.random.default_rng(seed + 7)
    query = rng.uniform(-2.0, 2.0, size=(num_q_heads, head_dim)).astype(np.float32)
    bound = context if max_context_len is None else int(max_context_len)

    expected = gemma4_attention_decode_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, context,
        block_size=block_size, token_positions=token_positions, evict_mask=evict_mask,
        row_position=row_position, sliding_window=sliding_window,
    )

    buffers: list = []
    try:
        query_b = _upload(buffers, query, runtime=runtime)
        key_b = _upload(buffers, key_cache, runtime=runtime)
        value_b = _upload(buffers, value_cache, runtime=runtime)
        k_scale_b = _upload(buffers, k_scale, runtime=runtime)
        v_scale_b = _upload(buffers, v_scale, runtime=runtime)
        table_b = _upload(buffers, block_table, runtime=runtime)
        live_b = _upload(buffers, np.asarray([context], dtype=np.int64), runtime=runtime)
        pos_b = None
        if token_positions is not None:
            pos_b = _upload(buffers, token_positions.astype(np.int64), runtime=runtime)
        evict_b = None
        if evict_mask is not None:
            evict_b = _upload(buffers, evict_mask.astype(np.bool_), runtime=runtime)
        row_b = None
        if row_position is not None:
            row_b = _upload(buffers, np.asarray([row_position], dtype=np.int64), runtime=runtime)
        poison = _poison((num_q_heads, head_dim))
        out_host = poison.copy()
        out_b = _upload(buffers, out_host, runtime=runtime)

        metadata = KVScaleMetadata(
            k_scale=_tensor(k_scale_b.ptr, k_scale.shape, _DTYPE_ENUM[scale_dtype]),
            v_scale=_tensor(v_scale_b.ptr, v_scale.shape, _DTYPE_ENUM[scale_dtype]),
            scale_dtype=_DTYPE_ENUM[scale_dtype],
        )
        spans = KVLiveSpans(
            base_offsets=_tensor(table_b.ptr, block_table.shape, DType.INT32),
            live_counts=_tensor(live_b.ptr, (1,), DType.INT64),
            max_live_count=bound,
            token_positions=None if pos_b is None else _tensor(pos_b.ptr, (bound,), DType.INT64),
            evict_mask=None if evict_b is None else _tensor(evict_b.ptr, (bound,), DType.BOOL),
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            spans_mode="uniform",
            row_positions=None if row_b is None else _tensor(row_b.ptr, (1,), DType.INT64),
            scale_metadata=metadata,
        )

        outputs: list[np.ndarray] = []
        for _ in range(runs):
            # Re-poison the output so a second run must rewrite every element.
            copy_host_to_device(
                out_b, host_array_ptr(poison), poison.nbytes, runtime=runtime
            )
            gemma4_attention_int8.gemma4_attention_decode_int8_per_token_head_spans(
                query_b.ptr, key_b.ptr, value_b.ptr, out_b.ptr, spans, bound, block_size,
                num_q_heads, num_kv_heads, head_dim, 1.0, sliding_window,
                library=library, runtime=runtime,
            )
            runtime.device_synchronize()
            _download(out_b, out_host, runtime=runtime)
            outputs.append(out_host.copy())
            assert not np.any(np.isnan(out_host)), "output not fully written"
            # Same representation, independent implementation: agreement is
            # FP32 exp/reduction-order level, not bit-exact (expf vs NumPy,
            # warp tree vs BLAS dot, head_dim-long accumulation).
            np.testing.assert_allclose(out_host, expected, rtol=1e-3, atol=3e-4)
        # Repeated launches must be bit-identical to each other (byte view, so
        # signed zeros and NaN payloads cannot hide behind a float compare).
        for other in outputs[1:]:
            np.testing.assert_array_equal(
                other.view(np.uint8), outputs[0].view(np.uint8)
            )
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)

    diagnostic = _float_attention(
        query, key, value, context, token_positions=token_positions,
        evict_mask=evict_mask, row_position=row_position, sliding_window=sliding_window,
    )
    return expected, diagnostic, outputs[-1]


@pytest.fixture(scope="module")
def _library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    return gemma4_attention_int8.build_gemma4_attention_int8(load=True)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim", _GEOMETRIES, ids=("h8-d256", "h2-d512")
)
@pytest.mark.parametrize("context", [1, 137, 256, 257, 1023, 1024, 1025, 4096])
def test_gemma4_int8_decode_matches_oracle_over_context_ladder(
    _library, num_q_heads, num_kv_heads, head_dim, scale_dtype, context
) -> None:
    expected, diagnostic, _ = _run_case(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        context=context,
        block_size=256,
        scale_dtype=scale_dtype,
        seed=0xD12 + context + num_kv_heads,
        library=_library,
    )
    # Diagnostic only: the quantized consumer must track the unquantized rows
    # within INT8 per-token/head error, not exactly. Correlation is the stable
    # signal; the absolute gap grows where a quantized logit shift moves softmax
    # mass between near-tied keys.
    assert np.all(np.isfinite(diagnostic))
    corr = float(np.corrcoef(expected.ravel(), diagnostic.ravel())[0, 1])
    assert corr > 0.95, f"quantized/unquantized correlation {corr} too low"


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim", _GEOMETRIES, ids=("h8-d256", "h2-d512")
)
def test_gemma4_int8_decode_positions_evictions_and_window_match_oracle(
    _library, num_q_heads, num_kv_heads, head_dim, scale_dtype
) -> None:
    context = 600
    bound = 700  # larger than the live count: unused scratch must not be read.
    rng = np.random.default_rng(0x5EED + num_kv_heads)
    # Non-monotonic absolute positions and a partial eviction mask.
    positions = np.full(bound, -1, dtype=np.int64)
    positions[:context] = rng.permutation(np.arange(context, dtype=np.int64))
    evict = np.zeros(bound, dtype=bool)
    evict[:context] = rng.random(context) < 0.25
    _run_case(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        context=context,
        block_size=256,
        scale_dtype=scale_dtype,
        seed=0x5EED + num_kv_heads + head_dim,
        token_positions=positions,
        evict_mask=evict,
        row_position=context - 1,
        sliding_window=128,
        max_context_len=bound,
        library=_library,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_decode_all_masked_returns_zeros(_library, scale_dtype) -> None:
    for num_q_heads, num_kv_heads, head_dim in _GEOMETRIES:
        context = 300
        expected, _, _ = _run_case(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            context=context,
            block_size=256,
            scale_dtype=scale_dtype,
            seed=0xA11 + num_kv_heads,
            evict_mask=np.ones(context, dtype=bool),
            library=_library,
        )
        np.testing.assert_array_equal(expected, np.zeros_like(expected))
        expected, _, _ = _run_case(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            context=context,
            block_size=256,
            scale_dtype=scale_dtype,
            seed=0xA12 + num_kv_heads,
            row_position=-1,
            library=_library,
        )
        np.testing.assert_array_equal(expected, np.zeros_like(expected))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_decode_repeated_poisoned_output_is_deterministic(
    _library, scale_dtype
) -> None:
    for num_q_heads, num_kv_heads, head_dim in _GEOMETRIES:
        _run_case(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            context=1025,
            block_size=256,
            scale_dtype=scale_dtype,
            seed=0xDEAD + num_kv_heads,
            runs=2,
            library=_library,
        )


def _run_zero_context_case(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale_dtype: str,
    live_count: int,
    library,
    runs: int = 2,
):
    """Run one launch with a non-positive ``live_counts[0]`` into a poisoned output.

    Returns the downloaded output plus the fixture arrays so the caller can build
    the CPU reference for the supported empty case.
    """

    runtime = get_hip_runtime()
    block_size = 1
    blocks = 1
    key_cache = np.full((blocks, block_size, num_kv_heads, head_dim), _SENTINEL_I8, np.int8)
    value_cache = np.full_like(key_cache, _SENTINEL_I8)
    np_dtype = _NP_SCALE[scale_dtype]
    k_scale = np.full(
        (blocks, block_size, num_kv_heads), np.asarray(_SENTINEL_SCALE, np_dtype), np_dtype
    )
    v_scale = np.full_like(k_scale, np.asarray(_SENTINEL_SCALE, np_dtype))
    block_table = np.asarray([0], dtype=np.int32)
    rng = np.random.default_rng(0xE0 + num_kv_heads)
    query = rng.uniform(-2.0, 2.0, size=(num_q_heads, head_dim)).astype(np.float32)
    bound = 1

    buffers: list = []
    try:
        query_b = _upload(buffers, query, runtime=runtime)
        key_b = _upload(buffers, key_cache, runtime=runtime)
        value_b = _upload(buffers, value_cache, runtime=runtime)
        k_scale_b = _upload(buffers, k_scale, runtime=runtime)
        v_scale_b = _upload(buffers, v_scale, runtime=runtime)
        table_b = _upload(buffers, block_table, runtime=runtime)
        live_b = _upload(buffers, np.asarray([live_count], dtype=np.int64), runtime=runtime)
        poison = _poison((num_q_heads, head_dim))
        out_host = poison.copy()
        out_b = _upload(buffers, out_host, runtime=runtime)
        metadata = KVScaleMetadata(
            k_scale=_tensor(k_scale_b.ptr, k_scale.shape, _DTYPE_ENUM[scale_dtype]),
            v_scale=_tensor(v_scale_b.ptr, v_scale.shape, _DTYPE_ENUM[scale_dtype]),
            scale_dtype=_DTYPE_ENUM[scale_dtype],
        )
        spans = KVLiveSpans(
            base_offsets=_tensor(table_b.ptr, block_table.shape, DType.INT32),
            live_counts=_tensor(live_b.ptr, (1,), DType.INT64),
            max_live_count=bound,
            token_positions=None,
            evict_mask=None,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            spans_mode="uniform",
            scale_metadata=metadata,
        )
        for _ in range(runs):
            copy_host_to_device(out_b, host_array_ptr(poison), poison.nbytes, runtime=runtime)
            gemma4_attention_int8.gemma4_attention_decode_int8_per_token_head_spans(
                query_b.ptr, key_b.ptr, value_b.ptr, out_b.ptr, spans, bound, block_size,
                num_q_heads, num_kv_heads, head_dim, 1.0, None,
                library=library, runtime=runtime,
            )
            runtime.device_synchronize()
            _download(out_b, out_host, runtime=runtime)
            assert not np.any(np.isnan(out_host)), "non-positive-context output not written"
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)
    return out_host, query, key_cache, value_cache, k_scale, v_scale, block_table


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim", _GEOMETRIES, ids=("h8-d256", "h2-d512")
)
def test_gemma4_int8_decode_empty_span_matches_cpu_reference(
    _library, num_q_heads, num_kv_heads, head_dim, scale_dtype
) -> None:
    """An empty span is supported and must return the CPU reference's zeros.

    The output buffer is poisoned before each of two runs, so a stale/partial
    write is caught rather than hidden by a previous run.
    """

    out, query, key_cache, value_cache, k_scale, v_scale, block_table = _run_zero_context_case(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale_dtype=scale_dtype,
        live_count=0,
        library=_library,
    )
    expected = gemma4_attention_decode_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, 0, block_size=1
    )
    np.testing.assert_array_equal(expected, np.zeros_like(expected))
    np.testing.assert_array_equal(out, expected)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim", _GEOMETRIES, ids=("h8-d256", "h2-d512")
)
def test_gemma4_int8_decode_negative_live_count_is_refused(
    _library, num_q_heads, num_kv_heads, head_dim, scale_dtype
) -> None:
    """A malformed negative count is refused, distinct from the supported empty span.

    The checked path reports it before launch; the kernel repeats the guard and
    writes a NaN row, so a direct caller is never left with a success value.
    """

    with pytest.raises(ValueError, match="is negative"):
        _checked_launch(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale_dtype=scale_dtype,
            live_count=-1,
            block_table=np.asarray([0], dtype=np.int32),
            cache_blocks=2,
            block_size=4,
            max_context_len=4,
            library=_library,
        )
    # The CPU reference refuses the same malformed count.
    with pytest.raises(ValueError, match="must not be negative"):
        gemma4_attention_decode_int8_per_token_head(
            np.zeros((num_q_heads, head_dim), np.float32),
            np.zeros((1, 1, num_kv_heads, head_dim), np.int8),
            np.zeros((1, 1, num_kv_heads, head_dim), np.int8),
            np.ones((1, 1, num_kv_heads), np.float32),
            np.ones((1, 1, num_kv_heads), np.float32),
            np.asarray([0], dtype=np.int32),
            -1,
            block_size=1,
        )


def _checked_launch(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale_dtype: str,
    live_count: int,
    block_table: np.ndarray,
    cache_blocks: int,
    block_size: int,
    max_context_len: int,
    library,
) -> None:
    """Upload real metadata and call the wrapper; the checked path must raise.

    Validation reads the metadata prefix and raises before any launch, so a
    deliberately invalid page id is never dereferenced by the kernel -- a safe
    checked-path test rather than a live out-of-bounds reproduction.
    """

    runtime = get_hip_runtime()
    key_cache = np.full(
        (cache_blocks, block_size, num_kv_heads, head_dim), _SENTINEL_I8, np.int8
    )
    value_cache = np.full_like(key_cache, _SENTINEL_I8)
    np_dtype = _NP_SCALE[scale_dtype]
    k_scale = np.full(
        (cache_blocks, block_size, num_kv_heads),
        np.asarray(_SENTINEL_SCALE, np_dtype),
        np_dtype,
    )
    v_scale = np.full_like(k_scale, np.asarray(_SENTINEL_SCALE, np_dtype))
    query = np.zeros((num_q_heads, head_dim), np.float32)
    out = np.zeros((num_q_heads, head_dim), np.float32)
    buffers: list = []
    try:
        query_b = _upload(buffers, query, runtime=runtime)
        key_b = _upload(buffers, key_cache, runtime=runtime)
        value_b = _upload(buffers, value_cache, runtime=runtime)
        k_scale_b = _upload(buffers, k_scale, runtime=runtime)
        v_scale_b = _upload(buffers, v_scale, runtime=runtime)
        table_b = _upload(buffers, block_table.astype(np.int32), runtime=runtime)
        live_b = _upload(buffers, np.asarray([live_count], dtype=np.int64), runtime=runtime)
        out_b = _upload(buffers, out, runtime=runtime)
        metadata = KVScaleMetadata(
            k_scale=_tensor(k_scale_b.ptr, k_scale.shape, _DTYPE_ENUM[scale_dtype]),
            v_scale=_tensor(v_scale_b.ptr, v_scale.shape, _DTYPE_ENUM[scale_dtype]),
            scale_dtype=_DTYPE_ENUM[scale_dtype],
        )
        spans = KVLiveSpans(
            base_offsets=_tensor(table_b.ptr, block_table.shape, DType.INT32),
            live_counts=_tensor(live_b.ptr, (1,), DType.INT64),
            max_live_count=max_context_len,
            token_positions=None,
            evict_mask=None,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            spans_mode="uniform",
            scale_metadata=metadata,
        )
        gemma4_attention_int8.gemma4_attention_decode_int8_per_token_head_spans(
            query_b.ptr, key_b.ptr, value_b.ptr, out_b.ptr, spans, max_context_len, block_size,
            num_q_heads, num_kv_heads, head_dim, 1.0, None, library=library, runtime=runtime,
        )
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_decode_over_capacity_live_count_is_refused(_library, scale_dtype) -> None:
    with pytest.raises(ValueError, match="exceeds max_context_len"):
        _checked_launch(
            num_q_heads=16, num_kv_heads=8, head_dim=256, scale_dtype=scale_dtype,
            live_count=5, block_table=np.asarray([0], dtype=np.int32), cache_blocks=2,
            block_size=4, max_context_len=4, library=_library,
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_decode_out_of_range_physical_page_is_refused(_library, scale_dtype) -> None:
    # cache_blocks == 2, so physical id 2 (and -1) are out of range.
    with pytest.raises(ValueError, match="outside"):
        _checked_launch(
            num_q_heads=16, num_kv_heads=8, head_dim=256, scale_dtype=scale_dtype,
            live_count=4, block_table=np.asarray([2], dtype=np.int32), cache_blocks=2,
            block_size=4, max_context_len=4, library=_library,
        )
    with pytest.raises(ValueError, match="outside"):
        _checked_launch(
            num_q_heads=16, num_kv_heads=8, head_dim=256, scale_dtype=scale_dtype,
            live_count=4, block_table=np.asarray([-1], dtype=np.int32), cache_blocks=2,
            block_size=4, max_context_len=4, library=_library,
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim", _GEOMETRIES, ids=("h8-d256", "h2-d512")
)
def test_gemma4_int8_decode_mixed_token_positions_match_oracle(
    _library, num_q_heads, num_kv_heads, head_dim, scale_dtype
) -> None:
    context = 600
    bound = 640
    positions = np.full(bound, -1, dtype=np.int64)
    positions[:200] = np.arange(200)  # visible
    positions[200:400] = np.arange(200, 400) + 1000  # future
    # Slots 400..599 stay negative (excluded).
    _run_case(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        context=context,
        block_size=256,
        scale_dtype=scale_dtype,
        seed=0xC0FFEE + num_kv_heads,
        token_positions=positions,
        row_position=199,
        max_context_len=bound,
        library=_library,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_decode_nonpositive_window_is_global(_library, scale_dtype) -> None:
    context = 600
    common = dict(
        num_q_heads=16, num_kv_heads=8, head_dim=256, context=context,
        block_size=256, scale_dtype=scale_dtype, seed=0x1DD0,
        library=_library,
    )
    base = _run_case(**common, sliding_window=None)[2]
    zero = _run_case(**common, sliding_window=0)[2]
    negative = _run_case(**common, sliding_window=-5)[2]
    np.testing.assert_array_equal(zero.view(np.uint8), base.view(np.uint8))
    np.testing.assert_array_equal(negative.view(np.uint8), base.view(np.uint8))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_decode_masked_poisoned_scales_do_not_leak(_library, scale_dtype) -> None:
    context = 600
    evict = np.zeros(context, dtype=bool)
    evict[::3] = True
    # The runner asserts the output is finite and matches the oracle, which
    # skips masked slots; the poisoned masked K/V scales must not leak.
    _run_case(
        num_q_heads=16, num_kv_heads=8, head_dim=256, context=context,
        block_size=256, scale_dtype=scale_dtype, seed=0xA01,
        evict_mask=evict, poison_masked_scales=True, library=_library,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_decode_visible_nonfinite_k_and_v_is_nan_with_canaries(
    _library, scale_dtype
) -> None:
    """A visible non-finite K or V must write NaN, proven against a finite sentinel.

    The output starts finite (sentinel 7.0) with canaries on both sides, so an
    all-NaN body is a real write and the canaries prove nothing escaped the row.
    """

    for poison in (np.inf, np.nan):
        # Visible K poison -> non-finite logit.
        query, kc, vc, ks, vs, table = _controlled_case(scale_dtype)
        ks[0, 0, :] = poison
        body, canary_ok = _run_direct(
            query=query, key_cache=kc, value_cache=vc, k_scale=ks, v_scale=vs,
            block_table=table, live_count=4, capacity=4, block_size=4,
            scale_dtype=scale_dtype, library=_library,
        )
        assert canary_ok, "K-poison write escaped the output row"
        assert np.all(np.isnan(body)), poison
        # Visible V poison -> NaN independent of the softmax weight.
        for underflow in (False, True):
            query, kc, vc, ks, vs, table = _controlled_case(scale_dtype, underflow=underflow)
            vs[0, 0, :] = poison
            body, canary_ok = _run_direct(
                query=query, key_cache=kc, value_cache=vc, k_scale=ks, v_scale=vs,
                block_table=table, live_count=4, capacity=4, block_size=4,
                scale_dtype=scale_dtype, library=_library,
            )
            assert canary_ok, "V-poison write escaped the output row"
            assert np.all(np.isnan(body)), (poison, underflow)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_decode_nonfinite_v_failure_is_head_local(_library, scale_dtype) -> None:
    num_q_heads, num_kv_heads, head_dim = 16, 8, 256
    block_size, capacity, cache_blocks = 4, 4, 1
    query = np.ones((num_q_heads, head_dim), np.float32)
    key_cache, value_cache, k_scale, v_scale = _controlled_cache(
        num_kv_heads=num_kv_heads, head_dim=head_dim, cache_blocks=cache_blocks,
        block_size=block_size,
    )
    value_cache[:] = 1
    v_scale[0, :, 0] = np.inf  # only KV head 0 is poisoned
    table = np.asarray([0], dtype=np.int32)
    body, canary_ok = _run_direct(
        query=query, key_cache=key_cache, value_cache=value_cache, k_scale=k_scale,
        v_scale=v_scale, block_table=table, live_count=capacity, capacity=capacity,
        block_size=block_size, scale_dtype=scale_dtype, library=_library,
    )
    assert canary_ok
    group = num_q_heads // num_kv_heads
    assert np.all(np.isnan(body[:group])), "query heads 0..group-1 map to KV head 0"
    assert np.all(np.isfinite(body[group:])), "unaffected heads must stay finite"


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_decode_direct_symbol_guards_are_bounded(_library, scale_dtype) -> None:
    """Exercise the raw kernel guards with allocated metadata and a bounded cache.

    The wrapper refuses these before launch, so this reaches the kernel's own
    guards directly: every invalid count/page must write NaN, an empty count
    zeros, and the output canaries check bounded writes. Read safety is enforced
    by the explicit physical bounds guard before cache dereference.
    """

    num_q_heads, num_kv_heads, head_dim = 16, 8, 256
    block_size, capacity, cache_blocks = 4, 4, 2
    query = np.ones((num_q_heads, head_dim), np.float32)
    key_cache, value_cache, k_scale, v_scale = _controlled_cache(
        num_kv_heads=num_kv_heads, head_dim=head_dim, cache_blocks=cache_blocks,
        block_size=block_size,
    )
    value_cache[:] = 1

    def _run(live_count: int, table: np.ndarray, **extra):
        return _run_direct(
            query=query, key_cache=key_cache, value_cache=value_cache, k_scale=k_scale,
            v_scale=v_scale, block_table=table, live_count=live_count, capacity=capacity,
            block_size=block_size, scale_dtype=scale_dtype, library=_library, **extra,
        )

    body, canary_ok = _run(0, np.asarray([0], dtype=np.int32))
    assert canary_ok and np.all(body == 0.0), "empty span must zero its row"

    body, canary_ok = _run(-1, np.asarray([0], dtype=np.int32))
    assert canary_ok and np.all(np.isnan(body)), "negative count must NaN"

    body, canary_ok = _run(capacity + 1, np.asarray([0], dtype=np.int32))
    assert canary_ok and np.all(np.isnan(body)), "over-capacity count must NaN"

    body, canary_ok = _run(capacity, np.asarray([cache_blocks], dtype=np.int32))
    assert canary_ok and np.all(np.isnan(body)), "out-of-range visible page must NaN"

    body, canary_ok = _run(capacity, np.asarray([-1], dtype=np.int32))
    assert canary_ok and np.all(np.isnan(body)), "negative visible page must NaN"


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_decode_direct_symbol_masked_invalid_page_is_not_failed(
    _library, scale_dtype
) -> None:
    """A masked out-of-range page is never dereferenced, so it is not a failure."""

    num_q_heads, num_kv_heads, head_dim = 16, 8, 256
    block_size, capacity, cache_blocks = 4, 4, 2
    query = np.ones((num_q_heads, head_dim), np.float32)
    key_cache, value_cache, k_scale, v_scale = _controlled_cache(
        num_kv_heads=num_kv_heads, head_dim=head_dim, cache_blocks=cache_blocks,
        block_size=block_size,
    )
    value_cache[:] = 1
    body, canary_ok = _run_direct(
        query=query, key_cache=key_cache, value_cache=value_cache, k_scale=k_scale,
        v_scale=v_scale, block_table=np.asarray([cache_blocks], dtype=np.int32),
        live_count=capacity, capacity=capacity, block_size=block_size,
        scale_dtype=scale_dtype, evict_mask=np.ones(capacity, dtype=bool),
        library=_library,
    )
    assert canary_ok
    assert np.all(body == 0.0), "all-masked row is zeros, not a failure"


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gemma4_int8_decode_readback_orders_against_nonblocking_stream(_library) -> None:
    """A non-blocking producer stream must be ordered before the metadata readback.

    The live count buffer is seeded with an over-capacity (invalid) value, then a
    valid count is enqueued on a non-blocking stream. The checked path must
    synchronize that stream before its synchronous D2H read, so the readback sees
    the valid count; an unordered read would raise ``ValueError``.
    """

    from hipengine.core.memory import enqueue_host_to_device

    runtime = get_hip_runtime()
    stream = runtime.stream_create(nonblocking=True)
    try:
        num_q_heads, num_kv_heads, head_dim = 16, 8, 256
        block_size, capacity, cache_blocks = 4, 4, 2
        query = np.ones((num_q_heads, head_dim), np.float32)
        key_cache, value_cache, k_scale, v_scale = _controlled_cache(
            num_kv_heads=num_kv_heads, head_dim=head_dim, cache_blocks=cache_blocks,
            block_size=block_size,
        )
        value_cache[:] = 1
        block_table = np.asarray([0], dtype=np.int32)
        buffers: list = []
        try:
            query_b = _upload(buffers, query, runtime=runtime)
            key_b = _upload(buffers, key_cache, runtime=runtime)
            value_b = _upload(buffers, value_cache, runtime=runtime)
            k_scale_b = _upload(buffers, k_scale, runtime=runtime)
            v_scale_b = _upload(buffers, v_scale, runtime=runtime)
            table_b = _upload(buffers, block_table, runtime=runtime)
            live_b = _upload(buffers, np.asarray([capacity + 1], dtype=np.int64), runtime=runtime)
            valid = np.asarray([capacity], dtype=np.int64)
            enqueue_host_to_device(
                live_b, host_array_ptr(valid), valid.nbytes, stream=stream, runtime=runtime
            )
            out_host = np.zeros((num_q_heads, head_dim), np.float32)
            out_b = _upload(buffers, out_host, runtime=runtime)
            metadata = KVScaleMetadata(
                k_scale=_tensor(k_scale_b.ptr, k_scale.shape, DType.FP32),
                v_scale=_tensor(v_scale_b.ptr, v_scale.shape, DType.FP32),
                scale_dtype=DType.FP32,
            )
            spans = KVLiveSpans(
                base_offsets=_tensor(table_b.ptr, block_table.shape, DType.INT32),
                live_counts=_tensor(live_b.ptr, (1,), DType.INT64),
                max_live_count=capacity,
                token_positions=None,
                evict_mask=None,
                storage_dtype=DType.INT8_PER_TOKEN_HEAD,
                spans_mode="uniform",
                scale_metadata=metadata,
            )
            gemma4_attention_int8.gemma4_attention_decode_int8_per_token_head_spans(
                query_b.ptr, key_b.ptr, value_b.ptr, out_b.ptr, spans, capacity, block_size,
                num_q_heads, num_kv_heads, head_dim, 1.0, None,
                stream=stream, library=_library, runtime=runtime,
            )
            runtime.device_synchronize()
            _download(out_b, out_host, runtime=runtime)
            assert not np.any(np.isnan(out_host)), "ordered readback must launch cleanly"
        finally:
            for buffer in buffers:
                free(buffer, runtime=runtime)
    finally:
        runtime.stream_destroy(stream)


# --------------------------------------------------------------------------
# Direct-symbol helpers (canaries + finite sentinel)
# --------------------------------------------------------------------------

_CANARY = 16
_CANARY_LO = np.float32(-3.0)
_CANARY_HI = np.float32(11.0)
_SENTINEL = np.float32(7.0)


def _controlled_cache(*, num_kv_heads: int, head_dim: int, cache_blocks: int, block_size: int):
    key_cache = np.zeros((cache_blocks, block_size, num_kv_heads, head_dim), np.int8)
    value_cache = np.zeros_like(key_cache)
    k_scale = np.ones((cache_blocks, block_size, num_kv_heads), np.float32)
    v_scale = np.ones_like(k_scale)
    return key_cache, value_cache, k_scale, v_scale


def _controlled_case(scale_dtype: str, *, underflow: bool = False):
    """Slot 0 visible (V poisoned by the caller); slot 1 visible and normal."""

    num_kv_heads, head_dim, cache_blocks, block_size = 2, 4, 2, 4
    query = np.ones((num_kv_heads * 2, head_dim), np.float32)
    key_cache, value_cache, k_scale, v_scale = _controlled_cache(
        num_kv_heads=num_kv_heads, head_dim=head_dim, cache_blocks=cache_blocks,
        block_size=block_size,
    )
    value_cache[0, 0, :, :] = 1
    value_cache[0, 1, :, :] = 1
    if underflow:
        # Slot 0 logit ~ -1000 so its softmax weight underflows to zero.
        key_cache[0, 0, :, :] = 127
        k_scale[0, 0, :] = np.float32(-1000.0 / (127 * head_dim))
    table = np.asarray([0], dtype=np.int32)
    return query, key_cache, value_cache, k_scale, v_scale, table


def _run_direct(
    *,
    query,
    key_cache,
    value_cache,
    k_scale,
    v_scale,
    block_table,
    live_count,
    capacity,
    block_size,
    scale_dtype,
    token_positions=None,
    evict_mask=None,
    row_position=None,
    sliding_window=None,
    library,
):
    """Call the raw ``extern "C"`` symbol with canaries around the output.

    The body starts at a finite sentinel and is surrounded by finite canaries, so
    an all-NaN body is a real write and the canaries prove the guard stayed in
    bounds. Returns ``(body, canary_ok)``.
    """

    runtime = get_hip_runtime()
    num_q_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[2]
    symbol = (
        gemma4_attention_int8._SYMBOL_DECODE_SCALE_FP16
        if scale_dtype == "fp16"
        else gemma4_attention_int8._SYMBOL_DECODE_SCALE_F32
    )
    scale_np = _NP_SCALE[scale_dtype]
    k_scale = np.asarray(k_scale, dtype=scale_np)
    v_scale = np.asarray(v_scale, dtype=scale_np)
    body_n = num_q_heads * head_dim
    host = np.full(_CANARY + body_n + _CANARY, _SENTINEL, np.float32)
    host[:_CANARY] = _CANARY_LO
    host[_CANARY + body_n:] = _CANARY_HI
    buffers: list = []
    try:
        query_b = _upload(buffers, query, runtime=runtime)
        key_b = _upload(buffers, key_cache, runtime=runtime)
        value_b = _upload(buffers, value_cache, runtime=runtime)
        k_scale_b = _upload(buffers, k_scale, runtime=runtime)
        v_scale_b = _upload(buffers, v_scale, runtime=runtime)
        table_b = _upload(buffers, block_table.astype(np.int32), runtime=runtime)
        live_b = _upload(buffers, np.asarray([live_count], dtype=np.int64), runtime=runtime)
        pos_b = None
        if token_positions is not None:
            pos_b = _upload(buffers, token_positions.astype(np.int64), runtime=runtime)
        evict_b = None
        if evict_mask is not None:
            evict_b = _upload(buffers, evict_mask.astype(np.bool_), runtime=runtime)
        row_b = None
        if row_position is not None:
            row_b = _upload(buffers, np.asarray([row_position], dtype=np.int64), runtime=runtime)
        out_b = _upload(buffers, host, runtime=runtime)
        out_ptr = out_b.ptr + _CANARY * 4
        fn = getattr(library, symbol)
        fn.argtypes = list(gemma4_attention_int8._ARGTYPES)
        fn.restype = ctypes.c_int
        fn(
            ctypes.c_void_p(query_b.ptr),
            ctypes.c_void_p(key_b.ptr),
            ctypes.c_void_p(value_b.ptr),
            ctypes.c_void_p(k_scale_b.ptr),
            ctypes.c_void_p(v_scale_b.ptr),
            ctypes.c_void_p(out_ptr),
            ctypes.c_void_p(table_b.ptr),
            ctypes.c_void_p(live_b.ptr),
            ctypes.c_void_p(0 if pos_b is None else pos_b.ptr),
            ctypes.c_void_p(0 if evict_b is None else evict_b.ptr),
            ctypes.c_void_p(0 if row_b is None else row_b.ptr),
            ctypes.c_int64(int(capacity)),
            ctypes.c_int64(int(key_cache.shape[0])),
            ctypes.c_int64(int(block_table.size)),
            ctypes.c_int64(int(block_size)),
            ctypes.c_int64(int(num_q_heads)),
            ctypes.c_int64(int(num_kv_heads)),
            ctypes.c_int64(int(head_dim)),
            ctypes.c_int64(0 if sliding_window is None else int(sliding_window)),
            ctypes.c_float(1.0),
            ctypes.c_void_p(0),
        )
        runtime.device_synchronize()
        _download(out_b, host, runtime=runtime)
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)
    canary_ok = bool(
        np.array_equal(host[:_CANARY], np.full(_CANARY, _CANARY_LO, np.float32))
        and np.array_equal(host[_CANARY + body_n:], np.full(_CANARY, _CANARY_HI, np.float32))
    )
    body = host[_CANARY:_CANARY + body_n].reshape(num_q_heads, head_dim)
    return body, canary_ok
