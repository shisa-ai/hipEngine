"""Gemma 4 multi-row INT8 prefill consumer vs the CPU representation oracle.

The GPU prefill consumer must match the independent CPU reference on the *same*
quantized representation: INT8 payload plus per-token/per-KV-head FP16 or FP32
scale, reconstructed as ``float32(int8) * float32(scale)``. The original
unquantized FP32 fixture rows are a diagnostic, never the binding gate.

Declared semantics: every query row attends over ONE SHARED LOGICAL KV PREFIX
through a single 1-D page table. Each row carries its own live count and its own
absolute row position.

Coverage: both Gemma 4 GQA geometries, FP16 and FP32 scales, query rows 1/3/7
and an unrelated width, a context ladder around the 1024 page boundaries plus
4096, causal/future/negative/evicted/window masks, shuffled physical pages,
per-row empty/all-masked rows, distinct row positions, repeated poisoned-output
byte equality, finite failure sentinels/canaries, direct kernel guards, and a
non-blocking producer stream.
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
    gemma4_attention_prefill_int8_per_token_head,
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
    """Distinct K/V rows with zero, negative, and wide-range values."""

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


def _float_prefill(query, key, value, live_counts, *, row_positions, token_positions=None,
                   evict_mask=None, sliding_window=None, scale=1.0):
    """Unquantized diagnostic attention over the original float rows."""

    q = query.astype(np.float64)
    rows, num_q_heads, head_dim = q.shape
    num_kv_heads = key.shape[1]
    kv_group = num_q_heads // num_kv_heads
    out = np.empty_like(q)
    for row in range(rows):
        context = int(live_counts[row])
        query_position = int(row_positions[row])
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
                logits.append(float(key[slot, kv_head] @ q[row, head]) * scale if visible else -np.inf)
                vectors.append(value[slot, kv_head].astype(np.float64))
            logits = np.asarray(logits)
            if not np.isfinite(logits).any():
                out[row, head] = 0.0
                continue
            m = np.max(logits)
            w = np.exp(logits - m)
            w = w / w.sum()
            out[row, head] = sum(w[i] * vectors[i] for i in range(context))
    return out


def _run_prefill_case(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    live_counts,
    block_size: int,
    scale_dtype: str,
    seed: int,
    token_positions: np.ndarray | None = None,
    evict_mask: np.ndarray | None = None,
    row_positions=None,
    sliding_window: int | None = None,
    max_context_len: int | None = None,
    runs: int = 1,
    poison_masked_scales: bool = False,
    library,
):
    runtime = get_hip_runtime()
    live = np.asarray(live_counts, dtype=np.int64).reshape(-1)
    rows = int(live.size)
    context = int(live.max()) if rows else 0
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
    if row_positions is None:
        row_positions = live - 1
    positions = np.asarray(row_positions, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(seed + 7)
    query = rng.uniform(-2.0, 2.0, size=(rows, num_q_heads, head_dim)).astype(np.float32)
    bound = context if max_context_len is None else int(max_context_len)

    expected = gemma4_attention_prefill_int8_per_token_head(
        query, key_cache, value_cache, k_scale, v_scale, block_table, live,
        block_size=block_size, row_positions=positions, token_positions=token_positions,
        evict_mask=evict_mask, sliding_window=sliding_window,
    )

    buffers: list = []
    try:
        query_b = _upload(buffers, query, runtime=runtime)
        key_b = _upload(buffers, key_cache, runtime=runtime)
        value_b = _upload(buffers, value_cache, runtime=runtime)
        k_scale_b = _upload(buffers, k_scale, runtime=runtime)
        v_scale_b = _upload(buffers, v_scale, runtime=runtime)
        table_b = _upload(buffers, block_table, runtime=runtime)
        live_b = _upload(buffers, live, runtime=runtime)
        row_b = _upload(buffers, positions, runtime=runtime)
        pos_b = None
        if token_positions is not None:
            pos_b = _upload(buffers, token_positions.astype(np.int64), runtime=runtime)
        evict_b = None
        if evict_mask is not None:
            evict_b = _upload(buffers, evict_mask.astype(np.bool_), runtime=runtime)
        poison = _poison((rows, num_q_heads, head_dim))
        out_host = poison.copy()
        out_b = _upload(buffers, out_host, runtime=runtime)

        metadata = KVScaleMetadata(
            k_scale=_tensor(k_scale_b.ptr, k_scale.shape, _DTYPE_ENUM[scale_dtype]),
            v_scale=_tensor(v_scale_b.ptr, v_scale.shape, _DTYPE_ENUM[scale_dtype]),
            scale_dtype=_DTYPE_ENUM[scale_dtype],
        )
        spans = KVLiveSpans(
            base_offsets=_tensor(table_b.ptr, block_table.shape, DType.INT32),
            live_counts=_tensor(live_b.ptr, (rows,), DType.INT64),
            max_live_count=bound,
            token_positions=None if pos_b is None else _tensor(pos_b.ptr, (bound,), DType.INT64),
            evict_mask=None if evict_b is None else _tensor(evict_b.ptr, (bound,), DType.BOOL),
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            spans_mode="uniform",
            row_positions=_tensor(row_b.ptr, (rows,), DType.INT64),
            scale_metadata=metadata,
        )

        outputs: list[np.ndarray] = []
        for _ in range(runs):
            copy_host_to_device(out_b, host_array_ptr(poison), poison.nbytes, runtime=runtime)
            gemma4_attention_int8.gemma4_attention_prefill_int8_per_token_head_spans(
                query_b.ptr, key_b.ptr, value_b.ptr, out_b.ptr, spans, rows, bound, block_size,
                num_q_heads, num_kv_heads, head_dim, 1.0, sliding_window,
                library=library, runtime=runtime,
            )
            runtime.device_synchronize()
            _download(out_b, out_host, runtime=runtime)
            outputs.append(out_host.copy())
            assert not np.any(np.isnan(out_host)), "output not fully written"
            np.testing.assert_allclose(out_host, expected, rtol=1e-3, atol=3e-4)
        for other in outputs[1:]:
            np.testing.assert_array_equal(other.view(np.uint8), outputs[0].view(np.uint8))
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)

    diagnostic = _float_prefill(
        query, key, value, live, row_positions=positions,
        token_positions=token_positions, evict_mask=evict_mask, sliding_window=sliding_window,
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
def test_gemma4_int8_prefill_matches_oracle_over_context_ladder(
    _library, num_q_heads, num_kv_heads, head_dim, scale_dtype, context
) -> None:
    rows = 3
    expected, diagnostic, _ = _run_prefill_case(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        live_counts=[context] * rows,
        block_size=256,
        scale_dtype=scale_dtype,
        seed=0xD12 + context + num_kv_heads,
        library=_library,
    )
    assert expected.shape == (rows, num_q_heads, head_dim)
    # Diagnostic only: the quantized consumer must track the unquantized rows
    # within INT8 per-token/head error, not exactly. Correlation is the stable
    # signal; the binding gate above is the same-representation oracle match.
    assert np.all(np.isfinite(diagnostic))
    corr = float(np.corrcoef(expected.ravel(), diagnostic.ravel())[0, 1])
    assert corr > 0.9, f"quantized/unquantized correlation {corr} too low"


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
@pytest.mark.parametrize("rows", [1, 3, 7])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim", _GEOMETRIES, ids=("h8-d256", "h2-d512")
)
def test_gemma4_int8_prefill_rows_ladder_matches_oracle(
    _library, num_q_heads, num_kv_heads, head_dim, scale_dtype, rows
) -> None:
    context = 600
    # Distinct per-row prefix lengths and absolute positions.
    live = np.full(rows, context, dtype=np.int64)
    live[::2] = np.maximum(context - 5, 1)
    positions = np.asarray([context - 1 - r for r in range(rows)], dtype=np.int64)
    _run_prefill_case(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        live_counts=live,
        block_size=256,
        scale_dtype=scale_dtype,
        seed=0x700 + rows,
        row_positions=positions,
        library=_library,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_prefill_masks_positions_and_window_match_oracle(
    _library, scale_dtype
) -> None:
    num_q_heads, num_kv_heads, head_dim = 16, 8, 256
    context = 600
    bound = 700
    rows = 3
    rng = np.random.default_rng(0x5EED)
    positions = np.full(bound, -1, dtype=np.int64)
    positions[:context] = rng.permutation(np.arange(context, dtype=np.int64))
    evict = np.zeros(bound, dtype=bool)
    evict[:context] = rng.random(context) < 0.25
    _run_prefill_case(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        live_counts=[context] * rows,
        block_size=256,
        scale_dtype=scale_dtype,
        seed=0x5EED,
        token_positions=positions,
        evict_mask=evict,
        row_positions=[context - 1, 300, 42],
        sliding_window=128,
        max_context_len=bound,
        library=_library,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_prefill_causal_mixtures_match_oracle(_library, scale_dtype) -> None:
    num_q_heads, num_kv_heads, head_dim = 16, 8, 256
    context = 600
    bound = 640
    positions = np.full(bound, -1, dtype=np.int64)
    positions[:200] = np.arange(200)  # visible
    positions[200:400] = np.arange(200, 400) + 1000  # future
    # Slots 400..599 stay negative (excluded).
    _run_prefill_case(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        live_counts=[600, 400, 200],
        block_size=256,
        scale_dtype=scale_dtype,
        seed=0xC0FFEE,
        token_positions=positions,
        row_positions=[199, 250, 199],
        max_context_len=bound,
        library=_library,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_prefill_per_row_empty_and_all_masked(_library, scale_dtype) -> None:
    num_q_heads, num_kv_heads, head_dim = 16, 8, 256
    context = 300
    rows = 4
    expected, _, _ = _run_prefill_case(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        live_counts=[0, context, context, context],
        block_size=256,
        scale_dtype=scale_dtype,
        seed=0xA11,
        row_positions=[-1, -1, context - 1, 100],
        library=_library,
    )
    np.testing.assert_array_equal(expected[0], np.zeros_like(expected[0]))
    np.testing.assert_array_equal(expected[1], np.zeros_like(expected[1]))
    assert np.any(expected[2] != 0.0)
    assert np.any(expected[3] != 0.0)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_prefill_repeated_poisoned_output_is_deterministic(
    _library, scale_dtype
) -> None:
    for num_q_heads, num_kv_heads, head_dim in _GEOMETRIES:
        _run_prefill_case(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            live_counts=[1025, 512, 17],
            block_size=256,
            scale_dtype=scale_dtype,
            seed=0xDEAD,
            runs=2,
            library=_library,
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_prefill_masked_poisoned_scales_do_not_leak(_library, scale_dtype) -> None:
    context = 600
    evict = np.zeros(context, dtype=bool)
    evict[::3] = True
    _run_prefill_case(
        num_q_heads=16, num_kv_heads=8, head_dim=256,
        live_counts=[context, context, context],
        block_size=256, scale_dtype=scale_dtype, seed=0xA01,
        evict_mask=evict, poison_masked_scales=True, library=_library,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_prefill_visible_nonfinite_is_head_local_nan_with_canaries(
    _library, scale_dtype
) -> None:
    """A visible non-finite K must write NaN only for the mapped query heads."""

    num_q_heads, num_kv_heads, head_dim = 16, 8, 256
    block_size, capacity, cache_blocks = 4, 4, 1
    for poison in (np.inf, np.nan):
        query = np.ones((3, num_q_heads, head_dim), np.float32)
        key_cache, value_cache, k_scale, v_scale = _controlled_cache(
            num_kv_heads=num_kv_heads, head_dim=head_dim, cache_blocks=cache_blocks,
            block_size=block_size,
        )
        value_cache[:] = 1
        k_scale[0, 0, 0] = poison  # KV head 0, visible slot 0
        body, canary_ok, _ = _run_direct_prefill(
            query=query, key_cache=key_cache, value_cache=value_cache, k_scale=k_scale,
            v_scale=v_scale, block_table=np.asarray([0], dtype=np.int32),
            live_counts=[capacity, capacity, capacity], capacity=capacity,
            block_size=block_size, scale_dtype=scale_dtype,
            row_positions=[capacity - 1, capacity - 1, capacity - 1], library=_library,
        )
        assert canary_ok, "K-poison write escaped the output body"
        group = num_q_heads // num_kv_heads
        assert np.all(np.isnan(body[:, :group])), poison
        assert np.all(np.isfinite(body[:, group:])), poison


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


def _run_direct_prefill(
    *,
    query,
    key_cache,
    value_cache,
    k_scale,
    v_scale,
    block_table,
    live_counts,
    capacity,
    block_size,
    scale_dtype,
    row_positions=None,
    token_positions=None,
    evict_mask=None,
    sliding_window=None,
    rows_arg: int | None = None,
    library,
):
    """Call the raw prefill ``extern "C"`` symbol with canaries around the output.

    ``rows_arg`` overrides the row count passed to the launcher while the
    allocated arrays still hold ``query.shape[0]`` rows; the caller is
    responsible for keeping the actual metadata in range.
    """

    runtime = get_hip_runtime()
    rows, num_q_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[2]
    symbol = (
        gemma4_attention_int8._SYMBOL_PREFILL_SCALE_FP16
        if scale_dtype == "fp16"
        else gemma4_attention_int8._SYMBOL_PREFILL_SCALE_F32
    )
    scale_np = _NP_SCALE[scale_dtype]
    k_scale = np.asarray(k_scale, dtype=scale_np)
    v_scale = np.asarray(v_scale, dtype=scale_np)
    live = np.asarray(live_counts, dtype=np.int64).reshape(-1)
    positions = None if row_positions is None else np.asarray(row_positions, dtype=np.int64).reshape(-1)
    body_n = rows * num_q_heads * head_dim
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
        table_b = _upload(buffers, np.asarray(block_table, dtype=np.int32), runtime=runtime)
        live_b = _upload(buffers, live, runtime=runtime)
        row_b = None
        if positions is not None:
            row_b = _upload(buffers, positions, runtime=runtime)
        pos_b = None
        if token_positions is not None:
            pos_b = _upload(buffers, np.asarray(token_positions, dtype=np.int64), runtime=runtime)
        evict_b = None
        if evict_mask is not None:
            evict_b = _upload(buffers, np.asarray(evict_mask, dtype=np.bool_), runtime=runtime)
        out_b = _upload(buffers, host, runtime=runtime)
        out_ptr = out_b.ptr + _CANARY * 4
        fn = getattr(library, symbol)
        fn.argtypes = list(gemma4_attention_int8._ARGTYPES_PREFILL)
        fn.restype = ctypes.c_int
        err = fn(
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
            ctypes.c_int64(int(rows) if rows_arg is None else int(rows_arg)),
            ctypes.c_int64(int(capacity)),
            ctypes.c_int64(int(key_cache.shape[0])),
            ctypes.c_int64(int(np.asarray(block_table).size)),
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
    body = host[_CANARY:_CANARY + body_n].reshape(rows, num_q_heads, head_dim)
    return body, canary_ok, int(err)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_prefill_direct_symbol_guards_are_bounded(_library, scale_dtype) -> None:
    """Exercise the raw kernel guards with allocated metadata and a bounded cache."""

    num_q_heads, num_kv_heads, head_dim = 16, 8, 256
    block_size, capacity, cache_blocks = 4, 4, 2
    query = np.ones((3, num_q_heads, head_dim), np.float32)
    key_cache, value_cache, k_scale, v_scale = _controlled_cache(
        num_kv_heads=num_kv_heads, head_dim=head_dim, cache_blocks=cache_blocks,
        block_size=block_size,
    )
    value_cache[:] = 1

    def _run(live_counts, table, **extra):
        return _run_direct_prefill(
            query=query, key_cache=key_cache, value_cache=value_cache, k_scale=k_scale,
            v_scale=v_scale, block_table=table, live_counts=live_counts, capacity=capacity,
            block_size=block_size, scale_dtype=scale_dtype, library=_library, **extra,
        )

    body, canary_ok, _ = _run([0, 0, 0], np.asarray([0], dtype=np.int32),
                           row_positions=[-1, -1, -1])
    assert canary_ok and np.all(body == 0.0), "empty rows must zero their body"

    body, canary_ok, _ = _run([capacity, -1, capacity], np.asarray([0], dtype=np.int32),
                           row_positions=[3, 0, 3])
    assert canary_ok
    assert np.all(np.isnan(body[1])), "negative count must NaN its row"
    assert np.all(np.isfinite(body[0])) and np.all(np.isfinite(body[2]))

    body, canary_ok, _ = _run([capacity, capacity + 1, capacity], np.asarray([0], dtype=np.int32),
                           row_positions=[3, 4, 3])
    assert canary_ok and np.all(np.isnan(body[1])), "over-capacity count must NaN its row"

    body, canary_ok, _ = _run([capacity, capacity, capacity], np.asarray([cache_blocks], dtype=np.int32),
                           row_positions=[3, 3, 3])
    assert canary_ok and np.all(np.isnan(body)), "out-of-range visible page must NaN"

    body, canary_ok, _ = _run([capacity, capacity, capacity], np.asarray([-1], dtype=np.int32),
                           row_positions=[3, 3, 3])
    assert canary_ok and np.all(np.isnan(body)), "negative visible page must NaN"


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_prefill_direct_symbol_masked_invalid_page_is_not_failed(
    _library, scale_dtype
) -> None:
    """A masked out-of-range page is never dereferenced, so it is not a failure."""

    num_q_heads, num_kv_heads, head_dim = 16, 8, 256
    block_size, capacity, cache_blocks = 4, 4, 2
    query = np.ones((3, num_q_heads, head_dim), np.float32)
    key_cache, value_cache, k_scale, v_scale = _controlled_cache(
        num_kv_heads=num_kv_heads, head_dim=head_dim, cache_blocks=cache_blocks,
        block_size=block_size,
    )
    value_cache[:] = 1
    body, canary_ok, _ = _run_direct_prefill(
        query=query, key_cache=key_cache, value_cache=value_cache, k_scale=k_scale,
        v_scale=v_scale, block_table=np.asarray([cache_blocks], dtype=np.int32),
        live_counts=[capacity, capacity, capacity], capacity=capacity, block_size=block_size,
        scale_dtype=scale_dtype, row_positions=[3, 3, 3],
        evict_mask=np.ones(capacity, dtype=bool), library=_library,
    )
    assert canary_ok
    assert np.all(body == 0.0), "all-masked rows are zeros, not a failure"


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_prefill_direct_symbol_rejects_unrepresentable_row_count(
    _library, scale_dtype
) -> None:
    """``rows = 2**32 + 1`` must be refused, not silently truncated to one row.

    The ``unsigned int`` grid.y cast would wrap ``2**32 + 1`` to ``1`` and launch
    only row 0. The launcher's row bound must refuse it. The inputs are sized for
    one safe row (count 1, position 1) so a launch would not read out of bounds;
    the output body stays at the finite sentinel, proving nothing was written.
    """

    num_q_heads, num_kv_heads, head_dim = 16, 8, 256
    block_size, capacity, cache_blocks = 4, 4, 1
    query = np.ones((1, num_q_heads, head_dim), np.float32)
    key_cache, value_cache, k_scale, v_scale = _controlled_cache(
        num_kv_heads=num_kv_heads, head_dim=head_dim, cache_blocks=cache_blocks,
        block_size=block_size,
    )
    value_cache[:] = 1
    body, canary_ok, err = _run_direct_prefill(
        query=query, key_cache=key_cache, value_cache=value_cache, k_scale=k_scale,
        v_scale=v_scale, block_table=np.asarray([0], dtype=np.int32),
        live_counts=[1], capacity=capacity, block_size=block_size,
        scale_dtype=scale_dtype, row_positions=[0], rows_arg=2**32 + 1, library=_library,
    )
    assert err != 0, "an unrepresentable row count must be refused before launch"
    assert canary_ok, "a refused launch must not write the output"
    np.testing.assert_array_equal(
        body.view(np.uint8), np.full(body.shape, _SENTINEL, np.float32).view(np.uint8)
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_prefill_visible_nonfinite_v_underflowed_weight_is_head_local_nan(
    _library, scale_dtype
) -> None:
    """A visible non-finite V must fail its heads even when its weight underflows."""

    num_q_heads, num_kv_heads, head_dim = 16, 8, 256
    block_size, capacity, cache_blocks = 4, 4, 1
    rows = 2
    query = np.ones((rows, num_q_heads, head_dim), np.float32)
    key_cache, value_cache, k_scale, v_scale = _controlled_cache(
        num_kv_heads=num_kv_heads, head_dim=head_dim, cache_blocks=cache_blocks,
        block_size=block_size,
    )
    value_cache[:] = 1
    # Slot 0 of KV head 0: K = 127 with a tiny scale -> logit ~ -1000, so its
    # softmax weight underflows to zero. Its visible V is poisoned regardless.
    key_cache[0, 0, 0, :] = 127
    k_scale[0, 0, 0] = np.float32(-1000.0 / (127 * head_dim))
    v_scale[0, 0, 0] = np.inf
    body, canary_ok, err = _run_direct_prefill(
        query=query, key_cache=key_cache, value_cache=value_cache, k_scale=k_scale,
        v_scale=v_scale, block_table=np.asarray([0], dtype=np.int32),
        live_counts=[capacity, capacity], capacity=capacity, block_size=block_size,
        scale_dtype=scale_dtype, row_positions=[capacity - 1, capacity - 1], library=_library,
    )
    assert err == 0 and canary_ok
    group = num_q_heads // num_kv_heads
    assert np.all(np.isnan(body[:, :group])), "visible poisoned V must NaN its heads"
    assert np.all(np.isfinite(body[:, group:])), "unaffected heads must stay finite"


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_prefill_poison_visible_to_one_row_only_is_contained_per_row(
    _library, scale_dtype
) -> None:
    """A slot visible to one row and masked for another must fail only that row."""

    num_q_heads, num_kv_heads, head_dim = 16, 8, 256
    block_size, capacity, cache_blocks = 4, 4, 1
    rows = 2
    query = np.ones((rows, num_q_heads, head_dim), np.float32)
    key_cache, value_cache, k_scale, v_scale = _controlled_cache(
        num_kv_heads=num_kv_heads, head_dim=head_dim, cache_blocks=cache_blocks,
        block_size=block_size,
    )
    value_cache[:] = 1
    # Slot 1 of KV head 0 has a visible non-finite V. Row 0 (position 3) sees it;
    # row 1 (position 0) treats it as future and never dereferences it.
    v_scale[0, 1, 0] = np.inf
    positions = np.arange(capacity, dtype=np.int64)
    body, canary_ok, err = _run_direct_prefill(
        query=query, key_cache=key_cache, value_cache=value_cache, k_scale=k_scale,
        v_scale=v_scale, block_table=np.asarray([0], dtype=np.int32),
        live_counts=[capacity, capacity], capacity=capacity, block_size=block_size,
        scale_dtype=scale_dtype, row_positions=[3, 0], token_positions=positions,
        library=_library,
    )
    assert err == 0 and canary_ok
    group = num_q_heads // num_kv_heads
    assert np.all(np.isnan(body[0, :group])), "row 0 sees the poisoned slot"
    assert np.all(np.isfinite(body[0, group:])), "row 0's other heads stay finite"
    assert np.all(np.isfinite(body[1])), "row 1 never sees the poisoned slot"


def _checked_launch(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale_dtype: str,
    live_counts,
    block_table: np.ndarray,
    cache_blocks: int,
    block_size: int,
    max_context_len: int,
    library,
) -> None:
    runtime = get_hip_runtime()
    rows = len(live_counts)
    key_cache = np.full(
        (cache_blocks, block_size, num_kv_heads, head_dim), _SENTINEL_I8, np.int8
    )
    value_cache = np.full_like(key_cache, _SENTINEL_I8)
    np_dtype = _NP_SCALE[scale_dtype]
    k_scale = np.full(
        (cache_blocks, block_size, num_kv_heads), np.asarray(_SENTINEL_SCALE, np_dtype), np_dtype
    )
    v_scale = np.full_like(k_scale, np.asarray(_SENTINEL_SCALE, np_dtype))
    query = np.zeros((rows, num_q_heads, head_dim), np.float32)
    out = np.zeros((rows, num_q_heads, head_dim), np.float32)
    buffers: list = []
    try:
        query_b = _upload(buffers, query, runtime=runtime)
        key_b = _upload(buffers, key_cache, runtime=runtime)
        value_b = _upload(buffers, value_cache, runtime=runtime)
        k_scale_b = _upload(buffers, k_scale, runtime=runtime)
        v_scale_b = _upload(buffers, v_scale, runtime=runtime)
        table_b = _upload(buffers, block_table.astype(np.int32), runtime=runtime)
        live_b = _upload(buffers, np.asarray(live_counts, dtype=np.int64), runtime=runtime)
        row_b = _upload(buffers, np.zeros(rows, dtype=np.int64), runtime=runtime)
        out_b = _upload(buffers, out, runtime=runtime)
        metadata = KVScaleMetadata(
            k_scale=_tensor(k_scale_b.ptr, k_scale.shape, _DTYPE_ENUM[scale_dtype]),
            v_scale=_tensor(v_scale_b.ptr, v_scale.shape, _DTYPE_ENUM[scale_dtype]),
            scale_dtype=_DTYPE_ENUM[scale_dtype],
        )
        spans = KVLiveSpans(
            base_offsets=_tensor(table_b.ptr, block_table.shape, DType.INT32),
            live_counts=_tensor(live_b.ptr, (rows,), DType.INT64),
            max_live_count=max_context_len,
            token_positions=None,
            evict_mask=None,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            spans_mode="uniform",
            row_positions=_tensor(row_b.ptr, (rows,), DType.INT64),
            scale_metadata=metadata,
        )
        gemma4_attention_int8.gemma4_attention_prefill_int8_per_token_head_spans(
            query_b.ptr, key_b.ptr, value_b.ptr, out_b.ptr, spans, rows, max_context_len,
            block_size, num_q_heads, num_kv_heads, head_dim, 1.0, None,
            library=library, runtime=runtime,
        )
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_gemma4_int8_prefill_checked_path_refuses_invalid_metadata(_library, scale_dtype) -> None:
    common = dict(
        num_q_heads=16, num_kv_heads=8, head_dim=256, scale_dtype=scale_dtype,
        cache_blocks=2, block_size=4, max_context_len=4, library=_library,
    )
    with pytest.raises(ValueError, match="is negative"):
        _checked_launch(live_counts=[4, -1, 4], block_table=np.asarray([0], dtype=np.int32), **common)
    with pytest.raises(ValueError, match="exceeds max_context_len"):
        _checked_launch(live_counts=[4, 5, 4], block_table=np.asarray([0], dtype=np.int32), **common)
    with pytest.raises(ValueError, match="outside"):
        _checked_launch(live_counts=[4, 4, 4], block_table=np.asarray([2], dtype=np.int32), **common)
    with pytest.raises(ValueError, match="outside"):
        _checked_launch(live_counts=[4, 4, 4], block_table=np.asarray([-1], dtype=np.int32), **common)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gemma4_int8_prefill_readback_orders_against_nonblocking_stream(_library) -> None:
    """A non-blocking producer stream must be ordered before the metadata readback."""

    from hipengine.core.memory import enqueue_host_to_device

    runtime = get_hip_runtime()
    stream = runtime.stream_create(nonblocking=True)
    try:
        num_q_heads, num_kv_heads, head_dim = 16, 8, 256
        block_size, capacity, cache_blocks = 4, 4, 2
        rows = 3
        query = np.ones((rows, num_q_heads, head_dim), np.float32)
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
            # Seed invalid counts, then enqueue valid ones on a non-blocking stream.
            live_b = _upload(buffers, np.asarray([capacity + 1] * rows, dtype=np.int64), runtime=runtime)
            valid = np.asarray([capacity] * rows, dtype=np.int64)
            enqueue_host_to_device(
                live_b, host_array_ptr(valid), valid.nbytes, stream=stream, runtime=runtime
            )
            row_b = _upload(buffers, np.asarray([capacity - 1] * rows, dtype=np.int64), runtime=runtime)
            out_host = np.zeros((rows, num_q_heads, head_dim), np.float32)
            out_b = _upload(buffers, out_host, runtime=runtime)
            metadata = KVScaleMetadata(
                k_scale=_tensor(k_scale_b.ptr, k_scale.shape, DType.FP32),
                v_scale=_tensor(v_scale_b.ptr, v_scale.shape, DType.FP32),
                scale_dtype=DType.FP32,
            )
            spans = KVLiveSpans(
                base_offsets=_tensor(table_b.ptr, block_table.shape, DType.INT32),
                live_counts=_tensor(live_b.ptr, (rows,), DType.INT64),
                max_live_count=capacity,
                token_positions=None,
                evict_mask=None,
                storage_dtype=DType.INT8_PER_TOKEN_HEAD,
                spans_mode="uniform",
                row_positions=_tensor(row_b.ptr, (rows,), DType.INT64),
                scale_metadata=metadata,
            )
            gemma4_attention_int8.gemma4_attention_prefill_int8_per_token_head_spans(
                query_b.ptr, key_b.ptr, value_b.ptr, out_b.ptr, spans, rows, capacity, block_size,
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
