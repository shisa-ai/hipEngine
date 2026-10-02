"""GPU route for the Gemma 4 INT8 KV cache: owner -> layer -> CPU oracle.

This is the integration gate the runner's INT8 path needs and the unit tests
cannot provide. It drives the exact function the runner calls
(``gemma4_layer._run_int8_attention``) against the real owner
(``Gemma4Int8KVCache``), so the registered writer and consumer run on the
owner's own buffers, spans and page table -- not on a fixture that reimplements
them. The independent CPU reference
(``gemma4_attention_prefill_int8_per_token_head`` /
``gemma4_attention_decode_int8_per_token_head``) is fed the quantized cache the
writer actually produced, so the gate is on the same representation, never on
the original BF16 rows.

Coverage, on both Gemma 4 geometries:

* prefill (multi-row writer + shared-prefix consumer) for a global and a
  sliding layer;
* page crossings -- a prefill wider than one cache page;
* a nonzero-offset multi-row prefill, and several such chunks appended in
  sequence so each later chunk attends over its predecessors;
* a full-capacity append, which is the case the owner's extra page exists for;
* decode continuity -- a single appended row attends over the prefill plus
  itself at its absolute position;
* reset/reuse with a shorter sequence, proving a later request reads only its
  own appended slots;
* the FP32-scale route, not just the default FP16 one;
* that the layer's BF16 context is the BF16 rounding of the FP32 consumer
  scratch, and that its selected route is the registered writer/consumer.

The owner is built per case from a small spec rather than a whole model, so the
suite stays a focused correctness fixture with no expensive ladder.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import (
    gemma4_attention_decode_int8_per_token_head,
    gemma4_attention_prefill_int8_per_token_head,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import (
    Gemma4LayerGeometry,
    _run_int8_attention,
    last_int8_kv_route,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.runtime.gemma4_int8_kv import Gemma4Int8KVCache

_GEOMETRY = ((16, 8, 256), (16, 2, 512))
_GEOMETRY_IDS = ("h8-d256", "h2-d512")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits_to_float(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _upload(buffers: list, array: np.ndarray, *, runtime):
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes, runtime=runtime)
    buffers.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes, runtime=runtime)
    return buffer


def _download(buffer, array: np.ndarray, *, runtime) -> np.ndarray:
    copy_device_to_host(host_array_ptr(array), buffer, array.nbytes, runtime=runtime)
    return array


def _fixture(rows: int, num_heads: int, num_kv_heads: int, head_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    key = rng.uniform(-6.0, 6.0, size=(rows, num_kv_heads, head_dim)).astype(np.float32)
    value = rng.uniform(-6.0, 6.0, size=(rows, num_kv_heads, head_dim)).astype(np.float32)
    query = rng.uniform(-2.0, 2.0, size=(rows, num_heads, head_dim)).astype(np.float32)
    key[0, 0] = 0.0
    value[0, 0] = 0.0
    return query, key, value


def _layer_run(
    *,
    owner: Gemma4Int8KVCache,
    layer: int,
    write_offset: int,
    rows: int,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    sliding_window: int | None,
    runtime,
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """Run one layer forward through the owner and return the consumer output.

    Returns ``(context_f32, context_bf16_bits, route)``.
    """

    num_heads, num_kv_heads, head_dim = _GEOMETRY[layer]
    block = owner.begin_block(write_offset=write_offset, rows=rows, stream=0)
    layer_kv = owner.layer_kv(layer, block)
    geometry = Gemma4LayerGeometry(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=1.0,
        k_eq_v=False,
        sliding_window=sliding_window,
    )
    buffers: list = []
    try:
        query_bf16 = _upload(buffers, float_array_to_bf16_bits(query), runtime=runtime)
        key_bf16 = _upload(buffers, float_array_to_bf16_bits(key), runtime=runtime)
        value_bf16 = _upload(buffers, float_array_to_bf16_bits(value), runtime=runtime)
        context_bf16 = _upload(
            buffers,
            np.zeros((rows, num_heads, head_dim), dtype=np.uint16),
            runtime=runtime,
        )
        _run_int8_attention(
            layer_kv,
            query_bf16=query_bf16.ptr,
            key_bf16=key_bf16.ptr,
            value_bf16=value_bf16.ptr,
            context_bf16=context_bf16.ptr,
            rows=rows,
            geometry=geometry,
            stream=0,
        )
        runtime.device_synchronize()
        context_f32_host = np.empty((rows, num_heads, head_dim), dtype=np.float32)
        _download(owner.context_f32_buffers[layer], context_f32_host, runtime=runtime)
        context_bf16_host = np.empty((rows, num_heads, head_dim), dtype=np.uint16)
        _download(context_bf16, context_bf16_host, runtime=runtime)
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)
    return context_f32_host, context_bf16_host, last_int8_kv_route()


def _download_quantized(owner: Gemma4Int8KVCache, layer: int, runtime):
    """Download the quantized cache the writer produced, in oracle layout."""

    blocks = owner.blocks
    block_size = owner.block_size
    _, num_kv_heads, head_dim = _GEOMETRY[layer]
    scale_np = np.float16 if owner.scale_dtype.itemsize == 2 else np.float32

    key_host = np.empty((blocks, block_size, num_kv_heads, head_dim), dtype=np.int8)
    value_host = np.empty_like(key_host)
    k_scale_host = np.empty((blocks, block_size, num_kv_heads), dtype=scale_np)
    v_scale_host = np.empty_like(k_scale_host)
    _download(owner.key_caches[layer], key_host, runtime=runtime)
    _download(owner.value_caches[layer], value_host, runtime=runtime)
    _download(owner.k_scale_buffers[layer], k_scale_host, runtime=runtime)
    _download(owner.v_scale_buffers[layer], v_scale_host, runtime=runtime)
    return key_host, value_host, k_scale_host, v_scale_host


def _install_page_table(owner: Gemma4Int8KVCache, permutation: np.ndarray, runtime) -> None:
    """Replace the owner's identity page table with ``permutation``.

    The writer's row-major view and the consumer's 1-D view alias this one
    buffer, so a non-identity table is the only way to prove neither hard-codes
    the identity mapping.
    """

    table = np.tile(
        np.asarray(permutation, dtype=np.int32), (owner.max_block, 1)
    ).reshape(-1)
    copy_host_to_device(
        owner._page_table, host_array_ptr(table), table.nbytes, runtime=runtime
    )
    runtime.device_synchronize()


def _build_owner(
    runtime,
    *,
    capacity: int = 256,
    block_size: int = 32,
    max_block: int = 64,
    scale_dtype: DType = DType.FP16,
) -> Gemma4Int8KVCache:
    """Build the owner and zero its planes so unwritten slots are finite.

    The owner allocates uninitialized device memory. Only slots a request wrote
    are ever read, but the oracle dequantizes the whole cache before masking, so
    a finite fill keeps the comparison free of NaN warnings from unwritten
    slots without weakening the gate.
    """

    owner = Gemma4Int8KVCache(
        capacity=capacity,
        max_block=max_block,
        attentions=_GEOMETRY,
        block_size=block_size,
        scale_dtype=scale_dtype,
    )
    buffers = (
        *owner.key_caches,
        *owner.value_caches,
        *owner.k_scale_buffers,
        *owner.v_scale_buffers,
    )
    for buffer in buffers:
        zeros = np.zeros(buffer.nbytes, dtype=np.uint8)
        copy_host_to_device(buffer, host_array_ptr(zeros), zeros.nbytes, runtime=runtime)
    runtime.device_synchronize()
    return owner


def _prefill_and_compare(
    owner: Gemma4Int8KVCache,
    layer: int,
    *,
    write_offset: int,
    rows: int,
    runtime,
    seed: int,
    sliding_window: int | None = None,
    block_table: np.ndarray | None = None,
) -> tuple[dict | None, np.ndarray, np.ndarray]:
    """Run a prefill through the owner and gate it on the representation oracle."""

    num_heads, num_kv_heads, head_dim = _GEOMETRY[layer]
    query, key, value = _fixture(rows, num_heads, num_kv_heads, head_dim, seed)
    context_f32, context_bf16, route = _layer_run(
        owner=owner,
        layer=layer,
        write_offset=write_offset,
        rows=rows,
        query=query,
        key=key,
        value=value,
        sliding_window=sliding_window,
        runtime=runtime,
    )
    key_cache, value_cache, k_scale, v_scale = _download_quantized(owner, layer, runtime)
    q_f32 = _bf16_bits_to_float(float_array_to_bf16_bits(query))
    counts = np.arange(write_offset + 1, write_offset + rows + 1, dtype=np.int64)
    positions = np.arange(write_offset, write_offset + rows, dtype=np.int64)
    if block_table is None:
        block_table = np.arange(owner.blocks, dtype=np.int32)
    expected = gemma4_attention_prefill_int8_per_token_head(
        q_f32,
        key_cache,
        value_cache,
        k_scale,
        v_scale,
        np.asarray(block_table, dtype=np.int32),
        counts,
        block_size=owner.block_size,
        row_positions=positions,
        scale=1.0,
        sliding_window=sliding_window,
    )
    np.testing.assert_allclose(context_f32, expected, rtol=1e-3, atol=3e-4)
    return route, context_f32, context_bf16


@pytest.fixture(scope="module")
def runtime():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    return get_hip_runtime()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("layer", [0, 1], ids=_GEOMETRY_IDS)
@pytest.mark.parametrize("sliding_window", [None, 3], ids=("global", "sliding"))
def test_prefill_route_matches_oracle_on_the_written_cache(
    runtime, layer: int, sliding_window: int | None
) -> None:
    owner = _build_owner(runtime)
    try:
        route, context_f32, context_bf16 = _prefill_and_compare(
            owner, layer, write_offset=0, rows=5, runtime=runtime,
            seed=0xD12 + layer, sliding_window=sliding_window,
        )
        assert route is not None, "the layer recorded no INT8 route"
        assert route["writer"] == "int8_per_token_head/per_token_head_bf16_prompt_spans"
        assert route["consumer"] == (
            "paged_attn_prefill/int8_per_token_head/gemma4_direct_spans"
        )
        # The layer's BF16 context is the BF16 rounding of the FP32 consumer
        # scratch, not a separately computed (and differently rounded) value.
        np.testing.assert_array_equal(
            float_array_to_bf16_bits(context_f32), context_bf16
        )
    finally:
        owner.close()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("layer", [0, 1], ids=_GEOMETRY_IDS)
def test_prefill_wider_than_a_page_crosses_pages(runtime, layer: int) -> None:
    """A 40-row prefill at block_size 32 spans two cache pages.

    Every row beyond the first page is addressed through the page table, so a
    consumer that assumed one page -- or a writer whose span generation did not
    advance a page -- would misplace or misread rows 32 onward while rows 0..31
    still passed. The owner's page table is the identity here, so the test is
    about the width, not the indirection; ``test_permuted_page_table_is_honoured``
    covers indirection.
    """

    owner = _build_owner(runtime, capacity=256, block_size=32, max_block=64)
    try:
        assert owner.blocks >= 3
        route, _, _ = _prefill_and_compare(
            owner, layer, write_offset=0, rows=40, runtime=runtime, seed=0xC705 + layer
        )
        assert route is not None
        assert route["writer"] == "int8_per_token_head/per_token_head_bf16_prompt_spans"
    finally:
        owner.close()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("layer", [0, 1], ids=_GEOMETRY_IDS)
def test_permuted_page_table_is_honoured(runtime, layer: int) -> None:
    """A non-identity page table must move rows, in both directions.

    Physical blocks are reversed relative to logical ones, so logical slot *i*
    lives at a different physical slot. The oracle is given the same permutation.
    A writer or consumer that assumed logical slot == physical slot would pass on
    the identity table and fail here, on prefill and on the decode that follows.
    """

    owner = _build_owner(runtime, capacity=128, block_size=16, max_block=64)
    try:
        permutation = np.arange(owner.blocks, dtype=np.int32)[::-1].copy()
        assert not np.array_equal(permutation, np.arange(owner.blocks))
        _install_page_table(owner, permutation, runtime)

        rows = 24
        _prefill_and_compare(
            owner,
            layer,
            write_offset=0,
            rows=rows,
            runtime=runtime,
            seed=0x9E4E + layer,
            block_table=permutation,
        )

        num_heads, num_kv_heads, head_dim = _GEOMETRY[layer]
        dq, dk, dv = _fixture(1, num_heads, num_kv_heads, head_dim, 0x9ED0 + layer)
        context_f32, _, route = _layer_run(
            owner=owner, layer=layer, write_offset=rows, rows=1, query=dq, key=dk, value=dv,
            sliding_window=None, runtime=runtime,
        )
        assert route is not None
        assert route["consumer"] == (
            "paged_attn_decode/int8_per_token_head/gemma4_direct_spans"
        )
        key_cache, value_cache, k_scale, v_scale = _download_quantized(owner, layer, runtime)
        expected = gemma4_attention_decode_int8_per_token_head(
            _bf16_bits_to_float(float_array_to_bf16_bits(dq))[0],
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            permutation,
            rows + 1,
            block_size=owner.block_size,
            scale=1.0,
            token_positions=np.arange(rows + 1, dtype=np.int64),
            row_position=rows,
        )
        np.testing.assert_allclose(context_f32[0], expected, rtol=1e-3, atol=3e-4)
    finally:
        owner.close()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("layer", [0, 1], ids=_GEOMETRY_IDS)
def test_chunked_nonzero_offset_prefill_matches_oracle(runtime, layer: int) -> None:
    """Three 8-row chunks appended at increasing offsets, each over its prefix.

    Chunks after the first are multi-row prefills whose ``write_offset`` is not
    zero, so their absolute positions differ from their slot indices. A consumer
    that treated the slot index as the position would get chunks two and three
    wrong while chunk one still passed.
    """

    owner = _build_owner(runtime, capacity=256, block_size=32, max_block=64)
    try:
        for chunk in range(3):
            _prefill_and_compare(
                owner,
                layer,
                write_offset=chunk * 8,
                rows=8,
                runtime=runtime,
                seed=0xC4A0 + layer * 16 + chunk,
            )
    finally:
        owner.close()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("layer", [0, 1], ids=_GEOMETRY_IDS)
def test_full_capacity_append_matches_oracle(runtime, layer: int) -> None:
    """Filling the cache exactly to ``capacity`` must not trip a page bound.

    ``blocks`` is one more than the minimum to cover ``capacity``, so
    ``block_size * blocks`` strictly exceeds the writer's ``max_live_count``.
    A full request is the case that bound exists for; a smaller page count would
    refuse exactly this request.
    """

    owner = _build_owner(runtime, capacity=64, block_size=16, max_block=64)
    try:
        assert owner.blocks * 16 > 64
        route, _, _ = _prefill_and_compare(
            owner, layer, write_offset=0, rows=64, runtime=runtime, seed=0xF0C4 + layer
        )
        assert route is not None
        assert route["writer"] == "int8_per_token_head/per_token_head_bf16_prompt_spans"
    finally:
        owner.close()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("layer", [0, 1], ids=_GEOMETRY_IDS)
def test_fp32_scale_route_matches_oracle(runtime, layer: int) -> None:
    """The FP32 scale plane is a separate kernel instantiation, not a default."""

    owner = _build_owner(runtime, scale_dtype=DType.FP32)
    try:
        assert owner.scale_dtype == DType.FP32
        _prefill_and_compare(
            owner, layer, write_offset=0, rows=5, runtime=runtime, seed=0xF32 + layer
        )
    finally:
        owner.close()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("layer", [0, 1], ids=_GEOMETRY_IDS)
def test_decode_continues_the_prefill_with_absolute_positions(runtime, layer: int) -> None:
    """A decode step at position ``rows`` attends over the prefill plus itself.

    The oracle must be told each slot's absolute position: the cache is indexed
    from slot 0, so a decode query at absolute position 5 reads slot 0 whose
    position is 5, not 0. This is exactly the case a slot-index oracle silently
    passes on a wrong answer.
    """

    owner = _build_owner(runtime)
    try:
        rows = 4
        num_heads, num_kv_heads, head_dim = _GEOMETRY[layer]
        query, key, value = _fixture(rows, num_heads, num_kv_heads, head_dim, 0xDE0 + layer)
        _layer_run(
            owner=owner, layer=layer, write_offset=0, rows=rows, query=query, key=key,
            value=value, sliding_window=None, runtime=runtime,
        )

        dq, dk, dv = _fixture(1, num_heads, num_kv_heads, head_dim, 0xDEC + layer)
        context_f32, _, route = _layer_run(
            owner=owner, layer=layer, write_offset=rows, rows=1, query=dq, key=dk, value=dv,
            sliding_window=None, runtime=runtime,
        )
        assert route is not None
        assert route["writer"] == "int8_per_token_head/per_token_head_bf16_spans"
        assert route["consumer"] == (
            "paged_attn_decode/int8_per_token_head/gemma4_direct_spans"
        )

        key_cache, value_cache, k_scale, v_scale = _download_quantized(owner, layer, runtime)
        q_f32 = _bf16_bits_to_float(float_array_to_bf16_bits(dq))
        expected = gemma4_attention_decode_int8_per_token_head(
            q_f32[0],
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            np.arange(owner.blocks, dtype=np.int32),
            rows + 1,
            block_size=owner.block_size,
            scale=1.0,
            token_positions=np.arange(rows + 1, dtype=np.int64),
            row_position=rows,
        )
        np.testing.assert_allclose(context_f32[0], expected, rtol=1e-3, atol=3e-4)
    finally:
        owner.close()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_reset_then_shorter_request_reads_only_its_own_slots(runtime) -> None:
    """A shorter request after a longer one must not read the previous slots."""

    owner = _build_owner(runtime)
    try:
        layer = 0
        num_heads, num_kv_heads, head_dim = _GEOMETRY[layer]
        query, key, value = _fixture(6, num_heads, num_kv_heads, head_dim, 0x10E)
        _layer_run(
            owner=owner, layer=layer, write_offset=0, rows=6, query=query, key=key,
            value=value, sliding_window=None, runtime=runtime,
        )
        owner.reset()

        short = 2
        query2, key2, value2 = _fixture(short, num_heads, num_kv_heads, head_dim, 0x5A0)
        context_f32, _, _ = _layer_run(
            owner=owner, layer=layer, write_offset=0, rows=short, query=query2, key=key2,
            value=value2, sliding_window=None, runtime=runtime,
        )

        key_cache, value_cache, k_scale, v_scale = _download_quantized(owner, layer, runtime)
        q_f32 = _bf16_bits_to_float(float_array_to_bf16_bits(query2))
        expected = gemma4_attention_prefill_int8_per_token_head(
            q_f32,
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            np.arange(owner.blocks, dtype=np.int32),
            np.arange(1, short + 1, dtype=np.int64),
            block_size=owner.block_size,
            row_positions=np.arange(short, dtype=np.int64),
            scale=1.0,
        )
        np.testing.assert_allclose(context_f32, expected, rtol=1e-3, atol=3e-4)

        # The second row of the short request is its own write, not a survivor
        # of the six-row request.
        assert np.any(expected[1] != 0.0)
    finally:
        owner.close()
