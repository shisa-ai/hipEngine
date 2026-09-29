"""The tiled flash prefill kernel against the exact Gemma 4 attention kernel.

The tiled route is a different association -- an online softmax over key tiles,
ported from llama.cpp's ``fattn-tile`` -- and it stages dtype on the way in
(bf16 -> f32 for Q, bf16 -> f16 for K/V), so agreement is bounded by
representation, not bitwise. What this test pins down is that the port computes
*the same attention*: the same mask, the same GQA head mapping, and the same
output row for every shape the kernel claims it can execute.

The two bugs this test exists to catch both vanished at head 0, so a probe that
only exercises the first head cannot see either: the stride to the next head
must be ``head_dim * tokens`` (ggml's ``nb[2] = ne[1] * nb[1]``), and the input
tensor the kernel indexes is ``(head_dim, tokens, heads)`` -- head *outermost* --
while hipEngine stores ``(tokens, num_heads, head_dim)``. Both are re-checked
here by including cases whose worst rows land on non-zero heads.

Shapes are chosen to cover the space rather than the gates: token counts on
either side of the 4-wide head tile and not a multiple of it, key counts at
several multiples of the 128 key tile, and the real global-layer geometry.
"""

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")

_HEADS = 16
_KV_HEADS = 2
_HEAD_DIM = 512
_KEY_TILE = 128
# BF16 output with f16-staged K/V: agreement is bounded by representation. The
# CPU-reference probe measured ~2 ULP on the largest magnitudes at this shape.
_TOLERANCE = {"rtol": 2.0e-2, "atol": 1.0e-2}


def _bf16(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    )


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float32)


def _causal_mask(rows: int, keys: int) -> np.ndarray:
    """``key <= query`` with the query block ending at the last key."""
    queries = np.arange(rows, dtype=np.int64)[:, None] + (keys - rows)
    key_positions = np.arange(keys, dtype=np.int64)[None, :]
    return np.ascontiguousarray((key_positions <= queries).astype(np.uint8))


def _windowed_mask(rows: int, keys: int, window: int) -> np.ndarray:
    queries = np.arange(rows, dtype=np.int64)[:, None] + (keys - rows)
    key_positions = np.arange(keys, dtype=np.int64)[None, :]
    keep = (key_positions <= queries) & ((queries - key_positions) < window)
    return np.ascontiguousarray(keep.astype(np.uint8))


def _run_both(rows: int, keys: int, *, window: int | None = None, seed: int = 5):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        Gemma4AttentionScratch,
        gemma4_attention_prefill_bf16,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_tiled import (
        gemma4_attention_prefill_tiled,
    )

    rng = np.random.default_rng(seed)
    query = _bf16(rng.normal(0.0, 0.5, (rows, _HEADS, _HEAD_DIM)))
    key = _bf16(rng.normal(0.0, 0.5, (keys, _KV_HEADS, _HEAD_DIM)))
    value = _bf16(rng.normal(0.0, 0.5, (keys, _KV_HEADS, _HEAD_DIM)))
    mask = (
        _causal_mask(rows, keys)
        if window is None
        else _windowed_mask(rows, keys, window)
    )

    runtime = get_hip_runtime()
    buffers = []
    scratch = Gemma4AttentionScratch()
    try:

        def upload(array: np.ndarray):
            buffer = malloc(array.nbytes, runtime=runtime)
            buffers.append(buffer)
            copy_host_to_device(
                buffer, host_array_ptr(array), array.nbytes, runtime=runtime
            )
            return buffer

        q_buf = upload(query)
        k_buf = upload(key)
        v_buf = upload(value)
        m_buf = upload(mask)
        exact_host = np.zeros((rows, _HEADS, _HEAD_DIM), dtype=np.uint16)
        exact_buf = upload(exact_host)
        tiled_host = np.zeros((rows, _HEADS, _HEAD_DIM), dtype=np.uint16)
        tiled_buf = upload(tiled_host)

        common = dict(
            tokens=rows,
            keys=keys,
            num_heads=_HEADS,
            num_kv_heads=_KV_HEADS,
            head_dim=_HEAD_DIM,
            scale=1.0,
            runtime=runtime,
        )
        # The exact launcher takes the shared scratch arena; the tiled launcher
        # stages its own dtype buffers inside the HIP source and takes none.
        gemma4_attention_prefill_bf16(
            q_buf.ptr,
            k_buf.ptr,
            v_buf.ptr,
            m_buf.ptr,
            exact_buf.ptr,
            scratch=scratch,
            **common,
        )
        gemma4_attention_prefill_tiled(
            q_buf.ptr,
            k_buf.ptr,
            v_buf.ptr,
            m_buf.ptr,
            tiled_buf.ptr,
            **common,
        )
        copy_device_to_host(
            host_array_ptr(exact_host), exact_buf, exact_host.nbytes, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(tiled_host), tiled_buf, tiled_host.nbytes, runtime=runtime
        )
        return _bf16_to_f32(exact_host), _bf16_to_f32(tiled_host)
    finally:
        scratch.close()
        for buffer in buffers:
            free(buffer, runtime=runtime)


@pytest.mark.parametrize(
    ("rows", "keys", "window"),
    [
        # Token counts on both sides of the 4-wide head tile, and not a multiple
        # of it: the tail tile must be dropped by the bounds check, not written.
        # tokens=1 and 3 sit *below* the tile width, which is the space the old
        # `tokens <= ncols1` guard used to refuse outright.
        pytest.param(1, 128, None, id="tokens-below-head-tile-1"),
        pytest.param(3, 128, None, id="tokens-below-head-tile-3"),
        pytest.param(4, 256, None, id="tokens-equal-head-tile-width"),
        pytest.param(5, 256, None, id="tokens-not-multiple-of-head-tile"),
        pytest.param(8, 256, None, id="tokens-8"),
        pytest.param(64, 128, None, id="keys-equal-key-tile"),
        pytest.param(64, 512, None, id="tokens-64-keys-512"),
        # Real global-layer geometry: 512 prefill tokens over a 1024-key range.
        pytest.param(512, 1024, None, id="global-layer-geometry"),
        # Non-causal mask: the kernel never derives causality, so a sliding
        # window must reach it intact.
        pytest.param(128, 512, 64, id="sliding-window-64"),
        pytest.param(128, 512, 1024, id="window-wider-than-range-is-causal"),
    ],
)
def test_tiled_prefill_matches_exact(rows: int, keys: int, window: int | None) -> None:
    exact, tiled = _run_both(rows, keys, window=window)
    np.testing.assert_allclose(tiled, exact, **_TOLERANCE)


def test_tiled_path_is_the_one_that_ran() -> None:
    """The symbol under test exists and is distinct from the exact kernel's.

    Parity is worthless if the call silently reached the kernel being compared
    against, so pin the two symbols apart before trusting the numbers above.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        _SYMBOL_PREFILL_BF16 as exact_symbol,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_tiled import (
        _SYMBOL_PREFILL_BF16 as tiled_symbol,
        build_gemma4_attention_tiled,
    )

    assert tiled_symbol != exact_symbol
    library = build_gemma4_attention_tiled(load=True)
    assert getattr(library, tiled_symbol) is not None