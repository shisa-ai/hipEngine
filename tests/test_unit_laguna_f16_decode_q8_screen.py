from __future__ import annotations

import numpy as np


def test_laguna_f16_decode_q8_screen_packs_raw_q8_0_blocks() -> None:
    from scripts.laguna_f16_decode_q8_screen import _layer_id, _quantize_f16_q8_0

    assert _layer_id("blk.23.attn_q.weight") == 23
    assert _layer_id("layers.23.attn_q") is None

    rng = np.random.default_rng(20260728)
    weight = rng.normal(0.0, 0.2, size=(5, 64)).astype(np.float16)
    packed = _quantize_f16_q8_0(weight)

    assert packed.shape == (5, 68)
    assert packed.dtype == np.uint8
    blocks = packed.reshape(5, 2, 34)
    scales = (
        np.ascontiguousarray(blocks[:, :, :2])
        .reshape(5, 2, 2)
        .copy()
        .view(np.float16)
        .reshape(5, 2)
        .astype(np.float32)
    )
    quants = blocks[:, :, 2:].view(np.int8).astype(np.float32)
    restored = (quants * scales[:, :, None]).reshape(weight.shape)
    block_error = np.max(
        np.abs(restored.reshape(5, 2, 32) - weight.astype(np.float32).reshape(5, 2, 32)),
        axis=2,
    )

    assert np.all(block_error <= scales * 0.501)
    np.testing.assert_array_equal(_quantize_f16_q8_0(weight), packed)
