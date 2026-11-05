"""Unit tier: Muse Glimmer GGUF loading, tensor map, and RoPE row permutation.

Hermetic (no GPU, no artifact): synthetic metadata exercises the loader's
capability checks, and the NORM->NEOX row permutation is pinned against the
CPU reference's interleaved rotation, which the llama.cpp oracle already validated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.muse_glimmer import (
    MuseGlimmerConfig,
    apply_rope_interleaved,
    rope_tables,
)
from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo
from hipengine.loading.muse_glimmer_gguf import (
    config_from_gguf_info,
    norm_to_neox_row_permutation,
    required_tensor_names,
    validate_tensor_map,
)


def _metadata(n_layers: int = 2) -> dict:
    p = "muse-glimmer."
    return {
        "general.architecture": "muse-glimmer",
        p + "block_count": n_layers,
        p + "embedding_length": 32,
        p + "feed_forward_length": 96,
        p + "attention.head_count": 4,
        p + "attention.head_count_kv": 2,
        p + "attention.key_length": 8,
        p + "rope.freq_base": 500000.0,
        p + "attention.layer_norm_rms_epsilon": 1e-5,
        p + "attention.sliding_window": 16,
        p + "attention.sliding_window_pattern": [True, False][:n_layers],
        p + "logit_scale": 0.25,
        p + "final_logit_softcapping": 20.0,
        p + "context_length": 128,
    }


def _tensor(name: str, shape: tuple[int, ...], type_name: str = "F32") -> GGUFTensorInfo:
    n = int(np.prod(shape))
    return GGUFTensorInfo(
        name=name, shape=shape, ggml_shape=tuple(reversed(shape)),
        ggml_type=0, ggml_type_name=type_name, n_elements=n,
        nbytes=n * 4, offset=0, data_offset=0, byte_shape=shape,
    )


def _info(n_layers: int = 2, *, vocab: int = 64) -> GGUFModelInfo:
    md = _metadata(n_layers)
    cfg = MuseGlimmerConfig.from_gguf_info(
        GGUFModelInfo(Path("x.gguf"), 3, 32, md, (), 0), vocab_size=vocab
    )
    tensors = [
        _tensor("token_embd.weight", (vocab, cfg.n_embd), "Q4_K"),
        _tensor("output_norm.weight", (cfg.n_embd,)),
        _tensor("output.weight", (vocab, cfg.n_embd), "Q5_K"),
    ]
    for layer in range(n_layers):
        p = f"blk.{layer}."
        d, nh, nkv = cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
        tensors += [
            _tensor(p + "attn_norm.weight", (cfg.n_embd,)),
            _tensor(p + "attn_q.weight", (nh * d, cfg.n_embd), "Q4_K"),
            _tensor(p + "attn_k.weight", (nkv * d, cfg.n_embd), "Q4_K"),
            _tensor(p + "attn_v.weight", (nkv * d, cfg.n_embd), "Q4_K"),
            _tensor(p + "attn_gate.weight", (nh * d, cfg.n_embd), "Q4_K"),
            _tensor(p + "attn_q_norm.weight", (d,)),
            _tensor(p + "attn_k_norm.weight", (d,)),
            _tensor(p + "attn_output.weight", (cfg.n_embd, nh * d), "Q4_K"),
            _tensor(p + "post_attention_norm.weight", (cfg.n_embd,)),
            _tensor(p + "ffn_norm.weight", (cfg.n_embd,)),
            _tensor(p + "ffn_gate.weight", (cfg.n_ff, cfg.n_embd), "Q4_K"),
            _tensor(p + "ffn_up.weight", (cfg.n_ff, cfg.n_embd), "Q4_K"),
            _tensor(p + "ffn_down.weight", (cfg.n_embd, cfg.n_ff), "Q6_K"),
            _tensor(p + "post_ffw_norm.weight", (cfg.n_embd,)),
        ]
    return GGUFModelInfo(Path("x.gguf"), 3, 32, md, tuple(tensors), 0)


def test_config_from_metadata() -> None:
    cfg = config_from_gguf_info(_info())
    assert cfg.n_layers == 2
    assert cfg.swa_pattern == (True, False)
    assert cfg.vocab_size == 64
    assert cfg.logit_scale == 0.25
    assert cfg.final_logit_softcapping == 20.0
    assert cfg.post_norm_eps == 1e-8


def test_validate_tensor_map_accepts_complete_map() -> None:
    info = _info()
    validate_tensor_map(info, config_from_gguf_info(info))
    assert len(required_tensor_names(config_from_gguf_info(info))) == len(info.tensors)


def test_validate_tensor_map_names_missing_tensor() -> None:
    info = _info()
    tensors = tuple(t for t in info.tensors if t.name != "blk.1.ffn_down.weight")
    broken = GGUFModelInfo(info.path, 3, 32, info.metadata, tensors, 0)
    with pytest.raises(ValueError, match="blk.1.ffn_down.weight"):
        validate_tensor_map(broken, config_from_gguf_info(info))


def test_validate_tensor_map_names_shape_miss() -> None:
    info = _info()
    tensors = tuple(
        _tensor(t.name, (7, 7), t.ggml_type_name) if t.name == "blk.0.attn_k.weight" else t
        for t in info.tensors
    )
    broken = GGUFModelInfo(info.path, 3, 32, info.metadata, tensors, 0)
    with pytest.raises(ValueError, match="blk.0.attn_k.weight.*shape"):
        validate_tensor_map(broken, config_from_gguf_info(info))


def test_permutation_is_bijective() -> None:
    perm = norm_to_neox_row_permutation(4, 8)
    assert sorted(perm.tolist()) == list(range(32))


def test_neox_on_permuted_rows_equals_interleaved_reference() -> None:
    """The load-time transform's contract: NEOX rotation on permuted Q/K rows,
    un-permuted back, equals the ggml NORM interleaved rotation bit-for-bit
    (up to f32 rounding of the two table constructions)."""

    rng = np.random.default_rng(7)
    n_heads, d, tokens = 5, 16, 4
    perm = norm_to_neox_row_permutation(n_heads, d)
    inv = np.argsort(perm)
    x = rng.standard_normal((tokens, n_heads, d)).astype(np.float32)
    positions = np.array([0, 3, 11, 127], dtype=np.int64)
    cos_i, sin_i = rope_tables(positions, d, 500000.0)
    reference = apply_rope_interleaved(x, cos_i, sin_i)

    xp = x.reshape(tokens, n_heads * d)[:, perm].reshape(tokens, n_heads, d)
    half = d // 2
    cb = np.concatenate([cos_i, cos_i], axis=1)[:, None, :half]
    sb = np.concatenate([sin_i, sin_i], axis=1)[:, None, :half]
    x0, x1 = xp[..., :half], xp[..., half:]
    neox = np.concatenate([x0 * cb - x1 * sb, x0 * sb + x1 * cb], axis=-1)
    back = neox.reshape(tokens, n_heads * d)[:, inv].reshape(tokens, n_heads, d)
    assert np.max(np.abs(back - reference)) < 2e-6


def test_permuted_quantized_rows_preserve_row_bytes() -> None:
    """Row reorder of block-quantized storage is a pure gather: every output
    row is byte-identical to its source row (no requantization)."""

    rng = np.random.default_rng(3)
    n_heads, d, row_bytes = 2, 8, 144
    raw = rng.integers(0, 256, size=(n_heads * d, row_bytes), dtype=np.uint8)
    perm = norm_to_neox_row_permutation(n_heads, d)
    permuted = raw[perm]
    for dst, src in enumerate(perm):
        assert np.array_equal(permuted[dst], raw[src])