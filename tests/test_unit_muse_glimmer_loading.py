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

def test_context_beyond_trained_positions_runs_with_a_warning(caplog) -> None:
    from types import SimpleNamespace

    from hipengine.generation.muse_glimmer_gguf import MuseGlimmerGGUFGenerator

    owner = SimpleNamespace(
        weight_index=SimpleNamespace(metadata={"muse-glimmer.context_length": 131072})
    )
    with caplog.at_level("WARNING"):
        assert MuseGlimmerGGUFGenerator._validate_context(owner, 131072) == 131072
    assert not caplog.records
    with caplog.at_level("WARNING"):
        assert MuseGlimmerGGUFGenerator._validate_context(owner, 262144) == 262144
    assert "trained 131072" in caplog.text
    with pytest.raises(ValueError):
        MuseGlimmerGGUFGenerator._validate_context(owner, 0)


@pytest.mark.parametrize("owner", ["target", "drafter"])
@pytest.mark.parametrize("fail_upload", [False, True])
def test_weight_owner_releases_each_tensor_allocation(owner, fail_upload):
    from dataclasses import replace
    from types import SimpleNamespace

    from hipengine.kernels.cpu_reference.muse_glimmer_dflash import MuseGlimmerDFlashConfig
    from hipengine.loading.muse_glimmer_gguf import materialize_muse_glimmer_weights
    from hipengine.runtime.muse_glimmer_dflash import MuseGlimmerDFlashDrafter

    info = _info(n_layers=1)
    info = replace(info, tensors=tuple(_tensor(t.name, t.shape) for t in info.tensors))
    reader = SimpleNamespace(
        info=info, tensor_info=info.tensor,
        tensor_data=lambda name: np.zeros(info.tensor(name).shape, dtype=np.float32),
    )

    class Runtime:
        def __init__(self):
            self.attempts = 0
            self.live = set()

        def malloc(self, nbytes):
            self.attempts += 1
            if fail_upload and self.attempts == 3:
                raise MemoryError("injected upload allocation failure")
            ptr = self.attempts * 0x10000
            self.live.add(ptr)
            return ptr

        def free(self, ptr):
            self.live.remove(ptr)

        def memcpy(self, dst, src, nbytes, kind):
            assert dst in self.live

    runtime = Runtime()

    def load():
        if owner == "target":
            weights = materialize_muse_glimmer_weights(
                reader, backend="hip_gfx1151", runtime=runtime,
            )
            assert len({weights.ptr(t.name) for t in info.tensors}) == len(info.tensors)
            weights.free(runtime=runtime)
        else:
            config = MuseGlimmerDFlashConfig(
                n_layers=1, n_embd=32, n_ff=96, n_heads=4, n_kv_heads=2,
                head_dim=8, block_size=4, sliding_window=8, rope_freq_base=500000.0,
                rms_eps=1e-5, target_layers=(0,), mask_token_id=0,
            )
            draft = MuseGlimmerDFlashDrafter(
                reader, config, target_weights=SimpleNamespace(config=SimpleNamespace(vocab_size=64)),
                backend="hip_gfx1151", max_context=8, runtime=runtime,
            )
            draft.close()

    if fail_upload:
        with pytest.raises(MemoryError, match="injected upload allocation failure"):
            load()
    else:
        load()
    assert runtime.live == set()


def test_session_close_releases_graph_resources_before_backing_storage(monkeypatch):
    from types import SimpleNamespace

    from hipengine.runtime import muse_glimmer

    events = []
    runtime = SimpleNamespace(**{
        name: lambda value, name=name: events.append((name, value))
        for name in ("stream_synchronize", "graph_exec_destroy", "stream_destroy")
    })
    session = object.__new__(muse_glimmer.MuseGlimmerResidentSession)
    session.runtime = runtime
    session._closed = False
    session._graph_stream = 1
    session._vg_stream = 2
    session._graph_exec = 3
    session._verify_graph_exec = {(2, ()): 4, (3, ()): 5}
    session._prefill_fullgemm_workspace = SimpleNamespace(
        close=lambda: events.append(("workspace", None)),
    )
    session._scratch = None
    session._kv = []
    for name in ("_g_staging", "_vg_key_end", "_vg_tap_buf", "_vg_logits_buf"):
        setattr(session, name, name)
    monkeypatch.setattr(muse_glimmer, "free", lambda buf, **kw: events.append(("free", buf)))

    session.close()
    session.close()  # Idempotent.
    assert events == [
        ("stream_synchronize", 1), ("stream_synchronize", 2),
        ("graph_exec_destroy", 3), ("graph_exec_destroy", 4), ("graph_exec_destroy", 5),
        ("stream_destroy", 1), ("stream_destroy", 2),
        ("free", "_g_staging"), ("free", "_vg_key_end"),
        ("free", "_vg_tap_buf"), ("free", "_vg_logits_buf"), ("workspace", None),
    ]
