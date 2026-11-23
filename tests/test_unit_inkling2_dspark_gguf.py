from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo, MissingGGUFTensorError
from hipengine.quant.gguf import GGMLQuantizationType as Q


def draft():
    metadata = {
        "general.architecture": "dflash", **{f"dflash.{k}": v for k, v in {
            "attention.causal": False, "attention.head_count": 2, "attention.head_count_kv": 1,
            "attention.key_length": 32, "attention.value_length": 32, "attention.layer_norm_rms_epsilon": 1e-6,
            "attention.sliding_window": 9, "attention.sliding_window_pattern": [True, False],
            "block_count": 2, "block_size": 4, "dspark.markov_rank": 4, "embedding_length": 32,
            "feed_forward_length": 64, "final_logit_softcapping": 30., "has_confidence_head": False,
            "rope.dimension_count": 32, "rope.freq_base": 1e6, "sample_from_anchor": False,
            "target_layers": [1, 3, 5], "vocab_size": 24, "vocab_size_unpadded": 17,
        }.items()},
    }
    tensors = []

    def add(name, shape, dtype=Q.F32):
        tensors.append(GGUFTensorInfo(name, shape, shape[::-1], dtype, dtype.name, 0, 0, 0, 0, shape))

    add("dspark_fuse.weight", (3, 3))
    add("dspark_hidden_out.bias", (32,))
    add("dspark_hidden_out.weight", (32, 4), Q.BF16)
    add("joint_proj.bias", (12,))
    add("joint_proj.weight", (12, 72), Q.BF16)
    add("markov_w1.weight", (24, 4), Q.BF16)
    for stem in ("dspark_norm", "mask_embd", "output_norm"):
        add(stem + ".weight", (32,))
    for layer in range(2):
        p = f"blk.{layer}."
        for stem, shape in {
            "attn_k": (32, 32), "attn_k_ctx": (32, 32), "attn_output": (32, 64),
            "attn_q": (64, 32), "attn_v": (32, 32), "attn_v_ctx": (32, 32),
            "ffn_down": (32, 64), "ffn_gate": (64, 32), "ffn_up": (64, 32),
        }.items():
            add(p + stem + ".weight", shape, Q.Q8_0)
        for stem in ("attn_k_ctx_norm", "attn_k_norm", "attn_norm", "attn_q_norm", "dflare_norm", "ffn_norm"):
            add(p + stem + ".weight", (32,))
    return GGUFModelInfo(None, 3, 32, metadata, tuple(tensors), 0)


def target():
    return SimpleNamespace(hidden_size=32, vocab_size=17, logit_cap=30., plan=SimpleNamespace(visits=[None] * 5))


def test_dspark_gguf_validates_header_and_slices_token_table_lazily():
    from hipengine.loading.inkling2_dspark_gguf import Inkling2DSparkGGUFWeights

    payload_reads = []

    def data(name):
        payload_reads.append(name)
        if name == "blk.1.attn_k_ctx.weight":
            raw = np.zeros((32, 34), dtype=np.uint8)
            raw[:, :2] = np.array([0.5], dtype="<f2").view(np.uint8)
            raw[4, 2:] = np.arange(32, dtype=np.uint8)
            return raw
        assert name == "markov_w1.weight"
        return (np.arange(96, dtype=np.float32).reshape(24, 4).view(np.uint32) >> 16).astype(np.uint16)

    w = Inkling2DSparkGGUFWeights(SimpleNamespace(info=draft(), tensor_data=data), target())
    assert payload_reads == []
    assert w.config["target_layer_ids"] == (1, 3, 5)
    assert w.config["local_layer_ids"] == (0,)
    assert w.names["layers.1.self_attn.k_proj_ctx.weight"].name == "blk.1.attn_k_ctx.weight"
    assert w.names["markov_head.rnn_proj.weight"].name == "joint_proj.weight"
    np.testing.assert_array_equal(w.dense("markov_head.markov_w1.weight", 16), [64, 65, 66, 67])
    np.testing.assert_array_equal(w.dense("layers.1.self_attn.k_proj_ctx.weight", 4), np.arange(32) * 0.5)
    assert payload_reads == ["markov_w1.weight", "blk.1.attn_k_ctx.weight"]


@pytest.mark.parametrize("key,value", [
    ("attention.causal", True), ("attention.sliding_window_pattern", [True]), ("rope.dimension_count", 16),
    ("target_layers", [1, 5, 3]), ("target_layers", [1, 3, 4]), ("vocab_size_unpadded", 18),
])
def test_dspark_gguf_rejects_incompatible_geometry(key, value):
    from hipengine.loading.inkling2_dspark_gguf import build_inkling2_dspark_gguf_map

    info = draft()
    info.metadata[f"dflash.{key}"] = value
    with pytest.raises(ValueError, match="DSpark"):
        build_inkling2_dspark_gguf_map(info, target())


def test_dspark_gguf_reports_missing_context_projection():
    from dataclasses import replace
    from hipengine.loading.inkling2_dspark_gguf import build_inkling2_dspark_gguf_map

    info = draft()
    info = replace(info, tensors=tuple(t for t in info.tensors if t.name != "blk.1.attn_v_ctx.weight"))
    with pytest.raises(MissingGGUFTensorError, match=r"blk\.1\.attn_v_ctx\.weight"):
        build_inkling2_dspark_gguf_map(info, target())
