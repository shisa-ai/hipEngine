from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from hipengine.loading.gguf import GGUFModelInfo
from hipengine.loading.inkling2_gguf import Inkling2GGUFWeights, build_inkling2_gguf_map
from hipengine.quant.gguf import GGMLQuantizationType as Q
from tests._laguna_synthetic import tensor_info


def _file(legacy):
    architecture = "inkling2" if legacy else "inkling2_loop"
    prefix = architecture + "."
    settings = {
        "block_count": 8 if legacy else 6,
        "context_length": 64,
        "embedding_length": 256,
        "attention.head_count": 8,
        "attention.head_count_kv": 4,
        "attention.key_length": 32,
        "attention.value_length": 32,
        "attention.sliding_window": 4,
        "attention.sliding_window_pattern": [True, False] * (4 if legacy else 3),
        "attention.layer_norm_rms_epsilon": 1e-6,
        "attention.relative_length": 2,
        "attention.relative_extent": 8,
        "attention.qk_clip": 22.0,
        "attention.relative_clip": 8.0,
        "attention.log_scale_floor": 16,
        "attention.log_scale_alpha": 0.1,
        "activation.situ_beta": 4.0,
        "activation.situ_linear_beta": 25.0,
        "expert_count": 2,
        "expert_used_count": 1,
        "expert_weights_scale": 8.0,
        "vocab_size": 8,
        "vocab_size_unpadded": 7,
        "expert_feed_forward_length": 256,
        "attn_res.block_size": 2,
        "shortconv.l_cache": 4,
        "final_logit_softcapping": 30.0,
        "loop.gate_rank": 4,
    }
    if legacy:
        settings.update({"num_loops": 2, "loop.core_start": 2, "loop.core_end": 4})
    else:
        settings.update({"loop.count": 2, "loop.start": 2, "loop.end": 4})
    metadata = {"general.architecture": architecture, **{prefix + k: v for k, v in settings.items()}}
    if not legacy:
        metadata["tokenizer.ggml.tokens"] = [str(i) for i in range(7)]
    tensors = []

    def add(name, shape, dtype=Q.F32):
        tensors.append(tensor_info(name, shape, dtype))

    for root in ("token_embd", "output"):
        add(root + ".weight", (8, 256), Q.Q8_0)
    for root in ("token_embd_norm", "output_norm", "output_res_query"):
        add(root + ".weight", (256,))
    if legacy:
        for root in ("loop_res_query", "loop_kv_res_query"):
            add(root + ".weight", (4, 256))
    for i in range(6):
        b = f"blk.{i}."
        local = i % 2 == 0
        for name, rows in (("attn_q", 256), ("attn_k", 128), ("attn_v", 128),
                           ("attn_gate", 256), ("attn_output", 256), ("attn_rel", 16)):
            add(b + name + ".weight", (rows, 256), Q.Q8_0)
        for name, shape in {
            "attn_q_clip": (8, 32), "attn_rel_clip": (8, 2), "attn_rel_bias": (4 if local else 8, 2),
            "attn_k_conv": (128, 4), "attn_v_conv": (128, 4), "attn_post_conv": (256, 4),
            "ffn_post_conv": (256, 4), "ffn_norm": (256,), "ffn_gate_inp": (2, 256),
            "ffn_res_query": (256,),
        }.items():
            add(b + name + ".weight", shape)
        add(b + "attn_sink_proj.weight", (8, 256), Q.BF16)
        if i:
            add(b + "attn_res_query.weight", (256,) if local else (3, 256))
        if local:
            add(b + "attn_norm.weight", (256,))
        else:
            for stem in ("q", "k", "v"):
                add(b + f"attn_{stem}_inp_norm.weight", (256,))
            for stem in ("k", "v"):
                add(b + f"attn_{stem}_conv_proj.weight", (16, 256), Q.BF16)
        if i in (2, 3):
            if not legacy:
                add(b + "loop_msg_query.weight", (4, 256))
            for name in ("attn", "k", "v", "mlp") if legacy else ("q_ctrl", "k_ctrl", "v_ctrl", "ffn_ctrl"):
                for op, shape in (("down", (4, 512)), ("forget", (256, 4)), ("gain", (256, 4))):
                    add(b + f"loop_{name}_{op}.weight", shape)
        for name, rows in (("gate_up", 512), ("down", 256)):
            dtype = Q.NVFP4 if i < 4 else Q.Q8_0 if legacy or name == "down" else Q.Q5_K
            add(b + f"ffn_{name}_exps.weight", (2, rows, 256), dtype)
            if dtype == Q.NVFP4:
                add(b + f"ffn_{name}_exps.scale", (2,))
    return GGUFModelInfo(Path("fixture.gguf"), 3, 32, metadata, tuple(tensors), 0)


@pytest.mark.parametrize("legacy", [True, False])
def test_visit_geometry_and_controller_layout(legacy):
    mapping = build_inkling2_gguf_map(_file(legacy))
    assert [v.weight_layer for v in mapping.plan.visits] == [0, 1, 2, 3, 2, 3, 4, 5]
    assert [v.state_slot for v in mapping.plan.visits] == (
        list(range(8)) if legacy else [0, 1, 2, 3, 6, 7, 4, 5]
    )
    assert [v.is_local for v in mapping.plan.visits] == [True, False] * 4
    assert [i for i, v in enumerate(mapping.plan.visits) if v.restart_from_prelude] == [4]
    assert mapping.layers[2].controllers[0].down.name == (
        "blk.2.loop_attn_down.weight" if legacy else "blk.2.loop_q_ctrl_down.weight"
    )
    assert mapping.layers[4].gate_up.weight.ggml_type == (Q.Q8_0 if legacy else Q.Q5_K)
    assert mapping.vocab_size == 7
    assert mapping.root["token_embd.weight"].shape[0] == 8


def test_loop_queries_are_read_in_q_k_v_ffn_order():
    for legacy in (True, False):
        reader = Mock()
        reader.info = _file(legacy)
        reader.tensor_data.side_effect = {
            "loop_res_query.weight": np.repeat(np.array([[0, 3, 4, 7]], dtype=np.float32).T, 256, axis=1),
            "loop_kv_res_query.weight": np.repeat(np.array([[1, 2, 5, 6]], dtype=np.float32).T, 256, axis=1),
            "blk.3.loop_msg_query.weight": np.repeat(np.arange(4, 8, dtype=np.float32)[:, None], 256, axis=1),
        }.__getitem__
        weights = Inkling2GGUFWeights(reader)
        np.testing.assert_array_equal(weights.loop_queries(weights.mapping.plan.visits[5])[:, 0], np.arange(4, 8))


@pytest.mark.parametrize("legacy", [True, False])
def test_repeated_visit_projects_with_the_physical_expert_scale(legacy):
    reader = Mock()
    reader.info = _file(legacy)
    # NVFP4: UE4M3 1.0, nibble 2 (value 1.0).
    block = np.array([0x38] * 4 + [0x22] * 32, dtype=np.uint8)
    raw = np.tile(block, (2, 256, 4))
    raw[0].fill(0)
    reader.tensor_data.side_effect = {"blk.2.ffn_down_exps.weight": raw,
                                      "blk.2.ffn_down_exps.scale": np.array([2, 3], np.float32)}.__getitem__
    weights = Inkling2GGUFWeights(reader)
    visits = weights.mapping.plan.visits
    for visit in (visits[2], visits[4]):
        result = weights.project_expert(visit, "down", 1, np.ones(256, np.float32), rows=slice(2, 5))
        np.testing.assert_array_equal(result, np.full(3, 768, np.float32))
    assert {call.args[0] for call in reader.tensor_data.call_args_list} == {
        "blk.2.ffn_down_exps.weight", "blk.2.ffn_down_exps.scale",
    }


def test_rejects_bad_weight_shape_and_conflicting_legacy_slot_metadata():
    info = _file(True)
    name = "blk.2.ffn_down_exps.scale"
    changed = tuple(replace(t, shape=(3,)) if t.name == name else t for t in info.tensors)
    with pytest.raises(ValueError, match=name):
        build_inkling2_gguf_map(replace(info, tensors=changed))
    unscaled = build_inkling2_gguf_map(replace(info, tensors=tuple(t for t in info.tensors if t.name != name)))
    assert unscaled.layers[2].down.scale is None
    pattern = "inkling2.attention.sliding_window_pattern"
    local = info.metadata[pattern].copy()
    local[4] = False
    with pytest.raises(ValueError, match="sliding_window_pattern"):
        build_inkling2_gguf_map(replace(info, metadata={**info.metadata, pattern: local}))


def test_rejects_inconsistent_tokenizer_size():
    info = _file(False)
    with pytest.raises(ValueError, match="vocabulary"):
        build_inkling2_gguf_map(replace(info, metadata={**info.metadata, "tokenizer.ggml.tokens": []}))
