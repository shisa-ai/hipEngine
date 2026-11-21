import numpy as np
import pytest

from hipengine.runtime.inkling2_mini_reference import Inkling2TextReference


class TinyWeights:
    def dense(self, name, selector=...):
        if name in ("embed.weight", "unembed.weight"):
            return np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)[selector]
        if name.endswith("rel_l2_clip.scale"):
            return np.ones(1, dtype=np.float32)
        if name.endswith(("norm.weight", "q_l2_clip.scale")):
            return np.ones(2, dtype=np.float32)
        if name.endswith("middle_cycle_cross_pass_queries") or name.endswith("middle_cycle_cross_pass_kv_queries"):
            return np.zeros((2, 2), dtype=np.float32)
        if ".controllers." in name and name.endswith("down_weight"):
            return np.zeros((1, 4), dtype=np.float32)
        if ".controllers." in name:
            return np.zeros((2, 1), dtype=np.float32)
        if "attn_res." in name:
            return np.zeros((3, 2) if name.endswith("attn_queries.0") else (2,), dtype=np.float32)
        if name.endswith(("wr_du.weight", "attn_sink_proj.weight", "mlp.gate.weight")):
            return np.zeros((1, 2), dtype=np.float32)
        if name.endswith("attn.rel_logits_proj.proj"):
            return np.zeros((1, 2), dtype=np.float32)
        if name.endswith("mlp.gate.bias"):
            return np.zeros(1, dtype=np.float32)
        if name.endswith("_sconv.static_w"):
            return np.zeros((2, 2), dtype=np.float32)
        if name.endswith("_sconv.weight_proj"):
            return np.zeros((2, 2), dtype=np.float32)
        if name.endswith("_sconv.weight"):
            return np.zeros((2, 1, 2), dtype=np.float32)
        if name.endswith(("wq_du.weight", "wk_dv.weight", "wv_dv.weight", "wo_ud.weight", "gate_attn.weight")):
            return np.eye(2, dtype=np.float32)
        raise AssertionError(name)

    def expert(self, layer, projection, index):
        return np.zeros((2, 2) if projection == "w13" else (2, 1), dtype=np.float32)


def _config():
    return {"text_config": {
        "num_hidden_layers": 2, "local_layer_ids": [0],
        "loop_config": {"core_start": 1, "core_end": 2, "num_passes": 2},
        "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 2,
        "swa_num_attention_heads": 1, "swa_num_key_value_heads": 1, "swa_head_dim": 2,
        "rms_norm_eps": 1e-6, "n_layers_per_attn_res_block": 1, "sconv_kernel_size": 2,
        "sliding_window_size": 2, "rel_extent": 2, "qk_l2_clip_threshold": 8,
        "rel_l2_clip_threshold": 8, "log_scaling_alpha": 0.1, "log_scaling_n_floor": 2,
        "num_experts_per_tok": 1, "route_scale": 8, "mlp_gate_softcap": 4, "mlp_up_softcap": 25,
        "unpadded_vocab_size": 2, "vocab_size": 3, "final_logit_softcapping": 30,
    }}


def test_reference_decodes_causally_with_separate_local_and_loop_states():
    reference = Inkling2TextReference(_config(), TinyWeights())
    first = reference.session(3)
    for token in [0, 1, 0]:
        logits = first.step(token)
    assert logits.shape == (2,)
    assert logits.dtype == np.float32 and np.isfinite(logits).all()
    assert logits[0] > logits[1]
    assert [s.keys.shape[0] for s in first.states] == [2, 3, 3]
    assert not np.shares_memory(first.states[1].keys, first.states[2].keys)
    with pytest.raises(ValueError, match="capacity"):
        first.step(0)
    other = reference.session(3)
    for token in [0, 1, 0]:
        same = other.step(token)
    np.testing.assert_array_equal(same, logits)
    fresh = reference.session(1).step(0)
    assert fresh[1] == 0 and logits[1] > 0


def test_layer_major_block_matches_serial_logits_taps_and_followup():
    model = Inkling2TextReference(_config(), TinyWeights())
    a, b = model.session(7), model.session(7)
    np.testing.assert_array_equal(a.step(0), b.step(0))
    per_token = []
    serial = []
    for token in (1, 0, 1):
        taps = {}
        serial.append(a.step(token, capture=lambda i, h: taps.update({i: h.copy()})))
        per_token.append(taps)
    block_taps = {}
    block = b.step_block((1, 0, 1), capture=lambda i, h: block_taps.update({i: h.copy()}))
    np.testing.assert_allclose(block, np.stack(serial), rtol=1e-6, atol=1e-6)
    for boundary, got in block_taps.items():
        np.testing.assert_allclose(got, np.stack([t[boundary] for t in per_token]), rtol=1e-6, atol=1e-6)
    assert a.position == b.position == 4
    np.testing.assert_allclose(a.step(0), b.step(0), rtol=1e-6, atol=1e-6)
