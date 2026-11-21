from math import sqrt

import numpy as np
import pytest

from hipengine.runtime.inkling2_mini_reference import Inkling2TextReference, _read_blocks, _rms, _softmax


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


@pytest.mark.parametrize("shape", [(65,), (3, 2048)])
@pytest.mark.parametrize("queries", [None, 4])
def test_read_blocks_reuses_completed_norms_without_rounding_changes(shape, queries):
    rng = np.random.default_rng(8)
    blocks = [rng.standard_normal(shape, dtype=np.float32) for _ in range(5)]
    partial = rng.standard_normal(shape, dtype=np.float32)
    width = shape[-1]
    query = rng.standard_normal((width,) if queries is None else (queries, width), dtype=np.float32)
    values = _rms(np.stack((*blocks, partial), axis=-2))
    if partial.ndim == 1:
        expected = _softmax((query @ values.T) / sqrt(width)) @ values
    elif query.ndim == 1:
        scores = values @ query / sqrt(width)
        expected = np.einsum("bp,bpd->bd", _softmax(scores), values)
    else:
        scores = query @ values.swapaxes(-1, -2) / sqrt(width)
        expected = _softmax(scores) @ values
    actual = _read_blocks(query, [_rms(block) for block in blocks], partial)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


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


def test_prefill_block_keeps_state_and_only_projects_final_row():
    model = Inkling2TextReference(_config(), TinyWeights())
    serial, packed = model.session(8), model.session(8)
    tokens = (0, 1, 2, 0, 1)
    for token in tokens:
        expected = serial.step(token)
    project = model.project_logits
    shapes = []

    def track(x):
        shapes.append(x.shape)
        return project(x)

    model.project_logits = track
    assert packed.prefill_block(tokens[:3], logits=False) is None
    assert not shapes
    taps = {}
    actual = packed.prefill_block(tokens[3:], capture=lambda i, h: taps.update({i: h.copy()}))
    assert shapes == [(2,)]
    assert all(hidden.shape == (2, 2) for hidden in taps.values())
    assert packed.position == serial.position == len(tokens)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(packed.step(0), serial.step(0), atol=1e-6)


@pytest.mark.parametrize("keep", [0, 1, 3])
def test_provisional_block_rolls_back_rejected_suffix_across_multiple_wraps(keep):
    model = Inkling2TextReference(_config(), TinyWeights())
    serial, packed = model.session(10), model.session(10)
    for token in (0, 1, 1):
        serial.step(token)
        packed.step(token)
    proposed = (1, 0, 1, 1, 1)
    with packed.provisional_block(proposed) as (logits, accept):
        assert len(logits) == len(proposed)
        accept(keep)
    for token in proposed[:keep]:
        serial.step(token)
    assert serial.position == packed.position == 3 + keep
    np.testing.assert_allclose(serial.step(0), packed.step(0), atol=1e-6)


def test_attention_message_controllers_group_projection_chains(monkeypatch):
    model = Inkling2TextReference(_config(), TinyWeights())
    project_chains = model.project_chains
    observed = []

    def track(chains):
        chains = tuple(chains)
        observed.append(tuple((first.split(".controllers.")[1], *(name.split(".controllers.")[1] for name in names))
                              for first, names, _ in chains))
        return project_chains(chains)

    monkeypatch.setattr(model, "project_chains", track)
    model.session(4).step_block((0, 1))
    assert observed == [
        tuple((f"{name}.down_weight", f"{name}.forget_weight", f"{name}.gain_weight") for name in ("attn", "k", "v")),
        (("mlp.down_weight", "mlp.forget_weight", "mlp.gain_weight"),),
    ]


def test_provisional_block_failure_discards_even_selected_prefix():
    model = Inkling2TextReference(_config(), TinyWeights())
    reference, packed = model.session(6), model.session(6)
    with pytest.raises(RuntimeError, match="rejected transaction"):
        with packed.provisional_block((1, 0, 1)) as (_, accept):
            accept(2)
            raise RuntimeError("rejected transaction")
    assert packed.position == 0
    assert all(not visit.conv for visit in packed.states)
    np.testing.assert_array_equal(reference.step(0), packed.step(0))
