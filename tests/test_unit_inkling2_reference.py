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


@pytest.mark.parametrize("mode", ["scalar", "matrix", "strided", "promoted", "broadcast"])
def test_controller_blend_keeps_numpy_bits_and_inputs(mode):
    from hipengine.runtime.inkling2_mini_reference import _blend_controller

    data = np.random.default_rng(92).standard_normal((4, 7, 146), dtype=np.float32)
    data[0, 0, :3] = (0.0, -0.0, 0.0)
    x, message, forget, gain = data
    if mode == "scalar":
        x, message, forget, gain = data[:, 0]
    elif mode == "strided":
        x, message, forget, gain = data[..., ::-2]
    elif mode == "promoted":
        message = message.astype(np.float64)
    elif mode == "broadcast":
        gain = gain[:1]
    for row in (x, message, forget, gain):
        row.setflags(write=False)
    expected = (1 - np.tanh(forget)) * x + np.tanh(gain) * message
    actual = _blend_controller(x, message, forget, gain)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("shape", [(65,), (8, 128), (3, 2048)])
@pytest.mark.parametrize("stride", [1, 2])
def test_rms_preserves_numpy_mean_reduction_bits(shape, stride):
    rng = np.random.default_rng(94)
    rows = rng.standard_normal((*shape[:-1], shape[-1] * stride), dtype=np.float32)[..., ::stride]
    eps = 7e-6
    expected = rows / np.sqrt(np.mean(rows * rows, axis=-1, keepdims=True) + eps)
    np.testing.assert_array_equal(_rms(rows, eps).view(np.uint32), expected.view(np.uint32))


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


@pytest.mark.parametrize("shape", [(65,), (2, 65)])
@pytest.mark.parametrize("queries", [None, 4])
def test_residual_reader_preserves_bits_when_switching_prefixes(shape, queries):
    from hipengine.runtime.inkling2_mini_reference import _ResidualRows

    rng = np.random.default_rng(82)
    partial = rng.standard_normal(shape, dtype=np.float32)
    rows = [_rms(rng.standard_normal(shape, dtype=np.float32)) for _ in range(5)]
    extra = _rms(rng.standard_normal(shape, dtype=np.float32))
    query = rng.standard_normal((shape[-1],) if queries is None else (queries, shape[-1]), dtype=np.float32)
    reader = _ResidualRows(partial, 6)
    for blocks in (rows[:2], rows[:4], rows[:1], [extra, *rows[2:]], rows, []):
        want = _read_blocks(query, blocks, partial)
        got = reader.read(query, blocks, partial)
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
    # A fallback after a row has already changed must not keep stale sources.
    promoted = [extra, rows[1].astype(np.float64)]
    np.testing.assert_array_equal(reader.read(query, promoted, partial), _read_blocks(query, promoted, partial))
    np.testing.assert_array_equal(reader.read(query, rows[:2], partial), _read_blocks(query, rows[:2], partial))


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


@pytest.mark.parametrize("mode", ["ordinary", "staged", "block"])
def test_distinct_local_inputs_consume_static_kv_before_download(mode):
    model = Inkling2TextReference(_config(), TinyWeights())
    session = model.session(3)
    visit = model.plan.visits[0]
    assert visit.is_local
    consumed = []

    class Convolution:
        def apply_static(self, slot, name, taps, ptr, shape, *, position):
            consumed.append((slot, name, ptr, shape, position))

        def step_kv(self, *args, **kwargs):
            pytest.fail("distinct local inputs used a second convolution exchange")

    def project(groups, transforms):
        outputs = model.project_groups(groups)
        assert transforms[0] is None
        for ptr, transform, (row,) in zip((1, 2), transforms[1:], outputs[1:], strict=True):
            transform((ptr, row.shape))
        return outputs

    session.convolution = Convolution()
    model.project_groups_transformed = project
    eye = np.eye(2, dtype=np.float32)
    q, k, v = eye[0], eye[1], np.ones(2, dtype=np.float32)
    if mode == "block":
        q, k, v = (np.stack((row, row)) for row in (q, k, v))
    layer = f"layers.{visit.weight_layer}."
    traversal = session._attention(visit, session.states[visit.state_slot], lambda name: model.weights.dense(layer + name),
                                   q, k, v, stage_projections=mode == "staged")
    if mode == "staged":
        actual_visit, kind, (groups, transforms) = next(traversal)
        assert (actual_visit, kind) == (visit, "attention_inputs")
        with pytest.raises(StopIteration):
            traversal.send(project(groups, transforms))
    else:
        with pytest.raises(StopIteration):
            next(traversal)
    assert consumed == [(visit.state_slot, "k", 1, k.shape, 0), (visit.state_slot, "v", 2, v.shape, 0)]


def test_staged_decode_keeps_independent_positions_and_scalar_math():
    model = Inkling2TextReference(_config(), TinyWeights())
    controls = [model.session(6) for _ in range(2)]
    staged = [model.session(6) for _ in range(2)]
    for expected, actual, prompt in zip(controls, staged, ((0,), (1, 2, 1)), strict=True):
        for token in prompt:
            np.testing.assert_array_equal(actual.step(token), expected.step(token))
    streams = [session.stage_decode(token) for session, token in zip(staged, (2, 0), strict=True)]
    jobs = [next(stream) for stream in streams]
    for visit in model.plan.visits:
        assert all(current == visit for current, _, _ in jobs)
        assert [s.position for s in staged] == [1, 3]
        if visit.message_pass is not None:
            assert all(kind == "controllers" and len(chains) == 3 for _, kind, chains in jobs)
            jobs = [stream.send(model.project_chains(chains)) for stream, (_, _, chains) in zip(streams, jobs, strict=True)]
        assert all(kind == "attention_inputs" and transforms is None for _, kind, (_, transforms) in jobs)
        jobs = [stream.send(model.project_groups(groups)) for stream, (_, _, (groups, _)) in zip(streams, jobs, strict=True)]
        if visit.message_pass is not None:
            assert all(kind == "controllers" and len(chains) == 1 for _, kind, chains in jobs)
            jobs = [stream.send(model.project_chains(chains)) for stream, (_, _, chains) in zip(streams, jobs, strict=True)]
        assert all(kind == "experts" for _, kind, _ in jobs)
        jobs = [stream.send(model.feed_forward(visit, hidden, transform_output=consume))
                for stream, (_, _, (hidden, consume)) in zip(streams, jobs, strict=True)]
    for expected, actual, hidden, token, stream in zip(controls, staged, jobs, (2, 0), streams, strict=True):
        np.testing.assert_array_equal(model.cap_logits(model.project_logits(hidden)), expected.step(token))
        with pytest.raises(StopIteration):
            next(stream)
        assert actual.position == expected.position
        np.testing.assert_array_equal(actual.step(1), expected.step(1))


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


def test_capture_preserves_the_ordinary_prefill_logit_shape_and_values():
    model = Inkling2TextReference(_config(), TinyWeights())
    ordinary, captured = model.session(8), model.session(8)
    finalize = captured._output_hidden
    taps = {}

    def shape_sensitive(h, blocks):
        # Model a different reduction order when all rows are normalized together.
        return finalize(h, blocks) + (np.float32(0.125) if h.ndim == 2 else np.float32(0.0))

    captured._output_hidden = shape_sensitive
    tokens = (0, 1, 2)
    want = ordinary.prefill_block(tokens)
    got = captured.prefill_block(tokens, capture=lambda i, h: taps.update({i: h.copy()}))
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
    assert taps[len(model.plan.visits)].shape == (3, 2)
    assert ordinary.position == captured.position == 3


def test_capture_points_are_not_visit_exit_states():
    """`capture(k, h)` observes entry to visit ``k``, and for a restart visit
    that is the restored prelude state, not the prior visit's exit. Comparing
    a captured entry against a truncated run's raw exit misreads the restart
    as a divergence. Guard the three points that must not be conflated."""
    from contextlib import closing
    from dataclasses import replace

    cfg, ids = _config(), [0, 1]
    reference = Inkling2TextReference(cfg, TinyWeights())
    h0 = _rms(reference.weights.dense('embed.weight', ids), reference.config['rms_norm_eps']) * reference.weights.dense('embed_norm.weight')

    def exit_at(n):
        plan = replace(reference.plan, visits=reference.plan.visits[:n])
        with closing(Inkling2TextReference(cfg, TinyWeights(), plan=plan).session(4)) as session:
            try:
                next(session._traverse(h0.copy(), None))
            except StopIteration as done:
                raw, blocks, _ = done.value
            else:
                raise AssertionError('unexpected staged traversal')
            return raw, session._output_hidden(raw, blocks)

    with closing(reference.session(4)) as session:
        seen = {}
        session.prefill_block(ids, capture=lambda i, h: seen.setdefault(i, h.copy()))

    visits = reference.plan.visits
    entry = next(i for i, visit in enumerate(visits) if visit.weight_layer == reference.config['loop_config']['core_start'])
    restart = next(i for i, visit in enumerate(visits) if visit.restart_from_prelude)
    # Restart re-enters with the same prelude state captured at the core entry.
    np.testing.assert_array_equal(seen[restart], seen[entry])
    # Truncating before the restart keeps the first pass's exit, which is not the restored entry.
    raw_before_restart, _ = exit_at(restart)
    assert not np.allclose(seen[restart], raw_before_restart)
    # The final capture observes the depth-mixed output, not the raw final residual.
    raw_final, final_hidden = exit_at(len(visits))
    np.testing.assert_array_equal(seen[len(visits)], final_hidden)
    assert not np.allclose(seen[len(visits)], raw_final)


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
