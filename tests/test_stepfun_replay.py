from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hipengine.runtime.stepfun_replay import (
    StepFunAttentionReplaySpec,
    StepFunDenseMLPReplaySpec,
    StepFunLayerReplaySpec,
    StepFunMoEMLPReplaySpec,
    StepFunReplayMismatch,
    StepFunReplayTolerance,
    assert_stepfun_replay_close,
    capture_stepfun_replay_reference,
    replay_stepfun_layer,
)


def _attention_spec(rng: np.random.Generator, *, hidden: int, hq: int, hkv: int, d: int, sliding_window: int | None) -> StepFunAttentionReplaySpec:
    return StepFunAttentionReplaySpec(
        q_weight=rng.normal(scale=0.05, size=(hq * d, hidden)).astype(np.float32),
        k_weight=rng.normal(scale=0.05, size=(hkv * d, hidden)).astype(np.float32),
        v_weight=rng.normal(scale=0.05, size=(hkv * d, hidden)).astype(np.float32),
        o_weight=rng.normal(scale=0.05, size=(hidden, hq * d)).astype(np.float32),
        gate_weight=rng.normal(scale=0.05, size=(hq, hidden)).astype(np.float32),
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_dim=d,
        rope_partial_factor=0.5 if sliding_window is None else 1.0,
        rope_theta=5_000_000.0 if sliding_window is None else 10_000.0,
        rope_llama3_scaling=sliding_window is None,
        sliding_window=sliding_window,
    )


def _dense_spec(rng: np.random.Generator, *, hidden: int, intermediate: int) -> StepFunDenseMLPReplaySpec:
    return StepFunDenseMLPReplaySpec(
        gate_weight=rng.normal(scale=0.08, size=(intermediate, hidden)).astype(np.float32),
        up_weight=rng.normal(scale=0.08, size=(intermediate, hidden)).astype(np.float32),
        down_weight=rng.normal(scale=0.08, size=(hidden, intermediate)).astype(np.float32),
    )


def _moe_spec(rng: np.random.Generator, *, hidden: int, intermediate: int, experts: int, top_k: int) -> StepFunMoEMLPReplaySpec:
    router_bias = np.zeros((experts,), dtype=np.float32)
    router_bias[[1, 5, experts - 1]] = np.asarray([1.0, 0.75, 0.5], dtype=np.float32)
    return StepFunMoEMLPReplaySpec(
        router_weight=rng.normal(scale=0.04, size=(experts, hidden)).astype(np.float32),
        router_bias=router_bias,
        expert_gate_weight=rng.normal(scale=0.08, size=(experts, intermediate, hidden)).astype(np.float32),
        expert_up_weight=rng.normal(scale=0.08, size=(experts, intermediate, hidden)).astype(np.float32),
        expert_down_weight=rng.normal(scale=0.08, size=(experts, hidden, intermediate)).astype(np.float32),
        shared_gate_weight=rng.normal(scale=0.08, size=(intermediate, hidden)).astype(np.float32),
        shared_up_weight=rng.normal(scale=0.08, size=(intermediate, hidden)).astype(np.float32),
        shared_down_weight=rng.normal(scale=0.08, size=(hidden, intermediate)).astype(np.float32),
        top_k=top_k,
        routing_scale=3.0,
        swiglu_limit=7.0,
        shared_swiglu_limit=16.0,
    )


def _layer_spec(rng: np.random.Generator, *, name: str, hidden: int, attention: StepFunAttentionReplaySpec, mlp: StepFunDenseMLPReplaySpec | StepFunMoEMLPReplaySpec) -> StepFunLayerReplaySpec:
    return StepFunLayerReplaySpec(
        name=name,
        input_norm_weight=rng.normal(scale=0.02, size=(hidden,)).astype(np.float32),
        ffn_norm_weight=rng.normal(scale=0.02, size=(hidden,)).astype(np.float32),
        attention=attention,
        mlp=mlp,
    )


def test_stepfun_replay_dense_layer_captures_cpu_reference_stages() -> None:
    rng = np.random.default_rng(100)
    hidden = 8
    spec = _layer_spec(
        rng,
        name="dense-full-layer",
        hidden=hidden,
        attention=_attention_spec(rng, hidden=hidden, hq=4, hkv=2, d=4, sliding_window=None),
        mlp=_dense_spec(rng, hidden=hidden, intermediate=12),
    )
    hidden_states = rng.normal(size=(2, hidden)).astype(np.float32)

    replay = replay_stepfun_layer(hidden_states, spec, positions=np.asarray([0, 1], dtype=np.int64))
    reference = capture_stepfun_replay_reference(replay)

    assert replay.output.shape == (2, hidden)
    assert replay.stage("q_rope").shape == (2, 4, 4)
    assert replay.stage("attention_output").shape == (2, hidden)
    assert replay.stage("ffn").shape == (2, hidden)
    assert_stepfun_replay_close(replay, reference, tolerance=StepFunReplayTolerance(atol=1e-5, rtol=1e-5))


def test_stepfun_replay_full_and_sliding_moe_layers_with_stage_gates() -> None:
    rng = np.random.default_rng(200)
    hidden = 8
    full_spec = _layer_spec(
        rng,
        name="full-attention-moe-layer",
        hidden=hidden,
        attention=_attention_spec(rng, hidden=hidden, hq=4, hkv=2, d=4, sliding_window=None),
        mlp=_moe_spec(rng, hidden=hidden, intermediate=6, experts=12, top_k=3),
    )
    sliding_spec = _layer_spec(
        rng,
        name="sliding-attention-moe-layer",
        hidden=hidden,
        attention=_attention_spec(rng, hidden=hidden, hq=6, hkv=2, d=4, sliding_window=3),
        mlp=_moe_spec(rng, hidden=hidden, intermediate=6, experts=12, top_k=3),
    )
    hidden_states = rng.normal(size=(1, hidden)).astype(np.float32)
    key_cache = rng.normal(size=(1, 5, 2, 4)).astype(np.float32)
    value_cache = rng.normal(size=(1, 5, 2, 4)).astype(np.float32)

    full = replay_stepfun_layer(hidden_states, full_spec, positions=np.asarray([4], dtype=np.int64))
    sliding = replay_stepfun_layer(
        hidden_states,
        sliding_spec,
        positions=np.asarray([4], dtype=np.int64),
        key_cache=key_cache,
        value_cache=value_cache,
        live_counts=np.asarray([5], dtype=np.int64),
    )
    no_window = replay_stepfun_layer(
        hidden_states,
        replace(sliding_spec, attention=replace(sliding_spec.attention, sliding_window=None)),
        positions=np.asarray([4], dtype=np.int64),
        key_cache=key_cache,
        value_cache=value_cache,
        live_counts=np.asarray([5], dtype=np.int64),
    )

    assert full.stage("router_indices").shape == (1, 3)
    assert sliding.stage("router_weights").shape == (1, 3)
    assert sliding.stage("key_cache").shape == (1, 5, 2, 4)
    assert sliding.output.shape == (1, hidden)
    assert not np.allclose(sliding.stage("attention"), no_window.stage("attention"))
    assert_stepfun_replay_close(full, capture_stepfun_replay_reference(full))
    assert_stepfun_replay_close(sliding, capture_stepfun_replay_reference(sliding))


def test_stepfun_replay_mismatch_reports_first_substage() -> None:
    rng = np.random.default_rng(300)
    hidden = 8
    spec = _layer_spec(
        rng,
        name="dense-mismatch-layer",
        hidden=hidden,
        attention=_attention_spec(rng, hidden=hidden, hq=4, hkv=2, d=4, sliding_window=None),
        mlp=_dense_spec(rng, hidden=hidden, intermediate=12),
    )
    replay = replay_stepfun_layer(
        rng.normal(size=(1, hidden)).astype(np.float32),
        spec,
        positions=np.asarray([0], dtype=np.int64),
    )
    reference = capture_stepfun_replay_reference(replay)
    reference["ffn_norm"] = reference["ffn_norm"].copy()
    reference["ffn_norm"][0, 0] += np.float32(0.25)

    with pytest.raises(StepFunReplayMismatch, match="ffn_norm") as excinfo:
        assert_stepfun_replay_close(replay, reference, tolerance=StepFunReplayTolerance(atol=1e-5, rtol=1e-5))

    assert excinfo.value.stage == "ffn_norm"
