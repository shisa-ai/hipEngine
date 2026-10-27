"""The GDN peer prefill owner's layer scope is request-local runner state.

The peer owner reads the compact Q/K/V projections inside a published global
layer scope, certified at layers 35-47. Every layer outside that scope, and
every plan that publishes no scope, keeps the columnwarp parent and the serial
sigmoid owner registered below both. The two process keys that used to carry the
gate and the scope are retired: a diagnostic arm binds the scope on the runner
it measures, and an all-layer or suffix arm is that same binding with a
different scope.
"""
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as runner

RETIRED_GATE_ENV = "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL"
RETIRED_SCOPE_ENV = "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL_LAYERS"


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.delenv(RETIRED_GATE_ENV, raising=False)
    monkeypatch.delenv(RETIRED_SCOPE_ENV, raising=False)


def weight(layer):
    return SimpleNamespace(spec=SimpleNamespace(slot_path=f"layers.{layer}.attn_qkv"))


def allowed(layer, *, published):
    return runner._qwen4_exp_gdn_peer_layer_allowed(
        {"attn_qkv": weight(layer)}, published=published
    )


def test_the_certified_scope_owns_layers_35_to_47():
    scope = runner.GDN_PEER_CERTIFIED_SCOPE
    assert [layer for layer in range(48) if allowed(layer, published=scope)] == list(
        range(35, 48)
    )


def test_the_certified_scope_matches_the_profile_tuple():
    from hipengine.generation.qwen4_exp_profiles import (
        PRODUCTION_GDN_PEER_PREFILL_LAYERS,
    )

    assert runner.GDN_PEER_CERTIFIED_SCOPE == ",".join(
        map(str, PRODUCTION_GDN_PEER_PREFILL_LAYERS)
    )


def test_an_explicit_suffix_scope_owns_only_those_layers():
    assert [layer for layer in range(48) if allowed(layer, published="0-34")] == list(
        range(35)
    )
    assert [layer for layer in range(48) if allowed(layer, published="35,40,47")] == [
        35,
        40,
        47,
    ]


def test_the_all_layer_control_owns_every_layer():
    assert all(allowed(layer, published="all") for layer in range(48))


@pytest.mark.parametrize("published", [None, ""])
def test_an_unpublished_scope_binds_the_serial_owners(published):
    assert not any(allowed(layer, published=published) for layer in range(48))


@pytest.mark.parametrize(
    "slot_path",
    ["attn_qkv", "root.head_hc_down", "layers.35", "layers.x.attn_qkv"],
)
def test_non_layer_slot_paths_never_take_the_peer_owner(slot_path):
    assert runner._qwen4_exp_layer_scope_contains(slot_path, "all") is False


def test_the_peer_owner_needs_a_128_wide_head():
    assert (
        runner.qwen4_exp_gdn_peer_prefill_selected(head_dim=128, published="35-47")
        is True
    )
    assert (
        runner.qwen4_exp_gdn_peer_prefill_selected(head_dim=64, published="35-47")
        is False
    )
    assert runner.qwen4_exp_gdn_peer_prefill_selected(head_dim=128, published=None) is False


def test_retired_process_keys_cannot_select_the_peer_owner(monkeypatch):
    """The scope is runner state; setting either retired key changes nothing."""

    monkeypatch.setenv(RETIRED_GATE_ENV, "1")
    monkeypatch.setenv(RETIRED_SCOPE_ENV, "all")
    assert (
        runner.qwen4_exp_gdn_peer_prefill_selected(head_dim=128, published=None)
        is False
    )
    assert not allowed(35, published=None)


def test_runner_defaults_to_no_published_scope():
    assert runner.Qwen4ExpGGUFResidentModelRunner.gdn_peer_prefill_layers is None


def test_scope_binds_off_and_restores_the_published_scope():
    owner = SimpleNamespace(gdn_peer_prefill_layers="35-47")
    with runner.gdn_peer_prefill(owner, False) as prior:
        assert prior == "35-47"
        assert owner.gdn_peer_prefill_layers is None
    assert owner.gdn_peer_prefill_layers == "35-47"


def test_scope_restores_the_certified_scope_from_true():
    owner = SimpleNamespace(gdn_peer_prefill_layers=None)
    with runner.gdn_peer_prefill(owner, True):
        assert owner.gdn_peer_prefill_layers == runner.GDN_PEER_CERTIFIED_SCOPE


def test_scope_names_an_explicit_diagnostic_scope():
    owner = SimpleNamespace(gdn_peer_prefill_layers=None)
    with runner.gdn_peer_prefill(owner, "all"):
        assert owner.gdn_peer_prefill_layers == "all"
    with runner.gdn_peer_prefill(owner, "0-34"):
        assert owner.gdn_peer_prefill_layers == "0-34"


def test_scope_leaves_an_unset_binding_alone():
    owner = SimpleNamespace(gdn_peer_prefill_layers="35-47")
    with runner.gdn_peer_prefill(owner):
        assert owner.gdn_peer_prefill_layers == "35-47"
    assert owner.gdn_peer_prefill_layers == "35-47"


def test_scope_restores_after_an_exception():
    owner = SimpleNamespace(gdn_peer_prefill_layers="35-47")
    with pytest.raises(RuntimeError, match="boom"):
        with runner.gdn_peer_prefill(owner, "all"):
            raise RuntimeError("boom")
    assert owner.gdn_peer_prefill_layers == "35-47"


def test_scope_is_per_runner():
    first = SimpleNamespace(gdn_peer_prefill_layers="35-47")
    second = SimpleNamespace(gdn_peer_prefill_layers="35-47")
    with runner.gdn_peer_prefill(first, False):
        assert second.gdn_peer_prefill_layers == "35-47"
        with runner.gdn_peer_prefill(second, "all"):
            assert first.gdn_peer_prefill_layers is None
            assert second.gdn_peer_prefill_layers == "all"
        assert second.gdn_peer_prefill_layers == "35-47"
    assert first.gdn_peer_prefill_layers == "35-47"


def test_a_serving_clone_inherits_the_published_scope_not_a_diagnostic_binding():
    """The arm is per-runner: a clone built inside it keeps the published scope."""

    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    generator = SimpleNamespace(gdn_peer_prefill_layers="35-47")
    measured = SimpleNamespace(gdn_peer_prefill_layers="35-47")
    with runner.gdn_peer_prefill(measured, "all"):
        clone = _inherit_lane_decisions(generator, SimpleNamespace())
        assert clone.gdn_peer_prefill_layers == "35-47"
        assert measured.gdn_peer_prefill_layers == "all"
    assert measured.gdn_peer_prefill_layers == "35-47"


@pytest.mark.parametrize("scope", ["35-47", None])
def test_layer_forwards_the_scope_to_the_mixer(monkeypatch, scope):
    """The layer's only mixer call site carries the runner's scope."""

    seen = []
    runtime = object()

    def gr_read(*args, **kwargs):
        return SimpleNamespace(normalized=None, gate=None,
                               mixed=SimpleNamespace(ptr=11),
                               inject_logits=SimpleNamespace(ptr=12))

    def mixer(*args, **kwargs):
        seen.append(kwargs.get("gdn_peer_prefill_layers", "absent"))
        return SimpleNamespace(ptr=13)

    def moe(*args, **kwargs):
        return SimpleNamespace(output=SimpleNamespace(ptr=14))

    monkeypatch.setattr(runner, "run_qwen4_exp_gr_read", gr_read)
    monkeypatch.setattr(runner, "run_qwen4_exp_gdn_token_mixer", mixer)
    monkeypatch.setattr(runner, "run_qwen4_exp_moe", moe)
    monkeypatch.setattr(runner, "_qwen4_exp_gr_pending_write", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_qwen4_exp_gr_defer_write", lambda *a, **k: False)
    monkeypatch.setattr(runner, "qwen4_exp_gr_write_bf16_f32", lambda *a, **k: None)
    monkeypatch.setattr(runner, "qwen4_exp_gr_write_bf16_bf16", lambda *a, **k: None)
    output = SimpleNamespace(ptr=15)
    scratch = SimpleNamespace(
        closed=False, runtime=runtime, attention_gr=SimpleNamespace(),
        gdn=SimpleNamespace(), after_attention=SimpleNamespace(ptr=16),
        ffn_gr=SimpleNamespace(), moe=SimpleNamespace(), output=output)
    weights = SimpleNamespace(
        attention_gr=SimpleNamespace(norm_weight_ptr=1, down=2, up=3, inject=4),
        mixer=SimpleNamespace(projections={}, conv_weight_ptr=5, dt_bias_ptr=6,
                              a_ptr=7, norm_weight_ptr=8),
        ffn_gr=SimpleNamespace(norm_weight_ptr=9, down=10, up=11, inject=12),
        moe=SimpleNamespace())
    kwargs = dict(conv_state_ptr=0, recurrent_state_ptr=0, scratch=scratch,
                  rows=4, branches=1, hidden=2, low_rank=2, num_k_heads=16,
                  num_v_heads=48, head_dim=128, conv_kernel=4, ffn=8,
                  experts=1, top_k=1, runtime=runtime)
    assert runner.run_qwen4_exp_gdn_layer(
        100, weights, **kwargs, gdn_peer_prefill_layers=scope) is output
    assert runner.run_qwen4_exp_gdn_layer(100, weights, **kwargs) is output
    assert seen == [scope, None]
