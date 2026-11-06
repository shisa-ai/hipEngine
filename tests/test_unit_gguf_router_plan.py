"""Construction snapshot and explicit diagnostic router precedence."""
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_runner as qgr


@pytest.mark.parametrize('coop,persistent', [(False, False), (True, False), (True, True)])
def test_router_plan_snapshot_and_explicit_diagnostic(monkeypatch, coop, persistent):
    owner = SimpleNamespace(router_f32w_coop=coop, router_f32w_persistent_counter=persistent)
    owner.router_plan = qgr._resolve_gguf_router_plan(owner)
    owner.router_f32w_coop = None
    owner.router_f32w_persistent_counter = None
    monkeypatch.setenv('HIPENGINE_GGUF_ROUTER_F32W_COOP', '0' if coop else '1')
    flags = qgr._layer_ffn_flags(owner)
    assert flags.router_f32w_coop_enabled() is coop
    assert flags.router_f32w_persistent_counter_enabled() is persistent
    owner.router_f32w_coop = not coop
    assert flags.router_f32w_coop_enabled() is not coop
    assert owner.router_plan.cooperative is coop
    assert (owner.router_plan.counter_owner != 'none') is (coop and persistent)


def test_direct_router_environment_is_snapshotted(monkeypatch):
    owner = SimpleNamespace(router_f32w_coop=None, router_f32w_persistent_counter=None)
    monkeypatch.setenv('HIPENGINE_GGUF_ROUTER_F32W_COOP', '0')
    plan = qgr._resolve_gguf_router_plan(owner)
    monkeypatch.setenv('HIPENGINE_GGUF_ROUTER_F32W_COOP', '1')
    assert plan.variant == 'coop_out_bf16_hidden_persistent'
