"""The f16 WMMA layer scope is owner state, not a process key.

An empty scope means every layer, ``0`` is a real layer-0 selection, and an
explicit diagnostic override is distinct from both. The owning generator and
runner publish one typed scope, and the dispatch memo and weight filter both
read it, so a teacher arm cannot replay the candidate arm's dispatch.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.runtime import gguf_linear as gl
from hipengine.runtime import qwen4_exp_runner as runner_module
from hipengine.runtime.gguf_linear import (
    Q8WMMALayerScope,
    q8_wmma_layer_scope,
    q8_wmma_layer_scope_session,
    resolve_q8_wmma_layers,
)

ENV = gl._Q8_WMMA_LAYERS_ENV


class _Spec:
    layout = "raw_gguf"
    quant_key = "gguf_q8_0"

    def __init__(self, layer: int = 20) -> None:
        self.slot_path = f"layers.{layer}.attn_qkv.weight"


class _Allocation:
    class tensor:  # noqa: N801 - mirrors the resident-weight attribute path
        ptr = 4096


class _Weight:
    backend = "hip_gfx1151"

    def __init__(self, layer: int = 20, quant: str = "gguf_q8_0") -> None:
        self.spec = _Spec(layer)
        self.spec.quant_key = quant

    @staticmethod
    def allocation(name):
        assert name == "raw"
        return _Allocation()


def test_scope_parse_distinguishes_empty_layer0_and_an_override() -> None:
    empty = Q8WMMALayerScope.parse("")
    assert empty.is_empty
    assert empty.to_env() == ""
    assert empty.to_report() == {"mode": "empty", "layers": None}

    layer0 = Q8WMMALayerScope.parse("0")
    assert not layer0.is_empty
    assert layer0.to_env() == "0"
    assert layer0.to_report() == {"mode": "layer0", "layers": [0]}

    scoped = Q8WMMALayerScope.parse("16,17,18")
    assert scoped.layers == frozenset({16, 17, 18})
    assert scoped.to_env() == "16,17,18"
    assert scoped.to_report() == {"mode": "layers", "layers": [16, 17, 18]}

    with pytest.raises(ValueError):
        Q8WMMALayerScope.parse("16-47")


def test_bound_owner_precedes_ambient_key_and_never_rereads_it(monkeypatch) -> None:
    monkeypatch.setenv(ENV, "32")
    owner = SimpleNamespace(q8_wmma_layer_scope=None)
    assert resolve_q8_wmma_layers(owner).layers == frozenset({32})

    with q8_wmma_layer_scope(owner, Q8WMMALayerScope.parse("0")):
        assert resolve_q8_wmma_layers(owner).layers == frozenset({0})
        monkeypatch.setenv(ENV, "33")
        assert resolve_q8_wmma_layers(owner).layers == frozenset({0})

    assert owner.q8_wmma_layer_scope is None


def test_owner_binding_reaches_the_current_runner_and_restores() -> None:
    runner = SimpleNamespace()
    generator = SimpleNamespace(runner=runner)
    with q8_wmma_layer_scope(generator, Q8WMMALayerScope.empty()):
        assert generator.q8_wmma_layer_scope.is_empty
        assert runner.q8_wmma_layer_scope.is_empty
    assert not hasattr(generator, "q8_wmma_layer_scope")
    assert not hasattr(runner, "q8_wmma_layer_scope")


def test_nested_binding_restores_after_exception() -> None:
    owner = SimpleNamespace(q8_wmma_layer_scope=Q8WMMALayerScope.parse("20"))
    with pytest.raises(RuntimeError, match="failed"):
        with q8_wmma_layer_scope(owner, Q8WMMALayerScope.empty()):
            assert owner.q8_wmma_layer_scope.is_empty
            with q8_wmma_layer_scope(owner, Q8WMMALayerScope.parse("0")):
                assert owner.q8_wmma_layer_scope.layers == frozenset({0})
            assert owner.q8_wmma_layer_scope.is_empty
            raise RuntimeError("failed")
    assert owner.q8_wmma_layer_scope.layers == frozenset({20})


def test_independent_owners_do_not_share_a_scope() -> None:
    first = SimpleNamespace()
    second = SimpleNamespace()
    with q8_wmma_layer_scope(first, Q8WMMALayerScope.parse("0")):
        with q8_wmma_layer_scope(second, Q8WMMALayerScope.empty()):
            assert resolve_q8_wmma_layers(first).layers == frozenset({0})
            assert resolve_q8_wmma_layers(second).is_empty
        assert resolve_q8_wmma_layers(first).layers == frozenset({0})
        assert resolve_q8_wmma_layers(second).is_empty
    assert resolve_q8_wmma_layers(first).is_empty


def test_session_scope_is_authoritative_over_the_ambient_key(monkeypatch) -> None:
    monkeypatch.setenv(ENV, "32")
    with q8_wmma_layer_scope_session(Q8WMMALayerScope.parse("0")):
        assert gl._q8_wmma_layers_scope() == frozenset({0})
        monkeypatch.setenv(ENV, "33")
        assert gl._q8_wmma_layers_scope() == frozenset({0})
        with q8_wmma_layer_scope_session(Q8WMMALayerScope.empty()):
            assert gl._q8_wmma_layers_scope() is None
        assert gl._q8_wmma_layers_scope() == frozenset({0})
    assert gl._q8_wmma_layers_scope() == frozenset({33})


def test_weight_filter_and_dispatch_key_use_the_same_scope(monkeypatch) -> None:
    scope = Q8WMMALayerScope.parse("20")
    recorded: list[tuple] = []

    class _Recording(dict):
        def get(self, key, default=None):
            recorded.append(key)
            return super().get(key, default)

    original = gl._DISPATCH_RESOLVE_CACHE
    monkeypatch.setattr(gl, "_DISPATCH_RESOLVE_CACHE", _Recording())
    try:
        with runner_module._q8_wmma_scope_sessions(scope):
            assert gl._q8_wmma_layers_scope() == frozenset({20})
            selector = gl._wmma_prefill_weight_filter.get()
            assert selector is not None
            assert selector(_Weight(20)) is True
            assert selector(_Weight(21)) is False
            assert selector(_Weight(20, quant="gguf_q4_k")) is False

            try:
                gl.launch_gguf_linear(
                    _Weight(20), 0, 0, 1024, 2560, 10240, libraries={}
                )
            except Exception:
                # Synthetic pointers cannot launch; the key is computed first.
                pass
    finally:
        gl._DISPATCH_RESOLVE_CACHE = original

    assert recorded, "the launch never reached the dispatch memoization"
    assert recorded[-1][3] == (scope, 20)


def test_empty_scope_removes_the_filter_and_keys_as_empty() -> None:
    empty = Q8WMMALayerScope.empty()
    with runner_module._q8_wmma_scope_sessions(empty):
        assert gl._wmma_prefill_weight_filter.get() is None
        assert gl._q8_wmma_layers_scope() is None
        assert runner_module._q8_wmma_weight_filter(empty)(_Weight(21)) is True


def test_profile_gap_rejects_the_owner_key_and_exposes_an_explicit_override() -> None:
    from scripts.qwen4exp_profile_gap import (
        _configuration_class,
        _named_profile_intact,
        _parse_overrides,
        build_parser,
    )

    with pytest.raises(ValueError, match="owner state"):
        _parse_overrides([f"{ENV}=0"])
    args = build_parser().parse_args(
        [
            "--model-root", "/tmp/model",
            "--mode", "decode",
            "--output", "/tmp/out.json",
            "--q8-wmma-layers", "0",
        ]
    )
    assert args.q8_wmma_layers == "0"
    assert _named_profile_intact({}, None, None, Q8WMMALayerScope.parse("0")) is False
    assert _configuration_class(
        {}, None, None, Q8WMMALayerScope.parse("0")
    ) == "diagnostic_post_binder_override"
    assert _named_profile_intact({}, None, None, None) is True


def test_disabled_family_sweep_sends_the_wmma_owner_as_an_explicit_arg(
    monkeypatch, tmp_path,
) -> None:
    from scripts import qwen4exp_disabled_family_sweep as sweep

    family = next(
        item for item in sweep.FAMILIES
        if item["family"] == "Q8_WMMA_LAYERS_16_47"
    )
    assert ENV not in family["env"]
    assert family["owner"]["q8_wmma_layers"].startswith("16,17")

    seen: list[list[str]] = []

    def fake_run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=1, stderr="stopped")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    result = sweep._run(
        "python",
        Path("/tmp/model"),
        Path("/tmp/fixture.json"),
        "case",
        1,
        {},
        tmp_path / "out.json",
        owner=family["owner"],
    )

    assert result["ok"] is False
    assert "--override" not in seen[0]
    layers_at = seen[0].index("--q8-wmma-layers")
    assert seen[0][layers_at + 1] == family["owner"]["q8_wmma_layers"]


def test_dense_route_ab_arm_binds_the_owner_scope(monkeypatch) -> None:
    from scripts.qwen4exp_dense_route_ab import ARMS, _apply_arm

    monkeypatch.delenv(ENV, raising=False)
    arm = next(item for item in ARMS if item.name == "wmma-16-47")
    owner = SimpleNamespace()
    with _apply_arm(arm, owner):
        assert resolve_q8_wmma_layers(owner).layers == frozenset(range(16, 48))
        assert ENV not in os.environ
    assert not hasattr(owner, "q8_wmma_layer_scope")


def test_prefill_last_row_probe_clears_the_owner_scope(monkeypatch) -> None:
    from scripts.qwen4exp_prefill_last_row_probe import route_environment

    monkeypatch.delenv(ENV, raising=False)
    owner = SimpleNamespace()
    with route_environment(candidate=True, runner=owner) as scope:
        assert scope.is_empty
        assert resolve_q8_wmma_layers(owner).is_empty
        assert ENV not in os.environ
    assert not hasattr(owner, "q8_wmma_layer_scope")
