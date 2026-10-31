"""The Q4 DP4A decode pair is owner state, with env only as a direct fallback."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile,
    clear_runtime_profile_registry_for_tests,
    resolve_runtime_profile,
)
from hipengine.generation import qwen4_exp_gguf
from hipengine.generation.qwen4_exp_gguf import (
    Qwen4ExpGGUFTextGenerator,
    _inherit_lane_decisions,
)
from hipengine.generation.qwen4_exp_profiles import (
    PRODUCTION_Q4_DP4A_DECODE_LAYERS,
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    QWEN4_EXP_QUANTS,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpGGUFResidentModelRunner,
    _qwen4_exp_q4_dp4a_decode_selected,
    _qwen4_exp_q4_dp4a_decode_key,
    _resolve_q4_dp4a_decode,
    q4_dp4a_decode,
    q4_dp4a_decode_graph_key,
)

ENV_ENABLED = "HIPENGINE_QWEN4_EXP_Q4_DP4A64"
ENV_LAYERS = "HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS"
SAFE43 = frozenset(PRODUCTION_Q4_DP4A_DECODE_LAYERS)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    clear_runtime_profile_registry_for_tests()
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    monkeypatch.delenv(ENV_LAYERS, raising=False)
    yield
    clear_runtime_profile_registry_for_tests()


def _resolve(profile: ExecutionProfile, *, quant: str = "gguf_ud_q4_k_xl"):
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=quant,
        profile=profile,
    )


def _bind(resolved, *, runner: SimpleNamespace | None = None):
    fake_runner = runner or SimpleNamespace(
        configure_mmq_prefill_resources=lambda: None)
    generator = SimpleNamespace(runner=fake_runner)
    resolved.binder(generator, resolved)
    return generator, fake_runner


def test_two_owners_keep_independent_decisions_in_both_bind_orders():
    production = _resolve(ExecutionProfile.PRODUCTION)
    strict = _resolve(ExecutionProfile.STRICT)
    first = SimpleNamespace(configure_mmq_prefill_resources=lambda: None)
    second = SimpleNamespace(configure_mmq_prefill_resources=lambda: None)
    first_generator = SimpleNamespace(runner=first)
    second_generator = SimpleNamespace(runner=second)

    production.binder(first_generator, production)
    strict.binder(second_generator, strict)
    assert first_generator.q4_dp4a_decode_enabled is False
    assert first.q4_dp4a_decode_enabled is False
    assert first_generator.q4_dp4a_decode_layers == SAFE43
    assert second_generator.q4_dp4a_decode_enabled is False
    assert second.q4_dp4a_decode_enabled is False
    assert second_generator.q4_dp4a_decode_layers == frozenset()

    # Re-binding one owner does not move its peer.
    strict.binder(first_generator, strict)
    assert first_generator.q4_dp4a_decode_layers == frozenset()
    assert second_generator.q4_dp4a_decode_layers == frozenset()

    production.binder(second_generator, production)
    assert first_generator.q4_dp4a_decode_layers == frozenset()
    assert second_generator.q4_dp4a_decode_layers == SAFE43


def test_binder_no_longer_publishes_the_q4_env_pair(monkeypatch):
    monkeypatch.setenv(ENV_ENABLED, "stale")
    monkeypatch.setenv(ENV_LAYERS, "7")
    production = _resolve(ExecutionProfile.PRODUCTION)
    generator, runner = _bind(production)
    assert os.environ[ENV_ENABLED] == "stale"
    assert os.environ[ENV_LAYERS] == "7"
    assert generator.q4_dp4a_decode_enabled is False
    assert runner.q4_dp4a_decode_enabled is False
    assert generator.q4_dp4a_decode_layers == SAFE43


def test_clone_and_constructor_carry_the_owner_decision(monkeypatch):
    calls: list[dict] = []

    def fake_runner(resident, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(qwen4_exp_gguf, "Qwen4ExpGGUFResidentModelRunner", fake_runner)
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        runner=SimpleNamespace(
            max_sequence_length=8, prefill_chunk_size=4, close=lambda: None),
    )
    generator.q4_dp4a_decode_enabled = True
    generator.q4_dp4a_decode_layers = frozenset({0, 2})
    try:
        clone = generator._new_resident_runner(
            SimpleNamespace(), max_sequence_length=8, prefill_chunk_size=4)
    finally:
        generator.close()
    assert calls[0]["q4_dp4a_decode_enabled"] is True
    assert calls[0]["q4_dp4a_decode_layers"] == frozenset({0, 2})
    assert clone.q4_dp4a_decode_enabled is True
    assert clone.q4_dp4a_decode_layers == frozenset({0, 2})

    inherited = _inherit_lane_decisions(
        generator, SimpleNamespace(q4_dp4a_decode_enabled=False,
                                   q4_dp4a_decode_layers=frozenset()))
    assert inherited.q4_dp4a_decode_enabled is True
    assert inherited.q4_dp4a_decode_layers == frozenset({0, 2})


def test_runner_and_generator_expose_the_owner_inputs():
    runner = inspect.signature(Qwen4ExpGGUFResidentModelRunner.__init__).parameters
    generator = inspect.signature(Qwen4ExpGGUFTextGenerator.__init__).parameters
    assert runner["q4_dp4a_decode_enabled"].default is None
    assert runner["q4_dp4a_decode_layers"].default is None
    assert "q4_dp4a_decode_enabled" not in generator


def test_owner_wins_over_ambient_and_explicit_override_wins_over_owner(monkeypatch):
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv(ENV_LAYERS, "1,3")
    assert _resolve_q4_dp4a_decode(None, None) == (True, frozenset({1, 3}))
    assert _resolve_q4_dp4a_decode(False, frozenset({2})) == (False, frozenset({2}))
    assert _resolve_q4_dp4a_decode(True, None) == (True, frozenset())

    owner = SimpleNamespace(
        q4_dp4a_decode_enabled=True,
        q4_dp4a_decode_layers=frozenset({1}),
    )
    with q4_dp4a_decode(owner, enabled=False, layers=frozenset({2})):
        assert owner.q4_dp4a_decode_enabled is False
        assert owner.q4_dp4a_decode_layers == frozenset({2})
    assert owner.q4_dp4a_decode_enabled is True
    assert owner.q4_dp4a_decode_layers == frozenset({1})


def test_operation_override_restores_after_failure_and_updates_generator_runner():
    generator = SimpleNamespace(
        q4_dp4a_decode_enabled=None,
        q4_dp4a_decode_layers=None,
        runner=SimpleNamespace(
            q4_dp4a_decode_enabled=None,
            q4_dp4a_decode_layers=None,
        ),
    )
    with pytest.raises(RuntimeError, match="failed arm"):
        with q4_dp4a_decode(generator, enabled=True, layers=frozenset({5})):
            assert generator.q4_dp4a_decode_enabled is True
            assert generator.runner.q4_dp4a_decode_enabled is True
            assert generator.runner.q4_dp4a_decode_layers == frozenset({5})
            raise RuntimeError("failed arm")
    assert generator.q4_dp4a_decode_enabled is None
    assert generator.runner.q4_dp4a_decode_enabled is None
    assert generator.runner.q4_dp4a_decode_layers is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", frozenset()),
        (" ", frozenset()),
        (",", frozenset()),
        ("0, 2,5", frozenset({0, 2, 5})),
        ("0,2,5,6,8,9,10,11,13", frozenset({0, 2, 5, 6, 8, 9, 10, 11, 13})),
    ],
)
def test_layer_parser_preserves_existing_semantics(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv(ENV_LAYERS, raw)
    assert _resolve_q4_dp4a_decode(None, None)[1] == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", False),
        ("0", False),
        ("false", False),
        ("False", False),
        ("1", True),
        ("true", True),
        ("FALSE", True),
    ],
)
def test_boolean_parser_preserves_existing_semantics(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV_ENABLED, raw)
    monkeypatch.setenv(ENV_LAYERS, "")
    assert _resolve_q4_dp4a_decode(None, None)[0] is expected


def test_invalid_layer_token_still_raises(monkeypatch):
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv(ENV_LAYERS, "0,nope")
    with pytest.raises(ValueError):
        _resolve_q4_dp4a_decode(None, None)


def test_graph_key_is_canonical_and_independent_of_ambient(monkeypatch):
    monkeypatch.setenv(ENV_ENABLED, "0")
    monkeypatch.setenv(ENV_LAYERS, "9")
    owner_key = q4_dp4a_decode_graph_key(
        "gdn", 7, (True, frozenset({2, 1})))
    peer_key = q4_dp4a_decode_graph_key(
        "gdn", 7, (False, frozenset()))
    ambient_key = q4_dp4a_decode_graph_key(
        "gdn", 7, _resolve_q4_dp4a_decode(None, None))
    assert owner_key == ("gdn", 7, True, (1, 2))
    assert peer_key == ("gdn", 7, False, ())
    assert ambient_key == ("gdn", 7, False, (9,))
    assert owner_key != peer_key != ambient_key


def test_leaf_key_is_the_dp4a_owner_and_fallback_is_registration_guarded():
    key = _qwen4_exp_q4_dp4a_decode_key("hip_gfx1151", "gguf_q4_k")
    assert key.layer == "linear"
    assert key.variant == (
        "selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out")

    common = dict(
        rows=1,
        gate_quant="gguf_q4_k",
        up_quant="gguf_q4_k",
        gate_layer=7,
        enabled=True,
        layers=frozenset({7}),
        registered=True,
    )
    assert _qwen4_exp_q4_dp4a_decode_selected(**common)
    assert not _qwen4_exp_q4_dp4a_decode_selected(**{**common, "rows": 2})
    assert not _qwen4_exp_q4_dp4a_decode_selected(
        **{**common, "gate_quant": "gguf_q5_1"})
    assert not _qwen4_exp_q4_dp4a_decode_selected(
        **{**common, "gate_layer": 8})
    assert not _qwen4_exp_q4_dp4a_decode_selected(
        **{**common, "enabled": False})
    assert not _qwen4_exp_q4_dp4a_decode_selected(
        **{**common, "registered": False})
    # Empty scope is no filter, not "no layers".
    assert _qwen4_exp_q4_dp4a_decode_selected(
        **{**common, "layers": frozenset()})


def test_diagnostic_scripts_bind_the_owner_instead_of_writing_the_dead_pair():
    from scripts.qwen4exp_layer2_profile_gate import CANDIDATES
    from scripts.qwen4exp_r2e_route_ablation import ROUTES, RUNNER_ROUTE_OFF

    candidate = CANDIDATES["decode_dp4a_linear"]
    assert candidate.q4_dp4a_decode_enabled is True
    assert candidate.q4_dp4a_decode_layers == SAFE43
    assert ENV_ENABLED not in candidate.environment
    assert ENV_LAYERS not in candidate.environment
    assert "decode_dp4a" not in ROUTES
    assert RUNNER_ROUTE_OFF["decode_dp4a"] == "q4_dp4a_decode"


def test_disabled_family_sweep_sends_the_q4_owner_as_explicit_args(monkeypatch, tmp_path):
    from scripts import qwen4exp_disabled_family_sweep as sweep

    family = next(item for item in sweep.FAMILIES if item["family"] == "Q4_DP4A64")
    assert ENV_ENABLED not in family["env"]
    assert ENV_LAYERS not in family["env"]
    assert family["owner"]["q4_dp4a_decode"] == "on"

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
    decode_at = seen[0].index("--q4-dp4a-decode")
    layers_at = seen[0].index("--q4-dp4a-layers")
    assert seen[0][decode_at + 1] == "on"
    assert seen[0][layers_at + 1] == family["owner"]["q4_dp4a_layers"]


def test_decode_sync_ab_uses_the_owner_override_not_the_env_pair():
    text = (Path(__file__).resolve().parents[1]
            / "scripts" / "qwen4exp_decode_sync_ab.py").read_text()
    assert ENV_ENABLED not in text
    assert ENV_LAYERS not in text
    assert "q4_dp4a_decode(generator, True, SAFE43_DP4A_LAYERS)" in text
