from __future__ import annotations

import os

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from scripts import execution_profile_gguf_c1_route_gate as gate


def _state(prompt_id: str, digest: str, **overrides):
    row = {
        "prompt_id": prompt_id,
        "position": 32,
        "finite": True,
        "linear_state_pairs": 30,
        "full_attention_kv_pairs": 10,
        "state_sha256": digest,
    }
    row.update(overrides)
    return row


def test_route_environment_is_complete_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = gate._ROUTE_ENV_KEYS
    monkeypatch.setenv(first, "caller")
    monkeypatch.delenv(second, raising=False)

    with gate.route_environment(gate.STRICT_ENVIRONMENT):
        assert os.environ[first] == "0"
        assert os.environ[second] == "0"

    assert os.environ[first] == "caller"
    assert second not in os.environ


@pytest.mark.parametrize("fail", [False, True])
def test_route_owner_is_nested_restored_and_environment_independent(monkeypatch, fail):
    for key in gate._ROUTE_ENV_KEYS:
        monkeypatch.setenv(key, "caller")
    runner = SimpleNamespace(router_f32w_coop=None, router_f32w_persistent_counter=True)
    session = SimpleNamespace(runner=runner)
    with pytest.raises(RuntimeError) if fail else nullcontext():
        with gate.route_owner(session, gate.STRICT_ENVIRONMENT):
            assert runner.router_f32w_coop is False
            assert runner.router_f32w_persistent_counter is False
            with gate.route_owner(session, gate.CANDIDATES['router_f32w_coop_persistent'].environment):
                assert runner.router_f32w_coop is True
                assert runner.router_f32w_persistent_counter is True
            assert runner.router_f32w_coop is False
            assert all(os.environ[key] == "caller" for key in gate._ROUTE_ENV_KEYS)
            if fail:
                raise RuntimeError("comparison failure")
    assert runner.router_f32w_coop is None
    assert runner.router_f32w_persistent_counter is True


@pytest.mark.parametrize("bad", ["invalid", "true", None])
def test_route_owner_rejects_invalid_values_without_mutation(bad):
    runner = SimpleNamespace(router_f32w_coop=None, router_f32w_persistent_counter=True)
    values = dict(gate.STRICT_ENVIRONMENT)
    values[gate._ROUTE_ENV_KEYS[0]] = bad
    with pytest.raises(ValueError, match="0 or 1"):
        with gate.route_owner(SimpleNamespace(runner=runner), values):
            pass
    assert runner.router_f32w_coop is None
    assert runner.router_f32w_persistent_counter is True


def test_route_owner_normalizes_binary_values_consistently():
    runner = SimpleNamespace(router_f32w_coop=None, router_f32w_persistent_counter=None)
    values = {key: 1 for key in gate._ROUTE_ENV_KEYS}
    with gate.route_owner(SimpleNamespace(runner=runner), values):
        assert runner.router_f32w_coop is True
        assert runner.router_f32w_persistent_counter is True


def test_route_environment_rejects_partial_manifest() -> None:
    with pytest.raises(ValueError, match="missing"):
        with gate.route_environment({gate._ROUTE_ENV_KEYS[0]: "1"}):
            pass


def test_state_repeat_gate_allows_candidate_numeric_drift_but_requires_repeatability() -> None:
    result = gate.build_state_repeat_gate(
        [_state("p0", "strict")],
        [[_state("p0", "candidate") for _ in range(3)]],
    )

    assert result["passed"] is True
    assert result["prompts"][0]["strict_and_candidate_bytes_equal"] is False
    assert result["prompts"][0]["repeatable"] is True


def test_state_repeat_gate_rejects_nonfinite_layout_or_digest_drift() -> None:
    result = gate.build_state_repeat_gate(
        [_state("p0", "strict")],
        [[
            _state("p0", "candidate"),
            _state("p0", "changed"),
            _state("p0", "candidate", finite=False, linear_state_pairs=29),
        ]],
    )

    assert result["passed"] is False
    assert result["mismatches"] == [
        {
            "prompt_id": "p0",
            "finite": False,
            "layout_stable": False,
            "repeatable": False,
        }
    ]


def test_route_owner_metadata_does_not_claim_launch_engagement():
    strict = gate.route_owner_metadata(gate.STRICT_ENVIRONMENT)
    candidate = gate.route_owner_metadata(gate.CANDIDATES['router_f32w_coop_persistent'].environment)
    assert strict['selection'] == 'separate_projections_topk'
    assert strict['counter'] == 'none'
    assert candidate['selection'] == 'cooperative_persistent'
    assert candidate['counter'] == 'self_resetting_device'
    assert candidate['environment_mutated'] is False
    assert candidate['engagement'] == 'requested; requires launch trace'


def test_router_candidate_declares_t2_mechanism_and_strict_fallback() -> None:
    candidate = gate.CANDIDATES["router_f32w_coop_persistent"]
    assert candidate.classification == "T2"
    assert "top-k" in candidate.mechanism
    assert "router_select" in candidate.strict_fallback
    assert candidate.environment != gate.STRICT_ENVIRONMENT
