from __future__ import annotations

import os

import numpy as np
import pytest

from scripts import execution_profile_gguf_batch_route_gate as gate
from scripts.execution_profile_gguf_batch_route_gate import (
    BatchRouteCapture,
    build_batch_route_quality,
    validate_width_schedule,
)


def _step(token_id: int, logits: list[float]) -> dict[str, object]:
    return {"token_id": token_id, "logits": np.asarray(logits, dtype=np.float32)}


def test_validate_width_schedule_requires_descending_supported_widths() -> None:
    assert validate_width_schedule(((0, 8), (2, 4), (5, 2), (7, 1)), decode_steps=9) == (
        (0, 8),
        (2, 4),
        (5, 2),
        (7, 1),
    )
    assert validate_width_schedule(
        ((0, 7), (2, 6), (5, 5), (7, 3)), decode_steps=9
    ) == (
        (0, 7),
        (2, 6),
        (5, 5),
        (7, 3),
    )
    with pytest.raises(ValueError, match="start at step zero"):
        validate_width_schedule(((1, 8),), decode_steps=9)
    with pytest.raises(ValueError, match="strictly descend"):
        validate_width_schedule(((0, 4), (2, 8)), decode_steps=9)
    with pytest.raises(ValueError, match="outside decode horizon"):
        validate_width_schedule(((0, 8), (9, 4)), decode_steps=9)


def test_bundled_router_policy_uses_package_rowtile_floor_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(gate.POLICY_ENV, "caller-rowtile")
    monkeypatch.setenv(gate.ROUTER_COOP_ENV, "caller")
    monkeypatch.delenv(gate.ROUTER_PERSISTENT_ENV, raising=False)

    with gate._candidate_bundle_policy(True, include_router_candidate=True):
        assert gate.POLICY_ENV not in os.environ
        assert os.environ[gate.ROUTER_COOP_ENV] == "1"
        assert os.environ[gate.ROUTER_PERSISTENT_ENV] == "1"

    assert os.environ[gate.POLICY_ENV] == "caller-rowtile"
    assert os.environ[gate.ROUTER_COOP_ENV] == "caller"
    assert gate.ROUTER_PERSISTENT_ENV not in os.environ

    variants = gate.candidate_variant_manifest(include_router_candidate=True)
    assert variants["c2"]["single"] == "t16_gemv_decode_bf16_bf16_out"
    assert variants["c4"]["single"] == "t16_gemv_decode_rowtile4_bf16_bf16_out"


def test_current_package_policy_clears_candidate_overrides_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(gate.POLICY_ENV, "caller-rowtile")
    monkeypatch.setenv(gate.ROUTER_COOP_ENV, "caller-coop")
    monkeypatch.setenv(gate.ROUTER_PERSISTENT_ENV, "caller-persistent")

    with gate._current_package_policy():
        assert gate.POLICY_ENV not in os.environ
        assert gate.ROUTER_COOP_ENV not in os.environ
        assert gate.ROUTER_PERSISTENT_ENV not in os.environ

    assert os.environ[gate.POLICY_ENV] == "caller-rowtile"
    assert os.environ[gate.ROUTER_COOP_ENV] == "caller-coop"
    assert os.environ[gate.ROUTER_PERSISTENT_ENV] == "caller-persistent"


@pytest.mark.parametrize("fail", (False, True))
def test_selected_pair_package_policy_clears_overrides_and_restores(
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    keys = (
        gate.POLICY_ENV,
        gate.PAIR_POLICY_ENV,
        gate.PAIR_COL8_ENV,
        gate.SELECTED_PAIR_ENV,
        gate.SELECTED_DOWN_PAIR_ENV,
        gate.SELECTED_Q6_DOWN_PAIR_ENV,
    )
    previous = {key: str(index % 2) for index, key in enumerate(keys)}
    for key, value in previous.items():
        monkeypatch.setenv(key, value)

    def exercise() -> None:
        with gate._candidate_route_policy(gate.ROUTE_PROFILE_Q4_SELECTED_PAIR):
            assert all(key not in os.environ for key in keys)
            if fail:
                raise RuntimeError("capture failed")

    if fail:
        with pytest.raises(RuntimeError, match="capture failed"):
            exercise()
    else:
        exercise()
    assert {key: os.environ.get(key) for key in keys} == previous


@pytest.mark.parametrize("fail", [False, True])
def test_pair_package_policy_reaches_backend_floor_and_restores(monkeypatch, fail) -> None:
    from hipengine.runtime import gguf_linear as linear

    keys = (gate.POLICY_ENV, gate.PAIR_POLICY_ENV, gate.PAIR_COL8_ENV)
    previous = dict(zip(keys, ("1", "0", "0")))
    for key, value in previous.items():
        monkeypatch.setenv(key, value)

    def exercise() -> None:
        with gate._candidate_route_policy(gate.ROUTE_PROFILE_Q8T16_PAIR):
            assert all(key not in os.environ for key in keys)
            with linear.q8_t16_pair_rowtile_min_rows_session(8):
                assert linear._resolve_use_q8_t16_pair_rowtile(rows=8)
                assert linear._resolve_use_q8_t16_pair_col8(rows=8)
            with gate._pair_package_policy():
                assert all(key not in os.environ for key in keys)
            if fail:
                raise RuntimeError("pair capture failed")

    if fail:
        with pytest.raises(RuntimeError, match="pair capture failed"):
            exercise()
    else:
        exercise()
    assert {key: os.environ.get(key) for key in keys} == previous


@pytest.mark.parametrize("bundled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_broad_route_gate_restores_ambient_policy_on_every_exit(monkeypatch, bundled, fail) -> None:
    monkeypatch.setenv(gate.POLICY_ENV, "caller-override")

    def exercise():
        with gate._candidate_bundle_policy(True, include_router_candidate=bundled):
            assert os.environ.get(gate.POLICY_ENV) == (None if bundled else "1")
            if fail:
                raise RuntimeError("broad capture failed")

    if fail:
        with pytest.raises(RuntimeError, match="broad capture failed"):
            exercise()
    else:
        exercise()
    assert os.environ[gate.POLICY_ENV] == "caller-override"
    assert gate._router_candidate_enabled is False


def test_build_batch_route_quality_preserves_shape_and_transition_attribution() -> None:
    strict = (
        _step(1, [0.0, 4.0, 1.0]),
        _step(2, [0.0, 1.0, 4.0]),
    )
    runs = (strict, strict, strict)
    capture = BatchRouteCapture(
        scenario_id="dynamic",
        request_id="request-0",
        category="code",
        strict=strict,
        candidate_runs=runs,
        shapes=("c8", "c4"),
        transitions=("steady", "width_8_to_4"),
        teacher_steps=(1, 2),
    )

    result = build_batch_route_quality((capture,))

    assert result["quality"]["hard_gates_passed"] is True
    assert result["repeat_determinism"]["passed"] is True
    assert result["quality"]["by_scope"]["shape"]["c8"]["rows"] == 1
    assert result["quality"]["by_scope"]["shape"]["c4"]["rows"] == 1
    assert (
        result["quality"]["by_scope"]["transition"]["width_8_to_4"]["rows"]
        == 1
    )
