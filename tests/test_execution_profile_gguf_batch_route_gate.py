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
