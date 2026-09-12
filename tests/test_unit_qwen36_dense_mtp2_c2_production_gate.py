from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from hipengine.core.specdec2_scope import (
    moe_physical_c2_exact_linear_enabled,
    moe_physical_c2_f32_residual_disabled,
    moe_physical_c2_pairreuse_enabled,
    q4_t16_physical_extra_rowtiles_enabled,
    q5_t16_physical_rowtile_enabled,
    q6_t16_physical_rowtile_enabled,
)
from scripts.qwen36_dense_mtp2_c2_production_gate import (
    _install_packed_capture,
    _physical_target_scope,
    _repeat_verdict,
)
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


def _capture(*, repeat: int, digest: str = "same") -> dict[str, object]:
    return {
        "repeat": repeat,
        "prompt_id": "code_merge_intervals",
        "start_position": 40,
        "inputs": (264, 7047, 1817),
        "candidate_logits_sha256": digest,
        "candidate_top1": (7047, 1817, 25),
    }


def test_physical_target_scope_replays_captured_production_arithmetic() -> None:
    captured = {
        "moe_f32_residual_disabled": True,
        "moe_pairreuse": True,
        "moe_exact_linear": True,
        "q4_extra_rowtiles": True,
        "q5_rowtile": True,
        "q6_rowtile": True,
    }

    with _physical_target_scope(captured):
        assert moe_physical_c2_f32_residual_disabled() is True
        assert moe_physical_c2_pairreuse_enabled() is True
        assert moe_physical_c2_exact_linear_enabled() is True
        assert q4_t16_physical_extra_rowtiles_enabled() is True
        assert q5_t16_physical_rowtile_enabled() is True
        assert q6_t16_physical_rowtile_enabled() is True

    assert moe_physical_c2_f32_residual_disabled() is False
    assert moe_physical_c2_pairreuse_enabled() is False
    assert moe_physical_c2_exact_linear_enabled() is False
    assert q4_t16_physical_extra_rowtiles_enabled() is False
    assert q5_t16_physical_rowtile_enabled() is False
    assert q6_t16_physical_rowtile_enabled() is False


def test_packed_capture_records_the_real_adapter_scope(monkeypatch) -> None:
    monkeypatch.setattr(
        Qwen35GGUFResidentSession,
        "verify_target_blocks_batch",
        lambda self, jobs, **kwargs: list(jobs),
    )
    context = {
        "enabled": True,
        "repeat": 0,
        "pair": 0,
        "cycle_counts": {},
        "captures": [],
    }
    original = _install_packed_capture(context)
    runtime = SimpleNamespace(device_synchronize=lambda: None)
    session = SimpleNamespace(
        runtime=runtime,
        runner=SimpleNamespace(vocab_size=100),
    )
    jobs = [
        {
            "session": SimpleNamespace(position=7),
            "request_id": 3,
            "input_token_ids": (1, 2),
        }
    ]

    try:
        with _physical_target_scope(
            {
                "moe_f32_residual_disabled": True,
                "moe_pairreuse": True,
                "moe_exact_linear": True,
            }
        ):
            Qwen35GGUFResidentSession.verify_target_blocks_batch(session, jobs)
    finally:
        Qwen35GGUFResidentSession.verify_target_blocks_batch = original

    assert context["captures"][0]["physical_scope"] == {
        "moe_f32_residual_disabled": True,
        "moe_pairreuse": True,
        "moe_exact_linear": True,
        "q4_extra_rowtiles": False,
        "q5_rowtile": False,
        "q6_rowtile": False,
    }


def test_repeat_verdict_requires_all_three_identical_physical_schedules() -> None:
    result = _repeat_verdict(tuple(_capture(repeat=index) for index in range(3)), 3)

    assert result["passed"] is True
    assert result["rows"] == [
        {
            "prompt_id": "code_merge_intervals",
            "start_position": 40,
            "repeats": [0, 1, 2],
            "inputs_equal": True,
            "passed": True,
        }
    ]


def test_repeat_verdict_fails_on_missing_repeat_or_input_drift() -> None:
    missing = _repeat_verdict((_capture(repeat=0), _capture(repeat=1)), 3)
    drift_rows = [_capture(repeat=index) for index in range(3)]
    drift_rows[-1] = deepcopy(drift_rows[-1])
    drift_rows[-1]["inputs"] = (264, 7047, 999)
    drift = _repeat_verdict(tuple(drift_rows), 3)

    assert missing["passed"] is False
    assert drift["passed"] is False
    assert drift["rows"][0]["inputs_equal"] is False
