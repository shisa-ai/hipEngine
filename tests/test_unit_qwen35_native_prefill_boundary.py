from __future__ import annotations

from scripts.qwen35_native_prefill_boundary import _boundary_payload


def test_qwen35_native_prefill_boundary_accepts_full_attention_layers() -> None:
    payload = _boundary_payload(
        ("linear_attention", "linear_attention", "full_attention", "linear_attention"),
        model="/tmp/model",
        max_layers=4,
        command="cmd",
    )

    assert payload["status"] == "accepted"
    assert payload["performance_claim"] is False
    assert payload["accepted_linear_prefix_layers"] == 2
    assert payload["first_unsupported_layer"] is None
    assert payload["first_unsupported_type"] is None
    assert payload["component_blockers"] == []
    assert payload["next_actions"] == []


def test_qwen35_native_prefill_boundary_accepts_all_linear_prefix() -> None:
    payload = _boundary_payload(
        ("linear_attention", "linear_attention"),
        model="/tmp/model",
        max_layers=2,
        command="cmd",
    )

    assert payload["status"] == "accepted"
    assert payload["blocked_reason"] is None
    assert payload["first_unsupported_layer"] is None
    assert payload["component_blockers"] == []
    assert payload["next_actions"] == []
