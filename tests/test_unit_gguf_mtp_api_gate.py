from __future__ import annotations

from scripts.gguf_mtp_api_gate import _mtp_contract, sse_payloads


def test_sse_payloads_extracts_error_and_ignores_done() -> None:
    rows = sse_payloads('data: {"error":{"code":"x"}}\n\ndata: [DONE]\n\n')
    assert rows == [{"error": {"code": "x"}}]


def test_mtp_contract_distinguishes_direct_usage_fields() -> None:
    mtp = {
        "usage": {
            "completion_tokens_details": {
                "accepted_prediction_tokens": 2,
                "rejected_prediction_tokens": 1,
            }
        },
        "hipengine": {
            "speculative_mtp": {
                "used": True,
                "effective_route": "speculative_mtp",
                "thinking_policy": "hint",
            }
        },
    }
    ar = {
        "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
        "hipengine": {"speculative_mtp": {"used": False, "effective_route": "default"}},
    }
    assert _mtp_contract(mtp, used=True) is True
    assert _mtp_contract(ar, used=False) is True
    assert _mtp_contract(ar, used=True) is False
