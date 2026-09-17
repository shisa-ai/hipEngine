from __future__ import annotations

from scripts.gguf_mtp_api_gate import _mtp_contract, sse_payloads


def _mtp_body(*, mtp_output_tokens: int = 2, ar_output_tokens: int = 1) -> dict:
    completion_tokens = mtp_output_tokens + ar_output_tokens
    return {
        "usage": {
            "completion_tokens": completion_tokens,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 2,
                "rejected_prediction_tokens": 1,
            },
        },
        "hipengine": {
            "speculative_mtp": {
                "used": True,
                "effective_route": "speculative_mtp",
                "thinking_policy": "hint",
                "mtp_output_tokens": mtp_output_tokens,
                "ar_output_tokens": ar_output_tokens,
                "output_accounting": {
                    "completion_tokens": completion_tokens,
                    "mtp_output_tokens": mtp_output_tokens,
                    "ar_output_tokens": ar_output_tokens,
                    "ar_output_tokens_in_cycles": ar_output_tokens,
                    "reconciled": True,
                },
            }
        },
    }


def test_sse_payloads_extracts_error_and_ignores_done() -> None:
    rows = sse_payloads('data: {"error":{"code":"x"}}\n\ndata: [DONE]\n\n')
    assert rows == [{"error": {"code": "x"}}]


def test_mtp_contract_distinguishes_direct_usage_fields() -> None:
    mtp = _mtp_body()
    ar = {
        "usage": {"completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 0}},
        "hipengine": {"speculative_mtp": {"used": False, "effective_route": "default"}},
    }
    assert _mtp_contract(mtp, used=True) is True
    assert _mtp_contract(ar, used=False) is True
    assert _mtp_contract(ar, used=True) is False


def test_mtp_contract_requires_the_output_split_to_reconcile() -> None:
    body = _mtp_body(mtp_output_tokens=2, ar_output_tokens=1)
    assert _mtp_contract(body, used=True) is True

    # A split that does not add up to the reported completion count fails.
    unreconciled = _mtp_body(mtp_output_tokens=2, ar_output_tokens=1)
    unreconciled["hipengine"]["speculative_mtp"]["output_accounting"]["reconciled"] = False
    assert _mtp_contract(unreconciled, used=True) is False

    mismatched = _mtp_body(mtp_output_tokens=2, ar_output_tokens=1)
    mismatched["hipengine"]["speculative_mtp"]["output_accounting"]["ar_output_tokens"] = 4
    assert _mtp_contract(mismatched, used=True) is False

    wrong_total = _mtp_body(mtp_output_tokens=2, ar_output_tokens=1)
    wrong_total["usage"]["completion_tokens"] = 9
    assert _mtp_contract(wrong_total, used=True) is False

    # "used" without any speculative output is exactly the case the split exists
    # to catch.
    empty = _mtp_body(mtp_output_tokens=0, ar_output_tokens=3)
    assert _mtp_contract(empty, used=True) is False


def test_mtp_contract_rejects_a_used_row_without_accounting() -> None:
    body = _mtp_body()
    del body["hipengine"]["speculative_mtp"]["output_accounting"]
    assert _mtp_contract(body, used=True) is False


def test_mtp_contract_rejects_more_plan_attributed_steps_than_ar_output() -> None:
    """Counting one emitted token twice must fail the gate, not inflate a count."""

    body = _mtp_body(mtp_output_tokens=2, ar_output_tokens=1)
    assert _mtp_contract(body, used=True) is True

    doubled = _mtp_body(mtp_output_tokens=2, ar_output_tokens=1)
    doubled["hipengine"]["speculative_mtp"]["output_accounting"][
        "ar_output_tokens_in_cycles"
    ] = 2
    assert _mtp_contract(doubled, used=True) is False

    missing = _mtp_body(mtp_output_tokens=2, ar_output_tokens=1)
    del missing["hipengine"]["speculative_mtp"]["output_accounting"][
        "ar_output_tokens_in_cycles"
    ]
    assert _mtp_contract(missing, used=True) is False

    # The plan may cover only part of the autoregressive output; the remainder
    # was emitted after the request left the plan.
    partial = _mtp_body(mtp_output_tokens=2, ar_output_tokens=3)
    partial["hipengine"]["speculative_mtp"]["output_accounting"][
        "ar_output_tokens_in_cycles"
    ] = 1
    assert _mtp_contract(partial, used=True) is True
