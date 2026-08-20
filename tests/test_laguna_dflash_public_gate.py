from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.laguna_dflash_public_gate import (
    _blocking_result,
    _capability_checks,
    _load_ar_oracle,
    _prompt_checks,
    _public_ids_until_stop,
    _state_is_reset,
    _stream_result,
)


def _oracle_row(prompt_id: str, ids: list[int], repetition: int) -> dict:
    return {
        "prompt": {
            "id": prompt_id,
            "category": "code",
            "prompt_tokens": 3,
            "prompt_ids_sha256": f"prompt-{prompt_id}",
        },
        "ar": {"generated_ids": ids},
        "repetition": repetition,
    }


def test_load_ar_oracle_requires_repeat_determinism(tmp_path: Path) -> None:
    path = tmp_path / "oracle.json"
    path.write_text(
        json.dumps(
            {
                "prompt_runs": [
                    _oracle_row("a", [1, 2], 0),
                    _oracle_row("a", [1, 2], 1),
                    _oracle_row("b", [3, 4], 0),
                    _oracle_row("b", [3, 4], 1),
                ]
            }
        ),
        encoding="utf-8",
    )

    oracle = _load_ar_oracle(path, expected_prompt_count=2)

    assert oracle["a"]["fixed_horizon_ids"] == (1, 2)
    assert oracle["a"]["repetitions"] == 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["prompt_runs"][1]["ar"]["generated_ids"] = [1, 9]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="nondeterministic"):
        _load_ar_oracle(path, expected_prompt_count=2)


def test_public_ids_include_first_stop_and_discard_suffix() -> None:
    assert _public_ids_until_stop((10, 11, 24, 13), (2, 24)) == (10, 11, 24)
    assert _public_ids_until_stop((10, 11, 12), (2, 24)) == (10, 11, 12)


def test_public_response_parsers_require_exact_cumulative_ids() -> None:
    blocking = _blocking_result(
        SimpleNamespace(
            status_code=200,
            text="unused",
            json=lambda: {
                "choices": [
                    {
                        "text": "AB",
                        "finish_reason": "length",
                        "finish_details": {"reason": "length"},
                        "hipengine": {
                            "generated_token_ids": [10, 11],
                            "decode_state": {
                                "execution_path": "laguna_dflash_b4_c1"
                            },
                            "diagnostics": {
                                "provider": "dflash",
                                "target_iq3_selected_down_tile": 4,
                            },
                        },
                    }
                ],
                "hipengine": {
                    "generation_shape": {"route": "speculative"}
                },
            },
        )
    )
    stream = _stream_result(
        SimpleNamespace(
            status_code=200,
            text=(
                'data: {"choices":[{"text":"A","finish_reason":null}]}\n\n'
                'data: {"choices":[{"text":"B","finish_reason":"length",'
                '"finish_details":{"reason":"length"},"hipengine":{'
                '"generated_token_ids":[10,11],"decode_state":{'
                '"execution_path":"laguna_dflash_b4_c1"},"diagnostics":{'
                '"provider":"dflash","target_iq3_selected_down_tile":4}}}]}\n\ndata: [DONE]\n\n'
            ),
        )
    )

    assert blocking["generated_ids"] == (10, 11)
    assert blocking["generation_shape"] == {"route": "speculative"}
    assert stream["text"] == "AB"
    assert stream["generated_ids"] == (10, 11)
    assert stream["done"] is True


def test_capability_gate_requires_dflash_iq3_tile4() -> None:
    payload = {
        "sampling": {
            "speculative": {
                "serving_route": True,
                "configured": True,
                "provider": "dflash",
                "configured_provider": "dflash",
                "request_field": "speculative",
                "policy": "explicit_only",
                "default_enabled": False,
                "streaming_compatible": True,
                "candidate_budget": 4,
                "exactness_mode": "target_corrected_greedy",
                "processed_target_verification": False,
                "target": {
                    "sha256": "7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f",
                    "iq3_selected_down_tile": 4,
                },
                "drafter": {
                    "sha256": "f24f08781c697c19952c02fb2e7e9bdf2071b79a711c2a44b836a74b9b62a1f4",
                    "revision": "b0486d1586daa0d56435c508108171fc1c8daff9",
                },
                "fallback_reason": "d4_full_suite_speedup_0p9469x_below_1p10",
                "economics_evidence": "benchmarks/results/2026-07-23-gfx1151-laguna-dflash-category-economics-post-prefill.json",
                "performance_claim": False,
            }
        }
    }

    checks = _capability_checks(payload)

    assert all(checks.values())
    payload["sampling"]["speculative"]["target"]["iq3_selected_down_tile"] = 1
    assert not _capability_checks(payload)["target_iq3_selected_down_tile"]


def _reset_state(owner_id: int = 1) -> dict:
    return {
        "provider_present": True,
        "provider_closed": False,
        "target_present": True,
        "target_position": -1,
        "target_closed": False,
        "drafter_present": True,
        "drafter_context_tokens": 0,
        "drafter_closed": False,
        "cycle_present": True,
        "owner_ids": {"target": owner_id, "drafter": 2, "cycle": 3},
    }


def test_prompt_gate_requires_ar_block_stream_state_and_route_equality() -> None:
    state = _reset_state()
    assert _state_is_reset(state)
    checks = _prompt_checks(
        expected_ids=(10, 11),
        ar={"generated_ids": (10, 11)},
        blocking={
            "generated_ids": (10, 11),
            "text": "AB",
            "generation_shape": {"route": "speculative"},
            "execution_path": "laguna_dflash_b4_c1",
            "diagnostics": {"target_iq3_selected_down_tile": 4},
        },
        streaming={
            "generated_ids": (10, 11),
            "text": "AB",
            "execution_path": "laguna_dflash_b4_c1",
            "diagnostics": {"target_iq3_selected_down_tile": 4},
            "done": True,
        },
        state_after_blocking=state,
        state_after_streaming=state,
        retained_owner_ids=state["owner_ids"],
    )

    assert all(checks.values())
    bad_state = dict(state, target_position=4)
    assert not _state_is_reset(bad_state)
