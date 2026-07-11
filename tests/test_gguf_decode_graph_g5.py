from __future__ import annotations

import pytest

from scripts.gguf_decode_graph_g5 import (
    _build_graph_key,
    _classify_candidate,
    _compare_checkpoints,
    _context_bucket,
    _summarize_runs,
)


def _fingerprint(value: str, *, nbytes: int = 16) -> dict[str, object]:
    return {
        "nbytes": nbytes,
        "blake2b_128": value,
        "finite": True,
        "rms": 1.0,
        "max_abs": 2.0,
    }


def _checkpoint(*, token: int = 9707, recurrent: str = "r0", kv: str = "k0") -> dict[str, object]:
    return {
        "position": 513,
        "input_token_id": 9707,
        "predicted_token_id": token,
        "finite": True,
        "hidden_seed": _fingerprint("h0"),
        "linear_states": [
            {
                "layer": 0,
                "conv": _fingerprint("c0"),
                "recurrent": _fingerprint(recurrent),
            }
        ],
        "kv_states": [
            {
                "layer": 3,
                "live_positions": 513,
                "key": _fingerprint(kv),
                "value": _fingerprint("v0"),
            }
        ],
    }


def test_context_bucket_rounds_replay_limit_and_checks_capacity() -> None:
    assert _context_bucket(position=512, replay_steps=17, block_size=256, max_positions=1024) == 768
    with pytest.raises(ValueError, match="capacity"):
        _context_bucket(position=900, replay_steps=200, block_size=256, max_positions=1024)


def test_graph_key_covers_state_generation_and_buffer_identity() -> None:
    common = {
        "backend": "hip_gfx1151",
        "target_arch": "gfx1151",
        "model_fingerprint": "model-sha",
        "quant": "gguf_q4_k_m",
        "kv_dtype": "bf16",
        "position": 512,
        "replay_steps": 16,
        "steps_per_launch": 1,
        "block_size": 256,
        "max_positions": 1024,
        "hidden_size": 2048,
        "vocab_size": 248320,
        "layer_types": ("linear_attention", "full_attention"),
        "weight_role_digest": "weights",
        "route": {"gemv_decode": True, "decode_repack": True},
        "buffer_ptrs": (101, 202, 303),
    }
    first = _build_graph_key(**common, state_generation=512)
    same = _build_graph_key(**common, state_generation=512)
    next_state = _build_graph_key(**common, state_generation=513)
    next_buffers = _build_graph_key(**{**common, "buffer_ptrs": (101, 202, 404)}, state_generation=512)

    assert first == same
    assert first["key_sha256"] != next_state["key_sha256"]
    assert first["key_sha256"] != next_buffers["key_sha256"]
    assert first["axes"]["active_rows"] == 1
    assert first["axes"]["context_bucket"] == 768


def test_checkpoint_comparison_localizes_recurrent_and_kv_drift() -> None:
    exact = _compare_checkpoints(_checkpoint(), _checkpoint())
    recurrent = _compare_checkpoints(_checkpoint(), _checkpoint(recurrent="r1"))
    kv = _compare_checkpoints(_checkpoint(), _checkpoint(kv="k1"))

    assert exact == {"passed": True, "mismatches": [], "first_divergence": None}
    assert recurrent["passed"] is False
    assert recurrent["first_divergence"] == {
        "component": "linear_state",
        "layer": 0,
        "part": "recurrent",
    }
    assert kv["first_divergence"] == {
        "component": "full_attention_kv",
        "layer": 3,
        "part": "key",
    }


def test_run_summary_requires_exact_tokens_and_reports_median() -> None:
    runs = [
        {"wall_ms": 40.0, "steps": 2, "generated_token_ids": [9707, 9707]},
        {"wall_ms": 36.0, "steps": 2, "generated_token_ids": [9707, 9707]},
        {"wall_ms": 38.0, "steps": 2, "generated_token_ids": [9707, 9707]},
    ]
    summary = _summarize_runs(runs, expected_token_id=9707)

    assert summary["all_tokens_exact"] is True
    assert summary["median_ms_per_token"] == pytest.approx(19.0)
    assert summary["median_tok_s"] == pytest.approx(1000.0 / 19.0)

    runs[0]["generated_token_ids"] = [9707, 9]
    with pytest.raises(ValueError, match="unexpected token"):
        _summarize_runs(runs, expected_token_id=9707)


def test_candidate_classification_rejects_third_launch_or_no_wall_win() -> None:
    eager = {"median_ms_per_token": 20.0}
    faster = {"median_ms_per_token": 18.0}
    slower = {"median_ms_per_token": 21.0}

    corrupt = _classify_candidate(
        relaunch_passed=False,
        relaunch_first_failure=3,
        recapture_passed=True,
        eager_summary=eager,
        recapture_summary=faster,
    )
    no_win = _classify_candidate(
        relaunch_passed=True,
        relaunch_first_failure=None,
        recapture_passed=True,
        eager_summary=eager,
        recapture_summary=slower,
    )
    accepted = _classify_candidate(
        relaunch_passed=True,
        relaunch_first_failure=None,
        recapture_passed=True,
        eager_summary=eager,
        recapture_summary=faster,
    )

    assert corrupt["status"] == "rejected"
    assert corrupt["decision"] == "reject_third_or_later_relaunch_state_corruption"
    assert no_win["decision"] == "reject_no_end_to_end_wall_win"
    assert accepted["status"] == "accepted"
    assert accepted["decision"] == "promote_state_keyed_graph_replay"
