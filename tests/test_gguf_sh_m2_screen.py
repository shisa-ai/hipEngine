from __future__ import annotations

from scripts.gguf_sh_m2_screen import ENV_NAME, _mode_environment, classify_screen


def _row(context: int) -> dict:
    return {
        "prompt_length": context,
        "dedicated": {
            "scratch": {"rows": min(context + 256, 4096), "allocation_mode": "dedicated"},
            "reclamation": {"delta_bytes": 0},
        },
        "liveness": {
            "scratch": {
                "rows": min(context + 256, 4096),
                "allocation_mode": "liveness_aliased" if context + 256 >= 4096 else "dedicated",
            },
            "reclamation": {"delta_bytes": 0},
        },
        "comparison": {
            "prefill_loss_pct": 0.2,
            "decode_loss_pct": -0.1,
            "tracked_savings_gib": 1.45,
            "whole_gtt_savings_gib": 1.43,
        },
    }


def test_mode_environment_changes_only_scratch_control() -> None:
    base = {"HIPENGINE_HIP_ARCH": "gfx1151", "UNCHANGED": "yes"}

    dedicated = _mode_environment(base, "dedicated")
    liveness = _mode_environment(base, "liveness")

    assert dedicated[ENV_NAME] == "0"
    assert liveness[ENV_NAME] == "1"
    assert dedicated["UNCHANGED"] == liveness["UNCHANGED"] == "yes"
    assert ENV_NAME not in base


def test_classify_screen_promotes_only_complete_exact_matrix() -> None:
    decision = classify_screen(
        [_row(512), _row(4096), _row(32768), _row(65536)],
        state_comparison={"passed": True},
    )

    assert decision["status"] == "promote_liveness_alias"
    assert decision["promotion_passed"] is True
    assert decision["selected_default"] == "liveness"


def test_classify_screen_fails_closed_on_nonpositive_whole_gtt_direction() -> None:
    rows = [_row(512), _row(4096), _row(32768), _row(65536)]
    rows[-1]["comparison"]["whole_gtt_savings_gib"] = 0.0

    decision = classify_screen(rows, state_comparison={"passed": True})

    assert decision["status"] == "reject_whole_gtt_direction"
    assert decision["promotion_passed"] is False
    assert decision["selected_default"] == "dedicated"
