from __future__ import annotations

from scripts.zbook_production_numerics_c1_profile import rank_candidate_roles


def test_rank_candidate_roles_excludes_unattributed_and_keeps_order() -> None:
    roles = [
        {"name": "selected_expert_gate_up", "gpu_us_per_token": 420.0, "share_pct": 32.0},
        {"name": "unattributed", "gpu_us_per_token": 300.0, "share_pct": 23.0},
        {"name": "lm_head", "gpu_us_per_token": 240.0, "share_pct": 18.0},
        {"name": "full_attention_qkv", "gpu_us_per_token": 120.0, "share_pct": 9.0},
    ]

    ranked = rank_candidate_roles(roles, limit=2)

    assert [row["name"] for row in ranked] == ["selected_expert_gate_up", "lm_head"]
    assert ranked[0]["gpu_us_per_token"] == 420.0
