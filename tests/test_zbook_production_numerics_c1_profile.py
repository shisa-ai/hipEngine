from __future__ import annotations

from scripts.zbook_production_numerics_c1_profile import (
    rank_candidate_roles,
    rank_device_stages,
    resolve_roctx_sdk,
    summarize_role_launches,
)


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


def test_resolve_roctx_sdk_falls_back_to_base_prefix(tmp_path) -> None:
    missing = tmp_path / "venv" / "librocprofiler-sdk-roctx.so.1"
    base = tmp_path / "base"
    fallback = (
        base
        / "lib"
        / "python3.13"
        / "site-packages"
        / "_rocm_sdk_core"
        / "lib"
        / "librocprofiler-sdk-roctx.so.1"
    )
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"sdk")

    assert resolve_roctx_sdk(missing, base_prefix=base, python_version="3.13") == fallback


def test_zero_duration_fallback_ranks_device_stages_and_launches() -> None:
    stages = rank_device_stages(
        {"decode_lm_head_sample": 12.0, "decode_layers": 36.0},
        decode_steps=24,
        limit=2,
    )
    assert [row["name"] for row in stages] == ["decode_layers", "decode_lm_head_sample"]
    assert stages[0]["ms_per_token"] == 1.5

    launches = summarize_role_launches(
        [
            {"role": "lm_head", "family": "q6", "kernel": "head"},
            {"role": "lm_head", "family": "q6", "kernel": "head"},
            {"role": "gdn_input_projections", "family": "q8", "kernel": "qkv"},
        ],
        decode_steps=2,
    )
    assert launches[0] == {
        "name": "lm_head",
        "calls": 2,
        "calls_per_token": 1.0,
        "kernel_families": ["q6"],
        "kernels": ["head"],
    }
