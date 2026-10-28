from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

import scripts.qwen35_readme_sweep as sweep


CONSERVATIVE_SELECTORS = {
    "HIPENGINE_GGUF_GDN_PREFILL_MODE": "chain",
    "HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE": "baseline",
    "HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE": "baseline",
    "HIPENGINE_GGUF_PREFILL_ROUTER_SELECT_THREADS": "512",
    "HIPENGINE_GGUF_PREFILL_DEVICE_METADATA": "0",
}

GDN_EXACT_SELECTORS = {
    **CONSERVATIVE_SELECTORS,
    "HIPENGINE_GGUF_GDN_PREFILL_MODE": "exact",
}

Q4_SHARED_X_SELECTORS = {
    **CONSERVATIVE_SELECTORS,
    "HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE": "shared_x",
}

def test_conservative_prefill_kernel_profile_sets_complete_selector_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in CONSERVATIVE_SELECTORS:
        monkeypatch.delenv(name, raising=False)

    selectors = sweep._apply_prefill_kernel_profile("conservative")

    assert selectors == CONSERVATIVE_SELECTORS
    assert {name: os.environ[name] for name in selectors} == CONSERVATIVE_SELECTORS


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("gdn_exact", GDN_EXACT_SELECTORS),
        ("q4_shared_x", Q4_SHARED_X_SELECTORS),
        ("q4_pair32", {**CONSERVATIVE_SELECTORS, "HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE": "pair32"}),
    ],
)
def test_split_prefill_kernel_profiles_reenable_exactly_one_family(
    profile: str,
    expected: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in CONSERVATIVE_SELECTORS:
        monkeypatch.delenv(name, raising=False)

    selectors = sweep._apply_prefill_kernel_profile(profile)

    assert selectors == expected
    assert {name: os.environ[name] for name in selectors} == expected


@pytest.mark.parametrize("profile,mode", [
    ("q4_baseline", "baseline"),
    ("q4_pair32_production", "pair32"),
])
def test_q4_prefill_kernel_profile_changes_only_requested_route(
    profile: str,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in CONSERVATIVE_SELECTORS:
        monkeypatch.delenv(name, raising=False)

    expected = {"HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE": mode}
    selectors = sweep._apply_prefill_kernel_profile(profile)

    assert selectors == expected
    assert os.environ["HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE"] == mode
    for name in CONSERVATIVE_SELECTORS.keys() - expected.keys():
        assert name not in os.environ


def test_default_prefill_kernel_profile_does_not_mutate_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", "fused")

    assert sweep._apply_prefill_kernel_profile("default") == {}
    assert os.environ["HIPENGINE_GGUF_GDN_PREFILL_MODE"] == "fused"


def test_conservative_prefill_kernel_profile_rejects_conflicting_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", "fused")

    with pytest.raises(ValueError, match="conflicts with"):
        sweep._apply_prefill_kernel_profile("conservative")


def test_readme_sweep_records_conservative_prefill_kernel_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in CONSERVATIVE_SELECTORS:
        monkeypatch.delenv(name, raising=False)

    captured: dict[str, object] = {}

    def fake_run(
        args,
        model,
        workloads,
        warmup_decode_tokens,
        max_sequence_length,
        compiler_version,
        prefill_config,
    ):
        del model, workloads, warmup_decode_tokens, max_sequence_length, compiler_version, prefill_config
        captured["profile"] = args.prefill_kernel_profile
        captured["selectors"] = args.prefill_kernel_selectors
        return {"ok": True}

    monkeypatch.setattr(sweep, "_run_gguf_sweep", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen35_readme_sweep.py",
            "--engine",
            "gguf",
            "--model",
            str(tmp_path / "model.gguf"),
            "--workloads",
            "512/0",
            "--prefill-kernel-profile",
            "conservative",
        ],
    )

    assert sweep.main() == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert captured == {
        "profile": "conservative",
        "selectors": CONSERVATIVE_SELECTORS,
    }


def test_readme_sweep_rejects_non_gguf_prefill_kernel_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen35_readme_sweep.py",
            "--engine",
            "paro",
            "--model",
            str(tmp_path / "model"),
            "--workloads",
            "512/0",
            "--prefill-kernel-profile",
            "conservative",
        ],
    )

    with pytest.raises(ValueError, match="GGUF-only"):
        sweep.main()
