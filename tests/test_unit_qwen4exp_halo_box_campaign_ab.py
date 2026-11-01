from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_halo_box_campaign_ab.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "qwen4exp_halo_box_campaign_ab", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(
    *, mode: str, case_id: str, category: str, prompt_tokens: int,
    repetition: int, prefill_ms: float, decode_ms: float, digest: str = "same"
) -> dict[str, object]:
    return {
        "mode": mode,
        "case_id": case_id,
        "category": category,
        "prompt_tokens": prompt_tokens,
        "prompt_token_ids_sha256": f"prompt-{case_id}",
        "repetition": repetition,
        "prefill_ms": prefill_ms,
        "prefill_tok_s": 1000.0 * prompt_tokens / prefill_ms,
        "decode_ms": decode_ms,
        "decode_transitions": 128,
        "decode_tok_s": 128000.0 / decode_ms,
        "client_wall_s": (prefill_ms + decode_ms) / 1000.0,
        "output_token_count": 129,
        "output_token_ids": [1] * 129,
        "output_token_ids_sha256": digest,
    }


def test_arm_sequence_is_balanced_and_reversed_by_case() -> None:
    module = _load_script()

    first = module.arm_sequence(0)
    second = module.arm_sequence(1)

    assert first == ("before", "after", "after", "before", "before", "after")
    assert second == tuple(reversed(first))
    assert first.count("before") == first.count("after") == 3
    assert second.count("before") == second.count("after") == 3


def test_summarize_campaign_ab_uses_complete_walls_and_requires_exact_outputs() -> None:
    module = _load_script()
    samples: list[dict[str, object]] = []
    for case_index, (case_id, category, prompt_tokens) in enumerate(
        (("code-p512", "code", 512), ("general_en-p512", "general_en", 512))
    ):
        for repetition, mode in enumerate(module.arm_sequence(case_index)):
            samples.append(
                _sample(
                    mode=mode,
                    case_id=case_id,
                    category=category,
                    prompt_tokens=prompt_tokens,
                    repetition=repetition,
                    prefill_ms=10.0 if mode == "before" else 8.0,
                    decode_ms=20.0,
                )
            )

    summary = module.summarize_campaign_ab(samples, repetitions_per_mode=3)

    assert summary["correctness"] == {
        "within_mode_deterministic": True,
        "cross_mode_output_exact": True,
        "mismatched_case_ids": [],
    }
    shape = summary["by_shape"]["512"]
    assert shape["before_prefill_tok_s_weighted"] == pytest.approx(51200.0)
    assert shape["after_prefill_tok_s_weighted"] == pytest.approx(64000.0)
    assert shape["after_over_before_prefill"] == pytest.approx(1.25)
    assert shape["after_over_before_decode"] == pytest.approx(1.0)
    assert summary["by_case"]["code-p512"]["samples_per_mode"] == 3

    for row in samples:
        if row["mode"] == "after":
            row["output_token_ids_sha256"] = "different"
    with pytest.raises(ValueError, match="cross-mode output mismatch"):
        module.summarize_campaign_ab(samples, repetitions_per_mode=3)


def test_summarize_campaign_ab_rejects_incomplete_arm_counts() -> None:
    module = _load_script()
    samples = [
        _sample(
            mode="before",
            case_id="code-p512",
            category="code",
            prompt_tokens=512,
            repetition=0,
            prefill_ms=10.0,
            decode_ms=20.0,
        ),
        _sample(
            mode="after",
            case_id="code-p512",
            category="code",
            prompt_tokens=512,
            repetition=1,
            prefill_ms=8.0,
            decode_ms=20.0,
        ),
    ]

    with pytest.raises(ValueError, match="expected 3 samples per mode"):
        module.summarize_campaign_ab(samples, repetitions_per_mode=3)


def test_gr_dpp_scope_binds_one_leaf_per_arm_and_restores_every_field() -> None:
    """``sample()`` enters this scope per arm; only the counted leaf differs.

    Both arms hold the wave-scale parent on and close the two preemptors, so
    the sibling and the parent counts are the same GR-read surface. The scope
    must put every field and the pinned key back on exit.
    """

    module = _load_script()
    # The plan may publish the parent off; the arm holds it on, so the binding
    # and its release are both observable.
    runner = SimpleNamespace(
        gr_wide=True, gr_wave_scale=False, gr_dpp_reduction=True)
    environment = {"HIPENGINE_QWEN4_EXP_GR_IU8": "1"}
    for mode, sibling in (("before", False), ("after", True)):
        with module.route_scope(
                "gr-dpp-reduction", mode, runner, environment=environment):
            assert runner.gr_dpp_reduction is sibling
            assert runner.gr_wave_scale is True
            assert runner.gr_wide is False
            assert environment["HIPENGINE_QWEN4_EXP_GR_IU8"] == "0"
        assert runner.gr_dpp_reduction is True
        assert runner.gr_wave_scale is False
        assert runner.gr_wide is True
        assert environment["HIPENGINE_QWEN4_EXP_GR_IU8"] == "1"


def test_gr_dpp_scope_refuses_an_invalid_arm_or_a_missing_runner() -> None:
    module = _load_script()
    with pytest.raises(ValueError, match="invalid campaign A/B mode"):
        module.route_scope("gr-dpp-reduction", "sideways", SimpleNamespace())
    with pytest.raises(ValueError, match="needs the owning runner"):
        module.route_scope("gr-dpp-reduction", "before", None)


def test_gr_dpp_control_record_names_the_arm_it_binds() -> None:
    """The record states the premise both arms ride, not the plan's default."""

    module = _load_script()
    control = module.gr_dpp_control_record(
        SimpleNamespace(gr_dpp_reduction=True))
    assert control["runner_field"] == "gr_dpp_reduction"
    assert (control["before"], control["after"]) == (False, True)
    assert control["published_value"] is True
    assert control["forced_runner_fields"] == {
        "gr_wide": False, "gr_wave_scale": True}
    assert control["forced_environment"] == module.GR_WAVE_ENV
    assert control["counted_variants"] == [
        "coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact",
        "coltile2_branch4_rowbatch4_wave_scale_dpp_k320_full_rows_f32_exact",
    ]
    assert control["counted_parent_variant"] == (
        "coltile2_branch4_rowbatch4_wave_scale_f32_exact")
