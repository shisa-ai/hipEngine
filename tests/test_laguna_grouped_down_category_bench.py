from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import scripts.laguna_grouped_down_category_bench as benchmark
from scripts.laguna_grouped_down_category_bench import (
    ATTENTION_HIPBLASLT_ABSOLUTE_COMPARISON,
    CUMULATIVE_CONTROL_COMPARISON,
    F16_WMMA_COMP_SWA_COMPARISON,
    GLOBAL_QROW2_ONLINE_COMPARISON,
    GROUPED_COMBINE_COMPARISON,
    MODES,
    PREFILL_350_COMPARISON,
    PRODUCTION_ABSOLUTE_COMPARISON,
    SWA_DECODE_BOUNDED_EXP_COMPARISON,
    SWA_QROW2_COMPARISON,
    SWA_QROW2_ONLINE_COMPARISON,
    _aggregate,
    _load_shape_screen,
    _mode_order,
    _oracle_for_candidate,
    _paired_free_running,
    _promotion_gate,
    _prompt_token_ids,
    _teacher_forced_quality,
)

HORIZONS = (16, 32)
CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")


def test_production_absolute_comparison_uses_current_defaults() -> None:
    assert PRODUCTION_ABSOLUTE_COMPARISON.modes == (
        "all_exact",
        "production_absolute_candidate",
    )
    assert not PRODUCTION_ABSOLUTE_COMPARISON.require_performance_gate
    lane = benchmark._PREFILL_LANE_CONFIGURATIONS[
        "production_absolute_candidate"
    ]
    assert (
        lane.selected_gate_up_mode
        == "mmq128x32_d8_f32_wavecols_direct_doublebuf_rawprefetch_ge512"
    )
    assert (
        lane.selected_down_mode
        == "mmq64x64_d4_f32_q6_wavecols_direct_q4"
    )
    assert (
        lane.global_prefill_variant
        == "global_context_rows_qrow4_m128_online_spans"
    )
    assert (
        lane.swa_prefill_variant
        == "swa_context_rows_qrow4_m128_online_spans"
    )
    assert lane.f16_projection_mode == "hipblaslt_range_direct"
    assert lane.dense_q4_prefill_mode == "wmma_pack8"


def test_attention_hipblaslt_comparison_adds_only_attention_candidate() -> None:
    assert ATTENTION_HIPBLASLT_ABSOLUTE_COMPARISON.modes == (
        "all_exact",
        "attention_hipblaslt_candidate",
    )
    lane = benchmark._PREFILL_LANE_CONFIGURATIONS[
        "attention_hipblaslt_candidate"
    ]
    production = benchmark._PREFILL_LANE_CONFIGURATIONS[
        "production_absolute_candidate"
    ]
    assert not production.attention_hipblaslt
    assert lane.attention_hipblaslt
    assert lane.selected_gate_up_mode == production.selected_gate_up_mode
    assert lane.selected_down_mode == production.selected_down_mode
    assert lane.f16_projection_mode == production.f16_projection_mode
    assert lane.dense_q4_prefill_mode == production.dense_q4_prefill_mode


def test_swa_bounded_exp_category_saturates_ring_and_switches_only_candidate(
    monkeypatch,
) -> None:
    calls: list[bool] = []

    class FakeSession:
        def set_decode_swa_bounded_exp(self, enabled: bool) -> None:
            calls.append(enabled)

    monkeypatch.setattr(
        benchmark,
        "_session",
        lambda _owner, _args: FakeSession(),
    )
    for mode in SWA_DECODE_BOUNDED_EXP_COMPARISON.modes:
        benchmark._session_for_mode(
            object(),
            SimpleNamespace(),
            mode,
            comparison=SWA_DECODE_BOUNDED_EXP_COMPARISON,
        )

    assert calls == [False, True]
    assert _prompt_token_ids(
        {"token_ids": (1, 2, 3)},
        comparison=SWA_DECODE_BOUNDED_EXP_COMPARISON,
    ) == (1, 2, 3) + (2, 3) * 254 + (2,)


def _rows(
    *,
    modes: tuple[str, str] = MODES,
    baseline_prefill: float = 2.0,
    candidate_prefill: float = 1.8,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prompt_index, category in enumerate(CATEGORIES):
        prompt_id = f"prompt_{prompt_index}"
        for repetition in range(3):
            generated = list(range(32))
            for mode in modes:
                prefill = baseline_prefill if mode == modes[0] else candidate_prefill
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "category": category,
                        "prompt_tokens": 96,
                        "mode": mode,
                        "repetition": repetition,
                        "prefill_seconds": prefill,
                        "ttft_seconds": prefill,
                        "checkpoints": {
                            str(horizon): {
                                "generated_token_ids": generated[:horizon],
                                "generated_ids_sha256": f"{mode}-{prompt_id}-{horizon}",
                                "decode_forward_calls": horizon - 1,
                                "decode_seconds": float(horizon - 1),
                                "output_tokens": horizon,
                                "total_seconds": prefill + horizon - 1,
                                "decode_tok_s": 1.0,
                                "e2e_output_tok_s": horizon / (prefill + horizon - 1),
                            }
                            for horizon in HORIZONS
                        },
                    }
                )
    return rows


def _teacher_rows(*, top1_agreement: float = 1.0, max_kl: float = 1e-3):
    rows = []
    for index, category in enumerate(CATEGORIES):
        steps = []
        for step in range(32):
            matched = step / 32 < top1_agreement
            steps.append(
                {
                    "index": step,
                    "kl_divergence": max_kl,
                    "direct_top1": step,
                    "adaptive_grouped_smallm_top1": step if matched else step + 100,
                    "top1_agreement": matched,
                    "finite": True,
                }
            )
        rows.append(
            {"prompt_id": f"prompt_{index}", "category": category, "steps": steps}
        )
    return rows


def test_grouped_down_category_mode_order_is_counterbalanced() -> None:
    for index in range(10):
        assert _mode_order(index, 1) == tuple(reversed(_mode_order(index, 0)))
        assert _mode_order(index, 2) == _mode_order(index, 0)


def test_grouped_combine_category_accepts_exact_nonregressive_wall() -> None:
    comparison = GROUPED_COMBINE_COMPARISON
    rows = _rows(
        modes=comparison.modes,
        baseline_prefill=2.0,
        candidate_prefill=2.001,
    )
    free_running = _paired_free_running(
        rows,
        HORIZONS,
        comparison=comparison,
    )
    aggregate = _aggregate(rows, HORIZONS, comparison=comparison)
    gate = _promotion_gate(
        aggregate,
        free_running,
        _teacher_forced_quality(_teacher_rows()),
        {"pass": True},
        {"pass": True},
        horizons=HORIZONS,
        recovered=True,
        comparison=comparison,
    )

    assert _mode_order(0, 0, comparison=comparison) == comparison.modes
    assert aggregate[comparison.aggregate_key]["prefill_speedup"] == pytest.approx(
        2.0 / 2.001
    )
    assert free_running["all_pairs_exact"] is True
    assert gate["pass"] is True
    assert gate["policy"]["performance"].startswith("aggregate/category prefill >=0.995x")


def test_grouped_combine_category_loads_matching_screen(tmp_path) -> None:
    screen = tmp_path / "combine-screen.json"
    screen.write_text(
        json.dumps(
            {
                "kind": GROUPED_COMBINE_COMPARISON.screen_kind,
                "status": GROUPED_COMBINE_COMPARISON.screen_status,
                "pass": True,
                "screen": {
                    "pass": True,
                    "regressed_rows": [],
                    "effective_speedup": 0.9997,
                },
                "model": {"sha256": "model-sha"},
                "repo": {"revision": "candidate-revision"},
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(shape_screen=screen, model_sha256="model-sha")

    result = _load_shape_screen(
        args,
        comparison=GROUPED_COMBINE_COMPARISON,
    )

    assert result["pass"] is True
    assert result["comparison"] == "grouped_combine"
    assert result["aggregate_speedup"] == pytest.approx(0.9997)


def test_swa_qrow2_category_resolves_explicit_session_variants(monkeypatch) -> None:
    calls: list[str | None] = []

    def fake_session(_owner, _args, *, swa_prefill_variant=None):
        calls.append(swa_prefill_variant)
        return SimpleNamespace()

    monkeypatch.setattr(benchmark, "_session", fake_session)
    for mode in SWA_QROW2_COMPARISON.modes:
        benchmark._session_for_mode(
            object(),
            SimpleNamespace(),
            mode,
            comparison=SWA_QROW2_COMPARISON,
        )

    assert calls == [
        "swa_context_rows_wave32_exact_spans",
        "swa_context_rows_qrow2_m128_c128_exact_spans",
    ]


def test_swa_qrow2_category_loads_exact_full_model_screen(tmp_path) -> None:
    screen = tmp_path / "swa-qrow2-screen.json"
    artifact = {
        "kind": SWA_QROW2_COMPARISON.screen_kind,
        "status": SWA_QROW2_COMPARISON.screen_status,
        "pass": True,
        "promotion": {"pass": True, "failed_checks": []},
        "model": {"sha256": "model-sha"},
        "repo": {"revision": "candidate-revision"},
    }
    screen.write_text(json.dumps(artifact), encoding="utf-8")
    args = SimpleNamespace(shape_screen=screen, model_sha256="model-sha")

    result = _load_shape_screen(args, comparison=SWA_QROW2_COMPARISON)

    assert result["pass"] is True
    assert result["comparison"] == "swa_qrow2"
    assert SWA_QROW2_COMPARISON.modes == ("wave32_exact", "qrow2_32_exact")

    artifact["model"]["sha256"] = "wrong-model"
    screen.write_text(json.dumps(artifact), encoding="utf-8")
    assert _load_shape_screen(args, comparison=SWA_QROW2_COMPARISON)["pass"] is False


def test_swa_qrow2_online_category_resolves_variants_and_screen(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[str | None] = []

    def fake_session(_owner, _args, *, swa_prefill_variant=None):
        calls.append(swa_prefill_variant)
        return SimpleNamespace()

    monkeypatch.setattr(benchmark, "_session", fake_session)
    for mode in SWA_QROW2_ONLINE_COMPARISON.modes:
        benchmark._session_for_mode(
            object(),
            SimpleNamespace(),
            mode,
            comparison=SWA_QROW2_ONLINE_COMPARISON,
        )
    assert calls == [
        "swa_context_rows_qrow2_m128_c128_exact_spans",
        "swa_context_rows_qrow2_online_spans",
    ]

    screen = tmp_path / "swa-online-screen.json"
    artifact = {
        "kind": SWA_QROW2_ONLINE_COMPARISON.screen_kind,
        "status": SWA_QROW2_ONLINE_COMPARISON.screen_status,
        "pass": True,
        "correctness": {"pass": True, "failed_checks": []},
        "candidate": {"variant": "swa_context_rows_qrow2_online_spans"},
        "model": {"sha256": "model-sha"},
        "repo": {"revision": "candidate-revision"},
    }
    screen.write_text(json.dumps(artifact), encoding="utf-8")
    args = SimpleNamespace(shape_screen=screen, model_sha256="model-sha")
    result = _load_shape_screen(
        args,
        comparison=SWA_QROW2_ONLINE_COMPARISON,
    )
    assert result["pass"] is True
    assert result["comparison"] == "swa_qrow2_online"
    assert result["candidate_variant"] == "swa_context_rows_qrow2_online_spans"

    artifact["candidate"]["variant"] = "wrong-variant"
    screen.write_text(json.dumps(artifact), encoding="utf-8")
    assert _load_shape_screen(
        args,
        comparison=SWA_QROW2_ONLINE_COMPARISON,
    )["pass"] is False


def test_global_qrow2_online_category_resolves_variants_and_screen(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[str | None] = []

    def fake_session(_owner, _args, *, global_prefill_variant=None):
        calls.append(global_prefill_variant)
        return SimpleNamespace()

    monkeypatch.setattr(benchmark, "_session", fake_session)
    for mode in GLOBAL_QROW2_ONLINE_COMPARISON.modes:
        benchmark._session_for_mode(
            object(),
            SimpleNamespace(),
            mode,
            comparison=GLOBAL_QROW2_ONLINE_COMPARISON,
        )
    assert calls == [
        "global_context_rows_spans",
        "global_context_rows_qrow2_online_spans",
    ]

    screen = tmp_path / "global-online-screen.json"
    artifact = {
        "kind": GLOBAL_QROW2_ONLINE_COMPARISON.screen_kind,
        "status": GLOBAL_QROW2_ONLINE_COMPARISON.screen_status,
        "pass": True,
        "promotion": {
            "pass": True,
            "failed_checks": [],
            "effective_speedup": 1.17,
        },
        "model": {"sha256": "model-sha"},
        "repo": {"revision": "candidate-revision"},
    }
    screen.write_text(json.dumps(artifact), encoding="utf-8")
    args = SimpleNamespace(shape_screen=screen, model_sha256="model-sha")
    result = _load_shape_screen(
        args,
        comparison=GLOBAL_QROW2_ONLINE_COMPARISON,
    )
    assert result["pass"] is True
    assert result["comparison"] == "global_qrow2_online"
    assert result["aggregate_speedup"] == pytest.approx(1.17)

    artifact["model"]["sha256"] = "wrong-model"
    screen.write_text(json.dumps(artifact), encoding="utf-8")
    assert _load_shape_screen(
        args,
        comparison=GLOBAL_QROW2_ONLINE_COMPARISON,
    )["pass"] is False


def test_cumulative_control_resolves_explicit_exact_and_shipping_lanes(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeSession:
        def __init__(self, row: dict[str, object]) -> None:
            self.row = row

        def set_selected_gate_up_mode(self, mode: str) -> None:
            self.row["selected_gate_up_mode"] = mode

        def set_selected_down_mode(self, mode: str) -> None:
            self.row["selected_down_mode"] = mode

        def set_f16_prefill_mode(self, mode: str) -> None:
            self.row["f16_projection_mode"] = mode

        def set_dense_q4_prefill_mode(self, mode: str) -> None:
            self.row["dense_q4_prefill_mode"] = mode

        def set_prefill_attention_hipblaslt(self, enabled: bool) -> None:
            self.row["attention_hipblaslt"] = enabled

        def prefill(self, _token_ids, *, use_bulk: bool):
            self.row["f16_prefill_mode"] = benchmark.os.environ.get(
                "HIPENGINE_LAGUNA_F16_PREFILL"
            )
            self.row["use_bulk"] = use_bulk
            return object()

    def fake_session(
        _owner,
        _args,
        *,
        global_prefill_variant=None,
        swa_prefill_variant=None,
    ):
        row = {
            "global_prefill_variant": global_prefill_variant,
            "swa_prefill_variant": swa_prefill_variant,
        }
        calls.append(row)
        return FakeSession(row)

    monkeypatch.setattr(benchmark, "_session", fake_session)
    monkeypatch.setenv("HIPENGINE_LAGUNA_F16_PREFILL", "sentinel")
    for mode in CUMULATIVE_CONTROL_COMPARISON.modes:
        session = benchmark._session_for_mode(
            object(),
            SimpleNamespace(),
            mode,
            comparison=CUMULATIVE_CONTROL_COMPARISON,
        )
        benchmark._prefill_for_mode(
            session,
            (1, 2),
            mode,
            CUMULATIVE_CONTROL_COMPARISON,
        )

    assert calls == [
        {
            "global_prefill_variant": "global_context_rows_spans",
            "swa_prefill_variant": "swa_context_rows_qrow2_m128_c128_exact_spans",
            "selected_gate_up_mode": "direct",
            "selected_down_mode": "adaptive_grouped_smallm_fused",
            "f16_projection_mode": "retained",
            "dense_q4_prefill_mode": "retained",
            "attention_hipblaslt": False,
            "f16_prefill_mode": "tiled",
            "use_bulk": True,
        },
        {
            "global_prefill_variant": "global_context_rows_qrow2_online_spans",
            "swa_prefill_variant": "swa_context_rows_qrow2_online_spans",
            "selected_gate_up_mode": "direct",
            "selected_down_mode": "adaptive_grouped_smallm_fused",
            "f16_projection_mode": "retained",
            "dense_q4_prefill_mode": "retained",
            "attention_hipblaslt": False,
            "f16_prefill_mode": "wmma_comp_swa",
            "use_bulk": True,
        },
    ]
    assert benchmark.os.environ["HIPENGINE_LAGUNA_F16_PREFILL"] == "sentinel"

    args = SimpleNamespace(
        shape_screen=tmp_path / "does-not-exist.json",
        model_sha256="model-sha",
    )
    screen = _load_shape_screen(
        args,
        comparison=CUMULATIVE_CONTROL_COMPARISON,
    )
    assert screen == {
        "pass": True,
        "path": None,
        "sha256": None,
        "revision": None,
        "aggregate_speedup": None,
        "grouped_min_rows": None,
        "model_sha256": "model-sha",
        "candidate_variant": None,
        "comparison": "cumulative_control",
        "role": "not_applicable_control_ledger",
    }


def test_cumulative_control_reports_performance_without_gating_it() -> None:
    comparison = CUMULATIVE_CONTROL_COMPARISON
    rows = _rows(
        modes=comparison.modes,
        baseline_prefill=1.0,
        candidate_prefill=1.2,
    )
    gate = _promotion_gate(
        _aggregate(rows, HORIZONS, comparison=comparison),
        _paired_free_running(rows, HORIZONS, comparison=comparison),
        _teacher_forced_quality(_teacher_rows()),
        {"pass": True},
        {"pass": True},
        horizons=HORIZONS,
        recovered=True,
        comparison=comparison,
    )

    assert gate["pass"] is True
    assert gate["failed_checks"] == []
    assert gate["policy"]["performance"] == "reported; no admission threshold"


def test_cumulative_control_oracle_uses_explicit_shipping_lane(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    configured: list[tuple[str, object]] = []

    class FakeSession:
        def set_selected_gate_up_mode(self, mode: str) -> None:
            configured.append(("selected_gate_up", mode))

        def set_selected_down_mode(self, mode: str) -> None:
            configured.append(("selected_down", mode))

        def set_f16_prefill_mode(self, mode: str) -> None:
            configured.append(("f16_projection", mode))

        def set_dense_q4_prefill_mode(self, mode: str) -> None:
            configured.append(("dense_q4", mode))

        def set_prefill_attention_hipblaslt(self, enabled: bool) -> None:
            configured.append(("attention_hipblaslt", enabled))

    def fake_oracle(_owner, _args, **kwargs):
        configurator = kwargs.pop("session_configurator")
        configurator(FakeSession())
        calls.append(
            {
                **kwargs,
                "f16_prefill_mode": benchmark.os.environ.get(
                    "HIPENGINE_LAGUNA_F16_PREFILL"
                ),
            }
        )
        return {"pass": True}

    monkeypatch.setattr(benchmark, "_oracle_gate", fake_oracle)
    monkeypatch.setenv("HIPENGINE_LAGUNA_F16_PREFILL", "sentinel")

    assert _oracle_for_candidate(
        object(),
        SimpleNamespace(),
        comparison=CUMULATIVE_CONTROL_COMPARISON,
    ) == {"pass": True}
    assert calls == [
        {
            "global_prefill_variant": "global_context_rows_qrow2_online_spans",
            "swa_prefill_variant": "swa_context_rows_qrow2_online_spans",
            "f16_prefill_mode": "wmma_comp_swa",
        }
    ]
    assert configured == [
        ("selected_gate_up", "direct"),
        ("selected_down", "adaptive_grouped_smallm_fused"),
        ("f16_projection", "retained"),
        ("dense_q4", "retained"),
        ("attention_hipblaslt", False),
    ]
    assert benchmark.os.environ["HIPENGINE_LAGUNA_F16_PREFILL"] == "sentinel"


def test_prefill_350_resolves_complete_shipping_and_candidate_lanes(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeSession:
        def __init__(self, row: dict[str, object]) -> None:
            self.row = row

        def set_selected_gate_up_mode(self, mode: str) -> None:
            self.row["selected_gate_up_mode"] = mode

        def set_selected_down_mode(self, mode: str) -> None:
            self.row["selected_down_mode"] = mode

        def set_f16_prefill_mode(self, mode: str) -> None:
            self.row["f16_projection_mode"] = mode

        def set_dense_q4_prefill_mode(self, mode: str) -> None:
            self.row["dense_q4_prefill_mode"] = mode

        def set_prefill_attention_hipblaslt(self, enabled: bool) -> None:
            self.row["attention_hipblaslt"] = enabled

        def prefill(self, _token_ids, *, use_bulk: bool):
            self.row["retained_f16_strategy"] = benchmark.os.environ.get(
                "HIPENGINE_LAGUNA_F16_PREFILL"
            )
            self.row["use_bulk"] = use_bulk
            return object()

    def fake_session(
        _owner,
        _args,
        *,
        global_prefill_variant=None,
        swa_prefill_variant=None,
    ):
        row = {
            "global_prefill_variant": global_prefill_variant,
            "swa_prefill_variant": swa_prefill_variant,
        }
        calls.append(row)
        return FakeSession(row)

    monkeypatch.setattr(benchmark, "_session", fake_session)
    monkeypatch.setenv("HIPENGINE_LAGUNA_F16_PREFILL", "sentinel")
    for mode in PREFILL_350_COMPARISON.modes:
        session = benchmark._session_for_mode(
            object(),
            SimpleNamespace(),
            mode,
            comparison=PREFILL_350_COMPARISON,
        )
        benchmark._prefill_for_mode(
            session,
            (1, 2),
            mode,
            PREFILL_350_COMPARISON,
        )

    common = {
        "global_prefill_variant": "global_context_rows_qrow2_online_spans",
        "swa_prefill_variant": "swa_context_rows_qrow2_online_spans",
        "retained_f16_strategy": "wmma_comp_swa",
        "use_bulk": True,
    }
    assert calls == [
        {
            **common,
            "selected_gate_up_mode": "direct",
            "selected_down_mode": "adaptive_grouped_smallm_fused",
            "f16_projection_mode": "retained",
            "dense_q4_prefill_mode": "retained",
            "attention_hipblaslt": False,
        },
        {
            **common,
            "selected_gate_up_mode": "mmq128x32_d8_f32",
            "selected_down_mode": "mmq64x32_d4_f32",
            "f16_projection_mode": "hipblaslt_scaled",
            "dense_q4_prefill_mode": "wmma_pack8",
            "attention_hipblaslt": False,
        },
    ]
    assert benchmark.os.environ["HIPENGINE_LAGUNA_F16_PREFILL"] == "sentinel"


def test_prefill_350_oracle_configures_the_candidate_session(monkeypatch) -> None:
    configured: list[tuple[str, object]] = []
    oracle_kwargs: list[dict[str, object]] = []

    class FakeSession:
        def set_selected_gate_up_mode(self, mode: str) -> None:
            configured.append(("selected_gate_up", mode))

        def set_selected_down_mode(self, mode: str) -> None:
            configured.append(("selected_down", mode))

        def set_f16_prefill_mode(self, mode: str) -> None:
            configured.append(("f16_projection", mode))

        def set_dense_q4_prefill_mode(self, mode: str) -> None:
            configured.append(("dense_q4", mode))

        def set_prefill_attention_hipblaslt(self, enabled: bool) -> None:
            configured.append(("attention_hipblaslt", enabled))

    def fake_oracle(_owner, _args, **kwargs):
        configurator = kwargs.pop("session_configurator")
        configurator(FakeSession())
        oracle_kwargs.append(
            {
                **kwargs,
                "retained_f16_strategy": benchmark.os.environ.get(
                    "HIPENGINE_LAGUNA_F16_PREFILL"
                ),
            }
        )
        return {"pass": True}

    monkeypatch.setattr(benchmark, "_oracle_gate", fake_oracle)
    monkeypatch.setenv("HIPENGINE_LAGUNA_F16_PREFILL", "sentinel")

    assert _oracle_for_candidate(
        object(),
        SimpleNamespace(),
        comparison=PREFILL_350_COMPARISON,
    ) == {"pass": True}
    assert configured == [
        ("selected_gate_up", "mmq128x32_d8_f32"),
        ("selected_down", "mmq64x32_d4_f32"),
        ("f16_projection", "hipblaslt_scaled"),
        ("dense_q4", "wmma_pack8"),
        ("attention_hipblaslt", False),
    ]
    assert oracle_kwargs == [
        {
            "global_prefill_variant": "global_context_rows_qrow2_online_spans",
            "swa_prefill_variant": "swa_context_rows_qrow2_online_spans",
            "retained_f16_strategy": "wmma_comp_swa",
        }
    ]
    assert benchmark.os.environ["HIPENGINE_LAGUNA_F16_PREFILL"] == "sentinel"


def test_f16_wmma_comp_swa_category_requires_matching_compensated_screen(
    tmp_path,
) -> None:
    screen = tmp_path / "f16-comp-screen.json"
    artifact = {
        "kind": F16_WMMA_COMP_SWA_COMPARISON.screen_kind,
        "status": F16_WMMA_COMP_SWA_COMPARISON.screen_status,
        "pass": True,
        "summary": {"pass": True, "failed_checks": []},
        "protocol": {"candidate_variant": "wmma_comp"},
        "repo": {"revision": "candidate-revision"},
    }
    screen.write_text(json.dumps(artifact), encoding="utf-8")
    args = SimpleNamespace(shape_screen=screen, model_sha256="model-sha")

    result = _load_shape_screen(
        args,
        comparison=F16_WMMA_COMP_SWA_COMPARISON,
    )
    assert result["pass"] is True
    assert result["comparison"] == "f16_wmma_comp_swa"
    assert result["model_sha256"] is None
    assert result["candidate_variant"] == "wmma_comp"

    artifact["protocol"]["candidate_variant"] = "wmma"
    screen.write_text(json.dumps(artifact), encoding="utf-8")
    assert _load_shape_screen(
        args,
        comparison=F16_WMMA_COMP_SWA_COMPARISON,
    )["pass"] is False


def test_f16_wmma_comp_swa_reports_but_does_not_require_exact_trajectories() -> None:
    comparison = F16_WMMA_COMP_SWA_COMPARISON
    rows = _rows(
        modes=comparison.modes,
        baseline_prefill=2.0,
        candidate_prefill=1.0,
    )
    for row in rows:
        if row["mode"] == comparison.modes[1]:
            for checkpoint in row["checkpoints"].values():
                checkpoint["generated_token_ids"][-1] = 999
    free_running = _paired_free_running(
        rows,
        HORIZONS,
        comparison=comparison,
    )
    gate = _promotion_gate(
        _aggregate(rows, HORIZONS, comparison=comparison),
        free_running,
        _teacher_forced_quality(_teacher_rows()),
        {"pass": True},
        {"pass": True},
        horizons=HORIZONS,
        recovered=True,
        comparison=comparison,
    )

    assert free_running["all_pairs_exact"] is False
    assert free_running["same_mode_repeat_deterministic"] is True
    assert gate["pass"] is True
    assert gate["policy"]["free_running_ids"].startswith("report complete")


def test_grouped_down_category_gate_accepts_quality_and_full_model_win() -> None:
    rows = _rows()
    free_running = _paired_free_running(rows, HORIZONS)
    teacher = _teacher_forced_quality(_teacher_rows())
    aggregate = _aggregate(rows, HORIZONS)
    promotion = _promotion_gate(
        aggregate,
        free_running,
        teacher,
        {"pass": True},
        {"pass": True},
        horizons=HORIZONS,
        recovered=True,
    )

    assert free_running["all_pairs_exact"] is True
    assert teacher["pass"] is True
    assert teacher["top1_agreement"] == 1.0
    comparison = aggregate["adaptive_grouped_smallm_vs_direct"]
    assert comparison["prefill_speedup"] == pytest.approx(2.0 / 1.8)
    assert promotion == {
        "pass": True,
        "failed_checks": [],
        "policy": promotion["policy"],
    }


def test_grouped_down_quality_fails_closed_on_category_or_kl() -> None:
    low_top1 = _teacher_forced_quality(_teacher_rows(top1_agreement=0.875))
    assert low_top1["pass"] is False
    assert "general_en_top1_below_0.9" in low_top1["failed_checks"]

    high_kl = _teacher_forced_quality(_teacher_rows(max_kl=0.051))
    assert high_kl["pass"] is False
    assert "max_kl_above_0.05" in high_kl["failed_checks"]


def test_grouped_down_free_running_ids_are_required() -> None:
    rows = _rows()
    mismatch_rows = deepcopy(rows)
    candidate = next(
        row for row in mismatch_rows if row["mode"] == "adaptive_grouped_smallm"
    )
    candidate["checkpoints"]["32"]["generated_token_ids"][-1] = 999
    free_running = _paired_free_running(mismatch_rows, HORIZONS)

    assert free_running["all_pairs_exact"] is False
    gate = _promotion_gate(
        _aggregate(mismatch_rows, HORIZONS),
        free_running,
        _teacher_forced_quality(_teacher_rows()),
        {"pass": True},
        {"pass": True},
        horizons=HORIZONS,
        recovered=True,
    )
    assert gate["pass"] is False
    assert "free_running_pairs_not_exact" in gate["failed_checks"]

    nondeterministic_rows = deepcopy(rows)
    repeat = next(
        row
        for row in nondeterministic_rows
        if row["mode"] == "adaptive_grouped_smallm" and row["repetition"] == 1
    )
    repeat["checkpoints"]["32"]["generated_ids_sha256"] = "changed"
    deterministic_gate = _promotion_gate(
        _aggregate(nondeterministic_rows, HORIZONS),
        _paired_free_running(nondeterministic_rows, HORIZONS),
        _teacher_forced_quality(_teacher_rows()),
        {"pass": True},
        {"pass": True},
        horizons=HORIZONS,
        recovered=True,
    )
    assert deterministic_gate["pass"] is False
    assert "free_running_repeat_not_deterministic" in deterministic_gate["failed_checks"]


def test_grouped_down_category_gate_rejects_regression_or_missing_screen() -> None:
    rows = _rows()
    for row in rows:
        if (
            row["mode"] == "adaptive_grouped_smallm"
            and row["category"] == "general_ja"
        ):
            row["prefill_seconds"] = 2.1
            row["ttft_seconds"] = 2.1
            for checkpoint in row["checkpoints"].values():
                checkpoint["total_seconds"] = 2.1 + checkpoint["decode_seconds"]
                checkpoint["e2e_output_tok_s"] = (
                    checkpoint["output_tokens"] / checkpoint["total_seconds"]
                )
    aggregate = _aggregate(rows, HORIZONS)
    gate = _promotion_gate(
        aggregate,
        _paired_free_running(rows, HORIZONS),
        _teacher_forced_quality(_teacher_rows()),
        {"pass": True},
        {"pass": False},
        horizons=HORIZONS,
        recovered=True,
    )

    assert gate["pass"] is False
    assert "general_ja_prefill_regressed" in gate["failed_checks"]
    assert "shape_screen_failed" in gate["failed_checks"]
