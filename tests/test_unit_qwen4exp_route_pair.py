from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_route_pair.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen4exp_route_pair", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_counterbalanced_order_alternates_first_route() -> None:
    module = _load_script()

    assert module._counterbalanced_order(0) == ("bound", "override")
    assert module._counterbalanced_order(1) == ("override", "bound")
    assert module._counterbalanced_order(2) == ("bound", "override")


def test_parser_accepts_selected_fixture_cases(tmp_path: Path) -> None:
    module = _load_script()

    args = module.build_parser().parse_args(
        [
            "--model-root",
            str(tmp_path / "model"),
            "--fixture",
            str(tmp_path / "fixture.json"),
            "--case-id",
            "code-p512",
            "--case-id",
            "general_en-p512",
            "--override",
            "HIPENGINE_ROUTE=1",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert args.prompt_file is None
    assert args.case_id == ["code-p512", "general_en-p512"]


def test_parser_accepts_decode_mode(tmp_path: Path) -> None:
    module = _load_script()

    args = module.build_parser().parse_args(
        [
            "--model-root",
            str(tmp_path / "model"),
            "--mode",
            "decode",
            "--decode-transitions",
            "64",
            "--prompt-file",
            str(tmp_path / "prompt.txt"),
            "--override",
            "HIPENGINE_ROUTE=1",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert args.mode == "decode"
    assert args.decode_transitions == 64


def test_decode_route_excludes_prefill_and_hashes_complete_sequence() -> None:
    module = _load_script()

    class Runtime:
        def __init__(self) -> None:
            self.synchronizations = 0

        def device_synchronize(self) -> None:
            self.synchronizations += 1

    class Runner:
        def __init__(self) -> None:
            self.runtime = Runtime()
            self.kwargs = []

        def prefill(self, _ids, **kwargs):
            self.kwargs.append(kwargs)
            return SimpleNamespace(token_id=10)

        def step(self, token_id, **kwargs):
            self.kwargs.append(kwargs)
            return SimpleNamespace(token_id=token_id + 1)

    runner = Runner()
    sample = module._run_route(
        runner, [1, 2, 3], mode="decode", decode_transitions=3
    )

    assert sample["output_token_ids"] == [10, 11, 12, 13]
    assert sample["token_id"] == 13
    assert sample["seconds"] >= 0
    assert len(sample["logits_sha256"]) == 64
    assert runner.runtime.synchronizations == 2
    assert runner.kwargs == [
        {"capture_logits": False, "capture_target_hidden": False},
        *[
            {
                "capture_logits": False,
                "capture_target_hidden": False,
                "token_id_resident": True,
            }
        ] * 3,
    ]


def test_parser_accepts_the_scoped_parent_arm_without_an_env_override(
    tmp_path: Path,
) -> None:
    module = _load_script()

    args = module.build_parser().parse_args(
        [
            "--model-root",
            str(tmp_path / "model"),
            "--fixture",
            str(tmp_path / "fixture.json"),
            "--gdn-tile16-parent",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert args.gdn_tile16_parent is True
    assert args.override == []


def test_retired_tile16_override_names_its_replacement() -> None:
    module = _load_script()

    with pytest.raises(SystemExit, match="--gdn-tile16-parent"):
        module._reject_retired_overrides(
            {"HIPENGINE_QWEN4_EXP_GDN_TILE16_PREFILL": "0"}
        )
    # Every live override stays accepted.
    module._reject_retired_overrides({"HIPENGINE_ROUTE": "1"})


def test_an_arm_without_a_control_is_refused() -> None:
    module = _load_script()

    with pytest.raises(SystemExit, match="--gdn-tile16-parent"):
        module._require_arm_control({}, gdn_tile16_parent=False)
    module._require_arm_control({}, gdn_tile16_parent=True)
    module._require_arm_control({"HIPENGINE_ROUTE": "1"}, gdn_tile16_parent=False)
    with pytest.raises(SystemExit, match="prefill mode only"):
        module._require_arm_control({}, gdn_tile16_parent=True, mode="decode")


def test_dispatch_counter_counts_the_launched_owner_and_restores(monkeypatch):
    module = _load_script()
    from hipengine.runtime import qwen4_exp_runner as runner

    def owner(*args, **kwargs):
        return "ran"

    monkeypatch.setattr(runner, "qwen4_exp_gdn_prefill_tiled16_f32", owner)
    monkeypatch.setattr(runner, "qwen4_exp_gdn_prefill_columnwarps_f32", owner)
    monkeypatch.setattr(runner, "qwen4_exp_gdn_prefill_f32", owner)
    original_resolve = runner.resolve

    def resolve(**kwargs):
        if kwargs.get("variant") == "qwen4exp_gdn_rowsplit_prefill":
            return owner
        return original_resolve(**kwargs, missing="none")

    monkeypatch.setattr(runner, "resolve", resolve)
    counts: dict[str, int] = {}
    with module._count_prefill_dispatches(counts):
        runner.qwen4_exp_gdn_prefill_columnwarps_f32(0)
        # The runtime resolver is keyword-only; the arm must count that shape.
        runner.resolve(
            backend="hip_gfx1100",
            layer="gdn_recurrence_norm_gate",
            quant="f32_state",
            variant="qwen4exp_gdn_rowsplit_prefill",
        )(0)
        runner.qwen4_exp_gdn_prefill_tiled16_f32(0)
        # A non-GDN resolution is passed through uncounted.
        runner.resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q4_k",
            variant="selected_dual_silu_logical128_t64_gemv_bf16_bf16_out",
        )
    assert counts == {
        "qwen4exp_gdn_columnwarps_prefill": 1,
        "qwen4exp_gdn_rowsplit_prefill": 1,
        "qwen4exp_gdn_tiled16_prefill": 1,
    }
    assert runner.qwen4_exp_gdn_prefill_columnwarps_f32 is owner
    assert runner.resolve is resolve


def test_dispatch_counter_preserves_the_owner_contract():
    """Wrapping must not change selection: attributes and rows 2/4 survive."""

    module = _load_script()
    from hipengine.kernels.registry import KernelKey, register, unregister
    from hipengine.runtime import qwen4_exp_runner as runner

    launched = []

    def rowsplit(*args, **kwargs):
        launched.append(args)
        return "ran"

    rowsplit.supports_any_rows = True
    rowsplit.publishes_f16_image = True
    rowsplit.supports_state_rows = True
    key = KernelKey(
        "hip_gfx1100",
        "gdn_recurrence_norm_gate",
        "f32_state",
        "qwen4exp_gdn_rowsplit_prefill",
    )
    register(key, rowsplit, replace=True)
    try:
        counts: dict[str, int] = {}
        with module._count_prefill_dispatches(counts):
            owner = runner.resolve(
                backend="hip_gfx1100",
                layer="gdn_recurrence_norm_gate",
                quant="f32_state",
                variant="qwen4exp_gdn_rowsplit_prefill",
            )
            assert getattr(owner, "supports_any_rows", False) is True
            assert getattr(owner, "publishes_f16_image", False) is True
            assert getattr(owner, "supports_state_rows", False) is True
            assert owner(1) == "ran"
            for rows in (2, 4, 15, 16, 512):
                assert (
                    runner.qwen4_exp_gdn_tile16_owner_admitted(
                        rows=rows,
                        num_k_heads=16,
                        num_v_heads=48,
                        head_dim=128,
                        supports_any_rows=getattr(
                            owner, "supports_any_rows", False),
                    )
                    is True
                )
        assert counts == {"qwen4exp_gdn_rowsplit_prefill": 1}
        assert launched
    finally:
        unregister(key)


def test_dispatch_summary_separates_the_two_arms() -> None:
    module = _load_script()
    samples = [
        {"route": "bound", "prefill_dispatches": {"tile16": 2}},
        {"route": "bound", "prefill_dispatches": {"tile16": 1}},
        {"route": "override", "prefill_dispatches": {"columnwarps": 3}},
    ]

    assert module._sum_prefill_dispatches(samples, "bound") == {"tile16": 3}
    assert module._sum_prefill_dispatches(samples, "override") == {"columnwarps": 3}


def test_parent_arm_requires_the_counted_owner_to_move() -> None:
    module = _load_script()

    assert module._parent_arm_counted(
        {
            "bound": {"qwen4exp_gdn_rowsplit_prefill": 36},
            "override": {"qwen4exp_gdn_columnwarps_prefill": 36},
        }
    )
    assert module._parent_arm_counted(
        {
            "bound": {"qwen4exp_gdn_tiled16_multi_prefill": 36},
            "override": {"qwen4exp_gdn_columnwarps_prefill": 36},
        }
    )
    # A self-comparison is not an arm, whichever side failed to move.
    assert not module._parent_arm_counted(
        {
            "bound": {"qwen4exp_gdn_rowsplit_prefill": 36},
            "override": {"qwen4exp_gdn_rowsplit_prefill": 36},
        }
    )
    assert not module._parent_arm_counted(
        {
            "bound": {"qwen4exp_gdn_columnwarps_prefill": 36},
            "override": {"qwen4exp_gdn_columnwarps_prefill": 36},
        }
    )
    assert not module._parent_arm_counted(
        {
            "bound": {"qwen4exp_gdn_rowsplit_prefill": 36},
            "override": {"qwen4exp_gdn_rowsplit_prefill": 1,
                         "qwen4exp_gdn_columnwarps_prefill": 35},
        }
    )


def test_measure_route_pins_the_parent_and_records_the_launch(monkeypatch):
    module = _load_script()
    from hipengine.runtime import qwen4_exp_runner as runner

    seen = {}

    def run_route(owner, token_ids, *, mode, decode_transitions):
        runner.qwen4_exp_gdn_prefill_columnwarps_f32(0)
        seen["enabled"] = owner.gdn_tile16_prefill
        return {"seconds": 1.0}

    monkeypatch.setattr(module, "_run_route", run_route)
    monkeypatch.setattr(runner, "qwen4_exp_gdn_prefill_columnwarps_f32", lambda *a: None)
    owner = SimpleNamespace(gdn_tile16_prefill=True)

    sample = module._measure_route(
        "override",
        owner,
        [1],
        mode="prefill",
        decode_transitions=0,
        gdn_tile16_parent=True,
    )
    assert seen["enabled"] is False
    assert sample["prefill_dispatches"] == {"qwen4exp_gdn_columnwarps_prefill": 1}
    assert owner.gdn_tile16_prefill is True

    module._measure_route(
        "bound",
        owner,
        [1],
        mode="prefill",
        decode_transitions=0,
        gdn_tile16_parent=True,
    )
    assert seen["enabled"] is True


def test_arm_scope_pins_only_the_override_arm() -> None:
    module = _load_script()
    runner = SimpleNamespace(gdn_tile16_prefill=True)

    with module._arm_scope("override", runner, gdn_tile16_parent=True):
        assert runner.gdn_tile16_prefill is False
    assert runner.gdn_tile16_prefill is True

    with module._arm_scope("bound", runner, gdn_tile16_parent=True):
        assert runner.gdn_tile16_prefill is True
    with module._arm_scope("override", runner, gdn_tile16_parent=False):
        assert runner.gdn_tile16_prefill is True


def test_paired_summary_reports_ratio_cv_and_output_identity() -> None:
    module = _load_script()
    samples = [
        {
            "pair": 0,
            "route": "bound",
            "seconds": 10.0,
            "token_id": 7,
            "logits_sha256": "a",
        },
        {
            "pair": 0,
            "route": "override",
            "seconds": 8.0,
            "token_id": 7,
            "logits_sha256": "b",
        },
        {
            "pair": 1,
            "route": "override",
            "seconds": 9.0,
            "token_id": 7,
            "logits_sha256": "b",
        },
        {
            "pair": 1,
            "route": "bound",
            "seconds": 11.0,
            "token_id": 7,
            "logits_sha256": "a",
        },
    ]

    summary = module._paired_summary(samples, prompt_tokens=100)

    assert summary["routes"]["bound"]["tok_s"] == 200 / 21
    assert summary["routes"]["override"]["tok_s"] == 200 / 17
    assert summary["routes"]["bound"]["repeat_exact"] is True
    assert summary["routes"]["override"]["repeat_exact"] is True
    assert summary["cross_route_logits_exact"] is False
    assert [row["throughput_ratio_override_vs_bound"] for row in summary["pairs"]] == [
        1.25,
        11 / 9,
    ]
    assert summary["ratio"]["min"] > 1.0
    assert summary["ratio"]["median"] > 1.0
    assert summary["ratio"]["mean_95ci"][0] < summary["ratio"]["mean"]
    assert summary["ratio"]["mean_95ci"][1] > summary["ratio"]["mean"]
    assert summary["routes"]["bound"]["cv_percent"] > 0
