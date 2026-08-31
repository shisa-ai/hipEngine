from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


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


def test_decode_route_excludes_prefill_and_hashes_complete_sequence(
    monkeypatch,
) -> None:
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

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_DEVICE_ARGMAX", "1")
    runner = Runner()
    sample = module._run_route(
        runner, [1, 2, 3], mode="decode", decode_transitions=3
    )

    assert sample["output_token_ids"] == [10, 11, 12, 13]
    assert sample["token_id"] == 13
    assert sample["seconds"] >= 0
    assert len(sample["logits_sha256"]) == 64
    assert runner.runtime.synchronizations == 2
    assert runner.kwargs == [{"capture_logits": False}] * 4


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
