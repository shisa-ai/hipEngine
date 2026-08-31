from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_qsa_h256_gate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen4exp_qsa_h256_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_summary_binds_cross_route_and_candidate_repeats() -> None:
    module = _load_script()
    strict = {"a": [1, 2, 3], "b": [4, 5]}
    candidate = {
        "a": [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        "b": [[4, 5], [4, 5], [4, 5]],
    }

    passed = module._task_summary(strict, candidate)
    candidate["b"][2] = [4, 6]
    failed = module._task_summary(strict, candidate)

    assert passed == {
        "prompts": 2,
        "strict_exact": 2,
        "candidate_repeat_exact": True,
        "passed": True,
        "divergences": [],
        "repeat_mismatches": [],
    }
    assert failed["passed"] is False
    assert failed["candidate_repeat_exact"] is False
    assert failed["repeat_mismatches"] == ["b"]


def test_parser_defaults_to_all_canonical_p4096_categories(tmp_path: Path) -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        ["--model-root", str(tmp_path / "model"), "--output", str(tmp_path / "o.json")]
    )

    assert args.case_id == [
        "code-p4096", "general_en-p4096", "general_ja-p4096", "mixed_ja_en-p4096"
    ]
    assert args.decode_steps == 24
    assert args.repeat_runs == 3
    assert args.free_tokens == 32
