from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_context_decode_profile.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "qwen4exp_context_decode_profile", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_case_and_live_count_prefix_contract() -> None:
    module = _load_script()
    fixture = {
        "cases": [
            {
                "id": "code-p4096",
                "category": "code",
                "prompt_tokens": 6,
                "prompt_token_ids": [10, 11, 12, 13, 14, 15],
            }
        ]
    }

    case = module._select_case(fixture, "code-p4096")

    assert module._prefix_for_live_count(case, 2) == (10,)
    assert module._prefix_for_live_count(case, 6) == (10, 11, 12, 13, 14)
    assert module._prefix_for_live_count(case, 7) == (10, 11, 12, 13, 14, 15)
    with pytest.raises(ValueError, match="at least 2"):
        module._prefix_for_live_count(case, 1)
    with pytest.raises(ValueError, match="exceeds"):
        module._prefix_for_live_count(case, 8)


def test_repeat_summary_requires_token_and_state_exactness() -> None:
    module = _load_script()
    rows = [
        {"token_id": 7, "state_sha256": "s", "wall_seconds": value}
        for value in (0.3, 0.2, 0.1)
    ]

    exact = module._repeat_summary(rows)
    token_mismatch = module._repeat_summary(
        [*rows[:2], {"token_id": 8, "state_sha256": "s", "wall_seconds": 0.1}]
    )
    state_mismatch = module._repeat_summary(
        [*rows[:2], {"token_id": 7, "state_sha256": "t", "wall_seconds": 0.1}]
    )

    assert exact == {
        "runs": 3,
        "token_exact": True,
        "state_exact": True,
        "passed": True,
        "wall_seconds": {
            "mean": pytest.approx(0.2),
            "median": pytest.approx(0.2),
            "min": pytest.approx(0.1),
            "max": pytest.approx(0.3),
        },
    }
    assert token_mismatch["passed"] is False
    assert token_mismatch["token_exact"] is False
    assert state_mismatch["passed"] is False
    assert state_mismatch["state_exact"] is False


def test_memory_growth_distinguishes_steady_window_from_bucket_transition() -> None:
    module = _load_script()
    before = {"active_allocations": 10, "current_allocated_bytes": 100}

    steady = module._memory_growth(
        before,
        {"active_allocations": 10, "current_allocated_bytes": 100},
    )
    resized = module._memory_growth(
        before,
        {"active_allocations": 10, "current_allocated_bytes": 120},
    )

    assert steady == {
        "allocation_growth": 0,
        "allocated_byte_growth": 0,
        "passed": True,
    }
    assert resized == {
        "allocation_growth": 0,
        "allocated_byte_growth": 20,
        "passed": False,
    }


def test_parser_defaults_to_binding_transition_live_counts(tmp_path: Path) -> None:
    module = _load_script()

    args = module.build_parser().parse_args(
        [
            "--model-root", str(tmp_path / "model"),
            "--output", str(tmp_path / "result.json"),
        ]
    )

    assert args.case_id == "code-p4096"
    assert args.live_count == [2051, 2052, 4097]
    assert args.repetitions == 3
    assert args.prefill_chunk_size == 512
