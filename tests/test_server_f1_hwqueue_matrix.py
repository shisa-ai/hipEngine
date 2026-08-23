"""Host-only tests for the gfx1151 F1 hardware-queue matrix driver."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "server_f1_hwqueue_matrix.py"
    module_name = "_server_f1_hwqueue_matrix_test_module"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module()


def test_queue_policy_parser_is_bounded_and_includes_true_unset() -> None:
    assert SCRIPT.parse_queue_policies("1,2,4,8,unset") == ("1", "2", "4", "8", "unset")
    with pytest.raises(ValueError, match="duplicate"):
        SCRIPT.parse_queue_policies("1,2,2")
    with pytest.raises(ValueError, match="1,2,4,8,unset"):
        SCRIPT.parse_queue_policies("1,16")


def test_counterbalanced_queue_blocks_reverse_and_rotate() -> None:
    policies = ("1", "2", "4", "8", "unset")

    blocks = SCRIPT.counterbalanced_queue_blocks(policies, blocks=3)

    assert blocks[0] == policies
    assert blocks[1] == tuple(reversed(policies))
    assert blocks[2] == ("2", "4", "8", "unset", "1")


def test_queue_environment_distinguishes_explicit_and_runtime_default() -> None:
    base = {
        "GPU_MAX_HW_QUEUES": "99",
        "HIPENGINE_GPU_MAX_HW_QUEUES_POLICY": "stale",
    }

    explicit = SCRIPT.queue_environment(base, "4")
    runtime_default = SCRIPT.queue_environment(base, "unset")

    assert explicit["GPU_MAX_HW_QUEUES"] == "4"
    assert explicit["HIPENGINE_GPU_MAX_HW_QUEUES_POLICY"] == "explicit"
    assert "GPU_MAX_HW_QUEUES" not in runtime_default
    assert runtime_default["HIPENGINE_GPU_MAX_HW_QUEUES_POLICY"] == "runtime_default"
    assert base["GPU_MAX_HW_QUEUES"] == "99"


def test_run_leaf_is_unique_by_commit_policy_suite_block_and_tag(tmp_path: Path) -> None:
    leaf = SCRIPT.run_leaf(
        tmp_path,
        commit="a" * 40,
        profile="fp16-production",
        queue_policy="unset",
        suite="core",
        run_tag="cycle-a",
        block_index=1,
    )

    assert leaf == (
        tmp_path
        / "runs"
        / ("a" * 40)
        / "fp16-production"
        / "hwq-unset"
        / "core"
        / "cycle-a-block01"
    )


def test_resume_requires_matching_spec_and_complete_schema2_result(tmp_path: Path) -> None:
    leaf = tmp_path / "leaf"
    leaf.mkdir()
    spec = {"schema_version": 1, "spec_sha256": "abc"}
    (leaf / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    result = {
        "schema": 2,
        "status": "failed_gate",
        "completed_at": "2026-08-23T00:00:00Z",
        "rows": {"17": {}, "32": {}},
    }
    (leaf / "result.json").write_text(json.dumps(result), encoding="utf-8")

    assert SCRIPT.resume_result(leaf, spec_sha256="abc", expected_widths=(17, 32)) == result
    with pytest.raises(ValueError, match="spec hash"):
        SCRIPT.resume_result(leaf, spec_sha256="different", expected_widths=(17, 32))


def test_child_result_classification_separates_mechanical_and_product_gates() -> None:
    row = {
        "correctness": {
            "warmups": {"passed": True},
            "measured": {"passed": True},
            "repeat_determinism": {"passed": True},
            "live_admission": None,
        },
        "execution": {"route_ok": True},
        "streaming": {"passed": True, "route": {"passed": True}},
        "memory": {"final_delta_bytes": 0},
    }
    payload = {
        "schema": 2,
        "status": "failed_gate",
        "passed": False,
        "rows": {"17": row, "32": row},
    }

    classification = SCRIPT.classify_child_result(payload, expected_widths=(17, 32))

    assert classification["mechanical_passed"] is True
    assert classification["product_gate_passed"] is False
    assert classification["safe_to_continue"] is True
