from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.qwen35_kv_int8_accuracy import _compare_path, main, run


def _args(**overrides):
    defaults = dict(
        device="cpu",
        contexts="4,9",
        block_size=4,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=8,
        scale_dtype="fp16",
        pseudo_vocab_size=16,
        seed=1234,
        max_abs_threshold=5.0e-3,
        kl_threshold=0.05,
        top1_threshold=0.90,
        compiler_version_file=None,
        require_cached_build=False,
        require_int8_hip=False,
        allow_missing_int8_hip=False,
        json=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_qwen35_kv_int8_accuracy_cpu_runs_short_and_page_boundary_cases() -> None:
    payload = run(_args())

    assert payload["status"] == "accepted"
    assert payload["passed"] is True
    assert payload["blocked_reasons"] == []
    assert [case["context_len"] for case in payload["cases"]] == [4, 9]
    assert payload["cases"][0]["crosses_page_boundary"] is False
    assert payload["cases"][1]["crosses_page_boundary"] is True
    for case in payload["cases"]:
        assert set(case["paths"]) == {"bf16", "int8_per_token_head"}
        assert case["paths"]["bf16"]["passed"] is True
        assert case["paths"]["int8_per_token_head"]["passed"] is True
        assert case["paths"]["bf16"]["pseudo_logit_gate"]["top1_agreement"] == 1.0
        assert case["paths"]["int8_per_token_head"]["pseudo_logit_gate"]["top1_agreement"] == 1.0
        assert "bf16_vs_int8_quantization" in case


def test_qwen35_kv_int8_accuracy_json_self_describes_artifact_path(tmp_path: Path) -> None:
    artifact_path = tmp_path / "int8-primitive.json"

    rc = main(
        [
            "--device",
            "cpu",
            "--contexts",
            "4,9",
            "--block-size",
            "4",
            "--num-q-heads",
            "4",
            "--num-kv-heads",
            "2",
            "--head-dim",
            "8",
            "--json",
            str(artifact_path),
        ]
    )

    assert rc == 0
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["artifact_path"] == str(artifact_path)
    assert payload["source_artifact_path"] == str(artifact_path)


def test_qwen35_kv_int8_accuracy_reports_numerical_mismatch_clearly() -> None:
    expected = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidate = expected + np.asarray([[0.1, 0.0], [0.0, 0.0]], dtype=np.float32)
    projection = np.eye(4, dtype=np.float32)

    check = _compare_path(
        "int8_per_token_head",
        "hip_gfx1100",
        expected,
        candidate,
        projection,
        max_abs_threshold=1.0e-3,
        kl_threshold=0.05,
        top1_threshold=0.90,
    )

    assert check.passed is False
    assert np.isclose(check.max_abs_attn, 0.1)
    assert check.mismatch_reason is not None
    assert "max_abs" in check.mismatch_reason


def test_qwen35_kv_int8_accuracy_hip_requires_qwen_block_size() -> None:
    args = _args(device="hip", block_size=4, allow_missing_int8_hip=True)

    try:
        run(args)
    except ValueError as exc:
        assert "block-size 256" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected device=hip shape validation to fail")
