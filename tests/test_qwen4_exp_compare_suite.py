from __future__ import annotations

import json

import pytest

from scripts.qwen4_exp_compare_suite import aggregate, load_prompts


def test_qwen4_exp_suite_loader_and_tail_aggregate(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps({"id": "a", "category": "code", "text": "alpha"}),
                json.dumps(
                    {
                        "id": "b",
                        "category": "ja",
                        "messages": [{"role": "user", "content": "beta"}],
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_prompts(path) == [
        {"id": "a", "category": "code", "prompt": "alpha"},
        {"id": "b", "category": "ja", "prompt": "beta"},
    ]
    report = aggregate(
        [
            {"kl_teacher_to_hipengine": 0.01, "top1_agreement": True},
            {"kl_teacher_to_hipengine": 0.03, "top1_agreement": False},
        ]
    )
    assert report["count"] == 2
    assert report["mean_kl"] == pytest.approx(0.02)
    assert report["p95_kl"] == pytest.approx(0.029)
    assert report["p99_kl"] == pytest.approx(0.0298)
    assert report["max_kl"] == pytest.approx(0.03)
    assert report["top1_agreement_rate"] == pytest.approx(0.5)
