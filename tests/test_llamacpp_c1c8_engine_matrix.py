from __future__ import annotations

import json
from pathlib import Path

from scripts import llamacpp_c1c8_engine_matrix as matrix


def test_load_prompts_matches_published_single_turn_template(tmp_path: Path) -> None:
    suite = tmp_path / "prompts.jsonl"
    suite.write_text(
        json.dumps(
            {
                "id": "p0",
                "category": "code",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert matrix.load_prompts(suite) == [
        {
            "id": "p0",
            "category": "code",
            "rendered": (
                "<|im_start|>user\nhello<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
        }
    ]


def test_workload_arms_can_run_prefill_without_decode() -> None:
    assert matrix._workload_arms(1, 0, 0) == (("prefill", 1),)
    assert matrix._workload_arms(1, 24, 0) == (("prefill", 1), ("ar", 24))
    assert matrix._workload_arms(1, 24, 3) == (("mtp", 24),)
