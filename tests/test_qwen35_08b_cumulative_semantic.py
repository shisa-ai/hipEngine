from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import qwen35_08b_cumulative_semantic as cumulative


def test_expand_to_512_repeats_complete_prompt_tokens() -> None:
    expanded = cumulative.expand_to_512([11, 22, 33])

    assert len(expanded) == 512
    assert expanded[:9] == [11, 22, 33] * 3
    assert expanded[-2:] == [11, 22]
    with pytest.raises(ValueError, match="empty prompt"):
        cumulative.expand_to_512([])


def test_role_environment_restores_every_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK", "before")
    monkeypatch.setenv(
        "HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL",
        "before-dual",
    )
    monkeypatch.delenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", raising=False)

    with cumulative.role_environment("q4", "strict_x2"):
        assert os.environ["HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK"] == "0"
        assert os.environ[
            "HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL"
        ] == "0"
        assert os.environ["HIPENGINE_GGUF_DENSE_WMMA_BULK"] == "0"
        assert os.environ["HIPENGINE_GGUF_GDN_PREFILL_MODE"] == "exact"
        assert os.environ["HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING"] == "0"

    assert os.environ["HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK"] == "before"
    assert os.environ[
        "HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL"
    ] == "before-dual"
    assert "HIPENGINE_GGUF_GDN_PREFILL_MODE" not in os.environ


def test_trajectory_digest_covers_tokens_and_logits() -> None:
    baseline = [
        {"token_id": 7, "logits": np.asarray([1.0, 2.0], dtype=np.float32)},
        {"token_id": 8, "logits": np.asarray([3.0, 4.0], dtype=np.float32)},
    ]
    changed = [dict(row) for row in baseline]
    changed[1] = {"token_id": 9, "logits": changed[1]["logits"]}

    assert cumulative.trajectory_digest(baseline) == cumulative.trajectory_digest(baseline)
    assert cumulative.trajectory_digest(baseline) != cumulative.trajectory_digest(changed)


def test_cumulative_semantic_artifact_closes_the_packet() -> None:
    artifact = (
        Path(__file__).parents[1]
        / "benchmarks/results/2026-08-15-gfx1151-qwen35-08b-cumulative-semantic.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["status"] == "retained_correctness"
    assert payload["gate_passed"] is True
    assert payload["source"]["tracked_source_clean"] is True
    assert payload["decision"]["current_top1_matches"] == 1794
    assert payload["decision"]["current_transitions"] == 1800
    assert payload["decision"]["max_current_kl"] < 0.01
    assert payload["decision"]["all_role_deterministic"] is True
    assert payload["decision"]["all_state_finite"] is True
    assert payload["decision"]["recorded_graph_prompt_role_pairs"] == 72
    assert payload["decision"]["all_recorded_graph_trajectories_exact"] is True
    assert payload["decision"]["min_recorded_graph_top1"] == 1.0

    for quant in payload["results"].values():
        for profile in quant["prompts"].values():
            assert len(profile) == 18
            for row in profile:
                for role in row["roles"].values():
                    assert role["correctness"]["passed"] is True
                    assert role["teacher_forced_deterministic"] is True
                    assert role["state_deterministic"] is True
                    assert role["state_finite"] is True
