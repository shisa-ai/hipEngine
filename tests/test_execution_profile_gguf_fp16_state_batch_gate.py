from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import execution_profile_gguf_fp16_state_batch_gate as gate


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counts = {
        "code": 5,
        "general_en": 5,
        "general_ja": 4,
        "mixed_ja_en": 4,
    }
    for category, count in counts.items():
        for index in range(count):
            rows.append(
                {
                    "id": f"{category}-{index}",
                    "category": category,
                    "prompt": f"{category} prompt {index}",
                    "suite": "/tmp/suite.jsonl",
                }
            )
    return rows


def _trajectory(token: int, delta: float = 0.0):
    return (
        {
            "token_id": token,
            "logits": np.asarray([0.0, 1.0 + delta, -1.0], dtype=np.float32),
        },
    )


def test_fp16_state_parser_records_quant_label() -> None:
    parser = gate.build_parser()
    default = parser.parse_args(("--json", "/tmp/out.json"))
    q4km = parser.parse_args(
        ("--quant-label", "gguf_q4_k_m", "--json", "/tmp/out.json")
    )

    assert default.quant_label == "gguf_q4_k_s"
    assert q4km.quant_label == "gguf_q4_k_m"


def test_fp16_state_environment_explicitly_disables_promoted_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(gate.FP16_STATE_ENV, raising=False)

    with gate.fp16_state_environment(False):
        assert os.environ[gate.FP16_STATE_ENV] == "0"

    assert gate.FP16_STATE_ENV not in os.environ
    monkeypatch.setenv(gate.FP16_STATE_ENV, "1")
    with gate.fp16_state_environment(False):
        assert os.environ[gate.FP16_STATE_ENV] == "0"
    assert os.environ[gate.FP16_STATE_ENV] == "1"


def test_cycle_tokens_reaches_exact_target_without_input_mutation() -> None:
    source = (3, 5, 7)

    assert gate._cycle_tokens(source, 8) == (3, 5, 7, 3, 5, 7, 3, 5)
    assert source == (3, 5, 7)


def test_static_matrix_covers_full_c4_c8_and_long_category_group() -> None:
    rows = _rows()
    prompt_tokens = {
        str(row["id"]): (1, index + 2, 3)
        for index, row in enumerate(rows)
    }

    scenarios = gate.build_static_scenarios(
        rows,
        prompt_tokens,
        widths=(4, 8),
        decode_steps=24,
        long_prompt_tokens=512,
        long_decode_steps=8,
    )

    assert len([row for row in scenarios if row.width == 4]) == 5
    assert len([row for row in scenarios if row.width == 8 and not row.long_context]) == 3
    long = [row for row in scenarios if row.long_context]
    assert len(long) == 1
    assert all(len(tokens) == 512 for tokens in long[0].token_rows)
    assert {
        str(row["category"]) for row in long[0].rows
    } == {"code", "general_en", "general_ja", "mixed_ja_en"}
    assert scenarios[4].actual_count == 2
    assert scenarios[7].actual_count == 2


def test_trajectory_exactness_checks_logits_and_selected_ids() -> None:
    reference = _trajectory(1)

    assert gate._trajectories_exact(reference, reference)
    assert not gate._trajectories_exact(reference, _trajectory(2))
    assert not gate._trajectories_exact(reference, _trajectory(1, 1.0e-4))


def test_manifest_gate_requires_indexed_singleton_for_every_cn_route() -> None:
    good = {
        "decode": {
            "physical_rows": 8,
            "linear_attention_decode_path": "indexed_batch",
            "gdn_recurrent_decode_path": "indexed_singleton",
        }
    }
    c1_tail = {"decode": {"physical_rows": 1}}
    bad = {
        "decode": {
            "physical_rows": 4,
            "linear_attention_decode_path": "indexed_batch",
            "gdn_recurrent_decode_path": "segments",
        }
    }

    assert gate._decode_manifests_use_indexed((good, c1_tail))
    assert not gate._decode_manifests_use_indexed((good, bad))
    assert not gate._decode_manifests_use_indexed((c1_tail,))


def test_step_normalizes_single_row_logits_and_rejects_nonfinite() -> None:
    result = SimpleNamespace(
        token_id=7,
        logits=np.asarray([[0.0, 2.0, -1.0]], dtype=np.float32),
    )

    row = gate._step(result)

    assert row["token_id"] == 7
    assert np.asarray(row["logits"]).shape == (3,)

    with pytest.raises(gate.GateError, match="invalid"):
        gate._step(SimpleNamespace(token_id=0, logits=np.asarray([np.nan])))
