from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cross_engine_quality_gate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("cross_engine_quality_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CharTokenizer:
    def decode(self, token_ids: Any) -> str:
        return "".join(chr(int(token_id)) for token_id in token_ids)


def _ids(values: list[int]) -> dict[str, Any]:
    return {"ids": values, "ids_sha256": "-".join(str(v) for v in values)}


def _teacher_artifact(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "engine": "hipengine-strict",
        "profile": "strict",
        "battery": {"suites": [{"path": "a.jsonl", "sha256": "a"}], "text_recipe": "recipe"},
        "prompts": prompts,
    }


def _engine_artifact(engine: str, prompts: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "engine": engine,
        "surface": "test",
        "battery": {"suites": [{"path": "a.jsonl", "sha256": "a"}], "text_recipe": "recipe"},
        "prompts": prompts,
    }
    payload.update(extra)
    return payload


def test_battery_loads_18_prompts_across_four_categories() -> None:
    module = _load_script()
    rows = module.load_battery()
    assert len(rows) == 18
    categories = {str(row["category"]) for row in rows}
    assert categories == {"code", "general_en", "general_ja", "mixed_ja_en"}
    identity = module.battery_identity()
    assert identity["text_recipe"] == module.TEXT_RECIPE
    assert len(identity["suites"]) == 2


def test_battery_text_recipe_renders_id_role_and_content() -> None:
    module = _load_script()
    row = {
        "id": "p1",
        "category": "code",
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert module.battery_text(row) == "[p1]\nuser: hello\n\n"


def test_first_divergence_helpers() -> None:
    module = _load_script()
    assert module._first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert module._first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert module._first_divergence([1, 2], [1, 2, 3]) == 2
    assert module._first_text_divergence("abc", "abc") is None
    assert module._first_text_divergence("abc", "abd") == 2


def test_truncated_kl_is_zero_for_identical_topk() -> None:
    module = _load_script()
    left = [[1, -0.1], [2, -2.0], [3, -5.0]]
    assert module._truncated_kl(left, [list(entry) for entry in left]) == pytest.approx(0.0, abs=1e-12)
    right = [[2, -0.1], [1, -2.0], [3, -5.0]]
    assert module._truncated_kl(left, right) > 0.0


def test_aligned_topk_metrics_stop_at_first_id_divergence() -> None:
    module = _load_script()
    teacher_top = [[[1, -0.1], [2, -1.0]], [[2, -0.2], [1, -1.0]]]
    engine_top = [[[1, -0.3], [2, -1.0]], [[2, -0.2], [1, -1.0]]]
    metrics = module._aligned_topk_metrics(
        teacher_top, engine_top, teacher_ids=[1, 2, 3], engine_ids=[1, 2, 9]
    )
    assert metrics is not None
    assert metrics["aligned_positions"] == 2
    assert metrics["top1_agreement"] == 1.0
    assert metrics["mean_kl"] > 0.0


def test_engine_gate_verdicts_ids_basis() -> None:
    module = _load_script()
    teacher = _teacher_artifact(
        [
            {
                "id": "a",
                "runs": [dict(_ids([10, 11, 12]), top20=[[[10, -0.5]], [[11, -0.1]]])],
            },
            {"id": "b", "runs": [dict(_ids([20, 21, 22]))]},
        ]
    )
    engine = _engine_artifact(
        "halobox",
        [
            {
                "id": "a",
                "runs": [dict(_ids([10, 11, 12]), top20=[[[10, -0.7]], [[11, -0.1]]]), _ids([10, 11, 12])],
            },
            {
                "id": "b",
                "runs": [_ids([20, 99, 22]), _ids([20, 99, 22])],
            },
        ],
        protocol={"ids_basis": "server_token_ids"},
    )
    verdict = module._engine_gate_verdicts(teacher, engine, tokenizer=None)
    assert verdict["g1_greedy_identity"]["prompt_count"] == 2
    assert verdict["g1_greedy_identity"]["exact_count"] == 1
    assert verdict["g1_greedy_identity"]["first_divergences"] == {"b": 1}
    assert verdict["g2_distribution"]["available"] is True
    assert verdict["g2_distribution"]["prompt_count"] == 1
    assert verdict["g2_distribution"]["top1_agreement_mean"] == 1.0
    assert verdict["g3_determinism"]["all_repeats_exact"] is True
    assert verdict["g4_task_battery"]["strict_exact_count"] == 1


def test_engine_gate_verdicts_text_basis_uses_decoded_text() -> None:
    module = _load_script()
    # Teacher ids decode to "XY"; the engine emitted identical text but its
    # re-encoded ids would disagree at a merge boundary - the text basis must
    # be primary and report exact agreement.
    teacher = _teacher_artifact(
        [{"id": "a", "runs": [_ids([88, 89])]}]
    )
    engine = _engine_artifact(
        "gufo",
        [
            {
                "id": "a",
                "runs": [
                    {"ids": [], "ids_sha256": "", "text": "XY", "text_sha256": "t"},
                    {"ids": [], "ids_sha256": "", "text": "XY", "text_sha256": "t"},
                ],
            }
        ],
        protocol={"ids_basis": "text_retokenized"},
    )
    verdict = module._engine_gate_verdicts(teacher, engine, tokenizer=_CharTokenizer())
    assert verdict["g1_greedy_identity"]["exact_count"] == 1
    assert verdict["g1_greedy_identity"]["per_prompt"]["a"]["comparison_basis"] == "decoded_text"
    assert verdict["g4_task_battery"]["strict_exact_count"] == 1
    assert verdict["g2_distribution"]["available"] is False

    divergent = _engine_artifact(
        "gufo",
        [
            {
                "id": "a",
                "runs": [
                    {"ids": [], "ids_sha256": "", "text": "XZ", "text_sha256": "t2"},
                    {"ids": [], "ids_sha256": "", "text": "XZ", "text_sha256": "t2"},
                ],
            }
        ],
        protocol={"ids_basis": "text_retokenized"},
    )
    verdict = module._engine_gate_verdicts(teacher, divergent, tokenizer=_CharTokenizer())
    assert verdict["g1_greedy_identity"]["exact_count"] == 0
    assert verdict["g1_greedy_identity"]["first_divergences"] == {"a": 1}


def test_engine_gate_verdicts_missing_prompt_requires_declared_reason() -> None:
    module = _load_script()
    teacher = _teacher_artifact([{"id": "a", "runs": [_ids([1])]}, {"id": "b", "runs": [_ids([2])]}])
    engine = _engine_artifact("halobox", [{"id": "a", "runs": [_ids([1])]}])
    with pytest.raises(ValueError, match="without a declared reason"):
        module._engine_gate_verdicts(teacher, engine, tokenizer=None)
    declared = _engine_artifact(
        "halobox",
        [{"id": "a", "runs": [_ids([1])]}],
        disqualified_prompts=[{"id": "b", "reason": "round-trip mismatch"}],
    )
    verdict = module._engine_gate_verdicts(teacher, declared, tokenizer=None)
    assert verdict["g1_greedy_identity"]["prompt_count"] == 1
    assert verdict["g4_task_battery"]["verdict"] == "partial_coverage"


def test_llamacpp_probs_parses_logprob_and_prob_fields() -> None:
    module = _load_script()
    response = {
        "probs": [
            {
                "id": 5,
                "prob": 0.25,
                "top_probs": [{"id": 5, "logprob": -1.0}, {"id": 6, "prob": 0.1}],
            }
        ]
    }
    parsed = module._llamacpp_probs(response)
    assert parsed == [[[5, -1.0], [6, pytest.approx(float(np.log(0.1)), abs=1e-9)]]]
    assert module._llamacpp_probs({"tokens": []}) is None
