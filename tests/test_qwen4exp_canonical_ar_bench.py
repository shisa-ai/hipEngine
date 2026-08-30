from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_canonical_ar_bench.py"
CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen4exp_canonical_ar_bench", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ByteTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


def _source_rows() -> list[dict[str, object]]:
    return [
        {
            "id": f"{category}_fixture",
            "category": category,
            "messages": [{"role": "user", "content": f"material for {category}"}],
        }
        for category in CATEGORIES
    ]


def test_build_fixture_covers_every_category_and_exact_shape() -> None:
    module = _load_script()
    fixture = module.build_fixture(
        tokenizer=_ByteTokenizer(),
        source_rows=_source_rows(),
        shapes=(64, 96),
        source_path=Path("suite.jsonl"),
        source_sha256="a" * 64,
        model_identity={"architecture": "qwen4exp"},
    )

    assert fixture["schema"] == 1
    assert fixture["shapes"] == [64, 96]
    assert len(fixture["cases"]) == 8
    assert {
        (row["category"], row["prompt_tokens"])
        for row in fixture["cases"]
    } == {(category, shape) for category in CATEGORIES for shape in (64, 96)}
    for row in fixture["cases"]:
        assert len(row["prompt_token_ids"]) == row["prompt_tokens"]
        assert row["prompt_token_ids_sha256"] == module.token_ids_sha256(
            row["prompt_token_ids"]
        )


def test_build_fixture_rejects_a_missing_category() -> None:
    module = _load_script()
    with pytest.raises(ValueError, match="missing canonical categories"):
        module.build_fixture(
            tokenizer=_ByteTokenizer(),
            source_rows=_source_rows()[:-1],
            shapes=(64,),
            source_path=Path("suite.jsonl"),
            source_sha256="b" * 64,
            model_identity={},
        )


def test_committed_fixture_is_pinned_and_valid() -> None:
    module = _load_script()
    fixture_path = (
        ROOT
        / "benchmarks"
        / "fixtures"
        / "qwen4exp_canonical_ar_p512_p1024_p4096.json"
    )
    fixture, digest = module.load_fixture(fixture_path)

    assert digest == "42b562bd8e9644bea5b8891c61633dce7f6e75daca64cf79e9cb45c432099da1"
    assert fixture["source"]["sha256"] == (
        "fac920be5e691fec2cb70fd8b7eedddab8926b89d6a1627f62ec4f441d86084a"
    )
    assert fixture["decode_transitions"] == 128
    assert fixture["shapes"] == [512, 1024, 4096]


def test_llamacpp_response_uses_128_transition_boundary() -> None:
    module = _load_script()
    case = {
        "id": "code-p512",
        "category": "code",
        "prompt_tokens": 512,
        "prompt_token_ids_sha256": "c" * 64,
    }
    response = {
        "tokens": list(range(129)),
        "tokens_evaluated": 512,
        "timings": {
            "prompt_n": 512,
            "prompt_ms": 1000.0,
            "predicted_n": 129,
            "predicted_ms": 2000.0,
        },
        "stop_type": "limit",
        "truncated": False,
    }

    row = module.llamacpp_response_sample(
        case=case,
        response=response,
        client_wall_s=3.1,
        repetition=2,
        expected_transitions=128,
    )

    assert row["prefill_tok_s"] == 512.0
    assert row["decode_transitions"] == 128
    assert row["decode_tok_s"] == 64.0
    assert row["output_token_count"] == 129
    assert row["output_token_ids_sha256"] == module.token_ids_sha256(range(129))


def test_llamacpp_response_rejects_missing_output_token() -> None:
    module = _load_script()
    with pytest.raises(ValueError, match="129 output token IDs"):
        module.llamacpp_response_sample(
            case={
                "id": "code-p512",
                "category": "code",
                "prompt_tokens": 512,
                "prompt_token_ids_sha256": "d" * 64,
            },
            response={
                "tokens": list(range(128)),
                "tokens_evaluated": 512,
                "timings": {
                    "prompt_n": 512,
                    "prompt_ms": 1000.0,
                    "predicted_n": 129,
                    "predicted_ms": 2000.0,
                },
            },
            client_wall_s=3.0,
            repetition=0,
            expected_transitions=128,
        )


def test_summarize_samples_reports_shape_rates_and_determinism() -> None:
    module = _load_script()
    samples = []
    for category_index, category in enumerate(CATEGORIES):
        digest = f"{category_index:064x}"
        for repetition in range(3):
            samples.append(
                {
                    "case_id": f"{category}-p512",
                    "category": category,
                    "prompt_tokens": 512,
                    "repetition": repetition,
                    "prefill_ms": 1000.0 + category_index,
                    "prefill_tok_s": 512000.0 / (1000.0 + category_index),
                    "decode_ms": 2000.0 + category_index,
                    "decode_transitions": 128,
                    "decode_tok_s": 128000.0 / (2000.0 + category_index),
                    "client_wall_s": 3.0,
                    "output_token_ids_sha256": digest,
                    "output_token_count": 129,
                }
            )

    summary = module.summarize_samples(samples)

    assert summary["all_cases_deterministic"] is True
    assert summary["shapes"]["512"]["sample_count"] == 12
    assert summary["shapes"]["512"]["case_count"] == 4
    assert summary["shapes"]["512"]["prefill_tok_s_weighted"] > 500
    assert summary["shapes"]["512"]["decode_tok_s_weighted"] > 60


def test_compare_rejects_different_case_sets(tmp_path: Path) -> None:
    module = _load_script()
    paths = []
    for engine, case_id in (("one", "code-p512"), ("two", "general_en-p512")):
        path = tmp_path / f"{engine}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "engine": engine,
                    "fixture_sha256": "f" * 64,
                    "samples": [{"case_id": case_id}],
                    "summary": {},
                }
            )
        )
        paths.append(path)

    with pytest.raises(ValueError, match="identical non-empty case sets"):
        module.compare_engine_artifacts(paths)
