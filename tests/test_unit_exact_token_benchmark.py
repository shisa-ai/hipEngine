from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from hipengine.generation import GenerationOutput, GenerationTelemetry
from hipengine.benchmark.exact_tokens import (
    ExactTokenOracle,
    load_exact_token_fixture,
    validate_exact_token_parity,
)
from hipengine.benchmark.prompts import token_ids_sha256


REPO_FIXTURE = Path("fixtures/qwen35_paro/parent_512_32_seed1234.json")
TOOL_PATH = Path("scripts/exact_token_generation.py")


def _load_tool():
    spec = importlib.util.spec_from_file_location("hipengine_exact_token_generation", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_exact_token_fixture_cycles_one_512_row() -> None:
    fixture = load_exact_token_fixture(REPO_FIXTURE, prompt_length=512, prompt_count=8)

    assert fixture.prompt_length == 512
    assert fixture.prompt_count == 8
    assert all(row == fixture.prompt_rows[0] for row in fixture.prompt_rows)
    assert fixture.row_sha256 == tuple(
        token_ids_sha256(row) for row in fixture.prompt_rows
    )


def test_exact_token_oracle_roundtrips_and_requires_generated_id_equality(tmp_path: Path) -> None:
    fixture = load_exact_token_fixture(REPO_FIXTURE, prompt_length=512, prompt_count=2)
    direct = ExactTokenOracle.from_rows(
        mode="direct",
        prompt_rows=fixture.prompt_rows,
        generated_rows=((101, 102), (201, 202)),
        max_tokens=2,
    )
    path = tmp_path / "direct-oracle.json"
    path.write_text(json.dumps(direct.to_json_dict()), encoding="utf-8")
    loaded = ExactTokenOracle.from_json_path(path)

    parity = validate_exact_token_parity(
        loaded,
        mode="http",
        prompt_rows=fixture.prompt_rows,
        generated_rows=((101, 102), (201, 202)),
        max_tokens=2,
    )
    assert parity["passed"] is True
    assert parity["prompt_ids_equal"] is True
    assert parity["generated_ids_equal"] is True

    with pytest.raises(ValueError, match="generated token IDs differ"):
        validate_exact_token_parity(
            loaded,
            mode="http",
            prompt_rows=fixture.prompt_rows,
            generated_rows=((101, 999), (201, 202)),
            max_tokens=2,
        )


def test_exact_token_tool_defaults_to_committed_512_128_gate(tmp_path: Path) -> None:
    tool = _load_tool()
    args = tool.build_parser().parse_args(
        [
            "direct",
            "--model-path",
            str(tmp_path / "model"),
            "--json",
            str(tmp_path / "oracle.json"),
        ]
    )

    assert args.fixture == REPO_FIXTURE
    assert args.prompt_length == 512
    assert args.prompt_count == 1
    assert args.max_tokens == 128


def test_exact_token_tool_validates_http_prompt_and_generated_accounting() -> None:
    tool = _load_tool()
    prompt_rows = ((10, 11, 12), (20, 21, 22))
    response = {
        "choices": [{"text": "a"}, {"text": "b"}],
        "usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
        "hipengine": {
            "prompt_token_accounting": {
                "schema_version": 1,
                "input_type": "token_ids",
                "prompt_token_ids_sha256": [token_ids_sha256(row) for row in prompt_rows],
                "prompt_tokens": [3, 3],
                "total_prompt_tokens": 6,
            },
            "token_accounting": {
                "choice_generated_token_ids": [[101, 102], [201, 202]],
                "choice_generated_tokens": [2, 2],
                "total_generated_tokens": 4,
            },
        },
    }

    assert tool.parse_http_response(response, prompt_rows=prompt_rows, max_tokens=2) == (
        (101, 102),
        (201, 202),
    )

    response["hipengine"]["prompt_token_accounting"]["prompt_token_ids_sha256"][0] = "bad"
    with pytest.raises(tool.ExactTokenBenchError, match="hashes differ"):
        tool.parse_http_response(response, prompt_rows=prompt_rows, max_tokens=2)


def test_exact_token_tool_preserves_http_choice_telemetry() -> None:
    tool = _load_tool()
    response = {
        "choices": [
            {
                "text": "a",
                "hipengine": {
                    "timing": {"batch_decode_ms": 8.0},
                    "timing_scope": "batch",
                    "batch_id": "batch-1",
                    "group_rows": 2,
                    "timing_owner": True,
                },
            },
            {
                "text": "b",
                "hipengine": {
                    "timing": {"batch_decode_ms": 8.0},
                    "timing_scope": "batch",
                    "batch_id": "batch-1",
                    "group_rows": 2,
                    "timing_owner": False,
                },
            },
        ]
    }

    assert tool._http_choice_telemetry(response) == [
        response["choices"][0]["hipengine"],
        response["choices"][1]["hipengine"],
    ]


def test_openai_concurrency_sweep_uses_same_committed_fixture() -> None:
    spec = importlib.util.spec_from_file_location(
        "hipengine_vllm_openai_concurrency_sweep",
        Path("scripts/vllm_openai_concurrency_sweep.py"),
    )
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    rows = tool.load_rows(tool.DEFAULT_FIXTURE, prompt_length=512, count=8)

    assert tool.DEFAULT_FIXTURE == REPO_FIXTURE
    assert len(rows) == 8
    assert all(row == rows[0] for row in rows)


def test_exact_token_direct_artifact_carries_scoped_timing_and_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()

    class _FakeLLM:
        resolved_backend = "hip_gfx1151"
        resolved_quant = "w4_paro"

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def generate_detailed(self, prompts, _sampling):
            return [
                GenerationOutput(
                    text="x",
                    generated_token_ids=(101, 102),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=len(prompt),
                        generated_tokens=2,
                        timing={"decode_ms": 5.0},
                        timing_scope="choice",
                    ),
                )
                for prompt in prompts
            ]

    monkeypatch.setattr(tool, "LLM", _FakeLLM)
    monkeypatch.setattr(
        tool,
        "memory_stats",
        lambda: {
            "current_allocated_bytes": 10,
            "peak_allocated_bytes": 20,
            "total_allocated_bytes": 30,
            "total_freed_bytes": 20,
            "active_allocations": 1,
            "peak_allocations": 2,
        },
    )
    monkeypatch.setattr(
        tool,
        "collect_artifact_provenance",
        lambda **_kwargs: {"kind": "hipengine_artifact_provenance"},
    )
    args = tool.build_parser().parse_args(
        [
            "direct",
            "--model-path",
            str(tmp_path / "model"),
            "--prompt-length",
            "512",
            "--prompt-count",
            "1",
            "--max-tokens",
            "2",
            "--json",
            str(tmp_path / "out.json"),
        ]
    )
    args.model = str(args.model_path)
    fixture = load_exact_token_fixture(REPO_FIXTURE, prompt_length=512, prompt_count=1)

    artifact = tool.run_direct(args, fixture)

    assert artifact["generation_telemetry"] == [
        {
            "decode_state": {
                "row_index": 0,
                "step_index": 2,
                "prompt_tokens": 512,
                "generated_tokens": 2,
                "phase": "done",
                "continuation_eligible": False,
            },
            "timing": {"decode_ms": 5.0},
            "timing_scope": "choice",
            "group_rows": 1,
            "timing_owner": True,
        }
    ]
    assert artifact["memory"] == {
        "scope": "hipengine_tracked_process",
        "current_allocated_bytes": 10,
        "peak_allocated_bytes": 20,
        "total_allocated_bytes": 30,
        "total_freed_bytes": 20,
        "active_allocations": 1,
        "peak_allocations": 2,
    }
