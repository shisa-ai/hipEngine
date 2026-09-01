from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_llamacpp_exact_profile.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "qwen4exp_llamacpp_exact_profile", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_case_and_decode_prompt_use_exact_ids() -> None:
    module = _load_script()
    fixture = {
        "cases": [
            {
                "id": "code-p512",
                "category": "code",
                "prompt_tokens": 3,
                "prompt_token_ids": [10, 11, 12],
            }
        ]
    }

    case = module._select_case(fixture, "code-p512")

    assert module._decode_prompt(case, {"tokens": [99]}) == [10, 11, 12, 99]
    with pytest.raises(ValueError, match="one output token"):
        module._decode_prompt(case, {"tokens": []})
    with pytest.raises(ValueError, match="exactly one"):
        module._select_case(fixture, "general_en-p512")


def test_completion_payload_pins_sampler_and_cache_policy() -> None:
    module = _load_script()

    payload = module._completion_payload(
        [10, 11, 12], n_predict=1, cache_prompt=True
    )

    assert payload == {
        "prompt": [10, 11, 12],
        "n_predict": 1,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": 12345,
        "ignore_eos": True,
        "cache_prompt": True,
        "stream": False,
        "return_tokens": True,
    }


def test_server_environment_enables_direct_profiler_attach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.delenv("ROCP_TOOL_ATTACH", raising=False)

    environment = module._server_environment()

    assert environment["ROCP_TOOL_ATTACH"] == "1"


def test_parser_accepts_repeated_cases_and_server_args(tmp_path: Path) -> None:
    module = _load_script()

    args = module.build_parser().parse_args(
        [
            "--server-bin",
            str(tmp_path / "llama-server"),
            "--source-root",
            str(tmp_path / "source"),
            "--model",
            str(tmp_path / "model.gguf"),
            "--case-id",
            "code-p512",
            "--case-id",
            "code-p4096",
            "--server-arg=-ngl",
            "--server-arg=999",
            "--trace-root",
            str(tmp_path / "trace"),
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert args.case_id == ["code-p512", "code-p4096"]
    assert args.server_arg == ["-ngl", "999"]
