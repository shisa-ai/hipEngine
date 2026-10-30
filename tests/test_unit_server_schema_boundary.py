"""Boundary contract for the OpenAI request schemas under ``server/schemas.py``.

The schemas were moved out of ``hipengine/server/api.py`` unchanged. These
checks pin the two properties the move depends on: the schema module has no
application/model/runtime dependency, and the HTTP boundary still exposes the
same fields, defaults, extra-field behavior, validation errors and JSON schema.
"""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from hipengine.server import schemas

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_PATH = REPO_ROOT / "hipengine" / "server" / "schemas.py"

EXPECTED_FIELDS = {
    "SongGenerationRequest": (
        "model", "style", "lyrics", "cot", "seed", "abc", "cfg_scale", "steps",
        "response_format", "request_id",
    ),
    "CompletionRequest": (
        "model", "prompt", "max_tokens", "max_completion_tokens", "temperature",
        "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
        "frequency_penalty", "logit_bias", "suppress_token_ids", "min_tokens",
        "eos_token_id", "n", "stream", "stream_options", "timeout_ms", "stop",
        "seed", "echo", "logprobs", "ignore_eos", "kv_storage", "kv_scale_dtype",
        "kv_scale_granularity", "response_format", "guided_json", "guided_regex",
        "guided_choice", "guided_patch", "guided_diff", "continuation_id",
        "session", "speculative_mtp", "speculative",
    ),
    "ChatMessage": ("role", "content", "name", "tool_call_id", "tool_calls"),
    "ChatCompletionRequest": (
        "model", "messages", "max_tokens", "max_completion_tokens", "temperature",
        "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
        "frequency_penalty", "logit_bias", "suppress_token_ids", "min_tokens",
        "eos_token_id", "n", "stream", "stream_options", "timeout_ms", "tools",
        "tool_choice", "parallel_tool_calls", "invalid_tool_call_error_mode",
        "reasoning_effort", "enable_thinking", "max_think_tokens",
        "min_answer_tokens", "hard_think_cap", "soft_close_window",
        "hard_close_message", "hard_close_sequence", "thinking_token_budget",
        "chat_template_kwargs", "thinking", "reasoning", "stop", "seed",
        "logprobs", "top_logprobs", "ignore_eos", "kv_storage", "kv_scale_dtype",
        "kv_scale_granularity", "response_format", "guided_json", "guided_regex",
        "guided_choice", "guided_patch", "guided_diff", "continuation_id",
        "session", "speculative_mtp", "speculative",
    ),
    "SessionForkRequest": ("id",),
    "SessionRollbackRequest": ("message_count",),
    "TokenizeRequest": ("text",),
    "DetokenizeRequest": ("token_ids", "skip_special"),
    "TokenDiagnosticRequest": (
        "text", "messages", "tools", "tool_choice", "reasoning_effort",
        "enable_thinking", "max_think_tokens", "min_answer_tokens",
        "hard_think_cap", "soft_close_window", "hard_close_message",
        "hard_close_sequence", "thinking_token_budget", "chat_template_kwargs",
        "thinking", "reasoning", "guided_json", "guided_regex", "guided_choice",
        "guided_patch", "guided_diff", "session",
    ),
    "FitContextRequest": (
        "text", "messages", "tools", "tool_choice", "reasoning_effort",
        "enable_thinking", "max_think_tokens", "min_answer_tokens",
        "hard_think_cap", "soft_close_window", "hard_close_message",
        "hard_close_sequence", "thinking_token_budget", "chat_template_kwargs",
        "thinking", "reasoning", "guided_json", "guided_regex", "guided_choice",
        "guided_patch", "guided_diff", "session", "max_tokens",
    ),
}

EXPECTED_REQUIRED = {
    "SongGenerationRequest": ("style", "lyrics"),
    "CompletionRequest": (),
    "ChatMessage": ("role",),
    "ChatCompletionRequest": (),
    "SessionForkRequest": ("id",),
    "SessionRollbackRequest": ("message_count",),
    "TokenizeRequest": ("text",),
    "DetokenizeRequest": ("token_ids",),
    "TokenDiagnosticRequest": (),
    "FitContextRequest": (),
}

VALID_CASES = {
    "SongGenerationRequest": (
        {
            "model": "song-model",
            "style": "warm pop",
            "lyrics": "[verse]\nhi",
            "cot": "melody",
            "seed": 11,
            "abc": "abc",
            "cfg_scale": 3.5,
            "steps": 32,
            "response_format": "wav",
            "request_id": "req-1",
        },
        {
            "model": "song-model",
            "style": "warm pop",
            "lyrics": "[verse]\nhi",
            "cot": "melody",
            "seed": 11,
            "abc": "abc",
            "cfg_scale": 3.5,
            "steps": 32,
            "response_format": "wav",
            "request_id": "req-1",
        },
    ),
    "CompletionRequest": (
        {
            "model": "m",
            "prompt": ["hi", "there"],
            "max_tokens": 4,
            "temperature": 0.25,
            "top_p": 0.9,
            "stop": ["\n"],
            "extra_field": {"kept": True},
        },
        {
            "model": "m",
            "prompt": ["hi", "there"],
            "max_tokens": 4,
            "max_completion_tokens": None,
            "temperature": 0.25,
            "top_p": 0.9,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logit_bias": None,
            "suppress_token_ids": None,
            "min_tokens": 0,
            "eos_token_id": None,
            "n": 1,
            "stream": False,
            "stream_options": None,
            "timeout_ms": None,
            "stop": ["\n"],
            "seed": None,
            "echo": False,
            "logprobs": None,
            "ignore_eos": False,
            "kv_storage": None,
            "kv_scale_dtype": None,
            "kv_scale_granularity": None,
            "response_format": None,
            "guided_json": None,
            "guided_regex": None,
            "guided_choice": None,
            "guided_patch": None,
            "guided_diff": None,
            "continuation_id": None,
            "session": None,
            "speculative_mtp": None,
            "speculative": None,
            "extra_field": {"kept": True},
        },
    ),
    "ChatMessage": (
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "name": "assistant",
            "tool_call_id": "call-1",
            "tool_calls": [{"id": "call-1"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "name": "assistant",
            "tool_call_id": "call-1",
            "tool_calls": [{"id": "call-1"}],
        },
    ),
    "ChatCompletionRequest": (
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "temperature": 0.2,
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "auto",
            "reasoning_effort": "low",
            "extra_field": 7,
        },
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": "hi",
                    "name": None,
                    "tool_call_id": None,
                    "tool_calls": None,
                }
            ],
            "max_tokens": 8,
            "max_completion_tokens": None,
            "temperature": 0.2,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logit_bias": None,
            "suppress_token_ids": None,
            "min_tokens": 0,
            "eos_token_id": None,
            "n": 1,
            "stream": False,
            "stream_options": None,
            "timeout_ms": None,
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "auto",
            "parallel_tool_calls": None,
            "invalid_tool_call_error_mode": None,
            "reasoning_effort": "low",
            "enable_thinking": None,
            "max_think_tokens": None,
            "min_answer_tokens": None,
            "hard_think_cap": None,
            "soft_close_window": None,
            "hard_close_message": None,
            "hard_close_sequence": None,
            "thinking_token_budget": None,
            "chat_template_kwargs": None,
            "thinking": None,
            "reasoning": None,
            "stop": None,
            "seed": None,
            "logprobs": None,
            "top_logprobs": None,
            "ignore_eos": False,
            "kv_storage": None,
            "kv_scale_dtype": None,
            "kv_scale_granularity": None,
            "response_format": None,
            "guided_json": None,
            "guided_regex": None,
            "guided_choice": None,
            "guided_patch": None,
            "guided_diff": None,
            "continuation_id": None,
            "session": None,
            "speculative_mtp": None,
            "speculative": None,
            "extra_field": 7,
        },
    ),
    "SessionForkRequest": ({"id": "session-1"}, {"id": "session-1"}),
    "SessionRollbackRequest": ({"message_count": 3}, {"message_count": 3}),
    "TokenizeRequest": ({"text": "hello"}, {"text": "hello"}),
    "DetokenizeRequest": (
        {"token_ids": [1, 2, 3], "skip_special": True},
        {"token_ids": [1, 2, 3], "skip_special": True},
    ),
    "TokenDiagnosticRequest": (
        {"text": "hello", "reasoning_effort": "low", "guided_json": {"type": "object"}},
        {
            "text": "hello",
            "messages": None,
            "tools": None,
            "tool_choice": None,
            "reasoning_effort": "low",
            "enable_thinking": None,
            "max_think_tokens": None,
            "min_answer_tokens": None,
            "hard_think_cap": None,
            "soft_close_window": None,
            "hard_close_message": None,
            "hard_close_sequence": None,
            "thinking_token_budget": None,
            "chat_template_kwargs": None,
            "thinking": None,
            "reasoning": None,
            "guided_json": {"type": "object"},
            "guided_regex": None,
            "guided_choice": None,
            "guided_patch": None,
            "guided_diff": None,
            "session": None,
        },
    ),
    "FitContextRequest": (
        {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64,
            "session": "session-1",
        },
        {
            "text": None,
            "messages": [
                {
                    "role": "user",
                    "content": "hello",
                    "name": None,
                    "tool_call_id": None,
                    "tool_calls": None,
                }
            ],
            "tools": None,
            "tool_choice": None,
            "reasoning_effort": None,
            "enable_thinking": None,
            "max_think_tokens": None,
            "min_answer_tokens": None,
            "hard_think_cap": None,
            "soft_close_window": None,
            "hard_close_message": None,
            "hard_close_sequence": None,
            "thinking_token_budget": None,
            "chat_template_kwargs": None,
            "thinking": None,
            "reasoning": None,
            "guided_json": None,
            "guided_regex": None,
            "guided_choice": None,
            "guided_patch": None,
            "guided_diff": None,
            "session": "session-1",
            "max_tokens": 64,
        },
    ),
}

INVALID_CASES = [
    ("SongGenerationRequest", {"style": "", "lyrics": "x"}, [("string_too_short", ["style"])]),
    ("SongGenerationRequest", {"style": "x", "lyrics": ""}, [("string_too_short", ["lyrics"])]),
    ("SongGenerationRequest", {"style": "x", "lyrics": "y", "seed": -1}, [("greater_than_equal", ["seed"])]),
    ("SongGenerationRequest", {"style": "x", "lyrics": "y", "cfg_scale": 20.5}, [("less_than_equal", ["cfg_scale"])]),
    ("SongGenerationRequest", {"style": "x", "lyrics": "y", "steps": 0}, [("greater_than_equal", ["steps"])]),
    ("CompletionRequest", {"max_tokens": -1}, [("greater_than_equal", ["max_tokens"])]),
    ("CompletionRequest", {"temperature": -0.1}, [("greater_than_equal", ["temperature"])]),
    ("CompletionRequest", {"top_p": 1.1}, [("less_than_equal", ["top_p"])]),
    ("CompletionRequest", {"top_k": -1}, [("greater_than_equal", ["top_k"])]),
    ("CompletionRequest", {"repetition_penalty": 0}, [("greater_than", ["repetition_penalty"])]),
    ("CompletionRequest", {"timeout_ms": 0}, [("greater_than", ["timeout_ms"])]),
    ("CompletionRequest", {"logprobs": 21}, [("less_than_equal", ["logprobs"])]),
    ("ChatMessage", {}, [("missing", ["role"])]),
    ("ChatMessage", {"role": 3}, [("string_type", ["role"])]),
    (
        "ChatCompletionRequest",
        {"messages": [{"content": "missing role"}]},
        [("missing", ["messages", 0, "role"])],
    ),
    ("ChatCompletionRequest", {"max_tokens": -1}, [("greater_than_equal", ["max_tokens"])]),
    ("ChatCompletionRequest", {"top_logprobs": 21}, [("less_than_equal", ["top_logprobs"])]),
    ("SessionForkRequest", {}, [("missing", ["id"])]),
    ("SessionRollbackRequest", {}, [("missing", ["message_count"])]),
    ("SessionRollbackRequest", {"message_count": -1}, [("greater_than_equal", ["message_count"])]),
    ("TokenizeRequest", {}, [("missing", ["text"])]),
    ("TokenizeRequest", {"text": 1}, [("string_type", ["text"])]),
    ("DetokenizeRequest", {"token_ids": "not-a-list"}, [("list_type", ["token_ids"])]),
    ("TokenDiagnosticRequest", {"max_think_tokens": -1}, [("greater_than_equal", ["max_think_tokens"])]),
    ("FitContextRequest", {"max_tokens": -1}, [("greater_than_equal", ["max_tokens"])]),
]


def test_schemas_module_depends_only_on_pydantic_and_stdlib() -> None:
    tree = ast.parse(SCHEMAS_PATH.read_text(encoding="utf-8"), filename=str(SCHEMAS_PATH))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.add((node.module or "").split(".", 1)[0])
        elif isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
    assert modules <= {"__future__", "typing", "pydantic"}, modules


def test_schemas_file_loads_without_importing_the_api_or_engine() -> None:
    probe = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('schema_probe', {str(SCHEMAS_PATH)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert 'hipengine.server.api' not in sys.modules\n"
        "assert not any(name == 'hipengine' or name.startswith('hipengine.') "
        "for name in sys.modules)\n"
        "assert tuple(module.ChatCompletionRequest.model_fields) == "
        "('model', 'messages', 'max_tokens', 'max_completion_tokens', "
        "'temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty', "
        "'presence_penalty', 'frequency_penalty', 'logit_bias', "
        "'suppress_token_ids', 'min_tokens', 'eos_token_id', 'n', 'stream', "
        "'stream_options', 'timeout_ms', 'tools', 'tool_choice', "
        "'parallel_tool_calls', 'invalid_tool_call_error_mode', "
        "'reasoning_effort', 'enable_thinking', 'max_think_tokens', "
        "'min_answer_tokens', 'hard_think_cap', 'soft_close_window', "
        "'hard_close_message', 'hard_close_sequence', 'thinking_token_budget', "
        "'chat_template_kwargs', 'thinking', 'reasoning', 'stop', 'seed', "
        "'logprobs', 'top_logprobs', 'ignore_eos', 'kv_storage', "
        "'kv_scale_dtype', 'kv_scale_granularity', 'response_format', "
        "'guided_json', 'guided_regex', 'guided_choice', 'guided_patch', "
        "'guided_diff', 'continuation_id', 'session', 'speculative_mtp', "
        "'speculative')\n"
        "assert 'torch' not in sys.modules\n"
        "print('SCHEMA_PROBE_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "SCHEMA_PROBE_OK"


def test_package_import_of_schemas_stays_schema_only() -> None:
    probe = (
        "import sys\n"
        "import hipengine.server.schemas as schemas\n"
        "blocked = ('fastapi', 'hipengine.server.api', 'hipengine.generation', "
        "'hipengine.runtime', 'hipengine.models', 'hipengine.kvcache', "
        "'hipengine.speculative', 'hipengine.llm', 'hipengine.kernels', "
        "'hipengine.dispatch', 'hipengine.loading', 'hipengine.quant', 'torch')\n"
        "loaded = [name for name in sys.modules if any(name == prefix or "
        "name.startswith(prefix + '.') for prefix in blocked)]\n"
        "assert loaded == [], loaded\n"
        "assert 'hipengine.server.schemas' in sys.modules\n"
        "assert tuple(schemas.ChatMessage.model_fields) == "
        "('role', 'content', 'name', 'tool_call_id', 'tool_calls')\n"
        "assert schemas.CompletionRequest.model_validate("
        "{'prompt': 'hi', 'max_tokens': 2}).max_tokens == 2\n"
        "print('SCHEMA_PACKAGE_IMPORT_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "SCHEMA_PACKAGE_IMPORT_OK"


def test_lazy_server_exports_resolve_to_api_owned_objects() -> None:
    probe = (
        "import sys\n"
        "import hipengine.server as server\n"
        "assert 'hipengine.server.api' not in sys.modules\n"
        "from hipengine.server import ServerConfig, create_app, render_chat_prompt\n"
        "import hipengine.server.api as api\n"
        "assert ServerConfig is api.ServerConfig\n"
        "assert create_app is api.create_app\n"
        "assert render_chat_prompt is api.render_chat_prompt\n"
        "assert server.ServerConfig is ServerConfig\n"
        "from hipengine.server import ServerConfig as again\n"
        "assert again is ServerConfig\n"
        "from hipengine.server import api as api_from_package\n"
        "assert api_from_package is api\n"
        "assert server.__all__ == ['ServerConfig', 'create_app', 'render_chat_prompt']\n"
        "assert {'ServerConfig', 'create_app', 'render_chat_prompt'} <= set(dir(server))\n"
        "assert ServerConfig(model='fake').model == 'fake'\n"
        "assert callable(create_app) and callable(render_chat_prompt)\n"
        "try:\n"
        "    server.not_an_export\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('unknown attribute resolved')\n"
        "print('SERVER_LAZY_EXPORT_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "SERVER_LAZY_EXPORT_OK"


def test_api_imports_the_schema_classes_its_request_handling_uses() -> None:
    from hipengine.server import api

    for name in schemas.__all__:
        assert getattr(api, name) is getattr(schemas, name), name
        assert getattr(api, name).__module__ == "hipengine.server.schemas"


@pytest.mark.parametrize("name", tuple(EXPECTED_FIELDS))
def test_schema_declares_the_pre_move_fields_in_order(name: str) -> None:
    cls = getattr(schemas, name)
    assert tuple(cls.model_fields) == EXPECTED_FIELDS[name]
    assert tuple(field for field, info in cls.model_fields.items() if info.is_required()) == (
        EXPECTED_REQUIRED[name]
    )


@pytest.mark.parametrize("name", tuple(EXPECTED_FIELDS))
def test_json_schema_keeps_openapi_shape(name: str) -> None:
    schema = getattr(schemas, name).model_json_schema()
    assert schema["title"] == name
    assert schema["additionalProperties"] is True
    assert tuple(schema["properties"]) == EXPECTED_FIELDS[name]
    assert tuple(schema.get("required", ())) == EXPECTED_REQUIRED[name]


def test_json_schema_keeps_constraints_used_by_the_http_contract() -> None:
    completion = schemas.CompletionRequest.model_json_schema()["properties"]
    assert completion["max_tokens"]["default"] == 16
    assert completion["max_tokens"]["anyOf"][0]["minimum"] == 0
    assert completion["temperature"]["anyOf"][0]["minimum"] == 0.0
    assert completion["top_p"]["anyOf"][0] == {
        "maximum": 1.0,
        "minimum": 0.0,
        "type": "number",
    }
    assert completion["timeout_ms"]["anyOf"][0]["exclusiveMinimum"] == 0.0
    assert completion["logprobs"]["anyOf"][0]["maximum"] == 20

    song = schemas.SongGenerationRequest.model_json_schema()["properties"]
    assert song["style"]["minLength"] == 1
    assert song["seed"]["anyOf"][0]["exclusiveMaximum"] == 2**63
    assert song["cfg_scale"]["anyOf"][0]["maximum"] == 20.0

    fit_context = schemas.FitContextRequest.model_json_schema()["properties"]
    assert fit_context["max_tokens"]["anyOf"][0]["minimum"] == 0
    assert fit_context["max_tokens"]["default"] is None


@pytest.mark.parametrize("name", tuple(VALID_CASES))
def test_valid_request_dump_is_unchanged(name: str) -> None:
    body, expected = VALID_CASES[name]
    request = getattr(schemas, name).model_validate(body)
    assert request.model_dump() == expected
    assert request.model_dump(mode="json") == expected


def test_extra_fields_stay_allowed_and_are_kept_in_the_dump() -> None:
    request = schemas.CompletionRequest.model_validate(
        {"prompt": "hi", "unknown_parameter": {"nested": True}}
    )
    assert request.model_extra == {"unknown_parameter": {"nested": True}}
    assert request.model_dump()["unknown_parameter"] == {"nested": True}
    assert schemas.CompletionRequest.model_config["extra"] == "allow"


def test_chat_message_default_list_is_not_shared_between_requests() -> None:
    first = schemas.ChatCompletionRequest()
    second = schemas.ChatCompletionRequest()
    assert first.messages == []
    first.messages.append(schemas.ChatMessage(role="user"))
    assert second.messages == []


@pytest.mark.parametrize(("name", "body", "expected"), INVALID_CASES)
def test_invalid_request_errors_are_unchanged(name: str, body: dict, expected: list) -> None:
    with pytest.raises(ValidationError) as excinfo:
        getattr(schemas, name).model_validate(body)
    assert [(error["type"], list(error["loc"])) for error in excinfo.value.errors()] == expected
