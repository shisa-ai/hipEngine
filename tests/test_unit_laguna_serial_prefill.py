from __future__ import annotations

from inspect import signature
from types import SimpleNamespace

import pytest

from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession


def test_serial_oracle_owns_loop_and_captures_only_last_token():
    from scripts.laguna_serial_prefill import serial_prefill

    calls = []
    captures = SimpleNamespace(rows=1)
    result = object()

    def forward(token, **kwargs):
        calls.append((token, kwargs))
        return result

    session = SimpleNamespace(forward_token=forward)
    assert serial_prefill(session, ("2", 3, 4), capture_last=captures, stream=7) is result
    assert calls == [
        (2, {"captures": None, "stream": 7}),
        (3, {"captures": None, "stream": 7}),
        (4, {"captures": captures, "stream": 7}),
    ]
    assert "use_bulk" not in signature(LagunaGGUFResidentSession.prefill).parameters


@pytest.mark.parametrize("tokens,captures,message", [
    ((), None, "at least one token"),
    ((1,), SimpleNamespace(rows=2), "one-row targets"),
])
def test_serial_oracle_rejects_invalid_inputs_before_forward(tokens, captures, message):
    from scripts.laguna_serial_prefill import serial_prefill

    session = SimpleNamespace(forward_token=lambda *args, **kwargs: pytest.fail("unexpected forward"))
    with pytest.raises(ValueError, match=message):
        serial_prefill(session, tokens, capture_last=captures)


def test_public_prefill_keeps_one_token_primitive_and_chunked_default():
    calls = []
    result = object()
    session = SimpleNamespace(
        _check_open=lambda: None, _check_no_staged_verifier=lambda: None,
        forward_token=lambda token, **kwargs: calls.append(("forward", token, kwargs)) or result,
        _execute_rows=lambda tokens, **kwargs: calls.append(("rows", tokens, kwargs)),
        _project_rows_last=lambda **kwargs: calls.append(("project", kwargs)) or result,
        prefill_chunk_size=2, position=4,
    )
    assert LagunaGGUFResidentSession.prefill(session, (1,), stream=3) is result
    assert calls == [("forward", 1, {"captures": None, "stream": 3})]
    calls.clear()
    captures = SimpleNamespace(rows=1)
    assert LagunaGGUFResidentSession.prefill(session, (1, 2, 3), capture_last=captures, stream=3) is result
    assert calls == [
        ("rows", (1, 2), {"capture_last": None, "stream": 3}),
        ("rows", (3,), {"capture_last": captures, "stream": 3}),
        ("project", {"input_token_id": 3, "position": 4, "row_index": 0, "stream": 3}),
    ]


@pytest.mark.parametrize("mode", ["serial", "bulk"])
def test_target_ar_control_and_warmup_use_independent_serial_helper(monkeypatch, mode):
    from scripts import laguna_target_ar_bench as benchmark

    calls = []
    result = SimpleNamespace(next_token_id=9)
    session = SimpleNamespace(
        prefill=lambda tokens: calls.append(("bulk", tuple(tokens))) or result,
        forward_token=lambda token, **kwargs: calls.append(("forward", token)) or result,
        close=lambda: calls.append(("close",)),
    )
    monkeypatch.setattr(benchmark, "_session", lambda *args: session)
    prompt = {"token_ids": (1, 2, 3), "id": "test", "category": "test",
              "prompt_tokens": 3, "token_ids_sha256": "fixture"}
    args = SimpleNamespace(warmup_output_tokens=1)
    row = benchmark._run_target(object(), prompt, mode=mode, horizons=(2,), repetition=0, args=args)
    prefill_calls = [("bulk", (1, 2, 3))] if mode == "bulk" else [("forward", token) for token in (1, 2, 3)]
    assert calls == [*prefill_calls, ("forward", 9), ("close",)]
    assert row["checkpoints"]["2"]["generated_token_ids"] == [9, 9]
    calls.clear()
    benchmark._warmup(object(), prompt, args)
    assert calls == [("forward", 1), ("forward", 2), ("forward", 3), ("close",),
                     ("bulk", (1, 2, 3)), ("close",)]


def test_laguna_scripts_do_not_pass_retired_prefill_keyword():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path in (root / "scripts").glob("laguna_*.py"):
        tree = ast.parse(path.read_text())
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "prefill":
                assert not any(keyword.arg == "use_bulk" for keyword in call.keywords), path
