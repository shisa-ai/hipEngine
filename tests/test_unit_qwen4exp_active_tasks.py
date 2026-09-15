from scripts.qwen4exp_active_task_checks import score_choice, task_verdict
from types import SimpleNamespace

import numpy as np

from scripts import qwen4exp_active_task_checks as tasks


def test_choice_requires_the_whole_visible_answer():
    assert score_choice(" B\n", "B")
    assert score_choice("b", "B")
    for text in ("A", "A B C D", "The answer is B", "B or C", ""):
        assert not score_choice(text, "B")


def test_reference_errors_are_not_silently_promoted_as_task_success():
    assert task_verdict([{"valid": True, "strict_correct": True, "candidate_correct": True}]) == "passed_supplemental"
    assert task_verdict([{"valid": True, "strict_correct": True, "candidate_correct": False}]) == "task_regression"
    assert task_verdict([{"valid": True, "strict_correct": False, "candidate_correct": False}]) == "reference_unscorable"
    assert task_verdict([{"valid": False, "strict_correct": True, "candidate_correct": True}]) == "invalid_capture"
    assert task_verdict([]) == "invalid_capture"


def test_completion_removes_only_eos_and_tracks_finiteness(monkeypatch):
    monkeypatch.setattr(tasks, "_state_summary", lambda runner: {
        "finite": True, "state_sha256": "state"})
    result = lambda token, value: SimpleNamespace(token_id=token, logits=np.array([value]))
    runner = SimpleNamespace(
        reset=lambda: None, prefill=lambda prompt: result(2, 1.0),
        step=lambda token: result(9, float("nan")))
    tokenizer = SimpleNamespace(eos_token_id=9, decode=lambda ids, **kw: repr(ids))
    capture = tasks.completion(runner, tokenizer, [1], 4)
    assert capture["ids"] == [2, 9]
    assert capture["text"] == "[2]"
    assert capture["finish"] == "eos"
    assert not capture["finite"]
    capture = tasks.completion(runner, tokenizer, [1], 1)
    assert capture["finish"] == "length"
    assert capture["ids"] == [2]


def test_traced_task_requires_actual_chunk_coverage(monkeypatch):
    import pytest

    class Runner:
        chunk = 2

        def _prefill_chunk(self, values):
            pass

    runner = Runner()

    def fake_completion(runner, tokenizer, prompt, limit):
        for start in range(0, len(prompt), runner.chunk):
            runner._prefill_chunk(prompt[start:start + runner.chunk])
        return {"ids": [9], "finish": "eos"}

    monkeypatch.setattr(tasks, "completion", fake_completion)
    result = tasks.traced_completion(runner, None, [1, 2, 3, 4, 5], 4, 2)
    assert result["prefill_chunks"] == [2, 2, 1]
    assert "_prefill_chunk" not in vars(runner)
    with pytest.raises(ValueError, match="chunk coverage"):
        tasks.traced_completion(runner, None, [1, 2, 3, 4, 5], 4, 4)
    assert "_prefill_chunk" not in vars(runner)


def test_active_prompt_uses_embedded_template_without_thinking_and_preserves_length():
    task = {
        "id": "test", "prefix": "QUESTION\n", "suffix": "\nANSWER:",
        "filler": "filler ", "evidence": [{"position": 0.5, "text": "FACT"}],
    }
    calls = []

    def render(messages, *, enable_thinking):
        calls.append((messages, enable_thinking))
        return "USER:" + messages[0]["content"] + ":ASSISTANT:CLOSED_THINK:"

    generator = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda text: list(text.encode()), chat_template="template"),
        render_chat_prompt=render)
    prompt, metadata = tasks.build_active_prompt(generator, task, context_tokens=256)
    text = bytes(prompt).decode()
    assert len(prompt) == 256
    assert text.startswith("USER:QUESTION\n")
    assert text.endswith("\nANSWER::ASSISTANT:CLOSED_THINK:")
    assert text.count("FACT") == 1
    assert calls[0][1] is False
    assert metadata["prompt_format"] == "qwen4exp_embedded"
    assert metadata["enable_thinking"] is False
    assert task["prefix"] == "QUESTION\n"


def test_active_prompt_rejects_template_that_drops_content():
    import pytest

    generator = SimpleNamespace(render_chat_prompt=lambda *args, **kwargs: "no content")
    with pytest.raises(ValueError, match="preserve"):
        tasks.build_active_prompt(generator, {"prefix": "", "suffix": ""})
