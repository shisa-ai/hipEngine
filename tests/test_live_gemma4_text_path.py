"""The Gemma 4 text path through ``LLM.generate`` with a chat template.

Every Gemma 4 benchmark in ``scripts/`` drives the model with raw token ids --
``gemma4_campaign_bench.py:484`` is ``llm.generate_detailed(list(prompt_ids), params)``
with synthetic ids from ``range(1000, 1000 + n)``. That is the right thing for a kernel
throughput measurement, but it means tokenization, the chat template, and text decoding
are exercised by none of them.

This covers that path. It exists because a chat-templated prompt and a raw one produce
completely different behaviour on this artifact -- the raw prompt terminates immediately
-- and nothing in the benchmark suite would have noticed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_live_gguf_int8_mtp import _hip_available

_MODEL = Path("/models/gguf/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf")

_PROMPT = "Explain in one sentence why the sky is blue."


def _chat(text: str) -> str:
    """Wrap a user turn in the Gemma chat framing."""
    return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"


pytestmark = pytest.mark.skipif(
    not _hip_available() or not _MODEL.is_file(), reason="HIP or GGUF model unavailable",
)


@pytest.fixture(scope="module")
def llm():
    import hipengine

    return hipengine.LLM(model=str(_MODEL))


def _text(out) -> str:
    first = out[0] if isinstance(out, list) and out else out
    return getattr(first, "text", str(first))


def test_chat_templated_prompt_generates_text(llm):
    """A templated prompt produces real output, not an immediate stop."""
    from hipengine.llm import SamplingParams

    text = _text(llm.generate([_chat(_PROMPT)], SamplingParams(max_tokens=32, temperature=0.0)))

    assert text, "chat-templated prompt produced empty output"
    assert "<eos>" not in text, f"chat-templated prompt stopped immediately: {text!r}"
    assert len(text.strip()) > 20, f"chat-templated prompt produced almost nothing: {text!r}"


def test_the_text_path_is_not_the_token_id_path(llm):
    """The two entry points are not interchangeable, which is why both need coverage.

    The benchmarks pass token ids and this passes text. They differ in tokenization and
    chat framing, so a benchmark run passing says nothing about this path. Asserted only
    as far as it is stable: text in, text out, and the token-id path still reachable.
    """
    from hipengine.llm import SamplingParams

    params = SamplingParams(max_tokens=16, temperature=0.0)

    by_text = _text(llm.generate([_chat(_PROMPT)], params))
    assert by_text.strip(), "text entry point produced nothing"

    by_ids = llm.generate_detailed([1000, 1001, 1002, 1003], params)
    assert by_ids, "token-id entry point produced nothing"
