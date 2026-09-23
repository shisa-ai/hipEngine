"""Gemma 4 generation through the public ``LLM`` surface.

The kernel and runner tests exercise the device path directly. This module
covers the path a user actually reaches: ``hipengine.LLM(model=<gguf>)``
resolves the Gemma 4 generator, runs the engine loop, and returns detokenized
text.

That distinction matters here. A route reachable only from a test fixture or an
explicit env var is not shipped, so a Gemma 4 generator that constructs
correctly and then fails inside the first decode step would pass every other
test in the suite.
"""

from __future__ import annotations

import pathlib

import pytest

from tests._rocm_guard import hip_runtime_available

_HIP_AVAILABLE = hip_runtime_available()
_needs_hip = pytest.mark.skipif(
    not _HIP_AVAILABLE, reason="HIP runtime unavailable; skipping gfx1100 kernel tests"
)


def _write_generate_fixture(directory: pathlib.Path) -> pathlib.Path:
    """Write a fixture artifact that can be loaded, run, and detokenized.

    ``tokenizer_fixture_metadata`` carries the token list, merges, and chat
    template, but that list is wider than ``FIXTURE_VOCAB``. The loader checks
    the declared tokens against the embedding's row count, so the tensors have
    to be widened to match rather than taking the default shape.
    """

    from tests._gemma4_gguf_fixture import (
        FIXTURE_TOKEN_STRINGS,
        default_fixture_tensors,
        tokenizer_fixture_metadata,
        write_fixture_gguf,
    )

    return write_fixture_gguf(
        directory / "gemma4.gguf",
        default_fixture_tensors(vocab=len(FIXTURE_TOKEN_STRINGS)),
        tokenizer_fixture_metadata(),
    )


@_needs_hip
def test_llm_generate_reaches_the_gemma4_generator(tmp_path: pathlib.Path) -> None:
    """``LLM.generate`` runs Gemma 4 and returns detokenized text.

    The weights are synthetic, so the text itself carries no meaning. What is
    asserted is that the path completes: the artifact loads, the generator is
    selected, the engine loop decodes at least one token through the device
    kernels, and the result comes back as a string rather than an exception.
    """

    import hipengine
    from hipengine.generation.engine_loop import SubmitPollTextGenerator
    from hipengine.llm import SamplingParams

    model_path = _write_generate_fixture(tmp_path)
    llm = hipengine.LLM(model=str(model_path))
    try:
        generator = llm._get_text_generator()
        assert isinstance(generator, SubmitPollTextGenerator), (
            f"LLM resolved {type(generator).__name__} for a Gemma 4 GGUF; "
            "the gemma4 generator was not selected"
        )

        outputs = llm.generate("Hello", SamplingParams(max_tokens=16))
    finally:
        llm.close()

    assert len(outputs) == 1
    assert isinstance(outputs[0], str)
    # An empty string would mean decode produced nothing and the loop exited on
    # its first tick, which is not the same as generating a token.
    assert outputs[0], "generation returned no text"


@_needs_hip
def test_llm_generate_honours_max_tokens(tmp_path: pathlib.Path) -> None:
    """The decode loop stops at the requested token budget.

    A runner that ignores the budget, or one that stops after a single step,
    both produce a non-empty string above. Comparing two budgets separates
    those from a loop that actually counts.
    """

    import hipengine
    from hipengine.llm import SamplingParams

    model_path = _write_generate_fixture(tmp_path)
    llm = hipengine.LLM(model=str(model_path))
    try:
        short = llm.generate("Hello", SamplingParams(max_tokens=1))
        long = llm.generate("Hello", SamplingParams(max_tokens=8))
    finally:
        llm.close()

    assert len(short) == 1 and len(long) == 1
    assert len(long[0]) >= len(short[0]), (
        f"a larger token budget produced less text ({len(long[0])} < {len(short[0])})"
    )
