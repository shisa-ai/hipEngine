"""Live wiring for the Gemma 4 session adapter: the adapter must be arithmetic-
transparent over a real runner.

Piece (B)'s whole justification for existing is that wrapping ``Gemma4Runner``
in a ``prefill``/``step`` contract changes no arithmetic -- so this drives one
greedy trajectory directly through ``runner.forward`` and the same trajectory
through the session, on the real artifact, and requires the logits and tokens
to be identical. If the adapter ever added, reordered or re-scaled anything,
the byte comparison fails here rather than in a gate result later.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.runtime.gemma4_session import Gemma4ResidentSession

_MODEL_DIR = Path("/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF")
_TARGET = _MODEL_DIR / "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available() or not _TARGET.exists(),
    reason="needs ROCm (libamdhip64.so) and the gemma4 fixture artifact",
)

# Four steps after prefill: enough for a chain to diverge if the adapter
# touched state, short enough that the test stays a minute-scale run.
_DECODE_STEPS = 4


def _build_runner():
    import hipengine

    llm = hipengine.LLM(model=str(_TARGET))
    generator = llm._get_text_generator()
    runner = generator._ensure_runner()
    return llm, runner


def _prompt(vocab: int) -> list[int]:
    # Fixed ids well inside the vocabulary; greedy decoding needs no sampling
    # seed, so the chain is reproducible by construction.
    return [pid % vocab for pid in (101, 202, 303, 404, 505)]


def test_live_gemma4_session_matches_direct_forward_chain():
    llm, runner = _build_runner()
    vocab = int(runner.weights.config.vocab_size or 0)
    assert vocab > 0, "fixture carries no vocab_size"
    prompt = _prompt(vocab)

    # --- Chain A: what the runner does with no adapter in between.
    direct_logits: list[np.ndarray] = []
    tokens_a: list[int] = []
    logits = runner.forward(prompt)
    direct_logits.append(logits)
    token = runner.next_token(logits)
    tokens_a.append(token)
    for _ in range(_DECODE_STEPS):
        logits = runner.forward([token])
        direct_logits.append(logits)
        token = runner.next_token(logits)
        tokens_a.append(token)

    runner.reset()
    assert runner.position == 0

    # --- Chain B: the identical trajectory through the session contract.
    session_logits: list[np.ndarray] = []
    tokens_b: list[int] = []
    with Gemma4ResidentSession(
        runner, backend="hip_gfx1100", target_arch="gemma4"
    ) as session:
        assert session.runner is runner
        assert session.backend == "hip_gfx1100"
        assert session.target_arch == "gemma4"

        result = session.prefill(prompt)
        session_logits.append(result.logits)
        tokens_b.append(result.token_id)
        assert session.position == len(prompt)

        for _ in range(_DECODE_STEPS):
            result = session.step(tokens_b[-1])
            session_logits.append(result.logits)
            tokens_b.append(result.token_id)
            assert session.position == len(prompt) + len(tokens_b) - 1

    # The adapter must not perturb a single value: same logits bytes, same
    # greedy decisions, same positions.
    assert tokens_b == tokens_a, (
        f"adapter changed greedy decisions: direct={tokens_a} session={tokens_b}"
    )
    assert len(session_logits) == len(direct_logits)
    for index, (expected, actual) in enumerate(zip(direct_logits, session_logits)):
        assert actual.shape == expected.shape, f"step {index}: shape drifted"
        np.testing.assert_array_equal(
            actual, expected, err_msg=f"step {index}: adapter perturbed logits"
        )


def test_live_gemma4_session_reset_starts_a_second_prompt():
    """reset() must rewind the runner, since the smoke runs several prompts."""
    _, runner = _build_runner()
    vocab = int(runner.weights.config.vocab_size or 0)
    first = _prompt(vocab)

    with Gemma4ResidentSession(
        runner, backend="hip_gfx1100", target_arch="gemma4"
    ) as session:
        session.prefill(first)
        position_after_first = session.position
        assert position_after_first == len(first)

        session.reset()
        assert session.position == 0

        second = [pid % vocab for pid in (9, 8, 7)]
        result = session.prefill(second)
        assert session.position == len(second)
        assert result.logits.shape[0] == vocab