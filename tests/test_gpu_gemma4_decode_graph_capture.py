"""Decode-graph replay must be the same step as the launched decode.

The graph path (``Gemma4DecodeGraphSession``) freezes one decode step's
launch shape to a context bucket, re-targets only the KV append destinations
each step, and stages the position-dependent content ahead of the replay.
The contract is that none of that is visible: the logits a replayed step
returns must be byte-identical to what ``runner.forward`` produced from the
same state, at every position, including bucket crossings.

Two sequences, chosen to exercise every shape the bucket maths produces:
60 + 6 steps crosses the width-grid boundary at 64 (a clamped bucket), and
120 + 12 steps crosses the tiled-route singleton at 127 (key count 128,
where the global layers switch to the tiled kernel) and the next grid
boundary at 128.

One test function on purpose — see ``test_live_gemma4_forward_hidden``: the
registry snapshot restore between tests drops kernels a model load registered
lazily, so the reference runs, the graph runs and the assertions stay in one
sequence. The reference runs double as the decode-route warm-up the capture
requires (a cold route would load modules while capture is active).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

_MODEL_DIR = Path("/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF")
_TARGET = _MODEL_DIR / "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"

# Sequence 1: 60 prompt tokens + 6 decode steps puts steps 60..63 in bucket
# [0, 64) and 64..65 in the clamped bucket [64, 127) — one width-grid
# crossing, and the clamp that keeps key count 128 out of an exact-route
# bucket.
_SEQ1_PROMPT = 60
_SEQ1_STEPS = 6
# Sequence 2: steps 120..131 cross the singleton bucket [127, 128) — key
# count 128 is where the global layers' tiled route admits — and then the
# [128, 192) grid bucket.
_SEQ2_PROMPT = 120
_SEQ2_STEPS = 12


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


def _launched(runner, prompt, decode):
    """Run the reference sequence: prefill, then one forward per token."""

    runner.reset()
    runner.forward(prompt)
    return [runner.forward([token]) for token in decode]


def test_gemma4_decode_graph_replay_matches_launched_decode():
    import hipengine

    from hipengine.runtime.gemma4_decode_graph import Gemma4DecodeGraphSession

    llm = hipengine.LLM(model=str(_TARGET))
    runner = llm._get_text_generator()._ensure_runner()
    vocab = int(runner.weights.config.vocab_size or 0)
    assert vocab > 0

    rng = np.random.default_rng(20260930)
    sequences = [
        (
            [int(t) for t in rng.integers(0, vocab, size=_SEQ1_PROMPT)],
            [int(t) for t in rng.integers(0, vocab, size=_SEQ1_STEPS)],
            2,  # [0, 64) then the clamped [64, 127)
        ),
        (
            [int(t) for t in rng.integers(0, vocab, size=_SEQ2_PROMPT)],
            [int(t) for t in rng.integers(0, vocab, size=_SEQ2_STEPS)],
            3,  # [64, 127), the [127, 128) singleton, then [128, 192)
        ),
    ]

    for prompt, decode, want_captures in sequences:
        start = len(prompt)
        # --- the launched path (production) ------------------------------
        expected = _launched(runner, prompt, decode)
        assert runner.position == start + len(decode)

        # --- the graph path, from an identical state ---------------------
        # Reset and re-prefill rebuild the same KV state: reset rewinds the
        # position and the append offsets, and the mask bounds every read by
        # the live context, so stale slots past it cannot contribute.
        runner.reset()
        runner.forward(prompt)
        with Gemma4DecodeGraphSession(runner) as session:
            replayed = [session.step(token) for token in decode]
            assert session.captures == want_captures, (
                f"steps {start}..{start + len(decode) - 1} span the bucket "
                f"boundaries at 127/128 as applicable; expected {want_captures} "
                f"captures, got {session.captures} — a capture per step means "
                "the bucket key is drifting every position"
            )
        assert runner.position == start + len(decode)

        # --- the contract ------------------------------------------------
        for step, (want, got) in enumerate(zip(expected, replayed, strict=True)):
            np.testing.assert_array_equal(
                got,
                want,
                err_msg=(
                    f"sequence at prompt {start}: step {step} "
                    f"(position {start + step}): replayed logits differ from "
                    "the launched path"
                ),
            )