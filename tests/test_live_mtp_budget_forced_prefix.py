"""Forced acceptance prefixes repair real GPU state, and device faults clean up.

``tests/test_unit_mtp_budget_session_lifecycle.py`` forces every acceptance
prefix through doubled resources, which proves control ordering only. This file
runs the same diagnostic budget protocol on the real dense GGUF target and NextN
provider: the host proposal is replaced by the true greedy continuation with a
wrong token at position ``k``, so the target must accept exactly ``k`` drafts at
every cycle and repair its Conv/GDN/KV state from that prefix. Generated IDs
must equal plain greedy decoding for every ``(budget, k)``, which they cannot
if any rejected-row state leaks into a later cycle.

A second case injects a failure into the target commit mid-request, then reuses
the same decoder for a clean request and requires the same exact output and a
drained tracked-allocation count once everything closes.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

try:  # pragma: no cover - platform guard
    ctypes.CDLL("libamdhip64.so")
    _HIP_AVAILABLE = True
except OSError:  # pragma: no cover - no ROCm on this runner
    _HIP_AVAILABLE = False

_MODEL = Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf")

pytestmark = [
    pytest.mark.skipif(not _HIP_AVAILABLE, reason="requires the HIP runtime"),
    pytest.mark.skipif(not _MODEL.exists(), reason=f"requires {_MODEL}"),
]

_PROMPT = tuple(
    (7734, 264, 12654, 709, 421, 4523, 279, 307, 7324, 76938, 1324, 1608, 20781, 1954, 13)[
        index % 15
    ]
    for index in range(96)
)
_NEW_TOKENS = 14


@pytest.fixture(scope="module")
def dense_mtp():
    from hipengine.core import memory
    from hipengine.runtime import qwen35_gguf_mtp as runtime
    from hipengine.runtime.qwen35_gguf_nextn import (
        Qwen35GGUFNextNDraftProvider,
        borrow_qwen35_gguf_nextn_fallback_weights,
    )
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    baseline = memory.live_allocation_histogram()["allocation_count"]
    target = Qwen35GGUFResidentSession(
        _MODEL, max_sequence_length=256, use_wmma_prefill=True, use_gemv_decode=True
    )
    target.__enter__()
    provider = None
    decoder = None
    try:
        target.select_prefill_quant("gguf_q4_k_m")
        root = int(target.prefill(_PROMPT, use_bulk=True, return_logits=False).token_id)
        expected = [root]
        while len(expected) < _NEW_TOKENS:
            expected.append(int(target.step(expected[-1], return_logits=False).token_id))
        target.reset()
        provider = Qwen35GGUFNextNDraftProvider.from_model(
            _MODEL,
            max_positions=256,
            max_requests=1,
            runtime=target.runtime,
            borrowed_fallback_weights=borrow_qwen35_gguf_nextn_fallback_weights(target),
        )
        decoder = runtime.Qwen35GGUFMTPDecodeSession(
            target, provider, candidate_budget=3, quant="gguf_q4_k_m"
        )
        yield decoder, provider, tuple(expected), int(target.runner.weights.config.vocab_size)
    finally:
        if decoder is not None:
            decoder.close()
        if provider is not None:
            provider.close()
        target.__exit__(None, None, None)
    assert memory.live_allocation_histogram()["allocation_count"] == baseline


def _force_prefix(monkeypatch, provider, expected, vocab, accepted):
    """Replace the host proposal with the true continuation, wrong at ``accepted``."""

    from hipengine.runtime import qwen35_gguf_mtp as runtime
    from hipengine.speculative.mtp import MtpDraftRequest, compile_mtp_chain

    monkeypatch.setattr(runtime, "_maybe_launch_device_proposal", lambda *a, **k: None)
    original = provider.propose
    forced: list[tuple[int, ...]] = []

    def propose(context, *, candidate_budget, return_logits=False, **kwargs):
        original(context, candidate_budget=candidate_budget, return_logits=return_logits, **kwargs)
        index = int(context.root_positions[0]) - len(_PROMPT)
        assert int(context.root_tokens[0]) == expected[index]
        tokens = []
        for depth in range(int(candidate_budget)):
            truth = expected[index + 1 + depth] if index + 1 + depth < len(expected) else 0
            tokens.append((truth + 1) % vocab if depth == accepted else truth)
        forced.append(tuple(tokens))
        return compile_mtp_chain(
            [MtpDraftRequest(
                request_id=int(context.request_ids[0]),
                root_position=int(context.root_positions[0]),
                candidate_tokens=tuple(tokens),
                active_count=len(tokens),
            )],
            candidate_budget=int(candidate_budget),
            pad_token_id=provider.pad_token_id,
        )

    monkeypatch.setattr(provider, "propose", propose)
    return forced


@pytest.mark.parametrize(
    "budget,accepted", [(b, k) for b in (1, 2, 3) for k in range(b + 1)]
)
def test_forced_prefix_repairs_target_state(dense_mtp, monkeypatch, budget, accepted) -> None:
    from scripts.mtp_budget_diagnostics import MtpBudgetSequencePolicy

    decoder, provider, expected, vocab = dense_mtp
    forced = _force_prefix(monkeypatch, provider, expected, vocab, accepted)
    policy = MtpBudgetSequencePolicy((budget,))
    result = decoder.generate(
        _PROMPT, max_new_tokens=_NEW_TOKENS, use_bulk_prefill=True, budget_policy=policy
    )

    assert tuple(result.token_ids) == expected
    assert forced, "the forced host proposal never ran"
    cycles = policy.summary()["results"]
    assert cycles
    # A cycle's chain reaches past the reference only at the request tail, where
    # the padded draft is wrong by construction; every other cycle must accept
    # exactly the forced prefix.
    produced = 1
    for cycle in cycles:
        reachable = min(accepted, cycle["budget"], len(expected) - produced - 1)
        assert cycle["accepted_count"] == max(reachable, 0), (produced, cycles)
        produced += cycle["visible_tokens"]


def test_commit_fault_cleans_up_and_the_decoder_is_reusable(dense_mtp, monkeypatch) -> None:
    decoder, provider, expected, vocab = dense_mtp
    _force_prefix(monkeypatch, provider, expected, vocab, accepted=1)
    original = decoder.verifier.commit
    calls = {"count": 0}

    def commit(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("injected target commit fault")
        return original(*args, **kwargs)

    monkeypatch.setattr(decoder.verifier, "commit", commit)
    with pytest.raises(Exception, match="injected target commit fault"):
        decoder.generate(_PROMPT, max_new_tokens=_NEW_TOKENS, use_bulk_prefill=True)
    monkeypatch.setattr(decoder.verifier, "commit", original)

    result = decoder.generate(_PROMPT, max_new_tokens=_NEW_TOKENS, use_bulk_prefill=True)
    assert tuple(result.token_ids) == expected
