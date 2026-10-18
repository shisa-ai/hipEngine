"""A request's thinking policy is part of what makes a batch compatible.

``_mtp_dispatch_sampling`` reads the policy that travels on the queue item, so
two requests that differ only in policy must not share one dispatch:

- the batcher's group key ignored the policy, so a hint request and a hard
  request with identical sampling and plan fingerprint coalesced, and the whole
  group took ``group[0]``'s policy -- hint-first cleared both budgets, hard-first
  enforced both;
- the streaming path never called the helper at all, so an MTP hint stream kept
  the budget the blocking path clears.

These tests drive the real batcher with a recording fake: the policy has to
survive both the coalescing decision and the streaming dispatch.
"""

from __future__ import annotations

import asyncio

import pytest

from hipengine.llm import SamplingParams
from hipengine.server.api import _GenerationBatcher
from tests.test_integration_server_api import SpeculativeMTPFakeLLM

_BUDGET = {"thinking_close_token_ids": (7,), "thinking_hard_token_cap": 8}


def _budgeted(**overrides) -> SamplingParams:
    values = {"max_tokens": 4, "temperature": 0.0, **_BUDGET}
    values.update(overrides)
    return SamplingParams(**values)


def _batcher(fake) -> _GenerationBatcher:
    return _GenerationBatcher(
        engine_factory=lambda: fake,
        batch_window_seconds=0.0,
        mtp_circuit_breaker=None,
    )


def _mtp_samplings(fake) -> list[SamplingParams]:
    return [call[1] for call in fake.mtp_calls]


@pytest.mark.parametrize("first", ["hint", "hard"])
def test_mixed_policy_requests_do_not_coalesce_into_one_dispatch(first: str) -> None:
    """Both arrival orders: the group key has to separate the two policies."""

    async def run() -> None:
        fake = SpeculativeMTPFakeLLM()
        batcher = _batcher(fake)
        policies = [first, "hard" if first == "hint" else "hint"]
        results = await asyncio.gather(
            *(
                batcher.submit(
                    (f"prompt-{policy}",),
                    _budgeted(),
                    route="speculative_mtp",
                    thinking_policy=policy,
                )
                for policy in policies
            )
        )
        assert len(results) == 2
        calls = fake.mtp_calls
        assert len(calls) == 2, {
            "mixed_policy_requests_coalesced": {
                "arrival_order": policies,
                "dispatch_rows": [len(call[0]) for call in calls],
            }
        }
        by_prompt = {
            str(prompt): sampling
            for prompts, sampling in calls
            for prompt in prompts
        }
        # The hard request keeps its budget; the hint request is relaxed.
        assert by_prompt["prompt-hard"].thinking_hard_token_cap == 8
        assert by_prompt["prompt-hint"].thinking_hard_token_cap is None
        assert by_prompt["prompt-hint"].thinking_close_token_ids == ()

    asyncio.run(run())


def test_same_policy_requests_still_coalesce() -> None:
    """The key change must not split a genuinely compatible group."""

    async def run() -> None:
        fake = SpeculativeMTPFakeLLM()
        batcher = _batcher(fake)
        await asyncio.gather(
            *(
                batcher.submit(
                    (f"prompt-{index}",),
                    _budgeted(),
                    route="speculative_mtp",
                    thinking_policy="hard",
                )
                for index in range(2)
            )
        )
        assert len(fake.mtp_calls) == 1, {
            "compatible_requests_stopped_coalescing": {
                "dispatch_rows": [len(call[0]) for call in fake.mtp_calls]
            }
        }
        assert len(fake.mtp_calls[0][0]) == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    "policy,expected_cap",
    [("hint", None), ("hard", 8)],
)
def test_streaming_mtp_applies_the_same_policy_as_blocking(
    policy: str, expected_cap: int | None
) -> None:
    """SSE must not disagree with the blocking path about the budget."""

    async def run() -> None:
        fake = SpeculativeMTPFakeLLM()
        batcher = _batcher(fake)
        chunks = [
            chunk
            async for chunk in batcher.stream(
                ("prompt",),
                _budgeted(),
                route="speculative_mtp",
                thinking_policy=policy,
            )
        ]
        assert chunks
        samplings = _mtp_samplings(fake)
        assert len(samplings) == 1, "the stream must take the MTP route"
        assert samplings[0].thinking_hard_token_cap == expected_cap
        if policy == "hint":
            assert samplings[0].thinking_close_token_ids == ()

    asyncio.run(run())


@pytest.mark.parametrize(
    "policy,expected_cap",
    [("hint", None), ("hard", 8)],
)
def test_blocking_and_streaming_agree_on_the_budget(
    policy: str, expected_cap: int | None
) -> None:
    """One policy, both transports, one answer about enforcement."""

    async def run() -> None:
        blocking = SpeculativeMTPFakeLLM()
        await _batcher(blocking).submit(
            ("prompt",),
            _budgeted(),
            route="speculative_mtp",
            thinking_policy=policy,
        )
        streaming = SpeculativeMTPFakeLLM()
        async for _ in _batcher(streaming).stream(
            ("prompt",),
            _budgeted(),
            route="speculative_mtp",
            thinking_policy=policy,
        ):
            pass
        blocking_cap = _mtp_samplings(blocking)[0].thinking_hard_token_cap
        streaming_cap = _mtp_samplings(streaming)[0].thinking_hard_token_cap
        assert blocking_cap == streaming_cap == expected_cap

    asyncio.run(run())
