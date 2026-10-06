"""The MTP batch route must not silently drop a request's thinking budget.

``_mtp_dispatch_sampling`` decides what the dispatched batch carries, and the
policy decides which form of the request the route is meant to serve:

- ``hint`` asks for the raw-argmax-exact form, so the sampler-level budget fields
  are cleared while the rendered prompt keeps its thinking hints;
- ``hard`` asks for the budget, which the sampled route applies per row.

The dispatch used to relax unconditionally. That was invisible while the hard
policy refused the route, and it served a live ``--speculative-mtp-thinking
hard`` request with its budget silently dropped once the sampled route began
serving one. A "sampled mode" condition cannot separate the two policies -- a
hint request is sampled too once the route is chosen -- so the policy itself is
what the dispatch reads.
"""

from __future__ import annotations

import pytest

from hipengine.generation.sampling import (
    relax_thinking_budget_for_mtp,
    speculative_serving_sampling_mode,
)
from hipengine.llm import SamplingParams
from hipengine.server.api import _mtp_dispatch_sampling


def _budget(**overrides) -> SamplingParams:
    values = {
        "temperature": 0.0,
        "thinking_close_token_ids": (7,),
        "thinking_hard_token_cap": 8,
    }
    values.update(overrides)
    return SamplingParams(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"temperature": 0.7},
        {"eos_token_id": 2},
        {"logprobs": True, "top_logprobs": 3},
        {"stop_token_ids": (5,)},
        {"temperature": 0.7, "repetition_penalty": 1.2},
    ],
)
def test_the_hard_policy_keeps_the_budget_in_every_servable_shape(overrides) -> None:
    params = _budget(**overrides)
    dispatched = _mtp_dispatch_sampling(params, thinking_policy="hard")
    assert dispatched is params
    assert dispatched.thinking_hard_token_cap == 8
    assert dispatched.thinking_close_token_ids == (7,)


def test_the_hint_policy_still_relaxes_the_budget() -> None:
    params = _budget()
    dispatched = _mtp_dispatch_sampling(params, thinking_policy="hint")
    assert dispatched is not params
    assert dispatched == relax_thinking_budget_for_mtp(params)
    assert dispatched.thinking_hard_token_cap is None
    assert dispatched.thinking_close_token_ids == ()
    # The request still samples through the route the hint policy chose for it.
    assert speculative_serving_sampling_mode(params) == "sampled"


def test_an_unset_policy_keeps_the_previous_relaxation() -> None:
    params = _budget()
    assert _mtp_dispatch_sampling(params) == relax_thinking_budget_for_mtp(params)


def test_a_request_without_enforcement_is_returned_unchanged() -> None:
    params = SamplingParams(temperature=0.7)
    assert _mtp_dispatch_sampling(params, thinking_policy="hard") is params
    assert _mtp_dispatch_sampling(params, thinking_policy="hint") is params
