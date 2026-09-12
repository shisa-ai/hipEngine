"""Request/resource contract tests for the Surya OCR generators.

Pure unit tests over ``hipengine.generation.surya_contract`` — no model
checkpoint, GPU, or torch required.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.deadline import GenerationDeadlineExceeded
from hipengine.generation.registry import GenerationRequest
from hipengine.generation.surya_contract import (
    SuryaGreedySettings,
    SuryaRequestError,
    check_prompt_capacity,
    greedy_decode_tokens,
    resolve_surya_greedy_settings,
)


def _spec() -> SimpleNamespace:
    return SimpleNamespace(eos_token_id=2)


def _request(**overrides) -> SimpleNamespace:
    base = dict(
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        logit_bias=(),
        suppress_token_ids=(),
        min_tokens=0,
        eos_token_id=None,
        stop_token_ids=(),
        stop_token_sequences=(),
        forced_tokens_pending=(),
        post_thinking_forced_tokens_pending=(),
        force_sequence_completion_token_sequences=(),
        thinking_close_token_ids=(),
        thinking_hard_token_cap=None,
        thinking_soft_close_window=0,
        json_object_close_forcing=False,
        tool_call_constraint=None,
        grammar=None,
        logprobs=False,
        top_logprobs=0,
        kv_storage="auto",
        kv_scale_dtype="fp16",
        kv_scale_granularity="per_token_head",
        ignore_eos=False,
        deadline_at=None,
        cancellation_token=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_defaults_are_accepted() -> None:
    settings = resolve_surya_greedy_settings(_request(), _spec())
    assert settings.max_tokens == 4
    assert settings.ignore_eos is False
    assert settings.eos_token_ids == frozenset({2})
    assert settings.stop_token_ids == frozenset()


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0.7),
        ("top_p", 0.9),
        ("top_k", 40),
        ("min_p", 0.1),
        ("repetition_penalty", 1.1),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
        ("logit_bias", ((1, 2.0),)),
        ("suppress_token_ids", (5,)),
        ("min_tokens", 3),
        ("grammar", {"type": "json"}),
        ("json_object_close_forcing", True),
        ("logprobs", True),
        ("kv_storage", "quantized"),
    ],
)
def test_unsupported_controls_are_rejected(field: str, value: object) -> None:
    with pytest.raises(SuryaRequestError, match=field):
        resolve_surya_greedy_settings(_request(**{field: value}), _spec())


def test_supported_controls_are_honored() -> None:
    settings = resolve_surya_greedy_settings(
        _request(ignore_eos=True, eos_token_id=7, stop_token_ids=(9, 11)),
        _spec(),
    )
    assert settings.ignore_eos is True
    assert settings.eos_token_ids == frozenset({7})
    assert settings.stop_token_ids == frozenset({9, 11})


def test_generation_request_roundtrip() -> None:
    request = GenerationRequest(
        prompts=("hi",),
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )
    settings = resolve_surya_greedy_settings(request, _spec())
    assert settings.max_tokens == 8 and settings.ignore_eos is True


def test_capacity_validation() -> None:
    check_prompt_capacity(10, 4, 16)  # exactly fits
    with pytest.raises(SuryaRequestError, match="exceeds"):
        check_prompt_capacity(10, 7, 16)
    with pytest.raises(SuryaRequestError, match="no tokens"):
        check_prompt_capacity(0, 1, 16)


# ---------------------------------------------------------------------------
# budget contract: the registered HIP generator must honor what it advertises
# ---------------------------------------------------------------------------


_MIB = 1024 * 1024


def test_vision_tile_plan_stays_inside_the_budget() -> None:
    """The vision score tile must never exceed the configured memory budget.

    The dense path allocated ``heads * n^2 * 4`` bytes: 56.5 GB for a 300-DPI
    A4 page (220x156 grid, 34320 patches) and 206 GB at the checkpoint's
    ``max_pixels`` ceiling (256x256, 65536 patches). Tiling by query rows makes
    the live tile ``heads * n * block * 4`` instead, and this pins the bound.
    """

    from hipengine.runtime.surya import plan_vision_attention

    heads = 12
    budget = 512 * _MIB
    for n in (256, 4096, 34320, 65536):
        block, scratch = plan_vision_attention(n, heads, budget)
        assert 1 <= block <= n, (n, block)
        assert scratch == heads * n * block * 4, (n, block, scratch)
        assert scratch <= budget, f"grid of {n} patches needs {scratch} > {budget}"

    # the two grids the dense path could not run are now well inside a
    # half-gigabyte tile budget
    assert plan_vision_attention(34320, heads, budget)[1] <= budget
    assert plan_vision_attention(65536, heads, budget)[1] <= budget


def test_vision_tile_plan_uses_one_tile_when_it_fits() -> None:
    """Small grids must keep the single dense tile, not be split for nothing."""

    from hipengine.runtime.surya import plan_vision_attention

    heads = 12
    # 4096 patches x 12 heads x 4 B = 196 KiB per query row; the whole grid is
    # 805 MiB, so a 1 GiB budget keeps it in one tile and a 512 MiB budget must
    # split it
    dense = heads * 4096 * 4096 * 4
    block, scratch = plan_vision_attention(4096, heads, 1024 * _MIB)
    assert block == 4096
    assert scratch == dense

    block, scratch = plan_vision_attention(4096, heads, 512 * _MIB)
    assert 1 <= block < 4096
    assert scratch == heads * 4096 * block * 4 <= 512 * _MIB


def test_vision_tile_plan_honors_a_disabled_budget() -> None:
    from hipengine.runtime.surya import plan_vision_attention

    block, scratch = plan_vision_attention(4096, 12, None)
    assert block == 4096
    assert scratch == 12 * 4096 * 4096 * 4


def test_vision_tile_plan_shrinks_to_one_row_under_a_tiny_budget() -> None:
    from hipengine.runtime.surya import plan_vision_attention

    heads, n = 12, 4096
    block, scratch = plan_vision_attention(n, heads, heads * n * 4)
    assert block == 1
    assert scratch == heads * n * 4
    # below a single row there is nothing left to give: the plan still returns
    # a valid tile and the caller's admission check is what rejects the grid
    block, scratch = plan_vision_attention(n, heads, 1)
    assert block == 1
    assert scratch == heads * n * 4


def test_gpu_factory_declares_the_budgets_it_honors() -> None:
    """The registered factory must not swallow capacity configuration.

    ``LLM._factory_capacity_kwargs`` forwards a limit only to a factory that
    declares the parameter by name. ``make_surya_generator_gpu`` used to take
    ``**_kwargs``, so ``LLM(max_sequence_length=...)`` was silently dropped and
    the runner stayed at its 2048-token default — below the 8580 image tokens
    a 300-DPI A4 page needs.
    """

    import inspect

    from hipengine.generation.surya_gpu import make_surya_generator_gpu
    from hipengine.llm import _factory_capacity_kwargs

    parameters = inspect.signature(make_surya_generator_gpu).parameters
    assert "max_sequence_length" in parameters
    assert "vision_max_scratch_bytes" in parameters

    forwarded = _factory_capacity_kwargs(
        make_surya_generator_gpu,
        max_sequence_length=16384,
        resident_capacity=None,
        vision_max_scratch_bytes=256 * _MIB,
    )
    assert forwarded["max_sequence_length"] == 16384
    assert forwarded["vision_max_scratch_bytes"] == 256 * _MIB


def _settings(**overrides) -> SuryaGreedySettings:
    base = dict(
        max_tokens=4,
        ignore_eos=False,
        eos_token_ids=frozenset({2}),
        stop_token_ids=frozenset(),
    )
    base.update(overrides)
    return SuryaGreedySettings(**base)


def test_greedy_stops_at_eos_without_appending() -> None:
    calls: list[int] = []

    def step_fn(token_id: int, step: int) -> np.ndarray:
        calls.append(token_id)
        return np.array([0.0, 0.0, 5.0, 0.0])  # token 2 = eos

    ids, reason = greedy_decode_tokens(
        np.array([0.0, 0.0, 5.0, 0.0]), _settings(), step_fn, _request()
    )
    assert ids == []
    assert reason == "eos"
    assert calls == []


def test_ignore_eos_continues_past_eos() -> None:
    def step_fn(token_id: int, step: int) -> np.ndarray:
        return np.array([0.0, 0.0, 5.0, 0.0])

    ids, reason = greedy_decode_tokens(
        np.array([0.0, 0.0, 5.0, 0.0]),
        _settings(ignore_eos=True, max_tokens=3),
        step_fn,
        _request(ignore_eos=True),
    )
    assert ids == [2, 2, 2]
    assert reason == "length"


def test_stop_token_ids_finish_reason() -> None:
    def step_fn(token_id: int, step: int) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 5.0])  # token 3

    ids, reason = greedy_decode_tokens(
        np.array([0.0, 0.0, 0.0, 5.0]),
        _settings(stop_token_ids=frozenset({3})),
        step_fn,
        _request(),
    )
    assert ids == []
    assert reason == "stop"


def test_no_trailing_decode_after_final_token() -> None:
    """The last requested token must not trigger an extra decode step."""

    calls: list[int] = []

    def step_fn(token_id: int, step: int) -> np.ndarray:
        calls.append(token_id)
        return np.array([0.0, 5.0, 0.0, 0.0])  # token 1

    ids, reason = greedy_decode_tokens(
        np.array([0.0, 5.0, 0.0, 0.0]), _settings(max_tokens=2), step_fn, _request()
    )
    assert ids == [1, 1]
    assert reason == "length"
    # one step between the two tokens, none after the second
    assert calls == [1]


def test_deadline_is_checked() -> None:
    def step_fn(token_id: int, step: int) -> np.ndarray:
        return np.array([0.0, 5.0, 0.0, 0.0])

    request = _request(deadline_at=0.0)  # monotonic clock is always past 0
    with pytest.raises(GenerationDeadlineExceeded):
        greedy_decode_tokens(
            np.array([0.0, 5.0, 0.0, 0.0]), _settings(max_tokens=3), step_fn, request
        )


def test_zero_max_tokens_returns_empty() -> None:
    def step_fn(token_id: int, step: int) -> np.ndarray:  # pragma: no cover
        raise AssertionError("must not decode when max_tokens is 0")

    ids, reason = greedy_decode_tokens(
        np.array([0.0, 5.0, 0.0, 0.0]), _settings(max_tokens=0), step_fn, _request()
    )
    assert ids == [] and reason == "length"
