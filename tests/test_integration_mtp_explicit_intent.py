"""Explicit MTP intent survives invalid requests and queue admission."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hipengine.llm import SamplingParams
from hipengine.server.api import (
    OpenAIHTTPError,
    _GenerationBatcher,
    _MTPCircuitBreaker,
    _mtp_unavailable_route_fallback,
    _serving_plan_route_decision,
)
from tests.test_integration_server_api import SpeculativeMTPFakeLLM


@pytest.mark.parametrize("error", [NotImplementedError("control"), ValueError("input")])
def test_client_errors_do_not_open_mtp_breaker(error):
    class Rejecting(SpeculativeMTPFakeLLM):
        def generate_speculative_mtp_detailed(self, prompts, sampling_params):
            if sampling_params.logprobs:
                raise error
            return super().generate_speculative_mtp_detailed(prompts, sampling_params)

    async def run():
        engine = Rejecting()
        breaker = _MTPCircuitBreaker(threshold=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: engine, batch_window_seconds=0,
            mtp_circuit_breaker=breaker,
        )
        for _ in range(3):
            with pytest.raises(type(error)):
                await batcher.submit(("prompt",), SamplingParams(max_tokens=4, logprobs=True), route="speculative_mtp")
        result = await batcher.submit(("prompt",), SamplingParams(max_tokens=4), route="speculative_mtp")
        assert result == ["mtp:prompt"]
        assert engine.calls == []
        assert breaker.snapshot()["failures"] == []

    asyncio.run(run())


@pytest.mark.parametrize("operator", [False, True])
def test_explicit_mtp_cannot_fall_back_when_unavailable(operator):
    engine = SimpleNamespace()
    breaker = _MTPCircuitBreaker(threshold=1)
    if operator:
        breaker.disable_new_mtp()
    else:
        breaker.record_failure(breaker.scope(engine, ("prompt",)), RuntimeError("kernel"))
    with pytest.raises(OpenAIHTTPError) as raised:
        _mtp_unavailable_route_fallback(
            "speculative_mtp", requested_route="speculative_mtp",
            route_decision={"request_mode": "explicit"}, breaker=breaker,
            engine=engine, prompts=("prompt",), group_rows=1, output_horizon_tokens=256,
        )
    assert raised.value.status_code == 503
    assert raised.value.param == "speculative_mtp"
    assert raised.value.extra["hipengine"]["speculative_mtp"]["reason"] == (
        "mtp_operator_rollback" if operator else "mtp_circuit_breaker_open"
    )


def test_automatic_mtp_retains_reported_ar_fallback():
    engine = SimpleNamespace()
    breaker = _MTPCircuitBreaker(threshold=1)
    breaker.record_failure(breaker.scope(engine, ("prompt",)), RuntimeError("kernel"))
    route, decision = _mtp_unavailable_route_fallback(
        "speculative_mtp", requested_route="speculative_mtp_auto",
        route_decision={"request_mode": "automatic"}, breaker=breaker,
        engine=engine, prompts=("prompt",), group_rows=1, output_horizon_tokens=256,
    )
    assert route == "default"
    assert decision["reason"] == "mtp_circuit_breaker_open"


def test_explicit_realized_capability_miss_cannot_become_ar():
    with pytest.raises(OpenAIHTTPError) as raised:
        _serving_plan_route_decision(
            {"admitted": False, "reason": "gemma4_assistant_group_above_offered_width",
             "key": {"realized_group_rows": 4}, "request_mode": "explicit"},
            requested_route="speculative_mtp", group_rows=4, output_horizon_tokens=32,
        )
    assert raised.value.status_code == 501
    assert "group_above_offered_width" in raised.value.message


def test_mtp_queue_respects_declared_single_prompt_provider():
    async def run():
        engine = SpeculativeMTPFakeLLM()
        engine.server_mtp_batch_max_active_requests = 1
        batcher = _GenerationBatcher(engine_factory=lambda: engine, batch_window_seconds=0, max_active_requests=4)
        results = await asyncio.gather(*(
            batcher.submit((f"p{i}",), SamplingParams(max_tokens=32), route="speculative_mtp")
            for i in range(4)
        ))
        assert results == [[f"mtp:p{i}"] for i in range(4)]
        assert len(engine.mtp_calls) == 4
        assert engine.calls == []
    asyncio.run(run())
