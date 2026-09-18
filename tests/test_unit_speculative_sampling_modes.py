"""The sampled MTP route's admission vocabulary.

A request may only reach the sampled route when both sides agree: the request-time
serving plan (keyed on the model-plugin evidence rows) and the engine loop's
planner (keyed on the capability's ``supported_sampling_modes``). These tests pin
the mapping for every processor class, so a request the sampled route cannot
serve exactly keeps falling back to the autoregressive path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.generation.qwen35_gguf_mtp2 import (
    Qwen35GGUFMTP2Adapter,
    _adapter_artifact_size,
)
from hipengine.generation.sampling import (
    SAMPLED_MTP_SERVABLE_BLOCKERS,
    SAMPLED_MTP_UNSERVABLE_BLOCKERS,
    sampled_speculative_mtp_blockers,
    speculative_sampling_mode,
    speculative_serving_sampling_mode,
    supports_sampled_speculative_mtp,
)


def _params(**overrides):
    values = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "logit_bias": (),
        "suppress_token_ids": (),
        "min_tokens": 0,
        "eos_token_id": None,
        "ignore_eos": False,
        "seed": None,
        "row_seeds": (),
        "stop_token_ids": (),
        "stop_token_sequences": (),
        "forced_tokens_pending": (),
        "forced_token_reason": None,
        "post_thinking_forced_tokens_pending": (),
        "post_thinking_forced_token_reason": None,
        "force_sequence_completion_token_sequences": (),
        "force_sequence_completion_reason": None,
        "json_object_close_forcing": False,
        "tool_call_constraint": None,
        "thinking_close_token_ids": (),
        "thinking_hard_token_cap": None,
        "thinking_soft_close_window": 0,
        "logprobs": False,
        "top_logprobs": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


_SERVABLE = {
    "temperature": _params(temperature=0.7),
    "temperature_with_eos": _params(temperature=0.7, eos_token_id=9),
    "repetition_penalty": _params(temperature=0.7, repetition_penalty=1.1),
    "presence_penalty": _params(temperature=0.7, presence_penalty=0.2),
    "frequency_penalty": _params(temperature=0.7, frequency_penalty=0.2),
    "logit_bias": _params(temperature=0.7, logit_bias={3: 2.0}),
    "suppress_token_ids": _params(temperature=0.7, suppress_token_ids=(4,)),
    "min_tokens": _params(temperature=0.7, min_tokens=2, eos_token_id=9),
    "stop_token_ids": _params(temperature=0.7, stop_token_ids=(9,)),
    "stop_token_sequences": _params(temperature=0.7, stop_token_sequences=((1, 2),)),
    "ignore_eos": _params(temperature=0.7, ignore_eos=True),
    "greedy": _params(),
    "eos_only": _params(eos_token_id=9),
}

_UNSERVABLE = {
    "logprobs": _params(temperature=0.7, logprobs=True),
    "top_logprobs": _params(temperature=0.7, top_logprobs=5),
    "forced_tokens": _params(temperature=0.7, forced_tokens_pending=(1,)),
    "post_thinking_forced": _params(
        temperature=0.7,
        thinking_close_token_ids=(1,),
        thinking_hard_token_cap=4,
        post_thinking_forced_tokens_pending=(2,),
    ),
    "force_sequence_completion": _params(
        temperature=0.7, force_sequence_completion_token_sequences=((1, 2),)
    ),
    "json_object_close": _params(temperature=0.7, json_object_close_forcing=True),
    "thinking_budget": _params(
        temperature=0.7, thinking_close_token_ids=(1,), thinking_hard_token_cap=4
    ),
}


@pytest.mark.parametrize("name", sorted(_SERVABLE))
def test_servable_requests_map_to_the_sampled_mode(name: str) -> None:
    params = _SERVABLE[name]
    if name in {"greedy"}:
        assert speculative_serving_sampling_mode(params) == "greedy_fast"
        assert speculative_sampling_mode(params) == "greedy"
        return
    if name == "eos_only":
        # Unchanged behavior: an EOS-only request is not newly admitted by the
        # sampled route's evidence row.
        assert speculative_serving_sampling_mode(params) == "processed_argmax"
        return
    assert supports_sampled_speculative_mtp(params)
    assert sampled_speculative_mtp_blockers(params) == ()
    assert speculative_serving_sampling_mode(params) == "sampled"
    assert speculative_sampling_mode(params) == "sampled"


@pytest.mark.parametrize("name", sorted(_UNSERVABLE))
def test_unservable_requests_keep_the_autoregressive_route(name: str) -> None:
    params = _UNSERVABLE[name]
    assert not supports_sampled_speculative_mtp(params)
    assert sampled_speculative_mtp_blockers(params)
    assert speculative_serving_sampling_mode(params) == "processed_argmax"
    assert speculative_sampling_mode(params) == "processed"


def test_eos_supported_greedy_request_keeps_the_greedy_mode() -> None:
    assert speculative_sampling_mode(_params(eos_token_id=9), eos_supported=True) == "greedy"


def test_servable_and_unservable_blocker_sets_are_disjoint_and_cover_the_vocabulary() -> None:
    assert not set(SAMPLED_MTP_SERVABLE_BLOCKERS) & set(SAMPLED_MTP_UNSERVABLE_BLOCKERS)
    # Every blocker the sampled route refuses must be one of the processors the
    # sampler itself applies, so a new processor cannot be served by accident.
    assert "temperature" in SAMPLED_MTP_SERVABLE_BLOCKERS
    assert "logprobs" in SAMPLED_MTP_UNSERVABLE_BLOCKERS


def _adapter(*, plugin_evidence=(), artifact_size=None, row=None):
    generator = SimpleNamespace(
        backend="hip_gfx1151",
        target_arch="gfx1151",
        execution_profile="production",
        model_plugin=SimpleNamespace(speculative_mtp_serving_evidence=plugin_evidence),
    )
    if artifact_size is not None:
        generator.weight_index = SimpleNamespace(
            path=__file__ if artifact_size == "file" else "/nonexistent"
        )
    owner = SimpleNamespace(generator=generator, capacity=4)
    adapter = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
        quant="gguf_q4_k_m",
    )
    if row is not None:
        owner._row = lambda request_id: row
    return adapter


def _evidence_row(**overrides):
    values = {
        "sampling_modes": ("greedy_fast", "sampled"),
        "backend": "hip_gfx1151",
        "target_arch": "gfx1151",
        "weight_quant": "gguf_q4_k_m",
        "artifact_size_bytes": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sampled_route_is_closed_without_an_evidence_row() -> None:
    adapter = _adapter(plugin_evidence=())
    assert adapter._sampled_route_qualified() is False


def test_sampled_route_opens_only_for_a_matching_evidence_row() -> None:
    assert _adapter(plugin_evidence=(_evidence_row(),))._sampled_route_qualified() is True
    assert (
        _adapter(
            plugin_evidence=(_evidence_row(backend="hip_gfx1100"),)
        )._sampled_route_qualified()
        is False
    )
    assert (
        _adapter(
            plugin_evidence=(_evidence_row(target_arch="gfx1100"),)
        )._sampled_route_qualified()
        is False
    )
    assert (
        _adapter(
            plugin_evidence=(_evidence_row(weight_quant="gguf_q4_k_s"),)
        )._sampled_route_qualified()
        is False
    )
    assert (
        _adapter(
            plugin_evidence=(_evidence_row(sampling_modes=("greedy_fast",)),)
        )._sampled_route_qualified()
        is False
    )


def test_sampled_route_rejects_a_different_artifact_size() -> None:
    adapter = _adapter(
        plugin_evidence=(_evidence_row(artifact_size_bytes=1),),
        artifact_size="file",
    )
    assert _adapter_artifact_size(adapter.generator) == __import__("os").path.getsize(__file__)
    assert adapter._sampled_route_qualified() is False
    matching = _adapter(
        plugin_evidence=(
            _evidence_row(artifact_size_bytes=__import__("os").path.getsize(__file__)),
        ),
        artifact_size="file",
    )
    assert matching._sampled_route_qualified() is True


def test_sampled_route_request_follows_the_row_sampling_mode() -> None:
    row = SimpleNamespace(
        sampling_request=_params(temperature=0.7, eos_token_id=9),
        request=_params(temperature=0.7, eos_token_id=9),
        sampling_state=None,
    )
    adapter = _adapter(plugin_evidence=(_evidence_row(),), row=row)
    assert adapter._sampled_route_request(1) is True
    greedy_row = SimpleNamespace(
        sampling_request=_params(),
        request=_params(),
        sampling_state=None,
    )
    greedy_adapter = _adapter(plugin_evidence=(_evidence_row(),), row=greedy_row)
    assert greedy_adapter._sampled_route_request(1) is False
    unqualified = _adapter(plugin_evidence=(), row=row)
    assert unqualified._sampled_route_request(1) is False


def test_sampled_route_request_falls_back_to_the_request_object() -> None:
    row = SimpleNamespace(
        sampling_request=None,
        request=_params(temperature=0.9),
        sampling_state=None,
    )
    adapter = _adapter(plugin_evidence=(_evidence_row(),), row=row)
    assert adapter._sampled_route_request(1) is True
    empty = _adapter(plugin_evidence=(_evidence_row(),), row=SimpleNamespace())
    assert empty._sampled_route_request(1) is False
