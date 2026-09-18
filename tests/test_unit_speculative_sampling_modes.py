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


def test_engine_loop_selects_the_sampled_mode_for_a_temperature_request() -> None:
    """The planner's branch must resolve, not raise: it names a helper the
    engine loop has to import, and a missing import is invisible until a
    temperature request reaches it."""

    from hipengine.generation.engine_loop import _speculative_sampling_mode

    runner = SimpleNamespace(speculative_eos_supported=lambda request_id: False)
    assert _speculative_sampling_mode(runner, 1, _params(temperature=0.7)) == "sampled"
    assert _speculative_sampling_mode(runner, 1, _params()) == "greedy"
    assert _speculative_sampling_mode(runner, 1, _params(logprobs=True)) == "processed"
    # An EOS gate is servable, and so is ignoring EOS: both are processors the
    # sampled law already applies, so this row takes the sampled route rather
    # than the processed one.
    assert (
        _speculative_sampling_mode(runner, 1, _params(eos_token_id=9, ignore_eos=True))
        == "sampled"
    )
    eos_runner = SimpleNamespace(speculative_eos_supported=lambda request_id: True)
    assert (
        _speculative_sampling_mode(eos_runner, 1, _params(eos_token_id=9))
        == "greedy"
    )


def test_the_server_module_imports_the_serving_mode_vocabulary() -> None:
    """The shipped server entry point imports these names from the package."""

    import hipengine.server.api as server_api

    from hipengine.generation import (
        speculative_serving_sampling_mode,
        supports_sampled_speculative_mtp,
    )

    assert server_api.speculative_serving_sampling_mode is speculative_serving_sampling_mode
    assert callable(supports_sampled_speculative_mtp)


def test_adapter_capability_admits_a_qualified_sampled_row() -> None:
    """The capability's row eligibility is the guard the route rides on.

    A temperature row is not `native_greedy`, which used to end the capability
    before any sampled check ran - the route could never be reached in serving.
    The guard now admits a row exactly when `_sampled_route_request` says so, and
    that answer is itself gated on the evidence row.
    """

    from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter

    params = _params(temperature=0.7)
    row = SimpleNamespace(
        native_greedy=False,
        native_sampler=False,
        sampling_request=params,
        request=params,
        sampling_state=SimpleNamespace(),
    )
    adapter = _adapter(plugin_evidence=(_evidence_row(),), row=row)
    assert adapter._sampled_route_request(1) is True

    # A device-sampler row cannot use the host accept rule.
    device_row = SimpleNamespace(
        native_greedy=False,
        native_sampler=True,
        sampling_request=params,
        request=params,
        sampling_state=None,
    )
    assert _adapter(plugin_evidence=(_evidence_row(),), row=device_row)._sampled_route_request(1) is False
    # No evidence row: the route stays closed even for a temperature row.
    assert _adapter(plugin_evidence=(), row=row)._sampled_route_request(1) is False
    # Greedy rows keep their own route.
    greedy = SimpleNamespace(
        native_greedy=True,
        native_sampler=False,
        sampling_request=_params(),
        request=_params(),
        sampling_state=None,
    )
    assert _adapter(plugin_evidence=(), row=greedy)._sampled_route_request(1) is False
    del Qwen35GGUFMTP2Adapter


def _server_config(**overrides):
    from hipengine.server.api import ServerConfig

    values = {"model": "test-model", "speculative_mtp_serving": "auto"}
    values.update(overrides)
    return ServerConfig(**values)


def _serving_engine(*modes: str):
    """An engine that advertises exactly the sampling modes its rows declare."""

    return SimpleNamespace(
        supports_speculative_mtp=True,
        generate_speculative_mtp_detailed=lambda *args, **kwargs: None,
        speculative_mtp_sampling_modes=tuple(modes),
        speculative_mtp_serving_capability={
            "admitted": True,
            "automatic_eligible": True,
        },
    )


def _serving_request(speculative_mtp=True):
    return SimpleNamespace(speculative_mtp=speculative_mtp)


def test_server_route_keeps_a_sampled_request_only_with_a_sampled_row() -> None:
    """The server's route choice is the sampled route's outermost gate.

    A temperature request carries sampling blockers, and a request with blockers
    used to be sent to K0 before the model-plugin resolver ran, so the route
    could not be reached in serving at all. It now keeps typed intent exactly
    when the artifact's own evidence lists the sampled mode, and the shipped
    table lists it for no artifact.
    """

    from hipengine.server.api import (
        _SPECULATIVE_MTP_AUTO_ROUTE,
        _SPECULATIVE_MTP_BATCH_ROUTE,
        _SPECULATIVE_MTP_K0_ROUTE,
        _speculative_mtp_route_for_request,
    )

    sampled = _params(temperature=0.7)
    unservable = _params(temperature=0.7, logprobs=True)
    greedy = _params()

    def route(engine, sampling, *, explicit=True, mode="auto"):
        return _speculative_mtp_route_for_request(
            _server_config(speculative_mtp_serving=mode),
            _serving_request(explicit),
            engine=engine,
            sampling=sampling,
        )

    # A row listing the sampled mode admits the request to the route.
    assert (
        route(_serving_engine("greedy_fast", "sampled"), sampled)
        == _SPECULATIVE_MTP_BATCH_ROUTE
    )
    # Shipped behavior: no row lists the sampled mode, so a temperature request
    # still ends at K0 before provider mutation.
    assert route(_serving_engine("greedy_fast"), sampled) == _SPECULATIVE_MTP_K0_ROUTE
    # A blocker the sampled route refuses stays K0 even with the row.
    assert (
        route(_serving_engine("greedy_fast", "sampled"), unservable)
        == _SPECULATIVE_MTP_K0_ROUTE
    )
    # Automatic mode selects the automatic route rather than the explicit one.
    assert (
        route(_serving_engine("greedy_fast", "sampled"), sampled, explicit=None)
        == _SPECULATIVE_MTP_AUTO_ROUTE
    )
    # Greedy requests keep their own route with or without the sampled row.
    assert (
        route(_serving_engine("greedy_fast", "sampled"), greedy, explicit=None)
        == _SPECULATIVE_MTP_AUTO_ROUTE
    )
    assert (
        route(_serving_engine("greedy_fast"), greedy, explicit=None)
        == _SPECULATIVE_MTP_AUTO_ROUTE
    )


def test_no_shipped_evidence_row_advertises_the_sampled_mode() -> None:
    """The sampled route's switch is an evidence row, and no shipped row sets it.

    Serving that route was measured and rejected: with the row in place the
    route engages (99% of output tokens from speculative cycles) and decodes at
    0.47x of its in-load true-AR control, because a sampled cycle costs 5.3x the
    same draft chain's greedy cycle. The default therefore stays autoregressive
    for every temperature request until a device-side sampled accept exists. A
    row that re-advertises the mode without that measurement is a regression in
    the shipped default, not a tuning choice.
    """

    from hipengine.models import qwen35

    tables = {
        name: value
        for name, value in vars(qwen35).items()
        if name.endswith("_MTP_SERVING_EVIDENCE")
    }
    assert tables, "the model plugin's serving evidence tables must be importable"
    for name, table in tables.items():
        assert table, name
        for row in table:
            assert "sampled" not in tuple(row.sampling_modes), (name, row.evidence_key)
