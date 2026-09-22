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
    SPECULATIVE_MTP_INCOMPATIBLE_FIELDS,
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
    "repetition_penalty": _params(temperature=0.7, repetition_penalty=1.1),
    "presence_penalty": _params(temperature=0.7, presence_penalty=0.2),
    "frequency_penalty": _params(temperature=0.7, frequency_penalty=0.2),
    "logit_bias": _params(temperature=0.7, logit_bias={3: 2.0}),
    "suppress_token_ids": _params(temperature=0.7, suppress_token_ids=(4,)),
    "ignore_eos": _params(temperature=0.7, ignore_eos=True),
    "min_tokens": _params(temperature=0.7, min_tokens=2, eos_token_id=9),
    "eos_gate": _params(temperature=0.7, eos_token_id=9),
    "stop_token_ids": _params(temperature=0.7, stop_token_ids=(9,)),
    "stop_token_sequences": _params(temperature=0.7, stop_token_sequences=((1, 2),)),
    "logprobs": _params(temperature=0.7, logprobs=True),
    "top_logprobs": _params(temperature=0.7, top_logprobs=5),
    "forced_tokens": _params(temperature=0.7, forced_tokens_pending=(1,)),
    "force_sequence_completion": _params(
        temperature=0.7, force_sequence_completion_token_sequences=((1, 2),)
    ),
    "greedy_logprobs": _params(logprobs=True),
    "greedy": _params(),
    "eos_only": _params(eos_token_id=9),
}

_UNSERVABLE = {
    # The post-thinking queue is served by the same per-row walk as
    # ``forced_tokens_pending``, but it can only be non-empty alongside a
    # thinking budget, and the budget is still refused below.
    "post_thinking_forced": _params(
        temperature=0.7,
        thinking_close_token_ids=(1,),
        thinking_hard_token_cap=4,
        post_thinking_forced_tokens_pending=(2,),
    ),
    "json_object_close": _params(temperature=0.7, json_object_close_forcing=True),
    "tool_call_constraint": _params(
        temperature=0.7, tool_call_constraint={"tool_names": ("read",)}
    ),
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
    # The servable set is the sampling law, the finish-rule relaxations, and the
    # two metadata fields; the refused set is the hook family. Every blocker in
    # the union must be one the sampler or the cycle commit can name, so a new
    # field cannot be served or refused by accident.
    assert "temperature" in SAMPLED_MTP_SERVABLE_BLOCKERS
    assert "logprobs" in SAMPLED_MTP_SERVABLE_BLOCKERS
    assert "top_logprobs" in SAMPLED_MTP_SERVABLE_BLOCKERS
    assert "thinking_budget" in SAMPLED_MTP_UNSERVABLE_BLOCKERS


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
        sampling_request=_params(temperature=0.7, logit_bias={3: 1.0}),
        request=_params(temperature=0.7, logit_bias={3: 1.0}),
        sampling_state=None,
    )
    adapter = _adapter(plugin_evidence=(_evidence_row(),), row=row)
    assert adapter._sampled_route_request(1) is True
    # An EOS finish policy is served by the cycle commit's finish rule, so a
    # qualified row carrying one stays on the sampled route.
    eos_row = SimpleNamespace(
        sampling_request=_params(temperature=0.7, eos_token_id=9),
        request=_params(temperature=0.7, eos_token_id=9),
        sampling_state=None,
    )
    assert _adapter(plugin_evidence=(_evidence_row(),), row=eos_row)._sampled_route_request(1) is True
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
    # Selection-preserving settings stay on the greedy route: the truncation
    # filters and the stream seed cannot change a raw argmax.
    assert (
        _speculative_sampling_mode(
            runner, 1, _params(top_p=0.1, top_k=4, min_p=0.5, seed=7)
        )
        == "greedy"
    )
    # A metadata-only request is served by the sampled route: its law is the
    # processed distribution the accept already builds per row, and the reported
    # value comes from the same row the autoregressive route would read.
    assert _speculative_sampling_mode(runner, 1, _params(logprobs=True)) == "sampled"
    # Both are finish-rule fields and both are served now: the cycle commit
    # applies EOS, stop ids, stop sequences, and the min-token EOS floor to the
    # whole verified chain and selects its terminal prefix, so a stochastic
    # accept that lands EOS or a stop mid-cycle publishes nothing after it.
    assert (
        _speculative_sampling_mode(runner, 1, _params(temperature=0.7, ignore_eos=True))
        == "sampled"
    )
    assert (
        _speculative_sampling_mode(runner, 1, _params(temperature=0.7, eos_token_id=9))
        == "sampled"
    )
    assert (
        _speculative_sampling_mode(
            runner, 1, _params(temperature=0.7, eos_token_id=9, ignore_eos=True)
        )
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

    # A device-sampler row uses target draws from the shared native sampler.
    device_row = SimpleNamespace(
        native_greedy=False,
        native_sampler=True,
        sampling_request=params,
        request=params,
        sampling_state=None,
    )
    assert _adapter(plugin_evidence=(_evidence_row(),), row=device_row)._sampled_route_request(1) is True
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
    # A hook-family request is what the sampled route still refuses, so it is the
    # case that must stay at K0 even when the row lists the sampled mode.
    unservable = _params(
        temperature=0.7, tool_call_constraint={"tool_names": ("read",)}
    )
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
    # An artifact whose row does not list the sampled mode still ends at K0
    # before provider mutation.
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


def test_sampled_policy_names_the_concurrent_native_routes() -> None:
    """Automatic sampled intent must not change the old greedy C2 policy."""

    from hipengine.models import qwen35

    tables = {
        name: value
        for name, value in vars(qwen35).items()
        if name.endswith("_MTP_SERVING_EVIDENCE")
    }
    assert tables, "the model plugin's serving evidence tables must be importable"
    advertising = [
        (name, row.evidence_key)
        for name, table in tables.items()
        for row in table
        if "sampled" in tuple(row.sampling_modes)
    ]
    assert advertising == [
        (
            "_QWEN38_Q4KM_MTP_SERVING_EVIDENCE",
            f"qwen38-q4km-gfx1151-native-sampled-c{width}-k3",
        )
        for width in (1, 2, 3, 4)
    ], advertising


# ------------------------------------------------------------- guard totality

# Request fields that cannot change token selection or post-accept finish
# behavior on the admitted route.  Every other field of the request vocabulary
# is an advertised MTP blocker; the two sets must partition the vocabulary so a
# new sampler field cannot be admitted by omission.
_SELECTION_PRESERVING_FIELDS = {
    # Inert while temperature <= 0: the autoregressive route selects the raw
    # argmax, so a truncation filter never removes the selected token.
    "top_k": "truncation filters are inert on the raw-argmax route",
    "top_p": "truncation filters are inert on the raw-argmax route",
    "min_p": "truncation filters are inert on the raw-argmax route",
    # Only the sampler stream differs; the selected token is still the argmax.
    "seed": "selects the sampler stream, not the greedy decision",
    # Reasons label a queue that is itself a blocker, so an empty queue carries
    # no behavior of its own.
    "forced_token_reason": "labels the served forced queue",
    "post_thinking_forced_token_reason": "labels a queue a thinking budget gates",
    "force_sequence_completion_reason": "labels the served forced queue",
    # The thinking budget is active only as a pair; a partial configuration
    # builds no budget state and enforces nothing.
    "thinking_close_token_ids": "inert without thinking_hard_token_cap",
    "thinking_hard_token_cap": "inert without thinking_close_token_ids",
    "thinking_soft_close_window": "inert without the active budget pair",
}


def test_sampled_servable_set_is_the_sampling_law_and_the_finish_rule() -> None:
    """The sampled route reproduces the sampler law, the finish rule, and the queue.

    ``hipengine/speculative/sampling.py`` requires the caller to apply the
    request's pipeline (bias, penalties, suppression, temperature, top-k,
    top-p, min-p) before the coupled accept, and the induced-law gate measures
    exactly that set.  ``ignore_eos`` is a finish-rule relaxation the cycle
    commit honors.  The four finish-rule fields are served by
    ``limit_chain_accept_finish`` in ``hipengine/speculative/streaming.py``,
    which applies EOS, stop token ids, multi-token stop sequences, and the
    min-token EOS floor to the whole verified chain and selects its terminal
    prefix, so they no longer have to be advertised blockers. The two metadata
    fields are served by ``reported_logprob``, which reads the logits row that
    predicted each published token with the same per-branch rule
    ``select_token`` reports with. The three forced-token fields are one queue:
    the row that predicts a position with a pending forced token gets a point
    mass on it instead of the sampled law, and the live queue is popped only for
    the tokens a cycle published.
    """

    finish_rule_fields = {
        "min_tokens",
        "eos_token_id",
        "stop_token_ids",
        "stop_token_sequences",
    }
    metadata_fields = {"logprobs", "top_logprobs"}
    forced_queue_fields = {
        "forced_tokens_pending",
        "post_thinking_forced_tokens_pending",
        "force_sequence_completion_token_sequences",
    }
    assert set(SAMPLED_MTP_SERVABLE_BLOCKERS) == {
        "temperature",
        "logit_bias",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "suppress_token_ids",
        "ignore_eos",
        *finish_rule_fields,
        *metadata_fields,
        *forced_queue_fields,
    }
    # The two sets stay disjoint and still partition every incompatible field, so
    # a field can never be silently in neither.
    assert finish_rule_fields <= set(SAMPLED_MTP_SERVABLE_BLOCKERS)
    assert not finish_rule_fields & set(SAMPLED_MTP_UNSERVABLE_BLOCKERS)
    assert metadata_fields <= set(SAMPLED_MTP_SERVABLE_BLOCKERS)
    assert not metadata_fields & set(SAMPLED_MTP_UNSERVABLE_BLOCKERS)
    assert forced_queue_fields <= set(SAMPLED_MTP_SERVABLE_BLOCKERS)
    assert not forced_queue_fields & set(SAMPLED_MTP_UNSERVABLE_BLOCKERS)
    assert set(SAMPLED_MTP_SERVABLE_BLOCKERS) | set(SAMPLED_MTP_UNSERVABLE_BLOCKERS) == set(
        SPECULATIVE_MTP_INCOMPATIBLE_FIELDS
    )


@pytest.mark.parametrize(
    "name",
    ["min_tokens", "eos_gate", "stop_token_ids", "stop_token_sequences"],
)
def test_finish_rule_fields_serve_on_the_sampled_route(name: str) -> None:
    """A temperature request that stops on anything now speculates.

    The cycle commit applies the autoregressive finish rule to the whole verified
    chain (``limit_chain_accept_finish``), so these four no longer have to fall
    back to AR to get the right finish.
    """

    params = {
        "min_tokens": _params(temperature=0.7, min_tokens=2, eos_token_id=9),
        "eos_gate": _params(temperature=0.7, eos_token_id=9),
        "stop_token_ids": _params(temperature=0.7, stop_token_ids=(9,)),
        "stop_token_sequences": _params(temperature=0.7, stop_token_sequences=((1, 2),)),
    }[name]
    assert supports_sampled_speculative_mtp(params) is True
    assert sampled_speculative_mtp_blockers(params) == ()
    assert speculative_serving_sampling_mode(params) == "sampled"


def test_request_vocabulary_is_fully_classified() -> None:
    """Every request field is an advertised blocker or selection-preserving."""

    import dataclasses

    from hipengine.generation.batch_scheduler import PerRowSamplingParams

    vocabulary = {field.name for field in dataclasses.fields(PerRowSamplingParams)}
    blockers = set(SPECULATIVE_MTP_INCOMPATIBLE_FIELDS)
    preserving = set(_SELECTION_PRESERVING_FIELDS)
    # The guard names two blockers by their request-side alias.
    aliases = {
        "suppress_tokens": "suppress_token_ids",
        "stop_tokens": "stop_token_ids",
    }
    classified = {aliases.get(name, name) for name in vocabulary}
    unclassified = classified - blockers - preserving
    assert not unclassified, sorted(unclassified)
    assert not (blockers & preserving)
    assert preserving <= set(vocabulary)
    assert all(reason for reason in _SELECTION_PRESERVING_FIELDS.values())
