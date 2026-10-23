"""PARO MTP serving-route contracts: explicit force semantics per scope.

The packed-PARO model plugin declares its C1 chain implementation scope so an
explicit ``speculative_mtp`` request resolves through the OpenAI route instead
of failing with no typed plan, and every axis outside the declaration is
refused by name. Sampling is deliberately absent from the declaration until a
sampled decider is real: an explicit sampled request is refused with
``mtp_sampling_unsupported`` rather than silently downgraded to greedy.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.generation.qwen35_paro import Qwen35ParoOneTokenGenerator
from hipengine.models.qwen35 import Qwen35ParoMoeModel
from hipengine.speculative.serving import SpeculativeMTPServingKey


def _key(**overrides) -> SpeculativeMTPServingKey:
    fields = dict(
        artifact_sha256=None,
        artifact_size_bytes=None,
        content_verified=False,
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="w4_paro",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=1,
        sampling_mode="greedy_fast",
        memory_fit=True,
    )
    fields.update(overrides)
    return SpeculativeMTPServingKey(**fields)


def _plugin() -> Qwen35ParoMoeModel:
    return Qwen35ParoMoeModel()


def _generator(
    *,
    backend: str = "hip_gfx1100",
    target_arch: str = "gfx1100",
) -> Qwen35ParoOneTokenGenerator:
    generator = object.__new__(Qwen35ParoOneTokenGenerator)
    generator.model_plugin = _plugin()
    generator.backend = backend
    generator._resident_model_runner = SimpleNamespace(
        generator=SimpleNamespace(backend=backend, target_arch=target_arch),
        capacity=8,
    )
    return generator


@pytest.mark.parametrize("index,arch", [(0, "gfx1100"), (1, "gfx1151")])
def test_paro_declares_its_c1_chain_implementation_scope(index, arch) -> None:
    declarations = _plugin().speculative_mtp_serving_implementations

    assert len(declarations) == 2
    declaration = declarations[index]
    assert declaration.name == f"paro_w4_{arch}_c1_native_chain"
    assert declaration.backends == ((f"hip_{arch}", arch),)
    assert declaration.max_candidate_count == 1
    assert declaration.max_group_rows == 1
    # Greedy only: a sampled decider does not exist yet, so the declaration
    # must not advertise a mode the runtime would silently downgrade.
    assert declaration.sampling_modes == ("greedy_fast",)
    assert declaration.automatic_eligible is True


@pytest.mark.parametrize("backend,arch", [("hip_gfx1100", "gfx1100"), ("hip_gfx1151", "gfx1151")])
@pytest.mark.parametrize("request_mode", ["explicit", "automatic"])
def test_an_explicit_or_automatic_greedy_c1_request_resolves(request_mode: str, backend: str, arch: str) -> None:
    decision = _plugin().resolve_speculative_mtp_serving_plan(
        key=_key(backend=backend, target_arch=arch), request_mode=request_mode
    )

    assert decision.admitted
    payload = decision.as_dict()
    assert payload["admission_basis"] == "implementation"
    assert payload["implementation_key"] == f"paro_w4_{arch}_c1_native_chain"


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1151"])
def test_a_sampled_request_is_refused_by_name_not_downgraded_to_greedy(arch) -> None:
    decision = _plugin().resolve_speculative_mtp_serving_plan(
        key=_key(backend=f"hip_{arch}", target_arch=arch, sampling_mode="sampled"),
        request_mode="explicit",
    )

    assert not decision.admitted
    assert decision.as_dict()["reason"] == "mtp_sampling_unsupported"


def test_an_unimplemented_backend_is_refused_by_name() -> None:
    decision = _plugin().resolve_speculative_mtp_serving_plan(
        key=_key(backend="cuda_sm86", target_arch="sm86"),
        request_mode="explicit",
    )

    assert not decision.admitted
    assert decision.as_dict()["reason"] == "mtp_backend_unsupported"


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1151"])
def test_a_wider_group_is_refused_by_the_named_width_reason(arch) -> None:
    decision = _plugin().resolve_speculative_mtp_serving_plan(
        key=_key(backend=f"hip_{arch}", target_arch=arch, realized_group_rows=2),
        request_mode="explicit",
    )

    assert not decision.admitted
    assert decision.as_dict()["reason"] == "paro_group_above_offered_width"


def test_the_generator_exposes_the_plugin_resolver_to_the_llm_route() -> None:
    generator = _generator()

    assert Qwen35ParoOneTokenGenerator.speculative_candidate_budget_default == 1
    decision = generator.resolve_speculative_mtp_serving_plan(
        realized_group_rows=1,
        sampling_mode="greedy_fast",
        request_mode="explicit",
    )
    assert decision.admitted
    assert (
        decision.as_dict()["implementation_key"] == "paro_w4_gfx1100_c1_native_chain"
    )

    # The refusal path reaches the caller by name too, so the server maps it
    # to a typed 501 instead of a silent AR downgrade.
    refused = generator.resolve_speculative_mtp_serving_plan(
        realized_group_rows=1,
        sampling_mode="sampled",
        request_mode="explicit",
    )
    assert not refused.admitted
    assert refused.as_dict()["reason"] == "mtp_sampling_unsupported"


def test_the_generator_resolves_the_resolved_runner_backend() -> None:
    # The dataclass default backend is "auto"; the resolver must read the
    # resolved backend off the resident runner so a loaded owner resolves
    # against the backend it actually runs on.
    generator = object.__new__(Qwen35ParoOneTokenGenerator)
    generator.model_plugin = _plugin()
    generator.backend = "auto"
    generator._resident_model_runner = SimpleNamespace(
        generator=SimpleNamespace(backend="hip_gfx1100", target_arch="gfx1100"),
        capacity=8,
    )
    decision = generator.resolve_speculative_mtp_serving_plan(
        realized_group_rows=1,
        sampling_mode="greedy_fast",
        request_mode="explicit",
    )
    assert decision.admitted

    unresolvable = _generator(backend="cuda_sm86", target_arch="sm86")
    refused = unresolvable.resolve_speculative_mtp_serving_plan(
        realized_group_rows=1,
        sampling_mode="greedy_fast",
        request_mode="explicit",
    )
    assert not refused.admitted
    assert refused.as_dict()["reason"] == "mtp_backend_unsupported"
