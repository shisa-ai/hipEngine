from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine import LLM, SamplingParams
from hipengine.generation import GenerationOutput, GenerationStreamChunk
from hipengine.speculative.serving import SpeculativeMTPStaticEligibility, SpeculativeMTPStaticState


def _eligibility():
    return SpeculativeMTPStaticEligibility(
        state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
        reason="implemented_native_int8_chain",
        max_candidate_count=3, max_realized_group_rows=1,
        automatic_eligible=True, strict_fallback_key="gguf_target_ar",
        implementation_key="gguf_dense_int8_native_chain",
    )


@pytest.mark.parametrize("stream", [False, True])
def test_library_mtp_resolves_static_eligibility_before_submission(monkeypatch, stream):
    seen = []
    resolved = []
    eligibility = _eligibility()
    generator = SimpleNamespace(
        supports_speculative_mtp=True,
        generate_speculative_mtp_detailed=lambda request: seen.append(request) or [GenerationOutput(text="ok")],
        stream_speculative_mtp_detailed=lambda request: seen.append(request) or iter([GenerationStreamChunk("ok")]),
    )
    llm = LLM("test")
    monkeypatch.setattr(llm, "_get_text_generator", lambda: generator)
    monkeypatch.setattr(
        llm, "resolve_speculative_mtp_serving_plan",
        lambda **kwargs: resolved.append(kwargs) or SimpleNamespace(static_eligibility=eligibility),
    )
    params = SamplingParams(max_tokens=4, temperature=0, kv_storage="int8_per_token_head", kv_scale_dtype="fp32")
    if stream:
        list(llm.stream_speculative_mtp_detailed("hi", params))
    else:
        llm.generate_speculative_mtp_detailed("hi", params)
    assert seen[0].speculative_mtp_static_eligibility is eligibility
    assert resolved[0]["realized_group_rows"] == 1
    assert resolved[0]["kv_storage"] == "int8_per_token_head"
    assert params.speculative_mtp_static_eligibility is None


def test_library_mtp_preserves_server_supplied_eligibility(monkeypatch):
    llm = LLM("test")
    seen = []
    generator = SimpleNamespace(
        generate_speculative_mtp_detailed=lambda request: seen.append(request) or [GenerationOutput(text="ok")],
    )
    monkeypatch.setattr(llm, "_get_text_generator", lambda: generator)
    monkeypatch.setattr(
        llm, "resolve_speculative_mtp_serving_plan",
        lambda **kwargs: pytest.fail("must not replace already-resolved request intent"),
    )
    eligibility = _eligibility()
    llm.generate_speculative_mtp_detailed("hi", replace(
        SamplingParams(), speculative_mtp_static_eligibility=eligibility,
    ))
    assert seen[0].speculative_mtp_static_eligibility is eligibility
