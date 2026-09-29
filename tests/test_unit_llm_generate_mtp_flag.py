"""Unit tests for the explicit ``speculative_mtp`` flag on ``LLM.generate()``.

M7's product rule is that an explicit request either runs the MTP route or
fails naming the capability -- never silently downgrades to plain
autoregressive decoding. These tests pin all three behaviours: routing when
supported, a naming failure when not, and an untouched default path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine import LLM, SamplingParams
from hipengine.generation import GenerationOutput


def _llm(monkeypatch, generator) -> LLM:
    # LLM("test") never loads weights: the generator is patched in, so the
    # dispatch under test is exercised without an artifact.
    llm = LLM("test")
    monkeypatch.setattr(llm, "_get_text_generator", lambda: generator)
    return llm


def test_unit_llm_generate_routes_explicit_mtp_flag_to_the_mtp_path(monkeypatch):
    seen = []
    generator = SimpleNamespace(
        supports_speculative_mtp=True,
        generate_speculative_mtp_detailed=lambda request: seen.append(request)
        or [GenerationOutput(text="ok")],
        generate_detailed=lambda request: pytest.fail("must not take the AR path"),
    )
    llm = _llm(monkeypatch, generator)

    outputs = llm.generate_detailed("hi", SamplingParams(speculative_mtp=True))

    assert [output.text for output in outputs] == ["ok"]
    assert len(seen) == 1, "the MTP route must run exactly once"


def test_unit_llm_generate_list_surface_honours_the_flag(monkeypatch):
    generator = SimpleNamespace(
        supports_speculative_mtp=True,
        generate_speculative_mtp_detailed=lambda request: [GenerationOutput(text="spec")],
    )
    llm = _llm(monkeypatch, generator)

    assert llm.generate("hi", SamplingParams(speculative_mtp=True)) == ["spec"]


def test_unit_llm_generate_unsupported_model_fails_naming_the_capability(monkeypatch):
    generator = SimpleNamespace(
        supports_speculative_mtp=False,
        generate_speculative_mtp_detailed=lambda request: pytest.fail(
            "must not submit unsupported work"
        ),
    )
    llm = _llm(monkeypatch, generator)

    # The name in the message is the contract: the caller is told which
    # capability is missing rather than being served plain AR output.
    with pytest.raises(NotImplementedError, match="speculative MTP"):
        llm.generate("hi", SamplingParams(speculative_mtp=True))


def test_unit_llm_generate_missing_mtp_callable_fails_loudly(monkeypatch):
    # A generator that reports support but exposes no route is still a refusal
    # case, not a reason to fall back.
    generator = SimpleNamespace(supports_speculative_mtp=True)
    llm = _llm(monkeypatch, generator)

    with pytest.raises(NotImplementedError, match="speculative MTP"):
        llm.generate("hi", SamplingParams(speculative_mtp=True))


def test_unit_llm_generate_flag_off_keeps_the_plain_path(monkeypatch):
    generator = SimpleNamespace(
        generate_detailed=lambda request: [GenerationOutput(text="plain")],
        generate_speculative_mtp_detailed=lambda request: pytest.fail(
            "default requests must not be diverted to MTP"
        ),
    )
    llm = _llm(monkeypatch, generator)

    assert llm.generate("hi") == ["plain"]
    assert llm.generate("hi", SamplingParams()) == ["plain"]
    assert llm.generate_detailed("hi", SamplingParams(speculative_mtp=False)) == [
        GenerationOutput(text="plain")
    ]


def test_unit_llm_generate_speculative_flag_defaults_to_false():
    # The flag is opt-in: shipping it must not change any existing caller's
    # arithmetic, so the default is asserted rather than assumed.
    assert SamplingParams().speculative_mtp is False
    assert "speculative_mtp" in SamplingParams.__dataclass_fields__