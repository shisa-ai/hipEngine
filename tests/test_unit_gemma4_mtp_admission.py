"""Gemma assistant backend resolution and paired admission before allocation."""
from types import SimpleNamespace

import pytest

from hipengine.generation import gemma4_mtp
from hipengine.loading import gemma4_mtp_gguf as admission
from hipengine.speculative.registry import SpeculativeProviderConfig, resolve_speculative_provider
from test_unit_gemma4_mtp_loader import _assistant_info, _target_info


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_provider_resolves_on_implemented_backends_without_quant_allowlist(backend):
    gemma4_mtp.register_gemma4_mtp_providers()
    factory = resolve_speculative_provider(
        provider="gemma4_mtp", target_model="gemma4_gguf",
        backend=backend, quant="unfamiliar_implemented_quant",
    )
    assert factory is gemma4_mtp.make_gemma4_mtp_provider


@pytest.mark.parametrize("metadata", [
    {"gemma4-assistant.embedding_length_out": 1024},
    {"general.architecture": "not-an-assistant"},
])
def test_incompatible_pair_is_rejected_before_target_or_assistant_allocation(monkeypatch, metadata):
    target = _target_info()
    assistant = _assistant_info(metadata_overrides=metadata)
    monkeypatch.setattr(admission, "scan_gguf", lambda path: target if str(path) == "target.gguf" else assistant)
    allocations = []

    def forbidden():
        allocations.append("target")
        raise AssertionError("target allocation occurred before pair admission")

    def forbidden_assistant(*args, **kwargs):
        allocations.append("assistant")
        raise AssertionError("assistant allocation occurred before pair admission")

    monkeypatch.setattr(gemma4_mtp, "load_gemma4_assistant_device_weights", forbidden_assistant)
    provider = gemma4_mtp.Gemma4MTPTextProvider(
        target_generator=SimpleNamespace(model_path="target.gguf", _ensure_runner=forbidden),
        config=SpeculativeProviderConfig(provider="gemma4_mtp", draft_model="assistant.gguf", candidate_budget=2),
    )
    with pytest.raises(admission.Gemma4MTPGGUFError, match="capability admission"):
        provider._ensure_drafter()
    assert allocations == []
    assert provider._head is None
    assert provider._drafter is None
    provider.close()


@pytest.mark.parametrize("compatible", [False, True])
def test_runtime_geometry_preflight_precedes_cold_construction(monkeypatch, compatible):
    events = []
    target_config, assistant_config = object(), object()
    runner = SimpleNamespace(capacity=4096)
    weights = SimpleNamespace(free=lambda: events.append("free"))
    head = SimpleNamespace(close=lambda: events.append("close"))
    monkeypatch.setattr(gemma4_mtp, "require_gemma4_mtp", lambda *a: events.append("paired"))
    monkeypatch.setattr(gemma4_mtp, "scan_gguf", lambda *a: SimpleNamespace(metadata={}))
    monkeypatch.setattr(gemma4_mtp, "gemma4_gguf_config_from_metadata", lambda *a: target_config)
    monkeypatch.setattr(gemma4_mtp, "gemma4_assistant_config_from_metadata", lambda *a: assistant_config)

    def geometry(assistant, target):
        assert assistant is assistant_config and target is target_config
        events.append("geometry")
        if not compatible:
            raise ValueError("assistant block has incompatible Q heads")

    def materialize():
        events.append("target")
        return runner

    def load(*a):
        events.append("assistant")
        return weights

    def make_head(**kwargs):
        assert kwargs["backbone"] is target_config
        events.append("head")
        return head

    monkeypatch.setattr(gemma4_mtp, "gemma4_assistant_geometry", geometry)
    monkeypatch.setattr(gemma4_mtp, "load_gemma4_assistant_device_weights", load)
    monkeypatch.setattr(gemma4_mtp, "Gemma4AssistantHead", make_head)
    monkeypatch.setattr(gemma4_mtp, "Gemma4MtpDrafter", lambda **kwargs: SimpleNamespace(**kwargs))
    generator = SimpleNamespace(
        model_path="target.gguf", _ensure_runner=materialize,
        _weights=SimpleNamespace(embed_tokens=object(), final_norm=SimpleNamespace(buffer=object())),
    )
    provider = gemma4_mtp.Gemma4MTPTextProvider(
        target_generator=generator,
        config=SpeculativeProviderConfig(provider="gemma4_mtp", draft_model="assistant.gguf", candidate_budget=2),
    )
    if compatible:
        drafter = provider._ensure_drafter()
        assert events == ["paired", "geometry", "target", "assistant", "head"]
        assert provider._ensure_drafter() is drafter
        assert events[-1] == "target"
        assert events.count("paired") == events.count("geometry") == 1
    else:
        with pytest.raises(ValueError, match="Q heads"):
            provider._ensure_drafter()
        assert events == ["paired", "geometry"]
    provider.close()
