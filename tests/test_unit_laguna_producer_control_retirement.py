"""Retire producer overrides without changing the output-gate comparison basis."""
import inspect
from types import SimpleNamespace

import pytest
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.kernels import hip_gfx1151
from scripts import laguna_attention_packed_output_gate_ab as harness


def test_producer_override_is_removed():
    name = "prefill_attention_hipblaslt_packed_query_producer"
    assert name not in inspect.signature(LagunaGGUFResidentSession).parameters
    for value in (False, True):
        with pytest.raises(TypeError, match="unexpected keyword"):
            LagunaGGUFResidentSession(**{name: value})


@pytest.mark.parametrize("fails", [False, True])
def test_output_gate_harness_preserves_producer_control_and_restores(monkeypatch, fails):
    key = "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERY_PRODUCER"
    original = getattr(hip_gfx1151, key)
    seen = []
    result = object()

    def construct(*args, **kwargs):
        seen.append(getattr(hip_gfx1151, key))
        assert "prefill_attention_hipblaslt_packed_query_producer" not in kwargs
        if fails:
            raise RuntimeError("construction failed")
        return result

    monkeypatch.setattr(harness, "LagunaGGUFResidentSession", construct)
    monkeypatch.setattr(harness, "_compiler_version", lambda path: "test")
    owner = SimpleNamespace(weights=object(), runtime=object())
    assert set(harness._COMPARISONS) == {"packed_output_gate"}
    for settings in harness._COMPARISONS["packed_output_gate"].values():
        if fails:
            with pytest.raises(RuntimeError, match="construction failed"):
                harness._child(owner, settings)
        else:
            assert harness._child(owner, settings) is result
        assert getattr(hip_gfx1151, key) == original
    assert seen == [False, False]
