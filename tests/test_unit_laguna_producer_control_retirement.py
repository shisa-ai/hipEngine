"""Retire producer overrides while preserving a distinct output-gate comparison."""
import inspect
from types import SimpleNamespace

import pytest
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.kernels import hip_gfx1151
from hipengine.kernels.hip_gfx1151 import policy_laguna
from scripts import laguna_attention_packed_output_gate_ab as harness


def test_producer_override_is_removed():
    name = "prefill_attention_hipblaslt_packed_query_producer"
    assert name not in inspect.signature(LagunaGGUFResidentSession).parameters
    for value in (False, True):
        with pytest.raises(TypeError, match="unexpected keyword"):
            LagunaGGUFResidentSession(**{name: value})


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("initial", ["absent", False, True])
def test_output_gate_harness_restores_producer_control_presence_and_value(
    monkeypatch, fails, initial
):
    key = "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERY_PRODUCER"
    owner_value = getattr(policy_laguna, key)

    if initial == "absent":
        monkeypatch.delattr(hip_gfx1151, key, raising=False)
        original_present = False
        original_value = None
    else:
        monkeypatch.setattr(hip_gfx1151, key, initial, raising=False)
        original_present = True
        original_value = initial
    assert hasattr(hip_gfx1151, key) is original_present

    seen = []
    result = object()
    resident_owner = SimpleNamespace(weights=object(), runtime=object())

    def construct(*args, **kwargs):
        seen.append(getattr(hip_gfx1151, key))
        assert "prefill_attention_hipblaslt_packed_query_producer" not in kwargs
        assert (
            kwargs["prefill_attention_hipblaslt_packed_output_gate"]
            is settings["packed_output_gate"]
        )
        assert kwargs["resident_weights"] is resident_owner.weights
        assert kwargs["runtime"] is resident_owner.runtime
        if fails:
            raise RuntimeError("construction failed")
        return result

    monkeypatch.setattr(harness, "LagunaGGUFResidentSession", construct)
    monkeypatch.setattr(harness, "_compiler_version", lambda path: "test")
    assert set(harness._COMPARISONS) == {"packed_output_gate"}
    for settings in harness._COMPARISONS["packed_output_gate"].values():
        if fails:
            with pytest.raises(RuntimeError, match="construction failed"):
                harness._child(resident_owner, settings)
        else:
            assert harness._child(resident_owner, settings) is result

        # Exact restoration after both success and construction failure.
        assert hasattr(hip_gfx1151, key) is original_present
        if original_present:
            assert getattr(hip_gfx1151, key) is original_value
        assert getattr(policy_laguna, key) is owner_value

    assert seen == [True, True]
    assert getattr(policy_laguna, key) is owner_value
