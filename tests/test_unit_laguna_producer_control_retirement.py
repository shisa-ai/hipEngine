"""Retire producer overrides while preserving a distinct output-gate comparison."""
import inspect
from types import SimpleNamespace

import pytest
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.kernels import hip_gfx1151
from hipengine.kernels.hip_gfx1151 import policy_laguna
from scripts import laguna_attention_packed_output_gate_ab as harness


@pytest.mark.parametrize("fails", [False, True])
def test_packed_attention_diagnostic_restores_nested_owner_choices(fails):
    owner = object.__new__(LagunaGGUFResidentSession)
    owner.prefill_attention_hipblaslt_packed_query_producer = False
    owner.prefill_attention_hipblaslt_packed_output_gate = True
    peer = object.__new__(LagunaGGUFResidentSession)
    peer.prefill_attention_hipblaslt_packed_query_producer = False
    peer.prefill_attention_hipblaslt_packed_output_gate = True
    key = "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERY_PRODUCER"
    package_value = getattr(policy_laguna, key)
    with owner.packed_attention_diagnostic(packed_query=True, packed_output_gate=False):
        assert owner.prefill_attention_hipblaslt_packed_query_producer is True
        assert owner.prefill_attention_hipblaslt_packed_output_gate is False
        try:
            with owner.packed_attention_diagnostic(packed_query=False, packed_output_gate=True):
                assert owner.prefill_attention_hipblaslt_packed_query_producer is False
                if fails:
                    raise RuntimeError("comparison failed")
        except RuntimeError:
            assert fails
        assert owner.prefill_attention_hipblaslt_packed_query_producer is True
        assert owner.prefill_attention_hipblaslt_packed_output_gate is False
        assert peer.prefill_attention_hipblaslt_packed_query_producer is False
        assert peer.prefill_attention_hipblaslt_packed_output_gate is True
        assert getattr(policy_laguna, key) is package_value
    assert owner.prefill_attention_hipblaslt_packed_query_producer is False
    assert owner.prefill_attention_hipblaslt_packed_output_gate is True


@pytest.mark.parametrize("name", [
    "prefill_attention_hipblaslt_packed_query_producer",
    "prefill_attention_hipblaslt_packed_output_gate",
])
def test_producer_override_is_removed(name):
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
        seen.append(hasattr(hip_gfx1151, key))
        assert hasattr(hip_gfx1151, key) is original_present
        if original_present:
            assert getattr(hip_gfx1151, key) is original_value
        assert "prefill_attention_hipblaslt_packed_query_producer" not in kwargs
        assert "prefill_attention_hipblaslt_packed_output_gate" not in kwargs
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

    assert seen == [original_present, original_present]
    assert getattr(policy_laguna, key) is owner_value
