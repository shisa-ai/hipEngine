"""Dense Q4 diagnostic selection belongs to the harness, not the runner API."""
from types import SimpleNamespace
import pytest
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from scripts import laguna_grouped_down_category_bench as harness


def test_runtime_setter_is_absent():
    assert not hasattr(LagunaGGUFResidentSession, "set_dense_q4_prefill_mode")


def test_diagnostic_mode_is_validated_and_session_local():
    first = SimpleNamespace(backend="hip_gfx1151", dense_q4_prefill_mode="wmma_pack8")
    other = SimpleNamespace(backend="hip_gfx1151", dense_q4_prefill_mode="wmma_pack8")
    harness._set_dense_q4_prefill_mode(first, "retained")
    assert first.dense_q4_prefill_mode == "retained"
    assert other.dense_q4_prefill_mode == "wmma_pack8"
    with pytest.raises(ValueError, match="unsupported"):
        harness._set_dense_q4_prefill_mode(first, "invalid")
    assert first.dense_q4_prefill_mode == "retained"
