"""Source-Q5 diagnostic controls restore the native selector on failure."""
import pytest
from hipengine.runtime import qwen35_gguf_runner as runner
from scripts.qwen38_q4_verifier_numerics import source_mmq_control


def test_source_control_restores_nested_selection_and_error(monkeypatch):
    monkeypatch.delenv('HIPENGINE_GGUF_C8_Q5_RAW_MMQ', raising=False)
    original = runner._gguf_c8_q5_source_mmq_enabled
    assert original('hip_gfx1100', request_count=8)
    with pytest.raises(RuntimeError):
        with source_mmq_control(False):
            assert not runner._gguf_c8_q5_source_mmq_enabled('hip_gfx1100', request_count=8)
            with source_mmq_control(False):
                assert not runner._gguf_c8_q5_source_mmq_enabled('hip_gfx1100', request_count=8)
            raise RuntimeError('diagnostic failed')
    assert runner._gguf_c8_q5_source_mmq_enabled is original
    with source_mmq_control(True):
        assert runner._gguf_c8_q5_source_mmq_enabled is original


def test_raw_control_restores_after_failure_and_preserves_capability():
    from scripts.qwen38_q4_verifier_numerics import raw_mmq_control

    original = runner._gguf_c8_q5_raw_mmq_enabled
    assert original('hip_gfx1100', request_count=8)
    assert not original('hip_gfx1151', request_count=8)
    with pytest.raises(RuntimeError):
        with raw_mmq_control(False):
            assert not runner._gguf_c8_q5_raw_mmq_enabled('hip_gfx1100', request_count=8)
            with raw_mmq_control(False):
                assert not runner._gguf_c8_q5_source_mmq_enabled('hip_gfx1100', request_count=8)
            raise RuntimeError('arm failed')
    assert runner._gguf_c8_q5_raw_mmq_enabled is original
