"""Ordinary INT8 attention correctness tests may build a missing cache entry."""
import pytest
from hipengine.core.dtype import DType
from tests import test_gpu_qwen38_int8_batch_attention_gpu as gate


def test_attention_gate_uses_normal_build_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("HIPENGINE_COMPILER_VERSION_FILE", str(tmp_path / "missing-version"))
    monkeypatch.setattr(gate, "get_hip_runtime", lambda: object())
    calls = []

    class StopAfterBuild(Exception):
        pass

    def build(**kwargs):
        calls.append(kwargs)
        raise StopAfterBuild

    monkeypatch.setattr(gate, "build_qwen35_paged_attn_decode", build)
    try:
        with pytest.raises(StopAfterBuild):
            gate._run_attention_gate(1, (1,), shared_table=False, scale_dtype=DType.FP32)
    except pytest.skip.Exception:
        pytest.fail("correctness gate skipped instead of resolving its build")
    assert calls == [{"load": True}]
