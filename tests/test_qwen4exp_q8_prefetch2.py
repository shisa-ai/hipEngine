import pytest

from tests import test_qwen4exp_q8_wave_scale as base

PARENT = base.CANDIDATE
CANDIDATE = "gguf_q8_0_gemv_coltile8_rowbatch4_prefetch2_f32_f32_out"


def test_registry(monkeypatch):
    monkeypatch.setattr(base, "PARENT", PARENT)
    monkeypatch.setattr(base, "CANDIDATE", CANDIDATE)
    base.test_registry()


@pytest.mark.skipif(not base.hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize("rows,k,n", [
    (1, 32, 8), (7, 96, 24), (17, 288, 32), (17, 2560, 32),
    (512, 2560, 6144), (1024, 2560, 6144),
])
def test_exact_and_cpu(monkeypatch, rows, k, n):
    monkeypatch.setattr(base, "PARENT", PARENT)
    monkeypatch.setattr(base, "CANDIDATE", CANDIDATE)
    base.test_exact_and_cpu(rows, k, n)
