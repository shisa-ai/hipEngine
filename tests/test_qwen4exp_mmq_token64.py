import pytest
from tests import test_qwen4exp_mmq_prepack as base

CANDIDATE="gguf_q8_0_mmq128_token64_q8_1_d4x3_guarded_f32_f32_out"


def test_registry(monkeypatch):
    monkeypatch.setattr(base,"RAW_VECTOR",CANDIDATE)
    base.test_registry()


@pytest.mark.skipif(not base.hip_available(),reason="HIP unavailable")
@pytest.mark.parametrize("rows,k,n",[(17,256,144),(128,2560,320),(1024,2560,640)])
def test_exact(monkeypatch,rows,k,n):
    monkeypatch.setattr(base,"RAW_VECTOR",CANDIDATE)
    base.test_exact_prepacked(rows,k,n,0.0)


@pytest.mark.skipif(not base.hip_available(),reason="HIP unavailable")
def test_active_repair_set(monkeypatch):
    monkeypatch.setattr(base,"RAW_VECTOR",CANDIDATE)
    base.test_exact_prepacked(65,256,144,0.0001)
