import pytest
from tests import test_gpu_qwen4exp_q51_pair as base

CANDIDATE = "qwen4_exp_q5_1_selected_grouped_prefill_pair2_row_publish_bf16_bf16_out"


def test_registry():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve
    register_gfx1151_kernels(replace=True)
    fn = resolve(backend="hip_gfx1151", layer="moe_linear", quant="gguf_q5_1",
                 variant="selected_grouped_prefill_pair2_row_publish_bf16_bf16_out")
    assert fn is getattr(base.q5, CANDIDATE)
    with pytest.raises(ValueError, match="640"):
        fn(1, 1, 1, 1, 1, 1, 256, 1)


@pytest.mark.skipif(not base.hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize("rows,experts,n", [
    (17, 8, 7), (65, 64, 33), (5120, 512, 2560), (10240, 512, 2560)])
def test_exact_and_cpu(monkeypatch, rows, experts, n):
    monkeypatch.setattr(base, "PARENT", base.REGISTER_CACHE)
    base.test_pair_exact(rows, experts, 640, n, CANDIDATE)
