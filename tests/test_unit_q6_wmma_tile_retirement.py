"""Q6 WMMA tile policy is independent of the retired ambient override."""
import pytest
from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_prefill as kernels


@pytest.mark.parametrize("value", ["16x32", "64x16", "invalid"])
@pytest.mark.parametrize("default", [(64, 16), (16, 32)])
def test_retired_q6_tile_environment_does_not_override_backend(monkeypatch, value, default):
    monkeypatch.setenv("HIPENGINE_GGUF_Q6_K_DENSE_WMMA_TILE", value)
    calls = []
    monkeypatch.setattr(kernels, "_launch", lambda *args, **kwargs: calls.append(kwargs))
    wrapper = kernels._make_q6_wrapper("wmma_prefill_bf16_bf16_out", default_tile=default)
    wrapper()
    assert (calls[-1]["tile_m"], calls[-1]["tile_n"]) == default
    wrapper(tile_m=32, tile_n=32)
    assert (calls[-1]["tile_m"], calls[-1]["tile_n"]) == (32, 32)
