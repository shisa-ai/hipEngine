"""Ambient Q4 tile retirement and scoped diagnostic policy restoration."""
import pytest
from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_prefill as kernels
from scripts import laguna_q4_pack8_shape_ab as harness


@pytest.mark.parametrize("value", ["64x16", "invalid"])
def test_retired_q4_environment_preserves_shape_policy(monkeypatch, value):
    monkeypatch.setenv("HIPENGINE_GGUF_Q4_K_DENSE_WMMA_TILE", value)
    assert kernels._default_q4_pack8_tiles(512, 1024, 3072) == (64, 32)
    assert kernels._default_q4_pack8_tiles(512, 3072, 12288) == (32, 32)
    assert kernels._default_q4_pack8_tiles(17, 256, 80) == (64, 16)


def test_diagnostic_scope_restores_policy_on_exception():
    original = kernels._default_q4_pack8_tiles
    with pytest.raises(RuntimeError):
        with harness._select("q4_64x16_rollback"):
            assert kernels._default_q4_pack8_tiles(512, 1024, 3072) == (64, 16)
            raise RuntimeError("launch failed")
    assert kernels._default_q4_pack8_tiles is original
    with harness._select("selector_unset_q4_shape_policy"):
        assert kernels._default_q4_pack8_tiles is original
    assert kernels._default_q4_pack8_tiles is original
