import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear

PARENT = "dense_wide256_f16in_f32_f32_out"
LONG_K = "dense_wide128x128_wgp_f16in_f32_f32_out"


def _key(variant=PARENT):
    return KernelKey("hip_gfx1100", "linear", "gguf_q8_0", variant)


@pytest.mark.parametrize(
    "rows,in_features,out_features,expected",
    [
        (4096, 6144, 2560, LONG_K),
        (1024, 6144, 2560, LONG_K),
        (512, 6144, 2560, PARENT),      # measured slower below 1024 rows
        (4096, 2560, 10240, PARENT),    # the builds tie at K=2560
        (4096, 10240, 320, PARENT),     # narrow output: CU mode wins
    ],
)
def test_long_k_rule_selects_by_shape(monkeypatch, rows, in_features, out_features, expected):
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_DENSE_WIDE_LONG_K", raising=False)
    key = linear._dense_wide_f16_long_k_key(
        _key(), rows=rows, in_features=in_features, out_features=out_features)
    assert key.variant == expected
    assert (key.backend, key.layer, key.quant) == ("hip_gfx1100", "linear", "gguf_q8_0")


def test_long_k_rule_opt_out_and_other_parents(monkeypatch):
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_DENSE_WIDE_LONG_K", "0")
    assert linear._dense_wide_f16_long_k_key(
        _key(), rows=4096, in_features=6144, out_features=2560).variant == PARENT
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_DENSE_WIDE_LONG_K")
    padded = "dense_wide256_padded_f16in_f32_f32_out"
    assert linear._dense_wide_f16_long_k_key(
        _key(padded), rows=4096, in_features=6144, out_features=2560).variant == padded
