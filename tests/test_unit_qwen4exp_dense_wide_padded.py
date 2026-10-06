"""Surface contract for the measurement-only padded-fragment wide sibling."""
from types import SimpleNamespace

import pytest

from scripts import qwen4exp_dense_wide_padded_micro as micro

from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw
from hipengine.kernels.registry import KernelKey, is_registered, resolve

VARIANTS = (
    "dense_wide256_padded_f32_f32_out",
    "dense_wide256_padded_f16in_f32_f32_out",
)


def test_real_weight_loader_returns_row_major_k_n_and_exact_bytes(tmp_path, monkeypatch):
    file = tmp_path / "weight.gguf"
    file.write_bytes(b"header!!" + b"pad!" + bytes(range(34)))
    tensor = SimpleNamespace(name="projection", ggml_type=8, shape=(1, 32),
                             offset=4, nbytes=34)
    monkeypatch.setattr(micro, "discover_gguf_files", lambda root: (file,))
    monkeypatch.setattr(micro, "load_gguf_index", lambda file: SimpleNamespace(
        tensors=(tensor,), tensor_data_offset=8))
    raw, shape = micro.load_weight(tmp_path, "projection")
    assert shape == (32, 1)
    assert raw.tobytes() == bytes(range(34))


def test_natural_owner_admission_is_shape_based_and_bounded():
    from scripts.qwen4exp_dense_wide_padded_owner import admitted_shape

    assert admitted_shape(512, 10240, 320)
    assert admitted_shape(4096, 10240, 320)
    assert not admitted_shape(511, 10240, 320)
    assert not admitted_shape(4097, 10240, 320)
    assert not admitted_shape(512, 320, 10240)
    assert not admitted_shape(512, 10240, 384)


def test_padded_siblings_are_registered_without_replacing_parent():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels()
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        for variant in (*VARIANTS, "dense_wide256_f16in_f32_f32_out"):
            assert is_registered(KernelKey(backend, "linear", "gguf_q8_0", variant))
            assert resolve(backend=backend, layer="linear", quant="gguf_q8_0",
                           variant=variant) is dw._WRAPPERS[variant]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("shape", [(0, 64, 16), (1, 96, 16), (1, 64, 0)])
def test_padded_sibling_rejects_invalid_shape_before_runtime(variant, shape):
    with pytest.raises(ValueError):
        dw._WRAPPERS[variant](1, 2, 3, *shape)
