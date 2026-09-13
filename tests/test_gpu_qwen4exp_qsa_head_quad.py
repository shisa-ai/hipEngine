import pytest
from tests import test_gpu_qwen4exp_qsa_head_pair as pair
from tests.test_gpu_qwen4exp_qsa_h256_wave import qsa,hip_available


def run_quad(f):
    qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_quad_rows_f32(
        f.dq.ptr,f.dk.ptr,f.dv.ptr,f.ds.ptr,f.dc.ptr,f.output.ptr,f.spans,
        rows=f.rows,selected_stride=f.stride,block_size=256,query_heads=24,
        kv_heads=2,head_dim=256,library=f.library,runtime=f.runtime)
    f.runtime.device_synchronize()


def test_registry_and_guard():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve
    from hipengine.core.dtype import DType
    from types import SimpleNamespace
    register_gfx1151_kernels(replace=True)
    fn=resolve(backend="hip_gfx1151",layer="qsa_sparse_attention",quant="bf16_kv",
               variant="strict_h256_head_quad_rows_spans")
    assert fn is qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_quad_rows_f32
    with pytest.raises(ValueError,match="divisible by4"):
        fn(*([1]*6),SimpleNamespace(spans_mode="uniform",storage_dtype=DType.BF16),
           rows=1,selected_stride=1,block_size=256,query_heads=12,kv_heads=2,head_dim=256)


@pytest.mark.skipif(not hip_available(),reason="HIP unavailable")
@pytest.mark.parametrize("rows,stride",[(3,33),(4,2051),(64,2048)])
def test_exact(monkeypatch,rows,stride):
    monkeypatch.setattr(pair,"run_pair",run_quad)
    pair.test_exact(rows,stride)


@pytest.mark.skipif(not hip_available(),reason="HIP unavailable")
def test_poisoned_unused_cells(monkeypatch):
    from tests import test_gpu_qwen4exp_qsa_poisoned_padding as poison
    monkeypatch.setattr(qsa,"qwen4_exp_qsa_sparse_attention_paged_bf16_h256_page256_wave_rows_f32",
                        qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_quad_rows_f32)
    poison.test_unselected_kv_and_selection_padding_are_inert("page256",0x7fc1)


@pytest.mark.skipif(not hip_available(),reason="HIP unavailable")
@pytest.mark.parametrize("scale",[128,0.00001])
def test_extreme_queries(scale):
    import numpy as np
    from hipengine.core.memory import copy_host_to_device,host_array_ptr
    f=pair.Fixture(2,257,page256=True)
    try:
        f.query*=np.float32(scale)
        copy_host_to_device(f.dq,host_array_ptr(f.query),runtime=f.runtime)
        f.run(True)
        expected=f.download(True)
        run_quad(f)
        np.testing.assert_array_equal(f.download(True).view(np.uint32),expected.view(np.uint32))
    finally:
        f.close()
