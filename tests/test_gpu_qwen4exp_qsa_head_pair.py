import pytest
import numpy as np
from tests.test_gpu_qwen4exp_qsa_h256_wave import Fixture,hip_available,qsa


def run_pair(f):
    qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_pair_rows_f32(
        f.dq.ptr,f.dk.ptr,f.dv.ptr,f.ds.ptr,f.dc.ptr,f.output.ptr,f.spans,
        rows=f.rows,selected_stride=f.stride,block_size=256,
        query_heads=24,kv_heads=2,head_dim=256,library=f.library,runtime=f.runtime)
    f.runtime.device_synchronize()


def test_registry():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve
    register_gfx1151_kernels(replace=True)
    assert resolve(backend="hip_gfx1151",layer="qsa_sparse_attention",quant="bf16_kv",
                   variant="strict_h256_head_pair_rows_spans") is (
        qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_pair_rows_f32)
    from types import SimpleNamespace
    from hipengine.core.dtype import DType
    spans=SimpleNamespace(spans_mode="uniform",storage_dtype=DType.BF16)
    with pytest.raises(ValueError,match="even GQA"):
        qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_pair_rows_f32(
            *([1]*6),spans,rows=1,selected_stride=1,block_size=256,
            query_heads=6,kv_heads=2,head_dim=256)


@pytest.mark.skipif(not hip_available(),reason="HIP unavailable")
@pytest.mark.parametrize("rows,stride",[(3,33),(4,2051),(64,2048)])
def test_exact(rows,stride):
    f=Fixture(rows,stride,edge=True,page256=True)
    try:
        f.run(True)
        parent=f.download(True)
        for _ in range(2):
            run_pair(f)
            np.testing.assert_array_equal(f.download(True).view(np.uint32),parent.view(np.uint32))
    finally:
        f.close()


@pytest.mark.skipif(not hip_available(),reason="HIP unavailable")
def test_poisoned_unused_cells(monkeypatch):
    from tests import test_gpu_qwen4exp_qsa_poisoned_padding as poison
    monkeypatch.setattr(qsa,"qwen4_exp_qsa_sparse_attention_paged_bf16_h256_page256_wave_rows_f32",
                        qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_pair_rows_f32)
    poison.test_unselected_kv_and_selection_padding_are_inert("page256",0x7fc1)
