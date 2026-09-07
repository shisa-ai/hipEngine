"""Raw dense K_M planning and concrete consumer integration."""
import pytest

from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.qwen35_gguf_materialize import _spec_for_tensor, LAYOUT_RAW_GGUF
from hipengine.loading.qwen35_gguf_consumer_surface import source_linear_dispatch_row
from hipengine.quant.gguf import GGMLQuantizationType


@pytest.mark.parametrize('name,stride', [('IQ4_XS', 136), ('IQ4_NL', 144), ('IQ3_S', 110), ('Q3_K', 110)])
@pytest.mark.parametrize('repack', [False, True])
def test_raw_dense_plan_and_consumer(name, stride, repack):
    tensor = GGUFTensorInfo('blk.0.ffn_gate.weight', (3, 5120), (5120, 3),
                           int(GGMLQuantizationType[name]), name, 3*5120,
                           3*20*stride, 0, 0, (3, 20*stride))
    spec = _spec_for_tensor('layers.0.ffn_gate', tensor, decode_repack=repack)
    assert spec.layout == LAYOUT_RAW_GGUF
    assert spec.quant_key == 'gguf_'+name.lower()
    assert spec.allocation_names == ('raw',)
    assert not spec.sidecar_layouts
    for output in ('bf16', 'f32'):
        row = source_linear_dispatch_row(name, spec.layout, 'bf16', output)
        assert row is not None
        assert row.quant == spec.quant_key
        assert row.abi == 'raw'
        assert row.variant_for_rows(1) == f'gemv_bf16_{output}_out'
        assert row.variant_for_rows(3) == f'prefill_bf16_{output}_out'
    assert source_linear_dispatch_row(name, spec.layout, 'bf16', 'fp16') is None
    assert source_linear_dispatch_row(name, spec.layout, 'f32', 'bf16') is None
