"""The DPP comparison control is a live raw-string dispatch memo dimension."""
from types import SimpleNamespace

from hipengine.runtime import gguf_linear as linear
from hipengine.kernels import registry
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF


def test_dpp_comparison_partitions_and_reuses_dispatch_cache(monkeypatch):
    prefix = 'coltile8_rowbatch4_wave_scale_pipelined'
    parent = prefix + '_f32_f32_out'
    dpp = prefix + '_dpp_f32_f32_out'
    launched = []
    monkeypatch.setattr(linear, 'raw_k_prefill_rowbatch', lambda: 32)
    monkeypatch.setattr(linear, 'raw_k_prefill_variant', lambda: 'coltile8_wave_scale')
    monkeypatch.setattr(linear, '_raw_k_prefill_rowbatch', SimpleNamespace(get=lambda: 32))
    monkeypatch.setattr(linear, '_raw_k_prefill_variant', SimpleNamespace(get=lambda: 'coltile8_wave_scale'))
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE', '0')
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_STAGED_F32_LAYERS', '')
    monkeypatch.setenv('HIPENGINE_GGUF_Q4K_ROWTILE', '0')
    weight = SimpleNamespace(
        spec=SimpleNamespace(layout=LAYOUT_RAW_GGUF, quant_key='gguf_q8_0'),
        allocation=lambda name='raw': SimpleNamespace(tensor=SimpleNamespace(ptr=16)))
    keys = [registry.KernelKey('hip_gfx1151', 'linear', 'gguf_q8_0', value)
            for value in (parent, dpp)]
    saved = {key: registry._KERNELS.get(key) for key in keys}
    linear.clear_gguf_linear_dispatch_cache()
    real = linear._raw_k_coltile_pipelined_dispatch
    resolutions = []
    def select(dispatch, **shape):
        resolutions.append(linear._q8_dpp_reduction_raw())
        return real(dispatch, **shape)
    monkeypatch.setattr(linear, '_raw_k_coltile_pipelined_dispatch', select)
    try:
        for key in keys:
            registry.register(key, lambda *a, _v=key.variant, **kw: launched.append(_v),
                              replace=True)
        for raw in ('0', '1', '1', '0', None):
            if raw is None:
                monkeypatch.delenv(linear._Q8_DPP_REDUCTION_ENV, raising=False)
            else:
                monkeypatch.setenv(linear._Q8_DPP_REDUCTION_ENV, raw)
            linear.launch_gguf_linear(
                weight, 1, 2, 73, 2560, 10240,
                activation_dtype='f32', output_dtype='f32',
                backend='hip_gfx1151', use_wmma_prefill=False,
                use_gemv_decode=False, runtime=object())
        assert launched == [parent, dpp, dpp, parent, dpp]
        assert resolutions == ['0', '1'], 'cached arms must not rerun selection'
    finally:
        for key, value in saved.items():
            if value is None:
                registry.unregister(key)
            else:
                registry.register(key, value, replace=True)
        linear.clear_gguf_linear_dispatch_cache()
