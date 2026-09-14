"""Peer dispatch and pre-mutation capability fallback without GPU access."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from hipengine.kernels.vibevoice import resolve_vibevoice_kernels,resolve_vibevoice_kernel
from hipengine.kernels.registry import KernelKey,register
from hipengine.runtime.vibevoice_qwen2 import VibevoiceQwen2Runtime
from hipengine.core.memory import DeviceBuffer


@pytest.mark.parametrize('backend',['hip_gfx1100','hip_gfx1151'])
def test_peer_primitives_and_strict_fallback(backend):
    ops=resolve_vibevoice_kernels(backend)
    assert ops.backend==backend
    assert callable(ops.vv_kv_write_spans)
    assert callable(resolve_vibevoice_kernel(backend,'vibevoice_prefill','strict'))
    assert callable(resolve_vibevoice_kernel(backend,'vibevoice_frontend_gemm','strict'))


def test_no_lt_capability_uses_registered_fallback():
    r=VibevoiceQwen2Runtime.__new__(VibevoiceQwen2Runtime)
    r.backend='hip_gfx1151';r.max_context=4;r.spec=SimpleNamespace(hidden_size=2)
    r.prefill_variant='hipblaslt';r._prepare_lt=Mock(side_effect=OSError('missing library'))
    fallback=Mock()
    register(KernelKey(r.backend,'vibevoice_prefill','bf16','strict'),fallback,replace=True)
    r._prefill_routes={'strict':resolve_vibevoice_kernel(r.backend,'vibevoice_prefill','strict')}
    r.prefill_rows(DeviceBuffer(1,8),2,0)
    fallback.assert_called_once()
    assert r.prefill_fallback_reason == 'missing library'
    assert r.variant_manifest['execution_profile']=='strict'
    assert len(r.variant_manifest_sha256)==64


def test_frontend_variant_selects_callable_depthwise_fallback():
    strict=resolve_vibevoice_kernels('hip_gfx1151',frontend_variant='strict')
    candidate=resolve_vibevoice_kernels('hip_gfx1151',frontend_variant='wmma')
    assert strict.vv_depthwise_conv_bf16 is resolve_vibevoice_kernel('hip_gfx1151','vv_depthwise_conv_bf16','strict')
    assert candidate.vv_depthwise_conv_bf16 is resolve_vibevoice_kernel('hip_gfx1151','vv_depthwise_conv_bf16','fused')
    assert strict.vv_depthwise_conv_bf16 is not candidate.vv_depthwise_conv_bf16
