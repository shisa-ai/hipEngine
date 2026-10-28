"""GDN chain launch schedule, numerical oracle and retired-selector controls."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.linear_attn import gdn
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from scripts.gdn_chain_microbench import gdn_chain_reference


_CASES = ((1, 64, 7), (3, 128, 8), (5, 192, 31), (2, 128, 128))


@pytest.fixture(scope="module")
def runtime():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime unavailable")
    from hipengine.core.hip import get_hip_runtime
    return get_hip_runtime()


@pytest.fixture(scope="module", params=[False, True], ids=["interleaved", "grouped"])
def grouped(request):
    return request.param


@pytest.fixture(scope="module")
def library(runtime, hip_test_target_arch, grouped):
    from hipengine.kernels.backends import hip_target_arch_environment
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    version = Path(version_file).read_text() if version_file else None
    with hip_target_arch_environment(hip_test_target_arch):
        build = gdn.build_qwen35_linear_attn_gdn_grouped_heads if grouped else gdn.build_qwen35_linear_attn_gdn
        return build(load=True, compiler_version=version)


def _run_case(runtime, library, scalar, rows, dk, dv, grouped=False):
    hk, hv, eps = 2, 4, 1e-6
    rng = np.random.default_rng(428 + rows + dk + dv)
    conv = (rng.standard_normal((rows, 2 * hk * dk + hv * dv)) * 0.2).astype(np.float32)
    lowp = [(rng.standard_normal(shape) * 0.2).astype(np.float32)
            for shape in ((rows, hv * dv), (rows, hv), (rows, hv))]
    if scalar == "bf16":
        packed = [float_array_to_bf16_bits(a) for a in lowp]
        rounded = [bf16_to_float32(a) for a in packed]
    else:
        packed = [a.astype(np.float16) for a in lowp]
        rounded = packed
    bias = (rng.standard_normal(hv) * 0.1).astype(np.float32)
    alog = (rng.standard_normal(hv) * 0.1).astype(np.float32)
    norm = (1 + rng.standard_normal(dv) * 0.1).astype(np.float32)
    base = (rng.standard_normal((hv, dk, dv)) * 0.1).astype(np.float32)
    # Expand Q/K for the CPU oracle so both supported head layouts map each
    # V head explicitly; the recurrence formula itself stays independent.
    indices = np.arange(hv) // (hv // hk) if grouped else np.arange(hv) % hk
    q = conv[:, :hk * dk].reshape(rows, hk, dk)[:, indices].reshape(rows, hv * dk)
    k = conv[:, hk * dk:2 * hk * dk].reshape(rows, hk, dk)[:, indices].reshape(rows, hv * dk)
    oracle_conv = np.concatenate((q, k, conv[:, 2 * hk * dk:]), axis=1)
    ref_out, ref_leaf = gdn_chain_reference(oracle_conv, *rounded, bias, alog, norm, base, hv, hv, dk, dv, eps)
    arrays = [conv, *packed, bias, alog, norm, base,
              np.full((rows, hv * dk * dv), np.nan, dtype=np.float32),
              np.full((rows, hv * dv), np.nan, dtype=np.float32),
              np.full((rows, hv * dv), np.nan, dtype=np.float32)]
    buffers = []
    stream = runtime.stream_create()
    try:
        for array in arrays:
            buf = malloc(array.nbytes, runtime=runtime)
            buffers.append(buf)
            copy_host_to_device(buf, host_array_ptr(array), array.nbytes, runtime=runtime)
        fn = getattr(gdn, f"qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_{scalar}")
        fn(*(buf.ptr for buf in buffers), eps, rows, hk, hv, dk, dv,
           stream=stream, library=library, runtime=runtime)
        runtime.stream_synchronize(stream)
        leaf, acc, out = [np.empty_like(a) for a in arrays[-3:]]
        for array, buf in zip((leaf, acc, out), buffers[-3:]):
            copy_device_to_host(host_array_ptr(array), buf, array.nbytes, runtime=runtime)
        assert all(np.isfinite(a).all() for a in (leaf, acc, out))
        np.testing.assert_allclose(leaf, ref_leaf, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(out, ref_out, atol=2e-5, rtol=2e-4)
        return out, leaf, acc
    finally:
        runtime.stream_synchronize(stream)
        for buf in reversed(buffers):
            free(buf, runtime=runtime)
        runtime.stream_destroy(stream)


@pytest.mark.parametrize("rows,dk,dv", _CASES)
@pytest.mark.parametrize("scalar", ["fp16", "bf16"])
def test_chain_tile_schedule_ignores_retired_selector(runtime, library, grouped, monkeypatch, scalar, rows, dk, dv):
    monkeypatch.setenv("HIPENGINE_GDN_TLOOP_C1_EXACT", "0")
    controls = None
    for retired in (None, "1", "4", "bogus"):
        if retired is None:
            monkeypatch.delenv("HIPENGINE_GDN_CHAIN_TLOOP_VTILE", raising=False)
        else:
            monkeypatch.setenv("HIPENGINE_GDN_CHAIN_TLOOP_VTILE", retired)
        observed = _run_case(runtime, library, scalar, rows, dk, dv, grouped)
        if controls is None:
            controls = observed
        for left, right in zip(observed, controls):
            np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
