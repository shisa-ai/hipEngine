"""IQ4 multirow tile geometry preserves scalar arithmetic and CPU safety."""
import ctypes
import json
from pathlib import Path
import numpy as np
import pytest

try:
    ctypes.CDLL('libamdhip64.so')
except OSError:
    pytest.skip('HIP runtime unavailable', allow_module_level=True)

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import malloc, free, copy_host_array_to_device, copy_device_to_host, host_array_ptr
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_dense as iq
from tests.test_gpu_gguf_iq_local32_rows import bf16, bf16_f32


@pytest.mark.parametrize('rows', [2, 3, 4])
@pytest.mark.parametrize('waves', [1, 2, 4])
@pytest.mark.parametrize('k,n', [(256, 8), (1024, 24), (5120, 128)])
def test_iq4_row_tiles_match_scalar_rows_and_cpu(rows, waves, k, n):
    fixture = Path(__file__).parent / 'fixtures/gguf_ud'
    entry = next(e for e in json.loads((fixture/'real_rows.json').read_text())['entries'] if e['type']=='IQ4_XS')
    with np.load(fixture/'real_rows.npz') as data:
        raw = data[entry['key']+'_raw'].reshape(-1,136)
        decoded = data[entry['key']+'_f32'].reshape(-1,256)
        indices = np.random.default_rng(31).integers(0,len(raw),size=(n,k//256))
        weights = np.ascontiguousarray(raw[indices])
        reference_weights = np.ascontiguousarray(decoded[indices]).reshape(n,k)
    x = bf16(np.random.default_rng(7).normal(0,0.03,(rows,k)))
    runtime = get_hip_runtime()
    lib = iq.build_gguf_iq_dense(load=True)
    scalar = lib.hipengine_gguf_iq4_xs_local32_gemv
    batch = lib.hipengine_gguf_iq4_xs_local32_rows_gemv
    for fn in (scalar,batch):
        fn.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int64]*4+[ctypes.c_void_p]
        fn.restype = ctypes.c_int
    bufs = []
    def upload(a):
        b=malloc(a.nbytes);bufs.append(b);copy_host_array_to_device(b,a);return b
    def read(b):
        a=np.empty((rows,n),dtype=np.uint16);copy_device_to_host(host_array_ptr(a),b);return a
    try:
        xd=upload(x);wd=upload(weights)
        od=upload(np.zeros((rows,n),dtype=np.uint16));rd=upload(np.zeros((rows,n),dtype=np.uint16))
        for r in range(rows):
            runtime.check(scalar(xd.ptr+r*k*2,wd.ptr,rd.ptr+r*n*2,1,k,n,waves,None))
        expected=read(rd)
        for _ in range(2):
            runtime.check(batch(xd.ptr,wd.ptr,od.ptr,rows,k,n,waves,None))
            got=read(od)
            np.testing.assert_array_equal(got,expected)
        cpu=bf16_f32(x).astype(np.float64)@reference_weights.astype(np.float64).T
        actual=bf16_f32(got).astype(np.float64)
        def softmax(a):
            e=np.exp(a-a.max(axis=1,keepdims=True));return e/e.sum(axis=1,keepdims=True)
        p,q=softmax(cpu),softmax(actual)
        assert np.max(np.sum(p*(np.log(p)-np.log(q)),axis=1))<=0.05
        assert np.mean(cpu.argmax(axis=1)==actual.argmax(axis=1))>=0.9
    finally:
        for b in reversed(bufs):free(b)
