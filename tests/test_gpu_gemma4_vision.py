"""FP32 Gemma image primitives versus independent NumPy equations."""
import ctypes

import numpy as np
import pytest

try:
    ctypes.CDLL('libamdhip64.so')
except OSError:
    pytest.skip('HIP runtime is unavailable', allow_module_level=True)

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import malloc, free, copy_host_to_device, copy_device_to_host, host_array_ptr
from hipengine.kernels.hip_gfx1100.vision.gemma4_vision import (
    gemma4_vision_norm, gemma4_vision_heads, gemma4_vision_softmax,
    gemma4_vision_pool, gemma4_vision_elementwise,
)


@pytest.fixture
def device():
    buffers = []
    try:
        get_hip_runtime().device_synchronize()
    except Exception as exc:
        pytest.skip(f'HIP device unavailable: {exc}')

    def upload(x):
        x = np.ascontiguousarray(x)
        buf = malloc(x.nbytes)
        buffers.append(buf)
        copy_host_to_device(buf, host_array_ptr(x), x.nbytes)
        return buf.ptr

    def download(ptr, shape):
        x = np.empty(shape, dtype=np.float32)
        from hipengine.core.memory import DeviceBuffer
        copy_device_to_host(host_array_ptr(x), DeviceBuffer(ptr, x.nbytes), x.nbytes)
        return x

    yield upload, download
    for buf in reversed(buffers):
        free(buf)


def test_norm_softmax_and_gelu(device):
    upload, download = device
    rng = np.random.default_rng(8)
    x = rng.normal(size=(7, 21)).astype(np.float32)
    w = rng.normal(size=21).astype(np.float32)
    xp, wp, yp = upload(x), upload(w), upload(np.zeros_like(x))
    gemma4_vision_norm(xp, wp, yp, 7, 21, 1e-6)
    expected = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+1e-6) * w
    np.testing.assert_allclose(download(yp, x.shape), expected, atol=2e-6, rtol=2e-6)
    gemma4_vision_norm(xp, 0, yp, 7, 21, 1e-6)
    np.testing.assert_allclose(download(yp, x.shape), expected/w, atol=2e-6, rtol=2e-6)
    gemma4_vision_softmax(xp, 7, 21)
    e = np.exp(x-x.max(axis=-1, keepdims=True))
    np.testing.assert_allclose(download(xp, x.shape), e/e.sum(axis=-1, keepdims=True), atol=1e-7)
    xp = upload(x)
    gemma4_vision_elementwise(xp, 0, yp, x.size, 1)
    expected = .5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*x**3)))
    np.testing.assert_allclose(download(yp, x.shape), expected, atol=3e-7, rtol=2e-6)


def test_xy_neox_rope_and_head_transpose(device):
    upload, download = device
    rng = np.random.default_rng(41)
    x = rng.normal(size=(5, 3, 12)).astype(np.float32)
    pos = np.array([[0,0], [1,0], [2,0], [0,1], [1,1]], dtype=np.int32)
    xp, pp, yp = upload(x), upload(pos), upload(np.zeros_like(x))
    gemma4_vision_heads(xp, pp, yp, 5, 3, 12, 10000., 1)
    expected = x.copy()
    for axis in range(2):
        a = x[..., axis*6:axis*6+3]
        b = x[..., axis*6+3:axis*6+6]
        angle = pos[:,axis,None,None] * 10000.**(-np.arange(3)[None,None,:]/3)
        expected[...,axis*6:axis*6+3] = a*np.cos(angle)-b*np.sin(angle)
        expected[...,axis*6+3:axis*6+6] = b*np.cos(angle)+a*np.sin(angle)
    np.testing.assert_allclose(download(yp, (3,5,12)), expected.transpose(1,0,2), atol=5e-7, rtol=2e-6)
    gemma4_vision_heads(xp, 0, yp, 5, 3, 12, 10000., 0)
    gemma4_vision_heads(yp, 0, xp, 5, 3, 12, 10000., 2)
    np.testing.assert_array_equal(download(xp, x.shape), x)


def test_rectangular_pool_and_standardize(device):
    upload, download = device
    x = np.arange(6*9*7, dtype=np.float32).reshape(6,9,7)/100
    bias = np.arange(7, dtype=np.float32)/10
    scale = np.linspace(.4, 1.1, 7, dtype=np.float32)
    xp,bp,sp,yp = upload(x),upload(bias),upload(scale),upload(np.zeros((6,7), np.float32))
    gemma4_vision_pool(xp,bp,sp,yp,6,9,7,3)
    expected = x.reshape(2,3,3,3,7).mean(axis=(1,3)).reshape(6,7)*np.sqrt(7)
    expected = (expected-bias)*scale
    np.testing.assert_allclose(download(yp,(6,7)),expected,atol=3e-6,rtol=2e-6)


def test_image_text_attention_reads_future_keys_in_its_explicit_mask(device):
    from types import SimpleNamespace
    from hipengine.runtime.gemma4 import _keep_mask, _bf16_bits
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import gemma4_attention_prefill_bf16
    from hipengine.core.memory import DeviceBuffer
    upload,_ = device
    def download(ptr,shape):
        bits = np.empty(shape,dtype=np.uint16)
        copy_device_to_host(host_array_ptr(bits),DeviceBuffer(ptr,bits.nbytes),bits.nbytes)
        return (bits.astype(np.uint32)<<16).view(np.float32)
    q = np.zeros((8,1,256),np.float32)
    k = np.zeros_like(q)
    v = np.repeat(np.arange(8,dtype=np.float32)[:,None,None],256,axis=-1)
    keep = _keep_mask(SimpleNamespace(sliding_window=64),0,8,image_spans=((2,6),))
    qp,kp,vp,mp,yp = upload(_bf16_bits(q)),upload(_bf16_bits(k)),upload(_bf16_bits(v)),upload(keep),upload(np.zeros(q.shape,np.uint16))
    gemma4_attention_prefill_bf16(qp,kp,vp,mp,yp,tokens=8,num_heads=1,num_kv_heads=1,
                                head_dim=256,scale=1.,window=0)
    expected = (keep.astype(np.float32)@v[:,0,:])/keep.sum(axis=1,keepdims=True)
    expected = (_bf16_bits(expected).astype(np.uint32)<<16).view(np.float32)
    np.testing.assert_allclose(download(yp,q.shape)[:,0,:],expected,atol=1e-6)


@pytest.mark.parametrize('grid', [(2,2), (4,6), (6,4)])
def test_tiny_native_tower_matches_numpy_oracle(tmp_path, device, grid):
    from test_unit_gemma4_vision import write_fixture_gguf
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.gemma4_vision_gguf import build_gemma4_vision_gguf_map, read_gemma4_vision_weights
    from hipengine.kernels.cpu_reference.gemma4_vision import (
        gemma4_vision_forward_positions, gemma4_vision_geometry_from_config,
        gemma4_vision_weights_from_slots,
    )
    from hipengine.runtime.gemma4_vision import Gemma4VisionRunner

    path = write_fixture_gguf(tmp_path/'vision.gguf', extra={
        'v.blk.0.out_scale.weight': ((1,), np.array([0.8],np.float32).tobytes()),
    })
    reader = GGUFReader(path)
    model_map = build_gemma4_vision_gguf_map(reader.info)
    c = model_map.config
    rng = np.random.default_rng(80)
    patches = rng.uniform(size=(grid[0]*grid[1],3*c.patch_size**2)).astype(np.float32)
    positions = np.stack(np.meshgrid(np.arange(grid[1]),np.arange(grid[0])),axis=-1).reshape(-1,2)
    geometry = gemma4_vision_geometry_from_config(c)
    weights = gemma4_vision_weights_from_slots(read_gemma4_vision_weights(reader.info, model_map), geometry)
    expected = gemma4_vision_forward_positions(weights,patches,positions,geometry,grid_size=grid).embeddings
    runner = Gemma4VisionRunner(reader,c,backend='hip_gfx1100')
    try:
        actual = runner.forward(patches,positions,grid)
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    finally:
        runner.close()
