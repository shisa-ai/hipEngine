"""Strict dense UD projection gates using independently decoded real rows."""
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import malloc, free, copy_host_to_device, copy_device_to_host, host_array_ptr
from hipengine.quant.gguf import bf16_to_float32

FIXTURE = Path(__file__).parent / 'fixtures/gguf_ud'
ENTRIES = json.loads((FIXTURE / 'real_rows.json').read_text())['entries']
CASES = [e for e in ENTRIES if e['type'] in ('IQ4_XS', 'IQ4_NL', 'IQ3_S', 'Q3_K', 'IQ3_XXS', 'IQ2_S')]


def bf16(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


@pytest.mark.parametrize('kwargs', ({'rows': 0}, {'hidden_size': 255}, {'vocab_size': 0},
                                    {'threads': 128}, {'token_ids_ptr': 0}))
def test_q3_embedding_rejects_invalid_args(kwargs, monkeypatch):
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_dense as dense
    monkeypatch.setattr(dense, 'build_gguf_iq_dense', lambda: pytest.fail('built before validation'))
    args = dict(token_ids_ptr=1, qweight_ptr=2, out_ptr=3, rows=1, hidden_size=256, vocab_size=2)
    args.update(kwargs)
    with pytest.raises(ValueError):
        dense.embedding(**args)


def test_q3_embedding_real_rows(library):
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import embedding
    entry = next(e for e in ENTRIES if e['type'] == 'Q3_K')
    with np.load(FIXTURE / 'real_rows.npz') as data:
        raw = np.ascontiguousarray(data[entry['key']+'_raw'])
        weights = data[entry['key']+'_f32']
    ids = np.array([1, 0, 1], dtype=np.int64)
    out = np.zeros((3, weights.shape[1]), dtype=np.uint16)
    buffers = []
    try:
        for array in (ids, raw, out):
            buf = malloc(array.nbytes)
            buffers.append(buf)
            copy_host_to_device(buf, host_array_ptr(array), array.nbytes)
        for _ in range(3):
            embedding(*(b.ptr for b in buffers), 3, weights.shape[1], len(weights), library=library)
            copy_device_to_host(host_array_ptr(out), buffers[2], out.nbytes)
            np.testing.assert_array_equal(out, bf16(weights[ids]))
    finally:
        for buf in reversed(buffers):
            free(buf)


def reference(x, w):
    # Contract: 128 strided accumulators, separate F32 multiply/add, wave32
    # shuffle tree, then four wave sums accumulated serially from zero.
    sums = np.zeros((len(x), len(w), 128), dtype=np.float32)
    for k in range(0, w.shape[1], 128):
        sums += x[:, None, k:k+128] * w[None, :, k:k+128]
    waves = sums.reshape(len(x), len(w), 4, 32)
    for delta in (16, 8, 4, 2, 1):
        old = waves.copy()
        waves[..., :32-delta] += old[..., delta:]
    out = np.zeros((len(x), len(w)), dtype=np.float32)
    for wave in range(4):
        out += waves[..., wave, 0]
    return out


@pytest.fixture(scope='module')
def library():
    try:
        ctypes.CDLL('libamdhip64.so')
    except OSError:
        pytest.skip('HIP runtime is unavailable')
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import build_gguf_iq_dense
    version_file = os.environ.get('HIPENGINE_COMPILER_VERSION_FILE')
    return build_gguf_iq_dense(compiler_version=Path(version_file).read_text() if version_file else None,
                              require_cached=os.environ.get('HIPENGINE_REQUIRE_CACHED_BUILD') == '1')


@pytest.mark.parametrize('backend', ('hip_gfx1100', 'hip_gfx1151'))
def test_dense_registry_strict_keys(backend):
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.registry import KernelKey, is_registered
    # gfx1100's root package declares capabilities; its quant package owns
    # registration. gfx1151's loader aliases those registered source keys.
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import register_gguf_iq_dense_kernels
    register_gguf_iq_dense_kernels()
    load_backend_kernel_package(backend)
    assert is_registered(KernelKey(backend, 'embedding', 'gguf_q3_k', 'lookup_bf16_out'))
    for quant in ('gguf_iq4_xs', 'gguf_iq4_nl', 'gguf_iq3_s', 'gguf_q3_k', 'gguf_iq3_xxs', 'gguf_iq2_s'):
        for output in ('bf16', 'f32'):
            for prefix in ('gemv', 'prefill'):
                assert is_registered(KernelKey(backend, 'linear', quant, f'{prefix}_bf16_{output}_out'))


@pytest.mark.parametrize('kwargs', ({'rows': 0}, {'in_features': 255}, {'out_features': 0},
                                    {'threads': 64}, {'quant': 'unknown'}, {'output': 'fp16'}))
def test_dense_rejects_invalid_shape_before_build(kwargs, monkeypatch):
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_dense as dense
    monkeypatch.setattr(dense, 'build_gguf_iq_dense', lambda: pytest.fail('built before validation'))
    args = dict(rows=1, in_features=256, out_features=3, quant='gguf_iq3_s', output='f32')
    args.update(kwargs)
    with pytest.raises(ValueError):
        dense.launch(1, 2, 3, **args)


@pytest.mark.parametrize('entry', CASES, ids=lambda e: e['model'] + ':' + e['tensor'])
@pytest.mark.parametrize('rows', (1, 3))
@pytest.mark.parametrize('output', ('f32', 'bf16'))
def test_dense_real_rows_exact_and_cpu_outer_gate(library, entry, rows, output):
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import launch
    with np.load(FIXTURE / 'real_rows.npz') as data:
        raw, weights = data[entry['key']+'_raw'], data[entry['key']+'_f32']
    # Odd N proves the output tail; weights retain actual dense K.
    raw = np.ascontiguousarray(raw[[0, 1, 0]])
    weights = weights[[0, 1, 0]]
    x = bf16(np.random.default_rng(27).normal(0, 0.125, (rows, weights.shape[1])))
    expected = reference(bf16_to_float32(x), weights)
    host = np.empty((rows, 3), dtype=np.float32 if output == 'f32' else np.uint16)
    buffers = []
    try:
        for array in (x, raw, host):
            buf = malloc(array.nbytes)
            buffers.append(buf)
            copy_host_to_device(buf, host_array_ptr(array), array.nbytes)
        for _ in range(3):
            launch(buffers[0].ptr, buffers[1].ptr, buffers[2].ptr, rows,
                   weights.shape[1], 3, quant='gguf_'+entry['type'].lower(),
                   output=output, library=library)
            copy_device_to_host(host_array_ptr(host), buffers[2], host.nbytes)
            want = expected if output == 'f32' else bf16(expected)
            np.testing.assert_array_equal(host.view(np.uint32 if output == 'f32' else np.uint16),
                                          want.view(np.uint32 if output == 'f32' else np.uint16))
        teacher = bf16_to_float32(x).astype(np.float64) @ weights.astype(np.float64).T
        actual = host if output == 'f32' else bf16_to_float32(host)
        def probabilities(a):
            e = np.exp(a - a.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        p, q = probabilities(teacher), probabilities(actual.astype(np.float64))
        assert np.max(np.sum(p * (np.log(p) - np.log(q)), axis=1)) <= 0.05
        assert np.mean(teacher.argmax(axis=1) == actual.argmax(axis=1)) >= 0.9
    finally:
        for buf in reversed(buffers):
            free(buf)
