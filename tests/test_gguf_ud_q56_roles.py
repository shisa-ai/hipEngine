"""Optional published-weight Q5/Q6 gates; sampled N, actual role K.

Set HIPENGINE_UD_ROLE_MODEL to the published K_M GGUF. These tests qualify
raw leaves, not loader repacks, full-N geometry, or NextN state semantics.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.quant.gguf import dequantize_gguf_data, bf16_to_float32


# SHA256 of concatenated first/middle/last/first/second compressed rows.
ROW_HASHES = {
    "blk.0.attn_gate.weight": "cd6f171cd7913e9f5a5da5ab7fc9a07e6aee13fea30d3fd32abf2a3badc05977",
    "blk.0.ssm_out.weight": "2d5381ffac7dcf1e242520dad2da2729b0aa104f99aaf72bc67fd3c342d34d9a",
    "blk.1.ssm_out.weight": "491eb80a18da3865a126fc07dac6392760c6bd1d17d9dc034aa878e40456efdd",
    "blk.11.attn_k.weight": "397257aa678fc209198a7cb2a15dc426aaed81d0da6c1448a55caeeaff40a339",
    "blk.24.ffn_down.weight": "90da161e1477e7064fc43fd05b10c870ccff8a6ba77dfdae9481e0ab13050513",
    "blk.63.ffn_down.weight": "61d77099f662e962773e4fd05f98704f37786631e1301407b4319df7eceb02a9",
    "blk.64.nextn.eh_proj.weight": "9802bf5a438d2e0d356ce9acdf32afba02524a2cc97d3d82ac91f2969dba5fd1",
}


@pytest.fixture(scope="module")
def model():
    path = Path(os.environ.get("HIPENGINE_UD_ROLE_MODEL", "/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf"))
    if not path.is_file():
        pytest.skip("published UD K_M model unavailable")
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime unavailable")
    from hipengine.loading.gguf import load_gguf_index
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import build_gguf_k_gemv
    return path, {t.name: t for t in load_gguf_index(path).tensors}, build_gguf_k_gemv(load=True)


def bf16(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


@pytest.mark.parametrize("name,quant,width", (
    ("blk.0.attn_gate.weight", "Q5_K", 5120),
    ("blk.0.ssm_out.weight", "Q5_K", 6144),
    ("blk.24.ffn_down.weight", "Q5_K", 17408),
    ("blk.11.attn_k.weight", "Q6_K", 5120),
    ("blk.1.ssm_out.weight", "Q6_K", 6144),
    ("blk.63.ffn_down.weight", "Q6_K", 17408),
    ("blk.64.nextn.eh_proj.weight", "Q6_K", 10240),
))
@pytest.mark.parametrize("rows", (1, 8, 32))
@pytest.mark.parametrize("output", ("f32", "bf16"))
def test_raw_role_rows(model, name, quant, width, rows, output):
    from hipengine.core.memory import malloc, free, copy_host_to_device, copy_device_to_host, host_array_ptr
    from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv
    path, tensors, library = model
    tensor = tensors[name]
    assert tensor.ggml_type_name == quant and tensor.shape[1] == width
    row_bytes = tensor.nbytes // tensor.shape[0]
    # Include a real first/middle/last row plus repeated rows and odd-N tail.
    indices = [0, tensor.shape[0] // 2, tensor.shape[0] - 1, 0, 1]
    with path.open("rb") as f:
        payloads = []
        for index in indices:
            f.seek(tensor.data_offset + index * row_bytes)
            payloads.append(f.read(row_bytes))
    assert all(len(p) == row_bytes for p in payloads)
    payload = b"".join(payloads)
    assert hashlib.sha256(payload).hexdigest() == ROW_HASHES[name]
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(5, row_bytes).copy()
    weights = dequantize_gguf_data(raw, tensor.ggml_type).reshape(5, width)
    x = bf16(np.random.default_rng(731).normal(0, 0.125, (rows, width)))
    teacher = bf16_to_float32(x).astype(np.float64) @ weights.astype(np.float64).T
    dtype = np.float32 if output == "f32" else np.uint16
    host = np.full((rows + 2, 5), 123, dtype=dtype)
    fn = getattr(gguf_k_gemv, "gguf_" + quant.lower() + "_gemv_bf16_" + output + "_out")
    buffers = []
    try:
        for array in (x, raw, host):
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        def launch(row, count):
            fn(buffers[0].ptr + row * x.strides[0], buffers[1].ptr,
               buffers[2].ptr + (row + 1) * host.strides[0], count, width, 5, library=library)
        launch(0, rows)
        copy_device_to_host(host_array_ptr(host), buffers[2], host.nbytes)
        original = host.copy()
        for _ in range(2):
            host[1:-1].view(np.uint8).fill(0xff)
            copy_host_to_device(buffers[2], host_array_ptr(host), host.nbytes)
            for row in range(rows):
                launch(row, 1)
            copy_device_to_host(host_array_ptr(host), buffers[2], host.nbytes)
            np.testing.assert_array_equal(host.view(np.uint8), original.view(np.uint8))
        np.testing.assert_array_equal(host[[0, -1]], np.full((2, 5), 123, dtype=dtype))
        actual = host[1:-1] if output == "f32" else bf16_to_float32(host[1:-1])
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, teacher,
                                   rtol=2e-5 if output == "f32" else 0.004,
                                   atol=2e-4)
        def logsoft(a):
            a = a - a.max(axis=1, keepdims=True)
            return a - np.log(np.exp(a).sum(axis=1, keepdims=True))
        p, q = logsoft(teacher), logsoft(actual.astype(np.float64))
        assert np.max(np.sum(np.exp(p) * (p - q), axis=1)) <= 0.05
        assert np.mean(teacher.argmax(axis=1) == actual.argmax(axis=1)) >= 0.9
    finally:
        for buffer in reversed(buffers):
            free(buffer)
