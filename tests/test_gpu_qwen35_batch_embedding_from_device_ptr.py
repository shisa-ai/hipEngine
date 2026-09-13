"""Gate for device-token-fed batch embedding (C3.0b piece A).

``Qwen35ParoResidentSession._set_batch_token_embeddings_from_ptr`` gathers the
batch token embeddings straight from a device int64 token buffer (e.g.
``batch_lm_out_index``) with no host token list / host->device copy / host
bounds check, so it is safe inside a captured c>1 decode graph.

The test exercises the gather against a duck-typed stand-in (tiny fp16
embedding table on a real HIP device, no 35B model) and asserts the result is
both byte-identical to the host-fed ``_set_batch_token_embeddings`` path and an
exact match for the numpy reference gather ``embedding[token_ids]``.
"""

from __future__ import annotations

import ctypes
import types

import numpy as np
import pytest


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")

_VOCAB = 16
_HIDDEN = 8
_MAX_BATCH = 8


def _make_stub(embedding_fp16: np.ndarray):
    from hipengine.core.device import Device
    from hipengine.core.dtype import DType
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc
    from hipengine.core.tensor import Tensor
    from hipengine.kernels.hip_gfx1100.runtime import build_runtime_state

    runtime = get_hip_runtime()
    device = Device("hip", 0)
    buffers: list = []

    emb_buf = malloc(embedding_fp16.nbytes, runtime=runtime)
    copy_host_to_device(emb_buf, host_array_ptr(embedding_fp16), embedding_fp16.nbytes, runtime=runtime)
    buffers.append(emb_buf)

    batch_hidden = malloc(_MAX_BATCH * _HIDDEN * DType.FP16.itemsize, runtime=runtime)
    runtime.memset(batch_hidden.ptr, 0, batch_hidden.nbytes)
    buffers.append(batch_hidden)

    token_id_buf = malloc(_MAX_BATCH * DType.INT64.itemsize, runtime=runtime)
    buffers.append(token_id_buf)

    stub = types.SimpleNamespace(
        runtime=runtime,
        device=device,
        max_batch_size=_MAX_BATCH,
        vocab_size=_VOCAB,
        config=types.SimpleNamespace(hidden_size=_HIDDEN),
        embedding=types.SimpleNamespace(tensor=Tensor.from_handle(emb_buf.ptr, embedding_fp16.shape, DType.FP16, device)),
        batch_hidden=batch_hidden,
        token_id_buf=token_id_buf,
        libraries={"runtime_state": build_runtime_state(load=True)},
        buffers=buffers,
    )
    return stub


def _read_hidden(stub, rows: int) -> np.ndarray:
    from hipengine.core.dtype import DType
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    out = np.empty((rows, _HIDDEN), dtype=np.float16)
    copy_device_to_host(
        host_array_ptr(out),
        DeviceBuffer(stub.batch_hidden.ptr, rows * _HIDDEN * DType.FP16.itemsize),
        runtime=stub.runtime,
    )
    return out


def test_embedding_from_device_ptr_matches_host_and_numpy() -> None:
    from hipengine.core.dtype import DType
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc
    from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession

    rng = np.random.default_rng(1234)
    embedding = rng.standard_normal((_VOCAB, _HIDDEN)).astype(np.float16)
    token_ids = np.asarray([3, 7, 0, 15], dtype=np.int64)
    rows = int(token_ids.size)
    stub = _make_stub(embedding)

    # Device-token-fed gather (piece A).
    tok_buf = malloc(rows * DType.INT64.itemsize, runtime=stub.runtime)
    stub.buffers.append(tok_buf)
    copy_host_to_device(tok_buf, host_array_ptr(token_ids), token_ids.nbytes, runtime=stub.runtime)
    Qwen35ParoResidentSession._set_batch_token_embeddings_from_ptr(stub, tok_buf.ptr, rows=rows)
    stub.runtime.device_synchronize()
    from_ptr = _read_hidden(stub, rows)

    # Reset, then host-fed gather for the same tokens.
    stub.runtime.memset(stub.batch_hidden.ptr, 0, stub.batch_hidden.nbytes)
    Qwen35ParoResidentSession._set_batch_token_embeddings(stub, tuple(int(t) for t in token_ids))
    stub.runtime.device_synchronize()
    host_fed = _read_hidden(stub, rows)

    reference = embedding[token_ids]

    # Raw 16-bit gather => bit-identical to numpy reference and to the host path.
    np.testing.assert_array_equal(from_ptr.view(np.uint16), reference.view(np.uint16))
    np.testing.assert_array_equal(from_ptr.view(np.uint16), host_fed.view(np.uint16))


def test_embedding_from_device_ptr_rejects_bad_rows() -> None:
    from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession

    embedding = np.zeros((_VOCAB, _HIDDEN), dtype=np.float16)
    stub = _make_stub(embedding)
    with pytest.raises(ValueError):
        Qwen35ParoResidentSession._set_batch_token_embeddings_from_ptr(stub, stub.token_id_buf.ptr, rows=0)
    with pytest.raises(ValueError):
        Qwen35ParoResidentSession._set_batch_token_embeddings_from_ptr(
            stub, stub.token_id_buf.ptr, rows=_MAX_BATCH + 1
        )
