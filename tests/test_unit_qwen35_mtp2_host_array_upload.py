"""Owning-array upload regressions for the dense/MoE Qwen3.5 MTP2 modules."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.generation import qwen35_gguf_moe_mtp2 as moe_mtp2
from hipengine.generation import qwen35_gguf_mtp2 as mtp2
from hipengine.generation.qwen35_gguf_moe_mtp2 import Qwen35GGUFMoEMTP2Adapter
from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    (
        "hipengine/generation/qwen35_gguf_mtp2.py",
        "hipengine/generation/qwen35_gguf_moe_mtp2.py",
    ),
)
def test_mtp2_generation_has_no_synchronous_raw_host_pointer_upload(
    relative_path: str,
) -> None:
    tree = ast.parse((_REPO_ROOT / relative_path).read_text())
    raw_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "copy_host_to_device"
    ]
    assert raw_calls == []

    owning_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "copy_host_array_to_device"
    ]
    assert owning_calls
    for call in owning_calls:
        assert len(call.args) >= 2
        source = call.args[1]
        assert not (
            isinstance(source, ast.Call)
            and isinstance(source.func, ast.Name)
            and source.func.id == "host_array_ptr"
        )


def test_stage_ngram_tokens_uploads_the_owning_array(monkeypatch) -> None:
    calls: list[tuple[DeviceBuffer, np.ndarray, int, object]] = []

    def upload(buffer, array, nbytes, *, runtime) -> None:
        calls.append((buffer, array, int(nbytes), runtime))

    monkeypatch.setattr(mtp2, "copy_host_array_to_device", upload)
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._cycle_ngram_tokens = SimpleNamespace(
        ptr=0x1000,
        numel=4,
        device=Device("hip", 0),
    )
    runtime = object()

    tensor = adapter._stage_ngram_tokens((5, 6, 7), runtime=runtime)

    assert (tensor.ptr, tensor.shape, tensor.dtype) == (0x1000, (3,), DType.INT32)
    assert len(calls) == 1
    buffer, array, nbytes, call_runtime = calls[0]
    assert (buffer.ptr, buffer.nbytes, nbytes, call_runtime) == (
        0x1000,
        12,
        12,
        runtime,
    )
    assert array.dtype == np.int32
    assert array.flags.c_contiguous
    assert array.tolist() == [5, 6, 7]


def test_target_pad_token_tensor_uploads_pads_at_the_row_offset(monkeypatch) -> None:
    calls: list[tuple[DeviceBuffer, np.ndarray, int]] = []
    async_calls: list[tuple[object, ...]] = []

    def upload(buffer, array, nbytes, *, runtime) -> None:
        _ = runtime
        calls.append((buffer, array, int(nbytes)))

    monkeypatch.setattr(mtp2, "copy_host_array_to_device", upload)
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._target_pad_token_scratch = DeviceBuffer(0x2000, 32)
    adapter._target_pad_token_capacity = 32
    proposal = SimpleNamespace(
        token_ids=SimpleNamespace(
            numel=2,
            ptr=0x3000,
            device=Device("hip", 0),
        )
    )
    runtime = SimpleNamespace(
        memcpy_async=lambda *args: async_calls.append(tuple(args))
    )

    tensor = adapter._target_pad_token_tensor(proposal, pad_rows=3, runtime=runtime)

    assert (tensor.ptr, tensor.shape, tensor.dtype) == (0x2000, (5,), DType.INT32)
    assert async_calls == [
        (0x2000, 0x3000, 8, HipMemcpyKind.DEVICE_TO_DEVICE, 0)
    ]
    assert len(calls) == 1
    buffer, array, nbytes = calls[0]
    assert (buffer.ptr, buffer.nbytes, nbytes) == (0x2008, 12, 12)
    assert array.dtype == np.int32
    assert array.flags.c_contiguous
    assert array.tolist() == [0, 0, 0]


def test_upload_accept_array_contiguizes_before_the_owning_upload(monkeypatch) -> None:
    calls: list[tuple[DeviceBuffer, np.ndarray, int]] = []

    def upload(buffer, array, nbytes, *, runtime) -> None:
        _ = runtime
        calls.append((buffer, array, int(nbytes)))

    monkeypatch.setattr(mtp2, "copy_host_array_to_device", upload)
    tensor = SimpleNamespace(ptr=0x4000, numel=4, dtype=DType.INT32)
    values = np.arange(6, dtype=np.int32)[::2]
    assert not values.flags.c_contiguous

    Qwen35GGUFMTP2Adapter._upload_accept_array(tensor, values, object())

    assert len(calls) == 1
    buffer, array, nbytes = calls[0]
    assert (buffer.ptr, buffer.nbytes, nbytes) == (0x4000, 12, 12)
    assert array.dtype == np.int32
    assert array.flags.c_contiguous
    assert array.tolist() == [0, 2, 4]


def test_upload_accept_array_rejects_capacity_overflow_before_upload(
    monkeypatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        mtp2,
        "copy_host_array_to_device",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    tensor = SimpleNamespace(ptr=0x4000, numel=4, dtype=DType.INT32)

    with pytest.raises(ValueError, match="exceeds tensor capacity"):
        Qwen35GGUFMTP2Adapter._upload_accept_array(
            tensor,
            np.arange(5, dtype=np.int32),
            object(),
        )

    assert calls == []


def test_moe_upload_accept_array_uses_the_owning_upload(monkeypatch) -> None:
    calls: list[tuple[DeviceBuffer, np.ndarray, int]] = []

    def upload(buffer, array, nbytes, *, runtime) -> None:
        _ = runtime
        calls.append((buffer, array, int(nbytes)))

    monkeypatch.setattr(moe_mtp2, "copy_host_array_to_device", upload)
    tensor = SimpleNamespace(ptr=0x5000, numel=3, dtype=DType.INT32)
    values = np.arange(6, dtype=np.int32)[::2]

    Qwen35GGUFMoEMTP2Adapter._upload_accept_array(tensor, values, object())

    assert len(calls) == 1
    buffer, array, nbytes = calls[0]
    assert (buffer.ptr, buffer.nbytes, nbytes) == (0x5000, 12, 12)
    assert array.dtype == np.int32
    assert array.flags.c_contiguous
    assert array.tolist() == [0, 2, 4]
