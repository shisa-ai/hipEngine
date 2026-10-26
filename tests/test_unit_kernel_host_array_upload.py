"""Kernel host wrappers must hand synchronous NumPy uploads to the owning helper.

``copy_host_array_to_device`` retains the source array through the copy and
checks both bounds; ``copy_host_to_device(buf, host_array_ptr(array))`` erases
the owner, so a later asynchronous upload or refactor can read freed memory.
These tests pin the reviewed kernel modules to the owning helper.

The VibeVoice site is exercised by driving ``q4_prefill`` with an empty layer
list: its entry validation, arena scratch request, position/count staging and
final context publication run without any kernel, model or hipBLASLt problem.
The Gemma 4 scratch map and the dense IQ metadata tail run through a recording
fake runtime. All three sites are therefore exercised on CPU, and each test
asserts the upload that actually happens (owning helper, pointer, byte count,
runtime and payload), not just the helper name.

The syntax guard below is bounded: it covers the reviewed modules, not arbitrary
pointer lifetimes. Direct ``runtime.memcpy`` calls whose source is not an array
pointer, asynchronous uploads, and D2H/D2D copies stay outside it.

``kernels/hip_gfx1100/speculative/mtp_nextn`` is deliberately absent; its
uploads and its Q6_K eh_proj branch are a separate unit.
"""

from __future__ import annotations

import ast
import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core import memory
from hipengine.core.memory import DeviceBuffer
from hipengine.core.runtime import MemcpyKind

KERNEL_MODULES = (
    "kernels/hip_gfx1100/vibevoice/q4",
    "kernels/hip_gfx1100/gemma4/gemma4_experts",
    "kernels/hip_gfx1100/quant/gguf_iq_source_mmq_prefill",
)


def _is_array_pointer_expression(node: ast.AST) -> bool:
    """Return whether ``node`` is a raw address taken from a host array."""

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "host_array_ptr":
            return True
        if isinstance(node.func, ast.Name) and node.func.id == "int":
            return bool(node.args) and _is_array_pointer_expression(node.args[0])
        if isinstance(node.func, ast.Attribute) and node.func.attr == "addressof":
            return bool(node.args) and _is_array_pointer_expression(node.args[0])
        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"data_as", "ctypes"}
        )
    if isinstance(node, ast.Attribute) and node.attr == "data":
        owner = node.value
        return isinstance(owner, ast.Attribute) and owner.attr == "ctypes"
    return False


def _argument(node: ast.Call, position: int, name: str) -> ast.AST | None:
    if len(node.args) > position:
        return node.args[position]
    return next((kw.value for kw in node.keywords if kw.arg == name), None)


def _host_to_device(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "HOST_TO_DEVICE"


def _array_pointer_hazards(tree: ast.AST) -> list[tuple[int, str]]:
    """Return synchronous H2D uploads whose source owner is only a raw pointer."""

    hazards: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "copy_host_to_device":
            source = _argument(node, 1, "host_ptr")
            if source is not None and _is_array_pointer_expression(source):
                hazards.append((node.lineno, "copy_host_to_device"))
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "memcpy"):
            continue
        if not _host_to_device(_argument(node, 3, "kind")):
            continue
        source = _argument(node, 1, "src")
        if source is not None and _is_array_pointer_expression(source):
            hazards.append((node.lineno, "runtime.memcpy"))
    return sorted(hazards)


@pytest.mark.parametrize("module", KERNEL_MODULES)
def test_reviewed_kernel_modules_use_the_owning_upload_helper(module: str) -> None:
    path = Path(__file__).resolve().parents[1] / "hipengine" / f"{module}.py"
    hazards = _array_pointer_hazards(ast.parse(path.read_text()))
    assert not hazards, f"{path.name}: use copy_host_array_to_device at {hazards}"


@pytest.fixture
def owning_uploads(monkeypatch):
    """Record every owning-helper call, then run the real helper."""

    calls = []
    real = memory.copy_host_array_to_device

    def spy(buffer, array, nbytes=None, *, runtime=None):
        calls.append((buffer, array, nbytes, runtime))
        return real(buffer, array, nbytes, runtime=runtime)

    monkeypatch.setattr(memory, "copy_host_array_to_device", spy)
    return calls


class _RecordingRuntime:
    """Minimal HIP runtime surface: records the copies it is asked for."""

    def __init__(self) -> None:
        self.copies: list[tuple[int, int, int, int]] = []

    def memcpy(self, dst: int, src: int, nbytes: int, kind: int) -> None:
        self.copies.append((int(dst), int(src), int(nbytes), int(kind)))


def test_gemma4_compact_to_source_upload_uses_the_owning_helper(
    monkeypatch, owning_uploads
) -> None:
    """The identity lane map is written once through the owning helper."""

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts

    copies = []

    def record_copy(buffer, host_ptr, nbytes=None, *, runtime=None):
        copies.append(
            (buffer.ptr, host_ptr, buffer.nbytes if nbytes is None else nbytes)
        )

    monkeypatch.setattr(memory, "copy_host_to_device", record_copy)
    monkeypatch.setattr(
        gemma4_experts, "malloc", lambda nbytes: DeviceBuffer(0x4000, int(nbytes))
    )
    scratch = gemma4_experts.Gemma4ExpertScratch(
        tokens=4, top_k=2, hidden_size=8, intermediate=16, num_experts=4
    )

    buffer = scratch.buffer("compact_to_source")

    assert buffer.ptr == 0x4000
    assert len(owning_uploads) == 1
    uploaded_buffer, array, nbytes, runtime = owning_uploads[0]
    assert uploaded_buffer is buffer
    assert isinstance(array, np.ndarray) and array.dtype == np.int64
    np.testing.assert_array_equal(array, np.arange(scratch.total_lanes))
    assert nbytes is None and runtime is None
    # The source size, not the destination size, is what reaches the device.
    assert copies == [(0x4000, array.ctypes.data, 64)]

    # Reuse keeps the cached buffer and does not re-upload.
    assert scratch.buffer("compact_to_source") is buffer
    assert len(owning_uploads) == 1
    assert len(copies) == 1


def test_dense_iq_metadata_upload_uses_the_owning_helper(owning_uploads) -> None:
    """The dense IQ metadata tail uploads once per row count, at fixed offsets."""

    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as module

    runtime = _RecordingRuntime()
    workspace = module.IQDenseMMQWorkspace(ptr=0x20000, nbytes=0x4000)
    base = 0x20000 + 0x4000 - module._METADATA_NBYTES
    assert workspace.metadata_ptr() == base

    compact, start, tile, total = module._upload_dense_metadata(workspace, 512, runtime)

    assert (compact, start, tile) == (base, base + 16, base + 32)
    assert total == 512
    assert workspace._uploaded_rows == 512
    assert len(owning_uploads) == 1
    buffer, payload, nbytes, passed_runtime = owning_uploads[0]
    assert isinstance(buffer, DeviceBuffer)
    assert (buffer.ptr, buffer.nbytes) == (base, module._METADATA_NBYTES)
    assert passed_runtime is runtime
    assert nbytes == payload.nbytes
    assert runtime.copies == [
        (base, payload.ctypes.data, payload.nbytes, int(MemcpyKind.HOST_TO_DEVICE))
    ]
    expected = np.concatenate(
        (
            np.array([0, 512], dtype=np.int64),
            np.array([0, 512], dtype=np.int64),
            np.zeros(4, dtype=np.int64),
        )
    )
    observed = np.frombuffer(
        ctypes.string_at(runtime.copies[0][1], payload.nbytes), dtype=np.int64
    )
    np.testing.assert_array_equal(observed, expected)

    # Same row count: no second upload.
    module._upload_dense_metadata(workspace, 512, runtime)
    assert len(owning_uploads) == 1
    assert len(runtime.copies) == 1

    # New row count: exactly one more upload, same offsets.
    module._upload_dense_metadata(workspace, 1024, runtime)
    assert len(owning_uploads) == 2
    assert owning_uploads[1][0].ptr == base
    assert workspace._uploaded_rows == 1024
    assert len(runtime.copies) == 2

    # Oversize refusal is unchanged and leaves the cached row count alone.
    with pytest.raises(ValueError, match="exceeds its workspace tail"):
        module._upload_dense_metadata(workspace, 131_200, runtime)
    assert workspace._uploaded_rows == 1024
    assert len(owning_uploads) == 2


def test_vibevoice_q4_prefill_publishes_owning_context_length(
    monkeypatch, owning_uploads
) -> None:
    """The ``q4_prefill`` tail publishes the new context length from the owner.

    ``layers`` is empty so the test exercises the entry validation, the arena
    scratch request, the position/count staging and the final context
    publication without any kernel, model or hipBLASLt problem.
    """

    from hipengine.kernels.hip_gfx1100.vibevoice import q4

    legacy = []

    def record_copy(buffer, host_ptr, nbytes=None, *, runtime=None):
        legacy.append(
            (buffer.ptr, host_ptr, buffer.nbytes if nbytes is None else nbytes)
        )

    monkeypatch.setattr(memory, "copy_host_to_device", record_copy)

    runner = SimpleNamespace(
        spec=SimpleNamespace(
            hidden_size=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            intermediate_size=8,
            rms_norm_eps=1e-6,
        ),
        runtime=None,
        max_context=16,
        layers=(),
        _validate_position=lambda start: None,
        _ctx_len=DeviceBuffer(0x8100, 8),
        _ctx_len_host=np.zeros(1, dtype=np.int64),
    )
    scratch = []

    def take(sizes):
        views = [
            DeviceBuffer(0x8000 + 0x100 * index, int(size))
            for index, size in enumerate(sizes)
        ]
        scratch.extend(views)
        return views

    runner._prefill_scratch = SimpleNamespace(take=take)
    runner._spans = lambda positions, counts, rows: object()
    hidden_rows = DeviceBuffer(0x8200, 4 * 4 * 2)

    # The rejected window must stage and publish nothing.
    with pytest.raises(ValueError, match="prefill exceeds max_context"):
        q4.q4_prefill(runner, hidden_rows, 14, 3)
    assert owning_uploads == []
    assert legacy == []
    assert runner._ctx_len_host.tolist() == [0]

    q4.q4_prefill(runner, hidden_rows, 4, 3)

    positions, counts = scratch[0], scratch[1]
    assert [
        (call[0].ptr, call[1].nbytes if call[2] is None else call[2], call[3])
        for call in owning_uploads
    ] == [
        (positions.ptr, 4 * 8, None),
        (counts.ptr, 4 * 8, None),
        (0x8100, 8, None),
    ]
    position_array, count_array, context_array = (call[1] for call in owning_uploads)
    assert position_array.tolist() == [3, 4, 5, 6]
    assert count_array.tolist() == [4, 5, 6, 7]
    # The publication retains the runner-owned host scalar and passes the
    # former destination-sized default explicitly.
    assert context_array is runner._ctx_len_host
    assert owning_uploads[2][2] == runner._ctx_len.nbytes == 8
    assert runner._ctx_len_host.tolist() == [7]
    # No raw pointer upload reached the device; every copy ran through the
    # owning helper with the default runtime.
    assert legacy == [
        (positions.ptr, position_array.ctypes.data, 32),
        (counts.ptr, count_array.ctypes.data, 32),
        (0x8100, context_array.ctypes.data, 8),
    ]

    # A second window re-stages positions/counts and republishes the tail.
    q4.q4_prefill(runner, hidden_rows, 2, 8)
    assert len(owning_uploads) == 6
    second_positions, second_counts = scratch[22], scratch[23]
    assert owning_uploads[3][0] is second_positions
    assert owning_uploads[3][1].tolist() == [8, 9]
    assert owning_uploads[4][0] is second_counts
    assert owning_uploads[4][1].tolist() == [9, 10]
    assert owning_uploads[5][0] is runner._ctx_len
    assert owning_uploads[5][1] is runner._ctx_len_host
    assert owning_uploads[5][2] == 8
    assert owning_uploads[5][3] is None
    assert runner._ctx_len_host.tolist() == [10]
    assert len(legacy) == 6
