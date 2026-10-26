"""CPU behavior coverage for the Qwen3.5 GGUF MTP NextN ``eh_proj`` wrapper.

``qwen35_gguf_mtp_eh_proj_f32`` runs two RMSNorms, a device-side concat and one
quantized GEMV. The concat is where this wrapper has been wrong: the Q6_K
branch read an undefined host array, and the two whole-array D2D copies it did
issue produced ``[e_norm; h_norm]`` (``[2*rows, hidden]``) instead of the
``[rows, 2*hidden]`` rows the GEMV reads. One row hides the layout defect, so
every case here runs rows 1, 2 and 3; the Q6_K branch failed at any row count.

A fake HIP boundary keeps the copy traffic, the concat layout and the Q6_K
staging visible: device memory is a byte mirror, ``runtime.memcpy`` records its
kind, and the fake kernels evaluate the mirror through the same
``cpu_reference`` dequantized oracle the GPU test uses. Nothing here touches
ROCm, and no test asserts on a real device pointer value.
"""

from __future__ import annotations

import ast
import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.kernels.cpu_reference.ops import gguf_quant_gemv, rmsnorm
from hipengine.kernels.hip_gfx1100.speculative import mtp_nextn as mod
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import make_q6_k_weight, make_q8_0_weight

# 2 * HIDDEN == QK_K, so one shape serves both block formats: Q6_K needs a
# multiple of 256 input features and Q8_0 a multiple of 32.
HIDDEN = 128
EPS = 1e-6
_Q6_K = GGMLQuantizationType.Q6_K
_Q8_0 = GGMLQuantizationType.Q8_0


class _FakeHip:
    """Byte-accurate stand-in for the HIP boundary of one wrapper call."""

    def __init__(self) -> None:
        self.mem: dict[int, np.ndarray] = {}
        self.arrays: dict[int, np.ndarray] = {}
        self.uploads: list[tuple[DeviceBuffer, np.ndarray, int | None, object]] = []
        self.raw_copies: list[tuple[DeviceBuffer, int, int | None, object]] = []
        self.device_copies: list[tuple[int, int, int, object]] = []
        self.mallocs: list[DeviceBuffer] = []
        self.frees: list[DeviceBuffer] = []
        self.cache_buffers: list[DeviceBuffer] = []
        self.cached_names: list[str] = []
        self.norm_calls: list[tuple[int, int, int, int, int]] = []
        self.cast_calls: list[tuple[int, int, int]] = []
        self.cast_outputs: list[np.ndarray] = []
        self.gemv_calls: list[tuple[str, int, int, int, int, int]] = []
        self.gemv_inputs: list[np.ndarray] = []
        self.syncs = 0
        self.fail_gemv = False
        self.runtime: object | None = None
        self._next_ptr = 0x1_0000_0000

    # ── device memory ───────────────────────────────────────────────────
    def _slice(self, ptr: int, nbytes: int) -> np.ndarray:
        base = max(candidate for candidate in self.mem if candidate <= ptr)
        offset = ptr - base
        assert offset + nbytes <= self.mem[base].size, "access outside a device allocation"
        return self.mem[base][offset : offset + nbytes]

    def view(self, ptr: int, dtype, count: int) -> np.ndarray:
        dtype = np.dtype(dtype)
        return np.frombuffer(self._slice(ptr, count * dtype.itemsize), dtype=dtype)

    def view_write(self, ptr: int, values: np.ndarray) -> None:
        values = np.ascontiguousarray(values)
        self._slice(ptr, values.nbytes)[:] = values.reshape(-1).view(np.uint8)

    def malloc(self, nbytes, *, runtime=None):
        size = int(nbytes)
        buffer = DeviceBuffer(self._next_ptr, size)
        self._next_ptr += size + 0x1000
        self.mem[buffer.ptr] = np.zeros(size, dtype=np.uint8)
        self.mallocs.append(buffer)
        return buffer

    def free(self, buffer, *, runtime=None):
        assert buffer.ptr in self.mem, "free of an unknown device pointer"
        assert buffer not in self.frees, "double free"
        del self.mem[buffer.ptr]
        self.frees.append(buffer)

    # ── copies ──────────────────────────────────────────────────────────
    def copy_host_array_to_device(self, buffer, array, nbytes=None, *, runtime=None):
        count = int(array.nbytes) if nbytes is None else int(nbytes)
        self.uploads.append((buffer, array, nbytes, runtime))
        source = np.ascontiguousarray(array).reshape(-1).view(np.uint8)
        self._slice(buffer.ptr, count)[:] = source[:count]

    def copy_host_to_device(self, buffer, host_ptr, nbytes=None, *, runtime=None):
        count = buffer.nbytes if nbytes is None else int(nbytes)
        self.raw_copies.append((buffer, host_ptr, nbytes, runtime))
        self._slice(buffer.ptr, count)[:] = np.frombuffer(
            ctypes.string_at(host_ptr, count), dtype=np.uint8
        )

    def copy_device_to_host(self, host_ptr, buffer, nbytes=None, *, runtime=None):
        count = buffer.nbytes if nbytes is None else int(nbytes)
        ctypes.memmove(host_ptr, self._slice(buffer.ptr, count).ctypes.data, count)

    def memcpy(self, dst, src, nbytes, kind):
        self.device_copies.append((dst, src, int(nbytes), kind))
        self._slice(dst, int(nbytes))[:] = self._slice(src, int(nbytes))

    def device_synchronize(self) -> None:
        self.syncs += 1

    # ── kernels ─────────────────────────────────────────────────────────
    def rmsnorm_f32(
        self, x_ptr, weight_ptr, out_ptr, rows, hidden, eps=1e-6, *, stream=0,
        library=None, runtime=None,
    ):
        self.norm_calls.append((x_ptr, weight_ptr, out_ptr, rows, hidden))
        x = self.view(x_ptr, np.float32, rows * hidden).reshape(rows, hidden)
        weight = self.view(weight_ptr, np.float32, hidden)
        self.view_write(out_ptr, rmsnorm(x, weight, eps))

    def cached_upload(self, name, data, *, runtime=None):
        buffer = self.malloc(data.nbytes, runtime=runtime)
        array = np.ascontiguousarray(data)
        self.copy_host_array_to_device(buffer, array, buffer.nbytes, runtime=runtime)
        self.cache_buffers.append(buffer)
        self.cached_names.append(name)
        self.arrays[buffer.ptr] = array
        return buffer

    def q8_0_gemv(
        self, x_ptr, qweight_ptr, out_ptr, rows, in_features, out_features, *,
        library=None, runtime=None,
    ):
        self.gemv_calls.append(
            ("q8_0", x_ptr, qweight_ptr, out_ptr, rows, in_features, out_features)
        )
        self._quantized_gemv(_Q8_0, np.float32, x_ptr, qweight_ptr, out_ptr, rows,
                             in_features, out_features)

    def q6_k_gemv(
        self, x_ptr, qweight_ptr, out_ptr, rows, in_features, out_features, *,
        library=None, runtime=None,
    ):
        self.gemv_calls.append(
            ("q6_k", x_ptr, qweight_ptr, out_ptr, rows, in_features, out_features)
        )
        # The fp16 ABI is the point: read the staging buffer as fp16, not f32.
        self._quantized_gemv(_Q6_K, np.float16, x_ptr, qweight_ptr, out_ptr, rows,
                             in_features, out_features)

    def _quantized_gemv(self, qtype, dtype, x_ptr, qweight_ptr, out_ptr, rows,
                        in_features, out_features):
        if self.fail_gemv:
            raise RuntimeError(f"{qtype.name} eh_proj gemv failed")
        # Snapshot the activations the kernel reads, before the wrapper frees
        # them: this is the layout and the dtype the GEMV actually consumed.
        x = self.view(x_ptr, dtype, rows * in_features).astype(np.float32)
        self.gemv_inputs.append(x.reshape(rows, in_features).copy())
        out = gguf_quant_gemv(
            x.reshape(rows, in_features), self.arrays[qweight_ptr], qtype
        )
        assert out.shape == (rows, out_features)
        self.view_write(out_ptr, out)

    def f32_to_fp16(self, x_ptr, out_ptr, count, *, stream=0, library=None, runtime=None):
        self.cast_calls.append((x_ptr, out_ptr, count))
        values = self.view(x_ptr, np.float32, count).astype(np.float16)
        self.cast_outputs.append(values.copy())
        self.view_write(out_ptr, values)


def _install(monkeypatch, hip: _FakeHip) -> _FakeHip:
    from hipengine.kernels.hip_gfx1100.convert import cast
    from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv, gguf_q6_k_pack8_gemv

    runtime = SimpleNamespace(memcpy=hip.memcpy, device_synchronize=hip.device_synchronize)
    hip.runtime = runtime
    monkeypatch.setattr(mod, "malloc", hip.malloc)
    monkeypatch.setattr(mod, "free", hip.free)
    # ``raising=False`` on both names: the raw helper disappears and the owning
    # one becomes a module-level import as part of this unit, and the harness
    # must describe the same boundary before and after that migration.
    monkeypatch.setattr(mod, "copy_host_to_device", hip.copy_host_to_device, raising=False)
    monkeypatch.setattr(
        mod, "copy_host_array_to_device", hip.copy_host_array_to_device, raising=False
    )
    monkeypatch.setattr(mod, "copy_device_to_host", hip.copy_device_to_host)
    monkeypatch.setattr(mod, "get_hip_runtime", lambda: runtime)
    monkeypatch.setattr(mod, "build_mtp_nextn", lambda **kwargs: object())
    monkeypatch.setattr(mod, "mtp_rmsnorm_f32", hip.rmsnorm_f32)
    monkeypatch.setattr(
        mod,
        "mtp_eh_proj_f32",
        lambda *args, **kwargs: pytest.fail("F32 eh_proj GEMV ran on a quantized route"),
    )
    monkeypatch.setattr(mod, "_cached_upload", hip.cached_upload)
    monkeypatch.setattr(gguf_k_gemv, "build_gguf_k_gemv", lambda **kwargs: object())
    monkeypatch.setattr(gguf_k_gemv, "gguf_q8_0_gemv_f32_f32_out", hip.q8_0_gemv)
    monkeypatch.setattr(
        gguf_q6_k_pack8_gemv, "gguf_q6_k_pack8_gemv_decode_fp16_f32_out", hip.q6_k_gemv
    )
    monkeypatch.setattr(cast, "f32_to_fp16", hip.f32_to_fp16)
    return runtime


def _synthetic_weight(qtype, hidden: int) -> np.ndarray:
    if qtype is _Q6_K:
        return make_q6_k_weight(hidden, 2 * hidden)
    return make_q8_0_weight(hidden, 2 * hidden)


def _inputs(seed: int, rows: int):
    rng = np.random.default_rng(seed)
    return {
        "hidden_seed": rng.normal(0.0, 0.5, (rows, HIDDEN)).astype(np.float32),
        "token_embedding": rng.normal(0.0, 0.5, (rows, HIDDEN)).astype(np.float32),
        "hnorm_weight": rng.normal(1.0, 0.1, HIDDEN).astype(np.float32),
        "enorm_weight": rng.normal(1.0, 0.1, HIDDEN).astype(np.float32),
    }


def _uploaded_ptr(hip: _FakeHip, array: np.ndarray) -> int:
    matches = [buffer.ptr for buffer, uploaded, _, _ in hip.uploads if uploaded is array]
    assert len(matches) == 1, f"expected exactly one upload of the given array, got {len(matches)}"
    return matches[0]


def _assert_every_buffer_freed(hip: _FakeHip) -> None:
    cached = {buffer.ptr for buffer in hip.cache_buffers}
    expected = sorted(buffer.ptr for buffer in hip.mallocs if buffer.ptr not in cached)
    assert sorted(buffer.ptr for buffer in hip.frees) == expected
    # Only the deliberately cached weight buffers survive the call.
    assert set(hip.mem) == cached


@pytest.mark.parametrize(
    "qtype", [_Q8_0, _Q6_K], ids=["q8_0", "q6_k"]
)
@pytest.mark.parametrize("rows", [1, 2, 3])
def test_quantized_route_stages_row_interleaved_concat(monkeypatch, qtype, rows):
    """The GEMV reads ``[rows, 2*hidden]`` rows ``[e_norm[r], h_norm[r]]``.

    Checked against the dequantized ``cpu_reference`` oracle, so a concat that
    only agrees at ``rows == 1`` fails at rows 2 and 3.
    """
    hip = _FakeHip()
    _install(monkeypatch, hip)
    inputs = _inputs(0xE4 + 7 * rows + int(qtype), rows)
    weight = _synthetic_weight(qtype, HIDDEN)

    actual = mod.qwen35_gguf_mtp_eh_proj_f32(
        inputs["hidden_seed"],
        inputs["token_embedding"],
        weight,
        inputs["hnorm_weight"],
        inputs["enorm_weight"],
        eh_proj_qtype=qtype,
        eps=EPS,
    )

    e_norm = rmsnorm(inputs["token_embedding"], inputs["enorm_weight"], EPS)
    h_norm = rmsnorm(inputs["hidden_seed"], inputs["hnorm_weight"], EPS)
    fused = np.concatenate([e_norm, h_norm], axis=-1)
    staged = fused.astype(np.float16).astype(np.float32) if qtype is _Q6_K else fused
    np.testing.assert_allclose(
        actual, gguf_quant_gemv(staged, weight, qtype), rtol=1e-6, atol=1e-6
    )

    embed_ptr = _uploaded_ptr(hip, inputs["token_embedding"])
    hidden_ptr = _uploaded_ptr(hip, inputs["hidden_seed"])
    assert [call[0] for call in hip.norm_calls] == [embed_ptr, hidden_ptr]
    e_norm_ptr = hip.norm_calls[0][2]
    h_norm_ptr = hip.norm_calls[1][2]

    gemv = hip.gemv_calls[0]
    assert gemv[0] == ("q6_k" if qtype is _Q6_K else "q8_0")
    assert (gemv[4], gemv[5], gemv[6]) == (rows, 2 * HIDDEN, HIDDEN)

    # The concat is built on the device, row by row, and every copy is D2D.
    concat_ptr = hip.cast_calls[0][0] if qtype is _Q6_K else gemv[1]
    row_bytes = HIDDEN * 4
    expected_copies = []
    for row in range(rows):
        dst = concat_ptr + row * 2 * row_bytes
        expected_copies.append(
            (dst, e_norm_ptr + row * row_bytes, row_bytes, HipMemcpyKind.DEVICE_TO_DEVICE)
        )
        expected_copies.append(
            (dst + row_bytes, h_norm_ptr + row * row_bytes, row_bytes,
             HipMemcpyKind.DEVICE_TO_DEVICE)
        )
    assert hip.device_copies == expected_copies

    if qtype is _Q6_K:
        # One device cast of the whole concat, and the GEMV reads that buffer:
        # the wrapper's existing fp16 input ABI is preserved rather than
        # silently re-routed.
        assert len(hip.cast_calls) == 1
        cast_src, cast_dst, cast_count = hip.cast_calls[0]
        assert (cast_src, cast_count) == (concat_ptr, rows * 2 * HIDDEN)
        assert gemv[1] == cast_dst
        staged = hip.cast_outputs[0].reshape(rows, 2 * HIDDEN)
        np.testing.assert_array_equal(staged, fused.astype(np.float16))
        np.testing.assert_array_equal(
            hip.gemv_inputs[0], fused.astype(np.float16).astype(np.float32)
        )
    else:
        assert hip.cast_calls == []
        assert gemv[1] == concat_ptr
        np.testing.assert_array_equal(hip.gemv_inputs[0], fused)

    # Both activations reach the device through the owning helper, sized by the
    # former destination default, and never through a raw host pointer.
    assert hip.raw_copies == []
    assert [buffer.ptr for buffer, _, _, _ in hip.uploads if buffer.ptr in
            (embed_ptr, hidden_ptr)] == [embed_ptr, hidden_ptr]
    for buffer, array, nbytes, runtime in hip.uploads:
        assert runtime is hip.runtime
        if buffer.ptr in (embed_ptr, hidden_ptr):
            assert nbytes == buffer.nbytes == rows * HIDDEN * 4
    _assert_every_buffer_freed(hip)
    assert hip.syncs == 1


@pytest.mark.parametrize("qtype", [_Q8_0, _Q6_K], ids=["q8_0", "q6_k"])
def test_gemv_failure_frees_every_staging_buffer(monkeypatch, qtype):
    """A launch failure must not leak the concat or the fp16 staging buffer."""
    hip = _FakeHip()
    hip.fail_gemv = True
    _install(monkeypatch, hip)
    inputs = _inputs(0x51 + int(qtype), 2)

    with pytest.raises(RuntimeError, match="gemv failed"):
        mod.qwen35_gguf_mtp_eh_proj_f32(
            inputs["hidden_seed"],
            inputs["token_embedding"],
            _synthetic_weight(qtype, HIDDEN),
            inputs["hnorm_weight"],
            inputs["enorm_weight"],
            eh_proj_qtype=qtype,
            eps=EPS,
        )

    _assert_every_buffer_freed(hip)


def test_module_has_no_synchronous_raw_host_pointer_upload() -> None:
    """Every array upload in the module goes through the owning helper.

    The owning call must either pass the former destination byte count
    explicitly or upload into a buffer freshly sized from the source (the
    weight cache), so no site silently changes the copied length.
    """
    tree = ast.parse(Path(mod.__file__).read_text())

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
    assert len(owning_calls) >= 20, f"expected the module's uploads, found {len(owning_calls)}"
    for call in owning_calls:
        source = call.args[1]
        assert not (
            isinstance(source, ast.Call)
            and isinstance(source.func, ast.Name)
            and source.func.id == "host_array_ptr"
        ), f"line {call.lineno}: raw host pointer upload"
        explicit_size = len(call.args) >= 3
        fresh_source_sized = (
            isinstance(source, ast.Call)
            and (
                (isinstance(source.func, ast.Name) and source.func.id == "ascontiguousarray")
                or (
                    isinstance(source.func, ast.Attribute)
                    and source.func.attr == "ascontiguousarray"
                )
            )
        )
        assert explicit_size or fresh_source_sized, (
            f"line {call.lineno}: upload neither passes the destination size "
            "nor targets a source-sized buffer"
        )
