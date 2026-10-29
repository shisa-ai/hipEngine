"""Retire obsolete dense/shared T16 comparison seams without losing dispatch."""

import ctypes
import inspect
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading import laguna_gguf_materialize as materializer
from hipengine.runtime.gguf_linear import launch_gguf_linear_pair_silu
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from scripts import laguna_long_context_profile as profiler
from tests._laguna_synthetic import laguna_tensors, make_laguna_info


@pytest.mark.parametrize("suffix", ["sidecar", "dual-interleaved"])
def test_retired_t16_comparison_rejects_cli_and_has_no_session_setter(monkeypatch, suffix):
    monkeypatch.setattr(sys, "argv", ["laguna_long_context_profile.py", "--compare-q4-decode-t16-" + suffix])
    with pytest.raises(SystemExit) as error:
        profiler._parse_args()
    assert error.value.code == 2
    name = "q4_decode_t16_" + suffix.replace("-", "_")
    assert "compare_" + name not in profiler.COMPARISON_ARGUMENTS
    assert not hasattr(LagunaGGUFResidentSession, "set_" + name)
    assert "use_" + name in inspect.signature(LagunaGGUFResidentSession).parameters


class _Runtime:
    def __init__(self):
        self.buffers = {}
        self.next_ptr = 0x10000

    def malloc(self, nbytes):
        ptr = self.next_ptr
        self.next_ptr += nbytes + 4096
        self.buffers[ptr] = bytearray(nbytes)
        return ptr

    def free(self, ptr):
        del self.buffers[ptr]

    def memcpy(self, dst, src, nbytes, kind):
        assert kind == HipMemcpyKind.HOST_TO_DEVICE
        self.buffers[dst][:nbytes] = ctypes.string_at(src, nbytes)


@pytest.mark.parametrize("sidecar,paired,expected", [
    (True, True, "t16_dual_interleaved_sidecar_decode_bf16_bf16_out"),
    (True, False, "pack8_dual_decode_bf16_bf16_out"),
    (False, True, "pack8_dual_decode_bf16_bf16_out"),
])
def test_materialized_dense_pair_keeps_default_and_pack8_rollback(sidecar, paired, expected):
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    rng = np.random.default_rng(355)
    arrays = {
        "blk.1.ffn_gate_shexp.weight": rng.integers(0, 256, (1024, 1728), dtype=np.uint8),
        "blk.1.ffn_up_shexp.weight": rng.integers(0, 256, (1024, 1728), dtype=np.uint8),
    }
    for array in arrays.values():
        blocks = array.reshape(1024, 12, 144)
        blocks[..., :4] = np.asarray([0, 60, 0, 60], dtype=np.uint8)
    reader = SimpleNamespace(
        info=make_laguna_info(tensors=laguna_tensors()),
        tensor_data=arrays.__getitem__,
    )
    runtime = _Runtime()
    weights = materializer.materialize_laguna_gguf_weights(
        reader, selected_slots=("layers.1.ffn_gate_shexp", "layers.1.ffn_up_shexp"),
        context_length=4096, available_bytes=120 * 2**30, runtime=runtime,
        backend="hip_gfx1151",
    )
    keys = [KernelKey("hip_gfx1151", "linear_pair_silu", "gguf_q4_k", variant) for variant in (
        "pack8_dual_decode_bf16_bf16_out", "t16_sidecar_dual_decode_bf16_bf16_out",
        "t16_dual_interleaved_sidecar_decode_bf16_bf16_out",
    )]
    originals = {key: resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant) for key in keys}
    calls = []
    try:
        for key in keys:
            register(key, lambda *a, _key=key, **kw: calls.append((_key.variant, a, kw)), replace=True)
        gate = weights.layer(1).weight("ffn_gate_shexp")
        up = weights.layer(1).weight("ffn_up_shexp")
        assert "decode_tiles_dual" in gate.allocations
        assert "decode_tiles" not in gate.allocations and "decode_tiles" not in up.allocations
        assert launch_gguf_linear_pair_silu(
            gate, up, x_ptr=100, out_ptr=200, rows=1, in_features=3072, out_features=1024,
            backend="hip_gfx1151", stream=7, runtime=runtime,
            use_gemv_decode=True, use_q4_t16_sidecar=sidecar,
            use_q4_t16_dual_interleaved=paired,
        )
        assert [call[0] for call in calls] == [expected]
        args, kwargs = calls[0][1:]
        assert args[0] == 100 and args[-4:] == (200, 1, 3072, 1024)
        assert kwargs == {"stream": 7, "runtime": runtime}
        if expected.startswith("t16_dual"):
            assert args[1] == gate.allocation("decode_tiles_dual").tensor.ptr
        else:
            assert args[1:7] == tuple(
                weight.allocation(name).tensor.ptr
                for weight in (gate, up) for name in ("qweight", "scales", "mins")
            )
    finally:
        for key, fn in originals.items():
            register(key, fn, replace=True)
        weights.free(runtime=runtime)
    assert not runtime.buffers
