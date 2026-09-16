"""CPU ownership/shape tests for the batched rank-local MLP shard.

No device and no model file: a fake runtime records allocations and D2D copies,
and the linear/SiLU launchers are recorded. These tests pin the batched-prefill
contract (row-sized buffers, rows*hidden input copies, unfused GEMM for rows>1,
unchanged single-row fused decode) before any GPU validation.
"""
import numpy as np
import pytest

import hipengine.distributed.shard_exec as shard_exec
import hipengine.runtime.gguf_linear as gguf_linear
from hipengine.distributed.shard_exec import MlpShardRank


class FakeRuntime:
    def __init__(self):
        self.device = 0
        self.allocated = {}
        self.next_ptr = 0x1000
        self.memcpys = []
        self.freed = []

    def get_device(self):
        return self.device

    def set_device(self, device):
        self.device = int(device)

    def malloc(self, nbytes):
        ptr = self.next_ptr
        self.next_ptr += int(nbytes) + 16
        self.allocated[ptr] = int(nbytes)
        return ptr

    def free(self, ptr):
        self.freed.append(int(ptr))
        self.allocated.pop(int(ptr), None)

    def memcpy_async(self, dst, src, nbytes, kind, stream):
        self.memcpys.append((int(dst), int(src), int(nbytes), int(stream)))

    def stream_synchronize(self, stream):
        return None


class FakeAllocation:
    def __init__(self):
        self.freed = False

    def free(self):
        self.freed = True


class FakeWeight:
    def __init__(self):
        self._allocation = FakeAllocation()

    def allocation(self, name=None):
        return self._allocation


def _weights():
    return {role: FakeWeight() for role in ("ffn_gate", "ffn_up", "ffn_down")}


def _rank(runtime, *, rows=1, mlp_decode_variant=None):
    return MlpShardRank(
        runtime,
        device=0,
        stream=7,
        weights=_weights(),
        hidden=8,
        per_rank_ffn=4,
        partial_dtype="f32",
        mlp_decode_variant=mlp_decode_variant,
        rows=rows,
    )


def _record_launchers(monkeypatch):
    launches = []

    def fake_linear(weight, x_ptr, out_ptr, rows, in_features, out_features, **kwargs):
        launches.append({"kind": "linear", "rows": rows, "in": in_features,
                         "out": out_features, "use_gemv_decode": kwargs.get("use_gemv_decode"),
                         "output_dtype": kwargs.get("output_dtype")})

    def fake_pair(a, b, x_ptr, out_ptr, rows, in_features, out_features, **kwargs):
        launches.append({"kind": "pair", "rows": rows, "use_gemv_decode": kwargs.get("use_gemv_decode")})
        return True

    def fake_silu(gate_ptr, up_ptr, out_ptr, rows, features, **kwargs):
        launches.append({"kind": "silu", "rows": rows, "features": features})

    monkeypatch.setattr(gguf_linear, "launch_gguf_linear", fake_linear)
    monkeypatch.setattr(gguf_linear, "launch_gguf_linear_pair_silu", fake_pair)
    monkeypatch.setattr(shard_exec, "silu_mul_separate_out_bf16", fake_silu)
    return launches


def test_rows_buffers_are_sized_for_the_batch():
    runtime = FakeRuntime()
    rank = _rank(runtime, rows=4)
    sizes = sorted(runtime.allocated.values())
    # x, gate, up, act (bf16: 4*hidden*2, 4*ffn*2) and down partial (f32: 4*hidden*4)
    assert 4 * 8 * 2 in sizes
    assert sizes.count(4 * 4 * 2) == 3
    assert 4 * 8 * 4 in sizes
    assert rank.rows == 4


def test_write_input_rows_copies_rows_times_hidden_bytes():
    runtime = FakeRuntime()
    rank = _rank(runtime, rows=4)
    rank.write_input_from_device(0xABCD, rows=3)
    assert runtime.memcpys[-1][2] == 3 * 8 * 2
    rank.write_input_from_device(0xABCD)
    assert runtime.memcpys[-1][2] == 4 * 8 * 2


def test_batched_forward_uses_unfused_gemm_with_rows(monkeypatch):
    runtime = FakeRuntime()
    launches = _record_launchers(monkeypatch)
    rank = _rank(runtime, rows=4, mlp_decode_variant="dense_dual_local32_bf16_bf16_out")
    ptr = rank.forward_partial(rows=4)
    assert ptr == rank.down_partial_ptr
    kinds = [entry["kind"] for entry in launches]
    assert kinds == ["linear", "linear", "silu", "linear"]
    assert all(entry["rows"] == 4 for entry in launches)
    linears = [entry for entry in launches if entry["kind"] == "linear"]
    assert all(entry["use_gemv_decode"] is False for entry in linears)
    assert launches[-1]["output_dtype"] == "f32"


def test_single_row_forward_preserves_the_fused_decode_route(monkeypatch):
    runtime = FakeRuntime()
    launches = _record_launchers(monkeypatch)
    rank = _rank(runtime, rows=1, mlp_decode_variant="dense_dual_local32_bf16_bf16_out")
    rank.forward_partial()
    assert [entry["kind"] for entry in launches] == ["pair", "linear"]
    assert all(entry["rows"] == 1 for entry in launches)
    assert all(entry["use_gemv_decode"] is True for entry in launches)


def test_rows_bounds_are_rejected():
    runtime = FakeRuntime()
    rank = _rank(runtime, rows=2)
    with pytest.raises(ValueError):
        rank.write_input_from_device(0x1, rows=3)
    with pytest.raises(ValueError):
        rank.forward_partial(rows=0)
    with pytest.raises(ValueError):
        MlpShardRank(runtime, device=0, stream=0, weights=_weights(), hidden=8,
                     per_rank_ffn=4, rows=0)


def test_close_frees_every_buffer_and_weight_once():
    runtime = FakeRuntime()
    rank = _rank(runtime, rows=2)
    allocated = set(runtime.allocated)
    rank.close()
    assert set(runtime.freed) == allocated
    assert runtime.allocated == {}
    rank.close()  # idempotent
    assert set(runtime.freed) == allocated
