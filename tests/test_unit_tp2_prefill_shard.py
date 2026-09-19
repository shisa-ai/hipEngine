"""CPU ownership/shape tests for the batched rank-local MLP shard.

No device and no model file: a fake runtime records allocations and D2D copies,
and the linear/SiLU launchers are recorded. These tests pin the batched-prefill
contract (row-sized buffers, active-row tracking, rows*hidden input copies, the
fused-pair attempt at batched rows with the unfused GEMM as its fallback,
unchanged single-row fused decode, and a preflight that refuses an unsupported
batched quant/dtype route before launch).
"""
import numpy as np
import pytest

import hipengine.distributed.shard_exec as shard_exec
import hipengine.runtime.gguf_linear as gguf_linear
from hipengine.distributed.shard_exec import MlpShardRank, MlpShardRankError


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

    def memcpy(self, dst, src, nbytes, kind):
        self.memcpys.append((int(dst), int(src), int(nbytes), -1))

    def stream_synchronize(self, stream):
        return None


class FakeAllocation:
    def __init__(self):
        self.name = "fake"
        self.freed = False

    def free(self):
        self.freed = True


class FakeSpec:
    def __init__(self, layout, quant_key):
        self.layout = layout
        self.quant_key = quant_key


class FakeWeight:
    def __init__(self, layout="dense_bf16", quant_key="dense_bf16"):
        self.spec = FakeSpec(layout, quant_key)
        self._allocation = FakeAllocation()

    def allocation(self, name=None):
        return self._allocation


def _weights(layout="dense_bf16", quant_key="dense_bf16"):
    return {
        role: FakeWeight(layout, quant_key)
        for role in ("ffn_gate", "ffn_up", "ffn_down")
    }


def _rank(runtime, *, rows=1, mlp_decode_variant=None, weights=None):
    return MlpShardRank(
        runtime,
        device=0,
        stream=7,
        weights=weights or _weights(),
        hidden=8,
        per_rank_ffn=4,
        partial_dtype="f32",
        mlp_decode_variant=mlp_decode_variant,
        rows=rows,
    )


def _record_launchers(monkeypatch, *, pair_launches: bool = True):
    launches = []

    def fake_linear(weight, x_ptr, out_ptr, rows, in_features, out_features, **kwargs):
        launches.append({"kind": "linear", "rows": rows, "in": in_features,
                         "out": out_features, "use_gemv_decode": kwargs.get("use_gemv_decode"),
                         "output_dtype": kwargs.get("output_dtype")})

    def fake_pair(a, b, x_ptr, out_ptr, rows, in_features, out_features, **kwargs):
        launches.append({"kind": "pair", "rows": rows, "in": in_features,
                         "out": out_features,
                         "use_gemv_decode": kwargs.get("use_gemv_decode")})
        return pair_launches

    def fake_silu(gate_ptr, up_ptr, out_ptr, rows, features, **kwargs):
        launches.append({"kind": "silu", "rows": rows, "features": features})

    monkeypatch.setattr(gguf_linear, "launch_gguf_linear", fake_linear)
    monkeypatch.setattr(gguf_linear, "launch_gguf_linear_pair_silu", fake_pair)
    monkeypatch.setattr(shard_exec, "silu_mul_separate_out_bf16", fake_silu)
    monkeypatch.setattr(MlpShardRank, "_require_batched_route", lambda self, rows: None)
    return launches


def test_capacity_buffers_are_sized_for_the_batch():
    runtime = FakeRuntime()
    rank = _rank(runtime, rows=4)
    sizes = sorted(runtime.allocated.values())
    assert 4 * 8 * 2 in sizes
    assert sizes.count(4 * 4 * 2) == 3
    assert 4 * 8 * 4 in sizes
    assert rank.rows == 4
    assert rank.active_rows == 1


def test_write_input_accepts_multi_row_payload_and_sets_active_rows():
    runtime = FakeRuntime()
    rank = _rank(runtime, rows=4)
    rank.write_input(np.zeros(3 * 8 * 2, dtype=np.uint8))
    assert rank.active_rows == 3
    rank.write_input(np.zeros(8 * 2, dtype=np.uint8))
    assert rank.active_rows == 1


def test_write_input_rejects_a_partial_row_payload():
    runtime = FakeRuntime()
    rank = _rank(runtime, rows=4)
    with pytest.raises(ValueError, match="whole number of"):
        rank.write_input(np.zeros(8 * 2 - 1, dtype=np.uint8))
    with pytest.raises(ValueError, match="rows"):
        rank.write_input(np.zeros(5 * 8 * 2, dtype=np.uint8))
    with pytest.raises(ValueError, match="declared"):
        rank.write_input(np.zeros(2 * 8 * 2, dtype=np.uint8), rows=3)


def test_write_input_from_device_defaults_to_active_rows():
    runtime = FakeRuntime()
    rank = _rank(runtime, rows=4)
    rank.write_input_from_device(0xABCD, rows=3)
    assert runtime.memcpys[-1][2] == 3 * 8 * 2
    assert rank.active_rows == 3
    rank.write_input_from_device(0xABCD)  # re-stage the active count
    assert runtime.memcpys[-1][2] == 3 * 8 * 2
    rank.write_input_from_device(0xABCD, rows=1)
    assert runtime.memcpys[-1][2] == 1 * 8 * 2
    assert rank.active_rows == 1


def test_batched_forward_attempts_the_fused_pair_when_a_variant_is_set(monkeypatch):
    """A batched rank with an admitted variant runs the fused pair.

    The fused prefill owner is bit-identical to the unfused chain and faster in
    every row band, so batched rows attempt it; the decline path is covered by
    the fallback test below.
    """

    runtime = FakeRuntime()
    launches = _record_launchers(monkeypatch)
    rank = _rank(runtime, rows=4, mlp_decode_variant="dense_dual_local32_bf16_bf16_out")
    rank.write_input_from_device(0x1, rows=4)
    ptr = rank.forward_partial(rows=4)
    assert ptr == rank.down_partial_ptr
    kinds = [entry["kind"] for entry in launches]
    assert kinds == ["pair", "linear"]
    assert all(entry["rows"] == 4 for entry in launches)
    pair = next(entry for entry in launches if entry["kind"] == "pair")
    assert pair["use_gemv_decode"] is False
    assert launches[-1]["output_dtype"] == "f32"


def test_batched_forward_falls_back_to_the_unfused_chain_when_the_pair_declines(
    monkeypatch,
):
    """A declined batched pair falls back instead of failing.

    Only some widths are admitted, so at batched rows the fused pair is an
    attempt. The unfused chain is the registered strict fallback the fused owner
    is bit-identical to, and it must keep the same rows and dtype contract.
    """

    runtime = FakeRuntime()
    launches = _record_launchers(monkeypatch, pair_launches=False)
    rank = _rank(runtime, rows=4, mlp_decode_variant="dense_dual_local32_bf16_bf16_out")
    rank.write_input_from_device(0x1, rows=4)
    ptr = rank.forward_partial(rows=4)
    assert ptr == rank.down_partial_ptr
    kinds = [entry["kind"] for entry in launches]
    assert kinds == ["pair", "linear", "linear", "silu", "linear"]
    linears = [entry for entry in launches if entry["kind"] == "linear"]
    assert all(entry["rows"] == 4 for entry in linears)
    assert all(entry["use_gemv_decode"] is False for entry in linears)
    assert launches[-1]["output_dtype"] == "f32"


def test_batched_forward_uses_unfused_gemm_without_an_admitted_variant(monkeypatch):
    """A rank the policy does not admit keeps the unfused chain, and never
    guesses a fused variant for batched rows."""

    runtime = FakeRuntime()
    launches = _record_launchers(monkeypatch)
    rank = _rank(runtime, rows=4)
    rank.write_input_from_device(0x1, rows=4)
    ptr = rank.forward_partial(rows=4)
    assert ptr == rank.down_partial_ptr
    kinds = [entry["kind"] for entry in launches]
    assert kinds == ["linear", "linear", "silu", "linear"]
    assert all(entry["rows"] == 4 for entry in launches)
    linears = [entry for entry in launches if entry["kind"] == "linear"]
    assert all(entry["use_gemv_decode"] is False for entry in linears)
    assert launches[-1]["output_dtype"] == "f32"


def test_single_row_pair_decline_is_still_an_error(monkeypatch):
    """The fallback is batched-only: at one row the fused route *was* resolved
    for this exact shape, so a decline is a real error rather than a shape the
    admission excludes."""

    runtime = FakeRuntime()
    _record_launchers(monkeypatch, pair_launches=False)
    rank = _rank(runtime, rows=1, mlp_decode_variant="dense_dual_local32_bf16_bf16_out")
    with pytest.raises(RuntimeError, match="did not launch"):
        rank.forward_partial()


def test_forward_rejects_an_active_row_mismatch_before_launch(monkeypatch):
    runtime = FakeRuntime()
    launches = _record_launchers(monkeypatch)
    rank = _rank(runtime, rows=4)
    rank.write_input_from_device(0x1, rows=3)
    with pytest.raises(ValueError, match="does not match"):
        rank.forward_partial(rows=2)
    assert launches == [], "no launch happens on a row-count mismatch"


def test_single_row_forward_preserves_the_fused_decode_route(monkeypatch):
    runtime = FakeRuntime()
    launches = _record_launchers(monkeypatch)
    rank = _rank(runtime, rows=1, mlp_decode_variant="dense_dual_local32_bf16_bf16_out")
    rank.forward_partial()
    assert [entry["kind"] for entry in launches] == ["pair", "linear"]
    assert all(entry["rows"] == 1 for entry in launches)
    assert all(entry["use_gemv_decode"] is True for entry in launches)


def test_rows_values_must_be_real_positive_ints():
    runtime = FakeRuntime()
    rank = _rank(runtime, rows=2)
    for bad in (True, 1.5, "2"):
        with pytest.raises(ValueError, match="integer"):
            rank.write_input_from_device(0x1, rows=bad)
        with pytest.raises(ValueError, match="integer"):
            rank.forward_partial(rows=bad)
    with pytest.raises(ValueError, match="positive"):
        rank.forward_partial(rows=0)
    with pytest.raises(ValueError, match="capacity"):
        rank.forward_partial(rows=3)
    with pytest.raises(ValueError, match="integer"):
        MlpShardRank(runtime, device=0, stream=0, weights=_weights(), hidden=8,
                     per_rank_ffn=4, rows=True)
    with pytest.raises(ValueError, match="integer"):
        MlpShardRank(runtime, device=0, stream=0, weights=_weights(), hidden=8,
                     per_rank_ffn=4, rows=1.5)
    with pytest.raises(ValueError, match="positive"):
        MlpShardRank(runtime, device=0, stream=0, weights=_weights(), hidden=8,
                     per_rank_ffn=4, rows=0)


def test_read_partial_and_read_input_respect_active_rows():
    runtime = FakeRuntime()
    rank = _rank(runtime, rows=4)
    rank.write_input_from_device(0x1, rows=3)
    assert rank.read_partial().shape == (3, 8)
    assert rank.read_input().shape == (3 * 8 * 2,)
    with pytest.raises(ValueError, match="does not match"):
        rank.read_partial(rows=2)
    rank.write_input_from_device(0x1, rows=1)
    assert rank.read_partial().shape == (8,)


def test_batched_route_preflight_accepts_a_registered_quant():
    runtime = FakeRuntime()
    rank = _rank(runtime, rows=4)
    rank._require_batched_route(4)  # dense_bf16 prefill_out is registered


def test_batched_route_preflight_admits_the_f32_down_partial_route():
    runtime = FakeRuntime()
    # 0e4966906 declared the gguf_q4_k_t16_v1 bf16 -> f32 dispatch surface that
    # the f32 down-partial route needs at multi-row prefill, so this is the
    # admitted case, not a refusal: the preflight resolves rows=4 through
    # t16_wmma_prefill_bf16_f32_out. Kept as the counterpart of the refusal
    # below so the pair reads as one contract.
    rank = _rank(runtime, rows=4, weights=_weights("gguf_q4_k_t16_v1", "gguf_q4_k_t16_v1"))
    rank._require_batched_route(4)


def test_batched_route_preflight_blocks_an_unsupported_dtype_route():
    runtime = FakeRuntime()
    # Q5_K t16 has no f32-output dispatch surface: the preflight must refuse
    # before any launch rather than stage a multi-row buffer into a bad route.
    rank = _rank(runtime, rows=4, weights=_weights("gguf_q5_k_t16_v1", "gguf_q5_k_t16_v1"))
    with pytest.raises(MlpShardRankError, match="no batched"):
        rank._require_batched_route(4)


def test_close_frees_every_buffer_and_weight_once():
    runtime = FakeRuntime()
    weights = _weights()
    rank = _rank(runtime, rows=2, weights=weights)
    allocated = set(runtime.allocated)
    rank.close()
    assert set(runtime.freed) == allocated
    assert runtime.allocated == {}
    assert all(w.allocation().freed for w in weights.values())
    rank.close()  # idempotent
    assert set(runtime.freed) == allocated
