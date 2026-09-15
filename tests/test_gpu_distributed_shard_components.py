"""Two-GPU integration: the runtime shard components against a CPU oracle.

This is the TP2-A correctness contract re-run through the runtime components
(``hipengine.distributed.shard_exec`` + ``hipengine.distributed.staged``)
instead of the slice script's ad-hoc device code: each rank uploads a dense
bf16 shard, runs the unfused chain into a persistent f32 partial, and the
staged exchange reduces both partials into every rank's device buffer. The
oracle is plain numpy arithmetic over the same host weights, with bf16
rounding exactly where the device chain rounds.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _device_count() -> int:
    if not _hip_available():
        return 0
    from hipengine.core.hip import get_hip_runtime

    try:
        return int(get_hip_runtime().device_count())
    except Exception:  # noqa: BLE001 - no usable device
        return 0


needs_two_gpus = pytest.mark.skipif(
    _device_count() < 2,
    reason="requires two HIP devices (W7900 + RX 7900 XTX target host)",
)


def _bf16_round(values: np.ndarray) -> np.ndarray:
    """Round f32 to bf16 precision by truncation through the f16-style mantissa."""

    bits = values.astype("<f4").view("<u4")
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.view("<f4")


def _bf16_bytes(values: np.ndarray) -> np.ndarray:
    """bf16-rounded values as little-endian bytes: the top half of each f32."""

    bits = _bf16_round(values).view("<u4")
    return (bits >> 16).astype("<u2").view(np.uint8).reshape(*values.shape, 2)


def _bf16_values_from_bytes(payload: np.ndarray) -> np.ndarray:
    """The f32 view of bf16 bytes (the value the device actually reads)."""

    u16 = np.ascontiguousarray(payload).view("<u2")
    return (u16.astype("<u4") << 16).view("<f4")


def _silu(values: np.ndarray) -> np.ndarray:
    return values / (1.0 + np.exp(-values.astype(np.float64))).astype(np.float32)


@needs_two_gpus
def test_two_rank_shard_chain_matches_the_cpu_oracle() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.runtime import MemcpyKind
    from hipengine.distributed.shard_exec import MlpShardRank, upload_shard_weight
    from hipengine.distributed.staged import StagedExchangeTransport

    runtime = get_hip_runtime()
    devices = (0, 1)
    from hipengine.core.device import scoped_current_device

    streams = {}
    for d in devices:
        # A stream belongs to the device that is current when it is created;
        # creating both under device 0 would give rank 1 an invalid handle.
        with scoped_current_device(runtime, d):
            streams[d] = runtime.stream_create(nonblocking=True)
    rng = np.random.default_rng(7)
    hidden, per_rank_ffn = 64, 32
    scale = 0.05

    gate = (rng.standard_normal((per_rank_ffn, hidden)) * scale).astype(np.float32)
    up = (rng.standard_normal((per_rank_ffn, hidden)) * scale).astype(np.float32)
    down = (rng.standard_normal((hidden, per_rank_ffn)) * scale).astype(np.float32)
    gate_bf16 = _bf16_bytes(gate)
    up_bf16 = _bf16_bytes(up)
    down_bf16 = _bf16_bytes(down)
    x_f32 = (rng.standard_normal(hidden) * scale).astype(np.float32)
    x_bf16 = _bf16_round(x_f32)
    x_bf16_bytes = _bf16_bytes(x_f32).reshape(-1)

    # The CPU contract: bf16 in, bf16 gate/up matmuls, SiLU-multiply in f32,
    # rounded back to bf16 (the device act buffer is bf16), then the f32 down
    # partial per rank. All matmuls run on the same bf16-value views the
    # device reads. The reduced sum is what one all-reduce would see.
    gate_v = _bf16_values_from_bytes(gate_bf16)
    up_v = _bf16_values_from_bytes(up_bf16)
    down_v = _bf16_values_from_bytes(down_bf16)
    partials = []
    for rank in range(2):
        g = (x_bf16 @ gate_v[rank * 16 : (rank + 1) * 16].T).astype(np.float32)
        u = (x_bf16 @ up_v[rank * 16 : (rank + 1) * 16].T).astype(np.float32)
        act = _bf16_round(_silu(g) * u)
        partials.append((act @ down_v[:, rank * 16 : (rank + 1) * 16].T).astype(np.float32))
    expected = partials[0] + partials[1]

    ranks = []
    transports = None
    try:
        for device in devices:
            weights = {
                "ffn_gate": upload_shard_weight(
                    runtime,
                    device=device,
                    name="raw",
                    layout="dense_bf16",
                    quant_key="dense_bf16",
                    payload=gate_bf16[device * 16 : (device + 1) * 16],
                ),
                "ffn_up": upload_shard_weight(
                    runtime,
                    device=device,
                    name="raw",
                    layout="dense_bf16",
                    quant_key="dense_bf16",
                    payload=up_bf16[device * 16 : (device + 1) * 16],
                ),
                "ffn_down": upload_shard_weight(
                    runtime,
                    device=device,
                    name="raw",
                    layout="dense_bf16",
                    quant_key="dense_bf16",
                    payload=down_bf16[:, device * 16 : (device + 1) * 16],
                ),
            }
            ranks.append(
                MlpShardRank(
                    runtime,
                    device=device,
                    stream=streams[device],
                    weights=weights,
                    hidden=hidden,
                    per_rank_ffn=per_rank_ffn // 2,
                )
            )
        transports = StagedExchangeTransport(
            runtime, devices=devices, streams=streams, hidden=hidden
        )

        # Both ranks enqueue their chains before either is awaited.
        partial_ptrs = {}
        for device, rank in zip(devices, ranks):
            rank.write_input(x_bf16_bytes)
            partial_ptrs[device] = rank.forward_partial()
        reduced = transports.reduce(partial_ptrs)

        for device in devices:
            out = np.empty(hidden, dtype="<f4")
            with scoped_current_device(runtime, device):
                runtime.memcpy(
                    out.ctypes.data,
                    reduced[device],
                    hidden * 4,
                    MemcpyKind.DEVICE_TO_HOST,
                )
            # The bf16 contract has ~3 significant digits; accumulation-order
            # noise adds a small tolerance on top.
            assert np.abs(out - expected).max() < 4e-2, (
                f"rank {device}'s reduced output left the bf16 contract: "
                f"max diff {np.abs(out - expected).max():.4g}"
            )
    finally:
        if transports is not None:
            transports.close()
        for rank in ranks:
            rank.close()
        for stream in streams.values():
            runtime.stream_destroy(stream)


@needs_two_gpus
def test_shard_rank_stages_its_input_from_a_device_row() -> None:
    """write_input_from_device is a same-device D2D copy with host staging."""

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.runtime import MemcpyKind
    from hipengine.distributed.shard_exec import MlpShardRank, upload_shard_weight

    runtime = get_hip_runtime()
    device = 1  # the second physical GPU, not the process default
    with scoped_current_device(runtime, device):
        stream = runtime.stream_create(nonblocking=True)
    weights = {
        role: upload_shard_weight(
            runtime,
            device=device,
            name=f"raw.{role}",
            layout="dense_bf16",
            quant_key="dense_bf16",
            payload=np.full((4, 8), 3, dtype=np.uint8),
        )
        for role in ("ffn_gate", "ffn_up", "ffn_down")
    }
    rank = MlpShardRank(
        runtime, device=device, stream=stream, weights=weights, hidden=8, per_rank_ffn=4
    )
    try:
        source = np.arange(16, dtype=np.uint8).reshape(-1)
        with scoped_current_device(runtime, device):
            src_buf = int(runtime.malloc(16))
            runtime.memcpy(
                src_buf,
                source.ctypes.data,
                16,
                MemcpyKind.HOST_TO_DEVICE,
            )
        rank.write_input_from_device(src_buf)
        readback = rank.read_input()
        assert np.array_equal(readback, source), (
            "the D2D stage must land the same bytes the device row holds"
        )
    finally:
        with scoped_current_device(runtime, device):
            runtime.free(src_buf)
        rank.close()
        runtime.stream_destroy(stream)


@needs_two_gpus
def test_the_shard_group_runs_both_ranks_and_reduces_to_bf16() -> None:
    """The group forward: real ranks, real exchange, bf16 boundary per rank.

    Tiny bf16 shards; the oracle is the same per-rank f32 chain the component
    test above pins, summed and rounded to bf16 at the group's boundary.
    """

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.runtime import MemcpyKind
    from hipengine.distributed.shard_exec import upload_shard_weight
    from hipengine.distributed.shard_group import MlpShardGroup

    runtime = get_hip_runtime()
    devices = (0, 1)
    hidden, full_ffn = 16, 8
    per_rank = full_ffn // 2
    scale = 0.05
    rng = np.random.default_rng(11)

    gate_f = (rng.standard_normal((full_ffn, hidden)) * scale).astype(np.float32)
    up_f = (rng.standard_normal((full_ffn, hidden)) * scale).astype(np.float32)
    down_f = (rng.standard_normal((hidden, full_ffn)) * scale).astype(np.float32)
    gate_b = _bf16_bytes(gate_f)
    up_b = _bf16_bytes(up_f)
    down_b = _bf16_bytes(down_f)
    x_f32 = (rng.standard_normal(hidden) * scale).astype(np.float32)
    x_bf16 = _bf16_round(x_f32)
    x_bf16_bytes = _bf16_bytes(x_f32).reshape(-1)

    gate_v = _bf16_values_from_bytes(gate_b)
    up_v = _bf16_values_from_bytes(up_b)
    down_v = _bf16_values_from_bytes(down_b)
    partials = []
    for rank in range(2):
        g = (x_bf16 @ gate_v[rank * per_rank : (rank + 1) * per_rank].T).astype(np.float32)
        u = (x_bf16 @ up_v[rank * per_rank : (rank + 1) * per_rank].T).astype(np.float32)
        act = _bf16_round(_silu(g) * u)
        partials.append((act @ down_v[:, rank * per_rank : (rank + 1) * per_rank].T).astype(np.float32))
    expected_bf16 = _bf16_round(partials[0] + partials[1])

    streams = {}
    for d in devices:
        with scoped_current_device(runtime, d):
            streams[d] = runtime.stream_create(nonblocking=True)
    weights = {
        0: {
            d: {
                role: upload_shard_weight(
                    runtime,
                    device=d,
                    name="raw",
                    layout="dense_bf16",
                    quant_key="dense_bf16",
                    payload=(
                        gate_b[d * per_rank : (d + 1) * per_rank]
                        if role == "ffn_gate"
                        else up_b[d * per_rank : (d + 1) * per_rank]
                        if role == "ffn_up"
                        else down_b[:, d * per_rank : (d + 1) * per_rank]
                    ),
                )
                for role in ("ffn_gate", "ffn_up", "ffn_down")
            }
            for d in devices
        }
    }
    group = MlpShardGroup(
        runtime,
        devices=devices,
        streams=streams,
        hidden=hidden,
        per_rank_ffn=per_rank,
        weights=weights,
    )
    inputs = {}
    try:
        for d in devices:
            with scoped_current_device(runtime, d):
                buf = runtime.malloc(hidden * 2)
                runtime.memcpy(
                    int(buf),
                    x_bf16_bytes.ctypes.data,
                    hidden * 2,
                    MemcpyKind.HOST_TO_DEVICE,
                )
                inputs[d] = int(buf)
        outputs = group.forward(0, inputs)
        assert group.reductions == 1
        for d in devices:
            out = np.empty(hidden, dtype="<u2")
            with scoped_current_device(runtime, d):
                runtime.memcpy(
                    out.ctypes.data, outputs[d], hidden * 2, MemcpyKind.DEVICE_TO_HOST
                )
            got = (out.astype("<u4") << 16).view("<f4")
            assert np.abs(got - expected_bf16).max() < 2e-2, (
                f"rank {d}'s bf16 output left the contract"
            )
    finally:
        group.close()
        for d in devices:
            with scoped_current_device(runtime, d):
                runtime.free(inputs[d])
                runtime.stream_destroy(streams[d])
