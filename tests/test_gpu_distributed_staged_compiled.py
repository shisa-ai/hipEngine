"""HIP-guarded hardware tests for the compiled staged-exchange transport.

These run the real hipcc-built host driver against two devices and pin what
the CPU fakes cannot: bit-parity of the reduced f32 payload against both the
Python route and a host oracle, the mapped payload surviving slot
alternation, and the zero-copy consumer contract - a device kernel
(``f32_to_bf16``, the boundary cast the shard group runs) reading the mapped
host row on the second device's own stream.

Skipped without ROCm or a second visible device, like every HIP test here.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

pytest.importorskip("numpy")

try:  # HIP availability guard: no-ROCm runners skip instead of failing.
    import ctypes as _ctypes

    _ctypes.CDLL("libamdhip64.so")
    HAVE_HIP = True
except OSError:  # pragma: no cover - CI/publish runners
    HAVE_HIP = False

pytestmark = pytest.mark.skipif(not HAVE_HIP, reason="HIP runtime unavailable")


def _two_devices(rt) -> bool:
    return rt.device_count() >= 2


def _bf16_bits_from_f32(values: np.ndarray) -> np.ndarray:
    """Round f32 to bf16 bits the way the device contract (RNE) does."""

    f = np.ascontiguousarray(values, dtype="<f4")
    u = f.view("<u4")
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")


def test_compiled_exchange_reduces_bit_identically_to_the_python_route() -> None:
    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.staged import StagedExchangeTransport
    from hipengine.distributed.staged_compiled import CompiledStagedExchangeTransport

    rt = get_hip_runtime()
    if not _two_devices(rt):
        pytest.skip("this test needs two visible devices")
    hidden = 4096
    rng = np.random.default_rng(11)
    streams = {}
    for device in (0, 1):
        with scoped_current_device(rt, device):
            streams[device] = rt.stream_create()

    for staging_dtype in ("f32", "bf16"):
        itemsize = 4 if staging_dtype == "f32" else 2
        if staging_dtype == "f32":
            rows = {
                0: rng.standard_normal(hidden).astype("<f4"),
                1: rng.standard_normal(hidden).astype("<f4"),
            }
        else:
            rows = {
                0: _bf16_bits_from_f32(rng.standard_normal(hidden)),
                1: _bf16_bits_from_f32(rng.standard_normal(hidden)),
            }
        dev_ptrs = {}
        for device in (0, 1):
            payload = np.ascontiguousarray(rows[device].view(np.uint8).reshape(-1))
            with scoped_current_device(rt, device):
                ptr = int(rt.malloc(hidden * itemsize))
            rt.memcpy(ptr, payload.ctypes.data, hidden * itemsize, 1)  # H2D
            dev_ptrs[device] = ptr

        python_route = StagedExchangeTransport(
            rt, devices=(0, 1), streams=streams, hidden=hidden, staging_dtype=staging_dtype
        )
        compiled = CompiledStagedExchangeTransport(
            rt, devices=(0, 1), streams=streams, hidden=hidden, staging_dtype=staging_dtype
        )
        if staging_dtype == "f32":
            expected = (rows[0] + rows[1]).astype("<f4")
        else:
            wide0 = (rows[0].astype(np.uint32) << 16).view(np.float32)
            wide1 = (rows[1].astype(np.uint32) << 16).view(np.float32)
            expected = (wide0 + wide1).astype("<f4")

        readback = np.empty(hidden, dtype="<f4")
        for _call in range(2):  # both slot sets must agree
            py_ptrs = python_route.reduce(dev_ptrs)
            compiled_ptrs = compiled.reduce(dev_ptrs)
            assert compiled_ptrs[0] == compiled_ptrs[1], (
                "both ranks consume the same mapped payload row"
            )
            with scoped_current_device(rt, 0):
                rt.memcpy(readback.ctypes.data, py_ptrs[0], hidden * 4, 2)  # D2H
            assert np.array_equal(readback.view("<u4"), expected.view("<u4"))
            got = np.frombuffer(ctypes.string_at(compiled_ptrs[0], hidden * 4), dtype="<f4")
            assert np.array_equal(got.view("<u4"), expected.view("<u4")), (
                "the compiled driver's payload is bit-identical to the host oracle"
            )
        python_route.close()
        compiled.close()

    for device in (0, 1):
        rt.stream_destroy(streams[device])


def test_the_mapped_payload_feeds_the_zero_copy_boundary_cast() -> None:
    """The production consumer contract on the second device's own stream."""

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.staged_compiled import CompiledStagedExchangeTransport
    from hipengine.kernels.hip_gfx1100.convert import f32_to_bf16

    rt = get_hip_runtime()
    if not _two_devices(rt):
        pytest.skip("this test needs two visible devices")
    hidden = 4096
    rng = np.random.default_rng(5)
    streams = {}
    for device in (0, 1):
        with scoped_current_device(rt, device):
            streams[device] = rt.stream_create()
    rows = {
        0: rng.standard_normal(hidden).astype("<f4"),
        1: rng.standard_normal(hidden).astype("<f4"),
    }
    dev_ptrs = {}
    for device in (0, 1):
        payload = np.ascontiguousarray(rows[device])
        with scoped_current_device(rt, device):
            ptr = int(rt.malloc(hidden * 4))
        rt.memcpy(ptr, payload.ctypes.data, hidden * 4, 1)  # H2D
        dev_ptrs[device] = ptr

    compiled = CompiledStagedExchangeTransport(
        rt, devices=(0, 1), streams=streams, hidden=hidden, staging_dtype="f32"
    )
    mapped = compiled.reduce(dev_ptrs)[1]

    # Device 1 casts the mapped host row (zero-copy over PCIe) into its own
    # device bf16 buffer on its own stream - exactly what MlpShardGroup's
    # boundary cast does after a compiled reduction.
    expected_sum = rows[0] + rows[1]
    exp_bits = (expected_sum.view("<u4") + 0x7FFF + ((expected_sum.view("<u4") >> 16) & 1)) >> 16
    got = np.empty(hidden, dtype="<u2")
    with scoped_current_device(rt, 1):
        out = int(rt.malloc(hidden * 2))
        f32_to_bf16(mapped, out, hidden, stream=streams[1], runtime=rt)
        rt.stream_synchronize(streams[1])
        rt.memcpy(got.ctypes.data, out, hidden * 2, 2)  # D2H
    assert np.array_equal(got, exp_bits.astype("<u2")), (
        "the boundary cast over the mapped payload produces the same bf16 bits"
    )
    compiled.close()
    for device in (0, 1):
        rt.stream_destroy(streams[device])


def test_reduce_at_publishes_into_fixed_slots_and_consumers_read_them() -> None:
    """Caller-chosen slots: two fixed publishes and a mapped-payload consumer.

    The graphed schedule reduces each layer into its own fixed payload slot
    and a captured consumer reads that slot's mapped host row later. This
    pins: the slot address computation, the internal alternation surviving
    slot-pinned reduces, and the zero-copy contract on the second device.
    """

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.staged_compiled import CompiledStagedExchangeTransport

    rt = get_hip_runtime()
    if not _two_devices(rt):
        pytest.skip("this test needs two visible devices")
    hidden = 1024
    rng = np.random.default_rng(23)
    streams = {}
    for device in (0, 1):
        with scoped_current_device(rt, device):
            streams[device] = rt.stream_create()
    transport = CompiledStagedExchangeTransport(
        rt, devices=(0, 1), streams=streams, hidden=hidden, slot_sets=4
    )
    try:
        partials = {}
        host_partials = {}
        for device in (0, 1):
            with scoped_current_device(rt, device):
                row = rng.standard_normal(hidden).astype("<f4")
                host_partials[device] = row
                buffer = rt.malloc(row.nbytes)
                rt.memcpy(
                    buffer,
                    row.ctypes.data,
                    row.nbytes,
                    3,  # hipMemcpyHostToDevice
                )
                partials[device] = buffer
        payload_1 = transport.payload_ptr(1)
        payload_3 = transport.payload_ptr(3)
        assert payload_3 - payload_1 == 2 * hidden * 4, (
            "fixed slots sit one payload row apart each"
        )
        reduced = transport.reduce(partials, slot=3)
        assert reduced[0] == payload_3 and reduced[1] == payload_3
        expected = (
            host_partials[0].astype("<f8") + host_partials[1].astype("<f8")
        ).astype("<f4")
        got = np.empty(hidden, dtype="<f4")
        ctypes.memmove(
            got.ctypes.data, ctypes.c_void_p(payload_3), got.nbytes
        )
        np.testing.assert_array_equal(got, expected)
        # The eager two-slot alternation still works after slot-pinned calls.
        alternated = transport.reduce(partials)
        assert alternated[0] not in (payload_1, payload_3)
        ctypes.memmove(got.ctypes.data, ctypes.c_void_p(alternated[0]), got.nbytes)
        np.testing.assert_array_equal(got, expected)
    finally:
        transport.close()
        for device in (0, 1):
            with scoped_current_device(rt, device):
                rt.stream_destroy(streams[device])


def test_device_exchange_sums_bit_identically_and_locksteps_on_flags() -> None:
    """The device-side reduction: bit-parity with the host sum, and the
    published-flag spin that lets captured graphs lockstep without any host
    wait - the graphed schedule's production reduction path."""

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.device_exchange_compiled import (
        CompiledDeviceExchange,
    )

    rt = get_hip_runtime()
    if not _two_devices(rt):
        pytest.skip("this test needs two visible devices")
    hidden = 4096
    rng = np.random.default_rng(41)
    streams = {}
    for device in (0, 1):
        with scoped_current_device(rt, device):
            streams[device] = rt.stream_create()
    exchange = CompiledDeviceExchange(
        rt, devices=(0, 1), streams=streams, num_layers=3, hidden=hidden
    )
    try:
        partials = {}
        host_rows = {}
        for device in (0, 1):
            bits = rng.integers(0, 2**16, size=hidden, dtype=np.uint16)
            host_rows[device] = bits
            with scoped_current_device(rt, device):
                buffer = rt.malloc(hidden * 2)
                rt.memcpy(buffer, bits.ctypes.data, hidden * 2, 3)
                partials[device] = buffer
        outs = {}
        for device in (0, 1):
            with scoped_current_device(rt, device):
                outs[device] = rt.malloc(hidden * 2)

        exchange.step_begin()
        for rank, device in enumerate((0, 1)):
            exchange.enqueue_rank(rank, partials[device], 0, outs[device])
        exchange.wait()
        got = {}
        for device in (0, 1):
            bits = np.empty(hidden, dtype="<u2")
            with scoped_current_device(rt, device):
                rt.memcpy(bits.ctypes.data, outs[device], hidden * 2, 2)
            got[device] = bits

        def widen(bits: np.ndarray) -> np.ndarray:
            return (bits.astype("<u4") << 16).view("<f4")

        expected = (widen(host_rows[0]).astype("<f4") + widen(host_rows[1]).astype("<f4"))
        # RNE narrow, the boundary-cast kernel's bit arithmetic.
        u = expected.view("<u4")
        expected_bits = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")
        for device in (0, 1):
            np.testing.assert_array_equal(
                got[device], expected_bits,
                err_msg=f"device exchange output on device {device} is not bit-identical to the host sum",
            )

        # A second step reuses slot 0; the lockstep must still hold.
        exchange.step_begin()
        for rank, device in enumerate((0, 1)):
            exchange.enqueue_rank(rank, partials[device], 0, outs[device])
        exchange.wait()
        for device in (0, 1):
            bits = np.empty(hidden, dtype="<u2")
            with scoped_current_device(rt, device):
                rt.memcpy(bits.ctypes.data, outs[device], hidden * 2, 2)
            np.testing.assert_array_equal(bits, expected_bits)
    finally:
        exchange.close()
        for device in (0, 1):
            with scoped_current_device(rt, device):
                rt.stream_destroy(streams[device])


def test_device_exchange_spin_timeout_fails_instead_of_hanging() -> None:
    """A missing peer must fail the exchange, not hang the rank group.

    Three things are pinned, in the order they matter: the spin kernel's bounded
    exit returns the rank's stream (so ``wait`` cannot block forever), the host
    surfaces the timeout as a transport error and poisons the handle (so the
    rank group cannot keep running), and the timed-out exchange publishes
    nothing (so a stale or partial row cannot pass as a result).
    """

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.device_exchange_compiled import (
        CompiledDeviceExchange,
    )
    from hipengine.distributed.transport import TransportStateError

    rt = get_hip_runtime()
    if not _two_devices(rt):
        pytest.skip("this test needs two visible devices")
    hidden = 5120
    streams = {}
    for device in (0, 1):
        with scoped_current_device(rt, device):
            streams[device] = rt.stream_create()
    exchange = CompiledDeviceExchange(
        rt, devices=(0, 1), streams=streams, num_layers=1, hidden=hidden,
        max_spins=20_000,
    )
    try:
        with scoped_current_device(rt, 0):
            partial = rt.malloc(hidden * 2)
            out = rt.malloc(hidden * 2)
            rt.memset(partial, 0, hidden * 2)
            # A byte pattern no exchange would produce, so any write is visible.
            rt.memset(out, 0xAB, hidden * 2)
        exchange.step_begin()
        # Only rank 0 exchanges; rank 1 never publishes, so rank 0's spin has no
        # flag to see and must give up within its budget.
        exchange.enqueue_rank(0, partial, 0, out)
        with pytest.raises(TransportStateError, match="spin timeout"):
            exchange.wait()
        # The failure is a handle-level failure: the rank group cannot go on and
        # read a row that was never summed.
        with pytest.raises(TransportStateError, match="poisoned"):
            exchange.step_begin()
        with pytest.raises(TransportStateError, match="poisoned"):
            exchange.wait()
        with scoped_current_device(rt, 0):
            published = np.empty(hidden, dtype="<u2")
            rt.memcpy(published.ctypes.data, out, hidden * 2, 2)
        assert np.all(published.view("<u1") == 0xAB), (
            "the timed-out exchange wrote an output row instead of publishing nothing"
        )
        with scoped_current_device(rt, 0):
            rt.free(partial)
            rt.free(out)
    finally:
        exchange.close()
        for device in (0, 1):
            with scoped_current_device(rt, device):
                rt.stream_destroy(streams[device])


def test_head_shard_gemv_rows_bit_identical_to_replicated_head() -> None:
    """The sharded head must produce bit-identical logit values per row.

    Both ranks' contiguous Q6_K block ranges are repacked to the
    runtime-resolved layout and compared against the full replicated head
    GEMV on the same hidden input; row independence makes every range
    bit-exact, which is what pins the exact greedy tie-break.
    """

    import ctypes
    import os

    import numpy as np

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device
    from hipengine.distributed.head_shard import materialize_head_shards
    from hipengine.distributed.shard_exec import upload_shard_weight
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
        gguf_rmsnorm_bf16_f32_weight,
    )
    from hipengine.loading.gguf import GGUFReader
    from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16_qmicro_planar
    from hipengine.runtime.gguf_linear import GGUF_OUTPUT_F32, launch_gguf_linear

    class _RawPtrProxy:
        def __init__(self, ptr, nbytes):
            self.ptr = int(ptr)
            self.nbytes = int(nbytes)

    rt = get_hip_runtime()
    if not _two_devices(rt):
        pytest.skip("this test needs two visible devices")
    model = os.environ.get("HIPENGINE_TP2_TEST_MODEL")
    if not model or not os.path.exists(model):
        pytest.skip("HIPENGINE_TP2_TEST_MODEL must point at the artifact GGUF")

    reader = GGUFReader(model)
    head_info = reader.info.tensor("output.weight")
    vocab_rows = head_info.shape[0] if hasattr(head_info, "shape") else None
    assert vocab_rows is not None
    hidden = 5120
    plan, rank_payloads = materialize_head_shards(model, world_size=2)
    assert plan.vocab_rows == int(vocab_rows)

    # The full replicated head, repacked to the same layout, on device 0.
    source = np.memmap(
        reader.path, dtype=np.uint8, mode="r",
        offset=head_info.data_offset,
        shape=(plan.vocab_rows * plan.source_row_bytes,),
    )
    full_local = np.ascontiguousarray(source).reshape(
        1, plan.vocab_rows, plan.source_row_bytes
    )
    full_tiles = np.ascontiguousarray(
        np.asarray(repack_gguf_q6_k_tile16_qmicro_planar(full_local).tiles)
    ).reshape(-1)

    rng = np.random.default_rng(23)
    x_host = (rng.standard_normal(hidden) * 0.5).astype(np.float32)
    import struct

    x_bits = np.asarray(
        [struct.unpack("<I", struct.pack("<f", v))[0] >> 16 for v in x_host],
        dtype=np.uint16,
    )
    streams = {}
    buffers = {}
    for device in (0, 1):
        with scoped_current_device(rt, device):
            streams[device] = rt.stream_create()
            x_dev = rt.malloc(hidden * 2)
            copy_host_to_device(
                _RawPtrProxy(x_dev, hidden * 2), x_bits.ctypes.data, hidden * 2,
                runtime=rt,
            )
            out_dev = rt.malloc(plan.vocab_rows * 4)
            buffers[device] = (x_dev, out_dev)
    try:
        with scoped_current_device(rt, 0):
            full_weight = upload_shard_weight(
                rt, device=0, name="tiles", layout=plan.layout,
                quant_key=plan.quant_key, payload=full_tiles,
            )
            launch_gguf_linear(
                full_weight, buffers[0][0], buffers[0][1], 1, hidden,
                plan.vocab_rows, output_dtype=GGUF_OUTPUT_F32,
                stream=streams[0], runtime=rt,
            )
        shard_outs = {}
        for device in (0, 1):
            with scoped_current_device(rt, device):
                weight = upload_shard_weight(
                    rt, device=device, name="tiles", layout=plan.layout,
                    quant_key=plan.quant_key, payload=rank_payloads[device]["tiles"],
                )
                out_dev = rt.malloc(plan.rows_per_rank * 4)
                launch_gguf_linear(
                    weight, buffers[device][0], out_dev, 1, hidden,
                    plan.rows_per_rank, output_dtype=GGUF_OUTPUT_F32,
                    stream=streams[device], runtime=rt,
                )
                rt.stream_synchronize(streams[device])
                row = np.empty(plan.rows_per_rank, dtype="<f4")
                copy_device_to_host(
                    row.ctypes.data, _RawPtrProxy(out_dev, plan.rows_per_rank * 4),
                    plan.rows_per_rank * 4, runtime=rt,
                )
                shard_outs[device] = row
                rt.free(out_dev)
        with scoped_current_device(rt, 0):
            rt.stream_synchronize(streams[0])
            full_row = np.empty(plan.vocab_rows, dtype="<f4")
            copy_device_to_host(
                full_row.ctypes.data,
                _RawPtrProxy(buffers[0][1], plan.vocab_rows * 4),
                plan.vocab_rows * 4, runtime=rt,
            )
        assert np.array_equal(
            full_row[: plan.rows_per_rank], shard_outs[0]
        ), "rank 0's head shard rows differ from the replicated head"
        assert np.array_equal(
            full_row[plan.rows_per_rank :], shard_outs[1]
        ), "rank 1's head shard rows differ from the replicated head"
    finally:
        for device in (0, 1):
            with scoped_current_device(rt, device):
                rt.stream_destroy(streams[device])
                rt.free(buffers[device][0])
                rt.free(buffers[device][1])


def _widen_bf16(bits: np.ndarray) -> np.ndarray:
    return (bits.astype("<u4") << 16).view("<f4")


def _narrow_bf16_rne(values: np.ndarray) -> np.ndarray:
    """RNE f32 -> bf16, the boundary-cast kernel's bit arithmetic."""

    u = values.view("<u4")
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")


def test_batched_device_exchange_sums_a_whole_prompt_bit_identically() -> None:
    """The batched form of the device reduction: ``rows`` prompt rows reduced
    in one exchange instead of one row per token step.

    The driver stages a slot as a single long row, so a batched caller passes
    ``rows`` and the element count scales with it. Parity is bit-exact against
    the host sum followed by the boundary cast, which is what lets the bulk
    prefill use this instead of the host-staged transport.
    """

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.device_exchange_compiled import CompiledDeviceExchange

    rt = get_hip_runtime()
    if not _two_devices(rt):
        pytest.skip("this test needs two visible devices")
    hidden, rows = 512, 8
    n = hidden * rows
    rng = np.random.default_rng(97)
    streams = {}
    for device in (0, 1):
        with scoped_current_device(rt, device):
            streams[device] = rt.stream_create()
    exchange = CompiledDeviceExchange(
        rt, devices=(0, 1), streams=streams, num_layers=2, hidden=hidden, rows=rows
    )
    assert exchange.hidden == n, "a batched exchange stages the whole batch as one row"
    assert exchange.rows == rows
    # Tracked per device: a buffer allocated on one rank cannot be freed from
    # another rank's context.
    buffers: dict[int, list[int]] = {0: [], 1: []}
    try:
        outs = {}
        for device in (0, 1):
            with scoped_current_device(rt, device):
                outs[device] = rt.malloc(n * 2)
                buffers[device].append(outs[device])

        # Two layers, two alternating slots, different data per layer: this is
        # what catches a missing per-layer counter bump, because the peer's
        # flag from layer 0 would already satisfy layer 1's comparison and the
        # spin would read layer 0's staging.
        exchange.reset_timeouts()
        for layer in range(2):
            host_rows = {}
            partials = {}
            for device in (0, 1):
                bits = rng.integers(
                    0, 2**16, size=n, dtype=np.uint16
                )
                host_rows[device] = bits
                with scoped_current_device(rt, device):
                    buffer = rt.malloc(n * 2)
                    buffers[device].append(buffer)
                    rt.memcpy(buffer, bits.ctypes.data, n * 2, 3)
                    partials[device] = buffer
            exchange.bump()
            for rank, device in enumerate((0, 1)):
                exchange.enqueue_rank(rank, partials[device], layer % 2, outs[device])
            exchange.wait()
            expected = _narrow_bf16_rne(
                _widen_bf16(host_rows[0]) + _widen_bf16(host_rows[1])
            )
            for device in (0, 1):
                got = np.empty(n, dtype="<u2")
                with scoped_current_device(rt, device):
                    rt.memcpy(got.ctypes.data, outs[device], n * 2, 2)
                np.testing.assert_array_equal(
                    got,
                    expected,
                    err_msg=(
                        f"batched device exchange, layer {layer}, device {device}: "
                        "not bit-identical to the host sum"
                    ),
                )
    finally:
        exchange.close()
        for device in (0, 1):
            with scoped_current_device(rt, device):
                rt.stream_synchronize(streams[device])
                for buffer in buffers[device]:
                    rt.free(buffer)
                rt.stream_destroy(streams[device])


def test_batched_device_exchange_keeps_a_timeout_visible_across_layers() -> None:
    """A per-layer bump must not clear an earlier layer's spin timeout.

    The spin kernel writes nothing when it times out, so a cleared flag would
    let a stale output row pass as a result. ``reset_timeouts`` exists to be
    called once per group for exactly this reason, and ``bump`` must not clear
    the flag the way ``step_begin`` does.
    """

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.device_exchange_compiled import CompiledDeviceExchange
    from hipengine.distributed.transport import TransportStateError

    rt = get_hip_runtime()
    if not _two_devices(rt):
        pytest.skip("this test needs two visible devices")
    hidden, rows = 256, 4
    n = hidden * rows
    streams = {}
    for device in (0, 1):
        with scoped_current_device(rt, device):
            streams[device] = rt.stream_create()
    exchange = CompiledDeviceExchange(
        rt,
        devices=(0, 1),
        streams=streams,
        num_layers=2,
        hidden=hidden,
        rows=rows,
        max_spins=20_000,
    )
    buffers: dict[int, list[int]] = {0: [], 1: []}
    try:
        outs = {}
        for device in (0, 1):
            with scoped_current_device(rt, device):
                outs[device] = rt.malloc(n * 2)
                buffers[device].append(outs[device])
        partials = {}
        for device in (0, 1):
            with scoped_current_device(rt, device):
                buffer = rt.malloc(n * 2)
                buffers[device].append(buffer)
                rt.memset(buffer, 0, n * 2)
                partials[device] = buffer

        exchange.reset_timeouts()
        # Only rank 0 exchanges, so its spin expires and sets its timeout flag.
        exchange.bump()
        exchange.enqueue_rank(0, partials[0], 0, outs[0])
        # The next layer's bump must leave that flag set: if it cleared it the
        # way step_begin does, the wait below would report success and the
        # unwritten output row would pass as a result.
        exchange.bump()
        exchange.enqueue_rank(0, partials[0], 1, outs[0])
        with pytest.raises(TransportStateError, match="spin timeout"):
            exchange.wait()
    finally:
        exchange.close()
        for device in (0, 1):
            with scoped_current_device(rt, device):
                rt.stream_synchronize(streams[device])
                for buffer in buffers[device]:
                    rt.free(buffer)
                rt.stream_destroy(streams[device])


def test_device_exchange_serves_two_reductions_in_one_layer() -> None:
    """Head sharding reduces twice per layer: the attention-output partial before
    the post-attention norm, then the MLP down partial. This drives that layout
    through the real device exchange - ``num_layers = 2 * reductions_per_layer``
    with ``slot = reductions_per_layer * (layer % 2) + phase`` - and checks each
    phase against its own host sum.

    What this establishes: the doubled slot space is real on the driver side and
    both reductions produce the correct sum through it. What it does **not**
    establish: that a shared slot would be caught here. The hazard is a timing
    race - the peer's spin exits as soon as its flag reaches the step, so the peer
    may overwrite its own staging while this rank's spin-add is still reading it -
    and with a payload this small the reader always wins. Mutating the slot to
    collide both phases leaves this test green, so the deterministic guard is the
    slot-arithmetic assertion in
    ``test_unit_distributed_shard_group.py::test_device_reduce_gives_each_phase_its_own_slot``,
    not this.
    """

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.device_exchange_compiled import CompiledDeviceExchange

    rt = get_hip_runtime()
    if not _two_devices(rt):
        pytest.skip("this test needs two visible devices")
    hidden, phases = 256, 2
    rng = np.random.default_rng(2026)
    streams = {}
    for device in (0, 1):
        with scoped_current_device(rt, device):
            streams[device] = rt.stream_create()
    exchange = CompiledDeviceExchange(
        rt,
        devices=(0, 1),
        streams=streams,
        num_layers=2 * phases,
        hidden=hidden,
    )
    buffers: dict[int, list[int]] = {0: [], 1: []}
    try:
        outs: dict[int, dict[int, int]] = {phase: {} for phase in range(phases)}
        for phase in range(phases):
            for device in (0, 1):
                with scoped_current_device(rt, device):
                    outs[phase][device] = rt.malloc(hidden * 2)
                    buffers[device].append(outs[phase][device])

        # Two layers exercise the set alternation as well as the phase split.
        exchange.reset_timeouts()
        for layer in range(2):
            payloads: dict[int, dict[int, tuple[int, np.ndarray]]] = {}
            for phase in range(phases):
                payloads[phase] = {}
                for device in (0, 1):
                    # Finite bf16, not raw bits: a random uint16 lands on the
                    # exponent 0xFF patterns often enough that the comparison
                    # would spend part of its budget on NaN == NaN.
                    values = rng.normal(0.0, 1.0, size=hidden).astype(np.float32)
                    bits = (values.view(np.uint32) >> 16).astype(np.uint16)
                    with scoped_current_device(rt, device):
                        buffer = rt.malloc(hidden * 2)
                        buffers[device].append(buffer)
                        rt.memcpy(buffer, bits.ctypes.data, hidden * 2, 3)
                        payloads[phase][device] = (buffer, bits)
            for phase in range(phases):
                exchange.bump()
                slot = phases * (layer % 2) + phase
                for rank, device in enumerate((0, 1)):
                    exchange.enqueue_rank(
                        rank, payloads[phase][device][0], slot, outs[phase][device]
                    )
            exchange.wait()
            for phase in range(phases):
                expected = _narrow_bf16_rne(
                    _widen_bf16(payloads[phase][0][1])
                    + _widen_bf16(payloads[phase][1][1])
                )
                for device in (0, 1):
                    got = np.empty(hidden, dtype="<u2")
                    with scoped_current_device(rt, device):
                        rt.memcpy(got.ctypes.data, outs[phase][device], hidden * 2, 2)
                    np.testing.assert_array_equal(
                        got,
                        expected,
                        err_msg=(
                            f"layer {layer} phase {phase} device {device}: a "
                            "per-layer reduction did not get its own staging slot"
                        ),
                    )
    finally:
        exchange.close()
        for device in (0, 1):
            with scoped_current_device(rt, device):
                rt.stream_synchronize(streams[device])
                for buffer in buffers[device]:
                    rt.free(buffer)
                rt.stream_destroy(streams[device])
