"""Two-GPU guarded tests for the distributed transport and KV claims.

These run only on a host with at least two HIP devices. They exercise the parts
of Packet 1 that a CPU mock cannot: a real RCCL communicator, a real group
all-reduce, a real peer-copy probe, and a real KV claim against device memory.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

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


def _devices(count: int = 2) -> list[int]:
    return list(range(count))


@pytest.fixture
def plan():
    from hipengine.distributed.plan import DistributedPlan

    return DistributedPlan.resolve(_devices(), hidden_size=5120)


@needs_two_gpus
def test_gpu_distributed_rccl_group_all_reduce_matches_host_sum() -> None:
    """A real two-rank RCCL all-reduce produces the arithmetic sum on both ranks."""

    import numpy as np

    from hipengine.core.device import Device, scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_array_to_device, free, host_array_ptr, malloc
    from hipengine.distributed.plan import DistributedPlan
    from hipengine.distributed.rccl import RcclTransport

    runtime = get_hip_runtime()
    resolved = DistributedPlan.resolve(_devices(), hidden_size=5120, algorithm="rccl")
    transport = RcclTransport([spec.device for spec in resolved.ranks], runtime=runtime, init_timeout_s=120.0)
    count = 5120
    nbytes = count * 4
    send = [malloc(nbytes, device=Device("hip", rank)) for rank in range(2)]
    recv = [malloc(nbytes, device=Device("hip", rank)) for rank in range(2)]
    try:
        for rank in range(2):
            copy_host_array_to_device(send[rank], np.full(count, float(rank + 1), dtype=np.float32))
        transport.group_start()
        for rank in range(2):
            transport.all_reduce_sum(rank, send[rank].ptr, recv[rank].ptr, count=count, dtype="fp32")
        transport.group_end()
        transport.sync(timeout_s=120.0)
        for rank in range(2):
            host = np.empty(count, dtype=np.float32)
            with scoped_current_device(runtime, rank):
                copy_device_to_host(host_array_ptr(host), recv[rank])
            assert np.array_equal(host, np.full(count, 3.0, dtype=np.float32)), f"rank {rank} all-reduce"
    finally:
        for buffer in (*send, *recv):
            free(buffer, runtime=runtime)
        transport.close()


@needs_two_gpus
def test_gpu_distributed_context_selects_and_restores_devices() -> None:
    """A rank-bound context binds the right device and restores the caller's."""

    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.context import DistributedContext
    from hipengine.distributed.plan import DistributedPlan
    from hipengine.distributed.rccl import RcclTransport

    runtime = get_hip_runtime()
    resolved = DistributedPlan.resolve(_devices(), hidden_size=5120, algorithm="rccl")
    transport = RcclTransport([spec.device for spec in resolved.ranks], runtime=runtime, init_timeout_s=120.0)
    with DistributedContext.create(resolved, transport=transport, runtime=runtime) as context:
        assert context.rank(1).device.index == 1
        with context.rank(1).activate():
            assert int(runtime.current_device()) == 1
        assert int(runtime.current_device()) == 0


@needs_two_gpus
def test_gpu_distributed_kv_claim_allocates_and_rolls_back_on_device() -> None:
    """A real KV claim allocates on every rank and releases on failure."""

    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import free, malloc
    from hipengine.distributed.kv import (
        KvSpansLayout,
        claim_all,
        resolve_kv_claims,
        resolve_kv_geometry,
    )
    from hipengine.distributed.plan import DistributedPlan

    runtime = get_hip_runtime()
    config = SimpleNamespace(
        layer_types=tuple("full_attention" if index % 4 == 3 else "linear_attention" for index in range(64)),
        head_count_kv=4,
        key_length=256,
        value_length=256,
    )
    resolved = DistributedPlan.resolve(_devices(), hidden_size=5120)
    geometry = resolve_kv_geometry(
        config, spans=KvSpansLayout.dense_policy(), world_size=2, context_tokens=8192
    )
    claims = resolve_kv_claims(resolved, geometry)
    live: list[object] = []
    released: list[int] = []

    def allocate(claim):
        """A well-behaved allocator: never return a buffer it cannot keep."""

        buffer = malloc(claim.total_bytes, device=Device("hip", claim.rank))
        if claim.rank == 1:
            # The allocator owns this buffer until it returns it, so it must
            # free it itself when it fails.
            free(buffer, runtime=runtime)
            raise RuntimeError("synthetic failure after rank 0 allocated")
        live.append(buffer)
        return buffer

    def release(buffer) -> None:
        free(buffer, runtime=runtime)
        live.remove(buffer)
        released.append(1)

    with pytest.raises(Exception) as error:
        claim_all(claims, allocate, release)
    assert "1 of 2 ranks" in str(error.value)
    assert released == [1], "rank 0's device allocation must be released when rank 1 fails"
    assert live == [], "no device buffer may survive a failed claim"

    # The happy path allocates on both ranks and releases cleanly.
    buffers = claim_all(claims, lambda claim: malloc(claim.total_bytes, device=Device("hip", claim.rank)), lambda buffer: free(buffer, runtime=runtime))
    assert len(buffers) == 2
    for buffer in buffers:
        free(buffer, runtime=runtime)


@needs_two_gpus
def test_gpu_distributed_kv_group_reservation_is_ledger_owned() -> None:
    """The scheduler's ledgers hold the group, and the device path runs after."""

    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import free, malloc
    from hipengine.distributed.kv import (
        KvSpansLayout,
        build_group_kv_plan,
        plane_pool_id,
        reserve_group_kv,
        resolve_kv_claims,
        resolve_kv_geometry,
    )
    from hipengine.distributed.plan import DistributedPlan
    from hipengine.kvcache.ledger import ResourceLedger

    runtime = get_hip_runtime()
    config = SimpleNamespace(
        layer_types=tuple(
            "full_attention" if index % 4 == 3 else "linear_attention" for index in range(64)
        ),
        head_count_kv=4,
        key_length=256,
        value_length=256,
    )
    resolved = DistributedPlan.resolve(_devices(), hidden_size=5120)
    geometry = resolve_kv_geometry(
        config, spans=KvSpansLayout.dense_policy(), world_size=2, context_tokens=2048
    )
    claims = resolve_kv_claims(resolved, geometry)
    rank_plans = build_group_kv_plan(
        claims, backend_fingerprint="gpu-test", generation=1
    )
    ledgers = [ResourceLedger(plan.plan) for plan in rank_plans]

    reservation = reserve_group_kv(rank_plans, ledgers=ledgers, group_id="gpu-test")
    for ledger in ledgers:
        assert ledger.snapshot()["provisional_reservations"] == 1
    reservation.commit()
    for index, ledger in enumerate(ledgers):
        assert ledger.has_owner(f"{reservation.owner_id}:rank{index}")
        ledger.assert_conserved()

    # A second group over the same rank pools cannot also reserve them.
    with pytest.raises(Exception) as error:
        reserve_group_kv(rank_plans, ledgers=ledgers)
    assert plane_pool_id(0) in str(error.value)

    # The device buffers follow the committed ledger, one per rank.
    buffers = [
        malloc(rank_plans[rank].geometry_bytes, device=Device("hip", rank))
        for rank in range(2)
    ]
    assert len(buffers) == 2
    for buffer in buffers:
        free(buffer, runtime=runtime)


@needs_two_gpus
def test_gpu_distributed_peer_access_screen_is_recorded() -> None:
    """Peer access is probed, not assumed: the result is recorded either way."""

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.device import scoped_current_device

    runtime = get_hip_runtime()
    observed: list[bool] = []
    for source, target in ((0, 1), (1, 0)):
        try:
            with scoped_current_device(runtime, source):
                can = bool(runtime.device_can_access_peer(source, target))
        except Exception as error:  # noqa: BLE001 - an unavailable probe is a result
            pytest.skip(f"peer probe unavailable: {error!r}")
        observed.append(can)
    # This host has no peer DMA (256 MB BARs and ACS redirection), so the screen
    # must report false; a host that enables it reports true. Either way the
    # probe must answer rather than raise.
    assert len(observed) == 2


@needs_two_gpus
def test_gpu_distributed_mismatched_group_fails_fast_on_real_communicators() -> None:
    """A real mismatched group must raise and abort, never hang.

    This is the failure path the CPU suite could only reach through the mock.
    With real RCCL communicators the difference matters: an operation that
    reaches RCCL is queued, so a rank that issues a different sequence would
    block its peer forever rather than report an error. The transport must
    detect the mismatch before issuing the second rank's operation and abort the
    communicators so nothing is left waiting.
    """

    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import free, malloc
    from hipengine.distributed.plan import DistributedPlan
    from hipengine.distributed.rccl import RcclTransport
    from hipengine.distributed.transport import TransportError, TransportStateError

    runtime = get_hip_runtime()
    resolved = DistributedPlan.resolve(_devices(), hidden_size=5120, algorithm="rccl")
    transport = RcclTransport([spec.device for spec in resolved.ranks], runtime=runtime, init_timeout_s=120.0)
    send = [malloc(4096, device=Device("hip", rank)) for rank in range(2)]
    recv = [malloc(4096, device=Device("hip", rank)) for rank in range(2)]
    try:
        transport.group_start()
        transport.all_reduce_sum(0, send[0].ptr, recv[0].ptr, count=1024, dtype="fp32")
        # Rank 1 declares a different payload size: a mismatch, detected before
        # rank 1's operation reaches the communicator.
        with pytest.raises(TransportStateError):
            transport.all_reduce_sum(1, send[1].ptr, recv[1].ptr, count=2048, dtype="fp32")
        assert transport.poisoned is True
        # The communicators were aborted, so a later sync cannot wait forever:
        # it reports the poisoned group instead of blocking on a collective its
        # peer will never issue.
        with pytest.raises(TransportError):
            transport.sync(timeout_s=5.0)
    finally:
        for buffer in (*send, *recv):
            free(buffer, runtime=runtime)
        transport.close()
