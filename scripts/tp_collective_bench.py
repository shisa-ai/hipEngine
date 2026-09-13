#!/usr/bin/env python3
"""Packet 0 topology/collective screen for the TP2 campaign.

Measures, on an explicitly ordered device list:

  * peer access in both directions and verified device-to-device copies,
  * broadcast and all-reduce-sum warm latency (p50/p95/p99) and bandwidth for
    FP32 and any proposed transport dtype,
  * payload shapes ``rows * hidden_size * dtype_bytes`` for rows 1..5 (decode)
    and realistic prefill chunks,
  * rank order and enqueue-mode effects, including sequential reductions with a
    local producer/consumer on each rank,
  * optional HIP graph capture of a whole chain (``--graph-chain-depth``) with a
    replay that must match the graph-disabled result bit-for-bit,
  * PCIe negotiated link width/speed sampled while traffic is running.

Capture size is bounded on ROCm 7.2 / RCCL 2.27.7: 49 captured nodes (a 24-op
chain with its producer/consumer memsets) captures and replays, while 65 nodes
(a 32-op chain) faults the device with a memory access error. Keep any captured
group well below that; a TP2 decode step's 36 collectives cannot be one graph.

Every number comes from a real device run and is written to a compact JSON
artifact. This script drives :mod:`hipengine.distributed` (the same plan and
transport objects the runtime will use) so the campaign does not grow a second,
temporary communication wrapper.

Usage:
    python3 scripts/tp_collective_bench.py --devices 0,1 --mode all \
        --hidden-size 5120 --json benchmarks/results/tp2_collective_bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_HIDDEN_SIZE = 5120
DEFAULT_DTYPES = ("fp32", "bf16")
DEFAULT_ROWS = (1, 2, 3, 4, 5)
DEFAULT_PREFILL_ROWS = (128, 512, 1024)
DEFAULT_PEER_SIZES = (1 << 12, 1 << 20, 1 << 24, 1 << 26)
DEFAULT_CHAIN_DEPTH = 4


# ---------------------------------------------------------------------------
# Statistics helpers (pure, CPU-testable)
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolation percentile over a non-empty sample."""

    if not values:
        raise ValueError("percentile requires at least one sample")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = float(quantile) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_samples(values: Sequence[float]) -> dict[str, float]:
    """Summarize a latency sample in milliseconds."""

    if not values:
        raise ValueError("summary requires at least one sample")
    samples = [float(value) for value in values]
    mean = statistics.fmean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return {
        "count": len(samples),
        "mean_ms": mean,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "p99_ms": percentile(samples, 0.99),
        "stdev_ms": stdev,
        "cv": (stdev / mean) if mean > 0 else 0.0,
    }


def bandwidth_gbs(*, payload_bytes: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return (float(payload_bytes) / 1e9) / (float(latency_ms) / 1e3)


def bus_bandwidth_gbs(*, payload_bytes: int, latency_ms: float, world_size: int) -> float:
    """NCCL-style bus bandwidth for all-reduce (2*(n-1)/n); identity for broadcast."""

    algorithm = bandwidth_gbs(payload_bytes=payload_bytes, latency_ms=latency_ms)
    if int(world_size) <= 1:
        return algorithm
    return algorithm * (2.0 * (int(world_size) - 1) / int(world_size))


def encode_values(values: Sequence[float], dtype: str):
    """Encode host float values into the wire dtype (bf16 as uint16 bit patterns)."""

    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if dtype == "fp32":
        return array.astype(np.float32)
    if dtype == "fp16":
        return array.astype(np.float16)
    if dtype == "bf16":
        bits = array.view(np.uint32)
        lsb = (bits >> 16) & np.uint32(1)
        rounded = (bits + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
        return (rounded >> np.uint32(16)).astype(np.uint16)
    raise ValueError(f"unsupported wire dtype {dtype!r}")


def decode_values(array, dtype: str):
    """Decode wire bytes into float32 for comparison."""

    import numpy as np

    if dtype == "fp32":
        return np.asarray(array, dtype=np.float32)
    if dtype == "fp16":
        return np.asarray(array, dtype=np.float16).astype(np.float32)
    if dtype == "bf16":
        bits = np.asarray(array, dtype=np.uint16).astype(np.uint32)
        return (bits << np.uint32(16)).view(np.float32)
    raise ValueError(f"unsupported wire dtype {dtype!r}")


def wire_itemsize(dtype: str) -> int:
    return {"fp32": 4, "fp16": 2, "bf16": 2}[dtype]


@dataclass(frozen=True)
class Case:
    """One measured payload shape."""

    op: str
    rows: int
    dtype: str
    hidden_size: int

    @property
    def count(self) -> int:
        return int(self.rows) * int(self.hidden_size)

    @property
    def payload_bytes(self) -> int:
        return self.count * wire_itemsize(self.dtype)

    def key(self) -> str:
        return f"{self.op}:rows{self.rows}:{self.dtype}"


def build_cases(
    *,
    hidden_size: int,
    dtypes: Sequence[str],
    rows: Sequence[int],
    prefill_rows: Sequence[int],
    ops: Sequence[str] = ("all_reduce", "broadcast"),
) -> list[Case]:
    """Build the decode rows first, then prefill chunks, for every dtype/op."""

    ordered_rows = [int(row) for row in rows]
    for row in prefill_rows:
        if int(row) not in ordered_rows:
            ordered_rows.append(int(row))
    cases: list[Case] = []
    for op in ops:
        if op not in {"all_reduce", "broadcast"}:
            raise ValueError(f"unknown collective op {op!r}")
        for dtype in dtypes:
            for row in ordered_rows:
                if int(row) < 1:
                    raise ValueError("rows must be positive")
                cases.append(Case(op=str(op), rows=int(row), dtype=str(dtype), hidden_size=int(hidden_size)))
    return cases


# ---------------------------------------------------------------------------
# PCIe link sampling
# ---------------------------------------------------------------------------


@dataclass
class LinkSampler:
    """Samples negotiated PCIe link state while traffic is running."""

    paths: dict[int, Path]
    interval_s: float = 0.05
    samples: dict[int, list[tuple[str, str]]] = field(default_factory=dict)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def __enter__(self) -> "LinkSampler":
        self._thread = threading.Thread(target=self._run, name="tp-link-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            for rank, path in self.paths.items():
                try:
                    width = (path / "current_link_width").read_text(encoding="utf-8").strip()
                    speed = (path / "current_link_speed").read_text(encoding="utf-8").strip()
                except OSError:
                    continue
                entry = (width, speed)
                bucket = self.samples.setdefault(rank, [])
                if not bucket or bucket[-1] != entry:
                    bucket.append(entry)
            self._stop.wait(self.interval_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            f"rank{rank}": {
                "observed": [
                    {"current_width_lanes": int(width), "current_speed": speed} for width, speed in samples
                ],
                "last_observed": (
                    {"current_width_lanes": int(samples[-1][0]), "current_speed": samples[-1][1]}
                    if samples
                    else None
                ),
            }
            for rank, samples in sorted(self.samples.items())
        }


# ---------------------------------------------------------------------------
# Peer screening
# ---------------------------------------------------------------------------


def screen_peer_access(
    devices: Sequence[int],
    *,
    sizes: Sequence[int],
    iterations: int = 20,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Test peer access and verified device-to-device copies in both directions."""

    import numpy as np

    from hipengine.core.device import Device, scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        host_array_ptr,
        malloc,
    )

    runtime = get_hip_runtime()
    result: dict[str, Any] = {
        "can_access_peer": {},
        "peer_access_enabled": [],
        "errors": [],
        "copies": {},
    }
    for source in devices:
        for destination in devices:
            if source == destination:
                continue
            key = f"{source}->{destination}"
            try:
                result["can_access_peer"][key] = bool(runtime.device_can_access_peer(source, destination))
            except Exception as error:  # noqa: BLE001
                result["errors"].append(f"can_access_peer {key}: {error!r}")
    for source in devices:
        for destination in devices:
            if source == destination:
                continue
            try:
                with scoped_current_device(runtime, source):
                    runtime.device_enable_peer_access(destination)
                result["peer_access_enabled"].append(f"{source}->{destination}")
            except Exception as error:  # noqa: BLE001
                result["errors"].append(f"enable_peer_access {source}->{destination}: {error!r}")

    for size in sizes:
        size = int(size)
        entry: dict[str, Any] = {"bytes": size}
        src_buffers = {device: malloc(size, device=Device("hip", device)) for device in devices}
        dst_buffers = {device: malloc(size, device=Device("hip", device)) for device in devices}
        try:
            values = np.arange(size // 4, dtype=np.float32) if size % 4 == 0 else np.zeros(size, dtype=np.uint8)
            host_pattern = values
            for device in devices:
                copy_host_array_to_device(src_buffers[device], host_pattern)

            readback = np.empty_like(host_pattern)
            for source in devices:
                for destination in devices:
                    if source == destination:
                        continue
                    key = f"{source}->{destination}"
                    try:
                        runtime.memcpy_peer(
                            dst_buffers[destination].ptr,
                            destination,
                            src_buffers[source].ptr,
                            source,
                            size,
                        )
                        copy_device_to_host(host_array_ptr(readback), dst_buffers[destination])
                        matches = bool(np.array_equal(readback, host_pattern))
                        latencies: list[float] = []
                        for _ in range(max(1, int(iterations))):
                            start = time.perf_counter()
                            runtime.memcpy_peer(
                                dst_buffers[destination].ptr,
                                destination,
                                src_buffers[source].ptr,
                                source,
                                size,
                            )
                            latencies.append((time.perf_counter() - start) * 1e3)
                        summary = summarize_samples(latencies)
                        summary["verified_bytes_match"] = matches
                        summary["bandwidth_gbs"] = bandwidth_gbs(payload_bytes=size, latency_ms=summary["p50_ms"])
                        entry.setdefault("unidirectional", {})[key] = summary
                    except Exception as error:  # noqa: BLE001
                        result["errors"].append(f"memcpy_peer {key} size {size}: {error!r}")
                        entry.setdefault("unidirectional", {})[key] = {"error": repr(error)}

            if len(devices) >= 2:
                source, destination = int(devices[0]), int(devices[1])
                streams: dict[int, int] = {}
                for device in (source, destination):
                    with scoped_current_device(runtime, device):
                        streams[device] = runtime.stream_create(nonblocking=True)
                try:
                    latencies = []
                    for _ in range(max(1, int(iterations))):
                        start = time.perf_counter()
                        with scoped_current_device(runtime, destination):
                            runtime.memcpy_peer_async(
                                dst_buffers[destination].ptr,
                                destination,
                                src_buffers[source].ptr,
                                source,
                                size,
                                streams[destination],
                            )
                        with scoped_current_device(runtime, source):
                            runtime.memcpy_peer_async(
                                dst_buffers[source].ptr,
                                source,
                                src_buffers[destination].ptr,
                                destination,
                                size,
                                streams[source],
                            )
                        for device in (source, destination):
                            with scoped_current_device(runtime, device):
                                runtime.stream_synchronize(streams[device])
                        latencies.append((time.perf_counter() - start) * 1e3)
                    summary = summarize_samples(latencies)
                    summary["bandwidth_gbs"] = bandwidth_gbs(payload_bytes=size * 2, latency_ms=summary["p50_ms"])
                    entry["bidirectional"] = summary
                except Exception as error:  # noqa: BLE001
                    result["errors"].append(f"bidirectional peer copy size {size}: {error!r}")
                    entry["bidirectional"] = {"error": repr(error)}
                finally:
                    for device in (source, destination):
                        with scoped_current_device(runtime, device):
                            runtime.stream_destroy(streams[device])
        except Exception as error:  # noqa: BLE001
            result["errors"].append(f"copy screening size {size} failed: {error!r}")
        finally:
            for buffer in (*src_buffers.values(), *dst_buffers.values()):
                free(buffer, runtime=runtime)
        result["copies"][str(size)] = entry
    return result


# ---------------------------------------------------------------------------
# Collective benchmark
# ---------------------------------------------------------------------------


def _collective_call(transport, case: Case, rank: int, send_ptr: int, recv_ptr: int, root: int) -> None:
    if case.op == "all_reduce":
        transport.all_reduce_sum(rank, send_ptr, recv_ptr, count=case.count, dtype=case.dtype)
    else:
        transport.broadcast(rank, send_ptr, recv_ptr, count=case.count, dtype=case.dtype, root=root)


def _enqueue_single_thread(
    *,
    transport,
    runtime,
    case: Case,
    send: Sequence[Any],
    recv: Sequence[Any],
    producer: Sequence[Any],
    consumer: Sequence[Any],
    chain_depth: int,
    root: int,
) -> None:
    from hipengine.core.device import scoped_current_device

    transport.group_start()
    try:
        for _ in range(max(1, int(chain_depth))):
            for rank in range(transport.world_size):
                with scoped_current_device(runtime, transport.devices[rank].index):
                    runtime.memset_async(producer[rank].ptr, 0, 4, transport.stream(rank))
                    _collective_call(transport, case, rank, send[rank].ptr, recv[rank].ptr, root)
                    runtime.memset_async(consumer[rank].ptr, 0, 4, transport.stream(rank))
    finally:
        transport.group_end()


def _enqueue_threaded(
    *,
    transport,
    runtime,
    case: Case,
    send: Sequence[Any],
    recv: Sequence[Any],
    producer: Sequence[Any],
    consumer: Sequence[Any],
    chain_depth: int,
    root: int,
) -> None:
    """One thread per rank; each thread owns its RCCL group context."""

    from hipengine.core.device import scoped_current_device

    errors: list[BaseException | None] = [None] * transport.world_size

    def worker(rank: int) -> None:
        try:
            with scoped_current_device(runtime, transport.devices[rank].index):
                transport.group_start()
                try:
                    for _ in range(max(1, int(chain_depth))):
                        runtime.memset_async(producer[rank].ptr, 0, 4, transport.stream(rank))
                        _collective_call(transport, case, rank, send[rank].ptr, recv[rank].ptr, root)
                        runtime.memset_async(consumer[rank].ptr, 0, 4, transport.stream(rank))
                finally:
                    transport.group_end()
        except BaseException as error:  # noqa: BLE001 - re-raised on the caller thread
            errors[rank] = error

    threads = [
        threading.Thread(target=worker, args=(rank,), name=f"tp-enqueue-{rank}")
        for rank in range(transport.world_size)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for rank, error in enumerate(errors):
        if error is not None:
            raise RuntimeError(f"threaded enqueue failed on rank {rank}: {error!r}") from error


def _capture_graph_probe(
    *,
    transport,
    runtime,
    case: Case,
    send: Sequence[Any],
    recv: Sequence[Any],
    producer: Sequence[Any],
    consumer: Sequence[Any],
    iterations: int,
    warmup: int,
    chain_depth: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Capture the collective group into a HIP graph per rank and time replay.

    This answers the Packet 1 question of whether RCCL work can be captured and
    replayed: communicator creation stays outside capture, each rank's whole
    group is captured on that rank's own stream, and the replayed result is
    checked against the graph-disabled (eager) result bit-for-bit.
    """

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import HIP_GRAPH_CAPTURE_MODE_RELAXED

    world = transport.world_size
    devices = transport.devices
    streams = [transport.stream(rank) for rank in range(world)]

    def enqueue_group() -> None:
        _enqueue_single_thread(
            transport=transport,
            runtime=runtime,
            case=case,
            send=send,
            recv=recv,
            producer=producer,
            consumer=consumer,
            chain_depth=int(chain_depth),
            root=0,
        )

    # Eager (graph-disabled) reference result.
    enqueue_group()
    transport.sync(timeout_s=timeout_s)
    reference = _snapshot_recv(case, recv)

    graphs: list[int] = []
    execs: list[int] = []
    node_counts: list[int] = []
    try:
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                runtime.stream_begin_capture(streams[rank], mode=HIP_GRAPH_CAPTURE_MODE_RELAXED)
        enqueue_group()
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                graph = runtime.stream_end_capture(streams[rank])
                graphs.append(graph)
                node_counts.append(len(runtime.graph_nodes(graph)))
                execs.append(runtime.graph_instantiate(graph))
    except Exception as error:  # noqa: BLE001 - capture support is the result
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                try:
                    runtime.stream_end_capture(streams[rank])
                except Exception:  # noqa: BLE001
                    pass
        for graph in graphs:
            try:
                runtime.graph_destroy(graph)
            except Exception:  # noqa: BLE001
                pass
        return {
            "captured": False,
            "error": f"{type(error).__name__}: {error}",
            "world_size": world,
            "chain_depth": int(chain_depth),
        }

    def launch_all() -> None:
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                runtime.graph_launch(execs[rank], streams[rank])

    def timed_launch() -> tuple[float, float]:
        start_events = []
        end_events = []
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                start_events.append(runtime.event_create())
                end_events.append(runtime.event_create())
                runtime.event_record(start_events[rank], streams[rank])
        host_start = time.perf_counter()
        launch_all()
        host_end = time.perf_counter()
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                runtime.event_record(end_events[rank], streams[rank])
        transport.sync(timeout_s=timeout_s)
        per_rank = []
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                per_rank.append(runtime.event_elapsed_time_ms(start_events[rank], end_events[rank]))
                runtime.event_destroy(start_events[rank])
                runtime.event_destroy(end_events[rank])
        return max(per_rank), (host_end - host_start) * 1e3

    try:
        # Poison the receive buffers so a graph that silently does nothing
        # cannot pass the replay check by leaving the eager result in place.
        _poison_recv(case, recv)
        for _ in range(max(0, int(warmup))):
            launch_all()
            transport.sync(timeout_s=timeout_s)
        latencies: list[float] = []
        launch_ms: list[float] = []
        for _ in range(max(1, int(iterations))):
            device_ms, host_ms = timed_launch()
            latencies.append(device_ms)
            launch_ms.append(host_ms)
        replayed = _snapshot_recv(case, recv)
        summary = summarize_samples(latencies)
        summary["launch"] = summarize_samples(launch_ms)
        summary["bandwidth_gbs"] = bandwidth_gbs(payload_bytes=case.payload_bytes, latency_ms=summary["p50_ms"])
        return {
            "captured": True,
            "world_size": world,
            "chain_depth": int(chain_depth),
            "graph_nodes": node_counts,
            "replay_matches_eager": all(
                replayed[rank] == reference[rank] for rank in range(world)
            ),
            **summary,
        }
    finally:
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                runtime.graph_exec_destroy(execs[rank])
        for graph in graphs:
            runtime.graph_destroy(graph)


def _poison_recv(case: Case, recv: Sequence[Any]) -> None:
    """Fill every receive buffer with a byte pattern that is never a result."""

    import numpy as np

    from hipengine.core.memory import copy_host_array_to_device

    poison = np.full(case.count * wire_itemsize(case.dtype), 0xFF, dtype=np.uint8)
    for buffer in recv:
        copy_host_array_to_device(buffer, poison)


def _snapshot_recv(case: Case, recv: Sequence[Any]) -> list[bytes]:
    """Read every rank's receive buffer back as raw bytes."""

    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    import numpy as np

    snapshots: list[bytes] = []
    for buffer in recv:
        raw = np.empty(case.count * wire_itemsize(case.dtype), dtype=np.uint8)
        copy_device_to_host(host_array_ptr(raw), buffer)
        snapshots.append(raw.tobytes())
    return snapshots


def _measure_case(
    *,
    transport,
    runtime,
    case: Case,
    send: Sequence[Any],
    recv: Sequence[Any],
    producer: Sequence[Any],
    consumer: Sequence[Any],
    iterations: int,
    warmup: int,
    enqueue_mode: str,
    chain_depth: int,
    timeout_s: float,
    root_order: str,
) -> dict[str, Any]:
    from hipengine.core.device import scoped_current_device

    world = transport.world_size
    devices = transport.devices
    root = (world - 1) if root_order == "reverse" else 0
    enqueue = _enqueue_threaded if enqueue_mode == "threaded" else _enqueue_single_thread

    def enqueue_group() -> None:
        enqueue(
            transport=transport,
            runtime=runtime,
            case=case,
            send=send,
            recv=recv,
            producer=producer,
            consumer=consumer,
            chain_depth=chain_depth,
            root=root,
        )

    def timed_iteration() -> tuple[float, list[float], float]:
        start_events = []
        end_events = []
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                start_events.append(runtime.event_create())
                end_events.append(runtime.event_create())
                runtime.event_record(start_events[rank], transport.stream(rank))
        host_start = time.perf_counter()
        enqueue_group()
        host_end = time.perf_counter()
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                runtime.event_record(end_events[rank], transport.stream(rank))
        transport.sync(timeout_s=timeout_s)
        per_rank = []
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                per_rank.append(runtime.event_elapsed_time_ms(start_events[rank], end_events[rank]))
                runtime.event_destroy(start_events[rank])
                runtime.event_destroy(end_events[rank])
        return max(per_rank), per_rank, (host_end - host_start) * 1e3

    for _ in range(max(0, int(warmup))):
        enqueue_group()
        transport.sync(timeout_s=timeout_s)

    latencies: list[float] = []
    per_rank_latencies: dict[int, list[float]] = {rank: [] for rank in range(world)}
    enqueue_ms: list[float] = []
    for _ in range(max(1, int(iterations))):
        exposed, per_rank, host_enqueue_ms = timed_iteration()
        latencies.append(exposed)
        enqueue_ms.append(host_enqueue_ms)
        for rank, value in enumerate(per_rank):
            per_rank_latencies[rank].append(value)

    summary = summarize_samples(latencies)
    summary["bandwidth_gbs"] = bandwidth_gbs(payload_bytes=case.payload_bytes, latency_ms=summary["p50_ms"])
    summary["bus_bandwidth_gbs"] = bus_bandwidth_gbs(
        payload_bytes=case.payload_bytes, latency_ms=summary["p50_ms"], world_size=world
    )
    summary["enqueue"] = summarize_samples(enqueue_ms)
    summary["per_rank_p50_ms"] = {str(rank): percentile(values, 0.5) for rank, values in per_rank_latencies.items()}
    summary["rank_skew_p50_ms"] = max(summary["per_rank_p50_ms"].values()) - min(
        summary["per_rank_p50_ms"].values()
    )
    return summary


def _verify_case(
    *,
    transport,
    case: Case,
    send: Sequence[Any],
    recv: Sequence[Any],
    timeout_s: float,
) -> dict[str, Any]:
    """Run one collective and check the payload arithmetic on every rank."""

    import numpy as np

    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    world = transport.world_size
    transport.group_start()
    for rank in range(world):
        _collective_call(transport, case, rank, send[rank].ptr, recv[rank].ptr, 0)
    transport.group_end()
    transport.sync(timeout_s=timeout_s)

    if case.op == "all_reduce":
        expected = encode_values([float(sum(range(1, world + 1)))] * case.count, case.dtype)
        expected_f32 = decode_values(expected, case.dtype)
    else:
        expected_f32 = decode_values(encode_values([1.0] * case.count, case.dtype), case.dtype)

    report: dict[str, Any] = {"op": case.op, "dtype": case.dtype, "rows": case.rows, "ranks": {}}
    for rank in range(world):
        raw = np.empty(case.count * wire_itemsize(case.dtype), dtype=np.uint8)
        host = raw.view(np.float32) if case.dtype == "fp32" else raw.view(np.uint16)
        copy_device_to_host(host_array_ptr(host), recv[rank])
        actual = decode_values(host, case.dtype)
        matches = bool(np.array_equal(actual, expected_f32))
        report["ranks"][str(rank)] = {"verified": matches}
    report["verified"] = all(entry["verified"] for entry in report["ranks"].values())
    return report


def run_collective_bench(
    devices: Sequence[int],
    *,
    cases: Sequence[Case],
    iterations: int,
    warmup: int,
    enqueue_modes: Sequence[str],
    chain_depths: Sequence[int],
    timeout_s: float,
    root_orders: Sequence[str],
    graph_chain_depths: Sequence[int] = (),
    graph_iterations: int = 20,
) -> dict[str, Any]:
    """Measure broadcast/all-reduce for every case and enqueue mode."""

    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_host_array_to_device, free, malloc
    from hipengine.distributed.plan import DistributedPlan
    from hipengine.distributed.rccl import RcclTransport

    runtime = get_hip_runtime()
    plan = DistributedPlan.resolve(list(devices), hidden_size=cases[0].hidden_size, algorithm="rccl")
    transport = RcclTransport([spec.device for spec in plan.ranks], runtime=runtime, init_timeout_s=timeout_s)
    world = transport.world_size
    results: dict[str, Any] = {"cases": {}, "errors": []}
    try:
        for case in cases:
            entry: dict[str, Any] = {
                "op": case.op,
                "rows": case.rows,
                "dtype": case.dtype,
                "count": case.count,
                "payload_bytes": case.payload_bytes,
                "enqueue_modes": {},
            }
            send = [malloc(case.payload_bytes, device=Device("hip", rank)) for rank in range(world)]
            recv = [malloc(case.payload_bytes, device=Device("hip", rank)) for rank in range(world)]
            producer = [malloc(4, device=Device("hip", rank)) for rank in range(world)]
            consumer = [malloc(4, device=Device("hip", rank)) for rank in range(world)]
            try:
                for rank in range(world):
                    copy_host_array_to_device(send[rank], encode_values([float(rank + 1)] * case.count, case.dtype))
                for mode in enqueue_modes:
                    for root_order in root_orders:
                        for chain_depth in chain_depths:
                            key = f"{mode}:root_{root_order}:chain_{int(chain_depth)}"
                            try:
                                entry["enqueue_modes"][key] = _measure_case(
                                    transport=transport,
                                    runtime=runtime,
                                    case=case,
                                    send=send,
                                    recv=recv,
                                    producer=producer,
                                    consumer=consumer,
                                    iterations=int(iterations),
                                    warmup=int(warmup),
                                    enqueue_mode=mode,
                                    chain_depth=int(chain_depth),
                                    timeout_s=float(timeout_s),
                                    root_order=root_order,
                                )
                            except Exception as error:  # noqa: BLE001
                                results["errors"].append(f"{case.key()} {key}: {error!r}")
                                entry["enqueue_modes"][key] = {"error": repr(error)}
                                if transport.poisoned:
                                    raise
                if graph_chain_depths:
                    entry["graph_capture"] = {}
                    for graph_depth in graph_chain_depths:
                        try:
                            entry["graph_capture"][f"chain_{int(graph_depth)}"] = _capture_graph_probe(
                                transport=transport,
                                runtime=runtime,
                                case=case,
                                send=send,
                                recv=recv,
                                producer=producer,
                                consumer=consumer,
                                iterations=int(graph_iterations),
                                warmup=max(2, int(warmup) // 4),
                                chain_depth=int(graph_depth),
                                timeout_s=float(timeout_s),
                            )
                        except Exception as error:  # noqa: BLE001
                            results["errors"].append(f"{case.key()} graph chain_{graph_depth}: {error!r}")
                            entry["graph_capture"][f"chain_{int(graph_depth)}"] = {"captured": False, "error": repr(error)}
                            if transport.poisoned:
                                raise
                entry["verification"] = _verify_case(
                    transport=transport,
                    case=case,
                    send=send,
                    recv=recv,
                    timeout_s=float(timeout_s),
                )
            finally:
                for buffer in (*send, *recv, *producer, *consumer):
                    free(buffer, runtime=runtime)
            results["cases"][case.key()] = entry
    finally:
        transport.close()
    return results


# ---------------------------------------------------------------------------
# Artifact assembly
# ---------------------------------------------------------------------------


def resolve_link_paths(devices: Sequence[int]) -> dict[int, Path]:
    """Map HIP device index to its DRM device directory for link sampling."""

    from hipengine.core.hip import get_hip_runtime
    from scripts.tp_host_inventory import card_device_path

    runtime = get_hip_runtime()
    paths: dict[int, Path] = {}
    for index in devices:
        try:
            info = runtime.device_info(int(index))
        except Exception:  # noqa: BLE001
            continue
        path = card_device_path(info.pci_bus_id)
        if path is not None:
            paths[int(index)] = path
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", default="0,1", help="Ordered HIP device indices (rank order)")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--dtypes", default=",".join(DEFAULT_DTYPES))
    parser.add_argument("--rows", default=",".join(str(row) for row in DEFAULT_ROWS))
    parser.add_argument("--prefill-rows", default=",".join(str(row) for row in DEFAULT_PREFILL_ROWS))
    parser.add_argument("--ops", default="all_reduce,broadcast")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--chain-depth", default=str(DEFAULT_CHAIN_DEPTH), help="Comma list of chain depths")
    parser.add_argument("--enqueue", default="single,threaded", help="Comma list of single,threaded")
    parser.add_argument("--root-order", default="forward", help="Comma list of forward,reverse")
    parser.add_argument("--peer-sizes", default=",".join(str(size) for size in DEFAULT_PEER_SIZES))
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-completion deadline in seconds")
    parser.add_argument("--mode", default="all", choices=("all", "peer", "collective"))
    parser.add_argument(
        "--graph-chain-depth",
        default="",
        help="Comma list of chain depths to also capture into a HIP graph (empty disables the capture probe)",
    )
    parser.add_argument("--graph-iterations", type=int, default=20, help="Graph replay iterations per case")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from scripts.tp_host_inventory import build_inventory, parse_device_list

    devices = parse_device_list(args.devices)
    dtypes = [chunk.strip() for chunk in args.dtypes.split(",") if chunk.strip()]
    rows = [int(chunk) for chunk in args.rows.split(",") if chunk.strip()]
    prefill_rows = [int(chunk) for chunk in args.prefill_rows.split(",") if chunk.strip()]
    ops = [chunk.strip() for chunk in args.ops.split(",") if chunk.strip()]
    enqueue_modes = [chunk.strip() for chunk in args.enqueue.split(",") if chunk.strip()]
    root_orders = [chunk.strip() for chunk in args.root_order.split(",") if chunk.strip()]
    peer_sizes = [int(chunk) for chunk in args.peer_sizes.split(",") if chunk.strip()]
    chain_depths = [int(chunk) for chunk in str(args.chain_depth).split(",") if chunk.strip()]
    if not chain_depths or any(depth < 1 for depth in chain_depths):
        parser.error("--chain-depth entries must be positive integers")

    artifact: dict[str, Any] = {
        "kind": "tp_collective_bench",
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device_order": devices,
        "hidden_size": int(args.hidden_size),
        "iterations": int(args.iterations),
        "warmup": int(args.warmup),
        "chain_depths": chain_depths,
        "modes": {"enqueue": enqueue_modes, "root_order": root_orders},
    }
    artifact["host_inventory"] = build_inventory(devices)
    link_paths = resolve_link_paths(devices)
    artifact["pcie_link_paths"] = {str(rank): str(path) for rank, path in link_paths.items()}

    exit_code = 0
    try:
        if args.mode in ("all", "peer"):
            artifact["peer_screen"] = screen_peer_access(devices, sizes=peer_sizes, timeout_s=args.timeout)
        if args.mode in ("all", "collective"):
            cases = build_cases(
                hidden_size=int(args.hidden_size),
                dtypes=dtypes,
                rows=rows,
                prefill_rows=prefill_rows,
                ops=ops,
            )
            graph_depths = [int(chunk) for chunk in str(args.graph_chain_depth).split(",") if chunk.strip()]
            artifact["modes"]["graph_chain_depth"] = graph_depths
            with LinkSampler(link_paths) as sampler:
                artifact["collective"] = run_collective_bench(
                    devices,
                    cases=cases,
                    iterations=int(args.iterations),
                    warmup=int(args.warmup),
                    enqueue_modes=enqueue_modes,
                    chain_depths=chain_depths,
                    timeout_s=float(args.timeout),
                    root_orders=root_orders,
                    graph_chain_depths=graph_depths,
                    graph_iterations=int(args.graph_iterations),
                )
            artifact["pcie_under_load"] = sampler.to_dict()
    except Exception as error:  # noqa: BLE001 - artifact must record the failure
        artifact["fatal_error"] = repr(error)
        exit_code = 1

    payload = json.dumps(artifact, indent=2, sort_keys=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    else:
        print(payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
