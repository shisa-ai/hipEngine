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
import math
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
DEFAULT_DEPENDENT_DEPTHS = (1, 4, 16, 64, 128)

#: Dependent-chain structures. ``per_step`` is the copy-bearing original,
#: ``per_chain`` the single-group contrast case, and the alternating modes remove
#: the copy and then the per-reduction host submission. Only ``per_step``,
#: ``per_step_alternating`` and ``staged_exchange_host_sync`` pass the dependency
#: check, so only those margins are per-layer costs.
DEFAULT_CHAIN_MODES = (
    "per_step",
    "per_chain",
    "per_step_alternating",
    "single_group_alternating",
    "per_step_alternating_graph",
    "staged_exchange_host_sync",
    "staged_exchange_batched",
)

#: Modes that are measured only when asked for. Capturing a single group for the
#: whole chain faults the GPU at depth 32 (``Memory access fault ... Page not
#: present``), and the structure is invalid anyway because the reductions
#: collapse, so it is not part of a default run. The per-step capture, which is
#: valid and dependency-verified, is a default.
OPT_IN_CHAIN_MODES = ("single_group_alternating_graph",)


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


def dependent_chain_expected(seed: float, *, world_size: int, depth: int, dtype: str = "fp64") -> float:
    """Final value of a chain where each step reduces the previous result.

    Every rank holds ``seed`` before the first step. A step copies the previous
    reduction result into the next input on each rank, then sums those inputs
    across ``world_size`` ranks, so the value is multiplied by the world size per
    step. A chain that does not consume the previous result cannot produce this
    value, which is what makes the check a dependency test rather than a
    smoke test.

    ``dtype`` applies the case's own arithmetic, so a chain whose value exceeds
    the dtype's range expects an infinity rather than a finite number. The
    overflow is still evidence of dependency: a chain that re-reads an unchanged
    input returns the single-step value and never saturates.
    """

    value = float(seed) * float(world_size) ** int(depth)
    if dtype in {"fp32", "bf16"}:
        import numpy as np

        with np.errstate(over="ignore"):
            # bf16 shares fp32's exponent range, so the saturation point matches.
            return float(np.float32(value))
    return value


def chain_marginal_ms(samples: Sequence[tuple[int, float]]) -> dict[str, Any]:
    """Per-step cost from a depth ladder.

    ``samples`` is ``(depth, p50_ms)`` in ascending depth order. The marginal
    between the first and last point divides by the depth difference, so a fixed
    per-measurement overhead cancels instead of inflating the per-step cost.
    """

    points = sorted((int(depth), float(value)) for depth, value in samples)
    if len(points) < 2:
        return {"points": points, "marginal_us_per_step": None, "overall_us_per_step": None}
    per_segment = [
        {
            "from_depth": first_depth,
            "to_depth": second_depth,
            "marginal_us_per_step": (second_value - first_value) * 1e3 / (second_depth - first_depth),
        }
        for (first_depth, first_value), (second_depth, second_value) in zip(points, points[1:])
        if second_depth != first_depth
    ]
    first_depth, first_value = points[0]
    last_depth, last_value = points[-1]
    overall = None
    if last_depth != first_depth:
        overall = (last_value - first_value) * 1e3 / (last_depth - first_depth)
    return {
        "points": points,
        "segments": per_segment,
        "marginal_us_per_step": per_segment[-1]["marginal_us_per_step"] if per_segment else None,
        "overall_us_per_step": overall,
    }


def _chain_final_buffer_index(depth: int) -> int:
    """Which of the two alternating buffers holds the result after ``depth`` steps.

    Step ``i`` reads buffer ``i % 2`` and writes buffer ``(i + 1) % 2``, so the
    result of a ``depth``-step chain sits in buffer ``depth % 2``.
    """

    return int(depth) % 2


#: Poison value for the alternating buffers. The chain's values are positive
#: powers of the seed, so a graph that did nothing leaves this in place and fails
#: the value check instead of passing on a previous iteration's result.
CHAIN_POISON_VALUE = -1234567.0


def _measure_alternating_chain(
    *,
    transport,
    runtime,
    case: Case,
    buffers: Sequence[Sequence[Any]],
    depths: Sequence[int],
    iterations: int,
    warmup: int,
    seed: float,
    timeout_s: float,
    graph: bool,
    group_boundary: str = "per_step",
) -> dict[str, Any]:
    """Measure a dependency chain that needs no copy between reductions.

    The earlier protocol forced the dependency with a device-to-device copy: the
    reduction wrote one buffer and a copy moved it into the next step's input.
    Alternating the two buffers instead makes step ``i + 1`` read exactly what
    step ``i`` wrote, so the chain is dependency-bound with zero extra traffic
    and the measured cost is the collective sequence rather than the collective
    sequence plus a copy that a real layer would not perform.

    ``group_boundary`` selects where the transport's validation group is opened:

    - ``per_step`` opens one group per reduction. This is the structure in which
      each reduction is separately ordered after its predecessor, and it pays one
      host submission per reduction.
    - ``single`` puts the whole chain in one group. The reductions are still
      stream-ordered on each rank - step ``i + 1`` reads the buffer step ``i``
      wrote - so the dependency is carried by stream order rather than by the
      group boundary, and only one host submission is paid for the chain. This is
      also the only structure of the two that the transport can capture: opening
      and closing a group repeatedly inside a capture fails with
      ``HIP error 900: operation not permitted when stream is capturing``.

    With ``graph=True`` the whole sequence is captured once per rank on that
    rank's own stream and replayed. The result is checked against the eager
    alternating chain and against the closed form, so a replay that lost the
    dependency chain or did nothing at all is rejected rather than reported as a
    fast number.
    """

    if group_boundary not in ("per_step", "single"):
        raise ValueError("group_boundary must be 'per_step' or 'single'")

    import numpy as np

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import HIP_GRAPH_CAPTURE_MODE_RELAXED
    from hipengine.core.memory import copy_device_to_host, copy_host_array_to_device, host_array_ptr

    world = transport.world_size
    devices = transport.devices
    nbytes = int(case.payload_bytes)
    streams = [transport.stream(rank) for rank in range(world)]

    def read_value(rank: int, buffer_index: int) -> float:
        raw = np.empty(nbytes, dtype=np.uint8)
        host = raw.view(np.float32) if case.dtype == "fp32" else raw.view(np.uint16)
        with scoped_current_device(runtime, devices[rank].index):
            copy_device_to_host(host_array_ptr(host), buffers[buffer_index][rank])
        values = decode_values(host, case.dtype)
        return float(np.asarray(values, dtype=np.float64)[0])

    def seed_chain() -> None:
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                copy_host_array_to_device(
                    buffers[0][rank], encode_values([float(seed)] * case.count, case.dtype)
                )

    def poison_tail() -> None:
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                copy_host_array_to_device(
                    buffers[1][rank],
                    encode_values([CHAIN_POISON_VALUE] * case.count, case.dtype),
                )

    def enqueue_chain(depth: int, *, sync: bool = True) -> None:
        single = group_boundary == "single"
        if single:
            transport.group_start()
        try:
            for step in range(int(depth)):
                source = buffers[step % 2]
                destination = buffers[(step + 1) % 2]
                if not single:
                    transport.group_start()
                try:
                    for rank in range(world):
                        with scoped_current_device(runtime, devices[rank].index):
                            transport.all_reduce_sum(
                                rank,
                                source[rank].ptr,
                                destination[rank].ptr,
                                count=case.count,
                                dtype=case.dtype,
                            )
                finally:
                    if not single:
                        transport.group_end()
        finally:
            if single:
                transport.group_end()
        # A stream synchronize is refused while the stream is capturing, so the
        # capture path enqueues without waiting.
        if sync:
            transport.sync(timeout_s=timeout_s)

    mode_report: dict[str, Any] = {
        "group_boundary": (
            "one native group per reduction" if group_boundary == "per_step"
            else "one native group for the whole chain"
        ),
        "dependency_carried_by": (
            "the group boundary and stream order" if group_boundary == "per_step"
            else "stream order within one group"
        ),
        "consumes_predecessor_via": "alternating buffers",
        "device_copy_per_reduction": False,
        "graph_replay": bool(graph),
        "depths": {},
    }

    if not graph:
        for depth in depths:
            depth = max(1, int(depth))
            expected = dependent_chain_expected(
                seed, world_size=world, depth=depth, dtype=case.dtype
            )
            seed_chain()
            for _ in range(max(0, int(warmup))):
                enqueue_chain(depth)
            latencies: list[float] = []
            per_rank_latencies: dict[int, list[float]] = {rank: [] for rank in range(world)}
            for _ in range(max(1, int(iterations))):
                seed_chain()
                start_events: list[int] = []
                end_events: list[int] = []
                for rank in range(world):
                    with scoped_current_device(runtime, devices[rank].index):
                        start_events.append(runtime.event_create())
                        end_events.append(runtime.event_create())
                        runtime.event_record(start_events[rank], streams[rank])
                enqueue_chain(depth)
                for rank in range(world):
                    with scoped_current_device(runtime, devices[rank].index):
                        runtime.event_record(end_events[rank], streams[rank])
                transport.sync(timeout_s=timeout_s)
                per_rank: list[float] = []
                for rank in range(world):
                    with scoped_current_device(runtime, devices[rank].index):
                        per_rank.append(
                            runtime.event_elapsed_time_ms(start_events[rank], end_events[rank])
                        )
                        runtime.event_destroy(start_events[rank])
                        runtime.event_destroy(end_events[rank])
                latencies.append(max(per_rank))
                for rank, value in enumerate(per_rank):
                    per_rank_latencies[rank].append(value)
            final_index = _chain_final_buffer_index(depth)
            observed = [read_value(rank, final_index) for rank in range(world)]
            summary = summarize_samples(latencies)
            mode_report["depths"][str(depth)] = {
                **summary,
                "expected_final_value": expected,
                "observed_final_value": observed,
                "final_value_matches": all(value == expected for value in observed),
                # ``inf == inf`` proves nothing, so the verdict needs this flag.
                "value_check_informative": math.isfinite(float(expected)),
                "final_buffer": final_index,
                "per_rank_p50_ms": {
                    str(rank): percentile(values, 0.5)
                    for rank, values in per_rank_latencies.items()
                },
                "per_step_us": summary["p50_ms"] * 1e3 / depth,
            }
    else:
        mode_report["captured"] = False

        def capture(depth: int) -> tuple[list[int], list[int], list[int]] | None:
            """Capture one graph for one chain depth.

            A separate graph per depth is required for the ladder to mean
            anything: one deep graph replayed for every depth would report the
            deepest chain's time divided by a depth it never ran.
            """

            graphs: list[int] = []
            execs: list[int] = []
            node_counts: list[int] = []
            capturing: set[int] = set()
            try:
                for rank in range(world):
                    with scoped_current_device(runtime, devices[rank].index):
                        runtime.stream_begin_capture(
                            streams[rank], mode=HIP_GRAPH_CAPTURE_MODE_RELAXED
                        )
                        capturing.add(rank)
                enqueue_chain(depth, sync=False)
                for rank in range(world):
                    with scoped_current_device(runtime, devices[rank].index):
                        graph_handle = runtime.stream_end_capture(streams[rank])
                        capturing.discard(rank)
                        graphs.append(graph_handle)
                        node_counts.append(len(runtime.graph_nodes(graph_handle)))
                        execs.append(runtime.graph_instantiate(graph_handle))
            except Exception as error:  # noqa: BLE001 - capture support is the result
                for rank in sorted(capturing):
                    with scoped_current_device(runtime, devices[rank].index):
                        try:
                            graphs.append(runtime.stream_end_capture(streams[rank]))
                        except Exception:  # noqa: BLE001
                            continue
                for rank, exec_ in enumerate(execs):
                    with scoped_current_device(runtime, devices[rank].index):
                        try:
                            runtime.graph_exec_destroy(exec_)
                        except Exception:  # noqa: BLE001
                            pass
                for handle in graphs:
                    try:
                        runtime.graph_destroy(handle)
                    except Exception:  # noqa: BLE001
                        pass
                mode_report["error"] = f"depth {depth}: {type(error).__name__}: {error}"
                return None
            mode_report["captured"] = True
            return graphs, execs, node_counts

        def release(graphs: list[int], execs: list[int]) -> None:
            for rank, exec_ in enumerate(execs):
                with scoped_current_device(runtime, devices[rank].index):
                    runtime.graph_exec_destroy(exec_)
            for handle in graphs:
                runtime.graph_destroy(handle)

        for depth in depths:
            depth = max(1, int(depth))
            captured = capture(depth)
            if captured is None:
                # A failed capture can leave the transport unusable, so the rest
                # of the ladder is abandoned rather than reported as measured.
                break
            graphs, execs, node_counts = captured

            def launch_all() -> None:
                for rank in range(world):
                    with scoped_current_device(runtime, devices[rank].index):
                        runtime.graph_launch(execs[rank], streams[rank])

            try:
                expected = dependent_chain_expected(
                    seed, world_size=world, depth=depth, dtype=case.dtype
                )
                seed_chain()
                poison_tail()
                for _ in range(max(0, int(warmup))):
                    seed_chain()
                    poison_tail()
                    launch_all()
                    transport.sync(timeout_s=timeout_s)
                latencies: list[float] = []
                launch_ms: list[float] = []
                per_rank_latencies: dict[int, list[float]] = {
                    rank: [] for rank in range(world)
                }
                for _ in range(max(1, int(iterations))):
                    seed_chain()
                    poison_tail()
                    start_events: list[int] = []
                    end_events: list[int] = []
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
                            per_rank.append(
                                runtime.event_elapsed_time_ms(
                                    start_events[rank], end_events[rank]
                                )
                            )
                            runtime.event_destroy(start_events[rank])
                            runtime.event_destroy(end_events[rank])
                    latencies.append(max(per_rank))
                    launch_ms.append((host_end - host_start) * 1e3)
                    for rank, value in enumerate(per_rank):
                        per_rank_latencies[rank].append(value)
                final_index = _chain_final_buffer_index(depth)
                observed = [read_value(rank, final_index) for rank in range(world)]
                summary = summarize_samples(latencies)
                summary["launch"] = summarize_samples(launch_ms)
                mode_report["depths"][str(depth)] = {
                    **summary,
                    "expected_final_value": expected,
                    "observed_final_value": observed,
                    "final_value_matches": all(value == expected for value in observed),
                    # ``inf == inf`` proves nothing, so the verdict needs this flag.
                    "value_check_informative": math.isfinite(float(expected)),
                    "final_buffer": final_index,
                    "poison_value": CHAIN_POISON_VALUE,
                    "graph_nodes": node_counts,
                    "per_rank_p50_ms": {
                        str(rank): percentile(values, 0.5)
                        for rank, values in per_rank_latencies.items()
                    },
                    "per_step_us": summary["p50_ms"] * 1e3 / depth,
                }
            finally:
                # One graph per depth, so each is released before the next is
                # captured rather than accumulating handles across the ladder.
                release(graphs, execs)

    single_step_value = dependent_chain_expected(seed, world_size=world, depth=1, dtype=case.dtype)
    mode_report["depends_on_every_step"], reason = _dependency_verdict(
        mode_report["depths"], single_step_value=single_step_value
    )
    mode_report["dependency_verdict_depth"] = _deepest_informative_depth(mode_report["depths"])
    if reason:
        mode_report["dependency_verdict_reason"] = reason
    mode_report["marginal"] = chain_marginal_ms(
        [(int(depth), report["p50_ms"]) for depth, report in mode_report["depths"].items()]
    )
    return mode_report


def _deepest_informative_depth(depths: dict[str, Any]) -> str | None:
    """The deepest depth whose value check can still distinguish the chain.

    ``seed * world ** depth`` leaves the fp32 range at depth 128, so both the
    expected and the observed value are ``inf`` and the check passes trivially.
    The verdict has to come from a depth where the closed form is finite.
    """

    usable = [key for key, entry in depths.items() if entry.get("value_check_informative")]
    return max(usable, key=int) if usable else None


def _dependency_verdict(
    depths: dict[str, Any], *, single_step_value: float
) -> tuple[bool, str | None]:
    """Whether the deepest informative depth proves the chain compounded."""

    deepest = _deepest_informative_depth(depths)
    if deepest is None:
        return False, "no depth has a finite expected value, so no check is informative"
    values = depths[deepest]["observed_final_value"]
    compounded = all(value != single_step_value for value in values)
    matched = bool(depths[deepest]["final_value_matches"])
    reason = None
    if not matched:
        reason = f"depth {deepest} did not reach the closed form"
    elif not compounded:
        reason = f"depth {deepest} returned the single-step value"
    return bool(matched and compounded), reason


def _measure_staged_exchange_chain(
    *,
    transport,
    runtime,
    case: Case,
    buffers: Sequence[Sequence[Any]],
    depths: Sequence[int],
    iterations: int,
    warmup: int,
    seed: float,
    protocol: str = "serial",
) -> dict[str, Any]:
    """Measure a dependency-bound two-rank exchange that does not use RCCL.

    This is the specialized-transport screen: the peer-access path is unavailable
    on this host, so the alternative to RCCL is host staging. Each step moves the
    rank's own partial to a page-locked host slot, the host accumulates the two
    partials, and the result is copied back as the next step's input. The value
    check is the same closed form the RCCL chain uses, so a structure that does
    not consume its predecessor is rejected here too.

    Two orchestration protocols are measured, because they differ only in host
    submission structure and that structure is the thing under test:

    ``serial``
        For each rank in turn: submit the device-to-host copy and wait for it.
        Then accumulate on the host, then for each rank in turn submit the
        host-to-device copy and wait for it. Four host waits per reduction, and
        the two transfers of a pair never overlap.
    ``batched``
        Submit both device-to-host copies before waiting for either, wait for
        both, accumulate, then submit both host-to-device copies and wait for
        neither. Two host waits per reduction, and the paired transfers overlap.

    The batched form is correct without the host-to-device waits, and that is
    worth stating because dropping them looks like removing a dependency:

    * the host-to-device copy for step ``i`` and the device-to-host copy for step
      ``i + 1`` read and write different device buffers and run on the same rank
      stream, so stream order orders them;
    * the host write into a staging slot for step ``i + 1`` must not race the
      host-to-device copy that read that slot at step ``i``. It cannot: the slot
      is reused two steps later, the device-to-host copy for that step is
      enqueued after that host-to-device copy on the same stream, and the host
      waits for the device-to-host copy before accumulating. The wait the host
      already performs is therefore also the slot-reuse guard, so the extra
      host-to-device wait buys no ordering.

    A drain wait at the end of the chain covers the final host-to-device copy.

    The host round trip is inside the timed region on purpose: a host-synchronized
    exchange pays it per reduction, and hiding it would report a number the
    structure cannot deliver.
    """

    import ctypes

    import numpy as np

    from hipengine.core.device import scoped_current_device
    from hipengine.core.memory import host_buffer_ptr
    from hipengine.core.runtime import MemcpyKind

    if protocol not in ("serial", "batched"):
        raise ValueError(f"unknown staged protocol {protocol!r}")

    devices = tuple(getattr(transport, "devices", ()) or ())
    world = int(len(devices))
    if not world:
        return {"skipped": "transport exposes no devices"}
    nbytes = int(case.payload_bytes)
    streams = [0 for _ in range(world)]
    for rank in range(world):
        with scoped_current_device(runtime, devices[rank].index):
            streams[rank] = runtime.stream_create()

    # One slot per rank for the serial protocol; two per rank for the batched
    # protocol, because a slot cannot be rewritten until the copy that read it has
    # completed and the batched form defers that check by one step.
    slot_count = 1 if protocol == "serial" else 2
    host_slots = [
        [ctypes.create_string_buffer(nbytes) for _ in range(slot_count)]
        for _ in range(world)
    ]
    registered = False
    try:
        for slots in host_slots:
            for slot in slots:
                runtime.host_register(host_buffer_ptr(slot), nbytes)
        registered = True
    except Exception as error:  # noqa: BLE001 - the screen result is the failure
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                runtime.stream_destroy(streams[rank])
        return {
            "skipped": "host_register unavailable",
            "error": f"{type(error).__name__}: {error}",
        }

    #: Host-side phase counters, separated so submission cost and exposed wait are
    #: not reported as one number. ``device`` terms come from a separate event
    #: probe rather than from these.
    phases = {
        "d2h_submit_us": 0.0,
        "d2h_wait_us": 0.0,
        "h2d_submit_us": 0.0,
        "h2d_wait_us": 0.0,
        "host_sum_us": 0.0,
        "steps": 0,
    }

    def submit_d2h_rank(rank: int, step: int, slot: int) -> None:
        source = buffers[step % 2]
        with scoped_current_device(runtime, devices[rank].index):
            started = time.perf_counter()
            runtime.memcpy_async(
                host_buffer_ptr(host_slots[rank][slot]),
                source[rank].ptr,
                nbytes,
                MemcpyKind.DEVICE_TO_HOST,
                streams[rank],
            )
            phases["d2h_submit_us"] += (time.perf_counter() - started) * 1e6

    def submit_d2h(step: int, slot: int) -> None:
        for rank in range(world):
            submit_d2h_rank(rank, step, slot)

    def wait_rank(rank: int, *, d2h: bool) -> None:
        key = "d2h_wait_us" if d2h else "h2d_wait_us"
        with scoped_current_device(runtime, devices[rank].index):
            started = time.perf_counter()
            runtime.stream_synchronize(streams[rank])
            phases[key] += (time.perf_counter() - started) * 1e6

    def wait_streams(*, d2h: bool) -> None:
        for rank in range(world):
            wait_rank(rank, d2h=d2h)

    def accumulate(slot: int) -> None:
        started = time.perf_counter()
        partials = [
            np.frombuffer(host_slots[rank][slot], dtype=np.float32) for rank in range(world)
        ]
        reduced = partials[0].copy()
        # The closed form leaves the fp32 range at depth 128; that saturation is
        # expected and the value check reports it as uninformative there.
        with np.errstate(over="ignore"):
            for partial in partials[1:]:
                reduced += partial
        for rank in range(world):
            partials[rank][:] = reduced
        phases["host_sum_us"] += (time.perf_counter() - started) * 1e6

    def submit_h2d_rank(rank: int, step: int, slot: int) -> None:
        destination = buffers[(step + 1) % 2]
        with scoped_current_device(runtime, devices[rank].index):
            started = time.perf_counter()
            runtime.memcpy_async(
                destination[rank].ptr,
                host_buffer_ptr(host_slots[rank][slot]),
                nbytes,
                MemcpyKind.HOST_TO_DEVICE,
                streams[rank],
            )
            phases["h2d_submit_us"] += (time.perf_counter() - started) * 1e6

    def submit_h2d(step: int, slot: int) -> None:
        for rank in range(world):
            submit_h2d_rank(rank, step, slot)

    def staged_step(step: int) -> None:
        if protocol == "serial":
            # One rank at a time: submit and wait, so the two ranks' transfers of a
            # pair never overlap. This is the orchestration structure under test,
            # so each wait is for the rank that was just submitted, not for all
            # ranks on every iteration.
            for rank in range(world):
                submit_d2h_rank(rank, step, 0)
                wait_rank(rank, d2h=True)
            accumulate(0)
            for rank in range(world):
                submit_h2d_rank(rank, step, 0)
                wait_rank(rank, d2h=False)
            phases["steps"] += 1
            return
        slot = step % slot_count
        submit_d2h(step, slot)
        # This wait is the slot-reuse guard as well as the accumulation barrier.
        wait_streams(d2h=True)
        accumulate(slot)
        # No host-to-device wait: see the docstring for why stream order covers it.
        submit_h2d(step, slot)
        phases["steps"] += 1

    def enqueue_chain(depth: int) -> None:
        for step in range(int(depth)):
            staged_step(step)
        if protocol == "batched":
            wait_streams(d2h=False)

    def read_value(rank: int, buffer_index: int) -> float:
        from hipengine.core.memory import copy_device_to_host, host_array_ptr

        host = np.zeros(int(case.count), dtype=np.float32)
        with scoped_current_device(runtime, devices[rank].index):
            copy_device_to_host(host_array_ptr(host), buffers[buffer_index][rank])
        return float(np.asarray(host, dtype=np.float64)[0])

    def seed_chain() -> None:
        from hipengine.core.memory import copy_host_array_to_device

        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                copy_host_array_to_device(
                    buffers[0][rank], encode_values([float(seed)] * case.count, case.dtype)
                )

    def device_probe(steps: int = 3) -> dict[str, Any]:
        """Measure the transfer itself with events, on the first staging slot.

        The host-side counters above time submission and waiting, which includes
        driver overhead and scheduling, so they cannot say how long the copy took.
        This probe records an event around one copy per direction per rank and
        reports the device-side elapsed time.
        """

        probe_slot = 0
        samples: dict[str, list[float]] = {"d2h_us": [], "h2d_us": []}
        for step in range(max(1, int(steps))):
            per_rank: dict[str, list[float]] = {"d2h_us": [], "h2d_us": []}
            for rank in range(world):
                with scoped_current_device(runtime, devices[rank].index):
                    start = runtime.event_create()
                    end = runtime.event_create()
                    runtime.event_record(start, streams[rank])
                    runtime.memcpy_async(
                        host_buffer_ptr(host_slots[rank][probe_slot]),
                        buffers[step % 2][rank].ptr,
                        nbytes,
                        MemcpyKind.DEVICE_TO_HOST,
                        streams[rank],
                    )
                    runtime.event_record(end, streams[rank])
                    runtime.stream_synchronize(streams[rank])
                    per_rank["d2h_us"].append(runtime.event_elapsed_time_ms(start, end) * 1e3)
                    runtime.event_destroy(start)
                    runtime.event_destroy(end)
            accumulate(probe_slot)
            for rank in range(world):
                with scoped_current_device(runtime, devices[rank].index):
                    start = runtime.event_create()
                    end = runtime.event_create()
                    runtime.event_record(start, streams[rank])
                    runtime.memcpy_async(
                        buffers[(step + 1) % 2][rank].ptr,
                        host_buffer_ptr(host_slots[rank][probe_slot]),
                        nbytes,
                        MemcpyKind.HOST_TO_DEVICE,
                        streams[rank],
                    )
                    runtime.event_record(end, streams[rank])
                    runtime.stream_synchronize(streams[rank])
                    per_rank["h2d_us"].append(runtime.event_elapsed_time_ms(start, end) * 1e3)
                    runtime.event_destroy(start)
                    runtime.event_destroy(end)
            for key, values in per_rank.items():
                samples[key].extend(values)
        return {
            key: round(float(np.median(values)), 3) if values else None
            for key, values in samples.items()
        }

    report: dict[str, Any] = {
        "transport": "page-locked host staging with host accumulation",
        "protocol": protocol,
        "group_boundary": "not applicable (no RCCL group)",
        "dependency_carried_by": "stream order plus the host round trip",
        "device_copy_per_reduction": True,
        "host_round_trip_per_reduction": True,
        "host_waits_per_reduction": 4 if protocol == "serial" else 2,
        "accumulation": "host",
        "graph_replay": False,
        "phase_attribution": (
            "host-side counters separate submission from exposed wait; the "
            "transfer itself is measured separately by device_probe with events"
        ),
        "depths": {},
    }
    try:
        for depth in depths:
            depth = max(1, int(depth))
            expected = dependent_chain_expected(
                seed, world_size=world, depth=depth, dtype=case.dtype
            )
            seed_chain()
            for _ in range(max(0, int(warmup))):
                enqueue_chain(depth)
            latencies: list[float] = []
            for _ in range(max(1, int(iterations))):
                seed_chain()
                start = time.perf_counter()
                enqueue_chain(depth)
                latencies.append((time.perf_counter() - start) * 1e3)
            final_index = _chain_final_buffer_index(depth)
            observed = [read_value(rank, final_index) for rank in range(world)]
            summary = summarize_samples(latencies)
            report["depths"][str(depth)] = {
                **summary,
                "expected_final_value": expected,
                "observed_final_value": observed,
                "final_value_matches": all(value == expected for value in observed),
                # ``inf == inf`` proves nothing, so the verdict needs this flag.
                "value_check_informative": math.isfinite(float(expected)),
                "final_buffer": final_index,
                "per_step_us": summary["p50_ms"] * 1e3 / depth,
            }
        report["device_probe"] = device_probe()
    finally:
        if registered:
            for slots in host_slots:
                for slot in slots:
                    try:
                        runtime.host_unregister(host_buffer_ptr(slot))
                    except Exception:  # noqa: BLE001 - teardown must not mask a result
                        continue
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                runtime.stream_destroy(streams[rank])
    single_step_value = dependent_chain_expected(seed, world_size=world, depth=1, dtype=case.dtype)
    report["depends_on_every_step"], reason = _dependency_verdict(
        report["depths"], single_step_value=single_step_value
    )
    report["dependency_verdict_depth"] = _deepest_informative_depth(report["depths"])
    if reason:
        report["dependency_verdict_reason"] = reason
    report["marginal"] = chain_marginal_ms(
        [(int(depth), entry["p50_ms"]) for depth, entry in report["depths"].items()]
    )
    steps = max(1, int(phases["steps"]))
    report["phases_per_step_us"] = {
        key: phases[key] / steps
        for key in (
            "d2h_submit_us",
            "d2h_wait_us",
            "h2d_submit_us",
            "h2d_wait_us",
            "host_sum_us",
        )
    }
    report["phases_per_step_us"]["host_total_us"] = sum(
        report["phases_per_step_us"].values()
    )
    return report


def _chain_attribution(modes: dict[str, Any]) -> dict[str, Any]:
    """Split the per-reduction cost into copy, submission, and execution terms.

    The structures differ by exactly one term each, so their marginals decompose
    the cost: the original adds a device copy per reduction, the copy-free variant
    removes it, and the single-group variant removes one host submission per
    reduction. What remains in the captured replay is the collective itself plus
    rank waiting.
    """

    def marginal_us(mode: str) -> float | None:
        report = modes.get(mode) or {}
        value = (report.get("marginal") or {}).get("overall_us_per_step")
        return None if value is None else float(value)

    copy_bearing = marginal_us("per_step")
    copy_free = marginal_us("per_step_alternating")
    single_group = marginal_us("single_group_alternating")
    # Two different replays, and they answer different questions. The per-step
    # replay keeps the dependency-bearing device structure and only removes host
    # submission, so its delta is the host cost. The single-group replay removes
    # the dependency as well, so its delta measures the device structure too and
    # cannot be attributed to the host.
    replayed_per_step = marginal_us("per_step_alternating_graph")
    replayed_single_group = marginal_us("single_group_alternating_graph")
    staged = marginal_us("staged_exchange_host_sync")
    staged_batched = marginal_us("staged_exchange_batched")
    attribution: dict[str, Any] = {
        "per_step_us": copy_bearing,
        "copy_free_us": copy_free,
        "single_group_us": single_group,
        "graph_replay_per_step_us": replayed_per_step,
        "graph_replay_single_group_us": replayed_single_group,
        "staged_exchange_us": staged,
        "staged_exchange_batched_us": staged_batched,
    }
    if copy_bearing is not None and copy_free is not None:
        attribution["device_copy_us"] = copy_bearing - copy_free
    if copy_free is not None and replayed_per_step is not None:
        # Measured host submission: the same device structure, replayed.
        attribution["host_submission_us"] = copy_free - replayed_per_step
    if copy_free is not None and single_group is not None:
        # Not a host cost: this is what collapsing N dependent reductions into one
        # group saves on the device. It is reported because the difference is
        # large enough to be mistaken for submission overhead.
        attribution["per_step_over_single_group_us"] = copy_free - single_group
    if single_group is not None and replayed_single_group is not None:
        attribution["graph_launch_amortized_us"] = single_group - replayed_single_group
    if replayed_per_step is not None:
        attribution["collective_and_wait_us"] = replayed_per_step
    if copy_free is not None and replayed_per_step is not None and replayed_per_step > 0:
        attribution["per_step_over_graph_replay"] = copy_free / replayed_per_step
    if staged is not None and staged_batched is not None:
        attribution["staged_host_orchestration_us"] = staged - staged_batched
    return attribution


def _measure_dependent_chain(
    *,
    transport,
    runtime,
    case: Case,
    work: Sequence[Any],
    scratch: Sequence[Any],
    depths: Sequence[int],
    iterations: int,
    warmup: int,
    seed: float,
    timeout_s: float,
    modes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Measure a chain in which every reduction depends on the previous one.

    Each step is one native group containing one all-reduce per rank, followed on
    the same stream by a device copy that consumes the reduction result and
    produces the next step's input. Stream order therefore makes step ``i+1``
    depend on step ``i``, and the final value depends on every step.

    The earlier ladder put every reduction inside a single group and re-read
    unchanged inputs, so RCCL was free to aggregate them and the marginal
    latency could not be multiplied into a per-layer cost. This protocol keeps
    the group boundary at each sum and checks the result arithmetic, so a chain
    that does not consume its predecessor is rejected rather than measured.
    """

    import numpy as np

    from hipengine.core.device import scoped_current_device
    from hipengine.core.memory import copy_device_to_host, copy_host_array_to_device, host_array_ptr
    from hipengine.core.runtime import MemcpyKind

    if case.op != "all_reduce":
        return {"skipped": f"dependent chain is defined for all_reduce, not {case.op}"}

    world = transport.world_size
    devices = transport.devices
    nbytes = int(case.payload_bytes)

    def read_value(rank: int) -> float:
        raw = np.empty(nbytes, dtype=np.uint8)
        host = raw.view(np.float32) if case.dtype == "fp32" else raw.view(np.uint16)
        with scoped_current_device(runtime, devices[rank].index):
            copy_device_to_host(host_array_ptr(host), work[rank])
        values = decode_values(host, case.dtype)
        return float(np.asarray(values, dtype=np.float64)[0])

    def seed_chain() -> None:
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                copy_host_array_to_device(
                    work[rank], encode_values([float(seed)] * case.count, case.dtype)
                )

    def enqueue_chain(depth: int, group_mode: str) -> None:
        """Enqueue the chain, with a group boundary at each reduction.

        ``per_step`` opens and closes one native group per reduction. That is the
        only structure in which the reduction for step ``i+1`` is ordered after
        the consumer copy for step ``i``: a group defers its collectives until
        ``group_end``, so any work enqueued between ``group_start`` and
        ``group_end`` runs *before* every collective in that group. ``per_chain``
        is kept as the contrast case - it reproduces the batched structure the
        previous ladder measured, and its value check is expected to fail.
        """

        open_group = group_mode != "per_chain"
        if not open_group:
            transport.group_start()
        try:
            for step in range(int(depth)):
                if step:
                    for rank in range(world):
                        with scoped_current_device(runtime, devices[rank].index):
                            # The consumer: read the previous reduction result and
                            # produce this step's input on the same stream.
                            runtime.memcpy_async(
                                scratch[rank].ptr,
                                work[rank].ptr,
                                nbytes,
                                MemcpyKind.DEVICE_TO_DEVICE,
                                transport.stream(rank),
                            )
                if open_group:
                    transport.group_start()
                try:
                    for rank in range(world):
                        with scoped_current_device(runtime, devices[rank].index):
                            source = scratch[rank].ptr if step else work[rank].ptr
                            transport.all_reduce_sum(
                                rank, source, work[rank].ptr, count=case.count, dtype=case.dtype
                            )
                finally:
                    if open_group:
                        transport.group_end()
        finally:
            if not open_group:
                transport.group_end()
        transport.sync(timeout_s=timeout_s)

    def timed_chain(depth: int, group_mode: str) -> tuple[float, list[float]]:
        start_events: list[int] = []
        end_events: list[int] = []
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                start_events.append(runtime.event_create())
                end_events.append(runtime.event_create())
                runtime.event_record(start_events[rank], transport.stream(rank))
        enqueue_chain(depth, group_mode)
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                runtime.event_record(end_events[rank], transport.stream(rank))
        transport.sync(timeout_s=timeout_s)
        per_rank: list[float] = []
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                per_rank.append(runtime.event_elapsed_time_ms(start_events[rank], end_events[rank]))
                runtime.event_destroy(start_events[rank])
                runtime.event_destroy(end_events[rank])
        return max(per_rank), per_rank

    report: dict[str, Any] = {
        "op": case.op,
        "dtype": case.dtype,
        "rows": case.rows,
        "count": case.count,
        "payload_bytes": nbytes,
        "world_size": world,
        "seed": float(seed),
        "group_boundary": "one native group per reduction",
        "modes": {},
    }
    alternating_buffers = (work, scratch)
    selected = tuple(modes) if modes else DEFAULT_CHAIN_MODES
    for group_mode in [mode for mode in ("per_step", "per_chain") if mode in selected]:
        samples: list[tuple[int, float]] = []
        mode_report: dict[str, Any] = {
            "group_boundary": (
                "one native group per reduction" if group_mode == "per_step" else "one native group for the whole chain"
            ),
            "depths": {},
        }
        for depth in depths:
            depth = max(1, int(depth))
            expected = dependent_chain_expected(
                seed, world_size=world, depth=depth, dtype=case.dtype
            )
            seed_chain()
            for _ in range(max(0, int(warmup))):
                enqueue_chain(depth, group_mode)
            latencies: list[float] = []
            per_rank_latencies: dict[int, list[float]] = {rank: [] for rank in range(world)}
            for _ in range(max(1, int(iterations))):
                # Re-seed outside the timed region: without this the value keeps
                # compounding across measurement iterations and the expected
                # final value would depend on the iteration count. The seed copy
                # is a synchronous host-to-device write, so it cannot overlap.
                seed_chain()
                exposed, per_rank = timed_chain(depth, group_mode)
                latencies.append(exposed)
                for rank, value in enumerate(per_rank):
                    per_rank_latencies[rank].append(value)
            observed = [read_value(rank) for rank in range(world)]
            summary = summarize_samples(latencies)
            samples.append((depth, summary["p50_ms"]))
            mode_report["depths"][str(depth)] = {
                **summary,
                "expected_final_value": expected,
                "observed_final_value": observed,
                "final_value_matches": all(value == expected for value in observed),
                "value_check_informative": math.isfinite(float(expected)),
                "per_rank_p50_ms": {
                    str(rank): percentile(values, 0.5) for rank, values in per_rank_latencies.items()
                },
                "per_step_us": summary["p50_ms"] * 1e3 / depth,
            }

        # A chain that ignores its predecessor returns the single-step value.
        single_step_value = dependent_chain_expected(seed, world_size=world, depth=1, dtype=case.dtype)
        mode_report["depends_on_every_step"], reason = _dependency_verdict(
            mode_report["depths"], single_step_value=single_step_value
        )
        mode_report["dependency_verdict_depth"] = _deepest_informative_depth(mode_report["depths"])
        if reason:
            mode_report["dependency_verdict_reason"] = reason
        mode_report["marginal"] = chain_marginal_ms(samples)
        report["modes"][group_mode] = mode_report
    alternating_modes = (
        ("per_step_alternating", False, "per_step"),
        ("single_group_alternating", False, "single"),
        # Captured replay runs last: a capture that fails part-way can leave the
        # transport unusable, and the eager measurements above must not depend on
        # whether capture worked. ``per_step`` captures the dependency-bearing
        # structure, in which every reduction has its own group.
        ("per_step_alternating_graph", True, "per_step"),
        ("single_group_alternating_graph", True, "single"),
    )
    for key, graph, group_boundary in alternating_modes:
        if key not in selected:
            continue
        report["modes"][key] = _measure_alternating_chain(
            transport=transport,
            runtime=runtime,
            case=case,
            buffers=alternating_buffers,
            depths=depths,
            iterations=iterations,
            warmup=warmup,
            seed=seed,
            timeout_s=timeout_s,
            graph=graph,
            group_boundary=group_boundary,
        )
    # The specialized-transport screen replaces RCCL entirely for the exchange, so
    # it runs last and cannot perturb the RCCL measurements.
    for mode, protocol in (
        ("staged_exchange_host_sync", "serial"),
        ("staged_exchange_batched", "batched"),
    ):
        if mode not in selected:
            continue
        report["modes"][mode] = _measure_staged_exchange_chain(
            transport=transport,
            runtime=runtime,
            case=case,
            buffers=alternating_buffers,
            depths=depths,
            iterations=iterations,
            warmup=warmup,
            seed=seed,
            protocol=protocol,
        )
    report["attribution"] = _chain_attribution(report["modes"])
    return report


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
    capturing: set[int] = set()
    try:
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                runtime.stream_begin_capture(streams[rank], mode=HIP_GRAPH_CAPTURE_MODE_RELAXED)
                capturing.add(rank)
        enqueue_group()
        for rank in range(world):
            with scoped_current_device(runtime, devices[rank].index):
                graph = runtime.stream_end_capture(streams[rank])
                capturing.discard(rank)
                graphs.append(graph)
                node_counts.append(len(runtime.graph_nodes(graph)))
                execs.append(runtime.graph_instantiate(graph))
    except Exception as error:  # noqa: BLE001 - capture support is the result
        # Release every handle this probe created, in reverse creation order: an
        # executable references its graph, so the graph cannot be destroyed
        # first, and a partial failure leaves the ranks that already succeeded
        # holding live handles. A stream whose capture already ended must not be
        # ended again, so only still-capturing streams are closed here.
        for rank in sorted(capturing):
            with scoped_current_device(runtime, devices[rank].index):
                try:
                    pending = runtime.stream_end_capture(streams[rank])
                except Exception:  # noqa: BLE001
                    continue
                graphs.append(pending)
        for rank, exec_ in enumerate(execs):
            with scoped_current_device(runtime, devices[rank].index):
                try:
                    runtime.graph_exec_destroy(exec_)
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
    dependent_depths: Sequence[int] = DEFAULT_DEPENDENT_DEPTHS,
    dependent_modes: Sequence[str] | None = None,
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
            chain_work = [malloc(case.payload_bytes, device=Device("hip", rank)) for rank in range(world)]
            chain_scratch = [malloc(case.payload_bytes, device=Device("hip", rank)) for rank in range(world)]
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
                if dependent_depths:
                    try:
                        entry["dependent_chain"] = _measure_dependent_chain(
                            transport=transport,
                            runtime=runtime,
                            case=case,
                            work=chain_work,
                            scratch=chain_scratch,
                            depths=dependent_depths,
                            iterations=int(iterations),
                            warmup=max(1, int(warmup) // 4),
                            seed=1.0,
                            timeout_s=float(timeout_s),
                            modes=dependent_modes,
                        )
                    except Exception as error:  # noqa: BLE001
                        results["errors"].append(f"{case.key()} dependent chain: {error!r}")
                        entry["dependent_chain"] = {"error": repr(error)}
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
                for buffer in (*send, *recv, *producer, *consumer, *chain_work, *chain_scratch):
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
    parser.add_argument(
        "--dependent-depths",
        default=",".join(str(depth) for depth in DEFAULT_DEPENDENT_DEPTHS),
        help=(
            "Comma list of depths for the dependent reduction chain, in which each "
            "sum runs in its own group and consumes the previous result "
            "(empty disables it)"
        ),
    )
    parser.add_argument(
        "--dependent-modes",
        default=",".join(DEFAULT_CHAIN_MODES),
        help=(
            "Comma list of dependent-chain structures to measure. The single-group "
            "modes fail the dependency check on this host, and a failed graph capture "
            "can leave the transport unusable, so the safe subset is "
            "per_step,per_chain,per_step_alternating,staged_exchange_host_sync"
        ),
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
    dependent_depths = [int(chunk) for chunk in str(args.dependent_depths).split(",") if chunk.strip()]
    dependent_modes = tuple(chunk for chunk in str(args.dependent_modes).split(",") if chunk.strip())
    if any(depth < 1 for depth in dependent_depths):
        parser.error("--dependent-depths entries must be positive integers")

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
            artifact["modes"]["dependent_depths"] = dependent_depths
            artifact["modes"]["dependent_modes"] = list(dependent_modes or DEFAULT_CHAIN_MODES)
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
                    dependent_depths=dependent_depths,
                    dependent_modes=dependent_modes,
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
