#!/usr/bin/env python3
"""Per-segment device attribution for one TP2 layer, on one rank at a time.

The end-to-end traffic accounting says the TP2 default route reads 9.6757 GiB per
rank per decode token and takes 24.155 ms, while the same bytes at the same
rank's measured resident bandwidth would take 19.629 ms. The per-layer span
attribution (`scripts/tp2_stage_device_attribution.py`) shows that gap is spread
evenly over all 64 layers - but "evenly over layers" is not "evenly over
kernels", because a full-attention layer and a GDN layer read very different
numbers of bytes (140.65 MB and 158.69 MB per rank) in the same ~350 us.

This script splits one layer into its segments and times each one with HIP
events recorded on that rank's own stream:

    attention -> post-attention norm -> MLP shard chain -> residual add

Every segment is queued work on one device with no cross-rank dependency, so
each rank is measured alone and the spans are that rank's own device timeline.
Segments are taken from the session's production entry points
(``_run_*_attention_attn_only``, ``_add_norm_kernel``,
``MlpShardGroup.enqueue_rank_chain``), so the kernels are the ones the graphed
default runs, not a reconstruction.

The eager schedule is used because each segment must be an individual queued
launch to be separable by events; the graphed schedule collapses a layer into
one graph. That is a measurement instrument difference, not a route difference:
the same kernels, the same shard variant, the same pointers.

Bytes per segment come from the GGUF index for the probed layer (MLP bytes
halved for the degree-2 shard) plus the full-attention KV read at the probed
position, which is not a weight and is therefore absent from the index.

Usage:

    python3 scripts/tp2_layer_segment_attribution.py MODEL [REPLAYS] [--json OUT]
        [--position N] [--devices 0,1] [--full-attn-layer 3] [--gdn-layer 0]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import platform
import re
import statistics
import sys
import time

import numpy as np

from hipengine.core.device import scoped_current_device
from hipengine.distributed.tp2_generate import MlpTP2GenerationSession
from hipengine.runtime.qwen35_gguf_runner import gguf_bf16_add

from hipengine.loading.gguf import scan_gguf
from hipengine.runtime.qwen35_gguf_runner import _HipEventStageRecorder

#: Tensor suffixes the degree-2 route splits, and therefore the ones whose bytes
#: per rank are half of the full model's.
SHARDED_SUFFIXES = ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")

#: Which segment each tensor's bytes belong to.
SEGMENT_OF_SUFFIX = {
    "attn_q.weight": "attention",
    "attn_k.weight": "attention",
    "attn_v.weight": "attention",
    "attn_output.weight": "attention",
    "attn_gate.weight": "attention",
    "attn_qkv.weight": "attention",
    "attn_q_norm.weight": "attention",
    "attn_k_norm.weight": "attention",
    "attn_norm.weight": "attention",
    "ssm_a": "attention",
    "ssm_alpha.weight": "attention",
    "ssm_beta.weight": "attention",
    "ssm_conv1d.weight": "attention",
    "ssm_dt.bias": "attention",
    "ssm_norm.weight": "attention",
    "ssm_out.weight": "attention",
    "post_attention_norm.weight": "post_norm",
    "ffn_gate.weight": "mlp",
    "ffn_up.weight": "mlp",
    "ffn_down.weight": "mlp",
}

#: The byte-carrying segments, in layer order. ``prime`` is a fixed-cost launch
#: that absorbs the cold-start submission latency and carries no layer bytes.
SEGMENT_ORDER = ("attention", "post_norm", "mlp", "residual")
REPORT_ORDER = ("prime", "attention", "post_norm", "mlp", "residual")


def _layer_bytes(model: str, layer_id: int) -> dict[str, object]:
    """Per-rank bytes for one layer, by segment, from the GGUF index.

    Also returns the layer's per-token KV read (bf16 K plus V, one token), which
    is not a weight and is therefore absent from the index.
    """

    info = scan_gguf(model)
    metadata = info.metadata or {}
    kv_heads = int(metadata.get("qwen35.attention.head_count_kv", 0))
    key_length = int(metadata.get("qwen35.attention.key_length", 0))
    value_length = int(metadata.get("qwen35.attention.value_length", key_length))
    per_segment: collections.Counter[str] = collections.Counter()
    by_suffix: collections.Counter[str] = collections.Counter()
    for tensor in info.tensors:
        match = re.match(r"^blk\.(\d+)\.(.+)$", tensor.name)
        if match is None or int(match.group(1)) != layer_id:
            continue
        suffix = match.group(2)
        nbytes = int(tensor.nbytes)
        if suffix in SHARDED_SUFFIXES:
            nbytes //= 2
        segment = SEGMENT_OF_SUFFIX.get(suffix, "residual")
        per_segment[segment] += nbytes
        by_suffix[suffix] += nbytes
    return {
        "bytes": dict(per_segment),
        "by_suffix": dict(by_suffix),
        "total": int(sum(per_segment.values())),
        "kv_bytes_per_token": kv_heads * (key_length + value_length) * 2,
    }


def _kv_bytes(accounting: dict[str, object], layer_type: str, position: int) -> int:
    """The full-attention KV read for one layer at ``position`` (not a weight)."""

    if layer_type != "full_attention":
        return 0
    return int(accounting["kv_bytes_per_token"]) * int(position + 1)


def _measure(
    session: object,
    device: int,
    layer_id: int,
    layer_type: str,
    *,
    replays: int,
    position: int,
) -> dict[str, list[float]]:
    """Time one layer's segments on one rank, in eager mode.

    One recorder spans the whole replay, so no segment pays a cold-queue start.
    The runner's own marks land between the segment boundaries, and the recorder
    reports intervals in order, each ending at the mark it is named after. The
    intervals are therefore partitioned by walking that order: an internal
    interval belongs to the segment that is open, and the boundary interval that
    closes a segment belongs to that segment. A segment's time is the sum of its
    intervals, not the last one.
    """

    runtime = session.runtime  # noqa: SLF001
    runner = session._runners[device]  # noqa: SLF001
    scratch = session._scratches[device]  # noqa: SLF001
    group = session._shard_group  # noqa: SLF001
    stream = session._rank_stream(device)  # noqa: SLF001
    src, _dst = session._hidden[device]  # noqa: SLF001
    add_norm = session._add_norm_kernel(runner)  # noqa: SLF001

    # The position lives in a device buffer: upload it once, outside the timed
    # replays. Doing it per replay would fold two H2D copies into the first
    # interval of every replay.
    with scoped_current_device(runtime, device):
        scratch.set_full_attention_position(position, runtime)

    samples: dict[str, list[float]] = collections.defaultdict(list)
    for _ in range(replays):
        recorder = _HipEventStageRecorder(runtime, enabled=True, stream=stream)
        with scoped_current_device(runtime, device):
            recorder.start()
            # Prime: a tiny launch whose span absorbs the cold-start submission
            # latency, so the attention span starts from a busy queue.
            gguf_bf16_add(
                scratch.residual.ptr,
                scratch.attn_out.ptr,
                _dst,
                runner.hidden_size,
                stream=stream,
                runtime=runtime,
            )
            recorder.mark("segment_prime")
            if layer_type == "linear_attention":
                runner._run_linear_attention_attn_only(  # noqa: SLF001
                    layer_id,
                    src,
                    scratch.attn_out.ptr,
                    scratch,
                    stream=stream,
                    gpu_stage_recorder=recorder,
                )
            else:
                runner._run_full_attention_attn_only(  # noqa: SLF001
                    layer_id,
                    src,
                    scratch.attn_out.ptr,
                    scratch,
                    position=position,
                    stream=stream,
                    gpu_stage_recorder=recorder,
                )
            recorder.mark("segment_attention")
            add_norm(
                src,
                scratch.attn_out.ptr,
                runner.weights.layer(layer_id)
                .weight("post_attention_norm")
                .allocation()
                .tensor.ptr,
                scratch.post_norm.ptr,
                scratch.residual.ptr,
                1,
                runner.hidden_size,
                runner.weights.config.rms_norm_eps,
                stream=stream,
                runtime=runtime,
            )
            recorder.mark("segment_post_norm")
            group.enqueue_rank_chain(layer_id, device, scratch.post_norm.ptr)
            recorder.mark("segment_mlp")
            # The residual add's second operand is the exchange's bf16 output row
            # - the exact buffer and dtype the captured layer graph adds - so the
            # segment times the production kernel without needing the peer.
            gguf_bf16_add(
                scratch.residual.ptr,
                group.output_ptr(device),
                _dst,
                runner.hidden_size,
                stream=stream,
                runtime=runtime,
            )
            recorder.mark("segment_residual")

        totals: collections.Counter[str] = collections.Counter()
        fine: collections.Counter[str] = collections.Counter()
        current = "prime"
        for names, start_event, stop_event in list(recorder._intervals):  # noqa: SLF001
            runtime.event_synchronize(stop_event)
            ms = float(runtime.event_elapsed_time_ms(start_event, stop_event))
            name = names[0]
            totals[current] += ms
            if name.startswith("segment_"):
                closed = name[len("segment_") :]
                index = REPORT_ORDER.index(closed)
                current = (
                    REPORT_ORDER[index + 1] if index + 1 < len(REPORT_ORDER) else ""
                )
            else:
                fine[name] += ms
        recorder.close()
        for name in REPORT_ORDER:
            if name in totals:
                samples[name].append(totals[name])
        for name, value in fine.items():
            samples[f"mark:{name}"].append(value)
    return samples


def _device_names(session: object) -> dict[str, str]:
    """The runtime's own name for each participating device."""

    return {
        str(device): session.runtime.device_get_name(int(device))  # noqa: SLF001
        for device in session.devices  # noqa: SLF001
    }


def _capture_subgraph(
    session: object,
    device: int,
    layer_id: int,
    layer_type: str,
    *,
    which: str,
    position: int,
) -> tuple[int, int]:
    """Capture one layer half into its own graph; return (graph, graph exec).

    The eager split pays a host submission gap per launch; a captured half has
    none, so its replay is a clean device time for that half's kernels. Both
    halves are the production kernels with the production pointers - the same
    calls the session's own per-layer capture makes.
    """

    runtime = session.runtime  # noqa: SLF001
    runner = session._runners[device]  # noqa: SLF001
    scratch = session._scratches[device]  # noqa: SLF001
    group = session._shard_group  # noqa: SLF001
    stream = session._rank_stream(device)  # noqa: SLF001
    src, dst = session._hidden[device]  # noqa: SLF001
    with scoped_current_device(runtime, device):
        runtime.stream_begin_capture(stream)
        try:
            if which == "attention":
                if layer_type == "linear_attention":
                    runner._run_linear_attention_attn_only(  # noqa: SLF001
                        layer_id, src, scratch.attn_out.ptr, scratch, stream=stream
                    )
                else:
                    runner._run_full_attention_attn_only(  # noqa: SLF001
                        layer_id,
                        src,
                        scratch.attn_out.ptr,
                        scratch,
                        position=position,
                        stream=stream,
                    )
            elif which == "rest":
                session._add_norm_kernel(runner)(  # noqa: SLF001
                    src,
                    scratch.attn_out.ptr,
                    runner.weights.layer(layer_id)
                    .weight("post_attention_norm")
                    .allocation()
                    .tensor.ptr,
                    scratch.post_norm.ptr,
                    scratch.residual.ptr,
                    1,
                    runner.hidden_size,
                    runner.weights.config.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
                group.enqueue_rank_chain(layer_id, device, scratch.post_norm.ptr)
                gguf_bf16_add(
                    scratch.residual.ptr,
                    group.output_ptr(device),
                    dst,
                    runner.hidden_size,
                    stream=stream,
                    runtime=runtime,
                )
            else:
                raise ValueError(f"unknown sub-graph {which!r}")
        except Exception:
            leaked = runtime.stream_end_capture(stream)
            if leaked:
                runtime.graph_destroy(leaked)
            raise
        graph = runtime.stream_end_capture(stream)
        try:
            graph_exec = runtime.graph_instantiate(graph)
        except Exception:
            runtime.graph_destroy(graph)
            raise
    return int(graph), int(graph_exec)


def _replay_rotation(
    session: object,
    device: int,
    entries: list[tuple[str, int]],
    *,
    rotations: int,
) -> dict[str, list[float]]:
    """Replay a list of captured graphs round-robin and time each one.

    Rotating over several layers' sub-graphs is the point, not an inconvenience:
    one layer's shard weights (81 MB of MLP plus 60-85 MB of attention) fit in
    gfx1100's 96 MB Infinity Cache, so replaying a single layer measures L2
    bandwidth. Cycling over a set whose total exceeds the cache keeps every
    replay on the same DRAM path the real 64-layer loop uses.

    The whole rotation is queued before any event is awaited, so no replay pays
    a cold-queue start either.
    """

    runtime = session.runtime  # noqa: SLF001
    stream = session._rank_stream(device)  # noqa: SLF001
    samples: dict[str, list[float]] = {name: [] for name, _ in entries}
    for _ in range(rotations):
        pending: list[tuple[str, int, int]] = []
        for name, graph_exec in entries:
            with scoped_current_device(runtime, device):
                # Events are device-owned resources: create them under the
                # device that will record them.
                start = runtime.event_create()
                stop = runtime.event_create()
                pending.append((name, start, stop))
                runtime.event_record(start, stream)
                runtime.graph_launch(graph_exec, stream)
                runtime.event_record(stop, stream)
        for name, start, stop in pending:
            runtime.event_synchronize(stop)
            samples[name].append(float(runtime.event_elapsed_time_ms(start, stop)))
            with scoped_current_device(runtime, device):
                runtime.event_destroy(start)
                runtime.event_destroy(stop)
    return samples


def _measure_graphed_layer(
    session: object,
    devices: list[int],
    layer_id: int,
    *,
    replays: int,
) -> dict[int, list[float]]:
    """Reference: the layer as the graphed default replays it, all ranks together.

    A layer above 0 carries the prior slot's device exchange at its head, which
    spin-waits on the peer's flag. Every rank's graph for the same layer is
    therefore launched before any is awaited - the real loop's order - so the
    span is a valid layer wall per rank and includes that rank's own exchange
    wait.
    """

    runtime = session.runtime  # noqa: SLF001
    samples: dict[int, list[float]] = {device: [] for device in devices}
    for _ in range(replays):
        events: dict[int, tuple[int, int]] = {}
        try:
            for device in devices:
                stream = session._rank_stream(device)  # noqa: SLF001
                with scoped_current_device(runtime, device):
                    # Events are device-owned resources: create them under the
                    # device that will record them.
                    start = runtime.event_create()
                    stop = runtime.event_create()
                    events[device] = (start, stop)
                    runtime.event_record(start, stream)
                    runtime.graph_launch(
                        session._layer_execs[(layer_id, device)], stream  # noqa: SLF001
                    )
                    runtime.event_record(stop, stream)
            for device in devices:
                start, stop = events[device]
                runtime.event_synchronize(stop)
                samples[device].append(
                    float(runtime.event_elapsed_time_ms(start, stop))
                )
        finally:
            for device, (start, stop) in events.items():
                with scoped_current_device(runtime, device):
                    runtime.event_destroy(start)
                    runtime.event_destroy(stop)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("replays", nargs="?", type=int, default=24)
    parser.add_argument("--json", type=pathlib.Path, default=None)
    parser.add_argument(
        "--position",
        type=int,
        default=None,
        help=(
            "full-attention position for the probe; defaults to the session's "
            "own capture position, which is what the captured graphs bake in"
        ),
    )
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--full-attn-layer", type=int, default=None)
    parser.add_argument("--gdn-layer", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument(
        "--probe-layers",
        default=None,
        help=(
            "comma-separated layer ids to capture and rotate over; the default "
            "spans both layer types across 8 layers so the rotation cannot fit "
            "in the Infinity Cache"
        ),
    )
    args = parser.parse_args(argv)

    devices = [int(part) for part in args.devices.split(",") if part.strip()]
    captured: list[tuple[int, int, int]] = []
    session = MlpTP2GenerationSession(
        args.model,
        devices=tuple(devices),
        mode="tp2",
        max_sequence_length=256,
    )
    # Build the captured per-layer graphs (the production default) so the same
    # session can also replay a whole graphed layer as the closure reference.
    session._ensure_graph_schedule()  # noqa: SLF001 - diagnostic introspection
    if args.position is None:
        args.position = int(session._capture_position)  # noqa: SLF001
    for device in devices:
        with scoped_current_device(session.runtime, device):  # noqa: SLF001
            session._scratches[device].set_full_attention_position(  # noqa: SLF001
                args.position, session.runtime  # noqa: SLF001
            )
    config = session._config  # noqa: SLF001
    layer_types = [str(t) for t in config.layer_types]
    full_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    gdn_layers = [i for i, t in enumerate(layer_types) if t != "full_attention"]
    probes = [
        (
            int(args.full_attn_layer)
            if args.full_attn_layer is not None
            else (full_layers[0] if full_layers else 0),
            "full_attention",
        ),
        (
            int(args.gdn_layer)
            if args.gdn_layer is not None
            else (gdn_layers[0] if gdn_layers else 0),
            "linear_attention",
        ),
    ]

    print(
        f"host {platform.node()} | schedule {session.schedule} | "
        f"reduce {session.reduce_mode} | devices {devices} | position {args.position}"
    )
    result: dict[str, object] = {
        "kind": "tp2-layer-segment-attribution",
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "model": str(args.model),
        "schedule": session.schedule,
        "reduce_mode": session.reduce_mode,
        "devices": devices,
        "device_names": _device_names(session),
        "position": args.position,
        "replays": args.replays,
        "probes": [],
        "note": (
            "Eager schedule: each segment is an individually queued launch so HIP "
            "events can separate it; the graphed default runs the same kernels as "
            "one captured graph per layer. Spans are one rank's own device "
            "timeline (events recorded on that rank's stream), measured with the "
            "peer idle, so they exclude the cross-rank exchange entirely. "
            "Bandwidth denominators are index bytes per rank plus the "
            "full-attention KV read at the probed position. Two measurement "
            "limits: the eager segment split repeats one layer, so its weights "
            "are Infinity-Cache-resident and it pays a host submission gap per "
            "launch - use the DRAM-rotation halves (subgraph_*) for rates; and "
            "the whole-layer graphed replay launches both ranks back to back on "
            "an otherwise idle device, so its in-graph exchange spin pays a full "
            "round trip that the real pipeline hides - use the real loop's "
            "per-layer spans (scripts/tp2_stage_device_attribution.py) for the "
            "layer wall, not this replay."
        ),
    }

    # Capture each probe layer's two halves once, then replay them round-robin.
    # The default probe set spans both layer types across 8 layers, so the
    # rotation's working set (~1.2 GiB) stays far above the 96 MB Infinity Cache
    # and every replay streams from DRAM the way the real 64-layer loop does.
    if args.probe_layers:
        probe_layers = [int(part) for part in args.probe_layers.split(",")]
    else:
        # Four of each type, spread over the stack: the rotation must carry both
        # layer types (their attention halves differ by ~23 MB per rank) and
        # enough layers that its working set cannot fit in the Infinity Cache.
        gdn_pick = [i for i, t in enumerate(layer_types) if t != "full_attention"]
        full_pick = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        probe_layers = sorted(
            gdn_pick[:: max(len(gdn_pick) // 4, 1)][:4]
            + full_pick[:: max(len(full_pick) // 4, 1)][:4]
        )
    rotating: dict[tuple[int, int], dict[str, list[float]]] = {}
    for device in devices:
        entries: list[tuple[str, int]] = []
        for layer_id in probe_layers:
            layer_type = layer_types[layer_id]
            for which in ("attention", "rest"):
                try:
                    graph, graph_exec = _capture_subgraph(
                        session,
                        device,
                        layer_id,
                        layer_type,
                        which=which,
                        position=args.position,
                    )
                except Exception as error:  # noqa: BLE001 - optional
                    print(
                        f"  dev{device} L{layer_id} {which} sub-graph capture "
                        f"unavailable: {error}"
                    )
                    continue
                captured.append((device, graph, graph_exec))
                entries.append((f"L{layer_id}:{which}", graph_exec))
        if not entries:
            continue
        # Warm up on the same rotation so the timed pass starts warm in the
        # queue sense without becoming L2-resident.
        _replay_rotation(session, device, entries, rotations=args.warmups)
        timed = _replay_rotation(session, device, entries, rotations=args.replays)
        for key, values in timed.items():
            layer_id = int(key.split(":", 1)[0][1:])
            which = key.split(":", 1)[1]
            rotating.setdefault((device, layer_id), {})[which] = values

    for device in devices:
        for layer_id, layer_type in probes:
            accounting = _layer_bytes(str(args.model), layer_id)
            kv = _kv_bytes(accounting, layer_type, args.position)
            for _ in range(args.warmups):
                _measure(
                    session,
                    device,
                    layer_id,
                    layer_type,
                    replays=1,
                    position=args.position,
                )
            samples = _measure(
                session,
                device,
                layer_id,
                layer_type,
                replays=args.replays,
                position=args.position,
            )
            subgraph_ms: dict[str, list[float]] = dict(
                rotating.get((device, layer_id), {})
            )
            rotation_summary = {
                str(other): {
                    "layer_type": layer_types[other],
                    "attention_ms": (
                        round(float(np.median(values["attention"])), 4)
                        if "attention" in values
                        else None
                    ),
                    "rest_ms": (
                        round(float(np.median(values["rest"])), 4)
                        if "rest" in values
                        else None
                    ),
                    "attention_mb_per_rank": round(
                        (
                            int(
                                _layer_bytes(str(args.model), other)["bytes"].get(
                                    "attention", 0
                                )
                            )
                            + _kv_bytes(
                                _layer_bytes(str(args.model), other),
                                layer_types[other],
                                args.position,
                            )
                        )
                        / 1e6,
                        2,
                    ),
                    "rest_mb_per_rank": round(
                        (
                            int(
                                _layer_bytes(str(args.model), other)["bytes"].get(
                                    "post_norm", 0
                                )
                            )
                            + int(
                                _layer_bytes(str(args.model), other)["bytes"].get(
                                    "mlp", 0
                                )
                            )
                        )
                        / 1e6,
                        2,
                    ),
                }
                for (dev, other), values in sorted(rotating.items())
                if dev == device
            }
            result.setdefault("rotations", {})
            result["rotations"][str(device)] = rotation_summary  # type: ignore[index]
            try:
                graphed_all = _measure_graphed_layer(
                    session, devices, layer_id, replays=args.replays
                )
            except Exception as error:  # noqa: BLE001 - the reference is optional
                print(f"  graphed layer replay unavailable: {error}")
                graphed_all = {}
            graphed = graphed_all.get(device, [])
            rows = []
            total_bytes = 0
            total_ms = 0.0
            for name in REPORT_ORDER:
                values = samples.get(name, [])
                if not values:
                    continue
                bytes_for_segment = int(accounting["bytes"].get(name, 0))
                if name == "attention":
                    bytes_for_segment += kv
                if name == "prime":
                    bytes_for_segment = 0
                ms = float(np.median(values))
                if name != "prime":
                    total_bytes += bytes_for_segment
                    total_ms += ms
                rows.append(
                    {
                        "segment": name,
                        "median_ms": round(ms, 4),
                        "min_ms": round(float(min(values)), 4),
                        "max_ms": round(float(max(values)), 4),
                        "stdev_ms": round(float(statistics.pstdev(values)), 4),
                        "mb_per_rank": round(bytes_for_segment / 1e6, 2),
                        "gb_per_s": (
                            round(bytes_for_segment / (ms / 1e3) / 1e9, 1)
                            if ms > 0
                            else None
                        ),
                    }
                )
            graphed_median = float(np.median(graphed)) if graphed else None
            entry = {
                "device": device,
                "device_name": _device_names(session)[str(device)],
                "layer_id": layer_id,
                "layer_type": layer_type,
                "segments": rows,
                "segment_sum_ms": round(total_ms, 4),
                "segment_sum_mb_per_rank": round(total_bytes / 1e6, 2),
                "segment_sum_gb_per_s": (
                    round(total_bytes / (total_ms / 1e3) / 1e9, 1)
                    if total_ms > 0
                    else None
                ),
                "graphed_layer_median_ms": (
                    round(graphed_median, 4) if graphed_median is not None else None
                ),
                "graphed_layer_min_ms": (
                    round(float(min(graphed)), 4) if graphed else None
                ),
                "graphed_layer_all_ranks_median_ms": {
                    str(d): round(float(np.median(v)), 4)
                    for d, v in sorted(graphed_all.items())
                },
                "segment_sum_over_graphed_ratio": (
                    round(total_ms / graphed_median, 3)
                    if graphed_median is not None
                    else None
                ),
                "kv_read_mb": round(kv / 1e6, 2),
                "subgraph_median_ms": {
                    name: round(float(np.median(values)), 4)
                    for name, values in sorted(subgraph_ms.items())
                },
                "subgraph_min_ms": {
                    name: round(float(min(values)), 4)
                    for name, values in sorted(subgraph_ms.items())
                },
                "subgraph_bytes_mb": {
                    "attention": round(
                        (int(accounting["bytes"].get("attention", 0)) + kv) / 1e6, 2
                    ),
                    "rest": round(
                        (
                            int(accounting["bytes"].get("post_norm", 0))
                            + int(accounting["bytes"].get("mlp", 0))
                        )
                        / 1e6,
                        2,
                    ),
                },
                "subgraph_gb_per_s": {
                    name: (
                        round(
                            (
                                int(accounting["bytes"].get("attention", 0)) + kv
                                if name == "attention"
                                else int(accounting["bytes"].get("post_norm", 0))
                                + int(accounting["bytes"].get("mlp", 0))
                            )
                            / (float(np.median(values)) / 1e3)
                            / 1e9,
                            1,
                        )
                        if float(np.median(values)) > 0
                        else None
                    )
                    for name, values in sorted(subgraph_ms.items())
                },
                "subgraph_sum_ms": (
                    round(
                        sum(float(np.median(v)) for v in subgraph_ms.values()), 4
                    )
                    if subgraph_ms
                    else None
                ),
                "fine_marks_ms": {
                    name.split(":", 1)[1]: round(float(np.median(values)), 4)
                    for name, values in sorted(samples.items())
                    if name.startswith("mark:")
                },
                "index_bytes_by_segment_mb": {
                    k: round(v / 1e6, 2)
                    for k, v in sorted(accounting["bytes"].items())
                },
            }
            result["probes"].append(entry)
            print(
                f"\n=== {_device_names(session)[str(device)]} L{layer_id} {layer_type} ==="
            )
            for row in rows:
                print(
                    f"  {row['segment']:10s} {row['median_ms']:7.3f} ms "
                    f"({row['min_ms']:.3f}-{row['max_ms']:.3f}, sd {row['stdev_ms']:.3f}) "
                    f"| {row['mb_per_rank']:7.2f} MB | {row['gb_per_s']:6.1f} GB/s"
                )
            print(
                f"  {'SUM(no prime)':10s} {total_ms:7.3f} ms | {total_bytes/1e6:7.2f} MB | "
                f"{total_bytes/(total_ms/1e3)/1e9:6.1f} GB/s"
            )
            if subgraph_ms:
                half_bytes = {
                    "attention": int(accounting["bytes"].get("attention", 0)) + kv,
                    "rest": int(accounting["bytes"].get("post_norm", 0))
                    + int(accounting["bytes"].get("mlp", 0)),
                }
                print(
                    "  captured halves (DRAM rotation): "
                    + ", ".join(
                        f"{name} {float(np.median(v)):.3f} ms / "
                        f"{half_bytes.get(name, 0)/1e6:.1f} MB / "
                        f"{half_bytes.get(name, 0)/(float(np.median(v))/1e3)/1e9:.0f} GB/s"
                        for name, v in sorted(subgraph_ms.items())
                    )
                    + f" | sum {sum(float(np.median(v)) for v in subgraph_ms.values()):.3f} ms"
                )
            fine = {
                name.split(":", 1)[1]: float(np.median(values))
                for name, values in sorted(samples.items())
                if name.startswith("mark:")
            }
            if fine:
                print(
                    "  fine marks: "
                    + ", ".join(f"{k} {v:.3f}" for k, v in fine.items())
                )
            if graphed_all:
                print(
                    f"  graphed layer replay (both ranks together): "
                    + ", ".join(
                        f"dev{d} {float(np.median(v)):.3f}"
                        for d, v in sorted(graphed_all.items())
                    )
                    + f" ms | segment sum / graphed = {total_ms/graphed_median:.3f}"
                )

    for device, graph, graph_exec in captured:
        with scoped_current_device(session.runtime, device):  # noqa: SLF001
            session.runtime.graph_exec_destroy(graph_exec)  # noqa: SLF001
            session.runtime.graph_destroy(graph)  # noqa: SLF001
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
