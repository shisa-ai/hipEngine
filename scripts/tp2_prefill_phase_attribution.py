"""TP2 bulk-prefill per-phase attribution via stream events.

The bulk prefill route runs, per layer, three production helpers on each rank:
the attention/GDN helper, the post-attention norm+residual helper, and the
batched sharded MLP with its staged exchange and residual add. The kernel
inventory (``tp2_prefill_kernel_inventory.py``) ranks kernels but cannot say how
much of the wall is *not* kernel execution, and its per-kernel durations are
demonstrably unreliable here: two profiled runs of the same prefill reported
kernel sums 1.7x apart at an identical region wall.

This tool answers the complementary question without a profiler. It wraps the
production layer helpers, records a stream event per phase boundary per layer
per rank *in stream order without intermediate syncs*, then calls the real
``bulk_prefill``.

Two timelines are reported, and keeping them apart is the point:

- **device** - each rank's own stream interval between the two boundaries. This
  is that rank's execution of the phase, including any bubble inside it.
- **host** - wall-clock deltas between the host crossing the same boundaries.
  Every boundary is crossed once, after the host has submitted that phase's work
  for *both* ranks, so the host spans partition the host timeline exactly as the
  device spans partition each stream timeline.

A phase whose host span is much larger than its device span is enqueue-bound:
the device is idle waiting for the host, and shrinking the kernels cannot help.
A phase whose device span is much larger than its host span is device-bound.
This distinction is why the host timeline is recorded at all - device event
spans alone cannot separate the two, and reading them as pure execution time
overstates what kernel work can recover.

Every repetition is kept as a *paired* record: one wall clock with the spans
measured inside that same wall. The headline row is the minimum-wall repetition
read together with its own spans. Mixing a minimum wall with another
repetition's spans would make the percentages and the unaccounted remainder
describe an execution that never happened.

The sharded MLP phase is split further, because "the MLP is slow" is not
actionable:

- ``mlp_chain``   - both ranks' shard chains (the quantized WMMA work);
- ``mlp_exchange``- the staged transport reduction;
- ``mlp_cast``    - the f32 -> bf16 boundary cast, which exists only on this
                    route;
- ``mlp_residual``- the single residual add.

``attn_reduce`` is the head-sharded route's second reduction per layer - the sum
of the ranks' attention-output partials that the post-attention norm consumes.
It is recorded on both routes so the two are read off the same phase table; on
the replicated route the helper is a no-op and the span is zero.

It also reports the group's own ``exchange_walls_s`` (the host-side wall of each
transport reduction), so a host-bound exchange can be told apart from a
device-bound one: if the device span is small while the host wall is large, the
reduce is limited by enqueue/synchronization rather than by device time.

Usage::

    python scripts/tp2_prefill_phase_attribution.py [--prompt-tokens 512]
        [--repeats 3] [--model MODEL] [--json OUT.json] [--per-layer]
        [--attention-shard]
"""

from __future__ import annotations
import pathlib

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

# Event slots per layer per rank. Every span is a pair of consecutive slots, so
# the numbering is the contract the report reads.
SLOT_LAYER_START = 0
SLOT_ATTN_STOP = 1
SLOT_ATTN_REDUCE_STOP = 2
SLOT_MLP_START = 3
SLOT_CHAIN_START = 4
SLOT_CHAIN_STOP = 5
SLOT_EXCHANGE_STOP = 6
SLOT_CAST_STOP = 7
SLOT_MLP_STOP = 8
SLOTS_PER_LAYER = 9
# The final norm plus the head, recorded once per prefill.
SLOT_TAIL_START = 9
SLOT_TAIL_STOP = 10
SLOTS_PER_PREFILL = SLOTS_PER_LAYER + 2

PHASES = (
    "attention",
    "attn_reduce",
    "norm_residual",
    "mlp",
    "mlp_chain",
    "mlp_exchange",
    "mlp_cast",
    "mlp_residual",
    "tail",
)
# ``mlp`` is the whole sharded-MLP phase and the four ``mlp_*`` entries are its
# decomposition, so only these four top-level phases may be summed: adding the
# breakdown as well double counts the MLP and reports a total above the wall.
TOP_LEVEL_PHASES = ("attention", "attn_reduce", "norm_residual", "mlp", "tail")
MLP_PARTS = ("mlp_chain", "mlp_exchange", "mlp_cast", "mlp_residual")
# The span pairs each phase reads, as (start slot, stop slot) offsets.
PHASE_SLOTS = {
    "attention": (SLOT_LAYER_START, SLOT_ATTN_STOP),
    "attn_reduce": (SLOT_ATTN_STOP, SLOT_ATTN_REDUCE_STOP),
    "norm_residual": (SLOT_ATTN_REDUCE_STOP, SLOT_MLP_START),
    "mlp": (SLOT_MLP_START, SLOT_MLP_STOP),
    "mlp_chain": (SLOT_CHAIN_START, SLOT_CHAIN_STOP),
    "mlp_exchange": (SLOT_CHAIN_STOP, SLOT_EXCHANGE_STOP),
    "mlp_cast": (SLOT_EXCHANGE_STOP, SLOT_CAST_STOP),
    "mlp_residual": (SLOT_CAST_STOP, SLOT_MLP_STOP),
}


class PhaseRecorder:
    """Wrap the production layer helpers and time each phase with stream events.

    Events are pre-created so the measured call never pays ``event_create``
    between enqueues, and recording is gated on ``enabled`` so the warmup pass
    through the same helpers does not consume slots.
    """

    def __init__(self, runtime: Any, session: Any) -> None:
        self.runtime = runtime
        self.session = session
        self.devices = tuple(int(device) for device in session.devices)
        self.layers = len(session._config.layer_types)
        self.enabled = False
        self.layer_index = 0
        self.group: Any = None
        self.wrapped_bulk_group = False
        self.device_reduce = False
        self._pools: dict[int, list[Any]] = {}
        self._originals: list[tuple[Any, str, Any]] = []
        # Host-side arrival time per absolute slot index. Shared across ranks
        # because a boundary is crossed once for both.
        self._host_stamps: dict[int, float] = {}

    # -- lifecycle ---------------------------------------------------------
    def install(self) -> None:
        for device in self.devices:
            from hipengine.core.device import scoped_current_device

            with scoped_current_device(self.runtime, device):
                self._pools[device] = [
                    self.runtime.event_create()
                    for _ in range(self.layers * SLOTS_PER_LAYER + 2)
                ]

        session = self.session
        # The bulk prefill drives its own group (``_bulk_shard_group``), built
        # separately from the decode group. Wrapping ``_shard_group`` instead
        # silently instruments a group the route never calls, which shows up as
        # unrecorded events rather than as an obvious error.
        group = getattr(session, "_bulk_shard_group", None) or getattr(
            session, "_shard_group", None
        )
        if group is None:
            raise SystemExit("this session has no shard group; it is not a TP2 route")
        self.group = group
        self.wrapped_bulk_group = group is getattr(session, "_bulk_shard_group", None)

        self._wrap(session, "_bulk_attention_layer", self._enter_attention, self._exit_attention)
        # The head-sharded route owes the layer a second reduction between its
        # attention partials and the norm that consumes their sum. It is called
        # on both routes (a no-op when attention is replicated), so recording it
        # unconditionally keeps one slot contract and reports the replicated
        # route's cost as the zero-width span it is.
        self._wrap(
            session,
            "_bulk_attention_reduce",
            self._enter_attention_reduce,
            self._exit_attention_reduce,
        )
        self._wrap(session, "_bulk_norm_residual_layer", self._enter_norm, self._exit_norm)
        self._wrap(session, "_bulk_sharded_mlp_layer", self._enter_mlp, self._exit_mlp)
        self._wrap(session, "_finish_bulk_prefill", self._enter_tail, self._exit_tail)
        self._wrap(group, "enqueue_chain", self._enter_chain, self._exit_chain)
        if getattr(group, "_device_exchange", None) is not None:
            # The device route has no transport reduce and no cast: the spin-add
            # kernel writes the bf16 boundary buffer directly. Recording both
            # boundaries here keeps the slot contract (and reports the cast as
            # the zero-width span it now is) instead of leaving events unset,
            # which surfaces as an invalid-resource-handle error.
            self.device_reduce = True
            self._wrap(group, "_forward_device_reduce", None, self._exit_device_reduce)
        else:
            self.device_reduce = False
            self._wrap(group._transport, "reduce", None, self._exit_exchange)
            self._wrap(group, "cast_reduced", None, self._exit_cast)

    def close(self) -> None:
        for owner, name, original in self._originals:
            setattr(owner, name, original)
        self._originals.clear()
        from hipengine.core.device import scoped_current_device

        for device, pool in self._pools.items():
            with scoped_current_device(self.runtime, device):
                for event in pool:
                    self.runtime.event_destroy(event)
        self._pools = {}

    def reset(self) -> None:
        self.layer_index = 0
        self._host_stamps.clear()

    def _wrap(self, owner: Any, name: str, before: Any, after: Any) -> None:
        original = getattr(owner, name)
        self._originals.append((owner, name, original))

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if before is not None:
                before(*args, **kwargs)
            try:
                return original(*args, **kwargs)
            finally:
                if after is not None:
                    after(*args, **kwargs)

        setattr(owner, name, wrapper)

    # -- recording ---------------------------------------------------------
    def _record(self, device: int, slot: int) -> None:
        if not self.enabled:
            return
        from hipengine.core.device import scoped_current_device

        stream = self.session._rank_stream(device)
        with scoped_current_device(self.runtime, device):
            self.runtime.event_record(self._pools[device][slot], stream)

    def _slot(self, slot: int) -> int:
        return self.layer_index * SLOTS_PER_LAYER + slot

    def _boundary_at(self, index: int) -> None:
        """Cross a phase boundary: one host stamp plus one event per rank.

        The host stamp is what separates device execution from an enqueue
        bubble. It is taken after the work for this phase has been submitted,
        so a phase's host span covers the submission of both ranks' work while
        its device span is a single rank's own stream interval. The gap between
        the two is the part of the phase the device spent waiting on the host.
        """
        if not self.enabled:
            return
        self._host_stamps[index] = time.perf_counter()
        for device in self.devices:
            self._record(device, index)

    def _boundary(self, slot: int) -> None:
        self._boundary_at(self._slot(slot))

    # -- phase hooks -------------------------------------------------------
    def _enter_attention(self, *_args: Any, **_kwargs: Any) -> None:
        if not self.enabled:
            return
        self._boundary(SLOT_LAYER_START)

    def _exit_attention(self, *_args: Any, **_kwargs: Any) -> None:
        self._boundary(SLOT_ATTN_STOP)

    def _enter_attention_reduce(self, *_args: Any, **_kwargs: Any) -> None:
        # attn_stop was recorded by the attention exit hook.
        return

    def _exit_attention_reduce(self, *_args: Any, **_kwargs: Any) -> None:
        self._boundary(SLOT_ATTN_REDUCE_STOP)

    def _enter_norm(self, *_args: Any, **_kwargs: Any) -> None:
        # The norm phase starts where the attention reduction stopped; no new
        # boundary needed.
        return

    def _exit_norm(self, *_args: Any, **_kwargs: Any) -> None:
        self._boundary(SLOT_MLP_START)

    def _enter_mlp(self, *_args: Any, **_kwargs: Any) -> None:
        # mlp_start was recorded by the norm exit hook.
        return

    def _exit_mlp(self, *_args: Any, **_kwargs: Any) -> None:
        self._boundary(SLOT_MLP_STOP)
        if self.enabled:
            self.layer_index += 1

    def _enter_chain(self, *_args: Any, **_kwargs: Any) -> None:
        self._boundary(SLOT_CHAIN_START)

    def _exit_chain(self, *_args: Any, **_kwargs: Any) -> None:
        self._boundary(SLOT_CHAIN_STOP)

    def _exit_exchange(self, *_args: Any, **_kwargs: Any) -> None:
        self._boundary(SLOT_EXCHANGE_STOP)

    def _exit_device_reduce(self, *_args: Any, **_kwargs: Any) -> None:
        """The device route's whole reduction, and its absent cast.

        ``_forward_device_reduce`` enqueues the staging copy, the flag publish
        and the spin-add for both ranks, so this span is the exchange. The cast
        boundary is recorded immediately after it because the kernel already
        wrote bf16: leaving it unset would make ``collect`` read an event that
        was never recorded.
        """

        self._boundary(SLOT_EXCHANGE_STOP)
        self._boundary(SLOT_CAST_STOP)

    def _exit_cast(self, device: Any, *_args: Any, **_kwargs: Any) -> None:
        # ``cast_reduced`` runs once per rank inside the group's forward, so this
        # records the calling rank's own cast boundary. The host stamp is
        # last-wins, so it lands after the final rank has been submitted.
        if not self.enabled:
            return
        index = self._slot(SLOT_CAST_STOP)
        self._host_stamps[index] = time.perf_counter()
        self._record(int(device), index)

    def _enter_tail(self, *_args: Any, **_kwargs: Any) -> None:
        self._boundary_at(self.layers * SLOTS_PER_LAYER + 0)

    def _exit_tail(self, *_args: Any, **_kwargs: Any) -> None:
        self._boundary_at(self.layers * SLOTS_PER_LAYER + 1)

    # -- reporting ---------------------------------------------------------
    def collect(self) -> dict[str, Any]:
        from hipengine.core.device import scoped_current_device

        spans: dict[int, dict[str, float]] = {
            device: {name: 0.0 for name in PHASES} for device in self.devices
        }
        per_layer: dict[int, list[dict[str, float]]] = {device: [] for device in self.devices}
        for device in self.devices:
            pool = self._pools[device]
            with scoped_current_device(self.runtime, device):
                for layer in range(self.layers):
                    base = layer * SLOTS_PER_LAYER
                    row = {
                        name: self.runtime.event_elapsed_time_ms(
                            pool[base + start], pool[base + stop]
                        )
                        for name, (start, stop) in PHASE_SLOTS.items()
                    }
                    per_layer[device].append(row)
                    for name, value in row.items():
                        spans[device][name] += value
                spans[device]["tail"] += self.runtime.event_elapsed_time_ms(
                    pool[self.layers * SLOTS_PER_LAYER + 0],
                    pool[self.layers * SLOTS_PER_LAYER + 1],
                )
        return {"spans": spans, "per_layer": per_layer, "host": self._host_spans()}

    def _host_spans(self) -> dict[str, float]:
        """Host-side spans for the same boundaries, in milliseconds.

        These are wall-clock deltas, so they are not attributable to a single
        rank: a boundary is crossed once, after both ranks' work for that phase
        has been submitted. They partition the host timeline, which is exactly
        what is needed to tell an enqueue-bound phase from a device-bound one.
        """
        stamps = self._host_stamps
        out = {name: 0.0 for name in PHASES}

        def gap(start: int, stop: int) -> float:
            if start in stamps and stop in stamps:
                return (stamps[stop] - stamps[start]) * 1000.0
            return 0.0

        for layer in range(self.layers):
            base = layer * SLOTS_PER_LAYER
            for name, (start, stop) in PHASE_SLOTS.items():
                out[name] += gap(base + start, base + stop)
        tail = self.layers * SLOTS_PER_LAYER
        out["tail"] += gap(tail + 0, tail + 1)
        return out


def _phase_sum(spans: dict[str, float]) -> float:
    return sum(spans[name] for name in TOP_LEVEL_PHASES)


def _rank_record(
    devices: tuple[int, ...],
    wall_ms: float,
    result: dict[str, Any],
    host_spans: dict[str, float],
) -> dict[str, Any]:
    """One repetition's wall paired with the spans measured inside it."""
    per_rank = {str(device): result["spans"][device] for device in devices}
    return {
        "wall_ms": round(wall_ms, 4),
        "prefill_tok_per_s": None,  # filled by the caller, which knows the tokens
        "phases": {
            device: {name: round(value, 4) for name, value in spans.items()}
            for device, spans in per_rank.items()
        },
        "mlp_parts": {
            device: {name: round(spans[name], 4) for name in MLP_PARTS}
            for device, spans in per_rank.items()
        },
        "host_phases": {name: round(value, 4) for name, value in host_spans.items()},
        "host_phase_sum_ms": round(sum(host_spans[name] for name in TOP_LEVEL_PHASES), 4),
        "totals": {
            device: {
                "phase_sum_ms": round(_phase_sum(spans), 4),
                "unaccounted_ms": round(wall_ms - _phase_sum(spans), 4),
                "share_of_wall": round(_phase_sum(spans) / wall_ms, 4),
            }
            for device, spans in per_rank.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--fractions", default=None, help="e.g. 0.44,0.56")
    parser.add_argument(
        "--logits-rows",
        type=int,
        default=1,
        help="rows the head projects; 1 is the product path and 512 is the full-row control",
    )
    parser.add_argument(
        "--attention-shard",
        action="store_true",
        help="head-shard the attention/GDN phase instead of replicating it",
    )
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="include every layer's spans in the JSON (large; for skew analysis)",
    )
    parser.add_argument(
        "--save-logits",
        type=Path,
        default=None,
        help="write the prefill's logits rows so two routes can be compared",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

    devices = tuple(int(part) for part in str(args.devices).split(","))
    runtime = get_hip_runtime()
    fractions = (
        tuple(float(part) for part in str(args.fractions).split(","))
        if args.fractions
        else None
    )

    session = MlpTP2GenerationSession(
        args.model,
        devices=devices,
        mode="tp2",
        max_sequence_length=int(args.max_sequence_length),
        uneven_split=fractions,
        bulk_prefill=True,
        attention_shard=bool(args.attention_shard),
    )
    prompt = [int(args.token_id)] * int(args.prompt_tokens)
    logits_rows = int(args.logits_rows)

    # Warm up through generate so the graph schedule, JIT variants and the
    # prompt-sized workspace all exist before the measured region.
    session.generate(prompt, max_new_tokens=2)
    runtime.device_synchronize()

    def prefill() -> float:
        runtime.device_synchronize()
        started = time.perf_counter()
        session.bulk_prefill(prompt, logits_rows=logits_rows)
        runtime.device_synchronize()
        return (time.perf_counter() - started) * 1000.0

    if args.save_logits is not None:
        # The logits are read once, outside the measured region: the two arms of
        # an attention-shard A/B have to be the same computation, and a timing
        # report cannot say that on its own.
        args.save_logits.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_logits, session.bulk_prefill(prompt, logits_rows=logits_rows))

    # Uninstrumented reference, measured before the wrappers exist at all: the
    # recorded run pays Python wrapper calls on every helper, and a phase share
    # read off an inflated wall would misattribute that overhead to the phases.
    prefill()  # one unrecorded prefill so the measured ones are steady-state
    uninstrumented = [prefill() for _ in range(int(args.repeats))]

    recorder = PhaseRecorder(runtime, session)
    recorder.install()
    # Read the counters from the group the recorder actually wrapped: the bulk
    # group is built lazily by the first bulk prefill, so capturing it before
    # that silently picks the decode group, whose exchange walls stay empty.
    group = recorder.group
    try:
        # One unrecorded prefill so every helper has run once with the wrappers
        # installed, then the recorded repeats.
        prefill()

        recorder.reset()
        recorder.enabled = True
        records: list[dict[str, Any]] = []
        per_layer_all: list[dict[int, list[dict[str, float]]]] = []
        try:
            for _ in range(int(args.repeats)):
                recorder.reset()
                runtime.device_synchronize()
                started = time.perf_counter()
                session.bulk_prefill(prompt, logits_rows=logits_rows)
                runtime.device_synchronize()
                wall_ms = (time.perf_counter() - started) * 1000.0
                result = recorder.collect()
                record = _rank_record(devices, wall_ms, result, result["host"])
                record["prefill_tok_per_s"] = round(
                    int(args.prompt_tokens) / (wall_ms / 1000.0), 3
                )
                records.append(record)
                per_layer_all.append(result["per_layer"])
        finally:
            recorder.enabled = False
    finally:
        recorder.close()

    # The headline is one repetition read whole: its own wall with its own
    # spans. Taking a minimum wall and another repetition's spans would report
    # percentages for an execution that never happened.
    headline_index = min(range(len(records)), key=lambda index: records[index]["wall_ms"])
    headline = records[headline_index]
    wall_ms = headline["wall_ms"]
    per_rank = headline["phases"]
    host_phases = headline["host_phases"]
    exchange_walls = list(getattr(group, "exchange_walls_s", []) or [])
    exchange_host = sum(exchange_walls) * 1000.0 / max(1, len(records))
    if not recorder.wrapped_bulk_group:
        raise SystemExit(
            "the recorder wrapped the decode group, not the bulk prefill group; "
            "the phase spans would describe the wrong route"
        )

    report: dict[str, Any] = {
        "schema": 2,
        "kind": "tp2_prefill_phase_attribution",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "model": str(args.model),
        "devices": list(devices),
        "fractions": None if fractions is None else list(fractions),
        "prompt_tokens": int(args.prompt_tokens),
        "logits_rows": logits_rows,
        "repeats": int(args.repeats),
        "layer_count": int(len(session._config.layer_types)),
        "reduce_mode": getattr(session, "reduce_mode", None),
        "device_reduce": bool(recorder.device_reduce),
        "headline_repeat": headline_index,
        "wall_ms": wall_ms,
        "wall_ms_samples": [round(record["wall_ms"], 3) for record in records],
        "prefill_tok_per_s": headline["prefill_tok_per_s"],
        "uninstrumented_wall_ms": [round(value, 3) for value in uninstrumented],
        "uninstrumented_wall_ms_min": round(min(uninstrumented), 3),
        # Added latency: how much *longer* the instrumented run took. The first
        # version subtracted the other way round and reported the wrapper calls
        # as making the prefill faster.
        "instrumentation_overhead_ms": round(wall_ms - min(uninstrumented), 3),
        "instrumentation_overhead_percent": round(
            100.0 * (wall_ms - min(uninstrumented)) / min(uninstrumented), 3
        ),
        "exchange_host_wall_ms_per_prefill": round(exchange_host, 3),
        "exchange_host_samples": len(exchange_walls),
        "mlp_parts": headline["mlp_parts"],
        "phases": per_rank,
        "host_phases": host_phases,
        "totals": headline["totals"],
        "repeats_paired": records,
    }
    if args.per_layer:
        report["per_layer"] = [
            {str(device): rows for device, rows in result.items()} for result in per_layer_all
        ]
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")

    print(f"prefill {args.prompt_tokens} tokens: wall {wall_ms:.1f} ms = "
          f"{report['prefill_tok_per_s']:.1f} tok/s "
          f"(paired samples {report['wall_ms_samples']})")
    print(f"headline repeat {headline_index}; uninstrumented "
          f"{min(uninstrumented):.1f} ms, instrumentation "
          f"{report['instrumentation_overhead_ms']:+.1f} ms "
          f"({report['instrumentation_overhead_percent']:+.2f}% added latency)")
    print(f"layers {report['layer_count']}  exchange host wall "
          f"{exchange_host:.2f} ms/prefill over {len(exchange_walls)} samples")
    print()
    header = (
        f"{'phase':<16}"
        + "".join(f"{'dev r' + str(d):>11}" for d in devices)
        + f"{'host':>10}{'% wall':>9}{'binding':>11}"
    )
    print(header)
    print("-" * len(header))
    for name in PHASES:
        values = [per_rank[str(device)][name] for device in devices]
        host_value = host_phases[name]
        # Which side is the constraint: the busier rank's device span or the
        # host span. Whichever is larger is what has to shrink.
        binding = "device" if max(values) > host_value else "host"
        print(
            f"{name:<16}"
            + "".join(f"{value:>11.2f}" for value in values)
            + f"{host_value:>10.2f}"
            + f"{100 * max(values) / wall_ms:>8.1f}%"
            + f"{binding:>11}"
        )
    print("-" * len(header))
    for device in devices:
        total = report["totals"][str(device)]
        print(
            f"{'sum':<16}{total['phase_sum_ms']:>11.2f}"
            + " " * (11 * (len(devices) - 1))
            + f"{headline['host_phase_sum_ms']:>10.2f}"
            + f"{100 * total['share_of_wall']:>8.1f}%"
        )
        print(f"{'unaccounted':<16}{total['unaccounted_ms']:>11.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
