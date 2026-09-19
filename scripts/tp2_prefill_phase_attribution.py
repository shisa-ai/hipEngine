"""TP2 bulk-prefill per-phase attribution via stream events.

The bulk prefill route runs, per layer, three production helpers on each rank:
the attention/GDN helper, the post-attention norm+residual helper, and the
batched sharded MLP with its staged exchange and residual add. The kernel
inventory (``tp2_prefill_kernel_inventory.py``) ranks kernels but cannot say how
much of the wall is *not* kernel execution, and its per-kernel durations are
demonstrably unreliable here: two profiled runs of the same prefill reported
kernel sums 1.7x apart at an identical region wall.

This tool answers the complementary question without a profiler. It wraps the
production layer helpers, records two stream events per phase boundary per layer
per rank *in stream order without intermediate syncs*, then calls the real
``bulk_prefill``. Each span is therefore the device execution of that phase on
that rank, including any bubble inside it, and the difference between the host
wall and the summed spans is the host-side/enqueue component.

The sharded MLP phase is split further, because "the MLP is slow" is not
actionable:

- ``mlp_chain``   - both ranks' shard chains (the quantized WMMA work);
- ``mlp_exchange``- the staged transport reduction;
- ``mlp_cast``    - the f32 -> bf16 boundary cast, which exists only on this
                    route;
- ``mlp_residual``- the single residual add.

It also reports the group's own ``exchange_walls_s`` (the host-side wall of each
transport reduction), so a host-bound exchange can be told apart from a
device-bound one: if the device span is small while the host wall is large, the
reduce is limited by enqueue/synchronization rather than by device time.

Usage::

    python scripts/tp2_prefill_phase_attribution.py [--prompt-tokens 512]
        [--repeats 3] [--model MODEL] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/lhl/hipEngine-main")
import numpy as np  # noqa: E402

# Event slots per layer per rank. Every span is a pair of consecutive slots, so
# the numbering is the contract the report reads.
SLOT_LAYER_START = 0
SLOT_ATTN_STOP = 1
SLOT_MLP_START = 2
SLOT_CHAIN_START = 3
SLOT_CHAIN_STOP = 4
SLOT_EXCHANGE_STOP = 5
SLOT_CAST_STOP = 6
SLOT_MLP_STOP = 7
SLOTS_PER_LAYER = 8
# The final norm plus the head, recorded once per prefill.
SLOT_TAIL_START = 8
SLOT_TAIL_STOP = 9
SLOTS_PER_PREFILL = SLOTS_PER_LAYER + 2

PHASES = (
    "attention",
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
TOP_LEVEL_PHASES = ("attention", "norm_residual", "mlp", "tail")
MLP_PARTS = ("mlp_chain", "mlp_exchange", "mlp_cast", "mlp_residual")


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

    def _record_all(self, slot: int) -> None:
        for device in self.devices:
            self._record(device, slot)

    def _slot(self, slot: int) -> int:
        return self.layer_index * SLOTS_PER_LAYER + slot

    # -- phase hooks -------------------------------------------------------
    def _enter_attention(self, *_args: Any, **_kwargs: Any) -> None:
        if not self.enabled:
            return
        self._record_all(self._slot(SLOT_LAYER_START))

    def _exit_attention(self, *_args: Any, **_kwargs: Any) -> None:
        self._record_all(self._slot(SLOT_ATTN_STOP))

    def _enter_norm(self, *_args: Any, **_kwargs: Any) -> None:
        # The norm phase starts where attention stopped; no new event needed.
        return

    def _exit_norm(self, *_args: Any, **_kwargs: Any) -> None:
        self._record_all(self._slot(SLOT_MLP_START))

    def _enter_mlp(self, *_args: Any, **_kwargs: Any) -> None:
        # mlp_start was recorded by the norm exit hook.
        return

    def _exit_mlp(self, *_args: Any, **_kwargs: Any) -> None:
        self._record_all(self._slot(SLOT_MLP_STOP))
        if self.enabled:
            self.layer_index += 1

    def _enter_chain(self, *_args: Any, **_kwargs: Any) -> None:
        self._record_all(self._slot(SLOT_CHAIN_START))

    def _exit_chain(self, *_args: Any, **_kwargs: Any) -> None:
        self._record_all(self._slot(SLOT_CHAIN_STOP))

    def _exit_exchange(self, *_args: Any, **_kwargs: Any) -> None:
        self._record_all(self._slot(SLOT_EXCHANGE_STOP))

    def _exit_device_reduce(self, *_args: Any, **_kwargs: Any) -> None:
        """The device route's whole reduction, and its absent cast.

        ``_forward_device_reduce`` enqueues the staging copy, the flag publish
        and the spin-add for both ranks, so this span is the exchange. The cast
        boundary is recorded immediately after it because the kernel already
        wrote bf16: leaving it unset would make ``collect`` read an event that
        was never recorded.
        """

        self._record_all(self._slot(SLOT_EXCHANGE_STOP))
        self._record_all(self._slot(SLOT_CAST_STOP))

    def _exit_cast(self, device: Any, *_args: Any, **_kwargs: Any) -> None:
        # ``cast_reduced`` runs once per rank inside the group's forward, so this
        # records the calling rank's own cast boundary.
        self._record(int(device), self._slot(SLOT_CAST_STOP))

    def _enter_tail(self, *_args: Any, **_kwargs: Any) -> None:
        if not self.enabled:
            return
        for device in self.devices:
            self._record(device, self.layers * SLOTS_PER_LAYER + 0)

    def _exit_tail(self, *_args: Any, **_kwargs: Any) -> None:
        for device in self.devices:
            self._record(device, self.layers * SLOTS_PER_LAYER + 1)

    # -- reporting ---------------------------------------------------------
    def collect(self) -> dict[str, Any]:
        from hipengine.core.device import scoped_current_device

        spans: dict[int, dict[str, float]] = {device: {name: 0.0 for name in PHASES} for device in self.devices}
        for device in self.devices:
            pool = self._pools[device]
            with scoped_current_device(self.runtime, device):
                for layer in range(self.layers):
                    base = layer * SLOTS_PER_LAYER
                    spans[device]["attention"] += self.runtime.event_elapsed_time_ms(
                        pool[base + SLOT_LAYER_START], pool[base + SLOT_ATTN_STOP]
                    )
                    spans[device]["norm_residual"] += self.runtime.event_elapsed_time_ms(
                        pool[base + SLOT_ATTN_STOP], pool[base + SLOT_MLP_START]
                    )
                    spans[device]["mlp"] += self.runtime.event_elapsed_time_ms(
                        pool[base + SLOT_MLP_START], pool[base + SLOT_MLP_STOP]
                    )
                    spans[device]["mlp_chain"] += self.runtime.event_elapsed_time_ms(
                        pool[base + SLOT_CHAIN_START], pool[base + SLOT_CHAIN_STOP]
                    )
                    spans[device]["mlp_exchange"] += self.runtime.event_elapsed_time_ms(
                        pool[base + SLOT_CHAIN_STOP], pool[base + SLOT_EXCHANGE_STOP]
                    )
                    spans[device]["mlp_cast"] += self.runtime.event_elapsed_time_ms(
                        pool[base + SLOT_EXCHANGE_STOP], pool[base + SLOT_CAST_STOP]
                    )
                    spans[device]["mlp_residual"] += self.runtime.event_elapsed_time_ms(
                        pool[base + SLOT_CAST_STOP], pool[base + SLOT_MLP_STOP]
                    )
                spans[device]["tail"] += self.runtime.event_elapsed_time_ms(
                    pool[self.layers * SLOTS_PER_LAYER + 0],
                    pool[self.layers * SLOTS_PER_LAYER + 1],
                )
        return spans


def _phase_sum(spans: dict[str, float]) -> float:
    return sum(spans[name] for name in TOP_LEVEL_PHASES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--fractions", default=None, help="e.g. 0.44,0.56")
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
    )
    prompt = [int(args.token_id)] * int(args.prompt_tokens)

    # Warm up through generate so the graph schedule, JIT variants and the
    # prompt-sized workspace all exist before the measured region.
    session.generate(prompt, max_new_tokens=2)
    runtime.device_synchronize()

    recorder = PhaseRecorder(runtime, session)
    recorder.install()
    # Read the counters from the group the recorder actually wrapped: the bulk
    # group is built lazily by the first bulk prefill, so capturing it before
    # that silently picks the decode group, whose exchange walls stay empty.
    group = recorder.group
    try:
        # One unrecorded prefill so every helper has run once with the wrappers
        # installed, then the recorded repeats.
        session.bulk_prefill(prompt, logits_rows=1)
        runtime.device_synchronize()

        recorder.reset()
        recorder.enabled = True
        walls: list[float] = []
        per_repeat: list[dict[int, dict[str, float]]] = []
        for _ in range(int(args.repeats)):
            recorder.reset()
            runtime.device_synchronize()
            started = time.perf_counter()
            session.bulk_prefill(prompt, logits_rows=1)
            runtime.device_synchronize()
            walls.append(time.perf_counter() - started)
            per_repeat.append(recorder.collect())
    finally:
        recorder.enabled = False
        recorder.close()

    wall_ms = min(walls) * 1000.0
    # The slowest rank's spans bound the step, so the report uses the max.
    worst = max(per_repeat, key=lambda spans: max(_phase_sum(v) for v in spans.values()))
    per_rank = {str(device): worst[device] for device in devices}
    exchange_walls = list(getattr(group, "exchange_walls_s", []) or [])
    exchange_host = sum(exchange_walls) * 1000.0 / max(1, len(walls))
    if not recorder.wrapped_bulk_group:
        raise SystemExit(
            "the recorder wrapped the decode group, not the bulk prefill group; "
            "the phase spans would describe the wrong route"
        )

    report: dict[str, Any] = {
        "schema": 1,
        "kind": "tp2_prefill_phase_attribution",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "model": str(args.model),
        "devices": list(devices),
        "fractions": None if fractions is None else list(fractions),
        "prompt_tokens": int(args.prompt_tokens),
        "repeats": int(args.repeats),
        "layer_count": int(len(session._config.layer_types)),
        "reduce_mode": getattr(session, "reduce_mode", None),
        "device_reduce": bool(recorder.device_reduce),
        "wall_ms": wall_ms,
        "wall_ms_samples": [round(value * 1000.0, 3) for value in walls],
        "prefill_tok_per_s": round(int(args.prompt_tokens) / (wall_ms / 1000.0), 3),
        "exchange_host_wall_ms_per_prefill": round(exchange_host, 3),
        "exchange_host_samples": len(exchange_walls),
        "mlp_parts": {
            device: {name: round(spans[name], 4) for name in MLP_PARTS}
            for device, spans in per_rank.items()
        },
        "phases": {
            device: {name: round(value, 4) for name, value in spans.items()}
            for device, spans in per_rank.items()
        },
        "totals": {
            device: {
                "phase_sum_ms": round(_phase_sum(spans), 4),
                "unaccounted_ms": round(wall_ms - _phase_sum(spans), 4),
                "share_of_wall": round(_phase_sum(spans) / wall_ms, 4),
            }
            for device, spans in per_rank.items()
        },
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")

    print(f"prefill {args.prompt_tokens} tokens: wall {wall_ms:.1f} ms = "
          f"{report['prefill_tok_per_s']:.1f} tok/s ({report['wall_ms_samples']})")
    print(f"layers {report['layer_count']}  exchange host wall "
          f"{exchange_host:.2f} ms/prefill over {len(exchange_walls)} samples")
    print()
    header = f"{'phase':<16}" + "".join(f"{'rank ' + str(d):>12}" for d in devices) + f"{'% wall':>9}"
    print(header)
    print("-" * len(header))
    for name in PHASES:
        values = [per_rank[str(device)][name] for device in devices]
        print(
            f"{name:<16}"
            + "".join(f"{value:>12.2f}" for value in values)
            + f"{100 * max(values) / wall_ms:>8.1f}%"
        )
    print("-" * len(header))
    for device in devices:
        total = report["totals"][str(device)]
        print(
            f"{'sum':<16}{total['phase_sum_ms']:>12.2f}"
            + " " * (12 * (len(devices) - 1))
            + f"{100 * total['share_of_wall']:>8.1f}%"
        )
        print(f"{'unaccounted':<16}{total['unaccounted_ms']:>12.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
