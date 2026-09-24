#!/usr/bin/env python3
"""A/B the two placements for the head-sharded attention-output reduction.

Head sharding makes ``attn_output`` and ``ssm_out`` row-parallel, so each rank
holds a hidden-size partial that must be summed before the post-attention norm
(``_bulk_norm_residual_layer`` consumes it). That is a second exchange round per
layer, and there are two ways to place it:

* ``split`` - a second :class:`CompiledDeviceExchange` instance for the
  attention partial, alongside the MLP's existing one;
* ``phases`` - extra slots on the single existing instance, with a
  phase-distinct slot index (``2 * (layer % 2) + phase``).

A third placement, ``shared``, reuses the two existing slots with an extra
``bump()`` per layer. It is included as a hazard probe rather than a candidate:
the peer can publish its next payload into the same slot while this rank's
spin-add is still reading the previous one, because the protocol has no
acknowledgement. If ``shared`` produces wrong outputs here, that is evidence the
slot separation in ``phases`` is load-bearing; if it does not, that is *not*
proof it is safe, only that this workload did not expose the window.

Everything is measured. All three variants run the same
``layers x (attention, mlp)`` reduction sequence and the probe reports wall time
per variant, the pinned staging each variant needs, and a bit-exact check of
every reduced output against the host sum.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The script lives one level down, so ``sys.path[0]`` is ``scripts/`` and a bare
# ``import hipengine`` would resolve to whatever is installed. Pin this tree.
sys.path.insert(0, str(REPO_ROOT))

VARIANTS = ("split", "phases", "shared")

#: The two ranks of the TP2 host, in rank order: W7900 then RX 7900 XTX.
DEVICES = (0, 1)


def _require_hip() -> None:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"HIP runtime unavailable: {exc}") from exc


def _finite_bf16_bits(rng, n: int):
    """Realistic activation partials: finite bf16 bit patterns, wide exponents.

    Arbitrary 16-bit patterns (as the unit test uses on 4096 elements) include
    non-canonical NaNs, and the RNE narrow below overflows the exponent field on
    those, so the expectation would not be the kernel's. Activation partials are
    finite, and the check is exact on them.
    """

    import numpy as np

    values = rng.normal(0.0, 4.0, size=int(n)).astype("<f4")
    # A sparse tail across a wide exponent range, so the rounding path is
    # exercised rather than only small values.
    tail = rng.random(int(n)) < 0.05
    values[tail] *= np.float32(1e9)
    u = values.view("<u4")
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")


def _widen(bits):
    """bf16 bits -> f32 values, the exchange kernel's own widening."""

    import numpy as np

    return (np.asarray(bits).astype("<u4") << 16).view("<f4")


def _narrow_rne(values):
    """f32 values -> bf16 bits with round-to-nearest-even, as the kernel does."""

    import numpy as np

    u = np.asarray(values, dtype="<f4").view("<u4")
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")


def build_variant(runtime, streams, variant: str, *, hidden: int, rows: int):
    """Return ``(exchanges, slot_for)`` for one placement.

    ``slot_for(layer, phase)`` gives ``(exchange_index, slot)``; phase 0 is the
    attention partial and phase 1 the MLP partial.
    """

    from hipengine.distributed.device_exchange_compiled import CompiledDeviceExchange

    def make(num_layers: int):
        return CompiledDeviceExchange(
            runtime,
            devices=DEVICES,
            streams=streams,
            num_layers=num_layers,
            hidden=int(hidden),
            rows=int(rows),
        )

    if variant == "split":
        attention, mlp = make(2), make(2)
        return [attention, mlp], lambda layer, phase: (0 if phase == 0 else 1, layer % 2)
    if variant == "phases":
        exchange = make(4)
        return [exchange], lambda layer, phase: (0, 2 * (layer % 2) + phase)
    if variant == "shared":
        exchange = make(2)
        return [exchange], lambda layer, phase: (0, layer % 2)
    raise ValueError(f"unknown variant {variant!r}")


def staging_bytes(variant: str, *, hidden: int, rows: int) -> int:
    """Pinned host staging per rank, from the host's own arena arithmetic."""

    row_bytes = int(hidden) * int(rows) * 2  # bf16
    slots = {"split": 2 + 2, "phases": 4, "shared": 2}[variant]
    return slots * row_bytes


def run_sequence(
    exchanges,
    slot_for,
    *,
    layers: int,
    partials,
    outputs,
) -> None:
    """Enqueue ``layers x 2`` reductions, then wait once.

    Mirrors the bulk-prefill discipline: the timeout flags are cleared once for
    the whole group, every exchange bumps before its enqueue, and the single
    ``wait`` at the end is the only host synchronization.
    """

    for exchange in exchanges:
        exchange.reset_timeouts()
    for layer in range(int(layers)):
        for phase in (0, 1):
            index, slot = slot_for(layer, phase)
            exchange = exchanges[index]
            exchange.bump()
            for rank, device in enumerate(DEVICES):
                exchange.enqueue_rank(rank, partials[device], slot, outputs[phase][device])
    for exchange in exchanges:
        exchange.wait()


def _allocate(runtime, *, nbytes: int, device: int) -> int:
    from hipengine.core.device import scoped_current_device

    with scoped_current_device(runtime, device):
        return int(runtime.malloc(nbytes))


def _upload(runtime, host, *, device: int) -> int:
    from hipengine.core.device import scoped_current_device

    with scoped_current_device(runtime, device):
        buffer = int(runtime.malloc(int(host.nbytes)))
        runtime.memcpy(buffer, host.ctypes.data, int(host.nbytes), 3)
    return buffer


def _download(runtime, buffer: int, *, nbytes: int, device: int):
    import numpy as np

    from hipengine.core.device import scoped_current_device

    host = np.empty(nbytes, dtype=np.uint8)
    with scoped_current_device(runtime, device):
        runtime.memcpy(host.ctypes.data, buffer, nbytes, 2)
    return host


def check_variant(runtime, streams, variant: str, *, hidden: int, rows: int, layers: int):
    """Verify every reduced output bit-matches the host sum, on small shapes."""

    import numpy as np

    rng = np.random.default_rng(20260923)
    elements = int(hidden) * int(rows)
    nbytes = elements * 2
    host = {}
    partials = {}
    for device in DEVICES:
        host[device] = _finite_bf16_bits(rng, elements)
        partials[device] = _upload(runtime, host[device], device=device)
    # One output buffer per reduction, so every layer's attention result stays
    # readable: that is what exposes a staging slot being overwritten under a
    # reader.
    outputs = {
        phase: {device: [_allocate(runtime, nbytes=nbytes, device=device) for _ in range(layers)]
                for device in DEVICES}
        for phase in (0, 1)
    }
    expected = _narrow_rne(_widen(host[0]).astype("<f4") + _widen(host[1]).astype("<f4"))

    exchanges, slot_for = build_variant(runtime, streams, variant, hidden=hidden, rows=rows)
    try:
        for exchange in exchanges:
            exchange.reset_timeouts()
        for layer in range(int(layers)):
            for phase in (0, 1):
                index, slot = slot_for(layer, phase)
                exchange = exchanges[index]
                exchange.bump()
                for rank, device in enumerate(DEVICES):
                    exchange.enqueue_rank(
                        rank, partials[device], slot, outputs[phase][device][layer]
                    )
        for exchange in exchanges:
            exchange.wait()
        mismatches = []
        for phase in (0, 1):
            for layer in range(int(layers)):
                for device in DEVICES:
                    got = np.frombuffer(
                        _download(
                            runtime,
                            outputs[phase][device][layer],
                            nbytes=nbytes,
                            device=device,
                        ).tobytes(),
                        dtype="<u2",
                    )
                    if not np.array_equal(got, expected):
                        bad = np.where(got != expected)[0]
                        first = int(bad[0])
                        mismatches.append(
                            {
                                "phase": phase,
                                "layer": layer,
                                "device": device,
                                "bad_elements": int(bad.size),
                                "first_index": first,
                                "got_bits": int(got[first]),
                                "expected_bits": int(expected[first]),
                                "own_bits": int(host[device][first]),
                            }
                        )
        return {"layers": int(layers), "reductions": 2 * int(layers), "mismatches": mismatches}
    except Exception as exc:  # the hazard probe's expected failure mode
        return {
            "layers": int(layers),
            "reductions": 2 * int(layers),
            "mismatches": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        for exchange in exchanges:
            exchange.close()
        for phase in (0, 1):
            for device in DEVICES:
                for buffer in outputs[phase][device]:
                    from hipengine.core.device import scoped_current_device

                    with scoped_current_device(runtime, device):
                        runtime.free(buffer)
        for device in DEVICES:
            from hipengine.core.device import scoped_current_device

            with scoped_current_device(runtime, device):
                runtime.free(partials[device])


def time_variant(
    runtime,
    streams,
    variant: str,
    *,
    hidden: int,
    rows: int,
    layers: int,
    repeats: int,
    warmup: int,
):
    """Wall time for one full sequence, min and median over ``repeats``."""

    import numpy as np

    elements = int(hidden) * int(rows)
    nbytes = elements * 2
    rng = np.random.default_rng(7)
    partials = {}
    for device in DEVICES:
        host = _finite_bf16_bits(rng, elements)
        partials[device] = _upload(runtime, host, device=device)
    outputs = {
        phase: {device: _allocate(runtime, nbytes=nbytes, device=device) for device in DEVICES}
        for phase in (0, 1)
    }
    exchanges, slot_for = build_variant(runtime, streams, variant, hidden=hidden, rows=rows)
    try:
        for _ in range(int(warmup)):
            run_sequence(
                exchanges, slot_for, layers=layers, partials=partials, outputs=outputs
            )
        walls = []
        for _ in range(int(repeats)):
            started = time.perf_counter()
            run_sequence(
                exchanges, slot_for, layers=layers, partials=partials, outputs=outputs
            )
            walls.append((time.perf_counter() - started) * 1e3)
        return {
            "variant": variant,
            "exchanges": len(exchanges),
            "staging_bytes_per_rank": staging_bytes(variant, hidden=hidden, rows=rows),
            "walls_ms": walls,
            "min_ms": min(walls),
            "median_ms": statistics.median(walls),
            "per_reduction_ms": min(walls) / (2 * int(layers)),
        }
    finally:
        for exchange in exchanges:
            exchange.close()
        from hipengine.core.device import scoped_current_device

        for phase in (0, 1):
            for device in DEVICES:
                with scoped_current_device(runtime, device):
                    runtime.free(outputs[phase][device])
        for device in DEVICES:
            with scoped_current_device(runtime, device):
                runtime.free(partials[device])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=512, help="partial rows (prompt tokens)")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--check-layers", type=int, default=16)
    parser.add_argument("--check-rows", type=int, default=16)
    parser.add_argument("--check-only", action="store_true", help="run the correctness pass only")
    parser.add_argument("--variants", default="split,phases", help="variants to time")
    parser.add_argument("--check-variants", default=",".join(VARIANTS), help="variants to check")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    _require_hip()
    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    streams = {}
    for device in DEVICES:
        with scoped_current_device(runtime, device):
            streams[device] = int(runtime.stream_create())

    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    check_variants = [
        item.strip() for item in str(args.check_variants).split(",") if item.strip()
    ]
    report: dict[str, object] = {
        "rows": int(args.rows),
        "hidden": int(args.hidden),
        "layers": int(args.layers),
        "variants": {},
        "checks": {},
    }
    try:
        if not args.skip_check:
            for variant in check_variants:
                print(f"[check] {variant} ...", flush=True)
                result = check_variant(
                    runtime,
                    streams,
                    variant,
                    hidden=int(args.hidden),
                    rows=int(args.check_rows),
                    layers=int(args.check_layers),
                )
                report["checks"][variant] = result
                if "error" in result:
                    print(f"[check] {variant}: FAILED - {result['error']}", flush=True)
                    continue
                bad = len(result["mismatches"])
                detail = ""
                if bad:
                    first = result["mismatches"][0]
                    detail = (
                        f" (first: {first['bad_elements']} elements, "
                        f"got {first['got_bits']:#06x} expected {first['expected_bits']:#06x} "
                        f"from own {first['own_bits']:#06x})"
                    )
                print(f"[check] {variant}: {bad} mismatched reduction(s){detail}", flush=True)
        if args.check_only:
            print(json.dumps(report, indent=1, sort_keys=True))
            return 0
        for variant in variants:
            print(f"[time]  {variant} ...", flush=True)
            result = time_variant(
                runtime,
                streams,
                variant,
                hidden=int(args.hidden),
                rows=int(args.rows),
                layers=int(args.layers),
                repeats=int(args.repeats),
                warmup=int(args.warmup),
            )
            report["variants"][variant] = result
            print(
                f"[time]  {variant}: min {result['min_ms']:.3f} ms, "
                f"median {result['median_ms']:.3f} ms, "
                f"{result['per_reduction_ms']:.4f} ms/reduction, "
                f"staging {result['staging_bytes_per_rank'] / 1e6:.2f} MB/rank, "
                f"{result['exchanges']} exchange(s)",
                flush=True,
            )
    finally:
        for device in DEVICES:
            with scoped_current_device(runtime, device):
                runtime.stream_destroy(streams[device])

    print(json.dumps(report, indent=1, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
