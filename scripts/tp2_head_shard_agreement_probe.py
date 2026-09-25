#!/usr/bin/env python3
"""Locate the head-sharded bulk prefill's divergence, route by route.

Head sharding sums two bf16 partials where the unsharded route runs one
full-width projection, so the two routes are not expected to be bit-identical.
They are, however, expected to be the *same computation*: a mean KL of 0.36 and
59% top-1 agreement at the output is a wrong answer, not bf16 partial rounding.

This probe narrows where that happens by running the same prompt through both
routes in separate processes and comparing the tensors at the seams:

* ``route``   - run one route's bulk prefill and save its logits;
* ``layers``  - additionally capture the tensor the post-attention norm consumes
  (the reduced sum on the sharded route, the rank's own full attention output on
  the unsharded one) for the requested layers, on every rank;
* ``compare`` - read the saved artifacts back and report, per layer, how far the
  two routes are apart and whether the ranks agree with each other.

Each route gets its own process. A session whose constructor raises cannot be
closed (``close`` is an instance method) and the buffers it already allocated are
raw HIP allocations that no finalizer returns, so retrying inside one process
leaks a rank's worth of VRAM per failed attempt and turns a transient OOM into a
permanent one. Dying with the process returns everything.

A second hazard is specific to capturing: the attention kernels run on each
rank's own non-blocking stream, so the copy that reads their output has to be
ordered behind them. A plain ``memcpy`` on the default stream reads a
partly-written buffer and reports NaNs and spurious rank mismatches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The script lives one level down, so ``sys.path[0]`` is ``scripts/`` and a bare
# ``import hipengine`` would resolve to whatever is installed. Pin this tree.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
# 64 tokens, arbitrary but in-vocab: the probe compares two routes on one prompt,
# so the prompt only has to be long enough to exercise the prefill path.
PROMPT = tuple(range(100, 164))
ROWS = len(PROMPT)
DEFAULT_OUT = Path("/tmp/tp2_head_shard_agreement")


def _require_hip() -> None:
    """Fail with a clear message instead of an import error on a no-ROCm host."""

    import ctypes

    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:  # pragma: no cover - host dependent
        raise SystemExit(f"HIP runtime unavailable: {exc}") from exc


def _parse_layers(text: str) -> tuple[int, ...]:
    if not text:
        return ()
    return tuple(int(part) for part in text.split(",") if part.strip())


def _run_route(
    route: str, layers: tuple[int, ...], out: Path, lockstep: bool = False
) -> int:
    import numpy as np

    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

    captured: dict[tuple[int, int], "np.ndarray"] = {}
    partials: dict[tuple[int, int], "np.ndarray"] = {}
    if layers:
        from hipengine.core.device import scoped_current_device
        from hipengine.core.hip import get_hip_runtime

        runtime = get_hip_runtime()
        original = MlpTP2GenerationSession._bulk_norm_residual_layer

        def patched(self, layer_id, src, rows):
            if layer_id in layers and not any(k[0] == layer_id for k in captured):
                hidden = self.hidden_size
                for device in self.devices:
                    ptr = self._bulk_attention_output_ptr(device)
                    scratch = self._bulk_chunk_scratch.get(
                        device, self._bulk_scratch[device]
                    )
                    host = np.empty(rows * hidden, dtype="<u2")
                    # On the sharded route this is the rank's own heads'
                    # partial; on the unsharded one it is already the full
                    # output. Comparing the two is what separates a wrong split
                    # from a wrong reduction: if the sharded partials sum to the
                    # unsharded output, the split computes the right thing.
                    part = np.empty(rows * hidden, dtype="<u2")
                    with scoped_current_device(runtime, device):
                        # Order the copy behind this rank's attention kernels.
                        runtime.stream_synchronize(self._rank_stream(device))
                        # Raw device pointers, so the runtime memcpy rather than
                        # the DeviceBuffer-typed wrapper. Kind 2 = device to host.
                        runtime.memcpy(host.ctypes.data, ptr, rows * hidden * 2, 2)
                        runtime.memcpy(
                            part.ctypes.data, int(scratch.attn_out.ptr), rows * hidden * 2, 2
                        )
                        runtime.device_synchronize()
                    captured[layer_id, device] = host.copy()
                    partials[layer_id, device] = part.copy()
            return original(self, layer_id, src, rows)

        MlpTP2GenerationSession._bulk_norm_residual_layer = patched

    if lockstep:
        # A device-reduced group enqueues every layer without a host wait, and
        # the staging slot set only alternates on ``layer_id % 2``. So a rank
        # that runs one layer ahead is safe but a rank that reaches layer L+2
        # overwrites the slot the peer's spin-add on layer L may still be
        # reading. Waiting after each layer holds the ranks to one layer apart,
        # which is the property the slot scheme assumes. It is a diagnostic, not
        # a fix: the point of the device reduction is not to sync per layer.
        mlp_layer = MlpTP2GenerationSession._bulk_sharded_mlp_layer

        def lockstep_mlp(self, layer_id, src, dst, rows):
            result = mlp_layer(self, layer_id, src, dst, rows)
            group = self._bulk_shard_group
            if group is not None:
                group.finish_device_group()
            return result

        MlpTP2GenerationSession._bulk_sharded_mlp_layer = lockstep_mlp

    session = MlpTP2GenerationSession(
        MODEL,
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=ROWS,
        bulk_prefill=True,
        bulk_prefill_rows=ROWS,
        attention_shard=(route == "sharded"),
    )
    try:
        logits = session.bulk_prefill(PROMPT)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"{route}.npy", logits)
        for (layer_id, device), host in captured.items():
            np.save(out / f"{route}_l{layer_id}_d{device}.npy", host)
        for (layer_id, device), part in partials.items():
            np.save(out / f"{route}_partial_l{layer_id}_d{device}.npy", part)
        info = {
            "route": route,
            "attention_shard": route == "sharded",
            "lockstep": lockstep,
            "captured_layers": sorted({layer for layer, _ in captured}),
            "shape": list(logits.shape),
            "finite": bool(np.isfinite(logits).all()),
            "argmax_head": logits.argmax(-1)[:8].tolist(),
            # Read after bulk_prefill: _ensure_graph_schedule creates the
            # session-level exchange lazily, so reading it earlier yields None.
            "bulk_reductions_per_layer": getattr(
                session._bulk_shard_group, "reductions_per_layer", None
            ),
            "session_exchange_num_layers": getattr(
                session._device_exchange, "num_layers", None
            ),
            "hidden_size": session.hidden_size,
        }
        (out / f"{route}.json").write_text(json.dumps(info, indent=1) + "\n")
        print(json.dumps(info, indent=1), flush=True)
        return 0
    finally:
        session.close()


def _kl_rows(lhs, rhs) -> tuple[float, float]:
    """Row-mean KL(lhs || rhs) over the flattened vocab axis, and top-1 agreement."""

    import numpy as np

    lhs = np.asarray(lhs, dtype=np.float64).reshape(-1, lhs.shape[-1])
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1, rhs.shape[-1])
    lhs = lhs - lhs.max(axis=-1, keepdims=True)
    rhs = rhs - rhs.max(axis=-1, keepdims=True)
    lhs = np.exp(lhs)
    lhs /= lhs.sum(axis=-1, keepdims=True)
    rhs = np.exp(rhs)
    rhs /= rhs.sum(axis=-1, keepdims=True)
    eps = 1e-12
    kl = float(np.mean(np.sum(lhs * (np.log(lhs + eps) - np.log(rhs + eps)), axis=-1)))
    top1 = float(np.mean(lhs.argmax(-1) == rhs.argmax(-1)))
    return kl, top1


def _load_bf16(path: Path):
    """Read a raw bf16 dump as floats.

    The capture writes the device buffer verbatim, so the file holds bf16 *bit
    patterns*. Casting those to float yields the integer value (0xBB80 becomes
    48000), which looks like a saturated tensor and compares two routes on their
    exponent bytes rather than their values.
    """

    import numpy as np

    bits = np.load(path)
    if bits.dtype != np.uint16:
        raise SystemExit(f"{path}: expected raw bf16 (uint16), got {bits.dtype}")
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def _widen_bf16(bits):
    import numpy as np

    return (np.asarray(bits).astype("<u4") << 16).view("<f4")


def _narrow_bf16_rne(values):
    """RNE f32 -> bf16, the boundary cast the exchange narrows with."""

    import numpy as np

    u = np.asarray(values).astype("<f4").view("<u4")
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")


def _reduction_check(out: Path, layer: int, report: dict) -> None:
    """Does the consumed value equal the host sum of the two real partials?

    Rank agreement is not this check. Both ranks consume the same reduced row,
    so two ranks agreeing with each other is also what a wrong slot, a doubled
    partial, or a stale row produces. The only thing that rules those out is
    comparing the consumed bytes against the sum of the two partials the ranks
    actually wrote, computed here on the host.
    """

    import numpy as np

    parts = [out / f"sharded_partial_l{layer}_d{device}.npy" for device in (0, 1)]
    reduced = [out / f"sharded_l{layer}_d{device}.npy" for device in (0, 1)]
    if not all(path.exists() for path in parts + reduced):
        return
    expected = _narrow_bf16_rne(_widen_bf16(np.load(parts[0])) + _widen_bf16(np.load(parts[1])))
    stats: dict[str, object] = {}
    for device, path in enumerate(reduced):
        got = np.load(path)
        stats[f"rank{device}_exact"] = bool(np.array_equal(got, expected))
        stats[f"rank{device}_mismatched_elems"] = int((got != expected).sum())
        if not stats[f"rank{device}_exact"]:
            delta = _load_bf16(path) - (
                _load_bf16(parts[0]) + _load_bf16(parts[1])
            )
            scale = np.sqrt(np.mean((_load_bf16(parts[0]) + _load_bf16(parts[1])) ** 2)) + 1e-12
            stats[f"rank{device}_rel_rms"] = float(
                np.sqrt(np.mean(delta**2)) / scale
            )
    report.setdefault("reduction", {})[str(layer)] = stats
    print(
        f"  L{layer} reduction: rank0 exact={stats['rank0_exact']} "
        f"({stats['rank0_mismatched_elems']} elems), "
        f"rank1 exact={stats['rank1_exact']} ({stats['rank1_mismatched_elems']} elems)"
    )


def _compare(out: Path, layers: tuple[int, ...]) -> int:
    import numpy as np

    report: dict[str, object] = {"layers": {}}
    # The layer capture and the logits come from separate commands, so a run that
    # only did one of them is still worth comparing.
    if (out / "baseline.npy").exists() and (out / "sharded.npy").exists():
        baseline = np.load(out / "baseline.npy")
        sharded = np.load(out / "sharded.npy")
        kl, top1 = _kl_rows(baseline, sharded)
        report["logits"] = {
            "mean_kl": kl,
            "top1_agreement": top1,
            "baseline_argmax_head": baseline.argmax(-1)[:8].tolist(),
            "sharded_argmax_head": sharded.argmax(-1)[:8].tolist(),
        }
        print(f"logits: mean KL {kl:.4g}, top-1 agreement {top1:.4f}")
    else:
        print(f"no logits in {out}; comparing captured layers only")

    for layer in layers:
        rows: dict[str, object] = {}
        arrays: dict[str, np.ndarray] = {}
        for route in ("baseline", "sharded"):
            for device in (0, 1):
                path = out / f"{route}_l{layer}_d{device}.npy"
                if path.exists():
                    arrays[f"{route}_d{device}"] = _load_bf16(path)

        # The split check: on the sharded route each rank holds half the heads'
        # contribution; on the unsharded route both ranks hold the whole thing.
        # If the two partials sum to the unsharded output, the split computes
        # the right value and any remaining error belongs to the reduction.
        base_part = out / f"baseline_partial_l{layer}_d0.npy"
        shard_parts = [
            out / f"sharded_partial_l{layer}_d{device}.npy" for device in (0, 1)
        ]
        if base_part.exists() and all(p.exists() for p in shard_parts):
            full = _load_bf16(base_part)
            total = sum(_load_bf16(p) for p in shard_parts)
            denom = np.sqrt(np.mean(full**2)) + 1e-12
            report.setdefault("split", {})[str(layer)] = {
                "partials_sum_vs_full_rel_rms": float(
                    np.sqrt(np.mean((total - full) ** 2)) / denom
                ),
                "partial_r0_vs_r1_rel_rms": float(
                    np.sqrt(
                        np.mean((_load_bf16(shard_parts[0]) - _load_bf16(shard_parts[1])) ** 2)
                    )
                    / denom
                ),
            }
            stats = report["split"][str(layer)]
            print(
                f"  L{layer} split: partials_sum_vs_full rel_rms="
                f"{stats['partials_sum_vs_full_rel_rms']:.4g}  "
                f"r0_vs_r1={stats['partial_r0_vs_r1_rel_rms']:.4g}"
            )
        if not arrays:
            continue
        _reduction_check(out, layer, report)
        # Both ranks compute the same tensor on the unsharded route and hold
        # complementary halves on the sharded one, so rank agreement is a
        # separate signal from route agreement: a rank that disagrees with its
        # peer is a state or synchronization bug, not a rounding difference.
        base = arrays.get("baseline_d0")
        if base is not None:
            for name, arr in arrays.items():
                if name == "baseline_d0":
                    continue
                rows[f"{name}_vs_baseline_d0"] = {
                    "max_abs": float(np.max(np.abs(arr - base))),
                    "rel_rms": float(
                        np.sqrt(np.mean((arr - base) ** 2))
                        / (np.sqrt(np.mean(base**2)) + 1e-12)
                    ),
                    "exact": bool(np.array_equal(arr, base)),
                }
        report["layers"][str(layer)] = rows
        for name, stats in rows.items():
            print(
                f"  L{layer} {name:34s} rel_rms={stats['rel_rms']:.4g} "
                f"max_abs={stats['max_abs']:.4g} exact={stats['exact']}"
            )

    (out / "compare.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("route", "layers", "compare"))
    parser.add_argument("route", nargs="?", choices=("baseline", "sharded"))
    parser.add_argument(
        "--layers",
        default="0,1,2,3,4",
        help="comma-separated layer ids to capture (layers command only)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--lockstep",
        action="store_true",
        help="wait for both ranks after every layer (diagnostic)",
    )
    args = parser.parse_args(argv)

    if args.command == "compare":
        return _compare(args.out, _parse_layers(args.layers))
    if args.route is None:
        parser.error(f"{args.command} needs a route")
    _require_hip()
    layers = _parse_layers(args.layers) if args.command == "layers" else ()
    return _run_route(args.route, layers, args.out, lockstep=args.lockstep)


if __name__ == "__main__":
    raise SystemExit(main())
