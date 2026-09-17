#!/usr/bin/env python3
"""Per-projection decode bandwidth audit on real weights, one rank at a time.

The TP2 campaign's layer attribution says a full-attention layer's attention half
runs at ~260 GB/s while its MLP shard half runs at ~500 GB/s, and that the
attention half is replicated (identical work on both ranks) so it is 45% of each
rank's per-token bytes. This script asks the next question: *which projection*
inside the attention half is slow, and how does it compare with the same device's
full-width MLP projections measured the same way?

Every launch goes through the production entry points (`launch_gguf_linear`,
`launch_gguf_linear_pair`) with the model's own weights and the session's own
scratch buffers, so the resolved kernel and shape are the ones decode uses. The
resolved dispatch key is recorded per role.

Two protocol details make the numbers mean something:

* **Rotation.** One layer's attention weights are 60-85 MB and gfx1100's Infinity
  Cache is 96 MB, so replaying a single layer measures L2, not the DRAM path the
  64-layer loop actually uses. Each role rotates over eight of its layers
  (hundreds of MB), so every timed launch streams from DRAM.
* **One queue, one sync.** A whole rotation is enqueued before any event is
  awaited, so no launch pays a cold-queue submission gap.

Usage:

    python3 scripts/gguf_decode_projection_bandwidth_audit.py MODEL [--json OUT]
        [--devices 0,1] [--rotations 12] [--layers 8]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import platform
import re
import sys
import time

import numpy as np

from hipengine.core.device import scoped_current_device
from hipengine.distributed.tp2_generate import MlpTP2GenerationSession
from hipengine.loading.gguf import scan_gguf
from hipengine.runtime.gguf_linear import (
    launch_gguf_linear,
    launch_gguf_linear_pair,
    resolve_gguf_linear_dispatch,
)

#: Roles to audit, in report order. ``want`` selects which layer type carries the
#: role; the shapes come from the GGUF index as (out_features, in_features). The
#: input and output buffers are the exact scratch buffers the runner's own decode
#: path uses for that projection, so every launch is a production-shaped one.
ROLES = (
    {
        "role": "attn_q",
        "suffixes": ("attn_q.weight",),
        "want": "full_attention",
        "inp": ("norm", 0),
        "out_a": ("full_q", 0),
        "out_b": None,
    },
    {
        "role": "attn_kv_pair",
        "suffixes": ("attn_k.weight", "attn_v.weight"),
        "want": "full_attention",
        "inp": ("norm", 0),
        "out_a": ("full_k", 0),
        "out_b": ("full_v", 0),
    },
    {
        "role": "attn_output",
        "suffixes": ("attn_output.weight",),
        "want": "full_attention",
        "inp": ("full_gated", 0),
        "out_a": ("attn_out", 0),
        "out_b": None,
    },
    {
        "role": "attn_qkv_gate_pair",
        "suffixes": ("attn_qkv.weight", "attn_gate.weight"),
        "want": "linear_attention",
        "inp": ("norm", 0),
        "out_a": ("linear_qkv", 0),
        "out_b": ("linear_z", 0),
    },
    {
        "role": "ssm_out",
        "suffixes": ("ssm_out.weight",),
        "want": "linear_attention",
        "inp": ("full_gated", 0),
        "out_a": ("attn_out", 0),
        "out_b": None,
    },
    {
        "role": "ffn_gate",
        "suffixes": ("ffn_gate.weight",),
        "want": "any",
        "inp": ("norm", 0),
        "out_a": ("ffn_gate_up", 0),
        "out_b": None,
    },
    {
        "role": "ffn_up",
        "suffixes": ("ffn_up.weight",),
        "want": "any",
        "inp": ("norm", 0),
        "out_a": ("ffn_gate_up", 0),
        "out_b": None,
    },
    {
        "role": "ffn_down",
        "suffixes": ("ffn_down.weight",),
        "want": "any",
        # The 17408-wide intermediate the gate/up pair just wrote.
        "inp": ("ffn_gate_up", 0),
        "out_a": ("attn_out", 0),
        "out_b": None,
    },
)


#: Non-projection decode kernels whose per-layer cost is the attention half's
#: other term: each is a tiny launch whose time is latency, not bytes.
KERNEL_ROLES = ("attn_norm", "post_attn_norm")


def _buffer_ptr(scratch: object, spec: tuple[str, int]) -> int:
    """Resolve a scratch buffer name (plus a bf16 element offset) to a pointer."""

    name, offset = spec
    return int(getattr(scratch, name).ptr) + int(offset) * 2


def _tensor_bytes(model: str) -> dict[str, dict[str, object]]:
    """Per-suffix tensor bytes and shapes from the GGUF index."""

    info = scan_gguf(model)
    out: dict[str, dict[str, object]] = collections.defaultdict(dict)
    for tensor in info.tensors:
        match = re.match(r"^blk\.(\d+)\.(.+)$", tensor.name)
        if match is None:
            continue
        layer_id = int(match.group(1))
        suffix = match.group(2)
        out[suffix][layer_id] = {
            "nbytes": int(tensor.nbytes),
            "shape": tuple(int(v) for v in tensor.shape),
        }
    return out


def _audit_device(
    model: str,
    device: int,
    *,
    rotations: int,
    layer_budget: int,
    suffixes: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Measure each role's projections on one rank and return its rows."""

    session = MlpTP2GenerationSession(
        model,
        devices=(device,),
        mode="tp1",
        schedule="eager",
        max_sequence_length=256,
    )
    runtime = session.runtime  # noqa: SLF001
    runner = session._runners[device]  # noqa: SLF001
    scratch = session._scratches[device]  # noqa: SLF001
    stream = session._rank_stream(device)  # noqa: SLF001
    layer_types = [str(t) for t in session._config.layer_types]  # noqa: SLF001
    rows: list[dict[str, object]] = []
    try:
        for spec in ROLES:
            role = str(spec["role"])
            suffix_list = tuple(spec["suffixes"])  # type: ignore[arg-type]
            want = str(spec["want"])
            available = [
                layer_id
                for layer_id in sorted(suffixes.get(suffix_list[0], {}))
                # The index carries the NextN/MTP block past the AR layer count;
                # only the autoregressive layers are part of this audit.
                if layer_id < len(layer_types)
                and (want == "any" or layer_types[layer_id] == want)
            ]
            if not available:
                continue
            picked = available[:: max(len(available) // layer_budget, 1)][
                :layer_budget
            ]
            weights = [
                [
                    runner.weights.layer(layer_id).weight(name.split(".")[0])
                    for name in suffix_list
                ]
                for layer_id in picked
            ]
            # One launch's bytes: the rotation total would overstate the rate.
            nbytes = sum(
                int(suffixes[name][picked[0]]["nbytes"])  # type: ignore[index]
                for name in suffix_list
            )
            shape = tuple(
                int(v)  # type: ignore[arg-type]
                for v in suffixes[suffix_list[0]][picked[0]]["shape"]  # type: ignore[index]
            )
            out_features, in_features = shape
            shape_b = (
                tuple(
                    int(v)  # type: ignore[arg-type]
                    for v in suffixes[suffix_list[1]][picked[0]]["shape"]  # type: ignore[index]
                )
                if len(suffix_list) == 2
                else None
            )
            dispatch = resolve_gguf_linear_dispatch(weights[0][0], rows=1)
            input_ptr = _buffer_ptr(scratch, spec["inp"])  # type: ignore[arg-type]
            out_a_ptr = _buffer_ptr(scratch, spec["out_a"])  # type: ignore[arg-type]
            out_b_ptr = (
                _buffer_ptr(scratch, spec["out_b"])  # type: ignore[arg-type]
                if spec["out_b"]
                else out_a_ptr
            )

            def enqueue(layer_weights: list[object]) -> None:
                if len(suffix_list) == 2:
                    launched = launch_gguf_linear_pair(
                        layer_weights[0],
                        layer_weights[1],
                        input_ptr,
                        out_a_ptr,
                        out_b_ptr,
                        rows=1,
                        in_features=in_features,
                        out_features=out_features,
                        out_features_b=int(shape_b[0]),
                        stream=stream,
                        runtime=runtime,
                    )
                    if not launched:
                        launch_gguf_linear(
                            layer_weights[0],
                            input_ptr,
                            out_a_ptr,
                            rows=1,
                            in_features=in_features,
                            out_features=out_features,
                            stream=stream,
                            runtime=runtime,
                        )
                        launch_gguf_linear(
                            layer_weights[1],
                            input_ptr,
                            out_b_ptr,
                            rows=1,
                            in_features=in_features,
                            out_features=int(shape_b[0]),
                            stream=stream,
                            runtime=runtime,
                        )
                else:
                    launch_gguf_linear(
                        layer_weights[0],
                        input_ptr,
                        out_a_ptr,
                        rows=1,
                        in_features=in_features,
                        out_features=out_features,
                        stream=stream,
                        runtime=runtime,
                    )

            samples: list[float] = []
            for _ in range(rotations + 2):
                pending: list[tuple[int, int]] = []
                for layer_weights in weights:
                    with scoped_current_device(runtime, device):
                        start = runtime.event_create()
                        stop = runtime.event_create()
                        pending.append((start, stop))
                        runtime.event_record(start, stream)
                        enqueue(layer_weights)
                        runtime.event_record(stop, stream)
                for start, stop in pending:
                    runtime.event_synchronize(stop)
                    samples.append(
                        float(runtime.event_elapsed_time_ms(start, stop))
                    )
                    with scoped_current_device(runtime, device):
                        runtime.event_destroy(start)
                        runtime.event_destroy(stop)
            # Drop the two warm-up rotations: the first fills the caches for
            # every picked layer, the second settles the rotation.
            measured = samples[2 * len(weights) :]
            ms = float(np.median(measured))
            rows.append(
                {
                    "device": device,
                    "device_name": runtime.device_get_name(device),
                    "role": role,
                    "suffixes": list(suffix_list),
                    "layers": picked,
                    "shape": [out_features, in_features],
                    "shape_b": list(shape_b) if shape_b else None,
                    "input_buffer": spec["inp"][0],
                    "output_buffer": spec["out_a"][0],
                    "quant_key": str(getattr(weights[0][0].spec, "quant_key", "?")),
                    "resolved_kernel": str(dispatch.key),
                    "median_ms": round(ms, 4),
                    "min_ms": round(float(min(measured)), 4),
                    "max_ms": round(float(max(measured)), 4),
                    "mb": round(nbytes / 1e6, 2),
                    "gb_per_s": round(nbytes / (ms / 1e3) / 1e9, 1) if ms else None,
                }
            )
        # Small kernels: the same rotation protocol, so their per-launch cost is
        # directly comparable with the projections above.
        add_norm = session._add_norm_kernel(runner)  # noqa: SLF001
        norm_layers = [
            i for i, t in enumerate(layer_types) if t == "full_attention"
        ]
        picked_norm = norm_layers[:: max(len(norm_layers) // layer_budget, 1)][
            :layer_budget
        ]
        for role in KERNEL_ROLES:
            samples = []
            for _ in range(rotations + 2):
                pending = []
                for layer_id in picked_norm:
                    layer = runner.weights.layer(layer_id)
                    with scoped_current_device(runtime, device):
                        start = runtime.event_create()
                        stop = runtime.event_create()
                        pending.append((start, stop))
                        runtime.event_record(start, stream)
                        if role == "attn_norm":
                            runner._run_attention_norm_rows(  # noqa: SLF001
                                hidden_ptr=session._hidden[device][0],  # noqa: SLF001
                                hidden_f32_ptr=None,
                                weight_ptr=layer.weight("attn_norm")
                                .allocation()
                                .tensor.ptr,
                                out_ptr=scratch.norm.ptr,
                                rows=1,
                                eps=runner.weights.config.rms_norm_eps,
                                stream=stream,
                                runtime=runtime,
                            )
                        else:
                            add_norm(
                                session._hidden[device][0],  # noqa: SLF001
                                scratch.attn_out.ptr,
                                layer.weight("post_attention_norm")
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
                        runtime.event_record(stop, stream)
                for start, stop in pending:
                    runtime.event_synchronize(stop)
                    samples.append(float(runtime.event_elapsed_time_ms(start, stop)))
                    with scoped_current_device(runtime, device):
                        runtime.event_destroy(start)
                        runtime.event_destroy(stop)
            measured = samples[2 * len(picked_norm) :]
            ms = float(np.median(measured))
            rows.append(
                {
                    "device": device,
                    "device_name": runtime.device_get_name(device),
                    "role": role,
                    "suffixes": [],
                    "layers": picked_norm,
                    "shape": [1, runner.hidden_size],
                    "shape_b": None,
                    "quant_key": "-",
                    "resolved_kernel": "-",
                    "median_ms": round(ms, 4),
                    "min_ms": round(float(min(measured)), 4),
                    "max_ms": round(float(max(measured)), 4),
                    "mb": 0.0,
                    "gb_per_s": None,
                }
            )
    finally:
        session.close()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--json", type=pathlib.Path, default=None)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--rotations", type=int, default=12)
    parser.add_argument("--layers", type=int, default=8)
    args = parser.parse_args(argv)

    devices = [int(part) for part in args.devices.split(",") if part.strip()]
    suffixes = _tensor_bytes(args.model)
    result: dict[str, object] = {
        "kind": "gguf-decode-projection-bandwidth-audit",
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "model": str(args.model),
        "rotations": args.rotations,
        "layers_per_role": args.layers,
        "rows": [],
        "note": (
            "Real weights through the production launch entry points, one rank "
            "at a time, one queue per rotation with a single sync, rotating over "
            "several layers per role so every timed launch streams from DRAM "
            "instead of the 96 MB Infinity Cache. Shapes are (out, in) from the "
            "GGUF index; sharded shapes are not audited here (the MLP shard "
            "chain is measured in scripts/tp2_layer_segment_attribution.py)."
        ),
    }
    print(f"host {platform.node()} | model {args.model}")
    for device in devices:
        rows = _audit_device(
            args.model,
            device,
            rotations=args.rotations,
            layer_budget=args.layers,
            suffixes=suffixes,
        )
        if not rows:
            continue
        print(f"\n=== device {device}: {rows[0]['device_name']} ===")
        print(
            f"  {'role':22s} {'shape':>16s} {'quant':>20s} {'MB':>8s} "
            f"{'ms':>8s} {'GB/s':>7s}  kernel"
        )
        for row in rows:
            shape = f"{row['shape'][0]}x{row['shape'][1]}"
            if row["shape_b"]:
                shape += f" + {row['shape_b'][0]}x{row['shape_b'][1]}"
            rate = (
                f"{row['gb_per_s']:7.1f}" if row["gb_per_s"] is not None else "      -"
            )
            print(
                f"  {row['role']:22s} {shape:>16s} {row['quant_key']:>20s} "
                f"{row['mb']:8.1f} {row['median_ms']:8.4f} "
                f"{rate}  {row['resolved_kernel']}"
            )
        result["rows"].extend(rows)  # type: ignore[attr-defined]
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
