"""Bounded GPU probe: batched rank-local MLP prefill vs per-row serial.

This validates the P1 batched MLP execution contract on the real GGUF shard
quant types (gate/up = Q4_K t16, down = Q6_K t16 qmicro planar for
Qwen3.8-27B Q4_K_M): for each active row count, the batched chain's per-row
result must match running that row alone, per rank and after the staged
reduction. It also checks that the inactive tail of a partially filled batch is
zeroed, never left as a stale or never-written row.

The serial per-row chain is the independent reference for batch composition;
the already-validated single-row CPU-oracle contract lives in
``scripts/tp2_mlp_slice_e2e.py``. This probe is correctness-only: it makes no
performance claim.

Run:

    python3 scripts/tp2_batched_prefill_probe.py \
        --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf --layer 0 --capacity 4 \
        --json benchmarks/results/2026-09-17-w7900-tp2-batched-prefill-probe.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _bf16_bytes(values: np.ndarray) -> np.ndarray:
    bits = values.astype("<f4").view("<u4")
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return (rounded.view("<u4") >> 16).astype("<u2").view(np.uint8)


def _bf16_values(payload: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(payload).view("<u2")
    return (u16.astype("<u4") << 16).view("<f4")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _upload(runtime, device: int, ptr: int, payload: np.ndarray) -> None:
    from hipengine.core.device import Device, scoped_current_device
    from hipengine.core.memory import copy_host_to_device

    class _Buf:
        def __init__(self, ptr: int, nbytes: int, device: int):
            self.ptr = int(ptr)
            self.nbytes = int(nbytes)
            self.device = Device("hip", int(device))

    with scoped_current_device(runtime, device):
        copy_host_to_device(
            _Buf(ptr, payload.nbytes, device),
            payload.ctypes.data,
            payload.nbytes,
            runtime=runtime,
        )


def _read_device(runtime, device: int, ptr: int, nbytes: int) -> np.ndarray:
    from hipengine.core.device import scoped_current_device
    from hipengine.core.runtime import MemcpyKind

    out = np.empty(nbytes, dtype=np.uint8)
    with scoped_current_device(runtime, device):
        runtime.memcpy(out.ctypes.data, int(ptr), nbytes, MemcpyKind.DEVICE_TO_HOST)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--json", default=None)
    parser.add_argument("--driver", choices=("python", "compiled"), default="python")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    if not _hip_available():
        print("no HIP runtime; skipping")
        return 0
    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.shard_group import MlpShardGroup
    from hipengine.distributed.shard_weights import (
        materialize_mlp_shards,
        upload_mlp_shard_weights,
    )

    runtime = get_hip_runtime()
    if int(runtime.device_count()) < 2:
        print(f"needs two HIP devices, found {runtime.device_count()}; skipping")
        return 0

    capacity = int(args.capacity)
    layer = int(args.layer)
    devices = (0, 1)

    t0 = time.time()
    shards = materialize_mlp_shards(args.model, world_size=2, layer_ids=(layer,))
    weights = upload_mlp_shard_weights(runtime, shards, devices=devices)
    layer_shards = shards[layer]
    hidden = int(layer_shards.ranks[0]["ffn_gate"].local_shape[1])
    per_rank_ffn = int(layer_shards.ranks[0]["ffn_gate"].local_shape[0])
    quant_keys = {
        role: layer_shards.ranks[0][role].quant_key
        for role in ("ffn_gate", "ffn_up", "ffn_down")
    }
    print(
        f"materialized layer {layer}: hidden={hidden} per_rank_ffn={per_rank_ffn} "
        f"quants={quant_keys} in {time.time() - t0:.1f}s"
    )

    streams: dict[int, int] = {}
    for device in devices:
        with scoped_current_device(runtime, device):
            streams[device] = int(runtime.stream_create(nonblocking=True))

    rng = np.random.default_rng(int(args.seed))
    x = (rng.standard_normal((capacity, hidden)) * 0.05).astype("<f4")
    x_bytes = np.ascontiguousarray(_bf16_bytes(x)).reshape(capacity, hidden, 2)

    row_buffers = {device: [] for device in devices}
    batched_buffers: dict[int, int] = {}
    for device in devices:
        with scoped_current_device(runtime, device):
            batched_buffers[device] = int(runtime.malloc(capacity * hidden * 2))
            row_buffers[device] = [
                int(runtime.malloc(hidden * 2)) for _ in range(capacity)
            ]
        for row in range(capacity):
            _upload(runtime, device, row_buffers[device][row], x_bytes[row].reshape(-1))
        _upload(runtime, device, batched_buffers[device], x_bytes.reshape(-1))

    group = MlpShardGroup(
        runtime,
        devices=devices,
        streams=streams,
        hidden=hidden,
        per_rank_ffn=per_rank_ffn,
        weights=weights,
        staging_dtype="bf16",
        driver=args.driver,
        rows=capacity,
    )

    report: dict = {
        "model": str(args.model),
        "layer": layer,
        "capacity": capacity,
        "hidden": hidden,
        "per_rank_ffn": per_rank_ffn,
        "quant_keys": quant_keys,
        "staging_dtype": "bf16",
        "driver": args.driver,
        "rows": {},
        "blocked": False,
    }

    def run(inputs, rows):
        partials = group.enqueue_chain(layer, inputs, rows=rows)
        reduced = group.reduce_partials(partials, rows=rows)
        for device in devices:
            group.cast_reduced(device, reduced[device])
        return partials

    try:
        for active in range(1, capacity + 1):
            batched_inputs = {d: batched_buffers[d] for d in devices}
            partial_ptrs = run(batched_inputs, active)
            batched_partials = {
                d: _bf16_values(
                    _read_device(runtime, d, partial_ptrs[d], active * hidden * 2)
                ).reshape(active, hidden)
                for d in devices
            }
            batched_out = {
                d: _bf16_values(
                    _read_device(runtime, d, group.output_ptr(d), active * hidden * 2)
                ).reshape(active, hidden)
                for d in devices
            }
            serial_partials = {d: [] for d in devices}
            serial_out = {d: [] for d in devices}
            for row in range(active):
                ptrs = run({d: row_buffers[d][row] for d in devices}, 1)
                for d in devices:
                    serial_partials[d].append(
                        _bf16_values(
                            _read_device(runtime, d, ptrs[d], hidden * 2)
                        ).reshape(hidden)
                    )
                    serial_out[d].append(
                        _bf16_values(
                            _read_device(runtime, d, group.output_ptr(d), hidden * 2)
                        ).reshape(hidden)
                    )
            entry = {"partials": {}, "outputs": {}, "tail_zero": None}
            for d in devices:
                sp = np.stack(serial_partials[d])
                so = np.stack(serial_out[d])
                p_scale = float(np.abs(sp).max()) or 1.0
                o_scale = float(np.abs(so).max()) or 1.0
                entry["partials"][str(d)] = {
                    "max_abs": float(np.abs(batched_partials[d] - sp).max()),
                    "scale": p_scale,
                    "rel": float(np.abs(batched_partials[d] - sp).max()) / p_scale,
                }
                entry["outputs"][str(d)] = {
                    "max_abs": float(np.abs(batched_out[d] - so).max()),
                    "scale": o_scale,
                    "rel": float(np.abs(batched_out[d] - so).max()) / o_scale,
                }
            if active < capacity:
                full_out = _bf16_values(
                    _read_device(
                        runtime, devices[0], group.output_ptr(devices[0]),
                        capacity * hidden * 2,
                    )
                ).reshape(capacity, hidden)
                entry["tail_zero"] = bool(
                    np.array_equal(full_out[active:], np.zeros_like(full_out[active:]))
                )
            report["rows"][str(active)] = entry
            print(
                f"active={active} partial rel="
                f"{max(entry['partials'][str(d)]['rel'] for d in devices):.3e} "
                f"output rel="
                f"{max(entry['outputs'][str(d)]['rel'] for d in devices):.3e} "
                f"tail_zero={entry['tail_zero']}"
            )
    finally:
        group.close()
        for device in devices:
            for ptr in row_buffers[device]:
                with scoped_current_device(runtime, device):
                    runtime.free(ptr)
            with scoped_current_device(runtime, device):
                runtime.free(batched_buffers[device])
                runtime.stream_destroy(streams[device])

    tolerance = 5e-3
    report["tolerance_rel"] = tolerance
    report["passed"] = all(
        entry["partials"][d]["rel"] <= tolerance
        and entry["outputs"][d]["rel"] <= tolerance
        and entry["tail_zero"] is not False
        for entry in report["rows"].values()
        for d in ("0", "1")
    )
    print("PASSED" if report["passed"] else "FAILED")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
