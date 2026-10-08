#!/usr/bin/env python3
"""Populate the runner's T16 gate/up tile cache for the A/B and gate stages.

Reads every Q4_K expert gate/up weight pair from the model's GGUF, repacks
each through the production T16 layout (repack_gguf_q4_k_tile16), uploads the
tiles, and inserts them into hipengine.runtime.qwen4_exp_runner's
_T16_GATE_UP_TILES cache keyed by the resident gate weight's raw device
pointer. Layers whose gate is not Q4_K are skipped. Call after generator
construction and before the measured request; the env-gated runner branch
(>HIPENGINE_QWEN4_EXP_T16_GATE_UP=1) then dispatches the T16 dual compensated
kernel, and a cache miss falls back to the production iu8 chain.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16


def populate_t16_gate_up_tiles(
    runner,
    model_root: str | Path,
) -> dict:
    """Insert T16 tiles for every Q4_K expert gate/up pair into the cache."""

    from hipengine.runtime import qwen4_exp_runner as rmod

    readers = [GGUFReader(p) for p in discover_gguf_files(model_root)]
    layer_weights = {}
    layer_weights.update(getattr(runner, "gdn_bindings", {}) or {})
    layer_weights.update(getattr(runner, "qsa_bindings", {}) or {})
    if not layer_weights:
        raise AssertionError("runner exposes no gdn/qsa layer bindings")

    inserted = {}
    for layer in sorted(layer_weights):
        lw = layer_weights[layer]
        moe = getattr(lw, "moe", None)
        if moe is None or "expert_gate" not in moe:
            continue
        gate_w = moe["expert_gate"]
        up_w = moe["expert_up"]
        if getattr(gate_w.spec, "quant_key", "") != "gguf_q4_k":
            continue
        # The binding's slot_path is the target name (layers.N.expert_gate);
        # the GGUF source names follow the blk.N.ffn_{gate,up}_exps.weight form
        # the natural-owner screen validated.
        slot = f"blk.{layer}.ffn_gate_exps.weight"
        up_slot = f"blk.{layer}.ffn_up_exps.weight"
        gate_reader = next(
            (r for r in readers if any(t.name == slot for t in r.info.tensors)), None
        )
        up_reader = next(
            (r for r in readers if any(t.name == up_slot for t in r.info.tensors)), None
        )
        if gate_reader is None or up_reader is None:
            print(f"layer {layer}: GGUF tensor not found ({slot}); skipped", file=sys.stderr)
            continue
        gate_raw = gate_reader.tensor_data(slot)
        up_raw = up_reader.tensor_data(up_slot)
        gate_tiles = np.ascontiguousarray(repack_gguf_q4_k_tile16(gate_raw).tiles)
        up_tiles = np.ascontiguousarray(repack_gguf_q4_k_tile16(up_raw).tiles)
        runtime = runner.runtime
        gate_dev = malloc(int(gate_tiles.nbytes))
        up_dev = malloc(int(up_tiles.nbytes))
        copy_host_to_device(gate_dev, host_array_ptr(gate_tiles), int(gate_tiles.nbytes))
        copy_host_to_device(up_dev, host_array_ptr(up_tiles), int(up_tiles.nbytes))
        gate_ptr = int(gate_w.allocation("raw").tensor.ptr)
        rmod._T16_GATE_UP_TILES[gate_ptr] = (gate_dev, up_dev)
        inserted[layer] = (gate_ptr, int(gate_tiles.nbytes), int(up_tiles.nbytes))
    return inserted


def populate_t16_gate_up_tiles_bounded(
    runner,
    model_root: str | Path,
    *,
    max_layers: int | None = None,
) -> dict:
    if max_layers is None:
        raw = os.environ.get("HIPENGINE_QWEN4_EXP_T16_GATE_UP_MAX_LAYERS")
        if raw:
            max_layers = int(raw)
    """Populate the cache, stopping gracefully at the memory bound.

    The model leaves a finite UMA slack after load; the full 47-layer tile set
    (~43 GiB) does not fit alongside the resident raw weights on a 120 GiB
    host. This variant inserts layers in order until an allocation fails or
    the optional ``max_layers`` is reached, frees any partially-uploaded pair,
    and returns what fit so the A/B can measure the partial window honestly.
    """

    from hipengine.runtime import qwen4_exp_runner as rmod

    readers = [GGUFReader(p) for p in discover_gguf_files(model_root)]
    layer_weights = {}
    layer_weights.update(getattr(runner, "gdn_bindings", {}) or {})
    layer_weights.update(getattr(runner, "qsa_bindings", {}) or {})
    if not layer_weights:
        raise AssertionError("runner exposes no gdn/qsa layer bindings")

    runtime = runner.runtime
    inserted = {}
    for layer in sorted(layer_weights):
        if max_layers is not None and len(inserted) >= max_layers:
            break
        lw = layer_weights[layer]
        moe = getattr(lw, "moe", None)
        if moe is None or "expert_gate" not in moe:
            continue
        gate_w = moe["expert_gate"]
        up_w = moe["expert_up"]
        if getattr(gate_w.spec, "quant_key", "") != "gguf_q4_k":
            continue
        slot = f"blk.{layer}.ffn_gate_exps.weight"
        up_slot = f"blk.{layer}.ffn_up_exps.weight"
        gate_reader = next(
            (r for r in readers if any(t.name == slot for t in r.info.tensors)), None
        )
        up_reader = next(
            (r for r in readers if any(t.name == up_slot for t in r.info.tensors)), None
        )
        if gate_reader is None or up_reader is None:
            continue
        gate_tiles = np.ascontiguousarray(
            repack_gguf_q4_k_tile16(gate_reader.tensor_data(slot)).tiles
        )
        up_tiles = np.ascontiguousarray(
            repack_gguf_q4_k_tile16(up_reader.tensor_data(up_slot)).tiles
        )
        try:
            gate_dev = malloc(int(gate_tiles.nbytes))
            up_dev = malloc(int(up_tiles.nbytes))
        except Exception as error:
            print(
                f"layer {layer}: memory bound reached ({error!r}); "
                f"stopping with {len(inserted)} layers cached",
                file=sys.stderr,
                flush=True,
            )
            break
        try:
            copy_host_to_device(gate_dev, host_array_ptr(gate_tiles), int(gate_tiles.nbytes))
            copy_host_to_device(up_dev, host_array_ptr(up_tiles), int(up_tiles.nbytes))
        except Exception as error:
            from hipengine.core.memory import free as mem_free
            mem_free(gate_dev, runtime=runtime)
            mem_free(up_dev, runtime=runtime)
            print(
                f"layer {layer}: upload failed ({error!r}); stopping with "
                f"{len(inserted)} layers cached",
                file=sys.stderr,
                flush=True,
            )
            break
        gate_ptr = int(gate_w.allocation("raw").tensor.ptr)
        rmod._T16_GATE_UP_TILES[gate_ptr] = (gate_dev, up_dev)
        inserted[layer] = (gate_ptr, int(gate_tiles.nbytes), int(up_tiles.nbytes))
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Populate the runner T16 cache (diagnostic entry point)."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        "This module is a helper; call populate_t16_gate_up_tiles(runner, model_root) "
        "after generator construction.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
