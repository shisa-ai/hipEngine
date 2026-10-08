#!/usr/bin/env python3
"""Natural-input owner screen: T16 dual gate/up vs the production iu8 chain.

The production iu8-risk gate/up kernel is hooked at its registry key during a
canonical prefill. On the first admitted call (a real layer's natural
activations, selected-expert maps and resident raw weights) the hook captures
the inputs, then runs both arms on the identical data across balanced
repetitions: the production iu8 chain (risk kernel + sparse exact repair,
into a private buffer reproducing the published parent) and the T16 dual
compensated kernel over the repacked real weights (into its own buffer). The
screen records per-arm timing, output drift statistics (T1 class: the T16
kernel is not bit-identical to the repair-published parent), and
input/weight/map immutability. Diagnostic screen; no default changes.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc, memory_stats
from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.qwen4_exp_profiles import (
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    QWEN4_EXP_QUANTS,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.gguf import GGUFReader, discover_gguf_files, load_gguf_index
from hipengine.models import resolve_model
from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, _git_metadata, _host_metadata, load_fixture
from scripts.qwen4exp_moe_owner_ab import _mkc

PARENT = "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"
T16 = "gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out"
BACKEND = "hip_gfx1151"
CHUNK_SIZE = 4096


def sha256_device(ptr: int, nbytes: int, runtime) -> str:
    import hashlib
    host = np.empty(int(nbytes), dtype=np.uint8)
    copy_device_to_host(host_array_ptr(host), DeviceBuffer(int(ptr), int(nbytes)), runtime=runtime)
    return hashlib.sha256(host.tobytes()).hexdigest()


def f32_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = (a.astype(np.float64) - b.astype(np.float64))
    denom = np.maximum(np.abs(b.astype(np.float64)), 1e-30)
    return {
        "max_abs": float(np.max(np.abs(d))),
        "rms_abs": float(np.sqrt(np.mean(d * d))),
        "max_rel": float(np.max(np.abs(d) / denom)),
        "bf16_floor_rms": float(np.sqrt(np.mean((b.astype(np.float32).view(np.uint16).astype(np.float64)
                                                 - np.round(b.astype(np.float64) * 255.0) / 255.0) ** 2))) if False else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", default="code-p4096")
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--layers", default="12,24,46", help="comma-separated layer indices to capture")
    args = parser.parse_args(argv=None) if False else parser.parse_args()
    if args.repetitions < 4:
        parser.error("repetitions must be at least 4")

    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    profile = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION,
    )
    fixture, fixture_sha = load_fixture(DEFAULT_FIXTURE)
    case = next(c for c in fixture["cases"] if c["id"] == args.case_id)
    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    generator = profile.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root,
        weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend=QWEN4_EXP_BACKEND,
        max_sequence_length=int(case["prompt_tokens"]) + 160,
        prefill_chunk_size=CHUNK_SIZE,
    ))
    runner = generator.runner
    runtime = runner.runtime

    # Real-weight T16 repack for the requested layers.
    readers = [GGUFReader(p) for p in discover_gguf_files(args.model_root)]
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

    def t16_for_layer(layer: int):
        tiles = {}
        for role in ("expert_gate", "expert_up"):
            name = f"blk.{layer}.ffn_{"gate_exps" if role == "expert_gate" else "up_exps"}.weight"
            reader = next((r for r in readers if any(t.name == name for t in r.info.tensors)), None)
            if reader is None:
                return None, None
            info = reader.tensor_info(name)
            if info.ggml_type_name != "Q4_K":
                return name, None
            raw = reader.tensor_data(name)
            repacked = repack_gguf_q4_k_tile16(raw).tiles
            dev = malloc(int(repacked.nbytes))
            copy_host_to_device(dev, host_array_ptr(np.ascontiguousarray(repacked)), int(repacked.nbytes))
            tiles[role] = dev
        return None, tiles

    parent_key = KernelKey(BACKEND, "moe_linear", "gguf_q4_k", PARENT)
    parent = resolve(backend=BACKEND, layer="moe_linear", quant="gguf_q4_k", variant=PARENT)
    repair_fn = resolve(backend=BACKEND, layer="moe_linear", quant="gguf_q4_k",
                        variant="selected_dual_sparse_exact_repair_bf16")
    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_t16_selected_prefill as t16mod
    t16_fn = t16mod.gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out

    start_ev, stop_ev = runtime.event_create(), runtime.event_create()
    wanted_layers = {int(v) for v in args.layers.split(",") if v.strip()}
    captures = []
    state = {"enabled": False, "current": None}

    def hooked(*call_args, **kwargs):
        if not state["enabled"]:
            return parent(*call_args, **kwargs)
        cur = state["current"]
        if cur is None:
            return parent(*call_args, **kwargs)
        # Match the requested layer by the gate weight pointer.
        gate_ptr = int(call_args[4])
        if gate_ptr != cur.get("gate_ptr", -1):
            return parent(*call_args, **kwargs)
        if cur.get("done"):
            return parent(*call_args, **kwargs)
        cur["done"] = True
        print("HOOK arg count:", len(call_args), "kwargs:", list(kwargs), file=sys.stderr, flush=True)
        print("HOOK args:", [hex(int(a)) if isinstance(a, int) and a > 4096 else a for a in call_args], file=sys.stderr, flush=True)
        # Args per the production call site:
        # x, group_expert_start, group_wmma_expert_start, group_tile_expert,
        # w_gate, w_up, out, risk_count, risk_indices, risk_capacity, mult,
        # compact, hidden, ffn_a, ffn_b, experts, static_tiles*16
        if len(call_args) < 17:
            raise AssertionError(f"iu8 gate/up hook received {len(call_args)} args")
        (x_ptr, ges_ptr, gwes_ptr, gte_ptr, wg_ptr, wu_ptr, out_ptr,
         rc_ptr, ri_ptr, rcap, mult, compact, hidden, ffn_a, ffn_b,
         experts, static_rows) = call_args[:17]
        compact = int(compact); hidden = int(hidden)
        ffn = int(ffn_a); experts = int(experts)
        x_nbytes = compact * hidden * 2
        ges_nbytes = (experts + 1) * 8
        gwes_nbytes = (experts + 1) * 8
        static_tiles = int(static_rows) // 16
        gte_nbytes = static_tiles * 8
        xh = sha256_device(x_ptr, x_nbytes, runtime)
        wgh = sha256_device(wg_ptr, int(cur["gate_weight_nbytes"]), runtime)
        private = malloc(compact * 2 * ffn * 2)
        parent_priv = malloc(compact * 2 * ffn * 2)
        p_samples, t_samples = [], []
        # copy maps
        def dup(ptr, nbytes):
            d = malloc(nbytes)
            runtime.memcpy_device_to_device(d.ptr, int(ptr), int(nbytes)) if hasattr(runtime, "memcpy_device_to_device") else None
            # fallback: host roundtrip
            if not hasattr(runtime, "memcpy_device_to_device"):
                host = np.empty(int(nbytes), dtype=np.uint8)
                copy_device_to_host(host_array_ptr(host), DeviceBuffer(int(ptr), int(nbytes)), runtime=runtime)
                copy_host_to_device(d, host_array_ptr(host), int(nbytes))
            return d
        x_dup = dup(x_ptr, x_nbytes)
        ges_dup = dup(ges_ptr, ges_nbytes)
        gwes_dup = dup(gwes_ptr, gwes_nbytes)
        gte_dup = dup(gte_ptr, gte_nbytes)
        # fresh risk buffers
        rc = malloc(4); ri = malloc(8 * int(rcap))
        t16_gate, t16_up = cur["t16_gate"], cur["t16_up"]
        try:
            parent(*call_args, **kwargs)  # publish for model
            t16_fn(x_dup.ptr, ges_dup.ptr, gwes_dup.ptr, gte_dup.ptr,
                   t16_gate.ptr, t16_up.ptr, private.ptr,
                   compact, hidden, ffn, ffn, experts, int(static_rows))
            repair_fn(x_ptr, ges_ptr, wg_ptr, wu_ptr, out_ptr,
                      rc.ptr, ri.ptr, int(rcap), compact, hidden, ffn, ffn, experts)
            runtime.device_synchronize()
            for rep in range(args.repetitions):
                order = ("parent", "t16") if (rep + len(captures)) % 2 == 0 else ("t16", "parent")
                for arm in order:
                    runtime.event_record(start_ev)
                    if arm == "parent":
                        parent(x_dup.ptr, ges_dup.ptr, gwes_dup.ptr, gte_dup.ptr,
                               wg_ptr, wu_ptr, parent_priv.ptr,
                               rc.ptr, ri.ptr, int(rcap), float(mult),
                               compact, hidden, ffn, ffn, experts, int(static_rows))
                        repair_fn(x_dup.ptr, ges_dup.ptr, wg_ptr, wu_ptr, parent_priv.ptr,
                                  rc.ptr, ri.ptr, int(rcap), compact, hidden, ffn, ffn, experts)
                    else:
                        t16_fn(x_dup.ptr, ges_dup.ptr, gwes_dup.ptr, gte_dup.ptr,
                               t16_gate.ptr, t16_up.ptr, private.ptr,
                               compact, hidden, ffn, ffn, experts, int(static_rows))
                    runtime.event_record(stop_ev)
                    runtime.event_synchronize(stop_ev)
                    elapsed = runtime.event_elapsed_time_ms(start_ev, stop_ev)
                    (p_samples if arm == "parent" else t_samples).append(elapsed)
            host_p = np.empty((compact, 2 * ffn), dtype=np.float32)
            host_t = np.empty((compact, 2 * ffn), dtype=np.float32)
            copy_device_to_host(host_array_ptr(host_p), parent_priv, runtime=runtime)
            copy_device_to_host(host_array_ptr(host_t), private, runtime=runtime)
            captures.append({
                "layer": cur["layer"],
                "compact_rows": compact, "hidden": hidden, "ffn": ffn,
                "experts": experts, "static_rows": int(static_rows),
                "x_sha256": xh, "gate_weight_sha256": wgh,
                "x_immutable": sha256_device(x_ptr, x_nbytes, runtime) == xh,
                "weight_immutable": sha256_device(wg_ptr, int(cur["gate_weight_nbytes"]), runtime) == wgh,
                "parent_ms": statistics.median(p_samples),
                "t16_ms": statistics.median(t_samples),
                "speedup": statistics.median(p_samples) / statistics.median(t_samples),
                "parent_samples": p_samples, "t16_samples": t_samples,
                "drift": f32_stats(host_t, host_p),
            })
            print(json.dumps({k: v for k, v in captures[-1].items() if k not in ("parent_samples", "t16_samples")}), flush=True)
        finally:
            for d in (private, parent_priv, x_dup, ges_dup, gwes_dup, gte_dup, rc, ri):
                free(d, runtime=runtime)
        return None  # already published

    report = {
        "schema": 1,
        "kind": "qwen4exp_t16_natural_owner_screen",
        "status": "running",
        "performance_claim": False,
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model_root": str(args.model_root),
        "fixture_sha256": fixture_sha,
        "manifest_sha256": profile.manifest_sha256,
        "case_id": case["id"],
        "parent_variant": PARENT, "t16_variant": T16,
        "repetitions": args.repetitions,
        "captures": captures,
    }
    try:
        # Prepare per-layer T16 tiles + resident weight pointers via the
        # runner's bound layer weights: layer_weights[layer].moe is the
        # MappingProxy of resident GGUFDeviceWeights the MoE call receives.
        prepared = {}
        layer_weights = {}
        layer_weights.update(getattr(runner, "gdn_bindings", {}) or {})
        layer_weights.update(getattr(runner, "qsa_bindings", {}) or {})
        if not layer_weights:
            raise AssertionError("runner exposes no gdn/qsa layer bindings")
        for layer in sorted(wanted_layers):
            lw = layer_weights.get(layer)
            if lw is None or "expert_gate" not in lw.moe:
                print(f"layer {layer}: no bound moe weights; skipped", file=sys.stderr)
                continue
            gate_w = lw.moe["expert_gate"]
            if getattr(gate_w.spec, "quant_key", "") != "gguf_q4_k":
                print(f"layer {layer}: gate is {getattr(gate_w.spec, 'quant_key', '?')}, skipped",
                      file=sys.stderr)
                continue
            _, tiles = t16_for_layer(layer)
            if tiles is None:
                print(f"layer {layer}: T16 repack unavailable; skipped", file=sys.stderr)
                continue
            raw_alloc = gate_w.allocation("raw")
            prepared[layer] = {
                "t16_gate": tiles["expert_gate"], "t16_up": tiles["expert_up"],
                "gate_ptr": raw_alloc.tensor.ptr,
                "gate_weight_nbytes": int(raw_alloc.buffer.nbytes),
                "gate_slot": str(getattr(gate_w.spec, "slot_path", "")),
                "layer": layer, "done": False,
                "tile_capacity": 8192, "risk_capacity": 1 << 20,
            }
        register(parent_key, hooked, replace=True)
        state["enabled"] = True
        for prep in prepared.values():
            state["current"] = prep
            runner.prefill([int(t) for t in case["prompt_token_ids"]])
            state["current"] = None
            if not prep["done"]:
                print(f"layer {prep['layer']}: hook did not fire (ptr mismatch?)", file=sys.stderr)
        report["status"] = "passed" if captures else "no-captures"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        register(parent_key, parent, replace=True)
        for prep in prepared.values():
            for role in ("t16_gate", "t16_up"):
                d = prep.get(role)
                if d is not None:
                    free(d, runtime=runtime)
        generator.close()
        report["memory_after_close"] = memory_stats()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
