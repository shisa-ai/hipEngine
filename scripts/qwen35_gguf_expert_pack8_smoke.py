#!/usr/bin/env python3
"""Compare qwen35moe GGUF expert pack8 sidecar kernels against raw selected GGUF kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.gguf_expert_pack8_gemv import build_gguf_expert_pack8_gemv
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.qwen35_gguf_expert_sidecar import build_or_load_qwen35moe_expert_sidecar
from hipengine.loading.qwen35_gguf_materialize import materialize_qwen35_gguf_weights
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.qwen35_gguf_runner import (
    _DeviceExpertPackedTensor,
    _launch_selected_expert_pack8_moe_linear,
    _launch_selected_expert_pack8_moe_pair,
    _launch_selected_raw_gguf_moe_linear,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--x-rows", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/hipengine-qwen35moe-expert-sidecar-smoke"))
    parser.add_argument("--overwrite-sidecar", action="store_true")
    parser.add_argument("--require-sidecar", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    compiler_version = args.compiler_version_file.read_text() if args.compiler_version_file else None
    expert_pack8_library = build_gguf_expert_pack8_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    runtime = get_hip_runtime()
    layer_id = int(args.layer)
    selected_slots = [
        f"layers.{layer_id}.ffn_gate_exps",
        f"layers.{layer_id}.ffn_up_exps",
        f"layers.{layer_id}.ffn_down_exps",
    ]
    weights = materialize_qwen35_gguf_weights(args.model, selected_slots=selected_slots, runtime=runtime)
    sidecar = build_or_load_qwen35moe_expert_sidecar(
        args.model,
        layer_id=layer_id,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite_sidecar,
        require_cached=args.require_sidecar,
    )
    try:
        cfg = weights.config
        rows = int(args.x_rows) * int(args.top_k)
        selected = (np.arange(rows, dtype=np.int64) * 37 + 11) % int(cfg.expert_count)
        results: list[dict[str, Any]] = []
        for slot, in_features, out_features, x_rows in (
            ("ffn_gate_exps", cfg.hidden_size, cfg.expert_feed_forward_length, int(args.x_rows)),
            ("ffn_up_exps", cfg.hidden_size, cfg.expert_feed_forward_length, int(args.x_rows)),
            ("ffn_down_exps", cfg.expert_feed_forward_length, cfg.hidden_size, rows),
        ):
            result = _run_slot(
                weights.layer(layer_id).weight(slot),
                sidecar.tensor(slot),
                selected,
                x_rows=x_rows,
                rows=rows,
                num_experts=cfg.expert_count,
                in_features=in_features,
                out_features=out_features,
                runtime=runtime,
                library=expert_pack8_library,
            )
            result["slot"] = slot
            result["quant_key"] = sidecar.tensor(slot).quant_key
            results.append(result)
            print(
                f"{slot}: quant={result['quant_key']} max_abs={result['max_abs']} "
                f"mean_abs={result['mean_abs']} bit_equal={result['bit_equal']}"
            )
        dual_result = _run_q4_gate_up_dual(
            weights.layer(layer_id).weight("ffn_gate_exps"),
            weights.layer(layer_id).weight("ffn_up_exps"),
            sidecar.tensor("ffn_gate_exps"),
            sidecar.tensor("ffn_up_exps"),
            selected,
            x_rows=int(args.x_rows),
            rows=rows,
            num_experts=cfg.expert_count,
            in_features=cfg.hidden_size,
            out_features=cfg.expert_feed_forward_length,
            runtime=runtime,
            library=expert_pack8_library,
        )
        results.append({"slot": "ffn_gate_up_exps_dual", "quant_key": "gguf_q4_k+gguf_q4_k", **dual_result})
        print(
            "ffn_gate_up_exps_dual: "
            f"gate_max_abs={dual_result['gate_max_abs']} up_max_abs={dual_result['up_max_abs']} "
            f"bit_equal={dual_result['bit_equal']} launched={dual_result['launched']}"
        )
    finally:
        weights.free(runtime=runtime)

    summary = {
        "model": str(args.model),
        "layer": layer_id,
        "rows": rows,
        "x_rows": int(args.x_rows),
        "top_k": int(args.top_k),
        "cache_dir": str(args.cache_dir),
        "all_within_gate": all(item.get("max_abs", max(item.get("gate_max_abs", 0.0), item.get("up_max_abs", 0.0))) <= 0.03125 for item in results),
        "results": results,
        "expected_kernel_substrings": [
            "gguf_expert_pack8_selected_prefill_kernel",
            "gguf_expert_pack8_q4_dual_selected_prefill_kernel",
        ],
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)
    return 0


def _run_slot(
    weight,
    packed_host,
    selected: np.ndarray,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    runtime,
    library,
) -> dict[str, Any]:
    x = ((np.arange(x_rows * in_features, dtype=np.float32).reshape(x_rows, in_features) % 17) - 8) / 64.0
    x_bits = float_array_to_bf16_bits(x)
    raw_out = np.empty((rows, out_features), dtype=np.uint16)
    packed_out = np.empty_like(raw_out)
    bufs = []
    try:
        x_dev = _dev(x_bits, runtime, bufs)
        selected_dev = _dev(selected, runtime, bufs)
        raw_out_dev = malloc(raw_out.nbytes, runtime=runtime)
        packed_out_dev = malloc(packed_out.nbytes, runtime=runtime)
        bufs.extend((raw_out_dev, packed_out_dev))
        _launch_selected_raw_gguf_moe_linear(
            weight,
            x_dev.ptr,
            selected_dev.ptr,
            raw_out_dev.ptr,
            x_rows=x_rows,
            rows=rows,
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            stream=0,
            runtime=runtime,
        )
        packed_dev = _DeviceExpertPackedTensor.from_host(packed_host, runtime=runtime)
        bufs.extend(packed_dev.buffers)
        _launch_selected_expert_pack8_moe_linear(
            packed_dev,
            x_dev.ptr,
            selected_dev.ptr,
            packed_out_dev.ptr,
            x_rows=x_rows,
            rows=rows,
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            stream=0,
            runtime=runtime,
            library=library,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(raw_out), raw_out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(packed_out), packed_out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)
    diff = bf16_to_float32(raw_out) - bf16_to_float32(packed_out)
    return {
        "bit_equal": bool(np.array_equal(raw_out, packed_out)),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "nonzero": int(np.count_nonzero(diff)),
        "raw_preview": [int(x) for x in raw_out.reshape(-1)[:8].tolist()],
        "packed_preview": [int(x) for x in packed_out.reshape(-1)[:8].tolist()],
    }


def _run_q4_gate_up_dual(
    gate_weight,
    up_weight,
    gate_packed_host,
    up_packed_host,
    selected: np.ndarray,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    runtime,
    library,
) -> dict[str, Any]:
    x = ((np.arange(x_rows * in_features, dtype=np.float32).reshape(x_rows, in_features) % 17) - 8) / 64.0
    x_bits = float_array_to_bf16_bits(x)
    raw_gate = np.empty((rows, out_features), dtype=np.uint16)
    raw_up = np.empty_like(raw_gate)
    dual_gate = np.empty_like(raw_gate)
    dual_up = np.empty_like(raw_gate)
    bufs = []
    try:
        x_dev = _dev(x_bits, runtime, bufs)
        selected_dev = _dev(selected, runtime, bufs)
        raw_gate_dev = malloc(raw_gate.nbytes, runtime=runtime)
        raw_up_dev = malloc(raw_up.nbytes, runtime=runtime)
        dual_gate_dev = malloc(dual_gate.nbytes, runtime=runtime)
        dual_up_dev = malloc(dual_up.nbytes, runtime=runtime)
        bufs.extend((raw_gate_dev, raw_up_dev, dual_gate_dev, dual_up_dev))
        _launch_selected_raw_gguf_moe_linear(
            gate_weight,
            x_dev.ptr,
            selected_dev.ptr,
            raw_gate_dev.ptr,
            x_rows=x_rows,
            rows=rows,
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            stream=0,
            runtime=runtime,
        )
        _launch_selected_raw_gguf_moe_linear(
            up_weight,
            x_dev.ptr,
            selected_dev.ptr,
            raw_up_dev.ptr,
            x_rows=x_rows,
            rows=rows,
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            stream=0,
            runtime=runtime,
        )
        gate_packed_dev = _DeviceExpertPackedTensor.from_host(gate_packed_host, runtime=runtime)
        up_packed_dev = _DeviceExpertPackedTensor.from_host(up_packed_host, runtime=runtime)
        bufs.extend((*gate_packed_dev.buffers, *up_packed_dev.buffers))
        launched = _launch_selected_expert_pack8_moe_pair(
            gate_packed_dev,
            up_packed_dev,
            x_dev.ptr,
            selected_dev.ptr,
            dual_gate_dev.ptr,
            dual_up_dev.ptr,
            x_rows=x_rows,
            rows=rows,
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            stream=0,
            runtime=runtime,
            library=library,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(raw_gate), raw_gate_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(raw_up), raw_up_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(dual_gate), dual_gate_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(dual_up), dual_up_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)
    gate_diff = bf16_to_float32(raw_gate) - bf16_to_float32(dual_gate)
    up_diff = bf16_to_float32(raw_up) - bf16_to_float32(dual_up)
    return {
        "launched": bool(launched),
        "bit_equal": bool(np.array_equal(raw_gate, dual_gate) and np.array_equal(raw_up, dual_up)),
        "gate_max_abs": float(np.max(np.abs(gate_diff))),
        "gate_mean_abs": float(np.mean(np.abs(gate_diff))),
        "gate_nonzero": int(np.count_nonzero(gate_diff)),
        "up_max_abs": float(np.max(np.abs(up_diff))),
        "up_mean_abs": float(np.mean(np.abs(up_diff))),
        "up_nonzero": int(np.count_nonzero(up_diff)),
    }


def _dev(array: np.ndarray, runtime, bufs: list):
    contiguous = np.ascontiguousarray(array)
    buf = malloc(contiguous.nbytes, runtime=runtime)
    bufs.append(buf)
    copy_host_to_device(buf, host_array_ptr(contiguous), runtime=runtime)
    return buf


if __name__ == "__main__":
    raise SystemExit(main())
