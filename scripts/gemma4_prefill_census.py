#!/usr/bin/env python3
"""Attribute a Gemma 4 prefill to kernel families with HIP event pairs.

Wraps the launch sites the layer loop reaches (dense projections, MoE selected
projections, attention prefill, the per-layer forward) and reports each family's
summed device time against the measured prefill wall. No profiler: event pairs
cost two calls per measured launch and do not perturb the schedule the way
``rocprofv3 --kernel-trace`` does on a launch-bound workload.

The families nest (the layer forward contains the others), so the shares are
not additive; ``unattributed`` is the layer total minus the families measured
inside it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = Path(
    "/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)


class Census:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.records: list[tuple[str, Any, Any]] = []

    def wrap(self, owner: Any, name: str, label: Callable[..., str]) -> None:
        original = getattr(owner, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = self.runtime.event_create()
            end = self.runtime.event_create()
            self.runtime.event_record(start)
            try:
                return original(*args, **kwargs)
            finally:
                self.runtime.event_record(end)
                self.records.append((label(*args, **kwargs), start, end))

        setattr(owner, name, wrapper)

    def summarize(self) -> dict[str, dict[str, float]]:
        self.runtime.device_synchronize()
        totals: dict[str, list[float]] = defaultdict(list)
        for label, start, end in self.records:
            totals[label].append(self.runtime.event_elapsed_time_ms(start, end))
        return {
            label: {"calls": len(values), "total_ms": sum(values), "mean_us": 1000.0 * sum(values) / len(values)}
            for label, values in totals.items()
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--expect-gpu", default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    import hipengine
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    import ctypes

    library = ctypes.CDLL("libamdhip64.so")
    name = ctypes.create_string_buffer(256)
    library.hipDeviceGetName(name, len(name), 0)
    device = name.value.decode()
    print(f"[prefill_census] device0={device}", flush=True)
    if args.expect_gpu and args.expect_gpu not in device:
        print(f"ERROR: device0 is {device!r}, expected {args.expect_gpu!r}", file=sys.stderr)
        return 2

    llm = hipengine.LLM(model=str(args.artifact))
    generator = llm._get_text_generator()
    generator.context_length = int(args.context)
    runner = generator._ensure_runner()

    prompt = [9707] * args.tokens
    runner.reset()
    runner.forward(prompt[:64])
    runner.reset()
    runner.forward(prompt)
    runtime.device_synchronize()

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts, gemma4_layer
    import hipengine.runtime.gemma4 as gemma4_runtime

    census = Census(runtime)

    def quant_of(weight: Any) -> str:
        spec = getattr(weight, "spec", None)
        return str(getattr(spec, "quant_key", "dense_bf16"))

    census.wrap(
        gemma4_runtime,
        "gemma4_layer_forward_bf16",
        lambda *a, **k: "layer_total",
    )
    census.wrap(
        gemma4_layer,
        "gemma4_attention_prefill_bf16",
        lambda *a, **k: "attention_prefill",
    )

    def dense_label(x_ptr: int, weight: Any, out_ptr: int, rows: int, in_features: int, out_features: int, **kw: Any) -> str:
        return f"dense:{quant_of(weight)}"

    census.wrap(gemma4_layer, "gemma4_project", dense_label)

    def selected_label(weight: Any, x_ptr: int, selected_ptr: int, out_ptr: int, x_rows: int, rows: int, num_experts: int, in_features: int, out_features: int, **kw: Any) -> str:
        return f"moe_selected:{quant_of(weight)}"

    census.wrap(gemma4_experts, "gemma4_project_experts_selected", selected_label)

    def grouped_label(weight: Any, x_ptr: int, expert_start_ptr: int, out_ptr: int, compact_rows: int, num_experts: int, in_features: int, out_features: int, **kw: Any) -> str:
        return f"moe_grouped:{quant_of(weight)}"

    census.wrap(gemma4_experts, "gemma4_project_experts_grouped", grouped_label)

    def wmma_label(weight: Any, x_ptr: int, expert_start_ptr: int, expert_start_wmma_ptr: int, tile_expert_ptr: int, out_ptr: int, compact_rows: int, num_experts: int, in_features: int, out_features: int, wmma_total_rows: int, **kw: Any) -> str:
        return f"moe_wmma:{quant_of(weight)}"

    census.wrap(gemma4_experts, "gemma4_project_experts_wmma", wmma_label)

    def grouped_dual_label(weight: Any, x_ptr: int, expert_start_ptr: int, out_ptr: int, compact_rows: int, num_experts: int, in_features: int, out_features: int, fused_width: int, **kw: Any) -> str:
        return f"moe_grouped_dual:{quant_of(weight)}"

    census.wrap(
        gemma4_experts, "gemma4_project_experts_grouped_dual", grouped_dual_label
    )

    def wmma_dual_label(weight: Any, x_ptr: int, expert_start_ptr: int, expert_start_wmma_ptr: int, tile_expert_ptr: int, out_ptr: int, compact_rows: int, num_experts: int, in_features: int, out_features: int, wmma_total_rows: int, **kw: Any) -> str:
        return f"moe_wmma_dual:{quant_of(weight)}"

    census.wrap(gemma4_experts, "gemma4_project_experts_wmma_dual", wmma_dual_label)
    census.wrap(
        gemma4_experts,
        "_build_wmma_tile_plan",
        lambda *a, **k: "moe_wmma:tile_plan",
    )

    def offset_label(weight: Any, x_ptr: int, out_ptr: int, expert_start: Any, num_experts: int, in_features: int, out_features: int, **kw: Any) -> str:
        return f"moe_offset:{quant_of(weight)}"

    census.wrap(gemma4_experts, "gemma4_project_experts_by_offset", offset_label)

    for kernel in (
        "qwen35_moe_group_count",
        "qwen35_moe_group_prefix_active",
        "qwen35_moe_group_compact_active",
        "qwen35_moe_gather_packed_hidden_lowp",
        "gemma4_gelu_tanh_mul_bf16",
        "gemma4_moe_lane_to_row_i32",
        "gemma4_moe_weighted_accumulate_bf16",
        "gemma4_router_topk_bf16",
    ):
        if hasattr(gemma4_experts, kernel):
            census.wrap(gemma4_experts, kernel, lambda *a, _n=kernel, **k: f"moe_misc:{_n}")

    import time

    runner.reset()
    runtime.device_synchronize()
    start = time.perf_counter()
    runner.forward(prompt)
    runtime.device_synchronize()
    prefill_s = time.perf_counter() - start

    summary = census.summarize()
    layer_ms = summary.get("layer_total", {}).get("total_ms", 0.0)
    report = {
        "device": device,
        "tokens": args.tokens,
        "prefill_s": prefill_s,
        "prefill_tps": args.tokens / prefill_s,
        "families": summary,
        "layer_total_ms": layer_ms,
    }
    print(f"[prefill_census] prefill_s={prefill_s:.6f} prefill_tps={report['prefill_tps']:.4f}")
    print(f"{'family':44s} {'calls':>8s} {'total_ms':>12s} {'mean_us':>10s} {'of layer':>9s}")
    for label, row in sorted(summary.items(), key=lambda kv: -kv[1]["total_ms"]):
        share = 100.0 * row["total_ms"] / layer_ms if layer_ms else 0.0
        print(f"{label:44s} {row['calls']:8d} {row['total_ms']:12.3f} {row['mean_us']:10.2f} {share:8.1f}%")
    inside = sum(
        row["total_ms"]
        for label, row in summary.items()
        if label != "layer_total" and not label.startswith("moe_misc:")
    )
    print(f"{'unattributed inside layers (layer_total - families)':44s} {'':8s} {layer_ms - inside:12.3f} {100.0 * (layer_ms - inside) / layer_ms if layer_ms else 0:9.1f}%")
    print(f"{'outside layers (embed + head + block gaps)':44s} {'':8s} {1000.0 * prefill_s - layer_ms:12.3f}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
