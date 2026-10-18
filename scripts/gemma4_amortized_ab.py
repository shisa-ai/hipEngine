"""A/B the amortized vs non-amortized grouped MoE owner on identical shapes.

gemma4_experts selects the amortized dual owner only while
in_features <= _GROUPED_DUAL_AMORTIZED_MAX_IN_FEATURES. Lowering that constant
below the Gemma expert width (2816) makes the same forward fall back to the
non-amortized row-batch owner, which re-reads each expert's input slice once per
output column. That is a 4x spread in input traffic (32.5 GB vs 8.12 GB for the
fused gate_up call) on the same kernel family and the same shapes.

If the amortized owner's time is ~4x lower, the input-traffic model from
iteration 99 is confirmed and a wider out_batch is worth building. If the two are
close, the kernel is limited by something else and the model is wrong.

Usage: PYTHONPATH=. .venv/bin/python scripts/gemma4_amortized_ab.py [max_in_features]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import hipengine
from hipengine.core.hip import get_hip_runtime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma4_prefill_census import Census  # noqa: E402

ARTIFACT = "/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"


def quant_of(weight) -> str:
    spec = getattr(weight, "spec", None)
    return str(getattr(spec, "quant_key", "dense_bf16"))


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 2048

    runtime = get_hip_runtime()
    llm = hipengine.LLM(model=ARTIFACT)
    generator = llm._get_text_generator()
    generator.context_length = 8192
    runner = generator._ensure_runner()

    from hipengine.kernels import registry
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts

    print(
        f"amortized limit: {gemma4_experts._GROUPED_DUAL_AMORTIZED_MAX_IN_FEATURES}"
        f" -> {limit}"
    )
    gemma4_experts._GROUPED_DUAL_AMORTIZED_MAX_IN_FEATURES = limit

    resolved: Counter[str] = Counter()
    original = registry.resolve

    def spy(*args, **kwargs):
        variant = kwargs.get("variant", args[3] if len(args) > 3 else "?")
        resolved[str(variant)] += 1
        return original(*args, **kwargs)

    registry.resolve = spy

    census = Census(runtime)

    def grouped_label(
        weight, x_ptr, expert_start_ptr, out_ptr, compact_rows, num_experts,
        in_features, out_features, **kw,
    ):
        return (
            f"moe_grouped:{quant_of(weight)} rows={compact_rows} "
            f"k={in_features} n={out_features}"
        )

    def grouped_dual_label(
        weight, x_ptr, expert_start_ptr, out_ptr, compact_rows, num_experts,
        in_features, out_features, fused_width, **kw,
    ):
        return (
            f"moe_grouped_dual:{quant_of(weight)} rows={compact_rows} "
            f"k={in_features} n={fused_width}"
        )

    census.wrap(gemma4_experts, "gemma4_project_experts_grouped", grouped_label)
    census.wrap(
        gemma4_experts, "gemma4_project_experts_grouped_dual", grouped_dual_label
    )

    try:
        runner.reset()
        runner.forward([9707] * 64)
        census.records.clear()
        resolved.clear()
        runner.reset()
        runner.forward([9707] * 1024)
        runtime.device_synchronize()
    finally:
        registry.resolve = original

    print("\nresolved MoE variants:")
    for key, n in resolved.most_common(8):
        print(f"  {n:5d}  {key}")
    print()
    summary = census.summarize()
    for key in sorted(summary, key=lambda k: -summary[k]["total_ms"]):
        row = summary[key]
        print(
            f"{key:58s} {row['calls']:5d} calls  {row['total_ms']:8.1f} ms  "
            f"{row['mean_us']:8.1f} us/call"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
