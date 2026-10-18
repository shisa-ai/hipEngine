"""Report per-family weight bytes per prefill and the memory floor for each.

The census gives time per family; this gives bytes per family, so each family can
be compared against the roofline instead of guessed at. Wraps the projection
owners during a forward and records the weight bytes each call reads.

Usage: PYTHONPATH=. .venv/bin/python scripts/gemma4_weight_bytes_census.py
"""

from __future__ import annotations

from collections import defaultdict

import hipengine
from hipengine.core.hip import get_hip_runtime

ARTIFACT = "/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
BW_GBPS = 864.0  # W7900 / XTX peak device bandwidth


def weight_bytes(weight) -> int:
    try:
        return int(weight.allocation("raw").buffer.nbytes)
    except Exception:  # noqa: BLE001
        return 0


def main() -> int:
    runtime = get_hip_runtime()
    llm = hipengine.LLM(model=ARTIFACT)
    generator = llm._get_text_generator()
    generator.context_length = 8192
    runner = generator._ensure_runner()

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts, gemma4_layer

    stats: dict[str, list[int]] = defaultdict(list)
    originals: list[tuple[object, str, object]] = []

    def quant_of(weight) -> str:
        spec = getattr(weight, "spec", None)
        return str(getattr(spec, "quant_key", "dense_bf16"))

    def wrap(module, name: str, label, weight_arg: int = 0):
        orig = getattr(module, name)
        originals.append((module, name, orig))

        def spy(*args, **kwargs):
            weight = args[weight_arg] if len(args) > weight_arg else None
            if not isinstance(weight, int) and weight is not None:
                stats[label(weight)].append(weight_bytes(weight))
            return orig(*args, **kwargs)

        setattr(module, name, spy)

    # gemma4_project takes (x_ptr, weight, out_ptr, ...) -- weight is argument 1.
    # The MoE owners take weight first. Getting this wrong silently records
    # nothing, which is what happened on the first run.
    wrap(gemma4_layer, "gemma4_project", lambda w: f"dense:{quant_of(w)}", weight_arg=1)
    wrap(
        gemma4_experts,
        "gemma4_project_experts_grouped",
        lambda w: f"moe_grouped:{quant_of(w)}",
    )
    wrap(
        gemma4_experts,
        "gemma4_project_experts_grouped_dual",
        lambda w: f"moe_grouped_dual:{quant_of(w)}",
    )
    wrap(
        gemma4_experts,
        "gemma4_project_experts_selected",
        lambda w: f"moe_selected:{quant_of(w)}",
    )

    try:
        runner.reset()
        runner.forward([9707] * 64)
        for key in stats:
            stats[key].clear()
        runner.reset()
        runner.forward([9707] * 1024)
        runtime.device_synchronize()
    finally:
        for module, name, orig in originals:
            setattr(module, name, orig)

    print(f"{'family':34s} {'calls':>6s} {'bytes/call':>12s} {'total MB':>10s} {'ceiling ms':>11s}")
    grand = 0
    for key in sorted(stats, key=lambda k: -sum(stats[k])):
        vals = stats[key]
        if not vals:
            continue
        total = sum(vals)
        grand += total
        floor_ms = total / (BW_GBPS * 1e9) * 1e3
        print(
            f"{key:34s} {len(vals):6d} {vals[0] / 1e6:11.3f}M "
            f"{total / 1e6:9.1f} {floor_ms:10.1f}"
        )
    print(
        f"{'TOTAL':34s} {'':6s} {'':12s} {grand / 1e6:9.1f} "
        f"{grand / (BW_GBPS * 1e9) * 1e3:10.1f}"
    )
    # These are *tensor* sizes, not bytes read: a grouped owner touches only the
    # activated experts' rows. Treat the column as a ceiling on weight traffic --
    # dividing measured time by these numbers gave impossible bandwidths, so the
    # read fraction has to be measured (profiler or routing histogram), not
    # derived.
    print("NOTE: bytes are tensor sizes (ceiling), not bytes read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
