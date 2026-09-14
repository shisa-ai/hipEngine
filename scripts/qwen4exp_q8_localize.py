"""Diagnostic Q8 layer/shape interventions; no production dispatch changes."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import qwen4exp_journey_localize as localize


SHAPE_FLAG = "HIPENGINE_DIAGNOSTIC_MMQ_SHAPE_OFF"
LAYER_FLAG = "HIPENGINE_DIAGNOSTIC_Q8_DOWN_LAYER_OFF"
SHAPES = ((2560, 10240), (2560, 12288), (6144, 2560), (10240, 320),
          (2560, 2560), (2560, 640), (2560, 512))


def main():
    from hipengine.runtime import gguf_linear as linear
    from hipengine.runtime import qwen4_exp_runner as runner

    for k, n in SHAPES:
        localize.ARMS[f"mmq_off_{k}x{n}"] = {
            **localize.ARMS["q8_selected_down_strict"],
            SHAPE_FLAG: f"{k}x{n}",
        }
    for layer in (2, 4, 30, 46, 47):
        localize.ARMS[f"down_off_{layer}"] = {
            localize.PREFIX + "Q8_MMQ_PREFILL": "0",
            LAYER_FLAG: str(layer),
        }
    counts = Counter()
    original_dispatch = linear._q8_mmq_prefill_dispatch
    original_moe = runner.run_qwen4_exp_moe
    original_resolve = runner.resolve
    original_clear = localize.clear_arm_graphs
    active_layer = None

    def dispatch(parent, *, rows, in_features, out_features):
        selected = original_dispatch(parent, rows=rows, in_features=in_features,
                                     out_features=out_features)
        shape = f"{in_features}x{out_features}"
        intervention = os.environ.get(SHAPE_FLAG, "")
        if selected != parent:
            disabled = intervention == shape
            counts[f"mmq:{intervention}:{shape}:disabled={disabled}"] += 1
            if disabled:
                return parent
        return selected

    def moe(mixed_ptr, weights, **kwargs):
        nonlocal active_layer
        previous_layer = active_layer
        active_layer = weights["expert_down"].spec.slot_path.split(".")[1]
        flag = localize.PREFIX + "Q8_0_SELECTED_WMMA_DOWN"
        previous = os.environ.get(flag)
        disable = os.environ.get(LAYER_FLAG) == active_layer
        try:
            if disable:
                os.environ[flag] = "0"
            return original_moe(mixed_ptr, weights, **kwargs)
        finally:
            localize.set_flags({flag: previous})
            active_layer = previous_layer

    def resolve(*args, **kwargs):
        fn = original_resolve(*args, **kwargs)
        if (kwargs.get("quant") == "gguf_q8_0"
                and kwargs.get("variant") == "selected_grouped_wmma_prefill_bf16_bf16_out"):
            def counted(*a, **kw):
                counts[f"down:{os.environ.get(LAYER_FLAG, '')}:layer={active_layer}"] += 1
                return fn(*a, **kw)
            return counted
        return fn

    def clear(runner_instance):
        original_clear(runner_instance)
        linear.clear_gguf_linear_dispatch_cache()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    linear._q8_mmq_prefill_dispatch = dispatch
    runner.run_qwen4_exp_moe = moe
    runner.resolve = resolve
    localize.clear_arm_graphs = clear
    try:
        localize.main()
    finally:
        linear._q8_mmq_prefill_dispatch = original_dispatch
        runner.run_qwen4_exp_moe = original_moe
        runner.resolve = original_resolve
        localize.clear_arm_graphs = original_clear
        args.output.with_suffix(".counts.json").write_text(
            json.dumps(dict(sorted(counts.items())), indent=2) + "\n")


if __name__ == "__main__":
    main()
