"""Sample real Q8-down operands without changing model outputs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import qwen4exp_journey_localize as localize


def dequant(raw, hidden):
    blocks = raw.reshape(-1, hidden // 32, 34)
    scales = np.ascontiguousarray(blocks[:, :, :2]).view(np.float16).reshape(
        blocks.shape[:2]).astype(np.float64)
    codes = blocks[:, :, 2:].view(np.int8).astype(np.float64)
    return (scales[:, :, None] * codes).reshape(-1, hidden)


def main():
    from hipengine.core.hip import MemcpyKind
    from hipengine.runtime import qwen4_exp_runner as runner

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    original_moe, original_resolve = runner.run_qwen4_exp_moe, runner.resolve
    layer = None
    records = []

    def download(ptr, shape, dtype, runtime):
        array = np.empty(shape, dtype=dtype)
        runtime.memcpy(array.ctypes.data, ptr, array.nbytes, MemcpyKind.DEVICE_TO_HOST)
        return array

    def moe(mixed_ptr, weights, **kwargs):
        nonlocal layer
        previous = layer
        layer = weights["expert_down"].spec.slot_path
        try:
            return original_moe(mixed_ptr, weights, **kwargs)
        finally:
            layer = previous

    def resolve(*a, **kw):
        fn = original_resolve(*a, **kw)
        if not (kw.get("quant") == "gguf_q8_0" and kw.get("variant") ==
                "selected_grouped_wmma_prefill_bf16_bf16_out"):
            return fn

        def replay(x, starts_ptr, wmma_starts, tiles, weights, output,
                   rows, experts, hidden, outputs, total, **kwargs):
            fn(x, starts_ptr, wmma_starts, tiles, weights, output,
               rows, experts, hidden, outputs, total, **kwargs)
            runtime = kwargs["runtime"]
            runtime.device_synchronize()
            starts = download(starts_ptr, (experts + 1,), np.int64, runtime)
            xb = download(x, (rows, hidden), np.uint16, runtime)
            xf = (xb.astype(np.uint32) << 16).view(np.float32)
            xh = xf.astype(np.float16).astype(np.float32)
            samples = []
            # Fixed geometry-based sampling, independent of token IDs or errors.
            for row in np.unique(np.linspace(0, rows - 1, 8, dtype=int)):
                expert = int(np.searchsorted(starts, row, side="right") - 1)
                cols = np.arange(0, outputs, max(1, outputs // 32))
                raw = np.stack([download(
                    weights + (expert * outputs + int(col)) * (hidden // 32 * 34),
                    (hidden // 32 * 34,), np.uint8, runtime) for col in cols])
                w = dequant(raw, hidden)
                wh = w.astype(np.float16).astype(np.float64)
                exact = w @ xf[row].astype(np.float64)
                rounded_w = wh @ xf[row].astype(np.float64)
                rounded_x = w @ xh[row].astype(np.float64)
                rounded_both = wh @ xh[row].astype(np.float64)
                got_bits = download(output + int(row) * outputs * 2,
                                    (outputs,), np.uint16, runtime)[cols]
                got = (got_bits.astype(np.uint32) << 16).view(np.float32)
                samples.append({
                    "row": int(row), "expert": expert,
                    "columns": cols.tolist(),
                    "fp64_raw": exact.tolist(),
                    "fp64_fp16_weights": rounded_w.tolist(),
                    "fp64_fp16_activations": rounded_x.tolist(),
                    "fp64_fp16_both": rounded_both.tolist(),
                    "kernel_bf16": got.tolist(),
                })
            records.append({
                "layer": layer, "compact_rows": rows, "hidden": hidden,
                "outputs": outputs, "activation_max_abs": float(np.max(np.abs(xf))),
                "activation_fp16_changed": int(np.count_nonzero(xf != xh)),
                "activation_elements": int(xf.size),
                "activation_fp16_nonfinite": int(np.count_nonzero(~np.isfinite(xh))),
                "activation_fp16_zeroed": int(np.count_nonzero((xf != 0) & (xh == 0))),
                "samples": samples,
            })
        return replay

    runner.run_qwen4_exp_moe, runner.resolve = moe, resolve
    try:
        localize.main()
    finally:
        runner.run_qwen4_exp_moe, runner.resolve = original_moe, original_resolve
        args.output.with_suffix(".operands.json").write_text(
            json.dumps(records, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
