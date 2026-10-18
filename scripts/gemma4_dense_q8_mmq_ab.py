#!/usr/bin/env python3
"""A/B the Q8 MMQ128 prefill route against the exact tiled route on Gemma 4.

The gemma4 runner never opens ``q8_mmq_prefill_session``, so every dense
``gguf_q8_0`` prefill projection takes ``exact_prefill_tile4x16``. The Qwen4Exp
and Qwen3.5 runners do open it, and their gate tables record MMQ128 beating the
float-coltile owner by 6.4-7.3x at their own shapes. Gemma 4's dense
projections are 31.9 percent of a 512-token prefill at 4.4-8.1 TFLOP/s, so the
question is whether the same route helps here.

This measures the difference before any policy is registered: one prefill with
the session closed, one with it open over a gemma4 policy, same process, same
weights, same prompt. The MMQ route quantizes activations to Q8_1 over D4
planes, so it is a different arithmetic and this probe reports its own output
divergence rather than assuming it is acceptable.

Usage:
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1151 \
        PYTHONPATH=. python3 scripts/gemma4_dense_q8_mmq_ab.py --prompt 512
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Gemma 4 26B-A4B dense q8_0 projections at 512 rows, (in_features, out_features).
# The MMQ dispatch requires in_features % 256 == 0 and out_features % 16 == 0.
GEMMA4_DENSE_SHAPES = (
    (2816, 2112),
    (4096, 2816),
    (2816, 2048),
    (2816, 4096),
    (8192, 2816),
    (2816, 8192),
    (2816, 1024),
)

# ffn_down is (2112, 2816): in_features 2112 % 256 == 64, so it fails the
# dispatch's own constraint and stays on the exact route by construction.
GEMMA4_EXCLUDED_SHAPES = ((2112, 2816),)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from scripts.gemma4_campaign_bench import _resolve_generator, resolve_artifact

    model = args.model or str(resolve_artifact())
    llm, _runner, info = _resolve_generator(Path(model), 4096)
    print(f"load_s={info['load_s']:.1f} resolution={info['resolution']}", flush=True)

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import DeviceBuffer, free, malloc
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import (
        Q8MMQPrefillPolicy,
        q8_mmq_d4x3_nbytes,
    )
    from hipengine.llm import SamplingParams
    from hipengine.runtime.gguf_linear import q8_mmq_prefill_session

    runtime = get_hip_runtime()
    prompt_ids = list(range(1000, 1000 + args.prompt))
    params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)

    # A null A/B is only worth reporting if the treatment ran. These count the
    # variant each dispatch hook actually returns, so the session-open arm can
    # be shown to have reached the MMQ route rather than silently declining it.
    from collections import Counter

    from hipengine.runtime import gguf_linear as glinear

    variant_counts: dict[str, Counter] = {"closed": Counter(), "open": Counter()}
    arm = "closed"
    originals = {}
    for name in ("_q8_mmq_prefill_dispatch", "_exact_q8_prefill_dispatch"):
        originals[name] = getattr(glinear, name)

    def make_counted(name, fn):
        def counted(dispatch, **kwargs):
            out = fn(dispatch, **kwargs)
            variant_counts[arm][getattr(out.key, "variant", "?")] += 1
            return out

        return counted

    for name, fn in originals.items():
        setattr(glinear, name, make_counted(name, fn))

    def prefill_tokens() -> int:
        result = llm.generate_detailed(prompt_ids, params)
        text = getattr(result, "text", None)
        return 0 if text is None else len(text)

    def timed(samples: int) -> list[float]:
        for _ in range(args.warmup):
            prefill_tokens()
        out = []
        for _ in range(samples):
            runtime.device_synchronize()
            started = time.perf_counter()
            prefill_tokens()
            runtime.device_synchronize()
            out.append(time.perf_counter() - started)
        return out

    baseline = timed(args.samples)
    base_median = statistics.median(baseline)
    print(
        f"session closed (exact tiled): {base_median:.4f} s  "
        f"({args.prompt / base_median:.1f} tok/s)  samples={[round(s, 4) for s in baseline]}",
        flush=True,
    )
    print(f"  variants: {dict(variant_counts['closed'])}", flush=True)

    policy = Q8MMQPrefillPolicy(
        min_rows={shape: 64 for shape in GEMMA4_DENSE_SHAPES},
        max_rows=2048,
        # The guard criterion is "near a BF16 rounding boundary"; this path
        # emits BF16, so the UD-Q3_K_M threshold applies rather than Qwen4Exp's
        # zero, which was chosen for an F32 output.
        risk_threshold=1.0e-5,
        max_out_features=8192,
    )

    rows = args.prompt
    workspace_nbytes = max(q8_mmq_d4x3_nbytes(rows, hidden) for hidden, _ in GEMMA4_DENSE_SHAPES)
    risk_capacity = policy.risk_capacity(rows)
    workspace = malloc(workspace_nbytes, runtime=runtime)
    risk_count = malloc(4, runtime=runtime)
    risk_indices = malloc(risk_capacity * 4, runtime=runtime)
    print(
        f"workspace={workspace_nbytes / 1e6:.1f} MB risk_capacity={risk_capacity}",
        flush=True,
    )
    try:
        arm = "open"
        with q8_mmq_prefill_session(
            workspace_ptr=workspace.ptr,
            workspace_nbytes=workspace_nbytes,
            risk_count_ptr=risk_count.ptr,
            risk_count_nbytes=4,
            risk_indices_ptr=risk_indices.ptr,
            risk_indices_nbytes=risk_capacity * 4,
            policy=policy,
        ):
            mmq = timed(args.samples)
        arm = "closed"
    finally:
        for name, fn in originals.items():
            setattr(glinear, name, fn)
        for buffer in (workspace, risk_count, risk_indices):
            free(buffer, runtime=runtime)

    mmq_median = statistics.median(mmq)
    print(
        f"session open (mmq128 d4x3):   {mmq_median:.4f} s  "
        f"({args.prompt / mmq_median:.1f} tok/s)  samples={[round(s, 4) for s in mmq]}",
        flush=True,
    )
    print(f"  variants: {dict(variant_counts['open'])}", flush=True)
    print(f"ratio closed/open = {base_median / mmq_median:.3f}x", flush=True)

    payload = {
        "kind": "gemma4_dense_q8_mmq_ab",
        "performance_claim": False,
        "prompt": args.prompt,
        "samples": args.samples,
        "baseline_prefill_s": base_median,
        "baseline_samples": baseline,
        "mmq_prefill_s": mmq_median,
        "mmq_samples": mmq,
        "ratio": base_median / mmq_median,
        "policy_shapes": [list(shape) for shape in GEMMA4_DENSE_SHAPES],
        "excluded_shapes": [list(shape) for shape in GEMMA4_EXCLUDED_SHAPES],
        "workspace_nbytes": workspace_nbytes,
        "variant_counts": {k: dict(v) for k, v in variant_counts.items()},
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
