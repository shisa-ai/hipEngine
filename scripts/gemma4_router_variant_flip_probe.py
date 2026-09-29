#!/usr/bin/env python3
"""Do the two router logits kernels ever select different experts on real data?

`74fb74ffe` sends prefill to `qwen35_router_logits_bf16_f32w_token_tile_16` and
leaves decode on `qwen35_router_logits_bf16_f32w`. Iteration 36 showed the two
select identical experts on 0 of 512 tokens and both match float64 -- but with
RANDOM weights. A trained router's logit distribution is not Gaussian: the
top-8 boundary can sit at a near-tie only real weights produce, so the
random-weight result is weak evidence against a flip in the actual run.

This probe closes that gap. It wraps the tiled kernel at the call site and, for
every prefill invocation during a real `LLM.generate()`, also runs the untiled
kernel into a scratch buffer from the same prescaled hidden. Both logits are
read back and their top-k selections compared token by token. The model's own
output is untouched: the scratch result is never written to the real buffer.

Decode is not the interesting case -- 1 token is below
`_TOKEN_TILE_16_MIN_TOKENS`, so both candidates take the untiled path by
construction. Prefill is the only place the two differ.

Usage:
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 PYTHONPATH=. \\
        .venv/bin/python scripts/gemma4_router_variant_flip_probe.py \\
        --prompt 6 --output 8
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def topk_sets(logits: np.ndarray, top_k: int) -> np.ndarray:
    """Which experts each row selects, order-insensitive."""
    order = np.argsort(-logits, axis=1, kind="stable")[:, :top_k]
    return np.sort(order, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/"
        "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
    )
    ap.add_argument("--prompt", type=int, default=6, help="times to repeat the base text")
    ap.add_argument("--output", type=int, default=8, help="tokens to generate")
    ap.add_argument("--top-k", type=int, default=8, help="Gemma 4 experts per token")
    ap.add_argument("--json-out", default="")
    ap.add_argument(
        "--prompt-tokens", type=int, default=0,
        help="replay the gate's frozen chain at this prompt length instead of "
             "generating from text; 5120 with --prefill 4096 matches V1",
    )
    ap.add_argument("--prefill", type=int, default=0,
                    help="ids pushed through the cache before scoring")
    ap.add_argument("--context", type=int, default=8192)
    args = ap.parse_args()

    from hipengine.core.memory import (
        copy_device_to_host,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_router as gr
    from hipengine.quant.gguf_q4_k import _bf16_u16_to_f32

    base_fn = gr.qwen35_router_logits_bf16_f32w
    tiled_fn = gr.qwen35_router_logits_bf16_f32w_token_tile_16

    stats: dict = {
        "tiled_calls": 0,
        "tokens_compared": 0,
        "tokens_flipped": 0,
        "calls_with_flips": [],
        "max_abs_logit_diff": 0.0,
        "mismatch_base_vs_float64": 0,
        "mismatch_tiled_vs_float64": 0,
        "reference_errors": [],
        "observe_errors": [],
        "flip_examples": [],
    }
    scratch = {"buf": None, "nbytes": 0}

    class _RawPtr:
        """Adapter so copy_device_to_host accepts the raw int the kernel gives us.

        That helper takes a DeviceBuffer and reads ``.nbytes`` off it; the call
        site only has an address, and passing it straight through raises
        ``AttributeError: 'int' object has no attribute 'nbytes'``.
        """

        def __init__(self, ptr: int, nbytes: int) -> None:
            self.ptr = ptr
            self.nbytes = nbytes

    def wrapped(hidden_ptr, weight_ptr, logits_ptr, tokens, hidden_size,
                num_rows, *, threads=256, stream=0, library=None, runtime=None):
        # 1. The model's real work, exactly as shipped. Everything after it is
        #    observation: if that raises, the model must still run, so the
        #    diagnostics are isolated and their errors recorded rather than
        #    allowed to kill the generation.
        tiled_fn(hidden_ptr, weight_ptr, logits_ptr, tokens, hidden_size, num_rows,
                 threads=threads, stream=stream, library=library, runtime=runtime)
        try:
            _observe(hidden_ptr, weight_ptr, logits_ptr, tokens, hidden_size,
                     num_rows, library=library, runtime=runtime)
        except Exception:
            stats["observe_errors"].append(traceback.format_exc(limit=8))
        return None

    def _observe(hidden_ptr, weight_ptr, logits_ptr, tokens, hidden_size,
                 num_rows, *, library, runtime):
        rows, width = int(tokens), int(num_rows)
        elems = rows * width
        bytes_ = elems * 4

        # 2. The path the baseline would have taken, into scratch only.
        if scratch["buf"] is None or scratch["nbytes"] < bytes_:
            if scratch["buf"] is not None:
                free(scratch["buf"])
            scratch["buf"] = malloc(bytes_)
            scratch["nbytes"] = bytes_
        alt_ptr = scratch["buf"].ptr
        base_fn(hidden_ptr, weight_ptr, alt_ptr, rows, int(hidden_size), width,
                library=library, runtime=runtime)

        # 3. Read both back.
        tiled_h = np.empty(elems, dtype=np.float32)
        base_h = np.empty(elems, dtype=np.float32)
        copy_device_to_host(host_array_ptr(tiled_h), _RawPtr(logits_ptr, bytes_), bytes_)
        copy_device_to_host(host_array_ptr(base_h), _RawPtr(alt_ptr, bytes_), bytes_)
        tiled_h = tiled_h.reshape(rows, width)
        base_h = base_h.reshape(rows, width)

        stats["tiled_calls"] += 1
        stats["tokens_compared"] += rows
        diff = float(np.abs(tiled_h - base_h).max())
        stats["max_abs_logit_diff"] = max(stats["max_abs_logit_diff"], diff)

        t_sel = topk_sets(tiled_h, args.top_k)
        b_sel = topk_sets(base_h, args.top_k)
        flipped = (t_sel != b_sel).any(axis=1)
        n_flip = int(flipped.sum())
        if n_flip:
            stats["tokens_flipped"] += n_flip
            stats["calls_with_flips"].append(stats["tiled_calls"])
            if len(stats["flip_examples"]) < 8:
                i = int(np.argmax(flipped))
                stats["flip_examples"].append({
                    "call": stats["tiled_calls"],
                    "row": i,
                    "base": b_sel[i].tolist(),
                    "tiled": t_sel[i].tolist(),
                })

        # 4. Float64 yardstick over the same inputs, to attribute any flip.
        try:
            # bf16 is 2 bytes, so rows*hidden uint16 elements == rows*hidden*2
            # bytes; sizing the array by the byte count doubles it and the
            # reshape below then fails.
            hidden_h = np.empty(rows * int(hidden_size), dtype=np.uint16)
            weight_h = np.empty(width * int(hidden_size) * 4, dtype=np.uint8)
            hb = rows * int(hidden_size) * 2
            wb = width * int(hidden_size) * 4
            copy_device_to_host(host_array_ptr(hidden_h), _RawPtr(hidden_ptr, hb), hb)
            copy_device_to_host(host_array_ptr(weight_h), _RawPtr(weight_ptr, wb), wb)
            h = _bf16_u16_to_f32(hidden_h.reshape(rows, int(hidden_size))).astype(np.float64)
            w = weight_h.view(np.float32).reshape(width, int(hidden_size)).astype(np.float64)
            ref = (h @ w.T).astype(np.float32)
            r_sel = topk_sets(ref, args.top_k)
            stats["mismatch_tiled_vs_float64"] += int((t_sel != r_sel).any(axis=1).sum())
            stats["mismatch_base_vs_float64"] += int((b_sel != r_sel).any(axis=1).sum())
        except Exception:  # pragma: no cover - diagnostic only
            stats["reference_errors"].append(traceback.format_exc(limit=6))

        return None

    gr.qwen35_router_logits_bf16_f32w_token_tile_16 = wrapped

    # --- run: either the gate's frozen chain, or a text generation ---------
    if args.prompt_tokens:
        from pathlib import Path as _P
        from scripts.gemma4_campaign_bench import (
            DEFAULT_ARTIFACT as _DEFAULT,
            exact_prompt_ids,
            _resolve_generator,
        )

        artifact = _P(args.model) if args.model else _P(_DEFAULT)
        llm, runner, loading = _resolve_generator(artifact, args.context)
        generator = llm._get_text_generator()
        prompt_ids = exact_prompt_ids(generator.tokenize, args.prompt_tokens)
        if not 0 <= args.prefill < len(prompt_ids) - 1:
            raise SystemExit(
                f"--prefill {args.prefill} must be < prompt {len(prompt_ids)} - 1"
            )
        # One forward of the whole prefill: this is the call that runs the
        # tiled router, once per layer per block.
        runner.forward(list(prompt_ids[: args.prefill]))
        text = f"replayed chain: {args.prefill} ids through prefill"
    else:
        import hipengine
        from hipengine.llm import SamplingParams
        from scripts.gemma4_campaign_bench import DEFAULT_ARTIFACT

        model = args.model or str(DEFAULT_ARTIFACT)
        llm = hipengine.LLM(model=model)
        base_text = (
            "The quarterly report covers revenue, operating cost, headcount, and "
            "retention across every region we serve, together with the assumptions "
            "that changed since the previous cycle and the follow-up questions the "
            "board asked us to resolve before the next review. "
        )
        prompt = base_text * max(1, args.prompt)
        out = llm.generate(prompt, SamplingParams(max_tokens=args.output, temperature=0.0))
        text = out[0] if out else ""

    if scratch["buf"] is not None:
        free(scratch["buf"])

    print("\n=== router variant flip probe (real activations, prefill) ===")
    print(f"  tiled-kernel calls observed  : {stats['tiled_calls']}")
    print(f"  tokens compared              : {stats['tokens_compared']}")
    print(f"  tokens selecting differently : {stats['tokens_flipped']}")
    print(f"  calls containing a flip      : {stats['calls_with_flips']}")
    print(f"  max |tiled - base| logits    : {stats['max_abs_logit_diff']:.3e}")
    print(f"  rows where base  != float64  : {stats['mismatch_base_vs_float64']}")
    print(f"  rows where tiled != float64  : {stats['mismatch_tiled_vs_float64']}")
    for ex in stats["flip_examples"]:
        print(f"    flip call={ex['call']} row={ex['row']} "
              f"base={ex['base']} tiled={ex['tiled']}")
    if stats["reference_errors"]:
        print(f"  reference errors: {len(stats['reference_errors'])}")
        print(stats["reference_errors"][0])
    if stats["observe_errors"]:
        print(f"  OBSERVATION ERRORS: {len(stats['observe_errors'])}")
        print(stats["observe_errors"][0])
    print(f"\n  generated {len(text)} chars ok")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(stats, indent=2))
        print(f"  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())