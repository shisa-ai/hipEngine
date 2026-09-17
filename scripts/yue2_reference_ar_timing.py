#!/usr/bin/env python3
"""Fixed-token AR decode timing for the pinned upstream reference (oracle only).

The counterpart of ``scripts/yue2_ar_matched_timing.py``: same case, same recorded
token trajectory, same positive and negative prefixes, same step count, same two CFG
branches. It drives the reference's own ``GraphAR`` decode path - the one its
``generate_tokens`` uses when a CUDA graph is available, which is what produced the
recorded timings this is compared against - with sampling replaced by the recorded
token, so each step is one batched forward covering both branches plus the
full-vocabulary head, exactly as its product loop pays.

This script is oracle-only: it imports torch and the pinned upstream package, and
nothing here is reachable from ``hipengine.LLM.generate()``. Run it with the oracle
venv:

    PYTHONPATH=~/yue2-shootout/shared/upstream \\
    ~/venvs/vibevoice-tts-oracle/bin/python scripts/yue2_reference_ar_timing.py \\
        --case mandarin-off-s1234 --json /tmp/yue2_ar_ref.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.yue2_ar_matched_timing import _digest, load_case  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default="reference")
    parser.add_argument("--case", default="mandarin-off-s1234")
    parser.add_argument("--oracle", default=str(REPO / "artifacts" / "yue2" / "oracle" / "cases"))
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    import time

    import torch

    from yue2.cuda_graph import GraphAR

    from scripts.yue2_oracle import load_model, load_pipeline

    pipe = load_pipeline()
    tokenizer = pipe.tokenizer
    case_data = load_case(Path(args.oracle), args.case, tokenizer.encode)
    steps = int(args.steps) or len(case_data["trajectory"])
    prefix = case_data["prefix"]
    negative = case_data["negative"]
    trajectory = case_data["trajectory"]
    if not negative:
        raise SystemExit("this protocol needs a negative branch; the case has none")

    model = load_model()[1]

    def decode_once() -> tuple[float, float]:
        # GraphAR.step allows max_tokens - 1 calls, and prefill already predicts the
        # first token, so a budget of ``steps`` gives ``steps`` head evaluations.
        graph = GraphAR(model, [prefix, negative], steps, capture=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        graph.prefill()
        torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - started
        started = time.perf_counter()
        for step in range(steps - 1):
            branch_logits = graph.step(trajectory[step])
            # The product loop reduces this to one sampled token per step; the
            # reduction is what forces the device to finish, so keep it.
            conditional = branch_logits[:1]
            unconditional = branch_logits[1:]
            logits = unconditional + case_data["guidance"] * (conditional - unconditional)
            int(torch.argmax(logits, dim=-1).item())
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - started
        graph.close() if hasattr(graph, "close") else None
        return prefill_seconds, decode_seconds

    for _ in range(args.warmup):
        decode_once()
    prefill_seconds, decode_seconds = decode_once()
    samples = [decode_once() for _ in range(max(0, args.repeats - 1))]

    payload = {
        "protocol": "yue2-ar-fixed-token-v1",
        "side": "reference",
        "case": case_data["case"],
        "steps": steps,
        "prefix_tokens": len(prefix),
        "negative_tokens": len(negative),
        "prefix_digest": _digest(prefix),
        "negative_digest": _digest(negative),
        "trajectory_digest": _digest(trajectory[:steps]),
        "guidance": case_data["guidance"],
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "repeat_prefill_seconds": [s[0] for s in samples],
        "repeat_decode_seconds": [s[1] for s in samples],
        "execution": "cuda_graph",
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
    }
    print(f"case {payload['case']}: {steps} steps, prefix {payload['prefix_tokens']} tokens, "
          f"negative {payload['negative_tokens']} tokens")
    print(f"prefill {prefill_seconds:.3f} s, decode {decode_seconds:.2f} s, "
          f"{decode_seconds / max(1, steps - 1) * 1000.0:.2f} ms per step")
    if samples:
        print(f"repeat decode: {['%.2f' % s[1] for s in samples]}")
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
