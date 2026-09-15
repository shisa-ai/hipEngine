#!/usr/bin/env python3
"""Compare the hipEngine Qwen2 CPU reference against the HF oracle export.

Engine venv, no torch. Loads the pinned checkpoint's backbone BF16 tensors,
runs the FP32 numpy reference on the oracle's token sequence, and reports
max-abs drift for hidden states and logits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hipengine.kernels.cpu_reference import vibevoice_qwen2 as q2
from hipengine.loading.safetensors import load_weight_index, read_tensor_storage_bytes

DEFAULT_MODEL = "/models/vibevoice/VibeVoice-1.5B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--oracle", default="benchmarks/results/2026-09-15-vibevoice-qwen2-oracle.npz")
    parser.add_argument("--json", default="benchmarks/results/2026-09-15-vibevoice-qwen2-cpu-reference.json")
    args = parser.parse_args()

    data = np.load(args.oracle)
    token_ids = data["token_ids"].astype(np.int64)
    oracle_hidden = data["hidden"]
    oracle_logits = data["logits"]

    index = load_weight_index(args.model)
    weights = q2.load_backbone_weights(
        lambda name: read_tensor_storage_bytes(index.require([name])[0]),
        q2.Qwen2Geometry(),
    )
    hidden, _ = q2.forward_hidden_states(token_ids, weights, q2.Qwen2Geometry())
    logits = hidden @ weights["embed_tokens.weight"].T

    def diff(got: np.ndarray, want: np.ndarray) -> dict:
        d = np.abs(got.astype(np.float64) - want)
        return {
            "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "max_rel_to_peak": float(d.max() / max(np.abs(want).max(), 1e-30)),
        }

    report = {
        "tokens": int(token_ids.shape[0]),
        "geometry": {
            "hidden_size": q2.Qwen2Geometry().hidden_size,
            "num_hidden_layers": q2.Qwen2Geometry().num_hidden_layers,
        },
        "hidden_vs_oracle": diff(hidden, oracle_hidden),
        "logits_vs_oracle": diff(logits, oracle_logits),
        "hidden_peak": float(np.abs(oracle_hidden).max()),
        "logits_peak": float(np.abs(oracle_logits).max()),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
