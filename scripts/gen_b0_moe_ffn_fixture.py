"""Generate the committed B0 golden fixture for the selected-expert GGUF MoE FFN.

Writes ``tests/fixtures/cpu_reference/moe_ffn_selected_gguf_q4_k.json`` with the
small deterministic inputs plus the golden ``expected_selected`` (megakernel
unit) and ``expected_block`` (full qwen35moe FFN block) outputs. Weights are
regenerated from ``tests/_gguf_synthetic_weights`` so they are not stored.

Run from the repo root:
    PYTHONPATH=. python3 scripts/gen_b0_moe_ffn_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hipengine.kernels.cpu_reference.ops import gguf_q4_k_moe_selected_ffn  # noqa: E402
from tests.test_cpu_reference_moe_ffn import (  # noqa: E402
    FFN_LEN,
    FIXTURE_PATH,
    HIDDEN,
    NUM_EXPERTS,
    SHARED_FFN_LEN,
    TOKENS,
    TOP_K,
    _oracle_block,
    build_b0_fixture,
)


def main() -> None:
    f = build_b0_fixture()
    expected_selected = gguf_q4_k_moe_selected_ffn(
        f["x"], f["selected_experts"], f["routing_weights"], f["gate_q"], f["up_q"], f["down_q"]
    )
    expected_block = _oracle_block(f)
    payload = {
        "schema": 1,
        "name": "moe_ffn_selected_gguf_q4_k",
        "description": (
            "B0 golden selected-expert GGUF MoE FFN (gate_up->silu*mul->down->"
            "routing-weighted combine) plus full block (+Q8_0 shared expert + "
            "sigmoid gate + residual). Weights regenerated from "
            "tests/_gguf_synthetic_weights; see build_b0_fixture()."
        ),
        "dims": {
            "tokens": TOKENS,
            "hidden": HIDDEN,
            "ffn_len": FFN_LEN,
            "shared_ffn_len": SHARED_FFN_LEN,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
        },
        "quant": {"gate": "Q4_K", "up": "Q4_K", "down": "Q4_K", "shared": "Q8_0"},
        "inputs": {
            "x": f["x"].tolist(),
            "residual": f["residual"].tolist(),
            "selected_experts": f["selected_experts"].tolist(),
            "routing_weights": f["routing_weights"].tolist(),
        },
        "expected_selected": np.asarray(expected_selected, dtype=np.float32).tolist(),
        "expected_block": np.asarray(expected_block, dtype=np.float32).tolist(),
    }
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {FIXTURE_PATH}")
    print(f"  expected_selected: shape={expected_selected.shape} "
          f"range=[{expected_selected.min():.5f}, {expected_selected.max():.5f}]")
    print(f"  expected_block:    shape={expected_block.shape} "
          f"range=[{expected_block.min():.5f}, {expected_block.max():.5f}]")


if __name__ == "__main__":
    main()
