"""Model-level parity gate for the rows==1 MoE FFN graph (task #15).

HIPENGINE_GGUF_MOE_GRAPH routes the stateless per-layer MoE FFN through
capture/replay while the stateful attention stays eager.  This gate asserts the
graphed decode is bit-exact greedy-identical to the eager decode on the real
model and that the graph actually engaged (captures + replays, zero parity
rejects).  Skips unless ROCm and the 35B-A3B GGUF model are both present.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
PROMPT = [760, 4087, 369, 220, 16, 17, 18, 19]
N_STEPS = 6


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.exists()),
    reason="requires ROCm + the 35B-A3B GGUF model",
)


def _decode(enable_graph: bool):
    os.environ["HIPENGINE_GGUF_MOE_GRAPH"] = "1" if enable_graph else "0"
    os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    toks: list[int] = []
    stats = None
    with Qwen35GGUFResidentSession(
        MODEL, max_sequence_length=128, use_wmma_prefill=True, use_gemv_decode=True
    ) as s:
        first = s.prefill(PROMPT, use_bulk=True, return_logits=False)
        cur = int(first.token_id)
        for _ in range(N_STEPS):
            r = s.step(cur, return_logits=False)
            toks.append(int(r.token_id))
            cur = int(r.token_id)
        if s._moe_graph is not None:
            stats = dict(s._moe_graph.stats)
    return toks, stats


def test_moe_graph_decode_matches_eager_and_engages() -> None:
    eager_toks, eager_stats = _decode(False)
    assert eager_stats is None  # flag off -> no graph cache created

    graph_toks, graph_stats = _decode(True)
    assert graph_toks == eager_toks, f"graph diverged: {graph_toks} != {eager_toks}"
    assert graph_stats is not None
    # The graph must have actually engaged (not silently fallen back to eager).
    assert graph_stats["capture"] > 0
    assert graph_stats["replay"] > 0
    assert graph_stats["reject"] == 0
    assert graph_stats["eager"] == 0
