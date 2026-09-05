#!/usr/bin/env python3
"""Natural expert-ID distribution diagnostic for the W7900 packet D gates.

Runs eight distinct natural prompts (first eight of the mtpbench category
suite) through packed-AR c8 decode and captures every MoE layer's selected
expert IDs by monkey-patching ``qwen35_router_select`` in the GGUF runner
namespace (diagnostic only; no kernel or route changes). Reports the
duplicate-lane fraction, unique-expert ratio, and multiplicity histogram per
layer and in aggregate, plus the same statistics for the p512 fixture bench
distribution (four distinct prompts repeated to eight rows) so the artifact
can show both distributions side by side.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr  # noqa: E402
from hipengine.loading.gguf import scan_gguf  # noqa: E402
from hipengine.runtime import qwen35_gguf_runner as qgr  # noqa: E402
from hipengine.runtime.prefill import PrefillConfig  # noqa: E402
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession  # noqa: E402
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer  # noqa: E402

MODEL = "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
SUITE = REPO / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
STEPS = 24
WIDTH = 8
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gfx1100-d-natural-reuse.json"

CAPTURE: list[np.ndarray] = []
_ORIG = qgr.qwen35_router_select


def _patched(logits_ptr, selected_ptr, routing_ptr, tokens, logits_stride, num_experts, top_k, **kwargs):
    _ORIG(logits_ptr, selected_ptr, routing_ptr, tokens, logits_stride, num_experts, top_k, **kwargs)
    host = np.empty((tokens, top_k), dtype=np.int64)
    runtime = kwargs["runtime"]
    runtime.device_synchronize()
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(ptr=int(selected_ptr), nbytes=host.nbytes),
        host.nbytes,
    )
    CAPTURE.append(host)


def stats(lanes: np.ndarray) -> dict:
    flat = lanes.reshape(-1)
    counts = Counter(flat.tolist())
    unique = len(counts)
    total = flat.size
    mult = Counter(counts.values())
    return {
        "lanes": int(total),
        "unique_experts": int(unique),
        "unique_ratio": round(unique / total, 6),
        "duplicate_lane_fraction": round((total - unique) / total, 6),
        "multiplicity_histogram": {str(k): int(v) for k, v in sorted(mult.items())},
    }


def main() -> None:
    # Mirror the batch-route gate environment: decode-repack materialization is
    # what the qualified c8 route consumes, and the GDN semantic-gate scratch
    # reservation must precede session construction.
    import os

    os.environ["HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE"] = "1"
    os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    rows = [json.loads(l) for l in SUITE.read_text().splitlines() if l.strip()][:WIDTH]
    tok = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(MODEL))
    from scripts.gguf_mtp_bench import build_chat_prompt  # noqa: E402

    def _user_text(row: dict) -> str:
        for message in row["messages"]:
            if message.get("role") == "user":
                return str(message["content"])
        raise KeyError(f"prompt row {row.get('id')} has no user message")

    prompts = {str(r["id"]): build_chat_prompt(tok, _user_text(r)) for r in rows}
    max_len = max(len(v) for v in prompts.values()) + STEPS + 2

    qgr.qwen35_router_select = _patched
    sessions = []
    owner = Qwen35GGUFResidentSession(
        MODEL,
        backend="hip_gfx1100",
        require_cached_build=True,
        max_sequence_length=max_len,
        use_wmma_prefill=True,
        use_gemv_decode=True,
        prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
    )
    sessions.append(owner)
    for _ in range(WIDTH - 1):
        sessions.append(
            Qwen35GGUFResidentSession(
                MODEL,
                backend="hip_gfx1100",
                runtime=owner.runtime,
                shared_runner=owner.runner,
                require_cached_build=True,
                max_sequence_length=max_len,
                use_wmma_prefill=True,
                use_gemv_decode=True,
                prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
            )
        )
    try:
        for session, row in zip(sessions, rows, strict=True):
            session.reset()
            session.prefill(
                [int(t) for t in prompts[str(row["id"])]],
                use_bulk=True,
                bulk_attention_mode="bulk",
                return_logits=True,
                capture_hidden_seed_fp32=False,
            )
        layer_count = None
        step_captures: list[list[np.ndarray]] = []
        for step in range(STEPS):
            CAPTURE.clear()
            # The distribution statistic does not depend on which real token
            # continues each row, only that the rows are distinct natural
            # prompts; use the direct bench's terminal ids.
            token_ids = [9707 + (i % 4) for i in range(WIDTH)]
            owner.step_batch_native(token_ids, sessions=sessions)
            n = len(CAPTURE)
            if layer_count is None:
                layer_count = n
            elif n != layer_count:
                raise RuntimeError(f"router capture count drift: {n} != {layer_count}")
            step_captures.append([arr.copy() for arr in CAPTURE])
        # The pair-reuse route pairs duplicate IDs WITHIN one model step: the
        # relevant unit is one layer-step's 64 lanes (8 rows x top-8).
        per_layer = {
            str(l): stats(np.stack([step[l] for step in step_captures]))
            for l in range(layer_count)
        }
        within_step_unique: list[int] = []
        within_step_dup_lanes = 0
        within_step_total_lanes = 0
        for step in step_captures:
            for arr in step:
                s = stats(arr)
                within_step_unique.append(s["unique_experts"])
                within_step_dup_lanes += s["lanes"] - s["unique_experts"]
                within_step_total_lanes += s["lanes"]
        unique_hist = Counter(within_step_unique)
        # fixture distribution (modeled bound): the p512/d128 direct bench
        # repeats four distinct prompt trajectories across the eight rows (rows
        # 0-3 and 4-7 share contexts), so each of the four distinct rows' 8
        # experts appears at least twice: >= 16 paired lanes of 64. Cross-prompt
        # coincidental collisions would only increase pairing.
        out = {
            "kind": "gfx1100_packet_d_natural_expert_reuse",
            "model": MODEL,
            "width": WIDTH,
            "decode_steps": STEPS,
            "prompts": [str(r["id"]) for r in rows],
            "layers_captured": layer_count,
            "within_step_unit": "one layer-step = 64 selected lanes (8 rows x top-8)",
            "within_step_unique_experts_histogram": {str(k): int(v) for k, v in sorted(unique_hist.items())},
            "within_step_duplicate_lane_fraction": round(within_step_dup_lanes / within_step_total_lanes, 6),
            "within_step_unique_mean": round(sum(within_step_unique) / len(within_step_unique), 4),
            "per_layer_cross_step": per_layer,
            "prompts_note": "eight distinct natural prompts; rows are independent contexts",
            "fixture_distribution_note": "direct p512/d128 bench repeats four distinct prompt trajectories to eight rows (two rows per prompt); modeled pairwise-duplicate bound below",
            "fixture_modeled_step": stats(np.tile(np.arange(64).reshape(8, 8)[:4], (2, 1)).reshape(8, 8)),
        }
        Path(OUT).write_text(json.dumps(out, indent=1))
        print(json.dumps({
            "within_step_unique_experts_histogram": out["within_step_unique_experts_histogram"],
            "within_step_duplicate_lane_fraction": out["within_step_duplicate_lane_fraction"],
            "within_step_unique_mean": out["within_step_unique_mean"],
            "fixture_modeled_step": out["fixture_modeled_step"],
        }, indent=1))
    finally:
        qgr.qwen35_router_select = _ORIG
        for s in sessions:
            s.close()


if __name__ == "__main__":
    main()
