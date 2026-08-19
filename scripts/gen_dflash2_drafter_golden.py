#!/usr/bin/env python3
"""Generate a DFlash2 drafter golden trace from the real reference model.

Test-time only (requires the ``torch`` extra and the z-lab/dflash reference at
``~/dflash``). Loads ``z-lab/Qwen3.8-27B-DFlash2`` via
``dflash.model.DFlash2DraftModel.from_pretrained`` in fp32 and runs a full
forward + greedy proposal over deterministic synthetic inputs. The saved golden
inputs/outputs pin the exact drafter contract that
``hipengine/speculative/dflash2_drafter.py`` (and later the native kernels)
must reproduce.

Writes tests/fixtures/cpu_reference/dflash2_drafter_golden.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SNAPSHOT = (
    "/home/lhl/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/"
    "snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=SNAPSHOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "cpu_reference" / "dflash2_drafter_golden.npz",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--block-len", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=256, help="synthetic output-head vocab")
    args = parser.parse_args()

    sys.path.insert(0, "/home/lhl/dflash")
    from dflash.model import DFlash2DraftModel  # noqa: E402

    model = DFlash2DraftModel.from_pretrained(args.snapshot, torch_dtype=torch.float32)
    model.eval()

    ctx, blk = args.ctx_len, args.block_len
    hidden_size = model.config.hidden_size
    n_taps = len(model.target_layer_ids)

    torch.manual_seed(args.seed)
    target_hidden_concat = torch.randn(1, ctx, n_taps * hidden_size, dtype=torch.float32)
    noise_embedding = torch.randn(1, blk, hidden_size, dtype=torch.float32)
    position_ids = torch.arange(ctx + blk, dtype=torch.long).unsqueeze(0)
    anchor_ids = torch.tensor([42], dtype=torch.long)

    head = nn.Linear(hidden_size, args.vocab, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.randn_like(head.weight))
    head.eval()

    with torch.no_grad():
        final_hidden = model(
            target_hidden=target_hidden_concat,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
        )
        draft_hidden = final_hidden[:, 1 - blk :, :]  # drop the anchor row
        logits = model.compute_logits(draft_hidden, head)
        path, candidates, q_rows = model.candidate_selector.select(
            draft_hidden, logits, anchor_ids, 0.0  # greedy
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        # inputs
        target_hidden_concat=target_hidden_concat.numpy(),
        noise_embedding=noise_embedding.numpy(),
        position_ids=position_ids.numpy(),
        anchor_ids=anchor_ids.numpy(),
        output_head_weight=head.weight.detach().numpy(),
        # outputs
        final_hidden=final_hidden.numpy(),
        draft_hidden=draft_hidden.numpy(),
        logits=logits.numpy(),
        path=path.numpy(),
        candidates=candidates.numpy(),
    )
    print(f"wrote {args.out}")
    print(f"block_size={blk} ctx={ctx} vocab={args.vocab} taps={n_taps} path={path.numpy().tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
