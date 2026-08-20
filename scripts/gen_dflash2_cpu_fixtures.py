#!/usr/bin/env python3
"""Generate DFlash2 CPU-reference golden fixtures from the z-lab/dflash torch reference.

Test-time only. Requires the ``torch`` extra (never used on the hot path). The
``GroupedDynamicCausalConv`` and ``CandidateSelector`` modules below are copied
verbatim from ``~/dflash/dflash/model.py`` (z-lab/dflash @ 07ebd93) so the
golden values come from the true reference math.

Writes:
  tests/fixtures/cpu_reference/dflash2_conv_prepare.json
  tests/fixtures/cpu_reference/dflash2_conv_finish.json
  tests/fixtures/cpu_reference/dflash2_selector_path.json
  tests/fixtures/cpu_reference/dflash2_selector_full.npz

The JSON fixtures follow the LayerFixture schema (single-array expected) so
they can be consumed by ``load_fixture`` / ``run_fixture``; the .npz carries
the full selector result (path, candidates, unary, scores) for the dedicated
pytest comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from hipengine.kernels.cpu_reference.fixtures import LayerFixture, save_fixture  # noqa: E402


# ---------------------------------------------------------------------------
# Reference math copied verbatim from ~/dflash/dflash/model.py @ 07ebd93
# ---------------------------------------------------------------------------


def _grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.view_as(hidden)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size, kernel_size, group_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * groups, bias=False)

    def prepare(self, hidden):
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).view(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        return (
            _grouped_dynamic_convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden, dynamic):
        return _grouped_dynamic_convolve(hidden, dynamic, self.base_kernel[1], self.group_size)


class CandidateSelector(nn.Module):
    def __init__(self, hidden_size, vocab_size, rank, top_k):
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Embedding(vocab_size, rank)
        self.successor_codebook = nn.Embedding(vocab_size, rank)
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)

    def select(self, hidden, logits, anchor_ids):
        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        hidden = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path, q_rows, scores_all = [], [], []
        for position in range(hidden.shape[1]):
            scores = unary[:, position] + torch.einsum(
                "br,bkr->bk",
                self.predecessor_codebook(predecessor) * hidden[:, position],
                self.successor_codebook(candidates[:, position]),
            )
            index = torch.argmax(scores, dim=-1)
            predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
            path.append(predecessor)
            scores_all.append(scores)
        return (
            torch.stack(path, dim=1),
            candidates,
            unary,
            torch.stack(scores_all, dim=1),
        )


def _make_conv_fixtures(seed: int, out_dir: Path) -> None:
    torch.manual_seed(seed)
    hidden_size, kernel_size, group_size = 32, 2, 8
    length = 5
    conv = GroupedDynamicCausalConv(hidden_size, kernel_size, group_size).eval()
    # The reference leaves base_kernel as torch.empty (uninitialized); initialize
    # it deterministically so the golden fixtures are reproducible and finite.
    with torch.no_grad():
        conv.base_kernel.copy_(torch.randn_like(conv.base_kernel))
    hidden = torch.randn(1, length, hidden_size, dtype=torch.float32)

    prepared, output_dynamic = conv.prepare(hidden)
    finished = conv.finish(prepared, output_dynamic)
    groups = hidden_size // group_size
    projected = conv.kernel_projection(hidden).view(1, length, 2, kernel_size, groups)
    input_dynamic = projected[..., 0, :, :]

    save_fixture(
        out_dir / "dflash2_conv_prepare.json",
        LayerFixture(
            name="dflash2_conv_prepare",
            layer="dflash2_grouped_conv",
            quant="fp32",
            inputs={
                "hidden": hidden.detach().numpy(),
                "dynamic": input_dynamic.detach().numpy(),
                "base": conv.base_kernel[0].detach().numpy(),
                "group_size": group_size,
            },
            expected=prepared.detach().numpy(),
            metadata={"source": "z-lab/dflash @ 07ebd93 GroupedDynamicCausalConv.prepare", "seed": str(seed)},
        ),
    )
    save_fixture(
        out_dir / "dflash2_conv_finish.json",
        LayerFixture(
            name="dflash2_conv_finish",
            layer="dflash2_grouped_conv",
            quant="fp32",
            inputs={
                "hidden": prepared.detach().numpy(),
                "dynamic": output_dynamic.detach().numpy(),
                "base": conv.base_kernel[1].detach().numpy(),
                "group_size": group_size,
            },
            expected=finished.detach().numpy(),
            metadata={"source": "z-lab/dflash @ 07ebd93 GroupedDynamicCausalConv.finish", "seed": str(seed)},
        ),
    )


def _make_selector_fixtures(seed: int, out_dir: Path) -> None:
    torch.manual_seed(seed)
    hidden_size, vocab_size, rank, top_k = 16, 40, 8, 6
    length, batch = 4, 2
    selector = CandidateSelector(hidden_size, vocab_size, rank, top_k).eval()
    hidden = torch.randn(batch, length, hidden_size, dtype=torch.float32)
    logits = torch.randn(batch, length, vocab_size, dtype=torch.float32)
    anchor_ids = torch.randint(0, vocab_size, (batch,))

    path, candidates, unary, scores = selector.select(hidden, logits, anchor_ids)

    path_arr = path.detach().numpy()
    save_fixture(
        out_dir / "dflash2_selector_path.json",
        LayerFixture(
            name="dflash2_selector_path",
            layer="dflash2_selector_path",
            quant="fp32",
            inputs={
                "hidden": hidden.detach().numpy(),
                "logits": logits.detach().numpy(),
                "anchor_ids": anchor_ids.detach().numpy(),
                "predecessor_codebook": selector.predecessor_codebook.weight.detach().numpy(),
                "successor_codebook": selector.successor_codebook.weight.detach().numpy(),
                "hidden_projection": selector.hidden_projection.weight.detach().numpy(),
                "top_k": top_k,
            },
            expected=path_arr,
            metadata={"source": "z-lab/dflash @ 07ebd93 CandidateSelector.select (greedy)", "seed": str(seed)},
        ),
    )
    np.savez(
        out_dir / "dflash2_selector_full.npz",
        path=path_arr,
        candidates=candidates.detach().numpy(),
        unary=unary.detach().numpy(),
        scores=scores.detach().numpy(),
        anchor_ids=anchor_ids.detach().numpy(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "cpu_reference",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _make_conv_fixtures(args.seed, args.out_dir)
    _make_selector_fixtures(args.seed, args.out_dir)
    print(f"wrote DFlash2 fixtures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
