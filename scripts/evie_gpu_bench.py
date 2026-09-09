"""EVIE-4.5B torch GPU baseline: reference wall-clock on the local HIP GPU.

Measures the deployment configuration (bfloat16, sdpa) and the strict fp32
path for the two inference workloads that matter for a retrieval encoder:

- document indexing: page image -> multi-vector embeddings,
- query encoding: query text -> multi-vector embeddings,
- plus the MaxSim scoring step.

Workload is fixed and synthetic (deterministic pages, seed 20260909) so the
same script can be re-run for same-host old->new comparisons.

Usage:
    PYTHONPATH=~/EVIE/colpali python3 scripts/evie_gpu_bench.py [--dtype bf16|fp32]
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np


def _resolve_repo_path() -> str:
    home = Path.home() / "EVIE" / "colpali"
    if (home / "colpali_engine").exists():
        return str(home)
    print("colpali_engine not found (expected ~/EVIE/colpali)", file=sys.stderr)
    raise SystemExit(2)


def _make_page(rng: np.random.Generator, height: int, width: int) -> "Image.Image":
    from PIL import Image

    page = np.full((height, width, 3), 255, dtype=np.uint8)
    y = 24
    while y < height - 24:
        w = int(rng.integers(80, max(81, width - 80)))
        x0 = int(rng.integers(16, 40))
        page[y : y + 8, x0 : x0 + w] = 0
        y += 16
    return Image.fromarray(page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--pages", type=int, default=8, help="batch of doc pages")
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--page-size", default="448x336", help="HxW of the synthetic page"
    )
    args = parser.parse_args()

    import torch
    from PIL import Image

    torch.manual_seed(0)
    repo = _resolve_repo_path()
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
    from colpali_engine.models.qwen3_5.colqwen3_5.modeling_colqwen3_5 import (
        set_active_head,
    )

    snapshot = Path(
        sorted(
            glob.glob(
                str(
                    Path.home()
                    / ".cache/huggingface/hub/models--tencent--EVIE-4.5B/snapshots/*/model.safetensors"
                )
            )
        )[0]
    ).parent

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"loading {snapshot} on cuda ({args.dtype})...")
    model = ColQwen3_5.from_pretrained(
        str(snapshot),
        torch_dtype=dtype,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    model.enable_bidirectional_attention()
    set_active_head(model, 128)
    processor = ColQwen3_5Processor.from_pretrained(str(snapshot))

    rng = np.random.default_rng(20260909)
    h, w = (int(x) for x in args.page_size.split("x"))
    images = [_make_page(rng, h, w) for _ in range(args.pages)]
    queries = [f"What is the value shown in figure {i}?" for i in range(args.queries)]

    doc_batch = processor.process_images(images).to("cuda")
    query_batch = processor.process_queries(queries).to("cuda")
    print("doc tokens:", doc_batch["input_ids"].shape,
          "query tokens:", query_batch["input_ids"].shape)

    def run_once(reset_rope: bool = True) -> tuple[float, float, float]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        doc_emb = model(**doc_batch)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if reset_rope:
            model.rope_deltas = None
        q_emb = model(**query_batch)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        scores = processor.score(q_emb, doc_emb)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return t1 - t0, t2 - t1, t3 - t2, scores

    with torch.inference_mode():
        run_once()  # warmup
        best = None
        for _ in range(args.repeats):
            doc_s, q_s, score_s, scores = run_once()
            total = doc_s + q_s + score_s
            row = (doc_s, q_s, score_s, total)
            print(
                f"doc {doc_s*1e3:8.1f} ms | query {q_s*1e3:7.1f} ms | "
                f"maxsim {score_s*1e3:6.1f} ms | total {total*1e3:8.1f} ms"
            )
            if best is None or total < best[3]:
                best = row
    doc_s, q_s, score_s, total = best
    print(
        f"BEST {args.dtype}: doc {doc_s*1e3:.1f} ms ({args.pages} pages, {h}x{w}) | "
        f"query {q_s*1e3:.1f} ms ({args.queries} queries) | maxsim {score_s*1e3:.1f} ms | "
        f"total {total*1e3:.1f} ms"
    )
    print("score[0,0]:", float(scores[0, 0]))


if __name__ == "__main__":
    main()
