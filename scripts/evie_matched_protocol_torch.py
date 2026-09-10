"""Matched-protocol torch reference for EVIE (reviewer-requested harness).

Generates the shared inputs ONCE with the real ColQwen3_5Processor (8 seeded
synthetic 448x336 pages + 8 queries), saves them for the hipEngine side,
and times torch forwards (bf16 + fp32) with preprocessing OUTSIDE the
timers and full 8x8 scoring. Run in the ~/EVIE/colpali torch env:

    PYTHONPATH=~/EVIE/colpali python3 scripts/evie_matched_protocol_torch.py \
        [--out-dir /tmp/evie_matched]

Produces <out-dir>/inputs.npz, scores_bf16.npy, scores_fp32.npy.
"""

import argparse
import sys
import glob
import time
import numpy as np
import torch
from PIL import Image
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
sys.path.insert(0, "/home/lhl/EVIE/colpali")

import os
OUT = os.environ.get("EVIE_MATCHED_DIR", "/tmp/evie_matched")
os.makedirs(OUT, exist_ok=True)
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--tencent--EVIE-4.5B/snapshots/*"))[0]
rng = np.random.default_rng(20260909)
h, w = 448, 336
images = []
for _ in range(8):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = rng.integers(0, 256, (h, w))
    # text-like lines
    for r in range(0, h, 28):
        img[r:r+4, :] = rng.integers(0, 256, (4, w, 1))
    images.append(Image.fromarray(img))
queries = [f"What is the value shown in figure {i}?" for i in range(8)]

def set_active_head(model, dim):
    from colpali_engine.models.qwen3_5.colqwen3_5 import AttentionOutput
    def head(self, hidden, *a, **k):
        out = self.proj(hidden)
        if dim is not None:
            out = out[..., :dim]
        return out
    import types
    for layer in self.model.layers:
        layer.self_attn.head_proj = types.MethodType(head, layer.self_attn) if hasattr(layer.self_attn, "head_proj") else None
    # fall back to attribute-based if patch fails
    model._head_dim = dim

out = {}
for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
    model = ColQwen3_5.from_pretrained(snap, torch_dtype=dtype, device_map="cuda", attn_implementation="sdpa").eval()
    model.enable_bidirectional_attention()
    try:
        model.set_active_head(128)
    except Exception as e:
        print("set_active_head:", e)
    processor = ColQwen3_5Processor.from_pretrained(snap)
    doc_batch = processor.process_images(images).to("cuda")
    query_batch = processor.process_queries(queries).to("cuda")
    if dtype_name == "bf16":
        np.savez(OUT + "/inputs.npz",
                 doc_pv=doc_batch["pixel_values"].cpu().numpy(),
                 doc_grid=doc_batch["image_grid_thw"].cpu().numpy(),
                 doc_ids=doc_batch["input_ids"].cpu().numpy(),
                 doc_mask=doc_batch["attention_mask"].cpu().numpy(),
                 qry_ids=query_batch["input_ids"].cpu().numpy(),
                 qry_mask=query_batch["attention_mask"].cpu().numpy())
        print("doc ids:", doc_batch["input_ids"].shape, "qry ids:", query_batch["input_ids"].shape,
              "pv:", doc_batch["pixel_values"].shape, "grid:", doc_batch["image_grid_thw"].cpu().numpy().tolist())
    def run_once(reset=True):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        doc_emb = model(**doc_batch); torch.cuda.synchronize(); t1 = time.perf_counter()
        if reset: model.rope_deltas = None
        q_emb = model(**query_batch); torch.cuda.synchronize(); t2 = time.perf_counter()
        scores = processor.score(q_emb, doc_emb); torch.cuda.synchronize(); t3 = time.perf_counter()
        return t1-t0, t2-t1, t3-t2, scores
    with torch.inference_mode():
        run_once()
        best = None
        for _ in range(3):
            d, q, s, scores = run_once()
            tot = d+q+s
            if best is None or tot < best[0]:
                best = (tot, d, q, s)
                best_scores = scores.cpu().numpy()
        print(f"{dtype_name}: doc {best[1]*1e3:.1f} ms | query {best[2]*1e3:.1f} ms | score {best[3]*1e3:.1f} ms | total {best[0]*1e3:.1f} ms")
    out[dtype_name] = best_scores
    if dtype_name == "bf16":
        np.save(OUT + "/scores_bf16.npy", best_scores)
    else:
        np.save(OUT + "/scores_fp32.npy", best_scores)
    del model
    torch.cuda.empty_cache()
print("done")
