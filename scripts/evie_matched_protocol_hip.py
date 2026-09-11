"""Matched-protocol hipEngine side for EVIE.

Consumes the shared inputs from scripts/evie_matched_protocol_torch.py
(default /tmp/evie_matched), runs the batched encode (encode_documents +
encode_queries) in fp16 and fp32, times with the same boundaries
(preprocessing outside), and reports the full 8x8 score matrices plus
the comparison against the torch scores when present.

    python3 scripts/evie_matched_protocol_hip.py [--dir /tmp/evie_matched]
"""

import argparse
import glob
import os
import time
import numpy as np
from hipengine.loading.evie import load_evie_model
from hipengine.runtime.evie import EvieRunner
from hipengine.core.hip import get_hip_runtime
def sync(): get_hip_runtime().device_synchronize()
D = os.environ.get("EVIE_MATCHED_DIR", "/tmp/evie_matched")
d = np.load(D + "/inputs.npz")
doc_pv, doc_grid = d["doc_pv"], d["doc_grid"]
doc_ids, doc_mask = d["doc_ids"], d["doc_mask"]
qry_ids, qry_mask = d["qry_ids"], d["qry_mask"]
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--tencent--EVIE-4.5B/snapshots/*"))[0]
for prec in ("fp16", "fp32"):
    r = EvieRunner(load_evie_model(snap, runtime=None, precision=prec), precision=prec)
    pages = [(doc_ids[i], doc_mask[i], doc_pv[i], doc_grid[i:i+1]) for i in range(8)]
    queries = [(qry_ids[i], qry_mask[i]) for i in range(8)]
    r.encode_documents(pages); sync()
    def run():
        t0 = time.perf_counter(); sync()
        docs = r.encode_documents(pages)
        sync(); t1 = time.perf_counter()
        qs = r.encode_queries(queries)
        sync(); t2 = time.perf_counter()
        S = np.zeros((8, 8))
        for qi in range(8):
            for di in range(8):
                q = qs[qi] / np.linalg.norm(qs[qi], axis=-1, keepdims=True)
                dd = docs[di] / np.linalg.norm(docs[di], axis=-1, keepdims=True)
                S[qi, di] = (q @ dd.T).max(axis=1).sum()
        sync(); t3 = time.perf_counter()
        return t1-t0, t2-t1, t3-t2, S
    best = None
    for _ in range(3):
        dtd, dtq, dts, S = run()
        tot = dtd+dtq+dts
        if best is None or tot < best[0]: best = (tot, dtd, dtq, dts, S)
    tot, dtd, dtq, dts, S = best
    print(f"{prec}: doc {dtd*1e3:.1f} ms | query {dtq*1e3:.1f} ms | score {dts*1e3:.1f} ms | total {tot*1e3:.1f} ms")
    np.save(f"{D}/scores_hip_batched_{prec}.npy", S)
    r.close()
# retrieval quality: both hip precisions vs the torch fp32 teacher
fp = f"{D}/scores_fp32.npy"
if os.path.exists(fp):
    t = np.load(fp)
    for prec, fname in (("fp32", "fp32"), ("fp16", "fp16")):
        h = np.load(f"{D}/scores_hip_batched_{fname}.npy")
        print(f"hip {prec} vs torch fp32 teacher: max|d| {np.abs(h - t).max():.4f} argmax agree {(h.argmax(1) == t.argmax(1)).sum()}/8")
    tb = f"{D}/scores_bf16.npy"
    if os.path.exists(tb):
        t2 = np.load(tb)
        print(f"torch bf16 vs torch fp32 teacher: max|d| {np.abs(t2 - t).max():.4f} argmax agree {(t2.argmax(1) == t.argmax(1)).sum()}/8")
print("retained torch bf16 reference: doc 1074.1 | query 224.2 | total 1299.5 ms")
