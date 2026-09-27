#!/usr/bin/env python3
"""Log the real full-attention launch geometry from a live prefill.

`run_qwen4_exp_attention` launches the dense context leaf as

    dense_attention(spans, dense_rows, start + dense_rows, ...)

so `rows` is the chunk's dense row count and `max_context_len` is the absolute
position the chunk ends at. Both are needed to build a microbench that matches
the production launch, and neither is readable from a kernel trace. This wrapper
records them, plus which of the two dense entry points ran and the live-count
distribution, then hands control to the harness.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from hipengine.runtime import qwen4_exp_runner as runner
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import host_array_ptr
from hipengine.core.runtime import MemcpyKind

RECORDS: list[dict] = []
_NAMES = (
    "qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_promoted_spans",
    "qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans",
    "qwen35_paged_full_attn_decode_context_bf16_batch_spans",
)


def _wrap(name: str, fn):
    def wrapped(*args, **kwargs):
        spans = args[4] if len(args) > 4 else kwargs.get("spans")
        rows = args[5] if len(args) > 5 else kwargs.get("rows")
        max_ctx = args[6] if len(args) > 6 else kwargs.get("max_context_len")
        rec = {"entry": name, "rows": rows, "max_context_len": max_ctx}
        try:
            n = int(spans.live_counts.numel)
            live = np.empty(n, dtype=np.int64)
            _rt = kwargs.get("runtime") or get_hip_runtime()
            _rt.memcpy(host_array_ptr(live), spans.live_counts.ptr, n * 8,
                       MemcpyKind.DEVICE_TO_HOST)
            rec["live_min"] = int(live.min())
            rec["live_mean"] = float(live.mean())
            rec["live_max"] = int(live.max())
            rec["live_sum"] = int(live.sum())
            rec["live_gt0"] = int((live > 0).sum())
            rec["live_gt_maxctx"] = int((live > int(max_ctx)).sum())
        except Exception as exc:  # pragma: no cover - diagnostics only
            rec["live_error"] = repr(exc)
        rt = kwargs.get("runtime")
        if rt is not None:
            rt.device_synchronize()
            t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if rt is not None:
            rt.device_synchronize()
            rec["ms"] = (time.perf_counter() - t0) * 1e3
        RECORDS.append(rec)
        _maybe_dump(args, rec)
        return out

    return wrapped


_DUMPED: set = set()


def _maybe_dump(args, rec) -> None:
    """Save one launch's exact inputs so a microbench can replay it."""

    target = os.environ.get("QWEN4EXP_DUMP_GEOM")
    if not target:
        return
    key = (rec.get("rows"), rec.get("max_context_len"))
    if str(key[0]) != target.split(":")[0] or str(key[1]) != target.split(":")[1]:
        return
    if key in _DUMPED:
        return
    _DUMPED.add(key)
    try:
        query_ptr, k_ptr, v_ptr, out_ptr = args[0], args[1], args[2], args[3]
        spans = args[4]
        rows, max_ctx = int(args[5]), int(args[6])
        block_size = int(args[7])
        # Python launcher signature: (q, k, v, out, spans, rows, max_context_len,
        # block_size, num_q_heads, num_kv_heads, head_dim, scale, *, ...)
        nqh, nkv, hd = int(args[8]), int(args[9]), int(args[10])
        if min(rows, max_ctx, block_size, nqh, nkv, hd) <= 0:
            raise ValueError(
                f"implausible geometry rows={rows} max_ctx={max_ctx} "
                f"block_size={block_size} nqh={nqh} nkv={nkv} hd={hd}"
            )
        blen = (max_ctx + block_size - 1) // block_size
        arrays = {
            "query": (query_ptr, rows * nqh * hd * 4, np.float32),
            "key": (k_ptr, blen * block_size * nkv * hd * 2, np.uint16),
            "value": (v_ptr, blen * block_size * nkv * hd * 2, np.uint16),
            "base_offsets": (spans.base_offsets.ptr,
                             int(spans.base_offsets.numel) * 4, np.int32),
            "live_counts": (spans.live_counts.ptr,
                            int(spans.live_counts.numel) * 8, np.int64),
        }
        out = {}
        for name, (ptr, nbytes, dt) in arrays.items():
            if nbytes <= 0:
                raise ValueError(f"{name}: computed {nbytes} bytes")
            a = np.empty(nbytes // np.dtype(dt).itemsize, dtype=dt)
            _rt = get_hip_runtime()
            _rt.memcpy(host_array_ptr(a), ptr, nbytes, MemcpyKind.DEVICE_TO_HOST)
            out[name] = a
        out["rows"] = np.array([rows], np.int64)
        out["max_context_len"] = np.array([max_ctx], np.int64)
        out["block_size"] = np.array([block_size], np.int64)
        out["nqh"] = np.array([nqh], np.int64)
        out["nkv"] = np.array([nkv], np.int64)
        out["hd"] = np.array([hd], np.int64)
        path = os.environ.get("QWEN4EXP_DUMP_PATH", "/tmp/fullattn-launch.npz")
        np.savez(path, **out)
        print(f"dumped launch rows={rows} max_ctx={max_ctx} block_size={block_size} "
              f"nqh={nqh} nkv={nkv} hd={hd} live[min/mean/max]="
              f"{out['live_counts'].min()}/{out['live_counts'].mean():.1f}/"
              f"{out['live_counts'].max()} -> {path}", file=sys.stderr)
    except Exception as exc:  # pragma: no cover - diagnostics only
        import traceback
        traceback.print_exc(file=sys.stderr)


for _n in _NAMES:
    if hasattr(runner, _n):
        setattr(runner, _n, _wrap(_n, getattr(runner, _n)))

from scripts import qwen4exp_profile_gap as harness  # noqa: E402


def _report():
    print("\n===== FULL-ATTN DENSE LAUNCH GEOMETRY =====", file=sys.stderr)
    print(f"  total launches recorded: {len(RECORDS)}", file=sys.stderr)
    for i, r in enumerate(RECORDS[:8]):
        short = r["entry"].replace("qwen35_paged_full_attn_decode_context_bf16_", "")
        live = ""
        if "live_mean" in r:
            live = (f" live[min/mean/max]={r['live_min']}/{r['live_mean']:.1f}/{r['live_max']}"
                    f" live_sum={r['live_sum']} gt0={r['live_gt0']}"
                    f" gt_maxctx={r['live_gt_maxctx']}")
        elif "live_error" in r:
            live = f" live_error={r['live_error'][:70]}"
        print(f"  [{i}] {short} rows={r['rows']} max_ctx={r['max_context_len']}"
              f" ms={r.get('ms')}{live}", file=sys.stderr)
    # Group by (entry, rows, max_ctx) and report per-group timing.
    groups: dict[tuple, list] = {}
    for r in RECORDS:
        groups.setdefault((r["entry"], r["rows"], r["max_context_len"]), []).append(r)
    print("  --- per-geometry (all launches in the run) ---", file=sys.stderr)
    for key, rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        short = key[0].replace("qwen35_paged_full_attn_decode_context_bf16_", "")
        ms = [x["ms"] for x in rs if "ms" in x]
        live_mean = [x["live_mean"] for x in rs if "live_mean" in x]
        live_sum = [x["live_sum"] for x in rs if "live_sum" in x]
        lmin = [x["live_min"] for x in rs if "live_min" in x]
        lmax = [x["live_max"] for x in rs if "live_max" in x]
        print(f"  {short} rows={key[1]} max_ctx={key[2]} n={len(rs)}"
              f" ms[min/med/max]={min(ms):.3f}/{sorted(ms)[len(ms)//2]:.3f}/{max(ms):.3f}"
              if ms else f"  {short} rows={key[1]} max_ctx={key[2]} n={len(rs)}"
              f" live[min/mean/max]={lmin[0]}/{live_mean[0]:.1f}/{lmax[0]}"
              f" live_sum={live_sum[0]}"
              if live_mean else "", file=sys.stderr)
    print("===========================================", file=sys.stderr)


if __name__ == "__main__":
    try:
        rc = harness.main()
    finally:
        _report()
    raise SystemExit(rc)
