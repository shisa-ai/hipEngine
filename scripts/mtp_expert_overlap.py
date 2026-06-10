#!/usr/bin/env python3
"""P0 expert-overlap diagnostic: does 35B-A3B MoE dedup across verify rows?

Decides whether 35B-A3B speculative decode is structurally viable or a
research/persistent-kernel project (reviewer 2 P0).  Captures the TARGET
verifier's per-layer routed-expert selections for the B+1 rows of real verify
cycles (root + MTP-head drafts) by monkey-patching the FP16 router so it copies
``selected_experts`` (rows x top_k INT64) to host on every call -- diagnostic
only, no kernel changes, default behaviour untouched.

Per B / prompt / LAYER we record rows=B+1, total_lanes=rows*top_k,
unique_experts, unique_ratio=unique_experts/top_k, root-vs-draft Jaccard,
adjacent-token overlap, duplicate-savings=total_lanes/unique_experts, plus a
byte-weighted bandwidth floor (1-f)+f*U using ACTUAL per-layer expert/shared/
attention bytes, and the acceptance E_B from the same run.

The bandwidth-limit floor for a batched verify is verify_bytes/AR_bytes =
(1-f) + f*U where f = expert byte fraction and U = unique_experts/top_k.  Spec
decode is BW-viable at budget B when visible_tokens_per_cycle (from acceptance)
exceeds that floor.  U scales harshly with rows, so it decides whether B=1/2
beats B=3/5/7.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# The MoE C-side dispatcher (M14.dispatch.1, default-on) launches the router
# from C, bypassing the Python router wrappers we capture. Expert SELECTIONS are
# identical regardless of dispatch path (same kernels, same math), so force the
# Python router path for this diagnostic only. Must be set before the runtime
# resolves the gate.
os.environ.setdefault("HIPENGINE_MOE_C1_C_DISPATCH", "0")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
import hipengine.runtime.qwen35_paro as paro_mod
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from hipengine.speculative import MTP_CHAIN_CANDIDATE_BUDGETS
from scripts.mtp_chain_e2e_smoke import (
    _capture_tensor,
    _read_capture_row,
    _target_batch,
    run_native_mtp_proposal,
)

DEFAULT_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")

# ---------------------------------------------------------------------------
# Router capture: monkey-patch the FP16 router so every call appends the
# rows x top_k selected experts (host copy) to a module-global buffer.  The
# verify forward fires this once per MoE layer in topological order, so a
# drained buffer of length num_layers is the per-layer routing for one cycle.
# ---------------------------------------------------------------------------
_CAPTURE: list[np.ndarray] | None = None
_ORIG_ROUTERS: dict[str, Any] = {}


def _install_router_capture(runtime) -> None:
    # The verify forward mixes FP16 (chain_tloop linear) and BF16 (run_moe_c1)
    # routers; the router-weights are BF16 even when hidden is FP16. Patch all
    # variants so every MoE layer's routing is captured regardless of path.
    names = [
        "qwen35_router_topk_shared_out_fp16", "qwen35_router_topk_shared_coop_out_fp16",
        "qwen35_router_topk_shared_out_bf16", "qwen35_router_topk_shared_coop_out_bf16",
        "qwen35_router_topk_shared_sigmoid_out_fp16",
    ]

    def make_wrapper(orig):
        def wrapper(*args, **kwargs):
            r = orig(*args, **kwargs)
            if _CAPTURE is not None:
                selected_ptr = int(args[3])
                tokens = int(args[5])
                top_k = int(args[9])
                host = np.empty((tokens, top_k), dtype=np.int64)
                runtime.device_synchronize()
                copy_device_to_host(
                    host_array_ptr(host),
                    DeviceBuffer(ptr=selected_ptr, nbytes=host.nbytes),
                    host.nbytes,
                )
                _CAPTURE.append(host)
            return r

        return wrapper

    for n in names:
        orig = getattr(paro_mod, n, None)
        if orig is None:
            continue
        _ORIG_ROUTERS[n] = orig
        setattr(paro_mod, n, make_wrapper(orig))


def _drain_capture() -> list[np.ndarray]:
    global _CAPTURE
    out = list(_CAPTURE or [])
    _CAPTURE = []
    return out


# ---------------------------------------------------------------------------
# Per-layer byte accounting from the safetensors header (data_offsets give
# exact byte ranges).  We classify every tensor of a target language-model
# layer into expert (per routed expert), shared expert, attention, router, and
# norm/other so f = expert byte fraction is grounded in the real packed model.
# ---------------------------------------------------------------------------
def _read_safetensors_layer_bytes(model_dir: Path) -> dict[str, Any]:
    files = sorted(glob.glob(str(model_dir / "*.safetensors")))
    # per layer: category -> bytes ; experts tracked per expert-id to average
    layer_cat: dict[int, dict[str, int]] = {}
    layer_expert_bytes: dict[int, dict[int, int]] = {}
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            off = v.get("data_offsets")
            if not off:
                continue
            nbytes = int(off[1]) - int(off[0])
            # Target language-model layers only -- exclude the MTP head
            # (mtp.layers.*) and any non-layer tensors (embed/lm_head/norm).
            m = re.search(r"language_model\.layers\.(\d+)\.", k)
            if not m:
                continue
            L = int(m.group(1))
            cat = layer_cat.setdefault(L, {})
            em = re.search(r"\.mlp\.experts\.(\d+)\.", k)
            if em:
                eid = int(em.group(1))
                layer_expert_bytes.setdefault(L, {}).setdefault(eid, 0)
                layer_expert_bytes[L][eid] += nbytes
                cat["expert_total"] = cat.get("expert_total", 0) + nbytes
            elif "shared_expert" in k or "shared_mlp" in k or (".mlp." in k and "experts" not in k and "gate" in k and "router" not in k):
                cat["shared"] = cat.get("shared", 0) + nbytes
            elif "router" in k or "gate" in k and ".mlp." in k:
                cat["router"] = cat.get("router", 0) + nbytes
            elif "attn" in k or "attention" in k:
                cat["attention"] = cat.get("attention", 0) + nbytes
            else:
                cat["other"] = cat.get("other", 0) + nbytes
    # average per-expert bytes per layer
    per_expert: dict[int, float] = {}
    for L, ed in layer_expert_bytes.items():
        if ed:
            per_expert[L] = sum(ed.values()) / len(ed)
    return {"layer_cat": layer_cat, "per_expert_bytes": per_expert}


def _layer_byte_model(model_dir: Path, top_k: int) -> dict[str, Any]:
    info = _read_safetensors_layer_bytes(model_dir)
    per_expert = info["per_expert_bytes"]
    layer_cat = info["layer_cat"]
    rows = []
    for L in sorted(layer_cat):
        cat = layer_cat[L]
        e = per_expert.get(L, 0.0)
        non_expert = float(cat.get("shared", 0) + cat.get("router", 0) + cat.get("attention", 0) + cat.get("other", 0))
        ar_bytes = non_expert + top_k * e  # per-token AR bytes for this layer
        f = (top_k * e) / ar_bytes if ar_bytes > 0 else 0.0
        rows.append({"layer": L, "per_expert_bytes": e, "non_expert_bytes": non_expert,
                     "ar_bytes": ar_bytes, "expert_fraction_f": f, "has_experts": e > 0})
    moe_rows = [r for r in rows if r["has_experts"]]
    f_mean = float(np.mean([r["expert_fraction_f"] for r in moe_rows])) if moe_rows else 0.0
    return {"per_layer": rows, "f_mean_moe_layers": f_mean,
            "moe_layer_count": len(moe_rows), "total_layers": len(rows)}


# ---------------------------------------------------------------------------
# Overlap metrics for one cycle's per-layer routing (list of [rows, top_k]).
# ---------------------------------------------------------------------------
def _cycle_layer_metrics(per_layer_selected: list[np.ndarray], top_k: int) -> list[dict[str, Any]]:
    out = []
    for L, sel in enumerate(per_layer_selected):
        rows = int(sel.shape[0])
        lanes = rows * top_k
        flat = sel.reshape(-1)
        unique = int(np.unique(flat).size)
        # root-vs-draft Jaccard (root=row0 vs union of draft rows)
        root = set(int(x) for x in sel[0].tolist())
        if rows > 1:
            draft = set()
            for r in range(1, rows):
                draft |= set(int(x) for x in sel[r].tolist())
            inter = len(root & draft)
            uni = len(root | draft)
            jacc = inter / uni if uni else 0.0
        else:
            jacc = 1.0
        # adjacent-row mean Jaccard (row i vs row i+1)
        adj = []
        for r in range(rows - 1):
            a = set(int(x) for x in sel[r].tolist())
            b = set(int(x) for x in sel[r + 1].tolist())
            u = len(a | b)
            adj.append((len(a & b) / u) if u else 0.0)
        out.append({
            "layer": L, "rows": rows, "total_lanes": lanes, "unique_experts": unique,
            "unique_ratio": unique / top_k, "U": unique / top_k,
            "root_draft_jaccard": jacc,
            "adjacent_token_overlap": float(np.mean(adj)) if adj else 1.0,
            "duplicate_savings_upper_bound": lanes / unique if unique else 1.0,
        })
    return out


def _run_budget(
    session: Qwen35ParoResidentSession,
    model: Path,
    prompt_tokens: Sequence[int],
    *,
    candidate_budget: int,
    decode_tokens: int,
    capture_buf: DeviceBuffer,
    capture_rows: int,
    hidden: int,
    capture_layer_id: int,
    top_k: int,
) -> dict[str, Any]:
    """Run real verify cycles at one budget, capturing per-layer routing."""
    global _CAPTURE
    capture = _capture_tensor(capture_buf, capture_rows, hidden)
    next_result = None
    for pos, token in enumerate(prompt_tokens):
        next_result = session.step_with_hidden_taps(
            int(token), position=pos, capture_layer_ids=(capture_layer_id,),
            capture_hidden_concat=capture, capture_row=pos, sample=(pos == len(prompt_tokens) - 1),
        )
    root = int(next_result.token_id)
    context = len(prompt_tokens)
    previous_hidden_row = context - 1
    cycle_layer_rows: list[list[dict[str, Any]]] = []
    accepted_lengths: list[int] = []
    generated = 0
    while generated < decode_tokens:
        remaining = decode_tokens - generated
        active_budget = min(candidate_budget, max(0, remaining - 1))
        allowed = [b for b in MTP_CHAIN_CANDIDATE_BUDGETS if b <= active_budget]
        active_budget = max(allowed) if allowed else 0
        if active_budget <= 0:
            step_result = session.step_with_hidden_taps(
                root, position=context, capture_layer_ids=(capture_layer_id,),
                capture_hidden_concat=capture, capture_row=context, sample=True)
            root = int(step_result.token_id)
            previous_hidden_row = context
            context += 1
            generated += 1
            continue
        target_hidden = _read_capture_row(capture_buf, previous_hidden_row, hidden)
        proposal = run_native_mtp_proposal(
            model, root_token=root, root_position=context, draft_budget=active_budget,
            torch_compare=False, target_hidden_bits_override=target_hidden)
        candidates = [int(t) for t in proposal["candidate_tokens"][:active_budget]]
        target_batch = _target_batch(root, context, candidates, active_budget)
        # capture only the verify forward's routing
        _CAPTURE = []
        verify = session.verify_chain_bulk_and_commit(
            target_batch, base_slot=0, capture_layer_ids=(capture_layer_id,),
            capture_hidden_concat=capture, capture_row_start=context, chain_attn_mode="batched")
        captured = _drain_capture()
        _CAPTURE = None
        if captured:
            cycle_layer_rows.append(_cycle_layer_metrics(captured, top_k))
        accepted = int(verify.accepted_count)
        accepted_lengths.append(accepted)
        committed = [root, *candidates[:accepted]]
        generated += len(committed)
        bonus = int(verify.next_token) if verify.next_token is not None else int(verify.target_top1[min(accepted, len(verify.target_top1) - 1)])
        previous_hidden_row = context + len(committed) - 1
        context += len(committed)
        root = bonus
    return {"cycle_layer_rows": cycle_layer_rows, "accepted_lengths": accepted_lengths}


def _aggregate(cycle_layer_rows: list[list[dict[str, Any]]], rows_expected: int) -> dict[str, Any]:
    """Average per-layer metrics across cycles; also overall (mean over layers)."""
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for cyc in cycle_layer_rows:
        for m in cyc:
            by_layer.setdefault(m["layer"], []).append(m)
    per_layer = []
    for L in sorted(by_layer):
        ms = by_layer[L]
        per_layer.append({
            "layer": L,
            "rows": ms[0]["rows"],
            "total_lanes": ms[0]["total_lanes"],
            "unique_experts": float(np.mean([m["unique_experts"] for m in ms])),
            "U": float(np.mean([m["U"] for m in ms])),
            "root_draft_jaccard": float(np.mean([m["root_draft_jaccard"] for m in ms])),
            "adjacent_token_overlap": float(np.mean([m["adjacent_token_overlap"] for m in ms])),
            "duplicate_savings_upper_bound": float(np.mean([m["duplicate_savings_upper_bound"] for m in ms])),
            "samples": len(ms),
        })
    return {"per_layer": per_layer,
            "U_mean": float(np.mean([r["U"] for r in per_layer])) if per_layer else None,
            "unique_experts_mean": float(np.mean([r["unique_experts"] for r in per_layer])) if per_layer else None,
            "root_draft_jaccard_mean": float(np.mean([r["root_draft_jaccard"] for r in per_layer])) if per_layer else None,
            "duplicate_savings_mean": float(np.mean([r["duplicate_savings_upper_bound"] for r in per_layer])) if per_layer else None}


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model)
    prompt_tokens = tuple(int(p.strip()) for p in str(args.prompt_tokens).split(",") if p.strip())
    budgets = [int(b) for b in str(args.candidate_budgets).split(",") if b.strip()]
    top_k = 8
    byte_model = _layer_byte_model(model, top_k)
    f = byte_model["f_mean_moe_layers"]

    runner = Qwen35ParoNextTokenRunner(model, backend=str(args.backend))
    _install_router_capture(runner.runtime)
    results = []
    skipped = []
    started = time.perf_counter()
    for B in budgets:
        if B not in MTP_CHAIN_CANDIDATE_BUDGETS:
            skipped.append({"candidate_budget": B, "reason": f"not an allowed MTP chain budget {sorted(MTP_CHAIN_CANDIDATE_BUDGETS)}; rows={B+1} not realizable by the native proposer"})
            continue
        max_sequence = len(prompt_tokens) + int(args.decode_tokens) + B + 4
        run_aggs = []
        accepted_all: list[int] = []
        for _run in range(int(args.runs)):
            with Qwen35ParoResidentSession(runner, max_sequence_length=max_sequence, max_batch_size=B + 1) as session:
                hidden = int(session.config.hidden_size)
                capture_layer_id = int(session.layer_limit) - 1
                capture_rows = max_sequence + B + 2
                capture_buf = malloc(capture_rows * hidden * DType.BF16.itemsize, runtime=session.runtime)
                try:
                    out = _run_budget(session, model, prompt_tokens, candidate_budget=B,
                                      decode_tokens=int(args.decode_tokens), capture_buf=capture_buf,
                                      capture_rows=capture_rows, hidden=hidden,
                                      capture_layer_id=capture_layer_id, top_k=top_k)
                finally:
                    free(capture_buf, runtime=session.runtime)
            run_aggs.extend(out["cycle_layer_rows"])
            accepted_all.extend(out["accepted_lengths"])
        agg = _aggregate(run_aggs, rows_expected=B + 1)
        ncyc = len(accepted_all) or 1
        avg_accept = sum(accepted_all) / ncyc
        visible = 1.0 + avg_accept
        U = agg["U_mean"] or 1.0
        floor = (1 - f) + f * U
        floor_nodedup = (1 - f) + f * (B + 1)  # worst case U=rows
        results.append({
            "candidate_budget": B, "rows": B + 1, "top_k": top_k,
            "runs": int(args.runs), "cycles": ncyc,
            "acceptance_E_B": {"avg_accepted": avg_accept, "visible_tokens_per_cycle": visible,
                               "accept_histogram": {str(k): accepted_all.count(k) for k in sorted(set(accepted_all))}},
            "U_mean": U, "unique_experts_mean": agg["unique_experts_mean"],
            "root_draft_jaccard_mean": agg["root_draft_jaccard_mean"],
            "duplicate_savings_mean": agg["duplicate_savings_mean"],
            "bw_floor_measured": floor, "bw_floor_no_dedup": floor_nodedup,
            "viable_measured": visible > floor, "viable_no_dedup": visible > floor_nodedup,
            "headroom_measured": visible - floor,
            "per_layer": agg["per_layer"],
        })
    elapsed = time.perf_counter() - started
    return {
        "status": "passed", "performance_claim": False, "kind": "diagnostic-expert-overlap",
        "model": str(model), "backend": str(args.backend), "prompt_tokens": list(prompt_tokens),
        "decode_tokens": int(args.decode_tokens), "runs": int(args.runs),
        "top_k": top_k, "num_experts": 256, "num_layers": byte_model["total_layers"],
        "expert_byte_fraction_f": f, "byte_model_summary": {
            "f_mean_moe_layers": f, "moe_layer_count": byte_model["moe_layer_count"]},
        "results": results, "skipped": skipped, "elapsed_seconds": elapsed,
        "note": ("BW floor = (1-f)+f*U, U=unique_experts/top_k. Spec decode is BW-viable at "
                 "budget B when visible_tokens_per_cycle > floor. U from REAL verify routing "
                 "captured via FP16-router monkey-patch (diagnostic only). B not in {1,2,3,5} is "
                 "skipped (native MTP proposer cannot draft it)."),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--prompt-tokens", default="727,3841,6646,19490,1590,198,262,4071,5423,264")
    p.add_argument("--candidate-budgets", default="1,2,3,5,7")
    p.add_argument("--decode-tokens", type=int, default=24)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--backend", default="hip_gfx1100")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = run(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {"status": result["status"], "f": round(result["expert_byte_fraction_f"], 4),
               "rows": [{"B": r["candidate_budget"], "U": round(r["U_mean"], 3),
                         "uniq": round(r["unique_experts_mean"], 1),
                         "visible": round(r["acceptance_E_B"]["visible_tokens_per_cycle"], 3),
                         "floor": round(r["bw_floor_measured"], 3),
                         "viable": r["viable_measured"]} for r in result["results"]],
               "skipped": [s["candidate_budget"] for s in result["skipped"]]}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
