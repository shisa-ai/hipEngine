#!/usr/bin/env python3
"""Prove compact INT8 KV residency has no persistent BF16 shadow, mid-flight.

Runs one INT8 per-token/head (FP32 scales) request through the LLM lane and
samples per-session `device_kv_layout_audit()` from a side thread while the
request row is still resident. The persistent-side claim: zero BF16 payload
and zero BF16 mirror bytes at every sample; INT8 payload and scale bytes
equal the declared geometry exactly. Pool-plane accounting is read after the
request. Graph scratch is zero by construction (packed decode graphs
hard-require BF16 KV).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.util.amdgpu_vram import select_card  # noqa: E402

# Declared geometry: 16 full-attention layers, 4 KV heads, head_dim 256,
# block 256 tokens. INT8 payload plane set per page:
#   2 * 16 * 4 * 256 * 256 = 8,388,608 B
# FP32 per-token/head scales per page:
#   2 * 16 * 4 * 4 * 256 = 131,072 B
# A BF16 shadow plane set per page would be 16,777,216 B.
FULL_ATTN_LAYERS = 16
KV_HEADS = 4
HEAD_DIM = 256
BLOCK = 256
PAGE_INT8_BYTES = 2 * FULL_ATTN_LAYERS * KV_HEADS * HEAD_DIM * BLOCK
PAGE_SCALE_BYTES = 2 * FULL_ATTN_LAYERS * KV_HEADS * 4 * BLOCK
PAGE_BF16_SHADOW_BYTES = 2 * FULL_ATTN_LAYERS * KV_HEADS * HEAD_DIM * 2 * BLOCK


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    model_path = Path(args.model).resolve()
    card = select_card(pci_id=args.pci_id)
    import os

    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ["GPU_MAX_HW_QUEUES"] = "1"

    from hipengine.llm import LLM, SamplingParams

    llm = LLM(
        str(model_path),
        backend=args.backend,
        quant=args.quant,
        max_active_requests=1,
        max_sequence_length=args.max_sequence_length,
        kv_storage="int8_per_token_head",
        kv_scale_dtype="fp32",
        kv_scale_granularity="per_token_head",
    )
    adapter = llm._get_text_generator()
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        kv_storage="int8_per_token_head",
        kv_scale_dtype="fp32",
        kv_scale_granularity="per_token_head",
    )
    llm.prepare(max_sequence_length=args.max_sequence_length, sampling_params=sampling)
    runner = adapter._runner
    tokenizer = getattr(adapter, "tokenizer", None)

    filler_ids = list(range(1000, 1000 + args.prompt_tokens))
    detok = getattr(adapter, "detokenize", None)
    tok_fn = getattr(adapter, "tokenize", None)
    if callable(detok) and callable(tok_fn):
        prompt = str(detok(filler_ids, skip_special=False)) + "\n\nAnswer: shadow"
        prompt_ids = [int(t) for t in tok_fn(prompt)]
    else:
        prompt = None
        prompt_ids = filler_ids

    midflight_audits: list[dict[str, Any]] = []
    stop = threading.Event()

    def _sample_audits() -> None:
        while not stop.is_set():
            try:
                for session in runner._resident_sessions():
                    audit_fn = getattr(session, "device_kv_layout_audit", None)
                    if callable(audit_fn):
                        audit = audit_fn()
                        if int(audit.get("request_pages", 0)) > 0:
                            midflight_audits.append(audit)
            except Exception as exc:  # diagnostic sampler must not kill the run
                midflight_audits.append({"sampler_error": repr(exc)})
            time.sleep(0.05)

    audit_thread = threading.Thread(target=_sample_audits, daemon=True)
    audit_thread.start()
    generate_started = time.perf_counter()
    try:
        if prompt is not None:
            outputs = llm.generate([prompt], sampling)
        else:
            outputs = llm.generate(prompt_ids, sampling)
        elapsed = time.perf_counter() - generate_started
    finally:
        stop.set()
        audit_thread.join(timeout=2.0)

    resident_samples = [a for a in midflight_audits if "request_pages" in a]
    sampler_errors = [a for a in midflight_audits if "sampler_error" in a]
    best_pages = max((int(a["request_pages"]) for a in resident_samples), default=0)
    best_audit = next(
        (a for a in resident_samples if int(a["request_pages"]) == best_pages), None
    )

    checks: dict[str, bool] = {}
    for index, audit in enumerate(
        sorted(resident_samples, key=lambda a: int(a["request_pages"]))
    ):
        pages = int(audit["request_pages"])
        checks[f"sample{index}_zero_bf16_payload"] = (
            int(audit["persistent_bf16_payload_bytes"]) == 0
        )
        checks[f"sample{index}_zero_bf16_mirror"] = (
            int(audit["persistent_bf16_mirror_bytes"]) == 0
        )
        checks[f"sample{index}_int8_matches_geometry"] = (
            int(audit["persistent_int8_payload_bytes"]) == pages * PAGE_INT8_BYTES
        )
        checks[f"sample{index}_scales_match_geometry"] = (
            int(audit["persistent_scale_bytes"]) == pages * PAGE_SCALE_BYTES
        )
    checks["midflight_residency_observed"] = best_pages > 0
    checks["no_sampler_errors"] = not sampler_errors

    from hipengine.core.memory import live_allocation_histogram

    pool_snapshot = runner.kv_pool_memory_snapshot()
    histogram = live_allocation_histogram(min_bytes=64 << 20)

    passed = all(checks.values())
    return {
        "schema": 1,
        "kind": "qwen38_int8_kv_no_shadow_residency_audit",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "passed" if passed else "failed",
        "performance_claim": False,
        "diagnostic": True,
        "model": {
            "path": str(model_path),
            "quant": args.quant,
            "size_bytes": model_path.stat().st_size,
        },
        "device": {
            "pci_id": card.pci_id,
            "drm_card": card.card_name,
            "gpu_index": args.gpu_index,
        },
        "geometry": {
            "full_attention_layers": FULL_ATTN_LAYERS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "block_tokens": BLOCK,
            "page_int8_payload_bytes": PAGE_INT8_BYTES,
            "page_scale_bytes": PAGE_SCALE_BYTES,
            "page_bf16_shadow_bytes_if_present": PAGE_BF16_SHADOW_BYTES,
        },
        "workload": {
            "prompt_tokens": len(prompt_ids),
            "max_tokens": args.max_tokens,
            "generate_elapsed_s": round(elapsed, 3),
            "output_chars": len(outputs[0]) if outputs else 0,
        },
        "midflight_samples": len(resident_samples),
        "max_observed_request_pages": best_pages,
        "representative_audit": best_audit,
        "sampler_errors": sampler_errors[:4],
        "pool_after_request": {
            "storage_view": pool_snapshot.get("storage_view"),
            "dynamic_pool_current_bytes": (
                (pool_snapshot.get("dynamic_pool") or {}).get("current_bytes")
            ),
            "dynamic_pool_current_pages": (
                (pool_snapshot.get("dynamic_pool") or {}).get("current_pages")
            ),
        },
        "tracked_peak_large_allocations": histogram,
        "checks": checks,
        "graph_scratch_note": "packed decode graphs hard-require BF16 KV "
        "(gguf_packed_decode_graph raises NotImplementedError); INT8 decode "
        "runs eager, so graph scratch ownership is zero by construction",
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf", type=Path)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--pci-id", default="0000:10:00.0")
    parser.add_argument("--prompt-tokens", type=int, default=3072)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
