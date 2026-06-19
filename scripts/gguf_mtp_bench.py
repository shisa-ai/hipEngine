#!/usr/bin/env python3
"""M6 MTP speculative decoding benchmark.

Runs the hipEngine GGUF MTP speculative decoding pipeline with a fixed prompt,
measures decode speed (tokens/sec) and acceptance metrics (accept_per_draft,
accepted_per_output), and saves a compact JSON artifact in benchmarks/results/.

Usage:
    python3 scripts/gguf_mtp_bench.py [--model GGUF_PATH] [--cycles N]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

GGUF_PATH = "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DEFAULT_PROMPT = "What is the capital of France?"
IM_START_TOKEN = 248045
IM_END_TOKEN = 248046
THINK_START_TOKEN = 248068
THINK_END_TOKEN = 248069


def build_chat_prompt(tokenizer, user_prompt: str = DEFAULT_PROMPT, *, reasoning: str = "off") -> list[int]:
    """Build the llama.cpp-compatible Qwen chat prompt for GGUF MTP.

    llama.cpp ``--reasoning off`` still renders an empty thinking block for this
    GGUF chat template.  Include that suffix by default so native accepted/output
    diagnostics compare against the same token stream as llama-server.
    """
    if reasoning not in {"off", "none"}:
        raise ValueError("build_chat_prompt currently supports only reasoning='off'/'none'")
    prompt = (
        [IM_START_TOKEN]
        + tokenizer.encode(f"user\n{user_prompt}")
        + [IM_END_TOKEN]
        + tokenizer.encode("\n")
        + [IM_START_TOKEN]
        + tokenizer.encode("assistant\n")
    )
    if reasoning == "off":
        prompt += [THINK_START_TOKEN] + tokenizer.encode("\n\n") + [THINK_END_TOKEN] + tokenizer.encode("\n\n\n")
    return prompt


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _get_hw_info() -> dict:
    """Get GPU hardware info."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["rocminfo"], stderr=subprocess.DEVNULL, text=True, timeout=10)
        for line in out.split("\n"):
            if "Marketing Name" in line:
                gpu = line.split(":", 1)[1].strip()
                return {"gpu": gpu, "arch": "gfx1151"}
    except Exception:
        pass
    return {"gpu": "unknown", "arch": "gfx1151"}


def select_topk_tokens(logits_row: "np.ndarray", *, k: int = 10) -> tuple[int, list[int]]:
    """Return greedy token and descending top-k token IDs for one logits row."""
    if logits_row.ndim != 1:
        raise ValueError("logits_row must be rank-1")
    limit = min(k, int(logits_row.shape[0]))
    top_idx = np.argpartition(logits_row, -limit)[-limit:]
    top_sorted = top_idx[np.argsort(logits_row[top_idx])[::-1]]
    top_tokens = [int(t) for t in top_sorted]
    return top_tokens[0], top_tokens


def validate_draft_n_max(draft_n_max: int) -> int:
    """Validate the benchmark's currently implemented draft depth.

    The native GGUF path in this script currently implements B1 and the first
    target-attached B2 driver. Refuse B3/B4-looking invocations until deeper
    target-attached multi-draft context is implemented, so artifacts cannot
    silently mislabel B2 measurements as B3/B4.
    """
    if draft_n_max not in (1, 2):
        raise ValueError(
            "scripts/gguf_mtp_bench.py currently implements only B1/B2 "
            "(draft_n_max=1 or 2); B3-B4 target-attached GGUF MTP is not wired yet"
        )
    return draft_n_max


def compute_speculative_metrics(cycles: list[dict]) -> dict:
    """Compute MTP metrics with explicit llama.cpp-compatible denominators.

    A verify cycle always emits one target/corrective token. Accepted draft
    tokens are also visible output tokens, so ``accepted_per_output`` uses
    ``accepted_draft_tokens / visible_output_token_count`` rather than dividing
    by verify-cycle count. This follows docs/MTP-gguf.md's denominator contract.
    """
    verify_cycle_count = len(cycles)
    total_drafts = sum(int(c.get("generated_draft_tokens", 1)) for c in cycles)
    total_accepted = sum(int(c.get("accepted_draft_tokens", int(bool(c.get("accepted"))))) for c in cycles)
    visible_output_tokens = sum(
        int(c.get("visible_output_tokens", 1 + int(c.get("accepted_draft_tokens", int(bool(c.get("accepted")))))))
        for c in cycles
    )
    total_cycle_ms = sum(float(c.get("ar_decode_ms", 0.0)) + float(c.get("mtp_draft_ms", 0.0)) for c in cycles)
    total_ar_ms = sum(float(c.get("ar_decode_ms", 0.0)) for c in cycles)
    total_draft_ms = sum(float(c.get("mtp_draft_ms", 0.0)) for c in cycles)

    accept_per_draft = total_accepted / total_drafts if total_drafts > 0 else 0.0
    accepted_per_output = total_accepted / visible_output_tokens if visible_output_tokens > 0 else 0.0
    visible_tokens_per_cycle = visible_output_tokens / verify_cycle_count if verify_cycle_count > 0 else 0.0
    avg_cycle_ms = total_cycle_ms / verify_cycle_count if verify_cycle_count > 0 else 0.0
    avg_ar_decode_ms = total_ar_ms / verify_cycle_count if verify_cycle_count > 0 else 0.0
    avg_mtp_draft_ms = total_draft_ms / verify_cycle_count if verify_cycle_count > 0 else 0.0
    avg_ms_per_visible_token = total_cycle_ms / visible_output_tokens if visible_output_tokens > 0 else 0.0
    tokens_per_sec = 1000.0 / avg_ms_per_visible_token if avg_ms_per_visible_token > 0 else 0.0
    ar_baseline_tokens_per_sec = (
        1000.0 * visible_output_tokens / total_ar_ms if total_ar_ms > 0 and visible_output_tokens > 0 else 0.0
    )
    speedup_vs_ar_visible = (
        tokens_per_sec / ar_baseline_tokens_per_sec if ar_baseline_tokens_per_sec > 0 else 0.0
    )

    return {
        "accept_per_draft": accept_per_draft,
        "accepted_per_output": accepted_per_output,
        "avg_cycle_ms": avg_cycle_ms,
        "avg_decode_ms": avg_cycle_ms,  # backward-compatible alias for older artifacts
        "avg_ar_decode_ms": avg_ar_decode_ms,
        "avg_mtp_draft_ms": avg_mtp_draft_ms,
        "avg_ms_per_visible_token": avg_ms_per_visible_token,
        "tokens_per_sec": tokens_per_sec,
        "ar_baseline_tokens_per_sec": ar_baseline_tokens_per_sec,
        "speedup_vs_ar_visible": speedup_vs_ar_visible,
        "visible_tokens_per_cycle": visible_tokens_per_cycle,
        "total_accepted": total_accepted,
        "total_drafts": total_drafts,
        "total_output_tokens": visible_output_tokens,
        "verify_cycle_count": verify_cycle_count,
        "total_cycle_ms": total_cycle_ms,
        "denominators": {
            "accept_per_draft": "accepted_draft_tokens / generated_draft_tokens",
            "accepted_per_output": "accepted_draft_tokens / visible_output_token_count",
            "visible_tokens_per_cycle": "visible_output_token_count / verify_cycle_count",
            "tokens_per_sec": "visible_output_token_count / total_cycle_wall_time",
        },
    }


def llama_cpp_acceptance_from_target_samples(
    draft_tokens: list[int],
    target_samples: list[int],
) -> dict[str, object]:
    """Summarize llama.cpp draft-MTP accept/commit semantics.

    ``target_samples`` are the target-model greedy samples for rows
    ``[sampled_token] + accepted_draft_prefix``.  The list must include the
    corrective target row after the accepted prefix, so a fully accepted B2 draft
    has three target samples: draft0, draft1, corrective.
    """
    if not draft_tokens:
        raise ValueError("draft_tokens must be non-empty")
    if not target_samples:
        raise ValueError("target_samples must be non-empty")

    drafts = [int(token) for token in draft_tokens]
    targets = [int(token) for token in target_samples]
    accepted = 0
    for draft_token, target_token in zip(drafts, targets, strict=False):
        if draft_token != target_token:
            break
        accepted += 1
        if accepted == len(drafts):
            break

    if len(targets) <= accepted:
        raise ValueError(
            "target_samples must include the corrective target token after the accepted prefix"
        )

    output_tokens = targets[:accepted] + [targets[accepted]]
    comparison_target_tokens = targets[: min(len(drafts), len(targets))]
    return {
        "accepted_draft_tokens": accepted,
        "visible_output_tokens": len(output_tokens),
        "output_tokens": output_tokens,
        "comparison_target_tokens": comparison_target_tokens,
        "pending_hidden_row_index": accepted,
    }


def main():
    parser = argparse.ArgumentParser(description="MTP speculative decoding benchmark")
    parser.add_argument("--model", default=GGUF_PATH, help="GGUF model path")
    parser.add_argument("--cycles", type=int, default=10, help="Number of speculate-verify cycles")
    parser.add_argument("--draft-n-max", type=int, default=1, help="Max draft tokens per cycle")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt text before the assistant turn")
    parser.add_argument("--output", default=None, help="Output JSON path (default: benchmarks/results/mtp-bench-<timestamp>.json)")
    args = parser.parse_args()
    try:
        args.draft_n_max = validate_draft_n_max(args.draft_n_max)
    except ValueError as exc:
        parser.error(str(exc))

    if not _hip_available():
        print("ERROR: ROCm/HIP not available", file=sys.stderr)
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    from hipengine.loading.gguf import GGUFReader
    from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
    from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import (
        qwen35_gguf_mtp_nextn_layer_logits_f32 as gpu_kernel,
    )
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.core.hip import get_hip_runtime, HipMemcpyKind

    # Load model info + weights
    r = GGUFReader(args.model)
    tok = Qwen35GGUFTokenizer.from_gguf_info(r.info)
    meta = r.info.metadata

    weights = {}
    for t in r.info.tensors:
        if "blk.40" in t.name or t.name == "output.weight" or t.name == "token_embd.weight":
            data = r.tensor_data(t.name)
            weights[t.name] = (data, t.ggml_type, t.shape)

    def get(name): return weights[name][0]
    def qt(name): return GGMLQuantizationType(weights[name][1])
    def dq(name): return dequantize_gguf_data(get(name), qt(name)).astype(np.float32)

    token_embd_f32 = dq("token_embd.weight")

    # Build chat-formatted prompt
    prompt = build_chat_prompt(tok, args.prompt)

    print(f"Prompt: {repr(tok.decode(prompt))}")
    print(f"Prompt tokens: {len(prompt)}")

    # Use raw Q6_K shared_head weight (398MB vs 2034MB F32 dequant)
    sh_raw = np.asarray(get("output.weight"), dtype=np.uint8)
    sh_qtype = qt("output.weight")
    print(f"Raw shared_head ({sh_qtype.name}): {sh_raw.nbytes/1e6:.0f}MB")

    # Build GPU kernel args
    def run_draft(hidden_seed, token_embed, *, return_hidden_seed: bool = False):
        gpu_args = [
            hidden_seed, token_embed,
            get("blk.40.nextn.eh_proj.weight"), get("blk.40.nextn.hnorm.weight"),
            get("blk.40.nextn.enorm.weight"), get("blk.40.attn_norm.weight"),
            get("blk.40.attn_q.weight"), get("blk.40.attn_k.weight"),
            get("blk.40.attn_v.weight"), get("blk.40.attn_output.weight"),
            get("blk.40.attn_q_norm.weight"), get("blk.40.attn_k_norm.weight"),
            get("blk.40.post_attention_norm.weight"), get("blk.40.ffn_gate_inp.weight"),
            get("blk.40.ffn_gate_exps.weight"), get("blk.40.ffn_up_exps.weight"),
            get("blk.40.ffn_down_exps.weight"),
            qt("blk.40.ffn_gate_exps.weight"), qt("blk.40.ffn_up_exps.weight"), qt("blk.40.ffn_down_exps.weight"),
            get("blk.40.ffn_gate_inp_shexp.weight"),
            get("blk.40.ffn_gate_shexp.weight"), get("blk.40.ffn_up_shexp.weight"),
            get("blk.40.ffn_down_shexp.weight"), qt("blk.40.ffn_gate_shexp.weight"),
            get("blk.40.nextn.shared_head_norm.weight"), sh_raw,
        ]
        gpu_kwargs = dict(
            num_heads=16, num_kv_heads=2, experts_used=8,
            eh_proj_qtype=qt("blk.40.nextn.eh_proj.weight"),
            wq_qtype=qt("blk.40.attn_q.weight"), wk_qtype=qt("blk.40.attn_k.weight"),
            wv_qtype=qt("blk.40.attn_v.weight"), wo_qtype=qt("blk.40.attn_output.weight"),
            eps=1e-6,
            shared_head_qtype=sh_qtype,
            return_hidden_seed=return_hidden_seed,
        )
        result = gpu_kernel(*gpu_args, **gpu_kwargs)
        if return_hidden_seed:
            logits, next_hidden_seed = result
            return (
                np.asarray(logits, dtype=np.float32),
                np.ascontiguousarray(next_hidden_seed, dtype=np.float32),
            )
        return np.asarray(result, dtype=np.float32)

    # Run benchmark
    session = Qwen35GGUFResidentSession(model_path=args.model)
    try:
        prefill_result = session.prefill(prompt, return_logits=False,
                                         capture_hidden_seed_fp32=True)
        prev_token = int(prefill_result.token_id)
        runtime = session.runtime or get_hip_runtime()
        hidden_size = 2048

        def copy_pending_hidden_seed() -> np.ndarray:
            hidden_seed = np.empty((1, hidden_size), dtype=np.float32)
            runtime.memcpy(
                hidden_seed.ctypes.data,
                session.fp32_hidden_seed_ptr(),
                hidden_size * 4,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
            return hidden_seed

        # llama.cpp carries the post-output_norm hidden row from the position
        # before ``prev_token`` and pairs it with ``prev_token`` in draft().
        # Prefill with capture_hidden_seed_fp32=True populated that row for the
        # final prompt token.
        pending_hidden_seed = copy_pending_hidden_seed()

        total_drafts = 0
        total_accepted = 0
        total_output_tokens = 0
        cycle_details = []
        decode_times = []

        for cycle in range(args.cycles):
            # MTP draft(s). The seed row is the pending target hidden row from
            # the previous accepted/corrective boundary, not the hidden row we
            # are about to verify. B2 chains the MTP post-FFN hidden row returned
            # by depth 1 and uses the depth-1 draft token embedding as the
            # depth-2 token row.
            t2 = time.perf_counter()
            draft_tokens = []
            draft_top10_tokens = []
            current_hidden_seed = pending_hidden_seed
            current_token = prev_token
            for draft_depth in range(args.draft_n_max):
                token_embed = token_embd_f32[current_token:current_token+1].copy()
                need_next_seed = draft_depth + 1 < args.draft_n_max
                if need_next_seed:
                    draft_logits, current_hidden_seed = run_draft(
                        current_hidden_seed, token_embed, return_hidden_seed=True
                    )
                else:
                    draft_logits = run_draft(current_hidden_seed, token_embed)
                # Top-k=10 greedy selection (llama.cpp contract)
                draft_token, top10_tokens = select_topk_tokens(draft_logits[0], k=10)
                draft_tokens.append(draft_token)
                draft_top10_tokens.append(top10_tokens)
                current_token = draft_token
            t3 = time.perf_counter()
            draft_ms = (t3 - t2) * 1000

            # Verify/account with llama.cpp semantics. The target evaluates the
            # sampled token plus accepted draft prefix and returns one final
            # corrective target token. Output tokens are therefore
            # accepted_drafts + 1, and accept(n) re-seeds pending_hidden_seed from
            # the hidden row at the accepted-prefix boundary.
            ar_decode_ms = 0.0
            target_tokens = []
            verify_input_token = prev_token
            while True:
                t0 = time.perf_counter()
                target_result = session.step(verify_input_token, capture_hidden_seed_fp32=True)
                t1 = time.perf_counter()
                ar_decode_ms += (t1 - t0) * 1000
                target_token = int(target_result.token_id)
                target_tokens.append(target_token)
                pending_hidden_seed = copy_pending_hidden_seed()

                depth = len(target_tokens) - 1
                if depth < len(draft_tokens) and target_token == draft_tokens[depth]:
                    verify_input_token = target_token
                    continue
                break

            acceptance = llama_cpp_acceptance_from_target_samples(draft_tokens, target_tokens)
            accepted_draft_tokens = int(acceptance["accepted_draft_tokens"])
            output_tokens = list(acceptance["output_tokens"])
            comparison_target_tokens = list(acceptance["comparison_target_tokens"])
            prev_token = int(output_tokens[-1])

            total_drafts += len(draft_tokens)
            visible_output_tokens = int(acceptance["visible_output_tokens"])
            total_output_tokens += visible_output_tokens
            total_accepted += accepted_draft_tokens
            accepted = accepted_draft_tokens == len(draft_tokens)

            target_in_draft_top10 = []
            target_rank_in_draft_top10 = []
            for depth, target in enumerate(comparison_target_tokens):
                top10 = draft_top10_tokens[depth]
                if target in top10:
                    target_in_draft_top10.append(True)
                    target_rank_in_draft_top10.append(top10.index(target) + 1)
                else:
                    target_in_draft_top10.append(False)
                    target_rank_in_draft_top10.append(None)

            cycle_details.append({
                "cycle": cycle,
                "target_token": target_tokens[0],
                "target_tokens": target_tokens,
                "comparison_target_tokens": comparison_target_tokens,
                "output_tokens": output_tokens,
                "draft_token": draft_tokens[0],
                "draft_tokens": draft_tokens,
                "draft_top10_tokens": draft_top10_tokens,
                "target_in_draft_top10": target_in_draft_top10,
                "target_rank_in_draft_top10": target_rank_in_draft_top10,
                "accepted": accepted,
                "generated_draft_tokens": len(draft_tokens),
                "accepted_draft_tokens": accepted_draft_tokens,
                "visible_output_tokens": visible_output_tokens,
                "pending_hidden_row_index": acceptance["pending_hidden_row_index"],
                "ar_decode_ms": round(ar_decode_ms, 2),
                "mtp_draft_ms": round(draft_ms, 2),
            })
            decode_times.append(ar_decode_ms + draft_ms)

    finally:
        session.close()

    # Compute metrics
    metrics = compute_speculative_metrics(cycle_details)
    warm_metrics = compute_speculative_metrics(cycle_details[1:]) if len(cycle_details) > 1 else None
    accept_per_draft = metrics["accept_per_draft"]
    accepted_per_output = metrics["accepted_per_output"]
    avg_decode_ms = metrics["avg_decode_ms"]
    avg_ms_per_visible_token = metrics["avg_ms_per_visible_token"]
    tokens_per_sec = metrics["tokens_per_sec"]
    speedup_vs_ar_visible = metrics["speedup_vs_ar_visible"]
    visible_tokens_per_cycle = metrics["visible_tokens_per_cycle"]

    # Print summary
    print(f"\n{'='*60}")
    print(f"MTP Speculative Decoding Benchmark")
    print(f"{'='*60}")
    print(f"Model: {Path(args.model).name}")
    print(f"Cycles: {args.cycles}")
    print(f"Draft n_max: {args.draft_n_max}")
    print(f"accept_per_draft: {accept_per_draft:.3f}")
    print(f"accepted_per_output: {accepted_per_output:.3f}")
    print(f"visible_tokens_per_cycle: {visible_tokens_per_cycle:.3f}")
    print(f"avg_cycle_ms: {avg_decode_ms:.2f}")
    print(f"avg_ms_per_visible_token: {avg_ms_per_visible_token:.2f}")
    print(f"tokens_per_sec: {tokens_per_sec:.2f}")
    print(f"speedup_vs_ar_visible: {speedup_vs_ar_visible:.3f}x")
    if warm_metrics is not None:
        print(
            "warm_excluding_cycle0: "
            f"avg_ms_per_visible_token={warm_metrics['avg_ms_per_visible_token']:.2f} "
            f"tokens_per_sec={warm_metrics['tokens_per_sec']:.2f} "
            f"speedup_vs_ar_visible={warm_metrics['speedup_vs_ar_visible']:.3f}x"
        )
    print(f"total_accepted: {total_accepted}/{total_drafts}")
    for d in cycle_details:
        acc = f"{d['accepted_draft_tokens']}/{d['generated_draft_tokens']}"
        print(f"  Cycle {d['cycle']}: accepted={acc} targets={d['target_tokens']} drafts={d['draft_tokens']} "
              f"visible={d['visible_output_tokens']} ar={d['ar_decode_ms']:.1f}ms draft={d['mtp_draft_ms']:.1f}ms")

    # Save benchmark artifact
    hw = _get_hw_info()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    result = {
        "schema": 4,
        "status": "ok",
        "timestamp": timestamp,
        "run_tag": f"mtp-bench-{int(time.time())}",
        "hardware": hw,
        "workload": {
            "model": Path(args.model).name,
            "model_path": args.model,
            "quant": "Q4_K_M",
            "prompt": args.prompt,
            "prompt_tokens": len(prompt),
            "cycles": args.cycles,
            "draft_n_max": args.draft_n_max,
            "engine": "hipEngine GGUF MTP",
        },
        "metrics": {
            "accept_per_draft": round(float(metrics["accept_per_draft"]), 4),
            "accepted_per_output": round(float(metrics["accepted_per_output"]), 4),
            "visible_tokens_per_cycle": round(float(metrics["visible_tokens_per_cycle"]), 4),
            "avg_cycle_ms": round(float(metrics["avg_cycle_ms"]), 2),
            "avg_decode_ms": round(float(metrics["avg_decode_ms"]), 2),
            "avg_ar_decode_ms": round(float(metrics["avg_ar_decode_ms"]), 2),
            "avg_mtp_draft_ms": round(float(metrics["avg_mtp_draft_ms"]), 2),
            "avg_ms_per_visible_token": round(float(metrics["avg_ms_per_visible_token"]), 2),
            "tokens_per_sec": round(float(metrics["tokens_per_sec"]), 2),
            "ar_baseline_tokens_per_sec": round(float(metrics["ar_baseline_tokens_per_sec"]), 2),
            "speedup_vs_ar_visible": round(float(metrics["speedup_vs_ar_visible"]), 4),
            "total_accepted": metrics["total_accepted"],
            "total_drafts": metrics["total_drafts"],
            "total_output_tokens": metrics["total_output_tokens"],
            "verify_cycle_count": metrics["verify_cycle_count"],
            "denominators": metrics["denominators"],
            "warm_excluding_cycle0": (
                {
                    "accept_per_draft": round(float(warm_metrics["accept_per_draft"]), 4),
                    "accepted_per_output": round(float(warm_metrics["accepted_per_output"]), 4),
                    "visible_tokens_per_cycle": round(float(warm_metrics["visible_tokens_per_cycle"]), 4),
                    "avg_cycle_ms": round(float(warm_metrics["avg_cycle_ms"]), 2),
                    "avg_ar_decode_ms": round(float(warm_metrics["avg_ar_decode_ms"]), 2),
                    "avg_mtp_draft_ms": round(float(warm_metrics["avg_mtp_draft_ms"]), 2),
                    "avg_ms_per_visible_token": round(float(warm_metrics["avg_ms_per_visible_token"]), 2),
                    "tokens_per_sec": round(float(warm_metrics["tokens_per_sec"]), 2),
                    "speedup_vs_ar_visible": round(float(warm_metrics["speedup_vs_ar_visible"]), 4),
                    "total_accepted": warm_metrics["total_accepted"],
                    "total_drafts": warm_metrics["total_drafts"],
                    "total_output_tokens": warm_metrics["total_output_tokens"],
                    "verify_cycle_count": warm_metrics["verify_cycle_count"],
                }
                if warm_metrics is not None else None
            ),
        },
        "cycles": cycle_details,
    }

    out_path = args.output or f"benchmarks/results/mtp-bench-{int(time.time())}.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nBenchmark saved to: {out_path}")


if __name__ == "__main__":
    main()