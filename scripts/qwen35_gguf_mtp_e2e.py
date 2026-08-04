#!/usr/bin/env python3
"""Benchmark a real trailing GGUF NextN block through the shared transactional verifier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Sequence

from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.loading import load_gguf_index
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFMTPDecodeSession
from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNDraftProvider
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf")
DEFAULT_PROMPT = (
    "Write a Python function that returns the n-th Fibonacci number using "
    "memoization. Include a docstring."
)


def _parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not budgets or any(item not in {1, 2, 3} for item in budgets):
        raise argparse.ArgumentTypeError("candidate budgets must be a comma-separated subset of 1,2,3")
    if len(set(budgets)) != len(budgets):
        raise argparse.ArgumentTypeError("candidate budgets must not contain duplicates")
    return budgets


def _parse_tokens(value: str) -> tuple[int, ...]:
    tokens = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not tokens or any(token < 0 for token in tokens):
        raise argparse.ArgumentTypeError("prompt tokens must be non-negative comma-separated integers")
    return tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--quant", default="gguf_ud_q3_k_m")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-tokens", type=_parse_tokens)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--candidate-budgets", type=_parse_budgets, default=(1, 2, 3))
    parser.add_argument(
        "--target-verify-mode",
        choices=("serial-exact", "native"),
        default="serial-exact",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=0)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--return-cycle-logits", action="store_true")
    parser.add_argument("--require-cached-build", action="store_true")
    return parser


def _ar_generate(
    target: Qwen35GGUFResidentSession,
    prompt: Sequence[int],
    *,
    max_new_tokens: int,
) -> dict[str, object]:
    prefill_started = time.perf_counter()
    root = int(target.prefill(prompt, use_bulk=False, return_logits=False).token_id)
    prefill_seconds = time.perf_counter() - prefill_started
    generated = [root]
    decode_started = time.perf_counter()
    while len(generated) < int(max_new_tokens):
        generated.append(int(target.step(generated[-1], return_logits=False).token_id))
    decode_seconds = time.perf_counter() - decode_started
    return {
        "token_ids": generated,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tok_s": len(generated) / decode_seconds,
    }


def _median_row(rows: Sequence[dict[str, object]], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _model_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    info = load_gguf_index(path)
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "architecture": info.architecture,
        "tensor_count": int(info.tensor_count),
        "file_type": info.file_type_name,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if args.runs <= 0:
        raise ValueError("runs must be positive")
    model = args.model.resolve()
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(model))
    prompt_tokens = (
        tuple(args.prompt_tokens)
        if args.prompt_tokens is not None
        else tuple(int(token) for token in tokenizer.encode(str(args.prompt)))
    )
    if not prompt_tokens:
        raise ValueError("prompt tokenization produced no tokens")
    max_sequence = int(args.max_sequence_length) or (
        len(prompt_tokens) + int(args.max_new_tokens) + max(args.candidate_budgets) + 8
    )
    reset_memory_stats()
    rows_by_budget: dict[str, list[dict[str, object]]] = {
        str(budget): [] for budget in args.candidate_budgets
    }
    with Qwen35GGUFResidentSession(
        model,
        max_sequence_length=max_sequence,
        require_cached_build=bool(args.require_cached_build),
    ) as target:
        target.select_prefill_quant(str(args.quant))
        if target.runner.weights is None:
            raise RuntimeError("target GGUF weights are unavailable")
        borrowed_fallback_weights = {
            slot: target.runner.weights.root(slot)
            for slot in ("token_embedding", "lm_head")
        }
        provider = Qwen35GGUFNextNDraftProvider.from_model(
            model,
            max_positions=max_sequence,
            max_requests=1,
            runtime=target.runtime,
            require_cached_build=bool(args.require_cached_build),
            borrowed_fallback_weights=borrowed_fallback_weights,
        )
        try:
            for run_index in range(int(args.runs)):
                for budget in args.candidate_budgets:
                    ar = _ar_generate(
                        target,
                        prompt_tokens,
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    with Qwen35GGUFMTPDecodeSession(
                        target,
                        provider,
                        candidate_budget=int(budget),
                        quant=str(args.quant),
                        target_verify_mode=str(args.target_verify_mode),
                    ) as decoder:
                        mtp = decoder.generate(
                            prompt_tokens,
                            max_new_tokens=int(args.max_new_tokens),
                            return_cycle_logits=bool(args.return_cycle_logits),
                            use_bulk_prefill=False,
                            prefill_draft=True,
                        )
                    exact = tuple(int(token) for token in ar["token_ids"]) == mtp.token_ids
                    mtp_payload = mtp.to_json_dict()
                    rows_by_budget[str(budget)].append(
                        {
                            "run": run_index,
                            "ar": ar,
                            "mtp": mtp_payload,
                            "exact_greedy_match": exact,
                            "speedup_vs_ar": float(mtp_payload["decode_tok_s"])
                            / float(ar["decode_tok_s"]),
                        }
                    )
        finally:
            provider.close()

    summaries: dict[str, dict[str, object]] = {}
    any_non_regressive = False
    all_exact = True
    for budget, rows in rows_by_budget.items():
        exact = all(bool(row["exact_greedy_match"]) for row in rows)
        all_exact = all_exact and exact
        speedup = _median_row(rows, "speedup_vs_ar")
        mtp_tok_s = float(
            statistics.median(float(row["mtp"]["decode_tok_s"]) for row in rows)  # type: ignore[index]
        )
        ar_tok_s = float(
            statistics.median(float(row["ar"]["decode_tok_s"]) for row in rows)  # type: ignore[index]
        )
        accepted = float(
            statistics.mean(float(row["mtp"]["accepted_draft_tokens"]) for row in rows)  # type: ignore[index]
        )
        cycles = float(
            statistics.mean(float(row["mtp"]["cycles"]) for row in rows)  # type: ignore[index]
        )
        visible = 1.0 + accepted / cycles if cycles > 0 else 1.0
        non_regressive = bool(exact and speedup >= 1.0)
        any_non_regressive = any_non_regressive or non_regressive
        summaries[budget] = {
            "runs": len(rows),
            "exact_greedy_match": exact,
            "gpu_accept_match_cpu": all(
                bool(row["mtp"]["gpu_accept_match_cpu"]) for row in rows  # type: ignore[index]
            ),
            "ar_decode_tok_s_median": ar_tok_s,
            "mtp_decode_tok_s_median": mtp_tok_s,
            "speedup_vs_ar_median": speedup,
            "accepted_draft_tokens_mean": accepted,
            "cycles_mean": cycles,
            "visible_tokens_per_cycle": visible,
            "non_regressive": non_regressive,
        }

    status = "accepted" if all_exact and any_non_regressive else "diagnostic_nohold"
    decision = (
        "retain explicit GGUF MTP route"
        if status == "accepted"
        else "keep GGUF MTP disabled by default; exact integration is economics-negative"
    )
    return {
        "schema": 1,
        "kind": "qwen35_gguf_mtp_e2e",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "performance_claim": status == "accepted",
        "decision": decision,
        "hardware": {
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "hip_arch": os.environ.get("HIPENGINE_HIP_ARCH"),
            "default_hardware_note": "device identity is recorded by the exact command and WORKLOG evidence",
        },
        "software": {
            "hipengine_commit": _git_output("rev-parse", "HEAD"),
            "hipengine_dirty": bool(_git_output("status", "--porcelain")),
            "script": "scripts/qwen35_gguf_mtp_e2e.py",
        },
        "model": _model_metadata(model),
        "workload": {
            "prompt": None if args.prompt_tokens is not None else str(args.prompt),
            "prompt_tokens": list(prompt_tokens),
            "prompt_length": len(prompt_tokens),
            "max_new_tokens": int(args.max_new_tokens),
            "candidate_budgets": list(args.candidate_budgets),
            "target_verify_mode": str(args.target_verify_mode),
            "runs": int(args.runs),
            "max_sequence_length": max_sequence,
            "quant": str(args.quant),
            "sampling": "exact greedy raw target top-1",
        },
        "contract": {
            "draft_batch": "candidate-only DraftBatch",
            "target_verify_batch": "root+candidate TargetVerifyBatch",
            "accept_result": "GPU TargetAcceptSummary validated against CPU AcceptResult oracle",
            "kv_spans": "KVLiveSpans(span_role=verify_chain)",
            "commit": "scheduler-owned KVTransaction plus target state/KV append journal",
            "graph_buckets": "BatchShapeKey scheduler buckets with stable TargetVerifyBuffers; eager target replay",
        },
        "summary": summaries,
        "runs": rows_by_budget,
        "memory": memory_stats(),
    }


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "decision": payload["decision"], "summary": payload["summary"]}, indent=2))
    return 0 if all(row["exact_greedy_match"] for row in payload["summary"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
