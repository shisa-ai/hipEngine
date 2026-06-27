#!/usr/bin/env python3
"""Produce true no-MTP AR baselines for the GGUF-MTP category suite.

This is the no-speculation counterpart to ``scripts/gguf_mtp_category_bench.py``.
It runs the resident GGUF autoregressive path on the same chat-formatted prompt
suite and emits the prompt-level ``prompt_metrics[]`` schema that
``gguf_mtp_category_bench.py --true-ar-baseline-json`` accepts.

The artifact is a baseline input for guarded MTP speed comparisons. It does not
claim an MTP win by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import DEFAULT_MODEL, DEFAULT_PROMPTS, BenchError, load_prompt_rows, prompt_sha256, repo_provenance, safe_name, validate_command_provenance


def exact_command_payload(argv: Sequence[object]) -> dict[str, Any]:
    argv_strings = [str(item) for item in argv]
    return {"argv": argv_strings, "command": shlex.join(argv_strings)}


def aggregate_prompt_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output_tokens = sum(int(row["output_tokens"]) for row in rows)
    decode_ms = sum(float(row["decode_ms"]) for row in rows)
    return {
        "prompts": len(rows),
        "total_output_tokens": output_tokens,
        "decode_ms": decode_ms,
        "decode_tok_s_weighted": 1000.0 * output_tokens / decode_ms if decode_ms > 0 else 0.0,
    }


def category_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    return {category: aggregate_prompt_metrics(group_rows) for category, group_rows in sorted(grouped.items())}


def build_true_ar_artifact(
    *,
    args: argparse.Namespace,
    prompts: list[dict[str, Any]],
    prompt_metrics: list[dict[str, Any]],
    commands: list[str],
) -> dict[str, Any]:
    prompt_ids = [str(row["id"]) for row in prompts]
    metric_ids = [str(row["id"]) for row in prompt_metrics]
    if metric_ids != prompt_ids:
        raise BenchError(f"prompt_metrics order/ids must match selected prompts: {metric_ids} != {prompt_ids}")
    for prompt_row, metric_row in zip(prompts, prompt_metrics, strict=True):
        metric_row.setdefault("prompt_chars", len(str(prompt_row["prompt"])))
        metric_row.setdefault("prompt_sha256", prompt_sha256(str(prompt_row["prompt"])))
    commands = validate_command_provenance({"commands": commands}, label="true AR artifact")
    return {
        "schema": 1,
        "kind": "hipengine_gguf_true_ar_category_baseline",
        "status": "complete",
        "performance_claim": False,
        "true_autoregressive_path": True,
        "same_timing_protocol": True,
        "same_prompt_suite": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repo": repo_provenance(),
        "model": str(args.model),
        "quant": "UD-Q4_K_M GGUF",
        "prompt_file": str(args.prompts),
        "decode_tokens": int(args.decode_tokens),
        "warmup_decode_tokens": int(args.warmup_decode_tokens),
        "timing_protocol": {
            "decode_path": "graph_replay" if bool(getattr(args, "graph_replay_decode", False)) else "eager_step",
            "graph_replay_decode": bool(getattr(args, "graph_replay_decode", False)),
            "graph_steps_per_replay": (
                int(getattr(args, "graph_steps_per_replay", 0) or 0)
                if bool(getattr(args, "graph_replay_decode", False))
                else 0
            ),
            "decode_repack": bool(getattr(args, "decode_repack", False)),
            "decode_repack_env": os.environ.get("HIPENGINE_GGUF_DECODE_REPACK"),
            "use_gemv_decode": bool(getattr(args, "use_gemv_decode", False)),
            "use_wmma_prefill": bool(getattr(args, "use_wmma_prefill", False)),
            "force_bulk_prefill": bool(getattr(args, "force_bulk_prefill", False)),
            "no_bulk_prefill": bool(getattr(args, "no_bulk_prefill", False)),
            "bulk_prefill_attention_mode": str(getattr(args, "bulk_prefill_attention_mode", "bulk")),
            "attn_aotriton_min_tokens": int(getattr(args, "attn_aotriton_min_tokens", 0) or 0),
        },
        "prompt_count": len(prompts),
        "prompt_ids": prompt_ids,
        "prompt_hashes": {str(row["id"]): prompt_sha256(str(row["prompt"])) for row in prompts},
        "commands": commands,
        "totals": aggregate_prompt_metrics(prompt_metrics),
        "categories": category_metrics(prompt_metrics),
        "prompt_metrics": prompt_metrics,
        "notes": [
            "True no-MTP autoregressive resident GGUF path; no draft/proposal/MTP kernels are invoked.",
            "Prompt tokens use scripts.gguf_mtp_bench.build_chat_prompt() so the prompt suite matches GGUF-MTP diagnostics.",
            "Default timing mirrors the retained production GGUF benchmark path: GEMV decode enabled and measured decode via HIP graph replay with graph capture excluded.",
            "decode_ms measures the autoregressive decode loop after prefill and optional warmup; model load, prefill, warmup, and graph capture are excluded.",
            "Use as --true-ar-baseline-json input for scripts/gguf_mtp_category_bench.py.",
        ],
    }


def run_prompt_true_ar(
    *,
    session: Any,
    tokenizer: Any,
    prompt_row: dict[str, Any],
    decode_tokens: int,
    warmup_decode_tokens: int,
    use_bulk_prefill: bool | None,
    bulk_attention_mode: str,
    graph_replay_decode: bool,
    graph_steps_per_replay: int,
) -> dict[str, Any]:
    prompt_tokens = build_chat_prompt(tokenizer, str(prompt_row["prompt"]))
    session.reset()
    prefill_start = time.perf_counter()
    first = session.prefill(
        prompt_tokens,
        use_bulk=use_bulk_prefill,
        bulk_attention_mode=bulk_attention_mode,
        return_logits=False,
    )
    prefill_ms = 1000.0 * (time.perf_counter() - prefill_start)
    next_token = int(first.token_id)
    generated: list[int] = [next_token]

    warmup_start = time.perf_counter()
    for _ in range(int(warmup_decode_tokens)):
        warmup = session.step(next_token, return_logits=False)
        next_token = int(warmup.token_id)
        generated.append(next_token)
    warmup_ms = 1000.0 * (time.perf_counter() - warmup_start)

    final = None
    graph_capture_ms = 0.0
    if graph_replay_decode:
        graph = None
        capture_start = time.perf_counter()
        try:
            graph = session.capture_decode_graph(
                position=session.position,
                steps_per_replay=int(graph_steps_per_replay),
                max_replay_steps=int(decode_tokens),
                record_steps=int(decode_tokens),
            )
            graph_capture_ms = 1000.0 * (time.perf_counter() - capture_start)
            decode_start = time.perf_counter()
            graph.replay(int(decode_tokens))
            decode_ms = 1000.0 * (time.perf_counter() - decode_start)
            generated.extend(int(token) for token in graph.read_generated_token_ids(int(decode_tokens)))
            final = graph.read_sample()
            if final is not None:
                next_token = int(final.token_id)
        finally:
            if graph is not None:
                graph.close()
    else:
        decode_start = time.perf_counter()
        for step_index in range(int(decode_tokens)):
            final = session.step(next_token, return_logits=(step_index == int(decode_tokens) - 1))
            next_token = int(final.token_id)
            generated.append(next_token)
        decode_ms = 1000.0 * (time.perf_counter() - decode_start)
    finite_logits = None if final is None else bool(np.all(np.isfinite(final.logits)))

    return {
        "id": str(prompt_row["id"]),
        "category": str(prompt_row["category"]),
        "prompt_chars": len(str(prompt_row["prompt"])),
        "prompt_sha256": prompt_sha256(str(prompt_row["prompt"])),
        "prompt_tokens": len(prompt_tokens),
        "output_tokens": int(decode_tokens),
        "decode_ms": decode_ms,
        "decode_tok_s": 1000.0 * int(decode_tokens) / decode_ms if decode_ms > 0 else 0.0,
        "prefill_ms": prefill_ms,
        "warmup_decode_ms": warmup_ms,
        "warmup_decode_tokens": int(warmup_decode_tokens),
        "graph_replay_decode": bool(graph_replay_decode),
        "graph_steps_per_replay": int(graph_steps_per_replay if graph_replay_decode else 0),
        "graph_capture_ms_excluded": graph_capture_ms,
        "finite_final_logits": finite_logits,
        "final_token_id": None if final is None else int(final.token_id),
        "generated_preview_token_ids": generated[:16],
        "generated_tail_token_ids": generated[-16:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup-decode-tokens", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--force-bulk-prefill", action="store_true")
    parser.add_argument("--no-bulk-prefill", action="store_true")
    parser.add_argument("--bulk-prefill-attention-mode", choices=("bulk", "native"), default="bulk")
    parser.add_argument("--graph-replay-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-steps-per-replay", type=int, default=1)
    parser.add_argument("--decode-repack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-wmma-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gemv-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-aotriton-min-tokens", type=int, default=512)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.decode_tokens <= 0:
        raise BenchError("--decode-tokens must be positive")
    if args.warmup_decode_tokens < 0:
        raise BenchError("--warmup-decode-tokens must be non-negative")
    if args.graph_steps_per_replay <= 0:
        raise BenchError("--graph-steps-per-replay must be positive")
    if args.graph_replay_decode and args.decode_tokens % args.graph_steps_per_replay != 0:
        raise BenchError("--decode-tokens must be divisible by --graph-steps-per-replay")
    if args.attn_aotriton_min_tokens < 0:
        raise BenchError("--attn-aotriton-min-tokens must be non-negative")
    if args.force_bulk_prefill and args.no_bulk_prefill:
        raise BenchError("--force-bulk-prefill and --no-bulk-prefill are mutually exclusive")
    if not args.model.exists():
        raise BenchError(f"model not found: {args.model}")
    prompts = load_prompt_rows(args.prompts)
    if args.limit is not None:
        prompts = prompts[: max(0, int(args.limit))]
    if not prompts:
        raise BenchError("selected prompt list is empty")

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.raw_root is None:
        args.raw_root = Path(f"/tmp/hipengine-gguf-true-ar-category-{run_tag}")
    if args.output is None:
        args.output = args.raw_root / "true-ar-baseline.json"
    args.raw_root.mkdir(parents=True, exist_ok=True)

    commands = [exact_command_payload(sys.argv)["command"]]
    if args.dry_run:
        prompt_metrics = [
            {
                "id": row["id"],
                "category": row["category"],
                "prompt_chars": len(row["prompt"]),
                "prompt_sha256": prompt_sha256(row["prompt"]),
                "prompt_tokens": None,
                "output_tokens": int(args.decode_tokens),
                "decode_ms": 1.0,
                "decode_tok_s": 1000.0 * int(args.decode_tokens),
                "warmup_decode_tokens": int(args.warmup_decode_tokens),
                "graph_replay_decode": bool(args.graph_replay_decode),
                "graph_steps_per_replay": int(args.graph_steps_per_replay if args.graph_replay_decode else 0),
                "decode_repack": bool(args.decode_repack),
                "graph_capture_ms_excluded": 0.0,
                "finite_final_logits": True,
                "dry_run": True,
            }
            for row in prompts
        ]
        artifact = build_true_ar_artifact(args=args, prompts=prompts, prompt_metrics=prompt_metrics, commands=commands)
        artifact["status"] = "dry_run"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(args.output)
        return 0

    if args.decode_repack:
        os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    else:
        os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "0"

    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8")
    gguf_info = scan_gguf(args.model)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(gguf_info)
    max_prompt_tokens = max(len(build_chat_prompt(tokenizer, row["prompt"])) for row in prompts)
    max_sequence_length = max_prompt_tokens + int(args.warmup_decode_tokens) + int(args.decode_tokens) + 1
    if args.force_bulk_prefill:
        use_bulk_prefill = True
    elif args.no_bulk_prefill:
        use_bulk_prefill = False
    else:
        use_bulk_prefill = None

    prefill_config = PrefillConfig(attn_aotriton_min_tokens=int(args.attn_aotriton_min_tokens))
    session = Qwen35GGUFResidentSession(
        args.model,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=bool(args.use_wmma_prefill),
        use_gemv_decode=bool(args.use_gemv_decode),
        prefill_config=prefill_config,
    )
    session_timing_protocol = {
        "effective_decode_repack": bool(args.decode_repack),
        "effective_use_wmma_prefill": bool(session.use_wmma_prefill),
        "effective_use_gemv_decode": bool(session.use_gemv_decode),
        "fastpath_safety": None if session.fastpath_safety is None else session.fastpath_safety.as_dict(),
    }
    prompt_metrics: list[dict[str, Any]] = []
    try:
        for row in prompts:
            metric = run_prompt_true_ar(
                session=session,
                tokenizer=tokenizer,
                prompt_row=row,
                decode_tokens=int(args.decode_tokens),
                warmup_decode_tokens=int(args.warmup_decode_tokens),
                use_bulk_prefill=use_bulk_prefill,
                bulk_attention_mode=str(args.bulk_prefill_attention_mode),
                graph_replay_decode=bool(args.graph_replay_decode),
                graph_steps_per_replay=int(args.graph_steps_per_replay),
            )
            prompt_metrics.append(metric)
            per_prompt_path = args.raw_root / f"{safe_name(row['id'])}.json"
            per_prompt_path.write_text(json.dumps(metric, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"AR {row['id']}: {per_prompt_path}", flush=True)
    finally:
        session.close()

    artifact = build_true_ar_artifact(args=args, prompts=prompts, prompt_metrics=prompt_metrics, commands=commands)
    artifact["timing_protocol"].update(session_timing_protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
