#!/usr/bin/env python3
"""Canonical SPECDEC2 C2/C4 strict category and true-AR gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

from hipengine import LLM
from hipengine.generation.registry import GenerationRequest
from hipengine.server import render_chat_prompt

_HELDOUT_IDS = frozenset(
    {
        "code_markdown_table",
        "general_en_explain",
        "general_ja_explain",
        "mixed_ja_en_review",
    }
)
_REQUIRED_CATEGORIES = frozenset({"code", "general_en", "general_ja", "mixed_ja_en"})


def _git(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_prompt_suite(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        prompt_id = payload.get("id")
        category = payload.get("category")
        messages = payload.get("messages")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"prompt line {line_number} has no strict id")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"prompt {prompt_id} has no strict category")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"prompt {prompt_id} has no messages")
        rendered = render_chat_prompt(messages)
        rows.append(
            {
                "id": prompt_id,
                "category": category,
                "messages": messages,
                "rendered_prompt": rendered,
                "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            }
        )
    ids = tuple(row["id"] for row in rows)
    if len(ids) != len(set(ids)):
        raise ValueError("prompt ids must be unique")
    categories = frozenset(row["category"] for row in rows)
    if categories != _REQUIRED_CATEGORIES:
        raise ValueError("canonical SPECDEC2 categories are incomplete")
    if not _HELDOUT_IDS.issubset(ids):
        raise ValueError("canonical heldout prompts are incomplete")
    return tuple(rows)


def counterbalanced_route_order(prompt_index: int) -> tuple[str, str]:
    """Alternate AR→MTP and MTP→AR without consulting prompt content."""

    return ("ar", "mtp") if int(prompt_index) % 2 == 0 else ("mtp", "ar")


def _request(prompt: str, max_tokens: int) -> GenerationRequest:
    return GenerationRequest(
        prompts=(prompt,),
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
    )


def _run_group(service: Any, request: GenerationRequest, concurrency: int, *, mtp: bool):
    requests = tuple(request for _ in range(int(concurrency)))
    started = time.perf_counter()
    handles = (
        service.submit_speculative_children(requests)
        if mtp
        else service.submit_children(requests)
    )
    outputs = tuple(handle.result(timeout=180) for handle in handles)
    wall = time.perf_counter() - started
    ids = tuple(tuple(int(token) for token in output.generated_token_ids or ()) for output in outputs)
    timings = tuple(dict(output.telemetry.timing or {}) for output in outputs)
    return ids, timings, wall


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompt_suite(args.prompts.resolve())
    os.environ["HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET"] = str(int(args.budget))
    os.environ["HIPENGINE_GGUF_FP16_RECURRENT_STATE"] = "0"
    llm = LLM(
        str(args.model.resolve()),
        backend="hip_gfx1151",
        execution_profile="strict",
        max_active_requests=4,
        max_sequence_length=1024,
    )
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        service = llm._get_text_generator()
        for prompt_index, prompt in enumerate(prompts):
            request = _request(prompt["rendered_prompt"], int(args.max_tokens))
            cells: dict[str, Any] = {}
            all_ids: list[tuple[int, ...]] = []
            for concurrency in (2, 4):
                measured: dict[str, tuple[Any, Any, float]] = {}
                execution_order = counterbalanced_route_order(prompt_index)
                mtp_rows: list[dict[str, Any]] = []
                for route in execution_order:
                    measured[route] = _run_group(
                        service,
                        request,
                        concurrency,
                        mtp=route == "mtp",
                    )
                    if route == "mtp":
                        snapshot = service.live_loop_snapshot()
                        recent = snapshot["runner"]["routes"]["recent_completed"]
                        mtp_rows = [
                            row
                            for row in recent[-concurrency:]
                            if row["specdec2_mtp2_used"]
                        ]
                ar_ids, ar_timings, ar_wall = measured["ar"]
                mtp_ids, mtp_timings, mtp_wall = measured["mtp"]
                exact = bool(
                    len(set(ar_ids)) == 1
                    and len(set(mtp_ids)) == 1
                    and ar_ids[0] == mtp_ids[0]
                    and len(mtp_rows) == concurrency
                )
                all_ids.extend((ar_ids[0], mtp_ids[0]))
                cells[f"c{concurrency}"] = {
                    "concurrency": concurrency,
                    "candidate_budget": int(args.budget),
                    "execution_order": execution_order,
                    "ar_ids": [list(row) for row in ar_ids],
                    "mtp_ids": [list(row) for row in mtp_ids],
                    "exact": exact,
                    "ar_wall_seconds": ar_wall,
                    "mtp_wall_seconds": mtp_wall,
                    "mtp_over_ar_wall_ratio": mtp_wall / ar_wall if ar_wall > 0 else None,
                    "ar_timings": ar_timings,
                    "mtp_timings": mtp_timings,
                    "accepted_counts": [
                        list(row["specdec2_mtp2_accepted_counts"])
                        for row in mtp_rows
                    ],
                    "candidate_counts": [
                        list(row["specdec2_mtp2_candidate_counts"])
                        for row in mtp_rows
                    ],
                    "execution_routes": [
                        list(row["specdec2_mtp2_execution_routes"])
                        for row in mtp_rows
                    ],
                }
            composition_exact = len(set(all_ids)) == 1
            results.append(
                {
                    **{key: prompt[key] for key in ("id", "category", "prompt_sha256")},
                    "heldout": prompt["id"] in _HELDOUT_IDS,
                    "cells": cells,
                    "composition_exact": composition_exact,
                    "passed": composition_exact and all(cell["exact"] for cell in cells.values()),
                }
            )
        final_snapshot = service.live_loop_snapshot()
    finally:
        llm.close()

    categories = sorted(_REQUIRED_CATEGORIES)
    category_passed = {
        category: all(row["passed"] for row in results if row["category"] == category)
        for category in categories
    }
    train_ids = tuple(row["id"] for row in prompts if row["id"] not in _HELDOUT_IDS)
    heldout_ids = tuple(row["id"] for row in prompts if row["id"] in _HELDOUT_IDS)
    full_passed = all(row["passed"] for row in results)
    heldout_passed = all(row["passed"] for row in results if row["heldout"])
    passed = bool(
        full_passed
        and heldout_passed
        and all(category_passed.values())
        and final_snapshot["loop"]["requests"]["active"] == 0
        and final_snapshot["loop"]["requests"]["pending"] == 0
    )
    return {
        "schema": 1,
        "kind": "specdec2_gfx1151_s6_category_gate",
        "status": "passed" if passed else "failed",
        "performance_claim": False,
        "speed_claim_eligible": False,
        "repo": {
            "commit": _git(("rev-parse", "HEAD")),
            "dirty": bool(_git(("status", "--porcelain", "--untracked-files=no"))),
            "shared_untracked_files_excluded": True,
        },
        "host": {
            "hostname": platform.node(),
            "device": "AMD Radeon 8060S Graphics",
            "backend": "hip_gfx1151",
        },
        "model": {
            "path": str(args.model.resolve()),
            "quant": "Q4_K_S",
            "kv": "bf16",
            "execution_profile": "strict",
            "execution_profile_manifest_sha256": llm.execution_profile_manifest_sha256,
        },
        "protocol": {
            "prompt_file": str(args.prompts.resolve()),
            "prompt_file_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
            "prompt_count": len(prompts),
            "categories": categories,
            "train_ids": train_ids,
            "heldout_ids": heldout_ids,
            "candidate_budget": int(args.budget),
            "concurrency": [2, 4],
            "max_tokens": int(args.max_tokens),
            "sampling": "raw_greedy",
            "true_ar_baseline": True,
            "same_prompt_suite": True,
            "same_process": True,
            "counterbalanced_order": "even prompts AR→MTP; odd prompts MTP→AR",
        },
        "results": results,
        "category_passed": category_passed,
        "full_passed": full_passed,
        "heldout_passed": heldout_passed,
        "final_snapshot": final_snapshot,
        "elapsed_seconds": time.perf_counter() - started,
        "passed": passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"),
    )
    parser.add_argument("--budget", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args(argv)
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "budget": args.budget,
                "elapsed_seconds": payload["elapsed_seconds"],
                "passed": payload["passed"],
            },
            sort_keys=True,
        )
    )
    return 1 if args.fail_on_fail and not payload["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
