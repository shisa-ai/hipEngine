#!/usr/bin/env python3
"""Long-context retrieval gate over the served HTTP route.

The committed fixture (``benchmarks/prompts/qwen4exp-long-retrieval-v1.jsonl``)
covers 8K and 16K. Serving qualification also needs a deeper point, so this
gate rebuilds the same seeded retrieval prompts at any target depth with the
committed builder and scores the served answer against the recorded facts.

Each prompt stores four answers at about 10%, 40%, 70%, and 95% depth, so a
pass means the model read the whole context rather than the head or tail.
Scoring is exact per question: the response must print ``N: <number>`` with the
recorded count. Nothing is compared against another engine's output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_ROOT = Path("/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL")


def _answers(text: str, expected: int) -> list[int | None]:
    """Read ``N: <number>`` lines in question order; None when a line is absent."""

    found: dict[int, int] = {}
    for line in text.splitlines():
        line = line.strip().strip("`")
        if ":" not in line:
            continue
        head, _, tail = line.partition(":")
        head = head.strip()
        if not head.isdigit():
            continue
        number = tail.strip().split()[0] if tail.strip() else ""
        number = number.strip(".,;:。、")
        if number.isdigit():
            found.setdefault(int(head), int(number))
    return [found.get(index + 1) for index in range(expected)]


def _build_cases(model_root: Path, targets: Sequence[int], langs: Sequence[str]) -> list[dict[str, Any]]:
    from scripts.build_qwen4exp_long_retrieval_suite import build_prompt
    from hipengine.loading.gguf import discover_gguf_files, scan_gguf
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    info = scan_gguf(discover_gguf_files(model_root)[0])
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(info)
    cases: list[dict[str, Any]] = []
    seed = 20261014
    for target in targets:
        for lang in langs:
            seed += 1
            cases.append(build_prompt(tokenizer, lang, target, seed))
    return cases


def _post(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    import httpx

    started = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{url.rstrip('/')}/v1/chat/completions", json=payload)
    elapsed = time.perf_counter() - started
    body = response.json()
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {json.dumps(body)[:400]}")
    return body, elapsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18395")
    parser.add_argument("--model", default="qwen38fn-capon")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--targets", default="8192,16384,65536")
    parser.add_argument("--langs", default="en,ja")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    targets = [int(value) for value in args.targets.split(",") if value.strip()]
    langs = [value.strip() for value in args.langs.split(",") if value.strip()]
    cases = _build_cases(args.model_root, targets, langs)

    results: list[dict[str, Any]] = []
    failures = 0
    for case in cases:
        payload = {
            "model": args.model,
            "messages": case["messages"],
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
        }
        try:
            body, elapsed = _post(args.url, payload, args.timeout)
        except Exception as exc:  # a served failure is a gate result, not a crash
            failures += 1
            results.append(
                {
                    "id": case["id"],
                    "passed": False,
                    "error": str(exc),
                    "target_chat_tokens": case["chat_tokens"],
                }
            )
            print(f"{case['id']}: ERROR {exc}")
            continue
        choice = (body.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        expected = list(case["answers"])
        got = _answers(text, len(expected))
        correct = sum(1 for want, have in zip(expected, got, strict=True) if want == have)
        usage = body.get("usage") or {}
        passed = correct == len(expected)
        failures += 0 if passed else 1
        results.append(
            {
                "id": case["id"],
                "passed": passed,
                "target_chat_tokens": case["chat_tokens"],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "expected": expected,
                "got": got,
                "correct": correct,
                "questions": len(expected),
                "elapsed_s": round(elapsed, 3),
                "prefill_tokens_per_s": (
                    None
                    if not usage.get("prompt_tokens") or elapsed <= 0
                    else round(int(usage["prompt_tokens"]) / elapsed, 1)
                ),
                "finish_reason": choice.get("finish_reason"),
                "finish_details": choice.get("finish_details"),
                "text": text[:400],
            }
        )
        print(
            f"{case['id']}: {'PASS' if passed else 'FAIL'} "
            f"{correct}/{len(expected)} prompt_tokens={usage.get('prompt_tokens')} "
            f"elapsed={elapsed:.1f}s"
        )

    summary = {
        "schema": "hipengine.qwen4exp.http_long_context_gate.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": args.url,
        "model": args.model,
        "targets": targets,
        "langs": langs,
        "max_tokens": args.max_tokens,
        "cases": results,
        "case_count": len(results),
        "failure_count": failures,
        "status": "passed" if failures == 0 else "failed",
    }
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "cases": len(results), "failures": failures}))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
