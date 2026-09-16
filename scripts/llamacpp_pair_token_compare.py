#!/usr/bin/env python3
"""Compare generated token sequences between two llama.cpp-family servers.

A comparator whose source changed needs a correctness check before its rate is
used as a target. One prefill token is not enough: the two builds here differ in
kernel arithmetic, so the question is whether the difference stays below the
level that changes what the model actually emits over a run of tokens.

This starts each server in turn, sends the same exact token ids with the same
sampling settings, and reports the first position where the sequences diverge
along with the agreement rate. Greedy sampling (temperature 0) is used so the
comparison is deterministic and any divergence is a real arithmetic difference
rather than sampling noise.

Example:
    python3 scripts/llamacpp_pair_token_compare.py \
        --server-a <base>/bin/llama-server --label-a 69946438a \
        --server-b <pr>/bin/llama-server   --label-b c4aa30229 \
        --model <gguf> --fixture benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json \
        --case-id code-p4096 --n-predict 48 --output /tmp/pair-tokens.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_ready(port: int, timeout: float, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early with {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(2.0)
    raise SystemExit("server did not become ready in time")


def run_one(
    server: Path,
    label: str,
    model: Path,
    prompt_ids: list[int],
    n_predict: int,
    kv_dtype: str,
    context: int,
    batch: int,
    ubatch: int,
    threads: int,
    startup_timeout: float,
    extra_args: list[str],
) -> dict:
    port = free_port()
    args = [
        str(server), "-m", str(model),
        "--host", "127.0.0.1", "--port", str(port),
        "--parallel", "1", "--no-webui", "-ngl", "999", "-fa", "on",
        "-ctk", kv_dtype, "-ctv", kv_dtype, "-c", str(context),
        "-b", str(batch), "-ub", str(ubatch), "-t", str(threads),
        *extra_args,
    ]
    log = open(f"/tmp/pair-tokens-{label}.log", "wb")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(port, startup_timeout, proc)
        body = json.dumps({
            "prompt": prompt_ids,
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "cache_prompt": False,
            "return_tokens": True,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as resp:
            payload = json.loads(resp.read())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    tokens = payload.get("tokens")
    if tokens is None:
        raise SystemExit(
            f"{label}: server response has no 'tokens' array; the comparison "
            "cannot be made and must not be reported as agreement"
        )
    tokens = [int(t) for t in tokens]
    if not tokens:
        raise SystemExit(
            f"{label}: server returned an empty 'tokens' array; an empty "
            "sequence would compare equal to another empty sequence, so this "
            "is a failure rather than a pass"
        )
    return {
        "label": label,
        "server": str(server),
        "port": port,
        "tokens": tokens,
        "tokens_predicted": payload.get("tokens_predicted"),
        "prompt_ms": (payload.get("timings") or {}).get("prompt_ms"),
        "predicted_ms": (payload.get("timings") or {}).get("predicted_ms"),
        "content_head": (payload.get("content") or "")[:200],
    }


def compare_arms(a: dict, b: dict, n_predict: int) -> dict:
    """Decide whether two arms emitted the same sequence.

    Raises rather than returning a result when the inputs cannot support a
    comparison. A checker that reports agreement for two empty sequences, or for
    a three-token sequence compared over its own length, is worse than no checker
    because it reads as evidence.
    """
    problems = []
    for arm in (a, b):
        tokens = arm.get("tokens")
        if tokens is None:
            problems.append(f"{arm.get('label')}: no 'tokens' array in the response")
            continue
        if not tokens:
            problems.append(
                f"{arm.get('label')}: empty 'tokens' array; two empty sequences "
                "would compare equal, so this is a failure rather than a pass"
            )
            continue
        if len(tokens) != n_predict:
            problems.append(
                f"{arm.get('label')}: emitted {len(tokens)} tokens, "
                f"expected {n_predict}"
            )
    if problems:
        raise SystemExit(
            "token comparison cannot be evaluated:\n  " + "\n  ".join(problems)
        )

    ta, tb = a["tokens"], b["tokens"]
    n = min(len(ta), len(tb))
    first_diff = next((i for i in range(n) if ta[i] != tb[i]), None)
    return {
        "compared_positions": n,
        "identical_positions": sum(1 for i in range(n) if ta[i] == tb[i]),
        "first_divergence": first_diff,
        "sequences_identical": first_diff is None and len(ta) == len(tb),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-a", type=Path, required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--server-b", type=Path, required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--n-predict", type=int, default=48)
    parser.add_argument("--kv-dtype", default="bf16")
    parser.add_argument("--context", type=int, default=4352)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--ubatch", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--server-arg", action="append", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text())
    case = next((c for c in fixture["cases"] if c["id"] == args.case_id), None)
    if case is None:
        raise SystemExit(f"case {args.case_id} not in {args.fixture}")
    prompt_ids = case["prompt_token_ids"]

    common = dict(
        model=args.model,
        prompt_ids=prompt_ids,
        n_predict=args.n_predict,
        kv_dtype=args.kv_dtype,
        context=args.context,
        batch=args.batch,
        ubatch=args.ubatch,
        threads=args.threads,
        startup_timeout=args.startup_timeout,
        extra_args=list(args.server_arg or []),
    )
    a = run_one(args.server_a, args.label_a, **common)
    b = run_one(args.server_b, args.label_b, **common)

    verdict = compare_arms(a, b, args.n_predict)
    ta, tb = a["tokens"], b["tokens"]
    n = verdict["compared_positions"]
    first_diff = verdict["first_divergence"]
    agree = verdict["identical_positions"]
    identical = verdict["sequences_identical"]

    print(f"{args.case_id}: {len(prompt_ids)} prompt tokens, "
          f"n_predict={args.n_predict}")
    print(f"  {a['label']:16s} emitted {len(ta)} tokens")
    print(f"  {b['label']:16s} emitted {len(tb)} tokens")
    print(f"  identical positions: {agree}/{n}")
    if first_diff is None and len(ta) == len(tb):
        print("  sequences are IDENTICAL")
    else:
        print(f"  first divergence at index {first_diff}")
        if first_diff is not None:
            lo = max(0, first_diff - 3)
            print(f"    {a['label']}: {ta[lo:first_diff + 4]}")
            print(f"    {b['label']}: {tb[lo:first_diff + 4]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema": 1,
        "kind": "llamacpp_pair_token_compare",
        "performance_claim": False,
        "status": "correctness_check",
        "case": args.case_id,
        "prompt_tokens": len(prompt_ids),
        "n_predict": args.n_predict,
        "identical_positions": agree,
        "compared_positions": n,
        "first_divergence": first_diff,
        "sequences_identical": identical,
        "arms": [
            {k: v for k, v in a.items() if k != "tokens"},
            {k: v for k, v in b.items() if k != "tokens"},
        ],
        "tokens": {a["label"]: ta, b["label"]: tb},
        "notes": [
            "Greedy sampling (temperature 0, top_k 1) so divergence is arithmetic, not sampling.",
            "Prompt ids are the fixture's exact token ids, identical for both arms.",
        ],
    }, indent=1) + "\n")
    print(f"\nwrote {args.output}")
    if not identical:
        print(
            f"FAIL: {a['label']} and {b['label']} diverge at index {first_diff}; "
            "a non-zero exit here is what stops this check from passing vacuously",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
