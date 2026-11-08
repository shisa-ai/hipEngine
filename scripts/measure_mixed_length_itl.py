#!/usr/bin/env python3
"""Mixed-length serving ITL measurement for the empty-split elimination A/B.

Protocol: one ~128K-token request streams 64 decode tokens. After 16 decode
chunks (alone phase), three ~520-token requests with 32 outputs are admitted
concurrently (mixed phase). Per-token arrival timestamps give ITL in each
phase; short-request wall times are recorded separately. Run identically
against the pre-fix and post-fix builds on the same host.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

PARA = ("The quick brown fox jumps over the lazy dog while the inventory system "
        "records every movement across the warehouse shelves and updates the "
        "ledger accordingly. ")


def post_completion(base, model, prompt, max_tokens, stream):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": stream,
    }).encode()
    return urllib.request.Request(
        base + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"})


def stream_long(base, model, prompt, max_tokens, state, cond):
    req = post_completion(base, model, prompt, max_tokens, True)
    t0 = time.perf_counter()
    usage = None
    text_parts = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choice = obj["choices"][0] if obj.get("choices") else {}
            delta = choice.get("text") or choice.get("delta", {}).get("content", "")
            if delta:
                now = time.perf_counter()
                text_parts.append(delta)
                with cond:
                    state["times"].append(now)
                    cond.notify_all()
    with cond:
        state["done"] = True
        state["usage"] = usage
        state["text"] = "".join(text_parts)
        state["wall"] = time.perf_counter() - t0
        cond.notify_all()


def run_short(base, model, prompt, max_tokens, results, idx):
    t0 = time.perf_counter()
    req = post_completion(base, model, prompt, max_tokens, False)
    with urllib.request.urlopen(req, timeout=1200) as resp:
        obj = json.loads(resp.read())
    results[idx] = {
        "wall_s": time.perf_counter() - t0,
        "usage": obj.get("usage"),
        "text_head": obj["choices"][0]["text"][:80],
    }


def itl_stats(times):
    if len(times) < 2:
        return None
    diffs = [(b - a) * 1000 for a, b in zip(times, times[1:])]
    return {
        "n_intervals": len(diffs),
        "median_ms": statistics.median(diffs),
        "mean_ms": statistics.mean(diffs),
        "min_ms": min(diffs),
        "max_ms": max(diffs),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8123")
    ap.add_argument("--model", default="Qwen3.8-27B-UD-Q4_K_M.gguf")
    ap.add_argument("--long-paras", type=int, default=4900,
                    help="~26.1 tokens/para; 4900 ~ 128K tokens")
    ap.add_argument("--long-tokens", type=int, default=64)
    ap.add_argument("--alone-chunks", type=int, default=16)
    ap.add_argument("--short-count", type=int, default=3)
    ap.add_argument("--short-tokens", type=int, default=32)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    long_prompt = ("Summarize the following log in one sentence.\n"
                   + PARA * args.long_paras + "\nSummary:")
    short_prompt = ("Answer briefly.\n" + PARA * 20
                    + "\nQuestion: what animal jumps over the dog?")

    state = {"times": [], "done": False}
    cond = threading.Condition()
    long_thread = threading.Thread(
        target=stream_long,
        args=(args.base, args.model, long_prompt, args.long_tokens, state, cond))
    long_thread.start()

    with cond:
        ok = cond.wait_for(lambda: len(state["times"]) >= args.alone_chunks or state["done"],
                           timeout=3000)
    if not ok:
        raise SystemExit("timed out waiting for long-request decode chunks")
    t_mixed_start = time.perf_counter()
    print(f"alone phase captured ({len(state['times'])} chunks); admitting shorts", flush=True)

    short_results = [None] * args.short_count
    short_threads = [
        threading.Thread(target=run_short,
                         args=(args.base, args.model, short_prompt, args.short_tokens,
                               short_results, i))
        for i in range(args.short_count)
    ]
    for t in short_threads:
        t.start()
    for t in short_threads:
        t.join()
    t_mixed_end = time.perf_counter()
    print(f"shorts done in {t_mixed_end - t_mixed_start:.2f}s", flush=True)

    long_thread.join()
    times = state["times"]
    alone = [t for t in times if t < t_mixed_start]
    mixed = [t for t in times if t_mixed_start <= t <= t_mixed_end]
    post = [t for t in times if t > t_mixed_end]
    # include boundary intervals: last alone chunk -> first mixed chunk etc.
    mixed_with_boundary = [t for t in times if t >= (alone[-1] if alone else t_mixed_start)
                           and t <= (post[0] if post else t_mixed_end)]

    artifact = {
        "label": args.label,
        "date": "2026-11-08",
        "protocol": {
            "long_prompt_paras": args.long_paras,
            "long_usage": state.get("usage"),
            "long_tokens_requested": args.long_tokens,
            "long_chunks_observed": len(times),
            "short_count": args.short_count,
            "short_tokens": args.short_tokens,
            "alone_chunks_target": args.alone_chunks,
        },
        "raw_chunk_times_s": times,
        "t_mixed_start_s": t_mixed_start,
        "t_mixed_end_s": t_mixed_end,
        "long_decode_avg_ms": ((times[-1] - times[0]) * 1000 / (len(times) - 1)) if len(times) > 1 else None,
        "long_decode_mixed_avg_ms": (
            (mixed_with_boundary[-1] - mixed_with_boundary[0]) * 1000
            / (len(mixed_with_boundary) - 1)) if len(mixed_with_boundary) > 1 else None,
        "itl_alone": itl_stats(alone),
        "itl_mixed": itl_stats(mixed_with_boundary),
        "itl_post": itl_stats(post),
        "short_requests": short_results,
        "mixed_window_s": t_mixed_end - t_mixed_start,
        "long_wall_s": state.get("wall"),
        "long_text_head": (state.get("text") or "")[:120],
    }
    with open(args.json_out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(json.dumps({k: artifact[k] for k in
                      ("itl_alone", "itl_mixed", "itl_post", "mixed_window_s")}, indent=2))
    print("artifact:", args.json_out)


if __name__ == "__main__":
    main()