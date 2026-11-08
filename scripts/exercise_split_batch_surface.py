#!/usr/bin/env python3
"""Mixed-length concurrent requests + decode-telemetry path confirmation.

Sends one long prompt (>=1024 tokens so decode selects the split-K batch
path) alongside two short prompts, then reads /v1/hipengine/decode to confirm
the shipping engine selected kv_live_spans_batch_split_gqa.
"""
import concurrent.futures
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8123"

para = ("The quick brown fox jumps over the lazy dog while the inventory system "
        "records every movement across the warehouse shelves and updates the "
        "ledger accordingly. ")
long_prompt = "Summarize the following log in one sentence.\n" + para * 90 + "\nSummary:"
short_prompts = ["Say hello in French.", "What is 2 + 2? Answer briefly."]


def req(prompt):
    body = json.dumps({
        "model": "Qwen3.8-27B-UD-Q4_K_M.gguf",
        "prompt": prompt,
        "max_tokens": 24,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    r = urllib.request.Request(BASE + "/v1/completions", data=body,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.loads(resp.read())


def main():
    prompts = [long_prompt] + short_prompts
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(req, prompts))
    ok = True
    for i, res in enumerate(results):
        text = res["choices"][0]["text"]
        usage = res.get("usage", {})
        print(f"request {i}: prompt_tokens={usage.get('prompt_tokens')} "
              f"completion_tokens={usage.get('completion_tokens')} text={text[:100]!r}")
        if not text.strip():
            ok = False
    with urllib.request.urlopen(BASE + "/v1/hipengine/decode", timeout=30) as resp:
        telemetry = json.loads(resp.read())
    print("decode telemetry:", json.dumps(telemetry, indent=2)[:2000])
    blob = json.dumps(telemetry)
    if "kv_live_spans_batch_split_gqa" not in blob:
        print("FAIL: split-K batch path not observed in telemetry", file=sys.stderr)
        ok = False
    print("SURFACE PASS" if ok else "SURFACE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())