#!/usr/bin/env python3
"""Require real target-prefix hits and working MTP after provider restoration."""

import argparse
import json
from pathlib import Path

import httpx


def assert_restored_pair(baseline, candidate, prefix_tokens):
    for result in (baseline, candidate):
        cache = result["diagnostics"]["prefix_cache"]
        assert cache["hit"] and cache["reused_tokens"] >= prefix_tokens, cache
    cycles = candidate.get("timing", {}).get("mtp_cycles_count", 0)
    assert cycles > 0, candidate["diagnostics"]
    assert baseline["generated_token_ids"] == candidate["generated_token_ids"], {
        "stage": "ids", "ar": baseline["generated_token_ids"],
        "mtp": candidate["generated_token_ids"],
    }
    return cycles


def run(args):
    report = {"passed": False, "rows": []}
    try:
        with httpx.Client(base_url=args.base_url, timeout=300) as client:
            readiness = client.get("/ready")
            readiness.raise_for_status()
            report["kv_capability"] = readiness.json()["model"]["kv_capability"]
            assert readiness.json()["prefix_cache"]["mode"] == "radix"
            assert report["kv_capability"]["effective_kv_storage"] == "int8_per_token_head"
            if report["kv_capability"].get("diagnostic_override"):
                assert args.allow_kv_diagnostic_override

            def encode(text):
                response = client.post("/v1/hipengine/tokenize", json={"text": text})
                response.raise_for_status()
                return response.json()["token_ids"]

            def request(tokens, speculative):
                response = client.post("/v1/completions", json={
                    "model": args.model, "prompt": tokens, "max_tokens": 24,
                    "temperature": 0, "speculative_mtp": speculative,
                })
                response.raise_for_status()
                body = response.json()
                metadata = body["choices"][0]["hipengine"]
                layout = metadata["diagnostics"]["kv_layout"]
                assert layout["storage_dtype"] == "int8_per_token_head"
                assert layout["persistent_bf16_mirror_bytes"] == 0
                return metadata

            prompts = {}
            prompt_file = Path(__file__).resolve().parents[1] / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
            for line in prompt_file.read_text().splitlines():
                row = json.loads(line)
                prompts.setdefault(row["category"], row["messages"][0]["content"])
            for label in ("code", "general_en", "general_ja", "mixed_ja_en"):
                text = prompts[label]
                seed = encode(text)
                prefix = (seed * ((512 + len(seed) - 1) // len(seed)))[:512]
                suffix = encode(" Now summarize the main invariant in one short paragraph.")
                source = request(prefix, True)
                assert source["diagnostics"].get("specdec2_mtp2", {}).get("prompt_streaming"), {
                    "stage": "source", "diagnostics": source["diagnostics"],
                }
                baseline = request(prefix + suffix, False)
                candidate = request(prefix + suffix, True)
                cycles = assert_restored_pair(baseline, candidate, len(prefix))
                report["rows"].append({
                    "category": label, "reused_tokens": candidate["diagnostics"]["prefix_cache"]["reused_tokens"],
                    "cycles": cycles, "ids": candidate["generated_token_ids"],
                })
                print("PASS", label, flush=True)
            report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--model", default="int8-mtp")
    parser.add_argument("--allow-kv-diagnostic-override", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
