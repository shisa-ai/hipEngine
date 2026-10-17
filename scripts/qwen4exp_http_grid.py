#!/usr/bin/env python3
"""Capture a bounded Flash-Next HTTP feature audit, not a performance qualification.

Run against an already running server. Requests, responses and event timestamps
are saved locally; the compact report preserves execution metadata and exact IDs.
A successful HTTP status is not treated as proof that speculation executed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

import httpx


def compact(row: dict) -> dict:
    result = {k: v for k, v in row.items() if k not in {"request", "body", "events"}}
    encoded = json.dumps(row["request"], sort_keys=True).encode()
    result["request_sha256"] = hashlib.sha256(encoded).hexdigest()
    result["request_options"] = {k: v for k, v in row["request"].items()
                                 if k not in {"prompt", "messages"}}
    body = row.get("body", {})
    if isinstance(body, dict):
        result["usage"] = body.get("usage")
        result["hipengine"] = body.get("hipengine")
        result["error"] = body.get("error")
        result["choices"] = body.get("choices", [])
        if "unparsed_body" in body:
            result["unparsed_body"] = body["unparsed_body"]
    if "events" in row:
        result["event_count"] = len(row["events"])
        result["events"] = row["events"]
    return result


def response_body(response):
    try:
        return response.json()
    except ValueError:
        return {"unparsed_body": response.text}


async def request(client, out, name, payload, endpoint="/v1/chat/completions"):
    started = time.perf_counter()
    row = {"name": name, "endpoint": endpoint, "request": payload,
           "started_unix": time.time()}
    try:
        if payload.get("stream"):
            row["events"] = []
            async with client.stream("POST", endpoint, json=payload) as response:
                row["status"] = response.status_code
                if response.status_code != 200:
                    await response.aread()
                    row["body"] = response_body(response)
                else:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            row["events"].append({"elapsed_s": time.perf_counter()-started,
                                "data": data if data == "[DONE]" else json.loads(data)})
        else:
            response = await client.post(endpoint, json=payload)
            row["status"] = response.status_code
            row["body"] = response_body(response)
    except Exception as exc:
        row["transport_error"] = f"{type(exc).__name__}: {exc}"
    row["wall_s"] = time.perf_counter()-started
    (out / f"{name}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: row.get(k) for k in ("name", "status", "wall_s", "transport_error")}), flush=True)
    return compact(row)


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    suite_bytes = args.prompts.read_bytes()
    suite = [json.loads(line) for line in suite_bytes.decode().splitlines() if line.strip()]
    rows = []
    report = {"schema": 1, "kind": "qwen4exp_http_feature_audit", "performance_claim": False,
              "suite_sha256": hashlib.sha256(suite_bytes).hexdigest(), "rows": rows,
              "protocol": vars(args) | {"output": str(args.output), "prompts": str(args.prompts)}}
    def save():
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    async with httpx.AsyncClient(base_url=args.url, timeout=600) as client:
        for endpoint in ("/ready", "/v1/hipengine/capabilities"):
            r = await client.get(endpoint)
            report[endpoint] = r.json()
            r.raise_for_status()
        base = {"model": args.model, "messages": suite[0]["messages"], "temperature": 0,
                "top_p": 1, "max_tokens": 128, "enable_thinking": False}
        async def probe(name, changes=None, *, payload=None, endpoint="/v1/chat/completions"):
            row = await request(client, args.output, name, payload if payload is not None else base | (changes or {}), endpoint)
            rows.append(row)
            save()
            return row
        await probe("default")
        # The complete category suite includes the four category-heldout prompts.
        for item in suite:
            for mode in (False, True):
                await probe(f"suite-{item['id']}-{'mtp' if mode else 'ar'}",
                            {"messages": item["messages"], "speculative_mtp": mode})
        await probe("mtp-alias", {"speculative": True})
        await probe("mtp-ignore-eos", {"speculative_mtp": True, "ignore_eos": True, "max_tokens": 32})
        for mode in (False, True):
            label = "mtp" if mode else "ar"
            route = {"speculative_mtp": mode}
            await probe(f"{label}-256", route | {"max_tokens": 256})
            await probe(f"{label}-stream", route | {"max_tokens": 64, "stream": True,
                "stream_options": {"include_usage": True, "include_hipengine": True}})
            await probe(f"{label}-thinking", route | {"enable_thinking": True, "max_tokens": 64})
            await probe(f"{label}-thinking-budget", route | {"enable_thinking": True,
                "max_think_tokens": 8, "min_answer_tokens": 8, "max_tokens": 64})
            await probe(f"{label}-logprobs", route | {"logprobs": True, "top_logprobs": 2, "max_tokens": 16})
            await probe(f"{label}-stop", route | {"messages": [{"role": "user", "content":
                "Repeat this exact sentence: alpha beta gamma delta epsilon."}],
                "stop": [" beta gamma"], "max_tokens": 32})
            await probe(f"{label}-sampling", route | {"temperature": 0.7, "max_tokens": 16})
            await probe(f"{label}-grammar", route | {"messages": [{"role": "user", "content":
                "Return a JSON object with the integer field answer equal to 42."}], "max_tokens": 32,
                "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True,
                    "schema": {"type": "object", "properties": {"answer": {"type": "integer"}},
                               "required": ["answer"], "additionalProperties": False}}}})
            group_started = time.perf_counter()
            group = await asyncio.gather(*(request(client, args.output, f"{label}-c4-{i}",
                base | route | {"max_tokens": 64, "messages": suite[i]["messages"]}) for i in range(4)))
            report[f"{label}_c4_wall_s"] = time.perf_counter()-group_started
            rows.extend(group)
            save()
        # A natural repeated document, with a known retrieval target near its middle.
        prefix = "<|im_start|>user\nRead this archive and return the access code at the end.\n"
        filler = "Archive entry: The library opens at nine and closes at five. This entry contains no access code.\n"
        encoded = (await client.post("/v1/hipengine/tokenize", json={"text": filler})).json()
        token_ids = encoded.get("token_ids", encoded.get("tokens"))
        if not isinstance(token_ids, list):
            raise ValueError(f"Unexpected tokenizer response: {encoded}")
        count = max(1, (args.long_tokens-128)//len(token_ids)//2)
        text = prefix + filler*count + "The access code is VIOLET-7391.\n" + filler*count
        text += "What is the access code? Return only the code.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        tokens = (await client.post("/v1/hipengine/tokenize", json={"text": text})).json()
        long_ids = tokens.get("token_ids", tokens.get("tokens"))
        report["long_prompt_tokens"] = len(long_ids)
        for mode in (False, True):
            await probe(f"{'mtp' if mode else 'ar'}-long", payload={"model": args.model,
                "prompt": long_ids, "temperature": 0, "max_tokens": 32, "speculative_mtp": mode},
                endpoint="/v1/completions")
        await probe("recovery-ar", {"speculative_mtp": False, "max_tokens": 16})
        report["ready_after"] = (await client.get("/ready")).json()
        report["status"] = "captured"
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18388")
    parser.add_argument("--model", default="qwen38fn-grid")
    parser.add_argument("--prompts", type=Path, default=Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"))
    parser.add_argument("--long-tokens", type=int, default=30000)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
