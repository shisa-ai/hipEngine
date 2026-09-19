#!/usr/bin/env python3
"""Same-server AR/MTP pair bench over the OpenAI streaming API.

One harness, one server process, one request shape: every arm differs only in
the request body (for hipEngine, the ``speculative_mtp`` field), so an AR and an
MTP reading from the same load are directly comparable.

Protocol:

  * one discarded warmup per shape, then N measured runs
  * streaming, so TTFT and the decode window are measured, not inferred
  * decode tok/s = (completion_tokens - 1) / (t_last_chunk - t_first_chunk)
  * prefill tok/s = prompt_tokens / TTFT
  * greedy: temperature 0
  * the prompt is a deterministic filler passage of roughly the requested token
    count, so the shape is a length rather than a content choice

Per-arm request extras come from ``BENCH_EXTRA_JSON`` (for example
``{"speculative_mtp": false}`` for the AR control arm), so the arm is recorded
in the artifact rather than in a copied harness. Each run records the routing
block the server reported about itself (effective route, decision reason, MTP
output coverage and cycles, fallback counts); ``--identity-tokens N`` adds one
non-streaming greedy probe per row whose exact generated ids let the arms be
compared token for token. Set ``BENCH_INCLUDE_HIPENGINE=0`` when pointing the
harness at an engine that rejects hipEngine's ``stream_options`` extension.

Usage:
  scripts/gguf_mtp_server_pair_bench.py --url http://127.0.0.1:8081 --model <id> \
      --tag mtp-auto --shapes 945,3530 --decode 128 --repeats 3 --out mtp.json
  BENCH_EXTRA_JSON='{"speculative_mtp": false}' \
  scripts/gguf_mtp_server_pair_bench.py --url http://127.0.0.1:8081 --model <id> \
      --tag ar-control --shapes 945,3530 --decode 128 --repeats 3 --out ar.json
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping

PARA = (
    "The Strix Halo package pairs sixteen Zen 5 cores with a Radeon 8060S "
    "integrated GPU over a unified memory pool, so weights and activations "
    "share the same physical DRAM as the host. Bandwidth, not capacity, sets "
    "the decode ceiling for a dense 27B model at four-bit weights. Kernel "
    "selection therefore matters more than arithmetic throughput: a fused "
    "recurrence or a better matrix shape buys more tokens per second than an "
    "extra compute unit would. "
)


def make_prompt(target_tokens: int) -> str:
    """Deterministic filler prompt of roughly `target_tokens` tokens.

    The task asks for a long continuation rather than an answer: an answered
    question ends at EOS after ~24 tokens, which leaves no decode window to
    time. The exact prompt length is read back from the server's usage block.
    """
    # ~4 chars per token for this English text.
    target_chars = target_tokens * 4
    body = (PARA * (target_chars // len(PARA) + 2))[:target_chars]
    return (
        "Continue the technical passage below in the same style and register. "
        "Write at least 400 more words and do not stop early; do not summarise "
        "and do not add a heading.\n\n" + body + "\n\nContinuation:"
    )


def request_extras() -> dict:
    """Per-engine request fields, e.g. hipEngine's ``speculative_mtp`` pin."""
    return json.loads(os.environ.get("BENCH_EXTRA_JSON", "{}"))


def stream_options() -> dict:
    """Stream options, including hipEngine's own extension channel.

    The routing/accounting block rides on ``include_hipengine``, which is what
    makes a row's own route and speculation coverage observable. Both servers
    this harness compares are OpenAI-compatible and ignore an unknown
    ``stream_options`` key, but a stricter engine can opt out with
    ``BENCH_INCLUDE_HIPENGINE=0`` rather than failing on the request.
    """
    options: dict[str, Any] = {"include_usage": True}
    if os.environ.get("BENCH_INCLUDE_HIPENGINE", "1") not in ("", "0"):
        options["include_hipengine"] = True
    return options


def _merge_extension(target: dict[str, Any], update: Mapping[str, Any]) -> None:
    """Merge a stream's routing extension, keeping earlier nested fields.

    The accounting arrives in pieces: a chunk may carry the route decision while
    the terminal usage chunk carries the cycle and coverage counters. Replacing
    the nested mapping outright would drop whichever half arrived first, so
    nested mappings are merged and only leaf values are overwritten.
    """

    for key, value in update.items():
        current = target.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged = dict(current)
            merged.update(value)
            target[key] = merged
        else:
            target[key] = value


def route_summary(extension: Mapping[str, Any]) -> dict:
    """The routing and MTP accounting a served row reports about itself.

    A serving rate is only comparable across arms if the row says which route
    ran and how much of its output came from speculative cycles, so the pair
    harness records the same fields the ShareGPT recorder reads rather than
    inferring coverage from the arm's intent. Streaming and non-streaming
    responses carry the same accounting under slightly different keys, so both
    channels are read.
    """
    route = (extension.get("generation_shape") or {}).get("route_decision") or {}
    mtp = extension.get("speculative_mtp") or {}
    accounting = mtp.get("output_accounting") or {}
    token_accounting = extension.get("token_accounting") or {}
    return {
        "effective_route": (
            route.get("effective_route")
            or mtp.get("effective_route")
            or extension.get("effective_route")
        ),
        "decision_reason": route.get("decision_reason") or mtp.get("decision_reason"),
        "mtp_used": bool(mtp.get("used")),
        "speculative_cycles": mtp.get("draft_cycles") or mtp.get("speculative_cycles"),
        "mtp_output_tokens": accounting.get("mtp_output_tokens")
        or mtp.get("mtp_output_tokens"),
        "ar_output_tokens": accounting.get("ar_output_tokens")
        or mtp.get("ar_output_tokens"),
        "mtp_coverage": accounting.get("mtp_coverage") or mtp.get("mtp_coverage"),
        "fallback_event_counts": mtp.get("fallback_event_counts"),
        "generated_ids": _flat_generated_ids(
            token_accounting.get("choice_generated_token_ids")
        ),
    }


def _flat_generated_ids(value: Any) -> list[int] | None:
    """Normalize ``choice_generated_token_ids`` to one choice's id list.

    The API reports ids per choice, so a single-choice response arrives as
    ``[[...]]``. Comparing two arms is a token-for-token comparison of the
    sequence each one emitted, so the nested single-choice form is flattened
    and a multi-choice response keeps its per-choice lists.
    """
    if not isinstance(value, list) or not value:
        return None
    if all(isinstance(item, list) for item in value):
        if len(value) != 1:
            return value
        return list(value[0])
    return list(value)


def identity_once(url: str, model: str, prompt: str, decode_tokens: int,
                  timeout: float = 900.0) -> dict:
    """Non-streaming greedy probe used as the cross-arm identity oracle.

    Exact generated ids are only attached to a non-streaming response, so the
    identity probe is a separate request from the timed streaming repeats. It
    runs the same route the repeats do (the request fields are identical apart
    from ``stream``) and reports the ids, the text and the routing block.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": decode_tokens,
        "temperature": 0.0,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
        **request_extras(),
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (body.get("usage") or {}).get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "text": message.get("content") or "",
        "reasoning_chars": len(message.get("reasoning_content") or ""),
        **route_summary(body.get("hipengine") or {}),
    }


def stream_once(url: str, model: str, prompt: str, decode_tokens: int,
                timeout: float = 900.0) -> dict:
    # Identical JSON to both engines, including the thinking pin: Atlas defaults
    # to thinking off, hipEngine defaults to on and streams `reasoning_content`.
    # Both accept the top-level field and the chat_template_kwargs mirror.
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": decode_tokens,
        "temperature": 0.0,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": stream_options(),
        # Optional per-engine extras (e.g. hipEngine's ``speculative_mtp``).
        **request_extras(),
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    t_last = None
    text = []
    reasoning = []
    usage = {}
    finish = None
    extension: dict[str, Any] = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk.get("hipengine"), Mapping):
                _merge_extension(extension, chunk["hipengine"])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if isinstance(choice.get("hipengine"), Mapping):
                    _merge_extension(extension, choice["hipengine"])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                piece = delta.get("content")
                if piece or delta.get("reasoning_content"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    t_last = now - t0
                if piece:
                    text.append(piece)
    total = time.perf_counter() - t0
    completion = usage.get("completion_tokens")
    if completion is None:
        completion = len(text)
    decode_s = (t_last - ttft) if (t_last is not None and ttft is not None) else None
    return {
        "ttft_s": ttft,
        "total_s": total,
        "decode_window_s": decode_s,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion,
        "finish_reason": finish,
        "reasoning_tokens_chars": len("".join(reasoning)),
        "decode_tok_s": ((completion - 1) / decode_s) if decode_s else None,
        "prefill_tok_s": (usage.get("prompt_tokens") / ttft) if ttft else None,
        "text": "".join(text),
        **route_summary(extension),
    }


def cv(values) -> float:
    vals = [v for v in values if v is not None]
    if len(vals) < 2 or statistics.mean(vals) == 0:
        return 0.0
    return statistics.stdev(vals) / statistics.mean(vals)


def load_prompt_suite(path) -> list[dict]:
    """Read the committed JSONL prompt suites (``id``/``category``/``messages``)."""
    prompts: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        name = str(item.get("id") or item.get("name") or "")
        if not name:
            raise SystemExit(f"{path}:{line_number}: prompt requires id/name")
        text = item.get("prompt")
        if not isinstance(text, str) or not text:
            messages = item.get("messages")
            if not isinstance(messages, list) or len(messages) != 1:
                raise SystemExit(
                    f"{path}:{line_number}: prompt requires text or one user message"
                )
            text = str(messages[0].get("content") or "")
        if not text:
            raise SystemExit(f"{path}:{line_number}: prompt text is empty")
        prompts.append(
            {
                "id": name,
                "category": str(item.get("category") or "uncategorized"),
                "text": text,
            }
        )
    if not prompts:
        raise SystemExit(f"{path} contains no prompts")
    return prompts


def pad_prompt(text: str, *, pad_to_tokens: int | None) -> str:
    """Prepend a deterministic filler passage to reach a target context length.

    Padding is character-approximate (~4 chars/token) and identical for every
    arm; the artifact records the token count the server reports. The suite
    prompt stays the tail, so the model still answers the original task.
    """
    if not pad_to_tokens or int(pad_to_tokens) <= 0:
        return text
    target_chars = int(pad_to_tokens) * 4
    body = (PARA * (target_chars // len(PARA) + 2))[:target_chars]
    return (
        "Read the reference passage below, then follow the instruction that "
        "comes after it.\n\n" + body + "\n\nInstruction:\n" + text
    )


def measure(
    args: argparse.Namespace,
    *,
    label: str,
    prompt: str,
    result: dict,
    key: str,
) -> dict:
    """Run the warmup plus measured repeats for one prompt and record them."""
    runs = []
    for i in range(args.repeats + 1):          # +1 discarded warmup
        r = stream_once(args.url, args.model, prompt, args.decode)
        if i == 0:
            print(f"[{args.tag}] warmup {label}: {r['decode_tok_s']:.2f} tok/s "
                  f"({r['completion_tokens']} tok)", flush=True)
            continue
        runs.append(r)
        short = " SHORT!" if r["completion_tokens"] < args.decode else ""
        print(f"[{args.tag}] {label} run{i}: "
              f"decode {r['decode_tok_s']:.2f} tok/s, ttft {r['ttft_s']*1000:.0f} ms, "
              f"prompt {r['prompt_tokens']} tok, gen {r['completion_tokens']} tok "
              f"({r['finish_reason']}){short}", flush=True)
    identity = None
    if args.identity_tokens:
        identity = identity_once(
            args.url, args.model, prompt, int(args.identity_tokens)
        )
        print(f"[{args.tag}] {label} identity: {len(identity['generated_ids'] or [])} ids, "
              f"route {identity['effective_route']}, mtp_used {identity['mtp_used']}", flush=True)
    return {
        "prompt_tokens": runs[0]["prompt_tokens"],
        "completion_tokens": runs[0]["completion_tokens"],
        "identity": identity,
        "route": {k: runs[0][k] for k in (
            "effective_route", "decision_reason", "mtp_used", "speculative_cycles",
            "mtp_output_tokens", "ar_output_tokens", "mtp_coverage",
            "fallback_event_counts",
        )},
        "decode_tok_s_median": statistics.median([r["decode_tok_s"] for r in runs]),
        "decode_tok_s_cv": cv([r["decode_tok_s"] for r in runs]),
        "prefill_tok_s_median": statistics.median([r["prefill_tok_s"] for r in runs]),
        "prefill_tok_s_cv": cv([r["prefill_tok_s"] for r in runs]),
        "ttft_ms_median": statistics.median([r["ttft_s"] * 1000 for r in runs]),
        "total_s_median": statistics.median([r["total_s"] for r in runs]),
        "sample_text": runs[0]["text"][:600],
        "runs": [{k: v for k, v in r.items() if k != "text"} for r in runs],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shapes", default="512,1024,4096")
    ap.add_argument("--prompts-file", type=Path,
                    help="JSONL suite; switches the run from shapes to prompt ids")
    ap.add_argument("--prompt-names", help="comma-separated subset of suite ids")
    ap.add_argument("--limit", type=int, help="run only the first N selected suite prompts")
    ap.add_argument("--pad-to-tokens", type=int,
                    help="prepend deterministic filler so the context reaches this size")
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--identity-tokens",
        type=int,
        default=0,
        help=(
            "Also run one non-streaming greedy probe per shape/prompt with this "
            "output budget and record its exact generated ids, so the two arms "
            "can be compared token for token. 0 disables the probe."
        ),
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Wait for the server, then report the model id it actually serves.
    for _ in range(300):
        try:
            with urllib.request.urlopen(args.url.rstrip("/") + "/v1/models", timeout=5) as r:
                served = [m["id"] for m in json.load(r)["data"]]
            break
        except Exception:
            time.sleep(1)
    else:
        print(f"{args.tag}: server never came up at {args.url}", file=sys.stderr)
        return 2
    print(f"[{args.tag}] /v1/models -> {served}", flush=True)

    result = {
        "tag": args.tag,
        "url": args.url,
        "served_models": served,
        "request_extras": json.loads(os.environ.get("BENCH_EXTRA_JSON", "{}")),
        "shapes": {},
        "prompts": {},
    }
    if args.prompts_file is not None:
        prompts = load_prompt_suite(args.prompts_file)
        if args.prompt_names:
            wanted = [name.strip() for name in args.prompt_names.split(",") if name.strip()]
            prompts = [p for p in prompts if p["id"] in wanted]
        if args.limit is not None:
            prompts = prompts[: int(args.limit)]
        if not prompts:
            print(f"[{args.tag}] no prompts selected", file=sys.stderr)
            return 2
        result["prompts_file"] = str(args.prompts_file)
        result["pad_to_tokens"] = args.pad_to_tokens
        for prompt in prompts:
            label = f"{prompt['id']}[{prompt['category']}]"
            result["prompts"][prompt["id"]] = {
                "category": prompt["category"],
                **measure(
                    args,
                    label=label,
                    prompt=pad_prompt(prompt["text"], pad_to_tokens=args.pad_to_tokens),
                    result=result,
                    key=prompt["id"],
                ),
            }
    else:
        for shape in [int(s) for s in args.shapes.split(",")]:
            result["shapes"][shape] = measure(
                args,
                label=str(shape),
                prompt=make_prompt(shape),
                result=result,
                key=str(shape),
            )
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[{args.tag}] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
