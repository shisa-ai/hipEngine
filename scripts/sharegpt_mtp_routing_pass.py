#!/usr/bin/env python3
"""Measure how a real ShareGPT load routes through the speculative MTP path.

The vLLM serving benchmark's ShareGPT loader is the load definition: prompt is
the first conversation turn, the expected output length is the second turn's
token count, and rows are filtered to 4..1024 prompt tokens, 4..2048 total. This
harness sends that same load to a running hipEngine server and records, for every
request, which execution mode produced its output:

* ``hipengine.speculative_mtp`` (used, MTP-committed versus autoregressive
  output tokens, the first fallback position, and the planner reason),
* ``hipengine.generation_shape.route_decision`` (the route the model plugin
  resolved, and why),
* a per-chunk decode timeline, so a request that leaves speculation mid-output
  shows the resulting latency step rather than only an aggregate rate.

The per-request ``reconciled`` field is an attribution result, not arithmetic:
``ar_output_tokens`` is ``completion - mtp`` by construction, so the row also
carries the committed cycle counts that independently bound the speculative
output, and the failed checks when it does not reconcile. Start the server with
``HIPENGINE_MTP2_OUTPUT_SPANS=1`` to have the backend record committed output
spans (execution mode, reason, emitted-token position, token count); the harness
then records whether those spans tile the emitted output, which is the strongest
form of the proof. ``fallback_event_counts`` counts non-speculative steps and
refusals (events) and is never a token count; the token-level attribution for
the same reasons is ``ar_output_tokens_by_reason`` from the spans.

The arms a routing comparison needs are command-line switches, not separate
harnesses:

* ``--speculative-mtp on|off`` sends the per-request ``speculative_mtp`` field,
  so the true autoregressive control is the same load on the same server
  process with the provider explicitly disabled (never a verifier-derived off
  row); ``auto`` omits the field and keeps the server policy.
* ``--route-mix off,auto`` assigns the arm per request inside one measured load,
  cycling in submission order, so the true-AR control and the speculative arm
  run concurrently under the same occupancy, thermal and cache conditions
  instead of in two separate runs. Every row carries its requested arm in
  ``route_arm``, and the summary reports each arm separately.
* ``--cases short,long`` selects a labelled mix of prompt lengths in one load:
  a healthy short request beside the boundary-crossing row that the provider's
  window may refuse, so the provider-ready and provider-refused paths are
  measured together and split by ``case`` in the summary. ``beyond`` selects a
  prompt past the loader's own 1,024-token pruning bound (raise
  ``--max-prompt-len`` with it) for the provider-window refusal; note that the
  head tree's packed paged autoregressive prefill rejects a context of 1,024 or
  more with ``unsupported_parameter``, so a ``beyond`` prompt only measures the
  refusal on a server whose prefill path accepts that length.
* ``--min-prompt-len`` selects the boundary-crossing row (for example 900) that
  the default 4-token floor excludes, so a healthy short request and a row that
  crosses the provider's context window can be measured together.
* ``--stagger-ms`` spaces submissions so requests arrive while others decode,
  which is the changing-occupancy case a single ``pool.map`` burst cannot show.
* ``--prompt-repeats`` sends every prompt more than once; repeat rows after the
  first reuse the prompt prefix and exercise the prefix-cache hit path beside
  the miss path. The summary splits them (``by_repeat``) so a hit's rate is not
  read as a miss's.
* ``--cancel-count`` aborts the last N requests after their first generated
  token, so the server's cancel/refill path runs beside completed requests;
  cancelled rows are reported separately from failures.
* ``--record-ids`` keeps the generated token ids per row (about 4 bytes per
  token) and reports ``arm_identity``: for every prompt served by more than one
  arm, whether the arms produced identical ids and how many leading tokens they
  share. That is the quality gate for a speculative route, measured in the same
  run rather than inferred from two artifacts.

The summary reports realized group composition (``groups_by_realized_rows``),
a decode rate that excludes prefill (``decode_tokens_per_second_excluding_prefill``),
and the serving metrics a routing change has to move: completed requests per
second, time to first token, inter-token latency, and end-to-end latency, each
with a tail percentile. Every completed row carries an ``output_sha256`` over
its thinking and answer text, so two arms of the same load can be checked for
greedy-exactness instead of assuming it. MTP usage alone does not say whether
the run was faster, so it is reported as a diagnostic beside those numbers.

Usage:
    python3 scripts/sharegpt_mtp_routing_pass.py \
        --server-url http://127.0.0.1:8030 --model qwen3.8-27b-q4km \
        --dataset /home/lhl/models/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --gguf /home/lhl/models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
        --num-prompts 24 --max-concurrency 1 --output-len 128 \
        --label c1-len128 --out /tmp/sharegpt-c1-len128.json

    # True-AR control beside the speculative arm, same load, same process:
    python3 scripts/sharegpt_mtp_routing_pass.py \
        --server-url http://127.0.0.1:8030 \
        --dataset /home/lhl/models/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 8 --max-concurrency 1 --output-len 0 \
        --route-mix off,auto --cases short,long --record-ids \
        --label c1-control --out /tmp/sharegpt-c1-control.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "qwen3.8-27b-q4km"
DEFAULT_GGUF = Path("/home/lhl/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
# vLLM's ShareGPTDataset pruning criteria (benchmarks/datasets/datasets.py).
MIN_LEN = 4
MAX_PROMPT_LEN = 1024
MAX_TOTAL_LEN = 2048
# Named prompt-length cases for ``--cases``. ``short`` is the healthy request a
# provider serves immediately; ``long`` is the boundary-crossing row whose
# prefill may still be priming the provider when decode starts.
CASE_BOUNDS: dict[str, tuple[int, int]] = {
    "short": (MIN_LEN, 64),
    "mid": (65, 511),
    "long": (512, MAX_PROMPT_LEN),
    # Deliberately past the loader's own pruning bound, so a prompt the
    # provider's window refuses can be measured beside a healthy short one.
    # Selecting it requires raising --max-prompt-len past 1024 as well.
    "beyond": (MAX_PROMPT_LEN + 1, 4096),
}


def _tokenizer(gguf: Path):
    from hipengine.loading.gguf import GGUFReader
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    return Qwen35GGUFTokenizer.from_gguf_info(GGUFReader(gguf).info)


def load_samples(
    dataset: Path,
    *,
    tokenizer,
    count: int,
    seed: int,
    output_len: int | None,
    max_prompt_len: int,
    max_total_len: int,
    min_prompt_len: int = MIN_LEN,
    cases: Sequence[str] = (),
    turns: int = 1,
) -> list[dict[str, Any]]:
    """Reproduce the vLLM ShareGPT sample set for one benchmark point.

    With ``cases`` the count is split evenly across the named prompt-length
    cases (``short``/``mid``/``long``) and every row is labelled with its case,
    so one load can hold a healthy short request beside a boundary-crossing
    row. Without it the selection is the original single-window sample.

    With ``turns > 1`` each dataset conversation becomes one growing multi-turn
    conversation: turn ``t`` submits the first ``t`` user turns and their
    dataset replies, so every turn after the first shares a token-exact prefix
    with the turn before it. ``count`` then counts conversations, not requests,
    and the load holds ``count * turns`` requests. The prompts come from the
    dataset rather than from the model's own replies so that both arms of a
    paired run submit byte-identical prompt sequences.
    """

    with dataset.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    rows = [
        row
        for row in rows
        if isinstance(row.get("conversations"), list) and len(row["conversations"]) >= 2
    ]
    random.Random(seed).shuffle(rows)
    if int(turns) > 1:
        return _multi_turn_candidates(
            rows,
            tokenizer=tokenizer,
            count=count,
            turns=int(turns),
            output_len=output_len,
            max_prompt_len=max_prompt_len,
            max_total_len=max_total_len,
            min_prompt_len=min_prompt_len,
        )
    candidates: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row["conversations"][0].get("value") or "")
        completion = str(row["conversations"][1].get("value") or "")
        prompt_len = len(tokenizer.encode(prompt))
        expected = (
            len(tokenizer.encode(completion)) if output_len is None else output_len
        )
        if prompt_len < min_prompt_len or prompt_len > max_prompt_len:
            continue
        if expected < MIN_LEN or prompt_len + expected > max_total_len:
            continue
        candidates.append(
            {
                "source_id": row.get("id"),
                "prompt": prompt,
                "prompt_tokens": prompt_len,
                "expected_output_tokens": expected,
            }
        )
    if not cases:
        samples = candidates[:count]
        if len(samples) < count:
            raise SystemExit(f"dataset produced {len(samples)} valid rows, need {count}")
        return samples
    samples = []
    per_case = -(-count // len(cases))
    for case in cases:
        low, high = CASE_BOUNDS[case]
        selected = [
            candidate
            for candidate in candidates
            if low <= int(candidate["prompt_tokens"]) <= high
        ][:per_case]
        if len(selected) < per_case:
            raise SystemExit(
                f"case {case!r} produced {len(selected)} valid rows, need {per_case}"
            )
        samples.extend({**candidate, "case": case} for candidate in selected)
    return samples


def _multi_turn_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer,
    count: int,
    turns: int,
    output_len: int | None,
    max_prompt_len: int,
    max_total_len: int,
    min_prompt_len: int,
) -> list[dict[str, Any]]:
    """Expand each conversation into its ``turns`` growing prompts.

    Turn ``t`` renders the dataset's first ``t`` user turns and the replies
    between them, so turn ``t + 1`` opens with turn ``t``'s prompt verbatim and
    a prefix cache has a token-exact prefix to reuse. Every turn of a
    conversation is kept or dropped together: a conversation whose later turns
    exceed the loader's length bounds would otherwise measure a different
    request than the one the bound describes.
    """

    conversations: list[list[dict[str, Any]]] = []
    for row in rows:
        values = [
            str(entry.get("value") or "")
            for entry in row["conversations"]
            if isinstance(entry, Mapping)
        ]
        if len(values) < 2 * turns - 1:
            continue
        parts: list[str] = []
        expanded: list[dict[str, Any]] = []
        usable = True
        for turn_index in range(turns):
            parts.append(values[2 * turn_index])
            prompt = "\n\n".join(parts)
            prompt_len = len(tokenizer.encode(prompt))
            expected = (
                len(tokenizer.encode(values[2 * turn_index + 1]))
                if output_len is None and 2 * turn_index + 1 < len(values)
                else (output_len if output_len is not None else 0)
            )
            if prompt_len < min_prompt_len or prompt_len > max_prompt_len:
                usable = False
                break
            if expected < MIN_LEN or prompt_len + expected > max_total_len:
                usable = False
                break
            expanded.append(
                {
                    "source_id": row.get("id"),
                    "conversation_id": row.get("id"),
                    "turn_index": turn_index,
                    "turns": turns,
                    "prompt": prompt,
                    "prompt_tokens": prompt_len,
                    "expected_output_tokens": expected,
                }
            )
            if 2 * turn_index + 1 < len(values):
                parts.append(values[2 * turn_index + 1])
        if not usable:
            continue
        conversations.append(expanded)
        if len(conversations) >= count:
            break
    if len(conversations) < count:
        raise SystemExit(
            f"dataset produced {len(conversations)} usable {turns}-turn "
            f"conversations, need {count}"
        )
    return [turn for conversation in conversations for turn in conversation]


def _stream_request(
    url: str,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float | None,
    timeout: float,
    speculative_mtp: bool | None = None,
    cancel_after_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": int(max_tokens),
        "stream": True,
        "stream_options": {"include_usage": True, "include_hipengine": True},
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if speculative_mtp is not None:
        # The request-level override is the only true-AR control: it makes the
        # same load on the same server process take the registered strict
        # fallback instead of a route decision that merely reports "ar".
        payload["speculative_mtp"] = bool(speculative_mtp)
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    timeline: list[list[float]] = []
    usage: Mapping[str, Any] | None = None
    terminal: Mapping[str, Any] | None = None
    error_body: Mapping[str, Any] | None = None
    # The realized-route and MTP extension can arrive on a choice (the terminal
    # chunk) or on the top-level event (the usage chunk). Keep whichever channel
    # carries it so a rejected or errored stream still reports its route.
    stream_extension: dict[str, Any] = {}
    answer: list[str] = []
    thinking: list[str] = []
    started = time.perf_counter()
    first_token_ms: float | None = None
    first_answer_ms: float | None = None
    first_thinking_ms: float | None = None
    cancelled = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if isinstance(event.get("error"), Mapping):
                # A request rejected mid-stream (for example min_tokens without
                # eos_token_id) arrives as a normal SSE chunk with
                # finish_reason="error" and this body. Recording it is the
                # difference between "stream ended without usage" and the cause.
                error_body = event["error"]
            if isinstance(event.get("hipengine"), Mapping):
                stream_extension.update(event["hipengine"])
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                # A Qwen-style thinking reply streams its reasoning on a
                # separate channel, so the first generated token is usually a
                # reasoning token. Both channels are timed and counted.
                if delta.get("reasoning_content"):
                    if first_thinking_ms is None:
                        first_thinking_ms = (time.perf_counter() - started) * 1000.0
                    if first_token_ms is None:
                        first_token_ms = first_thinking_ms
                    thinking.append(delta["reasoning_content"])
                if delta.get("content"):
                    if first_answer_ms is None:
                        first_answer_ms = (time.perf_counter() - started) * 1000.0
                    if first_token_ms is None:
                        first_token_ms = first_answer_ms
                    answer.append(delta["content"])
                if choice.get("finish_reason") is not None:
                    terminal = choice
                if isinstance(choice.get("hipengine"), Mapping):
                    stream_extension.update(choice["hipengine"])
                state = ((choice.get("hipengine") or {}).get("decode_state")) or {}
                generated = state.get("generated_tokens")
                if generated is not None:
                    timeline.append(
                        [
                            (time.perf_counter() - started) * 1000.0,
                            int(generated),
                        ]
                    )
                if (
                    cancel_after_tokens is not None
                    and generated is not None
                    and int(generated) >= int(cancel_after_tokens)
                ):
                    # Deliberate client cancellation: close the stream the way
                    # a disconnected client does, so the server's cancel/refill
                    # path is exercised. The row is marked cancelled rather
                    # than failed.
                    cancelled = True
                    break
            if cancelled:
                break
            if event.get("usage"):
                usage = event["usage"]
    e2e_ms = (time.perf_counter() - started) * 1000.0
    terminal_extension = (terminal or {}).get("hipengine") or {}
    extension = {**stream_extension, **terminal_extension}
    answer_text = "".join(answer)
    thinking_text = "".join(thinking)
    return {
        "ttft_ms": first_token_ms,
        "first_answer_ms": first_answer_ms,
        "first_thinking_ms": first_thinking_ms,
        "first_decode_state_ms": timeline[0][0] if timeline else None,
        "e2e_ms": e2e_ms,
        "cancelled": cancelled,
        "timeline": timeline,
        "usage": usage,
        "error_body": None if error_body is None else dict(error_body),
        "answer_characters": len(answer_text),
        "thinking_characters": len(thinking_text),
        # One digest over both channels. Greedy MTP is an exact verifier of the
        # autoregressive output, so two arms that disagree on the text for the
        # same prompt are a quality failure, not measurement noise. Hashing
        # keeps the artifact small enough to retain.
        "output_sha256": hashlib.sha256(
            (thinking_text + "\x00" + answer_text).encode("utf-8")
        ).hexdigest(),
        "speculative_mtp": extension.get("speculative_mtp"),
        "generation_shape": extension.get("generation_shape"),
        "routing": extension.get("routing"),
        # The backend's per-request diagnostics (prefix-cache outcome, MTP
        # accounting) and its own timing decomposition. Both are published on
        # the terminal chunk; keeping them here is what makes a prefix hit and
        # its prefill cost readable per row instead of inferred from a shorter
        # TTFT.
        "diagnostics": extension.get("diagnostics"),
        "timing": extension.get("timing"),
        # The greedy token stream, when the server reports it. Two arms of the
        # same prompt can then be compared for identity inside one run instead
        # of by digest across two artifacts.
        "generated_token_ids": [
            int(value) for value in (extension.get("generated_token_ids") or [])
        ],
    }

# Tokens per inter-token-latency window. Long enough that a burst's buffered
# reads average out, short enough that a mid-stream stall still shows up.
ITL_WINDOW_TOKENS = 16


def _inter_token_latencies_ms(
    timeline: Any, *, window: int = ITL_WINDOW_TOKENS
) -> list[float]:
    """Milliseconds per generated token, measured over token windows.

    The live c=2 artifact reported a 0.29 ms median because the decode reports
    arrive in buffered bursts: a speculative cycle's committed tokens are read
    microseconds apart while the wait for the next cycle is not reported at
    all. Dividing each consecutive pair therefore measures the client's read
    pattern, not the model's decode cost, and a stalled window disappears into
    one large interval that a median hides.

    Rate is measured over windows of ``window`` committed tokens instead, and
    the per-token cost is the window's span divided by its tokens. Buffering
    cannot fake a window's rate, and a stalled window is a slow window rather
    than an invisible one.
    """

    points: list[tuple[float, int]] = []
    for entry in timeline or ():
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            points.append((float(entry[0]), int(entry[1])))
        except (TypeError, ValueError):
            continue
    intervals: list[float] = []
    start = 0
    while start < len(points) - 1:
        end = start + 1
        while end < len(points) - 1 and points[end][1] - points[start][1] < window:
            end += 1
        tokens = points[end][1] - points[start][1]
        span_ms = points[end][0] - points[start][0]
        if tokens > 0 and span_ms > 0:
            intervals.append(span_ms / tokens)
        start = end
    return intervals


def _decode_ms_per_token(timeline: Any) -> float | None:
    """Milliseconds per generated token across the whole decode, no bursts.

    The aggregate is the cross-check on a burst-amortized median: if the two
    disagree by more than a small factor, the timeline is not a decode
    sequence and the row should be read with that in mind.
    """

    entries: list[tuple[float, int]] = []
    for entry in timeline or ():
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            entries.append((float(entry[0]), int(entry[1])))
        except (TypeError, ValueError):
            continue
    if len(entries) < 2:
        return None
    tokens = max(entry[1] for entry in entries) - entries[0][1]
    if tokens <= 0:
        return None
    return (entries[-1][0] - entries[0][0]) / tokens


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _error_row(sample: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Describe one request that produced no usage, without hiding the cause."""

    error = result.get("error_body") or {}
    extension = error.get("hipengine") if isinstance(error.get("hipengine"), Mapping) else {}
    message = error.get("message") or "stream ended without usage"
    detail = [str(message)]
    if error.get("code"):
        detail.append(f"code={error['code']}")
    if error.get("type"):
        detail.append(f"type={error['type']}")
    if extension.get("status_code") is not None:
        detail.append(f"status={extension['status_code']}")
    detail.append(
        f"events={len(result.get('timeline') or [])},"
        f" answer_characters={result.get('answer_characters')},"
        f" thinking_characters={result.get('thinking_characters')},"
        f" e2e_ms={result.get('e2e_ms')}"
    )
    return {
        "source_id": sample.get("source_id"),
        "case": sample.get("case"),
        "route_arm": sample.get("route_arm"),
        "prompt_tokens_local": sample.get("prompt_tokens"),
        "expected_output_tokens": sample.get("expected_output_tokens"),
        "e2e_ms": result.get("e2e_ms"),
        "cancelled": bool(result.get("cancelled")),
        "error": " ".join(detail),
        "error_code": error.get("code"),
        "error_type": error.get("type"),
        "error_status_code": extension.get("status_code"),
        "error_message": None if error.get("message") is None else str(error["message"]),
        "error_body": None if not error else dict(error),
        # Even a rejected request carries its realized route when the server
        # reported it, which is what makes a route-level rejection legible.
        **_route_fields(result),
    }


def _route_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    """Realized route, planned route, and refusal reason for one request."""

    shape = result.get("generation_shape") or {}
    decision = shape.get("route_decision") or {}
    return {
        # ``effective_route`` is the route the request actually executed, which
        # is what a reader needs to compare against ``selected_route``. Reporting
        # only the latter made an earlier pass read as "route=speculative_mtp
        # with mtp_used=false".
        "effective_route": shape.get("route"),
        "selected_route": decision.get("selected_route"),
        "requested_route": decision.get("requested_route"),
        "route_decision_reason": decision.get("reason"),
        "k0_class": decision.get("k0_class"),
        "policy_reason": decision.get("policy_reason"),
        "static_intent_allowed": decision.get("static_intent_allowed"),
        "realized_group_rows": decision.get("realized_group_rows"),
        "output_horizon_tokens": decision.get("output_horizon_tokens"),
        "selected_candidate_count": decision.get("selected_candidate_count"),
    }


def _prefix_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    """Per-request prefix-cache telemetry, when the backend published it.

    The terminal chunk carries the backend's ``diagnostics`` block, whose
    ``prefix_cache`` entry says whether this row reused a cached prefix, how
    many tokens it avoided prefilling, and why a lookup missed. Without it a
    hit is only inferable from a shorter TTFT, which is exactly the inference a
    prefix measurement must not have to make.
    """

    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return {}
    block = diagnostics.get("prefix_cache")
    if not isinstance(block, Mapping):
        return {}
    return {
        "prefix_mode": block.get("mode"),
        "prefix_eligible": block.get("eligible"),
        "prefix_lookup": block.get("lookup"),
        "prefix_hit": block.get("hit"),
        "prefix_source": block.get("source"),
        "prefix_matched_tokens": block.get("matched_tokens"),
        "prefix_reused_tokens": block.get("reused_tokens"),
        "prefix_executed_prefill_tokens": block.get("executed_prefill_tokens"),
        "prefix_state_clone_bytes": block.get("state_clone_bytes"),
        "prefix_snapshot_hit": block.get("snapshot_hit"),
        "prefix_fallback_reason": block.get("fallback_reason"),
        "prefix_cache_resident_bytes": block.get("cache_resident_bytes"),
        "prefix_cache_resident_entries": block.get("cache_resident_entries"),
    }


def _request_row(sample: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    usage = result.get("usage") or {}
    if usage.get("completion_tokens") is None:
        # A stream that ends without usage is a truncated or aborted response;
        # record it as a failure instead of an empty success row.
        return _error_row(sample, result)
    mtp = result.get("speculative_mtp") or {}
    execution = mtp.get("execution") or {}
    shape = result.get("generation_shape") or {}
    decision = shape.get("route_decision") or {}
    accounting = mtp.get("output_accounting") or {}
    itl = _inter_token_latencies_ms(result.get("timeline"))
    return {
        "source_id": sample.get("source_id"),
        "case": sample.get("case"),
        "route_arm": sample.get("route_arm"),
        # Multi-turn load identity. Turn 0 has no prefix to reuse and every
        # later turn shares the turn before it, so the turn index is the key the
        # summary splits the reuse evidence by.
        "turn_index": sample.get("turn_index"),
        "conversation_id": sample.get("conversation_id"),
        "prompt_tokens_local": sample.get("prompt_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "expected_output_tokens": sample.get("expected_output_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cancelled": bool(result.get("cancelled")),
        "ttft_ms": result.get("ttft_ms"),
        "first_answer_ms": result.get("first_answer_ms"),
        "first_thinking_ms": result.get("first_thinking_ms"),
        "e2e_ms": result.get("e2e_ms"),
        # Inter-token latency after the first token, from the decode-state
        # timeline. This is the latency a streaming client actually feels and
        # the quantity a coverage change must not degrade.
        # Inter-token latency for a client is the average gap between tokens it
        # received, so the row's number is the whole-timeline average and the
        # windowed values are the tail: buffering can only shorten a window's
        # measured span, so a high windowed p95 is real evidence of a stall
        # while a low windowed median is not evidence of speed.
        "itl_median_ms": (
            _decode_ms_per_token(result.get("timeline"))
            if _decode_ms_per_token(result.get("timeline")) is not None
            else (statistics.median(itl) if itl else None)
        ),
        "itl_p95_ms": _percentile(itl, 95.0),
        "itl_window_median_ms": statistics.median(itl) if itl else None,
        "itl_samples": len(itl),
        "output_sha256": result.get("output_sha256"),
        "generated_token_ids": result.get("generated_token_ids") or None,
        "answer_characters": result.get("answer_characters"),
        "thinking_characters": result.get("thinking_characters"),
        "route": shape.get("route"),
        **_route_fields(result),
        **_prefix_fields(result),
        # The backend's own prefill/decode split for this row, which separates
        # "the prompt cost this much" from "the client waited this long".
        "backend_timing": result.get("timing"),
        "mtp_used": mtp.get("used"),
        "mtp_output_tokens": mtp.get("mtp_output_tokens"),
        "ar_output_tokens": mtp.get("ar_output_tokens"),
        "mtp_coverage": accounting.get("mtp_coverage"),
        "first_fallback_position": mtp.get("first_fallback_position"),
        "fallback_reason": mtp.get("fallback_reason"),
        # The four refusal facts, kept apart from the folded ``fallback_reason``
        # above: which admission gate refused this row's prompt activation,
        # whether its draft provider existed at all, how wide the group it was
        # actually planned in was, and whether that group was planned
        # autoregressively.
        "activation_reason": execution.get("activation_reason"),
        "provider_readiness": execution.get("provider_readiness"),
        "provider_decline_reason": execution.get("provider_decline_reason"),
        "plan_group_rows": execution.get("plan_group_rows"),
        "plan_ar_only": execution.get("plan_ar_only"),
        "plan_reason": execution.get("plan_reason"),
        # Events, not tokens: one entry per non-speculative step or refusal. The
        # token-level attribution for the same reasons comes from
        # ``ar_output_tokens_by_reason`` when diagnostic spans were recorded.
        "fallback_event_counts": mtp.get("fallback_event_counts"),
        "fallback_event_total": mtp.get("fallback_event_total"),
        "selected_depth_histogram": mtp.get("selected_depth_histogram"),
        "draft_cycles": mtp.get("draft_cycles"),
        "accepted_draft_tokens": mtp.get("accepted_draft_tokens"),
        "ar_output_tokens_in_cycles": accounting.get(
            "ar_output_tokens_in_cycles"
        ),
        "mtp_output_tokens_explained_by_cycles": accounting.get(
            "mtp_output_tokens_explained_by_cycles"
        ),
        "unexplained_mtp_output_tokens": accounting.get(
            "unexplained_mtp_output_tokens"
        ),
        "reconciled": accounting.get("reconciled"),
        "reconciled_reasons": accounting.get("reconciled_reasons"),
        "span_accounting": accounting.get("span_accounting"),
        "ar_output_tokens_by_reason": (
            (accounting.get("span_accounting") or {}).get("ar_tokens_by_reason")
            if isinstance(accounting.get("span_accounting"), Mapping)
            else None
        ),
        "timeline": result.get("timeline"),
    }


def _rate_windows(timeline: Sequence[Sequence[float]], *, window: int = 16) -> list[dict[str, float]]:
    """Decode rate per window of committed tokens, from the chunk timeline."""

    points = [(float(ms), int(tokens)) for ms, tokens in (timeline or ())]
    if len(points) < 2:
        return []
    windows: list[dict[str, float]] = []
    start_index = 0
    while start_index < len(points) - 1:
        start_ms, start_tokens = points[start_index]
        end_index = min(start_index + window, len(points) - 1)
        end_ms, end_tokens = points[end_index]
        tokens = end_tokens - start_tokens
        seconds = (end_ms - start_ms) / 1000.0
        if tokens > 0 and seconds > 0:
            windows.append(
                {
                    "start_token": float(start_tokens),
                    "tokens_per_second": tokens / seconds,
                }
            )
        start_index = end_index
    return windows


def _speculation_case(row: Mapping[str, Any]) -> str:
    """Classify how far one request's speculation ran.

    ``first_fallback_position`` is ``0`` when speculation never covered a token,
    ``None`` when it covered every token after the prompt root (so the request
    ended inside speculation), and a middle value when the request left
    speculation mid-stream. Collapsing the last two into one empty block hid the
    difference between "never entered" and "stayed to the end".
    """

    if not row.get("mtp_used"):
        return "never_entered_speculation"
    position = row.get("first_fallback_position")
    if position is None:
        return "ended_inside_speculation"
    if int(position) <= 0:
        return "never_entered_speculation"
    return "left_mid_stream"


def _cliff(rows: Sequence[Mapping[str, Any]], *, window: int = 16) -> dict[str, Any]:
    """Quantify the latency step a request takes when it leaves speculation."""

    cases: Counter[str] = Counter(_speculation_case(row) for row in rows)
    measured: list[dict[str, Any]] = []
    for row in rows:
        if _speculation_case(row) != "left_mid_stream":
            continue
        position = int(row.get("first_fallback_position") or 0)
        timeline = row.get("timeline") or []
        if not timeline:
            continue
        windows = _rate_windows(timeline, window=window)
        before = [
            entry["tokens_per_second"]
            for entry in windows
            if entry["start_token"] < position - window
        ]
        after = [
            entry["tokens_per_second"]
            for entry in windows
            if entry["start_token"] >= position
        ]
        if not before or not after:
            continue
        measured.append(
            {
                "source_id": row.get("source_id"),
                "first_fallback_position": position,
                "completion_tokens": row.get("completion_tokens"),
                "before_tokens_per_second": statistics.median(before),
                "after_tokens_per_second": statistics.median(after),
                "ratio": statistics.median(after) / statistics.median(before),
                "rate_profile": windows,
            }
        )
    base: dict[str, Any] = {
        "case_counts": dict(sorted(cases.items())),
        "left_mid_stream_requests": cases.get("left_mid_stream", 0),
        "never_entered_speculation_requests": cases.get(
            "never_entered_speculation", 0
        ),
        "ended_inside_speculation_requests": cases.get(
            "ended_inside_speculation", 0
        ),
    }
    if not measured:
        # Say why there is no rate step instead of reporting an empty block that
        # reads like a measurement of zero.
        base["requests"] = 0
        base["note"] = (
            "no request both left speculation mid-stream and had rate windows on "
            "both sides of the crossing"
        )
        return base
    ratios = [entry["ratio"] for entry in measured]
    return {
        **base,
        "requests": len(measured),
        "median_before_tokens_per_second": statistics.median(
            entry["before_tokens_per_second"] for entry in measured
        ),
        "median_after_tokens_per_second": statistics.median(
            entry["after_tokens_per_second"] for entry in measured
        ),
        "median_ratio": statistics.median(ratios),
        "worst_ratio": min(ratios),
        "requests_slower_than_80pct_after_crossing": sum(
            1 for ratio in ratios if ratio < 0.8
        ),
        "detail": measured,
    }


def _serving_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Serving metrics for one slice of a run (one arm, one case, or a group).

    Throughput here is per-slice wall time, so a slice measured inside a run
    that also served other slices reads as a rate, not as a share of the run.
    """

    completed = [row for row in rows if not row.get("error")]
    wall = max((float(row.get("e2e_ms") or 0.0) for row in completed), default=0.0)
    tokens = sum(int(row.get("completion_tokens") or 0) for row in completed)
    itl = [
        float(row["itl_median_ms"])
        for row in completed
        if row.get("itl_median_ms") is not None
    ]
    ttft = [float(row["ttft_ms"]) for row in completed if row.get("ttft_ms")]
    return {
        "requests": len(rows),
        "completed_requests": len(completed),
        "failed_requests": sum(1 for row in rows if row.get("error")),
        "cancelled_requests": sum(1 for row in rows if row.get("cancelled")),
        "completion_tokens": tokens,
        "mtp_requests": sum(1 for row in completed if row.get("mtp_used")),
        "mtp_output_tokens": sum(
            int(row.get("mtp_output_tokens") or 0) for row in completed
        ),
        "prompt_tokens": sum(
            int(row.get("prompt_tokens") or row.get("prompt_tokens_local") or 0)
            for row in completed
        ),
        "median_prompt_tokens": (
            statistics.median(
                [
                    int(row.get("prompt_tokens") or row.get("prompt_tokens_local") or 0)
                    for row in completed
                ]
            )
            if completed
            else None
        ),
        "max_e2e_ms": wall or None,
        "decode_tokens_per_second": tokens / (wall / 1000.0) if wall else None,
        "completed_requests_per_second": len(completed) / (wall / 1000.0) if wall else None,
        "median_itl_ms": statistics.median(itl) if itl else None,
        "median_itl_window_ms": (
            statistics.median(
                [
                    float(value)
                    for row in completed
                    for value in [row.get("itl_window_median_ms")]
                    if value is not None
                ]
            )
            if any(row.get("itl_window_median_ms") is not None for row in completed)
            else None
        ),
        "median_ttft_ms": statistics.median(ttft) if ttft else None,
        "p95_ttft_ms": _percentile(ttft, 95.0),
    }


def _arm_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare the arms' greedy output for the prompts more than one arm served.

    The speculative route is a verifier: for the same prompt it must reproduce
    the autoregressive arm's tokens. A differing pair is a quality failure and
    is reported as one, with the shared-prefix length that localizes where the
    two streams part.
    """

    by_prompt: dict[tuple[Any, Any], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("error") or not row.get("route_arm"):
            continue
        # Keyed without the repeat index: with --prompt-repeats and a route mix
        # the arms alternate across repeats, so the same prompt is served by
        # each arm in a different repeat. The first occurrence per arm is the
        # comparison, and it is the same prompt either way.
        key = (row.get("source_id"), row.get("case"))
        arms = by_prompt.setdefault(key, {})
        arm = str(row["route_arm"])
        existing = arms.get(arm)
        if existing is None or int(row.get("repeat_index") or 0) < int(
            existing.get("repeat_index") or 0
        ):
            arms[arm] = row
    pairs: list[dict[str, Any]] = []
    for key, arms in sorted(by_prompt.items(), key=lambda item: str(item[0])):
        if len(arms) < 2:
            continue
        # The reference is the true-AR arm when the load carries one, so every
        # pair reads as "does the speculative arm reproduce the control".
        reference = "off" if "off" in arms else sorted(arms)[0]
        for name in sorted(arms):
            if name == reference:
                continue
            left, right = arms[reference], arms[name]
            left_name, right_name = reference, name
            left_ids = [int(value) for value in (left.get("generated_token_ids") or [])]
            right_ids = [int(value) for value in (right.get("generated_token_ids") or [])]
            shared = 0
            for a, b in zip(left_ids, right_ids):
                if a != b:
                    break
                shared += 1
            pairs.append(
                {
                    "source_id": key[0],
                    "case": key[1],
                    "left_repeat_index": left.get("repeat_index"),
                    "right_repeat_index": right.get("repeat_index"),
                    "left_arm": left_name,
                    "right_arm": right_name,
                    "left_tokens": len(left_ids),
                    "right_tokens": len(right_ids),
                    "ids_available": bool(left_ids and right_ids),
                    "shared_prefix_tokens": shared,
                    "identical": (
                        left_ids == right_ids
                        if left_ids and right_ids
                        else left.get("output_sha256") == right.get("output_sha256")
                    ),
                }
            )
    compared = [pair for pair in pairs if pair["ids_available"]]
    return {
        "compared_prompts": len(pairs),
        "compared_prompts_with_ids": len(compared),
        "identical_prompts": sum(1 for pair in pairs if pair["identical"]),
        "differing_prompts": sum(1 for pair in pairs if not pair["identical"]),
        "median_shared_prefix_tokens": (
            statistics.median([pair["shared_prefix_tokens"] for pair in compared])
            if compared
            else None
        ),
        "min_shared_prefix_tokens": (
            min(pair["shared_prefix_tokens"] for pair in compared) if compared else None
        ),
        "differing": [pair for pair in pairs if not pair["identical"]],
    }


def _group_rows(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        groups.setdefault(str(value), []).append(row)
    return groups


def summarize(rows: Sequence[Mapping[str, Any]], *, window: int = 16) -> dict[str, Any]:
    completion = sum(int(row.get("completion_tokens") or 0) for row in rows)
    mtp_outputs = sum(int(row.get("mtp_output_tokens") or 0) for row in rows)
    used = [row for row in rows if row.get("mtp_used")]
    reasons: Counter[str] = Counter()
    for row in rows:
        if row.get("mtp_used"):
            continue
        reason = row.get("fallback_reason") or row.get("route_decision_reason")
        reasons[str(reason or "unknown")] += 1
    prompt_buckets: dict[str, Counter[str]] = {}
    group_rows_histogram: dict[str, Counter[str]] = {}
    for row in rows:
        prompt_tokens = int(row.get("prompt_tokens") or row.get("prompt_tokens_local") or 0)
        bucket = (
            "0-511"
            if prompt_tokens < 512
            else "512-767"
            if prompt_tokens < 768
            else "768-1022"
            if prompt_tokens < 1023
            else "1023+"
        )
        prompt_buckets.setdefault(bucket, Counter())
        prompt_buckets[bucket]["requests"] += 1
        if row.get("mtp_used"):
            prompt_buckets[bucket]["mtp_requests"] += 1
        prompt_buckets[bucket]["mtp_output_tokens"] += int(
            row.get("mtp_output_tokens") or 0
        )
        prompt_buckets[bucket]["completion_tokens"] += int(
            row.get("completion_tokens") or 0
        )
        # Two refusals that a route-level share hides: a long prompt still
        # priming its provider refuses with prompt_activation_in_flight, and a
        # plan with no provider at all refuses with no_provider. Count both per
        # bucket, because the first pass showed long prompts losing the route
        # while mid-length ones kept it.
        events = row.get("fallback_event_counts") or {}
        if isinstance(events, Mapping):
            for reason in ("prompt_activation_in_flight", "no_provider"):
                count = int(events.get(reason) or 0)
                if count:
                    prompt_buckets[bucket][f"{reason}_requests"] += 1
                    prompt_buckets[bucket][f"{reason}_events"] += count
        reason = row.get("fallback_reason")
        if reason in {"prompt_activation_in_flight", "no_provider"}:
            prompt_buckets[bucket][f"primary_{reason}_requests"] += 1
        # Realized group composition: whether a request that reported a
        # speculative route was served at width 1 or inside a wider group is
        # the difference the concurrent-coverage work turns on.
        realized_rows = row.get("realized_group_rows")
        group_key = (
            "unknown"
            if realized_rows is None
            else str(int(realized_rows))
        )
        group_rows_histogram.setdefault(group_key, Counter())
        group_rows_histogram[group_key]["requests"] += 1
        group_rows_histogram[group_key]["completion_tokens"] += int(
            row.get("completion_tokens") or 0
        )
        if row.get("mtp_used"):
            group_rows_histogram[group_key]["mtp_requests"] += 1
            group_rows_histogram[group_key]["mtp_output_tokens"] += int(
                row.get("mtp_output_tokens") or 0
            )
    ttfts = [float(row["ttft_ms"]) for row in rows if row.get("ttft_ms")]
    latencies = [float(row["e2e_ms"]) for row in rows if row.get("e2e_ms")]
    # Inter-token latency across every decode interval in the run. A change
    # that raises aggregate throughput by batching but stretches each token is
    # not a serving win, so the tail is reported beside the median.
    itl_samples: list[float] = []
    for row in rows:
        itl_samples.extend(_inter_token_latencies_ms(row.get("timeline")))
    # The run's central inter-token latency is the per-request average, not the
    # pooled window median: a row whose response was fully buffered contributes
    # one fast window per token and would otherwise drag the pooled median down.
    row_itl = [
        float(row["itl_median_ms"])
        for row in rows
        if row.get("itl_median_ms") is not None
    ]
    completed_hashes = Counter(
        str(row["output_sha256"])
        for row in rows
        if row.get("output_sha256")
    )
    answers = [
        float(row["first_answer_ms"])
        for row in rows
        if row.get("first_answer_ms")
    ]
    thinking = [
        float(row["first_thinking_ms"])
        for row in rows
        if row.get("first_thinking_ms")
    ]
    accounted = [row for row in rows if row.get("reconciled") is not None]
    unreconciled = [
        {
            "source_id": row.get("source_id"),
            "reconciled_reasons": row.get("reconciled_reasons"),
        }
        for row in accounted
        if row.get("reconciled") is not True
    ]
    span_proofs = [
        row for row in rows if isinstance(row.get("span_accounting"), Mapping)
    ]
    event_total = sum(int(row.get("fallback_event_total") or 0) for row in rows)
    # Events are not tokens. A planning refusal that is retried on the next step
    # costs one autoregressive step and no route, so an event count alone reads
    # as a lost route when the request was in fact almost entirely speculative.
    # The spanned token attribution is the quantity that answers "what did this
    # refusal cost", so both are reported.
    ar_tokens_by_reason: Counter[str] = Counter()
    for row in rows:
        attributed = row.get("ar_output_tokens_by_reason")
        if isinstance(attributed, Mapping):
            for reason, count in attributed.items():
                ar_tokens_by_reason[str(reason)] += int(count or 0)
    # Per-request decode rate after the first token. This is the quantity a
    # reader needs to compare MTP against AR per request; the aggregate rate in
    # ``decode_tokens_per_second`` also carries prefill and queueing, so a
    # route change that only moves prefill would otherwise look like a decode
    # change. Concurrent requests overlap, so the median is per request and the
    # aggregate stays the wall-clock number.
    per_request_decode: list[float] = []
    for row in rows:
        row_completion = int(row.get("completion_tokens") or 0)
        e2e = row.get("e2e_ms")
        ttft = row.get("ttft_ms")
        if row_completion <= 0 or e2e is None or ttft is None:
            continue
        decode_seconds = (float(e2e) - float(ttft)) / 1000.0
        if decode_seconds > 0:
            per_request_decode.append(row_completion / decode_seconds)
    return {
        "requests": len(rows),
        "cancelled_requests": sum(1 for row in rows if row.get("cancelled")),
        "mtp_requests": len(used),
        "mtp_request_share": (len(used) / len(rows)) if rows else 0.0,
        "completion_tokens": completion,
        "mtp_output_tokens": mtp_outputs,
        "mtp_output_share": (mtp_outputs / completion) if completion else 0.0,
        "ar_output_tokens": sum(
            int(row.get("ar_output_tokens") or 0) for row in rows
        ),
        # Attribution gate. ``reconciled`` is not arithmetic consistency: it is
        # decided from the committed cycle counts and, in diagnostic mode, from
        # the committed output spans tiling the emitted output.
        "accounted_requests": len(accounted),
        "reconciled_requests": sum(
            1 for row in accounted if row.get("reconciled") is True
        ),
        "unreconciled_requests": unreconciled,
        "span_proof_requests": len(span_proofs),
        "span_proof_failures": [
            {
                "source_id": row.get("source_id"),
                "reconciled_reasons": (row.get("span_accounting") or {}).get(
                    "reconciled_reasons"
                ),
            }
            for row in span_proofs
            if (row.get("span_accounting") or {}).get("reconciled") is not True
        ],
        # Event counts stay separate from token counts: this is the number of
        # non-speculative steps and refusals, which is a different quantity from
        # ``ar_output_tokens`` and from ``mtp_output_tokens``.
        "fallback_event_total": event_total,
        "non_mtp_reasons": dict(sorted(reasons.items())),
        # Token-level attribution of the same refusals, from the committed output
        # spans (present when the server runs with HIPENGINE_MTP2_OUTPUT_SPANS=1).
        "ar_tokens_by_reason": dict(sorted(ar_tokens_by_reason.items())),
        "ar_tokens_attributed": sum(ar_tokens_by_reason.values()),
        "prompt_token_buckets": {
            bucket: dict(sorted(counts.items()))
            for bucket, counts in sorted(prompt_buckets.items())
        },
        # Realized physical group width per request, with the MTP share inside
        # each width. At c>1 this is where a route that is reported as
        # speculative while its rows run at width 1 shows up.
        "groups_by_realized_rows": {
            group: dict(sorted(counts.items()))
            for group, counts in sorted(group_rows_histogram.items())
        },
        # Per-slice serving metrics: one entry per requested arm and one per
        # prompt-length case, so the true-AR control and the boundary-crossing
        # row are readable without re-deriving them from the row list.
        "by_route_arm": {
            str(arm): _serving_metrics(group)
            for arm, group in sorted(_group_rows(rows, "route_arm").items())
        },
        "by_case": {
            str(case): _serving_metrics(group)
            for case, group in sorted(_group_rows(rows, "case").items())
        },
        # Prefix-cache hit beside miss, from the backend's own per-request
        # telemetry rather than inferred from a shorter TTFT.
        "by_prefix": {
            key: _serving_metrics(group)
            for key, group in (
                ("hit", [row for row in rows if row.get("prefix_hit") is True]),
                ("miss", [row for row in rows if row.get("prefix_hit") is False]),
            )
            if group
        },
        # Prefix-cache hit beside miss. With --prompt-repeats the rows after the
        # first for a prompt reuse its prefix; this splits the run by that fact
        # so a hit's rate is not read as a miss's.
        "by_repeat": {
            ("miss" if int(repeat) == 0 else f"hit-{repeat}"): _serving_metrics(group)
            for repeat, group in sorted(
                _group_rows(rows, "repeat_index").items(), key=lambda item: int(item[0])
            )
        },
        # Per-turn slice of a multi-turn load. Turn 0 has no prefix to reuse and
        # every later turn shares the turn before it, so this is where a
        # prefix-cache hit and its interaction with the route show up together.
        "by_turn": {
            str(turn): _serving_metrics(group)
            for turn, group in sorted(
                _group_rows(rows, "turn_index").items(), key=lambda item: int(item[0])
            )
        },
        "arm_identity": _arm_identity(rows),
        "median_decode_tokens_per_second_per_request": (
            statistics.median(per_request_decode) if per_request_decode else None
        ),
        "decode_requests": len(per_request_decode),
        # Output identity, so two arms of the same load can be compared for
        # greedy-exactness. ``duplicate_output_hashes`` above one means the
        # load contains repeated outputs (or a prefix-cache replay); it is not
        # an error, only context for reading the digest list.
        "output_hashes": dict(sorted(completed_hashes.items())),
        "duplicate_output_hashes": sum(
            count - 1 for count in completed_hashes.values() if count > 1
        ),
        # ttft_ms is time to the first generated token of either channel, which
        # for a thinking reply is a reasoning token.
        "median_ttft_ms": statistics.median(ttfts) if ttfts else None,
        "p95_ttft_ms": _percentile(ttfts, 95.0),
        "ttft_requests": len(ttfts),
        # Total request latency, end to end, including prefill and queueing.
        "median_e2e_ms": statistics.median(latencies) if latencies else None,
        "p95_e2e_ms": _percentile(latencies, 95.0),
        "total_e2e_ms": sum(latencies),
        "median_itl_ms": statistics.median(row_itl) if row_itl else None,
        "p95_itl_ms": _percentile(itl_samples, 95.0),
        "itl_samples": len(itl_samples),
        "itl_requests": len(row_itl),
        "median_itl_window_ms": (
            statistics.median(itl_samples) if itl_samples else None
        ),
        "median_first_answer_ms": statistics.median(answers) if answers else None,
        "answer_requests": len(answers),
        "median_first_thinking_ms": (
            statistics.median(thinking) if thinking else None
        ),
        "thinking_requests": len(thinking),
        "thinking_output_share": (
            sum(1 for row in rows if row.get("first_thinking_ms")) / len(rows)
            if rows
            else 0.0
        ),
        # Character split, so a reader can see how much of the output was
        # reasoning rather than answer text.
        "answer_characters": sum(int(row.get("answer_characters") or 0) for row in rows),
        "thinking_characters": sum(
            int(row.get("thinking_characters") or 0) for row in rows
        ),
        "thinking_character_share": (
            sum(int(row.get("thinking_characters") or 0) for row in rows)
            / max(
                1,
                sum(int(row.get("thinking_characters") or 0) for row in rows)
                + sum(int(row.get("answer_characters") or 0) for row in rows),
            )
        ),
        # Realized vs planned route, per request. A pass where every request
        # reports effective_route=ar while selected_route=speculative_mtp is a
        # route-level refusal, not a lying route.
        "effective_routes": dict(
            sorted(Counter(str(row.get("effective_route")) for row in rows).items())
        ),
        "selected_routes": dict(
            sorted(Counter(str(row.get("selected_route")) for row in rows).items())
        ),
        "route_disagreements": [
            {
                "source_id": row.get("source_id"),
                "effective_route": row.get("effective_route"),
                "selected_route": row.get("selected_route"),
                "route_decision_reason": row.get("route_decision_reason"),
                "k0_class": row.get("k0_class"),
                "fallback_reason": row.get("fallback_reason"),
            }
            for row in rows
            if row.get("selected_route")
            and row.get("effective_route")
            and row.get("selected_route") != row.get("effective_route")
        ],
        "cliff": _cliff(rows, window=window),
    }


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} tok/s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--gguf", default=DEFAULT_GGUF, type=Path)
    parser.add_argument("--num-prompts", type=int, default=24)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help=(
            "Override the dataset's expected output length (tokens). "
            "0 or a negative value uses the dataset's own answer length."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Omit to use the server default (hipEngine defaults to greedy).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompt-len", type=int, default=MAX_PROMPT_LEN)
    parser.add_argument(
        "--min-prompt-len",
        type=int,
        default=MIN_LEN,
        help=(
            "Minimum prompt length in tokens. Raise it to select the "
            "boundary-crossing row that the default 4-token floor excludes."
        ),
    )
    parser.add_argument("--max-total-len", type=int, default=MAX_TOTAL_LEN)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--label", default="sharegpt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--speculative-mtp",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Per-request speculative_mtp field: on sends true, off sends false "
            "(the true-AR control), auto omits it and keeps the server policy. "
            "--route-mix overrides this per request inside one run."
        ),
    )
    parser.add_argument(
        "--route-mix",
        default=None,
        help=(
            "Comma-separated arms assigned per request in submission order, "
            "for example 'off,auto'. The true-AR control and the speculative "
            "arm then run in one load under the same conditions, instead of in "
            "two runs on two servers."
        ),
    )
    parser.add_argument(
        "--cases",
        default=None,
        help=(
            "Comma-separated prompt-length cases selected in one load: any of "
            "short (4-64 tokens), mid (65-511), long (512-1024), or beyond "
            "(1025-4096, which also needs --max-prompt-len raised and a server "
            "whose autoregressive prefill accepts that context) for the row the "
            "provider's window refuses. Each row is labelled and the "
            "summary splits by case, so a healthy short request is measured "
            "beside the boundary-crossing row."
        ),
    )
    parser.add_argument(
        "--record-ids",
        action="store_true",
        help=(
            "Keep the generated token ids on every row and report "
            "arm_identity: whether the arms produced identical greedy output "
            "for the prompts more than one arm served, and the shared-prefix "
            "length where they differ."
        ),
    )
    parser.add_argument(
        "--stagger-ms",
        type=float,
        default=0.0,
        help=(
            "Delay between submissions so requests arrive while others decode "
            "(changing occupancy). 0 submits the whole load at once."
        ),
    )
    parser.add_argument(
        "--prompt-repeats",
        type=int,
        default=1,
        help=(
            "Send every prompt this many times. Rows after the first reuse the "
            "prompt prefix, so a server with prefix caching enabled shows the "
            "hit path beside the miss path. With ``--route-mix`` the arm is "
            "assigned by submission-index parity over the flattened repeat "
            "list, so an even ``--num-prompts`` gives each prompt the same arm "
            "in every repeat and ``arm_identity`` compares nothing; use an odd "
            "prompt count to put the same prompt on both arms."
        ),
    )
    parser.add_argument(
        "--cancel-count",
        type=int,
        default=0,
        help=(
            "Cancel this many of the last measured requests after their first "
            "generated token, so the server's cancel/refill path runs beside "
            "the completed requests. Cancelled rows are reported separately "
            "from failures."
        ),
    )
    parser.add_argument(
        "--cancel-after-tokens",
        type=int,
        default=4,
        help="Generated tokens to wait for before cancelling a request.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Requests to run (and discard) before the measured load.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=1,
        help=(
            "Expand each dataset conversation into this many growing turns, so "
            "every turn after the first shares a token-exact prefix with the "
            "turn before it and a prefix cache has something to reuse. "
            "--num-prompts then counts conversations and the load holds "
            "num_prompts * turns requests; --route-mix assigns the arm per "
            "conversation so both arms submit identical prompt sequences."
        ),
    )
    args = parser.parse_args()
    if args.prompt_repeats < 1:
        parser.error("--prompt-repeats must be positive")
    if args.turns < 1:
        parser.error("--turns must be positive")
    if args.cancel_count < 0 or args.cancel_count > args.num_prompts * args.prompt_repeats:
        parser.error("--cancel-count must be between 0 and the measured load size")
    route_mix: list[str] = []
    if args.route_mix:
        route_mix = [arm.strip() for arm in args.route_mix.split(",") if arm.strip()]
        invalid = [arm for arm in route_mix if arm not in {"auto", "on", "off"}]
        if invalid:
            parser.error(f"--route-mix has unknown arms: {invalid}")
    cases: list[str] = []
    if args.cases:
        cases = [case.strip() for case in args.cases.split(",") if case.strip()]
        invalid = [case for case in cases if case not in CASE_BOUNDS]
        if invalid:
            parser.error(f"--cases has unknown cases: {invalid}")
    speculative_mtp = {"auto": None, "on": True, "off": False}[args.speculative_mtp]

    tokenizer = _tokenizer(args.gguf)
    output_len = args.output_len if (args.output_len or 0) > 0 else None
    samples = load_samples(
        args.dataset,
        tokenizer=tokenizer,
        count=args.num_prompts + args.warmup,
        seed=args.seed,
        output_len=output_len,
        max_prompt_len=args.max_prompt_len,
        max_total_len=args.max_total_len,
        min_prompt_len=args.min_prompt_len,
        cases=cases,
        turns=args.turns,
    )
    warmup, measured = samples[: args.warmup], samples[args.warmup :]
    if args.turns > 1 and args.warmup:
        # A multi-turn warmup is whole conversations: a warmup that stopped
        # mid-conversation would leave its later turns in the measured load.
        conversations = {
            sample.get("conversation_id") for sample in warmup
        }
        measured = [
            sample
            for sample in samples
            if sample.get("conversation_id") not in conversations
        ]
    # Repeat rows keep the prompt and differ only in arrival order, which is
    # what a prefix-cache hit is: the same prefix, submitted again.
    measured = [
        {**sample, "repeat_index": repeat}
        for repeat in range(args.prompt_repeats)
        for sample in measured
    ]
    # Assign the arm per request, cycling in submission order. A mix makes the
    # true-AR control and the speculative arm part of one load: same process,
    # same occupancy history, same cache state.
    if args.turns > 1:
        # A conversation is one arm's whole prompt sequence. Assigning per
        # request would split a conversation across the arms and compare an
        # early turn against a later one.
        arm_by_conversation: dict[Any, str] = {}
        for sample in measured:
            conversation_id = sample.get("conversation_id")
            if conversation_id in arm_by_conversation:
                continue
            arm_by_conversation[conversation_id] = (
                route_mix[len(arm_by_conversation) % len(route_mix)]
                if route_mix
                else args.speculative_mtp
            )
        for sample in measured:
            sample["route_arm"] = arm_by_conversation[sample.get("conversation_id")]
    else:
        for index, sample in enumerate(measured):
            sample["route_arm"] = (
                route_mix[index % len(route_mix)] if route_mix else args.speculative_mtp
            )
    cancel_ids = {
        id(sample) for sample in measured[len(measured) - args.cancel_count :]
    }
    print(
        f"[{args.label}] {len(measured)} prompts, concurrency {args.max_concurrency}, "
        f"output_len={output_len or 'dataset'}, "
        f"speculative_mtp={args.speculative_mtp}, "
        f"route_mix={args.route_mix or '-'}, cases={args.cases or '-'}, "
        f"repeats={args.prompt_repeats}, stagger_ms={args.stagger_ms}, "
        f"turns={args.turns}, "
        f"temperature={args.temperature if args.temperature is not None else 'server-default'}"
    )

    def run(sample: Mapping[str, Any]) -> dict[str, Any]:
        cancel = id(sample) in cancel_ids
        arm = str(sample.get("route_arm") or args.speculative_mtp)
        arm_value = {"auto": None, "on": True, "off": False}[arm]
        try:
            result = _stream_request(
                args.server_url,
                model=args.model,
                prompt=str(sample["prompt"]),
                max_tokens=int(sample["expected_output_tokens"]),
                temperature=args.temperature,
                timeout=args.request_timeout,
                speculative_mtp=arm_value,
                cancel_after_tokens=(
                    args.cancel_after_tokens if cancel else None
                ),
            )
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")[:400]
            return {
                "source_id": sample.get("source_id"),
                "case": sample.get("case"),
                "route_arm": sample.get("route_arm"),
                "prompt_tokens_local": sample.get("prompt_tokens"),
                "expected_output_tokens": sample.get("expected_output_tokens"),
                "repeat_index": sample.get("repeat_index", 0),
                "speculative_mtp_request": arm,
                "cancelled": False,
                "error": f"HTTP {error.code}: {body}",
            }
        row = _request_row(sample, result)
        row["repeat_index"] = sample.get("repeat_index", 0)
        row["speculative_mtp_request"] = arm
        if not args.record_ids:
            # Ids are ~4 bytes per token; a 24-prompt suite is small but the
            # per-chunk timelines are not, so keep them opt-in.
            row.pop("generated_token_ids", None)
        return row

    for sample in warmup:
        run(sample)

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_concurrency
    ) as pool:
        if args.stagger_ms > 0:
            futures = []
            for sample in measured:
                futures.append(pool.submit(run, sample))
                time.sleep(args.stagger_ms / 1000.0)
            rows = [future.result() for future in futures]
        else:
            rows = list(pool.map(run, measured))
    wall = time.perf_counter() - started

    failures = [
        row for row in rows if row.get("error") and not row.get("cancelled")
    ]
    cancelled = [row for row in rows if row.get("cancelled")]
    good = [row for row in rows if not row.get("error")]
    report = {
        "label": args.label,
        "server_url": args.server_url,
        "model": args.model,
        "dataset": str(args.dataset),
        "num_prompts": len(measured),
        "max_concurrency": args.max_concurrency,
        "output_len": output_len,
        "temperature": args.temperature,
        "speculative_mtp_request": args.speculative_mtp,
        "route_mix": route_mix or None,
        "cases": cases or None,
        "record_ids": bool(args.record_ids),
        "prompt_repeats": args.prompt_repeats,
        "turns": args.turns,
        "stagger_ms": args.stagger_ms,
        "min_prompt_len": args.min_prompt_len,
        "cancel_count": args.cancel_count,
        "seed": args.seed,
        "wall_seconds": wall,
        "failed_requests": len(failures),
        "cancelled_requests": len(cancelled),
        "summary": summarize(good, window=args.window),
        "rows": rows,
    }
    report["summary"]["decode_tokens_per_second"] = (
        sum(int(row.get("completion_tokens") or 0) for row in good) / wall
        if wall
        else None
    )
    report["summary"]["completed_requests_per_second"] = (
        len(good) / wall if wall else None
    )
    report["summary"]["request_error_rate"] = (
        len(failures) / len(rows) if rows else 0.0
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = report["summary"]
    print(
        f"[{args.label}] requests={summary['requests']} failures={len(failures)} "
        f"cancelled={len(cancelled)} "
        f"mtp_requests={summary['mtp_requests']} "
        f"({summary['mtp_request_share']:.1%}) "
        f"mtp_output_share={summary['mtp_output_share']:.1%} "
        f"completed={summary['completed_requests_per_second']:.2f} req/s "
        f"decode={summary['decode_tokens_per_second']:.2f} tok/s "
        f"wall={wall:.1f}s"
    )
    print(
        f"[{args.label}] latency: ttft(med/p95)="
        f"{_fmt_ms(summary['median_ttft_ms'])}/{_fmt_ms(summary['p95_ttft_ms'])} "
        f"itl(med/p95)="
        f"{_fmt_ms(summary['median_itl_ms'])}/{_fmt_ms(summary['p95_itl_ms'])} "
        f"e2e(med/p95)="
        f"{_fmt_ms(summary['median_e2e_ms'])}/{_fmt_ms(summary['p95_e2e_ms'])}"
    )
    print(f"[{args.label}] non-MTP reasons: {json.dumps(summary['non_mtp_reasons'])}")
    for arm, metrics in summary["by_route_arm"].items():
        print(
            f"[{args.label}] arm {arm}: requests={metrics['requests']} "
            f"completed={metrics['completed_requests']} "
            f"decode={_fmt_rate(metrics['decode_tokens_per_second'])} "
            f"itl(avg/med-window)={_fmt_ms(metrics['median_itl_ms'])}/"
            f"{_fmt_ms(metrics['median_itl_window_ms'])} "
            f"ttft(med/p95)={_fmt_ms(metrics['median_ttft_ms'])}/"
            f"{_fmt_ms(metrics['p95_ttft_ms'])} "
            f"mtp_requests={metrics['mtp_requests']} "
            f"mtp_output_tokens={metrics['mtp_output_tokens']} "
            f"median_prompt_tokens={metrics['median_prompt_tokens']}"
        )
    for case, metrics in summary["by_case"].items():
        print(
            f"[{args.label}] case {case}: requests={metrics['requests']} "
            f"completed={metrics['completed_requests']} "
            f"decode={_fmt_rate(metrics['decode_tokens_per_second'])} "
            f"itl(avg)={_fmt_ms(metrics['median_itl_ms'])} "
            f"ttft(med)={_fmt_ms(metrics['median_ttft_ms'])} "
            f"median_prompt_tokens={metrics['median_prompt_tokens']} "
            f"mtp_requests={metrics['mtp_requests']}"
        )
    for repeat, metrics in summary["by_repeat"].items():
        print(
            f"[{args.label}] prefix {repeat}: requests={metrics['requests']} "
            f"decode={_fmt_rate(metrics['decode_tokens_per_second'])} "
            f"itl(avg)={_fmt_ms(metrics['median_itl_ms'])} "
            f"ttft(med)={_fmt_ms(metrics['median_ttft_ms'])} "
            f"mtp_requests={metrics['mtp_requests']} "
            f"median_prompt_tokens={metrics['median_prompt_tokens']}"
        )
    for hit, metrics in summary["by_prefix"].items():
        print(
            f"[{args.label}] cache {hit}: requests={metrics['requests']} "
            f"decode={_fmt_rate(metrics['decode_tokens_per_second'])} "
            f"itl(avg)={_fmt_ms(metrics['median_itl_ms'])} "
            f"ttft(med/p95)={_fmt_ms(metrics['median_ttft_ms'])}/"
            f"{_fmt_ms(metrics['p95_ttft_ms'])} "
            f"mtp_requests={metrics['mtp_requests']} "
            f"median_prompt_tokens={metrics['median_prompt_tokens']}"
        )
    for turn, metrics in summary["by_turn"].items():
        print(
            f"[{args.label}] turn {turn}: requests={metrics['requests']} "
            f"decode={_fmt_rate(metrics['decode_tokens_per_second'])} "
            f"ttft(med)={_fmt_ms(metrics['median_ttft_ms'])} "
            f"mtp_requests={metrics['mtp_requests']} "
            f"median_prompt_tokens={metrics['median_prompt_tokens']}"
        )
    identity = summary["arm_identity"]
    if identity["compared_prompts"]:
        print(
            f"[{args.label}] arm identity: compared={identity['compared_prompts']} "
            f"identical={identity['identical_prompts']} "
            f"differing={identity['differing_prompts']} "
            f"shared_prefix(med/min)="
            f"{identity['median_shared_prefix_tokens']}/"
            f"{identity['min_shared_prefix_tokens']}"
        )
        if identity["differing_prompts"]:
            print(
                f"[{args.label}] QUALITY: the arms disagree on greedy output for "
                f"{identity['differing_prompts']} prompt(s): "
                f"{json.dumps(identity['differing'][:5])}"
            )
    print(
        f"[{args.label}] groups by realized rows: "
        f"{json.dumps(summary['groups_by_realized_rows'])}"
    )
    if summary.get("median_decode_tokens_per_second_per_request") is not None:
        print(
            f"[{args.label}] median per-request decode "
            f"{summary['median_decode_tokens_per_second_per_request']:.2f} tok/s "
            f"over {summary['decode_requests']} requests"
        )
    print(
        f"[{args.label}] refusal cost: events={summary['fallback_event_total']} "
        f"spanned_ar_tokens={summary['ar_tokens_attributed']} "
        f"by_reason={json.dumps(summary['ar_tokens_by_reason'])} "
        f"(of {summary['ar_output_tokens']} autoregressive tokens)"
    )
    print(
        f"[{args.label}] attribution: "
        f"reconciled={summary['reconciled_requests']}/{summary['accounted_requests']} "
        f"span_proofs={summary['span_proof_requests']} "
        f"fallback_events={summary['fallback_event_total']}"
    )
    if summary["unreconciled_requests"]:
        print(
            f"[{args.label}] UNRECONCILED: "
            f"{json.dumps(summary['unreconciled_requests'])}"
        )
    if summary["span_proof_failures"]:
        print(
            f"[{args.label}] SPAN PROOF FAILURES: "
            f"{json.dumps(summary['span_proof_failures'])}"
        )
    elif not summary["span_proof_requests"]:
        print(
            f"[{args.label}] note: no committed output spans were reported; "
            "start the server with HIPENGINE_MTP2_OUTPUT_SPANS=1 to record "
            "the per-span attribution proof"
        )
    print(
        f"[{args.label}] first-token channels: "
        f"ttft={_fmt_ms(summary['median_ttft_ms'])} "
        f"over {summary['ttft_requests']} requests, "
        f"first_answer={_fmt_ms(summary['median_first_answer_ms'])} "
        f"over {summary['answer_requests']}, "
        f"first_thinking={_fmt_ms(summary['median_first_thinking_ms'])} "
        f"over {summary['thinking_requests']}"
    )
    print(
        f"[{args.label}] output characters: answer={summary['answer_characters']} "
        f"thinking={summary['thinking_characters']} "
        f"({summary['thinking_character_share']:.1%} reasoning)"
    )
    print(
        f"[{args.label}] routes: effective={json.dumps(summary['effective_routes'])} "
        f"selected={json.dumps(summary['selected_routes'])}"
    )
    if summary["route_disagreements"]:
        print(
            f"[{args.label}] ROUTE DISAGREEMENTS "
            f"({len(summary['route_disagreements'])}): "
            f"{json.dumps(summary['route_disagreements'][:5])}"
        )
    buckets = summary["prompt_token_buckets"]
    refused = {
        bucket: counts
        for bucket, counts in buckets.items()
        if any(
            key.endswith("_requests") and key.startswith(("prompt_activation", "no_provider"))
            for key in counts
        )
    }
    if refused:
        print(
            f"[{args.label}] route refusals by prompt bucket: "
            f"{json.dumps(refused)}"
        )
    cliff = summary["cliff"]
    print(
        f"[{args.label}] speculation coverage: "
        f"never_entered={cliff['never_entered_speculation_requests']} "
        f"left_mid_stream={cliff['left_mid_stream_requests']} "
        f"ended_inside={cliff['ended_inside_speculation_requests']}"
    )
    if cliff.get("requests"):
        print(
            f"[{args.label}] cliff over {cliff['requests']} requests: "
            f"before={cliff['median_before_tokens_per_second']:.2f} "
            f"after={cliff['median_after_tokens_per_second']:.2f} tok/s "
            f"median_ratio={cliff['median_ratio']:.2f} "
            f"slower_than_80pct={cliff['requests_slower_than_80pct_after_crossing']}"
        )
    elif cliff.get("note"):
        print(f"[{args.label}] cliff: {cliff['note']}")
    if failures:
        print(
            f"[{args.label}] failures: "
            f"{json.dumps([{k: row.get(k) for k in ('source_id', 'error', 'error_code', 'error_status_code', 'effective_route', 'selected_route')} for row in failures[:5]])}"
        )
    print(f"[{args.label}] wrote {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
