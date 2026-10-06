#!/usr/bin/env python3
"""Live acceptance for MTP logprob metadata: the value must be the AR value.

A speculative cycle publishes several tokens at once, so the metadata the OpenAI
schema wants for each of them has to come from the verified row that predicted
it. The failure this gate guards against is a route that reports *something*
plausible -- a point mass, a re-softmax over a truncated support, or a value read
from the wrong row -- which a shape check cannot see. It therefore compares
against the same request served as true autoregressive decoding on the same
server and the same KV cell, so the only difference between the arms is the
route.

Checks:

* ``greedy_metadata`` -- a ``temperature=0`` request with ``logprobs`` publishes
  the same tokens on both arms and, for every
  position, the same selected logprob, the same top-k token ids, and the same
  top-k logprobs. A greedy row reports the full-support softmax of the processed
  logits, so a route that reported ``log(1)`` from a point-mass support, or one
  that scored a token against the wrong row, fails here.
* ``sampled_metadata`` -- the same comparison for a sampling request, whose rows
  report the retained support's probability for the token the draw came from.
  Sampling arms consume the RNG stream differently, so the two arms are compared
  position by position up to their first token divergence and the gate requires a
  minimum shared prefix; the shared prefix is what proves the value, and its
  length is recorded rather than assumed.
* ``stream_metadata`` -- a streaming request carries per-token logprobs in its
  chunks and those values equal the non-streaming ones for the same request, so
  the stream path is not a second, silently different implementation. The
  streaming response does not repeat the admission summary, so this check
  compares against the non-streaming speculative arm and reads a mid-stream
  refusal as a failure of the streaming arm.
* ``chat_stream_metadata`` -- the chat surface applies its own reasoning splitter
  and logprob validator to the same chunks, so its streamed values are compared
  against its own blocking response, which also has to show that the request
  speculated.
* ``route_ran`` -- the speculative arm actually speculated, and the
  autoregressive arm did not, so neither comparison is between two AR runs.

Usage:
    .venv/bin/python scripts/mtp_logprobs_gate.py \
        --base-url http://127.0.0.1:18198 --model <served-name> \
        --json benchmarks/results/<date>-<cell>-mtp-logprobs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

# Repetitive text keeps the draft chains long, so most cycles publish more than
# one token and the metadata has to be right for a multi-token publication.
PROMPT = (
    "Count slowly, one number per word: one two three four five six seven "
    "eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty"
)

# The envelope a compared logprob must fall inside. Both arms compute the same
# float64 softmax of the same logits row, so they should agree far more closely;
# the tolerance is the production numeric envelope rather than a fitted value.
CHAT_PROMPT = "Say exactly: hello there my friend"

LOGPROB_TOLERANCE = 1e-3


def run_completion(
    client: httpx.Client,
    args: argparse.Namespace,
    *,
    speculative: bool,
    stream: bool = False,
    max_tokens: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        # ``/v1/completions`` uses the legacy spelling: an integer ``logprobs``
        # is both "report metadata" and the top-k width, and the endpoint has no
        # separate ``top_logprobs`` field.
        "logprobs": int(args.top_logprobs),
        "speculative_mtp": bool(speculative),
        "seed": int(args.seed),
        **extra,
    }
    if not stream:
        response = client.post("/v1/completions", json=payload)
        body = response.json()
        if response.status_code != 200:
            return {"status": response.status_code, "error": body.get("error"), "request": payload}
        choice = body["choices"][0]
        summary = body.get("hipengine", {}).get("speculative_mtp") or {}
        choice_engine = choice.get("hipengine") or {}
        return {
            "status": 200,
            "ids": [int(token) for token in choice_engine.get("generated_token_ids") or []],
            "logprobs": _normalize_logprobs(choice.get("logprobs")),
            "used": bool(summary.get("used")),
            "effective_route": summary.get("effective_route"),
            "draft_cycles": int(summary.get("draft_cycles", 0) or 0),
            "mtp_output_tokens": int(summary.get("mtp_output_tokens", 0) or 0),
            "request": payload,
        }

    ids: list[int] = []
    logprobs: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    with client.stream("POST", "/v1/completions", json={**payload, "stream": True}) as response:
        if response.status_code != 200:
            response.read()
            return {"status": response.status_code, "error": response.text, "request": payload}
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                # A mid-stream refusal still delivers a 200 response, so the arm
                # has to report it rather than collect a truncated prefix.
                return {
                    "status": 200,
                    "error": chunk["error"],
                    "ids": ids,
                    "logprobs": logprobs,
                    "used": bool(summary.get("used")),
                    "effective_route": summary.get("effective_route"),
                    "draft_cycles": int(summary.get("draft_cycles", 0) or 0),
                    "mtp_output_tokens": int(summary.get("mtp_output_tokens", 0) or 0),
                    "request": payload,
                }
            summary = (chunk.get("hipengine") or {}).get("speculative_mtp") or summary
            for choice in chunk.get("choices") or ():
                # ``/v1/completions`` reports metadata on the choice itself; the
                # chat endpoint nests it under ``delta``.
                payload_logprobs = choice.get("logprobs") or (
                    choice.get("delta") or {}
                ).get("logprobs")
                if not payload_logprobs:
                    continue
                tokens = payload_logprobs.get("tokens") or []
                for index, token_text in enumerate(tokens):
                    logprobs.append(
                        {
                            "token": token_text,
                            "logprob": (payload_logprobs.get("token_logprobs") or [None])[index],
                            "top": _top_pairs(payload_logprobs, index),
                        }
                    )
                ids.extend(
                    int(token)
                    for token in (chunk.get("hipengine") or {}).get("generated_token_ids") or ()
                )
    return {
        "status": 200,
        "ids": ids,
        "logprobs": logprobs,
        "used": bool(summary.get("used")),
        "effective_route": summary.get("effective_route"),
        "draft_cycles": int(summary.get("draft_cycles", 0) or 0),
        "mtp_output_tokens": int(summary.get("mtp_output_tokens", 0) or 0),
        "request": payload,
    }


def _top_pairs(payload_logprobs: dict[str, Any], index: int) -> list[list[Any]]:
    entry = (payload_logprobs.get("top_logprobs") or [None])[index]
    if not isinstance(entry, dict):
        return []
    return [[str(token), float(value)] for token, value in entry.items()]


def _normalize_logprobs(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    tokens = payload.get("tokens") or []
    values = payload.get("token_logprobs") or []
    return [
        {
            "token": str(token),
            "logprob": values[index] if index < len(values) else None,
            "top": _top_pairs(payload, index),
        }
        for index, token in enumerate(tokens)
    ]


def compare_prefix(
    mtp: dict[str, Any],
    ar: dict[str, Any],
    where: dict[str, Any],
    *,
    minimum_shared: int,
    require_same_tokens: bool,
) -> dict[str, Any]:
    """Compare the two arms position by position over their shared token prefix."""

    assert mtp["status"] == 200, {"speculative_arm_refused": {**where, "arm": mtp}}
    assert ar["status"] == 200, {"autoregressive_arm_refused": {**where, "arm": ar}}
    assert mtp["used"], {"speculative_arm_did_not_speculate": {**where, "arm": mtp}}
    assert mtp["draft_cycles"] > 0, {"speculative_arm_ran_no_cycles": {**where, "arm": mtp}}
    assert not ar["used"], {"autoregressive_arm_speculated": {**where, "arm": ar}}
    assert mtp["logprobs"], {"speculative_arm_reported_no_logprobs": {**where, "arm": mtp}}
    assert ar["logprobs"], {"autoregressive_arm_reported_no_logprobs": {**where, "arm": ar}}
    assert len(mtp["logprobs"]) == len(mtp["ids"]), {
        "metadata_does_not_cover_every_published_token": {
            **where,
            "published": len(mtp["ids"]),
            "reported": len(mtp["logprobs"]),
        },
    }
    assert len(ar["logprobs"]) == len(ar["ids"]), {
        "autoregressive_metadata_does_not_cover_every_published_token": {
            **where,
            "published": len(ar["ids"]),
            "reported": len(ar["logprobs"]),
        },
    }

    shared = 0
    for index, (left, right) in enumerate(zip(mtp["ids"], ar["ids"])):
        if left != right:
            break
        shared += 1
    if require_same_tokens:
        assert shared == len(mtp["ids"]) == len(ar["ids"]), {
            "arms_published_different_tokens": {
                **where,
                "speculative": mtp["ids"],
                "autoregressive": ar["ids"],
            },
        }
    assert shared >= minimum_shared, {
        "arms_share_too_short_a_prefix_to_compare": {
            **where,
            "shared": shared,
            "required": minimum_shared,
            "speculative": mtp["ids"],
            "autoregressive": ar["ids"],
        },
    }

    worst = 0.0
    worst_position = None
    for index in range(shared):
        expected = ar["logprobs"][index]
        actual = mtp["logprobs"][index]
        assert expected["logprob"] is not None, {
            "autoregressive_arm_omitted_a_selected_logprob": {**where, "position": index},
        }
        assert actual["logprob"] is not None, {
            "speculative_arm_omitted_a_selected_logprob": {**where, "position": index},
        }
        assert actual["token"] == expected["token"], {
            "metadata_token_text_differs": {
                **where,
                "position": index,
                "speculative": actual["token"],
                "autoregressive": expected["token"],
            },
        }
        delta = abs(float(actual["logprob"]) - float(expected["logprob"]))
        if delta > worst:
            worst, worst_position = delta, index
        assert delta <= LOGPROB_TOLERANCE, {
            "selected_logprob_differs_from_autoregressive": {
                **where,
                "position": index,
                "token_id": mtp["ids"][index],
                "speculative": actual["logprob"],
                "autoregressive": expected["logprob"],
                "delta": delta,
            },
        }
        expected_top = expected["top"]
        actual_top = actual["top"]
        assert len(actual_top) == len(expected_top), {
            "top_logprobs_width_differs": {
                **where,
                "position": index,
                "speculative": len(actual_top),
                "autoregressive": len(expected_top),
            },
        }
        for rank, (left_pair, right_pair) in enumerate(zip(actual_top, expected_top)):
            assert left_pair[0] == right_pair[0], {
                "top_logprobs_token_differs": {
                    **where,
                    "position": index,
                    "rank": rank,
                    "speculative": left_pair[0],
                    "autoregressive": right_pair[0],
                },
            }
            top_delta = abs(float(left_pair[1]) - float(right_pair[1]))
            assert top_delta <= LOGPROB_TOLERANCE, {
                "top_logprob_differs_from_autoregressive": {
                    **where,
                    "position": index,
                    "rank": rank,
                    "token": left_pair[0],
                    "speculative": left_pair[1],
                    "autoregressive": right_pair[1],
                    "delta": top_delta,
                },
            }
    return {
        **where,
        "published_tokens": len(mtp["ids"]),
        "autoregressive_tokens": len(ar["ids"]),
        "shared_prefix": shared,
        "tokens_match": mtp["ids"] == ar["ids"],
        "worst_logprob_delta": worst,
        "worst_position": worst_position,
        "speculative_cycles": mtp["draft_cycles"],
        "speculative_output_tokens": mtp["mtp_output_tokens"],
    }


def _normalize_chat_logprobs(payload: Any) -> list[dict[str, Any]]:
    """Normalize the chat shape, whose entries nest their own top candidates."""

    if not isinstance(payload, dict):
        return []
    entries = payload.get("content") or []
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        normalized.append(
            {
                "token": str(entry.get("token")),
                "logprob": entry.get("logprob"),
                "top": [
                    [str(candidate.get("token")), float(candidate.get("logprob"))]
                    for candidate in entry.get("top_logprobs") or ()
                    if isinstance(candidate, dict) and candidate.get("logprob") is not None
                ],
            }
        )
    return normalized


def run_chat_completion(
    client: httpx.Client,
    args: argparse.Namespace,
    *,
    speculative: bool,
    stream: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Exercise the chat surface, which has its own splitter and validator."""

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.chat_prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "logprobs": True,
        "top_logprobs": int(args.top_logprobs),
        "speculative_mtp": bool(speculative),
        "seed": int(args.seed),
        "chat_template_kwargs": {"enable_thinking": False},
        **extra,
    }
    if not stream:
        response = client.post("/v1/chat/completions", json=payload)
        body = response.json()
        if response.status_code != 200:
            return {"status": response.status_code, "error": body.get("error"), "request": payload}
        choice = body["choices"][0]
        summary = body.get("hipengine", {}).get("speculative_mtp") or {}
        return {
            "status": 200,
            "ids": [],
            "logprobs": _normalize_chat_logprobs(choice.get("logprobs")),
            "used": bool(summary.get("used")),
            "effective_route": summary.get("effective_route"),
            "draft_cycles": int(summary.get("draft_cycles", 0) or 0),
            "mtp_output_tokens": int(summary.get("mtp_output_tokens", 0) or 0),
            "request": payload,
        }

    logprobs: list[dict[str, Any]] = []
    with client.stream(
        "POST", "/v1/chat/completions", json={**payload, "stream": True}
    ) as response:
        if response.status_code != 200:
            response.read()
            return {"status": response.status_code, "error": response.text, "request": payload}
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                return {
                    "status": 200,
                    "error": chunk["error"],
                    "ids": [],
                    "logprobs": logprobs,
                    "used": False,
                    "effective_route": None,
                    "draft_cycles": 0,
                    "mtp_output_tokens": 0,
                    "request": payload,
                }
            for choice in chunk.get("choices") or ():
                logprobs.extend(_normalize_chat_logprobs(choice.get("logprobs")))
    return {
        "status": 200,
        "ids": [],
        "logprobs": logprobs,
        "used": False,
        "effective_route": None,
        "draft_cycles": 0,
        "mtp_output_tokens": 0,
        "request": payload,
    }


def compare_stream_to_blocking(
    streamed: dict[str, Any],
    blocking: dict[str, Any],
    where: dict[str, Any],
) -> dict[str, Any]:
    """Check streamed metadata against the blocking response for one request.

    The streaming arm publishes the same tokens with the same values, but a stop
    suffix that is still pending at the end of the stream is withheld, so the
    streamed token list is compared as a prefix of the blocking one.
    """

    assert streamed["status"] == 200, {"streaming_arm_refused": {**where, "arm": streamed}}
    assert not streamed.get("error"), {"streaming_arm_failed": {**where, "arm": streamed}}
    assert blocking["status"] == 200, {"blocking_arm_refused": {**where, "arm": blocking}}
    assert streamed["logprobs"], {"streaming_arm_reported_no_logprobs": {**where, "arm": streamed}}
    assert blocking["logprobs"], {"blocking_arm_reported_no_logprobs": {**where, "arm": blocking}}

    streamed_tokens = [entry["token"] for entry in streamed["logprobs"]]
    blocking_tokens = [entry["token"] for entry in blocking["logprobs"]]
    assert streamed_tokens == blocking_tokens[: len(streamed_tokens)], {
        "streaming_tokens_differ_from_blocking": {
            **where,
            "streamed": streamed_tokens,
            "blocking": blocking_tokens,
        },
    }
    worst = 0.0
    worst_position = None
    for index, entry in enumerate(streamed["logprobs"]):
        reference = blocking["logprobs"][index]
        if entry["logprob"] is None or reference["logprob"] is None:
            raise AssertionError(
                {
                    "streaming_metadata_missing_a_logprob": {
                        **where,
                        "position": index,
                        "token": entry["token"],
                        "streamed": entry["logprob"],
                        "blocking": reference["logprob"],
                    }
                }
            )
        delta = abs(float(entry["logprob"]) - float(reference["logprob"]))
        if delta > worst:
            worst = delta
            worst_position = index
        assert entry["top"] == reference["top"], {
            "streaming_top_logprobs_differ_from_blocking": {
                **where,
                "position": index,
                "token": entry["token"],
                "streamed": entry["top"],
                "blocking": reference["top"],
            },
        }
    assert worst <= LOGPROB_TOLERANCE, {
        "streaming_logprob_delta_exceeds_tolerance": {
            **where,
            "worst_delta": worst,
            "position": worst_position,
            "tolerance": LOGPROB_TOLERANCE,
        },
    }
    return {
        **where,
        "published_tokens": len(streamed["logprobs"]),
        "blocking_tokens": len(blocking["logprobs"]),
        "worst_logprob_delta": worst,
        "worst_position": worst_position,
    }


def server_storage(client: httpx.Client) -> dict[str, Any]:
    response = client.get("/ready")
    response.raise_for_status()
    ready = response.json()
    capability = ready["model"]["kv_capability"]
    return {
        "status": ready.get("status"),
        "served_model": ready["model"].get("id"),
        "effective_kv_storage": capability.get("effective_kv_storage"),
        "capability_id": capability.get("capability_id"),
        "capability_status": capability.get("status"),
        "runtime_action": capability.get("runtime_action"),
        "diagnostic_override": capability.get("diagnostic_override"),
        "persistent_bf16_mirror": capability.get("persistent_bf16_mirror"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--sampled-temperature",
        type=float,
        default=0.35,
        help=(
            "temperature for the sampled arm; kept low so both arms stay on the "
            "same trajectory long enough to compare more than one position"
        ),
    )
    parser.add_argument(
        "--minimum-shared-prefix",
        type=int,
        default=4,
        help="positions the sampled arms must share before their values are compared",
    )
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument(
        "--chat-prompt",
        default=CHAT_PROMPT,
        help=(
            "prompt for the chat arms; the chat template decides how much of a "
            "continuation a chat turn produces, so this one asks for a fixed "
            "sentence instead of a long count"
        ),
    )
    parser.add_argument(
        "--minimum-chat-tokens",
        type=int,
        default=2,
        help="positions the chat stream must carry before its values are compared",
    )
    parser.add_argument(
        "--expect-storage",
        default=None,
        help=(
            "require this effective KV storage (for example int8_per_token_head "
            "or bf16); without it the gate records whichever cell it measured"
        ),
    )
    parser.add_argument(
        "--allow-kv-diagnostic-override",
        action="store_true",
        help=(
            "accept a server running the rejected artifact under the documented KV "
            "diagnostic override; without this the gate refuses one, so an override "
            "run can never be read as a qualified measurement"
        ),
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {
        "scope": "MTP logprob metadata against the autoregressive route",
        "base_url": args.base_url,
        "prompt": args.prompt,
        "seed": args.seed,
        "logprob_tolerance": LOGPROB_TOLERANCE,
        "passed": False,
    }

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        storage = server_storage(client)
        report["server"] = storage
        if args.expect_storage and storage["effective_kv_storage"] != args.expect_storage:
            report["error"] = (
                f"this run pins {args.expect_storage!r}; the server reports "
                f"{storage['effective_kv_storage']!r}"
            )
            args.json.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["error"]))
            return 1
        if storage["diagnostic_override"] and not args.allow_kv_diagnostic_override:
            report["error"] = (
                "the server is running a rejected artifact under the documented KV "
                "diagnostic override; re-run with --allow-kv-diagnostic-override to "
                "record it as an override measurement"
            )
            args.json.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["error"]))
            return 1

        try:
            # 1. Greedy metadata: both arms are deterministic, so the tokens match
            #    and every reported value is comparable.
            mtp = run_completion(client, args, speculative=True)
            ar = run_completion(client, args, speculative=False)
            report["greedy_metadata"] = compare_prefix(
                mtp,
                ar,
                {"check": "greedy_metadata", "temperature": args.temperature},
                minimum_shared=1,
                require_same_tokens=True,
            )
            print(
                f"PASS greedy_metadata tokens={report['greedy_metadata']['published_tokens']} "
                f"cycles={report['greedy_metadata']['speculative_cycles']} "
                f"worst_delta={report['greedy_metadata']['worst_logprob_delta']:.3e}"
            )

            # 2. Sampled metadata: the sampled branch reports the retained
            #    support's probability, which is a different number from the
            #    greedy branch and is compared over the shared trajectory.
            sampled_args = argparse.Namespace(**vars(args))
            sampled_args.temperature = args.sampled_temperature
            mtp_sampled = run_completion(client, sampled_args, speculative=True)
            ar_sampled = run_completion(client, sampled_args, speculative=False)
            report["sampled_metadata"] = compare_prefix(
                mtp_sampled,
                ar_sampled,
                {"check": "sampled_metadata", "temperature": args.sampled_temperature},
                minimum_shared=args.minimum_shared_prefix,
                require_same_tokens=False,
            )
            print(
                f"PASS sampled_metadata shared={report['sampled_metadata']['shared_prefix']} "
                f"of {report['sampled_metadata']['published_tokens']} "
                f"worst_delta={report['sampled_metadata']['worst_logprob_delta']:.3e}"
            )

            # 3. Streaming metadata must equal the non-streaming values for the
            #    same request, so a second implementation cannot drift. The
            #    streaming response does not repeat the admission summary, so
            #    this compares against the arm whose speculation is already
            #    established above rather than re-asserting it here.
            streamed = run_completion(client, args, speculative=True, stream=True)
            report["stream_metadata"] = compare_stream_to_blocking(
                streamed,
                mtp,
                {"check": "stream_metadata", "temperature": args.temperature},
            )
            print(
                f"PASS stream_metadata chunks={report['stream_metadata']['published_tokens']} "
                f"worst_delta={report['stream_metadata']['worst_logprob_delta']:.3e}"
            )

            # 4. The chat surface runs its own reasoning splitter and logprob
            #    validator over the same chunks, so its streamed values are
            #    checked against its own blocking response.
            chat_blocking = run_chat_completion(client, args, speculative=True)
            chat_streamed = run_chat_completion(client, args, speculative=True, stream=True)
            report["chat_stream_metadata"] = compare_stream_to_blocking(
                chat_streamed,
                chat_blocking,
                {"check": "chat_stream_metadata", "temperature": args.temperature},
            )
            assert chat_blocking["used"], {
                "chat_arm_did_not_speculate": {"arm": chat_blocking}
            }
            assert (
                report["chat_stream_metadata"]["published_tokens"]
                >= args.minimum_chat_tokens
            ), {
                "chat_stream_too_short_to_compare": {
                    "published": report["chat_stream_metadata"]["published_tokens"],
                    "required": args.minimum_chat_tokens,
                    "arm": chat_streamed,
                }
            }
            print(
                f"PASS chat_stream_metadata chunks={report['chat_stream_metadata']['published_tokens']} "
                f"worst_delta={report['chat_stream_metadata']['worst_logprob_delta']:.3e}"
            )
        except AssertionError as error:
            report["error"] = error.args[0] if error.args else "assertion failed"
            args.json.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["error"], indent=1))
            return 1

    report["passed"] = True
    report["summary"] = {
        "storage": storage["effective_kv_storage"],
        "greedy_tokens": report["greedy_metadata"]["published_tokens"],
        "greedy_cycles": report["greedy_metadata"]["speculative_cycles"],
        "greedy_worst_delta": report["greedy_metadata"]["worst_logprob_delta"],
        "sampled_shared_prefix": report["sampled_metadata"]["shared_prefix"],
        "sampled_worst_delta": report["sampled_metadata"]["worst_logprob_delta"],
        "stream_tokens": report["stream_metadata"]["published_tokens"],
        "stream_worst_delta": report["stream_metadata"]["worst_logprob_delta"],
        "chat_stream_tokens": report["chat_stream_metadata"]["published_tokens"],
        "chat_stream_worst_delta": report["chat_stream_metadata"]["worst_logprob_delta"],
    }
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, **report["summary"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
