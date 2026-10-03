#!/usr/bin/env python3
"""Live acceptance for the MTP finish rule: EOS, stop ids, stop sequences, min_tokens.

The cycle commit applies the autoregressive finish rule to a whole verified chain,
so a stop can land *inside* an accepted draft chain instead of only at the last
visible token of a greedy one. A unit test pins the rule; this gate measures a
running server, because the failure it guards against is a commit that publishes
past the terminal token -- which only a real chain can produce.

The instrument is a seeded free trajectory. Seeded requests are reproducible on
both routes, so the trajectory the server generates with no stop is also the
reference for every stop: placing a stop at position ``k`` of that trajectory must
publish that prefix and nothing after it. Without a finish rule the commit would
publish the whole chain that contained the stop, which is one or more tokens too
many, and the sweep fails.

Checks:

* ``free_trajectory`` -- both arms agree on a no-stop run and the speculative arm
  actually speculated, so the reference is a real chain, not a fallback.
* ``stop_sweep`` -- a text stop built from each position of the trajectory stops
  there, on both arms, with ``finish_reason=stop``, publishing a prefix of the
  trajectory and no token after the stop.
* ``stop_sequence`` -- a two-token stop sequence stops before the sequence
  completes, on both arms.
* ``eos_floor`` -- ``eos_token_id`` at or above ``min_tokens`` finishes with
  ``eos``; below the floor the same token is emitted and generation continues.
* ``neighbour_isolation`` -- a request that stops early leaves a concurrent
  neighbour's published ids unchanged.

The rule reads tokens, not storage, so the gate runs against any KV cell. Run it
once per cell (INT8 and BF16) and record both.

Usage:
    .venv/bin/python scripts/mtp_finish_rule_gate.py \
        --base-url http://127.0.0.1:18198 --model <served-name> \
        --json /tmp/mtp-finish-int8.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any

import httpx

# Repetitive text keeps the draft chains long, so a stop placed inside the
# trajectory lands inside a chain rather than always at a chain boundary.
PROMPT = (
    "Count slowly, one number per word: one two three four five six seven "
    "eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty"
)

USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def run_arm(
    client: httpx.Client,
    args: argparse.Namespace,
    *,
    speculative: bool,
    seed: int,
    max_tokens: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "speculative_mtp": bool(speculative),
        "seed": int(seed),
        **extra,
    }
    response = client.post("/v1/completions", json=payload)
    body = response.json()
    if response.status_code != 200:
        return {
            "status": response.status_code,
            "error": body.get("error"),
            "request": payload,
        }
    choice = body["choices"][0]
    summary = body.get("hipengine", {}).get("speculative_mtp") or {}
    choice_engine = choice.get("hipengine") or {}
    decode_state = choice_engine.get("decode_state") or {}
    usage = body["usage"]
    return {
        "status": 200,
        "ids": choice_engine.get("generated_token_ids") or [],
        "text": choice.get("text") or "",
        "finish_reason": choice.get("finish_reason"),
        "finish_details": choice_engine.get("finish_details"),
        "finish_detail_reason": (choice_engine.get("finish_details") or {}).get("reason"),
        "usage": {field: int(usage.get(field, 0)) for field in USAGE_FIELDS},
        "used": bool(summary.get("used")),
        "draft_cycles": int(summary.get("draft_cycles", 0) or 0),
        "draft_tokens": int(summary.get("draft_tokens", 0) or 0),
        "accepted_draft_tokens": int(summary.get("accepted_draft_tokens", 0) or 0),
        "mtp_output_tokens": int(summary.get("mtp_output_tokens", 0) or 0),
        "ar_output_tokens": int(summary.get("ar_output_tokens", 0) or 0),
        "effective_route": summary.get("effective_route"),
        "decode_step_index": decode_state.get("step_index"),
        "decode_generated_tokens": decode_state.get("generated_tokens"),
        "cache_action": (choice_engine.get("finish_details") or {}).get("cache_action"),
    }


def detokenize(client: httpx.Client, token_ids: list[int]) -> str:
    response = client.post("/v1/hipengine/detokenize", json={"token_ids": token_ids})
    response.raise_for_status()
    return str(response.json()["text"])


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


def check_pair(
    mtp: dict[str, Any],
    ar: dict[str, Any],
    where: dict[str, Any],
    *,
    require_speculation: bool = True,
) -> None:
    assert mtp["status"] == 200, {"speculative_arm_refused": {**where, "arm": mtp}}
    assert ar["status"] == 200, {"autoregressive_arm_refused": {**where, "arm": ar}}
    if require_speculation:
        assert mtp["used"], {"speculative_arm_did_not_speculate": {**where, "arm": mtp}}
        assert mtp["draft_cycles"] > 0, {"speculative_arm_ran_no_cycles": {**where, "arm": mtp}}


def speculation_exempt(row: dict[str, Any]) -> str | None:
    """Why a row cannot be required to have speculated, or ``None`` if it can.

    A cycle needs a chain to verify, so a stop that matches the first generated
    token ends the request before one can run, and a row whose whole output came
    from the autoregressive tail had nothing to verify either. Both are recorded
    per row rather than silently skipped.
    """

    if len(row["ids"]) < 2:
        return "stopped_at_the_first_token"
    if row["mtp_output_tokens"] == 0:
        return "autoregressive_tail_only"
    return None


def check_finish(
    mtp: dict[str, Any],
    ar: dict[str, Any],
    where: dict[str, Any],
    *,
    trajectory: list[int],
    stop_position: int | None,
) -> None:
    """Require both arms to publish the same terminal prefix, with no token after it."""

    for arm, name in ((mtp, "speculative"), (ar, "autoregressive")):
        assert arm["finish_reason"] == "stop", {
            "finish_reason_not_stop": {
                **where, "arm": name, "finish_reason": arm["finish_reason"],
            },
        }
        # The public vocabulary reports every non-length reason as ``stop``, so
        # the rule that actually fired is what distinguishes a stop from an EOS.
        assert arm["finish_detail_reason"] == "stop", {
            "stop_did_not_fire_the_stop_rule": {
                **where, "arm": name, "finish_detail_reason": arm["finish_detail_reason"],
            },
        }
        assert arm["ids"] == trajectory[: len(arm["ids"])], {
            "published_token_outside_the_trajectory": {
                **where, "arm": name, "published": arm["ids"],
            },
        }
        if stop_position is not None:
            assert len(arm["ids"]) <= stop_position + 1, {
                "token_published_after_the_stop": {
                    **where,
                    "arm": name,
                    "stop_position": stop_position,
                    "published_tokens": len(arm["ids"]),
                },
            }
    assert mtp["ids"] == ar["ids"], {
        "route_parity_mismatch": {
            **where, "speculative": mtp["ids"], "autoregressive": ar["ids"],
        },
    }


def sweep_positions(trajectory: list[int], stride: int, limit: int) -> list[int]:
    positions = list(range(1, max(1, len(trajectory) - 1), max(1, stride)))
    return positions[:limit]


def distinct_stop_positions(
    trajectory: list[int], detokenize_one, stride: int, limit: int
) -> list[tuple[int, str]]:
    """Positions whose stop text is not one already taken.

    Repetitive text repeats tokens, so several positions lower to the same stop
    string and the route legitimately stops at the first of them -- the later
    position can only re-measure that same stop. Taking the limit over *distinct*
    stops spends the request budget on coverage instead of repeating one case.
    """

    seen: dict[str, int] = {}
    for position in sweep_positions(trajectory, stride, len(trajectory)):
        text = detokenize_one([trajectory[position]])
        if not text.strip() or text in seen:
            continue
        seen[text] = position
        if len(seen) >= limit:
            break
    return [(position, text) for text, position in sorted(seen.items(), key=lambda kv: kv[1])]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--sweep-stride", type=int, default=1)
    parser.add_argument("--sweep-limit", type=int, default=24)
    parser.add_argument(
        "--expect-storage",
        default=None,
        help=(
            "require this effective KV storage (for example int8_per_token_head "
            "or bf16); without it the gate records whichever cell it measured"
        ),
    )
    parser.add_argument("--prompt", default=PROMPT)
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
        "scope": "MTP finish rule: stop ids, stop sequences, EOS floor, neighbour isolation",
        "base_url": args.base_url,
        "prompt": args.prompt,
        "seed": args.seed,
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
            return 2
        if storage["diagnostic_override"] and not args.allow_kv_diagnostic_override:
            report["error"] = (
                "the server runs the artifact's rejected capability under a KV "
                "diagnostic override; pass --allow-kv-diagnostic-override to record "
                "that explicitly"
            )
            args.json.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["error"]))
            return 2

        try:
            # 1. The reference trajectory. Both arms run with no stop so the
            #    sweep's expectations come from a run the server actually made.
            free_mtp = run_arm(client, args, speculative=True, seed=args.seed)
            free_ar = run_arm(client, args, speculative=False, seed=args.seed)
            check_pair(free_mtp, free_ar, {"check": "free_trajectory"})
            assert free_mtp["finish_reason"] == free_ar["finish_reason"], {
                "free_trajectory_finish_mismatch": {
                    "speculative": free_mtp["finish_reason"],
                    "autoregressive": free_ar["finish_reason"],
                },
            }
            trajectory = list(free_ar["ids"])
            assert len(trajectory) >= 8, {
                "free_trajectory_too_short": {"tokens": len(trajectory)},
            }
            report["free_trajectory"] = {
                "ids": trajectory,
                "speculative": free_mtp,
                "autoregressive": free_ar,
                "ids_match": free_mtp["ids"] == free_ar["ids"],
            }
            print(
                f"PASS free_trajectory tokens={len(trajectory)} "
                f"cycles={free_mtp['draft_cycles']} accepted={free_mtp['accepted_draft_tokens']} "
                f"finish={free_mtp['finish_reason']} "
                f"ids_match={free_mtp['ids'] == free_ar['ids']}"
            )

            # 2. Stop sweep: a stop at each position of the trajectory stops there.
            sweep: list[dict[str, Any]] = []
            stop_positions = distinct_stop_positions(
                trajectory,
                lambda ids: detokenize(client, ids),
                args.sweep_stride,
                args.sweep_limit,
            )
            for position, stop_text in stop_positions:
                where = {"check": "stop_sweep", "position": position, "stop": stop_text}
                mtp = run_arm(
                    client, args, speculative=True, seed=args.seed, stop=[stop_text]
                )
                ar = run_arm(
                    client, args, speculative=False, seed=args.seed, stop=[stop_text]
                )
                exempt = speculation_exempt(mtp)
                check_pair(mtp, ar, where, require_speculation=exempt is None)
                check_finish(mtp, ar, where, trajectory=trajectory, stop_position=position)
                row = {
                    **where,
                    "published_tokens": len(mtp["ids"]),
                    # True when the stop text first matches exactly at the position
                    # it was built from; a prefix match on an earlier token makes
                    # it false, which is correct text-stop behaviour.
                    "stop_matched_at_build_position": mtp["ids"]
                    == trajectory[: position + 1],
                    "draft_cycles": mtp["draft_cycles"],
                    "accepted_draft_tokens": mtp["accepted_draft_tokens"],
                    "draft_tokens": mtp["draft_tokens"],
                    "mtp_output_tokens": mtp["mtp_output_tokens"],
                    "ar_output_tokens": mtp["ar_output_tokens"],
                    "speculation_exempt": exempt,
                    "cache_action": mtp["cache_action"],
                    "decode_step_index": mtp["decode_step_index"],
                    "stop_text_in_text": stop_text in mtp["text"],
                }
                sweep.append(row)
                print(
                    f"PASS stop_sweep position={position:2d} stop={stop_text!r:12s} "
                    f"published={len(mtp['ids']):2d} cycles={mtp['draft_cycles']:2d} "
                    f"mtp_out={mtp['mtp_output_tokens']:2d} "
                    f"matched_at_build_position={row['stop_matched_at_build_position']} "
                    f"exempt={exempt}"
                )
            assert sweep, {"stop_sweep_empty": True}
            assert any(row["speculation_exempt"] is None for row in sweep), {
                "stop_sweep_never_speculated": {
                    "exempt": [row["speculation_exempt"] for row in sweep],
                },
            }
            report["stop_sweep"] = sweep

            # 3. A two-token stop sequence: the sequence must not complete.
            sequence_rows: list[dict[str, Any]] = []
            for position in sweep_positions(trajectory, max(1, args.sweep_stride), 6):
                if position + 2 >= len(trajectory):
                    continue
                sequence_text = detokenize(client, trajectory[position : position + 2])
                if not sequence_text.strip():
                    continue
                where = {
                    "check": "stop_sequence",
                    "position": position,
                    "stop": sequence_text,
                }
                mtp = run_arm(
                    client, args, speculative=True, seed=args.seed, stop=[sequence_text]
                )
                ar = run_arm(
                    client, args, speculative=False, seed=args.seed, stop=[sequence_text]
                )
                exempt = speculation_exempt(mtp)
                check_pair(mtp, ar, where, require_speculation=exempt is None)
                check_finish(mtp, ar, where, trajectory=trajectory, stop_position=position + 1)
                row = {
                    **where,
                    "published_tokens": len(mtp["ids"]),
                    "draft_cycles": mtp["draft_cycles"],
                    "mtp_output_tokens": mtp["mtp_output_tokens"],
                    "speculation_exempt": exempt,
                    "stop_text_in_text": sequence_text in mtp["text"],
                }
                sequence_rows.append(row)
                print(
                    f"PASS stop_sequence position={position:2d} stop={sequence_text!r:24s} "
                    f"published={len(mtp['ids']):2d} cycles={mtp['draft_cycles']:2d} "
                    f"stop_text_in_text={row['stop_text_in_text']}"
                )
            assert sequence_rows, {"stop_sequence_not_exercised": True}
            report["stop_sequence"] = sequence_rows

            # 4. The EOS floor. The EOS token is one the trajectory emits, so the
            #    rule has something to suppress: at or above the floor it finishes,
            #    below it the same token is emitted and generation continues.
            eos_position = min(len(trajectory) - 3, max(2, len(trajectory) // 3))
            eos_token = trajectory[eos_position]
            floor_rows: list[dict[str, Any]] = []
            for label, minimum in (("at_or_above_floor", 0), ("below_floor", eos_position + 6)):
                where = {
                    "check": "eos_floor",
                    "position": eos_position,
                    "min_tokens": minimum,
                    "eos_token_id": eos_token,
                }
                mtp = run_arm(
                    client,
                    args,
                    speculative=True,
                    seed=args.seed,
                    eos_token_id=eos_token,
                    min_tokens=minimum,
                )
                ar = run_arm(
                    client,
                    args,
                    speculative=False,
                    seed=args.seed,
                    eos_token_id=eos_token,
                    min_tokens=minimum,
                )
                check_pair(mtp, ar, where)
                assert mtp["ids"] == ar["ids"], {
                    "route_parity_mismatch": {
                        **where, "speculative": mtp["ids"], "autoregressive": ar["ids"],
                    },
                }
                if minimum == 0:
                    # No floor: the trajectory is the free one and the EOS token
                    # itself is the terminal token.
                    assert mtp["finish_detail_reason"] == "eos", {
                        "eos_did_not_fire_the_eos_rule_above_the_floor": {
                            **where, "finish_detail_reason": mtp["finish_detail_reason"],
                        },
                    }
                    assert len(mtp["ids"]) <= eos_position + 1, {
                        "eos_published_past_the_terminal_token": {
                            **where, "published_tokens": len(mtp["ids"]),
                        },
                    }
                    assert mtp["ids"] == trajectory[: len(mtp["ids"])], {
                        "eos_run_left_the_trajectory": {**where, "published": mtp["ids"]},
                    }
                else:
                    # The floor suppresses the EOS *token* until it is reached, so
                    # generation continues -- and, when the model wants the token
                    # the floor withheld, with a different tokenization of the
                    # same text. That is why the prefix check applies only at
                    # floor 0 and this arm is checked on the floor itself.
                    assert len(mtp["ids"]) > eos_position + 1, {
                        "eos_floor_did_not_suppress_the_rule": {
                            **where, "published_tokens": len(mtp["ids"]),
                        },
                    }
                    assert eos_token not in mtp["ids"][:minimum], {
                        "eos_token_emitted_below_the_floor": {
                            **where, "published": mtp["ids"][:minimum],
                        },
                    }
                row = {
                    **where,
                    "label": label,
                    "finish_reason": mtp["finish_reason"],
                    "finish_detail_reason": mtp["finish_detail_reason"],
                    "published_tokens": len(mtp["ids"]),
                    "trajectory_prefix_matches": mtp["ids"] == trajectory[: len(mtp["ids"])],
                    "draft_cycles": mtp["draft_cycles"],
                    "mtp_output_tokens": mtp["mtp_output_tokens"],
                }
                floor_rows.append(row)
                print(
                    f"PASS eos_floor {label:16s} eos_token={eos_token} "
                    f"min_tokens={minimum:2d} detail_reason={mtp['finish_detail_reason']} "
                    f"published={len(mtp['ids']):2d} cycles={mtp['draft_cycles']:2d}"
                )
            report["eos_floor"] = floor_rows

            # 5. A stopping request must not move a concurrent neighbour's cursor.
            isolation_position = sweep_positions(trajectory, 1, 1)[0]
            isolation_text = detokenize(client, [trajectory[isolation_position]])
            solo_stopped = run_arm(
                client, args, speculative=True, seed=args.seed, stop=[isolation_text]
            )
            solo_neighbour = run_arm(client, args, speculative=True, seed=args.seed + 1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                stopped_future = pool.submit(
                    run_arm,
                    client,
                    args,
                    speculative=True,
                    seed=args.seed,
                    stop=[isolation_text],
                )
                neighbour_future = pool.submit(
                    run_arm, client, args, speculative=True, seed=args.seed + 1
                )
                stopped = stopped_future.result()
                neighbour = neighbour_future.result()
            where = {"check": "neighbour_isolation"}
            assert stopped["ids"] == solo_stopped["ids"], {
                "stopping_request_changed_under_concurrency": {
                    **where, "solo": solo_stopped["ids"], "concurrent": stopped["ids"],
                },
            }
            assert neighbour["ids"] == solo_neighbour["ids"], {
                "neighbour_cursor_moved_by_a_stopping_request": {
                    **where,
                    "solo": solo_neighbour["ids"],
                    "concurrent": neighbour["ids"],
                },
            }
            report["neighbour_isolation"] = {
                **where,
                "stopping_ids": stopped["ids"],
                "neighbour_ids": neighbour["ids"],
                "neighbour_cycles_solo": solo_neighbour["draft_cycles"],
                "neighbour_cycles_concurrent": neighbour["draft_cycles"],
                "neighbour_finish_solo": solo_neighbour["finish_reason"],
                "neighbour_finish_concurrent": neighbour["finish_reason"],
            }
            print(
                f"PASS neighbour_isolation stopping={len(stopped['ids'])} tokens "
                f"neighbour={len(neighbour['ids'])} tokens "
                f"neighbour_unchanged={neighbour['ids'] == solo_neighbour['ids']}"
            )
        except AssertionError as error:
            report["error"] = error.args[0] if error.args else "assertion failed"
            args.json.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["error"], indent=1))
            return 1

    report["passed"] = True
    report["summary"] = {
        "storage": storage["effective_kv_storage"],
        "trajectory_tokens": len(report["free_trajectory"]["ids"]),
        "sweep_positions": len(report["stop_sweep"]),
        "sweep_rows_that_speculated": sum(
            1 for row in report["stop_sweep"] if row["speculation_exempt"] is None
        ),
        "stops_matched_at_build_position": sum(
            1 for row in report["stop_sweep"] if row["stop_matched_at_build_position"]
        ),
        "stop_sequence_rows": len(report["stop_sequence"]),
        "eos_floor_rows": len(report["eos_floor"]),
        "neighbour_isolated": True,
    }
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, **report["summary"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
