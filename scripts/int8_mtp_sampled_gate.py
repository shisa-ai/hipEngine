#!/usr/bin/env python3
"""Live acceptance for the sampled MTP route on INT8 KV.

The declaration states that the INT8 chain executes ``temperature``, penalty,
``logit_bias`` and ``suppress_token_ids`` requests. A declaration cannot say
whether the route *actually* runs them, so this gate measures a running server:
for each sampler configuration it runs the speculative arm and the request's own
true no-MTP autoregressive baseline, requires the speculative arm to have
speculated, and requires both arms to agree on finish reason and usage.

Seeded requests are used throughout. A sampled accept is stochastic, so an
unseeded repeat would only be a tolerance; with a seed the route is reproducible,
which makes the second run a real check -- identical ids and identical cycle
counts -- instead of a band that hides drift.

The token ids of the two arms are reported but not compared: the speculative arm
draws for accept and resample as well as for the emitted token, so its draw
stream is not the baseline's even when both are seeded. Finish reason and usage
are the parity contract; the induced-law gate
(``scripts/mtp_sampled_accept_distribution_gate.py``) is what proves the
distribution itself.

Usage:
    .venv/bin/python scripts/int8_mtp_sampled_gate.py \
        --base-url http://127.0.0.1:18198 --model <served-name> \
        --json /tmp/int8-mtp-sampled.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import httpx


# Each entry is a request the INT8 declaration says it executes. The values are
# deliberately ordinary: the gate is about the route, not about a sampler edge.
SAMPLER_CONFIGS: tuple[dict[str, Any], ...] = (
    {"name": "temperature_top_p", "temperature": 0.7, "top_p": 0.95},
    {
        "name": "penalties",
        "temperature": 0.7,
        "top_p": 0.95,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "repetition_penalty": 1.1,
    },
    {
        "name": "logit_bias",
        "temperature": 0.7,
        "top_p": 0.95,
        # Bias the newline tokens down rather than a rare token up: the effect has
        # to be visible in emitted text for the parity comparison to mean anything.
        "logit_bias": {"13": -12.0, "198": -8.0},
    },
    {
        "name": "suppress_token_ids",
        "temperature": 0.7,
        "top_p": 0.95,
        "suppress_token_ids": [13, 198],
    },
)

PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    (
        "A resident KV pool is allocated once per session and reused across "
        "requests. Explain in two sentences why its pages must never be freed "
        "while a captured graph still binds them."
    ),
    (
        "Summarise the tradeoff between speculative decoding and plain "
        "autoregressive decoding for a single-user local server:"
    ),
)

USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def sampler_params(config: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in config.items() if name != "name"}


def run_arm(
    client: httpx.Client,
    args: argparse.Namespace,
    config: dict[str, Any],
    prompt: str,
    *,
    speculative: bool,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "speculative_mtp": bool(speculative),
        "seed": int(seed),
        **sampler_params(config),
    }
    response = client.post("/v1/completions", json=payload)
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    summary = body.get("hipengine", {}).get("speculative_mtp") or {}
    decode_state = choice.get("hipengine", {}).get("decode_state") or {}
    usage = body["usage"]
    return {
        "ids": choice["hipengine"]["generated_token_ids"],
        "finish_reason": choice["finish_reason"],
        "usage": {field: int(usage.get(field, 0)) for field in USAGE_FIELDS},
        "usage_details": usage.get("completion_tokens_details"),
        "used": bool(summary.get("used")),
        "effective_route": summary.get("effective_route"),
        "draft_cycles": int(summary.get("draft_cycles", 0) or 0),
        "draft_tokens": int(summary.get("draft_tokens", 0) or 0),
        "accepted_draft_tokens": int(summary.get("accepted_draft_tokens", 0) or 0),
        "sampler_mode": decode_state.get("sampler_mode"),
        "fast_path_blockers": decode_state.get("sampler_fast_path_blockers"),
    }


def check_pair(mtp: dict[str, Any], ar: dict[str, Any], where: dict[str, Any]) -> None:
    assert mtp["used"], {"sampled_arm_did_not_speculate": {**where, "arm": mtp}}
    assert mtp["draft_cycles"] > 0, {
        "sampled_arm_ran_no_cycles": {**where, "arm": mtp},
    }
    assert mtp["finish_reason"] == ar["finish_reason"], {
        "finish_reason_mismatch": {
            **where, "speculative": mtp["finish_reason"], "autoregressive": ar["finish_reason"],
        },
    }
    assert mtp["usage"] == ar["usage"], {
        "usage_mismatch": {**where, "speculative": mtp["usage"], "autoregressive": ar["usage"]},
    }


def check_repeatable(
    first: dict[str, Any], second: dict[str, Any], where: dict[str, Any]
) -> None:
    assert first["ids"] == second["ids"], {
        "sampled_ids_not_repeatable": {
            **where, "first": first["ids"], "second": second["ids"],
        },
    }
    assert first["draft_cycles"] == second["draft_cycles"], {
        "sampled_cycles_not_repeatable": {
            **where, "first": first["draft_cycles"], "second": second["draft_cycles"],
        },
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
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--runs", type=int, default=2, help="repeatability runs")
    parser.add_argument(
        "--allow-kv-diagnostic-override",
        action="store_true",
        help=(
            "accept a server running the rejected artifact under the documented KV "
            "diagnostic override; without this the gate refuses one, so an override "
            "run can never be read as a qualified measurement"
        ),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {
        "scope": "sampled MTP on INT8 KV: speculative cycles, baseline parity, repeatability",
        "base_url": args.base_url,
        "prompts": list(PROMPTS),
        "configs": [config["name"] for config in SAMPLER_CONFIGS],
        "runs": [],
        "passed": False,
    }

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        storage = server_storage(client)
        report["server"] = storage
        if storage["effective_kv_storage"] != "int8_per_token_head":
            report["error"] = (
                "this gate measures the INT8 cell; the server reports "
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
            for run_index in range(max(1, int(args.runs))):
                run: dict[str, Any] = {"index": run_index, "rows": []}
                for config in SAMPLER_CONFIGS:
                    for prompt_index, prompt in enumerate(PROMPTS):
                        seed = 1000 + 7 * prompt_index + run_index * 0
                        where = {
                            "run": run_index,
                            "config": config["name"],
                            "prompt_index": prompt_index,
                            "seed": seed,
                        }
                        mtp = run_arm(
                            client, args, config, prompt, speculative=True, seed=seed
                        )
                        ar = run_arm(
                            client, args, config, prompt, speculative=False, seed=seed
                        )
                        check_pair(mtp, ar, where)
                        run["rows"].append({
                            **where,
                            "speculative": mtp,
                            "autoregressive": ar,
                            "ids_match_baseline": mtp["ids"] == ar["ids"],
                        })
                        print(
                            f"PASS {config['name']:18s} prompt={prompt_index} "
                            f"cycles={mtp['draft_cycles']:2d} accepted={mtp['accepted_draft_tokens']:2d} "
                            f"finish={mtp['finish_reason']} sampler={mtp['sampler_mode']} "
                            f"ids_match_baseline={mtp['ids'] == ar['ids']}"
                        )
                report["runs"].append(run)

            if len(report["runs"]) > 1:
                first, second = report["runs"][0], report["runs"][1]
                for left, right in zip(first["rows"], second["rows"], strict=True):
                    where = {
                        "config": left["config"],
                        "prompt_index": left["prompt_index"],
                        "seed": left["seed"],
                    }
                    check_repeatable(left["speculative"], right["speculative"], where)
                    check_repeatable(left["autoregressive"], right["autoregressive"], where)
                report["repeatable"] = True
        except AssertionError as error:
            report["error"] = error.args[0] if error.args else "assertion failed"
            args.json.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["error"], indent=1))
            return 1

    report["passed"] = True
    report["summary"] = {
        "rows": sum(len(run["rows"]) for run in report["runs"]),
        "configs": len(SAMPLER_CONFIGS),
        "prompts": len(PROMPTS),
        "runs": len(report["runs"]),
        "repeatable": bool(report.get("repeatable")),
    }
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, **report["summary"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
