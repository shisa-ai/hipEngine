#!/usr/bin/env python3
"""RT-1 blocker (a): short and long requests through the selection path on the SAME owners.

`docs/REFACTOR.md` records the requirements for enabling automatic prefill-workspace
selection. Three are now cleared by measurement (numerical qualification at the new chunk,
arithmetic neutrality, and the short-request loss). The specific remaining measurement is:

    "measure short/long requests on the same prepared owners"

Every measurement so far, including RT-1's and RT-3's, used **one process per arm**, so the
selection path itself has never been exercised end-to-end on a prepared pair. This script
does that: it prepares a 1024 owner and a 4096 owner in one runner, configures the selector,
and then runs a short (p512) and a long (p4096) prompt through the same runner.

The selection rule is smallest-owner-that-fits:

    next((w for w in choices if len(tokens) <= w.prefill_chunk_size), choices[-1])

so p512 must select the 1024 owner and p4096 the 4096 owner. The script asserts which owner
served each request rather than inferring it from timing.

Diagnostic only. No runtime default changes, no promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.loading.qwen4_exp_scratch import qwen4_exp_scratch_breakdown  # noqa: E402
from scripts.qwen4exp_canonical_ar_bench import (  # noqa: E402
    DEFAULT_FIXTURE,
    _git_metadata,
    _host_metadata,
    load_fixture,
)
from scripts.qwen4exp_chunk_memory_probe import prepare_lazy_group_risk  # noqa: E402
from scripts.qwen4exp_chunk_workspace import capture_workspace  # noqa: E402
from scripts.qwen4exp_framework_family_refresh import model_identity  # noqa: E402
from scripts.qwen4exp_layer2_profile_gate import _make_generator  # noqa: E402

# The prefill-only owners whose extra allocation must be admitted before preparing an
# additional workspace. Same tuple the owned-workspace check uses.
PREFILL_OWNERS = (
    "gdn_prefill_scratch", "qsa_prefill_scratch", "ple_prefill_scratch",
    "qsa_prefill_metadata", "_prefill_buffers",
)
RESERVE_BYTES = 4 << 30
# Which owner served a request, recorded by the selector itself.
SERVED: list[int] = []


def memory_stats() -> dict:
    import ctypes

    lib = ctypes.CDLL("libamdhip64.so")
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    lib.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    return {"free_bytes": free.value, "total_bytes": total.value}


def install_owner_probe(runner) -> None:
    """Record which owner's chunk size served each prefill, from inside the selector."""

    original = runner._prefill_chunked_impl

    def counted(token_ids, **kwargs):
        SERVED.append(int(runner.prefill_chunk_size))
        return original(token_ids, **kwargs)

    runner._prefill_chunked_impl = counted


def timed(runner, prompt, *, steps, repeats):
    """Median wall seconds over `repeats`, plus the last logits hash.

    Calls the runner directly rather than going through the chunk-boundary gate's
    `trajectory`, because that helper validates the observed chunk coverage against a
    caller-supplied size. The selector under test is the thing that chooses the chunk, so
    imposing an expected size would defeat the measurement.
    """
    walls, digest = [], None
    for _ in range(repeats):
        runner.reset()
        start = time.perf_counter()
        result = runner.prefill(prompt)
        logits = result.logits.copy()
        token = int(result.token_id)
        for _ in range(steps):
            result = runner.step(token)
            logits = result.logits.copy()
            token = int(result.token_id)
        walls.append(time.perf_counter() - start)
        digest = hashlib.sha256(logits.tobytes()).hexdigest()
    return statistics.median(walls), walls, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--base-chunk", type=int, default=4096)
    parser.add_argument("--extra-chunk", type=int, default=1024)
    parser.add_argument("--max-sequence-length", type=int, default=4352)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--short-case", default="code-p512")
    parser.add_argument("--long-case", default="code-p4096")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixture, digest = load_fixture(args.fixture)
    sources = {case["id"]: case["prompt_token_ids"] for case in fixture["cases"]}
    short_prompt = list(sources[args.short_case])
    long_prompt = list(sources[args.long_case])

    arm = SimpleNamespace(**{
        **vars(args),
        "max_sequence_length": args.max_sequence_length,
        "prefill_chunk_size": args.base_chunk,
    })
    # The profile gate registers these before constructing; without them
    # resolve_runtime_profile raises MissingRuntimeProfilePlanError.
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    generator, profile, _ = _make_generator(arm, "production")
    runner = generator.runner

    report: dict = {
        "schema": 1,
        "kind": "qwen4exp_rt1_same_owner_selection",
        "performance_claim": False,
        "promotion_claim": False,
        "purpose": (
            "RT-1 blocker (a) from docs/REFACTOR.md: measure short/long requests on the "
            "SAME prepared owners, exercising the selection path end-to-end."
        ),
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "fixture_sha256": digest,
        "manifest_sha256": profile.manifest_sha256,
        "model": model_identity(args.model_root),
        "base_chunk": args.base_chunk,
        "extra_chunk": args.extra_chunk,
        "steps": args.steps,
        "repeats": args.repeats,
        "limits": (
            "One model, one quant, one host, one execution profile. Diagnostic: the "
            "selector is enabled only inside this process; no runtime default changes."
        ),
    }

    try:
        prepare_lazy_group_risk(runner)
        before = memory_stats()
        plan = qwen4_exp_scratch_breakdown(
            runner.config, context_tokens=args.max_sequence_length,
            prefill_chunk_size=args.extra_chunk)
        extra_bytes = sum(plan[key] for key in PREFILL_OWNERS)
        if before["free_bytes"] < extra_bytes + RESERVE_BYTES:
            raise MemoryError("extra workspace would consume the reserve")

        base_owner = runner._prefill_workspaces[0]
        extra_owner = runner._allocate_extra_prefill_workspace(args.extra_chunk)
        prepare_lazy_group_risk(extra_owner)
        after = memory_stats()

        report["base_owner"] = {
            "chunk_size": int(base_owner.prefill_chunk_size),
            "prepared": capture_workspace(base_owner) is not None,
        }
        report["extra_owner"] = {
            "chunk_size": int(extra_owner.prefill_chunk_size),
            "additional_bytes": before["free_bytes"] - after["free_bytes"],
        }
        report["owner_count"] = len(runner._prefill_workspaces)

        # Enable the selector over the two prepared owners. This is the wiring that
        # production does not do; doing it here is the point of the measurement.
        runner._configure_prefill_workspace_selection(
            [base_owner, extra_owner], reserve_bytes=RESERVE_BYTES)
        report["selection_enabled"] = runner._prefill_workspace_selection is not None
        report["selection_choices"] = [
            int(item.prefill_chunk_size)
            for item in (runner._prefill_workspace_selection or ())
        ]

        install_owner_probe(runner)
        report["cases"] = []
        # The base owner sets the staging capacity and the selector may only choose owners
        # at or below it, so the short request is expected to select the SMALL extra owner
        # and the long request the large base owner.
        for label, prompt, expect_chunk in (
            ("short", short_prompt, args.extra_chunk),
            ("long", long_prompt, args.base_chunk),
        ):
            for _ in range(args.warmups):
                SERVED.clear()
                timed(runner, prompt, steps=args.steps, repeats=1)
            SERVED.clear()
            median, walls, logits_sha = timed(
                runner, prompt, steps=args.steps, repeats=args.repeats)
            served = sorted(set(SERVED))
            entry = {
                "label": label,
                "prompt_tokens": len(prompt),
                "expected_owner_chunk": expect_chunk,
                "owners_that_served": served,
                "selection_correct": served == [expect_chunk],
                "wall_median_s": median,
                "wall_seconds": walls,
                "spread_pct": (max(walls) - min(walls)) / median * 100,
                "logits_sha256": logits_sha,
            }
            report["cases"].append(entry)
            print(json.dumps(entry, indent=1))

        report["all_selections_correct"] = all(
            case["selection_correct"] for case in report["cases"])

        # The decisive comparison: the SAME short prompt on the SAME prepared owners with
        # the selector turned off, so it is forced onto the large base owner. This isolates
        # what the shape policy buys a short request, which is the property the policy
        # exists to preserve.
        runner._configure_prefill_workspace_selection(None, reserve_bytes=RESERVE_BYTES)
        report["selector_disabled"] = True
        for _ in range(max(args.warmups, 3)):
            SERVED.clear()
            timed(runner, short_prompt, steps=args.steps, repeats=1)
        SERVED.clear()
        median_off, walls_off, logits_off = timed(
            runner, short_prompt, steps=args.steps, repeats=args.repeats)
        served_off = sorted(set(SERVED))
        short_on = report["cases"][0]
        report["short_without_selection"] = {
            "label": "short_no_selection",
            "prompt_tokens": len(short_prompt),
            "expected_owner_chunk": args.base_chunk,
            "owners_that_served": served_off,
            "selection_correct": served_off == [args.base_chunk],
            "wall_median_s": median_off,
            "wall_seconds": walls_off,
            "spread_pct": (max(walls_off) - min(walls_off)) / median_off * 100,
            "logits_sha256": logits_off,
        }
        report["short_policy_effect"] = {
            "with_selection_s": short_on["wall_median_s"],
            "without_selection_s": median_off,
            "delta_pct": (median_off - short_on["wall_median_s"]) / median_off * 100,
            "same_logits": short_on["logits_sha256"] == logits_off,
            "separation_vs_spread": (
                abs(median_off - short_on["wall_median_s"])
                / max(short_on["spread_pct"], report["short_without_selection"]["spread_pct"])
                if max(short_on["spread_pct"], report["short_without_selection"]["spread_pct"]) > 0
                else None),
        }
        print(json.dumps(report["short_without_selection"], indent=1))
        print(json.dumps(report["short_policy_effect"], indent=1))

        report["status"] = (
            "complete" if report["all_selections_correct"]
            and report["short_without_selection"]["selection_correct"] else "invalid")
    finally:
        report.setdefault("cases", [])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1) + "\n")
        runner.close()

    print(json.dumps({
        "status": report["status"],
        "selection_choices": report.get("selection_choices"),
        "extra_owner_bytes": (report.get("extra_owner") or {}).get("additional_bytes"),
        "cases": [
            {"label": c["label"], "tokens": c["prompt_tokens"],
             "owner": c["owners_that_served"], "median_s": round(c["wall_median_s"], 4)}
            for c in report["cases"]
        ],
        "short_without_selection": {
            "owner": (report.get("short_without_selection") or {}).get("owners_that_served"),
            "median_s": round(
                (report.get("short_without_selection") or {}).get("wall_median_s", 0), 4),
        },
        "short_policy_effect": report.get("short_policy_effect"),
    }, indent=1))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
