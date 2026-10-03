#!/usr/bin/env python3
"""Is the Q8 dense-wide prefill route actually engaging in production?

``benchmarks/README.md`` and ``benchmarks/results/2026-09-16-flashnext-per-role-cost``
record the dense projections of a 4096-token prefill at 2585 / 1580 / 1249 / 1960
/ 656 ms - 10553 ms of a 22046 ms prefill, 47.9% - running the scalar FP32
``gguf_k_prefill_out_coltile_rowbatch`` chain at 2366-2991 GFLOP/s, 8-10% of this
part's FP32 peak.  The wide-row route is 6.74x on the same operation and the
profile binds it for layers 16-47.

Those two statements cannot both describe the current prefill, so one of them is
stale.  This probe answers which, without a benchmark and without a gate: it runs
one real prefill through the production profile with the dispatch recorder on and
reports

* every route selector's engage/decline count and reason, and
* the launched kernel symbols with their call counts.

The second half is the one that cannot be fooled: a selector can report
``engaged`` and still not have replaced the leaf, and the symbol census shows
which kernel actually ran.

Usage:
    python3 scripts/qwen4exp_q8_dense_wide_route_probe.py [--case code-p4096]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DEFAULT_MODEL_ROOT = pathlib.Path(
    "/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
DEFAULT_FIXTURE = pathlib.Path(
    "/home/lhl/hipEngine/benchmarks/fixtures/"
    "qwen4exp_canonical_ar_p512_p1024_p4096.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=pathlib.Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--fixture", type=pathlib.Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case", default="code-p4096")
    parser.add_argument("--max-sequence-length", type=int, default=4160)
    parser.add_argument("--compiler-version-file", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def load_prompt(fixture: pathlib.Path, case: str) -> list[int]:
    payload = json.loads(fixture.read_text())
    for row in payload["cases"]:
        if row["id"] == case:
            return [int(token) for token in row["prompt_token_ids"]]
    raise SystemExit(
        f"case {case!r} not found in {fixture}; "
        f"have {[row['id'] for row in payload['cases']]}"
    )


def main() -> int:
    args = parse_args()
    tokens = load_prompt(args.fixture, args.case)

    from hipengine import LLM
    from hipengine.generation.engine_service import GenerationRequest
    from hipengine.kernels import launch_census
    from hipengine.runtime.gguf_linear import wmma_prefill_dispatch_recorder

    env = {"HIPENGINE_KERNEL_CENSUS": "1"}
    if args.compiler_version_file is not None:
        env["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    prior = {key: os.environ.get(key) for key in env}
    os.environ.update(env)

    report: dict[str, object] = {
        "schema": 1,
        "kind": "qwen4exp_q8_dense_wide_route_probe",
        "case": args.case,
        "prompt_tokens": len(tokens),
        "model_root": str(args.model_root),
    }

    records: list[dict[str, object]] = []
    try:
        llm = LLM(str(args.model_root), quant="gguf_ud_q4_k_xl", backend="hip_gfx1151")
        generator = llm._get_text_generator()
        try:
            llm.prepare(max_sequence_length=args.max_sequence_length)
            # Reset after construction and prepare, so allocator growth and
            # graph instantiation are not attributed to the measured pass.
            launch_census.reset()
            prefill_started = time.perf_counter()
            request = GenerationRequest(
                prompts=(tuple(int(t) for t in tokens),),
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                ignore_eos=True,
            )
            with wmma_prefill_dispatch_recorder(records):
                outputs = generator.generate_detailed(request)
            prefill_seconds = time.perf_counter() - prefill_started
            census = launch_census.snapshot()
            report["generated_token_ids"] = [
                list(row.generated_token_ids or ()) for row in outputs
            ]
            report["prefill_seconds"] = prefill_seconds
            report["selectors"] = summarise_selectors(records)
            report["dense_wide_engaged"] = any(
                row["selector"] == "dense_wide" and row["engaged"] for row in records
            )
            report["launched_symbols"] = summarise_launches(census)
            report["kernel_census_total_launches"] = census.get("total_launches")
            # Written here rather than after the teardown, so a failure in
            # teardown or in the summary printers cannot discard a completed
            # prefill.  Four earlier attempts lost the run that way.
            args.output.write_text(json.dumps(report, indent=1) + "\n")
        finally:
            generator.close()
            llm.close()
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    report.setdefault("kernel_census_total_launches", None)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"selectors", "launched_symbols"}}, indent=1))
    for row in report.get("selectors", []):
        print("SELECTOR", row)
    for row in report.get("launched_symbols", [])[:40]:
        print("LAUNCH", row)
    return 0


def summarise_launches(census: dict[str, object]) -> list[dict[str, object]]:
    """Collapse the launch census into one row per (quant, symbol)."""

    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    for row in census.get("rows", []) or []:
        counts[(str(row.get("quant")), str(row.get("symbol")))] += int(
            row.get("launches", 0)
        )
    return [
        {"quant": quant, "symbol": symbol, "launches": calls}
        for (quant, symbol), calls in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def summarise_selectors(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Collapse the raw records into one row per (selector, engaged, reason)."""

    counts: collections.Counter[tuple[str, bool, str]] = collections.Counter()
    for row in records:
        counts[(row["selector"], bool(row["engaged"]), row["reason"])] += 1
    return [
        {"selector": selector, "engaged": engaged, "reason": reason, "calls": calls}
        for (selector, engaged, reason), calls in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


if __name__ == "__main__":
    raise SystemExit(main())
