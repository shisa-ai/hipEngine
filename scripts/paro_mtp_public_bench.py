#!/usr/bin/env python3
"""Compare public PARO MTP depths with true AR on one loaded owner.

Timings include the complete request. This diagnostic does not replace the
strict-teacher numerical gate or report decode-only throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

HELDOUT = frozenset({
    "code_markdown_table", "general_en_explain", "general_ja_explain", "mixed_ja_en_review",
})


def pad_prompt(tail: list[int], filler: list[int], length: int) -> list[int]:
    """Pad before the task while preserving its complete tokenized suffix."""
    count = length - len(tail)
    if count < 0 or (count and not filler):
        raise ValueError("padded length must fit the task and filler must be nonempty")
    return (filler * ((count + len(filler) - 1) // len(filler)))[:count] + tail if count else list(tail)


def validate_arm(arm: str, row: dict[str, Any], outputs: int) -> None:
    """Check emitted counts and the selected route, not just request success."""
    if len(row["ids"]) != outputs:
        raise AssertionError("public request did not emit the exact output count")
    if arm == "ar":
        if row["route"].get("specdec2_mtp2_used"):
            raise AssertionError("AR control selected speculative execution")
        return
    depth = int(arm[1:])
    spec = row["spec"]
    if not spec or spec["requested_budget"] != depth or spec["candidate_budget"] != depth:
        raise AssertionError("public requested/effective depth differs from the requested arm")
    if spec["mtp_output_tokens"] + spec["ar_output_tokens_in_cycles"] != outputs:
        raise AssertionError("public speculative output accounting does not balance")
    if spec["recoverable_failures"]:
        raise AssertionError("a speculative cycle failed and silently recovered")
    counts = row["route"].get("specdec2_mtp2_candidate_counts", [])
    if outputs >= depth + 2:
        if not counts or max(counts) != depth or not row["route"].get("specdec2_mtp2_used"):
            raise AssertionError("requested depth did not execute")
    if any(count < 1 or count > depth for count in counts):
        raise AssertionError("selected depth is outside the requested range")


def strict_mismatches(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Retain each first divergence so a failed screen remains inspectable."""
    reference = row["arms"]["ar"]["ids"]
    return [{"id": row["id"], "max_tokens": row["max_tokens"], "run": row["run"],
             "arm": arm, "first_mismatch": next(i for i, (a, b) in enumerate(zip(reference, item["ids"], strict=True)) if a != b)}
            for arm, item in row["arms"].items() if item["ids"] != reference]


def pooled(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool tokens over elapsed time; never average per-request rates."""
    if not rows:
        return {}
    arms = rows[0]["arms"]
    return {
        arm: {
            "requests": len(rows),
            "tokens": sum(len(r["arms"][arm]["ids"]) for r in rows),
            "whole_request_seconds": sum(r["arms"][arm]["wall_s"] for r in rows),
            "whole_request_tok_s": sum(len(r["arms"][arm]["ids"]) for r in rows)
            / sum(r["arms"][arm]["wall_s"] for r in rows),
            "ids_equal_ar": sum(r["arms"][arm]["ids"] == r["arms"]["ar"]["ids"] for r in rows),
        }
        for arm in arms
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Packed PARO model directory with MTP weights")
    p.add_argument("--output", type=Path, required=True, help="Diagnostic JSON result")
    p.add_argument("--prompts", type=Path, default=Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"))
    p.add_argument("--prompt-names", default="", help="Comma-separated prompt IDs; empty selects all")
    p.add_argument("--depths", default="1,2,3", help="Comma-separated explicit candidate depths")
    p.add_argument("--outputs", default="25,129", help="Comma-separated exact output counts")
    p.add_argument("--contexts", default="127,128,257,512,4096", help="Padded input lengths; empty disables")
    p.add_argument("--repetitions", type=int, default=2)
    p.add_argument("--max-sequence-length", type=int, default=8192)
    p.add_argument("--profile", choices=("strict", "production"), default="production")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    depths = [int(x) for x in args.depths.split(",")]
    lengths = [int(x) for x in args.outputs.split(",")]
    contexts = [int(x) for x in args.contexts.split(",") if x]
    if not depths or any(d not in (1, 2, 3) for d in depths):
        raise ValueError("candidate depths must be 1, 2, or 3")
    if args.repetitions < 1 or min(lengths) < 1:
        raise ValueError("output counts and repetitions must be positive")

    from hipengine import LLM, SamplingParams
    from hipengine.benchmark.provenance import collect_artifact_provenance
    from scripts.gguf_mtp_category_bench import load_prompt_rows
    from scripts.specdec2_perf_gfx1100_child import _latest_route_metadata

    llm = LLM(args.model, backend="hip_gfx1100", execution_profile=args.profile,
              max_active_requests=1, max_sequence_length=args.max_sequence_length,
              speculative_candidate_budget=depths[0])
    runtime_sources = [Path(p) for p in (
        "hipengine/speculative/mtp_native.py", "hipengine/generation/qwen35_paro_mtp2.py",
        "hipengine/generation/qwen35_paro.py", "hipengine/models/qwen35.py",
        "hipengine/speculative/paro_mtp_profiles.py", "hipengine/generation/sampling.py",
        "hipengine/generation/engine_loop.py",
    )]
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_sources}
    record: dict[str, Any] = {
        "runtime_source_sha256": source_hashes,
        "schema": 1, "status": "diagnostic", "performance_claim": False,
        "protocol": {
            "timing": "Complete public-library request, including prompt and provider work; not decode-only",
            "arms": ["ar", *[f"b{d}" for d in depths]], "depths": depths,
            "outputs": lengths, "extra_contexts": contexts,
            "warmups_per_case": 1, "repetitions": args.repetitions,
            "sampling": "greedy, ignore_eos=True", "kv": "bf16", "capacity": 1,
            "profile": args.profile, "heldout_ids": sorted(HELDOUT),
            "order": "Rotate AR/B1/B2/B3 arms each repetition in one loaded LLM owner; reset every request",
            "limitations": ["Timing is diagnostic, not a numerical or task-quality verdict.",
                            "Existing production strict-teacher findings remain open.",
                            "Depth is changed only through the LLM's requested/resolved budget fields; no kernel or provider replacement."],
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provenance": collect_artifact_provenance(
            repo_root=Path.cwd(), configured_backend="hip_gfx1100", resolved_backend="hip_gfx1100",
            target_arch="gfx1100", model_path=args.model, quant="w4_paro", kv_dtype="bf16",
            command=None, timing_protocol="Rotated same-owner public requests; complete wall time",
            warmups=1, repetitions=args.repetitions,
        ),
        "prompts": {}, "rows": [], "complete": False, "strict_failures": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    try:
        llm.prepare(max_sequence_length=args.max_sequence_length)
        prompts = load_prompt_rows(args.prompts)
        if args.prompt_names:
            names = set(args.prompt_names.split(","))
            prompts = [r for r in prompts if r["id"] in names]
        if not prompts:
            raise ValueError("no selected prompts")
        cases = [{**r, "tokens": list(llm.tokenize(r["prompt"])), "padded": False} for r in prompts]
        first_code = next(r for r in cases if r["category"] == "code") if contexts else None
        filler = list(llm.tokenize("This is background context for the task. "))
        for context in contexts:
            assert first_code is not None
            cases.append({**first_code, "id": f'{first_code["id"]}_p{context}', "padded": True,
                          "tokens": pad_prompt(first_code["tokens"], filler, context)})
        previous: dict[tuple[str, int, str], list[int]] = {}
        for case in cases:
            record["prompts"][case["id"]] = {"ids": case["tokens"], "category": case["category"], "padded": case["padded"]}
            for length in lengths:
                if len(case["tokens"]) + length > args.max_sequence_length:
                    raise ValueError("prompt and output exceed configured sequence length")
                for run in range(-1, args.repetitions):
                    row: dict[str, Any] = {"id": case["id"], "category": case["category"],
                        "heldout": case["id"] in HELDOUT, "padded": case["padded"],
                        "context": len(case["tokens"]), "max_tokens": length, "run": run,
                        "measured": run >= 0, "arms": {}}
                    arms = ["ar", *[f"b{d}" for d in depths]]
                    offset = (run + 1) % len(arms)
                    for arm in arms[offset:] + arms[:offset]:
                        if arm != "ar":
                            depth = int(arm[1:])
                            llm.speculative_candidate_budget_requested = depth
                            llm.speculative_candidate_budget = depth
                        params = SamplingParams(max_tokens=length, temperature=0., top_p=1., ignore_eos=True)
                        fn = llm.generate_detailed if arm == "ar" else llm.generate_speculative_mtp_detailed
                        start = time.perf_counter()
                        result = fn((tuple(case["tokens"]),), params)[0]
                        elapsed = time.perf_counter() - start
                        route = _latest_route_metadata(llm.live_loop_snapshot())
                        route.pop("scheduler_chunks", None)
                        item = {"ids": list(result.generated_token_ids), "wall_s": elapsed,
                                "route": route, "timing": result.telemetry.timing,
                                "spec": (result.telemetry.diagnostics or {}).get("specdec2_mtp2")}
                        validate_arm(arm, item, length)
                        key = (case["id"], length, arm)
                        if key in previous and item["ids"] != previous[key]:
                            raise AssertionError(f"same-arm repeat IDs changed: {key}")
                        previous[key] = item["ids"]
                        row["arms"][arm] = item
                    record["rows"].append(row)
                    if args.profile == "strict":
                        record["strict_failures"].extend(strict_mismatches(row))
                    save()
                    print(case["id"], length, run, {a: round(x["wall_s"], 4) for a, x in row["arms"].items()}, flush=True)
        measured = [r for r in record["rows"] if r["measured"] and not r["padded"]]
        record["pooled_whole_request"] = {
            str(length): {
                "all": pooled([r for r in measured if r["max_tokens"] == length]),
                "train": pooled([r for r in measured if r["max_tokens"] == length and not r["heldout"]]),
                "heldout": pooled([r for r in measured if r["max_tokens"] == length and r["heldout"]]),
                "categories": {category: pooled([r for r in measured if r["max_tokens"] == length and r["category"] == category])
                               for category in sorted({r["category"] for r in measured})},
            } for length in lengths
        }
        if {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_sources} != source_hashes:
            raise AssertionError("runtime sources changed during the depth benchmark")
        record["complete"] = True
        if args.profile == "strict" and record["strict_failures"]:
            raise AssertionError(f"strict public IDs differ from AR in {len(record['strict_failures'])} arm requests; see saved strict_failures")
    finally:
        llm.close()
        save()


if __name__ == "__main__":
    main()
