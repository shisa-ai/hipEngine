"""Measure Surya full-page transcription quality against independent ground truth.

Runs the checkpoint's real full-page HTML prompt (``FULL_PAGE_HTML_PROMPT``)
over the transcription-acceptance fixtures and scores each page against the text
actually drawn on it (``scripts/surya_bench_pages.py``), emitting a compact
artifact. This is the measurement behind the acceptance test's thresholds and
the row that belongs in the benchmark rollup.

Usage:
    python3 scripts/surya_transcription_report.py \
        --out benchmarks/results/2026-09-12-gfx1151-surya-transcription-acceptance.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hipengine.generation.surya_protocol import (
    FULL_PAGE_HTML_PROMPT,
    parse_full_page_html,
)
from scripts.surya_bench_pages import GROUND_TRUTH, expected_lines, page_filename
from scripts.surya_transcription_score import evaluate, score_transcription
from scripts.surya_transcription_score import TranscriptionThresholds

MODEL_ID = "datalab-to/surya-ocr-2"
FIXTURES = Path("tests/fixtures/surya")

# The smallest budget each page needs to reach a natural EOS, measured on the
# fp32 HIP lane. A starved budget truncates, and truncation is scored as
# truncation rather than as a pass.
MAX_TOKENS = {
    "ja": 600,
    "mixed": 600,
    "dense": 1500,
    "table": 640,
    "blank": 64,
    "scan": 640,
    "long": 2600,
    # A real 300-DPI A4 page (grid 1x220x156, 8580 image tokens). It needs the
    # largest output budget in the suite because it is a full page of prose plus
    # a ruled table.
    "a4": 4000,
}

THRESHOLDS = TranscriptionThresholds(
    min_line_recall=1.0,
    min_line_exact_rate=0.95,
    max_cer=0.01,
    max_reading_order_violations=0,
    min_table_cell_accuracy=1.0,
    require_table_shape=True,
    require_eos=True,
)


def _host_identity() -> dict:
    cpu = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"cpu": cpu, "platform": platform.platform()}


def _stage_timer(runner: object) -> tuple[dict[str, float], list[tuple]]:
    """Wrap the runner's stage entry points to accumulate wall clock per stage.

    Reporting only: the wrappers call through unchanged, so the measured run is
    the production path. Decode steps are counted rather than individually
    timed, because per-step timing overhead would dominate a 20 ms step.
    """

    totals = {"vision_s": 0.0, "prefill_s": 0.0, "decode_s": 0.0, "decode_steps": 0.0}
    restore: list[tuple] = []

    def wrap(name: str, key: str):
        original = getattr(runner, name)

        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                totals[key] += time.perf_counter() - start
                if key == "decode_s":
                    totals["decode_steps"] += 1

        setattr(runner, name, timed)
        restore.append((name, original))

    wrap("vision_forward", "vision_s")
    wrap("prefill", "prefill_s")
    wrap("decode_step", "decode_s")
    return totals, restore


def _dump_hypothesis(
    directory: Path,
    page: str,
    raw_text: str,
    blocks: list,
    score: object,
    row: dict,
) -> None:
    """Write one page's raw output and its per-line verdict.

    A scored failure is otherwise only a count: this makes it readable, so the
    difference between "the model did not emit the text" and "the model emitted
    the text in a shape the line matcher cannot pair up" is a matter of reading
    rather than of guessing.
    """

    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{page}.html").write_text(raw_text, encoding="utf-8")
    lines = [
        f"score: {json.dumps({k: v for k, v in row.items() if k != 'failures'})}",
        f"blocks: {len(blocks)}",
        "",
        "--- candidate lines (model output, in order) ---",
    ]
    for index, candidate in enumerate(score.candidates):
        lines.append(f"[{index:3d}] blk{candidate.block_index:<4d} {candidate.text!r}")
    lines += ["", "--- expected lines and their assignment ---"]
    for match in score.matches:
        verdict = "EXACT" if match.exact else ("match" if match.present else "MISS ")
        lines.append(
            f"{verdict} sim={match.similarity:.3f} dist={match.distance:<4d} "
            f"exp={match.expected!r}"
        )
        if match.matched is not None and not match.exact:
            lines.append(f"          got={match.matched!r}")
    (directory / f"{page}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-seq", type=int, default=16384)
    parser.add_argument("--cases", default=",".join(MAX_TOKENS))
    parser.add_argument(
        "--stage-timings", action="store_true",
        help="record per-stage wall clock for each page (reporting only)",
    )
    parser.add_argument(
        "--dump", type=Path, default=None,
        help=(
            "write each page's raw hypothesis, its parsed blocks, and the "
            "per-line match verdict to this directory, so a scored failure can "
            "be read rather than guessed at"
        ),
    )
    args = parser.parse_args()

    from hipengine.generation.registry import GenerationRequest
    from hipengine.generation.surya_gpu import SuryaOCRGeneratorGPU

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in cases if c not in MAX_TOKENS]
    if unknown:
        raise SystemExit(f"unknown case(s) {unknown}; known: {sorted(MAX_TOKENS)}")

    generator = SuryaOCRGeneratorGPU(model_path=MODEL_ID, max_seq=args.max_seq)
    rows = []
    try:
        for page in cases:
            page_name = page_filename(page)
            budget = MAX_TOKENS[page]
            totals, restore = (
                _stage_timer(generator.runner) if args.stage_timings else ({}, [])
            )
            started = time.time()
            try:
                result = generator.generate_multimodal_detailed(
                    FULL_PAGE_HTML_PROMPT,
                    str(FIXTURES / page_name),
                    GenerationRequest(
                        prompts=[FULL_PAGE_HTML_PROMPT],
                        max_tokens=budget,
                        temperature=0.0,
                        top_p=1.0,
                        ignore_eos=False,
                    ),
                )
            finally:
                for name, original in restore:
                    setattr(generator.runner, name, original)
            elapsed = time.time() - started
            blocks = parse_full_page_html(result.text)
            score = score_transcription(
                page=page,
                blocks=blocks,
                expected_lines=expected_lines(page),
                finish_reason=result.finish_details.reason,
                expected_table=GROUND_TRUTH[page].get("table"),
            )
            passed, failures = evaluate(score, THRESHOLDS)
            row = score.as_dict()
            row.update(
                {
                    "page_fixture": page_name,
                    "max_tokens": budget,
                    "generated_tokens": len(result.generated_token_ids),
                    "seconds": round(elapsed, 3),
                    "passed": passed,
                    "failures": failures,
                }
            )
            if totals:
                steps = int(totals["decode_steps"])
                row["stages"] = {
                    "vision_s": round(totals["vision_s"], 3),
                    "prefill_s": round(totals["prefill_s"], 3),
                    "decode_s": round(totals["decode_s"], 3),
                    "decode_steps": steps,
                    "decode_ms_per_step": (
                        round(totals["decode_s"] / steps * 1000.0, 2) if steps else None
                    ),
                    "decode_tok_per_s": (
                        round(steps / totals["decode_s"], 2)
                        if totals["decode_s"] > 0 else None
                    ),
                }
            rows.append(row)
            if args.dump is not None:
                _dump_hypothesis(
                    args.dump, page, result.text, blocks, score, row
                )
            print(
                f"{page:6s} {row['generated_tokens']:5d} tok {row['finish_reason']:6s} "
                f"recall={score.line_recall:.4f} exact={score.line_exact_rate:.4f} "
                f"cer={score.cer:.4f} order_viol={score.reading_order_violations} "
                f"cells={score.table_cell_accuracy:.4f} passed={passed}"
            )
            for failure in failures:
                print(f"       FAIL {failure}")
            for omission in score.omissions:
                print(f"       OMIT {omission!r}")
    finally:
        generator.close()

    ok = all(row["passed"] for row in rows)
    artifact = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": MODEL_ID,
        "hardware": _host_identity(),
        "protocol": {
            "prompt": FULL_PAGE_HTML_PROMPT,
            "boundary": "page image plus prompt in, greedy full-page HTML out",
            "correctness_basis": (
                "text drawn on the fixture (scripts/surya_bench_pages.py), not a "
                "model run; every page is drawn through a fit assertion, so the "
                "reference lines are exactly the visible text"
            ),
            "metrics": (
                "line_recall/omissions use a presence bar of normalized "
                "similarity >= 0.60 with one-to-one assignment; line_exact_rate "
                "requires normalized equality; cer is total character edit "
                "distance over total expected characters"
            ),
            "max_seq": args.max_seq,
            "tiled": True,
            "stage_timings": bool(args.stage_timings),
        },
        "thresholds": {
            "min_line_recall": THRESHOLDS.min_line_recall,
            "min_line_exact_rate": THRESHOLDS.min_line_exact_rate,
            "max_cer": THRESHOLDS.max_cer,
            "max_reading_order_violations": THRESHOLDS.max_reading_order_violations,
            "min_table_cell_accuracy": THRESHOLDS.min_table_cell_accuracy,
            "require_table_shape": THRESHOLDS.require_table_shape,
            "require_eos": THRESHOLDS.require_eos,
        },
        "passed": ok,
        "results": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=1))
        print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
