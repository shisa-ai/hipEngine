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
from scripts.surya_bench_pages import GROUND_TRUTH, acceptance_page, expected_lines
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-seq", type=int, default=16384)
    parser.add_argument("--cases", default=",".join(MAX_TOKENS))
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
            page_name = acceptance_page(page)
            budget = MAX_TOKENS[page]
            started = time.time()
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
            rows.append(row)
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
                "model run; reference lines are the visible text, so the fixtures "
                "for ja/mixed/scan/long are the non-clipping page_*_fit variants"
            ),
            "metrics": (
                "line_recall/omissions use a presence bar of normalized "
                "similarity >= 0.60 with one-to-one assignment; line_exact_rate "
                "requires normalized equality; cer is total character edit "
                "distance over total expected characters"
            ),
            "max_seq": args.max_seq,
            "tiled": True,
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
