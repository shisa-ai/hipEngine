"""Serve one real 300-DPI A4 page through the OpenAI HTTP API and score it.

The serving path was qualified against a fake generator: transport, bounds,
media adaptation, and prompt selection. This is the missing half — a loaded
checkpoint, a real request, and the same ground-truth scorer the direct-call
path is gated by. It asserts the HTTP transcription is *identical* to the direct
call rather than merely plausible.

    python3 scripts/surya_http_e2e.py
    python3 scripts/surya_http_e2e.py --page page_a4.png --max-tokens 4000
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

FIXTURES = Path("tests/fixtures/surya")
MODEL_ID = "datalab-to/surya-ocr-2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", default="page_a4.png")
    parser.add_argument("--case", default="a4", help="ground-truth case name")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--max-seq", type=int, default=16384)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--compare-artifact",
        type=Path,
        default=Path(
            "benchmarks/results/2026-09-12-gfx1151-surya-transcription-acceptance.json"
        ),
        help=(
            "committed direct-call artifact to assert equality against; the "
            "serving claim is that HTTP reproduces it, not that it is plausible"
        ),
    )
    args = parser.parse_args()

    from fastapi.testclient import TestClient

    from hipengine.generation.surya_protocol import FULL_PAGE_HTML_PROMPT, parse_full_page_html
    from hipengine.server import ServerConfig, create_app
    from scripts.surya_bench_pages import GROUND_TRUTH, expected_lines
    from scripts.surya_transcription_score import evaluate, score_transcription
    from scripts.surya_transcription_report import THRESHOLDS

    page = FIXTURES / args.page
    if not page.exists():
        raise SystemExit(f"{page} is not present")

    data_url = "data:image/png;base64," + base64.b64encode(page.read_bytes()).decode()
    config = ServerConfig(
        model=MODEL_ID,
        served_model_name="surya-ocr-2",
        max_context_tokens=args.max_seq,
        backend="hip_gfx1151",
        quant="fp32",
    )
    app = create_app(config)

    payload = {
        "model": "surya-ocr-2",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FULL_PAGE_HTML_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }

    with TestClient(app) as client:
        engine = app.state.hipengine_llm
        print(f"engine: {type(engine).__name__}")
        print(f"  supports_vision={engine.supports_vision}")
        print(f"  vision_max_pixels={getattr(engine, 'vision_max_pixels', None)}")
        print(f"  vision_media_input={getattr(engine, 'vision_media_input', None)!r}")
        print(f"  vision_prompt_marker={getattr(engine, 'vision_prompt_marker', None)!r}")

        started = time.time()
        response = client.post("/v1/chat/completions", json=payload)
        elapsed = time.time() - started
        if response.status_code != 200:
            print(f"HTTP {response.status_code}: {response.text[:2000]}", file=sys.stderr)
            return 1
        body = response.json()

    choice = body["choices"][0]
    text = choice["message"]["content"]
    blocks = parse_full_page_html(text)
    # The OpenAI wire vocabulary is stop/length; the scorer speaks the engine's
    # eos/length. `stop` means the generator stopped on its own EOS token.
    wire_reason = choice["finish_reason"]
    score_finish = "eos" if wire_reason == "stop" else "length"
    score = score_transcription(
        page=args.case,
        blocks=blocks,
        expected_lines=expected_lines(args.case),
        finish_reason=score_finish,
        expected_table=GROUND_TRUTH[args.case].get("table"),
    )
    passed, failures = evaluate(score, THRESHOLDS)

    print()
    print(f"finish_reason     {wire_reason} (scored as {score_finish!r})")
    print(f"completion_tokens {body['usage']['completion_tokens']}")
    print(f"prompt_tokens     {body['usage']['prompt_tokens']}")
    print(f"multimodal flag   {body['hipengine']['multimodal']}")
    print(f"http wall         {elapsed:.1f} s")
    print(f"blocks            {score.n_blocks} ({score.n_candidate_lines} candidate lines)")
    print(f"line_recall       {score.line_recall:.4f}  ({score.lines_found}/{score.lines_expected})")
    print(f"line_exact_rate   {score.line_exact_rate:.4f}")
    print(f"cer               {score.cer:.4f}")
    print(f"reading_order     {score.reading_order_violations} violations")
    print(f"table_cells       {score.table_cell_accuracy:.4f} "
          f"({score.table_cells_matched}/{score.table_cells_expected})")
    print(f"passed            {passed}")
    for failure in failures:
        print(f"  FAIL {failure}")

    # The claim is equality with the direct-call path, so assert it rather than
    # reporting a score that merely looks right.
    if args.compare_artifact is not None and args.compare_artifact.exists():
        artifact = json.loads(args.compare_artifact.read_text())
        rows = [r for r in artifact["results"] if r["page"] == args.case]
        if not rows:
            print(f"\n{args.compare_artifact} has no {args.case} row to compare")
        else:
            direct = rows[0]
            mismatches = []
            if body["usage"]["completion_tokens"] != direct["generated_tokens"]:
                mismatches.append(
                    f"completion_tokens {body['usage']['completion_tokens']} != "
                    f"direct {direct['generated_tokens']}"
                )
            for key in (
                "finish_reason", "n_blocks", "n_candidate_lines",
                "lines_expected", "lines_found", "line_recall",
                "line_exact_rate", "cer", "reading_order_violations",
                "table_cells_matched", "table_cell_accuracy",
            ):
                got = score_finish if key == "finish_reason" else score.as_dict()[key]
                if got != direct[key]:
                    mismatches.append(f"{key} {got!r} != direct {direct[key]!r}")
            if mismatches:
                print(f"\nHTTP does NOT match {args.compare_artifact}:")
                for mismatch in mismatches:
                    print(f"  {mismatch}")
                passed = False
            else:
                print(
                    f"\nHTTP matches the committed direct call "
                    f"({args.compare_artifact.name}): "
                    f"{direct['generated_tokens']} tokens, {direct['n_blocks']} blocks, "
                    f"{direct['lines_found']}/{direct['lines_expected']} units, "
                    f"cer {direct['cer']}"
                )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "page": args.page,
                    "case": args.case,
                    "model": MODEL_ID,
                    "prompt": FULL_PAGE_HTML_PROMPT,
                    "max_tokens": args.max_tokens,
                    "max_seq": args.max_seq,
                    "finish_reason": choice["finish_reason"],
                    "completion_tokens": body["usage"]["completion_tokens"],
                    "http_wall_s": round(elapsed, 3),
                    "score": score.as_dict(),
                    "passed": passed,
                    "failures": failures,
                    "text": text,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.out}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
