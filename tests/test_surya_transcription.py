"""GPU (hip_gfx1151) Surya full-page transcription acceptance.

Two independent questions, both required before a transcription claim:

* **Implementation parity** — the HIP lane reproduces the torch fp32 reference
  ids exactly under the checkpoint's real full-page HTML prompt.
* **Transcription quality** — the output actually reads the page. The reference
  is the text drawn on the fixture (``scripts/surya_bench_pages.py``), never a
  model run, so reproducing the oracle cannot satisfy it.

The old gate answered only the first question, and with the ad-hoc prompt
``"Transcribe this page."``, whose continuation is layout JSON or a degenerate
repeat. It could pass while the page was transcribed as garbage.

Skips without ROCm or the model checkpoint. Thresholds are calibrated on the
measured fp32 baseline recorded in the unit's worklog entry.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import pytest

from hipengine.generation.surya_protocol import (
    FULL_PAGE_HTML_PROMPT,
    parse_full_page_html,
)
from scripts.surya_bench_pages import GROUND_TRUTH, acceptance_page, expected_lines
from scripts.surya_transcription_score import (
    TranscriptionThresholds,
    evaluate,
    score_transcription,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP not available"),
]

FIXTURES = Path("tests/fixtures/surya")
MODEL_ID = "datalab-to/surya-ocr-2"
ORACLE_PATH = FIXTURES / "oracle_fullpage_protocol.json"
CASES = ("ja", "mixed", "dense", "table", "blank", "scan", "long")

# Pages whose full greedy id chain matches the torch fp32 reference exactly.
EXACT_ID_CASES = ("ja", "mixed", "dense", "table", "blank", "long")

# The degraded scan is the one page where hipEngine fp32 and torch fp32 pick
# different bbox digits. Measured: 21 of 616 ids differ (3.4%), the parsed
# labels and text are identical, and the worst coordinate delta is 4 of 1000.
# The coordinates are intrinsically fragile on this page: torch bf16 against
# torch fp32 differs on 15 ids and even changes the decoded text, so the drift
# is not specific to the HIP route. The gate therefore bounds the coordinate
# drift instead of demanding bit equality, and the text gate still has to pass.
COORDINATE_CASES = ("scan",)
MAX_BBOX_DELTA = 8.0
MAX_COORDINATE_ID_FRACTION = 0.05

# Calibrated on the fp32 gfx1151 baseline: every page measured recall 1.000,
# exact 1.000, CER 0.0000, 0 reading-order violations, table shape and cells
# 1.000, and a natural EOS. The bars below leave room for one character slip
# per page while still failing on any omission, reordering, or table damage.
THRESHOLDS = TranscriptionThresholds(
    min_line_recall=1.0,
    min_line_exact_rate=0.95,
    max_cer=0.01,
    max_reading_order_violations=0,
    min_table_cell_accuracy=1.0,
    require_table_shape=True,
    require_eos=True,
)

# A deliberately starved budget on a page that needs ~1050 tokens: the gate
# must report truncation and omissions rather than pass the prefix.
TRUNCATION_CASE = ("dense", 256)


def _model_dir() -> Path:
    from hipengine.loading.surya import resolve_surya_path

    try:
        return resolve_surya_path(MODEL_ID)
    except FileNotFoundError:
        pytest.skip(f"{MODEL_ID} not in local HF cache")


def _oracle() -> dict:
    if not ORACLE_PATH.exists():
        pytest.skip(
            f"{ORACLE_PATH} not present; capture it with "
            "scripts/surya_oracle_greedy.py --case protocol"
        )
    return json.loads(ORACLE_PATH.read_text())


@pytest.fixture(scope="module")
def generator():
    from hipengine.generation.surya_gpu import SuryaOCRGeneratorGPU

    _model_dir()
    gen = SuryaOCRGeneratorGPU(model_path=MODEL_ID, max_seq=16384)
    yield gen
    gen.close()


def _run(generator, page: str, max_tokens: int):
    from hipengine.generation.registry import GenerationRequest

    return generator.generate_multimodal_detailed(
        FULL_PAGE_HTML_PROMPT,
        str(FIXTURES / acceptance_page(page)),
        GenerationRequest(
            prompts=[FULL_PAGE_HTML_PROMPT],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=False,
        ),
    )


@pytest.fixture(scope="module")
def outputs(generator) -> dict:
    """One generation per page, shared by the parity and quality tests."""

    oracle = _oracle()
    return {page: _run(generator, page, oracle[page]["max_tokens"]) for page in CASES}


@pytest.mark.parametrize("page", EXACT_ID_CASES)
def test_protocol_matches_torch_reference(page: str, outputs: dict) -> None:
    entry = _oracle()[page]

    assert entry["prompt"] == FULL_PAGE_HTML_PROMPT, "oracle used a different prompt"
    assert entry["page"] == acceptance_page(page), "oracle used a different page"

    result = outputs[page]
    assert list(result.generated_token_ids) == entry["ids"]
    assert result.finish_details.reason == entry["finish_reason"]


@pytest.mark.parametrize("page", COORDINATE_CASES)
def test_protocol_matches_torch_reference_up_to_coordinate_digits(
    page: str, outputs: dict
) -> None:
    """Bound the drift where the reference's own bbox digits are unstable."""

    entry = _oracle()[page]
    result = outputs[page]
    assert entry["page"] == acceptance_page(page)
    assert result.finish_details.reason == entry["finish_reason"]

    ours = list(result.generated_token_ids)
    diffs = sum(1 for a, b in zip(ours, entry["ids"]) if a != b)
    assert len(ours) == len(entry["ids"])
    assert diffs <= MAX_COORDINATE_ID_FRACTION * len(entry["ids"])

    # The transcription itself must be identical: same regions, same order,
    # same text. Only the coordinates may move, and only a little.
    hip_blocks = parse_full_page_html(result.text)
    ref_blocks = parse_full_page_html(entry["text"])
    assert [b.label for b in hip_blocks] == [b.label for b in ref_blocks]
    assert [b.text for b in hip_blocks] == [b.text for b in ref_blocks]
    for hip, ref in zip(hip_blocks, ref_blocks):
        assert hip.bbox is not None and ref.bbox is not None
        delta = max(abs(a - b) for a, b in zip(hip.bbox, ref.bbox))
        assert delta <= MAX_BBOX_DELTA, (
            f"bbox moved {delta} of 1000 for {hip.label}: {hip.bbox} vs {ref.bbox}"
        )


@pytest.mark.parametrize("page", CASES)
def test_transcription_meets_ground_truth(page: str, outputs: dict) -> None:
    entry = _oracle()[page]
    result = outputs[page]

    blocks = parse_full_page_html(result.text)
    score = score_transcription(
        page=page,
        blocks=blocks,
        expected_lines=expected_lines(page),
        finish_reason=result.finish_details.reason,
        expected_table=GROUND_TRUTH[page].get("table"),
    )
    passed, failures = evaluate(score, THRESHOLDS)

    assert passed, (
        f"{page}: {failures}\n"
        f"  recall={score.line_recall:.4f} exact={score.line_exact_rate:.4f} "
        f"cer={score.cer:.4f} order_violations={score.reading_order_violations}\n"
        f"  omissions={score.omissions}\n"
        f"  table: found={score.table_found} shape={score.table_shape_match} "
        f"header={score.table_header_match} cells={score.table_cell_accuracy:.4f}\n"
        f"  text={result.text[:400]!r}"
    )
    # The measured baseline is exact on every page; anything less is a real
    # regression in reading, not a threshold artifact.
    assert score.line_recall == 1.0
    assert score.line_exact_rate == 1.0
    assert score.cer == 0.0
    assert not score.truncated
    assert entry["finish_reason"] == "eos"


@pytest.mark.parametrize("page", CASES)
def test_reading_order_follows_the_drawn_order(page: str, outputs: dict) -> None:
    score = score_transcription(
        page=page,
        blocks=parse_full_page_html(outputs[page].text),
        expected_lines=expected_lines(page),
        finish_reason=outputs[page].finish_details.reason,
    )

    assert score.reading_order_violations == 0


def test_table_structure_is_read(generator) -> None:
    result = _run(generator, "table", _oracle()["table"]["max_tokens"])
    blocks = parse_full_page_html(result.text)

    score = score_transcription(
        page="table",
        blocks=blocks,
        expected_lines=expected_lines("table"),
        finish_reason=result.finish_details.reason,
        expected_table=GROUND_TRUTH["table"]["table"],
    )

    assert score.table_found
    assert score.table_shape_match
    assert score.table_header_match
    assert score.table_cell_accuracy == 1.0
    assert score.table_cells_matched == score.table_cells_expected == 32


def test_a_starved_budget_is_reported_as_truncation_not_a_pass(generator) -> None:
    """A prefix must fail the gate, not silently qualify the page."""

    page, budget = TRUNCATION_CASE
    result = _run(generator, page, budget)

    score = score_transcription(
        page=page,
        blocks=parse_full_page_html(result.text),
        expected_lines=expected_lines(page),
        finish_reason=result.finish_details.reason,
    )
    passed, failures = evaluate(score, THRESHOLDS)

    assert result.finish_details.reason == "length"
    assert score.truncated
    assert score.line_recall < 1.0
    assert score.omissions
    assert not passed
    assert any("truncated" in failure for failure in failures)
    assert any("line_recall" in failure for failure in failures)
