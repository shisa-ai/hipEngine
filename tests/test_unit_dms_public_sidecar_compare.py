"""CPU checks for the public DMS comparison protocol (no GPU imports)."""
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "dms_public_sidecar_compare", Path(__file__).parents[1] / "scripts/dms_public_sidecar_compare.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_timing_keeps_post_prefill_window_distinct_from_kernel_decode():
    row = _MODULE.timing_row({"prefill_ms": 1000, "request_total_ms": 3000}, 128, 33)
    assert row["prefill_tokens_per_second"] == 128
    assert row["post_prefill_tokens_per_second"] == 16
    assert row["post_prefill_inclusive_ms"] == 2000


@pytest.mark.parametrize("total", [0, 999, 1000, float("inf"), float("nan")])
def test_invalid_timing_is_not_reported_as_throughput(total):
    with pytest.raises(ValueError):
        _MODULE.timing_row({"prefill_ms": 1000, "request_total_ms": total}, 128, 33)


def test_prompt_selection_is_deterministic_not_result_dependent(tmp_path):
    rows = [{"id": "a", "category": "code"}, {"id": "b", "category": "code"},
            {"id": "c", "category": "general_ja"}]
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert _MODULE.select_prompts(path) == [rows[0], rows[2]]


def test_exact_prompt_length_and_request_suffix_are_preserved():
    class Tokenizer:
        @staticmethod
        def tokenize(text):
            return tuple(text.encode())
    row = {"id": "a", "messages": [{"content": "What is 2+2?"}]}
    ids = _MODULE.make_prompt(Tokenizer(), row, 512)
    assert len(ids) == 512
    assert b"What is 2+2?" in bytes(ids)
    assert _MODULE.digest(ids) == _MODULE.digest(list(ids))
    with pytest.raises(ValueError):
        _MODULE.make_prompt(Tokenizer(), row, 1)
