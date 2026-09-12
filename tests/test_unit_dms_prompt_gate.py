"""Full-suite codec evidence cannot silently omit heldouts or duplicate prompts."""
import json
from pathlib import Path

import pytest

from scripts.qwen38_dms_int8_prompt_gate import load_prompts


def test_full_canonical_suite_is_required(tmp_path):
    source = Path('benchmarks/prompts/mtpbench-code-general-ja.jsonl')
    rows = load_prompts(source)
    assert len(rows) == 10
    path = tmp_path / 'prompts.jsonl'
    for altered in (rows[:-1], rows + rows[:1]):
        path.write_text('\n'.join(json.dumps(row) for row in altered))
        with pytest.raises(ValueError, match='complete canonical'):
            load_prompts(path)
