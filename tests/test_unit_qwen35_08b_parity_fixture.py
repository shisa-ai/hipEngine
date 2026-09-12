from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.qwen35_08b_exact_core import _expand_rle, _load_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT / "benchmarks" / "fixtures" / "qwen35_08b_vulkan_parity_p512_t128.json"
)
LLAMA_HELPER = ROOT / "benchmarks" / "llama.cpp" / "qwen35_08b_exact_core.cpp"


def test_qwen35_08b_parity_fixture_hashes_and_shapes_are_exact() -> None:
    payload, prompt, continuation = _load_fixture(FIXTURE)
    assert prompt == [9707] * 512
    assert continuation == [9707] * 128
    assert payload["prompt"]["text"] == ".Q" * 512

    prompt_i64 = np.asarray(prompt, dtype="<i8")
    continuation_i64 = np.asarray(continuation, dtype="<i8")
    combined = np.concatenate((prompt_i64, continuation_i64))
    assert hashlib.sha256(payload["prompt"]["text"].encode()).hexdigest() == (
        payload["prompt"]["text_utf8_sha256"]
    )
    assert hashlib.sha256(prompt_i64.tobytes()).hexdigest() == (
        payload["prompt"]["token_ids_i64le_sha256"]
    )
    assert hashlib.sha256(continuation_i64.tobytes()).hexdigest() == (
        payload["teacher_forced_continuation"]["token_ids_i64le_sha256"]
    )
    assert hashlib.sha256(combined.tobytes()).hexdigest() == (
        payload["combined_token_ids_i64le_sha256"]
    )
    assert LLAMA_HELPER.is_file()


def test_qwen35_08b_parity_fixture_rejects_invalid_rle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _expand_rle([])
    with pytest.raises(ValueError, match="positive"):
        _expand_rle([[9707, 0]])

    payload = json.loads(FIXTURE.read_text())
    payload["prompt"]["count"] = 511
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="prompt RLE count"):
        _load_fixture(broken)
