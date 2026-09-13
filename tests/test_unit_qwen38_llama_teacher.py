from __future__ import annotations

import struct
from pathlib import Path

import pytest

from scripts.qwen38_llama_teacher import _teacher_rows


def test_llama_teacher_token_rows_require_fixed_shape(tmp_path: Path) -> None:
    path = tmp_path / "tokens.txt"
    path.write_text(
        ",".join(str(index) for index in range(9))
        + "\n"
        + ",".join(str(index) for index in range(9, 18))
        + "\n"
    )

    assert _teacher_rows(path, 2) == [list(range(9)), list(range(9, 18))]


def test_llama_teacher_token_rows_reject_wrong_count(tmp_path: Path) -> None:
    path = tmp_path / "tokens.txt"
    path.write_text("1,2,3\n")

    with pytest.raises(ValueError, match="teacher-step count"):
        _teacher_rows(path, 1)


def test_llama_teacher_prompt_binary_header_contract() -> None:
    encoded = b"Q38Q" + struct.pack("<II", 1, 10)
    assert encoded[:4] == b"Q38Q"
    assert struct.unpack("<II", encoded[4:]) == (1, 10)
