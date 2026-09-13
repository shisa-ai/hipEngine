from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.gguf_llamacpp_matched_context import compare_logit_rows, load_reference_logits


def _write_dump(path: Path, values: np.ndarray, *, magic: bytes = b"HKVLOG1\0") -> None:
    rows, columns = values.shape
    path.write_bytes(struct.pack("<8sIII", magic, 1, rows, columns) + values.astype("<f4").tobytes())


def test_load_reference_logits_round_trips_header_and_rows(tmp_path: Path) -> None:
    expected = np.asarray([[1.0, 2.0, 3.0], [-4.0, 0.5, 7.0]], dtype=np.float32)
    path = tmp_path / "reference.bin"
    _write_dump(path, expected)

    actual = load_reference_logits(path)

    np.testing.assert_array_equal(actual, expected)


def test_load_reference_logits_rejects_wrong_magic(tmp_path: Path) -> None:
    path = tmp_path / "reference.bin"
    _write_dump(path, np.ones((1, 2), dtype=np.float32), magic=b"BADLOG1\0")

    with pytest.raises(ValueError, match="magic"):
        load_reference_logits(path)


def test_load_reference_logits_rejects_truncated_payload(tmp_path: Path) -> None:
    path = tmp_path / "reference.bin"
    path.write_bytes(struct.pack("<8sIII", b"HKVLOG1\0", 1, 2, 3) + b"\0" * 8)

    with pytest.raises(ValueError, match="payload size"):
        load_reference_logits(path)


def test_compare_logit_rows_reports_position_metrics_and_first_mismatch() -> None:
    reference = np.asarray([[4.0, 0.0, -1.0], [0.0, 4.0, -1.0]], dtype=np.float32)
    candidate = np.asarray([[3.0, 0.0, -1.0], [5.0, 4.0, -1.0]], dtype=np.float32)

    result = compare_logit_rows(reference, candidate)

    assert result["reference_top1"] == [0, 1]
    assert result["candidate_top1"] == [0, 0]
    assert result["top1_matches"] == [True, False]
    assert result["first_top1_mismatch"] == {"index": 1, "reference": 1, "candidate": 0}
    assert result["candidate_reference_top1_rank"] == [1, 2]
    assert len(result["kl"]) == 2
    assert result["top1_agreement"] == 0.5
