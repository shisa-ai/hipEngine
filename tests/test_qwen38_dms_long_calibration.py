from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from scripts.qwen38_dms_build_long_manifest import _mixed_tokens, _source_digest
from scripts.qwen38_dms_calibrate_long_bias import (
    _bf16_float,
    _live_summary,
    _write_bf16_safetensors,
)


def test_long_manifest_mixed_stream_alternates_bounded_chunks() -> None:
    en = list(range(20))
    ja = list(range(100, 120))

    mixed = _mixed_tokens(en, ja, target_tokens=12, chunk_tokens=3)

    assert mixed == [0, 1, 2, 100, 101, 102, 3, 4, 5, 103, 104, 105]


def test_long_manifest_source_digest_is_key_order_stable() -> None:
    left = [{"source_id": "a", "sha256": "1"}, {"source_id": "b", "sha256": "2"}]
    right = [{"sha256": "1", "source_id": "a"}, {"sha256": "2", "source_id": "b"}]

    assert _source_digest(left) == _source_digest(right)


def test_long_calibration_live_summary_counts_per_head_evictions() -> None:
    scores = np.asarray(
        [
            [[0.0, 5.0], [1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0], [5.0, 0.0]],
            [[5.0, 0.0], [4.0, 1.0], [3.0, 2.0], [2.0, 3.0], [1.0, 4.0], [0.0, 5.0]],
        ],
        dtype=np.float32,
    )
    thresholds = np.full((2, 2), 2.5, dtype=np.float32)

    summary = _live_summary(scores, thresholds, window=2)

    # Only the first four rows are eligible. Per-head evictions are [1,3] and
    # [3,1], so live counts are [5,3] and [3,5].
    assert summary["logical_rows"] == 24
    assert summary["live_rows"] == 16
    assert summary["per_layer_head_live_counts"] == [[5, 3], [3, 5]]
    assert summary["live_compression_ratio"] == 24 / 16


def test_long_calibration_writes_canonical_bf16_safetensors(tmp_path: Path) -> None:
    bias = _bf16_float(np.asarray([[0.25, -1.5]], dtype=np.float32))
    weight = _bf16_float(np.arange(12, dtype=np.float32).reshape(1, 2, 6) / 8)
    path = tmp_path / "sidecar.safetensors"

    _write_bf16_safetensors(path, bias=bias, weight=weight)

    raw = path.read_bytes()
    header_size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_size].decode().rstrip())
    assert list(header) == ["bias", "weight"]
    assert header["bias"] == {
        "data_offsets": [0, 4],
        "dtype": "BF16",
        "shape": [1, 2],
    }
    assert header["weight"]["dtype"] == "BF16"
    assert header["weight"]["shape"] == [1, 2, 6]
    assert len(raw) == 8 + header_size + bias.nbytes // 2 + weight.nbytes // 2
