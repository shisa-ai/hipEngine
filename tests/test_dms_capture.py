from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kvcache.dms_capture import (
    DMS_CAPTURE_INPUT_STAGE,
    DMSCaptureWriter,
    load_dms_capture_manifest,
)


def _writer(tmp_path: Path) -> DMSCaptureWriter:
    return DMSCaptureWriter(
        tmp_path / "captures",
        model_path="/models/fixture.gguf",
        model_sha256="a" * 64,
        data_manifest_sha256="b" * 64,
        tokenizer_identity="fixture-tokenizer",
        tokenizer_sha256="c" * 64,
        physical_layer_ids=(3, 7),
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=4,
        hidden_size=4,
        input_stage=DMS_CAPTURE_INPUT_STAGE,
        qk_storage_dtype="float16",
        teacher_topk=3,
    )


def _chunk(start: int, *, rows: int = 2) -> tuple[np.ndarray, ...]:
    positions = np.arange(start, start + rows, dtype=np.int32)
    hidden = (np.arange(rows * 4, dtype=np.uint16) + start).reshape(rows, 4)
    query = (np.arange(rows * 4 * 4, dtype=np.float32) + start).reshape(rows, 4, 4)
    key = (np.arange(rows * 2 * 4, dtype=np.float32) + start).reshape(rows, 2, 4)
    return positions, hidden, query, key


def _capture_complete_sequence(writer: DMSCaptureWriter) -> None:
    writer.begin_sequence(
        sequence_id="train-code-0001",
        token_ids=(11, 12, 13, 14),
        category="code",
        provenance={"dataset": "fixture", "split": "train", "row": 1},
    )
    for physical_layer_id, compact_layer_index in ((3, 0), (7, 1)):
        for start in (0, 2):
            positions, hidden, query, key = _chunk(start)
            writer.capture_chunk(
                physical_layer_id=physical_layer_id,
                compact_layer_index=compact_layer_index,
                positions=positions,
                hidden_bf16=hidden,
                query_f32=query,
                key_f32=key,
            )
    writer.capture_teacher_logits(np.asarray([0.2, 2.0, -1.0, 1.0], dtype=np.float32))
    writer.finish_sequence()


def test_capture_writer_streams_checksummed_layer_chunks_and_topk(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _capture_complete_sequence(writer)

    manifest_path = writer.finalize()
    manifest = load_dms_capture_manifest(manifest_path, verify_shards=True)

    assert manifest["schema_version"] == 1
    assert manifest["model"]["sha256"] == "a" * 64
    assert manifest["geometry"]["physical_layer_ids"] == [3, 7]
    assert manifest["geometry"]["input_stage"] == DMS_CAPTURE_INPUT_STAGE
    assert manifest["storage"]["hidden_dtype"] == "bfloat16_bits_uint16"
    assert manifest["storage"]["qk_dtype"] == "float16"
    assert manifest["summary"] == {
        "sequence_count": 1,
        "token_count": 4,
        "shard_count": 4,
    }
    sequence = manifest["sequences"][0]
    assert sequence["sequence_id"] == "train-code-0001"
    assert sequence["teacher_logits"]["scope"] == "next_token_after_sequence"
    assert sequence["teacher_logits"]["topk_token_ids"] == [1, 3, 0]
    assert sequence["teacher_logits"]["topk_logits"] == pytest.approx([2.0, 1.0, 0.2])
    assert np.isfinite(sequence["teacher_logits"]["logsumexp"])

    first_shard = manifest_path.parent / sequence["shards"][0]["path"]
    with np.load(first_shard, allow_pickle=False) as payload:
        assert payload["hidden_bf16"].dtype == np.uint16
        assert payload["query"].dtype == np.float16
        assert payload["key"].dtype == np.float16
        np.testing.assert_array_equal(payload["token_ids"], [11, 12])
        np.testing.assert_array_equal(payload["positions"], [0, 1])

    companion = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    assert companion.read_text(encoding="ascii").strip() == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


def test_capture_writer_rejects_incomplete_or_overlapping_layer_coverage(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    writer.begin_sequence(
        sequence_id="incomplete",
        token_ids=(1, 2, 3, 4),
        category="general_en",
        provenance={"dataset": "fixture", "split": "train"},
    )
    positions, hidden, query, key = _chunk(0)
    writer.capture_chunk(
        physical_layer_id=3,
        compact_layer_index=0,
        positions=positions,
        hidden_bf16=hidden,
        query_f32=query,
        key_f32=key,
    )
    with pytest.raises(ValueError, match="contiguous full-sequence coverage"):
        writer.finish_sequence()

    writer = _writer(tmp_path / "other")
    writer.begin_sequence(
        sequence_id="overlap",
        token_ids=(1, 2, 3, 4),
        category="general_en",
        provenance={"dataset": "fixture", "split": "train"},
    )
    writer.capture_chunk(
        physical_layer_id=3,
        compact_layer_index=0,
        positions=positions,
        hidden_bf16=hidden,
        query_f32=query,
        key_f32=key,
    )
    with pytest.raises(ValueError, match="next uncaptured position"):
        writer.capture_chunk(
            physical_layer_id=3,
            compact_layer_index=0,
            positions=positions,
            hidden_bf16=hidden,
            query_f32=query,
            key_f32=key,
        )


def test_capture_writer_rejects_wrong_layer_mapping_and_tensor_shapes(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.begin_sequence(
        sequence_id="bad",
        token_ids=(1, 2, 3, 4),
        category="mixed_ja_en",
        provenance={"dataset": "fixture", "split": "train"},
    )
    positions, hidden, query, key = _chunk(0)
    with pytest.raises(ValueError, match="compact layer index"):
        writer.capture_chunk(
            physical_layer_id=3,
            compact_layer_index=1,
            positions=positions,
            hidden_bf16=hidden,
            query_f32=query,
            key_f32=key,
        )
    with pytest.raises(ValueError, match="query_f32 shape"):
        writer.capture_chunk(
            physical_layer_id=3,
            compact_layer_index=0,
            positions=positions,
            hidden_bf16=hidden,
            query_f32=query[:, :3],
            key_f32=key,
        )


def test_capture_manifest_loader_detects_tampered_manifest(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _capture_complete_sequence(writer)
    manifest_path = writer.finalize()
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="capture manifest hash mismatch"):
        load_dms_capture_manifest(manifest_path, verify_shards=True)


def test_capture_manifest_loader_detects_tampered_shard(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _capture_complete_sequence(writer)
    manifest_path = writer.finalize()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard = manifest_path.parent / raw["sequences"][0]["shards"][0]["path"]
    tampered = bytearray(shard.read_bytes())
    tampered[-1] ^= 1
    shard.write_bytes(tampered)

    with pytest.raises(ValueError, match="capture shard hash mismatch"):
        load_dms_capture_manifest(manifest_path, verify_shards=True)
