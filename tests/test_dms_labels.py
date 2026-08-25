from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kvcache.dms_capture import DMSCaptureWriter
from hipengine.kvcache.dms_labels import (
    build_dms_label_artifact,
    build_eviction_labels,
    future_attention_mass_cpu,
    load_dms_label_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_future_attention_mass_is_causal_and_excludes_protected_window() -> None:
    query = np.zeros((4, 1, 1), dtype=np.float32)
    key = np.zeros((4, 1, 1), dtype=np.float32)

    scores = future_attention_mass_cpu(query, key, window_size=1)

    np.testing.assert_allclose(scores[:, 0], [7.0 / 12.0, 1.0 / 4.0, 0.0, 0.0])


def test_future_attention_mass_keeps_kv_heads_independent() -> None:
    query = np.zeros((3, 2, 1), dtype=np.float32)
    key = np.zeros((3, 2, 1), dtype=np.float32)
    query[:, 0, 0] = 4.0
    key[:, 0, 0] = [4.0, -4.0, -4.0]
    query[:, 1, 0] = 4.0
    key[:, 1, 0] = [-4.0, 4.0, -4.0]

    scores = future_attention_mass_cpu(query, key, window_size=0)

    assert scores[0, 0] > scores[1, 0]
    assert scores[1, 1] > scores[0, 1]


def test_tiled_gpu_future_mass_matches_cpu_and_repeats_deterministically() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch HIP/CUDA device is unavailable")
    from scripts.qwen38_dms_build_labels import future_attention_mass_torch

    rng = np.random.default_rng(20260823)
    query = rng.normal(size=(6, 4, 3)).astype(np.float32)
    key = rng.normal(size=(6, 2, 3)).astype(np.float32)
    expected = future_attention_mass_cpu(query, key, window_size=1)

    first = future_attention_mass_torch(
        query,
        key,
        window_size=1,
        device="cuda",
        query_tile=2,
    )
    second = future_attention_mass_torch(
        query,
        key,
        window_size=1,
        device="cuda",
        query_tile=2,
    )

    np.testing.assert_allclose(first, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(first, second)

    strided_first = future_attention_mass_torch(
        query,
        key,
        window_size=1,
        device="cuda",
        query_tile=2,
        query_stride=2,
    )
    strided_second = future_attention_mass_torch(
        query,
        key,
        window_size=1,
        device="cuda",
        query_tile=2,
        query_stride=2,
    )
    assert strided_first.shape == expected.shape
    assert np.all(np.isfinite(strided_first))
    assert np.all(strided_first >= 0.0)
    np.testing.assert_array_equal(strided_first, strided_second)
    assert not np.array_equal(strided_first, first)


def test_eviction_labels_enforce_budget_window_and_position_tie_break() -> None:
    scores = np.asarray(
        [
            [0.0, 4.0],
            [0.0, 3.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    positions = np.arange(8, dtype=np.int32)

    labels, eligible, stats = build_eviction_labels(
        scores,
        positions=positions,
        current_position=7,
        window_size=2,
        target_compression_ratio=2,
    )

    np.testing.assert_array_equal(eligible, [True, True, True, True, True, False, False, False])
    np.testing.assert_array_equal(labels[:, 0], [True, True, False, False, False, False, False, False])
    np.testing.assert_array_equal(labels[:, 1], [False, False, False, True, True, False, False, False])
    assert stats[0]["evict_count"] == stats[1]["evict_count"] == 2
    assert stats[0]["target_historical_live_count"] == 3
    assert stats[0]["target_live_count"] == stats[1]["target_live_count"] == 6
    assert not np.any(labels[~eligible])


def test_eviction_budget_applies_cr_to_history_outside_window() -> None:
    positions = np.arange(768, dtype=np.int32)
    labels, eligible, stats = build_eviction_labels(
        np.zeros((768, 1), dtype=np.float64),
        positions=positions,
        current_position=767,
        window_size=256,
        target_compression_ratio=4,
    )

    assert int(np.count_nonzero(eligible)) == 511
    assert int(np.count_nonzero(labels)) == 383
    assert stats[0]["protected_count"] == 257
    assert stats[0]["target_historical_live_count"] == 128
    assert stats[0]["target_live_count"] == 385
    assert not np.any(labels[~eligible])


def _capture(tmp_path: Path) -> Path:
    writer = DMSCaptureWriter(
        tmp_path / "capture",
        model_path="/models/fixture.gguf",
        model_sha256="a" * 64,
        data_manifest_sha256="b" * 64,
        tokenizer_identity="fixture-tokenizer",
        tokenizer_sha256="c" * 64,
        physical_layer_ids=(3,),
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=2,
        hidden_size=2,
        teacher_topk=2,
    )
    writer.begin_sequence(
        sequence_id="fixture-sequence",
        token_ids=(10, 11, 12, 13),
        category="code",
        provenance={"dataset": "fixture", "split": "train"},
    )
    writer.capture_chunk(
        physical_layer_id=3,
        compact_layer_index=0,
        positions=np.arange(4, dtype=np.int32),
        hidden_bf16=np.arange(8, dtype=np.uint16).reshape(4, 2),
        query_f32=np.zeros((4, 2, 2), dtype=np.float32),
        key_f32=np.zeros((4, 1, 2), dtype=np.float32),
    )
    writer.capture_teacher_logits(np.asarray([0.0, 1.0, -1.0], dtype=np.float32))
    writer.finish_sequence()
    return writer.finalize()


def test_label_builder_writes_compact_hidden_labels_and_physical_map(tmp_path: Path) -> None:
    capture_manifest = _capture(tmp_path)

    label_manifest = build_dms_label_artifact(
        capture_manifest,
        tmp_path / "labels",
        target_compression_ratio=2,
        window_size=1,
    )
    manifest = load_dms_label_manifest(label_manifest, verify_shards=True)

    assert manifest["geometry"]["physical_layer_ids"] == [3]
    assert manifest["objective"] == {
        "method": "future_attention_distillation_v1",
        "target_compression_ratio": 2,
        "window_size": 1,
        "tie_break": "ascending_score_then_position",
    }
    assert manifest["summary"]["sequence_count"] == 1
    assert manifest["summary"]["shard_count"] == 1
    shard_path = label_manifest.parent / manifest["sequences"][0]["shards"][0]["path"]
    with np.load(shard_path, allow_pickle=False) as shard:
        assert shard["hidden_bf16"].dtype == np.uint16
        assert shard["future_attention_mass"].shape == (4, 1)
        assert shard["evict_labels"].dtype == np.bool_
        np.testing.assert_array_equal(shard["eligible_mask"], [True, True, False, False])
        np.testing.assert_array_equal(shard["evict_labels"][:, 0], [False, True, False, False])


def test_label_builder_rejects_structurally_malformed_capture_shard(tmp_path: Path) -> None:
    capture_manifest = _capture(tmp_path)
    manifest = json.loads(capture_manifest.read_text(encoding="utf-8"))
    shard_record = manifest["sequences"][0]["shards"][0]
    shard_path = capture_manifest.parent / shard_record["path"]
    with np.load(shard_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files if name != "key"}
    with shard_path.open("wb") as handle:
        np.savez(handle, **arrays)
    shard_record["nbytes"] = shard_path.stat().st_size
    shard_record["sha256"] = _sha256(shard_path)
    capture_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    capture_manifest.with_suffix(capture_manifest.suffix + ".sha256").write_text(
        _sha256(capture_manifest) + "\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="missing required arrays"):
        build_dms_label_artifact(
            capture_manifest,
            tmp_path / "labels",
            target_compression_ratio=2,
            window_size=1,
        )
