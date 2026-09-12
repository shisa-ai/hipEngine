from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from hipengine.kvcache import (
    DMSLinearSidecarSpec,
    DMSRetrofitConfig,
    DMSTrainingProvenance,
)
from hipengine.kvcache.dms_sidecar import (
    load_external_dms_sidecar,
    screen_external_sidecar,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded >> 16).astype(np.uint16)


def _write_sidecar(path: Path, weight: np.ndarray, bias: np.ndarray) -> None:
    tensors = {}
    data = bytearray()
    for name, values in (("bias", bias), ("weight", weight)):
        raw = _bf16_bits(values).tobytes(order="C")
        start = len(data)
        data.extend(raw)
        tensors[name] = {
            "dtype": "BF16",
            "shape": list(values.shape),
            "data_offsets": [start, len(data)],
        }
    header = json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()
    header += b" " * ((-len(header)) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def _config(tmp_path: Path) -> tuple[DMSRetrofitConfig, np.ndarray, np.ndarray]:
    weight = np.asarray(
        [[[1.0, -2.0, 0.5], [-1.0, 0.25, 2.0]]],
        dtype=np.float32,
    )
    bias = np.asarray([[0.5, -0.25]], dtype=np.float32)
    path = tmp_path / "sidecar.safetensors"
    _write_sidecar(path, weight, bias)
    sidecar = DMSLinearSidecarSpec(
        path=path.name,
        format="safetensors",
        dtype="bfloat16",
        weight_tensor="weight",
        bias_tensor="bias",
        weight_shape=weight.shape,
        bias_shape=bias.shape,
        sha256=_sha256(path),
        resolved_path=str(path),
    )
    training = DMSTrainingProvenance(
        method="future_attention_distillation_v1",
        data_manifest_sha256="a" * 64,
        trainer_commit="b" * 40,
        fastdms_reference_commit="c602b0ec3266da7f74d6a658b3dafcddb443fddd",
        seed=0,
    )
    config = DMSRetrofitConfig(
        schema_version=2,
        artifact_fingerprint="c" * 64,
        model_family="qwen35_dense_hybrid",
        decision_source="external_linear_sidecar_v1",
        physical_layer_ids=(3,),
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=2,
        hidden_size=3,
        input_stage="post_attn_rmsnorm_pre_q_projection",
        window_size=1,
        target_compression_ratio=2,
        alpha_scale=1.0,
        alpha_offset=0.0,
        borrowed_query_channel=None,
        zero_borrowed_query_channel=False,
        corrected_mask=False,
        trained_checkpoint=True,
        evidence_source="fixture",
        source_path=str(tmp_path / "dms_metadata.json"),
        sidecar=sidecar,
        training=training,
    )
    return config, weight, bias


def test_external_sidecar_loads_bf16_and_projects_without_mutating_hidden(tmp_path: Path) -> None:
    config, weight, bias = _config(tmp_path)
    source = load_external_dms_sidecar(config)
    hidden = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], dtype=np.float32)
    original = hidden.copy()

    logits, evict = source.project(hidden, physical_layer_id=3)

    expected = hidden @ weight[0].T + bias[0]
    np.testing.assert_allclose(logits, expected, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(evict, expected > 0.0)
    np.testing.assert_array_equal(hidden, original)
    assert source.compact_layer_index(3) == 0


def _label_manifest(tmp_path: Path, config: DMSRetrofitConfig) -> Path:
    root = tmp_path / "labels"
    root.mkdir()
    sequences = []
    categories = ("code", "general_en", "general_ja", "mixed_ja_en")
    hidden_template = np.asarray(
        [
            [-3.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    for index in range(4):
        path = root / f"seq-{index:06d}-layer-00.npz"
        mass = np.asarray(
            [[0.0, 5.0], [1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0], [5.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        )
        with path.open("wb") as handle:
            np.savez(
                handle,
                positions=np.arange(8, dtype=np.int32),
                token_ids=np.arange(8, dtype=np.int32) + index,
                hidden_bf16=_bf16_bits(hidden_template + index * 0.1),
                future_attention_mass=mass,
                eligible_mask=np.asarray([True] * 6 + [False] * 2, dtype=np.bool_),
                evict_labels=np.asarray(
                    [[True, False], [True, False], [True, True], [True, True], [False, True], [False, True], [False, False], [False, False]],
                    dtype=np.bool_,
                ),
            )
        sequences.append(
            {
                "sequence_index": index,
                "sequence_id": f"fixture-{index}",
                "category": categories[index],
                "token_count": 8,
                "token_ids_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                "provenance": {"dataset": "fixture", "split": "train" if index < 2 else "validation"},
                "teacher_logits": {"scope": "next_token_after_sequence", "vocab_size": 2, "topk": 2, "topk_token_ids": [0, 1], "topk_logits": [1.0, 0.0], "logsumexp": 1.313},
                "shards": [
                    {
                        "path": path.name,
                        "sha256": _sha256(path),
                        "nbytes": path.stat().st_size,
                        "physical_layer_id": 3,
                        "compact_layer_index": 0,
                        "rows": 8,
                        "hidden_shape": [8, 3],
                        "label_shape": [8, 2],
                        "per_head": [],
                    }
                ],
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "hipengine_dms_label_manifest",
        "created_at": "fixture",
        "source_capture": {"path": "fixture", "sha256": "d" * 64},
        "model": {"path": "/models/fixture.gguf", "sha256": config.artifact_fingerprint},
        "data_manifest_sha256": "a" * 64,
        "tokenizer": {"identity": "fixture", "sha256": "e" * 64},
        "geometry": {
            "physical_layer_ids": [3],
            "num_layers": 1,
            "num_q_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 2,
            "hidden_size": 3,
            "input_stage": "post_attn_rmsnorm_pre_q_projection",
        },
        "objective": {"method": "future_attention_distillation_v1", "target_compression_ratio": 2, "window_size": 1, "tie_break": "ascending_score_then_position"},
        "compute": {"backend": "fixture"},
        "summary": {"sequence_count": 4, "token_count": 32, "shard_count": 4, "categories": {}},
        "sequences": sequences,
    }
    manifest_path = root / "label_manifest.json"
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(payload)
    manifest_path.with_suffix(manifest_path.suffix + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n",
        encoding="ascii",
    )
    return manifest_path


def test_sidecar_replay_screens_no_evict_cr2_cr4_cr8_deterministically(tmp_path: Path) -> None:
    config, _, _ = _config(tmp_path)
    source = load_external_dms_sidecar(config)
    labels = _label_manifest(tmp_path, config)

    first = screen_external_sidecar(labels, source, compression_ratios=(2, 4, 8))
    second = screen_external_sidecar(labels, source, compression_ratios=(2, 4, 8))

    assert first == second
    assert set(first["scenarios"]) == {"no_evict", "cr2", "cr4", "cr8"}
    assert first["calibration"]["splits"] == ["train"]
    assert first["evaluation"]["splits"] == ["validation"]
    assert first["scenarios"]["no_evict"]["global"]["actual_compression_ratio"] == 1.0
    assert all(
        scenario["global"]["protected_window_violations"] == 0
        for scenario in first["scenarios"].values()
    )
    assert first["quality"]["dense_vs_masked_logits"] == "unavailable_without_runtime_replay"
    assert first["deterministic"] is True
