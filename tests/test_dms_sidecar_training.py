from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from safetensors.torch import save_file as save_safetensors

from hipengine.kvcache import load_dms_retrofit_config
from scripts.qwen38_dms_train_sidecar import train_sidecar


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded >> 16).astype(np.uint16)


def _label_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "labels"
    root.mkdir()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"exact-fixture-model")
    rng = np.random.default_rng(1234)
    physical_layers = (3, 7)
    target_weight = np.asarray(
        [
            [[1.5, -0.5, 0.75, 0.25], [-1.0, 1.25, 0.5, -0.75]],
            [[0.5, 1.0, -1.25, 0.75], [1.0, -1.0, 1.0, -1.0]],
        ],
        dtype=np.float32,
    )
    target_bias = np.asarray([[0.1, -0.2], [0.0, 0.3]], dtype=np.float32)
    sequences = []
    categories = ("code", "general_en", "general_ja", "mixed_ja_en")
    for sequence_index in range(6):
        split = "train" if sequence_index < 4 else "validation"
        shards = []
        for compact_index, physical_layer_id in enumerate(physical_layers):
            hidden = rng.normal(size=(32, 4)).astype(np.float32)
            logits = hidden @ target_weight[compact_index].T + target_bias[compact_index]
            labels = logits > 0.0
            filename = f"seq-{sequence_index:06d}-layer-{compact_index:02d}.npz"
            path = root / filename
            with path.open("wb") as handle:
                np.savez(
                    handle,
                    positions=np.arange(32, dtype=np.int32),
                    token_ids=np.arange(32, dtype=np.int32) + sequence_index,
                    hidden_bf16=_bf16_bits(hidden),
                    future_attention_mass=np.zeros((32, 2), dtype=np.float32),
                    eligible_mask=np.ones((32,), dtype=np.bool_),
                    evict_labels=labels.astype(np.bool_),
                )
            shards.append(
                {
                    "path": filename,
                    "sha256": _sha256(path),
                    "nbytes": path.stat().st_size,
                    "physical_layer_id": physical_layer_id,
                    "compact_layer_index": compact_index,
                    "rows": 32,
                    "hidden_shape": [32, 4],
                    "label_shape": [32, 2],
                    "per_head": [],
                }
            )
        category = categories[sequence_index % len(categories)]
        sequences.append(
            {
                "sequence_index": sequence_index,
                "sequence_id": f"fixture-{sequence_index}",
                "category": category,
                "token_count": 32,
                "token_ids_sha256": hashlib.sha256(str(sequence_index).encode()).hexdigest(),
                "provenance": {"dataset": "fixture", "split": split},
                "teacher_logits": {
                    "scope": "next_token_after_sequence",
                    "vocab_size": 2,
                    "topk": 2,
                    "topk_token_ids": [0, 1],
                    "topk_logits": [1.0, 0.0],
                    "logsumexp": 1.3132616875,
                },
                "shards": shards,
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "hipengine_dms_label_manifest",
        "created_at": "fixture",
        "source_capture": {"path": "fixture", "sha256": "f" * 64},
        "model": {"path": str(model), "sha256": _sha256(model)},
        "data_manifest_sha256": "d" * 64,
        "tokenizer": {"identity": "fixture", "sha256": "e" * 64},
        "geometry": {
            "physical_layer_ids": list(physical_layers),
            "num_layers": 2,
            "num_q_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 2,
            "hidden_size": 4,
            "input_stage": "post_attn_rmsnorm_pre_q_projection",
        },
        "objective": {
            "method": "future_attention_distillation_v1",
            "target_compression_ratio": 2,
            "window_size": 1,
            "tie_break": "ascending_score_then_position",
        },
        "compute": {"backend": "fixture", "score_dtype": "float64"},
        "summary": {
            "sequence_count": len(sequences),
            "token_count": 32 * len(sequences),
            "shard_count": 2 * len(sequences),
            "categories": {},
        },
        "sequences": sequences,
    }
    manifest_path = root / "label_manifest.json"
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(payload)
    manifest_path.with_suffix(manifest_path.suffix + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n",
        encoding="ascii",
    )
    return manifest_path, model


def test_sidecar_training_reduces_loss_and_exports_stable_strict_artifact(
    tmp_path: Path,
) -> None:
    labels, model = _label_fixture(tmp_path)
    common = {
        "device": "cpu",
        "epochs": 18,
        "batch_size": 16,
        "learning_rate": 0.08,
        "budget_weight": 0.1,
        "weight_decay": 0.0,
        "seed": 7,
        "resume": False,
    }

    first = train_sidecar(labels, tmp_path / "run-a", **common)
    second = train_sidecar(labels, tmp_path / "run-b", **common)

    assert first["parameter_count"] == 20
    assert first["optimizer_parameter_count"] == 20
    assert first["optimizer_state_elements"] <= 3 * first["parameter_count"] + 2
    assert first["loss_history"][-1]["train_loss"] < first["loss_history"][0]["train_loss"]
    assert first["sidecar_sha256"] == second["sidecar_sha256"]
    assert Path(first["sidecar_path"]).read_bytes() == Path(second["sidecar_path"]).read_bytes()
    assert first["calibration"]["target_evictions"] == first["calibration"]["calibrated_evictions"]
    assert first["calibration"]["representation"] == "bfloat16_export"
    assert first["validation"]["global"]["accuracy"] > 0.85
    assert first["validation"]["by_layer_head"]
    assert first["validation"]["by_category"]
    assert first["validation"]["by_context_bucket"]

    config = load_dms_retrofit_config(
        model,
        metadata_path=first["metadata_path"],
        expected_artifact_fingerprint=_sha256(model),
        expected_physical_layer_ids=(3, 7),
    )
    assert config.schema_version == 2
    assert config.sidecar is not None
    assert config.sidecar.sha256 == first["sidecar_sha256"]
    assert config.sidecar.weight_shape == (2, 2, 4)
    assert config.sidecar.bias_shape == (2, 2)
    assert config.alpha_offset == first["calibration"]["alpha_offset"]


def test_sidecar_training_derives_disjoint_row_validation_and_loads_initial_sidecar(
    tmp_path: Path,
) -> None:
    labels, _ = _label_fixture(tmp_path)
    manifest = json.loads(labels.read_text(encoding="utf-8"))
    for sequence in manifest["sequences"]:
        sequence["provenance"]["split"] = "train"
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    labels.write_bytes(payload)
    labels.with_suffix(labels.suffix + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n",
        encoding="ascii",
    )
    initial = tmp_path / "initial.safetensors"
    initial_weight = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) / 32
    initial_bias = torch.tensor([[0.25, -0.5], [0.75, -1.0]], dtype=torch.float32)
    save_safetensors(
        {
            "bias": initial_bias.to(dtype=torch.bfloat16),
            "weight": initial_weight.to(dtype=torch.bfloat16),
        },
        str(initial),
    )

    result = train_sidecar(
        labels,
        tmp_path / "derived-validation",
        device="cpu",
        epochs=1,
        batch_size=32,
        learning_rate=0.01,
        budget_weight=0.1,
        weight_decay=0.0,
        seed=3,
        resume=False,
        derive_validation_modulus=4,
        initial_sidecar=initial,
    )

    assert result["initial_sidecar"] == {
        "path": str(initial.resolve()),
        "sha256": _sha256(initial),
    }
    assert result["validation_split"] == {
        "source": "deterministic_row_modulus",
        "modulus": 4,
        "heldout_remainder": 0,
    }
    # 6 sequences x 2 layers x 8 heldout rows x 2 KV-head decisions.
    assert result["validation"]["global"]["count"] == 192
    metadata = json.loads(Path(result["metadata_path"]).read_text())
    assert metadata["training"]["initial_sidecar_sha256"] == _sha256(initial)
    assert metadata["training"]["derived_validation_modulus"] == 4


def test_sidecar_training_resumes_only_sidecar_optimizer_state(tmp_path: Path) -> None:
    labels, _ = _label_fixture(tmp_path)
    output = tmp_path / "resume"
    first = train_sidecar(
        labels,
        output,
        device="cpu",
        epochs=2,
        batch_size=32,
        learning_rate=0.05,
        budget_weight=0.1,
        weight_decay=0.0,
        seed=9,
        resume=False,
    )
    resumed = train_sidecar(
        labels,
        output,
        device="cpu",
        epochs=4,
        batch_size=32,
        learning_rate=0.05,
        budget_weight=0.1,
        weight_decay=0.0,
        seed=9,
        resume=True,
    )

    assert first["completed_epochs"] == 2
    assert resumed["resumed_from_epoch"] == 2
    assert resumed["completed_epochs"] == 4
    assert resumed["parameter_count"] == resumed["optimizer_parameter_count"] == 20
