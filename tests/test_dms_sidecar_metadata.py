from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path

import pytest

from hipengine.kvcache import (
    DMSLinearSidecarSpec,
    DMSTrainingProvenance,
    load_dms_retrofit_config,
)

_MODEL_FAMILY = "qwen35_dense_hybrid"
_DECISION_SOURCE = "external_linear_sidecar_v1"
_INPUT_STAGE = "post_attn_rmsnorm_pre_q_projection"
_PHYSICAL_LAYERS = (3, 7)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_safetensors(
    path: Path,
    *,
    weight_shape: tuple[int, ...] = (2, 2, 4),
    bias_shape: tuple[int, ...] = (2, 2),
    weight_dtype: str = "BF16",
    bias_dtype: str = "BF16",
) -> None:
    bytes_per_value = {"BF16": 2, "F32": 4}
    tensors: dict[str, dict[str, object]] = {}
    data = bytearray()
    for name, shape, dtype in (
        ("weight", weight_shape, weight_dtype),
        ("bias", bias_shape, bias_dtype),
    ):
        count = 1
        for dim in shape:
            count *= int(dim)
        start = len(data)
        data.extend(bytes(count * bytes_per_value[dtype]))
        tensors[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(data)],
        }
    header = json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode("utf-8")
    padding = (-len(header)) % 8
    header += b" " * padding
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def _metadata(model: Path, sidecar: Path) -> dict[str, object]:
    return {
        "schema_version": 2,
        "artifact_fingerprint": _sha256(model),
        "model_family": _MODEL_FAMILY,
        "decision_source": _DECISION_SOURCE,
        "physical_layer_ids": list(_PHYSICAL_LAYERS),
        "num_layers": 2,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 4,
        "hidden_size": 4,
        "input_stage": _INPUT_STAGE,
        "window_size": 2,
        "target_compression_ratio": 4,
        "alpha_scale": 1.0,
        "alpha_offset": 0.0,
        "borrowed_query_channel": None,
        "zero_borrowed_query_channel": False,
        "sidecar": {
            "path": sidecar.name,
            "format": "safetensors",
            "dtype": "bfloat16",
            "weight_tensor": "weight",
            "bias_tensor": "bias",
            "weight_shape": [2, 2, 4],
            "bias_shape": [2, 2],
            "sha256": _sha256(sidecar),
        },
        "training": {
            "method": "future_attention_distillation_v1",
            "data_manifest_sha256": "b" * 64,
            "trainer_commit": "c" * 40,
            "fastdms_reference_commit": "c602b0ec3266da7f74d6a658b3dafcddb443fddd",
            "seed": 0,
        },
        "trained_checkpoint": True,
        "evidence_source": "fixture://external-sidecar",
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"exact-q4-fixture")
    sidecar = tmp_path / "sidecar.safetensors"
    _write_safetensors(sidecar)
    payload = _metadata(model, sidecar)
    metadata = tmp_path / "dms_metadata.json"
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    return model, sidecar, metadata, payload


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_schema_v2_loads_and_binds_external_linear_sidecar(tmp_path: Path) -> None:
    model, sidecar, metadata, payload = _fixture(tmp_path)

    config = load_dms_retrofit_config(
        model,
        metadata_path=metadata,
        expected_artifact_fingerprint=str(payload["artifact_fingerprint"]),
        expected_physical_layer_ids=_PHYSICAL_LAYERS,
    )

    assert config.schema_version == 2
    assert config.decision_source == _DECISION_SOURCE
    assert config.physical_layer_ids == _PHYSICAL_LAYERS
    assert config.hidden_size == 4
    assert config.input_stage == _INPUT_STAGE
    assert config.borrowed_query_channel is None
    assert config.zero_borrowed_query_channel is False
    assert config.prefill_selection_mode == "threshold"
    assert config.corrected_mask is False
    assert isinstance(config.sidecar, DMSLinearSidecarSpec)
    assert config.sidecar.resolved_path == str(sidecar.resolve())
    assert config.sidecar.weight_shape == (2, 2, 4)
    assert config.sidecar.bias_shape == (2, 2)
    assert isinstance(config.training, DMSTrainingProvenance)
    assert config.training.seed == 0
    assert len(config.fingerprint) == 64


def test_schema_v2_binds_exact_budget_prefill_selection(tmp_path: Path) -> None:
    model, _, metadata, payload = _fixture(tmp_path)
    payload["prefill_selection_mode"] = "exact_budget"
    _write_metadata(metadata, payload)

    config = load_dms_retrofit_config(model, metadata_path=metadata)

    assert config.prefill_selection_mode == "exact_budget"

    payload["prefill_selection_mode"] = "adaptive_magic"
    _write_metadata(metadata, payload)
    with pytest.raises(ValueError, match="prefill_selection_mode"):
        load_dms_retrofit_config(model, metadata_path=metadata)


def test_schema_v2_rejects_unsupported_schema_before_interpreting_fields(
    tmp_path: Path,
) -> None:
    model, _, metadata, payload = _fixture(tmp_path)
    payload["schema_version"] = 3
    _write_metadata(metadata, payload)

    with pytest.raises(ValueError, match="unsupported DMS metadata schema 3"):
        load_dms_retrofit_config(model, metadata_path=metadata)


def test_schema_v2_rejects_unverified_non_file_model_identity(tmp_path: Path) -> None:
    model, _, metadata, _ = _fixture(tmp_path)
    model.unlink()
    model.mkdir()

    with pytest.raises(ValueError, match="verified model artifact fingerprint"):
        load_dms_retrofit_config(model, metadata_path=metadata)


def test_schema_v2_rejects_wrong_model_hash_and_layer_map(tmp_path: Path) -> None:
    model, _, metadata, payload = _fixture(tmp_path)

    with pytest.raises(ValueError, match="physical layer map"):
        load_dms_retrofit_config(
            model,
            metadata_path=metadata,
            expected_physical_layer_ids=(3, 11),
        )

    model.write_bytes(b"different-model-bytes")
    with pytest.raises(ValueError, match="model artifact hash"):
        load_dms_retrofit_config(
            model,
            metadata_path=metadata,
            expected_artifact_fingerprint=str(payload["artifact_fingerprint"]),
            expected_physical_layer_ids=_PHYSICAL_LAYERS,
        )


def test_schema_v2_rejects_invalid_internal_layer_map(tmp_path: Path) -> None:
    model, _, metadata, payload = _fixture(tmp_path)
    payload["physical_layer_ids"] = [3, 3]
    _write_metadata(metadata, payload)

    with pytest.raises(ValueError, match="physical_layer_ids"):
        load_dms_retrofit_config(model, metadata_path=metadata)


def test_schema_v2_rejects_missing_sidecar_and_hash_mismatch(tmp_path: Path) -> None:
    model, sidecar, metadata, payload = _fixture(tmp_path)
    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="sidecar"):
        load_dms_retrofit_config(model, metadata_path=metadata)

    _write_safetensors(sidecar)
    broken = copy.deepcopy(payload)
    broken["sidecar"]["sha256"] = "0" * 64  # type: ignore[index]
    _write_metadata(metadata, broken)
    with pytest.raises(ValueError, match="sidecar hash"):
        load_dms_retrofit_config(model, metadata_path=metadata)


def test_schema_v2_rejects_declared_and_file_tensor_shape_mismatches(tmp_path: Path) -> None:
    model, sidecar, metadata, payload = _fixture(tmp_path)
    broken = copy.deepcopy(payload)
    broken["sidecar"]["weight_shape"] = [2, 2, 5]  # type: ignore[index]
    _write_metadata(metadata, broken)
    with pytest.raises(ValueError, match="weight_shape"):
        load_dms_retrofit_config(model, metadata_path=metadata)

    _write_safetensors(sidecar, weight_shape=(2, 2, 5))
    payload["sidecar"]["sha256"] = _sha256(sidecar)  # type: ignore[index]
    _write_metadata(metadata, payload)
    with pytest.raises(ValueError, match="tensor shape"):
        load_dms_retrofit_config(model, metadata_path=metadata)


def test_schema_v2_rejects_safetensors_dtype_mismatch(tmp_path: Path) -> None:
    model, sidecar, metadata, payload = _fixture(tmp_path)
    _write_safetensors(sidecar, weight_dtype="F32")
    payload["sidecar"]["sha256"] = _sha256(sidecar)  # type: ignore[index]
    _write_metadata(metadata, payload)

    with pytest.raises(ValueError, match="tensor dtype"):
        load_dms_retrofit_config(model, metadata_path=metadata)
