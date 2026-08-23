"""FastDMS-derived compact retention backend and torch-free CPU oracle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil, prod
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from hipengine.core import Device, DType, Tensor
from hipengine.kvcache.backend import (
    ClaimLifetime,
    KVBackendSpec,
    KVBatchView,
    KVLease,
    KVPlaneView,
    KVPoolPlan,
    KVPoolSpec,
    KVStorageView,
    ResourceClaim,
    ResourceClaimSet,
    ResourceDelta,
)
from hipengine.kvcache.dms_device import (
    DMSDevicePayloadSnapshot,
    DMSDevicePayloadStore,
    DMSDeviceUnavailable,
    device_payloads_requested,
)
from hipengine.kvcache.ledger import FitAwareAdmissionController, ResourceLedger
from hipengine.kvcache.spans import KVLiveSpans, KVScaleMetadata

_CPU = Device("cpu", 0)
_DMS_SCHEMA_VERSION = 1
_DMS_SCHEMA_VERSIONS = {1, 2}
_DMS_BORROWED_QUERY_SOURCE = "borrowed_query_channel_v1"
_DMS_EXTERNAL_LINEAR_SOURCE = "external_linear_sidecar_v1"
_DMS_EXTERNAL_INPUT_STAGE = "post_attn_rmsnorm_pre_q_projection"
_DMS_CODECS = {"bf16", "int8_per_token_head"}
_HEX_DIGITS = frozenset("0123456789abcdef")


def _validated_hex(value: object, *, length: int, label: str) -> str:
    text = str(value)
    if len(text) != int(length) or any(char not in _HEX_DIGITS for char in text):
        raise ValueError(f"DMS {label} must be {length} lowercase hexadecimal characters")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DMSLinearSidecarSpec:
    """Strict external linear-decision sidecar tensor contract."""

    path: str
    format: str
    dtype: str
    weight_shape: tuple[int, ...]
    bias_shape: tuple[int, ...]
    sha256: str
    weight_tensor: str = "weight"
    bias_tensor: str = "bias"
    resolved_path: str = ""

    def __post_init__(self) -> None:
        for name in ("path", "format", "dtype", "weight_tensor", "bias_tensor"):
            value = str(getattr(self, name))
            if not value or value != value.strip():
                raise ValueError(f"DMS sidecar {name} must be a non-empty trimmed string")
            object.__setattr__(self, name, value)
        declared_path = Path(self.path)
        if declared_path.is_absolute() or ".." in declared_path.parts:
            raise ValueError("DMS sidecar path must stay relative to its metadata directory")
        if self.format != "safetensors":
            raise ValueError("DMS external sidecar format must be safetensors")
        if self.dtype != "bfloat16":
            raise ValueError("DMS external sidecar dtype must be bfloat16")
        if self.weight_tensor == self.bias_tensor:
            raise ValueError("DMS sidecar weight and bias tensor names must differ")
        for name in ("weight_shape", "bias_shape"):
            shape = tuple(int(dim) for dim in getattr(self, name))
            if not shape or any(dim <= 0 for dim in shape):
                raise ValueError(f"DMS sidecar {name} must contain positive dimensions")
            object.__setattr__(self, name, shape)
        object.__setattr__(
            self,
            "sha256",
            _validated_hex(self.sha256, length=64, label="sidecar sha256"),
        )
        if self.resolved_path:
            object.__setattr__(self, "resolved_path", str(Path(self.resolved_path).resolve()))

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "format": self.format,
            "dtype": self.dtype,
            "weight_tensor": self.weight_tensor,
            "bias_tensor": self.bias_tensor,
            "weight_shape": list(self.weight_shape),
            "bias_shape": list(self.bias_shape),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class DMSTrainingProvenance:
    """Immutable provenance required to admit a trained external sidecar."""

    method: str
    data_manifest_sha256: str
    trainer_commit: str
    fastdms_reference_commit: str
    seed: int

    def __post_init__(self) -> None:
        method = str(self.method)
        if method != "future_attention_distillation_v1":
            raise ValueError("unsupported DMS external-sidecar training method")
        object.__setattr__(self, "method", method)
        object.__setattr__(
            self,
            "data_manifest_sha256",
            _validated_hex(
                self.data_manifest_sha256,
                length=64,
                label="training data manifest sha256",
            ),
        )
        object.__setattr__(
            self,
            "trainer_commit",
            _validated_hex(self.trainer_commit, length=40, label="trainer commit"),
        )
        object.__setattr__(
            self,
            "fastdms_reference_commit",
            _validated_hex(
                self.fastdms_reference_commit,
                length=40,
                label="FastDMS reference commit",
            ),
        )
        seed = int(self.seed)
        if seed < 0:
            raise ValueError("DMS training seed must be non-negative")
        object.__setattr__(self, "seed", seed)

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "method": self.method,
            "data_manifest_sha256": self.data_manifest_sha256,
            "trainer_commit": self.trainer_commit,
            "fastdms_reference_commit": self.fastdms_reference_commit,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class DMSRetrofitConfig:
    """Checkpoint-bound borrowed-channel or external-sidecar DMS metadata."""

    artifact_fingerprint: str
    model_family: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    window_size: int
    target_compression_ratio: int
    alpha_scale: float
    alpha_offset: float
    borrowed_query_channel: int | None
    corrected_mask: bool
    trained_checkpoint: bool
    evidence_source: str
    source_path: str
    source_kind: str = "packaged_metadata"
    schema_version: int = _DMS_SCHEMA_VERSION
    decision_source: str = _DMS_BORROWED_QUERY_SOURCE
    physical_layer_ids: tuple[int, ...] = ()
    hidden_size: int | None = None
    input_stage: str | None = None
    zero_borrowed_query_channel: bool = True
    sidecar: DMSLinearSidecarSpec | None = None
    training: DMSTrainingProvenance | None = None

    def __post_init__(self) -> None:
        for name in (
            "artifact_fingerprint",
            "model_family",
            "evidence_source",
            "source_path",
            "source_kind",
            "decision_source",
        ):
            value = str(getattr(self, name))
            if not value or value != value.strip():
                raise ValueError(f"DMS {name} must be a non-empty trimmed string")
            object.__setattr__(self, name, value)
        schema_version = int(self.schema_version)
        if schema_version not in _DMS_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported DMS metadata schema {self.schema_version}; "
                f"expected one of {sorted(_DMS_SCHEMA_VERSIONS)}"
            )
        object.__setattr__(self, "schema_version", schema_version)
        for name in (
            "num_layers",
            "num_q_heads",
            "num_kv_heads",
            "head_dim",
            "window_size",
            "target_compression_ratio",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"DMS {name} must be positive")
            object.__setattr__(self, name, value)
        if self.num_q_heads % self.num_kv_heads:
            raise ValueError("DMS query heads must be divisible by KV heads")
        if not np.isfinite(float(self.alpha_scale)) or float(self.alpha_scale) == 0.0:
            raise ValueError("DMS alpha_scale must be finite and non-zero")
        if not np.isfinite(float(self.alpha_offset)):
            raise ValueError("DMS alpha_offset must be finite")
        object.__setattr__(self, "alpha_scale", float(self.alpha_scale))
        object.__setattr__(self, "alpha_offset", float(self.alpha_offset))
        object.__setattr__(
            self,
            "physical_layer_ids",
            tuple(int(layer_id) for layer_id in self.physical_layer_ids),
        )
        if not bool(self.trained_checkpoint):
            raise ValueError("DMS runtime requires a trained/retrofitted checkpoint")

        if schema_version == 1:
            self._validate_borrowed_query_schema()
        else:
            self._validate_external_sidecar_schema()

    def _validate_borrowed_query_schema(self) -> None:
        if self.decision_source != _DMS_BORROWED_QUERY_SOURCE:
            raise ValueError("DMS schema v1 only supports borrowed query decisions")
        if self.borrowed_query_channel is None:
            raise ValueError("DMS schema v1 requires a borrowed query channel")
        channel = int(self.borrowed_query_channel)
        if channel not in {-1, self.head_dim - 1}:
            raise ValueError("DMS borrowed query channel must be the last head channel")
        object.__setattr__(self, "borrowed_query_channel", self.head_dim - 1)
        if not bool(self.corrected_mask):
            raise ValueError("DMS runtime requires corrected-mask metadata")
        if not bool(self.zero_borrowed_query_channel):
            raise ValueError("DMS schema v1 must zero its borrowed query channel")
        if self.sidecar is not None or self.training is not None:
            raise ValueError("DMS schema v1 cannot declare an external sidecar")

    def _validate_external_sidecar_schema(self) -> None:
        if self.decision_source != _DMS_EXTERNAL_LINEAR_SOURCE:
            raise ValueError("DMS schema v2 requires external_linear_sidecar_v1")
        object.__setattr__(
            self,
            "artifact_fingerprint",
            _validated_hex(
                self.artifact_fingerprint,
                length=64,
                label="artifact fingerprint",
            ),
        )
        if self.borrowed_query_channel is not None:
            raise ValueError("external DMS sidecars cannot declare a borrowed query channel")
        if bool(self.zero_borrowed_query_channel):
            raise ValueError("external DMS sidecars must preserve ordinary query channels")
        if bool(self.corrected_mask):
            raise ValueError("external DMS sidecars cannot claim corrected-mask semantics")
        if self.source_kind == "training_log_diagnostic":
            raise ValueError("DMS schema v2 cannot load from a training-log fallback")
        layer_ids = self.physical_layer_ids
        if (
            len(layer_ids) != self.num_layers
            or len(set(layer_ids)) != len(layer_ids)
            or any(layer_id < 0 for layer_id in layer_ids)
            or tuple(sorted(layer_ids)) != layer_ids
        ):
            raise ValueError(
                "DMS schema v2 physical_layer_ids must be unique, sorted, non-negative, "
                "and match num_layers"
            )
        if self.hidden_size is None or int(self.hidden_size) <= 0:
            raise ValueError("DMS schema v2 hidden_size must be positive")
        object.__setattr__(self, "hidden_size", int(self.hidden_size))
        if self.input_stage != _DMS_EXTERNAL_INPUT_STAGE:
            raise ValueError(
                f"DMS schema v2 input_stage must be {_DMS_EXTERNAL_INPUT_STAGE!r}"
            )
        if self.sidecar is None or self.training is None:
            raise ValueError("DMS schema v2 requires sidecar and training provenance")
        expected_weight = (self.num_layers, self.num_kv_heads, self.hidden_size)
        expected_bias = (self.num_layers, self.num_kv_heads)
        if self.sidecar.weight_shape != expected_weight:
            raise ValueError(f"DMS sidecar weight_shape must be {expected_weight}")
        if self.sidecar.bias_shape != expected_bias:
            raise ValueError(f"DMS sidecar bias_shape must be {expected_bias}")

    @property
    def group_size(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def fingerprint(self) -> str:
        if self.schema_version == 1:
            # Preserve the schema-v1 identity byte-for-byte. Existing compact
            # backend artifacts use this fingerprint as a compatibility key.
            payload: dict[str, object] = {
                "schema_version": self.schema_version,
                "artifact_fingerprint": self.artifact_fingerprint,
                "model_family": self.model_family,
                "num_layers": self.num_layers,
                "num_q_heads": self.num_q_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "window_size": self.window_size,
                "target_compression_ratio": self.target_compression_ratio,
                "alpha_scale": self.alpha_scale,
                "alpha_offset": self.alpha_offset,
                "borrowed_query_channel": self.borrowed_query_channel,
                "corrected_mask": self.corrected_mask,
                "trained_checkpoint": self.trained_checkpoint,
            }
        else:
            payload = {
                "schema_version": self.schema_version,
                "artifact_fingerprint": self.artifact_fingerprint,
                "model_family": self.model_family,
                "decision_source": self.decision_source,
                "physical_layer_ids": list(self.physical_layer_ids),
                "num_layers": self.num_layers,
                "num_q_heads": self.num_q_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "hidden_size": self.hidden_size,
                "input_stage": self.input_stage,
                "window_size": self.window_size,
                "target_compression_ratio": self.target_compression_ratio,
                "alpha_scale": self.alpha_scale,
                "alpha_offset": self.alpha_offset,
                "borrowed_query_channel": self.borrowed_query_channel,
                "zero_borrowed_query_channel": self.zero_borrowed_query_channel,
                "corrected_mask": self.corrected_mask,
                "trained_checkpoint": self.trained_checkpoint,
            }
            assert self.sidecar is not None and self.training is not None
            payload["sidecar"] = self.sidecar.fingerprint_payload()
            payload["training"] = self.training.fingerprint_payload()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _resolve_sidecar_path(source: Path, declared_path: str) -> Path:
    base = source.parent.resolve()
    resolved = (base / declared_path).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError("DMS sidecar path escapes its metadata directory") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"DMS sidecar file is missing: {resolved}")
    return resolved


def _validate_safetensors_sidecar(spec: DMSLinearSidecarSpec) -> None:
    path = Path(spec.resolved_path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("DMS sidecar is not a complete safetensors file")
        header_size = int.from_bytes(prefix, "little", signed=False)
        if header_size <= 1 or header_size > size - 8:
            raise ValueError("DMS sidecar safetensors header length is invalid")
        try:
            header = json.loads(handle.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("DMS sidecar safetensors header is invalid") from exc
    if not isinstance(header, Mapping):
        raise TypeError("DMS sidecar safetensors header must be an object")
    tensor_names = {name for name in header if name != "__metadata__"}
    expected_names = {spec.weight_tensor, spec.bias_tensor}
    if tensor_names != expected_names:
        raise ValueError(
            f"DMS sidecar tensors must be exactly {sorted(expected_names)}, "
            f"got {sorted(tensor_names)}"
        )
    data_size = size - 8 - header_size
    ranges: list[tuple[int, int, str]] = []
    for name, expected_shape in (
        (spec.weight_tensor, spec.weight_shape),
        (spec.bias_tensor, spec.bias_shape),
    ):
        descriptor = header.get(name)
        if not isinstance(descriptor, Mapping):
            raise TypeError(f"DMS sidecar tensor descriptor is invalid for {name}")
        if descriptor.get("dtype") != "BF16":
            raise ValueError(f"DMS sidecar tensor dtype mismatch for {name}; expected BF16")
        shape_raw = descriptor.get("shape")
        if not isinstance(shape_raw, list) or tuple(int(dim) for dim in shape_raw) != expected_shape:
            raise ValueError(
                f"DMS sidecar tensor shape mismatch for {name}; expected {expected_shape}"
            )
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) for value in offsets)
        ):
            raise ValueError(f"DMS sidecar tensor offsets are invalid for {name}")
        start, end = (int(offsets[0]), int(offsets[1]))
        expected_bytes = prod(expected_shape) * 2
        if start < 0 or end - start != expected_bytes or end > data_size:
            raise ValueError(f"DMS sidecar tensor byte range is invalid for {name}")
        ranges.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise ValueError(f"DMS sidecar tensor ranges are not contiguous at {name}")
        cursor = end
    if cursor != data_size:
        raise ValueError("DMS sidecar contains undeclared trailing tensor bytes")


def _load_external_sidecar_spec(source: Path, raw: Mapping[str, Any]) -> DMSLinearSidecarSpec:
    sidecar_raw = raw.get("sidecar")
    if not isinstance(sidecar_raw, Mapping):
        raise TypeError("DMS schema v2 sidecar must be an object")
    declared_path = str(sidecar_raw.get("path", ""))
    resolved_path = _resolve_sidecar_path(source, declared_path)
    spec = DMSLinearSidecarSpec(
        path=declared_path,
        format=str(sidecar_raw.get("format", "")),
        dtype=str(sidecar_raw.get("dtype", "")),
        weight_tensor=str(sidecar_raw.get("weight_tensor", "weight")),
        bias_tensor=str(sidecar_raw.get("bias_tensor", "bias")),
        weight_shape=tuple(sidecar_raw.get("weight_shape", ())),
        bias_shape=tuple(sidecar_raw.get("bias_shape", ())),
        sha256=str(sidecar_raw.get("sha256", "")),
        resolved_path=str(resolved_path),
    )
    if _sha256_file(resolved_path) != spec.sha256:
        raise ValueError("DMS sidecar hash does not match metadata")
    return spec


def _load_training_provenance(raw: Mapping[str, Any]) -> DMSTrainingProvenance:
    training_raw = raw.get("training")
    if not isinstance(training_raw, Mapping):
        raise TypeError("DMS schema v2 training provenance must be an object")
    return DMSTrainingProvenance(
        method=str(training_raw.get("method", "")),
        data_manifest_sha256=str(training_raw.get("data_manifest_sha256", "")),
        trainer_commit=str(training_raw.get("trainer_commit", "")),
        fastdms_reference_commit=str(training_raw.get("fastdms_reference_commit", "")),
        seed=int(training_raw.get("seed", -1)),
    )


def load_dms_retrofit_config(
    model_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    expected_artifact_fingerprint: str | None = None,
    expected_physical_layer_ids: Sequence[int] | None = None,
    allow_training_log_fallback: bool = False,
) -> DMSRetrofitConfig:
    """Load strict checkpoint metadata; never infer DMS from an ordinary model."""

    model = Path(model_path).expanduser().resolve()
    if metadata_path is not None:
        source = Path(metadata_path).expanduser().resolve()
        source_kind = "explicit_metadata"
    else:
        root = model if model.is_dir() else model.parent
        packaged = root / "dms_metadata.json"
        training = root.parent / "training_log.json"
        if packaged.is_file():
            source = packaged
            source_kind = "packaged_metadata"
        elif allow_training_log_fallback and training.is_file():
            source = training
            source_kind = "training_log_diagnostic"
        else:
            raise FileNotFoundError(
                f"no packaged DMS metadata for {model}; expected {packaged}"
            )
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw = payload.get("dms", payload.get("config", payload))
    if not isinstance(raw, Mapping):
        raise TypeError("DMS metadata must contain an object")
    schema_version = int(payload.get("schema_version", raw.get("schema_version", 1)))
    if schema_version not in _DMS_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported DMS metadata schema {schema_version}; "
            f"expected one of {sorted(_DMS_SCHEMA_VERSIONS)}"
        )
    artifact = str(
        raw.get("artifact_fingerprint", payload.get("artifact_fingerprint", ""))
    )
    if expected_artifact_fingerprint is not None and artifact != str(
        expected_artifact_fingerprint
    ):
        raise ValueError("DMS metadata artifact fingerprint does not match checkpoint")

    sidecar = None
    training_provenance = None
    borrowed_query_channel: int | None
    if schema_version == 2:
        sidecar = _load_external_sidecar_spec(source, raw)
        training_provenance = _load_training_provenance(raw)
        borrowed_raw = raw.get("borrowed_query_channel")
        borrowed_query_channel = None if borrowed_raw is None else int(borrowed_raw)
    else:
        borrowed_query_channel = int(raw.get("borrowed_query_channel", -1))

    config = DMSRetrofitConfig(
        schema_version=schema_version,
        artifact_fingerprint=artifact,
        model_family=str(raw.get("model_family", "")),
        decision_source=str(
            raw.get(
                "decision_source",
                _DMS_EXTERNAL_LINEAR_SOURCE
                if schema_version == 2
                else _DMS_BORROWED_QUERY_SOURCE,
            )
        ),
        physical_layer_ids=tuple(raw.get("physical_layer_ids", ())),
        num_layers=int(raw.get("num_layers", 0)),
        num_q_heads=int(raw.get("num_q_heads", 0)),
        num_kv_heads=int(raw.get("num_kv_heads", 0)),
        head_dim=int(raw.get("head_dim", 0)),
        hidden_size=(None if raw.get("hidden_size") is None else int(raw["hidden_size"])),
        input_stage=(None if raw.get("input_stage") is None else str(raw["input_stage"])),
        window_size=int(raw.get("window_size", raw.get("dms_window_size", 0))),
        target_compression_ratio=int(
            raw.get("target_compression_ratio", raw.get("target_cr", 0))
        ),
        alpha_scale=float(raw.get("alpha_scale", raw.get("dms_alpha_scale", 0.0))),
        alpha_offset=float(raw.get("alpha_offset", raw.get("dms_alpha_offset", 0.0))),
        borrowed_query_channel=borrowed_query_channel,
        zero_borrowed_query_channel=bool(
            raw.get("zero_borrowed_query_channel", schema_version == 1)
        ),
        corrected_mask=bool(raw.get("corrected_mask", False)),
        trained_checkpoint=bool(raw.get("trained_checkpoint", False)),
        evidence_source=str(raw.get("evidence_source", "")),
        source_path=str(source),
        source_kind=source_kind,
        sidecar=sidecar,
        training=training_provenance,
    )
    if schema_version == 2:
        if model.is_file():
            if _sha256_file(model) != config.artifact_fingerprint:
                raise ValueError("DMS model artifact hash does not match metadata")
        elif expected_artifact_fingerprint is None:
            raise ValueError(
                "DMS schema v2 requires a verified model artifact fingerprint for non-file models"
            )
        if expected_physical_layer_ids is not None and config.physical_layer_ids != tuple(
            int(layer_id) for layer_id in expected_physical_layer_ids
        ):
            raise ValueError("DMS metadata physical layer map does not match model capability")
        assert config.sidecar is not None
        _validate_safetensors_sidecar(config.sidecar)
    if source_kind == "training_log_diagnostic" and not allow_training_log_fallback:
        raise AssertionError("unreachable DMS training-log admission")
    return config


@dataclass(frozen=True, slots=True)
class DMSCodecQualification:
    codec: str
    artifact_fingerprint: str
    kl_divergence: float
    top1_agreement: float
    no_dense_shadow: bool
    evidence_source: str

    def __post_init__(self) -> None:
        if self.codec != "int8_per_token_head":
            raise ValueError("only compact INT8 currently requires codec qualification")
        if not str(self.artifact_fingerprint).strip() or not str(
            self.evidence_source
        ).strip():
            raise ValueError("DMS codec qualification identity must be non-empty")
        if not 0.0 <= float(self.kl_divergence) <= 0.05:
            raise ValueError("DMS codec qualification requires KL <= 0.05")
        if not 0.90 <= float(self.top1_agreement) <= 1.0:
            raise ValueError("DMS codec qualification requires top-1 >= 90%")
        if not self.no_dense_shadow:
            raise ValueError("DMS codec qualification must prove no dense shadow")


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even BF16 bits (identical arithmetic to the bf16 payload codec)."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return (rounded >> np.uint32(16)).astype(np.uint16)


def encode_dms_payload(
    values: np.ndarray,
    *,
    codec: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 2 or array.shape[-1] <= 0:
        raise ValueError("DMS payload must have a non-empty feature dimension")
    if codec == "bf16":
        # NumPy has no portable BF16 scalar. FP32 with BF16-rounded mantissa is
        # the CPU oracle representation; device storage remains declared BF16.
        bits = array.view(np.uint32)
        rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000))
        return rounded.view(np.float32), None
    if codec != "int8_per_token_head":
        raise ValueError(f"unsupported DMS codec {codec!r}")
    maximum = np.max(np.abs(array), axis=-1, keepdims=True)
    scale = np.maximum(maximum / 127.0, np.finfo(np.float32).tiny).astype(np.float32)
    quantized = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
    return quantized, np.squeeze(scale, axis=-1)


def decode_dms_payload(
    payload: np.ndarray,
    scales: np.ndarray | None,
    *,
    codec: str,
) -> np.ndarray:
    if codec == "bf16":
        if scales is not None:
            raise ValueError("BF16 DMS payload cannot carry scales")
        return np.asarray(payload, dtype=np.float32)
    if codec != "int8_per_token_head" or scales is None:
        raise ValueError("INT8 DMS payload requires scales")
    return np.asarray(payload, dtype=np.float32) * np.expand_dims(
        np.asarray(scales, dtype=np.float32), axis=-1
    )


def extract_dms_eviction_decisions(
    q: np.ndarray,
    config: DMSRetrofitConfig,
    *,
    inplace: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one decision channel per GQA group and zero it before attention."""

    if config.decision_source != _DMS_BORROWED_QUERY_SOURCE:
        raise ValueError(
            "external DMS sidecars require hidden-state projection, not borrowed-Q extraction"
        )
    if config.borrowed_query_channel is None:
        raise AssertionError("borrowed-query DMS config lacks its decision channel")
    array = np.asarray(q)
    if array.ndim == 2:
        expected = config.num_q_heads * config.head_dim
        if array.shape[1] != expected:
            raise ValueError(f"expected flat Q width {expected}")
        view = array.reshape(array.shape[0], config.num_q_heads, config.head_dim)
    elif array.ndim == 3 and array.shape[1:] == (
        config.num_q_heads,
        config.head_dim,
    ):
        view = array
    else:
        raise ValueError("DMS Q must be [tokens,q_heads,head_dim] or flattened")
    cleaned = array if inplace else array.copy()
    clean_view = cleaned.reshape(view.shape)
    grouped = clean_view.reshape(
        clean_view.shape[0],
        config.num_kv_heads,
        config.group_size,
        config.head_dim,
    )
    decisions = grouped[:, :, 0, config.borrowed_query_channel].astype(np.float32)
    evict = decisions > (config.alpha_offset / config.alpha_scale)
    if config.alpha_scale < 0:
        evict = decisions * config.alpha_scale - config.alpha_offset > 0.0
    grouped[:, :, 0, config.borrowed_query_channel] = 0
    return cleaned, np.asarray(evict, dtype=np.bool_)


def build_dms_live_mask(
    evict_mask: np.ndarray,
    *,
    current_position: int,
    window_size: int,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    evict = np.asarray(evict_mask, dtype=np.bool_)
    if evict.ndim not in {2, 3}:
        raise ValueError("DMS eviction mask must be [heads,tokens] or [layers,heads,tokens]")
    if int(window_size) < 0:
        raise ValueError("DMS window_size must be non-negative")
    tokens = evict.shape[-1]
    token_positions = (
        np.arange(tokens, dtype=np.int64)
        if positions is None
        else np.asarray(positions, dtype=np.int64)
    )
    if token_positions.shape != (tokens,):
        raise ValueError("DMS positions must align with token axis")
    inside = int(current_position) - token_positions <= int(window_size)
    return (~evict) | inside.reshape((1,) * (evict.ndim - 1) + (tokens,))


def compact_attention_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    live_counts: np.ndarray,
    *,
    scale: float | None = None,
) -> np.ndarray:
    """Grouped-GQA compact decode oracle over variable per-head live counts."""

    query = np.asarray(q, dtype=np.float32)
    keys = np.asarray(k, dtype=np.float32)
    values = np.asarray(v, dtype=np.float32)
    counts = np.asarray(live_counts, dtype=np.int32)
    if query.ndim != 3 or keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("compact attention expects Q[B,QH,D], K/V[B,KVH,N,D]")
    batch, q_heads, dim = query.shape
    if keys.shape[0] != batch or keys.shape[3] != dim:
        raise ValueError("compact attention Q/K/V dimensions do not align")
    kv_heads = keys.shape[1]
    if q_heads % kv_heads or counts.shape != (batch, kv_heads):
        raise ValueError("compact attention requires aligned GQA live counts")
    factor = dim**-0.5 if scale is None else float(scale)
    output = np.zeros_like(query, dtype=np.float32)
    group = q_heads // kv_heads
    for row in range(batch):
        for kv_head in range(kv_heads):
            live = int(counts[row, kv_head])
            if live <= 0 or live > keys.shape[2]:
                raise ValueError("compact attention live count is outside storage")
            key_rows = keys[row, kv_head, :live]
            value_rows = values[row, kv_head, :live]
            q_slice = query[row, kv_head * group : (kv_head + 1) * group]
            logits = q_slice @ key_rows.T * factor
            logits -= np.max(logits, axis=-1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
            output[row, kv_head * group : (kv_head + 1) * group] = (
                probabilities @ value_rows
            )
    return output


@dataclass(frozen=True, slots=True)
class CompactExtent:
    layer_id: int
    head_id: int
    start: int
    length: int


@dataclass(slots=True)
class DMSSequenceState:
    request_id: int
    lease: KVLease
    extents: tuple[CompactExtent, ...]
    base_offsets: np.ndarray
    range_capacity: np.ndarray
    live_counts: np.ndarray
    token_positions: np.ndarray
    evict_mask: np.ndarray
    logical_tokens: int = 0
    k_payload: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    v_payload: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    k_scales: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    v_scales: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)


class CompactExtentPool:
    """Atomic per-layer extent allocator with deterministic coalescing."""

    def __init__(self, *, num_layers: int, slots_per_layer: int) -> None:
        if int(num_layers) <= 0 or int(slots_per_layer) <= 0:
            raise ValueError("compact extent dimensions must be positive")
        self.num_layers = int(num_layers)
        self.slots_per_layer = int(slots_per_layer)
        self._free = [
            [(0, self.slots_per_layer)] for _ in range(self.num_layers)
        ]
        self._owners: dict[str, tuple[CompactExtent, ...]] = {}
        self.allocation_failures = 0
        self.high_water_slots = 0

    def can_allocate(self, *, per_head_slots: int, num_heads: int) -> bool:
        needed = int(per_head_slots) * int(num_heads)
        return all(sum(length for _start, length in ranges) >= needed for ranges in self._free)

    def allocate(
        self,
        owner_id: str,
        *,
        per_head_slots: int,
        num_heads: int,
    ) -> tuple[CompactExtent, ...]:
        identifier = str(owner_id)
        if not identifier or identifier in self._owners:
            raise ValueError("compact extent owner must be new and non-empty")
        length = int(per_head_slots)
        heads = int(num_heads)
        if length <= 0 or heads <= 0:
            raise ValueError("compact extent request must be positive")
        allocated: list[CompactExtent] = []
        try:
            for layer in range(self.num_layers):
                for head in range(heads):
                    start = self._take(layer, length)
                    allocated.append(CompactExtent(layer, head, start, length))
        except Exception:
            for extent in reversed(allocated):
                self._give(extent.layer_id, extent.start, extent.length)
            self.allocation_failures += 1
            raise MemoryError("compact DMS extent capacity exhausted")
        result = tuple(allocated)
        self._owners[identifier] = result
        used = self.num_layers * self.slots_per_layer - self.free_slots
        self.high_water_slots = max(self.high_water_slots, used)
        return result

    def release(self, owner_id: str) -> tuple[CompactExtent, ...]:
        try:
            extents = self._owners.pop(str(owner_id))
        except KeyError as exc:
            raise KeyError(f"unknown compact extent owner {owner_id!r}") from exc
        for extent in extents:
            self._give(extent.layer_id, extent.start, extent.length)
        return extents

    @property
    def free_slots(self) -> int:
        return sum(
            length
            for ranges in self._free
            for _start, length in ranges
        )

    @property
    def largest_free_extent(self) -> int:
        return max(
            (length for ranges in self._free for _start, length in ranges),
            default=0,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "slots_per_layer": self.slots_per_layer,
            "capacity_slots": self.num_layers * self.slots_per_layer,
            "free_slots": self.free_slots,
            "largest_free_extent": self.largest_free_extent,
            "owner_count": len(self._owners),
            "allocation_failures": self.allocation_failures,
            "high_water_slots": self.high_water_slots,
            "free_ranges_by_layer": [list(ranges) for ranges in self._free],
        }

    def assert_conserved(self) -> None:
        occupied = sum(
            extent.length
            for extents in self._owners.values()
            for extent in extents
        )
        capacity = self.num_layers * self.slots_per_layer
        if occupied + self.free_slots != capacity:
            raise AssertionError("compact DMS extent conservation failure")
        for ranges in self._free:
            for index, (start, length) in enumerate(ranges):
                if start < 0 or length <= 0 or start + length > self.slots_per_layer:
                    raise AssertionError("compact DMS free extent is out of bounds")
                if index and ranges[index - 1][0] + ranges[index - 1][1] >= start:
                    raise AssertionError("compact DMS free extents are not coalesced")

    def _take(self, layer: int, length: int) -> int:
        ranges = self._free[layer]
        candidates = [
            (available, start, index)
            for index, (start, available) in enumerate(ranges)
            if available >= length
        ]
        if not candidates:
            raise MemoryError("compact DMS layer has no fitting extent")
        available, start, index = min(candidates)
        if available == length:
            ranges.pop(index)
        else:
            ranges[index] = (start + length, available - length)
        return start

    def _give(self, layer: int, start: int, length: int) -> None:
        ranges = self._free[layer]
        ranges.append((int(start), int(length)))
        ranges.sort()
        merged: list[tuple[int, int]] = []
        for current_start, current_length in ranges:
            if not merged:
                merged.append((current_start, current_length))
                continue
            prior_start, prior_length = merged[-1]
            prior_end = prior_start + prior_length
            if current_start <= prior_end:
                merged[-1] = (
                    prior_start,
                    max(prior_end, current_start + current_length) - prior_start,
                )
            else:
                merged.append((current_start, current_length))
        self._free[layer] = merged


@dataclass(frozen=True, slots=True)
class DMSOperation:
    operation_id: str
    lease: KVLease
    state_snapshot: tuple[np.ndarray, np.ndarray, np.ndarray]
    payload_snapshot: tuple[
        dict[tuple[int, int], np.ndarray],
        dict[tuple[int, int], np.ndarray],
        dict[tuple[int, int], np.ndarray],
        dict[tuple[int, int], np.ndarray],
    ]
    logical_tokens: int
    device_snapshot: DMSDevicePayloadSnapshot | None
    counter_snapshot: tuple[int, int, int]


class DMSCompactBackend:
    """Compact DMS topology whose codec does not alter scheduler lifecycle."""

    def __init__(
        self,
        *,
        retrofit: DMSRetrofitConfig,
        codec: str,
        slots_per_layer: int,
        max_request_rows: int,
        max_pack_rows: int,
        physical_widths: tuple[int, ...] = (1, 2, 4, 8),
        generation: int = 1,
        codec_qualification: DMSCodecQualification | None = None,
        device_payloads: bool | None = None,
    ) -> None:
        if codec not in _DMS_CODECS:
            raise ValueError(f"unsupported compact DMS codec {codec!r}")
        if device_payloads_requested(device_payloads) and codec != "bf16":
            raise ValueError("compact DMS device payloads are BF16-only")
        if codec == "int8_per_token_head":
            if codec_qualification is None:
                raise ValueError("compact INT8 DMS requires artifact qualification")
            if codec_qualification.artifact_fingerprint != retrofit.artifact_fingerprint:
                raise ValueError("compact INT8 qualification artifact mismatch")
        elif codec_qualification is not None:
            raise ValueError("BF16 compact DMS does not accept codec qualification")
        self.retrofit = retrofit
        self.codec = codec
        self.codec_qualification = codec_qualification
        self.slots_per_layer = int(slots_per_layer)
        self.max_request_rows = int(max_request_rows)
        self.max_pack_rows = int(max_pack_rows)
        self.generation = int(generation)
        if min(
            self.slots_per_layer,
            self.max_request_rows,
            self.max_pack_rows,
            self.generation,
        ) <= 0:
            raise ValueError("compact DMS capacities/generation must be positive")
        decision_bundle = (
            ""
            if retrofit.schema_version == 1
            else f"_{retrofit.decision_source}"
        )
        self.spec = KVBackendSpec(
            topology_key="dms_compact",
            hot_codec_key=codec,
            tier_key="device_only",
            layout_fingerprint=(
                f"dms-compact:{retrofit.fingerprint}:{codec}:v1"
            ),
            artifact_fingerprint=retrofit.artifact_fingerprint,
            prefix_mode="unsupported",
            transaction_mode="journal",
            kernel_bundle_key=f"dms_compact_{codec}{decision_bundle}_streaming_v1",
            physical_widths=physical_widths,
            max_context_tokens=self.slots_per_layer,
        )
        self.extents = CompactExtentPool(
            num_layers=retrofit.num_layers,
            slots_per_layer=self.slots_per_layer,
        )
        self._plan = self._build_plan()
        self.ledger = ResourceLedger(self._plan)
        self._states: dict[int, DMSSequenceState] = {}
        self._leases: dict[str, KVLease] = {}
        self._storage_view = self._build_storage_view()
        self.pack_calls = 0
        self.decode_appends = 0
        self.evicted_tokens = 0
        self._device_store: DMSDevicePayloadStore | None = None
        if device_payloads_requested(device_payloads):
            try:
                self._device_store = DMSDevicePayloadStore(
                    retrofit=retrofit,
                    slots_per_layer=self.slots_per_layer,
                    max_pack_rows=self.max_pack_rows,
                )
            except DMSDeviceUnavailable:
                # Host parent remains the registered fallback.
                self._device_store = None

    @property
    def device_payloads_enabled(self) -> bool:
        """True when the registered dms_compact kernels own the payloads."""
        return self._device_store is not None

    @property
    def storage_dtype(self) -> DType:
        return DType.BF16 if self.codec == "bf16" else DType.INT8_PER_TOKEN_HEAD

    @property
    def reservation_pool_ids(self) -> tuple[str, ...]:
        roles = ["dms.k_slots", "dms.v_slots", "dms.position_slots", "dms.evict_slots"]
        if self.codec == "int8_per_token_head":
            roles.extend(("dms.k_scale_slots", "dms.v_scale_slots"))
        return tuple(roles)

    def plan_pools(self, load_plan: Any) -> KVPoolPlan:
        del load_plan
        return self._plan

    def estimate(self, request: Any, prefix: Any, stage: Any) -> ResourceClaimSet:
        if prefix is not None:
            raise ValueError("compact DMS prefix reuse is disabled until overlay qualification")
        request_id = int(getattr(request, "request_id"))
        prompt_tokens = tuple(int(token) for token in getattr(request, "prompt_tokens", ()))
        stage_map: Mapping[str, Any] = stage if isinstance(stage, Mapping) else {}
        if str(stage_map.get("kind", "admission")) == "work_item":
            workspace_rows = int(stage_map.get("rows", stage_map.get("tokens", 1)))
            if workspace_rows <= 0 or workspace_rows > self.max_pack_rows:
                raise ValueError("compact DMS work-item rows exceed pack workspace")
            return ResourceClaimSet(
                claim_id=f"dms-work:{request_id}:{workspace_rows}",
                request_id=request_id,
                claims=(
                    ResourceClaim(
                        "dms.pack_workspace_rows",
                        workspace_rows,
                        ClaimLifetime.WORK_ITEM,
                    ),
                ),
                metadata=(("workspace_rows", workspace_rows),),
            )
        logical_prompt = int(stage_map.get("tokens", len(prompt_tokens)))
        max_new = int(stage_map.get("max_new_tokens", getattr(request, "max_new_tokens", 1)))
        if logical_prompt < 0 or max_new < 0:
            raise ValueError("compact DMS token counts must be non-negative")
        protected_prompt = min(
            logical_prompt,
            self.retrofit.window_size + 1,
        )
        eligible_prompt = logical_prompt - protected_prompt
        retained_prompt = protected_prompt + ceil(
            eligible_prompt / self.retrofit.target_compression_ratio
        )
        per_head = max(1, min(logical_prompt + max_new, retained_prompt + max_new))
        total_slots = (
            self.retrofit.num_layers * self.retrofit.num_kv_heads * per_head
        )
        claims = [
            ResourceClaim(pool_id, total_slots, ClaimLifetime.LEASE)
            for pool_id in self.reservation_pool_ids
        ]
        claims.extend(
            (
                ResourceClaim("dms.extent_descriptors", self.retrofit.num_layers * self.retrofit.num_kv_heads),
                ResourceClaim("dms.request_rows", 1),
            )
        )
        return ResourceClaimSet(
            claim_id=f"dms-admission:{request_id}:{per_head}",
            request_id=request_id,
            claims=tuple(claims),
            metadata=(
                ("per_head_slots", per_head),
                ("logical_prompt_tokens", logical_prompt),
                ("max_new_tokens", max_new),
                ("prefix_mode", "off"),
            ),
        )

    def reserve(self, claims: ResourceClaimSet) -> KVLease:
        if claims.request_id is None:
            raise ValueError("compact DMS reservation requires request_id")
        request_id = int(claims.request_id)
        lease_id = f"dms-lease:{request_id}"
        if request_id in self._states:
            raise ValueError("compact DMS request already has a lease")
        metadata = claims.metadata_dict()
        per_head = int(metadata.get("per_head_slots", 0))
        reservation = self.ledger.reserve_provisional(claims)
        extents: tuple[CompactExtent, ...] = ()
        try:
            extents = self.extents.allocate(
                lease_id,
                per_head_slots=per_head,
                num_heads=self.retrofit.num_kv_heads,
            )
            lease = KVLease(
                lease_id=lease_id,
                request_id=request_id,
                backend_fingerprint=self.spec.fingerprint,
                generation=self.generation,
                claims=claims.with_claim_id(f"dms-ownership:{request_id}"),
                private_handles=tuple(
                    f"extent:l{extent.layer_id}:h{extent.head_id}:"
                    f"{extent.start}+{extent.length}"
                    for extent in extents
                ),
                metadata_handles=(f"dms-row:{request_id}",),
            )
            shape = (self.retrofit.num_layers, self.retrofit.num_kv_heads)
            bases = np.asarray(
                [extent.start for extent in extents], dtype=np.int32
            ).reshape(shape)
            capacities = np.asarray(
                [extent.length for extent in extents], dtype=np.int32
            ).reshape(shape)
            state = DMSSequenceState(
                request_id=request_id,
                lease=lease,
                extents=extents,
                base_offsets=bases,
                range_capacity=capacities,
                live_counts=np.zeros(shape, dtype=np.int32),
                token_positions=np.full((*shape, per_head), -1, dtype=np.int32),
                evict_mask=np.zeros((*shape, per_head), dtype=np.bool_),
            )
            self._states[request_id] = state
            self._leases[lease_id] = lease
            self.ledger.commit(reservation, owner_id=lease_id)
            return lease
        except Exception:
            if extents:
                self.extents.release(lease_id)
            self.ledger.rollback(reservation)
            raise

    def streaming_pack(
        self,
        request_id: int,
        k: np.ndarray,
        v: np.ndarray,
        eviction: np.ndarray,
    ) -> None:
        """Pack surviving prompt rows directly; inputs are never retained."""

        state = self.state_for_request(request_id)
        key = np.asarray(k, dtype=np.float32)
        value = np.asarray(v, dtype=np.float32)
        evict = np.asarray(eviction, dtype=np.bool_)
        expected_prefix = (
            key.shape[0],
            self.retrofit.num_layers,
            self.retrofit.num_kv_heads,
        )
        if (
            key.ndim != 4
            or value.shape != key.shape
            or key.shape[1:3] != expected_prefix[1:]
            or key.shape[3] != self.retrofit.head_dim
        ):
            raise ValueError("DMS streaming pack expects K/V [tokens,layers,heads,dim]")
        if evict.shape != expected_prefix:
            raise ValueError("DMS streaming pack eviction shape mismatch")
        tokens = key.shape[0]
        positions = np.arange(tokens, dtype=np.int32)
        device_store = self._device_store
        for layer in range(self.retrofit.num_layers):
            for head in range(self.retrofit.num_kv_heads):
                keep = build_dms_live_mask(
                    evict[:, layer, head][None, :],
                    current_position=max(0, tokens - 1),
                    window_size=self.retrofit.window_size,
                    positions=positions,
                )[0]
                selected = positions[keep]
                live = len(selected)
                capacity = int(state.range_capacity[layer, head])
                if live > capacity:
                    raise MemoryError("DMS packed live rows exceed reserved extent")
                if device_store is None:
                    k_payload, k_scale = encode_dms_payload(
                        key[keep, layer, head, :], codec=self.codec
                    )
                    v_payload, v_scale = encode_dms_payload(
                        value[keep, layer, head, :], codec=self.codec
                    )
                    state.k_payload[(layer, head)] = k_payload
                    state.v_payload[(layer, head)] = v_payload
                    if k_scale is not None:
                        state.k_scales[(layer, head)] = k_scale
                        state.v_scales[(layer, head)] = v_scale
                state.live_counts[layer, head] = live
                state.token_positions[layer, head, :live] = selected
                state.token_positions[layer, head, live:] = -1
                state.evict_mask[layer, head, :live] = evict[keep, layer, head]
                state.evict_mask[layer, head, live:] = False
            if device_store is not None:
                device_store.pack_layer(
                    layer,
                    _bf16_bits(key[:, layer, :, :]),
                    _bf16_bits(value[:, layer, :, :]),
                    evict[:, layer, :].astype(np.uint8),
                    np.ascontiguousarray(state.base_offsets[layer, :]),
                    np.ascontiguousarray(state.range_capacity[layer, :]),
                )
        state.logical_tokens = tokens
        self.pack_calls += 1

    def append_decode(
        self,
        request_id: int,
        k: np.ndarray,
        v: np.ndarray,
        eviction: np.ndarray,
        *,
        position: int,
    ) -> None:
        state = self.state_for_request(request_id)
        key = np.asarray(k, dtype=np.float32)
        value = np.asarray(v, dtype=np.float32)
        evict_new = np.asarray(eviction, dtype=np.bool_)
        shape = (
            self.retrofit.num_layers,
            self.retrofit.num_kv_heads,
            self.retrofit.head_dim,
        )
        if key.shape != shape or value.shape != shape:
            raise ValueError("DMS decode append K/V shape mismatch")
        if evict_new.shape != shape[:2]:
            raise ValueError("DMS decode eviction shape mismatch")
        device_store = self._device_store
        if device_store is not None:
            # Device payload path: verify the full keep-recompute for every
            # (layer, head) before any device mutation, so overflow fails
            # atomically (the kernel would also fail closed per head; device
            # mode does not replicate the host parent's partial-update-on-
            # overflow artifact).
            recomputed: list[tuple[int, int, np.ndarray, np.ndarray]] = []
            for layer in range(shape[0]):
                for head in range(shape[1]):
                    live = int(state.live_counts[layer, head])
                    positions = state.token_positions[layer, head, :live]
                    prior_evict = state.evict_mask[layer, head, :live]
                    keep = (~prior_evict) | (
                        int(position) - positions <= self.retrofit.window_size
                    )
                    removed = int(live - np.count_nonzero(keep))
                    self.evicted_tokens += removed
                    combined_positions = np.concatenate(
                        (positions[keep], np.asarray([position], dtype=np.int32))
                    )
                    combined_evict = np.concatenate(
                        (prior_evict[keep], np.asarray([evict_new[layer, head]]))
                    )
                    if len(combined_positions) > int(state.range_capacity[layer, head]):
                        raise MemoryError("DMS decode extent has no evictable slot")
                    recomputed.append(
                        (layer, head, combined_positions, combined_evict)
                    )
            for layer in range(shape[0]):
                device_store.append_layer(
                    layer,
                    _bf16_bits(key[layer, :, :]),
                    _bf16_bits(value[layer, :, :]),
                    evict_new[layer, :].astype(np.uint8),
                    position,
                    np.ascontiguousarray(state.base_offsets[layer, :]),
                    np.ascontiguousarray(state.range_capacity[layer, :]),
                    np.ascontiguousarray(state.live_counts[layer, :]),
                )
            for layer, head, combined_positions, combined_evict in recomputed:
                live = int(len(combined_positions))
                capacity = int(state.range_capacity[layer, head])
                state.live_counts[layer, head] = live
                state.token_positions[layer, head, :live] = combined_positions
                state.token_positions[layer, head, live:capacity] = -1
                state.evict_mask[layer, head, :live] = combined_evict
                state.evict_mask[layer, head, live:capacity] = False
        else:
            for layer in range(shape[0]):
                for head in range(shape[1]):
                    live = int(state.live_counts[layer, head])
                    positions = state.token_positions[layer, head, :live]
                    prior_evict = state.evict_mask[layer, head, :live]
                    keep = (~prior_evict) | (
                        int(position) - positions <= self.retrofit.window_size
                    )
                    removed = int(live - np.count_nonzero(keep))
                    self.evicted_tokens += removed
                    prior_k = state.k_payload.get(
                        (layer, head), np.empty((0, shape[2]), dtype=np.float32)
                    )[keep]
                    prior_v = state.v_payload.get(
                        (layer, head), np.empty((0, shape[2]), dtype=np.float32)
                    )[keep]
                    if self.codec == "int8_per_token_head":
                        prior_k = decode_dms_payload(
                            prior_k,
                            state.k_scales[(layer, head)][keep],
                            codec=self.codec,
                        )
                        prior_v = decode_dms_payload(
                            prior_v,
                            state.v_scales[(layer, head)][keep],
                            codec=self.codec,
                        )
                    combined_k = np.concatenate((prior_k, key[layer, head][None, :]))
                    combined_v = np.concatenate((prior_v, value[layer, head][None, :]))
                    combined_positions = np.concatenate(
                        (positions[keep], np.asarray([position], dtype=np.int32))
                    )
                    combined_evict = np.concatenate(
                        (prior_evict[keep], np.asarray([evict_new[layer, head]]))
                    )
                    capacity = int(state.range_capacity[layer, head])
                    while len(combined_positions) > capacity:
                        eligible = np.flatnonzero(
                            combined_evict
                            & (
                                int(position) - combined_positions
                                > self.retrofit.window_size
                            )
                        )
                        if not len(eligible):
                            raise MemoryError("DMS decode extent has no evictable slot")
                        victim = int(eligible[0])
                        combined_k = np.delete(combined_k, victim, axis=0)
                        combined_v = np.delete(combined_v, victim, axis=0)
                        combined_positions = np.delete(combined_positions, victim)
                        combined_evict = np.delete(combined_evict, victim)
                        self.evicted_tokens += 1
                    k_payload, k_scale = encode_dms_payload(combined_k, codec=self.codec)
                    v_payload, v_scale = encode_dms_payload(combined_v, codec=self.codec)
                    state.k_payload[(layer, head)] = k_payload
                    state.v_payload[(layer, head)] = v_payload
                    if k_scale is not None:
                        state.k_scales[(layer, head)] = k_scale
                        state.v_scales[(layer, head)] = v_scale
                    live = len(combined_positions)
                    state.live_counts[layer, head] = live
                    state.token_positions[layer, head, :live] = combined_positions
                    state.token_positions[layer, head, live:] = -1
                    state.evict_mask[layer, head, :live] = combined_evict
                    state.evict_mask[layer, head, live:] = False
        state.logical_tokens = max(state.logical_tokens, int(position) + 1)
        self.decode_appends += 1

    def compact_decode_attention(
        self,
        request_id: int,
        layer: int,
        q: np.ndarray | None = None,
        *,
        q_ptr: int | None = None,
        out_ptr: int | None = None,
        scale: float | None = None,
    ) -> np.ndarray | None:
        """GQA decode attention over one request's compact extents on one layer.

        Takes either a host ``q`` (``[q_heads, dim]`` FP32, uploaded) or a
        device ``q_ptr``; returns the FP32 output (``[q_heads, dim]``) when
        no ``out_ptr`` is given, else writes to the device ``out_ptr`` and
        returns None. Device payloads must be enabled.
        """
        if self._device_store is None:
            raise ValueError("device payloads are not enabled on this backend")
        state = self.state_for_request(request_id)
        layer = int(layer)
        if not 0 <= layer < self.retrofit.num_layers:
            raise ValueError("layer out of range")
        if q is not None:
            query = np.ascontiguousarray(q, dtype=np.float32)
            if query.shape != (self.retrofit.num_q_heads, self.retrofit.head_dim):
                raise ValueError(
                    "DMS compact attention expects Q [q_heads, dim]"
                )
        out = (
            None
            if out_ptr is not None
            else np.zeros(
                (self.retrofit.num_q_heads, self.retrofit.head_dim),
                dtype=np.float32,
            )
        )
        self._device_store.attention_layer(
            layer,
            q=query if q is not None else None,
            q_ptr=q_ptr,
            out=out,
            out_ptr=out_ptr,
            base=np.ascontiguousarray(state.base_offsets[layer, :]),
            live=np.ascontiguousarray(state.live_counts[layer, :]),
            scale=scale,
        )
        return out

    def device_layer_view(self, request_id: int, layer: int) -> Any:
        """Read back one layer's device slot buffers (test/observability)."""
        if self._device_store is None:
            raise ValueError("device payloads are not enabled on this backend")
        self.state_for_request(request_id)
        layer = int(layer)
        if not 0 <= layer < self.retrofit.num_layers:
            raise ValueError("layer out of range")
        return self._device_store.layer_view(layer)

    def close(self) -> None:
        """Free device payload storage; the backend is unusable afterwards."""
        if self._device_store is not None:
            self._device_store.close()

    def prepare(self, work_item: Any) -> KVBatchView:
        request_ids = tuple(int(value) for value in getattr(work_item, "request_ids"))
        if not request_ids:
            raise ValueError("compact DMS prepare requires request IDs")
        states = [self.state_for_request(request_id) for request_id in request_ids]
        rows = len(states)
        layers = self.retrofit.num_layers
        heads = self.retrofit.num_kv_heads
        capacity = max(int(np.max(state.range_capacity)) for state in states)
        base = 0x6D000000 + self.generation * 0x100000
        scales = None
        if self.codec == "int8_per_token_head":
            scales = KVScaleMetadata(
                k_scale=Tensor.from_handle(base + 0x7000, (rows, layers, heads, capacity), DType.FP16, _CPU),
                v_scale=Tensor.from_handle(base + 0x8000, (rows, layers, heads, capacity), DType.FP16, _CPU),
            )
        spans = KVLiveSpans(
            base_offsets=Tensor.from_handle(base + 0x1000, (rows, layers, heads), DType.INT32, _CPU),
            live_counts=Tensor.from_handle(base + 0x2000, (rows, layers, heads), DType.INT32, _CPU),
            max_live_count=max(int(np.max(state.live_counts)) for state in states),
            token_positions=Tensor.from_handle(base + 0x3000, (rows, layers, heads, capacity), DType.INT32, _CPU),
            evict_mask=Tensor.from_handle(base + 0x4000, (rows, layers, heads, capacity), DType.BOOL, _CPU),
            storage_dtype=self.storage_dtype,
            spans_mode="per_head_variable",
            request_ids=Tensor.from_handle(base + 0x5000, (rows,), DType.INT64, _CPU),
            row_positions=Tensor.from_handle(base + 0x6000, (rows,), DType.INT32, _CPU),
            span_role=str(getattr(work_item, "span_role", "decode")),
            scale_metadata=scales,
        )
        return KVBatchView(
            live_spans=spans,
            storage_view=self._storage_view,
            kernel_bundle_key=self.spec.kernel_bundle_key,
            execution_compatibility_key=(*self.spec.compatibility_key, spans.span_role),
        )

    def begin_transaction(self, rows: Sequence[Any], draft: Any) -> DMSOperation:
        del draft
        if len(rows) != 1:
            raise ValueError("compact DMS transaction expects one row")
        lease = getattr(rows[0], "lease", rows[0])
        if not isinstance(lease, KVLease):
            raise TypeError("compact DMS transaction row must contain KVLease")
        state = self.state_for_request(lease.request_id)
        return DMSOperation(
            operation_id=f"dms-transaction:{lease.request_id}",
            lease=lease,
            state_snapshot=(
                state.live_counts.copy(),
                state.token_positions.copy(),
                state.evict_mask.copy(),
            ),
            payload_snapshot=(
                {key: value.copy() for key, value in state.k_payload.items()},
                {key: value.copy() for key, value in state.v_payload.items()},
                {key: value.copy() for key, value in state.k_scales.items()},
                {key: value.copy() for key, value in state.v_scales.items()},
            ),
            logical_tokens=int(state.logical_tokens),
            device_snapshot=(
                None
                if self._device_store is None
                else self._device_store.snapshot(
                    state.base_offsets,
                    state.range_capacity,
                )
            ),
            counter_snapshot=(
                int(self.pack_calls),
                int(self.decode_appends),
                int(self.evicted_tokens),
            ),
        )

    def commit(self, operation: Any, result: Any) -> ResourceDelta:
        del result
        if not isinstance(operation, DMSOperation):
            raise TypeError("compact DMS commit requires DMSOperation")
        return ResourceDelta(
            operation_id=f"commit:{operation.operation_id}",
            lease_id=operation.lease.lease_id,
            request_id=operation.lease.request_id,
        )

    def rollback(self, operation: Any) -> ResourceDelta:
        if not isinstance(operation, DMSOperation):
            raise TypeError("compact DMS rollback requires DMSOperation")
        state = self.state_for_request(operation.lease.request_id)
        state.live_counts[...] = operation.state_snapshot[0]
        state.token_positions[...] = operation.state_snapshot[1]
        state.evict_mask[...] = operation.state_snapshot[2]
        state.k_payload = {
            key: value.copy() for key, value in operation.payload_snapshot[0].items()
        }
        state.v_payload = {
            key: value.copy() for key, value in operation.payload_snapshot[1].items()
        }
        state.k_scales = {
            key: value.copy() for key, value in operation.payload_snapshot[2].items()
        }
        state.v_scales = {
            key: value.copy() for key, value in operation.payload_snapshot[3].items()
        }
        state.logical_tokens = int(operation.logical_tokens)
        if operation.device_snapshot is not None:
            if self._device_store is None:
                raise RuntimeError("DMS device store disappeared before rollback")
            self._device_store.restore(operation.device_snapshot)
        self.pack_calls, self.decode_appends, self.evicted_tokens = operation.counter_snapshot
        return ResourceDelta(
            operation_id=f"rollback:{operation.operation_id}",
            lease_id=operation.lease.lease_id,
            request_id=operation.lease.request_id,
        )

    def reclaim(self, lease: KVLease) -> ResourceDelta:
        state = self.state_for_request(lease.request_id)
        if state.lease != lease:
            raise ValueError("compact DMS lease identity mismatch")
        self.extents.release(lease.lease_id)
        delta = self.ledger.release(
            lease.lease_id,
            operation_id=f"dms-reclaim:{lease.request_id}",
        )
        self._states.pop(lease.request_id)
        self._leases.pop(lease.lease_id)
        return delta

    def prefix_lookup(self, tokens: Sequence[int]) -> Any:
        return SimpleNamespace(
            hit=False,
            matched_tokens=(),
            remaining_tokens=tuple(int(token) for token in tokens),
            reason="dms_prefix_overlay_unqualified",
        )

    def maintenance(self, budget: Any) -> list[Any]:
        del budget
        return []

    def state_for_request(self, request_id: int) -> DMSSequenceState:
        try:
            return self._states[int(request_id)]
        except KeyError as exc:
            raise KeyError(f"request_id {request_id} has no compact DMS state") from exc

    def lease_for_request(self, request_id: int) -> KVLease:
        return self.state_for_request(request_id).lease

    def has_request(self, request_id: int) -> bool:
        return int(request_id) in self._states

    def observability_snapshot(self) -> dict[str, Any]:
        logical = sum(
            state.logical_tokens
            * self.retrofit.num_layers
            * self.retrofit.num_kv_heads
            for state in self._states.values()
        )
        live = sum(int(np.sum(state.live_counts)) for state in self._states.values())
        payload_itemsize = 2 if self.codec == "bf16" else 1
        payload_bytes = live * self.retrofit.head_dim * payload_itemsize * 2
        scale_bytes = live * 4 * 2 if self.codec == "int8_per_token_head" else 0
        return {
            "backend": {
                "topology": "dms_compact",
                "codec": self.codec,
                "artifact_fingerprint": self.retrofit.artifact_fingerprint,
                "retrofit_fingerprint": self.retrofit.fingerprint,
                "decision_source": self.retrofit.decision_source,
                "physical_layer_ids": list(self.retrofit.physical_layer_ids),
                "prefix_mode": "off",
                "no_dense_shadow": True,
                "device_payloads": self.device_payloads_enabled,
                "physical_widths": list(self.spec.physical_widths),
            },
            "capacity": {
                "logical_token_rows": logical,
                "live_token_rows": live,
                "actual_compression_ratio": (
                    None if live == 0 else logical / live
                ),
                "target_compression_ratio": self.retrofit.target_compression_ratio,
                "max_live_count": max(
                    (int(np.max(state.live_counts)) for state in self._states.values()),
                    default=0,
                ),
                "payload_bytes": payload_bytes,
                "scale_bytes": scale_bytes,
            },
            "operations": {
                "streaming_pack_calls": self.pack_calls,
                "decode_appends": self.decode_appends,
                "evicted_tokens": self.evicted_tokens,
            },
            "extent_pool": self.extents.snapshot(),
            "ledger": self.ledger.snapshot(),
        }

    def assert_conserved(self) -> None:
        self.extents.assert_conserved()
        self.ledger.assert_conserved()
        if bool(self._states) != bool(self._leases):
            raise AssertionError("compact DMS state/lease ownership drift")

    def _build_plan(self) -> KVPoolPlan:
        total_slots = self.retrofit.num_layers * self.slots_per_layer
        lease_lifetime = (ClaimLifetime.LEASE,)
        pools = [
            KVPoolSpec(pool_id, total_slots, unit="token_slots", plane_role=pool_id.split(".", 1)[1], lifetimes=lease_lifetime)
            for pool_id in self.reservation_pool_ids
        ]
        pools.extend(
            (
                KVPoolSpec(
                    "dms.extent_descriptors",
                    self.max_request_rows * self.retrofit.num_layers * self.retrofit.num_kv_heads,
                    unit="descriptors",
                    plane_role="extent_metadata",
                    lifetimes=lease_lifetime,
                ),
                KVPoolSpec(
                    "dms.request_rows",
                    self.max_request_rows,
                    unit="rows",
                    plane_role="row_metadata",
                    lifetimes=lease_lifetime,
                ),
                KVPoolSpec(
                    "dms.pack_workspace_rows",
                    self.max_pack_rows,
                    unit="rows",
                    plane_role="streaming_workspace",
                    lifetimes=(ClaimLifetime.WORK_ITEM,),
                ),
            )
        )
        return KVPoolPlan(
            backend_fingerprint=self.spec.fingerprint,
            generation=self.generation,
            pools=tuple(pools),
        )

    def _build_storage_view(self) -> KVStorageView:
        base = 0x6C000000 + self.generation * 0x100000
        total_slots = self.retrofit.num_layers * self.slots_per_layer
        payload_dtype = "bf16" if self.codec == "bf16" else "int8"
        planes = [
            KVPlaneView("k_payload", payload_dtype, base + 0x1000, (total_slots, self.retrofit.head_dim), (self.retrofit.head_dim, 1)),
            KVPlaneView("v_payload", payload_dtype, base + 0x2000, (total_slots, self.retrofit.head_dim), (self.retrofit.head_dim, 1)),
            KVPlaneView("base_offsets", "int32", base + 0x3000, (self.max_request_rows, self.retrofit.num_layers, self.retrofit.num_kv_heads), (self.retrofit.num_layers * self.retrofit.num_kv_heads, self.retrofit.num_kv_heads, 1)),
            KVPlaneView("range_capacity", "int32", base + 0x4000, (self.max_request_rows, self.retrofit.num_layers, self.retrofit.num_kv_heads), (self.retrofit.num_layers * self.retrofit.num_kv_heads, self.retrofit.num_kv_heads, 1)),
            KVPlaneView("live_counts", "int32", base + 0x5000, (self.max_request_rows, self.retrofit.num_layers, self.retrofit.num_kv_heads), (self.retrofit.num_layers * self.retrofit.num_kv_heads, self.retrofit.num_kv_heads, 1)),
            KVPlaneView("token_positions", "int32", base + 0x6000, (total_slots,), (1,)),
            KVPlaneView("evict_mask", "bool", base + 0x7000, (total_slots,), (1,)),
        ]
        if self.codec == "int8_per_token_head":
            planes.extend(
                (
                    KVPlaneView("k_scale", "fp32", base + 0x8000, (total_slots,), (1,)),
                    KVPlaneView("v_scale", "fp32", base + 0x9000, (total_slots,), (1,)),
                )
            )
        return KVStorageView(
            layout_key=f"dms-compact:{self.codec}:g{self.generation}",
            generation=self.generation,
            planes=tuple(planes),
            artifact_fingerprint=self.retrofit.artifact_fingerprint,
            metadata_descriptor_ptr=base + 0xA000,
            metadata_descriptor_bytes=256,
        )


class DMSAdmissionManager:
    """Fit-aware scheduler callbacks for compact extents; prefix stays off."""

    def __init__(
        self,
        backend: DMSCompactBackend,
        *,
        lookahead: int = 32,
        max_bypasses: int = 8,
    ) -> None:
        self.backend = backend
        self.controller = FitAwareAdmissionController(
            backend.ledger,
            lookahead=lookahead,
            max_bypasses=max_bypasses,
        )

    def plan_admission(self, pending_requests: Sequence[Any], *, max_items: int) -> tuple[int, ...]:
        pending_by_id = {int(request.request_id): request for request in pending_requests}
        for request_id in self.controller.pending_request_ids:
            if request_id not in pending_by_id:
                self.controller.cancel(request_id)
        known = set(self.controller.pending_request_ids)
        known.update(
            request_id for request_id in pending_by_id if self.backend.has_request(request_id)
        )
        for request_id, request in pending_by_id.items():
            if request_id in known:
                continue
            self.controller.enqueue(
                request_id,
                self.backend.estimate(request, None, {"kind": "admission"}),
                owner_id=f"dms-lease:{request_id}",
            )
        grants = self.controller.admit(max_items=max_items)
        admitted: list[int] = []
        for grant in grants:
            try:
                # Ledger ownership is already committed by the controller. Give
                # the backend an equivalent atomic path by releasing then using
                # its reserve operation, preserving one implementation of extents.
                self.backend.ledger.release(
                    grant.owner_id,
                    operation_id=f"dms-controller-rematerialize:{grant.request_id}",
                )
                self.backend.reserve(grant.reservation.claims)
            except Exception:
                if self.backend.has_request(grant.request_id):
                    self.backend.reclaim(self.backend.lease_for_request(grant.request_id))
                raise
            admitted.append(grant.request_id)
        return tuple(admitted)

    def reserve_admission(self, request: Any) -> None:
        if not self.backend.has_request(int(request.request_id)):
            raise RuntimeError("compact DMS admission lacks materialized extent lease")

    def rollback_admission(self, request: Any) -> None:
        request_id = int(request.request_id)
        if self.backend.has_request(request_id):
            self.backend.reclaim(self.backend.lease_for_request(request_id))
        else:
            self.controller.cancel(request_id)

    def reclaim_request(self, request: Any) -> ResourceDelta | None:
        request_id = int(request.request_id)
        if not self.backend.has_request(request_id):
            self.controller.cancel(request_id)
            return None
        return self.backend.reclaim(self.backend.lease_for_request(request_id))

    def resource_observability_snapshot(self) -> dict[str, Any]:
        snapshot = self.backend.observability_snapshot()
        snapshot["admission"] = self.controller.snapshot()
        return snapshot


class DMSCompactResidentRunnerAdapter:
    """Bind a common resident runner to compact DMS KVBatchView inputs."""

    supports_prefill_decode_same_round = True
    supports_multiple_prefill_quanta_per_round = True

    def __init__(self, runner: Any, admission: DMSAdmissionManager) -> None:
        for name in ("prefill_batch_with_kv", "decode_batch_with_kv"):
            if not callable(getattr(runner, name, None)):
                raise TypeError(f"compact DMS runner requires {name}")
        if str(getattr(runner, "kv_kernel_bundle_key", "")) != admission.backend.spec.kernel_bundle_key:
            raise ValueError("compact DMS runner kernel bundle mismatch")
        self.runner = runner
        self.admission = admission
        self.backend = admission.backend
        self.capacity = int(getattr(runner, "capacity"))

    def plan_admission(self, pending_requests: Sequence[Any], *, max_items: int) -> tuple[int, ...]:
        return self.admission.plan_admission(pending_requests, max_items=max_items)

    def reserve_admission(self, request: Any) -> None:
        self.admission.reserve_admission(request)

    def rollback_admission(self, request: Any) -> None:
        self.admission.rollback_admission(request)

    def prefill_batch(self, work: Any, *, commit: bool) -> Any:
        view = self.backend.prepare(work)
        return self.runner.prefill_batch_with_kv(work, kv_batch_view=view, commit=commit)

    def decode_batch(self, work: Any, *, commit: bool) -> Any:
        view = self.backend.prepare(work)
        return self.runner.decode_batch_with_kv(work, kv_batch_view=view, commit=commit)

    def compact_batch(self, moves: Any) -> Any:
        compact = getattr(self.runner, "compact_batch", None)
        return None if not callable(compact) else compact(moves)

    def reclaim(self, completed: Any) -> None:
        reclaim = getattr(self.runner, "reclaim", None)
        if callable(reclaim):
            reclaim(completed)
        self.admission.reclaim_request(completed)

    def resource_observability_snapshot(self) -> dict[str, Any]:
        return self.admission.resource_observability_snapshot()


def create_dms_bf16_backend(**kwargs: Any) -> DMSCompactBackend:
    return DMSCompactBackend(codec="bf16", **kwargs)


def create_dms_int8_backend(
    *,
    codec_qualification: DMSCodecQualification,
    **kwargs: Any,
) -> DMSCompactBackend:
    return DMSCompactBackend(
        codec="int8_per_token_head",
        codec_qualification=codec_qualification,
        **kwargs,
    )


__all__ = [
    "CompactExtent",
    "CompactExtentPool",
    "DMSAdmissionManager",
    "DMSCodecQualification",
    "DMSCompactBackend",
    "DMSCompactResidentRunnerAdapter",
    "DMSLinearSidecarSpec",
    "DMSOperation",
    "DMSRetrofitConfig",
    "DMSSequenceState",
    "DMSTrainingProvenance",
    "build_dms_live_mask",
    "compact_attention_reference",
    "create_dms_bf16_backend",
    "create_dms_int8_backend",
    "decode_dms_payload",
    "encode_dms_payload",
    "extract_dms_eviction_decisions",
    "load_dms_retrofit_config",
]
