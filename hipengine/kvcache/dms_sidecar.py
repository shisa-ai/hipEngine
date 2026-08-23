"""Torch-free external DMS sidecar loading, projection, and capture replay."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.kernels.registry import resolve
from hipengine.kvcache.dms import DMSRetrofitConfig
from hipengine.kvcache.dms_labels import build_eviction_labels, load_dms_label_manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(bits, dtype=np.uint16)
    return (values.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _read_sidecar_tensors(config: DMSRetrofitConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.schema_version != 2 or config.sidecar is None:
        raise ValueError("external DMS projection requires schema-v2 sidecar metadata")
    path = Path(config.sidecar.resolved_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256_file(path) != config.sidecar.sha256:
        raise ValueError("external DMS sidecar hash changed after metadata validation")
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError("external DMS sidecar safetensors file is truncated")
    header_size = int.from_bytes(raw[:8], "little", signed=False)
    if header_size <= 0 or 8 + header_size > len(raw):
        raise ValueError("external DMS sidecar safetensors header is invalid")
    header = json.loads(raw[8 : 8 + header_size].decode("utf-8"))
    data_offset = 8 + header_size

    def tensor(name: str, shape: tuple[int, ...]) -> np.ndarray:
        descriptor = header.get(name)
        if not isinstance(descriptor, dict):
            raise TypeError(f"external DMS sidecar tensor {name!r} is missing")
        if descriptor.get("dtype") != "BF16" or tuple(descriptor.get("shape", ())) != shape:
            raise ValueError(f"external DMS sidecar tensor {name!r} dtype/shape mismatch")
        offsets = descriptor.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"external DMS sidecar tensor {name!r} offsets are invalid")
        start, end = int(offsets[0]), int(offsets[1])
        payload = raw[data_offset + start : data_offset + end]
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * 2
        if len(payload) != expected_bytes:
            raise ValueError(f"external DMS sidecar tensor {name!r} byte size mismatch")
        bits = np.frombuffer(payload, dtype=np.uint16).copy().reshape(shape)
        return np.ascontiguousarray(_bf16_bits_to_float32(bits), dtype=np.float32)

    return (
        tensor(config.sidecar.weight_tensor, config.sidecar.weight_shape),
        tensor(config.sidecar.bias_tensor, config.sidecar.bias_shape),
    )


@dataclass(frozen=True, slots=True)
class ExternalDMSLinearSidecar:
    config: DMSRetrofitConfig
    weight: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        if self.config.schema_version != 2 or self.config.hidden_size is None:
            raise ValueError("external DMS source requires schema-v2 metadata")
        expected_weight = (
            self.config.num_layers,
            self.config.num_kv_heads,
            self.config.hidden_size,
        )
        expected_bias = (self.config.num_layers, self.config.num_kv_heads)
        weight = np.ascontiguousarray(self.weight, dtype=np.float32)
        bias = np.ascontiguousarray(self.bias, dtype=np.float32)
        if weight.shape != expected_weight or bias.shape != expected_bias:
            raise ValueError("external DMS sidecar tensor geometry mismatch")
        if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
            raise ValueError("external DMS sidecar tensors must be finite")
        weight.setflags(write=False)
        bias.setflags(write=False)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)

    def compact_layer_index(self, physical_layer_id: int) -> int:
        try:
            return self.config.physical_layer_ids.index(int(physical_layer_id))
        except ValueError as exc:
            raise ValueError(
                f"physical layer {physical_layer_id} is not mapped by the DMS sidecar"
            ) from exc

    def project(
        self,
        hidden: np.ndarray,
        *,
        physical_layer_id: int | None = None,
        compact_layer_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (physical_layer_id is None) == (compact_layer_index is None):
            raise ValueError("provide exactly one physical or compact DMS layer index")
        layer = (
            self.compact_layer_index(int(physical_layer_id))
            if compact_layer_index is None
            else int(compact_layer_index)
        )
        if layer < 0 or layer >= self.config.num_layers:
            raise ValueError("compact DMS layer index is out of range")
        values = np.asarray(hidden)
        if values.dtype == np.uint16:
            values = _bf16_bits_to_float32(values)
        else:
            values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != int(self.config.hidden_size):
            raise ValueError("external DMS hidden input shape mismatch")
        logits = np.ascontiguousarray(
            values @ self.weight[layer].T + self.bias[layer],
            dtype=np.float32,
        )
        decisions = logits * self.config.alpha_scale - self.config.alpha_offset > 0.0
        return logits, np.asarray(decisions, dtype=np.bool_)


def load_external_dms_sidecar(config: DMSRetrofitConfig) -> ExternalDMSLinearSidecar:
    weight, bias = _read_sidecar_tensors(config)
    return ExternalDMSLinearSidecar(config=config, weight=weight, bias=bias)


class DMSExternalDecisionRuntime:
    """Registered torch-free decision source feeding compact backend commits."""

    def __init__(self, source: ExternalDMSLinearSidecar) -> None:
        self.source = source
        kernel = resolve(
            backend="cpu_reference",
            layer="dms_decision_source",
            quant="bf16",
            variant="external_linear_sidecar_v1",
            missing="none",
        )
        if kernel is None:
            from hipengine.kernels.cpu_reference.dms import (
                register_dms_cpu_reference_kernels,
            )

            register_dms_cpu_reference_kernels(replace=True)
            kernel = resolve(
                backend="cpu_reference",
                layer="dms_decision_source",
                quant="bf16",
                variant="external_linear_sidecar_v1",
            )
        self._kernel = kernel

    def _require_backend(self, backend: Any) -> None:
        retrofit = getattr(backend, "retrofit", None)
        if not isinstance(retrofit, DMSRetrofitConfig):
            raise TypeError("external DMS runtime requires a compact DMS backend")
        if retrofit.fingerprint != self.source.config.fingerprint:
            raise ValueError("external DMS runtime/backend metadata fingerprints differ")
        if bool(getattr(backend, "device_payloads_enabled", False)):
            raise ValueError(
                "external DMS device payload mutation requires a qualified device journal; "
                "use the strict host fallback"
            )

    @staticmethod
    def _require_span_role(span_role: str, expected: str) -> None:
        role = str(span_role)
        if role != expected:
            raise ValueError(
                f"external DMS decision source has unsupported span role {role!r}; "
                "prefix/speculative modes fail closed"
            )

    def prefill_decisions(self, hidden: np.ndarray) -> np.ndarray:
        values = np.asarray(hidden)
        hidden_size = int(self.source.config.hidden_size)
        expected = (self.source.config.num_layers, hidden_size)
        if values.ndim != 3 or values.shape[1:] != expected:
            raise ValueError(
                "external DMS prefill hidden must be [tokens,compact_layers,hidden]"
            )
        decisions = np.zeros(
            (values.shape[0], self.source.config.num_layers, self.source.config.num_kv_heads),
            dtype=np.bool_,
        )
        for layer in range(self.source.config.num_layers):
            layer_values = values[:, layer]
            if layer_values.dtype == np.uint16:
                layer_values = _bf16_bits_to_float32(layer_values)
            logits, layer_decisions = self._kernel(
                np.asarray(layer_values, dtype=np.float32),
                self.source.weight[layer],
                self.source.bias[layer],
                alpha_scale=self.source.config.alpha_scale,
                alpha_offset=self.source.config.alpha_offset,
            )
            if logits.shape != layer_decisions.shape or logits.shape != (
                values.shape[0],
                self.source.config.num_kv_heads,
            ):
                raise RuntimeError("registered external DMS decision kernel returned wrong shape")
            decisions[:, layer, :] = layer_decisions
        return decisions

    def decode_decisions(self, hidden: np.ndarray) -> np.ndarray:
        values = np.asarray(hidden)
        expected = (
            self.source.config.num_layers,
            int(self.source.config.hidden_size),
        )
        if values.shape != expected:
            raise ValueError("external DMS decode hidden must be [compact_layers,hidden]")
        return self.prefill_decisions(values[None, ...])[0]

    def streaming_pack(
        self,
        backend: Any,
        *,
        request_id: int,
        hidden: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        span_role: str = "prefill",
    ) -> np.ndarray:
        self._require_span_role(span_role, "prefill")
        self._require_backend(backend)
        decisions = self.prefill_decisions(hidden)
        lease = backend.lease_for_request(int(request_id))
        operation = backend.begin_transaction((lease,), None)
        try:
            backend.streaming_pack(int(request_id), k, v, decisions)
        except Exception:
            backend.rollback(operation)
            raise
        backend.commit(operation, None)
        return decisions

    def append_decode(
        self,
        backend: Any,
        *,
        request_id: int,
        hidden: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        position: int,
        span_role: str = "decode",
    ) -> np.ndarray:
        self._require_span_role(span_role, "decode")
        self._require_backend(backend)
        decisions = self.decode_decisions(hidden)
        lease = backend.lease_for_request(int(request_id))
        operation = backend.begin_transaction((lease,), None)
        try:
            backend.append_decode(
                int(request_id),
                k,
                v,
                decisions,
                position=int(position),
            )
        except Exception:
            backend.rollback(operation)
            raise
        backend.commit(operation, None)
        return decisions


class ExternalDMSDecisionCollector:
    """Exact-stage GGUF diagnostic sink producing per-token compact decisions."""

    def __init__(self, source: ExternalDMSLinearSidecar, *, token_count: int) -> None:
        self.source = source
        self.physical_layer_ids = source.config.physical_layer_ids
        self.hidden_size = int(source.config.hidden_size)
        self.num_q_heads = source.config.num_q_heads
        self.num_kv_heads = source.config.num_kv_heads
        self.head_dim = source.config.head_dim
        self.input_stage = str(source.config.input_stage)
        self.token_count = int(token_count)
        if self.token_count <= 0:
            raise ValueError("external DMS collector token_count must be positive")
        self._next = np.zeros(source.config.num_layers, dtype=np.int32)
        self._decisions = np.zeros(
            (self.token_count, source.config.num_layers, source.config.num_kv_heads),
            dtype=np.bool_,
        )
        self.teacher_vocab_size: int | None = None

    def capture_chunk(
        self,
        *,
        physical_layer_id: int,
        compact_layer_index: int,
        positions: np.ndarray,
        hidden_bf16: np.ndarray,
        query_f32: np.ndarray,
        key_f32: np.ndarray,
    ) -> None:
        del query_f32, key_f32
        layer = int(compact_layer_index)
        if self.source.compact_layer_index(int(physical_layer_id)) != layer:
            raise ValueError("external DMS collector physical/compact layer mismatch")
        pos = np.asarray(positions, dtype=np.int32)
        start = int(self._next[layer])
        if not np.array_equal(pos, np.arange(start, start + pos.size, dtype=np.int32)):
            raise ValueError("external DMS collector chunks must be contiguous")
        if start + pos.size > self.token_count:
            raise ValueError("external DMS collector chunk exceeds token_count")
        _, decisions = self.source.project(
            hidden_bf16,
            compact_layer_index=layer,
        )
        if decisions.shape != (pos.size, self.num_kv_heads):
            raise RuntimeError("external DMS collector decision shape mismatch")
        self._decisions[start : start + pos.size, layer, :] = decisions
        self._next[layer] = start + pos.size

    def capture_teacher_logits(self, logits: np.ndarray) -> None:
        values = np.asarray(logits)
        if values.size <= 0:
            raise ValueError("external DMS collector teacher logits must be non-empty")
        self.teacher_vocab_size = int(values.size)

    def finalize(self) -> np.ndarray:
        if np.any(self._next != self.token_count):
            raise ValueError("external DMS collector lacks full layer/token coverage")
        if self.teacher_vocab_size is None:
            raise ValueError("external DMS collector lacks teacher logits")
        return self._decisions.copy()


@dataclass(frozen=True, slots=True)
class _ReplayRecord:
    split: str
    category: str
    context_bucket: str
    compact_layer_index: int
    physical_layer_id: int
    positions: np.ndarray
    hidden_bf16: np.ndarray
    eligible: np.ndarray
    future_mass: np.ndarray


def _context_bucket(tokens: int) -> str:
    if tokens <= 512:
        return "le512"
    if tokens <= 2048:
        return "513_2048"
    if tokens <= 8192:
        return "2049_8192"
    return "gt8192"


def _replay_records(path: Path, manifest: dict[str, Any], source: ExternalDMSLinearSidecar) -> list[_ReplayRecord]:
    root = path.parent
    records: list[_ReplayRecord] = []
    for sequence in manifest["sequences"]:
        provenance = sequence.get("provenance")
        if not isinstance(provenance, dict):
            raise TypeError("DMS replay sequence provenance must be an object")
        split = str(provenance.get("split", ""))
        if split not in {"train", "validation"}:
            raise ValueError("DMS replay requires train/validation splits")
        tokens = int(sequence["token_count"])
        for shard_record in sequence["shards"]:
            shard_path = (root / str(shard_record["path"])).resolve()
            with np.load(shard_path, allow_pickle=False) as shard:
                required = {
                    "positions",
                    "hidden_bf16",
                    "eligible_mask",
                    "future_attention_mass",
                }
                missing = sorted(required - set(shard.files))
                if missing:
                    raise ValueError(f"DMS replay label shard is missing {missing}")
                positions = np.asarray(shard["positions"])
                hidden = np.asarray(shard["hidden_bf16"])
                eligible = np.asarray(shard["eligible_mask"])
                future_mass = np.asarray(shard["future_attention_mass"])
            compact_index = int(shard_record["compact_layer_index"])
            physical_layer = int(shard_record["physical_layer_id"])
            if source.compact_layer_index(physical_layer) != compact_index:
                raise ValueError("DMS replay physical/compact layer mapping mismatch")
            if positions.shape != (tokens,) or positions.dtype != np.int32:
                raise ValueError("DMS replay positions shape/dtype mismatch")
            if hidden.shape != (tokens, int(source.config.hidden_size)) or hidden.dtype != np.uint16:
                raise ValueError("DMS replay hidden shape/dtype mismatch")
            if eligible.shape != (tokens,) or eligible.dtype != np.bool_:
                raise ValueError("DMS replay eligibility shape/dtype mismatch")
            if future_mass.shape != (tokens, source.config.num_kv_heads):
                raise ValueError("DMS replay future-mass shape mismatch")
            records.append(
                _ReplayRecord(
                    split=split,
                    category=str(sequence["category"]),
                    context_bucket=_context_bucket(tokens),
                    compact_layer_index=compact_index,
                    physical_layer_id=physical_layer,
                    positions=np.ascontiguousarray(positions),
                    hidden_bf16=np.ascontiguousarray(hidden),
                    eligible=np.ascontiguousarray(eligible),
                    future_mass=np.ascontiguousarray(future_mass, dtype=np.float32),
                )
            )
    if not records:
        raise ValueError("DMS replay has no label records")
    return records


def _desired_evictions(record: _ReplayRecord, *, compression_ratio: int, window_size: int) -> np.ndarray:
    labels, eligible, _ = build_eviction_labels(
        record.future_mass,
        positions=record.positions,
        current_position=int(record.positions[-1]),
        window_size=window_size,
        target_compression_ratio=compression_ratio,
    )
    if not np.array_equal(eligible, record.eligible):
        raise ValueError("DMS replay eligibility differs from the configured window")
    return labels


def _threshold_for_count(values: np.ndarray, desired_count: int) -> float:
    scores = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(scores)):
        raise ValueError("DMS calibration logits must be finite")
    desired = int(desired_count)
    if desired <= 0:
        return float("inf")
    if desired >= scores.size:
        return float(np.nextafter(np.min(scores), -np.inf))
    ordered = np.sort(scores)[::-1]
    upper = float(ordered[desired - 1])
    lower = float(ordered[desired])
    if upper > lower:
        return (upper + lower) * 0.5
    return float(np.nextafter(upper, -np.inf))


class _ScreenMetric:
    def __init__(self) -> None:
        self.cells = 0
        self.evictions = 0
        self.protected_violations = 0
        self.suppressed_protected = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(
        self,
        *,
        decision: np.ndarray,
        raw_decision: np.ndarray,
        eligible: np.ndarray,
        oracle: np.ndarray,
    ) -> None:
        active = np.asarray(decision, dtype=np.bool_)
        raw = np.asarray(raw_decision, dtype=np.bool_)
        mask = np.asarray(eligible, dtype=np.bool_)[:, None]
        target = np.asarray(oracle, dtype=np.bool_)
        self.cells += int(active.size)
        self.evictions += int(np.count_nonzero(active))
        self.protected_violations += int(np.count_nonzero(active & ~mask))
        self.suppressed_protected += int(np.count_nonzero(raw & ~mask))
        self.tp += int(np.count_nonzero(active & target))
        self.fp += int(np.count_nonzero(active & ~target))
        self.fn += int(np.count_nonzero(~active & target))

    def result(self) -> dict[str, int | float]:
        live = self.cells - self.evictions
        return {
            "cells": self.cells,
            "evictions": self.evictions,
            "live_cells": live,
            "actual_compression_ratio": self.cells / max(1, live),
            "precision": self.tp / max(1, self.tp + self.fp),
            "recall": self.tp / max(1, self.tp + self.fn),
            "protected_window_violations": self.protected_violations,
            "suppressed_protected_decisions": self.suppressed_protected,
        }


def _evaluate_scenarios(
    records: list[_ReplayRecord],
    source: ExternalDMSLinearSidecar,
    *,
    thresholds: dict[str, float | None],
    compression_ratios: tuple[int, ...],
    window_size: int,
) -> dict[str, Any]:
    scenario_cr = {f"cr{cr}": int(cr) for cr in compression_ratios}
    scenario_cr["no_evict"] = 1
    metrics: dict[str, dict[str, Any]] = {}
    for scenario in ("no_evict", *(f"cr{cr}" for cr in compression_ratios)):
        global_metric = _ScreenMetric()
        by_layer_head: dict[str, _ScreenMetric] = {}
        by_category: dict[str, _ScreenMetric] = {}
        by_context: dict[str, _ScreenMetric] = {}
        for record in records:
            logits, _ = source.project(
                record.hidden_bf16,
                compact_layer_index=record.compact_layer_index,
            )
            threshold = thresholds[scenario]
            raw = (
                np.zeros_like(logits, dtype=np.bool_)
                if threshold is None
                else logits > float(threshold)
            )
            decision = raw & record.eligible[:, None]
            oracle = (
                np.zeros_like(decision)
                if scenario == "no_evict"
                else _desired_evictions(
                    record,
                    compression_ratio=scenario_cr[scenario],
                    window_size=window_size,
                )
            )
            global_metric.update(
                decision=decision,
                raw_decision=raw,
                eligible=record.eligible,
                oracle=oracle,
            )
            by_category.setdefault(record.category, _ScreenMetric()).update(
                decision=decision,
                raw_decision=raw,
                eligible=record.eligible,
                oracle=oracle,
            )
            by_context.setdefault(record.context_bucket, _ScreenMetric()).update(
                decision=decision,
                raw_decision=raw,
                eligible=record.eligible,
                oracle=oracle,
            )
            for head in range(source.config.num_kv_heads):
                key = f"layer{record.compact_layer_index}:head{head}"
                by_layer_head.setdefault(key, _ScreenMetric()).update(
                    decision=decision[:, head : head + 1],
                    raw_decision=raw[:, head : head + 1],
                    eligible=record.eligible,
                    oracle=oracle[:, head : head + 1],
                )
        metrics[scenario] = {
            "threshold": thresholds[scenario],
            "target_compression_ratio": scenario_cr[scenario],
            "global": global_metric.result(),
            "by_layer_head": {
                key: value.result() for key, value in sorted(by_layer_head.items())
            },
            "by_category": {
                key: value.result() for key, value in sorted(by_category.items())
            },
            "by_context_bucket": {
                key: value.result() for key, value in sorted(by_context.items())
            },
        }
    return metrics


def screen_external_sidecar(
    label_manifest_path: str | Path,
    source: ExternalDMSLinearSidecar,
    *,
    compression_ratios: tuple[int, ...] = (2, 4, 8),
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    label_path = Path(label_manifest_path).expanduser().resolve()
    manifest = load_dms_label_manifest(label_path, verify_shards=True)
    geometry = manifest["geometry"]
    if tuple(int(layer) for layer in geometry["physical_layer_ids"]) != source.config.physical_layer_ids:
        raise ValueError("DMS replay label/sidecar physical layer maps differ")
    if (
        int(geometry["hidden_size"]) != int(source.config.hidden_size)
        or int(geometry["num_kv_heads"]) != source.config.num_kv_heads
        or str(geometry["input_stage"]) != str(source.config.input_stage)
    ):
        raise ValueError("DMS replay label/sidecar geometry differs")
    ratios = tuple(int(cr) for cr in compression_ratios)
    if not ratios or len(set(ratios)) != len(ratios) or any(cr <= 1 for cr in ratios):
        raise ValueError("DMS replay compression ratios must be unique and greater than one")
    window = int(source.config.window_size)
    records = _replay_records(label_path, manifest, source)
    train = [record for record in records if record.split == "train"]
    validation = [record for record in records if record.split == "validation"]
    if not train or not validation:
        raise ValueError("DMS replay requires both train and validation records")
    thresholds: dict[str, float | None] = {"no_evict": None}
    calibration: dict[str, Any] = {"splits": ["train"], "scenarios": {}}
    for cr in ratios:
        values: list[np.ndarray] = []
        desired = 0
        for record in train:
            logits, _ = source.project(
                record.hidden_bf16,
                compact_layer_index=record.compact_layer_index,
            )
            values.append(logits[record.eligible])
            desired += int(
                np.count_nonzero(
                    _desired_evictions(
                        record,
                        compression_ratio=cr,
                        window_size=window,
                    )
                )
            )
        joined = np.concatenate([value.reshape(-1) for value in values])
        threshold = _threshold_for_count(joined, desired)
        key = f"cr{cr}"
        thresholds[key] = threshold
        calibration["scenarios"][key] = {
            "threshold": threshold,
            "eligible_decisions": int(joined.size),
            "target_evictions": desired,
            "calibrated_evictions": int(np.count_nonzero(joined > threshold)),
        }
    first = _evaluate_scenarios(
        validation,
        source,
        thresholds=thresholds,
        compression_ratios=ratios,
        window_size=window,
    )
    second = _evaluate_scenarios(
        validation,
        source,
        thresholds=thresholds,
        compression_ratios=ratios,
        window_size=window,
    )
    if first != second:
        raise RuntimeError("DMS sidecar replay is not deterministic")
    result = {
        "schema_version": 1,
        "kind": "hipengine_dms_sidecar_replay",
        "sidecar_fingerprint": source.config.fingerprint,
        "sidecar_sha256": source.config.sidecar.sha256 if source.config.sidecar else None,
        "label_manifest": {"path": str(label_path), "sha256": _sha256_file(label_path)},
        "calibration": calibration,
        "evaluation": {"splits": ["validation"]},
        "scenarios": first,
        "quality": {
            "dense_vs_masked_logits": "unavailable_without_runtime_replay",
        },
        "deterministic": True,
    }
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
        destination.with_suffix(destination.suffix + ".sha256").write_text(
            hashlib.sha256(payload).hexdigest() + "\n",
            encoding="ascii",
        )
    return result


__all__ = [
    "DMSExternalDecisionRuntime",
    "ExternalDMSDecisionCollector",
    "ExternalDMSLinearSidecar",
    "load_external_dms_sidecar",
    "screen_external_sidecar",
]
