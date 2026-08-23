#!/usr/bin/env python3
"""Train only an external linear DMS sidecar from checksummed label shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file as save_safetensors

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.kvcache.dms_labels import load_dms_label_manifest

_FASTDMS_REFERENCE_COMMIT = "c602b0ec3266da7f74d6a658b3dafcddb443fddd"
_DECISION_SOURCE = "external_linear_sidecar_v1"
_INPUT_STAGE = "post_attn_rmsnorm_pre_q_projection"


@dataclass(frozen=True, slots=True)
class _ShardRecord:
    path: Path
    sequence_id: str
    split: str
    category: str
    context_bucket: str
    physical_layer_id: int
    compact_layer_index: int
    rows: int


class _ExternalLinearSidecar(torch.nn.Module):
    def __init__(self, *, num_layers: int, num_kv_heads: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros((num_layers, num_kv_heads, hidden_size), dtype=torch.float32)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros((num_layers, num_kv_heads), dtype=torch.float32)
        )

    def forward(self, hidden: torch.Tensor, compact_layer_index: int) -> torch.Tensor:
        layer = int(compact_layer_index)
        return hidden @ self.weight[layer].T + self.bias[layer]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(bits, dtype=np.uint16)
    return (values.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _context_bucket(tokens: int) -> str:
    if tokens <= 512:
        return "le512"
    if tokens <= 2048:
        return "513_2048"
    if tokens <= 8192:
        return "2049_8192"
    return "gt8192"


def _records(label_manifest: Path, manifest: dict[str, Any]) -> tuple[list[_ShardRecord], list[_ShardRecord]]:
    root = label_manifest.parent
    train: list[_ShardRecord] = []
    validation: list[_ShardRecord] = []
    for sequence in manifest["sequences"]:
        provenance = sequence.get("provenance")
        if not isinstance(provenance, dict):
            raise TypeError("DMS label sequence provenance must be an object")
        split = str(provenance.get("split", ""))
        if split not in {"train", "validation"}:
            raise ValueError("DMS sidecar training requires train/validation splits")
        rows = int(sequence["token_count"])
        target = train if split == "train" else validation
        for shard in sequence["shards"]:
            target.append(
                _ShardRecord(
                    path=(root / str(shard["path"])).resolve(),
                    sequence_id=str(sequence["sequence_id"]),
                    split=split,
                    category=str(sequence["category"]),
                    context_bucket=_context_bucket(rows),
                    physical_layer_id=int(shard["physical_layer_id"]),
                    compact_layer_index=int(shard["compact_layer_index"]),
                    rows=int(shard["rows"]),
                )
            )
    if not train:
        raise ValueError("DMS sidecar training requires at least one train shard")
    if not validation:
        raise ValueError("DMS sidecar training requires at least one validation shard")
    return train, validation


def _load_shard(record: _ShardRecord, *, hidden_size: int, num_kv_heads: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"hidden_bf16", "eligible_mask", "evict_labels"}
    with np.load(record.path, allow_pickle=False) as shard:
        missing = sorted(required - set(shard.files))
        if missing:
            raise ValueError(f"DMS label shard is missing required arrays: {missing}")
        hidden_bits = np.asarray(shard["hidden_bf16"])
        eligible = np.asarray(shard["eligible_mask"])
        labels = np.asarray(shard["evict_labels"])
    if hidden_bits.dtype != np.uint16 or hidden_bits.shape != (record.rows, hidden_size):
        raise ValueError("DMS label hidden tensor shape/dtype mismatch")
    if eligible.dtype != np.bool_ or eligible.shape != (record.rows,):
        raise ValueError("DMS label eligibility tensor shape/dtype mismatch")
    if labels.dtype != np.bool_ or labels.shape != (record.rows, num_kv_heads):
        raise ValueError("DMS label target tensor shape/dtype mismatch")
    if np.any(labels[~eligible]):
        raise ValueError("DMS label shard contains a protected-window eviction")
    return _bf16_bits_to_float32(hidden_bits), eligible, labels


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _optimizer_parameter_count(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        int(parameter.numel())
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def _optimizer_state_elements(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        int(value.numel())
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def _train_epoch(
    model: _ExternalLinearSidecar,
    optimizer: torch.optim.Optimizer,
    records: list[_ShardRecord],
    *,
    epoch: int,
    seed: int,
    batch_size: int,
    budget_weight: float,
    max_grad_norm: float,
    hidden_size: int,
    num_kv_heads: int,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    order_rng = np.random.default_rng(int(seed) + int(epoch) * 1_000_003)
    record_order = order_rng.permutation(len(records)).tolist()
    total_loss = 0.0
    total_bce = 0.0
    total_budget = 0.0
    total_examples = 0
    optimizer_steps = 0
    for record_index in record_order:
        record = records[int(record_index)]
        hidden, eligible, labels = _load_shard(
            record,
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
        )
        eligible_rows = np.flatnonzero(eligible)
        token_rng = np.random.default_rng(
            int(seed) + int(epoch) * 10_000_019 + int(record_index) * 97
        )
        token_rng.shuffle(eligible_rows)
        for start in range(0, int(eligible_rows.size), int(batch_size)):
            indices = eligible_rows[start : start + int(batch_size)]
            if indices.size == 0:
                continue
            x = torch.as_tensor(hidden[indices], dtype=torch.float32, device=device)
            target = torch.as_tensor(
                labels[indices],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, record.compact_layer_index)
            bce = F.binary_cross_entropy_with_logits(logits, target)
            predicted_rate = torch.sigmoid(logits).mean(dim=0)
            target_rate = target.mean(dim=0)
            budget = torch.square(predicted_rate - target_rate).mean()
            loss = bce + float(budget_weight) * budget
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step()
            count = int(indices.size) * int(num_kv_heads)
            total_loss += float(loss.detach().cpu()) * count
            total_bce += float(bce.detach().cpu()) * count
            total_budget += float(budget.detach().cpu()) * count
            total_examples += count
            optimizer_steps += 1
    if total_examples <= 0:
        raise ValueError("DMS training found no eligible label rows")
    return {
        "train_loss": total_loss / total_examples,
        "train_bce": total_bce / total_examples,
        "train_budget": total_budget / total_examples,
        "optimizer_steps": float(optimizer_steps),
    }


class _Metric:
    def __init__(self) -> None:
        self.count = 0
        self.loss_sum = 0.0
        self.correct = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.predicted_positive = 0
        self.label_positive = 0

    def update(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        *,
        decision_threshold: float,
    ) -> None:
        values = np.asarray(logits, dtype=np.float64).reshape(-1)
        target = np.asarray(labels, dtype=np.bool_).reshape(-1)
        prediction = values > float(decision_threshold)
        losses = np.maximum(values, 0.0) - values * target.astype(np.float64) + np.log1p(
            np.exp(-np.abs(values))
        )
        self.count += int(values.size)
        self.loss_sum += float(np.sum(losses))
        self.correct += int(np.count_nonzero(prediction == target))
        self.tp += int(np.count_nonzero(prediction & target))
        self.fp += int(np.count_nonzero(prediction & ~target))
        self.fn += int(np.count_nonzero(~prediction & target))
        self.predicted_positive += int(np.count_nonzero(prediction))
        self.label_positive += int(np.count_nonzero(target))

    def result(self) -> dict[str, int | float]:
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        return {
            "count": self.count,
            "bce": self.loss_sum / max(1, self.count),
            "accuracy": self.correct / max(1, self.count),
            "precision": precision,
            "recall": recall,
            "predicted_eviction_rate": self.predicted_positive / max(1, self.count),
            "label_eviction_rate": self.label_positive / max(1, self.count),
        }


def _validation_metrics(
    model: _ExternalLinearSidecar,
    records: list[_ShardRecord],
    *,
    hidden_size: int,
    num_kv_heads: int,
    device: torch.device,
    decision_threshold: float = 0.0,
) -> dict[str, Any]:
    model.eval()
    global_metric = _Metric()
    by_layer_head: dict[str, _Metric] = {}
    by_category: dict[str, _Metric] = {}
    by_context: dict[str, _Metric] = {}
    with torch.no_grad():
        for record in records:
            hidden, eligible, labels = _load_shard(
                record,
                hidden_size=hidden_size,
                num_kv_heads=num_kv_heads,
            )
            indices = np.flatnonzero(eligible)
            if indices.size == 0:
                continue
            logits = model(
                torch.as_tensor(hidden[indices], dtype=torch.float32, device=device),
                record.compact_layer_index,
            ).to(device="cpu", dtype=torch.float32).numpy()
            targets = labels[indices]
            global_metric.update(
                logits,
                targets,
                decision_threshold=decision_threshold,
            )
            by_category.setdefault(record.category, _Metric()).update(
                logits,
                targets,
                decision_threshold=decision_threshold,
            )
            by_context.setdefault(record.context_bucket, _Metric()).update(
                logits,
                targets,
                decision_threshold=decision_threshold,
            )
            for kv_head in range(num_kv_heads):
                key = f"layer{record.compact_layer_index}:head{kv_head}"
                by_layer_head.setdefault(key, _Metric()).update(
                    logits[:, kv_head],
                    targets[:, kv_head],
                    decision_threshold=decision_threshold,
                )
    if global_metric.count == 0:
        raise ValueError("DMS validation found no eligible label rows")
    return {
        "global": global_metric.result(),
        "by_layer_head": {key: value.result() for key, value in sorted(by_layer_head.items())},
        "by_category": {key: value.result() for key, value in sorted(by_category.items())},
        "by_context_bucket": {key: value.result() for key, value in sorted(by_context.items())},
    }


def _export_arithmetic_model(
    model: _ExternalLinearSidecar,
    *,
    num_layers: int,
    num_kv_heads: int,
    hidden_size: int,
    device: torch.device,
) -> _ExternalLinearSidecar:
    exported = _ExternalLinearSidecar(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
    ).to(device)
    with torch.no_grad():
        exported.weight.copy_(model.weight.to(dtype=torch.bfloat16).to(dtype=torch.float32))
        exported.bias.copy_(model.bias.to(dtype=torch.bfloat16).to(dtype=torch.float32))
    return exported


def _threshold_for_count(values: np.ndarray, desired_count: int) -> float:
    scores = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    desired = int(desired_count)
    if scores.size <= 0 or not np.all(np.isfinite(scores)):
        raise ValueError("DMS calibration logits must be non-empty and finite")
    if desired <= 0:
        return float(np.nextafter(np.max(scores), np.inf))
    if desired >= scores.size:
        return float(np.nextafter(np.min(scores), -np.inf))
    ordered = np.sort(scores)[::-1]
    upper = float(ordered[desired - 1])
    lower = float(ordered[desired])
    if upper > lower:
        return (upper + lower) * 0.5
    return float(np.nextafter(upper, -np.inf))


def _calibrate_alpha_offset(
    model: _ExternalLinearSidecar,
    records: list[_ShardRecord],
    *,
    hidden_size: int,
    num_kv_heads: int,
    device: torch.device,
) -> dict[str, int | float]:
    values: list[np.ndarray] = []
    desired = 0
    model.eval()
    with torch.no_grad():
        for record in records:
            hidden, eligible, labels = _load_shard(
                record,
                hidden_size=hidden_size,
                num_kv_heads=num_kv_heads,
            )
            indices = np.flatnonzero(eligible)
            if indices.size == 0:
                continue
            logits = model(
                torch.as_tensor(hidden[indices], dtype=torch.float32, device=device),
                record.compact_layer_index,
            ).to(device="cpu", dtype=torch.float32).numpy()
            values.append(logits.reshape(-1))
            desired += int(np.count_nonzero(labels[indices]))
    joined = np.concatenate(values)
    offset = _threshold_for_count(joined, desired)
    return {
        "alpha_scale": 1.0,
        "alpha_offset": offset,
        "eligible_decisions": int(joined.size),
        "target_evictions": desired,
        "calibrated_evictions": int(np.count_nonzero(joined > offset)),
    }


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_checkpoint(
    path: Path,
    *,
    model: _ExternalLinearSidecar,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
    loss_history: list[dict[str, Any]],
    identity: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_epochs": int(completed_epochs),
            "loss_history": loss_history,
            "identity": identity,
        },
        temporary,
    )
    os.replace(temporary, path)


def train_sidecar(
    label_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    budget_weight: float = 0.1,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    seed: int = 0,
    resume: bool = False,
) -> dict[str, Any]:
    """Train sidecar-only parameters and export strict schema-v2 artifacts."""

    started = time.perf_counter()
    label_path = Path(label_manifest_path).expanduser().resolve()
    manifest = load_dms_label_manifest(label_path, verify_shards=True)
    label_sha256 = _sha256_file(label_path)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.iterdir()):
        raise FileExistsError(f"DMS trainer output directory must be empty: {output}")
    geometry = manifest["geometry"]
    physical_layer_ids = tuple(int(layer) for layer in geometry["physical_layer_ids"])
    num_layers = len(physical_layer_ids)
    num_q_heads = int(geometry["num_q_heads"])
    num_kv_heads = int(geometry["num_kv_heads"])
    head_dim = int(geometry["head_dim"])
    hidden_size = int(geometry["hidden_size"])
    if int(geometry["num_layers"]) != num_layers:
        raise ValueError("DMS label num_layers does not match physical layer map")
    train_records, validation_records = _records(label_path, manifest)
    completed_target = int(epochs)
    if completed_target <= 0 or int(batch_size) <= 0:
        raise ValueError("DMS epochs and batch_size must be positive")
    if float(learning_rate) <= 0.0 or float(budget_weight) < 0.0:
        raise ValueError("DMS learning rate must be positive and budget weight non-negative")
    if int(seed) < 0:
        raise ValueError("DMS seed must be non-negative")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.use_deterministic_algorithms(True)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DMS trainer requested CUDA/HIP but no device is available")
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(target_device)
    model = _ExternalLinearSidecar(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        betas=(0.9, 0.95),
        weight_decay=float(weight_decay),
    )
    identity = {
        "label_manifest_sha256": label_sha256,
        "geometry": {
            "physical_layer_ids": list(physical_layer_ids),
            "num_kv_heads": num_kv_heads,
            "hidden_size": hidden_size,
        },
        "config": {
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "budget_weight": float(budget_weight),
            "weight_decay": float(weight_decay),
            "max_grad_norm": float(max_grad_norm),
            "seed": int(seed),
        },
    }
    checkpoint_path = output / "sidecar_checkpoint.pt"
    start_epoch = 0
    loss_history: list[dict[str, Any]] = []
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DMS resume checkpoint is missing: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
        if checkpoint.get("identity") != identity:
            raise ValueError("DMS resume checkpoint identity/config does not match")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["completed_epochs"])
        loss_history = list(checkpoint["loss_history"])
        if start_epoch > completed_target:
            raise ValueError("DMS resume checkpoint is beyond requested epochs")

    for epoch in range(start_epoch, completed_target):
        epoch_started = time.perf_counter()
        row = _train_epoch(
            model,
            optimizer,
            train_records,
            epoch=epoch,
            seed=int(seed),
            batch_size=int(batch_size),
            budget_weight=float(budget_weight),
            max_grad_norm=float(max_grad_norm),
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            device=target_device,
        )
        validation = _validation_metrics(
            model,
            validation_records,
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            device=target_device,
        )
        row.update(
            {
                "epoch": epoch + 1,
                "validation_bce": validation["global"]["bce"],
                "validation_accuracy": validation["global"]["accuracy"],
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
        )
        loss_history.append(row)
        _save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            completed_epochs=epoch + 1,
            loss_history=loss_history,
            identity=identity,
        )

    export_model = _export_arithmetic_model(
        model,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        device=target_device,
    )
    calibration = _calibrate_alpha_offset(
        export_model,
        train_records,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        device=target_device,
    )
    validation = _validation_metrics(
        export_model,
        validation_records,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        device=target_device,
        decision_threshold=float(calibration["alpha_offset"]),
    )
    sidecar_path = output / "qwen38-27b-q4km-dms-sidecar.safetensors"
    save_safetensors(
        {
            "bias": export_model.bias.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
            "weight": export_model.weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
        },
        str(sidecar_path),
    )
    sidecar_sha256 = _sha256_file(sidecar_path)
    trainer_commit = _git_commit()
    model_record = manifest["model"]
    objective = manifest["objective"]
    metadata_path = output / "dms_metadata.json"
    metadata = {
        "schema_version": 2,
        "artifact_fingerprint": str(model_record["sha256"]),
        "model_family": "qwen35_dense_hybrid",
        "decision_source": _DECISION_SOURCE,
        "physical_layer_ids": list(physical_layer_ids),
        "num_layers": num_layers,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "input_stage": _INPUT_STAGE,
        "window_size": int(objective["window_size"]),
        "target_compression_ratio": int(objective["target_compression_ratio"]),
        "alpha_scale": float(calibration["alpha_scale"]),
        "alpha_offset": float(calibration["alpha_offset"]),
        "borrowed_query_channel": None,
        "zero_borrowed_query_channel": False,
        "sidecar": {
            "path": sidecar_path.name,
            "format": "safetensors",
            "dtype": "bfloat16",
            "weight_tensor": "weight",
            "bias_tensor": "bias",
            "weight_shape": [num_layers, num_kv_heads, hidden_size],
            "bias_shape": [num_layers, num_kv_heads],
            "sha256": sidecar_sha256,
        },
        "training": {
            "method": "future_attention_distillation_v1",
            "data_manifest_sha256": str(manifest["data_manifest_sha256"]),
            "trainer_commit": trainer_commit,
            "fastdms_reference_commit": _FASTDMS_REFERENCE_COMMIT,
            "seed": int(seed),
        },
        "trained_checkpoint": True,
        "evidence_source": "training_summary.json",
    }
    _write_json(metadata_path, metadata)
    metadata_sha256 = _sha256_file(metadata_path)
    parameter_count = _parameter_count(model)
    optimizer_parameter_count = _optimizer_parameter_count(optimizer)
    optimizer_state_elements = _optimizer_state_elements(optimizer)
    duration = time.perf_counter() - started
    peak_device_bytes = (
        int(torch.cuda.max_memory_allocated(target_device))
        if target_device.type == "cuda"
        else 0
    )
    summary_path = output / "training_summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "hipengine_dms_sidecar_training",
        "status": "trained_candidate",
        "model": model_record,
        "label_manifest": {"path": str(label_path), "sha256": label_sha256},
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sidecar_sha256,
        "sidecar_nbytes": sidecar_path.stat().st_size,
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_sha256,
        "checkpoint_path": str(checkpoint_path),
        "parameter_count": parameter_count,
        "optimizer_parameter_count": optimizer_parameter_count,
        "optimizer_state_elements": optimizer_state_elements,
        "completed_epochs": completed_target,
        "resumed_from_epoch": start_epoch,
        "loss_history": loss_history,
        "calibration": {
            **calibration,
            "representation": "bfloat16_export",
        },
        "validation": validation,
        "duration_seconds": duration,
        "peak_device_bytes": peak_device_bytes,
        "config": identity["config"],
        "provenance": {
            "trainer_commit": trainer_commit,
            "fastdms_reference_commit": _FASTDMS_REFERENCE_COMMIT,
            "command": [str(value) for value in sys.argv],
            "python": sys.version,
            "torch": {"version": torch.__version__, "hip": torch.version.hip},
            "device": str(target_device),
            "host": platform.node(),
        },
    }
    _write_json(summary_path, summary)
    summary["training_summary_path"] = str(summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--budget-weight", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = args.labels.expanduser().resolve()
    if labels.is_dir():
        labels = labels / "label_manifest.json"
    result = train_sidecar(
        labels,
        args.output_dir,
        device=str(args.device),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        budget_weight=float(args.budget_weight),
        weight_decay=float(args.weight_decay),
        max_grad_norm=float(args.max_grad_norm),
        seed=int(args.seed),
        resume=bool(args.resume),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
