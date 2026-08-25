#!/usr/bin/env python3
"""Capture bounded exact-Q4 full-attention intermediates for external DMS training."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.kvcache.dms_capture import DMS_CAPTURE_INPUT_STAGE, DMSCaptureWriter
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen35_gguf import (
    FULL_ATTENTION,
    qwen35_gguf_config_from_metadata,
)
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

_EXPECTED_FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))
_REQUIRED_CATEGORIES = frozenset({"code", "general_en", "general_ja", "mixed_ja_en"})
_EVALUATION_ONLY_BASENAME = "mtpbench-code-general-ja.jsonl"
_DATA_MANIFEST_KIND = "hipengine_dms_training_data_manifest"
_DATA_MANIFEST_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class CaptureSequence:
    sequence_id: str
    category: str
    token_ids: tuple[int, ...]
    provenance: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_identity(reader: GGUFReader) -> tuple[str, str]:
    metadata = reader.info.metadata
    tokenizer_metadata = {
        key: metadata[key]
        for key in sorted(metadata)
        if key.startswith("tokenizer.")
    }
    encoded = json.dumps(
        tokenizer_metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    identity = ":".join(
        (
            str(reader.info.architecture),
            str(metadata.get("tokenizer.ggml.model", "")),
            str(metadata.get("tokenizer.ggml.pre", "")),
        )
    )
    return identity, hashlib.sha256(encoded).hexdigest()


def _contains_evaluation_source(value: Any) -> bool:
    if isinstance(value, str):
        return _EVALUATION_ONLY_BASENAME.lower() in value.lower()
    if isinstance(value, dict):
        return any(
            _contains_evaluation_source(key) or _contains_evaluation_source(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_evaluation_source(item) for item in value)
    return False


def _validate_dataset_records(payload: dict[str, Any]) -> None:
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("DMS data manifest requires a non-empty datasets list")
    required = {"name", "revision", "license", "splits", "filters"}
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise TypeError(f"DMS dataset record {index} must be an object")
        missing = sorted(required - set(dataset))
        if missing:
            raise ValueError(f"DMS dataset record {index} is missing {missing}")
        for field in ("name", "revision", "license"):
            if not str(dataset[field]).strip():
                raise ValueError(f"DMS dataset record {index} has empty {field}")
        if not isinstance(dataset["splits"], list) or not dataset["splits"]:
            raise ValueError(f"DMS dataset record {index} requires splits")
        if not isinstance(dataset["filters"], (dict, list)):
            raise TypeError(f"DMS dataset record {index} filters must be an object or list")
    for field in ("sequence_construction", "context_length_distribution", "split_policy"):
        if not isinstance(payload.get(field), dict) or not payload[field]:
            raise ValueError(f"DMS data manifest requires non-empty {field}")
    if int(payload.get("seed", -1)) < 0:
        raise ValueError("DMS data manifest seed must be non-negative")


def load_capture_sequences(
    path: str | Path,
    *,
    tokenizer: Qwen35GGUFTokenizer,
    tokenizer_identity_value: str,
    tokenizer_sha256: str,
    max_sequence_tokens: int,
    max_sequences: int | None = None,
    require_all_categories: bool = True,
) -> tuple[str, list[CaptureSequence]]:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.name.lower() == _EVALUATION_ONLY_BASENAME:
        raise ValueError("mtp-bench is evaluation-only and cannot be a DMS training manifest")
    raw_bytes = manifest_path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("DMS data manifest must be an object")
    if payload.get("schema_version") != _DATA_MANIFEST_SCHEMA:
        raise ValueError("unsupported DMS data manifest schema")
    if payload.get("kind") != _DATA_MANIFEST_KIND:
        raise ValueError("invalid DMS data manifest kind")
    if _contains_evaluation_source(payload):
        raise ValueError("DMS data manifest references the evaluation-only mtp-bench suite")
    _validate_dataset_records(payload)
    tokenizer_record = payload.get("tokenizer")
    if not isinstance(tokenizer_record, dict):
        raise TypeError("DMS data manifest tokenizer must be an object")
    if str(tokenizer_record.get("identity", "")) != str(tokenizer_identity_value):
        raise ValueError("DMS data manifest tokenizer identity does not match the GGUF")
    if str(tokenizer_record.get("sha256", "")) != str(tokenizer_sha256):
        raise ValueError("DMS data manifest tokenizer hash does not match the GGUF")
    records = payload.get("sequences")
    if not isinstance(records, list) or not records:
        raise ValueError("DMS data manifest requires non-empty sequences")
    limit = len(records) if max_sequences is None else min(len(records), int(max_sequences))
    if limit <= 0:
        raise ValueError("max_sequences must select at least one sequence")
    max_tokens = int(max_sequence_tokens)
    if max_tokens < 4:
        raise ValueError("max_sequence_tokens must be at least four")
    sequences: list[CaptureSequence] = []
    seen: set[str] = set()
    for index, record in enumerate(records[:limit]):
        if not isinstance(record, dict):
            raise TypeError(f"DMS sequence record {index} must be an object")
        sequence_id = str(record.get("sequence_id", ""))
        category = str(record.get("category", ""))
        if not sequence_id.strip() or sequence_id in seen:
            raise ValueError(f"DMS sequence record {index} has empty/duplicate sequence_id")
        seen.add(sequence_id)
        if category not in _REQUIRED_CATEGORIES:
            raise ValueError(f"DMS sequence record {index} has unsupported category {category!r}")
        split = str(record.get("split", ""))
        if split not in {"train", "validation"}:
            raise ValueError("DMS capture sequences must use train or validation splits")
        has_tokens = "token_ids" in record
        has_text = "text" in record
        if has_tokens == has_text:
            raise ValueError("each DMS sequence requires exactly one of token_ids or text")
        if has_tokens:
            token_values = record["token_ids"]
            if not isinstance(token_values, list):
                raise TypeError("DMS sequence token_ids must be a list")
            token_ids = [int(token) for token in token_values]
        else:
            token_ids = [int(token) for token in tokenizer.encode(str(record["text"]))]
        original_token_count = len(token_ids)
        token_ids = token_ids[:max_tokens]
        if len(token_ids) < 4 or any(token < 0 for token in token_ids):
            raise ValueError("DMS capture sequences require at least four non-negative tokens")
        provenance = record.get("provenance")
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("DMS sequence provenance must be a non-empty object")
        provenance = {
            **provenance,
            "split": split,
            "manifest_sequence_index": index,
            "original_token_count": original_token_count,
            "captured_token_count": len(token_ids),
        }
        sequences.append(
            CaptureSequence(
                sequence_id=sequence_id,
                category=category,
                token_ids=tuple(token_ids),
                provenance=provenance,
            )
        )
    categories = {sequence.category for sequence in sequences}
    if require_all_categories and categories != _REQUIRED_CATEGORIES:
        raise ValueError(
            "DMS capture requires all training categories; "
            f"missing={sorted(_REQUIRED_CATEGORIES - categories)}"
        )
    return hashlib.sha256(raw_bytes).hexdigest(), sequences


def capture_provenance(args: argparse.Namespace, *, compiler_version: str | None) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    uname = platform.uname()
    return {
        "source_commit": head,
        "working_tree_clean": not bool(status.strip()),
        "command": [str(value) for value in sys.argv],
        "backend": str(args.backend),
        "compiler_version": compiler_version,
        "python": sys.version,
        "host": {
            "node": uname.node,
            "system": uname.system,
            "release": uname.release,
            "machine": uname.machine,
        },
    }


def validate_qwen38_geometry(reader: GGUFReader) -> tuple[Any, tuple[int, ...]]:
    config = qwen35_gguf_config_from_metadata(reader.info)
    full_attention_layer_ids = tuple(
        index
        for index, layer_type in enumerate(config.layer_types)
        if layer_type == FULL_ATTENTION
    )
    actual = {
        "architecture": reader.info.architecture,
        "file_type": reader.info.file_type_name,
        "declared_block_count": int(config.declared_block_count),
        "block_count": int(config.block_count),
        "ignored_block_ids": tuple(int(value) for value in config.ignored_block_ids),
        "hidden_size": int(config.hidden_size),
        "head_count": int(config.head_count),
        "head_count_kv": int(config.head_count_kv),
        "key_length": int(config.key_length),
        "context_length": int(config.context_length),
        "full_attention_layer_ids": full_attention_layer_ids,
    }
    expected = {
        "architecture": "qwen35",
        "file_type": "MOSTLY_Q4_K_M",
        "declared_block_count": 65,
        "block_count": 64,
        "ignored_block_ids": (64,),
        "hidden_size": 5120,
        "head_count": 24,
        "head_count_kv": 4,
        "key_length": 256,
        "context_length": 262144,
        "full_attention_layer_ids": _EXPECTED_FULL_ATTENTION_LAYERS,
    }
    if actual != expected:
        raise ValueError(f"GGUF geometry does not match Qwen3.8-27B Q4_K_M: {actual}")
    return config, full_attention_layer_ids


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    model_sha256 = _sha256_file(model)
    if args.expected_artifact is not None and model_sha256 != args.expected_artifact:
        raise ValueError(
            f"GGUF SHA-256 mismatch: expected {args.expected_artifact}, got {model_sha256}"
        )
    reader = GGUFReader(model)
    config, physical_layer_ids = validate_qwen38_geometry(reader)
    tokenizer_id, tokenizer_hash = tokenizer_identity(reader)
    if args.print_tokenizer_identity:
        return {
            "model_sha256": model_sha256,
            "tokenizer": {"identity": tokenizer_id, "sha256": tokenizer_hash},
        }
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(reader.info)
    data_manifest_sha256, sequences = load_capture_sequences(
        args.data_manifest,
        tokenizer=tokenizer,
        tokenizer_identity_value=tokenizer_id,
        tokenizer_sha256=tokenizer_hash,
        max_sequence_tokens=int(args.max_sequence_tokens),
        max_sequences=args.max_sequences,
        require_all_categories=not bool(args.allow_incomplete_categories),
    )
    max_rows = max(len(sequence.token_ids) for sequence in sequences)
    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8").strip()
    writer = DMSCaptureWriter(
        args.output_dir,
        model_path=str(model),
        model_sha256=model_sha256,
        data_manifest_sha256=data_manifest_sha256,
        tokenizer_identity=tokenizer_id,
        tokenizer_sha256=tokenizer_hash,
        physical_layer_ids=physical_layer_ids,
        num_q_heads=int(config.head_count),
        num_kv_heads=int(config.head_count_kv),
        head_dim=int(config.key_length),
        hidden_size=int(config.hidden_size),
        input_stage=DMS_CAPTURE_INPUT_STAGE,
        qk_storage_dtype=str(args.qk_storage_dtype),
        teacher_topk=int(args.teacher_topk),
        max_shard_bytes=int(args.max_shard_bytes),
        capture_provenance=capture_provenance(
            args,
            compiler_version=compiler_version,
        ),
    )
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    with Qwen35GGUFResidentSession(
        model,
        backend=str(args.backend),
        max_sequence_length=max_rows + 1,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        for sequence in sequences:
            writer.begin_sequence(
                sequence_id=sequence.sequence_id,
                token_ids=sequence.token_ids,
                category=sequence.category,
                provenance=sequence.provenance,
            )
            session.prefill(
                list(sequence.token_ids),
                use_bulk=True,
                bulk_attention_mode="bulk",
                return_logits=True,
                dms_capture=writer,
            )
            writer.finish_sequence()
    capture_manifest = writer.finalize()
    capture_sha256 = _sha256_file(capture_manifest)
    return {
        "status": "captured",
        "model_sha256": model_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "capture_manifest": str(capture_manifest),
        "capture_manifest_sha256": capture_sha256,
        "sequence_count": len(sequences),
        "token_count": sum(len(sequence.token_ids) for sequence in sequences),
        "physical_layer_ids": list(physical_layer_ids),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"),
    )
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-artifact", default=None)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--max-sequence-tokens", type=int, default=4096)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--qk-storage-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--teacher-topk", type=int, default=64)
    parser.add_argument("--max-shard-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--allow-incomplete-categories", action="store_true")
    parser.add_argument("--print-tokenizer-identity", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.print_tokenizer_identity:
        if args.data_manifest is None:
            parser.error("--data-manifest is required unless --print-tokenizer-identity is used")
        if args.output_dir is None:
            parser.error("--output-dir is required unless --print-tokenizer-identity is used")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
