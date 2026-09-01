#!/usr/bin/env python3
"""Build a model-bound, corpus-ranked selected vocabulary for GGUF MTP."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable

from hipengine.loading.gguf import GGUFReader
from hipengine.loading.gguf_mtp_hot_vocab import (
    HOT_VOCAB_KIND,
    HOT_VOCAB_SCHEMA_VERSION,
    gguf_tokenizer_tokens_sha256,
)
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

_TEXT_EXTENSIONS = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".json",
        ".jsonl",
        ".jsx",
        ".md",
        ".py",
        ".rs",
        ".sh",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }
)


def _corpus_paths(roots: Iterable[Path]) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for root in roots:
        if root.is_file():
            paths.add(root)
            continue
        if not root.is_dir():
            raise ValueError(f"corpus path does not exist: {root}")
        paths.update(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in _TEXT_EXTENSIONS
        )
    return tuple(sorted(paths, key=lambda path: str(path)))


def _matches_script(text: str, script: str) -> bool:
    if script == "cjk":
        return any(
            "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff"
            for char in text
        )
    raise ValueError(f"unsupported tokenizer script: {script}")


def _script_token_ids(
    tokenizer: Qwen35GGUFTokenizer,
    scripts: Iterable[str],
) -> set[int]:
    requested = tuple(dict.fromkeys(str(script) for script in scripts))
    return {
        token_id
        for token_id in range(len(tokenizer.tokens))
        if any(
            _matches_script(tokenizer.decode([token_id]), script)
            for script in requested
        )
    }


def _required_token_ids(info) -> set[int]:
    metadata = info.metadata
    required = {
        index
        for index, kind in enumerate(metadata.get("tokenizer.ggml.token_type", ()))
        if int(kind) in {3, 4}
    }
    for key, value in metadata.items():
        if key.startswith("tokenizer.ggml.") and key.endswith("_token_id"):
            if value is not None:
                required.add(int(value))
    return required


def build(args: argparse.Namespace) -> dict[str, object]:
    if args.tokens <= 0 or args.tokens % 16:
        raise ValueError("--tokens must be positive and divisible by 16")
    reader = GGUFReader(args.model)
    info = reader.info
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(info)
    vocab_size = len(tokenizer.tokens)
    if args.tokens > vocab_size:
        raise ValueError("--tokens exceeds the model vocabulary")

    paths = _corpus_paths(Path(value) for value in args.corpus)
    counts: Counter[int] = Counter()
    corpus_digest = hashlib.sha256()
    corpus_tokens = 0
    corpus_chars = 0
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")[: args.max_chars_per_file]
        encoded_path = str(path).encode("utf-8")
        encoded_text = text.encode("utf-8")
        corpus_digest.update(len(encoded_path).to_bytes(8, "little"))
        corpus_digest.update(encoded_path)
        corpus_digest.update(hashlib.sha256(encoded_text).digest())
        if not text:
            continue
        ids = tokenizer.encode(text)
        counts.update(ids)
        corpus_tokens += len(ids)
        corpus_chars += len(text)

    special = _required_token_ids(info)
    script_ids = _script_token_ids(tokenizer, args.include_script)
    required = special | script_ids
    if len(required) > args.tokens:
        raise ValueError("--tokens does not leave room for all required tokenizer rows")
    selected = set(required)
    for token_id, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if len(selected) >= args.tokens:
            break
        selected.add(int(token_id))
    for token_id in range(vocab_size):
        if len(selected) >= args.tokens:
            break
        selected.add(token_id)
    token_ids = sorted(selected)
    covered = sum(count for token_id, count in counts.items() if token_id in selected)

    payload: dict[str, object] = {
        "schema_version": HOT_VOCAB_SCHEMA_VERSION,
        "kind": HOT_VOCAB_KIND,
        "model": {
            "path": str(Path(args.model)),
            "architecture": info.metadata.get("general.architecture"),
            "basename": info.metadata.get("general.basename"),
            "block_count": info.metadata.get("qwen35.block_count"),
            "file_type": info.file_type_name,
            "vocab_size": vocab_size,
            "tokenizer_tokens_sha256": gguf_tokenizer_tokens_sha256(info),
            "output_weight": next(
                {
                    "name": tensor.name,
                    "shape": list(tensor.shape),
                    "ggml_type": tensor.ggml_type_name,
                }
                for tensor in info.tensors
                if tensor.name == "output.weight"
            ),
        },
        "selection": {
            "strategy": "required_scripts_then_corpus_frequency_then_token_id",
            "corpus_roots": [str(Path(value)) for value in args.corpus],
            "corpus_files": len(paths),
            "corpus_chars": corpus_chars,
            "corpus_tokens": corpus_tokens,
            "corpus_sha256": corpus_digest.hexdigest(),
            "required_special_tokens": len(special),
            "required_script_tokens": len(script_ids),
            "included_scripts": list(args.include_script),
            "selected_tokens": len(token_ids),
            "corpus_token_coverage": covered / max(corpus_tokens, 1),
        },
        "token_ids": token_ids,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", action="append", required=True)
    parser.add_argument("--tokens", type=int, default=65_536)
    parser.add_argument("--max-chars-per-file", type=int, default=1_000_000)
    parser.add_argument(
        "--include-script",
        action="append",
        choices=("cjk",),
        default=[],
        help="Require every tokenizer token containing this Unicode script",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build(args)
    print(json.dumps(payload["selection"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
