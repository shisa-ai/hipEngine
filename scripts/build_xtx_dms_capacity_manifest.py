#!/usr/bin/env python3
"""Build the XTX-lane deterministic DMS capacity manifest (local adaptation).

Adapted from the fastdms-tree ``scripts/qwen38_dms_build_long_manifest.py``
construction (wikimedia/wikipedia 20231101 en/ja streaming pools, CPython
stdlib code pool, deterministic sha256(seed:label) selection, one train and
one validation sequence per category). Differences, recorded here and in the
output payload:

- No prior training-time source manifest is available on this host, so no
  exclusion sets are applied and the corpus is NOT train-disjoint. The output
  is authorized for same-inputs memory accounting, dispatch/trace evidence,
  and indicative same-owner KL/top-1 comparisons only; authoritative heldout
  DMS quality remains the gfx1151 lane qualification.
- Seed 20260907 (training-time seed was 20260824).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from scripts.qwen38_dms_capture import tokenizer_identity

_SEED = 20260907
_WIKI_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(label: str) -> str:
    return _sha256_bytes(f"{_SEED}:{label}".encode())


def _git() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    )
    return {"commit": commit, "working_tree_clean": not dirty}


def _code_records(*, required_pools: int) -> list[list[dict[str, Any]]]:
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    candidates: list[dict[str, Any]] = []
    for path in sorted(stdlib.rglob("*.py")):
        relative = str(path.relative_to(stdlib))
        lowered = relative.lower().split("/")
        if (
            "site-packages" in lowered
            or "test" in lowered
            or "tests" in lowered
            or "__pycache__" in lowered
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if len(text) < 1200:
            continue
        candidates.append(
            {
                "dataset": "python-stdlib-xtx-capacity-local",
                "source_id": relative,
                "path": relative,
                "chars": len(text),
                "sha256": _sha256_bytes(text.encode()),
                "text": f"# file: {relative}\n\n{text}\n",
            }
        )
    candidates.sort(key=lambda row: _rank(f"code:{row['source_id']}"))
    pools = [[] for _ in range(int(required_pools))]
    for index, record in enumerate(candidates):
        pools[index % len(pools)].append(record)
    return pools


def _wiki_records(
    *,
    config: str,
    record_limit: int,
    required_pools: int,
) -> list[list[dict[str, Any]]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "wikimedia/wikipedia",
        config,
        split="train",
        streaming=True,
    )
    candidates: list[dict[str, Any]] = []
    scanned = 0
    for row in dataset:
        scanned += 1
        text = str(row["text"])
        if len(text) >= 1200:
            candidates.append(
                {
                    "dataset": f"wikimedia-wikipedia-{config}-xtx-capacity-local",
                    "source_id": str(row["id"]),
                    "title": str(row["title"]),
                    "chars": len(text),
                    "sha256": _sha256_bytes(text.encode()),
                    "text": f"{row['title']}\n\n{text}\n",
                }
            )
        if len(candidates) >= int(record_limit):
            break
    if len(candidates) < int(record_limit):
        raise RuntimeError(
            f"{config} yielded {len(candidates)} eligible records after scanning {scanned}"
        )
    candidates.sort(key=lambda row: _rank(f"{config}:{row['source_id']}"))
    pools = [[] for _ in range(int(required_pools))]
    for index, record in enumerate(candidates):
        pools[index % len(pools)].append(record)
    return pools


def _encode_documents(
    tokenizer: Qwen35GGUFTokenizer,
    records: list[dict[str, Any]],
    *,
    target_tokens: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    output: list[int] = []
    used: list[dict[str, Any]] = []
    for record in records:
        text = str(record["text"])
        tokens = [int(token) for token in tokenizer.encode(text)]
        if not tokens:
            continue
        take = min(len(tokens), int(target_tokens) - len(output))
        output.extend(tokens[:take])
        used.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"text"}
            }
            | {"used_tokens": take}
        )
        if len(output) >= int(target_tokens):
            break
    if len(output) != int(target_tokens):
        raise RuntimeError(
            f"source pool produced {len(output)} tokens; need {int(target_tokens)}"
        )
    return output, used


def _mixed_tokens(
    en_tokens: list[int],
    ja_tokens: list[int],
    *,
    target_tokens: int,
    chunk_tokens: int = 1024,
) -> list[int]:
    output: list[int] = []
    offsets = [0, 0]
    sources = (en_tokens, ja_tokens)
    turn = 0
    while len(output) < int(target_tokens):
        source = sources[turn]
        offset = offsets[turn]
        chunk = source[offset : offset + int(chunk_tokens)]
        offsets[turn] += len(chunk)
        output.extend(chunk)
        turn = (turn + 1) % len(sources)
    return output[: int(target_tokens)]


def _source_digest(used: list[dict[str, Any]]) -> str:
    encoded = json.dumps(used, ensure_ascii=False, sort_keys=True).encode()
    return _sha256_bytes(encoded)


def _sequence(
    *,
    sequence_id: str,
    category: str,
    split: str,
    tokens: list[int],
    used: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "category": category,
        "split": split,
        "token_ids": tokens,
        "provenance": {
            "construction": "deterministic long token stream (not train-disjoint; see manifest-level scope caveat)",
            "source_ids_sha256": _source_digest(used),
            "source_count": len(used),
            "source_records": used,
            "benchmark_prompts_used": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--wiki-records", type=int, default=128)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    target = int(args.tokens)
    if target < 8192:
        raise ValueError("long DMS corpus tokens must be at least 8192")
    reader = GGUFReader(model)
    tokenizer_id, tokenizer_hash = tokenizer_identity(reader)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(reader.info)

    code_pools = _code_records(required_pools=2)
    en_pools = _wiki_records(
        config="20231101.en",
        record_limit=int(args.wiki_records),
        required_pools=4,
    )
    ja_pools = _wiki_records(
        config="20231101.ja",
        record_limit=int(args.wiki_records),
        required_pools=4,
    )

    sequences: list[dict[str, Any]] = []
    source_records: dict[str, Any] = {}
    for split_index, split in enumerate(("train", "validation")):
        code_tokens, code_used = _encode_documents(
            tokenizer, code_pools[split_index], target_tokens=target
        )
        en_tokens, en_used = _encode_documents(
            tokenizer, en_pools[split_index], target_tokens=target
        )
        ja_tokens, ja_used = _encode_documents(
            tokenizer, ja_pools[split_index], target_tokens=target
        )
        mixed_en, mixed_en_used = _encode_documents(
            tokenizer, en_pools[2 + split_index], target_tokens=target
        )
        mixed_ja, _mixed_ja_used = _encode_documents(
            tokenizer, ja_pools[2 + split_index], target_tokens=target
        )
        mixed = _mixed_tokens(mixed_en, mixed_ja, target_tokens=target)
        rows = (
            ("code", code_tokens, code_used),
            ("general_en", en_tokens, en_used),
            ("general_ja", ja_tokens, ja_used),
            ("mixed_ja_en", mixed, mixed_en_used + _mixed_ja_used),
        )
        for category, tokens, used in rows:
            sequence_id = f"xtx-{split}-{category}-{target}"
            sequences.append(
                _sequence(
                    sequence_id=sequence_id,
                    category=category,
                    split=split,
                    tokens=tokens,
                    used=used,
                )
            )
            source_records[sequence_id] = used

    payload = {
        "schema_version": 1,
        "kind": "hipengine_dms_training_data_manifest",
        "seed": _SEED,
        "scope_caveat": {
            "train_disjoint": False,
            "reason": "no training-time source manifest is available on this host; no exclusion sets applied",
            "authorized_uses": [
                "same-inputs memory accounting across dense/no-evict/sidecar arms",
                "dispatch and compact trace evidence",
                "indicative same-owner KL/top-1 comparison against the dense teacher",
            ],
            "not_authorized": "authoritative DMS heldout quality claims (gfx1151 lane qualification remains authoritative)",
        },
        "tokenizer": {"identity": tokenizer_id, "sha256": tokenizer_hash},
        "datasets": [
            {
                "name": "CPython standard library",
                "revision": f"{sys.version.split()[0]} at {sysconfig.get_path('stdlib')}",
                "license": "PSF-2.0",
                "splits": ["local-source"],
                "filters": {
                    "selection": "sha256(seed:path) ranking",
                    "excluded": ["site-packages", "test", "tests", "__pycache__"],
                },
            },
            {
                "name": "wikimedia/wikipedia",
                "revision": _WIKI_REVISION,
                "license": "CC-BY-SA-3.0 and GFDL",
                "splits": ["20231101.en/train", "20231101.ja/train"],
                "filters": {
                    "min_chars": 1200,
                    "selection": "first eligible records, then sha256(seed:config:id) ranking",
                    "record_limit_per_language": int(args.wiki_records),
                },
            },
        ],
        "sequence_construction": {
            "benchmark_prompts_used": False,
            "excluded_source_manifests": [],
            "categories": ["code", "general_en", "general_ja", "mixed_ja_en"],
            "length_tokens": target,
            "method": "one calibration and one validation sequence per category; mixed alternates 1024-token en/ja chunks",
        },
        "context_length_distribution": {str(target): len(sequences)},
        "split_policy": {
            "method": "one per-category calibration and one disjoint validation sequence",
            "train_per_category": 1,
            "validation_per_category": 1,
        },
        "provenance": _git(),
        "sequences": sequences,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_path = output.with_name(output.stem + "-sources.json")
    source_payload = {
        "schema_version": 1,
        "kind": "hipengine_dms_xtx_capacity_source_manifest",
        "seed": _SEED,
        "data_manifest": str(output),
        "provenance": _git(),
        "sequences": source_records,
    }
    source_path.write_text(
        json.dumps(source_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(output),
        "sha256": _sha256_file(output),
        "bytes": output.stat().st_size,
        "source_manifest": str(source_path),
        "source_manifest_sha256": _sha256_file(source_path),
        "sequences": len(sequences),
        "tokens": len(sequences) * target,
        "splits": {
            split: sum(row["split"] == split for row in sequences)
            for split in ("train", "validation")
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
