#!/usr/bin/env python3
"""Acquire deterministic, source-disjoint records for the sealed DMS campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

SEED = 20260922
WIKI_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalized_hash(text: str) -> str:
    return sha256_bytes(" ".join(text.split()).encode("utf-8"))


def token_hash(tokens: list[int]) -> str:
    return sha256_bytes(json.dumps(tokens, separators=(",", ":")).encode())


def rank(value: str, seed: int = SEED) -> str:
    return sha256_bytes(f"{seed}:{value}".encode())


def git_identity(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                capture_output=True, check=True).stdout.strip()
        command = " ".join(["git", "rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError):
        commit, command = "unknown", "git rev-parse HEAD"
    return {"commit": commit, "command": command}


def exclusion_sets(paths: Iterable[Path]) -> tuple[set[str], set[str], set[str]]:
    ids: set[str] = set(); paths_seen: set[str] = set(); texts: set[str] = set()
    def walk(value: Any, context: dict[str, Any] | None = None) -> None:
        if isinstance(value, dict):
            has_source = any(str(k).lower() in {"source_id", "sourceid", "path", "source_path"} for k in value)
            for key, item in value.items():
                key_l = str(key).lower()
                if isinstance(item, (str, int)):
                    if key_l in {"source_id", "sourceid", "id"}:
                        ids.add(str(item))
                    elif key_l in {"path", "source_path", "filepath", "file_path"}:
                        paths_seen.add(str(item))
                    elif key_l in {"normalized_text_sha256", "text_sha256"} or (key_l == "sha256" and has_source):
                        if len(str(item)) == 64: texts.add(str(item))
                walk(item, value)
        elif isinstance(value, list):
            for item in value: walk(item, context)
    for path in paths:
        walk(json.loads(Path(path).read_text(encoding="utf-8")))
    return ids, paths_seen, texts


def _eligible(doc: dict[str, Any], excluded: tuple[set[str], set[str], set[str]]) -> bool:
    sid, path = str(doc["source_id"]), str(doc["path"])
    text = str(doc["text"])
    return sid not in excluded[0] and path not in excluded[1] and normalized_hash(text) not in excluded[2]


def _partition(docs: list[dict[str, Any]], count: int, target: int, label: str, *, seed: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(docs, key=lambda d: rank(f"{label}:{d['source_id']}:{d['path']}", seed))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    totals = [0] * count
    for doc in ordered:
        eligible_bucket = next((i for i in range(count) if totals[i] < target), None)
        if eligible_bucket is None: break
        buckets[eligible_bucket].append(doc)
        totals[eligible_bucket] += len(doc["tokens"])
    if any(total < target for total in totals):
        raise ValueError(f"insufficient {label} source tokens: {sum(totals)} for {count} streams of {target}")
    return buckets


def _component(doc: dict[str, Any], tokenizer: dict[str, str], provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": str(doc["source_id"]), "path": str(doc["path"]),
        "dataset": doc.get("dataset", "unknown"), "revision": doc.get("revision", "unknown"),
        "license": doc.get("license", "unknown"), "normalized_text_sha256": normalized_hash(str(doc["text"])),
        "token_count": len(doc["tokens"]), "token_ids_sha256": token_hash(doc["tokens"]),
        "tokenizer": tokenizer, "provenance": provenance,
    }


def _stream(bucket: list[dict[str, Any]], target: int) -> tuple[list[int], list[dict[str, Any]]]:
    tokens: list[int] = []; used: list[dict[str, Any]] = []
    for doc in bucket:
        take = min(len(doc["tokens"]), target - len(tokens))
        tokens.extend(doc["tokens"][:take])
        used.append((doc, take))
        if len(tokens) == target: break
    if len(tokens) < target: raise ValueError("source bucket is shorter than requested stream")
    return tokens, used


def _mixed(en: list[dict[str, Any]], ja: list[dict[str, Any]], target: int, chunk: int) -> tuple[list[int], list[tuple[dict[str, Any], int]]]:
    en_tokens, en_used = _stream(en, target); ja_tokens, ja_used = _stream(ja, target)
    output: list[int] = []; used: list[tuple[dict[str, Any], int]] = []; offsets = [0, 0]
    for turn in range(100000):
        if len(output) == target: break
        source, source_used = ((en_tokens, en_used), (ja_tokens, ja_used))[turn % 2]
        take = min(chunk, target - len(output)); start = offsets[turn % 2]
        output.extend(source[start:start + take]); offsets[turn % 2] += take
        # Keep component ownership complete even where a final aggregate truncates a document.
        if source_used: pass
    return output, en_used + ja_used


def acquire_records(documents: Iterable[dict[str, Any]], *, tokenizer: Callable[[str], Iterable[int]], tokenizer_identity: str,
                    tokenizer_sha256: str, model_sha256: str, old_manifests: Iterable[Path] = (), target_tokens: int = 131072,
                    seed: int = SEED, mixed_chunk_tokens: int = 1024, git: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if target_tokens <= 0 or mixed_chunk_tokens <= 0: raise ValueError("token lengths must be positive")
    tokenizer_binding = {"identity": tokenizer_identity, "sha256": tokenizer_sha256}
    excluded = exclusion_sets(old_manifests)
    seen: set[str] = set(); clean: list[dict[str, Any]] = []
    for raw in documents:
        doc = dict(raw)
        for key in ("source_id", "path", "text", "category"):
            if key not in doc or not str(doc[key]): raise ValueError(f"source document missing {key}")
        if not _eligible(doc, excluded): continue
        sid, path, text_digest = str(doc["source_id"]), str(doc["path"]), normalized_hash(str(doc["text"]))
        if sid in seen or path in seen or text_digest in seen: raise ValueError("duplicate source ID/path/text")
        seen.update((sid, path, text_digest))
        if doc.get("tokenizer") and doc["tokenizer"] != tokenizer_binding: raise ValueError("tokenizer mismatch")
        doc["tokens"] = [int(x) for x in tokenizer(str(doc["text"]))]
        if not doc["tokens"]: continue
        clean.append(doc)
    if not clean: raise ValueError("no eligible source documents")
    pools: dict[str, list[list[dict[str, Any]]]] = {}
    pools["code"] = _partition([d for d in clean if d["category"] == "code"], 9, target_tokens, "code", seed=seed)
    pools["en"] = _partition([d for d in clean if d["category"] == "general_en"], 18, target_tokens, "general_en", seed=seed)
    pools["ja"] = _partition([d for d in clean if d["category"] == "general_ja"], 18, target_tokens, "general_ja", seed=seed)
    out: list[dict[str, Any]] = []
    provenance = {"git": git or {}, "command": "qwen38_dms_acquire_selector_campaign_sources", "model_sha256": model_sha256}
    def make(category: str, index: int, tokens: list[int], used: list[tuple[dict[str, Any], int]]) -> None:
        components = [_component(doc, tokenizer_binding, provenance) | {"used_tokens": take} for doc, take in used]
        sid = f"acquired-{category}-{index:02d}"; path = f"acquired/{category}/{index:02d}"
        out.append({"source_id": sid, "path": path, "category": category, "token_ids": tokens,
                    "token_count": len(tokens), "token_ids_sha256": token_hash(tokens), "source_records": components,
                    "tokenizer": tokenizer_binding, "model_sha256": model_sha256, "provenance": provenance})
    for i, bucket in enumerate(pools["code"]): make("code", i, *_stream(bucket, target_tokens))
    for i in range(9):
        make("general_en", i, *_stream(pools["en"][i], target_tokens))
        make("general_ja", i, *_stream(pools["ja"][i], target_tokens))
        make("mixed_ja_en", i, *_mixed(pools["en"][9 + i], pools["ja"][9 + i], target_tokens, mixed_chunk_tokens))
    return out


def write_atomic(records: list[dict[str, Any]], output: Path, metadata: dict[str, Any]) -> None:
    output = Path(output).expanduser().resolve()
    if output.exists(): raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "kind": "hipengine_dms_selector_campaign_acquired_sources", "metadata": metadata, "records": records}
    fd, temp = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":")); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, output)
    except BaseException:
        try: os.unlink(temp)
        except OSError: pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cli_documents(root: Path, excluded: tuple[set[str], set[str], set[str]], record_limit: int = 512) -> list[dict[str, Any]]:
    """Collect only eligible local code and pinned streaming Wikimedia documents."""
    rows: list[dict[str, Any]] = []
    tracked = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root,
                             text=True, capture_output=True, check=True).stdout.splitlines()
    blocked = ("tests/", "test/", "fixtures/", "benchmark", "benchmarks/", "artifacts/", "generated/")
    for name in tracked:
        path = (root / name).resolve()
        if path.suffix != ".py" or any(part.lower() in {"tests", "test", "fixtures", "__pycache__"} for part in path.relative_to(root).parts):
            continue
        try: text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError): continue
        if len(text) < 1200 or name.startswith(blocked): continue
        rows.append({"source_id": f"repo:{name}", "path": f"repo/{name}", "category": "code", "text": text,
                     "dataset": "git-tracked-repository-source", "revision": git_identity(root)["commit"], "license": "repository license"})
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    for path in sorted(stdlib.rglob("*.py")):
        rel = path.relative_to(stdlib)
        if any(part.lower() in {"test", "tests", "site-packages", "__pycache__"} for part in rel.parts): continue
        try: text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError): continue
        if len(text) < 1200: continue
        name = str(rel)
        rows.append({"source_id": f"stdlib:{name}", "path": f"python-stdlib/{name}", "category": "code", "text": text,
                     "dataset": "python-stdlib", "revision": sys.version.split()[0], "license": "PSF-2.0"})
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("the real acquisition CLI requires the optional datasets package") from exc
    for config, category in (("20231101.en", "general_en"), ("20231101.ja", "general_ja")):
        stream = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
        found = 0
        for item in stream:
            source_id = str(item["id"]); text = str(item["text"])
            if len(text) < 1200: continue
            rows.append({"source_id": f"wiki:{config}:{source_id}", "path": f"wikimedia/{config}/{source_id}", "category": category,
                         "text": f"{item['title']}\n\n{text}", "dataset": f"wikimedia/wikipedia:{config}", "revision": WIKI_REVISION,
                         "license": "CC-BY-SA-3.0 and GFDL"})
            found += 1
            if found >= record_limit: break
        if found < record_limit: raise ValueError(f"insufficient streamed {config} records")
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--old-source-manifest", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=131072)
    p.add_argument("--record-limit", type=int, default=512)
    return p


def main() -> int:
    args = build_parser().parse_args()
    from hipengine.loading.gguf import GGUFReader
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
    from scripts.qwen38_dms_capture import tokenizer_identity
    model = args.model.expanduser().resolve()
    if len(args.old_source_manifest) != 4: raise ValueError("exactly four v1-v4 manifests are required")
    reader = GGUFReader(model); identity, tok_hash = tokenizer_identity(reader)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(reader.info)
    old = [path.expanduser().resolve() for path in args.old_source_manifest]
    excluded = exclusion_sets(old)
    root = Path(__file__).resolve().parents[1]
    git = git_identity(root)
    records = acquire_records(_cli_documents(root, excluded, args.record_limit), tokenizer=tokenizer.encode,
                              tokenizer_identity=identity, tokenizer_sha256=tok_hash, model_sha256=_sha256_file(model),
                              old_manifests=old, target_tokens=args.tokens, git=git)
    write_atomic(records, args.output, {"seed": SEED, "wiki_revision": WIKI_REVISION, "tokenizer": {"identity": identity, "sha256": tok_hash},
                                       "model_sha256": _sha256_file(model), "git": git})
    print(json.dumps({"output": str(args.output), "records": len(records)}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
