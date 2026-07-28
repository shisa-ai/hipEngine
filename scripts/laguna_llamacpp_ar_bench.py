#!/usr/bin/env python3
"""Benchmark llama.cpp against Laguna's matched post-TTFT AR contract.

The primary metric counts only synchronized model transitions after the first
sampled token.  llama.cpp starts ``predicted_ms`` after that first token while
``predicted_n`` still includes it, so the comparable rate is
``sum(predicted_n - 1) / sum(predicted_ms)``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from hipengine.loading.gguf import GGUFReader
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_target_ar_bench import (
    EXPECTED_CATEGORIES,
    EXPECTED_PROMPT_COUNT,
    RETAINED_HORIZONS,
    _load_prompts,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = Path("/tmp/llama-c0bc8591-hip-build/bin/llama-server")
DEFAULT_SOURCE = Path("/tmp/llama-c0bc8591-hip-src")
DEFAULT_SOURCE_REVISION = "c0bc8591e8815c63cb01dd3f051a8b0df02501c9"
DEFAULT_REFERENCE_REPO = Path("/home/lhl/llama.cpp/llama.cpp-hip")
DEFAULT_MODEL = Path("/models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf")
DEFAULT_MODEL_SHA256 = "8fe1170f012723f6f7d6c9b08d8f928b0b3d8bffc32926f33a930148a1d62679"
DEFAULT_PROMPTS = ROOT / "benchmarks/prompts/laguna-target-ar-code-general-ja-heldout.jsonl"
DEFAULT_HIPENGINE_ARTIFACT = Path(
    "/tmp/laguna-iq3-wave10-fused-category-candidate-a.json"
)
DEFAULT_SERVER_LOG = Path("/tmp/laguna-llamacpp-hip-matched-server.log")
DEFAULT_GTT_PATH = Path("/sys/class/drm/card1/device/mem_info_gtt_used")
MATCHED_PROMPT_SHA256 = "3097ed25c6f4cf3c2986c1da90e61d1600c3b291745224313dba5100fa7a8e76"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--source-patch",
        type=Path,
        action="append",
        help="declared measurement-only patch applied after the named source revision",
    )
    parser.add_argument("--reference-repo", type=Path, default=DEFAULT_REFERENCE_REPO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--hipengine-artifact",
        type=Path,
        action="append",
        help="raw passing hipEngine category artifact; repeat to pool process orders",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument(
        "--output-horizons",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        default=RETAINED_HORIZONS,
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--cache-type-k", default="bf16")
    parser.add_argument("--cache-type-v", default="bf16")
    parser.add_argument("--flash-attention", choices=("on", "off"), default="on")
    parser.add_argument("--mmap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--repack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--skip-chat-parsing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--hip-visible-devices", default="0")
    parser.add_argument("--gpu-max-hw-queues", default="1")
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--server-log", type=Path, default=DEFAULT_SERVER_LOG)
    parser.add_argument("--gtt-path", type=Path, default=DEFAULT_GTT_PATH)
    parser.add_argument("--allow-dirty-harness", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _git_state(path: Path) -> dict[str, Any]:
    root = path.resolve()
    revision = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=root, text=True
    ).strip()
    tracked = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=root,
        text=True,
    ).strip()
    return {
        "path": str(root),
        "revision": revision,
        "tracked_clean": not bool(tracked),
        "tracked_status": tracked.splitlines(),
    }


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _verify_source_archive(
    source_dir: Path,
    reference_repo: Path,
    revision: str,
    *,
    patches: list[Path] | None = None,
) -> dict[str, Any]:
    source = source_dir.resolve()
    reference = reference_repo.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"llama.cpp source directory not found: {source}")
    if not reference.is_dir():
        raise FileNotFoundError(f"llama.cpp reference repository not found: {reference}")
    full_revision = subprocess.check_output(
        ("git", "rev-parse", f"{revision}^{{commit}}"), cwd=reference, text=True
    ).strip()
    tree = subprocess.check_output(
        ("git", "ls-tree", "-rz", "--full-tree", full_revision), cwd=reference
    )
    expected: dict[str, tuple[str, str]] = {}
    for record in tree.split(b"\0"):
        if not record:
            continue
        header, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id = header.decode("ascii").split()
        if object_type != "blob":
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        expected[relative] = (mode, object_id)

    declared_patches = [path.resolve() for path in (patches or [])]
    patch_records = [
        {"path": str(path), "sha256": _sha256_file(path)}
        for path in declared_patches
    ]
    if declared_patches:
        touched: set[str] = set()
        for patch in declared_patches:
            if not patch.is_file():
                raise FileNotFoundError(f"declared source patch not found: {patch}")
            numstat = subprocess.check_output(
                ("git", "apply", "--numstat", str(patch)), cwd=reference, text=True
            )
            for line in numstat.splitlines():
                fields = line.split("\t", 2)
                if len(fields) != 3 or " => " in fields[2]:
                    raise ValueError(f"unsupported source patch path record: {line!r}")
                touched.add(fields[2])
        with tempfile.TemporaryDirectory(prefix="laguna-llamacpp-source-patch-") as raw:
            staging = Path(raw)
            for relative in sorted(touched):
                if relative not in expected:
                    raise ValueError(f"source patch touches unknown path: {relative}")
                mode, object_id = expected[relative]
                payload = subprocess.check_output(
                    ("git", "cat-file", "blob", object_id), cwd=reference
                )
                path = staging / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if mode == "120000":
                    path.symlink_to(payload.decode("utf-8", errors="surrogateescape"))
                else:
                    path.write_bytes(payload)
                    if mode == "100755":
                        path.chmod(0o755)
            for patch in declared_patches:
                subprocess.run(
                    ("git", "apply", "--check", "--unsafe-paths", str(patch)),
                    cwd=staging,
                    check=True,
                )
                subprocess.run(
                    ("git", "apply", "--unsafe-paths", str(patch)),
                    cwd=staging,
                    check=True,
                )
            for relative in sorted(touched):
                mode, _object_id = expected[relative]
                path = staging / relative
                if mode == "120000":
                    payload = os.readlink(path).encode(
                        "utf-8", errors="surrogateescape"
                    )
                else:
                    payload = path.read_bytes()
                expected[relative] = (mode, _git_blob_sha1(payload))

    mismatches: list[str] = []
    for relative, (mode, object_id) in expected.items():
        path = source / relative
        if mode == "120000":
            if not path.is_symlink():
                mismatches.append(f"{relative}: expected symlink")
                continue
            payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
        else:
            if not path.is_file() or path.is_symlink():
                mismatches.append(f"{relative}: expected regular file")
                continue
            payload = path.read_bytes()
        if _git_blob_sha1(payload) != object_id:
            mismatches.append(f"{relative}: blob differs")

    actual: set[str] = set()
    for directory, dirnames, filenames in os.walk(source, followlinks=False):
        root = Path(directory)
        for name in tuple(dirnames):
            path = root / name
            if path.is_symlink():
                actual.add(path.relative_to(source).as_posix())
                dirnames.remove(name)
        for name in filenames:
            actual.add((root / name).relative_to(source).as_posix())
    extras = sorted(actual - set(expected))
    mismatches.extend(f"{relative}: unexpected file" for relative in extras[:20])
    if mismatches:
        detail = "; ".join(mismatches[:20])
        raise ValueError(f"source archive mismatch for {full_revision}: {detail}")
    return {
        "path": str(source),
        "reference_repo": str(reference),
        "revision": full_revision,
        "archive_matches_revision": not declared_patches,
        "archive_matches_revision_plus_patches": True,
        "patches": patch_records,
        "tracked_files": len(expected),
        "git_tree_listing_sha256": _sha256(tree),
    }


def _binary_bundle(server_bin: Path) -> dict[str, Any]:
    server = server_bin.resolve()
    bin_dir = server.parent
    libraries = sorted(
        path
        for path in bin_dir.glob("libggml-hip.so*")
        if path.is_file() and not path.is_symlink()
    )
    aggregate = hashlib.sha256()
    total_bytes = 0
    primary: list[dict[str, Any]] = []
    for path in libraries:
        digest = _sha256_file(path)
        size = path.stat().st_size
        total_bytes += size
        aggregate.update(path.name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
        if ".hipv4-" not in path.name and ".host-" not in path.name:
            primary.append({"path": str(path), "sha256": digest, "bytes": size})
    cmake_cache = server.parents[1] / "CMakeCache.txt"
    return {
        "server": {
            "path": str(server),
            "sha256": _sha256_file(server),
            "bytes": server.stat().st_size,
        },
        "hip_library_files": len(libraries),
        "hip_library_bytes": total_bytes,
        "hip_library_bundle_sha256": aggregate.hexdigest(),
        "primary_hip_libraries": primary,
        "cmake_cache": (
            {"path": str(cmake_cache), "sha256": _sha256_file(cmake_cache)}
            if cmake_cache.is_file()
            else None
        ),
    }


def _post_json(
    url: str, payload: dict[str, Any], timeout: float, *, retries: int = 1
) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    call = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(call, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            if response.headers.get_content_type() != "text/event-stream":
                return json.loads(raw)
            events = [
                json.loads(line[6:])
                for line in raw.splitlines()
                if line.startswith("data: ") and line[6:] != "[DONE]"
            ]
            if not events:
                raise RuntimeError(f"empty SSE response from {url}")
            final = dict(events[-1])
            if "timings" not in final:
                if retries > 0:
                    return _post_json(url, payload, timeout, retries=retries - 1)
                raise RuntimeError(f"SSE response ended without timings: {final}")
            final["tokens"] = [
                int(token)
                for event in events
                for token in (event.get("tokens") or ())
            ]
            final["content"] = "".join(
                str(event.get("content") or "") for event in events
            )
            return final
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _wait_for_server(
    *, host: str, port: int, timeout: float, process: subprocess.Popen
) -> float:
    started = time.perf_counter()
    deadline = started + float(timeout)
    url = f"http://{host}:{port}/health"
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup with {process.returncode}")
        try:
            with request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return time.perf_counter() - started
        except (error.URLError, TimeoutError):
            pass
        time.sleep(0.1)
    raise TimeoutError(f"llama-server did not become ready within {timeout:.1f}s")


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


def _completion_payload(token_ids: tuple[int, ...], horizon: int) -> dict[str, Any]:
    return {
        "prompt": [int(token) for token in token_ids],
        "n_predict": int(horizon),
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "repeat_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "dry_multiplier": 0.0,
        "seed": 4242,
        "return_tokens": True,
        "cache_prompt": False,
        "ignore_eos": True,
        "stream": True,
        "reasoning_format": "none",
    }


def _matching_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for lhs, rhs in zip(left, right):
        if int(lhs) != int(rhs):
            break
        count += 1
    return count


def _response_row(
    *,
    prompt: dict[str, Any],
    horizon: int,
    repetition: int,
    response: dict[str, Any],
    wall_seconds: float,
    hipengine_ids: list[int],
) -> dict[str, Any]:
    timings = response.get("timings") or {}
    tokens = [int(token) for token in (response.get("tokens") or ())]
    predicted_n = int(
        timings.get("predicted_n", response.get("tokens_predicted", 0)) or 0
    )
    prompt_n = int(timings.get("prompt_n", 0) or 0)
    prompt_seconds = float(timings.get("prompt_ms", 0.0) or 0.0) / 1_000.0
    predicted_seconds = float(timings.get("predicted_ms", 0.0) or 0.0) / 1_000.0
    transitions = max(0, predicted_n - 1)
    valid_native_count = predicted_n == int(horizon)
    valid_returned_count = len(tokens) == int(horizon)
    valid_prompt_count = prompt_n == int(prompt["prompt_tokens"])
    return {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "prompt_tokens": prompt["prompt_tokens"],
        "prompt_token_ids_sha256": prompt["token_ids_sha256"],
        "horizon": int(horizon),
        "repetition": int(repetition),
        "generated_token_ids": tokens,
        "generated_ids_sha256": _sha256_json(tokens),
        "valid_token_count": valid_native_count,
        "valid_native_predicted_count": valid_native_count,
        "returned_token_array_complete": valid_returned_count,
        "valid_prompt_count": valid_prompt_count,
        "prompt_n": prompt_n,
        "prompt_seconds": prompt_seconds,
        "prompt_tok_s": prompt_n / prompt_seconds if prompt_seconds > 0 else 0.0,
        "predicted_n": predicted_n,
        "timed_decode_transitions": transitions,
        "predicted_seconds": predicted_seconds,
        "predicted_tok_s": (
            predicted_n / predicted_seconds if predicted_seconds > 0 else 0.0
        ),
        "transition_normalized_tok_s": (
            transitions / predicted_seconds if predicted_seconds > 0 else 0.0
        ),
        "wall_seconds": float(wall_seconds),
        "wall_output_tok_s": predicted_n / wall_seconds if wall_seconds > 0 else 0.0,
        "matches_hipengine": tokens == hipengine_ids,
        "matching_hipengine_prefix_tokens": _matching_prefix(tokens, hipengine_ids),
        "stop_type": response.get("stop_type"),
        "timings": timings,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("llama.cpp aggregate requires at least one row")
    prompt_tokens = sum(int(row["prompt_n"]) for row in rows)
    prompt_seconds = sum(float(row["prompt_seconds"]) for row in rows)
    predicted_tokens = sum(int(row["predicted_n"]) for row in rows)
    transitions = sum(int(row["timed_decode_transitions"]) for row in rows)
    predicted_seconds = sum(float(row["predicted_seconds"]) for row in rows)
    wall_seconds = sum(float(row["wall_seconds"]) for row in rows)
    return {
        "runs": len(rows),
        "prompt_tokens": prompt_tokens,
        "prompt_seconds": prompt_seconds,
        "prompt_tok_s": prompt_tokens / prompt_seconds,
        "predicted_tokens": predicted_tokens,
        "predicted_seconds": predicted_seconds,
        "predicted_tok_s": predicted_tokens / predicted_seconds,
        "timed_decode_transitions": transitions,
        "transition_normalized_tok_s": transitions / predicted_seconds,
        "wall_seconds": wall_seconds,
        "wall_output_tok_s": predicted_tokens / wall_seconds,
        "wall_median_seconds": statistics.median(
            float(row["wall_seconds"]) for row in rows
        ),
        "valid_token_counts": all(bool(row["valid_token_count"]) for row in rows),
        "valid_native_predicted_counts": all(
            bool(row["valid_native_predicted_count"]) for row in rows
        ),
        "returned_token_arrays_complete": all(
            bool(row["returned_token_array_complete"]) for row in rows
        ),
        "valid_prompt_counts": all(bool(row["valid_prompt_count"]) for row in rows),
        "hipengine_exact_runs": sum(bool(row["matches_hipengine"]) for row in rows),
    }


def _rollups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_horizon = {
        str(horizon): _aggregate([row for row in rows if row["horizon"] == horizon])
        for horizon in sorted({int(row["horizon"]) for row in rows})
    }
    by_category: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        by_category[category] = {
            str(horizon): _aggregate(
                [row for row in selected if row["horizon"] == horizon]
            )
            for horizon in sorted({int(row["horizon"]) for row in selected})
        }
    return {"horizons": by_horizon, "categories": by_category}


def _read_optional_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _process_rss_bytes(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _load_hipengine_reference(
    paths: list[Path],
    *,
    model_sha256: str,
    prompt_sha256: str,
    prompts: list[dict[str, Any]],
    context_length: int,
    horizons: tuple[int, ...],
    repetitions: int,
) -> tuple[dict[str, Any], dict[tuple[str, int], list[int]]]:
    selected_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for path in paths:
        payload = path.read_bytes()
        artifact = json.loads(payload)
        if not artifact.get("pass"):
            raise ValueError(f"hipEngine artifact did not pass: {path}")
        if artifact.get("model", {}).get("sha256") != model_sha256:
            raise ValueError(f"hipEngine model hash mismatch: {path}")
        protocol = artifact.get("protocol", {})
        if protocol.get("prompt_suite_sha256") != prompt_sha256:
            raise ValueError(f"hipEngine prompt-suite hash mismatch: {path}")
        if int(protocol.get("prompt_count", 0)) != len(prompts):
            raise ValueError(f"hipEngine prompt count mismatch: {path}")
        if int(protocol.get("context_length", 0)) != context_length:
            raise ValueError(f"hipEngine context length mismatch: {path}")
        if tuple(int(value) for value in protocol.get("output_horizons", ())) != horizons:
            raise ValueError(f"hipEngine horizon mismatch: {path}")
        rows = [row for row in artifact.get("prompt_runs", ()) if row.get("mode") == "bulk"]
        if not rows:
            raise ValueError(f"hipEngine artifact has no bulk prompt rows: {path}")
        selected_rows.extend(rows)
        artifacts.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(payload),
                "revision": artifact.get("repo", {}).get("revision"),
                "rows": len(rows),
            }
        )

    prompt_ids = {str(prompt["id"]) for prompt in prompts}
    if {str(row["prompt_id"]) for row in selected_rows} != prompt_ids:
        raise ValueError("hipEngine prompt IDs do not match the matched suite")
    rows_per_prompt = {
        prompt_id: sum(str(row["prompt_id"]) == prompt_id for row in selected_rows)
        for prompt_id in prompt_ids
    }
    if set(rows_per_prompt.values()) != {repetitions}:
        raise ValueError(
            f"hipEngine repetition count does not match llama.cpp: {rows_per_prompt}"
        )

    oracle: dict[tuple[str, int], list[int]] = {}
    rates: dict[str, Any] = {}
    for horizon in horizons:
        checkpoints: list[dict[str, Any]] = []
        for row in selected_rows:
            checkpoint = row["checkpoints"][str(horizon)]
            checkpoints.append(checkpoint)
            key = (str(row["prompt_id"]), int(horizon))
            ids = [int(token) for token in checkpoint["generated_token_ids"]]
            prior = oracle.setdefault(key, ids)
            if ids != prior:
                raise ValueError(f"hipEngine generated IDs are not deterministic for {key}")
        transitions = sum(int(item["decode_forward_calls"]) for item in checkpoints)
        seconds = sum(float(item["decode_seconds"]) for item in checkpoints)
        rates[str(horizon)] = {
            "runs": len(checkpoints),
            "timed_decode_transitions": transitions,
            "decode_seconds": seconds,
            "transition_tok_s": transitions / seconds,
        }
    return (
        {
            "artifacts": artifacts,
            "repetitions": repetitions,
            "prompt_rows": len(selected_rows),
            "horizons": rates,
            "generated_ids_deterministic": True,
        },
        oracle,
    )


def _build_server_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.server_bin.resolve()),
        "-m",
        str(args.model.resolve()),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "-c",
        str(args.context_length),
        "-ngl",
        str(args.gpu_layers),
        "-fa",
        args.flash_attention,
        "-ctk",
        args.cache_type_k,
        "-ctv",
        args.cache_type_v,
        "--parallel",
        "1",
        "--no-warmup",
        "--metrics",
    ]
    command.append("--mmap" if args.mmap else "--no-mmap")
    if not args.repack:
        command.append("--no-repack")
    if args.skip_chat_parsing:
        command.append("--skip-chat-parsing")
    return command


def _resolved_source(args: argparse.Namespace) -> tuple[Path, str]:
    if args.source_dir is not None:
        source = args.source_dir
    elif args.server_bin.resolve() == DEFAULT_SERVER.resolve():
        source = DEFAULT_SOURCE
    else:
        source = args.server_bin.resolve().parents[2]
    if args.source_revision:
        revision = args.source_revision
    elif source.resolve() == DEFAULT_SOURCE.resolve():
        revision = DEFAULT_SOURCE_REVISION
    elif (source / ".git").exists():
        revision = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=source, text=True
        ).strip()
    else:
        raise ValueError("--source-revision is required for a non-Git source archive")
    return source.resolve(), revision


def _server_version(server: Path, env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        (str(server.resolve()), "--version"),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30.0,
        check=False,
    )
    return {
        "command": [str(server.resolve()), "--version"],
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def _deterministic_rows(rows: list[dict[str, Any]]) -> bool:
    groups: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        key = (str(row["prompt_id"]), int(row["horizon"]))
        groups.setdefault(key, set()).add(str(row["generated_ids_sha256"]))
    return all(len(values) == 1 for values in groups.values())


def run(args: argparse.Namespace) -> dict[str, Any]:
    horizons = tuple(sorted({int(value) for value in args.output_horizons}))
    if horizons != RETAINED_HORIZONS:
        raise ValueError(f"matched Laguna horizons must be {RETAINED_HORIZONS}")
    if args.repetitions < 2:
        raise ValueError("matched Laguna comparison requires at least two repetitions")
    if not args.server_bin.is_file() or not os.access(args.server_bin, os.X_OK):
        raise FileNotFoundError(f"llama-server is not executable: {args.server_bin}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if args.cache_type_k != "bf16" or args.cache_type_v != "bf16":
        raise ValueError("matched HIP comparison requires BF16 K and V storage")
    if args.flash_attention != "on":
        raise ValueError("matched HIP comparison requires FlashAttention on")
    if args.mmap or not args.repack or not args.skip_chat_parsing:
        raise ValueError(
            "matched HIP comparison requires no mmap, default repack, and skip-chat-parsing"
        )

    harness_repo = _git_state(ROOT)
    if not harness_repo["tracked_clean"] and not args.allow_dirty_harness:
        raise RuntimeError("matched benchmark requires a clean tracked hipEngine tree")

    prompt_sha256 = _sha256_file(args.prompts)
    if prompt_sha256 != MATCHED_PROMPT_SHA256:
        raise ValueError(
            f"matched prompt hash mismatch: {prompt_sha256} != {MATCHED_PROMPT_SHA256}"
        )
    model_sha256 = _sha256_file(args.model)
    if model_sha256 != args.model_sha256:
        raise ValueError(f"model SHA-256 mismatch: {model_sha256} != {args.model_sha256}")

    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    if len(prompts) != EXPECTED_PROMPT_COUNT or {
        prompt["category"] for prompt in prompts
    } != EXPECTED_CATEGORIES:
        raise ValueError("matched comparison requires all 18 train+heldout prompts")
    if max(prompt["prompt_tokens"] for prompt in prompts) + max(horizons) - 1 > args.context_length:
        raise ValueError("prompt/output shape exceeds matched context length")

    hipengine_paths = args.hipengine_artifact or [DEFAULT_HIPENGINE_ARTIFACT]
    hipengine, oracle = _load_hipengine_reference(
        hipengine_paths,
        model_sha256=model_sha256,
        prompt_sha256=prompt_sha256,
        prompts=prompts,
        context_length=args.context_length,
        horizons=horizons,
        repetitions=args.repetitions,
    )
    source_dir, revision = _resolved_source(args)
    source = _verify_source_archive(
        source_dir,
        args.reference_repo,
        revision,
        patches=args.source_patch,
    )
    bundle = _binary_bundle(args.server_bin)

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = args.hip_visible_devices
    env["GPU_MAX_HW_QUEUES"] = args.gpu_max_hw_queues
    version = _server_version(args.server_bin, env)
    if version["returncode"] != 0:
        raise RuntimeError(f"llama-server --version failed: {version['output']}")

    server_command = _build_server_command(args)
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    gtt_samples: list[int] = []
    rss_samples: list[int] = []
    with args.server_log.open("wb") as log:
        process = subprocess.Popen(
            server_command,
            cwd=source_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        startup_seconds = _wait_for_server(
            host=args.host,
            port=args.port,
            timeout=args.startup_timeout,
            process=process,
        )
        ready_gtt = _read_optional_int(args.gtt_path)
        ready_rss = _process_rss_bytes(process.pid)
        if ready_gtt is not None:
            gtt_samples.append(ready_gtt)
        if ready_rss is not None:
            rss_samples.append(ready_rss)
        _post_json(
            f"http://{args.host}:{args.port}/completion",
            _completion_payload(prompts[0]["token_ids"], max(horizons)),
            args.request_timeout,
        )
        for repetition in range(args.repetitions):
            for prompt_index, prompt in enumerate(prompts):
                order = horizons if (repetition + prompt_index) % 2 == 0 else horizons[::-1]
                for horizon in order:
                    started = time.perf_counter()
                    response = _post_json(
                        f"http://{args.host}:{args.port}/completion",
                        _completion_payload(prompt["token_ids"], horizon),
                        args.request_timeout,
                    )
                    wall_seconds = time.perf_counter() - started
                    row = _response_row(
                        prompt=prompt,
                        horizon=horizon,
                        repetition=repetition,
                        response=response,
                        wall_seconds=wall_seconds,
                        hipengine_ids=oracle[(str(prompt["id"]), int(horizon))],
                    )
                    rows.append(row)
                    gtt = _read_optional_int(args.gtt_path)
                    rss = _process_rss_bytes(process.pid)
                    if gtt is not None:
                        gtt_samples.append(gtt)
                    if rss is not None:
                        rss_samples.append(rss)
                    print(
                        f"rep={repetition} prompt={prompt['id']} h={horizon} "
                        f"prefill={row['prompt_tok_s']:.2f} tok/s "
                        f"transitions={row['transition_normalized_tok_s']:.2f} tok/s "
                        f"match={row['matches_hipengine']}",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        _terminate(process)

    aggregate = _rollups(rows)
    deterministic = _deterministic_rows(rows)
    expected_rows = len(prompts) * len(horizons) * args.repetitions
    timing_rows_valid = all(
        row["valid_native_predicted_count"]
        and row["valid_prompt_count"]
        and row["predicted_seconds"] > 0.0
        for row in rows
    )
    returned_arrays_complete = all(row["returned_token_array_complete"] for row in rows)
    comparisons: dict[str, Any] = {}
    for horizon in horizons:
        key = str(horizon)
        llama = aggregate["horizons"][key]
        hip = hipengine["horizons"][key]
        if llama["timed_decode_transitions"] != hip["timed_decode_transitions"]:
            raise ValueError(f"transition count mismatch at h{horizon}")
        llama_rate = float(llama["transition_normalized_tok_s"])
        hip_rate = float(hip["transition_tok_s"])
        comparisons[key] = {
            "timed_decode_transitions_each": llama["timed_decode_transitions"],
            "hipengine_tok_s": hip_rate,
            "llamacpp_hip_tok_s": llama_rate,
            "hipengine_vs_llamacpp_pct": (hip_rate / llama_rate - 1.0) * 100.0,
            "faster_engine": "hipEngine" if hip_rate > llama_rate else "llama.cpp HIP",
        }

    eligible = bool(
        harness_repo["tracked_clean"]
        and source["archive_matches_revision_plus_patches"]
        and model_sha256 == args.model_sha256
        and prompt_sha256 == MATCHED_PROMPT_SHA256
        and len(rows) == expected_rows
        and timing_rows_valid
        and args.cache_type_k == args.cache_type_v == "bf16"
        and args.flash_attention == "on"
        and args.hip_visible_devices == "0"
        and args.gpu_max_hw_queues == "1"
    )
    server_log_sha256 = _sha256_file(args.server_log)
    created_at = datetime.now(timezone.utc).isoformat()  # noqa: UP017 - Python 3.10
    return {
        "schema": 1,
        "created_at": created_at,
        "kind": "llamacpp_hip_laguna_matched_post_ttft_ar",
        "status": "accepted" if eligible else "rejected",
        "pass": eligible,
        "performance_claim": eligible,
        "performance_claim_scope": (
            "same W7900, model bytes, 18 prompt token streams, natural greedy sampling, "
            "BF16 K/V, FA on, context 4096, h16/h32, and post-TTFT transition count"
        ),
        "comparison": comparisons,
        "comparison_eligibility": {
            "eligible": eligible,
            "primary_metric": "sum(predicted_n - 1) / sum(predicted_ms)",
            "same_model_sha256": True,
            "same_prompt_token_streams": True,
            "same_prompt_and_horizon_run_counts": len(rows) == expected_rows,
            "same_timed_transition_counts": all(
                comparisons[str(horizon)]["timed_decode_transitions_each"]
                == hipengine["horizons"][str(horizon)]["timed_decode_transitions"]
                for horizon in horizons
            ),
            "same_kv_storage_dtype": True,
            "same_flash_attention_policy": True,
            "same_context_length": True,
            "same_natural_greedy_sampling": True,
            "llamacpp_native_timing_rows_valid": timing_rows_valid,
            "returned_token_arrays_complete": returned_arrays_complete,
            "llamacpp_repeat_deterministic_for_returned_arrays": deterministic,
            "generated_ids_match_reported_not_required": True,
            "native_timing_is_authoritative_when_sse_token_arrays_are_incomplete": True,
            "remaining_engine_difference": (
                "hipEngine uses KVLiveSpans and its own exact kernels; llama.cpp uses ggml "
                "graph/KV scheduling. The protocol and storage dtype match, not arithmetic order."
            ),
        },
        "model": {
            "path": str(args.model.resolve()),
            "sha256": model_sha256,
            "quant": "UD-Q2_K_XL",
            "bytes": args.model.stat().st_size,
        },
        "prompt_suite": {
            "path": str(args.prompts.resolve()),
            "sha256": prompt_sha256,
            "count": len(prompts),
            "categories": sorted(EXPECTED_CATEGORIES),
            "prompt_tokens_min": min(prompt["prompt_tokens"] for prompt in prompts),
            "prompt_tokens_max": max(prompt["prompt_tokens"] for prompt in prompts),
        },
        "protocol": {
            "context_length": args.context_length,
            "horizons": list(horizons),
            "repetitions": args.repetitions,
            "runs_per_horizon": len(prompts) * args.repetitions,
            "warmups": 1,
            "sampling": "natural greedy; temperature=0; neutral penalties; seed=4242",
            "cache_prompt": False,
            "ignore_eos": True,
            "flash_attention": args.flash_attention,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "mmap": args.mmap,
            "repack": args.repack,
            "timing_scope": (
                "primary post-TTFT synchronized transitions; model load, prompt prefill, "
                "first sampled token, HTTP wall, and tokenization excluded"
            ),
            "order": "alternating h16/h32 by repetition plus prompt index",
        },
        "hardware": {
            "HIP_VISIBLE_DEVICES": args.hip_visible_devices,
            "GPU_MAX_HW_QUEUES": args.gpu_max_hw_queues,
            "gtt_path": str(args.gtt_path),
            "gtt_peak_sampled_bytes": max(gtt_samples) if gtt_samples else None,
            "server_rss_peak_sampled_bytes": max(rss_samples) if rss_samples else None,
        },
        "harness_repo": harness_repo,
        "hipengine": hipengine,
        "llamacpp": {
            "source": source,
            "build": bundle,
            "version": version,
            "command": server_command,
            "environment": {
                "HIP_VISIBLE_DEVICES": args.hip_visible_devices,
                "GPU_MAX_HW_QUEUES": args.gpu_max_hw_queues,
            },
            "startup_seconds": startup_seconds,
            "server_log": str(args.server_log.resolve()),
            "server_log_sha256": server_log_sha256,
            "aggregate": aggregate,
            "rows": rows,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
