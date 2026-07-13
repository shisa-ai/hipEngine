#!/usr/bin/env python3
# ruff: noqa: E402
"""Compare hipEngine GGUF BF16-KV logits with llama.cpp F16-KV logits.

Both sides consume the same Q4_K_M GGUF model, repeated-token prompt, and
llama.cpp-reference teacher tokens. The llama.cpp reference is a compact binary
export from ``llamacpp_kv_matched_context.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.benchmark.provenance import collect_artifact_provenance

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
_MAGIC = b"HKVLOG1\0"
_HEADER = struct.Struct("<8sIII")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_logits(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise ValueError("reference logits header is truncated")
        magic, schema, rows, columns = _HEADER.unpack(header)
        if magic != _MAGIC:
            raise ValueError("reference logits magic does not match")
        if schema != 1:
            raise ValueError(f"unsupported reference logits schema: {schema}")
        if rows < 1 or columns < 1:
            raise ValueError("reference logits shape must be non-empty")
        expected_bytes = int(rows) * int(columns) * np.dtype("<f4").itemsize
        data = handle.read()
    if len(data) != expected_bytes:
        raise ValueError(f"reference logits payload size mismatch: {len(data)} != {expected_bytes}")
    logits = np.frombuffer(data, dtype="<f4").reshape(int(rows), int(columns)).copy()
    if not np.all(np.isfinite(logits)):
        raise ValueError("reference logits contain non-finite values")
    return logits


def compare_logit_rows(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    kl_threshold: float = 0.05,
    top1_threshold: float = 0.90,
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    aggregate = evaluate_logits(
        reference,
        candidate,
        kl_threshold=kl_threshold,
        top1_threshold=top1_threshold,
    )
    reference_top1 = np.argmax(reference, axis=1).astype(np.int64)
    candidate_top1 = np.argmax(candidate, axis=1).astype(np.int64)
    top1_matches = reference_top1 == candidate_top1
    per_position_kl = [
        evaluate_logits(reference[index], candidate[index]).kl_mean
        for index in range(reference.shape[0])
    ]
    mismatch_indices = np.flatnonzero(~top1_matches)
    first_mismatch = None
    if mismatch_indices.size:
        index = int(mismatch_indices[0])
        first_mismatch = {
            "index": index,
            "reference": int(reference_top1[index]),
            "candidate": int(candidate_top1[index]),
        }
    reference_ranks = []
    for index, token_id in enumerate(reference_top1.tolist()):
        reference_ranks.append(int(1 + np.count_nonzero(candidate[index] > candidate[index, token_id])))
    return {
        "mean_kl": aggregate.kl_mean,
        "max_kl": aggregate.kl_max,
        "top1_agreement": aggregate.top1_agreement,
        "passed": aggregate.passed,
        "kl": per_position_kl,
        "reference_top1": reference_top1.tolist(),
        "candidate_top1": candidate_top1.tolist(),
        "top1_matches": top1_matches.tolist(),
        "candidate_reference_top1_rank": reference_ranks,
        "first_top1_mismatch": first_mismatch,
    }


def _load_llama_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "llamacpp_kv_matched_context_driver":
        raise ValueError("llama.cpp metadata has unexpected mode")
    result = payload.get("result", {})
    if result.get("mode") != "llamacpp_kv_matched_context":
        raise ValueError("llama.cpp metadata is missing the C++ matched-context result")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    llama_payload = _load_llama_metadata(args.llama_reference_json)
    llama_result = llama_payload["result"]
    reference_logits = load_reference_logits(args.llama_reference_logits)
    expected_shape = (int(llama_result["positions"]), int(llama_result["reference"]["n_vocab"]))
    if reference_logits.shape != expected_shape:
        raise ValueError(f"reference logits shape mismatch: {reference_logits.shape} != {expected_shape}")
    reference_top1 = np.argmax(reference_logits, axis=1).astype(np.int64).tolist()
    recorded_top1 = [int(token) for token in llama_result["reference"]["top1_ids"]]
    if reference_top1 != recorded_top1:
        raise ValueError("reference logit argmax does not match llama.cpp metadata")
    if Path(llama_result["model"]).resolve() != args.model.resolve():
        raise ValueError("llama.cpp and hipEngine model paths differ")
    if int(llama_result["prompt_length"]) != args.prompt_length:
        raise ValueError("llama.cpp and hipEngine prompt lengths differ")
    if int(llama_result["decode_steps"]) != args.decode_steps:
        raise ValueError("llama.cpp and hipEngine decode lengths differ")
    if int(llama_result["prompt_token_id"]) != args.prompt_token_id:
        raise ValueError("llama.cpp and hipEngine prompt token IDs differ")

    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8")
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())

    from hipengine.core.dtype import DType
    from hipengine.kvcache.policy import FixedPagedKVPolicy
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    prompt = [int(args.prompt_token_id)] * int(args.prompt_length)
    teacher_inputs = recorded_top1[: int(args.decode_steps)]
    candidate_rows: list[np.ndarray] = []
    candidate_top1: list[int] = []
    start = time.perf_counter()
    with Qwen35GGUFResidentSession(
        args.model,
        backend=args.backend,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=int(args.max_sequence_length or args.prompt_length + args.decode_steps + 1),
        use_wmma_prefill=True,
        use_gemv_decode=True,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        prefill_start = time.perf_counter()
        first = session.prefill(prompt, use_bulk=True, bulk_attention_mode="bulk", return_logits=True)
        prefill_seconds = time.perf_counter() - prefill_start
        candidate_rows.append(np.asarray(first.logits, dtype=np.float32).copy())
        candidate_top1.append(int(first.token_id))
        decode_start = time.perf_counter()
        for token_id in teacher_inputs:
            current = session.step(int(token_id), return_logits=True)
            candidate_rows.append(np.asarray(current.logits, dtype=np.float32).copy())
            candidate_top1.append(int(current.token_id))
        decode_seconds = time.perf_counter() - decode_start
        effective_kv = session.kv_storage_dtype.value
    total_seconds = time.perf_counter() - start
    candidate_logits = np.vstack(candidate_rows).astype(np.float32, copy=False)
    comparison = compare_logit_rows(
        reference_logits,
        candidate_logits,
        kl_threshold=args.kl_threshold,
        top1_threshold=args.top1_threshold,
    )
    if candidate_top1 != comparison["candidate_top1"]:
        raise ValueError("hipEngine sampled token IDs do not match captured-logit argmax")

    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch="gfx1100",
        model_path=args.model,
        quant="Q4_K_M",
        kv_dtype="bf16",
        command=["python3", "scripts/gguf_llamacpp_matched_context.py", *sys.argv[1:]],
        timing_protocol="diagnostic wall; quality claim only",
        warmups=0,
        repetitions=1,
    )
    return {
        "schema": 1,
        "kind": "w7900_gguf_llamacpp_matched_context",
        "status": "accepted" if comparison["passed"] else "rejected_correctness",
        "performance_claim": False,
        "date": "2026-07-13",
        "protocol": {
            "model": str(args.model.resolve()),
            "model_weights_identical": True,
            "prompt_source": f"token {args.prompt_token_id} repeated",
            "prompt_tokens": args.prompt_length,
            "decode_steps": args.decode_steps,
            "positions": args.decode_steps + 1,
            "teacher": "llama.cpp F16-KV reference top-1 history",
            "reference": "llama.cpp F16 K/V cache",
            "candidate": "hipEngine BF16 K/V cache",
            "quality_thresholds": {
                "kl_mean_max": args.kl_threshold,
                "top1_agreement_min": args.top1_threshold,
            },
        },
        "comparison": comparison,
        "hipengine": {
            "effective_kv_storage": effective_kv,
            "candidate_top1": candidate_top1,
            "teacher_inputs": teacher_inputs,
            "finite_logits": bool(np.all(np.isfinite(candidate_logits))),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
        },
        "llama_cpp_reference": {
            "json_path": str(args.llama_reference_json),
            "json_sha256": _sha256_file(args.llama_reference_json),
            "logits_path": str(args.llama_reference_logits),
            "logits_sha256": _sha256_file(args.llama_reference_logits),
            "logits_size_bytes": args.llama_reference_logits.stat().st_size,
            "build": llama_payload["build"],
            "source": llama_payload["provenance"]["llama_cpp"],
            "libraries": llama_payload["provenance"]["libraries"],
        },
        "provenance": provenance,
        "notes": [
            "This same-weight bridge measures cross-engine numerical parity; it does not isolate FP16-vs-BF16 KV from other implementation arithmetic.",
            "All scored positions consume exactly the llama.cpp F16-reference token history.",
            "Timing fields are diagnostics only and are not an engine performance comparison.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--llama-reference-json", type=Path, required=True)
    parser.add_argument("--llama-reference-logits", type=Path, required=True)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=131072)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--max-sequence-length", type=int, default=0)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
