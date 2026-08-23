#!/usr/bin/env python3
# ruff: noqa: E402
"""Run the torch-free compact-DMS metadata, lifecycle, and codec gate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from hipengine.kvcache import (
    DMSCodecQualification,
    create_dms_bf16_backend,
    create_dms_int8_backend,
    load_dms_retrofit_config,
)


def _qualification(path: Path, artifact: str) -> DMSCodecQualification:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DMSCodecQualification(
        codec="int8_per_token_head",
        artifact_fingerprint=str(payload.get("artifact_fingerprint", artifact)),
        kl_divergence=float(payload["kl_divergence"]),
        top1_agreement=float(payload["top1_agreement"]),
        no_dense_shadow=bool(payload["no_dense_shadow"]),
        evidence_source=str(payload["evidence_source"]),
    )


def _request(request_id: int, prompt_tokens: int, decode_tokens: int):
    return SimpleNamespace(
        request_id=int(request_id),
        prompt_tokens=tuple(range(int(prompt_tokens))),
        max_new_tokens=int(decode_tokens),
    )


def _arrays(config, tokens: int, *, seed: int):
    rng = np.random.default_rng(seed)
    shape = (
        int(tokens),
        config.num_layers,
        config.num_kv_heads,
        config.head_dim,
    )
    k = rng.normal(0.0, 0.2, size=shape).astype(np.float32)
    v = rng.normal(0.0, 0.2, size=shape).astype(np.float32)
    evict = np.ones(shape[:3], dtype=np.bool_)
    evict[:: config.target_compression_ratio] = False
    return k, v, evict


def _backend(args: argparse.Namespace, config, *, max_rows: int | None = None):
    common = dict(
        retrofit=config,
        slots_per_layer=int(args.slots_per_layer),
        max_request_rows=int(args.max_requests if max_rows is None else max_rows),
        max_pack_rows=int(args.max_pack_rows),
        physical_widths=(1, 2, 4, 8),
    )
    if args.codec == "bf16":
        return create_dms_bf16_backend(**common)
    if args.codec_qualification is None:
        raise ValueError("INT8 DMS gate requires --codec-qualification")
    return create_dms_int8_backend(
        codec_qualification=_qualification(
            args.codec_qualification,
            config.artifact_fingerprint,
        ),
        **common,
    )


def _run_width(config, args: argparse.Namespace, width: int) -> dict[str, Any]:
    backend = _backend(args, config, max_rows=width)
    leases = []
    for request_id in range(width):
        request = _request(
            request_id,
            int(args.prompt_tokens),
            int(args.decode_tokens),
        )
        lease = backend.reserve(backend.estimate(request, None, {}))
        k, v, evict = _arrays(
            config,
            int(args.prompt_tokens),
            seed=1000 + request_id,
        )
        backend.streaming_pack(request_id, k, v, evict)
        leases.append(lease)
    active = backend.observability_snapshot()
    for lease in leases[::2]:
        backend.reclaim(lease)
    for request_id in range(width, width + len(leases[::2])):
        request = _request(
            request_id,
            int(args.prompt_tokens),
            int(args.decode_tokens),
        )
        replacement = backend.reserve(backend.estimate(request, None, {}))
        k, v, evict = _arrays(
            config,
            int(args.prompt_tokens),
            seed=1000 + request_id,
        )
        backend.streaming_pack(request_id, k, v, evict)
        leases.append(replacement)
    for lease in leases:
        if backend.has_request(lease.request_id):
            backend.reclaim(lease)
    backend.assert_conserved()
    final = backend.observability_snapshot()
    return {
        "width": int(width),
        "passed": bool(
            active["extent_pool"]["owner_count"] == width
            and final["extent_pool"]["owner_count"] == 0
            and final["extent_pool"]["free_slots"]
            == final["extent_pool"]["capacity_slots"]
        ),
        "active": active,
        "final": final,
    }


def _pressure_gate(config, args: argparse.Namespace) -> dict[str, Any]:
    prompt_tokens = int(args.prompt_tokens)
    # Admission is correctness-first provisional capacity. Streaming pack
    # shrinks to protected + compressed history only after actual decisions.
    per_head = prompt_tokens + int(args.decode_tokens)
    capacity = config.num_kv_heads * per_head
    pressure_args = argparse.Namespace(**vars(args))
    pressure_args.slots_per_layer = capacity
    pressure_args.max_requests = 2
    backend = _backend(pressure_args, config)
    first = _request(0, int(args.prompt_tokens), int(args.decode_tokens))
    lease = backend.reserve(backend.estimate(first, None, {}))
    rejected = False
    try:
        second = _request(1, int(args.prompt_tokens), int(args.decode_tokens))
        backend.reserve(backend.estimate(second, None, {}))
    except (MemoryError, RuntimeError):
        rejected = True
    backend.reclaim(lease)
    backend.assert_conserved()
    final = backend.observability_snapshot()
    return {
        "passed": bool(
            rejected
            and final["extent_pool"]["owner_count"] == 0
            and (
                final["extent_pool"]["allocation_failures"] >= 1
                or final["ledger"]["stats"]["rejections"] >= 1
            )
        ),
        "retryable_rejection_observed": rejected,
        "final": final,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        config = load_dms_retrofit_config(
            args.model,
            metadata_path=args.metadata,
            expected_artifact_fingerprint=args.expected_artifact,
            allow_training_log_fallback=bool(args.allow_training_log_fallback),
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        return {
            "schema": 1,
            "kind": "hipengine_dms_backend_gate",
            "created_at": created_at,
            "status": "blocked_metadata",
            "passed": False,
            "performance_claim": False,
            "model": str(args.model),
            "metadata": None if args.metadata is None else str(args.metadata),
            "blocker": f"{type(exc).__name__}: {exc}",
        }
    try:
        rows = [_run_width(config, args, width) for width in (1, 2, 4, 8, 16, 32)]
        pressure = _pressure_gate(config, args)
        passed = bool(all(row["passed"] for row in rows) and pressure["passed"])
        blocker = None
    except (MemoryError, RuntimeError, ValueError) as exc:
        rows = []
        pressure = {}
        passed = False
        blocker = f"{type(exc).__name__}: {exc}"
    return {
        "schema": 1,
        "kind": "hipengine_dms_backend_gate",
        "created_at": created_at,
        "status": "accepted_host_backend" if passed else "blocked_backend",
        "passed": passed,
        "performance_claim": False,
        "model": str(args.model),
        "codec": str(args.codec),
        "metadata": asdict(config),
        "retrofit_fingerprint": config.fingerprint,
        "widths": rows,
        "pressure": pressure,
        "prefix_mode": "off",
        "no_dense_shadow": True,
        "blocker": blocker,
        "limitations": [
            "This is a torch-free CPU/reference and host-lifecycle gate.",
            "HIP streaming-pack and compact-attention promotion requires a fresh stable GPU generation.",
            "Quality is scoped to the exact metadata and optional codec qualification artifacts.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--expected-artifact")
    parser.add_argument("--allow-training-log-fallback", action="store_true")
    parser.add_argument("--codec", choices=("bf16", "int8_per_token_head"), default="bf16")
    parser.add_argument("--codec-qualification", type=Path)
    parser.add_argument("--slots-per-layer", type=int, default=4096)
    parser.add_argument("--max-requests", type=int, default=32)
    parser.add_argument("--max-pack-rows", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
