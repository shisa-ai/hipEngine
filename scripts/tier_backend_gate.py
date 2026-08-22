#!/usr/bin/env python3
# ruff: noqa: E402
"""Run optional cold-tier lifecycle and restore-vs-recompute host economics."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hipengine.kvcache import (
    ColdTierStore,
    DenseKVCacheBackend,
    KVTCColdCodec,
    TieredKVCacheBackend,
    evaluate_restore_economics,
)


def _request(request_id: int):
    return SimpleNamespace(
        request_id=int(request_id),
        prompt_tokens=tuple(range(64)),
        max_new_tokens=8,
    )


def _lease(backend: DenseKVCacheBackend, request_id: int):
    request = _request(request_id)
    return backend.reserve(backend.estimate(request, None, {}))


def _measure_economics(
    codec: KVTCColdCodec,
    key,
    encoded: bytes,
    payload: bytes,
    *,
    repetitions: int,
    recompute_rounds: int,
) -> dict[str, Any]:
    restore_samples: list[float] = []
    recompute_samples: list[float] = []
    for _ in range(int(repetitions)):
        started = time.perf_counter()
        restored = codec.decode(key, encoded)
        restore_samples.append(time.perf_counter() - started)
        assert restored == payload

        started = time.perf_counter()
        digest = b""
        for round_index in range(int(recompute_rounds)):
            digest = hashlib.sha256(
                payload + round_index.to_bytes(4, "little") + digest
            ).digest()
        if not digest:
            raise AssertionError("recompute digest was not produced")
        recompute_samples.append(time.perf_counter() - started)
    restore_median = statistics.median(restore_samples)
    recompute_median = statistics.median(recompute_samples)
    economics = evaluate_restore_economics(
        restore_seconds=restore_median,
        recompute_seconds=recompute_median,
    )
    return {
        "restore_seconds": restore_samples,
        "recompute_seconds": recompute_samples,
        "restore_median_seconds": restore_median,
        "recompute_median_seconds": recompute_median,
        "savings_median_seconds": economics.savings_seconds,
        "use_restore": economics.use_restore,
        "scope": "synthetic host TTFT proxy; not a model performance claim",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = (b"hipengine-cold-kv-state\0" * ((int(args.payload_bytes) // 24) + 1))[
        : int(args.payload_bytes)
    ]
    with tempfile.TemporaryDirectory(prefix="hipengine-tier-gate-") as directory:
        hot = DenseKVCacheBackend(
            codec="bf16",
            page_capacity=256,
            block_size=16,
            artifact_fingerprint="fixture:tier-hot",
        )
        store = ColdTierStore(
            host_capacity_bytes=max(int(args.payload_bytes), 4096),
            nvme_capacity_bytes=max(int(args.payload_bytes) * 2, 8192),
            nvme_directory=Path(directory) / "nvme",
            tenant_quota_bytes={"gate": max(int(args.payload_bytes) * 2, 8192)},
        )
        codec = KVTCColdCodec(level=int(args.compression_level))
        tiered = TieredKVCacheBackend(
            hot,
            store=store,
            codec=codec,
            transfer_workspace_bytes=max(int(args.payload_bytes) * 2, 4096),
            maintenance_budget_bytes=max(int(args.payload_bytes) * 2, 4096),
        )
        lease = _lease(hot, 1)
        key = tiered.cold_key_for_tokens(
            token_ids=tuple(range(64)),
            request_scope="gate-request",
            state_fingerprint="gate-state-v1",
        )
        encoded_result = codec.encode(key, payload)
        economics = _measure_economics(
            codec,
            key,
            encoded_result.encoded,
            payload,
            repetitions=int(args.repetitions),
            recompute_rounds=int(args.recompute_rounds),
        )
        tiered.enqueue_offload(
            key=key,
            lease=lease,
            hot_payload=payload,
            tenant_id="gate",
        )
        offload = tiered.maintenance()
        restored: list[bytes] = []
        tiered.enqueue_restore(
            key=key,
            request=_request(2),
            restore_callback=lambda _lease, data: restored.append(data),
        )
        restore = tiered.maintenance()
        restored_lease = restore[0].lease if restore else None
        if restored_lease is not None:
            hot.reclaim(restored_lease)
        tiered.assert_conserved()
        before_drain = tiered.observability_snapshot()
        tiered.drain()
        final = tiered.observability_snapshot()
        passed = bool(
            offload
            and offload[0].passed
            and restore
            and restore[0].passed
            and restored == [payload]
            and final["tier"]["store"]["object_count"] == 0
            and final["tier"]["ledger"]["owners"] == {}
        )
    return {
        "schema": 1,
        "kind": "hipengine_optional_cold_tier_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted_host_tier" if passed else "failed",
        "passed": passed,
        "performance_claim": False,
        "hot_backend": {
            "topology": hot.spec.topology_key,
            "hot_codec": hot.spec.hot_codec_key,
            "tier_key": tiered.spec.tier_key,
            "attention_storage_view_remains_hot": True,
        },
        "cold_codec": {
            "name": codec.name,
            "original_bytes": encoded_result.original_bytes,
            "encoded_bytes": encoded_result.encoded_bytes,
            "compression_ratio": (
                encoded_result.original_bytes / encoded_result.encoded_bytes
            ),
        },
        "economics": economics,
        "offload": [result.passed for result in offload],
        "restore": [result.passed for result in restore],
        "before_drain": before_drain["tier"],
        "final": final["tier"],
        "limitations": [
            "Host/NVMe and recompute timings are synthetic host proxies.",
            "Cold bytes always restore into the resolved hot backend before attention.",
            "No product default or model TTFT claim is made.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-bytes", type=int, default=1 << 20)
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--recompute-rounds", type=int, default=16)
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.payload_bytes, args.repetitions, args.recompute_rounds) <= 0:
        raise ValueError("payload/repetition/recompute values must be positive")
    payload = run(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
