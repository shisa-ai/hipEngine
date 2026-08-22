#!/usr/bin/env python3
# ruff: noqa: E402
"""Measure optional cold-tier economics with actual model-produced BF16 KV.

The gate prefills a resident Qwen/PARO session, copies only the live full-
attention K/V prefix bytes to host, and uses that exact payload for cold codec,
pressure, restore, cancellation, and drain checks. Raw KV bytes remain local and
are never written to the repository. Tiering remains default-off regardless of
this gate's result.
"""

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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.kvcache import ColdTierStore, DenseKVCacheBackend, KVTCColdCodec, TieredKVCacheBackend, evaluate_restore_economics
from hipengine.kvcache import resolve_kv_policy
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_paro_bench import _prompt_tokens

DEFAULT_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")


def _request(request_id: int, prompt_tokens: int):
    return SimpleNamespace(
        request_id=int(request_id),
        prompt_tokens=tuple(range(int(prompt_tokens))),
        max_new_tokens=1,
    )


def _capture_actual_kv(session: Qwen35ParoResidentSession, tokens: int) -> tuple[bytes, dict[str, int]]:
    rows = int(tokens)
    heads = int(session.config.num_key_value_heads)
    dim = int(session.config.head_dim)
    shape = (rows, heads, dim)
    nbytes = int(np.prod(shape)) * np.dtype(np.uint16).itemsize
    chunks: list[bytes] = []
    for layer_id in sorted(session.full_caches):
        _kt, _vt, key_buf, value_buf = session.full_caches[layer_id]
        key_bits = np.empty(shape, dtype=np.uint16)
        value_bits = np.empty(shape, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(key_bits), DeviceBuffer(key_buf.ptr, nbytes), nbytes, runtime=session.runtime)
        copy_device_to_host(host_array_ptr(value_bits), DeviceBuffer(value_buf.ptr, nbytes), nbytes, runtime=session.runtime)
        chunks.extend((key_bits.tobytes(), value_bits.tobytes()))
    return b"".join(chunks), {
        "full_attention_layers": len(session.full_caches),
        "tokens": rows,
        "kv_heads": heads,
        "head_dim": dim,
        "bytes_per_layer_kv_pair": 2 * nbytes,
    }


def _prefill(session: Qwen35ParoResidentSession, prompt: list[int]) -> float:
    session.reset()
    session._resolve_prefill_config_for_length(len(prompt))
    started = time.perf_counter()
    result = session.prefill_native(prompt, sample=True)
    session.runtime.device_synchronize()
    elapsed = time.perf_counter() - started
    if result is None:
        raise RuntimeError("native prefill produced no seed")
    return elapsed


def run(args: argparse.Namespace) -> dict[str, object]:
    model = args.model.expanduser().resolve()
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8").strip()
    runner = Qwen35ParoNextTokenRunner(model, backend=args.backend)
    prompt = _prompt_tokens(model, "Hello", args.token_id, args.prompt_length)
    policy = resolve_kv_policy("bf16", block_size=256, scale_dtype="fp16")
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=args.prompt_length + 2,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        prefill_config=PrefillConfig(attn_aotriton_min_tokens=128),
        kv_policy=policy.create_policy(),
        kv_scale_dtype=policy.scale_dtype,
        kv_scale_granularity=policy.scale_granularity,
    ) as session:
        warmup_seconds = _prefill(session, prompt)
        payload, geometry = _capture_actual_kv(session, args.prompt_length)
        recompute = [_prefill(session, prompt) for _ in range(args.repetitions)]

    codec = KVTCColdCodec(level=args.compression_level)
    hot = DenseKVCacheBackend(
        codec="bf16", page_capacity=1024, block_size=16,
        artifact_fingerprint=f"model:{hashlib.sha256(str(model).encode()).hexdigest()}",
    )
    with tempfile.TemporaryDirectory(prefix="hipengine-real-tier-") as directory:
        # Encode once to size pressure pools from the real object.
        probe_key = TieredKVCacheBackend(
            hot,
            store=ColdTierStore(
                host_capacity_bytes=max(len(payload), 4096),
                nvme_capacity_bytes=max(len(payload) * 2, 8192),
                nvme_directory=Path(directory) / "probe",
            ),
            codec=codec,
            transfer_workspace_bytes=max(len(payload), 4096),
            maintenance_budget_bytes=max(len(payload), 4096),
        ).cold_key_for_tokens(
            token_ids=prompt,
            request_scope="real-kv-probe",
            state_fingerprint="prefill-bf16-v1",
        )
        encoded_probe = codec.encode(probe_key, payload)
        encoded_bytes = int(encoded_probe.encoded_bytes)
        store = ColdTierStore(
            host_capacity_bytes=max(encoded_bytes + encoded_bytes // 2, 4096),
            nvme_capacity_bytes=max(encoded_bytes * 3, 8192),
            nvme_directory=Path(directory) / "nvme",
            tenant_quota_bytes={"model": max(encoded_bytes * 3, 8192)},
        )
        tiered = TieredKVCacheBackend(
            hot, store=store, codec=codec,
            transfer_workspace_bytes=max(len(payload), 4096),
            maintenance_budget_bytes=max(len(payload), 4096),
        )
        keys = [
            tiered.cold_key_for_tokens(
                token_ids=prompt,
                request_scope=f"real-kv-{index}",
                state_fingerprint="prefill-bf16-v1",
            )
            for index in range(3)
        ]
        offload_results = []
        for index, key in enumerate(keys):
            request = _request(index + 1, args.prompt_length)
            lease = hot.reserve(hot.estimate(request, None, {}))
            tiered.enqueue_offload(key=key, lease=lease, hot_payload=payload, tenant_id="model")
            offload_results.extend(tiered.maintenance())
        pressure_snapshot = tiered.observability_snapshot()["tier"]

        restore_samples = []
        restored: list[bytes] = []
        restore_key = next(key for key in keys if store.contains(key))
        for repetition in range(args.repetitions):
            encoded = codec.encode(restore_key, payload).encoded
            started = time.perf_counter()
            decoded = codec.decode(restore_key, encoded)
            restore_samples.append(time.perf_counter() - started)
            if decoded != payload:
                raise AssertionError("real KV codec roundtrip mismatch")
        tiered.enqueue_restore(
            key=restore_key,
            request=_request(100, args.prompt_length),
            restore_callback=lambda _lease, data: restored.append(data),
        )
        restore_result = tiered.maintenance()
        if restore_result and restore_result[0].lease is not None:
            hot.reclaim(restore_result[0].lease)

        # Queue one more restore, cancel it before maintenance, and prove that
        # cancellation preserves the cold object for another request.
        remaining = next((key for key in keys if store.contains(key)), None)
        cancel_work_id = None
        cancelled: tuple[str, ...] = ()
        if remaining is not None:
            cancel_work_id = tiered.enqueue_restore(
                key=remaining,
                request=_request(101, args.prompt_length),
                restore_callback=lambda _lease, _data: None,
            )
            cancelled = tiered.cancel_request(101)
            if cancelled != (cancel_work_id,) or not store.contains(remaining):
                raise AssertionError("tier restore cancellation contract failed")
        before_drain = tiered.observability_snapshot()["tier"]
        tiered.drain()
        tiered.assert_conserved()
        final = tiered.observability_snapshot()["tier"]

    restore_median = statistics.median(restore_samples)
    recompute_median = statistics.median(recompute)
    economics = evaluate_restore_economics(
        restore_seconds=restore_median,
        recompute_seconds=recompute_median,
    )
    passed = bool(
        payload and all(result.passed for result in offload_results)
        and restore_result and restore_result[0].passed
        and restored == [payload]
        and final["store"]["object_count"] == 0
        and final["ledger"]["owners"] == {}
    )
    return {
        "schema": 1,
        "kind": "hipengine_real_model_kv_tier_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted_real_model_tier" if passed else "failed",
        "passed": passed,
        "performance_claim": False,
        "model": str(model),
        "prompt_length": args.prompt_length,
        "prompt_sha256": hashlib.sha256(np.asarray(prompt, dtype=np.int32).tobytes()).hexdigest(),
        "kv_geometry": geometry,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "raw_payload_retained": False,
        "cold_codec": {
            "name": codec.name,
            "encoded_bytes": encoded_bytes,
            "compression_ratio": len(payload) / encoded_bytes,
        },
        "economics": {
            "warmup_prefill_seconds": warmup_seconds,
            "restore_seconds": restore_samples,
            "recompute_prefill_seconds": recompute,
            "restore_median_seconds": restore_median,
            "recompute_median_seconds": recompute_median,
            "savings_median_seconds": economics.savings_seconds,
            "use_restore": economics.use_restore,
            "scope": "actual model-produced BF16 KV and same-loaded-model native prefill; not end-user TTFT",
        },
        "pressure": pressure_snapshot,
        "offload_passed": [result.passed for result in offload_results],
        "restore_passed": [result.passed for result in restore_result],
        "cancelled_restore_work_ids": list(cancelled),
        "pending_before_final_drain": before_drain["pending_maintenance"],
        "final": final,
        "decision": "implementation qualified; remain default-off pending integrated request-level TTFT/SLO policy",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--max-layers", type=int, default=64)
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
