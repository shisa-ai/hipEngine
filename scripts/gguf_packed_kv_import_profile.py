#!/usr/bin/env python3
"""Quantify the per-slab whole-history KV import on packed slot-local prefill.

The packed slot-local INT8 prefill route (the server path for ``int8_direct``
sessions) calls ``_sync_packed_decode_initial_state`` once per slab, and each
call imports every session's whole prior KV history into packed storage --
even though slot-local attention reads request-owned KV and the end-of-slab
scatter already skips packed KV on that route
(``copy_kv=not slot_local_full_prefill``). At fixed chunk size the cumulative
import work is quadratic in the chunk count.

This probe measures the import directly instead of inferring it from
throughput. For each row count (1/2/4/8 chunks on the dense H5120 Q4_K_M
geometry) it reports, for one ``prefill_batch_native`` call with the shipping
selectors:

- KV import calls / rows / ``memcpy_async`` dispatches / bytes
  (session -> packed), and the packed -> session scatter side for contrast;
- ``_sync_packed_decode_initial_state`` call count (one per slab) and the
  Conv/GDN linear-state import path (fused kernel calls + fallback memcpys);
- wall time of the synchronized prefill, and CPU submission time of the same
  call with stream/device synchronization neutralized (``sample_output=False``
  arm, own session, real sync restored before teardown);
- chunk plan from ``last_packed_prefill_plan``;
- sha256 of the full final logits and 8 greedy decode IDs, so a before/after
  comparison of an import-removal patch can demand exact equality on the
  same schedule.

Decode-phase counters (the import the packed decode round performs after
prefill) are recorded separately and are not part of the prefill budget.

Example:

    HIP_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1 \
    HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1 \
    HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS=none \
    python scripts/gguf_packed_kv_import_profile.py --rows 1024,2048,4096,8192 \
        --json /tmp/kv-import-profile.json

Under rocprofv3, do NOT launch this script bare: the profiled process must
never probe the compiler or it can stall in hipcc --version (see
benchmarks/HARNESSES.md "Profiling these harnesses under rocprofv3" and
docs/KERNELS.md). Prewarm first, then profile cache-only:

    hipcc --version > /tmp/hipcc-version.txt   # on the host, not under rocprofv3
    HIP_VISIBLE_DEVICES=0 ... python scripts/gguf_packed_kv_import_profile.py \
        --rows 2048 --skip-cpu-submission --json /tmp/warm.json      # prewarm
    HIP_VISIBLE_DEVICES=0 ... \
    HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipcc-version.txt \
    HIPENGINE_REQUIRE_CACHED_BUILD=1 \
    rocprofv3 --kernel-trace --memory-copy-trace --output-format csv \
        -d /tmp/rocprof-out -- \
        python scripts/gguf_packed_kv_import_profile.py \
        --rows 2048 --skip-cpu-submission --json /tmp/profiled.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    ap.add_argument("--rows", default="1024,2048,4096,8192")
    ap.add_argument("--max-sequence-length", type=int, default=16384)
    ap.add_argument("--decode-tokens", type=int, default=8)
    ap.add_argument("--vocab-span", type=int, default=32000)
    ap.add_argument("--kv-storage", default="int8_per_token_head")
    ap.add_argument("--kv-scale-dtype", default="fp32")
    ap.add_argument("--skip-cpu-submission", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.kvcache import resolve_kv_policy
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    runtime = get_hip_runtime()
    policy = resolve_kv_policy(str(args.kv_storage), scale_dtype=str(args.kv_scale_dtype))
    rng = np.random.default_rng(20260910)

    # ---- counters ---------------------------------------------------------
    def new_bucket() -> dict:
        return {
            "kv_import": {"calls": 0, "rows": 0, "memcpys": 0, "bytes": 0},
            "kv_scatter": {"calls": 0, "rows": 0, "memcpys": 0, "bytes": 0},
            "sync_calls": 0,
            "sync_kv_skipped": 0,
            "fused_linear_copies": 0,
            "sync_other_memcpys": 0,
        }

    counters = {"prefill": new_bucket(), "decode": new_bucket(), "other": new_bucket()}
    phase = {"name": "other"}
    kv_direction = {"name": None}

    orig_kv_copy = Qwen35GGUFResidentSession._copy_session_packed_kv_segments
    orig_sync = Qwen35GGUFResidentSession._sync_packed_decode_initial_state
    orig_fused = Qwen35GGUFResidentSession._fused_linear_state_pair_copy
    orig_memcpy = runtime.memcpy_async

    def counting_kv_copy(
        self, session, packed_state, slot_index, layer_id, *,
        start_position, rows, packed_to_session, runtime, stream,
    ):
        bucket = counters[phase["name"]]["kv_scatter" if packed_to_session else "kv_import"]
        bucket["calls"] += 1
        bucket["rows"] += int(rows)
        kv_direction["name"] = "kv_scatter" if packed_to_session else "kv_import"
        try:
            return orig_kv_copy(
                self, session, packed_state, slot_index, layer_id,
                start_position=start_position, rows=rows,
                packed_to_session=packed_to_session, runtime=runtime, stream=stream,
            )
        finally:
            kv_direction["name"] = None

    def counting_sync(self, sessions, layout, packed_state, *, runtime, stream, copy_linear_state=True, copy_kv=True):
        bucket = counters[phase["name"]]
        bucket["sync_calls"] += 1
        if not copy_kv:
            bucket["sync_kv_skipped"] += 1
        return orig_sync(
            self, sessions, layout, packed_state,
            runtime=runtime, stream=stream,
            copy_linear_state=copy_linear_state, copy_kv=copy_kv,
        )

    def counting_fused(self, copies, *, runtime, stream):
        counters[phase["name"]]["fused_linear_copies"] += 1
        return orig_fused(self, copies, runtime=runtime, stream=stream)

    def counting_memcpy(destination, source, nbytes, kind, stream):
        name = kv_direction["name"]
        if name is not None:
            bucket = counters[phase["name"]][name]
            bucket["memcpys"] += 1
            bucket["bytes"] += int(nbytes)
        else:
            counters[phase["name"]]["sync_other_memcpys"] += 1
        return orig_memcpy(destination, source, nbytes, kind, stream)

    Qwen35GGUFResidentSession._copy_session_packed_kv_segments = counting_kv_copy
    Qwen35GGUFResidentSession._sync_packed_decode_initial_state = counting_sync
    Qwen35GGUFResidentSession._fused_linear_state_pair_copy = counting_fused
    runtime.memcpy_async = counting_memcpy

    def reset_counters() -> None:
        for name in tuple(counters):
            counters[name] = new_bucket()

    def snapshot() -> dict:
        return json.loads(json.dumps(counters))

    # ---- measurement passes ----------------------------------------------
    def open_session() -> Qwen35GGUFResidentSession:
        return Qwen35GGUFResidentSession(
            args.model, runtime=runtime,
            max_sequence_length=int(args.max_sequence_length),
            prefill_config=PrefillConfig(),
            kv_policy=policy.create_policy(),
            kv_scale_dtype=str(args.kv_scale_dtype),
            kv_scale_granularity=str(policy.scale_granularity),
            # Match the shipping serving owner's low-level selectors (see
            # scripts/gguf_prefill_route_ab.py for why omitting them
            # invalidates rate attribution).
            use_wmma_prefill=True,
            use_gemv_decode=True,
        )

    out = {
        "kind": "packed_slot_local_kv_import_profile",
        "model": args.model,
        "max_sequence_length": int(args.max_sequence_length),
        "kv_storage": str(args.kv_storage),
        "kv_scale_dtype": str(args.kv_scale_dtype),
        "prompt_kind": "deterministic_varied_rng20260910",
        "decode_tokens": int(args.decode_tokens),
        "env": {k: os.environ.get(k) for k in (
            "HIP_VISIBLE_DEVICES",
            "GPU_MAX_HW_QUEUES",
            "HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG",
            "HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS",
        )},
        "configs": [],
    }

    for row_count in (int(value) for value in str(args.rows).split(",") if value.strip()):
        prompt = [int(t) for t in rng.integers(1000, args.vocab_span, size=row_count)]
        config: dict = {"rows": row_count}

        # Pass 1: synchronized wall + counters + logits + decode continuation.
        with open_session() as session:
            reset_counters()
            phase["name"] = "prefill"
            t0 = time.perf_counter()
            result = session.prefill_batch_native(
                [prompt], sessions=[session],
                full_prompt_lengths=[len(prompt)], return_logits=True,
            )[0]
            wall = time.perf_counter() - t0
            phase["name"] = "decode"
            logits = np.asarray(result.logits, dtype=np.float32).reshape(-1)
            ids = [int(result.token_id)]
            nxt = ids[0]
            for _ in range(int(args.decode_tokens) - 1):
                step = session.step(nxt, return_logits=False)
                nxt = int(step.token_id)
                ids.append(nxt)
            phase["name"] = "other"
            plan = dict(session.last_packed_prefill_plan)
            peak_gib = int(memory_stats().get("peak_allocated_bytes", 0)) / 2**30
            kv_source = getattr(session, "kv_attention_source", None)
        config.update(
            chunk_count=plan.get("chunk_count"),
            chunk_rows=plan.get("chunk_rows"),
            kv_attention_source=kv_source,
            prefill_wall_seconds=round(wall, 3),
            prefill_tok_s=round(row_count / wall, 2),
            tracked_peak_gib=round(peak_gib, 4),
            logits_sha256=hashlib.sha256(
                np.ascontiguousarray(logits).tobytes()
            ).hexdigest(),
            generated_ids=ids,
            prefill_counters=snapshot()["prefill"],
            decode_counters=snapshot()["decode"],
        )

        # Pass 2: CPU submission time (sync neutralized, outputs discarded).
        if not args.skip_cpu_submission:
            with open_session() as session:
                saved_sync = (runtime.stream_synchronize, runtime.device_synchronize)
                runtime.stream_synchronize = lambda *a, **k: None
                runtime.device_synchronize = lambda *a, **k: None
                try:
                    t0 = time.perf_counter()
                    session.prefill_batch_native(
                        [prompt], sessions=[session],
                        full_prompt_lengths=[len(prompt)], sample_output=False,
                    )
                    cpu_submission = time.perf_counter() - t0
                finally:
                    (runtime.stream_synchronize, runtime.device_synchronize) = saved_sync
                    runtime.device_synchronize()
                plan2 = dict(session.last_packed_prefill_plan)
            config["cpu_submission_seconds"] = round(cpu_submission, 3)
            config["cpu_submission_chunk_count"] = plan2.get("chunk_count")

        out["configs"].append(config)
        print(
            f"[rows={row_count} chunks={config['chunk_count']}] "
            f"wall {config['prefill_wall_seconds']} s ({config['prefill_tok_s']} tok/s) "
            f"import {config['prefill_counters']['kv_import']['calls']} calls / "
            f"{config['prefill_counters']['kv_import']['rows']} rows / "
            f"{config['prefill_counters']['kv_import']['bytes'] / 2**20:.1f} MiB",
            flush=True,
        )

    payload = json.dumps(out, indent=2, default=str)
    print(payload)
    if args.json:
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
