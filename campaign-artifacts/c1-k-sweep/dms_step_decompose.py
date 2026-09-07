#!/usr/bin/env python3
"""Decompose one DMS decode step: host sync, finalize, kernel enqueue costs.

Runs the standard dense_vs_dms arm flow (16K prompt, N decode steps) with
instance-level timing wrappers around the per-step hot path:

- ``HipRuntime.stream_synchronize`` (gated to the decode window)
- ``DMSCompactBackend.finalize_device_append`` (host bookkeeping)
- ``DMSDevicePayloadStore.live_counts`` (per-layer D2H reads inside finalize)
- ``attention_layer_device`` / ``append_layer_device`` (enqueue cost)

Reports per-step means for each phase plus the step wall, decomposing the
DMS-vs-dense delta. Diagnostics only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _validation_stream(path: Path) -> list[int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sequences = sorted(
        (row for row in raw["sequences"] if str(row.get("split")) == "validation"),
        key=lambda row: str(row["sequence_id"]),
    )
    return [int(t) for row in sequences for t in row["token_ids"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--metadata", type=Path,
                        default=Path("/models/dms/qwen38-27b-q4km-dms-w8192-local/dms_metadata.json"))
    parser.add_argument("--data-manifest", type=Path,
                        default=Path("/models/dms/xtx-manifests/qwen38-27b-q4km-xtx-capacity-manifest.json"))
    parser.add_argument("--prompt-tokens", type=int, default=16384)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--arm", default="dms-bf16")  # or dms-int8
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from hipengine.kvcache.dms import (
        create_dms_bf16_backend,
        create_dms_int8_evaluation_backend,
    )
    from hipengine.runtime.qwen35_gguf_runner import (
        Qwen35GGUFFullStackRunner,
        Qwen35GGUFResidentSession,
    )
    backend_factory = (
        create_dms_int8_evaluation_backend
        if args.arm == "dms-int8"
        else create_dms_bf16_backend
    )

    stream = _validation_stream(args.data_manifest)
    prompt = stream[: args.prompt_tokens]
    steps = args.decode_steps

    runner = Qwen35GGUFFullStackRunner(args.model, backend="hip_gfx1100")

    # Timing state.
    buckets = {
        "sync_ms": [],
        "finalize_ms": [],
        "live_counts_ms": [],
        "attn_enqueue_ms": [],
        "append_enqueue_ms": [],
        "rope_enqueue_ms": [],
        "projector_ms": [],
        "cast_ms": [],
        "gate_mul_ms": [],
    }
    recording = {"on": False}

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    original_sync = runtime.stream_synchronize
    original_dev_sync = runtime.device_synchronize

    def timed_sync(stream_id: int = 0):
        if recording["on"]:
            started = time.perf_counter()
            original_sync(int(stream_id))
            buckets["sync_ms"].append((time.perf_counter() - started) * 1000)
        else:
            original_sync(int(stream_id))

    def timed_dev_sync():
        if recording["on"]:
            started = time.perf_counter()
            original_dev_sync()
            buckets["sync_ms"].append((time.perf_counter() - started) * 1000)
        else:
            original_dev_sync()

    runtime.stream_synchronize = timed_sync
    runtime.device_synchronize = timed_dev_sync

    max_len = len(prompt) + steps + 1
    session = Qwen35GGUFResidentSession(
        args.model,
        backend="hip_gfx1100",
        shared_runner=runner,
        max_sequence_length=max_len,
        dms_metadata_path=args.metadata,
        dms_backend_factory=backend_factory,
        dms_max_new_tokens=steps + 1,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    )
    session.__enter__()
    try:
        session.prefill(prompt, use_bulk=True, bulk_attention_mode="bulk",
                        return_logits=False, record_gpu_stage_timings=False)

        backend = session._dms_backend
        store = backend._device_store
        if store is None:
            raise RuntimeError("device payloads not enabled")

        original_finalize = backend.finalize_device_append
        original_live = store.live_counts
        original_attn = store.attention_layer_device
        original_append = store.append_layer_device

        def timed_finalize(request_id, *, eviction, position):
            if recording["on"]:
                started = time.perf_counter()
                result = original_finalize(request_id, eviction=eviction,
                                           position=position)
                buckets["finalize_ms"].append((time.perf_counter() - started) * 1000)
                return result
            return original_finalize(request_id, eviction=eviction,
                                     position=position)

        def timed_live(layer):
            if recording["on"]:
                started = time.perf_counter()
                result = original_live(layer)
                buckets["live_counts_ms"].append((time.perf_counter() - started) * 1000)
                return result
            return original_live(layer)

        def timed_attn(layer, **kwargs):
            if recording["on"]:
                started = time.perf_counter()
                result = original_attn(layer, **kwargs)
                buckets["attn_enqueue_ms"].append((time.perf_counter() - started) * 1000)
                return result
            return original_attn(layer, **kwargs)

        def timed_append(layer, **kwargs):
            if recording["on"]:
                started = time.perf_counter()
                result = original_append(layer, **kwargs)
                buckets["append_enqueue_ms"].append((time.perf_counter() - started) * 1000)
                return result
            return original_append(layer, **kwargs)

        backend.finalize_device_append = timed_finalize
        store.live_counts = timed_live
        store.attention_layer_device = timed_attn
        store.append_layer_device = timed_append

        # Per-layer small-kernel launch costs (the FastDMS fusion targets).
        import hipengine.runtime.qwen35_gguf_runner as runner_mod
        from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
            qwen35_full_attn_gate_mul_bf16 as gate_mul_fn,
        )
        from hipengine.kernels.hip_gfx1100.convert.cast import f32_to_bf16 as cast_fn

        shared_runner = session.runner
        original_qk_resolver = shared_runner._full_attn_qk_postprocess_fn
        original_gate = runner_mod.qwen35_full_attn_gate_mul_bf16
        original_cast = runner_mod.f32_to_bf16

        def timed_qk_resolver():
            fn = original_qk_resolver()

            def timed_rope(*a, **kw):
                if recording["on"]:
                    started = time.perf_counter()
                    result = fn(*a, **kw)
                    buckets["rope_enqueue_ms"].append((time.perf_counter() - started) * 1000)
                    return result
                return fn(*a, **kw)

            return timed_rope

        def timed_gate(*a, **kw):
            if recording["on"]:
                started = time.perf_counter()
                result = original_gate(*a, **kw)
                buckets["gate_mul_ms"].append((time.perf_counter() - started) * 1000)
                return result
            return original_gate(*a, **kw)

        def timed_cast(*a, **kw):
            if recording["on"]:
                started = time.perf_counter()
                result = original_cast(*a, **kw)
                buckets["cast_ms"].append((time.perf_counter() - started) * 1000)
                return result
            return original_cast(*a, **kw)

        shared_runner._full_attn_qk_postprocess_fn = timed_qk_resolver
        runner_mod.qwen35_full_attn_gate_mul_bf16 = timed_gate
        runner_mod.f32_to_bf16 = timed_cast

        projector = getattr(session, "_dms_decode_projector", None)
        if projector is not None and hasattr(projector, "project"):
            original_project = projector.project

            def timed_project(*a, **kw):
                if recording["on"]:
                    started = time.perf_counter()
                    result = original_project(*a, **kw)
                    buckets["projector_ms"].append((time.perf_counter() - started) * 1000)
                    return result
                return original_project(*a, **kw)

            projector.project = timed_project

        current = int(prompt[-1])
        step_walls = []
        for _ in range(steps):
            for key in buckets:
                buckets[key].clear()
            recording["on"] = True
            step_started = time.perf_counter()
            result = session.step(current, return_logits=True)
            step_walls.append(time.perf_counter() - step_started)
            recording["on"] = False
            current = int(result.token_id)
            per_step = {k: (sum(v) / len(v)) if v else 0.0 for k, v in buckets.items()}
            counts = {k: len(v) for k, v in buckets.items()}
            print(
                f"[decompose] wall={step_walls[-1]*1000:.2f}ms "
                f"sync={per_step['sync_ms']:.2f}({counts['sync_ms']}) "
                f"finalize={per_step['finalize_ms']:.2f}({counts['finalize_ms']}) "
                f"attn_enq={per_step['attn_enqueue_ms']:.2f}({counts['attn_enqueue_ms']}) "
                f"append_enq={per_step['append_enqueue_ms']:.2f}({counts['append_enqueue_ms']}) "
                f"rope={per_step['rope_enqueue_ms']:.2f}({counts['rope_enqueue_ms']}) "
                f"proj={per_step['projector_ms']:.2f}({counts['projector_ms']}) "
                f"cast={per_step['cast_ms']:.2f}({counts['cast_ms']}) "
                f"gate={per_step['gate_mul_ms']:.2f}({counts['gate_mul_ms']})",
                flush=True,
            )

        walls = np.asarray(step_walls) * 1000
        out = {
            "kind": "dms_step_decompose",
            "arm": args.arm,
            "prompt_tokens": args.prompt_tokens,
            "decode_steps": steps,
            "step_wall_ms_median": float(np.median(walls)),
            "step_wall_ms_mean": float(walls.mean()),
        }
        args.output.write_text(json.dumps(out, indent=2) + "\n")
        print(f"[decompose] wrote {args.output}", flush=True)
    finally:
        session.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
