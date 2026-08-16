#!/usr/bin/env python3
"""Measure the D08 shared-token no-sampler hipEngine core scope.

This closure diagnostic consumes the committed p512/tg128 token fixture. Prompt
processing replaces the public sampler with a synchronization-only boundary.
Decode captures the normal one-step production graph while temporarily making
the device sampler a no-op; the fixed teacher token remains in the graph input
buffer, so each replay consumes the exact committed continuation ID. Graph
capture and the final untimed logits check are excluded from throughput.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.runtime.prefill import PrefillConfig
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from scripts.qwen35_gguf_bench import _memory_snapshot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = (
    ROOT / "benchmarks" / "fixtures" / "qwen35_08b_vulkan_parity_p512_t128.json"
)
DEFAULT_COMPILER_VERSION = Path("/tmp/d08-c0/hipcc-version.txt")


def _expand_rle(rows: Any) -> list[int]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("token_ids_rle must be a non-empty list")
    result: list[int] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("each token_ids_rle row must be [token_id, count]")
        token_id, count = int(row[0]), int(row[1])
        if token_id < 0 or count <= 0:
            raise ValueError("token IDs must be non-negative and counts positive")
        result.extend([token_id] * count)
    return result


def _load_fixture(path: Path) -> tuple[dict[str, Any], list[int], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema", -1)) != 1:
        raise ValueError("unsupported parity fixture schema")
    prompt = _expand_rle(payload["prompt"]["token_ids_rle"])
    continuation = _expand_rle(
        payload["teacher_forced_continuation"]["token_ids_rle"]
    )
    if len(prompt) != int(payload["prompt"]["count"]):
        raise ValueError("prompt RLE count does not match fixture")
    if len(continuation) != int(payload["teacher_forced_continuation"]["count"]):
        raise ValueError("continuation RLE count does not match fixture")
    if len(set(continuation)) != 1:
        raise ValueError("the captured no-sampler graph requires one fixed continuation ID")
    return payload, prompt, continuation


def _stats(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples": values,
        "median": float(statistics.median(values)),
        "mean": float(statistics.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _run(
    model: Path,
    *,
    fixture_path: Path,
    compiler_version: str,
    repetitions: int,
) -> dict[str, Any]:
    fixture, prompt, continuation = _load_fixture(fixture_path)
    teacher_token = int(continuation[0])
    steps = len(continuation)
    session = Qwen35GGUFResidentSession(
        model,
        backend="hip_gfx1151",
        compiler_version=compiler_version,
        require_cached_build=True,
        max_sequence_length=len(prompt) + steps + 8,
        token_embedding_placement="device",
        use_wmma_prefill=True,
        use_gemv_decode=True,
        prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
    )
    runtime = session.runtime
    if runtime is None:
        session.close()
        raise RuntimeError("HIP runtime is unavailable")

    try:
        def core_once() -> dict[str, Any]:
            original_host_sample = session._sample_from_hidden

            def sync_only(
                hidden_ptr: int,
                *,
                return_logits: bool = True,
                stream: int = 0,
            ) -> None:
                del hidden_ptr, return_logits
                if stream:
                    runtime.stream_synchronize(stream)
                else:
                    runtime.device_synchronize()

            session._sample_from_hidden = sync_only
            try:
                runtime.device_synchronize()
                started = time.perf_counter_ns()
                session.prefill(
                    prompt,
                    use_bulk=True,
                    bulk_attention_mode="bulk",
                    return_logits=False,
                    capture_hidden_seed_fp32=False,
                )
                runtime.device_synchronize()
                prefill_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            finally:
                session._sample_from_hidden = original_host_sample

            original_device_sample = session._sample_device_from_hidden
            session._sample_device_from_hidden = (
                lambda hidden_ptr, *, stream=0: None
            )
            try:
                graph = session.capture_decode_graph(
                    position=session.position,
                    steps_per_replay=1,
                    max_replay_steps=steps,
                    record_steps=0,
                    input_token_id=teacher_token,
                )
            finally:
                session._sample_device_from_hidden = original_device_sample

            try:
                core_graph_nodes = len(runtime.graph_nodes(graph.graph))
                runtime.device_synchronize()
                started = time.perf_counter_ns()
                graph.replay(steps)
                runtime.device_synchronize()
                decode_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                final = original_host_sample(session.scratch.norm.ptr, return_logits=True)
                return {
                    "prefill_ms": prefill_ms,
                    "decode_ms": decode_ms,
                    "final_top1": int(final.token_id),
                    "finite": bool(np.isfinite(final.logits).all()),
                    "core_graph_nodes": core_graph_nodes,
                }
            finally:
                graph.close()

        def public_once() -> dict[str, Any]:
            runtime.device_synchronize()
            started = time.perf_counter_ns()
            first = session.prefill(
                prompt,
                use_bulk=True,
                bulk_attention_mode="bulk",
                return_logits=False,
                capture_hidden_seed_fp32=False,
            )
            runtime.device_synchronize()
            prefill_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            graph = session.capture_decode_graph(
                position=session.position,
                steps_per_replay=1,
                max_replay_steps=steps,
                record_steps=0,
                input_token_id=int(first.token_id),
            )
            try:
                graph_nodes = len(runtime.graph_nodes(graph.graph))
                runtime.device_synchronize()
                started = time.perf_counter_ns()
                graph.replay(steps)
                runtime.device_synchronize()
                decode_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                final = graph.read_sample(return_logits=True)
                return {
                    "prefill_ms": prefill_ms,
                    "decode_ms": decode_ms,
                    "final_top1": int(final.token_id),
                    "finite": bool(np.isfinite(final.logits).all()),
                    "graph_nodes": graph_nodes,
                }
            finally:
                graph.close()

        core_once()
        public_once()
        rows = [core_once() for _ in range(repetitions)]
        public_rows = [public_once() for _ in range(repetitions)]

        def correctness() -> tuple[list[int], bool]:
            first = session.prefill(
                prompt,
                use_bulk=True,
                bulk_attention_mode="bulk",
                return_logits=True,
                capture_hidden_seed_fp32=False,
            )
            token_ids = [int(first.token_id)]
            finite = bool(np.isfinite(first.logits).all())
            for token_id in continuation:
                row = session.step(int(token_id), return_logits=True)
                token_ids.append(int(row.token_id))
                finite = finite and bool(np.isfinite(row.logits).all())
            return token_ids, finite

        def public_correctness() -> tuple[list[int], bool]:
            first = session.prefill(
                prompt,
                use_bulk=True,
                bulk_attention_mode="bulk",
                return_logits=True,
                capture_hidden_seed_fp32=False,
            )
            token_ids = [int(first.token_id)]
            finite = bool(np.isfinite(first.logits).all())
            current = int(first.token_id)
            for _ in continuation:
                row = session.step(current, return_logits=True)
                current = int(row.token_id)
                token_ids.append(current)
                finite = finite and bool(np.isfinite(row.logits).all())
            return token_ids, finite

        top1_ids, finite_first = correctness()
        top1_repeat, finite_repeat = correctness()
        public_top1_ids, public_finite_first = public_correctness()
        public_top1_repeat, public_finite_repeat = public_correctness()
        snapshot = _memory_snapshot("closure", runtime, session)
        prefill_ms = [float(row["prefill_ms"]) for row in rows]
        decode_ms = [float(row["decode_ms"]) for row in rows]
        public_prefill_ms = [float(row["prefill_ms"]) for row in public_rows]
        public_decode_ms = [float(row["decode_ms"]) for row in public_rows]
        return {
            "schema": 1,
            "engine": "hipengine",
            "model": str(model.resolve()),
            "fixture": str(fixture_path.resolve()),
            "fixture_name": fixture["name"],
            "protocol": (
                "exact fixture prompt plus forced continuation; no-sampler bulk "
                "prefill and one-step production graph core; one warmup plus "
                f"{repetitions} measures; separate public-path top-1 repeats"
            ),
            "prefill_ms": _stats(prefill_ms),
            "prefill_tok_s": _stats([len(prompt) * 1000.0 / value for value in prefill_ms]),
            "decode_ms": _stats(decode_ms),
            "decode_tok_s": _stats([steps * 1000.0 / value for value in decode_ms]),
            "core_graph_nodes": sorted({int(row["core_graph_nodes"]) for row in rows}),
            "public_prefill_ms": _stats(public_prefill_ms),
            "public_prefill_tok_s": _stats(
                [len(prompt) * 1000.0 / value for value in public_prefill_ms]
            ),
            "public_decode_ms": _stats(public_decode_ms),
            "public_decode_tok_s": _stats(
                [steps * 1000.0 / value for value in public_decode_ms]
            ),
            "public_graph_nodes": sorted(
                {int(row["graph_nodes"]) for row in public_rows}
            ),
            "timed_final_top1": [int(row["final_top1"]) for row in rows],
            "public_final_top1": [int(row["final_top1"]) for row in public_rows],
            "timed_all_finite": all(bool(row["finite"]) for row in rows),
            "public_all_finite": all(bool(row["finite"]) for row in public_rows),
            "top1_ids": top1_ids,
            "top1_repeat_exact": top1_ids == top1_repeat,
            "top1_all_finite": finite_first and finite_repeat,
            "public_top1_ids": public_top1_ids,
            "public_top1_repeat_exact": public_top1_ids == public_top1_repeat,
            "public_top1_all_finite": public_finite_first and public_finite_repeat,
            "memory": {
                "owned_session_bytes": int(snapshot["owned_session_bytes"]),
                "tracked_current_bytes": int(snapshot["tracked"]["current_allocated_bytes"]),
                "tracked_peak_bytes": int(snapshot["tracked"]["peak_allocated_bytes"]),
                "hip_used_bytes": int(snapshot["hip"]["used_bytes"]),
            },
        }
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--compiler-version-file", type=Path, default=DEFAULT_COMPILER_VERSION)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    payload = _run(
        args.model,
        fixture_path=args.fixture,
        compiler_version=args.compiler_version_file.read_text(encoding="utf-8"),
        repetitions=args.repetitions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: payload[key]
        for key in (
            "prefill_tok_s",
            "decode_tok_s",
            "core_graph_nodes",
            "public_prefill_tok_s",
            "public_decode_tok_s",
            "public_graph_nodes",
            "timed_all_finite",
            "public_all_finite",
            "top1_repeat_exact",
            "top1_all_finite",
            "public_top1_repeat_exact",
            "public_top1_all_finite",
            "memory",
        )
    }, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
