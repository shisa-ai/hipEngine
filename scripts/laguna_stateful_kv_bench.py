#!/usr/bin/env python3
"""Gate exact Laguna resident-session KV reuse at 128/512/1K prefixes."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Sequence

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import memory_stats
from hipengine.generation import GenerationRequest
from hipengine.generation.laguna_gguf import LagunaGGUFGenerator
from hipengine.loading.gguf import scan_gguf
from hipengine.models.laguna import LAGUNA_GGUF

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")
DEFAULT_MODEL_SHA256 = "7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f"
DEFAULT_PROMPTS = ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--prefix-lengths", default="128,512,1024")
    parser.add_argument("--suffix-tokens", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _prefix_lengths(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in str(raw).split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("prefix lengths must be positive")
    return values


def _natural_tokens(generator: LagunaGGUFGenerator, path: Path, required: int) -> tuple[int, ...]:
    contents: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for message in row["messages"]:
            contents.append(str(message.get("content", "")))
    corpus = "\n\n".join(contents)
    values: list[int] = []
    while len(values) < required:
        values.extend(generator.tokenize(corpus))
    return tuple(values[:required])


def _request(
    prompt: Sequence[int],
    *,
    key: str,
) -> GenerationRequest:
    return GenerationRequest(
        prompts=(tuple(int(token) for token in prompt),),
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        resident_session_key=key,
        resident_session_cache_action="append_visible_only",
    )


def _device_digest(generator: LagunaGGUFGenerator) -> dict[str, Any]:
    session = generator._session
    if session is None or session.kv_cache is None or generator._runtime is None:
        raise RuntimeError("Laguna resident KV is unavailable for digest")
    generator._runtime.device_synchronize()
    digest = hashlib.sha256()
    seen: set[tuple[int, int]] = set()
    copied = 0

    def update(label: str, ptr: int, nbytes: int) -> None:
        nonlocal copied
        signature = (int(ptr), int(nbytes))
        if signature in seen:
            return
        seen.add(signature)
        digest.update(label.encode("utf-8"))
        digest.update(int(nbytes).to_bytes(8, "little"))
        chunk_nbytes = 8 * 1024 * 1024
        offset = 0
        while offset < int(nbytes):
            width = min(chunk_nbytes, int(nbytes) - offset)
            host = (ctypes.c_ubyte * width)()
            generator._runtime.memcpy(
                ctypes.addressof(host),
                int(ptr) + offset,
                width,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
            digest.update(host)
            offset += width
            copied += width

    for layer in session.kv_cache.layers:
        update(f"layer{layer.layer_id}.key", layer.key_cache.ptr, layer.key_cache.nbytes)
        update(f"layer{layer.layer_id}.value", layer.value_cache.ptr, layer.value_cache.nbytes)
        spans = layer.spans
        for name in ("base_offsets", "live_counts", "token_positions", "evict_mask"):
            tensor = getattr(spans, name)
            if tensor is not None:
                update(
                    f"layer{layer.layer_id}.{name}",
                    tensor.ptr,
                    tensor.numel * tensor.dtype.itemsize,
                )
    digest.update(int(session.position).to_bytes(8, "little", signed=True))
    digest.update(int(session.kv_cache.position).to_bytes(8, "little", signed=True))
    return {
        "sha256": digest.hexdigest(),
        "copied_nbytes": copied,
        "session_position": int(session.position),
        "kv_position": int(session.kv_cache.position),
        "pending_positions": list(session.kv_cache.pending_positions),
    }


def _timing(output: Any, wall_ms: float) -> dict[str, Any]:
    telemetry = output.telemetry
    if telemetry is None or telemetry.timing is None or telemetry.diagnostics is None:
        raise RuntimeError("Laguna continuation benchmark requires timing diagnostics")
    return {
        "wall_ms": float(wall_ms),
        "session_prepare_ms": float(telemetry.timing["session_prepare_ms"]),
        "prefill_ms": float(telemetry.timing["prefill_ms"]),
        "request_total_ms": float(telemetry.timing["request_total_ms"]),
        "session_prepare_mode": str(telemetry.diagnostics["session_prepare_mode"]),
        "resident_kv_reused": bool(telemetry.diagnostics["resident_kv_reused"]),
        "prefix_reused_tokens": int(telemetry.diagnostics["prefix_reused_tokens"]),
        "generated_token_ids": list(output.generated_token_ids or ()),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    prefixes = _prefix_lengths(args.prefix_lengths)
    if args.warmups < 1 or args.repetitions < 1:
        raise ValueError("warmups and repetitions must be at least one")
    if args.suffix_tokens <= 0:
        raise ValueError("suffix-tokens must be positive")
    if max(prefixes) + args.suffix_tokens + 1 > args.context_length:
        raise ValueError("prefix plus continuation suffix exceeds context")

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="Q4_K_M mixed GGUF v3",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_stateful_kv",
        timing_protocol="seed_then_exact_suffix_reuse_then_full_reset_control",
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    tracked_before = memory_stats()
    generator = LagunaGGUFGenerator(
        model_path=args.model,
        weight_index=scan_gguf(args.model),
        model_plugin=LAGUNA_GGUF,
        backend=args.backend,
        context_length=args.context_length,
    )
    generator.bind_repacked_cache_source_sha256(args.model_sha256)
    load_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    try:
        generator.prepare(max_sequence_length=args.context_length)
        load_seconds = time.perf_counter() - load_started
        token_stream = _natural_tokens(
            generator,
            args.prompts,
            max(prefixes) + args.suffix_tokens,
        )
        for prefix_length in prefixes:
            for repetition in range(-args.warmups, args.repetitions):
                prefix = token_stream[:prefix_length]
                seed_key = f"reuse-{prefix_length}-{repetition}"
                seed_started = time.perf_counter()
                seed = generator.generate_detailed(_request(prefix, key=seed_key))[0]
                seed_wall_ms = (time.perf_counter() - seed_started) * 1_000.0
                pending = int(seed.generated_token_ids[-1])
                suffix = token_stream[
                    prefix_length : prefix_length + args.suffix_tokens
                ]
                full_prompt = (*prefix, pending, *suffix)

                reuse_started = time.perf_counter()
                reuse = generator.generate_detailed(_request(full_prompt, key=seed_key))[0]
                reuse_wall_ms = (time.perf_counter() - reuse_started) * 1_000.0
                reuse_state = _device_digest(generator) if repetition == 0 else None

                control_started = time.perf_counter()
                control = generator.generate_detailed(
                    _request(full_prompt, key=f"control-{prefix_length}-{repetition}")
                )[0]
                control_wall_ms = (time.perf_counter() - control_started) * 1_000.0
                control_state = _device_digest(generator) if repetition == 0 else None

                if reuse.generated_token_ids != control.generated_token_ids:
                    raise RuntimeError("reused/full continuation token mismatch")
                if reuse_state is not None and reuse_state != control_state:
                    raise RuntimeError("reused/full continuation KV digest mismatch")
                if repetition >= 0:
                    row = {
                        "prefix_tokens": prefix_length,
                        "suffix_tokens": len(full_prompt) - prefix_length,
                        "repetition": repetition,
                        "seed_wall_ms": seed_wall_ms,
                        "seed_token_id": pending,
                        "reuse": _timing(reuse, reuse_wall_ms),
                        "control": _timing(control, control_wall_ms),
                        "state_digest": reuse_state,
                    }
                    rows.append(row)
                    print(
                        f"prefix={prefix_length} rep={repetition} "
                        f"reuse={reuse_wall_ms:.3f} ms control={control_wall_ms:.3f} ms",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        generator.close()
    tracked_after = memory_stats()
    recovered = (
        tracked_before["current_allocated_bytes"]
        == tracked_after["current_allocated_bytes"]
        and tracked_before["active_allocations"] == tracked_after["active_allocations"]
    )

    summaries = {}
    for prefix_length in prefixes:
        selected = [row for row in rows if row["prefix_tokens"] == prefix_length]
        reuse_wall = [row["reuse"]["wall_ms"] for row in selected]
        control_wall = [row["control"]["wall_ms"] for row in selected]
        summaries[str(prefix_length)] = {
            "samples": len(selected),
            "suffix_tokens": selected[0]["suffix_tokens"],
            "reuse_wall_ms_median": statistics.median(reuse_wall),
            "control_wall_ms_median": statistics.median(control_wall),
            "saved_wall_ms_median": statistics.median(control_wall)
            - statistics.median(reuse_wall),
            "speedup": statistics.median(control_wall) / statistics.median(reuse_wall),
            "all_ids_equal": all(
                row["reuse"]["generated_token_ids"]
                == row["control"]["generated_token_ids"]
                for row in selected
            ),
            "all_reuse_modes": all(
                row["reuse"]["session_prepare_mode"] == "reuse"
                and row["reuse"]["prefix_reused_tokens"] == prefix_length
                for row in selected
            ),
        }
    return {
        "schema": "hipengine.laguna_stateful_kv_bench.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "platform": {"python": platform.python_version(), "platform": platform.platform()},
        "load_seconds": load_seconds,
        "prefix_lengths": list(prefixes),
        "suffix_tokens": args.suffix_tokens + 1,
        "rows": rows,
        "summaries": summaries,
        "lifecycle": {"before": tracked_before, "after": tracked_after, "recovered": recovered},
        "gates": {
            "all_ids_equal": all(item["all_ids_equal"] for item in summaries.values()),
            "all_state_digests_equal": all(
                row["state_digest"] is None or bool(row["state_digest"]["sha256"])
                for row in rows
            ),
            "all_reuse_modes": all(item["all_reuse_modes"] for item in summaries.values()),
            "all_reuse_faster": all(item["speedup"] > 1.0 for item in summaries.values()),
            "lifecycle_recovered": recovered,
        },
    }


def main() -> int:
    args = _parse_args()
    result = _run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not all(bool(value) for value in result["gates"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
