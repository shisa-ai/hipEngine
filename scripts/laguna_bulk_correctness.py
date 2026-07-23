#!/usr/bin/env python3
"""Compare Laguna chunked prefill and B+1 rows with exact eager execution."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import free, host_array_ptr, malloc, memory_stats
from hipengine.loading.laguna_gguf import FULL_ATTENTION
from hipengine.runtime.laguna_gguf_runner import (
    LAGUNA_DFLASH_CAPTURE_DEPTHS,
    LagunaGGUFResidentSession,
    LagunaHiddenCaptureTargets,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")
DEFAULT_TEMPLATE = ROOT / "tests/fixtures/laguna_poolside_v1_template.json"
DEFAULT_ORACLE = ROOT / "tests/fixtures/laguna_poolside_v1_oracle.json"
DEFAULT_CACHE = Path(
    "/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.hipengine-repacked-v1"
)
DEFAULT_LENGTHS = (1, 2, 7, 55, 65)


@dataclass(frozen=True)
class _Snapshot:
    elapsed_seconds: float
    next_token_id: int
    logits: np.ndarray
    final_hidden: np.ndarray
    post_layer_hidden: np.ndarray
    captures: dict[int, np.ndarray]
    kv_sha256: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--lengths",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        default=DEFAULT_LENGTHS,
    )
    parser.add_argument("--verifier-prefix-length", type=int, default=7)
    parser.add_argument("--verifier-rows", type=int, default=5)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--direct-gguf", action="store_true")
    parser.add_argument("--safety-reserve-gib", type=float, default=8.0)
    parser.add_argument("--model-sha256")
    parser.add_argument("--quant-label", default="Q4_K_M mixed GGUF v3")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _compiler_version(path: Path | None) -> str | None:
    return None if path is None else path.read_text(encoding="utf-8")


def _progress(completed: int, total: int, spec) -> None:
    if completed == 1 or completed == total or completed % 100 == 0:
        print(
            f"load {completed}/{total}: {spec.source.name} ({spec.layout})",
            file=sys.stderr,
            flush=True,
        )


def _sequence(template_path: Path, oracle_path: Path) -> tuple[int, ...]:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    prompt_case = next(
        case for case in template["cases"] if case["name"] == oracle["prompt"]["case"]
    )
    prompt = tuple(int(value) for value in prompt_case["token_ids"])
    continuation = tuple(int(value) for value in oracle["greedy32"]["token_ids"])
    return prompt + continuation


def _copy_array(runtime, buffer, dtype, shape: tuple[int, ...]) -> np.ndarray:
    out = np.empty(shape, dtype=dtype)
    expected = int(out.nbytes)
    if expected > int(buffer.nbytes):
        raise ValueError("requested host array exceeds the borrowed device buffer")
    runtime.memcpy(
        host_array_ptr(out),
        buffer.ptr,
        expected,
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return out


def _capture_targets(runtime, hidden_size: int, rows: int):
    row_nbytes = int(hidden_size) * np.dtype(np.uint16).itemsize
    buffers = [
        malloc(int(rows) * row_nbytes, runtime=runtime)
        for _ in LAGUNA_DFLASH_CAPTURE_DEPTHS
    ]
    targets = LagunaHiddenCaptureTargets(
        hidden_size=int(hidden_size),
        buffers=dict(zip(LAGUNA_DFLASH_CAPTURE_DEPTHS, buffers, strict=True)),
        rows=int(rows),
    )
    return targets, buffers


def _capture_arrays(runtime, targets: LagunaHiddenCaptureTargets) -> dict[int, np.ndarray]:
    return {
        int(depth): _copy_array(
            runtime,
            buffer,
            np.uint16,
            (targets.rows, targets.hidden_size),
        )
        for depth, buffer in targets.buffers.items()
    }


def _kv_digest(session: LagunaGGUFResidentSession) -> str:
    assert session.kv_cache is not None
    runtime = session.runtime
    digest = hashlib.sha256()
    width = session.config.head_count_kv * session.config.key_length
    row_nbytes = width * np.dtype(np.uint16).itemsize
    for state in session.kv_cache.layers:
        offsets = np.empty(state.spans.base_offsets.numel, dtype=np.int32)
        live_counts = np.empty(state.spans.live_counts.numel, dtype=np.int64)
        positions = np.empty(state.capacity, dtype=np.int64)
        mask = np.empty(state.capacity, dtype=np.bool_)
        for tensor, destination in (
            (state.spans.base_offsets, offsets),
            (state.spans.live_counts, live_counts),
            (state.spans.token_positions, positions),
            (state.spans.evict_mask, mask),
        ):
            assert tensor is not None
            runtime.memcpy(
                host_array_ptr(destination),
                tensor.ptr,
                destination.nbytes,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
        digest.update(int(state.layer_id).to_bytes(4, "little", signed=False))
        digest.update(state.attention_type.encode("utf-8"))
        digest.update(offsets.tobytes())
        digest.update(live_counts.tobytes())
        digest.update(positions.tobytes())
        digest.update(mask.tobytes())
        visible_slots = np.flatnonzero((~mask) & (positions >= 0))
        for logical_slot in visible_slots.tolist():
            if state.attention_type == FULL_ATTENTION:
                block_size = 256
                physical_slot = (
                    int(offsets[logical_slot // block_size]) * block_size
                    + logical_slot % block_size
                )
            else:
                physical_slot = int(offsets[logical_slot])
            digest.update(int(logical_slot).to_bytes(4, "little", signed=False))
            digest.update(int(positions[logical_slot]).to_bytes(8, "little", signed=True))
            for cache in (state.key_cache, state.value_cache):
                payload = np.empty(width, dtype=np.uint16)
                runtime.memcpy(
                    host_array_ptr(payload),
                    cache.ptr + physical_slot * row_nbytes,
                    row_nbytes,
                    HipMemcpyKind.DEVICE_TO_HOST,
                )
                digest.update(payload.tobytes())
    return digest.hexdigest()


def _session(owner: LagunaGGUFResidentSession, args: argparse.Namespace):
    assert owner.weights is not None
    return LagunaGGUFResidentSession(
        resident_weights=owner.weights,
        context_length=args.context_length,
        backend=args.backend,
        runtime=owner.runtime,
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached_build=args.require_cached_build,
        prefill_chunk_size=args.chunk_size,
    )


def _prefill_snapshot(
    owner: LagunaGGUFResidentSession,
    tokens: tuple[int, ...],
    *,
    use_bulk: bool,
    args: argparse.Namespace,
) -> _Snapshot:
    runtime = owner.runtime
    session = _session(owner, args)
    targets, buffers = _capture_targets(runtime, session.config.hidden_size, 1)
    try:
        started = time.perf_counter()
        result = session.prefill(tokens, capture_last=targets, use_bulk=use_bulk)
        elapsed = time.perf_counter() - started
        snapshot = _Snapshot(
            elapsed_seconds=elapsed,
            next_token_id=int(result.next_token_id),
            logits=_copy_array(runtime, result.logits, np.float32, (session.config.vocab_size,)),
            final_hidden=_copy_array(
                runtime,
                result.final_hidden,
                np.uint16,
                (session.config.hidden_size,),
            ),
            post_layer_hidden=_copy_array(
                runtime,
                result.post_layer_hidden,
                np.uint16,
                (session.config.hidden_size,),
            ),
            captures={depth: rows[0] for depth, rows in _capture_arrays(runtime, targets).items()},
            kv_sha256=_kv_digest(session),
        )
        return snapshot
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        session.close()


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _compare_snapshot(
    serial: _Snapshot,
    bulk: _Snapshot,
    *,
    length: int,
) -> dict[str, Any]:
    logits_equal = bool(np.array_equal(serial.logits, bulk.logits))
    hidden_equal = bool(np.array_equal(serial.final_hidden, bulk.final_hidden))
    post_equal = bool(np.array_equal(serial.post_layer_hidden, bulk.post_layer_hidden))
    capture_equal = {
        str(depth): bool(np.array_equal(serial.captures[depth], bulk.captures[depth]))
        for depth in LAGUNA_DFLASH_CAPTURE_DEPTHS
    }
    passed = bool(
        serial.next_token_id == bulk.next_token_id
        and logits_equal
        and hidden_equal
        and post_equal
        and all(capture_equal.values())
        and serial.kv_sha256 == bulk.kv_sha256
    )
    return {
        "length": int(length),
        "pass": passed,
        "serial_seconds": serial.elapsed_seconds,
        "bulk_seconds": bulk.elapsed_seconds,
        "speedup_diagnostic": serial.elapsed_seconds / bulk.elapsed_seconds,
        "next_token_id": bulk.next_token_id,
        "next_token_equal": serial.next_token_id == bulk.next_token_id,
        "logits_exact": logits_equal,
        "logits_max_abs": float(np.max(np.abs(serial.logits - bulk.logits))),
        "logits_sha256": _sha256(bulk.logits),
        "final_hidden_exact": hidden_equal,
        "post_layer_hidden_exact": post_equal,
        "capture_exact": capture_equal,
        "kv_exact": serial.kv_sha256 == bulk.kv_sha256,
        "kv_sha256": bulk.kv_sha256,
    }


def _serial_verifier_snapshot(
    owner: LagunaGGUFResidentSession,
    prefix: tuple[int, ...],
    rows: tuple[int, ...],
    args: argparse.Namespace,
) -> _Snapshot:
    runtime = owner.runtime
    session = _session(owner, args)
    targets, buffers = _capture_targets(runtime, session.config.hidden_size, 1)
    logits = np.empty((len(rows), session.config.vocab_size), dtype=np.float32)
    final_hidden = np.empty((len(rows), session.config.hidden_size), dtype=np.uint16)
    post_hidden = np.empty_like(final_hidden)
    captures = {
        depth: np.empty_like(final_hidden) for depth in LAGUNA_DFLASH_CAPTURE_DEPTHS
    }
    try:
        session.prefill(prefix, use_bulk=False)
        started = time.perf_counter()
        next_token_id = -1
        for row_index, token in enumerate(rows):
            result = session.forward_token(token, captures=targets)
            next_token_id = int(result.next_token_id)
            logits[row_index] = _copy_array(
                runtime,
                result.logits,
                np.float32,
                (session.config.vocab_size,),
            )
            final_hidden[row_index] = _copy_array(
                runtime,
                result.final_hidden,
                np.uint16,
                (session.config.hidden_size,),
            )
            post_hidden[row_index] = _copy_array(
                runtime,
                result.post_layer_hidden,
                np.uint16,
                (session.config.hidden_size,),
            )
            captured = _capture_arrays(runtime, targets)
            for depth in LAGUNA_DFLASH_CAPTURE_DEPTHS:
                captures[depth][row_index] = captured[depth][0]
        elapsed = time.perf_counter() - started
        snapshot = _Snapshot(
            elapsed_seconds=elapsed,
            next_token_id=next_token_id,
            logits=logits,
            final_hidden=final_hidden,
            post_layer_hidden=post_hidden,
            captures=captures,
            kv_sha256=_kv_digest(session),
        )
        return snapshot
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        session.close()


def _bulk_verifier_snapshot(
    owner: LagunaGGUFResidentSession,
    prefix: tuple[int, ...],
    rows: tuple[int, ...],
    args: argparse.Namespace,
) -> _Snapshot:
    runtime = owner.runtime
    session = _session(owner, args)
    targets, buffers = _capture_targets(runtime, session.config.hidden_size, len(rows))
    try:
        session.prefill(prefix, use_bulk=False)
        started = time.perf_counter()
        result = session.verify_rows(rows[0], rows[1:], captures=targets)
        elapsed = time.perf_counter() - started
        logits = _copy_array(
            runtime,
            result.logits,
            np.float32,
            (len(rows), session.config.vocab_size),
        )
        return _Snapshot(
            elapsed_seconds=elapsed,
            next_token_id=int(np.argmax(logits[-1])),
            logits=logits,
            final_hidden=_copy_array(
                runtime,
                result.final_hidden,
                np.uint16,
                (len(rows), session.config.hidden_size),
            ),
            post_layer_hidden=_copy_array(
                runtime,
                result.post_layer_hidden,
                np.uint16,
                (len(rows), session.config.hidden_size),
            ),
            captures=_capture_arrays(runtime, targets),
            kv_sha256=_kv_digest(session),
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        session.close()


def _compare_verifier(serial: _Snapshot, bulk: _Snapshot) -> dict[str, Any]:
    capture_equal = {
        str(depth): bool(np.array_equal(serial.captures[depth], bulk.captures[depth]))
        for depth in LAGUNA_DFLASH_CAPTURE_DEPTHS
    }
    checks = {
        "logits_exact": bool(np.array_equal(serial.logits, bulk.logits)),
        "final_hidden_exact": bool(np.array_equal(serial.final_hidden, bulk.final_hidden)),
        "post_layer_hidden_exact": bool(
            np.array_equal(serial.post_layer_hidden, bulk.post_layer_hidden)
        ),
        "kv_exact": serial.kv_sha256 == bulk.kv_sha256,
        "capture_exact": capture_equal,
    }
    passed = bool(
        checks["logits_exact"]
        and checks["final_hidden_exact"]
        and checks["post_layer_hidden_exact"]
        and checks["kv_exact"]
        and all(capture_equal.values())
    )
    return {
        "pass": passed,
        **checks,
        "logits_max_abs": float(np.max(np.abs(serial.logits - bulk.logits))),
        "logits_sha256": _sha256(bulk.logits),
        "kv_sha256": bulk.kv_sha256,
        "serial_seconds": serial.elapsed_seconds,
        "bulk_seconds": bulk.elapsed_seconds,
        "speedup_diagnostic": serial.elapsed_seconds / bulk.elapsed_seconds,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.lengths or min(args.lengths) <= 0:
        raise ValueError("--lengths must contain positive integers")
    if args.verifier_prefix_length <= 0 or args.verifier_rows <= 0:
        raise ValueError("verifier prefix/rows must be positive")
    if args.safety_reserve_gib <= 0.0:
        raise ValueError("--safety-reserve-gib must be positive")
    sequence = _sequence(args.template, args.oracle)
    required = max(max(args.lengths), args.verifier_prefix_length + args.verifier_rows)
    if required > len(sequence):
        raise ValueError(f"fixtures provide {len(sequence)} tokens but {required} are required")
    runtime = get_hip_runtime()
    tracked_before = memory_stats()
    owner = None
    load_started = time.perf_counter()
    try:
        owner = LagunaGGUFResidentSession(
            args.model,
            context_length=args.context_length,
            backend=args.backend,
            runtime=runtime,
            compiler_version=_compiler_version(args.compiler_version_file),
            require_cached_build=args.require_cached_build,
            safety_reserve_nbytes=int(args.safety_reserve_gib * 2**30),
            progress=_progress,
            repacked_cache=None if args.direct_gguf else args.repacked_cache,
            model_sha256=args.model_sha256,
            prefill_chunk_size=args.chunk_size,
        )
        load_seconds = time.perf_counter() - load_started
        cases = []
        for length in args.lengths:
            tokens = sequence[: int(length)]
            serial = _prefill_snapshot(owner, tokens, use_bulk=False, args=args)
            bulk = _prefill_snapshot(owner, tokens, use_bulk=True, args=args)
            case = _compare_snapshot(serial, bulk, length=int(length))
            cases.append(case)
            print(
                f"length={length} pass={case['pass']} "
                f"serial={case['serial_seconds']:.6f}s bulk={case['bulk_seconds']:.6f}s",
                file=sys.stderr,
                flush=True,
            )

        prefix_start = args.verifier_prefix_length
        prefix = sequence[:prefix_start]
        verifier_tokens = sequence[prefix_start : prefix_start + args.verifier_rows]
        serial_verifier = _serial_verifier_snapshot(
            owner,
            prefix,
            verifier_tokens,
            args,
        )
        bulk_verifier = _bulk_verifier_snapshot(owner, prefix, verifier_tokens, args)
        verifier = _compare_verifier(serial_verifier, bulk_verifier)
        verifier.update(
            {
                "prefix_length": len(prefix),
                "rows": len(verifier_tokens),
                "input_token_ids": list(verifier_tokens),
            }
        )
        resident_nbytes = owner.resident_nbytes
    finally:
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    recovered = bool(
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    passed = bool(all(case["pass"] for case in cases) and verifier["pass"] and recovered)
    try:
        source_revision = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = "unknown"
    return {
        "schema": 1,
        "date": "2026-07-23",
        "status": "accepted" if passed else "rejected",
        "pass": passed,
        "performance_claim": False,
        "scope": "Laguna exact bulk prefill and committed B+1 target rows",
        "source_revision": source_revision,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "quant": args.quant_label,
        },
        "backend": args.backend,
        "context_length": args.context_length,
        "prefill_chunk_size": args.chunk_size,
        "load_seconds": load_seconds,
        "resident_nbytes": resident_nbytes,
        "prefill_cases": cases,
        "verifier": verifier,
        "tracked_before": tracked_before,
        "tracked_after": tracked_after,
        "tracked_returned_to_baseline": recovered,
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "acceptance": {
            "required": (
                "bit-exact logits, final/post-layer hidden, six DFlash taps, and live KV "
                "versus token-serial execution at every declared length and B+1 row"
            ),
            "not_claimed": (
                "retained target throughput, speculative accept/rollback safety, or DFlash economics"
            ),
        },
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
