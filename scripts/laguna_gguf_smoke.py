#!/usr/bin/env python3
"""Run one all-resident eager Laguna S 2.1 token/greedy smoke on gfx11."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.laguna_gguf_runner import (
    LAGUNA_DFLASH_CAPTURE_DEPTHS,
    LagunaGGUFResidentSession,
    LagunaHiddenCaptureTargets,
    load_laguna_eager_libraries,
)

DEFAULT_MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")


def _compiler_version(path: Path | None) -> str | None:
    return None if path is None else path.read_text(encoding="utf-8")


def _progress(completed: int, total: int, spec) -> None:
    if completed == 1 or completed == total or completed % 25 == 0:
        print(
            f"load {completed}/{total}: {spec.source.name} ({spec.layout})",
            file=sys.stderr,
            flush=True,
        )


def run_smoke(
    model: str | Path,
    *,
    backend: str,
    context_length: int,
    prompt_token_ids: tuple[int, ...],
    max_new_tokens: int,
    capture_hidden: bool,
    compiler_version: str | None,
    require_cached: bool,
) -> dict[str, object]:
    runtime = get_hip_runtime()
    tracked_before = memory_stats()
    free_before, total_before = runtime.mem_get_info()
    capture_buffers = []
    session = None
    started = time.perf_counter()
    try:
        session = LagunaGGUFResidentSession(
            model,
            context_length=context_length,
            backend=backend,
            runtime=runtime,
            compiler_version=compiler_version,
            require_cached_build=require_cached,
            progress=_progress,
        )
        loaded_seconds = time.perf_counter() - started
        capture_targets = None
        if capture_hidden:
            row_nbytes = session.config.hidden_size * 2
            capture_buffers = [
                malloc(row_nbytes, runtime=runtime) for _ in LAGUNA_DFLASH_CAPTURE_DEPTHS
            ]
            capture_targets = LagunaHiddenCaptureTargets(
                hidden_size=session.config.hidden_size,
                buffers=dict(zip(LAGUNA_DFLASH_CAPTURE_DEPTHS, capture_buffers, strict=True)),
            )

        inference_started = time.perf_counter()
        first = session.prefill(prompt_token_ids, capture_last=capture_targets)
        generated = [first.next_token_id]
        result = first
        while len(generated) < max_new_tokens:
            result = session.forward_token(result.next_token_id)
            generated.append(result.next_token_id)
        inference_seconds = time.perf_counter() - inference_started
        logits = np.empty(session.config.vocab_size, dtype=np.float32)
        copy_device_to_host(host_array_ptr(logits), result.logits, runtime=runtime)
        logits_finite = bool(np.isfinite(logits).all())

        capture_finite: dict[str, bool] = {}
        if capture_targets is not None:
            for depth, buffer in capture_targets.buffers.items():
                bits = np.empty(session.config.hidden_size, dtype=np.uint16)
                copy_device_to_host(
                    host_array_ptr(bits),
                    buffer,
                    runtime=runtime,
                )
                capture_finite[str(depth)] = bool(np.isfinite(bf16_to_float32(bits)).all())
        return {
            "schema": 1,
            "model": str(model),
            "backend": backend,
            "context_length": context_length,
            "prompt_token_ids": list(prompt_token_ids),
            "generated_token_ids": generated,
            "loaded_seconds": loaded_seconds,
            "inference_seconds": inference_seconds,
            "resident_nbytes": session.resident_nbytes,
            "finite_logits": logits_finite,
            "capture_finite": capture_finite,
            "hip_total_bytes": total_before,
            "hip_free_before": free_before,
            "tracked_before": tracked_before,
        }
    finally:
        for buffer in reversed(capture_buffers):
            free(buffer, runtime=runtime)
        if session is not None:
            session.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4_096)
    parser.add_argument("--prompt-token-id", type=int, action="append", dest="prompt_ids")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--capture-hidden", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--prebuild-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    compiler_version = _compiler_version(args.compiler_version_file)
    if args.prebuild_only:
        libraries = load_laguna_eager_libraries(
            backend=args.backend,
            compiler_version=compiler_version,
            require_cached=args.require_cached_build,
        )
        result: dict[str, object] = {
            "schema": 1,
            "backend": args.backend,
            "prebuilt": True,
            "library_fields": list(libraries.__dataclass_fields__),
        }
    else:
        if args.max_new_tokens <= 0:
            raise ValueError("--max-new-tokens must be positive")
        result = run_smoke(
            args.model,
            backend=args.backend,
            context_length=args.context_length,
            prompt_token_ids=tuple(args.prompt_ids or (2,)),
            max_new_tokens=args.max_new_tokens,
            capture_hidden=args.capture_hidden,
            compiler_version=compiler_version,
            require_cached=args.require_cached_build,
        )
        runtime = get_hip_runtime()
        runtime.device_synchronize()
        result["tracked_after"] = memory_stats()
        result["hip_free_after"] = runtime.mem_get_info()[0]
        result["tracked_returned_to_baseline"] = (
            result["tracked_after"]["current_allocated_bytes"]
            == result["tracked_before"]["current_allocated_bytes"]
            and result["tracked_after"]["active_allocations"]
            == result["tracked_before"]["active_allocations"]
        )
        result["pass"] = (
            bool(result["finite_logits"])
            and all(bool(value) for value in result["capture_finite"].values())
            and bool(result["tracked_returned_to_baseline"])
        )
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if args.prebuild_only or bool(result["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
