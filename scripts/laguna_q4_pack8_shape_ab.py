#!/usr/bin/env python3
"""Compare Laguna's gfx1151 Q4 pack8 shape policy with one fixed tile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

from hipengine.core.hip import get_hip_runtime
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_prefill_profile import _profile_token_stream
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_PROMPTS,
    _compiler_version,
    _load_prompts,
    _progress,
)

_TILE_ENV = "HIPENGINE_GGUF_Q4_K_DENSE_WMMA_TILE"
_MODES = {
    "q4_64x16_rollback": "64x16",
    "selector_unset_q4_shape_policy": None,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _select(mode: str) -> None:
    tile = _MODES[mode]
    if tile is None:
        os.environ.pop(_TILE_ENV, None)
    else:
        os.environ[_TILE_ENV] = tile


def _child_session(
    owner: LagunaGGUFResidentSession,
) -> LagunaGGUFResidentSession:
    assert owner.weights is not None
    return LagunaGGUFResidentSession(
        resident_weights=owner.weights,
        context_length=512,
        backend="hip_gfx1151",
        runtime=owner.runtime,
        compiler_version=_compiler_version(
            Path("/tmp/laguna_hipcc_version.txt")
        ),
        require_cached_build=True,
        prefill_chunk_size=512,
        prefill_attention_chunk_size=128,
    )


def _run(
    owner: LagunaGGUFResidentSession,
    mode: str,
    tokens: list[int],
) -> tuple[float, int]:
    _select(mode)
    child = _child_session(owner)
    try:
        started = time.perf_counter()
        result = child.prefill(tokens, use_bulk=True)
        child.runtime.device_synchronize()
        elapsed = time.perf_counter() - started
        return 512.0 / elapsed, int(result.next_token_id)
    finally:
        child.close()


def main() -> int:
    args = _parse_args()
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    reader = GGUFReader(DEFAULT_MODEL)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(DEFAULT_PROMPTS, tokenizer)
    token_stream, _ = _profile_token_stream(prompts, 512)
    tokens = list(token_stream)
    runtime = get_hip_runtime()
    samples = {mode: [] for mode in _MODES}
    next_tokens = {mode: [] for mode in _MODES}
    owner = LagunaGGUFResidentSession(
        DEFAULT_MODEL,
        context_length=512,
        backend="hip_gfx1151",
        runtime=runtime,
        compiler_version=_compiler_version(
            Path("/tmp/laguna_hipcc_version.txt")
        ),
        require_cached_build=False,
        progress=_progress,
        repacked_cache=DEFAULT_CACHE,
        model_sha256=DEFAULT_MODEL_SHA256,
        prefill_chunk_size=512,
        prefill_attention_chunk_size=128,
    )
    try:
        for mode in _MODES:
            _select(mode)
            warm_session = _child_session(owner)
            try:
                warm_session.prefill(tokens[:128], use_bulk=True)
                runtime.device_synchronize()
            finally:
                warm_session.close()
        for repetition in range(args.repetitions):
            mode_order = tuple(_MODES)
            order = (
                mode_order
                if repetition % 2 == 0
                else tuple(reversed(mode_order))
            )
            for mode in order:
                tok_s, next_token = _run(owner, mode, tokens)
                samples[mode].append(tok_s)
                next_tokens[mode].append(next_token)
                print(
                    mode,
                    repetition,
                    f"{tok_s:.6f}",
                    next_token,
                    flush=True,
                )
    finally:
        owner.close()
        os.environ.pop(_TILE_ENV, None)
    result = {
        mode: {
            "samples_tok_s": values,
            "median_tok_s": statistics.median(values),
            "next_tokens": next_tokens[mode],
            "q4_dense_wmma_tile_override": _MODES[mode],
        }
        for mode, values in samples.items()
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
