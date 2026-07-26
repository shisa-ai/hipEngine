#!/usr/bin/env python3
"""Compare unfused and exact fused MMQ selected-down combine on pp512."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime import laguna_moe
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_matrix_chunk_bench import _device_digest, _kv_digest
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

_MODES = {
    "unfused_rollback": False,
    "fused_candidate": True,
}
_PRODUCTION_SHOULD_FUSE = laguna_moe._should_fuse_grouped_weighted_sum


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _child_session(
    owner: LagunaGGUFResidentSession,
) -> LagunaGGUFResidentSession:
    assert owner.weights is not None
    return LagunaGGUFResidentSession(
        resident_weights=owner.weights,
        context_length=512,
        backend="hip_gfx1151",
        runtime=owner.runtime,
        compiler_version=_compiler_version(Path("/tmp/laguna_hipcc_version.txt")),
        require_cached_build=True,
        prefill_chunk_size=512,
        prefill_attention_chunk_size=128,
    )


def _run(
    owner: LagunaGGUFResidentSession,
    mode: str,
    tokens: list[int],
) -> dict[str, float | int | str]:
    should_fuse = _MODES[mode]
    prior = laguna_moe._should_fuse_grouped_weighted_sum
    laguna_moe._should_fuse_grouped_weighted_sum = (
        _PRODUCTION_SHOULD_FUSE if should_fuse else lambda **_: False
    )
    child = _child_session(owner)
    try:
        started = time.perf_counter()
        result = child.prefill(tokens, use_bulk=True)
        child.runtime.device_synchronize()
        elapsed = time.perf_counter() - started
        return {
            "tok_s": 512.0 / elapsed,
            "wall_ms": elapsed * 1000.0,
            "next_token": int(result.next_token_id),
            "next_token_logit_hex": float(result.next_token_logit).hex(),
            "logits_sha256": _device_digest(child.runtime, result.logits, np.float32),
            "final_hidden_sha256": _device_digest(
                child.runtime, result.final_hidden, np.uint16
            ),
            "post_layer_hidden_sha256": _device_digest(
                child.runtime, result.post_layer_hidden, np.uint16
            ),
            "kv_sha256": _kv_digest(child),
            "final_position": int(child.position),
        }
    finally:
        child.close()
        laguna_moe._should_fuse_grouped_weighted_sum = prior


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
    records = {mode: [] for mode in _MODES}
    owner = LagunaGGUFResidentSession(
        DEFAULT_MODEL,
        context_length=512,
        backend="hip_gfx1151",
        runtime=runtime,
        compiler_version=_compiler_version(Path("/tmp/laguna_hipcc_version.txt")),
        require_cached_build=args.require_cached_build,
        progress=_progress,
        repacked_cache=DEFAULT_CACHE,
        model_sha256=DEFAULT_MODEL_SHA256,
        prefill_chunk_size=512,
        prefill_attention_chunk_size=128,
    )
    try:
        for mode in _MODES:
            _run(owner, mode, tokens[:128])
        for repetition in range(args.repetitions):
            mode_order = tuple(_MODES)
            order = mode_order if repetition % 2 == 0 else tuple(reversed(mode_order))
            for mode in order:
                record = _run(owner, mode, tokens)
                samples[mode].append(float(record["tok_s"]))
                records[mode].append(record)
                print(
                    mode,
                    repetition,
                    f"{float(record['tok_s']):.6f}",
                    int(record["next_token"]),
                    flush=True,
                )
    finally:
        laguna_moe._should_fuse_grouped_weighted_sum = _PRODUCTION_SHOULD_FUSE
        owner.close()
    state_fields = (
        "next_token",
        "next_token_logit_hex",
        "logits_sha256",
        "final_hidden_sha256",
        "post_layer_hidden_sha256",
        "kv_sha256",
        "final_position",
    )
    exact_pairs = [
        all(
            records["unfused_rollback"][index][field]
            == records["fused_candidate"][index][field]
            for field in state_fields
        )
        for index in range(args.repetitions)
    ]
    result = {
        "schema_version": 1,
        "kind": "hipengine_laguna_mmq_combine_ab",
        "protocol": {
            "rows": 512,
            "repetitions": args.repetitions,
            "timed_order": "counter-rotated",
            "unfused_rollback": (
                "MMQ down -> sorted-lane weighted sum -> shared projections -> BF16 add"
            ),
            "fused_candidate": (
                "MMQ down -> shared projections -> exact sorted-lane weighted sum + shared add"
            ),
        },
        "modes": {
            mode: {
                "samples_tok_s": values,
                "median_tok_s": statistics.median(values),
                "median_wall_ms": statistics.median(
                    float(record["wall_ms"]) for record in records[mode]
                ),
                "records": records[mode],
            }
            for mode, values in samples.items()
        },
        "candidate_wins": sum(
            candidate > rollback
            for rollback, candidate in zip(
                samples["unfused_rollback"], samples["fused_candidate"]
            )
        ),
        "exact_pairs": exact_pairs,
        "pass": all(exact_pairs),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
