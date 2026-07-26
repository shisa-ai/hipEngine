#!/usr/bin/env python3
"""Compare production interleaved Q6 qmicro with byte-neutral planar Q6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats
from hipengine.loading.gguf import GGUFReader
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

_MODES = ("current_permute", "planar_candidate")


def _parse_mode_order(value: str) -> tuple[str, str]:
    result = tuple(item.strip() for item in value.split(","))
    if len(result) != 2 or set(result) != set(_MODES):
        raise argparse.ArgumentTypeError(
            "mode order must contain current_permute,planar_candidate"
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument(
        "--mode-order",
        type=_parse_mode_order,
        default=_MODES,
    )
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _run_once(
    owner: LagunaGGUFResidentSession,
    tokens: list[int],
) -> dict[str, float | int | str]:
    owner.reset_state()
    started = time.perf_counter()
    result = owner.prefill(tokens, use_bulk=True)
    owner.runtime.device_synchronize()
    elapsed = time.perf_counter() - started
    return {
        "tok_s": 512.0 / elapsed,
        "elapsed_s": elapsed,
        "next_token": int(result.next_token_id),
        "next_token_logit_hex": float(result.next_token_logit).hex(),
        "logits_sha256": _device_digest(
            owner.runtime,
            result.logits,
            np.float32,
        ),
        "final_hidden_sha256": _device_digest(
            owner.runtime,
            result.final_hidden,
            np.uint16,
        ),
        "post_layer_hidden_sha256": _device_digest(
            owner.runtime,
            result.post_layer_hidden,
            np.uint16,
        ),
        "kv_sha256": _kv_digest(owner),
        "final_position": int(owner.position),
    }


def _run_mode(
    mode: str,
    *,
    args: argparse.Namespace,
    tokens: list[int],
) -> dict:
    planar = mode == "planar_candidate"
    runtime = get_hip_runtime()
    tracked_before = memory_stats()
    load_started = time.perf_counter()
    owner = LagunaGGUFResidentSession(
        DEFAULT_MODEL,
        context_length=4_096,
        backend="hip_gfx1151",
        runtime=runtime,
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached_build=args.require_cached_build,
        progress=_progress,
        repacked_cache=DEFAULT_CACHE,
        model_sha256=DEFAULT_MODEL_SHA256,
        prefill_chunk_size=2_048,
        prefill_attention_chunk_size=128,
        q6_qmicro=True,
        q6_qmicro_planar=planar,
        q6_qmicro_permute=not planar,
    )
    load_seconds = time.perf_counter() - load_started
    try:
        owner.prefill(tokens[:128], use_bulk=True)
        runtime.device_synchronize()
        records = []
        for repetition in range(args.repetitions):
            record = _run_once(owner, tokens)
            records.append(record)
            print(
                mode,
                repetition,
                f"{float(record['tok_s']):.6f}",
                int(record["next_token"]),
                flush=True,
            )
    finally:
        owner.close()
    tracked_after = memory_stats()
    samples = [float(record["tok_s"]) for record in records]
    return {
        "q6_qmicro_planar": planar,
        "q6_qmicro_permute": not planar,
        "load_seconds_excluded": load_seconds,
        "samples_tok_s": samples,
        "median_tok_s": statistics.median(samples),
        "records": records,
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
        },
    }


def main() -> int:
    args = _parse_args()
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    reader = GGUFReader(DEFAULT_MODEL)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(DEFAULT_PROMPTS, tokenizer)
    token_stream, source = _profile_token_stream(prompts, 512)
    tokens = list(token_stream)

    result = {
        mode: _run_mode(mode, args=args, tokens=tokens)
        for mode in args.mode_order
    }
    current = result["current_permute"]
    planar = result["planar_candidate"]
    state_keys = (
        "next_token",
        "next_token_logit_hex",
        "logits_sha256",
        "final_hidden_sha256",
        "post_layer_hidden_sha256",
        "kv_sha256",
        "final_position",
    )
    result["protocol"] = {
        "repetitions": args.repetitions,
        "mode_order": list(args.mode_order),
        "timing_scope": (
            "reset through synchronized first-token projection; "
            "one resident owner per byte layout; load excluded"
        ),
        "matrix_rows": 2_048,
        "attention_rows": 128,
        "token_source": source,
    }
    result["comparison"] = {
        "speedup": (
            float(planar["median_tok_s"])
            / float(current["median_tok_s"])
        ),
        "delta_percent": (
            float(planar["median_tok_s"])
            / float(current["median_tok_s"])
            - 1.0
        )
        * 100.0,
        "complete_state_exact": all(
            {key: candidate_record[key] for key in state_keys}
            == {key: production_record[key] for key in state_keys}
            for candidate_record, production_record in zip(
                planar["records"],
                current["records"],
                strict=True,
            )
        ),
        "all_mode_repeats_deterministic": all(
            len(
                {
                    tuple(record[key] for key in state_keys)
                    for record in result[mode]["records"]
                }
            )
            == 1
            for mode in _MODES
        ),
    }
    rendered = json.dumps(result, indent=2)
    args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if result["comparison"]["complete_state_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
