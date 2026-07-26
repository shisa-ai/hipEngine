#!/usr/bin/env python3
"""Compare exact global qrow4 and qrow6 cached-metadata prefill policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from hipengine.core.hip import get_hip_runtime
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

_MODES = {
    "global_qrow4_rollback": False,
    "global_qrow6_candidate": True,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _child_session(
    owner: LagunaGGUFResidentSession,
    *,
    prefill_global_qrow6: bool,
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
        prefill_cached_meta=True,
        prefill_global_qrow6=prefill_global_qrow6,
    )


def _run(
    owner: LagunaGGUFResidentSession,
    mode: str,
    tokens: list[int],
) -> dict[str, float | int | str]:
    child = _child_session(owner, prefill_global_qrow6=_MODES[mode])
    try:
        started = time.perf_counter()
        result = child.prefill(tokens, use_bulk=True)
        child.runtime.device_synchronize()
        elapsed = time.perf_counter() - started
        return {
            "tok_s": 512.0 / elapsed,
            "elapsed_s": elapsed,
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
        for prefill_global_qrow6 in _MODES.values():
            warm_session = _child_session(
                owner,
                prefill_global_qrow6=prefill_global_qrow6,
            )
            try:
                warm_session.prefill(tokens[:128], use_bulk=True)
                runtime.device_synchronize()
            finally:
                warm_session.close()
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
        owner.close()
    result = {
        mode: {
            "samples_tok_s": values,
            "median_tok_s": statistics.median(values),
            "records": records[mode],
            "prefill_cached_meta": True,
            "prefill_global_qrow6": _MODES[mode],
        }
        for mode, values in samples.items()
    }
    rollback = float(result["global_qrow4_rollback"]["median_tok_s"])
    candidate = float(result["global_qrow6_candidate"]["median_tok_s"])
    result["comparison"] = {
        "speedup": candidate / rollback,
        "delta_percent": (candidate / rollback - 1.0) * 100.0,
        "candidate_wins": sum(
            candidate_value > rollback_value
            for candidate_value, rollback_value in zip(
                samples["global_qrow6_candidate"],
                samples["global_qrow4_rollback"],
                strict=True,
            )
        ),
        "complete_state_exact": all(
            {
                key: candidate_record[key]
                for key in (
                    "next_token",
                    "next_token_logit_hex",
                    "logits_sha256",
                    "final_hidden_sha256",
                    "post_layer_hidden_sha256",
                    "kv_sha256",
                    "final_position",
                )
            }
            == {
                key: rollback_record[key]
                for key in (
                    "next_token",
                    "next_token_logit_hex",
                    "logits_sha256",
                    "final_hidden_sha256",
                    "post_layer_hidden_sha256",
                    "kv_sha256",
                    "final_position",
                )
            }
            for candidate_record, rollback_record in zip(
                records["global_qrow6_candidate"],
                records["global_qrow4_rollback"],
                strict=True,
            )
        ),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
