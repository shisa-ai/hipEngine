#!/usr/bin/env python3
"""Compare exact packed attention input/output boundary candidates."""

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

_COMPARISONS = {
    "packed_output_gate": {
        "production_rollback": {
            "packed_output_gate": False,
            "packed_query_producer": False,
        },
        "packed_output_gate_candidate": {
            "packed_output_gate": True,
            "packed_query_producer": False,
        },
    },
    "packed_query_producer": {
        "production_rollback": {
            "packed_output_gate": True,
            "packed_query_producer": False,
        },
        "packed_query_producer_candidate": {
            "packed_output_gate": True,
            "packed_query_producer": True,
        },
    },
}
_STATE_KEYS = (
    "next_token",
    "next_token_logit_hex",
    "logits_sha256",
    "final_hidden_sha256",
    "post_layer_hidden_sha256",
    "kv_sha256",
    "final_position",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        choices=tuple(_COMPARISONS),
        default="packed_output_gate",
    )
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _child(
    owner: LagunaGGUFResidentSession,
    settings: dict[str, bool],
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
        prefill_attention_hipblaslt=True,
        prefill_attention_hipblaslt_packed_queries=True,
        prefill_attention_hipblaslt_packed_query_producer=(
            settings["packed_query_producer"]
        ),
        prefill_attention_hipblaslt_wave_rows_softmax=True,
        prefill_attention_hipblaslt_packed_output_gate=(
            settings["packed_output_gate"]
        ),
    )


def _run(
    owner: LagunaGGUFResidentSession,
    mode: str,
    tokens: list[int],
    settings: dict[str, bool],
) -> dict[str, float | int | str]:
    child = _child(owner, settings)
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
            "logits_sha256": _device_digest(
                child.runtime,
                result.logits,
                np.float32,
            ),
            "final_hidden_sha256": _device_digest(
                child.runtime,
                result.final_hidden,
                np.uint16,
            ),
            "post_layer_hidden_sha256": _device_digest(
                child.runtime,
                result.post_layer_hidden,
                np.uint16,
            ),
            "kv_sha256": _kv_digest(child),
            "final_position": int(child.position),
        }
    finally:
        child.close()


def main() -> int:
    args = _parse_args()
    reader = GGUFReader(DEFAULT_MODEL)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(DEFAULT_PROMPTS, tokenizer)
    tokens = list(_profile_token_stream(prompts, 512)[0])
    runtime = get_hip_runtime()
    modes = _COMPARISONS[args.comparison]
    samples = {mode: [] for mode in modes}
    records = {mode: [] for mode in modes}
    owner = LagunaGGUFResidentSession(
        DEFAULT_MODEL,
        context_length=512,
        backend="hip_gfx1151",
        runtime=runtime,
        compiler_version=_compiler_version(
            Path("/tmp/laguna_hipcc_version.txt")
        ),
        require_cached_build=args.require_cached_build,
        progress=_progress,
        repacked_cache=DEFAULT_CACHE,
        model_sha256=DEFAULT_MODEL_SHA256,
        prefill_chunk_size=512,
        prefill_attention_chunk_size=128,
    )
    try:
        for settings in modes.values():
            warm = _child(owner, settings)
            try:
                warm.prefill(tokens[:128], use_bulk=True)
                runtime.device_synchronize()
            finally:
                warm.close()
        for repetition in range(args.repetitions):
            mode_names = tuple(modes)
            order = (
                mode_names
                if repetition % 2 == 0
                else tuple(reversed(mode_names))
            )
            for mode in order:
                record = _run(owner, mode, tokens, modes[mode])
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
    result: dict[str, object] = {
        mode: {
            "samples_tok_s": values,
            "median_tok_s": statistics.median(values),
            "records": records[mode],
            **modes[mode],
        }
        for mode, values in samples.items()
    }
    candidate_mode = next(
        mode for mode in modes if mode != "production_rollback"
    )
    rollback = float(result["production_rollback"]["median_tok_s"])
    candidate = float(result[candidate_mode]["median_tok_s"])
    result["comparison"] = {
        "speedup": candidate / rollback,
        "delta_percent": (candidate / rollback - 1.0) * 100.0,
        "candidate_wins": sum(
            candidate_value > rollback_value
            for candidate_value, rollback_value in zip(
                samples[candidate_mode],
                samples["production_rollback"],
                strict=True,
            )
        ),
        "complete_state_exact": all(
            {key: candidate_record[key] for key in _STATE_KEYS}
            == {key: rollback_record[key] for key in _STATE_KEYS}
            for candidate_record, rollback_record in zip(
                records[candidate_mode],
                records["production_rollback"],
                strict=True,
            )
        ),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
