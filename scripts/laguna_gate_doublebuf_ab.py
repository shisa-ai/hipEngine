#!/usr/bin/env python3
"""Compare exact Laguna prefill runtime modes on one resident model."""

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
    "direct_rollback": "mmq128x32_d8_f32_wavecols_direct",
    "doublebuf_candidate": "mmq128x32_d8_f32_wavecols_direct_doublebuf",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument(
        "--mode",
        action="append",
        metavar="LABEL=RUNTIME_MODE",
        help=(
            "runtime mode arm; repeat for a multi-arm comparison "
            "(defaults to the retained direct/double-buffer pair)"
        ),
    )
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _modes_from_args(values: list[str] | None) -> dict[str, str]:
    if values is None:
        return dict(_MODES)
    modes: dict[str, str] = {}
    for value in values:
        label, separator, runtime_mode = value.partition("=")
        if not separator or not label or not runtime_mode:
            raise ValueError("--mode must use LABEL=RUNTIME_MODE")
        if label in modes:
            raise ValueError(f"duplicate --mode label: {label!r}")
        modes[label] = runtime_mode
    if len(modes) < 2:
        raise ValueError("at least two --mode arms are required")
    return modes


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
    selected_gate_up_mode: str,
    tokens: list[int],
) -> dict[str, float | int | str]:
    child = _child_session(owner)
    try:
        _apply_mode(child, selected_gate_up_mode)
        started = time.perf_counter()
        result = child.prefill(tokens, use_bulk=True)
        child.runtime.device_synchronize()
        elapsed = time.perf_counter() - started
        return {
            "tok_s": 512.0 / elapsed,
            "next_token": int(result.next_token_id),
            "next_token_logit_hex": float(result.next_token_logit).hex(),
            "logits_sha256": _device_digest(
                child.runtime, result.logits, np.float32
            ),
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


def _apply_mode(
    session: LagunaGGUFResidentSession,
    runtime_mode: str,
) -> None:
    if runtime_mode == "q6_wmma_current":
        session.set_q6_wmma_prefetch_weight(False)
    elif runtime_mode == "q6_wmma_prefetch":
        session.set_q6_wmma_prefetch_weight(True)
    else:
        session.set_selected_gate_up_mode(runtime_mode)


def main() -> int:
    args = _parse_args()
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    modes = _modes_from_args(args.mode)
    reader = GGUFReader(DEFAULT_MODEL)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(DEFAULT_PROMPTS, tokenizer)
    token_stream, _ = _profile_token_stream(prompts, 512)
    tokens = list(token_stream)
    runtime = get_hip_runtime()
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
        for gate_up_mode in modes.values():
            warm_session = _child_session(owner)
            try:
                _apply_mode(warm_session, gate_up_mode)
                warm_session.prefill(tokens[:128], use_bulk=True)
                runtime.device_synchronize()
            finally:
                warm_session.close()
        for repetition in range(args.repetitions):
            mode_order = tuple(modes)
            order = tuple(
                mode_order[(repetition + offset) % len(mode_order)]
                for offset in range(len(mode_order))
            )
            for mode in order:
                record = _run(owner, modes[mode], tokens)
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
            "selected_gate_up_mode": modes[mode],
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
