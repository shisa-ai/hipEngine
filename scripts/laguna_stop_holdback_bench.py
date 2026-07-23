#!/usr/bin/env python3
"""Measure Laguna useful-content delay from multi-token stop holdback.

This is a deterministic host-only integration probe. A fake resident session
replays fixed token IDs at a configurable decode interval while the production
Laguna streaming path owns stop matching and incremental emission. It uses no
GPU or model weights; the default 61 ms interval approximates the retained
16.384 tok/s Laguna decode rate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from hipengine.generation import GenerationRequest
from hipengine.models.laguna import LAGUNA_GGUF


_TOKENS = [chr(ord("a") + index % 26) for index in range(32)]
_TOKENS[18] = "<think>"
_TOKENS[19] = "</think>"
_TOKENS[23] = "<assistant>"
_TOKENS[24] = "</assistant>"


class _Tokenizer:
    eos_token_id = 2
    eot_token_id = 24
    stop_token_ids = (2, 24)
    tokens = tuple(_TOKENS)
    token_to_id = {token: index for index, token in enumerate(tokens)}
    token_types = tuple(1 for _ in tokens)
    byte_decoder: dict[str, int] = {}

    _text = {10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G"}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del text
        if add_special_tokens:
            raise ValueError("the stop-holdback probe does not add special tokens")
        return [7, 8]

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special: bool = False,
    ) -> str:
        values = []
        for token_id in token_ids:
            value = int(token_id)
            if skip_special and value in self.stop_token_ids:
                continue
            values.append(self._text.get(value, f"T{value}"))
        return "".join(values)


class _Weights:
    def __init__(self) -> None:
        self.config = SimpleNamespace(context_length=4_096)
        self.backend = "hip_gfx1151"

    def free(self, *, runtime: Any = None) -> None:
        del runtime


class _DelayedSession:
    sequence: tuple[int, ...] = ()
    decode_interval_seconds: float = 0.0
    resident_nbytes = 1

    def __init__(self, **kwargs: Any) -> None:
        del kwargs
        self.index = 0

    @staticmethod
    def _result(token_id: int) -> SimpleNamespace:
        return SimpleNamespace(next_token_id=int(token_id), next_token_logit=1.0)

    def prefill(self, token_ids: Sequence[int]) -> SimpleNamespace:
        del token_ids
        self.index = 1
        return self._result(self.sequence[0])

    def forward_token(self, token_id: int) -> SimpleNamespace:
        del token_id
        time.sleep(self.decode_interval_seconds)
        result = self._result(self.sequence[self.index])
        self.index += 1
        return result

    def reset_state(self) -> None:
        self.index = 0

    def close(self) -> None:
        return None



def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"



def _summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "maximum": ordered[-1],
    }



def _request(
    sequence: Sequence[int],
    stop_sequences: Sequence[Sequence[int]],
) -> GenerationRequest:
    return GenerationRequest(
        prompts=((7, 8),),
        max_tokens=len(sequence),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
        stop_token_sequences=tuple(tuple(int(token) for token in row) for row in stop_sequences),
    )



def _sample(
    generator: Any,
    *,
    sequence: tuple[int, ...],
    stop_sequences: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    _DelayedSession.sequence = sequence
    started = time.perf_counter()
    useful_content_ttft_ms: float | None = None
    chunks = []
    for chunk in generator.stream_detailed(_request(sequence, stop_sequences)):
        chunks.append(chunk)
        if chunk.text and useful_content_ttft_ms is None:
            useful_content_ttft_ms = (time.perf_counter() - started) * 1_000.0
    end_to_end_ms = (time.perf_counter() - started) * 1_000.0
    terminal = chunks[-1]
    if terminal.finish_details is None:
        raise RuntimeError("stop-holdback probe produced no terminal metadata")
    return {
        "useful_content_ttft_ms": useful_content_ttft_ms,
        "end_to_end_ms": end_to_end_ms,
        "text": "".join(chunk.text for chunk in chunks),
        "generated_token_ids": list(terminal.generated_token_ids or ()),
        "finish": terminal.finish_details.to_json_dict(),
    }



def _run_workload(
    generator: Any,
    *,
    sequence: tuple[int, ...],
    stop_sequences: tuple[tuple[int, ...], ...],
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    for _ in range(int(warmups)):
        _sample(generator, sequence=sequence, stop_sequences=stop_sequences)
    samples = [
        _sample(generator, sequence=sequence, stop_sequences=stop_sequences)
        for _ in range(int(repetitions))
    ]
    useful = [
        float(sample["useful_content_ttft_ms"])
        for sample in samples
        if sample["useful_content_ttft_ms"] is not None
    ]
    return {
        "sequence": list(sequence),
        "stop_sequences": [list(row) for row in stop_sequences],
        "samples": samples,
        "useful_content_ttft_ms": None if not useful else _summary(useful),
        "end_to_end_ms": _summary([sample["end_to_end_ms"] for sample in samples]),
        "texts": sorted({sample["text"] for sample in samples}),
        "generated_token_ids": sorted(
            {tuple(sample["generated_token_ids"]) for sample in samples}
        ),
        "finishes": [
            json.loads(value)
            for value in sorted(
                {json.dumps(sample["finish"], sort_keys=True) for sample in samples}
            )
        ],
    }



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-interval-ms", type=float, default=61.0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.decode_interval_ms < 0.0:
        parser.error("decode-interval-ms must be non-negative")
    if args.warmups < 0 or args.repetitions <= 0:
        parser.error("warmups must be non-negative and repetitions positive")

    from hipengine.generation import laguna_gguf

    original_tokenizer_factory = laguna_gguf.LagunaGGUFTokenizer.__dict__["from_gguf_info"]
    original_materialize = laguna_gguf.materialize_laguna_gguf_weights
    original_session = laguna_gguf.LagunaGGUFResidentSession
    original_runtime = laguna_gguf.get_hip_runtime
    _DelayedSession.decode_interval_seconds = float(args.decode_interval_ms) / 1_000.0
    generator = None
    try:
        laguna_gguf.LagunaGGUFTokenizer.from_gguf_info = classmethod(
            lambda cls, info: _Tokenizer()
        )
        laguna_gguf.materialize_laguna_gguf_weights = lambda path, **kwargs: _Weights()
        laguna_gguf.LagunaGGUFResidentSession = _DelayedSession
        laguna_gguf.get_hip_runtime = lambda: SimpleNamespace(device_synchronize=lambda: None)
        with tempfile.TemporaryDirectory(prefix="hipengine-laguna-stop-") as directory:
            model = Path(directory) / "laguna.gguf"
            model.touch()
            model.with_suffix(".hipengine-repacked-v1").mkdir()
            generator = laguna_gguf.LagunaGGUFGenerator(
                model_path=model,
                weight_index=SimpleNamespace(metadata={}),
                model_plugin=LAGUNA_GGUF,
                backend="hip_gfx1151",
            )
            workloads = {
                "nonmatching_first_token": ((10, 11, 12, 13), ((13, 14, 15, 16),)),
                "failed_prefix_after_one": ((13, 10, 11, 12), ((13, 14, 15, 16),)),
                "exact_stop": ((13, 14, 15, 16), ((13, 14, 15, 16),)),
            }
            results = {
                name: _run_workload(
                    generator,
                    sequence=sequence,
                    stop_sequences=stops,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                )
                for name, (sequence, stops) in workloads.items()
            }
    finally:
        if generator is not None:
            generator.close()
        laguna_gguf.LagunaGGUFTokenizer.from_gguf_info = original_tokenizer_factory
        laguna_gguf.materialize_laguna_gguf_weights = original_materialize
        laguna_gguf.LagunaGGUFResidentSession = original_session
        laguna_gguf.get_hip_runtime = original_runtime

    result = {
        "schema": "hipengine.laguna_stop_holdback_bench.v1",
        "revision": _revision(),
        "scope": "host_only_delayed_fake_resident_session",
        "decode_interval_ms": float(args.decode_interval_ms),
        "warmups": int(args.warmups),
        "repetitions": int(args.repetitions),
        "workloads": results,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
