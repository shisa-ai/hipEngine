#!/usr/bin/env python3
"""Measure Poolside request preprocessing and fake-model useful-content TTFT.

The model response is intentionally immediate: this isolates chat rendering,
GGUF tokenization, admission, usage accounting, and SSE emission from GPU model
wall. Run the same script against baseline and candidate revisions with the same
GGUF and host environment; model loading and tensor materialization are absent.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from fastapi.testclient import TestClient

from hipengine.chat.poolside_v1 import (
    PoolsideV1ReasoningParser,
    render_poolside_v1_chat,
)
from hipengine.generation import GenerationOutput
from hipengine.loading.gguf import scan_gguf
from hipengine.server import ServerConfig, create_app
from hipengine.tokenization.gguf import LagunaGGUFTokenizer

_DEFAULT_PROMPTS = Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl")
_SHAPES = (128, 512, 1_024, 4_096)


class _CountingTokenizer:
    def __init__(self, inner: LagunaGGUFTokenizer) -> None:
        self.inner = inner
        self.encode_records: list[tuple[str, float]] = []

    def encode(self, text: str) -> list[int]:
        value = str(text)
        started = time.perf_counter()
        result = self.inner.encode(value)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        self.encode_records.append((value, elapsed_ms))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _ImmediatePoolsideEngine:
    """Poolside protocol/tokenizer owner with an immediate exact fake output."""

    chat_template_family = "poolside_v1"
    reasoning_parser_name = "poolside_v1"
    tool_parser_name = "poolside_v1"
    max_sequence_length = 8_192

    def __init__(self, tokenizer: LagunaGGUFTokenizer) -> None:
        self.tokenizer = _CountingTokenizer(tokenizer)
        self.chat_reasoning_parser = PoolsideV1ReasoningParser(self.tokenizer)

    def render_chat_prompt(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] | None = None,
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        return render_poolside_v1_chat(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            add_generation_prompt=add_generation_prompt,
        )

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(int(token) for token in self.tokenizer.encode(str(text)))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(str(text)))

    def detokenize(
        self,
        token_ids: Sequence[int],
        *,
        skip_special: bool = False,
    ) -> str:
        return self.tokenizer.decode(token_ids, skip_special=skip_special)

    def _consume_prompt(self, prompt: Any) -> tuple[int, ...]:
        if isinstance(prompt, str):
            return self.tokenize(prompt)
        return tuple(int(token) for token in prompt)

    def generate_detailed(self, prompts: Sequence[Any], sampling_params: Any):
        del sampling_params
        return [
            GenerationOutput(text="ready", generated_token_ids=(1,))
            for prompt in prompts
            if self._consume_prompt(prompt)
        ]

    def stream(self, prompt: Any, sampling_params: Any):
        del sampling_params
        self._consume_prompt(prompt)
        yield "ready"



def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"



def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]



def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }



def _synthetic_messages(
    tokenizer: LagunaGGUFTokenizer,
    target_tokens: int,
) -> list[dict[str, str]]:
    empty_prompt = render_poolside_v1_chat(
        [{"role": "user", "content": ""}],
        enable_thinking=False,
    )
    empty_tokens = len(tokenizer.encode(empty_prompt))
    repeat_count = max(1, int(target_tokens) - empty_tokens - 1)
    content = " x" * repeat_count
    prompt = render_poolside_v1_chat(
        [{"role": "user", "content": content}],
        enable_thinking=False,
    )
    actual_tokens = len(tokenizer.encode(prompt))
    if actual_tokens != int(target_tokens):
        raise RuntimeError(
            f"could not construct exact {target_tokens}-token prompt; got {actual_tokens}"
        )
    return [{"role": "user", "content": content}]



def _load_suite(path: Path) -> list[tuple[str, list[dict[str, Any]]]]:
    rows: list[tuple[str, list[dict[str, Any]]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append((str(payload["id"]), list(payload["messages"])))
    if len(rows) != 10:
        raise ValueError(f"expected the canonical ten-prompt suite, got {len(rows)}")
    return rows



def _stream_request(
    client: TestClient,
    payload: dict[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    started = time.perf_counter()
    useful_ms: float | None = None
    terminal_payload: dict[str, Any] = {}
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            terminal_payload = event
            for choice in event.get("choices", ()):
                delta = choice.get("delta") or {}
                if any(delta.get(key) for key in ("content", "reasoning_content", "tool_calls")):
                    if useful_ms is None:
                        useful_ms = (time.perf_counter() - started) * 1_000.0
    end_to_end_ms = (time.perf_counter() - started) * 1_000.0
    if useful_ms is None:
        raise RuntimeError("stream produced no useful-content delta")
    return useful_ms, end_to_end_ms, terminal_payload



def _blocking_request(
    client: TestClient,
    payload: dict[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    started = time.perf_counter()
    response = client.post("/v1/chat/completions", json=payload)
    end_to_end_ms = (time.perf_counter() - started) * 1_000.0
    response.raise_for_status()
    body = response.json()
    if not any(choice.get("message", {}).get("content") for choice in body["choices"]):
        raise RuntimeError("blocking response produced no useful content")
    return end_to_end_ms, end_to_end_ms, body



def _run_case(
    client: TestClient,
    engine: _ImmediatePoolsideEngine,
    messages: list[dict[str, Any]],
    *,
    stream: bool,
    repetitions: int,
    warmups: int,
) -> dict[str, Any]:
    payload = {
        "model": "laguna-preprocess-probe",
        "messages": messages,
        "max_tokens": 1,
        "stream": bool(stream),
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    rendered_prompt = engine.render_chat_prompt(messages)
    request = _stream_request if stream else _blocking_request
    for _ in range(int(warmups)):
        request(client, payload)

    samples: list[dict[str, Any]] = []
    for _ in range(int(repetitions)):
        records_before = len(engine.tokenizer.encode_records)
        useful_ms, end_to_end_ms, body = request(client, payload)
        records = engine.tokenizer.encode_records[records_before:]
        prompt_records = [elapsed for text, elapsed in records if text == rendered_prompt]
        samples.append(
            {
                "prompt_encoder_calls": len(prompt_records),
                "prompt_encoder_ms": sum(prompt_records),
                "other_encoder_calls": len(records) - len(prompt_records),
                "useful_content_ttft_ms": useful_ms,
                "end_to_end_ms": end_to_end_ms,
                "prompt_tokens": int(body.get("usage", {}).get("prompt_tokens", 0)),
            }
        )
    return {
        "samples": samples,
        "prompt_encoder_calls": sorted(
            {int(sample["prompt_encoder_calls"]) for sample in samples}
        ),
        "other_encoder_calls": sorted(
            {int(sample["other_encoder_calls"]) for sample in samples}
        ),
        "prompt_encoder_ms": _summary(
            [sample["prompt_encoder_ms"] for sample in samples]
        ),
        "useful_content_ttft_ms": _summary(
            [sample["useful_content_ttft_ms"] for sample in samples]
        ),
        "end_to_end_ms": _summary([sample["end_to_end_ms"] for sample in samples]),
        "prompt_tokens": sorted({int(sample["prompt_tokens"]) for sample in samples}),
    }



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Laguna GGUF used only for tokenizer metadata")
    parser.add_argument("--prompts", type=Path, default=_DEFAULT_PROMPTS)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repetitions <= 0 or args.warmups < 0:
        parser.error("repetitions must be positive and warmups non-negative")

    tokenizer = LagunaGGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    suite = _load_suite(args.prompts)
    result: dict[str, Any] = {
        "schema": "hipengine.laguna_prepared_prompt_bench.v1",
        "revision": _revision(),
        "scope": "fastapi_testclient_immediate_model",
        "model": str(args.model.expanduser().resolve()),
        "prompts": str(args.prompts),
        "repetitions": int(args.repetitions),
        "warmups": int(args.warmups),
        "modes": {},
    }

    for stream in (False, True):
        mode = "streaming" if stream else "blocking"
        engine = _ImmediatePoolsideEngine(tokenizer)
        app = create_app(
            ServerConfig(
                model=str(args.model),
                served_model_name="laguna-preprocess-probe",
                eager_load=False,
                max_context_tokens=8_192,
            ),
            llm=engine,
        )
        with TestClient(app) as client:
            shapes = {
                str(shape): _run_case(
                    client,
                    engine,
                    _synthetic_messages(tokenizer, shape),
                    stream=stream,
                    repetitions=args.repetitions,
                    warmups=args.warmups,
                )
                for shape in _SHAPES
            }
            suite_rows = []
            for prompt_id, messages in suite:
                row = _run_case(
                    client,
                    engine,
                    messages,
                    stream=stream,
                    repetitions=args.repetitions,
                    warmups=args.warmups,
                )
                row["id"] = prompt_id
                suite_rows.append(row)
            result["modes"][mode] = {"shapes": shapes, "suite": suite_rows}

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
