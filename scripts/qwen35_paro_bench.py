#!/usr/bin/env python3
"""Actual autoregressive Qwen3.5/PARO resident benchmark harness.

This runs real prompt-token prefill and generated-token decode with persistent
per-layer linear-attention state and full-attention KV cache. By default,
prefill is token-by-token c=1. ``--native-prefill`` requests a diagnostic native
batched prefill only for linear-attention-only layer prefixes; because the
current helper has rejected correctness vs serial c=1, it also requires
``--allow-rejected-native-prefill`` and is not a throughput path.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--token-id", type=int, default=9707, help="Repeated token id for fixed-length prompt")
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--warmup-decode-tokens", type=int, default=1)
    parser.add_argument("--max-layers", type=int, default=0, help="Debug limit; 0 means all layers")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--roctx", action="store_true", help="Emit ROCTX ranges for profiler correlation")
    parser.add_argument("--graph-replay-decode", action="store_true", help="Replay measured decode with a captured HIP graph")
    parser.add_argument("--graph-steps-per-replay", type=int, default=1, help="Decode token steps captured per graph replay")
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=None,
        help="Read precomputed hipcc --version text so profiled runs do not spawn hipcc.",
    )
    parser.add_argument(
        "--require-cached-build",
        action="store_true",
        help="Fail instead of invoking hipcc if any resident HIP library is missing from cache.",
    )
    parser.add_argument(
        "--native-prefill",
        action="store_true",
        help="Request diagnostic native batched prefill for a linear-attention-only layer prefix.",
    )
    parser.add_argument(
        "--allow-rejected-native-prefill",
        action="store_true",
        help="Opt into the current rejected_correctness native prefill helper for blocker/profiling diagnostics only.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    if args.prompt_length <= 0:
        raise ValueError("--prompt-length must be positive")
    if args.decode_tokens < 0 or args.warmup_decode_tokens < 0:
        raise ValueError("decode token counts must be non-negative")
    if args.graph_steps_per_replay <= 0:
        raise ValueError("--graph-steps-per-replay must be positive")
    if args.graph_replay_decode and args.decode_tokens and (args.decode_tokens % args.graph_steps_per_replay) != 0:
        raise ValueError("--decode-tokens must be divisible by --graph-steps-per-replay")
    if args.allow_rejected_native_prefill and not args.native_prefill:
        raise ValueError("--allow-rejected-native-prefill requires --native-prefill")
    if args.native_prefill and not args.allow_rejected_native_prefill:
        raise ValueError(
            "--native-prefill is currently rejected_correctness vs serial c=1; "
            "add --allow-rejected-native-prefill only for diagnostic blocker artifacts"
        )

    model = Path(args.model)
    compiler_version = _read_compiler_version(args.compiler_version_file) if args.compiler_version_file is not None else None
    prompt_tokens = _prompt_tokens(model, args.prompt, args.token_id, args.prompt_length)
    max_sequence = len(prompt_tokens) + args.warmup_decode_tokens + args.decode_tokens + 1

    progress = _emit_progress if args.progress else None
    roctx = _Roctx(enabled=args.roctx)
    runner = Qwen35ParoNextTokenRunner(model)

    load_start = time.perf_counter()
    with roctx.range("hipengine:resident_build"):
        session = Qwen35ParoResidentSession(
            runner,
            max_sequence_length=max_sequence,
            max_layers=args.max_layers,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            progress=progress,
        )
    load_seconds = time.perf_counter() - load_start

    generated: list[dict[str, Any]] = []
    decode_samples: list[float] = []
    try:
        prefill_start = time.perf_counter()
        next_result = None
        with roctx.range("hipengine:prefill"):
            if args.native_prefill:
                with roctx.range("hipengine:native_prefill_batch"):
                    next_result = session.prefill_linear_tokens_native(
                        prompt_tokens,
                        sample=True,
                        allow_rejected_correctness=args.allow_rejected_native_prefill,
                    )
            else:
                for pos, token in enumerate(prompt_tokens):
                    with roctx.range("hipengine:prefill_step"):
                        next_result = session.step(token, position=pos, sample=(pos == len(prompt_tokens) - 1))
        prefill_seconds = time.perf_counter() - prefill_start
        if next_result is None:
            raise RuntimeError("prefill did not produce next-token logits")
        next_token = next_result.token_id
        generated.append(next_result.to_json_dict())

        warmup_start = time.perf_counter()
        with roctx.range("hipengine:warmup_decode"):
            for offset in range(args.warmup_decode_tokens):
                with roctx.range("hipengine:warmup_decode_step"):
                    result = session.step(next_token, position=len(prompt_tokens) + offset, sample=True)
                if result is None:
                    raise RuntimeError("decode warmup did not produce a token")
                next_token = result.token_id
        warmup_seconds = time.perf_counter() - warmup_start

        decode_start_pos = len(prompt_tokens) + args.warmup_decode_tokens
        if args.graph_replay_decode and args.decode_tokens:
            with roctx.range("hipengine:capture_decode_graph"):
                graph = session.capture_decode_graph(
                    position=decode_start_pos,
                    steps_per_replay=args.graph_steps_per_replay,
                )
            try:
                decode_start = time.perf_counter()
                with roctx.range("hipengine:measured_decode_graph"):
                    graph.replay(args.decode_tokens)
                decode_seconds = time.perf_counter() - decode_start
                result = graph.read_sample()
                avg_step = decode_seconds / args.decode_tokens
                decode_samples.extend([avg_step] * args.decode_tokens)
                next_token = result.token_id
                generated.append(result.to_json_dict())
            finally:
                graph.close()
        else:
            decode_start = time.perf_counter()
            with roctx.range("hipengine:measured_decode"):
                for offset in range(args.decode_tokens):
                    step_start = time.perf_counter()
                    with roctx.range("hipengine:measured_decode_step"):
                        result = session.step(next_token, position=decode_start_pos + offset, sample=True)
                    step_seconds = time.perf_counter() - step_start
                    if result is None:
                        raise RuntimeError("decode step did not produce a token")
                    decode_samples.append(step_seconds)
                    next_token = result.token_id
                    generated.append(result.to_json_dict())
            decode_seconds = time.perf_counter() - decode_start
    finally:
        session.close()

    output = {
        "schema": 1,
        "model": str(model),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": "actual_autoregressive_resident",
        "prompt_source": "repeated_token_id" if args.token_id is not None else "prompt_tokenized_repeat",
        "prompt": args.prompt,
        "prompt_length": len(prompt_tokens),
        "decode_tokens": args.decode_tokens,
        "warmup_decode_tokens": args.warmup_decode_tokens,
        "max_layers": args.max_layers or runner.config.num_hidden_layers,
        "tokens_per_step": 1,
        "native_batched_prefill": bool(args.native_prefill),
        "allow_rejected_native_prefill": bool(args.allow_rejected_native_prefill),
        "graph_replay": bool(args.graph_replay_decode),
        "graph_steps_per_replay": args.graph_steps_per_replay if args.graph_replay_decode else 0,
        "prefill_comparable_to_plan_moe2": False,
        "decode_comparable_to_plan_moe2": "graph_replay_diagnostic" if args.graph_replay_decode else "partial_no_graph_replay",
        "timings": {
            "load_seconds": load_seconds,
            "prefill_seconds": prefill_seconds,
            "warmup_decode_seconds": warmup_seconds,
            "decode_seconds": decode_seconds,
            "decode_step_seconds": decode_samples,
        },
        "throughput": {
            "prefill_tok_s": len(prompt_tokens) / prefill_seconds if prefill_seconds > 0 else None,
            "token_by_token_prefill_tok_s": (None if args.native_prefill else (len(prompt_tokens) / prefill_seconds if prefill_seconds > 0 else None)),
            "warmed_decode_tok_s": args.decode_tokens / decode_seconds if decode_seconds > 0 and args.decode_tokens else None,
            "warmed_decode_step_median_s": statistics.median(decode_samples) if decode_samples else None,
        },
        "generated_preview": generated[:16],
        "notes": [
            (
                "Prefill uses the rejected_correctness native linear-prefix helper under explicit diagnostic opt-in; not compact/grouped WMMA and not a throughput claim."
                if args.native_prefill
                else "Prefill is actual autoregressive token-by-token c=1, not native batched/compact prefill."
            ),
            (
                f"Measured decode uses HIP graph replay ({args.graph_steps_per_replay} step(s) per replay) with device token/position state."
                if args.graph_replay_decode
                else "Decode uses persistent per-layer state/KV and GPU lm-head/argmax, but no graph replay yet."
            ),
        ],
    }
    text = json.dumps(output, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0


class _Roctx:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._lib = None
        if not self.enabled:
            return
        try:
            self._lib = ctypes.CDLL("libroctx64.so")
            self._lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
            self._lib.roctxRangePushA.restype = ctypes.c_int
            self._lib.roctxRangePop.argtypes = []
            self._lib.roctxRangePop.restype = ctypes.c_int
        except OSError as exc:
            print(f"warning: --roctx requested but libroctx64.so could not be loaded: {exc}", file=sys.stderr)
            self._lib = None

    def range(self, name: str) -> "_RoctxRange":
        return _RoctxRange(self, name)

    def push(self, name: str) -> None:
        if self._lib is not None:
            self._lib.roctxRangePushA(name.encode("utf-8"))

    def pop(self) -> None:
        if self._lib is not None:
            self._lib.roctxRangePop()


class _RoctxRange:
    def __init__(self, roctx: _Roctx, name: str) -> None:
        self.roctx = roctx
        self.name = name

    def __enter__(self) -> None:
        self.roctx.push(self.name)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.roctx.pop()


def _read_compiler_version(path: Path) -> str:
    text = path.read_text()
    if not text.strip():
        raise ValueError(f"compiler version file {path} is empty")
    return text


def _prompt_tokens(model: Path, prompt: str, token_id: int | None, prompt_length: int) -> list[int]:
    if token_id is not None:
        return [int(token_id)] * prompt_length
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    ids = [int(x) for x in tokenizer.encode(prompt).ids]
    if not ids:
        raise ValueError("prompt produced no tokens")
    out: list[int] = []
    while len(out) < prompt_length:
        out.extend(ids)
    return out[:prompt_length]


def _emit_progress(payload: dict[str, Any]) -> None:
    event = payload.get("event", "progress")
    layer = payload.get("layer")
    prefix = f"layer {layer}: " if layer is not None else ""
    if event in {"resident_build_start", "resident_build_done"}:
        msg = f"{event} layers={payload.get('layers')} max_sequence_length={payload.get('max_sequence_length', '')}"
    elif event in {"materialize_layer_start", "materialize_layer_done", "layer_start", "layer_done"}:
        msg = f"{prefix}{event} {payload.get('type')}"
    elif event == "expert_stack_progress":
        msg = f"{prefix}stack {payload.get('proj')}.{payload.get('suffix')} {payload.get('expert')}/{payload.get('total')}"
    elif event in {"materialize_tensor_start", "materialize_prepared_tensor_start"}:
        msg = f"{prefix}{event} {payload.get('index')}/{payload.get('total')} {payload.get('name')}"
    else:
        msg = event
    print(msg, file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
