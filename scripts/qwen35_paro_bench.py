#!/usr/bin/env python3
"""Actual autoregressive Qwen3.5/PARO resident benchmark harness.

This runs real prompt-token prefill and generated-token decode with persistent
per-layer linear-attention state and full-attention KV cache. By default,
prefill uses ``prefill_native(...)`` with the retained single-request native
path labels. ``--serial-prefill-diagnostic`` is the explicit token-by-token c=1
fallback for correctness/debug comparison and is not a native throughput row.
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

from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.runtime import PrefillConfig
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
        "--attn-aotriton-min-tokens",
        type=int,
        default=0,
        help="Opt into AOTriton per-Q-head full-attention prefill at prompts with at least this many tokens (0 disables).",
    )
    parser.add_argument("--prefill-linear-chunk-size", type=int, default=0, help="Chunk single-request linear-attention prefill layers (0 disables).")
    parser.add_argument("--prefill-moe-chunk-size", type=int, default=0, help="Chunk grouped-MoE prefill scratch/users by limiting layer chunks (0 disables).")
    parser.add_argument("--prefill-full-attn-query-chunk-size", type=int, default=0, help="Chunk single-request full-attention query rows (0 disables).")
    parser.add_argument("--prefill-full-attn-post-chunk-size", type=int, default=0, help="Limit full-attention post/MoE chunk rows (0 disables).")
    parser.add_argument("--prefill-full-attn-rope-chunk-size", type=int, default=0, help="Limit full-attention RoPE/norm chunk rows (0 disables).")
    parser.add_argument(
        "--native-prefill",
        action="store_true",
        help="Deprecated compatibility no-op: native single-request prefill is the default.",
    )
    parser.add_argument(
        "--serial-prefill-diagnostic",
        action="store_true",
        help="Use explicit token-by-token c=1 prompt prefill instead of the native prefill path.",
    )
    parser.add_argument(
        "--allow-rejected-native-prefill",
        action="store_true",
        help="Deprecated compatibility no-op retained for old scripts; rejected native-prefill diagnostics use dedicated correctness helpers.",
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
    if args.attn_aotriton_min_tokens < 0:
        raise ValueError("--attn-aotriton-min-tokens must be non-negative")
    for name in (
        "prefill_linear_chunk_size",
        "prefill_moe_chunk_size",
        "prefill_full_attn_query_chunk_size",
        "prefill_full_attn_post_chunk_size",
        "prefill_full_attn_rope_chunk_size",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.native_prefill and args.serial_prefill_diagnostic:
        raise ValueError("--native-prefill cannot be combined with --serial-prefill-diagnostic")

    model = Path(args.model)
    compiler_version = _read_compiler_version(args.compiler_version_file) if args.compiler_version_file is not None else None
    prompt_tokens = _prompt_tokens(model, args.prompt, args.token_id, args.prompt_length)
    max_sequence = len(prompt_tokens) + args.warmup_decode_tokens + args.decode_tokens + 1

    progress = _emit_progress if args.progress else None
    roctx = _Roctx(enabled=args.roctx)
    runner = Qwen35ParoNextTokenRunner(model)
    reset_memory_stats()
    memory_snapshots: dict[str, Any] = {
        "before_load": _memory_snapshot("before_load", runner.runtime),
    }

    load_start = time.perf_counter()
    with roctx.range("hipengine:resident_build"):
        session = Qwen35ParoResidentSession(
            runner,
            max_sequence_length=max_sequence,
            max_layers=args.max_layers,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            progress=progress,
            prefill_config=PrefillConfig(
                linear_chunk_size=args.prefill_linear_chunk_size,
                moe_chunk_size=args.prefill_moe_chunk_size,
                full_attn_query_chunk_size=args.prefill_full_attn_query_chunk_size,
                full_attn_post_chunk_size=args.prefill_full_attn_post_chunk_size,
                full_attn_rope_chunk_size=args.prefill_full_attn_rope_chunk_size,
                attn_aotriton_min_tokens=args.attn_aotriton_min_tokens,
            ),
        )
    load_seconds = time.perf_counter() - load_start
    memory_snapshots["after_load"] = _memory_snapshot("after_load", session.runtime, session)

    native_prefill_plan = session.native_prefill_plan()
    native_prefill_execution = (
        "serial_c1_prefill_diagnostic" if args.serial_prefill_diagnostic else native_prefill_plan.path
    )

    generated: list[dict[str, Any]] = []
    decode_samples: list[float] = []
    try:
        prefill_start = time.perf_counter()
        next_result = None
        with roctx.range("hipengine:prefill"):
            if args.serial_prefill_diagnostic:
                for pos, token in enumerate(prompt_tokens):
                    with roctx.range("hipengine:prefill_step"):
                        next_result = session.step(token, position=pos, sample=(pos == len(prompt_tokens) - 1))
            else:
                with roctx.range("hipengine:native_prefill_single_request"):
                    _ = (args.native_prefill, args.allow_rejected_native_prefill)
                    next_result = session.prefill_native(prompt_tokens, sample=True)
        prefill_seconds = time.perf_counter() - prefill_start
        if next_result is None:
            raise RuntimeError("prefill did not produce next-token logits")
        memory_snapshots["after_prefill"] = _memory_snapshot("after_prefill", session.runtime, session)
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
        memory_snapshots["after_warmup_decode"] = _memory_snapshot("after_warmup_decode", session.runtime, session)

        decode_start_pos = len(prompt_tokens) + args.warmup_decode_tokens
        if args.graph_replay_decode and args.decode_tokens:
            with roctx.range("hipengine:capture_decode_graph"):
                graph = session.capture_decode_graph(
                    position=decode_start_pos,
                    steps_per_replay=args.graph_steps_per_replay,
                    max_replay_steps=args.decode_tokens,
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
        memory_snapshots["after_decode"] = _memory_snapshot("after_decode", session.runtime, session)
    finally:
        memory_snapshots["before_close"] = _memory_snapshot("before_close", session.runtime, session)
        session.close()
        memory_snapshots["after_close"] = _memory_snapshot("after_close", runner.runtime)

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
        "native_batched_prefill": not bool(args.serial_prefill_diagnostic),
        "native_prefill_execution": native_prefill_execution,
        "native_prefill_plan": native_prefill_plan.to_json_dict(),
        "prefill_execution_detail": getattr(session, "last_prefill_execution", None),
        "serial_prefill_diagnostic": bool(args.serial_prefill_diagnostic),
        "allow_rejected_native_prefill": bool(args.allow_rejected_native_prefill),
        "attn_aotriton_min_tokens": args.attn_aotriton_min_tokens,
        "prefill_chunk_sizes": {
            "linear": args.prefill_linear_chunk_size,
            "moe": args.prefill_moe_chunk_size,
            "full_attn_query": args.prefill_full_attn_query_chunk_size,
            "full_attn_post": args.prefill_full_attn_post_chunk_size,
            "full_attn_rope": args.prefill_full_attn_rope_chunk_size,
        },
        "graph_replay": bool(args.graph_replay_decode),
        "graph_steps_per_replay": args.graph_steps_per_replay if args.graph_replay_decode else 0,
        "prefill_comparable_to_plan_moe2": bool((not args.serial_prefill_diagnostic) and native_prefill_plan.full_layer_limit_native),
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
            "token_by_token_prefill_tok_s": (
                len(prompt_tokens) / prefill_seconds if args.serial_prefill_diagnostic and prefill_seconds > 0 else None
            ),
            "warmed_decode_tok_s": args.decode_tokens / decode_seconds if decode_seconds > 0 and args.decode_tokens else None,
            "warmed_decode_step_median_s": statistics.median(decode_samples) if decode_samples else None,
        },
        "memory": _memory_summary(memory_snapshots),
        "memory_snapshots": memory_snapshots,
        "generated_preview": generated[:16],
        "notes": [
            (
                "Prefill is explicit token-by-token c=1 diagnostic mode, not native prefill."
                if args.serial_prefill_diagnostic
                else f"Prefill execution is {native_prefill_execution} via prefill_native(...); c>N compact slabs remain separate work."
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


def _owned_device_bytes(session: Qwen35ParoResidentSession) -> int:
    allocation_bytes = sum(int(allocation.buffer.nbytes) for allocation in session.allocations)
    buffer_bytes = sum(int(buffer.nbytes) for buffer in session.buffers)
    state_workspace_bytes = sum(
        int(state.workspace.allocation(name).buffer.nbytes)
        for state in session.states
        for name in state.workspace.names
    )
    prefill_workspace = getattr(session, "prefill_workspace", None)
    prefill_workspace_bytes = 0
    if prefill_workspace is not None:
        prefill_workspace_bytes = sum(
            int(prefill_workspace.allocation(name).buffer.nbytes)
            for name in prefill_workspace.names
        )
    return allocation_bytes + buffer_bytes + state_workspace_bytes + prefill_workspace_bytes


def _memory_snapshot(
    label: str,
    runtime,
    session: Qwen35ParoResidentSession | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": label,
        "tracked": memory_stats(),
        "hip": _hip_memory_info(runtime),
    }
    if session is not None:
        payload["owned_session_bytes"] = _owned_device_bytes(session)
        payload["owned_session_gib"] = _bytes_to_gib(payload["owned_session_bytes"])
    return payload


def _hip_memory_info(runtime) -> dict[str, Any]:
    try:
        free_bytes, total_bytes = runtime.mem_get_info()
    except Exception as exc:  # pragma: no cover - exercised only on HIP runtime failures
        return {"available": False, "error": str(exc)}
    used_bytes = total_bytes - free_bytes
    return {
        "available": True,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
        "free_gib": _bytes_to_gib(free_bytes),
        "total_gib": _bytes_to_gib(total_bytes),
        "used_gib": _bytes_to_gib(used_bytes),
    }


def _memory_summary(snapshots: dict[str, Any]) -> dict[str, Any]:
    tracked_peak = max(
        int(snapshot.get("tracked", {}).get("peak_allocated_bytes", 0))
        for snapshot in snapshots.values()
    ) if snapshots else 0
    tracked_current_before_close = int(
        snapshots.get("before_close", {}).get("tracked", {}).get("current_allocated_bytes", 0)
    )
    tracked_current_after_close = int(
        snapshots.get("after_close", {}).get("tracked", {}).get("current_allocated_bytes", 0)
    )
    owned_peak = max(
        int(snapshot.get("owned_session_bytes", 0))
        for snapshot in snapshots.values()
    ) if snapshots else 0
    hip_used_values = [
        int(snapshot.get("hip", {}).get("used_bytes", 0))
        for snapshot in snapshots.values()
        if snapshot.get("hip", {}).get("available")
    ]
    hip_used_peak = max(hip_used_values) if hip_used_values else None
    summary = {
        "tracked_peak_allocated_bytes": tracked_peak,
        "tracked_peak_allocated_gib": _bytes_to_gib(tracked_peak),
        "tracked_current_allocated_bytes_before_close": tracked_current_before_close,
        "tracked_current_allocated_gib_before_close": _bytes_to_gib(tracked_current_before_close),
        "tracked_current_allocated_bytes_after_close": tracked_current_after_close,
        "tracked_current_allocated_gib_after_close": _bytes_to_gib(tracked_current_after_close),
        "owned_session_peak_bytes": owned_peak,
        "owned_session_peak_gib": _bytes_to_gib(owned_peak),
        "hip_used_peak_sampled_bytes": hip_used_peak,
        "hip_used_peak_sampled_gib": _bytes_to_gib(hip_used_peak) if hip_used_peak is not None else None,
        "notes": [
            "tracked_* covers hipENGINE allocations made through hipengine.core.memory.malloc and keeps a high-water mark across freed prefill workspaces.",
            "hip_used_peak_sampled_* is sampled via hipMemGetInfo at phase boundaries, not a continuous device-wide peak.",
            "owned_session_* sums buffers owned by the resident session at each sampled point and excludes external HIP/AOTriton internal allocations.",
        ],
    }
    return summary


def _bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1 << 30)


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
