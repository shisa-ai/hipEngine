#!/usr/bin/env python3
"""Capture and replay PARO prefill convolution domains around a full-attention boundary.

The production 4K profile shows the same 256-row convolution dispatch running
quickly in linear layers 0-2 and much more slowly after full-attention layer 3.
This diagnostic captures the first convolution chunk from one linear layer on
each side of that boundary, reconstructs the FP32 accumulator immediately
before SiLU, and replays the unchanged production kernel over a 2x2 cross of
captured activation and weight layers.

The capture synchronizations are intentionally outside any performance claim.
Only the isolated HIP-event replay timings are compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.linear_attn.conv import qwen35_linear_attn_conv_prefill_fp16
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro import Qwen35ParoDecodeState
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_paro_bench import _prompt_tokens, _read_compiler_version

DEFAULT_MODEL = Path(
    "/home/lhl/.cache/huggingface/hub/"
    "models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/"
    "snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1"
)


@dataclass(frozen=True)
class ConvCapture:
    layer_id: int
    qkv: np.ndarray
    conv_state: np.ndarray
    conv_weight: np.ndarray
    gpu_output: np.ndarray | None

    @property
    def tokens(self) -> int:
        return int(self.qkv.shape[0])

    @property
    def channels(self) -> int:
        return int(self.qkv.shape[1])

    @property
    def kernel_size(self) -> int:
        return int(self.conv_weight.shape[1])


@dataclass
class ReplayBuffers:
    qkv: DeviceBuffer
    conv_state: DeviceBuffer
    conv_weight: DeviceBuffer
    output: DeviceBuffer

    def close(self, runtime) -> None:
        for buffer in (self.output, self.conv_weight, self.conv_state, self.qkv):
            free(buffer, runtime=runtime)


class CaptureBoundaryReached(RuntimeError):
    """Internal control flow used to stop a diagnostic before a target conv."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("hip_gfx1100", "hip_gfx1151"), default="hip_gfx1151")
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--before-layer", type=int, default=2)
    parser.add_argument("--after-layer", type=int, default=4)
    parser.add_argument(
        "--stop-before-layer",
        type=int,
        default=None,
        help="stop prefill after capturing this layer's pre-conv inputs",
    )
    parser.add_argument(
        "--stop-after-full-stage",
        choices=("kv_append", "aotriton", "o_projection", "post_norm", "moe"),
        default=None,
        help="stop after this stage in the first full-attention layer following --before-layer",
    )
    parser.add_argument(
        "--aotriton-stream",
        choices=("default", "isolated"),
        default="default",
        help="run the captured AOTriton boundary on stream 0 or an isolated nonblocking stream",
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--burst-repeats", type=int, default=64)
    parser.add_argument(
        "--replay-stream",
        choices=("default", "fresh", "default_then_fresh", "fresh_then_default"),
        default="default",
        help="replay on legacy stream 0, a fresh nonblocking stream, or both in the stated order",
    )
    parser.add_argument(
        "--pre-replay-delay-seconds",
        type=float,
        default=0.0,
        help="idle on the host after the capture boundary and before replay",
    )
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="write --json without printing the full artifact")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.prompt_length <= 0:
        raise ValueError("--prompt-length must be positive")
    if args.before_layer < 0 or args.after_layer < 0 or args.before_layer == args.after_layer:
        raise ValueError("capture layers must be distinct non-negative integers")
    if args.stop_before_layer not in (None, args.before_layer, args.after_layer):
        raise ValueError("--stop-before-layer must match --before-layer or --after-layer")
    if args.stop_before_layer is not None and args.stop_after_full_stage is not None:
        raise ValueError("--stop-before-layer and --stop-after-full-stage are mutually exclusive")
    if args.stop_after_full_stage is not None and args.after_layer <= 0:
        raise ValueError("--after-layer must follow a non-negative full-attention layer")
    if args.aotriton_stream == "isolated" and args.stop_after_full_stage != "aotriton":
        raise ValueError("--aotriton-stream isolated requires --stop-after-full-stage aotriton")
    if args.warmups < 0 or args.repetitions <= 0 or args.burst_repeats <= 0:
        raise ValueError("--warmups must be non-negative and repetitions/burst-repeats must be positive")
    if args.pre_replay_delay_seconds < 0:
        raise ValueError("--pre-replay-delay-seconds must be non-negative")

    compiler_version = _read_compiler_version(args.compiler_version_file) if args.compiler_version_file else None
    capture_layer_ids = (
        (args.before_layer,)
        if args.stop_before_layer == args.before_layer or args.stop_after_full_stage is not None
        else (args.before_layer, args.after_layer)
    )
    captures, session, runner, isolated_aotriton_stream = _capture_layers(
        model=args.model,
        backend=args.backend,
        prompt_length=args.prompt_length,
        token_id=args.token_id,
        layer_ids=capture_layer_ids,
        stop_before_layer=args.stop_before_layer,
        stop_after_full_stage=args.stop_after_full_stage,
        aotriton_stream=args.aotriton_stream,
        full_layer_id=args.after_layer - 1,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )
    try:
        fresh_stream = 0 if args.replay_stream == "default" else session.runtime.stream_create()
        stream_order = {
            "default": (("default", 0),),
            "fresh": (("fresh", fresh_stream),),
            "default_then_fresh": (("default", 0), ("fresh", fresh_stream)),
            "fresh_then_default": (("fresh", fresh_stream), ("default", 0)),
        }[args.replay_stream]
        try:
            if args.pre_replay_delay_seconds:
                time.sleep(args.pre_replay_delay_seconds)
            replay = _run_replay_matrix(
                captures,
                layer_ids=capture_layer_ids,
                session=session,
                warmups=args.warmups,
                repetitions=args.repetitions,
                burst_repeats=args.burst_repeats,
                streams=stream_order,
            )
        finally:
            if fresh_stream:
                session.runtime.stream_destroy(fresh_stream)
        capture_json = {
            str(layer_id): _capture_summary(capture)
            for layer_id, capture in sorted(captures.items())
        }
        output = {
            "schema": 1,
            "status": "diagnostic",
            "performance_claim": False,
            "correctness_claim": False,
            "mode": "paro_prefill_conv_domain",
            "workload": {
                "model": str(args.model.resolve()),
                "quant": "w4_paro",
                "prompt_length": int(args.prompt_length),
                "token_id": int(args.token_id),
                "capture_layers": [int(layer_id) for layer_id in capture_layer_ids],
                "stop_before_layer": None if args.stop_before_layer is None else int(args.stop_before_layer),
                "stop_after_full_stage": args.stop_after_full_stage,
                "aotriton_stream": args.aotriton_stream,
                "captured_chunk_rows": int(captures[args.before_layer].tokens),
                "pre_replay_delay_seconds": float(args.pre_replay_delay_seconds),
            },
            "resolved": {
                "backend": runner.backend,
                "target_arch": runner.target_arch,
                "replay_stream_order": [name for name, _stream in stream_order],
                "prefill_chunks": {
                    "linear": session.prefill_config.linear_chunk_size,
                    "moe": session.prefill_config.moe_chunk_size,
                    "full_attn_query": session.prefill_config.full_attn_query_chunk_size,
                    "full_attn_post": session.prefill_config.full_attn_post_chunk_size,
                    "full_attn_rope": session.prefill_config.full_attn_rope_chunk_size,
                },
            },
            "captures": capture_json,
            "replay": replay,
            "provenance": collect_artifact_provenance(
                repo_root=REPO_ROOT,
                configured_backend=args.backend,
                resolved_backend=runner.backend,
                target_arch=runner.target_arch,
                model_path=args.model,
                quant="w4_paro",
                kv_dtype="bf16",
                command=sys.argv,
                environment={
                    "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
                    "tuned_profile": "accelerator-performance",
                },
                build_profile="production linear-attention conv replay",
                timing_protocol=(
                    "capture first exact 256-row chunk at layers around first full-attention boundary; "
                    "HIP-event round-robin queued-burst 2x2 input/weight replay"
                ),
                warmups=args.warmups,
                repetitions=args.repetitions,
                profiler={"enabled": False, "reason": "HIP-event replay; production kernel identity is already traced"},
                hipcc_version=compiler_version,
            ),
            "notes": [
                "Capture copies and stream synchronizations are diagnostic and excluded from replay timings.",
                "The 2x2 replay separates activation-layer input from layer-specific convolution weights.",
                "The second replay phase frees the session prefill workspace while keeping the same replay buffers live.",
                "Accumulator histograms are computed with ordered FP32 products/adds matching the four-tap kernel structure.",
            ],
        }
    finally:
        if isolated_aotriton_stream:
            session.runtime.stream_destroy(isolated_aotriton_stream)
        session.close()

    text = json.dumps(output, indent=2, sort_keys=True)
    if not args.quiet:
        print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


def conv_accumulators(
    qkv: np.ndarray,
    conv_state: np.ndarray,
    conv_weight: np.ndarray,
) -> np.ndarray:
    """Reconstruct ordered FP32 convolution accumulators before SiLU."""

    qkv_f32 = np.asarray(qkv, dtype=np.float16).astype(np.float32)
    if qkv_f32.ndim != 2:
        raise ValueError("qkv must be a [tokens, channels] matrix")
    tokens, channels = qkv_f32.shape
    state_f32 = np.asarray(conv_state, dtype=np.float32).reshape(channels, -1)
    weight_f32 = np.asarray(conv_weight, dtype=np.float32).reshape(channels, -1)
    if state_f32.shape != weight_f32.shape:
        raise ValueError("conv_state and conv_weight must have matching [channels, kernel] shape")
    kernel_size = int(weight_f32.shape[1])
    if kernel_size <= 0:
        raise ValueError("kernel size must be positive")

    acc = np.zeros((tokens, channels), dtype=np.float32)
    token_indices = np.arange(tokens, dtype=np.int64)
    for tap in range(kernel_size):
        source_rows = token_indices + tap - (kernel_size - 1)
        values = np.empty((tokens, channels), dtype=np.float32)
        history = source_rows < 0
        if np.any(history):
            history_columns = source_rows[history] + kernel_size
            values[history] = state_f32[:, history_columns].T
        if np.any(~history):
            values[~history] = qkv_f32[source_rows[~history]]
        product = np.asarray(values * weight_f32[:, tap][None, :], dtype=np.float32)
        acc = np.asarray(acc + product, dtype=np.float32)
    return acc


def precise_silu(values: np.ndarray) -> np.ndarray:
    values_f32 = np.asarray(values, dtype=np.float32)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        denominator = np.asarray(np.float32(1.0) + np.exp(-values_f32), dtype=np.float32)
        return np.asarray(values_f32 / denominator, dtype=np.float32)


def accumulator_summary(acc: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(acc, dtype=np.float32).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if not finite.size:
        raise ValueError("accumulator capture has no finite values")
    quantiles = (0.0, 0.0001, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999, 1.0)
    q_values = np.quantile(finite.astype(np.float64), quantiles)
    thresholds = (-104.0, -90.0, -88.0, -87.0, -80.0, -40.0, -20.0, -16.0, 16.0, 20.0, 40.0, 80.0, 87.0, 88.0, 90.0, 104.0)
    return {
        "count": int(flat.size),
        "finite_count": int(finite.size),
        "nan_count": int(np.count_nonzero(np.isnan(flat))),
        "positive_inf_count": int(np.count_nonzero(np.isposinf(flat))),
        "negative_inf_count": int(np.count_nonzero(np.isneginf(flat))),
        "mean": float(np.mean(finite, dtype=np.float64)),
        "stddev": float(np.std(finite, dtype=np.float64)),
        "quantiles": {f"p{100.0 * q:g}": float(value) for q, value in zip(quantiles, q_values, strict=True)},
        "threshold_counts": {
            **{f"le_{abs(threshold):g}": int(np.count_nonzero(finite <= threshold)) for threshold in thresholds if threshold < 0},
            **{f"ge_{threshold:g}": int(np.count_nonzero(finite >= threshold)) for threshold in thresholds if threshold > 0},
        },
        "sha256": hashlib.sha256(np.ascontiguousarray(flat).tobytes()).hexdigest(),
    }


def _capture_layers(
    *,
    model: Path,
    backend: str,
    prompt_length: int,
    token_id: int,
    layer_ids: Sequence[int],
    stop_before_layer: int | None,
    stop_after_full_stage: str | None,
    aotriton_stream: str,
    full_layer_id: int,
    compiler_version: str | None,
    require_cached_build: bool,
) -> tuple[dict[int, ConvCapture], Qwen35ParoResidentSession, Qwen35ParoNextTokenRunner, int]:
    target_layers = frozenset(int(layer_id) for layer_id in layer_ids)
    captures: dict[int, ConvCapture] = {}
    capture_inputs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    original = Qwen35ParoDecodeState.run_linear_attention_prefill_recurrent_fp16
    stage_original = None
    stage_method_name = None
    isolated_aotriton_stream = 0
    if stop_after_full_stage is not None:
        stage_method_name = {
            "kv_append": "append_full_attention_kv_fp16_batch",
            "aotriton": "prefill_full_attention_aotriton_varlen_gqa_bf16",
            "o_projection": "project_full_attention_o_bf16_attn_gate_fp16",
            "post_norm": "post_attention_add_rmsnorm_fp16",
            "moe": "run_moe_grouped_compact_fp16",
        }[stop_after_full_stage]
        stage_original = getattr(Qwen35ParoDecodeState, stage_method_name)

        def stage_wrapped(state: Qwen35ParoDecodeState, *stage_args, **stage_kwargs):
            nonlocal isolated_aotriton_stream
            if aotriton_stream == "isolated" and int(state.layer_weights.layer_id) == int(full_layer_id):
                source_stream = int(stage_kwargs.get("stream", 0))
                state.runtime.stream_synchronize(source_stream)
                if isolated_aotriton_stream == 0:
                    isolated_aotriton_stream = state.runtime.stream_create()
                stage_kwargs["stream"] = isolated_aotriton_stream
            result = stage_original(state, *stage_args, **stage_kwargs)
            if int(state.layer_weights.layer_id) == int(full_layer_id):
                stream = int(stage_kwargs.get("stream", 0))
                state.runtime.stream_synchronize(stream)
                raise CaptureBoundaryReached(
                    f"stopped after {stop_after_full_stage} in full-attention layer {full_layer_id}"
                )
            return result

        setattr(Qwen35ParoDecodeState, stage_method_name, stage_wrapped)

    def wrapped(
        state: Qwen35ParoDecodeState,
        scratch,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        tokens: int,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        layer_id = int(state.layer_weights.layer_id)
        should_capture = layer_id in target_layers and layer_id not in capture_inputs
        if should_capture:
            state.runtime.stream_synchronize(stream)
            prefix = f"layers.{layer_id}.linear_attn"
            weight = state.tensor(f"{prefix}.conv1d.weight")
            channels = int(scratch.qkv.shape[-1])
            kernel_size = int(state.config.linear_conv_kernel_dim)
            qkv = _copy_device_array(
                scratch.qkv.ptr,
                shape=(int(tokens), channels),
                dtype=np.float16,
                runtime=state.runtime,
            )
            conv_state_host = _copy_device_array(
                conv_state.ptr,
                shape=(channels, kernel_size),
                dtype=np.float32,
                runtime=state.runtime,
            )
            conv_weight = _copy_device_array(
                weight.ptr,
                shape=(channels, kernel_size),
                dtype=np.float32,
                runtime=state.runtime,
            )
            capture_inputs[layer_id] = (qkv, conv_state_host, conv_weight)
            if stop_before_layer == layer_id:
                captures[layer_id] = ConvCapture(
                    layer_id=layer_id,
                    qkv=qkv,
                    conv_state=conv_state_host,
                    conv_weight=conv_weight,
                    gpu_output=None,
                )
                raise CaptureBoundaryReached(f"stopped before layer {layer_id} convolution")

        result = original(
            state,
            scratch,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        if should_capture:
            state.runtime.stream_synchronize(stream)
            qkv, conv_state_host, conv_weight = capture_inputs[layer_id]
            gpu_output = _copy_device_array(
                scratch.conv_out.ptr,
                shape=(int(tokens), int(scratch.qkv.shape[-1])),
                dtype=np.float32,
                runtime=state.runtime,
            )
            captures[layer_id] = ConvCapture(
                layer_id=layer_id,
                qkv=qkv,
                conv_state=conv_state_host,
                conv_weight=conv_weight,
                gpu_output=gpu_output,
            )
        return result

    Qwen35ParoDecodeState.run_linear_attention_prefill_recurrent_fp16 = wrapped
    runner = Qwen35ParoNextTokenRunner(model, shared_expert_format="packed_paro_w4", backend=backend)
    session: Qwen35ParoResidentSession | None = None
    try:
        session = Qwen35ParoResidentSession(
            runner,
            max_sequence_length=prompt_length + 1,
            compiler_version=compiler_version,
            require_cached_build=require_cached_build,
            prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
        )
        prompt_tokens = _prompt_tokens(model, "Hello", token_id, prompt_length)
        session.reset()
        session._resolve_prefill_config_for_length(len(prompt_tokens))
        try:
            session.prefill_native(prompt_tokens, sample=False)
        except CaptureBoundaryReached:
            if stop_before_layer is None and stop_after_full_stage is None:
                raise
    except Exception:
        if session is not None:
            if isolated_aotriton_stream:
                session.runtime.stream_destroy(isolated_aotriton_stream)
            session.close()
        raise
    finally:
        Qwen35ParoDecodeState.run_linear_attention_prefill_recurrent_fp16 = original
        if stage_method_name is not None and stage_original is not None:
            setattr(Qwen35ParoDecodeState, stage_method_name, stage_original)

    missing = sorted(target_layers.difference(captures))
    if missing:
        if isolated_aotriton_stream:
            session.runtime.stream_destroy(isolated_aotriton_stream)
        session.close()
        raise RuntimeError(f"did not capture requested convolution layers: {missing}")
    return captures, session, runner, isolated_aotriton_stream


def _capture_summary(capture: ConvCapture) -> dict[str, Any]:
    acc = conv_accumulators(capture.qkv, capture.conv_state, capture.conv_weight)
    summary: dict[str, Any] = {
        "tokens": capture.tokens,
        "channels": capture.channels,
        "kernel_size": capture.kernel_size,
        "qkv_sha256": hashlib.sha256(np.ascontiguousarray(capture.qkv).tobytes()).hexdigest(),
        "conv_state_sha256": hashlib.sha256(np.ascontiguousarray(capture.conv_state).tobytes()).hexdigest(),
        "conv_weight_sha256": hashlib.sha256(np.ascontiguousarray(capture.conv_weight).tobytes()).hexdigest(),
        "accumulator": accumulator_summary(acc),
    }
    if capture.gpu_output is None:
        summary["gpu_output"] = {"available": False, "reason": "capture stopped before production conv"}
        return summary
    cpu_output = precise_silu(acc)
    finite_delta = np.abs(cpu_output.astype(np.float64) - capture.gpu_output.astype(np.float64))
    summary.update(
        {
            "gpu_output_sha256": hashlib.sha256(np.ascontiguousarray(capture.gpu_output).tobytes()).hexdigest(),
            "cpu_formula_vs_gpu_output": {
                "bit_equal_count": int(
                    np.count_nonzero(cpu_output.view(np.uint32) == capture.gpu_output.view(np.uint32))
                ),
                "count": int(cpu_output.size),
                "max_abs": float(np.nanmax(finite_delta)),
                "mean_abs": float(np.nanmean(finite_delta)),
            },
        }
    )
    return summary


def _run_replay_matrix(
    captures: dict[int, ConvCapture],
    *,
    layer_ids: tuple[int, ...],
    session: Qwen35ParoResidentSession,
    warmups: int,
    repetitions: int,
    burst_repeats: int,
    streams: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    if not layer_ids:
        raise ValueError("at least one captured layer is required")
    before = captures[layer_ids[0]]
    combinations = {f"input_L{before.layer_id}_weight_L{before.layer_id}": (before, before)}
    if len(layer_ids) > 1:
        after = captures[layer_ids[1]]
        combinations.update(
            {
                f"input_L{after.layer_id}_weight_L{after.layer_id}": (after, after),
                f"input_L{after.layer_id}_weight_L{before.layer_id}": (after, before),
                f"input_L{before.layer_id}_weight_L{after.layer_id}": (before, after),
            }
        )
    runtime = session.runtime
    library = session.libraries["linear_conv"]
    replay_buffers: dict[str, ReplayBuffers] = {}
    try:
        for name, (input_capture, weight_capture) in combinations.items():
            if input_capture.qkv.shape != weight_capture.qkv.shape:
                raise ValueError("cross replay requires matching capture shapes")
            replay_buffers[name] = _make_replay_buffers(input_capture, weight_capture, runtime=runtime)
        phases = {"workspace_live": {}}
        for stream_name, stream in streams:
            phases["workspace_live"][stream_name] = _measure_replay_phase(
                replay_buffers,
                combinations,
                library=library,
                runtime=runtime,
                warmups=warmups,
                repetitions=repetitions,
                burst_repeats=burst_repeats,
                stream=stream,
            )
        session._release_prefill_workspace()
        phases["workspace_released"] = {}
        for stream_name, stream in streams:
            phases["workspace_released"][stream_name] = _measure_replay_phase(
                replay_buffers,
                combinations,
                library=library,
                runtime=runtime,
                warmups=warmups,
                repetitions=repetitions,
                burst_repeats=burst_repeats,
                stream=stream,
            )
        return phases
    finally:
        for buffers in replay_buffers.values():
            buffers.close(runtime)


def _measure_replay_phase(
    replay_buffers: dict[str, ReplayBuffers],
    combinations: dict[str, tuple[ConvCapture, ConvCapture]],
    *,
    library,
    runtime,
    warmups: int,
    repetitions: int,
    burst_repeats: int,
    stream: int,
) -> dict[str, Any]:
    names = tuple(combinations)
    for cycle in range(warmups):
        for name in names[cycle % len(names) :] + names[: cycle % len(names)]:
            buffers = replay_buffers[name]
            state_host = combinations[name][1].conv_state
            copy_host_to_device(
                buffers.conv_state,
                host_array_ptr(state_host),
                state_host.nbytes,
                runtime=runtime,
            )
            for _ in range(burst_repeats):
                _launch_replay(
                    buffers,
                    capture=combinations[name][0],
                    library=library,
                    runtime=runtime,
                    stream=stream,
                )
    runtime.stream_synchronize(stream)

    samples: dict[str, list[float]] = {name: [] for name in names}
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        for cycle in range(repetitions):
            ordered = names[cycle % len(names) :] + names[: cycle % len(names)]
            for name in ordered:
                buffers = replay_buffers[name]
                state_host = combinations[name][1].conv_state
                copy_host_to_device(
                    buffers.conv_state,
                    host_array_ptr(state_host),
                    state_host.nbytes,
                    runtime=runtime,
                )
                runtime.event_record(start, stream)
                for _ in range(burst_repeats):
                    _launch_replay(
                        buffers,
                        capture=combinations[name][0],
                        library=library,
                        runtime=runtime,
                        stream=stream,
                    )
                runtime.event_record(stop, stream)
                runtime.event_synchronize(stop)
                samples[name].append(runtime.event_elapsed_time_ms(start, stop) * 1000.0 / burst_repeats)
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)

    results: dict[str, Any] = {}
    for name, values in samples.items():
        input_capture, weight_capture = combinations[name]
        acc = conv_accumulators(input_capture.qkv, weight_capture.conv_state, weight_capture.conv_weight)
        _reset_and_launch(
            replay_buffers[name],
            weight_capture.conv_state,
            capture=input_capture,
            library=library,
            runtime=runtime,
            stream=stream,
        )
        runtime.stream_synchronize(stream)
        output = _copy_device_array(
            replay_buffers[name].output.ptr,
            shape=acc.shape,
            dtype=np.float32,
            runtime=runtime,
        )
        expected = precise_silu(acc)
        results[name] = {
            "input_layer": input_capture.layer_id,
            "weight_layer": weight_capture.layer_id,
            "warmups": int(warmups),
            "repetitions": int(repetitions),
            "burst_repeats": int(burst_repeats),
            "duration_us": _sample_summary(values),
            "accumulator": accumulator_summary(acc),
            "output_sha256": hashlib.sha256(np.ascontiguousarray(output).tobytes()).hexdigest(),
            "cpu_formula_vs_gpu_output": {
                "bit_equal_count": int(np.count_nonzero(expected.view(np.uint32) == output.view(np.uint32))),
                "count": int(expected.size),
                "max_abs": float(np.nanmax(np.abs(expected.astype(np.float64) - output.astype(np.float64)))),
            },
        }
    return results


def _make_replay_buffers(
    input_capture: ConvCapture,
    weight_capture: ConvCapture,
    *,
    runtime,
) -> ReplayBuffers:
    buffers = ReplayBuffers(
        qkv=malloc(input_capture.qkv.nbytes, runtime=runtime),
        conv_state=malloc(weight_capture.conv_state.nbytes, runtime=runtime),
        conv_weight=malloc(weight_capture.conv_weight.nbytes, runtime=runtime),
        output=malloc(input_capture.tokens * input_capture.channels * np.dtype(np.float32).itemsize, runtime=runtime),
    )
    copy_host_to_device(buffers.qkv, host_array_ptr(input_capture.qkv), runtime=runtime)
    copy_host_to_device(buffers.conv_state, host_array_ptr(weight_capture.conv_state), runtime=runtime)
    copy_host_to_device(buffers.conv_weight, host_array_ptr(weight_capture.conv_weight), runtime=runtime)
    return buffers


def _reset_and_launch(
    buffers: ReplayBuffers,
    state_host: np.ndarray,
    *,
    capture: ConvCapture,
    library,
    runtime,
    stream: int,
) -> None:
    copy_host_to_device(buffers.conv_state, host_array_ptr(state_host), state_host.nbytes, runtime=runtime)
    _launch_replay(buffers, capture=capture, library=library, runtime=runtime, stream=stream)


def _launch_replay(buffers: ReplayBuffers, *, capture: ConvCapture, library, runtime, stream: int) -> None:
    qwen35_linear_attn_conv_prefill_fp16(
        buffers.qkv.ptr,
        buffers.conv_state.ptr,
        buffers.conv_weight.ptr,
        buffers.output.ptr,
        capture.tokens,
        capture.channels,
        capture.kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _copy_device_array(ptr: int, *, shape: Sequence[int], dtype: np.dtype[Any], runtime) -> np.ndarray:
    host = np.empty(tuple(int(dim) for dim in shape), dtype=dtype)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(int(ptr), int(host.nbytes)),
        host.nbytes,
        runtime=runtime,
    )
    return host


def _sample_summary(values: Sequence[float]) -> dict[str, float | int]:
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "median": float(statistics.median(numeric)),
        "min": float(min(numeric)),
        "max": float(max(numeric)),
        "mean": float(statistics.fmean(numeric)),
        "stdev": float(statistics.pstdev(numeric)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
