#!/usr/bin/env python3
"""Screen Q8_0-weight/q8_1-activation decode on real Laguna F16 inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
from typing import Callable

import numpy as np

from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
    laguna_f16w_onebarrier_gemv_bf16_bf16_out,
    laguna_f16w_onebarrier_gemv_bf16_f32_out,
    laguna_f16w_triple_onebarrier_gemv_bf16_f32_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_quantize_bf16_q8_1,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dp4a_gemv import (
    build_gguf_q8_0_dp4a_gemv,
    gguf_q8_0_dp4a_gemv_bf16_bf16_out,
    gguf_q8_0_dp4a_gemv_f32_f32_out,
    gguf_q8_0_dp4a_triple_split_rowtile4_gemv_f32_f32_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
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

_BLOCK = 32
_Q8_0_BYTES = 34
_Q8_1_BYTES = 36
_LAYER_RE = re.compile(r"^blk\.(\d+)\.")


def _parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(item) for item in value.split(",") if item.strip())
    if not layers or len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("capture layers must be distinct integers")
    if any(layer < 0 or layer >= 48 for layer in layers):
        raise argparse.ArgumentTypeError("capture layers must be within [0, 48)")
    return layers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layers", type=_parse_layers, default=(0, 23, 44, 47))
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--burst", type=int, default=20)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _quantize_f16_q8_0(weight: np.ndarray, *, row_batch: int = 128) -> np.ndarray:
    source = np.ascontiguousarray(weight, dtype=np.float16)
    if source.ndim != 2 or source.shape[1] % _BLOCK:
        raise ValueError("F16 Q8_0 screen requires [out, K] with K divisible by 32")
    out_features, in_features = source.shape
    blocks = in_features // _BLOCK
    packed = np.empty((out_features, blocks, _Q8_0_BYTES), dtype=np.uint8)
    for start in range(0, out_features, row_batch):
        end = min(start + row_batch, out_features)
        values = source[start:end].astype(np.float32).reshape(-1, blocks, _BLOCK)
        amax = np.max(np.abs(values), axis=2)
        scales = (amax / 127.0).astype(np.float16)
        stored_scales = scales.astype(np.float32)
        inverse = np.zeros_like(stored_scales)
        np.divide(
            1.0,
            stored_scales,
            out=inverse,
            where=stored_scales != 0.0,
        )
        quants = np.clip(
            np.rint(values * inverse[:, :, None]),
            -127,
            127,
        ).astype(np.int8)
        scale_bytes = np.ascontiguousarray(scales).view(np.uint8).reshape(-1, blocks, 2)
        packed[start:end, :, :2] = scale_bytes
        packed[start:end, :, 2:] = quants.view(np.uint8)
    return packed.reshape(out_features, blocks * _Q8_0_BYTES)


def _copy_ptr(runtime, ptr: int, shape: tuple[int, ...], dtype) -> np.ndarray:
    host = np.empty(shape, dtype=dtype)
    runtime.memcpy(
        host_array_ptr(host),
        ptr,
        host.nbytes,
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return host


def _upload(runtime, host: np.ndarray):
    contiguous = np.ascontiguousarray(host)
    device = malloc(contiguous.nbytes, runtime=runtime)
    copy_host_to_device(
        device,
        host_array_ptr(contiguous),
        contiguous.nbytes,
        runtime=runtime,
    )
    return device


def _download(runtime, device, shape: tuple[int, ...], dtype) -> np.ndarray:
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(
        host_array_ptr(host),
        device,
        host.nbytes,
        runtime=runtime,
    )
    return host


def _time_samples(
    runtime,
    launch: Callable[[], None],
    *,
    samples: int,
    warmups: int,
    burst: int,
) -> list[float]:
    for _ in range(warmups):
        launch()
    runtime.device_synchronize()
    timings = []
    for _ in range(samples):
        start = runtime.event_create()
        stop = runtime.event_create()
        try:
            runtime.event_record(start)
            for _ in range(burst):
                launch()
            runtime.event_record(stop)
            runtime.event_synchronize(stop)
            timings.append(float(runtime.event_elapsed_time_ms(start, stop)) / burst)
        finally:
            runtime.event_destroy(stop)
            runtime.event_destroy(start)
    return timings


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
    }


def _quality(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float32).reshape(-1)
    delta = cand - ref
    ref_rms = float(np.sqrt(np.mean(ref * ref)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(cand))
    cosine = 1.0 if denominator == 0.0 else float(np.dot(ref, cand) / denominator)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": rmse,
        "reference_rms": ref_rms,
        "normalized_rmse": rmse / max(ref_rms, 1.0e-30),
        "cosine": cosine,
    }


def _layer_id(slot_path: str) -> int | None:
    match = _LAYER_RE.match(slot_path)
    return None if match is None else int(match.group(1))


def _capture_decode_inputs(
    owner: LagunaGGUFResidentSession,
    token_id: int,
    layers: tuple[int, ...],
) -> tuple[dict[int, dict[str, np.ndarray]], object]:
    import hipengine.runtime.laguna_gguf_runner as runner

    selected = set(layers)
    captures: dict[int, dict[str, np.ndarray]] = {
        layer: {} for layer in layers
    }
    runtime = owner.runtime
    original_single = runner.launch_f16_weight_linear
    original_triple = runner.launch_f16_weight_linear_triple

    def single(weight, x_ptr, out_ptr, rows, in_features, out_features, **kwargs):
        source_name = weight.spec.source.name
        layer = _layer_id(source_name)
        capture = layer in selected and rows == 1
        if capture and source_name.endswith(".attn_output.weight"):
            captures[layer]["output_input"] = _copy_ptr(
                runtime, x_ptr, (1, in_features), np.uint16
            )
        original_single(
            weight,
            x_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            **kwargs,
        )
        if capture and source_name.endswith(".attn_gate.weight"):
            captures[layer]["gate"] = _copy_ptr(
                runtime, out_ptr, (1, out_features), np.float32
            )
        elif capture and source_name.endswith(".attn_output.weight"):
            captures[layer]["output"] = _copy_ptr(
                runtime, out_ptr, (1, out_features), np.uint16
            )

    def triple(
        q_weight,
        k_weight,
        v_weight,
        x_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        rows,
        in_features,
        q_features,
        k_features,
        v_features,
        **kwargs,
    ):
        layer = _layer_id(q_weight.spec.source.name)
        capture = layer in selected and rows == 1
        if capture:
            captures[layer]["norm_input"] = _copy_ptr(
                runtime, x_ptr, (1, in_features), np.uint16
            )
        original_triple(
            q_weight,
            k_weight,
            v_weight,
            x_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            rows,
            in_features,
            q_features,
            k_features,
            v_features,
            **kwargs,
        )
        if capture:
            captures[layer]["q"] = _copy_ptr(
                runtime, q_ptr, (1, q_features), np.float32
            )
            captures[layer]["k"] = _copy_ptr(
                runtime, k_ptr, (1, k_features), np.float32
            )
            captures[layer]["v"] = _copy_ptr(
                runtime, v_ptr, (1, v_features), np.float32
            )

    runner.launch_f16_weight_linear = single
    runner.launch_f16_weight_linear_triple = triple
    try:
        result = owner.forward_token(token_id)
    finally:
        runner.launch_f16_weight_linear = original_single
        runner.launch_f16_weight_linear_triple = original_triple
    missing = {
        layer: sorted(
            {"norm_input", "output_input", "q", "k", "v", "gate", "output"}
            - set(captures[layer])
        )
        for layer in layers
        if len(captures[layer]) != 7
    }
    if missing:
        raise RuntimeError(f"incomplete Laguna F16 decode capture: {missing}")
    return captures, result


def _screen_layer(
    owner: LagunaGGUFResidentSession,
    reader: GGUFReader,
    layer_id: int,
    capture: dict[str, np.ndarray],
    *,
    q4_library,
    dp4a_library,
    samples: int,
    warmups: int,
    burst: int,
) -> dict[str, object]:
    assert owner.weights is not None
    assert owner.libraries is not None
    runtime = owner.runtime
    layer = owner.weights.layer(layer_id)
    slots = ("attn_q", "attn_k", "attn_v", "attn_gate", "attn_output")
    source_weights = {
        slot: np.asarray(reader.tensor_data(layer.weight(slot).spec.source.name))
        for slot in slots
    }
    qweights = {
        slot: _quantize_f16_q8_0(source_weights[slot]) for slot in slots
    }
    allocations = []
    try:
        dx_norm = _upload(runtime, capture["norm_input"])
        dx_output = _upload(runtime, capture["output_input"])
        allocations.extend((dx_norm, dx_output))
        device_qweights = {
            slot: _upload(runtime, qweight) for slot, qweight in qweights.items()
        }
        allocations.extend(device_qweights.values())
        hidden = capture["norm_input"].shape[1]
        output_in = capture["output_input"].shape[1]
        xq_norm = malloc(hidden // _BLOCK * _Q8_1_BYTES, runtime=runtime)
        xq_output = malloc(output_in // _BLOCK * _Q8_1_BYTES, runtime=runtime)
        allocations.extend((xq_norm, xq_output))
        widths = {
            slot: int(source_weights[slot].shape[0]) for slot in slots
        }
        candidate_outputs = {
            slot: malloc(
                widths[slot] * (2 if slot == "attn_output" else 4),
                runtime=runtime,
            )
            for slot in slots
        }
        baseline_outputs = {
            slot: malloc(
                widths[slot] * (2 if slot == "attn_output" else 4),
                runtime=runtime,
            )
            for slot in slots
        }
        allocations.extend(candidate_outputs.values())
        allocations.extend(baseline_outputs.values())
        f16_library = owner.libraries.f16_projection

        def baseline_norm() -> None:
            laguna_f16w_triple_onebarrier_gemv_bf16_f32_out(
                dx_norm.ptr,
                *(layer.weight(slot).allocation("raw").tensor.ptr for slot in slots[:3]),
                *(baseline_outputs[slot].ptr for slot in slots[:3]),
                1,
                hidden,
                *(widths[slot] for slot in slots[:3]),
                library=f16_library,
                runtime=runtime,
            )
            laguna_f16w_onebarrier_gemv_bf16_f32_out(
                dx_norm.ptr,
                layer.weight("attn_gate").allocation("raw").tensor.ptr,
                baseline_outputs["attn_gate"].ptr,
                1,
                hidden,
                widths["attn_gate"],
                library=f16_library,
                runtime=runtime,
            )

        def candidate_norm() -> None:
            gguf_q4_k_quantize_bf16_q8_1(
                dx_norm.ptr,
                xq_norm.ptr,
                1,
                hidden,
                library=q4_library,
                runtime=runtime,
            )
            gguf_q8_0_dp4a_triple_split_rowtile4_gemv_f32_f32_out(
                xq_norm.ptr,
                *(device_qweights[slot].ptr for slot in slots[:3]),
                *(candidate_outputs[slot].ptr for slot in slots[:3]),
                1,
                hidden,
                *(widths[slot] for slot in slots[:3]),
                library=dp4a_library,
                runtime=runtime,
            )
            gguf_q8_0_dp4a_gemv_f32_f32_out(
                xq_norm.ptr,
                device_qweights["attn_gate"].ptr,
                candidate_outputs["attn_gate"].ptr,
                1,
                hidden,
                widths["attn_gate"],
                library=dp4a_library,
                runtime=runtime,
            )

        def baseline_output() -> None:
            laguna_f16w_onebarrier_gemv_bf16_bf16_out(
                dx_output.ptr,
                layer.weight("attn_output").allocation("raw").tensor.ptr,
                baseline_outputs["attn_output"].ptr,
                1,
                output_in,
                widths["attn_output"],
                library=f16_library,
                runtime=runtime,
            )

        def candidate_output() -> None:
            gguf_q4_k_quantize_bf16_q8_1(
                dx_output.ptr,
                xq_output.ptr,
                1,
                output_in,
                library=q4_library,
                runtime=runtime,
            )
            gguf_q8_0_dp4a_gemv_bf16_bf16_out(
                xq_output.ptr,
                device_qweights["attn_output"].ptr,
                candidate_outputs["attn_output"].ptr,
                1,
                output_in,
                widths["attn_output"],
                library=dp4a_library,
                runtime=runtime,
            )

        baseline_norm()
        candidate_norm()
        baseline_output()
        candidate_output()
        runtime.device_synchronize()
        quality = {}
        reference_names = {
            "attn_q": "q",
            "attn_k": "k",
            "attn_v": "v",
            "attn_gate": "gate",
            "attn_output": "output",
        }
        for slot, reference_name in reference_names.items():
            dtype = np.uint16 if slot == "attn_output" else np.float32
            candidate = _download(
                runtime,
                candidate_outputs[slot],
                (1, widths[slot]),
                dtype,
            )
            reference = capture[reference_name]
            if dtype == np.uint16:
                candidate = bf16_to_float32(candidate)
                reference = bf16_to_float32(reference)
            quality[slot] = _quality(reference, candidate)

        timings = {}
        for name, baseline, candidate in (
            ("qkv_gate", baseline_norm, candidate_norm),
            ("output", baseline_output, candidate_output),
        ):
            mode_samples = {"baseline": [], "q8": []}
            for repetition in range(samples):
                order = (
                    (("baseline", baseline), ("q8", candidate))
                    if repetition % 2 == 0
                    else (("q8", candidate), ("baseline", baseline))
                )
                for mode, launch in order:
                    mode_samples[mode].extend(
                        _time_samples(
                            runtime,
                            launch,
                            samples=1,
                            warmups=warmups if repetition == 0 else 0,
                            burst=burst,
                        )
                    )
            baseline_ms = statistics.median(mode_samples["baseline"])
            candidate_ms = statistics.median(mode_samples["q8"])
            timings[name] = {
                "baseline": _summary(mode_samples["baseline"]),
                "q8": _summary(mode_samples["q8"]),
                "latency_change_percent": (candidate_ms / baseline_ms - 1.0) * 100.0,
            }
        source_bytes = sum(weight.nbytes for weight in source_weights.values())
        q8_bytes = sum(weight.nbytes for weight in qweights.values())
        return {
            "layer_id": layer_id,
            "attention_type": layer.attention_type,
            "source_f16_bytes": source_bytes,
            "q8_0_bytes": q8_bytes,
            "resident_byte_change_percent": (q8_bytes / source_bytes - 1.0) * 100.0,
            "activation_stats": {
                "norm_abs_max": float(
                    np.max(np.abs(bf16_to_float32(capture["norm_input"])))
                ),
                "output_abs_max": float(
                    np.max(np.abs(bf16_to_float32(capture["output_input"])))
                ),
            },
            "quality": quality,
            "timings": timings,
        }
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.samples <= 0 or args.warmups < 0 or args.burst <= 0:
        raise ValueError("samples/burst must be positive and warmups non-negative")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )
    runtime = get_hip_runtime()
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(DEFAULT_PROMPTS, tokenizer)
    token_stream, _ = _profile_token_stream(prompts, 512)
    owner = LagunaGGUFResidentSession(
        args.model,
        context_length=640,
        backend="hip_gfx1151",
        runtime=runtime,
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached_build=args.require_cached_build,
        progress=_progress,
        repacked_cache=DEFAULT_CACHE,
        model_sha256=(
            DEFAULT_MODEL_SHA256
            if Path(args.model).resolve() == Path(DEFAULT_MODEL).resolve()
            else None
        ),
        prefill_chunk_size=512,
        prefill_attention_chunk_size=128,
    )
    try:
        prefill = owner.prefill(token_stream)
        captures, decoded = _capture_decode_inputs(
            owner,
            int(prefill.next_token_id),
            args.layers,
        )
        q4_library = build_gguf_q4_k_gemv(
            load=True,
            require_cached=args.require_cached_build,
        )
        dp4a_library = build_gguf_q8_0_dp4a_gemv(
            load=True,
            require_cached=args.require_cached_build,
        )
        results = [
            _screen_layer(
                owner,
                reader,
                layer_id,
                captures[layer_id],
                q4_library=q4_library,
                dp4a_library=dp4a_library,
                samples=args.samples,
                warmups=args.warmups,
                burst=args.burst,
            )
            for layer_id in args.layers
        ]
    finally:
        owner.close()
    by_type = {}
    for attention_type, calls in (
        (FULL_ATTENTION, 12),
        (SLIDING_ATTENTION, 36),
    ):
        selected = [
            row for row in results if row["attention_type"] == attention_type
        ]
        if not selected:
            continue
        baseline = statistics.mean(
            sum(
                float(row["timings"][family]["baseline"]["median_ms"])
                for family in ("qkv_gate", "output")
            )
            for row in selected
        )
        candidate = statistics.mean(
            sum(
                float(row["timings"][family]["q8"]["median_ms"])
                for family in ("qkv_gate", "output")
            )
            for row in selected
        )
        by_type[attention_type] = {
            "sampled_layers": [int(row["layer_id"]) for row in selected],
            "calls_per_token": calls,
            "baseline_ms_per_layer": baseline,
            "q8_ms_per_layer": candidate,
            "latency_change_percent": (candidate / baseline - 1.0) * 100.0,
        }
    modeled_baseline = sum(
        row["baseline_ms_per_layer"] * row["calls_per_token"]
        for row in by_type.values()
    )
    modeled_q8 = sum(
        row["q8_ms_per_layer"] * row["calls_per_token"]
        for row in by_type.values()
    )
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_f16_decode_q8_real_input_screen",
        "status": "candidate",
        "source": {
            "revision": _revision(),
            "model": str(Path(args.model).resolve()),
            "model_sha256": DEFAULT_MODEL_SHA256,
        },
        "protocol": {
            "layers": list(args.layers),
            "samples": args.samples,
            "warmups": args.warmups,
            "burst": args.burst,
            "capture": "one production transition after exact p512",
            "candidate": "raw Q8_0 weights plus one q8_1 activation pack per QKV+gate and output family",
        },
        "trajectory": {
            "prefill_next_token": int(prefill.next_token_id),
            "decoded_next_token": int(decoded.next_token_id),
            "decoded_position": int(decoded.position),
        },
        "layers": results,
        "modeled_family": {
            "attention_types": by_type,
            "baseline_ms_per_token": modeled_baseline,
            "q8_ms_per_token": modeled_q8,
            "latency_change_percent": (modeled_q8 / modeled_baseline - 1.0) * 100.0,
            "modeled_saving_ms_per_token": modeled_baseline - modeled_q8,
        },
    }


def main() -> None:
    args = _parse_args()
    payload = json.dumps(run(args), indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
