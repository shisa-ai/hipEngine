#!/usr/bin/env python3
"""Compare ordinary Q6T16 logits+argmax with producer-owned top-1 control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
from typing import Callable

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    build_gguf_ops,
    gguf_rmsnorm_bf16_f32_weight,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    argmax_f32_publish_control,
    argmax_tile_stage2_i32_publish_control,
    build_lm_head,
    lm_head_argmax_stage1_blocks,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding import (
    build_gguf_q6_k_embedding,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    build_gguf_q6_k_t16_gemv,
    gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.laguna_gguf_materialize import (
    materialize_laguna_gguf_weights,
)
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import (
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
)


DEFAULT_MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")
_ROOT_SLOTS = ("root.token_embedding", "root.output_norm", "root.lm_head")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--token-id", type=int, default=100257)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--burst", type=int, default=8)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _upload(runtime, host: np.ndarray):
    device = malloc(host.nbytes, runtime=runtime)
    copy_host_to_device(
        device,
        host_array_ptr(host),
        host.nbytes,
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


def _time_ms(runtime, launch: Callable[[], None], burst: int) -> float:
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(burst):
            launch()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        return float(runtime.event_elapsed_time_ms(start, stop)) / burst
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.samples <= 0 or args.warmups < 0 or args.burst <= 0:
        raise ValueError("samples/burst must be positive and warmups non-negative")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )
    runtime = get_hip_runtime()
    compiler_version = (
        None
        if args.compiler_version_file is None
        else args.compiler_version_file.read_text(encoding="utf-8")
    )
    build_kwargs = {
        "compiler_version": compiler_version,
        "require_cached": args.require_cached_build,
        "load": True,
    }
    libraries = {
        "embedding": build_gguf_q6_k_embedding(**build_kwargs),
        "norm": build_gguf_ops(**build_kwargs),
        "lm_head": build_gguf_q6_k_t16_gemv(**build_kwargs),
        "argmax": build_lm_head(**build_kwargs),
    }
    reader = GGUFReader(args.model)
    free_before, _ = runtime.mem_get_info()
    resident = materialize_laguna_gguf_weights(
        reader,
        selected_slots=_ROOT_SLOTS,
        context_length=4096,
        available_bytes=free_before,
        runtime=runtime,
        backend="hip_gfx1151",
    )
    buffers = []
    try:
        config = resident.config
        token = np.asarray([args.token_id], dtype=np.int64)
        token_d = _upload(runtime, token)
        embedding_d = malloc(config.hidden_size * 2, runtime=runtime)
        norm_d = malloc(config.hidden_size * 2, runtime=runtime)
        control_logits_d = malloc(config.vocab_size * 4, runtime=runtime)
        candidate_logits_d = malloc(config.vocab_size * 4, runtime=runtime)
        stage1_blocks = lm_head_argmax_stage1_blocks(config.vocab_size)
        control_values_d = malloc(stage1_blocks * 4, runtime=runtime)
        control_indices_d = malloc(stage1_blocks * 8, runtime=runtime)
        tile_count = config.vocab_size // 16
        tile_values_d = malloc(tile_count * 4, runtime=runtime)
        tile_indices_d = malloc(tile_count * 4, runtime=runtime)
        control_id_d = malloc(8, runtime=runtime)
        control_value_d = malloc(4, runtime=runtime)
        candidate_id_d = malloc(8, runtime=runtime)
        candidate_value_d = malloc(4, runtime=runtime)
        control_token_d = malloc(8, runtime=runtime)
        control_position_d = malloc(8, runtime=runtime)
        control_kv_position_d = malloc(8, runtime=runtime)
        candidate_token_d = malloc(8, runtime=runtime)
        candidate_position_d = malloc(8, runtime=runtime)
        candidate_kv_position_d = malloc(8, runtime=runtime)
        buffers.extend(
            (
                token_d,
                embedding_d,
                norm_d,
                control_logits_d,
                candidate_logits_d,
                control_values_d,
                control_indices_d,
                tile_values_d,
                tile_indices_d,
                control_id_d,
                control_value_d,
                candidate_id_d,
                candidate_value_d,
                control_token_d,
                control_position_d,
                control_kv_position_d,
                candidate_token_d,
                candidate_position_d,
                candidate_kv_position_d,
            )
        )

        launch_gguf_embedding(
            resident.root("token_embedding"),
            token_d.ptr,
            embedding_d.ptr,
            1,
            config.hidden_size,
            config.vocab_size,
            backend="hip_gfx1151",
            libraries={"gguf_q4_k": libraries["embedding"]},
            runtime=runtime,
        )
        gguf_rmsnorm_bf16_f32_weight(
            embedding_d.ptr,
            resident.root("output_norm").allocation().tensor.ptr,
            norm_d.ptr,
            1,
            config.hidden_size,
            config.rms_norm_eps,
            library=libraries["norm"],
            runtime=runtime,
        )
        head_weight = resident.root("lm_head")
        head_tiles = head_weight.allocation("tiles").tensor.ptr

        def control() -> None:
            launch_gguf_linear(
                head_weight,
                norm_d.ptr,
                control_logits_d.ptr,
                1,
                config.hidden_size,
                config.vocab_size,
                output_dtype=GGUF_OUTPUT_F32,
                backend="hip_gfx1151",
                libraries={"gguf_q6_k_t16_v1": libraries["lm_head"]},
                runtime=runtime,
            )
            argmax_f32_publish_control(
                control_logits_d.ptr,
                control_values_d.ptr,
                control_indices_d.ptr,
                control_id_d.ptr,
                control_value_d.ptr,
                control_token_d.ptr,
                control_position_d.ptr,
                control_kv_position_d.ptr,
                config.vocab_size,
                513,
                library=libraries["argmax"],
                runtime=runtime,
            )

        def candidate() -> None:
            gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1(
                norm_d.ptr,
                head_tiles,
                candidate_logits_d.ptr,
                tile_values_d.ptr,
                tile_indices_d.ptr,
                config.hidden_size,
                config.vocab_size,
                library=libraries["lm_head"],
                runtime=runtime,
            )
            argmax_tile_stage2_i32_publish_control(
                tile_values_d.ptr,
                tile_indices_d.ptr,
                candidate_id_d.ptr,
                candidate_value_d.ptr,
                candidate_token_d.ptr,
                candidate_position_d.ptr,
                candidate_kv_position_d.ptr,
                tile_count,
                513,
                library=libraries["argmax"],
                runtime=runtime,
            )

        control()
        candidate()
        runtime.device_synchronize()
        control_logits = _download(
            runtime, control_logits_d, (config.vocab_size,), np.float32
        )
        candidate_logits = _download(
            runtime, candidate_logits_d, (config.vocab_size,), np.float32
        )
        control_id = _download(runtime, control_id_d, (1,), np.int64)
        candidate_id = _download(runtime, candidate_id_d, (1,), np.int64)
        control_value = _download(runtime, control_value_d, (1,), np.float32)
        candidate_value = _download(
            runtime, candidate_value_d, (1,), np.float32
        )
        for _ in range(args.warmups):
            control()
            candidate()
        runtime.device_synchronize()
        control_samples: list[float] = []
        candidate_samples: list[float] = []
        paired_delta: list[float] = []
        for sample in range(args.samples):
            if sample % 2 == 0:
                control_ms = _time_ms(runtime, control, args.burst)
                candidate_ms = _time_ms(runtime, candidate, args.burst)
            else:
                candidate_ms = _time_ms(runtime, candidate, args.burst)
                control_ms = _time_ms(runtime, control, args.burst)
            control_samples.append(control_ms)
            candidate_samples.append(candidate_ms)
            paired_delta.append(candidate_ms - control_ms)

        control_summary = _summary(control_samples)
        candidate_summary = _summary(candidate_samples)
        control_median = float(control_summary["median_ms"])
        candidate_median = float(candidate_summary["median_ms"])
        exact = bool(
            np.array_equal(control_logits.view(np.uint32), candidate_logits.view(np.uint32))
            and np.array_equal(control_id, candidate_id)
            and np.array_equal(
                control_value.view(np.uint32), candidate_value.view(np.uint32)
            )
        )
        return {
            "schema": "hipengine.laguna_q6_lm_head_top1_leaf.v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "model": str(args.model),
            "backend": "hip_gfx1151",
            "shape": {
                "rows": 1,
                "hidden_size": config.hidden_size,
                "vocab_size": config.vocab_size,
                "producer_tiles": tile_count,
            },
            "protocol": {
                "samples": args.samples,
                "warmups": args.warmups,
                "burst": args.burst,
                "counterbalanced": True,
            },
            "correctness": {
                "exact_logits": bool(
                    np.array_equal(
                        control_logits.view(np.uint32),
                        candidate_logits.view(np.uint32),
                    )
                ),
                "exact_top1_id": bool(np.array_equal(control_id, candidate_id)),
                "exact_top1_value": bool(
                    np.array_equal(
                        control_value.view(np.uint32),
                        candidate_value.view(np.uint32),
                    )
                ),
                "control_id": int(control_id[0]),
                "candidate_id": int(candidate_id[0]),
                "passed": exact,
            },
            "control": control_summary,
            "candidate": candidate_summary,
            "paired_delta_ms": paired_delta,
            "median_paired_delta_ms": statistics.median(paired_delta),
            "median_delta_percent": (
                (candidate_median / control_median - 1.0) * 100.0
            ),
            "passed": exact and candidate_median < control_median,
        }
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        resident.free(runtime=runtime)


def main() -> int:
    args = _parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
