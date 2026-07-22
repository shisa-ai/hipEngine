#!/usr/bin/env python3
"""Probe Laguna's Q4_K embedding, final norm, Q6_K LM head, and argmax.

The probe materializes only the three target-owned root tensors.  It compares
native gfx11 output with raw-GGUF CPU-reference math and is suitable for a
cache-only ``rocprofv3 --kernel-trace`` launch after ``--prebuild-only``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_q4_k_embedding, rmsnorm
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    build_gguf_ops,
    gguf_rmsnorm_bf16_f32_weight,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    argmax_f32,
    build_lm_head,
    lm_head_argmax_stage1_blocks,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding import (
    build_gguf_q6_k_embedding,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    build_gguf_q6_k_t16_gemv,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.laguna_gguf_materialize import materialize_laguna_gguf_weights
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import GGMLQuantizationType, bf16_to_float32, dequantize_gguf_data
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import GGUF_OUTPUT_F32, launch_gguf_linear

DEFAULT_MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")
_ROOT_SLOTS = ("root.token_embedding", "root.output_norm", "root.lm_head")


def _compiler_version(path: Path | None) -> str | None:
    return None if path is None else path.read_text(encoding="utf-8")


def build_probe_libraries(
    *,
    compiler_version: str | None = None,
    require_cached: bool = False,
    load: bool = True,
) -> dict[str, Any]:
    """Build/load every library used by the root probe."""

    kwargs = {
        "compiler_version": compiler_version,
        "require_cached": require_cached,
        "load": load,
    }
    return {
        "embedding": build_gguf_q6_k_embedding(**kwargs),
        "norm": build_gguf_ops(**kwargs),
        "lm_head": build_gguf_q6_k_t16_gemv(**kwargs),
        "argmax": build_lm_head(**kwargs),
    }


def _copy_to_device(array: np.ndarray, runtime):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _copy_from_device(device, shape: tuple[int, ...], dtype: np.dtype, runtime) -> np.ndarray:
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host


def _raw_q6_logits(
    hidden_bf16_bits: np.ndarray,
    raw_weight: np.ndarray,
    *,
    chunk_rows: int,
) -> np.ndarray:
    """Compute a full raw-Q6_K CPU-reference LM head with bounded host RSS."""

    hidden = bf16_to_float32(np.asarray(hidden_bf16_bits, dtype=np.uint16)).reshape(-1)
    logits = np.empty(raw_weight.shape[0], dtype=np.float32)
    for start in range(0, raw_weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, raw_weight.shape[0])
        dense = dequantize_gguf_data(
            np.ascontiguousarray(raw_weight[start:end]),
            GGMLQuantizationType.Q6_K,
        ).astype(np.float32, copy=False)
        logits[start:end] = np.matmul(dense, hidden).astype(np.float32, copy=False)
    return logits


def _kl_divergence(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    ref = np.asarray(reference_logits, dtype=np.float64)
    cand = np.asarray(candidate_logits, dtype=np.float64)
    ref_shift = ref - float(np.max(ref))
    cand_shift = cand - float(np.max(cand))
    ref_log_z = float(np.log(np.sum(np.exp(ref_shift))))
    cand_log_z = float(np.log(np.sum(np.exp(cand_shift))))
    ref_logp = ref_shift - ref_log_z
    cand_logp = cand_shift - cand_log_z
    ref_p = np.exp(ref_logp)
    return float(np.sum(ref_p * (ref_logp - cand_logp)))


def run_laguna_root_probe(
    model_path: str | Path,
    *,
    backend: str = "hip_gfx1151",
    token_id: int = 100257,
    cpu_chunk_rows: int = 2048,
    compiler_version: str | None = None,
    require_cached: bool = False,
) -> dict[str, Any]:
    """Run all root primitives and return machine-readable correctness metrics."""

    model_path = Path(model_path)
    reader = GGUFReader(model_path)
    runtime = get_hip_runtime()
    libraries = build_probe_libraries(
        compiler_version=compiler_version,
        require_cached=require_cached,
        load=True,
    )
    free_before, _ = runtime.mem_get_info()
    resident = materialize_laguna_gguf_weights(
        reader,
        selected_slots=_ROOT_SLOTS,
        context_length=4096,
        available_bytes=free_before,
        runtime=runtime,
        backend=backend,
    )
    buffers = []
    try:
        config = resident.config
        if token_id < 0 or token_id >= config.vocab_size:
            raise ValueError(f"token_id must be in [0, {config.vocab_size})")
        token_ids = np.asarray([token_id], dtype=np.int64)
        token_dev = _copy_to_device(token_ids, runtime)
        embedding_dev = malloc(config.hidden_size * np.dtype(np.uint16).itemsize, runtime=runtime)
        norm_dev = malloc(config.hidden_size * np.dtype(np.uint16).itemsize, runtime=runtime)
        logits_dev = malloc(config.vocab_size * np.dtype(np.float32).itemsize, runtime=runtime)
        stage1_blocks = lm_head_argmax_stage1_blocks(config.vocab_size)
        block_values_dev = malloc(stage1_blocks * np.dtype(np.float32).itemsize, runtime=runtime)
        block_indices_dev = malloc(stage1_blocks * np.dtype(np.int64).itemsize, runtime=runtime)
        argmax_id_dev = malloc(np.dtype(np.int64).itemsize, runtime=runtime)
        argmax_value_dev = malloc(np.dtype(np.float32).itemsize, runtime=runtime)
        buffers.extend(
            (
                token_dev,
                embedding_dev,
                norm_dev,
                logits_dev,
                block_values_dev,
                block_indices_dev,
                argmax_id_dev,
                argmax_value_dev,
            )
        )

        launch_gguf_embedding(
            resident.root("token_embedding"),
            token_dev.ptr,
            embedding_dev.ptr,
            1,
            config.hidden_size,
            config.vocab_size,
            backend=backend,
            libraries={"gguf_q4_k": libraries["embedding"]},
            runtime=runtime,
        )
        gguf_rmsnorm_bf16_f32_weight(
            embedding_dev.ptr,
            resident.root("output_norm").allocation().tensor.ptr,
            norm_dev.ptr,
            1,
            config.hidden_size,
            config.rms_norm_eps,
            library=libraries["norm"],
            runtime=runtime,
        )
        launch_gguf_linear(
            resident.root("lm_head"),
            norm_dev.ptr,
            logits_dev.ptr,
            1,
            config.hidden_size,
            config.vocab_size,
            output_dtype=GGUF_OUTPUT_F32,
            backend=backend,
            libraries={"gguf_q6_k_t16_v1": libraries["lm_head"]},
            runtime=runtime,
        )
        argmax_f32(
            logits_dev.ptr,
            block_values_dev.ptr,
            block_indices_dev.ptr,
            argmax_id_dev.ptr,
            argmax_value_dev.ptr,
            config.vocab_size,
            library=libraries["argmax"],
            runtime=runtime,
        )
        runtime.device_synchronize()

        embedding_bits = _copy_from_device(
            embedding_dev, (1, config.hidden_size), np.dtype(np.uint16), runtime
        )
        norm_bits = _copy_from_device(norm_dev, (1, config.hidden_size), np.dtype(np.uint16), runtime)
        gpu_logits = _copy_from_device(logits_dev, (config.vocab_size,), np.dtype(np.float32), runtime)
        gpu_argmax = int(_copy_from_device(argmax_id_dev, (1,), np.dtype(np.int64), runtime)[0])
        gpu_argmax_value = float(
            _copy_from_device(argmax_value_dev, (1,), np.dtype(np.float32), runtime)[0]
        )

        embedding_raw = np.ascontiguousarray(reader.tensor_data("token_embd.weight"))
        output_norm = np.asarray(reader.tensor_data("output_norm.weight"), dtype=np.float32)
        lm_head_raw = np.ascontiguousarray(reader.tensor_data("output.weight"))
        cpu_embedding = gguf_q4_k_embedding(token_ids, embedding_raw)
        cpu_embedding_bits = float_array_to_bf16_bits(cpu_embedding)
        cpu_norm = rmsnorm(
            bf16_to_float32(cpu_embedding_bits),
            output_norm,
            config.rms_norm_eps,
        )
        cpu_norm_bits = float_array_to_bf16_bits(cpu_norm)
        cpu_logits = _raw_q6_logits(
            cpu_norm_bits,
            lm_head_raw,
            chunk_rows=cpu_chunk_rows,
        )
        cpu_argmax = int(np.argmax(cpu_logits))
        embedding_abs = float(
            np.max(np.abs(bf16_to_float32(embedding_bits) - bf16_to_float32(cpu_embedding_bits)))
        )
        norm_abs = float(np.max(np.abs(bf16_to_float32(norm_bits) - bf16_to_float32(cpu_norm_bits))))
        kl = _kl_divergence(cpu_logits, gpu_logits)
        top1_agreement = float(cpu_argmax == gpu_argmax)
        finite = bool(np.all(np.isfinite(gpu_logits)))
        passed = embedding_abs == 0.0 and norm_abs == 0.0 and finite and kl <= 0.05 and top1_agreement >= 0.9
        return {
            "schema": 1,
            "model": str(model_path),
            "backend": backend,
            "token_id": token_id,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "embedding_max_abs": embedding_abs,
            "output_norm_max_abs": norm_abs,
            "logits_max_abs": float(np.max(np.abs(cpu_logits - gpu_logits))),
            "logits_mean_abs": float(np.mean(np.abs(cpu_logits - gpu_logits))),
            "kl_divergence": kl,
            "cpu_top1": cpu_argmax,
            "gpu_top1": gpu_argmax,
            "gpu_top1_value": gpu_argmax_value,
            "top1_agreement": top1_agreement,
            "finite_logits": finite,
            "pass": passed,
        }
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        resident.free(runtime=runtime)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--token-id", type=int, default=100257)
    parser.add_argument("--cpu-chunk-rows", type=int, default=2048)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--prebuild-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    compiler_version = _compiler_version(args.compiler_version_file)
    if args.prebuild_only:
        artifacts = build_probe_libraries(
            compiler_version=compiler_version,
            require_cached=args.require_cached_build,
            load=False,
        )
        result = {name: str(artifact.output_path) for name, artifact in artifacts.items()}
    else:
        result = run_laguna_root_probe(
            args.model,
            backend=args.backend,
            token_id=args.token_id,
            cpu_chunk_rows=args.cpu_chunk_rows,
            compiler_version=compiler_version,
            require_cached=args.require_cached_build,
        )
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if args.prebuild_only or bool(result.get("pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
