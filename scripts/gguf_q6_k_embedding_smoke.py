#!/usr/bin/env python3
"""HIP correctness smoke for GGUF Q6_K/Q8_0 token embedding lookup."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.cpu_reference import gguf_q6_k_embedding
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding import (
    build_gguf_q6_k_embedding,
    gguf_q6_k_embedding_bf16_out,
    gguf_q8_0_embedding_bf16_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32, dequantize_gguf_data
from tests.test_gguf_k_gemv import make_q6_k_weight

DEFAULT_MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
DEFAULT_Q8_MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q8_0.gguf")


def _compiler_version(path: Path | None) -> str | None:
    return None if path is None else path.read_text()


def _run_case(
    name: str,
    qweight: np.ndarray,
    token_ids: np.ndarray,
    *,
    hidden_size: int,
    vocab_size: int,
    expected: np.ndarray,
    kernel,
    library,
) -> float:
    runtime = get_hip_runtime()
    expected_bits = float_array_to_bf16_bits(expected)
    out = np.empty((token_ids.shape[0], hidden_size), dtype=np.uint16)
    bufs = []
    try:
        token_dev = malloc(token_ids.nbytes, runtime=runtime)
        qweight_dev = malloc(qweight.nbytes, runtime=runtime)
        out_dev = malloc(out.nbytes, runtime=runtime)
        bufs.extend((token_dev, qweight_dev, out_dev))
        copy_host_to_device(token_dev, host_array_ptr(np.ascontiguousarray(token_ids)), runtime=runtime)
        copy_host_to_device(qweight_dev, host_array_ptr(np.ascontiguousarray(qweight)), runtime=runtime)
        kernel(
            token_dev.ptr,
            qweight_dev.ptr,
            out_dev.ptr,
            int(token_ids.shape[0]),
            int(hidden_size),
            int(vocab_size),
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)
    max_abs = float(np.max(np.abs(bf16_to_float32(out) - bf16_to_float32(expected_bits))))
    print(f"{name} rows={token_ids.shape[0]} hidden_size={hidden_size} max_abs={max_abs}")
    return max_abs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--q8-model", type=Path, default=DEFAULT_Q8_MODEL)
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    args = parser.parse_args()

    library = build_gguf_q6_k_embedding(
        load=True,
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached=args.require_cached_build,
    )
    worst = 0.0
    synthetic = make_q6_k_weight(7, 512)
    synthetic_tokens = np.asarray([0, 3, 6, 3], dtype=np.int64)
    worst = max(
        worst,
        _run_case(
            "synthetic_q6_k_embedding",
            synthetic,
            synthetic_tokens,
            hidden_size=512,
            vocab_size=7,
            expected=gguf_q6_k_embedding(synthetic_tokens, synthetic),
            kernel=gguf_q6_k_embedding_bf16_out,
            library=library,
        ),
    )
    if not args.skip_real and args.model.exists():
        reader = GGUFReader(args.model)
        tensor = reader.tensor_info("token_embd.weight")
        qweight = np.asarray(reader.tensor_data("token_embd.weight"))
        token_ids = np.asarray([760, 4087, 369, 760], dtype=np.int64)
        worst = max(
            worst,
            _run_case(
                "real_q6_k_token_embd",
                qweight,
                token_ids,
                hidden_size=tensor.shape[1],
                vocab_size=tensor.shape[0],
                expected=gguf_q6_k_embedding(token_ids, qweight),
                kernel=gguf_q6_k_embedding_bf16_out,
                library=library,
            ),
        )
    if not args.skip_real and args.q8_model.exists():
        reader = GGUFReader(args.q8_model)
        tensor = reader.tensor_info("token_embd.weight")
        qweight = np.asarray(reader.tensor_data("token_embd.weight"))
        token_ids = np.asarray([760, 4087, 369, 760], dtype=np.int64)
        worst = max(
            worst,
            _run_case(
                "real_q8_0_token_embd",
                qweight,
                token_ids,
                hidden_size=tensor.shape[1],
                vocab_size=tensor.shape[0],
                expected=dequantize_gguf_data(qweight[token_ids], tensor.ggml_type),
                kernel=gguf_q8_0_embedding_bf16_out,
                library=library,
            ),
        )
    print(f"worst_max_abs={worst}")
    if worst != 0.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
