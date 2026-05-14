#!/usr/bin/env python3
"""Early scaffold smokes.

Default modes are CPU-only and safe before GPU clearance. ``smoke-add-hip`` is the explicit
GPU/JIT path for the first raw-pointer HIP smoke kernel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.dispatch.fusion import FusionPlanner, resolve_plan
from hipengine.kernels.registry import MissingKernelError
from hipengine.models import resolve_model

DEFAULT_QWEN35_PARO_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "registry",
            "cpu-fixtures",
            "smoke-add-plan",
            "smoke-add-hip",
            "qwen35-rmsnorm-hip",
            "paro-rmsnorm-hip",
            "qwen35-router-hip",
            "qwen35-rotary-hip",
            "qwen35-linear-attn-conv-hip",
            "qwen35-linear-attn-gdn-hip",
            "qwen35-linear-attn-prefill-hip",
            "qwen35-paged-kv-write-hip",
            "qwen35-full-attn-decode-hip",
            "qwen35-paged-attn-decode-hip",
            "qwen35-paged-attn-split-k-hip",
            "qwen35-paged-attn-gate-hip",
            "qwen35-paged-attn-gate-bf16-hip",
            "qwen35-paged-attn-gqa-hip",
            "qwen35-paged-attn-gqa-state-hip",
            "paro-selected-gemv-hip",
            "paro-selected-gemv-rotate-hip",
            "paro-pack8-gemv-hip",
            "paro-rotate-hip",
            "paro-silu-hip",
            "paro-combine-hip",
            "dense-gemv-hip",
            "lm-head-hip",
            "w8a16-linear-hip",
            "w8a16-shared-expert-hip",
            "paro-moe-c1-hip",
            "paro-moe-c1-state-hip",
            "qwen35-paro-generate-hip",
        ),
        default="registry",
    )
    parser.add_argument("--n", type=int, default=1024, help="Element count for smoke-add-hip.")
    parser.add_argument("--rows", type=int, default=2, help="Rows/tokens for HIP smoke modes.")
    parser.add_argument(
        "--model",
        default=DEFAULT_QWEN35_PARO_MODEL,
        help="Model path for qwen35-paro-generate-hip.",
    )
    parser.add_argument("--prompt", default="Hello", help="Prompt for qwen35-paro-generate-hip.")
    parser.add_argument("--max-tokens", type=int, default=1, help="Max tokens for generate smoke.")
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=16,
        help="Hidden/input feature size for HIP smoke modes.",
    )
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=None,
        help=(
            "Read the precomputed hipcc --version text from this file before building/loading "
            "HIP smoke libraries. Use under rocprofv3 to avoid spawning hipcc inside the profiler."
        ),
    )
    parser.add_argument(
        "--require-cached-build",
        action="store_true",
        help="Fail instead of invoking hipcc if the expected HIP cache artifact is absent.",
    )
    args = parser.parse_args()
    if args.mode == "registry":
        return registry_smoke()
    if args.mode == "cpu-fixtures":
        return cpu_fixture_smoke()
    if args.mode == "smoke-add-plan":
        return smoke_add_plan_smoke()
    if args.mode == "qwen35-paro-generate-hip":
        return qwen35_paro_generate_hip_smoke(args.model, args.prompt, args.max_tokens)
    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = _read_compiler_version(args.compiler_version_file)
    if args.mode == "smoke-add-hip":
        return smoke_add_hip_smoke(
            args.n,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "qwen35-rmsnorm-hip":
        return qwen35_rmsnorm_hip_smoke(
            args.rows,
            args.hidden_size,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "paro-rmsnorm-hip":
        return paro_rmsnorm_hip_smoke(
            args.rows,
            args.hidden_size,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "qwen35-router-hip":
        return qwen35_router_hip_smoke(
            args.rows,
            args.hidden_size,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "qwen35-rotary-hip":
        return qwen35_rotary_hip_smoke(
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "qwen35-linear-attn-conv-hip":
        return qwen35_linear_attn_conv_hip_smoke(
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "qwen35-linear-attn-gdn-hip":
        return qwen35_linear_attn_gdn_hip_smoke(
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "qwen35-linear-attn-prefill-hip":
        return qwen35_linear_attn_prefill_hip_smoke(
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "qwen35-paged-kv-write-hip":
        return qwen35_paged_kv_write_hip_smoke(
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "qwen35-full-attn-decode-hip":
        return qwen35_full_attn_decode_hip_smoke(
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "qwen35-paged-attn-decode-hip":
        return qwen35_paged_attn_decode_hip_smoke(
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "qwen35-paged-attn-split-k-hip":
        return qwen35_paged_attn_split_k_hip_smoke(
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "qwen35-paged-attn-gate-hip":
        return qwen35_paged_attn_gate_hip_smoke(
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "qwen35-paged-attn-gate-bf16-hip":
        return qwen35_paged_attn_gate_bf16_hip_smoke(
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "qwen35-paged-attn-gqa-hip":
        return qwen35_paged_attn_gqa_hip_smoke(
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "qwen35-paged-attn-gqa-state-hip":
        return qwen35_paged_attn_gqa_state_hip_smoke(
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "paro-selected-gemv-hip":
        return paro_selected_gemv_hip_smoke(
            args.rows,
            args.hidden_size,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "paro-selected-gemv-rotate-hip":
        return paro_selected_gemv_rotate_hip_smoke(
            args.rows,
            args.hidden_size,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "paro-pack8-gemv-hip":
        return paro_pack8_gemv_hip_smoke(
            args.rows,
            args.hidden_size,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "paro-rotate-hip":
        return paro_rotate_hip_smoke(
            args.rows,
            args.hidden_size,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "paro-silu-hip":
        return paro_silu_hip_smoke(
            args.rows,
            args.hidden_size,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "paro-combine-hip":
        return paro_combine_hip_smoke(
            args.rows,
            args.hidden_size,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "dense-gemv-hip":
        return dense_gemv_hip_smoke(
            args.rows,
            args.hidden_size,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "lm-head-hip":
        return lm_head_hip_smoke(
            args.hidden_size,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "w8a16-linear-hip":
        return w8a16_linear_hip_smoke(
            args.rows,
            args.hidden_size,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "w8a16-shared-expert-hip":
        return w8a16_shared_expert_hip_smoke(
            args.rows,
            args.hidden_size,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    if args.mode == "paro-moe-c1-state-hip":
        return paro_moe_c1_state_hip_smoke(
            args.hidden_size,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
    return paro_moe_c1_hip_smoke(
        args.hidden_size,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )


def registry_smoke() -> int:
    model = resolve_model("HipEngineToyForCausalLM")
    planner = FusionPlanner(backend="hip_gfx1100", quant="fp16")
    plan = planner.plan(model.layer_sequence())
    print("plan:", " -> ".join(step.layer for step in plan))
    try:
        resolve_plan(plan)
    except MissingKernelError as exc:
        print("expected missing kernel:", exc)
        return 0
    print("unexpected: plan resolved even though no kernels are registered")
    return 1


def cpu_fixture_smoke() -> int:
    from hipengine.kernels.cpu_reference import register_cpu_reference_kernels
    from hipengine.kernels.cpu_reference.fixtures import load_fixture, run_fixture

    register_cpu_reference_kernels()
    fixture_dir = Path("tests/fixtures/cpu_reference")
    failed = 0
    for path in sorted(fixture_dir.glob("*.json")):
        result = run_fixture(load_fixture(path))
        print(f"{'PASS' if result.passed else 'FAIL'} {path} max_abs={result.max_abs:.6g}")
        failed += 0 if result.passed else 1
    return 1 if failed else 0


def smoke_add_plan_smoke() -> int:
    from hipengine.kernels.hip_gfx1100.smoke import plan_smoke_add_build

    artifact = plan_smoke_add_build()
    print("family:", artifact.family)
    print("profile:", artifact.profile.name)
    print("output:", artifact.output_path)
    print("command:", " ".join(artifact.command))
    print("dry-run only: no hipcc invocation, no GPU access")
    return 0


def qwen35_paro_generate_hip_smoke(model: str, prompt: str, max_tokens: int) -> int:
    from hipengine import LLM, SamplingParams

    llm = LLM(model, backend="hip_gfx1100", quant="w4_paro")
    outputs = llm.generate(
        prompt,
        SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0),
    )
    print(
        json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "outputs": outputs,
                "max_tokens": max_tokens,
                "path": "LLM.generate/qwen3_5_moe_paro/hip_gfx1100/w4_paro",
            },
            ensure_ascii=False,
        )
    )
    return 0


def lm_head_hip_smoke(
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.linear import (
        build_lm_head,
        lm_head_argmax_stage1_blocks,
        lm_head_fp16_argmax_bf16,
    )

    vocab_size = 257
    hidden_f32 = np.linspace(-0.75, 0.9, hidden_size, dtype=np.float32)
    hidden_bits = _float32_to_bf16_bits(hidden_f32)
    hidden_bf32 = _bf16_bits_to_float32(hidden_bits)
    weight_f32 = np.asarray(
        [[((row * 17 + col * 5) % 31 - 15) / 16.0 for col in range(hidden_size)] for row in range(vocab_size)],
        dtype=np.float32,
    )
    weight_f16 = np.ascontiguousarray(weight_f32.astype(np.float16))
    expected_logits = weight_f16.astype(np.float32) @ hidden_bf32.astype(np.float32)
    expected_id = int(np.argmax(expected_logits))
    expected_logit = float(expected_logits[expected_id])

    runtime = get_hip_runtime()
    library = build_lm_head(load=True, compiler_version=compiler_version, require_cached=require_cached_build)
    threads = 256
    stage1_blocks = lm_head_argmax_stage1_blocks(vocab_size, threads=threads)
    buffers = []
    try:
        hidden_dev = malloc(hidden_bits.nbytes, runtime=runtime); buffers.append(hidden_dev)
        weight_dev = malloc(weight_f16.nbytes, runtime=runtime); buffers.append(weight_dev)
        logits_dev = malloc(vocab_size * np.dtype(np.float32).itemsize, runtime=runtime); buffers.append(logits_dev)
        block_values_dev = malloc(stage1_blocks * np.dtype(np.float32).itemsize, runtime=runtime); buffers.append(block_values_dev)
        block_indices_dev = malloc(stage1_blocks * np.dtype(np.int64).itemsize, runtime=runtime); buffers.append(block_indices_dev)
        out_index_dev = malloc(np.dtype(np.int64).itemsize, runtime=runtime); buffers.append(out_index_dev)
        out_value_dev = malloc(np.dtype(np.float32).itemsize, runtime=runtime); buffers.append(out_value_dev)
        copy_host_to_device(hidden_dev, host_array_ptr(hidden_bits), runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(weight_f16), runtime=runtime)
        lm_head_fp16_argmax_bf16(
            hidden_dev.ptr,
            weight_dev.ptr,
            logits_dev.ptr,
            block_values_dev.ptr,
            block_indices_dev.ptr,
            out_index_dev.ptr,
            out_value_dev.ptr,
            hidden_size,
            vocab_size,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        out_index = np.empty((1,), dtype=np.int64)
        out_value = np.empty((1,), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out_index), out_index_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(out_value), out_value_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    index_match = int(out_index[0]) == expected_id
    logit_abs = abs(float(out_value[0]) - expected_logit)
    print(f"lm_head_id={int(out_index[0])} expected_id={expected_id} index_match={index_match}")
    print(f"lm_head_logit={float(out_value[0])} expected_logit={expected_logit} abs={logit_abs}")
    return 0 if index_match and logit_abs <= 1.0e-5 else 1


def _read_compiler_version(path: Path) -> str:
    text = path.expanduser().read_text().strip()
    if not text:
        raise ValueError(f"compiler version file is empty: {path}")
    return text


def smoke_add_hip_smoke(
    n: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.smoke import build_smoke_add, smoke_add_f32

    if n < 1:
        raise ValueError("--n must be >= 1")

    a_host = np.arange(n, dtype=np.float32)
    b_host = np.arange(n, dtype=np.float32) * 2.0 + 1.0
    out_host = np.empty_like(a_host)

    runtime = get_hip_runtime()
    library = build_smoke_add(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    a_dev = b_dev = out_dev = None
    try:
        a_dev = malloc(a_host.nbytes, runtime=runtime)
        b_dev = malloc(b_host.nbytes, runtime=runtime)
        out_dev = malloc(out_host.nbytes, runtime=runtime)
        copy_host_to_device(a_dev, host_array_ptr(a_host), runtime=runtime)
        copy_host_to_device(b_dev, host_array_ptr(b_host), runtime=runtime)
        smoke_add_f32(a_dev.ptr, b_dev.ptr, out_dev.ptr, n, library=library, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_host), out_dev, runtime=runtime)
    finally:
        for buffer in (out_dev, b_dev, a_dev):
            if buffer is not None:
                free(buffer, runtime=runtime)

    expected = a_host + b_host
    max_abs = float(np.max(np.abs(out_host - expected)))
    print(f"n={n} max_abs={max_abs}")
    print("first5=", out_host[:5].tolist())
    return 0 if np.allclose(out_host, expected) else 1


def qwen35_rmsnorm_hip_smoke(
    rows: int,
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.norm import build_qwen35_rmsnorm, qwen35_rmsnorm_bf16

    if rows < 1:
        raise ValueError("--rows must be >= 1")
    if hidden_size < 1:
        raise ValueError("--hidden-size must be >= 1")

    x_f32 = np.linspace(-1.5, 2.0, rows * hidden_size, dtype=np.float32).reshape(
        rows, hidden_size
    )
    weight_delta_f32 = np.linspace(-0.25, 0.25, hidden_size, dtype=np.float32)
    x_bits = _float32_to_bf16_bits(x_f32)
    weight_bits = _float32_to_bf16_bits(weight_delta_f32)
    out_bits = np.empty_like(x_bits)

    x_bf32 = _bf16_bits_to_float32(x_bits)
    weight_delta_bf32 = _bf16_bits_to_float32(weight_bits)
    inv_rms = np.reciprocal(np.sqrt(np.mean(x_bf32 * x_bf32, axis=-1, keepdims=True) + 1e-6))
    expected_bits = _float32_to_bf16_bits(x_bf32 * inv_rms * (1.0 + weight_delta_bf32))
    expected = _bf16_bits_to_float32(expected_bits)

    runtime = get_hip_runtime()
    library = build_qwen35_rmsnorm(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    x_dev = weight_dev = out_dev = None
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        weight_dev = malloc(weight_bits.nbytes, runtime=runtime)
        out_dev = malloc(out_bits.nbytes, runtime=runtime)
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(weight_bits), runtime=runtime)
        qwen35_rmsnorm_bf16(
            x_dev.ptr,
            weight_dev.ptr,
            out_dev.ptr,
            rows,
            hidden_size,
            1e-6,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_bits), out_dev, runtime=runtime)
    finally:
        for buffer in (out_dev, weight_dev, x_dev):
            if buffer is not None:
                free(buffer, runtime=runtime)

    out = _bf16_bits_to_float32(out_bits)
    max_abs = float(np.max(np.abs(out - expected)))
    bit_mismatch = int(np.count_nonzero(out_bits != expected_bits))
    print(f"rows={rows} hidden_size={hidden_size} max_abs={max_abs} bit_mismatch={bit_mismatch}")
    print("first_row=", out[0, : min(5, hidden_size)].tolist())
    return 0 if np.allclose(out, expected, atol=2e-2, rtol=2e-2) else 1


def paro_rmsnorm_hip_smoke(
    rows: int,
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.norm import (
        build_qwen35_rmsnorm,
        paro_add_rmsnorm_out_bf16,
        paro_add_rmsnorm_out_fp16,
        paro_rmsnorm_out_bf16,
        paro_rmsnorm_out_fp16,
    )

    if rows < 1:
        raise ValueError("--rows must be >= 1")
    if hidden_size < 1:
        raise ValueError("--hidden-size must be >= 1")

    x_f32 = np.linspace(-1.25, 1.75, rows * hidden_size, dtype=np.float32).reshape(
        rows, hidden_size
    )
    add_f32 = np.linspace(0.5, -0.75, rows * hidden_size, dtype=np.float32).reshape(
        rows, hidden_size
    )
    weight_f32 = np.linspace(0.75, 1.25, hidden_size, dtype=np.float32)
    x_bits = _float32_to_bf16_bits(x_f32)
    add_bits = _float32_to_bf16_bits(add_f32)
    weight_bits = _float32_to_bf16_bits(weight_f32)
    norm_out_bits = np.empty_like(x_bits)
    add_norm_out_bits = np.empty_like(x_bits)
    residual_out_bits = np.empty_like(x_bits)

    x_bf32 = _bf16_bits_to_float32(x_bits)
    add_bf32 = _bf16_bits_to_float32(add_bits)
    weight_bf32 = _bf16_bits_to_float32(weight_bits)

    inv_rms = np.reciprocal(np.sqrt(np.mean(x_bf32 * x_bf32, axis=-1, keepdims=True) + 1e-6))
    expected_norm_bits = _float32_to_bf16_bits(x_bf32 * inv_rms * weight_bf32)
    expected_norm = _bf16_bits_to_float32(expected_norm_bits)

    residual_bits = _float32_to_bf16_bits(x_bf32 + add_bf32)
    residual_bf32 = _bf16_bits_to_float32(residual_bits)
    add_inv_rms = np.reciprocal(
        np.sqrt(np.mean(residual_bf32 * residual_bf32, axis=-1, keepdims=True) + 1e-6)
    )
    expected_add_norm_bits = _float32_to_bf16_bits(residual_bf32 * add_inv_rms * weight_bf32)
    expected_add_norm = _bf16_bits_to_float32(expected_add_norm_bits)

    runtime = get_hip_runtime()
    library = build_qwen35_rmsnorm(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    x_dev = add_dev = weight_dev = norm_out_dev = add_norm_out_dev = residual_out_dev = None
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        add_dev = malloc(add_bits.nbytes, runtime=runtime)
        weight_dev = malloc(weight_bits.nbytes, runtime=runtime)
        norm_out_dev = malloc(norm_out_bits.nbytes, runtime=runtime)
        add_norm_out_dev = malloc(add_norm_out_bits.nbytes, runtime=runtime)
        residual_out_dev = malloc(residual_out_bits.nbytes, runtime=runtime)
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(add_dev, host_array_ptr(add_bits), runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(weight_bits), runtime=runtime)
        paro_rmsnorm_out_bf16(
            x_dev.ptr,
            weight_dev.ptr,
            norm_out_dev.ptr,
            rows,
            hidden_size,
            1e-6,
            library=library,
            runtime=runtime,
        )
        paro_add_rmsnorm_out_bf16(
            x_dev.ptr,
            add_dev.ptr,
            weight_dev.ptr,
            add_norm_out_dev.ptr,
            residual_out_dev.ptr,
            rows,
            hidden_size,
            1e-6,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(norm_out_bits), norm_out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(add_norm_out_bits), add_norm_out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(residual_out_bits), residual_out_dev, runtime=runtime)
    finally:
        for buffer in (
            residual_out_dev,
            add_norm_out_dev,
            norm_out_dev,
            weight_dev,
            add_dev,
            x_dev,
        ):
            if buffer is not None:
                free(buffer, runtime=runtime)

    norm_out = _bf16_bits_to_float32(norm_out_bits)
    add_norm_out = _bf16_bits_to_float32(add_norm_out_bits)
    residual_out = _bf16_bits_to_float32(residual_out_bits)
    norm_max_abs = float(np.max(np.abs(norm_out - expected_norm)))
    add_norm_max_abs = float(np.max(np.abs(add_norm_out - expected_add_norm)))
    residual_max_abs = float(np.max(np.abs(residual_out - residual_bf32)))
    norm_bit_mismatch = int(np.count_nonzero(norm_out_bits != expected_norm_bits))
    add_norm_bit_mismatch = int(np.count_nonzero(add_norm_out_bits != expected_add_norm_bits))
    residual_bit_mismatch = int(np.count_nonzero(residual_out_bits != residual_bits))

    x_fp16 = x_f32.astype(np.float16)
    add_fp16 = add_f32.astype(np.float16)
    weight_fp16 = weight_f32.astype(np.float16)
    norm_out_fp16 = np.empty_like(x_fp16)
    add_norm_out_fp16 = np.empty_like(x_fp16)
    residual_out_fp16 = np.empty_like(x_fp16)
    x_fp32 = x_fp16.astype(np.float32)
    add_fp32 = add_fp16.astype(np.float32)
    weight_fp32 = weight_fp16.astype(np.float32)
    fp16_inv_rms = np.reciprocal(
        np.sqrt(np.mean(x_fp32 * x_fp32, axis=-1, keepdims=True) + 1e-6)
    )
    expected_norm_fp16 = (x_fp32 * fp16_inv_rms * weight_fp32).astype(np.float16)
    expected_residual_fp16 = (x_fp32 + add_fp32).astype(np.float16)
    expected_residual_fp32 = expected_residual_fp16.astype(np.float32)
    fp16_add_inv_rms = np.reciprocal(
        np.sqrt(
            np.mean(
                expected_residual_fp32 * expected_residual_fp32,
                axis=-1,
                keepdims=True,
            )
            + 1e-6
        )
    )
    expected_add_norm_fp16 = (
        expected_residual_fp32 * fp16_add_inv_rms * weight_fp32
    ).astype(np.float16)

    fp16_buffers = []
    try:
        for array in (
            x_fp16,
            add_fp16,
            weight_fp16,
            norm_out_fp16,
            add_norm_out_fp16,
            residual_out_fp16,
        ):
            buffer = malloc(array.nbytes, runtime=runtime)
            fp16_buffers.append(buffer)
        copy_host_to_device(fp16_buffers[0], host_array_ptr(x_fp16), runtime=runtime)
        copy_host_to_device(fp16_buffers[1], host_array_ptr(add_fp16), runtime=runtime)
        copy_host_to_device(fp16_buffers[2], host_array_ptr(weight_fp16), runtime=runtime)
        paro_rmsnorm_out_fp16(
            fp16_buffers[0].ptr,
            fp16_buffers[2].ptr,
            fp16_buffers[3].ptr,
            rows,
            hidden_size,
            1e-6,
            library=library,
            runtime=runtime,
        )
        paro_add_rmsnorm_out_fp16(
            fp16_buffers[0].ptr,
            fp16_buffers[1].ptr,
            fp16_buffers[2].ptr,
            fp16_buffers[4].ptr,
            fp16_buffers[5].ptr,
            rows,
            hidden_size,
            1e-6,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(norm_out_fp16), fp16_buffers[3], runtime=runtime)
        copy_device_to_host(host_array_ptr(add_norm_out_fp16), fp16_buffers[4], runtime=runtime)
        copy_device_to_host(host_array_ptr(residual_out_fp16), fp16_buffers[5], runtime=runtime)
    finally:
        for buffer in reversed(fp16_buffers):
            free(buffer, runtime=runtime)

    norm_fp16_mismatch = int(
        np.count_nonzero(norm_out_fp16.view(np.uint16) != expected_norm_fp16.view(np.uint16))
    )
    add_norm_fp16_mismatch = int(
        np.count_nonzero(
            add_norm_out_fp16.view(np.uint16) != expected_add_norm_fp16.view(np.uint16)
        )
    )
    residual_fp16_mismatch = int(
        np.count_nonzero(
            residual_out_fp16.view(np.uint16) != expected_residual_fp16.view(np.uint16)
        )
    )
    print(
        f"rows={rows} hidden_size={hidden_size} "
        f"norm_max_abs={norm_max_abs} norm_bit_mismatch={norm_bit_mismatch} "
        f"add_norm_max_abs={add_norm_max_abs} add_norm_bit_mismatch={add_norm_bit_mismatch} "
        f"residual_max_abs={residual_max_abs} residual_bit_mismatch={residual_bit_mismatch} "
        f"fp16_norm_mismatch={norm_fp16_mismatch} "
        f"fp16_add_norm_mismatch={add_norm_fp16_mismatch} "
        f"fp16_residual_mismatch={residual_fp16_mismatch}"
    )
    print("first_row=", norm_out[0, : min(5, hidden_size)].tolist())
    return 0 if (
        norm_bit_mismatch == 0
        and add_norm_bit_mismatch == 0
        and residual_bit_mismatch == 0
        and norm_fp16_mismatch == 0
        and add_norm_fp16_mismatch == 0
        and residual_fp16_mismatch == 0
    ) else 1












def qwen35_paged_attn_gqa_state_hip_smoke(
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.core.tensor import Tensor
    from hipengine.kernels.hip_gfx1100.attention import build_qwen35_paged_attn_decode, build_qwen35_paged_kv_write
    from hipengine.kvcache import KVLiveSpans
    from hipengine.loading.materialize import DeviceWeightMap
    from hipengine.loading.qwen35_paro import Qwen35ParoConfig, Qwen35ParoLayerDeviceWeights
    from hipengine.runtime import Qwen35ParoDecodeState, RuntimeWorkspace

    block_size = 256
    blocks = 2
    context_len = 512
    append_position = context_len - 1
    chunk_size = 256
    num_splits = 2
    num_q_heads = 16
    num_kv_heads = 2
    head_dim = 256
    hidden_size = num_q_heads * head_dim
    scale = head_dim ** -0.5
    block_table = np.asarray([0, 1], dtype=np.int32)
    live_counts = np.asarray([append_position], dtype=np.int64)
    live_counts_decode = np.asarray([context_len], dtype=np.int64)

    query_grid = np.arange(num_q_heads * head_dim, dtype=np.float32).reshape(num_q_heads, head_dim)
    query = ((query_grid % 97.0) - 48.0) / 128.0
    token_grid = np.arange(context_len * num_kv_heads * head_dim, dtype=np.float32).reshape(
        context_len, num_kv_heads, head_dim
    )
    key_tokens = ((token_grid % 89.0) - 44.0) / 96.0
    value_tokens = (37.0 - (token_grid % 83.0)) / 80.0
    gate_grid = np.arange(num_q_heads * head_dim, dtype=np.float32).reshape(num_q_heads, head_dim)
    gate_f32 = ((gate_grid % 31.0) - 15.0) / 8.0
    gate_bits = _float32_to_bf16_bits(gate_f32)
    key_bits = _float32_to_bf16_bits(key_tokens)
    value_bits = _float32_to_bf16_bits(value_tokens)
    key_cache = np.zeros((blocks, block_size, num_kv_heads, head_dim), dtype=np.uint16)
    value_cache = np.zeros_like(key_cache)
    for token in range(context_len - 1):
        key_cache[token // block_size, token % block_size] = key_bits[token]
        value_cache[token // block_size, token % block_size] = value_bits[token]

    key_ref = _bf16_bits_to_float32(key_bits)
    value_ref = _bf16_bits_to_float32(value_bits)
    gate_ref = _bf16_bits_to_float32(gate_bits)
    expected = np.empty((num_q_heads, head_dim), dtype=np.float32)
    for q_head in range(num_q_heads):
        kv_head = q_head // 8
        scores = np.asarray(
            [np.dot(query[q_head], key_ref[token, kv_head]) * scale for token in range(context_len)],
            dtype=np.float32,
        )
        scores = scores - np.max(scores)
        probs = np.exp(scores, dtype=np.float32)
        probs = probs / np.sum(probs, dtype=np.float32)
        expected[q_head] = probs @ value_ref[:, kv_head, :]
    expected_gate = _float32_to_bf16_bits(expected * (1.0 / (1.0 + np.exp(-gate_ref, dtype=np.float32))))

    runtime = get_hip_runtime()
    libraries = {
        "kv": build_qwen35_paged_kv_write(load=True, compiler_version=compiler_version, require_cached=require_cached_build),
        "attention": build_qwen35_paged_attn_decode(load=True, compiler_version=compiler_version, require_cached=require_cached_build),
    }
    buffers = []
    state = None

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        block_table_dev = dev(block_table)
        live_counts_dev = dev(live_counts)
        key_cache_dev = dev(key_cache)
        value_cache_dev = dev(value_cache)
        config = Qwen35ParoConfig(
            architecture="Qwen3_5MoeForConditionalGeneration",
            num_hidden_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            num_experts=0,
            num_experts_per_tok=0,
            moe_intermediate_size=0,
            shared_expert_intermediate_size=0,
            layer_types=("full_attention",),
            quant_method="paroquant",
        )
        state = Qwen35ParoDecodeState(
            layer_weights=Qwen35ParoLayerDeviceWeights(config=config, layer_id=0, weights=DeviceWeightMap({})),
            workspace=RuntimeWorkspace(runtime=runtime),
            runtime=runtime,
        )
        scratch = state.reserve_full_attention_scratch(tokens=1, num_splits=num_splits, gated_dtype="bf16")
        copy_host_to_device(state.workspace.allocation("attn.query").buffer, host_array_ptr(query.reshape(1, num_q_heads, head_dim)), runtime=runtime)
        copy_host_to_device(state.workspace.allocation("attn.key").buffer, host_array_ptr(key_tokens[append_position : append_position + 1]), runtime=runtime)
        copy_host_to_device(state.workspace.allocation("attn.value").buffer, host_array_ptr(value_bits[append_position : append_position + 1]), runtime=runtime)
        copy_host_to_device(state.workspace.allocation("attn.gate").buffer, host_array_ptr(gate_bits.reshape(1, num_q_heads, head_dim)), runtime=runtime)
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(block_table_dev.ptr, block_table.shape, "int32", Device("hip", 0)),
            live_counts=Tensor.from_handle(live_counts_dev.ptr, live_counts.shape, "int64", Device("hip", 0)),
            max_live_count=int(append_position),
            storage_dtype="bf16",
        )
        state.append_full_attention_kv(scratch, key_cache=Tensor.from_handle(key_cache_dev.ptr, key_cache.shape, "bf16", Device("hip", 0)), value_cache=Tensor.from_handle(value_cache_dev.ptr, value_cache.shape, "bf16", Device("hip", 0)), spans=spans, block_size=block_size, library=libraries)
        copy_host_to_device(live_counts_dev, host_array_ptr(live_counts_decode), runtime=runtime)
        out = state.decode_full_attention_gqa_gate_bf16(
            scratch,
            key_cache=Tensor.from_handle(key_cache_dev.ptr, key_cache.shape, "bf16", Device("hip", 0)),
            value_cache=Tensor.from_handle(value_cache_dev.ptr, value_cache.shape, "bf16", Device("hip", 0)),
            spans=spans,
            chunk_size=chunk_size,
            num_splits=num_splits,
            block_size=block_size,
            scale=scale,
            library=libraries,
        )
        runtime.device_synchronize()
        out_bits = np.empty((num_q_heads, head_dim), dtype=np.uint16)
        appended_key_cache = np.empty_like(key_cache)
        appended_value_cache = np.empty_like(value_cache)
        copy_device_to_host(host_array_ptr(out_bits), state.workspace.allocation("attn.gated").buffer, runtime=runtime)
        copy_device_to_host(host_array_ptr(appended_key_cache), key_cache_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(appended_value_cache), value_cache_dev, runtime=runtime)
        _ = out
        state.free()
        state = None
    finally:
        if state is not None:
            state.free()
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    appended_key_mismatch = int(
        np.count_nonzero(appended_key_cache[append_position // block_size, append_position % block_size] != key_bits[append_position])
    )
    appended_value_mismatch = int(
        np.count_nonzero(appended_value_cache[append_position // block_size, append_position % block_size] != value_bits[append_position])
    )
    gate_mismatch = int(np.count_nonzero(out_bits != expected_gate))
    gate_max_abs = float(np.max(np.abs(_bf16_bits_to_float32(out_bits) - _bf16_bits_to_float32(expected_gate))))
    print(
        f"context_len={context_len} chunk_size={chunk_size} num_splits={num_splits} state_path=1 "
        f"appended_key_mismatch={appended_key_mismatch} appended_value_mismatch={appended_value_mismatch} "
        f"gqa_gate_bf16_mismatch={gate_mismatch} gqa_gate_bf16_max_abs={gate_max_abs:.3g}"
    )
    print("gqa_state_attn_head0=", _bf16_bits_to_float32(out_bits)[0, :8].tolist())
    return 0 if appended_key_mismatch == 0 and appended_value_mismatch == 0 and gate_mismatch == 0 else 1

def qwen35_paged_attn_gqa_hip_smoke(
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.core.tensor import Tensor
    from hipengine.kernels.hip_gfx1100.attention import (
        build_qwen35_paged_attn_decode,
        qwen35_paged_full_attn_decode_split_k_gqa_bf16_spans,
        qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans,
        qwen35_paged_full_attn_decode_split_k_warp_bf16_spans,
    )
    from hipengine.kvcache import KVLiveSpans

    block_size = 256
    blocks = 2
    context_len = 512
    chunk_size = 256
    num_splits = 2
    num_q_heads = 16
    num_kv_heads = 2
    head_dim = 256
    scale = head_dim ** -0.5
    block_table = np.asarray([0, 1], dtype=np.int32)
    live_counts = np.asarray([context_len], dtype=np.int64)
    query_grid = np.arange(num_q_heads * head_dim, dtype=np.float32).reshape(num_q_heads, head_dim)
    query = ((query_grid % 97.0) - 48.0) / 128.0
    token_grid = np.arange(context_len * num_kv_heads * head_dim, dtype=np.float32).reshape(
        context_len, num_kv_heads, head_dim
    )
    key_tokens = ((token_grid % 89.0) - 44.0) / 96.0
    value_tokens = (37.0 - (token_grid % 83.0)) / 80.0
    gate_grid = np.arange(num_q_heads * head_dim, dtype=np.float32).reshape(num_q_heads, head_dim)
    gate_f32 = ((gate_grid % 31.0) - 15.0) / 8.0
    gate = _float32_to_bf16_bits(gate_f32)
    key_cache = np.zeros((blocks, block_size, num_kv_heads, head_dim), dtype=np.uint16)
    value_cache = np.zeros_like(key_cache)
    key_bits = _float32_to_bf16_bits(key_tokens)
    value_bits = _float32_to_bf16_bits(value_tokens)
    for token in range(context_len):
        key_cache[token // block_size, token % block_size] = key_bits[token]
        value_cache[token // block_size, token % block_size] = value_bits[token]
    warp_out = np.empty((num_q_heads, head_dim), dtype=np.float32)
    gqa_out = np.empty_like(warp_out)
    gqa_gate_out = np.empty((num_q_heads, head_dim), dtype=np.uint16)
    warp_partial_out = np.zeros((num_q_heads, num_splits, head_dim), dtype=np.float32)
    warp_partial_m = np.zeros((num_q_heads, num_splits), dtype=np.float32)
    warp_partial_l = np.zeros((num_q_heads, num_splits), dtype=np.float32)
    gqa_partial_out = np.zeros_like(warp_partial_out)
    gqa_partial_m = np.zeros_like(warp_partial_m)
    gqa_partial_l = np.zeros_like(warp_partial_l)
    gate_partial_out = np.zeros_like(warp_partial_out)
    gate_partial_m = np.zeros_like(warp_partial_m)
    gate_partial_l = np.zeros_like(warp_partial_l)

    key_ref = _bf16_bits_to_float32(key_bits)
    value_ref = _bf16_bits_to_float32(value_bits)
    gate_ref = _bf16_bits_to_float32(gate)
    expected = np.empty((num_q_heads, head_dim), dtype=np.float32)
    for q_head in range(num_q_heads):
        kv_head = q_head // 8
        scores = np.asarray(
            [np.dot(query[q_head], key_ref[token, kv_head]) * scale for token in range(context_len)],
            dtype=np.float32,
        )
        scores = scores - np.max(scores)
        probs = np.exp(scores, dtype=np.float32)
        probs = probs / np.sum(probs, dtype=np.float32)
        expected[q_head] = probs @ value_ref[:, kv_head, :]
    expected_gate = _float32_to_bf16_bits(expected * (1.0 / (1.0 + np.exp(-gate_ref, dtype=np.float32))))

    runtime = get_hip_runtime()
    library = build_qwen35_paged_attn_decode(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        block_table_dev = dev(block_table)
        live_counts_dev = dev(live_counts)
        query_dev = dev(query)
        gate_dev = dev(gate)
        key_cache_dev = dev(key_cache)
        value_cache_dev = dev(value_cache)
        warp_out_dev = out_dev(warp_out)
        gqa_out_dev = out_dev(gqa_out)
        gqa_gate_out_dev = out_dev(gqa_gate_out)
        warp_partial_out_dev = out_dev(warp_partial_out)
        warp_partial_m_dev = out_dev(warp_partial_m)
        warp_partial_l_dev = out_dev(warp_partial_l)
        gqa_partial_out_dev = out_dev(gqa_partial_out)
        gqa_partial_m_dev = out_dev(gqa_partial_m)
        gqa_partial_l_dev = out_dev(gqa_partial_l)
        gate_partial_out_dev = out_dev(gate_partial_out)
        gate_partial_m_dev = out_dev(gate_partial_m)
        gate_partial_l_dev = out_dev(gate_partial_l)
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(block_table_dev.ptr, block_table.shape, "int32", Device("hip", 0)),
            live_counts=Tensor.from_handle(live_counts_dev.ptr, live_counts.shape, "int64", Device("hip", 0)),
            max_live_count=int(context_len),
            storage_dtype="bf16",
        )
        qwen35_paged_full_attn_decode_split_k_warp_bf16_spans(
            query_dev.ptr,
            key_cache_dev.ptr,
            value_cache_dev.ptr,
            warp_out_dev.ptr,
            warp_partial_out_dev.ptr,
            warp_partial_m_dev.ptr,
            warp_partial_l_dev.ptr,
            spans,
            chunk_size,
            num_splits,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            library=library,
            runtime=runtime,
        )
        qwen35_paged_full_attn_decode_split_k_gqa_bf16_spans(
            query_dev.ptr,
            key_cache_dev.ptr,
            value_cache_dev.ptr,
            gqa_out_dev.ptr,
            gqa_partial_out_dev.ptr,
            gqa_partial_m_dev.ptr,
            gqa_partial_l_dev.ptr,
            spans,
            chunk_size,
            num_splits,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            library=library,
            runtime=runtime,
        )
        qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans(
            query_dev.ptr,
            key_cache_dev.ptr,
            value_cache_dev.ptr,
            gate_dev.ptr,
            gqa_gate_out_dev.ptr,
            gate_partial_out_dev.ptr,
            gate_partial_m_dev.ptr,
            gate_partial_l_dev.ptr,
            spans,
            chunk_size,
            num_splits,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            1,
            scale,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(warp_out), warp_out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(gqa_out), gqa_out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(gqa_gate_out), gqa_gate_out_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    warp_max_abs = float(np.max(np.abs(warp_out - expected)))
    gqa_max_abs = float(np.max(np.abs(gqa_out - expected)))
    gate_mismatch = int(np.count_nonzero(gqa_gate_out != expected_gate))
    gate_max_abs = float(
        np.max(np.abs(_bf16_bits_to_float32(gqa_gate_out) - _bf16_bits_to_float32(expected_gate)))
    )
    print(
        f"context_len={context_len} chunk_size={chunk_size} num_splits={num_splits} "
        f"shape={num_q_heads}x{head_dim}/{num_kv_heads} "
        f"warp_max_abs={warp_max_abs:.3g} gqa_max_abs={gqa_max_abs:.3g} "
        f"gqa_gate_bf16_mismatch={gate_mismatch} gqa_gate_bf16_max_abs={gate_max_abs:.3g}"
    )
    print("gqa_attn_head0=", gqa_out[0, :8].tolist())
    return 0 if warp_max_abs <= 1.0e-5 and gqa_max_abs <= 1.0e-5 and gate_mismatch == 0 else 1

def qwen35_paged_attn_gate_bf16_hip_smoke(
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.core.tensor import Tensor
    from hipengine.kernels.hip_gfx1100.attention import (
        build_qwen35_paged_attn_decode,
        qwen35_paged_full_attn_decode_split_k_gate_bf16_spans,
    )
    from hipengine.kvcache import KVLiveSpans

    block_size = 256
    blocks = 1
    context_len = 4
    chunk_size = 2
    num_splits = 2
    num_q_heads = 2
    num_kv_heads = 1
    head_dim = 8
    scale = 0.3535533905932738
    block_table = np.asarray([0], dtype=np.int32)
    live_counts = np.asarray([context_len], dtype=np.int64)
    query = np.asarray(
        [
            [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0],
            [-0.125, 0.375, -0.625, 0.875, -1.125, 1.375, -1.625, 1.875],
        ],
        dtype=np.float32,
    )
    gate_f32 = np.asarray(
        [
            [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
            [1.0, 0.5, 0.0, -0.5, -1.0, 1.5, -1.5, 0.25],
        ],
        dtype=np.float32,
    )
    gate = _float32_to_bf16_bits(gate_f32)
    token_grid = np.arange(context_len * num_kv_heads * head_dim, dtype=np.float32).reshape(
        context_len, num_kv_heads, head_dim
    )
    key_tokens = (token_grid - 11.0) / 16.0
    value_tokens = (13.0 - token_grid) / 12.0
    key_cache = np.zeros((blocks, block_size, num_kv_heads, head_dim), dtype=np.uint16)
    value_cache = np.zeros_like(key_cache)
    key_cache[0, :context_len] = _float32_to_bf16_bits(key_tokens)
    value_cache[0, :context_len] = _float32_to_bf16_bits(value_tokens)
    out = np.empty((num_q_heads, head_dim), dtype=np.uint16)
    partial_out = np.zeros((num_q_heads, num_splits, head_dim), dtype=np.float32)
    partial_m = np.zeros((num_q_heads, num_splits), dtype=np.float32)
    partial_l = np.zeros((num_q_heads, num_splits), dtype=np.float32)

    key_ref = _bf16_bits_to_float32(key_cache[0, :context_len, 0])
    value_ref = _bf16_bits_to_float32(value_cache[0, :context_len, 0])
    gate_ref = _bf16_bits_to_float32(gate)
    attn = np.empty((num_q_heads, head_dim), dtype=np.float32)
    for q_head in range(num_q_heads):
        scores = np.asarray([np.dot(query[q_head], key_ref[token]) * scale for token in range(context_len)], dtype=np.float32)
        scores = scores - np.max(scores)
        probs = np.exp(scores, dtype=np.float32)
        probs = probs / np.sum(probs, dtype=np.float32)
        attn[q_head] = probs @ value_ref
    expected = _float32_to_bf16_bits(attn * (1.0 / (1.0 + np.exp(-gate_ref, dtype=np.float32))))

    runtime = get_hip_runtime()
    library = build_qwen35_paged_attn_decode(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        block_table_dev = dev(block_table)
        live_counts_dev = dev(live_counts)
        query_dev = dev(query)
        gate_dev = dev(gate)
        key_cache_dev = dev(key_cache)
        value_cache_dev = dev(value_cache)
        out_dev_buf = out_dev(out)
        partial_out_dev = out_dev(partial_out)
        partial_m_dev = out_dev(partial_m)
        partial_l_dev = out_dev(partial_l)
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(block_table_dev.ptr, block_table.shape, "int32", Device("hip", 0)),
            live_counts=Tensor.from_handle(live_counts_dev.ptr, live_counts.shape, "int64", Device("hip", 0)),
            max_live_count=int(context_len),
            storage_dtype="bf16",
        )
        qwen35_paged_full_attn_decode_split_k_gate_bf16_spans(
            query_dev.ptr,
            key_cache_dev.ptr,
            value_cache_dev.ptr,
            gate_dev.ptr,
            out_dev_buf.ptr,
            partial_out_dev.ptr,
            partial_m_dev.ptr,
            partial_l_dev.ptr,
            spans,
            chunk_size,
            num_splits,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            1,
            scale,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev_buf, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    mismatch = int(np.count_nonzero(out != expected))
    max_abs = float(np.max(np.abs(_bf16_bits_to_float32(out) - _bf16_bits_to_float32(expected))))
    print(
        f"context_len={context_len} chunk_size={chunk_size} num_splits={num_splits} "
        f"head_dim={head_dim} bf16_mismatch={mismatch} bf16_max_abs={max_abs:.3g}"
    )
    print("gated_attn_bf16=", _bf16_bits_to_float32(out).reshape(-1).tolist())
    return 0 if mismatch == 0 else 1

def qwen35_paged_attn_gate_hip_smoke(
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.core.tensor import Tensor
    from hipengine.kernels.hip_gfx1100.attention import (
        build_qwen35_paged_attn_decode,
        qwen35_paged_full_attn_decode_split_k_gate_f32_spans,
    )
    from hipengine.kvcache import KVLiveSpans

    block_size = 256
    blocks = 1
    context_len = 4
    chunk_size = 2
    num_splits = 2
    num_q_heads = 2
    num_kv_heads = 1
    head_dim = 8
    scale = 0.3535533905932738
    block_table = np.asarray([0], dtype=np.int32)
    live_counts = np.asarray([context_len], dtype=np.int64)
    query = np.asarray(
        [
            [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0],
            [-0.125, 0.375, -0.625, 0.875, -1.125, 1.375, -1.625, 1.875],
        ],
        dtype=np.float32,
    )
    gate = np.asarray(
        [
            [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
            [1.0, 0.5, 0.0, -0.5, -1.0, 1.5, -1.5, 0.25],
        ],
        dtype=np.float32,
    )
    token_grid = np.arange(context_len * num_kv_heads * head_dim, dtype=np.float32).reshape(
        context_len, num_kv_heads, head_dim
    )
    key_tokens = (token_grid - 11.0) / 16.0
    value_tokens = (13.0 - token_grid) / 12.0
    key_cache = np.zeros((blocks, block_size, num_kv_heads, head_dim), dtype=np.uint16)
    value_cache = np.zeros_like(key_cache)
    key_cache[0, :context_len] = _float32_to_bf16_bits(key_tokens)
    value_cache[0, :context_len] = _float32_to_bf16_bits(value_tokens)
    out = np.empty((num_q_heads, head_dim), dtype=np.float32)
    partial_out = np.zeros((num_q_heads, num_splits, head_dim), dtype=np.float32)
    partial_m = np.zeros((num_q_heads, num_splits), dtype=np.float32)
    partial_l = np.zeros((num_q_heads, num_splits), dtype=np.float32)

    key_ref = _bf16_bits_to_float32(key_cache[0, :context_len, 0])
    value_ref = _bf16_bits_to_float32(value_cache[0, :context_len, 0])
    attn = np.empty_like(out)
    for q_head in range(num_q_heads):
        scores = np.asarray([np.dot(query[q_head], key_ref[token]) * scale for token in range(context_len)], dtype=np.float32)
        scores = scores - np.max(scores)
        probs = np.exp(scores, dtype=np.float32)
        probs = probs / np.sum(probs, dtype=np.float32)
        attn[q_head] = probs @ value_ref
    expected = attn * (1.0 / (1.0 + np.exp(-gate, dtype=np.float32)))

    runtime = get_hip_runtime()
    library = build_qwen35_paged_attn_decode(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        block_table_dev = dev(block_table)
        live_counts_dev = dev(live_counts)
        query_dev = dev(query)
        gate_dev = dev(gate)
        key_cache_dev = dev(key_cache)
        value_cache_dev = dev(value_cache)
        out_dev_buf = out_dev(out)
        partial_out_dev = out_dev(partial_out)
        partial_m_dev = out_dev(partial_m)
        partial_l_dev = out_dev(partial_l)
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(block_table_dev.ptr, block_table.shape, "int32", Device("hip", 0)),
            live_counts=Tensor.from_handle(live_counts_dev.ptr, live_counts.shape, "int64", Device("hip", 0)),
            max_live_count=int(context_len),
            storage_dtype="bf16",
        )
        qwen35_paged_full_attn_decode_split_k_gate_f32_spans(
            query_dev.ptr,
            key_cache_dev.ptr,
            value_cache_dev.ptr,
            gate_dev.ptr,
            out_dev_buf.ptr,
            partial_out_dev.ptr,
            partial_m_dev.ptr,
            partial_l_dev.ptr,
            spans,
            chunk_size,
            num_splits,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            1,
            scale,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev_buf, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    max_abs = float(np.max(np.abs(out - expected)))
    print(
        f"context_len={context_len} chunk_size={chunk_size} num_splits={num_splits} "
        f"head_dim={head_dim} gated_max_abs={max_abs:.3g}"
    )
    print("gated_attn_out=", out.reshape(-1).tolist())
    return 0 if max_abs <= 1.0e-6 else 1

def qwen35_paged_attn_split_k_hip_smoke(
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.core.tensor import Tensor
    from hipengine.kernels.hip_gfx1100.attention import (
        build_qwen35_paged_attn_decode,
        qwen35_paged_full_attn_decode_split_k_bf16_spans,
    )
    from hipengine.kvcache import KVLiveSpans

    block_size = 256
    blocks = 1
    context_len = 4
    chunk_size = 2
    num_splits = 2
    num_q_heads = 2
    num_kv_heads = 1
    head_dim = 8
    scale = 0.3535533905932738
    block_table = np.asarray([0], dtype=np.int32)
    live_counts = np.asarray([context_len], dtype=np.int64)
    query = np.asarray(
        [
            [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0],
            [-0.125, 0.375, -0.625, 0.875, -1.125, 1.375, -1.625, 1.875],
        ],
        dtype=np.float32,
    )
    token_grid = np.arange(context_len * num_kv_heads * head_dim, dtype=np.float32).reshape(
        context_len, num_kv_heads, head_dim
    )
    key_tokens = (token_grid - 11.0) / 16.0
    value_tokens = (13.0 - token_grid) / 12.0
    key_cache = np.zeros((blocks, block_size, num_kv_heads, head_dim), dtype=np.uint16)
    value_cache = np.zeros_like(key_cache)
    key_cache[0, :context_len] = _float32_to_bf16_bits(key_tokens)
    value_cache[0, :context_len] = _float32_to_bf16_bits(value_tokens)
    out = np.empty((num_q_heads, head_dim), dtype=np.float32)
    partial_out = np.zeros((num_q_heads, num_splits, head_dim), dtype=np.float32)
    partial_m = np.zeros((num_q_heads, num_splits), dtype=np.float32)
    partial_l = np.zeros((num_q_heads, num_splits), dtype=np.float32)

    key_ref = _bf16_bits_to_float32(key_cache[0, :context_len, 0])
    value_ref = _bf16_bits_to_float32(value_cache[0, :context_len, 0])
    expected = np.empty_like(out)
    for q_head in range(num_q_heads):
        scores = np.asarray([np.dot(query[q_head], key_ref[token]) * scale for token in range(context_len)], dtype=np.float32)
        scores = scores - np.max(scores)
        probs = np.exp(scores, dtype=np.float32)
        probs = probs / np.sum(probs, dtype=np.float32)
        expected[q_head] = probs @ value_ref

    runtime = get_hip_runtime()
    library = build_qwen35_paged_attn_decode(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        block_table_dev = dev(block_table)
        live_counts_dev = dev(live_counts)
        query_dev = dev(query)
        key_cache_dev = dev(key_cache)
        value_cache_dev = dev(value_cache)
        out_dev_buf = out_dev(out)
        partial_out_dev = out_dev(partial_out)
        partial_m_dev = out_dev(partial_m)
        partial_l_dev = out_dev(partial_l)
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(block_table_dev.ptr, block_table.shape, "int32", Device("hip", 0)),
            live_counts=Tensor.from_handle(live_counts_dev.ptr, live_counts.shape, "int64", Device("hip", 0)),
            max_live_count=int(context_len),
            storage_dtype="bf16",
        )
        qwen35_paged_full_attn_decode_split_k_bf16_spans(
            query_dev.ptr,
            key_cache_dev.ptr,
            value_cache_dev.ptr,
            out_dev_buf.ptr,
            partial_out_dev.ptr,
            partial_m_dev.ptr,
            partial_l_dev.ptr,
            spans,
            chunk_size,
            num_splits,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev_buf, runtime=runtime)
        copy_device_to_host(host_array_ptr(partial_m), partial_m_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(partial_l), partial_l_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    max_abs = float(np.max(np.abs(out - expected)))
    finite_partials = bool(np.all(np.isfinite(partial_m)) and np.all(partial_l > 0.0))
    print(
        f"context_len={context_len} chunk_size={chunk_size} num_splits={num_splits} "
        f"head_dim={head_dim} max_abs={max_abs:.3g} finite_partials={finite_partials}"
    )
    print("split_attn_out=", out.reshape(-1).tolist())
    return 0 if max_abs <= 1.0e-6 and finite_partials else 1

def qwen35_full_attn_decode_hip_smoke(
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention import (
        build_qwen35_paged_attn_decode,
        qwen35_full_attn_decode_context_bf16,
    )

    context_len = 3
    max_context_len = 4
    num_q_heads = 2
    num_kv_heads = 1
    head_dim = 4
    scale = 0.5
    live_counts = np.asarray([context_len], dtype=np.int64)
    query = np.asarray(
        [[0.25, -0.5, 0.75, -1.0], [1.25, -1.5, 1.75, -2.0]], dtype=np.float32
    )
    key_tokens = np.asarray(
        [[[0.5, -0.25, 1.0, -0.75]], [[-1.0, 0.5, -0.5, 0.25]], [[0.75, 0.25, -1.25, 1.5]]],
        dtype=np.float32,
    )
    value_tokens = np.asarray(
        [[[0.125, -0.375, 0.625, -0.875]], [[-1.125, 1.375, -1.625, 1.875]], [[0.5, -0.25, 0.75, -1.0]]],
        dtype=np.float32,
    )
    key_cache = np.zeros((max_context_len, num_kv_heads, head_dim), dtype=np.uint16)
    value_cache = np.zeros_like(key_cache)
    key_cache[:context_len] = _float32_to_bf16_bits(key_tokens)
    value_cache[:context_len] = _float32_to_bf16_bits(value_tokens)
    out = np.empty((num_q_heads, head_dim), dtype=np.float32)

    key_ref = _bf16_bits_to_float32(key_cache[:context_len, 0])
    value_ref = _bf16_bits_to_float32(value_cache[:context_len, 0])
    expected = np.empty_like(out)
    for q_head in range(num_q_heads):
        scores = np.asarray([np.dot(query[q_head], key_ref[token]) * scale for token in range(context_len)], dtype=np.float32)
        scores = scores - np.max(scores)
        probs = np.exp(scores, dtype=np.float32)
        probs = probs / np.sum(probs, dtype=np.float32)
        expected[q_head] = probs @ value_ref

    runtime = get_hip_runtime()
    library = build_qwen35_paged_attn_decode(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        live_counts_dev = dev(live_counts)
        query_dev = dev(query)
        key_cache_dev = dev(key_cache)
        value_cache_dev = dev(value_cache)
        out_dev_buf = out_dev(out)
        qwen35_full_attn_decode_context_bf16(
            query_dev.ptr,
            key_cache_dev.ptr,
            value_cache_dev.ptr,
            out_dev_buf.ptr,
            live_counts_dev.ptr,
            max_context_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev_buf, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    max_abs = float(np.max(np.abs(out - expected)))
    print(
        f"context_len={context_len} max_context_len={max_context_len} num_q_heads={num_q_heads} "
        f"num_kv_heads={num_kv_heads} head_dim={head_dim} max_abs={max_abs:.3g}"
    )
    print("full_attn_out=", out.reshape(-1).tolist())
    return 0 if max_abs <= 1.0e-6 else 1


def qwen35_paged_attn_decode_hip_smoke(
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.core.tensor import Tensor
    from hipengine.kernels.hip_gfx1100.attention import (
        build_qwen35_paged_attn_decode,
        qwen35_paged_full_attn_decode_context_bf16_spans,
    )
    from hipengine.kvcache import KVLiveSpans

    block_size = 256
    blocks = 1
    context_len = 2
    max_context_len = 2
    num_q_heads = 2
    num_kv_heads = 1
    head_dim = 4
    scale = 0.5
    block_table = np.asarray([0], dtype=np.int32)
    live_counts = np.asarray([context_len], dtype=np.int64)
    query = np.asarray(
        [[0.25, -0.5, 0.75, -1.0], [1.25, -1.5, 1.75, -2.0]], dtype=np.float32
    )
    key_tokens = np.asarray(
        [[[0.5, -0.25, 1.0, -0.75]], [[-1.0, 0.5, -0.5, 0.25]]], dtype=np.float32
    )
    value_tokens = np.asarray(
        [[[0.125, -0.375, 0.625, -0.875]], [[-1.125, 1.375, -1.625, 1.875]]], dtype=np.float32
    )
    key_cache = np.zeros((blocks, block_size, num_kv_heads, head_dim), dtype=np.uint16)
    value_cache = np.zeros_like(key_cache)
    key_cache[0, :context_len] = _float32_to_bf16_bits(key_tokens)
    value_cache[0, :context_len] = _float32_to_bf16_bits(value_tokens)
    out = np.empty((num_q_heads, head_dim), dtype=np.float32)

    key_ref = _bf16_bits_to_float32(key_cache[0, :context_len, 0])
    value_ref = _bf16_bits_to_float32(value_cache[0, :context_len, 0])
    expected = np.empty_like(out)
    for q_head in range(num_q_heads):
        scores = np.asarray([np.dot(query[q_head], key_ref[token]) * scale for token in range(context_len)], dtype=np.float32)
        scores = scores - np.max(scores)
        probs = np.exp(scores, dtype=np.float32)
        probs = probs / np.sum(probs, dtype=np.float32)
        expected[q_head] = probs @ value_ref

    runtime = get_hip_runtime()
    library = build_qwen35_paged_attn_decode(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        block_table_dev = dev(block_table)
        live_counts_dev = dev(live_counts)
        query_dev = dev(query)
        key_cache_dev = dev(key_cache)
        value_cache_dev = dev(value_cache)
        out_dev_buf = out_dev(out)
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(block_table_dev.ptr, block_table.shape, "int32", Device("hip", 0)),
            live_counts=Tensor.from_handle(live_counts_dev.ptr, live_counts.shape, "int64", Device("hip", 0)),
            max_live_count=int(context_len),
            storage_dtype="bf16",
        )
        qwen35_paged_full_attn_decode_context_bf16_spans(
            query_dev.ptr,
            key_cache_dev.ptr,
            value_cache_dev.ptr,
            out_dev_buf.ptr,
            spans,
            max_context_len,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev_buf, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    max_abs = float(np.max(np.abs(out - expected)))
    print(
        f"context_len={context_len} block_size={block_size} num_q_heads={num_q_heads} "
        f"num_kv_heads={num_kv_heads} head_dim={head_dim} max_abs={max_abs:.3g}"
    )
    print("attn_out=", out.reshape(-1).tolist())
    return 0 if max_abs <= 1.0e-6 else 1

def qwen35_paged_kv_write_hip_smoke(
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.core.tensor import Tensor
    from hipengine.kernels.hip_gfx1100.attention import (
        build_qwen35_paged_kv_write,
        qwen35_write_paged_kv_f32_spans,
        qwen35_write_paged_kv_mixed_value_bf16_spans,
    )
    from hipengine.kvcache import KVLiveSpans

    block_size = 4
    blocks = 2
    num_kv_heads = 2
    head_dim = 4
    position = 5
    logical_block = position // block_size
    block_offset = position - logical_block * block_size
    physical_block = 0
    block_table = np.asarray([1, physical_block], dtype=np.int32)
    live_counts = np.asarray([position], dtype=np.int64)
    key = np.asarray(
        [[0.25, -0.5, 0.75, -1.0], [1.25, -1.5, 1.75, -2.0]], dtype=np.float32
    )
    value_f32 = np.asarray(
        [[-0.125, 0.375, -0.625, 0.875], [1.125, -1.375, 1.625, -1.875]], dtype=np.float32
    )
    value_bf16_bits = _float32_to_bf16_bits(value_f32)
    mixed_key_cache = np.zeros((blocks, block_size, num_kv_heads, head_dim), dtype=np.uint16)
    mixed_value_cache = np.zeros_like(mixed_key_cache)
    f32_key_cache = np.zeros_like(mixed_key_cache)
    f32_value_cache = np.zeros_like(mixed_key_cache)
    expected_key_bits = _float32_to_bf16_bits(key)
    expected_value_bf16_bits = value_bf16_bits
    expected_value_f32_bits = _float32_to_bf16_bits(value_f32)

    runtime = get_hip_runtime()
    library = build_qwen35_paged_kv_write(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    try:
        block_table_dev = dev(block_table)
        live_counts_dev = dev(live_counts)
        key_dev = dev(key)
        value_f32_dev = dev(value_f32)
        value_bf16_dev = dev(value_bf16_bits)
        mixed_key_cache_dev = out_dev(mixed_key_cache)
        mixed_value_cache_dev = out_dev(mixed_value_cache)
        f32_key_cache_dev = out_dev(f32_key_cache)
        f32_value_cache_dev = out_dev(f32_value_cache)
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(
                block_table_dev.ptr, block_table.shape, "int32", Device("hip", 0)
            ),
            live_counts=Tensor.from_handle(
                live_counts_dev.ptr, live_counts.shape, "int64", Device("hip", 0)
            ),
            max_live_count=int(position),
            storage_dtype="bf16",
        )
        qwen35_write_paged_kv_mixed_value_bf16_spans(
            key_dev.ptr,
            value_bf16_dev.ptr,
            mixed_key_cache_dev.ptr,
            mixed_value_cache_dev.ptr,
            spans,
            block_size,
            num_kv_heads,
            head_dim,
            library=library,
            runtime=runtime,
        )
        qwen35_write_paged_kv_f32_spans(
            key_dev.ptr,
            value_f32_dev.ptr,
            f32_key_cache_dev.ptr,
            f32_value_cache_dev.ptr,
            spans,
            block_size,
            num_kv_heads,
            head_dim,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(mixed_key_cache), mixed_key_cache_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(mixed_value_cache), mixed_value_cache_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(f32_key_cache), f32_key_cache_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(f32_value_cache), f32_value_cache_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    target = (physical_block, block_offset)
    mixed_key_mismatch = int(np.count_nonzero(mixed_key_cache[target] != expected_key_bits))
    mixed_value_mismatch = int(np.count_nonzero(mixed_value_cache[target] != expected_value_bf16_bits))
    f32_key_mismatch = int(np.count_nonzero(f32_key_cache[target] != expected_key_bits))
    f32_value_mismatch = int(np.count_nonzero(f32_value_cache[target] != expected_value_f32_bits))
    untouched_mask = np.ones(mixed_key_cache.shape, dtype=bool)
    untouched_mask[physical_block, block_offset, :, :] = False
    untouched = int(np.count_nonzero(mixed_key_cache[untouched_mask]))
    print(
        f"block_size={block_size} position={position} physical_block={physical_block} "
        f"mixed_mismatch={mixed_key_mismatch}/{mixed_value_mismatch} "
        f"f32_mismatch={f32_key_mismatch}/{f32_value_mismatch} untouched_nonzero={untouched}"
    )
    print("kv_key=", _bf16_bits_to_float32(mixed_key_cache[target]).reshape(-1).tolist())
    return 0 if (
        mixed_key_mismatch == 0
        and mixed_value_mismatch == 0
        and f32_key_mismatch == 0
        and f32_value_mismatch == 0
    ) else 1

def qwen35_linear_attn_prefill_hip_smoke(
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.linear_attn import (
        build_qwen35_linear_attn_conv,
        build_qwen35_linear_attn_gdn,
        qwen35_gdn_prefill_recurrent_f32,
        qwen35_gdn_prefill_recurrent_k2_f32,
        qwen35_gdn_prefill_rmsnorm_gate_bf16,
        qwen35_linear_attn_conv_prefill_f32,
        qwen35_linear_attn_prefill_prepare_f32_bf16,
    )

    runtime = get_hip_runtime()
    conv_library = build_qwen35_linear_attn_conv(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    gdn_library = build_qwen35_linear_attn_gdn(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    def conv_prefill_ref(hidden: np.ndarray, state: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tokens, channels = hidden.shape
        kernel_size = state.shape[1]
        out = np.empty_like(hidden, dtype=np.float32)
        for token in range(tokens):
            for channel in range(channels):
                acc = np.float32(0.0)
                for k in range(kernel_size):
                    padded = token + k
                    if padded < kernel_size - 1:
                        value = np.float32(state[channel, padded + 1])
                    else:
                        value = np.float32(hidden[padded - (kernel_size - 1), channel])
                    acc = np.float32(acc + np.float32(value * weight[channel, k]))
                out[token, channel] = _silu_np(acc)
        new_state = state.copy()
        for channel in range(channels):
            for k in range(kernel_size):
                new_state[channel, k] = hidden[tokens - kernel_size + k, channel]
        return out, new_state

    def gdn_prefill_ref(
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        beta: np.ndarray,
        decay: np.ndarray,
        state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens, num_v_heads, head_k_dim = query.shape
        head_v_dim = value.shape[2]
        out = np.empty((tokens, num_v_heads, head_v_dim), dtype=np.float32)
        new_state = state.copy()
        for token in range(tokens):
            for v_head in range(num_v_heads):
                for value_idx in range(head_v_dim):
                    state_vec = new_state[v_head, :, value_idx].copy()
                    state_vec = np.asarray(state_vec * decay[token, v_head], dtype=np.float32)
                    kv_mem = np.sum(key[token, v_head] * state_vec, dtype=np.float32)
                    delta = np.float32(np.float32(value[token, v_head, value_idx] - kv_mem) * beta[token, v_head])
                    state_vec = np.asarray(state_vec + key[token, v_head] * delta, dtype=np.float32)
                    new_state[v_head, :, value_idx] = state_vec
                    out[token, v_head, value_idx] = np.sum(query[token, v_head] * state_vec, dtype=np.float32)
        return out, new_state

    def linear_prefill_prepare_ref(
        conv_out_src: np.ndarray,
        a_bits: np.ndarray,
        b_bits: np.ndarray,
        dt_bias_src: np.ndarray,
        a_log_src: np.ndarray,
        *,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tokens = conv_out_src.shape[0]
        repeat = num_v_heads // num_k_heads
        key_dim = num_k_heads * head_k_dim
        query_out = np.empty((tokens, num_v_heads, head_k_dim), dtype=np.float32)
        key_out = np.empty_like(query_out)
        value_out = np.empty((tokens, num_v_heads, head_v_dim), dtype=np.float32)
        beta_out = np.empty((tokens, num_v_heads), dtype=np.float32)
        decay_out = np.empty((tokens, num_v_heads), dtype=np.float32)
        a_f32 = _bf16_bits_to_float32(a_bits)
        b_f32 = _bf16_bits_to_float32(b_bits)
        for token in range(tokens):
            for v_head in range(num_v_heads):
                k_head = v_head // repeat
                q = conv_out_src[token, k_head * head_k_dim : (k_head + 1) * head_k_dim].astype(np.float32)
                k = conv_out_src[
                    token,
                    key_dim + k_head * head_k_dim : key_dim + (k_head + 1) * head_k_dim,
                ].astype(np.float32)
                q_scale = np.float32(1.0) / np.sqrt(np.sum(q * q, dtype=np.float32) + np.float32(1.0e-6))
                q_scale = np.float32(q_scale * (np.float32(1.0) / np.sqrt(np.float32(head_k_dim))))
                k_scale = np.float32(1.0) / np.sqrt(np.sum(k * k, dtype=np.float32) + np.float32(1.0e-6))
                query_out[token, v_head] = q * q_scale
                key_out[token, v_head] = k * k_scale
                value_out[token, v_head] = conv_out_src[
                    token,
                    2 * key_dim + v_head * head_v_dim : 2 * key_dim + (v_head + 1) * head_v_dim,
                ]
                beta_out[token, v_head] = np.float32(1.0) / (
                    np.float32(1.0) + np.exp(-b_f32[token, v_head], dtype=np.float32)
                )
                decay_out[token, v_head] = np.exp(
                    -np.exp(a_log_src[v_head], dtype=np.float32)
                    * np.float32(
                        a_f32[token, v_head] + dt_bias_src[v_head]
                        if a_f32[token, v_head] + dt_bias_src[v_head] > np.float32(20.0)
                        else np.log1p(np.exp(a_f32[token, v_head] + dt_bias_src[v_head], dtype=np.float32), dtype=np.float32)
                    ),
                    dtype=np.float32,
                )
        return query_out, key_out, value_out, beta_out, decay_out

    def rmsnorm_gate_ref(recurrent: np.ndarray, gate_bits: np.ndarray, norm_weight: np.ndarray, eps: float) -> np.ndarray:
        gate_f32 = _bf16_bits_to_float32(gate_bits).reshape(recurrent.shape)
        out = np.empty_like(recurrent, dtype=np.float32)
        for token in range(recurrent.shape[0]):
            for head in range(recurrent.shape[1]):
                row = recurrent[token, head]
                inv = np.float32(1.0) / np.sqrt(
                    np.sum(row * row, dtype=np.float32) / np.float32(row.shape[0]) + np.float32(eps)
                )
                out[token, head] = row * inv * norm_weight * _silu_np(gate_f32[token, head])
        return _float32_to_bf16_bits(out.reshape(-1)).reshape(recurrent.shape)

    conv_tokens = 5
    channels = 8
    kernel_size = 4
    hidden = np.asarray(
        [[((token * 7 + channel * 3) % 11 - 5) * 0.0625 for channel in range(channels)] for token in range(conv_tokens)],
        dtype=np.float32,
    )
    conv_state = np.asarray(
        [[0.125 * ((channel + k) % 5 - 2) for k in range(kernel_size)] for channel in range(channels)],
        dtype=np.float32,
    )
    conv_weight = np.asarray(
        [[0.25 * ((channel + 2 * k) % 5 - 2) for k in range(kernel_size)] for channel in range(channels)],
        dtype=np.float32,
    )
    conv_out = np.empty_like(hidden)
    expected_conv_out, expected_conv_state = conv_prefill_ref(hidden, conv_state, conv_weight)

    tokens = 3
    num_k_heads = 1
    num_v_heads = 2
    head_k_dim = 128
    head_v_dim = 4
    qkv_width = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    conv_gdn = np.asarray(
        [[((token * 29 + col * 7) % 23 - 11) * 0.01 for col in range(qkv_width)] for token in range(tokens)],
        dtype=np.float32,
    )
    a_bits = _float32_to_bf16_bits(
        np.asarray([[((token * 3 + head) % 5 - 2) * 0.125 for head in range(num_v_heads)] for token in range(tokens)], dtype=np.float32)
    )
    b_bits = _float32_to_bf16_bits(
        np.asarray([[((token * 5 + head * 2) % 7 - 3) * 0.1 for head in range(num_v_heads)] for token in range(tokens)], dtype=np.float32)
    )
    dt_bias = np.asarray([0.125, -0.25], dtype=np.float32)
    a_log = np.asarray([-1.0, -0.5], dtype=np.float32)
    query, key, value, beta, decay = linear_prefill_prepare_ref(
        conv_gdn,
        a_bits,
        b_bits,
        dt_bias,
        a_log,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
    )
    query_actual = np.empty_like(query)
    key_actual = np.empty_like(key)
    value_actual = np.empty_like(value)
    beta_actual = np.empty_like(beta)
    decay_actual = np.empty_like(decay)
    state = np.asarray(
        [
            [[((head * 23 + k * 5 + d * 3) % 19 - 9) * 0.01 for d in range(head_v_dim)] for k in range(head_k_dim)]
            for head in range(num_v_heads)
        ],
        dtype=np.float32,
    )
    gate_bits = _float32_to_bf16_bits(
        np.asarray(
            [
                [[((token * 7 + head * 3 + d) % 9 - 4) * 0.0625 for d in range(head_v_dim)] for head in range(num_v_heads)]
                for token in range(tokens)
            ],
            dtype=np.float32,
        )
    )
    norm_weight = np.asarray([1.0, 0.5, 0.25, 2.0], dtype=np.float32)
    eps = 1.0e-6
    gdn_out = np.empty((tokens, num_v_heads, head_v_dim), dtype=np.float32)
    gdn_k2_out = np.empty_like(gdn_out)
    gated_bits = np.empty((tokens, num_v_heads, head_v_dim), dtype=np.uint16)
    expected_gdn_out, expected_gdn_state = gdn_prefill_ref(query, key, value, beta, decay, state)
    expected_gated_bits = rmsnorm_gate_ref(expected_gdn_out, gate_bits, norm_weight, eps)
    state_regular = state.copy()
    state_k2 = state.copy()

    try:
        hidden_dev = dev(hidden)
        conv_state_dev = dev(conv_state.copy())
        conv_weight_dev = dev(conv_weight)
        conv_out_dev = out_dev(conv_out)
        qwen35_linear_attn_conv_prefill_f32(
            hidden_dev.ptr,
            conv_state_dev.ptr,
            conv_weight_dev.ptr,
            conv_out_dev.ptr,
            conv_tokens,
            channels,
            kernel_size,
            library=conv_library,
            runtime=runtime,
        )
        conv_gdn_dev = dev(conv_gdn)
        a_dev = dev(a_bits)
        b_dev = dev(b_bits)
        dt_bias_dev = dev(dt_bias)
        a_log_dev = dev(a_log)
        query_dev = out_dev(query_actual)
        key_dev = out_dev(key_actual)
        value_dev = out_dev(value_actual)
        beta_dev = out_dev(beta_actual)
        decay_dev = out_dev(decay_actual)
        state_regular_dev = dev(state_regular)
        state_k2_dev = dev(state_k2)
        gdn_out_dev = out_dev(gdn_out)
        gdn_k2_out_dev = out_dev(gdn_k2_out)
        gated_bits_dev = out_dev(gated_bits)
        norm_weight_dev = dev(norm_weight)
        gate_dev = dev(gate_bits)
        qwen35_linear_attn_prefill_prepare_f32_bf16(
            conv_gdn_dev.ptr,
            a_dev.ptr,
            b_dev.ptr,
            dt_bias_dev.ptr,
            a_log_dev.ptr,
            query_dev.ptr,
            key_dev.ptr,
            value_dev.ptr,
            beta_dev.ptr,
            decay_dev.ptr,
            tokens,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=gdn_library,
            runtime=runtime,
        )
        qwen35_gdn_prefill_recurrent_f32(
            query_dev.ptr,
            key_dev.ptr,
            value_dev.ptr,
            beta_dev.ptr,
            decay_dev.ptr,
            state_regular_dev.ptr,
            gdn_out_dev.ptr,
            tokens,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=gdn_library,
            runtime=runtime,
        )
        qwen35_gdn_prefill_recurrent_k2_f32(
            query_dev.ptr,
            key_dev.ptr,
            value_dev.ptr,
            beta_dev.ptr,
            decay_dev.ptr,
            state_k2_dev.ptr,
            gdn_k2_out_dev.ptr,
            tokens,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=gdn_library,
            runtime=runtime,
        )
        qwen35_gdn_prefill_rmsnorm_gate_bf16(
            gdn_k2_out_dev.ptr,
            gate_dev.ptr,
            norm_weight_dev.ptr,
            gated_bits_dev.ptr,
            eps,
            tokens,
            num_v_heads,
            head_v_dim,
            library=gdn_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(conv_out), conv_out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(conv_state), conv_state_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(query_actual), query_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(key_actual), key_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(value_actual), value_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(beta_actual), beta_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(decay_actual), decay_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(gdn_out), gdn_out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(gdn_k2_out), gdn_k2_out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(gated_bits), gated_bits_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(state_regular), state_regular_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(state_k2), state_k2_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    conv_out_max_abs = float(np.max(np.abs(conv_out - expected_conv_out)))
    conv_state_max_abs = float(np.max(np.abs(conv_state - expected_conv_state)))
    query_max_abs = float(np.max(np.abs(query_actual - query)))
    key_max_abs = float(np.max(np.abs(key_actual - key)))
    value_max_abs = float(np.max(np.abs(value_actual - value)))
    beta_max_abs = float(np.max(np.abs(beta_actual - beta)))
    decay_max_abs = float(np.max(np.abs(decay_actual - decay)))
    gdn_out_max_abs = float(np.max(np.abs(gdn_out - expected_gdn_out)))
    gdn_state_max_abs = float(np.max(np.abs(state_regular - expected_gdn_state)))
    gdn_k2_out_max_abs = float(np.max(np.abs(gdn_k2_out - expected_gdn_out)))
    gdn_k2_state_max_abs = float(np.max(np.abs(state_k2 - expected_gdn_state)))
    gated_mismatch = int(np.count_nonzero(gated_bits != expected_gated_bits))
    print(
        f"conv_out_max_abs={conv_out_max_abs:.3g} conv_state_max_abs={conv_state_max_abs:.3g} "
        f"prepare_max_abs={max(query_max_abs, key_max_abs, value_max_abs, beta_max_abs, decay_max_abs):.3g} "
        f"gdn_out_max_abs={gdn_out_max_abs:.3g} gdn_state_max_abs={gdn_state_max_abs:.3g} "
        f"gdn_k2_out_max_abs={gdn_k2_out_max_abs:.3g} gdn_k2_state_max_abs={gdn_k2_state_max_abs:.3g} "
        f"gated_mismatch={gated_mismatch}"
    )
    return 0 if max(
        conv_out_max_abs,
        conv_state_max_abs,
        query_max_abs,
        key_max_abs,
        value_max_abs,
        beta_max_abs,
        decay_max_abs,
        gdn_out_max_abs,
        gdn_state_max_abs,
        gdn_k2_out_max_abs,
        gdn_k2_state_max_abs,
    ) <= 1.0e-5 and gated_mismatch == 0 else 1


def qwen35_linear_attn_gdn_hip_smoke(
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.linear_attn import (
        build_qwen35_linear_attn_gdn,
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
    )

    num_k_heads = 1
    num_v_heads = 2
    head_k_dim = 8
    head_v_dim = 4
    eps = 1.0e-6
    key_dim = num_k_heads * head_k_dim
    conv_dim = 2 * key_dim + num_v_heads * head_v_dim
    conv_out = np.asarray([((idx % 7) - 3) * 0.125 for idx in range(conv_dim)], dtype=np.float32)
    gate = _float32_to_bf16_bits(
        np.asarray([((idx % 5) - 2) * 0.25 for idx in range(num_v_heads * head_v_dim)], dtype=np.float32)
    )
    a = _float32_to_bf16_bits(np.asarray([-0.25, 0.5], dtype=np.float32))
    b = _float32_to_bf16_bits(np.asarray([0.25, -0.5], dtype=np.float32))
    dt_bias = np.asarray([0.125, -0.25], dtype=np.float32)
    a_log = np.asarray([-1.0, -0.5], dtype=np.float32)
    norm_weight = np.asarray([1.0, 0.5, 0.25, 2.0], dtype=np.float32)
    recurrent_state = np.asarray(
        [
            ((v * 17 + k * 5 + d * 3) % 11 - 5) * 0.03125
            for v in range(num_v_heads)
            for k in range(head_k_dim)
            for d in range(head_v_dim)
        ],
        dtype=np.float32,
    ).reshape(num_v_heads, head_k_dim, head_v_dim)
    out = np.empty((num_v_heads, head_v_dim), dtype=np.float32)

    def softplus(x: np.float32) -> np.float32:
        return np.float32(x if x > np.float32(20.0) else np.log1p(np.exp(x, dtype=np.float32), dtype=np.float32))

    def sigmoid(x: np.float32) -> np.float32:
        return np.float32(1.0) / (np.float32(1.0) + np.exp(-x, dtype=np.float32))

    expected_state = recurrent_state.copy()
    expected_acc = np.empty_like(out)
    gate_f32 = _bf16_bits_to_float32(gate).reshape(num_v_heads, head_v_dim)
    a_f32 = _bf16_bits_to_float32(a)
    b_f32 = _bf16_bits_to_float32(b)
    for v_head in range(num_v_heads):
        k_head = v_head // (num_v_heads // num_k_heads)
        q_base = k_head * head_k_dim
        k_base = key_dim + q_base
        q = conv_out[q_base : q_base + head_k_dim].astype(np.float32)
        k = conv_out[k_base : k_base + head_k_dim].astype(np.float32)
        q_sum = np.sum(q * q, dtype=np.float32)
        k_sum = np.sum(k * k, dtype=np.float32)
        q_scale = np.float32(1.0) / np.sqrt(q_sum + np.float32(1.0e-6), dtype=np.float32)
        q_scale = np.float32(q_scale * (np.float32(1.0) / np.sqrt(np.float32(head_k_dim), dtype=np.float32)))
        k_scale = np.float32(1.0) / np.sqrt(k_sum + np.float32(1.0e-6), dtype=np.float32)
        beta = sigmoid(b_f32[v_head])
        decay = np.exp(
            -np.exp(a_log[v_head], dtype=np.float32)
            * softplus(np.float32(a_f32[v_head] + dt_bias[v_head])),
            dtype=np.float32,
        )
        values = conv_out[2 * key_dim + v_head * head_v_dim : 2 * key_dim + (v_head + 1) * head_v_dim]
        for value_idx in range(head_v_dim):
            state_vec = expected_state[v_head, :, value_idx].copy()
            kv_mem = np.float32(0.0)
            for idx in range(head_k_dim):
                kv_mem = np.float32(
                    kv_mem + np.float32(np.float32(k[idx] * k_scale) * np.float32(state_vec[idx] * decay))
                )
            delta = np.float32(np.float32(values[value_idx] - kv_mem) * beta)
            out_acc = np.float32(0.0)
            for idx in range(head_k_dim):
                k_norm = np.float32(k[idx] * k_scale)
                q_norm = np.float32(q[idx] * q_scale)
                new_state = np.float32(np.float32(state_vec[idx] * decay) + np.float32(k_norm * delta))
                expected_state[v_head, idx, value_idx] = new_state
                out_acc = np.float32(out_acc + np.float32(q_norm * new_state))
            expected_acc[v_head, value_idx] = out_acc
    expected_out = np.empty_like(out)
    for v_head in range(num_v_heads):
        inv_rms = np.float32(1.0) / np.sqrt(
            np.sum(expected_acc[v_head] * expected_acc[v_head], dtype=np.float32) / np.float32(head_v_dim)
            + np.float32(eps),
            dtype=np.float32,
        )
        for value_idx in range(head_v_dim):
            expected_out[v_head, value_idx] = np.float32(
                expected_acc[v_head, value_idx]
                * inv_rms
                * norm_weight[value_idx]
                * _silu_np(gate_f32[v_head, value_idx])
            )

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_gdn(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        conv_out_dev = dev(conv_out)
        gate_dev = dev(gate)
        a_dev = dev(a)
        b_dev = dev(b)
        dt_bias_dev = dev(dt_bias)
        a_log_dev = dev(a_log)
        norm_weight_dev = dev(norm_weight)
        recurrent_state_dev = dev(recurrent_state)
        out_dev_buf = out_dev(out)
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
            conv_out_dev.ptr,
            gate_dev.ptr,
            a_dev.ptr,
            b_dev.ptr,
            dt_bias_dev.ptr,
            a_log_dev.ptr,
            norm_weight_dev.ptr,
            recurrent_state_dev.ptr,
            out_dev_buf.ptr,
            eps,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev_buf, runtime=runtime)
        copy_device_to_host(host_array_ptr(recurrent_state), recurrent_state_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    out_max_abs = float(np.max(np.abs(out - expected_out)))
    state_max_abs = float(np.max(np.abs(recurrent_state - expected_state)))
    print(
        f"num_k_heads={num_k_heads} num_v_heads={num_v_heads} head_k_dim={head_k_dim} "
        f"head_v_dim={head_v_dim} out_max_abs={out_max_abs:.3g} state_max_abs={state_max_abs:.3g}"
    )
    print("gdn_out=", out.reshape(-1).tolist())
    return 0 if max(out_max_abs, state_max_abs) <= 1.0e-6 else 1

def qwen35_linear_attn_conv_hip_smoke(
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.linear_attn import (
        build_qwen35_linear_attn_conv,
        qwen35_linear_attn_conv_decode_bf16,
        qwen35_linear_attn_conv_decode_f32,
    )

    channels = 8
    kernel_size = 4
    hidden_f32 = np.asarray([-0.5, -0.25, 0.25, 0.5, -0.5, -0.25, 0.25, 0.5], dtype=np.float32)
    hidden_bits = _float32_to_bf16_bits(hidden_f32)
    conv_state_base = np.asarray(
        [[0.125 * ((channel + k) % 5 - 2) for k in range(kernel_size)] for channel in range(channels)],
        dtype=np.float32,
    )
    conv_weight = np.asarray(
        [[0.25 * ((channel + 2 * k) % 5 - 2) for k in range(kernel_size)] for channel in range(channels)],
        dtype=np.float32,
    )
    state_f32 = conv_state_base.copy()
    state_bf16 = conv_state_base.copy()
    out_f32 = np.empty(channels, dtype=np.float32)
    out_bf16 = np.empty(channels, dtype=np.float32)

    def conv_ref(hidden: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        new_state = state.copy()
        out = np.empty(channels, dtype=np.float32)
        for channel in range(channels):
            acc = np.float32(0.0)
            for idx in range(kernel_size - 1):
                value = np.float32(new_state[channel, idx + 1])
                acc = np.float32(acc + np.float32(value * conv_weight[channel, idx]))
                new_state[channel, idx] = value
            newest = np.float32(hidden[channel])
            acc = np.float32(acc + np.float32(newest * conv_weight[channel, kernel_size - 1]))
            new_state[channel, kernel_size - 1] = newest
            out[channel] = _silu_np(acc)
        return out, new_state

    expected_f32, expected_state_f32 = conv_ref(hidden_f32, conv_state_base)
    expected_bf16, expected_state_bf16 = conv_ref(_bf16_bits_to_float32(hidden_bits), conv_state_base)

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_conv(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        hidden_f32_dev = dev(hidden_f32)
        hidden_bf16_dev = dev(hidden_bits)
        state_f32_dev = dev(state_f32)
        state_bf16_dev = dev(state_bf16)
        weight_dev = dev(conv_weight)
        out_f32_dev = out_dev(out_f32)
        out_bf16_dev = out_dev(out_bf16)
        qwen35_linear_attn_conv_decode_f32(
            hidden_f32_dev.ptr,
            state_f32_dev.ptr,
            weight_dev.ptr,
            out_f32_dev.ptr,
            channels,
            kernel_size,
            library=library,
            runtime=runtime,
        )
        qwen35_linear_attn_conv_decode_bf16(
            hidden_bf16_dev.ptr,
            state_bf16_dev.ptr,
            weight_dev.ptr,
            out_bf16_dev.ptr,
            channels,
            kernel_size,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_f32), out_f32_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(out_bf16), out_bf16_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(state_f32), state_f32_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(state_bf16), state_bf16_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    f32_out_max_abs = float(np.max(np.abs(out_f32 - expected_f32)))
    f32_state_max_abs = float(np.max(np.abs(state_f32 - expected_state_f32)))
    bf16_out_max_abs = float(np.max(np.abs(out_bf16 - expected_bf16)))
    bf16_state_max_abs = float(np.max(np.abs(state_bf16 - expected_state_bf16)))
    print(
        f"channels={channels} kernel_size={kernel_size} "
        f"f32_out_max_abs={f32_out_max_abs:.3g} f32_state_max_abs={f32_state_max_abs:.3g} "
        f"bf16_out_max_abs={bf16_out_max_abs:.3g} bf16_state_max_abs={bf16_state_max_abs:.3g}"
    )
    print("conv_f32=", out_f32.tolist())
    return 0 if max(f32_out_max_abs, f32_state_max_abs, bf16_out_max_abs, bf16_state_max_abs) <= 1.0e-6 else 1

def qwen35_rotary_hip_smoke(
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.rotary import (
        build_qwen35_rotary,
        qwen35_head_rmsnorm_partial_rotary_f32_bf16,
        qwen35_head_rmsnorm_partial_rotary_position_f32_bf16,
        qwen35_partial_rotary_f32,
    )

    num_q_heads = 2
    num_kv_heads = 1
    head_dim = 8
    rotary_dim = 4
    eps = 1.0e-6
    query = np.asarray(
        [[0.25, -0.5, 0.75, -1.0, 0.5, -0.25, 1.25, -0.75],
         [-0.125, 0.375, -0.625, 0.875, -1.125, 1.375, -1.625, 1.875]],
        dtype=np.float32,
    )
    key = np.asarray([[0.5, -0.25, 1.0, -0.75, 1.5, -1.25, 0.125, -0.375]], dtype=np.float32)
    cos = np.ones(rotary_dim, dtype=np.float32)
    sin = np.zeros(rotary_dim, dtype=np.float32)
    cos_table = np.vstack([np.zeros(rotary_dim, dtype=np.float32), cos]).astype(np.float32)
    sin_table = np.vstack([np.ones(rotary_dim, dtype=np.float32), sin]).astype(np.float32)
    position = np.asarray([1], dtype=np.int64)
    q_weight = _float32_to_bf16_bits(
        np.asarray([0.0, 0.125, -0.125, 0.25, 0.0, -0.25, 0.125, -0.125], dtype=np.float32)
    )
    k_weight = _float32_to_bf16_bits(
        np.asarray([0.125, 0.0, -0.125, 0.25, -0.25, 0.125, 0.0, -0.125], dtype=np.float32)
    )
    partial_query = np.empty_like(query)
    partial_key = np.empty_like(key)
    head_query = np.empty_like(query)
    head_key = np.empty_like(key)
    position_query = np.empty_like(query)
    position_key = np.empty_like(key)

    def head_ref(src: np.ndarray, weight_bits: np.ndarray) -> np.ndarray:
        weight = _bf16_bits_to_float32(weight_bits).reshape(1, head_dim)
        out = np.empty_like(src, dtype=np.float32)
        for head in range(src.shape[0]):
            row = src[head].astype(np.float32)
            inv = np.float32(1.0) / np.sqrt(
                np.sum(row * row, dtype=np.float32) / np.float32(head_dim) + np.float32(eps)
            )
            out[head] = row * inv * (np.float32(1.0) + weight[0])
        return out

    expected_partial_query = query.copy()
    expected_partial_key = key.copy()
    expected_head_query = head_ref(query, q_weight)
    expected_head_key = head_ref(key, k_weight)

    runtime = get_hip_runtime()
    library = build_qwen35_rotary(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        query_dev = dev(query)
        key_dev = dev(key)
        cos_dev = dev(cos)
        sin_dev = dev(sin)
        cos_table_dev = dev(cos_table)
        sin_table_dev = dev(sin_table)
        position_dev = dev(position)
        q_weight_dev = dev(q_weight)
        k_weight_dev = dev(k_weight)
        partial_query_dev = out_dev(partial_query)
        partial_key_dev = out_dev(partial_key)
        head_query_dev = out_dev(head_query)
        head_key_dev = out_dev(head_key)
        position_query_dev = out_dev(position_query)
        position_key_dev = out_dev(position_key)
        qwen35_partial_rotary_f32(
            query_dev.ptr,
            key_dev.ptr,
            cos_dev.ptr,
            sin_dev.ptr,
            partial_query_dev.ptr,
            partial_key_dev.ptr,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            library=library,
            runtime=runtime,
        )
        qwen35_head_rmsnorm_partial_rotary_f32_bf16(
            query_dev.ptr,
            key_dev.ptr,
            q_weight_dev.ptr,
            k_weight_dev.ptr,
            cos_dev.ptr,
            sin_dev.ptr,
            head_query_dev.ptr,
            head_key_dev.ptr,
            eps,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            library=library,
            runtime=runtime,
        )
        qwen35_head_rmsnorm_partial_rotary_position_f32_bf16(
            query_dev.ptr,
            key_dev.ptr,
            q_weight_dev.ptr,
            k_weight_dev.ptr,
            cos_table_dev.ptr,
            sin_table_dev.ptr,
            position_dev.ptr,
            position_query_dev.ptr,
            position_key_dev.ptr,
            eps,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            cos_table.shape[0],
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(partial_query), partial_query_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(partial_key), partial_key_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(head_query), head_query_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(head_key), head_key_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(position_query), position_query_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(position_key), position_key_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    partial_max_abs = float(
        max(np.max(np.abs(partial_query - expected_partial_query)), np.max(np.abs(partial_key - expected_partial_key)))
    )
    head_max_abs = float(
        max(np.max(np.abs(head_query - expected_head_query)), np.max(np.abs(head_key - expected_head_key)))
    )
    position_max_abs = float(
        max(
            np.max(np.abs(position_query - expected_head_query)),
            np.max(np.abs(position_key - expected_head_key)),
        )
    )
    print(
        f"num_q_heads={num_q_heads} num_kv_heads={num_kv_heads} head_dim={head_dim} "
        f"rotary_dim={rotary_dim} partial_max_abs={partial_max_abs:.3g} "
        f"head_max_abs={head_max_abs:.3g} position_max_abs={position_max_abs:.3g}"
    )
    print("head_query0=", head_query[0].tolist())
    return 0 if partial_max_abs == 0.0 and head_max_abs <= 1.0e-6 and position_max_abs <= 1.0e-6 else 1

def qwen35_router_hip_smoke(
    rows: int,
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.moe import (
        build_qwen35_router,
        qwen35_router_topk_shared_out_bf16,
    )

    if rows < 1:
        raise ValueError("--rows must be >= 1")
    if hidden_size < 1:
        raise ValueError("--hidden-size must be >= 1")

    num_experts = 8
    num_rows = num_experts + 1
    top_k = 4
    threads = 64
    x_f32 = np.linspace(-0.75, 1.25, rows * hidden_size, dtype=np.float32).reshape(
        rows, hidden_size
    )
    # Make expert rows separated enough that top-k order is stable despite reduction-order noise.
    weight_f32 = np.empty((num_rows, hidden_size), dtype=np.float32)
    base = np.linspace(-0.5, 0.75, hidden_size, dtype=np.float32)
    for expert in range(num_rows):
        weight_f32[expert] = base * (0.25 + expert * 0.125) + expert * 0.05
    x_bits = _float32_to_bf16_bits(x_f32)
    weight_bits = _float32_to_bf16_bits(weight_f32)
    logits = np.empty((rows, num_rows), dtype=np.float32)
    selected = np.empty((rows, top_k), dtype=np.int64)
    routing = np.empty((rows, top_k), dtype=np.float32)

    x_bf32 = _bf16_bits_to_float32(x_bits)
    weight_bf32 = _bf16_bits_to_float32(weight_bits)
    expected_logits = (x_bf32.astype(np.float32) @ weight_bf32.astype(np.float32).T).astype(
        np.float32
    )
    router_logits = expected_logits[:, :num_experts]
    expected_selected = np.argsort(-router_logits, axis=1)[:, :top_k].astype(np.int64)
    topk_logits = np.take_along_axis(router_logits, expected_selected, axis=1)
    shifted = topk_logits - np.max(topk_logits, axis=1, keepdims=True)
    expected_routing = np.exp(shifted).astype(np.float32)
    expected_routing = (expected_routing / np.sum(expected_routing, axis=1, keepdims=True)).astype(
        np.float32
    )

    runtime = get_hip_runtime()
    library = build_qwen35_router(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    x_dev = weight_dev = logits_dev = selected_dev = routing_dev = None
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        weight_dev = malloc(weight_bits.nbytes, runtime=runtime)
        logits_dev = malloc(logits.nbytes, runtime=runtime)
        selected_dev = malloc(selected.nbytes, runtime=runtime)
        routing_dev = malloc(routing.nbytes, runtime=runtime)
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(weight_bits), runtime=runtime)
        qwen35_router_topk_shared_out_bf16(
            x_dev.ptr,
            weight_dev.ptr,
            logits_dev.ptr,
            selected_dev.ptr,
            routing_dev.ptr,
            rows,
            hidden_size,
            num_rows,
            num_experts,
            top_k,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(logits), logits_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(selected), selected_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(routing), routing_dev, runtime=runtime)
    finally:
        for buffer in (routing_dev, selected_dev, logits_dev, weight_dev, x_dev):
            if buffer is not None:
                free(buffer, runtime=runtime)

    logits_max_abs = float(np.max(np.abs(logits - expected_logits)))
    routing_max_abs = float(np.max(np.abs(routing - expected_routing)))
    selected_match = bool(np.array_equal(selected, expected_selected))
    print(
        f"rows={rows} hidden_size={hidden_size} num_experts={num_experts} top_k={top_k} "
        f"logits_max_abs={logits_max_abs} routing_max_abs={routing_max_abs} "
        f"selected_match={selected_match}"
    )
    print("selected0=", selected[0].tolist())
    print("routing0=", routing[0].tolist())
    return 0 if selected_match and logits_max_abs <= 2e-5 and routing_max_abs <= 2e-5 else 1



def paro_moe_c1_hip_smoke(
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.fused import (
        build_paro_combine,
        build_paro_silu,
        silu_mul_dual_out_bf16,
        weighted_sum_shared_gate_combine_residual_out_bf16_f32w,
    )
    from hipengine.kernels.hip_gfx1100.moe import (
        build_qwen35_router,
        qwen35_router_topk_shared_out_bf16,
    )
    from hipengine.kernels.hip_gfx1100.norm import (
        build_qwen35_rmsnorm,
        paro_rmsnorm_out_bf16,
    )
    from hipengine.kernels.hip_gfx1100.quant import (
        build_paro_awq_gemv,
        build_w8a16_linear,
        gemv_awq_selected_dual_pack8_strided_bf16,
        gemv_awq_selected_pack8_strided_bf16,
        w8a16_linear_bf16_lowp_out,
    )

    if hidden_size != 8:
        raise ValueError("paro-moe-c1-hip currently uses --hidden-size 8")

    tokens = 1
    top_k = 2
    num_experts = 3
    num_router_rows = num_experts + 1
    intermediate_size = 8
    group_size = 8
    out_packed = hidden_size // 8
    threads_gemv = 64
    threads_small = 64
    threads_combine = 256
    eps = 1.0e-6

    hidden_f32 = np.asarray(
        [[-1.0, -0.75, -0.375, -0.125, 0.25, 0.5, 0.875, 1.125]], dtype=np.float32
    )
    residual_bits = _float32_to_bf16_bits(hidden_f32)
    norm_weight_bits = _float32_to_bf16_bits(
        np.asarray([1.0, 0.875, 1.125, 0.75, 1.25, 0.625, 1.375, 0.5], dtype=np.float32)
    )
    router_weight_bits = _float32_to_bf16_bits(
        np.asarray(
            [
                [0.125, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1.0],
                [-0.5, 0.375, -0.25, 0.125, 1.0, -0.875, 0.75, -0.625],
                [0.875, 0.125, -0.75, -0.25, 0.5, 0.375, -1.0, -0.625],
                [0.25, 0.5, -0.125, -0.375, 0.75, -0.625, 1.0, -0.875],
            ],
            dtype=np.float32,
        )
    )
    qweight_gate, qzeros_gate, scales_gate_bits = _make_pack8_fixture(
        num_experts, hidden_size, out_packed, group_size, salt=5
    )
    qweight_up, qzeros_up, scales_up_bits = _make_pack8_fixture(
        num_experts, hidden_size, out_packed, group_size, salt=7
    )
    qweight_down, qzeros_down, scales_down_bits = _make_pack8_fixture(
        num_experts, intermediate_size, out_packed, group_size, salt=9
    )
    shared_gate_up_weight = _int8_pattern(2 * intermediate_size, hidden_size, salt=11)
    shared_gate_up_scale = np.asarray(
        [0.125, 0.25, 0.5, 1.0] * ((2 * intermediate_size + 3) // 4),
        dtype=np.float32,
    )[: 2 * intermediate_size]
    shared_down_weight = _int8_pattern(hidden_size, intermediate_size, salt=13)
    shared_down_scale = np.asarray(
        [0.0625, 0.125, 0.25, 0.5] * ((hidden_size + 3) // 4), dtype=np.float32
    )[:hidden_size]

    norm_bits = np.empty((tokens, hidden_size), dtype=np.uint16)
    router_logits = np.empty((tokens, num_router_rows), dtype=np.float32)
    selected = np.empty((tokens * top_k,), dtype=np.int64)
    routing = np.empty((tokens * top_k,), dtype=np.float32)
    selected_gate_up_bits = np.empty((top_k, 2 * intermediate_size), dtype=np.uint16)
    selected_act_bits = np.empty((top_k, intermediate_size), dtype=np.uint16)
    selected_down_bits = np.empty((top_k, hidden_size), dtype=np.uint16)
    shared_gate_up_bits = np.empty((tokens, 2 * intermediate_size), dtype=np.uint16)
    shared_act_bits = np.empty((tokens, intermediate_size), dtype=np.uint16)
    shared_out_bits = np.empty((tokens, hidden_size), dtype=np.uint16)
    final_bits = np.empty((tokens, hidden_size), dtype=np.uint16)

    expected_norm_bits = _paro_rmsnorm_reference(residual_bits, norm_weight_bits, eps)
    expected_norm = _bf16_bits_to_float32(expected_norm_bits)
    expected_logits = expected_norm @ _bf16_bits_to_float32(router_weight_bits).T
    expected_selected, expected_routing = _router_topk_reference(
        expected_logits[0, :num_experts], top_k
    )
    expected_gate_bits = _selected_pack8_reference(
        expected_norm_bits,
        expected_selected,
        qweight_gate,
        qzeros_gate,
        scales_gate_bits,
        group_size,
        qweight_transposed=False,
    )
    expected_up_bits = _selected_pack8_reference(
        expected_norm_bits,
        expected_selected,
        qweight_up,
        qzeros_up,
        scales_up_bits,
        group_size,
        qweight_transposed=False,
    )
    expected_gate_up_bits = np.concatenate([expected_gate_bits, expected_up_bits], axis=1)
    expected_gate_up = _bf16_bits_to_float32(expected_gate_up_bits)
    expected_selected_act_bits = _float32_to_bf16_bits(
        _silu_np(expected_gate_up[:, :intermediate_size]) * expected_gate_up[:, intermediate_size:]
    )
    expected_selected_down_bits = _selected_pack8_reference(
        expected_selected_act_bits,
        expected_selected,
        qweight_down,
        qzeros_down,
        scales_down_bits,
        group_size,
        qweight_transposed=False,
    )
    expected_shared_gate_up_bits = _float32_to_bf16_bits(
        (expected_norm @ shared_gate_up_weight.astype(np.float32).T).astype(np.float32)
        * shared_gate_up_scale.reshape(1, 2 * intermediate_size)
    )
    expected_shared_gate_up = _bf16_bits_to_float32(expected_shared_gate_up_bits)
    expected_shared_act_bits = _float32_to_bf16_bits(
        _silu_np(expected_shared_gate_up[:, :intermediate_size])
        * expected_shared_gate_up[:, intermediate_size:]
    )
    expected_shared_act = _bf16_bits_to_float32(expected_shared_act_bits)
    expected_shared_out_bits = _float32_to_bf16_bits(
        (expected_shared_act @ shared_down_weight.astype(np.float32).T).astype(np.float32)
        * shared_down_scale.reshape(1, hidden_size)
    )
    selected_down_bf32 = _bf16_bits_to_float32(expected_selected_down_bits)
    expected_weighted_bits = _float32_to_bf16_bits(
        np.sum(selected_down_bf32 * expected_routing.reshape(top_k, 1), axis=0, dtype=np.float32).reshape(
            1, hidden_size
        )
    )
    expected_weighted = _bf16_bits_to_float32(expected_weighted_bits)
    shared_gate = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-np.float32(expected_logits[0, num_experts]), dtype=np.float32)
    )
    expected_final_bits = _float32_to_bf16_bits(
        _bf16_bits_to_float32(residual_bits)
        + expected_weighted
        + shared_gate * _bf16_bits_to_float32(expected_shared_out_bits)
    )

    runtime = get_hip_runtime()
    norm_library = build_qwen35_rmsnorm(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    router_library = build_qwen35_router(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    awq_library = build_paro_awq_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    silu_library = build_paro_silu(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    w8a16_library = build_w8a16_linear(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    combine_library = build_paro_combine(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        hidden_dev = dev(residual_bits)
        norm_weight_dev = dev(norm_weight_bits)
        router_weight_dev = dev(router_weight_bits)
        qweight_gate_dev = dev(qweight_gate)
        qzeros_gate_dev = dev(qzeros_gate)
        scales_gate_dev = dev(scales_gate_bits)
        qweight_up_dev = dev(qweight_up)
        qzeros_up_dev = dev(qzeros_up)
        scales_up_dev = dev(scales_up_bits)
        qweight_down_dev = dev(qweight_down)
        qzeros_down_dev = dev(qzeros_down)
        scales_down_dev = dev(scales_down_bits)
        shared_gate_up_weight_dev = dev(shared_gate_up_weight)
        shared_gate_up_scale_dev = dev(shared_gate_up_scale)
        shared_down_weight_dev = dev(shared_down_weight)
        shared_down_scale_dev = dev(shared_down_scale)
        norm_dev = out_dev(norm_bits)
        logits_dev = out_dev(router_logits)
        selected_dev = out_dev(selected)
        routing_dev = out_dev(routing)
        selected_gate_up_dev = out_dev(selected_gate_up_bits)
        selected_act_dev = out_dev(selected_act_bits)
        selected_down_dev = out_dev(selected_down_bits)
        shared_gate_up_dev = out_dev(shared_gate_up_bits)
        shared_act_dev = out_dev(shared_act_bits)
        shared_out_dev = out_dev(shared_out_bits)
        final_dev = out_dev(final_bits)

        paro_rmsnorm_out_bf16(
            hidden_dev.ptr,
            norm_weight_dev.ptr,
            norm_dev.ptr,
            tokens,
            hidden_size,
            eps,
            library=norm_library,
            runtime=runtime,
        )
        qwen35_router_topk_shared_out_bf16(
            norm_dev.ptr,
            router_weight_dev.ptr,
            logits_dev.ptr,
            selected_dev.ptr,
            routing_dev.ptr,
            tokens,
            hidden_size,
            num_router_rows,
            num_experts,
            top_k,
            threads=512,
            library=router_library,
            runtime=runtime,
        )
        gemv_awq_selected_dual_pack8_strided_bf16(
            norm_dev.ptr,
            selected_dev.ptr,
            qweight_gate_dev.ptr,
            qzeros_gate_dev.ptr,
            scales_gate_dev.ptr,
            qweight_up_dev.ptr,
            qzeros_up_dev.ptr,
            scales_up_dev.ptr,
            selected_gate_up_dev.ptr,
            tokens,
            top_k,
            hidden_size,
            out_packed,
            out_packed,
            num_experts,
            group_size,
            threads=threads_gemv,
            library=awq_library,
            runtime=runtime,
        )
        silu_mul_dual_out_bf16(
            selected_gate_up_dev.ptr,
            selected_act_dev.ptr,
            top_k,
            intermediate_size,
            threads=threads_small,
            library=silu_library,
            runtime=runtime,
        )
        gemv_awq_selected_pack8_strided_bf16(
            selected_act_dev.ptr,
            selected_dev.ptr,
            qweight_down_dev.ptr,
            qzeros_down_dev.ptr,
            scales_down_dev.ptr,
            selected_down_dev.ptr,
            top_k,
            intermediate_size,
            out_packed,
            num_experts,
            group_size,
            threads=threads_gemv,
            library=awq_library,
            runtime=runtime,
        )
        w8a16_linear_bf16_lowp_out(
            norm_dev.ptr,
            shared_gate_up_weight_dev.ptr,
            shared_gate_up_scale_dev.ptr,
            shared_gate_up_dev.ptr,
            tokens,
            hidden_size,
            2 * intermediate_size,
            threads=threads_small,
            library=w8a16_library,
            runtime=runtime,
        )
        silu_mul_dual_out_bf16(
            shared_gate_up_dev.ptr,
            shared_act_dev.ptr,
            tokens,
            intermediate_size,
            threads=threads_small,
            library=silu_library,
            runtime=runtime,
        )
        w8a16_linear_bf16_lowp_out(
            shared_act_dev.ptr,
            shared_down_weight_dev.ptr,
            shared_down_scale_dev.ptr,
            shared_out_dev.ptr,
            tokens,
            intermediate_size,
            hidden_size,
            threads=threads_small,
            library=w8a16_library,
            runtime=runtime,
        )
        weighted_sum_shared_gate_combine_residual_out_bf16_f32w(
            selected_down_dev.ptr,
            routing_dev.ptr,
            shared_out_dev.ptr,
            logits_dev.ptr + num_experts * np.dtype(np.float32).itemsize,
            hidden_dev.ptr,
            final_dev.ptr,
            top_k,
            hidden_size,
            threads=threads_combine,
            library=combine_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(norm_bits), norm_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(router_logits), logits_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(selected), selected_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(routing), routing_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(selected_gate_up_bits), selected_gate_up_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(selected_act_bits), selected_act_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(selected_down_bits), selected_down_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(shared_out_bits), shared_out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(final_bits), final_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    norm_mismatch = int(np.count_nonzero(norm_bits != expected_norm_bits))
    selected_match = bool(np.array_equal(selected, expected_selected))
    routing_max_abs = float(np.max(np.abs(routing - expected_routing)))
    logits_max_abs = float(np.max(np.abs(router_logits - expected_logits)))
    selected_gate_up_mismatch = int(
        np.count_nonzero(selected_gate_up_bits != expected_gate_up_bits)
    )
    selected_act_mismatch = int(np.count_nonzero(selected_act_bits != expected_selected_act_bits))
    selected_down_mismatch = int(
        np.count_nonzero(selected_down_bits != expected_selected_down_bits)
    )
    shared_out_mismatch = int(np.count_nonzero(shared_out_bits != expected_shared_out_bits))
    final_mismatch = int(np.count_nonzero(final_bits != expected_final_bits))
    final_max_abs = float(
        np.max(np.abs(_bf16_bits_to_float32(final_bits) - _bf16_bits_to_float32(expected_final_bits)))
    )
    print(
        f"hidden_size={hidden_size} top_k={top_k} "
        f"norm_mismatch={norm_mismatch} selected_match={selected_match} "
        f"logits_max_abs={logits_max_abs} routing_max_abs={routing_max_abs} "
        f"selected_gate_up_mismatch={selected_gate_up_mismatch} "
        f"selected_act_mismatch={selected_act_mismatch} "
        f"selected_down_mismatch={selected_down_mismatch} "
        f"shared_out_mismatch={shared_out_mismatch} "
        f"final_mismatch={final_mismatch} final_max_abs={final_max_abs}"
    )
    print("selected=", selected.tolist(), "routing=", routing.tolist())
    print("final=", _bf16_bits_to_float32(final_bits)[0].tolist())
    return (
        0
        if norm_mismatch == 0
        and selected_match
        and logits_max_abs <= 2e-5
        and routing_max_abs <= 2e-5
        and selected_gate_up_mismatch == 0
        and selected_act_mismatch == 0
        and selected_down_mismatch == 0
        and shared_out_mismatch == 0
        and final_mismatch == 0
        else 1
    )



def paro_moe_c1_state_hip_smoke(
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.core.tensor import Tensor
    from hipengine.kernels.hip_gfx1100.fused import build_paro_combine, build_paro_silu
    from hipengine.kernels.hip_gfx1100.moe import build_qwen35_router
    from hipengine.kernels.hip_gfx1100.norm import build_qwen35_rmsnorm, paro_rmsnorm_out_bf16
    from hipengine.kernels.hip_gfx1100.quant import build_paro_awq_gemv, build_w8a16_linear
    from hipengine.loading.materialize import DeviceTensorAllocation, DeviceWeightMap
    from hipengine.loading.qwen35_paro import Qwen35ParoConfig, Qwen35ParoLayerDeviceWeights
    from hipengine.loading.safetensors import TensorInfo
    from hipengine.runtime import Qwen35ParoDecodeState, RuntimeWorkspace

    if hidden_size != 8:
        raise ValueError("paro-moe-c1-state-hip currently uses --hidden-size 8")

    tokens = 1
    top_k = 2
    num_experts = 3
    num_router_rows = num_experts + 1
    intermediate_size = 8
    group_size = 8
    out_packed = hidden_size // 8
    eps = 1.0e-6

    hidden_f32 = np.asarray(
        [[-1.0, -0.75, -0.375, -0.125, 0.25, 0.5, 0.875, 1.125]], dtype=np.float32
    )
    residual_bits = _float32_to_bf16_bits(hidden_f32)
    norm_weight_bits = _float32_to_bf16_bits(
        np.asarray([1.0, 0.875, 1.125, 0.75, 1.25, 0.625, 1.375, 0.5], dtype=np.float32)
    )
    router_weight_bits = _float32_to_bf16_bits(
        np.asarray(
            [
                [0.125, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1.0],
                [-0.5, 0.375, -0.25, 0.125, 1.0, -0.875, 0.75, -0.625],
                [0.875, 0.125, -0.75, -0.25, 0.5, 0.375, -1.0, -0.625],
                [0.25, 0.5, -0.125, -0.375, 0.75, -0.625, 1.0, -0.875],
            ],
            dtype=np.float32,
        )
    )
    qweight_gate, qzeros_gate, scales_gate_bits = _make_pack8_fixture(
        num_experts, hidden_size, out_packed, group_size, salt=5
    )
    qweight_up, qzeros_up, scales_up_bits = _make_pack8_fixture(
        num_experts, hidden_size, out_packed, group_size, salt=7
    )
    qweight_down, qzeros_down, scales_down_bits = _make_pack8_fixture(
        num_experts, intermediate_size, out_packed, group_size, salt=9
    )
    qweight_gate_t = np.ascontiguousarray(np.swapaxes(qweight_gate, 1, 2))
    qweight_up_t = np.ascontiguousarray(np.swapaxes(qweight_up, 1, 2))
    qweight_down_t = np.ascontiguousarray(np.swapaxes(qweight_down, 1, 2))
    down_pairs = np.asarray([[0, 4, 1, 5, 2, 6, 3, 7]], dtype=np.int16)
    down_theta_bits = _float32_to_bf16_bits(np.zeros((1, hidden_size // 2), dtype=np.float32))
    down_scales_bits = _float32_to_bf16_bits(np.ones((intermediate_size,), dtype=np.float32))
    shared_gate_up_weight = _int8_pattern(2 * intermediate_size, hidden_size, salt=11)
    shared_gate_up_scale = np.asarray(
        [0.125, 0.25, 0.5, 1.0] * ((2 * intermediate_size + 3) // 4), dtype=np.float32
    )[: 2 * intermediate_size]
    shared_down_weight = _int8_pattern(hidden_size, intermediate_size, salt=13)
    shared_down_scale = np.asarray([0.0625, 0.125, 0.25, 0.5] * 2, dtype=np.float32)[:hidden_size]

    expected_norm_bits = _paro_rmsnorm_reference(residual_bits, norm_weight_bits, eps)
    expected_norm = _bf16_bits_to_float32(expected_norm_bits)
    expected_logits = expected_norm @ _bf16_bits_to_float32(router_weight_bits).T
    expected_selected, expected_routing = _router_topk_reference(expected_logits[0, :num_experts], top_k)
    expected_gate_bits = _selected_pack8_reference(
        expected_norm_bits,
        expected_selected,
        qweight_gate_t,
        qzeros_gate,
        scales_gate_bits,
        group_size,
        qweight_transposed=True,
    )
    expected_up_bits = _selected_pack8_reference(
        expected_norm_bits,
        expected_selected,
        qweight_up_t,
        qzeros_up,
        scales_up_bits,
        group_size,
        qweight_transposed=True,
    )
    expected_gate_up_bits = np.concatenate([expected_gate_bits, expected_up_bits], axis=1)
    expected_gate_up = _bf16_bits_to_float32(expected_gate_up_bits)
    expected_down_input_bits = _float32_to_bf16_bits(
        _silu_np(expected_gate_up[:, :intermediate_size]) * expected_gate_up[:, intermediate_size:]
    )
    expected_selected_down_bits = _selected_pack8_reference(
        expected_down_input_bits,
        expected_selected,
        qweight_down_t,
        qzeros_down,
        scales_down_bits,
        group_size,
        qweight_transposed=True,
    )
    expected_shared_gate_up_bits = _float32_to_bf16_bits(
        (expected_norm @ shared_gate_up_weight.astype(np.float32).T).astype(np.float32)
        * shared_gate_up_scale.reshape(1, 2 * intermediate_size)
    )
    expected_shared_gate_up = _bf16_bits_to_float32(expected_shared_gate_up_bits)
    expected_shared_act_bits = _float32_to_bf16_bits(
        _silu_np(expected_shared_gate_up[:, :intermediate_size]) * expected_shared_gate_up[:, intermediate_size:]
    )
    expected_shared_act = _bf16_bits_to_float32(expected_shared_act_bits)
    expected_shared_out_bits = _float32_to_bf16_bits(
        (expected_shared_act @ shared_down_weight.astype(np.float32).T).astype(np.float32)
        * shared_down_scale.reshape(1, hidden_size)
    )
    selected_down_f32 = _bf16_bits_to_float32(expected_selected_down_bits)
    expected_weighted_bits = _float32_to_bf16_bits(
        np.sum(selected_down_f32 * expected_routing.reshape(top_k, 1), axis=0, dtype=np.float32).reshape(1, hidden_size)
    )
    expected_weighted = _bf16_bits_to_float32(expected_weighted_bits)
    shared_gate = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-np.float32(expected_logits[0, num_experts]), dtype=np.float32)
    )
    expected_final_bits = _float32_to_bf16_bits(
        _bf16_bits_to_float32(residual_bits)
        + expected_weighted
        + shared_gate * _bf16_bits_to_float32(expected_shared_out_bits)
    )

    runtime = get_hip_runtime()
    libraries = {
        "router": build_qwen35_router(load=True, compiler_version=compiler_version, require_cached=require_cached_build),
        "awq": build_paro_awq_gemv(load=True, compiler_version=compiler_version, require_cached=require_cached_build),
        "silu": build_paro_silu(load=True, compiler_version=compiler_version, require_cached=require_cached_build),
        "w8a16": build_w8a16_linear(load=True, compiler_version=compiler_version, require_cached=require_cached_build),
        "combine": build_paro_combine(load=True, compiler_version=compiler_version, require_cached=require_cached_build),
    }
    norm_library = build_qwen35_rmsnorm(load=True, compiler_version=compiler_version, require_cached=require_cached_build)
    buffers = []
    state = None

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    def weight(name: str, array: np.ndarray, dtype: str) -> DeviceTensorAllocation:
        buffer = malloc(array.nbytes, runtime=runtime)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        source_dtype = {"bf16": "BF16", "fp32": "F32", "int8": "I8", "int16": "I16", "int32": "I32"}[dtype]
        return DeviceTensorAllocation(
            name=name,
            source=TensorInfo(name=name, shard_path=Path(__file__), dtype=source_dtype, shape=tuple(array.shape)),
            buffer=buffer,
            tensor=Tensor.from_handle(buffer.ptr, array.shape, dtype, Device("hip", 0)),
        )

    try:
        hidden_dev = dev(residual_bits)
        norm_weight_dev = dev(norm_weight_bits)
        norm_bits = np.empty((tokens, hidden_size), dtype=np.uint16)
        norm_dev = out_dev(norm_bits)
        paro_rmsnorm_out_bf16(
            hidden_dev.ptr,
            norm_weight_dev.ptr,
            norm_dev.ptr,
            tokens,
            hidden_size,
            eps,
            library=norm_library,
            runtime=runtime,
        )
        weights = DeviceWeightMap(
            {
                "layers.0.mlp.router_shared_gate.weight": weight("layers.0.mlp.router_shared_gate.weight", router_weight_bits, "bf16"),
                "layers.0.mlp.experts.stacked_gate_qweight_pack8_decode": weight("layers.0.mlp.experts.stacked_gate_qweight_pack8_decode", qweight_gate_t, "int32"),
                "layers.0.mlp.experts.stacked_gate_qzeros": weight("layers.0.mlp.experts.stacked_gate_qzeros", qzeros_gate, "int32"),
                "layers.0.mlp.experts.stacked_gate_scales": weight("layers.0.mlp.experts.stacked_gate_scales", scales_gate_bits, "bf16"),
                "layers.0.mlp.experts.stacked_up_qweight_pack8_decode": weight("layers.0.mlp.experts.stacked_up_qweight_pack8_decode", qweight_up_t, "int32"),
                "layers.0.mlp.experts.stacked_up_qzeros": weight("layers.0.mlp.experts.stacked_up_qzeros", qzeros_up, "int32"),
                "layers.0.mlp.experts.stacked_up_scales": weight("layers.0.mlp.experts.stacked_up_scales", scales_up_bits, "bf16"),
                "layers.0.mlp.experts.stacked_down_qweight_pack8_decode": weight("layers.0.mlp.experts.stacked_down_qweight_pack8_decode", qweight_down_t, "int32"),
                "layers.0.mlp.experts.stacked_down_qzeros": weight("layers.0.mlp.experts.stacked_down_qzeros", qzeros_down, "int32"),
                "layers.0.mlp.experts.stacked_down_scales": weight("layers.0.mlp.experts.stacked_down_scales", scales_down_bits, "bf16"),
                "layers.0.mlp.experts.down_weight_pairs": weight("layers.0.mlp.experts.down_weight_pairs", down_pairs, "int16"),
                "layers.0.mlp.experts.down_weight_theta": weight("layers.0.mlp.experts.down_weight_theta", down_theta_bits, "bf16"),
                "layers.0.mlp.experts.down_weight_channel_scales": weight("layers.0.mlp.experts.down_weight_channel_scales", down_scales_bits, "bf16"),
                "layers.0.mlp.shared_expert.gate_up_weight_w8a16": weight("layers.0.mlp.shared_expert.gate_up_weight_w8a16", shared_gate_up_weight, "int8"),
                "layers.0.mlp.shared_expert.gate_up_weight_w8a16_scale": weight("layers.0.mlp.shared_expert.gate_up_weight_w8a16_scale", shared_gate_up_scale, "fp32"),
                "layers.0.mlp.shared_expert.down_weight_w8a16": weight("layers.0.mlp.shared_expert.down_weight_w8a16", shared_down_weight, "int8"),
                "layers.0.mlp.shared_expert.down_weight_w8a16_scale": weight("layers.0.mlp.shared_expert.down_weight_w8a16_scale", shared_down_scale, "fp32"),
            }
        )
        config = Qwen35ParoConfig(
            architecture="Qwen3_5MoeForConditionalGeneration",
            num_hidden_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            moe_intermediate_size=intermediate_size,
            shared_expert_intermediate_size=intermediate_size,
            layer_types=("full_attention",),
            quant_method="paroquant",
        )
        state = Qwen35ParoDecodeState(
            layer_weights=Qwen35ParoLayerDeviceWeights(config=config, layer_id=0, weights=weights),
            workspace=RuntimeWorkspace(runtime=runtime),
            runtime=runtime,
        )
        norm_tensor = Tensor.from_handle(norm_dev.ptr, norm_bits.shape, "bf16", Device("hip", 0))
        residual_tensor = Tensor.from_handle(hidden_dev.ptr, residual_bits.shape, "bf16", Device("hip", 0))
        scratch = state.reserve_moe_c1_scratch(tokens=tokens)
        final_tensor = state.run_moe_c1_bf16(norm_tensor, residual_tensor, scratch=scratch, group_size=group_size, library=libraries)
        runtime.device_synchronize()
        final_bits = np.empty((tokens, hidden_size), dtype=np.uint16)
        selected = np.empty((tokens, top_k), dtype=np.int64)
        routing = np.empty((tokens, top_k), dtype=np.float32)
        router_logits = np.empty((tokens, num_router_rows), dtype=np.float32)
        gate_up_bits = np.empty((top_k, 2 * intermediate_size), dtype=np.uint16)
        down_input_bits = np.empty((top_k, intermediate_size), dtype=np.uint16)
        down_out_bits = np.empty((top_k, hidden_size), dtype=np.uint16)
        shared_out_bits = np.empty((tokens, hidden_size), dtype=np.uint16)
        copy_device_to_host(host_array_ptr(norm_bits), norm_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(final_bits), DeviceBuffer(final_tensor.ptr, final_bits.nbytes), runtime=runtime)
        copy_device_to_host(host_array_ptr(selected), state.workspace.allocation("moe.selected_experts").buffer, runtime=runtime)
        copy_device_to_host(host_array_ptr(routing), state.workspace.allocation("moe.routing_weights").buffer, runtime=runtime)
        copy_device_to_host(host_array_ptr(router_logits), state.workspace.allocation("moe.router_logits").buffer, runtime=runtime)
        copy_device_to_host(host_array_ptr(gate_up_bits), state.workspace.allocation("moe.gate_up").buffer, runtime=runtime)
        copy_device_to_host(host_array_ptr(down_input_bits), state.workspace.allocation("moe.down_input").buffer, runtime=runtime)
        copy_device_to_host(host_array_ptr(down_out_bits), state.workspace.allocation("moe.down_out").buffer, runtime=runtime)
        copy_device_to_host(host_array_ptr(shared_out_bits), state.workspace.allocation("moe.shared_out").buffer, runtime=runtime)
        state.free()
        state = None
    finally:
        if state is not None:
            state.free()
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    norm_mismatch = int(np.count_nonzero(norm_bits != expected_norm_bits))
    selected_match = bool(np.array_equal(selected.reshape(-1), expected_selected))
    routing_max_abs = float(np.max(np.abs(routing.reshape(-1) - expected_routing)))
    logits_max_abs = float(np.max(np.abs(router_logits - expected_logits)))
    gate_up_mismatch = int(np.count_nonzero(gate_up_bits != expected_gate_up_bits))
    down_input_mismatch = int(np.count_nonzero(down_input_bits != expected_down_input_bits))
    down_out_mismatch = int(np.count_nonzero(down_out_bits != expected_selected_down_bits))
    shared_out_mismatch = int(np.count_nonzero(shared_out_bits != expected_shared_out_bits))
    final_mismatch = int(np.count_nonzero(final_bits != expected_final_bits))
    final_max_abs = float(np.max(np.abs(_bf16_bits_to_float32(final_bits) - _bf16_bits_to_float32(expected_final_bits))))
    print(
        f"hidden_size={hidden_size} top_k={top_k} state_path=1 "
        f"norm_mismatch={norm_mismatch} selected_match={selected_match} "
        f"logits_max_abs={logits_max_abs} routing_max_abs={routing_max_abs} "
        f"gate_up_mismatch={gate_up_mismatch} down_input_mismatch={down_input_mismatch} "
        f"down_out_mismatch={down_out_mismatch} shared_out_mismatch={shared_out_mismatch} "
        f"final_mismatch={final_mismatch} final_max_abs={final_max_abs}"
    )
    print("selected=", selected.reshape(-1).tolist(), "routing=", routing.reshape(-1).tolist())
    print("final=", _bf16_bits_to_float32(final_bits)[0].tolist())
    return (
        0
        if norm_mismatch == 0
        and selected_match
        and logits_max_abs <= 2e-5
        and routing_max_abs <= 2e-5
        and gate_up_mismatch == 0
        and down_input_mismatch == 0
        and down_out_mismatch == 0
        and shared_out_mismatch == 0
        and final_mismatch == 0
        else 1
    )

def dense_gemv_hip_smoke(
    rows: int,
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.linear import build_dense_gemv, dense_gemv_out_bf16

    if rows < 1:
        raise ValueError("--rows must be >= 1")
    if hidden_size < 1:
        raise ValueError("--hidden-size must be >= 1")

    out_features = 8
    threads = 256
    x_f32 = np.linspace(-0.75, 1.0, rows * hidden_size, dtype=np.float32).reshape(
        rows, hidden_size
    )
    weight_f32 = np.empty((out_features, hidden_size), dtype=np.float32)
    for row in range(out_features):
        weight_f32[row] = np.asarray(
            [[-0.5, -0.25, 0.25, 0.5][(row + col) % 4] for col in range(hidden_size)],
            dtype=np.float32,
        )
    x_bits = _float32_to_bf16_bits(x_f32)
    weight_bits = _float32_to_bf16_bits(weight_f32)
    out_bits = np.empty((rows, out_features), dtype=np.uint16)
    expected_bits = _float32_to_bf16_bits(
        _bf16_bits_to_float32(x_bits) @ _bf16_bits_to_float32(weight_bits).T
    )

    runtime = get_hip_runtime()
    library = build_dense_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        x_dev = dev(x_bits)
        weight_dev = dev(weight_bits)
        out_dev_buf = out_dev(out_bits)
        dense_gemv_out_bf16(
            x_dev.ptr,
            weight_dev.ptr,
            out_dev_buf.ptr,
            rows,
            hidden_size,
            out_features,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_bits), out_dev_buf, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    mismatch = int(np.count_nonzero(out_bits != expected_bits))
    max_abs = float(
        np.max(np.abs(_bf16_bits_to_float32(out_bits) - _bf16_bits_to_float32(expected_bits)))
    )
    print(
        f"rows={rows} hidden_size={hidden_size} out_features={out_features} "
        f"mismatch={mismatch} max_abs={max_abs}"
    )
    print("dense_gemv_row0=", _bf16_bits_to_float32(out_bits)[0, : min(8, out_features)].tolist())
    return 0 if mismatch == 0 else 1

def w8a16_shared_expert_hip_smoke(
    rows: int,
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.fused import (
        build_paro_silu,
        silu_mul_dual_out_bf16,
    )
    from hipengine.kernels.hip_gfx1100.quant import (
        build_w8a16_linear,
        w8a16_linear_bf16_lowp_out,
    )

    if rows < 1:
        raise ValueError("--rows must be >= 1")
    if hidden_size < 2 or hidden_size % 2 != 0:
        raise ValueError("--hidden-size must be >= 2 and even")

    intermediate_size = hidden_size // 2
    gate_up_features = intermediate_size * 2
    threads = 64
    x_f32 = np.linspace(-1.0, 0.75, rows * hidden_size, dtype=np.float32).reshape(
        rows, hidden_size
    )
    x_bits = _float32_to_bf16_bits(x_f32)
    gate_up_weight = _int8_pattern(gate_up_features, hidden_size, salt=1)
    down_weight = _int8_pattern(hidden_size, intermediate_size, salt=3)
    gate_up_scale = np.asarray(
        [0.125, 0.25, 0.5, 1.0] * ((gate_up_features + 3) // 4), dtype=np.float32
    )[:gate_up_features]
    down_scale = np.asarray(
        [0.0625, 0.125, 0.25, 0.5] * ((hidden_size + 3) // 4), dtype=np.float32
    )[:hidden_size]
    gate_up_bits = np.empty((rows, gate_up_features), dtype=np.uint16)
    intermediate_bits = np.empty((rows, intermediate_size), dtype=np.uint16)
    out_bits = np.empty((rows, hidden_size), dtype=np.uint16)

    x_bf32 = _bf16_bits_to_float32(x_bits)
    gate_up_f32 = (x_bf32 @ gate_up_weight.astype(np.float32).T).astype(np.float32)
    gate_up_f32 *= gate_up_scale.reshape(1, gate_up_features)
    expected_gate_up_bits = _float32_to_bf16_bits(gate_up_f32)
    expected_gate_up = _bf16_bits_to_float32(expected_gate_up_bits)
    gate = expected_gate_up[:, :intermediate_size]
    up = expected_gate_up[:, intermediate_size:]
    expected_intermediate_bits = _float32_to_bf16_bits(_silu_np(gate) * up)
    expected_intermediate = _bf16_bits_to_float32(expected_intermediate_bits)
    expected_out_f32 = (expected_intermediate @ down_weight.astype(np.float32).T).astype(
        np.float32
    )
    expected_out_f32 *= down_scale.reshape(1, hidden_size)
    expected_out_bits = _float32_to_bf16_bits(expected_out_f32)

    runtime = get_hip_runtime()
    w8a16_library = build_w8a16_linear(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    silu_library = build_paro_silu(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        x_dev = dev(x_bits)
        gate_up_weight_dev = dev(gate_up_weight)
        gate_up_scale_dev = dev(gate_up_scale)
        down_weight_dev = dev(down_weight)
        down_scale_dev = dev(down_scale)
        gate_up_dev = out_dev(gate_up_bits)
        intermediate_dev = out_dev(intermediate_bits)
        out_dev_buf = out_dev(out_bits)
        w8a16_linear_bf16_lowp_out(
            x_dev.ptr,
            gate_up_weight_dev.ptr,
            gate_up_scale_dev.ptr,
            gate_up_dev.ptr,
            rows,
            hidden_size,
            gate_up_features,
            threads=threads,
            library=w8a16_library,
            runtime=runtime,
        )
        silu_mul_dual_out_bf16(
            gate_up_dev.ptr,
            intermediate_dev.ptr,
            rows,
            intermediate_size,
            threads=threads,
            library=silu_library,
            runtime=runtime,
        )
        w8a16_linear_bf16_lowp_out(
            intermediate_dev.ptr,
            down_weight_dev.ptr,
            down_scale_dev.ptr,
            out_dev_buf.ptr,
            rows,
            intermediate_size,
            hidden_size,
            threads=threads,
            library=w8a16_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(gate_up_bits), gate_up_dev, runtime=runtime)
        copy_device_to_host(
            host_array_ptr(intermediate_bits), intermediate_dev, runtime=runtime
        )
        copy_device_to_host(host_array_ptr(out_bits), out_dev_buf, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    gate_up_mismatch = int(np.count_nonzero(gate_up_bits != expected_gate_up_bits))
    intermediate_mismatch = int(
        np.count_nonzero(intermediate_bits != expected_intermediate_bits)
    )
    out_mismatch = int(np.count_nonzero(out_bits != expected_out_bits))
    out_max_abs = float(
        np.max(
            np.abs(_bf16_bits_to_float32(out_bits) - _bf16_bits_to_float32(expected_out_bits))
        )
    )
    print(
        f"rows={rows} hidden_size={hidden_size} intermediate_size={intermediate_size} "
        f"gate_up_mismatch={gate_up_mismatch} "
        f"intermediate_mismatch={intermediate_mismatch} "
        f"out_mismatch={out_mismatch} out_max_abs={out_max_abs}"
    )
    print(
        "shared_out_row0=",
        _bf16_bits_to_float32(out_bits)[0, : min(8, hidden_size)].tolist(),
    )
    return (
        0
        if gate_up_mismatch == 0 and intermediate_mismatch == 0 and out_mismatch == 0
        else 1
    )

def w8a16_linear_hip_smoke(
    rows: int,
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant import (
        build_w8a16_linear,
        w8a16_linear_bf16_f32_out,
        w8a16_linear_bf16_lowp_out,
        w8a16_linear_f32_f32_out,
    )

    if rows < 1:
        raise ValueError("--rows must be >= 1")
    if hidden_size < 1:
        raise ValueError("--hidden-size must be >= 1")

    out_features = 8
    threads = 64
    x_f32 = np.linspace(-1.0, 1.0, rows * hidden_size, dtype=np.float32).reshape(
        rows, hidden_size
    )
    x_bits = _float32_to_bf16_bits(x_f32)
    x_bf32 = _bf16_bits_to_float32(x_bits)
    weight = np.empty((out_features, hidden_size), dtype=np.int8)
    for out_row in range(out_features):
        weight[out_row] = np.asarray(
            [((out_row + col) % 7) - 3 for col in range(hidden_size)], dtype=np.int8
        )
    weight_scale = np.asarray(
        [0.125, 0.25, 0.5, 1.0] * ((out_features + 3) // 4), dtype=np.float32
    )[:out_features]
    bf16_f32_out = np.empty((rows, out_features), dtype=np.float32)
    bf16_lowp_bits = np.empty((rows, out_features), dtype=np.uint16)
    f32_f32_out = np.empty((rows, out_features), dtype=np.float32)

    expected_bf16_f32 = (x_bf32.astype(np.float32) @ weight.astype(np.float32).T).astype(
        np.float32
    ) * weight_scale.reshape(1, out_features)
    expected_lowp_bits = _float32_to_bf16_bits(expected_bf16_f32)
    expected_f32_f32 = (x_f32.astype(np.float32) @ weight.astype(np.float32).T).astype(
        np.float32
    ) * weight_scale.reshape(1, out_features)

    runtime = get_hip_runtime()
    library = build_w8a16_linear(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        x_bits_dev = dev(x_bits)
        x_f32_dev = dev(x_f32)
        weight_dev = dev(weight)
        weight_scale_dev = dev(weight_scale)
        bf16_f32_dev = out_dev(bf16_f32_out)
        bf16_lowp_dev = out_dev(bf16_lowp_bits)
        f32_f32_dev = out_dev(f32_f32_out)
        w8a16_linear_bf16_f32_out(
            x_bits_dev.ptr,
            weight_dev.ptr,
            weight_scale_dev.ptr,
            bf16_f32_dev.ptr,
            rows,
            hidden_size,
            out_features,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        w8a16_linear_bf16_lowp_out(
            x_bits_dev.ptr,
            weight_dev.ptr,
            weight_scale_dev.ptr,
            bf16_lowp_dev.ptr,
            rows,
            hidden_size,
            out_features,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        w8a16_linear_f32_f32_out(
            x_f32_dev.ptr,
            weight_dev.ptr,
            weight_scale_dev.ptr,
            f32_f32_dev.ptr,
            rows,
            hidden_size,
            out_features,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(bf16_f32_out), bf16_f32_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(bf16_lowp_bits), bf16_lowp_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(f32_f32_out), f32_f32_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    bf16_f32_max_abs = float(np.max(np.abs(bf16_f32_out - expected_bf16_f32)))
    f32_f32_max_abs = float(np.max(np.abs(f32_f32_out - expected_f32_f32)))
    lowp_mismatch = int(np.count_nonzero(bf16_lowp_bits != expected_lowp_bits))
    lowp_max_abs = float(
        np.max(
            np.abs(_bf16_bits_to_float32(bf16_lowp_bits) - _bf16_bits_to_float32(expected_lowp_bits))
        )
    )
    print(
        f"rows={rows} hidden_size={hidden_size} out_features={out_features} "
        f"bf16_f32_max_abs={bf16_f32_max_abs} "
        f"f32_f32_max_abs={f32_f32_max_abs} "
        f"lowp_mismatch={lowp_mismatch} lowp_max_abs={lowp_max_abs}"
    )
    print("bf16_f32_row0=", bf16_f32_out[0, : min(8, out_features)].tolist())
    print("lowp_row0=", _bf16_bits_to_float32(bf16_lowp_bits)[0, : min(8, out_features)].tolist())
    return 0 if bf16_f32_max_abs <= 1e-5 and f32_f32_max_abs <= 1e-5 and lowp_mismatch == 0 else 1


def paro_combine_hip_smoke(
    rows: int,
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.fused import (
        build_paro_combine,
        shared_gate_combine_out_bf16,
        shared_gate_combine_residual_out_bf16,
        weighted_sum_out_bf16_f32w,
        weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w,
        weighted_sum_shared_gate_combine_residual_out_bf16_f32w,
    )

    if rows < 1:
        raise ValueError("--rows must be >= 1")
    if hidden_size < 1:
        raise ValueError("--hidden-size must be >= 1")

    features = hidden_size
    threads = 256
    values = np.empty((rows, features), dtype=np.float32)
    for row in range(rows):
        values[row] = np.asarray(
            [[-0.5, -0.25, 0.25, 0.5][(row + col) % 4] for col in range(features)],
            dtype=np.float32,
        )
    weights = np.asarray([0.125, 0.25, -0.5, 1.0] * ((rows + 3) // 4), dtype=np.float32)[:rows]
    expert = np.asarray([[0.125, -0.25, 0.5, -1.0] * ((features + 3) // 4)], dtype=np.float32)[:, :features]
    shared = np.asarray([[0.5, -0.5, 0.25, -0.25] * ((features + 3) // 4)], dtype=np.float32)[:, :features]
    residual = np.asarray([[1.0, -0.75, 0.375, -0.125] * ((features + 3) // 4)], dtype=np.float32)[:, :features]
    gate_logits = np.asarray([0.0], dtype=np.float32)

    values_bits = _float32_to_bf16_bits(values)
    expert_bits = _float32_to_bf16_bits(expert)
    shared_bits = _float32_to_bf16_bits(shared)
    residual_bits = _float32_to_bf16_bits(residual)
    weighted_bits = np.empty((1, features), dtype=np.uint16)
    weighted_shared_residual_bits = np.empty_like(weighted_bits)
    shared_combine_bits = np.empty_like(weighted_bits)
    shared_residual_bits = np.empty_like(weighted_bits)

    values_bf32 = _bf16_bits_to_float32(values_bits)
    expert_bf32 = _bf16_bits_to_float32(expert_bits)
    shared_bf32 = _bf16_bits_to_float32(shared_bits)
    residual_bf32 = _bf16_bits_to_float32(residual_bits)
    weighted_acc = np.sum(values_bf32 * weights.reshape(rows, 1), axis=0, dtype=np.float32).reshape(1, features)
    expected_weighted_bits = _float32_to_bf16_bits(weighted_acc)
    expected_weighted = _bf16_bits_to_float32(expected_weighted_bits)
    gate = np.float32(0.5)
    expected_shared_combine_bits = _float32_to_bf16_bits(expert_bf32 + gate * shared_bf32)
    expected_shared_residual_bits = _float32_to_bf16_bits(
        residual_bf32 + expert_bf32 + gate * shared_bf32
    )
    expected_weighted_shared_residual_bits = _float32_to_bf16_bits(
        residual_bf32 + expected_weighted + gate * shared_bf32
    )

    batch_tokens = 2
    batch_rows_per_token = rows
    batch_values = np.vstack([values + np.float32(0.0625 * token) for token in range(batch_tokens)]).astype(np.float32)
    batch_weights = np.tile(weights, batch_tokens).astype(np.float32)
    batch_shared = np.vstack([shared[0] + np.float32(0.125 * token) for token in range(batch_tokens)]).astype(np.float32)
    batch_residual = np.vstack([residual[0] - np.float32(0.0625 * token) for token in range(batch_tokens)]).astype(np.float32)
    batch_gate_stride = 3
    batch_gate_logits = np.asarray([[0.25 * token, -0.125, 0.5 - 0.25 * token] for token in range(batch_tokens)], dtype=np.float32)
    batch_values_bits = _float32_to_bf16_bits(batch_values)
    batch_shared_bits = _float32_to_bf16_bits(batch_shared)
    batch_residual_bits = _float32_to_bf16_bits(batch_residual)
    batch_weighted_shared_residual_bits = np.empty((batch_tokens, features), dtype=np.uint16)
    batch_values_bf32 = _bf16_bits_to_float32(batch_values_bits).reshape(batch_tokens, batch_rows_per_token, features)
    batch_shared_bf32 = _bf16_bits_to_float32(batch_shared_bits)
    batch_residual_bf32 = _bf16_bits_to_float32(batch_residual_bits)
    expected_batch = np.empty((batch_tokens, features), dtype=np.float32)
    for token in range(batch_tokens):
        selected_acc = np.sum(
            batch_values_bf32[token] * batch_weights[token * rows : (token + 1) * rows].reshape(rows, 1),
            axis=0,
            dtype=np.float32,
        ).reshape(1, features)
        selected_bits = _float32_to_bf16_bits(selected_acc)
        selected = _bf16_bits_to_float32(selected_bits)[0]
        gate_t = np.float32(1.0) / (np.float32(1.0) + np.exp(-batch_gate_logits[token, 2], dtype=np.float32))
        expected_batch[token] = batch_residual_bf32[token] + selected + gate_t * batch_shared_bf32[token]
    expected_batch_bits = _float32_to_bf16_bits(expected_batch)

    runtime = get_hip_runtime()
    library = build_paro_combine(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        values_dev = dev(values_bits)
        weights_dev = dev(weights)
        expert_dev = dev(expert_bits)
        shared_dev = dev(shared_bits)
        residual_dev = dev(residual_bits)
        gate_logits_dev = dev(gate_logits)
        batch_values_dev = dev(batch_values_bits)
        batch_weights_dev = dev(batch_weights)
        batch_shared_dev = dev(batch_shared_bits)
        batch_residual_dev = dev(batch_residual_bits)
        batch_gate_logits_dev = dev(batch_gate_logits)
        weighted_dev = out_dev(weighted_bits)
        weighted_shared_residual_dev = out_dev(weighted_shared_residual_bits)
        batch_weighted_shared_residual_dev = out_dev(batch_weighted_shared_residual_bits)
        shared_combine_dev = out_dev(shared_combine_bits)
        shared_residual_dev = out_dev(shared_residual_bits)
        weighted_sum_out_bf16_f32w(
            values_dev.ptr,
            weights_dev.ptr,
            weighted_dev.ptr,
            rows,
            features,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        weighted_sum_shared_gate_combine_residual_out_bf16_f32w(
            values_dev.ptr,
            weights_dev.ptr,
            shared_dev.ptr,
            gate_logits_dev.ptr,
            residual_dev.ptr,
            weighted_shared_residual_dev.ptr,
            rows,
            features,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w(
            batch_values_dev.ptr,
            batch_weights_dev.ptr,
            batch_shared_dev.ptr,
            batch_gate_logits_dev.ptr + 2 * batch_gate_logits.itemsize,
            batch_residual_dev.ptr,
            batch_weighted_shared_residual_dev.ptr,
            batch_tokens,
            batch_rows_per_token,
            features,
            batch_gate_stride,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        shared_gate_combine_out_bf16(
            expert_dev.ptr,
            shared_dev.ptr,
            gate_logits_dev.ptr,
            shared_combine_dev.ptr,
            features,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        shared_gate_combine_residual_out_bf16(
            expert_dev.ptr,
            shared_dev.ptr,
            gate_logits_dev.ptr,
            residual_dev.ptr,
            shared_residual_dev.ptr,
            features,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(weighted_bits), weighted_dev, runtime=runtime)
        copy_device_to_host(
            host_array_ptr(weighted_shared_residual_bits),
            weighted_shared_residual_dev,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(batch_weighted_shared_residual_bits),
            batch_weighted_shared_residual_dev,
            runtime=runtime,
        )
        copy_device_to_host(host_array_ptr(shared_combine_bits), shared_combine_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(shared_residual_bits), shared_residual_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    weighted_mismatch = int(np.count_nonzero(weighted_bits != expected_weighted_bits))
    fused_mismatch = int(
        np.count_nonzero(weighted_shared_residual_bits != expected_weighted_shared_residual_bits)
    )
    batch_fused_mismatch = int(
        np.count_nonzero(batch_weighted_shared_residual_bits != expected_batch_bits)
    )
    shared_mismatch = int(np.count_nonzero(shared_combine_bits != expected_shared_combine_bits))
    shared_residual_mismatch = int(
        np.count_nonzero(shared_residual_bits != expected_shared_residual_bits)
    )
    weighted_max_abs = float(
        np.max(np.abs(_bf16_bits_to_float32(weighted_bits) - _bf16_bits_to_float32(expected_weighted_bits)))
    )
    fused_max_abs = float(
        np.max(
            np.abs(
                _bf16_bits_to_float32(weighted_shared_residual_bits)
                - _bf16_bits_to_float32(expected_weighted_shared_residual_bits)
            )
        )
    )
    batch_fused_max_abs = float(
        np.max(
            np.abs(
                _bf16_bits_to_float32(batch_weighted_shared_residual_bits)
                - _bf16_bits_to_float32(expected_batch_bits)
            )
        )
    )
    shared_max_abs = float(
        np.max(
            np.abs(
                _bf16_bits_to_float32(shared_combine_bits)
                - _bf16_bits_to_float32(expected_shared_combine_bits)
            )
        )
    )
    shared_residual_max_abs = float(
        np.max(
            np.abs(
                _bf16_bits_to_float32(shared_residual_bits)
                - _bf16_bits_to_float32(expected_shared_residual_bits)
            )
        )
    )
    print(
        f"rows={rows} hidden_size={hidden_size} "
        f"weighted_mismatch={weighted_mismatch} weighted_max_abs={weighted_max_abs} "
        f"fused_mismatch={fused_mismatch} fused_max_abs={fused_max_abs} "
        f"batch_fused_mismatch={batch_fused_mismatch} batch_fused_max_abs={batch_fused_max_abs} "
        f"shared_mismatch={shared_mismatch} shared_max_abs={shared_max_abs} "
        f"shared_residual_mismatch={shared_residual_mismatch} "
        f"shared_residual_max_abs={shared_residual_max_abs}"
    )
    print("weighted=", _bf16_bits_to_float32(weighted_bits)[0, : min(8, features)].tolist())
    print("fused=", _bf16_bits_to_float32(weighted_shared_residual_bits)[0, : min(8, features)].tolist())
    return 0 if (
        weighted_mismatch == 0
        and fused_mismatch == 0
        and batch_fused_mismatch == 0
        and shared_mismatch == 0
        and shared_residual_mismatch == 0
    ) else 1


def paro_silu_hip_smoke(
    rows: int,
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.fused import (
        build_paro_silu,
        silu_mul_dual_out_bf16,
        silu_mul_dual_rotate_out_bf16,
        silu_mul_pair_rotate_out_bf16,
    )

    if rows < 1:
        raise ValueError("--rows must be >= 1")
    if hidden_size < 2 or hidden_size % 2 != 0:
        raise ValueError("--hidden-size must be >= 2 and even")

    features = hidden_size
    group_size = hidden_size
    krot = 1
    gate = np.linspace(-1.0, 1.0, rows * features, dtype=np.float32).reshape(rows, features)
    up = np.linspace(0.75, -0.5, rows * features, dtype=np.float32).reshape(rows, features)
    gate_up = np.concatenate([gate, up], axis=1)
    scales = np.asarray([0.25, 0.5, 1.0, 0.125] * ((features + 3) // 4), dtype=np.float32)[:features]
    pairs = np.empty((krot, features), dtype=np.int16)
    half_group = group_size // 2
    for lane in range(half_group):
        pairs[0, 2 * lane] = lane
        pairs[0, 2 * lane + 1] = lane + half_group
    theta = np.zeros((krot, features // 2), dtype=np.float32)

    gate_bits = _float32_to_bf16_bits(gate)
    up_bits = _float32_to_bf16_bits(up)
    gate_up_bits = _float32_to_bf16_bits(gate_up)
    scales_bits = _float32_to_bf16_bits(scales)
    theta_bits = _float32_to_bf16_bits(theta)
    dual_out_bits = np.empty((rows, features), dtype=np.uint16)
    dual_rotate_bits = np.empty_like(dual_out_bits)
    pair_rotate_bits = np.empty_like(dual_out_bits)

    gate_bf32 = _bf16_bits_to_float32(gate_bits)
    up_bf32 = _bf16_bits_to_float32(up_bits)
    scales_bf32 = _bf16_bits_to_float32(scales_bits)
    act = gate_bf32 * (1.0 / (1.0 + np.exp(-gate_bf32, dtype=np.float32))) * up_bf32
    expected_dual_bits = _float32_to_bf16_bits(act)
    rounded_act = _bf16_bits_to_float32(expected_dual_bits)
    expected_rotate_bits = _float32_to_bf16_bits(rounded_act * scales_bf32.reshape(1, features))

    runtime = get_hip_runtime()
    library = build_paro_silu(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        gate_up_dev = dev(gate_up_bits)
        gate_dev = dev(gate_bits)
        up_dev = dev(up_bits)
        pairs_dev = dev(pairs)
        theta_dev = dev(theta_bits)
        scales_dev = dev(scales_bits)
        dual_out_dev = out_dev(dual_out_bits)
        dual_rotate_dev = out_dev(dual_rotate_bits)
        pair_rotate_dev = out_dev(pair_rotate_bits)
        silu_mul_dual_out_bf16(
            gate_up_dev.ptr,
            dual_out_dev.ptr,
            rows,
            features,
            threads=256,
            library=library,
            runtime=runtime,
        )
        silu_mul_dual_rotate_out_bf16(
            gate_up_dev.ptr,
            pairs_dev.ptr,
            theta_dev.ptr,
            scales_dev.ptr,
            dual_rotate_dev.ptr,
            rows,
            features,
            group_size,
            krot,
            library=library,
            runtime=runtime,
        )
        silu_mul_pair_rotate_out_bf16(
            gate_dev.ptr,
            up_dev.ptr,
            pairs_dev.ptr,
            theta_dev.ptr,
            scales_dev.ptr,
            pair_rotate_dev.ptr,
            rows,
            features,
            group_size,
            krot,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(dual_out_bits), dual_out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(dual_rotate_bits), dual_rotate_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(pair_rotate_bits), pair_rotate_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    dual_out = _bf16_bits_to_float32(dual_out_bits)
    dual_rotate = _bf16_bits_to_float32(dual_rotate_bits)
    pair_rotate = _bf16_bits_to_float32(pair_rotate_bits)
    expected_dual = _bf16_bits_to_float32(expected_dual_bits)
    expected_rotate = _bf16_bits_to_float32(expected_rotate_bits)
    dual_max_abs = float(np.max(np.abs(dual_out - expected_dual)))
    dual_rotate_max_abs = float(np.max(np.abs(dual_rotate - expected_rotate)))
    pair_rotate_max_abs = float(np.max(np.abs(pair_rotate - expected_rotate)))
    dual_mismatch = int(np.count_nonzero(dual_out_bits != expected_dual_bits))
    dual_rotate_mismatch = int(np.count_nonzero(dual_rotate_bits != expected_rotate_bits))
    pair_rotate_mismatch = int(np.count_nonzero(pair_rotate_bits != expected_rotate_bits))
    print(
        f"rows={rows} hidden_size={hidden_size} "
        f"dual_max_abs={dual_max_abs} dual_mismatch={dual_mismatch} "
        f"dual_rotate_max_abs={dual_rotate_max_abs} dual_rotate_mismatch={dual_rotate_mismatch} "
        f"pair_rotate_max_abs={pair_rotate_max_abs} pair_rotate_mismatch={pair_rotate_mismatch}"
    )
    print("dual0=", dual_out[0, : min(8, features)].tolist())
    print("rotate0=", dual_rotate[0, : min(8, features)].tolist())
    return 0 if (
        dual_max_abs <= 2e-2
        and dual_rotate_max_abs <= 2e-2
        and pair_rotate_max_abs <= 2e-2
    ) else 1



def paro_pack8_gemv_hip_smoke(
    rows: int,
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant import (
        build_paro_awq_gemv,
        gemv_awq_dual_pack8_strided_bf16,
        gemv_awq_dual_pack8_transposed_bf16,
        gemv_awq_pack8_strided_bf16,
        gemv_awq_pack8_transposed_bf16,
    )

    if rows < 1:
        raise ValueError("--rows must be >= 1")
    if hidden_size < 8 or hidden_size % 8 != 0:
        raise ValueError("--hidden-size must be >= 8 and divisible by 8")

    group_size = 8
    threads = 64
    out_packed_a = out_packed_b = out_packed = 1
    selected = np.zeros(rows, dtype=np.int64)
    x_f32 = np.empty((rows, hidden_size), dtype=np.float32)
    x_b_f32 = np.empty_like(x_f32)
    for row in range(rows):
        x_f32[row] = np.asarray(
            [[-0.5, -0.25, 0.25, 0.5][(row + col) % 4] for col in range(hidden_size)],
            dtype=np.float32,
        )
        x_b_f32[row] = -x_f32[row]
    x_bits = _float32_to_bf16_bits(x_f32)
    x_b_bits = _float32_to_bf16_bits(x_b_f32)

    qweight_a_3d, qzeros_a_3d, scales_a_3d_bits = _make_pack8_fixture(
        1, hidden_size, out_packed_a, group_size, salt=11
    )
    qweight_b_3d, qzeros_b_3d, scales_b_3d_bits = _make_pack8_fixture(
        1, hidden_size, out_packed_b, group_size, salt=13
    )
    qweight_single_3d, qzeros_single_3d, scales_single_3d_bits = _make_pack8_fixture(
        1, hidden_size, out_packed, group_size, salt=17
    )
    qweight_a = qweight_a_3d[0].copy()
    qzeros_a = qzeros_a_3d[0].copy()
    scales_a_bits = scales_a_3d_bits[0].copy()
    qweight_b = qweight_b_3d[0].copy()
    qzeros_b = qzeros_b_3d[0].copy()
    scales_b_bits = scales_b_3d_bits[0].copy()
    qweight_single = qweight_single_3d[0].copy()
    qzeros_single = qzeros_single_3d[0].copy()
    scales_single_bits = scales_single_3d_bits[0].copy()
    qweight_a_t = np.transpose(qweight_a).copy()
    qweight_b_t = np.transpose(qweight_b).copy()
    qweight_single_t = np.transpose(qweight_single).copy()

    single_strided_bits = np.empty((rows, out_packed * 8), dtype=np.uint16)
    single_transposed_bits = np.empty_like(single_strided_bits)
    dual_strided_bits = np.empty((rows, (out_packed_a + out_packed_b) * 8), dtype=np.uint16)
    dual_transposed_bits = np.empty_like(dual_strided_bits)

    expected_single_bits = _selected_pack8_reference(
        x_bits,
        selected,
        qweight_single_3d,
        qzeros_single_3d,
        scales_single_3d_bits,
        group_size,
        qweight_transposed=False,
    )
    expected_dual_a_bits = _selected_pack8_reference(
        x_bits,
        selected,
        qweight_a_3d,
        qzeros_a_3d,
        scales_a_3d_bits,
        group_size,
        qweight_transposed=False,
    )
    expected_dual_b_strided_bits = _selected_pack8_reference(
        x_bits,
        selected,
        qweight_b_3d,
        qzeros_b_3d,
        scales_b_3d_bits,
        group_size,
        qweight_transposed=False,
    )
    expected_dual_b_transposed_bits = _selected_pack8_reference(
        x_b_bits,
        selected,
        np.transpose(qweight_b_3d, (0, 2, 1)).copy(),
        qzeros_b_3d,
        scales_b_3d_bits,
        group_size,
        qweight_transposed=True,
    )
    expected_dual_strided_bits = np.concatenate(
        [expected_dual_a_bits, expected_dual_b_strided_bits], axis=1
    )
    expected_dual_transposed_bits = np.concatenate(
        [expected_dual_a_bits, expected_dual_b_transposed_bits], axis=1
    )

    runtime = get_hip_runtime()
    library = build_paro_awq_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        x_dev = dev(x_bits)
        x_b_dev = dev(x_b_bits)
        qweight_single_dev = dev(qweight_single)
        qweight_single_t_dev = dev(qweight_single_t)
        qzeros_single_dev = dev(qzeros_single)
        scales_single_dev = dev(scales_single_bits)
        qweight_a_dev = dev(qweight_a)
        qweight_a_t_dev = dev(qweight_a_t)
        qzeros_a_dev = dev(qzeros_a)
        scales_a_dev = dev(scales_a_bits)
        qweight_b_dev = dev(qweight_b)
        qweight_b_t_dev = dev(qweight_b_t)
        qzeros_b_dev = dev(qzeros_b)
        scales_b_dev = dev(scales_b_bits)
        single_strided_dev = out_dev(single_strided_bits)
        single_transposed_dev = out_dev(single_transposed_bits)
        dual_strided_dev = out_dev(dual_strided_bits)
        dual_transposed_dev = out_dev(dual_transposed_bits)
        gemv_awq_pack8_strided_bf16(
            x_dev.ptr,
            qweight_single_dev.ptr,
            qzeros_single_dev.ptr,
            scales_single_dev.ptr,
            single_strided_dev.ptr,
            rows,
            hidden_size,
            out_packed,
            group_size,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        gemv_awq_pack8_transposed_bf16(
            x_dev.ptr,
            qweight_single_t_dev.ptr,
            qzeros_single_dev.ptr,
            scales_single_dev.ptr,
            single_transposed_dev.ptr,
            rows,
            hidden_size,
            out_packed,
            group_size,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        gemv_awq_dual_pack8_strided_bf16(
            x_dev.ptr,
            qweight_a_dev.ptr,
            qzeros_a_dev.ptr,
            scales_a_dev.ptr,
            qweight_b_dev.ptr,
            qzeros_b_dev.ptr,
            scales_b_dev.ptr,
            dual_strided_dev.ptr,
            rows,
            hidden_size,
            out_packed_a,
            out_packed_b,
            group_size,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        gemv_awq_dual_pack8_transposed_bf16(
            x_dev.ptr,
            x_b_dev.ptr,
            qweight_a_t_dev.ptr,
            qzeros_a_dev.ptr,
            scales_a_dev.ptr,
            qweight_b_t_dev.ptr,
            qzeros_b_dev.ptr,
            scales_b_dev.ptr,
            dual_transposed_dev.ptr,
            rows,
            hidden_size,
            out_packed_a,
            out_packed_b,
            group_size,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(single_strided_bits), single_strided_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(single_transposed_bits), single_transposed_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(dual_strided_bits), dual_strided_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(dual_transposed_bits), dual_transposed_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    single_strided_mismatch = int(np.count_nonzero(single_strided_bits != expected_single_bits))
    single_transposed_mismatch = int(np.count_nonzero(single_transposed_bits != expected_single_bits))
    dual_strided_mismatch = int(np.count_nonzero(dual_strided_bits != expected_dual_strided_bits))
    dual_transposed_mismatch = int(
        np.count_nonzero(dual_transposed_bits != expected_dual_transposed_bits)
    )
    max_abs = float(
        max(
            np.max(np.abs(_bf16_bits_to_float32(single_strided_bits) - _bf16_bits_to_float32(expected_single_bits))),
            np.max(np.abs(_bf16_bits_to_float32(single_transposed_bits) - _bf16_bits_to_float32(expected_single_bits))),
            np.max(np.abs(_bf16_bits_to_float32(dual_strided_bits) - _bf16_bits_to_float32(expected_dual_strided_bits))),
            np.max(np.abs(_bf16_bits_to_float32(dual_transposed_bits) - _bf16_bits_to_float32(expected_dual_transposed_bits))),
        )
    )
    print(
        f"rows={rows} hidden_size={hidden_size} threads={threads} "
        f"single_mismatch={single_strided_mismatch}/{single_transposed_mismatch} "
        f"dual_mismatch={dual_strided_mismatch}/{dual_transposed_mismatch} "
        f"max_abs={max_abs}"
    )
    print("generic_single0=", _bf16_bits_to_float32(single_strided_bits)[0].tolist())
    print("generic_dual0=", _bf16_bits_to_float32(dual_strided_bits)[0].tolist())
    return (
        0
        if single_strided_mismatch == 0
        and single_transposed_mismatch == 0
        and dual_strided_mismatch == 0
        and dual_transposed_mismatch == 0
        else 1
    )



def paro_rotate_hip_smoke(
    rows: int,
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.rotary import (
        build_paro_rotate,
        paro_rotate1_fp16,
        paro_rotate2_bf16,
        paro_rotate2_fp16,
        paro_rotate3_bf16,
        paro_rotate3_fp16,
    )

    if rows < 1:
        raise ValueError("--rows must be >= 1")
    if hidden_size < 8 or hidden_size % 8 != 0:
        raise ValueError("--hidden-size must be >= 8 and divisible by 8")

    group_size = 8
    krot = 1
    x_f32 = np.empty((rows, hidden_size), dtype=np.float32)
    for row in range(rows):
        x_f32[row] = np.asarray(
            [[-0.5, -0.25, 0.25, 0.5][(row + col) % 4] for col in range(hidden_size)],
            dtype=np.float32,
        )
    x_bits = _float32_to_bf16_bits(x_f32)
    pairs = np.empty((krot, hidden_size), dtype=np.int16)
    half_group = group_size // 2
    for group in range(hidden_size // group_size):
        base = group * group_size
        for lane in range(half_group):
            pairs[0, base + 2 * lane] = lane
            pairs[0, base + 2 * lane + 1] = lane + half_group
    theta_bits = _float32_to_bf16_bits(np.zeros((krot, hidden_size // 2), dtype=np.float32))
    pattern = [1.0, 0.5, 0.25, 2.0, 1.0, 0.5, 0.25, 2.0]
    scale_sets = [
        np.asarray([pattern[(i + salt) % 8] for i in range(hidden_size)], dtype=np.float32)
        for salt in range(3)
    ]
    scale_bits = [_float32_to_bf16_bits(values) for values in scale_sets]
    expected = [_float32_to_bf16_bits(x_f32 * values.reshape(1, hidden_size)) for values in scale_sets]
    x_fp16 = x_f32.astype(np.float16)
    theta_fp16 = np.zeros((krot, hidden_size // 2), dtype=np.float16)
    scale_fp16 = [values.astype(np.float16) for values in scale_sets]
    expected_fp16 = [
        (x_fp16.astype(np.float32) * values.astype(np.float32).reshape(1, hidden_size)).astype(
            np.float16
        )
        for values in scale_fp16
    ]
    rotate1_fp16 = np.empty_like(x_fp16)
    rotate2_fp16_out0 = np.empty_like(x_fp16)
    rotate2_fp16_out1 = np.empty_like(x_fp16)
    rotate3_fp16_out0 = np.empty_like(x_fp16)
    rotate3_fp16_out1 = np.empty_like(x_fp16)
    rotate3_fp16_out2 = np.empty_like(x_fp16)
    rotate2_out0 = np.empty_like(x_bits)
    rotate2_out1 = np.empty_like(x_bits)
    rotate3_out0 = np.empty_like(x_bits)
    rotate3_out1 = np.empty_like(x_bits)
    rotate3_out2 = np.empty_like(x_bits)

    runtime = get_hip_runtime()
    library = build_paro_rotate(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        x_dev = dev(x_bits)
        pairs_dev = dev(pairs)
        theta_dev = dev(theta_bits)
        scales_dev = [dev(bits) for bits in scale_bits]
        r2o0_dev = out_dev(rotate2_out0)
        r2o1_dev = out_dev(rotate2_out1)
        r3o0_dev = out_dev(rotate3_out0)
        r3o1_dev = out_dev(rotate3_out1)
        r3o2_dev = out_dev(rotate3_out2)
        x_fp16_dev = dev(x_fp16)
        theta_fp16_dev = dev(theta_fp16)
        scale_fp16_dev = [dev(values) for values in scale_fp16]
        r1_fp16_dev = out_dev(rotate1_fp16)
        r2_fp16_o0_dev = out_dev(rotate2_fp16_out0)
        r2_fp16_o1_dev = out_dev(rotate2_fp16_out1)
        r3_fp16_o0_dev = out_dev(rotate3_fp16_out0)
        r3_fp16_o1_dev = out_dev(rotate3_fp16_out1)
        r3_fp16_o2_dev = out_dev(rotate3_fp16_out2)
        paro_rotate2_bf16(
            x_dev.ptr,
            r2o0_dev.ptr,
            r2o1_dev.ptr,
            pairs_dev.ptr,
            pairs_dev.ptr,
            theta_dev.ptr,
            theta_dev.ptr,
            scales_dev[0].ptr,
            scales_dev[1].ptr,
            rows,
            hidden_size,
            group_size,
            krot,
            library=library,
            runtime=runtime,
        )
        paro_rotate3_bf16(
            x_dev.ptr,
            r3o0_dev.ptr,
            r3o1_dev.ptr,
            r3o2_dev.ptr,
            pairs_dev.ptr,
            pairs_dev.ptr,
            pairs_dev.ptr,
            theta_dev.ptr,
            theta_dev.ptr,
            theta_dev.ptr,
            scales_dev[0].ptr,
            scales_dev[1].ptr,
            scales_dev[2].ptr,
            rows,
            hidden_size,
            group_size,
            krot,
            library=library,
            runtime=runtime,
        )
        paro_rotate1_fp16(
            x_fp16_dev.ptr,
            r1_fp16_dev.ptr,
            pairs_dev.ptr,
            theta_fp16_dev.ptr,
            scale_fp16_dev[0].ptr,
            rows,
            hidden_size,
            group_size,
            krot,
            library=library,
            runtime=runtime,
        )
        paro_rotate2_fp16(
            x_fp16_dev.ptr,
            r2_fp16_o0_dev.ptr,
            r2_fp16_o1_dev.ptr,
            pairs_dev.ptr,
            pairs_dev.ptr,
            theta_fp16_dev.ptr,
            theta_fp16_dev.ptr,
            scale_fp16_dev[0].ptr,
            scale_fp16_dev[1].ptr,
            rows,
            hidden_size,
            group_size,
            krot,
            library=library,
            runtime=runtime,
        )
        paro_rotate3_fp16(
            x_fp16_dev.ptr,
            r3_fp16_o0_dev.ptr,
            r3_fp16_o1_dev.ptr,
            r3_fp16_o2_dev.ptr,
            pairs_dev.ptr,
            pairs_dev.ptr,
            pairs_dev.ptr,
            theta_fp16_dev.ptr,
            theta_fp16_dev.ptr,
            theta_fp16_dev.ptr,
            scale_fp16_dev[0].ptr,
            scale_fp16_dev[1].ptr,
            scale_fp16_dev[2].ptr,
            rows,
            hidden_size,
            group_size,
            krot,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(rotate2_out0), r2o0_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(rotate2_out1), r2o1_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(rotate3_out0), r3o0_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(rotate3_out1), r3o1_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(rotate3_out2), r3o2_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(rotate1_fp16), r1_fp16_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(rotate2_fp16_out0), r2_fp16_o0_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(rotate2_fp16_out1), r2_fp16_o1_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(rotate3_fp16_out0), r3_fp16_o0_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(rotate3_fp16_out1), r3_fp16_o1_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(rotate3_fp16_out2), r3_fp16_o2_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    outputs = [rotate2_out0, rotate2_out1, rotate3_out0, rotate3_out1, rotate3_out2]
    expect = [expected[0], expected[1], expected[0], expected[1], expected[2]]
    mismatches = [int(np.count_nonzero(out != exp)) for out, exp in zip(outputs, expect, strict=True)]
    max_abs = float(
        max(
            np.max(np.abs(_bf16_bits_to_float32(out) - _bf16_bits_to_float32(exp)))
            for out, exp in zip(outputs, expect, strict=True)
        )
    )
    fp16_outputs = [
        rotate1_fp16,
        rotate2_fp16_out0,
        rotate2_fp16_out1,
        rotate3_fp16_out0,
        rotate3_fp16_out1,
        rotate3_fp16_out2,
    ]
    fp16_expect = [
        expected_fp16[0],
        expected_fp16[0],
        expected_fp16[1],
        expected_fp16[0],
        expected_fp16[1],
        expected_fp16[2],
    ]
    fp16_mismatches = [
        int(np.count_nonzero(out.view(np.uint16) != exp.view(np.uint16)))
        for out, exp in zip(fp16_outputs, fp16_expect, strict=True)
    ]
    fp16_max_abs = float(
        max(
            np.max(np.abs(out.astype(np.float32) - exp.astype(np.float32)))
            for out, exp in zip(fp16_outputs, fp16_expect, strict=True)
        )
    )
    print(
        f"rows={rows} hidden_size={hidden_size} group_size={group_size} krot={krot} "
        f"mismatches={mismatches} max_abs={max_abs} "
        f"fp16_mismatches={fp16_mismatches} fp16_max_abs={fp16_max_abs}"
    )
    print("rotate2_0=", _bf16_bits_to_float32(rotate2_out0)[0].tolist())
    print("rotate3_2=", _bf16_bits_to_float32(rotate3_out2)[0].tolist())
    print("rotate1_fp16=", rotate1_fp16[0].astype(np.float32).tolist())
    return 0 if max(mismatches) == 0 and max(fp16_mismatches) == 0 else 1

def paro_selected_gemv_rotate_hip_smoke(
    rows: int,
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant import (
        build_paro_awq_gemv,
        gemv_awq_selected_dual_pack8_strided_rotate_out_bf16,
    )

    if rows < 1:
        raise ValueError("--rows must be >= 1")
    if hidden_size < 8 or hidden_size % 8 != 0:
        raise ValueError("--hidden-size must be >= 8 and divisible by 8")

    group_size = 8
    threads = 64
    num_experts = 2
    out_packed_a = out_packed_b = 1
    krot = 1
    selected = np.arange(rows, dtype=np.int64) % num_experts
    x_f32 = np.empty((rows, hidden_size), dtype=np.float32)
    for row in range(rows):
        x_f32[row] = np.asarray(
            [[-0.5, -0.25, 0.25, 0.5][(row + col) % 4] for col in range(hidden_size)],
            dtype=np.float32,
        )
    x_bits = _float32_to_bf16_bits(x_f32)
    channel_scales_f32 = np.asarray(
        [1.0, 0.5, 0.25, 2.0, 1.0, 0.5, 0.25, 2.0] * (hidden_size // 8),
        dtype=np.float32,
    )
    channel_scales_bits = _float32_to_bf16_bits(channel_scales_f32)
    pairs = np.empty((krot, hidden_size), dtype=np.int16)
    theta = np.zeros((krot, hidden_size // 2), dtype=np.float32)
    half_group = group_size // 2
    for group in range(hidden_size // group_size):
        base = group * group_size
        for lane in range(half_group):
            pairs[0, base + 2 * lane] = lane
            pairs[0, base + 2 * lane + 1] = lane + half_group
    theta_bits = _float32_to_bf16_bits(theta)

    qweight_a, qzeros_a, scales_a_bits = _make_pack8_fixture(
        num_experts, hidden_size, out_packed_a, group_size, salt=23
    )
    qweight_b, qzeros_b, scales_b_bits = _make_pack8_fixture(
        num_experts, hidden_size, out_packed_b, group_size, salt=29
    )
    out_bits = np.empty((rows, (out_packed_a + out_packed_b) * 8), dtype=np.uint16)

    rotated = _bf16_bits_to_float32(x_bits) * _bf16_bits_to_float32(channel_scales_bits).reshape(
        1, hidden_size
    )
    rotated_bits = _float32_to_bf16_bits(rotated)
    expected_a = _selected_pack8_reference(
        rotated_bits,
        selected,
        qweight_a,
        qzeros_a,
        scales_a_bits,
        group_size,
        qweight_transposed=False,
    )
    expected_b = _selected_pack8_reference(
        rotated_bits,
        selected,
        qweight_b,
        qzeros_b,
        scales_b_bits,
        group_size,
        qweight_transposed=False,
    )
    expected_bits = np.concatenate([expected_a, expected_b], axis=1)

    runtime = get_hip_runtime()
    library = build_paro_awq_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        x_dev = dev(x_bits)
        selected_dev = dev(selected)
        pairs_dev = dev(pairs)
        theta_dev = dev(theta_bits)
        channel_scales_dev = dev(channel_scales_bits)
        qweight_a_dev = dev(qweight_a)
        qzeros_a_dev = dev(qzeros_a)
        scales_a_dev = dev(scales_a_bits)
        qweight_b_dev = dev(qweight_b)
        qzeros_b_dev = dev(qzeros_b)
        scales_b_dev = dev(scales_b_bits)
        out_dev_buf = out_dev(out_bits)
        gemv_awq_selected_dual_pack8_strided_rotate_out_bf16(
            x_dev.ptr,
            selected_dev.ptr,
            pairs_dev.ptr,
            theta_dev.ptr,
            channel_scales_dev.ptr,
            qweight_a_dev.ptr,
            qzeros_a_dev.ptr,
            scales_a_dev.ptr,
            qweight_b_dev.ptr,
            qzeros_b_dev.ptr,
            scales_b_dev.ptr,
            out_dev_buf.ptr,
            rows,
            rows,
            hidden_size,
            out_packed_a,
            out_packed_b,
            num_experts,
            group_size,
            krot,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_bits), out_dev_buf, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    mismatch = int(np.count_nonzero(out_bits != expected_bits))
    max_abs = float(np.max(np.abs(_bf16_bits_to_float32(out_bits) - _bf16_bits_to_float32(expected_bits))))
    print(
        f"rows={rows} hidden_size={hidden_size} threads={threads} krot={krot} "
        f"mismatch={mismatch} max_abs={max_abs}"
    )
    print("rotate_selected0=", _bf16_bits_to_float32(out_bits)[0].tolist())
    return 0 if mismatch == 0 else 1

def paro_selected_gemv_hip_smoke(
    rows: int,
    hidden_size: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant import (
        build_paro_awq_gemv,
        gemv_awq_selected_dual_pack8_strided_bf16,
        gemv_awq_selected_dual_pack8_transposed_bf16,
        gemv_awq_selected_pack8_strided_bf16,
        gemv_awq_selected_pack8_transposed_bf16,
    )

    if rows < 1:
        raise ValueError("--rows must be >= 1")
    if hidden_size < 8 or hidden_size % 8 != 0:
        raise ValueError("--hidden-size must be >= 8 and divisible by 8")

    group_size = 8
    threads = 64
    num_experts = 3
    out_packed_a = out_packed_b = out_packed = 1
    selected = (np.arange(rows, dtype=np.int64) % num_experts).astype(np.int64)
    x_dual_f32 = np.array(
        [[[-0.5, -0.25, 0.25, 0.5][i % 4] for i in range(hidden_size)]],
        dtype=np.float32,
    )
    x_single_f32 = np.empty((rows, hidden_size), dtype=np.float32)
    for row in range(rows):
        x_single_f32[row] = x_dual_f32[0] * (1.0 if (row % 2) == 0 else -1.0)
    x_dual_bits = _float32_to_bf16_bits(x_dual_f32)
    x_single_bits = _float32_to_bf16_bits(x_single_f32)

    qweight_a, qzeros_a, scales_a_bits = _make_pack8_fixture(
        num_experts, hidden_size, out_packed_a, group_size, salt=0
    )
    qweight_b, qzeros_b, scales_b_bits = _make_pack8_fixture(
        num_experts, hidden_size, out_packed_b, group_size, salt=2
    )
    qweight_single, qzeros_single, scales_single_bits = _make_pack8_fixture(
        num_experts, hidden_size, out_packed, group_size, salt=4
    )
    qweight_a_t = np.transpose(qweight_a, (0, 2, 1)).copy()
    qweight_b_t = np.transpose(qweight_b, (0, 2, 1)).copy()
    qweight_single_t = np.transpose(qweight_single, (0, 2, 1)).copy()

    dual_strided_bits = np.empty((rows, (out_packed_a + out_packed_b) * 8), dtype=np.uint16)
    dual_transposed_bits = np.empty_like(dual_strided_bits)
    single_strided_bits = np.empty((rows, out_packed * 8), dtype=np.uint16)
    single_transposed_bits = np.empty_like(single_strided_bits)

    expected_dual_a = _selected_pack8_reference(
        x_dual_bits,
        selected,
        qweight_a,
        qzeros_a,
        scales_a_bits,
        group_size,
        qweight_transposed=False,
    )
    expected_dual_b = _selected_pack8_reference(
        x_dual_bits,
        selected,
        qweight_b,
        qzeros_b,
        scales_b_bits,
        group_size,
        qweight_transposed=False,
    )
    expected_dual_bits = np.concatenate([expected_dual_a, expected_dual_b], axis=1)
    expected_single_bits = _selected_pack8_reference(
        x_single_bits,
        selected,
        qweight_single,
        qzeros_single,
        scales_single_bits,
        group_size,
        qweight_transposed=False,
    )

    runtime = get_hip_runtime()
    library = build_paro_awq_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        return buffer

    def out_dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        x_dual_dev = dev(x_dual_bits)
        x_single_dev = dev(x_single_bits)
        selected_dev = dev(selected)
        qweight_a_dev = dev(qweight_a)
        qweight_a_t_dev = dev(qweight_a_t)
        qzeros_a_dev = dev(qzeros_a)
        scales_a_dev = dev(scales_a_bits)
        qweight_b_dev = dev(qweight_b)
        qweight_b_t_dev = dev(qweight_b_t)
        qzeros_b_dev = dev(qzeros_b)
        scales_b_dev = dev(scales_b_bits)
        qweight_single_dev = dev(qweight_single)
        qweight_single_t_dev = dev(qweight_single_t)
        qzeros_single_dev = dev(qzeros_single)
        scales_single_dev = dev(scales_single_bits)
        dual_strided_dev = out_dev(dual_strided_bits)
        dual_transposed_dev = out_dev(dual_transposed_bits)
        single_strided_dev = out_dev(single_strided_bits)
        single_transposed_dev = out_dev(single_transposed_bits)

        gemv_awq_selected_dual_pack8_strided_bf16(
            x_dual_dev.ptr,
            selected_dev.ptr,
            qweight_a_dev.ptr,
            qzeros_a_dev.ptr,
            scales_a_dev.ptr,
            qweight_b_dev.ptr,
            qzeros_b_dev.ptr,
            scales_b_dev.ptr,
            dual_strided_dev.ptr,
            1,
            rows,
            hidden_size,
            out_packed_a,
            out_packed_b,
            num_experts,
            group_size,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        gemv_awq_selected_dual_pack8_transposed_bf16(
            x_dual_dev.ptr,
            selected_dev.ptr,
            qweight_a_t_dev.ptr,
            qzeros_a_dev.ptr,
            scales_a_dev.ptr,
            qweight_b_t_dev.ptr,
            qzeros_b_dev.ptr,
            scales_b_dev.ptr,
            dual_transposed_dev.ptr,
            1,
            rows,
            hidden_size,
            out_packed_a,
            out_packed_b,
            num_experts,
            group_size,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        gemv_awq_selected_pack8_strided_bf16(
            x_single_dev.ptr,
            selected_dev.ptr,
            qweight_single_dev.ptr,
            qzeros_single_dev.ptr,
            scales_single_dev.ptr,
            single_strided_dev.ptr,
            rows,
            hidden_size,
            out_packed,
            num_experts,
            group_size,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        gemv_awq_selected_pack8_transposed_bf16(
            x_single_dev.ptr,
            selected_dev.ptr,
            qweight_single_t_dev.ptr,
            qzeros_single_dev.ptr,
            scales_single_dev.ptr,
            single_transposed_dev.ptr,
            rows,
            hidden_size,
            out_packed,
            num_experts,
            group_size,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(dual_strided_bits), dual_strided_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(dual_transposed_bits), dual_transposed_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(single_strided_bits), single_strided_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(single_transposed_bits), single_transposed_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    dual_strided_mismatch = int(np.count_nonzero(dual_strided_bits != expected_dual_bits))
    dual_transposed_mismatch = int(np.count_nonzero(dual_transposed_bits != expected_dual_bits))
    single_strided_mismatch = int(np.count_nonzero(single_strided_bits != expected_single_bits))
    single_transposed_mismatch = int(np.count_nonzero(single_transposed_bits != expected_single_bits))
    dual_max_abs = float(
        max(
            np.max(np.abs(_bf16_bits_to_float32(dual_strided_bits) - _bf16_bits_to_float32(expected_dual_bits))),
            np.max(np.abs(_bf16_bits_to_float32(dual_transposed_bits) - _bf16_bits_to_float32(expected_dual_bits))),
        )
    )
    single_max_abs = float(
        max(
            np.max(np.abs(_bf16_bits_to_float32(single_strided_bits) - _bf16_bits_to_float32(expected_single_bits))),
            np.max(np.abs(_bf16_bits_to_float32(single_transposed_bits) - _bf16_bits_to_float32(expected_single_bits))),
        )
    )
    print(
        f"rows={rows} hidden_size={hidden_size} threads={threads} "
        f"dual_mismatch={dual_strided_mismatch}/{dual_transposed_mismatch} "
        f"single_mismatch={single_strided_mismatch}/{single_transposed_mismatch} "
        f"dual_max_abs={dual_max_abs} single_max_abs={single_max_abs}"
    )
    print("dual0=", _bf16_bits_to_float32(dual_strided_bits)[0].tolist())
    print("single0=", _bf16_bits_to_float32(single_strided_bits)[0].tolist())
    return 0 if (
        dual_strided_mismatch == 0
        and dual_transposed_mismatch == 0
        and single_strided_mismatch == 0
        and single_transposed_mismatch == 0
    ) else 1


def _make_pack8_fixture(
    num_experts: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    salt: int,
):
    import numpy as np

    groups = in_features // group_size
    qweight = np.empty((num_experts, in_features, out_packed), dtype=np.int32)
    qzeros = np.empty((num_experts, groups, out_packed), dtype=np.int32)
    scales = np.empty((num_experts, groups, out_packed * 8), dtype=np.float32)
    scale_choices = np.asarray([0.125, 0.25, 0.5, 1.0], dtype=np.float32)
    for expert in range(num_experts):
        for group in range(groups):
            for out_pack in range(out_packed):
                zeros = np.full(8, 8, dtype=np.int32)
                qzeros[expert, group, out_pack] = _pack_awq_lanes(zeros)
                for lane in range(out_pack * 8, out_pack * 8 + 8):
                    scales[expert, group, lane] = scale_choices[(expert + group + lane + salt) % 4]
        for in_col in range(in_features):
            for out_pack in range(out_packed):
                deltas = np.asarray(
                    [((expert + in_col + lane + salt) % 5) - 2 for lane in range(8)],
                    dtype=np.int32,
                )
                qweight[expert, in_col, out_pack] = _pack_awq_lanes(8 + deltas)
    return qweight, qzeros, _float32_to_bf16_bits(scales)


def _selected_pack8_reference(
    x_bits: object,
    selected: object,
    qweight: object,
    qzeros: object,
    scales_bits: object,
    group_size: int,
    *,
    qweight_transposed: bool,
):
    import numpy as np

    x = _bf16_bits_to_float32(x_bits)
    selected_arr = np.asarray(selected, dtype=np.int64)
    qweight_arr = np.asarray(qweight, dtype=np.int32)
    qzeros_arr = np.asarray(qzeros, dtype=np.int32)
    scales = _bf16_bits_to_float32(scales_bits)
    rows = selected_arr.shape[0]
    num_experts = qweight_arr.shape[0]
    out_packed = qweight_arr.shape[1] if qweight_transposed else qweight_arr.shape[2]
    in_features = qweight_arr.shape[2] if qweight_transposed else qweight_arr.shape[1]
    out = np.empty((rows, out_packed * 8), dtype=np.float32)
    for row in range(rows):
        expert = int(selected_arr[row])
        if expert < 0 or expert >= num_experts:
            out[row].fill(0.0)
            continue
        x_row = 0 if x.shape[0] == 1 else row
        for out_pack in range(out_packed):
            acc = np.zeros(8, dtype=np.float32)
            for in_col in range(in_features):
                group = in_col // group_size
                packed_w = int(
                    qweight_arr[expert, out_pack, in_col]
                    if qweight_transposed
                    else qweight_arr[expert, in_col, out_pack]
                )
                packed_z = int(qzeros_arr[expert, group, out_pack])
                xv = np.float32(x[x_row, in_col])
                for lane in range(8):
                    q = (packed_w >> _awq_shift_for_pack_lane(lane)) & 0xF
                    z = (packed_z >> _awq_shift_for_pack_lane(lane)) & 0xF
                    scale = np.float32(scales[expert, group, out_pack * 8 + lane])
                    acc[lane] = np.float32(acc[lane] + np.float32(xv * np.float32(q - z) * scale))
            out[row, out_pack * 8 : out_pack * 8 + 8] = acc
    return _float32_to_bf16_bits(out)


def _pack_awq_lanes(lanes: object):
    import numpy as np

    packed = 0
    for lane, value in enumerate(np.asarray(lanes, dtype=np.int32).tolist()):
        packed |= (int(value) & 0xF) << _awq_shift_for_pack_lane(lane)
    return np.asarray([packed], dtype=np.uint32).view(np.int32)[0]


def _awq_shift_for_pack_lane(lane: int) -> int:
    packed_pos = (4 + (lane >> 1)) if (lane & 1) else (lane >> 1)
    return packed_pos * 4



def _paro_rmsnorm_reference(x_bits: object, weight_bits: object, eps: float):
    import numpy as np

    x = _bf16_bits_to_float32(x_bits)
    weight = _bf16_bits_to_float32(weight_bits)
    sumsq = np.sum(x * x, axis=1, keepdims=True, dtype=np.float32)
    inv_rms = np.float32(1.0) / np.sqrt(sumsq / np.float32(x.shape[1]) + np.float32(eps))
    return _float32_to_bf16_bits(x * inv_rms * weight.reshape(1, x.shape[1]))


def _router_topk_reference(logits: object, top_k: int):
    import numpy as np

    logits_arr = np.asarray(logits, dtype=np.float32)
    selected = np.argsort(-logits_arr, kind="stable")[:top_k].astype(np.int64)
    values = logits_arr[selected]
    shifted = values - np.max(values)
    exp_values = np.exp(shifted, dtype=np.float32)
    routing = (exp_values / np.sum(exp_values, dtype=np.float32)).astype(np.float32)
    return selected, routing

def _int8_pattern(rows: int, cols: int, *, salt: int):
    import numpy as np

    weight = np.empty((rows, cols), dtype=np.int8)
    for row in range(rows):
        weight[row] = np.asarray(
            [((row * 3 + col + salt) % 7) - 3 for col in range(cols)], dtype=np.int8
        )
    return weight


def _silu_np(values):
    import numpy as np

    return values / (np.float32(1.0) + np.exp(-values, dtype=np.float32))

def _float32_to_bf16_bits(values: object):
    import numpy as np

    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype(np.uint16)


def _bf16_bits_to_float32(bits: object):
    import numpy as np

    u32 = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return u32.view(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
