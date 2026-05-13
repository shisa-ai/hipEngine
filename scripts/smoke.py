#!/usr/bin/env python3
"""Early scaffold smokes.

Default modes are CPU-only and safe before GPU clearance. ``smoke-add-hip`` is the explicit
GPU/JIT path for the first raw-pointer HIP smoke kernel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.dispatch.fusion import FusionPlanner, resolve_plan
from hipengine.kernels.registry import MissingKernelError
from hipengine.models import resolve_model


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
            "paro-selected-gemv-hip",
            "paro-selected-gemv-rotate-hip",
            "paro-pack8-gemv-hip",
            "paro-rotate-hip",
            "paro-silu-hip",
            "paro-combine-hip",
            "dense-gemv-hip",
            "w8a16-linear-hip",
            "w8a16-shared-expert-hip",
            "paro-moe-c1-hip",
        ),
        default="registry",
    )
    parser.add_argument("--n", type=int, default=1024, help="Element count for smoke-add-hip.")
    parser.add_argument("--rows", type=int, default=2, help="Rows/tokens for HIP smoke modes.")
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
        paro_rmsnorm_out_bf16,
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
    print(
        f"rows={rows} hidden_size={hidden_size} "
        f"norm_max_abs={norm_max_abs} norm_bit_mismatch={norm_bit_mismatch} "
        f"add_norm_max_abs={add_norm_max_abs} add_norm_bit_mismatch={add_norm_bit_mismatch} "
        f"residual_max_abs={residual_max_abs} residual_bit_mismatch={residual_bit_mismatch}"
    )
    print("first_row=", norm_out[0, : min(5, hidden_size)].tolist())
    return 0 if (
        norm_bit_mismatch == 0
        and add_norm_bit_mismatch == 0
        and residual_bit_mismatch == 0
    ) else 1





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
        weighted_dev = out_dev(weighted_bits)
        weighted_shared_residual_dev = out_dev(weighted_shared_residual_bits)
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
        copy_device_to_host(host_array_ptr(shared_combine_bits), shared_combine_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(shared_residual_bits), shared_residual_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    weighted_mismatch = int(np.count_nonzero(weighted_bits != expected_weighted_bits))
    fused_mismatch = int(
        np.count_nonzero(weighted_shared_residual_bits != expected_weighted_shared_residual_bits)
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
        f"shared_mismatch={shared_mismatch} shared_max_abs={shared_max_abs} "
        f"shared_residual_mismatch={shared_residual_mismatch} "
        f"shared_residual_max_abs={shared_residual_max_abs}"
    )
    print("weighted=", _bf16_bits_to_float32(weighted_bits)[0, : min(8, features)].tolist())
    print("fused=", _bf16_bits_to_float32(weighted_shared_residual_bits)[0, : min(8, features)].tolist())
    return 0 if (
        weighted_mismatch == 0
        and fused_mismatch == 0
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
        paro_rotate2_bf16,
        paro_rotate3_bf16,
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
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(rotate2_out0), r2o0_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(rotate2_out1), r2o1_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(rotate3_out0), r3o0_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(rotate3_out1), r3o1_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(rotate3_out2), r3o2_dev, runtime=runtime)
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
    print(
        f"rows={rows} hidden_size={hidden_size} group_size={group_size} krot={krot} "
        f"mismatches={mismatches} max_abs={max_abs}"
    )
    print("rotate2_0=", _bf16_bits_to_float32(rotate2_out0)[0].tolist())
    print("rotate3_2=", _bf16_bits_to_float32(rotate3_out2)[0].tolist())
    return 0 if max(mismatches) == 0 else 1

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
