from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
    qwen4_exp_gr_write_bf16_f32,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpGDNLayerDeviceWeights,
    Qwen4ExpGDNLayerScratch,
    Qwen4ExpGDNMixerDeviceWeights,
    Qwen4ExpGRDeviceWeights,
    run_qwen4_exp_gdn_layer,
    run_qwen4_exp_gdn_token_mixer,
    run_qwen4_exp_gr_read,
    run_qwen4_exp_moe,
)
from tests._gguf_synthetic_weights import make_q4_k_weight
from tests.test_qwen4_exp_runner_gr import _dense_f32_weight
from tests.test_qwen4_exp_runner_moe import _q4_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_complete_gdn_layer_matches_individually_gated_components() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rng = np.random.default_rng(40385)
    rows, branches, hidden, low_rank = 1, 2, 256, 16
    k_heads, v_heads, head_dim, conv_kernel = 2, 4, 64, 4
    qkv_width = 2 * k_heads * head_dim + v_heads * head_dim
    core_width = v_heads * head_dim
    ffn, experts, top_k = 256, 4, 2
    residual = float_array_to_bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, branches, hidden)).astype(np.float32)
    )

    allocations = []
    scratch = None
    manual_after = manual_moe_f32 = None
    try:
        d_residual = _upload(residual, runtime, allocations)

        def gr(name: str) -> Qwen4ExpGRDeviceWeights:
            norm = rng.normal(1.0, 0.05, size=(branches, hidden)).astype(np.float32)
            down = rng.normal(
                0.0, 0.05, size=(low_rank, branches * hidden)
            ).astype(np.float32)
            up = rng.normal(
                0.0, 0.05, size=(branches * hidden, low_rank)
            ).astype(np.float32)
            inject = rng.normal(
                0.0, 0.05, size=(branches, branches * hidden)
            ).astype(np.float32)
            return Qwen4ExpGRDeviceWeights(
                norm_weight_ptr=_upload(norm, runtime, allocations).ptr,
                down=_dense_f32_weight(f"{name}_down", down, runtime, allocations),
                up=_dense_f32_weight(f"{name}_up", up, runtime, allocations),
                inject=_dense_f32_weight(
                    f"{name}_inject", inject, runtime, allocations
                ),
            )

        gdn_arrays = {
            "attn_qkv": rng.normal(0.0, 0.05, size=(qkv_width, hidden)).astype(np.float32),
            "attn_gate": rng.normal(0.0, 0.05, size=(core_width, hidden)).astype(np.float32),
            "ssm_alpha": rng.normal(0.0, 0.05, size=(v_heads, hidden)).astype(np.float32),
            "ssm_beta": rng.normal(0.0, 0.05, size=(v_heads, hidden)).astype(np.float32),
            "ssm_out": rng.normal(0.0, 0.05, size=(hidden, core_width)).astype(np.float32),
        }
        gdn_weights = {
            name: _dense_f32_weight(name, value, runtime, allocations)
            for name, value in gdn_arrays.items()
        }
        conv = rng.normal(0.0, 0.05, size=(qkv_width, conv_kernel)).astype(np.float32)
        dt = rng.normal(-1.0, 0.05, size=v_heads).astype(np.float32)
        a_log = rng.normal(-0.5, 0.05, size=v_heads).astype(np.float32)
        norm = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
        conv_state = rng.normal(
            0.0, 0.02, size=(qkv_width, conv_kernel)
        ).astype(np.float32)
        matrix_state = rng.normal(
            0.0, 0.005, size=(v_heads, head_dim, head_dim)
        ).astype(np.float32)
        d_conv_weight = _upload(conv, runtime, allocations)
        d_dt = _upload(dt, runtime, allocations)
        d_a = _upload(a_log, runtime, allocations)
        d_norm = _upload(norm, runtime, allocations)
        d_conv_state = _upload(conv_state, runtime, allocations)
        d_matrix_state = _upload(matrix_state, runtime, allocations)

        router = rng.normal(0.0, 0.05, size=(experts, hidden)).astype(np.float32)
        gate_raw = np.stack([make_q4_k_weight(ffn, hidden) for _ in range(experts)])
        up_raw = np.stack([make_q4_k_weight(ffn, hidden) for _ in range(experts)])
        down_raw = np.stack([make_q4_k_weight(hidden, ffn) for _ in range(experts)])
        moe_weights = {
            "router": _dense_f32_weight("router", router, runtime, allocations),
            "expert_gate": _q4_weight("expert_gate", gate_raw, runtime, allocations),
            "expert_up": _q4_weight("expert_up", up_raw, runtime, allocations),
            "expert_down": _q4_weight("expert_down", down_raw, runtime, allocations),
            "shared_gate": _dense_f32_weight(
                "shared_gate",
                rng.normal(0.0, 0.05, size=(ffn, hidden)).astype(np.float32),
                runtime,
                allocations,
            ),
            "shared_up": _dense_f32_weight(
                "shared_up",
                rng.normal(0.0, 0.05, size=(ffn, hidden)).astype(np.float32),
                runtime,
                allocations,
            ),
            "shared_down": _dense_f32_weight(
                "shared_down",
                rng.normal(0.0, 0.05, size=(hidden, ffn)).astype(np.float32),
                runtime,
                allocations,
            ),
            "shared_gate_weight": _dense_f32_weight(
                "shared_gate_weight",
                rng.normal(0.0, 0.05, size=(1, hidden)).astype(np.float32),
                runtime,
                allocations,
            ),
        }
        layer_weights = Qwen4ExpGDNLayerDeviceWeights(
            attention_gr=gr("attention"),
            mixer=Qwen4ExpGDNMixerDeviceWeights(
                projections=gdn_weights,
                conv_weight_ptr=d_conv_weight.ptr,
                dt_bias_ptr=d_dt.ptr,
                a_log_ptr=d_a.ptr,
                norm_weight_ptr=d_norm.ptr,
            ),
            ffn_gr=gr("ffn"),
            moe=moe_weights,
        )
        scratch = Qwen4ExpGDNLayerScratch.allocate(
            rows=rows,
            branches=branches,
            hidden=hidden,
            low_rank=low_rank,
            qkv_width=qkv_width,
            core_width=core_width,
            scalar_width=v_heads,
            ffn=ffn,
            experts=experts,
            top_k=top_k,
            runtime=runtime,
        )

        initial_conv = _download(d_conv_state, conv_state.shape, np.float32, runtime)
        initial_matrix = _download(
            d_matrix_state, matrix_state.shape, np.float32, runtime
        )
        attn_read = run_qwen4_exp_gr_read(
            d_residual.ptr,
            layer_weights.attention_gr.norm_weight_ptr,
            layer_weights.attention_gr.down,
            layer_weights.attention_gr.up,
            layer_weights.attention_gr.inject,
            scratch.attention_gr,
            rows=rows,
            branches=branches,
            hidden=hidden,
            low_rank=low_rank,
            runtime=runtime,
        )
        mixer = run_qwen4_exp_gdn_token_mixer(
            attn_read.mixed.ptr,
            layer_weights.mixer.projections,
            conv_weight_ptr=layer_weights.mixer.conv_weight_ptr,
            dt_bias_ptr=layer_weights.mixer.dt_bias_ptr,
            a_log_ptr=layer_weights.mixer.a_log_ptr,
            norm_weight_ptr=layer_weights.mixer.norm_weight_ptr,
            conv_state_ptr=d_conv_state.ptr,
            recurrent_state_ptr=d_matrix_state.ptr,
            scratch=scratch.gdn,
            rows=rows,
            hidden=hidden,
            num_k_heads=k_heads,
            num_v_heads=v_heads,
            head_dim=head_dim,
            conv_kernel=conv_kernel,
            runtime=runtime,
        )
        manual_after = malloc(residual.nbytes, runtime=runtime)
        qwen4_exp_gr_write_bf16_f32(
            d_residual.ptr, mixer.ptr, attn_read.inject_logits.ptr, manual_after.ptr,
            rows, branches, hidden, runtime=runtime,
        )
        ffn_read = run_qwen4_exp_gr_read(
            manual_after.ptr,
            layer_weights.ffn_gr.norm_weight_ptr,
            layer_weights.ffn_gr.down,
            layer_weights.ffn_gr.up,
            layer_weights.ffn_gr.inject,
            scratch.ffn_gr,
            rows=rows,
            branches=branches,
            hidden=hidden,
            low_rank=low_rank,
            runtime=runtime,
        )
        manual_moe = run_qwen4_exp_moe(
            ffn_read.mixed.ptr,
            layer_weights.moe,
            scratch=scratch.moe,
            rows=rows,
            hidden=hidden,
            ffn=ffn,
            experts=experts,
            top_k=top_k,
            runtime=runtime,
        )
        manual_moe_f32 = malloc(rows * hidden * 4, runtime=runtime)
        bf16_to_f32(
            manual_moe.output.ptr,
            manual_moe_f32.ptr,
            rows * hidden,
            runtime=runtime,
        )
        manual_final = np.empty_like(residual)
        qwen4_exp_gr_write_bf16_f32(
            manual_after.ptr,
            manual_moe_f32.ptr,
            ffn_read.inject_logits.ptr,
            scratch.output.ptr,
            rows,
            branches,
            hidden,
            runtime=runtime,
        )
        runtime.device_synchronize()
        manual_final = _download(scratch.output, residual.shape, np.uint16, runtime)
        manual_conv = _download(d_conv_state, conv_state.shape, np.float32, runtime)
        manual_matrix = _download(
            d_matrix_state, matrix_state.shape, np.float32, runtime
        )
        copy_host_to_device(
            d_conv_state, host_array_ptr(initial_conv), runtime=runtime
        )
        copy_host_to_device(
            d_matrix_state, host_array_ptr(initial_matrix), runtime=runtime
        )

        integrated = run_qwen4_exp_gdn_layer(
            d_residual.ptr,
            layer_weights,
            conv_state_ptr=d_conv_state.ptr,
            recurrent_state_ptr=d_matrix_state.ptr,
            scratch=scratch,
            rows=rows,
            branches=branches,
            hidden=hidden,
            low_rank=low_rank,
            num_k_heads=k_heads,
            num_v_heads=v_heads,
            head_dim=head_dim,
            conv_kernel=conv_kernel,
            ffn=ffn,
            experts=experts,
            top_k=top_k,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(integrated, residual.shape, np.uint16, runtime)
        actual_conv = _download(d_conv_state, conv_state.shape, np.float32, runtime)
        actual_matrix = _download(
            d_matrix_state, matrix_state.shape, np.float32, runtime
        )
    finally:
        if manual_after is not None:
            free(manual_after, runtime=runtime)
        if manual_moe_f32 is not None:
            free(manual_moe_f32, runtime=runtime)
        if scratch is not None:
            scratch.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(actual, manual_final)
    np.testing.assert_array_equal(actual_conv, manual_conv)
    np.testing.assert_array_equal(actual_matrix, manual_matrix)


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
