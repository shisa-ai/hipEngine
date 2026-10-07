#!/usr/bin/env python3
"""MT-1 verifier-body composition probe (diagnostic, no model load).

MT-1 wants one target pass over the draft rows that uses the c1 decode
arithmetic *per row*. Two halves already exist and are tested in isolation:

* the q8_0 dense-linear rowbatch owner is bit-exact width-invariant at rows
  2..8 (``test_q8_0_coltile8_rowbatch4_is_width_invariant_to_serial_c1``);
* the head is already batched
  (``Qwen4ExpGGUFResidentModelRunner.verify_target_block_deferred_head``).

The body in between is not composed. This probe measures the three body
primitives that a rows>1 c1 pass needs and reports, per primitive, whether the
rows>1 launch reproduces the serial c1 launch bit-for-bit:

* QSA attention: strict decode mixer is rows==1-only (the hard blocker), and the
  rows-capable prefill mixer is numerically close but not bit-exact;
* GDN recurrent layer: rows>1 output/conv/recurrent state deviate from serial;
* MoE: rows=3 non-grouped is bit-exact against serial (already covered by the
  runner test, re-measured here for the same artifact).

Usage:
    LD_LIBRARY_PATH=<rocm devel lib> PATH=<rocm bin>:$PATH \
      python3 scripts/qwen4exp_mt1_verifier_body_probe.py --output <json>
"""

from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from hipengine.loading.materialize import float_array_to_bf16_bits  # noqa: E402
from hipengine.quant.gguf import bf16_to_float32  # noqa: E402


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def measure_qsa_attention() -> dict:
    """Rows>1 prefill QSA mixer vs serial rows=1 decode mixer, and the guard."""

    from hipengine.core.memory import free
    from hipengine.runtime.qwen4_exp_runner import (
        Qwen4ExpDenseAttentionState,
        Qwen4ExpQSAIndexDeviceState,
        Qwen4ExpQSAMixerDeviceWeights,
        Qwen4ExpQSAPrefillMetadata,
        Qwen4ExpQSAScratch,
        run_qwen4_exp_dense_qsa_token_mixer,
        run_qwen4_exp_qsa_prefill_token_mixer,
    )
    from tests.test_gpu_qwen4_exp_runner_gr import _dense_f32_weight
    from tests.test_gpu_qwen4_exp_runner_qsa import _download, _upload

    runtime = __import__("hipengine.core.hip", fromlist=["get_hip_runtime"]).get_hip_runtime()
    rng = np.random.default_rng(4320)
    rows, hidden, query_heads, kv_heads, head_dim = 6, 8, 2, 1, 4
    index_heads, index_dim, ratio, block_budget = 2, 4, 2, 2
    q_width = query_heads * head_dim
    kv_width = kv_heads * head_dim
    arrays = {
        "attn_q": rng.normal(0.0, 0.1, size=(q_width * 2, hidden)).astype(np.float32),
        "attn_k": rng.normal(0.0, 0.1, size=(kv_width, hidden)).astype(np.float32),
        "attn_v": rng.normal(0.0, 0.1, size=(kv_width, hidden)).astype(np.float32),
        "attn_output": rng.normal(0.0, 0.1, size=(hidden, q_width)).astype(np.float32),
        "index_q": rng.normal(0.0, 0.1, size=(index_heads * index_dim, hidden)).astype(np.float32),
        "index_k": rng.normal(0.0, 0.1, size=(index_dim, hidden)).astype(np.float32),
    }
    q_norm = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    k_norm = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    index_q_norm = rng.normal(1.0, 0.05, size=index_dim).astype(np.float32)
    index_k_norm = rng.normal(1.0, 0.05, size=index_dim).astype(np.float32)
    mixed_rows = rng.normal(0.0, 0.2, size=(rows, hidden)).astype(np.float32)

    allocations = []
    owners = []
    try:
        weights = Qwen4ExpQSAMixerDeviceWeights(
            projections={
                name: _dense_f32_weight(name, array, runtime, allocations)
                for name, array in arrays.items()
            },
            q_norm_weight_ptr=_upload(q_norm, runtime, allocations).ptr,
            k_norm_weight_ptr=_upload(k_norm, runtime, allocations).ptr,
            index_q_norm_weight_ptr=_upload(index_q_norm, runtime, allocations).ptr,
            index_k_norm_weight_ptr=_upload(index_k_norm, runtime, allocations).ptr,
        )
        serial_state = Qwen4ExpDenseAttentionState.allocate(
            max_positions=8, block_size=256, kv_heads=kv_heads, head_dim=head_dim,
            runtime=runtime,
        )
        bulk_state = Qwen4ExpDenseAttentionState.allocate(
            max_positions=8, block_size=256, kv_heads=kv_heads, head_dim=head_dim,
            runtime=runtime,
        )
        serial_index = Qwen4ExpQSAIndexDeviceState.allocate(
            attention_state=serial_state, index_heads=index_heads, index_dim=index_dim,
            compression_ratio=ratio, block_budget=block_budget, runtime=runtime,
        )
        bulk_index = Qwen4ExpQSAIndexDeviceState.allocate(
            attention_state=bulk_state, index_heads=index_heads, index_dim=index_dim,
            compression_ratio=ratio, block_budget=block_budget, runtime=runtime,
        )
        serial_scratch = Qwen4ExpQSAScratch.allocate(
            rows=1, hidden=hidden, query_heads=query_heads, kv_heads=kv_heads,
            head_dim=head_dim, index_heads=index_heads, index_dim=index_dim,
            runtime=runtime,
        )
        bulk_scratch = Qwen4ExpQSAScratch.allocate(
            rows=rows, hidden=hidden, query_heads=query_heads, kv_heads=kv_heads,
            head_dim=head_dim, index_heads=index_heads, index_dim=index_dim,
            runtime=runtime,
        )
        metadata = Qwen4ExpQSAPrefillMetadata.allocate(
            bulk_state, rows=rows,
            selection_capacity=block_budget * ratio + ratio - 1,
        )
        owners.extend([metadata, bulk_scratch, serial_scratch, bulk_index,
                       serial_index, bulk_state, serial_state])
        d_mixed = _upload(mixed_rows, runtime, allocations)
        serial_outputs = []
        for position in range(rows):
            output = run_qwen4_exp_dense_qsa_token_mixer(
                d_mixed.ptr + position * hidden * 4, weights,
                attention_state=serial_state, index_state=serial_index,
                scratch=serial_scratch, position=position, rows=1, hidden=hidden,
                query_heads=query_heads, kv_heads=kv_heads, head_dim=head_dim,
                rotary_dim=4, theta=100.0, index_heads=index_heads,
                index_dim=index_dim, index_rotary_dim=4, runtime=runtime,
            )
            runtime.device_synchronize()
            serial_outputs.append(_download(output, (hidden,), np.float32, runtime))
        bulk_output = run_qwen4_exp_qsa_prefill_token_mixer(
            d_mixed.ptr, weights, attention_state=bulk_state,
            index_state=bulk_index, scratch=bulk_scratch, metadata=metadata,
            start_position=0, rows=rows, hidden=hidden, query_heads=query_heads,
            kv_heads=kv_heads, head_dim=head_dim, rotary_dim=4, theta=100.0,
            index_heads=index_heads, index_dim=index_dim, index_rotary_dim=4,
            runtime=runtime,
        )
        runtime.device_synchronize()
        bulk_outputs = _download(bulk_output, (rows, hidden), np.float32, runtime)
        serial_key = _download(
            serial_state.key_cache,
            (serial_state.max_positions, kv_heads, head_dim), np.uint16, runtime,
        )
        bulk_key = _download(
            bulk_state.key_cache,
            (bulk_state.max_positions, kv_heads, head_dim), np.uint16, runtime,
        )
        serial_raw = _download(
            serial_index.raw_keys, (serial_index.capacity, index_dim), np.float32, runtime,
        )
        bulk_raw = _download(
            bulk_index.raw_keys, (bulk_index.capacity, index_dim), np.float32, runtime,
        )
        output_delta = float(
            np.max(np.abs(bulk_outputs - np.asarray(serial_outputs)))
        )
        return {
            "primitive": "qsa_attention",
            "rows": rows,
            "decode_mixer_rows_one_only": True,
            "prefill_mixer_rows_capable": True,
            "output_max_abs_delta": output_delta,
            "output_bit_exact": bool(output_delta == 0.0),
            "kv_cache_bit_exact": bool(np.array_equal(bulk_key, serial_key)),
            "index_raw_keys_bit_exact": bool(np.array_equal(bulk_raw, serial_raw)),
        }
    finally:
        for owner in reversed(owners):
            owner.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


def measure_gdn_layer() -> dict:
    """Rows>1 GDN layer (output + conv/recurrent state) vs serial rows=1."""

    from hipengine.core.memory import free
    from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32
    from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
        qwen4_exp_gr_write_bf16_f32,
    )
    from hipengine.loading.materialize import float_array_to_bf16_bits
    from hipengine.quant.gguf import bf16_to_float32
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
    from tests.test_gpu_qwen4_exp_runner_gr import _dense_f32_weight, _download, _upload
    from tests.test_gpu_qwen4_exp_runner_moe import _q4_weight

    runtime = __import__("hipengine.core.hip", fromlist=["get_hip_runtime"]).get_hip_runtime()
    rng = np.random.default_rng(40385)
    rows, branches, hidden, low_rank = 3, 2, 256, 16
    k_heads, v_heads, head_dim, conv_kernel = 2, 4, 64, 4
    qkv_width = 2 * k_heads * head_dim + v_heads * head_dim
    core_width = v_heads * head_dim
    ffn, experts, top_k = 256, 4, 2
    residual = float_array_to_bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, branches, hidden)).astype(np.float32)
    )
    allocations: list = []
    scratch = serial_scratch = None
    try:
        d_residual = _upload(residual, runtime, allocations)

        def gr(name: str) -> Qwen4ExpGRDeviceWeights:
            return Qwen4ExpGRDeviceWeights(
                norm_weight_ptr=_upload(
                    rng.normal(1.0, 0.05, size=(branches, hidden)).astype(np.float32),
                    runtime, allocations,
                ).ptr,
                down=_dense_f32_weight(
                    f"{name}_down",
                    rng.normal(0.0, 0.05, size=(low_rank, branches * hidden)).astype(np.float32),
                    runtime, allocations,
                ),
                up=_dense_f32_weight(
                    f"{name}_up",
                    rng.normal(0.0, 0.05, size=(branches * hidden, low_rank)).astype(np.float32),
                    runtime, allocations,
                ),
                inject=_dense_f32_weight(
                    f"{name}_inject",
                    rng.normal(0.0, 0.05, size=(branches, branches * hidden)).astype(np.float32),
                    runtime, allocations,
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
        conv_state = rng.normal(0.0, 0.02, size=(qkv_width, conv_kernel)).astype(np.float32)
        matrix_state = rng.normal(
            0.0, 0.005, size=(v_heads, head_dim, head_dim)
        ).astype(np.float32)
        d_conv_state = _upload(conv_state, runtime, allocations)
        d_matrix_state = _upload(matrix_state, runtime, allocations)
        d_serial_conv_state = _upload(conv_state, runtime, allocations)
        d_serial_matrix_state = _upload(matrix_state, runtime, allocations)

        gate_raw = np.stack([make_q4_k_weight(ffn, hidden) for _ in range(experts)])
        up_raw = np.stack([make_q4_k_weight(ffn, hidden) for _ in range(experts)])
        down_raw = np.stack([make_q4_k_weight(hidden, ffn) for _ in range(experts)])
        moe_weights = {
            "router": _dense_f32_weight(
                "router", rng.normal(0.0, 0.05, size=(experts, hidden)).astype(np.float32),
                runtime, allocations,
            ),
            "expert_gate": _q4_weight("expert_gate", gate_raw, runtime, allocations),
            "expert_up": _q4_weight("expert_up", up_raw, runtime, allocations),
            "expert_down": _q4_weight("expert_down", down_raw, runtime, allocations),
            "shared_gate": _dense_f32_weight(
                "shared_gate", rng.normal(0.0, 0.05, size=(ffn, hidden)).astype(np.float32),
                runtime, allocations,
            ),
            "shared_up": _dense_f32_weight(
                "shared_up", rng.normal(0.0, 0.05, size=(ffn, hidden)).astype(np.float32),
                runtime, allocations,
            ),
            "shared_down": _dense_f32_weight(
                "shared_down", rng.normal(0.0, 0.05, size=(hidden, ffn)).astype(np.float32),
                runtime, allocations,
            ),
            "shared_gate_weight": _dense_f32_weight(
                "shared_gate_weight", rng.normal(0.0, 0.05, size=(1, hidden)).astype(np.float32),
                runtime, allocations,
            ),
        }
        layer_weights = Qwen4ExpGDNLayerDeviceWeights(
            attention_gr=gr("attention"),
            mixer=Qwen4ExpGDNMixerDeviceWeights(
                projections=gdn_weights,
                conv_weight_ptr=_upload(
                    rng.normal(0.0, 0.05, size=(qkv_width, conv_kernel)).astype(np.float32),
                    runtime, allocations,
                ).ptr,
                dt_bias_ptr=_upload(
                    rng.normal(-1.0, 0.05, size=v_heads).astype(np.float32), runtime, allocations
                ).ptr,
                a_ptr=_upload(
                    rng.normal(-0.5, 0.05, size=v_heads).astype(np.float32), runtime, allocations
                ).ptr,
                norm_weight_ptr=_upload(
                    rng.normal(1.0, 0.05, size=head_dim).astype(np.float32), runtime, allocations
                ).ptr,
            ),
            ffn_gr=gr("ffn"),
            moe=moe_weights,
        )
        scratch = Qwen4ExpGDNLayerScratch.allocate(
            rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
            qkv_width=qkv_width, core_width=core_width, scalar_width=v_heads,
            ffn=ffn, experts=experts, top_k=top_k, runtime=runtime,
        )
        serial_scratch = Qwen4ExpGDNLayerScratch.allocate(
            rows=1, branches=branches, hidden=hidden, low_rank=low_rank,
            qkv_width=qkv_width, core_width=core_width, scalar_width=v_heads,
            ffn=ffn, experts=experts, top_k=top_k, runtime=runtime,
        )
        integrated = run_qwen4_exp_gdn_layer(
            d_residual.ptr, layer_weights,
            conv_state_ptr=d_conv_state.ptr, recurrent_state_ptr=d_matrix_state.ptr,
            scratch=scratch, rows=rows, branches=branches, hidden=hidden,
            low_rank=low_rank, num_k_heads=k_heads, num_v_heads=v_heads,
            head_dim=head_dim, conv_kernel=conv_kernel, ffn=ffn, experts=experts,
            top_k=top_k, runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(integrated, residual.shape, np.uint16, runtime)
        actual_conv = _download(d_conv_state, conv_state.shape, np.float32, runtime)
        actual_matrix = _download(d_matrix_state, matrix_state.shape, np.float32, runtime)
        serial_rows = []
        for row in range(rows):
            serial_output = run_qwen4_exp_gdn_layer(
                d_residual.ptr + row * branches * hidden * 2, layer_weights,
                conv_state_ptr=d_serial_conv_state.ptr,
                recurrent_state_ptr=d_serial_matrix_state.ptr,
                scratch=serial_scratch, rows=1, branches=branches, hidden=hidden,
                low_rank=low_rank, num_k_heads=k_heads, num_v_heads=v_heads,
                head_dim=head_dim, conv_kernel=conv_kernel, ffn=ffn, experts=experts,
                top_k=top_k, runtime=runtime,
            )
            runtime.device_synchronize()
            serial_rows.append(_download(serial_output, (branches, hidden), np.uint16, runtime))
        serial = np.asarray(serial_rows)
        serial_conv = _download(d_serial_conv_state, conv_state.shape, np.float32, runtime)
        serial_matrix = _download(
            d_serial_matrix_state, matrix_state.shape, np.float32, runtime
        )
        out_delta = float(
            np.max(np.abs(bf16_to_float32(actual) - bf16_to_float32(serial)))
        )
        conv_delta = float(np.max(np.abs(actual_conv - serial_conv)))
        matrix_delta = float(np.max(np.abs(actual_matrix - serial_matrix)))
        return {
            "primitive": "gdn_recurrent_layer",
            "rows": rows,
            "output_bit_exact": bool(np.array_equal(actual, serial)),
            "output_max_abs_delta": out_delta,
            "conv_state_bit_exact": bool(np.array_equal(actual_conv, serial_conv)),
            "conv_state_max_abs_delta": conv_delta,
            "recurrent_state_bit_exact": bool(np.array_equal(actual_matrix, serial_matrix)),
            "recurrent_state_max_abs_delta": matrix_delta,
        }
    finally:
        if serial_scratch is not None:
            serial_scratch.close()
        if scratch is not None:
            scratch.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


def measure_qsa_layer() -> dict:
    """Full QSA layer (GR+attention+MoE) rows>1 prefill vs serial rows=1 decode."""

    from hipengine.core.memory import free
    from hipengine.runtime.qwen4_exp_runner import (
        Qwen4ExpDenseAttentionState,
        Qwen4ExpQSAIndexDeviceState,
        Qwen4ExpQSALayerDeviceWeights,
        Qwen4ExpQSALayerScratch,
        Qwen4ExpQSAMixerDeviceWeights,
        Qwen4ExpQSAPrefillMetadata,
        run_qwen4_exp_dense_qsa_layer,
        run_qwen4_exp_qsa_prefill_layer,
    )
    from tests._gguf_synthetic_weights import make_q4_k_weight
    from tests.test_gpu_qwen4_exp_runner_gr import _dense_f32_weight, _download, _upload
    from tests.test_gpu_qwen4_exp_runner_moe import _q4_weight

    runtime = __import__("hipengine.core.hip", fromlist=["get_hip_runtime"]).get_hip_runtime()
    rng = np.random.default_rng(20261007)
    rows, branches, hidden, low_rank = 4, 2, 256, 16
    query_heads, kv_heads, head_dim, rotary_dim = 2, 1, 4, 4
    index_heads, index_dim, ratio, block_budget = 2, 4, 2, 2
    ffn, experts, top_k = 256, 4, 2
    allocations: list = []
    owners: list = []
    try:
        residual = float_array_to_bf16_bits(
            rng.normal(0.0, 0.1, size=(rows, branches, hidden)).astype(np.float32)
        )
        d_residual = _upload(residual, runtime, allocations)

        def gr(name: str):
            from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGRDeviceWeights

            return Qwen4ExpGRDeviceWeights(
                norm_weight_ptr=_upload(
                    rng.normal(1.0, 0.05, size=(branches, hidden)).astype(np.float32),
                    runtime, allocations,
                ).ptr,
                down=_dense_f32_weight(
                    f"{name}_down",
                    rng.normal(0.0, 0.05, size=(low_rank, branches * hidden)).astype(np.float32),
                    runtime, allocations,
                ),
                up=_dense_f32_weight(
                    f"{name}_up",
                    rng.normal(0.0, 0.05, size=(branches * hidden, low_rank)).astype(np.float32),
                    runtime, allocations,
                ),
                inject=_dense_f32_weight(
                    f"{name}_inject",
                    rng.normal(0.0, 0.05, size=(branches, branches * hidden)).astype(np.float32),
                    runtime, allocations,
                ),
            )

        q_width = query_heads * head_dim
        kv_width = kv_heads * head_dim
        projections = {
            "attn_q": _dense_f32_weight(
                "attn_q", rng.normal(0.0, 0.1, size=(q_width * 2, hidden)).astype(np.float32),
                runtime, allocations,
            ),
            "attn_k": _dense_f32_weight(
                "attn_k", rng.normal(0.0, 0.1, size=(kv_width, hidden)).astype(np.float32),
                runtime, allocations,
            ),
            "attn_v": _dense_f32_weight(
                "attn_v", rng.normal(0.0, 0.1, size=(kv_width, hidden)).astype(np.float32),
                runtime, allocations,
            ),
            "attn_output": _dense_f32_weight(
                "attn_output", rng.normal(0.0, 0.1, size=(hidden, q_width)).astype(np.float32),
                runtime, allocations,
            ),
            "index_q": _dense_f32_weight(
                "index_q", rng.normal(0.0, 0.1, size=(index_heads * index_dim, hidden)).astype(np.float32),
                runtime, allocations,
            ),
            "index_k": _dense_f32_weight(
                "index_k", rng.normal(0.0, 0.1, size=(index_dim, hidden)).astype(np.float32),
                runtime, allocations,
            ),
        }
        mixer = Qwen4ExpQSAMixerDeviceWeights(
            projections=projections,
            q_norm_weight_ptr=_upload(
                rng.normal(1.0, 0.05, size=head_dim).astype(np.float32), runtime, allocations
            ).ptr,
            k_norm_weight_ptr=_upload(
                rng.normal(1.0, 0.05, size=head_dim).astype(np.float32), runtime, allocations
            ).ptr,
            index_q_norm_weight_ptr=_upload(
                rng.normal(1.0, 0.05, size=index_dim).astype(np.float32), runtime, allocations
            ).ptr,
            index_k_norm_weight_ptr=_upload(
                rng.normal(1.0, 0.05, size=index_dim).astype(np.float32), runtime, allocations
            ).ptr,
        )
        gate_raw = np.stack([make_q4_k_weight(ffn, hidden) for _ in range(experts)])
        up_raw = np.stack([make_q4_k_weight(ffn, hidden) for _ in range(experts)])
        down_raw = np.stack([make_q4_k_weight(hidden, ffn) for _ in range(experts)])
        moe_weights = {
            "router": _dense_f32_weight(
                "router", rng.normal(0.0, 0.05, size=(experts, hidden)).astype(np.float32),
                runtime, allocations,
            ),
            "expert_gate": _q4_weight("expert_gate", gate_raw, runtime, allocations),
            "expert_up": _q4_weight("expert_up", up_raw, runtime, allocations),
            "expert_down": _q4_weight("expert_down", down_raw, runtime, allocations),
            "shared_gate": _dense_f32_weight(
                "shared_gate", rng.normal(0.0, 0.05, size=(ffn, hidden)).astype(np.float32),
                runtime, allocations,
            ),
            "shared_up": _dense_f32_weight(
                "shared_up", rng.normal(0.0, 0.05, size=(ffn, hidden)).astype(np.float32),
                runtime, allocations,
            ),
            "shared_down": _dense_f32_weight(
                "shared_down", rng.normal(0.0, 0.05, size=(hidden, ffn)).astype(np.float32),
                runtime, allocations,
            ),
            "shared_gate_weight": _dense_f32_weight(
                "shared_gate_weight", rng.normal(0.0, 0.05, size=(1, hidden)).astype(np.float32),
                runtime, allocations,
            ),
        }
        layer_weights = Qwen4ExpQSALayerDeviceWeights(
            attention_gr=gr("attention"),
            mixer=mixer,
            ffn_gr=gr("ffn"),
            moe=moe_weights,
            layer_id=0,
            layer_type="qsa",
            qsa_state_index=0,
        )
        bulk_state = Qwen4ExpDenseAttentionState.allocate(
            max_positions=16, block_size=256, kv_heads=kv_heads, head_dim=head_dim,
            runtime=runtime,
        )
        serial_state = Qwen4ExpDenseAttentionState.allocate(
            max_positions=16, block_size=256, kv_heads=kv_heads, head_dim=head_dim,
            runtime=runtime,
        )
        bulk_index = Qwen4ExpQSAIndexDeviceState.allocate(
            attention_state=bulk_state, index_heads=index_heads, index_dim=index_dim,
            compression_ratio=ratio, block_budget=block_budget, runtime=runtime,
        )
        serial_index = Qwen4ExpQSAIndexDeviceState.allocate(
            attention_state=serial_state, index_heads=index_heads, index_dim=index_dim,
            compression_ratio=ratio, block_budget=block_budget, runtime=runtime,
        )
        bulk_scratch = Qwen4ExpQSALayerScratch.allocate(
            rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
            query_heads=query_heads, kv_heads=kv_heads, head_dim=head_dim,
            ffn=ffn, experts=experts, top_k=top_k, index_heads=index_heads,
            index_dim=index_dim, runtime=runtime,
        )
        serial_scratch = Qwen4ExpQSALayerScratch.allocate(
            rows=1, branches=branches, hidden=hidden, low_rank=low_rank,
            query_heads=query_heads, kv_heads=kv_heads, head_dim=head_dim,
            ffn=ffn, experts=experts, top_k=top_k, index_heads=index_heads,
            index_dim=index_dim, runtime=runtime,
        )
        metadata = Qwen4ExpQSAPrefillMetadata.allocate(
            bulk_state, rows=rows, selection_capacity=block_budget * ratio + ratio - 1,
        )
        owners.extend([metadata, serial_scratch, bulk_scratch, serial_index,
                       bulk_index, serial_state, bulk_state])
        bulk_output = run_qwen4_exp_qsa_prefill_layer(
            d_residual.ptr, layer_weights, attention_state=bulk_state,
            index_state=bulk_index, scratch=bulk_scratch, metadata=metadata,
            start_position=0, rows=rows, branches=branches, hidden=hidden,
            low_rank=low_rank, query_heads=query_heads, kv_heads=kv_heads,
            head_dim=head_dim, rotary_dim=rotary_dim, theta=100.0,
            index_heads=index_heads, index_dim=index_dim, index_rotary_dim=rotary_dim,
            ffn=ffn, experts=experts, top_k=top_k, runtime=runtime,
        )
        runtime.device_synchronize()
        bulk = _download(bulk_output, residual.shape, np.uint16, runtime)
        serial_rows = []
        for position in range(rows):
            serial_output = run_qwen4_exp_dense_qsa_layer(
                d_residual.ptr + position * branches * hidden * 2, layer_weights,
                attention_state=serial_state, index_state=serial_index,
                scratch=serial_scratch, position=position, rows=1, branches=branches,
                hidden=hidden, low_rank=low_rank, query_heads=query_heads,
                kv_heads=kv_heads, head_dim=head_dim, rotary_dim=rotary_dim,
                theta=100.0, index_heads=index_heads, index_dim=index_dim,
                index_rotary_dim=rotary_dim, ffn=ffn, experts=experts, top_k=top_k,
                runtime=runtime,
            )
            runtime.device_synchronize()
            serial_rows.append(_download(serial_output, (branches, hidden), np.uint16, runtime))
        serial = np.asarray(serial_rows)
        bulk_key = _download(
            bulk_state.key_cache,
            (bulk_state.max_positions, kv_heads, head_dim), np.uint16, runtime,
        )
        serial_key = _download(
            serial_state.key_cache,
            (serial_state.max_positions, kv_heads, head_dim), np.uint16, runtime,
        )
        bulk_value = _download(
            bulk_state.value_cache,
            (bulk_state.max_positions, kv_heads, head_dim), np.uint16, runtime,
        )
        serial_value = _download(
            serial_state.value_cache,
            (serial_state.max_positions, kv_heads, head_dim), np.uint16, runtime,
        )
        return {
            "primitive": "qsa_full_layer",
            "rows": rows,
            "output_bit_exact": bool(np.array_equal(bulk, serial)),
            "output_max_abs_delta": float(
                np.max(np.abs(bf16_to_float32(bulk) - bf16_to_float32(serial)))
            ),
            "kv_key_bit_exact": bool(np.array_equal(bulk_key, serial_key)),
            "kv_value_bit_exact": bool(np.array_equal(bulk_value, serial_value)),
        }
    finally:
        for owner in reversed(owners):
            owner.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


def measure_decode_guard() -> dict:
    """The strict decode QSA mixer refuses every verifier width."""

    from hipengine.runtime.qwen4_exp_runner import run_qwen4_exp_dense_qsa_token_mixer

    rejected = []
    for rows in range(2, 9):
        try:
            run_qwen4_exp_dense_qsa_token_mixer(
                0, {}, attention_state=None, scratch=None, position=0, rows=rows,
                hidden=8, query_heads=2, kv_heads=1, head_dim=4, rotary_dim=4,
                theta=100.0,
            )
        except ValueError as exc:
            if "requires rows == 1" in str(exc):
                rejected.append(rows)
    return {
        "primitive": "qsa_attention_decode_guard",
        "widths_rejected": rejected,
        "all_verifier_widths_rejected": rejected == list(range(2, 9)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument(
        "--only",
        default=None,
        help="Run only this measurement (qsa_attention|qsa_full_layer|gdn_recurrent_layer).",
    )
    args = parser.parse_args()
    payload: dict = {
        "schema": 1,
        "probe": "qwen4exp_mt1_verifier_body_boundary",
        "hip_available": _hip_available(),
        "primitives": [],
    }
    payload["primitives"].append(measure_decode_guard())
    if not _hip_available():
        payload["status"] = "blocked_no_hip"
    else:
        for name, fn in (
            ("qsa_attention", measure_qsa_attention),
            ("qsa_full_layer", measure_qsa_layer),
            ("gdn_recurrent_layer", measure_gdn_layer),
        ):
            if args.only is not None and args.only != name:
                continue
            try:
                payload["primitives"].append(fn())
            except BaseException as exc:  # noqa: BLE001 - diagnostic probe
                payload["primitives"].append(
                    {"primitive": name, "status": "error", "error": repr(exc)}
                )
        payload["status"] = "measured"
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
