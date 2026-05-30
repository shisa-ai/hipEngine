from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.kernels.cpu_reference import (
    gguf_q3_k_gemv,
    gguf_q5_k_gemv,
    gguf_q8_0_embedding,
    gguf_q8_0_gemv,
    step_moe_router,
)
from hipengine.kernels.cpu_reference.ops import step_rmsnorm
from hipengine.loading.gguf import GGUFReader, scan_gguf_splits
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.stepfun_gguf import build_stepfun_gguf_tensor_map
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.stepfun_gguf_runner import StepFunResidentSession

DEFAULT_STEPFUN_GGUF_DIR = Path("/data/models/gguf")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


def _stepfun_gguf_paths() -> tuple[Path, ...]:
    root = Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", DEFAULT_STEPFUN_GGUF_DIR))
    paths = tuple(sorted(root.glob("Step-3.7-flash-Q3_K_L-*.gguf")))
    if len(paths) != 3:
        pytest.skip(
            "StepFun GGUF Q3_K_L shards not found; set HIPENGINE_STEPFUN_GGUF_DIR "
            "to a directory containing Step-3.7-flash-Q3_K_L-00001..00003.gguf"
        )
    return paths


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_embeds_real_q8_tokens() -> None:
    had_torch = "torch" in sys.modules
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    token_tensor = model_map.root("token_embedding")
    token_ids = np.asarray(
        [
            0,
            model_map.config.bos_token_id,
            model_map.config.eos_token_id,
            128007,
            model_map.config.vocab_size - 1,
            model_map.config.eos_token_id,
        ],
        dtype=np.int64,
    )
    raw = GGUFReader(token_tensor.source_path).tensor_data(token_tensor.name)
    assert token_tensor.ggml_type_name == "Q8_0"
    assert raw.shape[0] == model_map.config.vocab_size
    expected_bits = float_array_to_bf16_bits(gguf_q8_0_embedding(token_ids, raw))
    runtime = get_hip_runtime()
    reset_memory_stats()

    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=("root.token_embedding",),
        runtime=runtime,
    )
    try:
        assert session.weights.allocated_nbytes == token_tensor.nbytes
        actual_bits = session.embed_token_ids_bf16(token_ids, runtime=runtime)
        assert actual_bits.shape == (token_ids.size, model_map.config.hidden_size)
        assert actual_bits.dtype == np.uint16
        np.testing.assert_array_equal(actual_bits, expected_bits)
        np.testing.assert_array_equal(actual_bits[2], actual_bits[5])
        stats = memory_stats()
        assert stats["current_allocated_bytes"] == token_tensor.nbytes
        assert stats["active_allocations"] == 1
    finally:
        session.free(runtime=runtime)

    assert memory_stats()["current_allocated_bytes"] == 0
    if not had_torch:
        assert "torch" not in sys.modules


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_embeds_rendered_chat_prompt() -> None:
    had_torch = "torch" in sys.modules
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    token_tensor = model_map.root("token_embedding")
    raw = GGUFReader(token_tensor.source_path).tensor_data(token_tensor.name)
    runtime = get_hip_runtime()
    reset_memory_stats()
    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=("root.token_embedding",),
        runtime=runtime,
    )
    try:
        prompt = session.embed_chat_prompt_bf16(
            [{"role": "user", "content": "hello"}],
            reasoning_effort="low",
            runtime=runtime,
        )
        assert prompt.rendered_prompt.endswith("<|im_start|>assistant\n<think>\n")
        assert prompt.prompt_length == len(prompt.input_ids) > 0
        assert prompt.embeddings_bf16.shape == (prompt.prompt_length, model_map.config.hidden_size)
        expected_bits = float_array_to_bf16_bits(
            gguf_q8_0_embedding(np.asarray(prompt.input_ids, dtype=np.int64), raw)
        )
        np.testing.assert_array_equal(prompt.embeddings_bf16, expected_bits)
        assert memory_stats()["current_allocated_bytes"] == token_tensor.nbytes
        assert memory_stats()["active_allocations"] == 1
    finally:
        session.free(runtime=runtime)

    assert memory_stats()["current_allocated_bytes"] == 0
    if not had_torch:
        assert "torch" not in sys.modules


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_projects_real_q3_and_q5_layer_weights() -> None:
    had_torch = "torch" in sys.modules
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    q3_tensor = model_map.layer(0).tensor("attn_q")
    q5_tensor = model_map.layer(0).tensor("attn_output")
    assert q3_tensor.ggml_type_name == "Q3_K"
    assert q5_tensor.ggml_type_name == "Q5_K"
    q3_out_features, q3_in_features = (int(dim) for dim in q3_tensor.shape)
    q5_out_features, q5_in_features = (int(dim) for dim in q5_tensor.shape)
    runtime = get_hip_runtime()
    reset_memory_stats()
    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=("layers.0.attn_q", "layers.0.attn_output"),
        runtime=runtime,
    )
    try:
        x_q3 = (
            (np.arange(2 * q3_in_features, dtype=np.float32).reshape(2, q3_in_features) % 31)
            - 15
        ) / 128.0
        x_q3_bits = float_array_to_bf16_bits(x_q3)
        raw_q3 = GGUFReader(q3_tensor.source_path).tensor_data(q3_tensor.name)
        expected_q3 = gguf_q3_k_gemv(bf16_to_float32(x_q3_bits), raw_q3)
        actual_q3 = session.linear_slot_bf16("layers.0.attn_q", x_q3_bits, runtime=runtime)
        assert actual_q3.shape == expected_q3.shape == (2, q3_out_features)
        assert actual_q3.dtype == np.float32
        np.testing.assert_allclose(actual_q3, expected_q3, rtol=2.0e-3, atol=2.0e-3)

        x_q5 = (
            (np.arange(2 * q5_in_features, dtype=np.float32).reshape(2, q5_in_features) % 37)
            - 18
        ) / 96.0
        x_q5_bits = float_array_to_bf16_bits(x_q5)
        raw_q5 = GGUFReader(q5_tensor.source_path).tensor_data(q5_tensor.name)
        expected_q5 = gguf_q5_k_gemv(bf16_to_float32(x_q5_bits), raw_q5)
        actual_q5 = session.linear_slot_bf16("layers.0.attn_output", x_q5_bits, runtime=runtime)
        assert actual_q5.shape == expected_q5.shape == (2, q5_out_features)
        assert actual_q5.dtype == np.float32
        np.testing.assert_allclose(actual_q5, expected_q5, rtol=2.0e-3, atol=2.0e-3)

        expected_nbytes = q3_tensor.nbytes + q5_tensor.nbytes
        assert session.weights.allocated_nbytes == expected_nbytes
        stats = memory_stats()
        assert stats["current_allocated_bytes"] == expected_nbytes
        assert stats["active_allocations"] == 2
    finally:
        session.free(runtime=runtime)

    stats = memory_stats()
    assert stats["current_allocated_bytes"] == 0
    assert stats["active_allocations"] == 0
    if not had_torch:
        assert "torch" not in sys.modules


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_projects_attention_inputs() -> None:
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    tensors = {
        "q": model_map.layer(0).tensor("attn_q"),
        "k": model_map.layer(0).tensor("attn_k"),
        "v": model_map.layer(0).tensor("attn_v"),
        "gate": model_map.layer(0).tensor("attn_gate"),
    }
    assert {name: tensor.ggml_type_name for name, tensor in tensors.items()} == {
        "q": "Q3_K",
        "k": "Q3_K",
        "v": "Q5_K",
        "gate": "Q3_K",
    }
    x = ((np.arange(4096, dtype=np.float32).reshape(1, 4096) % 29) - 14) / 112.0
    x_bits = float_array_to_bf16_bits(x)
    x_bf16 = bf16_to_float32(x_bits)
    expected = {}
    for name, tensor in tensors.items():
        raw = GGUFReader(tensor.source_path).tensor_data(tensor.name)
        ref_fn = gguf_q5_k_gemv if tensor.ggml_type_name == "Q5_K" else gguf_q3_k_gemv
        expected[name] = ref_fn(x_bf16, raw)
    runtime = get_hip_runtime()
    reset_memory_stats()
    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=(
            "layers.0.attn_q",
            "layers.0.attn_k",
            "layers.0.attn_v",
            "layers.0.attn_gate",
        ),
        runtime=runtime,
    )
    try:
        actual = session.project_attention_inputs_bf16(0, x_bits, runtime=runtime)
        assert set(actual) == {"q", "k", "v", "gate"}
        for name, expected_value in expected.items():
            assert actual[name].shape == expected_value.shape
            np.testing.assert_allclose(actual[name], expected_value, rtol=2.0e-3, atol=2.0e-3)
        expected_nbytes = sum(tensor.nbytes for tensor in tensors.values())
        assert session.weights.allocated_nbytes == expected_nbytes
        assert memory_stats()["current_allocated_bytes"] == expected_nbytes
        assert memory_stats()["active_allocations"] == 4
    finally:
        session.free(runtime=runtime)

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_projects_dense_mlp_inputs() -> None:
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    tensors = {
        "gate": model_map.layer(0).tensor("ffn_gate"),
        "up": model_map.layer(0).tensor("ffn_up"),
    }
    assert model_map.layer(0).mlp_type == "dense_mlp"
    assert {name: tensor.ggml_type_name for name, tensor in tensors.items()} == {
        "gate": "Q3_K",
        "up": "Q3_K",
    }
    x = ((np.arange(2 * 4096, dtype=np.float32).reshape(2, 4096) % 41) - 20) / 144.0
    x_bits = float_array_to_bf16_bits(x)
    x_bf16 = bf16_to_float32(x_bits)
    expected = {}
    for name, tensor in tensors.items():
        raw = GGUFReader(tensor.source_path).tensor_data(tensor.name)
        expected[name] = gguf_q3_k_gemv(x_bf16, raw)
    runtime = get_hip_runtime()
    reset_memory_stats()
    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=("layers.0.ffn_gate", "layers.0.ffn_up"),
        runtime=runtime,
    )
    try:
        actual = session.project_dense_mlp_inputs_bf16(0, x_bits, runtime=runtime)
        assert set(actual) == {"gate", "up"}
        for name, expected_value in expected.items():
            assert actual[name].shape == expected_value.shape == (2, 11264)
            np.testing.assert_allclose(actual[name], expected_value, rtol=2.0e-3, atol=2.0e-3)
        expected_nbytes = sum(tensor.nbytes for tensor in tensors.values())
        assert session.weights.allocated_nbytes == expected_nbytes
        assert memory_stats()["current_allocated_bytes"] == expected_nbytes
        assert memory_stats()["active_allocations"] == 2
    finally:
        session.free(runtime=runtime)

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_moe_router_probe_matches_cpu_reference() -> None:
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    router_tensor = model_map.layer(3).tensor("ffn_gate_inp")
    bias_tensor = model_map.layer(3).tensor("exp_probs_bias")
    assert model_map.layer(3).mlp_type == "moe"
    assert router_tensor.ggml_type_name == "F32"
    assert bias_tensor.ggml_type_name == "F32"
    x = ((np.arange(2 * 4096, dtype=np.float32).reshape(2, 4096) % 47) - 23) / 160.0
    x_bits = float_array_to_bf16_bits(x)
    x_bf16 = bf16_to_float32(x_bits)
    router_weight = GGUFReader(router_tensor.source_path).tensor_data(router_tensor.name)
    router_bias = GGUFReader(bias_tensor.source_path).tensor_data(bias_tensor.name)
    expected_weights, expected_experts, expected_logits = step_moe_router(
        x_bf16,
        router_weight,
        router_bias=router_bias,
        top_k=model_map.config.expert_used_count,
        routing_scale=model_map.config.expert_weights_scale,
        normalize_selected=model_map.config.expert_weights_norm,
    )
    runtime = get_hip_runtime()
    reset_memory_stats()
    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=("layers.3.ffn_gate_inp", "layers.3.exp_probs_bias"),
        runtime=runtime,
    )
    try:
        actual = session.moe_router_probe_bf16(3, x_bits, runtime=runtime)
        assert actual.routing_weights.shape == (2, model_map.config.expert_used_count)
        assert actual.selected_experts.shape == (2, model_map.config.expert_used_count)
        assert actual.logits.shape == (2, model_map.config.expert_count)
        np.testing.assert_array_equal(actual.selected_experts, expected_experts)
        np.testing.assert_allclose(actual.routing_weights, expected_weights, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actual.logits, expected_logits, rtol=0.0, atol=0.0)
        expected_nbytes = router_tensor.nbytes + bias_tensor.nbytes
        assert session.weights.allocated_nbytes == expected_nbytes
        assert memory_stats()["current_allocated_bytes"] == expected_nbytes
        assert memory_stats()["active_allocations"] == 2
    finally:
        session.free(runtime=runtime)

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_selected_expert_gate_projection_matches_cpu_reference() -> None:
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    tensor = model_map.layer(3).tensor("ffn_gate_exps")
    assert tensor.ggml_type_name == "Q3_K"
    x = ((np.arange(2 * 4096, dtype=np.float32).reshape(2, 4096) % 53) - 26) / 192.0
    x_bits = float_array_to_bf16_bits(x)
    x_bf16 = bf16_to_float32(x_bits)
    selected = np.asarray([0, 7, 13, 21, 2, 5, 8, 11], dtype=np.int64)
    raw = GGUFReader(tensor.source_path).tensor_data(tensor.name)
    expected_f32 = np.empty((selected.size, 1280), dtype=np.float32)
    rows_per_x = selected.size // x_bf16.shape[0]
    for row, expert_id in enumerate(selected):
        x_row = row // rows_per_x
        expected_f32[row] = gguf_q3_k_gemv(x_bf16[x_row : x_row + 1], raw[int(expert_id)])[0]
    expected_bits = float_array_to_bf16_bits(expected_f32)
    runtime = get_hip_runtime()
    reset_memory_stats()
    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=("layers.3.ffn_gate_exps",),
        runtime=runtime,
    )
    try:
        actual_bits = session.selected_expert_linear_bf16(
            "layers.3.ffn_gate_exps",
            x_bits,
            selected,
            runtime=runtime,
        )
        assert actual_bits.shape == expected_bits.shape == (selected.size, 1280)
        np.testing.assert_allclose(
            bf16_to_float32(actual_bits),
            bf16_to_float32(expected_bits),
            rtol=2.0e-3,
            atol=2.0e-3,
        )
        assert session.weights.allocated_nbytes == tensor.nbytes
        assert memory_stats()["current_allocated_bytes"] == tensor.nbytes
        assert memory_stats()["active_allocations"] == 1
    finally:
        session.free(runtime=runtime)

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_projects_moe_expert_inputs() -> None:
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    tensors = {
        "expert_gate": model_map.layer(3).tensor("ffn_gate_exps"),
        "expert_up": model_map.layer(3).tensor("ffn_up_exps"),
        "shared_gate": model_map.layer(3).tensor("ffn_gate_shexp"),
        "shared_up": model_map.layer(3).tensor("ffn_up_shexp"),
    }
    assert {name: tensor.ggml_type_name for name, tensor in tensors.items()} == {
        "expert_gate": "Q3_K",
        "expert_up": "Q3_K",
        "shared_gate": "Q3_K",
        "shared_up": "Q3_K",
    }
    x = ((np.arange(2 * 4096, dtype=np.float32).reshape(2, 4096) % 59) - 29) / 224.0
    x_bits = float_array_to_bf16_bits(x)
    x_bf16 = bf16_to_float32(x_bits)
    selected = np.asarray([0, 7, 13, 21, 2, 5, 8, 11], dtype=np.int64)
    rows_per_x = selected.size // x_bf16.shape[0]
    expected: dict[str, np.ndarray] = {}
    for name in ("expert_gate", "expert_up"):
        raw = GGUFReader(tensors[name].source_path).tensor_data(tensors[name].name)
        out = np.empty((selected.size, 1280), dtype=np.float32)
        for row, expert_id in enumerate(selected):
            x_row = row // rows_per_x
            out[row] = gguf_q3_k_gemv(x_bf16[x_row : x_row + 1], raw[int(expert_id)])[0]
        expected[name] = float_array_to_bf16_bits(out)
    for name in ("shared_gate", "shared_up"):
        raw = GGUFReader(tensors[name].source_path).tensor_data(tensors[name].name)
        expected[name] = float_array_to_bf16_bits(gguf_q3_k_gemv(x_bf16, raw))
    runtime = get_hip_runtime()
    reset_memory_stats()
    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=(
            "layers.3.ffn_gate_exps",
            "layers.3.ffn_up_exps",
            "layers.3.ffn_gate_shexp",
            "layers.3.ffn_up_shexp",
        ),
        runtime=runtime,
    )
    try:
        actual = session.project_moe_expert_inputs_bf16(3, x_bits, selected, runtime=runtime)
        assert set(actual) == {"expert_gate", "expert_up", "shared_gate", "shared_up"}
        assert actual["expert_gate"].shape == actual["expert_up"].shape == (selected.size, 1280)
        assert actual["shared_gate"].shape == actual["shared_up"].shape == (2, 1280)
        for name, expected_bits in expected.items():
            np.testing.assert_allclose(
                bf16_to_float32(actual[name]),
                bf16_to_float32(expected_bits),
                rtol=2.0e-3,
                atol=2.0e-3,
            )
        expected_nbytes = sum(tensor.nbytes for tensor in tensors.values())
        assert session.weights.allocated_nbytes == expected_nbytes
        assert memory_stats()["current_allocated_bytes"] == expected_nbytes
        assert memory_stats()["active_allocations"] == 4
    finally:
        session.free(runtime=runtime)

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_moe_mlp_probe_matches_cpu_reference() -> None:
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    layer = model_map.layer(3)
    tensors = {
        "router": layer.tensor("ffn_gate_inp"),
        "bias": layer.tensor("exp_probs_bias"),
        "expert_gate": layer.tensor("ffn_gate_exps"),
        "expert_up": layer.tensor("ffn_up_exps"),
        "expert_down": layer.tensor("ffn_down_exps"),
        "shared_gate": layer.tensor("ffn_gate_shexp"),
        "shared_up": layer.tensor("ffn_up_shexp"),
        "shared_down": layer.tensor("ffn_down_shexp"),
    }
    assert tensors["expert_down"].ggml_type_name == "Q5_K"
    assert tensors["shared_down"].ggml_type_name == "Q5_K"
    x = ((np.arange(2 * 4096, dtype=np.float32).reshape(2, 4096) % 61) - 30) / 240.0
    x_bits = float_array_to_bf16_bits(x)
    x_bf16 = bf16_to_float32(x_bits)
    router_weight = GGUFReader(tensors["router"].source_path).tensor_data(tensors["router"].name)
    router_bias = GGUFReader(tensors["bias"].source_path).tensor_data(tensors["bias"].name)
    routing, selected, _ = step_moe_router(
        x_bf16,
        router_weight,
        router_bias=router_bias,
        top_k=model_map.config.expert_used_count,
        routing_scale=model_map.config.expert_weights_scale,
        normalize_selected=model_map.config.expert_weights_norm,
    )
    selected_flat = selected.reshape(-1)
    rows_per_x = selected_flat.size // x_bf16.shape[0]
    expert_gate_raw = GGUFReader(tensors["expert_gate"].source_path).tensor_data(tensors["expert_gate"].name)
    expert_up_raw = GGUFReader(tensors["expert_up"].source_path).tensor_data(tensors["expert_up"].name)
    expert_down_raw = GGUFReader(tensors["expert_down"].source_path).tensor_data(tensors["expert_down"].name)
    expert_gate = np.empty((selected_flat.size, 1280), dtype=np.float32)
    expert_up = np.empty_like(expert_gate)
    for row, expert_id in enumerate(selected_flat):
        x_row = row // rows_per_x
        expert_gate[row] = gguf_q3_k_gemv(x_bf16[x_row : x_row + 1], expert_gate_raw[int(expert_id)])[0]
        expert_up[row] = gguf_q3_k_gemv(x_bf16[x_row : x_row + 1], expert_up_raw[int(expert_id)])[0]
    expert_gate = bf16_to_float32(float_array_to_bf16_bits(expert_gate))
    expert_up = bf16_to_float32(float_array_to_bf16_bits(expert_up))
    expert_fused = expert_gate / (np.float32(1.0) + np.exp(-expert_gate).astype(np.float32)) * expert_up
    expert_fused_bits = float_array_to_bf16_bits(expert_fused)
    expert_down = np.empty((selected_flat.size, 4096), dtype=np.float32)
    for row, expert_id in enumerate(selected_flat):
        expert_down[row] = gguf_q5_k_gemv(
            bf16_to_float32(expert_fused_bits[row : row + 1]),
            expert_down_raw[int(expert_id)],
        )[0]
    expert_down = bf16_to_float32(float_array_to_bf16_bits(expert_down)).reshape(2, -1, 4096)
    expected = np.sum(expert_down * routing[..., None], axis=1, dtype=np.float32)

    shared_gate = gguf_q3_k_gemv(
        x_bf16,
        GGUFReader(tensors["shared_gate"].source_path).tensor_data(tensors["shared_gate"].name),
    )
    shared_up = gguf_q3_k_gemv(
        x_bf16,
        GGUFReader(tensors["shared_up"].source_path).tensor_data(tensors["shared_up"].name),
    )
    shared_gate = bf16_to_float32(float_array_to_bf16_bits(shared_gate))
    shared_up = bf16_to_float32(float_array_to_bf16_bits(shared_up))
    shared_fused = shared_gate / (np.float32(1.0) + np.exp(-shared_gate).astype(np.float32)) * shared_up
    shared_down = gguf_q5_k_gemv(
        bf16_to_float32(float_array_to_bf16_bits(shared_fused)),
        GGUFReader(tensors["shared_down"].source_path).tensor_data(tensors["shared_down"].name),
    )
    expected += bf16_to_float32(float_array_to_bf16_bits(shared_down))

    runtime = get_hip_runtime()
    reset_memory_stats()
    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=(
            "layers.3.ffn_gate_inp",
            "layers.3.exp_probs_bias",
            "layers.3.ffn_gate_exps",
            "layers.3.ffn_up_exps",
            "layers.3.ffn_down_exps",
            "layers.3.ffn_gate_shexp",
            "layers.3.ffn_up_shexp",
            "layers.3.ffn_down_shexp",
        ),
        runtime=runtime,
    )
    try:
        actual = session.moe_mlp_probe_bf16(3, x_bits, runtime=runtime)
        assert actual.shape == expected.shape == (2, 4096)
        np.testing.assert_allclose(actual, expected, rtol=2.5e-3, atol=2.5e-3)
        expected_nbytes = sum(tensor.nbytes for tensor in tensors.values())
        assert session.weights.allocated_nbytes == expected_nbytes
        assert memory_stats()["current_allocated_bytes"] == expected_nbytes
        assert memory_stats()["active_allocations"] == len(tensors)
    finally:
        session.free(runtime=runtime)

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_dense_mlp_probe_matches_cpu_reference() -> None:
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    tensors = {
        "gate": model_map.layer(0).tensor("ffn_gate"),
        "up": model_map.layer(0).tensor("ffn_up"),
        "down": model_map.layer(0).tensor("ffn_down"),
    }
    assert {name: tensor.ggml_type_name for name, tensor in tensors.items()} == {
        "gate": "Q3_K",
        "up": "Q3_K",
        "down": "Q5_K",
    }
    x = ((np.arange(4096, dtype=np.float32).reshape(1, 4096) % 43) - 21) / 192.0
    x_bits = float_array_to_bf16_bits(x)
    x_bf16 = bf16_to_float32(x_bits)
    raw_gate = GGUFReader(tensors["gate"].source_path).tensor_data(tensors["gate"].name)
    raw_up = GGUFReader(tensors["up"].source_path).tensor_data(tensors["up"].name)
    raw_down = GGUFReader(tensors["down"].source_path).tensor_data(tensors["down"].name)
    gate = gguf_q3_k_gemv(x_bf16, raw_gate)
    up = gguf_q3_k_gemv(x_bf16, raw_up)
    fused = gate / (np.float32(1.0) + np.exp(-gate).astype(np.float32)) * up
    fused_bits = float_array_to_bf16_bits(fused)
    expected = gguf_q5_k_gemv(bf16_to_float32(fused_bits), raw_down)
    runtime = get_hip_runtime()
    reset_memory_stats()
    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=("layers.0.ffn_gate", "layers.0.ffn_up", "layers.0.ffn_down"),
        runtime=runtime,
    )
    try:
        actual = session.dense_mlp_probe_bf16(0, x_bits, runtime=runtime)
        assert actual.shape == expected.shape == (1, 4096)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-3, atol=2.0e-3)
        expected_nbytes = sum(tensor.nbytes for tensor in tensors.values())
        assert session.weights.allocated_nbytes == expected_nbytes
        assert memory_stats()["current_allocated_bytes"] == expected_nbytes
        assert memory_stats()["active_allocations"] == 3
    finally:
        session.free(runtime=runtime)

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_rejects_dense_mlp_projection_for_moe_layer() -> None:
    runtime = get_hip_runtime()
    session = StepFunResidentSession.from_gguf_paths(
        _stepfun_gguf_paths(),
        selected_slots=("layers.3.ffn_gate_inp",),
        runtime=runtime,
    )
    try:
        x_bits = float_array_to_bf16_bits(np.zeros((1, 4096), dtype=np.float32))
        with pytest.raises(RuntimeError, match="dense ffn_gate/ffn_up"):
            session.project_dense_mlp_inputs_bf16(3, x_bits, runtime=runtime)
    finally:
        session.free(runtime=runtime)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_embedding_is_torch_free() -> None:
    had_torch = "torch" in sys.modules
    runtime = get_hip_runtime()
    session = StepFunResidentSession.from_gguf_paths(
        _stepfun_gguf_paths(),
        selected_slots=("root.token_embedding",),
        runtime=runtime,
    )
    try:
        session.embed_token_ids_bf16([0], runtime=runtime)
    finally:
        session.free(runtime=runtime)

    if not had_torch:
        assert "torch" not in sys.modules


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_allocates_and_frees_kv_cache() -> None:
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    weight_nbytes = model_map.root("output_norm").nbytes
    page_size = 16
    context_pages = 1
    expected_kv_nbytes = sum(
        context_pages
        * page_size
        * kv_heads
        * (model_map.config.head_dim + model_map.config.value_dim)
        * 2
        for kv_heads in model_map.config.kv_head_counts
    )
    runtime = get_hip_runtime()
    reset_memory_stats()
    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=("root.output_norm",),
        runtime=runtime,
    )
    kv = None
    try:
        kv = session.allocate_kv_cache(
            context_pages=context_pages,
            page_size=page_size,
            runtime=runtime,
        )
        assert kv.tokens == page_size
        assert kv.buffer_count == model_map.config.block_count * 2
        assert kv.nbytes == expected_kv_nbytes
        assert len(kv.layer_nbytes) == model_map.config.block_count
        stats = memory_stats()
        assert stats["current_allocated_bytes"] == weight_nbytes + expected_kv_nbytes
        assert stats["active_allocations"] == 1 + kv.buffer_count
        kv.free(runtime=runtime)
        kv = None
        stats = memory_stats()
        assert stats["current_allocated_bytes"] == weight_nbytes
        assert stats["active_allocations"] == 1
    finally:
        if kv is not None:
            kv.free(runtime=runtime)
        session.free(runtime=runtime)

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_final_logits_probe_matches_cpu_rows() -> None:
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    norm_tensor = model_map.root("output_norm")
    head_tensor = model_map.root("lm_head")
    assert norm_tensor.ggml_type_name == "F32"
    assert head_tensor.ggml_type_name == "Q8_0"
    x = ((np.arange(4096, dtype=np.float32).reshape(1, 4096) % 67) - 33) / 256.0
    x_bits = float_array_to_bf16_bits(x)
    hidden = bf16_to_float32(x_bits)
    norm_weight = GGUFReader(norm_tensor.source_path).tensor_data(norm_tensor.name)
    normed = step_rmsnorm(hidden, norm_weight, eps=model_map.config.rms_norm_eps)
    normed_bits = float_array_to_bf16_bits(normed)
    head_rows = np.asarray([0, 1, 128007, model_map.config.vocab_size - 1], dtype=np.int64)
    head_raw = GGUFReader(head_tensor.source_path).tensor_data(head_tensor.name)
    expected_rows = gguf_q8_0_gemv(bf16_to_float32(normed_bits), head_raw[head_rows])
    runtime = get_hip_runtime()
    reset_memory_stats()
    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=("root.output_norm", "root.lm_head"),
        runtime=runtime,
    )
    try:
        logits = session.final_logits_probe_bf16(x_bits, runtime=runtime)
        assert logits.shape == (1, model_map.config.vocab_size)
        np.testing.assert_allclose(
            logits[:, head_rows],
            expected_rows,
            rtol=2.0e-3,
            atol=2.0e-3,
        )
        expected_nbytes = norm_tensor.nbytes + head_tensor.nbytes
        assert session.weights.allocated_nbytes == expected_nbytes
        assert memory_stats()["current_allocated_bytes"] == expected_nbytes
        assert memory_stats()["active_allocations"] == 2
    finally:
        session.free(runtime=runtime)

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_requires_resident_embedding_weight() -> None:
    runtime = get_hip_runtime()
    session = StepFunResidentSession.from_gguf_paths(
        _stepfun_gguf_paths(),
        selected_slots=("root.output_norm",),
        runtime=runtime,
    )
    try:
        with pytest.raises(RuntimeError, match="token_embedding"):
            session.embed_token_ids_bf16([0], runtime=runtime)
    finally:
        session.free(runtime=runtime)
