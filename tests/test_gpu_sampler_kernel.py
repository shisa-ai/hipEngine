from __future__ import annotations

import os
import pathlib
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.sampling import (
    apply_processors_f32_rows,
    plan_sampler_build,
    register_sampler_kernels,
    sample_temperature_f32_rows_i32,
    sample_temperature_top_logprobs_f32_rows_i32,
    sample_top_p_temperature_f32_rows_i32,
    sample_topk_temperature_f32_rows_i32,
)
from hipengine.kernels.registry import resolve


_NEG_INF = np.float32(-3.4028234663852886e38)
_MASK64 = (1 << 64) - 1
_SPLITMIX_INC = 0x9E3779B97F4A7C15
_SPLITMIX_MUL1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL2 = 0x94D049BB133111EB


def _has_gfx1100() -> bool:
    try:
        from hipengine.core.hip import get_hip_runtime
    except Exception:
        return False
    try:
        get_hip_runtime()
        return True
    except Exception:
        return False


def test_sampler_build_plan_uses_native_arch(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")

    plan = plan_sampler_build(compiler_version="hipcc:test")

    assert "--offload-arch=gfx1151" in plan.command
    assert plan.target_arch == "gfx1151"
    assert plan.output_path.name == "sampler.so"


def test_sampler_registers_for_gfx1151_alias() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_sampler_kernels(replace=True)
    register_gfx1151_kernels(replace=True)

    assert (
        resolve(backend="hip_gfx1151", layer="sampler", quant="f32", variant="processors_rows")
        is apply_processors_f32_rows
    )
    assert (
        resolve(backend="hip_gfx1151", layer="sampler", quant="f32", variant="temperature_rows_i32")
        is sample_temperature_f32_rows_i32
    )
    assert (
        resolve(backend="hip_gfx1151", layer="sampler", quant="f32", variant="temperature_top_logprobs_rows_i32")
        is sample_temperature_top_logprobs_f32_rows_i32
    )
    assert (
        resolve(backend="hip_gfx1151", layer="sampler", quant="f32", variant="top_p_temperature_rows_i32")
        is sample_top_p_temperature_f32_rows_i32
    )
    assert (
        resolve(backend="hip_gfx1151", layer="sampler", quant="f32", variant="topk_temperature_rows_i32")
        is sample_topk_temperature_f32_rows_i32
    )


def test_sampler_wrapper_validates_shapes_before_loading_hip() -> None:
    with pytest.raises(ValueError, match="rows"):
        apply_processors_f32_rows(0, 0, 0, None, None, 0, None, None, 0, 0, 0, rows=0, vocab_size=16)
    with pytest.raises(ValueError, match="vocab_size"):
        apply_processors_f32_rows(0, 0, 0, None, None, 0, None, None, 0, 0, 0, rows=1, vocab_size=0)
    with pytest.raises(ValueError, match="threads"):
        apply_processors_f32_rows(0, 0, 0, None, None, 0, None, None, 0, 0, 0, rows=1, vocab_size=16, threads=256)
    with pytest.raises(ValueError, match="rows"):
        sample_temperature_f32_rows_i32(0, 0, 0, 0, None, rows=0, vocab_size=16)
    with pytest.raises(ValueError, match="vocab_size"):
        sample_temperature_f32_rows_i32(0, 0, 0, 0, None, rows=1, vocab_size=0)
    with pytest.raises(ValueError, match="threads"):
        sample_temperature_f32_rows_i32(0, 0, 0, 0, None, rows=1, vocab_size=16, threads=256)
    with pytest.raises(ValueError, match="step_index"):
        sample_temperature_f32_rows_i32(0, 0, 0, 0, None, rows=1, vocab_size=16, step_index=-1)
    with pytest.raises(ValueError, match="top_logprobs"):
        sample_temperature_top_logprobs_f32_rows_i32(0, 0, 0, 0, rows=1, vocab_size=16, top_logprobs=0)
    with pytest.raises(ValueError, match="top_logprobs"):
        sample_temperature_top_logprobs_f32_rows_i32(0, 0, 0, 0, rows=1, vocab_size=16, top_logprobs=65)
    with pytest.raises(ValueError, match="rows"):
        sample_top_p_temperature_f32_rows_i32(0, 0, 0, 0, 0, 0, None, None, rows=0, vocab_size=16)
    with pytest.raises(ValueError, match="vocab_size"):
        sample_top_p_temperature_f32_rows_i32(0, 0, 0, 0, 0, 0, None, None, rows=1, vocab_size=0)
    with pytest.raises(ValueError, match="threads"):
        sample_top_p_temperature_f32_rows_i32(0, 0, 0, 0, 0, 0, None, None, rows=1, vocab_size=16, threads=256)
    with pytest.raises(ValueError, match="step_index"):
        sample_top_p_temperature_f32_rows_i32(0, 0, 0, 0, 0, 0, None, None, rows=1, vocab_size=16, step_index=-1)
    with pytest.raises(ValueError, match="rows"):
        sample_topk_temperature_f32_rows_i32(0, 0, 0, 0, None, None, None, rows=0, vocab_size=16, top_k=4)
    with pytest.raises(ValueError, match="vocab_size"):
        sample_topk_temperature_f32_rows_i32(0, 0, 0, 0, None, None, None, rows=1, vocab_size=0, top_k=4)
    with pytest.raises(ValueError, match="top_k"):
        sample_topk_temperature_f32_rows_i32(0, 0, 0, 0, None, None, None, rows=1, vocab_size=16, top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        sample_topk_temperature_f32_rows_i32(0, 0, 0, 0, None, None, None, rows=1, vocab_size=16, top_k=65)
    with pytest.raises(ValueError, match="threads"):
        sample_topk_temperature_f32_rows_i32(0, 0, 0, 0, None, None, None, rows=1, vocab_size=16, top_k=4, threads=256)
    with pytest.raises(ValueError, match="step_index"):
        sample_topk_temperature_f32_rows_i32(0, 0, 0, 0, None, None, None, rows=1, vocab_size=16, top_k=4, step_index=-1)


def test_native_sampler_route_falls_back_to_host_for_forced_tokens() -> None:
    from hipengine.generation.sampling import RowSamplingState
    from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession

    session = object.__new__(Qwen35ParoResidentSession)
    state = RowSamplingState(forced_tokens_pending=(7,), forced_token_reason="grammar")
    params = _request_params(temperature=0.7)
    hidden = SimpleNamespace(ptr=123)
    calls: list[tuple[object, object, RowSamplingState]] = []

    def fake_host_sample(self, hidden_arg, params_arg, state_arg):
        calls.append((hidden_arg, params_arg, state_arg))
        return SimpleNamespace(token_id=7)

    session._native_sampling_params = params
    session._native_sampling_state = state
    session._host_sampling_params = None
    session._host_sampling_state = None
    session._host_sampling_states_by_slot = None
    session._sample_from_hidden_host = MethodType(fake_host_sample, session)

    result = session._sample_from_hidden(hidden)

    assert result.token_id == 7
    assert calls == [(hidden, params, state)]


def test_native_sampler_rows_route_by_slot_and_clear_host_state() -> None:
    from hipengine.generation.sampling import RowSamplingState
    from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession

    session = object.__new__(Qwen35ParoResidentSession)
    params = _request_params(temperature=0.7)
    hidden = SimpleNamespace(ptr=456)
    native_state = RowSamplingState(prompt_tokens=(1,), generated_tokens=(4,), seed=9)
    host_state = RowSamplingState(prompt_tokens=(2,), generated_tokens=(5,), seed=10)
    calls: list[tuple[str, object, object | None, RowSamplingState | None]] = []

    def fake_native_sample(self, hidden_arg, params_arg, state_arg):
        calls.append(("native", hidden_arg, params_arg, state_arg))
        return SimpleNamespace(token_id=4)

    def fake_host_sample(self, hidden_arg, params_arg, state_arg):  # pragma: no cover - should be cleared
        calls.append(("host", hidden_arg, params_arg, state_arg))
        return SimpleNamespace(token_id=5)

    def fake_argmax_sample(self, hidden_arg):
        calls.append(("argmax", hidden_arg, None, None))
        return SimpleNamespace(token_id=6)

    session._sample_from_hidden_native = MethodType(fake_native_sample, session)
    session._sample_from_hidden_host = MethodType(fake_host_sample, session)
    session._sample_from_hidden = MethodType(fake_argmax_sample, session)
    session.configure_host_sampler_rows(params, {3: host_state})
    session.configure_native_sampler_rows(params, {3: native_state})

    result = session._sample_from_hidden_for_slot(hidden, 3)
    missing_slot_result = session._sample_from_hidden_for_slot(hidden, 4)
    session.configure_native_sampler_rows(None, None)

    assert result.token_id == 4
    assert missing_slot_result.token_id == 6
    assert session._native_sampling_params is None
    assert session._native_sampling_states_by_slot is None
    assert session._host_sampling_params is None
    assert session._host_sampling_states_by_slot is None
    assert calls == [
        ("native", hidden, params, native_state),
        ("argmax", hidden, None, None),
    ]


def _splitmix64(value: int) -> int:
    z = (value + _SPLITMIX_INC) & _MASK64
    z = ((z ^ (z >> 30)) * _SPLITMIX_MUL1) & _MASK64
    z = ((z ^ (z >> 27)) * _SPLITMIX_MUL2) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _uniform01(row_seed: int, step_index: int, row: int) -> np.float32:
    row_component = (((row + 1) & _MASK64) * _SPLITMIX_MUL1) & _MASK64
    step_component = (((step_index + 1) & _MASK64) * _SPLITMIX_INC) & _MASK64
    bits = _splitmix64((int(row_seed) ^ row_component ^ step_component) & _MASK64)
    return np.float32((bits >> 11) * (1.0 / 9007199254740992.0))


def _cpu_reference(
    logits: np.ndarray,
    temperatures: np.ndarray,
    seeds: np.ndarray,
    *,
    top_k: int,
    step_index: int,
    top_p: float = 1.0,
    min_p: float = 0.0,
):
    rows, _vocab = logits.shape
    selected = np.full((rows,), -1, dtype=np.int32)
    selected_logprobs = np.full((rows,), _NEG_INF, dtype=np.float32)
    top_indices = np.full((rows, top_k), -1, dtype=np.int32)
    top_logprobs = np.full((rows, top_k), _NEG_INF, dtype=np.float32)

    for row in range(rows):
        row_logits = logits[row]
        finite_ids = np.flatnonzero(np.isfinite(row_logits)).astype(np.int64, copy=False)
        order = np.lexsort((finite_ids, -row_logits[finite_ids]))
        candidates = finite_ids[order][: min(top_k, finite_ids.size)]
        if candidates.size == 0:
            continue
        top_indices[row, : candidates.size] = candidates.astype(np.int32)
        temp = np.float32(temperatures[row])
        if not np.isfinite(temp) or not (temp > np.float32(0.0)):
            selected[row] = np.int32(candidates[0])
            selected_logprobs[row] = np.float32(0.0)
            top_logprobs[row, 0] = np.float32(0.0)
            continue

        scaled = (row_logits[candidates].astype(np.float32) / temp).astype(np.float32)
        max_scaled = np.float32(scaled[0])
        weights = np.empty_like(scaled, dtype=np.float32)
        weight_sum = np.float32(0.0)
        for idx, value in enumerate(scaled):
            weight = np.float32(np.exp(np.float32(value - max_scaled)))
            weights[idx] = weight
            weight_sum = np.float32(weight_sum + weight)
        retained_count = candidates.size
        top_p_value = np.float32(top_p)
        if top_p_value < np.float32(1.0):
            if top_p_value <= np.float32(0.0):
                retained_count = 1
            else:
                target = np.float32(top_p_value * weight_sum)
                cumulative_full = np.float32(0.0)
                for idx, weight in enumerate(weights):
                    cumulative_full = np.float32(cumulative_full + weight)
                    if cumulative_full >= target:
                        retained_count = idx + 1
                        break
        min_p_value = np.float32(min_p)
        if min_p_value > np.float32(0.0):
            min_p_count = int(np.count_nonzero(weights[:retained_count] >= min_p_value))
            retained_count = min_p_count if min_p_count > 0 else 1
        retained_weights = weights[:retained_count]
        retained_sum = np.float32(retained_weights.sum(dtype=np.float32))
        log_denom = np.float32(np.log(retained_sum) + max_scaled)
        logprobs = (scaled[:retained_count] - log_denom).astype(np.float32)
        top_logprobs[row, :retained_count] = logprobs

        threshold = np.float32(_uniform01(int(seeds[row]), step_index, row) * retained_sum)
        cumulative = np.float32(0.0)
        selected_pos = retained_count - 1
        for idx, weight in enumerate(retained_weights):
            cumulative = np.float32(cumulative + weight)
            if threshold <= cumulative:
                selected_pos = idx
                break
        selected[row] = np.int32(candidates[selected_pos])
        selected_logprobs[row] = logprobs[selected_pos]

    return selected, selected_logprobs, top_indices, top_logprobs


def _cpu_process_reference(
    logits: np.ndarray,
    bias_offsets: np.ndarray,
    bias_token_ids: np.ndarray,
    bias_values: np.ndarray,
    history_offsets: np.ndarray,
    history_token_ids: np.ndarray,
    history_counts: np.ndarray,
    repetition_penalties: np.ndarray,
    presence_penalties: np.ndarray,
    frequency_penalties: np.ndarray,
    *,
    suppress_offsets: np.ndarray | None = None,
    suppress_token_ids: np.ndarray | None = None,
    min_tokens: np.ndarray | None = None,
    eos_token_ids: np.ndarray | None = None,
    step_indices: np.ndarray | None = None,
) -> np.ndarray:
    rows, vocab = logits.shape
    processed = logits.astype(np.float32, copy=True)
    processed[~np.isfinite(processed)] = -np.inf
    for row in range(rows):
        for item in range(int(bias_offsets[row]), int(bias_offsets[row + 1])):
            token = int(bias_token_ids[item])
            if 0 <= token < vocab:
                processed[row, token] = np.float32(processed[row, token] + np.float32(bias_values[item]))
        rep = np.float32(repetition_penalties[row])
        presence = np.float32(presence_penalties[row])
        frequency = np.float32(frequency_penalties[row])
        for item in range(int(history_offsets[row]), int(history_offsets[row + 1])):
            token = int(history_token_ids[item])
            if token < 0 or token >= vocab:
                continue
            value = np.float32(processed[row, token])
            if rep != np.float32(1.0):
                if value < np.float32(0.0):
                    value = np.float32(value * rep)
                else:
                    value = np.float32(value / rep)
            if presence != np.float32(0.0):
                value = np.float32(value - presence)
            if frequency != np.float32(0.0):
                value = np.float32(value - frequency * np.float32(history_counts[item]))
            processed[row, token] = value
        if suppress_offsets is not None and suppress_token_ids is not None:
            for item in range(int(suppress_offsets[row]), int(suppress_offsets[row + 1])):
                token = int(suppress_token_ids[item])
                if 0 <= token < vocab:
                    processed[row, token] = -np.inf
        if min_tokens is not None and eos_token_ids is not None and step_indices is not None:
            eos_token = int(eos_token_ids[row])
            if int(min_tokens[row]) > 0 and int(step_indices[row]) < int(min_tokens[row]) and 0 <= eos_token < vocab:
                processed[row, eos_token] = -np.inf
    return processed


@pytest.mark.skipif(not _has_gfx1100(), reason="gfx1100 HIP runtime not available")
def test_logits_processors_match_host_order() -> None:
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.sampling import build_sampler

    logits = np.array(
        [
            [2.0, -1.0, 0.5, np.nan, 4.0, -3.0, 0.0, 7.0],
            [1.0, 2.0, 3.0, 4.0, -5.0, np.inf, 0.25, -0.5],
            [-2.0, 0.0, 2.0, -4.0, 6.0, 8.0, 10.0, -np.inf],
        ],
        dtype=np.float32,
    )
    rows, vocab_size = logits.shape
    bias_offsets = np.array([0, 2, 2, 3], dtype=np.int32)
    bias_token_ids = np.array([0, 2, 4], dtype=np.int32)
    bias_values = np.array([1.5, -0.25, -3.0], dtype=np.float32)
    history_offsets = np.array([0, 3, 3, 6], dtype=np.int32)
    history_token_ids = np.array([0, 1, 99, 0, 4, 7], dtype=np.int32)
    history_counts = np.array([2, 1, 5, 3, 1, 2], dtype=np.int32)
    repetition_penalties = np.array([2.0, 1.0, 1.5], dtype=np.float32)
    presence_penalties = np.array([0.75, 0.0, -0.5], dtype=np.float32)
    frequency_penalties = np.array([0.25, 0.0, 0.1], dtype=np.float32)
    suppress_offsets = np.array([0, 1, 2, 2], dtype=np.int32)
    suppress_token_ids = np.array([7, 2], dtype=np.int32)
    min_tokens = np.array([1, 0, 3], dtype=np.int32)
    eos_token_ids = np.array([4, 0, 5], dtype=np.int32)
    step_indices = np.array([0, 0, 2], dtype=np.uint64)
    expected = _cpu_process_reference(
        logits,
        bias_offsets,
        bias_token_ids,
        bias_values,
        history_offsets,
        history_token_ids,
        history_counts,
        repetition_penalties,
        presence_penalties,
        frequency_penalties,
        suppress_offsets=suppress_offsets,
        suppress_token_ids=suppress_token_ids,
        min_tokens=min_tokens,
        eos_token_ids=eos_token_ids,
        step_indices=step_indices,
    )

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        lib = build_sampler(load=True, compiler_version=compiler_version)

    bufs = []

    def upload(array: np.ndarray):
        arr = np.ascontiguousarray(array)
        buf = malloc(max(arr.nbytes, 4))
        bufs.append(buf)
        copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
        return buf

    def alloc(nbytes: int):
        buf = malloc(max(nbytes, 4))
        bufs.append(buf)
        return buf

    try:
        logits_d = upload(logits)
        processed_d = alloc(logits.nbytes)
        bias_offsets_d = upload(bias_offsets)
        bias_ids_d = upload(bias_token_ids)
        bias_values_d = upload(bias_values)
        history_offsets_d = upload(history_offsets)
        history_ids_d = upload(history_token_ids)
        history_counts_d = upload(history_counts)
        repetition_d = upload(repetition_penalties)
        presence_d = upload(presence_penalties)
        frequency_d = upload(frequency_penalties)
        suppress_offsets_d = upload(suppress_offsets)
        suppress_ids_d = upload(suppress_token_ids)
        min_tokens_d = upload(min_tokens)
        eos_token_ids_d = upload(eos_token_ids)
        step_indices_d = upload(step_indices)

        apply_processors_f32_rows(
            logits_d.ptr,
            processed_d.ptr,
            bias_offsets_d.ptr,
            bias_ids_d.ptr,
            bias_values_d.ptr,
            history_offsets_d.ptr,
            history_ids_d.ptr,
            history_counts_d.ptr,
            repetition_d.ptr,
            presence_d.ptr,
            frequency_d.ptr,
            rows,
            vocab_size,
            suppress_offsets_i32_ptr=suppress_offsets_d.ptr,
            suppress_token_ids_i32_ptr=suppress_ids_d.ptr,
            min_tokens_i32_ptr=min_tokens_d.ptr,
            eos_token_ids_i32_ptr=eos_token_ids_d.ptr,
            step_indices_u64_ptr=step_indices_d.ptr,
            threads=128,
            library=lib,
        )
        observed = np.empty_like(logits)
        copy_device_to_host(host_array_ptr(observed), processed_d, observed.nbytes)

        assert np.array_equal(np.isneginf(observed), np.isneginf(expected))
        finite = np.isfinite(expected)
        np.testing.assert_allclose(observed[finite], expected[finite], rtol=0, atol=1e-6)
    finally:
        for buf in reversed(bufs):
            free(buf)


def _request_params(**overrides):
    values = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "logit_bias": (),
        "suppress_token_ids": (),
        "min_tokens": 0,
        "eos_token_id": None,
        "logprobs": True,
        "top_logprobs": 0,
        "stop_token_ids": (),
        "stop_token_sequences": (),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.skipif(not _has_gfx1100(), reason="gfx1100 HIP runtime not available")
def test_c1_paro_native_sampler_route_matches_cpu_reference_and_updates_state() -> None:
    from hipengine.core.dtype import DType
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.generation.sampling import RowSamplingState
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.sampling import build_sampler
    from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession

    logits = np.array(
        [[2.0, 0.5, 1.5, -np.inf, 4.0, 3.0, np.nan, 2.5, -1.0, 0.0, 1.0, 3.5]],
        dtype=np.float32,
    )
    vocab_size = int(logits.shape[1])
    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        lib = build_sampler(load=True, compiler_version=compiler_version)

    runtime = None
    buffers = []

    def alloc(nbytes: int):
        buf = malloc(max(int(nbytes), 4), runtime=runtime)
        buffers.append(buf)
        return buf

    session = object.__new__(Qwen35ParoResidentSession)
    try:
        from hipengine.core.hip import get_hip_runtime

        runtime = get_hip_runtime()
        session.runtime = runtime
        session.vocab_size = vocab_size
        session.buffers = buffers
        session.lm_logits = alloc(logits.nbytes)
        session.lm_out_index = alloc(DType.INT64.itemsize)
        session.lm_out_value = alloc(DType.FP32.itemsize)
        session._native_sampler_library = lib
        session._native_sampling_params = None
        session._native_sampling_state = None
        session._host_sampling_params = None
        session._host_sampling_state = None
        session._host_sampling_states_by_slot = None
        session.tokenizer = SimpleNamespace(decode=lambda ids: f"T{int(ids[0])}")
        session._project_logits_device_from_hidden = MethodType(lambda self, hidden: None, session)

        def run_case(
            params,
            state,
            expected_id: int,
            expected_logprob: np.float32,
            expected_logit: np.float32,
            expected_top_logprobs: tuple[tuple[int, np.float32], ...] = (),
        ):
            copy_host_to_device(session.lm_logits, host_array_ptr(logits), logits.nbytes, runtime=runtime)
            session.configure_native_sampler(params, state)
            result = session._sample_from_hidden(SimpleNamespace())
            observed_index = np.empty((1,), dtype=np.int64)
            observed_value = np.empty((1,), dtype=np.float32)
            copy_device_to_host(host_array_ptr(observed_index), session.lm_out_index, runtime=runtime)
            copy_device_to_host(host_array_ptr(observed_value), session.lm_out_value, runtime=runtime)

            assert result.token_id == int(expected_id)
            assert result.token_text == f"T{int(expected_id)}"
            np.testing.assert_allclose(np.array([result.logprob], dtype=np.float32), np.array([expected_logprob], dtype=np.float32), rtol=0, atol=2e-5)
            np.testing.assert_allclose(np.array([result.logit], dtype=np.float32), np.array([expected_logit], dtype=np.float32), rtol=0, atol=1e-6)
            assert tuple(token_id for token_id, _logprob in result.top_logprobs) == tuple(
                int(token_id) for token_id, _logprob in expected_top_logprobs
            )
            np.testing.assert_allclose(
                np.array([logprob for _token_id, logprob in result.top_logprobs], dtype=np.float32),
                np.array([logprob for _token_id, logprob in expected_top_logprobs], dtype=np.float32),
                rtol=0,
                atol=2e-5,
            )
            assert int(observed_index[0]) == int(expected_id)
            np.testing.assert_allclose(observed_value, np.array([expected_logit], dtype=np.float32), rtol=0, atol=1e-6)
            assert state.generated_tokens[-1] == int(expected_id)
            return result

        seed = 0x100
        step_index = 4
        full_params = _request_params(temperature=0.9, top_k=0)
        full_state = RowSamplingState(prompt_tokens=(0, 1), seed=seed, step_index=step_index)
        full_expected = _cpu_full_vocab_reference(
            logits,
            np.array([0.9], dtype=np.float32),
            np.array([seed], dtype=np.uint64),
            step_index=step_index,
        )
        full_first = run_case(
            full_params,
            full_state,
            int(full_expected[0][0]),
            full_expected[1][0],
            logits[0, int(full_expected[0][0])],
        )
        full_second = run_case(
            full_params,
            RowSamplingState(prompt_tokens=(0, 1), seed=seed, step_index=step_index),
            int(full_expected[0][0]),
            full_expected[1][0],
            logits[0, int(full_expected[0][0])],
        )
        assert full_second.token_id == full_first.token_id
        assert full_state.step_index == step_index + 1

        full_top_seed = 0x170
        full_top_step = 3
        full_top_params = _request_params(temperature=0.9, top_k=0, top_logprobs=3)
        full_top_state = RowSamplingState(prompt_tokens=(0,), seed=full_top_seed, step_index=full_top_step)
        full_top_expected = _cpu_full_vocab_reference(
            logits,
            np.array([0.9], dtype=np.float32),
            np.array([full_top_seed], dtype=np.uint64),
            step_index=full_top_step,
        )
        full_top_expected_top = _cpu_full_vocab_top_logprobs_reference(
            logits,
            np.array([0.9], dtype=np.float32),
            3,
        )
        run_case(
            full_top_params,
            full_top_state,
            int(full_top_expected[0][0]),
            full_top_expected[1][0],
            logits[0, int(full_top_expected[0][0])],
            tuple(
                (int(token_id), np.float32(logprob))
                for token_id, logprob in zip(
                    full_top_expected_top[0][0],
                    full_top_expected_top[1][0],
                    strict=True,
                )
            ),
        )
        assert full_top_state.step_index == full_top_step + 1

        top_logprobs_seed = 0x180
        top_logprobs_step = 2
        top_logprobs_params = _request_params(temperature=0.85, top_k=4, top_logprobs=3)
        top_logprobs_state = RowSamplingState(prompt_tokens=(0,), seed=top_logprobs_seed, step_index=top_logprobs_step)
        top_logprobs_expected = _cpu_reference(
            logits,
            np.array([0.85], dtype=np.float32),
            np.array([top_logprobs_seed], dtype=np.uint64),
            top_k=4,
            step_index=top_logprobs_step,
        )
        run_case(
            top_logprobs_params,
            top_logprobs_state,
            int(top_logprobs_expected[0][0]),
            top_logprobs_expected[1][0],
            logits[0, int(top_logprobs_expected[0][0])],
            tuple(
                (int(token_id), np.float32(logprob))
                for token_id, logprob in zip(
                    top_logprobs_expected[2][0, :3],
                    top_logprobs_expected[3][0, :3],
                    strict=True,
                )
            ),
        )
        assert top_logprobs_state.step_index == top_logprobs_step + 1

        bounded_filter_seed = 0x190
        bounded_filter_step = 1
        bounded_filter_params = _request_params(
            temperature=0.85,
            top_k=5,
            top_p=0.75,
            min_p=0.12,
            top_logprobs=4,
        )
        bounded_filter_state = RowSamplingState(prompt_tokens=(0,), seed=bounded_filter_seed, step_index=bounded_filter_step)
        bounded_filter_expected = _cpu_reference(
            logits,
            np.array([0.85], dtype=np.float32),
            np.array([bounded_filter_seed], dtype=np.uint64),
            top_k=5,
            step_index=bounded_filter_step,
            top_p=0.75,
            min_p=0.12,
        )
        bounded_expected_pairs = tuple(
            (int(token_id), np.float32(logprob))
            for token_id, logprob in zip(
                bounded_filter_expected[2][0, :4],
                bounded_filter_expected[3][0, :4],
                strict=True,
            )
            if np.isfinite(logprob)
        )
        run_case(
            bounded_filter_params,
            bounded_filter_state,
            int(bounded_filter_expected[0][0]),
            bounded_filter_expected[1][0],
            logits[0, int(bounded_filter_expected[0][0])],
            bounded_expected_pairs,
        )
        assert bounded_filter_state.step_index == bounded_filter_step + 1

        proc_seed = 0x200
        proc_step = 3
        proc_params = _request_params(
            temperature=0.75,
            top_k=4,
            logit_bias=((1, 3.0), (4, -2.0)),
            suppress_token_ids=(5,),
            min_tokens=4,
            eos_token_id=4,
            repetition_penalty=1.2,
            presence_penalty=0.25,
            frequency_penalty=0.1,
        )
        proc_state = RowSamplingState(prompt_tokens=(4, 4, 8), generated_tokens=(1,), seed=proc_seed, step_index=proc_step)
        proc_logits = _cpu_process_reference(
            logits,
            np.array([0, 2], dtype=np.int32),
            np.array([1, 4], dtype=np.int32),
            np.array([3.0, -2.0], dtype=np.float32),
            np.array([0, 3], dtype=np.int32),
            np.array([1, 4, 8], dtype=np.int32),
            np.array([1, 2, 1], dtype=np.int32),
            np.array([1.2], dtype=np.float32),
            np.array([0.25], dtype=np.float32),
            np.array([0.1], dtype=np.float32),
            suppress_offsets=np.array([0, 1], dtype=np.int32),
            suppress_token_ids=np.array([5], dtype=np.int32),
            min_tokens=np.array([4], dtype=np.int32),
            eos_token_ids=np.array([4], dtype=np.int32),
            step_indices=np.array([proc_step], dtype=np.uint64),
        )
        proc_expected = _cpu_reference(
            proc_logits,
            np.array([0.75], dtype=np.float32),
            np.array([proc_seed], dtype=np.uint64),
            top_k=4,
            step_index=proc_step,
        )
        run_case(
            proc_params,
            proc_state,
            int(proc_expected[0][0]),
            proc_expected[1][0],
            proc_logits[0, int(proc_expected[0][0])],
        )
        assert proc_state.step_index == proc_step + 1

        top_p_seed = 0x300
        top_p_step = 5
        top_p_params = _request_params(temperature=1.0, top_p=0.7, top_k=0)
        top_p_state = RowSamplingState(prompt_tokens=(2,), seed=top_p_seed, step_index=top_p_step)
        top_p_expected = _cpu_top_p_reference(
            logits,
            np.array([1.0], dtype=np.float32),
            np.array([0.7], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([top_p_seed], dtype=np.uint64),
            step_index=top_p_step,
        )
        run_case(
            top_p_params,
            top_p_state,
            int(top_p_expected[0][0]),
            top_p_expected[1][0],
            logits[0, int(top_p_expected[0][0])],
        )
        assert top_p_state.step_index == top_p_step + 1
    finally:
        for buf in reversed(buffers):
            free(buf, runtime=runtime)


def _cpu_top_p_reference(
    logits: np.ndarray,
    temperatures: np.ndarray,
    top_ps: np.ndarray,
    min_ps: np.ndarray,
    seeds: np.ndarray,
    *,
    step_index: int,
    top_logprobs: int = 0,
):
    rows, _vocab = logits.shape
    selected = np.full((rows,), -1, dtype=np.int32)
    selected_logprobs = np.full((rows,), _NEG_INF, dtype=np.float32)
    retained_counts = np.zeros((rows,), dtype=np.int32)
    top_indices = np.full((rows, max(int(top_logprobs), 1)), -1, dtype=np.int32)
    top_logprob_values = np.full((rows, max(int(top_logprobs), 1)), _NEG_INF, dtype=np.float32)

    for row in range(rows):
        finite_ids = np.flatnonzero(np.isfinite(logits[row])).astype(np.int64, copy=False)
        if finite_ids.size == 0:
            continue
        order = np.lexsort((finite_ids, -logits[row, finite_ids]))
        sorted_ids = finite_ids[order]
        temp = np.float32(temperatures[row])
        if not np.isfinite(temp) or not (temp > np.float32(0.0)):
            selected[row] = np.int32(sorted_ids[0])
            selected_logprobs[row] = np.float32(0.0)
            retained_counts[row] = np.int32(1)
            if top_logprobs > 0:
                top_indices[row, 0] = np.int32(sorted_ids[0])
                top_logprob_values[row, 0] = np.float32(0.0)
            continue
        scaled = (logits[row, sorted_ids].astype(np.float32) / temp).astype(np.float32)
        max_scaled = np.float32(scaled[0])
        weights = np.exp((scaled - max_scaled).astype(np.float32)).astype(np.float32)
        full_sum = np.float32(weights.sum(dtype=np.float32))
        top_p = np.float32(top_ps[row])
        if top_p < np.float32(1.0):
            if top_p <= np.float32(0.0):
                keep_count = 1
            else:
                keep_count = int(np.searchsorted(np.cumsum(weights / full_sum, dtype=np.float32), top_p, side="left")) + 1
                keep_count = max(1, min(keep_count, sorted_ids.size))
            sorted_ids = sorted_ids[:keep_count]
            weights = weights[:keep_count]
        min_p = np.float32(min_ps[row])
        if min_p > np.float32(0.0):
            mask = weights >= min_p
            if np.any(mask):
                sorted_ids = sorted_ids[mask]
                weights = weights[mask]
            else:
                sorted_ids = sorted_ids[:1]
                weights = weights[:1]
        retained_counts[row] = np.int32(sorted_ids.size)
        retained_sum = np.float32(weights.sum(dtype=np.float32))
        if top_logprobs > 0:
            limit = min(int(top_logprobs), int(sorted_ids.size))
            top_indices[row, :limit] = sorted_ids[:limit].astype(np.int32)
            top_logprob_values[row, :limit] = np.log(
                (weights[:limit] / retained_sum).astype(np.float32)
            ).astype(np.float32)
        draw = np.float32(_uniform01(int(seeds[row]), step_index, row) * retained_sum)
        cumulative = np.cumsum(weights, dtype=np.float32)
        choice = int(np.searchsorted(cumulative, draw, side="right"))
        if choice >= sorted_ids.size:
            choice = sorted_ids.size - 1
        selected[row] = np.int32(sorted_ids[choice])
        selected_logprobs[row] = np.float32(np.log(np.float32(weights[choice] / retained_sum)))

    if top_logprobs <= 0:
        return selected, selected_logprobs, retained_counts
    return selected, selected_logprobs, retained_counts, top_indices[:, :top_logprobs], top_logprob_values[:, :top_logprobs]


@pytest.mark.skipif(not _has_gfx1100(), reason="gfx1100 HIP runtime not available")
def test_top_p_temperature_sampler_matches_cpu_reference_and_is_deterministic() -> None:
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.sampling import build_sampler

    logits = np.array(
        [
            [3.0, 2.0, 1.0, 0.0, -1.0, np.nan, -np.inf, 0.5],
            [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
            [3.0, 2.4, 1.0, 0.0, -2.0, 2.4, -np.inf, np.nan],
            [2.0, 2.0, 1.0, -1.0, 0.0, -np.inf, np.nan, 0.5],
        ],
        dtype=np.float32,
    )
    rows, vocab_size = logits.shape
    temperatures = np.array([1.0, 0.8, 1.0, 1.0], dtype=np.float32)
    top_ps = np.array([0.7, 0.0, 1.0, 0.8], dtype=np.float32)
    min_ps = np.array([0.0, 0.0, 0.5, 0.0], dtype=np.float32)
    seeds = np.array([0x101, 0x202, 0x303, 0x404], dtype=np.uint64)
    step_index = 13
    top_logprobs = 3
    expected = _cpu_top_p_reference(
        logits,
        temperatures,
        top_ps,
        min_ps,
        seeds,
        step_index=step_index,
        top_logprobs=top_logprobs,
    )

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        lib = build_sampler(load=True, compiler_version=compiler_version)

    bufs = []

    def upload(array: np.ndarray):
        arr = np.ascontiguousarray(array)
        buf = malloc(max(arr.nbytes, 4))
        bufs.append(buf)
        copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
        return buf

    def alloc(nbytes: int):
        buf = malloc(max(nbytes, 4))
        bufs.append(buf)
        return buf

    def download(buf, shape, dtype):
        out = np.empty(shape, dtype=dtype)
        copy_device_to_host(host_array_ptr(out), buf, out.nbytes)
        return out

    try:
        logits_d = upload(logits)
        temperatures_d = upload(temperatures)
        top_ps_d = upload(top_ps)
        min_ps_d = upload(min_ps)
        seeds_d = upload(seeds)
        selected_d = alloc(rows * np.dtype(np.int32).itemsize)
        selected_logprobs_d = alloc(rows * np.dtype(np.float32).itemsize)
        retained_counts_d = alloc(rows * np.dtype(np.int32).itemsize)
        top_indices_d = alloc(rows * top_logprobs * np.dtype(np.int32).itemsize)
        top_logprobs_d = alloc(rows * top_logprobs * np.dtype(np.float32).itemsize)

        sample_top_p_temperature_f32_rows_i32(
            logits_d.ptr,
            temperatures_d.ptr,
            top_ps_d.ptr,
            min_ps_d.ptr,
            seeds_d.ptr,
            selected_d.ptr,
            selected_logprobs_d.ptr,
            retained_counts_d.ptr,
            rows,
            vocab_size,
            out_top_indices_i32_ptr=top_indices_d.ptr,
            out_top_logprobs_f32_ptr=top_logprobs_d.ptr,
            top_logprobs=top_logprobs,
            step_index=step_index,
            threads=128,
            library=lib,
        )
        first = (
            download(selected_d, (rows,), np.int32),
            download(selected_logprobs_d, (rows,), np.float32),
            download(retained_counts_d, (rows,), np.int32),
            download(top_indices_d, (rows, top_logprobs), np.int32),
            download(top_logprobs_d, (rows, top_logprobs), np.float32),
        )

        sample_top_p_temperature_f32_rows_i32(
            logits_d.ptr,
            temperatures_d.ptr,
            top_ps_d.ptr,
            min_ps_d.ptr,
            seeds_d.ptr,
            selected_d.ptr,
            selected_logprobs_d.ptr,
            retained_counts_d.ptr,
            rows,
            vocab_size,
            out_top_indices_i32_ptr=top_indices_d.ptr,
            out_top_logprobs_f32_ptr=top_logprobs_d.ptr,
            top_logprobs=top_logprobs,
            step_index=step_index,
            threads=128,
            library=lib,
        )
        second = (
            download(selected_d, (rows,), np.int32),
            download(selected_logprobs_d, (rows,), np.float32),
            download(retained_counts_d, (rows,), np.int32),
            download(top_indices_d, (rows, top_logprobs), np.int32),
            download(top_logprobs_d, (rows, top_logprobs), np.float32),
        )

        assert np.array_equal(first[0], expected[0])
        np.testing.assert_allclose(first[1], expected[1], rtol=0, atol=2e-5)
        assert np.array_equal(first[2], expected[2])
        assert np.array_equal(first[3], expected[3])
        np.testing.assert_allclose(first[4], expected[4], rtol=0, atol=2e-5)
        assert np.array_equal(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1], rtol=0, atol=0)
        assert np.array_equal(first[2], second[2])
        assert np.array_equal(first[3], second[3])
        np.testing.assert_allclose(first[4], second[4], rtol=0, atol=0)
    finally:
        for buf in reversed(bufs):
            free(buf)


def _cpu_full_vocab_reference(logits: np.ndarray, temperatures: np.ndarray, seeds: np.ndarray, *, step_index: int):
    rows, vocab = logits.shape
    selected = np.full((rows,), -1, dtype=np.int32)
    selected_logprobs = np.full((rows,), _NEG_INF, dtype=np.float32)

    for row in range(rows):
        row_logits = logits[row]
        finite_ids = np.flatnonzero(np.isfinite(row_logits)).astype(np.int64, copy=False)
        if finite_ids.size == 0:
            continue
        finite_values = row_logits[finite_ids]
        order = np.lexsort((finite_ids, -finite_values))
        argmax_id = np.int32(finite_ids[order[0]])
        max_value = np.float32(row_logits[int(argmax_id)])
        temp = np.float32(temperatures[row])
        if not np.isfinite(temp) or not (temp > np.float32(0.0)):
            selected[row] = argmax_id
            selected_logprobs[row] = np.float32(0.0)
            continue

        weights_by_id = np.zeros((vocab,), dtype=np.float32)
        weight_sum = np.float32(0.0)
        max_scaled = np.float32(max_value / temp)
        for token_id in finite_ids:
            value = np.float32(row_logits[int(token_id)] / temp - max_scaled)
            weight = np.float32(np.exp(value))
            weights_by_id[int(token_id)] = weight
            weight_sum = np.float32(weight_sum + weight)
        log_denom = np.float32(np.log(weight_sum) + max_scaled)
        threshold = np.float32(_uniform01(int(seeds[row]), step_index, row) * weight_sum)
        cumulative = np.float32(0.0)
        selected_id = argmax_id
        for token_id in range(vocab):
            weight = weights_by_id[token_id]
            if weight == np.float32(0.0):
                continue
            cumulative = np.float32(cumulative + weight)
            if threshold <= cumulative:
                selected_id = np.int32(token_id)
                break
        selected[row] = selected_id
        selected_logprobs[row] = np.float32(row_logits[int(selected_id)] / temp - log_denom)

    return selected, selected_logprobs


def _cpu_full_vocab_top_logprobs_reference(logits: np.ndarray, temperatures: np.ndarray, top_logprobs: int):
    rows, vocab = logits.shape
    top_indices = np.full((rows, top_logprobs), -1, dtype=np.int32)
    top_logprob_values = np.full((rows, top_logprobs), _NEG_INF, dtype=np.float32)

    for row in range(rows):
        row_logits = logits[row]
        finite_ids = np.flatnonzero(np.isfinite(row_logits)).astype(np.int64, copy=False)
        if finite_ids.size == 0:
            continue
        finite_values = row_logits[finite_ids]
        order = np.lexsort((finite_ids, -finite_values))
        sorted_ids = finite_ids[order]
        limit = min(int(top_logprobs), int(sorted_ids.size))
        top_indices[row, :limit] = sorted_ids[:limit].astype(np.int32)
        temp = np.float32(temperatures[row])
        if not np.isfinite(temp) or not (temp > np.float32(0.0)):
            top_logprob_values[row, 0] = np.float32(0.0)
            continue
        scaled = (row_logits[finite_ids].astype(np.float32) / temp).astype(np.float32)
        max_scaled = np.float32(np.max(scaled))
        weights = np.exp((scaled - max_scaled).astype(np.float32)).astype(np.float32)
        weight_sum = np.float32(weights.sum(dtype=np.float32))
        log_denom = np.float32(np.log(weight_sum) + max_scaled)
        top_logprob_values[row, :limit] = (
            row_logits[sorted_ids[:limit]].astype(np.float32) / temp - log_denom
        ).astype(np.float32)

    return top_indices, top_logprob_values


@pytest.mark.skipif(not _has_gfx1100(), reason="gfx1100 HIP runtime not available")
def test_temperature_sampler_matches_cpu_reference_and_is_deterministic() -> None:
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.sampling import build_sampler

    rows = 3
    vocab_size = 257
    step_index = 11
    rng = np.random.default_rng(0x7109A11)
    logits = (rng.standard_normal((rows, vocab_size), dtype=np.float32) * np.float32(0.9)).astype(np.float32)
    logits[0, 3] = np.float32(8.0)
    logits[0, 7] = np.float32(8.0)  # Argmax tie: lower id wins for temp<=0 fallback only.
    logits[1, 20] = np.float32(5.0)
    logits[1, 21] = np.float32(4.75)
    logits[2, 4] = np.float32(np.nan)
    logits[2, 9] = np.float32(-np.inf)
    temperatures = np.array([0.65, 1.2, 2.0], dtype=np.float32)
    seeds = np.array([0xAAAA_1111, 0xBBBB_2222, 0xCCCC_3333_4444], dtype=np.uint64)
    top_logprobs = 5
    expected = _cpu_full_vocab_reference(logits, temperatures, seeds, step_index=step_index)
    expected_top = _cpu_full_vocab_top_logprobs_reference(logits, temperatures, top_logprobs)

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        lib = build_sampler(load=True, compiler_version=compiler_version)

    bufs = []

    def upload(array: np.ndarray):
        arr = np.ascontiguousarray(array)
        buf = malloc(max(arr.nbytes, 4))
        bufs.append(buf)
        copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
        return buf

    def alloc(nbytes: int):
        buf = malloc(max(nbytes, 4))
        bufs.append(buf)
        return buf

    def download(buf, shape, dtype):
        out = np.empty(shape, dtype=dtype)
        copy_device_to_host(host_array_ptr(out), buf, out.nbytes)
        return out

    try:
        logits_d = upload(logits)
        temperatures_d = upload(temperatures)
        seeds_d = upload(seeds)
        selected_d = alloc(rows * np.dtype(np.int32).itemsize)
        selected_logprobs_d = alloc(rows * np.dtype(np.float32).itemsize)
        top_indices_d = alloc(rows * top_logprobs * np.dtype(np.int32).itemsize)
        top_logprobs_d = alloc(rows * top_logprobs * np.dtype(np.float32).itemsize)

        sample_temperature_f32_rows_i32(
            logits_d.ptr,
            temperatures_d.ptr,
            seeds_d.ptr,
            selected_d.ptr,
            selected_logprobs_d.ptr,
            rows,
            vocab_size,
            step_index=step_index,
            threads=128,
            library=lib,
        )
        sample_temperature_top_logprobs_f32_rows_i32(
            logits_d.ptr,
            temperatures_d.ptr,
            top_indices_d.ptr,
            top_logprobs_d.ptr,
            rows,
            vocab_size,
            top_logprobs,
            threads=128,
            library=lib,
        )
        first = (
            download(selected_d, (rows,), np.int32),
            download(selected_logprobs_d, (rows,), np.float32),
            download(top_indices_d, (rows, top_logprobs), np.int32),
            download(top_logprobs_d, (rows, top_logprobs), np.float32),
        )

        sample_temperature_f32_rows_i32(
            logits_d.ptr,
            temperatures_d.ptr,
            seeds_d.ptr,
            selected_d.ptr,
            selected_logprobs_d.ptr,
            rows,
            vocab_size,
            step_index=step_index,
            threads=128,
            library=lib,
        )
        sample_temperature_top_logprobs_f32_rows_i32(
            logits_d.ptr,
            temperatures_d.ptr,
            top_indices_d.ptr,
            top_logprobs_d.ptr,
            rows,
            vocab_size,
            top_logprobs,
            threads=128,
            library=lib,
        )
        second = (
            download(selected_d, (rows,), np.int32),
            download(selected_logprobs_d, (rows,), np.float32),
            download(top_indices_d, (rows, top_logprobs), np.int32),
            download(top_logprobs_d, (rows, top_logprobs), np.float32),
        )

        assert np.array_equal(first[0], expected[0])
        np.testing.assert_allclose(first[1], expected[1], rtol=0, atol=2e-5)
        assert np.array_equal(first[2], expected_top[0])
        np.testing.assert_allclose(first[3], expected_top[1], rtol=0, atol=2e-5)
        assert np.array_equal(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1], rtol=0, atol=0)
        assert np.array_equal(first[2], second[2])
        np.testing.assert_allclose(first[3], second[3], rtol=0, atol=0)
    finally:
        for buf in reversed(bufs):
            free(buf)


@pytest.mark.skipif(not _has_gfx1100(), reason="gfx1100 HIP runtime not available")
def test_topk_temperature_sampler_matches_cpu_reference_and_is_deterministic() -> None:
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.sampling import build_sampler

    rows = 3
    vocab_size = 257
    top_k = 16  # Deliberately beyond the older lm_head top-k helper's k<=8 cap.
    step_index = 7
    rng = np.random.default_rng(0x5A6D1E)
    logits = (rng.standard_normal((rows, vocab_size), dtype=np.float32) * np.float32(1.25)).astype(np.float32)
    logits[0, 3] = np.float32(10.0)
    logits[0, 7] = np.float32(10.0)  # Tie: lower id must sort first.
    logits[0, 11] = np.float32(9.5)
    logits[1, 20] = np.float32(6.0)
    logits[1, 21] = np.float32(5.75)
    logits[2, 5] = np.float32(np.nan)  # Non-finite logits are ignored like the host sampler.
    logits[2, 6] = np.float32(-np.inf)
    temperatures = np.array([0.7, 1.0, 1.8], dtype=np.float32)
    seeds = np.array([0x1234_5678, 0xCAFE_BABE, 0xDEAD_BEEF_1234], dtype=np.uint64)
    expected = _cpu_reference(logits, temperatures, seeds, top_k=top_k, step_index=step_index)

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        lib = build_sampler(load=True, compiler_version=compiler_version)

    bufs = []

    def upload(array: np.ndarray):
        arr = np.ascontiguousarray(array)
        buf = malloc(max(arr.nbytes, 4))
        bufs.append(buf)
        copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
        return buf

    def alloc(nbytes: int):
        buf = malloc(max(nbytes, 4))
        bufs.append(buf)
        return buf

    def download(buf, shape, dtype):
        out = np.empty(shape, dtype=dtype)
        copy_device_to_host(host_array_ptr(out), buf, out.nbytes)
        return out

    try:
        logits_d = upload(logits)
        temperatures_d = upload(temperatures)
        seeds_d = upload(seeds)
        selected_d = alloc(rows * np.dtype(np.int32).itemsize)
        selected_logprobs_d = alloc(rows * np.dtype(np.float32).itemsize)
        top_indices_d = alloc(rows * top_k * np.dtype(np.int32).itemsize)
        top_logprobs_d = alloc(rows * top_k * np.dtype(np.float32).itemsize)

        sample_topk_temperature_f32_rows_i32(
            logits_d.ptr,
            temperatures_d.ptr,
            seeds_d.ptr,
            selected_d.ptr,
            selected_logprobs_d.ptr,
            top_indices_d.ptr,
            top_logprobs_d.ptr,
            rows,
            vocab_size,
            top_k,
            step_index=step_index,
            threads=128,
            library=lib,
        )
        first = (
            download(selected_d, (rows,), np.int32),
            download(selected_logprobs_d, (rows,), np.float32),
            download(top_indices_d, (rows, top_k), np.int32),
            download(top_logprobs_d, (rows, top_k), np.float32),
        )

        # Same row seeds + step must be deterministic across launches.
        sample_topk_temperature_f32_rows_i32(
            logits_d.ptr,
            temperatures_d.ptr,
            seeds_d.ptr,
            selected_d.ptr,
            selected_logprobs_d.ptr,
            top_indices_d.ptr,
            top_logprobs_d.ptr,
            rows,
            vocab_size,
            top_k,
            step_index=step_index,
            threads=128,
            library=lib,
        )
        second = (
            download(selected_d, (rows,), np.int32),
            download(selected_logprobs_d, (rows,), np.float32),
            download(top_indices_d, (rows, top_k), np.int32),
            download(top_logprobs_d, (rows, top_k), np.float32),
        )

        assert np.array_equal(first[0], expected[0])
        np.testing.assert_allclose(first[1], expected[1], rtol=0, atol=2e-5)
        assert np.array_equal(first[2], expected[2])
        np.testing.assert_allclose(first[3], expected[3], rtol=0, atol=2e-5)
        for observed, repeated in zip(first, second, strict=True):
            if observed.dtype.kind in {"i", "u"}:
                assert np.array_equal(observed, repeated)
            else:
                np.testing.assert_allclose(observed, repeated, rtol=0, atol=0)
    finally:
        for buf in reversed(bufs):
            free(buf)
