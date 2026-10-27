"""Synthetic H128 dispatch coverage for the runner-local QSA wave32 selection.

The promoted H128 arithmetic is the wave32 paged sparse-attention owner, on the
c1 decode dispatch and on the prefill rows fallback below the H256 wave-prefill
owner. The kernel A/B against the strict parents lives in
``tests/test_gpu_qwen4_exp_qsa_sparse_hip.py``
(``test_qwen4_exp_sparse_wave32_h128_matches_strict_production_envelope``, 24/2
heads at head_dim 128). What that file cannot show is that the *runner* still
reaches those owners: the selection is a runner-local decision now
(``qwen4_exp_runner.qsa_wave32``), and a decision that silently stopped reaching
the dispatch would leave the kernel A/B passing while production ran the parent.

So this file drives the two real mixer entry points at head_dim 128 with a
synthetic mixer fixture, counts the module attributes the mixers read, and
asserts both arms:

- ``qsa_wave32=True`` (the default, production): exactly one wave32 launch on
  the sparse step (decode) / one wave32 rows launch (prefill), zero parent.
- ``qsa_wave32=False``: zero wave32, exactly one parent launch.

The two arms are compared to each other on the same fixture; the only
difference is the attention reduction order, and the tolerance is the one the
sibling mixer test already uses against its CPU oracle.

A real H128 model artifact is not required: ``head_dim`` comes from
``qwen4exp.attention.key_length``, and this fixture sets it to 128 directly.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import free
from hipengine.runtime import qwen4_exp_runner as runner_module
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

HIDDEN, QUERY_HEADS, KV_HEADS, HEAD_DIM = 8, 2, 1, 128
INDEX_HEADS, INDEX_DIM, RATIO, BLOCK_BUDGET = 2, 4, 2, 2
ROTARY_DIM, THETA = 4, 100.0
#: One row past the index owner's dense-equivalent limit (2*2 + 2 - 1 = 5), so
#: the last decode step and the last prefill row take the sparse arm.
ROWS = 6
DENSE_EQUIVALENT_LIMIT = BLOCK_BUDGET * RATIO + RATIO - 1
#: The tolerance the sibling mixer test uses against its CPU oracle.
TOLERANCE = 3e-4

WAVE32_DECODE = "qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32"
PARENT_DECODE = "qwen4_exp_qsa_sparse_attention_paged_bf16_f32"
WAVE32_ROWS = "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_wave32_f32"
PARENT_ROWS = "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither WMMA nor flash prefill may own the chunk in these arms."""

    for key in (
        "HIPENGINE_QWEN4_EXP_QSA_WMMA",
        "HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL",
        "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE",
        "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2",
        "HIPENGINE_QWEN4_EXP_QSA_WAVE32",
    ):
        monkeypatch.delenv(key, raising=False)


def _count(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count the four owners the mixers select between."""

    calls = {
        WAVE32_DECODE: 0, PARENT_DECODE: 0,
        WAVE32_ROWS: 0, PARENT_ROWS: 0,
    }
    for name in calls:
        original = getattr(runner_module, name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(runner_module, name, wrapper)
    return calls


def _weights(runtime, rng, allocations) -> Qwen4ExpQSAMixerDeviceWeights:
    q_width = QUERY_HEADS * HEAD_DIM
    kv_width = KV_HEADS * HEAD_DIM
    arrays = {
        "attn_q": rng.normal(0.0, 0.1, size=(q_width * 2, HIDDEN)).astype(np.float32),
        "attn_k": rng.normal(0.0, 0.1, size=(kv_width, HIDDEN)).astype(np.float32),
        "attn_v": rng.normal(0.0, 0.1, size=(kv_width, HIDDEN)).astype(np.float32),
        "attn_output": rng.normal(0.0, 0.1, size=(HIDDEN, q_width)).astype(np.float32),
        "index_q": rng.normal(
            0.0, 0.1, size=(INDEX_HEADS * INDEX_DIM, HIDDEN)).astype(np.float32),
        "index_k": rng.normal(0.0, 0.1, size=(INDEX_DIM, HIDDEN)).astype(np.float32),
    }
    return Qwen4ExpQSAMixerDeviceWeights(
        projections={
            name: _dense_f32_weight(name, array, runtime, allocations)
            for name, array in arrays.items()
        },
        q_norm_weight_ptr=_upload(
            rng.normal(1.0, 0.05, size=HEAD_DIM).astype(np.float32),
            runtime, allocations).ptr,
        k_norm_weight_ptr=_upload(
            rng.normal(1.0, 0.05, size=HEAD_DIM).astype(np.float32),
            runtime, allocations).ptr,
        index_q_norm_weight_ptr=_upload(
            rng.normal(1.0, 0.05, size=INDEX_DIM).astype(np.float32),
            runtime, allocations).ptr,
        index_k_norm_weight_ptr=_upload(
            rng.normal(1.0, 0.05, size=INDEX_DIM).astype(np.float32),
            runtime, allocations).ptr,
    )


def _run_decode(*, wave32: bool | None) -> np.ndarray:
    """Walk the sparse boundary one row at a time through the c1 mixer."""

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4701)
    mixed_rows = rng.normal(
        0.0, 0.2, size=(ROWS, HIDDEN)).astype(np.float32)
    allocations: list = []
    state = index_state = scratch = None
    try:
        weights = _weights(runtime, rng, allocations)
        state = Qwen4ExpDenseAttentionState.allocate(
            max_positions=8, block_size=256, kv_heads=KV_HEADS,
            head_dim=HEAD_DIM, runtime=runtime,
        )
        index_state = Qwen4ExpQSAIndexDeviceState.allocate(
            attention_state=state, index_heads=INDEX_HEADS, index_dim=INDEX_DIM,
            compression_ratio=RATIO, block_budget=BLOCK_BUDGET, runtime=runtime,
        )
        scratch = Qwen4ExpQSAScratch.allocate(
            rows=1, hidden=HIDDEN, query_heads=QUERY_HEADS, kv_heads=KV_HEADS,
            head_dim=HEAD_DIM, index_heads=INDEX_HEADS, index_dim=INDEX_DIM,
            runtime=runtime,
        )
        outputs = []
        for position in range(ROWS):
            d_mixed = _upload(mixed_rows[position][None], runtime, allocations)
            output = run_qwen4_exp_dense_qsa_token_mixer(
                d_mixed.ptr, weights,
                attention_state=state, index_state=index_state, scratch=scratch,
                position=position, rows=1, hidden=HIDDEN,
                query_heads=QUERY_HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM,
                rotary_dim=ROTARY_DIM, theta=THETA,
                index_heads=INDEX_HEADS, index_dim=INDEX_DIM,
                index_rotary_dim=ROTARY_DIM, runtime=runtime,
                **({} if wave32 is None else {"qsa_wave32": wave32}),
            )
            runtime.device_synchronize()
            outputs.append(_download(output, (HIDDEN,), np.float32, runtime))
        return np.stack(outputs)
    finally:
        for owner in (scratch, index_state, state):
            if owner is not None:
                owner.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


def _run_prefill(*, wave32: bool | None) -> np.ndarray:
    """Run the same rows as one chunk through the prefill mixer."""

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4701)
    mixed_rows = rng.normal(
        0.0, 0.2, size=(ROWS, HIDDEN)).astype(np.float32)
    allocations: list = []
    state = index_state = scratch = metadata = None
    try:
        weights = _weights(runtime, rng, allocations)
        state = Qwen4ExpDenseAttentionState.allocate(
            max_positions=8, block_size=256, kv_heads=KV_HEADS,
            head_dim=HEAD_DIM, runtime=runtime,
        )
        index_state = Qwen4ExpQSAIndexDeviceState.allocate(
            attention_state=state, index_heads=INDEX_HEADS, index_dim=INDEX_DIM,
            compression_ratio=RATIO, block_budget=BLOCK_BUDGET, runtime=runtime,
        )
        scratch = Qwen4ExpQSAScratch.allocate(
            rows=ROWS, hidden=HIDDEN, query_heads=QUERY_HEADS, kv_heads=KV_HEADS,
            head_dim=HEAD_DIM, index_heads=INDEX_HEADS, index_dim=INDEX_DIM,
            runtime=runtime,
        )
        metadata = Qwen4ExpQSAPrefillMetadata.allocate(
            state, rows=ROWS, selection_capacity=DENSE_EQUIVALENT_LIMIT,
        )
        d_mixed = _upload(mixed_rows, runtime, allocations)
        output = run_qwen4_exp_qsa_prefill_token_mixer(
            d_mixed.ptr, weights,
            attention_state=state, index_state=index_state, scratch=scratch,
            metadata=metadata, start_position=0, rows=ROWS, hidden=HIDDEN,
            query_heads=QUERY_HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM, theta=THETA,
            index_heads=INDEX_HEADS, index_dim=INDEX_DIM,
            index_rotary_dim=ROTARY_DIM, runtime=runtime,
            **({} if wave32 is None else {"qsa_wave32": wave32}),
        )
        runtime.device_synchronize()
        return _download(output, (ROWS, HIDDEN), np.float32, runtime)
    finally:
        for owner in (metadata, scratch, index_state, state):
            if owner is not None:
                owner.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h128_decode_dispatches_the_selected_owner_on_both_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the sparse step dispatches, and it dispatches the selected owner."""

    # The on arm: the production default at head_dim 128.
    calls = _count(monkeypatch)
    wave32_outputs = _run_decode(wave32=True)
    assert calls[WAVE32_DECODE] == 1, calls
    assert calls[PARENT_DECODE] == 0, calls
    # The dense steps above the index bound never reach either sparse owner.
    assert calls[WAVE32_ROWS] == 0 and calls[PARENT_ROWS] == 0, calls

    # The off arm: the wave64 parent, same fixture.
    calls = _count(monkeypatch)
    parent_outputs = _run_decode(wave32=False)
    assert calls[WAVE32_DECODE] == 0, calls
    assert calls[PARENT_DECODE] == 1, calls

    # Same selection, different attention reduction order.
    np.testing.assert_allclose(
        wave32_outputs, parent_outputs, rtol=TOLERANCE, atol=TOLERANCE)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h128_unbound_callers_keep_the_wave32_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default is the production arithmetic, on both dispatch sites.

    ``qsa_wave32`` defaults to True, so a caller that names nothing (an
    unbound harness, a fresh runner) must still reach the wave32 owners rather
    than the parent.
    """

    calls = _count(monkeypatch)
    _run_decode(wave32=None)
    assert calls[WAVE32_DECODE] == 1, calls
    assert calls[PARENT_DECODE] == 0, calls

    calls = _count(monkeypatch)
    _run_prefill(wave32=None)
    assert calls[WAVE32_ROWS] == 1, calls
    assert calls[PARENT_ROWS] == 0, calls


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h128_prefill_dispatches_the_selected_rows_owner_on_both_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prefill fallback below the H256 owner selects on the same decision."""

    calls = _count(monkeypatch)
    wave32_outputs = _run_prefill(wave32=True)
    assert calls[WAVE32_ROWS] == 1, calls
    assert calls[PARENT_ROWS] == 0, calls
    assert calls[WAVE32_DECODE] == 0 and calls[PARENT_DECODE] == 0, calls

    calls = _count(monkeypatch)
    parent_outputs = _run_prefill(wave32=False)
    assert calls[WAVE32_ROWS] == 0, calls
    assert calls[PARENT_ROWS] == 1, calls

    np.testing.assert_allclose(
        wave32_outputs, parent_outputs, rtol=TOLERANCE, atol=TOLERANCE)
