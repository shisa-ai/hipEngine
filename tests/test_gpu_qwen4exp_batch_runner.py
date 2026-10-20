"""GPU: full-model N-row Qwen4Exp decode against independent c1 runners.

The batch runner composes committed per-stage owners into one 3-layer synthetic
model: a PLE layer, two GDN layers, one QSA layer, real Q4_K expert weights, and
a Q8_0 embedding table. Every assertion is separated by contract
(``docs/EXECUTION-PROFILES.md``):

* **Control (exact).** Row -> slot mapping, untouched slots, permuted row order,
  deterministic repeats, release/re-admit, and the import cursor bookkeeping are
  byte-level assertions. A failure is a bug, not drift.
* **Arithmetic (measured, then bounded).** The batch path runs the mixed
  dense/sparse QSA rows mixer, the indexed GDN/PLE owners, and N-row GEMM
  variants, so it is not bit-identical to a c1 runner by construction. Each case
  measures and prints the deviation against an independent c1 runner for the
  same request, and the asserted bound is the measured envelope with margin.
  This is a fixture envelope, not the production-profile gate.
* **Admission.** A rejected step or import writes nothing: the slot slabs are
  byte-identical after a malformed row set, a foreign owner token, or a
  mismatched import source.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.kernels.backends import HIP_TARGET_ARCH_BACKEND
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.materialize import DeviceTensorAllocation
from hipengine.loading.qwen4_exp_gguf import Qwen4ExpGGUFTensorRef
from hipengine.loading.qwen4_exp_materialize import (
    LAYOUT_RAW_GGUF,
    Qwen4ExpDeviceWeight,
    Qwen4ExpGGUFWeightSpec,
    Qwen4ExpPLEStagingRing,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
from tests import _qwen4_exp_batch_fixtures as fixture
from tests._gguf_synthetic_weights import make_q4_k_weight, make_q8_0_weight
from tests.test_gpu_qwen4_exp_runner_gr import _dense_f32_weight
from tests.test_gpu_qwen4_exp_runner_moe import _q4_weight

# Measured on this fixture (see the module docstring): the bound is the printed
# envelope with margin, not a production numerical gate.
# Measured on this fixture: the two-row batch step reproduces each independent
# c1 runner bit-for-bit (worst logit deviation 0.0 over seven steps, GDN
# recurrent state 9.3e-10, conv/index/PLE state 0.0). The bounds stay wider than
# that because the GEMM/GDN variants are selected per target arch and only
# gfx1151 has been measured; they are fixture envelopes, not profile gates.
LOGIT_ATOL = 2e-2
LOGIT_RTOL = 2e-2
STATE_ATOL = 5e-2
TOP1_AGREEMENT = 1.0
# Two rows with different tokens and identical history must not produce the same
# logits, or the row/slot mapping assertions would be vacuous. Measured 3.4e-3 to
# 1.1e-2 on this fixture.
ROW_DISTINCT_MIN = 1e-4


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)


# -- synthetic resident ---------------------------------------------------


def _f32(name, shape, runtime, allocations, *, scale=0.05, seed=0):
    rng = np.random.default_rng(seed)
    array = rng.normal(0.0, scale, size=shape).astype(np.float32)
    return _dense_f32_weight(name, array, runtime, allocations)


class _Source:
    def __init__(self, name: str) -> None:
        self.name = name


class _Tensor:
    def __init__(self, ptr: int) -> None:
        self.ptr = ptr


def _raw_weight(
    name, raw, quant_key, ggml_type, shape, runtime, allocations, backend
):
    """Upload quantised bytes and describe them with their logical shape."""

    host = np.ascontiguousarray(raw, dtype=np.uint8)
    buffer = malloc(host.nbytes, runtime=runtime)
    allocations.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(host), runtime=runtime)
    tensor = GGUFTensorInfo(
        name=f"{name}.weight",
        shape=tuple(int(value) for value in shape),
        ggml_shape=tuple(reversed(tuple(int(value) for value in shape))),
        ggml_type=int(ggml_type),
        ggml_type_name=ggml_type.name,
        n_elements=int(np.prod(shape)),
        nbytes=int(host.nbytes),
        offset=0,
        data_offset=0,
        byte_shape=tuple(int(value) for value in host.shape),
    )
    spec = Qwen4ExpGGUFWeightSpec(
        slot_path=name,
        source_ref=Qwen4ExpGGUFTensorRef(0, Path("synthetic.gguf"), tensor),
        quant_key=quant_key,
        layout=LAYOUT_RAW_GGUF,
        allocation_names=("raw",),
        device_resident=True,
        device_nbytes=int(host.nbytes),
    )
    allocation = DeviceTensorAllocation(
        name=name,
        source=_Source(name),
        buffer=buffer,
        tensor=_Tensor(buffer.ptr),
        owns_buffer=False,
    )
    return Qwen4ExpDeviceWeight(spec, backend, {"raw": allocation})


def _build_weights(runtime, backend, config=None):
    """Every weight the synthetic layers and the head resolve."""

    config = config or fixture.config()
    allocations: list = []
    weights: dict[str, object] = {}
    seed = 100

    def dense(name, shape, scale=0.05):
        nonlocal seed
        seed += 1
        weights[name] = _f32(
            name, shape, runtime, allocations, scale=scale, seed=seed
        )

    residual = fixture.RESIDUAL_WIDTH
    weights["root.token_embedding"] = _raw_weight(
        "root.token_embedding",
        make_q8_0_weight(fixture.VOCAB, fixture.HIDDEN),
        "gguf_q8_0",
        GGMLQuantizationType.Q8_0,
        (fixture.VOCAB, fixture.HIDDEN),
        runtime,
        allocations,
        backend,
    )
    dense("root.head_hc_norm", (residual,), scale=0.1)
    dense("root.head_hc_down", (fixture.LOW_RANK, residual), scale=0.1)
    dense("root.head_hc_up", (residual, fixture.LOW_RANK), scale=0.1)
    dense("root.lm_head", (fixture.VOCAB, fixture.HIDDEN), scale=0.1)
    for layer, kind in enumerate(config.layer_types):
        prefix = f"layers.{layer}."
        dense(prefix + "hc_attn_norm", (residual,), scale=0.1)
        dense(prefix + "hc_attn_down", (fixture.LOW_RANK, residual), scale=0.1)
        dense(prefix + "hc_attn_up", (residual, fixture.LOW_RANK), scale=0.1)
        dense(prefix + "hc_attn_inject", (fixture.BRANCHES, residual), scale=0.1)
        dense(prefix + "hc_ffn_norm", (residual,), scale=0.1)
        dense(prefix + "hc_ffn_down", (fixture.LOW_RANK, residual), scale=0.1)
        dense(prefix + "hc_ffn_up", (residual, fixture.LOW_RANK), scale=0.1)
        dense(prefix + "hc_ffn_inject", (fixture.BRANCHES, residual), scale=0.1)
        dense(prefix + "router", (fixture.EXPERTS, fixture.HIDDEN), scale=0.1)
        weights[prefix + "expert_gate"] = _q4_weight(
            prefix + "expert_gate",
            np.stack(
                [
                    make_q4_k_weight(fixture.FFN, fixture.HIDDEN)
                    for _ in range(fixture.EXPERTS)
                ]
            ),
            runtime,
            allocations,
        )
        weights[prefix + "expert_up"] = _q4_weight(
            prefix + "expert_up",
            np.stack(
                [
                    make_q4_k_weight(fixture.FFN, fixture.HIDDEN)
                    for _ in range(fixture.EXPERTS)
                ]
            ),
            runtime,
            allocations,
        )
        weights[prefix + "expert_down"] = _q4_weight(
            prefix + "expert_down",
            np.stack(
                [
                    make_q4_k_weight(fixture.HIDDEN, fixture.FFN)
                    for _ in range(fixture.EXPERTS)
                ]
            ),
            runtime,
            allocations,
        )
        dense(prefix + "shared_gate", (fixture.FFN, fixture.HIDDEN), scale=0.1)
        dense(prefix + "shared_up", (fixture.FFN, fixture.HIDDEN), scale=0.1)
        dense(prefix + "shared_down", (fixture.HIDDEN, fixture.FFN), scale=0.1)
        dense(prefix + "shared_expert_gate", (fixture.HIDDEN,), scale=0.1)
        if kind == "qsa":
            dense(
                prefix + "attn_q",
                (2 * fixture.Q_WIDTH, fixture.HIDDEN),
                scale=0.1,
            )
            dense(prefix + "attn_k", (fixture.KV_WIDTH, fixture.HIDDEN), scale=0.1)
            dense(prefix + "attn_v", (fixture.KV_WIDTH, fixture.HIDDEN), scale=0.1)
            dense(prefix + "attn_output", (fixture.HIDDEN, fixture.Q_WIDTH), scale=0.1)
            dense(
                prefix + "index_q",
                (fixture.INDEX_HEADS * fixture.INDEX_DIM, fixture.HIDDEN),
                scale=0.1,
            )
            dense(prefix + "index_k", (fixture.INDEX_DIM, fixture.HIDDEN), scale=0.1)
            dense(prefix + "attn_q_norm", (fixture.HEAD_DIM,), scale=0.1)
            dense(prefix + "attn_k_norm", (fixture.HEAD_DIM,), scale=0.1)
            dense(prefix + "index_q_norm", (fixture.INDEX_DIM,), scale=0.1)
            dense(prefix + "index_k_norm", (fixture.INDEX_DIM,), scale=0.1)
        else:
            dense(
                prefix + "attn_qkv", (fixture.GDN_QKV_WIDTH, fixture.HIDDEN), scale=0.1
            )
            dense(
                prefix + "attn_gate", (fixture.GDN_CORE_WIDTH, fixture.HIDDEN), scale=0.1
            )
            dense(prefix + "ssm_alpha", (fixture.GDN_V_HEADS, fixture.HIDDEN), scale=0.1)
            dense(prefix + "ssm_beta", (fixture.GDN_V_HEADS, fixture.HIDDEN), scale=0.1)
            dense(
                prefix + "ssm_out", (fixture.HIDDEN, fixture.GDN_CORE_WIDTH), scale=0.1
            )
            dense(
                prefix + "ssm_conv1d",
                (fixture.GDN_QKV_WIDTH, fixture.GDN_CONV_KERNEL),
                scale=0.1,
            )
            dense(prefix + "ssm_dt_bias", (fixture.GDN_V_HEADS,), scale=0.05)
            dense(prefix + "ssm_a", (fixture.GDN_V_HEADS,), scale=0.05)
            dense(prefix + "ssm_norm", (fixture.GDN_HEAD_DIM,), scale=0.1)
        if layer in config.ple_layers:
            dense(prefix + "ple_key", (residual, fixture.HIDDEN), scale=0.1)
            dense(prefix + "ple_value", (fixture.HIDDEN, fixture.HIDDEN), scale=0.1)
            dense(prefix + "ple_norm_key", (residual,), scale=0.1)
            dense(prefix + "ple_norm_query", (residual,), scale=0.1)
            dense(prefix + "ple_norm_conv", (residual,), scale=0.1)
            dense(
                prefix + "ple_conv1d",
                (residual, fixture.PLE_CONV_KERNEL),
                scale=0.1,
            )
    return weights, allocations


def _build_bundle(monkey, backend, config):
    runtime = get_hip_runtime()
    weights, allocations = _build_weights(runtime, backend, config)
    table = fixture.FakePleTable()
    staging = Qwen4ExpPLEStagingRing.create(
        table, row_capacity=4 * fixture.PLE_HEAD_COUNT, runtime=None
    )
    resident = SimpleNamespace(
        plan=SimpleNamespace(config=config),
        weights=weights,
        ple_staging=staging,
        ple_table=table,
    )
    resident.weight = weights.__getitem__
    return SimpleNamespace(
        runtime=runtime,
        resident=resident,
        weights=weights,
        allocations=allocations,
        staging=staging,
        backend=backend,
    )


def _release_bundle(bundle) -> None:
    bundle.staging.close()
    for buffer in reversed(bundle.allocations):
        free(buffer, runtime=bundle.runtime)


def _module_monkeypatch() -> pytest.MonkeyPatch:
    monkey = pytest.MonkeyPatch()
    # The c1 reference is the eager committed path; graph capture is a c1
    # single-request optimization these fixtures do not exercise.
    monkey.setenv("HIPENGINE_QWEN4_EXP_MOE_GRAPH", "0")
    monkey.setenv("HIPENGINE_QWEN4_EXP_LAYER_GRAPHS", "0")
    monkey.delenv("HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL", raising=False)
    monkey.delenv("HIPENGINE_QWEN4_EXP_Q4_K_MMQ_PREFILL", raising=False)
    return monkey


@pytest.fixture(scope="module")
def model(hip_test_target_arch):
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    monkey = _module_monkeypatch()
    bundle = _build_bundle(
        monkey, HIP_TARGET_ARCH_BACKEND[hip_test_target_arch], fixture.config()
    )
    try:
        yield bundle
    finally:
        _release_bundle(bundle)
        monkey.undo()


@pytest.fixture(scope="module")
def two_qsa_model(hip_test_target_arch):
    """The same geometry with two QSA layers, as the real model has 12 of 48."""

    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    monkey = _module_monkeypatch()
    bundle = _build_bundle(
        monkey,
        HIP_TARGET_ARCH_BACKEND[hip_test_target_arch],
        fixture.two_qsa_config(),
    )
    try:
        yield bundle
    finally:
        _release_bundle(bundle)
        monkey.undo()


def _c1(model, *, max_positions: int = fixture.MAX_POSITIONS):
    return Qwen4ExpGGUFResidentModelRunner(
        model.resident,
        max_sequence_length=max_positions,
        prefill_chunk_size=4,
        backend=model.backend,
        runtime=model.runtime,
    )


def _batch(
    model,
    *,
    slots: int = 3,
    max_positions: int = fixture.MAX_POSITIONS,
    graph_segments: bool = True,
    pool_blocks=None,
):
    return Qwen4ExpGGUFBatchRunner(
        model.resident,
        state_slots=slots,
        max_positions=max_positions,
        backend=model.backend,
        runtime=model.runtime,
        graph_segments=graph_segments,
        pool_blocks=pool_blocks,
    )


def _download(buffer, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    # An explicit size: the default is the source buffer's size, which is larger
    # than one row's host array whenever the buffer holds several rows.
    assert host.nbytes <= buffer.nbytes
    copy_device_to_host(host_array_ptr(host), buffer, host.nbytes, runtime=runtime)
    return host


def _slot_kv(batch_runner, slot, *, layer: int = 0):
    """One QSA layer's full paged K/V bytes for a slot, in logical block order."""

    attention = batch_runner.attention_states[layer]
    row_bytes = attention.kv_heads * attention.head_dim * DType.BF16.itemsize
    block_bytes = attention.block_size * row_bytes
    blocks = attention.block_host[slot]
    keys = []
    values = []
    for logical in blocks.tolist():
        offset = int(logical) * block_bytes
        keys.append(
            _download(
                SimpleNamespace(
                    ptr=attention.key_cache.ptr + offset,
                    nbytes=block_bytes,
                ),
                (block_bytes,),
                np.uint8,
                batch_runner.runtime,
            )
        )
        values.append(
            _download(
                SimpleNamespace(
                    ptr=attention.value_cache.ptr + offset,
                    nbytes=block_bytes,
                ),
                (block_bytes,),
                np.uint8,
                batch_runner.runtime,
            )
        )
    return np.concatenate(keys), np.concatenate(values)


def _c1_kv(runner, *, layer: int = 0):
    attention = runner.attention_states[layer]
    row_bytes = fixture.KV_HEADS * fixture.HEAD_DIM * DType.BF16.itemsize
    block_bytes = attention.block_size * row_bytes
    blocks = attention.block_host
    keys = []
    values = []
    for logical in blocks.tolist():
        offset = int(logical) * block_bytes
        keys.append(
            _download(
                SimpleNamespace(
                    ptr=attention.key_cache.ptr + offset, nbytes=block_bytes
                ),
                (block_bytes,),
                np.uint8,
                runner.runtime,
            )
        )
        values.append(
            _download(
                SimpleNamespace(
                    ptr=attention.value_cache.ptr + offset, nbytes=block_bytes
                ),
                (block_bytes,),
                np.uint8,
                runner.runtime,
            )
        )
    return np.concatenate(keys), np.concatenate(values)


def _bf16_float(bits: np.ndarray) -> np.ndarray:
    raw = np.ascontiguousarray(bits, dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def _assert_control_equal(left: bytes, right: bytes, *, label: str) -> None:
    assert left == right, f"{label} bytes changed"


def _slot_buffer(ptr: int, nbytes: int):
    return DeviceBuffer(int(ptr), int(nbytes))


def _float_view(raw: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(raw.view(np.uint16))
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def test_batch_step_matches_independent_c1_runners(model) -> None:
    """Two requests advance seven steps and keep each one's own state.

    Seven steps is deliberate: the index owner's dense-equivalence bound is five
    contexts, so the sixth and seventh steps run the sparse selection for both
    rows, and the keys those rows captured while dense are what it selects from.
    """

    batch = _batch(model)
    references = [_c1(model), _c1(model)]
    tokens = [3, 11, 25, 7, 19, 31, 42]
    try:
        owners = [batch.admit(slot) for slot in (2, 0)]
        for step, token in enumerate(tokens):
            expected = [reference.step(token) for reference in references]
            result = batch.step(
                [token, token], slots=[2, 0], owners=owners
            )
            assert result.slot_ids == (2, 0)
            assert result.positions.tolist() == [step, step]
            assert result.logits is not None
            assert result.logits.shape == (2, fixture.VOCAB)
            for row, reference_result in enumerate(expected):
                deviation = _max_abs(result.logits[row], reference_result.logits)
                assert int(np.argmax(result.logits[row])) == int(
                    np.argmax(reference_result.logits)
                ), f"step {step} row {row} top-1 disagrees with the c1 runner"
                assert deviation <= LOGIT_ATOL, (
                    f"step {step} row {row} logit deviation {deviation:.3e} "
                    f"exceeds {LOGIT_ATOL:.3e}"
                )
                assert np.allclose(
                    result.logits[row],
                    reference_result.logits,
                    rtol=LOGIT_RTOL,
                    atol=LOGIT_ATOL,
                )
        # Every persistent state agrees with the request it belongs to.
        for row, slot in enumerate((2, 0)):
            reference = references[row]
            batch_keys, batch_values = _slot_kv(batch, slot)
            c1_keys, c1_values = _c1_kv(reference)
            assert _max_abs(_float_view(batch_keys), _float_view(c1_keys)) <= STATE_ATOL
            assert (
                _max_abs(_float_view(batch_values), _float_view(c1_values))
                <= STATE_ATOL
            )
            gdn_state = batch.gdn_states[0]
            slot_gdn = _download(
                _slot_buffer(
                    gdn_state.recurrent_slot_ptr(slot),
                    gdn_state.recurrent_bytes_per_slot,
                ),
                (gdn_state.recurrent_bytes_per_slot // 4,),
                np.float32,
                model.runtime,
            )
            # The c1 decode state holds one row per GDN layer.
            reference_gdn = _download(
                _slot_buffer(
                    reference.state.gdn_matrix.ptr,
                    gdn_state.recurrent_bytes_per_slot,
                ),
                (gdn_state.recurrent_bytes_per_slot // 4,),
                np.float32,
                model.runtime,
            )
            assert _max_abs(slot_gdn, reference_gdn) <= STATE_ATOL
            history = _download(
                _slot_buffer(
                    batch.ple_state.history_slot_ptr(slot),
                    batch.ple_state.history_bytes_per_slot,
                ),
                (batch.ple_state.history_bytes_per_slot // 4,),
                np.float32,
                model.runtime,
            )
            reference_history = _download(
                reference.state.ple_conv,
                (reference.state.ple_conv.nbytes // 4,),
                np.float32,
                model.runtime,
            )
            assert _max_abs(history, reference_history) <= STATE_ATOL
            index_state = batch.index_states[0]
            assert int(index_state.cursor_host[slot]) == len(tokens)
            reference_index = reference.index_states[0]
            assert int(reference_index.count) == len(tokens)
            raw_slot = _download(
                _slot_buffer(
                    index_state.raw_position_ptr(slot, 0),
                    len(tokens) * fixture.INDEX_DIM * 4,
                ),
                (len(tokens), fixture.INDEX_DIM),
                np.float32,
                model.runtime,
            )
            raw_reference = _download(
                _slot_buffer(
                    reference_index.raw_keys.ptr,
                    len(tokens) * fixture.INDEX_DIM * 4,
                ),
                (len(tokens), fixture.INDEX_DIM),
                np.float32,
                model.runtime,
            )
            assert _max_abs(raw_slot, raw_reference) <= STATE_ATOL
        assert [batch.position(slot) for slot in (2, 0)] == [7, 7]
        assert batch.free_slots == (1,)
        assert [batch.owner_of(slot) for slot in (2, 0)] == owners
        # The dense-equivalence bound was crossed, so both rows ran the sparse
        # selection at least once.
        assert batch.index_states[0].pooled_count_host[2] > fixture.BLOCK_BUDGET
    finally:
        for reference in references:
            reference.close()
        batch.close()


def test_batch_step_row_order_does_not_change_slot_state(model) -> None:
    """A permuted row set reproduces each slot's own bytes exactly."""

    tokens = [(5, 17), (29, 6), (13, 41)]
    forward = _batch(model)
    reverse = _batch(model)
    try:
        forward.admit(0)
        forward.admit(1)
        reverse.admit(0)
        reverse.admit(1)
        forward_owners = [forward.owner_of(0), forward.owner_of(1)]
        reverse_owners = [reverse.owner_of(0), reverse.owner_of(1)]
        for step, (left, right) in enumerate(tokens):
            first = forward.step([left, right], slots=[0, 1], owners=forward_owners)
            second = reverse.step(
                [right, left], slots=[1, 0], owners=reverse_owners[::-1]
            )
            # Row order is the caller's: row r belongs to slot_ids[r].
            assert second.slot_ids == (1, 0)
            assert np.array_equal(first.logits[0], second.logits[1])
            assert np.array_equal(first.logits[1], second.logits[0])
            assert first.positions.tolist() == second.positions.tolist() == [step, step]
            # The two rows carry different tokens, so a transposed row/slot map
            # cannot hide behind identical logits.
            assert _max_abs(first.logits[0], first.logits[1]) > ROW_DISTINCT_MIN
        for slot in (0, 1):
            forward_keys, forward_values = _slot_kv(forward, slot)
            reverse_keys, reverse_values = _slot_kv(reverse, slot)
            assert forward_keys.tobytes() == reverse_keys.tobytes()
            assert forward_values.tobytes() == reverse_values.tobytes()
            assert (
                forward.gdn_states[0]
                .recurrent_state.nbytes
                == reverse.gdn_states[0].recurrent_state.nbytes
            )
        # Slot 2 was never admitted, so its whole slab is still zero.
        keys, values = _slot_kv(forward, 2)
        assert not np.any(keys)
        assert not np.any(values)
    finally:
        forward.close()
        reverse.close()


def test_batch_step_is_deterministic_across_runs(model) -> None:
    """Two independent owners produce byte-identical steps and state."""

    left = _batch(model)
    right = _batch(model)
    tokens = [13, 21, 33, 2, 60, 44]
    try:
        for slot in (0, 2):
            left.admit(slot)
            right.admit(slot)
        left_owners = [left.owner_of(0), left.owner_of(2)]
        right_owners = [right.owner_of(0), right.owner_of(2)]
        for token in tokens:
            first = left.step([token, token], slots=[0, 2], owners=left_owners)
            second = right.step([token, token], slots=[0, 2], owners=right_owners)
            assert first.logits.tobytes() == second.logits.tobytes()
            assert first.tokens.tolist() == second.tokens.tolist()
            assert first.positions.tolist() == second.positions.tolist()
        for slot in (0, 2):
            left_keys, left_values = _slot_kv(left, slot)
            right_keys, right_values = _slot_kv(right, slot)
            assert left_keys.tobytes() == right_keys.tobytes()
            assert left_values.tobytes() == right_values.tobytes()
        assert (
            left.ple_hash_states[0] == right.ple_hash_states[0]
        ), "PLE n-gram history diverged between identical runs"
    finally:
        left.close()
        right.close()


def test_batch_step_records_projection_dispatch(model) -> None:
    """A real batch step records the exact projection dispatch it launched."""

    from hipengine.runtime.gguf_linear import record_gguf_linear_dispatch

    batch = _batch(model)
    records: list[dict] = []
    try:
        owners = [batch.admit(slot) for slot in (0, 1)]
        with record_gguf_linear_dispatch(records):
            result = batch.step([3, 11], slots=[0, 1], owners=owners)
        assert result.logits is not None
        assert records, "no projection dispatch was recorded"
        assert {int(record["rows"]) for record in records} == {2}
        assert {record["activation_dtype"] for record in records} == {"f32"}
        assert {record["output_dtype"] for record in records} == {"f32"}
        for record in records:
            key = record["key"]
            assert str(key["backend"]).startswith("hip_gfx")
            assert key["layer"]
            assert key["quant"]
            assert key["variant"]
            assert record["abi"]
            assert int(record["in_features"]) > 0
            assert int(record["out_features"]) > 0
            assert record["activation_dtype"]
            assert record["output_dtype"]
    finally:
        batch.close()


def test_batch_slot_reuse_starts_from_zero_state(model) -> None:
    """Release and re-admit zero every slab, and a stale token is refused."""

    batch = _batch(model)
    reference = _c1(model)
    try:
        owner = batch.admit(1)
        for token in (13, 21, 33):
            reference.step(token)
            batch.step([token])
        keys, values = _slot_kv(batch, 1)
        assert np.any(keys) and np.any(values)
        batch.release(1, owner)
        assert batch.free_slots == (0, 1, 2)
        keys, values = _slot_kv(batch, 1)
        assert not np.any(keys)
        assert not np.any(values)
        assert int(batch.index_states[0].cursor_host[1]) == 0
        assert int(batch.index_states[0].pooled_count_host[1]) == 0
        assert 1 not in batch.ple_hash_states
        assert not np.any(
            _download(
                _slot_buffer(
                    batch.gdn_states[0].recurrent_slot_ptr(1),
                    batch.gdn_states[0].recurrent_bytes_per_slot,
                ),
                (batch.gdn_states[0].recurrent_bytes_per_slot // 4,),
                np.float32,
                model.runtime,
            )
        )
        assert not np.any(
            _download(
                _slot_buffer(
                    batch.ple_state.history_slot_ptr(1),
                    batch.ple_state.history_bytes_per_slot,
                ),
                (batch.ple_state.history_bytes_per_slot // 4,),
                np.float32,
                model.runtime,
            )
        )
        fresh_owner = batch.admit(1)
        assert fresh_owner != owner
        # A reused slot steps from zero state, like a brand-new c1 runner.
        fresh = _c1(model)
        try:
            expected = fresh.step(13)
            actual = batch.step([13])
            assert _max_abs(actual.logits[0], expected.logits) <= LOGIT_ATOL
        finally:
            fresh.close()
        with pytest.raises(RuntimeError, match="owned by request"):
            batch.release(1, owner)
    finally:
        reference.close()
        batch.close()


def test_batch_step_rejects_bad_rows_before_mutating(model) -> None:
    """A rejected step leaves every slot's bytes and cursor untouched."""

    batch = _batch(model)
    try:
        owner = batch.admit(2)
        before_keys, before_values = _slot_kv(batch, 2)
        before_cursor = int(batch.index_states[0].cursor_host[2])
        with pytest.raises(RuntimeError, match="not admitted"):
            batch.step([1], slots=[0], owners=[owner])
        with pytest.raises(ValueError, match="outside"):
            batch.step([1], slots=[9], owners=[owner])
        with pytest.raises(ValueError, match="equal length"):
            batch.step([1, 2], slots=[2], owners=[owner])
        with pytest.raises(ValueError, match="appears twice"):
            batch.step([1, 2], slots=[2, 2], owners=[owner, owner])
        with pytest.raises(ValueError, match="vocabulary"):
            batch.step([fixture.VOCAB], slots=[2], owners=[owner])
        with pytest.raises(RuntimeError, match="no slots are admitted"):
            _batch(model).step([1])
        after_keys, after_values = _slot_kv(batch, 2)
        assert before_keys.tobytes() == after_keys.tobytes()
        assert before_values.tobytes() == after_values.tobytes()
        assert int(batch.index_states[0].cursor_host[2]) == before_cursor
        assert batch.position(2) == 0
        # A rejected call leaves the slot healthy, not poisoned.
        assert batch.poisoned_slots == ()
        assert not batch.is_poisoned(2)
        with pytest.raises(RuntimeError, match="owned by request"):
            batch.release(2, owner + 1)
        batch.release(2, owner)
    finally:
        batch.close()


def test_batch_import_from_c1_prefill_matches_the_reference(model) -> None:
    """A c1-prefilled request decodes on the batch runner and stays in step."""

    batch = _batch(model)
    reference = _c1(model)
    prefix = [9, 4, 27, 15]
    try:
        reference.prefill_serial(prefix)
        owner = batch.admit(0)
        position = batch.import_prefilled(0, owner, source=reference)
        assert position == len(prefix)
        assert batch.position(0) == len(prefix)
        assert int(batch.index_states[0].cursor_host[0]) == len(prefix)
        # Pooled keys are re-derived from the copied raw keys, not copied.
        assert int(batch.index_states[0].pooled_count_host[0]) == 0
        assert batch.ple_hash_states[0] == reference.ple_hash_states[0]
        # The next token advances both sides from the same prefilled state.
        next_token = 22
        expected = reference.step(next_token)
        actual = batch.step([next_token])
        assert actual.positions.tolist() == [len(prefix)]
        deviation = _max_abs(actual.logits[0], expected.logits)
        assert deviation <= LOGIT_ATOL, f"imported logit deviation {deviation:.3e}"
        assert int(np.argmax(actual.logits[0])) == int(np.argmax(expected.logits))
        assert batch.position(0) == len(prefix) + 1
        batch_keys, batch_values = _slot_kv(batch, 0)
        c1_keys, c1_values = _c1_kv(reference)
        assert _max_abs(_float_view(batch_keys), _float_view(c1_keys)) <= STATE_ATOL
        assert _max_abs(_float_view(batch_values), _float_view(c1_values)) <= STATE_ATOL
        # Importing again into the same slot is refused: it already has history.
        with pytest.raises(ValueError, match="already holds request history"):
            batch.import_prefilled(0, owner, source=reference)
        # A closed source and a foreign owner token are refused.
        other = _c1(model)
        other.close()
        with pytest.raises(ValueError, match="closed"):
            batch.import_prefilled(1, batch.admit(1), source=other)
        batch.admit(2)
        with pytest.raises(RuntimeError, match="owned by request"):
            batch.import_prefilled(2, owner, source=reference)
    finally:
        reference.close()
        batch.close()


def test_step_argmax_path_matches_the_captured_logits(model) -> None:
    """The no-readback path returns the greedy ids of the same step.

    ``capture_logits=False`` skips the ``[rows, vocab]`` download and takes the
    committed row-wise argmax kernel instead. Both paths must select the same id
    and leave identical state, or a serving caller could not choose between them.
    """

    capturing = _batch(model)
    greedy = _batch(model)
    tokens = [7, 19, 33, 2, 60]
    try:
        for slot in (0, 2):
            capturing.admit(slot)
            greedy.admit(slot)
        capturing_owners = [capturing.owner_of(0), capturing.owner_of(2)]
        greedy_owners = [greedy.owner_of(0), greedy.owner_of(2)]
        for token in tokens:
            with_logits = capturing.step(
                [token, token], slots=[0, 2], owners=capturing_owners
            )
            without = greedy.step(
                [token, token],
                slots=[0, 2],
                owners=greedy_owners,
                capture_logits=False,
            )
            assert without.logits is None
            expected = np.argmax(with_logits.logits, axis=1)
            assert without.tokens.tolist() == expected.tolist()
            assert without.tokens.tolist() == with_logits.tokens.tolist()
            assert without.positions.tolist() == with_logits.positions.tolist()
        for slot in (0, 2):
            first_keys, first_values = _slot_kv(capturing, slot)
            second_keys, second_values = _slot_kv(greedy, slot)
            assert first_keys.tobytes() == second_keys.tobytes()
            assert first_values.tobytes() == second_values.tobytes()
        assert capturing.ple_hash_states[0] == greedy.ple_hash_states[0]
    finally:
        capturing.close()
        greedy.close()


@pytest.mark.parametrize("fault", ["gdn", "index"])
def test_failed_step_poisons_its_rows_and_spares_the_others(model, fault, monkeypatch):
    """A step that dies after mutating refuses a retry until an explicit reset.

    The failure is injected *after* the real stage ran, so the rows' device state
    really is partially advanced while the position mirror still says 0. A retry
    would append and advance on top of that, so the rows must refuse work while a
    slot that was not in the row set keeps decoding.
    """

    from hipengine.runtime import qwen4_exp_batch_runner as runner_module

    batch = _batch(model)
    reference = _c1(model)
    token = 9
    try:
        owners = [batch.admit(slot) for slot in (0, 1, 2)]
        # One-shot fault: the stage runs for real, then the step fails, so later
        # steps on healthy slots take the normal path.
        armed = {"value": True}
        if fault == "gdn":
            real = runner_module.run_qwen4_exp_batch_gdn_layer

            def failing(*args, **kwargs):
                output = real(*args, **kwargs)
                if armed["value"]:
                    armed["value"] = False
                    raise RuntimeError("injected failure after the GDN mutation")
                return output

            monkeypatch.setattr(
                runner_module, "run_qwen4_exp_batch_gdn_layer", failing
            )
        else:
            real = runner_module.run_qwen4_exp_batch_qsa_layer

            def failing(*args, **kwargs):
                output = real(*args, **kwargs)
                if armed["value"]:
                    armed["value"] = False
                    raise RuntimeError("injected failure after the index mutation")
                return output

            monkeypatch.setattr(
                runner_module, "run_qwen4_exp_batch_qsa_layer", failing
            )
        with pytest.raises(RuntimeError, match="injected failure"):
            batch.step([token, token], slots=[0, 2], owners=[owners[0], owners[2]])
        assert batch.poisoned_slots == (0, 2)
        assert not batch.is_poisoned(1)
        # The position mirror did not advance, which is exactly why a retry
        # cannot be allowed: the device state did.
        assert batch.position(0) == 0 and batch.position(2) == 0
        if fault == "gdn":
            gdn_state = batch.gdn_states[0]
            recurrent = _download(
                _slot_buffer(
                    gdn_state.recurrent_slot_ptr(0),
                    gdn_state.recurrent_bytes_per_slot,
                ),
                (gdn_state.recurrent_bytes_per_slot // 4,),
                np.float32,
                model.runtime,
            )
            assert np.any(recurrent), "the GDN row really was advanced before failing"
        else:
            assert int(batch.index_states[0].cursor_host[0]) == 1
        with pytest.raises(RuntimeError, match="poisoned"):
            batch.step([token, token], slots=[0, 2], owners=[owners[0], owners[2]])
        with pytest.raises(RuntimeError, match="poisoned"):
            batch.step([token], slots=[2], owners=[owners[2]])
        # A slot outside the failed row set decodes normally.
        expected = reference.step(token)
        healthy = batch.step([token], slots=[1], owners=[owners[1]])
        deviation = _max_abs(healthy.logits[0], expected.logits)
        assert deviation <= LOGIT_ATOL, f"healthy slot deviation {deviation:.3e}"
        assert int(np.argmax(healthy.logits[0])) == int(np.argmax(expected.logits))
        assert batch.position(1) == 1
        assert batch.poisoned_slots == (0, 2)
        # reset_slot clears one slot and only that slot.
        batch.reset_slot(0, owners[0])
        assert batch.poisoned_slots == (2,)
        fresh = _c1(model)
        try:
            expected = fresh.step(token)
            recovered = batch.step([token], slots=[0], owners=[owners[0]])
            deviation = _max_abs(recovered.logits[0], expected.logits)
            assert deviation <= LOGIT_ATOL, f"reset slot deviation {deviation:.3e}"
        finally:
            fresh.close()
        assert batch.position(0) == 1
        # release + admit is the other recovery path.
        batch.release(2, owners[2])
        assert batch.poisoned_slots == ()
        replacement = batch.admit(2)
        assert replacement not in owners
        fresh = _c1(model)
        try:
            expected = fresh.step(token)
            recovered = batch.step([token], slots=[2], owners=[replacement])
            deviation = _max_abs(recovered.logits[0], expected.logits)
            assert deviation <= LOGIT_ATOL, f"re-admit deviation {deviation:.3e}"
        finally:
            fresh.close()
    finally:
        reference.close()
        batch.close()


def test_failed_import_poisons_only_the_target_slot(model, monkeypatch) -> None:
    """A copy that dies part-way refuses a retry until the slot is reset."""

    batch = _batch(model)
    reference = _c1(model)
    prefix = [9, 4, 27, 15]
    try:
        reference.prefill_serial(prefix)
        owner = batch.admit(0)
        spare = batch.admit(2)

        class _FailingRuntime:
            def __init__(self, inner, fail_at):
                self._inner = inner
                self._fail_at = fail_at
                self._calls = 0

            def memcpy(self, *args, **kwargs):
                self._calls += 1
                if self._calls == self._fail_at:
                    raise RuntimeError("injected device copy failure")
                return self._inner.memcpy(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        batch.runtime = _FailingRuntime(model.runtime, 3)
        with pytest.raises(RuntimeError, match="injected device copy failure"):
            batch.import_prefilled(0, owner, source=reference)
        batch.runtime = model.runtime
        assert batch.poisoned_slots == (0,)
        assert not batch.is_poisoned(2)
        assert batch.position(0) == 0
        with pytest.raises(RuntimeError, match="poisoned"):
            batch.import_prefilled(0, owner, source=reference)
        with pytest.raises(RuntimeError, match="poisoned"):
            batch.step([9], slots=[0], owners=[owner])
        # The spare slot is untouched by the failed import.
        assert batch.step([9], slots=[2], owners=[spare]).positions.tolist() == [0]
        # A reset clears the poison and the import then completes.
        batch.reset_slot(0, owner)
        assert batch.poisoned_slots == ()
        assert batch.import_prefilled(0, owner, source=reference) == len(prefix)
        assert batch.position(0) == len(prefix)
        assert int(batch.index_states[0].cursor_host[0]) == len(prefix)
        expected = reference.step(22)
        actual = batch.step([22], slots=[0], owners=[owner])
        deviation = _max_abs(actual.logits[0], expected.logits)
        assert deviation <= LOGIT_ATOL, f"imported logit deviation {deviation:.3e}"
        batch_keys, _ = _slot_kv(batch, 0)
        c1_keys, _ = _c1_kv(reference)
        assert _max_abs(_float_view(batch_keys), _float_view(c1_keys)) <= STATE_ATOL
    finally:
        reference.close()
        batch.close()


def test_two_qsa_layers_keep_separate_caches_and_match_c1(two_qsa_model) -> None:
    """A model with two QSA layers keeps one K/V cache per layer.

    The RED signal is the per-layer cache bytes: with one shared cache the
    second QSA layer writes its blocks over the first layer's, so layer 0's
    cache holds layer 1's keys (measured 0.18 max deviation from the c1
    reference at weight scale 0.1). The logits comparison is a non-regression
    guard: in this synthetic geometry the QSA mixer's contribution does not
    reach the BF16 residual as a visible logit change, so logit parity alone
    cannot see a cache-content bug. The default one-QSA-layer fixture cannot
    express the collision at all, and the real artifact has 12 QSA layers of 48.
    """

    model = two_qsa_model
    config = model.resident.plan.config
    assert config.qsa_layer_count == 2
    batch = _batch(model, slots=2)
    references = [_c1(model), _c1(model)]
    tokens = [5, 17, 29, 6]
    try:
        assert len(batch.attention_states) == 2, (
            "each QSA layer needs its own paged K/V owner"
        )
        first, second = batch.attention_states
        assert first is not second
        assert first.key_cache.ptr != second.key_cache.ptr
        assert first.value_cache.ptr != second.value_cache.ptr
        owners = [batch.admit(slot) for slot in (1, 0)]
        for row, reference in enumerate(references):
            reference.prefill_serial([3, 11, 25][: 2 + row])
        for slot, reference in zip((1, 0), references):
            batch.import_prefilled(slot, owners[(1, 0).index(slot)], source=reference)
        for step, token in enumerate(tokens):
            expected = [reference.step(token) for reference in references]
            actual = batch.step([token, token], slots=[1, 0], owners=owners)
            for row, reference in enumerate(references):
                deviation = _max_abs(actual.logits[row], expected[row].logits)
                assert deviation <= LOGIT_ATOL, (
                    f"step {step} slot {actual.slot_ids[row]} deviation "
                    f"{deviation:.3e} (shared QSA cache would diverge here)"
                )
                assert int(np.argmax(actual.logits[row])) == int(
                    np.argmax(expected[row].logits)
                )
                assert int(actual.positions[row]) == int(reference.position) - 1
            # Every layer's cache must match its own c1 counterpart: a shared
            # cache would leave layer 0 holding layer 1's keys here.
            for layer in range(config.qsa_layer_count):
                keys, values = _slot_kv(batch, 1, layer=layer)
                c1_keys, c1_values = _c1_kv(references[0], layer=layer)
                assert (
                    _max_abs(_float_view(keys), _float_view(c1_keys)) <= STATE_ATOL
                )
                assert (
                    _max_abs(_float_view(values), _float_view(c1_values))
                    <= STATE_ATOL
                )
    finally:
        for reference in references:
            reference.close()
        batch.close()


def test_packed_step_metadata_uses_one_upload_and_stable_views(two_qsa_model, monkeypatch):
    """Read the packed bytes after permutations, shrinking and slot reuse."""
    import hipengine.runtime.qwen4_exp_batch_state as metadata_module

    batch = _batch(two_qsa_model, slots=3)
    uploads = []
    original_copy = metadata_module.copy_host_to_device

    def record_copy(destination, *args, **kwargs):
        if destination.ptr == batch.step_metadata.buffer.ptr:
            uploads.append(destination.ptr)
        return original_copy(destination, *args, **kwargs)

    monkeypatch.setattr(metadata_module, "copy_host_to_device", record_copy)
    try:
        owners = {slot: batch.admit(slot) for slot in (0, 1, 2)}
        pointers = [
            (state.position.ptr, state.context.ptr, index.positions.ptr)
            for state, index in zip(batch.attention_states, batch.index_states, strict=True)
        ]
        schedule = (((2, 0), (3, 11)), ((0,), (7,)), ((1, 2, 0), (9, 5, 13)))
        for slots, tokens in schedule:
            before = len(uploads)
            batch.step(tokens, slots=slots, owners=[owners[slot] for slot in slots])
            assert len(uploads) == before + 1
            host = batch.step_metadata.host
            actual = np.empty_like(host)
            copy_device_to_host(host_array_ptr(actual), batch.step_metadata.buffer, runtime=batch.runtime)
            np.testing.assert_array_equal(actual, host)
            for layer, (attention, index) in enumerate(zip(
                batch.attention_states, batch.index_states, strict=True,
            )):
                rows = len(slots)
                np.testing.assert_array_equal(actual[layer, 0, :rows], attention.position_host[:rows])
                np.testing.assert_array_equal(actual[layer, 1, :rows], attention.context_host[:rows])
                np.testing.assert_array_equal(actual[layer, 2, :rows], attention.position_host[:rows] - 1)
                assert pointers[layer] == (attention.position.ptr, attention.context.ptr, index.positions.ptr)
        batch.release(2, owners[2])
        owners[2] = batch.admit(2)
        before = len(uploads)
        batch.step([17], slots=[2], owners=[owners[2]])
        assert len(uploads) == before + 1
        np.testing.assert_array_equal(batch.step_metadata.host[:, 0, 0], 0)
        np.testing.assert_array_equal(batch.step_metadata.host[:, 1, 0], 1)
        before_bytes = batch.step_metadata.host.copy()
        with pytest.raises(ValueError):
            batch.step([17, 19], slots=[2, 2], owners=[owners[2], owners[2]])
        assert len(uploads) == before + 1
        np.testing.assert_array_equal(batch.step_metadata.host, before_bytes)
    finally:
        batch.close()


def test_graph_segments_replay_bit_exact_against_eager(model) -> None:
    """Captured layer segments reproduce the eager step on every later step.

    Capture validation compares one replay against one eager run at the capture
    step. A launch argument that varied per step and was baked into a graph
    would only diverge afterwards, so this drives both owners through a schedule
    that changes width and row order and compares every step's logits and the
    slots' KV bytes.
    """

    graphed = _batch(model)
    eager = _batch(model, graph_segments=False)
    schedule = (
        ((0,), (5,)),
        ((0, 1), (9, 3)),
        ((2, 0, 1), (4, 17, 8)),
        ((1,), (30,)),
        ((1, 2), (6, 12)),
        ((0, 1, 2), (2, 60, 44)),
        ((2, 0, 1), (19, 7, 33)),
        ((0,), (21,)),
    )
    try:
        for slot in range(3):
            graphed.admit(slot)
            eager.admit(slot)
        for slots, tokens in schedule + schedule:
            left = graphed.step(
                list(tokens), slots=list(slots),
                owners=[graphed.owner_of(slot) for slot in slots],
            )
            right = eager.step(
                list(tokens), slots=list(slots),
                owners=[eager.owner_of(slot) for slot in slots],
            )
            assert left.logits.tobytes() == right.logits.tobytes(), slots
            assert left.tokens.tolist() == right.tokens.tolist()
        for slot in range(3):
            graphed_keys, graphed_values = _slot_kv(graphed, slot)
            eager_keys, eager_values = _slot_kv(eager, slot)
            assert graphed_keys.tobytes() == eager_keys.tobytes()
            assert graphed_values.tobytes() == eager_values.tobytes()
        stats = graphed.graph_stats
        assert stats["capture"] > 0 and stats["replay"] > 0, stats
        assert eager.graph_stats["capture"] == 0
    finally:
        graphed.close()
        eager.close()


def test_argmax_scratch_covers_every_slot(model) -> None:
    """The row-wise argmax writes one stage-1 partial per row and block.

    The scratch was sized for one row, so a step wider than the allocation's
    rounding slack wrote past it and corrupted the ids of the real vocabulary
    at width 5 and above.
    """

    from hipengine.kernels.hip_gfx1100.linear.lm_head import (
        lm_head_argmax_stage1_blocks,
    )

    slots = 6
    batch = _batch(model, slots=slots)
    reference = _batch(model, slots=slots)
    try:
        blocks = lm_head_argmax_stage1_blocks(fixture.VOCAB, threads=256)
        assert batch.argmax_values.nbytes >= slots * blocks * 4
        assert batch.argmax_blocks_i32.nbytes >= slots * blocks * 4
        for slot in range(slots):
            batch.admit(slot)
            reference.admit(slot)
        # The fixture's PLE staging ring holds four rows per step, so step the
        # four highest slots: the rows that map past the old one-row scratch.
        rows = list(range(slots - 4, slots))
        for token in (3, 11, 29):
            tokens = [(token + 7 * row) % fixture.VOCAB for row in rows]
            greedy = batch.step(
                tokens, slots=rows, owners=[batch.owner_of(s) for s in rows],
                capture_logits=False,
            )
            full = reference.step(
                tokens, slots=rows, owners=[reference.owner_of(s) for s in rows],
            )
            assert greedy.tokens.tolist() == np.argmax(full.logits, axis=1).tolist()
    finally:
        batch.close()
        reference.close()


def test_pooled_slots_match_the_static_layout_and_return_their_blocks(model) -> None:
    """A shared block pool changes where K/V lives, not what a step computes.

    Slots admitted with their own extents take only the blocks those extents
    need, decode bit-identically to the static one-window-per-slot layout, and
    hand the blocks back on release. Admission refuses an extent the free pool
    cannot hold before it touches anything.
    """

    static = _batch(model)
    pooled = _batch(model, pool_blocks=3)
    block = pooled.block_size
    try:
        assert pooled.pooled and not static.pooled
        assert pooled.pool_blocks == 3
        extents = {0: 2 * block, 2: 20}
        for slot, extent in extents.items():
            static.admit(slot)
            pooled.admit(slot, positions=extent)
        assert pooled.free_block_count == 0
        assert pooled.slot_capacity(0) == 2 * block
        assert pooled.slot_capacity(2) == block
        assert sorted(
            int(b) for slot, n in ((0, 2), (2, 1))
            for b in pooled._block_host[slot, :n]
        ) == [0, 1, 2], "every pooled block is held by exactly one slot"
        with pytest.raises(RuntimeError, match="free blocks"):
            pooled.admit(1, positions=1)
        assert pooled.owner_of(1) == 0
        for token in (7, 19, 33, 2, 60, 44):
            left = static.step(
                [token, token], slots=[0, 2],
                owners=[static.owner_of(0), static.owner_of(2)],
            )
            right = pooled.step(
                [token, token], slots=[0, 2],
                owners=[pooled.owner_of(0), pooled.owner_of(2)],
            )
            assert left.logits.tobytes() == right.logits.tobytes()
        pooled.release(2, pooled.owner_of(2))
        assert pooled.free_block_count == 1
        owner = pooled.admit(1, positions=block)
        assert pooled.slot_capacity(1) == block
        # A step past the slot's own extent is refused before any mutation.
        while pooled.position(1) < block - 1:
            pooled.step([5], slots=[1], owners=[owner])
        pooled.step([5], slots=[1], owners=[owner])
        with pytest.raises(ValueError, match="paged blocks hold"):
            pooled.step([5], slots=[1], owners=[owner])
        assert pooled.poisoned_slots == ()
    finally:
        static.close()
        pooled.close()
