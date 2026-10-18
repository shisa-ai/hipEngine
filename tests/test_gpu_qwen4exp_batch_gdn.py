"""N-row GDN token mixer over request-owned recurrent/Conv state slots.

Three contracts are asserted separately, per ``docs/EXECUTION-PROFILES.md``:

* **Ownership (exact in every profile).** Row -> slot mapping, untouched slots,
  permuted and changing active sets, and deterministic repeats are byte-level
  assertions. A failure is a bug, not drift.
* **Arithmetic.** The indexed recurrence owner is the row-split form and the
  projections pick a row-count-dependent GEMM variant, so the declared
  expectation against the c1 mixer is FP32 round-off, not bit equality. Each
  case measures and prints the deviation instead of asserting an invented bound,
  and the bound asserted is the measured envelope with margin.
* **Admission.** A rejected call writes nothing: the slabs are byte-identical
  after a malformed row count, geometry, or slot set.

A CPU oracle (the committed segmented recurrence plus the sigmoid-gated RMSNorm)
runs the whole step over the whole slab, so the mixer is not its own oracle and
slot addressing is checked at the arithmetic level, not only by canaries.
"""

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
from hipengine.kernels.cpu_reference import gdn_prefill_recurrent_segments
from hipengine.kernels.cpu_reference.qwen4_exp import sigmoid_gated_rmsnorm
from hipengine.runtime.qwen4_exp_batch_gdn import (
    Qwen4ExpBatchGdnState,
    run_qwen4_exp_batch_gdn_token_mixer,
)
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpGDNScratch,
    run_qwen4_exp_gdn_token_mixer,
)
from tests.test_gpu_qwen4_exp_runner_gr import _dense_f32_weight

SLOTS = 4
HIDDEN = 8
K_HEADS = 2
V_HEADS = 4
HEAD_DIM = 128
KERNEL = 4
QKV_WIDTH = 2 * K_HEADS * HEAD_DIM + V_HEADS * HEAD_DIM
CORE_WIDTH = V_HEADS * HEAD_DIM
# Measured envelope for the batch-vs-c1 comparison (see the printed maxima: the
# row-split recurrence owner and the row-count-dependent GEMM variant agree with
# the c1 mixer to <= 1.1e-6 absolute / 6.1e-7 relative on these fixtures). The
# asserted bound is that envelope with margin; it is deliberately not bit equality.
C1_ATOL = 4e-6
C1_RTOL = 4e-6
# The CPU oracle shares the projections but not the recurrence's summation order
# (measured <= 9.6e-7 absolute).
ORACLE_ATOL = 4e-6
ORACLE_RTOL = 4e-6


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()

pytestmark = pytest.mark.skipif(
    not HIP_AVAILABLE, reason="HIP runtime is not available"
)


@pytest.fixture(scope="module", autouse=True)
def _build_for_detected_target(hip_test_target_arch):
    from hipengine.kernels.backends import hip_target_arch_environment

    with hip_target_arch_environment(hip_test_target_arch):
        yield


def _upload(array: np.ndarray, runtime, allocations) -> object:
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _download(device, shape, dtype, runtime) -> np.ndarray:
    host = np.empty(shape, dtype=dtype)
    # ``copy_device_to_host`` defaults to the *device* buffer's size, so a scratch
    # wider than this call's row count would overrun the host array.
    assert host.nbytes <= device.nbytes, (host.nbytes, device.nbytes)
    copy_device_to_host(
        host_array_ptr(host), device, nbytes=host.nbytes, runtime=runtime
    )
    return host


def _silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-value))


class Weights:
    """One layer's dense-F32 GGUF projections plus the GDN operand vectors."""

    def __init__(self, rng: np.random.Generator, runtime) -> None:
        self.arrays = {
            "attn_qkv": rng.normal(0.0, 0.15, size=(QKV_WIDTH, HIDDEN)).astype(np.float32),
            "attn_gate": rng.normal(0.0, 0.15, size=(CORE_WIDTH, HIDDEN)).astype(np.float32),
            "ssm_alpha": rng.normal(0.0, 0.15, size=(V_HEADS, HIDDEN)).astype(np.float32),
            "ssm_beta": rng.normal(0.0, 0.15, size=(V_HEADS, HIDDEN)).astype(np.float32),
            "ssm_out": rng.normal(0.0, 0.15, size=(HIDDEN, CORE_WIDTH)).astype(np.float32),
        }
        self.conv_weight = rng.normal(0.0, 0.1, size=(QKV_WIDTH, KERNEL)).astype(np.float32)
        self.dt_bias = rng.normal(-1.0, 0.1, size=V_HEADS).astype(np.float32)
        self.a = (-np.exp(rng.normal(-0.5, 0.1, size=V_HEADS))).astype(np.float32)
        self.norm = rng.normal(1.0, 0.05, size=HEAD_DIM).astype(np.float32)
        self._allocations: list = []
        self._runtime = runtime
        self.device = {
            name: _dense_f32_weight(name, array, runtime, self._allocations)
            for name, array in self.arrays.items()
        }
        self.d_conv_weight = _upload(self.conv_weight, runtime, self._allocations)
        self.d_dt_bias = _upload(self.dt_bias, runtime, self._allocations)
        self.d_a = _upload(self.a, runtime, self._allocations)
        self.d_norm = _upload(self.norm, runtime, self._allocations)

    def close(self) -> None:
        for allocation in reversed(self._allocations):
            free(allocation, runtime=self._runtime)


def _initial_slabs(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Distinct per-slot initial Conv and recurrent state.

    Each slot gets a different scale so a row reading the wrong slot cannot
    accidentally agree with the oracle.
    """

    conv = np.stack(
        [
            rng.normal(0.0, 0.05 * (slot + 1), size=(QKV_WIDTH, KERNEL)).astype(np.float32)
            for slot in range(SLOTS)
        ]
    )
    recurrent = np.stack(
        [
            rng.normal(
                0.0, 0.01 * (slot + 1), size=(V_HEADS, HEAD_DIM, HEAD_DIM)
            ).astype(np.float32)
            for slot in range(SLOTS)
        ]
    )
    return conv, recurrent


def _cpu_step(
    weights: Weights,
    mixed: np.ndarray,
    conv_slab: np.ndarray,
    recurrent_slab: np.ndarray,
    slots: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One step for ``slots`` rows over the whole slab, on the CPU."""

    count = len(slots)
    qkv = mixed @ weights.arrays["attn_qkv"].T
    gate = (mixed @ weights.arrays["attn_gate"].T).reshape(count, V_HEADS, HEAD_DIM)
    alpha = mixed @ weights.arrays["ssm_alpha"].T
    beta_logits = mixed @ weights.arrays["ssm_beta"].T

    conv_next = conv_slab.copy()
    conv_out = np.zeros((count, QKV_WIDTH), np.float32)
    for row, slot in enumerate(slots):
        shifted = np.concatenate(
            (conv_slab[slot][:, 1:], qkv[row][:, None]), axis=1
        ).astype(np.float32)
        conv_next[slot] = shifted
        conv_out[row] = _silu(np.sum(shifted * weights.conv_weight, axis=1))

    value_offset = 2 * K_HEADS * HEAD_DIM
    q_raw = conv_out[:, : K_HEADS * HEAD_DIM].reshape(count, K_HEADS, HEAD_DIM)
    k_raw = conv_out[:, K_HEADS * HEAD_DIM : value_offset].reshape(
        count, K_HEADS, HEAD_DIM
    )
    value = conv_out[:, value_offset:].reshape(count, V_HEADS, HEAD_DIM)
    mapping = np.arange(V_HEADS) % K_HEADS
    query = q_raw[:, mapping]
    key = k_raw[:, mapping]
    query = query / np.sqrt(
        np.sum(query * query, axis=-1, keepdims=True) + np.float32(1e-6)
    ) / np.sqrt(np.float32(HEAD_DIM))
    key = key / np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + np.float32(1e-6))
    beta = 1.0 / (1.0 + np.exp(-beta_logits))
    decay = np.exp(
        weights.a * np.log1p(np.exp(alpha + weights.dt_bias))
    ).astype(np.float32)
    core, recurrent_next = gdn_prefill_recurrent_segments(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_slab,
        list(range(count + 1)),
        slots,
    )
    gated = sigmoid_gated_rmsnorm(core, weights.norm, gate)
    output = gated.reshape(count, CORE_WIDTH) @ weights.arrays["ssm_out"].T
    return output.astype(np.float32), conv_next, recurrent_next


def _run_c1(
    weights: Weights,
    mixed_row: np.ndarray,
    conv_slot: np.ndarray,
    recurrent_slot: np.ndarray,
    runtime,
    allocations: list,
):
    """One independent c1 mixer step for a single request."""

    d_mixed = _upload(mixed_row[None], runtime, allocations)
    d_conv = _upload(conv_slot, runtime, allocations)
    d_recurrent = _upload(recurrent_slot, runtime, allocations)
    scratch = Qwen4ExpGDNScratch.allocate(
        rows=1,
        qkv_width=QKV_WIDTH,
        core_width=CORE_WIDTH,
        scalar_width=V_HEADS,
        hidden=HIDDEN,
        runtime=runtime,
    )
    try:
        output = run_qwen4_exp_gdn_token_mixer(
            d_mixed.ptr,
            weights.device,
            conv_weight_ptr=weights.d_conv_weight.ptr,
            dt_bias_ptr=weights.d_dt_bias.ptr,
            a_ptr=weights.d_a.ptr,
            norm_weight_ptr=weights.d_norm.ptr,
            conv_state_ptr=d_conv.ptr,
            recurrent_state_ptr=d_recurrent.ptr,
            scratch=scratch,
            rows=1,
            hidden=HIDDEN,
            num_k_heads=K_HEADS,
            num_v_heads=V_HEADS,
            head_dim=HEAD_DIM,
            conv_kernel=KERNEL,
            runtime=runtime,
        )
        runtime.device_synchronize()
        return (
            _download(output, (HIDDEN,), np.float32, runtime),
            _download(d_conv, (QKV_WIDTH, KERNEL), np.float32, runtime),
            _download(d_recurrent, (V_HEADS, HEAD_DIM, HEAD_DIM), np.float32, runtime),
        )
    finally:
        scratch.close()


class Harness:
    def __init__(self, rng: np.random.Generator) -> None:
        from hipengine.core.hip import get_hip_runtime

        self.runtime = get_hip_runtime()
        self.weights = Weights(rng, self.runtime)
        self.allocations: list = []
        self.state = Qwen4ExpBatchGdnState.allocate(
            state_slots=SLOTS,
            num_k_heads=K_HEADS,
            num_v_heads=V_HEADS,
            head_dim=HEAD_DIM,
            conv_kernel=KERNEL,
            runtime=self.runtime,
        )
        self.scratch = Qwen4ExpGDNScratch.allocate(
            rows=SLOTS,
            qkv_width=QKV_WIDTH,
            core_width=CORE_WIDTH,
            scalar_width=V_HEADS,
            hidden=HIDDEN,
            runtime=self.runtime,
        )

    def seed(self, conv_slab: np.ndarray, recurrent_slab: np.ndarray) -> None:
        copy_host_to_device(
            self.state.conv_state, host_array_ptr(np.ascontiguousarray(conv_slab)),
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.state.recurrent_state,
            host_array_ptr(np.ascontiguousarray(recurrent_slab)),
            runtime=self.runtime,
        )

    def step(self, slots: list[int], mixed: np.ndarray) -> np.ndarray:
        self.state.prepare(slots)
        d_mixed = _upload(mixed, self.runtime, self.allocations)
        output = run_qwen4_exp_batch_gdn_token_mixer(
            d_mixed.ptr,
            self.weights.device,
            state=self.state,
            scratch=self.scratch,
            rows=len(slots),
            hidden=HIDDEN,
            num_k_heads=K_HEADS,
            num_v_heads=V_HEADS,
            head_dim=HEAD_DIM,
            conv_kernel=KERNEL,
            conv_weight_ptr=self.weights.d_conv_weight.ptr,
            dt_bias_ptr=self.weights.d_dt_bias.ptr,
            a_ptr=self.weights.d_a.ptr,
            norm_weight_ptr=self.weights.d_norm.ptr,
            runtime=self.runtime,
        )
        self.runtime.device_synchronize()
        return _download(output, (len(slots), HIDDEN), np.float32, self.runtime)

    def slabs(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            _download(
                self.state.conv_state, (SLOTS, QKV_WIDTH, KERNEL), np.float32,
                self.runtime,
            ),
            _download(
                self.state.recurrent_state,
                (SLOTS, V_HEADS, HEAD_DIM, HEAD_DIM),
                np.float32,
                self.runtime,
            ),
        )

    def close(self) -> None:
        self.scratch.close()
        self.state.close()
        self.weights.close()
        for allocation in reversed(self.allocations):
            free(allocation, runtime=self.runtime)


@pytest.fixture
def harness():
    built = Harness(np.random.default_rng(5231))
    try:
        yield built
    finally:
        built.close()


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left.astype(np.float64) - right.astype(np.float64)).max())


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.abs(right).max()), 1.0)
    return _max_abs(left, right) / denominator


@pytest.mark.parametrize(
    ("seed", "slots", "steps"),
    [
        (5111, [1], 1),
        (5112, [3, 0], 1),
        (5113, [2, 0, 3, 1], 1),
        (5114, [2, 0, 3, 1], 3),
        (5115, [0, 3], 2),
    ],
)
def test_batch_gdn_step_matches_cpu_oracle_and_c1(seed, slots, steps) -> None:
    rng = np.random.default_rng(seed)
    conv_slab, recurrent_slab = _initial_slabs(rng)
    harness = Harness(rng)
    try:
        harness.seed(conv_slab, recurrent_slab)
        for step in range(steps):
            mixed = rng.normal(0.0, 0.1, size=(len(slots), HIDDEN)).astype(np.float32)
            expected_out, expected_conv, expected_recurrent = _cpu_step(
                harness.weights, mixed, conv_slab, recurrent_slab, slots
            )
            c1_out = np.zeros((len(slots), HIDDEN), np.float32)
            c1_conv = np.zeros((len(slots), QKV_WIDTH, KERNEL), np.float32)
            c1_recurrent = np.zeros(
                (len(slots), V_HEADS, HEAD_DIM, HEAD_DIM), np.float32
            )
            for row, slot in enumerate(slots):
                c1_out[row], c1_conv[row], c1_recurrent[row] = _run_c1(
                    harness.weights, mixed[row], conv_slab[slot],
                    recurrent_slab[slot], harness.runtime, harness.allocations,
                )

            actual_out = harness.step(slots, mixed)
            actual_conv, actual_recurrent = harness.slabs()

            assert np.isfinite(actual_out).all()
            assert np.isfinite(actual_conv).all()
            assert np.isfinite(actual_recurrent).all()
            # Arithmetic: CPU oracle and the independent c1 mixer.
            np.testing.assert_allclose(
                actual_out, expected_out, rtol=ORACLE_RTOL, atol=ORACLE_ATOL
            )
            np.testing.assert_allclose(
                actual_out, c1_out, rtol=C1_RTOL, atol=C1_ATOL
            )
            # Exact ownership: only the selected slots moved, and each moved slot
            # is its own row's result.
            untouched = sorted(set(range(SLOTS)) - set(slots))
            for slot in untouched:
                np.testing.assert_array_equal(
                    actual_conv[slot].view(np.uint32),
                    conv_slab[slot].view(np.uint32),
                    err_msg=f"step {step} wrote Conv slot {slot}",
                )
                np.testing.assert_array_equal(
                    actual_recurrent[slot].view(np.uint32),
                    recurrent_slab[slot].view(np.uint32),
                    err_msg=f"step {step} wrote recurrent slot {slot}",
                )
            for row, slot in enumerate(slots):
                np.testing.assert_allclose(
                    actual_conv[slot], expected_conv[slot], rtol=ORACLE_RTOL,
                    atol=ORACLE_ATOL,
                )
                np.testing.assert_allclose(
                    actual_conv[slot], c1_conv[row], rtol=C1_RTOL, atol=C1_ATOL
                )
                np.testing.assert_allclose(
                    actual_recurrent[slot], expected_recurrent[slot],
                    rtol=ORACLE_RTOL, atol=ORACLE_ATOL,
                )
                np.testing.assert_allclose(
                    actual_recurrent[slot], c1_recurrent[row], rtol=C1_RTOL,
                    atol=C1_ATOL,
                )
            print(
                f"rows={len(slots)} step={step} "
                f"max|expected|={np.abs(expected_out).max():.3e} "
                f"max|oracle-batch|={_max_abs(actual_out, expected_out):.3e} "
                f"max|c1-batch|={_max_abs(actual_out, c1_out):.3e} "
                f"rel|c1-batch|={_relative(actual_out, c1_out):.3e} "
                f"state|c1-batch|={_max_abs(actual_recurrent, expected_recurrent):.3e}"
            )
            # Feed the next step from the state the device actually holds, so the
            # oracle keeps describing the real input rather than its own copy.
            conv_slab, recurrent_slab = actual_conv, actual_recurrent
    finally:
        harness.close()


def test_batch_gdn_repeat_is_byte_identical(harness) -> None:
    rng = np.random.default_rng(5241)
    conv_slab, recurrent_slab = _initial_slabs(rng)
    mixed = rng.normal(0.0, 0.1, size=(2, HIDDEN)).astype(np.float32)
    slots = [2, 0]

    harness.seed(conv_slab, recurrent_slab)
    first = harness.step(slots, mixed)
    first_conv, first_recurrent = harness.slabs()

    harness.seed(conv_slab, recurrent_slab)
    second = harness.step(slots, mixed)
    second_conv, second_recurrent = harness.slabs()

    np.testing.assert_array_equal(first.view(np.uint32), second.view(np.uint32))
    np.testing.assert_array_equal(first_conv.view(np.uint32), second_conv.view(np.uint32))
    np.testing.assert_array_equal(
        first_recurrent.view(np.uint32), second_recurrent.view(np.uint32)
    )


def test_batch_gdn_changing_active_sets_advance_only_their_slots(harness) -> None:
    rng = np.random.default_rng(5251)
    conv_slab, recurrent_slab = _initial_slabs(rng)
    harness.seed(conv_slab, recurrent_slab)

    schedule = [[0, 1], [2, 3], [1], [3, 2, 0]]
    for slots in schedule:
        mixed = rng.normal(0.0, 0.1, size=(len(slots), HIDDEN)).astype(np.float32)
        before_conv, before_recurrent = conv_slab.copy(), recurrent_slab.copy()
        expected_out, expected_conv, expected_recurrent = _cpu_step(
            harness.weights, mixed, conv_slab, recurrent_slab, slots
        )
        actual_out = harness.step(slots, mixed)
        conv_slab, recurrent_slab = harness.slabs()

        np.testing.assert_allclose(
            actual_out, expected_out, rtol=ORACLE_RTOL, atol=ORACLE_ATOL
        )
        for slot in range(SLOTS):
            if slot in slots:
                np.testing.assert_allclose(
                    conv_slab[slot], expected_conv[slot], rtol=ORACLE_RTOL,
                    atol=ORACLE_ATOL,
                )
                np.testing.assert_allclose(
                    recurrent_slab[slot], expected_recurrent[slot],
                    rtol=ORACLE_RTOL, atol=ORACLE_ATOL,
                )
            else:
                # A slot outside this step's active set keeps every byte.
                np.testing.assert_array_equal(
                    conv_slab[slot].view(np.uint32),
                    before_conv[slot].view(np.uint32),
                )
                np.testing.assert_array_equal(
                    recurrent_slab[slot].view(np.uint32),
                    before_recurrent[slot].view(np.uint32),
                )
        # The row-to-slot vector follows the caller's order, not slot order.
        assert harness.state.active.slot_ids == tuple(slots)
        assert harness.state.active.indices_host.tolist() == slots


def test_batch_gdn_rejected_calls_write_nothing(harness) -> None:
    rng = np.random.default_rng(5261)
    conv_slab, recurrent_slab = _initial_slabs(rng)
    harness.seed(conv_slab, recurrent_slab)
    mixed = rng.normal(0.0, 0.1, size=(2, HIDDEN)).astype(np.float32)
    d_mixed = _upload(mixed, harness.runtime, harness.allocations)
    harness.state.prepare([0, 1])

    def _call(**overrides):
        kwargs = dict(
            state=harness.state,
            scratch=harness.scratch,
            rows=2,
            hidden=HIDDEN,
            num_k_heads=K_HEADS,
            num_v_heads=V_HEADS,
            head_dim=HEAD_DIM,
            conv_kernel=KERNEL,
            conv_weight_ptr=harness.weights.d_conv_weight.ptr,
            dt_bias_ptr=harness.weights.d_dt_bias.ptr,
            a_ptr=harness.weights.d_a.ptr,
            norm_weight_ptr=harness.weights.d_norm.ptr,
            runtime=harness.runtime,
        )
        kwargs.update(overrides)
        return run_qwen4_exp_batch_gdn_token_mixer(
            d_mixed.ptr, harness.weights.device, **kwargs
        )

    with pytest.raises(ValueError, match="does not match"):
        _call(rows=1)
    with pytest.raises(ValueError, match="head_dim 128"):
        _call(head_dim=64)
    with pytest.raises(ValueError, match="match the state owner"):
        _call(conv_kernel=KERNEL + 1)
    with pytest.raises(ValueError, match="duplicate slot"):
        harness.state.prepare([0, 0])
    harness.runtime.device_synchronize()

    # Nothing above reached a kernel: the slabs are byte-identical.
    actual_conv, actual_recurrent = harness.slabs()
    np.testing.assert_array_equal(
        actual_conv.view(np.uint32), conv_slab.view(np.uint32)
    )
    np.testing.assert_array_equal(
        actual_recurrent.view(np.uint32), recurrent_slab.view(np.uint32)
    )
