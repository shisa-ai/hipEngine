"""N-row PLE stage over request-owned Conv-history slots.

Three contracts are asserted separately, per ``docs/EXECUTION-PROFILES.md``:

* **Ownership (exact in every profile).** Row -> slot mapping, untouched slots,
  permuted and changing active sets, and deterministic repeats are byte-level
  assertions. A failure is a bug, not drift.
* **Arithmetic.** Stages 1-7, 9, and 10 are the same kernels the single-request
  oracle runs; the projection GEMM picks a row-count-dependent variant and the
  Conv stage changes owner (strict single-row -> request-indexed decode). The
  declared expectation is therefore BF16 rounding equality, not bit equality:
  each case measures and prints the deviation, and the asserted bound is one
  BF16 ULP of the expected magnitude.
* **Admission.** A rejected call writes nothing: the history slab is
  byte-identical after a malformed row count, geometry, or slot set.

A CPU oracle (the committed PLE reference, run per row with that row's slot
history) checks the whole chain, so the stage is not its own oracle and slot
addressing is verified arithmetically, not only by canaries.
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
from hipengine.kernels.cpu_reference.qwen4_exp import PLEConvState, ple_injection
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.qwen4_exp_batch_ple import (
    Qwen4ExpBatchPleState,
    run_qwen4_exp_batch_ple_stage,
)
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpPLEScratch,
    run_qwen4_exp_ple,
)
from tests.test_gpu_qwen4_exp_runner_gr import _dense_f32_weight

SLOTS = 4
BRANCHES = 2
HIDDEN = 4
CHANNELS = BRANCHES * HIDDEN
KERNEL = 4
# BF16 keeps 8 significand bits, so one ULP of a magnitude-``m`` value is
# ``2**-8 * m``; the asserted bound is one ULP with a small margin.
BF16_ULP = 2.0**-8


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


def _upload(array: np.ndarray, runtime, allocations):
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


class Weights:
    """Compact dense-F32 GGUF projections plus the PLE operand vectors."""

    def __init__(self, rng: np.random.Generator, runtime) -> None:
        self.key = rng.normal(0.0, 0.15, size=(CHANNELS, HIDDEN)).astype(np.float32)
        self.value = rng.normal(0.0, 0.15, size=(HIDDEN, HIDDEN)).astype(np.float32)
        self.norm_key = rng.normal(1.0, 0.05, size=(BRANCHES, HIDDEN)).astype(np.float32)
        self.norm_query = rng.normal(1.0, 0.05, size=(BRANCHES, HIDDEN)).astype(np.float32)
        self.norm_conv = rng.normal(1.0, 0.05, size=(BRANCHES, HIDDEN)).astype(np.float32)
        self.conv = rng.normal(0.0, 0.1, size=(CHANNELS, KERNEL)).astype(np.float32)
        self._allocations: list = []
        self._runtime = runtime
        self.device = {
            "ple_key": _dense_f32_weight("ple_key", self.key, runtime, self._allocations),
            "ple_value": _dense_f32_weight(
                "ple_value", self.value, runtime, self._allocations
            ),
        }
        self.d_norm_key = _upload(self.norm_key, runtime, self._allocations)
        self.d_norm_query = _upload(self.norm_query, runtime, self._allocations)
        self.d_norm_conv = _upload(self.norm_conv, runtime, self._allocations)
        self.d_conv = _upload(self.conv, runtime, self._allocations)

    def close(self) -> None:
        for allocation in reversed(self._allocations):
            free(allocation, runtime=self._runtime)


def _initial_history(
    rng: np.random.Generator, history_rows: int
) -> np.ndarray:
    """Distinct per-slot initial history, so a row reading the wrong slot shows."""

    return np.stack(
        [
            rng.normal(
                0.0, 0.05 * (slot + 1), size=(history_rows, CHANNELS)
            ).astype(np.float32)
            for slot in range(SLOTS)
        ]
    )


def _cpu_row(
    weights: Weights,
    residual_bits: np.ndarray,
    embedding: np.ndarray,
    history: np.ndarray,
    dilation: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The committed PLE reference for one row with its own slot history."""

    expected = ple_injection(
        bf16_to_float32(residual_bits)[None],
        embedding[None],
        weights.key,
        weights.value,
        weights.norm_key,
        weights.norm_query,
        weights.norm_conv,
        weights.conv,
        positions=[0],
        state=PLEConvState(history, 0),
        dilation=dilation,
    )
    return float_array_to_bf16_bits(expected.residual), expected.state.history


def _run_c1(
    weights: Weights,
    residual_bits: np.ndarray,
    embedding: np.ndarray,
    history: np.ndarray,
    dilation: int,
    runtime,
    allocations: list,
) -> tuple[np.ndarray, np.ndarray]:
    """One independent single-request stage run for one row."""

    d_residual = _upload(residual_bits, runtime, allocations)
    d_embedding = _upload(embedding, runtime, allocations)
    d_history = _upload(history, runtime, allocations)
    scratch = Qwen4ExpPLEScratch.allocate(
        rows=1, branches=BRANCHES, hidden=HIDDEN, runtime=runtime
    )
    try:
        output = run_qwen4_exp_ple(
            d_residual.ptr,
            d_embedding.ptr,
            weights.device,
            norm_key_ptr=weights.d_norm_key.ptr,
            norm_query_ptr=weights.d_norm_query.ptr,
            norm_conv_ptr=weights.d_norm_conv.ptr,
            conv_weight_ptr=weights.d_conv.ptr,
            conv_history_ptr=d_history.ptr,
            scratch=scratch,
            rows=1,
            branches=BRANCHES,
            hidden=HIDDEN,
            conv_kernel=KERNEL,
            dilation=dilation,
            runtime=runtime,
        )
        runtime.device_synchronize()
        return (
            _download(output, (BRANCHES, HIDDEN), np.uint16, runtime),
            _download(d_history, history.shape, np.float32, runtime),
        )
    finally:
        scratch.close()


class Harness:
    def __init__(self, rng: np.random.Generator, dilation: int) -> None:
        from hipengine.core.hip import get_hip_runtime

        self.runtime = get_hip_runtime()
        self.dilation = dilation
        self.history_rows = (KERNEL - 1) * dilation
        self.weights = Weights(rng, self.runtime)
        self.allocations: list = []
        self.state = Qwen4ExpBatchPleState.allocate(
            state_slots=SLOTS,
            branches=BRANCHES,
            hidden=HIDDEN,
            conv_kernel=KERNEL,
            dilation=dilation,
            runtime=self.runtime,
        )
        self.scratch = Qwen4ExpPLEScratch.allocate(
            rows=SLOTS, branches=BRANCHES, hidden=HIDDEN, runtime=self.runtime
        )

    def seed(self, history: np.ndarray) -> None:
        copy_host_to_device(
            self.state.history,
            host_array_ptr(np.ascontiguousarray(history)),
            runtime=self.runtime,
        )

    def step(self, slots: list[int], residual_bits: np.ndarray, embedding: np.ndarray):
        self.state.prepare(slots)
        d_residual = _upload(residual_bits, self.runtime, self.allocations)
        d_embedding = _upload(embedding, self.runtime, self.allocations)
        output = run_qwen4_exp_batch_ple_stage(
            d_residual.ptr,
            d_embedding.ptr,
            self.weights.device,
            state=self.state,
            scratch=self.scratch,
            norm_key_ptr=self.weights.d_norm_key.ptr,
            norm_query_ptr=self.weights.d_norm_query.ptr,
            norm_conv_ptr=self.weights.d_norm_conv.ptr,
            conv_weight_ptr=self.weights.d_conv.ptr,
            rows=len(slots),
            branches=BRANCHES,
            hidden=HIDDEN,
            conv_kernel=KERNEL,
            dilation=self.dilation,
            runtime=self.runtime,
        )
        self.runtime.device_synchronize()
        return _download(output, (len(slots), BRANCHES, HIDDEN), np.uint16, self.runtime)

    def history(self) -> np.ndarray:
        return _download(
            self.state.history,
            (SLOTS, self.history_rows, CHANNELS),
            np.float32,
            self.runtime,
        )

    def close(self) -> None:
        self.scratch.close()
        self.state.close()
        self.weights.close()
        for allocation in reversed(self.allocations):
            free(allocation, runtime=self.runtime)


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left.astype(np.float64) - right.astype(np.float64)).max())


def _bits_differing(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left.reshape(-1) != right.reshape(-1)))


def _bf16_bound(expected_bits: np.ndarray) -> float:
    return BF16_ULP * max(float(np.abs(bf16_to_float32(expected_bits)).max()), 1e-3)


@pytest.mark.parametrize(
    ("seed", "slots", "steps", "dilation"),
    [
        (5211, [1], 1, 1),
        (5212, [3, 0], 1, 1),
        (5213, [2, 0, 3, 1], 1, 1),
        (5214, [2, 0, 3, 1], 1, 3),
        # History-boundary cases: dilation 1 holds 3 rows, dilation 3 holds 9.
        (5215, [0, 2], 12, 1),
        (5216, [3, 1], 8, 3),
        (5217, [2, 0, 3, 1], 12, 3),
    ],
)
def test_batch_ple_stage_matches_cpu_oracle_and_c1(seed, slots, steps, dilation) -> None:
    rng = np.random.default_rng(seed)
    harness = Harness(rng, dilation)
    try:
        history = _initial_history(rng, harness.history_rows)
        harness.seed(history)
        for step in range(steps):
            residual_bits = float_array_to_bf16_bits(
                rng.normal(0.0, 0.2, size=(len(slots), BRANCHES, HIDDEN)).astype(np.float32)
            )
            embedding = rng.normal(0.0, 0.2, size=(len(slots), HIDDEN)).astype(np.float32)

            expected_bits = np.zeros((len(slots), BRANCHES, HIDDEN), np.uint16)
            expected_history = history.copy()
            c1_bits = np.zeros((len(slots), BRANCHES, HIDDEN), np.uint16)
            c1_history = np.zeros(
                (len(slots), harness.history_rows, CHANNELS), np.float32
            )
            for row, slot in enumerate(slots):
                expected_bits[row], expected_history[slot] = _cpu_row(
                    harness.weights,
                    residual_bits[row],
                    embedding[row],
                    history[slot],
                    dilation,
                )
                c1_bits[row], c1_history[row] = _run_c1(
                    harness.weights,
                    residual_bits[row],
                    embedding[row],
                    history[slot],
                    dilation,
                    harness.runtime,
                    harness.allocations,
                )

            actual_bits = harness.step(slots, residual_bits, embedding)
            actual_history = harness.history()

            actual = bf16_to_float32(actual_bits)
            expected = bf16_to_float32(expected_bits)
            c1 = bf16_to_float32(c1_bits)
            assert np.isfinite(actual).all() and np.isfinite(actual_history).all()
            # Arithmetic: BF16 rounding equality with the CPU oracle and with the
            # independent single-request stage.
            np.testing.assert_allclose(
                actual, expected, rtol=BF16_ULP, atol=_bf16_bound(expected_bits)
            )
            np.testing.assert_allclose(
                actual, c1, rtol=BF16_ULP, atol=_bf16_bound(c1_bits)
            )
            # Exact ownership: only the selected slots moved, and each moved slot
            # is its own row's history.
            for slot in range(SLOTS):
                if slot in slots:
                    continue
                np.testing.assert_array_equal(
                    actual_history[slot].view(np.uint32),
                    history[slot].view(np.uint32),
                    err_msg=f"step {step} wrote history slot {slot}",
                )
            for row, slot in enumerate(slots):
                np.testing.assert_allclose(
                    actual_history[slot], expected_history[slot], rtol=2e-6, atol=2e-6
                )
                np.testing.assert_allclose(
                    actual_history[slot], c1_history[row], rtol=2e-6, atol=2e-6
                )
            print(
                f"rows={len(slots)} dilation={dilation} step={step} "
                f"max|expected|={np.abs(expected).max():.3e} "
                f"max|oracle-batch|={_max_abs(actual, expected):.3e} "
                f"max|c1-batch|={_max_abs(actual, c1):.3e} "
                f"bits-differing={_bits_differing(actual_bits, c1_bits)}/{actual_bits.size} "
                "history|c1-batch|="
                f"{max(_max_abs(actual_history[slot], c1_history[row]) for row, slot in enumerate(slots)):.3e}"
            )
            history = actual_history
    finally:
        harness.close()


def test_batch_ple_conv_route_is_the_indexed_decode() -> None:
    """The stage resolves the committed request-indexed owner, not a fallback."""

    from types import SimpleNamespace

    from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_ple import (
        qwen4_exp_ple_dilated_depthwise_conv_indexed_f32,
    )
    from hipengine.runtime.qwen4_exp_batch_ple import _resolve_indexed_conv

    resolved = _resolve_indexed_conv(
        {"ple_key": SimpleNamespace(backend="hip_gfx1151")}
    )
    assert resolved is qwen4_exp_ple_dilated_depthwise_conv_indexed_f32


def test_batch_ple_repeat_is_byte_identical() -> None:
    rng = np.random.default_rng(5221)
    harness = Harness(rng, 3)
    try:
        history = _initial_history(rng, harness.history_rows)
        residual_bits = float_array_to_bf16_bits(
            rng.normal(0.0, 0.2, size=(2, BRANCHES, HIDDEN)).astype(np.float32)
        )
        embedding = rng.normal(0.0, 0.2, size=(2, HIDDEN)).astype(np.float32)
        slots = [3, 0]

        harness.seed(history)
        first = harness.step(slots, residual_bits, embedding)
        first_history = harness.history()

        harness.seed(history)
        second = harness.step(slots, residual_bits, embedding)
        second_history = harness.history()

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(
            first_history.view(np.uint32), second_history.view(np.uint32)
        )
    finally:
        harness.close()


def test_batch_ple_changing_active_sets_advance_only_their_slots() -> None:
    rng = np.random.default_rng(5231)
    harness = Harness(rng, 1)
    try:
        history = _initial_history(rng, harness.history_rows)
        harness.seed(history)
        for slots in ([0, 1], [2, 3], [1], [3, 2, 0]):
            residual_bits = float_array_to_bf16_bits(
                rng.normal(0.0, 0.2, size=(len(slots), BRANCHES, HIDDEN)).astype(np.float32)
            )
            embedding = rng.normal(0.0, 0.2, size=(len(slots), HIDDEN)).astype(np.float32)
            before = history.copy()
            expected_bits = np.zeros((len(slots), BRANCHES, HIDDEN), np.uint16)
            expected_history = history.copy()
            for row, slot in enumerate(slots):
                expected_bits[row], expected_history[slot] = _cpu_row(
                    harness.weights,
                    residual_bits[row],
                    embedding[row],
                    history[slot],
                    harness.dilation,
                )
            actual_bits = harness.step(slots, residual_bits, embedding)
            history = harness.history()

            np.testing.assert_allclose(
                bf16_to_float32(actual_bits),
                bf16_to_float32(expected_bits),
                rtol=BF16_ULP,
                atol=_bf16_bound(expected_bits),
            )
            for slot in range(SLOTS):
                if slot in slots:
                    np.testing.assert_allclose(
                        history[slot], expected_history[slot], rtol=2e-6, atol=2e-6
                    )
                else:
                    # A slot outside this step's active set keeps every byte.
                    np.testing.assert_array_equal(
                        history[slot].view(np.uint32), before[slot].view(np.uint32)
                    )
            assert harness.state.active.slot_ids == tuple(slots)
            assert harness.state.active.indices_host.tolist() == slots
    finally:
        harness.close()


def test_batch_ple_rejected_calls_write_nothing() -> None:
    rng = np.random.default_rng(5241)
    harness = Harness(rng, 3)
    try:
        history = _initial_history(rng, harness.history_rows)
        harness.seed(history)
        residual_bits = float_array_to_bf16_bits(
            rng.normal(0.0, 0.2, size=(2, BRANCHES, HIDDEN)).astype(np.float32)
        )
        embedding = rng.normal(0.0, 0.2, size=(2, HIDDEN)).astype(np.float32)
        d_residual = _upload(residual_bits, harness.runtime, harness.allocations)
        d_embedding = _upload(embedding, harness.runtime, harness.allocations)
        harness.state.prepare([0, 1])

        def _call(**overrides):
            kwargs = dict(
                state=harness.state,
                scratch=harness.scratch,
                norm_key_ptr=harness.weights.d_norm_key.ptr,
                norm_query_ptr=harness.weights.d_norm_query.ptr,
                norm_conv_ptr=harness.weights.d_norm_conv.ptr,
                conv_weight_ptr=harness.weights.d_conv.ptr,
                rows=2,
                branches=BRANCHES,
                hidden=HIDDEN,
                conv_kernel=KERNEL,
                dilation=harness.dilation,
                runtime=harness.runtime,
            )
            kwargs.update(overrides)
            return run_qwen4_exp_batch_ple_stage(
                d_residual.ptr, d_embedding.ptr, harness.weights.device, **kwargs
            )

        with pytest.raises(ValueError, match="does not match"):
            _call(rows=1)
        with pytest.raises(ValueError, match="match the state owner"):
            _call(dilation=harness.dilation + 1)
        with pytest.raises(ValueError, match="conv_weight_ptr"):
            _call(conv_weight_ptr=0)
        with pytest.raises(ValueError, match="duplicate slot"):
            harness.state.prepare([0, 0])
        harness.runtime.device_synchronize()

        actual = harness.history()
        np.testing.assert_array_equal(actual.view(np.uint32), history.view(np.uint32))
    finally:
        harness.close()
