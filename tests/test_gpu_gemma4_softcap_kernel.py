"""The device softcap must stay inside numpy's f32 semantics.

X7 measured the host softcap — ``np.tanh(logits / cap) * cap`` in
``Gemma4Runner._collect_logits`` — at 0.528 ms of host time per decode step,
about a third of the measured 1.7 ms host wall gap. Applying it on the logits
device buffer before the D2H copy removes that host work, but the device's
f32 ``tanhf`` need not round bit-exactly like numpy's libm, so this gate pins
where the two must agree for the sampler: the greedy argmax position, exact
ties, and the saturating tail. Battery-only — no model artifact needed.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
    build_gemma4_norm,
    gemma4_logit_softcap_f32,
)

CAP = 30.0
# The capped range is [-cap, cap]; two f32 ulps of that range is the widest
# disagreement a correctly-rounded device tanhf may show against numpy's.
_ULP_TOLERANCE = 2.0 * CAP * np.float32(2.0**-23)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(),
    reason="needs ROCm (libamdhip64.so)",
)


def _numpy_reference(values: np.ndarray, cap: float) -> np.ndarray:
    """The production host expression, verbatim (``gemma4.py``)."""

    cap32 = np.float32(cap)
    return (np.tanh(values / cap32) * cap32).astype(np.float32)


def _device_softcap(values: np.ndarray, cap: float) -> np.ndarray:
    runtime = get_hip_runtime()
    library = build_gemma4_norm(load=True)
    array = np.ascontiguousarray(values, dtype=np.float32)
    buffer = malloc(array.nbytes, runtime=runtime)
    try:
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        gemma4_logit_softcap_f32(
            buffer.ptr, array.size, float(cap), library=library, runtime=runtime
        )
        runtime.stream_synchronize(0)
        result = np.empty_like(array)
        copy_device_to_host(host_array_ptr(result), buffer, runtime=runtime)
        return result
    finally:
        free(buffer, runtime=runtime)


def test_device_softcap_matches_numpy_within_two_ulps() -> None:
    """Every element of every battery lands within two f32 ulps of numpy."""

    rng = np.random.default_rng(20260930)
    vocab = 262144
    batteries = {
        "gaussian": rng.normal(0.0, 4.0, vocab).astype(np.float32),
        "heavy-tail": (rng.normal(0.0, 8.0, vocab) * rng.standard_normal(vocab)).astype(
            np.float32
        ),
        # tanh saturates near x/cap ~= 9; exercise the band on both sides.
        "boundary-band": rng.uniform(-10.0, 10.0, vocab).astype(np.float32) * CAP,
        "saturation": rng.uniform(12.0, 400.0, vocab).astype(np.float32) * CAP,
        "signed-ties": np.repeat(rng.normal(0.0, 6.0, vocab // 2).astype(np.float32), 2),
        "extremes": np.array([0.0, -0.0, 1e-45, -1e-45, 3.4e38, -3.4e38] * 40, dtype=np.float32),
    }
    for name, values in batteries.items():
        expected = _numpy_reference(values, CAP)
        actual = _device_softcap(values, CAP)
        assert np.isfinite(actual).all(), f"{name}: non-finite output"
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        assert difference.max() <= _ULP_TOLERANCE, (
            f"{name}: max |device-numpy| = {difference.max()} exceeds "
            f"{_ULP_TOLERANCE}"
        )
        # The saturated tail must collapse to exactly +cap in both paths;
        # that equality is what keeps saturation a tie rather than an order.
        saturated = values >= 12.0 * CAP
        assert np.array_equal(actual[saturated], expected[saturated]), (
            f"{name}: saturated elements disagree bitwise"
        )


def test_device_softcap_keeps_the_greedy_argmax() -> None:
    """Argmax position stays exact when the top is unique, in-set when tied."""

    rng = np.random.default_rng(613)
    vocab = 262144

    # Unique top with a gap wider than the elementwise tolerance: the
    # device must land on the same token.
    values = rng.normal(0.0, 3.0, vocab).astype(np.float32)
    top = int(rng.integers(0, vocab))
    # tanh barely compresses below ~20, so the top must clear the random
    # tail outright: 40 caps to ~29.85 while a 4-sigma draw caps near ~12.
    values[top] = 40.0
    values[(top + 1) % vocab] = -40.0
    expected = _numpy_reference(values, CAP)
    assert np.isfinite(values).all()
    actual = _device_softcap(values, CAP)
    tie_set = np.flatnonzero(expected == expected.max())
    assert tie_set.size == 1, "battery must have a unique top"
    assert int(np.argmax(actual)) == top

    # Exact ties (bitwise-identical tops): device argmax must stay inside
    # the tie set, matching D10's greedy-tie contract.
    tied = rng.normal(0.0, 2.0, vocab).astype(np.float32)
    # Above any 10-sigma draw from the sigma=2 tail, so the pair is the top.
    tied[7] = 20.0
    tied[100003] = 20.0
    expected_ties = _numpy_reference(tied, CAP)
    tie_set = np.flatnonzero(expected_ties == expected_ties.max())
    assert tie_set.size == 2, "battery must have a two-way tie"
    actual_ties = _device_softcap(tied, CAP)
    assert int(np.argmax(actual_ties)) in tie_set

    # Saturation ties: everything past the tanh ceiling caps to +cap in
    # both paths, so the whole saturated set is one tie.
    saturated = rng.normal(0.0, 1.0, vocab).astype(np.float32)
    saturated[:64] = rng.uniform(12.0, 300.0, 64).astype(np.float32) * CAP
    expected_sat = _numpy_reference(saturated, CAP)
    tie_set = np.flatnonzero(expected_sat == expected_sat.max())
    assert tie_set.size == 64, "battery must saturate exactly the first 64"
    actual_sat = _device_softcap(saturated, CAP)
    assert int(np.argmax(actual_sat)) in tie_set


def test_device_softcap_runs_in_place_on_single_element() -> None:
    """The in-place contract holds for the smallest legal launch."""

    values = np.array([1.5], dtype=np.float32)
    expected = _numpy_reference(values, CAP)
    actual = _device_softcap(values, CAP)
    assert actual.size == 1
    assert abs(float(actual[0]) - float(expected[0])) <= _ULP_TOLERANCE