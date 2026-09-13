from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.hf_cache import resolve_model_path

FIXTURE = Path(__file__).parent / "fixtures" / "cpu_reference" / "timesfm_2p5_200m_decode.npz"
PINNED_MODEL_ID = "google/timesfm-2.5-200m-pytorch"


def _gpu_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    snapshot = _snapshot()
    if snapshot is None:
        return False
    return True


def _snapshot() -> Path | None:
    try:
        path = resolve_model_path(PINNED_MODEL_ID)
    except Exception:
        return None
    return path if path.is_dir() else None


pytestmark = pytest.mark.skipif(
    not FIXTURE.is_file() or not _gpu_available(),
    reason="TimesFM fixture, checkpoint, or ROCm/HIP runtime not available",
)


def _run_decode(precision: str):
    from hipengine.loading.timesfm import load_timesfm_model
    from hipengine.runtime.timesfm_decode import TimesFMGPUDecoder

    fixture = np.load(FIXTURE)
    inputs, masks, horizon = fixture["inputs"], fixture["masks"], int(fixture["horizon"])
    local = load_timesfm_model(str(_snapshot()))
    try:
        decoder = TimesFMGPUDecoder(local, precision=precision)
        try:
            return decoder.decode(horizon, inputs, masks), fixture
        finally:
            decoder.close()
    finally:
        local.free()


def test_gpu_decode_fp32_strict_parity() -> None:
    (pf, qs, ar), fixture = _run_decode("fp32")
    np.testing.assert_allclose(pf, fixture["renormed_outputs"], atol=5.0e-4, rtol=1.0e-2)
    np.testing.assert_allclose(qs, fixture["quantile_spread"], atol=5.0e-4, rtol=1.0e-2)
    np.testing.assert_allclose(ar, fixture["ar_outputs"], atol=5.0e-4, rtol=1.0e-2)


def test_gpu_decode_fp16_production_gate() -> None:
    """FP16 production path: max <= 2%, mean <= 0.5% of per-series signal scale."""

    (pf, qs, ar), fixture = _run_decode("fp16")
    inputs, masks = fixture["inputs"], fixture["masks"]
    for name, actual, expected in (
        ("renormed_outputs", pf, fixture["renormed_outputs"]),
        ("quantile_spread", qs, fixture["quantile_spread"]),
        ("ar_outputs", ar, fixture["ar_outputs"]),
    ):
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        for b in range(expected.shape[0]):
            scale = float(np.std(inputs[b][~masks[b]]))
            assert error[b].max() / scale <= 0.02, f"{name} b{b} exceeds 2% of signal scale"
            assert error[b].mean() / scale <= 0.005, f"{name} b{b} exceeds 0.5% mean of signal scale"


def test_gpu_decode_rejects_unknown_precision() -> None:
    from hipengine.loading.timesfm import load_timesfm_model
    from hipengine.runtime.timesfm_decode import TimesFMGPUDecoder

    local = load_timesfm_model(str(_snapshot()))
    try:
        with pytest.raises(ValueError, match="precision"):
            TimesFMGPUDecoder(local, precision="bf16")
    finally:
        local.free()


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
@pytest.mark.parametrize("failed_history", [False, True])
def test_request_reuse_matches_fresh_runner(precision, failed_history):
    """Finite and non-finite prior forecasts cannot contaminate a new call."""
    from hipengine.loading.timesfm import load_timesfm_model
    from hipengine.runtime.timesfm_decode import TimesFMGPUDecoder

    with np.load(FIXTURE) as fixture:
        inputs = fixture["inputs"].copy()
        masks = fixture["masks"].copy()
        horizon = int(fixture["horizon"])
    local = load_timesfm_model(str(_snapshot()))
    decoder = TimesFMGPUDecoder(local, precision=precision)
    try:
        masks[:, :decoder.spec.patch_length] = True
        expected = decoder.decode(horizon, inputs, masks)
        history = np.ascontiguousarray(inputs[:, ::-1])
        if failed_history:
            history[:] = np.nan
        previous = decoder.decode(horizon, history, np.zeros_like(masks))
        assert all(np.isfinite(x).all() for x in previous if x is not None) != failed_history
        actual = decoder.decode(horizon, inputs, masks)
        for a, b in zip(actual, expected, strict=True):
            if b is None:
                assert a is None
            else:
                np.testing.assert_array_equal(a, b)
    finally:
        decoder.close()
        local.free()


def test_gpu_decode_is_invariant_to_poisoned_device_memory() -> None:
    """A decode must not depend on recycled device memory being zero.

    The batched attention GEMMs sweep the whole KV cache, including the AR slots
    the prefill has not written yet, and mask those slots with a zero weight:
    ``0 * garbage`` is 0 for finite garbage but ``NaN`` for a NaN, so the sweep
    is only correct while every slot is non-NaN.  The caches were zero-inited
    when they were *allocated*, which made that true for the first call and left
    every later one to chance -- the "correct only by accident of allocation"
    rule in ``docs/KERNELS.md``.  A recycled block holding a NaN (any prior test
    or kernel can leave one) then turned the whole decode NaN.

    The probe in ``tests/_poison_probe.py`` overwrites every per-call buffer
    with ``0xFF`` and requires the next decode to be bit-identical.  Poisoning
    one family at a time localized it to the V caches alone (20480 of 40960
    output values not finite) while every other buffer was bit-identical, which
    is why the fix re-zeros the caches per decode with a memset rather than
    keeping the allocation-time zero.
    """

    from hipengine.core.hip import get_hip_runtime
    from hipengine.loading.timesfm import load_timesfm_model
    from hipengine.runtime.timesfm_decode import TimesFMGPUDecoder

    from _poison_probe import (
        assert_poison_invariant,
        collect_device_buffers,
        group_by_prefix,
    )

    fixture = np.load(FIXTURE)
    inputs, masks, horizon = fixture["inputs"], fixture["masks"], int(fixture["horizon"])
    local = load_timesfm_model(str(_snapshot()))
    try:
        holder: dict[str, object] = {}

        def reset() -> None:
            """Rebuild the decoder, which releases and reallocates its buffers.

            A poison is not undoable, so localization needs a clean state per
            group; a fresh decoder is the public-API way to get one.
            """

            existing = holder.pop("decoder", None)
            if existing is not None:
                existing.close()
            decoder = TimesFMGPUDecoder(local, precision="fp16")
            holder["decoder"] = decoder
            decoder.decode(horizon, inputs, masks)

        def run():
            return holder["decoder"].decode(horizon, inputs, masks)

        reset()

        def collect() -> dict[str, list]:
            found = collect_device_buffers(holder["decoder"])
            names = {path for path, _ in found}
            assert any("caches_k" in name for name in names), sorted(names)
            assert any("caches_v" in name for name in names), sorted(names)
            return group_by_prefix(found)

        groups = collect()
        assert "caches_v" in "".join(groups), sorted(groups)
        try:
            report = assert_poison_invariant(
                get_hip_runtime(),
                collect,
                run,
                label="TimesFM fp16 decode",
                reset=reset,
            )
        finally:
            existing = holder.pop("decoder", None)
            if existing is not None:
                existing.close()
        # The probe is only meaningful if it poisoned the whole per-call set.
        assert report["buffers"] >= 20, report
        assert report["bytes"] >= 4 * 1024 * 1024, report
    finally:
        local.free()
