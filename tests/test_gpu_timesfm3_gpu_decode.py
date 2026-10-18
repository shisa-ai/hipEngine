from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.hf_cache import resolve_model_path

FIXTURES = (
    Path(__file__).parent / "fixtures" / "cpu_reference" / "timesfm_3p0_decode.npz",
    Path(__file__).parent / "fixtures" / "cpu_reference" / "timesfm_3p0_decode_edge.npz",
    Path(__file__).parent / "fixtures" / "cpu_reference" / "timesfm_3p0_decode_covmask.npz",
)
PINNED_MODEL_ID = "google/timesfm-3.0-pytorch"


def _snapshot() -> Path | None:
    try:
        path = resolve_model_path(PINNED_MODEL_ID)
    except Exception:
        return None
    return path if path.is_dir() else None


def _gpu_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return _snapshot() is not None


pytestmark = pytest.mark.skipif(
    not all(f.is_file() for f in FIXTURES) or not _gpu_available(),
    reason="TimesFM 3.0 fixtures, checkpoint, or ROCm/HIP runtime not available",
)


def _fixture_kwargs(fixture) -> dict:
    kw: dict = {}
    if "past_only_covariates" in fixture.files:
        kw["past_only_covariates"] = fixture["past_only_covariates"]
    if "past_future_covariates" in fixture.files:
        kw["past_future_covariates"] = fixture["past_future_covariates"]
    if "target_mask" in fixture.files:
        kw["target_mask"] = fixture["target_mask"]
    if "global_mask" in fixture.files:
        kw["mask"] = fixture["global_mask"]
    if "past_only_mask" in fixture.files:
        kw["past_only_mask"] = fixture["past_only_mask"]
    if "past_future_mask" in fixture.files:
        kw["past_future_mask"] = fixture["past_future_mask"]
    return kw


def _run_decode(precision: str, fixture_path: Path):
    from hipengine.loading.timesfm3 import load_timesfm3_model
    from hipengine.runtime.timesfm3_decode import TimesFM3GPUDecoder

    fixture = np.load(fixture_path)
    local = load_timesfm3_model(str(_snapshot()))
    try:
        decoder = TimesFM3GPUDecoder(local, precision=precision)
        try:
            return (
                decoder.decode(fixture["target"], int(fixture["horizon"]), **_fixture_kwargs(fixture)),
                fixture,
            )
        finally:
            decoder.close()
    finally:
        local.free()


@pytest.mark.parametrize("fixture_path", FIXTURES, ids=["base", "edge", "covmask"])
def test_gpu_decode_fp32_strict_parity(fixture_path) -> None:
    out, fixture = _run_decode("fp32", fixture_path)
    assert bool(np.isfinite(out).all())
    # Absolute bound only (measured max 1.3e-5): a relative component would
    # loosen the gate near large forecast values.
    np.testing.assert_allclose(
        out, fixture["decode_logits"], atol=1.0e-4, rtol=0.0
    )
    assert np.abs(out - fixture["decode_logits"]).max() < 5.0e-5


def test_sgemm_f64acc_matches_float64_reference() -> None:
    """The strict fp32 GEMM accumulates in double, not fp32 (NT layout pinned).

    The fp32 decode path routes every Linear through this kernel so its GEMMs
    match the CPU/torch fp32 oracle instead of the arch-dependent rocBLAS
    SGEMM accumulation order.  A regression to fp32 accumulation (or a wrong
    transposed layout) fails this bound: fp64 accumulation leaves only the
    final fp32 rounding (~5e-8 of the output max), while fp32 accumulation
    lands ~1e-6 relative and a layout error is O(1).
    """

    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.timesfm3.timesfm3 import timesfm3_sgemm_f64acc

    rng = np.random.default_rng(0)
    # rows=152 exercises the M-tail (not a multiple of the 32-row tile).
    for rows, in_features, out_features in ((152, 1280, 576), (37, 192, 1280), (64, 96, 96)):
        x = rng.standard_normal((rows, in_features)).astype(np.float32)
        weight = rng.standard_normal((out_features, in_features)).astype(np.float32)
        expected = x.astype(np.float64) @ weight.astype(np.float64).T
        xb, wb, ob = malloc(x.nbytes), malloc(weight.nbytes), malloc(rows * out_features * 4)
        try:
            copy_host_to_device(xb, host_array_ptr(x))
            copy_host_to_device(wb, host_array_ptr(weight))
            timesfm3_sgemm_f64acc(xb.ptr, wb.ptr, ob.ptr, rows, in_features, out_features)
            out = np.empty((rows, out_features), dtype=np.float32)
            copy_device_to_host(host_array_ptr(out), ob)
        finally:
            free(xb)
            free(wb)
            free(ob)
        assert np.abs(out - expected).max() <= 1.0e-6 * np.abs(expected).max()


@pytest.mark.parametrize("fixture_path", FIXTURES, ids=["base", "edge", "covmask"])
def test_gpu_decode_fp16_production_gate(fixture_path) -> None:
    """FP16: max <= 2%, mean <= 0.5% of per-series signal scale vs the oracle."""

    out, fixture = _run_decode("fp16", fixture_path)
    error = np.abs(out.astype(np.float64) - fixture["decode_logits"].astype(np.float64))
    assert bool(np.isfinite(error).all())
    target = fixture["target"]
    mask = fixture["target_mask"] if "target_mask" in fixture.files else None
    for b in range(target.shape[0]):
        for u in range(target.shape[1]):
            m = mask[b, u] if mask is not None else None
            series = target[b, u] if m is None else target[b, u][~m]
            scale = float(np.std(series))
            assert error[b, u].max() / scale <= 0.02, f"b{b} u{u} exceeds 2% of signal scale"
            assert error[b, u].mean() / scale <= 0.005, f"b{b} u{u} exceeds 0.5% mean"


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
@pytest.mark.parametrize("fixture_path", FIXTURES, ids=["base", "edge", "covmask"])
def test_gpu_decode_deterministic(precision, fixture_path) -> None:
    """Repeated decodes must be bitwise identical (same launch schedule)."""

    (out1, fixture) = _run_decode(precision, fixture_path)
    (out2, _) = _run_decode(precision, fixture_path)
    np.testing.assert_array_equal(out1, out2)


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_cached_runner_isolates_request_history(precision) -> None:
    """Reuse real buffers without modifying the persistent zero-offset constant."""
    from hipengine.loading.timesfm3 import load_timesfm3_model
    from hipengine.runtime.timesfm3_decode import TimesFM3GPUDecoder

    with np.load(FIXTURES[2]) as fixture:
        target = fixture["target"]
        horizon = int(fixture["horizon"])
        kwargs = _fixture_kwargs(fixture)
    local = load_timesfm3_model(str(_snapshot()))
    try:
        decoder = TimesFM3GPUDecoder(local, precision=precision)
        try:
            expected = decoder.decode(target, horizon, **kwargs)
            assert np.isfinite(expected).all()
            for history in (target[..., ::-1].copy(), np.full_like(target, np.nan)):
                decoder.decode(history, horizon, **kwargs)
                actual = decoder.decode(target, horizon, **kwargs)
                np.testing.assert_array_equal(actual, expected)
        finally:
            decoder.close()
    finally:
        local.free()


def test_gpu_decode_rejects_unknown_precision() -> None:
    from hipengine.loading.timesfm3 import load_timesfm3_model
    from hipengine.runtime.timesfm3_decode import TimesFM3GPUDecoder

    local = load_timesfm3_model(str(_snapshot()))
    try:
        with pytest.raises(ValueError, match="precision"):
            TimesFM3GPUDecoder(local, precision="bf16")
    finally:
        local.free()


def test_gpu_decode_is_invariant_to_poisoned_device_memory() -> None:
    """A decode must not depend on recycled device memory being zero.

    TimesFM 3.0 writes its single scratch cache pair once and then reads it, so
    this path has no unwritten-slot sweep.  The probe still runs because the
    class of defect is "any per-call buffer whose unread region is assumed to be
    zero" (``docs/KERNELS.md`` "Device-memory hygiene"), and the sibling 2.5
    decoder turned out to have exactly that in its V cache: the batched
    attention GEMMs sweep the not-yet-written AR slots and rely on
    ``0 * garbage == 0``, which is false for a NaN.

    Poison every per-call buffer with ``0xFF`` and require the next decode to be
    bit-identical.
    """

    from hipengine.core.hip import get_hip_runtime
    from hipengine.loading.timesfm3 import load_timesfm3_model
    from hipengine.runtime.timesfm3_decode import TimesFM3GPUDecoder

    from _poison_probe import (
        assert_poison_invariant,
        collect_device_buffers,
        group_by_prefix,
    )

    fixture = np.load(FIXTURES[0])
    target, horizon = fixture["target"], int(fixture["horizon"])
    kwargs = _fixture_kwargs(fixture)
    local = load_timesfm3_model(str(_snapshot()))
    try:
        holder: dict[str, object] = {}

        def reset() -> None:
            existing = holder.pop("decoder", None)
            if existing is not None:
                existing.close()
            decoder = TimesFM3GPUDecoder(local, precision="fp16")
            holder["decoder"] = decoder
            decoder.decode(target, horizon, **kwargs)

        def run():
            return holder["decoder"].decode(target, horizon, **kwargs)

        def collect() -> dict[str, list]:
            # q_offset is initialized once and is never written by decode.
            # Treat it like weights; poisoning it changes the model inputs.
            found = collect_device_buffers(holder["decoder"])
            constants = [(path, buf) for path, buf in found if path.endswith(".q_offset")]
            assert len(constants) == 1, [path for path, _ in found]
            from hipengine.core.memory import copy_device_to_host, host_array_ptr
            offset = np.empty(constants[0][1].nbytes // 4, dtype=np.int32)
            copy_device_to_host(host_array_ptr(offset), constants[0][1], offset.nbytes)
            np.testing.assert_array_equal(offset, np.zeros_like(offset))
            found = [(path, buf) for path, buf in found if not path.endswith(".q_offset")]
            assert found, "no device buffers reachable from the decoder"
            return group_by_prefix(found)

        reset()
        try:
            report = assert_poison_invariant(
                get_hip_runtime(),
                collect,
                run,
                label="TimesFM 3.0 fp16 decode",
                reset=reset,
            )
        finally:
            existing = holder.pop("decoder", None)
            if existing is not None:
                existing.close()
        assert report["buffers"] >= 10, report
    finally:
        local.free()
