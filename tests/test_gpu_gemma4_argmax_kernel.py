"""Device greedy argmax must reproduce ``np.argmax`` exactly (D10, RED first).

``Gemma4Runner.next_token`` picks greedy tokens with ``int(np.argmax(logits))``
over the host copy of the full vocab logits — 0.016 ms of host argmax plus the
1 MB sync D2H that delivers it (the single ``hipMemcpy`` measured at 1.51 ms
per decode step in the d8 attribution, of which ~0.118 ms is the transfer and
the rest is the queue drain the token dependency forces either way). D10's
first half moves the argmax onto the device after the softcap so the greedy
route transfers (index, value) instead of 262144 floats.

The contract this battery pins, before any kernel exists:

- the device index equals ``np.argmax`` **exactly** on the same buffer —
  first-maximum semantics, not merely a member of the tie set (the softcap
  battery allowed in-set; the token must be the very one the host path picks);
- the device value equals ``x[index]`` bitwise, so a future logprob consumer
  sees the same number ``next_token``'s caller would have read;
- NaN follows ``np.argmax`` (first NaN wins over every finite value);
- chained after the production softcap, the pair matches
  ``np.argmax`` over the device-capped buffer read back — the exact comparator
  the current host path uses, not numpy's independently-rounded tanh.

Battery-only: no model artifact needed.
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
    gemma4_logit_argmax_f32,
    gemma4_logit_argmax_scratch_bytes,
    gemma4_logit_softcap_f32,
)

CAP = 30.0


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


def _device_argmax(values: np.ndarray) -> tuple[int, float]:
    """Run the device argmax over ``values``; return (index, value)."""

    runtime = get_hip_runtime()
    library = build_gemma4_norm(load=True)
    array = np.ascontiguousarray(values, dtype=np.float32)
    buffer = malloc(array.nbytes, runtime=runtime)
    out = malloc(16, runtime=runtime)  # int64 index + f32 value, padded
    scratch_nbytes = gemma4_logit_argmax_scratch_bytes(array.size)
    scratch = malloc(scratch_nbytes, runtime=runtime)
    try:
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        gemma4_logit_argmax_f32(
            buffer.ptr,
            array.size,
            out.ptr,
            scratch_ptr=scratch.ptr,
            scratch_blocks=scratch_nbytes // 12,
            library=library,
            runtime=runtime,
        )
        runtime.stream_synchronize(0)
        raw = np.empty(4, dtype=np.int64)
        copy_device_to_host(host_array_ptr(raw), out, runtime=runtime)
        return int(raw[0]), float(np.float32(raw.view(np.float32)[2]))
    finally:
        free(buffer, runtime=runtime)
        free(out, runtime=runtime)
        free(scratch, runtime=runtime)


def _batteries(vocab: int, seed: int = 20260930) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    tie_heavy = rng.normal(0.0, 2.0, vocab).astype(np.float32)
    tie_heavy[[7, 100003, vocab - 1]] = 20.0  # three-way exact tie, spread out
    saturated = rng.normal(0.0, 1.0, vocab).astype(np.float32)
    saturated[:64] = rng.uniform(12.0, 300.0, 64).astype(np.float32) * CAP
    with_nan = rng.normal(0.0, 3.0, vocab).astype(np.float32)
    with_nan[5] = np.nan
    with_nan[90000] = np.nan
    with_nan[1000] = np.float32(1e30)  # finite value loses to first NaN
    signed_zeros = rng.normal(0.0, 1.0, vocab).astype(np.float32)
    signed_zeros[11] = np.float32(0.0)
    signed_zeros[12] = np.float32(-0.0)  # +0 == -0: index 11 must win
    return {
        "gaussian": rng.normal(0.0, 4.0, vocab).astype(np.float32),
        "heavy-tail": (rng.normal(0.0, 8.0, vocab) * rng.standard_normal(vocab)).astype(
            np.float32
        ),
        "exact-ties": tie_heavy,
        "saturation": saturated,
        "with-nan": with_nan,
        "signed-zeros": signed_zeros,
        "extremes": np.array(
            [0.0, -0.0, 1e-45, -1e-45, 3.4e38, -3.4e38] * 40, dtype=np.float32
        ),
        "single": np.array([1.5], dtype=np.float32),
    }


def test_device_argmax_matches_numpy_argmax_exactly() -> None:
    """Index equality is exact on every battery — first maximum, NaN included."""

    vocab = 262144
    for name, values in _batteries(vocab).items():
        expected_index = int(np.argmax(values))
        expected_value = float(values[expected_index])
        index, value = _device_argmax(values)
        assert index == expected_index, (
            f"{name}: device index {index} != np.argmax {expected_index}"
        )
        raw = np.float32(value)
        source = np.float32(values[index])
        assert raw.view(np.uint32) == source.view(np.uint32), (
            f"{name}: device value not bitwise equal to x[index]"
        )


def test_device_argmax_prefers_first_of_equal_maxima() -> None:
    """A flat top resolves to the lowest index, like np.argmax — not in-set."""

    vocab = 4096
    values = np.full(vocab, np.float32(-1.0), dtype=np.float32)
    values[3000] = np.float32(7.0)
    values[3005] = np.float32(7.0)
    values[3010] = np.float32(7.0)
    assert int(np.argmax(values)) == 3000
    index, value = _device_argmax(values)
    assert index == 3000
    assert float(np.float32(value)) == 7.0

    # Every element equal: index 0 wins.
    flat = np.ones(vocab, dtype=np.float32)
    assert int(np.argmax(flat)) == 0
    index, _ = _device_argmax(flat)
    assert index == 0


def test_device_argmax_after_softcap_matches_the_host_production_path() -> None:
    """softcap -> device argmax equals np.argmax over the same capped buffer.

    The production host path is: device softcap, D2H, ``np.argmax`` over what
    came back. The chained device path must return the identical token — the
    comparator is numpy run over the device-produced bytes, not numpy's own
    tanh (which rounds independently; that tolerance is the softcap battery's
    contract, already pinned at two ulps with in-set tie semantics).
    """

    runtime = get_hip_runtime()
    library = build_gemma4_norm(load=True)
    vocab = 262144
    rng = np.random.default_rng(613)

    batteries = {
        "gaussian": rng.normal(0.0, 3.0, vocab).astype(np.float32),
        "boundary-band": rng.uniform(-10.0, 10.0, vocab).astype(np.float32) * CAP,
        "ties": np.repeat(rng.normal(0.0, 6.0, vocab // 2).astype(np.float32), 2),
    }
    for name, raw_values in batteries.items():
        array = np.ascontiguousarray(raw_values, dtype=np.float32)
        buffer = malloc(array.nbytes, runtime=runtime)
        out = malloc(16, runtime=runtime)
        scratch_nbytes = gemma4_logit_argmax_scratch_bytes(array.size)
        scratch = malloc(scratch_nbytes, runtime=runtime)
        try:
            copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
            gemma4_logit_softcap_f32(
                buffer.ptr, array.size, CAP, library=library, runtime=runtime
            )
            gemma4_logit_argmax_f32(
                buffer.ptr,
                array.size,
                out.ptr,
                scratch_ptr=scratch.ptr,
                scratch_blocks=scratch_nbytes // 12,
                library=library,
                runtime=runtime,
            )
            runtime.stream_synchronize(0)
            capped = np.empty_like(array)
            copy_device_to_host(host_array_ptr(capped), buffer, runtime=runtime)
            raw = np.empty(4, dtype=np.int64)
            copy_device_to_host(host_array_ptr(raw), out, runtime=runtime)
        finally:
            free(buffer, runtime=runtime)
            free(out, runtime=runtime)
            free(scratch, runtime=runtime)

        # Production comparator: np.argmax over the device-capped bytes.
        host_index = int(np.argmax(capped))
        device_index = int(raw[0])
        assert device_index == host_index, (
            f"{name}: device {device_index} != host path {host_index}"
        )
        device_value = float(np.float32(raw.view(np.float32)[2]))
        assert np.float32(device_value).view(np.uint32) == capped[host_index].view(
            np.uint32
        ), f"{name}: value not bitwise equal to the capped logit"