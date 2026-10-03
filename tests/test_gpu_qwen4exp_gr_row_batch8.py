"""RED/GREEN for the widened-row-reuse GR up composite (ROW_BATCH 8/16).

``ROW_BATCH`` is a template parameter of
``q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32_kernel``. It only changes
how many rows share one dequantized weight set; the per-output K order, the
shuffle tree, the four-wave LDS publication, the serial wave sum and the epilogue
are unchanged, so every instantiation must reproduce the rowbatch4 parent bit for
bit -- including at rows that are not a multiple of the wider batch, where the
tail rows must be masked exactly as the parent masks them.

The 2026-09-06 ``gr-row-reuse`` rejection measured ROW_BATCH 8 with HIDDEN_TILE 1,
a constant 32-accumulator budget that halved weight decodes while doubling
activation loads. These instantiations keep HIDDEN_TILE 2, so the accumulator
budget is 64/128 and neither operand's load count per FMA rises.

**Both instantiations are measured slower than the rowbatch4 parent**
(0.833-0.844x at ROW_BATCH 8, 0.430-0.437x at 16, occupancy-bound: 130 and 256
VGPR against 72, which is 15 and 8 waves per CU against 28) and are therefore
**not registered for dispatch** -- the registry carries no key for them and
nothing selects them. This file exists so the retile's bit-exactness and its
negative are both on record, and so a re-measurement on a machine class with a
wider register file has a checked starting point. Evidence:
``benchmarks/results/2026-10-04-qwen4exp-gr-row-batch-width/``.
"""

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q8
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload

PARENT = "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32"
WAVE_PARENT = "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_wave_scale_f32"
CANDIDATES = (
    "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch8_f32",
    "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch8_wave_scale_f32",
    "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch16_wave_scale_f32",
)


def hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def test_row_batch_rejects_unknown_width():
    with pytest.raises(ValueError):
        q8.gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32(
            0, 0, 0, 0, 0, 4, 32, 4, 8, _row_batch=12
        )


def test_row_batch_rejects_widened_no_gate():
    # ``no_gate`` exists only as the rowbatch4 strict variant; there is no widened
    # no-gate export, so the combination must be refused rather than silently
    # resolving the rowbatch4 symbol.
    with pytest.raises(ValueError):
        q8.gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32(
            0, 0, 0, 0, 0, 4, 32, 4, 8, _row_batch=8, _no_gate=True
        )


def test_row_batch_is_not_reachable_from_dispatch():
    # The retile is bit-exact but measured slower, so it must not be reachable from
    # dispatch: no registered key anywhere may resolve to one of its wrappers. Scanned
    # through the public registry API rather than by naming the three keys, so a
    # future alias or fallback that picked one up would fail here too.
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import register as _register  # noqa: F401
    from hipengine.kernels.registry import registered_keys, resolve

    register_gfx1151_kernels(replace=True)
    widened = tuple(getattr(q8, name) for name in CANDIDATES)
    reachable = []
    for key in registered_keys():
        kernel = resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
            missing="none",
        )
        # Identity, not equality: some registry entries are policy dataclasses with
        # dict fields, which are unhashable and not comparable to a function.
        if any(kernel is candidate for candidate in widened):
            reachable.append(key.display())
    assert reachable == [], reachable


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize(
    "rows,k,h",
    [
        (1, 32, 6),
        (7, 96, 18),
        (9, 320, 2560),
        (37, 320, 2560),
        (257, 320, 2560),
        (512, 320, 2560),
        (1021, 320, 2560),
        (1024, 320, 2560),
    ],
)
def test_row_batch_is_bit_exact_to_rowbatch4(rows, k, h):
    rng = np.random.default_rng(9147)
    x = rng.normal(0, 0.2, (rows, k)).astype(np.float32)
    norm = rng.normal(0, 0.2, (rows, 4, h)).astype(np.float32)
    raw = rng.integers(0, 256, (4 * h, k // 32, 34), dtype=np.uint8)
    scales = rng.uniform(0.0001, 0.003, (4 * h, k // 32)).astype(np.float16)
    raw[..., :2] = scales.view(np.uint8).reshape(4 * h, k // 32, 2)
    raw = raw.reshape(4 * h, -1)
    runtime = get_hip_runtime()
    allocations = []
    # Engagement, not assumption: record the symbol each call actually resolves. The
    # DP-8 failure mode is a comparison between two calls to the SAME kernel that
    # reads as a perfect bit-exact pass, so the bit-exactness result is only
    # meaningful alongside proof that the candidate's own symbol ran.
    resolved: list[str] = []
    real_cached_fn = q8._cached_fn

    def recording_cached_fn(library, symbol, argtypes):
        resolved.append(symbol)
        return real_cached_fn(library, symbol, argtypes)

    q8._cached_fn = recording_cached_fn
    try:
        dx, dn, dw = [_upload(v, runtime, allocations) for v in (x, norm, raw)]
        gates = [_alloc((rows, 4, h), np.float32, runtime, allocations) for _ in range(2)]
        mixed = [_alloc((rows, h), np.float32, runtime, allocations) for _ in range(2)]

        def run(name, i):
            before = len(resolved)
            getattr(q8, name)(
                dx.ptr, dw.ptr, dn.ptr, gates[i].ptr, mixed[i].ptr, rows, k, 4, h, runtime=runtime
            )
            runtime.device_synchronize()
            # The wrapper must have resolved exactly one symbol, and it must be its own.
            assert resolved[before:] == [f"hipengine_{name}"], (name, resolved[before:])
            return (
                _download(gates[i], (rows, 4, h), np.float32, runtime),
                _download(mixed[i], (rows, h), np.float32, runtime),
            )

        expected = run(WAVE_PARENT, 0)
        for candidate in CANDIDATES:
            for _ in range(2):
                for actual, reference in zip(run(candidate, 1), expected):
                    np.testing.assert_array_equal(
                        actual.view(np.uint32), reference.view(np.uint32)
                    )
        # The plain (non-wave-scale) parent is also bit-identical, so the widened
        # variant must match it too -- this is the strict-chain comparison.
        plain = run(PARENT, 1)
        for actual, reference in zip(plain, expected):
            np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
    finally:
        q8._cached_fn = real_cached_fn
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)
