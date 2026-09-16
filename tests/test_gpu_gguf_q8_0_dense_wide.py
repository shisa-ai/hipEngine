"""Correctness tests for the wide-row GGUF Q8_0 prefill GEMM.

``gguf_q8_0_dense_wide.hip`` stages a dequantized weight tile and a converted
activation tile in LDS so each weight byte is read once per 256 rows instead of
once per 4-32 rows. See the ``.hip`` header for the port provenance.

The kernel's arithmetic is f16 operands with f32 accumulation, so bit parity
with the strict f32 coltile family is not the contract. Two oracles are used:

* **f16-simulated reference** (operands rounded to f16, product accumulated in
  float64). This is the binding check: a kernel that claims f16 operands must
  track it to accumulation-order error, and a structural bug shows up here as a
  miss of the f16 rounding scale rather than a small drift.
* **exact float64 reference** (``gguf_q8_0_gemv``). This is the outer safety
  gate; the tolerance is the f16 operand-rounding envelope, not an f32 one.

Coverage:

1. **No-GPU surface** - registry binding for the whole tile family, build plan,
   and the ``in_features % 64`` contract.
2. **GPU correctness** - the tile family against both oracles, over shapes with
   and without tails in rows and output columns.
3. **gfx1151 aliasing** - the production runtime requests
   ``backend="hip_gfx1151"``, so the alias pass must make these reachable under
   that key or the candidate can never be dispatched.
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
from hipengine.kernels.cpu_reference import gguf_q8_0_gemv
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw
from hipengine.kernels.registry import KernelKey, is_registered, resolve

# Reuse the synthetic Q8_0 generator from the existing GEMV test file.
from tests.test_gpu_gguf_k_gemv import make_q8_0_weight

_VARIANTS = {
    "dense_wide256_f32_f32_out": dw.gguf_q8_0_dense_wide256_f32_f32_out,
    "dense_wide64x256_f32_f32_out": dw._WRAPPERS["dense_wide64x256_f32_f32_out"],
    "dense_wide128x128_f32_f32_out": dw._WRAPPERS["dense_wide128x128_f32_f32_out"],
    "dense_wide64x128_f32_f32_out": dw._WRAPPERS["dense_wide64x128_f32_f32_out"],
}


def _hip_available() -> bool:
    """Whether the HIP runtime ``libamdhip64.so`` can be loaded."""

    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _dequantize_weight(qweight: np.ndarray, in_features: int) -> np.ndarray:
    """Dequantize a raw Q8_0 byte matrix to float32 (rows, in_features)."""

    out_features = qweight.shape[0]
    blocks = qweight.reshape(out_features, in_features // 32, 34)
    scales = (
        np.ascontiguousarray(blocks[:, :, :2])
        .view(np.float16)
        .reshape(blocks.shape[:2])
        .astype(np.float32)
    )
    codes = blocks[:, :, 2:].view(np.int8).astype(np.float32)
    return (scales[:, :, None] * codes).reshape(out_features, in_features)


def _f16_reference(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Round both operands to f16 and accumulate the product in float64."""

    xh = x.astype(np.float16).astype(np.float64)
    wh = weight.astype(np.float16).astype(np.float64)
    return xh @ wh.T


# ---------------------------------------------------------------------------
# 1. No-GPU surface: build plan, registry, contracts.
# ---------------------------------------------------------------------------


def test_dense_wide_registry_and_build_plan() -> None:
    """Every tile in the family binds to its wrapper and resolves by exact key."""

    for variant, wrapper in _VARIANTS.items():
        key = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", variant)
        assert is_registered(key), f"{variant} is not registered"
        assert resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q8_0",
            variant=variant,
        ) is wrapper


def test_dense_wide_build_plan_targets_the_wide_source() -> None:
    """The build plan compiles the wide-row source under its own family."""

    plan = dw.plan_gguf_q8_0_dense_wide_build()
    assert plan.family == "gguf_q8_0_dense_wide"
    assert any(str(path).endswith("gguf_q8_0_dense_wide.hip") for path in plan.sources)
    assert str(plan.output_path).endswith("gguf_q8_0_dense_wide.so")


@pytest.mark.parametrize("in_features", [0, -64, 32, 96])
def test_dense_wide_rejects_in_features_that_are_not_k_tile_multiples(
    in_features: int,
) -> None:
    """The kernel stages K in 64-element tiles, so 32 is not enough."""

    with pytest.raises(ValueError, match="multiple of 64"):
        dw._launch(
            "hipengine_gguf_q8_0_dense_wide256_f32_f32_out",
            1,
            2,
            3,
            8,
            in_features,
            16,
        )


def test_dense_wide_is_aliased_under_the_gfx1151_backend() -> None:
    """The runtime requests gfx1151, so the alias pass must carry the family."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels()
    for variant in _VARIANTS:
        key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0", variant)
        assert is_registered(key), (
            f"{variant} is not reachable under the backend the runtime requests; "
            "add the module to _GFX1100_MODULES"
        )


# ---------------------------------------------------------------------------
# 2. GPU correctness.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
@pytest.mark.parametrize("variant", sorted(_VARIANTS))
@pytest.mark.parametrize(
    ("rows", "in_features", "out_features"),
    [
        (256, 256, 128),  # exactly one row block and one column block
        (512, 256, 256),  # multiple row and column blocks
        (300, 192, 200),  # tails in rows and output columns
    ],
)
def test_dense_wide_matches_the_f16_reference(
    variant: str, rows: int, in_features: int, out_features: int
) -> None:
    """The binding check: the kernel implements the f16 arithmetic it claims."""

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    wrapper = _VARIANTS[variant]
    rng = np.random.default_rng(0xD3E5E)

    qweight = make_q8_0_weight(out_features, in_features)
    weight = _dequantize_weight(qweight, in_features)
    x = rng.standard_normal((rows, in_features), dtype=np.float32)
    out = np.zeros((rows, out_features), dtype=np.float32)

    library = dw.build_gguf_q8_0_dense_wide(load=True)
    x_dev = malloc(x.nbytes)
    w_dev = malloc(qweight.nbytes)
    out_dev = malloc(out.nbytes)
    try:
        copy_host_to_device(x_dev, host_array_ptr(x), runtime=runtime)
        copy_host_to_device(w_dev, host_array_ptr(qweight), runtime=runtime)
        wrapper(
            x_dev.ptr,
            w_dev.ptr,
            out_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
    finally:
        for buffer in (x_dev, w_dev, out_dev):
            free(buffer)

    f16_ref = _f16_reference(x, weight)
    exact_ref = gguf_q8_0_gemv(x, qweight)

    # Accumulation-order error only: f32 accumulation over K terms with f16
    # operands. A structural bug misses by the f16 rounding scale instead.
    f16_scale = float(np.max(np.abs(f16_ref)))
    f16_error = float(np.max(np.abs(out - f16_ref)))
    assert f16_error <= 1e-3 * f16_scale, (
        f"{variant}: max|d| vs the f16 reference is {f16_error:.3e} "
        f"({f16_error / f16_scale:.2e} relative), which is the f16 rounding "
        "scale, not accumulation order"
    )

    # Outer gate: within the f16 operand-rounding envelope of the exact result.
    exact_scale = float(np.max(np.abs(exact_ref)))
    exact_error = float(np.max(np.abs(out - exact_ref)))
    assert exact_error <= 1e-2 * exact_scale, (
        f"{variant}: max|d| vs the exact reference is {exact_error:.3e} "
        f"({exact_error / exact_scale:.2e} relative), outside the f16 envelope"
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dense_wide_rejects_a_shape_the_kernel_cannot_stage() -> None:
    """A K that is not a whole number of 64-element tiles must not launch."""

    with pytest.raises(ValueError, match="multiple of 64"):
        dw.gguf_q8_0_dense_wide256_f32_f32_out(1, 2, 3, 8, 96, 16)
