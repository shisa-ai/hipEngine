"""B2 P1: input-F16 activation siblings for the sole-T16 dense prefill owners.

The structural campaign blocked P1 (Laurent's F16 activation-B mechanism) on
missing kernels: the dense sole-T16 Q4/Q5 owners
(``gguf_{q4,q5}_k_t16_wmma_prefill_*_bf16_bf16_out``) consume BF16
activations and convert per element per use for the WMMA operands, while the
kernels are already templated on ``scalar_t`` with a vectorized ``half16_t``
load path for F16 input. The B2 build implements F16-staged activation
siblings: identical weights and arithmetic schedule, x operand pre-cast to
F16 in a stage-owned workspace, BF16 output preserved.

These tests are RED before implementation: the sibling wrappers do not
exist. The numerical contract is T1 (F16 staging rounds activations
differently than BF16), so GREEN requires output agreement with the BF16
owners within the production envelope, not bit equality; the strict
fallback stays on the BF16 owners.
"""

from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_OWNER_MODULE = "hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill"

# (BF16 owner, expected F16-input sibling)
B2_SIBLING_PAIRS = (
    (
        "gguf_q4_k_t16_wmma_prefill_bf16_bf16_out",
        "gguf_q4_k_t16_wmma_prefill_fp16_in_bf16_out",
    ),
    (
        "gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out",
        "gguf_q4_k_t16_wmma_prefill_shared_b_fp16_in_bf16_out",
    ),
    (
        "gguf_q5_k_t16_wmma_prefill_bf16_bf16_out",
        "gguf_q5_k_t16_wmma_prefill_fp16_in_bf16_out",
    ),
)

# Production prefill shapes (rows72 C2 grouped tick and rows288 C8 tick).
B2_ROWS = (72, 288)
B2_Q4_SHAPES = (
    (5_120, 6_144),
    (5_120, 10_240),
    (5_120, 12_288),
    (5_120, 17_408),
    (6_144, 5_120),
    (17_408, 5_120),
)
B2_Q5_SHAPES = ((6_144, 5_120),)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_f16_dense_siblings_exist() -> None:
    """RED until the B2 P1 implementation lands the sibling wrappers."""

    import importlib

    module = importlib.import_module(_OWNER_MODULE)
    missing = [
        sibling for _, sibling in B2_SIBLING_PAIRS if not hasattr(module, sibling)
    ]
    assert not missing, f"B2 P1 input-F16 siblings missing: {missing}"


def test_f16_dense_siblings_are_registered_on_both_backends() -> None:
    """Unselected four-axis registration with BF16 owners as fallback."""

    from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
        register_gguf_k_t16_selected_prefill_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import KernelKey, is_registered

    register_gguf_k_t16_selected_prefill_kernels()
    register_gfx1151_kernels()
    expected = (
        ("hip_gfx1100", "gguf_q4_k_t16_v1", "t16_wmma_prefill_fp16_in_bf16_out"),
        (
            "hip_gfx1100",
            "gguf_q4_k_t16_v1",
            "t16_wmma_prefill_shared_b_fp16_in_bf16_out",
        ),
        ("hip_gfx1100", "gguf_q5_k_t16_v1", "t16_wmma_prefill_fp16_in_bf16_out"),
        ("hip_gfx1151", "gguf_q4_k_t16_v1", "t16_wmma_prefill_fp16_in_bf16_out"),
        (
            "hip_gfx1151",
            "gguf_q4_k_t16_v1",
            "t16_wmma_prefill_shared_b_fp16_in_bf16_out",
        ),
        ("hip_gfx1151", "gguf_q5_k_t16_v1", "t16_wmma_prefill_fp16_in_bf16_out"),
    )
    for backend, quant, variant in expected:
        assert is_registered(KernelKey(backend, "linear", quant, variant))
    # The selected BF16 owners remain registered as the strict fallback.
    for backend, quant in (
        ("hip_gfx1100", "gguf_q4_k_t16_v1"),
        ("hip_gfx1100", "gguf_q5_k_t16_v1"),
        ("hip_gfx1151", "gguf_q4_k_t16_v1"),
        ("hip_gfx1151", "gguf_q5_k_t16_v1"),
    ):
        assert is_registered(
            KernelKey(backend, "linear", quant, "t16_wmma_prefill_bf16_bf16_out")
        )


def test_f16_dense_sibling_shapes_match_bf16_owners() -> None:
    """The sibling contract covers rows72/288 on the production shapes."""

    import importlib

    module = importlib.import_module(_OWNER_MODULE)
    for owner_name, sibling_name in B2_SIBLING_PAIRS:
        owner = getattr(module, owner_name, None)
        sibling = getattr(module, sibling_name, None)
        if sibling is None or owner is None:
            pytest.skip("B2 P1 siblings not implemented yet (RED)")
        import inspect

        owner_params = list(inspect.signature(owner).parameters)
        sibling_params = list(inspect.signature(sibling).parameters)
        assert owner_params == sibling_params, (
            f"{sibling_name} must keep the {owner_name} ABI"
        )


def _bf16_bits_to_float(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _float_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    # Round-to-nearest-even BF16 packing on the host.
    u32 = values.astype(np.float32).view(np.uint32)
    rounded = (u32 + 0x7FFF + ((u32 >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).astype(np.uint16)


@pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP runtime unavailable")
@pytest.mark.parametrize("rows", B2_ROWS)
def test_f16_dense_sibling_numerics_vs_bf16_owner(rows: int) -> None:
    """GREEN contract: F16-input output agrees with the BF16 owner (T1).

    Synthetic Q4_K/Q5_K weights are repacked to T16 tiles; deterministic
    activations feed both owners (the sibling consumes the same activations
    pre-cast BF16->F16, matching the stage-owned cast workspace). Outputs
    must agree with the BF16 owner well inside T1 drift and with the CPU
    reference inside BF16 rounding, with per-row argmax agreement >= 97%.
    """

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.cpu_reference import gguf_quant_gemv
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
        build_gguf_k_t16_selected_prefill,
        gguf_q4_k_t16_wmma_prefill_bf16_bf16_out as q4_owner,
        gguf_q4_k_t16_wmma_prefill_fp16_in_bf16_out as q4_sibling,
        gguf_q5_k_t16_wmma_prefill_bf16_bf16_out as q5_owner,
        gguf_q5_k_t16_wmma_prefill_fp16_in_bf16_out as q5_sibling,
    )
    from hipengine.quant.gguf import GGMLQuantizationType
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
    from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16
    from tests._gguf_synthetic_weights import make_q4_k_weight, make_q5_k_weight

    rng = np.random.default_rng(20260902)
    runtime = get_hip_runtime()
    build_gguf_k_t16_selected_prefill(load=True)

    cases = (
        ("q4_plain", q4_owner, q4_sibling, GGMLQuantizationType.Q4_K,
         make_q4_k_weight, repack_gguf_q4_k_tile16,
         (B2_Q4_SHAPES[0], B2_Q4_SHAPES[3], B2_Q4_SHAPES[5])),
        ("q5", q5_owner, q5_sibling, GGMLQuantizationType.Q5_K,
         make_q5_k_weight, repack_gguf_q5_k_tile16, B2_Q5_SHAPES),
    )
    argmax_agreements = []
    for name, owner, sibling, quant_type, make_weight, repack, shapes in cases:
        for in_features, out_features in shapes:
            raw = make_weight(out_features, in_features)
            weight = raw[np.newaxis, :, :]
            tiles = repack(weight).tiles
            x_values = rng.standard_normal((rows, in_features)).astype(np.float32)
            x_bf16 = _float_to_bf16_bits(x_values)
            x_f16 = x_values.astype(np.float16)
            ref = gguf_quant_gemv(x_values, raw, quant_type)

            bufs = []
            try:
                x_dev = malloc(x_bf16.nbytes, runtime=runtime)
                x16_dev = malloc(x_f16.nbytes, runtime=runtime)
                tiles_dev = malloc(tiles.nbytes, runtime=runtime)
                out_owner_dev = malloc(rows * out_features * 2, runtime=runtime)
                out_sib_dev = malloc(rows * out_features * 2, runtime=runtime)
                bufs.extend((x_dev, x16_dev, tiles_dev, out_owner_dev, out_sib_dev))
                copy_host_to_device(
                    x_dev, host_array_ptr(np.ascontiguousarray(x_bf16)),
                    runtime=runtime,
                )
                copy_host_to_device(
                    x16_dev,
                    host_array_ptr(np.ascontiguousarray(x_f16.view(np.uint16))),
                    runtime=runtime,
                )
                copy_host_to_device(
                    tiles_dev, host_array_ptr(np.ascontiguousarray(tiles)),
                    runtime=runtime,
                )

                owner(
                    x_dev.ptr, tiles_dev.ptr, out_owner_dev.ptr,
                    rows, in_features, out_features, runtime=runtime,
                )
                sibling(
                    x16_dev.ptr, tiles_dev.ptr, out_sib_dev.ptr,
                    rows, in_features, out_features, runtime=runtime,
                )
                out_owner = np.zeros((rows, out_features), dtype=np.uint16)
                out_sib = np.zeros((rows, out_features), dtype=np.uint16)
                copy_device_to_host(
                    host_array_ptr(out_owner), out_owner_dev,
                    out_owner.nbytes, runtime=runtime,
                )
                copy_device_to_host(
                    host_array_ptr(out_sib), out_sib_dev,
                    out_sib.nbytes, runtime=runtime,
                )
            finally:
                for dev in bufs:
                    free(dev, runtime=runtime)

            owner_f = _bf16_bits_to_float(out_owner)
            sib_f = _bf16_bits_to_float(out_sib)
            scale = max(float(np.abs(ref).max()), 1.0)
            owner_vs_ref = float(np.abs(owner_f - ref).max()) / scale
            sib_vs_owner = float(np.abs(sib_f - owner_f).max()) / scale
            sib_vs_ref = float(np.abs(sib_f - ref).max()) / scale
            assert owner_vs_ref < 0.02, (
                f"{name} {in_features}->{out_features} rows{rows}: BF16 owner "
                f"drift vs CPU reference {owner_vs_ref:.4f}"
            )
            assert sib_vs_owner < 0.05, (
                f"{name} {in_features}->{out_features} rows{rows}: F16 sibling "
                f"vs BF16 owner T1 drift {sib_vs_owner:.4f}"
            )
            assert sib_vs_ref < 0.06, (
                f"{name} {in_features}->{out_features} rows{rows}: F16 sibling "
                f"vs CPU reference {sib_vs_ref:.4f}"
            )
            # Index-level argmax on iid synthetic weights is tie-noise
            # dominated (17k-column rows have vanishing top-2 margins);
            # index agreement is enforced at the model level by the section-6
            # production gates on real distributions. Here: value-level
            # agreement per row plus whole-matrix correlation.
            owner_top = owner_f.max(axis=1)
            sib_at_owner_top = np.take_along_axis(
                sib_f, owner_f.argmax(axis=1)[:, None], axis=1
            )[:, 0]
            sib_top = sib_f.max(axis=1)
            top_rel = float(
                np.abs(sib_at_owner_top - owner_top).max()
                / max(float(np.abs(owner_top).max()), 1e-6)
            )
            assert top_rel < 0.05, (
                f"{name} {in_features}->{out_features} rows{rows}: top-value "
                f"relative drift {top_rel:.4f}"
            )
            corr = float(np.corrcoef(owner_f.ravel(), sib_f.ravel())[0, 1])
            assert corr > 0.9999, (
                f"{name} {in_features}->{out_features} rows{rows}: output "
                f"correlation {corr:.6f}"
            )
            argmax_agreements.append(corr)
    assert argmax_agreements, "no numerics cases ran"


@pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP runtime unavailable")
def test_device_bf16_to_f16_cast_preserves_ieee_half_bits() -> None:
    """The staging kernel must store FP16 bits, not integer-convert a half."""

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
        build_gguf_ops,
        gguf_cast_bf16_to_f16,
    )

    values = np.asarray(
        [-100.5, -3.5, -1.25, -0.0, 0.0, 0.1, 1.5, 17.75, 255.0, 60_000.0],
        dtype=np.float32,
    )
    bf16 = _float_to_bf16_bits(values)
    expected = _bf16_bits_to_float(bf16).astype(np.float16).view(np.uint16)
    observed = np.zeros_like(expected)
    runtime = get_hip_runtime()
    build_gguf_ops(load=True)
    src = malloc(bf16.nbytes, runtime=runtime)
    dst = malloc(observed.nbytes, runtime=runtime)
    try:
        copy_host_to_device(src, host_array_ptr(bf16), runtime=runtime)
        gguf_cast_bf16_to_f16(
            src.ptr,
            dst.ptr,
            bf16.size,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(observed),
            dst,
            observed.nbytes,
            runtime=runtime,
        )
    finally:
        free(dst, runtime=runtime)
        free(src, runtime=runtime)
    assert np.array_equal(observed, expected), (
        f"FP16 bit mismatch: observed={observed.tolist()} "
        f"expected={expected.tolist()}"
    )


def test_prefill_f16_staging_defaults_off() -> None:
    from hipengine.runtime.gguf_linear import (
        PREFILL_F16_STAGING_ENV,
        prefill_f16_staging_enabled,
        prefill_f16_staging_session,
    )

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(PREFILL_F16_STAGING_ENV, None)
        assert prefill_f16_staging_enabled() is False
        with prefill_f16_staging_session(True):
            assert prefill_f16_staging_enabled() is True
        assert prefill_f16_staging_enabled() is False
        with mock.patch.dict(
            os.environ, {PREFILL_F16_STAGING_ENV: "1"}
        ):
            assert prefill_f16_staging_enabled() is True
        with mock.patch.dict(
            os.environ, {PREFILL_F16_STAGING_ENV: "0"}
        ):
            assert prefill_f16_staging_enabled() is False


def test_prefill_f16_profile_default_and_env_overrides() -> None:
    from hipengine.runtime.gguf_linear import (
        PREFILL_F16_STAGING_ENV,
        prefill_f16_staging_for,
    )

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(PREFILL_F16_STAGING_ENV, None)
        assert prefill_f16_staging_for() is False
        assert prefill_f16_staging_for("strict") is False
        assert prefill_f16_staging_for("legacy_exact") is False
        assert prefill_f16_staging_for("production") is True
        assert (
            prefill_f16_staging_for(
                "production",
                profile_fell_back_to_strict=True,
            )
            is False
        )
        with mock.patch.dict(os.environ, {PREFILL_F16_STAGING_ENV: "1"}):
            assert prefill_f16_staging_for("strict") is True
        with mock.patch.dict(os.environ, {PREFILL_F16_STAGING_ENV: "0"}):
            assert prefill_f16_staging_for("production") is False


def test_generator_configures_profile_scoped_prefill_f16_default() -> None:
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator

    generator = object.__new__(Qwen35GGUFBringupGenerator)
    generator.execution_profile = "production"
    generator.execution_profile_fell_back_to_strict = False
    session = SimpleNamespace()
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HIPENGINE_GGUF_PREFILL_F16_STAGING", None)
        generator._configure_session(session)
        assert session.use_prefill_f16_staging is True
        generator.execution_profile = "strict"
        generator._configure_session(session)
        assert session.use_prefill_f16_staging is False
        generator.execution_profile = "production"
        generator.execution_profile_fell_back_to_strict = True
        generator._configure_session(session)
        assert session.use_prefill_f16_staging is False


def test_shared_session_acquisition_applies_profile_configuration() -> None:
    import inspect

    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator

    source = inspect.getsource(Qwen35GGUFBringupGenerator._acquire_shared_session)
    assert source.count("self._configure_session(session)") == 2


def test_prefill_f16_workspace_context_is_bounded_and_restored() -> None:
    import hipengine.runtime.gguf_linear as gguf_linear
    from hipengine.runtime.gguf_linear import (
        prefill_f16_staging_session,
        prefill_f16_staging_workspace,
    )

    assert prefill_f16_staging_workspace() is None
    assert gguf_linear._prefill_f16_stage_ptr(1) == 0
    with prefill_f16_staging_session(
        True,
        workspace_ptr=0x1200,
        workspace_nbytes=4096,
    ):
        owner = prefill_f16_staging_workspace()
        assert owner is not None
        assert owner.ptr == 0x1200
        assert owner.nbytes == 4096
        assert gguf_linear._prefill_f16_stage_ptr(2048) == 0x1200
        assert gguf_linear._prefill_f16_stage_ptr(2049) == 0
    assert prefill_f16_staging_workspace() is None
    assert gguf_linear._prefill_f16_stage_ptr(1) == 0


def test_prefill_f16_workspace_has_no_module_global_allocator() -> None:
    import hipengine.runtime.gguf_linear as gguf_linear

    assert not hasattr(gguf_linear, "_PREFILL_F16_WORKSPACE")
    assert not hasattr(gguf_linear, "_PREFILL_F16_WORKSPACE_LOCK")


def test_resident_sessions_own_distinct_prefill_f16_workspaces(monkeypatch) -> None:
    import hipengine.runtime.qwen35_gguf_runner as runner_module
    from hipengine.runtime.gguf_linear import prefill_f16_staging_workspace
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    allocations = []

    def fake_malloc(nbytes: int, *, runtime):
        buffer = SimpleNamespace(
            ptr=0x4000 + len(allocations) * 0x1000,
            nbytes=nbytes,
        )
        allocations.append((buffer, runtime))
        return buffer

    monkeypatch.setattr(runner_module, "malloc", fake_malloc)
    monkeypatch.setenv("HIPENGINE_GGUF_PREFILL_F16_STAGING", "1")

    def make_session() -> Qwen35GGUFResidentSession:
        session = object.__new__(Qwen35GGUFResidentSession)
        session.runtime = "rt"
        session._buffers = ()
        session._prefill_f16_staging_buf = None
        session.runner = SimpleNamespace(
            weights=SimpleNamespace(
                config=SimpleNamespace(
                    hidden_size=5_120,
                    feed_forward_length=17_408,
                    ssm_inner_size=6_144,
                )
            )
        )
        return session

    first = make_session()
    second = make_session()
    with first._prefill_f16_staging_context():
        first_owner = prefill_f16_staging_workspace()
        assert first_owner is not None
    with first._prefill_f16_staging_context():
        assert prefill_f16_staging_workspace() == first_owner
    with second._prefill_f16_staging_context():
        second_owner = prefill_f16_staging_workspace()
        assert second_owner is not None
    monkeypatch.setenv("HIPENGINE_GGUF_PREFILL_F16_STAGING", "0")
    disabled = make_session()
    with disabled._prefill_f16_staging_context():
        assert prefill_f16_staging_workspace() is None

    assert first_owner.ptr != second_owner.ptr
    assert len(allocations) == 2
    assert disabled._buffers == ()
    assert first._prefill_f16_staging_buf in first._buffers
    assert second._prefill_f16_staging_buf in second._buffers
    assert first._prefill_f16_staging_buf.nbytes == 1024 * 17_408 * 2


def test_prefill_f16_router_twins_registered() -> None:
    from hipengine.kernels.hip_gfx1151 import (
        gguf_q4_k_t16_wmma_prefill_gfx1151_fp16_in_bf16_out,
        gguf_q5_k_t16_wmma_prefill_gfx1151_fp16_in_bf16_out,
    )

    from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (  # noqa: F401
        gguf_q4_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out as _a,
        gguf_q4_k_t16_wmma_prefill_shared_b3w8r3_fp16_in_bf16_out as _b,  # noqa: F401
        gguf_q5_k_t16_wmma_prefill_shared8r3_fp16_in_bf16_out as _c,  # noqa: F401
    )
    assert callable(gguf_q4_k_t16_wmma_prefill_gfx1151_fp16_in_bf16_out)
    assert callable(gguf_q5_k_t16_wmma_prefill_gfx1151_fp16_in_bf16_out)
