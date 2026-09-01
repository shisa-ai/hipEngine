from __future__ import annotations

import ctypes
from types import MappingProxyType

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.cpu_reference.ops import gguf_quant_gemv
from hipengine.kernels import hip_gfx1100 as gfx1100_backend
from hipengine.kernels.hip_gfx1100.quant import gguf_k_t16_selected_prefill as t16_prefill
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    build_gguf_k_t16_selected_prefill,
    gguf_q4_k_qmicro_t16_dense_dual_wmma_prefill_expanded_meta_silu_bf16_bf16_out,
    gguf_q4_k_qmicro_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out,
    gguf_q4_k_qmicro_t16_wmma_prefill_bf16_bf16_out,
    gguf_q4_k_t16_dense_dual_wmma_prefill_row48_silu_bf16_bf16_out,
    gguf_q4_k_t16_dense_dual_wmma_prefill_row64_silu_bf16_bf16_out,
    gguf_q4_k_t16_dense_dual_wmma_prefill_row128_silu_bf16_bf16_out,
    gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out,
    gguf_q4_k_t16_dense_rowtile_bf16_bf16_out,
    gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.materialize import DeviceTensorAllocation
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q4_K_QMICRO_T16,
    LAYOUT_GGUF_Q4_K_T16,
    Qwen35GGUFDeviceWeight,
    Qwen35GGUFWeightSpec,
    plan_qwen35_gguf_weight_spec,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import (
    repack_gguf_q4_k_tile16,
    repack_gguf_q4_k_tile16_qmicro,
)
from hipengine.core.specdec2_scope import (
    physical_exact_rowtiles_session,
    q4_t16_physical_extra_rowtiles_session,
)
from hipengine.runtime.gguf_linear import (
    clear_gguf_linear_dispatch_cache,
    launch_gguf_linear,
    launch_gguf_linear_pair,
    launch_gguf_linear_pair_silu,
    native_batch_decode_session,
    q4_t16_unequal_pair_prefill_session,
    wmma_prefill_session,
    resolve_gguf_linear_dispatch,
)
from tests._gguf_synthetic_weights import make_q4_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _f32_to_bf16_bits(value: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype(np.uint16)


def _bf16_bits_to_f32(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def _weight(
    ptr: int,
    *,
    in_features: int = 256,
    out_features: int = 16,
    qmicro: bool = False,
    backend: str = "hip_gfx1100",
) -> Qwen35GGUFDeviceWeight:
    source = GGUFTensorInfo(
        name=f"weight.{ptr:x}",
        shape=(out_features, in_features),
        ggml_shape=(in_features, out_features),
        ggml_type=int(GGMLQuantizationType.Q4_K),
        ggml_type_name="Q4_K",
        n_elements=out_features * in_features,
        nbytes=out_features * (in_features // 256) * 144,
        offset=0,
        data_offset=0,
        byte_shape=(out_features, (in_features // 256) * 144),
    )
    quant_key = (
        "gguf_q4_k_qmicro_t16_v1" if qmicro else "gguf_q4_k_t16_v1"
    )
    layout = (
        LAYOUT_GGUF_Q4_K_QMICRO_T16 if qmicro else LAYOUT_GGUF_Q4_K_T16
    )
    spec = Qwen35GGUFWeightSpec(
        slot_path=f"layers.0.weight_{ptr:x}",
        source=source,
        quant_key=quant_key,
        layout=layout,
        allocation_names=("tiles",),
    )
    tile_bytes = 2304 if qmicro else 2368
    nbytes = out_features // 16 * (in_features // 256) * tile_bytes
    buffer = DeviceBuffer(ptr=ptr, nbytes=nbytes)
    allocation = DeviceTensorAllocation(
        name=f"{source.name}.t16.tiles",
        source=source,
        buffer=buffer,
        tensor=Tensor.from_handle(ptr, (nbytes,), DType.INT8, Device("hip", 0)),
    )
    return Qwen35GGUFDeviceWeight(
        spec=spec,
        allocations=MappingProxyType({"tiles": allocation}),
        backend=backend,
    )


def test_gfx1100_routes_physical_r6_q4_shapes_to_c1_rowtile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2", "0")
    monkeypatch.setattr(t16_prefill, "_Q4_ROWTILE16_W2_RESOLVED", None)
    selector = getattr(
        t16_prefill,
        "gguf_q4_k_t16_physical_c1_rowtile_gfx1100_bf16_bf16_out",
        None,
    )
    rows_policy = getattr(
        gfx1100_backend,
        "GGUF_Q4_T16_PHYSICAL_C1_ROWTILE_ROWS",
        None,
    )
    shape_policy = getattr(
        gfx1100_backend,
        "GGUF_Q4_T16_PHYSICAL_C1_ROWTILE_SHAPES",
        None,
    )
    single_wave_policy = getattr(
        gfx1100_backend,
        "GGUF_Q4_T16_PHYSICAL_SINGLE_WAVE_SHAPES",
        None,
    )
    single_wave_max_rows = getattr(
        gfx1100_backend,
        "GGUF_Q4_T16_PHYSICAL_SINGLE_WAVE_MAX_ROWS",
        None,
    )
    single_wave_max_rows_by_shape = getattr(
        gfx1100_backend,
        "GGUF_Q4_T16_PHYSICAL_SINGLE_WAVE_MAX_ROWS_BY_SHAPE",
        None,
    )
    shared_b_row64_shapes = getattr(
        gfx1100_backend,
        "GGUF_Q4_T16_PHYSICAL_SHARED_B_ROW64_SHAPES",
        None,
    )
    shared_b_row64_rows = getattr(
        gfx1100_backend,
        "GGUF_Q4_T16_PHYSICAL_SHARED_B_ROW64_ROWS",
        None,
    )
    shared_b_row64_fn = getattr(
        t16_prefill,
        "gguf_q4_k_t16_wmma_prefill_shared_b_row64_bf16_bf16_out",
        None,
    )
    assert callable(selector)
    assert callable(shared_b_row64_fn)
    assert rows_policy == frozenset({6})
    assert shape_policy == frozenset(
        {
            (5_120, 1_024),
            (5_120, 6_144),
            (5_120, 10_240),
            (5_120, 12_288),
            (17_408, 5_120),
        }
    )
    assert single_wave_policy == frozenset(
        {(5_120, 17_408), (5_120, 10_240), (5_120, 12_288)}
    )
    # Measured on W7900 at (rows, 5120, 17408) against the 256-row shared-B tile:
    # single-wave is bit-identical and 1.43x-1.04x faster for rows 2..128, and
    # 0.87x slower from 144 rows up. See
    # benchmarks/results/2026-08-30-w7900-q4km-t16-single-wave-rows-accepted.json.
    assert single_wave_max_rows == 128
    assert single_wave_max_rows_by_shape == {(5_120, 12_288): 112}
    assert shared_b_row64_shapes == frozenset({(17_408, 5_120)})
    assert shared_b_row64_rows == range(33, 193)

    calls: list[str] = []
    monkeypatch.setattr(
        t16_prefill,
        "gguf_q4_k_t16_dense_rowtile_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("rowtile"),
        raising=False,
    )
    monkeypatch.setattr(
        t16_prefill,
        "gguf_q4_k_t16_wmma_prefill_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("single_wave"),
    )
    monkeypatch.setattr(
        t16_prefill,
        "gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("shared_b"),
    )
    monkeypatch.setattr(
        t16_prefill,
        "gguf_q4_k_t16_wmma_prefill_shared_b_row64_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("shared_b_row64"),
    )
    for in_features, out_features in shape_policy:
        selector(1, 2, 3, 6, in_features, out_features)
    selector(1, 2, 3, 6, 5_120, 17_408)
    with q4_t16_physical_extra_rowtiles_session(True):
        selector(1, 2, 3, 6, 5_120, 17_408)
        selector(1, 2, 3, 6, 6_144, 5_120)
    selector(1, 2, 3, 5, 5_120, 17_408)
    selector(1, 2, 3, 6, 5_120, 5_120)
    selector(1, 2, 3, 35, 5_120, 17_408)
    selector(1, 2, 3, 128, 5_120, 17_408)
    selector(1, 2, 3, 129, 5_120, 17_408)
    selector(1, 2, 3, 35, 17_408, 5_120)
    selector(1, 2, 3, 6, 5_120, 10_240)
    selector(1, 2, 3, 35, 5_120, 10_240)
    selector(1, 2, 3, 35, 5_120, 12_288)
    selector(1, 2, 3, 35, 5_120, 6_144)
    selector(1, 2, 3, 112, 5_120, 12_288)
    selector(1, 2, 3, 113, 5_120, 12_288)
    selector(1, 2, 3, 128, 5_120, 10_240)
    selector(1, 2, 3, 67, 17_408, 5_120)
    selector(1, 2, 3, 192, 17_408, 5_120)
    selector(1, 2, 3, 193, 17_408, 5_120)
    assert calls == ["rowtile"] * len(shape_policy) + [
        "single_wave",
        "rowtile",
        "rowtile",
        "single_wave",
        "shared_b",
        "single_wave",
        "single_wave",
        "shared_b",
        "shared_b_row64",
        "rowtile",
        "single_wave",
        "single_wave",
        "shared_b",
        "single_wave",
        "shared_b",
        "single_wave",
        "shared_b_row64",
        "shared_b_row64",
        "shared_b",
    ]

    # Bisection switch: forcing the band to 0 must return every non-row-6 call to
    # the strict shared-B sibling without touching the qualified row-6 routes.
    calls.clear()
    monkeypatch.setenv(t16_prefill._ENV_SINGLE_WAVE_MAX_ROWS, "0")
    monkeypatch.setattr(t16_prefill, "_SINGLE_WAVE_MAX_ROWS_RESOLVED", None)
    selector(1, 2, 3, 5, 5_120, 17_408)
    selector(1, 2, 3, 35, 5_120, 17_408)
    selector(1, 2, 3, 6, 5_120, 17_408)
    assert calls == ["shared_b", "shared_b", "single_wave"]
    monkeypatch.delenv(t16_prefill._ENV_SINGLE_WAVE_MAX_ROWS)
    monkeypatch.setattr(t16_prefill, "_SINGLE_WAVE_MAX_ROWS_RESOLVED", None)

    # Same-build bisection: zero restores the 256-row parent without changing
    # row-6 precedence or widening the qualified 33..67 policy.
    calls.clear()
    monkeypatch.setenv(t16_prefill._ENV_SHARED_B_ROW64_MAX_ROWS, "0")
    monkeypatch.setattr(t16_prefill, "_SHARED_B_ROW64_MAX_ROWS_RESOLVED", None)
    selector(1, 2, 3, 35, 17_408, 5_120)
    selector(1, 2, 3, 192, 17_408, 5_120)
    selector(1, 2, 3, 6, 17_408, 5_120)
    assert calls == ["shared_b", "shared_b", "rowtile"]
    monkeypatch.delenv(t16_prefill._ENV_SHARED_B_ROW64_MAX_ROWS)
    monkeypatch.setattr(t16_prefill, "_SHARED_B_ROW64_MAX_ROWS_RESOLVED", None)

    selected = resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k_t16_v1",
        variant="t16_physical_c1_rowtile_bf16_bf16_out",
    )
    single_wave = resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k_t16_v1",
        variant="t16_wmma_prefill_single_wave_bf16_bf16_out",
    )
    fallback = resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k_t16_v1",
        variant="t16_wmma_prefill_shared_b_bf16_bf16_out",
    )
    row64 = resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k_t16_v1",
        variant="t16_wmma_prefill_shared_b_row64_bf16_bf16_out",
    )
    default = resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k_t16_v1",
        variant="t16_wmma_prefill_bf16_bf16_out",
    )
    assert selected is gguf_q4_k_t16_dense_rowtile_bf16_bf16_out
    assert single_wave is gguf_q4_k_t16_wmma_prefill_bf16_bf16_out
    assert fallback is gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out
    assert row64 is shared_b_row64_fn
    assert default is selector


def test_gfx1100_physical_r6_two_wave_policy_routes_target_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = (
        t16_prefill.gguf_q4_k_t16_physical_c1_rowtile_gfx1100_bf16_bf16_out
    )
    calls: list[str] = []
    monkeypatch.setattr(
        t16_prefill,
        "gguf_q4_k_t16_dense_rowtile_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("parent"),
    )
    monkeypatch.setattr(
        t16_prefill,
        "gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("two_wave"),
        raising=False,
    )

    monkeypatch.setenv("HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2", "1")
    monkeypatch.setattr(
        t16_prefill,
        "_Q4_ROWTILE16_W2_RESOLVED",
        None,
        raising=False,
    )
    with q4_t16_physical_extra_rowtiles_session(True):
        selector(1, 2, 3, 6, 5_120, 17_408)
        selector(1, 2, 3, 24, 5_120, 17_408)
        selector(1, 2, 3, 36, 5_120, 17_408)
    selector(1, 2, 3, 6, 17_408, 5_120)

    monkeypatch.setenv("HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2", "0")
    monkeypatch.setattr(t16_prefill, "_Q4_ROWTILE16_W2_RESOLVED", None)
    with q4_t16_physical_extra_rowtiles_session(True):
        selector(1, 2, 3, 6, 5_120, 17_408)

    assert calls == ["two_wave"] * 11 + ["parent", "parent"]


def test_gfx1100_physical_r6_two_wave_qkv_shape_defaults_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = (
        t16_prefill.gguf_q4_k_t16_physical_c1_rowtile_gfx1100_bf16_bf16_out
    )
    calls: list[str] = []
    monkeypatch.setattr(
        t16_prefill,
        "gguf_q4_k_t16_dense_rowtile_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("parent"),
    )
    monkeypatch.setattr(
        t16_prefill,
        "gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("two_wave"),
    )
    monkeypatch.delenv("HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2", raising=False)
    monkeypatch.setattr(t16_prefill, "_Q4_ROWTILE16_W2_RESOLVED", None)

    with q4_t16_physical_extra_rowtiles_session(True):
        selector(1, 2, 3, 6, 5_120, 10_240)
        monkeypatch.setenv("HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2", "0")
        monkeypatch.setattr(t16_prefill, "_Q4_ROWTILE16_W2_RESOLVED", None)
        selector(1, 2, 3, 6, 5_120, 10_240)

    assert calls == ["two_wave", "parent"]


def test_gfx1100_unpadded_r8_q4_rowtile_is_candidate_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = (
        t16_prefill.gguf_q4_k_t16_physical_c1_rowtile_gfx1100_bf16_bf16_out
    )
    calls: list[str] = []
    monkeypatch.setattr(
        t16_prefill,
        "gguf_q4_k_t16_dense_rowtile_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("rowtile"),
    )
    monkeypatch.setattr(
        t16_prefill,
        "gguf_q4_k_t16_wmma_prefill_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("single_wave"),
    )
    monkeypatch.setattr(
        t16_prefill,
        "gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("shared_b"),
    )

    with q4_t16_physical_extra_rowtiles_session(True):
        selector(1, 2, 3, 8, 5_120, 6_144)
        selector(1, 2, 3, 8, 5_120, 17_408)
        with physical_exact_rowtiles_session(True):
            selector(1, 2, 3, 8, 5_120, 6_144)
            selector(1, 2, 3, 8, 5_120, 17_408)

    assert calls == ["shared_b", "single_wave", "rowtile", "rowtile"]


def test_q4_t16_unequal_dual_prefill_leaf_contract() -> None:
    wrapper = getattr(
        t16_prefill,
        "gguf_q4_k_t16_dense_unequal_dual_wmma_prefill_bf16_bf16_out",
        None,
    )
    assert callable(wrapper)
    key = KernelKey(
        "hip_gfx1100",
        "linear_pair",
        "gguf_q4_k_t16_v1",
        "dense_unequal_dual_wmma_prefill_bf16_bf16_out",
    )
    assert resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    ) is wrapper
    source = t16_prefill._SOURCE.read_text(encoding="utf-8")
    assert "torch::Tensor" not in source
    assert (
        "hipengine_gguf_q4_k_t16_dense_unequal_dual_wmma_prefill_bf16_bf16_out"
        in source
    )
    with pytest.raises(ValueError, match="out_features_a must be at least out_features_b"):
        wrapper(1, 2, 3, 4, 5, 512, 5_120, 6_144, 10_240)
    with pytest.raises(ValueError, match="multiples of 32"):
        wrapper(1, 2, 3, 4, 5, 512, 5_120, 10_240, 6_160)


def test_q4_qmicro_dense_gate_up_planning_is_role_and_shape_bounded() -> None:
    def source(out_features: int, in_features: int) -> GGUFTensorInfo:
        return GGUFTensorInfo(
            name=f"weight.{out_features}.{in_features}",
            shape=(out_features, in_features),
            ggml_shape=(in_features, out_features),
            ggml_type=int(GGMLQuantizationType.Q4_K),
            ggml_type_name="Q4_K",
            n_elements=out_features * in_features,
            nbytes=out_features * (in_features // 256) * 144,
            offset=0,
            data_offset=0,
            byte_shape=(out_features, (in_features // 256) * 144),
        )

    for role in ("ffn_gate", "ffn_up"):
        spec = plan_qwen35_gguf_weight_spec(
            f"layers.0.{role}",
            source(17_408, 5_120),
            decode_repack=True,
            dense_q4_t16=True,
            dense_q4_qmicro_t16_gate_up=True,
        )
        assert spec.layout == LAYOUT_GGUF_Q4_K_QMICRO_T16
        assert spec.quant_key == "gguf_q4_k_qmicro_t16_v1"
        assert spec.allocation_names == ("tiles",)

    for slot_path, shape, expected_layout in (
        ("layers.0.ffn_down", (5_120, 17_408), LAYOUT_GGUF_Q4_K_T16),
        ("layers.0.attn_q", (12_288, 5_120), LAYOUT_GGUF_Q4_K_T16),
        ("layers.0.ffn_gate", (3_584, 1_024), "q4_k_pack8"),
    ):
        spec = plan_qwen35_gguf_weight_spec(
            slot_path,
            source(*shape),
            decode_repack=True,
            dense_q4_t16=True,
            dense_q4_qmicro_t16_gate_up=True,
        )
        assert spec.layout == expected_layout
        assert spec.quant_key == (
            "gguf_q4_k_t16_v1"
            if expected_layout == LAYOUT_GGUF_Q4_K_T16
            else "gguf_q4_k"
        )


def test_q4_t16_dense_dispatch_uses_one_tiles_abi_for_decode_and_prefill() -> None:
    weight = _weight(0x1000)

    decode = resolve_gguf_linear_dispatch(weight, rows=1)
    prefill = resolve_gguf_linear_dispatch(weight, rows=512)

    assert decode.key == KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "dense_single_local32_bf16_bf16_out",
    )
    assert prefill.key == KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "t16_wmma_prefill_bf16_bf16_out",
    )
    assert decode.abi == prefill.abi == "t16"


@pytest.mark.parametrize(
    "qmicro,quant",
    [
        (False, "gguf_q4_k_t16_v1"),
        (True, "gguf_q4_k_qmicro_t16_v1"),
    ],
)
@pytest.mark.parametrize("native_rows", [2, 3, 4, 5, 6, 7, 8])
def test_q4_dense_launch_routes_c1_small_rows_and_bulk_without_shadow_allocations(
    qmicro: bool,
    quant: str,
    native_rows: int,
) -> None:
    weight = _weight(0x1000, qmicro=qmicro)
    keys = (
        KernelKey(
            "hip_gfx1100",
            "linear",
            quant,
            "dense_single_local32_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1100",
            "linear",
            quant,
            "dense_rowtile_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1100",
            "linear",
            quant,
            "t16_wmma_prefill_bf16_bf16_out",
        ),
    )
    originals = {key: resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant) for key in keys}
    calls: list[tuple[str, tuple]] = []

    try:
        for key in keys:
            register(
                key,
                lambda *args, _variant=key.variant, **kwargs: calls.append((_variant, args)),
                replace=True,
            )
        clear_gguf_linear_dispatch_cache()
        launch_gguf_linear(weight, 0x2000, 0x3000, 1, 256, 16)
        with native_batch_decode_session(True):
            launch_gguf_linear(
                weight,
                0x2000,
                0x3000,
                native_rows,
                256,
                16,
            )
        launch_gguf_linear(weight, 0x2000, 0x3000, 512, 256, 16)
    finally:
        for key, fn in originals.items():
            register(key, fn, replace=True)
        clear_gguf_linear_dispatch_cache()

    native_variant = (
        "dense_rowtile_bf16_bf16_out"
        if not qmicro or native_rows <= 4
        else "t16_wmma_prefill_bf16_bf16_out"
    )
    assert [variant for variant, _args in calls] == [
        "dense_single_local32_bf16_bf16_out",
        native_variant,
        "t16_wmma_prefill_bf16_bf16_out",
    ]
    assert all(args[:3] == (0x2000, 0x1000, 0x3000) for _variant, args in calls)


@pytest.mark.parametrize(
    ("native_rows", "native_label"),
    [(3, "native"), (5, None), (8, None)],
)
def test_q4_qmicro_dense_pair_silu_routes_c1_native_rows_and_prefill(
    monkeypatch: pytest.MonkeyPatch,
    native_rows: int,
    native_label: str | None,
) -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.runtime import gguf_linear as gguf_linear_module

    register_gfx1151_kernels(replace=True)
    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_quantize_bf16_q8_1x2",
        lambda *args, **kwargs: None,
    )
    weight_a = _weight(
        0x1000,
        in_features=5_120,
        out_features=17_408,
        qmicro=True,
        backend="hip_gfx1151",
    )
    weight_b = _weight(
        0x2000,
        in_features=5_120,
        out_features=17_408,
        qmicro=True,
        backend="hip_gfx1151",
    )
    quant = "gguf_q4_k_qmicro_t16_v1"
    keys = {
        "c1": KernelKey(
            "hip_gfx1151",
            "linear_pair_silu",
            quant,
            "dense_dual_q8_1x2_split_weight_dp4a_bf16_bf16_out",
        ),
        "native": KernelKey(
            "hip_gfx1151",
            "linear_pair_silu",
            quant,
            "dense_dual_rowtile_bf16_bf16_out",
        ),
        "direct": KernelKey(
            "hip_gfx1151",
            "linear_pair_silu",
            quant,
            "dense_dual_wmma_prefill_bf16_bf16_out",
        ),
        "expanded": KernelKey(
            "hip_gfx1151",
            "linear_pair_silu",
            quant,
            "dense_dual_wmma_prefill_expanded_meta_bf16_bf16_out",
        ),
    }
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in keys.values()
    }
    calls: list[tuple[str, tuple]] = []
    try:
        for label, key in keys.items():
            register(
                key,
                lambda *args, _label=label, **kwargs: calls.append(
                    (_label, args)
                ),
                replace=True,
            )
        assert launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            0x3000,
            0x4000,
            1,
            5_120,
            17_408,
            use_gemv_decode=True,
            registered_decode_variant=keys["c1"].variant,
            q8_1_workspace_ptr=0x5000,
        )
        with native_batch_decode_session(True):
            native_launched = launch_gguf_linear_pair_silu(
                weight_a,
                weight_b,
                0x3000,
                0x4000,
                native_rows,
                5_120,
                17_408,
            )
        assert native_launched is (native_label is not None)
        assert launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            0x3000,
            0x4000,
            512,
            5_120,
            17_408,
        )
        metadata_nbytes = (17_408 // 16) * (5_120 // 256) * 256
        assert launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            0x3000,
            0x4000,
            4_096,
            5_120,
            17_408,
            pair_workspace_ptr=0x6000,
            pair_workspace_nbytes=2 * metadata_nbytes,
        )
    finally:
        for key, fn in originals.items():
            register(key, fn, replace=True)
        clear_gguf_linear_dispatch_cache()

    expected_labels = ["c1"]
    if native_label is not None:
        expected_labels.append(native_label)
    expected_labels.extend(["direct", "expanded"])
    assert [label for label, _args in calls] == expected_labels
    assert calls[0][1][:4] == (0x5000, 0x1000, 0x2000, 0x4000)
    direct_index = 2 if native_label is not None else 1
    assert calls[direct_index][1][:4] == (
        0x3000,
        0x1000,
        0x2000,
        0x4000,
    )
    assert calls[direct_index + 1][1][:6] == (
        0x3000,
        0x1000,
        0x2000,
        0x6000,
        0x6000 + metadata_nbytes,
        0x4000,
    )


def test_q4_t16_dense_c1_pair_silu_uses_canonical_tiles() -> None:
    weight_a = _weight(0x1000, in_features=5_120, out_features=17_408)
    weight_b = _weight(0x2000, in_features=5_120, out_features=17_408)
    key = KernelKey(
        "hip_gfx1100",
        "linear_pair_silu",
        "gguf_q4_k_t16_v1",
        "dense_dual_local32_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls: list[tuple] = []
    try:
        register(key, lambda *args, **kwargs: calls.append(args), replace=True)
        launched = launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            0x3000,
            0x4000,
            1,
            5_120,
            17_408,
            use_gemv_decode=True,
        )
    finally:
        register(key, original, replace=True)

    assert launched
    assert calls[0][:4] == (0x3000, 0x1000, 0x2000, 0x4000)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q4_t16_dense_c1_pair_silu_matches_unfused_chain() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
        build_paro_silu,
        silu_mul_separate_out_bf16,
    )

    runtime = get_hip_runtime()
    in_features = 256
    out_features = 32
    raw_a = make_q4_k_weight(out_features, in_features)
    raw_b = np.roll(raw_a, shift=1, axis=0).copy()
    tiles_a = repack_gguf_q4_k_tile16(raw_a[None, ...]).tiles
    tiles_b = repack_gguf_q4_k_tile16(raw_b[None, ...]).tiles
    rng = np.random.default_rng(0x38D51)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(1, in_features)).astype(np.float32)
    )
    expected_bits = np.zeros((1, out_features), dtype=np.uint16)
    actual_bits = np.zeros_like(expected_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        tiles_a_dev = malloc(tiles_a.nbytes, runtime=runtime)
        tiles_b_dev = malloc(tiles_b.nbytes, runtime=runtime)
        gate_dev = malloc(expected_bits.nbytes, runtime=runtime)
        up_dev = malloc(expected_bits.nbytes, runtime=runtime)
        expected_dev = malloc(expected_bits.nbytes, runtime=runtime)
        actual_dev = malloc(actual_bits.nbytes, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                tiles_a_dev,
                tiles_b_dev,
                gate_dev,
                up_dev,
                expected_dev,
                actual_dev,
            )
        )
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(
            tiles_a_dev, host_array_ptr(tiles_a), runtime=runtime
        )
        copy_host_to_device(
            tiles_b_dev, host_array_ptr(tiles_b), runtime=runtime
        )
        library = build_gguf_t16_selected_gemv(load=True)
        for tiles_dev, out_dev in (
            (tiles_a_dev, gate_dev),
            (tiles_b_dev, up_dev),
        ):
            gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
                x_dev.ptr,
                tiles_dev.ptr,
                out_dev.ptr,
                1,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
        silu_mul_separate_out_bf16(
            gate_dev.ptr,
            up_dev.ptr,
            expected_dev.ptr,
            1,
            out_features,
            library=build_paro_silu(load=True),
            runtime=runtime,
        )
        gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out(
            x_dev.ptr,
            tiles_a_dev.ptr,
            tiles_b_dev.ptr,
            actual_dev.ptr,
            1,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(expected_bits), expected_dev, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(actual_bits), actual_dev, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(actual_bits, expected_bits)
    assert np.isfinite(_bf16_bits_to_f32(actual_bits)).all()


def test_q4_t16_dense_small_row_pair_silu_uses_canonical_tiles() -> None:
    weight_a = _weight(0x1000, in_features=5_120, out_features=17_408)
    weight_b = _weight(0x2000, in_features=5_120, out_features=17_408)
    key = KernelKey(
        "hip_gfx1100",
        "linear_pair_silu",
        "gguf_q4_k_t16_v1",
        "dense_dual_rowtile_bf16_bf16_out",
    )
    original = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
    calls: list[tuple] = []
    try:
        register(key, lambda *args, **kwargs: calls.append(args), replace=True)
        with native_batch_decode_session(True):
            launched = launch_gguf_linear_pair_silu(
                weight_a,
                weight_b,
                0x3000,
                0x4000,
                3,
                5_120,
                17_408,
            )
    finally:
        register(key, original, replace=True)

    assert launched
    assert calls[0][:4] == (0x3000, 0x1000, 0x2000, 0x4000)


@pytest.mark.parametrize(
    ("rows", "expected_variant"),
    (
        (33, "dense_dual_wmma_prefill_row64_bf16_bf16_out"),
        (64, "dense_dual_wmma_prefill_row64_bf16_bf16_out"),
        (65, "dense_dual_wmma_prefill_row128_bf16_bf16_out"),
        (128, "dense_dual_wmma_prefill_row128_bf16_bf16_out"),
        (129, "dense_dual_wmma_prefill_bf16_bf16_out"),
        (256, "dense_dual_wmma_prefill_bf16_bf16_out"),
        (511, "dense_dual_wmma_prefill_bf16_bf16_out"),
        (512, "dense_dual_wmma_prefill_bf16_bf16_out"),
    ),
)
def test_q4_t16_dense_bulk_pair_silu_uses_measured_row_retile(
    rows: int,
    expected_variant: str,
    monkeypatch,
) -> None:
    from hipengine.runtime import gguf_linear as gguf_linear_module

    monkeypatch.delenv("HIPENGINE_GGUF_Q4_T16_DUAL_SILU_RETILE", raising=False)
    monkeypatch.setattr(
        gguf_linear_module,
        "_Q4_T16_DUAL_SILU_RETILE_RESOLVED",
        None,
    )
    weight_a = _weight(0x1000, in_features=5_120, out_features=17_408)
    weight_b = _weight(0x2000, in_features=5_120, out_features=17_408)
    variants = (
        "dense_dual_wmma_prefill_bf16_bf16_out",
        "dense_dual_wmma_prefill_row64_bf16_bf16_out",
        "dense_dual_wmma_prefill_row128_bf16_bf16_out",
    )
    keys = {
        variant: KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_t16_v1",
            variant,
        )
        for variant in variants
    }
    originals = {
        variant: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for variant, key in keys.items()
    }
    calls: list[tuple[str, tuple]] = []
    try:
        for variant, key in keys.items():
            register(
                key,
                lambda *args, _variant=variant, **kwargs: calls.append(
                    (_variant, args)
                ),
                replace=True,
            )
        launched = launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            0x3000,
            0x4000,
            rows,
            5_120,
            17_408,
        )
    finally:
        for variant, key in keys.items():
            register(key, originals[variant], replace=True)
        monkeypatch.setattr(
            gguf_linear_module,
            "_Q4_T16_DUAL_SILU_RETILE_RESOLVED",
            None,
        )
        clear_gguf_linear_dispatch_cache()

    assert launched
    assert calls == [
        (expected_variant, (0x3000, 0x1000, 0x2000, 0x4000, rows, 5_120, 17_408))
    ]


def test_q4_t16_physical_r36_pair_silu_defaults_row48_with_rollback(monkeypatch) -> None:
    from hipengine.runtime import gguf_linear as gguf_linear_module

    monkeypatch.delenv("HIPENGINE_GGUF_Q4_T16_DUAL_SILU_ROW48", raising=False)
    gguf_linear_module._rowtile_variant_policy_env_cache.clear()
    weight_a = _weight(0x1000, in_features=5_120, out_features=17_408)
    weight_b = _weight(0x2000, in_features=5_120, out_features=17_408)
    variants = (
        "dense_dual_wmma_prefill_row48_bf16_bf16_out",
        "dense_dual_wmma_prefill_row64_bf16_bf16_out",
    )
    keys = {
        variant: KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_t16_v1",
            variant,
        )
        for variant in variants
    }
    originals = {
        variant: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for variant, key in keys.items()
    }
    calls: list[str] = []

    def launch() -> None:
        assert launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            0x3000,
            0x4000,
            36,
            5_120,
            17_408,
        )

    try:
        for variant, key in keys.items():
            register(
                key,
                lambda *args, _variant=variant, **kwargs: calls.append(_variant),
                replace=True,
            )
        with q4_t16_physical_extra_rowtiles_session(True):
            launch()
            monkeypatch.setenv("HIPENGINE_GGUF_Q4_T16_DUAL_SILU_ROW48", "0")
            gguf_linear_module._rowtile_variant_policy_env_cache.clear()
            launch()
        launch()
    finally:
        for variant, key in keys.items():
            register(key, originals[variant], replace=True)
        gguf_linear_module._rowtile_variant_policy_env_cache.clear()
        clear_gguf_linear_dispatch_cache()

    assert calls == [
        "dense_dual_wmma_prefill_row48_bf16_bf16_out",
        "dense_dual_wmma_prefill_row64_bf16_bf16_out",
        "dense_dual_wmma_prefill_row64_bf16_bf16_out",
    ]


def test_q4_t16_physical_row48_flag_is_not_resolved_on_row_miss(monkeypatch) -> None:
    from hipengine.runtime import gguf_linear as gguf_linear_module

    monkeypatch.setenv("HIPENGINE_GGUF_Q4_T16_DUAL_SILU_ROW48", "invalid")
    gguf_linear_module._rowtile_variant_policy_env_cache.clear()
    dispatch = gguf_linear_module.GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            "t16_gemv_decode_bf16_bf16_out",
        ),
        "t16",
    )
    with q4_t16_physical_extra_rowtiles_session(True):
        assert (
            gguf_linear_module._q4_t16_physical_dual_silu_variant(
                dispatch.key.backend,
                30,
            )
            is None
        )
        with pytest.raises(ValueError, match="must be a boolean value"):
            gguf_linear_module._q4_t16_physical_dual_silu_variant(
                dispatch.key.backend,
                36,
            )


def test_q4_t16_dense_pair_silu_retile_rollback(monkeypatch) -> None:
    from hipengine.runtime import gguf_linear as gguf_linear_module

    monkeypatch.setenv("HIPENGINE_GGUF_Q4_T16_DUAL_SILU_RETILE", "0")
    monkeypatch.setattr(
        gguf_linear_module,
        "_Q4_T16_DUAL_SILU_RETILE_RESOLVED",
        None,
        raising=False,
    )
    weight_a = _weight(0x1000, in_features=5_120, out_features=17_408)
    weight_b = _weight(0x2000, in_features=5_120, out_features=17_408)
    key = KernelKey(
        "hip_gfx1100",
        "linear_pair_silu",
        "gguf_q4_k_t16_v1",
        "dense_dual_wmma_prefill_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls: list[tuple] = []
    try:
        register(key, lambda *args, **kwargs: calls.append(args), replace=True)
        assert launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            0x3000,
            0x4000,
            35,
            5_120,
            17_408,
        )
    finally:
        register(key, original, replace=True)
        monkeypatch.setattr(
            gguf_linear_module,
            "_Q4_T16_DUAL_SILU_RETILE_RESOLVED",
            None,
            raising=False,
        )
        clear_gguf_linear_dispatch_cache()

    assert len(calls) == 1


@pytest.mark.parametrize("rows", [2, 5, 8, 11, 16, 32])
def test_q4_t16_dense_bulk_pair_silu_keeps_unfused_fallback_below_33(
    rows: int,
) -> None:
    # rows<=8 keep their dedicated small-B rowtile/GEMV owners, and rows 16/32
    # are the captured target-verify group widths, which measured slower fused.
    weight_a = _weight(0x1000, in_features=5_120, out_features=17_408)
    weight_b = _weight(0x2000, in_features=5_120, out_features=17_408)

    assert not launch_gguf_linear_pair_silu(
        weight_a,
        weight_b,
        0x3000,
        0x4000,
        rows,
        5_120,
        17_408,
    )


_UNEQUAL_PAIR_KEY = KernelKey(
    "hip_gfx1100",
    "linear_pair",
    "gguf_q4_k_t16_v1",
    "dense_unequal_dual_wmma_prefill_bf16_bf16_out",
)


def _unequal_pair_launch(rows: int, *, session: bool = True) -> tuple[bool, list]:
    """Launch the 5120 -> 10240/6144 pair under an explicit prefill session."""

    weight_a = _weight(0x1000, in_features=5_120, out_features=10_240)
    weight_b = _weight(0x2000, in_features=5_120, out_features=6_144)
    original = resolve(
        backend=_UNEQUAL_PAIR_KEY.backend,
        layer=_UNEQUAL_PAIR_KEY.layer,
        quant=_UNEQUAL_PAIR_KEY.quant,
        variant=_UNEQUAL_PAIR_KEY.variant,
    )
    calls: list[tuple] = []
    try:
        register(_UNEQUAL_PAIR_KEY, lambda *args, **kwargs: calls.append(args), replace=True)
        # The resident prefill entry opens both sessions; use_wmma defaults off
        # outside it, so the pair route needs the same nesting as production.
        with wmma_prefill_session(True), q4_t16_unequal_pair_prefill_session(session):
            launched = launch_gguf_linear_pair(
                weight_a,
                weight_b,
                0x3000,
                0x4000,
                0x5000,
                rows,
                5_120,
                10_240,
                out_features_b=6_144,
            )
    finally:
        register(_UNEQUAL_PAIR_KEY, original, replace=True)
    return launched, calls


@pytest.mark.parametrize("rows", [16, 24, 32, 45, 96, 512])
def test_q4_t16_dense_unequal_pair_prefill_uses_dual_owner_from_16(rows: int) -> None:
    launched, calls = _unequal_pair_launch(rows)

    assert launched
    assert len(calls) == 1
    # (x, tiles_a, tiles_b, out_a, out_b, rows, in, out_a, out_b)
    assert (calls[0][0], calls[0][3], calls[0][4]) == (0x3000, 0x4000, 0x5000)
    assert calls[0][5:] == (rows, 5_120, 10_240, 6_144)


@pytest.mark.parametrize("rows", [2, 8, 15])
def test_q4_t16_dense_unequal_pair_prefill_keeps_singletons_below_16(
    rows: int,
) -> None:
    # rows<=8 keep their dedicated GEMV/rowtile decode owners; 15 is the last row
    # count under the qualified dual floor.
    launched, calls = _unequal_pair_launch(rows)

    assert not launched
    assert calls == []


def test_q4_t16_dense_unequal_pair_prefill_requires_prefill_session() -> None:
    """The dual is prefill-scoped, which is what keeps target verification off it.

    ``q4_t16_unequal_pair_prefill_session`` is opened by the resident prefill entry
    only, so a row-count floor alone does not have to protect the captured verify
    groups the way the shared ``linear_pair_silu`` gate does.
    """

    launched, calls = _unequal_pair_launch(512, session=False)

    assert not launched
    assert calls == []


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q4_t16_r6_rowtile_matches_two_retained_c1_r3_owners() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rows = 6
    c1_rows = 3
    in_features = 256
    out_features = 96
    raw = make_q4_k_weight(out_features, in_features)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    rng = np.random.default_rng(0xC1C206)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    r6_bits = np.zeros((rows, out_features), dtype=np.uint16)
    split_bits = np.zeros_like(r6_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        tiles_dev = malloc(tiles.nbytes, runtime=runtime)
        r6_dev = malloc(r6_bits.nbytes, runtime=runtime)
        split_dev = malloc(split_bits.nbytes, runtime=runtime)
        buffers.extend((x_dev, tiles_dev, r6_dev, split_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(tiles_dev, host_array_ptr(tiles), runtime=runtime)
        library = build_gguf_t16_selected_gemv(load=True)
        gguf_q4_k_t16_dense_rowtile_bf16_bf16_out(
            x_dev.ptr,
            tiles_dev.ptr,
            r6_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        for row_start in (0, c1_rows):
            gguf_q4_k_t16_dense_rowtile_bf16_bf16_out(
                x_dev.ptr + row_start * in_features * 2,
                tiles_dev.ptr,
                split_dev.ptr + row_start * out_features * 2,
                c1_rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(r6_bits), r6_dev, runtime=runtime)
        copy_device_to_host(
            host_array_ptr(split_bits), split_dev, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(r6_bits, split_bits)
    assert np.isfinite(_bf16_bits_to_f32(r6_bits)).all()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [6, 8, 12, 16])
def test_q4_t16_smallm_wmma_matches_current_wmma_and_cpu(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    candidate = getattr(
        t16_prefill,
        "gguf_q4_k_t16_wmma_prefill_smallm_bf16_bf16_out",
        None,
    )
    assert callable(candidate)
    runtime = get_hip_runtime()
    in_features = 256
    out_features = 96
    raw = make_q4_k_weight(out_features, in_features)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    rng = np.random.default_rng(0x115100 + rows)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    candidate_bits = np.zeros((rows, out_features), dtype=np.uint16)
    single_bits = np.zeros_like(candidate_bits)
    shared_bits = np.zeros_like(candidate_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        tiles_dev = malloc(tiles.nbytes, runtime=runtime)
        candidate_dev = malloc(candidate_bits.nbytes, runtime=runtime)
        single_dev = malloc(single_bits.nbytes, runtime=runtime)
        shared_dev = malloc(shared_bits.nbytes, runtime=runtime)
        buffers.extend(
            (x_dev, tiles_dev, candidate_dev, single_dev, shared_dev)
        )
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(tiles_dev, host_array_ptr(tiles), runtime=runtime)
        library = build_gguf_k_t16_selected_prefill(load=True)
        for fn, out_dev in (
            (candidate, candidate_dev),
            (gguf_q4_k_t16_wmma_prefill_bf16_bf16_out, single_dev),
            (gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out, shared_dev),
        ):
            fn(
                x_dev.ptr,
                tiles_dev.ptr,
                out_dev.ptr,
                rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        for host, device in (
            (candidate_bits, candidate_dev),
            (single_bits, single_dev),
            (shared_bits, shared_dev),
        ):
            copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(candidate_bits, single_bits)
    np.testing.assert_array_equal(candidate_bits, shared_bits)
    actual = _bf16_bits_to_f32(candidate_bits)
    expected = _bf16_bits_to_f32(
        _f32_to_bf16_bits(
            gguf_quant_gemv(
                _bf16_bits_to_f32(x_bits),
                raw,
                GGMLQuantizationType.Q4_K,
            )
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=0.012, atol=0.5)
    assert np.isfinite(actual).all()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [6, 8, 12, 16, 33, 512, 1_024, 4_096])
def test_q4_t16_dense_wmma_prefill_matches_cpu_reference(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    in_features = 256
    out_features = 32
    raw = make_q4_k_weight(out_features, in_features)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    rng = np.random.default_rng(36 + rows)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    out_bits = np.zeros((rows, out_features), dtype=np.uint16)
    baseline_bits = np.zeros_like(out_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        tiles_dev = malloc(tiles.nbytes, runtime=runtime)
        out_dev = malloc(out_bits.nbytes, runtime=runtime)
        baseline_dev = malloc(baseline_bits.nbytes, runtime=runtime)
        buffers.extend((x_dev, tiles_dev, out_dev, baseline_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(tiles_dev, host_array_ptr(tiles), runtime=runtime)
        library = build_gguf_k_t16_selected_prefill(load=True)
        gguf_q4_k_t16_wmma_prefill_bf16_bf16_out(
            x_dev.ptr,
            tiles_dev.ptr,
            baseline_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
            x_dev.ptr,
            tiles_dev.ptr,
            out_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_bits), out_dev, runtime=runtime)
        copy_device_to_host(
            host_array_ptr(baseline_bits), baseline_dev, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(out_bits, baseline_bits)
    x_f32 = _bf16_bits_to_f32(x_bits)
    actual = _bf16_bits_to_f32(out_bits)
    expected = _bf16_bits_to_f32(
        _f32_to_bf16_bits(
            gguf_quant_gemv(x_f32, raw, GGMLQuantizationType.Q4_K)
        )
    )
    # Match the established Q4_K WMMA gate: independent FP16 fragments may
    # differ near zero while BF16-scale outputs remain bounded.
    np.testing.assert_allclose(actual, expected, rtol=0.012, atol=0.5)
    assert np.isfinite(actual).all()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [35, 48, 67, 192])
def test_q4_t16_shared_b_row64_matches_shared_b_parent(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    candidate = getattr(
        t16_prefill,
        "gguf_q4_k_t16_wmma_prefill_shared_b_row64_bf16_bf16_out",
        None,
    )
    assert callable(candidate)
    runtime = get_hip_runtime()
    in_features = 256
    out_features = 96
    raw = make_q4_k_weight(out_features, in_features)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    rng = np.random.default_rng(0x640000 + rows)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    parent_bits = np.zeros((rows, out_features), dtype=np.uint16)
    candidate_bits = np.zeros_like(parent_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        tiles_dev = malloc(tiles.nbytes, runtime=runtime)
        parent_dev = malloc(parent_bits.nbytes, runtime=runtime)
        candidate_dev = malloc(candidate_bits.nbytes, runtime=runtime)
        buffers.extend((x_dev, tiles_dev, parent_dev, candidate_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(tiles_dev, host_array_ptr(tiles), runtime=runtime)
        library = build_gguf_k_t16_selected_prefill(load=True)
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
            x_dev.ptr,
            tiles_dev.ptr,
            parent_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        candidate(
            x_dev.ptr,
            tiles_dev.ptr,
            candidate_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(parent_bits), parent_dev, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(candidate_bits), candidate_dev, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(candidate_bits, parent_bits)
    actual = _bf16_bits_to_f32(candidate_bits)
    expected = _bf16_bits_to_f32(
        _f32_to_bf16_bits(
            gguf_quant_gemv(
                _bf16_bits_to_f32(x_bits),
                raw,
                GGMLQuantizationType.Q4_K,
            )
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=0.012, atol=0.5)
    assert np.isfinite(actual).all()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [512, 513, 1_024])
def test_q4_t16_dense_unequal_dual_wmma_matches_singletons(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    wrapper = getattr(
        t16_prefill,
        "gguf_q4_k_t16_dense_unequal_dual_wmma_prefill_bf16_bf16_out",
        None,
    )
    assert callable(wrapper)
    runtime = get_hip_runtime()
    in_features = 256
    out_features_a = 96
    out_features_b = 64
    raw_a = make_q4_k_weight(out_features_a, in_features)
    raw_b = np.roll(
        make_q4_k_weight(out_features_b, in_features), shift=1, axis=0
    ).copy()
    tiles_a = repack_gguf_q4_k_tile16(raw_a[None, ...]).tiles
    tiles_b = repack_gguf_q4_k_tile16(raw_b[None, ...]).tiles
    rng = np.random.default_rng(0x36D00 + rows)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    expected_a = np.zeros((rows, out_features_a), dtype=np.uint16)
    expected_b = np.zeros((rows, out_features_b), dtype=np.uint16)
    actual_a = np.zeros_like(expected_a)
    actual_b = np.zeros_like(expected_b)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        tiles_a_dev = malloc(tiles_a.nbytes, runtime=runtime)
        tiles_b_dev = malloc(tiles_b.nbytes, runtime=runtime)
        expected_a_dev = malloc(expected_a.nbytes, runtime=runtime)
        expected_b_dev = malloc(expected_b.nbytes, runtime=runtime)
        actual_a_dev = malloc(actual_a.nbytes, runtime=runtime)
        actual_b_dev = malloc(actual_b.nbytes, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                tiles_a_dev,
                tiles_b_dev,
                expected_a_dev,
                expected_b_dev,
                actual_a_dev,
                actual_b_dev,
            )
        )
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(
            tiles_a_dev, host_array_ptr(tiles_a), runtime=runtime
        )
        copy_host_to_device(
            tiles_b_dev, host_array_ptr(tiles_b), runtime=runtime
        )
        library = build_gguf_k_t16_selected_prefill(load=True)
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
            x_dev.ptr,
            tiles_a_dev.ptr,
            expected_a_dev.ptr,
            rows,
            in_features,
            out_features_a,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
            x_dev.ptr,
            tiles_b_dev.ptr,
            expected_b_dev.ptr,
            rows,
            in_features,
            out_features_b,
            library=library,
            runtime=runtime,
        )
        wrapper(
            x_dev.ptr,
            tiles_a_dev.ptr,
            tiles_b_dev.ptr,
            actual_a_dev.ptr,
            actual_b_dev.ptr,
            rows,
            in_features,
            out_features_a,
            out_features_b,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        for host, device in (
            (expected_a, expected_a_dev),
            (expected_b, expected_b_dev),
            (actual_a, actual_a_dev),
            (actual_b, actual_b_dev),
        ):
            copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [16, 24, 32, 45, 96, 512])
def test_q4_t16_dense_unequal_dual_wmma_matches_singletons_at_production_shape(
    rows: int,
) -> None:
    """Unequal dual must equal the two singletons at the only shape it dispatches.

    ``_Q4_T16_UNEQUAL_DUAL_WMMA_SHAPE`` is (5120 -> 10240, 6144); the fixture
    geometry above is 256 -> 96, 64, so it cannot qualify a row-gate change on its
    own - tile counts and the shared-x plan differ per N.
    """

    from hipengine.core.hip import get_hip_runtime

    wrapper = getattr(
        t16_prefill,
        "gguf_q4_k_t16_dense_unequal_dual_wmma_prefill_bf16_bf16_out",
        None,
    )
    assert callable(wrapper)
    runtime = get_hip_runtime()
    in_features = 5_120
    out_features_a = 10_240
    out_features_b = 6_144
    raw_a = make_q4_k_weight(out_features_a, in_features)
    raw_b = np.roll(make_q4_k_weight(out_features_b, in_features), shift=1, axis=0).copy()
    tiles_a = repack_gguf_q4_k_tile16(raw_a[None, ...]).tiles
    tiles_b = repack_gguf_q4_k_tile16(raw_b[None, ...]).tiles
    rng = np.random.default_rng(0x7A1E0 + rows)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    expected_a = np.zeros((rows, out_features_a), dtype=np.uint16)
    expected_b = np.zeros((rows, out_features_b), dtype=np.uint16)
    actual_a = np.zeros_like(expected_a)
    actual_b = np.zeros_like(expected_b)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        tiles_a_dev = malloc(tiles_a.nbytes, runtime=runtime)
        tiles_b_dev = malloc(tiles_b.nbytes, runtime=runtime)
        expected_a_dev = malloc(expected_a.nbytes, runtime=runtime)
        expected_b_dev = malloc(expected_b.nbytes, runtime=runtime)
        actual_a_dev = malloc(actual_a.nbytes, runtime=runtime)
        actual_b_dev = malloc(actual_b.nbytes, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                tiles_a_dev,
                tiles_b_dev,
                expected_a_dev,
                expected_b_dev,
                actual_a_dev,
                actual_b_dev,
            )
        )
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(tiles_a_dev, host_array_ptr(tiles_a), runtime=runtime)
        copy_host_to_device(tiles_b_dev, host_array_ptr(tiles_b), runtime=runtime)
        library = build_gguf_k_t16_selected_prefill(load=True)
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
            x_dev.ptr,
            tiles_a_dev.ptr,
            expected_a_dev.ptr,
            rows,
            in_features,
            out_features_a,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
            x_dev.ptr,
            tiles_b_dev.ptr,
            expected_b_dev.ptr,
            rows,
            in_features,
            out_features_b,
            library=library,
            runtime=runtime,
        )
        wrapper(
            x_dev.ptr,
            tiles_a_dev.ptr,
            tiles_b_dev.ptr,
            actual_a_dev.ptr,
            actual_b_dev.ptr,
            rows,
            in_features,
            out_features_a,
            out_features_b,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        for host, device in (
            (expected_a, expected_a_dev),
            (expected_b, expected_b_dev),
            (actual_a, actual_a_dev),
            (actual_b, actual_b_dev),
        ):
            copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [2, 12, 45, 48, 128, 256, 511, 512, 513, 1_024])
def test_q4_t16_dense_dual_wmma_silu_matches_unfused_chain(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
        build_paro_silu,
        silu_mul_separate_out_bf16,
    )

    runtime = get_hip_runtime()
    in_features = 256
    out_features = 32
    raw_a = make_q4_k_weight(out_features, in_features)
    raw_b = np.roll(raw_a, shift=1, axis=0).copy()
    tiles_a = repack_gguf_q4_k_tile16(raw_a[None, ...]).tiles
    tiles_b = repack_gguf_q4_k_tile16(raw_b[None, ...]).tiles
    rng = np.random.default_rng(3600 + rows)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    expected_bits = np.zeros((rows, out_features), dtype=np.uint16)
    actual_bits = np.zeros_like(expected_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        tiles_a_dev = malloc(tiles_a.nbytes, runtime=runtime)
        tiles_b_dev = malloc(tiles_b.nbytes, runtime=runtime)
        gate_dev = malloc(expected_bits.nbytes, runtime=runtime)
        up_dev = malloc(expected_bits.nbytes, runtime=runtime)
        expected_dev = malloc(expected_bits.nbytes, runtime=runtime)
        actual_dev = malloc(actual_bits.nbytes, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                tiles_a_dev,
                tiles_b_dev,
                gate_dev,
                up_dev,
                expected_dev,
                actual_dev,
            )
        )
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(
            tiles_a_dev, host_array_ptr(tiles_a), runtime=runtime
        )
        copy_host_to_device(
            tiles_b_dev, host_array_ptr(tiles_b), runtime=runtime
        )
        library = build_gguf_k_t16_selected_prefill(load=True)
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
            x_dev.ptr,
            tiles_a_dev.ptr,
            gate_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
            x_dev.ptr,
            tiles_b_dev.ptr,
            up_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        silu_mul_separate_out_bf16(
            gate_dev.ptr,
            up_dev.ptr,
            expected_dev.ptr,
            rows,
            out_features,
            library=build_paro_silu(load=True),
            runtime=runtime,
        )
        gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out(
            x_dev.ptr,
            tiles_a_dev.ptr,
            tiles_b_dev.ptr,
            actual_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(expected_bits), expected_dev, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(actual_bits), actual_dev, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(actual_bits, expected_bits)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("rows", "candidate"),
    [
        (45, gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out),
        (96, gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out),
        (192, gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out),
        (511, gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out),
        (512, gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out),
        (36, gguf_q4_k_t16_dense_dual_wmma_prefill_row48_silu_bf16_bf16_out),
        (64, gguf_q4_k_t16_dense_dual_wmma_prefill_row64_silu_bf16_bf16_out),
        (67, gguf_q4_k_t16_dense_dual_wmma_prefill_row128_silu_bf16_bf16_out),
    ],
)
def test_q4_t16_dense_dual_wmma_silu_matches_unfused_chain_at_production_shape(
    rows: int,
    candidate,
) -> None:
    """Fused dual+SiLU prefill must equal the unfused chain at the dispatched shape.

    ``_q4_t16_dual_wmma_silu_dispatch`` only admits (5120 -> 17408), so fixture
    parity at tiny shapes cannot qualify the row gate by itself: tile counts and
    the shared-x epilogue are shape dependent.
    """

    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
        build_paro_silu,
        silu_mul_separate_out_bf16,
    )

    runtime = get_hip_runtime()
    in_features = 5_120
    out_features = 17_408
    raw_a = make_q4_k_weight(out_features, in_features)
    raw_b = np.roll(raw_a, shift=1, axis=0).copy()
    tiles_a = repack_gguf_q4_k_tile16(raw_a[None, ...]).tiles
    tiles_b = repack_gguf_q4_k_tile16(raw_b[None, ...]).tiles
    rng = np.random.default_rng(4900 + rows)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    expected_bits = np.zeros((rows, out_features), dtype=np.uint16)
    actual_bits = np.zeros_like(expected_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        tiles_a_dev = malloc(tiles_a.nbytes, runtime=runtime)
        tiles_b_dev = malloc(tiles_b.nbytes, runtime=runtime)
        gate_dev = malloc(expected_bits.nbytes, runtime=runtime)
        up_dev = malloc(expected_bits.nbytes, runtime=runtime)
        expected_dev = malloc(expected_bits.nbytes, runtime=runtime)
        actual_dev = malloc(actual_bits.nbytes, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                tiles_a_dev,
                tiles_b_dev,
                gate_dev,
                up_dev,
                expected_dev,
                actual_dev,
            )
        )
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(tiles_a_dev, host_array_ptr(tiles_a), runtime=runtime)
        copy_host_to_device(tiles_b_dev, host_array_ptr(tiles_b), runtime=runtime)
        library = build_gguf_k_t16_selected_prefill(load=True)
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
            x_dev.ptr,
            tiles_a_dev.ptr,
            gate_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
            x_dev.ptr,
            tiles_b_dev.ptr,
            up_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        silu_mul_separate_out_bf16(
            gate_dev.ptr,
            up_dev.ptr,
            expected_dev.ptr,
            rows,
            out_features,
            library=build_paro_silu(load=True),
            runtime=runtime,
        )
        candidate(
            x_dev.ptr,
            tiles_a_dev.ptr,
            tiles_b_dev.ptr,
            actual_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(expected_bits), expected_dev, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(actual_bits), actual_dev, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(actual_bits, expected_bits)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [16, 33, 512, 513])
@pytest.mark.parametrize("expanded_metadata", [False, True])
def test_q4_qmicro_dense_dual_wmma_silu_matches_t16_bits(
    rows: int,
    expanded_metadata: bool,
) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
        build_paro_silu,
        silu_mul_separate_out_bf16,
    )

    runtime = get_hip_runtime()
    in_features = 256
    out_features = 32
    raw_a = make_q4_k_weight(out_features, in_features)
    raw_b = np.roll(raw_a, shift=3, axis=0).copy()
    control_tiles_a = repack_gguf_q4_k_tile16(raw_a[None, ...]).tiles
    control_tiles_b = repack_gguf_q4_k_tile16(raw_b[None, ...]).tiles
    candidate_tiles_a = repack_gguf_q4_k_tile16_qmicro(raw_a[None, ...]).tiles
    candidate_tiles_b = repack_gguf_q4_k_tile16_qmicro(raw_b[None, ...]).tiles
    rng = np.random.default_rng(3800 + rows)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    control_bits = np.zeros((rows, out_features), dtype=np.uint16)
    candidate_bits = np.zeros_like(control_bits)
    fallback_bits = np.zeros_like(control_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        control_a_dev = malloc(control_tiles_a.nbytes, runtime=runtime)
        control_b_dev = malloc(control_tiles_b.nbytes, runtime=runtime)
        candidate_a_dev = malloc(candidate_tiles_a.nbytes, runtime=runtime)
        candidate_b_dev = malloc(candidate_tiles_b.nbytes, runtime=runtime)
        control_dev = malloc(control_bits.nbytes, runtime=runtime)
        candidate_dev = malloc(candidate_bits.nbytes, runtime=runtime)
        fallback_gate_dev = malloc(fallback_bits.nbytes, runtime=runtime)
        fallback_up_dev = malloc(fallback_bits.nbytes, runtime=runtime)
        fallback_dev = malloc(fallback_bits.nbytes, runtime=runtime)
        metadata_nbytes = (out_features // 16) * (in_features // 256) * 256
        metadata_a_dev = malloc(metadata_nbytes, runtime=runtime)
        metadata_b_dev = malloc(metadata_nbytes, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                control_a_dev,
                control_b_dev,
                candidate_a_dev,
                candidate_b_dev,
                control_dev,
                candidate_dev,
                fallback_gate_dev,
                fallback_up_dev,
                fallback_dev,
                metadata_a_dev,
                metadata_b_dev,
            )
        )
        for device, host in (
            (x_dev, x_bits),
            (control_a_dev, control_tiles_a),
            (control_b_dev, control_tiles_b),
            (candidate_a_dev, candidate_tiles_a),
            (candidate_b_dev, candidate_tiles_b),
        ):
            copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
        library = build_gguf_k_t16_selected_prefill(load=True)
        gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out(
            x_dev.ptr,
            control_a_dev.ptr,
            control_b_dev.ptr,
            control_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_qmicro_t16_wmma_prefill_bf16_bf16_out(
            x_dev.ptr,
            candidate_a_dev.ptr,
            fallback_gate_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_qmicro_t16_wmma_prefill_bf16_bf16_out(
            x_dev.ptr,
            candidate_b_dev.ptr,
            fallback_up_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        silu_mul_separate_out_bf16(
            fallback_gate_dev.ptr,
            fallback_up_dev.ptr,
            fallback_dev.ptr,
            rows,
            out_features,
            library=build_paro_silu(load=True),
            runtime=runtime,
        )
        if expanded_metadata:
            gguf_q4_k_qmicro_t16_dense_dual_wmma_prefill_expanded_meta_silu_bf16_bf16_out(
                x_dev.ptr,
                candidate_a_dev.ptr,
                candidate_b_dev.ptr,
                metadata_a_dev.ptr,
                metadata_b_dev.ptr,
                candidate_dev.ptr,
                rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
        else:
            gguf_q4_k_qmicro_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out(
                x_dev.ptr,
                candidate_a_dev.ptr,
                candidate_b_dev.ptr,
                candidate_dev.ptr,
                rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(control_bits), control_dev, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(candidate_bits), candidate_dev, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(fallback_bits), fallback_dev, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(candidate_bits, control_bits)
    np.testing.assert_array_equal(candidate_bits, fallback_bits)
