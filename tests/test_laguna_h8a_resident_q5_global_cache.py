"""WPF-H8A exact resident global-Q5 F32 cache ownership RED."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.hip_gfx1100.quant import (
    gguf_q5_k_f32_rocblas_prefill as q5_f32,
)
from hipengine.kernels.registry import KernelKey, is_registered, register, resolve
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION
from hipengine.loading.laguna_gguf_materialize import LAYOUT_RAW_GGUF
from hipengine.runtime import gguf_linear
from hipengine.runtime import laguna_gguf_runner as runner_module
from tests.test_gguf_q5_k_f32_rocblas_prefill import (
    _bf16_bits,
    _bf16_to_f32,
    _device,
)
from tests.test_laguna_h5y_q5_activation_tile_k_row import (
    _activation_plane_shape,
    _expected_activation_plane,
    _hip_available,
    _suffix,
)
from tests.test_laguna_h7g_q5_padded_compute import (
    _production_q5_weight,
    _sampled_cpu_gate,
)

_ROOT = Path(__file__).resolve().parents[1]
_TARGET_ARTIFACT = _ROOT / (
    "benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h7y-"
    "resident-q5-global-f32-cache-target.json"
)
_TARGET_ARTIFACT_SHA256 = (
    "2cdb146629e10a2ae5992d8010dc5aa80d99a513f6d1a21209be4a236e84778f"
)
_HIP_SOURCE_SHA256 = (
    "1a06011ea6e7bda8e0b48fd357cbcbadaff76793a1b5c49bd217cc83d32b7110"
)
_RETAINED_PRODUCER_WRAPPER_SHA256 = (
    "62fc1fadbcf325b841c3d08995c42c973d3e9f4538bf61363f2595024588e3e2"
)
_RETAINED_H7G_PRIMITIVE_WRAPPER_SHA256 = (
    "b385d1353a85783cb49c60928bf8762b57d0d17e42702315eb80b61b1400f99c"
)
_RETAINED_H7G_COMPOSITE_WRAPPER_SHA256 = (
    "be6748b74afa3c2fe519dd2389a2a83459ce3f900071b97f13c7f9f678c71875"
)

_LAYERS = tuple(range(0, 48, 4))
# Slot, output dtype, K, N, source shape.
_ROLES = (
    ("attn_q", "f32", 3_072, 6_144, (6_144, 3_072)),
    ("attn_output", "bf16", 6_144, 3_072, (3_072, 6_144)),
)
_TARGETS = tuple(
    (layer_id, slot, output_dtype, in_features, out_features, source_shape)
    for layer_id in _LAYERS
    for slot, output_dtype, in_features, out_features, source_shape in _ROLES
)
_ROWS = (1, 7, 8, 9, 512)
_COL_TILE = 16
_ROW_BATCH = 5
_WEIGHT_LAYOUT = "tile_k_col"
_PLANE_BYTES = 3_072 * 6_144 * np.dtype(np.float32).itemsize
_PLANE_COUNT = 24
_RESIDENT_BYTES = _PLANE_COUNT * _PLANE_BYTES

_SUPPORTED_CAPABILITY = "LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE_SUPPORTED"
_SOURCE_CAPABILITY = "LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE"
_OWNER_CLASS = "LagunaQ5F32ResidentGlobalCache"
_RESOLVER = "resolve_laguna_q5_f32_resident_global_cache"
_SESSION_CACHE_PARAMETER = "resident_q5_f32_cache"
_SESSION_ENABLE_PARAMETER = "use_q5_f32_resident_global_cache"
_PLANE_CLASS = "Q5F32ResidentPlane"
_SESSION_PLANE_FIELD = "resident_weight_f32_planes"

_EXPECTED_TOPOLOGY = {
    "setup_coltile16_producers": 24,
    "request_coltile16_producers": 0,
    "request_activation_packs": 24,
    "request_h7g_consumers": 24,
    "request_application_dispatches": 2_262,
    "queues": 1,
    "streams": 1,
    "compiler_processes_allowed": 0,
}
_RUNTIME_ADMISSION = {
    "fixed_warmups": 1,
    "fixed_samples": 5,
    "lengths": (512, 1_024, 4_096),
    "paired_samples_per_length": 3,
    "complete_m512_state_byte_exact": True,
    "fixed_median_must_win": True,
    "every_length_median_must_win": True,
    "setup_excluded_from_request_timing": True,
    "source_promotion_separately_red_gated": True,
    "no_subset_or_favorable_rerun": True,
}
_NO_SALVAGE = (
    "attn_q-only",
    "attn_output-only",
    "layer",
    "prompt",
    "token",
    "length",
    "layout",
    "sidecar-size",
    "threshold",
    "favorable-rerun",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_source(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _producer_name(output_dtype: str) -> str:
    return (
        "gguf_q5_k_dequantize_f32_exact_tile_k_col_"
        + _suffix(_COL_TILE, _ROW_BATCH, output_dtype)
    )


def _producer_variant(output_dtype: str) -> str:
    return "raw_f32_exact_tile_k_col_" + _suffix(
        _COL_TILE,
        _ROW_BATCH,
        output_dtype,
    )


def _h7g_primitive_name(output_dtype: str) -> str:
    return (
        "gguf_q5_k_f32_weight_ordered_weight_major_tile_k_col_"
        "activation_tile_k_row_padded_compute_"
        + _suffix(_COL_TILE, _ROW_BATCH, output_dtype)
    )


def _h7g_composite_name(output_dtype: str) -> str:
    return (
        "gguf_q5_k_f32_ordered_weight_major_tile_k_col_"
        "activation_tile_k_row_padded_compute_"
        + _suffix(_COL_TILE, _ROW_BATCH, output_dtype)
    )


def _h7g_composite_variant(output_dtype: str) -> str:
    return (
        "f32_ordered_weight_major_tile_k_col_activation_tile_k_row_"
        "padded_compute_"
        + _suffix(_COL_TILE, _ROW_BATCH, output_dtype)
    )


def _candidate_name(output_dtype: str) -> str:
    return (
        "gguf_q5_k_f32_resident_ordered_weight_major_tile_k_col_"
        "activation_tile_k_row_padded_compute_"
        + _suffix(_COL_TILE, _ROW_BATCH, output_dtype)
    )


def _candidate_variant(output_dtype: str) -> str:
    return (
        "f32_resident_ordered_weight_major_tile_k_col_activation_tile_k_row_"
        "padded_compute_"
        + _suffix(_COL_TILE, _ROW_BATCH, output_dtype)
    )


def _candidate_key(output_dtype: str, *, backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(
        backend,
        "linear",
        "gguf_q5_k",
        _candidate_variant(output_dtype),
    )


def _retained_keys(output_dtype: str) -> tuple[KernelKey, KernelKey]:
    return (
        KernelKey(
            "hip_gfx1100",
            "dequant",
            "gguf_q5_k",
            _producer_variant(output_dtype),
        ),
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k",
            _h7g_composite_variant(output_dtype),
        ),
    )


def _require_cached_build() -> bool:
    return os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class _FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x4000_0000
        self.allocations: dict[int, int] = {}
        self.malloc_order: list[int] = []
        self.free_order: list[int] = []
        self.synchronizations = 0

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += int(nbytes) + 0x1000
        self.allocations[ptr] = int(nbytes)
        self.malloc_order.append(ptr)
        return ptr

    def free(self, ptr: int) -> None:
        parsed = int(ptr)
        assert parsed in self.allocations
        self.allocations.pop(parsed)
        self.free_order.append(parsed)

    def device_synchronize(self) -> None:
        self.synchronizations += 1


class _FakeWeight:
    def __init__(
        self,
        *,
        raw_ptr: int,
        source_shape: tuple[int, int],
        backend: str = "hip_gfx1100",
    ) -> None:
        self.backend = backend
        self.spec = SimpleNamespace(
            quant_key="gguf_q5_k",
            layout=LAYOUT_RAW_GGUF,
            source=SimpleNamespace(shape=source_shape),
        )
        self._raw = SimpleNamespace(
            tensor=SimpleNamespace(ptr=int(raw_ptr)),
            buffer=DeviceBuffer(int(raw_ptr), 1),
        )

    def allocation(self, name: str | None = None):
        assert name in (None, "raw")
        return self._raw


class _FakeLayer:
    def __init__(self, layer_id: int) -> None:
        self.layer_id = int(layer_id)
        self.attention_type = (
            FULL_ATTENTION if self.layer_id in _LAYERS else SLIDING_ATTENTION
        )
        self.weights: dict[str, _FakeWeight] = {}
        if self.attention_type == FULL_ATTENTION:
            for role_index, (slot, _dtype, _in, _out, shape) in enumerate(_ROLES):
                self.weights[slot] = _FakeWeight(
                    raw_ptr=0x1000_0000 + self.layer_id * 0x10000 + role_index * 0x1000,
                    source_shape=shape,
                )

    def weight(self, slot: str) -> _FakeWeight:
        return self.weights[slot]


class _FakeWeights:
    def __init__(self) -> None:
        self.backend = "hip_gfx1100"
        self.layers = tuple(_FakeLayer(layer_id) for layer_id in range(48))

    def layer(self, layer_id: int) -> _FakeLayer:
        return self.layers[int(layer_id)]


def _install_fake_producers(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, Any]],
    *,
    fail_at: int | None = None,
) -> None:
    def fake_resolve(
        *,
        backend: str,
        layer: str,
        quant: str,
        variant: str,
        missing: str = "raise",
    ):
        del missing
        assert backend == "hip_gfx1100"
        assert layer == "dequant"
        assert quant == "gguf_q5_k"
        assert variant in {_producer_variant("bf16"), _producer_variant("f32")}

        def producer(
            qweight_ptr: int,
            out_ptr: int,
            in_features: int,
            out_features: int,
            **kwargs: Any,
        ) -> None:
            calls.append(
                {
                    "variant": variant,
                    "qweight_ptr": int(qweight_ptr),
                    "out_ptr": int(out_ptr),
                    "in_features": int(in_features),
                    "out_features": int(out_features),
                    "kwargs": dict(kwargs),
                }
            )
            if fail_at is not None and len(calls) == fail_at:
                raise RuntimeError("injected H8A producer failure")

        return producer

    monkeypatch.setattr(runner_module, "resolve", fake_resolve)


def test_h8a_frozen_target_current_sources_topology_and_admission_contract() -> None:
    from hipengine.kernels import hip_gfx1100

    artifact = json.loads(_TARGET_ARTIFACT.read_text())
    assert _sha256(_TARGET_ARTIFACT) == _TARGET_ARTIFACT_SHA256
    assert artifact["target"]["id"] == "WPF-H8A"
    assert artifact["status"] == "accepted_target_only_no_candidate_implementation"
    assert artifact["target"]["layers"] == list(_LAYERS)
    assert artifact["target"]["tensor_count"] == _PLANE_COUNT
    assert artifact["target"]["resident_allocations"] == _PLANE_COUNT
    assert artifact["target"]["f32_plane_bytes_each"] == _PLANE_BYTES
    assert artifact["target"]["resident_f32_bytes"] == _RESIDENT_BYTES
    assert artifact["target"]["resident_f32_gib"] == 1.6875
    assert artifact["target"]["hip_source_change_allowed"] is False
    assert artifact["target"]["new_device_body"] is False
    assert artifact["target"]["new_jit_object"] is False
    assert artifact["target"]["request_dispatches_before"] == 2_286
    assert artifact["target"]["request_dispatches_after_model"] == 2_262
    assert artifact["target"]["setup_producer_dispatches"] == 24
    assert artifact["target"]["request_producer_dispatches_after"] == 0
    assert artifact["decision"]["candidate_implemented"] is False
    assert artifact["decision"]["speed_result_exists"] is False
    assert artifact["memory_feasibility"]["audit_status"] == "pass"
    assert artifact["memory_feasibility"]["exact_m512_token"] == 2930
    assert artifact["memory_feasibility"]["after_exact_m512_free_bytes"] > _RESIDENT_BYTES

    assert len(_TARGETS) == _PLANE_COUNT
    assert {target[0] for target in _TARGETS} == set(_LAYERS)
    assert sum(1 for target in _TARGETS if target[1] == "attn_q") == 12
    assert sum(1 for target in _TARGETS if target[1] == "attn_output") == 12
    assert _PLANE_BYTES == 75_497_472
    assert _RESIDENT_BYTES == 1_811_939_328
    assert _EXPECTED_TOPOLOGY == {
        "setup_coltile16_producers": 24,
        "request_coltile16_producers": 0,
        "request_activation_packs": 24,
        "request_h7g_consumers": 24,
        "request_application_dispatches": 2_262,
        "queues": 1,
        "streams": 1,
        "compiler_processes_allowed": 0,
    }
    assert _RUNTIME_ADMISSION["lengths"] == (512, 1_024, 4_096)
    assert _RUNTIME_ADMISSION["complete_m512_state_byte_exact"] is True
    assert _RUNTIME_ADMISSION["setup_excluded_from_request_timing"] is True
    assert _RUNTIME_ADMISSION["source_promotion_separately_red_gated"] is True
    assert _RUNTIME_ADMISSION["no_subset_or_favorable_rerun"] is True
    assert _NO_SALVAGE == (
        "attn_q-only",
        "attn_output-only",
        "layer",
        "prompt",
        "token",
        "length",
        "layout",
        "sidecar-size",
        "threshold",
        "favorable-rerun",
    )

    hip_source = Path(q5_f32.__file__).with_suffix(".hip")
    assert _sha256(hip_source) == _HIP_SOURCE_SHA256
    assert artifact["source_sha256"][str(hip_source.relative_to(_ROOT))] == (
        _HIP_SOURCE_SHA256
    )
    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY[
        ("bf16", 6_144, 3_072)
    ] == (
        "weight_major_tile_k_col_activation_tile_k_row_"
        "padded_compute_coltile16_rowbatch5"
    )
    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY[
        ("f32", 3_072, 6_144)
    ] == (
        "weight_major_tile_k_col_activation_tile_k_row_"
        "padded_compute_coltile16_rowbatch5"
    )
    for output_dtype in ("bf16", "f32"):
        producer = getattr(q5_f32, _producer_name(output_dtype))
        primitive = getattr(q5_f32, _h7g_primitive_name(output_dtype))
        composite = getattr(q5_f32, _h7g_composite_name(output_dtype))
        assert _sha256_source(producer) == _RETAINED_PRODUCER_WRAPPER_SHA256
        assert _sha256_source(primitive) == _RETAINED_H7G_PRIMITIVE_WRAPPER_SHA256
        assert _sha256_source(composite) == _RETAINED_H7G_COMPOSITE_WRAPPER_SHA256
        producer_key, composite_key = _retained_keys(output_dtype)
        assert resolve(
            backend=producer_key.backend,
            layer=producer_key.layer,
            quant=producer_key.quant,
            variant=producer_key.variant,
        ) is producer
        assert resolve(
            backend=composite_key.backend,
            layer=composite_key.layer,
            quant=composite_key.quant,
            variant=composite_key.variant,
        ) is composite


def test_h8a_composite_registry_preflight_and_gfx1151_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    for output_dtype in ("bf16", "f32"):
        producer_key, composite_key = _retained_keys(output_dtype)
        assert is_registered(producer_key)
        assert is_registered(composite_key)
        assert not is_registered(
            KernelKey(
                "hip_gfx1151",
                composite_key.layer,
                composite_key.quant,
                composite_key.variant,
            )
        )

        # Intentional RED after retained producer/H7G controls.
        candidate = getattr(q5_f32, _candidate_name(output_dtype))
        candidate_key = _candidate_key(output_dtype)
        assert resolve(
            backend=candidate_key.backend,
            layer=candidate_key.layer,
            quant=candidate_key.quant,
            variant=candidate_key.variant,
        ) is candidate
        assert not is_registered(_candidate_key(output_dtype, backend="hip_gfx1151"))

        def unexpected_build(**_kwargs: Any) -> None:
            raise AssertionError("invalid H8A role reached HIP loading")

        monkeypatch.setattr(
            q5_f32,
            "build_gguf_q5_k_f32_rocblas_prefill",
            unexpected_build,
        )
        role = next(role for role in _ROLES if role[1] == output_dtype)
        _, _, in_features, out_features, _ = role
        with pytest.raises(ValueError, match="rows must be positive"):
            candidate(1, 2, 3, 4, 0, in_features, out_features)
        with pytest.raises(ValueError, match=f"exactly {in_features}"):
            candidate(1, 2, 3, 4, 17, in_features - 256, out_features)
        with pytest.raises(ValueError, match=f"exactly {out_features}"):
            candidate(1, 2, 3, 4, 17, in_features, out_features - 16)
    assert _sha256(Path(q5_f32.__file__).with_suffix(".hip")) == _HIP_SOURCE_SHA256


def test_h8a_owner_allocates_publishes_and_frees_exact_24_plane_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = _FakeWeights()
    runtime = _FakeRuntime()
    producer_calls: list[dict[str, Any]] = []
    _install_fake_producers(monkeypatch, producer_calls)
    before = memory_stats()

    # Intentional RED after exact target inventory and current registry controls.
    owner_class = getattr(runner_module, _OWNER_CLASS)
    library = object()
    owner = owner_class.allocate(
        weights,
        backend="hip_gfx1100",
        library=library,
        runtime=runtime,
        stream=7,
    )
    try:
        expected_raw_ptrs = tuple(
            weights.layer(layer_id).weight(slot).allocation("raw").tensor.ptr
            for layer_id, slot, *_ in _TARGETS
        )
        assert owner.weights is weights
        assert owner.backend == "hip_gfx1100"
        assert owner.closed is False
        assert owner.allocation_count == _PLANE_COUNT
        assert owner.nbytes == _RESIDENT_BYTES
        assert len(owner.buffers) == _PLANE_COUNT
        assert len(owner.resident_planes) == _PLANE_COUNT
        assert tuple(owner.resident_planes) == expected_raw_ptrs
        assert len(set(owner.resident_planes)) == _PLANE_COUNT
        assert all(buffer.nbytes == _PLANE_BYTES for buffer in owner.buffers)
        assert len(set(buffer.ptr for buffer in owner.buffers)) == _PLANE_COUNT
        assert runtime.synchronizations == 1
        assert len(producer_calls) == _PLANE_COUNT
        assert [call["qweight_ptr"] for call in producer_calls] == list(
            expected_raw_ptrs
        )
        assert [call["out_ptr"] for call in producer_calls] == [
            buffer.ptr for buffer in owner.buffers
        ]
        assert [call["in_features"] for call in producer_calls] == [
            target[3] for target in _TARGETS
        ]
        assert [call["out_features"] for call in producer_calls] == [
            target[4] for target in _TARGETS
        ]
        assert [call["variant"] for call in producer_calls] == [
            _producer_variant(target[2]) for target in _TARGETS
        ]
        assert all(call["kwargs"]["library"] is library for call in producer_calls)
        assert all(call["kwargs"]["runtime"] is runtime for call in producer_calls)
        assert all(call["kwargs"]["stream"] == 7 for call in producer_calls)
        for target, raw_ptr in zip(_TARGETS, expected_raw_ptrs, strict=True):
            _, _, output_dtype, in_features, out_features, _ = target
            plane = owner.resident_planes[raw_ptr]
            assert plane.raw_weight_ptr == raw_ptr
            assert plane.weight_f32_ptr > 0
            assert plane.weight_f32_nbytes == _PLANE_BYTES
            assert plane.in_features == in_features
            assert plane.out_features == out_features
            assert plane.output_dtype == output_dtype
            assert plane.weight_layout == _WEIGHT_LAYOUT
            assert plane.col_tile == _COL_TILE
            assert plane.row_batch == _ROW_BATCH
        with pytest.raises(TypeError):
            owner.resident_planes[expected_raw_ptrs[0]] = object()
    finally:
        owner.free(runtime=runtime)
    assert owner.closed is True
    owner.free(runtime=runtime)
    assert runtime.allocations == {}
    assert runtime.free_order == list(reversed(runtime.malloc_order))
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


def test_h8a_owner_failed_setup_rolls_back_without_publishing_partial_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = _FakeWeights()
    runtime = _FakeRuntime()
    producer_calls: list[dict[str, Any]] = []
    _install_fake_producers(monkeypatch, producer_calls, fail_at=7)
    before = memory_stats()

    # Intentional RED at the same absent all-or-nothing owner surface.
    owner_class = getattr(runner_module, _OWNER_CLASS)
    with pytest.raises(RuntimeError, match="injected H8A producer failure"):
        owner_class.allocate(
            weights,
            backend="hip_gfx1100",
            library=object(),
            runtime=runtime,
        )
    assert len(producer_calls) == 7
    assert runtime.synchronizations == 0
    assert runtime.allocations == {}
    assert runtime.free_order == list(reversed(runtime.malloc_order))
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


def test_h8a_source_selected_shared_session_dispatch_and_unshared_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151

    assert not hasattr(hip_gfx1151, _SUPPORTED_CAPABILITY)
    # Source selection keeps explicit disable and unshared transient fallbacks.
    policy_resolver = getattr(runner_module, _RESOLVER)
    assert getattr(hip_gfx1100, _SUPPORTED_CAPABILITY) is True
    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) is True
    assert policy_resolver("hip_gfx1100", None) is True
    assert policy_resolver("hip_gfx1100", False) is False
    assert policy_resolver("hip_gfx1100", True) is True
    assert policy_resolver("hip_gfx1151", None) is False
    with pytest.raises(ValueError, match="not supported"):
        policy_resolver("hip_gfx1151", True)

    parameters = inspect.signature(
        runner_module.LagunaGGUFResidentSession.__init__
    ).parameters
    assert _SESSION_CACHE_PARAMETER in parameters
    assert _SESSION_ENABLE_PARAMETER in parameters

    plane_class = getattr(gguf_linear, _PLANE_CLASS)
    raw_ptr = 0x1000_0000
    resident_ptr = 0x5000_0000
    activation_ptr = 0x6000_0000
    transient_ptr = 0x7000_0000
    plane = plane_class(
        raw_weight_ptr=raw_ptr,
        weight_f32_ptr=resident_ptr,
        weight_f32_nbytes=_PLANE_BYTES,
        in_features=3_072,
        out_features=6_144,
        output_dtype="f32",
        weight_layout=_WEIGHT_LAYOUT,
        col_tile=_COL_TILE,
        row_batch=_ROW_BATCH,
    )
    immutable_planes = MappingProxyType({raw_ptr: plane})
    with pytest.raises(TypeError):
        immutable_planes[raw_ptr] = plane

    candidate_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    fallback_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def candidate(*args: Any, **kwargs: Any) -> None:
        candidate_calls.append((args, dict(kwargs)))

    def fallback(*args: Any, **kwargs: Any) -> None:
        fallback_calls.append((args, dict(kwargs)))

    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register(_candidate_key("f32"), candidate, replace=True)
    register(_retained_keys("f32")[1], fallback, replace=True)
    assert not is_registered(_candidate_key("f32", backend="hip_gfx1151"))
    gguf_linear.clear_gguf_linear_dispatch_cache()
    library = object()
    session = gguf_linear.Q5F32OrderedPrefillSession(
        min_rows=512,
        max_rows=512,
        weight_f32_ptr=transient_ptr,
        weight_f32_nbytes=150_994_944,
        activation_bf16_ptr=activation_ptr,
        activation_bf16_nbytes=10_125_312,
        library=library,
        **{_SESSION_PLANE_FIELD: immutable_planes},
    )
    matching = _FakeWeight(raw_ptr=raw_ptr, source_shape=(6_144, 3_072))
    unshared = _FakeWeight(raw_ptr=raw_ptr + 0x1000, source_shape=(6_144, 3_072))
    launch_kwargs = {
        "rows": 512,
        "in_features": 3_072,
        "out_features": 6_144,
        "output_dtype": "f32",
        "backend": "hip_gfx1100",
        "libraries": {
            "gguf_q5_k": library,
            f"gguf_q5_k:{_candidate_variant('f32')}": library,
        },
    }
    with gguf_linear.q5_f32_ordered_prefill_session(session):
        gguf_linear.launch_gguf_linear(
            matching,
            0x8000_0000,
            0x8100_0000,
            **launch_kwargs,
        )
        gguf_linear.launch_gguf_linear(
            unshared,
            0x8200_0000,
            0x8300_0000,
            **launch_kwargs,
        )
    no_cache_session = gguf_linear.Q5F32OrderedPrefillSession(
        min_rows=512,
        max_rows=512,
        weight_f32_ptr=transient_ptr,
        weight_f32_nbytes=150_994_944,
        activation_bf16_ptr=activation_ptr,
        activation_bf16_nbytes=10_125_312,
        library=library,
    )
    with gguf_linear.q5_f32_ordered_prefill_session(no_cache_session):
        gguf_linear.launch_gguf_linear(
            matching,
            0x8400_0000,
            0x8500_0000,
            **launch_kwargs,
        )

    assert len(candidate_calls) == 1
    assert len(fallback_calls) == 2
    candidate_args = candidate_calls[0][0]
    assert candidate_args[:4] == (
        0x8000_0000,
        resident_ptr,
        0x8100_0000,
        activation_ptr,
    )
    assert candidate_args[4:] == (512, 3_072, 6_144)
    for (args, _kwargs), expected_raw in zip(
        fallback_calls,
        (raw_ptr + 0x1000, raw_ptr),
        strict=True,
    ):
        assert args[1] == expected_raw
        assert args[3] == transient_ptr
        assert args[4] == activation_ptr


@pytest.fixture(scope="module")
def h8a_library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = (
        Path(version_file).read_text(encoding="utf-8").strip()
        if version_file
        else None
    )
    return q5_f32.build_gguf_q5_k_f32_rocblas_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=_require_cached_build(),
    )


@pytest.mark.parametrize(
    ("slot", "output_dtype", "in_features", "out_features", "source_shape"),
    _ROLES,
    ids=("global-attn-q-f32-k3072-n6144", "global-attn-output-bf16-k6144-n3072"),
)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h8a_complete_resident_plane_and_all_row_outputs_match_h7g(
    slot: str,
    output_dtype: str,
    in_features: int,
    out_features: int,
    source_shape: tuple[int, int],
    h8a_library: Any,
) -> None:
    from hipengine.core.hip import get_hip_runtime

    assert (slot, source_shape) in {
        ("attn_q", (6_144, 3_072)),
        ("attn_output", (3_072, 6_144)),
    }
    runtime = get_hip_runtime()
    before = memory_stats()
    rng = np.random.default_rng(0x8A00 + in_features + out_features)
    x_bf16 = _bf16_bits(
        rng.normal(0.0, 0.2, size=(max(_ROWS), in_features)).astype(np.float32)
    )
    qweight = _production_q5_weight(out_features, in_features)
    host_dtype = np.uint16 if output_dtype == "bf16" else np.float32
    max_output = np.empty((max(_ROWS), out_features), dtype=host_dtype)
    activation_shape = _activation_plane_shape(max(_ROWS), in_features, _ROW_BATCH)
    activation_host = np.empty(activation_shape, dtype=np.uint16)
    row_major_host = np.empty((out_features, in_features), dtype=np.float32)
    resident_host = np.empty(
        (out_features // _COL_TILE, in_features, _COL_TILE),
        dtype=np.float32,
    )
    buffers: list[DeviceBuffer] = []
    baseline_outputs: dict[int, np.ndarray] = {}
    try:
        x_dev = _device(x_bf16, runtime)
        qweight_dev = _device(qweight, runtime)
        row_major_dev = malloc(_PLANE_BYTES, runtime=runtime)
        resident_dev = malloc(_PLANE_BYTES, runtime=runtime)
        activation_dev = malloc(activation_host.nbytes, runtime=runtime)
        baseline_dev = malloc(max_output.nbytes, runtime=runtime)
        candidate_dev = malloc(max_output.nbytes, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                qweight_dev,
                row_major_dev,
                resident_dev,
                activation_dev,
                baseline_dev,
                candidate_dev,
            )
        )
        runtime.memset(row_major_dev.ptr, 0xA5, row_major_dev.nbytes)
        runtime.memset(resident_dev.ptr, 0x5A, resident_dev.nbytes)
        q5_f32.gguf_q5_k_dequantize_f32_exact(
            qweight_dev.ptr,
            row_major_dev.ptr,
            in_features,
            out_features,
            library=h8a_library,
            runtime=runtime,
        )
        retained_producer = getattr(q5_f32, _producer_name(output_dtype))
        retained_producer(
            qweight_dev.ptr,
            resident_dev.ptr,
            in_features,
            out_features,
            library=h8a_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(row_major_host),
            row_major_dev,
            row_major_host.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(resident_host),
            resident_dev,
            resident_host.nbytes,
            runtime=runtime,
        )
        expected_tile_k_col = row_major_host.reshape(
            out_features // _COL_TILE,
            _COL_TILE,
            in_features,
        ).transpose(0, 2, 1)
        np.testing.assert_array_equal(
            resident_host.view(np.uint32),
            expected_tile_k_col.view(np.uint32),
        )
        resident_sha256 = hashlib.sha256(resident_host.tobytes()).hexdigest()
        assert not np.all(resident_host.view(np.uint8) == np.uint8(0x5A))

        retained_h7g = getattr(q5_f32, _h7g_composite_name(output_dtype))
        for rows in _ROWS:
            runtime.memset(activation_dev.ptr, 0xA5, activation_dev.nbytes)
            runtime.memset(baseline_dev.ptr, 0x5A, max_output.nbytes)
            retained_h7g(
                x_dev.ptr,
                qweight_dev.ptr,
                baseline_dev.ptr,
                row_major_dev.ptr,
                activation_dev.ptr,
                rows,
                in_features,
                out_features,
                library=h8a_library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            baseline = np.empty((rows, out_features), dtype=host_dtype)
            activation = np.empty(
                _activation_plane_shape(rows, in_features, _ROW_BATCH),
                dtype=np.uint16,
            )
            copy_device_to_host(
                host_array_ptr(baseline),
                baseline_dev,
                baseline.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(activation),
                activation_dev,
                activation.nbytes,
                runtime=runtime,
            )
            np.testing.assert_array_equal(
                activation,
                _expected_activation_plane(x_bf16[:rows], _ROW_BATCH),
            )
            _sampled_cpu_gate(
                baseline,
                x_bf16[:rows],
                qweight,
                row_batch=_ROW_BATCH,
                output_dtype=output_dtype,
                out_features=out_features,
            )
            baseline_f32 = (
                _bf16_to_f32(baseline)
                if output_dtype == "bf16"
                else np.asarray(baseline, dtype=np.float32)
            )
            assert np.isfinite(baseline_f32).all()
            baseline_outputs[rows] = baseline.copy()

        # Intentional RED only after the complete retained plane, independent
        # row-major permutation, H7G outputs, CPU values, activation packing,
        # poison overwrite, finiteness, and all five row widths have passed.
        candidate = getattr(q5_f32, _candidate_name(output_dtype))
        for rows in _ROWS:
            expected = baseline_outputs[rows]
            for _repeat in range(2):
                runtime.memset(activation_dev.ptr, 0xA5, activation_dev.nbytes)
                runtime.memset(candidate_dev.ptr, 0x5A, max_output.nbytes)
                candidate(
                    x_dev.ptr,
                    resident_dev.ptr,
                    candidate_dev.ptr,
                    activation_dev.ptr,
                    rows,
                    in_features,
                    out_features,
                    library=h8a_library,
                    runtime=runtime,
                )
                runtime.device_synchronize()
                actual = np.empty_like(expected)
                activation = np.empty(
                    _activation_plane_shape(rows, in_features, _ROW_BATCH),
                    dtype=np.uint16,
                )
                copy_device_to_host(
                    host_array_ptr(actual),
                    candidate_dev,
                    actual.nbytes,
                    runtime=runtime,
                )
                copy_device_to_host(
                    host_array_ptr(activation),
                    activation_dev,
                    activation.nbytes,
                    runtime=runtime,
                )
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(
                    activation,
                    _expected_activation_plane(x_bf16[:rows], _ROW_BATCH),
                )
                actual_f32 = (
                    _bf16_to_f32(actual)
                    if output_dtype == "bf16"
                    else np.asarray(actual, dtype=np.float32)
                )
                assert np.isfinite(actual_f32).all()

        copy_device_to_host(
            host_array_ptr(resident_host),
            resident_dev,
            resident_host.nbytes,
            runtime=runtime,
        )
        assert hashlib.sha256(resident_host.tobytes()).hexdigest() == resident_sha256
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
