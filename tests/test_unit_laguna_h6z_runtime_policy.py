"""WPF-H6Z bounded runtime ownership contract."""

from __future__ import annotations

import inspect
from collections import Counter
from types import SimpleNamespace

import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION

_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6Z_ROLE_VARIANTS"
_H6W_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6W_ROLE_VARIANTS"
_H6A_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6A_ROLE_VARIANTS"
_SOURCE_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS"
_GLOBAL_ROLE = "global_m128_c4096_first_fill_exact"
_SWA_ROLE = "swa_qrow4_m128_c512_no_wrap_exact"
_H6N_GLOBAL = "global_context_rows_dense_initial_fixed512_cached_exact_spans"
_H6Z_GLOBAL = (
    "global_context_rows_qrow4_dense_initial_global_score_weight_replay_"
    "exact_spans"
)
_H6A_SWA = "swa_context_rows_qrow4_dense_initial_cached_exact_spans"
_H6W_SWA = (
    "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans"
)
_H6A_POLICY = {_GLOBAL_ROLE: _H6N_GLOBAL, _SWA_ROLE: _H6A_SWA}
_SOURCE_POLICY = {_GLOBAL_ROLE: _H6N_GLOBAL, _SWA_ROLE: _H6W_SWA}
_H6Z_POLICY = {_GLOBAL_ROLE: _H6Z_GLOBAL, _SWA_ROLE: _H6W_SWA}
_FALLBACK_STARTS = (0, 128)
_H6Z_STARTS = (256, 384)
_H6Z_SCORE_WEIGHT_BYTES = 12_582_912
_BOUND_SCORE_SCRATCH_BYTES = 18_874_368
_Q5_WEIGHT_PLANE_BYTES = 150_994_944
_Q5_ACTIVATION_PLANE_BYTES = 10_125_312
_Q5_WORKSPACE_BYTES = 161_120_256
_SOURCE_TOPOLOGY = {"H6N": 48, "H6A": 72, "H6W": 72}
_CANDIDATE_TOPOLOGY = {"H6N": 24, "H6Z": 24, "H6A": 72, "H6W": 72}
_ROLLBACK_TOPOLOGY = {"H6N": 48, "H6A": 144}


class _FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x10000000
        self.allocations: dict[int, int] = {}
        self.malloc_calls = 0

    def malloc(self, nbytes: int) -> int:
        self.malloc_calls += 1
        ptr = self.next_ptr
        self.next_ptr += max(0x1000, int(nbytes) + 0x100)
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.allocations.pop(int(ptr), None)

    def memcpy(self, dst: int, src: int, count: int, kind: HipMemcpyKind) -> None:
        del src, count
        assert kind == HipMemcpyKind.HOST_TO_DEVICE
        assert any(
            base <= int(dst) < base + max(size, 1)
            for base, size in self.allocations.items()
        )

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        del value, nbytes
        assert any(
            base <= int(dst) < base + max(size, 1)
            for base, size in self.allocations.items()
        )


def _production_config() -> SimpleNamespace:
    layer_types = tuple(
        FULL_ATTENTION if layer_id % 4 == 0 else SLIDING_ATTENTION
        for layer_id in range(48)
    )
    return SimpleNamespace(
        block_count=48,
        layer_types=layer_types,
        head_counts=tuple(
            48 if attention_type == FULL_ATTENTION else 72
            for attention_type in layer_types
        ),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )


def _install_fake_dispatch(cache, calls: list[tuple[str, tuple, dict]]) -> None:
    def resolve(layer: str, variant: str):
        calls.append((f"resolve:{layer}:{variant}", (), {}))

        def launch(*args: object, **kwargs: object) -> None:
            calls.append((f"launch:{variant}", args, dict(kwargs)))

        return launch

    cache._resolve = resolve


def _dispatch(cache, layer_id: int, start: int, rows: int = 128) -> bool:
    cache.position = start - 1
    cache.prepare_rows(tuple(range(start, start + rows)))
    qualified = cache.can_preappend_attention_prefill(layer_id, rows)
    if qualified:
        cache.append_rows(layer_id, 0x2000, 0x3000, rows)
        cache.attend_prefill_cached(
            layer_id,
            0x1000,
            0x2000,
            0x3000,
            0x4000,
            rows,
        )
    else:
        cache.attend_prefill(
            layer_id,
            0x1000,
            0x2000,
            0x3000,
            0x4000,
            rows,
        )
        cache.append_rows(layer_id, 0x2000, 0x3000, rows)
    cache.discard_rows()
    return qualified


def _attention_topology(launches: list[str]) -> dict[str, int]:
    selected = Counter(
        "H6N"
        if variant == _H6N_GLOBAL
        else "H6Z"
        if variant == _H6Z_GLOBAL
        else "H6A"
        if variant == _H6A_SWA
        else "H6W"
        if variant == _H6W_SWA
        else "other"
        for variant in launches
        if variant in {_H6N_GLOBAL, _H6Z_GLOBAL, _H6A_SWA, _H6W_SWA}
    )
    return dict(selected)


def _run_topology(
    monkeypatch: pytest.MonkeyPatch,
    policy: dict[str, str],
    *,
    bind_score_scratch: bool,
) -> tuple[dict[str, int], int]:
    from hipengine.kernels import hip_gfx1100
    from hipengine.runtime import laguna_kv as module

    monkeypatch.setattr(hip_gfx1100, _SOURCE_CAPABILITY, dict(policy))
    runtime = _FakeRuntime()
    cache = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    calls: list[tuple[str, tuple, dict]] = []
    _install_fake_dispatch(cache, calls)
    try:
        assert cache.prefill_preappend_role_variants == policy
        malloc_calls = runtime.malloc_calls
        if bind_score_scratch:
            cache.bind_prefill_score_scratch(
                0x40000000,
                _Q5_WEIGHT_PLANE_BYTES,
            )
            assert cache.prefill_score_scratch_bound
            assert cache.prefill_score_scratch_nbytes == (
                _BOUND_SCORE_SCRATCH_BYTES
            )
        assert runtime.malloc_calls == malloc_calls
        for layer_id in range(48):
            for start in (0, 128, 256, 384):
                assert _dispatch(cache, layer_id, start)
        launches = [
            call[0].removeprefix("launch:")
            for call in calls
            if call[0].startswith("launch:")
        ]
        return _attention_topology(launches), runtime.malloc_calls
    finally:
        cache.free()
        assert runtime.allocations == {}


def test_h6z_runtime_frozen_workspace_topology_and_promotion_contract() -> None:
    assert _FALLBACK_STARTS == (0, 128)
    assert _H6Z_STARTS == (256, 384)
    assert _H6Z_SCORE_WEIGHT_BYTES == 1_536 * 512 * 16
    assert _BOUND_SCORE_SCRATCH_BYTES == 2_304 * 512 * 16
    assert _H6Z_SCORE_WEIGHT_BYTES < _BOUND_SCORE_SCRATCH_BYTES
    assert _BOUND_SCORE_SCRATCH_BYTES < _Q5_WEIGHT_PLANE_BYTES
    assert _Q5_WEIGHT_PLANE_BYTES + _Q5_ACTIVATION_PLANE_BYTES == (
        _Q5_WORKSPACE_BYTES
    )
    assert _SOURCE_TOPOLOGY == {"H6N": 48, "H6A": 72, "H6W": 72}
    assert _CANDIDATE_TOPOLOGY == {
        "H6N": 24,
        "H6Z": 24,
        "H6A": 72,
        "H6W": 72,
    }
    assert sum(_SOURCE_TOPOLOGY.values()) == 192
    assert sum(_CANDIDATE_TOPOLOGY.values()) == 192
    assert _ROLLBACK_TOPOLOGY == {"H6N": 48, "H6A": 144}
    assert _SOURCE_POLICY[_GLOBAL_ROLE] == _H6N_GLOBAL
    assert _H6Z_POLICY[_GLOBAL_ROLE] == _H6Z_GLOBAL
    assert _SOURCE_POLICY[_SWA_ROLE] == _H6W_SWA
    assert _H6Z_POLICY[_SWA_ROLE] == _H6W_SWA


def test_h6z_runtime_capability_and_rollbacks_survive_source_promotion() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.kernels.registry import KernelKey, is_registered
    from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch

    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) == _H6Z_POLICY
    assert getattr(hip_gfx1100, _H6W_CAPABILITY) == _SOURCE_POLICY
    assert getattr(hip_gfx1100, _H6A_CAPABILITY) == _H6A_POLICY
    assert not hasattr(hip_gfx1151, _CAPABILITY)
    assert not hasattr(hip_gfx1151, _H6W_CAPABILITY)
    assert not hasattr(hip_gfx1151, _H6A_CAPABILITY)
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "laguna_attention_prefill",
            "bf16",
            _H6Z_GLOBAL,
        )
    )
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            "laguna_attention_prefill",
            "bf16",
            _H6Z_GLOBAL,
        )
    )
    assert LagunaQ5F32OrderedScratch.weight_f32_planned_nbytes() == (
        _Q5_WEIGHT_PLANE_BYTES
    )
    assert LagunaQ5F32OrderedScratch.activation_bf16_planned_nbytes(
        max_rows=512
    ) == _Q5_ACTIVATION_PLANE_BYTES
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == _Q5_WORKSPACE_BYTES

    # Intentional RED after leaf, source, rollback, backend, and workspace
    # controls: runtime qualification adds only a separate H6Z capability.
    assert getattr(hip_gfx1100, _CAPABILITY) == _H6Z_POLICY


def test_h6z_runtime_routes_only_late_global_starts_and_reuses_h6w_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.runtime import laguna_kv as module

    # Exercise the explicit H6N/H6A/H6W rollback before selected H6Z.
    monkeypatch.setattr(
        hip_gfx1100,
        _SOURCE_CAPABILITY,
        dict(_SOURCE_POLICY),
    )
    source_runtime = _FakeRuntime()
    source_cache = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=source_runtime,
    )
    source_calls: list[tuple[str, tuple, dict]] = []
    _install_fake_dispatch(source_cache, source_calls)
    try:
        assert source_cache.prefill_preappend_role_variants == _SOURCE_POLICY
        malloc_calls = source_runtime.malloc_calls
        source_cache.bind_prefill_score_scratch(
            0x40000000,
            _Q5_WEIGHT_PLANE_BYTES,
        )
        assert source_runtime.malloc_calls == malloc_calls
        for layer_id in (0, 1):
            for start in (0, 128, 256, 384):
                assert _dispatch(source_cache, layer_id, start)
        source_launches = [
            call[0]
            for call in source_calls
            if call[0].startswith("launch:")
        ]
        assert all(f"launch:{_H6Z_GLOBAL}" != call for call in source_launches)
        assert source_launches.count(f"launch:{_H6N_GLOBAL}") == 4
        assert source_launches.count(f"launch:{_H6A_SWA}") == 2
        assert source_launches.count(f"launch:{_H6W_SWA}") == 2
    finally:
        source_cache.free()
    assert source_runtime.allocations == {}

    capability = getattr(hip_gfx1100, _CAPABILITY)
    assert capability == _H6Z_POLICY
    monkeypatch.setattr(hip_gfx1100, _SOURCE_CAPABILITY, capability)
    runtime = _FakeRuntime()
    cache = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    calls: list[tuple[str, tuple, dict]] = []
    _install_fake_dispatch(cache, calls)
    score_ptr = 0x40000000
    try:
        assert cache.prefill_preappend_role_variants == _H6Z_POLICY
        assert cache.prefill_score_scratch_bound is False
        malloc_calls = runtime.malloc_calls
        cache.bind_prefill_score_scratch(score_ptr, _Q5_WEIGHT_PLANE_BYTES)
        assert runtime.malloc_calls == malloc_calls
        assert cache.prefill_score_scratch_bound is True
        assert cache.prefill_score_scratch_ptr == score_ptr
        assert cache.prefill_score_scratch_nbytes == _BOUND_SCORE_SCRATCH_BYTES
        assert cache.resident_nbytes == sum(
            buffer.nbytes for buffer in cache._buffers
        )

        for start in _FALLBACK_STARTS:
            before = len(calls)
            assert _dispatch(cache, 0, start)
            launches = [
                call for call in calls[before:] if call[0].startswith("launch:")
            ]
            assert [call[0] for call in launches] == [
                "launch:global_f32_rows_spans",
                f"launch:{_H6N_GLOBAL}",
            ]
            assert launches[-1][1][6].span_role == cache.layer(0).spans.span_role
            assert "score_scratch_nbytes" not in launches[-1][2]

        for start in _H6Z_STARTS:
            before = len(calls)
            assert _dispatch(cache, 0, start)
            launches = [
                call for call in calls[before:] if call[0].startswith("launch:")
            ]
            assert [call[0] for call in launches] == [
                "launch:global_f32_rows_spans",
                f"launch:{_H6Z_GLOBAL}",
            ]
            h6z_args = launches[-1][1]
            h6z_kwargs = launches[-1][2]
            assert h6z_args[6] == score_ptr
            assert h6z_args[7].span_role == cache.layer(0).spans.span_role
            assert h6z_args[8] == 128
            assert h6z_args[9] == 4096
            assert h6z_kwargs["score_scratch_nbytes"] == (
                _H6Z_SCORE_WEIGHT_BYTES
            )
            assert h6z_kwargs["start_position"] == start

        for start in _FALLBACK_STARTS:
            before = len(calls)
            assert _dispatch(cache, 1, start)
            launches = [
                call for call in calls[before:] if call[0].startswith("launch:")
            ]
            assert [call[0] for call in launches] == [
                "launch:swa_f32_rows_spans",
                f"launch:{_H6A_SWA}",
            ]

        for start in _H6Z_STARTS:
            before = len(calls)
            assert _dispatch(cache, 1, start)
            launches = [
                call for call in calls[before:] if call[0].startswith("launch:")
            ]
            assert [call[0] for call in launches] == [
                "launch:swa_f32_rows_spans",
                f"launch:{_H6W_SWA}",
            ]
            assert launches[-1][1][6] == score_ptr
            assert launches[-1][2]["score_scratch_nbytes"] == (
                _BOUND_SCORE_SCRATCH_BYTES
            )

        # Existing owner contract remains idempotent and binding.
        cache.bind_prefill_score_scratch(score_ptr, _Q5_WEIGHT_PLANE_BYTES)
        for ptr, nbytes, message in (
            (0, _Q5_WEIGHT_PLANE_BYTES, "non-zero"),
            (score_ptr + 8, _Q5_WEIGHT_PLANE_BYTES, "16-byte aligned"),
            (score_ptr, _BOUND_SCORE_SCRATCH_BYTES - 16, "at least"),
            (score_ptr + 0x1000, _Q5_WEIGHT_PLANE_BYTES, "already bound"),
        ):
            with pytest.raises(ValueError, match=message):
                cache.bind_prefill_score_scratch(ptr, nbytes)
    finally:
        cache.free()
    assert runtime.allocations == {}


def test_h6z_runtime_topology_and_unbound_policy_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100

    source, source_mallocs = _run_topology(
        monkeypatch,
        _SOURCE_POLICY,
        bind_score_scratch=True,
    )
    assert source == _SOURCE_TOPOLOGY

    capability = getattr(hip_gfx1100, _CAPABILITY)
    assert capability == _H6Z_POLICY
    candidate, candidate_mallocs = _run_topology(
        monkeypatch,
        capability,
        bind_score_scratch=True,
    )
    assert candidate == _CANDIDATE_TOPOLOGY
    assert candidate_mallocs == source_mallocs

    unbound, unbound_mallocs = _run_topology(
        monkeypatch,
        capability,
        bind_score_scratch=False,
    )
    assert unbound == _ROLLBACK_TOPOLOGY
    assert unbound_mallocs == source_mallocs


def test_h6z_runner_reuses_q5_plane_after_allocation_in_same_stream_order() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.runtime.laguna_gguf_runner import (
        LagunaGGUFResidentSession,
        LagunaQ5F32OrderedScratch,
    )

    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) == _H6Z_POLICY
    init_source = inspect.getsource(LagunaGGUFResidentSession.__init__)
    rows_source = inspect.getsource(LagunaGGUFResidentSession._run_layer_rows)
    cache_position = init_source.index("self.kv_cache = allocate_laguna_kv_cache(")
    q5_position = init_source.index("LagunaQ5F32OrderedScratch.allocate(")
    bind_position = init_source.index(".bind_prefill_score_scratch(")
    assert cache_position < q5_position < bind_position
    assert "q5_scratch.weight_f32_ptr" in init_source[bind_position:]
    assert "q5_scratch.weight_f32_nbytes" in init_source[bind_position:]
    assert LagunaQ5F32OrderedScratch.weight_f32_planned_nbytes() == (
        _Q5_WEIGHT_PLANE_BYTES
    )

    projection_position = rows_source.index(
        "self._launch_attention_projections_rows("
    )
    attention_position = rows_source.index("self.kv_cache.attend_prefill_cached(")
    ffn_position = rows_source.index("self._run_sparse_ffn_rows(")
    assert projection_position < attention_position < ffn_position
    projection_region = rows_source[projection_position:attention_position]
    attention_region = rows_source[attention_position:ffn_position]
    assert "stream=stream" in projection_region
    assert "stream=stream" in attention_region

    # Intentional RED after existing owner/order checks: H6Z must require no
    # new runner allocation or sidecar and keeps source policy unchanged.
    assert getattr(hip_gfx1100, _CAPABILITY) == _H6Z_POLICY
