from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION

_SOURCE_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS"
_H6A_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6A_ROLE_VARIANTS"
_H6W_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6W_ROLE_VARIANTS"
_H6Z_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6Z_ROLE_VARIANTS"
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
_H6W_POLICY = {_GLOBAL_ROLE: _H6N_GLOBAL, _SWA_ROLE: _H6W_SWA}
_H6Z_POLICY = {_GLOBAL_ROLE: _H6Z_GLOBAL, _SWA_ROLE: _H6W_SWA}
_SCORE_SCRATCH_BYTES = 18_874_368
_Q5_WEIGHT_PLANE_BYTES = 150_994_944
_Q5_WORKSPACE_BYTES = 161_120_256
_PRODUCTION_TOPOLOGY = {"H6N": 48, "H6A": 72, "H6W": 72}
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


def _install_dispatch(cache, launches: list[str]) -> None:
    def resolve(layer: str, variant: str):
        del layer

        def launch(*args: object, **kwargs: object) -> None:
            del args, kwargs
            launches.append(variant)

        return launch

    cache._resolve = resolve


def _dispatch(cache, layer_id: int, start: int) -> None:
    rows = 128
    cache.position = start - 1
    cache.prepare_rows(tuple(range(start, start + rows)))
    assert cache.can_preappend_attention_prefill(layer_id, rows)
    cache.append_rows(layer_id, 0x2000, 0x3000, rows)
    cache.attend_prefill_cached(
        layer_id,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        rows,
    )
    cache.discard_rows()


def _attention_topology(launches: list[str]) -> dict[str, int]:
    selected = Counter(
        "H6N"
        if variant == _H6N_GLOBAL
        else "H6A"
        if variant == _H6A_SWA
        else "H6W"
        if variant == _H6W_SWA
        else "other"
        for variant in launches
        if variant in {_H6N_GLOBAL, _H6A_SWA, _H6W_SWA}
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
    launches: list[str] = []
    _install_dispatch(cache, launches)
    try:
        assert cache.prefill_preappend_role_variants == policy
        malloc_calls = runtime.malloc_calls
        if bind_score_scratch:
            cache.bind_prefill_score_scratch(
                0x40000000,
                _Q5_WEIGHT_PLANE_BYTES,
            )
            assert cache.prefill_score_scratch_bound
            assert cache.prefill_score_scratch_nbytes == _SCORE_SCRATCH_BYTES
        assert runtime.malloc_calls == malloc_calls
        for layer_id in range(48):
            for start in (0, 128, 256, 384):
                _dispatch(cache, layer_id, start)
        return _attention_topology(launches), runtime.malloc_calls
    finally:
        cache.free()
        assert runtime.allocations == {}


def test_h6w_source_default_promotes_only_late_swa_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.kernels.registry import KernelKey, is_registered
    from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch

    live_source = getattr(hip_gfx1100, _SOURCE_CAPABILITY)
    assert live_source == _H6Z_POLICY
    assert getattr(hip_gfx1100, _H6W_CAPABILITY) == _H6W_POLICY
    assert getattr(hip_gfx1100, _H6Z_CAPABILITY) == _H6Z_POLICY
    assert not hasattr(hip_gfx1151, _H6Z_CAPABILITY)
    assert not hasattr(hip_gfx1151, _H6W_CAPABILITY)
    assert not hasattr(hip_gfx1151, _H6A_CAPABILITY)
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "laguna_attention_prefill",
            "bf16",
            _H6W_SWA,
        )
    )
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            "laguna_attention_prefill",
            "bf16",
            _H6W_SWA,
        )
    )
    assert set(_H6A_POLICY) == set(_H6W_POLICY)
    assert {
        role for role in _H6A_POLICY if _H6A_POLICY[role] != _H6W_POLICY[role]
    } == {_SWA_ROLE}
    assert _PRODUCTION_TOPOLOGY == {"H6N": 48, "H6A": 72, "H6W": 72}
    assert _ROLLBACK_TOPOLOGY == {"H6N": 48, "H6A": 144}
    assert LagunaQ5F32OrderedScratch.weight_f32_planned_nbytes() == (
        _Q5_WEIGHT_PLANE_BYTES
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == _Q5_WORKSPACE_BYTES

    production, production_mallocs = _run_topology(
        monkeypatch,
        _H6W_POLICY,
        bind_score_scratch=True,
    )
    assert production == _PRODUCTION_TOPOLOGY

    rollback, rollback_mallocs = _run_topology(
        monkeypatch,
        _H6A_POLICY,
        bind_score_scratch=False,
    )
    assert rollback == _ROLLBACK_TOPOLOGY
    assert production_mallocs == rollback_mallocs

    # Candidate policy without its externally bound plane fails safely back to
    # H6A at both late starts rather than launching H6W with an invalid pointer.
    unbound, unbound_mallocs = _run_topology(
        monkeypatch,
        _H6W_POLICY,
        bind_score_scratch=False,
    )
    assert unbound == _ROLLBACK_TOPOLOGY
    assert unbound_mallocs == rollback_mallocs

    # H6W remains the explicit H6N-global rollback after H6Z changes only the
    # selected global role; H6A remains the complete rollback.
    assert (
        getattr(hip_gfx1100, _H6A_CAPABILITY, None),
        getattr(hip_gfx1100, _H6W_CAPABILITY, None),
        live_source,
    ) == (_H6A_POLICY, _H6W_POLICY, _H6Z_POLICY)
