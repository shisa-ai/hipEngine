"""Prefill byte estimates are the allocation plan, not a cached extrapolation."""
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_runner as runtime
from tests.test_unit_qwen35_gguf_capacity_route_terms import _real_allocation_bytes
from tests.test_unit_qwen35_gguf_prefill_scratch_liveness import (
    _clear_diagnostic_environment, _fake_dense_qwen36_runner, _fake_runner,
)


@pytest.mark.parametrize('shape', ['dense', 'moe', 'dedicated', 'owner_slots'])
@pytest.mark.parametrize('rows,capacity,segments', [
    (7, 7, 1), (255, 256, 1), (255, 257, 3), (257, 1025, 1),
    (769, 4353, 3), (1025, 1025, 1),
])
def test_prefill_estimate_exact_at_block_and_route_boundaries(monkeypatch, shape, rows, capacity, segments):
    _clear_diagnostic_environment(monkeypatch)
    runner = _fake_dense_qwen36_runner() if shape == 'dense' else _fake_runner(
        {'dedicated': 'cpu_reference', 'owner_slots': 'hip_gfx1151'}.get(shape, 'hip_gfx1100'))
    expected = _real_allocation_bytes(monkeypatch, runner, rows=rows, capacity=capacity, segments=segments)
    assert runtime._gguf_prefill_scratch_nbytes(runner, rows=rows, capacity=capacity, segments=segments) == expected


def test_prefill_estimator_does_not_allocate_upload_or_build_host_metadata(monkeypatch):
    _clear_diagnostic_environment(monkeypatch)
    runner = _fake_runner()
    expected = _real_allocation_bytes(monkeypatch, runner, rows=257, capacity=4353, segments=3)

    def forbidden(*args, **kwargs):
        raise AssertionError('size-only planning must not materialize scratch')

    monkeypatch.setattr(runtime, 'malloc', forbidden)
    monkeypatch.setattr(runtime, 'copy_host_to_device', forbidden)
    monkeypatch.setattr(runtime, '_rope_tables', forbidden)
    monkeypatch.setattr(runtime.np, 'zeros', forbidden)
    monkeypatch.setattr(runtime.np, 'empty', forbidden)
    monkeypatch.setattr(runtime.np, 'arange', forbidden)
    assert runtime._gguf_prefill_scratch_nbytes(runner, rows=257, capacity=4353, segments=3) == expected


def test_prefill_estimator_observes_changed_geometry_and_policy(monkeypatch):
    _clear_diagnostic_environment(monkeypatch)
    runner = _fake_runner('cpu_reference')
    runtime._gguf_prefill_scratch_nbytes(runner, rows=257, capacity=4353)
    runner.fp16_recurrent_state = True
    runner.ffn_size = 768
    expected = _real_allocation_bytes(monkeypatch, runner, rows=257, capacity=4353)
    assert runtime._gguf_prefill_scratch_nbytes(runner, rows=257, capacity=4353) == expected


@pytest.mark.parametrize('capacity,expected', [
    (7, 7), (256, 256), (257, 256), (1024, 256), (1025, 1025),
    (65536, 1024), (131071, 1024), (131072, 768), (262144, 768),
])
def test_capacity_rows_use_the_resolved_architecture_and_memory_policy(capacity, expected):
    from hipengine.generation.qwen35_gguf import _capacity_prefill_rows_resolver

    runner = _fake_dense_qwen36_runner()
    runner.backend = 'hip_gfx1151'
    runner.target_arch = 'gfx1151'
    runner.runtime = SimpleNamespace(mem_get_info=lambda: (12 << 30, 24 << 30))
    assert _capacity_prefill_rows_resolver(runner)(capacity) == expected


def test_capacity_rows_preserve_geometry_chunk_precedence(monkeypatch):
    from hipengine.generation.qwen35_gguf import _capacity_prefill_rows_resolver

    runner = _fake_dense_qwen36_runner()
    runner.target_arch = 'gfx1151'
    monkeypatch.setattr(runtime, '_gguf_prefill_chunk_sizes_for', lambda *args: (123, 123))
    assert _capacity_prefill_rows_resolver(runner)(1024) == 123
