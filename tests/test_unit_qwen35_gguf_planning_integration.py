"""Planning owns sizes; executor allocation and capacity share that implementation."""
from __future__ import annotations

import ast
import pickle
from pathlib import Path

from hipengine.runtime import qwen35_gguf_planning as planning
from hipengine.runtime import qwen35_gguf_runner as execution
from tests.test_unit_qwen35_gguf_capacity import (
    _FakeRuntime,
    _fake_dense_qwen36_runner,
    _install_fake_device,
)


def test_allocator_and_capacity_use_the_same_plan(monkeypatch):
    runner = _fake_dense_qwen36_runner()
    _install_fake_device(monkeypatch)
    calls = []
    original = planning._full_stack_scratch_plan
    assert execution._full_stack_scratch_plan is original

    def observed(config, **kwargs):
        assert config is runner.weights.config
        plan = original(config, **kwargs)
        calls.append((kwargs.get('materialize', True), plan))
        return plan

    monkeypatch.setattr(planning, '_full_stack_scratch_plan', observed)
    monkeypatch.setattr(execution, '_full_stack_scratch_plan', observed)
    scratch = execution._FullStackScratch.allocate(
        runner, runtime=_FakeRuntime(), max_sequence_length=640, allocate_kv_cache=False,
    )
    breakdown = planning.qwen35_gguf_resident_breakdown(
        runner.weights.config, context_tokens=640, hidden_size=runner.hidden_size,
        ffn_size=runner.ffn_size, q_width=runner.q_width, kv_width=runner.kv_width,
        linear_qkv_width=runner.linear_qkv_width, kv_storage_dtype='bf16',
        kv_scale_dtype='fp16', allocate_kv_cache=False,
    )
    assert [materialize for materialize, _ in calls] == [True, False]
    assert calls[0][1].owner_sizes == calls[1][1].owner_sizes
    assert sum(buffer.nbytes for buffer in scratch.buffers) == breakdown.scratch_bytes


def test_planning_imports_do_not_depend_on_execution_or_device_allocation():
    tree = ast.parse(Path(planning.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or '')
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
    forbidden = ('hipengine.generation', 'hipengine.kernels', 'hipengine.core.memory',
                 'hipengine.core.hip', 'hipengine.runtime.qwen35_gguf_runner')
    assert not [name for name in imports if name.startswith(forbidden)]


def test_legacy_plan_type_and_capacity_imports_remain_aliases():
    for name in ('_FullStackScratchPlan', 'Qwen35GGUFResidentBreakdown',
                 'Qwen35GGUFKVCapacityEstimate', 'qwen35_gguf_resident_breakdown',
                 'estimate_qwen35_gguf_kv_capacity'):
        assert getattr(execution, name) is getattr(planning, name)
    from tests.test_unit_qwen35_gguf_capacity import _plan_for
    plan = _plan_for(_fake_dense_qwen36_runner(), max_sequence_length=17, materialize=False)
    legacy = pickle.dumps(plan, protocol=0).replace(
        b'hipengine.runtime.qwen35_gguf_planning\n', b'hipengine.runtime.qwen35_gguf_runner\n',
    )
    assert pickle.loads(legacy).to_json_dict() == plan.to_json_dict()
