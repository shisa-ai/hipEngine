"""Routed F16 always indexes token rows; other grouped routes still gather."""

import ast
import inspect
import os
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as runner

RETIRED = "HIPENGINE_QWEN4_EXP_ROUTED_F16_INDEXED"


def _tree():
    return ast.parse(inspect.getsource(runner.run_qwen4_exp_moe))


@pytest.mark.parametrize("value", ["0", "1", "", "false", "caller-sentinel"])
@pytest.mark.parametrize("routed", [False, True])
def test_scatter_selection_ignores_retired_environment(monkeypatch, value, routed):
    monkeypatch.setenv(RETIRED, value)
    grouped = next(node for node in ast.walk(_tree())
                   if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                   and node.test.id == "grouped_prefill")
    # Execute the production scatter selection with inert launchers. Include
    # the old assignment when present so this regression fails before removal.
    selected = []
    for node in grouped.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "routed_indexed"
                for target in node.targets):
            selected.append(node)
        if isinstance(node, ast.If) and any(
                isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "qwen35_moe_group_scatter"
                for call in ast.walk(node)):
            selected.append(node)
            break
    assert selected
    calls = []
    names = {node.attr for statement in selected for node in ast.walk(statement)
             if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
             and node.value.id == "scratch"}
    scratch = SimpleNamespace(**{name: SimpleNamespace(ptr=index)
                                for index, name in enumerate(sorted(names), 1)})
    namespace = dict(os=os, routed_f16=routed, scratch=scratch,
                     compact=12, experts=4, top_k=2, hidden=256,
                     stream=7, active_runtime=object(),
                     qwen35_moe_group_scatter=lambda *a, **k: calls.append("indexed"),
                     qwen35_moe_group_scatter_gather_lowp=lambda *a, **k: calls.append("gathered"))
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<scatter-route>", "exec"), namespace)
    assert calls == (["indexed"] if routed else ["gathered"])
    assert os.environ[RETIRED] == value


def test_routed_gate_up_receives_token_rows_and_lane_map():
    source = inspect.getsource(runner.run_qwen4_exp_moe)
    assert RETIRED not in source
    calls = [node for node in ast.walk(_tree()) if isinstance(node, ast.Call)
             and any(keyword.arg == "row_lanes_ptr" for keyword in node.keywords)]
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[0]) == "scratch.hidden_bf16.ptr"
    lanes = next(keyword.value for keyword in calls[0].keywords if keyword.arg == "row_lanes_ptr")
    assert ast.unparse(lanes) == "scratch.group_sorted_lanes.ptr"
