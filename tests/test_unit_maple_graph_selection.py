"""Maple graph identity includes the decode fusion choices sampled per step."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from hipengine.runtime.maple import MapleRunner
from hipengine.runtime.maple_graph import MapleGraphCache


@pytest.mark.parametrize("key,changed", [
    ((False, False), (False, True)),
    ((False, False), (True, False)),
    ((True, True), (False, False)),
])
def test_graph_cache_recaptures_changed_fusion_key_and_reuses_unchanged(key, changed):
    calls = []
    runtime = SimpleNamespace(
        stream_create=lambda **kw: 17,
        stream_destroy=lambda stream: calls.append(("stream_destroy", stream)),
        graph_launch=lambda graph, stream: calls.append(("launch", graph)),
        graph_exec_destroy=lambda graph: calls.append(("exec_destroy", graph)),
        graph_destroy=lambda graph: calls.append(("graph_destroy", graph)),
    )
    cache = MapleGraphCache(runtime)
    captured = []
    def capture(selected, *args):
        captured.append(selected)
        cache._graph_exec = len(captured) * 10
        cache._graph = len(captured)
        return "capture"
    cache._capture = capture
    def run(selected):
        return cache.run(selected, eager=lambda stream: None,
            argmax_index_ptr=1, argmax_value_ptr=2, stream=0)
    try:
        assert run(key) == "capture"
        assert run(key) == "replay"
        assert run(changed) == "capture"
        assert captured == [key, changed]
        assert ("exec_destroy", 10) in calls and ("graph_destroy", 1) in calls
        assert run(changed) == "replay"
        assert calls[-1] == ("launch", 20)
    finally:
        cache.close()
    assert ("exec_destroy", 20) in calls and ("graph_destroy", 2) in calls


def test_changed_fusion_key_can_retry_after_a_rejected_capture():
    runtime = SimpleNamespace(stream_create=lambda **kw: 17, stream_destroy=lambda stream: None)
    cache = MapleGraphCache(runtime)
    captures, eagers = [], []
    def capture(key, *args):
        captures.append(key)
        cache._eager_only = True
        return "eager"
    cache._capture = capture
    def run(key):
        return cache.run(key, eager=lambda stream: eagers.append(key),
            argmax_index_ptr=1, argmax_value_ptr=2, stream=0)
    try:
        assert run((False, False)) == "eager"
        assert run((False, False)) == "eager"
        assert captures == [(False, False)]
        assert eagers == [(False, False)]
        assert run((False, True)) == "eager"
        assert captures == [(False, False), (False, True)]
    finally:
        cache.close()


def test_step_graph_key_uses_both_sampled_fusion_values():
    tree = ast.parse(textwrap.dedent(inspect.getsource(MapleRunner.step)))
    run = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "run")
    assert isinstance(run.args[0], ast.Tuple)
    assert [node.id for node in run.args[0].elts] == ["fuse_qkattn", "fuse_moe"]
