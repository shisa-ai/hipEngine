"""The batch owner carries the Q4 DP4A decode pair as request-local state.

The c1 runner already resolves the pair from its own fields. A batch owner is a
second owner over the same resident weights, so it must receive the same pair
when it is built, forward it to both layer owners, and include it in the
captured-segment identity. A bound pair must never read the ambient process
keys; an unbound owner keeps the historical direct fallback for a c1 row only.

These are CPU contract tests over recording fakes. No GPU and no kernels.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from types import SimpleNamespace

import pytest

from hipengine.generation.qwen4_exp_ar_batch import make_qwen4_exp_ar_batch_runner
from hipengine.runtime import qwen4_exp_batch_runner as runner_module
from hipengine.runtime import qwen4_exp_runner as c1_runner
from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner
from tests.test_unit_qwen4exp_batch_runner import _fake_runner, _install_step_fakes

Q4_ENV = "HIPENGINE_QWEN4_EXP_Q4_DP4A64"
Q4_LAYERS_ENV = "HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS"
Q4_FIELDS = ("q4_dp4a_decode_enabled", "q4_dp4a_decode_layers")


class _FakeBatch:
    """Capture the carrier's constructor and the fields it publishes."""

    def __init__(self, resident, **kwargs):
        self.resident = resident
        self.kwargs = kwargs


def _generator(*, enabled, layers):
    return SimpleNamespace(
        _resident=object(),
        runner=SimpleNamespace(max_sequence_length=4096, runtime=object()),
        backend="hip_gfx1151",
        q4_dp4a_decode_enabled=enabled,
        q4_dp4a_decode_layers=layers,
    )


def _carrier(monkeypatch, *, enabled, layers):
    monkeypatch.setattr(
        runner_module, "Qwen4ExpGGUFBatchRunner", _FakeBatch)
    generator = _generator(enabled=enabled, layers=layers)
    batch = make_qwen4_exp_ar_batch_runner(
        generator, state_slots=2, max_positions=4096)
    return generator, batch


def _owner(*, enabled=None, layers=None):
    owner = object.__new__(Qwen4ExpGGUFBatchRunner)
    owner.q4_dp4a_decode_enabled = enabled
    owner.q4_dp4a_decode_layers = layers
    return owner


def _recording_layers(monkeypatch):
    """Replace both layer owners with recorders of what ``step`` forwarded."""

    _install_step_fakes(monkeypatch)
    seen: list[tuple] = []

    def layer(name):
        def run(residual_ptr, weights, *, rows, **kwargs):
            seen.append(
                (name, int(rows),
                 kwargs.get("q4_dp4a_decode_enabled"),
                 kwargs.get("q4_dp4a_decode_layers"),
                 kwargs.get("graph_key"))
            )
            return SimpleNamespace(ptr=residual_ptr + 0x10)

        return run

    monkeypatch.setattr(
        runner_module, "run_qwen4_exp_batch_gdn_layer", layer("gdn"))
    monkeypatch.setattr(
        runner_module, "run_qwen4_exp_batch_qsa_layer", layer("qsa"))
    return seen


def _step(runner, rows: int):
    owners = [runner.admit(slot) for slot in range(rows)]
    try:
        return runner.step(
            [1] * rows,
            slots=list(range(rows)),
            owners=owners,
            capture_logits=False,
        )
    finally:
        for slot, owner in enumerate(owners):
            runner.release(slot, owner)


def test_the_carrier_copies_the_pair_in_both_construction_orders(monkeypatch):
    _first_generator, first = _carrier(
        monkeypatch, enabled=True, layers=frozenset({0, 2}))
    _second_generator, second = _carrier(
        monkeypatch, enabled=False, layers=frozenset())
    _third_generator, third = _carrier(
        monkeypatch, enabled=True, layers=frozenset({0, 2}))

    assert (first.q4_dp4a_decode_enabled,
            first.q4_dp4a_decode_layers) == (True, frozenset({0, 2}))
    assert (second.q4_dp4a_decode_enabled,
            second.q4_dp4a_decode_layers) == (False, frozenset())
    assert (third.q4_dp4a_decode_enabled,
            third.q4_dp4a_decode_layers) == (True, frozenset({0, 2}))


def test_the_carrier_copies_the_pair_instead_of_aliasing_the_generator(
        monkeypatch):
    generator, batch = _carrier(
        monkeypatch, enabled=True, layers=frozenset({0, 2}))
    generator.q4_dp4a_decode_enabled = False
    generator.q4_dp4a_decode_layers = frozenset()
    assert batch.q4_dp4a_decode_enabled is True
    assert batch.q4_dp4a_decode_layers == frozenset({0, 2})


@pytest.mark.parametrize("enabled,layers,expected", [
    (False, frozenset({0, 2}), (False, (0, 2))),
    (True, frozenset({5}), (True, (5,))),
    (True, frozenset(), (True, ())),
])
def test_a_bound_c1_segment_never_reads_the_ambient_pair(
        monkeypatch, enabled, layers, expected):
    monkeypatch.setenv(Q4_ENV, "1")
    monkeypatch.setenv(Q4_LAYERS_ENV, "not-an-int")
    assert _owner(enabled=enabled, layers=layers)._q4_dp4a_policy(1) == expected


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("0", False), ("false", False), ("FALSE", True), ("", False),
])
def test_an_unbound_c1_segment_keeps_the_historical_env_parser(
        monkeypatch, value, expected):
    monkeypatch.setenv(Q4_ENV, value)
    monkeypatch.setenv(Q4_LAYERS_ENV, "0,2")
    assert _owner()._q4_dp4a_policy(1) == (expected, (0, 2))


def test_a_wide_segment_does_not_resolve_the_ambient_pair(monkeypatch):
    # DP4A only selects at rows == 1. A malformed ambient layer list must not
    # newly fail an unrelated multi-row prefill-width capture key.
    monkeypatch.setenv(Q4_LAYERS_ENV, "not-an-int")
    assert _owner()._q4_dp4a_policy(2) == (None, ())
    assert _owner()._q4_dp4a_policy(64) == (None, ())


@pytest.mark.parametrize("rows", [1, 2, 64])
def test_step_forwards_the_bound_pair_to_both_layer_owners(monkeypatch, rows):
    monkeypatch.setenv(Q4_ENV, "1")
    monkeypatch.setenv(Q4_LAYERS_ENV, "not-an-int")
    seen = _recording_layers(monkeypatch)
    runner = _fake_runner(slots=rows)
    runner.q4_dp4a_decode_enabled = False
    runner.q4_dp4a_decode_layers = frozenset({0, 2})

    _step(runner, rows)

    assert seen, "the step dispatched no layer"
    for name, width, enabled, layers, _key in seen:
        assert width == rows, (name, width)
        assert enabled is False, name
        assert layers == frozenset({0, 2}), name


def test_repeated_off_on_restores_the_capture_identity(monkeypatch):
    seen = _recording_layers(monkeypatch)
    runner = _fake_runner(slots=1)
    runner.q4_dp4a_decode_enabled = False
    runner.q4_dp4a_decode_layers = frozenset({0, 2})

    _step(runner, 1)
    off = [key for _name, _width, _enabled, _layers, key in seen]
    seen.clear()
    runner.q4_dp4a_decode_enabled = True
    _step(runner, 1)
    on = [key for _name, _width, _enabled, _layers, key in seen]
    seen.clear()
    runner.q4_dp4a_decode_enabled = False
    _step(runner, 1)
    off_again = [key for _name, _width, _enabled, _layers, key in seen]

    assert off and on and off_again
    assert off != on
    assert off == off_again


def test_a_layer_scope_flip_changes_the_capture_identity(monkeypatch):
    seen = _recording_layers(monkeypatch)
    runner = _fake_runner(slots=1)
    runner.q4_dp4a_decode_enabled = True
    runner.q4_dp4a_decode_layers = frozenset({0, 2})

    _step(runner, 1)
    first = [key for _name, _width, _enabled, _layers, key in seen]
    seen.clear()
    runner.q4_dp4a_decode_layers = frozenset({5})
    _step(runner, 1)
    second = [key for _name, _width, _enabled, _layers, key in seen]

    assert first and second
    assert first != second


def test_an_unbound_c1_batch_key_follows_the_ambient_pair(monkeypatch):
    seen = _recording_layers(monkeypatch)
    runner = _fake_runner(slots=1)
    monkeypatch.setenv(Q4_ENV, "0")
    monkeypatch.setenv(Q4_LAYERS_ENV, "0,2")

    _step(runner, 1)
    off = [key for _name, _width, _enabled, _layers, key in seen]
    seen.clear()
    monkeypatch.setenv(Q4_ENV, "1")
    _step(runner, 1)
    on = [key for _name, _width, _enabled, _layers, key in seen]

    assert off and on
    assert off != on


def test_the_mixed_target_pass_forwards_the_pair_from_the_batch():
    from hipengine.runtime.qwen4_exp_mixed_target import Qwen4ExpMixedTargetPass

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(Qwen4ExpMixedTargetPass._verify_rows)))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_run_layer_ffn"
    ]
    assert calls, "the pass no longer calls the layer FFN"
    for call in calls:
        forwarded = {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg in Q4_FIELDS
        }
        assert set(forwarded) == set(Q4_FIELDS)
        for field, value in forwarded.items():
            assert isinstance(value, ast.Call), field
            assert getattr(value.func, "id", None) == "getattr", field
            assert getattr(value.args[0], "id", None) == "batch", field
            assert value.args[1].value == field, field
            assert isinstance(value.args[2], ast.Constant), field
            assert value.args[2].value is None, field


def test_a_graph_cache_hit_does_not_resolve_the_ambient_pair(monkeypatch):
    # The graph-cache branch used to resolve the pair before asking the cache.
    # A hit must not turn a malformed ambient list into a new failure.
    monkeypatch.setenv(Q4_LAYERS_ENV, "not-an-int")
    scratch = SimpleNamespace(
        closed=False,
        runtime=object(),
        output=SimpleNamespace(ptr=1),
        selected=object(),
        routing=object(),
    )
    cache = SimpleNamespace(run=lambda key, **kwargs: None)
    c1_runner.run_qwen4_exp_moe(
        0, {}, scratch=scratch, rows=1, hidden=2, ffn=2, experts=2, top_k=1,
        graph_cache=cache, graph_key="layer",
    )


def test_the_grouped_prefill_branch_precedes_the_non_graph_lookup():
    # The old lookup lived in the non-grouped else arm. Resolving at function
    # entry would make a malformed ambient list fail a grouped multi-row
    # prefill that never reads the pair.
    source = inspect.getsource(c1_runner.run_qwen4_exp_moe)
    grouped = source.index("if grouped_prefill:")
    else_block = source.index("    else:\n        gate_weight", grouped)
    eager = source.index("        def eager(")
    graph_run = source.index("        graph_cache.run(", eager)
    lookups = [match.start() for match in re.finditer(
        r"_resolve_q4_dp4a_decode\(", source)]
    assert len(lookups) == 2
    assert eager < lookups[0] < graph_run
    assert lookups[1] > else_block
    scope = source.index("_qwen4_exp_q4_dp4a_decode_scope(", else_block)
    registered = source.index("is_registered(dp4a_key)", scope)
    assert registered > scope
