"""Qwen4Exp QSA chunk selection is request-local state.

``HIPENGINE_QWEN4_EXP_QSA_BATCHED_SELECTION`` used to be the only way to reach
the per-row selection fallback, and it was a process-wide switch. The decision
is now the owning runner's ``qsa_batched_selection`` field, threaded through the
prefill layer to the token mixer:

- ``True``  - the batched owner: one score pass, the top-k mask, and one
  expansion per row (production).
- ``False`` - the strict per-row fallback: one ``select_positions_device`` call
  per sparse row, keeping its own selection-capacity check.

Both arms keep the shape and count guards: the score-width capacity check, the
H4/D128 direct-scoring guard, and the WMMA mask-direct condition. These tests
pin the field, the scoped diagnostic control, the forwarding chain from the
prefill chunk, the surviving fallback, and that no process switch is read.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as runner_module

# The prefill layer's stub set is shared with the head-grouping tests.
from test_unit_qwen4exp_qsa_head_scope import _layer_stubs

RETIRED_ENV = "HIPENGINE_QWEN4_EXP_QSA_BATCHED_SELECTION"


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RETIRED_ENV, raising=False)


def _layer(monkeypatch: pytest.MonkeyPatch, recorded: dict[str, object], **extra):
    scratch, runtime, weights = _layer_stubs(monkeypatch, recorded)

    def mixer(*args, **kwargs):
        recorded["batched_selection"] = kwargs.get("batched_selection", "<absent>")
        return SimpleNamespace(ptr=31)

    monkeypatch.setattr(
        runner_module, "run_qwen4_exp_qsa_prefill_token_mixer", mixer
    )
    runner_module.run_qwen4_exp_qsa_prefill_layer(
        5,
        weights,
        attention_state=object(),
        index_state=object(),
        scratch=scratch,
        metadata=object(),
        start_position=0,
        rows=4,
        branches=2,
        hidden=64,
        low_rank=8,
        query_heads=24,
        kv_heads=2,
        head_dim=256,
        rotary_dim=64,
        theta=10000.0,
        index_heads=4,
        index_dim=128,
        index_rotary_dim=64,
        ffn=32,
        experts=8,
        top_k=2,
        runtime=runtime,
        **extra,
    )


def test_runner_defaults_to_the_batched_owner() -> None:
    runner = object.__new__(runner_module.Qwen4ExpGGUFResidentModelRunner)
    assert runner.qsa_batched_selection is True


def test_scope_binds_the_per_row_fallback_and_restores() -> None:
    runner = SimpleNamespace(qsa_batched_selection=True)

    with runner_module.qsa_batched_selection(runner, False) as prior:
        assert prior is True
        assert runner.qsa_batched_selection is False
        with runner_module.qsa_batched_selection(runner, True):
            assert runner.qsa_batched_selection is True
        assert runner.qsa_batched_selection is False
    assert runner.qsa_batched_selection is True


def test_scope_leaves_an_unset_binding_alone() -> None:
    runner = SimpleNamespace(qsa_batched_selection=True)

    with runner_module.qsa_batched_selection(runner):
        assert runner.qsa_batched_selection is True
    assert runner.qsa_batched_selection is True


def test_scope_restores_after_an_exception() -> None:
    runner = SimpleNamespace(qsa_batched_selection=True)

    with pytest.raises(RuntimeError, match="boom"):
        with runner_module.qsa_batched_selection(runner, False):
            raise RuntimeError("boom")
    assert runner.qsa_batched_selection is True


def test_scope_is_per_runner() -> None:
    first = SimpleNamespace(qsa_batched_selection=True)
    second = SimpleNamespace(qsa_batched_selection=True)

    with runner_module.qsa_batched_selection(first, False):
        assert second.qsa_batched_selection is True
        with runner_module.qsa_batched_selection(second, False):
            assert first.qsa_batched_selection is False
            assert second.qsa_batched_selection is False
        assert second.qsa_batched_selection is True
    assert first.qsa_batched_selection is True


@pytest.mark.parametrize("enabled", [True, False])
def test_prefill_layer_forwards_the_selection_to_the_mixer(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    recorded: dict[str, object] = {}

    _layer(monkeypatch, recorded, batched_selection=enabled)

    assert recorded["batched_selection"] is enabled


def test_prefill_layer_defaults_to_the_batched_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    _layer(monkeypatch, recorded)

    assert recorded["batched_selection"] is True


def test_prefill_chunk_forwards_the_runner_field() -> None:
    """The one prefill chunk path serves c1 and mixed chunks alike."""

    source = textwrap.dedent(
        inspect.getsource(
            runner_module.Qwen4ExpGGUFResidentModelRunner._prefill_chunk_steps
        )
    )
    call = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_qwen4_exp_qsa_prefill_layer"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}

    assert isinstance(keywords["batched_selection"], ast.Attribute)
    assert keywords["batched_selection"].attr == "qsa_batched_selection"


def test_retired_process_switch_is_not_read() -> None:
    mixer = textwrap.dedent(
        inspect.getsource(runner_module.run_qwen4_exp_qsa_prefill_token_mixer)
    )
    assert RETIRED_ENV not in mixer
    # Nothing else in the runtime may reintroduce the process transport.
    assert RETIRED_ENV not in inspect.getsource(runner_module)


def test_per_row_fallback_survives_the_transport_change() -> None:
    tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(runner_module.run_qwen4_exp_qsa_prefill_token_mixer)
        )
    )

    def calls(node: ast.AST, name: str) -> bool:
        return any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == name
            for child in ast.walk(node)
        )

    # One device selection per sparse row, inside a row loop.
    assert any(
        isinstance(node, ast.For) and calls(node, "select_positions_device")
        for node in ast.walk(tree)
    )
    # The score-width capacity guard.
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "score_blocks"
        and any(
            isinstance(side, ast.Attribute) and side.attr == "score_blocks"
            for side in node.comparators
        )
        for node in ast.walk(tree)
    )
    # The H4/D128 direct-scoring guard stays a runtime decision.
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_qwen4_exp_qsa_direct_select_enabled"
        for node in ast.walk(tree)
    )


def test_cache_only_verifier_call_site_never_selects() -> None:
    """The K-only verifier append returns before any selection dispatch."""

    from hipengine.runtime import qwen4_exp_mtp_prefill as mtp_module

    call = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(mtp_module)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_qwen4_exp_qsa_prefill_token_mixer"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}

    assert "batched_selection" not in keywords
    assert isinstance(keywords["cache_only"], ast.Constant)
    assert keywords["cache_only"].value is True
