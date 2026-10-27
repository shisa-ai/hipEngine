"""The Q8_0 dense WMMA prefill tile pair is execution-owned, not a process key.

``HIPENGINE_GGUF_Q8_0_WMMA_TILE_M`` / ``_N`` used to be read by
``gguf_q8_0_prefill._default_tiles``, and the Qwen4Exp profile binder wrote
``64``/``32`` into the process environment for every plan. One dense Q8_0
projection family is shared by every GGUF model, so that pin reached whatever
else ran in the same process: a co-resident Gemma 4 runner launched its dense
projections at the Qwen plan's tile instead of its own arch heuristic, and on
gfx1151 the arch rule was bypassed outright.

Both readers are retired. The pair is now a wrapper-level ``ContextVar`` that a
runner binds for the scope it launches in, so it is visible to the launches that
scope makes and to nothing else:

* an explicit ``tile_m`` / ``tile_n`` launch argument still wins, per side;
* a scoped pair replaces the arch heuristic for the sides left open;
* ``(None, None)`` masks an outer pair and restores the heuristic;
* the binder publishes ``(64, 32)`` on the Qwen4Exp carrier and writes no
  environment variable.

The Qwen4Exp entries that can reach that family under an existing opt-in bind
it: the chunked/serial prefill driver (which also carries the state-row verify
pass), the batch step, and the mixed-target pass. The serial entries cannot
reach it - they launch one row at a time, and the family requires ``rows > 1`` -
so they carry no bind.
"""

from __future__ import annotations

import ast
import contextvars
import inspect
import os
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation import qwen4_exp_ar_batch, qwen4_exp_gguf
from hipengine.generation.qwen4_exp_profiles import (
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_prefill as q8
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime import qwen4_exp_batch_runner as batch_module
from hipengine.runtime import qwen4_exp_mixed_target as mixed_module
from hipengine.runtime import qwen4_exp_runner as runner_module

RETIRED_KEYS = (
    "HIPENGINE_GGUF_Q8_0_WMMA_TILE_M",
    "HIPENGINE_GGUF_Q8_0_WMMA_TILE_N",
)
FIELD = "q8_wmma_tiles"
ROOT = Path(__file__).resolve().parents[1]
SCOPE = "_qwen4_exp_q8_tile_scope"

#: The plan pair, and the gfx1100 heuristic a Gemma-shaped launch falls back to.
QWEN_PAIR = (64, 32)
GEMMA_SHAPE = dict(rows=512, in_features=2816, out_features=4096)
GFX1100_HEURISTIC = (16, 32)


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in RETIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1100")


@pytest.fixture
def profiles():
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    return None


# --- the retired transport --------------------------------------------------


def _environ_reads(root: Path) -> list[tuple[str, int, object]]:
    """Every ``os.environ.get(...)`` / ``os.environ[...]`` literal in a tree."""

    found: list[tuple[str, int, object]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if not (
                    isinstance(func, ast.Attribute)
                    and func.attr in {"get", "setdefault", "pop"}
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "environ"
                ):
                    continue
                if node.args and isinstance(node.args[0], ast.Constant):
                    found.append((str(path), node.lineno, node.args[0].value))
            elif isinstance(node, ast.Subscript):
                value = node.value
                if (
                    isinstance(value, ast.Attribute)
                    and value.attr == "environ"
                    and isinstance(node.slice, ast.Constant)
                ):
                    found.append((str(path), node.lineno, node.slice.value))
    return found


def test_no_module_reads_the_retired_tile_keys() -> None:
    for root in (ROOT / "hipengine", ROOT / "scripts"):
        offenders = [
            (path, line, key)
            for path, line, key in _environ_reads(root)
            if key in RETIRED_KEYS
        ]
        assert offenders == []


def test_the_default_tiles_helper_reads_no_environment() -> None:
    assert "os.environ" not in inspect.getsource(q8._default_tiles)


def test_the_retired_keys_are_documented_as_retired() -> None:
    source = inspect.getsource(q8.wmma_tile_policy)
    assert "retired" in source
    for key in RETIRED_KEYS:
        assert key in source


# --- resolution -------------------------------------------------------------


def test_an_unbound_launch_uses_the_arch_heuristic(monkeypatch) -> None:
    assert q8._default_tiles(**GEMMA_SHAPE) == GFX1100_HEURISTIC
    assert q8._default_tiles(rows=8, in_features=2816, out_features=4096) == (16, 16)
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")
    assert q8._default_tiles(rows=512, in_features=2816, out_features=4096) == (32, 64)
    assert q8._default_tiles(rows=8, in_features=2816, out_features=4096) == (32, 16)


def test_a_scoped_pair_replaces_the_heuristic() -> None:
    with q8.wmma_tile_policy(*QWEN_PAIR):
        assert q8._default_tiles(**GEMMA_SHAPE) == QWEN_PAIR
    assert q8._default_tiles(**GEMMA_SHAPE) == GFX1100_HEURISTIC


def test_none_masks_an_outer_pair() -> None:
    with q8.wmma_tile_policy(*QWEN_PAIR):
        with q8.wmma_tile_policy(None, None):
            assert q8._default_tiles(**GEMMA_SHAPE) == GFX1100_HEURISTIC
        assert q8._default_tiles(**GEMMA_SHAPE) == QWEN_PAIR
    assert q8.current_wmma_tile_policy() is None


def test_explicit_launch_dimensions_win_independently() -> None:
    resolve = q8._resolve_launch_tiles
    with q8.wmma_tile_policy(*QWEN_PAIR):
        # Both explicit: the scope is not consulted at all.
        assert resolve(16, 16, **GEMMA_SHAPE) == (16, 16)
        # One explicit side keeps its value; the other comes from the scope.
        assert resolve(16, None, **GEMMA_SHAPE) == (16, 32)
        assert resolve(None, 16, **GEMMA_SHAPE) == (64, 16)
        # Neither explicit: the scope supplies both.
        assert resolve(None, None, **GEMMA_SHAPE) == QWEN_PAIR
    assert resolve(16, None, **GEMMA_SHAPE) == (16, 32)
    assert resolve(None, 16, **GEMMA_SHAPE) == (16, 16)


def test_the_dual_owner_resolves_the_same_way() -> None:
    source = inspect.getsource(q8._launch_dual)
    assert "_resolve_launch_tiles(" in source
    assert "max(out_features_a, out_features_b)" in source


def test_the_fused_dual_consumer_keeps_its_explicit_pair() -> None:
    """The dense Q8 dual gate/up call is not a policy consumer."""

    from hipengine.runtime import gguf_linear

    source = inspect.getsource(gguf_linear)
    tree = ast.parse(textwrap.dedent(source))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert isinstance(keywords["tile_m"], ast.Constant) and keywords["tile_m"].value == 16
    assert isinstance(keywords["tile_n"], ast.Constant) and keywords["tile_n"].value == 32


# --- validation -------------------------------------------------------------


def test_a_lone_side_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires both"):
        with q8.wmma_tile_policy(64, None):
            pass
    with pytest.raises(ValueError, match="requires both"):
        with q8.wmma_tile_policy(None, 32):
            pass
    assert q8.current_wmma_tile_policy() is None


def test_an_unsupported_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="not supported"):
        with q8.wmma_tile_policy(48, 16):
            pass
    with pytest.raises(ValueError, match="not supported"):
        with q8.wmma_tile_policy(128, 128):
            pass
    assert q8.normalize_wmma_tile_pair(None, None) is None


def test_the_binding_is_restored_after_an_exception() -> None:
    with q8.wmma_tile_policy(*QWEN_PAIR):
        with pytest.raises(RuntimeError, match="arm failed"):
            with q8.wmma_tile_policy(16, 16):
                assert q8.current_wmma_tile_policy() == (16, 16)
                raise RuntimeError("arm failed")
        assert q8.current_wmma_tile_policy() == QWEN_PAIR
    assert q8.current_wmma_tile_policy() is None


# --- suspension, interleaving and cancellation ------------------------------


def _binding_generator(pair, seen):
    with q8.wmma_tile_policy(*pair):
        seen.append(q8.current_wmma_tile_policy())
        yield "first"
        seen.append(q8.current_wmma_tile_policy())
        yield "second"


def test_a_suspended_step_does_not_leak_into_the_caller() -> None:
    """The resumable driver runs each ``next()`` in a copied context."""

    seen: list[object] = []
    steps = _binding_generator(QWEN_PAIR, seen)
    context = contextvars.copy_context()

    assert context.run(next, steps) == "first"
    # The caller's context is untouched, so a co-resident model keeps its own
    # default while the Qwen step is suspended.
    assert q8.current_wmma_tile_policy() is None
    assert q8._default_tiles(**GEMMA_SHAPE) == GFX1100_HEURISTIC

    assert context.run(next, steps) == "second"
    assert seen == [QWEN_PAIR, QWEN_PAIR]
    assert q8.current_wmma_tile_policy() is None
    context.run(steps.close)


def test_closing_a_suspended_step_restores_the_caller() -> None:
    seen: list[object] = []
    steps = _binding_generator(QWEN_PAIR, seen)
    context = contextvars.copy_context()
    assert context.run(next, steps) == "first"

    context.run(steps.close)

    assert seen == [QWEN_PAIR]
    assert q8.current_wmma_tile_policy() is None
    assert q8._default_tiles(**GEMMA_SHAPE) == GFX1100_HEURISTIC


def test_cancellation_closes_in_the_owning_context() -> None:
    """The lifecycle closes a suspended step where it was entered."""

    steps = _binding_generator(QWEN_PAIR, [])
    context = contextvars.copy_context()
    assert context.run(next, steps) == "first"

    # This is what ``reset``/``close``/the resumable driver's failure path do.
    context.run(steps.close)

    assert q8.current_wmma_tile_policy() is None
    assert q8._default_tiles(**GEMMA_SHAPE) == GFX1100_HEURISTIC


def test_a_cross_context_exit_is_not_silenced() -> None:
    """A scope exited elsewhere is a lifecycle bug, not something to absorb."""

    scope = q8.wmma_tile_policy(*QWEN_PAIR)
    context = contextvars.copy_context()
    context.run(scope.__enter__)
    assert context.run(q8.current_wmma_tile_policy) == QWEN_PAIR

    with pytest.raises(ValueError, match="different Context"):
        scope.__exit__(None, None, None)

    assert q8.current_wmma_tile_policy() is None


def test_two_copied_contexts_do_not_see_each_other() -> None:
    qwen_seen: list[object] = []
    other_seen: list[object] = []
    qwen = _binding_generator(QWEN_PAIR, qwen_seen)
    other = _binding_generator((16, 16), other_seen)
    first, second = contextvars.copy_context(), contextvars.copy_context()

    assert first.run(next, qwen) == "first"
    assert second.run(next, other) == "first"
    assert first.run(next, qwen) == "second"
    assert second.run(next, other) == "second"
    assert qwen_seen == [QWEN_PAIR, QWEN_PAIR]
    assert other_seen == [(16, 16), (16, 16)]
    first.run(qwen.close)
    second.run(other.close)
    assert q8.current_wmma_tile_policy() is None


# --- the Qwen4Exp carrier ---------------------------------------------------


def _resolve(profile: ExecutionProfile):
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant="gguf_ud_q4_k_xl",
        profile=profile,
    )


@pytest.mark.parametrize(
    "profile", [ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION])
def test_the_binder_publishes_the_pair_and_writes_no_key(profiles, profile) -> None:
    resolved = _resolve(profile)
    generator = SimpleNamespace()
    resolved.binder(generator, resolved)

    assert getattr(generator, FIELD) == QWEN_PAIR
    for key in RETIRED_KEYS:
        assert key not in os.environ


def test_the_binder_reaches_a_runner_it_finds(profiles) -> None:
    resolved = _resolve(ExecutionProfile.PRODUCTION)
    runner = SimpleNamespace()
    resolved.binder(SimpleNamespace(runner=runner), resolved)
    assert getattr(runner, FIELD) == QWEN_PAIR


def test_a_qwen_bind_leaves_a_gemma_launch_on_its_heuristic(profiles) -> None:
    """The failure this guards: a bound Qwen plan moving another model's tile."""

    resolved = _resolve(ExecutionProfile.PRODUCTION)
    generator = SimpleNamespace()
    resolved.binder(generator, resolved)
    assert generator.q8_wmma_tiles == QWEN_PAIR
    # Binding the plan does not bind the context: the co-resident model's
    # default launch still resolves the arch heuristic.
    assert q8._default_tiles(**GEMMA_SHAPE) == GFX1100_HEURISTIC
    assert q8.current_wmma_tile_policy() is None


def test_the_runner_defaults_unbound() -> None:
    runner = object.__new__(runner_module.Qwen4ExpGGUFResidentModelRunner)
    assert getattr(runner, FIELD) is None


def test_the_scope_binds_and_restores() -> None:
    runner = SimpleNamespace(**{FIELD: QWEN_PAIR})
    with runner_module.q8_wmma_tiles(runner, (16, 16)) as prior:
        assert prior == QWEN_PAIR
        assert runner.q8_wmma_tiles == (16, 16)
    assert runner.q8_wmma_tiles == QWEN_PAIR


def test_the_scope_keeps_an_unset_binding() -> None:
    runner = SimpleNamespace(**{FIELD: QWEN_PAIR})
    with runner_module.q8_wmma_tiles(runner):
        assert runner.q8_wmma_tiles == QWEN_PAIR


def test_the_scope_validates_and_restores_on_failure() -> None:
    runner = SimpleNamespace(**{FIELD: QWEN_PAIR})
    with pytest.raises(ValueError, match="not supported"):
        with runner_module.q8_wmma_tiles(runner, (48, 16)):
            pass
    assert runner.q8_wmma_tiles == QWEN_PAIR
    with pytest.raises(ValueError, match="requires both"):
        with runner_module.q8_wmma_tiles(runner, (64, None)):
            pass
    assert runner.q8_wmma_tiles == QWEN_PAIR
    with pytest.raises(RuntimeError, match="arm failed"):
        with runner_module.q8_wmma_tiles(runner, (16, 16)):
            raise RuntimeError("arm failed")
    assert runner.q8_wmma_tiles == QWEN_PAIR


def test_the_scope_can_mask_the_plan_pair() -> None:
    runner = SimpleNamespace(**{FIELD: QWEN_PAIR})
    with runner_module.q8_wmma_tiles(runner, None):
        assert runner.q8_wmma_tiles is None
    assert runner.q8_wmma_tiles == QWEN_PAIR


def test_an_unbound_owner_masks_the_callers_pair() -> None:
    """Ownership, not an optional override: a nested runner cannot inherit."""

    unbound = SimpleNamespace()
    with q8.wmma_tile_policy(64, 32):
        with runner_module._qwen4_exp_q8_tile_scope(unbound):
            assert q8.current_wmma_tile_policy() is None
            assert q8._default_tiles(**GEMMA_SHAPE) == GFX1100_HEURISTIC
        # The caller's pair is restored when the runner's entry returns.
        assert q8.current_wmma_tile_policy() == (64, 32)
    assert q8.current_wmma_tile_policy() is None


def test_a_published_owner_replaces_the_callers_pair() -> None:
    published = SimpleNamespace(**{FIELD: (16, 16)})
    with q8.wmma_tile_policy(64, 32):
        with runner_module._qwen4_exp_q8_tile_scope(published):
            assert q8.current_wmma_tile_policy() == (16, 16)
        assert q8.current_wmma_tile_policy() == (64, 32)


def test_the_clone_and_batch_carriers_forward_the_pair() -> None:
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    clone = SimpleNamespace()
    _inherit_lane_decisions(SimpleNamespace(**{FIELD: QWEN_PAIR}), clone)
    assert getattr(clone, FIELD) == QWEN_PAIR
    _inherit_lane_decisions(SimpleNamespace(), clone)
    assert getattr(clone, FIELD) is None

    source = inspect.getsource(qwen4_exp_ar_batch)
    assert f"batch.{FIELD} = " in source
    assert f'getattr(generator, "{FIELD}", None)' in source


# --- the bind sites ---------------------------------------------------------


def test_the_chunk_loop_binds_the_plan_pair() -> None:
    source = inspect.getsource(
        runner_module.Qwen4ExpGGUFResidentModelRunner._prefill_chunked_steps)
    assert f"{SCOPE}(self)" in source
    # Inside the existing per-chunk session tuple, not once per request.
    assert "_wide_f16_activation_context()," in source


def test_the_synchronous_chunk_driver_binds_the_plan_pair() -> None:
    source = inspect.getsource(
        runner_module.Qwen4ExpGGUFResidentModelRunner._prefill_chunk)
    assert f"with {SCOPE}(self):" in source


def test_the_state_row_verify_pass_binds_for_the_whole_entry() -> None:
    """It launches rows>1 projections after the prefill body as well."""

    entry = inspect.getsource(
        runner_module.Qwen4ExpGGUFResidentModelRunner.verify_target_block_state_rows)
    assert f"with {SCOPE}(self):" in entry
    assert "self._verify_state_rows(" in entry
    body = inspect.getsource(
        runner_module.Qwen4ExpGGUFResidentModelRunner._verify_state_rows)
    assert "self._prefill_chunk(" in body
    assert "run_qwen4_exp_gr_read(" in body


def test_the_batch_step_binds_and_keys_the_plan_pair() -> None:
    step_source = inspect.getsource(
        batch_module.Qwen4ExpGGUFBatchRunner.step)
    assert f"with {SCOPE}(self):" in step_source
    assert "self._execute_step(" in step_source

    owner = object.__new__(batch_module.Qwen4ExpGGUFBatchRunner)
    owner.q8_wmma_tiles = QWEN_PAIR
    assert owner._q8_tile_policy() == (QWEN_PAIR,)
    owner.q8_wmma_tiles = None
    assert owner._q8_tile_policy() == (None,)

    execute_source = inspect.getsource(
        batch_module.Qwen4ExpGGUFBatchRunner._execute_step)
    assert "self._q8_tile_policy()" in execute_source


def test_the_mixed_target_pass_binds_the_batch_pair() -> None:
    source = inspect.getsource(mixed_module.Qwen4ExpMixedTargetPass.verify)
    assert f"{SCOPE}(self.batch)" in source
    assert "self._verify_rows()" in source


@pytest.mark.parametrize(
    "name",
    ["step", "prefill_serial", "verify_target_block_serial_exact"],
)
def test_a_one_row_entry_carries_no_bind(name) -> None:
    """These launch one row at a time, and the family requires ``rows > 1``."""

    source = inspect.getsource(
        getattr(runner_module.Qwen4ExpGGUFResidentModelRunner, name))
    assert SCOPE not in source
    assert "_prefill_chunk(" not in source


def test_the_deferred_head_verify_binds_for_the_whole_entry() -> None:
    """Its bodies are serial, but the deferred head read is one rows>1 pass."""

    entry = inspect.getsource(
        runner_module.Qwen4ExpGGUFResidentModelRunner.verify_target_block_deferred_head)
    assert f"with {SCOPE}(self):" in entry
    assert "self._deferred_head_rows(" in entry
    body = inspect.getsource(
        runner_module.Qwen4ExpGGUFResidentModelRunner._deferred_head_rows)
    assert "rows=len(tokens)," in body
    assert "launch_gguf_linear(" in body


def test_the_acceptance_entry_reaches_the_bound_deferred_head() -> None:
    """It owns no launches of its own; the head read is the bound entry."""

    source = inspect.getsource(
        runner_module.Qwen4ExpGGUFResidentModelRunner
        .verify_target_block_with_acceptance)
    assert "self.verify_target_block_deferred_head(" in source
    assert "self.verify_target_block_serial_exact(" in source
    assert SCOPE not in source


def test_the_prefill_final_head_stays_rows_one() -> None:
    """The final head runs after the per-chunk scope, so it must not need one.

    ``_wmma_prefill_dispatch`` declines ``rows <= 1`` before it looks at the
    quant or at ``use_wmma``, so a one-row launch never resolves the Q8_0 tile
    pair (see ``test_unit_qwen4exp_wmma_engagement_recorder.py``). A multirow
    head here would sit outside the bound scope; pinning the literal makes that
    change fail here instead of silently measuring the arch heuristic.
    """

    tree = ast.parse(textwrap.dedent(inspect.getsource(
        runner_module.Qwen4ExpGGUFResidentModelRunner._prefill_chunked_steps)))
    reads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "run_qwen4_exp_gr_read"
    ]
    launches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "launch_gguf_linear"
    ]
    assert len(reads) == 1 and len(launches) == 1
    rows_kw = [kw.value for kw in reads[0].keywords if kw.arg == "rows"]
    assert len(rows_kw) == 1
    assert ast.literal_eval(rows_kw[0]) == 1
    # ``launch_gguf_linear(weight, x, out, rows, in_features, out_features, ...)``
    assert ast.literal_eval(launches[0].args[3]) == 1


def _resumable_runner() -> Any:
    """A runner stub with only the state ``reset``/``close`` touch."""

    runner = object.__new__(runner_module.Qwen4ExpGGUFResidentModelRunner)
    for name, value in dict(
        closed=False,
        _pending_prefill_append=None,
        _verify_rows_buffers={},
        _verify_rows_capacity=0,
        _pending_verify_block=None,
        _q8_mmq_buffers=(),
        _prefill_workspaces=[],
        _prefill_buffers=[],
        _buffers=[],
        index_states=(),
        attention_states=(),
        _ple_hash_states={},
        position=0,
        moe_graph_cache=None,
        layer_graph_cache=None,
        _target_verify_output=None,
        _device_transaction_snapshot=None,
        _device_transaction_lease=False,
        _q8_mmq_weight_sidecars=None,
        _q8_mmq_policy=None,
        _q8_mmq_library=None,
        _wide_f16_activation_buffer=None,
        _shared_position_context=None,
        _shared_position_context_host=None,
        head_scratch=None,
        ple_scratch=None,
        qsa_scratch=None,
        gdn_scratch=None,
    ).items():
        setattr(runner, name, value)
    runner.runtime = SimpleNamespace()
    runner.state = SimpleNamespace(zero=lambda: None, close=lambda: None)
    return runner


def test_a_suspended_resumable_prefill_keeps_its_pair_and_restores_the_caller() -> None:
    """The real resumable driver, not a toy generator, owns the pair.

    ``prefill_append_resumable`` saves a copied Context and drives the layer
    generator from it. The pair must therefore be the owner's while suspended,
    the caller's while the scheduler runs something else, and gone once the
    pending append is closed - in its owning context, as ``reset`` does.
    """

    runner = _resumable_runner()
    runner.q8_wmma_tiles = QWEN_PAIR
    inside: list[tuple[int, int] | None] = []

    def steps(tokens, **kwargs):
        with runner_module._qwen4_exp_q8_tile_scope(runner):
            for _ in range(2):
                inside.append(q8.current_wmma_tile_policy())
                yield None
            return SimpleNamespace(token_id=7)

    runner._prefill_chunked_steps = steps
    with q8.wmma_tile_policy(16, 16):
        assert runner.prefill_append_resumable(
            (1, 2), time_budget_seconds=1e-9) is None
        # The peer's pair is untouched while this owner is suspended.
        assert q8.current_wmma_tile_policy() == (16, 16)
        assert runner._pending_prefill_append is not None
        # Ownership: binding never writes the published field.
        assert runner.q8_wmma_tiles == QWEN_PAIR
        # ``reset`` closes a pending append through its own Context; the strict
        # token reset would raise here if it ran in the caller's.
        runner.reset()
        assert runner._pending_prefill_append is None
        assert q8.current_wmma_tile_policy() == (16, 16)
        assert inside == [QWEN_PAIR]
    assert q8.current_wmma_tile_policy() is None
    assert runner.q8_wmma_tiles == QWEN_PAIR
    runner.close()
    assert runner.closed is True
    assert runner.q8_wmma_tiles == QWEN_PAIR
