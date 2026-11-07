"""GR-up wide tile ownership, shape guards, and strict fallback contracts.

``HIPENGINE_QWEN4_EXP_GR_WIDE`` was the launch-time handoff for the F16 route
composition's GR-up wide tile. The resolved plan's
own manifest selection now publishes the decision on the owning runner
(``gr_wide``), the dispatch reads that field, and the diagnostic harnesses bind
it with ``gr_wide(owner, enabled)``.

These tests pin three separate contracts:

- the shape guards stay at the consumer, so a published route never widens
  below ``rows > 256`` or off the ``low_rank % 64 == 0`` alignment;
- the publication is the plan's, so a stale process key cannot select the route
  and a second bind in the same process publishes its own decision;
- the diagnostic scope restores the prior value on unwind, including after an
  exception, and never touches another owner.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile,
    clear_runtime_profile_registry_for_tests,
    resolve_runtime_profile,
)
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.generation.qwen4_exp_profiles import (
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    QWEN4_EXP_QUANTS,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime import qwen4_exp_runner as module

ENV = "HIPENGINE_QWEN4_EXP_GR_WIDE"
GR_SCOPE = "prefill_rows_ge257_branch4_k_multiple64_gr_up_projection"
WIDE = "dense_wide256_f32_f32_out"
PARENT = "coltile8_rowbatch4_f32_f32_out"
#: The UD-Q4_K_XL production plan carries the F16 route composition.
F16_QUANT = "gguf_ud_q4_k_xl"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV, raising=False)
    clear_runtime_profile_registry_for_tests()
    yield
    clear_runtime_profile_registry_for_tests()


def _resolve(profile: ExecutionProfile, *, quant: str = F16_QUANT):
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=quant,
        profile=profile,
    )


def _selection(resolved, variant: str):
    """Return the plan with its GR-up projection row forced to ``variant``."""

    from dataclasses import replace

    manifest = dict(resolved.manifest)
    manifest["selections"] = [
        dict(row, selected_variant=variant) if row["scope"] == GR_SCOPE else row
        for row in resolved.manifest["selections"]
    ]
    return replace(resolved, manifest=manifest)


# --------------------------------------------------------------------------
# Shape guards and the published choice
# --------------------------------------------------------------------------


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize(
    ("rows", "low_rank"),
    [
        (64, 320),    # decode/batch width
        (256, 320),   # the row threshold itself
        (257, 320),   # one row above it
        (1024, 320),  # canonical depth
        (1024, 384),  # a different 64-aligned rank
        (1024, 300),  # not 64-aligned
        (1024, 64),   # the narrowest 64-aligned rank
    ],
)
def test_selector_keeps_shape_guards_and_the_published_choice(
    published, rows, low_rank
):
    expected = bool(published and rows > 256 and low_rank % 64 == 0)
    assert module._qwen4_exp_gr_wide_enabled(
        rows, low_rank, published=published) is expected


def test_stale_process_key_cannot_select_the_route(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert module._qwen4_exp_gr_wide_enabled(1024, 320) is False
    assert module._qwen4_exp_gr_wide_enabled(
        1024, 320, published=False) is False


def test_runner_module_has_no_reader_for_the_retired_key():
    """The variable must be dead, not merely shadowed by a published field."""

    source = Path(inspect.getfile(module)).read_text()
    assert ENV not in source


# --------------------------------------------------------------------------
# Profile construction
# --------------------------------------------------------------------------


def test_production_bind_publishes_the_wide_route_from_the_manifest():
    resolved = _resolve(ExecutionProfile.PRODUCTION)
    row = next(
        row for row in resolved.manifest["selections"] if row["scope"] == GR_SCOPE)
    assert row["selected_variant"] == WIDE
    assert row["strict_fallback_variant"] == PARENT

    owner = SimpleNamespace(runner=SimpleNamespace())
    resolved.binder(owner, resolved)
    assert owner.gr_wide is True
    assert owner.runner.gr_wide is True
    # The retired process key is not written, so nothing downstream can read it.
    assert ENV not in os.environ


def test_bind_without_the_wide_row_publishes_the_exact_parent():
    resolved = _resolve(ExecutionProfile.PRODUCTION)
    fallback = _selection(resolved, PARENT)
    owner = SimpleNamespace(runner=SimpleNamespace())
    fallback.binder(owner, fallback)
    assert owner.gr_wide is False
    assert owner.runner.gr_wide is False


def test_strict_bind_publishes_the_exact_parent():
    resolved = _resolve(ExecutionProfile.STRICT)
    row = next(
        row for row in resolved.manifest["selections"] if row["scope"] == GR_SCOPE)
    assert row["selected_variant"] == PARENT
    owner = SimpleNamespace(runner=SimpleNamespace())
    resolved.binder(owner, resolved)
    assert owner.gr_wide is False
    assert owner.runner.gr_wide is False


def test_a_later_bind_restores_its_own_decision():
    """Profile precedence: each bind publishes its plan, never the last one's."""

    production = _resolve(ExecutionProfile.PRODUCTION)
    strict = _resolve(ExecutionProfile.STRICT)
    first = SimpleNamespace(runner=SimpleNamespace())
    second = SimpleNamespace(runner=SimpleNamespace())
    production.binder(first, production)
    strict.binder(second, strict)
    assert first.gr_wide is True
    assert second.gr_wide is False
    # Re-binding the first owner keeps the second owner's own decision.
    production.binder(second, production)
    assert second.gr_wide is True
    assert first.gr_wide is True


def test_binder_records_a_discarded_prebinder_value(monkeypatch):
    from hipengine.generation.qwen4_exp_profiles import profile_ignored_overrides

    resolved = _resolve(ExecutionProfile.STRICT)
    monkeypatch.setenv(ENV, "1")
    owner = SimpleNamespace(runner=SimpleNamespace())
    resolved.binder(owner, resolved)
    assert profile_ignored_overrides(owner)[ENV] == ("1", "0")
    assert owner.gr_wide is False


# --------------------------------------------------------------------------
# Diagnostic scope
# --------------------------------------------------------------------------


def test_diagnostic_scope_restores_nested_failure_and_isolates_other_owner():
    owner = SimpleNamespace(gr_wide=True)
    other = SimpleNamespace(gr_wide=True)
    with pytest.raises(RuntimeError, match="failed"):
        with module.gr_wide(owner, False):
            with module.gr_wide(owner, True):
                assert owner.gr_wide is True
            assert owner.gr_wide is False
            assert other.gr_wide is True
            raise RuntimeError("failed")
    assert owner.gr_wide is True
    assert other.gr_wide is True


def test_diagnostic_scope_binds_a_generator_and_its_current_runner():
    generator = SimpleNamespace(gr_wide=False, runner=SimpleNamespace(gr_wide=False))
    with module.gr_wide(generator, True):
        assert generator.gr_wide is True
        assert generator.runner.gr_wide is True
    assert generator.gr_wide is False
    assert generator.runner.gr_wide is False


def test_diagnostic_scope_restores_a_runner_built_inside_it():
    """A runner the owner builds during the scope inherits and then loses it.

    The binder, the constructor and clone inheritance all copy the generator's
    field, so a runner that appears mid-scope starts on the arm's route. Leaving
    it there would keep the arm's decision on the owner that serves later rows.
    """

    generator = SimpleNamespace(gr_wide=False, runner=SimpleNamespace(gr_wide=False))
    with module.gr_wide(generator, True):
        generator.runner = SimpleNamespace(gr_wide=True)
    assert generator.gr_wide is False
    assert generator.runner.gr_wide is False


def test_diagnostic_scope_restores_a_runner_swapped_in_and_raised():
    generator = SimpleNamespace(gr_wide=False, runner=SimpleNamespace(gr_wide=False))
    with pytest.raises(RuntimeError, match="failed"):
        with module.gr_wide(generator, True):
            generator.runner = SimpleNamespace(gr_wide=True)
            raise RuntimeError("failed")
    assert generator.gr_wide is False
    assert generator.runner.gr_wide is False


def test_diagnostic_scope_reaches_the_consumer_decision(monkeypatch):
    """A bound scope, not a process key, is what the dispatch reads."""

    monkeypatch.setenv(ENV, "1")
    owner = SimpleNamespace(gr_wide=False)
    with module.gr_wide(owner, True):
        assert module._qwen4_exp_gr_wide_enabled(
            1024, 320, published=owner.gr_wide) is True
    assert module._qwen4_exp_gr_wide_enabled(
        1024, 320, published=owner.gr_wide) is False


# --------------------------------------------------------------------------
# Owner transport
# --------------------------------------------------------------------------


@pytest.mark.parametrize("published", [False, True])
def test_late_clone_inherits_choice(published):
    clone = _inherit_lane_decisions(
        SimpleNamespace(gr_wide=published), SimpleNamespace())
    assert clone.gr_wide is published


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("name", [
    "run_qwen4_exp_dense_qsa_layer", "run_qwen4_exp_qsa_prefill_layer",
    "run_qwen4_exp_gdn_layer",
])
def test_layer_forwards_owner_choice_to_gr(monkeypatch, published, name):
    runtime = object()
    gr = SimpleNamespace(norm_weight_ptr=1, down=object(), up=object(), inject=None)
    weights = SimpleNamespace(attention_gr=gr, ffn_gr=gr)
    scratch = SimpleNamespace(closed=False, runtime=runtime, attention_gr=object())
    seen = []

    def read(*args, **kwargs):
        seen.append(kwargs["gr_wide"])
        raise RuntimeError("observed GR")

    monkeypatch.setattr(module, "run_qwen4_exp_gr_read", read)
    layer = getattr(module, name)
    kwargs = {key: 1 for key, parameter in inspect.signature(layer).parameters.items()
              if parameter.default is inspect.Parameter.empty}
    kwargs.update(weights=weights, scratch=scratch, runtime=runtime,
                  rows=389, gr_wide=published)
    with pytest.raises(RuntimeError, match="observed GR"):
        layer(**kwargs)
    assert seen == [published]


def test_runner_class_declares_the_unbound_default():
    """An unbound runner keeps the fused exact parent, like its siblings."""

    runner = module.Qwen4ExpGGUFResidentModelRunner
    assert runner.__annotations__["gr_wide"] == "bool"
    assert runner.gr_wide is False


def _load_depth_gate():
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "qwen4exp_q8_repair_depth_gate",
        root / "scripts" / "qwen4exp_q8_repair_depth_gate.py",
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_depth_gate_binds_the_arm_route_and_restores_the_plan():
    """The canonical depth gate measures the arm's route, not the plan's.

    It applies ``CandidateSpec.environment`` and binds the owner fields the arm
    names; the GR-up wide tile is one of those fields now, so the
    ``production_gr_up_wide`` arm has to reach it and the plan's route has to
    come back for the next arm.
    """

    import contextlib

    gate = _load_depth_gate()
    assert gate.CANDIDATES["production_gr_up_wide"].gr_wide is True
    generator = SimpleNamespace(
        gr_wide=False, runner=SimpleNamespace(gr_wide=False))
    spec = SimpleNamespace(gr_wide=True)
    with contextlib.ExitStack() as stack:
        assert gate.bind_candidate_gr_wide(stack, generator, spec) is True
        assert generator.runner.gr_wide is True
    assert generator.runner.gr_wide is False

    unbound = SimpleNamespace(gr_wide=False)
    with contextlib.ExitStack() as stack:
        assert gate.bind_candidate_gr_wide(
            stack, SimpleNamespace(runner=unbound),
            SimpleNamespace(gr_wide=None)) is False
        assert unbound.gr_wide is False
    assert unbound.gr_wide is False
