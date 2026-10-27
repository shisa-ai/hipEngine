"""R0254 cleanup contract: generic owners for exact ordered F32 prefill.

The ordered-F32 raw-K prefill route is admitted per quant through
``GGUF_F32_ORDERED_PREFILL_QUANTS`` (``gguf_q5_k`` and ``gguf_q6_k``), and both
quants reach the same weight plane, the same session, and the same Laguna
scratch allocation. The owner names therefore carry no quant label:
``F32OrderedPrefillSession``, ``f32_ordered_prefill_session``,
``LagunaF32OrderedScratch``, ``LagunaPrefillScratchPlan.f32_ordered_nbytes``,
and the runner's ``f32_ordered_dispatch`` / ``f32_ordered_scratch`` bindings.

Q5 stays where the thing is Q5-only: ``Q5F32ResidentPlane`` and the resident
global cache, the ``use_q5_activation_tile_k_row`` and resident-cache flags, the
``LagunaEagerLibraries.q5_f32_ordered`` library built by
``build_gguf_q5_k_f32_rocblas_prefill``, and the per-quant
``q5_k_f32_ordered_workspace_nbytes`` / ``q6_k_f32_ordered_workspace_nbytes``
pair.
"""

from __future__ import annotations

import contextvars
import dataclasses
import inspect

from hipengine.runtime import gguf_linear
from hipengine.runtime import laguna_gguf_runner as runner

_RETIRED = (
    "Q5F32OrderedPrefillSession",
    "LagunaQ5F32OrderedScratch",
    "q5_f32_ordered_prefill_session",
    "q5_f32_ordered_session",
    "q5_f32_ordered_scratch",
    "use_q5_f32_ordered",
    "q5_f32_ordered_nbytes",
    "q5_f32_ordered_dispatch",
)


def test_ordered_f32_prefill_session_has_exactly_one_contextvar_owner() -> None:
    owners = {
        name: value
        for name, value in vars(gguf_linear).items()
        if isinstance(value, contextvars.ContextVar)
        and "f32_ordered_prefill" in str(value.name)
    }
    assert list(owners) == ["_f32_ordered_prefill_session"]
    owner = owners["_f32_ordered_prefill_session"]
    assert str(owner.name) == "f32_ordered_prefill_session"

    session = gguf_linear.F32OrderedPrefillSession(
        max_rows=512,
        weight_f32_ptr=0x1000,
        weight_f32_nbytes=4096,
        library=object(),
    )
    assert owner.get() is None
    with gguf_linear.f32_ordered_prefill_session(session):
        # The public manager writes the one owner var; no shadow ContextVar
        # can carry a second session for the same route.
        assert owner.get() is session
    assert owner.get() is None


def test_retired_quant_labelled_owner_names_are_not_exported() -> None:
    for name in _RETIRED:
        assert not hasattr(gguf_linear, name)
        assert not hasattr(runner, name)
        assert name not in gguf_linear.__all__
        assert name not in runner.__all__
    assert "f32_ordered_prefill_session" in gguf_linear.__all__
    assert "F32OrderedPrefillSession" in gguf_linear.__all__
    assert "LagunaF32OrderedScratch" in runner.__all__


def test_runner_binds_the_generic_ordered_f32_owners() -> None:
    init_source = inspect.getsource(runner.LagunaGGUFResidentSession.__init__)
    close_source = inspect.getsource(runner.LagunaGGUFResidentSession._close)
    assert "self.f32_ordered_dispatch" in init_source
    assert "self.f32_ordered_scratch" in init_source
    assert "LagunaF32OrderedScratch.allocate(" in init_source
    assert "f32_scratch.weight_f32_ptr" in init_source
    assert "self.f32_ordered_scratch" in close_source
    for retired in (
        "self.q5_f32_ordered_dispatch",
        "self.q5_f32_ordered_scratch",
        "q5_scratch.",
    ):
        assert retired not in init_source
        assert retired not in close_source


def test_scratch_plan_names_ordered_f32_bytes_once() -> None:
    fields = [
        field.name for field in dataclasses.fields(runner.LagunaPrefillScratchPlan)
    ]
    assert fields.count("f32_ordered_nbytes") == 1
    assert not [name for name in fields if name.startswith("q5_f32_ordered")]

    parameters = inspect.signature(runner.LagunaPrefillScratchPlan.build).parameters
    assert "use_f32_ordered" in parameters
    assert "use_q5_f32_ordered" not in parameters

    plan = runner.LagunaPrefillScratchPlan(
        matrix_rows=1,
        attention_rows=1,
        rows_nbytes=2,
        moe_nbytes=3,
        f32_ordered_nbytes=5,
    )
    assert plan.total_nbytes == 10


def test_q5_only_policy_and_loader_names_stay_quant_labelled() -> None:
    library_fields = {
        field.name for field in dataclasses.fields(runner.LagunaEagerLibraries)
    }
    assert "q5_f32_ordered" in library_fields

    parameters = inspect.signature(runner.LagunaPrefillScratchPlan.build).parameters
    assert "use_q5_activation_tile_k_row" in parameters

    session_fields = {
        field.name
        for field in dataclasses.fields(gguf_linear.F32OrderedPrefillSession)
    }
    assert "resident_weight_f32_planes" in session_fields
    assert hasattr(gguf_linear, "Q5F32ResidentPlane")
    assert hasattr(runner, "LagunaQ5F32ResidentGlobalCache")
