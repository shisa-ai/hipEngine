"""Route contract for the layer-scoped Q8_0 dense prefill gate.

The gate is the only instrument that decides whether a layer-scoped dense Q8_0
prefill route passes the calibrated production envelope, and it decides it from
what its arms bind. Two properties are load-bearing and were both wrong or
missing on 2026-09-17:

* the ``dense_wide`` arm has to clear the f16 WMMA selector, because the
  production default binds it to the same layer window and the comparison would
  otherwise carry both routes' arithmetic;
* the teacher arm has to be the exact coltile chain, because every certified
  figure for these routes is measured against exact, so the numbers are only
  comparable if the teacher is the same.

These tests bind the route table, not a GPU run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.execution_profile_q8_wmma_prefill_layers_gate import (  # noqa: E402
    DENSE_WIDE_ENV,
    DENSE_WIDE_LAYERS_ENV,
    SELECTORS,
    WMMA_LAYERS_ENV,
    _parse_layers,
    _select_route,
)

LAYERS = "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47"


def test_both_layer_scoped_routes_are_selectable():
    assert set(SELECTORS) == {"wmma", "dense_wide"}
    assert _select_route("wmma") is SELECTORS["wmma"]
    assert _select_route("dense_wide") is SELECTORS["dense_wide"]


def test_unknown_route_is_rejected_rather_than_defaulted():
    from scripts.execution_profile_q8_wmma_prefill_layers_gate import (
        CalibrationError,
    )

    with pytest.raises(CalibrationError):
        _select_route("dense_wide256")


def test_the_wmma_arms_are_unchanged_for_the_certified_figures():
    """The 16-47 certification was produced by these exact bindings."""

    route = SELECTORS["wmma"]

    assert route.env_for(candidate=True, layers=LAYERS) == {WMMA_LAYERS_ENV: LAYERS}
    assert route.env_for(candidate=False, layers=LAYERS) == {WMMA_LAYERS_ENV: ""}
    assert route.candidate_chain == "wmma_prefill_f32_f32_out"


def test_the_dense_wide_arm_clears_the_wmma_selector():
    """Otherwise the window under test runs WMMA and the candidate's own route
    only contends for it, which is what left the wide kernel unreachable."""

    route = SELECTORS["dense_wide"]
    candidate = route.env_for(candidate=True, layers=LAYERS)

    assert candidate == {
        WMMA_LAYERS_ENV: "",
        DENSE_WIDE_ENV: "1",
        DENSE_WIDE_LAYERS_ENV: LAYERS,
    }
    assert route.candidate_chain == "dense_wide256_f32_f32_out"


def test_the_dense_wide_teacher_is_the_exact_chain():
    route = SELECTORS["dense_wide"]
    teacher = route.env_for(candidate=False, layers=LAYERS)

    assert teacher == {
        WMMA_LAYERS_ENV: "",
        DENSE_WIDE_ENV: "0",
        DENSE_WIDE_LAYERS_ENV: "",
    }


def test_restoring_the_route_env_removes_keys_that_were_unset():
    """A gate must not leave the process with a selector it invented."""

    import os

    from scripts.execution_profile_q8_wmma_prefill_layers_gate import (
        _apply_route_env,
        _restore_env,
    )

    route = SELECTORS["dense_wide"]
    bound = {key: os.environ.get(key) for key in route.env_keys}
    assert bound == {WMMA_LAYERS_ENV: None, DENSE_WIDE_ENV: None, DENSE_WIDE_LAYERS_ENV: None}
    try:
        _apply_route_env(route, candidate=True, layers=LAYERS)
        assert os.environ[DENSE_WIDE_ENV] == "1"
    finally:
        _restore_env(bound)

    assert DENSE_WIDE_ENV not in os.environ
    assert DENSE_WIDE_LAYERS_ENV not in os.environ
    assert WMMA_LAYERS_ENV not in os.environ


def test_layers_argument_normalizes_to_the_selector_form():
    assert _parse_layers("16-19") == "16,17,18,19"
    assert _parse_layers("20,16,20") == "16,20"
