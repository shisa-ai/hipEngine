"""The WMMA gate's route must clear the wide route in both arms.

The reciprocal of the clearing the ``dense_wide`` route already does, and its
absence produced a **green result covering a route that never ran**.

The certified plan binds ``HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE`` to 16-47, the same
window this route is gated at, and ``_q8_dense_wide_claims_parent`` accepts the
WMMA-rewritten parent (``wmma_<parent>`` under ABI ``wmma_raw``) as well as the
exact parent. With the wide route left bound it claims the dispatch whichever way
the WMMA selector is set, so both arms resolve to ``dense_wide256_f32_f32_out``:

```
strict_logits_sha256    = 83732d4542e1847fd8975ca6...
candidate_logits_sha256 = 83732d4542e1847fd8975ca6...
```

KL exactly ``0.0`` on every statistic, ``top1 1.0``, and ``status: complete``
with ``measurement_valid: true``. The 2026-09-16 certification of the same route
had a genuine exact-chain teacher (``550bb9b8...``) and mean KL 3.796e-4, so this
was a regression that made the certification non-reproducible at HEAD.

Confirmed by elimination rather than assumed: with ``dense_wide`` bound, the
route's two arms differ only in the WMMA selector and produce identical logits,
so the selector is inert; with ``dense_wide`` cleared (the composed route), the
WMMA arm produces ``87dbf445...`` against an exact-chain teacher of
``550bb9b8...``, so WMMA does change the arithmetic when it is the only route
bound. The hash the broken gate reported, ``83732d45...``, is therefore the wide
route's output.

Evidence: ``benchmarks/results/2026-09-29-q8-staged-f32-dense-prefill-layers-gate/WMMA-ROUTE-FALSE-PASS.md``.
"""

import importlib.util
import pathlib
import sys

import pytest

GATE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "execution_profile_q8_wmma_prefill_layers_gate.py"
)


def _gate():
    if not GATE.exists():
        pytest.skip(f"gate script not present: {GATE}")
    spec = importlib.util.spec_from_file_location("_wmma_gate_under_test", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wmma_route_clears_the_wide_route_in_both_arms():
    """Both arms, because clearing one only moves which arm is mislabelled."""

    gate = _gate()
    route = gate.SELECTORS["wmma"]
    for arm in (True, False):
        env = route.env_for(candidate=arm, layers="16-47")
        assert env.get(gate.DENSE_WIDE_ENV) == "0", (
            f"the {'candidate' if arm else 'teacher'} arm leaves "
            f"{gate.DENSE_WIDE_ENV} bound, so the wide route claims the parent "
            "and this arm measures it instead of the WMMA route"
        )
        assert env.get(gate.DENSE_WIDE_LAYERS_ENV) == "", (
            "the wide route's layer scope is left bound; clearing only the "
            "enable flag is not enough if anything reads the scope back"
        )


def test_wmma_route_arms_differ_only_in_the_wmma_selector():
    """The measured drift must be the WMMA route's own contribution.

    If the arms differed in more than the selector, the KL would be a
    composition's drift attributed to one route.
    """

    gate = _gate()
    route = gate.SELECTORS["wmma"]
    candidate = route.env_for(candidate=True, layers="16-47")
    teacher = route.env_for(candidate=False, layers="16-47")
    assert candidate != teacher, (
        "the two arms are identical, so the run measures one configuration "
        "against itself and reports a KL of exactly zero as a pass"
    )
    differing = {
        key
        for key in set(candidate) | set(teacher)
        if candidate.get(key) != teacher.get(key)
    }
    assert differing == {gate.WMMA_LAYERS_ENV}, (
        f"the arms differ in {sorted(differing)}, not only "
        f"{gate.WMMA_LAYERS_ENV}; the drift would not be attributable to this "
        "route alone"
    )
    assert candidate[gate.WMMA_LAYERS_ENV] == "16-47"
    assert teacher[gate.WMMA_LAYERS_ENV] == ""
