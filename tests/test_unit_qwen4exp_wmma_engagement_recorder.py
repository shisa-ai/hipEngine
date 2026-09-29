"""Engagement must be recorded at the dispatch, not inferred from a KL.

Four times in one packet a measurement was attributed to a route that had not
engaged, and every one survived because nothing recorded which kernel ran:

- two bit-identical candidate/teacher false passes (the staged-f32 and wmma
  cache-key incidents), where KL was exactly 0.0 and the status was green;
- a `category_heldout_qualification_claim` asserted on a run that failed;
- the `wmma` gate itself, whose post-prefill output is identical to f32
  precision (5.8014e-12 on an isolated case) while its entire drift sits on the
  decode path - so its recorded `candidate_chain` of `wmma_prefill_f32_f32_out`
  describes a route that contributed nothing measurable.

A nonzero KL cannot distinguish "the route ran and moved the output" from
"another selector moved the output". Recording the decision at the dispatch can.
"""

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "execution_profile_q8_wmma_prefill_layers_gate.py"


def _gguf_linear():
    try:
        from hipengine.runtime import gguf_linear
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"hipengine runtime unavailable: {exc}")
    return gguf_linear


def _gate():
    if not GATE.exists():
        pytest.skip(f"gate script not present: {GATE}")
    spec = importlib.util.spec_from_file_location("_wmma_gate_engagement", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dispatch(variant, abi):
    from hipengine.kernels.registry import KernelKey

    module = _gguf_linear()
    return module.GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0", variant), abi
    )


def _decide(variant, abi, *, rows, use_wmma, in_features=2560):
    module = _gguf_linear()
    records: list[dict] = []
    with module.wmma_prefill_dispatch_recorder(records):
        out = module._wmma_prefill_dispatch(
            _dispatch(variant, abi),
            rows=rows,
            in_features=in_features,
            use_wmma=use_wmma,
        )
    assert len(records) == 1, "every dispatch decision must record exactly once"
    return out, records[0]


def test_recorder_is_inert_when_not_installed():
    """The hot path must not accumulate records nobody asked for."""

    module = _gguf_linear()
    assert module._wmma_prefill_dispatch_records.get() is None
    module._wmma_prefill_dispatch(
        _dispatch("prefill_q8_0_f32_f32_out", "raw"),
        rows=512,
        in_features=2560,
        use_wmma=True,
    )
    assert module._wmma_prefill_dispatch_records.get() is None


def test_engaged_rewrite_is_recorded_with_the_outgoing_variant():
    out, record = _decide("prefill_q8_0_f32_f32_out", "raw", rows=512, use_wmma=True)
    assert record["engaged"] is True
    assert record["reason"] == "engaged"
    assert record["variant_out"] == out.key.variant
    assert record["variant_out"].startswith("wmma_")


def test_the_decline_that_was_invisible_is_now_named():
    """The exact leaf fails the predicate at every row count.

    `_dispatch_can_use_wmma_prefill` requires a `prefill_` variant. The exact
    coltile leaf never has one, so the route declines - and before this recorder
    existed nothing in the artifact distinguished that from "the route ran and
    the output happened not to move".
    """

    for rows, expected in (
        (1, "rows_le_1"),
        (64, "raw_abi_not_eligible"),
        (512, "raw_abi_not_eligible"),
        (1024, "raw_abi_not_eligible"),
    ):
        out, record = _decide(
            "coltile8_rowbatch4_f32_f32_out", "raw", rows=rows, use_wmma=True
        )
        assert record["engaged"] is False
        # The order of the guards is itself information: at rows == 1 the row
        # check fires before the variant predicate, so the recorded reason names
        # the condition that actually declined it.
        assert record["reason"] == expected
        assert out.key.variant == "coltile8_rowbatch4_f32_f32_out"


def test_each_decline_condition_reports_itself():
    """A single generic "declined" reason would not localize anything."""

    cases = {
        "use_wmma_false": dict(
            variant="prefill_q8_0_f32_f32_out", abi="raw", rows=512, use_wmma=False
        ),
        "rows_le_1": dict(
            variant="prefill_q8_0_f32_f32_out", abi="raw", rows=1, use_wmma=True
        ),
        "raw_abi_not_eligible": dict(
            variant="gemv_q8_0_f32_f32_out", abi="raw", rows=512, use_wmma=True
        ),
        "t16_abi_not_eligible": dict(
            variant="prefill_q8_0_f32_f32_out", abi="t16", rows=512, use_wmma=True
        ),
    }
    for expected, kwargs in cases.items():
        _, record = _decide(**kwargs)
        assert record["reason"] == expected, (
            f"{kwargs['variant']}/{kwargs['abi']}/rows={kwargs['rows']} reported "
            f"{record['reason']!r}, not {expected!r}"
        )
        assert record["engaged"] is False


def test_summary_flags_a_route_that_never_engaged():
    """The verdict must refuse to attribute drift to a route that did not run."""

    gate = _gate()
    summary = gate._summarise_engagement(
        {
            "teacher": [{"reason": "use_wmma_false", "engaged": False, "rows": 512}],
            "candidate": [
                {"reason": "raw_abi_not_eligible", "engaged": False, "rows": 512}
            ],
        }
    )
    assert summary["candidate_route_engaged"] is False
    assert summary["arms_isolated"] is False
    assert "must not be attributed" in summary["verdict"]
    assert summary["candidate"]["decline_reasons"] == {"raw_abi_not_eligible": 1}


def test_summary_accepts_a_correctly_isolated_pair():
    gate = _gate()
    summary = gate._summarise_engagement(
        {
            "teacher": [{"reason": "use_wmma_false", "engaged": False, "rows": 512}],
            "candidate": [
                {
                    "reason": "engaged",
                    "engaged": True,
                    "rows": 512,
                    "variant_out": "wmma_prefill_q8_0_f32_f32_out",
                }
            ],
        }
    )
    assert summary["candidate_route_engaged"] is True
    assert summary["teacher_route_engaged"] is False
    assert summary["arms_isolated"] is True
    assert summary["verdict"] == "candidate engaged and teacher cleared"


def test_summary_rejects_a_teacher_arm_that_also_engaged():
    """The teacher is supposed to be the cleared arm.

    If it engages, the comparison is not against the exact chain and the run is
    invalid however good its KL looks.
    """

    gate = _gate()
    summary = gate._summarise_engagement(
        {
            "teacher": [
                {
                    "reason": "engaged",
                    "engaged": True,
                    "rows": 512,
                    "variant_out": "wmma_prefill_q8_0_f32_f32_out",
                }
            ],
            "candidate": [
                {
                    "reason": "engaged",
                    "engaged": True,
                    "rows": 512,
                    "variant_out": "wmma_prefill_q8_0_f32_f32_out",
                }
            ],
        }
    )
    assert summary["arms_isolated"] is False
    assert "not against the cleared chain" in summary["verdict"]
