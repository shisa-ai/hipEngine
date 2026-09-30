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


class _SyntheticWeight:
    """Minimal resident-weight shape; only ``spec``/``backend`` are read early."""

    class _Spec:
        layout = "raw_gguf"
        quant_key = "gguf_q8_0"
        slot_path = "layers.20.attn_qkv.weight"

    spec = _Spec()
    backend = "hip_gfx1151"

    @staticmethod
    def allocation(name):
        class _Alloc:
            class tensor:
                ptr = 4096

        return _Alloc()


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


def test_summary_does_not_claim_arm_isolation():
    """A name that promises more than the check delivers is this packet's bug.

    Measured on the canonical single case: the route engaged 264 times at 512
    rows and the post-prefill output was still bit-identical (5.8014e-12), so the
    drift this gate reports comes from the *other* selectors `use_wmma` gates -
    `f_rowtile`, `enabled=not use_wmma`, `use_rowtile`, the rows-2-8 batch-decode
    session. The recorder cannot see any of them, so it must not report that the
    arms are isolated.
    """

    gate = _gate()
    summary = gate._summarise_engagement(
        {
            "teacher": [{"selector": "wmma_prefill", "reason": "use_wmma_false",
                         "engaged": False, "rows": 512}],
            "candidate": [
                {
                    "selector": "wmma_prefill",
                    "reason": "engaged",
                    "engaged": True,
                    "rows": 512,
                    "variant_out": "wmma_prefill_q8_0_f32_f32_out",
                }
            ],
        },
        route_selector="wmma_prefill",
    )
    assert "arms_isolated" not in summary, (
        "the flag was renamed because it promised isolation this check cannot "
        "establish; reintroducing the name reintroduces the misreading"
    )
    assert summary["covers_other_use_wmma_selectors"] is False


def test_recorder_captures_the_other_selectors_use_wmma_moves():
    """`use_wmma` is not a WMMA-route flag.

    It also forces `f_rowtile` off, which drags `use_rowtile` with it, and
    disables the rows-2-8 native batch-decode session. Those are decode
    decisions taken at row counts the WMMA route refuses, which is how a gate
    named `wmma` reports real drift the route did not cause. Recording them is
    what turns "not this route" into a name.
    """

    module = _gguf_linear()
    records: list[dict] = []
    with module.wmma_prefill_dispatch_recorder(records):
        try:
            module.launch_gguf_linear(
                _SyntheticWeight(), 0, 0, 4, 2560, 10240, libraries={}
            )
        except Exception:
            # Synthetic pointers cannot launch. The records are appended before
            # any of that, which is the part under test.
            pass
    selectors = {record["selector"] for record in records}
    assert {"wmma_prefill", "q4k_rowtile", "native_batch_decode_session"} <= selectors, (
        f"only {sorted(selectors)} were recorded; the other use_wmma-gated "
        "selectors are what the gate's drift actually comes from"
    )
    rowtile = [r for r in records if r["selector"] == "q4k_rowtile"]
    assert rowtile, "the rowtile decision was not recorded"
    assert "use_wmma" in rowtile[0]["detail"], (
        "the record must say whether use_wmma is what moved it"
    )


def test_summary_names_the_selectors_that_differ_between_arms():
    """Positive attribution: a drift can only come from a selector that differs.

    The WMMA route engaging is not sufficient - the canonical case engages 264
    times and its post-prefill output is still bit-identical. So the useful
    output is the list of selectors that actually disagree between the arms.
    """

    gate = _gate()
    summary = gate._summarise_engagement(
        {
            "teacher": [
                {"selector": "wmma_prefill", "reason": "use_wmma_false",
                 "engaged": False, "rows": 512},
                {"selector": "q4k_rowtile", "reason": "active",
                 "engaged": True, "rows": 512},
                {"selector": "native_batch_decode_session", "reason": "inactive",
                 "engaged": False, "rows": 4},
            ],
            "candidate": [
                {"selector": "wmma_prefill", "reason": "engaged", "engaged": True,
                 "rows": 512, "variant_out": "wmma_prefill_f32_f32_out"},
                {"selector": "q4k_rowtile", "reason": "inactive",
                 "engaged": False, "rows": 512},
                {"selector": "native_batch_decode_session", "reason": "inactive",
                 "engaged": False, "rows": 4},
            ],
        },
        route_selector="wmma_prefill",
    )
    assert summary["selectors_differing_between_arms"] == [
        "q4k_rowtile",
        "wmma_prefill",
    ]
    assert "q4k_rowtile" in summary["attribution"]
    # A selector that agrees between the arms cannot be the source.
    assert "native_batch_decode_session" not in summary["attribution"]


def test_selector_observed_in_one_arm_only_is_not_called_isolated():
    """Absence of a record is not evidence that a selector agreed.

    Counting an unobserved selector as "not differing" would manufacture the
    same false isolation this whole recorder exists to prevent.
    """

    gate = _gate()
    summary = gate._summarise_engagement(
        {
            "teacher": [
                {"selector": "wmma_prefill", "reason": "use_wmma_false",
                 "engaged": False, "rows": 512},
            ],
            "candidate": [
                {"selector": "wmma_prefill", "reason": "engaged", "engaged": True,
                 "rows": 512, "variant_out": "wmma_prefill_f32_f32_out"},
                {"selector": "q4k_rowtile", "reason": "inactive",
                 "engaged": False, "rows": 512},
            ],
        },
        route_selector="wmma_prefill",
    )
    assert summary["selectors_observed_in_one_arm_only"] == ["q4k_rowtile"]
    assert "q4k_rowtile" not in summary["selectors_differing_between_arms"]


def test_route_engagement_is_about_the_route_not_every_selector():
    """The teacher arm always has q4k_rowtile active, and that is not the route.

    Measured on the canonical case: the teacher arm's wmma_prefill count was
    exactly 0, yet the verdict read "the teacher arm also engaged the route"
    because the flag was derived from total engagement across every selector.
    q4k_rowtile is active in the teacher arm by design, so that derivation made
    the flag unconditionally true and the verdict unconditionally wrong.
    """

    gate = _gate()
    summary = gate._summarise_engagement(
        {
            "teacher": [
                {"selector": "wmma_prefill", "reason": "use_wmma_false",
                 "engaged": False, "rows": 512},
                {"selector": "q4k_rowtile", "reason": "active",
                 "engaged": True, "rows": 512},
            ],
            "candidate": [
                {"selector": "wmma_prefill", "reason": "engaged", "engaged": True,
                 "rows": 512, "variant_out": "wmma_prefill_f32_f32_out"},
                {"selector": "q4k_rowtile", "reason": "active",
                 "engaged": True, "rows": 512},
            ],
        },
        route_selector="wmma_prefill",
    )
    assert summary["teacher_route_engaged"] is False, (
        "the teacher's route count is zero; a true flag here makes the verdict "
        "unconditionally wrong"
    )
    assert summary["candidate_route_engaged"] is True
    assert "teacher arm also engaged" not in summary["verdict"]


def test_differing_selector_is_found_even_when_both_arms_engage_it():
    """A collapsed boolean hides a real difference.

    Measured on the canonical case: q4k_rowtile was active 1654/1654 in the
    teacher arm and 2730/3522 in the candidate arm. Both are "active at least
    once", so a boolean comparison calls them equal and the drift gets attributed
    to a selector that did not move.
    """

    gate = _gate()
    summary = gate._summarise_engagement(
        {
            "teacher": [
                {"selector": "q4k_rowtile", "reason": "active",
                 "engaged": True, "rows": 512} for _ in range(100)
            ]
            + [
                {"selector": "wmma_prefill", "reason": "use_wmma_false",
                 "engaged": False, "rows": 512} for _ in range(100)
            ],
            "candidate": [
                {"selector": "q4k_rowtile", "reason": "active",
                 "engaged": True, "rows": 512} for _ in range(80)
            ]
            + [
                {"selector": "q4k_rowtile", "reason": "inactive",
                 "engaged": False, "rows": 512} for _ in range(20)
            ]
            + [
                {"selector": "wmma_prefill", "reason": "engaged", "engaged": True,
                 "rows": 512, "variant_out": "wmma_prefill_f32_f32_out"}
                for _ in range(100)
            ],
        },
        route_selector="wmma_prefill",
    )
    assert "q4k_rowtile" in summary["selectors_differing_between_arms"], (
        "both arms engaged it at least once, but at different rates; a boolean "
        "comparison would call this isolated"
    )
    state = summary["selector_states"]["q4k_rowtile"]
    assert state["teacher_engaged_rate"] == 1.0
    assert state["candidate_engaged_rate"] == 0.8


def test_summary_flags_a_route_that_never_engaged():
    """The verdict must refuse to attribute drift to a route that did not run."""

    gate = _gate()
    summary = gate._summarise_engagement(
        {
            "teacher": [{"selector": "wmma_prefill", "reason": "use_wmma_false",
                         "engaged": False, "rows": 512}],
            "candidate": [
                {"reason": "raw_abi_not_eligible", "engaged": False, "rows": 512}
            ],
        },
        route_selector="wmma_prefill",
    )
    assert summary["candidate_route_engaged"] is False
    assert summary["wmma_route_engaged_candidate_only"] is False
    assert "must not be attributed" in summary["verdict"]
    assert summary["candidate"]["decline_reasons"] == {"raw_abi_not_eligible": 1}


def test_summary_accepts_a_correctly_isolated_pair():
    gate = _gate()
    summary = gate._summarise_engagement(
        {
            "teacher": [{"selector": "wmma_prefill", "reason": "use_wmma_false",
                         "engaged": False, "rows": 512}],
            "candidate": [
                {
                    "selector": "wmma_prefill",
                    "reason": "engaged",
                    "engaged": True,
                    "rows": 512,
                    "variant_out": "wmma_prefill_q8_0_f32_f32_out",
                }
            ],
        },
        route_selector="wmma_prefill",
    )
    assert summary["candidate_route_engaged"] is True
    assert summary["teacher_route_engaged"] is False
    assert summary["wmma_route_engaged_candidate_only"] is True
    # The flag must not promise arm isolation, because it cannot deliver it.
    assert "arms_isolated" not in summary
    assert summary["covers_other_use_wmma_selectors"] is False
    assert "does NOT mean the arms are isolated" in summary["verdict"]


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
                    "selector": "wmma_prefill",
                    "reason": "engaged",
                    "engaged": True,
                    "rows": 512,
                    "variant_out": "wmma_prefill_q8_0_f32_f32_out",
                }
            ],
            "candidate": [
                {
                    "selector": "wmma_prefill",
                    "reason": "engaged",
                    "engaged": True,
                    "rows": 512,
                    "variant_out": "wmma_prefill_q8_0_f32_f32_out",
                }
            ],
        },
        route_selector="wmma_prefill",
    )
    assert summary["wmma_route_engaged_candidate_only"] is False
    assert "not against the cleared chain" in summary["verdict"]


def test_route_flag_follows_the_named_route_not_wmma():
    """A `--route dense_wide` run must be able to report its own route engaged.

    The flag was hardcoded to `wmma_prefill`, so the route that is the live
    production default at 16-47 reported `candidate_route_engaged: False`
    whatever the recorder saw. It was the one route the false-pass detector
    could not see, and it was the live one.
    """

    gate = _gate()
    engagement = {
        "teacher": [
            {"selector": "dense_wide", "reason": "selector off", "engaged": False,
             "rows": 512},
        ],
        "candidate": [
            {"selector": "dense_wide", "reason": "claimed", "engaged": True,
             "rows": 512, "variant_out": "dense_wide256_f32_f32_out"},
        ],
    }

    dense = gate._summarise_engagement(engagement, route_selector="dense_wide")
    assert dense["route"] == "dense_wide"
    assert dense["candidate_route_engaged"] is True
    assert dense["teacher_route_engaged"] is False
    assert dense["route_engagement_recorded"] is True

    # The same records asked about the WMMA route answer about that route: the
    # wmma selector never appears, so it did not engage. This is the assertion
    # that pins "the flag is about the route you named".
    wmma = gate._summarise_engagement(engagement, route_selector="wmma_prefill")
    assert wmma["candidate_route_engaged"] is False


def test_an_unrecorded_route_says_so_instead_of_borrowing_another():
    """No recorder must read as *unanswered*, not as a pass or a failure."""

    gate = _gate()
    summary = gate._summarise_engagement(
        {
            "teacher": [{"selector": "wmma_prefill", "reason": "x", "engaged": False,
                         "rows": 512}],
            "candidate": [{"selector": "wmma_prefill", "reason": "y", "engaged": True,
                           "rows": 512}],
        },
        route_selector=None,
    )
    assert summary["route_engagement_recorded"] is False
    assert summary["route"] is None
    assert summary["candidate_route_engaged"] is False
    assert "cannot say whether" in summary["route_engagement_note"]
    # It must not answer with another route's selector.
    assert summary["candidate_route_engaged"] != summary["selectors_differing_between_arms"]


def test_every_route_with_a_recorder_is_mapped():
    """The map is the only place a route's selector is named; keep it honest."""

    gate = _gate()
    assert gate.ROUTE_ENGAGEMENT_SELECTORS["wmma"] == "wmma_prefill"
    assert gate.ROUTE_ENGAGEMENT_SELECTORS["dense_wide"] == "dense_wide"
    # Every mapped route must also have a row floor, so the two registries
    # cannot drift apart silently.
    assert set(gate.ROUTE_ENGAGEMENT_SELECTORS) <= set(gate.ROUTE_ROW_FLOORS)
