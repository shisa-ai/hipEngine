"""R0146: the superseded grouped-MoE prefill switch is gone.

``HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT`` is the exact, own-selector layer-2 Q5_K
route (``benchmarks/results/2026-09-09-framework-qwen4exp-q5k-iu8-*``), and the
certified WMMA-MoE27 transport is ``HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL``
with its capability layer list. The grouped-WMMA Q5_K candidate those routes
replaced failed its complete profile gate
(``benchmarks/results/2026-08-31-...p1-layer2-grouped-profile-rejected.json``:
``status: rejected_correctness_default_off``, binding ``prefill_last`` /
``prefill_to_c1`` mean KL 0.0011791 > 0.001, ``next`` = pursue a new exact/T1
Q5_K dataflow), so its selector is removed rather than left default-off.

These are source-level assertions because the selector's only effect was to
enter the grouped branch inside ``run_qwen4_exp_moe``; the branch itself stays
covered by the device tests.
"""

from __future__ import annotations

import inspect

from hipengine.runtime import qwen4_exp_runner

RETIRED = "HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL"


def _grouped_selection_expression(source: str) -> str:
    """Return the ``grouped_prefill = (...)`` expression text."""

    start = source.index("grouped_prefill = (")
    depth = 0
    for offset, character in enumerate(source[start:], start):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return source[start:offset + 1]
    raise AssertionError("unterminated grouped_prefill expression")


def test_runner_no_longer_reads_the_retired_grouped_moe_prefill_switch() -> None:
    source = inspect.getsource(qwen4_exp_runner.run_qwen4_exp_moe)

    assert RETIRED not in source
    assert "GROUPED_MOE_PREFILL" not in source


def test_grouped_selection_is_not_an_environment_switch() -> None:
    """The grouped branch is selected by route predicates and state, not env."""

    grouped = _grouped_selection_expression(
        inspect.getsource(qwen4_exp_runner.run_qwen4_exp_moe)
    )

    assert "os.environ" not in grouped
    # The exact Q5_K owner and the certified WMMA-MoE27 transport remain.
    assert "q5_k_iu8_exact" in grouped
    assert "production_grouped_moe" in grouped
    assert "exact_grouped_q4_gate" in grouped
    assert "exact_grouped_down" in grouped
    assert "routed_f16" in grouped
