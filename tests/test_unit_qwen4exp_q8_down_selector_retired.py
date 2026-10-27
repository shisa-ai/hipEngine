"""The two legacy Q8_0 expert-down group selectors are retired.

``HIPENGINE_QWEN4_EXP_Q8_0_GROUPED`` selected the device-driven grouped Q8_0
down owner (``gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out``) and
``HIPENGINE_QWEN4_EXP_Q8_0_GROUPED_WMMA`` selected a per-expert D2H copy plus a
Python loop over the experts calling the shared dense
``gguf_q8_0_wmma_prefill_bf16_bf16_out``. Both sat below the shipped Q8_0 down
owners (``HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN`` and the fork-b grouped
runner field). The engine no longer exposes these legacy selectors.

The engine branches are gone. What remains is the registered kernel (still
resolved by the Gemma4 grouped-prefill expert route and called directly by the
grouped-down microbench, the rocprof replay, and the kernel-level GPU tests),
the shared dense owner, and the strict per-expert selected gemv the chain now
falls through to.

These tests pin the retirement and the surviving strict fallback. They make no
performance or numerical claim.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap

import pytest

from hipengine.runtime import qwen4_exp_runner

RETIRED_WMMA = "HIPENGINE_QWEN4_EXP_Q8_0_GROUPED_WMMA"
RETIRED_GROUPED = "HIPENGINE_QWEN4_EXP_Q8_0_GROUPED"
RETIRED_KEYS = (RETIRED_GROUPED, RETIRED_WMMA)
ROOT = Path(__file__).resolve().parents[1]

#: Direct consumers of the grouped kernel that must keep working without the
#: engine branch: the two evidence harnesses and the kernel-level GPU tests.
DIRECT_DIAGNOSTICS = (
    ROOT / "scripts" / "gguf_q8_0_grouped_down_microbench.py",
    ROOT / "scripts" / "gguf_q8_0_grouped_down_rocprof.py",
    ROOT / "tests" / "test_gpu_qwen4_exp_q8_0_grouped_down.py",
)


def _moe_source() -> str:
    return inspect.getsource(qwen4_exp_runner.run_qwen4_exp_moe)


def _code_constants(code: object) -> set[str]:
    """Every string constant in a code object and its nested code objects."""

    found: set[str] = set()
    for value in code.co_consts:  # type: ignore[attr-defined]
        if isinstance(value, str):
            found.add(value)
        elif inspect.iscode(value):
            found |= _code_constants(value)
    return found


def _q8_down_chain() -> ast.If:
    """The owner chain inside ``elif weights["expert_down"]... == "gguf_q8_0"``.

    The outer node is the ``elif`` itself; its body starts with the chain of
    down-owner tests whose terminal ``else`` must be the strict fallback.
    """

    tree = ast.parse(textwrap.dedent(_moe_source()))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if (
            'expert_down' in test
            and 'gguf_q8_0' in test
            and node.body
            and isinstance(node.body[0], ast.If)
        ):
            return node.body[0]
    raise AssertionError("Q8_0 expert-down dispatch chain not found")


def _chain_arms(node: ast.If) -> tuple[list[str], ast.If]:
    """Return ``(arm tests in order, terminal node holding the final else)``."""

    tests: list[str] = []
    current = node
    while True:
        tests.append(ast.unparse(current.test))
        nested = [
            statement
            for statement in current.orelse
            if isinstance(statement, ast.If)
        ]
        if nested:
            current = nested[0]
            continue
        return tests, current


# --- retirement -------------------------------------------------------------


@pytest.mark.parametrize("key", RETIRED_KEYS)
def test_the_grouped_prefill_dispatch_no_longer_reads_either_key(key: str) -> None:
    assert key not in _moe_source()


def test_the_family_prefix_is_gone_from_the_dispatch() -> None:
    """No ``Q8_0_GROUPED`` spelling survives, so no prefix match can select."""

    assert "Q8_0_GROUPED" not in _moe_source()


@pytest.mark.parametrize("key", RETIRED_KEYS)
def test_the_compiled_dispatch_carries_no_such_constant(key: str) -> None:
    """Setting the key cannot select a route: the constant is not in the code.

    This is the runtime-level counterpart to the source assertions. A nested
    closure or a code object built from a template would still carry the
    literal, so the scan walks nested code objects too.
    """

    assert key not in _code_constants(qwen4_exp_runner.run_qwen4_exp_moe.__code__)


def test_the_runner_module_has_no_reader_of_either_key() -> None:
    source = inspect.getsource(qwen4_exp_runner)

    for key in RETIRED_KEYS:
        assert key not in source


# --- the surviving chain and its strict fallback ----------------------------


def test_the_q8_down_chain_keeps_only_the_shipped_owners() -> None:
    tests, _ = _chain_arms(_q8_down_chain())

    assert len(tests) == 2, tests
    assert "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN" in tests[0]
    assert "forkb_grouped_down" in tests[1]
    for key in RETIRED_KEYS:
        assert all(key not in test for test in tests)


def test_the_q8_down_chain_ends_at_the_strict_selected_projection() -> None:
    """The regression the P1 restructure once caused: no unwritten down output.

    With every shipped Q8_0 down owner off, the chain must reach the strict
    per-expert selected gemv for ``expert_down`` rather than fall off the end
    of the branch.
    """

    _, terminal = _chain_arms(_q8_down_chain())

    assert terminal.orelse, "the Q8_0 down chain has no terminal else"
    calls = [
        node
        for statement in terminal.orelse
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    ]
    selected = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "selected_projection"
    ]
    assert selected, "the terminal else does not call selected_projection"
    assert ast.unparse(selected[-1].args[0]) == "'expert_down'"


# --- what the retirement preserves -----------------------------------------


def test_the_registered_q8_kernels_survive() -> None:
    """The branch goes; the kernels stay for Gemma4 and the direct harnesses."""

    # The kernel module self-registers its wrappers on import.
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_prefill  # noqa: F401
    from hipengine.kernels.registry import KernelKey, is_registered

    for layer, variant in (
        # MoE expert route, grouped owner (Gemma4 _GROUPED_PREFILL_VARIANTS).
        ("moe_linear", "selected_grouped_prefill_compact_bf16_bf16_out"),
        # MoE expert route, grouped WMMA owner.
        ("moe_linear", "selected_grouped_wmma_prefill_compact_bf16_bf16_out"),
        # Dense owner the retired D2H loop called.
        ("linear", "wmma_prefill_bf16_bf16_out"),
        ("linear", "selected_grouped_wmma_prefill_bf16_bf16_out"),
    ):
        key = KernelKey("hip_gfx1100", layer, "gguf_q8_0", variant)
        assert is_registered(key), (layer, variant)


@pytest.mark.parametrize("path", DIRECT_DIAGNOSTICS, ids=lambda p: p.name)
def test_the_direct_diagnostic_consumer_still_names_the_kernel(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    grouped = "gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out"
    dense = "gguf_q8_0_wmma_prefill_bf16_bf16_out"

    assert grouped in source or dense in source, path


def test_no_profile_pin_or_preemptor_entry_remains() -> None:
    profiles = (ROOT / "hipengine" / "generation" / "qwen4_exp_profiles.py")
    campaign = ROOT / "scripts" / "qwen4exp_halo_box_campaign_ab.py"

    assert "Q8_0_GROUPED" not in profiles.read_text(encoding="utf-8")
    assert "Q8_0_GROUPED" not in campaign.read_text(encoding="utf-8")
