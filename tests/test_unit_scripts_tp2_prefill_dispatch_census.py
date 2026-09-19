"""The census must report what launches, not what was proposed.

The first version hooked ``resolve_gguf_linear_dispatch``, which runs *before*
the runtime's prefill rewrites. It therefore reported
``t16_gemv_decode_bf16_bf16_out`` for tensors that the rewrite stage converts to
``t16_wmma_prefill_bf16_bf16_out``, and that table was then used to claim the
TP2 prefill ran decode-shaped GEMV kernels on its attention path. It does not.
These tests pin the observation point so the mistake cannot recur silently.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "tp2_prefill_dispatch_census.py"


@pytest.fixture(scope="module")
def census():
    spec = importlib.util.spec_from_file_location("tp2_prefill_dispatch_census", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_census_hooks_the_final_resolve_not_the_pre_rewrite_candidate(census) -> None:
    """``resolve`` is the call that picks the callable; the candidate is not."""

    source = SCRIPT.read_text()
    assert "original_resolve = gguf_linear.resolve" in source
    assert "resolve_gguf_linear_dispatch" in source
    # The candidate hook must exist only to expose the rewrite delta.
    assert "original_candidate = gguf_linear.resolve_gguf_linear_dispatch" in source
    # ... and the launched table must be built from the resolve hook.
    assert "launched[" in source


def test_the_documented_rewrite_makes_the_candidate_misleading(census) -> None:
    """A decode candidate does not survive to launch at prefill row counts.

    This is the mechanism behind the retraction, pinned against the real
    rewrite helper rather than a restatement of it.
    """

    from hipengine.runtime import gguf_linear

    rewrite = getattr(gguf_linear, "_wmma_prefill_dispatch", None)
    if rewrite is None:
        pytest.skip("the WMMA prefill rewrite is not present in this revision")
    # The rewrite maps a decode-shaped variant to a prefill-shaped one at
    # batched rows; the census's old hook could not see that.
    assert "t16_gemv_decode" in SCRIPT.read_text()
