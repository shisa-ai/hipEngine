"""Host contract for the three unregistered widened-row-reuse GR up exports.

``gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32`` carries a
``_row_batch`` selector, and three named wrappers pin it to 8, 8 + wave scale and
16 + wave scale. They are retained deliberately as the recorded comparison for a
different register-resource class, are not registered for dispatch, and must stay
*nameable* without becoming *selectable*.

The device-side file (``tests/test_gpu_qwen4exp_gr_row_batch8.py``) proves the
bit-exactness, the whole-registry exclusion and the per-call symbol engagement on
a GPU. This module is the CPU-only half of the same contract, so it runs in the
default tier: each named wrapper resolves exactly its own C symbol and nothing
else, refuses the combinations that have no instantiation instead of resolving a
neighbouring symbol, cannot be retargeted to the production width, and the
production entry still defaults to width 4. No kernel is built or launched -
``_cached_fn`` is replaced with a recorder, so the assertions are about routing
and refusal only.
"""

from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q8

PARENT = "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32"

CANDIDATES = (
    "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch8_f32",
    "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch8_wave_scale_f32",
    "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch16_wave_scale_f32",
)

# x, qweight, normalized, gate, mixed, rows, in_features, branches, hidden
LAUNCH_ARGS = (0, 0, 0, 0, 0, 4, 32, 4, 8)


class _FakeLibrary:
    """Stand-in for a loaded ``gguf_k_gemv.so``; never queried."""


class _FakeRuntime:
    """Stand-in for ``HipRuntime``; only read when a launch reports an error."""


@pytest.fixture
def resolved(monkeypatch):
    """Record every symbol the wrapper asks the library for."""
    symbols: list[str] = []

    def fake_cached_fn(library, symbol, argtypes):
        symbols.append(symbol)

        def launch(*args):
            return 0

        return launch

    monkeypatch.setattr(q8, "_cached_fn", fake_cached_fn)
    return symbols


def _call(name: str, **kwargs) -> None:
    getattr(q8, name)(
        *LAUNCH_ARGS, library=_FakeLibrary(), runtime=_FakeRuntime(), **kwargs
    )


def test_widened_exports_are_public_and_distinct():
    # Retaining the comparison requires the exports to stay importable by name; it
    # must not become a silent alias of the production parent.
    for name in CANDIDATES:
        assert name in q8.__all__, name
        assert callable(getattr(q8, name)), name
    assert len({getattr(q8, name) for name in CANDIDATES}) == len(CANDIDATES)
    for name in CANDIDATES:
        assert getattr(q8, name) is not getattr(q8, PARENT), name


@pytest.mark.parametrize("name", CANDIDATES)
def test_widened_wrapper_routes_its_own_symbol(name, resolved):
    # One call, one symbol, and it is the wrapper's own. A wrapper that dropped its
    # width would resolve the rowbatch4 parent and still "work", which is the
    # failure mode this catches.
    _call(name)
    assert resolved == [f"hipengine_{name}"]


@pytest.mark.parametrize("name", CANDIDATES)
def test_widened_wrapper_refuses_no_gate(name, resolved):
    # ``no_gate`` exists only at width 4, so the widened wrappers must refuse rather
    # than resolve a neighbouring symbol.
    with pytest.raises(ValueError):
        _call(name, _no_gate=True)
    assert resolved == []


@pytest.mark.parametrize("name", CANDIDATES)
def test_widened_wrapper_cannot_be_retargeted(name, resolved):
    # The width is pinned by the wrapper: passing it again is a duplicate keyword,
    # not an override, so no caller can quietly re-point a diagnostic at production.
    with pytest.raises(TypeError):
        _call(name, _row_batch=4)
    assert resolved == []


def test_production_entry_still_defaults_to_width_4(resolved):
    # Retention is not promotion: the unspecialized entry is still the width-4
    # production symbol.
    _call(PARENT)
    assert resolved == [f"hipengine_{PARENT}"]
