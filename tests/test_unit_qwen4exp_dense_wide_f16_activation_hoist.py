"""Launch contract for the wide-row Q8_0 dense prefill activation hoist.

The wide kernel converts its activation from f32 to f16 during LDS staging and
re-reads the f32 activation once per column block. Hoisting that conversion into
one bounded pass per launch and dispatching the ``*_f16in_*`` sibling was
measured at 590.6 ms over the route's 1056 launches in a 4096-token prefill,
with bit-identical output, but nothing wired it: the route shipped on the
f32-input owner.

These tests pin the launch-time swap and, just as importantly, every condition
under which it must keep the f32-input owner. The swap is arithmetic-preserving
either way, so a missed swap costs the saving and never correctness; a *wrong*
swap would feed the f16-input kernel f32 bytes, which is why the workspace
bound is checked before the conversion rather than after.
"""

from __future__ import annotations

import pytest

from hipengine.kernels.registry import KernelKey, register
from hipengine.runtime.gguf_linear import (
    WIDE_F16_ACTIVATION_ENV,
    _wide_f16_activation_target_variant,
    wide_f16_activation_enabled,
    wide_f16_activation_session,
    wide_f16_activation_workspace,
)

BACKEND = "hip_gfx1151"
QUANT = "gguf_q8_0"
LAYER = "linear"
SOURCE = "dense_wide256_f32_f32_out"
TARGET = "dense_wide256_f16in_f32_f32_out"
ROWS = 1024
K = 2560
N = 10240
# Exactly the f16 bytes the f16-input kernel reads for this launch.
COUNT = ROWS * K
REQUIRED_NBYTES = COUNT * 2


class _Weight:
    """Stand-in for a resident weight with one raw allocation."""

    class _Allocation:
        class _Tensor:
            ptr = 0x9000

        tensor = _Tensor()

    def allocation(self, name: str):
        assert name == "raw"
        return self._Allocation()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(WIDE_F16_ACTIVATION_ENV, raising=False)


def _staged_launch(monkeypatch, *, variant=SOURCE, rows=ROWS, in_features=K, libraries=None):
    """Run ``_launch_dense_wide_f16_staged`` with both ABIs instrumented.

    Returns ``(swapped, converter_calls, sibling_calls)``.
    """

    from hipengine.runtime import gguf_linear

    converter_calls: list[tuple] = []
    sibling_calls: list[tuple] = []

    def _fake_converter(x_ptr, out_ptr, n, **kwargs):
        converter_calls.append((x_ptr, out_ptr, n, kwargs))

    monkeypatch.setattr(
        "hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dense_wide.f32_to_f16",
        _fake_converter,
    )
    register(
        KernelKey(BACKEND, LAYER, QUANT, TARGET),
        lambda *args, **kwargs: None,
        replace=True,
    )
    monkeypatch.setattr(
        gguf_linear,
        "resolve",
        lambda **kwargs: lambda *args, **kwargs: sibling_calls.append((args, kwargs)),
    )

    swapped = gguf_linear._launch_dense_wide_f16_staged(
        lambda *args, **kwargs: None,
        _Weight(),
        0x1000,
        0x2000,
        rows,
        in_features,
        N,
        {"stream": 7},
        backend=BACKEND,
        quant=QUANT,
        layer=LAYER,
        variant=variant,
        runtime=object(),
        libraries=libraries,
    )
    return swapped, converter_calls, sibling_calls


def test_variant_suffix_maps_to_the_registered_sibling():
    assert _wide_f16_activation_target_variant(SOURCE) == TARGET
    assert (
        _wide_f16_activation_target_variant("dense_wide128x128_f32_f32_out")
        == "dense_wide128x128_f16in_f32_f32_out"
    )
    # Tiles with no f16-input export, and every non-wide variant, must decline.
    assert _wide_f16_activation_target_variant("dense_wide64x256_f32_f32_out") == (
        "dense_wide64x256_f16in_f32_f32_out"
    )
    assert _wide_f16_activation_target_variant("coltile8_rowbatch4_f32_f32_out") is None
    assert _wide_f16_activation_target_variant("wmma_prefill_f32_f32_out") is None
    assert _wide_f16_activation_target_variant("dense_wide256_f16in_f32_f32_out") is None


def test_no_workspace_keeps_the_f32_input_owner(monkeypatch):
    """The shipped route runs without a bound workspace; it must not convert."""

    swapped, converter, sibling = _staged_launch(monkeypatch)

    assert swapped is False
    assert converter == []
    assert sibling == []


def test_workspace_swaps_to_the_f16_sibling(monkeypatch):
    with wide_f16_activation_session(
        True, workspace_ptr=0x5000, workspace_nbytes=REQUIRED_NBYTES
    ):
        swapped, converter, sibling = _staged_launch(monkeypatch)

    assert swapped is True
    assert len(converter) == 1
    x_ptr, out_ptr, n, kwargs = converter[0]
    assert (x_ptr, out_ptr, n) == (0x1000, 0x5000, COUNT)
    assert kwargs["stream"] == 7
    assert len(sibling) == 1
    # The sibling reads the staged buffer, not the f32 activation.
    sibling_args, _ = sibling[0]
    assert sibling_args[0] == 0x5000
    assert sibling_args[1] == 0x9000
    assert sibling_args[3:] == (ROWS, K, N)


def test_undersized_workspace_keeps_the_f32_input_owner(monkeypatch):
    """One byte short is short: the f16 kernel would read past the buffer."""

    with wide_f16_activation_session(
        True, workspace_ptr=0x5000, workspace_nbytes=REQUIRED_NBYTES - 1
    ):
        swapped, converter, sibling = _staged_launch(monkeypatch)

    assert swapped is False
    assert converter == []
    assert sibling == []


def test_larger_workspace_is_usable(monkeypatch):
    """The buffer is shared with the B2 staging route and is sized for its cap."""

    with wide_f16_activation_session(
        True, workspace_ptr=0x5000, workspace_nbytes=REQUIRED_NBYTES * 4
    ):
        swapped, _, _ = _staged_launch(monkeypatch)

    assert swapped is True


def test_env_override_disables_the_hoist(monkeypatch):
    monkeypatch.setenv(WIDE_F16_ACTIVATION_ENV, "0")
    with wide_f16_activation_session(
        True, workspace_ptr=0x5000, workspace_nbytes=REQUIRED_NBYTES
    ):
        swapped, converter, _ = _staged_launch(monkeypatch)

    assert swapped is False
    assert converter == []
    monkeypatch.setenv(WIDE_F16_ACTIVATION_ENV, "1")
    assert wide_f16_activation_enabled(False) is True


def test_session_clears_the_workspace_on_exit():
    assert wide_f16_activation_workspace() is None
    with wide_f16_activation_session(
        True, workspace_ptr=0x5000, workspace_nbytes=REQUIRED_NBYTES
    ):
        workspace = wide_f16_activation_workspace()
        assert workspace is not None and workspace.ptr == 0x5000
    assert wide_f16_activation_workspace() is None


def test_disabled_session_binds_no_workspace():
    with wide_f16_activation_session(
        False, workspace_ptr=0x5000, workspace_nbytes=REQUIRED_NBYTES
    ):
        assert wide_f16_activation_workspace() is None


def test_backend_package_is_populated_before_the_sibling_lookup(monkeypatch):
    """The sibling lives in a package the dispatch populates, not in the cache.

    ``register_gguf_q8_0_dense_wide_kernels`` registers under ``hip_gfx1100``
    while the dispatch key's backend is the *package* (``hip_gfx1151``), so a
    bare ``is_registered`` on the sibling key is False until
    ``_ensure_linear_kernel_registered`` loads the package. Skipping that step
    is how this swap silently declined on every launch while the census showed
    the f32-input symbol 792 times.
    """

    from hipengine.runtime import gguf_linear

    state = {"registered": False, "populated": 0}

    def _is_registered(key):
        return state["registered"]

    def _populate(key):
        state["populated"] += 1
        state["registered"] = True

    monkeypatch.setattr(gguf_linear, "is_registered", _is_registered)
    monkeypatch.setattr(gguf_linear, "_ensure_linear_kernel_registered", _populate)
    with wide_f16_activation_session(
        True, workspace_ptr=0x5000, workspace_nbytes=REQUIRED_NBYTES
    ):
        swapped, converter, _ = _staged_launch(monkeypatch)

    assert state["populated"] == 1
    assert swapped is True
    assert len(converter) == 1


def test_unregistered_sibling_keeps_the_f32_input_owner(monkeypatch):
    """A tile with no f16-input export must not be swapped to a missing symbol."""

    from hipengine.runtime import gguf_linear

    register(
        KernelKey(BACKEND, LAYER, QUANT, "dense_wide64x256_f16in_f32_f32_out"),
        None,
        replace=True,
    )
    monkeypatch.setattr(gguf_linear, "is_registered", lambda key: False)
    with wide_f16_activation_session(
        True, workspace_ptr=0x5000, workspace_nbytes=REQUIRED_NBYTES
    ):
        swapped, converter, _ = _staged_launch(
            monkeypatch, variant="dense_wide64x256_f32_f32_out"
        )

    assert swapped is False
    assert converter == []


def test_variant_scoped_library_is_used_for_both_calls(monkeypatch):
    """The cast and the tile live in the same variant-scoped shared object."""

    wide_library = object()
    libraries = {f"{QUANT}:{TARGET}": wide_library, QUANT: object()}
    with wide_f16_activation_session(
        True, workspace_ptr=0x5000, workspace_nbytes=REQUIRED_NBYTES
    ):
        swapped, converter, sibling = _staged_launch(monkeypatch, libraries=libraries)

    assert swapped is True
    assert converter[0][3]["library"] is wide_library
    _, sibling_kwargs = sibling[0]
    assert sibling_kwargs["library"] is wide_library
