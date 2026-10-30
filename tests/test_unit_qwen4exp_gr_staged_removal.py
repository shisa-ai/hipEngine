"""Removal contract for the rejected HC-1 GR staged-f32 route.

The GR-up LDS-staged f32 tile was wired as an opt-in production candidate behind
``HIPENGINE_QWEN4_EXP_GR_STAGED``. Its whole-route gate ran and rejected it
(``benchmarks/results/2026-09-29-qwen4exp-hc1-gr-staged-gate/gate-final.json``:
``status: failed``, ``measurement_valid: true``, 1920 counted dispatches,
``mixed_ja_en`` mean KL 1.499e-3 against a 1e-3 limit), so the selector, its
dispatch branch and its gate candidate were removed rather than left
default-off.

Two things survive, and these tests pin them:

- the GR selection chain (``gr_iu8`` > ``gr_wide`` > the fused exact parent)
  must be unchanged by the removal, and the removed variable must move none of
  it;
- the kernel module stays, because ``register_gfx1151_kernels`` registers the
  *dense* staged-f32 key from the same wrapper. Deleting the module would
  silently drop that dense registration.

The kernel's arithmetic contract lives in
``tests/test_gpu_qwen4exp_gr_staged_f32.py``.
"""

import inspect
from pathlib import Path

from hipengine.kernels.registry import KernelKey, is_registered, resolve

REMOVED_ENV = "HIPENGINE_QWEN4_EXP_GR_STAGED"
GR_LAYER = "linear+gr_gated_mean"
GR_QUANT = "gguf_q8_0"


def _runner():
    from hipengine.runtime import qwen4_exp_runner

    return qwen4_exp_runner


def test_selector_helper_is_gone() -> None:
    assert not hasattr(_runner(), "_qwen4_exp_gr_staged_enabled")


def test_runner_no_longer_reads_or_launches_the_removed_selector() -> None:
    """The variable must be dead, not merely unreachable.

    Unreachability was the state that let a rejected route keep a live env read:
    on the UD production plan ``GR_WIDE`` binds ahead of the staged branch, so
    the branch could not engage there while the flag and its launch stayed
    wired. A re-added read or launch is what this pins out.
    """

    runner = _runner()
    source = Path(inspect.getfile(runner)).read_text()
    assert REMOVED_ENV not in source
    assert "gr_staged_f32_up" not in runner.__dict__


def test_removed_selector_does_not_move_the_surviving_gr_chain(monkeypatch) -> None:
    """With nothing else bound, the GR route still takes the fused exact parent.

    The fused exact kernel is the strict fallback for every GR candidate, so
    which arm a prefill-sized row set lands on is the contract the removal could
    have broken. The environment below is the rejected gate candidate's own
    (``HIPENGINE_QWEN4_EXP_GR_STAGED=1``, the competing GR routes off); that arm
    used to launch a different projection at exactly these shapes, and it now
    has to be indistinguishable from the recovery plan.
    """

    runner = _runner()
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_GR_WIDE", raising=False)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_GR_IU8", raising=False)
    monkeypatch.delenv(REMOVED_ENV, raising=False)

    assert runner._qwen4_exp_gr_up_sigmoid_mean_enabled(1024) is True
    assert runner._qwen4_exp_gr_iu8_enabled(1024) is False
    assert runner._qwen4_exp_gr_wide_enabled(1024, 320) is False

    monkeypatch.setenv(REMOVED_ENV, "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_WIDE", "0")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_IU8", "0")
    assert runner._qwen4_exp_gr_iu8_enabled(1024) is False
    assert runner._qwen4_exp_gr_wide_enabled(1024, 320) is False


def test_strict_gr_fallback_key_stays_registered() -> None:
    """The fused exact GR parent is what the removed route fell back to."""

    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        register_gguf_k_gemv_kernels,
    )

    # ``tests/conftest.py`` restores the registry to its collection-time
    # baseline after every test, so the module-level registration is re-applied
    # explicitly here rather than relied on.
    register_gguf_k_gemv_kernels(replace=True)

    assert is_registered(
        KernelKey(
            "hip_gfx1151", GR_LAYER, GR_QUANT,
            "coltile2_branch4_rowbatch4_f32_exact",
        )
    )


def test_gr_prototype_key_remains_registered() -> None:
    """The prototype is retained; only the route that selected it is gone."""

    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_gr_staged_f32 as staged

    staged.register_gr_staged_f32_kernel(replace=True)
    key = KernelKey("hip_gfx1100", GR_LAYER, GR_QUANT, staged.GR_STAGED_F32_VARIANT)

    assert is_registered(key)
    assert resolve(
        backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant
    ) is staged.gr_staged_f32_up


def test_dense_staged_registration_survives_the_gr_removal() -> None:
    """The dense staged-f32 key is registered from the same wrapper.

    This is the coupling that makes deleting the module unsafe: the dense key
    would stop resolving and ``HIPENGINE_QWEN4_EXP_Q8_STAGED_F32_LAYERS`` would
    silently select nothing while still looking bound.
    """

    from hipengine.kernels.backends import backend_package_capability
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_gr_staged_f32 as staged
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    key = KernelKey(
        "hip_gfx1151",
        "linear",
        GR_QUANT,
        backend_package_capability("hip_gfx1151", "STAGED_F32_VARIANT"),
    )

    assert is_registered(key)
    assert resolve(
        backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant
    ) is staged.gr_staged_f32_up
