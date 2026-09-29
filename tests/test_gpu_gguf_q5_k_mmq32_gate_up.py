"""RED: the grouped rowbatch gate_up owner has no Q5_K entry.

Punchlist P3 (``docs/campaigns/GEMMA4-26B-A4B-PUNCHLIST.md``). Layer 29 carries
the artifact's only Q5_K gate_up (``scripts/gemma4_family_census.py:53``), and
while every other layer's gate_up runs the grouped rowbatch leaf,
``gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out``
(6 launches / 66.501 ms in the recorded 1024-token trace), that owner was
registered for ``gguf_q4_k`` only. Q5_K therefore resolved no dual owner on
Gemma 4's dispatch and fell through to the per-row GEMV.

Gemma 4's dispatch asks for
``selected_dual_wmma_prefill_compact_bf16_bf16_out`` and, when that is absent,
the selected GEMV -- it never selects a variant name that Q5_K provides. The
down half of this row was a routing defect and is fixed (``7a91b5735``); this
half is a genuinely missing owner, so the row never reaches a grouped owner at
all.

The registry is four-axis -- ``(backend, layer, quant, variant)`` -- so the leaf
does not need a new variant *name*: it needs to be reachable with
``quant="gguf_q5_k"``. That is what this test asserts.

Scope note (an earlier revision of this task targeted the MMQ32 leaf instead;
``selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out`` appears in
neither the registry inventory nor the punchlist's named target, so the
assertion was retargeted here). Q5_K already has ``mmq32_q8_1_*`` and
``mmq_i128_j128_k256_*`` registrations -- it was never devoid of MMQ kernels.

What GREEN must add (recorded here so the numerical gate cannot be quietly
dropped): drive the owner with Q5_K weights and hold it to the
``docs/OPTIMIZATION.md`` §4.2 outer floor against the CPU reference.

  - weight fixture: ``make_q5_k_weight`` from
    ``tests.test_gpu_gguf_k_gemv`` (raw GGUF ``block_q5_K`` bytes)
  - oracle: ``hipengine.kernels.cpu_reference.gguf_q5_k_gemv``
    (``ops.py:499``), an exact ``dequantize`` + ``np.matmul`` over the raw bytes
  - thresholds: ``np.testing.assert_allclose(..., _TOLERANCE_BF16)`` plus
    max-softmax-KL ``<= 0.05`` and argmax agreement ``>= 0.9``, exactly the
    assertions the Q4_K sibling already applies
  - §8 also requires a ``rocprofv3 --kernel-trace`` smoke showing the kernel
    runs under its expected name with a plausible ``DurationNs``

Q5_K carries an extra ``qh[32]`` plane over Q4_K: its fifth weight bit comes
from the block's ``qh`` slab and its superblock stride is 176 bytes against
Q4_K's 144. The shared kernel already encodes this behind its ``Q5K`` template
parameter, which is why the owner is one symbol in one translation unit rather
than a source port. Pre-flight and scope are in worklog entry
``20260929T195035.482780Z-lhl-gemma4-q5k-gateup-mmq32-preflight-1447a7.md``.
"""

from __future__ import annotations

import ctypes

import pytest

_GATE_UP_VARIANT = "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out"
_BACKEND = "hip_gfx1100"
_LAYER = "moe_linear"
_QUANT = "gguf_q5_k"


def _hip_available() -> bool:
    """Explicit HIP guard so no-ROCm CI and publish runners skip, not fail."""
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q5_k_gate_up_reaches_the_grouped_rowbatch_owner() -> None:
    """RED: the grouped rowbatch gate_up owner must resolve for ``gguf_q5_k``.

    The registry keys kernels on ``(backend, layer, quant, variant)``, so a
    successful lookup under ``gguf_q5_k`` is precisely the contract P3
    establishes. ``missing="none"`` is used rather than a static import so this
    module still *collects* while the owner is absent -- the missing leaf then
    fails as an assertion instead of an ImportError, which keeps the failure
    legible as a genuine RED.
    Registration is import-time and lazy, so importing the owning module is
    what makes the key visible to ``resolve`` -- without it a pre-import lookup
    returns ``None`` whether or not the owner exists.
    """
    import hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill  # noqa: F401
    from hipengine.kernels.registry import resolve

    leaf = resolve(
        backend=_BACKEND,
        layer=_LAYER,
        quant=_QUANT,
        variant=_GATE_UP_VARIANT,
        missing="none",
    )

    assert leaf is not None, (
        f"no {_QUANT} owner for ({_BACKEND}, {_LAYER}, {_GATE_UP_VARIANT}): the "
        "grouped rowbatch gate_up owner is registered for gguf_q4_k only, so "
        "layer 29's Q5_K gate_up has no dual owner and falls through to the "
        "slow per-row GEMV. Add the Q5_K symbol to "
        "gguf_q4_k_selected_prefill.hip launching the grouped kernel with "
        "Q5K=true, expose its sibling function in the module, and register it "
        "under quant axis 'gguf_q5_k'."
    )