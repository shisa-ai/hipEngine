"""RED: the int8 MMQ32 gate_up leaf has no Q5_K tile loader.

Punchlist P3 Step 2 (``docs/campaigns/GEMMA4-26B-A4B-PUNCHLIST.md``). Layer 29
carries the artifact's only Q5_K gate_up (``scripts/gemma4_family_census.py:53``),
which today runs the BF16 grouped rowbatch owner at 21.9 ms / 1024 tokens, while
llama.cpp runs the same projection through its ordinary MMQ in 2.6 ms.

The fast path exists and is measured: ``gemma4_project_experts_mmq_dual`` packs
activations to llama.cpp-style DS4 ``block_q8_1_mmq`` and runs the 32x32
packed-dot int8 leaf, whose docstring records it at **2.07x** the BF16 WMMA owner
at the Gemma4 MoE shape (2.00x even paying the packing cost). But its route guard
rejects everything except Q4_K:

    # gemma4_experts.py:_mmq_dual_route
    if weight.spec.quant_key != _MMQ_DUAL_QUANT_KEY:   # "gguf_q4_k"
        return False

So the leaf resolves on the ``gguf_q4_k`` axis and raises ``MissingKernelError``
on ``gguf_q5_k``. Verified directly: after importing
``hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill``, the
key ``(hip_gfx1100, moe_linear, gguf_q4_k, selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out)``
resolves to a function, while the identical key under ``gguf_q5_k`` raises.

The registry is four-axis -- ``(backend, layer, quant, variant)`` -- so the leaf
does not need a new variant *name*: it needs a Q5_K tile loader reachable with
``quant="gguf_q5_k"``, and then the route guard widened. That is what this test
asserts, and it is the RED half of RED/GREEN: it fails while the loader is absent
and passes once the port lands.

Q5_K carries an extra ``qh[32]`` plane over Q4_K -- each weight's fifth bit sits
in the block's ``qh`` slab and the superblock is 176 bytes against Q4_K's 144 --
which is why widening the guard alone would misdecode rather than merely run
slow. This is a source port from ``gguf_q5_k_q8_1_selected_prefill.hip``, not a
guard change. Scope and pre-flight are in worklog entry
``20260929T195035.482780Z-lhl-gemma4-q5k-gateup-mmq32-preflight-1447a7.md``.

What GREEN must add (recorded here so the numerical gate cannot be quietly
dropped): drive the leaf with Q5_K weights and hold it to the
``docs/OPTIMIZATION.md`` §4.2 outer floor against the CPU reference.

  - weight fixture: ``make_q5_k_weight`` from
    ``tests.test_gpu_gguf_k_gemv`` (raw GGUF ``block_q5_K`` bytes)
  - activation packing: ``pack_q8_1_mmq_ds4_from_bf16``, which is
    quant-agnostic -- it packs activations, not weights
  - oracle: ``hipengine.kernels.cpu_reference.gguf_q5_k_gemv``
    (``ops.py:499``), an exact ``dequantize`` + ``np.matmul`` over the raw bytes
  - thresholds: ``np.testing.assert_allclose(..., _TOLERANCE_BF16)`` plus
    max-softmax-KL ``<= 0.05`` and argmax agreement ``>= 0.9``, exactly the
    assertions ``test_q4_k_q8_1_ds4_mmq32_selected_prefill_bf16_matches_ds4_cpu_reference``
    already applies to the Q4_K leaf
  - §8 also requires a ``rocprofv3 --kernel-trace`` smoke showing the kernel
    runs under its expected name with a plausible ``DurationNs``

Scope note: an interim revision of this test retargeted it to
``selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out``. That was a
false negative -- the grouped variant family was checked without importing
``gguf_q4_k_q8_1_selected_prefill``, and registration here is import-time and
lazy, so a pre-import ``resolve`` returns nothing whether or not the key exists.
Gate_up also does not need a grouped owner: P3's own evidence says it already
runs one. The target below is restored to the MMQ32 leaf.
"""

from __future__ import annotations

import ctypes

import pytest

_MMQ32_VARIANT = "selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out"
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
def test_q5_k_gate_up_reaches_the_mmq32_leaf() -> None:
    """RED: the MMQ32 gate_up leaf must resolve for ``gguf_q5_k``.

    The registry keys kernels on ``(backend, layer, quant, variant)``, so the
    same lookup that succeeds for ``gguf_q4_k`` must succeed for
    ``gguf_q5_k``. Importing the owning module is load-bearing: registration is
    import-time and lazy, so without it ``resolve`` returns ``None`` whether or
    not the leaf exists. ``missing="none"`` keeps this module *collectable*
    while it is absent, so the failure reads as a genuine RED assertion rather
    than an ImportError.
    """
    import hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill  # noqa: F401
    from hipengine.kernels.registry import resolve

    leaf = resolve(
        backend=_BACKEND,
        layer=_LAYER,
        quant=_QUANT,
        variant=_MMQ32_VARIANT,
        missing="none",
    )

    assert leaf is not None, (
        f"no {_QUANT} owner for ({_BACKEND}, {_LAYER}, {_MMQ32_VARIANT}): the "
        "int8 MMQ32 gate_up leaf has no Q5_K tile loader, so layer 29's Q5_K "
        "gate_up stays on the BF16 grouped owner at 21.9 ms / 1024 instead of "
        "the int8 leaf measured at 2.07x it. Port the Q5_K decode (qh slab, "
        "176-byte superblock) from gguf_q5_k_q8_1_selected_prefill.hip, then "
        "widen _mmq_dual_route's _MMQ_DUAL_QUANT_KEY guard in "
        "gemma4_experts.py."
    )