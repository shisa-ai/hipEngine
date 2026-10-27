"""Raw Q5/Q6 prefill RB4/RB16 retirement is a closed surface.

The gfx1100 raw-K prefill selector derives its variant from the requested row
slab (``rowbatch{n}_<output>`` or ``<coltile geometry>_<output>``). Only RB8 and
RB32 have a retained end-to-end policy, so RB4/RB16 lost their wrappers,
registry entries, and HIP exports. This module pins the result: the retired
keys are unregistered and unexported, an explicit request for one raises the
registry's missing-implementation error, every retained neighbour still selects
a registered kernel, and Laguna refuses the retired widths before it can build
a dispatch.

The shared selector keeps accepting 4/16 because gfx1151's raw-Q8 family still
ships them. What it no longer does is hand back a Q5/Q6 slab that no kernel in
that family implements. The check runs on the *final* selection only: the row
slab is an intermediate name, and a coltile or role override supplies the key
that actually launches.
"""

from __future__ import annotations

import pytest

import hipengine.kernels.hip_gfx1100.quant  # noqa: F401 - registers the family
from hipengine.kernels import hip_gfx1100
from hipengine.kernels.registry import KernelKey, is_registered

RETIRED_OUTPUTS = (
    ("gguf_q5_k", "bf16_bf16_out"),
    ("gguf_q5_k", "bf16_f32_out"),
    ("gguf_q6_k", "bf16_bf16_out"),
    ("gguf_q6_k", "bf16_f32_out"),
)
RETIRED_WIDTHS = (4, 16)
RETAINED_ROW_SLABS = (8, 32)
RETAINED_COLTILES = ("coltile2_rowbatch16", "coltile4_rowbatch8")


def _retired_variants() -> list[tuple[str, str]]:
    return [
        (quant, f"rowbatch{width}_{output}")
        for width in RETIRED_WIDTHS
        for quant, output in RETIRED_OUTPUTS
    ]


def _key(quant: str, variant: str, *, backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(backend, "linear", quant, variant)


def _rowbatch_selection(
    quant: str,
    *,
    variant: str = "prefill_bf16_f32_out",
    backend: str = "hip_gfx1100",
    width: int = 8,
    geometry: str = "rowbatch",
    rows: int = 128,
    in_features: int = 3072,
    out_features: int = 72,
):
    from hipengine.runtime.gguf_linear import (
        GGUFLinearDispatch,
        _raw_k_prefill_rowbatch_dispatch,
    )

    base = GGUFLinearDispatch(_key(quant, variant, backend=backend), "raw")
    return _raw_k_prefill_rowbatch_dispatch(
        base,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        row_batch=width,
        variant=geometry,
    )


@pytest.mark.parametrize(("quant", "variant"), _retired_variants())
def test_retired_raw_k_rowbatch_keys_are_unregistered(
    quant: str, variant: str
) -> None:
    assert not is_registered(_key(quant, variant))


@pytest.mark.parametrize(("quant", "variant"), _retired_variants())
def test_retired_raw_k_rowbatch_wrappers_are_gone(
    quant: str, variant: str
) -> None:
    from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv

    name = f"{quant}_gemv_{variant}"
    assert not hasattr(gguf_k_gemv, name)
    assert name not in gguf_k_gemv.__all__


@pytest.mark.parametrize("quant", ("gguf_q5_k", "gguf_q6_k"))
@pytest.mark.parametrize("width", RETAINED_ROW_SLABS)
def test_retained_raw_k_row_slabs_still_register(quant: str, width: int) -> None:
    for output in ("bf16_bf16_out", "bf16_f32_out"):
        assert is_registered(_key(quant, f"rowbatch{width}_{output}"))


@pytest.mark.parametrize("quant", ("gguf_q5_k", "gguf_q6_k"))
@pytest.mark.parametrize("geometry", RETAINED_COLTILES)
def test_retained_coltile_shapes_still_register(
    quant: str, geometry: str
) -> None:
    for output in ("bf16_bf16_out", "bf16_f32_out"):
        assert is_registered(_key(quant, f"{geometry}_{output}"))


def test_gfx1100_coltile_package_policy_is_unchanged() -> None:
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_VARIANT == "coltile"
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_ROWBATCH == 32
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_COLTILE2_SHAPES >= {
        ("gguf_q5_k", "bf16_bf16_out", 3072, 12288),
        ("gguf_q5_k", "bf16_f32_out", 3072, 6144),
        ("gguf_q5_k", "bf16_f32_out", 3072, 9216),
        ("gguf_q6_k", "bf16_f32_out", 3072, 9216),
    }


@pytest.mark.parametrize("quant", ("gguf_q5_k", "gguf_q6_k"))
@pytest.mark.parametrize("width", RETIRED_WIDTHS)
def test_explicit_retired_q5_q6_slab_raises_missing_implementation(
    quant: str, width: int
) -> None:
    from hipengine.kernels.registry import MissingKernelError

    with pytest.raises(MissingKernelError) as raised:
        _rowbatch_selection(quant, width=width)

    # The Q8 4/16 success counterpart lives in
    # ``test_gfx1151_raw_q8_widths_are_preserved``: the raw-Q8 slab route is
    # declared by gfx1151 only, so gfx1100 never selects one.
    requested = raised.value.requested
    assert requested == _key(quant, f"rowbatch{width}_bf16_f32_out")
    assert requested.backend == "hip_gfx1100"
    assert requested.quant == quant
    assert f"rowbatch{width}_bf16_f32_out" in str(raised.value)


@pytest.mark.parametrize(
    ("quant", "output", "in_features", "out_features"),
    sorted(hip_gfx1100.GGUF_RAW_K_PREFILL_COLTILE2_SHAPES),
)
def test_retained_coltile2_selection_is_registered(
    quant: str, output: str, in_features: int, out_features: int
) -> None:
    selected = _rowbatch_selection(
        quant,
        variant=f"prefill_{output}",
        width=32,
        geometry="coltile",
        rows=256,
        in_features=in_features,
        out_features=out_features,
    )
    assert selected.key.variant == f"coltile2_rowbatch16_{output}"
    assert is_registered(selected.key)


@pytest.mark.parametrize("quant", ("gguf_q5_k", "gguf_q6_k"))
def test_retained_coltile4_fallback_is_registered(quant: str) -> None:
    selected = _rowbatch_selection(
        quant,
        width=32,
        geometry="coltile",
        rows=256,
        out_features=4096,
    )
    assert selected.key.variant == "coltile4_rowbatch8_bf16_f32_out"
    assert is_registered(selected.key)


def test_absent_intermediate_slab_keeps_supported_coltile_final() -> None:
    """Only the final selection is validated, so a coltile override survives.

    Dropping the plain RB32 family leaves the selector's intermediate
    ``rowbatch32_*`` name unregistered. The (4,8) override still supplies a
    registered final key, so the selection must succeed rather than refuse the
    intermediate.
    """

    from hipengine.kernels.registry import unregister

    for output in ("bf16_bf16_out", "bf16_f32_out"):
        unregister(_key("gguf_q5_k", f"rowbatch32_{output}"))
    selected = _rowbatch_selection(
        "gguf_q5_k",
        width=32,
        geometry="coltile",
        rows=256,
        out_features=4096,
    )
    assert selected.key.variant == "coltile4_rowbatch8_bf16_f32_out"
    assert is_registered(selected.key)


def test_absent_intermediate_slab_keeps_supported_role_final() -> None:
    """A role-qualified final key is equally outside the retirement."""

    from hipengine.kernels.registry import unregister

    for output in ("bf16_bf16_out", "bf16_f32_out"):
        unregister(_key("gguf_q6_k", f"rowbatch32_{output}"))
    selected = _rowbatch_selection(
        "gguf_q6_k",
        width=32,
        geometry="coltile",
        rows=512,
        out_features=9216,
    )
    assert selected.key.variant == (
        "dpp_wave_reduction_full_group_compute_coltile2_rowbatch16_bf16_f32_out"
    )
    assert is_registered(selected.key)


def test_gfx1151_raw_q8_widths_are_preserved() -> None:
    """Qwen4Exp selects raw-Q8 slabs 4/16 on gfx1151; both stay registered.

    This is the Q8 success counterpart to the Q5/Q6 refusal above: the same
    shared selector, the same widths, a family that implements them.
    ``register_backend_kernels`` is called explicitly because the registry is
    restored between tests while the backend module import happens once.
    """

    from hipengine.kernels.backends import load_backend_kernel_package

    load_backend_kernel_package("hip_gfx1151")
    for width in RETIRED_WIDTHS:
        key = _key(
            "gguf_q8_0", f"rowbatch{width}_f32_f32_out", backend="hip_gfx1151"
        )
        assert is_registered(key)
        selected = _rowbatch_selection(
            "gguf_q8_0",
            variant="prefill_f32_f32_out",
            backend="hip_gfx1151",
            width=width,
        )
        assert selected.key == key
        assert is_registered(selected.key)


def test_gfx1151_does_not_inherit_retired_q5_q6_widths() -> None:
    """gfx1151 aliases the gfx1100 key space, so retirement propagates."""

    from hipengine.kernels.backends import load_backend_kernel_package

    load_backend_kernel_package("hip_gfx1151")
    for quant, variant in _retired_variants():
        assert not is_registered(_key(quant, variant, backend="hip_gfx1151"))


def test_iq_down_rowbatch_variants_are_preserved() -> None:
    """The IQ selected-prefill family owns its own slabs; keep them.

    Registered from ``hip_gfx1100/quant/gguf_iq_selected_prefill.py`` (the
    ``moe_linear`` block) and imported by ``hip_gfx1100.quant``.
    """

    for variant in (
        "selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out",
        "selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out",
    ):
        assert is_registered(
            KernelKey("hip_gfx1100", "moe_linear", "gguf_iq3_xxs", variant)
        )


@pytest.mark.parametrize("width", RETIRED_WIDTHS)
def test_laguna_refuses_retired_raw_k_rowbatch_widths(width: int) -> None:
    from hipengine.runtime.laguna_gguf_runner import (
        resolve_laguna_raw_k_prefill_rowbatch,
    )

    with pytest.raises(ValueError, match="row batch"):
        resolve_laguna_raw_k_prefill_rowbatch("hip_gfx1100", width)
    for retained in (0, *RETAINED_ROW_SLABS):
        assert (
            resolve_laguna_raw_k_prefill_rowbatch("hip_gfx1100", retained)
            == retained
        )
