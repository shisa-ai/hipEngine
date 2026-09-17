"""CPU tests for the shared resident bulk-prefill GGUF dispatch context.

``Qwen35GGUFResidentSession`` and the rank-local TP2 bulk prefill both have to
resolve the *same* GGUF linear kernels for the same weight and rows. The
resident session owns eleven session-scoped owners; six of them are plain
process-global toggles that change dispatch, and
:func:`resident_prefill_dispatch_session` is the single source of that set.

These tests pin three things without hardware:

* the owner set is entered and restored exactly, including on failure and when
  nested, and it reports the same state for every rank of one group;
* the *actual* resolved variant changes for the layer-0 Q6_K ``attn_qkv`` shape
  (the measured first divergence) and does not change for the Q4_K
  ``attn_gate`` shape from the same launch group;
* the shipped policy resolver is the one both routes read.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv  # noqa: F401
import hipengine.runtime.gguf_linear as gguf_linear
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
)
from hipengine.runtime.gguf_linear import (
    gguf_prefill_dispatch_context,
    resident_session_wmma_prefill_default,
)
from hipengine.runtime.qwen35_gguf_runner import resident_prefill_dispatch_session

# The layer-0 shapes from the Qwen3.8-27B Q4_K_M artifact: ``attn_qkv`` is Q6_K
# (5120 -> 10240) and ``attn_gate`` is Q4_K (5120 -> 6144).
QKV_SHAPE = (5120, 10240)
GATE_SHAPE = (5120, 6144)
BULK_ROWS = 64

_OWNER_KEYS = (
    "wmma_prefill",
    "gemv_decode",
    "q8_t16_two_wave_prefill",
    "q8_t16_dual_wmma_prefill",
    "q4_pack8_dual_wmma_silu_prefill",
    "q4_t16_unequal_pair_prefill",
)

#: The six owners are the device-free half of the resident's eleven-owner
#: stack; the five device-pointer owners stay with the session that owns their
#: scratch. Every key here must be reported by the accessor.
assert set(_OWNER_KEYS) <= set(gguf_prefill_dispatch_context())


class _FakeWeight:
    def __init__(self, layout: str, quant_key: str) -> None:
        allocations = {
            "raw": SimpleNamespace(tensor=SimpleNamespace(ptr=10)),
            "qweight": SimpleNamespace(tensor=SimpleNamespace(ptr=11)),
            "scales": SimpleNamespace(tensor=SimpleNamespace(ptr=12)),
            "mins": SimpleNamespace(tensor=SimpleNamespace(ptr=13)),
            "tiles": SimpleNamespace(tensor=SimpleNamespace(ptr=14)),
        }
        self.spec = SimpleNamespace(layout=layout, quant_key=quant_key)
        self.allocations = allocations

    def allocation(self, name: str = "raw"):
        return self.allocations[name]


class _FakeRunner:
    """Minimal runner surface the context resolver reads."""

    def __init__(self, *, backend: str = "hip_gfx1100", weights=None) -> None:
        self.backend = backend
        self.weights = weights


def _resolve_variant(
    weight: _FakeWeight,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    output_dtype: int | None = None,
):
    """Resolve one GGUF linear launch to its registered ``(quant, variant)``."""

    seen: list[tuple[str, str]] = []
    original = gguf_linear.resolve

    def _stub(**kwargs):
        seen.append((kwargs.get("quant"), kwargs.get("variant")))
        return lambda *args, **kw: None

    gguf_linear.resolve = _stub
    gguf_linear._DISPATCH_RESOLVE_CACHE.clear()
    try:
        gguf_linear.launch_gguf_linear(
            weight,
            0,
            0,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            output_dtype=(
                gguf_linear.GGUF_OUTPUT_BF16 if output_dtype is None else output_dtype
            ),
            stream=0,
            runtime=None,
        )
    finally:
        gguf_linear.resolve = original
        gguf_linear._DISPATCH_RESOLVE_CACHE.clear()
    assert seen, "the launch never resolved a kernel"
    return seen[-1]


@pytest.fixture(autouse=True)
def _reset_dispatch_context():
    before = dict(gguf_prefill_dispatch_context())
    yield
    after = dict(gguf_prefill_dispatch_context())
    assert after == before, f"the context leaked: {before} -> {after}"


# ---------------------------------------------------------------------------
# the shipped policy resolver
# ---------------------------------------------------------------------------


def test_shipped_resident_prefill_policy_is_wmma_on(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL", raising=False)
    assert resident_session_wmma_prefill_default() is True


def test_shipped_resident_prefill_policy_honours_the_diagnostic_env(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL", "0")
    assert resident_session_wmma_prefill_default() is False
    monkeypatch.setenv("HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL", "1")
    assert resident_session_wmma_prefill_default() is True
    monkeypatch.setenv("HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL", "yes")
    assert resident_session_wmma_prefill_default() is True


def test_shipped_resident_prefill_policy_rejects_a_typo(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL", "ture")
    with pytest.raises(ValueError, match="HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL"):
        resident_session_wmma_prefill_default()


# ---------------------------------------------------------------------------
# enter / restore semantics
# ---------------------------------------------------------------------------


def test_context_sets_and_restores_every_owner() -> None:
    runner = _FakeRunner()
    before = dict(gguf_prefill_dispatch_context())
    with resident_prefill_dispatch_session(
        runner,
        prompt_tokens=BULK_ROWS,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ):
        inside = dict(gguf_prefill_dispatch_context())
        assert inside["wmma_prefill"] is True
        assert inside["gemv_decode"] is True
        # The geometry-gated owners stay off for a runner with no weights.
        assert inside["q8_t16_dual_wmma_prefill"] is False
        assert inside["q4_pack8_dual_wmma_silu_prefill"] is False
        assert inside["q4_t16_unequal_pair_prefill"] is False
        assert inside["t16_f16_rocblas_prefill"] is False
    assert dict(gguf_prefill_dispatch_context()) == before


def test_context_restores_after_an_exception() -> None:
    runner = _FakeRunner()
    before = dict(gguf_prefill_dispatch_context())
    with pytest.raises(RuntimeError, match="boom"):
        with resident_prefill_dispatch_session(
            runner,
            prompt_tokens=BULK_ROWS,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ):
            assert gguf_prefill_dispatch_context()["wmma_prefill"] is True
            raise RuntimeError("boom")
    assert dict(gguf_prefill_dispatch_context()) == before


def test_context_nests_and_unwinds_outermost_last() -> None:
    runner = _FakeRunner()
    before = dict(gguf_prefill_dispatch_context())
    with resident_prefill_dispatch_session(
        runner,
        prompt_tokens=BULK_ROWS,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ):
        with resident_prefill_dispatch_session(
            runner,
            prompt_tokens=BULK_ROWS,
            use_wmma_prefill=False,
            use_gemv_decode=False,
        ):
            inner = dict(gguf_prefill_dispatch_context())
            assert inner["wmma_prefill"] is False
            assert inner["gemv_decode"] is False
        outer = dict(gguf_prefill_dispatch_context())
        assert outer["wmma_prefill"] is True
        assert outer["gemv_decode"] is True
    assert dict(gguf_prefill_dispatch_context()) == before


def test_context_is_identical_for_every_rank_of_one_group() -> None:
    """The owners resolve from ``(backend, geometry, rows)``, not from a rank."""

    left = _FakeRunner()
    right = _FakeRunner()
    with resident_prefill_dispatch_session(
        left,
        prompt_tokens=BULK_ROWS,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ):
        first = dict(gguf_prefill_dispatch_context())
    with resident_prefill_dispatch_session(
        right,
        prompt_tokens=BULK_ROWS,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ):
        second = dict(gguf_prefill_dispatch_context())
    assert first == second


def test_context_reports_the_two_wave_owner_it_enters() -> None:
    """``q8_t16_two_wave_prefill`` is admitted for gfx1100 at these rows."""

    runner = _FakeRunner()
    assert gguf_prefill_dispatch_context()["q8_t16_two_wave_prefill"] is False
    with resident_prefill_dispatch_session(
        runner,
        prompt_tokens=BULK_ROWS,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ):
        assert gguf_prefill_dispatch_context()["q8_t16_two_wave_prefill"] is True
    assert gguf_prefill_dispatch_context()["q8_t16_two_wave_prefill"] is False


# ---------------------------------------------------------------------------
# actual route selection
# ---------------------------------------------------------------------------


def test_layer0_qkv_route_changes_with_the_context() -> None:
    """The measured first divergence: Q6_K ``attn_qkv`` at rows=64."""

    weight = _FakeWeight(LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR, "gguf_q6_k_t16_qmicro_planar_v1")
    outside = _resolve_variant(
        weight, rows=BULK_ROWS, in_features=QKV_SHAPE[0], out_features=QKV_SHAPE[1]
    )
    runner = _FakeRunner()
    with resident_prefill_dispatch_session(
        runner,
        prompt_tokens=BULK_ROWS,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ):
        inside = _resolve_variant(
            weight,
            rows=BULK_ROWS,
            in_features=QKV_SHAPE[0],
            out_features=QKV_SHAPE[1],
        )
    assert outside == ("gguf_q6_k_t16_qmicro_planar_v1", "t16_gemv_decode_bf16_bf16_out")
    assert inside == ("gguf_q6_k_t16_qmicro_planar_v1", "t16_wmma_prefill_bf16_bf16_out")
    assert outside != inside


def test_layer0_gate_route_is_already_the_wmma_variant_outside_the_context() -> None:
    """Why the Q4_K ``attn_gate`` projection was bit-identical on both routes."""

    weight = _FakeWeight(LAYOUT_GGUF_Q4_K_T16, "gguf_q4_k_t16_v1")
    outside = _resolve_variant(
        weight, rows=BULK_ROWS, in_features=GATE_SHAPE[0], out_features=GATE_SHAPE[1]
    )
    runner = _FakeRunner()
    with resident_prefill_dispatch_session(
        runner,
        prompt_tokens=BULK_ROWS,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ):
        inside = _resolve_variant(
            weight,
            rows=BULK_ROWS,
            in_features=GATE_SHAPE[0],
            out_features=GATE_SHAPE[1],
        )
    assert outside == inside
    assert outside == ("gguf_q4_k_t16_v1", "t16_wmma_prefill_bf16_bf16_out")


def test_wmma_override_is_read_from_the_context_not_the_environment(monkeypatch) -> None:
    """An explicit ``use_wmma_prefill`` wins over the ambient env var."""

    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "0")
    weight = _FakeWeight(LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR, "gguf_q6_k_t16_qmicro_planar_v1")
    runner = _FakeRunner()
    with resident_prefill_dispatch_session(
        runner,
        prompt_tokens=BULK_ROWS,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ):
        inside = _resolve_variant(
            weight,
            rows=BULK_ROWS,
            in_features=QKV_SHAPE[0],
            out_features=QKV_SHAPE[1],
        )
    assert inside == ("gguf_q6_k_t16_qmicro_planar_v1", "t16_wmma_prefill_bf16_bf16_out")


# ---------------------------------------------------------------------------
# the head projection is deliberately outside the context
# ---------------------------------------------------------------------------


def test_f32_out_wmma_prefill_variant_is_unregistered() -> None:
    """Why the TP2 bulk head must not enter the resident dispatch context.

    The Q6_K planar ``t16_wmma_prefill_bf16_f32_out`` leaf has no registered
    kernel. The registry's ``resolve`` silently returns the CPU reference
    ``linear``, so entering the context around the head projection turns the
    launch into a ``TypeError`` instead of a kernel. The resident session never
    sees this because it samples through its dedicated ``lm_head`` kernel
    rather than ``launch_gguf_linear``.
    """

    from hipengine.kernels.registry import KernelKey, is_registered

    weight = _FakeWeight(LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR, "gguf_q6_k_t16_qmicro_planar_v1")
    runner = _FakeRunner()
    with resident_prefill_dispatch_session(
        runner,
        prompt_tokens=BULK_ROWS,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ):
        variant = _resolve_variant(
            weight,
            rows=BULK_ROWS,
            in_features=5120,
            out_features=248320,
            output_dtype=gguf_linear.GGUF_OUTPUT_F32,
        )
    assert variant == ("gguf_q6_k_t16_qmicro_planar_v1", "t16_wmma_prefill_bf16_f32_out")
    assert not is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_wmma_prefill_bf16_f32_out",
        )
    )


def test_ambient_f32_out_head_route_is_registered() -> None:
    """The route the TP2 bulk head keeps is a real kernel."""

    from hipengine.kernels.registry import KernelKey, is_registered

    weight = _FakeWeight(LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR, "gguf_q6_k_t16_qmicro_planar_v1")
    assert gguf_prefill_dispatch_context()["wmma_prefill"] is False
    variant = _resolve_variant(
        weight,
        rows=BULK_ROWS,
        in_features=5120,
        out_features=248320,
        output_dtype=gguf_linear.GGUF_OUTPUT_F32,
    )
    assert variant == ("gguf_q6_k_t16_qmicro_planar_v1", "t16_gemv_decode_bf16_f32_out")
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_decode_bf16_f32_out",
        )
    )


def test_shard_down_projection_needs_the_context() -> None:
    """The MLP shard's Q6_K down projection is a resident-helper launch."""

    weight = _FakeWeight(LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR, "gguf_q6_k_t16_qmicro_planar_v1")
    outside = _resolve_variant(weight, rows=BULK_ROWS, in_features=8704, out_features=5120)
    runner = _FakeRunner()
    with resident_prefill_dispatch_session(
        runner,
        prompt_tokens=BULK_ROWS,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ):
        inside = _resolve_variant(weight, rows=BULK_ROWS, in_features=8704, out_features=5120)
    assert outside == ("gguf_q6_k_t16_qmicro_planar_v1", "t16_gemv_decode_bf16_bf16_out")
    assert inside == ("gguf_q6_k_t16_qmicro_planar_v1", "t16_wmma_prefill_bf16_bf16_out")


def test_rows_one_keeps_the_decode_route_under_the_context() -> None:
    """``wmma_prefill`` is a batched-prefill rewrite; rows<=1 is untouched."""

    weight = _FakeWeight(LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR, "gguf_q6_k_t16_qmicro_planar_v1")
    runner = _FakeRunner()
    with resident_prefill_dispatch_session(
        runner,
        prompt_tokens=1,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ):
        variant = _resolve_variant(
            weight, rows=1, in_features=QKV_SHAPE[0], out_features=QKV_SHAPE[1]
        )
    assert variant[1].startswith("t16_gemv_decode")


def test_context_does_not_touch_os_environ() -> None:
    before = dict(os.environ)
    runner = _FakeRunner()
    with resident_prefill_dispatch_session(
        runner,
        prompt_tokens=BULK_ROWS,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ):
        pass
    assert dict(os.environ) == before
