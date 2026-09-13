"""Verifier qualification must not silently enable AR prefill arithmetic."""

from types import SimpleNamespace

from hipengine.runtime import qwen35_gguf_runner as runner
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    q6_dense_integer_mmq_workspace,
)


def test_ar_prefill_does_not_allocate_or_enter_integer_mmq():
    def unexpected_allocation():
        raise AssertionError("AR prefill entered verifier-only MMQ")

    session = SimpleNamespace(
        use_q6_integer_mmq=True,
        _ensure_prefill_f16_staging_buffer=unexpected_allocation,
    )
    with runner.Qwen35GGUFResidentSession._q6_integer_mmq_context(session):
        assert q6_dense_integer_mmq_workspace() is None


def test_explicit_verifier_scope_keeps_registered_mmq_workspace():
    session = SimpleNamespace(
        use_q6_integer_mmq=True, _q6_integer_mmq_library="cached-library",
        _ensure_prefill_f16_staging_buffer=lambda: SimpleNamespace(ptr=4096, nbytes=65536),
    )
    with runner.Qwen35GGUFResidentSession._q6_integer_mmq_context(
        session, target_verifier=True
    ):
        assert q6_dense_integer_mmq_workspace().ptr == 4096
    assert q6_dense_integer_mmq_workspace() is None
