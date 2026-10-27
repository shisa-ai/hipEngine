"""A named padded-tail production manifest must not depend on ambient rollback."""
from hipengine.generation import qwen4_exp_profiles as profiles


def test_named_tail_selection_ignores_legacy_environment(monkeypatch):
    for value in ('dense_wide256_f16in_f32_f32_out', 'unknown'):
        monkeypatch.setenv('HIPENGINE_QWEN4_EXP_DENSE_WIDE_TAIL_VARIANT', value)
        assert profiles._dense_wide_tail_selection(production=True).selected_variant == (
            'dense_wide256_padded_f16in_f32_f32_out')
