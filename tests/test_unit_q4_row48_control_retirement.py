"""Physical Q4 row48 policy no longer needs a process-global opt-out."""
import pytest
from hipengine.core.specdec2_scope import q4_t16_physical_extra_rowtiles_session
from hipengine.runtime import gguf_linear as linear


@pytest.mark.parametrize('value', ['0', 'invalid'])
def test_retired_row48_environment_does_not_change_physical_policy(monkeypatch, value):
    monkeypatch.setenv('HIPENGINE_GGUF_Q4_T16_DUAL_SILU_ROW48', value)
    linear._rowtile_variant_policy_env_cache.clear()
    with q4_t16_physical_extra_rowtiles_session(True):
        assert linear._q4_t16_physical_dual_silu_variant('hip_gfx1100', 36) == (
            'dense_dual_wmma_prefill_row48_bf16_bf16_out')
    assert linear._q4_t16_physical_dual_silu_variant('hip_gfx1100', 36) is None
