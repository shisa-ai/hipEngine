"""Campaign vector prerequisites must not mutate ambient prepack input."""
from types import SimpleNamespace
import pytest
from scripts.qwen4exp_halo_box_campaign_ab import (
    mmq_vector_prerequisite, mmq_vector_control_record, mmq_token64_control_record,
)

ENV = 'HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK'


@pytest.mark.parametrize('package,packed', [('q8-mmq-vec4', True), ('q8-mmq-raw-vector', False)])
def test_owner_precondition_preserves_environment_and_restores_exception(package, packed):
    env = {ENV: '0' if packed else '1'}
    calls = []
    owner = SimpleNamespace(q8_mmq_prefill_enabled=None, q8_mmq_prepack_enabled=None,
                            _q8_mmq_buffers=(), _q8_mmq_weight_sidecars=None)
    def configure():
        calls.append((owner.q8_mmq_prefill_enabled, owner.q8_mmq_prepack_enabled))
        owner._q8_mmq_buffers = (object(),)
        if owner.q8_mmq_prepack_enabled:
            owner._q8_mmq_weight_sidecars = SimpleNamespace(mapping={(1,2,3): (4,5)})
    owner.configure_mmq_prefill_resources = configure
    with pytest.raises(RuntimeError, match='injected'):
        with mmq_vector_prerequisite(owner, package, environment=env) as record:
            assert env == {ENV: '0' if packed else '1'}
            assert calls == [(True, packed)]
            assert record['prerequisite_owner']['q8_mmq_prepack_enabled'] is packed
            raise RuntimeError('injected')
    assert owner.q8_mmq_prepack_enabled is owner.q8_mmq_prefill_enabled is None
    assert env == {ENV: '0' if packed else '1'}


@pytest.mark.parametrize('package,packed', [('q8-mmq-vec4', True), ('q8-mmq-raw-vector', False), ('mmq-token64', False)])
def test_record_names_owner_prepack_not_environment(package, packed):
    owner = SimpleNamespace(q8_mmq_prepack_enabled=None)
    record = (mmq_token64_control_record(owner) if package == 'mmq-token64'
              else mmq_vector_control_record(package, owner))
    assert record['prerequisite_environment'] == {}
    assert record['prerequisite_owner']['q8_mmq_prepack_enabled'] is packed
    assert record['prerequisite_previous_owner']['q8_mmq_prepack_enabled'] is None
