"""The inert GDN decode-layer key must not survive as profile process state."""
import os
from pathlib import Path
import pytest
from hipengine.execution_profiles import ExecutionProfile
from scripts import qwen4exp_layer2_profile_gate as gate
from scripts import qwen4exp_decode_sync_ab as sync
from scripts import qwen4exp_profile_gap as gap
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind, _isolate

ENV = 'HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS'


@pytest.mark.parametrize('profile', [ExecutionProfile.PRODUCTION, ExecutionProfile.STRICT])
def test_binder_leaves_unused_caller_key_alone(monkeypatch, profile):
    monkeypatch.setenv(ENV, '999')
    gen, runner = _bind(_resolve(profile))
    assert os.environ[ENV] == '999'
    assert gen.gdn_decode_rowsplit == runner.gdn_decode_rowsplit


def test_no_diagnostic_advertises_the_dead_key():
    assert all(ENV not in spec.environment for spec in gate.CANDIDATES.values())
    assert ENV not in gap.ROUTE_ENV_KEYS
    with pytest.raises(ValueError, match='no runtime reader'):
        gap._parse_overrides([ENV + '=0'])
    assert ENV not in Path(sync.__file__).read_text()
