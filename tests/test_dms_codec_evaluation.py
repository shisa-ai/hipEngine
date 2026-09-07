"""Evaluation must not manufacture codec qualification or change serving defaults."""
import pytest

from hipengine.kvcache import dms
from tests.test_kvcache_dms_device_hip import _make_backend


def test_evaluation_factory_reports_unqualified_without_fabricated_scores():
    parent = _make_backend(num_layers=1, heads=2, dim=16, window=2, slots=128, device=False)
    backend = dms.create_dms_int8_evaluation_backend(
        retrofit=parent.retrofit, slots_per_layer=128, max_request_rows=1,
        max_pack_rows=32, device_payloads=False)
    assert backend.codec_qualification is None
    assert backend.codec == 'int8_per_token_head'
    assert backend.observability_snapshot()['backend']['codec_evaluation_only'] is True
    with pytest.raises(ValueError, match='requires artifact qualification'):
        dms.DMSCompactBackend(retrofit=parent.retrofit, codec='int8_per_token_head',
                             slots_per_layer=128, max_request_rows=1,
                             max_pack_rows=32, device_payloads=False)


def test_resident_backend_factory_defaults_to_bf16():
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    assert Qwen35GGUFResidentSession.__dataclass_fields__['dms_backend_factory'].default is None
