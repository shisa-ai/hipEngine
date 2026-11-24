"""Compact DMS configuration and observed store metadata are distinct."""
from types import SimpleNamespace

from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator, Qwen35GGUFResidentModelRunner
from hipengine.kvcache.dms_config import DMSConfig


def _generator(storage='int8_per_token_head'):
    gen = Qwen35GGUFBringupGenerator.__new__(Qwen35GGUFBringupGenerator)
    gen.model_path = 'model.gguf'
    gen.weight_index = SimpleNamespace(path='model.gguf')
    gen.dms_metadata_path = 'sidecar.json'
    gen.dms_config = DMSConfig('sidecar.json', prefill_mode='layer_outer')
    gen._prepared_kv_signature = (storage, 'uniform', 'fp32', 'per_token_head')
    return gen


def _session(storage='int8_per_token_head'):
    return SimpleNamespace(
        _dms_source=SimpleNamespace(config=SimpleNamespace(
            decision_source='sidecar', window_size=8192, target_compression_ratio=2.0,
            prefill_selection_mode='layer_outer',
        )),
        _dms_backend=SimpleNamespace(codec=storage, device_payloads_enabled=True),
        dms_decision_mode='sidecar', dms_prefill_mode='layer_outer',
    )


def test_compact_dms_capability_provenance_does_not_claim_default_bf16():
    gen = _generator()
    provenance = gen.kv_capability_provenance
    assert provenance['effective_kv_storage'] == 'int8_per_token_head'
    assert provenance['scope'] == 'compact_dms_configuration'
    assert provenance['status'] == 'not_applicable'
    assert not provenance['promotion_eligible']
    assert 'BF16/default' not in provenance['reason']


def test_unprepared_dms_configuration_does_not_claim_actual_storage():
    gen = _generator()
    gen._prepared_kv_signature = None
    assert gen.kv_capability_provenance['effective_kv_storage'] is None


def test_prepared_policy_uses_resolved_scales_and_loaded_sidecar_metadata():
    gen = _generator()
    gen._record_dms_serving_policy(_session())
    policy = gen._dms_serving_policy
    assert policy['metadata_path'] == 'sidecar.json'
    assert policy['route'] == 'compact_dms'
    assert policy['storage'] == 'int8_per_token_head'
    assert policy['scale_dtype'] == 'fp32'
    assert policy['scale_granularity'] == 'per_token_head'
    assert policy['window_size'] == 8192
    assert policy['target_compression_ratio'] == 2.0


def test_runtime_snapshot_distinguishes_configuration_active_and_last_observed():
    gen = _generator()
    session = _session()
    gen._record_dms_serving_policy(session)
    runner = Qwen35GGUFResidentModelRunner.__new__(Qwen35GGUFResidentModelRunner)
    runner.generator = gen
    first = runner._dms_retention_snapshot([])
    assert first['active_stores'] == []
    assert first['last_observed_store'] is None
    active = runner._dms_retention_snapshot([session])
    store = active['active_stores'][0]
    assert store['storage'] == 'int8_per_token_head'
    assert store['scale_dtype'] == 'fp32'
    assert store['source'] == 'observed_backend'
    assert store['no_dense_shadow'] is True
    runner._remember_dms_store(session)
    idle = runner._dms_retention_snapshot([])
    assert idle['active_stores'] == []
    assert idle['last_observed_store'] == store
    # A caller must not be able to mutate the remembered evidence.
    idle['last_observed_store']['storage'] = 'mutated'
    assert runner._dms_retention_snapshot([])['last_observed_store']['storage'] == 'int8_per_token_head'


def test_observed_backend_is_not_inferred_from_prepared_configuration():
    gen = _generator()
    gen._record_dms_serving_policy(_session())
    runner = Qwen35GGUFResidentModelRunner.__new__(Qwen35GGUFResidentModelRunner)
    runner.generator = gen
    snapshot = runner._dms_retention_snapshot([_session('bf16')])
    assert snapshot['configuration']['storage'] == 'int8_per_token_head'
    assert snapshot['active_stores'][0]['storage'] == 'bf16'
    assert snapshot['active_stores'][0]['scale_dtype'] is None


def test_release_records_observed_format_before_closing_the_store():
    gen = _generator()
    session = _session()
    gen._record_dms_serving_policy(session)
    runner = Qwen35GGUFResidentModelRunner.__new__(Qwen35GGUFResidentModelRunner)
    runner.generator = gen
    runner._prefix_cache = None
    runner._drop_prefix_snapshots_for_row = lambda request_id: None
    runner._close_c1_decode_graph = lambda row: None
    runner._resident_batch_owner = None
    closed = []
    session.close = lambda: closed.append(True)
    row = SimpleNamespace(resumable_prefill=None, lease=SimpleNamespace(session=session),
                          kv_allocation=None, request_id=7)
    runner._release_row_resources(row)
    assert closed == [True]
    assert row.lease is None
    snapshot = runner._dms_retention_snapshot([])
    assert snapshot['active_stores'] == []
    assert snapshot['last_observed_store']['storage'] == 'int8_per_token_head'


def test_dense_runner_does_not_report_compact_dms():
    runner = Qwen35GGUFResidentModelRunner.__new__(Qwen35GGUFResidentModelRunner)
    runner.generator = SimpleNamespace()
    assert runner._dms_retention_snapshot([]) is None
