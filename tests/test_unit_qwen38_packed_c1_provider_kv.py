"""Provider committed-prefix addressing independent of rollback cursors."""
from types import SimpleNamespace as NS
import pytest


@pytest.mark.parametrize('changed', [False, True])
def test_provider_probe_compares_against_preproposal_capture(monkeypatch, changed):
    from scripts.qwen38_packed_c1_recovery import PrecommitProbe
    from scripts import qwen38_packed_c1_provider_kv as helper
    before = dict(position=3, buffers={'key': 'old', 'value': 'old'})
    after = dict(position=3, buffers={'key': 'bad' if changed else 'old', 'value': 'old'})
    snapshots = iter([before, after])
    monkeypatch.setattr(helper, 'snapshot_provider_kv', lambda *args: next(snapshots))
    probe = PrecommitProbe()
    checkpoint = object()
    assert probe.capture(lambda *args: checkpoint, object(), 7) is checkpoint
    probe.evidence['request_id'] = 7
    if changed:
        with pytest.raises(ValueError, match='committed KV prefix changed'): probe.assert_provider_prefix()
    else:
        probe.assert_provider_prefix()
        assert probe.evidence['provider_kv_planes'] == 2
        assert probe.evidence['provider_kv_position'] == 3


@pytest.mark.parametrize('fault', [None, 'missing_plane', 'slot', 'negative'])
def test_provider_prefix_uses_checkpoint_position_and_actual_slot(monkeypatch, fault):
    from hipengine.core import DType
    from scripts import qwen38_packed_c1_provider_kv as helper
    checkpoint = NS(request_id=7, slot=2, position=-1 if fault == 'negative' else 3)
    key, value = NS(ptr=10, nbytes=512), NS(ptr=20, nbytes=512)
    scratch = NS(full_key_caches=[key], full_value_caches=[value])
    if fault == 'missing_plane': scratch.full_key_caches = [None]
    session = NS(scratch=scratch, kv_storage_dtype=DType.BF16,
        _resident_slot_index=2, _resident_batch_owner=NS(_target_scratch_owner=NS(max_positions=16)),
        runner=NS(weights=NS(config=NS(head_count_kv=1, key_length=2))),
        runtime=NS(device_synchronize=lambda: None))
    executor = NS(_request_slots={7: 1 if fault == 'slot' else 2},
                  _batch_sessions=[None, None, session])
    calls = []
    monkeypatch.setattr(helper, 'hash_rows', lambda s, b, rows, stride:
                        calls.append((b.ptr, rows, stride)) or str(b.ptr))
    if fault:
        with pytest.raises(ValueError): helper.snapshot_provider_kv(executor, checkpoint)
    else:
        result = helper.snapshot_provider_kv(executor, checkpoint)
        assert result['position'] == 3
        assert calls == [(10, (32, 33, 34), 4), (20, (32, 33, 34), 4)]
        assert len(result['buffers']) == 2
