"""Fail closed on live request ownership after engine lifecycle completion."""
import pytest


def snapshot():
    return dict(engine_service=dict(active_children=0, command_queue_depth=0),
                loop=dict(requests=dict(pending=0, active=0, admitted_current=0),
                          physical_bucket=dict(capacity=2, occupied_slots=0, free_slots=2,
                                               active_mask=[False, False])),
                runner=dict(model_runner=dict(capacity=2, active_requests=0,
                    active_request_ids=[], available_sessions=2)))


@pytest.mark.parametrize('fault', [None, 'child', 'command', 'pending', 'active',
    'admitted', 'occupied', 'free', 'mask', 'runner', 'ids', 'lease', 'missing'])
def test_request_drain_rejects_residual_ownership(fault):
    from scripts.qwen38_packed_c1_drain import validate_request_drain
    row = snapshot()
    if fault == 'child': row['engine_service']['active_children'] = 1
    elif fault == 'command': row['engine_service']['command_queue_depth'] = 1
    elif fault == 'pending': row['loop']['requests']['pending'] = 1
    elif fault == 'active': row['loop']['requests']['active'] = 1
    elif fault == 'admitted': row['loop']['requests']['admitted_current'] = 1
    elif fault == 'occupied': row['loop']['physical_bucket']['occupied_slots'] = 1
    elif fault == 'free': row['loop']['physical_bucket']['free_slots'] = 1
    elif fault == 'mask': row['loop']['physical_bucket']['active_mask'][0] = True
    elif fault == 'runner': row['runner']['model_runner']['active_requests'] = 1
    elif fault == 'ids': row['runner']['model_runner']['active_request_ids'] = [4]
    elif fault == 'lease': row['runner']['model_runner']['available_sessions'] = 1
    elif fault == 'missing': del row['runner']
    if fault is None:
        validate_request_drain(row, capacity=2)
    else:
        with pytest.raises(ValueError):
            validate_request_drain(row, capacity=2)
