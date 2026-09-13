"""Request-ownership drain checks, not a claim that cached GPU memory is freed."""
from __future__ import annotations


def validate_request_drain(snapshot: dict, *, capacity: int) -> None:
    """Require empty service/scheduler ownership and return of resident leases."""
    try:
        service = snapshot['engine_service']
        requests = snapshot['loop']['requests']
        bucket = snapshot['loop']['physical_bucket']
        runner = snapshot['runner']['model_runner']
        empty = (service['active_children'], service['command_queue_depth'],
                 requests['pending'], requests['active'], requests['admitted_current'],
                 bucket['occupied_slots'], runner['active_requests'])
        if any(value != 0 for value in empty):
            raise ValueError('request ownership remains after completion')
        if (capacity < 1 or bucket['capacity'] != capacity or runner['capacity'] != capacity
                or bucket['free_slots'] != capacity or runner['available_sessions'] != capacity
                or runner['active_request_ids'] != []
                or list(bucket['active_mask']) != [False] * capacity):
            raise ValueError('resident leases or scheduler slots did not drain')
    except (KeyError, TypeError) as error:
        raise ValueError('incomplete request drain snapshot') from error
