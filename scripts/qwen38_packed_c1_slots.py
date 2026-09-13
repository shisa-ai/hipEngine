"""Diagnostic-only lease choice for sequential C1 physical-slot coverage.

Uses the actual service owner's existing free leases. It does not change session
pointers, page tables, dispatch, acceptance, or output. Not a concurrency test.
"""


def acquire_slot(owner, slot: int):
    if not 0 <= slot < int(owner.capacity):
        raise ValueError('diagnostic slot outside resident capacity')
    indices = [i for i, lease in enumerate(owner._available)
               if int(getattr(lease.session, '_resident_slot_index', 0) or 0) == slot]
    if len(indices) != 1:
        raise ValueError('diagnostic slot requires exactly one free lease')
    return owner._available.pop(indices[0])
