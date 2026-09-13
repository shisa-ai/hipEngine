"""Controller admission and real target-trace checks for survivor refill."""
from __future__ import annotations


def collect_refilled_pair(service, handles, refill_request, singleton_ready):
    if len(handles) != 2:
        raise ValueError('refill requires two initial handles')
    if not singleton_ready.wait(timeout=60):
        raise TimeoutError('no packed C1 survivor before refill')
    initial_ids = [h.backend_request_id for h in handles]
    added = service.submit_speculative_children((refill_request,))
    if len(added) != 1:
        raise ValueError('refill did not return exactly one child')
    outputs = [h.result(timeout=120) for h in (*handles, *added)]
    refill_id = added[0].backend_request_id
    if len(set((*initial_ids, refill_id))) != 3:
        raise ValueError('refill reused a request identity')
    return outputs, dict(initial_request_ids=initial_ids, refill_request_id=refill_id)


def validate_refill_transition(rows, evidence):
    initial = evidence['initial_request_ids']
    refill = evidence['refill_request_id']
    if len(initial) != 2 or len(set((*initial, refill))) != 3:
        raise ValueError('refill requires three independent request IDs')
    survivor = initial[1]  # D24 child; the first child has horizon D8.
    slot = None
    singleton = False
    found = None
    for row in rows:
        ids = row['request_ids']
        if (row.get('error') or row['packed_request_ids'] != ids
                or row['packed_group_sizes'] != [len(ids)]
                or len(row['resident_slots']) != len(ids)
                or len(set(ids)) != len(ids)
                or len(set(row['resident_slots'])) != len(ids)):
            raise ValueError('invalid packed execution during refill')
        if slot is not None and survivor in ids:
            if row['resident_slots'][ids.index(survivor)] != slot:
                raise ValueError('refill moved survivor storage')
        if row['active_request_ids'] != ids:
            continue
        if set(ids) == set(initial):
            slot = row['resident_slots'][ids.index(survivor)]
        elif ids == [survivor] and slot is not None:
            if not row['native_c1']:
                raise ValueError('refill survivor used legacy C1')
            singleton = True
        elif set(ids) == {survivor, refill} and singleton:
            found = dict(survivor_request_id=survivor, resident_slot=slot,
                         refill_request_id=refill)
    if found is None:
        raise ValueError('no real C2-to-C1-to-C2 survivor refill')
    return found
