"""Actual width-eight target transition checks for the engine diagnostic."""
from __future__ import annotations


def resolve_wide_intents(llm, prompt):
    """Resolve every reached width independently; never promote diagnostic rows."""
    from dataclasses import replace
    import hashlib
    import json
    intents, sources = [], []
    for horizon in (8, 24):
        decisions = [llm.resolve_speculative_mtp_serving_plan(
            realized_group_rows=w, sampling_mode='greedy_fast',
            context_tokens=llm.count_tokens(prompt), output_horizon_tokens=horizon)
            for w in range(1, 9)]
        if any(d is None or not d.admitted for d in decisions):
            raise ValueError('wide refill lacks admission: ' + json.dumps([
                None if d is None else d.as_dict() for d in decisions]))
        eligibility = [d.static_eligibility for d in decisions]
        if (not eligibility[0].packed_c1_target or eligibility[0].max_realized_group_rows != 1
                or any(not e.eligible or e.max_candidate_count < 3
                       or e.max_realized_group_rows < w
                       or e.strict_fallback_key != eligibility[0].strict_fallback_key
                       for w, e in enumerate(eligibility, 1))):
            raise ValueError('wide refill ownership does not cover every width')
        source = [d.as_dict() for d in decisions]
        digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        intents.append(replace(eligibility[-1], packed_c1_target=True,
            automatic_eligible=False, max_candidate_count=min(e.max_candidate_count for e in eligibility),
            reason='unqualified diagnostic intersection of C1 through C8 ownership',
            evidence_key='diagnostic-wide-refill:' + digest,
            evidence_fingerprint='sha256:' + digest,
            evidence_artifacts=tuple(dict.fromkeys(a for e in eligibility for a in e.evidence_artifacts))))
        sources.append(source)
    return tuple(intents), sources


def submit_wide_refill(service, prompt, intents, singleton_ready):
    """Submit eight children, then seven new children after the survivor event."""
    from hipengine.generation.registry import GenerationRequest
    short, long = [GenerationRequest(prompts=(prompt,), max_tokens=horizon,
        temperature=0.0, top_p=1.0, ignore_eos=False,
        speculative_mtp_static_eligibility=intent)
        for horizon, intent in zip((8, 24), intents, strict=True)]
    handles = service.submit_speculative_children((short,) * 7 + (long,))
    if len(handles) != 8:
        raise ValueError('wide admission did not return eight children')
    if not singleton_ready.wait(timeout=60):
        raise TimeoutError('no native C1 survivor after the initial C8 group')
    initial_ids = [h.backend_request_id for h in handles]
    added = service.submit_speculative_children((short,) * 7)
    if len(added) != 7:
        raise ValueError('wide refill did not return seven children')
    outputs = [h.result(timeout=120) for h in (*handles, *added)]
    if any(o.generated_token_ids is None for o in outputs):
        raise ValueError('wide refill omitted authoritative output IDs')
    evidence = dict(initial_request_ids=initial_ids,
                    refill_request_ids=[h.backend_request_id for h in added])
    return [dict(generated_ids=list(o.generated_token_ids), usage=None,
                 route='engine_service_submission', mtp=None) for o in outputs], evidence


def validate_wide_refill(rows, initial_ids, refill_ids):
    if (len(initial_ids) != 8 or len(refill_ids) != 7
            or len(set((*initial_ids, *refill_ids))) != 15):
        raise ValueError('wide refill requires eight initial and seven new identities')
    survivor = initial_ids[-1]
    initial = set(initial_ids)
    refilled = {survivor, *refill_ids}
    slot = None
    singleton_seen = False
    complete = False
    for row in rows:
        ids = row['request_ids']
        slots = row['resident_slots']
        if (row.get('error') or len(ids) != len(set(ids))
                or len(ids) != len(slots) or len(slots) != len(set(slots))
                or row['packed_request_ids'] != ids
                or row['packed_group_sizes'] != [len(ids)]):
            raise ValueError('invalid packed ownership in wide refill')
        if slot is not None and survivor in ids and slots[ids.index(survivor)] != slot:
            raise ValueError('wide refill moved survivor storage')
        if row['active_request_ids'] != ids:
            continue
        if set(ids) == initial:
            slot = slots[ids.index(survivor)]
        elif ids == [survivor] and slot is not None:
            if not row['native_c1']:
                raise ValueError('wide survivor used legacy target')
            singleton_seen = True
        elif set(ids) == refilled and singleton_seen:
            complete = True
    if not complete:
        raise ValueError('no actual C8-to-C1-to-C8 refill transition')
    return dict(survivor_request_id=survivor, resident_slot=slot,
                initial_width=8, refill_width=8)
