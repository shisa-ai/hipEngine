"""Diagnostic concurrent C2-to-C1 service transition; no performance claim.

With --engine-boundary, submits two independent service children atomically and
checks them against independent HTTP AR output. Default horizons are D8/D24;
--cancel-peer uses two D24 children and cancels one after paired target execution.
The cancelled prefix comes from its terminal collector, not a public response.
--refill-peer admits a D8 child after a native C1 survivor, requires a new packed
pair, and checks request ownership and resident leases after all children finish.
--wide-refill tests eight initial children and seven refills at capacity eight,
with a real native C1 survivor between the two packed width-eight groups.
--precommit-failure injects after native C1 packed verification and checks unchanged
canonical target state, provider checkpoint/cursor restoration and AR recovery.
The original HTTP mode cannot batch unequal horizons and reproduces that failure.
No lease swap, HTTP lifecycle, postcommit recovery, provider live-KV rollback, EOS,
full numerical qualification, or promotion claim.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import faulthandler
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_transition(rows: list[dict]) -> dict:
    """Require a successful physical pair followed by its native C1 survivor."""
    pairs = []
    found = None
    for row in rows:
        ids = row['request_ids']
        if row.get('error'):
            raise ValueError('target failure invalidates transition')
        if len(ids) != len(row['resident_slots']) or len(ids) != len(row['scheduler_slots']):
            raise ValueError('incomplete slot identity')
        if row['packed_request_ids'] != ids or row['packed_group_sizes'] != [len(ids)]:
            raise ValueError('target did not execute the requested packed group')
        # A speculative singleton alongside an AR peer is not physical C1.
        if row['active_request_ids'] != ids:
            continue
        if len(ids) == 2:
            pairs.append(dict(zip(ids, row['resident_slots'], strict=True)))
        elif len(ids) == 1:
            if not row['native_c1']:
                raise ValueError('legacy C1 target is not a transition pass')
            for pair in reversed(pairs):
                if ids[0] in pair:
                    if pair[ids[0]] != row['resident_slots'][0]:
                        raise ValueError('survivor storage identity changed')
                    found = dict(survivor_request_id=ids[0],
                                 resident_slot=row['resident_slots'][0],
                                 scheduler_slot=row['scheduler_slots'][0])
                    break
    if found is None:
        raise ValueError('no real C2-to-C1 packed survivor transition')
    return found


def validate_response(expected: dict, actual: dict) -> None:
    """Compare authoritative IDs and public token accounting with true AR."""
    if expected['generated_ids'] != actual['generated_ids']:
        raise ValueError('concurrent generated IDs differ from independent AR')
    for row in (expected, actual):
        usage = row.get('usage')
        if not isinstance(usage, dict):
            raise ValueError('response omitted token usage')
        if usage.get('completion_tokens') != len(row['generated_ids']):
            raise ValueError('completion usage does not match generated IDs')
        prompt = usage.get('prompt_tokens')
        if not isinstance(prompt, int) or prompt < 1:
            raise ValueError('invalid prompt token usage')
        if usage.get('total_tokens') != prompt + len(row['generated_ids']):
            raise ValueError('total usage does not match prompt and output')
    if expected['usage'] != actual['usage']:
        raise ValueError('concurrent token usage differs from independent AR')


def install_lifecycle_evidence(capacity: int, widths=(1, 2), *, budget=3) -> None:
    """Extend only runtime diagnostic clones to cover the short retirement child."""
    from dataclasses import replace
    from hipengine.models import qwen35
    from scripts.qwen38_packet5_k4_watchdog_probe import _inject_k4_evidence_row
    for width in widths:
        _inject_k4_evidence_row(width, budget, capacity=capacity)
        plugin = qwen35.QWEN35_GGUF
        rows = plugin.speculative_mtp_serving_evidence
        diagnostic = replace(rows[-1], min_output_horizon_tokens=8,
                             evidence_key=f'lifecycle-d8-d24-n{capacity}-c{width}-k{budget}-diagnostic',
                             reason='unqualified engine lifecycle D8-D24 diagnostic',
                             evidence_artifacts=('scripts/qwen38_packed_c1_lifecycle.py',),
                             automatic_eligible=False)
        object.__setattr__(plugin, 'speculative_mtp_serving_evidence', rows[:-1] + (diagnostic,))


def combine_engine_intent(c1, c2, *, budget=3):
    """Bind separately resolved C1 and C2 permissions for this diagnostic only."""
    from dataclasses import replace
    import hashlib
    if (not c1.eligible or not c2.eligible or not c1.packed_c1_target
            or c1.max_realized_group_rows != 1 or c2.max_realized_group_rows < 2
            or not 1 <= budget <= 7
            or min(c1.max_candidate_count, c2.max_candidate_count) < budget
            or c1.strict_fallback_key != c2.strict_fallback_key):
        raise ValueError('separate C1 and C2 eligibility is required')
    sources = [c1.as_dict(), c2.as_dict()]
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return replace(c2, packed_c1_target=True, automatic_eligible=False,
                   max_candidate_count=min(c1.max_candidate_count, c2.max_candidate_count),
                   reason='diagnostic C1 and C2 ownership intersection',
                   evidence_key=f'diagnostic-c1-c2:{c1.evidence_key}:{c2.evidence_key}',
                   evidence_fingerprint=f'sha256:{digest}',
                   evidence_artifacts=tuple(dict.fromkeys(c1.evidence_artifacts + c2.evidence_artifacts)))


def resolve_engine_intents(llm, prompt: str, horizons=(8, 24), *, budget=3):
    intents, sources = [], []
    for horizon in horizons:
        decisions = [llm.resolve_speculative_mtp_serving_plan(
            realized_group_rows=width, sampling_mode='greedy_fast',
            context_tokens=llm.count_tokens(prompt), output_horizon_tokens=horizon)
            for width in (1, 2)]
        if any(d is None or not d.admitted for d in decisions):
            raise ValueError('engine diagnostic scope lacks C1 or C2 admission: ' +
                             json.dumps({'horizon': horizon, 'decisions': [
                                 None if d is None else d.as_dict() for d in decisions]}))
        intents.append(combine_engine_intent(*(d.static_eligibility for d in decisions), budget=budget))
        sources.append([d.as_dict() for d in decisions])
    return tuple(intents), sources


def submit_engine_pair(service, prompt: str, intents, *, horizons=(8, 24),
                       paired_ready=None, cancellation=None,
                       singleton_ready=None, refill=None, eos=None) -> list[dict]:
    """Admit both real children before polling either; no HTTP usage claim."""
    from hipengine.generation.registry import GenerationRequest
    if len(intents) != 2 or any(not i.eligible or not i.packed_c1_target for i in intents):
        raise ValueError('engine pair requires explicit combined ownership')
    requests = tuple(GenerationRequest(prompts=(prompt,), max_tokens=n, temperature=0.0,
                                       top_p=1.0, ignore_eos=False,
                                       speculative_mtp_static_eligibility=intent)
                     for n, intent in zip(horizons, intents, strict=True))
    if eos is not None:
        from scripts.qwen38_packed_c1_eos import configure_eos_request
        requests = (requests[0], configure_eos_request(requests[1], eos['oracle_ids'], index=eos['index']))
    handles = service.submit_speculative_children(requests)
    if len(handles) != 2:
        raise ValueError('engine did not return two independent handles')
    if singleton_ready is not None:
        from scripts.qwen38_packed_c1_refill import collect_refilled_pair
        outputs, evidence = collect_refilled_pair(service, handles, requests[0], singleton_ready)
        refill.update(evidence)
    elif paired_ready is None:
        outputs = [handle.result() for handle in handles]
    else:
        from scripts.qwen38_packed_c1_cancel import collect_cancelled_pair
        outputs, evidence = collect_cancelled_pair(handles, paired_ready)
        cancellation.update(evidence)
    if eos is not None:
        from scripts.qwen38_packed_c1_eos import validate_eos_terminal
        eos.update(validate_eos_terminal(handles[1], outputs[1], eos['oracle_ids'], index=eos['index']))
    if any(output.generated_token_ids is None for output in outputs):
        raise ValueError('engine omitted authoritative generated IDs')
    return [dict(generated_ids=list(output.generated_token_ids), usage=None,
                 route='engine_service_submission', mtp=None) for output in outputs]


def close_and_report(llm, output: Path, result: dict) -> None:
    """Preserve diagnostics but never publish success after failed teardown."""
    result['teardown_error'] = None
    try:
        if llm is not None:
            llm.close()
    except BaseException as error:
        result['passed'] = False
        result['teardown_error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/models/gguf/Qwen3.8-27B-Q4_K_M.gguf')
    parser.add_argument('--capacity', type=int, choices=(2, 8), default=8)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=int, choices=range(1,8), default=3)
    parser.add_argument('--engine-boundary', action='store_true',
                        help='Submit unequal-horizon children atomically below HTTP batching')
    parser.add_argument('--cancel-peer', action='store_true',
                        help='Cancel one D24 engine child after a successful paired target')
    parser.add_argument('--refill-peer', action='store_true',
                        help='Admit a new D8 child after the D24 child becomes a native C1 survivor')
    parser.add_argument('--wide-refill', action='store_true',
                        help='Test eight initial children, their C1 survivor, then seven new peers')
    parser.add_argument('--precommit-failure', action='store_true',
                        help='Inject one native C1 failure after packed verification, before acceptance')
    parser.add_argument('--eos-survivor', action='store_true',
                        help='Configure EOS at a first-occurrence AR token after index 11 and check the collector')
    parser.add_argument('--eos-state', action='store_true',
                        help='Check canonical isolation and selected-state bytes for EOS C1 cycles')
    args = parser.parse_args()
    if args.budget != 3 and not args.eos_state:
        parser.error('nondefault --budget requires --eos-state')
    if args.eos_state and not args.eos_survivor:
        parser.error('--eos-state requires --eos-survivor')
    if args.eos_survivor and (not args.engine_boundary or args.wide_refill
                             or args.cancel_peer or args.refill_peer or args.precommit_failure):
        parser.error('--eos-survivor requires engine boundary and no other scenario')
    if args.precommit_failure and (not args.engine_boundary or args.wide_refill
                                  or args.cancel_peer or args.refill_peer):
        parser.error('--precommit-failure requires engine boundary and no other scenario')
    if args.wide_refill and (not args.engine_boundary or args.capacity != 8
                            or args.cancel_peer or args.refill_peer):
        parser.error('--wide-refill requires --engine-boundary --capacity 8 and no other scenario')
    if (args.cancel_peer or args.refill_peer) and not args.engine_boundary:
        parser.error('--cancel-peer and --refill-peer require --engine-boundary')
    if args.cancel_peer and args.refill_peer:
        parser.error('choose cancellation or refill, not both')
    horizons = (24, 24) if args.cancel_peer else (8, 24)
    from scripts import gguf_mtp_c1c8_server_bench as bench
    from hipengine.generation import qwen35_gguf_mtp2 as mtp2
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    os.environ['HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS'] = '1'
    install_lifecycle_evidence(args.capacity, range(1, 9) if args.wide_refill else (1, 2), budget=args.budget)
    traces = []
    paired_ready = None
    singleton_ready = None
    current = ContextVar('lifecycle_target', default=None)
    execute = mtp2.Qwen35GGUFMTP2Adapter._execute_target_frontier_batch
    verify = Qwen35GGUFResidentSession.verify_target_blocks_batch
    legacy = mtp2.Qwen35GGUFTransactionalVerifier
    from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNExecutor
    from scripts.qwen38_packed_c1_recovery import PrecommitProbe, InjectedPrecommitFailure
    capture_checkpoint = Qwen35GGUFNextNExecutor.capture_request_checkpoint
    restore_checkpoint = Qwen35GGUFNextNExecutor.restore_request_checkpoint
    recover_failure = mtp2.Qwen35GGUFMTP2Adapter.recover_cycle_failure
    recovery = None
    eos_probes = {}
    commit_states = Qwen35GGUFResidentSession._commit_deferred_packed_verify_states_batch_device

    def checked_eos_commit(owner, results, sessions, **kwargs):
        context = current.get()
        probe = None if context is None else eos_probes.get(id(context[0]))
        if probe is not None:
            return probe.commit(commit_states, owner, results, sessions, **kwargs)
        return commit_states(owner, results, sessions, **kwargs)

    def captured(executor, request_id):
        if recovery is not None:
            return recovery.capture(capture_checkpoint, executor, request_id)
        return capture_checkpoint(executor, request_id)

    def restored(executor, checkpoint):
        if recovery is not None and recovery.evidence['injected']:
            recovery.evidence.setdefault('restore_request_ids', []).append(int(checkpoint.request_id))
        if (recovery is not None and recovery.evidence['injected']
                and checkpoint.request_id == recovery.evidence['request_id']):
            return recovery.restore(restore_checkpoint, executor, checkpoint)
        return restore_checkpoint(executor, checkpoint)

    def recovered(adapter, plan, error):
        if recovery is not None and type(error) is InjectedPrecommitFailure:
            if list(plan.speculative_request_ids) != [recovery.evidence['request_id']]:
                raise ValueError('recovery changed request identity')
            recovery.assert_target()
            result = recover_failure(adapter, plan, error)
            recovery.evidence['recovered'] = bool(result)
            return result
        return recover_failure(adapter, plan, error)

    def target(adapter, plan, *positional, **kwargs):
        ids = list(plan.speculative_request_ids)
        sessions = [adapter.owner._row(rid).slot.session for rid in ids]
        record = dict(request_ids=ids, active_request_ids=list(plan.request_ids),
                      scheduler_slots=[int(plan.resident_slots[plan.request_ids.index(rid)]) for rid in ids],
                      resident_slots=[int(getattr(s, '_resident_slot_index', 0) or 0) for s in sessions],
                      native_c1=len(ids) == 1 and adapter._physical_c1_request(ids[0]),
                      packed_request_ids=[], packed_group_sizes=[], error=None)
        if args.eos_state and record['native_c1']:
            row = adapter.owner._row(ids[0])
            record['eos_logical_rows'] = 1 + int(plan.candidate_counts[plan.request_ids.index(ids[0])])
            record['eos_token_id'] = row.request.eos_token_id
            record['remaining_decode'] = row.request.max_tokens - len(row.slot.generated_ids)
        traces.append(record)
        token = current.set((record, sessions))
        try:
            result = execute(adapter, plan, *positional, **kwargs)
            if args.eos_state and record.get('eos_token_id') is not None and not record.get('eos_state', {}).get('passed'):
                raise ValueError('EOS cycle omitted selected-state verification')
            if (paired_ready is not None and len(ids) == 2
                    and record['active_request_ids'] == ids
                    and record['packed_request_ids'] == ids
                    and record['packed_group_sizes'] == [2]):
                paired_ready.set()
            if (singleton_ready is not None and len(ids) == 1
                    and record['active_request_ids'] == ids and record['native_c1']):
                if args.wide_refill:
                    if any(len(t['request_ids']) == 8 and ids[0] in t['request_ids']
                           and t['active_request_ids'] == t['request_ids']
                           and t['packed_request_ids'] == t['request_ids']
                           and t['packed_group_sizes'] == [8] and not t['error']
                           for t in traces[start:-1]):
                        singleton_ready.set()
                else:
                    validate_transition(traces[start:])
                    singleton_ready.set()
            return result
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            current.reset(token)

    def packed(owner, jobs, **kwargs):
        context = current.get()
        inject = (recovery is not None and not recovery.evidence['injected']
                  and context is not None and context[0]['native_c1']
                  and context[0]['active_request_ids'] == context[0]['request_ids'])
        if inject:
            recovery.prepare(context[1][0], context[0]['request_ids'][0])
        probe = None
        if args.eos_state and context is not None and context[0].get('eos_token_id') is not None:
            from scripts.qwen38_packed_c1_eos_state import EosStateProbe
            from scripts.qwen38_packed_c1_logits import _read_device
            import numpy as np
            if len(jobs) != 1 or not kwargs.get('device_result'):
                raise ValueError('EOS state check requires one native device-result job')
            tokens = list(jobs[0]['input_token_ids'])
            device = jobs[0].get('candidate_token_ids_device')
            if device is not None:
                tokens[1:] = _read_device(device.ptr,(len(tokens)-1,),np.int32,owner.runtime).tolist()
            tokens = tokens[:context[0]['eos_logical_rows']]
            probe = EosStateProbe(context[1][0], eos=context[0]['eos_token_id'],
                                  remaining=context[0]['remaining_decode'])
            eos_probes[id(context[0])] = probe
            context[0]['eos_state'] = probe.evidence
        result = verify(owner, jobs, **kwargs)
        if probe is not None:
            probe.verified(owner, result[0], tokens)
        if context is not None:
            record, sessions = context
            record['packed_group_sizes'].append(len(jobs))
            for job in jobs:
                matches = [i for i, s in enumerate(sessions) if s is job['session']]
                if len(matches) != 1:
                    raise ValueError('packed job does not own a traced request session')
                record['packed_request_ids'].append(record['request_ids'][matches[0]])
        if inject:
            recovery.inject()
        return result

    def forbidden(*positional, **kwargs):
        raise AssertionError('native lifecycle invoked legacy target')

    mtp2.Qwen35GGUFMTP2Adapter._execute_target_frontier_batch = target
    Qwen35GGUFResidentSession.verify_target_blocks_batch = packed
    mtp2.Qwen35GGUFTransactionalVerifier = forbidden
    if args.eos_state:
        Qwen35GGUFResidentSession._commit_deferred_packed_verify_states_batch_device = checked_eos_commit
    if args.precommit_failure:
        Qwen35GGUFNextNExecutor.capture_request_checkpoint = captured
        Qwen35GGUFNextNExecutor.restore_request_checkpoint = restored
        mtp2.Qwen35GGUFMTP2Adapter.recover_cycle_failure = recovered
    cells = []
    eos = None
    actual = None
    llm = None
    passed = False
    faulthandler.dump_traceback_later(900, exit=True)
    try:
        llm = bench.LLM(args.model, backend='hip_gfx1100', execution_profile='production',
                        max_active_requests=args.capacity, max_sequence_length=1024,
                        speculative_candidate_budget=args.budget)
        llm.prepare(max_sequence_length=1024)
        app = bench.create_app(bench.ServerConfig(
            model=args.model, backend='hip_gfx1100', quant='gguf_q4_k_m',
            served_model_name='lifecycle', eager_load=False, generation_batch_window_ms=20,
            max_context_tokens=1024, max_active_requests=args.capacity,
            speculative_mtp_serving='opt_in', speculative_candidate_budget=args.budget,
            shutdown_grace_seconds=5.0), llm=llm)
        suite = bench.load_prompt_suite(ROOT / 'benchmarks/prompts/mtpbench-code-general-ja.jsonl')
        with bench.TestClient(app) as client:
            for prompt in suite:
                # Independent non-MTP requests are the output oracle.
                expected = [bench._request(client, model='lifecycle', prompt=prompt['rendered_prompt'],
                            max_tokens=n, mtp=False, barrier=threading.Barrier(1))
                            for n in horizons]
                start = len(traces)
                recovery = PrecommitProbe() if args.precommit_failure else None
                paired_ready = threading.Event() if args.cancel_peer else None
                cancellation = {} if args.cancel_peer else None
                singleton_ready = threading.Event() if args.refill_peer or args.wide_refill else None
                refill = {} if args.refill_peer else None
                drain = None
                eos = None
                if args.eos_survivor:
                    oracle_ids = expected[1]['generated_ids']
                    from scripts.qwen38_packed_c1_eos import select_survivor_eos_index
                    index = select_survivor_eos_index(oracle_ids, budget=args.budget)
                    eos = dict(oracle_ids=oracle_ids, index=index)
                if args.refill_peer:
                    expected.append(expected[0])  # Same prompt and D8 as the first independent AR arm.
                intent_sources = None
                if args.wide_refill:
                    from scripts.qwen38_packed_c1_wide_refill import resolve_wide_intents, submit_wide_refill
                    expected = [expected[0]] * 7 + [expected[1]] + [expected[0]] * 7
                    intents, intent_sources = resolve_wide_intents(llm, prompt['rendered_prompt'])
                    actual, refill = submit_wide_refill(llm._get_text_generator(),
                        prompt['rendered_prompt'], intents, singleton_ready)
                elif args.engine_boundary:
                    intents, intent_sources = resolve_engine_intents(llm, prompt['rendered_prompt'], horizons, budget=args.budget)
                    actual = submit_engine_pair(llm._get_text_generator(), prompt['rendered_prompt'], intents,
                                                horizons=horizons, paired_ready=paired_ready,
                                                cancellation=cancellation,
                                                singleton_ready=singleton_ready, refill=refill, eos=eos)
                else:
                    barrier = threading.Barrier(3)
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futures = [pool.submit(bench._request, client, model='lifecycle',
                                   prompt=prompt['rendered_prompt'], max_tokens=n, mtp=True,
                                   barrier=barrier) for n in (8, 24)]
                        barrier.wait(timeout=30)
                        actual = [future.result() for future in futures]
                trace = traces[start:]
                if args.wide_refill:
                    from scripts.qwen38_packed_c1_wide_refill import validate_wide_refill
                    transition = validate_wide_refill(trace, refill['initial_request_ids'],
                                                      refill['refill_request_ids'])
                elif args.precommit_failure:
                    from scripts.qwen38_packed_c1_recovery import validate_recovery
                    transition = validate_recovery(trace, recovery.evidence)
                    if not recovery.evidence.get('provider_kv_restored') or recovery.evidence.get('provider_kv_position', 0) < 1:
                        raise ValueError('provider live KV recovery is unverified')
                else:
                    transition = validate_transition(trace)
                if args.refill_peer or args.wide_refill or args.precommit_failure or args.eos_survivor:
                    from scripts.qwen38_packed_c1_drain import validate_request_drain
                    if args.refill_peer:
                        from scripts.qwen38_packed_c1_refill import validate_refill_transition
                        transition = validate_refill_transition(trace, refill)
                    snapshot = llm._get_text_generator().live_loop_snapshot()
                    validate_request_drain(snapshot, capacity=args.capacity)
                    drain = dict(engine_service=snapshot['engine_service'],
                                 loop={k: snapshot['loop'][k] for k in ('requests', 'physical_bucket')},
                                 runner=dict(model_runner={k: snapshot['runner']['model_runner'][k]
                                     for k in ('capacity', 'active_requests', 'active_request_ids', 'available_sessions')}))
                if args.eos_state and not any(t.get('eos_state', {}).get('passed') and
                                                  t['eos_state'].get('terminal') for t in trace):
                    raise ValueError('no verified native terminal EOS state commit')
                if args.eos_survivor:
                    if eos['backend_request_id'] != transition['survivor_request_id']:
                        raise ValueError('EOS collector does not belong to the traced survivor')
                    exact = (actual[0]['generated_ids'] == expected[0]['generated_ids']
                             and actual[1]['generated_ids'] == expected[1]['generated_ids'][:eos['index'] + 1])
                elif args.cancel_peer:
                    from scripts.qwen38_packed_c1_cancel import validate_cancel_outputs
                    validate_cancel_outputs(expected, actual, cancellation, transition)
                    exact = True  # Cancelled child is an exact prefix; peer is full exact.
                else:
                    exact = all(a['generated_ids'] == b['generated_ids'] for a, b in zip(expected, actual, strict=True))
                engaged = (True if args.engine_boundary else
                           all(bench._mtp_engaged(r['route'], r['mtp']) for r in actual))
                if not args.engine_boundary:
                    for oracle, candidate in zip(expected, actual, strict=True):
                        validate_response(oracle, candidate)
                cell = dict(prompt_id=prompt['id'], category=prompt['category'], exact=exact,
                            usage_exact=None if args.engine_boundary else True,
                            engaged=engaged, transition=transition, trace=trace,
                            intent_sources=intent_sources, cancellation=cancellation,
                            refill=refill, request_drain=drain, eos=eos,
                            recovery=None if recovery is None else recovery.evidence,
                            responses=[{k: r[k] for k in ('generated_ids', 'usage', 'route', 'mtp')}
                                       for r in actual],
                            oracle_responses=[{k: r[k] for k in ('generated_ids', 'usage')}
                                              for r in expected])
                cells.append(cell)
                print(json.dumps({k: v for k, v in cell.items() if k not in ('trace', 'responses', 'oracle_responses', 'intent_sources')}), flush=True)
                if not exact or not engaged:
                    raise ValueError('concurrent outputs or engagement failed')
        if args.eos_state and not any(t.get('eos_logical_rows') == args.budget + 1
                                      and t.get('eos_state', {}).get('passed') for t in traces):
            raise ValueError('requested EOS frontier depth never executed')
        passed = len(cells) == len(suite) == 10
    finally:
        try:
            close_and_report(llm, args.output, dict(
                diagnostic_only=True, performance_claim=False, full_profile_qualification=False,
                passed=passed, capacity=args.capacity, budget=args.budget, horizons=list(horizons),
                cancellation_requested=args.cancel_peer, refill_requested=args.refill_peer,
                wide_refill_requested=args.wide_refill, precommit_failure_requested=args.precommit_failure,
                eos_requested=args.eos_survivor, eos_state_requested=args.eos_state,
                last_eos=eos, last_responses=actual,
                last_recovery=None if recovery is None else recovery.evidence,
                boundary='engine_service' if args.engine_boundary else 'http',
                cells=cells, all_target_traces=traces))
        finally:
            mtp2.Qwen35GGUFMTP2Adapter._execute_target_frontier_batch = execute
            Qwen35GGUFResidentSession.verify_target_blocks_batch = verify
            mtp2.Qwen35GGUFTransactionalVerifier = legacy
            Qwen35GGUFResidentSession._commit_deferred_packed_verify_states_batch_device = commit_states
            Qwen35GGUFNextNExecutor.capture_request_checkpoint = capture_checkpoint
            Qwen35GGUFNextNExecutor.restore_request_checkpoint = restore_checkpoint
            mtp2.Qwen35GGUFMTP2Adapter.recover_cycle_failure = recover_failure
            faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
