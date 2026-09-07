"""Native C1 diagnostic provider restoration checks before checkpoint release."""
from __future__ import annotations
import hashlib
import numpy as np
from scripts.gguf_packed_ar_state_oracle import _device_hash


class InjectedPrecommitFailure(RuntimeError):
    """One deliberate failure after packed target work, before acceptance."""


def validate_recovery(rows, evidence):
    from scripts.qwen38_packed_c1_lifecycle import validate_transition
    if (not evidence['injected'] or evidence['provider_restores'] != 1
            or not evidence['target_unchanged'] or not evidence['recovered']):
        import json
        raise ValueError('incomplete precommit recovery evidence: ' + json.dumps(evidence, sort_keys=True))
    errors = [r for r in rows if r.get('error')]
    if (len(errors) != 1 or errors[0]['request_ids'] != [evidence['request_id']]
            or errors[0]['error'] != 'InjectedPrecommitFailure: diagnostic'):
        raise ValueError('unexpected failure in recovery trace')
    # Preserve the raw error; only the deliberately injected failure is expected.
    return validate_transition([dict(r, error=None) if r is errors[0] else r for r in rows])


class PrecommitProbe:
    def __init__(self):
        self.evidence = dict(request_id=None, injected=False, provider_restores=0,
                             target_unchanged=False, recovered=False)
        self.session = None
        self.before = None
        self.provider_prefixes = {}

    def capture(self, original, executor, request_id):
        from scripts.qwen38_packed_c1_provider_kv import snapshot_provider_kv
        checkpoint = original(executor, request_id)
        snapshot = snapshot_provider_kv(executor, checkpoint)
        self.provider_prefixes[request_id] = (executor, checkpoint, snapshot)
        return checkpoint

    def assert_provider_prefix(self):
        from scripts.qwen38_packed_c1_provider_kv import snapshot_provider_kv
        rid = self.evidence['request_id']
        if rid not in self.provider_prefixes:
            raise ValueError('no provider KV snapshot before proposal')
        executor, checkpoint, before = self.provider_prefixes[rid]
        if snapshot_provider_kv(executor, checkpoint) != before:
            raise ValueError('provider committed KV prefix changed')
        self.evidence['provider_kv_position'] = before['position']
        self.evidence['provider_kv_planes'] = len(before['buffers'])

    def prepare(self, session, request_id):
        from scripts.qwen38_packed_c1_state import snapshot_committed_state
        self.session = session
        self.before = snapshot_committed_state(session)
        self.evidence['request_id'] = request_id

    def assert_target(self):
        from scripts.qwen38_packed_c1_state import snapshot_committed_state
        if snapshot_committed_state(self.session) != self.before:
            raise ValueError('precommit failure changed canonical target state')
        self.evidence['target_unchanged'] = True
        self.evidence['target_buffers_checked'] = len(self.before['buffers'])

    def inject(self):
        self.assert_target()
        self.assert_provider_prefix()
        self.evidence['injected'] = True
        raise InjectedPrecommitFailure('diagnostic')

    def restore(self, original, executor, checkpoint):
        try:
            expected = provider_restore_expected(executor, checkpoint)
            result = original(executor, checkpoint)
            assert_provider_restored(executor, checkpoint, expected)
            self.assert_provider_prefix()
            self.evidence['provider_kv_restored'] = True
        except Exception as error:
            self.evidence['provider_restore_error'] = f'{type(error).__name__}: {error}'
            raise
        self.evidence['provider_restores'] += 1
        self.evidence['provider_buffers_checked'] = len(expected['buffers'])
        return result


def provider_restore_expected(executor, checkpoint):
    executor.runtime.device_synchronize()
    if checkpoint.released:
        raise ValueError('released provider checkpoint')
    scratch = executor.scratch.for_slot(checkpoint.slot, span_role='decode')
    live = [state for pair in zip(scratch.layer_conv_states, scratch.layer_recurrent_states, strict=True)
            for state in pair if state is not None]
    actual = [(int(state.ptr), int(state.nbytes)) for state in live]
    saved = [(int(state.ptr), int(state.nbytes)) for state, _ in checkpoint.state_pairs]
    if actual != saved or any(state.nbytes != backup.nbytes for state, backup in checkpoint.state_pairs):
        raise ValueError('provider checkpoint does not cover actual mutable states')
    return dict(request_id=int(checkpoint.request_id), slot=int(checkpoint.slot),
                position=int(checkpoint.position), context=int(checkpoint.context_length),
                buffers=[dict(ptr=int(state.ptr), nbytes=int(state.nbytes),
                              hash=_device_hash(executor, backup))
                         for state, backup in checkpoint.state_pairs])


def assert_provider_restored(executor, checkpoint, expected):
    executor.runtime.device_synchronize()
    if (checkpoint.released or checkpoint.request_id != expected['request_id']
            or checkpoint.slot != expected['slot']
            or executor._request_slots.get(expected['request_id']) != expected['slot']):
        raise ValueError('provider checkpoint ownership changed')
    if len(checkpoint.state_pairs) != len(expected['buffers']):
        raise ValueError('provider checkpoint buffer count changed')
    for (state, _), wanted in zip(checkpoint.state_pairs, expected['buffers'], strict=True):
        if (state.ptr != wanted['ptr'] or state.nbytes != wanted['nbytes']
                or _device_hash(executor, state) != wanted['hash']):
            raise ValueError('provider state did not match checkpoint bytes')
    scratch = executor.scratch.for_slot(expected['slot'], span_role='decode')
    sessions = getattr(executor, '_batch_sessions', None)
    if (int(scratch.position_host[0]) != expected['position']
            or int(scratch.context_host[0]) != expected['context']
            or (sessions is not None and int(sessions[expected['slot']].position) != expected['position'])):
        raise ValueError('provider logical cursor did not restore')
    for buffer, value in ((scratch.position_buf, expected['position']),
                          (scratch.context_buf, expected['context'])):
        raw = np.array([value], dtype=np.int64).tobytes()
        if buffer.nbytes != len(raw) or _device_hash(executor, buffer) != hashlib.blake2b(raw, digest_size=16).hexdigest():
            raise ValueError('provider device cursor did not restore')
