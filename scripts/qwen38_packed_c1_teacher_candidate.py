"""Teacher-driven packed target capture, not a provider/service certificate.

Caller owns profile/variant contexts, validates the teacher fixture and prompt
tokenization, and constructs compatible owner/session objects. Every window
consumes teacher inputs and commits its last row with the host-selected state
helper; this isolates target arithmetic from speculative proposal decisions.
"""
import numpy as np

from scripts.qwen38_packed_c1_state import (
    snapshot_committed_state, assert_committed_state_unchanged,
)
from scripts.qwen38_packed_c1_teacher import teacher_windows, _owned_logits


def _read_logits(owner, rows):
    from scripts.qwen38_packed_c1_logits import _read_device
    return _read_device(owner._verify_logits_buf.ptr,
                        (rows, owner.runner.vocab_size), np.float32, owner.runtime)


def capture_teacher_candidate(owner, session, teacher, *, budget, use_wmma_prefill):
    """Return owned prefill/decode logits and window coordinates.

    No numerical envelope, runtime provenance or serving qualification is claimed.
    The head hook is instance-scoped and restored on success and failure.
    """
    windows = teacher_windows(teacher, budget=budget)
    session.reset()
    prefill = _owned_logits(session.prefill(teacher['prompt_ids'], return_logits=True))
    if int(session.position) != len(teacher['prompt_ids']):
        raise ValueError('candidate prefill cursor differs from teacher')
    original = owner._enqueue_target_block_rows_from_hidden
    had_override = '_enqueue_target_block_rows_from_hidden' in owner.__dict__
    fresh = []

    def head(*args, **kwargs):
        result = original(*args, **kwargs)
        path = owner._last_packed_lm_head_decode_path
        if path not in {'q6_rowtile_f32_logits', 'row_linear_f32_logits'}:
            raise ValueError(f'packed head did not produce fresh full logits: {path}')
        fresh.append(path)
        return result

    owner._enqueue_target_block_rows_from_hidden = head
    arrays, records = [], []
    try:
        for transaction, window in enumerate(windows):
            position = int(window['position'])
            tokens = tuple(window['tokens'])
            if int(session.position) != position:
                raise ValueError('candidate decode cursor differs from teacher window')
            slot = int(getattr(session, '_resident_slot_index', 0) or 0)
            job = dict(session=session, request_id=0, resident_slot=slot,
                       transaction_id=transaction, input_token_ids=tokens,
                       bulk_attention_mode='bulk', use_wmma_prefill=use_wmma_prefill,
                       capture_linear_state_rows=True, defer_linear_state_commit=True,
                       defer_state_scatter=True)
            before = snapshot_committed_state(session)
            fresh.clear()
            results = owner.verify_target_blocks_batch([job], device_result=True)
            assert_committed_state_unchanged(before, snapshot_committed_state(session))
            if len(results) != 1:
                raise ValueError('packed target returned wrong request count')
            result = results[0]
            if (result.request_id != 0 or result.resident_slot != slot
                    or result.transaction_id != transaction or result.start_position != position
                    or result.row_start != 0 or result.row_end != len(tokens)):
                raise ValueError('packed target changed teacher window ownership')
            if len(fresh) != 1 or owner._verify_logits_buf is None:
                raise ValueError('packed target did not execute exactly one fresh full-logit head')
            logits = _read_logits(owner, len(tokens))
            if (logits.dtype != np.float32 or logits.shape != window['logits'].shape
                    or logits.shape[1:] != prefill.shape or not np.isfinite(logits).all()):
                raise ValueError('candidate requires aligned finite full-vocabulary FP32 logits')
            logits = logits.copy()
            end = position + len(tokens)
            owner._commit_deferred_packed_verify_state(
                result.deferred_packed_state, session,
                commit_row_index=len(tokens) - 1, position=end, hidden_rows=len(tokens))
            if int(session.position) != end:
                raise ValueError('selected commit did not advance to teacher cursor')
            arrays.append(logits)
            records.append(dict(position=position, tokens=list(tokens), rows=len(tokens),
                                head_path=fresh[0], resident_slot=slot))
    finally:
        if had_override:
            owner._enqueue_target_block_rows_from_hidden = original
        else:
            del owner._enqueue_target_block_rows_from_hidden
    return dict(prefill_logits=prefill, logits=np.concatenate(arrays), windows=records,
                full_profile_qualification=False, commit_mode='host_selected_teacher_window')
