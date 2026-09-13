"""Controller-side cancellation checks for the native C1 engine diagnostic."""
from __future__ import annotations


def validate_cancel_outputs(expected, actual, evidence, transition):
    """Bind the acknowledged cancellation to exact prefix and survivor output."""
    prefix = actual[0]['generated_ids']
    full = expected[0]['generated_ids']
    if not 0 < len(prefix) < len(full) or prefix != full[:len(prefix)]:
        raise ValueError('cancelled output is not a nonempty proper AR prefix')
    if actual[1]['generated_ids'] != expected[1]['generated_ids']:
        raise ValueError('survivor output differs from independent AR')
    if (not evidence['acknowledged']
            or evidence['survivor_request_id'] != transition['survivor_request_id']
            or evidence['cancelled_request_id'] == transition['survivor_request_id']):
        raise ValueError('cancellation does not match the traced survivor')


def collect_cancelled_pair(handles, paired_ready):
    """Wait for paired target execution, then cancel only the first real child.

    Call from the controller, never the service/GPU worker: cancel waits for the
    service command to be acknowledged. The event must follow successful paired
    execution, not submission or a timer.
    """
    if len(handles) != 2:
        raise ValueError('cancellation requires two independent handles')
    if not paired_ready.wait(timeout=60):
        raise TimeoutError('no successful paired target before cancellation')
    cancelled_id, survivor_id = (h.backend_request_id for h in handles)
    if cancelled_id == survivor_id:
        raise ValueError('cancellation handles share a request identity')
    if not handles[0].cancel(reason='cancel'):
        raise ValueError('peer cancellation was not acknowledged')
    from types import SimpleNamespace
    from hipengine.generation.deadline import GenerationCancelled
    source = None
    try:
        cancelled = handles[0].result(timeout=120)
    except GenerationCancelled as error:
        # Cancellation has no normal GenerationOutput. Read the actual terminal
        # collector history, retaining its distinct provenance in the artifact.
        terminal = handles[0]._state.collector.result
        if (terminal is None or terminal.request_id != handles[0].request_id
                or terminal.error is not error or terminal.finish_reason != 'cancelled'):
            raise ValueError('cancelled terminal collector identity mismatch') from error
        cancelled = SimpleNamespace(generated_token_ids=terminal.generated_token_ids,
                                    finish_details=error.finish_details)
        source = 'terminal_collector'
    outputs = [cancelled, handles[1].result(timeout=120)]
    if not bool(getattr(outputs[0].finish_details, 'cancelled', False)):
        raise ValueError('cancelled child lacks cancelled finish details')
    if bool(getattr(outputs[1].finish_details, 'cancelled', False)):
        raise ValueError('cancellation leaked into survivor')
    evidence = dict(cancelled_request_id=cancelled_id,
                    survivor_request_id=survivor_id, acknowledged=True)
    if source is not None:
        evidence['cancelled_output_source'] = source
    return outputs, evidence
