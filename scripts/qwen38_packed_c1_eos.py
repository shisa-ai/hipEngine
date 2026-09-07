"""Configured EOS boundary from an independent AR trajectory; no logit edits."""
from dataclasses import replace


def configure_eos_request(request, oracle_ids, *, index):
    if not 0 <= index < len(oracle_ids) < request.max_tokens + 1:
        raise ValueError('invalid EOS oracle boundary')
    if oracle_ids[index] in oracle_ids[:index] or request.min_tokens != 0:
        raise ValueError('EOS fixture requires a first-occurrence marker and no minimum-token suppression')
    return replace(request, eos_token_id=int(oracle_ids[index]), ignore_eos=False)


def validate_eos_terminal(handle, output, oracle_ids, *, index):
    if not 0 <= index < len(oracle_ids):
        raise ValueError('invalid EOS oracle boundary')
    wanted = tuple(oracle_ids[:index + 1])
    if tuple(output.generated_token_ids or ()) != wanted or output.finish_details.reason != 'eos':
        raise ValueError('EOS output is not the expected AR prefix')
    terminal = handle._state.collector.result
    if (terminal is None or terminal.request_id != handle.request_id
            or terminal.error is not None or terminal.finish_reason != 'eos'
            or tuple(terminal.generated_token_ids) != wanted):
        raise ValueError('EOS terminal collector differs from returned output')
    return dict(backend_request_id=int(handle.backend_request_id),
                service_request_id=int(handle.request_id), eos_token_id=int(oracle_ids[index]),
                eos_index=index, visible_tokens=len(wanted), collector_exact=True)
