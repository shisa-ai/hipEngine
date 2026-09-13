"""Summarize observed selected-state coverage without inferring missing cases."""
from collections import Counter


def summarize_depth(capture, *, budget):
    records = capture.get('records', [])
    if not capture.get('complete') or capture.get('target_errors') or not records or not 1 <= budget <= 7:
        raise ValueError('incomplete depth capture')
    outcomes, counts, rejected, depths = Counter(), Counter(), Counter(), Counter()
    for row in records:
        if any(row.get(key) for key in ('commit_error', 'kv_error')):
            raise ValueError('depth capture contains errors')
        for key in ('pre_accept_state_isolation', 'selected_commit', 'aux_commit', 'kv_commit'):
            check = row.get(key, {})
            if check.get('passed') is not True or check.get('checked_buffers', 0) < 1:
                raise ValueError('depth capture lacks nonempty ownership proof')
        depth = row['logical_rows'] - 1
        remaining = row['remaining_decode']
        accepted = row['selected_commit']['accepted']
        if (not 1 <= depth <= budget or len(row['tokens']) != depth + 1
                or remaining < 1 or not 0 <= accepted <= min(depth, remaining - 1)):
            raise ValueError('invalid depth/acceptance coordinates')
        counts[str(accepted)] += 1
        depths[str(depth)] += 1
        if accepted == depth:
            outcomes['full_accept'] += 1
        elif accepted == remaining - 1:
            outcomes['horizon_clip'] += 1
        else:
            outcomes['rejection'] += 1
            rejected[f'K{depth}:candidate{accepted + 1}'] += 1
    if str(budget) not in depths:
        raise ValueError('requested depth never executed')
    return dict(cycles=len(records), prompts=sorted({r['prompt_id'] for r in records}),
                accepted_counts=dict(sorted(counts.items())), outcomes=dict(outcomes),
                logical_depths=dict(sorted(depths.items())), observed_rejections=dict(sorted(rejected.items())),
                exhaustive_rejection_or_eos_qualification=False)
