"""Run only the hash-pinned, manually reviewed output in an isolated sandbox.

Run with the bwrap command in README.md, never against arbitrary model output.
"""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

capture = json.loads(Path('/capture.json').read_text())
outputs = [a['cases'][0]['output'] for a in capture['arms'].values()]
assert len(outputs) == 2 and outputs[0] == outputs[1]
code = outputs[0].removeprefix('```python\n').removesuffix('```<|im_end|>')
assert hashlib.sha256(code.encode()).hexdigest() == (
    'b8685e995cfe9800efd9ddc82a5aa204ff56aad07b9350ce564dc856b5de3fbd')
namespace = {'__name__': '__main__'}
exec(compile(code, '<reviewed-booking-output>', 'exec'), namespace)
reconcile = namespace['reconcile']


def event(bid, revision, start, end, cancelled=False, room='R'):
    return dict(booking_id=bid, revision=revision, start=start, end=end,
                cancelled=cancelled, room=room)


cases = [
    # Choosing the wrong older revision would incorrectly accept B.
    ([event('A', 2, 0, 10), event('A', 1, 0, 5), event('B', 1, 5, 7)],
     {'accepted': ['A'], 'rejected': ['B']}),
    # Equal revision must choose the later interval, making B acceptable.
    ([event('A', 1, 0, 10), event('A', 1, 10, 15), event('B', 1, 5, 7)],
     {'accepted': ['B', 'A'], 'rejected': []}),
    # Remove the winning cancellation before allowing B; ignore older active A.
    ([event('A', 3, 0, 10, True), event('A', 1, 0, 20), event('B', 1, 5, 7)],
     {'accepted': ['B'], 'rejected': []}),
    # Shorter end first, then ID, regardless of arrival order.
    ([event('B', 1, 0, 5), event('C', 1, 0, 10), event('A', 1, 0, 5)],
     {'accepted': ['A'], 'rejected': ['B', 'C']}),
    # Rejected B must not block C; a second room is independent.
    ([event('C', 1, 5, 7), event('B', 1, 3, 10), event('A', 1, 0, 5),
      event('D', 1, 0, 5, room='S')],
     {'accepted': ['A', 'D', 'C'], 'rejected': ['B']}),
    ([], {'accepted': [], 'rejected': []}),
]
for events, expected in cases:
    original = deepcopy(events)
    assert reconcile(events) == expected
    assert events == original
print('11 generated tests and 6 independent cases passed; input immutability checked')
