---
status: closed
owns: Public success, cancellation and checkpoint-capture regression evidence for the Qwen4 single-request MTP state journal.
---
# Qwen4 MTP state-journal validation

The single-request Qwen4 MTP route uses `SpeculativeStateJournal` to capture
completed checkpoints and attempt every restore even if another restore fails.
The target transaction commits before the journal drops its undo records.
Iterator cleanup runs in a `finally` block. Kernels, sampling, acceptance and
the mixed-request route are unchanged.

## Public comparison

[`public.json`](public.json) records 12 calls on the local Framework Desktop
with Radeon 8060S (gfx1151), Flash-Next GGUF UD-Q4_K_XL target and Q8_0 MTP
sidecar. At prompt lengths 37 and 513, temperatures 0 and 0.7, parent
`07fba0543`, candidate and repeated candidate calls produced identical output
IDs. Each call exercised `_advance_mtp` five times. Tracked allocations and
bytes returned to zero after close. This is correctness evidence, not a
performance comparison.

[`compare_public.py`](compare_public.py) loads the parent's method from Git
and substitutes only that method for the baseline. Requests use public
`LLM.generate_detailed()` with `speculative_mtp_automatic=True`, the intent
used by automatic server selection. Explicit MTP requests normally use the
separate mixed-target path, which this change does not modify.

The first attempt stopped before generation because sequence capacity 768
provided fewer QSA index blocks than the selection budget. The successful
attempt used capacity 4096 with the same prompt lengths. The exact command,
source digest and attempt record are in [`protocol.json`](protocol.json).

## Failure and cancellation checks

`tests/test_live_qwen4exp_mtp_advance_journal.py`: **2 passed**.

- Cancellation after target verification and draft catch-up restored live KV,
  recurrent state, index state, cursors and sampling state. Recurrent bytes
  matched the real device checkpoint. The rejected step published no tokens,
  closed its iterator and released its transaction lease. A following public
  request completed successfully.
- Draft checkpoint-capture failure restored earlier captured owners and
  closed the iterator without leaving a target lease. The engine reported the
  injected error; the target could acquire and commit another transaction.

The oracle compares live rows, not block-rounded spare capacity that a rejected
step may legally have written. Run with the local ROCm Python environment:

```bash
bash /tmp/hipengine-paro-refresh-20261023-env.sh env PYTHONPATH="$PWD" \
  "$HOME/miniforge3/envs/therock/bin/python" -m pytest \
  tests/test_live_qwen4exp_mtp_advance_journal.py -v -o addopts=''
```

## CPU checks

The new fault-injection module failed seven of eight cases against the parent
and passed all eight against the candidate. The child ran 169 focused cases;
the parent reran 127 directly affected cases. The default unit suite passed,
as did compileall, fixture checks, test-tier checks, registry smoke, CPU-fixture
smoke and smoke-add planning. The diff introduces no Torch import.

The audit gate passed after recording the same three existing Qwen4 graph-key
findings under their Python 3.12 identities. Python 3.13 and 3.12 produce
different AST context hashes for those duplicate sites; this is unrelated to
MTP runtime behavior. The budget did not change.
