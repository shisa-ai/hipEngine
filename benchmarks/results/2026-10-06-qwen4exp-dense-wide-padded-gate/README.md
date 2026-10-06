# Full-model padded HC-down tail qualification

On 2026-10-06 UTC, the diagnostic padded dense-wide sibling passed the
full-state qualification on Radeon 8060S host `gfx1151`, machine-id
`55ea6c509d0b49eea8de7094a1023668`.

The gate covered 12 canonical cases and eight category-heldout fixtures. Each
case ran five arms: parent, candidate, candidate, candidate, parent. The
candidate was admitted only for the predeclared Q8_0 HC-down geometry
K10240/N320 and 512–4096 rows.

| Gate | Result |
| --- | --- |
| Cases | 20/20 |
| Candidate calls | 4,200 total; 70 per enabled arm/case |
| Candidate decode calls | 0 |
| Full prefill logits | Exact across all arms |
| Post-decode logits/state | Exact across all arms and 16 transitions |
| Prefill state/layout | Exact across all arms |
| Full KV hashes | Exact across all arms |
| Output token IDs | Exact across all arms |
| Teardown | 0 active allocations, 0 current bytes |

The eight short heldouts were repeated/truncated to 512 token IDs only to
exercise the admitted HC-down shape. They are state/isolation fixtures, not
task-quality measurements. The canonical fixture and heldout-suite hashes,
profile manifest hashes, commands and per-arm digests are in
[`full-state.json`](full-state.json).

This qualifies the diagnostic leaf's model control/state boundary. It does
not qualify a production default or establish model-wall speedup. The next
required result is a tracked-clean same-host three-pair engine A/B with the
same parent/candidate substitution and complete request timing.

The run was captured from source `9c668e702` while the gate harness was being
committed; the harness is now committed in `035338ed6`. The kernel and model
code are unchanged between those commits. The source record's dirty flag is
preserved rather than rewritten.
