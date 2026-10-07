# Qwen4Exp MT-8 rollback/replay accounting

This diagnostic inventories the current serial Qwen4Exp MTP provider at proposal
budget 4. A CPU-only instrumented screen covers all rejection depths, all-reject
and full-accept.

| case | target forwards | draft forwards | restored bytes | per-cycle allocations |
|---|---:|---:|---:|---:|
| all-reject | 1 | 4 | 0 | 0 |
| reject after 1 | 2 | 4 | 0 | 0 |
| reject after 2 | 3 | 4 | 0 | 0 |
| reject after 3 | 4 | 4 | 0 | 0 |
| full-accept | 4 | 4 | 0 | 0 |

The provider stops target verification at the first mismatch and trims the draft
cursor after each cycle. No duplicate target replay or dirty-destination restore
is present in this route. The screen is diagnostic only; production defaults and
performance claims are unchanged.

Artifact: [`accounting.json`](accounting.json).
