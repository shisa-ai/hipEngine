# R7 post-promotion retention - v8 packet (three wrapper-host promotions)

Capture at the promoted HEAD ef0625e4c (batched QSA positions + compiler-version
memoization + PLE page-cache warm sweep, plus the review's override-identity
cache fix). All samples validate; fixture/host/binary hashes match.

**Window quality**: the host ran degraded during this capture (and the first v8
attempt): 19 GB swap in use, MemFree as low as ~0-1.5 GB mid-capture, load
2.8-3.9 with background processes. Memwatch logged every 30s during v8b2
(/tmp capture logs). Both halo-box comparators dropped 5-10% vs their v7b
window rates across two independent captures - the comparators cannot be
affected by hipEngine code, so this is host drift, not regression.

## Rates (v7b -> v8)

| Engine | p512 PP / TG | p1024 PP / TG | p4096 PP / TG | Max PP / TG CV |
| --- | ---: | ---: | ---: | ---: |
| hipengine | 296.28 -> 288.87 / 20.02 -> 19.96 | 294.99 -> 283.68 / 19.53 -> 19.30 | 262.21 -> 259.07 / 19.14 -> 19.21 | 3.68% / 3.62% |
| halo-box-vulkan | 303.90 / 24.86 | 349.25 / 23.99 | 404.68 / 23.07 | 15.95% / 7.76% |
| halo-box-hip | 276.69 / 21.26 | 345.52 / 20.03 | 340.21 / 18.55 | 12.18% / 3.26% |

## Why hipEngine TG is flat vs v7b despite ~12% promoted gains

Clean per-process A/B on the CURRENT (degraded) host, same day
(SUPERSEDED by 2026-09-10-r7-combined-tg-attribution.json, which uses a
true all-off arm at 8a770b782 and correct units):
- flags OFF (BATCHED_POSITION=0, PLE_WARM=0; version cache still active -
  this arm did NOT isolate the compiler cache): TG median 59.68 ms/token
- flags ON (defaults): TG median 52.06 ms/token
- delta: -7.62 ms/token = **-12.8% latency = +14.6% throughput** (the
  original label '+12.8% TG' conflated latency with throughput)

The promotions' effect is fully present; the packet's flat cross-window
comparison reflects ~10% host drift since the v7b window (comparator-
corroborated). Cross-window packet comparisons on this shared host are not
reliable at the few-percent level; same-process interleaved A/Bs remain the
gate evidence of record.

## Methodology note

A mid-process flag-toggle A/B (switching BATCHED_POSITION/PLE_WARM between
case groups without runner reset) produced a token-digest mismatch.

**Root cause identified (task #33, 2026-09-10)**: harness contract violation,
not a flag effect and not a runtime defect. The A/B harness called
`runner.prefill(...)` with default `capture_logits=True` (token computed
host-side via `np.argmax`; `token_id_buffer` never written) but then drove
decode steps with `token_id_resident=True` (embedding reads `token_id_buffer`).
Every case's first decode step therefore embedded whatever token was last
written to that buffer: for case 1 of repetition 1 the fresh (deterministic)
allocation contents; for later cases/repetitions the previous sequence's
final argmax token. Repeating the 12-case set in one process flips case 1's
step-1 embedded token, yielding exactly the two digests observed. The flag
toggles were coincidental to run order: a dedicated reproducer shows toggling
BATCHED_POSITION/PLE_WARM mid-process with the contract-correct flow is
bit-exact. Production never enters the broken regime (the generator's compact
path pairs `capture_logits=False` prefill with resident steps; the controlled
path pairs `capture_logits=True` with host round-trips). The batched-position
promotion is unaffected. See worklog 20260910T094500.000000Z-lhl-qwen4exp-digest-mismatch-repro-af418a.md.
