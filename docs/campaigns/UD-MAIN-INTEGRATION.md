---
status: closed
owns: All 236 UD commits were rebased onto main at d26cd792f.
---
# UD/Main Integration Report

Date: 2026-09-13 UTC.

## Integration

All 236 UD commits were rebased onto main at `d26cd792f`. The original
`3eaab7c30` tip is preserved locally as
`backup/ud-quants-pre-main-rebase-20260913-3eaab7c30`.
The repair commit containing this report completes the integration changes.

Main's newer behavior was the baseline:

- Kept profile selection, graph break-even admission, DMS owner refreshes,
  Qwen4Exp raw-Q8 F32 capabilities, and gfx1151 raw-Q5/Q6 policy declines.
- Used UD's shared consumer-contract table, restoring main's raw-F32 and
  dense-BF16/F32-input dispatch entries.
- Kept main's compact primary journal and lazy, separately owned serial
  journal. UD's serial-capability check runs after journal selection.
- Confined the 256-thread Q8 dual-split choice to gfx1100 rows 1-4 at
  `(K, N_a, N_b) = (5120, 48, 48)`. Standalone reference calls, other shapes,
  and peer backends keep main's 128-thread default. Existing overrides work.
- Fixed a cold-import cycle exposed by the lazy root API and restored the
  Laguna activation-pack export lost during automatic merging.
- Migrated 41 UD tests into main's execution tiers. Historical command paths
  resolve through a separate rename record; original result artifacts and
  committed worklog entries were not rewritten.
- Repaired stale fixture assumptions, fake-memory tracker isolation, and an
  unexecutable Qwen4 Q8 fixture (packing, tile map and output dtype), without
  loosening its numerical tolerance or changing that kernel.

## Validation

[Machine-readable evidence](../../benchmarks/results/2026-09-13-ud-main-integration-validation.json)
contains commands, provenance, numerical metrics and test outcomes.

- Final unit tier: **8,583 passed, 13 skipped**, including the eleven new
  Q8 scope tests.
- The all-tier attempt completed 7,643 cases before a HIP `malloc` abort:
  7,327 passed, 18 failed, 294 skipped, and 4 expected failures.
- All 18 failures were repaired and covered by focused reruns.
- The aborting state-only test passed in a fresh process. The remaining
  live-test groups passed: 92 passed and 25 skipped across 117 cases,
  including that separately executed test.
- Composed coverage is **16,020 passed, 332 skipped, 4 expected failures**.
  This combines the completed prefix, focused repairs and isolated remaining
  groups; it is **not a single all-green monolithic run**.
- The final Q8 GPU bundle passes. Compilation, test tiers, CPU fixtures,
  registry smoke, published-command/provenance checks, README synchronization,
  and worklog validation pass.

The monolithic HIP allocator abort's root cause was not established.
It did not reproduce in the fresh state-only/rollback tests. No GPU reset or
cache purge was performed. All four paired model runs reported zero live
allocations after close.

## Model Results

Host `epyc`, physical GPU1, RX 7900 XTX / gfx1100. Qwen3.8-27B, canonical ten
prompts, two repeats, c1/natural25/B3, true graph-replay AR denominator.
These working-tree measurements are integration diagnostics, not new speed
promotions or clean-tree performance claims.

| Artifact | AR tok/s | MTP tok/s | MTP / AR | AR/MTP IDs |
| --- | ---: | ---: | ---: | --- |
| UD Q4_K_M | 33.149 | 51.033 | 1.5395x | Exact |
| Plain Q4_K_M | 38.671 | 67.001 | 1.7326x | Exact |
| UD Q4_K_S | 32.507 | 49.926 | 1.5358x | Known 2/20 divergences |
| Plain Q4_K_S | 41.320 | 70.841 | 1.7144x | Known 2/20 divergences |

All repeats are deterministic and all timing/acceptance predicates pass.
Against the pre-rebase clean raw payloads, all four artifacts have identical
token IDs, accepted/proposed counts and cycle counts: 20 AR and 20 MTP rows
each, 160 rows total. The final Q8 scope guard preserves the measured SSM
kernel arguments; other shapes return to main's reference default.

The 18-prompt category/heldout teacher-forced gate, B1-B3 and two repeats,
passes at these measured points:

| Artifact / root tokens | Mean KL | Max KL | Top-1 | Deterministic |
| --- | ---: | ---: | ---: | --- |
| UD K_M / 64 | 3.287e-5 | 2.432e-4 | 100% | Yes |
| UD K_M / 512 | 4.275e-5 | 7.806e-4 | 99.38% | Yes |
| UD K_S / 64 | 3.008e-5 | 5.714e-4 | 100% | Yes |

## Limits

- The queued K_S 512-token rerun was not started; no fresh pass is claimed.
- K_S speed-claim eligibility remains restricted by its known generated-ID
  divergence. Numerical gate passes do not silently rewrite that status.
- No gfx1151/CUDA hardware qualification, broader serving admission, or
  long-context MTP promotion follows from this integration.
- Automatic UD admission remains scoped to c1/capacity1, BF16 KV, greedy B3,
  context 4-95 and output horizon 24. Existing fallbacks remain available.
- Historical per-lever rates remain historical; current headline tables were
  not replaced by the working-tree integration timings.
