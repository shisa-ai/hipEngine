# Multi-Column GDN On Corrected Production

Frozen source `dfd5f9842`, Framework machine
`55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S/gfx1151,
Flash-Next UD-Q4_K_XL/BF16 KV, chunk 1024.

The full 18-prompt/594-row short numerical gate passes:
mean KL 0.00010724, p95 0.00039221, maximum 0.00967834,
593/594 top-1. Three numerical repeats, finite state, ownership metadata
and teardown pass, with 1350 counted multi-column dispatches.

Seventeen short free trajectories match strict. The remaining difference
is the rate-limiter code response, not the Japanese explanation that
blocked the separate single-column restoration. Both code prefixes are
incomplete; differing imports or wording do not establish a task regression.

## Canonical Depth

Clean source `3b739b674`, same physical host and model/BF16 configuration,
12 canonical512/1K/4K cases,64 teacher-forced decode transitions and three
candidate repeats. The current candidate declaration is T2; historical short
capture metadata is preserved unchanged.

| Metric | Measured | Limit |
| --- | ---: | ---: |
| Mean KL | 0.0000892215 | <=0.001 |
| p95 KL | 0.0004206676 | <=0.005 |
| p99 KL | 0.0012666802 | <=0.02 |
| Maximum KL | 0.0044548289 | <=0.05 |
| Top-1 | 777/780 (99.615%) | >=99% |

All global and declared scope numerical limits pass. Deterministic repeats,
sampled state layout/metadata/finiteness and zero-allocation teardown pass.
The registered multi-column kernel executes1080 times. Three top1 differences
are permitted by the production envelope; exact generated IDs are not the
admission criterion. These state checks do not certify dynamic isolation.

**No default promotion or current performance claim.** The numerical result
alone did not qualify tasks, isolation or current-model performance.
The earlier old-stack numerical rejection is recorded separately.

**September15 task decision:** not admitted under the existing per-prompt
criterion. The completed quantization response introduces a materially
misleading native-INT8 support explanation for AMD RX7000. This is a task
finding, not a numerical rejection; see `task-review.json` and the complete
paired outputs below. No isolation/timing promotion follows this finding.

## Targeted Complete Code Review

Clean `cc592ceca`, the rate-limiter heldout reaches EOS at656 strict and680
candidate tokens, with two identical repeats per arm,30 counted candidate
GDN calls and zero tracked allocations after teardown.

Both implementations pass36000 independent `allow(key,timestamp)` oracle
checks covering capacities, windows, expiry, repeated monotone timestamps and
independent keys. Both use the same amortized deque expiry/append algorithm.
Both omit the explicitly requested pytest tests and emit a code fence: these
are shared instruction-following defects, not new candidate regressions.

The extra, unrequested `remaining()` method differs: candidate scans the
stored window, whereas strict expires deque heads then reads its length.
A128-entry/100-query probe counts12800 candidate iteration visits versus0
strict iteration visits. This is a real optional-method complexity regression,
not a timing measurement; it is not used as a veto for the requested `allow`
API. The constructor keyword-name difference is outside the specified API.

**No new requested-API regression found on this prompt.** This scoped paired
review is not a full task-suite pass or proof of complete instruction following.
The remaining task results appear below. `rate-limiter-review.json`
preserves both complete outputs and the shared defects. Its reproduction
script executes only the inspected source hashes:

```bash
.venv/bin/python benchmarks/results/2026-09-15-journey-gdn-multi-restoration/review_rate_limiter.py \
  --capture /tmp/hipengine-journey-execute-20260914/resume-gdn-multi-rate-limiter-task.json
```

## Remaining Category And Heldout Tasks

Clean `b6f841832`:17 additional prompt pairs, two repeats per arm.
Six complete pairs match IDs exactly; ten other pairs reach EOS with changed
outputs. Both Japanese-plan responses hit the2048-token cap. All within-arm
repeats are identical,510 candidate GDN calls are counted, and teardown closes
to zero. Combined with the rate-limiter capture,17 of18 unique prompt pairs
have complete EOS evidence; the truncated pair is not counted as passed.

Both changed Markdown-table programs pass their generated tests plus1000
shared oracle cases. Both interval-scheduling programs pass generated tests
plus500 exhaustive-subset-oracle cases and a tie check. The candidate does
not reproduce the single-column Japanese speculative-decoding error.

The binding task finding is instead in `heldout_general_en_quant`: the
candidate conflates native AMD INT8 capability with library support and FP16
emulation. AMD's primary WMMA documentation demonstrates native IU8 and a
HIP path on RX7900XTX. That source establishes availability, not the frequency
of fallback across every library; the review states this limit explicitly.
Strict's own errors and shared omissions remain disclosed. This one paired
finding is not proof of worse expected quality or a GDN kernel defect.

The unchanged candidate is not promoted. No Japanese-plan extension or
performance arm is run after the task finding. Full paired texts, code
execution evidence, provenance and review are preserved in
`remaining-task-capture.json` and `task-review.json`.

```bash
.venv/bin/python benchmarks/results/2026-09-15-journey-gdn-multi-restoration/review_remaining_tasks.py \
  --capture /tmp/hipengine-journey-execute-20260914/resume-gdn-multi-remaining-tasks.json
```

Exact commands, overrides, source/host/profile metadata and raw hash are
preserved in `artifact.json` and its referenced task captures. The original
short qualification ran in its own clean worktree, so main-tree backend-cache
preparation did not change that capture's source.

```bash
.venv/bin/python benchmarks/results/2026-09-15-journey-gdn-multi-restoration/assemble.py \
  --capture /tmp/hipengine-journey-execute-20260914/resume-gdn-multi-profile.json \
  --depth-capture /tmp/hipengine-journey-execute-20260914/resume-gdn-multi-depth.json
```

Depth capture uses `scripts/qwen4exp_q8_repair_depth_gate.py --candidate
production_gdn_multi_restore` with the same model/compiler arguments and
cached-build environment recorded in the generic Q8 artifact. Exact argv,
host/model identity, strict/production manifests and overrides are preserved
in `depth_capture`. No tests/builds or other GPU workload overlapped this
numerical run; it is not a throughput measurement.
