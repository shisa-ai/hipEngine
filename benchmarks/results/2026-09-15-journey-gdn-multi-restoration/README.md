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

**No default promotion or current performance claim.** Complete task review,
applicable isolation and a current-model performance comparison remain.
The earlier old-stack numerical rejection remains recorded separately.

Exact commands, overrides, source/host/profile metadata and raw hash are
preserved in `artifact.json`. The qualification ran in its own clean
worktree, so main-tree backend-cache preparation did not change its source.

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
