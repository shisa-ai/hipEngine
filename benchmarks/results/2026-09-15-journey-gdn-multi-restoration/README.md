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

**No default promotion or current performance claim.** Canonical depth,
complete task review and a current-model performance comparison remain.
The earlier old-stack numerical rejection remains recorded separately.

Exact commands, overrides, source/host/profile metadata and raw hash are
preserved in `artifact.json`. The qualification ran in its own clean
worktree, so main-tree backend-cache preparation did not change its source.

```bash
.venv/bin/python benchmarks/results/2026-09-15-journey-gdn-multi-restoration/assemble.py \
  --capture /tmp/hipengine-journey-execute-20260914/resume-gdn-multi-profile.json
```
