# MoE Backend Refresh Cache

Measured September 14 UTC, Framework machine
`55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S/gfx1151,
Flash-Next UD-Q4_K_XL/BF16 KV. Source `93b15d89e`.

The MoE host path previously refreshed the full backend registration table
on every eager call. It now reuses a stable registry generation. Mutations
force refresh, a generation changed during refresh is not cached, failed
registration does not create an entry, and caller wrappers are preserved.
No GPU kernel, arithmetic, graph node or device allocation changes.
The experiment switch is removed; the cache is unconditional.

## Same-Session Result

One residency, shared unchanged decode graphs, chunk 1024, four categories,
512/1K/4K prompt lengths and 128 post-first-output decode transitions.
Three counterbalanced pairs per case: 72 measured samples, 24 warmups.
No CPU tests, compiler or other model workloads overlapped timing.

| Configuration | 512 PP / TG | 1K PP / TG | 4K PP / TG |
| --- | ---: | ---: | ---: |
| Production without host cache | 177.703 / 17.520 | 186.190 / 16.749 | 178.411 / 10.158 |
| Production with host cache | 183.546 / 17.492 | 189.153 / 16.719 | 181.759 / 10.019 |
| PP gain | +3.29% | +1.59% | +1.88% |

Token/time-weighted tok/s. All 12 complete requests improve by 0.22-1.54%.
All samples match across arms and repeats in generated IDs, final full
logits and recurrent state. Tracked ownership closes to zero.

**Decode is lower by 0.16/0.17/1.36%.** This is a prefill and complete-request
win at the measured 128-transition workload, not an across-the-board
latency win or a claim for arbitrary output lengths. The current run does
not independently attribute the phase tradeoff to CPU-frequency recovery.
No clock or affinity workaround was used.

## Validation And Scope

79 focused CPU tests cover cache/mutation/failed-refresh/wrapper semantics,
runner transactions and backend registration. Four MoE GPU tests compare
against CPU reference and exercise the grouped Q8 fallback. The fresh
no-override code/4K smoke matches strict on 9/9 rows with three repeats,
state and teardown passing. It is labeled a worktree smoke.

T0 qualification preserves the already-qualified production computation;
it does not relax its numerical or task gates. GDN suffix, dense/GR iu8,
DP4A and other unqualified arithmetic paths are not restored by this change.
No new MTP, multimodal or long-form factual-quality certificate is claimed.

Commands, environment, source/host/model/profile identities, runtime library
hashes, all compact timing samples and paired variability are in `artifact.json`.
The historical A/B command uses the experiment selector at its pinned source;
that selector is intentionally absent from the unconditional implementation.

```bash
.venv/bin/python benchmarks/results/2026-09-14-journey-backend-cache/assemble.py \
  --raw-root /tmp/hipengine-journey-execute-20260914
```
