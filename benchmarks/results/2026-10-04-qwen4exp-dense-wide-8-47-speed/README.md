# The dense-wide route at layers 8-47 is +8.52% prefill, and its gate failure is one category

The dense-wide Q8 prefill route binds to layers 16-47. Extending it to **8-47** was
gated before but **its speed was never measured**. This measures it with the
three-arm A/B/C design (baseline on both sides, so the same-code drift is the
control), `code-p4096`, chunk 1024, one process per arm, 1 warmup + 3 repetitions:

| arm | scope | prefill tok/s | decode tok/s | samples |
| --- | --- | ---: | ---: | --- |
| A | 16-47 | 271.84 | 17.964 | 271.88 / 271.81 / 271.81 |
| B | **8-47** | **295.01** | **18.281** | 295.09 / 294.92 / 295.02 |
| C | 16-47 | 271.86 | 17.854 | 271.65 / 271.75 / 272.19 |

**Same-code drift A against C is +0.01%.** Against `mean(A,C) = 271.851`, 8-47 is
**+8.52% prefill and +2.08% decode** - a **signal-to-drift ratio of 852x**. Every
arm is separated on both columns and the baselines agree to 0.02 tok/s.

## Why this point mattered

Per-layer speedup is roughly linear in scope (14-47 adds 2 layers for +1.69%,
0-47 adds 16 for +18.88%), so the deepest *passing* scope 14-47 was worth almost
nothing while the speed sat in the low layers. **8-47 covers 8 new layers and
delivers +8.52%** - consistent with the per-layer prediction of ~+7-9% and about
**4x the deepest passing scope**.

## Its gate result, which is the whole question

From the recorded scope ladder, 8-47 **passes every aggregate gate** - `kl_mean`
8.839e-4 (limit 1e-3), `kl_p95` 4.647e-3 (5e-3), `kl_max` 4.146e-2 (5e-2), top-1
0.99096 (0.99) - and **fails exactly one category**: `general_en` mean KL
**1.1812e-3, 1.18x over the 1e-3 limit**.

**So this is the only point on the frontier where a large, drift-free speedup sits
behind a single-category numerical failure.** The next unit is to find where that
one category's drift enters and whether it is fixable; the row is **not promoted**
by this measurement.

## Method notes

- The lever is the file constant `PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS`. The
  profile binder force-sets it, so an env override cannot reach it; each arm set
  the constant and the script printed it per arm. **The constant was restored to
  `range(16, 48)` and `git status` confirms the file is unmodified.**
- Decode moves **+2.08%** in the same direction, so this is not a prefill-only
  trade.

## Limits

One host (gfx1151 Strix Halo, Radeon 8060S, not the gfx1100 default hardware),
one model (`unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL`, UD-Q4_K_XL), one profile, one
case (`code-p4096`), one chunk size (1024), 3 repetitions per arm. The gate result
is quoted from the recorded ladder run, not re-run here, so speed and gate come
from different sessions at different scopes' worth of configuration.
