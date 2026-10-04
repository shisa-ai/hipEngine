# The dense-wide scope frontier is pinned at 14-47; no passing scope reaches the speed

The dense-wide Q8 prefill route delivers **+8.52% prefill and +2.08% decode** at
layers 8-47 (see `results/2026-10-04-qwen4exp-dense-wide-8-47-speed/`), but its
gate failure has to be resolved by scope choice or by arithmetic. **This closes
the scope-choice branch.**

## The frontier, ordered by scope

| scope | `kl_mean` (1e-3) | top-1 (0.99) | `general_en` mean (1e-3) | verdict |
| --- | ---: | ---: | ---: | --- |
| 8-47 | 8.839e-4 | 0.99096 | **1.1812e-3** | FAIL `general_en` 1.18x |
| **10-47** | 8.370e-4 | **0.98708** | **1.0752e-3** | **FAIL top-1 and `general_en` 1.08x** |
| 12-47 | 6.406e-4 | 0.98966 | — | FAIL top-1 by one row (1532/1548) |
| 14-47 | 5.190e-4 | — | — | **PASS** |
| 16-47 | 4.527e-4 | — | — | PASS (default) |

**10-47 is *worse* than 12-47 on top-1** (0.98708 against 0.98966), so the failure
is not a narrow band around 12 and there is **no passing scope between 14-47 and
0-47**.

**The passing frontier is 14-47, worth only +1.69% prefill** - below this host's
same-code drift floor of 1.38% (recorded in the scope ladder artifact), so it is
not a retainable speed claim either.

## Consequence

**The +8.52% at 8-47 is real but inaccessible by scope choice.** It is gated by
numerical drift introduced by layers 8-13 that no narrower scope avoids. The route
must be made **numerically admissible, not narrowed.**

The recorded mechanism is that the drift enters through **decode**, because the
route changes the *rounding* of the projections feeding the KV cache - the
`prefill_last` scope scores `kl_mean` ~1e-13 while `steady` carries all of it.
Candidate fixes are therefore a more accurate accumulation or a wider intermediate
in the wide route for the low layers, or restricting the route to the projections
that do not feed the KV cache.

## 10-47 in full

`scripts/execution_profile_q8_wmma_prefill_layers_gate.py --route dense_wide
--layers 10-47 --decode-steps 128 --repeat-runs 3 --execution-profile production
--prefill-chunk-size 1024 --require-cached-build`

`kl_mean` **8.370e-4** (limit 1e-3, PASS), `kl_p95` **4.291e-3** (5e-3, PASS),
`kl_max` **3.554e-2** (5e-2, PASS), top-1 **0.98708** (0.99, **FAIL**),
`hard_gates_passed` **false**.

| category | mean KL | top-1 |
| --- | ---: | ---: |
| code | 6.268e-4 | 0.98708 |
| general_en | **1.0752e-3** | 0.97416 |
| general_ja | 9.980e-4 | 0.99483 |
| mixed_ja_en | 6.481e-4 | 0.99225 |

## Method note

The scope gate **cannot** be run by editing `PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS`
in place: it raises `profile qualification requires a clean committed tracked
tree`. The correct harness takes the scope as an argument
(`--route dense_wide --layers N-M`) and needs no edit. That invocation is recorded
in the scope ladder artifact's `provenance.command` field.

## Limits

One host (gfx1151 Strix Halo, Radeon 8060S, not the gfx1100 default hardware),
one model (`unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL`, UD-Q4_K_XL), one profile, one
chunk size (1024), 3 candidate repeats. 13-47 was not run, so the frontier is
pinned to within one layer rather than to an exact layer. The 8-47, 12-47, 14-47
and 16-47 rows are quoted from the recorded scope ladder; only 10-47 was measured
here.
