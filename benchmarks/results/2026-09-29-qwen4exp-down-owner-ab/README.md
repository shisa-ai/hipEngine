# Q5_1 expert-down owners: the production route is the measured-fastest of its three owners

The three registered owners of `moe_linear` /
`prefill_rows_ge512_k640_exact_grouped_q5_1_down` are timed against each other on the main
kernel. **The owner production actually runs is faster than both alternatives in all 12
cases**: -21.0% to -79.2% against the owner it replaced (median -50.4%), and +82.5% to
+562.0% against the strict default.

So this is a **validation, not an opportunity**. No promotion is available here: the route
is already the measured-best of its three owners.

**Correction to this packet's first version.** It originally read as "a registered owner that
no manifest entry selects is faster than the production default". That was **wrong**. The
variant is selected by no `_selection()` entry, but production does run it, because
`hipengine/generation/qwen4_exp_profiles.py:712` binds
`"HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT": "1" if production else "0"` and
`hipengine/runtime/qwen4_exp_runner.py:4574` then takes the `q51_iu8_exact` branch, which
calls `q51_iu8_risk_down` and **bypasses the variant-selection path entirely**. Scanning
`_selection()` blocks alone cannot see an env-gated route - a manifest-reading trap worth
remembering. The measurements below were unaffected; only the interpretation was.

## The three owners

| owner | design | what runs it |
| --- | --- | --- |
| `selected_wmma_iu8_risk_prefill_bf16_bf16_out` | approximate iu8 WMMA; carries `risk_count` / `risk_indices`, repaired by `selected_sparse_exact_repair_row_publish` | **production** - via `HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT=1`, set by the production profile, not by a `_selection()` entry |
| `selected_grouped_prefill_pair2_row_publish_bf16_bf16_out` | exact; no risk/repair arguments at all | the `_selection()` entry, i.e. the strict path and the pre-iu8 production owner |
| `selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out` | exact | strict default and production fallback |

The first owner is the family the 2026-09-29 ED-1 artifacts screened for LDS staging. **ED-1
compared staged prototypes against that owner but never against the `pair2` owner it
replaced, which is what this packet supplies.**

## Result

Physical host `gfx1151`, machine ID `55ea6c509d0b49eea8de7094a1023668`, kernel
`7.1.6-1-cachyos`, Radeon 8060S, TheRock `therock10-staging-20260828` - the same physical
host as the 2026-09-29 ED-1 artifacts. Geometry `in=640`, `out=2560`, 512 experts, Q5_1
(480-byte weight rows). Fifteen HIP-event samples after two primed launches per arm; the
primary arm is measured again last to expose drift.

| case | pair2 (production) ms | compact8 ms | wmma ms | compact8 vs prod | wmma vs prod | drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced-512 | 6.909 | 12.611 | 5.418 | +82.5% | **-21.6%** | -0.02% |
| balanced-1024 | 7.235 | 13.714 | 5.493 | +89.6% | **-24.1%** | +0.21% |
| balanced-4096 | 9.826 | 25.762 | 5.821 | +162.2% | **-40.8%** | +0.66% |
| balanced-10240 | 17.263 | 61.236 | 8.904 | +254.7% | **-48.4%** | +1.09% |
| sparse-512 | 2.052 | 4.167 | 1.621 | +103.1% | **-21.0%** | -0.04% |
| sparse-1024 | 2.245 | 5.567 | 1.625 | +148.0% | **-27.6%** | +1.89% |
| sparse-4096 | 5.916 | 23.094 | 2.814 | +290.4% | **-52.4%** | -0.05% |
| sparse-10240 | 13.598 | 60.205 | 6.463 | +342.7% | **-52.5%** | -0.14% |
| single-512 | 1.286 | 6.638 | 0.398 | +416.2% | **-69.0%** | -4.53% |
| single-1024 | 2.411 | 14.259 | 0.695 | +491.3% | **-71.2%** | -0.27% |
| single-4096 | 9.582 | 61.781 | 2.560 | +544.8% | **-73.3%** | -0.13% |
| single-10240 | 30.640 | 202.825 | 6.365 | +562.0% | **-79.2%** | -0.17% |

What this establishes:

- **Both production choices on this route are validated by direct measurement.** The iu8
  owner beats the `pair2` owner it replaced by 21.0-79.2% on the main kernel, and the `pair2`
  owner beats the strict fallback by 82.5-562.0%.
- **The main-kernel gap explains the recorded operation-complete gain.** The 2026-09-08
  promotion `a57a3e418` recorded +1.9-3.0% prefill per case over 12 cases with 72/72
  trajectories exact and a 12/12 bit-identical state gate. **The difference between -50.4%
  main-kernel and +1.9-3.0% operation-complete is the repair**, which this packet excludes
  and production does pay.
- **The owners of this route had never been put in an arms pair**, so none of this was
  previously recorded.

## What the repair costs, and why the main-kernel gap is not a route saving

The `pair2` owner is exact and has **no repair pass**; the iu8 owner is approximate and its
correctness depends on the sparse repair the runner launches after it. That repair is not
timed here, so the -50.4% is not a route-level saving - and it is why the operation-complete
figure is +1.9-3.0% rather than ~50%.

Sizing it from recorded numbers, and labelling every step:

| term | value | provenance |
| --- | ---: | --- |
| q5_1 down main-kernel total | 1,482.958 ms | `q51_fast_ms`, `2026-09-29-qwen4exp-repair-rate/repair-rate.json` |
| q5_1 repair (inferred) | 661.21 ms | `q51_repair_ms`, **the same artifact marks this an inference** - "q5_1 route rate NOT observed; the following assumes it matches q4_k" |
| wmma main kernel, scaled by this packet's median | ~741 ms | 1,482.958 x (1 - 0.504), from the measured ratio above |

So a rough route-level comparison is `1,482.958` against `~741 + 661.21 = ~1,403 ms`, i.e.
**roughly 80 ms, or about 5.4% of the route** - an order of magnitude smaller than the
main-kernel figure, and built on two uncertain terms.

**The repair figure is the weaker one.** It assumes the q5_1 route carries the q4_k risk
rate; that rate has never been observed for q5_1, and the risk rate is a property of the
arithmetic and activations, not of the main kernel, so it may or may not hold.

**What would settle it:** measuring the iu8 owner end-to-end including its repair at real
routing, with the q5_1 risk rate observed rather than assumed. **This is confirmation, not a
promotion path** - the CHANGELOG already records the operation-complete result, so the
experiment would only quantify the repair. A future down-route lever would have to attack
the repair cost itself, which is `EG-3`'s subject and is closed there.

## Limits

- **No correctness gate was run.** No output was compared between any two owners, so
  nothing here says the wmma owner is numerically acceptable - and it is by construction an
  approximate path that depends on its repair, which this packet does not exercise.
- **Weights are synthesized** at the real shapes and byte layout; the access pattern and
  byte counts are representative, the values are not. Populations come from the harness's
  own count synthesizers, so **there is no real-router population**.
- `max_risks=0`, so the risk queue is never written for the wmma arm. No repair, no packing,
  no engine work.
- Main kernel only. Not a complete-owner, route or engine rate.
- One host, one geometry, one quant, one process.
- `single-512` repeats at 4.53% drift, the largest in the sweep; every other case repeats
  within 1.89%. The `single` distribution is degenerate and included only to bracket
  parallelism behaviour.
- The `1,482.958` and `661.21` figures come from a different measurement context than this
  packet. They are used only to size the repair gap, and the resulting ~5.4% is an
  estimate, not a measurement. This ledger carries three prefill-window denominators
  (22,698.6 / 16,319.9 / 15,742.0 ms, a 44% spread), so the two are not treated as
  interchangeable.

## Reproduce

```bash
E=/home/lhl/miniforge3/envs/therock10-staging-20260828/lib/python3.12/site-packages
export PATH="$E/_rocm_sdk_devel/bin:$E/_rocm_sdk_core/bin:$PATH"
export HIP_PATH="$E/_rocm_sdk_devel"
export LD_LIBRARY_PATH="$E/_rocm_sdk_devel/lib:$E/_rocm_sdk_core/lib:$LD_LIBRARY_PATH"
/home/lhl/miniforge3/envs/therock10-staging-20260828/bin/python3 \
  scripts/qwen4exp_moe_owner_ab.py --route down \
  --output benchmarks/results/2026-09-29-qwen4exp-down-owner-ab/artifact.json \
  --rows 512,1024,4096,10240 --distribution balanced,sparse,single --repetitions 15
```

The same harness runs the gate/up route with `--route gate-up`; its result is
`../2026-09-29-qwen4exp-gate-up-owner-ab/README.md`.
