# Q5_1 expert-down owners: an unselected registered owner is faster on the main kernel, and the repair it needs is not timed here

The three registered owners of `moe_linear` /
`prefill_rows_ge512_k640_exact_grouped_q5_1_down` are timed against each other on the main
kernel. **`selected_wmma_iu8_risk_prefill_bf16_bf16_out` - which no manifest entry selects -
is faster than the production default in all 12 cases, median -50.4%.** The strict default
is slower in all 12, median +272.6%.

**This is not a promotion candidate as it stands.** The winning arm is an
approximate-then-repair design and the repair pass is **not timed here**, so the number
below is the whole owner cost for the production default and only part of it for the wmma
owner. The section "What the repair costs" sizes that gap and shows it is most of the
effect. No correctness gate was run and no default changed.

## The three owners

| owner | design | selected for this shape? |
| --- | --- | --- |
| `selected_grouped_prefill_pair2_row_publish_bf16_bf16_out` | exact; no risk/repair arguments at all | **production default** |
| `selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out` | exact | strict default and production fallback |
| `selected_wmma_iu8_risk_prefill_bf16_bf16_out` | approximate iu8 WMMA; carries `risk_count` / `risk_indices` | **no manifest entry selects it** |

The third owner is registered (`KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1",
"selected_wmma_iu8_risk_prefill_bf16_bf16_out")`) and is the family the 2026-09-29 ED-1
artifacts screened for LDS staging. **ED-1 compared staged prototypes against that owner but
never against the owner production actually selects for this shape.**

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

Two things this establishes independent of the repair question:

- **The production default is a good choice against its own fallback.** The strict default /
  production fallback is slower in 12 of 12 cases by +82.5% to +562.0%. That is a measured
  justification for the promotion that the archived evidence
  (`2026-09-07-framework-qwen4exp-q4-iu8-exact-production.json`) did not supply for this
  route, since its arms compared against `selected_gemv_bf16_bf16_out`.
- **The owners of this route had never been put in an arms pair**, so none of this was
  previously recorded.

## What the repair costs, and why the headline number is not the whole story

The production default is exact and has **no repair pass**; the wmma owner is approximate
and its correctness depends on one. Its repair is not timed here, so the -50.4% is not a
route-level saving.

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

**What would settle it:** measuring the wmma owner end-to-end including its repair at real
routing, with the q5_1 risk rate observed rather than assumed. That is a bounded
experiment and the only thing that can turn this into a promotion candidate.

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
