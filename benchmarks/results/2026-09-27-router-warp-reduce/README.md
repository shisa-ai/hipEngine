# Router warp-per-token reduction: same-session A/B

**Date:** 2026-09-27
**Verdict: REJECTED by the production gate.** The kernel win is real and large -
`+1.65%` prefill (`247.739 -> 251.823` tok/s, `-0.272 s`), with the router kernel
itself `2.16x` faster - and it still fails admission. The gate rejects it on
top-1 agreement (98.67% against a 99% floor) and on mean KL in the `prefill_last`
scope (`3.515e-3` against a `1e-3` budget), and **all six top-1 mismatches are
flip-eligible**, so the cause is the router's discrete expert selection
amplifying a 1-ulp logit difference, not arithmetic drift. See section 6.

## Protocol

Same-host, same-session two-arm A/B on `code-p4096` (prefill only, 1024-token
chunks, four chunks). Both arms are the shipped production profile; the arm is
selected by the harness's post-binder `--override`, which records the arm in the
artifact.

| | |
| --- | --- |
| Host | `gfx1151`, machine `55ea6c509d0b49eea8de7094a1023668`, kernel `7.1.6-1-cachyos` |
| GPU | AMD Radeon 8060S (Strix Halo APU), ROCm platform `10.0.0` |
| Model | `unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL`, BF16 KV |
| Case | `code-p4096`, 4096 prompt tokens, fixture sha256 `42b562bd...` |
| Harness | `scripts/qwen4exp_profile_gap.py --mode prefill --case-id code-p4096 --prefill-chunk-size 1024` |
| Source head | `5aaafb52d` (the arm wiring is working-tree, see Provenance) |

```bash
/tmp/run-gpu.sh scripts/qwen4exp_profile_gap.py \
  --model-root <UD-Q4_K_XL> --mode prefill --case-id code-p4096 \
  --repetitions 3 --prefill-chunk-size 1024 \
  --override HIPENGINE_QWEN4_EXP_ROUTER_VARIANT=<variant> \
  --output plain-<arm>.json
```

## 1. Wall clock (the decision metric)

Three repetitions per arm, unprofiled:

| arm | variant | tok/s | median s | per-run walls (s) |
| --- | --- | ---: | ---: | --- |
| A | `f32_hidden_token_tile4_dense_exact` (shipped) | 247.739 | 16.534 | 16.546 / 16.534 / 16.521 |
| B | `f32_hidden_warp_reduce_dense_exact` (candidate) | **251.823** | **16.261** | 16.261 / 16.250 / 16.284 |

`+4.083 tok/s`, `+1.65%`, `-0.272 s` per prefill. Arm B's slowest run is 0.237 s
faster than arm A's fastest, so the two arms do not overlap. Within-arm spread is
0.15% (A) and 0.21% (B).

## 2. Kernel proof (the arms really differ)

The launch census cannot answer this — routed/MoE kernels do not record. The
kernel trace can, and it shows two different kernels at identical launch counts:

| arm | kernel | 1x ms | n | ms/launch |
| --- | --- | ---: | ---: | ---: |
| A | `qwen35_router_logits_f32_token_tile_dense_exact_kernel` | 619.45 | 192 | 3.226 |
| B | `qwen35_router_logits_f32_warp_reduce_dense_exact_kernel` | **287.11** | 192 | **1.495** |

`2.158x` on the router kernel, saving `332.3 ms`. Total kernel time over the same
traces: `16189.5 -> 15759.3 ms` (`-430.2 ms`).

The profiled delta (`-430 ms`) is larger than the unprofiled one (`-272 ms`).
The unprofiled figure is the decision metric; the profiled one is attribution.

## 3. Scaling note - the raw CSV holds two prefill passes

`--repetitions 1` still produces **384** router launches, not 192, because the
child runs a warmup pass and a measured pass and rocprofv3 traces both. The
authority for the 1x count is
`2026-09-17-qwen4exp-shipped-default-attribution/role-analysis.json`
(`exact_role_kernels`: `qwen35_router_logits_f32_token_tile_dense_exact_kernel`
rows = 192 = 48 MoE layers x 4 chunks).

**Every `ms` and `n` in this artifact is 1x. Divide the raw CSV totals by 2.**

This artifact was itself written after finding the same class of error twice
elsewhere: `2026-09-27-qsa-prefill-grid-resolution` and
`2026-09-27-moe-reduce-attribution` both halved `ms` but left `n` at the raw 2x
value. Both are corrected.

## 4. Provenance caveats

- **The manifest hash does not distinguish the arms.** Both arms report
  `manifest_sha256 35e57360...` because `--override` is applied after the profile
  binder. The arms are distinguished by the `overrides` field and by
  `configuration_class: diagnostic_post_binder_override` with
  `named_profile_intact: false`. Reading the shared manifest as "the arms were
  identical" is the mistake that produced the first, invalid version of this A/B.
- **The first A/B was invalid.** It exported the variant on the command line, and
  `/tmp/run-gpu.sh` builds the child environment with `env -i`, which drops it.
  The committed `warmup-*.json` artifacts show it: `route_env` carries no
  `HIPENGINE_QWEN4_EXP_ROUTER_VARIANT` at all, and the two arms read 246.259 and
  246.265 tok/s - a 0.002% difference between arms that were supposed to differ
  by 2x on the router. The runner also hardcoded the variant at the router call
  site, so the profile-level switch was inert regardless. Fixed before this run:
  the runner resolves the variant at launch and raises on an unrecognized value.
  The real arms are distinguishable in the same field: `plain-a.json` records
  `f32_hidden_token_tile4_dense_exact`, `plain-b.json` records
  `f32_hidden_warp_reduce_dense_exact`.
- The arm wiring is a working-tree change, so `source.head` is the base commit
  rather than the measured tree.
- `--override` is a diagnostic class. A promotion run must select the candidate
  through the profile (so the manifest carries it), which is why the binder now
  publishes the resolved variant.

## 5. What is not established

- ~~**Numerical admission.**~~ Resolved by the gate run in section 6: rejected.
- The screen is one case (`code-p4096`) on one host. `docs/BENCHMARK.md`
  "Anti-gaming" requires the full multi-prompt suite for a retained claim; this
  is a screen, not a retained benchmark row.

## 6. Production gate: rejected

`scripts/qwen4exp_layer2_profile_gate.py --candidate router_warp_reduce` over 450
rows. `measurement_valid: true`, `candidate_dispatch_calls: 4320` across
`(rows, 2560, 512)` shapes with rows 39-52, so the candidate genuinely ran - the
route admits at `rows >= 2` and the suite's 39-71 token prompts clear it. This is
not the silent no-op that `docs/REFACTOR.md` warns about for `rows > 256` routes.

| gate | result |
| --- | --- |
| numerical | **FAILED** |
| task | **FAILED** (`requires_review`, 12/18 exact) |
| state / determinism | passed (3 runs, byte-identical trajectories) |
| lifecycle | passed |

| metric | value | budget | |
| --- | ---: | ---: | --- |
| mean KL | 5.714e-4 | 1e-3 | pass |
| p95 KL | 2.322e-3 | 5e-3 | pass |
| p99 KL | 6.228e-3 | 2e-2 | pass |
| max KL | 3.344e-2 | 5e-2 | pass |
| **top-1 agreement** | **98.67%** (444/450) | >= 99% | **fail** |

`scope_failures`: `category/mixed_ja_en`, `shape/prefill_last`,
`transition/prefill_to_c1`. The two `prefill_last` scopes fail on **mean KL
3.515e-3**, 3.5x over budget, not on top-1 (100% there). `mixed_ja_en` fails
narrowly at mean KL 1.083e-3.

### The failure mechanism is discrete, not drift

The router candidate runs only on the 18 `prefill_last` rows; the 432 `c1` rows
take the linear path and inherit their divergence from the prefill state.

- `flip_eligible_rows`: 48 (10.7%)
- `top1_mismatches_flip_eligible`: **6**
- `top1_mismatches_outside_flip_eligible`: **0**
- `max_abs_logit_delta`: 1.673

The router logits feed `qwen35_router_select`, so a near-tie can change which of
the 512 experts is selected. That turns a 4.9e-7 logit difference into an O(1)
model-logit change: every top-1 mismatch sits on a row where strict's own top-1
margin is small enough for the observed drift to explain it. The KL envelope
passes comfortably at 57% of budget while top-1 fails, which is the signature of
a discrete amplification node rather than a numerically sloppy kernel.

**Tuning the kernel cannot fix this.** Any change to the router's reduction order
moves the logits by ~1 ulp and will flip the same near-ties. The route is closed
as arithmetic; the reopening path is to make the *selection* invariant, which
re-baselines strict and is a campaign-level decision. See `docs/REFACTOR.md`
"Router Producer Variant".

The task gate's 6 divergences are stylistic ("generation process" against
"decoding process", docstring and section-heading differences), but the gate
counts them.

## Files

| file | content |
| --- | --- |
| `plain-a.json`, `plain-b.json` | unprofiled 3-rep wall clock; `route_env` carries the arm |
| `trace-a_kernel_trace.csv`, `trace-b_kernel_trace.csv` | rocprofv3 kernel traces, 2 passes each - **gitignored raw dumps** (`.gitignore` line 66), not committed; the derived per-kernel figures are in section 2 |
| `child-a.json`, `child-b.json` | profiled child reports |
| `production-gate.json` | the 450-row production gate: rejection and mechanism |
| `warmup-a.json`, `warmup-b.json` | the invalid first attempt, kept because `route_env` is the evidence that the env var was dropped |
| `hipcc-version.txt` | compiler identity for the cached-build guard |

The raw traces are not committed, so the per-kernel `ms`/`n` table in section 2
cannot be re-derived from this directory alone. It is reproducible with
`rocprofv3 --kernel-trace` on the same command; the launch counts are checkable
independently against `role-analysis.json` `exact_role_kernels`.
