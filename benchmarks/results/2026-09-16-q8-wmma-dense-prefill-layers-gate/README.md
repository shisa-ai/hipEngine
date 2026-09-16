# f16 WMMA dense Q8_0 prefill: layer-scope numerical gate

The f16 WMMA dense Q8_0 prefill route was gated against the exact coltile chain
at four layer scopes on the Qwen4Exp UD-Q4_K_XL canonical exact-token fixture.

| Scope | Cases | Rows | Mean KL | p95 KL | p99 KL | Max KL | Top-1 | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| layers 16-47 | 12 (4 categories) | 1548 | 3.80e-4 | 1.76e-3 | 5.42e-3 | 1.53e-2 | 0.99354 | **pass** |
| layers 20-47 | 12 (4 categories) | 1548 | 2.53e-4 | 1.19e-3 | 3.70e-3 | 1.99e-2 | 0.99419 | **pass** |
| layers 28-47 | 12 (4 categories) | 1548 | 9.34e-5 | 4.06e-4 | 1.51e-3 | 9.67e-3 | 0.99742 | **pass** |
| layers 32-47 | 12 (4 categories) | 1548 | 5.81e-5 | 1.75e-4 | 6.90e-4 | 1.36e-2 | 0.99677 | **pass** |
| layers 0-47 | 3 (`code` only) | 387 | 1.099e-3 | 5.017e-3 | 1.510e-2 | 3.913e-2 | 0.9922 | fail |
| limit | | | 1e-3 | 5e-3 | 2e-2 | 5e-2 | 0.99 | |

**Layers 16-47 is the deepest certified scope.** It passes every calibrated
gate with 2.6x headroom on the mean and 2.8x on p95, top-1 1538/1548, three
identical trajectory hashes, no scope failures, and `measurement_valid: true`
with no blockers. All ten top-1 misses are inside the 154-row flip-eligible set
and none outside it. Its saving is **not measured**: this gate records no
timing, so the byte-share interpolation of ~+4.42 s remains a prediction.

The 16-47 screen overestimated the full arm by 1.8x (`6.728e-4` against
`3.796e-4`), the second deep-scope control point to do so, which is why screens
are read as an interval rather than through a single factor.

**Layers 20-47 was the deepest certified scope until 16-47.** It passes every
calibrated gate with 4.0x headroom on the mean and 4.2x on p95, top-1
1539/1548, three identical trajectory hashes, no scope failures, and
`measurement_valid: true` with no blockers. Its nine top-1 misses are all inside
the 135-row flip-eligible set and none outside it. The **+4.15 s** quoted for it
is a byte-share interpolation, not a measurement; see
[`../2026-09-16-q8-wmma-layers-recoverable-time/README.md`](../2026-09-16-q8-wmma-layers-recoverable-time/README.md).

The first 20-47 run was invalidated on provenance grounds — `scripts/check_lineage.py`
was edited while it was in flight, which `execution_affecting_paths` named
directly. The clean-tree re-run reproduced it exactly: identical strict and
candidate logit digests and a byte-identical quality summary, which also
demonstrates cross-process determinism of the gate.

**Layers 28-47** It passes every calibrated
gate with 10.7x headroom on the mean and 12.3x on p95, `hard_gates_passed` and
`eligible_for_automatic_admission` both true, zero scope failures, and — unlike
the two arms above it — `measurement_valid: true` with no qualification
blockers, because the provenance rule now distinguishes a documentation-only
dirty worktree from one that can change execution. Its **+3.6 s** is a byte-share
interpolation against 32-47's **measured +2.436 s**.

The certified scope passes every calibrated gate with 17x headroom on the mean
and every category, shape and transition scope passing individually. The maximal
scope fails the mean and p95 limits marginally, and that failure is scope, not
arithmetic: extending the route from layer 32 down to layer 0 multiplies the
`code`-category mean KL by 53x (2.08e-5 to 1.099e-3).

`wmma_prefill_f32_f32_out` is the pre-existing variant this gate selects. The
`dense_wide256` kernel developed in
`../2026-09-16-dense-wide-q8-prefill-candidate/` is bit-identical to it on the
one packet compared so far (`layers.8.attn_qkv`, K2560/N10240), so this result
transfers to that kernel only once per-shape bit-identity across the model's Q8
dense shapes is established.

## Protocol

| | |
| --- | --- |
| host | `gfx1151`, AMD Radeon 8060S Graphics |
| model | unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL`, fingerprint `fb1f2fbf73d588c9…` |
| profile | production, with the Q8 dense prefill selector flipped post-binder |
| fixture | `qwen4exp_canonical_ar_p512_p1024_p4096.json` |
| teacher | the same profile with the selector cleared: exact `coltile8_rowbatch4_f32_f32_out` |
| candidate | `wmma_prefill_f32_f32_out`, `HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` = `32-47` or `0-47` |
| shapes | `p512`, `p1024`, `p4096` |
| repeats | 3 candidate runs, teacher-forced onto the strict token chain |

The teacher is the production profile with the changed route disabled, not the
strict profile. That isolates the Q8 dense prefill route's own contribution, the
same candidate-local convention the Q8 MMQ plane gate uses. Absolute
strict-profile comparison remains a separate step.

The selector must be applied **after** the named production binder: the binder
writes `HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` as `""` during its own pass, so a
value set before generator construction is silently inert. The gate constructs
the generator first and flips the selector per arm.

The gate loads `libamdhip64.so` by bare soname, so it needs the TheRock library
paths on `LD_LIBRARY_PATH`; without them it fails at generator construction with
`OSError: libamdhip64.so: cannot open shared object file`. The recorded
provenance carries `HIPENGINE_HIP_ARCH` but not `LD_LIBRARY_PATH`, so this is not
recoverable from an artifact. Use the host setup in
[`docs/THEROCK.md`](../../../docs/THEROCK.md) rather than a literal ROCm prefix:

```bash
ENV_PREFIX=/home/lhl/miniforge3/envs/therock10-staging-20260828
PY=$ENV_PREFIX/bin/python
SITE=$ENV_PREFIX/lib/python3.12/site-packages
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIPENGINE_HIP_ARCH=gfx1151

# layers 32-47, all 12 cases -> artifact-layers32-47-4cat.json
$PY scripts/execution_profile_q8_wmma_prefill_layers_gate.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --decode-steps 128 --repeat-runs 3 --layers 32-47 --prefill-chunk-size 1024 \
  --output benchmarks/results/2026-09-16-q8-wmma-dense-prefill-layers-gate/artifact-layers32-47-4cat.json

# layers 0-47, code cases only -> artifact.json
$PY scripts/execution_profile_q8_wmma_prefill_layers_gate.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --case-id code-p512 --case-id code-p1024 --case-id code-p4096 \
  --decode-steps 128 --repeat-runs 3 --layers 0-47 --prefill-chunk-size 1024 \
  --output benchmarks/results/2026-09-16-q8-wmma-dense-prefill-layers-gate/artifact.json
```

One full arm is roughly twenty minutes of wall time on a quiet host, of which a
few minutes are the 111 GB model load. Do not add `--require-cached-build`: it is
not part of this gate's protocol, and it fails closed on any kernel that has not
been built under the current environment's cache key. Compiling on demand with
the same `hipcc` produces the same kernels.

## Layers 32-47: pass

1548 full-vocabulary rows, 12 cases, 4 categories, 3 deterministic repeats.

| Gate | Value | Limit | Verdict |
| --- | ---: | ---: | --- |
| mean KL | 5.808e-5 | 1e-3 | pass (17x) |
| p95 KL | 1.751e-4 | 5e-3 | pass (29x) |
| p99 KL | 6.895e-4 | 2e-2 | pass (29x) |
| max KL | 1.362e-2 | 5e-2 | pass (3.7x) |
| top-1 agreement | 1543/1548 = 0.99677 | 0.99 | pass |
| top-5 overlap | 0.99406 | — | — |
| teacher NLL delta | +7.09e-6 | — | — |
| repeat determinism | 3/3 identical trajectory hashes | exact | pass |
| outlier review | 0 rows over 2e-2 | none | not requested |

`hard_gates_passed` and `eligible_for_automatic_admission` are both true with no
scope failures.

| Category | Rows | Mean KL | p95 KL | Max KL | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `code` | 387 | 2.076e-5 | 1.102e-4 | 5.627e-4 | 1.00000 |
| `general_en` | 387 | 1.250e-4 | 2.625e-4 | 1.362e-2 | 0.98966 |
| `general_ja` | 387 | 5.036e-5 | 1.935e-4 | 4.077e-3 | 0.99742 |
| `mixed_ja_en` | 387 | 3.615e-5 | 1.666e-4 | 1.081e-3 | 1.00000 |
| shape `c1` (decode) | 1536 | 5.853e-5 | 1.784e-4 | 1.362e-2 | 0.99674 |
| shape `prefill_last` | 12 | 3.54e-9 | 1.91e-8 | 3.27e-8 | 1.00000 |

`general_en` carries the tail and all five top-1 disagreements; `code` and
`mixed_ja_en` are exact. That ordering matches the 2026-08-29 certification for
the same scope, where `general_en` was also the worst category.

### Relationship to the 2026-08-29 certification

`../2026-08-29-gfx1151-qwen38-flash-next-moe27-q8-32-production.json` records
"dense raw-Q8 WMMA layers 32-47" at 450 rows over four categories with mean KL
`1.197e-4`, top-1 `449/450`, and `hard_gates_passed: true`; its `code` category
mean was `4.57e-5`.

This run agrees and tightens it on 3.4x more rows: mean KL `5.81e-5` (2.1x
lower), `code` mean `2.08e-5`, top-1 `1543/1548`. The top-1 miss rates (0.32%
here against 0.22% there) are consistent within sampling noise at these sample
sizes. The certified scope therefore reproduces on the current profile stack from
an independent harness.

That artifact also carries a suffix screen showing where the cost sits:

| Layer suffix | Mean KL |
| --- | ---: |
| 27-47 | 1.26e-3 |
| 32-47 | 7.26e-5 |
| 36-47 | 1.43e-4 |
| 40-47 | 4.13e-5 |
| 44-47 | 5.65e-6 |

The all-layer `code`-only mean of `1.099e-3` measured here lands on the suffix27
value, so extending below layer 32 adds little beyond what layers 27-31 already
cost. The two scopes are consistent with each other and with that screen.

## Layers 0-47: fail

387 rows, `code` cases only, 3 of the fixture's 12 cases.

| Gate | Value | Limit | Verdict |
| --- | ---: | ---: | --- |
| mean KL | 1.099e-3 | 1e-3 | **fail** |
| p95 KL | 5.017e-3 | 5e-3 | **fail** |
| p99 KL | 1.510e-2 | 2e-2 | pass |
| max KL | 3.913e-2 | 5e-2 | pass |
| top-1 agreement | 384/387 = 0.9922 | 0.99 | pass |
| repeat determinism | 3/3 identical trajectory hashes | exact | pass |
| outlier review | 1 row over 2e-2 | none | requested |

The failures are marginal: 10% over the mean limit and 0.3% over the p95 limit.
This artifact is also `code`-only, so it is a screen rather than the
retention-grade protocol, and its `provenance.command` field is a character list
rather than a command list — the gate joined its argument vector into one string
where the provenance schema wants a sequence. The gate is fixed and the 32-47
artifact carries a correct `command`; the exact command for this one is in the
protocol section above.

## Open anomaly: the prefill row is nearly unaffected

At both scopes the prefill row's distribution is nearly unchanged (layers 32-47:
KL 3.54e-9, teacher NLL delta -6.4e-11; layers 0-47: KL 1.09e-11, NLL delta
5.2e-11) while its logits are not identical (max absolute logit delta 1.86 and
1.34 respectively, on entries whose probability is negligible). All 12 prefill
rows are within 3.3e-8 of the teacher's distribution while the decode rows carry
essentially all of the measured drift. The asymmetry reproduces at both scopes
and across all 12 cases, so it is a property of the model and harness rather than
of one layer set.

Two readings are open and neither is measured:

- The dense Q8 prefill perturbation is real but its effect on the prefill's
  last-token distribution is small, and the drift is amplified by the decode
  state. If so the measured mean and p95 are the honest cost of the route.
- Part of the prefill path is not taking the candidate arithmetic — for example
  a captured MoE graph replaying the teacher's kernels — in which case the
  measured drift is a lower bound and the route is only partly exercised.

A counted kernel census per owner over one prefill pass is the required next
evidence before either reading is adopted. The layer-level packet comparison does
show the route is live at the leaf: on the same `layers.8.attn_qkv` packet the
exact and WMMA routes differ elementwise by max 2.65e-3 with mean signed 1.2e-7
and no constant shift, which is the f16 rounding signature rather than a
scale/offset bug.

## Provenance

Both artifacts record `untracked_dirty`, so `measurement_valid` is false on
provenance grounds even where every numerical gate passed. An untracked
`docs/superpowers/` directory belonging to a different workstream is present in
the worktree. It is not removed here because it is not this workstream's to
remove.

## What this does not establish

- No promotion of `dense_wide256`. Per-shape bit-identity against
  `wmma_prefill_f32_f32_out` across the model's Q8 dense shapes is the missing
  link, currently established on one packet.
- No statement about the current production default. The route gated here is the
  f16 WMMA dense Q8_0 *arithmetic class*; the named production profile currently
  selects the admitted MMQ stack for Q8 prefill.
- No performance claim. These are full-logit captures; leaf timings live in
  `../2026-09-16-dense-wide-q8-prefill-candidate/`.
- No task-quality, isolation, BF16-relative or c>N gate.

## Correction, 2026-09-16: how the two arms compare

Three statements above are corrected here rather than edited in place, so the
original reading and its correction are both visible.

**The arms were compared on different prompt mixes.** The 32-47 arm ran 12 cases
across four categories; the 0-47 arm ran three `code` cases. Any metric quoted
across the two therefore mixes a scope difference with a prompt-mix difference.
Restricted to the one category both arms ran, produced by
`scripts/q8_wmma_layers_gate_compare.py`:

| Metric | layers 32-47 | layers 0-47 | ratio |
| --- | ---: | ---: | ---: |
| mean KL | 2.076e-5 | 1.099e-3 | 52.9x |
| p95 KL | 1.102e-4 | 5.017e-3 | 45.5x |
| max KL | 5.627e-4 | 3.913e-2 | 69.5x |
| top-1 agreement | 387/387 | 384/387 | — |
| max abs logit delta | 1.0028 | 3.1656 | 3.2x |

**Adding layers 0-31 enlarges the perturbation itself; it does not merely expose
more near-ties.** The `1.86 / 1.34` pair quoted in the open-anomaly section is
the *prefill row* of two different case sets, not a scope comparison. The
category-level figures above are the controlled ones, and they show the largest
per-row logit perturbation growing 3.2x while the minimum teacher margin in the
0-47 arm's decode rows is *wider* than in the 32-47 arm (0.0308 against
0.0021). More rows crossing small margins therefore does not explain the
failure. The early layers change the arithmetic's output more, which is
consistent with an accumulation-range effect and is the hypothesis the layer
bisect should test.

**The "prefill is not taking the candidate arithmetic" reading is refuted by
these artifacts and needs no kernel census.** In the 32-47 arm the prefill row
carries the *largest* `max_abs_logit_delta` of any row in the run — 1.861 against
1.409 for the c1 rows. A bypassed route returns a zero delta, so the candidate
arithmetic is demonstrably live in prefill. The near-zero prefill KL is a
confidence artifact: `prefill_last` has `strict_margin_min` 14.67 (32-47) and
23.10 (0-47) against `c1`'s 0.0021 and 0.0308, and a perturbation of ~1.9 logits
cannot move a distribution whose top-1 leads by 14.7. What remains open is
*coverage* — which weights take the route — not liveness. That is bounded
statically by the selector's own filter, which requires `quant_key ==
"gguf_q8_0"` and a `layers.` slot path
(`hipengine/runtime/qwen4_exp_runner.py`), so the embedding, the output head,
and every Q4_K expert GEMM are excluded by construction.

Gates run after this correction record `flip_eligible_rows` and
`top1_mismatches_flip_eligible` per scope, which makes a mismatch count readable
without this reconstruction. These two arms predate that field.

### Provenance, restated

The `Provenance` section above understates the problem for the arm that passed.
`artifact-layers32-47-4cat.json` records `unstaged_dirty: true` with
`untracked_count: 5`; `artifact.json` records `unstaged_dirty: false` with
`untracked_count: 2`. So the passing arm ran against modified **tracked** files,
and the artifact does not say which — that is unrecoverable after the fact.
Artifact provenance is now schema 3 and records `dirty_paths`,
`untracked_paths`, and an `execution_affecting_dirty` flag that excuses
documentation-only dirt, so a re-run clears the untracked `docs/superpowers/`
objection on its own. Neither arm here can be retro-classified; re-running is
the only way to clear their provenance blocker.

### On the missing `dense_wide256` link

Per-shape bit-identity against `wmma_prefill_f32_f32_out` is not the promotion
bar — `docs/EXECUTION-PROFILES.md` makes production correctness the contract and
treats strict parity as a debugging oracle. Bit-identity is being used here to
*transfer* this verdict to another route without re-measuring, which requires
proving equivalence over an open-ended shape set from a single established
packet. Running this gate directly against `dense_wide256` is cheaper and does
not create a claim that has to be re-defended whenever a new shape appears.

## Route coverage: what this selector can own, read statically

`scripts/qwen4exp_q8_route_coverage.py` reads the selector's own filter
(`quant_key == "gguf_q8_0"` and a `layers.` slot path) off the GGUF index, so
the route's blast radius is a static fact rather than something a kernel census
has to discover. Artifact: `route-coverage.json`.

```
python3 scripts/qwen4exp_q8_route_coverage.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --scope 32-47 --scope 0-47 --scope 28-47 --scope 24-47
```

The route's candidate set is **499 Q8_0 tensors, 7.68 GiB**, all inside
transformer blocks.

| Scope | Tensors | Share | Route bytes | Share |
| --- | ---: | ---: | ---: | ---: |
| 32-47 | 166/499 | 33.3% | 2.83 GiB | 36.8% |
| 28-47 | 208/499 | 41.7% | 3.95 GiB | 51.4% |
| 24-47 | 249/499 | 49.9% | 4.24 GiB | 55.2% |
| 0-47 | 499/499 | 100% | 7.68 GiB | 100% |

Two exclusions are worth stating because they remove hypotheses rather than
just describing the model.

**The MoE experts are not Q8_0.** `ffn_gate_exps` and `ffn_up_exps` are Q4_K on
47 of 48 layers, and only five layers carry a Q8_0 `ffn_down_exps`. The expert
GEMMs are therefore excluded from this route by the quant filter. The
open-anomaly section's "a captured MoE graph replaying the teacher's kernels"
explanation is not needed to account for the MoE path not moving: those weights
were never eligible.

**The largest Q8_0 matrix in the model never takes the route.** `output.weight`
is 248320x2560 Q8_0, and together with `token_embd.weight`,
`output_hc_down.weight`, and `output_hc_up.weight` it is excluded by the
slot-path filter. Any prefill-time expectation for this selector should exclude
the LM head.

### The layer axis is periodic, so a bisect must respect it

The Q8_0 roles are not uniform across depth. Layers congruent to 3 mod 4
(3, 7, 11, ... 47) carry separate `attn_q` / `attn_k` / `attn_v` /
`attn_output`; the other 36 layers carry fused `attn_qkv` plus `attn_gate` and
`ssm_out`. A scope boundary that is not a multiple of 4 therefore changes the
attention-role composition of the arm as well as its depth, and the two effects
would be confounded. Bisect boundaries should be multiples of 4 — 28, 24, 20,
16 — which is also why 32-47 and 0-47 are cleanly comparable.

## What each scope is worth, measured

The performance half of this decision is now measured in
`../2026-09-16-q8-wmma-layers-recoverable-time/`, on the same host and model
with the `code-p4096` production prefill.

| Scope | Median s | Saved s | Speedup | Numerical verdict here |
| --- | ---: | ---: | ---: | --- |
| fallback | 23.840 | — | — | — |
| layers 32-47 | 21.405 | +2.436 | 1.114x | passes every calibrated gate |
| layers 0-47 | 16.799 | +7.041 | 1.419x | fails mean and p95 marginally |

The maximal scope is the largest single recoverable-time family measured in this
model, ahead of `Q8_IU8_WMM`'s +6.614 s. The admissible scope captures 34.6% of
it, so the numerical boundary between layer 0 and layer 32 is withholding about
4.6 s. That makes the layer bisect worth running on its own merits rather than
as a tidiness exercise, and `route-coverage.json` predicts what each boundary is
worth: recoverable time tracks route bytes owned to within 2.2 points.

## Lane

Everything here is `hip_gfx1151` / Radeon 8060S. Nothing in this directory
constrains gfx1100; that is a separate machine and a separate lane, and its
absolute rates must not be compared with these.

## The `dense_wide256` link, restated

The missing link is not evidence, it is a dispatch route. `dense_wide256_f32_f32_out`
is registered in the kernel registry
(`hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_dense_wide.py`, pulled into
gfx1151 through that backend's module list) but **nothing in
`hipengine/runtime/` or `hipengine/dispatch/` references it**, so no dispatch
path can select it and neither this gate nor any end-to-end measurement can
reach it. Establishing per-shape bit-identity would not change that. The next
step for that kernel is a selector plus its RED test and registered strict
fallback, after which this gate runs against it directly.

The arithmetic-class question is already settled in the candidate's own
evidence: `dense_wide256` is bit-identical to the pre-existing f16 WMMA family
on the compared packet, and both negative controls differ on 99.99% of elements.

## Layers 28-47: pass, and a monotonicity caveat

| Gate | Value | Limit | Headroom | Verdict |
| --- | ---: | ---: | ---: | --- |
| mean KL | 9.342e-5 | 1e-3 | 10.7x | pass |
| p95 KL | 4.063e-4 | 5e-3 | 12.3x | pass |
| p99 KL | 1.510e-3 | 2e-2 | 13.2x | pass |
| max KL | 9.671e-3 | 5e-2 | 5.2x | pass |
| top-1 agreement | 1544/1548 = 0.99742 | 0.99 | — | pass |
| repeat determinism | 3/3 identical trajectory hashes | exact | — | pass |
| scope failures | none | none | — | pass |

`hard_gates_passed`, `eligible_for_automatic_admission`, and `measurement_valid`
are all true and `qualification_blockers` is empty.

The flip-eligible population makes the four top-1 misses readable: 87 of 1548
rows (5.6%) have a teacher top-2 gap narrow enough for the observed
perturbation to close, all four misses fall inside that set, and none fall
outside it. The misses are margin-limited rather than evidence of drift.

**Adding layers 28-31 does not degrade every metric, so the layer axis is not
monotone.** Mean KL rises in all four categories, but at the deeper scope
`general_en`'s max KL *falls* from 1.362e-2 to 5.848e-3 and its top-1 agreement
*rises* from 0.98966 to 0.99225, and `general_ja`'s max KL falls from 4.077e-3
to 2.025e-3. A search that assumes more layers can only be worse is therefore
unsafe on the tail metrics; it holds so far on the category means, which is what
the screens in `scripts/qwen4exp_q8_wmma_layer_bisect.py` read. The practical
consequence is that the boundary is a mean-KL boundary, and a scope rejected on
a tail metric alone would need its own arm rather than an inference from a
neighbour.

## Bracketing the boundary with screens

The full protocol costs about twenty minutes per arm, so the admissible
boundary was bracketed with reduced-protocol screens and the full gate spent
only on candidates the screens had already placed. Driver:
`scripts/qwen4exp_q8_wmma_layer_bisect.py`. Ledger: `screens/ledger.json`.

A screen is this same gate at one prompt per category at p512 and 32 decode
steps — 132 rows against the full arm's 1548, roughly a twelfth of the cost.
`general_en` is mandatory in the screen set because it carries the tail and
every top-1 miss in the 32-47 arm.

### What a screen may decide

Screens decide on mean and p95 KL only. They do not decide top-1, and the
reason is recorded normatively in `docs/EXECUTION-PROFILES.md` 6.4: a rate gate
tested against the value it equals is near a coin flip at any sample size this
fixture can reach, and the per-category `0.97` floor is sharper still at screen
size, where 33 rows per category means a single miss scores `32/33 = 0.9697`.

That is not a hypothetical. The screen of layers **32-47 returned FAIL** on the
per-scope floor for a scope this same gate had already certified at top-1
`0.99677`. Two control scopes with known full-gate verdicts were screened for
exactly this purpose, and they also calibrated the correction: screens read low
because they run the shortest prompts, by a factor of `1.085` at 28-47
(`8.606e-5` against `9.342e-5`) and `1.181` at 32-47 (`4.916e-5` against
`5.808e-5`). A screen must clear the limit with that correction applied.

### Series

| Scope | Screen mean KL | x1.15 | vs `1e-3` | Screen p95 | Top-1 misses | 95% CI | Predicted saving |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: |
| 8-47 | 1.356e-3 | 1.560e-3 | fail | 5.976e-3 | 4/132 | 0.9247-0.9882 | 4.955 s |
| 12-47 | 9.591e-4 | 1.103e-3 | fail | 4.087e-3 | 4/132 | 0.9247-0.9882 | 4.688 s |
| 16-47 | 6.728e-4 | 7.738e-4 | pass 1.29x | 2.305e-3 | 3/132 | 0.9353-0.9922 | 4.421 s |
| 20-47 | 4.986e-4 | 5.734e-4 | pass 1.74x | 1.787e-3 | 3/132 | 0.9353-0.9922 | 4.153 s |
| 24-47 | 2.869e-4 | 3.299e-4 | pass 3.03x | 1.405e-3 | 2/132 | 0.9464-0.9958 | 3.886 s |
| 28-47 *(control)* | 8.606e-5 | — | certified | 4.128e-4 | 0/132 | — | 3.618 s |
| 32-47 *(control)* | 4.916e-5 | — | certified | 2.557e-4 | 1/132 | — | 2.591 s |

Mean KL is monotone across all seven scopes, from `5.81e-5` at 32-47 to
`1.356e-3` at 8-47, so the screens order the layer axis reliably.

**Correction, after the 20-47 full arm.** An earlier revision of this section
read the screens through a single `1.15x` correction and concluded that the
mean-KL boundary sits between layers 12 and 16, and therefore that layers 0-7
were out of contiguous range. That was wrong, because the screen-to-full ratio
is not a constant:

| Scope | Screen mean KL | Full mean KL | Full / screen |
| --- | ---: | ---: | ---: |
| 32-47 | 4.916e-5 | 5.808e-5 | 1.181 |
| 28-47 | 8.606e-5 | 9.342e-5 | 1.086 |
| 20-47 | 4.986e-4 | **2.527e-4** | **0.507** |
| 16-47 | 6.728e-4 | **3.796e-4** | **0.564** |

At depth the screen *overestimates* by about 2x rather than underestimating by
15%. Both deep control points land near a half (`0.507` and `0.564`), so the low
end of the interval now rests on two independent scopes rather than one. Reading
the 20-47 screen through the shallow factor predicted `5.7e-4` against an actual
`2.5e-4`. Screens are therefore treated as an interval over the observed ratio
range `0.50`-`1.20`, and decide only when that whole interval falls on one side
of the limit. Under that treatment 16-47 and 24-47 screen as passes, while
**8-47 and 12-47 are inconclusive rather than failing** — their intervals
straddle the `1e-3` mean limit. Layers 0-7 are not ruled out by any evidence now
in hand; they are unmeasured. The normative rule is
`docs/EXECUTION-PROFILES.md` 6.4.

Top-1 is the other question the screens cannot answer. The miss counts trend the
right way — 0 and 1 at the certified scopes, 2 at 24-47, 3 at 20-47 and 16-47,
4 below — but every screen interval spans 0.99, so none of them separates a
passing scope from a failing one. Only a 1548-row arm can.


## Open: layers 12-47 and 8-47 are unmeasured, not excluded

Layers **16-47** is the deepest **certified** scope, measured 2026-09-17 at
`artifact-layers16-47-4cat.json`. Two scopes below it remain open and neither
has been ruled out:

| Scope | Screen mean KL | Full-arm range (0.50-1.20x) | Screen verdict | Predicted saving |
| --- | ---: | --- | --- | ---: |
| 12-47 | 9.591e-4 | 4.80e-4 - 1.15e-3 | inconclusive | ~4.69 s |
| 8-47 | 1.356e-3 | 6.78e-4 - 1.63e-3 | inconclusive | ~4.96 s |

Both are screen-**inconclusive**, not failing: their intervals straddle the
`1e-3` mean limit, so neither can be decided by a screen and neither is excluded.
A 1548-row arm is the only way to place either one.

The 16-47 full arm was attempted twice on 2026-09-17 and both attempts were
killed before writing an artifact. The cause is host memory, not the route: the
`UD-Q4_K_XL` weights are 111 GB against 125 GB of system memory, so a full-model
gate maps nearly the whole machine into page cache. No process was holding
memory at the time — used was 16 GB with 108 GB available, while `buff/cache`
climbed from 21 GB to 41 GB as the model paged in. The earlier arms in this
campaign ran on the same margin and succeeded; that was luck rather than
headroom. The third attempt succeeded after the host was quieted and the run was
launched directly rather than under a separate polling shell.

Remaining contiguous upside is about **+0.5 s across two arms**, against the
**+4.15 s** predicted at 20-47 and the **+4.42 s** predicted at 16-47, out of
the +7.04 s the maximal scope offers. Both remaining arms are screen-inconclusive
and each costs about twenty minutes on a host that has no headroom for a
concurrent job, so the honest comparison is +0.5 s of predicted gain against two
full-model runs.
