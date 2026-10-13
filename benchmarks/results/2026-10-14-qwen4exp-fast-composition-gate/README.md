# Qwen4Exp `fast` composition: full-vocabulary teacher-forced gate

Date: 2026-10-14 UTC. Host: AMD Radeon 8060S / gfx1151, machine
`55ea6c509d0b49eea8de7094a1023668`, ROCm 10.0.0 (TheRock). Model: unsloth
`Qwen3.8-Flash-Next` `UD-Q4_K_XL`. Tree at `98aaf1a8f`, tracked-clean.

The `fast_composition` candidate of `scripts/qwen4exp_layer2_profile_gate.py`
applies every route the opt-in `fast` profile selected at that commit to the
`production` plan and scores it against the `strict` teacher: wide-tile
routed F16 GEMMs for every grouped routed-expert dispatch, the F16 dense wide
route at layers 0-47, the hyper-connection up projection on the wide route,
the small-matrix F16 GEMM for the indexer projections, and the F16-operand
router-logits producer. The WMMA attention kernel added afterwards is not
part of this run.

Protocol: 18 prompts (mtp-bench 10 + GDN heldouts 8), prefill chunk 4096,
24 teacher-forced decode steps, full-vocabulary logits, 450 rows, 3
candidate repeats, 2 free-running 32-token trajectories. Artifact:
[`gate.json`](gate.json).

## Result against the calibrated production envelope

| Metric over 450 rows | Measured | Production limit | |
| --- | ---: | ---: | --- |
| Mean KL | 7.121e-04 | 1e-3 | pass |
| p95 KL | 3.387e-03 | 5e-3 | pass |
| p99 KL | 8.295e-03 | 2e-2 | pass |
| Max KL | 3.444e-02 | 5e-2 | pass; one row over the 2e-2 review boundary |
| Top-1 agreement | 99.11% (446/450) | 99% | pass |
| Top-5 overlap | 0.9827 | | |

All four top-1 mismatches are on flip-eligible rows (strict margin below the
gate's flip threshold); none is outside. The one row over the review
boundary (`heldout_general_ja_speculative`, step 12, KL 0.0344) keeps its
top-1 token and full top-5 overlap. Three candidate repeats are
bit-identical; the state and lifecycle gates pass.

| Scope | Rows | Mean KL | Max KL | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| category = code | 150 | 2.202e-04 | 7.515e-03 | 99.33% |
| category = general_en | 100 | 6.266e-04 | 5.421e-03 | 99.00% |
| category = general_ja | 100 | 1.253e-03 | 3.444e-02 | 99.00% |
| category = mixed_ja_en | 100 | 9.940e-04 | 1.024e-02 | 99.00% |
| shape = c1 | 432 | 6.829e-04 | 3.444e-02 | 99.07% |
| shape = prefill_last | 18 | 1.412e-03 | 1.024e-02 | 100.00% |
| transition = prefill_to_c1 | 18 | 1.412e-03 | 1.024e-02 | 100.00% |
| transition = steady | 432 | 6.829e-04 | 3.444e-02 | 99.07% |

Three scopes exceed the per-scope mean-KL limit of 1e-3: category
`general_ja` (1.25e-3), shape `prefill_last` and transition `prefill_to_c1`
(the same 18 rows, 1.41e-3, top-1 18/18). Every scope passes its top-1 floor
of 97%. The task axis is 13 of 18 strict-exact free-running trajectories
(`production`: 15 of 18).

## Reading

- The `fast` contract's floor (KL <= 0.05 and top-1 >= 90% against the
  strict teacher, exact control, determinism) is met with wide margin.
- Against the **production** envelope the composition passes every global
  threshold and fails on scope means only: `prefill_last` at 1.41x its
  limit and `general_ja` at 1.25x. These are the scopes that also rejected
  the earlier compensated-T16 candidates at 1.53x and 1.69x. The gate's
  automatic decision is therefore "reject" for production, and `fast`
  remains a separate opt-in profile.
- The artifact records `measurement_valid: false` because the candidate spec
  declares no dispatch-shape argument positions for the routed kernel; the
  dispatch counter itself recorded 4,230 routed gate/up launches, so the
  candidate ran. The numerical rows are unaffected. A shape-qualified rerun
  is needed before this artifact is used for a production admission.

## Reproducing

```bash
# Environment: TheRock ROCm clean-process wrapper (docs/ENVS.md); tracked tree must be clean.
MODEL=/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
python scripts/qwen4exp_layer2_profile_gate.py --candidate fast_composition \
  --model-root "$MODEL" --prefill-chunk-size 4096 \
  --output benchmarks/results/2026-10-14-qwen4exp-fast-composition-gate/gate.json
```
