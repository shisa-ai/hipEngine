# Qwen4Exp `fast` profile: full-vocabulary teacher-forced gate on the complete composition

Date: 2026-10-13 UTC. Host: AMD Radeon 8060S / gfx1151, machine
`55ea6c509d0b49eea8de7094a1023668`, ROCm 10.0.0 (TheRock). Model: unsloth
`Qwen3.8-Flash-Next` `UD-Q4_K_XL`. Tree at `1213ddd7a`, tracked-clean.

The `fast_composition` candidate of `scripts/qwen4exp_layer2_profile_gate.py`
applies every route the opt-in `fast` profile binds to the `production` plan
and scores it against the `strict` teacher: wide-tile routed F16 GEMMs, the
F16 dense wide route at layers 0-47, the hyper-connection up projection on
the wide route, the small-matrix F16 GEMM for the indexer projections, the
F16-operand router, the WMMA prefill attention, and the tiled GDN recurrence
at every GDN layer. This composition measures 769.8 / 935.1 / 1104.8 tok/s
prefill at p512/p1024/p4096 on the canonical protocol, against 306.6 / 331.4
/ 337.2 for `production`
([rates](../2026-10-14-qwen4exp-gdn-norm-gate-waves/README.md)).

Protocol: 18 prompts (mtp-bench 10 + GDN heldouts 8), prefill chunk 4096,
24 teacher-forced decode steps, full-vocabulary logits, 450 rows, 3
candidate repeats, 2 free-running 32-token trajectories. Artifact:
[`gate.json`](gate.json).

## Result against the calibrated production envelope

| Metric over 450 rows | Measured | Production limit | |
| --- | ---: | ---: | --- |
| Mean KL | 6.493e-04 | 1e-3 | pass |
| p95 KL | 3.452e-03 | 5e-3 | pass |
| p99 KL | 7.889e-03 | 2e-2 | pass |
| Max KL | 1.186e-02 | 5e-2 | pass; no row over the 2e-2 review boundary |
| Top-1 agreement | 99.56% (448/450) | 99% | pass |
| Top-5 overlap | 0.9791 | | |
| Free-running strict-exact tasks | 15 / 18 | | same count as `production` |

Both top-1 mismatches are on flip-eligible rows. Three candidate repeats are
bit-identical; the state and lifecycle gates pass.

| Scope | Rows | Mean KL | Max KL | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| category = code | 150 | 2.627e-04 | 5.117e-03 | 99.33% |
| category = general_en | 100 | 8.082e-04 | 8.868e-03 | 100.00% |
| category = general_ja | 100 | 6.115e-04 | 1.186e-02 | 99.00% |
| category = mixed_ja_en | 100 | 1.108e-03 | 1.013e-02 | 100.00% |
| shape = c1 | 432 | 6.261e-04 | 1.186e-02 | 99.54% |
| shape = prefill_last | 18 | 1.206e-03 | 8.331e-03 | 100.00% |
| transition = prefill_to_c1 | 18 | 1.206e-03 | 8.331e-03 | 100.00% |
| transition = steady | 432 | 6.261e-04 | 1.186e-02 | 99.54% |

Two scope groups exceed the per-scope mean-KL limit of 1e-3: shape
`prefill_last` / transition `prefill_to_c1` (the same 18 rows, 1.21e-3,
top-1 18/18) and category `mixed_ja_en` (1.11e-3, top-1 100/100). Every scope
passes its top-1 floor.

## Reading

- The `fast` contract's floor (KL <= 0.05, top-1 >= 90%, exact control,
  determinism) is met with wide margin.
- Against the **production** envelope the composition passes every global
  threshold and every top-1 threshold, and matches `production` on the task
  axis. It fails on two scope means, at 1.21x and 1.11x of the 1e-3 limit.
  The gate's automatic decision is therefore "reject" for production, and
  `fast` remains opt-in.
- Compared with the earlier gate of the composition before attention and
  GDN ([`../2026-10-14-qwen4exp-fast-composition-gate/`](../2026-10-14-qwen4exp-fast-composition-gate/README.md)):
  mean KL 7.12e-4 to 6.49e-04, max 3.44e-2 to 1.19e-02, top-1 99.11% to
  99.56%, `prefill_last` mean 1.41e-3 to 1.21e-3. These are two runs
  of different compositions, not a controlled comparison of one change.
- Limits of this evidence: the battery's prompts are short, so the attention
  kernel is exercised in its dense window, not its masked sparse window; and
  the artifact records `measurement_valid: false` because the candidate spec
  declares no dispatch-shape positions for the routed kernel (the dispatch
  counter recorded 4,230 routed gate/up launches). A long-context gate case
  and a shape-qualified rerun are owed before any production admission.

## Reproducing

```bash
# Environment: TheRock ROCm clean-process wrapper (docs/ENVS.md); tracked tree must be clean.
MODEL=/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
python scripts/qwen4exp_layer2_profile_gate.py --candidate fast_composition \
  --model-root "$MODEL" --prefill-chunk-size 4096 \
  --output benchmarks/results/2026-10-14-qwen4exp-fast-composition-gate-2/gate.json
```
