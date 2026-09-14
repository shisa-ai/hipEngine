# Q8 Prefill Numerical Diagnosis

Hardware: Framework Desktop `gfx1151`, Radeon 8060S, physical machine
`55ea6c509d0b49eea8de7094a1023668`. Target: Flash-Next UD-Q4_K_XL,
unchanged four-shard GGUF and BF16 KV. Commands, model fingerprint,
compiler, overrides and provenance are in the companion artifact.

## Findings

The previously failing production envelope is not explained by GDN alone.
Reverse-isolated GDN passed its numerical gate in the preceding experiment.
This experiment locates the major tail contribution in prefill arithmetic
and identifies two Q8 paths whose combined fallback passes the automatic
short-prompt numerical envelope.

| 50-row diagnostic | Max KL |
| --- | ---: |
| Production prefill and decode | 0.054642454 |
| Strict prefill, production decode | 0.005908949 |
| Production prefill, strict decode | 0.050698936 |
| All strict | 0 |
| Production matrix family over strict other arithmetic | 0.053097584 |

Each hybrid reconstructs its prefix in one runner, then switches flags after
draining/clearing graphs. No incomplete snapshot is transferred between
runners. All arms use the same strict teacher token chain.

The largest production prefill rows are `heldout_mixed_review`
(KL0.010250340) and `heldout_mixed_summary` (0.006508525). These are
different from the Japanese decode-step12 outlier. All18 prefill row
summaries are now exported, even below the normal0.02 review threshold.

Q8 selected-down WMMA contributes strongly to the known decode tail.
Turning it off alone leaves a mixed-language prefill failure. Dense Q8 MMQ
alone reproduces the separate dense-family drift on the two prompts that
reach its64-row admission threshold. The iu8/GR alternatives are inactive
at these short rows, not proved exact at their larger admitted shapes.

## Real Q8-Down Inputs

Read-only instrumentation of90 real calls reproduces the production
50-row metrics exactly. Across28,896,000 BF16 activations:

- Maximum absolute activation124.5; no FP16 conversion overflow/nonfinite.
- 8,666 conversions change value;73 nonzero values become zero.
- A fixed, geometry-based sample contains23,040 outputs.
- FP64 product MSE from activation conversion:1.85e-20.
- FP64 product MSE from weight conversion:3.21e-10.
- FP16 operand rounding changes2,332 sampled ideal BF16 outputs.
- The kernel differs from the FP16-operand ideal on14 sampled outputs.

This sample supports FP16 weight rounding as the dominant leaf difference,
not activation overflow or an indexing error. It is not an all-element
proof or a long-context claim. Small leaf errors can be amplified by later
layers and routing; these measurements are not an additive decomposition
of final-logit KL. The earlier weight-residual candidate remains rejected
under its full gate, despite better leaf MSE.

## Bisection Caveat

The first custom shape sequence is INVALID for shape attribution: its
intervention was absent from the host linear-dispatch cache key, so later
arms reused earlier selections. Missing selection counts exposed this.
The corrected run clears both graph and host-dispatch caches per arm and
records every intended shape intervention. Only that rerun is used.

Single-layer Q8-down interventions show stronger effects in layers2/4 than
46/47. No single-layer production restriction is qualified. Corrected MMQ
shape interventions are non-monotonic; excluding QSA value alone worsens
the prefill error. No prompt-conditioned branch, threshold relaxation,
or row-limit adjustment to exclude the measured prompts is introduced.

## Two-Q8 Fallback

Candidate overrides, with every other production flag unchanged:

```text
HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN=0
HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL=0
```

Clean revision `c20b72f2f`,18 natural suite/heldout prompts,32 teacher-forced
decode steps,594 scored rows,three numerical repeats:

| Metric | Candidate |
| --- | ---: |
| Mean KL | 0.000372133 |
| p95 KL | 0.001856306 |
| p99 KL | 0.004416409 |
| Max KL | 0.026879392 |
| Top-1 | 589/594 (99.158%) |
| Category/shape/transition failures | 0 |

Automatic numerical limits, determinism, state and teardown checks pass.
The English step25 row above0.02 retains top1/top5 and has strict
margin2.254; it requires review rather than automatic admission.
Four32-token free trajectories differ and require complete task review.
This does not by itself qualify a default repair or long-prefill behavior.

## Remaining Gates

The two-Q8 fallback was not promoted. Complete-output capture reached all18
strict prompts and six candidate prompts before the candidate's English
bandwidth explanation introduced an incorrect model-specific GQA example.
That violates the predeclared per-prompt factual non-inferiority criterion.
`task-review.json` records the source, hashes, failure and incomplete-capture
status. This is an observed task regression, not rejection for ID inequality.

## Conservative Recovery

The UD-Q4_K_XL production binder now disables the unqualified arithmetic
composition while retaining established exact grouped/risk-repair owners,
register-state GDN/wave reductions and mapping-only random PLE advice.
Approximate variants remain registered for explicit experiments. The
Q4_K_M profile, model files, KV ABI and other backends are unchanged.
The manifest selects the recovery owners with stable strict-compatible scopes.

Clean candidate qualification at `0ff037fcc`:

- Full natural suite:594/594 logits/top1 match strict, three repeats,
  state/lifecycle pass,18/18 short32-token free trajectories match strict.
- Canonical depth:780/780 logits/top1 match strict across four categories
  at512/1024/4096,64 decode transitions, three repeats; state/teardown pass.
- This is observed exactness on the tested scope, not a new universal
  bit-identity promotion rule or proof for every possible request.

The longer EOS task capture belongs to the rejected two-Q8 candidate, not
this conservative candidate. Do not claim a new complete-EOS task suite for
the recovery from those outputs. The recovery uses the qualified strict
arithmetic chains and passes the existing short-output task gate.

## Recovery Cost

Same Framework host, unchanged model/BF16 KV/chunk1024, three counterbalanced
pairs in one residency,128 decode transitions,72 measured samples and24
warmups. Each arm owns separate warmed graph caches. The old configuration
has failed quality and is only a throughput diagnostic.

| Shape | Previous PP / TG | Recovery PP / TG | PP change | TG change |
| --- | ---: | ---: | ---: | ---: |
| 512 | 292.70 / 20.17 | 172.87 / 17.63 | -40.94% | -12.60% |
| 1024 | 309.27 / 19.43 | 179.72 / 17.01 | -41.89% | -12.43% |
| 4096 | 285.79 / 18.06 | 129.95 / 11.40 | -54.53% | -36.88% |

Within-arm outputs repeat; cross-arm differences are expected and reported.
The original run completed all samples and teardown, then its shared summary
helper rejected cross-arm ID differences. The fixed summary explicitly allows
cross-arm drift while continuing to reject within-arm nondeterminism. It is
reassembled from unchanged raw samples; no timing rerun or sample exclusion.

Profile source edits occurred after the A/B process had loaded/bound its
original configurations. The measured process used its original revision
and explicit arm overrides; no kernel source changed during the run.
CPU-focused tests were also run during portions of the measurement; these
rates are a diagnostic recovery-cost comparison, not a precision tuning
claim. All samples are retained, including visible decode-rate variation.

This is a conservative correctness recovery, not the minimal precision
repair. Restore individual arithmetic families only after a full
same-suite numerical/task/depth qualification. Tiled GDN is currently
inactive, so DPP's earlier exact parent-relative speedup is not active in
this profile. MTP/multimodal throughput is not requalified by this text-AR work.
