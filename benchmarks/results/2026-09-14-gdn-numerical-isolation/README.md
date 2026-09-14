# GDN Numerical Isolation

This experiment separates GDN recurrence changes from quantized matrix
arithmetic. It does not change kernels, production defaults, or thresholds.

## Protocol

Framework Desktop `gfx1151`, Radeon 8060S, machine ID
`55ea6c509d0b49eea8de7094a1023668`; Flash-Next UD-Q4_K_XL, BF16 KV.
Model identity, host, compiler, exact commands and environment overrides are
recorded in `artifact.json`. The screen uses 18 natural-prompt prefill-last
rows plus 32 strict-teacher decode rows for the prior Japanese outlier.
The full run uses all 18 suite/heldout prompts, 32 decode steps, three
numerical repeats and two 32-token free-generation repeats: 594 scored rows.
Natural prompts are short; these are not 512/1024/4096-token quality gates.

Non-GDN flags are held at their strict values. The current GDN arm enables
the existing register-state prefix and admitted tiled suffix (layers 27..47);
it does not widen the suffix to earlier layers. The serial arm disables
peer/column-warp recurrence but keeps register-state/wave-norm optimizations.
The multi-column screen changes only the suffix variant.

## Screen

| Strict-backed arm | Mean KL | Max KL | Top-1 |
| --- | ---: | ---: | ---: |
| All strict | 0 | 0 | 50/50 |
| Serial-register GDN | 0 | 0 | 50/50 |
| Current tiled GDN | 0.000116956 | 0.001469807 | 49/50 |
| Experimental multi-column GDN | 0.000182469 | 0.005209519 | 50/50 |

The current tiled mismatch is Japanese decode step 19, KL0.000186472,
with the strict top-1 at candidate rank2 and identical top-5 membership.
This is not the incumbent's large-tail row at step12.
The screen is diagnostic only, not a promotion or determinism certificate.

## Clean Full Run

Revision `501d7d2db31223cccf23820a967eff6e71baee4f`, clean tracked tree:

| Metric | Current GDN over strict non-GDN arithmetic |
| --- | ---: |
| Rows | 594 |
| Mean KL | 0.0000643436 |
| p95 KL | 0.000229458 |
| p99 KL | 0.001228302 |
| Max KL | 0.003782627 |
| Top-1 agreement | 593/594 (99.83165%) |
| Category/shape/transition failures | 0 |

Numerical thresholds, three-repeat determinism, state metadata/layout,
finite-state checks and teardown pass. The numerical summary exactly
reproduces the initial uncommitted diagnostic. Measurement validity is true.
The process exits2 with `requires_task_review`, not a numerical failure.

Seventeen of18 free trajectories match strict IDs. The remaining Japanese
speculative-decoding explanation swaps its phrasing; both32-token prefixes
are incomplete. This is not evidence of a semantic regression, but it also
is not a completed paired task-quality/non-inferiority evaluation.
No automatic task pass or rejection is inferred from ID inequality.

## Interpretation

Disabling GDN in the previous production-composition screen did not eliminate
the large tail (max KL0.05633994). Conversely, the reverse-isolation screen
above does not reproduce that tail. GDN is not established as its sole cause.
Interactions with quantized matrix arithmetic remain possible; these
ablations are not an additive error decomposition.

The source shows changed reduction order and decay reassociation in the
tiled recurrence compared with serial strict arithmetic. No new indexing
or recurrence-formula defect was identified. The DPP substitution preserves
the tiled parent's reduction order and has separate exactness evidence.

The first full run was invalid for qualification because the harness was
uncommitted. Its raw hash and diagnostic summary are preserved separately;
it must not be represented as a valid clean-revision run.

No production repair or multi-column promotion is claimed here. A
free-generation difference requires task review, not automatic rejection
for failure to match strict token IDs.

## Reproduce

```bash
python3 benchmarks/results/2026-09-14-gdn-numerical-isolation/assemble.py \
  --raw-root /tmp/hipengine-journey-execute-20260914
```

The assembler requires a completed screen, a valid clean full measurement,
an exact strict control, and all594 scored rows. It preserves failed
qualification outcomes rather than converting them to a pass.
