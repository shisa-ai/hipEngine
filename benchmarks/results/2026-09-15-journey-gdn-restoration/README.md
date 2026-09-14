# GDN Restoration On Corrected Production

Source `f0b521468`, Framework machine `55ea6c509d0b49eea8de7094a1023668`,
Radeon 8060S/gfx1151, Flash-Next UD-Q4_K_XL and BF16 KV, chunk 1024.
Only the existing GDN/DPP suffix is restored over guarded Q8 and QSA.
No default changes or performance claim. The measured composition fails the
predeclared task criterion despite passing numerics.

| Gate | Rows / top-1 | Mean KL | p95 KL | Maximum KL | Counted suffix calls |
| --- | --- | ---: | ---: | ---: | ---: |
| Natural categories and heldouts, 32 decode steps | 593/594 | 0.00006434 | 0.0002295 | 0.003783 | 1350 |
| Canonical 512/1K/4K, 64 decode steps | 777/780 | 0.00009336 | 0.0004115 | 0.010320 | 1080 |

All numerical scopes pass, with three deterministic repeats, finite state,
matching ownership metadata and zero tracked allocations after teardown.
These results do not establish universal equality. The short numerical
metrics reproduce the prior strict-backed isolated GDN result.

Seventeen of eighteen short free trajectories match strict. The Japanese
speculative-decoding explanation differs in wording, and both captured
prefixes are incomplete. It requires complete paired task review, not
automatic rejection for differing token IDs and not automatic acceptance.
Canonical depth passes independently of that unresolved task decision.

## Complete Task Review

The targeted Japanese explanation reaches EOS at 1187 strict tokens and
1121 candidate tokens, with two identical repeats per arm and zero teardown.
Runtime source is unchanged between numerical and task capture revisions.

The candidate newly attributes limited verification parallelism to
causal-mask sequential dependence. Known draft tokens allow parallel
target evaluation of their prefixes; this is a material error in the
requested explanation of verification costs. Strict does not make that
claim. The primary PMLR speculative-decoding paper and the exact review
scope are recorded in `task-review.json`; the compact packet preserves
both complete texts.

**The composition is not promoted under the existing no-new-material-error
per-prompt rule.** This is one observation, not evidence of lower expected
quality or a blanket rejection of GDN. Strict also contains simplifications.
No new criterion was added, and numerical thresholds were not widened.
After this binding task failure, the remaining task prompts and performance
comparison were not run. Multi-column GDN is a separate candidate.

Commands, host/compiler identity, model fingerprint, overrides, full scoped
metrics and raw hashes are in `artifact.json`. Reproduce the compact packet:

```bash
.venv/bin/python benchmarks/results/2026-09-15-journey-gdn-restoration/assemble.py \
  --raw-root /tmp/hipengine-journey-execute-20260914
```

No GPU trace rerun was needed for this unchanged kernel; the DPP fixture,
ISA and trace evidence is in the prior journey GDN packet. New counted
model dispatches establish engagement of the restored composition.
