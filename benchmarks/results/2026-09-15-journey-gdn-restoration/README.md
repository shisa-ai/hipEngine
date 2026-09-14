# GDN Restoration On Corrected Production

Source `f0b521468`, Framework machine `55ea6c509d0b49eea8de7094a1023668`,
Radeon 8060S/gfx1151, Flash-Next UD-Q4_K_XL and BF16 KV, chunk 1024.
Only the existing GDN/DPP suffix is restored over guarded Q8 and QSA.
No default changes or performance claim yet.

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

Commands, host/compiler identity, model fingerprint, overrides, full scoped
metrics and raw hashes are in `artifact.json`. Reproduce the compact packet:

```bash
.venv/bin/python benchmarks/results/2026-09-15-journey-gdn-restoration/assemble.py \
  --raw-root /tmp/hipengine-journey-execute-20260914
```

No GPU trace rerun was needed for this unchanged kernel; the DPP fixture,
ISA and trace evidence is in the prior journey GDN packet. New counted
model dispatches establish engagement of the restored composition.
