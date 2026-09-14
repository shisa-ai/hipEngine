# Current-Profile Chunk Admission

Framework machine `55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S/gfx1151,
Flash-Next UD-Q4_K_XL/BF16, c1/context 4352. Source `9ce67327a`.
Chunk 1024 remains the default. Allocation and bounded numerical results
are recorded below; no performance or native-context inference claim.

Both probes allocate worst-case grouped repair queues for the GDN and QSA
scratch owners, not just constructor memory. Queue width covers the largest
gate/up/down output and every compact row. Existing scratch and reserve
allowances are unchanged.

| Chunk | Tracked GB | Scratch GB | Scratch margin GB | Verdict |
| --- | ---: | ---: | ---: | --- |
| 2048 | 86.055 | 3.229 | +1.066 | Prepared allocation passes |
| 4096 | 89.322 | 6.455 | -2.160 | Scratch accounting fails |

GB are decimal. Both close with zero tracked allocations. The 4096 failure
is not device OOM: allocation succeeds, but exceeds the modeled scratch
allowance. Correct the accounting from actual allocation structure before
using that shape; do not spend the reserve to hide it.

The earlier 2048 constructor-only pass is preserved as diagnostic evidence.
These current-profile results differ from the pre-recovery probes because
the resource configuration changed. They do not retroactively invalidate
those measurements.

## Bounded Numerical Gate

Clean `676e3b6bc`, same model/host/profile, 12 canonical cases at512/1K/4K,
64 teacher-forced decode steps and three candidate repeats:
**780/780 top-1, zero KL and zero maximum logit delta**. All sampled recurrent
state hashes match strict, repeat determinism and ownership metadata pass,
and tracked allocations close to zero.

Observed calls confirm strict chunk1024 versus candidate chunk2048:
four1024-token reference calls versus two2048-token candidate calls at4K.
All48 case/arm/repeat chunk records are checked. The strict reference's chunk
size did not move with the candidate.

This passes the production numerical envelope, without changing its limits.
Performance, active-shape task/isolation and wider admission remain before
default promotion. Native-context, c2, hidden-seed export, graph-capture and
driver-owned scratch claims are outside this packet. State summaries do not
copy the complete append-only KV payload.

```bash
.venv/bin/python benchmarks/results/2026-09-14-journey-chunk-admission/assemble.py \
  --raw-root /tmp/hipengine-journey-execute-20260914
```
