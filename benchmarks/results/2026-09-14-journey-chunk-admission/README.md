# Current-Profile Chunk Admission

Framework machine `55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S/gfx1151,
Flash-Next UD-Q4_K_XL/BF16. The initial bounded allocation probes use
c1/context 4352 at `9ce67327a`; later scopes and pins are stated below.
Chunk 1024 remains the default. Allocation and bounded numerical results
are recorded below, followed by the bounded performance comparison.
No native-context inference or new-default claim.

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
Active-shape task/isolation and wider admission remain before
default promotion. Native-context/c2 inference, hidden-seed export, graph-capture and
driver-owned scratch claims are outside this packet. State summaries do not
copy the complete append-only KV payload.

## Performance Harness Check

At clean `8ad5fae51`, native production chunk1024 and the active runner
borrowing those same prefill buffers match all five scored4K logit rows
and sampled state, with zero teardown allocations. Recorded metadata/token
capacities are1024 versus2048, and map capacities1120 versus1760.
The timed comparison can therefore use correctly sized prefill workspaces
while keeping active decode graphs, recurrent state and KV ownership shared.
Maximum-size PLE staging and unused donor non-prefill allocations remain
common to both arms; this is not a cold-start allocation-cost comparison.

## Measured Performance

Clean `958f2d2d0`, same physical host/model/BF16 configuration, three pairs
per case, 72 measured samples and 24 warmups, 128 decode transitions.
No tests, builds or other model workloads overlapped timing.

| Chunk | 512 PP / TG | 1K PP / TG | 4K PP / TG |
| --- | ---: | ---: | ---: |
| 1024 | 185.008 / 17.504 | 190.334 / 16.731 | 182.802 / 10.075 |
| 2048 | 184.374 / 17.505 | 189.997 / 16.732 | 187.042 / 10.198 |
| PP delta | -0.34% | -0.18% | +2.32% |
| TG delta | +0.005% | +0.005% | +1.22% |

All 4K complete requests improve1.52-2.14%. Short complete-request changes
range from0.40% lower throughput to0.25% higher. These small costs are not
discarded or relabeled as wins. All72 samples match IDs/final logits/sampled
state across arms and repeats, teardown is zero, and the donor executes no
graphs or eager graph-cache calls during timing.

This is a measured workload tradeoff, not a universal speedup. The active
decode owners/graphs are shared; observed TG movement is not attributed
to a changed decode kernel or an independently measured frequency mechanism.
Chunk1024 remains default while remaining admission checks are completed.

## Native c2 Allocation

Clean `350cf1abc`, chunk2048, two prepared runners at262144 positions:
tracked allocation107,332,608,432 bytes, unused modeled scratch1,017,794,480
bytes, unchanged4,294,967,296-byte reserve and zero teardown allocations.
All four worst-case repair queues are prepared and reconcile to the tracked
allocation increase.

This is an allocation-only result, not262K generation, retrieval or c2
inference qualification. Boundary/isolation and active long-task checks remain.

```bash
.venv/bin/python benchmarks/results/2026-09-14-journey-chunk-admission/assemble.py \
  --raw-root /tmp/hipengine-journey-execute-20260914
```
