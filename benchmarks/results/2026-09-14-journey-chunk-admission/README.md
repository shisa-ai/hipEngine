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

GB are decimal. These are the original probe's margins, which include the
host-staging reservation in its non-scratch components. Both close with zero
tracked allocations. The original4096 failure
is not device OOM: allocation succeeds, but exceeds the modeled scratch
allowance. Correct the accounting from actual allocation structure before
using that shape; do not spend the reserve to hide it.

**September15 follow-up:** this accounting is now repaired and the native-c2
allocation passes without consuming reserve. The new device-only check does
not credit host staging against device scratch.
[Derived accounting and allocation evidence](../2026-09-15-chunk4096-accounting/README.md).

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
The default stays1024 because of the short-request performance tradeoff.
Native-context inference, hidden-seed export, graph-capture and
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
inference qualification. Bounded c2 checks are recorded below.

## Active 4K Tasks

Clean `bb043e38b`, same Framework/model/BF16 lane: all six long-task
fixtures pass at strict chunk1024 and production chunk2048, three repeats
per arm. Each generation reaches EOS, stays finite and produces the correct
whole option letter. All36 outputs repeat identically; cross-arm IDs and
sampled state hashes match.

Actual calls are recorded for every generation: strict uses four1024-token
chunks and candidate uses two2048-token chunks. Prompt contents and embedded
non-thinking chat-template hashes match across arms. Tracked allocations
close to zero. These are ordinary autoregressive requests despite the
fixture IDs' historical `mtp_` prefix; no MTP performance claim is made.

The first capture was stopped during strict because the legacy hand-built
chat wrapper emitted thinking and exhausted the output cap. It never ran
the candidate and is not a candidate failure. The repaired harness uses the
model's embedded renderer with thinking disabled, preserves all task facts
and adjusts only filler budget to keep4096 total tokens. The invalid capture
is documented in `../2026-09-15-chunk-active-task-template.json`.

This is supplementary task evidence, not a complete long-form factual-quality
certificate or full KV/isolation gate. Chunk1024 is still the default.

## Boundaries And c1 Reuse

Clean `51e3e9a2b`, same host/model/BF16 lane, capacity4352. Canonical code
and Japanese prefixes cover2047/2048/2049/2051/2052/4095/4097 tokens.
Strict1024 supplies64 teacher transitions for each case; production2048
repeats three times with unrelated257-token prefills between repeats.

All910 scored rows have zero KL, zero logit delta and100% top1.
Control metadata, finiteness and determinism pass. Recurrent state, complete
BF16 KV-buffer hashes and live raw/pooled QSA index hashes match strict at
both prefill and final decode boundaries. Main and intervening prefill calls
are traced; ownership closes to zero.

Full KV bytes are read explicitly; `snapshot()` is not treated as an
append-only KV copy. Only live index keys are compared, since inactive
index storage is not a semantic input. Temporary selection scores are
outside these payload hashes. Arithmetic identity is an observed result,
separate from the production numerical/control gates.

This closes c1 boundary/reuse evidence, not simultaneous c2 ownership,
cancellation or native-depth inference. The real serving pool accumulates
scheduler chunks before invoking model prefill; no GPU-prefill interleaving
claim is made. Bounded c2 pool isolation is recorded next.

## Two Live Resident Runners

Detailed inspection at `15ead1097` and deferred inspection at `2003660a9`
pass on the same host/model/profile, chunk2048, context capacity4352.
Code2052, Japanese4097 and mixed2049 requests are compared with isolated
execution of the same production arithmetic.

Three repeats cover delayed peer prefill, decode-order permutation, peer
cancellation after two outputs, admission rollback, partial-prefix
cancellation, reuse of the released peer and cancellation while another
request remains live. Each request exercises both physical owners.

The detailed arm compares84 checkpoints. Because those reads synchronize
the device, the deferred arm adds no diagnostic reads during interleaving:
it verifies six final A/C payloads afterward. Both preserve emitted tokens,
full logits, recurrent state, complete KV and live index payloads against
their isolated references, which also match across runs.

Observed state/logit/KV/index/repair-buffer ranges do not overlap across
owners. Both arms trace12 model prefills/28 chunks; three cancelled partial
scheduler prefixes never reach model prefill. Peak tracked memory is
89,586,110,480 bytes, returning to zero after close.

Normal compact prefill/decode calls stay enabled; diagnostics do not force
full-output or logit-bias routes. These are native pool work items, not an
HTTP/SSE load test or concurrent GPU-prefill schedule. Native-depth, MTP and
multimodal qualification are not implied.

## Default Decision

Chunk2048 is a qualified explicit choice for the measured long-prefill
workloads; the global default stays1024. The4K PP/TG gains are2.32%/1.22%,
but512/1K PP is0.34%/0.18% lower. Short complete-request throughput changes
range from-0.40% to+0.25%, while every4K request improves1.52-2.14%.
These short costs cannot be averaged away to call the whole policy
non-regressive.

The concrete automatic-selection blocker is workspace residency: a
shape-dependent policy must keep the smaller owner's short-request costs
while using2048 for long prefill. The existing explicit
`prefill_chunk_size=2048` factory/benchmark setting is preserved. Do not
repeat unchanged arithmetic or claim every shape benefits.

Chunk4096's separate accounting problem is now repaired with a derived
mandatory footprint and a native-c2 allocation check. Its inference
qualification remains separate from2048.

```bash
.venv/bin/python benchmarks/results/2026-09-14-journey-chunk-admission/assemble.py \
  --raw-root /tmp/hipengine-journey-execute-20260914
```
