# Chunk4096 Scratch Accounting

Framework machine `55ea6c509d0b49eea8de7094a1023668`, Radeon8060S/gfx1151,
Flash-Next UD-Q4_K_XL revision `8bdc666649440e9bdc97e16f3f75782c98478ff5`,
BF16 KV. Accounting repair `ae4d8a7ca`; no default or arithmetic change.

## Cause And Repair

Admission previously reserved a fixed4GiB scratch allowance per runner.
The CPU census executes the real allocation recipes with virtual pointers.
Its device totals exactly reproduce the earlier GPU probes after adding
resident weight bytes:

| Chunk / Context | Runner Device Bytes | Mandatory Device Scratch |
| --- | ---: | ---: |
| 2048 / 4352 | 3,531,309,320 | 3,270,785,880 |
| 4096 / 4352 | 6,798,971,144 | 6,538,447,704 |

The old4K reported deficit of2,159,594,328 bytes credited83,886,080 bytes
of host-staging reservation against device allocations. Without that
credit, the device-scratch deficit was2,243,480,408 bytes.

The public generator now supplies chunk size to admission. Scratch is
`max(4GiB, mandatory footprint)` per runner, including context-scaled QSA
metadata and full-capacity repair queues. The separate4GiB reserve and
host-staging reservation are unchanged. Direct resolver callers without
chunk information retain the legacy floor-only behavior.

This follows the existing pure memory-planner pattern; allocator-census
tests guard the sizing equations. Optional MMQ, graph, verification and
transaction allocations are not covered by this mandatory-footprint claim.

## Native c2 Allocation

Clean `ae4d8a7ca`, chunk4096, two prepared runners at262144 capacity:

| Item | Bytes |
| --- | ---: |
| Tracked device allocations | 114,940,346,800 |
| Accounted scratch, both runners | 15,221,821,520 |
| Separately reserved host staging | 83,886,080 |
| Unchanged reserve | 4,294,967,296 |
| Total admission requirement | 119,319,200,176 |
| Device-scratch margin | 0 |

The CPU native-context census predicts the device total exactly.
All four worst-case queues are prepared; tracked allocations return to zero.
The probe completes in52.1 seconds. That duration is allocation evidence,
not inference throughput.

**The accounting blocker is repaired.** This does not qualify native-depth
generation or chunk4096 numerics, task quality, isolation or performance.
Those inference experiments can proceed using this allocation evidence.

## Canonical Numerical Gate

Clean `4f3a3547e`, same host/model/BF16 configuration, strict chunk1024
versus production chunk4096. All12 canonical512/1K/4K cases,64 shared-teacher
decode transitions and three candidate repeats complete.

All780 rows match strict logits exactly: zero KL, zero maximum logit
difference and100% top1. Every numerical scope, repeatability, sampled
recurrent-state/metadata check and zero-allocation teardown passes.
All48 chunk traces match the declared sizes; at4K the candidate executes
one4096-token call versus four1024-token reference calls.

This is a bounded numerical pass; the separate timing result is below.
Exact capture argv and manifests are in `artifact.json`.

## Boundary And Reuse Gate

Clean `5c5dc5dbe`, capacity4352, code and Japanese prefixes at
2047/2048/2049/2051/2052/4095/4097 tokens,64 teacher transitions and three
candidate repeats. All910 logits match strict exactly. Control metadata,
finiteness and deterministic repeats pass.

Complete BF16 KV buffers, live raw/pooled index keys and recurrent state
match strict at prefill and final decode endpoints. Intervening257-token
prefills do not change the repeated result. All declared chunk traces
match and tracked allocations return to zero.

This is c1 full-payload/reuse evidence; c2 is recorded separately below.

## Active Tasks

Clean `949447117`: all six4K active tasks return the correct whole option
letter and reach EOS at strict1024 and production4096, three repeats each.
Within-arm repeats, cross-arm output IDs and sampled state hashes match;
finiteness and tracked teardown pass. All18 candidate generations use
one4096-token chunk, versus four1024 calls for each strict generation.

The existing embedded non-thinking template and task criterion are unchanged.
This supplements the numerical and boundary gates; it is not a new full-EOS
factual-quality certificate.

## Two-Runner Lifecycle

Clean `1df91bdbc`: detailed and deferred-inspection c2 gates pass, both with
three repeats at capacity4352. They exercise both physical runners, delayed
peer prefill, decode-order permutation, peer cancellation, admission rollback,
partial-prefix cancellation and reuse of the released runner.

The detailed arm compares84 checkpoints. The deferred arm makes no diagnostic
reads during interleaving, then checks six final A/C states. Compared logits,
full KV, live index and recurrent state match isolated production references;
those references also match across arms. Token accounting and disjoint observed
owner ranges pass. Both trace12 model prefills/16 chunks; cancelled partial
prefixes execute no model prefill.

Peak tracked allocation is96,121,434,128 bytes, with zero after close.
Normal compact output calls stay enabled. This is native-pool evidence,
not HTTP/SSE load testing, native-depth generation or throughput measurement.

## Matched Throughput

Clean `b4ecdd98f`, same host/model/BF16 lane, three counterbalanced pairs
per canonical case:72 measured samples and24 warmups,128 decode transitions.
Correctly sized1024/4096 prefill workspaces share one active runner's
unchanged KV/recurrent owners and decode graphs. The donor's non-prefill
allocations and maximum-size PLE staging are common to both arms; the donor
executes no graphs during timing. No tests/builds overlapped the run.

| Chunk | 512 PP / TG | 1K PP / TG | 4K PP / TG |
| --- | ---: | ---: | ---: |
| 1024 | 185.245 / 17.501 | 190.573 / 16.745 | 183.116 / 10.091 |
| 4096 | 184.988 / 17.502 | 190.641 / 16.721 | 190.333 / 10.203 |
| PP delta | -0.14% | +0.04% | +3.94% |
| TG delta | +0.006% | -0.14% | +1.11% |

Weighted tok/s. Every4K complete-request case improves2.26-3.82%.
Short complete-request changes range from-0.31% to+0.32%; those costs are
not averaged away. All72 samples match in output IDs,final logits and
sampled state; finiteness and teardown pass. Measured disk reads and major
faults are zero. Shared graphs control graph-instance differences, not
frequency; no decode-kernel or clock mechanism is inferred from TG movement.

The pre-timing borrowed-workspace control matches five native1024 rows and
sampled state. Reserve checks pass before/after donor allocation.

**Qualified explicit option; default remains1024.** Chunk4096 provides the
measured4K gain but is not universally non-regressive at short workloads.
Shape-dependent workspace selection remains the automatic-selection task.
This is not a direct2048-versus4096 trial, so do not turn the independent
session rates into a claimed paired gain between those sizes.

## Reproduction

Use the same cached-build ROCm environment recorded in the adjacent
chunk-admission artifact. The GPU probe was serialized with
`flock -n /tmp/hipengine-gfx1151-benchmark.lock`.

```bash
.venv/bin/python scripts/qwen4exp_chunk_memory_probe.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipengine-journey-hipcc-version-20260913.txt \
  --chunk-size 4096 --capacity 2 \
  --output /tmp/hipengine-journey-execute-20260914/resume-chunk4096-native-c2-accounted.json

.venv/bin/python benchmarks/results/2026-09-15-chunk4096-accounting/assemble.py \
  --raw-root /tmp/hipengine-journey-execute-20260914
```

The artifact contains both census commands, historical probes, complete
new probe, source/host/model identities and raw hashes. Seven RED tests
preceded implementation; the37-test focused bundle passes.
