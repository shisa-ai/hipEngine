# Bounded HC-down grouped-norm F16 publication

On Radeon 8060S / gfx1151 host `55ea6c509d0b49eea8de7094a1023668`, this
Qwen3.8-Flash-Next UD-Q4_K_XL diagnostic compares parent grouped norm →
F32-to-F16 cast → padded HC-down GEMM with dual-publication grouped norm →
the same GEMM. Both arms charge the complete operation through HIP events.
The producer preserves the existing F32 output and publishes the identical
staging F16 bits. The operation screen and native production qualification
are recorded separately below.

## Natural-input result

Each case replays 70 HC-down calls from layers 13–47, covering the attention
and FFN projections at K10240/N320, branches4/H2560. Each call has one warm
pair and two measured pairs, with both arm orders represented. The table sums
each call's mean measured duration; these are operation sub-windows, not
prefill or request-wall rates.

| Natural input | Parent ms | Publisher ms | Speedup | Duration change |
| --- | ---: | ---: | ---: | ---: |
| code p512 | 57.277 | 54.074 | 1.059x | −5.591% |
| code p1024 | 93.045 | 99.133 | 0.939x | +6.542% |
| code p4096 | 359.037 | 327.398 | 1.097x | −8.812% |
| English p4096 | 358.844 | 327.377 | 1.096x | −8.769% |
| Japanese p4096 | 360.867 | 326.934 | 1.104x | −9.403% |
| mixed Japanese/English p4096 | 359.763 | 328.139 | 1.096x | −8.790% |

All 420 calls match F32 norm, F16 publication and GEMM output bytes and preserve
residual/norm-weight inputs. Teardown has zero tracked bytes and allocations.
All 70 calls improve their mean at 512 and each 4096 case; only 8/70 do so at
1024. Both measured-pair strata show the same direction. Synthetic
complete-operation medians agree: 1.053x/0.928x/1.090x at 512/1024/4096.
The 1024 loss is retained, not averaged away.

## Native production qualification

The gfx1151 UD-Q4_K_XL production profile selects publication only at exactly
4096 rows, K10240/N320, branches4/H2560, layers13–47 HC-down roles. Strict,
Q4_K_M, 512/1024 and unmeasured shapes keep parent norm+cast. The manifest names
`gr_grouped_rmsnorm` / `bf16_f32` / `f32_f16_publish`, with the registered strict
norm fallback; the complete norm+cast chain also stays registered as
`f32_f16_publish_unfused`. Set
`HIPENGINE_QWEN4_EXP_GR_NORM_F16_PUBLISH=0` before profile registration for a
manifested rollback, or after binding for diagnostic arm comparisons.

The native qualification at tracked-clean commit `a42aa0dd8` covers 14
canonical/heldout cases with three paired parent/candidate repeats. It records
2,520 candidate publisher/consumer calls (70 per candidate 4096 prefill), zero
decode calls, exact full logits, recurrent state, full KV and output IDs, and
zero active allocations/current bytes at teardown. Heldouts are ownership and
state fixtures, not task-quality scores. Evidence is in
[`native-state.json`](native-state.json).

The production-binder integration check also passes at code-p512/code-p4096:
its first candidate request uses the binder without a flag override. It records
zero publications at512 and70 publishers=accepted consumers with zero casts
at4096; parent4096 performs70 casts. Both cases match full logits/state/fullKV/
outputIDs, with no decode engagement and zero allocations/bytes after close.
This two-case check uses the uncommitted promotion declarations and is not a
new clean performance certificate. Production/strict manifest hashes are
`be32e7193f30c4c4a65f5cbaf99b62033fd6458edd295fc2d7806e734001469b` /
`1fcbf992271251220cc4707eb8d24d28506cd3714550756cf2e32f387d879670`.
See [`binder-smoke.json`](binder-smoke.json).

The tracked-clean 12-case engine A/B at commit `a42aa0dd8` records 72 exact
and deterministic trajectories. Weighted parent→candidate prefill is
294.109→294.011 tok/s at 512, 317.546→317.473 at 1024, and 322.618→323.115 at
4096 (candidate/parent ratios 0.9997x, 0.9998x and 1.0015x). The aggregate request wall is
0.9970x, and general-English p4096 is 2.77% slower; no all-request or
decode-speed claim is made. The runtime handoff is accepted only as a
bounded prefill owner transfer at exactly 4096 rows; 512/1024 remain on the
parent norm+cast path and no other shape is interpolated. See the compact
qualification artifact for command, provenance and hashes.

The mean 4096 prefill time is 12,696.123→12,676.608 ms (−19.515 ms). The
three paired aggregate prefill ratios are 1.002025x/0.999885x/1.002709x;
maximum within-arm prefill CV is 0.608%, so the +0.154% engine result is a
small signal within run variation, not an independently stable end-to-end
speedup. Maximum decode CV is 27.216%; general-English p4096 decode samples
range from 7.0 to 11.7 s while publication never engages in decode. That does
not establish the cause of its request regression. The acceptance basis is
the repeated 1.096–1.104x complete-operation win, exact model results and 70
eliminated casts with no added allocations, not the aggregate request ratio.
All 72 timings and their time order are in [`timing-samples.json`](timing-samples.json).

## Kernel checks

The 43-test focused bundle passes; a subsequent expanded gfx1151 registry
assertion passes all 16 unit nodes. Coverage includes independent CPU norm
values and the KL≤0.05/top-1≥90% outer floor, exact parent F32/existing-cast F16
bits, finite ties/signed-zero/overflow/underflow, IEEE-special groups, tails,
repeat writes, guards, input immutability, nondefault stream, dimension/address
bounds, overlap refusal and the registered strict norm+cast chain.

The first implementation failed F16 parity on halfway and signed-zero inputs
although F32 output matched. Explicit `v_cvt_f16_f32` from the published F32
register fixes those cases. The static native census finds parent/publisher
14/16 VGPR, 28/30 SGPR and 185/183 instructions; both private scratch0 and the
same 16-register allocation bucket. The cached profiler smoke launches each
norm six times at 4096 rows, with publisher median 1.914 ms and scratch0.
Profiler durations prove engagement, not an independent timing comparison.
Dynamic LDS is 1024 bytes; the trace's LDS field reports zero, so it is not
used as allocation evidence. Register-only occupancy is not runtime occupancy.

## Immediate-use ownership contract

The runtime borrows the existing session-owned F16 staging buffer; it does not
allocate or retain a per-activation cache:

1. Admit only an enabled, registered F16-input consumer with sufficient active
   workspace, the qualified model-plan scope and exactly 4096 rows at
   K10240/N320. Strict, missing registration, disabled staging and every shape
   miss keep parent norm and ordinary cast/fallback.
2. Publish F32 norm and F16 bytes on the consumer's stream. Bind a scoped,
   single-use loan naming exact F32 source pointer, element count, stream,
   runtime and active workspace identity/capacity. The source and destination
   ranges must be disjoint from all producer operands and consumer output.
3. Only the immediate matching F16-input launch may accept the loan. A mismatched
   launch cannot accept stale bytes; any ordinary staging write invalidates
   outstanding publication. Consumption is single-use even if pointers recur.
4. Context exit and exceptions invalidate the loan. Nested sessions/publications
   and independent contexts cannot restore a consumed/overwritten loan.
   The session owner keeps the buffer alive until same-stream work completes;
   no cross-stream wait is implied by publication.
5. Ownership RED/GREEN tests cover successful cast elision and exact GEMM input;
   source/count/stream/runtime/workspace/variant misses; alias and capacity
   rejection; single-use, intervening overwrite, nesting, exception cleanup and
   independent-context isolation; parent/strict fallbacks; and zero added
   allocations. Refused workspace aliases are shadowed during fallback so an
   ordinary staging cast cannot overwrite producer operands or consumer output.
   Native state/KV/full-logit qualification and cached engagement checks pass.

The cached native loan trace runs parent/candidate/candidate/parent on a
nondefault stream: two publisher and two parent norm launches, two staging
casts (parent only), and four padded GEMMs, all scratch0. Publisher durations
are 1,584,482/1,578,433 ns. This verifies engagement and plausible duration;
profiler timing is not an independent speed comparison. See the trace summary
in [`artifact.json`](artifact.json).

## Reproduction and provenance

`screen.json` retains exact commands, environment, host/model/fixture/manifest
identity, per-call samples and hashes, synthetic samples, resources and raw
artifact fingerprints. `artifact.json` and `native-state.json` record the
tracked-clean full-model qualifications at `a42aa0dd8`; the promoted manifest
adds a producer scope, without changing the earlier arithmetic scopes. Native
loan trace and binder-smoke artifacts remain under
`/tmp/qwen4exp-dp5-loan-native-20261006/`. Raw screen traces, code objects and
logs remain in `/tmp/qwen4exp-dp5-norm-publish-20261006/`; they are not
committed. The natural screen ran at base `6c1e6a818` with the candidate
kernel/wrapper uncommitted; its `tracked_clean=false` is retained. This is
diagnostic operation evidence, not a clean full-engine performance certificate.
