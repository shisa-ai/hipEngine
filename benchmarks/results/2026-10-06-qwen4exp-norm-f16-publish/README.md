# HC-down grouped-norm F16 publication screen

On Radeon 8060S / gfx1151 host `55ea6c509d0b49eea8de7094a1023668`, this
Qwen3.8-Flash-Next UD-Q4_K_XL diagnostic compares parent grouped norm →
F32-to-F16 cast → padded HC-down GEMM with dual-publication grouped norm →
the same GEMM. Both arms charge the complete operation through HIP events.
The producer preserves the existing F32 output and publishes the identical
staging F16 bits. No runtime handoff, profile or default changes in this packet.

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

**Next integration scope: exactly 4096 rows**, the four-category-qualified
shape; 512 stays a separate follow-up and 1024 stays on parent norm+cast.
No interpolation to unmeasured rows is supported. Native model correctness,
engagement and counterbalanced engine A/B remain required before promotion.

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

## Proposed immediate-use ownership contract

The runtime integration will borrow the existing session-owned F16 staging
buffer, not allocate or retain a per-activation cache:

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
5. RED tests precede runtime implementation: successful cast elision and exact
   GEMM input; source/count/stream/runtime/workspace/variant misses; alias and
   capacity rejection; single-use, intervening overwrite, nesting, exception
   cleanup and independent-context isolation; parent/strict fallbacks; zero
   added allocations. Native state/KV/full-logit and cached engagement checks
   follow before any default selection.

## Reproduction and provenance

`screen.json` retains exact commands, environment, host/model/fixture/manifest
identity, per-call samples and hashes, synthetic samples, resources and raw
artifact fingerprints. Raw traces, code objects and logs remain in
`/tmp/qwen4exp-dp5-norm-publish-20261006/`; they are not committed. The natural
screen ran at base `6c1e6a818` with the candidate kernel/wrapper uncommitted;
its `tracked_clean=false` is retained. This is diagnostic owner evidence, not
a clean full-engine performance certificate.
