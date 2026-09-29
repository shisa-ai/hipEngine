# RT-1: prefill chunk 4096 is 4.94% faster than today's 1024 default

**Date:** 2026-09-29
**Host:** Radeon 8060S (gfx1151), machine-id `55ea6c509d0b49eea8de7094a1023668`,
kernel 7.1.6-1-cachyos.
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL, Q8_0, BF16 KV, production execution
profile, case `code-p4096` (4096 prompt tokens), cached build.
**Artifacts:** `chunk-{1024,2048,4096}.json` (1 rep), `rep3-chunk-{1024,4096}.json`
(3 reps).

## Result

Three interleaved-cold repetitions per arm, same process-per-arm protocol:

| chunk | samples (s) | median (s) | within-arm spread |
|---|---:|---:|---:|
| 1024 (today's default) | 15.642, 15.625, 15.613 | **15.625** | 29.3 ms |
| 4096 | 14.977, 14.854, 14.841 | **14.854** | 135.4 ms |

**−771.9 ms, −4.94%.** The separation between medians is **5.70x** the larger
within-arm spread, and the arms do not overlap (`min(1024) = 15.613 >
max(4096) = 14.977`). So the effect is established at this sample size rather
than read against zero.

A single-repetition screen also covered 2048: 15.665 / 14.957 / 14.761 s for
1024 / 2048 / 4096, i.e. −4.52% and −5.77%. **The 3-rep median supersedes the
−5.77% figure**; the single sample overstated the effect.

## The output is identical

`token_id = 248068` and `logits_sha256 = 7270a972da43ff4642e557b1` on **every**
arm at every repetition, with `fell_back_to_strict = False` and
`named_profile_intact = True`. Chunk size changes how the work is scheduled, not
what it computes — which is what makes it safe to consider at all.

## Cost

| chunk | peak allocation |
|---|---:|
| 1024 | 84,439,702,280 B |
| 2048 | 86,094,246,664 B |
| 4096 | 89,403,335,432 B |

Chunk 4096 costs **+4.96 GB peak (+5.9%)**. That is an admission trade, not a
free win, and it belongs in any decision.

## Protocol

Recovered from the recorded command in
`benchmarks/results/2026-09-28-qwen4exp-current-attribution/artifact.json`, so
all arms share the case, fixture, execution profile and compiler-version file:

```bash
qwen4exp_profile_gap.py --model-root <UD-Q4_K_XL> --mode prefill \
  --case-id code-p4096 --repetitions 3 --execution-profile production \
  --compiler-version-file /tmp/qwen4exp-ledger-20260928-current/hipcc-version.txt \
  --require-cached-build --prefill-chunk-size {1024,4096} --output <path>
```

`--require-cached-build` keeps `hipcc` out of the measured process. One process
per arm, under the benchmark lock.

## Not claimed, and what blocks promotion

- **Not promoted.** No default changed. Two concrete blockers:
  1. The ledger's own note requires re-running the numerical gate at the new
     chunk size (`X-6`). A chunk change moves every row count the gate exercises.
  2. `RT-3` records that short-request costs historically kept 1024 as the
     default, and this measurement is one long case. The right form is a
     **shape-based chunk policy** — 4096 for long prompts, 1024 for short — not
     a blanket default flip.
- **One case, one host, prefill only.** Not a category/heldout qualification and
  not a decode or MTP result.
- The earlier September 15 engine A/B recorded **+3.94%** at a larger chunk.
  That is not contradicted here — different date, and the ledger notes the
  September 14 recovery changed the composition — but this measurement does not
  confirm it either, and is not presented as doing so.

---

# X-6 gate re-run at chunk 4096: passes, and both arms are bit-identical

RT-1's first promotion blocker is the ledger's `X-6` requirement that the
numerical gate be re-run at the new chunk size — and the gate states its own
scope explicitly: *"A complete run at one chunk size never qualifies another
chunk size."* So the chunk-1024 certification does not transfer by default.

Re-run at chunk 4096 for the promoted route at canonical depth:

```bash
execution_profile_q8_wmma_prefill_layers_gate.py --route wmma --layers 16-47 \
  --prefill-chunk-size 4096 --decode-steps 128 --repeat-runs 3 \
  --require-cached-build --output gate-wmma-16-47-chunk4096.json
```

## Result

| | chunk 1024 (recorded) | chunk 4096 (this run) |
|---|---:|---:|
| teacher logits sha256 | `550bb9b8…` | **`550bb9b832bc7ba2…`** |
| candidate logits sha256 | `65370710…` | **`65370710c0b9c40f…`** |
| mean KL | 3.79552e-4 | **3.795524e-4** |
| status | complete | complete |
| `hard_gates_passed` | true | true |

**Both arms reproduce the chunk-1024 certification bit-identically.** Not merely
the mean KL agreeing to six figures — the full trajectories hash the same. So the
route's arithmetic and its drift are both **invariant to the prefill chunk size**
over 1024 → 4096.

Other gates at chunk 4096: `p95` 1.762e-3, `max` 1.530e-2, `top1_rate` 0.99354,
`repeat_deterministic: true`, `canonical_numerical_qualification_claim: true`,
teacher rows 1548, chunk recorded as 4096.

## This is not the false-pass mode

The gate previously returned **KL exactly 0.0 with `measurement_valid: true`** on
a route that never ran, so a clean pass is the failure to rule out *first*. Here
the engagement is recorded, not inferred:

- `candidate_route_engaged: true`, `teacher_route_engaged: false`
- `wmma_prefill` candidate engaged rate **0.373585** against the teacher's **0.0**
- **792** launches of `wmma_prefill_f32_f32_out`
- `selectors_observed_in_one_arm_only: []` — the attribution rests on both arms
  being observed
- `selectors_differing_between_arms: ["q4k_rowtile", "wmma_prefill"]`

The KL is nonzero, the arms differ, the teacher is genuinely the cleared exact
chain (`teacher_chain: exact coltile8_rowbatch4_f32_f32_out`), and the hashes
match a certification taken at a different chunk size in a different session.

## What this does and does not clear

**Clears** RT-1's first blocker for the promoted route: the chunk-4096
configuration passes the canonical numerical gate and reproduces the chunk-1024
result bit-identically, so the chunk change does not perturb this route.

**Does not clear** RT-1's second blocker (`RT-3`: short-request behaviour and the
+4.96 GB allocation), and does not clear `X-7`'s remaining profile blockers,
which are unchanged and reported by the gate itself: category-heldout,
strict-teacher/BF16-relative, control/isolation/graph/lifecycle, task-quality,
and engagement/repair-oracle coverage. Those are pre-existing and are not
created or worsened by the chunk change.

**Still not promoted.** No default changed.

---

# Short-request cost: does not exist, and the mechanism says why

RT-1's second blocker was `RT-3`'s recorded short-request cost at a larger chunk.
Measured with `code-p512`, three repetitions per arm, same protocol:

| code-p512 | samples (s) | median | peak allocation |
|---|---:|---:|---:|
| chunk 1024 | 2.0074, 2.0059, 2.0525 | 2.0074 | 83,857,109,608 B |
| chunk 4096 | 2.0114, 2.0057, 2.0379 | 2.0114 | **83,857,109,608 B** |

Wall time is indistinguishable: +0.20%, which is **0.08x** the larger within-arm
spread, and the arms overlap. `token_id` and `logits_sha256` are identical.

**Peak allocation is byte-identical**, not +4.96 GB. So a short request pays
*nothing* for the larger chunk size - neither time nor memory.

## Why: the workspace is already shape-aware

`Qwen4ExpPrefillWorkspace.allocate_scratch` sizes every owner at
`rows = min(chunk_size, runner.max_sequence_length)`, and the harness sets
`max_sequence_length = max(768, prompt_tokens + 1)`:

| case | `max_sequence_length` | chunk 1024 | chunk 4096 | workspace |
|---|---:|---:|---:|---|
| `code-p512` | **768** | min(1024, 768) = **768** | min(4096, 768) = **768** | identical |
| `code-p4096` | 4097 | min(1024, 4097) = **1024** | min(4096, 4097) = **4096** | 4x |

So for any prompt where `max(768, prompt + 1) <= 1024` - i.e. **prompt <= 1023
tokens** - both chunk sizes allocate the same workspace and run the same kernels
on the same rows. They are the same execution. The +4.96 GB appears only once the
sequence length exceeds 1024, which is exactly when the larger chunk does more
work and buys the -4.94%.

**Correction to my own earlier reasoning.** I wrote that a shape-based policy
would have to make *the allocation* shape-aware rather than the chunk loop, and
that this would be "a different and larger change than RT-3 assumes". That was
wrong: the allocation is already shape-aware, through `max_sequence_length`. I had
the `min(chunk, max_seq_len)` expression right and assumed `max_seq_len` was
constant across runs; it is prompt-derived. The measurement corrected me, which is
what it was for.

## What this means for promotion

`RT-3`'s short-request objection **does not reproduce** on this host and
configuration: there is no short-request penalty to trade against. The remaining
question is purely **memory admission on long prompts** - can the host afford a
89.40 GB peak? - and that is a property of the long prompt that pays for it, not a
cost imposed on short ones.

**Still not promoted.** No default changed. This section removes one blocker and
narrows the remaining one to a single memory-admission question; it does not
settle it.

---

# Qualification: the "default" in this packet is the harness's, not the server's

Everything above compares **1024 -> 4096**, because 1024 is what the benchmark
protocol uses and what the recorded command in the attribution artifact passed.
That is a valid measurement of the protocol it ran, but it is **not a description
of the production change**, and the difference matters.

The engine has more than one chunk-size default:

| site | value | governs |
|---|---:|---|
| `EngineLoopConfig.max_prefill_chunk_tokens` (`DEFAULT_MAX_PREFILL_CHUNK_TOKENS`) | **256** | the serving/CLI path |
| `EngineLoop.__init__(prefill_chunk_size=...)` | 1024 | only the no-config fallback |
| `EngineLoop` CLI default | 256 | `HIPENGINE_MAX_PREFILL_CHUNK_TOKENS` |
| `qwen4exp_profile_gap.py --prefill-chunk-size` | 512 | this harness's own default |
| what this packet passed | 1024 | the recorded attribution command |

`EngineLoop.__init__` line 313's 1024 is a *fallback*: line 1907 sets
`self.prefill_chunk_size = int(resolved_config.max_prefill_chunk_tokens)`, so a
supplied config overrides it with 256 by default. The CLI agrees (line 1665:
"default: 256").

**Consequence.** The serving-relevant comparison is **256 -> 4096**, which is 16x
the chunk rather than 4x, and it has **not been measured**. So:

- The -4.94% is a measured result for **1024 -> 4096 under this harness protocol**.
- It is **not** evidence for the size of the production change, which may be
  larger (more chunks collapsed) or smaller (different row-count regime).
- Nothing here should be cited as "the server would be 4.94% faster".

The X-6 gate result above is unaffected in kind: it establishes that the
**route's arithmetic and drift are invariant to the chunk size** over the range
tested, and that conclusion does not depend on which default is in play. But the
speedup figure does.

**Still not promoted.** No default changed. If this is pursued, the next
measurement is 256 -> 4096 on the same case and protocol, not 1024 -> 4096 again.
