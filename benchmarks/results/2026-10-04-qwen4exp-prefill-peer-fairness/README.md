# Qwen4Exp prefill chunk 4096: no peer-fairness cost, and the win IS reachable through the serving loop

RT-3 listed "no peer-fairness measurement" as a real gap, and RT-1's promotion was
gated on it. This packet measures it, and then goes further: it also exercises the
**generator-level chunk 4096** through the engine loop and reproduces RT-1's win.

**Two independent results.**

1. **Raising the engine-loop cap 1024 → 4096 does not degrade a decoding peer and
   does not change the prefill wall.** That falsifies the corrected mechanism that
   was blocking the promotion — the concern that raising the cap "quadruples the
   per-tick prefill bound that protects decode interleaving". `protect_decode`
   protects decode by *priority*, not by bounding the tick, and `prefill_batch`
   hands the runner the **whole prompt** once the last slice arrives, so the cap
   changes how many loop ticks accumulate the prefill, not how much work any tick
   does.
2. **Raising the generator chunk 1024 → 4096 does change the executed workspace,
   and it delivers RT-1's prefill win through the serving loop: −5.74% prefill
   wall.** The peer is not harmed — its time-to-first-token *improves* by 6.43%
   because it waits behind a shorter prefill, its inter-token latency median is
   unchanged, and its worst single gap halves (147.9 → 75.1 ms).

Result 1 alone would have left the −4.94% with no serving-loop route. Result 2
shows the route exists and it is the **workspace**, not the cap. The two knobs are
not interchangeable and this packet separates them cleanly.

## Harness

`scripts/qwen4exp_prefill_peer_fairness.py`, one process per arm, one arm per
model load, production profile, cached build. Two rows on a shared engine loop:

- **B** — the canonical `code-p4096` prompt (4096 tokens), `max_tokens=1`.
- **A** — the peer, `code-p512` truncated to a 16-token prompt.

`max_sequence_length=4097`, matching RT-3's protocol (`max(768, prompt_tokens+1)`),
because the workspace is `min(prefill_chunk_size, max_sequence_length)` rows.

Engagement is recorded from inside the run, not assumed: the session's
`_prefill_workspaces[*].prefill_chunk_size` and the engine loop's
`prefill_chunk_size` are read back after `prepare()`, and every `prefill_batch`
call is traced with its row count.

Host: gfx1151 APU, `tuned-adm` `accelerator-performance`, CPU governor
`performance`. Model: `unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL`.

## Result 1 — a decoding peer, long prefill submitted mid-stream

The peer decodes continuously (400 tokens, ~20 s); B is submitted after 8 peer
tokens. `--order peer_first`.

| arm | cap | prefill ticks | peer ITL median | peer ITL max | prefill wall |
|---|---:|---|---:|---:|---:|
| `g1024_l1024` | 1024 | `[16, 1024, 1024, 1024, 1024]` | **49.662 ms** | 54.978 ms | 35.268 s |
| `g1024_l4096` | 4096 | `[16, 4096]` | **49.608 ms** | 54.753 ms | 35.250 s |
| **delta** | | | **-0.054 ms (-0.11%)** | -0.225 ms (-0.41%) | -0.018 s (-0.05%) |

The peer's inter-token latency is unchanged. So is the prefill wall.

**What the trace actually shows, and it matters for reading the table.** The peer
finished all 400 tokens at +19.4 s and B's prefill did not start until then — the
two rows never overlap, because `protect_decode` keeps scheduling decode while a
decode row is due. Under this policy a continuously decoding peer **starves the
prefill**; the prefill never gets to interrupt the peer. The 35.3 s prefill wall
(against ~16 s solo) is the prefill absorbing that contention, not the peer.

## Result 2 — a second request arriving behind a long prefill

Both rows are queued before the first tick, long row first, so the loop's own
FIFO ordering decides and the arms stay comparable. `--order queued`.

| arm | cap | prefill ticks | peer TTFT | peer tokens | prefill wall |
|---|---:|---|---:|---:|---:|
| `g1024_l1024` | 1024 | `[1024, 1024, 1024, 1024, 16]` | **16.860 s** | 64 | 15.999 s |
| `g1024_l4096` | 4096 | `[4096, 16]` | **16.841 s** | 64 | 16.012 s |
| **delta** | | | **-0.019 s (-0.11%)** | | +0.014 s (+0.09%) |

A request landing behind the prefill waits the prefill in both arms, to within
0.11%. The peer's worst inter-token gap is 151.7 ms at cap 1024 and 128.4 ms at
cap 4096 — a single outlier against a 16.8 s wait, and smaller at the larger cap.

`--order prefill_first` was **abandoned as unusable**: `poll()` runs a whole tick
synchronously, so with cap 4096 the first `poll()` does not return until the whole
prefill is done and the peer gets submitted *after* it. That produced a spurious
"peer TTFT 0.88 s vs 16.88 s" that was purely an artifact of when the harness
managed to enqueue the peer. It is not reported as a result.

## Engagement: the executed chunk really changed

`loop_prefill_batch_chunks` goes `[1024, 1024, 1024, 1024, 16]` →
`[4096, 16]`, and `loop_prefill_batch_calls` 5 → 2. The cap reached the scheduler
and changed the tick decomposition by 4x, which is exactly the change the ledger's
mechanism predicted would cost peer fairness. It did not.

The session workspace is `[[1024], [1024]]` in Results 1 and 2 — both sessions of
the c2 serving runner, unchanged, because those arms vary the cap. In Result 3 it
is `[[4096], [4096]]`, and `prefill_token_ids.nbytes` goes 8,192 → 32,768, so the
workspace change is observed rather than inferred.

## Result 3 — the generator chunk 4096 through the serving loop

Results 1 and 2 vary the **cap**, which changes the tick decomposition and not the
workspace. This result varies the **generator chunk**, which is what actually sizes
the workspace. Same protocol, same `--order queued`.

| arm | gen chunk | loop cap | workspace | `prefill_token_ids` | prefill wall | peer TTFT | peer ITL median | peer max gap |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `g1024_l1024` | 1024 | 1024 | `[1024]` | 8,192 B | 15.992 / 15.998 s | 16.854 / 16.864 s | 49.19 / 49.24 ms | 147.9 / 155.5 ms |
| `g4096_l1024` | **4096** | 1024 | **`[4096]`** | **32,768 B** | **15.075 / 15.070 s** | 15.770 / 15.777 s | 49.09 / 49.15 ms | **75.1 / 71.9 ms** |
| `g4096_l4096` | **4096** | 4096 | **`[4096]`** | **32,768 B** | **15.001 / 15.072 s** | 15.717 / 15.796 s | 49.17 / 49.10 ms | **70.5 / 73.3 ms** |
| **delta** | | | | | **−5.74% / −5.80%** | −6.43% / −6.44% | −0.2% | **−49% / −54%** |

Two values per cell are two independent repetitions, one process per arm each.

**The prefill win reproduces through the engine loop, and it is solid at this sample
size.** RT-1 measured −4.94% by calling `runner.prefill(ids)` directly, bypassing the
loop; at the same prompt and host the serving loop measures **−5.74% / −5.80%**.

**Separation is far beyond RT-1's own bar.** Within-arm spread is **0.005 s (0.03%)**
for both arms across the two repetitions, and the arms **do not overlap**: the
slowest `g4096_l1024` run (15.075 s) is 5.74% faster than the fastest
`g1024_l1024` run (15.992 s). The median delta is **183x** the larger within-arm
spread, against the 5.70x RT-1 required for the original claim.

**The cap adds nothing on top of the chunk.** At repetition 2, `g4096_l4096`
(−5.79%) and `g4096_l1024` (−5.80%) are indistinguishable, and `g4096_l4096` is the
*more* variable arm (0.071 s spread vs 0.005 s). That is the same conclusion
Results 1 and 2 reach from the other direction: the cap is not the lever.

**The peer is not harmed — it benefits.** Its TTFT falls 6.43% because it is queued
behind a prefill that finishes sooner, its ITL median is unchanged (−0.2%), and its
worst single gap **roughly halves**, from 147.9/155.5 ms to 75.1/71.9 ms, in both
repetitions.

**Engagement is recorded in-run:** the session's
`_prefill_workspaces[*].prefill_chunk_size` goes `[1024]` → `[4096]`,
`prefill_token_ids.nbytes` goes 8,192 → 32,768, and the traced session prefill
reports `prefill_chunk_size` 1024 → 4096 with `workspace_chunks` `[1024]` →
`[4096]`.

### The override that was needed to run this

The first attempt at this arm failed with `copy size 32768 exceeds device buffer
size 8192` and was recorded as a limit. The cause: patching
`Qwen4ExpGGUFTextGenerator.__init__` alone is **not** sufficient. `LLM` reaches the
generator through the registry factory, which sets
`_implicit_profile_chunk_size = prefill_chunk_size is None` — True whenever no chunk
is passed, which is always on this path — and
`qwen4_exp_profiles._bind_default_chunk` then runs *after* construction and resets
`generator.runner.prefill_chunk_size` to 1024, silently reverting the override while
the already-allocated workspace stayed at 1024 rows. Neutralising
`_bind_default_chunk` as well makes the chunk survive to the workspace allocation.

**This is a measurement-harness fix, not a production change.** In production the
chunk is 1024 for exactly this reason, so a promotion has to move
`_bind_default_chunk`'s constant too — it is the real binding point, and this packet
is the first thing to exercise it.

## What this means for RT-1

The blocking claim — "either policy reaches 4096 only by quadrupling the per-tick
prefill bound that protects decode interleaving" — is **falsified as stated**. The
per-tick prefill bound is not what protects decode interleaving; decode priority
is. Measured cost of the 4x tick decomposition: **nothing**, on both the peer's
ITL and its TTFT, and nothing on the prefill wall either.

Two consequences worth separating:

- **The `-4.94%` IS reachable through the serving loop — via the generator chunk.**
  RT-1 measured it by calling `runner.prefill(ids)` directly, and Result 3 above
  now reproduces it through the engine loop at −5.74%. The route is the workspace,
  not the cap.
- **The cap is a scheduling knob that does not change the executed chunk**, so a cap
  promotion is not the route to that number — and this packet found no
  peer-fairness reason to withhold one either. Cap and workspace are separate
  knobs.

**What is still open is the memory-admission cost, which this packet did not
measure.** RT-1 and RT-3 measured the 4096-row workspace at **+4.96 GB peak** on the
p4096 case (+5.9%), paid by the prompt that receives the win. The task that produced
this packet explicitly forbids promoting 4096 on a memory-admission argument, and
nothing here changes that: this packet establishes that the win is real, reachable,
and free of peer-fairness cost, and leaves the admission decision where it was.

## Limits

- **Two repetitions per arm.** Within-arm spread is 0.03% and the arms do not
  overlap, so the prefill deltas (−5.74%, −5.80%) are established; the peer-ITL
  medians are still not resolved beyond "unchanged".
- **No memory measurement.** The +4.96 GB workspace cost is quoted from RT-1/RT-3,
  not re-measured here.
- **One peer shape** (16-token prompt, 64 or 400 output tokens), one long prompt
  (4096), one host, one quant, one policy (`protect_decode`). The `token_budget`
  policy was not exercised.
- **The peer's TTFT in Results 2 and 3 is dominated by queue position**, not by
  chunk size: its own prompt is behind B's. It measures "a request arriving during
  a long prefill", which is the question asked, but it is not a measure of the
  peer's own prefill cost.
- **Result 1's ITL window includes no overlap between the rows**, because under
  `protect_decode` there is none to measure. That is the finding, but it means
  "peer ITL during prefill" is measured on a peer that is *not* being contended,
  by construction of the policy.
