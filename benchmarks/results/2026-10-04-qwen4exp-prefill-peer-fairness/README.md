# Qwen4Exp prefill chunk 4096: no peer-fairness cost (RT-1's missing measurement)

RT-3 listed "no peer-fairness measurement" as a real gap, and RT-1's promotion was
gated on it. This packet measures it. **Raising the engine-loop prefill cap from
1024 to 4096 does not degrade a decoding peer, and does not change the prefill
wall.**

That falsifies the corrected mechanism that has been blocking the promotion. The
concern was that raising the cap "quadruples the per-tick prefill bound that
protects decode interleaving". It does not: `protect_decode` protects decode by
*priority*, not by bounding the tick, and `prefill_batch` hands the runner the
**whole prompt** once the last slice arrives, so the executed GPU prefill is one
4096-token call at either cap. The cap changes how many loop ticks accumulate it,
not how much work any tick does.

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

The session workspace is `[[1024], [1024]]` in every arm reported here — both
sessions of the c2 serving runner, unchanged, because this packet varies the cap.

## What this means for RT-1

The blocking claim — "either policy reaches 4096 only by quadrupling the per-tick
prefill bound that protects decode interleaving" — is **falsified as stated**. The
per-tick prefill bound is not what protects decode interleaving; decode priority
is. Measured cost of the 4x tick decomposition: **nothing**, on both the peer's
ITL and its TTFT, and nothing on the prefill wall either.

Two consequences worth separating:

- **The `-4.94%` still has not been demonstrated through the serving loop.** RT-1
  measured it by calling `runner.prefill(ids)` directly. This packet shows the
  loop's cap is a *scheduling* knob that does not change the executed GPU prefill,
  so a cap promotion is not the route to that number. Whether the generator-level
  chunk 4096 (which does change the workspace) delivers it in the serving path is
  **not measured here** — see Limits.
- **The fairness objection to raising the cap is gone.** If a cap promotion is
  wanted for its own scheduling reasons, this packet found no peer-fairness
  reason to withhold it.

## Limits

- **The generator-chunk 4096 arms did not run.** Forcing
  `Qwen4ExpGGUFTextGenerator(prefill_chunk_size=4096)` in the serving path fails
  with `copy size 32768 exceeds device buffer size 8192` at
  `runner.prefill_token_ids`. The workspace was sized 1024 rows, not 4096, so the
  override did not survive
  `qwen4_exp_profiles._bind_default_chunk` (which resets
  `generator.runner.prefill_chunk_size = 1024` after construction whenever the
  factory left `_implicit_profile_chunk_size` True — always, when `LLM` passes no
  chunk). **So this packet varies the cap, not the workspace, and does not
  reproduce the `-4.94%`.** The two are different knobs and the ledger should not
  treat them as interchangeable.
- **One peer shape** (16-token prompt, 64 or 400 output tokens), one long prompt
  (4096), one host, one quant, one policy (`protect_decode`). The `token_budget`
  policy was not exercised.
- **One repetition per arm.** The deltas are all ≤0.11%, far inside any plausible
  run-to-run spread, so a single repetition cannot resolve them — but the claim
  here is *no measurable degradation*, and a cost large enough to matter would
  have shown.
- **The peer's TTFT in Result 2 is dominated by queue position**, not by chunk
  size: its own prompt is behind B's. It measures "a request arriving during a
  long prefill", which is the question asked, but it is not a measure of the
  peer's own prefill cost.
- **Result 1's ITL window includes no overlap between the rows**, because under
  `protect_decode` there is none to measure. That is the finding, but it means
  "peer ITL during prefill" is measured on a peer that is *not* being contended,
  by construction of the policy.
