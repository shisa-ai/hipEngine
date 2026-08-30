# Qwen3.8-27B DFlash2 GGUF Campaign (gfx1151 first, gfx1100 functional)

Status: **decision — not promoted (D3/D4 complete, measured blocker; loss attribution corrected 2026-08-22)** — D0 complete (metadata/validation/lineage/CPU oracles);
D1 complete: NumPy drafter reproduces the reference greedy chain exactly (D0
RED pin) **and** the GGUF-target tap capture + cycle driver runs end-to-end
(`scripts/dflash2_gguf_cycle.py`: full-prompt 5-layer tap capture at prefill,
mask-noise block proposal, sequential greedy commit-only verify, projected-context
cache; DFlash2 greedy == pure-AR greedy 20/20 on the smoke prompt). D2a
complete: native grouped dynamic conv + top-16 + candidate-selector kernels
RED-pinned vs the CPU oracles (6 GPU tests) and registered for gfx1100 +
gfx1151. D2b complete: native drafter forward + selector wiring
(`hipengine/speculative/dflash2_native.py`): conv+attention+MLP forward and
top-16 selector reproduce the numpy oracle to BF16 tolerance and are
deterministic; the native select path equals numpy propose; sliding-attention
kernel RED-pinned; `rocprofv3 --kernel-trace` smoke shows all expected kernels
under expected names (10 GPU tests: 7 kernel RED + 3 forward wiring).
D3 complete: native drafter wired into `scripts/dflash2_gguf_cycle.py`
(`--native`) with end-to-end correctness (native greedy == AR 40/40 on the
smoke) and the B7 chain-batched verifier (`_run_dflash2_cycle_batch`) is
AR-exact on all 10 mtpbench prompts.
D4 complete (2026-08-19), attribution corrected (2026-08-22): full 10-prompt
mtpbench suite measured — DFlash2 B3 (the measured optimum) **8.85 tok/s =
0.66x AR** after the retained Q6 amortized select fix (7.70 = 0.575x pre-fix);
B5 4.26, B7 3.58; every row AR-exact. Against exact native MTP B3
(23.85 tok/s = 1.7845x AR) the best DFlash2 point is ~2.7x down, so the
**promotion rule is NOT met** and DFlash2 stays a diagnostic on this lane.
A follow-up B-sweep root-caused why the earlier "B-sweep" was flat:
`--block-size` was force-clamped to the drafter config 8, so every B ran the
full 8-row verify; after truncating the verify chain to the CLI block size, B3
is the measured optimum. A smaller-block forward was a measured net loss
(acceptance -7%, recall@16 drops) and its path was removed (`597dbd4ad`); the
drafter always runs its trained block geometry.

**Why it loses — corrected 2026-08-22.** The pre-2026-08-22 record attributed
the loss to an acceptance gap (DFlash2 2.80 vs "MTP 3.85", ~0.96/row) plus an
O(N^2) APU verify, and concluded that no operating point on this lane could
reach MTP B3 regardless of optimization. **That attribution was wrong.** `3.85`
is MTP's `target_forward_rows / cycles` — verify rows per cycle — not its
accepted tokens per cycle, which is **2.85**. Corrected, DFlash2 B3 (2.80
tokens/cycle, 0.70 per verify row) is **at parity** with MTP B3 (2.85, 0.74).
The entire 2.7x deficit is cost, not drafting quality:

- drafter forward + select ~96 ms/cycle that MTP does not pay (MTP's proposal
  is 2.4 ms), running at roughly a fifth of the achievable bandwidth for its
  measured 3.584 GiB residency;
- a verify that costs 166 ms for 4 rows where MTP's costs 111 ms for 3.85 rows
  — and is **not** the same code path (different target file, different
  harness, plus DFlash2-only 5-layer tap capture inside the timed region);
- a rowtile admission cliff at four rows (`_PACK8_ROWTILE_MAX_ROWS = 4`) that
  makes the 8-row verify re-read the full weights once per row (620 ms ≈ 8.0
  weight sweeps), which is what forces the shallow chain — not quadratic
  attention, whose contribution here is microseconds. The reverted rowtile-8
  experiment (620 -> 310 ms, AR-divergent on `code_lru_cache`, never
  root-caused) is the top open item.

So the campaign's decision stands on the measured 2.7x gap, **not** on a claim
that the gap is unclosable. The cross-lane FP8-BLOCK / PRO 6000 datapoint
previously used to argue "hardware economics that do not transfer" is
unverifiable from this host and its MTP-strength half is refuted by the
corrected MTP number — see the Economics section.
Single clean run per B (all 10 prompts AR-exact), artifacts under
`benchmarks/results/`.
Remaining: N1-N4 rerun plan in the Economics section (GPU-blocked); D5 gfx1100
functional (optional given non-promotion).

The complete quantitative record (corrected acceptance model, per-cycle cost,
cost model in weight sweeps, the rowtile admission cliff, protocol-mismatch
caveats, retained cost reductions, reproducibility gaps, and the N1-N4 rerun
plan with falsifiable predictions) is consolidated in the **Economics**
section below — the single source for the DFlash2-vs-MTP analysis on this lane.

This document defines the campaign to bring `z-lab/Qwen3.8-27B-DFlash2`
drafting to the closed Qwen3.8-27B GGUF production path on Radeon 8060S /
`gfx1151`, with **functional (correctness-gated, untuned) support on
gfx1100/W7900** as an explicit deliverable. Milestone status lives in the
worklog.

References:

- DFlash 2 blog: https://inco.ai/blog/dflash2/ (2026-08-18)
- Reference implementation: `~/dflash` (cloned from `github.com/z-lab/dflash`,
  commit `07ebd93`, read-only). `dflash/model.py::DFlash2DraftModel`,
  `GroupedDynamicCausalConv`, `CandidateSelector`, `dflash_generate`;
  `dflash/model_mlx.py` is the quantized-target variant.
- llama.cpp DFlash2 GGUF path: `ggml-org/llama.cpp` PR #27342 and the
  `incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M` drafter weights (comparator lane).
- **Cross-lane datapoint (2026-08-19) — UNVERIFIED FROM THIS HOST:**
  `~/sm120-tuning/QWEN38-27B.md` "Correct benchmark", cited as matched
  FP8-BLOCK MTP3 vs FP8-BLOCK DFlash2 on PRO 6000 (vLLM PR #52816, 262144 ctx,
  greedy): DFlash2 wins decode/E2E at c<=16 (accept length 3.88 vs 3.04; c=1
  total 216.6 vs 167.1 tok/s; MTP-Bench decode +36%, E2E -46%), MTP3 wins only
  at queue-saturated c>=24. **That path does not exist on `strixhalo`, the host
  this campaign ran on, and the repo is not among the declared read-only
  reference peers in `AGENTS.md`.** It cannot carry a claim here until it is
  re-cited with a physical host identity and a reachable path. It was
  previously used to argue the 8060S loss is hardware economics that does not
  transfer; that argument is withdrawn (see Economics).
- Our DFlash history and verifier/accept/commit infrastructure:
  [`DFLASH.md`](DFLASH.md). Closed Qwen3.8-27B target campaign:
  [`QWEN38-27B-GFX1151-CAMPAIGN.md`](QWEN38-27B-GFX1151-CAMPAIGN.md) (AR
  13.10/12.92/13.10 tok/s at 512/1K/4K; exact native MTP B3 23.85 tok/s =
  1.7845x own AR).

## Economics — why DFlash2 as implemented loses to MTP on this lane

> **Correction (2026-08-22) — this section was rewritten; the pre-2026-08-22
> version's central MTP number was wrong.** `3.85` was read out of the MTP B3
> artifact as MTP's accepted tokens per cycle. It is `target_forward_rows /
> cycles` = **verify rows per cycle**. It was then divided by the 4-row block
> to manufacture a "0.96 per-row acceptance", compounding the same error.
> MTP B3's real accepted tokens per cycle is **2.85**, and its real per-row
> acceptance is **0.74**. Against DFlash2 B3's 2.80 / 0.70 the acceptance gap
> is **~2%, not ~27%**. Every conclusion that rested on that gap is superseded:
> "MTP B3 sits near the acceptance cap", "DFlash2's acceptance is below MTP's",
> the *not-an-optimization-problem* framing, and the MTP-strength half of the
> cross-lane reconciliation. The measured throughput rows are unchanged; only
> the attribution changes. Superseded worklog entries: `20260819T152835`,
> `20260819T165514`, `20260820T005019`. Correction entry:
> `worklog/entries/20260822T041749.084809Z-lhl-dflash2-economics-attribution-correction-901e48.md`.

This section is the single self-contained record of the DFlash2-vs-MTP
analysis. All DFlash2 numbers are the closed Qwen3.8-27B **Q4_K_M** GGUF
target, gfx1151 (Radeon 8060S, host `strixhalo`), resident session, greedy,
full 10-prompt mtpbench suite at `--max-new-tokens 40`, AR-exact on every row.
All MTP numbers are read from
[`exact Q4_K_S native B3`](../benchmarks/results/2026-08-17-gfx1151-qwen38-27b-q4ks-exact-native-b3.json),
which is a **different target file and a different protocol** — see
"Protocol mismatch" below before treating any absolute rate as same-lane.

### Baselines

| Lane | tok/s | ratio AR | protocol |
| --- | --- | --- | --- |
| Pure autoregressive (Q4_K_M, in-session) | 13.4 | 1.00x | `dflash2_gguf_suite_smoke.py`, 40 tok |
| Exact native MTP B3 (Q4_K_S) | 23.85 | 1.7845x | `qwen36_dense_gguf_suite.py`, 25 tok |
| DFlash2 B3 (post-Q6 select) | 8.85 | 0.66x | 40 tok, 4-row verify |
| DFlash2 B5 (pre-Q6 select) | 4.26 | 0.32x | 40 tok, 6-row verify |
| DFlash2 B7 (pre-Q6 select) | 3.58 | 0.27x | 40 tok, 8-row verify |

DFlash2's best point is ~2.7x below exact MTP B3. That headline is unchanged
and remains the reason the path is not promoted. What follows is why.

### What the numbers actually say

Per-cycle decomposition, both at their B3 operating point. MTP's columns are
derived from the artifact's own counters (`cycles` 87, `accepted_draft_tokens`
161, `proposed_draft_tokens` 248, `target_forward_rows` 335,
`stage_seconds.target_verify` 9.663 s, `stage_seconds.proposal` 0.206 s);
DFlash2's from the fox-prompt cycle split plus `mean_acceptance` (which is
produced tokens per cycle including the bonus token, so the two are the same
quantity):

| | verify rows/cycle | tokens/cycle | tokens per verify row | verify ms | draft+select ms | cycle ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact native MTP B3 | 3.85 | **2.85** | **0.740** | 111 | 2.4 | 118 |
| DFlash2 B3 | 4.00 | **2.80** | **0.700** | 166 | 96 | 264 |

Three things follow immediately.

1. **Acceptance is at parity.** 2.80 vs 2.85 tokens per cycle; 0.70 vs 0.74
   per verify row. DFlash2's drafter is not out-accepted by MTP on this lane
   in any meaningful sense.
2. **MTP proposes adaptively.** 248 proposed drafts over 87 cycles is 2.85 of
   a 3-draft budget — MTP's confidence gate declines to draft when it expects
   rejection, which is why it verifies 3.85 rows rather than 4. DFlash2 runs a
   fixed B and always pays for all 4 rows. That is a mechanism DFlash2 does not
   have, not a quality difference.
3. **The whole deficit is cost.** ~96 ms of drafter+select that MTP does not
   pay, plus ~55 ms of verify that MTP does not pay for the same work.

### The cost model in weight sweeps

Decode on this target is weight-bandwidth-bound and already at the roof.
`docs/ROOFLINE-gfx1151.md` gives ~221 GB/s practical read; Q4_K_M is
17,106,775,008 bytes, so one full weight sweep is **77.4 ms**, and measured
AR is 74.6 ms/token — i.e. AR runs at ~1.0 sweeps per token. Additional verify
rows add compute (a few ms for 4-8 rows of a 27B dense forward) but **no
additional weight traffic** if the kernel reuses the loaded weights across
rows. So sweeps-per-cycle is the right unit:

| path | rows | ms | weight sweeps/cycle |
| --- | ---: | ---: | ---: |
| AR (per token) | 1 | 74.6 | 0.96 |
| MTP B3 verify (Q4_K_S, 72.9 ms/sweep) | 3.85 | 111 | **1.52** |
| DFlash2 verify | 4 | 166 | **2.14** |
| DFlash2 verify | 8 | 620 | **8.01** |
| ideal amortized verify | any | — | ~1.0-1.1 |

The 8-row number is one full weight re-read per row. That is the signature of
**no cross-row reuse at all**, not of quadratic attention: the 4→8 row step
adds 26 causal query-key pairs over a few hundred KV tokens, which is
microseconds against a 620 ms cycle — the previously recorded "O(N^2) causal
block attention" explanation is off by roughly three orders of magnitude and
is withdrawn.

### Where the 8-row cliff comes from

`hipengine/runtime/gguf_linear.py` caps the amortized resident-pack8 route at
four rows (line numbers as of `4699392d0`):

```
_DENSE_BF16_ROWTILE_MAX_ROWS = 4      # line 150
_PACK8_ROWTILE_MAX_ROWS      = 4      # line 159
```

Above four rows the weight-reuse rowtile is not admitted and the verify falls
back to a per-row route. This is a dispatch admission bound, and it is
consistent with the D4 experiment that extended the Q4T16 dense rowtile to
eight rows and cut the 8-row verify **620 ms -> 310 ms** before being reverted
as AR-divergent on `code_lru_cache` (root cause never found). That revert is
the single most consequential open item in this campaign: it is what makes the
deep chain — the operating point where DFlash2's longer accept length would
pay — unreachable.

### The ceiling analysis, corrected

The old text called this "the airtight ceiling (not an optimization problem)".
The arithmetic in it was right; the causal attribution was not.

1. **Free-drafter bound (still true as stated).** Even at zero drafter cost,
   DFlash2 B3 = 2.80 tok / 0.166 s = 16.9 tok/s = 1.26x AR < MTP B3's 1.7845x.
   **But it binds because DFlash2's verify costs 166 ms for 4 rows while MTP's
   costs 111 ms for 3.85 rows — not because of acceptance.**
2. **Required-acceptance bound, recomputed.** At DFlash2's *own* 166 ms verify,
   matching MTP B3 needs 23.853 x 0.166 = 3.96 tokens/cycle, above the B3 cap
   of 4.0 — hence the old "no operating point" claim. At **MTP's** verify
   economics (111 ms) the requirement is 23.853 x 0.111 = **2.65** tokens/cycle,
   and DFlash2 already delivers **2.80**. The bound flips on the verify cost,
   which is an implementation number, not a property of the drafter.
3. **Modeled headroom** (from measured components; not itself a measurement).
   D6 measured drafter BF16 residency at 3.584 GiB; with the session's 0.6 GiB
   Q6 head that is ~4.5 GB touched per draft+select, or ~20 ms at the practical
   roof, against 96 ms measured (~5x). Combining an ideal ~1.1-sweep verify
   with a ~20 ms drafter gives roughly 27 tok/s at B3 and roughly 31 tok/s at
   B7 (3.49 tokens over ~113 ms) — i.e. **at or above MTP B3, with B7 becoming
   the optimum again**. These are model outputs, not results; they define what
   the rerun below would test, and nothing may be claimed from them.

**Corrected conclusion.** On Q4_K_M / 8060S, DFlash2's per-row acceptance is
within ~5% of hipEngine's MTP. The 2.7x throughput deficit is (a) an untuned
drafter forward running at roughly a fifth of achievable bandwidth, (b) a
verify that costs 1.4x MTP's for the same rows, and (c) a rowtile admission
cliff above four rows that makes the deep chain uneconomical. All three are
optimization problems of the kind this repo routinely closes. The campaign
result — **not promoted, diagnostic** — stands on the measured 2.7x gap, not
on any claim that the gap is unclosable.

### Why B3 is the optimum — with the caveat that matters

Throughput = acceptance(B) / (draft + select + verify(B) + commit). Acceptance
saturates with B (2.80 @ B3, 3.30 @ B5, 3.49 @ B7) while measured verify(B)
explodes past four rows, so the measured optimum is B3:

| B | verify rows | cycle cost | outcome |
| --- | ---: | --- | --- |
| 3 | 4 | 264 ms | **8.85 tok/s (0.66x AR)** — measured optimum |
| 5 | 6 | ~0.5-0.8 s | 4.26 (0.32x) |
| 7 | 8 | ~0.75-0.97 s | 3.58 (0.27x) |

This optimum is defined **against the current verify**, whose cost per row
above four rows is an artifact of the admission cliff above. If verify(B)
were amortized, acceptance(B) would still saturate but the denominator would
grow only ~10% from B3 to B7, and the optimum would move to the deep chain.
The B-sweep therefore does not establish that shallow chains are right for
DFlash2 on this hardware; it establishes that they are right for *this*
verify.

### Protocol mismatch — read before quoting any absolute rate

The MTP B3 row and the DFlash2 rows are not the same lane, and the campaign
previously presented them as if they were:

| axis | MTP B3 | DFlash2 |
| --- | --- | --- |
| target file | `Qwen3.8-27B-Q4_K_S.gguf` (16.12 GB) | `Qwen3.8-27B-Q4_K_M.gguf` (17.11 GB, +6.1%) |
| harness | `qwen36_dense_gguf_suite.py`, reusable target graphs | `dflash2_gguf_suite_smoke.py`, per-cycle Python + explicit syncs |
| tokens/prompt | 25 | 40 |
| verify call | plain native verify | `verify_target_block(capture_layer_output_hidden=[5 taps], capture_linear_state_rows=True)` |

So "verify (166 ms) is shared with MTP, not DFlash2-specific" — previously
recorded here — is **false on three counts**: different weight bytes, different
harness/graph-capture regime, and DFlash2-only 5-layer hidden-state capture
inside the timed region (`scripts/dflash2_gguf_cycle.py:417-422`). The ratio
normalization (0.66x vs 1.78x AR) absorbs most of the file-size difference but
not the harness or capture difference.

Three further measurement notes:

- `scripts/dflash2_gguf_suite_smoke.py:138` computes `max_new_tokens / ar_s`
  while `_run_ar` times only `max_new_tokens - 1` decode steps (the first token
  comes from prefill, untimed). The AR denominator is inflated ~2.6%, making
  DFlash2's ratios slightly conservative. Fix before the rerun.
- The B3 cycle split (draft 74 + select 22 + verify 166 + commit 2 = 264 ms) is
  **host wall-clock** from the since-removed `DF2_CYCLE_DEBUG` timers on the fox
  prompt (removed with the reduced-block cleanup, `597dbd4ad`), not a
  `rocprofv3 --kernel-trace`. The 10-prompt suite implies 2.799/8.845 =
  **316 ms/cycle**, so ~52 ms/cycle of host work (numpy tap projection, D2H,
  context append) is unattributed in that split.
- `max_accept = max_new_tokens - produced_total` truncates the final cycle's
  acceptance accounting, depressing `mean_acceptance` ~1-2% at 40 tokens.

### Cross-lane acceptance — unverified from this host

The 2026-08-20 reconciliation cited `~/sm120-tuning/QWEN38-27B.md` (matched
vLLM FP8-BLOCK MTP3 vs DFlash2 on RTX PRO 6000) for accept length 3.88 vs 3.04
and a DFlash2 decode/E2E win at c<=16. **That file does not exist on
`strixhalo`**, the host this campaign ran on; it is presumably on the SM120
box. It is also not among the declared read-only reference repos in
`AGENTS.md`. Per the evidence policy it cannot carry a claim here until it is
cited with a physical host identity and a reachable path.

| lane (engine) | DFlash2 accept length | MTP accept length | status |
| --- | --- | --- | --- |
| Q4_K_M / 8060S (hipEngine, greedy, B7 / B3) | 3.49 (8 rows) | 2.85 (3.85 rows) | **measured in-tree** |
| FP8-BLOCK / PRO 6000 (vLLM, greedy) | 3.88 | 3.04 | **unverified from this host** |
| BF16 ref (blog, T>0, block 8) | 4.80 | 4.28 | external, not reproduced |

With MTP corrected to 2.85, the reconciliation's second pillar also fails:
hipEngine's MTP is **not** unusually strong — at 2.85 tokens/cycle it is
slightly *below* the vLLM MTP3 datapoint's 3.04. The honest cross-lane
statement is now much weaker than what was recorded: DFlash2's accept length
is broadly similar across quantization levels (3.49 Q4 / 3.88 FP8 / 4.80 BF16,
though those are three engines, three protocols and two temperatures, and the
trend is monotone in target fidelity), and nothing in the in-tree data
supports "the 8060S loss is hardware economics that does not transfer".

### Retained cost reductions

| Fix | Effect | Status |
| --- | --- | --- |
| Drop redundant full-vocab logit host copy (top-1 of top-16 == argmax) | select 70 -> 56 ms | retained |
| **Q6 amortized select** (draft logits via the session's Q6_K head, read once across rows, vs the 2.54 GiB dequantized BF16 head) | **select 70 -> 22 ms**; B3 suite 7.70 -> 8.85 (+15%), acceptance unchanged, AR-exact | retained |
| Variable-block forward (smaller drafter block) | acceptance -7%, recall@16 down, throughput flat | rejected; path removed in `597dbd4ad`, drafter keeps its trained block geometry |
| rowtile-8 verify (620 -> 310 ms) | AR-divergent on `code_lru_cache` | **reverted, un-root-caused — top open item** |

Note on the variable-block finding: the recorded conclusion "the forward is
launch-bound, not compute-bound" does not follow from bs 8->4 being flat. A
weight-bound kernel predicts exactly the same flatness, because fewer block
rows read the same drafter weights. The *decision* (keep the config block
size) is still correct — acceptance -7% is the real signal — but the mechanism
claim is unsupported and the drafter's ~50 GB/s effective bandwidth points at
weight-bound-but-inefficient, not launch-bound.

### Memory economics (D6)

DFlash2 B3 pipeline ~19.5 GiB GTT (+3.6 GiB over the closed campaign's 15.899
GiB B3 budget) — inside the 8060S unified-memory capacity, so the BF16 drafter
does not exceed the APU budget and the GGUF-quantized drafter is not a
required follow-up. Unchanged by this correction.

### Reproducibility gaps on this host

Recorded so the rerun is scoped honestly. As of 2026-08-22 on `strixhalo`:

- The three DFlash2 artifacts carried `"gpu": "AMD Radeon Pro W7900-compute-lane
  (gfx1151 resident)"`, which names the wrong machine (W7900 is gfx1100) where
  every other artifact on this host says `"AMD Radeon 8060S Graphics"`.
  Corrected in place 2026-08-22, with `hardware.host: strixhalo` added.
- The drafter snapshot
  (`models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c...`) is **not
  present** on this host.
- The reference clone `~/dflash` is **not present**.
- `Qwen3.8-27B-Q4_K_S.gguf` (the MTP B3 baseline target) is **not present**;
  only `Qwen3.8-27B-Q4_K_M.gguf` remains under `/models/gguf/`.
- `~/sm120-tuning/` is **not present**.

None of this invalidates the recorded measurements, but the campaign cannot be
re-executed from this host as documented without re-fetching the drafter and
the Q4_K_S target.

### Next steps and what a matched rerun would settle

Blocked on GPU availability as of 2026-08-22. Ordered by leverage.

**N1 — Verify row-cost microbenchmark (cheap, highest information).** Sweep
`verify_target_block` rows 1..8 standalone on Q4_K_M, outside any decode loop,
with and without `capture_layer_output_hidden`, and report ms and
sweeps-per-cycle per row count. This converts the central disputed claim into
one curve. Falsifiable predictions: the curve is near-flat 2..4 rows and
**steps >=2.5x between 4 and 5 rows** (the `_PACK8_ROWTILE_MAX_ROWS = 4`
admission cliff); it does **not** grow smoothly as N^2. If the curve is
instead smooth and superlinear from 2 rows up, the original O(N^2) explanation
was right after all and the pre-2026-08-22 conclusion should be restored.

**N2 — Complete 2026-08-30: rowtile-8 AR divergence root-caused.** The old
experiment broadly widened rows<=8 dispatch. The safe replacement added the
physical-c8 scope, valid Q6 lm-head chunks, exact standard-Q4 rows5-8
single/dual/down owners, and quant/layout-specific standard-Q4 rows2-8 versus
qmicro rows2-4 caps. Current `code_lru_cache` R8 target IDs, pre/post-norm
hidden, five taps, selected row7 commit, and the next AR token are bit-exact;
serial/native wall is **721.5/204.4 ms (3.53x)**. This closes correctness, not
DFlash product economics.
[`artifact`](../benchmarks/results/2026-08-30-gfx1151-qwen38-dflash-row8-root-cause-closed.json)

**N3 — Blocked 2026-08-30: matched-protocol DFlash2-vs-MTP rerun.** The exact
Qwen3.8 DFlash2 snapshot `50307d4c...` and `~/dflash` reference remain absent;
the available Qwen3.6-35B-A3B drafter is incompatible and is not substituted.
Restore those assets before running this matrix. The current comparison
conflates four independent differences. Hold all four fixed: (i) one target
file, Q4_K_M for both; (ii) one harness and one timing boundary, with the
`max_new_tokens / (max_new_tokens - 1)` AR off-by-one fixed; (iii) the same
verify entry point, run with and without DFlash2's tap capture; (iv) the same
prompt suite and token budget. Cost is small — 10 prompts x 40 tokens x
{AR, MTP B3, DFlash2 B3/B5/B7} in one resident session, well under an hour of
GPU time plus N1.
[`closeout`](../benchmarks/results/2026-08-30-gfx1151-qwen38-dflash-p5-closeout.json)

What N3 would settle that the current data cannot:

- **Is acceptance parity real?** 2.80 vs 2.85 is currently a comparison across
  two target files, two harnesses and two token budgets. Matched, it becomes
  the number that decides whether DFlash2's drafter is ever worth its cost on
  any lane. Prediction: within +/-0.15 tokens/cycle of MTP at matched depth.
- **How is the 55 ms verify delta split** between weight bytes (Q4_K_S vs
  Q4_K_M), harness/graph-capture, and DFlash2's 5-layer tap capture? (i)+(ii)
  +(iii) decompose it directly. This is the difference between "DFlash2 needs
  a cheaper verify" and "DFlash2's tap capture is inherently expensive".
- **Where is the true B optimum?** With N1's row curve, the optimum is
  computed from acceptance(B)/cost(B) rather than swept against a broken
  denominator.
- **Does the drafter have ~5x headroom?** A `rocprofv3 --kernel-trace` of the
  drafter forward gives achieved bandwidth against the measured 3.584 GiB
  residency. Prediction: currently <60 GB/s effective; a hoisted/fused forward
  should exceed 120 GB/s.
- **Does the deep-chain hypothesis hold?** If N2 lands an amortized 8-row
  verify and DFlash2 retains 3.49 tokens/cycle, DFlash2 B7 would exceed MTP B3
  on this lane — which would **reverse this campaign's conclusion**. That is
  the outcome the current record actively forecloses and should not.

What the rerun would **not** settle: anything about the FP8-BLOCK / PRO 6000
lane (different hardware, engine and target format), or whether DFlash2's
acceptance advantage over MTP at BF16 fidelity reproduces in hipEngine. Those
need the SM120 host and a BF16 or FP8 target respectively, and remain out of
this campaign's scope.

**N4 — Adaptive proposal gate for the DFlash2 chain.** MTP verifies 3.85 rows
instead of 4 because it declines low-confidence drafts. DFlash2 has strictly
more signal available for the same decision (selector scores and top-16
margins per position) and its per-position recall decays with depth, so the
gate should pay more there than it does for MTP. Independent of N1-N3.

**N5 — gfx1100 functional smoke (D5).** Unchanged: hardware-blocked, optional.

## 1. What DFlash2 is, relative to what we already have
Our existing native DFlash path (chain/tree verifier, native-bulk
`TargetVerifyBatch`, `dflash_accept_chain_i32` / `dflash_commit_chain_i32`,
whole-cycle confidence gate) was built against DFlash **1** drafters
(`Qwen3.6-*` HF drafters and the Laguna DFlash GGUF lane). The released
`Qwen3.8-27B-DFlash2` drafter is the same qwen-family parallel block drafter
plus exactly two new mechanisms, and both live entirely on the **drafter**
side; verification stays a chain verify over root + accepted-path rows:

1. **Two-tap grouped dynamic convolution** (`attention_conv`, `mlp_conv`):
   before and after each attention and MLP sublayer of each of the 5 drafter
   layers (20 conv applications per draft forward),
   `Conv_k(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t-1}` where each coefficient is
   a learned `base_kernel` (per-channel) plus a dynamic correction from
   `kernel_projection(h)` shared per group of `conv_group_size=16` channels
   (320 groups, H=5120, `kernel_size=2`, projection output 1280). The first
   block position reads the last verified token's representation. This is
   block-local and stateless: it replaces depth (suffix-decay fix) without a
   sequential backbone.
2. **Low-rank bilinear path selector** (`candidate_selector`): keep top
   `selector_top_k=16` candidates per position (draft logits from the
   **target's** output head), then score adjacent pairs
   `S_t(a,b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>` with `selector_rank=256`
   codebooks A/B (vocab 248320 × 256) and a context gate
   `H = hidden_projection(h_t)` (5120→256). A greedy walk (T=0) or
   rejection-sampled walk (T>0) starting from the anchor (last verified token)
   traces one chain through the per-position candidate lists. The selected
   path is a **chain**, so the existing chain verifier/accept/commit machinery
   applies unchanged; tree/DDTree work is not required.

Reference acceptance (blog, block size 8, Qwen3.8-27B): mean acceptance
length 4.80 vs 4.28 for the model's native MTP. Our binding comparison is
against **our own exact native MTP B3 (1.7845x own AR)**, not external
headlines.

### 1.1 Model identity

| Field | Value |
| --- | --- |
| Drafter | `z-lab/Qwen3.8-27B-DFlash2` snapshot `50307d4c` |
| Architecture | `DFlash2DraftModel`, BF16, 81 tensors, ~3.85 GiB safetensors |
| Backbone | 5 × `sliding_attention` (window 2048, non-causal bidirectional window), 32 Q / 8 KV heads, head_dim 128, q/k per-head RMSNorm, RoPE theta 1e7 |
| Geometry | hidden 5120, intermediate 17408, vocab 248320 (uses **target** output head) |
| DFlash config | block_size 8, mask_token_id 248070, target_layer_ids `[5,19,33,47,61]` (5×5120 concat → `fc` 25600→5120 → `hidden_norm`) |
| DFlash2 config | conv_kernel_size 2, conv_group_size 16, selector_rank 256, selector_top_k 16 |
| Target | `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` (identity frozen in the closed campaign), gfx1151 resident session |

Full safetensors inventory (layer 0 shown; layers 0–4 identical):
`fc.weight [5120,25600]`, `hidden_norm`, `norm`,
`layers.N.{input_layernorm, post_attention_layernorm}`,
`layers.N.self_attn.{q,k,v,o}_proj` (+`q_norm`,`k_norm` [128]),
`layers.N.mlp.{gate,up,down}_proj`,
`layers.N.attention_conv.{base_kernel [2,2,5120], kernel_projection [1280,5120]}`,
`layers.N.mlp_conv.{...}` (same shapes),
`candidate_selector.{predecessor_codebook [248320,256], successor_codebook
[248320,256], hidden_projection [256,5120]}`.

### 1.2 Drafter dataflow per cycle (reference semantics)

From `dflash/model.py::dflash_generate`:

- Draft input embedding: **target** input embeddings of the block tokens
  (mask token at each draft position), optionally `input_embedding_scale`
  (absent here → 1.0).
- Context K/V: each layer projects `k/v` from the fc'd target hidden for all
  context rows **every cycle** plus the noise rows for the block; our DFlash1
  drafter already caches projected/rotated context rows across cycles
  (Phase A/B/C in [`DFLASH.md`](DFLASH.md)) — reuse that discipline.
- Positions span context + block; attention mask is the 2048 bidirectional
  sliding window (non-causal for the noise rows; context rows are always
  visible predecessors).
- Unary logits for top-16 come from the target output head applied to drafter
  hidden rows (8 rows × Q6 output head on the GGUF target — the native
  rows>1 output-head path exists for MTP B2/B3).
- Selector walk output fills `block[1..7]` draft tokens; verify is the
  existing greedy chain verify (T=0): accepted = prefix of the selected path
  matching target argmax; bonus token from the first rejected row.

## 2. Gap analysis vs current tree

| Area | Current state | Gap for DFlash2 on Qwen3.8 GGUF |
| --- | --- | --- |
| Metadata/loading | `hipengine/loading/dflash.py` parses `DFlashDraftModel` (DFlash1) HF configs; validates qwen-drafter tensor sets | New `DFlash2DraftModel` architecture, new config keys (conv/selector), new tensors (20 conv parameter pairs + 3 selector tensors), per-head q/k norms on a drafter |
| Hidden taps | GGUF target session exposes a layer-output capture map (built for Laguna/MTP); Qwen3.8 MTP used last-layer hidden only | Capture taps at 5 intermediate layers `[5,19,33,47,61]` of the 64-layer Qwen3.8 GGUF session, row-aligned, for prefill and for each verify cycle's accepted rows (compaction on partial accept) |
| Drafter kernels | `hip_gfx1100/speculative/dflash_drafter.hip` implements DFlash1 qwen drafter math (RMSNorm, RoPE, projections, context caching, add+rmsnorm fusion) used by the Laguna GGUF lane | (a) register/select per backend properly (see §5); (b) two-tap grouped dynamic conv kernels (prepare + finish); (c) q/k head-norm attention shape for 32Q/8KV/D128/window 2048; (d) selector scoring kernel: top-16 gather + `A(a)⊙H(h_t)·B(b)` over 16 candidates × 8 positions + greedy walk; (e) fc over 5×5120 tap concat |
| Output head | Native rows>1 quantized output-head path exists (MTP B1–B3 exact) | Reuse for 8-row draft logits (top-16 needed, not argmax: extend to top-K) |
| Verifier/accept/commit | Native GGUF rows path exact at B1–B3 (MTP); DFlash chain verify + `dflash_accept_chain_i32`/commit proven on other lanes | Extend the native rows path to B=7 draft rows (8-row verify incl. root); acceptance kernel already handles chain topology |
| CPU oracle | `kernels/cpu_reference/` has DFlash1 pieces | CPU-reference oracle for grouped dynamic conv and candidate selector (RED-first) |
| Bench harness | `scripts/dflash_chain_e2e_bench.py` (HF/PARO lanes), `scripts/mtp-bench.py` modes, closed-campaign GGUF bench scripts | New GGUF-target DFlash2 chain mode; same-session AR, exact MTP B3, and DFlash2 B7/B8 rows on the full mtpbench category suite |
| gfx1100 | DFlash speculative kernels physically live in the `hip_gfx1100` package; no gfx1151-native speculative module | Backend plan in §5: gfx1151 first-class, gfx1100 functional-with-correctness-gate |

## 3. Non-goals / order rules

- No tree/DDTree work: the selector emits a chain; chain verify first and
  only.
- Greedy T=0 is the binding correctness contract (exact same-session AR
  equality on the full suite). Rejection-sampled T>0 drafting is a follow-on
  and never blocks the campaign.
- No drafter quantization before the BF16 drafter path is exact and measured;
  the `incoai/Qwen3.8-27B-DFlash2-GGUF` weights are a memory comparator and a
  possible later lane, not an initial deliverable.
- gfx1100 receives **correctness and smoke only** (KL/top-1 gates + one
  rocprofv3 kernel-trace + one bench smoke row); no tuning iterations are run
  on W7900 for this campaign.
- All speed claims follow `docs/BENCHMARK.md` anti-gaming: full mtpbench
  category suite + heldouts, same-session AR denominator, no prompt
  conditioning.

## 4. Milestones

Each milestone is one or more atomic units with worklog entries; gates follow
`docs/TESTING.md` / `docs/EXECUTION-PROFILES.md`.

### D0 — Metadata, lineage, and CPU oracle (no GPU)

1. Extend `hipengine/loading/dflash.py`: `DFlash2DraftModel` config parser
   (registered, not branched by string anywhere else), tensor requirements
   for the conv/selector tensors, `candidate_selector.*` normalization
   (codebooks load as plain 2-D BF16 weights), per-head q/k norm tensors.
2. Extend `scripts/dflash_validate_artifacts.py` (or a GGUF-target variant)
   to validate the DFlash2 drafter against the Qwen3.8-27B Q4_K_M GGUF
   target: hidden sizes, vocab, mask token, target layer ids vs the GGUF's 64
   layers, output-head sharing.
3. Add `docs/source_lineage.json` entries for the reference files
   (`~/dflash/dflash/model.py`, `model_mlx.py` @ `07ebd93`) and port notes.
4. CPU reference in `kernels/cpu_reference/`: grouped dynamic conv
   (prepare/finish, group 16, first-position predecessor tap) and candidate
   selector (top-16, bilinear score, greedy walk) with RED tests against
   small golden fixtures derived from the reference implementation run on
   CPU (torch allowed in the *test fixture generator only*, never on the
   hot path).

**Gate:** metadata validator passes on the real snapshot; CPU oracles match
the reference implementation on fixtures; `python3 scripts/worklog.py check`.

### D1 — Drafter backbone on GGUF target (exact math, perf-naive)

Materialize BF16 drafter weights; implement the 5-layer forward with the
existing drafter kernel discipline (persistent scratch, context K/V caching,
`KVLiveSpans`-consistent draft attention) plus:

- target-side hidden taps at `[5,19,33,47,61]` during prefill and verify
  cycles, with accepted-row compaction;
- noise embedding from the GGUF target embedding table;
- fc + hidden_norm over the 5×5120 tap concat;
- sliding-window (2048) bidirectional mask for block rows.

A Python/numpy slow path is acceptable for the first exactness rows; the
native kernels arrive in D2. **Gate:** single-prompt greedy smoke reproduces
the reference `dflash` (transformers backend) greedy chain selection on
identical inputs (draft-top-K tables and selected path identical), finite
logits.

**D1 status (2026-08-19):** GGUF-target tap capture added to the resident
session (`DFlash2HiddenCaptureTargets`, `DFLASH2_TAP_DEPTHS=(6,20,34,48,62)`
= tap layer ids + 1, threaded through `prefill(dflash2_capture=...)` in both
bulk-prefill layer loops). `scripts/dflash2_gguf_cycle.py` drives the cycle:
prefill taps → drafter forward (mask-token noise, positions spanning context +
block, per-row projected-context cache) → top-16 selector → sequential greedy
verify via `session.step(capture_layer_output_hidden=tap_ids)` committing only
accepted rows (rejected rows are never run, so no KV rollback is needed) →
accepted-row taps extend the projected context. Smoke result: 20/20 greedy
tokens identical to pure-AR on the same session; drafts are sensible (e.g.
`' French'` for a translation prompt). Acceptance on the smoke prompt is low
(mean 1.67, ~0.095 accepted/draft) — a short-context prompt property, not a
mechanism failure; D4 measures acceptance on the full suite. Throughput is
CPU-bound by the NumPy drafter (2.44 vs 10.91 tok/s AR) — D2 native kernels
are the speed path.

### D2 — Native kernels: conv + selector + 8-row output head

1. Two-tap grouped dynamic conv kernel (one launch per conv application, or a
   fused prepare/projection variant later): strict exact/parent-parity RED
   contract vs the D0 CPU oracle; registered under
   `(hip_gfx1151, dflash2_conv, bf16, ...)` and the gfx1100 peer key.
2. Selector scoring kernel: gather A/B codebook rows (256-dim), compute
   `H(h_t)`, score 16 candidates × 8 positions in parallel, greedy-walk in
   one small kernel. Codebooks are 2×254 MB BF16 — resident, read-only,
   gathered rows only per cycle (16+16 rows × 256 per position).
3. Top-16 extension of the native quantized output-head rows path (currently
   argmax-oriented for MTP): emit top-K ids+logits for 8 draft rows without a
   full-logit host copy.

**Gates:** strict exact RED vs oracle for each kernel; `rocprofv3
--kernel-trace` smoke showing each kernel under its expected name;
KL ≤ 0.05 + top-1 ≥ 90% outer floor for the composite drafter vs reference
(production-profile tightening only if arithmetic reassociation is proposed).

### D3 — Chain verify at B=7 and wiring

- Extend the native GGUF rows verify path from B1–B3 to B=7 draft rows
  (8 rows incl. root) using the existing chain batched verifier semantics;
  accept/commit via `dflash_accept_chain_i32` + native commit (or the
  established GGUF MTP commit path — whichever the closed campaign retained).
- Wire DFlash2 as a speculative plugin choice beside MTP for the Qwen3.8 GGUF
  session (registry/plugin boundary, no engine-level `if arch ==` branches).
- Port the whole-cycle confidence gate (depth-1 `p1` from the drafter's top-16
  unary logits) as the online fallback policy, default-off until measured.

**Gate:** GPU accept summary == CPU oracle for reject/partial/full on real
weights; exact same-session AR equality on a 2-prompt smoke at B=1..7.

### D4 — Measurement campaign (gfx1151)

- Full mtpbench category suite (`code`/`general_en`/`general_ja`/`mixed_ja_en`
  + heldouts), greedy, B ∈ {3, 5, 7}: same-session AR, exact MTP B3, and
  DFlash2 rows from one clean source each; three runs; artifacts under
  `benchmarks/results/`.
- Promotion rule: DFlash2 must beat **our exact MTP B3 (23.85 tok/s,
  1.7845x)** and own AR with exact AR equality and GPU-accept==CPU on every
  row, or it stays a diagnostic with a recorded blocker. Acceptance-length
  instrumentation (per-position recall@1/@16) to quantify selector gains vs
  the DFlash1-style top-1 baseline lane (run a top-1 ablation with the
  selector disabled as a diagnostic).
- Ledger: `rocprofv3` verifier/drafter split via
  `scripts/mtp_verifier_rocprof.py`-style child profiling (never the parent
  suite harness), cycle-wall, rows/output, draft/verify seconds, GTT.

**D4 status (2026-08-19; attribution corrected 2026-08-22).** The promotion
rule is not met on the measured throughput (best point B3 8.85 tok/s = 0.66x
AR vs exact MTP B3 1.7845x), and that decision stands. Two ledger items were
**not** delivered and are re-opened as N1/N3 in the Economics section: the
`rocprofv3` verifier/drafter split was never produced — the recorded per-cycle
split is host wall-clock from the since-removed `DF2_CYCLE_DEBUG` timers — and
the DFlash2 rows were compared against an MTP B3 row measured on a different
target file (`Q4_K_S`), a different harness, and a different token budget. The
top-1 selector-disabled ablation was also not run. The corrected reading of the
MTP artifact (2.85 accepted tokens/cycle, not 3.85) puts DFlash2's acceptance
at parity with MTP, so the blocker is cost, not drafting quality.

### D5 — gfx1100 functional support

**D5 status (2026-08-19): registration complete, smoke hardware-blocked.**
The DFlash2 kernels live in `hipengine/kernels/hip_gfx1100/speculative/dflash2.{hip,py}`
and the native drafter (`hipengine/speculative/dflash2_native.py`) imports
them from the gfx1100 package — the gfx1100 peer registration and shared
source are in place. The remaining D5 deliverable is a correctness-gated
smoke on a real W7900/gfx1100 host (CPU-oracle gates, KL/top-1 floor, one
exact 2-prompt AR-equality smoke, one `rocprofv3 --kernel-trace` entry).
The measurement host in this campaign has only a gfx1151 Radeon 8060S (no
W7900), so the D5 hardware smoke is **blocked on hardware availability**, not
on code; it is a recorded, un-tuned pending row. Given DFlash2 is not
promoted (D4), D5 remains optional per the campaign decision.

- Register the DFlash2 kernels for gfx1100 (peer keys, shared source; the
  existing DFlash kernels already live in the gfx1100 package — reconcile the
  module layout per §5).
- Run on W7900: CPU-oracle gates, KL/top-1 floor, one exact 2-prompt smoke
  vs same-session AR, one `rocprofv3 --kernel-trace` entry. **No tuning**;
  record the smoke row as un-tuned gfx1100 support evidence.

### D6 — Closeout

- Rollup: `benchmarks/README.md` row + `Last updated`, changelog one-liner,
  artifact JSONs, worklog entries, `docs/DFLASH.md` status update, root
  README export via `scripts/sync_benchmark_readme.py --check`.
- Memory accounting: drafter BF16 residency (~3.6 GiB + taps + codebooks
  already included) vs the closed campaign's 15.9-GiB B3 GTT budget; state
  the GTT delta as a first-class result. If the APU GTT budget is exceeded,
  the GGUF-quantized drafter becomes a scoped follow-up, not silent scope
  creep.

**D6 status (2026-08-19): closeout complete (memory accounting + history).**
Memory accounting: DFlash2 drafter BF16 residency measured **3.584 GiB**
(layers 3.18 GiB incl. conv/MLP codebooks + `fc` 250 MiB + `candidate_selector`
245 MiB), plus ~0.15 GiB projected-context taps + drafter KV + scratch at
4K context. Against the closed campaign's B3 process GTT of **15.899 GiB**,
the full DFlash2 B3 pipeline is **~19.5 GiB (+3.6 GiB / +23%)** — inside the
Radeon 8060S unified-memory GTT capacity (the closed campaign's whole-device
4K peak was ~20 GiB with headroom), so the BF16 drafter does NOT exceed the
APU budget and the GGUF-quantized drafter is NOT a required follow-up.
Rollup: DFlash2 is a rejected/diagnostic lane (not promoted), so per
`AGENTS.md` it gets no `benchmarks/README.md` scoreboard row and no
`CHANGELOG.md` retained-row one-liner; it is recorded as a diagnostic in
`benchmarks/HISTORY.md` and as artifacts under `benchmarks/results/`, with
the campaign status in this doc and `docs/DFLASH.md`. Root `README.md` is
a product page and is not touched (no retained claim). The D4 "three runs"
statistical repeat is deliberately not repeated: on this lane the result is
decisively negative (~2.7x below MTP B3 at the optimum B3, post-Q6 select)
and every row is AR-exact, so additional full-suite runs would be a wasteful
rerun per `AGENTS.md`; B7 has multiple session runs, B3/B5 one clean
full-suite run each.

**D6 amendment (2026-08-22).** The lane-scope note previously read: "the
negative is specific to Q4_K_M / 8060S economics — see the Economics
cross-lane table for the FP8-BLOCK / PRO 6000 result where DFlash2 beats
MTP3." That framing is withdrawn. The negative is specific to **this
implementation** on Q4_K_M / 8060S: DFlash2's acceptance is at parity with
MTP's (2.80 vs 2.85 tokens/cycle), and the deficit is drafter cost plus a
verify that falls off its amortized rowtile above four rows. The cross-lane
PRO 6000 datapoint is unverifiable from this host. The "no further runs
needed" judgement also no longer holds in full: the result is decisively
negative for the current code, but the comparison it rests on is
cross-target-file and cross-harness, and the `rocprofv3` verifier/drafter
split in the D4 ledger was never produced. The N1-N3 rerun in the Economics
section is the scoped repair, blocked on GPU availability — not a wasteful
repeat of an existing measurement.

## 5. Backend ownership note

All DFlash speculative kernels currently live in
`hipengine/kernels/hip_gfx1100/speculative/` and are consumed by the gfx1151
Laguna lane by direct import. Before adding DFlash2 kernels, decide (small
refactor unit, first): either (a) keep the shared-source package as the
canonical home for both backends and make compilation/registration explicit
per target arch (documented in `docs/KERNELS.md`), or (b) give gfx1151 its
own speculative module re-exporting shared sources. Do **not** add
`if backend ==` branches; keep the four-axis registry keys authoritative.
gfx1151 remains the tuned lane; gfx1100 gets functional correctness only.

## 6. Risks / open questions

1. **Intermediate-layer hidden taps on the dense GGUF target**: the capture
   map exists, but per-cycle accepted-row compaction of 5 taps through the
   48-GDN/16-full-attention mixed stack is new plumbing; D1 proves it.
2. **8-row output-head cost**: 8 rows × ~1 GiB Q6 head read per draft cycle
   is potentially the dominant drafter cost on gfx1151's ~221 GB/s; the
   native rows path mitigates, and top-16 (not full logits) keeps host
   traffic bounded. If it dominates, a draft-vocab/lm-head-sharing diagnostic
   (à la MTP M12.2) is the follow-up, gated on exactness.
3. **Acceptance transfer**: blog numbers are sampling-mode (T=1) acceptance;
   our binding greedy lane may differ. The D4 top-1-vs-selector ablation
   quantifies this before any tuning.
4. **is_causal=false / bidirectional 2048 window** in the drafter differs
   from our causal-mask drafter attention kernels; verify mask semantics
   against the reference in D1 before kernel work.
5. **Drafter GTT on APU**: +3.6 GiB over the closed-campaign budget may
   change which KV capacities fit; measure before promoting defaults.
