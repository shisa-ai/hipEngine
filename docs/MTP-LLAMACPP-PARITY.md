# GGUF MTP llama.cpp Parity Trace and Roadmap

- Date: 2026-06-29 (deferred hidden-copy rejection; device top-k40 rejection; resident top-k40 full-suite update; production verifier/full-suite update; systemic workbench update; performance-path update 2026-06-27; correctness-solved update 2026-06-26; original trace 2026-06-25)
- Branch: `mtp-gguf`
- Hardware for all runtime numbers below: **gfx1151 / AMD Radeon 8060S (Ryzen AI Max+ 395)**, not the default W7900. Numbers state their scope; the current authoritative MTP numbers are full-suite retained diagnostics, not speed rows.
- hipEngine source baseline for the current performance review: `579112c860d8191cfcdd639b0debad86252531b7`
- llama.cpp checkout used for source/runtime evidence: `6e9007ae61f4e994c27484759caac6ef2aa32b30`

## 2026-06-29 — HANDOFF: current state, per-stage gap, tried levers, how to continue

This section is the current, authoritative snapshot. The dated sections below it
are the historical record of how we got here; where they conflict with this
section, **this section wins** (several older numbers were measured with stale
tooling or a since-corrected methodology — flagged inline below).

### TL;DR

- **Correctness is solved.** Target AR first-token + 12-token greedy trace and
  strict B3 draft acceptance match llama.cpp on the merge-sort prompt.
- **AR decode (no MTP) is already FASTER than llama.cpp's AR.** Current eager
  resident path measures **~55 tok/s** (54.65 tok/s, code prompt, gfx1151, this
  session, `scripts/gguf_ar_mtp_suite.py --scope smoke`). llama.cpp's AR is
  ~47 tok/s (its 89.55 tok/s MTP ÷ its own 1.9× MTP-over-AR ratio).
- **MTP is the entire gap.** Fresh full-suite measurement on current code
  (`gguf_ar_mtp_suite.py --scope full`, resident-serial-fallback, 10 prompts):
  **AR 54.51 tok/s; MTP best is B1 at 50.18 tok/s = 0.921× AR — MTP does NOT beat
  AR at any budget** (B1 0.921× → B5 0.759×). llama.cpp's MTP is **1.9× its AR**.
  The 1.9× we chase is **speculative amortization**, not kernel throughput.
- **There is no single bandwidth-starved GEMV to fix.** Measured cold-DRAM
  (MALL-defeated): dense Q8_0 c=1 GEMV ~51–70% of peak, selected-MoE GEMV
  ~70–80%. Every kernel micro-lever (dp4a, split-K, fusion, MoE-graph, cache
  hints) is real in isolation and **flat e2e** (table below).
- **Verifier host-vs-GPU split is resolved for the current suite route.** Fresh
  GGUF serial-target rocprof (`scripts/gguf_mtp_verifier_rocprof.py`, 12
  measured target steps, post no-logits cleanup) shows **18.63 ms host wall /
  16.56 ms kernel time per target step = 89% kernel time**, ~709 launches/step.
  The retained
  `resident-serial-fallback` route is GPU/weight-streaming bound, not
  host-launch-bound.
- **New standard measurement:** `scripts/gguf_ar_mtp_suite.py` produces ONE
  apple-to-apple AR-vs-MTP artifact under an enforced config (see "How to
  continue").

### Measurement reset — what to distrust in the history below

1. **"1.9× = selected-GEMV bandwidth" is RETRACTED.** It rested on a microbench
   that reported dense Q8_0 at ~20% of peak. That was an 8× byte-count bug
   (Q8_0 T16 block spans 32 k-values, not the 256 K-quant super-block) compounded
   by the 32 MB MALL caching the looped weight buffer. Corrected
   (`scripts/gguf_q8_0_dense_bw_microbench.py`, >2×-MALL weight pool): dense Q8_0
   is ~51–70% of peak. See `docs/ROOFLINE-gfx1151.md` §6.6.
2. **The "verifier is ~50/50 host-dispatch-bound (875 launches / ~54 ms host
   floor)" diagnostic is superseded.** It predates #9 and the current suite
   route. Re-measurement on current code with
   `scripts/gguf_mtp_verifier_rocprof.py` shows the retained
   `resident-serial-fallback` target verifier is GPU-bound after the no-logits
   cleanup: 18.63 ms host wall / 16.56 ms kernel time per target step (89%
   kernel share), ~709 launches/step. The pre-cleanup call-site profile was
   18.99 ms host / 16.68 ms kernel with unused full-logits D2H.
3. **The `--true-ar-baseline-json` apple-to-apple path is BROKEN.** Since #8
   retired the HIP decode graph, the production AR path emits `decode_path:
   eager_step`, but `gguf_mtp_category_bench.py`'s `TRUE_AR_PRODUCTION_TIMING_REQUIRED`
   (and a parallel speed-claim contract + tests) still demand the retired
   `graph_replay`. So that attach rejects every current AR baseline. The new
   suite sidesteps it (computes the ratio itself); the contracts need a proper
   eager-path fix — tracked in `docs/REFACTOR.md`.

### Per-stage gap vs llama.cpp (AR + MTP pipeline)

Numbers marked (S) are stale single-prompt diagnostics that need re-measurement
on current code; (M) are current-session measurements.

| Pipeline stage | hipEngine | llama.cpp | Gap | Status |
| --- | --- | --- | --- | --- |
| Target AR decode (c=1) | (M) ~18.2 ms/tok (~55 tok/s) | ~21 ms/tok (~47 tok/s) | **hipEngine faster** | kernels near-peak; not the problem |
| — dense Q8_0 GEMV (47% of decode) | (M) 51–70% of peak BW | — | small | not starved (was mis-measured as 20%) |
| — selected-MoE GEMV (26% of decode) | (M) 70–80% of peak BW | — | small | already amortized at rows>1 |
| — lm-head Q6_K (10% of decode) | (M) ~1.8 ms/tok | — | ? | once/token; not yet attacked |
| MTP draft (resident NextN, c=1×B) | (M) ~3.3 ms/depth (B3) | folded in graph | ~2× | device-resident + device-chained (#3) |
| MTP target verify (current resident-serial route) | (M) 18.63 ms host / 16.56 ms kernel per target step; ~709 launches/step | folded into ~9 ms 4-token fused graph | structural | GPU-bound; launch collapse alone is not the first lever on current route |
| Partial-accept rollback (B5) | (M) replay-forward dominates; LM-head skip landed (#4) | n/a | — | replay forward is the same GPU wall |
| **Net MTP throughput (full suite)** | **B1 50.2 / B3 45.4 / B5 41.4 tok/s (0.921× → 0.759× AR)** | **~89.6 tok/s (1.9× AR)** | **2–2.5×** | **amortization gap = the whole story** |

### Everything we tried — expected vs actual

| Lever | Hypothesis / expected | Actual measured | Verdict |
| --- | --- | --- | --- |
| dp4a q8_1+sudot4, selected MoE | 2.6× isolated kernel | flat e2e (BW already saturated) | diagnostic only |
| dp4a dense Q8_0 attention | faster verify | 1.2× isolated, flat e2e | not promoted |
| split-K dense Q8_0 (c=1) | more MLP → more BW | **0.74× (negative)** | rejected |
| non-temporal weight loads (c=1) | +14% via cache-bypass | +14% isolated, **flat/worse e2e** | not promoted, reverted |
| MoE-FFN HIP graph (launch cut) | fewer launches | −0.84% e2e (slight regress) | not promoted |
| dense small-B rowtile (verify) | 3× microbench at B=4 | flat e2e | kept (kernel-level win) |
| device-chain resident draft (#3) | cut per-depth host sync | bit-exact, flat e2e | kept default-off (clean arch) |
| partial-accept LM-head skip (#4) | cut discardable replay work | **+3.5% B5, bit-exact** | **kept, default-on** |
| serial verifier no-logits cleanup | remove unused full-logits D2H | **+0.7% B1 full-suite, acceptance unchanged** | **kept, default-on** |
| deferred serial hidden-seed D2H copies | avoid copying intermediate verifier hidden rows that production route does not consume | full-suite flat/noise: B1 **50.18 → 50.19 tok/s**, ratio **0.9206 → 0.9202x AR** | rejected/reverted |
| resident top-k40 draft route | avoid full legacy draft fallback for root top-k40 | **+2.9% B1 full-suite, acceptance unchanged** | **kept, default-on** |
| one-block device top-k40 | avoid resident root-K40 host logits readback + NumPy top-k | correctness passed, but smoke B3 **45.58 → 24.74 tok/s** at identical acceptance | rejected/reverted; serial K40 merge dominates |
| dispatch-resolve cache (#9) | ~15 µs/launch host | landed | kept |
| X8 selected-down repack (Q5/Q6) | sidecar-free dp4a layout | mixed; ≤ default B3 | diagnostic |
| T16 Q4/Q5 selected dp4a variants | faster MoE GEMV | 1.04–1.10× iso, flat/regress B3 | diagnostic gates |
| 32k draft vocab cap | ~5 ms/cycle draft | prompt-sensitive | diagnostic |
| adaptive AR fallback after zero-accept | avoid catastrophic block replay | robust full-suite route | **kept (production selector)** |
| HIP graph capture of verify | collapse the ~875 launches | blocked: 3rd-relaunch GDN state corruption | blocked (see WORKLOG 2026-06-28) |

Pattern: **every GPU/kernel/launch micro-lever is real in isolation and flat at
e2e.** The retained e2e wins are route/amortization cleanups (#4 LM-head skip,
serial no-logits, resident top-k40, adaptive fallback), not raw kernel
micro-optimization. That is the signal to stop optimizing kernels and work the
amortization.

### Decode-wall composition (rocprof, current code, c=1, this session)

`scripts/gguf_decode_rocprof.py`: dense_q8_0_gemv **47%**, selected-MoE GEMV
**26%**, lm-head Q6_K **10%**, GDN linear-attn **6%**, router **4%**, rmsnorm/rope
**3%**, rest <2%. Both dominant GEMV families are near-peak BW, so this wall is
mostly irreducible weight streaming — consistent with AR already beating
llama.cpp's AR.

### The new validation suite (`scripts/gguf_ar_mtp_suite.py`)

One entry point, one artifact, apple-to-apple enforced:

- Pins ONE canonical decode config on both AR and MTP: `HIPENGINE_GGUF_DECODE_REPACK=1`,
  `--decode-repack --use-gemv-decode --use-wmma-prefill`, eager decode, greedy,
  `--prompt-reasoning off` forced on both sides.
- Runs the true no-MTP AR baseline (`gguf_true_ar_category_bench.py`) and the MTP
  category suite (`gguf_mtp_category_bench.py`) — reusing the validated
  measurement code — then **computes the MTP/AR ratio itself** (does not rely on
  the stale `--true-ar-baseline-json` attach).
- **Enforces** the apple-to-apple invariants and records every problem: same
  decode protocol (`timing_protocol`), same prompt-set hashes; fails loudly with
  `apple_to_apple_ok=false` otherwise.
- Emits one artifact: `shared_config`, full provenance (git commit, hardware,
  host), the AR row, per-budget MTP rows with `vs_ar_ratio`, and a `verdict`
  (`best_mtp_budget`, `best_mtp_vs_ar_ratio`, `mtp_beats_ar`).
- Scope presets: `smoke` (1 prompt / 3 cycles / B3), `partial`
  (4 prompts / 5 cycles / B1,B3,B5), `full` (all 10 prompts / 10 cycles / B1–B5).
  The MTP suite loads the model **once** and loops all (prompt × budget)
  in-process (opt-in resident-session cache + per-prompt `reset()`; bit-exact
  validated vs the per-subprocess path — identical acceptance/token metrics, 1.89×
  faster on 2 prompts). So `full` runs in ~2–3 min instead of ~40+ min of repeated
  ~50 s model loads. The AR baseline already loads once.

```bash
# Quick directional check during development (1 prompt, ~1 min after first load):
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py --scope smoke

# Authoritative real-world number before retaining any change (~3-4 min, load-once):
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
    --scope full --output benchmarks/results/<date>-ar-mtp-suite-full.json
```

### Validation protocol — run the suite for EVERY change (mandatory)

**The only number that counts is the full-suite apple-to-apple result. Microbenches
and partials routinely do NOT translate to real-world e2e.** This session is the
proof: dp4a (2.6× isolated), split-K, dense rowtile (3× at B=4), and non-temporal
loads (+14% cold-DRAM) were all real wins in isolation and **flat or negative at
e2e** (see the tried-levers table). A kernel/host/launch microbench is a hypothesis,
not a result. So:

1. **Every GGUF AR/MTP optimization is gated by `scripts/gguf_ar_mtp_suite.py`,
   not by a microbench.** A change is a "win" only if `--scope full` improves AR
   tok/s and/or the MTP `vs_ar_ratio` **without regressing acceptance**
   (`accepted_per_output`), measured against the committed baseline.
2. **Cadence:** `--scope smoke` for a fast directional read while iterating →
   `--scope full` before promoting/committing anything as a win or making it
   default. Never retain a speed claim off a microbench, a single prompt, or a
   `partial` run alone.
3. **Compare to the committed baseline:** `benchmarks/results/2026-06-29-ar-mtp-suite-full-resident-topk40.json`
   (AR 54.51 tok/s; MTP B1 0.921× → B5 0.759× AR; `mtp_beats_ar=false`). Diff the
   `verdict` + per-budget `vs_ar_ratio`/`accepted_per_output`. The suite asserts
   `apple_to_apple_ok=true` (same decode protocol + prompt-set hashes) — if it is
   false, the comparison is invalid, full stop.
4. **Record it** per the evidence policy: drop the artifact under
   `benchmarks/results/`, update `benchmarks/README.md` + `benchmarks/CHANGELOG.md`,
   and note the before→after `vs_ar_ratio` in `WORKLOG.md`. A flat/negative e2e
   result is a *retained finding* too (it tells the next person not to re-chase it).
5. **Anti-gaming:** the suite runs the full multi-prompt category suite (code /
   general_en / general_ja / mixed_ja_en), never the single merge-sort prompt, and
   the true-AR denominator comes from the **same run** under the same config. Do
   not tune to one prompt.

This is the gate that stops the recurring trap of shipping an isolated win that
disappears at e2e.

**Scope:** `gguf_ar_mtp_suite.py` covers the **GGUF Q4_K_M path only**
(`Qwen35GGUFResidentSession`). The **PARO path** (BF16 / W4-PARO safetensors) is a
separate MTP/AR codepath with its own harnesses (`qwen35_paro_bench.py` AR;
`mtp_chain_e2e_bench.py` / `mtp_verifier_economics.py` MTP) and is **not** covered
by this suite — a PARO change needs e2e validation there. See `docs/BENCHMARK.md`
"Honest native GGUF-MTP category diagnostics" for the cross-path scope note.

### How to continue (ordered, all gated by the suite)

1. **Done: verifier host-vs-GPU split is settled for current code.**
   `scripts/gguf_mtp_verifier_rocprof.py` shows the retained
   `resident-serial-fallback` target verifier is GPU-bound (18.63 ms host /
   16.56 ms kernel per target step, 89% kernel share). Do not start with a
   launch-collapse project unless a new route/profile proves host residual is
   back on the critical path.
2. **Work the GPU-bound branch: acceptance/amortization.** Raise
   accepted-tokens-per-verify so each weight-read pass yields more output tokens.
   The full-suite sweep makes the problem concrete: acceptance *rises* with
   budget (acc/out 0.48 → 0.64 from B1→B5) but tok/s *falls* (50.2 → 41.4) —
   drafting more currently costs more than the extra acceptance saves. The win is
   higher acceptance **without** more draft/verify work per output token. Work
   draft quality on the full category suite (not the single merge-sort prompt —
   anti-gaming).
3. **Only if a future profile becomes host-bound:** collapse the per-layer
   launches — unblock HIP graph capture (fix the 3rd-relaunch GDN state
   corruption) or a C-level multi-layer dispatch loop. This is llama.cpp's
   structural advantage (one fused graph ≈ 9 ms for the 4-token verify), but it is
   not where current `resident-serial-fallback` wall time goes.
4. **Make MTP actually beat AR before any retained speedup claim.** Current best
   B1 needs roughly **+8.6% relative throughput** to beat the same-run AR
   denominator. Use `--scope full`; a retained claim needs `mtp_beats_ar=true`
   on the full suite with the true-AR denominator from the same run.
5. **Fix the stale AR-baseline contracts** (`TRUE_AR_PRODUCTION_TIMING_REQUIRED`
   + speed-claim contract + tests) to the eager path so the category bench's own
   `--true-ar-baseline-json` comparison works again (REFACTOR.md).

### Don't re-chase (closed lines of work)

GEMV instruction efficiency (dp4a/rowtile), split-K, MoE-FFN graph, cache
hints, deferred hidden-seed D2H copies, and the one-block device top-k40
extension are all measured flat or negative e2e and are not the lever. The
per-kernel GEMV bandwidth is already near-peak. Kernel micro-optimization is
exhausted; the gap is amortization.

## Production verifier status (2026-06-28)

### Update 2026-06-28 (later) — graph replay retired; AR denominator corrected; bandwidth-bound

The "AR denominator blocked by graph replay token divergence" framing **below is
superseded**. The GGUF decode-graph machinery (the divergent `--graph-replay-decode`
path) was **retired** (task #8). The current no-MTP AR path is the **eager** resident
`step()` loop with `HIPENGINE_GGUF_DECODE_REPACK=1` + `--use-gemv-decode`, with no graph
on the hot path. Measured this session (35B-A3B Q4_K_M, gfx1151, prompt-12 + 32 steps,
short-context diagnostic):

| Path | tok/s | Notes |
| --- | ---: | --- |
| **Eager AR (repack + gemv-decode), current production** | **~55.1** | no graph; the ~55.5 "divergent graph AR" row below was the now-retired graph path |
| MoE-FFN graph replay (`HIPENGINE_GGUF_MOE_GRAPH`, default off) | ~54.7 | bit-exact (KL=0, 40 cap / 3800 replay / 0 reject) but **−0.84% wall** — launch-count is not the bottleneck |

**Today's decisive finding: the decode/verify wall is weight-bandwidth bound, and every
kernel-compute/launch lever is flat.** A one-model-load AR flag sweep toggling every gated
path — `RAW`/`Q4K`/`T16` selected dp4a, `FUSED_MOE_FFN`, `COMPACT_MOE_C1`, `MOE_GRAPH`,
all-dp4a — moved AR tok/s within **−0.9%..+0.0% with bit-identical tokens** (baseline 55.15).
Bandwidth arithmetic: ~1.6–1.7 GB active Q4_K weights/token at 18.1 ms/token ≈ **~90 GB/s
achieved on ~256 GB/s peak LPDDR5X ≈ ~35% of peak**; llama.cpp's 1.9× implies ~68% of peak.
**The 1.9× gap is a memory-bandwidth-efficiency gap, not compute or launch count.** dp4a
(compute), fusion (launches), and graph (launches) are therefore exhausted as levers and
not promotable (matches the prior full-B3 dp4a −0.4% e2e; the "1.31x verifier" was an
env-toggle dispatch-thrash artifact). Artifacts:
`benchmarks/results/2026-06-28-ar-flag-sweep-bandwidth-bound.json`,
`benchmarks/results/2026-06-28-moe-graph-rows1-ab.json`.

**Open denominator question (task #5, in progress):** the honest fast eager AR is ~55 tok/s,
NOT the 19.67 "exact eager" slow control quoted below. The MTP ratio must be recomputed on the
**same protocol** with this eager-repack denominator: if AR is ~55 and resident-serial MTP is
~47.6, MTP is currently **~0.86× AR (not winning)** rather than the 2.42× implied by the 19.67
denominator. Settling this same-protocol (true-AR category bench with repack + gemv-decode vs the
MTP category bench) is the #5 deliverable. Caveat: the raw (`repack=0`) eager path is currently
**broken** by the committed `ssm_out` f32-activation fusion (`a12d8c4c`) — no `(raw_gguf, f32,
bf16)` dispatch — so the exact reference must come via the T16-repack path, and a clean eager
token-trace re-validation vs the established llama.cpp reference is part of #5.

**Re-pointed next work:** (1) #10 raise the selected-expert GEMV's *achieved* bandwidth toward
peak (coalesced/vectorized Q4_K block loads, occupancy, llama.cpp `mul_mat_vec_q` RDNA3 layout)
— the actual 1.9×; (2) #4/#3 speculative amortization (cut the ~303 ms partial-accept rollback,
keep the draft chain on-device) — fewer weight-read passes per output token. Kernel-compute and
launch-count micro-optimization is closed as a line of work.

---

_Historical (superseded above):_

**Full-suite broad verifier path exists, but the production AR denominator is
currently blocked by graph replay token divergence.**

The most robust measured route is the resident GGUF MTP draft chain with serial
target graph probing and adaptive AR fallback after zero-accept cycles:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_mtp_category_bench.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --budgets 3 --cycles 5 \
  --raw-root /tmp/hipengine-gguf-mtp-parity-workbench/2026-06-28-resident-serial-fallback-category-b3-c5/category/resident-serial-fallback \
  --output benchmarks/results/2026-06-28-resident-serial-fallback-category-b3-c5-eager-ar-summary.json \
  --true-ar-baseline-json benchmarks/results/2026-06-28-true-ar-eager-b3-c5.json \
  --reuse-existing \
  --extra-arg=--prompt-reasoning --extra-arg=off \
  --extra-arg=--root-topk-accept --extra-arg=1 \
  --extra-arg=--mtp-context-replay --extra-arg=--mtp-device-kv-cache \
  --extra-arg=--target-block-verify --extra-arg=--mtp-draft-vocab-cap \
  --extra-arg=32768 --extra-arg=--resident-mtp-draft \
  --extra-arg=--adaptive-ar-fallback --extra-arg=--no-target-block-verify
```

Result on the full default 10-prompt `mtpbench-code-general-ja.jsonl` suite,
B3/C5, gfx1151:

| Route / baseline | tok/s | Ratio | accepted/output | draft accept | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Exact no-MTP eager AR | 19.67 | 1.00x exact eager control | n/a | n/a | `--no-graph-replay-decode`; token-correct, but not the production speed denominator |
| Production graph no-MTP AR | ~55.5 | invalid denominator | n/a | n/a | graph replay settings; currently token-divergent |
| Resident serial-fallback MTP | 47.62 | 2.42x exact eager / 0.858x divergent graph AR | 0.438 | 0.542 | best robust full-suite MTP route measured |
| Always-block resident MTP | 16.60 | 0.84x exact eager | 0.597 | 0.493 | partial-accept block replay is too expensive |

The exact eager artifact is useful because it emits the expected merge-sort AR
trace. It is not evidence that production AR regressed to 19.67 tok/s; it is the
slow non-graph decode path. Artifacts:
`benchmarks/results/2026-06-28-true-ar-eager-b3-c5.json` and
`benchmarks/results/2026-06-28-resident-serial-fallback-category-b3-c5-eager-ar-summary.json`.

Important caveat: the faster graph-replay true-AR baseline measured about
`55.5 tok/s`, but it is currently token-divergent from exact eager AR on the
merge-sort diagnostic. It is not a valid speed denominator until graph replay
correctness is fixed. This is a graph correctness bug/denominator issue, not a
ROCm regression and not evidence that AR is actually 19.67 tok/s in production.

Rejected verifier routes from this update:

- Always-block resident draft is not production-safe: it reaches high acceptance
  but falls to `16.60 tok/s` full-suite because every partial accept triggers
  expensive block rollback/replay.
- B5 block promotion after a full B3 serial probe failed on the merge-sort smoke:
  `38.40 tok/s`, with two B5 partial cycles costing `~137-141 ms`. Do not make
  B5 block promotion a default without a stronger predictor and rollback fix.

Next performance work is now unambiguous: fix graph replay correctness so the
fast AR path is eligible as the denominator, then continue reducing verifier
GEMV cost and improving draft acceptance. The current full-suite route is a
useful robust MTP baseline, but it is not yet faster than the production graph
AR path and remains far from llama.cpp's ~90 tok/s MTP diagnostic.

## Executive summary (2026-06-27)

**Correctness is solved. The remaining gap is GGUF quantized GEMV performance,
roughly 1.9x on the single-prompt gfx1151 diagnostic.**

| Milestone | Status |
| --- | --- |
| Target AR first-token parity | ✅ `71093` matches llama.cpp (Qwen3.5 GDN K-head broadcast fix) |
| Target AR 12-token greedy trace | ✅ identical sequence `[71093,12305,198,727,10562,17885,10620,25,1103,8,1411,1103]` |
| Strict B3 draft acceptance | ✅ `2/9` → `9/9`, and `15/15` over 5 cycles (context replay + device MTP KV) |
| F32 router/alpha/beta retention | ✅ landed (registry-dispatched mixed kernels) |

The earlier blocker — hipEngine's target autoregressive stream diverging from
llama.cpp at the first sampled token — is fixed. The root cause was Qwen3.5
linear-attention Gated-Delta-Net K-head mapping: GGML maps value head `v_head` to
key head `v_head % num_k_heads`, while hipEngine inherited grouped `v_head /
repeat`. With the interleaved mapping, target AR and strict B3 acceptance both
match llama.cpp on the merge-sort prompt.

### Performance: current numbers (single-prompt diagnostic, gfx1151)

llama.cpp B3 MTP on the same reasoning-off 12-token trace:
**`eval time = 89.55 tok/s`** (`134.01 ms / 12 tokens`), 100% strict draft
acceptance, from `/tmp/hipengine-llamacpp-mtp-cli-reasoning-off-debug.log:3813`.

hipEngine best diagnostic configs (all `15/15` strict accepts, B3/C5, merge-sort
prompt):

| Configuration | tok/s | vs AR | verify ms/cycle | draft ms/cycle | accept |
| --- | ---: | ---: | ---: | ---: | ---: |
| Block verify GEMV prefill + dense rowtile + 32k draft cap | 48.8 | 0.80x | ~61 | ~17 | 15/15 |
| Block verify GEMV prefill + 32k draft cap, pre-rowtile | 48.1 | 0.80x | ~61–66 | ~17 | 15/15 |
| One-step graph + 32k draft cap | 44.5 | 0.81x | ~72 | ~17 | 15/15 |
| One-step graph, full vocab | 42.3 | 0.77x | ~73 | ~22 | 15/15 |

Gap to llama.cpp: **~48.8 vs ~89.6 tok/s ≈ 1.8-1.9x slower**, and it is almost entirely
target verification overhead, not acceptance and not draft quality.

### Where the time goes (per B3 cycle)

| Stage | hipEngine | llama.cpp | Gap |
| --- | --- | --- | --- |
| Target verify (4 tokens) | ~64 ms (block GEMV) / ~73 ms (graph) | ~8.9 ms (`dur(g)=26.7 ms / 3 calls`) | 7–8x |
| MTP draft (3 tokens) | ~17 ms (32k cap) / ~22 ms (full vocab) | included in `dur(g)` | ~2x |
| Commit / bookkeeping | ~1.6 ms | negligible | minor |

A synchronized per-layer probe over the first B3 verifier block showed most time
inside the 30 linear-attention layers, but a later sync-free rocprof trace
narrowed the actual hot bucket: selected-expert MoE GEMV is ~54% of verifier GPU
time (`gguf_q4_k_selected_dual_prefill_out_kernel` gate+up ~36% plus
`gguf_k_selected_pack8_prefill_out_kernel` down ~18%). Dense rowtile kernels are
now default-on and are ~3x faster on their microbench share, but end-to-end is
flat because dense projections are only ~11-17% of the verifier after clean
profiling.

**dp4a POC result (2026-06-27): positive, not runtime-default.** A bounded
q8_1+sudot4 selected-dual Q4_K variant now exists as a diagnostic wrapper. At
the qwen35moe verifier shape (`x_rows=4`, `rows=32`, `experts=256`, `in=2048`,
`out=512`, gfx1151), the existing raw selected-dual kernel measured `0.946 ms`
vs q8_1 quantize+dp4a at `0.357 ms` (**2.65x**). q8_1 quantization alone was
`0.0025 ms`. Correctness vs the existing float-dequant kernel on that diagnostic
was `KL_mean=0.0031`, top-1 `1.0` for both gate/up outputs. Disassembly confirms
`v_dot4_i32_iu8` emission, and `rocprofv3 --kernel-trace` shows
`gguf_q4_k_selected_dual_q8_1_dp4a_prefill_out_kernel` averaging `~338 us` vs
`~1007 us` for `gguf_q4_k_selected_dual_prefill_out_kernel` in the same short
trace. Artifact:
`benchmarks/results/2026-06-27-hipengine-gguf-q4-k-selected-dual-dp4a-poc.json`.

**Verifier integration diagnostic (2026-06-27): exact, but not the production
hot path.** The rows>1 verifier now has a default-off
`HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A=1` path with caller-owned q8_1 workspace.
B3/C5 merge-sort smoke with the production decode-repack route stayed exact
(`15/15`) and measured `50.44 tok/s` (`50.73 tok/s` warm), but rocprof showed
no q8_1/dp4a kernels in that production trace. The active selected-MoE verifier
route is T16 decode-repack (`q4_k_t16_selected_dual_*` and
`qk_t16_selected_direct_gemv_kernel`), not the raw Q4_K fallback. With
`--no-decode-repack`, the raw fallback does launch `40` q8_1 quantize calls and
`40` `gguf_q4_k_selected_dual_q8_1_dp4a_prefill_out_kernel` calls, but that mode
is much slower overall (`35.66 tok/s`, verifier `96.2 ms`) because it disables
the production T16 materialization.

**T16 selected-dual dp4a diagnostic (2026-06-27): launches in production, but
too small to promote.** The same env gate now also has a T16 Q4_K selected-dual
q8_1+sudot4 variant for the rows>1 split gate/up path. The isolated T16
microbench at the verifier shape measured current T16 split dual `0.198 ms` vs
q8_1 quantize+dp4a `0.191 ms` (**1.04x**), with gate/up `KL_mean=9.25e-05` and
top-1 `1.0`; disassembly confirms `v_dot4_i32_iu8`. The callable fused-SiLU
T16 dp4a variant is retained as a diagnostic but is **not routed** in production
because the c1 profile regressed it. Split-only B3/C5 smoke stayed exact
(`15/15`) but remained flat (`49.31 tok/s`, warm `50.60 tok/s`). A short
production trace confirms only the row-bulk split path uses dp4a: `80`
`q4_k_t16_selected_dual_q8_1_dp4a_direct_gemv_kernel<unsigned short,false>`
calls at `141.8 us` avg plus `80` q8_1 quantize calls at `3.35 us`; c1 fused
stays on `q4_k_t16_selected_dual_silu_direct_gemv_kernel` at `62.5 us` avg. The
next material bucket is still selected-down Q5_K T16 (`851` calls, `51.6 us`
avg, `43.9 ms` in the same two-cycle trace). Artifacts:
`benchmarks/results/2026-06-27-hipengine-gguf-q4-k-t16-selected-dual-dp4a-poc.json`
and
`benchmarks/results/2026-06-27-hipengine-mtp-b3-q4k-t16-dp4a-verifier-diagnostic.json`.

**T16 selected-down Q5_K dp4a diagnostic (2026-06-27): kernel-positive, not a
runtime win.** The next bucket was ported under a new default-off broad env gate:
`HIPENGINE_GGUF_T16_SELECTED_DP4A=1`. The Q5T16 selected-down microbench at the
c1-like down shape (`rows=8`, `E=256`, `in=512`, `out=2048`, gfx1151) measured
current T16 `0.0335 ms` vs q8_1 quantize+dp4a `0.0306 ms` (**1.10x**),
`KL_mean=0.00678`, `KL_max=0.03093`, but only `0.875` top-1 on that small
synthetic fixture. `rocprofv3 --kernel-trace` confirms
`qk_t16_selected_q8_1_dp4a_direct_gemv_kernel<unsigned short>` launches, and
extracted device ISA contains `v_dot4_i32_iu8`. B3/C5 merge-sort smoke remained
exact (`15/15`) but regressed to `47.62 tok/s` (warm `48.44`), so the Q5 path is
kept diagnostic/default-off. Q6_K was not routed: a synthetic probe had
acceptable KL but only `0.75` top-1 vs the T16 float path. Artifacts:
`benchmarks/results/2026-06-27-hipengine-gguf-q5-k-t16-selected-down-dp4a-poc.json`
and
`benchmarks/results/2026-06-27-hipengine-mtp-b3-q5-t16-dp4a-verifier-diagnostic.json`.

**Raw selected-down Q5_K/Q6_K dp4a diagnostic (2026-06-27): broad raw layout
is promising, but not enough yet.** The raw no-decode-repack selected-down path
now has Q5_K and Q6_K q8_1+sudot4 variants under the default-off
`HIPENGINE_GGUF_RAW_SELECTED_DP4A=1` gate. On the selected-down microshape
(`rows=8`, `E=256`, `in=512`, `out=2048`, gfx1151), Q5_K measured raw
float-dequant `0.0916 ms` vs q8_1 quantize+dp4a `0.0395 ms` (**2.32x**),
and Q6_K measured `0.0419 ms` vs `0.0259 ms` (**1.62x**). Correctness vs the
existing float-dequant path cleared the project gate on the diagnostic:
Q5_K `KL_mean=0.00011`, top-1 `1.0`; Q6_K `KL_mean=0.00512`, top-1 `1.0`.
A cached `rocprofv3 --kernel-trace` microbench confirms
`gguf_k_selected_pack8_q8_1_dp4a_prefill_out_kernel<unsigned short,5/6>`
launches, with q8_1 quantization at `~2.1 us` average and dp4a dot kernels at
`~44.7 us` (Q5) / `~19.5 us` (Q6) in the short trace. B3/C5 raw-layout smoke
stayed exact (`15/15`) and improved no-decode-repack from `31.63 tok/s` to
`39.61 tok/s` (warm `31.86 -> 40.29`), but the production decode-repack
baseline on the same short smoke was still `51.31 tok/s` (warm `52.00`). Keep
this as a diagnostic proof that GGML-style raw q8_1 vector-dot is worth a broad
layout port; do not promote the raw env as a runtime default yet. Artifacts:
`benchmarks/results/2026-06-27-hipengine-gguf-raw-q5-q6-selected-pack8-dp4a-poc.json`,
`benchmarks/results/2026-06-27-hipengine-mtp-b3-raw-selected-dp4a-verifier-diagnostic.json`,
`benchmarks/results/2026-06-27-hipengine-mtp-b3-raw-selected-float-verifier-baseline.json`,
and
`benchmarks/results/2026-06-27-hipengine-mtp-b3-default-verifier-baseline-for-raw-dp4a.json`.

**X8 selected-down production-layout slice (2026-06-27): correct and
sidecar-free, not default yet.** The first broad-port slice now has a
byte-neutral X8 replacement layout for selected-down Q5_K/Q6_K experts:
`tiles[expert, out_pack8, k_block, 8 * block_bytes]`. It preserves the raw GGUF
block bytes while giving the production decode-repack materializer the same
eight-output q8_1+sudot4 dot shape as the raw sidecar diagnostic. It is opt-in
via `HIPENGINE_GGUF_SELECTED_X8_REPACK=1`; gate/up remains on the current T16
Q4_K path. On the selected-down microshape (`rows=8`, `E=256`, `in=512`,
`out=2048`, gfx1151), X8 matched raw dp4a outputs exactly and cleared the
quality gate versus production T16 float, but the timing is mixed: Q5_K
production T16 `0.03352 ms` vs X8 q8_1 quantize+dot `0.03864 ms` (**0.87x**),
while Q6_K production T16 `0.03206 ms` vs X8 q8_1 quantize+dot `0.02602 ms`
(**1.23x**). A cached `rocprofv3 --kernel-trace` microbench confirms
`gguf_x8_selected_q8_1_dp4a_gemv_kernel<unsigned short,5/6>` launches; the
short trace averaged `~37.2 us` for Q5 X8, `~22.9 us` for Q6 X8, and `~1.9 us`
for q8_1 quantization. B3/C5 merge-sort smoke with X8 materialization stayed
exact (`15/15`) but was slower than the same-tree default control:
`49.74 tok/s` (`50.65` warm) vs default `51.43 tok/s` (`53.09` warm). Keep X8
default-off until the Q5 path beats T16 or a quant-selective production route
improves the same B3/full-suite protocol. Artifacts:
`benchmarks/results/2026-06-27-hipengine-gguf-x8-selected-down-dp4a-poc.json`,
`benchmarks/results/2026-06-27-hipengine-mtp-b3-x8-selected-down-verifier-diagnostic.json`,
and
`benchmarks/results/2026-06-27-hipengine-mtp-b3-default-verifier-control-for-x8.json`.

**X8 Q5 tuning / quant-selective route (2026-06-28): useful diagnostic, still
not a default.** Reducing X8 selected-down launches from 128 to 64 threads helps
the synthetic small-B shape: Q5_K X8 dot moved to `0.03026 ms` and q8_1
quantize+dot to `0.03378 ms` versus production T16 `0.03364 ms` (roughly
break-even), while Q6_K X8 quantize+dot moved to `0.02014 ms` versus T16
`0.03304 ms` (**1.64x**). The materializer now accepts
`HIPENGINE_GGUF_SELECTED_X8_REPACK=q5|q6|both`; `=1` remains `both`. This lets
diagnostics route by quant family, but the B3 verifier still does not improve:
full X8 with the 64-thread body measured `49.08 tok/s` (`49.41` warm), q6-only
X8 measured `50.32 tok/s` (`51.07` warm), and same-tree default T16 measured
`51.77 tok/s` (`52.56` warm), all exact `15/15`. Keep the selector opt-in and
do not promote X8 until the production verifier, not just the microshape, wins.
Artifacts:
`benchmarks/results/2026-06-28-hipengine-gguf-x8-selected-down-t64-dp4a-poc.json`,
`benchmarks/results/2026-06-28-hipengine-mtp-b3-x8-t64-selected-down-verifier-diagnostic.json`,
`benchmarks/results/2026-06-28-hipengine-mtp-b3-x8-q6-only-selected-down-verifier-diagnostic.json`,
and
`benchmarks/results/2026-06-28-hipengine-mtp-b3-default-verifier-control-for-x8-t64.json`.

**Systemic E2E/per-piece workbench (2026-06-28): landed.**
`scripts/gguf_mtp_parity_workbench.py` is now the standard local gate for the
GGML-style broad port. It runs the same B3/C5 E2E command shape across named
runtime candidates (`default`, `x8-q5`, `x8-q6`, `x8-both`, `t16-dp4a`,
`q4-t16-dp4a`, `raw-dp4a`), runs the selected-MoE per-piece microbenches
(`Q4_K` gate/up, raw `Q5_K/Q6_K` down, X8 `Q5_K/Q6_K` down), and can optionally
run rocprof bucket summaries and category-suite diagnostics. The first smoke
validated the wrapper on gfx1151 with one default B3 cycle plus low-iteration
piece runs:
`PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_mtp_parity_workbench.py --tag 2026-06-28-gguf-mtp-parity-workbench-smoke --raw-root /tmp/hipengine-gguf-mtp-parity-workbench --output benchmarks/results/2026-06-28-hipengine-gguf-mtp-parity-workbench-smoke.json --stages e2e,pieces --candidates default --cycles 1 --draft-n-max 3 --piece-iters 4 --piece-warmup 1`.
That smoke measured default E2E `49.3 tok/s`, AR baseline `60.62 tok/s`, exact
`3/3` accepts for the one cycle. Treat the piece timings in this smoke as
harness validation only because `--piece-iters 4` is intentionally noisy; use the
full default `--piece-iters 80`/`--cycles 5` workbench or a higher-iteration run
before making kernel decisions. Artifact:
`benchmarks/results/2026-06-28-hipengine-gguf-mtp-parity-workbench-smoke.json`.
The first full B3/C5 workbench matrix then showed why same-protocol repeats are
required before routing decisions: `default,x8-q6,x8-both` measured `46.19`,
`49.74`, and `50.49 tok/s`, but the reversed-order E2E repeat measured
`x8-both=48.07 tok/s` and `default=51.33 tok/s`, all exact `15/15`. This keeps
X8 diagnostic/default-off and confirms the workbench should be used as a
multi-run gate, not a single-run promotion oracle. Artifacts:
`benchmarks/results/2026-06-28-hipengine-gguf-mtp-parity-workbench-b3-current.json`
and
`benchmarks/results/2026-06-28-hipengine-gguf-mtp-parity-workbench-b3-repeat.json`.

### Next steps, ordered by impact

1. **Do not promote the current straight dp4a diagnostics.** Raw Q4_K/Q5_K/Q6_K
   q8_1+sudot4 is strong in isolation and improves the raw no-decode-repack
   verifier, but production B3 still uses T16 and remains faster. The first
   production-compatible X8 selected-down slice removes the raw sidecar and the
   64-thread body helps the isolated Q5/Q6 microshape, but full-X8 and q6-only
   X8 still trail default B3. T16 Q4 split is only `1.04x` in its small
   row-bulk bucket, T16 Q5 selected-down is only `1.10x` in isolation while
   regressing B3, and raw selected-down still trails default decode-repack at
   the verifier level. Keep
   `HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A` and
   `HIPENGINE_GGUF_T16_SELECTED_DP4A` / `HIPENGINE_GGUF_RAW_SELECTED_DP4A` /
   `HIPENGINE_GGUF_SELECTED_X8_REPACK` as diagnostic gates only.
2. **Broad port target: match GGML's q8_1/x4 vector-dot layout, gated through
   the workbench.** The next implementation should make the production verifier consume a GGML-like
   q8_1 activation plus x4 packed K-quant dot path for the selected-MoE and dense
   GGUF GEMVs, instead of continuing one-off T16 ports. The raw Q4/Q5/Q6 and X8
   results prove the instruction path and a sidecar-free materialization route;
   the missing piece is making the Q5 selected-down body and the remaining hot
   GGUF GEMVs faster than T16 on the same production verifier protocol. Use
   `scripts/gguf_mtp_parity_workbench.py --stages e2e,pieces,rocprof` for local
   acceptance of each broad-port slice before promoting any default.
3. **Extend only proven GGUF GEMVs into defaults.** Carry q8_1+sudot4 into
   dense/raw Q4_K/Q5_K/Q6_K/Q8_0 GEMVs when the local shape clears the quality
   gate and improves the same B3/full-suite protocol. The existing small-B
   rowtile dense kernels are complementary and should be combined with dp4a where
   rows 2..8 share an activation tile.
4. **MTP draft resident path.** Keep all MTP intermediates (embeddings,
   projections, KV, hidden seeds) on device across draft depths; only D2H the
   final top-1 token ID. Chain the B draft steps in one call instead of B separate
   `run_draft()` calls with full alloc/copy per depth. Validate the 32k draft
   vocab cap on the full suite before promoting (saved ~5 ms/cycle here but is
   prompt-sensitive).
5. **Partial-accept rollback is catastrophic (~303 ms for a B5 partial cycle).**
   Track which linear-attention buffers were modified and copy-on-write only
   those, or replay only the accepted prefix instead of full target decodes. Or
   just keep B3 (100% accept on this prompt) and skip B5 until rollback is cheap.
6. **Full-suite validation before any retained speed claim.** Everything above is
   single-prompt merge-sort diagnostics. Need the full
   `mtpbench-code-general-ja.jsonl` category suite, category heldouts, a true
   no-MTP AR baseline from the same protocol, and the draft vocab cap validated
   for non-regressive acceptance across prompts.
7. **Longer-term: match llama.cpp's architecture.** Both target verification and
   MTP drafting run through one optimized GGML compute graph in a single process
   with shared weight memory. C-level dispatch or HIP graph capture remains a
   later layer, after the hot GEMV kernels stop wasting instruction issue on
   float dequant-then-FMA.

The historical trace evidence below is retained as the record of how correctness
parity was reached.

## Source evidence: what llama.cpp does

All llama.cpp source links below point to commit
`6e9007ae61f4e994c27484759caac6ef2aa32b30`.

### 1. Qwen35MoE MTP graph

The Qwen35MoE MTP graph is built as a one-layer decoder graph:
[`src/models/qwen35moe.cpp#L550-L736`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/src/models/qwen35moe.cpp#L550-L736).
Important details:

- It requires one NextN/MTP block.
- It chooses `nextn.embed_tokens` when present, otherwise `model.tok_embd`.
- It takes a separate hidden-state input tensor named `mtp_h_input`.
- It calls `build_attn_inp_kv()`, so the MTP block has its own draft-context K/V state.
- It computes:
  1. `h_norm = RMSNorm(h_input, nextn.hnorm)`
  2. `e_norm = RMSNorm(token_embedding, nextn.enorm)`
  3. `concat = [e_norm, h_norm]`
  4. `eh_proj`
  5. attention + gated output projection + residual
  6. MoE/shared-expert FFN + residual
  7. shared-head norm, then LM head fallback to `model.output`.

This graph shape matches our Python/GPU wrapper at a high level.  The gap is in
**state lifecycle and numerical/runtime parity**, not the obvious concat order or
which head/embedding tensors are chosen.

### 1b. GGUF GEMV inner loop

The current performance-path delta is below the graph shape: llama.cpp/GGML
quantizes activations to q8_1 and runs quantized weight x q8_1 dot products,
while hipEngine's raw GGUF kernels dequantize weights to float and then FMA.
Local source evidence in `/home/lhl/llama.cpp/llama.cpp-hip/ggml/src`:

- `ggml-common.h` defines `block_q8_1` as 32 signed int8 activation quants plus
  `d` and `s` fp16 metadata.
- `ggml-cuda/mmvq.cu` dispatches `GGML_TYPE_Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0`
  through `vec_dot_*_q8_1` functions and allocates/quantizes `src1_q8_1` before
  `mul_mat_vec_q_switch_type(...)`.
- `ggml-cuda/vecdotq.cuh` uses repeated `ggml_cuda_dp4a(...)` calls in those
  vector-dot functions.
- `ggml-cuda/common.cuh` maps ROCm `ggml_cuda_dp4a(...)` to
  `__builtin_amdgcn_sudot4(...)` on AMD targets.

hipEngine's corresponding hot raw kernels are in
`hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_gemv.hip` and
`gguf_k_gemv.hip`; they currently unpack scales/mins/nibbles and accumulate in
float. This is why the bounded POC targets q8_1 activation quantization plus
sudot4 inside the raw selected Q4_K dual gate+up kernel before any broad port.

### 2. MTP state maintained by llama.cpp

The MTP speculative implementation stores per-sequence state in
[`common/speculative.cpp#L816-L918`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L816-L918):

- `pending_h`: hidden row used to seed the next MTP draft.
- `verify_h`: hidden rows captured from the target verifier batch.
- `verify_h_rows`: how many verifier hidden rows are available.
- `last_n_drafted`: last draft length, used for recurrent/rollback bookkeeping.

This is the critical lifecycle we only partially approximate today.

### 3. `process()` mirrors target verifier rows into the draft/MTP context

llama.cpp's MTP `process()` is in
[`common/speculative.cpp#L955-L1045`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L955-L1045).
The important behavior:

- It copies target `h_nextn` rows from the target context.
- It builds an MTP batch with token/hidden pairs.
- It calls `llama_decode(ctx_dft, batch)` on the draft/MTP context.
- That decode advances the MTP graph and its K/V state, not just a single isolated
  row.
- It stashes verifier hidden rows in `verify_h` and refreshes `pending_h`.

This is what our old no-context path lacked.  Our new `--mtp-device-kv-cache`
implements a first B1 approximation of the K/V portion, but not the full
llama.cpp process lifecycle or B>1 rollback/transactional semantics.

### 4. `draft()` seeds from `pending_h`, samples from `ctx_dft`, and chains `h_nextn`

llama.cpp's MTP `draft()` is in
[`common/speculative.cpp#L1048-L1168`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L1048-L1168):

- It adds the last accepted token `dp.id_last` at `dp.n_past`.
- It overwrites the draft batch embedding with `pending_h`.
- It calls `llama_decode(ctx_dft, batch)`.
- It samples a draft token from the draft/MTP logits.
- It reads `llama_get_embeddings_nextn_ith(ctx_dft, i_batch)` and uses that as
  the hidden seed for the next draft step.
- It repeats up to `n_max`, respecting `p_min`.

This is where llama.cpp gets an actual predictive draft chain.  hipEngine's
`run_draft()` also chains `return_hidden_seed`, but our state before/around that
chain has not matched llama.cpp's `process()`/draft context yet.

### 5. `accept()` chooses the verifier hidden row for the next seed

llama.cpp's MTP `accept()` is in
[`common/speculative.cpp#L1171-L1184`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L1171-L1184):

- It chooses `i_h = min(n_accepted, n_rows - 1)`.
- It copies `verify_h[i_h]` into `pending_h`.

This matches our conceptual `pending_hidden_row_index = accepted` logic, but we
must still validate that our captured row is numerically the same row at the same
point in the graph.

### 6. Runtime stats are reported by common speculative stats

The aggregate counters are printed by
[`common/speculative.cpp#L2079-L2103`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L2079-L2103):

- `#gen drafts`
- `#acc drafts`
- `#gen tokens`
- `#acc tokens`
- begin/draft/accept durations

These counters are the cleanest runtime evidence we have without editing the
read-only llama.cpp checkout.

## Source evidence: what hipEngine currently does

All hipEngine source links below point to commit
`98df03ddd00ae682c07e302721343040373e1b55`.

### 1. Acceptance accounting

hipEngine's benchmark implements llama.cpp-style strict acceptance in
[`scripts/gguf_mtp_bench.py#L259-L297`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/scripts/gguf_mtp_bench.py#L259-L297):

- The target samples `[last_token] + accepted_draft_prefix`.
- The first mismatch emits a corrective target token.
- Visible output tokens are accepted draft targets plus the corrective token.

The benchmark also has root/sibling top-K acceptance diagnostics; those are useful
for measuring whether the target is somewhere in the draft distribution, but they
are **not** evidence that the draft chain matches llama.cpp.

### 2. Device-resident MTP KV cache, default off

The new opt-in dense device cache is in
[`hipengine/kernels/hip_gfx1100/speculative/mtp_nextn.py#L636-L760`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/hipengine/kernels/hip_gfx1100/speculative/mtp_nextn.py#L636-L760),
with the device-to-device write and dense attention read in
[`mtp_nextn.py#L975-L1002`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/hipengine/kernels/hip_gfx1100/speculative/mtp_nextn.py#L975-L1002).

Accepted-row cheap commit is handled via `kv_write_only` in
[`mtp_nextn.py#L880-L930`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/hipengine/kernels/hip_gfx1100/speculative/mtp_nextn.py#L880-L930),
and the benchmark uses it in
[`scripts/gguf_mtp_bench.py#L1126-L1155`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/scripts/gguf_mtp_bench.py#L1126-L1155).

The fixture proving sequential cache writes match two-row dense attention is
[`tests/test_mtp_dense_device_kv_cache.py#L1-L120`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/tests/test_mtp_dense_device_kv_cache.py#L1-L120).

This is useful infrastructure, but it remains default-off because it has not yet
improved same-suite speed/acceptance.

## Runtime trace commands and artifacts

### llama.cpp CLI MTP debug trace

Command:

```bash
/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-cli \
  -m /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0.0 \
  -p 'Write a Python function that implements merge sort:' \
  -n 12 \
  -ngl 99 \
  --spec-draft-ngl 99 \
  --temp 0 \
  --no-warmup \
  --no-display-prompt \
  --single-turn \
  --simple-io \
  --log-file /tmp/hipengine-llamacpp-mtp-cli-debug.log \
  --log-verbosity 5
```

Artifact: `/tmp/hipengine-llamacpp-mtp-cli-debug.log`.

Caveat: `llama-cli --no-conversation` is not supported by this binary.  The
working CLI path is server/chat-style.  The debug trace had `task.n_tokens = 19`.
A `--no-jinja` probe used `task.n_tokens = 17` and still had 100% draft
acceptance, but generation timing collapsed to 0.88 tok/s, so it is not used for
performance comparison.

Aggregate llama.cpp result for the debug trace:

```text
draft acceptance = 1.00000 (8 accepted / 8 generated)
statistics draft-mtp: #calls(b,g,a) = 1 3 3,
  #gen drafts = 3, #acc drafts = 3,
  #gen tokens = 8, #acc tokens = 8,
  dur(b,g,a) = 0.004, 26.710, 0.001 ms
```

Per-draft-call table parsed from the debug log:

| call | history size before draft | drafted | accepted | top-1 draft IDs | corrective / sampled token | new token count |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 19 | 3 | 3 | `[579, 264, 7047]` | 1817 | 23 |
| 2 | 23 | 3 | 3 | `[25, 271, 16]` | 13 | 27 |
| 3 | 27 | 2 | 2 | `[220, 2972, 15771]` | 15771 | 30 |

Interpretation:

- `accepted == drafted` for every MTP call in the trace.
- The verifier call commits `accepted_draft_tokens + 1` visible tokens: 4, 4, and
  3 respectively.
- Visible output / verifier call is therefore `11 / 3 = 3.67`.
- Accepted draft tokens / verifier call is `8 / 3 = 2.67`.

### Target-AR parity trace (new primary blocker)

The cleanest apples-to-apples prompt mode is llama.cpp `--reasoning off`, which
renders the same 21-token text as hipEngine's retained `reasoning='off'` prompt:

```text
<|im_start|>user
Write a Python function that implements merge sort:<|im_end|>
<|im_start|>assistant
<think>

</think>

```

llama.cpp verbose prompt evidence:

```text
common_sampler_init prefill tail:
  248045 <|im_start|>, 74455 assistant, 198 \n,
  248068 <think>, 271 \n\n, 248069 </think>, 271 \n\n
task.n_tokens = 21
next token: 71093 '```'
```

Command/artifact:

```bash
/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-cli \
  -m /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0.0 \
  -p 'Write a Python function that implements merge sort:' \
  -n 1 \
  -ngl 99 \
  --spec-draft-ngl 99 \
  --temp 0 \
  --no-warmup \
  --no-display-prompt \
  --single-turn \
  --simple-io \
  --reasoning off \
  --verbose-prompt \
  --log-file /tmp/hipengine-llamacpp-reasoning-off-verbose-prompt.log \
  --log-verbosity 5
```

hipEngine target traces for the same 21-token prompt:

| hipEngine mode | First token after prefill | Next verifier target | Notes |
| --- | --- | --- | --- |
| retained default (`WMMA prefill + GEMV + graph`) | `760` = `The` | `198` = `\n` | `/tmp/hipengine-mtp-target-parity-off-default.json` |
| no WMMA prefill | `248069` = `</think>` | `271, 16` = `\n\n1` | `/tmp/hipengine-mtp-target-parity-off-no_wmma.json` |
| no WMMA/GEMV/graph/decode-repack | `248069` = `</think>` | `271, 16` = `\n\n1` | `/tmp/hipengine-mtp-target-parity-off-no_fast.json` |
| true token-serial `prefill(..., use_bulk=False)` probe | `1919` = `This` | n/a | top-1 from direct session probe |

None match llama.cpp's `71093` code-fence first token.  Therefore the first
confirmed divergence is **target AR prefill/decode/logit parity**, before MTP
draft acceptance.  The MTP acceptance gap is downstream of this target mismatch.

### hipEngine strict B3 trace

Command:

```bash
python3 scripts/gguf_mtp_bench.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --prompt "Write a Python function that implements merge sort:" \
  --cycles 3 \
  --draft-n-max 3 \
  --root-topk-accept 1 \
  --output /tmp/hipengine-mtp-b3-strict-trace.json
```

Artifact: `/tmp/hipengine-mtp-b3-strict-trace.json`.

Caveat: the hipEngine benchmark applies the Qwen chat prompt wrapper used by its
GGUF harness and reported `Prompt tokens: 21`; this is close but not byte-for-byte
identical to the llama.cpp CLI trace (`19` chat/server tokens).  The strict B3
numbers are still useful because the acceptance gap is large and consistent with
full-suite behavior.

Metrics:

```text
accept_per_draft     = 0.2222
accepted_per_output  = 0.4000
visible/cycle        = 1.6667
tokens_per_sec       = 33.38
speedup_vs_ar_visible= 0.598x
total_accepted       = 2 / 9 draft tokens
```

Per-cycle table:

| cycle | accepted / drafted | target samples | draft IDs | target rank in draft top-10 | visible output | target verify ms | MTP draft ms |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| 0 | 0/3 | `[198]` | `[803, 328, 760]` | `[None]` | 1 | 17.94 | 20.31 |
| 1 | 0/3 | `[17]` | `[760, 21397, 25]` | `[2]` | 1 | 18.00 | 19.51 |
| 2 | 2/3 | `[15, 15, 15]` | `[15, 15, 248046]` | `[1, 1, 2]` | 3 | 53.60 | 20.42 |

Interpretation:

- hipEngine's MTP top-1 is often wrong even when the target is near the top of
  the distribution (`target_rank_in_draft_top10 = 2` in cycles 1 and 2).
- This is exactly why root-top40 raised `accepted_per_output` while strict
  `draft_acceptance` stayed extremely low: the target is often in the top-K but
  not the actual draft token.
- B3 strict verification currently commits only `5/3 = 1.67` visible tokens per
  verifier call, far below llama.cpp's `3.67` in the debug trace.

### hipEngine retained/default and device-KV smoke context

Retained root-top40 B1 smoke artifact: `/tmp/hipengine-mtp-with-attn-smoke.json`

```text
accept_per_draft    = 0.0225
accepted_per_output = 0.4737
visible/cycle       = 1.9
tokens_per_sec      = 46.6
total_accepted      = 9 / 400 candidate-count denominator
```

Device-KV B1 smoke artifact:
`/tmp/hipengine-mtp-device-kv-smoke-fastcommit.json`

```text
accept_per_draft    = 0.0187
accepted_per_output = 0.4286
visible/cycle       = 1.75
tokens_per_sec      = 43.68
total_accepted      = 3 / 160 candidate-count denominator
KV rows             = 7 / 12
commit cost         = ~1.2-1.9 ms per accepted-row KV write
```

The device-KV path is much faster than prior host replay/prefix diagnostics, but
it did not reproduce llama.cpp's high B3 acceptance and remains default-off.

## What llama.cpp is doing that hipEngine is not yet doing

### 0. Target AR parity before speculation

llama.cpp and hipEngine must first agree on the target model's greedy token after
the prompt.  They currently do not.  For the same reasoning-off prompt tail,
llama.cpp picks code fence token `71093`; hipEngine picks `760`, `248069`, or
`1919` depending on prefill path.  This points to a target runtime issue, not an
MTP model-quality issue.

Likely places to investigate in order:

1. Prompt/output-row scheduling: llama.cpp decodes the 21-token prompt as a 17-row
   cached prefix plus a 4-row tail; hipEngine bulk/serial row selection may be
   sampling the wrong hidden row.
2. Qwen3.6 hybrid recurrent/Gated Delta Net state: fastpath toggles change the
   first sampled token, which means recurrent/prefill state is affecting target
   semantics.
3. LM-head/argmax parity: direct token-serial hipEngine top-10 does not contain
   llama.cpp's code fence token, so verify output logits against llama.cpp after
   the prompt.
4. Logit processors/biases: llama.cpp biases EOG tokens to `-inf`; confirm
   hipEngine has equivalent generation-time biasing.  This is unlikely to explain
   `71093` vs `760`, but should be checked.

Until this stage matches, MTP token acceptance is not the primary bug.

### A. Full draft-context lifecycle, not just K/V rows

llama.cpp's `process()` decodes verifier rows through `ctx_dft` and updates all
relevant draft-model state.  For Qwen35MoE MTP this primarily means attention K/V,
but it also means the exact graph scheduling, output IDs, and hidden-row selection
are controlled by the same decode path as `draft()`.

hipEngine now has device K/V row writes, but still drives MTP from a Python wrapper
that repeatedly uploads/downloads intermediates and manually chooses which rows to
commit.  It does not yet have the same transactional draft context abstraction.

**Roadmap item:** add an in-tree `GGUFMTPDraftContext` owning device K/V, position,
pending hidden row, accepted verifier rows, and rollback/commit state.  The
benchmark should call this object rather than open-coding row bookkeeping.

### B. B>1 transactional semantics

llama.cpp B3 drafts can be generated, verified, accepted, and rolled forward while
preserving draft context.  hipEngine's `--mtp-device-kv-cache` intentionally
rejects `--draft-n-max != 1` today because we do not yet have safe rollback for
unaccepted draft rows.

**Roadmap item:** implement draft transaction:

1. Save `kv_len_before_draft`.
2. Append draft rows while generating B tokens.
3. Verify target batch.
4. Roll back unaccepted draft rows.
5. Commit accepted target rows and the corrective pending hidden row exactly like
   llama.cpp's `accept()`.

### C. Numeric parity of MTP logits has not been proven

The largest unexplained delta is that llama.cpp's top-1 MTP tokens are accepted
in the debug trace, while hipEngine's top-1 tokens often miss even when the target
is rank 2.  That could be due to:

- hidden seed captured at the wrong point,
- RoPE position/context count mismatch,
- missing or stale MTP K/V context,
- output ID / row selection mismatch,
- quantized GEMV/layout differences in attention, FFN, or shared head,
- sampler/logit post-processing differences.

**Roadmap item:** create a one-step parity harness that records, for the same
prompt/token position:

- token ID entering MTP,
- `pending_h` checksum/norm,
- K/V cache length,
- MTP top-10 logits/tokens,
- `h_nextn` checksum/norm,
- accepted prefix length.

Without editing the read-only llama.cpp checkout, we can only get aggregate and
some debug candidate logs.  For true tensor parity we need either a temporary
instrumented llama.cpp worktree/copy or a local patch that is not committed to the
reference repo.

### D. hipEngine wrapper overhead is still high

Even when B1 device K/V is active, hipEngine draft time is ~8.5 ms/cycle on the
smoke.  The source-level issue is that the correctness-first Python wrapper still
allocates/copies many intermediates.  The WORKLOG follow-up already identified:

- remove Q/gate D2H split,
- avoid Q6_K temporary H2D uploads in attention,
- keep more MTP intermediates resident,
- move from Python orchestration to one or a few persistent launch wrappers.

**Roadmap item:** after numeric parity, port MTP attention+FFN+head into a real
resident path.  Do not optimize the wrong math first.

### E. Root-topK is not a substitute for draft quality

Root-top40 showed the target is frequently *near* the draft distribution, but the
speculative algorithm commits actual draft tokens.  llama.cpp's debug trace has
true top-1 acceptance.  hipEngine's root-topK acceptance is therefore a diagnostic
for rank quality, not a path to B3/B5 break-even.

**Roadmap item:** keep root-topK as diagnostic only.  Promote only changes that
raise strict top-1 chain acceptance and committed tokens/verifier call.

## What we can adopt from llama.cpp

| llama.cpp behavior | Adopt in hipEngine? | Notes |
| --- | --- | --- |
| `pending_h` / `verify_h` lifecycle | Yes | We already use a similar concept; needs parity checksum tests. |
| Draft context with persistent MTP K/V | Yes | Started with default-off B1 dense device cache; must become transactional and resident. |
| `process()` verifier-row mirroring | Yes | Need a resident `process_verifier_rows()` equivalent. |
| B>1 rollback/commit semantics | Yes | Required before meaningful MTP speedups. |
| `p_min` early stop | Yes, diagnostic first | We already have `--draft-p-min`; tune after top-1 parity. |
| Backend sampling | Maybe | llama.cpp logs backend TOP_K support missing on ROCm in this run; hipEngine top-k is already explicit. |
| Chat/server prompt handling | No as-is | hipEngine benchmark prompt protocol must stay fixed and anti-gaming compliant. |
| Loading full model twice for MTP | No | Must keep hipEngine torch-free/lean and use in-model MTP weights only. |

## Prioritized roadmap to effective MTP

### Phase 0 — target AR parity on one prompt

1. Reproduce llama.cpp's 21-token reasoning-off prompt exactly.
2. Add a hipEngine target-only trace that emits:
   - prompt token IDs,
   - chunking/prefill schedule,
   - final hidden-row index sampled,
   - top-20 target logits after prefill,
   - first generated token.
3. Instrument a temporary llama.cpp copy or use verbose prompt + a small tensor
   dump to get the same target top-20 logits.
4. Fix target parity before changing MTP acceptance logic.

Success criterion: hipEngine target prefill chooses `71093` for the documented
reasoning-off prompt, matching llama.cpp, under the narrowest correctness-first
path.  Then optimize back toward the retained fast path.

**2026-06-25 status:** achieved for both correctness-first and retained fast
paths.  The blocker was Qwen3.5 linear-attention GDN K-head broadcast semantics:
llama.cpp/GGML maps value head `v_head` to key head `v_head % num_k_heads`, while
hipEngine inherited the grouped `v_head / repeat` mapping.  After switching the
GDN decode/prefill kernels and CPU replay oracles to the interleaved mapping, the
same 21-token reasoning-off prompt has `initial_prev_token=71093`.  A follow-up
12-token greedy target trace also matches llama.cpp exactly:
`[71093, 12305, 198, 727, 10562, 17885, 10620, 25, 1103, 8, 1411, 1103]`
(decoded as a Python code fence followed by `def merge_sort(arr: list) -> list`).
The single-prompt B3 smoke improves from
the prior `2/9` accepted drafts / `5` visible output tokens to `7/9` accepted
drafts / `10` visible output tokens.

Evidence command:

```bash
python3 scripts/gguf_mtp_bench.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --prompt "Write a Python function that implements merge sort:" \
  --prompt-reasoning off --cycles 3 --draft-n-max 3 --root-topk-accept 1 \
  --output /tmp/hipengine-mtp-target-parity-final-c3.json
```

### Phase 1 — exact MTP trace parity on one prompt

1. Add a hipEngine trace mode that emits per-step JSON:
   - prompt token IDs,
   - previous token,
   - position,
   - pending hidden norm/checksum,
   - MTP KV length,
   - MTP top-10 IDs/logits/probs,
   - target samples,
   - accepted prefix length,
   - committed output tokens.
2. Produce a temporary instrumented llama.cpp copy or local patch that emits the
   same fields from `common_speculative_impl_draft_mtp`.
3. Compare the first divergence.
4. Fix math/state mismatches before doing more performance work.

Success criterion: on the same prompt/token positions, hipEngine and llama.cpp
produce the same MTP top-1/top-K tokens for at least the first several draft
steps, or we can explain every difference.

### Phase 2 — B3 transactional device KV

1. Promote the B1 device cache into a draft-context object.
2. Add rollback/commit around B>1 draft rows.
3. Validate with a CPU/synthetic fixture and then a GGUF smoke.
4. Run strict B3, no root-topK, same prompt.

Success criterion: strict B3 `accepted_draft_tokens / generated_draft_tokens`
substantially improves over the old `2/9 = 22.2%` smoke and approaches the
llama.cpp debug trace on the same prompt.

**2026-06-25 status:** achieved for the diagnostic llama.cpp-lifecycle path.  The
missing piece after target parity was the draft model context lifecycle: replay
the shifted prompt rows into a device-resident MTP KV cache, keep the cycle-start
row, roll back rejected speculative rows, and commit accepted rows with
verifier-derived target hidden seeds.  With `--mtp-context-replay`,
`--mtp-device-kv-cache`, `--draft-n-max 3`, and `--root-topk-accept 1`, the same
single-prompt smoke reaches `9/9 = 100%` accepted drafts and `12` visible output
tokens over three verifier calls.

### Phase 3 — full-suite strict acceptance before speed claims

Run `mtpbench-code-general-ja.jsonl` in strict mode and record:

- accepted draft tokens / verifier call,
- visible output tokens / verifier call,
- strict draft acceptance,
- rank histogram for target token in MTP top-K,
- raw tok/s.

Success criterion: committed tokens/verifier call rises enough that speed work is
worthwhile.  If strict acceptance remains low, return to Phase 1.

### Phase 4 — performance optimization only after parity

Once strict acceptance is credible:

- fuse resident MTP attention/FFN/head launches,
- eliminate host-side intermediate copies,
- pre-upload/cache Q6_K weights and scratch buffers,
- replace sequential target verification with a rollback-safe block verifier,
- profile verifier MoE grouping/budgeting to reduce `eta`,
- revisit B2/B3/B5 economics.

**2026-06-25 status:** first draft-side performance wins landed, and a
rollback-safe target continuation block verifier now exists, but performance
parity is still blocked by verifier kernel shape.  Batching accepted-row MTP KV
commit into one `kv_write_only` pass improved the corrected B3 merge-sort smoke
from `41.7` to `42.3 tok/s` (`15/15` strict accepts over five cycles).  A
hot-token draft LM-head cap of `32768` improved the same one-step-graph smoke to
`44.5 tok/s` with unchanged `15/15`, but it is prompt-sensitive and remains
diagnostic until full-suite validation.  The new `--target-block-verify` path
snapshots linear recurrent state, runs the target over `[prev]+drafts` as a
continuation block, records target IDs + FP32 hidden seeds, and restores/replays
the consumed prefix on partial accepts.  Its first version was exact (`15/15`) but
slow on the B3+32k smoke (`37.8 tok/s`, verifier `~90 ms/cycle`) because the
selected/WMMA prefill kernels are the wrong shape for tiny B.  The verifier now
defaults to the GEMV prefill fallback internally (`--no-target-block-wmma-prefill`)
while leaving normal prompt prefill WMMA enabled; that lifts the same B3+32k
smoke to `48.1 tok/s` with unchanged `15/15` and verifier `~61-66 ms/cycle`
(except variance on late cycles).  B5 remains unattractive because a partial
rollback cycle costs hundreds of ms in the generic restore/replay path.

**2026-06-26 profiling — the verifier is WORK-bound, not launch-bound.**  Two
single-process diagnostics overturn the earlier "captured HIP graph / C-level
dispatch loop" hypothesis for the #1 verifier fix:

- *Row-scaling* (`verify_rowscale.py`): `verify_target_block` GEMV wall-time is
  ~flat per row (`24 ms/row`, fit `23 ms + 24 ms·rows`); rows=128 costs **26× rows=4**.
  If launch-overhead-bound, rows=4 and rows=128 would cost nearly the same
  (~420 launches either way).  WMMA per-row falls `31.5 → 8.86 ms/row` (amortizes
  but high fixed cost at B=4).
- *Per-family* (`verify_family.py`, rows=4 GEMV): dense Q4_K projections
  (`launch_gguf_linear`) **44%**, MoE selected-expert GEMV **28%**, GDN 6%,
  router 7%, Q6_K lm-head sample 5%.  72% is quantized matmuls run per-row.
  Cross-check: `launch_gguf_linear` ≈ 89 µs/call vs ~20 µs B=4 weight-bandwidth
  floor ⇒ **~4× over floor**, i.e. the Q4_K weight is reloaded once per row.

Initial root cause: at rows>1 with WMMA off, `launch_gguf_linear` uses the decode-shaped
`dense_gemv:prefill_out` = `dense_gemv_out_kernel`
(`hipengine/kernels/hip_gfx1100/linear/dense_gemv.hip:122`), grid `(out_col, row)`
— one block per (column,row), so the column is re-dequantized per row.  This is
exactly llama.cpp's advantage: GGML batches the 4 rows into one weight-load-
amortized matmul (~8.9 ms total ≈ 2.2 ms/row).

**2026-06-27 update: dense rowtile landed, but the bottleneck moved.**
The small-B rowtile idea is implemented for raw Q4_K and raw K-family
Q8_0/Q5_K/Q6_K dense GEMVs, bit-exact against the per-row kernels, and default-on
for rows 2..8 when WMMA is off. Microbench speedups at B=4 are ~3x on dense
projection shapes, and a B3 verifier smoke with the 32k draft cap stayed exact at
`48.77 tok/s` (`15/15`, verifier ~61 ms/cycle), flat vs the pre-rowtile `48.1`
within run noise.

A clean sync-free rocprof pass corrected the family attribution: selected-expert
MoE GEMV is the real top bucket, not dense projection row reload. The hot verifier
GPU-time shares are:

| Kernel family | Share |
| --- | ---: |
| `gguf_q4_k_selected_dual_prefill_out_kernel` (MoE gate+up) | ~36% |
| `gguf_k_selected_pack8_prefill_out_kernel` (MoE down, Q5_K) | ~18% |
| residual per-row dense `gguf_k_prefill_out_kernel` | ~17% |
| dense rowtile `gguf_k_prefill_out_rowtile_kernel` | ~11% |
| GDN recurrent/rmsnorm-gate | ~8% |
| Q6_K lm-head pack8 | ~6% |

Two cheap MoE ideas are now ruled out:

- Row amortization/group-by-expert does not apply at B=4. A microbench with
  qwen35moe shapes showed 32 same-expert rows at `0.567 ms` vs 32 distinct
  experts at `0.882 ms`; B=4/top_k=8 selects ~30 distinct experts, so there is
  essentially no expert overlap to reuse.
- `expert_sidecar`/pack8 gate+up for the verifier is ~15x slower (`103.4 ms`
  raw vs `1588.4 ms` sidecar) because per-layer H2D movement dominates.

**Current #1 verifier task:** selected-MoE remains the verifier bottleneck, but
the straightforward T16 dp4a ports are not retainable defaults. The raw
selected-dual Q4_K POC is positive (`0.946 ms -> 0.357 ms` at the qwen35moe
verifier shape), but production B3 uses T16 decode-repack. The T16 Q4_K split
gate/up port launches under `HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A=1` and cuts
the row-bulk split kernel in the short trace (`~172 us -> ~142 us`), but B3
stays flat. The T16 Q5_K selected-down port launches under
`HIPENGINE_GGUF_T16_SELECTED_DP4A=1` and is `1.10x` faster in isolation, but B3
regresses (`47.62 tok/s`, warm `48.44`) and the c1 synthetic top-1 is marginal
(`0.875`). Next work should either adapt the layout closer to GGML's q8_1/x4
vector-dot path or find a selected-down reduction/layout change that improves
B3 without top-1 drift; do not keep porting Q6/dense dp4a as a default path
without that gate.

Captured-graph/C-loop work is deprioritized to a later launch-overhead layer
after GEMV instruction efficiency improves. Cheaper partial-accept rollback
remains important for B5, but it does not address the full-accept B3 verifier
hot path.

**2026-06-28 correction — the verifier is ~50/50 HOST-dispatch-bound; the
deprioritization above was wrong.**  A warm `verify_target_block` (rows=4)
issues **875 kernel launches** (~22/layer × 40 layers); the pure host launch
dispatch is **~54 ms** (~52% of the wall).  A dp4a A/B under `rocprofv3` shows
dp4a genuinely cuts GPU kernel time −35% (MoE dual `1256→400 ms`, 3.14×) yet the
E2E wall stays flat/worse because dp4a *adds* launches (per-layer q8_1 quantize)
and the host-dispatch floor dominates.  So GEMV instruction efficiency (dp4a,
rowtile) cannot move E2E until the ~54 ms host-launch floor is removed.  The
**primary lever is collapsing the 875 launches** — HIP graph capture (gated by
the 3rd-relaunch GDN corruption, see WORKLOG 2026-06-28) or a C-level multi-layer
dispatch loop — exactly the original plan.  dp4a/rowtile are complementary GPU
wins that materialize *after* the launch floor is cut.  llama.cpp runs the whole
4-token verifier as one fused GGML graph (~9 ms); the 875-launch host floor is
the core of the gap.

Success criterion: same-protocol full-suite row improves all three: raw weighted
decode tok/s, accepted/output, and strict draft acceptance.

## Bottom line

llama.cpp is not just using a wider candidate set.  It is running a real target
and MTP draft context with verifier-row processing, persistent draft K/V state,
hidden-row handoff, and B>1 accept/rollback semantics.  In the short debug trace
it commits `3.67` visible tokens per verifier call with `100%` strict draft
acceptance.

hipEngine now matches llama.cpp's documented reasoning-off target AR trace and,
with the llama.cpp-style context replay + device MTP KV lifecycle, reaches strict
B3 `9/9` (and `15/15` over five cycles) on the merge-sort smoke. Correctness
parity is therefore solved.

The remaining gap is performance: ~48.8 vs ~89.6 tok/s (~1.8-1.9x) on gfx1151.
The latest evidence says the q8_1+sudot4 recipe is valid, but the layout
decision matters more than the intrinsic itself: raw Q4_K selected-dual is
`~2.65x` faster in isolation, raw Q5_K/Q6_K selected-down is `~2.32x`/`~1.62x`
faster including q8_1 quantization, and the raw B3 verifier improves
`31.63 -> 39.61 tok/s`; meanwhile T16 Q4_K split gate/up is only `~1.04x`, T16
Q5_K selected-down is only `~1.10x`, and the production decode-repack smoke is
still faster at `51.31 tok/s`. Dense rowtile is already landed and retained as a
kernel-level win, but selected MoE dominates. Next: broad-port a GGML-like
q8_1/x4 vector-dot layout into the production GGUF verifier path, then promote
only the same-protocol B3/full-suite non-regressive pieces. Graph/C loop work,
resident MTP draft consolidation, and rollback improvements remain on the
roadmap after the GEMV instruction path is de-risked. These remain single-prompt
diagnostics, not benchmark rows.
