---
status: current
owns: Qwen3.8-27B speculative-decode optimization plan: which Muse Glimmer wins transfer, and the DSpine/EAGLE-4 evaluation case against the shipping MTP and the rejected DFlash2.
---
# Qwen3.8-27B speculative optimization and method-comparison campaign

This campaign records (1) which optimizations landed in the Muse Glimmer
long-context track apply to Qwen3.8-27B, and (2) the case for evaluating two
external drafters — **DSpine** and **EAGLE-4** — against the methods already in
the tree (native **MTP**, shipping; **DFlash2**, evaluated and rejected). It is a
plan and evidence register, not a result: no DSpine/EAGLE-4 code is implemented
here, and the external speedups below are vendor measurements on different
hardware, not hipEngine rows.

## 1. Baseline and prior decision

Qwen3.8-27B `Q4_K_S` on Radeon 8060S / `gfx1151` (closed campaign
[`QWEN38-27B-GFX1151-CAMPAIGN.md`](QWEN38-27B-GFX1151-CAMPAIGN.md)):

- True AR ≈ **13.1 tok/s** (512/1K/4K).
- Native **MTP B3 = 23.85 tok/s (1.78× own AR)** — the shipping speculative
  route, AR-exact.
- **DFlash2 rejected** ([`QWEN38-27B-DFLASH2-CAMPAIGN.md`](QWEN38-27B-DFLASH2-CAMPAIGN.md)):
  best point B3 **8.85 tok/s (0.66× AR)**, ~2.7× below MTP B3. The corrected
  attribution is **cost, not drafting quality** — DFlash2 B3 acceptance (2.80
  tokens/cycle) is at parity with MTP B3 (2.85). The deficit is:
  - drafter forward + select ≈ **96 ms/cycle** (MTP proposal ≈ 2.4 ms), running
    at roughly a fifth of achievable bandwidth for its 3.584 GiB residency;
  - a verify costing 166 ms for 4 rows vs MTP's 111 ms for 3.85 rows, on a
    different code path with DFlash2-only 5-layer tap capture in the timed region;
  - a rowtile admission cliff at four rows (`_PACK8_ROWTILE_MAX_ROWS = 4`) that
    makes the 8-row verify re-read the full weights per row (≈8 weight sweeps).

Any method that beats MTP on this lane must attack those three costs, not just
acceptance. Qwen3.8-27B is a **hybrid with linear-attention (Mamba-style) state**,
which constrains drafter/verify design (see §3 porting notes).

## 2. Applicable optimizations from the Muse Glimmer track

The Muse Glimmer long-context track (branch `muse-glimmer`, merged to `main`)
landed seven units. Status here means **implemented in hipEngine**; applicability
notes whether Qwen3.8-27B gets it for free, needs wiring, or needs a port. The
Muse Glimmer numbers are gfx1151 measurements on that model and are **not**
predictions for Qwen3.8-27B.

| Unit | What landed | Status | Applies to Qwen3.8-27B |
| --- | --- | --- | --- |
| Host sampler penalties | Incremental `history_counts` + vectorized repetition/presence/frequency penalties in `hipengine/generation/sampling.py` | shared path | **Free.** Any sampled request with active penalties; O(history)→flat. DFlash/MTP greedy decode is unaffected. |
| Q6 rowtile metadata hoist | Hoisted `d`/scales + wide record loads in `gguf_q6_k_t16_gemv.hip` (bit-identical, −8.5% on the Muse Glimmer Q6 verify) | shared kernel | **Free where used.** Applies to any Q6_K T16 exact-verify rowtile path; Qwen3.8-27B's verify uses Q5 col8 rowtile, so check the quant mix before claiming a win. |
| Draft head single-wave | Draft `lm_head` routed to the single-wave Q5 production rowtile (draft proposes, owes no exactness) | Muse Glimmer wiring; kernel shared | **Port the routing.** Directly attacks the DFlash2 drafter-head cost; applies to a DFlash2/DSpine draft head of matching quant. |
| Non-causal WMMA block-flash | `muse_glimmer_draft_attention` (4:1 GQA, d128, windowed, non-causal) | new kernel | **Port/adapt.** Fits a parallel DFlash-family drafter (DSpine) with the same geometry; not the MTP path. |
| Verify graph capture | HIP-graph-captured exact block verify (740→1 launch, bit-identical) | Muse Glimmer wiring | **Port the pattern.** Attacks the DFlash2 verify host-dispatch cost; wall benefit is host-speed dependent. |
| Large-chunk inject | Draft-KV injection in 1024-row WMMA chunks (injection owes no exactness) | Muse Glimmer wiring | **Port the pattern.** Attacks DFlash2's 5-layer tap-capture/inject cost in the timed prefill region. |
| Context-aware budget (finding) | Budget optimum is **acceptance-driven, not context-driven**; default 4 robust | measurement, no code | **Methodology.** Re-run the identical-KV budget sweep per method; do not assume the optimum shifts with context. |

Records: worklog entries `20261108T150626…-3f84bc` (Q6 hoist),
`20261108T162214…-12f246` (budget), `20261108T173703…-f4516d` (draft head),
`20261108T184336…-3a38a2` (verify graph), `20261108T185439…-c0a41a` (sampler),
`20261108T192501…-c7a63e` (inject), `20261108T201319…-392318` (draft attention),
all under `worklog/entries/`.

**Read-across to the DFlash2 cost gaps:** the draft head, draft attention, inject,
and verify-graph units target exactly the drafter-forward and verify costs that
sank DFlash2. They are a plausible path to re-evaluating a DFlash-family drafter
on this lane — but that is a hypothesis to measure, not a result. The rowtile
admission cliff (`_PACK8_ROWTILE_MAX_ROWS = 4`) is a separate DFlash2-specific
blocker the Muse Glimmer work did not address.

## 3. Candidate methods: DSpine and EAGLE-4

Both are external drafters with public Qwen3.8-27B artifacts and vendor
benchmarks. Neither is implemented in hipEngine. External numbers are on
NVIDIA B300 / AMD MI355X class hardware and are **not comparable** to the
gfx1151 MTP row without a matched local measurement.

### 3.1 DSpine (Tencent)

- **Mechanism.** A parallel drafter with *adjacent causal injection*: at every
  draft layer, a gated injection writes each predecessor's predicted feature
  into its successor, so the causal chain unfolds over network depth while all
  positions update in parallel. A unified transfer space built from the target's
  output embeddings plus layer-wise output-embedding supervision. Fused kernels
  and a transition cache run it in SGLang. It is a DFlash-family successor
  (parallel block drafting), not a feature-autoregression method.
- **Evidence (vendor).** On **Qwen3-8B**, seven-benchmark mean acceptance length
  at T=0: DFlash 5.92, DFlash2 7.55, **DSpine 7.83** (README table); the paper
  reports +27.8% acceptance vs DFlash and +23.3% SGLang serving throughput.
  These are Qwen3-8B figures, not Qwen3.8-27B.
- **Qwen3.8-27B artifact.** `tencent/Qwen3.8-27B-DSpine-b16` (released 2026-11),
  a five-layer drafter with an 8192-token sliding window. The Qwen3.8-27B
  serving recipe uses **DSpark-based dynamic verification** and **ReplaySSM** for
  the linear-attention state — i.e. it explicitly handles the hybrid recurrence
  this model has. A trade press item reports "up to 3.84× agent throughput"
  (secondary, unverified here).
- **License.** Apache 2.0 (weights and code) — compatible with vendoring/ports.
- **Porting notes.** Needs the adjacent-injection drafter forward, the
  output-embedding transfer space, and a ReplaySSM-equivalent for the linear
  state. The non-causal WMMA block-flash and inject units (§2) are the closest
  existing hipEngine pieces. Acceptance gain over DFlash2 is modest on Qwen3-8B
  (7.83 vs 7.55); the win case rests on cost, like DFlash2.

### 3.2 EAGLE-4 (Eagle AI)

- **Mechanism.** Feature-level (second-to-top-layer) autoregression in the
  EAGLE-1/2/3 lineage, with two additions that **independently mirror this
  campaign's Muse Glimmer findings**:
  - **EAGLE-Harness** profiles draft/target cost across batch size, context
    length and speculative-token count, caches lookup tables, and picks tree
    depth/width/verify budget from measured cost + draft confidence — the same
    conclusion as our budget finding (§2): the optimum is cost/acceptance-driven,
    not a fixed or context-only rule.
  - **EAGLE Efficient LM Head**: full-vocabulary logits were 65.4% of a 574 µs
    draft forward; a low-rank head retrieves candidates and the full-rank head
    scores the shortlist (−43% to −56% draft-forward latency). This is the same
    lever as our draft-head unit, taken further.
  - The drafter is a **single Transformer layer** (0.44B params vs DFlash2's
    1.66B / DSpark's 1.65B across five layers, excluding fusion/embeddings/head).
- **Evidence (vendor, matched Qwen3.8-27B).** SPEED-Bench qualitative, 4× B300,
  TP4, T=0, same engine, MegaKernel disabled (paper Table 2 / blog):

  | Method | 1 request tok/s | ×vanilla | 512 req aggregate tok/s | ×vanilla |
  | --- | ---: | ---: | ---: | ---: |
  | Vanilla | 240.5 | 1.00× | 21,203 | 1.00× |
  | DSpark | 618.6 | 2.57× | 14,229 | 0.67× |
  | DFlash2 | 672.6 | 2.80× | 16,641 | 0.78× |
  | **EAGLE-4** | **863.8** | **3.59×** | **28,340** | **1.34×** |

  EAGLE-4 is 1.28× DFlash2 at one request and 1.70× at 512. On a single **AMD
  MI355X** (ROCm port exists), SGLang+EAGLE-4 reports 3.49× vanilla at one
  request, 2.30× at 16 concurrent, 1.70× at 32. Long-context (TokenSpeed, 4×B300):
  759–822 tok/s decode at 32K–250K.
- **Artifact.** `EAGLE-Corp/Qwen3.8-27B-EAGLE4`; runs via `eagle-harness` on
  SGLang/vLLM/TokenSpeed.
- **License.** **Source-available, not open source.** Commercial use of a
  competing offering needs a separate license; other commercial use is bounded
  by a US$20M aggregate-revenue threshold. **This restricts vendoring into
  hipEngine** — treat EAGLE-4 as a comparison baseline and an idea source, not a
  code dependency, unless licensing is resolved.
- **Porting notes.** The single-layer feature drafter, dynamic tree verify, and
  efficient LM head are all rebuildable ideas (the LM head and cost-aware budget
  already have hipEngine analogues). The MI355X ROCm port shows AMD feasibility
  but is GFX950, not RDNA3 (gfx1151/gfx1100).

### 3.3 Comparison framing

| Method | In hipEngine? | Draft style | Layers | License | Qwen3.8-27B evidence |
| --- | --- | --- | --- | --- | --- |
| MTP (native) | shipping, AR-exact | multi-token prediction | 1 | repo | local: 23.85 tok/s, 1.78× AR (gfx1151) |
| DFlash2 | evaluated, rejected | parallel block | 5 | repo | local: 8.85 tok/s, 0.66× AR; vendor 2.80× vanilla (B300) |
| DSpine | not implemented | parallel + adjacent causal injection | 5 | Apache 2.0 | vendor acceptance > DFlash2 (Qwen3-8B); 27B weights released |
| EAGLE-4 | not implemented | feature autoregression + dynamic tree | 1 | source-available | vendor 3.59× vanilla, 1.28× DFlash2 (B300); MI355X ROCm port |

The vendor tables agree on ordering at low concurrency (EAGLE-4 > DFlash2 >
DSpark > vanilla) and show DFlash2/DSpark **losing to vanilla at high concurrency**
while EAGLE-4 stays above — consistent with EAGLE-Harness adapting the budget to
load. None of these external rows is a hipEngine measurement; the local MTP row
is the only number measured on our hardware in this doc.

## 4. Evaluation plan and decision gates

Proposed order (each a separate unit, profile-first per
[`OPTIMIZATION.md`](../OPTIMIZATION.md)):

1. **Transfer the free/shared wins** (§2 host sampler, Q6 hoist where the quant
   mix uses it) and confirm no regression on the Qwen3.8-27B MTP suite.
2. **Re-evaluate a DFlash-family drafter with the cost toolkit** (draft head,
   non-causal WMMA attention, inject, verify graph) against the DFlash2 cost
   gaps. Gate: must close the 2.7× cost deficit vs MTP B3 on a matched local
   suite, AR-exact, acceptance within band. Resolve the `_PACK8_ROWTILE_MAX_ROWS`
   cliff first or bound the verify chain below it.
3. **DSpine port trial** (Apache 2.0): implement adjacent-injection drafting +
   ReplaySSM-equivalent for the linear state; measure acceptance and tok/s vs
   MTP and the re-evaluated DFlash2 on the identical-KV mtpbench suite.
4. **EAGLE-4 as a baseline only** until licensing is resolved: reproduce its
   cost-aware budget and efficient-LM-head ideas as native hipEngine units
   (already partly present), and compare against the vendor numbers qualitatively,
   not as a vendored dependency.

Binding gates for any promotion: AR-exact generated IDs (or the calibrated
production numerical envelope per [`EXECUTION-PROFILES.md`](../EXECUTION-PROFILES.md)),
acceptance within a pre-registered band, a true no-speculation AR baseline from
the same protocol, and full multi-prompt mtpbench coverage (anti-gaming,
[`BENCHMARK.md`](../BENCHMARK.md)). External vendor speedups are context, not
evidence for a hipEngine claim.

## 5. Measured progress (ZBook relative baselines, gfx1151, Q4_K_M)

Internal relative rows (thermally throttled ZBook; NOT the official publishing
lane — final validation on the stable-clock Desktop). Full mtpbench, greedy,
3 reps, AR-exact:

| Method | decode tok/s | vs AR | acceptance | task |
| --- | ---: | ---: | ---: | --- |
| true AR | 10.85 | 1.00x | — | #13 |
| native MTP B3 | 18.05 | 1.665x | 0.603 | #13 |
| DFlash2 B3 (baseline) | 9.50 | 0.85x | 2.32 tok/cyc | #13 |
| DFlash2 B3 + QKV-WMMA | 9.72 | 0.86x | 2.32 tok/cyc | #15 |
| DSpine (native, GPU refine) | 4.30 | 0.38x | 0.214 | #17–18 |
| EAGLE-4-style chain n1 (native) | 8.92 | 0.78x | 0.445 | #19–21 |

**2026-11-09 refresh (matched relative-baseline, current code):** a fresh
apples-to-apples run supersedes the stale DSpine/EAGLE-4 rows above. MTP B3
**1.63x** AR (14.71 tok/s) > EAGLE-4 chain n_draft=2 **1.31x** (13.90) > AR >
DSpine verify-prefix-5 + GPU refine **0.99x** (11.13) > DFlash2 B3 **0.88x**
(9.86); all AR-exact. EAGLE-4 now beats AR after the P1 graph-verify + device
fc-projection + device head/draft work; DSpine reached ~AR parity after the
prefix-verify fix. Ratios are vs each method's own back-to-back AR (throttled
ZBook, absolute tok/s drifts). Bottlenecks: MTP/EAGLE-4 verify-GEMV-bound; DSpine
eager-verify + tap-D2H bound (65% of cycle, lacks the P1 device route); DFlash2
5-layer drafter-forward bound. See
[head-to-head artifact](../../benchmarks/results/2026-11-09-gfx1151-qwen38-27b-specdec-head-to-head.json).

Rows #17–21 use the relative-baseline protocol (25 outputs, ten prompts, median
of prompts, each vs its own back-to-back AR reference ~11.2–11.4 tok/s); #13/#15
use the suite-economics protocol (750 outputs, 3 reps, weighted, AR 10.85). All
rows AR-exact on the full ten-prompt mtpbench suite. The ordering is robust across
the two protocols.

- **MTP pass (#14, measured explanation):** the MTP B3 cycle is target-verify
  dominated (132 ms/cycle, ~85%; NextN proposal ~23 ms; host <1 ms). The verify
  is GPU-weight-bound and AR-exactness-constrained (exact rowtile; reassociating
  WMMA would break exact IDs) and already graph-captured. The transferable Muse
  Glimmer wins (Q6 rowtile hoist, host-sampler penalty fix) are already in the
  base, alongside graphs + hot-vocab fused head + production Q4 rowtile. No large
  missed MTP win respecting exactness.
- **DFlash2 revival (#15):** rocprof root-caused the cost gaps. The block QKV
  projection used a naive scalar GEMV (`dflash_qkv_proj_bf16_mixed`, ~5.4 ms/call,
  ~18x the bandwidth floor); routed it to the existing WMMA dense kernels
  (commit 99e2008bc). Full-suite acceptance IDENTICAL (2.32), AR-exact,
  ratio_vs_ar 0.848 -> 0.864. The remaining drafter dense projections (fc/MLP/o_proj/
  context) ALREADY dispatch to WMMA by default (`HIPENGINE_DFLASH_DRAFTER_DENSE=
  wmma`); the drafter attention is tiny (~12 ms); the target verify is
  exactness-constrained. **The residual DFlash2 deficit vs MTP is structural**:
  a 5-layer drafter forward (~96 ms/cycle, conv + selector) that MTP's 1-layer
  proposal (~2.4 ms) does not pay. The toolkit closes the margins, not this
  architectural gap. The `_PACK8_ROWTILE_MAX_ROWS=4` cliff bounds DFlash2 to
  shallow chains (B3 is already the optimum); the reverted rowtile-8 experiment
  was AR-divergent on `code_lru_cache` and remains un-root-caused.
- **Implication for DSpine/EAGLE-4:** DFlash2's drafter cost is architectural, so
  the win case for DSpine rests on its higher acceptance (adjacent causal
  injection) amortizing that cost over more committed tokens, and for EAGLE-4 on
  its single-layer feature drafter being far cheaper than DFlash2's 5-layer
  forward. Acceptance and drafter cost per committed token are the metrics to
  beat, not raw kernel tuning.
- **DSpine/EAGLE-4 measured (#17–21, clean-room):** both implemented independently
  (no vendoring) and AR-exact on the full suite, but neither beats AR on this
  throttled lane. DSpine's 5-layer backbone + 16-row block verify cost outweighs
  its 0.214 acceptance (0.38x AR); a GPU transition-cache refine gave +20%
  (3.57→4.30 tok/s) and a candidate-budget sweep confirmed smaller budgets are
  slower (fixed per-block overhead). The EAGLE-4-style single-layer feature
  drafter is cheaper than DFlash2's 5 layers and reaches 0.78x AR at its chain
  optimum (n_draft=1, 8.92 tok/s, acceptance 0.445); a cost-aware budget sweep
  (n1–8) peaks at n1 because marginal deeper drafts are not accepted enough to
  amortize their per-step cost. The dynamic tree + efficient-head sharding that
  give the vendor EAGLE-4 its 3.59x on B300 are the remaining levers; on gfx1151
  the chain draft does not yet amortize. **MTP B3 remains the shipping winner**
  (1.665x AR); the external drafters do not displace it on this hardware.

## 6. Optimization analysis: what MTP has that DSpine/EAGLE-4 lack

Code review of the measured routes, 2026-11-09. Per-cycle numbers below are
derived from the §5 rows and the worklog stage profiles. Rows marked *inferred*
are estimates, not measurements.

### 6.1 Cost model: drafting quality is not the problem for DSpine

| Method | tok/cycle | ms/cycle | verify rows | where the time goes |
| --- | ---: | ---: | ---: | --- |
| MTP B3 | ~2.8 | ~155 | 4 | verify 132 (graph, exact rowtile, ≤4-row bucket) + NextN 23 |
| DSpine full block | **~4.2** | ~1,010 | 16 | verify 879 + refine 51 + forward 46 + project/append 37 |
| EAGLE-4 chain n1 | ~1.45 | ~162 | 2 | eager verify + unprofiled drafter/host (*inferred* ~50–60 ms drafter+host) |

The true AR step costs ~88 ms on this host. **DSpine already predicts better per
cycle than MTP** (≈4.2 vs ≈2.8 committed tokens). Its 0.214 "acceptance" is
measured against 15 drafts. It loses because it verifies 16 rows on a path that
costs ~10 AR steps. **EAGLE-4 currently predicts worse than MTP**, and §6.3 shows
that this is largely a bring-up defect, not the method.

On this verify-bound lane, any method's ceiling is roughly
`tok/cycle ÷ (verify(rows) + draft)`. The verify cost for a given row count is
shared by all methods. A drafter beats MTP only by committing more tokens within
the cheap ≤4-row verify, or by making 5–8-row verifies cheap.

### 6.2 What the MTP route has that the generic drafter routes do not

| # | MTP B3 route | DSpine / EAGLE-4 generic route (`*_cycle.py`) |
| --- | --- | --- |
| 1 | Target verify is **HIP-graph captured** (enqueue-only, fixed-row buckets 2–8, device cursor advance); host residual <1 ms | Eager `verify_target_block` every cycle (Python dispatch of the full layer stack; Muse Glimmer measured 740→1 launches for the equivalent capture) |
| 2 | **Device-side accept** (`device_graph_i32x2` handoff); no logits or token readback | Target token IDs read back to host; Python accept loop; sync per cycle |
| 3 | Target hidden seed stays in **graph-owned device buffers** | `capture_layer_output_hidden` copies 3 (EAGLE) or 5 (DSpine) layers × rows × 5120 to host numpy every cycle. Enqueue-only verify explicitly refuses layer host captures, so this blocks #1. |
| 4 | Drafter weights **GGUF-quantized** with production rowtile GEMVs | BF16 drafter weights (EAGLE ≈0.93 GB read per draft step; DSpine 5 layers), DFlash WMMA dense with 16-row tiles at rows=1. Unprofiled per step. |
| 5 | NextN head is **hot-vocab + fused linear+argmax**, top-1 on device | EAGLE: full-V low-rank estimate (248320×128), top-128, then **D2H of candidates, host gather of 128 `lm_head` rows from fp32 numpy, H2D**, shortlist GEMV, D2H, host softmax/argsort, **per draft step**. DSpine: full-V 15×5120×248320 head + device top-16 + host chain walk. |
| 6 | Proposal is **graph captured** with a device proposal handoff (draft IDs never leave the device) | EAGLE: per draft step H2D of features, embeddings (host numpy gather) and positions; D2H of the output feature; 2+ `device_synchronize` |
| 7 | NextN keeps **private full-attention KV over the context** | **EAGLE drafter has no context** (see §6.3) |
| 8 | Fusion/projection of target features happens on device | EAGLE `fc` (15360→5120) and DSpine `project_target_hidden` (25600→5120) run as **fp32 numpy matmuls on the CPU every cycle** (DSpine project+append = 37 ms measured) |
| 9 | Verify width (4 rows) **sits inside the `_PACK8_ROWTILE_MAX_ROWS=4` cliff** and the rows 2–8 rowtile LM-head path | DSpine verifies 16 rows: past the rowtile cliff *and* past the `2 <= rows <= 8` row LM-head path |
| 10 | Budget B3 tuned on a correct drafter | EAGLE n1 tuned on the context-less drafter; the DSpine budget sweep is invalid (§6.3) |

### 6.3 Defects that invalidate current conclusions

These change the method verdicts, so they come before any kernel tuning.

1. **EAGLE-4 drafter ignores the context (acceptance bug).**
   `run_eagle4_cycle` passes only `ctx_aux[-1]`, and `draft_greedy_chain`
   attends only over the 1..n draft rows. It never builds a drafter KV over the
   committed context, although the artifact declares `sliding_window=4096` and
   `use_cache=true`. EAGLE-class drafters are trained to attend over their own
   KV for every prior position. The full `ctx_aux` grows every cycle (an O(n²)
   host concat) and is otherwise unused. The measured 0.445 depth-1 acceptance
   and the collapse at depth (0.099 at n8) are what a context-less drafter would
   produce. The n1 "cost-aware optimum" is therefore an artifact of the bug.
   **Fix:** prefill a drafter KV from the prompt's fused features (one rows=T
   forward). Append the K/V of the accepted rows after each verify. Decode each
   draft step as rows=1 against that KV. This is also the speed fix (it removes
   the per-step full-sequence recompute).
2. **The EAGLE-4 oracle was never pinned to a reference trace.** The design doc
   (§1 "RED-oracle questions") required checking the `fc` chunk order, the
   2H-input composition, whether `hidden_norm` applies before fusion, and the
   `norm_output` semantics against a reference forward. The landed oracle test
   checks only against its own serial numpy. Open items: whether the
   next-step feature should be pre- or post-`norm` (`EAGLE4NumpyDrafter.forward`
   has a no-op `norm_output` branch), and whether aux capture depth `l+1` in
   prefill matches `capture_layer_output_hidden=l` in verify. One trace
   comparison would close these. It is the cheapest acceptance check available.
3. **The DSpine candidate-budget sweep measured a broken configuration.**
   `candidate_budget` truncates the *drafter's input block*
   (`[bonus] + [mask]*n_drafts`) instead of the verify. The drafter was trained
   on 16-row blocks, and `positions` is still sized `ctx_len + block_size`.
   Budgets 3 and 4 show **acceptance 0.0** in their artifacts
   (`…dspine-budget{3,4}.json`). The conclusion "budget is not the lever" does
   not follow. **Fix:** always draft the full 16-row block, then verify only the
   first `n` drafts (or a confidence-truncated prefix).
4. **Drafter host memory on a unified-memory APU.** Both drafters keep fp32
   numpy copies of every weight in addition to the device BF16 copies. For
   EAGLE-4 this includes `lm_head` and `embed_tokens`, each ≈5 GB in fp32. On
   gfx1151 that host RAM is the same pool the GPU reads, so it competes for
   capacity and possibly bandwidth (*inferred*; not measured).

### 6.4 Ranked optimization list

Ordered by expected impact per unit of work. "Shared" items also help MTP or
future drafters.

| Pri | Item | Methods | Expected effect | Notes |
| --- | --- | --- | --- | --- |
| **P0** | EAGLE-4 drafter KV over context (prefill + append + rows=1 decode) | EAGLE-4 | Acceptance: the main unknown. Speed: removes per-step sequence recompute | §6.3-1. Re-run the n_draft sweep after it; the n1 optimum is void |
| **P0** | Pin the EAGLE-4 oracle to one reference forward trace (feature norm, fc order, aux depth) | EAGLE-4 | Correctness of drafting; may lift acceptance further | §6.3-2. CPU-only |
| **P0** | DSpine: draft full block, verify a prefix (fixed n ≤3, then confidence-truncated) | DSpine | Cuts verify from 16 rows (879 ms) to a ≤4-row bucket (MTP pays 132 ms for 4). *Inferred* cycle ~270 ms → ~11–12 tok/s before drafter work | §6.3-3. The vendor recipe's "DSpark dynamic verification" is this lever |
| **P1** | Graph-captured verify for the generic route: extend enqueue-only verify to write aux-layer hidden rows into graph-owned device buffers (DFlash2 already has device capture targets for prefill) | DSpine, EAGLE-4, DFlash2 | Brings verify to MTP's 132 ms/4 rows; removes eager dispatch and per-cycle D2H of layer rows | §6.2-1/3. Unlocks device accept |
| **P1** | Device-side accept + commit for the generic route (reuse MTP's `device_graph_i32x2` path) | all generic | Removes host round-trip and sync per cycle | §6.2-2 |
| **P1** | Move the `fc` / `project_target_hidden` fusion to the GPU (WMMA dense from the device aux buffers) | EAGLE-4, DSpine | DSpine −37 ms/cycle (measured cost); EAGLE removes CPU fp32 15360×5120 per cycle | §6.2-8 |
| **P1** | EAGLE efficient head fully on device: gather the 128 `lm_head` rows by device index from the resident BF16 head; device softmax/argmax (top-1 for chain) | EAGLE-4 | Removes 2 syncs + 2 transfers + a host gather per draft step | §6.2-5 |
| **P1** | Device-resident draft step: device embedding gather, on-device feature carry, no per-step H2D/D2H; then graph-capture the draft step | EAGLE-4 | Draft step approaches its bandwidth floor (≈0.93 GB BF16 ≈ 4–5 ms *inferred* at this host's bandwidth) | §6.2-6. Mirrors MTP's graph proposal |
| **P2** | Quantize the drafter weights (Q8_0 / Q4 GGUF, reusing the target's rowtile GEMVs). Drafts owe no exactness | EAGLE-4, DSpine | ~2–4× fewer bytes per draft step | §6.2-4. Acceptance must stay within band |
| **P2** | Hot-vocab draft head (MTP's lever) as an alternative or complement to the low-rank head | EAGLE-4, DSpine | Cuts the full-V low-rank pass (63 MB/step) | Measure against the low-rank head |
| **P2** | Drop the host fp32 weight copies after upload (keep only what the CPU oracle needs, lazily) | EAGLE-4, DSpine | Frees ~10+ GB of unified memory | §6.3-4 |
| **P2** | Lift the `_PACK8_ROWTILE_MAX_ROWS=4` cliff to 8 exactly (root-cause the `code_lru_cache` divergence first) | **shared** (MTP B4+, DSpine, EAGLE, trees) | Makes 5–8-row verifies cheap. This is the only lever that raises every method's ceiling | Open since the DFlash2 campaign |
| **P2** | Early-exit chunked verify: verify ≤4 rows; if all accepted, verify the next ≤4 from the committed state | DSpine (long blocks), EAGLE deep chains | Pays the 16-row cost only when acceptance would use it | Exact: each chunk is a normal exact verify |
| **P3** | EAGLE-4 dynamic tree + tree-mask verify | EAGLE-4 | Vendor's main lever | **Hard on this hybrid target.** GDN/linear-attention state is per-path, so a tree needs per-branch recurrent state (ReplaySSM-like) as well as a full-attention tree mask. Do this only after P0/P1 show a strong chain |
| **P3** | Cost tables per method (draft ms/step, verify ms/rows) feeding the budget choice (the EAGLE-Harness idea) | all | Picks depth/width per context and load | After P1, so the tables measure the final paths |
| **P3** | Protocol: run final rows on 750 outputs (suite-economics protocol) rather than 25 tokens per prompt | measurement | Removes short-run warm-up bias from the §5 relative rows | Desktop stable-clock lane |

### 6.5 Expected outcome (inferred, to be measured)

- **DSpine** with prefix verify (P0) plus graph verify and GPU projection (P1)
  costs about 132 ms verify + ~100 ms drafter per cycle and commits ≤4 tokens.
  That is roughly AR to MTP parity. Beating MTP additionally needs drafter
  forward cuts (WMMA backbone, quantized weights) or a cheaper 5–8-row verify
  (the shared cliff).
- **EAGLE-4** is the larger unknown and the larger upside. With context KV, its
  acceptance should move toward EAGLE-class chain acceptance. With a
  device-resident, graph-captured draft step (~5 ms/step *inferred*), a depth-3
  chain costs about 132 + 15 ms. That beats MTP if its depth-3 chain commits
  more than ~2.8 tokens/cycle.
- **MTP** gains only from shared levers: the rowtile cliff for B4+, or the
  production-profile verify. It is not exactness-free (see the §5 MTP pass).

## 7. Open questions

- Does Qwen3.8-27B's verify quant mix actually use the Q6 T16 rowtile (so the
  hoist transfers), or only Q5 col8?
- Can the DFlash2 rowtile admission cliff be lifted without the AR divergence
  that reverted the earlier rowtile-8 experiment (`code_lru_cache`, never
  root-caused)?
- What is the local acceptance/cost of DSpine's adjacent injection on gfx1151,
  and does ReplaySSM map onto hipEngine's linear-attention serving profile?
- Is EAGLE-4's license compatible with any hipEngine use beyond clean-room idea
  reproduction?

## 8. Sources

External, accessed 2026-11-09 (vendor claims; hardware differs from hipEngine):

- DSpine paper: arXiv:2609.36173v2 (2026-10-10),
  <https://arxiv.org/abs/2609.36173>. Code/weights/results table:
  <https://github.com/tencent/dspine/>; weights
  <https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16>.
- EAGLE-4 code: <https://github.com/eagle-hq/EAGLE-4>; matched Qwen3.8-27B
  comparison and MI355X/long-context results:
  <https://blog.eagle.ai/blog/introducing-eagle-4>; weights
  <https://huggingface.co/EAGLE-Corp/Qwen3.8-27B-EAGLE4>. License:
  Eagle 4 Source-Available License (see repo LICENSE).

Internal: [`QWEN38-27B-GFX1151-CAMPAIGN.md`](QWEN38-27B-GFX1151-CAMPAIGN.md)
(MTP baseline), [`QWEN38-27B-DFLASH2-CAMPAIGN.md`](QWEN38-27B-DFLASH2-CAMPAIGN.md)
(DFlash2 rejection + cost attribution),
[`SPECULATIVE-DECODE.md`](../reference/SPECULATIVE-DECODE.md) (EAGLE-2/3 tree
background), and the Muse Glimmer worklog entries listed in §2.