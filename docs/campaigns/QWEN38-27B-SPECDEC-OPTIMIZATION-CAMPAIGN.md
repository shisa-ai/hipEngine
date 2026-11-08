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

## 6. Open questions

- Does Qwen3.8-27B's verify quant mix actually use the Q6 T16 rowtile (so the
  hoist transfers), or only Q5 col8?
- Can the DFlash2 rowtile admission cliff be lifted without the AR divergence
  that reverted the earlier rowtile-8 experiment (`code_lru_cache`, never
  root-caused)?
- What is the local acceptance/cost of DSpine's adjacent injection on gfx1151,
  and does ReplaySSM map onto hipEngine's linear-attention serving profile?
- Is EAGLE-4's license compatible with any hipEngine use beyond clean-room idea
  reproduction?

## 7. Sources

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