# GGUF MTP llama.cpp Parity Trace and Roadmap

## 2026-06-30 — DUAL-ENGINE PER-STAGE ATTRIBUTION (active goal): where the MTP gap actually is

New goal: stand up clean per-stage profiling for BOTH engines on the same model
(Qwen3.6-35B-A3B-UD-Q4_K_M, gfx1151) and attribute the MTP tok/s gap to specific
stages/kernels. Full write access to `~/llama.cpp` granted; HIP and Vulkan both in
scope. Builds: llama.cpp HIP+Vulkan at `6e9007ae6` (master, clean). Model 21.1 GiB.

### SUMMARY — full tok/s ladder + gap decomposition (read this first)

| config (same model, gfx1151) | AR tok/s | MTP tok/s | uplift |
| --- | --- | --- | --- |
| hipEngine (HIP/ROCm, exact) | 54.95 | 60.8 (suite) | 1.114× |
| llama.cpp HIP/ROCm (dp4a) | 51.38 | 67.3 (suite) / 75.4 (cli prompt) | ~1.31–1.47× |
| llama.cpp **Vulkan** (dp4a) | **62.65** | **84.6 (cli prompt)** | ~1.35× |

**There are two separate comparisons:**
1. **HIP-vs-HIP parity (the 67.3 tok/s row):** hipEngine's base decode is not behind:
   AR is **54.95 vs llama HIP 51.38 tok/s**. The remaining HIP-vs-HIP gap is MTP
   uplift/economics: llama's pipeline uses dp4a/q8_1 verify and can run no-probe
   full-block speculation at **0.402 target passes/output**; hipEngine's exact route
   needs a B1 probe and spends **0.567 passes/output**. Copying dp4a into hipEngine
   reaches only **61.3-61.6 tok/s** and fails the ja correctness gate.
2. **Best llama.cpp parity (Vulkan rows):** Vulkan adds a separate backend factor on
   Strix Halo: llama Vulkan AR is **62.65 tok/s** vs hipEngine HIP **54.95 tok/s**.
   The large lm-head is equally BW-efficient (566 vs 550 GFLOPS), but Vulkan's driver
   and fused ggml op shapes are stronger on the smaller ops. hipEngine is HIP-only, so
   matching llama Vulkan is a backend project, not an MTP-policy fix.

**Correctness-preserving levers (exact precision) — TESTED 2026-06-30, both already captured:**
- **Fusion: already done** — qkv is a single fused `attn_qkv` GEMV; selected-expert
  MoE is pack8-consolidated with gate+up+silu fused. Mirrors Vulkan's qkv/`MUL_MAT_ID`.
- **Verify vec-rowtile: built+bit-exact but REFUTED** (0.93× vs the existing
  `grid.y`-occupancy rows-kernel; reverted). The dense verify GEMV is already
  occupancy-amortized at rows>1. Rowtile is the right tool only for the lm-head
  (already shipped).
- A **Vulkan backend for hipEngine** would directly capture the (now-dominant)
  backend factor, but is a large architectural undertaking.

=> No remaining correctness-preserving HIP-kernel lever for AR/verify; the residual
gap is the **Vulkan-vs-HIP backend** + llama's dp4a precision.

### FINAL STAGE LEDGER — hipEngine GGUF HIP vs llama.cpp HIP

This is the current authoritative stage-by-stage attribution. Older historical
sections below are retained for archaeology; where they conflict with this table, this
table wins.

| Stage | hipEngine GGUF HIP | llama.cpp HIP | What it means |
| --- | --- | --- | --- |
| AR wall | **54.95 tok/s** (~18.2 ms/tok) | 51.38 tok/s (~19.5 ms wall; 17.26 ms GPU + host exposed) | hipEngine wins base decode. |
| AR launch shape | **762 launches/tok**, larger exact kernels, host mostly hidden | **1632 launches/tok**, `mul_mat_vec_q` dp4a dominates, ~2.2 ms host exposed | llama's dp4a kernel is good, but HIP launch shape costs it. |
| AR kernel mix | q8_0 attention proj **42%**, q4_K MoE **21%**, q6_K lm-head **9.6%**, GDN **8%** | `mul_mat_vec_q` dp4a **76.5%**, `mul_mat_vec_f` **5.8%**, `quantize_q8_1` **2.2%**, GDN **1.4%** | No hidden AR-stage deficit in hipEngine. |
| Large lm-head bandwidth | q6_K lm-head ~1850 us, **~550 GFLOPS** | Vulkan comparison: 1794 us, **566 GFLOPS** | Large contiguous GEMV is already at parity-class BW. |
| Current exact block verify | rows=4: **42.40 ms wall**, **38.08 ms GPU**, **875 launches**, only **10.2% host exposed** | llama MTP rocprof deadlocks at finalize; 4-row `llama-bench -p 4 -b 4` proxy shows dp4a matmuls dominate | The old host/graph hypothesis is dead; hipEngine verify is GPU-bound. |
| hipEngine verify GPU mix | q8_0 attention **32.7%**, GDN **16.1%**, q4/qK MoE selected **25.9%**, rowtile lm-head **5.9%**, router/norm/misc **19.4%** | proxy: `mul_mat_vec_q_moe` **40.5%** + `mul_mat_vec_q` **33.8%** | llama's advantage is cheaper dp4a/q8_1 verify, not missing hipEngine fusion. |
| Exact MTP economics | B5 **60.78 tok/s**, **1.1134x**, acc/out **0.535**, passes/out **0.567** | B2 **67.3 tok/s**, ~**1.31x**, acc/out **0.598**, passes/out **0.402** | hipEngine does **41% more target-pass work/output**. |
| dp4a transplant | B5 **61.61 tok/s**, **1.1322x**, +1.3% E2E; block verify **42.9 -> 41.2 ms** (-3.9%) | native llama HIP still **67.3 tok/s** | dp4a helps, but does not close the gap. |
| no-probe llama recipe | B5 **56.42 tok/s**, acc/out **0.324** | llama succeeds with no-probe economy | The recipe does not transfer; hipEngine needs the B1 probe. |
| Correctness | exact path passes; dp4a ja top-1 **0.700 < 0.90** gate | llama speed row uses dp4a/q8_1 | Matching llama's precision regime violates hipEngine's guard. |

**Deal:** every stage has now been accounted for. hipEngine is not missing a secret
llama.cpp HIP kernel stage. It has a faster exact AR pipeline, an exact verifier that
is already GPU-bound and already has the useful fusion/amortization, and a speculative
policy that needs one extra cheap probe because exact failed rows are expensive. llama
HIP's remaining advantage is a whole-pipeline dp4a/no-probe economy; reproducing only
the dp4a kernel in hipEngine gives ~61.6 tok/s, not 67.3, and fails Japanese.

### FINAL RESULT — the MTP gap vs llama HIP is the dp4a verify, with an exact accuracy cost

**Bottom line:** hipEngine's GGUF AR decode is *faster* than llama.cpp HIP's
(54.95 vs 51.38 tok/s) — our exact HIP kernels are genuinely good, and fusion /
verify-amortization are already captured. The **only** place we trail llama HIP is
the **MTP verify loop**, and the entire deficit is llama's **dp4a verify pass**,
which **does not pass hipEngine's correctness gate**.

**Exact performance cost (what dp4a buys, what it doesn't):**

| HIP-vs-HIP, full suite | hipEngine (exact) | llama HIP (dp4a) |
| --- | --- | --- |
| AR tok/s | **54.95** | 51.38 |
| MTP tok/s | 60.8 | **67.3** |
| ms / output token | 16.4 | 14.9 |
| MTP uplift over own AR | 1.114× | **1.31×** |
| target-verify passes / output | **0.567** | **0.402** |
| acc / output | 0.535 | 0.598 |

- We do **41% more target-verify work per output token** (0.567 vs 0.402). llama runs
  **1 verify pass/cycle** (passes/out = `1 − acc/out` = 0.402); hipEngine runs ~1.22
  (a cheap B1-probe pass + the block pass).
- **Why:** llama's dp4a verify is cheaper per row, so it (a) pays less per pass and
  (b) can speculate full blocks with **no probe** (wasted rows are cheap). Our *exact*
  verify is pricier per row, so a wasted block-row is costly → the B1-probe is the best
  route (removing it regressed to **1.069×**, acc/out collapsing 0.535→0.379).
- **Swapping only the verify to dp4a on hipEngine buys +1.3% E2E** (60.8 → **61.61
  tok/s**, 1.114×→**1.1322×**, `results/2026-06-30-ar-mtp-suite-full-dp4a-verify-diagnostic.json`)
  — still **8.5% behind llama HIP (67.3)**, because our exact AR is already fast (a
  fixed verify saving is a *smaller ratio* over a fast AR) and the no-probe structure
  needs whole-pipeline dp4a. On the GPU-bound block verify the wall barely moves
  (exact 42.9 ms → all-dp4a 41.2 ms = **−3.9%**); dp4a does **not** speed AR at all
  (54.97 ≈ 54.95). The isolated MoE-GEMV dp4a is ~2–3× but does not translate E2E
  (GPU-bound + added per-layer `quantize_q8_1` launches).

**Exact accuracy cost — llama's dp4a verify FAILS hipEngine's correctness gate:**

- Gate (`AGENTS.md`/`docs/TESTING.md`): **KL ≤ 0.05 AND top-1 agreement ≥ 90%** vs
  `kernels/cpu_reference/` on fixture inputs.
- Measured greedy top-1 agreement of the dp4a (q8_1) verify vs the exact path
  (`scratchpad/dp4a_correctness.py`, flag `HIPENGINE_GGUF_T16_SELECTED_DP4A=1`, real
  ja+code context, 30 tokens):

  | category | dp4a greedy top-1 agreement | gate ≥ 0.90 | first divergence |
  | --- | --- | --- | --- |
  | code | **1.000** (30/30) | PASS | none |
  | **general_ja** | **0.700** (21/30) | **FAIL** | token 20 |

- **dp4a is a hard FAIL on Japanese: 0.700 < 0.90** (q8_1 activation quantization
  loses CJK precision; the greedy path diverges from exact at token 20 and compounds).
  Code is unaffected (1.000). So llama's MTP speed advantage is bought with an accuracy
  loss that violates hipEngine's stated correctness guard — it is **not** a free win.

**Conclusion:** within the correctness gate, hipEngine's MTP (1.114× / 60.8 tok/s) is
at its exact-precision optimum and **already beats llama HIP on AR and on accuracy**.
Matching llama HIP's MTP tok/s requires its dp4a verify, which fails our ja gate
(0.700 top-1) and even then only reaches ~61.6 tok/s here (still < 67.3). The two
honest paths to actually exceed llama remain: relax the ja accuracy gate for dp4a
(not recommended — fails CJK, and insufficient alone), or add a **Vulkan backend**
(beats llama on both AR and MTP on this APU). The exact-precision HIP design point is
documented as closed.

### COFFIN NAIL — dp4a is NECESSARY but NOT SUFFICIENT to match llama HIP MTP

A default-off opt-in **`--verify-dp4a`** mode (bench flag + suite route
`resident-b1-probe-block-direct-cap32k-minrows2-pmin05-dp4a`) was added so anyone who
accepts llama's precision loss can get the max accuracy-traded perf. Measured, full
suite, gfx1151 (artifact `results/2026-06-30-ar-mtp-suite-full-dp4a-verify-diagnostic.json`):

| config | B3 | B4 | **B5 (best)** | vs llama HIP 67.3 |
| --- | --- | --- | --- | --- |
| dp4a + b1-probe (`--verify-dp4a`, the mode) | 59.85 | 60.10 | **61.3–61.6** (1.13×) | **−8.5%** |
| dp4a + no-probe (the "1 pass/cycle" recipe) | 55.71 | 56.34 | 56.42 (1.04×) | −16% |
| exact default (shipped) | 58.83 | 59.53 | 60.76 (1.114×) | −9.7% |

**Two findings nail the claim:**
1. **The "dp4a + 1 verify pass/cycle" hypothesis is FALSE on hipEngine.** No-probe is
   *worse* (56.4, acc/out collapses to 0.324) — our exact/dp4a draft + adaptive-AR-
   fallback latches to AR on a rejected block; the b1-probe is essential. dp4a's
   slightly-less-accurate drafts make no-probe slightly worse, not better.
2. **dp4a + b1-probe (best dp4a) reaches only ~61.6 tok/s — still 8.5% short of llama
   HIP 67.3.** So dp4a is *necessary but not sufficient*: llama's 1.31× uplift also
   needs its **slower AR baseline** (51.38 — a fixed verify saving is a bigger *ratio*
   over a slower AR) and its **no-probe acceptance economy** (which doesn't transfer
   to our fast-AR setup). hipEngine's faster exact AR (54.95) structurally caps the
   uplift ratio even with dp4a.

**Accuracy cost of using the mode** (unchanged): ja greedy top-1 **0.700 < 0.90 gate
FAIL** (first divergence token 20), code 1.000. So `--verify-dp4a` is correctly
default-off and labelled accuracy-degrading; it buys ~+1.3% over the exact default at
the cost of failing the ja gate, and does **not** reach llama HIP. The mode exists for
users who explicitly accept that trade; the shipped default stays exact (1.114×).

### MEASURED CYCLE-STAGE BUCKETS — same buckets on hipEngine and llama.cpp HIP

The deeper instrumentation is now in place on both sides:

- hipEngine: `--record-cycle-stage-timings` on `scripts/gguf_ar_mtp_suite.py`.
- llama.cpp HIP: local diagnostic patch in
  `/home/lhl/llama.cpp/llama.cpp-hip/tools/server/server-context.cpp` plus
  `/home/lhl/llama.cpp/llama.cpp-hip/common/speculative.cpp`; set
  `LLAMA_MTP_STAGE_TIMINGS=/path/file.jsonl` to emit one JSONL record per MTP verify
  cycle. The hipEngine harness summarizes it via `--stage-timings-jsonl`.

Measured setup: Qwen3.6-35B-A3B-UD-Q4_K_M GGUF, `gfx1151` / Radeon 8060S, prompt
suite `benchmarks/prompts/mtpbench-code-general-ja.jsonl`, greedy sampling,
reasoning off. These are **diagnostic timing runs**, not replacement headline rows:
hipEngine timing adds bookkeeping overhead, and the llama natural-24 server trace
measures a slightly faster protocol than the retained 67.3 tok/s HIP row. Use the
stage buckets for attribution; keep the retained non-instrumented rows for the
official tok/s ladder.

Artifacts:

- hipEngine exact B5 deep: `benchmarks/results/2026-06-30-ar-mtp-stage-timing-b5-exact-deep.json`
- hipEngine dp4a+B1 B5 deep: `benchmarks/results/2026-06-30-ar-mtp-stage-timing-b5-dp4a-deep.json`
- hipEngine llama-compat dp4a B2 after top-1 diagnostic fix:
  `benchmarks/results/2026-06-30-ar-mtp-llama-compat-dp4a-b2-top1-deep.json`
- hipEngine llama-compat dp4a device-chain smoke:
  `benchmarks/results/2026-06-30-ar-mtp-llama-compat-dp4a-b2-devicechain-smoke.json`
- hipEngine llama-compat dp4a prewarmed device-chain split:
  `benchmarks/results/2026-06-30-ar-mtp-llama-compat-device-chain-dp4a-b2-full-split.json`
- hipEngine llama-compat dp4a prewarmed device-chain sync-stage draft attribution:
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-draftsync-full.json`
- hipEngine llama-compat dp4a prewarmed device-chain after exact Q6_K top-1/gather
  specialization:
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-q6top1-full.json`
  and same-tree disabled control
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-q6top1-control-full.json`
- hipEngine llama-compat dp4a prewarmed device-chain after Q6_K top-1/gather,
  sync-stage draft attribution:
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-q6top1-draftsync-full.json`
- hipEngine llama-compat dp4a prewarmed device-chain after verifier direct-state
  cleanup:
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-skip-snapshot-full.json`
- hipEngine llama-compat dp4a all-sync fine-grained verifier attribution after
  verifier direct-state cleanup:
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-skip-snapshot-allsync-smoke.json`
- hipEngine fused-B1 block probe B5:
  `benchmarks/results/2026-06-30-ar-mtp-fused-b1-block-direct-cap32k-minrows2-pmin05-b5-full.json`
  and non-stage check
  `benchmarks/results/2026-06-30-ar-mtp-fused-b1-block-direct-cap32k-minrows2-pmin05-b5-full-nostage.json`
- hipEngine fused-B1 block probe smoke after non-llama direct-state snapshot-skip
  carryover:
  `benchmarks/results/2026-07-01-ar-mtp-fused-b1-block-direct-cap32k-minrows2-pmin05-snapshot-skip-smoke.json`
- llama.cpp HIP B2 deep:
  `benchmarks/results/2026-06-30-llamacpp-mtp-stage-timing-b2-natural24-deep.json`
  and `.jsonl`

#### Instrumented economics

| config | AR tok/s | MTP tok/s | uplift | cycle wall / output | accepted / output | draft acceptance | target passes / output | target rows / output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine exact B5 | 54.56 | 59.61 | 1.093× | 16.800 ms | 0.535 | 0.723 | 0.567 | 1.163 |
| hipEngine dp4a+B1 B5 | 54.60 | 60.01 | 1.099× | 16.690 ms | 0.533 | 0.735 | 0.570 | 1.154 |
| llama.cpp HIP B2 | 52.13 | 72.12 | 1.383× | 14.231 ms traced / 13.866 ms server | 0.567 server / 0.610 traced | 0.805 | 0.390 | 1.148 |

Denominator note: llama's server summary reports accepted/output over 240 predicted
tokens (`0.567`); the per-cycle trace excludes the first warmup task and reports 223
visible traced tokens (`0.610`). Stage ms/output uses the traced denominator.

#### Stage ms / output token

| bucket | hipEngine exact B5 | hipEngine dp4a+B1 B5 | llama.cpp HIP B2 | interpretation |
| --- | ---: | ---: | ---: | --- |
| `cycle_wall_ms_per_output` | 16.800 | 16.690 | 14.231 | Instrumented wall. dp4a closes only 0.110 ms/output in this run. |
| `draft_initial` | 1.937 | 1.943 | 2.140 | Draft is not the retained B5 gap; hipEngine is slightly faster here. |
| `draft_topk_readback` | 1.158 | 1.134 | n/a | hipEngine name is a synchronization drain + top-k readback, not pure top-k kernel time. |
| `llama_draft_sample_topk` | n/a | n/a | 1.886 | llama draft is sampler/top-k dominated; MTP decode itself is small (`0.118 + 0.134`). |
| `target_serial_verify_step` | **6.682** | **6.647** | 0.000 | This is the hipEngine B1 probe / serial verifier cost. llama has no equivalent bucket. |
| `target_block_verify_total` | 8.157 | 8.073 | 12.083 | Compare verifier total, not raw `target_block_forward` alone. |
| `target_block_layer_total` | 7.022 | 6.864 | n/a | hipEngine block verifier is GPU layer work: mostly linear-attn layers. |
| `target_block_linear_attn_layers` | 5.195 | 5.049 | n/a | Biggest hipEngine block sub-bucket. |
| `target_block_full_attn_layers` | 1.827 | 1.816 | n/a | Secondary hipEngine block sub-bucket. |
| `target_block_lm_head_sample` | 0.573 | 0.586 | n/a | Not the gap. |
| `target_block_forward` | 8.065 | 7.985 | 0.549 | llama's raw `llama_decode(ctx_tgt)` is async; its GPU drain lands below. |
| `mtp_context_replay_append` | 0.000 | 0.000 | **11.348** | llama's `common_speculative_process()` cost; this is part of verifier total. |
| `llama_process_build_draft_batch` | n/a | n/a | **11.235** | This is the newly split llama bucket. It effectively includes target decode drain + target nextn embedding handoff. |
| `llama_process_decode_ctx_dft` | n/a | n/a | 0.112 | Draft-context catch-up decode is not the big llama cost. |
| `target_block_snapshot` | 0.060 | 0.056 | 0.001 | Not the gap. |
| `target_block_acceptance_accounting` | 0.001 | 0.001 | 0.181 | Visible in llama, still too small to explain the delta. |
| `target_block_replay_or_commit` | 0.029 | 0.029 | 0.004 | Not the gap. |
| `accept_policy_and_seed` | 0.002 | 0.002 | 0.002 | Not the gap. |
| `cycle_wall_over_legacy_ms_per_output` | 0.026 | 0.026 | n/a | hipEngine has no hidden wall outside the legacy timing denominator. |

**Answer:** after adopting dp4a, the measured gap is not draft, snapshot, commit,
policy bookkeeping, or hidden host wall. The gap is the extra hipEngine verification
economy:

- hipEngine dp4a verifier work = `target_serial_verify_step + target_block_verify_total`
  = **14.720 ms/output**.
- llama verifier work = `target_block_verify_total` = **12.083 ms/output**.
- Difference = **+2.637 ms/output** for hipEngine, mostly the B1 serial probe.
- hipEngine draft is **0.197 ms/output faster**, so the net instrumented wall gap is
  ~**2.46 ms/output** (16.690 - 14.231), which is fully explained by verifier
  economics.

This is the fine-grained version of the earlier retained-row conclusion. The retained
non-instrumented gap is smaller (**~1.37 ms/output**, 61.61 vs 67.3 tok/s) because the
diagnostic protocols and instrumentation overhead differ, but the attribution is the
same: llama gets its speedup by avoiding the hipEngine B1 serial probe and spending
fewer target passes/output (`0.390` traced here, `~0.402` retained) while maintaining
higher draft acceptance. Directly copying `target_block_forward` is the wrong target;
in llama most verifier time is under `mtp_context_replay_append`, and the deep split
puts that cost specifically in `llama_process_build_draft_batch` (target decode drain
and nextn embedding handoff), not in the draft-context decode.

#### Compat draft split: prewarm fixes initialization; steady-state draft drain remains

The first compat-dp4a deep split showed `draft_initial ~= 4.03 ms/output` and
`draft_topk_readback ~= 3.80 ms/output`. A top-1 diagnostic fix was implemented so
`--llama-compat` no longer forces top-10 proposal readback, but the full-suite result
was flat:

| config | MTP tok/s | cycle wall / output | `draft_initial` | `draft_topk_readback` | `target_block_verify_total` |
| --- | ---: | ---: | ---: | ---: | ---: |
| compat dp4a B2, old top-10 diagnostic | 52.42 | 19.096 ms | 4.031 | 3.799 | 14.749 |
| compat dp4a B2, top-1 diagnostic | 52.48 | 19.074 ms | 4.043 | 3.833 | 14.715 |
| compat dp4a B2 + prewarmed device-chain | **52.79** | **18.963 ms** | 4.028 | 3.839 | **14.620** |
| compat dp4a B2 + prewarmed device-seed-chain | 52.53 | 19.065 ms | 4.020 | 3.827 | 14.724 |

So the draft-side slowdown is not top-k width by itself. The first device-chain smoke
also showed the wrong bottleneck because it measured the lazy 268 MB full-vocab
embedding-table upload inside the short run:

| probe | MTP tok/s | cycle wall / output | `draft_initial` | `draft_device_chain_ensure_embed_table` | result |
| --- | ---: | ---: | ---: | ---: | --- |
| compat dp4a B2 + device-chain, smoke | 36.12 | 27.704 ms | 14.855 | 11.888 | full-vocab embed-table upload dominates the short run |
| compat dp4a B2 + prewarmed device-chain, full | **52.79** | **18.963 ms** | 4.028 | 0.000 | upload removed; steady-state still slow |

The prewarm/cache fix removes that initialization artifact, but it does **not** close
the llama gap. A split-bucket rerun of the same device-chain route shows why:

| split bucket, compat dp4a B2 + device-chain | ms/output |
| --- | ---: |
| `draft_initial` | 4.033 |
| `draft_topk_readback` | 3.839 |
| `draft_device_chain_drain` | **3.830** |
| `draft_topk_d2h` | **0.008** |

The "readback" bucket is therefore almost entirely a GPU drain, not host copy time.
This is the draft-side target for replication: hipEngine is draining roughly
**3.83 ms/output** of resident draft GPU work where llama's draft sampler/top-k bucket
is **1.886 ms/output** and total `draft_initial` is **2.140 ms/output**.
Persistent/prewarmed device-chain and resident `pending_h` semantics are now explicit
routes, but the remaining win requires reducing the actual device draft work/drain
or fusing it differently; avoiding D2H alone cannot produce the missing tokens.

#### Sync-stage draft attribution: where that GPU drain actually goes

Follow-up diagnostic route:
`llama-compat-device-chain-dp4a-draftsync` =
`--llama-compat --resident-mtp-device-chain --resident-mtp-draft-sync-stage-timings --verify-dp4a`.
This route inserts `hipDeviceSynchronize()` boundaries inside each resident MTP draft
layer section, so it changes timing and is **not** a retained performance route. Its
only purpose is attribution of the previous `draft_device_chain_drain` bucket.

Command:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a-draftsync \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-draftsync-full.json
```

Measured full-suite result, Qwen3.6-35B-A3B-UD-Q4_K_M, gfx1151/Radeon 8060S,
`benchmarks/prompts/mtpbench-code-general-ja.jsonl`, greedy, reasoning off, 10 prompts:

| config | AR tok/s | MTP tok/s | vs AR | cycle wall / output | acc / output | draft acceptance | passes / output | rows / output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine compat device-chain dp4a B2, sync-stage | 54.72 | **52.37** | 0.957x | 19.122 ms | 0.561 | 0.640 | 0.439 | 1.316 |
| llama.cpp HIP B2 deep trace | 52.13 | **72.12** | 1.383x | 14.231 ms | 0.610 traced | 0.805 | 0.390 | 1.148 |

Draft-side split, ms/output:

| bucket | hipEngine compat device-chain dp4a B2 sync-stage | llama.cpp HIP B2 deep | delta / meaning |
| --- | ---: | ---: | --- |
| `draft_initial` | **4.084** | **2.140** | hipEngine draft costs **+1.944 ms/output**. |
| `draft_mtp_layer_forward` | 3.639 | n/a | Sum of the synchronized hipEngine draft layer sections. |
| `draft_run_project` | 0.101 | n/a | Not the gap. |
| `draft_run_qkv_kvwrite` | 0.211 | n/a | Not the gap. |
| `draft_run_attention` | 0.718 | n/a | Material, but smaller than lm-head. |
| `draft_run_ffn_up_shared` | 0.557 | n/a | Material, secondary. |
| `draft_run_moe_down_combine` | 0.166 | n/a | Not the gap. |
| `draft_run_lm_head` | **1.882** | n/a | Biggest hipEngine draft section; roughly equals llama's whole `llama_draft_sample_topk` bucket. |
| `draft_device_topk_gather` | 0.357 | n/a | Device top-k + embedding gather for the next draft depth. |
| `draft_topk_readback` | 0.007 | n/a | D2H remains tiny after sync splitting. |
| `llama_draft_decode_initial` | n/a | 0.118 | llama MTP decode itself is small. |
| `llama_draft_decode_next` | n/a | 0.134 | llama MTP decode itself is small. |
| `llama_draft_sample_topk` | n/a | **1.886** | llama draft is sampler/top-k dominated. |

Verifier-side split, ms/output:

| bucket | hipEngine compat device-chain dp4a B2 sync-stage | llama.cpp HIP B2 deep | delta / meaning |
| --- | ---: | ---: | --- |
| `target_block_verify_total` | **14.715** | **12.083** | hipEngine verifier costs **+2.632 ms/output**. |
| `target_block_layer_total` | 12.827 | n/a | hipEngine's block verifier is still real target-layer work. |
| `target_block_linear_attn_layers` | **9.451** | n/a | Biggest hipEngine verifier sub-bucket. |
| `target_block_full_attn_layers` | 3.375 | n/a | Secondary hipEngine verifier sub-bucket. |
| `target_block_lm_head_sample` | 1.198 | n/a | Material, but not the biggest verifier delta. |
| `mtp_device_kv_commit` | 0.297 | n/a | Small compat lifecycle overhead. |
| `target_block_forward` | 14.573 | 0.549 | Raw bucket is async-misaligned across engines. |
| `mtp_context_replay_append` | 0.009 | 11.348 | llama's target decode drain and nextn embedding handoff live here. |
| `llama_process_build_draft_batch` | n/a | 11.235 | Main llama verifier drain is in process/build, not draft decode. |
| `llama_process_decode_ctx_dft` | n/a | 0.112 | Draft-context catch-up is not the big llama cost. |

This answers the current parity question precisely:

- hipEngine now matches the **observable llama.cpp MTP semantics** in the compat lane:
  B2, no B1 probe, p_min 0, shifted MTP context replay, device MTP KV, resident
  device-chain drafting, and optional dp4a verify.
- hipEngine does **not** yet match llama.cpp's **operation cost**. The measured gap is
  still about **4.89 ms/output** in the diagnostic trace (`19.122 - 14.231`):
  **+1.94 ms/output draft** and **+2.63 ms/output verifier**, with the rest from
  acceptance/pass economy and small lifecycle/accounting differences.
- The draft gap is no longer a black box. Inside the prior GPU drain, the largest
  section is the full-vocab draft LM head (**1.882 ms/output**), followed by draft
  attention (**0.718**), FFN/up/shared (**0.557**), and device top-k/gather
  (**0.357**). D2H is still negligible.
- The verifier gap is target-layer work, especially hipEngine's B2 linear-attention
  layer bucket (**9.451 ms/output**) and full-attention layers (**3.375 ms/output**),
  not a missing `pending_h` handoff or a hidden host copy.

#### First gap-closing fix: exact Q6_K top-1 + embedding gather for compat draft

Implemented an exact Q6_K lm-head specialization for the llama-compat device-chain
draft path:

- New kernel: `hipengine_gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32`.
- It preserves the same per-output Q6_K dot-product reduction and top-1 tie-break as
  `gguf_q6_k_pack8_gemv_decode_bf16_f32_out -> topk_f32_rows_i32`, but writes one
  winner per pack8 block, reduces those winners, and optionally gathers the selected
  FP32 embedding row for the next draft depth.
- Runtime flag: `HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_GATHER=1` by default, scoped to
  resident MTP draft `top_k == 1`. Set it to `0` for same-tree A/B.

Validation:

```bash
python3 -m py_compile \
  hipengine/kernels/hip_gfx1100/quant/gguf_q6_k_pack8_gemv.py \
  hipengine/speculative/mtp_resident_draft.py \
  tests/test_gguf_q6_k_pack8_gemv_decode.py

PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 pytest -q \
  tests/test_gguf_q6_k_pack8_gemv_decode.py

PYTHONPATH=. pytest -q \
  tests/test_mtp_resident_draft_device_commit.py \
  tests/test_gguf_mtp_bench_metrics.py \
  tests/test_gguf_ar_mtp_suite.py
```

The new unit gate compares the fused kernel against the old logits -> top-k ->
gather chain and requires identical selected id, selected value, and embedding row.

Same-tree full-suite A/B, Qwen3.6-35B-A3B-UD-Q4_K_M, gfx1151/Radeon 8060S,
`benchmarks/prompts/mtpbench-code-general-ja.jsonl`, greedy, reasoning off,
`--scope full --mtp-route llama-compat-device-chain-dp4a --record-cycle-stage-timings
--require-cached-build`:

| config | AR tok/s | MTP tok/s | vs AR | cycle wall / output | acc / output | draft acceptance | `draft_initial` | `draft_topk_readback` | `target_block_verify_total` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q6 top-1/gather disabled | 54.74 | 52.60 | 0.961x | 19.033 ms | 0.561 | 0.640 | 4.033 ms | 3.838 ms | 14.682 ms |
| Q6 top-1/gather enabled | 54.75 | **53.34** | **0.974x** | **18.772 ms** | 0.561 | 0.640 | **3.712 ms** | **3.518 ms** | 14.737 ms |

Measured effect:

- Headline: **52.60 -> 53.34 tok/s** on the llama-compat dp4a B2 diagnostic route
  (**+1.4%**), with unchanged acceptance.
- Cycle wall: **-0.261 ms/output**.
- Draft drain: **-0.321 ms/output** (`draft_initial`), almost exactly the section this
  fix targeted.
- Verifier: unchanged within noise (**+0.056 ms/output** in this A/B).

Sync-stage rerun after the fix:

| bucket | before Q6 top-1/gather | after Q6 top-1/gather | delta |
| --- | ---: | ---: | ---: |
| MTP tok/s | 52.37 | **53.43** | +2.0% |
| `cycle_wall_ms_per_output` | 19.122 | **18.737** | **-0.385 ms** |
| `draft_initial` | 4.084 | **3.758** | **-0.326 ms** |
| `draft_run_lm_head` | 1.882 | 1.916 | +0.034 ms |
| `draft_device_topk_gather` | 0.357 | **0.001** | **-0.356 ms** |
| `draft_topk_readback` | 0.007 | 0.007 | flat |
| `target_block_verify_total` | 14.715 | **14.661** | -0.055 ms |
| `target_block_linear_attn_layers` | 9.451 | **9.422** | -0.029 ms |
| `target_block_full_attn_layers` | 3.375 | **3.367** | -0.008 ms |

This closes the obvious top-k/gather waste, but it does **not** close the llama.cpp
gap. After the fix, the sync-stage diagnostic is still **18.737 ms/output** vs
llama.cpp HIP B2 trace **14.231 ms/output**, a remaining **+4.51 ms/output**:

- draft side: `draft_initial` **3.758** vs llama **2.140** = **+1.62 ms/output**;
- verifier side: `target_block_verify_total` **14.661** vs llama **12.083** =
  **+2.58 ms/output**;
- the rest is small lifecycle/accounting plus acceptance/pass economy.

The next compat-lane target is therefore no longer device top-k/gather. It is the
actual target verifier layer cost, especially `target_block_linear_attn_layers`
(still **9.42 ms/output**) and `target_block_full_attn_layers` (**3.37 ms/output**),
plus any remaining draft lm-head/attention/FFN work that differs from llama.cpp's
MTP draft decode shape.

#### Second gap-closing fix: defer exact direct-state writes and skip unnecessary snapshots

The next cleanup targeted verifier lifecycle overhead that hipEngine was still
paying even though the block verifier already captures per-row linear states:

- In the direct-state block verifier, the linear-attention direct branch no longer
  runs the BF16-to-F32 QKV conversion used only by the non-direct prefill conv path.
- `verify_target_block(..., defer_linear_state_commit=True)` no longer copies the
  final captured Conv/GDN state back into the resident state when the caller will
  immediately commit an accepted captured row or restore/replay.
- Shared block-verifier callers now skip `_linear_state_snapshot()` when direct
  commit is exact for the block (`bulk` verifier with
  `start_position + rows < 1024`, which covers the measured B2 suite). Rollback
  still keeps the snapshot on non-exact paths.
- New diagnostic flag `--target-block-sync-stage-timings` and suite route
  `llama-compat-device-chain-dp4a-allsync` add verifier-internal sync buckets for
  attribution only.

This is not llama-only. The retained non-llama `can_block_verify` path already
uses the shared snapshot policy. A follow-up carried the same policy into the
non-llama B1 branch-safe/fused-B1 block verifier: exact direct-commit outcomes
commit captured row 1 on strict B1 accept and row 0 on reject/root-topK branch,
so the rollback snapshot is unnecessary there as well. Smoke validation for
`resident-fused-b1-block-direct-cap32k-minrows2-pmin05` passed on 2026-07-01
(`benchmarks/results/2026-07-01-ar-mtp-fused-b1-block-direct-cap32k-minrows2-pmin05-snapshot-skip-smoke.json`);
that route remains diagnostic and below AR/default, but it now uses the same
direct-state waste policy.

Validation:

```bash
python3 -m py_compile \
  hipengine/runtime/qwen35_gguf_runner.py \
  scripts/gguf_mtp_bench.py \
  scripts/gguf_ar_mtp_suite.py \
  tests/test_gguf_mtp_bench_metrics.py \
  tests/test_gguf_ar_mtp_suite.py

PYTHONPATH=. pytest -q \
  tests/test_gguf_mtp_bench_metrics.py \
  tests/test_gguf_ar_mtp_suite.py \
  tests/test_mtp_resident_draft_device_commit.py
```

Full-suite A/B against the prior Q6 top-1/gather row, same command family:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-skip-snapshot-full.json
```

| config | AR tok/s | MTP tok/s | vs AR | cycle wall / output | acc / output | draft acceptance | `target_block_verify_total` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| after Q6 top-1/gather | 54.75 | 53.34 | 0.974x | 18.772 ms | 0.561 | 0.640 | 14.737 ms |
| + direct-state cleanup | 54.67 | **55.41** | **1.014x** | **18.069 ms** | 0.561 | 0.640 | **14.044 ms** |

Measured effect:

- Headline: **53.34 -> 55.41 tok/s** on the llama-compat dp4a B2 route
  (**+3.9%**), with unchanged acceptance.
- Cycle wall: **18.772 -> 18.069 ms/output** (**-0.702 ms/output**).
- Verifier: `target_block_verify_total` **14.737 -> 14.044 ms/output**
  (**-0.694 ms/output**).
- The fixed cost was not draft-side: `draft_initial` stayed flat
  (**3.712 -> 3.708 ms/output**).

Stage deltas vs the Q6 top-1/gather row:

| bucket | before | after | delta |
| --- | ---: | ---: | ---: |
| `target_block_verify_total` | 14.737 | **14.044** | **-0.694 ms** |
| `target_block_forward` | 14.590 | **13.997** | **-0.593 ms** |
| `target_block_layer_total` | 12.847 | **12.477** | **-0.370 ms** |
| `target_block_linear_attn_layers` | 9.467 | **9.185** | **-0.282 ms** |
| `target_block_full_attn_layers` | 3.380 | **3.292** | **-0.088 ms** |
| `target_block_setup` | 0.270 | **0.049** | **-0.221 ms** |
| `target_block_snapshot` | 0.090 | **0.000** | **-0.090 ms** |
| `target_block_replay_or_commit` | 0.051 | **0.042** | -0.009 ms |

Final all-sync smoke attribution after this cleanup, diagnostic-only:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope smoke \
  --mtp-route llama-compat-device-chain-dp4a-allsync \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-skip-snapshot-allsync-smoke.json
```

Top synchronized buckets, ms/output:

| bucket | ms/output |
| --- | ---: |
| `target_block_verify_total` | 15.581 |
| `target_block_layer_total` | 13.317 |
| `target_block_linear_attn_layers` | **10.191** |
| `target_block_full_attn_layers` | 3.126 |
| `draft_initial` | 2.763 |
| `target_block_linear_attn_norm_qkv_gate` | **2.429** |
| `target_block_linear_attn_ffn_moe_expert_gate_up` | **1.563** |
| `draft_run_lm_head` | 1.472 |
| `target_block_linear_attn_ffn_moe_expert_down` | **1.241** |
| `target_block_lm_head_sample` | 0.969 |
| `target_block_linear_attn_ssm_out` | 0.851 |
| `target_block_linear_attn_chain_gdn` | 0.790 |
| `target_block_full_attn_norm_qkv_split` | 0.700 |
| `target_block_linear_attn_ffn_moe_router` | 0.581 |

This is now the clearest operation-level target list: after semantic replication and
direct-state cleanup, the remaining verifier cost is dominated by target linear
attention projection (`norm_qkv_gate`) plus selected-MoE expert gate/up/down in the
linear-attention layers. The remaining draft cost is still mainly the MTP lm-head.

Updated remaining gap vs llama.cpp HIP B2 deep trace:

| bucket | hipEngine compat B2 after cleanup | llama.cpp HIP B2 deep | remaining delta |
| --- | ---: | ---: | ---: |
| cycle wall / output | 18.069 ms | 14.231 ms | **+3.838 ms** |
| `draft_initial` | 3.708 ms | 2.140 ms | **+1.568 ms** |
| `target_block_verify_total` | 14.044 ms | 12.083 ms | **+1.961 ms** |

So the remaining replication work is concrete: reduce the resident draft LM-head /
top-k section and the B2 target block layer time. Simply copying the llama.cpp
high-level no-probe lifecycle has already been tested and does not make the speed
match.

The next optimization question is therefore specific: keep the no-probe
`llama-compat` semantics and reduce the measured operation costs. The older
approximate no-probe route collapsed acceptance (`56.42 tok/s`, acc/output `0.324`).
The true `llama-compat-device-chain-dp4a` route now keeps acceptance near llama's
retained row (`0.561` acc/output) and finally beats its same-run AR baseline
(`55.41 tok/s`, `1.014x`), but it remains well short of llama.cpp HIP MTP. The
current blocker is operation cost in draft lm-head and target linear-attention/MoE
verifier sections, not an unattributed llama.cpp kernel bucket.

#### Queued fixes, ordered by expected impact

| priority | fix | why this is next | success gate |
| ---: | --- | --- | --- |
| 1 | **Fused B1/block verifier path** | Current dp4a B5 pays `target_serial_verify_step` **6.647 ms/output** plus block verify **8.073 ms/output**. A useful implementation must preserve the B1 probe's acceptance economy while avoiding a separate full serial target pass. | **Implemented and rejected for promotion 2026-06-30.** It cuts B1 serial work but moves too much work into 2-row blocks; exact B5 is **60.40 tok/s**, below the retained exact **60.78** and dp4a **61.61** rows. |
| 2 | Compat target block layer-time reduction | After direct-state cleanup, compat B2 still spends **14.044 ms/output** in target block verify. The all-sync split points to linear-attn `norm_qkv_gate` plus selected-MoE expert gate/up/down as the dominant sub-buckets. | Reduce `target_block_layer_total` / `target_block_linear_attn_layers` with acceptance unchanged and full-suite B2 moving toward llama's **12.083 ms/output** verifier trace. |
| 3 | Compat draft GPU-drain reduction | Compat B2 still spends **3.708 ms/output** in draft after Q6 top-1/gather. All-sync attribution keeps MTP lm-head as the largest draft sub-bucket. | Continue cutting compat `draft_initial` toward llama's **2.140 ms/output** without lowering full-suite acceptance. Remaining draft work is actual lm-head/attention/FFN cost, not D2H. |
| 4 | Confidence-gated no-probe policy | True llama-compat now proves no-probe can keep acc/output near **0.561**, but still needs operation-cost work to compete with the retained exact B5 route. | After operation-cost fixes, revisit whether a confidence gate can improve rows/output without reintroducing the B1 serial probe. |
| 5 | Keep llama.cpp deep instrumentation aligned | The current split proved llama's verifier drain lives in `llama_process_build_draft_batch`, not raw `target_block_forward`. Keep this patch available for A/B after every major hipEngine verifier change. | Re-run llama deep trace when upstream or local diagnostic patch changes; do not compare raw async buckets. |

**Fused-B1 implementation result (2026-06-30):** added default-off
`--fused-b1-block-probe` and suite route
`resident-fused-b1-block-direct-cap32k-minrows2-pmin05`. The flag lets adaptive B1
probe cycles use one strict two-row block over `[prev, draft0]` instead of entering
the serial verifier loop. Row-state commit uses the existing exact direct-commit
block path.

| route / artifact | MTP tok/s | vs AR | acc / output | passes / output | rows / output | `target_serial_verify_step` | `target_block_verify_total` | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| retained exact B5, non-stage rowtile confirm | **60.78** | **1.113×** | 0.535 | n/a | n/a | n/a | n/a | current exact default |
| fused-B1 B5, non-stage | **60.40** | **1.107×** | 0.535 | n/a | n/a | n/a | n/a | **do not promote** |
| retained exact B5, stage-timed | 59.61 | 1.093× | 0.535 | 0.567 | 1.163 | **6.682 ms/out** | **8.157 ms/out** | baseline attribution |
| fused-B1 B5, stage-timed | 60.40 | 1.107× | 0.535 | 0.465 | 1.205 | **2.095 ms/out** | **12.447 ms/out** | serial mostly removed, block cost rises |
| retained dp4a B5, non-stage | **61.61** | **1.132×** | 0.533 | n/a | n/a | n/a | n/a | accuracy-traded ceiling still higher |

Why it fails the promotion gate:

- It does what it says mechanically: stage-timed serial rows fall from **78** to
  **25** on B5, and those remaining serial rows are p_min zero-draft AR cycles
  (`linear_draft_tokens=0`), not missed fused B1 probes.
- But it turns many B1 probes into two-row target blocks: block passes rise from
  **44** to **75**, block rows from **172** to **234**, and
  `target_block_verify_total` rises by **+4.29 ms/output**. The serial bucket falls
  by **-4.59 ms/output**, so the verifier bucket only improves by about
  **0.30 ms/output** in the instrumented run.
- The non-stage full-suite row is **60.40 tok/s**, below the retained exact
  **60.78 tok/s** and far below the accuracy-traded dp4a **61.61 tok/s**; it is not
  a retained speed win.

**Replication-lane next unit:** keep the llama-compatible no-probe B2 shape and cut
the measured compat costs directly: resident draft GPU drain first, then B2 block
verifier layer time. Confidence-gated no-probe may still be useful for the default
hipEngine policy, but it is not the current llama.cpp replication task.

#### Can we adopt a true llama.cpp mode?

Yes, as an explicit opt-in **llama-compat / accuracy-traded mode**. No, not as the
shipped exact default. The current "no-probe" hipEngine experiments should not be
over-read as a full llama.cpp clone: they tested the high-level idea (one block pass,
no B1 probe) inside hipEngine's existing policy stack, not every llama.cpp semantic.

What a true llama mode needs to replicate:

| llama.cpp piece | why it matters | current hipEngine status |
| --- | --- | --- |
| `--spec-draft-n-max 2`, `--spec-draft-p-min 0.0` lifecycle | llama drafts every cycle up to B2 unless the MTP sampler itself stops; no hipEngine p_min gate. | Implemented in opt-in `--llama-compat`; suite routes are fixed to B2 to avoid mislabeled artifacts. |
| No B1 probe / one target block verify per cycle | This removes the `target_serial_verify_step` bucket entirely. | Implemented in `--llama-compat`: disables adaptive B1 probe/fallback and forces block verify with `--target-block-min-rows 2`. |
| llama MTP context handoff (`common_speculative_process` / `pending_h` / `verify_h`) | Draft quality depends on how target verify hidden rows seed the next MTP draft. | Shifted prompt catch-up via `--mtp-context-replay` plus device-resident MTP KV is implemented in `--llama-compat`. Explicit subroutes now add prewarmed resident device-chain drafting and optional resident device seed (`pending_h`) starts. |
| llama accept/checkpoint semantics | Partial accepts restore/commit through llama's checkpoint and `common_speculative_accept` path. | hipEngine has rollback/direct-commit paths, but they are not mechanically identical. |
| q8_1 / dp4a verify economy | This is part of llama's speed/acceptance economics, and it fails hipEngine's ja gate. | Exact compat route stays precision-preserving; `llama-compat-dp4a` adds default-off `--verify-dp4a` for llama's accuracy-traded regime. |

Implemented opt-in routes (2026-06-30):

| route | extra args | budget | purpose |
| --- | --- | ---: | --- |
| `llama-compat` | `--llama-compat` | B2 fixed | Precision-preserving closest semantic replica: B2, p_min 0, full draft vocab, shifted context replay + device KV, no B1 probe/fallback, one block verify/cycle. |
| `llama-compat-dp4a` | `--llama-compat --verify-dp4a` | B2 fixed | Same semantics plus llama-style q8_1/dp4a selected-expert verify. Accuracy-traded; ja gate failure remains expected until proven otherwise. |
| `llama-compat-device-chain` | `--llama-compat --resident-mtp-device-chain` | B2 fixed | Adds prewarmed resident device-chain drafting, mirroring llama's resident `ctx_dft` lifecycle more closely than per-depth host embedding handoff. |
| `llama-compat-device-chain-dp4a` | `--llama-compat --resident-mtp-device-chain --verify-dp4a` | B2 fixed | Accuracy-traded device-chain route; best measured compat replication row so far. |
| `llama-compat-device-chain-dp4a-draftsync` | `--llama-compat --resident-mtp-device-chain --resident-mtp-draft-sync-stage-timings --verify-dp4a` | B2 fixed | Diagnostic-only sync-stage route that attributes the resident draft GPU drain. Not a performance route. |
| `llama-compat-device-chain-dp4a-allsync` | `--llama-compat --resident-mtp-device-chain --resident-mtp-draft-sync-stage-timings --target-block-sync-stage-timings --verify-dp4a` | B2 fixed | Diagnostic-only route that sync-splits both resident draft and target block verifier sections. Not a performance route. |
| `llama-compat-device-seed-chain` | `--llama-compat --resident-mtp-device-seed --resident-mtp-device-chain` | B2 fixed | Also starts each draft from resident target `pending_h` rather than a host-copied seed. |
| `llama-compat-device-seed-chain-dp4a` | `--llama-compat --resident-mtp-device-seed --resident-mtp-device-chain --verify-dp4a` | B2 fixed | Full llama-lifecycle diagnostic: B2 no-probe, context replay + device KV, resident device seed, prewarmed device chain, and dp4a verify. |

`--llama-compat` is deliberately an override flag: if a wrapper passes conflicting
draft/policy knobs first, the bench normalizes them after parsing. The suite also
refuses non-B2 budget overrides for these routes because the child bench would force
`draft_n_max=2`; allowing a `B5` label would make the artifact misleading.

Exact route command:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-06-30-ar-mtp-llama-compat-b2.json
```

Accuracy-traded dp4a route command:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-dp4a \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-06-30-ar-mtp-llama-compat-dp4a-b2.json
```

Accuracy-traded prewarmed device-chain route command:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-06-30-ar-mtp-llama-compat-device-chain-dp4a-b2-full.json
```

Accuracy-traded resident device-seed + device-chain route command:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-seed-chain-dp4a \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-06-30-ar-mtp-llama-compat-device-seed-chain-dp4a-b2-full.json
```

Split-bucket attribution rerun for `draft_device_chain_drain` / `draft_topk_d2h`:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-06-30-ar-mtp-llama-compat-device-chain-dp4a-b2-full-split.json
```

Sync-stage attribution rerun for the inside of `draft_device_chain_drain`:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a-draftsync \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-draftsync-full.json
```

Measured full-suite result (2026-06-30, same model/gfx1151, stage timings enabled):

| config | AR tok/s | MTP tok/s | vs AR | cycle wall / output | acc / output | draft acceptance | passes / output | rows / output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama-compat` exact B2 | 54.76 | **51.16** | 0.934× | 19.570 ms | 0.559 | 0.635 | 0.441 | 1.322 |
| `llama-compat-dp4a` B2 (top-1 diagnostic) | 54.77 | **52.48** | 0.958× | 19.074 ms | 0.561 | 0.640 | 0.439 | 1.316 |
| `llama-compat-device-chain-dp4a` B2 | 54.71 | **52.79** | 0.965× | 18.963 ms | 0.561 | 0.640 | 0.439 | 1.316 |
| `llama-compat-device-chain-dp4a-draftsync` B2 | 54.72 | **52.37** | 0.957× | 19.122 ms | 0.561 | 0.640 | 0.439 | 1.316 |
| `llama-compat-device-seed-chain-dp4a` B2 | 54.74 | **52.53** | 0.960× | 19.065 ms | 0.561 | 0.640 | 0.439 | 1.316 |
| prior dp4a+B1-probe B5 | 54.60 | **60.01** | 1.099× | 16.690 ms | 0.533 | 0.735 | 0.570 | 1.154 |
| llama.cpp HIP B2 trace | 52.13 | **72.12** | 1.383× | 14.231 ms | 0.610 traced | 0.805 | 0.390 | 1.148 |

Stage ms/output:

| bucket | compat exact B2 | compat dp4a B2 | compat device-chain dp4a B2 | prior dp4a+B1 B5 | llama.cpp HIP B2 | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `draft_initial` | 4.084 | 4.043 | 4.033 split / 4.028 headline | 1.943 | 2.140 | hipEngine compat's shifted-context/full-vocab B2 draft is expensive; prewarmed device-chain does not reduce steady-state draft drain. |
| `target_serial_verify_step` | 0.000 | 0.000 | 0.000 | 6.660 | 0.000 | compat successfully removes the B1 serial probe. |
| `draft_topk_readback` | n/a | 3.833 | 3.839 | 1.134 | n/a | now split: this is almost all GPU drain, not copy time. |
| `draft_device_chain_drain` | n/a | n/a | **3.830** | n/a | n/a | resident device-chain still waits on the full draft GPU work at chain end. |
| `draft_topk_d2h` | n/a | n/a | **0.008** | n/a | n/a | D2H is too small to be the missing llama gap. |
| `target_block_verify_total` | 15.164 | 14.715 | 14.714 split / 14.620 headline | 8.073 | 12.083 | the saved serial probe is paid back by a much more expensive B2 block verifier. |
| `target_block_forward` | 15.021 | 14.585 | 14.581 | 7.985 | 0.549 | llama's raw forward bucket is not comparable; most llama verify/state work is in `mtp_context_replay_append`. |
| `mtp_context_replay_append` | 0.008 | 0.008 | 0.009 | 0.000 | 11.348 | hipEngine's context replay cost is not in this bucket; its cost manifests in draft/block wall. |
| `mtp_device_kv_commit` | 0.299 | 0.296 | 0.297 | 0.000 | n/a | small but nonzero compat lifecycle overhead. |
| `cycle_wall_ms_per_output` | 19.570 | 19.074 | 19.066 split / 18.963 headline | 16.690 | 14.231 | best compat replication is still ~4.73 ms/output slower than llama's traced path. |

**Result:** copying the observable llama policy is not sufficient. Adding the next
llama lifecycle pieces also does not close the gap: prewarmed device-chain improves
the compat dp4a headline only **52.48 -> 52.79 tok/s**, and resident device seed is
slightly worse (**52.53 tok/s**). The route does remove the B1 probe and preserves
decent full-suite acceptance, but the hipEngine realization of that lifecycle is
slower than the retained B1-probe path. Compared with prior dp4a+B1 B5, compat saves
**6.65 ms/output** of serial verify, but adds roughly **+6.64 ms/output** in block
verify and **+2.09 ms/output** in draft work. Versus llama.cpp HIP B2, the best
replication row is still slower by about **4.73 ms/output**: **~1.89 ms/output** in
draft, **~2.54 ms/output** in block verify, and the rest in small lifecycle/accounting
differences plus weaker acceptance/pass economy. The residual gap is now even more
concrete: reduce the actual resident draft GPU drain and B2 block verifier cost, not
just the llama no-probe policy flag or host readbacks.

So the real answer is: there is no architectural reason we cannot add a true
`llama-compat` mode while keeping exact mode as default. The reasons not to promote
it by default are the known correctness tradeoff (dp4a ja top-1 **0.700 < 0.90**) and
the fact that the current approximate no-probe routes did not reproduce llama's
economics. The clean experiment is now implemented and measured: the exact route lands
at **51.16 tok/s**, the dp4a route lands at **52.48 tok/s**, and the best prewarmed
device-chain dp4a replication row lands at **52.79 tok/s**, all below hipEngine AR and
well below llama HIP. Semantic parity alone was not the missing piece; the gap is
implementation/backend cost in the compat draft/verifier lifecycle.

Commands used:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route resident-b1-probe-block-direct-cap32k-minrows2-pmin05 \
  --budgets 5 \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-06-30-ar-mtp-stage-timing-b5-exact-deep.json

PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route resident-b1-probe-block-direct-cap32k-minrows2-pmin05-dp4a \
  --budgets 5 \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-06-30-ar-mtp-stage-timing-b5-dp4a-deep.json

PYTHONPATH=. python3 scripts/llamacpp_mtp_bench.py \
  --server-bin /home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-server \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --alias qwen36-35b \
  --port 8013 \
  --ctx-size 8192 \
  --gpu-layers 99 \
  --draft-max 2 \
  --mode both \
  --protocol natural \
  --max-tokens 24 \
  --server-extra-arg=--reasoning \
  --server-extra-arg=off \
  --stage-timings-jsonl benchmarks/results/2026-06-30-llamacpp-mtp-stage-timing-b2-natural24-deep.jsonl \
  --output benchmarks/results/2026-06-30-llamacpp-mtp-stage-timing-b2-natural24-deep.json \
  --log-dir /tmp/llamacpp-mtp-stage-timing-b2-natural24-deep-logs
```

Important: these stage windows are **diagnostic**, not a new retained benchmark
denominator. Some fields are nested by design (`target_block_verify_total` includes
snapshot/forward/accounting/replay sub-windows), so totals should be used for
attribution and ranking, not summed as disjoint wall time. The retained tok/s still
uses the existing suite protocol; `cycle_wall_*` is there to expose hidden overhead
that the legacy counters may miss.

Profiling harness (both engines, reproducible): llama HIP via `rocprofv3
--kernel-trace` on `llama-bench`/`llama-cli` (MTP path deadlocks rocprof at finalize
→ use the batched-forward proxy `llama-bench -p 4 -b 4`); llama Vulkan via
`GGML_VK_PERF_LOGGER=1` (per-op GFLOPS); hipEngine via the AR/MTP suite + rocprof.


| same model, same prompt where noted | AR (tg128) | MTP B2 (llama-cli, explain_concept prompt) |
| --- | --- | --- |
| llama.cpp **HIP/ROCm** | 51.38 | 75.4 |
| llama.cpp **Vulkan** | **62.65** | **84.6** |
| hipEngine (HIP/ROCm only) | 54.95 | 60.8 (full suite; same-prompt TBD) |

Two attributions:
1. **hipEngine's HIP kernels are FASTER than llama's HIP** (AR 54.95 > 51.38). The
   earlier "hipEngine wins AR" holds only against llama's *slower* (HIP) backend.
2. **llama's Vulkan AR (62.65) beats hipEngine's best HIP (54.95) by ~14%**, and
   Vulkan MTP (84.6) vs hipEngine (60.8). On this RDNA3.5 APU the **Vulkan shader
   compiler/driver is materially more efficient for these GEMVs than ROCm/HIP**. So
   a large part of "where we lose" is the **ROCm-vs-Vulkan backend gap**, which is
   SEPARATE from the MTP algorithm. hipEngine is HIP-only, i.e. structurally on the
   disadvantaged backend for this hardware. Closing it means either (a) a hipEngine
   Vulkan backend, or (b) raising the HIP GEMV efficiency toward Vulkan's.

The gap therefore decomposes into **(backend: HIP vs Vulkan) + (MTP uplift)** — not
uplift alone. The AR/verify analysis below is within the HIP/ROCm backend.

### Why Vulkan is faster: kernel FUSION, not raw BW (lm-head is equal on both)

Vulkan per-op timing (`GGML_VK_PERF_LOGGER=1`, AR decode) gives effective GFLOPS,
which for these memory-bound GEMVs tracks effective bandwidth:

| op (AR decode) | Vulkan | hipEngine | note |
| --- | --- | --- | --- |
| lm-head `q6_K m=248320 k=2048` | 1794 µs, **566 GFLOPS** | ~1850 µs, **~550 GFLOPS** | **EQUAL** — both saturate BW on a large contiguous GEMV |
| attn proj `q8_0 m=8192 k=2048` | 90.8 µs, 369 GFLOPS (**qkv FUSED into one m=8192 op**) | current audit: qkv already fused as one `attn_qkv` GEMV | no missing qkv-fusion lever remains |
| MoE `MUL_MAT_ID_VEC q4_K m=512 k=2048 n=8 n_expert=256` | 24.9 µs, **674 GFLOPS** (**all 8 selected experts in ONE call**) | current audit: pack8 selected MoE already consolidated with gate+up+silu fused | high-level MoE consolidation already captured |

**Finding:** on the *large* op (lm-head) the two backends are **equally BW-efficient
(566 vs 550 GFLOPS)** — Vulkan has no magic raw-bandwidth edge. The initial read was
that hipEngine lacked Vulkan's qkv/selected-expert fusion. The follow-up audit below
closed that: hipEngine already has fused `attn_qkv` and pack8 selected-MoE. So the
remaining Vulkan advantage is backend/compiler/op-shape efficiency on the small ops,
not a missing high-level fusion item in the HIP path.

**Concrete implication:** a hipEngine Vulkan backend is the clean way to capture this
backend factor. More HIP-side qkv/MoE fusion is not an available correctness-preserving
lever for the current path.

### 2026-06-30 IMPLEMENTED+TESTED: both levers are ALREADY captured by existing kernels

Acted on the two levers above (build/test, not just propose). Result: **both are
already implemented in hipEngine's HIP path; neither has remaining headroom.**

- **Fusion — already done.** Attention qkv is a single fused `attn_qkv` weight/GEMV
  (`qwen35_gguf_runner.py:1753`), not split q/k/v. The selected-expert MoE is already
  consolidated (`_launch_selected_expert_pack8_moe_pair` ids-GEMV) with gate+up fused
  (`dual`) and silu fused (`q4_k_t16_selected_dual_silu_direct`). Mirrors Vulkan's
  fused qkv + `MUL_MAT_ID`. No new fusion to add.
- **Verify amortization — built, bit-exact, but REFUTED (reverted).** Wrote a q8_0
  t16 **rowtile** (read each weight tile once, accumulate ROW_TILE rows — the lm-head
  rowtile pattern). Bit-exact vs per-row decode (rows 2-6). A/B vs the runner's
  *actual* verify kernel (`q8_0_t16_gemv_kernel` at rows=R, single launch `grid.y=R`):
  rowtile **0.93-0.94×** (rows=4: 31.0 vs 28.7 µs; rows=6: 40.3 vs 38.0 µs). The
  existing `grid.y` kernel already amortizes via **occupancy** — at rows=R it launches
  R× more blocks (better GPU utilization on these small weights) than the rowtile's
  single block/tile. The rowtile only beats *naive 4×-separate-launches* (1.36-1.58×),
  which the runner doesn't do. Right tool only for the huge lm-head (already shipped).
  Not landed.

**Net:** hipEngine's HIP kernels already capture both fusion and verify amortization
(the dense GEMV at rows>1 is already occupancy-amortized: `q8_0_t16_dual_split` 141 µs
at rows=1 → 220 µs at rows=4 = 1.56× for 4× rows = 2.5× cheaper per row). The residual
MTP gap is therefore the **Vulkan-vs-HIP backend** (which hipEngine can't close without
a Vulkan backend) plus llama's dp4a precision. No remaining correctness-preserving
HIP-kernel lever for the AR/verify dense path.

### AR decode (within HIP): hipEngine beats llama HIP; gap to Vulkan is backend

| AR decode (single-token), same model | llama.cpp HIP | hipEngine GGUF |
| --- | --- | --- |
| Wall tok/s (`llama-bench tg128` / hipEngine suite) | **51.38** | **54.95** |
| GPU kernel ms/token (rocprof kernel-trace) | 17.26 | ~18 |
| Kernel launches / token | **1632** | **762** |
| Dominant kernel(s) | `mul_mat_vec_q` (dp4a) **76.5%**, one unified GEMV for attn+MoE+lm-head; `quantize_q8_1` 2.2% | specialized EXACT kernels: q8_0 attn proj ~42% (`q8_0_t16_dual_split`+`_gemv`+`_triple`), q4_k MoE ~21%, q6_k lm-head ~9.6%, GDN ~8% |
| Bound by | host launch overhead (1632 small dp4a launches → ~2.2ms host exposed on top of 17.26ms GPU = ~19.5ms wall) | GPU-bound, host hidden (762 larger exact launches; 18.2ms wall ≈ GPU time) |

**Finding:** hipEngine's AR decode is **faster** than llama's (54.95 vs 51.38).
hipEngine uses fewer, larger *exact* kernels that are GPU-bound (host hidden);
llama uses a single highly-optimized *dp4a* `mul_mat_vec_q` for 76.5% of decode but
issues 2× the launches, exposing ~2.2ms/token of host overhead. So the base decode
is not where hipEngine loses — **the entire MTP gap (60.8 vs 67.3 tok/s) is in the
speculative machinery / uplift** (llama 1.342× vs hipEngine 1.114× over their own
AR). This redirects the investigation from AR GEMVs (we win) to the verify/draft
economics. (Method: `rocprofv3 --kernel-trace` on `llama-bench -p 0 -n 64 -r 1`
and the hipEngine AR step loop; normalized per-token.)

**Next:** profile llama's MTP verify (batched B+1 rows) — hypothesis: llama's
batched dp4a `mul_mat_q`/`mul_mat_vec_q` amortizes weight reads across verify rows
more cheaply *relative to its slower AR* than hipEngine's per-row exact verify does
relative to its faster AR. That relative-amortization is the suspected uplift lever.

### Verify mechanism: initial hypothesis, then refuted by direct test

llama-cli MTP B2 on the explain_concept prompt = **75–78 tok/s** (uplift ~1.47–1.52×
over its 51 AR; even higher than the server-suite 67.3 because this is a favorable
English prompt). rocprof of the MTP path itself DEADLOCKS at finalize (the draft-mtp
second-context/queue setup; not size- or graph-dependent), so the verify was profiled
via its equivalent **batched B+1-row forward** (`llama-bench -p 4 -b 4 -ub 4`, which
finalizes cleanly):

| verify-shape (4-row) forward | llama.cpp HIP | hipEngine initial read |
| --- | --- | --- |
| matmul kernels | `mul_mat_vec_q_moe` 40.5% + `mul_mat_vec_q` 33.8% (dp4a vec; batch is a grid dim → **each block reads a weight tile ONCE, computes all B+1 rows** = weight-read amortized across verify rows) | `q8_0_t16_dual_split` etc. with `blockIdx.y = row` → **a separate block per row, weight re-read B+1×** (NO cross-row amortization). The only amortized hipEngine path (WMMA prefill) is SLOWER at rows=4 (56.9 vs 42.3 ms) due to tile-setup overhead. |

**2026-06-30 correction:** this was the right hypothesis to test, but it is no longer
an open lever. The exact q8_0 rowtile was built and bit-exact for rows 2-6, then
lost to the existing rows kernel: rows=4 **31.0 vs 28.7 us** (0.93x), rows=6 **40.3
vs 38.0 us** (0.94x). The current `grid.y=R` kernel already gets the useful
multi-row benefit through occupancy; rowtile underutilizes the small q8_0 weights.
Only the huge shared lm-head benefits, and that rowtile is already shipped.

**Current attribution:** llama's verify is cheaper because it runs the whole
speculative economy in dp4a/q8_1 and can afford no-probe full-block attempts.
hipEngine's exact verifier is GPU-bound and already at its exact-precision floor.

---

# GGUF MTP llama.cpp Parity Trace (history)

- Date: 2026-06-29 (Goal — Part 1 set: target-verify amortization is the sole remaining gap; acceptance shown already at llama.cpp parity via cap32k-recover; shootout order inverted, verify-wall promoted to P0; llama.cpp parity shootout matrix update; B1-probe/block-direct/cap32k AR-beating route retained; bulk row-1 direct-commit exactness diagnostic; native row-1 direct-commit diagnostic; context replay + device-seed route rejection; device-seed + draft-KV route rejection; resident draft p_min strict-block rejection; direct verifier row-state commit diagnostic; resident device hidden-seed diagnostic; hybrid strict-block cap32k rejection; cap32k recovery full-suite diagnostic; strict-context route added; deferred hidden-copy rejection; device top-k40 rejection; resident top-k40 full-suite update; production verifier/full-suite update; systemic workbench update; performance-path update 2026-06-27; correctness-solved update 2026-06-26; original trace 2026-06-25)
- Branch: `mtp-gguf`
- Hardware for all runtime numbers below: **gfx1151 / AMD Radeon 8060S (Ryzen AI Max+ 395)**, not the default W7900. Numbers state their scope; the current authoritative MTP numbers are full-suite AR/MTP suite rows.
- hipEngine source baseline for the current performance review: `cfb584615b801ce0be7f622ea695327950018f74`
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
  session, `scripts/gguf_ar_mtp_suite.py --scope smoke`). llama.cpp's retained
  full-suite HIP AR reference is **50.13 tok/s**.
- **MTP is still the parity gap, but the same-protocol AR-beat gate is now
  closed.** The retained default full-suite route
  `resident-b1-probe-block-direct-cap32k` measures **AR 54.59 tok/s; best MTP B3
  56.54 tok/s = 1.0356× AR**, `apple_to_apple_ok=true`, `mtp_beats_ar=true`.
  It combines a cheap strict B1 cap32k probe with direct-commit B3 block
  verification after a full B1 accept. B3 acceptance is **40/140 = 0.286
  accepted/output**, draft acceptance is **0.645**, target layer passes drop to
  **0.779/output**, direct commit rows are **15**, and replay rows are **0**.
  The previous best retained diagnostic was B1 **52.08 tok/s = 0.9540× AR** via
  resident device hidden seed. llama.cpp's retained full-suite reference is
  still **67.29 tok/s at B2 = 1.342× its AR**. hipEngine now beats its own AR,
  but needs about **+19% relative tok/s** from 56.54 tok/s to match that
  llama.cpp row.
- **Current next goal:** close the llama.cpp MTP parity gap by improving the
  retained `resident-b1-probe-block-direct-cap32k` family, not by reworking AR
  kernels. `Qwen35GGUFMTPContext` already covers the
  `process_verifier_rows()`/`draft()`/`accept()` seed lifecycle shape, and direct
  row-state commit has proven exact enough to retain a B3 speed route. The
  remaining llama.cpp patterns to adopt are the target/draft memory economics:
  keep `pending_h`/`verify_h`-style rows resident across the target batch,
  promote B2/B3 block verification without serial fallback waste, and lift
  accepted/output while keeping target layer passes below the current
  **0.779/output**. Success for the next goal is a full-suite artifact that
  approaches or beats llama.cpp's **67.29 tok/s B2** row under the same
  no-gaming category protocol.
- **There is no single bandwidth-starved GEMV to fix.** Measured cold-DRAM
  (MALL-defeated): dense Q8_0 c=1 GEMV ~51–70% of peak, selected-MoE GEMV
  ~70–80%. Every kernel micro-lever (dp4a, split-K, fusion, MoE-graph, cache
  hints) is real in isolation and **flat e2e** (table below).
- **Verifier host-vs-GPU split is resolved for the current suite route.** Fresh
  GGUF serial-target rocprof (`scripts/gguf_mtp_verifier_rocprof.py`, 12
  measured target steps, post no-logits cleanup) shows **18.63 ms host wall /
  16.56 ms kernel time per target step = 89% kernel time**, ~709 launches/step.
  A same-day rerun after the capped/short-block probes remains the same shape:
  **19.37 ms host / 16.95 ms kernel = 87.5% kernel time**, **708.9
  launches/step** (`benchmarks/results/2026-06-29-gguf-mtp-verifier-rocprof-rerun.json`).
  A current 8-step rerun is unchanged: **19.03 ms host / 16.76 ms kernel =
  88.0% kernel time**, **708.5 launches/step**, with dense Q8_0 GEMV **48.9%**
  and selected-MoE GEMV **24.0%** of kernel time
  (`benchmarks/results/2026-06-29-gguf-mtp-verifier-rocprof-current.json`).
  A post bulk-row1-exactness rerun remains the same shape: **18.65 ms host /
  16.75 ms kernel = 89.8% kernel time**, **708.6 launches/step**, dense Q8_0
  GEMV **49.0%** and selected-MoE GEMV **24.6%**
  (`benchmarks/results/2026-06-29-gguf-mtp-verifier-rocprof-post-bulk-row1.json`).
  The retained
  `resident-serial-fallback` route is GPU/weight-streaming bound, not
  host-launch-bound.
- **New standard measurement:** `scripts/gguf_ar_mtp_suite.py` produces ONE
  apple-to-apple AR-vs-MTP artifact under an enforced config (see "How to
  continue").

### Where hipEngine still falls short vs llama.cpp

The milestone is real: hipEngine GGUF MTP now beats the same-run hipEngine AR
baseline. The remaining parity target is llama.cpp's MTP uplift and category
coverage, not hipEngine AR speed.

| Dimension | hipEngine current default | llama.cpp retained reference | Gap / interpretation |
| --- | --- | --- | --- |
| Best total MTP throughput | B3 **56.54 tok/s** | B2 **67.29 tok/s** | llama.cpp is **+10.75 tok/s / +19.0%** faster in absolute decode throughput. |
| AR-normalized uplift | **1.0356x AR** | **1.3423x AR** | llama.cpp gets **+29.6%** more uplift relative to its own AR. |
| Accepted/output at speed winner | B3 **0.286** (`40/140`) | B2 **0.598** (`3064/5120`) | hipEngine accepts about **2.1x fewer** draft tokens per visible output. |
| Draft acceptance at comparable B3 | B3 **0.645** | B3 **0.660** | Per-attempt B3 quality is close; the bigger miss is how often useful B2/B3 drafting is attempted and retained. |
| Target pass amortization | B3 measured **0.779 target layer passes/output** | B2 inferred **0.402 target batches/output** from `1 - accepted/output` | hipEngine still streams target layers about **1.9x** more often per output token. llama.cpp does not expose layer-pass counters in the retained artifact, so this is an inference from accepted/output. |
| Category coverage | Code B3 wins (**57.55 tok/s**, **0.500 accepted/output**); `general_en`, `general_ja`, and `mixed_ja_en` have **0 accepted drafts** under the retained route | Code winner B3 **72.59 tok/s**, non-code winners are B2: `general_en` **63.83**, `general_ja` **62.25**, `mixed_ja_en` **67.27 tok/s** with **0.56-0.60 accepted/output** | hipEngine's current route is effectively a code-category win plus near-AR fallback elsewhere. llama.cpp's B2 works across all categories. |
| Budget shape | B1/B2 remain below AR; B3 is the only winning budget; B4/B5 regress | B1/B2/B3 all beat AR strongly; B2 is fastest, B5 maximizes acceptance | The next target is a robust B2/B3 policy, not deeper B4/B5 drafting. |

Bottom line (corrected by the artifact evidence in "Goal — Part 1" below):
**acceptance is already at llama.cpp parity on every category; the only
remaining gap is target-verify amortization.** The retained default route's
non-code zeros are a *policy artifact* (`--adaptive-ar-fallback` stops drafting
after one miss), not a draft-quality deficit — `cap32k-recover` already matches
llama.cpp's per-category acceptance. The current route pays too many target
layer passes per visible token, and so does every high-acceptance route we have.

### Goal — Part 1 (HIGHEST PRIORITY): close the target-verify amortization gap

**STATUS 2026-06-30 — CLOSED (banked) by owner decision.** After an exhaustive,
measurement-backed investigation (every uplift lever tried/refuted, AR multiplier
profiled, dp4a verify ~1.13x AND dp4a AR == exact AR measured, verify on its fast
path, correctness validated bottom-up incl. a new mtp_dense_attn_f32 gate, baseline
audited), llama's absolute **67.3 tok/s was shown unreachable on hipEngine in ANY
precision regime** within the correctness guard: it is a property of llama's
slower-AR (50.1) x higher-uplift (1.342x) profile, which hipEngine's faster-AR
(54.95) x exact-precision (1.114x) profile cannot reproduce. The retained, shipped
result is the bit-exact Q6_K T16 rowtile lm-head kernel: GGUF MTP **1.0534x ->
1.1134x AR (60.8 tok/s = 90.3% of llama's 67.3)**, beating llama on AR and on the ja
correctness gate. **Owner chose to bank this exact-precision win** rather than relax
the ja gate for dp4a (which reaches only ~62 tok/s anyway) or fund a speculative,
multi-session AR-decode kernel-R&D project (the only correctness-preserving path that
could raise the absolute number, with no high-confidence optimization identified, on
a path where hipEngine already beats llama). The parity goal is therefore closed as
**structurally bounded at the exact-precision design point**, not as an open
engineering gap. See the 2026-06-30 entries in the "Bottom line" section and WORKLOG.

The original P0 framing below is retained for history.

This is the first part of the llama.cpp-parity goal. Resolve these P0
determinations **before** running any S1-S3 policy probe. The evidence below
re-frames the gap and is the reason the shootout order in the next section is
inverted from how it was first written.

**Evidence that re-frames the gap (from the 2026-06-29 full-suite artifacts):**

1. The retained default route's non-code "0 accepted" is drafting being switched
   off, not bad drafts. Per-category for
   `2026-06-29-ar-mtp-suite-full-b1-probe-block-direct-cap32k.json`: code B3
   `drafts=56, accepted=40`; `general_en`/`general_ja`/`mixed_ja_en` each
   `drafts=2, accepted=0` — i.e. two attempts, then `--adaptive-ar-fallback`
   runs pure AR for the rest. The headline **1.0356x AR is a code-only win
   averaged up**, not a cross-category win.
2. **Acceptance is already solved.** A route that keeps drafting
   (`cap32k-recover`, child of
   `2026-06-29-ar-mtp-suite-full-cap32k-recover.json`) matches llama.cpp B2
   per category:

   | Category | hipEngine `cap32k-recover` acc/out | llama.cpp B2 acc/out |
   | --- | ---: | ---: |
   | general_en | 0.608 | 0.576 |
   | general_ja | 0.459 | 0.563 |
   | mixed_ja_en | 0.615 | 0.599 |

   Yet `cap32k-recover` measures **~0.95x AR (below AR)**. So hipEngine already
   drafts as well as llama.cpp on every category and **still cannot turn that
   acceptance into a speedup**. The bottleneck is cost per visible token, not
   acceptance.
3. The cost is target-verify amortization. hipEngine's best is **0.779 target
   layer passes/output** (one near-full target weight stream per ~1.3 visible
   tokens); llama.cpp's fused 4-token verify graph (~9 ms) implies **~0.25-0.40
   passes/output**. AR is already faster than llama.cpp's AR, GEMV is near-peak
   BW, draft acceptance is matched — **target-verify amortization is the only
   structural advantage llama.cpp has left.**

**P0 determinations (the highest-priority list):**

| ID | Determination | What to measure / decide | Done when |
| --- | --- | --- | --- |
| P0.1 | Amortization ceiling | Compute the tok/s a single fused B-token target verify would yield at `cap32k-recover` acceptance: hold acc/out fixed, drop `target_verify_layer_passes_per_output` from 0.779 to ~0.25-0.40, and project total tok/s. Confirms the lever is sufficient to reach ~67 tok/s before building it. | A back-of-envelope + 1 measured block-verify route row showing projected tok/s ≥ llama.cpp B2 at matched acceptance. |
| P0.2 | ~~Unblock the fused multi-token block verifier~~ **CLOSED 2026-06-30 — REFUTED, do not build.** | The premise (host-launch floor → collapse 875 launches via graph capture / GDN-fix / C-loop) was the OLD serial route. The current block verify is **GPU-kernel-BOUND (38.1 ms GPU / 42.4 ms wall, only 10.2% host exposed; see 2026-06-30 correction below)**. Graph capture / C-loop cap at ≤10% and ROCm 7.x re-pays per-node (M12.1). The GDN-corruption fix would be wasted effort. | n/a — closed. Remaining gap is GPU compute, only cuttable by dp4a (fails ja gate) or FLOP/quality loss. |
| P0.3 | Re-baseline the verify work on a keep-drafting route | Stop using the code-only `b1-probe-block-direct-cap32k` as the input for verify-wall work; use `cap32k-recover` (already high acc/out, all categories, ~0.95x AR). It already satisfies the old S5 precondition ("good acc/out, poor tok/s"). | The shootout scoreboard records `cap32k-recover` as the verify-wall starting baseline with per-category acc/out. |

Success for Goal — Part 1: a full-suite artifact whose **best budget keeps
`cap32k-recover`-class per-category acceptance AND drives target layer
passes/output toward ~0.4 or below**, lifting non-code budgets above AR. Only
after that lands do the S1-S3 acceptance/policy probes below become worth
running — until then they will reproduce `cap32k-recover` (acceptance up, tok/s
pinned at ~0.95x AR).

#### P0 RESULTS (2026-06-30, gfx1151) — measured, and they reframe the lever

**P0.1 block-verify cost model (measured).** `scratchpad/p01_block_cost_probe.py`
times `verify_target_block` at fixed sequence position (snapshot/restore), bulk
mode + repack, realistic tokens:

| call | rows | ms | x c1 |
| --- | --- | ---: | ---: |
| c1 step (AR) | 1 | 18.9 | 1.00 |
| block B1 | 2 | 30.0 | 1.58 |
| block B2 | 3 | 36.6 | 1.93 |
| block B3 | 4 | 43.5 | 2.30 |
| block B5 | 6 | 57.3 | 3.03 |

Fit: `block(rows) ≈ 16.7 + 6.82·rows ms`. The block verifier **already does true
single-weight-stream amortization** (one Python layer loop; dense weights read
once) — it does **not** need graph capture. But only ~60% of per-step cost
amortizes: the marginal **6.82 ms/row** decomposes (via `advance_state_only`,
which skips lm-head) into **5.60 ms MoE expert over-read + attn compute** and
**1.23 ms lm-head**. This matches the decode rocprof split (dense GEMV 47%
amortizes; MoE 26% + lm-head 10% are paid per row). **The per-row cost is paid on
every *attempted* row, including rejected drafts** — that waste, not the pass
count, is the bottleneck.

**P0 acceptance (measured, decisive).** Route
`resident-strict-block-direct-nofallback` (strict top-1 + block verify + direct
commit, **no AR fallback so it keeps drafting**), `--scope full`
(`scratchpad/p01-strict-block-nofallback-full.json`), per-category best-budget
acc/out vs llama.cpp B2:

| Category | hipEngine strict-top-1 | llama.cpp B2 | Verdict |
| --- | ---: | ---: | --- |
| code | 0.64 (B3) | 0.627 | match |
| general_en | 0.60 (B3) / 0.556 (B2) | 0.576 | match |
| mixed_ja_en | 0.592 (B2) | 0.599 | match |
| general_ja | 0.394 (B3) | 0.563 | lags (Japanese only) |

**The "0 accepted on non-code" in the retained default was entirely
`--adaptive-ar-fallback` quitting after 2 drafts — not draft quality.** Under
identical strict-top-1 greedy (which is exactly what llama.cpp uses; its
`accept()` only reseeds `pending_h`), hipEngine matches llama.cpp acceptance on 3
of 4 categories. Confirmed: llama.cpp's root acceptance is strict argmax, so
hipEngine's `--root-topk-accept 40` relaxation is **not** apple-to-apple greedy
and is not the parity path; strict top-1 is.

**Why the strict-keep-drafting route is still 0.77× AR at B3** (and the reframed
levers): target time/output = `passes/out 0.418 × 43.5 ms ≈ 18.2 ms` ≈ a full AR
step, plus draft. Two structural costs, both fixable:

1. **Block verify is gated to B≥3** (`can_block_verify` needs
   `len(draft_tokens)+1 ≥ ssm_conv_kernel = 4`), so B1/B2 fall to serial
   (passes/out = 1.0, no amortization), and B3 must attempt **4 rows** even when
   ~2.4 are accepted — paying the 6.82 ms/row over-read on ~1.6 wasted rows/cycle.
   Lever: enable block verify at B1/B2 (2–3 rows).
2. **Draft cost ≈ 4.4 ms/depth** (backed out: total 23.7 ms/out − 18.2 ms target
   = 5.5 ms/out ÷ ... ≈ 4.4 ms/draft step), vs llama.cpp's NextN head ~1.5 ms.
   At B3 that is ~13 ms/cycle of draft. Lever: cut draft cost.
3. **general_ja draft quality** (0.39 vs 0.56) — the one true acceptance gap.

**Reframed P0.2:** the lever is **not** a new fused verifier or graph capture
(amortization already works). It is: (a) allow block verify at B1/B2, (b) reduce
the per-row block over-read (MoE+lm-head) and/or the draft cost, (c) replace
`--adaptive-ar-fallback` with a keep-drafting policy now that acceptance is known
good, (d) close general_ja draft quality. The cost model says: at the measured
block structure with cheap drafts and the measured acceptance, B2 block verify
reaches ≈ AR–1.1× today and clears llama parity once the per-row over-read or
draft cost drops.

#### P0 RETAINED WIN + settled conclusion (2026-06-30)

**Retained:** `--target-block-min-rows 2` promoted to the default route
(`resident-b1-probe-block-direct-cap32k-minrows2`). Full suite: best **B2 56.8
tok/s = 1.0399× AR** (confirm 1.0385×), beating the prior B3 1.0356× default. B2
moved from serial **0.9845×** to block-amortized **1.0399×** (+5.6%); B3
unchanged. Exact (bit-exact vs serial-exact rows 2–3), `apple_to_apple_ok=true`.

**Policy space is now exhausted — acceptance is not the lever.** Two further
full-suite diagnostics settle it:
- Keep-drafting (no fallback, cheap B1 probe) **restored** non-code acceptance
  (en/mixed 0.41, B2 acc/out 0.482 vs default 0.265) yet tok/s **fell to 1.007×**.
- A larger draft cap (98304) was **worse** (1.0276×): costlier drafts, fallback
  still latches.

This confirms P0.1 at the route level: **raising acceptance does not raise tok/s
while per-token verify cost (block over-read 6.82 ms/row + probe) eats the gain.**
The selective-fallback default is faster because it skips verify work on low-yield
prompts. The aggregate is dragged below the amortization threshold by
**general_ja** (draft acc 0.167 capped / ~0.5 full vs llama **0.563**).

**The remaining gap to llama 1.34× is now kernel/model work, not benchmark policy:**
1. **general_ja draft quality + coverage** (highest leverage) — cap32k halves ja
   draft acc (Japanese token IDs > 32768); full vocab ~0.5 (vs llama 0.56) but
   ~4 ms/depth. Needs a cheap full-vocab or CJK-covering draft.
2. **Draft lm-head cost** — full vocab ~3 ms (reads 638 MB Q6_K lm-head);
   cap32k ~0.7 ms but drops CJK. A smaller-quant/shortlist draft lm-head that
   preserves CJK lets ja/mixed use full coverage cheaply.
3. **Per-row block over-read** 5.6 ms/row (MoE distinct-expert, top-8/256) — the
   same constraint llama faces; only a more BW-efficient small-batch MoE verify
   GEMV reduces it.

The S1–S3 acceptance/policy probes in the shootout below are therefore
**confirmed dead-ends** for tok/s (acceptance restored, speed flat/down); skip
them and go straight to the three kernel/model levers above.

**Quantified roadmap (where each lever lands).** Cost model fit to the measured
block structure (`block(rows) ≈ 16.7 + 6.82·rows ms`, c1 = 18.9 ms) and llama's
own B2 (block(3)≈34 ms, draft ~1.5 ms, acc/out 0.598 → 14.86 ms/out = 1.34×):
- **Today:** B2 1.0399× (code-only contribution; en/mixed/ja ≈ AR).
- **+ general_ja draft quality to ~llama (0.56) realized cheaply:** aggregate
  acc/out ≈ 0.58 → B2 ≈ (36.6 + ~1.4)/2.4 ≈ 15.8 ms/out ≈ **1.16×**. This is the
  single highest-leverage lever. Blocker: ja full-vocab draft is ~0.5 draft_acc
  but only 0.13 acc/out when escalated (the chain collapses) AND full-vocab draft
  costs ~4 ms — so it needs BOTH cheaper full-vocab draft AND a draft-quality fix.
- **+ block ~llama (34 vs 36.6 ms) via a more BW-efficient small-batch MoE verify
  GEMV:** closes the rest toward **~1.34×**. (WMMA confirmed slower than
  gemv-decode here; gemv-decode is already near-peak, so this is hard.)
Net: ~1.16× is reachable with the draft levers; the last ~1.16→1.34× is the
hardware-limited MoE verify GEMV. Each is a correctness-gated kernel/model
sub-project (new kernel ⇒ RED test + `kernels/cpu_reference/` gate), not a
benchmark-policy change.

### Next shootout matrix

> **Order note (2026-06-29):** the S5 precondition ("a route with good
> accepted/output but poor tok/s") is **already met** by `cap32k-recover`, so
> the target-verify-wall work in S5 is promoted into "Goal — Part 1" above and
> runs **first**. S1-S3 are demoted: they are acceptance/policy reshuffles
> inside a Pareto frontier already known to sit at ≤ AR on non-code, and should
> only run after the verify wall drops.

Every row below is a full-suite shootout candidate, not a single-prompt probe.
Run `scripts/gguf_ar_mtp_suite.py --scope full` (with a named route for variants)
and compare against both the current hipEngine default artifact and the retained
llama.cpp matrix.

For this shootout, the retained evidence must include category rows. If the
compact suite artifact still records only aggregate `mtp_by_budget`, either copy
the `child_artifacts.mtp_category` summary into `benchmarks/results/` or extend
the suite artifact before promoting a result.

Current hipEngine baseline:
`benchmarks/results/2026-06-29-ar-mtp-suite-full-b1-probe-block-direct-cap32k.json`
(B3 **56.54 tok/s**, **1.0356x AR**, **0.286 accepted/output**,
**0.779 target layer passes/output**).

llama.cpp target:
`benchmarks/results/2026-06-22-llamacpp-35b-mtp-category-off-b1-b5-gfx1151.json`
(B2 **67.29 tok/s**, **1.3423x AR**, **0.598 accepted/output**).

| ID | Candidate | Priority | Hypothesis | Required evidence | Promote / reject rule |
| --- | --- | --- | --- | --- | --- |
| S5 | Target verifier wall reduction (fused B-token verify) | **P0 — do first (see Goal — Part 1)** | One target weight stream verifies the whole `[prev]+drafts` block, cutting target layer passes/output from 0.779 toward llama.cpp's ~0.25-0.40 at unchanged acceptance. | rocprof + full-suite row for a `cap32k-recover`-based block-verify route, reporting per-category acc/out and target passes/output. | Promote if full-suite best beats AR on non-code budgets while keeping `cap32k-recover`-class acceptance; this is the parity lever. |
| S0 | Current default rerun | After P0 | Establish noise band for `resident-b1-probe-block-direct-cap32k` before changing policy. | Full-suite total and category rows for B1-B5; confirm B3 stays around **56.5 tok/s** and **1.03x AR**. | Baseline only. Do not retune from a single rerun unless it reproduces the retained shape. |
| S1 | Non-code rescue after zero strict probe | After P0 (will reproduce `cap32k-recover` until the wall drops) | Keep the code-path B1 probe + B3 direct block, but when a category/prompt gets zero strict accepts, fall back to a cheap root-topK/cap32k B1/B2 route instead of pure AR. | `general_en`, `general_ja`, and `mixed_ja_en` accepted/output must move from **0.000** toward llama.cpp B2's **0.56-0.60** without lowering code B3 below current. | Promote only if full-suite best beats **56.54 tok/s** and no non-code category remains at zero accepted drafts. |
| S2 | B2 direct-block promotion | After P0 | llama.cpp is fastest at B2, so test a direct-commit B2 verifier after the cheap B1 probe rather than jumping to B3. | B2 total tok/s, accepted/output, discarded rows, direct commit rows, target layer passes/output. | Promote if B2 beats the current B3 row or materially raises accepted/output with no total tok/s regression. |
| S3 | B3 promotion threshold sweep | After P0 | The current route's B3 win is code-heavy; try stricter/looser promotion criteria that preserve cheap cap32k drafting but increase safe block use on non-code prompts. | Per-category accepted/output and target passes/output, not just aggregate tok/s. | Keep only if non-code accepted/output rises and aggregate B3 remains above AR and above the current baseline. |
| S4 | llama.cpp lifecycle parity route | After P0 | Re-test context replay + device MTP KV + resident `pending_h`/`verify_h` only with exact row-state commit and prompt catch-up aligned; earlier dense-KV routes collapsed acceptance. | One artifact with per-category acceptance plus a narrow trace showing draft context parity on at least one non-code prompt. | Promote only after full-suite acceptance improves; otherwise record as rejected lifecycle evidence. |

Fill the shootout scoreboard with these columns for every retained or rejected
attempt:

| Candidate | Best budget | Total tok/s | vs AR | Code acc/out | General EN acc/out | General JA acc/out | Mixed acc/out | Target passes/output | Direct commits | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default + min-rows 2 (2026-06-30) | B2 | 56.8 | 1.0399 | — | — | — | — | 0.794 | — | **RETAINED default** (B2 0.9845→1.0399 via 3-row block) |
| Prior default (cap32k) | B3 | 56.54 | 1.0356 | 0.500 | 0.000 | 0.000 | 0.000 | 0.779 | 15 | superseded (code-only win) |
| `cap32k-recover` (P0.3 verify-wall input) | B1 | ~51.7 | ~0.948 | 0.640 | 0.608 | 0.459 | 0.615 | ~1.0 | 0 | acceptance solved, serial (no block) |
| llama.cpp reference | B2 | 67.29 | 1.3423 | 0.627 | 0.576 | 0.563 | 0.599 | inferred 0.402 | n/a | target |

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
   kernel share), ~709 launches/step. A 2026-06-29 rerun measured 19.37 ms host /
   16.95 ms kernel (87.5% kernel share), 708.9 calls/step, with dense Q8_0 GEMV
   48.8% and selected MoE GEMV 24.7% of kernel time. A current artifact measured
   19.03 ms host / 16.76 ms kernel (88.0% kernel share), 708.5 calls/step, with
   dense Q8_0 GEMV 48.9% and selected MoE GEMV 24.0% of kernel time. A post
   bulk-row1-exactness artifact measured 18.65 ms host / 16.75 ms kernel (89.8%
   kernel share), 708.6 calls/step, with dense Q8_0 GEMV 49.0% and selected MoE
   GEMV 24.6% of kernel time. The
   pre-cleanup call-site profile was 18.99 ms host / 16.68 ms kernel with unused
   full-logits D2H.
3. **The `--true-ar-baseline-json` apple-to-apple path is BROKEN.** Since #8
   retired the HIP decode graph, the production AR path emits `decode_path:
   eager_step`, but `gguf_mtp_category_bench.py`'s `TRUE_AR_PRODUCTION_TIMING_REQUIRED`
   (and a parallel speed-claim contract + tests) still demand the retired
   `graph_replay`. So that attach rejects every current AR baseline. The new
   suite sidesteps it (computes the ratio itself); the contracts need a proper
   eager-path fix — tracked in `docs/REFACTOR.md`.

### Per-stage gap vs llama.cpp (AR + MTP pipeline)

Superseded by the final stage ledger at the top, but retained here in the historical
section with current numbers instead of stale single-prompt estimates.

| Pipeline stage | hipEngine | llama.cpp HIP | Gap / status |
| --- | --- | --- | --- |
| Target AR decode (c=1) | **54.95 tok/s**, ~18.2 ms/tok, 762 launches/tok | 51.38 tok/s, 17.26 ms GPU plus ~2.2 ms exposed host, 1632 launches/tok | **hipEngine faster**; AR is not the MTP gap. |
| AR kernel mix | q8_0 attention **42%**, q4_K MoE **21%**, q6_K lm-head **9.6%**, GDN **8%** | `mul_mat_vec_q` dp4a **76.5%**, `mul_mat_vec_f` **5.8%**, `quantize_q8_1` **2.2%** | Different precision/layout regimes; no hidden hipEngine AR deficit. |
| Large lm-head | ~1850 us, **~550 GFLOPS** | Vulkan comparison: 1794 us, **566 GFLOPS** | Large GEMV bandwidth is parity-class. |
| Current block verify | rows=4 **42.40 ms wall**, **38.08 ms GPU**, **875 launches**, **10.2% host exposed** | MTP path deadlocks `rocprofv3` finalize; 4-row proxy matmuls: `mul_mat_vec_q_moe` **40.5%** + `mul_mat_vec_q` **33.8%** | hipEngine verify is GPU-bound; graph/launch collapse is not the lever. |
| Verify GPU breakdown | q8_0 attention **32.7%**, GDN **16.1%**, selected MoE **25.9%**, rowtile lm-head **5.9%**, misc **19.4%** | dp4a/q8_1 matmuls dominate proxy | Exact components are already optimized, unquantizable, or dp4a-gated. |
| Exact MTP throughput | B5 **60.78 tok/s**, **1.1134x**, acc/out **0.535**, passes/out **0.567** | B2 **67.3 tok/s**, ~**1.31x**, acc/out **0.598**, passes/out **0.402** | llama spends fewer target passes/output and uses cheaper dp4a rows. |
| Accuracy-traded dp4a transplant | B5 **61.61 tok/s**, **1.1322x**; ja top-1 **0.700** gate fail | native llama HIP still **67.3 tok/s** | dp4a is not sufficient and not correctness-retainable. |

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
| strict-context route | existing llama.cpp-style prompt replay + device MTP KV with root/sibling top-1 | smoke B3 **42.81 tok/s = 0.780x AR**; partial best B1 **48.69 tok/s = 0.889x AR**, B3 **45.16 = 0.825x AR** | route is a valid diagnostic but not production-competitive; build resident lifecycle abstraction |
| adaptive full-vocab recovery after capped miss | keep cheap capped-vocab draft normally, switch to full vocab after a generic capped zero-accept miss instead of permanent AR fallback | partial route `resident-cap32k-recover`: AR **54.76 tok/s**, best B1 **52.45 tok/s = 0.958x AR**, accepted/output **19/39 = 0.487**; full suite: AR **54.55 tok/s**, best B1 **51.71 tok/s = 0.9478x AR**, accepted/output **78/178 = 0.438**; cap sweep B1 diagnostics peaked around cap18k/24k at **~52.6 tok/s** but still below AR | diagnostic only; B1 throughput improves, but acceptance regresses vs resident top-k40 and the serial verifier route remains bounded by target wall + draft overhead |
| short B1 target block verify with confidence gate | use 2-row target block verify for high-confidence exact B1 drafts, rollback to serial/root-topK on mismatch | direct rows=2 block probe was exact and faster than two serial steps (**32.8 ms vs 39.7 ms**), but partial B1 p=0.8 had 15 attempts/14 hits/1 rollback and regressed to **50.07 tok/s**; p=0.9 had 11/11 hits but still **51.84 tok/s**, below capped recovery **52.45 tok/s** | rejected; savings per hit too small and rollback/noise erases it |
| branch-safe B1 root-topK block verifier | batch `[prev, draft0]`; use row 1 only on strict draft top-1 accept; for root-topK branch/reject restore and replay row 0 unless a captured row-0 direct commit is requested | original restore/replay route smoke `resident-b1-branch-safe-block-cap32k-device-seed` B1 measured AR **54.93 tok/s**, MTP **31.11 tok/s = 0.566x AR**, accepted/output **0.400**; after fixing captured row-0 FP32 `ssm_out` exactness, direct row-0 branch route smoke `resident-b1-branch-safe-direct-cap32k-device-seed` B1 measured AR **54.97 tok/s**, MTP **26.66 tok/s = 0.4849x AR**, accepted/output **0.400** | rejected/default-off diagnostic; direct row-0 commit is now serial-exact, but row 1+ still needs replay and the route is slower than restore/replay, so it is not the amortization path |
| serial-exact verifier row baseline | use the normal token-serial decode scheduler to stage per-row `h_nextn` plus Conv/GDN state and prove direct row commits are exact before optimizing the batched path | focused wrong-branch gate passes: direct row-0 commit after `[prev, wrong_child]` matches serial hidden/state bit-for-bit and the corrective next step remains exact | correctness oracle/scaffold only; it does not amortize target weight loads and is not a speed route |
| hybrid strict-block/cap32k route | begin with strict top-1 block-promotion probe, then fall back generically to root-topK B1 + cap32k recovery if probe acceptance is weak | smoke B3 **48.94 tok/s = 0.890x AR**; partial best B3 **54.63 tok/s = 0.9973x AR** looked close, but full suite dropped to AR **54.58 tok/s**, best B3 **50.91 tok/s = 0.9328x AR**, B4 **48.94 = 0.8967x**, B5 **48.52 = 0.8890x**, accepted/output **94/194 = 0.485** | rejected/default-off diagnostic; partial was not predictive, and the route is worse than cap32k recovery B1 full-suite **51.71 tok/s = 0.9478x AR** |
| strict-context/block `draft_p_min=0.8` selector | suppress weak resident drafts before expensive strict block verification | smoke route `resident-strict-context-block-pmin08`: AR **55.00 tok/s**, B3 **38.44 tok/s = 0.6991x AR**, accepted/output **0.571** | rejected/default-off diagnostic; probability gating cannot fix strict block economics when low-accept cycles still pay target block work |
| direct verifier row-state commit | adopt llama.cpp-style verifier-row materialization for strict block verification: capture per-row GGUF linear-attention Conv/GDN state and commit the accepted row without rollback replay | row-0 wrong-branch commit is serial-exact after aligning captured `ssm_out` to the serial FP32 activation path; row 1 is serial-exact in `target-block-verify-mode=native` after fixing the native row-serial full-attention verifier to use absolute continuation positions and capture row states. Default `bulk` row 1 is now exact for short verifier blocks (`end < 1024`) after replacing the drifting suffix full-attention prefill reduction with a c1-exact row-batch decode context path and fixing the batch context kernel to honor shared physical block IDs. Standalone smoke remained negative: bulk hybrid direct B3 **49.01 tok/s = 0.893x AR**; native hybrid direct B3 **48.17 tok/s = 0.875x AR**; old pure strict B3 **37.20 tok/s = 0.678x AR**; B1 branch-safe direct **26.66 tok/s = 0.4849x AR**. | exactness scaffold retained; direct commit by itself was not a speed route, but it is required by the later B1-probe/block-direct/cap32k route that beats AR |
| resident device hidden seed | adopt llama.cpp-style resident `pending_h` and avoid target hidden-seed D2H/H2D before resident draft | full suite route `resident-cap32k-device-seed`: AR **54.59 tok/s**, best B1 **52.08 tok/s = 0.9540x AR**, accepted/output **78/178 = 0.438**; cap32k recovery control was B1 **51.71 tok/s = 0.9478x AR** with the same acceptance | retained default-off structural diagnostic; +0.7% over cap32k recovery, not enough to beat AR; confirms lifecycle direction but remaining lever must cut target verifier work per visible token |
| B1-probe/block-direct/cap32k route | use a cheap strict B1 cap32k probe to avoid non-code B3 block waste, then promote to direct-commit B3 block verification after a full B1 accept | full suite route `resident-b1-probe-block-direct-cap32k`: AR **54.59 tok/s**, best B3 **56.54 tok/s = 1.0356x AR**, accepted/output **40/140 = 0.286**, draft acceptance **0.645**, target layer passes **0.779/output**, direct commit rows **15**, replay rows **0** | retained default route; closes the same-protocol AR-beat gate, but still trails llama.cpp B2 **67.29 tok/s** by ~19% relative tok/s |
| resident device seed + dense draft KV | combine resident `pending_h` with device-resident draft KV and commit accepted verifier rows from staged device hidden rows instead of host hidden arrays | route `resident-cap32k-device-seed-kv`: B3 smoke AR **54.66 tok/s**, MTP **38.94 tok/s = 0.7124x AR**, draft_acceptance **0.032**; B1 smoke AR **54.92 tok/s**, MTP **39.73 tok/s = 0.7235x AR**, draft_acceptance **0.017** | rejected/default-off route; keep verifier-row staging + device-base KV commit primitives, but dense draft KV without llama.cpp prompt/context catch-up changes drafts and collapses acceptance |
| context replay + resident device seed | combine llama.cpp shifted prompt catch-up, device MTP KV, resident `pending_h`, staged verifier rows, and cap32k recovery | route `resident-context-cap32k-device-seed`: B1 smoke AR **54.93 tok/s**, MTP **50.84 tok/s = 0.9257x AR**, accepted/output **0.400**; B3 smoke AR **54.87 tok/s**, MTP **46.97 tok/s = 0.856x AR**, accepted/output **0.571** | rejected/default-off structural diagnostic; prompt/context lifecycle is now wired, but serial target verification still runs one target pass per visible token and draft overhead keeps it below true AR |
| dispatch-resolve cache (#9) | ~15 µs/launch host | landed | kept |
| X8 selected-down repack (Q5/Q6) | sidecar-free dp4a layout | mixed; ≤ default B3 | diagnostic |
| T16 Q4/Q5 selected dp4a variants | faster MoE GEMV | 1.04–1.10× iso, flat/regress B3 | diagnostic gates |
| 32k draft vocab cap | ~5 ms/cycle draft | prompt-sensitive | diagnostic |
| adaptive AR fallback after zero-accept | avoid catastrophic block replay | robust full-suite route | **kept (production selector)** |
| HIP graph capture of verify | collapse the ~875 launches | **refuted 2026-06-30: block verify is GPU-bound (10.2% host exposed); ROCm 7.x re-pays per-node (M12.1)** | **rejected — not a lever** |

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
3. **Compare to the committed hipEngine baseline and the llama.cpp target:**
   hipEngine current default is
   `benchmarks/results/2026-06-29-ar-mtp-suite-full-b1-probe-block-direct-cap32k.json`
   (AR 54.59 tok/s; MTP B3 56.54 tok/s = 1.0356× AR;
   `mtp_beats_ar=true`). The external target is
   `benchmarks/results/2026-06-22-llamacpp-35b-mtp-category-off-b1-b5-gfx1151.json`
   (llama.cpp B2 67.29 tok/s = 1.3423× AR). Diff the `verdict`,
   per-budget `vs_ar_ratio`/`accepted_per_output`, and the per-category
   accepted/output shootout columns. The suite asserts `apple_to_apple_ok=true`
   (same decode protocol + prompt-set hashes) — if it is false, the comparison is
   invalid, full stop.
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
   16.56 ms kernel per target step, 89% kernel share; latest current rerun
   19.03 ms host / 16.76 ms kernel, 88.0% kernel share). Do not start with a
   launch-collapse project unless a new route/profile proves host residual is
   back on the critical path.
2. **Treat strict-context as a diagnostic baseline, not the next optimization
   target.** The existing `resident-strict-context` route records
   `--resident-mtp-draft --root-topk-accept 1 --sibling-topk-accept 1
   --mtp-context-replay --mtp-device-kv-cache --no-target-block-verify`.
   Initial evidence: smoke B3 is **42.81 tok/s = 0.780× AR**; partial best is B1
   **48.69 tok/s = 0.889× AR** with B3 accepted/output **0.697** but only
   **0.825× AR**. A full run is useful after lifecycle changes, but the existing
   diagnostic hooks do **not** generalize into a competitive route by
   themselves.
3. **Port the llama.cpp target-memory pattern, not another micro-lever.**
   `Qwen35GGUFMTPContext` already owns the seed lifecycle (`pending_h` /
   verifier hidden rows / `accept()` reseed). The missing adoption target is a
   branch-safe transactional target verifier with llama.cpp-like recurrent
   rollback slots: run `[prev]+drafts` through scratch target state, materialize
   exact per-row `h_nextn` plus GGUF Conv/GDN state, and advance the resident
   target to the accepted row without serial restore/replay. In llama.cpp terms,
   this is the `llama_memory_recurrent::seq_rm()` / bounded `n_rs_seq` behavior,
   not merely a renamed draft context. Success is lower target passes per
   visible token on the full category suite, not a wider candidate-rank or
   confidence diagnostic.
   Source anchor: llama.cpp commit `6e9007ae61f4e994c27484759caac6ef2aa32b30`
   defines this lifecycle in `common/speculative.h` (`common_speculative_process`,
   `common_speculative_draft`, `common_speculative_accept`), implements the MTP
   state in `common/speculative.cpp::common_speculative_impl_draft_mtp`
   (`pending_h`, `verify_h`, `last_n_drafted`), and invokes it from
   `tools/server/server-context.cpp` (`draft()` before target batch construction,
   `process()` after target decode, `accept()` after accepted-row sampling).
   The same checkout builds Qwen3.5/Qwen3.6 MTP as a first-class
   `qwen35moe::graph_mtp` and exports target/draft `t_h_nextn`; the retained
   llama.cpp speed row is plain `--spec-type draft-mtp --spec-draft-n-max N`, not
   an ngram-stack or prompt-history trick.
   The capped-vocab recovery, hybrid strict-block, direct row-state commit, and
   device hidden-seed probes confirm this direction. The payoff became positive
   only after combining them as `resident-b1-probe-block-direct-cap32k`: full
   suite B3 **56.54 tok/s = 1.0356× AR**, target layer passes
   **0.779/output**, replay rows **0**. The route is now the baseline to improve,
   not a reason to restart from micro-kernel work.
4. **Close the llama.cpp parity gap from the retained route.** The remaining
   lever is better verifier amortization and acceptance economics: make B2/B3
   block promotion pay on more prompts, keep direct commits exact without serial
   fallback waste, and preserve the cheap cap32k draft cost. The concrete target
   is moving from B3 **56.54 tok/s** toward llama.cpp's B2 **67.29 tok/s** on
   the same full category suite. **Order superseded by "Goal — Part 1" above:**
   the verify-wall reduction (old S5) runs first because `cap32k-recover`
   already meets its precondition; non-code rescue / B2 / B3 policy sweeps
   (S1-S3) run only after the wall drops.
5. **~~The fused multi-token target verifier is now P0~~ — CLOSED/REFUTED
   2026-06-30.** The "collapse 875 launches into one weight stream" lever assumed
   a host-launch floor. Measured: the block verify is **GPU-kernel-bound** (38.1 ms
   GPU / 42.4 ms wall, 10.2% host exposed). HIP graph capture / C-loop / GDN-fix
   are **not levers** (≤10% ceiling; ROCm 7.x re-pays per-node, M12.1). llama's
   "~9 ms fused graph" advantage is its **dp4a/q8_1 cheaper kernels**, not graph
   topology — and dp4a fails hipEngine's ja correctness gate (top-1 0.700). The
   exact-precision GPU-compute ceiling is reached at `1.1134×`. See P0.2 + the
   2026-06-30 correction.
6. **Use llama.cpp parity, not AR-beat, as the next retained speed gate.** The
   current route already satisfies `mtp_beats_ar=true` on `--scope full`. The
   next retained claim should either move materially toward llama.cpp's
   **67.29 tok/s B2** row with the same no-gaming full-suite protocol, or record
   why a directly adopted llama.cpp pattern fails in hipEngine.
7. **Fix the stale AR-baseline contracts** (`TRUE_AR_PRODUCTION_TIMING_REQUIRED`
   + speed-claim contract + tests) to the eager path so the category bench's own
   `--true-ar-baseline-json` comparison works again (REFACTOR.md).

### Don't re-chase (closed lines of work)

GEMV instruction efficiency (dp4a/rowtile), split-K, MoE-FFN graph, cache
hints, deferred hidden-seed D2H copies, the one-block device top-k40 extension,
cap-only/rootK sweeps, resident device hidden-seed copy avoidance by itself,
device-seed + dense draft-KV without context catch-up, context replay + device
seed under serial target verification, strict block `draft_p_min` gating, short
B1 confidence-gated target block verify, and branch-safe B1 root-topK block
verify are all measured too small, acceptance-regressive, or negative e2e and
are not the lever. Direct row-state commit by itself was not a speed win, but it
is now part of the retained `resident-b1-probe-block-direct-cap32k` route; do not
re-test it as isolated exactness scaffolding unless a correctness regression
appears.
The per-kernel GEMV bandwidth is already near-peak. Kernel micro-optimization is
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

**2026-06-30 correction — the 2026-06-28 "host-bound" claim was the OLD serial
per-row route; the CURRENT block verify is GPU-kernel-BOUND (~90%).**  Decisive
differential measurement (`scratchpad/launch_overhead_decomp_blockverify.py`,
wall = clean `perf_counter` over the block loop, GPU = `rocprofv3 --kernel-trace`
DurationNs sum, both differenced over N=8 vs N=32 to cancel prefill) on the
production `verify_target_block(rows=4, bulk)` path with the landed rowtile
lm-head:

| per-block (rows=4) | ms |
| --- | --- |
| wall (host+GPU overlapped+sync) | **42.40** |
| GPU kernel-sum | **38.08** |
| host EXPOSED (wall − GPU) | **4.33 (10.2%)** |

A standalone async-issue probe (`scratchpad/launch_overhead_decomp.py`) confirms
per-kernel-launch dispatch is **~12 µs**, so 875 launches ≈ **10.5 ms** of host
issue — fully overlapped behind the 38 ms of GPU work, leaving only ~4.3 ms
exposed.  The "~54 ms host floor" did not reproduce on the block path; it was the
serial route's per-row-synced dispatch.  **Consequences, all evidence-backed:**

- **Graph capture / fused draft+verify graph is REFUTED as a lever** (and the
  GDN-corruption fix it requires would be wasted effort): only 10.2% host is
  exposed, and ROCm 7.x `hipGraphLaunch` re-pays per-node overhead at ~1000-node
  DAGs (M12.1 `2026-05-22-...graph-capture-diagnostic.json`, L3/L13 in DFLASH).
  Best case — eliminating *all* exposed host — caps the verify at 38.1 ms (≈ +11%
  → ~1.22× absolute ceiling), and that is physically unreachable.
- **C-loop / Python dispatch memoization is REFUTED**: same ≤10% host ceiling.
- **The 38.1 ms GPU kernel-sum IS the wall.**  Only cheaper kernels cut it:
  pipeline-wide **dp4a/q8_1** (REFUTED — ja greedy top-1 0.700 < 0.90 gate, even
  MoE-selected, `scratchpad/dp4a-verify-full.json`) or fewer FLOPs (quality loss).
- The **lm-head rowtile** (landed, bit-exact, `1.0534×→1.1134×`) captured the
  only shared-weight GPU amortization (all verify rows read the same head).  The
  MoE is per-row **disjoint**-weight (top-8 of 256, rarely shared across rows) →
  no cross-row amortization (grouping de-risk: all-distinct only 1.40–1.54× of
  all-same, L2-served) → near its efficient exact point.

**Net:** hipEngine's GGUF block verify is at its **exact-precision GPU-compute
ceiling**.  The residual gap to llama 1.342× is purely llama's pipeline-wide
dp4a/q8_1 precision tradeoff, which violates hipEngine's ja correctness gate.
hipEngine reaches `1.1134×` (60.8 tok/s, 90.3% of llama's 67.3) while **beating
llama on AR** (54.6 vs 50.1 tok/s) and on precision (exact; passes the ja gate
llama's recipe fails).  Closing the rest is not a config/kernel/graph lever — it
requires accepting llama's precision loss, which the stated correctness guard
forbids.

Success criterion: same-protocol full-suite row improves all three: raw weighted
decode tok/s, accepted/output, and strict draft acceptance.

## Bottom line

**2026-06-30 FINAL — investigation complete, root cause confirmed bottom-up.**
Retained win: bit-exact Q6_K T16 rowtile lm-head kernel, GGUF MTP `1.0534x ->
1.1134x AR` (60.8 tok/s = 90.3% of llama.cpp's 67.3; hipEngine AR 54.6 > llama AR
50.1). Every llama MTP pipeline lever was implemented/tested and either shipped or
refuted with committed full-suite artifacts: dp4a (only -4% on the GPU-bound verify
AND fails the ja gate, top-1 0.700), HIP graph capture (verify is 90% GPU-bound,
ROCm re-pays per-node, M12.1), MoE grouping (L2-served), vocab-cap recover (-1.3%),
p_min 0/0.3/0.5 (0.5 optimal), probe/no-probe (probe optimal), budgets 1-8 (plateau
at B5), generation length (uplift stable), and context (validated CORRECT via a new
mtp_dense_attn_f32 gate; the model's NextN simply does not benefit from it). The
llama baseline was audited apples-to-apples (matching metric defs; gap is real).

**The absolute number 67.3 tok/s is unreachable on hipEngine in ANY precision
regime**, not merely the exact one. Matching it needs a `1.233x` uplift over
hipEngine's 54.6 AR; measured uplift ceilings are exact `1.114x` and dp4a `~1.13x`
(prior session) - both below 1.233x. llama reaches 67.3 via a *slower* AR (50.1) x
a *higher* uplift (1.342x); hipEngine's faster-AR / exact-precision profile has a
different optimum (higher AR, lower uplift). A cross-tool draft-logit comparison vs
a captured llama oracle (`benchmarks/fixtures/llamacpp_mtp_explain_concept_draft_trace.json`)
confirmed the residual is the exact-vs-dp4a PRECISION REGIME manifesting through the
whole speculative economy (seed hiddens, draft logits, verification targets all
differ because hipEngine is exact and llama is dp4a) - NOT a hipEngine bug. Closing
to llama requires adopting llama's dp4a regime end-to-end, which fails hipEngine's
ja correctness gate (the stated guard) and still would not reach 67.3 given
hipEngine's already-faster exact AR. 1.1134x is the exact-precision optimum.

---


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
