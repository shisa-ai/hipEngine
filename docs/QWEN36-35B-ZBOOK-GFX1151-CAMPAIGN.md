# Qwen3.6-35B-A3B ZBook gfx1151 Campaign

Status: **active; all four artifacts are admitted, the exact-artifact quality
gate and hipEngine PARO implementation-correctness gate are complete, and Z1
quality is closed. Labeled quality-traded profiling may proceed.** The campaign
is specific to the HP ZBook Ultra G1a at its current 60/60/45 W limits. It must
not reuse absolute throughput from the higher-power 120/160/140 W Radeon 8060S host.

The objective is to answer three questions with matched local evidence:

1. How fast are current hipEngine GGUF Q4_K_M and PARO W4 on this ZBook versus
   current clean llama.cpp HIP and Vulkan?
2. How much quality does GGUF Q4_K_M, PARO W4, and ROCmFP4 STRIX_LEAN lose
   against the same Qwen BF16 source?
3. If ROCmFP4 is both accurate enough and faster, which measured mechanism is
   worth implementing in hipEngine rather than copying a foreign runtime?

This remains a campaign plan, not a performance claim. The exact-artifact
quality result is retained; no new ZBook throughput row has been retained yet.

Related authorities:

- [`BENCHMARK.md`](BENCHMARK.md) and [`TESTING.md`](TESTING.md) — evidence,
  anti-gaming, and correctness contracts.
- [`TUNING-gfx1151.md`](TUNING-gfx1151.md) and
  [`ROOFLINE-gfx1151.md`](ROOFLINE-gfx1151.md) — gfx1151 profiling and
  bandwidth/occupancy rules.
- [`MTP-LLAMACPP-PARITY.md`](MTP-LLAMACPP-PARITY.md) and
  [`NATIVE_SPEC_CYCLE.md`](NATIVE_SPEC_CYCLE.md) — complete-cycle MTP timing
  and true-AR denominator rules.
- [`QWEN38-27B-GFX1151-CAMPAIGN.md`](QWEN38-27B-GFX1151-CAMPAIGN.md) — the
  nearest current campaign template; its model and speed rows do not transfer.
- `/home/lhl/ROCmFPX` at the pinned revision below — read-only implementation
  reference. Development and retained kernels remain in this repository.

---

## 1. Fixed platform identity

| Field | ZBook campaign value |
| --- | --- |
| Host | HP ZBook Ultra G1a 14-inch Mobile Workstation |
| CPU/APU | AMD Ryzen AI MAX+ PRO 395 |
| GPU | Radeon 8060S, `gfx1151`, PCI `0000:c3:00.0` |
| Compute geometry | 40 CUs, wave32 default, unified LPDDR5X memory |
| System memory | 125.1 GiB |
| Kernel | `7.1.8-1-cachyos` at campaign creation |
| HIP compiler/runtime | PyPI ROCm/TheRock `7.15.0-0000000`; exact package and compiler hashes must be refreshed in Z0 |
| Vulkan | RADV, Mesa `26.1.6-arch3.1` at campaign creation |
| Power limits | STAPM 60 W, fast PPT 60 W, slow PPT 45 W |
| Power/clock policy | AC power; TuneD `accelerator-performance`; amdgpu `sched_policy=0`; DPM `auto` |
| TTM page limit | 16,392,564 pages (about 62.5 GiB); do not silently raise it for a benchmark |

The 60/60/45 W limits are part of every result's workload identity. Record
`ryzenadj -i` before and after each retained block. A row from the other
120/160/140 W gfx1151 system is historical context only, even when every other
field matches.

Do not alter firmware memory carve-outs, TTM limits, power limits, IOMMU,
clocking, fan policy, or thermal controls in this campaign without a separately
approved and labelled system-configuration experiment.

### 1.1 Idle-state gate

Run one engine at a time. Before a retained block:

- all model downloads, hash jobs, builds, JIT compilation, and profiler jobs
  have exited;
- `/dev/kfd` has no unrelated owner and no llama/hipEngine server is alive;
- CPU and GPU utilization have returned to idle;
- the machine is on AC power and the three power limits still read 60/60/45 W;
- the initial temperature and available-memory sample are recorded;
- model files are on the same local storage class and are warmed consistently.

Counter-rotate engine order. Use at least one warmup and five measured samples
for final rows. A thermally drifting block is diagnostic even if its median is
fast.

---

## 2. Fixed model set and provenance

All four artifacts derive from `Qwen/Qwen3.6-35B-A3B`. Cross-quant rows are
useful product comparisons only after quality passes; they are not same-file
engine comparisons.

| Role | Artifact | Pinned identity | Admission state |
| --- | --- | --- | --- |
| BF16 oracle | `Qwen/Qwen3.6-35B-A3B` | HF revision `995ad96eacd98c81ed38be0c5b274b04031597b0`; 26 shards / 71,903,776,776 file bytes / 71,903,645,408 tensor bytes; Apache-2.0 | **Admitted 2026-08-15 UTC:** every shard SHA-256, safetensors index/header map, 1,045 BF16 tensors, CPU Transformers full-logit forward, and greedy smoke passed. hipEngine can ingest its metadata but has no BF16 Qwen generation registration; this remains an external CPU quality oracle. Evidence: [`2026-08-15-zbook-qwen36-bf16-gguf-cross-runtime-correctness.json`](../benchmarks/results/2026-08-15-zbook-qwen36-bf16-gguf-cross-runtime-correctness.json). |
| GGUF baseline | `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`; 22,663,387,424 bytes; SHA-256 `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b` | **Admitted 2026-08-15 UTC:** local hash passed; 753-tensor MTP-bearing file; strict AR map, MTP inventory, all-type CPU dequant smoke, ten-prompt tokenizer roundtrip, and a llama.cpp exact-ID spot check passed. Evidence: [`2026-08-15-zbook-qwen36-35b-q4km-admission.json`](../benchmarks/results/2026-08-15-zbook-qwen36-35b-q4km-admission.json). |
| PARO W4 | `shisa-ai/Qwen3.6-35B-A3B-PARO-packed` | HF revision `437eba06df05aad71a4dacdcaf3fff70ae1ee8a1`; `model.safetensors` 20,474,495,512 bytes, SHA-256 `a5c9100b17846ff0b2b507dc16dfc3ff1d622adbfc4782f30b4f1b9fac58cc60`; Apache-2.0 | **Admitted and runtime-correct 2026-08-16 UTC:** exact-artifact BF16-relative mean KL/top-1 is `0.027038/92.222%`; exact-checkpoint ParoQuant-vs-hipEngine is mean/max KL `0.001151/0.023016`, top-1 `98.889%`, and repeat bit-identical. Verdict: **quality-traded** versus Q4_K_M; speed rows require that label. |
| ROCmFP4 | `gsrunion/Qwen3.6-35B-A3B-ROCmFP4-STRIX_LEAN-GGUF` | HF revision `f3be5a9c166640f973213d9077ec637ef0875da0`; 19,046,930,720-byte GGUF, SHA-256 `703a0e4af8f2d1e9ecb50f1c3507d7344189a0eb5dbab4796ff69261a47cb03b`; Apache-2.0 | **Admitted 2026-08-16 UTC:** transfer/hash and ROCmFPX HIP full-logit capture passed. Exact-artifact mean KL is `0.045984`, top-1 `97.778%`; verdict: **quality-traded** versus matched-runtime Q4_K_M. |

`plunderstruck/...embF16-headQ6.gguf` and the public ROCmFP4 `FAST` artifact are
not the campaign's binding ROCmFP4 model: they use different protected tensors
or a different recipe. They may be screened later only as explicit additional
quants after the four-way baseline closes.

### 2.1 Token and architecture identity

Z0 must record for every artifact:

- full model/file hashes and snapshot revision;
- tokenizer, vocabulary, special-token, chat-template, and EOS identity;
- model geometry, layer mix, expert count/top-k, MTP/NextN inventory, and
  context declaration;
- tensor count, per-type count/bytes, and file type metadata;
- whether embeddings/output are tied and their physical tensor types.

Render the ten committed category prompts once with the BF16 tokenizer and
store raw token IDs plus hashes in a compact fixture. Every engine consumes the
same IDs; no engine is allowed to independently reinterpret chat text for a
binding comparison.

### 2.2 Initial cross-runtime load smoke

The first representative smoke is retained in
[`2026-08-15-zbook-qwen36-bf16-gguf-cross-runtime-correctness.json`](../benchmarks/results/2026-08-15-zbook-qwen36-bf16-gguf-cross-runtime-correctness.json).
It is an intake checkpoint, not the full ten-prompt quality gate or a speed
claim.

- Every runtime consumed the same 40 token IDs rendered by the official BF16
  template for `general_en_explain`. The native official and Unsloth GGUF chat
  templates differ, so the comparison did not let each runtime render text
  independently.
- The BF16 CPU oracle, hipEngine Q4_K_M, llama.cpp HIP, and llama.cpp Vulkan all
  selected token `44812` on the first row. hipEngine Q4_K_M's one-row
  `KL(BF16 || Q4_K_M)` was `0.00385084`, with finite full-vocabulary logits.
- hipEngine's public GGUF route was deterministic across fresh processes and
  matched llama.cpp HIP on all eight generated IDs.
- llama.cpp Vulkan matched the first six generated IDs, then selected a close
  alternative. On that shared context Vulkan preferred `45239` over `10813` by
  `0.17485` log-prob; llama.cpp HIP server preferred the same token by only
  `0.00378`, while HIP completion and hipEngine selected `10813`. Close argmax
  changes also occurred between llama.cpp HIP execution shapes.
- The HIP and Vulkan llama.cpp smoke binaries were eight upstream commits apart
  (`a94d563ed` versus `1d2869c6e`); those intervening commits do not touch the
  tested kernels, but the binding gate must rebuild both at one revision.

Verdict: **preliminary pass with Vulkan numerical follow-up**. This rules out a
gross load/tokenizer/state failure and strongly supports the hipEngine HIP GGUF
path, but does not replace teacher-forced full-logit rows over the complete
category suite.

---

## 3. ROCmFPX audit boundary

The read-only reference checkout is:

| Field | Value |
| --- | --- |
| Path | `/home/lhl/ROCmFPX` |
| Current head | `0d313da1849f73c5a7f8c5f7e5b8d7d278fbb69d` |
| Upstream llama.cpp ancestor | `15586e2d7165570fb3aa7c26e0d442e289ef69de` (b10297 plus one commit) |
| Code license | MIT |
| Source audit | `python3 scripts/check-rocmfpx-preservation.py` passes at the audited head |

ROCmFP4 is a real custom GGUF weight format, not native FP4 matrix-core
execution:

- `Q4_0_ROCMFP4` stores 32 weights in 18 bytes: 16 packed Codebook10 nibbles
  plus two finite unsigned UE4M3 half-scales, one per 16 weights (4.50 bpw).
- `Q4_0_ROCMFP4_FAST` stores one UE4M3 scale per 32 weights in 17 bytes
  (4.25 bpw).
- STRIX_LEAN intends FAST dense tensors, dual-scale attention K/V, and Q5_K
  embedding/output protection.
- CPU reference/quantization lives under `ggml/rocmfp4/`. HIP uses custom
  MMVQ/MMQ, Codebook10-to-int8 DP4A expansion, conversion/get-rows/copy, and
  vector FlashAttention paths. Vulkan supplies generated matvec/MMQ,
  dequant/copy/SET_ROWS, and scalar FlashAttention paths.
- The speed premise is fewer streamed bytes plus format-specific integer dot
  and decode paths. The tree explicitly does **not** claim native FP4 WMMA.

### 3.1 Current-source quantizer blocker

The audited current `main` runtime can load the pinned prebuilt artifact, but
it must not be used to create a fresh STRIX_LEAN reference until this blocker
is resolved upstream.

A mechanical comparison with pre-sync source snapshot `4e8f35aef` shows that
the b10297 synchronization removed the `src/llama-quant.cpp` branches which:

- protected STRIX/STRIX_LEAN attention K/V with dual-scale ROCmFP4; and
- selected Q5_K token/output protection for STRIX_LEAN.

Current head retains the ftype names and default FAST mapping, so the preset can
be accepted while no longer producing its documented tensor-role mixture. The
current preservation script does not test this semantic contract. Z0 therefore
must inspect the downloaded artifact's actual tensor census and record this as
an external-source blocker; hipEngine does not repair the read-only checkout.

### 3.2 Historical evidence is hypothesis-only

ROCmFPX publishes 35B STRIX_LEAN rows around 1301.21 pp512 / 66.42 tg128 on
HIP and 1200.81 / 76.71 on Vulkan, plus later single-prompt MTP rows of 116.1
Vulkan and 106.2 HIP tok/s. Those runs used another machine/power envelope and
mostly `llama-bench` or single-prompt protocols. They nominate profiling
questions but are not ZBook denominators or retainable MTP evidence.

---

## 4. Definition of done

The baseline campaign closes only when all of the following are true:

1. **Identity:** all model, tokenizer, source, binary, driver, power, and
   workload identities are pinned and machine-readable.
2. **Quality:** the exact-artifact BF16-teacher full-logit gate is complete for
   GGUF Q4_K_M, PARO W4, and ROCmFP4 STRIX_LEAN. Each quant has a predeclared
   `Q4-equivalent` or `quality-traded` verdict, category breakdown, and finite
   rows. Quantization loss is not confused with implementation correctness.
3. **Same-file engine baseline:** hipEngine, clean llama.cpp HIP, and clean
   llama.cpp Vulkan run the exact Q4_K_M file at every declared shape with
   matched raw token IDs and timing scopes.
4. **Cross-quant product baseline:** hipEngine PARO and ROCmFPX HIP/Vulkan
   STRIX_LEAN rows use the same prompt/decode shapes and disclose that model
   bytes and runtime both differ.
5. **MTP honesty:** every MTP claim uses the complete ten-prompt category suite,
   six-train/four-heldout split, and a true no-MTP AR denominator from the same
   runtime. All greedy output IDs match that runtime's AR output.
6. **Profiles:** each leading route has a reconciled device/host ledger within
   10% of complete wall, or an explicit measured explanation of overlap.
7. **Memory:** file bytes, model residency, process GTT delta, RSS, system
   available delta, peak transient, and teardown are reported separately.
8. **Port admission:** no external mechanism enters hipEngine without a
   measured >=1% complete-request ceiling, an in-tree CPU oracle/RED fixture,
   four-axis registration, operation-complete fallback, and implementation
   correctness of maximum KL <= 0.05 plus at least 90% top-1 agreement versus
   `kernels/cpu_reference/` on fixture inputs.
9. **Durability:** retained rows update compact artifacts, worklog,
   `benchmarks/README.md`, and `benchmarks/CHANGELOG.md` in the same logical
   evidence unit.

Passing quality does not mean all quants are equally accurate. Report KL
distribution, top-1, teacher-trajectory NLL/PPL ratio, file size, and speed
together; never collapse them into one unlabeled score.

---

## 5. Quality protocol

### Q0 — Build one engine-neutral fixture

Use `benchmarks/prompts/mtpbench-code-general-ja.jsonl`, preserving all four
categories and heldouts. The BF16 oracle produces:

- exact rendered raw token IDs for every prompt;
- one free-running greedy continuation per prompt;
- teacher-forced probe prefixes consisting of the final prompt row plus at
  least eight BF16 continuation rows;
- fixture/source/tokenizer hashes.

Candidates consume the BF16 continuation tokens rather than their own sampled
tokens. This compares distributions at identical contexts and prevents early
argmax drift from changing the question.

Q0 is implemented under [`benchmarks/quant/`](../benchmarks/quant/) and
`scripts/quant_quality/`. Its NumPy metric/manifest path was RED-tested; the
Torch BF16/ParoQuant code remains an optional boundary tool and adds no Torch to
modules reached by `hipengine.LLM.generate()`. Large logits stay outside Git;
compact summaries remain under `benchmarks/results/`.

### Q1 — Full-logit metrics

Capture FP32 full-vocabulary logits from:

- BF16 HF reference;
- hipEngine PARO W4;
- hipEngine and llama.cpp Q4_K_M as a runtime cross-check;
- ROCmFPX STRIX_LEAN on its selected backend.

For every quant vs BF16 report:

- `KL(BF16 || candidate)`: mean, p95, and maximum;
- top-1 agreement: overall, train, heldout, and each category;
- top-5 overlap and first mismatch position as diagnostics;
- finite row count and maximum absolute logit difference;
- repeat determinism and exact raw-token identity.

Freeze the quant-quality decision rule before looking at candidate results.
Q4_K_M is the product-quality baseline. PARO or ROCmFP4 is `Q4-equivalent`
only when bootstrap 95% confidence intervals support all three non-inferiority
margins: top-1 is no more than 2 percentage points below Q4_K_M, mean KL is no
more than 0.005 above Q4_K_M, and teacher-trajectory PPL/BF16 is no more than
0.01 above the Q4_K_M ratio. Otherwise label it `quality-traded` and still
report the measured tradeoff. Per-category rows are mandatory and can veto
equivalence when one category is an obvious outlier.

#### Q1 checkpoint — 2026-08-16

The binding ten-prompt / 90-row exact-artifact gate is retained in
[`2026-08-16-zbook-qwen36-quant-quality.json`](../benchmarks/results/2026-08-16-zbook-qwen36-quant-quality.json):

- Matched ROCmFPX HIP Q4_K_M is mean/p95/max KL
  `0.013713/0.067523/0.269131`, top-1 `92.222%`.
- ROCmFP4 is `0.045984/0.205053/1.272484`, top-1 `97.778%`, and 15.96%
  smaller. Its paired prompt-block mean-KL delta 95% CI is
  `[+0.00690,+0.07168]`; Japanese mean KL `0.135091` vetoes equivalence.
  Verdict: **quality-traded**.
- hipEngine Q4_K_M is mean KL `0.011807`, top-1 `95.556%`; the same-quant
  ROCmFPX-HIP-to-hipEngine control is mean/max KL `0.005550/0.136877`, so
  runtime-shape attribution remains open despite passing top-1.
- PARO's packed-runtime bug is fixed. The first divergent layer-0 operation was
  GDN: packed safetensors retain Transformers' grouped V heads, while hipEngine
  used the tiled modulo mapping valid only after llama.cpp GGUF conversion.
  Exact-checkpoint ParoQuant-vs-hipEngine is now mean/max KL
  `0.001151/0.023016`, top-1 `98.889%`, and two 90-row captures are
  bit-identical. BF16-relative mean KL/top-1 is `0.027038/92.222%`, but paired
  Q4_K_M noninferiority still fails all gates. Verdict:
  **implementation-correct, quality-traded**.

This exact-artifact result closes the campaign's Z1 quality gate. Its 90
teacher-forced positions are not held-out-corpus PPL or downstream task
accuracy; larger exact-artifact suites may be added as supplemental evidence.

These BF16-relative margins characterize quantization quality. They do not
replace the project gate for a new implementation: the candidate kernel/runtime
must separately achieve maximum KL <= 0.05 and at least 90% top-1 agreement
against that format's CPU reference on fixture inputs. A same-quant
hipEngine-vs-llama difference is diagnosed separately from quantization loss.

### Q2 — Optional extended validation

A larger exact-artifact corpus or task suite may be added when a deployment
decision needs more coverage. It must use hash-pinned token IDs, identical
positions and scoring windows for every candidate, and separate corpus/task
metrics from the 90-row table. Report token count, tokenization and BOS/EOS
policy, stride/window policy, NLL/PPL where applicable, and category/task
results.

Extended evidence is additive: it cannot erase a failed exact-artifact KL,
top-1, or category gate. The BF16 model is larger than the current 62.5-GiB TTM
mapping limit, so any extended BF16 oracle runs on CPU or an explicitly
recorded CPU/GPU split; do not alter the system limit or compare its speed with
fully offloaded quants.

---

## 6. Performance protocol

### 6.1 Fixed shape matrix

Initial production shapes are:

- `512/128`
- `4096/128`
- `32768/128`
- `65536/128`

Use token ID `9707` repeated to the exact prompt length, greedy top-1, EOS
ignored, one request at a time, full practical offload, FlashAttention on, and
right-sized context. Report:

- prompt processing ms and tok/s;
- 128 true autoregressive transition ms/token and tok/s;
- one-time load/capture/setup outside steady decode;
- all raw samples and output-ID hashes;
- F16 versus BF16 K/V explicitly. Do not call those paths bit-identical.

`llama-bench` pp/tg rows remain useful kernel-oriented diagnostics. Binding
cross-engine decode comes from stateful context-matched requests with the same
raw token IDs.

### 6.2 Required engine rows

| Runtime | Weight format | Backend(s) | Comparison role |
| --- | --- | --- | --- |
| hipEngine | UD-Q4_K_M | `hip_gfx1151` | Product path and same-file engine candidate |
| llama.cpp | same UD-Q4_K_M | HIP, Vulkan | Same-file engine comparators |
| ROCmFPX | same UD-Q4_K_M | HIP, Vulkan | Fork overhead/control before attributing any ROCmFP4 win |
| hipEngine | PARO W4 | `hip_gfx1151` | Current hipEngine product ceiling; cross-quant only |
| ROCmFPX | STRIX_LEAN | HIP, Vulkan | Custom-format product row; cross-quant only |

The decisive ROCmFP4 attribution is Q4_K_M versus STRIX_LEAN in the **same
ROCmFPX binary**, followed by clean-fork controls. A comparison between
hipEngine PARO and ROCmFPX Vulkan is useful to users but cannot identify a
kernel or quant cause.

### 6.3 Memory on UMA

Report these scopes independently:

- on-disk model bytes;
- runtime-owned model/KV/workspace bytes when available;
- process-run GTT delta and peak;
- process RSS peak;
- system-available delta and swap;
- post-close residual.

GTT, RSS, and available-memory changes overlap physically on this APU. Never
sum them into a fictitious total.

---

## 7. MTP protocol

MTP starts only after AR and quality baselines close.

- Run the complete ten-prompt category fixture directly.
- Use the fixed train/heldout split and report every category.
- Generate a separate true no-MTP AR row from the same binary, model, prompts,
  sampling, cache, context, and timing boundary.
- Sweep draft depth 1-5. A small declared confidence set may include `p-min=0`
  and ROCmFPX's historical `0.55` starting point; it is evaluated on the full
  suite, never tuned to one prompt.
- Report complete proposal + target verify + accept/commit + correction +
  scheduler wall, proposals, accepted drafts, target passes, accepted/output,
  draft acceptance, visible outputs, and MTP/AR ratio.
- Greedy MTP output IDs must match the runtime's corresponding AR IDs. GPU
  acceptance must match the CPU transaction oracle where hipEngine exposes it.
- Repeated-token MTP is a transaction/perfect-acceptance diagnostic only.

A verifier-derived B0/off row, one prompt, or engine-reported generation time
without complete-cycle wall cannot support a speedup claim.

---

## 8. Profiling and port-admission protocol

Toplines and profiles are separate runs.

### 8.1 Tools

- **hipEngine:** prebuild every JIT object, require the cached compiler-version
  file, use semantic `wall_clock64()` markers for stage ownership, and use
  `rocprofv3` for symbol/count/grid/workgroup/VGPR/SGPR/LDS/scratch census.
- **ROCmFPX/llama HIP:** warm build and model caches outside the profiler, then
  profile the final child with `rocprofv3`; no profiled process may launch
  `hipcc` or clang.
- **ROCmFPX/llama Vulkan:** use `GGML_VK_PERF_LOGGER=1` query timestamps for
  operation attribution. Its synchronization perturbs throughput, so it never
  supplies a topline.
- **MTP:** profile only the final request child, not a parent prompt-suite
  harness.

For every profile:

```text
complete wall = device-stage/activity span + host/submission/sync residual
```

Reconcile to within 10% before selecting a target.

### 8.2 Candidate order

Only measured owners are eligible. The initial audit suggests this order of
questions, not implementation:

1. Does ROCmFP4 save complete decode wall in proportion to its 15.96% smaller
   file, and which tensor roles realize or miss that bandwidth saving?
2. Does Codebook10-to-int8 DP4A expansion beat hipEngine's current Q4/PARO
   decode owners at actual c1 and verifier shapes, or merely replace one
   efficient dot path with another?
3. Does role-mixed single/dual-scale quantization explain quality without
   erasing the dense FAST speed gain?
4. Are attention, runtime conversion/copy, or scheduling material after the
   weight-streaming delta is removed?
5. Does MTP improve because target rows are cheaper, because acceptance differs,
   or because timing boundaries differ?

Do not begin with native FP4 WMMA: the reference implementation does not use it,
and the installed rocWMMA path has no demonstrated native FP4 input primitive.
Do not port the external host scheduler wholesale; isolate one measured
mechanism.

Any admitted implementation lives under `kernels/<backend>/`, takes raw device
pointers, registers on `(backend, layer, quant, variant)`, and keeps an unfused
numerically equivalent fallback. External source path and commit are cited in
the eventual commit.

---

## 9. Execution ladder

### Z0 — Input, source, and idle admission

1. Let current downloads finish serially; do no GPU timing during transfer or
   SHA verification.
2. Verify every model hash/revision and tensor/tokenizer inventory.
3. Record the STRIX_LEAN tensor census, especially FAST, dual-scale, Q5_K,
   embedding/output, attention K/V, and MTP tensors.
4. Build clean ROCmFPX binaries for gfx1151 against the local ROCm 7.15 stack;
   record configure command and binary hashes. Run CPU/reference and focused
   backend correctness before loading the full model.
5. Preserve the current quantizer-routing blocker. Do not regenerate the pinned
   ROCmFP4 file from current `main`.
6. Freeze idle/power/thermal/memory preflight commands and the counter-rotation
   schedule.

Exit: all identities are machine-readable, all runtimes load their intended
models, and no benchmark is contaminated by transfer/build work.

### Z1 — Quality before speed (**complete**)

1. Add the engine-neutral raw-token/full-logit fixture and RED tests.
2. Capture BF16 teacher prefixes and logits.
3. Capture all three quant families at identical prefixes.
4. Publish the exact-artifact category matrix, paired noninferiority verdicts,
   and same-artifact runtime controls.

Exit: each quant has an explicit `Q4-equivalent` or `quality-traded` verdict.
A quality-traded quant can still be profiled and reported with its tradeoff, but
cannot become an unlabeled hipEngine default or parity target.

### Z2 — Clean AR/product baseline

Run the five required runtime/format rows at all four shapes, counter-rotated,
then add the same-file Q4_K_M stateful comparison. Keep load, capture, steady
AR, and memory scopes separate.

Exit: one compact baseline artifact establishes this ZBook's actual speed and
memory; no high-power row appears in the denominator.

### Z3 — Reconciled profiles

Profile short/4K AR first, then 32K/64K if attention changes the ranking.
Compare Q4_K_M and ROCmFP4 within the same fork before comparing forks. Join
semantic roles, symbols, resources, counts, bytes, and complete wall.

Exit: the top candidate has a measured >=1% complete-request ceiling and a
plausible in-tree design, or the port lane closes with no candidate.

### Z4 — One measured mechanism at a time

For each admitted candidate: RED CPU/reference fixture, implementation,
operation-complete correctness, quality gate, exact A/B, complete-model A/B,
re-profile, then keep or fully remove. No duplicate persistent weight sidecar is
allowed as a production shortcut.

### Z5 — Exact MTP economics

Run true AR and B1-B5 over the complete suite for hipEngine GGUF/PARO and
ROCmFPX HIP/Vulkan as supported. Optimize only after proposal, target, commit,
and scheduler ownership reconcile.

### Z6 — Closure

Promote only exact or explicitly quality-qualified non-regressive wins. Update
the benchmark rollup/changelog, compact artifacts, immutable worklog, relevant
kernel catalog/lineage, refactor ledger, and package defaults in atomic commits.

---

## 10. Initial command surfaces

These are starting surfaces, not completed evidence. Z0 must update paths and
binary hashes before execution.

```bash
# Power/identity preflight (read-only).
sudo ryzenadj -i
cat /sys/module/amdgpu/parameters/sched_policy
cat /sys/module/ttm/parameters/pages_limit
tuned-adm active
vulkaninfo --summary
rocminfo | grep -E 'Name:|gfx'

# hipEngine Q4_K_M/PARO split rows use the established resident sweep.
python3 scripts/qwen35_readme_sweep.py --help

# llama/ROCmFPX split rows, with the selected llama-bench passed explicitly.
python3 scripts/llamacpp_bench_with_peak.py --help

# Stateful llama/ROCmFPX true-AR + MTP category/token controls.
python3 scripts/llamacpp_mtp_bench.py --help
```

The existing `scripts/run_gfx1151_readme_refresh.sh` is a useful orchestration
reference, but its default model paths, binary paths, old host assumptions, and
three-versus-five-run split must not be used unchanged for this campaign.
