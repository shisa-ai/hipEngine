# Qwen3.8-Flash-Next Implementation Campaign

Status: **bounded 512/1K functionality gates are closed (F0–F10); fresh production is 1.10x behind llama.cpp HIP decode and 3.22x behind HIP prefill; closing the measured projection/MoE/GDN dataflow and submission gaps outranks feature expansion**

This campaign brings the open-weight `Qwen/Qwen3.8-Flash-Next` checkpoint to
hipEngine as a torch-free, registry-composed, text-generation path first, then
adds native QSA long-context execution, MTP, multimodal input, serving, and
performance qualification. The first hardware lane is the local Radeon 8060S /
`gfx1151` host with 128 GiB unified memory. `gfx1100` remains a required peer
correctness lane, but no W7900 result may be inferred from this host.

The operational bring-up artifact is the independently published, revision-
pinned Unsloth `UD-Q4_K_XL` split GGUF. It passed all-part SHA-256, complete
tensor-map, sparse-PLE, real memory, same-artifact llama.cpp full-logit, and
public `LLM.generate()` gates on gfx1151. A local conventional `Q4_K_M` remains
a reproducibility/quant-quality follow-up from the now-complete official BF16
snapshot; it is not required to relabel the verified working artifact. The
51.2B-parameter n-gram table remains one IQ4_NL sparse mmap/host owner rather
than consuming accelerator-resident capacity.

This document is the campaign authority. Architecture-wide decisions also stay
consistent with [`PLAN.md`](PLAN.md); numerical and evidence rules remain
normative in [`TESTING.md`](TESTING.md),
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md),
[`KERNELS.md`](KERNELS.md), and [`BENCHMARK.md`](BENCHMARK.md). The active
gap-closure plan, profiling recipe, external-source audit, and punchlist are
[`QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md`](QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md).
The matched cross-engine speed, static-logit, autoregressive-repeatability,
MTP-equivalence, and absolute-quality evidence is consolidated in
[`QWEN3.8-FLASH-NEXT-STRIX-HALO-SURVEY.md`](QWEN3.8-FLASH-NEXT-STRIX-HALO-SURVEY.md).

---

## 0. Campaign checkpoint (2026-08-30)

Consolidated position after the fresh remote-HEAD/API/role profile. All rows
are same-host `zbook` / gfx1151 / `UD-Q4_K_XL` unless stated. Strict is the
exact-bit profile; production is the certified T2 profile with manifest
`9e27fec0…` (strict `42509601…`).

### 0.1 Where we are

| Row | Strict | Production / certified opt-ins | Beat first: llama.cpp HIP, same host + GGUF | Stretch: llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| p508 prefill | 61.40 (chunk 512) | **84.83** (5.96–6.03 s) | **272.83** remote HEAD | **331.03** remote HEAD |
| p1012 prefill | 60.20 (chunk 512) | **~79–95** (mixed-session; not rerun) | 307.2–318.9 prior master | 308.4–310.5 prior master |
| tg32 decode | 13.880 | **15.19** admitted safe43 steady-state | **16.64** remote HEAD | **24.22** remote HEAD |
| Natural 16K prefill | 47.989 (chunk 512, gate passed) | — | ≥100 tok/s unlocks the 64K rung | — |
| MTP vs true AR | 0.955x aggregate (opt-in; 10/10 exact, 84.28% acceptance) | — | ≥1.0x to promote; ≥1.5x real target | external MTP fork ~2.7x |

Production = guarded dense-Q8 MMQ, compact f16-WMMA MoE on layers 27–47
with the iu8-WMMA gate/up kernel (exact q + 3 residual planes) on 35–47,
column-warp GDN prefill on 27–47 (llama layout, supersedes peer-GDN),
key-parallel QSA flash on 35–47, and Q8_1 DP4A decode on calibrated layers
`0,2,5,6,8,9,10,11,13–47`; layers 0–26 keep the strict exact owners. The
WMMA-MoE27 packet (450 rows, three repeats) passes at KL
mean/p95/p99/max `2.79e-4/1.53e-3/3.49e-3/5.98e-3`, 446/450 top-1, all
scopes ≥ 98.67%, exact repeat/state, 18/18 repeat-exact free generation with
4 task-valid divergences, exact c2 with zero teardown; paired p508/p1012
`6.572→6.287 (-4.34%, 80.82 tok/s)` / `13.398→12.694 s (-5.26%, 79.73 tok/s)`.
The fresh role-resolved pp508 trace is hipEngine **5.959 s / 3,328 kernel
rows** versus llama.cpp remote-HEAD HIP **1.625 s / 5,119 rows**: **3.67x**
device-kernel gap. End to end is **84.83 vs 272.83 tok/s (3.22x)**. The
corresponding decode trace is **48.63 vs 38.90 ms/output (1.25x)** in kernels;
end to end is **15.19 vs 16.64 tok/s (1.10x)**. Kernel rows are not host
launches: decode expands 625 MoE graph nodes but still issues 1,195 direct
launches and 48 graph launches per token.

### 0.2 What we learned

1. **Mine dataflow first, but count submission correctly.** llama.cpp prefill
   remains faster while executing more kernel rows; hipEngine's profiled p508
   span exceeds kernel sum by only 72 ms. Decode is different: 1,195 direct
   launches plus 48 small graphs leave 37.1 ms/token between kernel sum and
   span under profiling, so operation-complete fusion and larger state-safe
   graphs are material there.
2. **Numerical admissibility is layer- and composition-specific.** Current
   certified scopes are f16-WMMA MoE 27–47, iu8 gate/up 35–47, GDN colwarps
   27–47, QSA flash 35–47, and DP4A excluding `1,3,4,7,12`. Full-layer and
   adjacent boundary candidates were measured and rejected. Cheap screens
   guide; only complete 450-row packets promote.
3. **Exact thread contractions beat naive packing.** Mapping logical lanes
   onto fewer physical threads while preserving the declared reduction tree
   (Q5_1 t128→t64, Q4_K physical64, fused weighted down) kept every bit exact
   and won repeatedly. Naive widening (pack8, Q4 pack2, residual Q8_1x2)
   regressed decode each time and was removed with measured evidence (pack2:
   8.312 vs 11.515 tok/s baseline).
4. **The greedy-identity discipline works.** AR≡MTP generated-ID equality,
   450-row full-vocab KL packets, physical c2 exactness, and teardown-zero
   checks caught every bad candidate before promotion; nothing regressed
   silently.
5. **Vulkan beats HIP on this host at every depth** (same-host llama.cpp rows
   plus the external fork below). HIP parity is therefore the binding target
   and Vulkan the ceiling hypothesis, with Vulkan geometry as the shape proof
   for our own HIP ports.
6. **External fork hypotheses to validate in-tree** (see §1.2): distinct-stream
   MTP hyper-connection combiner, n-max 6, GPU radix top-k, incremental pooled
   QSA keys, gathered attention, skinny-m inject routing, and mat-vec epilog
   fusion. Several are already implemented; role attribution now identifies
   GR projection epilogs as the largest remaining direct-fusion surface.
7. **The prior GDN decode-all claim is invalid.** Its selector was unreachable
   below the `rows == 1` branch, so its packet compared the strict owner to
   itself. Wiring the actual candidate costs 6.832 ms/token plus a 0.117-ms
   tail versus 2.454 ms/token retained. Commit `15a436766` clears the dead
   binder route; prefill colwarps remains certified.

### 0.3 Next units (priority order)

The executable phase order and current punchlist are owned by
[`QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md`](QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md).
Audit note (2026-08-29, `qwen4exp-fork-idea-audit`): the fork's radix top-k,
incremental pooled-key cache, gathered decode attention, GDN concat fix,
permute-free scoring, and distinct-stream MTP combiner are **already covered
in-tree** — verified in source with retained evidence. The remaining campaigns:

1. **Fresh profile order:** (a) move to early MoE layers 0–26, which own
   **2.526 s** of p508. The existing layer-2 grouped-WMMA route is closed as a
   rejection: it cuts Q5_K gate/up **279.86→16.66 ms** and wins about 5% at
   p512, but its complete 450-row gate fails prefill-last mean KL
   (**0.001179 > 0.001**). Do not rescreen unchanged T2 arithmetic; revisit
   layer 2 only with a materially new exact/T1 dataflow. (b) decode GR
   operation-complete
   down+inject and projection-epilog fusion, up to 387 direct launches/token;
   (d) normalized-Q/K, transposed-state GDN decode (**2.659 vs 0.465 ms/token**);
   (e) a state-safe larger decode graph after the historical third-replay state
   bug is resolved. QSA/GDN prefill suffix widening follows only after these
   larger Amdahl units and requires fresh composition gates.
2. **MTP economics:** verification is a serial per-candidate target loop
   (budget 1..4), so every drafted token costs a full target decode row; the
   multirow candidate was rejected because `rows >= 2` switches MoE to the
   grouped-prefill arithmetic and breaks per-row bit equality. Build a
   rows≤8 batch-invariant verify path (per-row decode-order kernels in one
   launch, or proven bit-equal batched kernels), then sweep budget 4→6
   (fork reference: n-max 6 at ~0.9 code acceptance). Promote at ≥1.0x AR,
   target ≥1.5x, gated on the full mtp-bench suite against same-protocol AR.
3. **Depth/64K rung:** natural 16K prefill must reach ≥100 tok/s (strict
   chunk-512 is 47.989) to re-open the 64K rung. The final production profile
   has not been rerun at 16K, so no long-context production rate is claimed.

---

## 1. Fixed identity and references

### 1.1 Official checkpoint

| Field | Value |
| --- | --- |
| Hugging Face model | `Qwen/Qwen3.8-Flash-Next` |
| Frozen revision | `f5d08274bafd880402bd16f5e3e6c514136ec06c` |
| HF architecture | `Qwen4ExpForConditionalGeneration` |
| HF model type | `qwen4_exp` (`qwen4_exp_text`) |
| GGUF architecture | `qwen4exp` |
| License | Qwen Community 1.0; preserve the official model license beside local artifacts |
| Source format | BF16 safetensors, 131 shards |
| Source tensor bytes | `359,999,963,128` |
| Repository storage | `360,023,351,155` bytes across 144 frozen files |
| Frozen tree manifest | SHA-256 `dfd29ff3e73cd8fac3c10531d0d61196fa5f4af67ad75df5d2c96401544a7502` over the local HF tree record |
| Source path | `/models/hf/Qwen3.8-Flash-Next` |
| Operational GGUF path | `/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL` (4 parts, revision `8bdc666649440e9bdc97e16f3f75782c98478ff5`) |
| Planned local conventional quant | `/models/gguf/Qwen3.8-Flash-Next-Q4_K_M.gguf` (not yet produced) |
| Higher-quality GGUF candidate | Unsloth `UD-Q5_K_XL`, frozen revision `ff34bcdd8a6ecffbe75b392e57b866df8f6bba8f`: optional/deferred; it is not part of the active bring-up and cannot displace `UD-Q4_K_XL` performance parity |
| Native context | 262,144 tokens |
| Extended context | up to 1,000,000 tokens; not an initial support claim |

The completed local source resolves the exact revision above: 144 repository
files / 360,023,351,155 bytes, including 131 safetensor shards /
360,000,192,888 bytes. All shards pass pinned Hugging Face LFS size+SHA-256 in
the final complete rehash. The local manifest SHA-256 is
`973433d38d86f0e771855b74a170f55f07290bc446ff9bc7a42a688a152b912b`;
compact evidence is in
[`2026-08-27-qwen38-flash-next-official-source.json`](../benchmarks/results/2026-08-27-qwen38-flash-next-official-source.json).
The pinned MTP sidecar is local at
`/models/gguf/Qwen3.8-Flash-Next-MTP-Q8_0/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`;
its 34 tensors carry 4,126,482,432 payload bytes (4,117,775,360 Q8_0,
5,430,272 F32, 3,276,800 BF16), with no PLE table or alternate device layout.
Artifact:
[`2026-08-28-qwen38-flash-next-mtp-q8-sidecar.json`](../benchmarks/results/2026-08-28-qwen38-flash-next-mtp-q8-sidecar.json).
Model weights and converted GGUF files are local artifacts and are never
committed.

### 1.2 Read-only architecture and comparator sources

| Source | Frozen identity / use |
| --- | --- |
| Qwen technical report | `QwenLM/Qwen3.8-Flash-Next/tech_report.pdf`, local SHA-256 `04f263446d74a35cb7cea368574e0c561f3b05c133be2c777ac884404063655d`; architecture/formula authority |
| Transformers | `huggingface/transformers@fc5c5bde8e656dad91cbf34e61940d984b1c7b91`, `src/transformers/models/qwen4_exp/modular_qwen4_exp.py`; executable formula/control oracle outside the hot path |
| llama.cpp PR #27742 | merged as `ggml-org/llama.cpp@6c84c7d5d8833c6e0df69628f75a0f599797934e`; **primary basic implementation guide** for converter, target text graph, QSA, PLE, hyper-connections, quantizer fixes, and stock Qwen3-VL vision. The already-frozen comparator binary remains `bea3b12daee45876b0129a3602dc8f534ce30bf0`; do not mix those identities. |
| llama.cpp PR #27739 | `dfa0c0fee2b704fd2ac228d365d40502c3006c40`; MTP design reference, used through EngramHalo's Qwen4Exp port rather than as the target-text/quantizer authority |
| EngramHalo.cpp | `Aristo94/EngramHalo.cpp@4ff3affc2ac5861f7dda42bcf5ff653c776b816f`; PR #27742-based gfx1151 reference. MTP runtime `0f1c3e2ef41117033d91a83d7634fca4dfe12107`, MTP converter `2cc66f08ca03c6e3f385ec15412f92ad6d490794`; performance patches are ideas to validate in-tree, not inherited evidence. |
| EngramHalo Q8_0 MTP sidecar | `EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF@6f7900648b1c6b14f067a182c640e47971e9ab35`; one 4,137,429,088-byte GGUF, SHA-256 `9db03a687670608286e99b563fcc86d0ee76c8dd863f64b2afc0b54eb0eb975d`; strict 34-tensor inventory/shape/qtype map passes; execution remains unqualified |
| apepojken llama.cpp fork | `apepojken/llama.cpp@843d5750579a15ed4a42d73eb862855c271021ac`, branch `qwen4exp-spec-mtp` (PR #27742 base + speculative-decode rollback fixes, native MTP head with distinct-stream combiner, 4-pass 8-bit radix top-k, incremental pooled-key cache, gathered decode attention, GDN concat fix, epilog fusion). The clean local Vulkan build passes TOP_K 445/445 and is fast on matched Q4, but only 8/12 canonical AR cases repeat and MTP is 9/10 AR-message exact; see the Strix Halo survey. Reported Q3 rates (50.4 tok/s code at 0.963 acceptance, 338 tok/s p32K) remain author-reported. Sidecar: `jockevaupptaget/Qwen3.8-Flash-Next-MTP-GGUF@69da733459b737b79273c0a322340de9c9c08fa2`, 4,135,893,184 bytes, SHA-256 `713109a7f0dfd5bde305c296b4252daf576aa4c2e380f043f3323aa00dc2cde8`. |
| vLLM PR #53896 | `vllm-project/vllm@2a4cd640ff1a61b66124ddbaaf02a73781f7295a`; paged raw/persistent-compressed QSA caches, GPU scoring/top-k/expansion, split-k sparse attention, MTP step-0 index reuse, and AMD path reference |
| vLLM PR #53899 | `vllm-project/vllm@95dc96d1d012a25ff5c3823a1e77197c8dae4654`; PLE CPU-offload protocol/reference; known TP1 warmup deadlock is explicitly not inherited |
| SGLang PR #36497 | `sgl-project/sglang@7c66045d71f067c1c5da2b85baad3c47d9a19cb7`; persistent compressed-QSA cache, fused exact index prep/compression, fast top-k, sparse attention, PLE offload, HC and MTP reference |
| SGLang AMD PR #36601 | `sgl-project/sglang@3003ddf1574ef5004e21a10e36aaabc364766921`; gfx942/gfx950 QSA graph replay and MTP correctness reference, not a gfx1151 performance claim |
| SGLang PD PR #36651 | `sgl-project/sglang@7bcf5ba35654131cd2ff49bee73eca9677a33ca1`; PLE/GDN/QSA pending/compressed-state transfer and exact aggregate/PD parity reference |
| Unsloth guide/repository | Memory-size expectation and independent GGUF comparator; no unverified quality claim transfers into hipEngine |

The historical comparator worktree is local and external to hipEngine:
`/home/lhl/llama.cpp/llama.cpp-qwen4exp` at pre-merge PR #27742 commit
`bea3b12da`. It remains the frozen oracle for evidence already recorded against
that binary. New implementation decisions use the final merged PR identity
`6c84c7d5d` from the read-only `/tmp/EngramHalo.cpp` clone (or a separately
pinned read-only checkout), then EngramHalo `4ff3affc2`. Ported ideas cite exact
commit + source path. Kernel development still occurs only in this tree.

### 1.3 Basic implementation parity checklist from #27742 and EngramHalo

This checklist is reviewed before adding new Qwen4Exp code. It prevents local
optimization work from replacing missing product functionality.

| Reference surface | Reference implementation | hipEngine short-context requirement |
| --- | --- | --- |
| Target text graph | #27742 `src/models/qwen4exp.cpp`: widened HC residual, PLE, 36 GDN + 12 attention layers, 512/top-10 MoE, final HC mixer | Already present; preserve same-artifact 512/1K logits and public AR while completing the rows below. |
| Conversion/artifact | #27742 `conversion/qwen4exp.py` and quantizer fixes | Keep frozen target GGUF identity; support distinct MTP/mmproj artifacts rather than hiding them inside target mapping. |
| QSA/cache | #27742 final `llama-memory-hybrid-idx.*` plus `qwen4exp.cpp` | Existing native QSA remains historical/available, but no more full-model long-context runs before the ladder permits them. |
| Vision | #27742 `Qwen4ExpVisionModel(Qwen3VLVisionModel)`, stock Qwen3-VL clip path, plus image-placeholder PLE hashing | Add optional processor/vision/mmproj plugins and one 512/1K image generation gate; do not advertise vision before it passes. |
| Target→draft handoff | EngramHalo `0f1c3e2ef`, `src/models/qwen4exp.cpp` | Expose the authoritative pre-final-mix widened target hidden row (`4×2560`) to the MTP cycle without host reconstruction. |
| MTP input | EngramHalo: RMSNorm token embedding and widened hidden independently; concatenate and apply fused `eh_proj=[fc_embedding|fc_hidden]` | Loader validates both source projections or one fused sidecar projection; CPU/HIP fixture proves the fused equation. |
| MTP layer | EngramHalo: one dense-attention Qwen4Exp HC+MoE block, no PLE, own transactional K/V; indexer tensors may exist but are not used | Implement a model-owned one-layer draft runner with independent state and strict target rollback/commit. QSA/IndexShare is deferred. |
| MTP output | EngramHalo: preserve widened draft hidden for chaining, then its own final HC mixer and shared output head | Draft emits logits + next widened hidden; candidate budgets 1–4 use existing proposal/verify/accept interfaces. |
| MTP sidecar conversion | EngramHalo `2cc66f08c`: filter `mtp.*`, remap final HC mixer, pad compression ratios with dense `0`, omit PLE, fuse `fc_embedding/fc_hidden` | First accept the pinned external Q8_0 sidecar after exact inventory/hash checks; add a reproducible local converter only if mapping cannot be proven directly. |
| Public speculative route | EngramHalo generic draft-MTP driver with `n_max=4`; Q8_0 sidecar and confidence gate are external findings | Wire hipEngine's existing provider/transaction/streaming surfaces at 512/1K. AR stays default until the natural multi-prompt exact/economics gate passes. |
| Public server | #27742/EngramHalo run through normal llama-server surfaces | Prove hipEngine blocking + SSE, chat/reasoning/tool template basics, cancellation/reset/close, and exact response-owned token accounting at 512/1K. |

EngramHalo's gfx1151 performance patches are tracked after functionality:
wide GPU top-k (`79b55adce`), H256/GQA2 quantized-KV FA selection
(`21be595b7`), chunked GDN prefill (`083dde319`), decode-graph reuse
(`954198cfd`), and true selected-K/V gather (`942b9cc74`). hipEngine already
has exact GPU QSA selection and selected sparse attention; that is not evidence
that MTP, serving, or vision works. Q8_0 KV, Q8_0 draft quantization,
`ubatch=2048`, hipBLASLt, confidence threshold `0.75`, SSD-lazy PLE, and the
reported rates are external hypotheses only. Each needs an in-tree RED gate and
same-host complete-model evidence before promotion.

---

## 2. Reality check: A6B is not a 6B model file

Qwen3.8-Flash-Next has:

- **125B** backbone parameters;
- **6B activated** backbone parameters per token;
- **51B** additional n-gram embedding parameters;
- approximately **4B** MTP parameters; and
- a Qwen3-VL-derived vision encoder.

The active-parameter count predicts per-token compute, not storage. The official
checkpoint is 360.0 GB BF16. Unsloth currently estimates roughly **110 GB** for
a 4-bit GGUF and 75 GB for its smallest dynamic 1-bit file. The local 128 GiB
unified-memory host can run the Q4 target only if PLE is offloaded sparsely and
the runtime avoids duplicate raw/repacked weight owners.

The operational `UD-Q4_K_XL` artifact was selected when it became available
because it provides independent imatrix provenance and one file that actually
fits the 128-GiB lane while retaining high-bit sensitive roles. Its exact mixed
payload is 44.35 GB Q4_K, 27.05 GB Q5_1, 9.61 GB Q8_0, 1.15 GB Q5_K, 313 MB
F32, 39 MB BF16, and 28.80 GB IQ4_NL PLE. `Q4_K_M` remains the conventional
local follow-up; `Q4_K_S` is produced only if that follow-up has a concrete
capacity blocker. The campaign never reports “fits” from file size alone: the
working artifact measured 82,718,198,780 peak hipEngine-owned bytes at 2,051,
ran generation, and returned to zero tracked allocations after close.

### 2.1 Expected memory model (planning values, not measurements)

- The PLE table is `320,000,xxx × 160` values: 51.2B parameters. A standard K
  recipe cannot use a 256-value K block on a 160-wide row, so llama.cpp falls
  back to a 32-value format (normally `Q4_0`), approximately **28.8 GB**.
- The remaining Q4_K_M text backbone is expected to be roughly **75–82 GB**,
  depending on root/output, higher-bit K_M tensors, and whether MTP is included.
- PLE touches exactly 16 rows per token (8 bigram and 8 trigram heads), only
  `16 × 160 = 2,560` values. It therefore remains mmap-backed and is gathered
  into a bounded pinned staging ring; the whole table is not registered or
  materialized on device.
- Full-attention BF16 K/V is `12 layers × 2 KV heads × 256 dims × K/V × 2 bytes`
  = **24,576 bytes/token**: exactly 6.0 GiB at 262,144 tokens.
- A compressed BF16 QSA index cache is approximately **768 bytes/token** when
  one 128-dim key is retained per four tokens per QSA layer: 0.1875 GiB at
  262,144 tokens. The current strict owner retains raw per-token FP32 keys
  (**6,144 bytes/token**) plus complete-block member/start, pooled-FP32-key,
  score, and selected-position workspaces: about **1.90 GiB/request** at 262,144
  tokens. Memory admission accounts this exact current owner; the earlier
  0.75-GiB raw-BF16 and 0.1875-GiB compressed forms remain later promotions.
- The 36 GDN FP32 matrix states are approximately **108 MiB/request**; Conv,
  PLE history, and four residual branches are comparatively small.

These estimates support the user’s KV intuition: only 12 of 48 layers own
unbounded K/V and each has two K/V heads. The improvement is caused by the
hybrid/QSA geometry, not by “A6B” itself.

---

## 3. Architecture contract

### 3.1 Frozen text geometry

| Component | Geometry / semantics |
| --- | --- |
| Layers | 48: repeating `GDN, GDN, GDN, QSA` |
| Hidden | 2,560 |
| Residual | 4 branches × 2,560 = 10,240 values |
| GR bottleneck | 320 |
| GDN | 16 Q/K heads, 48 V heads, head dim 128, Conv kernel 4, FP32 recurrence |
| GDN output gate | sigmoid, not Qwen3.5’s SiLU |
| QSA core | 24 Q heads, 2 K/V heads, head dim 256, gated output |
| RoPE | interleaved MRoPE, first 64/256 dims, theta 10,000,000 |
| QSA indexer | 4 Q heads, 1 shared K head, dim 128, partial RoPE 64 |
| QSA compression | 4-token micro-blocks; mean in FP32 before norm/RoPE |
| QSA budget | best 512 complete blocks = 2,048 tokens, plus 0–3 tail tokens |
| MoE | 512 routed experts, normalized softmax top-10, one gated shared expert |
| Expert widths | routed 640, shared 640, SiLU/SwiGLU |
| PLE | layer ID 2 (zero-based decoder index 1), bigram+trigram, 8 heads/order |
| PLE row | 160 values/head, 16 rows concatenated to 2,560 |
| Vocabulary | 248,320 |
| Output | untied 248,320 × 2,560 head |
| MTP | one full/QSA-attention MoE layer, no PLE, consumes widened target hidden state |

### 3.2 Gated Residual (GR)

Every token mixer and every MoE has a separate GR read/write. For widened
residual `R ∈ [C=4, H=2560]`:

1. independently zero-centered-RMS-normalize each branch;
2. flatten to 10,240;
3. compute `g = sigmoid(W_up(silu(W_down(norm(R)) / C)))`;
4. read `x = mean(g * norm(R), branch_axis)`;
5. compute scalar branch injection `s = 2 * sigmoid(W_inject(norm(R)) / C)`;
6. write `R' = R + s[:, None] * block(x)`.

The final GR read replaces output RMSNorm. There is no normal residual add and
no Qwen3.5 input/post-attention norm on this architecture. Treating GR as a
minor norm variant would be a model bug.

### 3.3 PLE hash and injection

For token `x_t`, the PLE host path preserves the official uint64 overflow/XOR
hash exactly. Missing predecessors and post-EOS history use the model EOS token.
Each bigram/trigram order owns eight distinct prime-sized heads and row offsets.
The 16 gathered rows concatenate to `E_t ∈ R2560`.

PLE computes:

- `K = grouped_rmsnorm(W_key E)` over four 2,560-wide branches;
- `V = W_value E`;
- `Q = grouped_rmsnorm(R)`;
- `score = sum(K * Q) / sqrt(2560)` per branch;
- `gate = sigmoid(sign(score) * sqrt(max(abs(score), 1e-6)))`;
- `U = gate * V` per branch;
- `delta = U + silu(dilated_depthwise_conv(grouped_rmsnorm(U)))`;
- `R = R + delta` before the layer-2 token-mixer GR read.

The depthwise Conv has kernel 4, dilation 3, and nine historical rows. PLE also
owns the two preceding token IDs. Both states obey request identity,
cancellation, compaction, transaction, and prefix-reuse semantics.

### 3.4 QSA

For each complete 4-token block, raw indexer keys are averaged in FP32,
zero-centered-RMS-normalized, and partial-RoPE-rotated at the block’s first
position. Each query scores a block as:

`score(q, block) = sum_h relu(dot(q_h, pooled_k)) / sqrt(128)`.

QSA selects 512 complete blocks, expands them back to their original token
positions, and includes the current incomplete tail. Sparse softmax attends to
the **original uncompressed K/V**. Below 2,052 visible tokens, QSA is dense by
construction and is the first exact end-to-end oracle.

QSA storage and selection must remain under the normal attention
`KVLiveSpans` ownership contract. A separate index cache may mirror those spans,
but may not invent an append-only `(context_len, contiguous cache)` shortcut.

---

## 4. What can be reused and what is new

| Area | Reuse | Required delta |
| --- | --- | --- |
| GGUF scanner | v2/v3 scanner, uint64/int64 metadata arrays, lazy memmap tensors | `qwen4exp` metadata/config and tensor map; split-GGUF discovery if conversion is sharded |
| Quant | Q4/Q5/Q6/Q8 layouts and projection kernels | host Q4_0 PLE row dequant/gather; geometry admission for H2560/E512/top10 |
| GDN | existing Conv, normalized Q/K recurrence, FP32 state, prefill/decode kernels | sigmoid output-gated norm variant and H2560 projection policies |
| Full attention | partial MRoPE, gated GQA, paged KV write/read | 24Q/2KV group-12 geometry and QSA selected-span execution |
| MoE | router, selected expert Q4 K-family kernels, shared expert | 512-expert/top-10 capacities, H2560/F640 shapes, no hardcoded H5120 policies |
| Runtime | persistent session, graph/eager, global KV pool, sampling/serving | widened residual owner and two extra PLE states; QSA index-cache lifecycle |
| Tokenizer | GGUF BPE/tokenizer infrastructure | validate tokenizer identity/chat template and reasoning-effort behavior |
| SpecDec | provider/target transactions, chain accept/commit, MTP2 interfaces | one-layer Qwen4Exp MTP, widened hidden seed, QSA/IndexShare semantics |
| Vision | none in the public Qwen GGUF generation path | Qwen3-VL-compatible vision encoder/processor and multimodal MRoPE admission |

The first short-context target can reuse dense attention because QSA and dense
are identical below the budget. This is an explicitly bounded bring-up route,
not a long-context fallback: for context above 2,051, missing QSA must reject
rather than silently claim correct generation.

---

## 5. Non-negotiable implementation rules

1. The runtime reached by `LLM.generate()` remains torch-free.
2. Register a distinct model plugin (`qwen4_exp_gguf`); do not fold Qwen4Exp
   into `Qwen35GGUFModel` or add engine-level architecture branches.
3. Resolve backend/model/quant/profile once. No `if backend ==`, `if quant ==`,
   or prompt-conditioned routing in model/engine hot paths.
4. New kernels use raw pointers and live in the backend peer tree. Every fused
   or production variant has a registered strict primitive fallback.
5. QSA K/V and index-cache writes consume complete `KVLiveSpans` metadata.
6. PLE is one sparse host/mmap owner. Never create a device copy, BF16 shadow,
   or second quant layout of the 51.2B table.
7. The four GR branches are authoritative state. They may not be collapsed
   between layers or reconstructed from the mixed hidden row.
8. FP32 GDN state is strict. Lower-precision state is a later T1/T3 campaign,
   not part of basic model bring-up.
9. Text-only support is declared separately from vision support. A working text
   path must not advertise multimodal input until F9 passes.
10. MTP is opt-in until it passes full-suite exact target economics. AR remains
    the strict fallback.
11. No benchmark is tuned to one token ID or prompt. Performance and speculative
    claims use the complete category+heldout suites.
12. No model weight, GGUF, JIT `.so`, profiler dump, or raw benchmark log is
    committed.

---

## 6. Artifact and quantization plan

### F0 — Download and make the primary quant

1. Download exact revision `f5d082...` to
   `/models/hf/Qwen3.8-Flash-Next` with `hf download`.
2. Verify the model index total, all 131 shards, and local storage. Record a
   manifest hash over `(relative path, size, HF/LFS identity)`; do not hash
   360 GB redundantly if the hub client already verifies each content object,
   but full-hash the final GGUF.
3. Use llama.cpp PR #27742 at `bea3b12da` because it:
   - converts `qwen4exp` target text;
   - writes PLE shards through a bounded mmap path rather than concatenating a
     100+ GB tensor in RAM;
   - fixes 32-block fallback for 160-wide PLE rows;
   - permits explicit `per_layer_token_embd.weight` quant selection; and
   - fixes the quantizer’s otherwise enormous temporary work buffer.
4. Convert the target text to F16 GGUF. Split only if the converter requires it;
   hipEngine F1 must support the produced form rather than hand-editing files.
5. Quantize to `Q4_K_M`. Preserve the converter’s exact PLE hash metadata.
6. Scan the complete GGUF and record:
   - file bytes and SHA-256;
   - metadata/tensor-inventory hashes;
   - actual PLE tensor name, shape, qtype, and bytes;
   - qtype bytes by role (root, GR, GDN, QSA, expert, shared expert, PLE);
   - whether MTP and vision are absent/present; and
   - projected hot-resident vs mmap-owned bytes.
7. Run llama.cpp’s metadata load and a bounded text smoke at context <2,052.
   If PR #27742 changes during the campaign, the frozen commit remains the
   comparator until a deliberate refresh is logged.

Planned commands (final commands and results belong in F0’s worklog):

```bash
hf download Qwen/Qwen3.8-Flash-Next \
  --revision f5d08274bafd880402bd16f5e3e6c514136ec06c \
  --local-dir /models/hf/Qwen3.8-Flash-Next

cd /home/lhl/llama.cpp/llama.cpp-qwen4exp
python3 convert_hf_to_gguf.py \
  --outtype f16 \
  --outfile /models/gguf/Qwen3.8-Flash-Next-F16.gguf \
  /models/hf/Qwen3.8-Flash-Next

build-qwen4exp/bin/llama-quantize \
  /models/gguf/Qwen3.8-Flash-Next-F16.gguf \
  /models/gguf/Qwen3.8-Flash-Next-Q4_K_M.gguf \
  Q4_K_M
```

Before the expensive conversion, use converter `--dry-run`. Before final
quantization, use `llama-quantize --dry-run` to confirm the 160-wide PLE table
falls to a supported 32-block type and no critical GR/indexer/norm tensor is
accidentally quantized. Delete the F16 intermediate only after the Q4 GGUF,
its hash, scanner gate, and comparator smoke pass.

**F0 gate:** exact source revision complete; Q4_K_M GGUF scanned and hashed;
PLE is one mmap-compatible tensor; llama comparator emits finite logits/text;
local disk retains enough margin for implementation fixtures and profiling.

### Quant-quality follow-up

The first local Q4_K_M is a bring-up artifact, not automatically the final
quality quant. Once Unsloth publishes its Q4_K_M/UD-Q4_K_XL with imatrix
provenance, compare it as an independent artifact. Do not silently replace the
campaign target. Promotion requires the same model/plugin gates and explicit
artifact identity.

---

## 7. RED/GREEN implementation milestones

Each milestone is one or more atomic, validated commits with a unique immutable
worklog entry. Mark a milestone complete only when all named gates pass.

### F1 — Model plugin and GGUF metadata (CPU, no model execution)

Add:

- `hipengine/models/qwen4_exp.py`: `Qwen4ExpGGUFModel`, architecture
  `qwen4exp`, default quant `gguf_q4_k_m`, representative layer sequence and
  tensor templates;
- `hipengine/loading/qwen4_exp_gguf.py`: immutable config, root/layer/PLE/MTP
  tensor maps and strict validation;
- synthetic GGUF fixture support for qwen4exp metadata; and
- model/loader exports.

The config parser validates every frozen geometry field, the exact 36/12 layer
mix, PLE layer 1, GR width/rank, QSA budget/compression, all 128 PLE shards only
at conversion input (one joined GGUF tensor at runtime), 512/top-10 MoE, and
untied head. Unknown geometry fails closed.

**RED:** current registry cannot resolve `qwen4exp`; current Qwen35 mapper
rejects the architecture. Tests first pin both failures, required/optional
roles, shape errors, unknown tensor rejection, and uint64 PLE metadata.

**Gate:** model registry + synthetic/real header map pass; no tensor bytes are
materialized; Qwen35/Qwen35MoE mapping tests stay green.

### F2 — CPU-reference formulas and state ownership

Add clear NumPy oracles for:

1. grouped zero-centered RMSNorm over four branches;
2. GR read and write;
3. uint64 PLE bigram/trigram hashing with EOS reset;
4. PLE gate, dilated depthwise Conv, and state update;
5. QSA block pooling, partial RoPE, scoring, selection, tail inclusion and
   sparse attention;
6. sigmoid-gated GDN output norm; and
7. one reduced-geometry complete Qwen4Exp layer.

Fixtures are hand-checkable and include one-token, 3/4/5-token QSA boundaries,
2,048/2,051/2,052 selection boundaries via reduced analogues, EOS reset,
non-contiguous `KVLiveSpans`, page boundaries, two requests, cancellation, and
state copy/rollback.

**Gate:** analytic fixtures pass; QSA equals dense below budget; sparse indices
match the Transformers formula; wrong branch, row, position, or tail selection
is caught by RED fixtures.

### F3 — Residency, PLE mmap owner, and quant admission

Extend materialization without copying Qwen35’s hardcoded H5120 policies:

- roots and small F32/BF16/quant tensors have one device owner;
- rank-3 512-expert Q4/Q5/Q6 tensors use an existing compatible raw/selected
  layout first;
- PLE is deferred from device materialization and exposed as a dedicated sparse
  mmap table;
- a bounded double-buffered pinned row-gather stages only current PLE rows;
- planner reports raw payload, replacement payload, alternate bytes, PLE mmap
  bytes, device bytes, scratch, and state separately; and
- no allocation is duplicated merely to satisfy decode and prefill.

CPU Q4_0 row dequant is the strict first PLE implementation. A GPU mapped-row
consumer is optional later and must beat this path without pinning the whole
file.

**Gate:** complete real-file planning fits the declared 128-GiB lane; one PLE
owner, zero PLE device/shadow bytes, zero alternate weight bytes; selected PLE
rows match GGUF CPU dequant; allocation and mmap owners close cleanly after
injected failures and normal teardown.

### F4 — Native GR, PLE, and reused GDN/MoE primitives

Before any kernel port, finish reading `KERNELS.md`, run:

```bash
python3 scripts/check_lineage.py --kind kernel --diff stat
```

Then add only missing primitive families:

- GR grouped norm/read gate/mean and injection write, each with strict unfused
  dense-linear + CPU-reference fallback;
- PLE row upload, key/value projection, grouped gate and dilated Conv/state;
- sigmoid GDN output-gate sibling; and
- shape/capacity extensions for 512-expert top-10 MoE where existing kernels
  are otherwise operation-compatible.

Use existing quant projection kernels through registry keys. Do not fork an
H2560 copy solely to change constants when a generic kernel already handles the
shape.

**Gate per kernel:** strict/parent-parity RED, CPU outer floor, launch smoke,
registered strict fallback, expected `rocprofv3 --kernel-trace` symbol and
plausible duration. Complete F4 gate runs a reduced real layer and verifies GR,
GDN/MoE output, PLE state, and lifecycle.

### F5 — Text AR below the QSA budget

Build a dedicated `Qwen4ExpGGUFResidentModelRunner` rather than adding Qwen4Exp
branches to `qwen35_gguf_runner.py`. Shared quant/runtime helpers may be
factored only at stable plugin seams.

First supported scope:

- text-only;
- c1;
- greedy;
- strict profile;
- BF16 K/V and FP32 GDN state;
- prompt + generated context <=2,051; and
- dense full-attention execution, which is mathematically identical to QSA in
  this range.

The widened GR state, PLE history, GDN state, attention K/V, token positions and
sampler ownership are persistent and request-local. Prefill may begin
correctness-first but may not replay the entire prefix per decode token.

**Gate:** same-Q4 llama.cpp teacher-forced full logits on category+heldout
prompts; mean/tail/max KL and top-1 under the strict/production policy; exact
selected IDs where the strict comparator contract declares it; repeat
determinism; prompt chunk boundaries; fresh-prefix recomputation; zero teardown.
`LLM.generate()` and the CLI resolve the new plugin without architecture
branches.

### Current retained text evidence (2026-08-27)

On physical host `zbook` (`machine-id 87c566d30a5645cf8d12ed7ef6b6e1e8`),
AMD Ryzen AI Max+ Pro 395 / Radeon 8060S (`gfx1151`, 133,143,986,176 device
bytes), the pinned four-part artifact is 111,334,654,784 bytes / 1,224 tensors.
All four expected part hashes pass. The scanner reports no unknown or duplicate
tensors, 82,523,491,840 hot-device bytes, zero alternate/replacement layouts,
and one 28,800,138,240-byte IQ4_NL PLE mmap tensor shaped
`[320001536, 160]` (raw bytes `[320001536, 90]`).

Frozen llama.cpp PR #27742 (`bea3b12da`, `llama-debug` SHA-256
`e36ea6554f6112c02ef0c2d7bf20549a5f4b5ef6530f433ba4c0915b10bf5426`)
versus hipEngine on all 10 prompts in
`benchmarks/prompts/mtpbench-code-general-ja.jsonl` measured mean/p95/p99/max
teacher→hipEngine KL `0.01406 / 0.04154 / 0.04776 / 0.04931` and `10/10`
top-1 agreement. The separately predeclared eight category-heldout prompts,
run with matched BF16 K/V in the corrected merged process, pass mean/p95/p99/max
KL `0.00987 / 0.02331 / 0.02766 / 0.02874` and top-1 `8/8`; the complete
merged 18-row diagnostic is not called a pass because one canonical repeat
exceeded the ceiling. The public API resolved `gguf_ud_q4_k_xl` and generated
`" 4.\n\n"` for `The answer to 2 + 2 is`. Measured tracked peak was
82,718,198,780 bytes; close returned active allocations/current bytes to zero.
Exact commands, hashes, category rows, memory plan, and unsupported-scope list
are retained in
[`2026-08-27-gfx1151-qwen38-flash-next-text-bringup.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-text-bringup.json)
and the heldout subset is in
[`2026-08-27-gfx1151-qwen38-flash-next-heldout-logits.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-heldout-logits.json).
These are correctness/bring-up results, not a speed or 262K-capacity claim.

The first real native sparse-QSA row is additionally qualified at token 2,052
with a repeated-token structural prompt: frozen llama.cpp, strict serial, and
size-2 chunked hipEngine all select token 264; teacher→serial, teacher→chunk,
and serial→chunk KL are `7.65e-5`, `4.98e-5`, and `3.95e-5`, respectively,
with zero tracked bytes after close. On the promoted exact chunk64 route, the
same first sparse row is bit-exact to serial (KL/max error 0) and improves
strict→chunk wall `370.565→136.129 s` (2.722x); versus the historical size-2
chunk wall, this is a 44.630% reduction. A repeated-token structural 4,096-token checkpoint
also passes: teacher→serial/chunk KL `4.40e-5/4.78e-5`, serial→chunk KL
`3.19e-5`, top-1 264 exact, and zero tracked bytes after close; diagnostic
serial/chunk walls are `854.982/574.759` seconds. A practical chunk-only 16K
checkpoint also passes teacher KL `7.55e-5`, top-1 264 exact, and zero teardown
bytes in 2,434.172 seconds; a chunk-only repeated-token 64K checkpoint passes
teacher KL `5.74e-6`, top-1 264 exact, and zero teardown bytes in 10,336.580
seconds. Strict serial remains measured through 4K. This does not close natural
retrieval, selected-index, strict-above-4K, 262K inference, or lifecycle/
isolation gates. Separately, the real complete
262,144-token owner allocates successfully at 91,126,119,496 tracked bytes,
leaves 38,915,162,112 physical bytes free, and returns to zero tracked bytes on
close; this is capacity/lifecycle evidence only. Exact evidence is in
[`2026-08-27-gfx1151-qwen38-flash-next-qsa-2052-transition.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-2052-transition.json),
[`2026-08-27-gfx1151-qwen38-flash-next-qsa-2052-exact-prefill.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-2052-exact-prefill.json),
and
[`2026-08-27-gfx1151-qwen38-flash-next-qsa-4k.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-4k.json), and
[`2026-08-27-gfx1151-qwen38-flash-next-qsa-16k.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-16k.json),
[`2026-08-27-gfx1151-qwen38-flash-next-qsa-64k.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-64k.json), and
[`2026-08-27-gfx1151-qwen38-flash-next-262k-capacity.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-262k-capacity.json).

A natural chat-formatted 4K archive-retrieval gate now passes: the code at token
720 is selected by all 12 QSA layers and generated exactly, final-layer selected
positions match the pinned Transformers CPU formula oracle 2,048/2,048, replay
and abandoned-branch rollback are bit-exact, and teardown is zero. A matched
special-token llama PR #27742 diagnostic has KL `3.31e-12` and exact top-1.
Correctness-first prefill is 304.944 seconds before persistent pooling and
303.528 seconds after persistent pooling, then 294.434 seconds after exact
device top-k. The retained structural wins are pool launches `24,540→384`,
prepared block work `18,849,792→12,288`, and removal of 24,540 score D2H
synchronizations plus 403.341 MB of selection-metadata H2D. Production wave32
H128 sparse attention then improves the primitive 9.41% and paired natural 4K
298.078→290.941 seconds; four sparse categories and the retrieval task have
bit-exact final logits/control. Exact chunk-batched score/top-k then reduces
launches `49,080→768` and paired natural 4K `295.706→290.971 s` (1.60%).
Exact grouped Q4_K gate/up then reuses each dequantized weight across adjacent
expert rows and cuts paired natural 4K `291.624→231.798 s` (20.51%). Exact
output4 scheduling then cuts full-shape CTAs 75% and paired natural 4K
`235.774→228.569 s` (3.06%). Exact Q5_1-down output8 scheduling then cuts
its full-shape CTAs 87.5% and paired natural 4K `237.131→222.228 s` (6.28%).
Evidence:
[`2026-08-27-gfx1151-qwen38-flash-next-natural-4k-qsa.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-natural-4k-qsa.json) and
[`2026-08-27-gfx1151-qwen38-flash-next-persistent-qsa-pool.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-persistent-qsa-pool.json), and
[`2026-08-27-gfx1151-qwen38-flash-next-device-qsa-topk.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-device-qsa-topk.json), and
[`2026-08-27-gfx1151-qwen38-flash-next-wave32-sparse-attention.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-wave32-sparse-attention.json), and
[`2026-08-27-gfx1151-qwen38-flash-next-batched-qsa-selection.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-batched-qsa-selection.json), and
[`2026-08-27-gfx1151-qwen38-flash-next-exact-grouped-q4-gate-up.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-exact-grouped-q4-gate-up.json), and
[`2026-08-27-gfx1151-qwen38-flash-next-exact-grouped-q4-out4.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-exact-grouped-q4-out4.json), and
[`2026-08-27-gfx1151-qwen38-flash-next-exact-grouped-q5-1-out8.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-exact-grouped-q5-1-out8.json).

### Implementation-first context escalation guardrails

The binding execution order is:

1. **Fully working first at 512/1K.** Complete basic text AR, MTP, public
   serving/lifecycle, vision, and c2 isolation at short context.
2. **Optimize short-context prefill second.** Keep performance work and its
   complete-model gates at 512/1K until the next threshold is reached.
3. **Qualify longer context last.** Move up one permitted context rung at a
   time, running that rung's full correctness/lifecycle/memory gate only after
   short-context functionality and speed permit it.

These are hard campaign stop rules, not aspirational targets:

| Full-model context proposed | Minimum retained prompt-prefill rate required first |
| --- | ---: |
| 4K | `20 tok/s` |
| 16K | `50 tok/s` (**met: p508 first/steady 51.220/58.466; p1006 55.046**) |
| 64K | `100 tok/s` |
| 128K or greater (including 262K) | `200 tok/s` |

- Until the basic product gate below is complete, implementation and full-model
  validation stay at **512 or 1,024 tokens** regardless of measured speed.
- The threshold is a same-host, natural-prompt, complete-model prompt-prefill
  result from the current retained default at an already-allowed context. A
  microbenchmark, repeated-token-only row, rejected numerical profile, or
  external engine result cannot unlock a larger context.
- Reduced primitive fixtures may exercise position arithmetic above 1K, but
  must not allocate/run a full long-context model or become a performance row.
- Existing 4K/16K/64K artifacts remain historical correctness evidence. Do not
  rerun them until the corresponding threshold is met.
- Crossing a threshold permits the next scale; it does not replace that scale's
  correctness, lifecycle, and memory gates. Only an explicit user override may
  waive these stop rules.

### 512/1K basic product gate

Before returning to long-context verification, complete the basic end-to-end
implementation at 512/1K:

1. Audit llama.cpp PR #27742 as the primary formula/converter/runtime guide and
   maintain a hipEngine parity checklist; then audit EngramHalo's delta.
2. Keep target text AR working through `LLM.generate()` and the public blocking
   and SSE server surfaces, including reasoning/chat-template/tool-call basics,
   cancellation, reset, and clean teardown.
3. Produce/load the official MTP sidecar and pass proposal, target verification,
   accept/commit/rollback, deterministic greedy output, and a small natural
   multi-prompt economics smoke against true AR.
4. Add the basic vision path: one supported image fixture through processor,
   placeholders/MRoPE, encoder/mmproj, generation, and text-only non-regression.
5. Pass c1 plus a basic c2 isolation/cancellation smoke at 512/1K.

Long-context work resumes only after all five items work and the throughput
ladder above permits the requested context.

**Basic-gate status (2026-08-28): complete in the bounded declared scope.**

- Reference audit: final #27742 and pinned EngramHalo MTP deltas are mapped.
- Text/public serving: AR, blocking/SSE completion, chat reasoning/tool rendering,
  cancellation-before-mutation, reset/close, and exact tokenizer controls pass.
- MTP: Q8 sidecar loads; natural p512/p1008 IDs equal AR; draft acceptance is
  82.26%/93.33%; blocking/SSE pass. Serial verification is slower, so AR is
  still default and MTP is not a performance promotion.
- Vision: one exactly 32×32 RGB image passes the 27-layer encoder, independent
  Transformers embedding parity, public generation, and text-only isolation.
- c2: two varied p512 rows preserve exact independent AR output under isolated
  sequential target/draft state. Native parallel c2 is not claimed.

Evidence is linked under F8/F9 and in
[`2026-08-28-gfx1151-qwen38-flash-next-short-serving-basics.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-short-serving-basics.json).
The campaign now stays at 512/1K for optimization. Do not turn completion of
this bounded gate into an automatic long-context run.

### F6 — Native QSA and 262K context ownership

Add a QSA index-cache backend or model-attention state that mirrors
`KVLiveSpans` exactly. Correctness-first may retain raw per-token BF16 index
keys. Promotion compresses complete blocks to one BF16 key per four tokens and
keeps a bounded 0–3-token tail ring.

Required kernels/operations:

- raw index K write;
- FP32 block mean, grouped RMSNorm and partial MRoPE;
- 4-head block scoring;
- deterministic top-512 selection;
- selected original-K/V compaction or sparse paged attention; and
- chunked prefill with exact block/tail semantics.

Scopes include c1 first, then c2/c4/c8 only after row ownership is proven.
QSA index selection is control metadata and must be exact for a fixed strict
schedule; production arithmetic may change scores only under a declared T1/T2
profile gate.

**Gate:** QSA vs dense exact below budget; indices against Transformers/llama
reference above budget; long category/retrieval rows at 4K/16K/64K/262K;
non-contiguous pages, prefix reuse, cancellation, compaction, transaction
rollback, graph/eager repeats; no index/KV cross-request contamination. Capacity
and speed claims report K/V and index bytes independently.

### F7 — Prefill, graphs, batching, and performance

After F5/F6 correctness, add:

- bulk GR and MoE prefill;
- chunked PLE prefetch overlapped with layer 0 where measurable;
- QSA prefill by context bucket;
- c-aware quant projection selection;
- stable scratch arenas; and
- graph/PM4 capture only for fully qualified fixed shapes.

Benchmark exact repeated-token shapes for structural profiling and the full
natural category suite for product claims. Baselines are same-artifact
llama.cpp PR #27742 HIP and Vulkan, plus hipEngine AR before each retained
change. Physical host identity is mandatory. A user-reported, independently
hosted Strix Halo 128-GB llama.cpp PR #27742 run (`UD-Q4_K_XL`, Ubuntu
24.04.4, kernel 7.2.0, ROCm 10.1.0) reports pp33k `285 tok/s` and decode
`13 tok/s`. A second independent Strix Halo report using Vulkan and
`UD-IQ4_XS` records pp512 `390.3 tok/s`, pp4096 `357.5 tok/s`, tg128
`23.0 tok/s`, and at depth 16K pp512/pp4096/tg128
`305.3/317.9/19.4 tok/s`; Q4-XL comments report roughly `260–400 tok/s`
prefill and `16–17 tok/s` short-context decode, falling toward `12 tok/s`
around 100K. The first binding same-host/same-GGUF PR #27742 run now measures
Vulkan/HIP pp508 `316.380/274.996 tok/s`, pp1006 `290.450/284.485`, and tg32
`18.716/15.848`, versus retained hipEngine `58.466/55.046/5.890`. Vulkan is
therefore 5.41x/5.28x/3.18x faster on the shape-matched rows; closing this gap
is the active F7 priority. Evidence:
[`2026-08-28-gfx1151-qwen38-flash-next-llamacpp-matched-baseline.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-llamacpp-matched-baseline.json).
External report sources:
[`r/StrixHalo benchmark`](https://www.reddit.com/r/StrixHalo/comments/1vz5yb3/qwen38flashnext_125ba6b_running_on_strix_halo/)
and
[`r/LocalLLM PLE mmap/layout`](https://www.reddit.com/r/LocalLLM/comments/1vz927j/got_qwen38nextflash_ngram_ssd_offload_working_in/).
Treat external rows as target lines only—not old→new comparisons to this host.
The PR #27742 HIP/Vulkan rows above are now the reproduced same-host references.

#### Active Q4 parity order (review reset, 2026-08-28)

The completed baseline and decode trace supersede the earlier feature/checklist
priority. Before graph replay, exact Q8/Q4 packing/fusion brought controlled decode to
about **6.42 tok/s** versus llama.cpp HIP/Vulkan `15.85/18.72 tok/s`; eager
trace ownership remained about `1,861 launches/token`. Request-owned stateless
MoE graphs plus exact Q5_1 logical256/physical64 and Q4_K
logical128/physical64 decode owners plus exact fused Q5 down+weighted combine
now reach **13.523 tok/s** counterbalanced, with all generated IDs and full-logit
SHA rows exact. Exact strict remains 1.17x behind llama.cpp HIP; explicit
`production` reaches **15.543 tok/s** and closes that gap to **1.020x**. Current paired
strict/production p508 is **61.40/78.31 tok/s** versus same-host llama.cpp
Vulkan/HIP `316/275 tok/s`.

Binding implementation order:

1. **Contract exact prefill plumbing first.** PLE now flattens all
   `16 × chunk_rows` indices and gathers/dequantizes once per chunk; registered
   decode-order-exact K4 bulk Conv replaces row-serial launches. Together they
   cut p508 kernel launches `29,341→11,053` and improve counterbalanced p508
   `57.825→58.408 tok/s` (+1.01%) with bit-exact logits. Block-table-aware QSA
   index scatter then replaces 6,096 p508 per-row D2D copies with 24 kernels;
   the paired wall is neutral (`58.678→58.669 tok/s`) but p512 trace launches
   fall another `11,053→4,933`. Only after grouped decode ownership should
   asynchronous PLE next-chunk overlap be revisited.
2. **Contract MoE submission and then its grids.** Audit corrected an important
   premise: selected Q4/Q5 already launch once per layer and cover all 10 routed
   rows; workgroups, not Python calls, own individual output elements. The
   existing self-validating `MoeGraphCache` now collapses each exact stateless
   per-layer chain to one graph launch and improves eager `6.511→11.515 tok/s`
   (1.769x), with stateful GDN/QSA excluded. The exact Q5_1 decode owner now
   maps two logical partials per physical lane, preserves the strict 256-slot
   tree, and improves graph decode `11.380→12.140 tok/s` (+6.27%). The analogous
   exact Q4_K physical64 owner then reaches `13.167 tok/s` (+8.84%), and a
   second exact Q5 contraction to physical64 reaches `13.302 tok/s` (+1.69%);
   fusing exact per-route BF16 down publication with ordered weighted `fmaf`
   reaches `13.523 tok/s` (+1.06%). Production one-plane Q8_1 DP4A is rejected
   on all Q4 layers at 445/450 top-1. Contiguous bisection finds suffix13 passes
   447/450 while suffix12 fails 445/450; a complete early-layer screen then
   certifies static layers `0,2,5,6,8,9,10,11,13–47` together at 447/450.
   Layers `1,3,4,7,12` remain exact. The 43-layer profile reaches
   `15.543 tok/s` (+10.70%).
   Prefill now uses certified dense-Q8 and selected Q5_1/Q4_K MMQ routes;
   omitted layers stay strict. Do not widen any calibrated prefill/decode set
   without a fresh complete profile gate.
3. **Retain exact Q4/Q5 micro-wins only atop the grouped shape.** The Q5_1
   logical256 contraction is now the selected default at 64 physical threads:
   the first t128 step cut Q5 cycle-wall `692.930→410.364 ms`, and t64 then cuts
   its matched trace `444.699→362.525 ms` across 1,806 launches while output bits
   stay exact. Q4_K maps logical lanes `tid` and `tid+64` onto each physical
   lane while publishing the same four wave sums; its cycle-wall falls
   `1,076.767→814.906 ms` across 1,974 launches and graph decode improves
   `12.003→13.167 tok/s` (+8.84%). Q5 down+weighted fusion then removes 1,806
   traced launches and contracts its target cycle-wall `369.241→313.535 ms`
   while preserving every route's BF16 publication and combine `fmaf` order.
   Further packing work must retain these trees.
   The naïve raw
   Q4 selected pack8 (`+0.05%`), Q5_1 pack8 (`-6.3%`), Q5 metadata LDS cache
   (`-2.9%`), and device argmax (`-0.09%`) screens are rejected and removed.
4. **Mine Vulkan cooperative-matrix geometry first, HIP second.** The same-host
   HIP profile is complete. pp508 has `1.798 s` kernels / 5,543 launches:
   Q4_K/Q5_1/Q8_0 `0.554/0.382/0.290 s`, dense rocBLAS `0.119 s`, GDN
   `0.085 s`. tg32 has `40.71 ms/token` kernels: Q8/Q4/Q5_1/dense
   `21.63/4.76/3.06/2.74 ms`. llama.cpp emits ~4,362 kernel rows/token under
   tracing—more than hipEngine—so fewer rows alone are not the explanation;
   MMQ/cooperative grid dataflow is. Use Vulkan's faster path as the primary
   shape proof and report achieved bytes/s before calling a projection
   bandwidth-bound.
5. **Measure submission honestly.** HIP-event instrumentation shows median
   unprofiled step wall/stream/post-stream-host `193.91/193.74/0.18 ms`; logits
   D2H/NumPy is not the old inferred 55 ms gap. Inter-launch stalls live inside
   the stream span, while `rocprofv3` inflates them further. Continue graph/
   grouped submission work, but never report profiler wall-minus-kernel as
   unprofiled Python overhead.
6. **Maintain strict and production targets.** T0 packing/fusion stays exact.
   T1/T2 math requires full category+heldout rows, three repeats, state/task/c2,
   a manifest, and registered strict fallbacks. The final named profile selects
   Q8 MMQ, Q5_1 MMQ 32–47, Q4_K MMQ 35–47, peer-GDN 35–47, and DP4A safe-43.
   Its complete gate passes mean/p95/p99/max KL
   `3.16e-4/1.61e-3/4.25e-3/9.92e-3`, 448/450 top-1, all scopes, exact
   state/repeat/c2, and deterministic task generation. Production/strict
   manifest hashes are `3b7a0644…` / `9e648eb8…`; omitted routes stay strict.
7. **Keep MTP separate.** It may improve serving economics only under the full
   anti-gaming suite and same-protocol no-MTP denominator; it cannot mask the
   base AR or 5× prefill gap.

Same-host HIP kernel evidence:
[`2026-08-29-gfx1151-qwen38-flash-next-llamacpp-hip-kernel-profile.json`](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-llamacpp-hip-kernel-profile.json).

All sub-3% A/Bs are counterbalanced in one residency after warmup; fixed clocks
are preferred, and globally shifted profiles are used only for symbol/launch
structure, not speed claims. Current T0 PLE/Conv evidence:
[`2026-08-29-gfx1151-qwen38-flash-next-exact-ple-conv-bulk.json`](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-exact-ple-conv-bulk.json) and
[`2026-08-29-gfx1151-qwen38-flash-next-exact-qsa-index-scatter.json`](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-exact-qsa-index-scatter.json).

The pinned vLLM/SGLang implementations establish the long-context performance
design. hipEngine now updates its persistent compressed-QSA K cache only when a
four-token group completes and performs deterministic GPU top-k/block-tail
expansion without score D2H or host sorting. The production H128 sparse kernel
uses one barrier-free wave32 per query head and retains the strict shared-memory
fallback. Prompt score generation/top-k and strict Q4_K gate/up are now
batched/grouped per chunk. Split-k attention is optional follow-up only if its
production packet beats wave32. Natural 16K now passes in `946.9997 s`
(`17.301 tok/s`): exact retrieval, all-layer needle control, CPU index oracle,
replay/rollback, and zero teardown. Evidence:
[`2026-08-27-gfx1151-qwen38-flash-next-natural-qsa-16k.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-natural-qsa-16k.json).
Natural 64K also passes in `3832.663 s` (`17.099 tok/s`) with exact
retrieval/control/CPU-oracle/replay/rollback and zero teardown. Evidence:
[`2026-08-27-gfx1151-qwen38-flash-next-natural-qsa-64k.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-natural-qsa-64k.json).
Further long-context execution is deferred by the implementation-first
escalation guardrails: 128K+ requires a retained `>=200 tok/s` prompt-prefill
row and the 512/1K basic product gate. MTP step 0
selects target-aligned QSA rows and later draft steps reuse those indices.

Current exact F7 default (updated 2026-08-28): immediate PLE ring ownership,
batched projections, exact row-serial causal Conv, exact fixed-worker Q4_K/
Q5_1 experts, and raw-Q8 FP32 coltile8/rowbatch4. The expert kernels iterate the
512-entry prefix map through 64 worker CTAs; Q4 shares one block decode across
paired strict columns, while Q5 uses 128 physical threads to materialize the
same 256 logical partials and original reduction tree. The Q8 tile preserves
each scalar thread's K order/reduction tree while sharing scheduling across
eight outputs. Relative to the preceding exact default, natural p508 improves
`14.718→11.988 s` (`42.376 tok/s`, 1.228x) and paired p1012
`31.346→24.432 s` (`41.422 tok/s`, 1.283x), with bit-exact full logits. Cached
p512 kernel wall falls `14.230→11.692 s`; grouped Q4/Q5/Q8 buckets fall
`3.971→2.870`, `3.470→2.534`, and `3.121→2.482 s`. Extending the exact grouped
Q4 gate/up owner from Q5_1-down layers to all Q4 gate/up layers then removes 64
direct launches and improves paired p508 another `12.021→11.189 s` (6.92%,
**45.404 tok/s**) with bit-exact logits. Evidence:
[`2026-08-28-gfx1151-qwen38-flash-next-all-q4-grouped.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-all-q4-grouped.json).
The earlier chunk64 path remains bit-exact on all 687
teacher-forced category+heldout rows; its pre-Q8 natural-suite row was
`5.265→12.117 tok/s` and is now historical. Public generation/lifecycle pass.
Evidence:
[`2026-08-28-gfx1151-qwen38-flash-next-exact-scheduling-wave.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-exact-scheduling-wave.json),
[`2026-08-28-gfx1151-qwen38-flash-next-exact-q8-coltile-prefill.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-exact-q8-coltile-prefill.json), and
[`2026-08-27-gfx1151-qwen38-flash-next-exact-prefill-promotion.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-exact-prefill-promotion.json).
After the bounded basic/short optimization gate closed, the permitted current-
default natural 4K requalification passes at `146.883 s` / `27.886 tok/s`,
versus the prior exact `222.228 s` (33.90% lower, 1.513x). Retrieval, all 12
needle controls, CPU top-512 selection, replay/rollback, and teardown pass.
Evidence:
[`2026-08-28-gfx1151-qwen38-flash-next-natural-qsa-4k-current.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-natural-qsa-4k-current.json).
The subsequent exact Q4-metadata/chunk256 promotion reaches **51.220 tok/s on
the first p508 run and 58.466 tok/s steady median**, plus **55.046 tok/s at
p1006**, all with bit-exact full logits. The newly eligible current-default
natural 16K gate then passes in `364.306 s` / **44.973 tok/s**, versus prior
exact chunk64 `946.999 s` (61.53% lower, 2.599x). Retrieval, all 12 controls,
CPU top-512, replay/rollback, and teardown pass. Current 64K is not rerun because
`44.973 < 100 tok/s`; historical 64K remains retained. Evidence:
[`2026-08-28-gfx1151-qwen38-flash-next-exact-q4-metadata-chunk256.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-exact-q4-metadata-chunk256.json) and
[`2026-08-28-gfx1151-qwen38-flash-next-natural-qsa-16k-current.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-natural-qsa-16k-current.json).

Explicit gfx1151 `production` selects guarded dense-Q8 MMQ, Q5_1 down MMQ
layers 32–47, Q4_K dual MMQ layers 35–47, compact peer-GDN global layers
35–47, and DP4A Q4 gate/up+SiLU on calibrated decode layers
`0,2,5,6,8,9,10,11,13–47`. Manifest `3b7a0644...` falls back to strict
`9e648eb8...`. The combined 450-row gate passes mean/p95/p99/max KL
`3.16e-4/1.61e-3/4.25e-3/9.92e-3`, **448/450 top-1**, exact state/repeat/c2,
and deterministic task generation. Direct paired p508/p1012 improves
**8.273→6.487 s (-21.59%, 78.31 tok/s)** /
**16.810→13.178 s (-21.61%, 76.79 tok/s)**; decode remains **15.543 tok/s**.
Omitted routes remain exact strict. Evidence:
[`2026-08-29-gfx1151-qwen38-flash-next-prefill-mmq-campaign-final.json`](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-prefill-mmq-campaign-final.json),
[`2026-08-29-gfx1151-qwen38-flash-next-production-gdn-peer35.json`](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-production-gdn-peer35.json), and
[`2026-08-29-gfx1151-qwen38-flash-next-production-mmq-profile-manifest.json`](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-production-mmq-profile-manifest.json).

The broader default-off F7 research candidate still remains rejected: grouped Q4/Q5/Q8 MoE,
Q5_1 WMMA down, all-layer peer-GDN, and tuned Q4/Q8 tiles raise warm repeated-token 512
prefill from `8.67` to `211.76 tok/s` and the 18-prompt natural suite from
`5.36` to `35.43 tok/s` after the PLE ring-ownership fix. It is not promoted:
strict→candidate mean/p95/p99/max KL are now
`0.00476/0.01458/0.01530/0.01548` and top-1 is `100%`; mean/p95 still miss the
production envelope. The binding 687-row teacher-forced packet rejects the
same profile more strongly: mean/p95/p99/max KL
`0.01280/0.05553/0.12148/0.82237`, top-1 `94.47%`, and every category below its
97% top-1 floor. Warm eager
decode baseline was `5.74 tok/s`; an exact Q5_1 wave-tail reduction first raised
the strict path to `5.89 tok/s`. The subsequent exact raw-Q8 F32 output-pack8
owner cuts its traced Q8 bucket `2.620→1.171 s` and a counterbalanced complete-
model decode median `5.698→6.305 tok/s` (+10.66%). Registered exact Q4 selected
dual gate/up then halves Q4 launches `94→47/token` and improves its paired
median `6.065→6.223 tok/s` (+2.61%). Fusing the exact BF16-boundary SiLU/product
into that owner removes another 47 launches/token and improves the next paired
median `6.400→6.420 tok/s` (+0.31%). Exact request-owned per-layer MoE graph
replay then gives the structural jump: `6.511→11.515 tok/s` (1.769x), 192/192
full-logit rows exact, 48 captures/zero rejects, c2 exact, and teardown zero.
The exact Q5_1 logical256/physical128 selected owner then improves complete graph
decode `11.380→12.140 tok/s` (+6.27%) and contracts Q5 cycle-wall
`692.930→410.364 ms` (-40.78%) across 1,806 launches with strict BF16 bits,
full logits, and IDs exact. The analogous exact Q4_K logical128/physical64
owner then improves graph decode `12.003→13.167 tok/s` (+8.84%) and contracts
Q4 cycle-wall `1,076.767→814.906 ms` (-24.32%) across 1,974 launches, again
with strict bits, full logits, and IDs exact. A second exact Q5_1 contraction
to physical64 then improves `13.077→13.302 tok/s` (+1.69%) and contracts its
matched Q5 cycle-wall `444.699→362.525 ms` (-18.48%), again with every strict
partial and output bit preserved. Exact Q5 down+weighted fusion then improves
`13.379→13.523 tok/s` (+1.06%), removes 1,806 traced launches, and contracts
matched target cycle-wall `369.241→313.535 ms` (-15.09%) with fused/unfused
BF16 bits exact. All add no tracked resident bytes. A Q5_1
wave64 decode candidate reaches
`6.10-6.22 tok/s` but is rejected at production mean/p95 KL
`0.002565/0.007202`. Exact evidence and rejected sub-experiments are in
[`2026-08-29-gfx1151-qwen38-flash-next-exact-moe-graph-decode.json`](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-exact-moe-graph-decode.json),
[`2026-08-29-gfx1151-qwen38-flash-next-exact-q5-logical256-t128-decode.json`](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-exact-q5-logical256-t128-decode.json),
[`2026-08-29-gfx1151-qwen38-flash-next-exact-q4-logical128-t64-decode.json`](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-exact-q4-logical128-t64-decode.json),
[`2026-08-29-gfx1151-qwen38-flash-next-exact-q5-logical256-t64-decode.json`](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-exact-q5-logical256-t64-decode.json),
[`2026-08-29-gfx1151-qwen38-flash-next-exact-q5-fused-weighted-decode.json`](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-exact-q5-fused-weighted-decode.json),
[`2026-08-27-gfx1151-qwen38-flash-next-prefill-grouped-candidate.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-prefill-grouped-candidate.json),
[`2026-08-27-gfx1151-qwen38-flash-next-ple-staging-fix.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-ple-staging-fix.json),
[`2026-08-27-gfx1151-qwen38-flash-next-fast-allrows-rejected.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-fast-allrows-rejected.json),
[`2026-08-27-gfx1151-qwen38-flash-next-q5-wave-tail.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-q5-wave-tail.json), and
[`2026-08-27-gfx1151-qwen38-flash-next-decode-wave64-candidate.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-decode-wave64-candidate.json).
A one-layout 45.587-GB Q4T16 replacement was also measured and removed:
optimized p512 was neutral (`213.52` vs `211.76 tok/s`), paired decode regressed
`5.925→3.615 tok/s`, and mean/p95 KL failed (`0.003010/0.008338`). See
[`2026-08-27-gfx1151-qwen38-flash-next-t16-replacement-rejected.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-t16-replacement-rejected.json).
A one-layout pack8 replacement was likewise removed: the sampled layer-0 kernel
was bit-exact and 4.47x faster, but full load/repack took 979 seconds and paired
whole-model mean/p95 KL failed at `0.002089/0.006529`. See
[`2026-08-27-gfx1151-qwen38-flash-next-pack8-replacement-rejected.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-pack8-replacement-rejected.json).

**Promotion:** keep every exact/non-regressive measured win in scope. Update
benchmark rollups only for retained complete-model rows, never microbench-only
or single-prompt results.

### F8 — MTP

Export or convert the official one-layer MTP sidecar with a frozen converter
identity. Do not block target AR on MTP artifact production.

MTP differences from Qwen3.8 dense NextN:

- hidden seed is the four-branch 10,240-wide target state;
- separate embedding/hidden norms and projections fuse to the MTP input;
- one GR/QSA/MoE layer, no PLE;
- the draft has its own transactional K/V/index state; and
- QSA top-k may use IndexShare only after exact non-reuse semantics pass.

Reuse Generation-2 proposal/verify/accept/commit interfaces. Default remains
AR/K0 until the complete multi-prompt suite proves exact target results,
GPU-accept==CPU, rollback/repair, lifecycle, and positive economics against
true same-session AR.

Current short-context component status (2026-08-28): the pinned 34-tensor Q8_0
sidecar map/materializer, authoritative widened target-hidden handoff, and
request-owned one-layer draft runner exist. The draft independently normalizes
embedding/H10240 target hidden, applies fused `eh_proj`, executes one dense HC +
attention + 512/top-10 MoE block, retains its own K/V cursor, chains widened
hidden for budgets 1–4, and emits final logits. Its CPU input formula,
sidecar-only finite output, deterministic repeat, and snapshot/restore replay
pass at a 16-token reduced capacity. This is **not yet a working MTP product**:
real target prompt priming, sequential/batched target verification, public
blocking/SSE, cancellation, exact generated output, and natural economics are
still binding. The first public registry-attached smoke now also passes at
prompt 16/output 4: AR and MTP both emit IDs `[264,264,264,264]`, provider
capabilities resolve, exact serial verification trims the draft cursor, and the
buffered public stream returns the same IDs. This proves integration only; its
cold/warm synthetic timings are not a speed comparison. Evidence:
[`2026-08-28-qwen38-flash-next-mtp-q8-sidecar.json`](../benchmarks/results/2026-08-28-qwen38-flash-next-mtp-q8-sidecar.json),
[`2026-08-28-gfx1151-qwen38-flash-next-mtp-draft-smoke.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-mtp-draft-smoke.json), and
[`2026-08-28-gfx1151-qwen38-flash-next-mtp-public-smoke.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-mtp-public-smoke.json).
The complete 10-prompt category+heldout short-context packet now supersedes
the earlier four-row smoke: **10/10 exact AR generated-ID rows**, `134/159`
accepted drafts (`84.28%`), and clean teardown. One code prompt reaches
`1.198x` AR, but aggregate MTP remains slower at `0.955x`, so MTP stays
explicit opt-in and AR stays default. This completes F8 with a successful
functionality/correctness gate and a failed promotion/economics gate. A
multirow target-verification candidate was removed after it changed a
continuation-row target token; serial exact verification remains binding.
Evidence:
[`2026-08-28-gfx1151-qwen38-flash-next-mtp-fullsuite-short.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-mtp-fullsuite-short.json),
[`2026-08-28-gfx1151-qwen38-flash-next-mtp-natural-512.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-mtp-natural-512.json), and
[`2026-08-28-gfx1151-qwen38-flash-next-mtp-batch-verify-rejected.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-mtp-batch-verify-rejected.json).
The 1K edge and real OpenAI completion surfaces pass too: natural
`1008+16` IDs equal AR with `14/15` draft acceptance; explicit speculative
`/v1/completions` returns HTTP 200 on blocking and buffered SSE, reports the
`speculative` route, and closes to zero tracked bytes. Serial verification is
still slower (`0.807x` AR), so this closes basic 1K/blocking/SSE functionality,
not economics or default promotion. Evidence:
[`2026-08-28-gfx1151-qwen38-flash-next-mtp-1k-http.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-mtp-1k-http.json).

### F9 — Vision

Add the Qwen3-VL-compatible 27-layer vision encoder and multimodal processor as
separate model/layer plugins. Text completion remains available without the
optional vision artifact.

Required semantics include patch embedding, learned position interpolation,
vision RoPE, 27 attention/MLP blocks, spatial merge, image/video placeholders,
multimodal position IDs, and PLE hashing of the official placeholder token.
Vision input must not import torch on the hot path; torch/DLPack remains an
optional user-boundary bridge.

**Gate:** image and video fixtures vs Transformers/llama mmproj, placeholder
count failures, MRoPE positions, text-only non-regression, and multimodal
category smokes. Only then advertise multimodal support.

Current <=1K scope (2026-08-28): general merge-compatible RGB grids (positive
height/width multiples of 32, at most 256 patches per temporal pair), multiple
images, and videos are supported through `LLM.generate_multimodal_detailed()`.
Images duplicate the official temporal width 2; videos consume adjacent frame
pairs and duplicate an odd final frame. Patch rows use official 2×2 block-major
order, align-corners learned-position interpolation, frame-isolated ViT
attention, and the stock merger. Typed image/video placeholder groups are
expanded and count-checked. Text QSA uses explicit interleaved T/H/W MRoPE
sections `[11,11,10]`; continuation decode resumes at the compressed multimodal
position rather than the physical token count.

An independent Transformers 32×64/two-token oracle passes at relative L2
`1.48e-6`, cosine `1.0`, and max error `3.87e-7`. Public rectangular-image,
two-image, and three-frame-video requests pass; a 16-token black-vs-pattern
comparison produces different IDs and the patterned row begins “Based on the
image provided”. Text-only IDs remain exact before/after and teardown is zero.
Bounded `image/png` data URLs now work on non-streaming
`POST /v1/chat/completions` when `--vision-model` is configured; remote URL
fetch, HTTP multimodal SSE, and multimodal context above 1K remain explicitly
unsupported. The local 334-tensor/907,523,008-byte mmproj SHA-256 is
`375f156fdc1232f994c42f43813861fac4fdc791f0440a36c85e87b6907a7eee`.
Evidence:
[`2026-08-28-gfx1151-qwen38-flash-next-general-multimodal.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-general-multimodal.json) and
[`2026-08-28-gfx1151-qwen38-flash-next-basic-vision.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-basic-vision.json).

### F10 — Public serving and closure

Current <=1K native serving status (2026-08-28): Qwen4Exp now exposes a c2
resident runner pool over one shared weight/PLE layout. Residual, GDN, PLE,
QSA K/V/index, scratch, and cursors are request-owned; scheduler compaction
preserves runner identity. Serial c1 and c2 IDs match exactly on both varied
8-token prompts. Native stream-many emits three correctly owned chunks per
row. Two simultaneous blocking chats and two simultaneous SSE chats return
HTTP 200; both streams end in `[DONE]`. Admission rollback and cancellation
before mutation pass. c2 adds 550,283,960 tracked bytes over c1 and shutdown
returns to zero. Current model transitions remain per-request serial within
one scheduler tick, so this is a functionality/isolation result, not a c-aware
kernel throughput claim. Evidence:
[`2026-08-28-gfx1151-qwen38-flash-next-native-c2-serving.json`](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-native-c2-serving.json).

Qualify:

- `LLM.generate()` and `hipengine serve`;
- reasoning effort (`xhigh`, `medium`, `low`, disabled), preserve-thinking and
  official sampling defaults;
- tool-call/chat-template behavior;
- blocking and SSE token ownership;
- c1 then c>N admission/cancellation/lifecycle;
- declared 4K/16K/64K/262K capacities; and
- AR plus optional MTP under explicit profile/quant/KV identities.

Closure updates the campaign status, immutable worklogs, compact result
artifacts, benchmark README/changelog, kernel catalog, refactor ledger, PLAN,
and the public README export only where user-facing results are retained.

---

## 8. Required test matrix

| Surface | Minimum binding evidence |
| --- | --- |
| Quant | final GGUF hash/inventory; per-role qtypes; pack/dequant rows; llama metadata/text smoke |
| Plugin/loader | exact resolution; duplicate/missing paths; real header map; shape/metadata negatives |
| GR | grouped norm/read/write analytic fixtures; four-branch state/lifecycle; strict fallback |
| PLE | uint64 hash; EOS/reset; 16-row gather; Q4_0 dequant; Conv/history; mmap lifecycle |
| GDN | sigmoid output gate; Conv and FP32 recurrence; prefill/decode chunk parity |
| MoE | 512/top-10 route; shared gate; selected expert outputs; no row/expert overflow |
| QSA | dense-equality boundary; exact block/tail indices; sparse K/V; non-contiguous spans; long context |
| Runtime | prompt chunking; c1/cN state isolation; cancellation; prefix; graph/eager; teardown |
| Model quality | same-Q4 full logits; category/heldout mean/p95/p99/max KL and top-1; deterministic repeats |
| MTP | same-session AR denominator; exact output; CPU/GPU accept; rollback/repair; full suite economics |
| Vision | processor/placeholder/MRoPE/encoder fixtures; text-only isolation |
| Perf | exact command, host, hardware, source/model/binary hashes, shape, route, correctness and memory |

No “model works” claim is made from registry resolution, one layer, finite
logits, or one generated prompt alone.

---

## 9. Initial file ownership plan

The names may be refined at F1, but responsibility does not move into generic
engine branches.

| Path | Responsibility |
| --- | --- |
| `hipengine/models/qwen4_exp.py` | model plugin, layer sequence, architecture capabilities |
| `hipengine/loading/qwen4_exp_gguf.py` | GGUF config/tensor map/validation |
| `hipengine/loading/qwen4_exp_materialize.py` | one-layout residency and sparse PLE owner |
| `hipengine/kernels/cpu_reference/qwen4_exp.py` | GR, PLE, QSA and reduced-layer oracles |
| `hipengine/runtime/qwen4_exp_runner.py` | physical resident target execution |
| `hipengine/generation/qwen4_exp_gguf.py` | registered public generator factory |
| `hipengine/kvcache/qsa.py` | QSA index-cache/spans lifecycle if it is generic enough; otherwise model runtime owns it |
| `hipengine/kernels/hip_gfx1100/fused/qwen4_exp_gr.{hip,py}` | shared gfx11 GR primitives |
| `hipengine/kernels/hip_gfx1100/attention/qwen4_exp_qsa.{hip,py}` | shared gfx11 QSA primitives |
| `hipengine/kernels/hip_gfx1100/embedding/qwen4_exp_ple.{hip,py}` | sparse PLE staging/compute primitives |
| `tests/test_qwen4_exp_*.py` | metadata/oracle/runtime/model/profile tests |
| `scripts/qwen4_exp_*.py` | artifact inspection, comparator, correctness and benchmark drivers |

If QSA proves reusable across future architectures, extract its generic
backend only after the Qwen4Exp strict path works. Do not prematurely widen
core interfaces.

---

## 10. Risks and stop conditions

| Risk | Required response |
| --- | --- |
| Q4_K_M whole process exceeds capacity | prove PLE is sparse and no duplicate payload exists; then compare Q4_K_S. Do not hide OOM with swap thrash and call it fit. |
| Standard Q4 damages PLE quality | compare an imatrix/dynamic Q4 artifact when available; keep artifact identities separate. |
| PLE random mmap reads stall | use bounded read-ahead/pinned double buffer and layer-0 overlap; never materialize the table merely to make a benchmark pass. |
| Existing MoE kernels assume <=256 experts/top-8 | RED capacity tests before real load; extend generic metadata/scratch bounds, not architecture branches. |
| Group-12 GQA unsupported | use strict per-query-head fallback first; add grouped reuse only after correctness. |
| QSA selection diverges | bisect pooled keys, norm, MRoPE, scoring, top-k ties, block expansion and tail separately before touching sparse attention. |
| Dense fallback used above 2,051 | reject the request; this is a correctness blocker, not a slow fallback. |
| GR state collapsed accidentally | stop; add first-boundary branch-state capture and repair ownership before optimization. |
| GPU hang/0% | treat stale JIT cache per `KERNELS.md`; do not alter math blindly. |
| llama PR/reference changes | retain frozen commit as oracle; refresh only in a separate documented unit. |
| Vision delays text path | keep explicit text-only capability; do not block F5/F6, and do not overclaim F9. |
| MTP is slower than AR | keep AR default and record exact economics; no acceptance-only promotion. |

---

## 11. Campaign completion checklist

- [x] F0 exact source is pinned; the operational Unsloth `UD-Q4_K_XL` is
      scanned, all-part hashed, same-artifact llama-smoked, and public-tested.
      Local conventional Q4_K_M remains a non-blocking reproducibility follow-up.
- [x] F1 qwen4exp plugin and real GGUF tensor map pass.
- [x] F2 CPU GR/PLE/QSA/GDN reduced-model oracles pass.
- [x] F3 one-layout residency fits; PLE is sparse mmap-owned; teardown is zero.
- [x] F4 missing native primitives pass RED/GREEN and kernel traces.
- [x] F5 strict text AR works below the QSA budget through public APIs.
- [x] F6 QSA passes current natural 16K plus retained historical natural 64K;
      262K physical ownership passes but is explicitly capacity-only, not an
      inference support claim. 64K current rerun and 128K+/262K inference stay
      behind the declared throughput ladder.
- [x] F7 retained prefill/decode/batching paths are measured and documented.
- [x] F8 official MTP is correct and honestly retained as opt-in/rejected on
      aggregate economics.
- [x] F9 <=1K multimodal image/video/HTTP support passes with explicit bounds
      and text-only isolation.
- [x] F10 <=1K reasoning/tool/chat, blocking/SSE, multimodal HTTP, request-owned
      c2, admission/cancellation/lifecycle, rollups, and public scope close.

The campaign is complete only when the checked items match committed code and
durable evidence. A pending background download, a quant file, a campaign doc,
or a single generated answer is not completion.
