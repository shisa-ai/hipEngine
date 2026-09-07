# Dense UD Q4_K_S / Q4_K_M Support Campaign

Last updated: 2026-09-06.
Status: verified analysis and coder handoff; **no implementation or GPU validation**.
hipEngine source audited: `bf46abefc5ad8fbb00608cd5fb274ca1af21f716`.

**Goal:** support the pinned Qwen3.8-27B Unsloth Dynamic Q4_K_M and Q4_K_S
files as shipped, with compact, operation-complete dense execution on separately
qualified gfx1100 and gfx1151 backends.

**Architecture:** retain per-tensor GGML storage identities, resolve role/layout/
shape capabilities through the existing four-axis registry, and establish raw
compressed strict consumers before optional repacking or quantized-activation
optimizations. Model admission and execution-profile certification are separate.

**Technology:** GGUF metadata, NumPy CPU oracles, Python/ctypes host, raw-pointer
HIP kernels, existing kernel/profile registries. No torch dependency, quantizer
implementation, runtime replacement, or new registry axis.

This corrects conclusions of the initial audit in `8bd27fe` and `bf46abe`;
their immutable worklogs remain historical evidence, not current guidance.
Execution requires a future GPU allocation coordinated with the profiling owner.
Checkboxes below are future work unless explicitly marked complete.

Related: [architecture](PLAN.md), [intake](GGUF.md), [type portfolio](QUANTS.md),
[MoE Q3 work](GGUF-Q3-OPT.md), [kernels](KERNELS.md),
[profiles](EXECUTION-PROFILES.md), [testing](TESTING.md),
[benchmark protocol](BENCHMARK.md), [cleanup ledger](REFACTOR.md).
Existing K_M execution/performance charter:
[QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md](QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md).
Existing multi-engine evidence:
[QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md](QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md).
Reproduction: [executable analysis appendix](UD-QUANTS-REPRO.md) and
[metadata/accounting snapshot](UD-QUANTS-REVIEW.json).

## 1. Decisions And Findings

1. **The 18/41 refusal counts are real.** Published UD K_M requires Q3_K,
   IQ4_NL and IQ3_S; K_S additionally requires IQ3_XXS and IQ2_S. IQ4_XS
   and IQ2_XS currently expand to BF16 on this dense route.
2. **Do not reduce this campaign to a different two-format model.** That is a
   useful optional integration fixture, but it does not support the requested
   published UD files. Deliver K_M first, then the additional K_S surface.
3. **Raw Q5_K/Q6_K dense kernels already exist.** Their expansion in some roles
   is a materializer/consumer-qualification restriction, not a missing decoder
   or universal lack of a raw fallback.
4. **Generic selected IQ/Q3 wrappers are not fixed to expert widths.** Their
   variable, block-aligned K arithmetic is reusable. Selected registration,
   expert metadata and BF16-only output still do not constitute dense support.
5. **IQ repacking is not a prerequisite.** Separate the current file-wide repack
   veto from per-tensor layout selection. Keep IQ raw while independently
   selecting qualified Q4/Q5/Q6/Q8 residents. A T16 IQ layout is an optimization
   candidate, not an already-designed or necessary ABI.
6. **The prior memory table was not resident accounting.** It omitted refused-
   tensor expansion, pack8/layout growth, sidecars and AR/NextN separation.
   Corrected hypothetical all-refusal BF16 AR weight plans are 39.726 GiB
   for K_M and 42.620 GiB for K_S, before runtime allocations.
7. **Stamp hazards extend beyond two switches and beyond K_S.** A histogram
   alone is also insufficient: swapping the same types between sensitive roles
   preserves the histogram. Bind policy to the actual role/shape/type/layout
   manifest and its qualified model/profile identity.
8. **Upstream has useful missing-format implementations.** Use llama.cpp's
   IQ3_S/IQ2_S and standalone IQ4_NL math as independent references. Halo has
   specific IQ3_S staging ideas, with shape and arithmetic caveats. Pwilkin's
   pinned fork has no quant-kernel delta against its identified upstream base.

### 1.1 Relationship To The Existing K_M Campaign

The August 28 dedicated K_M campaign already defines strict-first admission,
`gguf_ud_q4_k_m` as an artifact-qualified preset key, exact unsupported slots,
single resident ownership, public/batch/NextN coverage and binding performance
comparisons. Preserve those decisions. Its "IQ4_XS existing supported family"
row means selected-MoE arithmetic, not an available dense compressed consumer.
This review supplies that missing work, repairs the audit, and adds K_S.

Use **UD-U0..U7** for this document's task IDs and **KM-U0..U8** for the older
document's tasks when reporting progress. The numbering is not interchangeable:

| This handoff | Existing K_M work it supplies or constrains |
| --- | --- |
| UD-U0/U1 | KM-U0 identities and KM-U3 artifact/materialization admission |
| UD-U2 | KM-U1 independent codec contracts |
| UD-U3 | Previously understated dense IQ4_XS and expanded Q5/Q6 dependencies; complete before KM-U3 model bring-up |
| UD-U4 | KM-U2/U3 refused-type and public K_M bring-up |
| UD-U5 | New K_S-only type/embedding extension; does not replace K_M tasks |
| UD-U6 | KM-U4/U5 operation coverage, plus NextN draft/serving scope |
| UD-U7 | KM-U5..U8 profiling, optimization and binding comparisons |

The existing K_M charter requires same-host wins over both hipEngine plain
Q4_K_M and the faster correctness-valid llama.cpp HIP/Vulkan result, for both
prefill and true AR decode at **512/128, 1024/128 and 4096/128**, independently
on each backend. Preserve its common-KV/timing/cache contract, one discarded
warmup, at least five counterbalanced paired samples, median improvement and
either five positive pairs or a positive 95% paired-bootstrap ratio bound.
Existing survey rows with different quant/KV/timing are context, not substitute
binding baselines.

Functional support and performance-charter closure remain distinct. A correct
route can be reported as supported in its qualified scope while those speed
targets remain open. No equivalent numeric speed promise is invented for K_S:
freeze its comparator/evaluator before tuning and report losses honestly.
This review does not loosen the older charter or claim it has passed.

## 2. Audit Of The Previous Claims

| Previous statement | Verdict and correction |
| --- | --- |
| UD files contain mixed tensor formats | Confirmed. This is a recipe over existing GGML types, not a new block encoding. Ordinary K presets also mix types; neither filename nor `general.file_type` is a tensor inventory. |
| Dense models cannot run UD | Overbroad. These two dense-27B artifacts fail; `GGUF.md` already records dense 0.8B UD-Q4_K_XL fallback support. Admission is artifact/role dependent. |
| 851 expected tensors validate; the family table counts 65 layers | The validation count is correct but AR-scoped. Files have 866 tensors; AR uses 851 and excludes 15 block-64 tensors. The 65-layer family table includes MTP: 64 AR layers are 16 full-attention and 48 recurrent. |
| Stored sizes 15.92/15.32/14.29 GiB | Confirmed as sums of tensor payload sizes including MTP. Not file size, AR resident size, or allocator peak. |
| 277/270 expansions and 18/41 refusals | Confirmed for the script's all-disk-tensor loop. Real AR-map expansions are 271/264; refusals remain 18/41. |
| Every table uses exactly the loader plan | False. The script invents slots, includes ignored MTP tensors, omits model-wide F32 contraction, Q5 raw sidecars and the gfx1151 Q6 exclusion, and does not resolve actual consumers. |
| `token_embd` is mapped correctly | False. Audit uses `root.token_embd`; loader uses `root.token_embedding`. Its test pins the incorrect spelling. Plain Q4 embedding is misreported as pack8 instead of raw. |
| Partial-header parsing skips only final range validation | False. `read_header` also omits version, duplicate metadata/tensor and alignment validation; it catches failure reading a tensor name and returns a partial table. `mapping_result` hardcodes version 3. |
| Q4 remains raw when repack is disabled | False for ordinary rank-2 projections: `_spec_for_tensor` uses `q4_k_pack8`. Raw embedding and rank-3 cases differ. Pack8 is 0.75 byte/weight, not Q4_K's 0.5625. |
| Q5/Q6 have no raw dense fallback | False. `gguf_k_gemv.{py,hip}` contains and registers raw dense Q5/Q6 decode and prefill. Qualification and planner rules need reconciliation. |
| IQ4/IQ3/Q3 device code only handles fixed expert widths | False for generic GEMV wrappers. Grouped IQ prefill does have K<=3072, so it cannot simply be relabeled for dense K5120/6144/17408. |
| IQ4_NL is nearly free | Unproven effort estimate. CPU decoder exists and codebook is shared, but block stride, scales, kernel, dense ABI and gates remain separate work. |
| Only IQ3_S and IQ2_S need decoders | True for missing CPU codecs among these seven types; incomplete for GPU work. IQ4_NL needs standalone device decoding/consumption too. |
| Repack alone recovers <1 GiB; Phase 1 leaves 80 expansions | Not a valid current plan. Corrected AR counterfactuals differ by backend and include IQ2_XS. The old 244-to-80 count also cannot represent removal of all 172 IQ4_XS tensors. |
| W7900 has 32 GiB, so neither host can fit 35.8 GiB | Wrong hardware premise and unsupported capacity conclusion. Canonical W7900 is 48 GB-class. Available allocations, other users, UMA limits, context and scratch determine fit; no device capacity was queried here. |
| A plain Q4_K_S always has four formats | Not a format guarantee. Quantizer revision, model, imatrix and overrides affect the mix. A historical benchmark needs its exact artifact/provenance. |
| IQ2_S is both required in Phase 2 and out of scope | Contradictory. It is required for published K_S. IQ2_XS also occurs in K_S; either supply compact execution or explicitly gate a bounded fallback. |
| Two-format model eliminates all other work | Only for that different artifact, after every role/dtype works. Its small alpha/beta matrices, Q8 embedding/head and MTP remain integration work. |
| Pwilkin has no new IQ decoder | Confirmed for the pinned fork versus its pinned upstream base, now by diff rather than commit-message search. Not a claim about all branches or absence of reusable runtime ideas. |

The five original audit tests pass but do not cover these discrepancies.
No production code or test was repaired during this analysis.

## 3. Pinned Local Inventories

All files are under `/models/gguf/`, read on September 6, 2026.
The JSON records header hashes, exact bytes, endpoints, format counts, current
AR rejection names/shapes and plan routes. Header hashes identify the inspected
metadata, **not the complete model bytes**.

| File suffix | Header claim | File bytes at final scan | All payload GiB | AR source GiB | Refused AR |
| --- | --- | ---: | ---: | ---: | ---: |
| `Q4_K_M.gguf` | MOSTLY_Q4_K_M | 17,106,775,008 | 15.921684 | 15.652040 | 0 |
| `UD-Q4_K_M.gguf` | MOSTLY_Q4_K_M | 16,464,440,224 | 15.323463 | 14.996561 | 18 |
| `UD-Q4_K_S.gguf` | MOSTLY_Q4_K_S | 15,358,213,024 | 14.293209 | 13.966307 | 41 |

K_S was growing during this review. At final scan its size equaled the required
final tensor endpoint, and production `scan_gguf` accepted all three files.
The earlier "9.64 GiB partial download" is historical, not the handoff state.
No downloader was started or changed here. Payload checksum verification is
still required before real fixtures or model-quality claims.

The existing K_M campaign records full SHA256
`322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`
for this 16,464,440,224-byte artifact. That is prior pinned evidence, not a new
payload rehash here. U0 should verify it against the execution-host copy and
reuse its provenance rather than invent a fresh identity. K_S still needs its
equivalent full-artifact pin.

### 3.1 All-file Format Histograms

Counts include block 64; each column sums to 866.

| GGML type | Plain K_M | UD K_M | UD K_S |
| --- | ---: | ---: | ---: |
| F32 | 456 | 360 | 360 |
| Q4_K | 294 | 104 | 95 |
| Q5_K | 48 | 131 | 80 |
| Q6_K | 67 | 30 | 18 |
| Q8_0 | 1 | 106 | 99 |
| IQ4_XS | 0 | 117 | 172 |
| Q3_K | 0 | 7 | 13 |
| IQ4_NL | 0 | 7 | 7 |
| IQ3_S | 0 | 4 | 15 |
| IQ3_XXS | 0 | 0 | 5 |
| IQ2_XS | 0 | 0 | 1 |
| IQ2_S | 0 | 0 | 1 |

All 18/41 refused tensors are AR tensors. K_S examples:
`token_embd.weight` Q3_K, `blk.0.ffn_up.weight` IQ2_S,
`blk.1.ffn_gate.weight` IQ3_S, `blk.3.attn_q.weight` IQ4_NL,
`blk.10.ffn_gate.weight` IQ3_XXS. JSON has the complete list.

### 3.2 Role And Shape Coverage

Shapes are logical `(N,K)` output-by-input, not serialized GGML order.
Avoid suffix-based grouping that collides across output/attention/gate families.

| AR role | Shape | Count | Concern |
| --- | --- | ---: | --- |
| Full attention Q | 12288x5120 | 16 | Mixed IQ4_XS/Q4/IQ4_NL/Q6 requires complete dense output contracts |
| Full attention K, V | 1024x5120 each | 16 each | Narrow-N Q5/Q6 differs from wide FFN |
| Full attention output | 5120x6144 | 16 | Distinct from LM head and recurrent output |
| Recurrent QKV | 10240x5120 | 48 | gfx1151 excludes `attn_qkv` from planar-Q6 default |
| Recurrent gate | 6144x5120 | 48 | Pair fusion cannot assume QKV/gate identical types |
| Recurrent output | 5120x6144 | 48 | Q5 sidecars and F16/BF16 prefill policy |
| Recurrent alpha, beta | 48x5120 each | 48 each | Plain F32 becomes UD Q8_0; N=48 is not a 128-row MMQ tile |
| FFN gate, up | 17408x5120 each | 64 each | Mixed pairs need independent projections plus SiLU fallback |
| FFN down | 5120x17408 | 64 | Long K, split/reduction and accumulation bounds |
| Token embedding | 248320x5120 | 1 | K_S Q3_K lookup, not selected GEMV; LM head is equally large |
| LM head | 248320x5120 | 1 | Q6_K; F32 full logits and top-1/sampling |
| Trailing NextN | Separate block 64 | 15 tensors | Separate loader, aliases and precision policy |

Old 17-attention/65-FFN counts included MTP; they are not AR launch counts.
Both UD files move `nextn.eh_proj` from plain-file Q8_0 to Q6_K,
shape 5120x10240. Inspect its plan separately before claiming MTP support.

The local 35B-A3B UD-Q4_K_M header was independently re-read: 753 tensors,
F32 368, Q8_0 259, Q4_K 82, Q5_K 38, Q6_K 4, BF16 2. This confirms the
previous histogram but not arbitrary UD MoE support. Existing IQ-selected MoE
plugins have distinct artifact, rank-3 routing and consumer contracts.

## 4. Corrected Weight Accounting

These are **CPU-calculated weight plans**, not measured GPU residency.
They use actual AR slots, production allocation formulas, default sidecars,
and source capability values without importing backend packages.

Count unique `(source, layout)` allocations, including raw/tiles/sidecars, with
`planned_qwen35_gguf_weight_allocation_nbytes`. Refusals have two explicit
hypothetical treatments:

```text
native-refusal lower bound = accepted planned bytes + refused source bytes
BF16-refusal scenario      = accepted planned bytes + 2 * refused elements
```

The first assumes missing compact kernels exist; the second assumes missing
CPU decoders and fallback admission exist. Neither is a working load. Neither
includes scratch, state, KV, graphs, allocator overhead or load peak.

| File / backend | Current expansions | Refusals | Native-refusal lower bound GiB | BF16-refusal scenario GiB |
| --- | ---: | ---: | ---: | ---: |
| Plain K_M / gfx1100 | 0 | 0 | 16.912416 | 16.912416 |
| Plain K_M / gfx1151 | 0 | 0 | 15.945620 | 15.945620 |
| UD K_M / either | 271 | 18 | 37.506021 | 39.725985 |
| UD K_S / either | 264 | 41 | 35.882925 | 42.619814 |

Plain-file lane differences include gfx1100 Q5 raw-MMQ sidecars and gfx1151
standard-versus-planar Q6 selection. For UD, the raw-IQ repack veto makes
accepted Q4 projections grow into pack8 and Q5/some Q6 expand to BF16.

### 4.1 Staged Counterfactuals

`Repack` lifts only the global veto for supported formats, not IQ repacking.
`+IQ4` additionally replaces every AR IQ4_XS expansion with a hypothetical
source-sized compressed resident. Other policies stay at pinned defaults,
including stamp choices; these are opportunity estimates, not certified plans.

| File / backend | Stage | BF16 tensors left | Native-refusal lower bound GiB | BF16-refusal scenario GiB |
| --- | --- | ---: | ---: | ---: |
| UD K_M / gfx1100 | Repack | 234 | 35.834040 | 38.054003 |
| UD K_M / gfx1100 | Repack + IQ4 | 117 | 23.577723 | 25.797686 |
| UD K_M / gfx1151 | Repack | 194 | 32.003168 | 34.223131 |
| UD K_M / gfx1151 | Repack + IQ4 | 77 | 19.746851 | 21.966814 |
| UD K_S / gfx1100 | Repack | 239 | 34.719099 | 41.455988 |
| UD K_S / gfx1100 | Repack + IQ4 | 67 | 18.511213 | 25.248102 |
| UD K_S / gfx1151 | Repack | 216 | 32.730116 | 39.467005 |
| UD K_S / gfx1151 | Repack + IQ4 | 44 | 16.522230 | 23.259119 |

Compact IQ4_XS removes about 16.208 GiB from the K_S counterfactual and
12.256 GiB from K_M. It is still the largest identified memory opportunity.
But "repack <1 GiB" is not portable across current backend policies. Residual
K_S BF16 includes Q5/Q6/IQ2_XS. Recover known-format roles before complex IQ
repacking. Admission/type coverage and memory optimization are separate gates.

The original 35.76/37.20 figures left refused tensors compressed while
describing a widened BF16 fallback. K_S refused source bytes are 2,009,518,080;
BF16 would require 9,243,197,440, a missing 6.737 GiB increment.
K_M's missing increment is 2.220 GiB. The old 18.65 GiB Phase-1 result is not a
validated allocation budget.

### 4.2 Capacity And Final Targets

The canonical W7900 is 48 GB-class, not 32 GiB. Strix usable space depends on
the physical machine and OS/runtime configuration. No hardware probe was run;
"39 GiB usable" was not independently verified. These sums alone cannot prove
either machine will or will not fit a requested context.

Compact AR source floors are 14.997 GiB for K_M and 13.966 GiB for K_S.
They are targets before layout overhead, not promises. Reconcile:

```text
steady device = unique weights + justified sidecars + persistent scratch
             + KV payload/metadata/mirrors + Conv/GDN state + graph pools
peak device  = steady device + temporary upload/repack/coexistence high-water
peak host    = mapped/resident source + bounded conversion/transfer staging
```

Every term needs an owner and measured or formula-derived bytes. On UMA, avoid
double-counting shared physical allocations. Do not stage a whole-model FP32/
BF16 copy. A one-tensor IQ2_XS BF16 interim fallback can be explicitly budgeted,
but is not fully compressed K_S completion and needs a removal trigger.

## 5. Existing Code And Actual Gaps

CPU functions are in `hipengine/quant/gguf.py`; device files below are in
`hipengine/kernels/hip_gfx1100/quant/`, with gfx1151 peer wrappers/registration.
Source sharing does not transfer hardware qualification.

| Format | CPU oracle | Existing device math | Required work |
| --- | --- | --- | --- |
| IQ4_XS | `_dequant_iq4_xs_blocks` | `gguf_iq_gemv.hip:iq4_xs_subblock_dot`; selected, grouped, source-MMQ | Raw dense BF16/F32 outputs, rows/prefill, role admission |
| Q3_K | `_dequant_q3_k_blocks` | `gguf_q3_k_gemv.{hip,py}`, selected single/dual-SiLU | Dense linear plus K_S embedding lookup |
| IQ3_XXS | `_dequant_iq3_xxs_blocks` | `gguf_iq_gemv.hip:iq3_xxs_group_dot`; selected/source-MMQ | Dense consumers and K_S admission |
| IQ2_XS | `_dequant_iq2_xs_blocks` | `gguf_iq_gemv.hip`, `gguf_iq2_xs_mmq_prefill.hip` | Dense consumers or explicit interim fallback |
| IQ4_NL | `_dequant_iq4_nl_blocks` | Shared codebook inside IQ4_XS, not standalone IQ4_NL | Separate block decoder, dense execution, independent tests |
| IQ3_S | Layout only | No matching family found | CPU oracle, device decoder and consumers |
| IQ2_S | Layout only | No matching family found | CPU oracle, device decoder and consumers |
| Q5_K / Q6_K | Existing | `gguf_k_gemv.{py,hip}` raw `linear`; T16/MMQ too | Qualify existing raw/T16 routes for remaining roles |

Generic selected Q3/IQ GEMV validates K>0 divisible by 256, N/E>0 and compatible
selected/activation rows. It is not inherently limited to K1024/K3072. However:

- Registrations are `moe_linear`, not dense `linear`. APIs require expert IDs
  and metadata; selected model consumers require rank-3 source storage.
- BF16-in/BF16-out selected GEMV does not supply F32 logits, F16 input, arbitrary
  output strides, embedding gather or every prefill shape.
- `gguf_iq_selected_prefill.py:_validate_common` rejects K>3072.
- `gguf_iq_source_mmq_prefill.py` allows larger block-aligned K but requires
  N/padded rows divisible by 128, selected metadata and Q8_1 activations.
  N=48 alpha/beta cannot be passed unmodified.
- `runtime/gguf_linear.py` resolves dtype/row variants and `prefill_*` for
  multirow execution. Mixed gate/up must keep a strict unfused chain.
- `runtime/gguf_embedding.py:_RAW_EMBEDDING_QUANTS` admits Q4/Q5/Q6/Q8 only.
  Q3 embedding lookup is a separate mandatory K_S primitive.

`gguf_iq_source_mmq_prefill.hip` already has `iq4_xs_expand_group32`,
`iq3_xxs_expand_group32` and
`gguf_iq_selected_mmq_i128_j128_k256_q8_1_ds4_kernel`, with lineage
`llama.cpp@c0bc8591e8815c63cb01dd3f051a8b0df02501c9`.
Integer-tile expansion of raw IQ is not a new discovery. IQ2_XS MMQ and pipelined
K-quant MMQ also exist; reuse them where applicable, rather than duplicate ports.

## 6. Upstream Review

Performance figures here are **author-reported, not reproduced**. Source
inspection establishes code, not actual GPU selection, speed or qualification.

### 6.1 Pins

| Reference | Reviewed identity |
| --- | --- |
| Local llama.cpp HIP/Vulkan | `/home/lhl/llama.cpp/llama.cpp-hip` and `llama.cpp-vulkan`, `4d9176092d00586775af140581bb0b558ddc4389` |
| Halo local master | `/home/lhl/halo-box-strix-llama`, `b212548e0ddbf0a14e5a1d81b6ffcf8e4d098faf` |
| Halo cached optimization branch | `origin/import/fork-master-optimizations`, `7c877db647c2148b0ec9213dd0d46d80a3de9412` |
| Halo remote master verified September 6 | `c7af5c6c29902eb1f7b3bd7952607e2349e1c668`; fresh read-only reference clone in `/tmp/ud-review-halo-current` |
| Pwilkin engine | `d3b5cc43d1fcfce891f2de94d5274ee40eceb21c`, `strix-halo`; fetched into `/tmp` only |
| Pwilkin upstream base | `427291b5b34cd914a31b3fd3b61a68f6184f4b9f` |
| Pwilkin installer/site | `4d0bf821cab29734dacce5321fcd73add72908c0` |
| Installer ROCm pin | `78d1160060bb6ada29b3b21e20c998a48161b257` |
| ilintar model-card revision | `96c04f96a641f25e56deb3cadefe5399e6b7960b` |

Source locations for immutable lookup:

```text
https://github.com/ggml-org/llama.cpp
https://github.com/halo-box/strix-llama.cpp
https://github.com/pwilkin/llama.cpp
https://github.com/pwilkin/strix-halo
https://github.com/pwilkin/rocm-systems
https://huggingface.co/ilintar/qwen3.8-27b-gguf-strix-halo
```

Pins are inspected snapshots, not an assertion that local peers are current.
No peer checkout, fetch, build or runtime configuration was changed.
The inspected llama.cpp/Halo trees carry the ggml MIT notice; preserve donor
copyright/permission notices and source commit attribution when porting.
Model/fixture licensing is separate from kernel-source provenance.

### 6.2 llama.cpp Codec And Execution Map

Under upstream `ggml/src/`, structs/tables are in `ggml-common.h`; CPU truth is
`ggml-quants.c:dequantize_row_<suffix>`. HIP compiles CUDA-directory sources
through `ggml-hip/CMakeLists.txt`; that is the correct HIP reference.

| Format / exact suffix | Bytes / weights | CPU line | `dequantize.cuh` line | `vecdotq.cuh` line | `mmq-load-tiles.cuh` line |
| --- | --- | ---: | ---: | ---: | ---: |
| IQ4_XS / `iq4_xs` | 136 / 256 | 2743 | 424 | 1348 | 1428 |
| IQ4_NL / `iq4_nl` | 18 / 32 | 2725 | 408 | 1324 | 1495 |
| Q3_K / `q3_K` | 110 / 256 | 1305 | 147 | 891 | 598 |
| IQ3_XXS / `iq3_xxs` | 98 / 256 | 2575 | 326 | 1163 | 1295 |
| IQ3_S / `iq3_s` | 110 / 256 | 2607 | 347 | 1202 | 1359 |
| IQ2_S / `iq2_s` | 82 / 256 | 2543 | 310 | 1115 | 1227 |
| IQ2_XS / `iq2_xs` | 74 / 256 | 2516 | 293 | 1074 | 1162 |

The last three files are under `ggml-cuda/`; symbols are `dequantize_<suffix>`,
`vec_dot_<suffix>_q8_1`, `ggml_cuda_mmq_load_tiles_<suffix>`.
Lines refer to `4d9176092`; use symbols and commit pins as durable references.

- MMVQ: `mmvq.cu:get_vec_dot_q_cuda`, `mul_mat_vec_q`,
  `ggml_cuda_mul_mat_vec_q`; uses `quantize_row_q8_1_cuda`.
- MMQ: `mmq.cu:ggml_cuda_mul_mat_q` and `quantize_mmq_q8_1_cuda`.
  K-major `block_q8_1_mmq` is not MMVQ's activation ABI. Weight expansion is
  tile-local, not necessarily a permanent dense copy.
- Dequant/lookup: `convert.cu`, `getrows.cu`, `dequantize.cuh`.
- Actual dispatch: `ggml-cuda.cu:ggml_cuda_mul_mat` checks dtype/stride/padding/
  backend policy before MMVQ/MMQ/BLAS. Type presence does not establish dispatch.
- Vulkan: per-format `mul_mat_vec_*` shaders, `dequant_funcs.glsl` and
  cooperative-matrix code are alternative schedules, not identical arithmetic
  to HIP's quantized-activation path.

### 6.3 Halo-box Candidates And Corrections

The local master snapshot has no HIP/CUDA kernel delta against shared upstream
anchor `9723942adc518b43c4b95dc4dce6906903eb5e09`, but it is stale.
Remote master `c7af5c6c` merged the optimization branch as PR 18 on
**September 5, 2026**. A fresh `/tmp` clone confirms the HIP work is now on
remote master. The cached branch is a provenance aid, not its current
availability boundary. A remote branch query no longer returned that import
branch; use commit IDs, not its cached ref, for reproducibility.

| Candidate / commit | Contribution | Transfer limit |
| --- | --- | --- |
| IQ3_S Vulkan spill fix `c4c03d97b0db3fd6ca110979d7ce5c3d34d8662b` | `mul_mat_vec_iq3_s.comp:calc_superblock`, 16 versus 8 invocations at NUM_COLS>4 | Already upstream `ba8818cbf3ad2f27f6b50e85b959ada4734f34c3`; shader hashes match. Accumulation grouping changes |
| IQ3_S grid staging `6130b7262ae97d353556903e0175b8993db77bef` | `mmvq.cu:vec_dot_iq3_s_q8_1_grid`, `mul_mat_vec_iq3_s_grid_rdna3_5`, row/LDS alternatives; 2 KiB grid in LDS | Reviewed master dispatch requires selected IDs, K2560, one sample/column, RDNA3.5; not dense 27B |
| Four-column IQ3_S reuse `5b6308d71a5dff94995af67a2598f230ee8178fe` | IQ3_S arm of `mul_mat_vec_q4_columns_rdna3_5` reuses decoded groups | Already dense-dispatched on RDNA3.5 for exactly four columns, no IDs or gate/bias/scale fusion; no K2560 restriction. Requalify Q8_1 arithmetic and registers |
| MMQ prefetch `90ad6cd26753b1eab62ff3ee39e17bfdcca3b6b5` | `ggml_cuda_mmq_use_prefetch`, register tile load/store | IQ2_S/IQ3_XXS J128 whitelist, not IQ4_XS; in-tree pipelining exists |
| Fused activation quant `ab55b8fdc393d7afa83a47354bf2a24939286946`, `37f02eb14ec5d58461596fb9f0ce6314cf6fb2d4` | `mul_mat_vec_q_fq` avoids separate pack launch | Q8_0/Q6_K eligible with F32 input/output, one column, alignment/stride/fusion guards and <=16 KiB quantized activation buffer; IQ compile-time default remains disabled |
| IQ4_NL weighted selected down `6130b7262` | `mul_mat_id_iq4_nl_weighted_rdna3_5` | K640/N2560/E512/top10; not dense 27B |

Read corrective commits before a donor port:

- `551ce30fe5c371258333d5681be76b8f3f440d65` restores one-wave batch-one
  reduction after author-observed changed logits/decode divergence.
- `930a8bdad3d6a1ed7011df038962a188e7432606` removes disabled experiments
  and narrows gates after the initial large patch.
- `732484c20c8c7a61885c2e48ea28f90bdd7bd1e1` concerns Vulkan Q8_1
  caching versus column chunking. The `would_quantize_y` guard protects
  extended-width admission, but the inner small-width wrapper also chunks
  and passes `allow_quantize_y=false`. This does not establish universally
  arithmetic-preserving chunking; inspect both layers.
- `a4d8e83e37761b9f18a826a45ecb6b39c2d436f3` driver-gates Vulkan LDS
  padding. Do not copy RADV version gates or pad=2 into HIP; alignment/lowering
  needs independent proof.
- `312ea53c6c637cff78df25bbd5e6bbb26b3c76be`, newer than the cached
  optimization tip, guards final-tile prefetched register assignments and
  handles all-zero activation blocks in `quantize_mmq_q8_1_swiglu`
  (`amax==0` gives zero scale/inverse scale). Include zero blocks and final
  incomplete tiles in any fused-pack port's RED tests.
- `cde4bf7dd3f8bcd16dae8526c5406c645e3a4d6d` and
  `4021b991c6245044ecda10f1e9759e28308610e0` fix ROCm-7.2 build warnings
  and prevent RDNA3.5 Q8 attention compilation on CDNA. These are compiler/
  architecture-scope cautions, not reasons to import unrelated attention code.

Grid staging defaults to four rows only for fused gate; ungated execution uses
the direct row kernel. Full-row LDS staging is off by default and needs the
higher-priority grid path disabled. They are alternatives, not cumulative
optimizations. The four-column dense donor is a separate, more directly
applicable reuse candidate.

Useful missing-format ideas are IQ3_S grid staging and four-column reuse, not
an unexplored IQ4_XS decoder. Source-only findings do not authorize speed or
hipEngine dispatch constants.

### 6.4 Pwilkin Verification

The site has installer/site/data assets, not quant kernels. A direct engine diff
against `427291b5` has 19 changed files, covering scheduler/allocator/sanitizer/
graph support and top-k. Quant definitions, CPU decoders, device dequant,
MMVQ/MMQ and HIP directory have no differences:

```bash
git -C /tmp/ud-review-pwilkin-llama diff --exit-code \
  427291b5b34cd914a31b3fd3b61a68f6184f4b9f \
  d3b5cc43d1fcfce891f2de94d5274ee40eceb21c -- \
  ggml/src/ggml-quants.c ggml/src/ggml-common.h \
  ggml/src/ggml-cuda/dequantize.cuh ggml/src/ggml-cuda/convert.cu \
  ggml/src/ggml-cuda/vecdotq.cuh ggml/src/ggml-cuda/mmq.cuh \
  ggml/src/ggml-cuda/mmvq.cu ggml/src/ggml-cuda/mmvq.cuh \
  ggml/src/ggml-cuda/getrows.cu ggml/src/ggml-hip
```

Exit 0. Full diff file-list review also bounds the claim, rather than relying
only on handpicked paths. Top-k and graph UID/source-pointer validation are
real reusable ideas, but not missing UD decoders or drop-in runtime replacements.
"Nothing portable" was too strong.

The installer pins custom ROCr/HIP and checks library resolution. We inspected
the installer and ROCm pin metadata, not the complete retained-PM4 implementation.
Do not install it into the profiling environment or claim a reviewed runtime port.

The pinned ilintar card and site publish:

| Author-reported evidence | Confirmed publication content, not local reproduction |
| --- | --- |
| Main recipe | 496 IQ4_XS + 10 Q8_0 + 360 F32; 16,110,851,680-byte file; output, embedding and eight uncalibrated MTP matrices stay Q8 |
| Calibration | Explicit tensor map avoids normal IQ4_XS preset Q5 substitutions |
| Small local PPL | IQ4_XS 15.3977 +/- 0.73357; source BF16 15.1721 +/- 0.72292; 16 held-out 512-token chunks |
| Draft comparison | +1.79% to +5.67% over six prose/reasoning/JSON width-3/6 cells; author reports matching target hashes |
| No speculation | tg128 14.0976 tok/s, retained PM4; about +3.2% over ordinary submission |
| Long-prompt site row | 31,497 prompt/256 generated; 256.838 prompt and 26.256 decode tok/s |

The main-model inventory was also independently checked by a bounded HTTP range
read at the pinned model revision: 866/866 descriptors, 496/10/360 histogram,
the exact ten Q8 role names, 851 valid AR-map tensors and expected file extent
16,110,851,680 bytes. Header length is 10,995,296 bytes, SHA256
`c85a1699ddeeb2e3cdd9302eca14bdc3db8fea3bd7cb197e310404cf64766b22`.
Only this inventory, not tensor contents, calibration or quality, is independently
verified. Reproduce the bounded fetch without downloading the full model:

```bash
curl -fsSL --range 0-12000000 --max-filesize 16000000 \
  https://huggingface.co/ilintar/qwen3.8-27b-gguf-strix-halo/resolve/96c04f96a641f25e56deb3cadefe5399e6b7960b/Qwen3.8-27B-IQ4_XS-ALL-IMATRIX-Q8-OUT-MTP.gguf \
  -o /tmp/ud-pwilkin-model-header.gguf
.venv/bin/python scripts/gguf_quant_route_audit.py \
  /tmp/ud-pwilkin-model-header.gguf --json /tmp/ud-pwilkin-header-audit.json
```

Use only the descriptor/mapping fields of that original audit, subject to U0's
documented parser limits; its resident-route accounting is not authoritative.

The draft is **not pure IQ4_XS**: 44 IQ4_XS, 5 Q5_K, 32 F32. It was
requantized from Q8_0 with transferred/uniform importance vectors. That is not a
high-precision-source recipe to copy without a separate quality investigation.
Equal greedy target hashes do not prove equal draft quality, acceptance or
stochastic correctness. These small prompt sets do not satisfy hipEngine's
full-category/heldout protocol. No external row enters our scoreboard.

## 7. Implementation Gotchas

### 7.1 Storage And Arithmetic

- Reverse GGML dimensions into `(N,K)`; preserve row/block/expert strides,
  little-endian fields and unaligned-safe loads. IQ4_NL has 32-weight blocks,
  not IQ4_XS's 256.
- IQ4_XS combines high/low bits into signed six-bit scales minus 32. IQ4_NL
  has separate per-32 F16 scale; equal nonlinear values do not imply equal ABI.
- IQ3_S and Q3_K both use 110 bytes/256 but unrelated encodings. IQ3_S has
  9-bit grid indices, explicit signs and odd subscales; IQ3_XXS has byte indices
  and packed seven-bit sign selectors.
- IQ2_S uses 10-bit indices/explicit signs; IQ2_XS has 9-bit indices/compressed
  sign selectors. Do not derive either decoder by changing a type ID.
- Derive sizes from struct assertions, not comments; the upstream IQ3_S
  decoder has a historical bpw comment inconsistent with its 110-byte ABI.
- Test all codebook/sign/high-index bits, zero scales, valid negative/extreme
  finite scales, row boundaries, bad K/rank, tail-N and row tails. Fail malformed
  metadata before allocation; avoid unchecked packed reads.
- Separate dequant arithmetic, activation conversion, accumulation, final cast
  and SiLU boundaries. CPU-decode parity does not imply dot parity; BF16 output
  can hide F32 discrepancies.
- Replacing an existing BF16-expanded weight with a raw IQ consumer may remove
  the old dequant-to-BF16 rounding boundary. Define and test that boundary
  explicitly; source-byte preservation alone does not prove parent arithmetic
  parity or authorize relabeling the new path as strict.
- Upstream Q8_1 MMVQ/MMQ changes activations relative to BF16 strict arithmetic.
  It needs a separate production gate, not just storage byte parity.
- Check signedness, integer overflow bounds, FMA, BF16/F16 rounding and scale
  order at K17408. Short expert-width evidence does not qualify long dense K.

### 7.2 Consumer Completeness

- c1 GEMV, independent c2/c4/c8 rows, verifier rows and bulk prefill are different
  shapes. A bounded correct fallback precedes optimization.
- Mixed gate/up and QKV/gate resolve operands independently; keep the strict
  unfused chain when a fused mixed-format kernel is absent.
- F32 LM logits/full-vocabulary teacher output and BF16 intermediate outputs
  are separate contracts, even if greedy top-1 is the first smoke.
- Quantized alpha/beta must feed recurrent gates correctly; N48 cannot use an
  N128-assuming MMQ kernel without padding/ownership proof.
- Concrete native-row hazard: `_run_linear_attention_decode_rows_native` in
  `runtime/qwen35_gguf_runner.py` passes alpha/beta `.allocation("raw")` directly
  to `dense_gemv_out_bf16`. If new UD admission reaches this route, raw Q8 bytes
  would be interpreted as BF16, while sole-T16 storage has no `raw` allocation.
  Route through layout-aware registered consumers or reject the scope before
  execution. A materializer-only change is unsafe.
- NextN shares roots but has its own planner. Preserve aliases and teardown;
  AR-only support must decline MTP until its operation set resolves.
- Concrete NextN blocker: `_EXPECTED_COMMON_QTYPES` currently expects Q8_0
  `eh_proj`. Its Q6_K exception belongs to the separate native-XL manifest.
  Published UD needs its own exact draft-manifest admission; do not bypass
  validation or spoof the native-XL identity.
- Embedding gather needs prompt/decode IDs, repeated IDs and bounds tests.
  Do not expand the entire 248320x5120 embedding as a shortcut.
- If any BF16 embedding fallback is qualified, test its multirow adapter:
  `gguf_embedding.py:_launch_dense_bf16` does not forward `rows` and resolves a
  singleton lookup. Distinct-token rows and sentinel-filled outputs must either
  all be written correctly or fail preflight. Native Q3 lookup does not fix that
  separate fallback issue.
- Chunked prefill/graphs must consume compatible persistent weights. No
  request-time whole-model upload or dequantization.

### 7.3 Policy And Ownership

Audit all K_M/K_S users in backend dictionaries,
`loading/qwen35_gguf_nextn.py`, `loading/gguf_mtp_hot_vocab.py`,
`runtime/qwen35_gguf_runner.py` and profile/automatic-MTP admission.
Scopes include recurrent storage, GDN association, scratch/source-F16,
gate/up, Q6 DP4A and speculative depth, not just two original flags.

The original flags indicate eligibility, not observed execution. The global
repack veto prevents Q4 qmicro selection for current UD, and rejected loads
execute no recurrent kernel. This is an admission hazard, not measured corruption.

Resolve cold-path capabilities from the full
`(role, shape, source type, resident layout, dtype, operation)` manifest, bound
to artifact and profile evidence. Preserve qualified plain-file behavior.
Neither successful loading nor matching histogram permits UD to inherit
plain-file numerical or automatic-MTP authorization.

The global IQ predicate also sets `contract_f32_linear`. Decouple its precision
effect from repack eligibility; test F32 alpha/beta/router plus unrelated IQ
so a layout repair cannot silently change precision or existing MoE Q3 behavior.

Graph/cache keys bind layout/variant identity and stable pointers. Aliases,
temporary metadata, active rows, output ownership and rollback remain exact.
Shared source does not transfer gfx1100/gfx1151 evidence, even with wave32.

## 8. Campaign Punchlist

Each unit: write the stated RED fixture, run its focused node and confirm the
intended failure, implement minimally, run GREEN and applicable gates, record
handoff, commit explicit paths. Future test/kernel names below are proposals.
No speculative optimization precedes its strict fallback.

### U0. Repair Audit And Pin Artifacts

Dependencies: none; CPU-only.
Files: `scripts/gguf_quant_route_audit.py`,
`tests/test_scripts_gguf_quant_route_audit.py`;
loader scanner/model-map modules are references, not automatic edits.

- [ ] Pin upstream model revision, payload checksum, file size and header identity
  for both UD files and the plain control.
- [ ] Use actual AR and separate NextN maps; report disk/AR/ignored counts and
  aliases without double counting.
- [ ] RED: `root.token_embedding`, 64 AR layers, ignored block64, raw Q4
  embedding, and MTP-only tensor exclusion.
- [ ] RED: unsupported version, duplicates, invalid alignment, incomplete table
  versus incomplete data. Partial diagnostic mode cannot report loadable.
- [ ] Report allocation-formula bytes, sidecars, reasons, and both hypothetical
  refusal treatments.
- [ ] RED: gfx1100 Q5 sidecar, gfx1151 Q6 exclusion, missing/nonliteral
  capabilities, env overrides and F32 contraction. Prefer a pure shared policy
  API over an expanding regex mirror; no backend import for metadata audit.
- [ ] Reproduce this snapshot and version the repaired report schema.

Run: `.venv/bin/python -m pytest tests/test_scripts_gguf_quant_route_audit.py -q`.
Exit: real planner accounting without device use, clearly distinct from consumer
qualification. Current five tests are insufficient.

### U1. Role-safe Admission And Policy

Dependencies: U0; CPU tests first.
Files: `hipengine/loading/qwen35_gguf_materialize.py`,
`hipengine/loading/qwen35_gguf_nextn.py`, backend policy/capabilities,
profile binders; proposed `tests/test_gguf_ud_admission.py`.

- [ ] Define cold-path operation coverage records with role/shape/type/layout/
  input-output dtype/rows/fallback, using existing registry conventions.
- [ ] Preserve the existing planned artifact preset `gguf_ud_q4_k_m` for
  model/session admission and concrete per-tensor kernel quant keys; choose an
  equally explicit K_S preset identity. This is not a fifth registry axis.
- [ ] RED: same stamp/different maps; same histogram/types swapped between
  recurrent and FFN roles. No unqualified arithmetic or automatic MTP inheritance.
- [ ] Inventory and constrain all K_M/K_S identity callers while preserving
  qualified plain controls and rollback.
- [ ] Separate per-tensor repack from global F32 contraction; retain existing
  MoE semantics for unchanged manifests.
- [ ] Preflight all requested operations before allocating; report every
  unsupported slot, not just first exception.
- [ ] RED: raw-Q8 and sole-T16 alpha/beta cannot enter a BF16-pointer native-row
  owner; unsupported multirow BF16 embedding cannot silently use a singleton.
- [ ] Keep AR-only and AR+MTP capabilities distinct.

Run after creating the file:
`.venv/bin/python -m pytest tests/test_gguf_ud_admission.py -q`.
Exit: unknown layouts fail closed, no plain-artifact certificate reuse for UD.

### U2. Independent Codec Oracles

Dependencies: U0; CPU-only before GPU leaf tests.
Files: `hipengine/quant/gguf.py`, codebook module only if existing conventions
justify it, proposed `tests/test_gguf_ud_codecs.py`, `tests/fixtures/gguf_ud/`.

- [ ] Add llama.cpp-pinned IQ3_S/IQ2_S byte-to-F32 fixtures before CPU decoders;
  add standalone IQ4_NL independent coverage.
- [ ] Cover all codebook/sign/high bits, struct sizes, scale corners, row
  transitions and dimension order.
- [ ] After payload checksums, extract small real-row fixtures from multiple
  roles/layers, first/last blocks and actual dense widths.
- [ ] Record donor revision, fixture hash, generation command and license.
  New decoder/tables cannot be their own sole oracle.
- [ ] Gate decoded F32 independently; separately specify BF16/FP16 rounding
  and strict accumulation/output contracts.

Run: `.venv/bin/python -m pytest tests/test_gguf_ud_codecs.py -q`.
Exit: independent oracles for all seven type ABIs.

### U3. Recover Q5/Q6 Roles And Add Raw Dense IQ4_XS

Dependencies: U1/U2; GPU access needed for device GREEN.
Files: existing `hipengine/kernels/hip_gfx11*/quant/gguf_k_gemv.py`,
IQ decode/source-MMQ donors, proposed
`hipengine/kernels/hip_gfx1100/quant/gguf_iq_dense.{hip,py}`,
gfx1151 peer registration, materializer/dense capability maps;
proposed `tests/test_gguf_ud_dense.py`.

- [ ] RED role-shaped dispatch/math tests for raw Q5/Q6 in expanded slots;
  use existing registered variants where operation-complete.
- [ ] Add raw IQ4_XS dense c1 strict GEMV for actual K/N, BF16/F32 outputs,
  and supported tail-N; reuse block math without forcing rank-3 runtime paths.
- [ ] Supply correct row-batched and bounded prefill execution. A bring-up
  row loop is explicitly a fallback, not a native-batch speed claim.
- [ ] Keep mixed-pair strict chains, dtypes, graph safety and root behavior
  while selecting residents per tensor.
- [ ] Verify no duplicate dense resident; account planned/measured bytes,
  transient load peak and scratch owners.
- [ ] Register strict fallbacks; qualify gfx1100 and gfx1151 independently.

Run: `.venv/bin/python -m pytest tests/test_gguf_ud_dense.py -q`.
Every GPU test needs a HIP availability guard.
Exit: IQ4_XS/known-format coverage, not yet complete published UD support.

### U4. Published UD K_M

Dependencies: U3 and U2 oracles.
Files: Q3 donor, proposed dense IQ family, materializer/registry consumers;
proposed `tests/test_gguf_ud_km.py`.

- [ ] Add dense Q3_K from existing math and explicit dtype contracts.
- [ ] Add standalone IQ4_NL and IQ3_S strict consumers and independent leaf gates.
- [ ] Cover all 18 refused AR slots and all other incomplete dense operations.
- [ ] Validate c1 AR, teacher logits and compact memory before broader modes.
- [ ] Reject unsupported requested modes in preflight; c1 is not full serving.

Run: `.venv/bin/python -m pytest tests/test_gguf_ud_km.py -q`.
Exit: qualified declared K_M AR scope; U6 controls public/batch/MTP completion.

### U5. Published UD K_S And Embedding

Dependencies: U4 (including its shared Q3_K/IQ4_NL/IQ3_S dense consumers) and
U3/U2. K_S-only codec/embedding leaf work may run in parallel; K_S model
completion cannot precede those shared prerequisites.
Files: IQ2/IQ3 donors, proposed dense family,
`hipengine/kernels/hip_gfx1100/quant/gguf_q3_k_embedding.{hip,py}`,
gfx1151 peer, `runtime/gguf_embedding.py`;
proposed `tests/test_gguf_ud_ks.py`.

- [ ] Q3 embedding lookup RED: repeated/boundary IDs, prompt/decode gather,
  row output ownership and vocabulary bounds.
- [ ] Dense IQ3_XXS and IQ2_XS from existing math.
- [ ] IQ2_S device decoder/strict consumers from U2 oracle.
- [ ] Clear 41 refusals and IQ2_XS expansion; any temporary fallback reports
  bytes/removal trigger, not silent permanent debt.
- [ ] Same c1 AR/logit/memory gates as K_M before broader admission.

Run: `.venv/bin/python -m pytest tests/test_gguf_ud_ks.py -q`.
Exit: both artifacts have complete declared AR coverage; no one-tensor omission.

### U6. Prefill, Batch, NextN And Serving

Dependencies: U4/U5 per artifact.
Files: existing row/bulk-prefill, NextN, resident runner, profile and server tests.

- [ ] Caller ABI coverage for rows 1/2/3/4/5/7/8, verifier rows such as
  6/9/12/16/28/32, prefill tile/chunk boundaries.
- [ ] Short/512/4096 prompts and separately budgeted long-context point such
  as 32768. Short gates do not authorize long trajectories.
- [ ] Q8 alpha/beta recurrent transitions, full attention, mixed FFN pairs,
  F32 logits, sampling, eager/graph repeat parity. Replace or exclude the direct
  BF16 alpha/beta calls in `_run_linear_attention_decode_rows_native`.
- [ ] Block64 Q6 `eh_proj`, attention/FFN, aliases/teardown, exact speculative
  accept/reject commit/rollback; exact UD NextN admission must not use the
  unrelated native-XL manifest exception.
- [ ] c1/c2/c4/c8, ragged/sparse rows, neighbor replacement, permutations,
  delayed arrivals, cancellation/reclaim and width transitions.
- [ ] Artifact-scoped strict manifest first; production requires section 9.
  Unknown identity/profile/shape uses only certified fallback or rejects.
- [ ] Complete category/heldout suite and true no-MTP AR denominator before
  automatic speculative admission.

Exit: explicit AR/MTP/serving/profile/backend/context records, no stamp inheritance.

### U7. Measured Optimization

Dependencies: operation-complete strict baseline for the same artifact.

- [ ] Profile family/launch/transfer/weight bytes before tuning; prebuild cached
  binaries outside rocprof, never wrap the nested suite parent.
- [ ] Compare raw decode versus lossless repack by shape, not assumed T16
  superiority. Include load time, steady bytes and peak coexistence.
- [ ] Reuse source-MMQ for dense prefill/verifier where useful. Q8_1 changes
  need full production gates, not just decoded-weight parity.
- [ ] Try Halo IQ3_S grid/four-column ideas only when its family matters;
  re-establish dense geometry, waves, LDS/register bounds and tails.
- [ ] Prefer compact replacement before permanent sidecars; record exclusions.
- [ ] Stop neutral/negative tuning and re-audit the profile. Promote every
  qualified non-regressive win in scope, including measured subwindows.
- [ ] Update result artifact, benchmark rollup/date/changelog only for measured
  retained results. Update kernel catalog/lineage for actual ports.
- [ ] Complete the existing KM-U8 common-KV same-host hipEngine/plain-Q4 and
  exact-file llama.cpp HIP/Vulkan comparisons at 512/128, 1024/128 and
  4096/128, using its statistical win rule. Do not close that performance
  charter based on decoder correctness or a microbenchmark win.

Exit: no unqualified experimental path without concrete blocker/removal trigger.

## 9. Gates And Completion

Keep three questions distinct:

1. **Codec correctness:** identical compressed bytes decode per independent oracle.
2. **Implementation fidelity:** the same UD artifact matches its strict teacher
   under the declared profile.
3. **Quantization quality:** UD versus BF16/plain K_M is a different representation
   comparison, with separate category/task quality.

Different quantized files need not generate identical IDs. Weight-quality loss
does not excuse implementation errors. Plain-K_M logits are not UD strict bytes.

Normative numerical gates:

- Leaf outer floor: KL<=0.05, top-1>=90% versus independent CPU reference, plus
  declared exact/parent-parity RED contract for strict.
- Production same-artifact teacher: mean KL<=1e-3, p95<=5e-3, p99<=2e-2,
  max<=5e-2; top-1>=99% overall and >=97% in each category/shape/transition.
- Rows above 2e-2 require diagnosis, not automatic promotion. All state/logits
  finite, three same-schedule deterministic repeats, exact control/ownership/
  isolation in every profile.
- Report strict/candidate versus BF16 when available; predeclare paired task
  criteria, never relax thresholds after observing results.
- All `code`, `general_en`, `general_ja`, `mixed_ja_en` categories from
  `benchmarks/prompts/mtpbench-code-general-ja.jsonl`, plus category heldouts.
  No prompt/token/candidate-specific scoring or acceptance shortcuts.

Each new/ported kernel needs a HIP guard, independent oracle, registered strict
fallback, expected-symbol/plausible-duration `rocprofv3 --kernel-trace` proof
on its physical host. Record model hash, quant, profile/manifest, workload,
command, host/hardware/software, correctness, wall time and allocation budget.
No cross-host old-to-new rate or peer-backend qualification.

Use milestone full-test protocol when appropriate after GPU access is granted.
Focused repair after isolated broad-suite failure follows AGENTS.md; avoid
automatic expensive broad reruns.

### Final Checklist

- [x] Reproduce original inventory/refusal observations on CPU.
- [x] Correct AR/NextN, slots, raw kernels, memory and scope claims.
- [x] Inspect upstream/Halo sources and Pwilkin quant diff.
- [x] Preserve runnable accounting appendix and snapshot.
- [ ] U0-U2 audit/identity/CPU oracles complete.
- [ ] K_M/K_S compact role/dtype/operation coverage.
- [ ] No unexplained BF16 expansion, sidecar or duplicate owner.
- [ ] Independent gates for every advertised backend/mode/profile/context.
- [ ] Plain controls, MoE Q3 and unknown-UD fail-closed regressions checked.
- [ ] NextN, batching, speculation and serving have explicit qualified scopes.
- [ ] Retained performance has complete same-host evidence.
- [ ] Catalog, lineage, refactor ledger, immutable handoff and atomic commits current.

## 10. Handoff Boundaries

Start U0, not a new IQ4 repacker. CPU fixtures can proceed while GPU is occupied;
this analysis does not itself authorize ROCm probes, backend registration,
kernel compilation or benchmarks. Coordinate GPU and shared-file ownership.

Historical plain Q4_K_S scoreboard reconstruction is separate: a complete UD
download cannot reproduce a plain-file result. Optional two-format IQ4_XS/Q8
has separate checksum/calibration/quality requirements and cannot close U4/U5.

Not validated here: GPU numerics, actual dispatch, throughput, resident peaks,
full payload checksums, or complete Pwilkin ROCm implementation. These remain
explicit future requirements.
