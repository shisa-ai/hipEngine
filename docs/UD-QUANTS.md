# Dense UD Q4_K_S / Q4_K_M Support Campaign

Last updated: 2026-09-07.
Status: **U0 complete; CPU IQ3_S/IQ2_S decoders and synthetic/real-row oracles
added; raw IQ4_XS/IQ4_NL/IQ3_S/Q3_K/IQ3_XXS/IQ2_S dense GPU leaves and
Q3_K embedding pass bounded gfx1151 checks. Published K_M and K_S run through
public `LLM.generate()` on gfx1151.**
The eight-token eager smoke returns finite logits; public generation returns the
same text without importing torch. Teacher-logit quality and broader
serving are not qualified. Loader preflight has zero K_M or K_S refusals. Q5/Q6 expansion still needs compact resident integration.
**No gfx1100 numerical validation is claimed.** General certificate
cleanup is deferred in favor of end-to-end inference. Generic profile lifecycle
transactions and per-call native authorization scans have been removed.
hipEngine source audited: `bf46abefc5ad8fbb00608cd5fb274ca1af21f716`;
U0 audit/identity repair landed on the `ud-quants` branch (see
[UD-QUANTS-REVIEW-v2.json](UD-QUANTS-REVIEW-v2.json) for its schema-v2
snapshot and [UD-QUANTS-U0-IDENTITY.json](UD-QUANTS-U0-IDENTITY.json) for
artifact identity pins).

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
The active bring-up lane is zbook / gfx1151; gfx1100 needs a separate hardware
allocation. Unchecked items remain open unless explicitly excluded by the scoped
admission decision below. Historical audit and review sections describe their
stated snapshots, not the integrated model's current refusal inventory.

Related: [architecture](PLAN.md), [intake](GGUF.md), [type portfolio](QUANTS.md),
[MoE Q3 work](GGUF-Q3-OPT.md), [kernels](KERNELS.md),
[profiles](EXECUTION-PROFILES.md), [testing](TESTING.md),
[benchmark protocol](BENCHMARK.md), [cleanup ledger](REFACTOR.md).
Existing K_M execution/performance charter:
[QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md](QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md).
Existing multi-engine evidence:
[QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md](QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md).
Reproduction: [executable analysis appendix](UD-QUANTS-REPRO.md),
[metadata/accounting snapshot](UD-QUANTS-REVIEW.json) (historical, schema
implicit v1), [versioned audit snapshot](UD-QUANTS-REVIEW-v2.json) (schema
v2, current: artifact-qualified policy identity, reproducible from the
documented CLI command),
[pre-F5 audit snapshot](UD-QUANTS-REVIEW-v2-pre-f5.json) (byte-frozen
historical v2, stamp-only policy semantics; see
[UD-QUANTS-REPRO.md](UD-QUANTS-REPRO.md) "Snapshot history"), and
[artifact identity pins](UD-QUANTS-U0-IDENTITY.json).

## 1. Decisions And Findings

1. **The baseline had 18/41 refusals.** Published UD K_M requires Q3_K,
   IQ4_NL and IQ3_S; K_S additionally requires IQ3_XXS and IQ2_S. Raw dense
   IQ4_XS/Q3_K/IQ4_NL/IQ3_S/IQ3_XXS/IQ2_S and Q3_K embedding integration
   removes all K_M and K_S base-operation refusals. IQ2_XS still expands to BF16; historical audit tables
   below describe the pre-integration planner.
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

### 1.2 Bring-up Scope And Review Lessons

The immediate milestone is published-file eager c1 generation, followed by
compact residents and numerical validation. Both published files now generate
through the public API on gfx1151; this does not close the broader campaign.
Implementation continues directly, without subagents.

- Retain pre-allocation checks for actual dtype, shape, layout, ordered operands
  and requested operations. These prevented concrete invalid-pointer routes.
- Do not turn a valid review counterexample into an automatic requirement.
  Arbitrary mutation of private pointers, custom-factory transactions, model
  hot replacement and concurrent profile isolation are not supported bring-up
  operations. A new session is the reconfiguration boundary.
- Generic profile lifecycle transactions and repeated native pointer/certificate
  scans were removed. The remaining cold-path certificate machinery is deferred
  cleanup, not a prerequisite for adding codecs or running published models.
- Classify review findings by supported call path, reproducible failure and
  milestone impact before implementing a remedy. A reviewer can identify a
  technically real limitation without establishing that a new framework belongs
  in this task. The coordinator owns that scope decision.
- A green metadata suite proves only the tested structural contract. Keep
  published-file generation, independent codec/math checks, model-logit quality
  and measured memory/performance as separate evidence.

Initial execution uses a fresh process with no named profile or production
overrides. Process-scoped binders are unchanged; generic profile authorization
and lifecycle isolation are not claimed. Numerical profile qualification remains
required before promoting arithmetic-changing optimizations. See the immutable
`ud-direct-cleanup` and `ud-native-cleanup` worklog entries for removal evidence.

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
No downloader was started or changed here.

All three artifacts are now identity-pinned (UD-U0); see
[UD-QUANTS-U0-IDENTITY.json](UD-QUANTS-U0-IDENTITY.json):

| File | Upstream revision | Payload SHA256 | Bytes |
| --- | --- | --- | ---: |
| `Qwen3.8-27B-Q4_K_M.gguf` | `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe` | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` | 17,106,775,008 |
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | `4ca720788d1e01f1bff70c033e0d0028fd02e502` | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` | 16,464,440,224 |
| `Qwen3.8-27B-UD-Q4_K_S.gguf` | `4ca720788d1e01f1bff70c033e0d0028fd02e502` | `75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017` | 15,358,213,024 |

Payload checksums are prior pinned evidence (the K_M hash from the existing
K_M campaign; plain K_M and K_S verified against the official pinned GitLFS
pointers in the parent session); this U0 pass re-fetched all three tiny
pointers at the pinned revisions on 2026-09-07, re-checked local sizes
(exact match for all three), and did not rehash the payloads. Header identity
(SHA256 over the `[0, data_start)` header region, payload bytes excluded; the
plain K_M value reproduces the historical snapshot byte-for-byte) is recorded
per file in the identity pins and in the schema-v2 snapshot's
`header_identity` sections. These pins establish published payload equality,
not original download history.

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
and source capability values without importing backend packages. The
schema-v2 snapshot [UD-QUANTS-REVIEW-v2.json](UD-QUANTS-REVIEW-v2.json)
reproduces every number in this section and section 4.1 from the pinned local
files via the documented CLI command.

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
`tests/test_scripts_gguf_quant_route_audit.py`,
`tests/test_hipengine_public_api.py`, `hipengine/__init__.py` (startup
isolation only; no runtime arithmetic or dispatch change);
loader scanner/model-map modules are references, not automatic edits.

- [x] Pin upstream model revision, payload checksum, file size and header identity
  for both UD files and the plain control.
- [x] Use actual AR and separate NextN maps; report disk/AR/ignored counts and
  aliases without double counting.
- [x] RED: `root.token_embedding`, 64 AR layers, ignored block64, raw Q4
  embedding, and MTP-only tensor exclusion.
- [x] RED: unsupported version, duplicates, invalid alignment, incomplete table
  versus incomplete data. Partial diagnostic mode cannot report loadable.
- [x] Report allocation-formula bytes, sidecars, reasons, and both hypothetical
  refusal treatments. Per-scope `allocation` sections in the schema-v2 report
  use `planned_qwen35_gguf_weight_allocation_nbytes` over unique
  `(source, layout)` residents, aggregate sidecar counts/bytes with reasons,
  and report both refusal treatments; real-file totals reproduce the section-4
  table exactly. The optional planar-Q5 sidecar initially had no production
  formula and was reported as formula-unavailable rather than guessed; since
  the 2026-09-07 admission repair it is sized exactly (the INT8 planar-tiles
  payload of the real converter chain, block formula below in the U1 repair
  note) and joins the sidecar accounting with the exact bytes.
- [x] RED: gfx1100 Q5 sidecar, gfx1151 Q6 exclusion, missing/nonliteral
  capabilities, env overrides and F32 contraction. Capability constants are
  read from backend source by a bounded AST literal reader (never an import,
  never expression evaluation); missing/nonliteral constants are reported per
  capability and resolve to the runtime default. Flag resolution goes through
  the shared pure policy API `hipengine.loading.qwen35_gguf_policy` used by
  both the runtime loader and this audit; the per-slot fallback now applies
  the model-wide F32 contraction, and cross-caller parity against the real
  backend packages is HIP-guarded in
  `tests/test_qwen35_gguf_policy_capability_parity.py`.
- [x] Reproduce this snapshot and version the repaired report schema.
  `docs/UD-QUANTS-REVIEW-v2.json` (`schema_version` 2) is the exact CLI output
  over the three real files; regeneration commands and the snapshot's
  producing source revision are in
  [UD-QUANTS-REPRO.md](UD-QUANTS-REPRO.md). The snapshot was refreshed at the
  UD-U1 F5 repair so `fp16_recurrent_state_default_on` reports the
  artifact-qualified policy identity (each backend lane carries
  `artifact_preset_key`); the earlier stamp-only snapshot is preserved
  byte-frozen as [UD-QUANTS-REVIEW-v2-pre-f5.json](UD-QUANTS-REVIEW-v2-pre-f5.json).
  Allocation accounting is identical between the two (the refresh changed no
  planned byte or route).

Run: `.venv/bin/python -m pytest tests/test_scripts_gguf_quant_route_audit.py
tests/test_loading_qwen35_gguf_policy.py -q`.
Exit: real planner accounting without device use, clearly distinct from consumer
qualification. U0 is complete (parser validation, production AR/NextN maps,
shared-policy capability resolution, allocation accounting, identity pins,
versioned schema, startup isolation); U1+ admission work follows.

Startup isolation (U0 review follow-up, 2026-09-07): a fresh process running
the audit must load no GPU backend package at all. The engine root
`hipengine/__init__.py` eagerly imported `hipengine.llm`, whose speculative
chain loads `hipengine.kernels.hip_gfx1100`; the root package now resolves
`LLM`/`SamplingParams` lazily (PEP 562) so a bare `import hipengine` is
CPU-safe and `from hipengine import LLM, SamplingParams, ExecutionProfile` is
unchanged. Bound by
`tests/test_scripts_gguf_quant_route_audit.py::test_fresh_process_audit_never_imports_gpu_backend_packages`
(fresh subprocess installs a meta-path guard rejecting
`hipengine.kernels.hip_*` / `cuda_*` / torch before any import, then runs the
actual CLI over a deterministic fixture) and the lazy-export contracts in
`tests/test_hipengine_public_api.py`. Deliberately untouched: the audit's
planner, policy, and report logic, and every runtime numerical or dispatch
path.

### U1. Role-safe Admission And Policy

**Scoped admission status: implemented and tested.** Load-time role/layout/
operation checks, resident prerequisites and native route exclusions pass the
136-test admission/invocation/execution bundle on 2026-09-07. Both published
files have zero default AR preflight refusals and public eager c1 smoke evidence
on gfx1151. This establishes the scoped U1 deliverable, not numerical, native
batch, NextN, named-profile or gfx1100 qualification.

The checked items and review rounds below are implementation history. Their
18/41 refusal inventories and pending review verdicts describe earlier commits.
The section 1.2 scope decision excludes generic lifecycle/private-mutation
frameworks; cold certificate simplification is deferred. Remaining numerical
and serving acceptance belongs to U2–U6, not further generic U1 authorization.

Dependencies: U0; CPU tests first.
Files: `hipengine/loading/qwen35_gguf_materialize.py`,
`hipengine/loading/qwen35_gguf_nextn.py`, backend policy/capabilities,
profile binders; proposed `tests/test_gguf_ud_admission.py`.

- [x] Define cold-path operation coverage records with role/shape/type/layout/
  input-output dtype/rows/fallback, using existing registry conventions.
  `hipengine/loading/qwen35_gguf_admission.py` ships `CERTIFIED_OPERATION_COVERAGE`:
  per `(operation, role class, resident layout)` records naming rows scope,
  input/output dtype, the four-axis `(layer, quant, variant)` consumer family,
  and a strict fallback; `test_coverage_records_use_existing_registry_layer_names`
  pins registry conformance. Certified consumers mirror the plain-lane dispatch
  tables, so unknown layouts/types/roles fail closed.
- [x] Preserve the existing planned artifact preset `gguf_ud_q4_k_m` for
  model/session admission and concrete per-tensor kernel quant keys; choose an
  equally explicit K_S preset identity. This is not a fifth registry axis.
  Both presets resolve only through pinned role-manifest fingerprints (UD K_M
  `5535c5bd…`, UD K_S `91130e16…`; distinct from both plain controls despite
  identical stamps), carry explicit `("ar",)` scopes, and never enter the
  kernel registry: per-tensor quant keys stay `gguf_*`
  (`test_preset_keys_are_session_identities_not_registry_axes`,
  `test_pinned_ud_q4_k_s_has_an_equally_explicit_preset_identity`).
- [x] RED: same stamp/different maps; same histogram/types swapped between
  recurrent and FFN roles. No unqualified arithmetic or automatic MTP inheritance.
  Manifest fingerprints differ for same-stamp/different-map and
  histogram-equal swapped-role fixtures, and a certificate bound to one
  manifest never covers the other
  (`test_role_manifest_fingerprint_distinguishes_same_stamp_different_maps`,
  `…_swapped_recurrent_ffn_types`); UD preset keys resolve to no execution
  profile and no MTP inheritance
  (`test_ud_quant_key_does_not_inherit_execution_profiles`,
  `test_mtp_scope_requires_an_explicitly_certified_preset`).
- [x] Inventory and constrain all K_M/K_S identity callers while preserving
  qualified plain controls and rollback. Identity callers: the runner policy
  identity (`_gguf_policy_identity`, all call sites) now appends the resident
  `artifact_preset_key` so same-stamp UD artifacts miss plain-certified policy
  tables while plain controls keep the historical key; the packaged
  hot-vocabulary selection appends the preset key to its identity; MTP serving
  evidence already binds `artifact.sha256`; execution profiles fail closed for
  unregistered quants. Plain K_M/K_S/0.8B/MoE-35B controls pass preflight.
  Review repairs tightened this further: unknown (non-pinned) manifests now
  get an unqualified sentinel identity instead of inheriting plain policy
  rows or the packaged hot-vocabulary selection (pinned plain-control
  fingerprints), the packed-decode-graph caller no longer crashes on
  preset-bound identities, and every certified coverage family is checked
  against actual registry registrations.
- [x] Separate per-tensor repack from global F32 contraction; retain existing
  MoE semantics for unchanged manifests. `gguf_ar_decode_repack_veto` and
  `gguf_ar_f32_linear_contraction` are independent predicates over the shared
  raw-IQ contract (rank-3 MoE exemptions preserved in both);
  `plan_qwen35_gguf_materialization` takes `repack_veto`/`contract_f32_linear`
  overrides and derives the previous combined behavior when unset. RED tests
  in `tests/test_loading_qwen35_gguf_policy.py` and
  `tests/test_qwen35_gguf_materialize_helpers.py`.
- [x] Preflight all requested operations before allocating; report every
  unsupported slot, not just first exception. `materialize_qwen35_gguf_weights`
  runs `preflight_qwen35_gguf_artifact` over the full plan (NextN-aware
  fingerprint) before `plan_qwen35_gguf_materialization` and any `malloc`, and
  raises one aggregated `Qwen35GGUFAdmissionError` naming every refused
  slot/mode. Measured: UD K_M refuses with exactly its 18 pinned unsupported
  AR slots, UD K_S with 41 (40 projections + Q3_K embedding), with zero
  allocator invocations before the refusal.
- [x] RED: raw-Q8 and sole-T16 alpha/beta cannot enter a BF16-pointer native-row
  owner; unsupported multirow BF16 embedding cannot silently use a singleton.
  `ar_decode_native_rows` is a distinct operation: only a dense-BF16 resident
  qualifies as the `dense_gemv_out_bf16` BF16-pointer owner; raw Q8_0 (UD),
  sole-T16 with no raw allocation, and dense-F32 (plain) alpha/beta are all
  refused; dense-BF16 embeddings have deliberately no certified record (the
  consumer drops rows), so a multirow gather is refused instead of silently
  resolving a singleton. Review repair: the refusal is now also bound at load
  time (`requested_operations`) and enforced at the actual native execution
  entries (`step_rows_native` / `capture_native_rows_graph`) before any state
  mutation or device call, using the real resident records.
- [x] Keep AR-only and AR+MTP capabilities distinct. Both UD presets carry
  `("ar",)` scopes; `mtp_nextn_draft` is scope-refused for them and for
  unresolved plain artifacts, and `materialize_qwen35_gguf_nextn_weights`
  refuses non-MTP-certified presets before draft validation, planning, or
  allocation.

Run after creating the file:
`.venv/bin/python -m pytest tests/test_gguf_ud_admission.py -q`.
Exit: unknown layouts fail closed, no plain-artifact certificate reuse for UD.

#### U1 review repairs (2026-09-07)

The U1 completion claim above was premature. An independent review found six
real defects in the landed work; the closure paragraph overstated what the
tests proved, and one claim (allocation-sentinel tests for the plain
K_M/K_S/0.8B/MoE-35B controls reaching allocation) had no test behind it at
all. Initial repairs were added with RED-then-GREEN CPU tests and immutable
worklog entries on `ud-quants`; later reviews reopened several findings.
The table records repair history, not current closure:

| # | Severity | Defect in the original U1 work | Repair (commit) |
| --- | --- | --- | --- |
| F1 | HIGH | `ar_decode_native_rows` was never requested by the materializer nor checked at native execution entry; raw-Q8/sole-T16/dense-F32 alpha/beta could reach the BF16-pointer owner. | Load-time binding + entry checks in `step_rows_native`/`capture_native_rows_graph` (`6a9712ee7`) |
| F2 | HIGH | The moe_experts whitelist omitted rank-3 IQ3_XXS although the materializer keeps it raw and `gguf_iq_gemv` registers selected-expert consumers. | Raw-only IQ3_XXS moe_experts coverage records with rank/shape/backend/dtype contract; rank-2 dense still refused (`cd73d1cef`) |
| F3 | HIGH | Coverage qualified op/role/layout/source only: unknown backends earned certificates, `cuda_sm120a` (no GGUF consumers) was positively certified, records carried placeholder consumer keys, the dense-F32 lm-head was certified with a convenient F32 input the default caller never supplies, and shapes were checked only via allocation bytes. | Round-1: backend keys restricted to the registered metadata surface; per-slot allocation-accounting aggregates planner refusals before allocation (`d665c01eb`). Round-6 consumer-key repair (not closure): concrete per-backend `GGUF_CONSUMER_LAYERS` declarations (AST-read, parity-tested), every record bound to a concrete four-axis consumer via the parity-tested dispatch-surface mirror, cuda_sm120a refused, dense-F32 lm-head refused under the actual BF16 caller dtype with the declared-F32-input route preserved (see the round-6 subsection) |
| F4 | HIGH | Certificates ignored slot filters and the effective plan contract: a one-slot preflight certified the full artifact, and contraction-enabled certificates were reusable on uncontracted plans. | `Qwen35GGUFPlanContract` + slot-scope recorded on every certificate; `certificate_covers_artifact` verifies both, legacy certificates fail closed (`39a7d40ae`). Third-round repair: the contract now binds the actual planned residents and coverage always requires the intended plan (see the round-3 subsection below). Fourth-round repair: intended-operation defaults and complete qualification accounting — refused reports can no longer supply authorizing contracts (see the round-4 subsection below) |
| F5 | HIGH | Unknown manifests (preset=None) inherited the plain stamp-based policy identity and the packaged hot-vocabulary selection. | `GGUF_UNQUALIFIED_MANIFEST_PRESET` sentinel + seven pinned plain-control fingerprints; loader and NextN materializer bind the qualification (`95f9fa2fd`). Fifth-round repair: the actual stamp-only policy callers (runner FP16 recurrent-state default, graph submission transport, private-c1 arena admissions) now bind the artifact qualification too (see the round-5 subsection below) |
| F6 | MEDIUM | `packed_decode_graph_min_replay_steps` destructively unpacked a 2-tuple identity that now optionally carries a preset key → `ValueError` for preset-bound residents. | Arity-safe unpack; preset-bound identities resolve only preset-keyed rows, never plain rows (`e2bb6d5e2`) |

These repairs closed the round-1 findings: `tests/test_gguf_ud_admission.py`
is green on CPU (real-artifact tests skip without the pinned files), admission binds to
actual role/shape/type manifests rather than stamps or histograms, the loader
preflights backend/materializability/scope before any allocation, both
K_M/K_S identity callers constrain UD, unknown manifests fall back generically
instead of inheriting plain-certified behavior, certificates carry their exact
slot/plan scope, and the native-row BF16-pointer owner contract is enforced at
the real execution entries. UD artifacts remain refused for execution until
U2-U5 deliver the missing dense consumers; the refusal lists are the honest
per-slot inventory for those units. Later review rounds (below) found further
defects, so U1 remains not accepted until every pending finding passes review.

#### U1 review repair regressions (2026-09-07, second review round)

A follow-up review accepted F6 and found two functionality regressions that
the repair commits themselves introduced; the remaining findings (F1, the
concrete-consumer substance of F3, F4, F5) stay pending re-review. Both
regressions are fixed with RED-then-GREEN CPU tests; U1 remains not accepted
until the pending findings pass review.

- **Raw moe_experts coverage collision (from the F2 repair).** The repair-2
  IQ3_XXS records reused the earlier raw records' `(operation, role_class,
  resident_layout)` coverage-index key, and the index was last-write-wins, so
  the IQ3_XXS record silently displaced Q3_K/Q4_K/Q5_K/Q6_K/IQ2_XS/IQ4_XS raw
  rank-3 experts: a rank-3 Q4_K MoE map with `decode_repack=False` was refused
  `consumer_unqualified`, and mixed-IQ expert maps broke. The index is now
  keyed per concrete source type (`_build_coverage_index` over
  `(operation, role_class, resident_layout, source_ggml_type)`) and rejects
  two different records claiming one source type at construction; each raw
  expert format keeps its own consumer record (gguf_q\*\_k raw family vs
  gguf_iq3_xxs selected consumers) instead of one merged blob.
- **Mandatory allocation-formula validation refused the Q5 planar sidecar
  (from the F3 repair).** The per-slot materializability check added with the
  F3 repair calls `planned_qwen35_gguf_weight_allocation_nbytes`, which had no
  formula for the `qmicro_planar` allocation name — so the env-gated
  (`HIPENGINE_C8_Q5_PLANAR_DP4A=1`) planar Q5 T16 `ssm_out` resident that the
  C8-P2 route materializes today became a hard `planner_refused`. The exact
  formula is derived from the actual converter/allocation ABI — the INT8
  `planar.tiles` payload of
  `convert_gguf_q5_k_qmicro_tile16_to_planar(repack_gguf_q5_k_qmicro_tile16(raw[None, ...]))`
  with shape `[experts, out/16, bytes_per_row/176,
  GGUF_Q5_K_QMICRO_PLANAR_T16_BLOCK_BYTES=3328]` (for the measured 5120x6144
  `ssm_out`: tiles 22,118,400 + raw 21,626,880 + planar 25,559,040 bytes) —
  gated to Q5_K T16 residents exactly like the materializer's own sidecar
  check. Tile-alignment and unsupported-layout refusals are unchanged, and the
  never-produced standalone `gguf_q5_k_qmicro_planar_v1` resident layout stays
  refused. The route audit now reports the sidecar formula-sized instead of
  formula-unavailable.

Re-review status: F6 accepted; F1, F3 (concrete consumer), F5 pending;
F4 was re-opened by round 3 and repaired there; this regression round is
subject to the same review.

#### U1 review repair round 3 (2026-09-07): F4 resident-plan binding

A third review accepted the F2/Q5-planar regression repairs and found F4
still open on two points, both now repaired on CPU with RED-then-GREEN tests
(`worklog` entry `ud-u1-f4-resident-plan-certificate`):

- **Coverage approval could skip plan verification.**
  `certificate_covers_artifact` compared the effective plan contract only
  when the caller volunteered one; omitting it still accepted an
  override-specific certificate as operation coverage. The intended
  `plan_contract` is now a required argument and is always verified; a
  certificate without recorded plan metadata fails closed; the pure
  source-identity check moved to a distinct
  `certificate_matches_artifact_identity` whose contract states it can never
  authorize operations.
- **The recorded contract missed env-resolved layout selectors.**
  `Qwen35GGUFPlanContract` recorded caller kwargs and env-resolved booleans,
  but the layout selectors read inside the planner
  (`HIPENGINE_GGUF_SELECTED_GATE_UP_X8`, `..._GATE_UP_RAW`,
  `..._SELECTED_X8_REPACK`, `..._SELECTED_DOWN_RAW`,
  `HIPENGINE_GGUF_Q8_0_RAW_SIDECAR`, `HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL`,
  `HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR`) changed actual residents without
  changing any recorded field: a rank-3 Q4_K MoE preflight with
  `decode_repack=True` records identical contracts with the gate/up X8 env
  off (T16 residents) and on (X8 residents). The contract now carries a
  canonical sorted record per checked slot — logical slot path, source
  identity (name/shape/GGML type), resident layout, per-tensor quant key,
  allocation names, planned per-allocation byte counts, sidecar layouts —
  plus a deterministic sha256 digest that is always re-derived from the
  records. Coverage verifies the intended contract per slot: full-artifact
  intent requires a full-artifact certificate whose records subsume the
  intended records; subset intent requires slot-scope nesting plus exact
  per-slot record equality (narrowing preserved, enlargement refused);
  contracts whose records do not cover their claimed slot scope fail closed.
  The loader (`materialize_qwen35_gguf_weights`) is a real certificate
  consumer: an `admission_certificate` argument is re-verified against the
  fresh admission report before any allocation, and residents carry the
  minted certificate for downstream re-verification. Identical effective
  plans compare stably regardless of input spelling/order; the pinned UD
  refusal inventories (18 K_M planner-refused slots / 41 K_S) are unchanged.

Re-review status after round 3: F6, F2, and the Q5-planar regression repair
accepted; F4 repaired here (two round-3 points above) and F1, F3 (concrete
consumer), F5 still pending re-review. U1 remains open.

#### U1 review repair round 4 (2026-09-07): F4 operation defaults + complete qualification accounting

A fourth review found two HIGH scope holes in the round-3 certificate work
(`56dba7d3c`), both repaired on CPU with RED-then-GREEN tests (worklog entry
`ud-u1-f4-operations-and-completeness`):

- **Operation membership was only checked when the caller volunteered an
  operation set.** Omitted, `certificate_covers_artifact` verified no
  operations at all: an `ar_decode_c1`-only certificate returned True for a
  prefill intended plan with identical residents, and (neighboring
  counterexample) identical residents would likewise have covered an
  `ar_decode_native_rows` intent. The intended operation set now defaults to
  the intended contract's own checked operations — the whole intended
  contract must be certified — and an explicit set must be non-empty and
  narrow within BOTH the certificate's certified operations and the intended
  contract's checked operations. Planned resident bytes establish
  allocations, never row-operation or dtype qualification.
- **Refused preflight reports could supply authorizing contracts.** The
  preflight records only successfully qualified slots, and contract
  self-consistency merely required a nonempty record set, so the refused
  F32-alpha/beta native-rows report (alpha/beta omitted from its records)
  compared True against the contracted certificate under an explicit native
  operation. The contract now binds complete expected-versus-actual
  qualification: `required_plan_slots` enumerates every slot the preflight
  tried to qualify (participating or refused — planner, consumer, unknown
  role class, and unknown `slot_filter` entries, which are now refused
  instead of silently ignored) and `operation_scope_refusals` records
  operation-level scope gates (MTP draft). A contract is
  authorization-capable (`Qwen35GGUFPlanContract.is_complete`) only when its
  records cover exactly the required accounting and nothing was refused;
  `certificate_covers_artifact` requires completeness on both the recorded
  and intended sides, `Qwen35GGUFAdmissionReport.certificate` refuses to
  mint an incomplete contract, and a named nonempty slot scope with no
  verified residents fails closed. Refused reports keep their partial
  contracts for debugging — they can no longer authorize. The digest still
  covers only successful records (unchanged format); completeness is bound
  by the enumeration accounting, so the proof is not circular.

Re-review status after round 4: F6, F2, and the Q5-planar regression repair
accepted; F4 repaired here (the two round-4 points above) and awaiting
re-review together with F1, F3 (concrete consumer), and F5. U1 remains open.

#### U1 review repair round 5 (2026-09-07): F5 stamp-only policy callers

A fifth review round accepted the F4 round-4 repair and confirmed the
remaining F5 hole unchanged since round 1: the sentinel and the 3-tuple
policy identity existed, but actual production callers still passed only the
header stamp (or ``(geometry, stamp)``) into artifact-qualified policy
selection, so an unknown non-pinned manifest sharing a qualified control's
stamp silently inherited plain-certified policy. All stamp-only callers of
artifact-qualified tables were inventoried and repaired on CPU with
RED-then-GREEN tests (worklog entry `ud-u1-f5-policy-callers`):

- **Runner FP16 recurrent-state default**
  (`Qwen35GGUFFullStackRunner` initialization →
  `_gguf_fp16_recurrent_state_enabled`): the initializer now passes the
  loader-resolved `artifact_preset_key` of the actual resident weights. Only
  a qualified plain control (`artifact_preset_key=None` from admission) may
  inherit the stamp-certified backend default
  (`GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES`); UD-preset and
  unknown-manifest artifacts resolve the generic strict FP32 storage default.
  The `HIPENGINE_GGUF_FP16_RECURRENT_STATE` environment value remains the
  documented developer opt-out for every identity — it is an explicit
  override, not a certified default, and no supported arithmetic claims were
  widened.
- **Decode-graph submission transport**
  (`_resolve_gguf_decode_graph_submission_transport` and its two production
callers, single-slot and packed capture): the policy row lookup is now
identity-keyed exactly like the F6 packed-floor fix — plain identities keep
the historical ``(geometry, stamp)`` rows; preset-bound and unknown
identities resolve only an exact preset-keyed row (none ship today) and
otherwise the generic hipgraph fallback. No tuple truncation or identity
stripping; explicit `submission_transport` requests still override for any
identity.
- **Private-c1 arena admissions** (same bypass class, found by the caller
inventory): `_resolve_gguf_private_c1_small_weight_arena`,
  `_resolve_gguf_private_c1_decode_scratch_arena`, and
  `_resolve_gguf_private_c1_weight_arena_max_allocation_bytes` are
  identity-keyed the same way, and `Qwen35GGUFResidentSession`
  initialization derives the artifact qualification from the actual manifest
  (header-only tensor map + structural NextN map, exactly like the
  materializer) before materialization instead of trusting the header stamp.
  Lazy startup isolation is preserved: the manifest derivation happens only
  when a backend actually ships arena policy rows.
- **Caller inventory outcome**: every other stamp/policy-table consumer in
  the runner already resolves through the full `_gguf_policy_identity`
  3-tuple (dense-pair/norm-residual/dual-WMMA/rocblas/chunk/scratch tables,
  whose 3-tuple keys miss plain 2-tuple rows by construction); the packaged
  hot-vocabulary and NextN materializer paths were already bound by the F5
  round-1 repair; the remaining stamp consumers are generic type-family
  policies deliberately left untouched (GDN quant+head-shape recurrence
  modes, host-token-embedding GGML type gates, shape-keyed rowtile/QK
  postprocess tables, and default-off env-gated probes) — no blanket
  disabling and no artifact renaming.
- **Audit parity**: `gguf_fp16_recurrent_state_default` (the shared pure
  policy mirror) carries the same artifact binding, and the quant-route
  audit reports `artifact_preset_key` and an artifact-qualified
  `fp16_recurrent_state_default_on`, still resolving capabilities from
  backend sources without importing a backend package. Evidence follow-up
  (accepted in the round-5 static review as the sole remaining item): the
  schema-v2 snapshot [UD-QUANTS-REVIEW-v2.json](UD-QUANTS-REVIEW-v2.json) was
  regenerated at the F5 source revision so the stored evidence matches the
  repaired audit — the exact-UD K_S file on gfx1151 now reports the FP16
  default `false` with `artifact_preset_key: gguf_ud_q4_k_s`, the plain K_M
  control keeps `null`/`false`; the pre-F5 snapshot is preserved byte-frozen
  as [UD-QUANTS-REVIEW-v2-pre-f5.json](UD-QUANTS-REVIEW-v2-pre-f5.json)
  (snapshot history and producing revisions in
  [UD-QUANTS-REPRO.md](UD-QUANTS-REPRO.md)).

Re-review status after round 5: F2, F6, the Q5-planar regression repair, and
F4 accepted; F5's named stamp-only bypass closure was accepted in static
review at `c5b9588b8` (172 focused tests pass), with the snapshot-evidence
mismatch repaired as the follow-up above (current snapshot regenerated at the
F5 source revision; pre-F5 snapshot frozen); F1 (native-row execution
dependencies) and F3 (concrete backend/dtype consumer substance) remain
pending. U1 remains open.

#### U1 review repair round 6 (2026-09-07): F3 concrete consumer qualification

Round 6 added the following consumer-key and dtype metadata repairs on CPU
(worklog entry `ud-u1-f3-concrete-consumers`; no GPU, no new kernels, no
numerical change). Its closure claim was rejected by the subsequent integrated
review: key existence does not prove resident prerequisites or actual operand
ABIs, and storing an F32-input declaration does not bind authorization.

- **Backend binding is now concrete registration, not a known arch name.**
  Each hardware backend package declares its `GGUF_CONSUMER_LAYERS` as a
  source literal (`hip_gfx1100`/`hip_gfx1151` declare the full certified
  surface; `cuda_sm120a` explicitly declares none after a real inventory —
  its scaffold registers moonshine/maple/PARO families but no `gguf_*`
  linear/embedding/dense/router/GDN/selected-expert consumer). Admission
  reads the declarations with a bounded AST literal reader (the audit
  capability-reader convention: never an import, never evaluation;
  annotated and plain assignments); a slot whose certified consumer layer
  is not declared is refused with the missing layer named, aggregated
  before any allocation. `tests/test_qwen35_gguf_consumer_surface_parity.py`
  proves the declarations are neither lies nor stale: every concrete
  consumer key named by every certified record (both row-mode variants) is
  actually registered on every declaring backend — with no placeholder
  skips — the cuda inventory holds, and both HIP declarations equal exactly
  the certified consumer-layer set.
- **Coverage now names concrete four-axis consumers bound to the actual
  runtime dispatch surface.** `hipengine/loading/qwen35_gguf_consumer_surface.py`
  mirrors `runtime/gguf_linear._DISPATCH_TABLE` row-for-row (layout,
  activation, output, layer, quant token, variant, ABI, plus the
  `_variant_for_rows` multirow rewrite including the Q4-T16 special case);
  a parity test asserts exact equality, so coverage can never name a
  variant the dispatcher would not resolve. Placeholder records (the
  former `<from-weight>`/`None` quant/variant forms and the nonexistent
  `moe_selected`/`gdn_chain` layers) are gone; the selected-expert records
  name the real registered keys per format (`moe_linear` for Q3_K/IQ* and
  the T16/X8 repack families, the `linear` raw selected keys for Q5_K/Q6_K),
  with one documented exception: raw rank-3 Q4_K selected experts are
  consumed through the direct `gguf_q4_k_gemv` module wrapper the runtime
  actually calls, so that record names the module symbol (parity-checked)
  and the registry-mediation gap is filed in `docs/REFACTOR.md`.
- **The dtype repro is fixed at the actual caller contract.** The default
  records now mirror what the production callers actually supply: BF16
  activations without an input override. A dense-F32 lm-head therefore has
  NO default `lm_head_f32_logits` record — the dispatch table has no
  `(dense_f32, bf16, f32)` row — and the preflight refuses `root.lm_head`
  with that reason before any allocation (the F32-head counterexample now
  RED-fails on the old code and passes on the new). The registered
  F32-activation route (`dense_gemv/f32/f32_hidden_f32_out`, the c1/verifier
  F32-input route) stays certifiable when the caller declares
  `f32_input_operations=(lm_head_f32_logits,)`; the declaration is
  recorded on the plan contract. However, the integrated review found that
  BF16 records take precedence even under this declaration and verification
  does not compare it: certificates can transfer across activation contracts.
  Only the bounded default dense-F32-head refusal was established here.
- **Preserved exactly:** the pinned UD refusal inventories (18 K_M / 41 K_S
  planner-refused slots), all qualified plain/MoE controls on both HIP
  backends (real-artifact preflights), the Q6 head (257,256) T16 shape
  refusal, the Q5 planar sidecar formula, the raw MoE 7-format index, the
  F4 certificate completeness machinery (the contract gains only the
  additive `f32_input_operations` field), and the audit schema-v2 snapshot
  (regenerated identically; the audit does not consume admission records).
  The loader forwards `f32_input_operations` to the preflight, but no
  production caller supplies it. Native head execution currently passes BF16
  `scratch.norm`; declaring F32 without a real compatible operand/adapter is
  not a valid repair. Loader/entry execution dependency closure remains F1.

#### Integrated expert review and resident-prerequisite repair

The integrated review of `a6f79f562` rejected the round-6 F3/F4 closure and
found an additional F5 profile-binder bypass. F2's seven-format raw-expert index,
F6's qualified identity propagation, the Q5 planar-sidecar byte formula, and the
bounded F4 completeness/F5 policy-caller repairs passed focused preservation
checks. These bounded successes do not establish U1 acceptance.

The first authorized implementation unit adds
`validate_qwen35_gguf_resident_prerequisites` in
`loading/qwen35_gguf_materialize.py`. It checks canonical source geometry and
every selected primary/allocated-sidecar route before allocation-byte
accounting. The actual materializer converter dispatch and the CPU repackers
share immutable `GGUFRepackShape` contracts from `quant/gguf_repack.py`.
Admission aggregates resident failures even when the requested operation does
not consume that selected slot; single-weight materialization also checks the
boundary before reading payloads. No allocation formula, arithmetic, kernel,
repack-selection flag, or policy default changes.

The byte-neutral Q6 planar `(5121,256)`, Q5 qmicro expert `(2,257,256)`, and
Q4/Q5/Q6 X8 expert `(2,257,256)` cases now refuse at the same rank/tile/block
boundary as their CPU converters. Legal neighbors, all 11 primary repack
routes, allocated Q4 T16/Q6 X8/Q5 planar/raw sidecars, and non-repacked
residents have focused CPU coverage. Optional expert-sidecar eligibility is
not an allocated sidecar and does not impose tile alignment on raw experts.
The real tiny-GGUF loader negative covers both full and filtered loads with
zero payload reads or allocator calls. Exact test outcomes and the full review
handoff are in the `ud-u1-resident-prerequisites` worklog entry.

**Current status (resident/NextN prerequisites and integrated F3/F4 accepted after GDN geometry closure; F1 pending independent review):**

| Finding | Status and remaining work |
| --- | --- |
| F1 | Implemented pending review: additive native execution dependency closure, real session/engine → loader requests, F4 consumption at native eager/capture/replay/direct-layer entries, exact row/state/physical-owner gates. CPU structural evidence only; see the F1 section below. |
| F2 | Bounded repair accepted: preserve the seven raw expert formats and rank-2 IQ3_XXS refusal. |
| F3 | Accepted after GDN geometry closure: shared production invocation owners and explicit selected-call intents. C1/indexed/prefill auxiliary ABIs, GDN handoffs, native alpha/beta and ordered selected partners are represented without changing kernels or numerical defaults. |
| F4 | Accepted: backend plus resolved invocation/partner identity binds alongside actual residents. Independent required slots/calls and scope refusals prevent partial/filter/operation narrowing from promoting incomplete plans. |
| F5 | Bounded identity-aware policy callers repaired; profile authorization OPEN. Actual LLM auto quant resolution can select the plain production profile for an unknown manifest; its binder sets FP16 state through the environment, bypassing the qualified default without a user FP16 override. |
| F6 | Bounded repair accepted: retain preset-qualified identity in packed replay policy calls. |

### Integrated F3/F4 invocation-contract repair

The CPU-safe production owners are
`loading/qwen35_gguf_consumer_surface.py` and
`loading/gguf_selected_contract.py`. Runtime linear/embedding/router dispatch,
selected FFN topology and argument marshalling, native alpha/beta, and named
conv/GDN/RMSNorm wrappers consume their metadata. The former independent
20-row linear dispatch table is now a generated compatibility view. Consumer
availability remains distinct from numerical/profile qualification.

`SelectedCallIntent` explicitly names `single`, `dual`, `dual_silu`, or
`weighted_down`, ordered weight slots, actual input/output types, routing and
buffer owners, and selected lanes per token. Full-model defaults use the same
caller-plan owner as production. Load scope is independent: filtering to the
gate slot cannot silently replace a paired gate/up intent with a singleton.
A missing required partner refuses before payload access/allocation. An
explicit singleton diagnostic can qualify a supported single-weight load,
but its certificate cannot cover a full-model paired intent. An explicit empty
intent tuple supports non-expert diagnostics, not selected calls with missing
dependencies. The dual caller now owns the existing ordered two-single fallback
as well as the direct dual launch; this is host factoring, not new arithmetic.

Certificates bind the canonical resolved invocation records and independently
required calls as well as the resident records: backend, ordered partners and
their complete source/resident/allocation/sidecar identity, operand types and
owners, adapters, operation/row limits, geometry and relevant scalar parameters.
The intended contract is required; omitted backend/operation arguments derive
from it. Completeness is checked before any narrowing. Equivalent effective
contracts can compare equal despite different declarations; a declared F32
input cannot be backed by a BF16 record. This is ordinary immutable internal
metadata, not a cryptographic proof against fabricated contracts.

Auxiliary boundaries are explicit: c1 conv/GDN differ from indexed native
rows; conv writes F32, segmented GDN has F32 conv/state/output and BF16
gate/alpha/beta operands, and prefill GDN publishes BF16 through its composite
caller. `ssm_norm` belongs to that composite. The row-local `ssm_out` handoff
resolves actual F32 input where supported or records the executed BF16 cast.
Native alpha/beta name the direct BF16 owner, not a rewritten prefill kernel.
The P2 review follow-up validates GDN geometry through the same CPU-safe rules
used by the native exports and Python wrappers: positive head counts/dimensions,
value-head count divisible by key-head count, and a value-head limit of 128
for both c1/row-local and segmented recurrence (F32 or FP16 state). Only the
baseline prefill export uses the uncapped positive value-head contract.
Native-source predicate tests distinguish these boundaries independently of
mocked launch returns. The runner and admission share exact inner-size/value-head
division; conv channel/kernel positivity is shared too. Geometry refusals retain required
slot/invocation accounting and cannot mint or transfer certificates. This
follow-up is pending independent review, not F1/F5 closure.
Native head input stays BF16; no F32 declaration was added to force admission.
The historical Q8T16 `(bf16 selector, fp16 output)` row keeps its runtime key,
but metadata identifies its actual FP16 pointer ABI and does not qualify a
supplied BF16 pointer against it.

Limits are fail-closed: optional raw/T16 selected DP4A adapter experiments
require their own qualified intent; mandatory X8 adapters are represented.
The baseline prefill GDN contract requires F32 state rather than guessing an
FP16 prefill plan. Existing runtime implementations and profile/variant
selectors remain available and unchanged; these metadata contracts do not
authorize a numerical profile or an unrepresented caller override. Internal
optimized alternatives share a caller contract only when its supplied operands,
output/state ownership, geometry and row semantics are identical; arithmetic
selection still belongs to the existing profile/variant policy. A baseline key
alone does not certify a different caller adapter or external workspace.
Borrowed selected partners have no implicit authorization path: every partner
must be present in the bound resident plan.

F1 complete execution dependencies and runtime-entry certificate consumption,
and F5 artifact-bound profile authorization, remain OPEN and require the next
units after review. No new entry guards, profile authorization, codec, kernel,
math or performance result is claimed here. U2+ and published dense UD execution
are not implemented; the pinned 18 K_M / 41 K_S planner-refusal inventories
remain the expected default result.

#### F1 native load dependencies — implemented; runtime authorization simplified

`loading/qwen35_gguf_execution.py` declares the full native weight-consuming
route, using the accepted linear/embedding/auxiliary and selected-call owners.
Native operations now include embedding, general projections, recurrent
auxiliaries, the BF16-input/F32-output head, router and ordered selected
gate/up/down partners where present. A native-only materializer request adds
to `DEFAULT_AR_OPERATIONS`; it cannot erase the ordinary required calls.
`execution_routes=("native_rows",)` or `("native_graph",)` on the resident
session is passed through the actual full-stack runner to the materializer.
The generator's selected native batch path passes both routes explicitly.
Default `("eager",)` preserves the separate existing c1/row-local/prefill scope.
A later native request on eager-only residents refuses, rather than replanning
after allocation.

Before payload reads or device allocation, the loader checks the intended
calls. Session construction also checks shared residents against the full native
load contract. This check is not repeated during inference. Native entry points
check the construction-time route choice, row bounds and unsupported adapters;
graphs also check closed owners, token IDs and their context bound. Private
resident pointers, scratch views and captured geometry are session-owned and
must not be replaced while live. Reconfiguration requires a new session.

Unsupported combinations stay explicit: dense-BF16 multirow embedding (the
registered leaf is singleton-only), native dense-F32 head with BF16 scratch,
non-BF16 native alpha/beta, rows outside 2–8, non-BF16 native KV,
unrepresented host/deferred or
expert-sidecar adapters, and the accepted F4 selected-adapter/state restrictions.
Known session-level misses refuse before loading; unanticipated later entry
requests refuse before mutation/device work. The embedding runtime itself now
uses the shared row predicate before registry resolution and refuses multirow
BF16 before the singleton leaf; no row loop, numerical adapter or kernel was
invented. Raw embedding still forwards rows.

The new tests use real tiny GGUF files and mocked allocations/calls. They prove
structural control/ABI properties, **not numerical or GPU certification**.
F2/F6, repacks, NextN ordering, GDN geometry and F4 transfer gates are preserved
by the affected CPU bundle. F5's integrated LLM profile-binder hole remains
**OPEN**; this unit does not grant named-profile permission or change a profile,
math, environment flag or numerical default. Exact validation and scope limits
are recorded in the `ud-u1-f1-execution` worklog entry.

#### F1 physical-operand authorization experiment — historical, removed

The following records the experiment in `6fb94aae9`, not the current runtime.
Direct cleanup removed its pointer inventories, index receipts, invocation
contexts and repeated full-model authorization. The experiment tested arbitrary
private-field mutation as though it were a supported reconfiguration API.
Construction-time compatibility checks and existing buffer lifetime rules are
the boundary used for UD bring-up. No GPU speed claim follows from this removal.

Review of `1b3ee1417` accepted the load dependency closure but found two HIGH
physical-binding gaps: sampler output replacement and named KV/position/rotary
view replacement could evade the graph key, and direct native layer methods
did not validate their supplied hidden/output/index pointers. The earlier
physical-ownership claim was therefore incomplete.

The experiment's `loading/qwen35_gguf_native_operands.py` owned `NativeRowsOperands`, the
actual native launch **and host-readback** plan: token IDs, BF16 ping-pong
hidden rows, logits, segmented-index inputs, sampler block values/indices,
sampler output IDs/values, and the persistent host token destination. Native
enqueue and replay consume that plan. A recursive inventory walks every named
scratch dataclass field, including nested KV spans, cache tuples, position and
rotary Tensor views, their dtype/shape/strides/device, owning buffers and static
geometry. Merely retaining `scratch.buffers` cannot hide a changed named view.
Host array contents are dynamic; the invocation's decode bound, graph context
bounds, physical owners and view geometry are not.

Both direct native layer methods require an owner-issued
`NativeInvocationContext`. The real enqueue supplies it. Validation checks the
current certificate/owner plan, actual runner and scratch views, layer geometry,
BF16 row-prefix inputs/outputs, buffer ranges/dtypes/non-aliasing, and exact
compact index pointers before any kernel or library access. Prefix views of
larger resident buffers and the real compact KV/position subview constructor
are covered; arbitrary nonzero pointers or unrequested offsets are not authority.
`NativeIndexBinding` records the canonical host metadata and destination after
the allocator's successful H2D publication. This is producer/ownership evidence,
**not an assertion that device contents were read or numerically verified**.

Capture stores the issued context. Replay checks it, including freshly derived
compact views and all sampler/readback operands, before H2D, position publication
or graph launch. Missing legacy contexts fail closed. CPU tests exercise both
real layer entries, real enqueue forwarding, legal prefix views, changed named
operands with unchanged owning lists, and changing contents with stable owners.
That correction did not include GPU, kernel, math, profile, default-route or
U2 work. Its mutation-defense tests are historical, not current acceptance gates.

#### F5 named-profile authorization experiment — historical, removed

The following describes the rejected experiment, not the current API. Generic
factory qualification/rollback was removed during direct cleanup. The
cancelled lifecycle prototype was also discarded after a local backup. Existing
profile binders retain their pre-experiment process-scoped behavior; use a fresh
process without a named profile for initial UD bring-up. Model hot replacement,
custom-factory transactions and concurrent profile isolation are not bring-up
requirements. Numerical production-profile validation remains a later gate.

The final known integrated-review bypass was profile-generated permission:
`LLM` auto or explicit plain quant lookup selected a plain production plan for
an unknown same-stamp manifest; its binder set FP16 state through the environment
although the artifact-qualified runner default was false. That internal write
was not an explicit user override.

`RuntimeProfilePlan.qualifier` now owns the plugin's cold qualification gate.
`LLM` forwards actual header metadata; resolution checks it before variant
package loading or factory invocation. `ResolvedRuntimeProfile` binds the
artifact identity separately from its unchanged variant-manifest hash and
rechecks construction inputs, custom-generator metadata, and direct public
binder calls. Missing context fails closed. Generator metadata conflicts and
invalid artifacts refuse before any binder environment write; failed binder
application restores the previous environment.

`loading.qwen35_gguf_admission.qwen35_gguf_artifact_identity_from_info` is the
shared header identity adapter used by both runtime policy and profile callers.
It includes the same structural NextN map as loader admission and preserves
pinned plain `None`, unknown sentinel, and exact UD preset identities. Existing
dense-27B/MoE-35B Q4_K_M profile scopes remain supported. Small-model and Q4_K_S
controls keep their independent admission/policy identities but do not acquire
numerical certification of these verifier-shaped plans. Unknown/UD artifacts
clearly refuse these named profiles (strict, production, and batch-invariant);
there is no registered qualified generic strict profile to substitute. Generic
AR admission, native physical qualification and MTP scope remain separate.
UD K_M/K_S still have their 18/41 missing-consumer refusals.

CPU tests cover actual public LLM auto/explicit selectors, truthful tiny GGUF
metadata, same-stamp role swaps preserving histograms, exact UD artifacts,
known plain dense/MoE controls, custom factories, missing/stale context, direct
resolved binders, strict fallback, environment failure/restore behavior, and
explicit developer overrides. No synthetic manifest is declared numerically
certified. Real-header positive tests skip when local control files are absent;
unknown-header negatives are portable. No GPU, new kernel, math, performance,
or U2 work is included. Successful binders retain their existing process-scoped
lifecycle; explicit overrides are not a qualification bypass. **F5 awaits
independent review; U1 is not complete until its integrated audit.**

### U2. Independent Codec Oracles

Dependencies: U0; CPU-only before GPU leaf tests.
Files: `hipengine/quant/gguf.py`, codebook module only if existing conventions
justify it, proposed `tests/test_gguf_ud_codecs.py`, `tests/fixtures/gguf_ud/`.

- [x] Add llama.cpp-pinned IQ3_S/IQ2_S byte-to-F32 fixtures before CPU decoders;
  add standalone IQ4_NL independent coverage. Synthetic fixtures cover all seven
  type ABIs at `llama.cpp@17252c769a63c1cb650ce98ae309cf4de0da7778`.
  The earlier donor pin is unavailable in this host's checkout. Fixtures compile
  an exact Git archive, not the moving worktree. RED: two missing decoders;
  GREEN: 136 codec/reader/admission tests. See the `ud-codecs` worklog entry.
- [x] Cover all codebook/sign/high bits, struct sizes, scale corners, row
  transitions and dimension order. Selector-domain assertions exposed missing
  IQ2_XS grids and IQ3_XXS sign selectors in the original random fixture; the
  pinned C-oracle generator now supplies exhaustive selector coverage at
  nonzero scale. This is not exhaustive Cartesian-product coverage.
- [x] After payload checksums, extract small real-row fixtures from multiple
  roles/layers, first/last blocks and actual dense widths. `real_rows.npz`
  covers first/last full rows of 20 tensors across both published files, all
  seven types, and widths 5120/17408. Both full payload hashes were recomputed
  on zbook before extraction and matched the identity pins.
- [x] Record donor revision, fixture hash, generation command and license.
  New decoder/tables cannot be their own sole oracle. The fixture generator
  compiles the pinned C implementation; tests need neither model nor compiler.
- [x] Gate decoded F32 independently; separately specify BF16/FP16 rounding
  and strict accumulation/output contracts. See
  `tests/fixtures/gguf_ud/README.md`: raw dense leaves keep F32 decoded weights,
  consume BF16, and use a declared separate-multiply/add F32 reduction with
  F32 or final RNE-BF16 output. FP16 is unsupported, not implicitly qualified.
  The CPU codec suite passes 42 tests; model-quality gates remain separate.

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
  Raw BF16-input BF16/F32-output leaf passes exact real-row gates on gfx1151;
  Raw dense layer materialization is integrated; gfx1100 numerical validation
  remains open.
- [ ] Add standalone IQ4_NL and IQ3_S strict consumers and independent leaf gates.
  The same bounded gfx1151 leaf evidence exists for both IQ types and IQ4_XS.
  Combined codec/dense suite: 102 passed; cache-only trace: 72 passed with all
  eight expected instantiations. See the `ud-dense-iq-leaves` and `ud-q3-dense`
  worklog entries. This is not model or full-shape qualification.
- [ ] Cover all 18 refused AR slots and all other incomplete dense operations.
  Preflight now accepts all K_M AR slots on both backend declarations. Only
  gfx1151 eager c1 execution has model evidence; other modes remain unqualified.
- [ ] Validate c1 AR, teacher logits and compact memory before broader modes.
  gfx1151 published-file smoke generates eight tokens with finite full logits
  from `The capital of France is`: ` Paris.\nThe capital of Germany is`.
  This is a smoke, not a teacher-quality gate or compact-memory claim. See
  `worklog/entries/20260907T035440.022336Z-lhl-ud-km-c1-integration-c5bbd6.md`.
  Public `LLM.generate()` matches this completion in a fresh process without
  importing torch; see the `ud-km-public-smoke` worklog entry.
  K_S public generation also returns this completion without torch after raw
  IQ3_XXS/IQ2_S and Q3_K embedding integration; see
  `worklog/entries/20260907T041214.806691Z-lhl-ud-ks-integration-e0d117.md`.
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
- [ ] Dense IQ3_XXS and IQ2_XS from existing math. Both raw leaves have
  bounded gfx1151 numerical evidence. IQ2_XS passes exact BF16/F32 real-row
  projection gates and synthetic one-hot decode coverage; model residency still
  uses its BF16 fallback pending integration. gfx1100 remains unverified.
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
