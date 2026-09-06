# Unsloth Dynamic (UD) GGUF formats: what hipEngine can run, and what supporting them would take

Last updated: 2026-09-06

Status: analysis and campaign plan. This document contains no new kernels and no
GPU measurement. Every number here comes from reading files and from the repo's
existing loader/plan code paths; nothing was loaded onto a device to produce them.

Related: [`GGUF.md`](GGUF.md) (intake and native-quant plan),
[`QUANTS.md`](QUANTS.md) (the wider quantization portfolio, including the
per-type coverage table this document's status lists must be read against),
[`GGUF-Q3-OPT.md`](GGUF-Q3-OPT.md) (the merged `UD-Q3_K_M` work, which is the
Qwen3.6-35B-A3B mixture-of-experts route),
[`KERNELS.md`](KERNELS.md) (kernel catalog),
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) (numerical gates a new kernel must pass).

## Summary

- **"UD-" (Unsloth Dynamic) GGUFs are mixed-quantization files.** Instead of one
  quantized format for every weight, each tensor gets its own format, chosen by a
  per-tensor sensitivity study. A UD file therefore names many formats in one
  model.
- **hipEngine runs UD formats on the Qwen3.6-35B-A3B mixture-of-experts route and
  cannot run them on the dense route.** Dense means an ordinary
  one-model-many-layers network with no expert routing: Qwen3.5-0.8B, Qwen3.6-9B,
  and Qwen3.8-27B. On the dense route, a UD file whose tensors include formats the
  dense matrix-multiply kernels do not implement is refused at load.
- **The refusal is correct today, and widening it is not a small change.**
  Qwen3.8-27B `UD-Q4_K_S` has 41 tensors in five formats that no dense kernel
  decodes (Q3_K, IQ4_NL, IQ3_S, IQ3_XXS, IQ2_S); `UD-Q4_K_M` has 18 tensors in
  three of them; plain `Q4_K_M` has none.
- **The cheap workaround is disqualified on memory.** The existing
  "convert-to-dense-BF16" fallback, if it were widened to cover these formats,
  would raise Qwen3.8-27B `UD-Q4_K_S` from 14.29 GiB stored on disk to about
  35.8 GiB resident on the GPU — more than the plain `Q4_K_M` file's 15.9 GiB
  resident, and beyond what either target machine can hold alongside a context.
- **One practical trap:** two switches key on the GGUF header's claimed format
  rather than the file's actual format mix, so a UD file that cannot load at all
  still switches those options on. See "The header stamp is not a layout check".
- **Consequence for the scoreboard:** the missing Q4_K_S-format row stays blocked
  until a plain (non-UD) Q4_K_S file is on disk. That is the format the retained
  2026-08-16/17 evidence describes. Supporting `UD-Q4_K_S` is kernel work, tracked
  as the campaign below.

## How this was measured

```bash
.venv/bin/python scripts/gguf_quant_route_audit.py \
  /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  /models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf \
  /models/gguf/Qwen3.8-27B-UD-Q4_K_S.gguf \
  --json /tmp/ud-quant-audit.json
```

[`scripts/gguf_quant_route_audit.py`](../scripts/gguf_quant_route_audit.py) reads
the GGUF metadata and tensor table, validates the tensor map against the
dense-Qwen model plugin, and then runs every tensor through
`plan_qwen35_gguf_weight_spec` — the same planner the loader uses — with the T16
switches set from each backend's declared package capabilities. It prints, per
format, how many tensors get a quantized kernel, how many get converted to dense
BF16, and how many make the planner refuse.

Verification state: reads metadata and tensor descriptors only. No tensor bytes
are read, no model is loaded, no device memory is allocated, and no GPU is
touched. It reuses the repo's own decoders, skipping only the final byte-range
size check that `hipengine.loading.gguf.scan_gguf` performs, which is what makes
it usable on a partially downloaded file.

## What the files contain

Three Qwen3.8-27B files, same architecture, same tensor names, same shapes. All
three pass the dense-Qwen plugin's tensor-map validation identically
(851 expected tensors present, 0 missing, 0 unexpected, 0 shape errors). What
differs is only the format assigned to each tensor.

| | `Qwen3.8-27B-Q4_K_M` | `UD-Q4_K_M` | `UD-Q4_K_S` |
| --- | --- | --- | --- |
| Header format claim | `MOSTLY_Q4_K_M` | `MOSTLY_Q4_K_M` | `MOSTLY_Q4_K_S` |
| Distinct weight formats (excluding F32) | 4 | 8 | 11 |
| Stored weight bytes | 15.92 GiB | 15.32 GiB | 14.29 GiB |
| Format mix, excluding 360–456 F32 structural tensors | Q4_K x294, Q6_K x67, Q5_K x48, Q8_0 x1 | Q5_K x131, IQ4_XS x117, Q8_0 x106, Q4_K x104, Q6_K x30, Q3_K x7, IQ4_NL x7, IQ3_S x4 | IQ4_XS x172, Q8_0 x99, Q4_K x95, Q5_K x80, Q6_K x18, IQ3_S x15, Q3_K x13, IQ4_NL x7, IQ3_XXS x5, IQ2_XS x1, IQ2_S x1 |
| Tensors converted to dense BF16 at load | 0 | 277 | 270 |
| Tensors the dense loader refuses | 0 | 18 (Q3_K, IQ4_NL, IQ3_S) | 41 (Q3_K, IQ4_NL, IQ3_S, IQ3_XXS, IQ2_S) |

The last two rows are planner verdicts per tensor. In a real run, one refused
tensor aborts the whole load before any conversion happens, so the conversion
counts describe what the file would need, not what currently lands on a device.

Two files are compared tensor family by tensor family below. Shapes are in this
repository's output-by-input order, and `n` is the number of layers carrying that
tensor: 65 layers in total, of which 17 use full attention and 48 use the recurrent
Gated Delta Network path (named `ssm_*` in the checkpoint). Both files are the same
65 layers with the same shapes; only the assigned formats differ.

| Family | Shape | n | `Q4_K_M` | `UD-Q4_K_S` |
| --- | --- | ---: | --- | --- |
| attention query | 12288x5120 | 17 | Q4_K | IQ4_XS x11, Q4_K x4, IQ4_NL, Q6_K |
| attention key | 1024x5120 | 17 | Q4_K | Q5_K x9, Q6_K x4, Q4_K x3, Q8_0 |
| attention value | 1024x5120 | 17 | Q6_K x9, Q4_K x8 | Q5_K x9, Q6_K x5, Q8_0 x2, Q4_K |
| attention output | 5120x6144 | 17 | Q4_K | Q5_K x11, Q4_K x2, IQ4_XS x2, IQ4_NL, Q6_K |
| recurrent query-key-value | 10240x5120 | 48 | Q6_K x24, Q4_K x24 | Q4_K x25, IQ4_XS x21, IQ4_NL x2 |
| recurrent gate | 6144x5120 | 48 | Q4_K | IQ4_XS x25, Q4_K x20, Q5_K x3 |
| recurrent output | 5120x6144 | 48 | Q5_K | Q5_K x20, IQ4_XS x15, Q4_K x11, Q6_K, IQ4_NL |
| recurrent alpha / beta | 48x5120 | 48 | F32 | Q8_0 |
| feed-forward gate | 17408x5120 | 65 | Q4_K | IQ4_XS x35, Q4_K x7, Q5_K x7, Q3_K x6, IQ3_S x5, IQ3_XXS x3, IQ2_XS, Q6_K |
| feed-forward up | 17408x5120 | 65 | Q4_K | IQ4_XS x34, Q4_K x10, Q5_K x7, Q3_K x5, IQ3_S x4, Q6_K x2, IQ2_S, IQ3_XXS, IQ4_NL |
| feed-forward down | 5120x17408 | 65 | Q6_K x33, Q4_K x32 | IQ4_XS x29, Q5_K x14, Q4_K x12, IQ3_S x6, IQ4_NL, Q3_K, IQ3_XXS, Q6_K |
| token embedding | 248320x5120 | 1 | Q4_K | Q3_K |
| output projection | 248320x5120 | 1 | Q6_K | Q6_K |
| MTP projection (`nextn.eh_proj`) | 5120x10240 | 1 | Q8_0 | Q6_K |
| norms, conv, routing scalars | various | — | F32 | F32 |

Two things fall out of this table beyond the format count. The token embedding
moves from Q4_K to Q3_K, which is the single largest tensor in the model and a
format the dense route cannot decode at all. And the tensors that stay loadable
still change owner: recurrent alpha and beta go from F32 to Q8_0, and the MTP
projection goes from Q8_0 to Q6_K, so a UD file is not "the same weights with a
few layers squeezed" — it re-points paths keyed on specific formats.

## What the dense route can run today

Implemented on both gfx1100 (Radeon Pro W7900) and gfx1151 (Strix Halo):

| Format | Dense read path |
| --- | --- |
| Q4_K | direct quantized kernel, including the T16 fused split-K variant, plus the raw-block fallback kernel |
| Q5_K | T16 fused split-K variant only |
| Q6_K | T16 fused split-K variant on the planar Q6 layout only |
| Q8_0 | T16 variant, and the raw-block path used when decode-time repacking is off |
| F32 | kept as F32 where the layer reads it (router weights), or converted to BF16 |
| BF16 | dense BF16 kernel |

Unreachable on the dense route, split by *why*. Read these two tables against the
per-type coverage table in [`QUANTS.md`](QUANTS.md): that table's word "native" can
mean a kernel for the selected-expert (mixture-of-experts) route, which is a wider
set than the dense route's, so a format can be listed there as native and still
refuse a dense load. What differs is the consumer, not the quantizer.

These five formats have no dense decode path at all, so any tensor using them stops
the load:

| Format | Count in `UD-Q4_K_S` | Count in `UD-Q4_K_M` |
| --- | ---: | ---: |
| Q3_K | 13 | 7 |
| IQ3_S | 15 | 4 |
| IQ4_NL | 7 | 7 |
| IQ3_XXS | 5 | 0 |
| IQ2_S | 1 | 0 |

These formats decode in software only, so they load by being converted to dense
BF16, which is what moves the memory in the table below:

| Format | Count in `UD-Q4_K_S` | Why it is converted instead of read as quantized |
| --- | ---: | --- |
| IQ4_XS | 172 | both backends list it among raw GGUF formats they can store, but the dense materializer excludes it from its own type set, with an in-code note that the dense-27B consumer has to exist first |
| IQ2_XS | 1 | same exclusion |
| Q5_K | 80 | no raw-block fallback kernel, and the fast T16 variant needs a repacked block — see the repack rule below. 20 of them, the recurrent-output tensors, qualify for T16 once repacking is available |
| Q6_K | 17 of 18 | same reason; one tensor reaches a raw-block path today, and repacking would move it and six others onto the planar-Q6 path |

The repack rule is the second reason formats that *do* have kernels still get
converted: the loader turns decode-time repacking off for the whole file when the
file carries raw IQ2_XS, IQ3_XXS, or IQ4_XS residents, because repacking those
layouts is not implemented. Q4_K and Q8_0 keep their quantized paths under that
rule because they have raw-block fallback kernels; Q5_K and Q6_K do not, so
switching repack back on (which only becomes possible once IQ blocks can be
repacked) recovers 26 of these tensors.

Converting to BF16 costs a lot of memory, because it keeps 2 bytes per weight
instead of the compressed block — 4.25 bits per weight for IQ4_XS, 5.5 for Q5_K,
6.6 for Q6_K:

| File | Stored weights | Tensors converted | Resident as compressed | Resident after conversion | Total resident |
| --- | --- | --- | --- | --- | --- |
| `Qwen3.8-27B-Q4_K_M` | 15.92 GiB | 0 | — | — | 15.92 GiB |
| `UD-Q4_K_M` (if the refusals were tolerated) | 15.32 GiB | 277 (9.65 GiB) | 9.65 GiB | 31.53 GiB | 37.20 GiB |
| `UD-Q4_K_S` (if the refusals were tolerated) | 14.29 GiB | 270 (8.67 GiB) | 8.67 GiB | 30.14 GiB | 35.76 GiB |

Targets: 32 GiB of VRAM on the W7900 and 39 GiB usable on this Strix Halo. So
neither UD layout fits, and it fits worse than the plain `Q4_K_M` file it is
supposed to improve on. This is why the existing expansion fallback was not
widened.

## Why the 35B mixture-of-experts route handles these files

Two structural differences, not two different quantizers:

1. **The experts go through a different kernel.** The routed expert matrices use
   GGML-block experts, and that path accepts Q4_K, Q5_K, Q6_K, Q8_0, Q3_K, IQ4_XS,
   BF16 and F32, reading the compressed blocks directly. The dense route's matrix
   kernels are a separate, narrower set.
2. **The mixed formats are pre-decided by a plugin.** The 35B `UD-Q3_K_M` route is
   a registered model plugin (`gguf_ud_q3_k_m`) with one fixed resident layout
   built for that file, and the fused path that reads it — see
   [`GGUF-Q3-OPT.md`](GGUF-Q3-OPT.md). It is not a runtime "accept any format per
   tensor" mechanism.

A useful accident to note: the 35B UD file currently on disk,
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, does not contain any IQ or Q3_K tensors at all —
its mix is F32 x368, Q8_0 x259, Q4_K x82, Q5_K x38, Q6_K x4, BF16 x2. So it is a
"UD" file that runs here only because its chosen mix happens to sit inside the
formats we implemented. That is the whole difference between a UD file that works
and one that does not, and the stamped family name in the filename predicts none of
it.

Dense 27B has neither of the two structural advantages: no expert path to absorb
the odd formats, and no per-layout plugin. Supporting one UD dense layout therefore
means either implementing the missing formats as kernels or registering a
layout-specific plugin, and both are kernel-level work.

## The header stamp is not a layout check

Two switches read the GGUF header's claimed format (`general.file_type`) rather
than the file's actual format mix:

- `GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES` is `{mostly_q4_k_s}` on gfx1151.
- `GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES` is `("MOSTLY_Q4_K_S",)` on gfx1151.

Qwen3.8-27B `UD-Q4_K_S` reports `MOSTLY_Q4_K_S` in its header, so both switches
turn on for it — even though 41 of its tensors cannot be loaded at all. A plain
`Q4_K_S` file and a UD file with the same stamp get the same switches today.
Anything that decides behavior from the stamp should check the actual format
histogram instead; that belongs in the campaign as a correctness item, not a
cleanup detail.

## Scope if we execute this campaign

Ordered so that the cheapest useful subset lands first. Each phase needs the
repo's standard evidence: a numerical RED contract before implementation, the CPU
reference correctness gate with a registered strict fallback, the execution-profile
gates in [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md), and
`rocprofv3 --kernel-trace` proof that the intended kernel actually ran.
Throughput claims follow [`BENCHMARK.md`](BENCHMARK.md) and are recorded per
machine — the W7900 and Strix Halo results are separate lanes, never merged into
one old-to-new comparison.

**Phase 0 — pick the target layout, do not chase arbitrary mixes.** The number of
new kernels needed is set by how many distinct formats a file uses, so a layout
with few formats is worth more than a layout with good theoretical bits-per-weight.
See "External reference" below: a third-party Strix Halo study on this same model
converged on two formats for the whole model (IQ4_XS for all calibrated matrices,
Q8_0 for the uncalibrated ones). If we target that shape, one new format buys a
complete model.

**Phase 1 — IQ4_XS as a dense quantized format.** Highest value: it is the most
frequent non-standard format in both UD files studied (117 and 172 tensors), it is
what the external study chose, and it needs one decoder plus one repack policy.
Scope is smaller than "write a new quantized GEMM": the block decode already runs
on device in the selected-expert family (`gguf_iq4_xs_selected_*` and
`gguf_iq3_xxs_selected_*` kernels, at fixed expert widths such as K=1024 and
K=3072/N=1024/E=256). What is missing is a dense consumer: a resident layout the
fused T16 split-K path reads for projection-width matrices, plus decode-time
repacking for raw IQ blocks, which is also what currently forces the whole file off
the fast path.
The measured payoff, in resident bytes for `UD-Q4_K_S` assuming the refusals are
also resolved, is the reason to order the phases this way:

| Step | Tensors converted to BF16 (stored -> resident) | Total resident weights | Against plain `Q4_K_M` |
| --- | --- | ---: | ---: |
| Nothing implemented, if refusals were tolerated | 270 (8.67 -> 30.14 GiB) | 35.76 GiB | +19.84 GiB |
| Repack for raw IQ blocks only | 244 (8.18 -> 28.75 GiB) | 34.86 GiB | +18.94 GiB |
| Repack plus IQ4_XS as a dense read path | 80 (2.32 -> 6.68 GiB) | 18.65 GiB | +2.73 GiB |

IQ4_XS is what makes the layout affordable; repack alone is under a gigabyte. The
remaining 80 converted tensors are Q5_K and Q6_K in slots the T16 variants do not
accept, and the 41 refused tensors still need Phase 2. Acceptance for Phase 1: a
file in the two-format shape below must load with zero refusals, hold resident
weight bytes at or below its stored bytes, and pass the profile gates against the
CPU reference.

**Phase 2 — Q3_K and the IQ3 family.** Needed only if we want the published UD
mixes as shipped rather than a chosen layout: these are the five formats that
refuse the load above. They are not one job, because the decode side differs:
a Q3_K GEMV builder already exists in the tree (`build_gguf_q3_k_gemv`) and
[`QUANTS.md`](QUANTS.md) records Q3_K as native on the selected-expert route, so
Q3_K needs a dense consumer; `IQ4_NL` needs little beyond Phase 1, because the
`IQ4_NL` codebook values are already embedded in the IQ4_XS implementation, which
[`QUANTS.md`](QUANTS.md) notes is not itself an `IQ4_NL` kernel; `IQ3_S` and
`IQ2_S` are layout-only today, so those two need a real decoder written and gated.
Each format needs its own RED contract; codebook formats are not a copy of the
K-quant code.

**Phase 3 — replace stamp-based switches with layout detection.** Read the actual
format histogram before deciding FP16-recurrent-state and Q4_K_S scope behavior.
Independent of Phases 1–2 and useful even if this campaign stops early.

**Out of scope unless evidence changes:** IQ2_S/IQ2_XS on the dense route (one
tensor each in these files, and a two-bit format needs its own quality case);
per-tensor kernel dispatch for every published UD variant; widening the
convert-to-BF16 fallback, which the table above disqualifies on memory.

## External reference: `pwilkin/strix-halo`

Reviewed 2026-09-06 at the user's request, to check whether its IQ quantization
work helps this campaign. What it is: an installer plus a GitHub Pages guide for
running Qwen3.8-27B on AMD Strix Halo (`gfx1151`), on a custom ROCr/HIP runtime,
with llama.cpp as the engine.

**There is no quantization kernel to port from it.** The repository holds
`install.sh`, the site files, and `data/benchmarks.json` /
`data/reproduction-results.json`. The engine is
[`pwilkin/llama.cpp` branch `strix-halo`](https://github.com/pwilkin/llama.cpp/tree/strix-halo),
whose own README and model card describe its custom pieces as a unified-memory
scheduler ring, a ROCm top-k implementation merged from upstream llama.cpp pull
request 28313, and a "retained PM4" command-list path in a
[forked ROCm](https://github.com/pwilkin/rocm-systems/tree/ilintar-experiments).
A GitHub commit search for `IQ4_XS` in that fork returns zero commits, so its
IQ4_XS mat-vec path is upstream llama.cpp's, not theirs. That search is a weak
signal by itself, because it reads commit messages and may only cover the default
branch. Confidence: high for the zero-result query; medium for the attribution of
their custom work, which rests on their own README and model card rather than a
diff I read — the fork's full diff was not reviewed.

**What is useful to us is the methodology and the calibration evidence, not code:**

1. **A deliberately narrow format mix.** Their main model,
   `Qwen3.8-27B-IQ4_XS-ALL-IMATRIX-Q8-OUT-MTP.gguf`, is 496 IQ4_XS + 10 Q8_0 + 360
   F32 tensors: IQ4_XS for every matrix covered by the published importance matrix,
   Q8_0 for `output.weight`, `token_embd.weight` and the eight block-64/MTP
   matrices the matrix does not calibrate, F32 for one-dimensional and
   convolutional structural tensors — and an explicit tensor map specifically to
   avoid the Q5_K substitutions that llama.cpp's ordinary `MOSTLY_IQ4_XS` preset
   would introduce. That is the Phase 0 argument in measured form: two formats for
   a whole 27B model, which is one new dense kernel instead of six.
2. **Quality cost of that choice on this exact model.** They report perplexity
   `15.3977 ± 0.73357` against `15.1721 ± 0.72292` for the source BF16 GGUF on 16
   held-out 512-token chunks, and note it is a small local evaluation rather than a
   quality benchmark. Useful as a prior for a Phase 1 acceptance target, not as
   proof.
3. **A quantized draft was faster than a bigger draft with the same output.** Their
   IQ4_XS DFlash2 draft beat the Q8_0 draft by +1.79% to +5.67% across prose,
   reasoning and JSON workloads at draft widths 3 and 6, while the deterministic
   target-output hash matched on every workload. That supports keeping our own
   speculative drafts small, which is a separate interest from this campaign.
4. **External performance anchors for gfx1151 on this model** — different engine,
   different quantization, different protocol, so these are reference points and not
   comparisons with our rows, per the evidence policy in
   [`AGENTS.md`](../AGENTS.md): no speculative decoding, `tg128` 14.0976 tok/s;
   IQ4_XS target plus IQ4_XS draft at width 6, 25.666 (prose) / 39.143 (reasoning)
   / 58.529 (JSON) tok/s; and a matched 31,497-token prompt plus 256 generated
   tokens run giving 256.838 prompt tok/s and 26.256 decode tok/s on one Radeon
   8060S. They also attribute about +3.2% decode to their retained command-list path
   over ordinary submission, which is runtime work rather than quantization work.

**Not portable:** the quantization speed itself (it is upstream llama.cpp), the
runtime changes (a forked ROCr/HIP stack, whereas our PM4 submission work lives in
this repository's runtime path), and the scheduler ring and top-k changes (llama.cpp
internals with no counterpart here). For the IQ4_XS block layout and codebook math
that Phase 1 needs, read upstream llama.cpp's quantized format definitions rather
than this fork.

Third-party figures in this section are as they published them, unverified by us.

## Decision carried forward

- Dense 27B `Q4_K_S` evidence stays blocked on a plain `Qwen3.8-27B-Q4_K_S.gguf`
  (llama.cpp's own quantizer, four formats, no importance-matrix formats), which is
  the layout the retained 2026-08-16/17 rows describe. What was on disk at review
  time was an incomplete download of the UD variant: its tensor table was complete
  and readable, but its tensor data was not, and its format mix could not have
  loaded either way.
- UD-format support on the dense route is a campaign, not a flag: Phase 0 decides
  whether one new format (IQ4_XS) is enough, and Phases 1–3 follow only if that
  decision says yes.

The counts in this document are a function of the backends' current capability
flags and the materializer's current type sets, so re-run the audit command above
rather than trusting these tables after any change to
`hipengine/kernels/hip_gfx11*/__init__.py` or
`hipengine/loading/qwen35_gguf_materialize.py`.
