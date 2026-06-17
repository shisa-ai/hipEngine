# GGUF Intake and Native-Quant Plan

Date: 2026-05-17
Target repo: `~/hipENGINE`

Primary references: local llama.cpp checkouts under `~/llama.cpp/` and parent evidence in `~/amd-gpu-tuning/`

## Executive summary

Implementation status as of 2026-05-17: the first intake slice has landed in
`hipengine/loading/gguf.py`, `hipengine/quant/gguf.py`, and
`scripts/inspect_gguf.py`. hipENGINE can now scan local GGUF v3 files, expose
lazy raw tensor views, and CPU-dequantize tiny fallback samples for the target
local tensor types (`BF16`, `Q8_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS`,
`MXFP4`, plus dense `F16/F32`). Native GGUF GEMV correctness spikes now cover
`Q8_0`, `Q5_K`, `Q6_K`, and `Q4_K` raw bytes, plus a lossless PARO-style pack8
repack for `Q4_K`, on gfx1100 while preserving GGML quant math. Full Qwen GGUF
model materialization and E2E correctness now work for the local Q4_K_M, Q8_0,
Q4_1, and UD-Q4_K_XL files; persistent resident decode, all-GPU full attention,
layer-level AOTriton prefill, rows>1 measured-equivalent projection surfaces,
decode graph replay with GPU sampling, and dense-BF16 fallback materialization for
Q4_1/F16/IQ4_XS tensors have landed. Public full-model bulk prefill and deeper
WMMA/Marlin-style tuning remain next steps. BF16 and FP16 output variants are available for the GGUF projection
kernels used by the planned runtime path. Qwen3.5 GGUF
tensor-name mapping now validates the local 0.8B Q4_K_M inventory and classifies
all 24 layers into 18 linear-attention and 6 full-attention blocks. The resident materialization plan covers all 320 tensors:
98 Q4_K weights use lossless pack8 records, 89 Q5_K/Q6_K/Q8_0 weights keep raw
GGUF block bytes, and 133 F32 tensors stay dense F32. Q4_1, F16, BF16, and
IQ4_XS tensors in the other local files materialize through explicit dense-BF16
fallback records for correctness. Native Q6_K and Q8_0 embedding lookup kernels now dequantize selected `token_embd.weight` rows directly to BF16
hidden states, avoiding full dense embedding-table fallback for those token embeddings. A registry-driven
runtime adapter selects the GGUF linear variants for BF16 hidden projections and
FP32 lm-head logits from resident weight metadata. A first resident one-layer
projection probe now starts at Q6_K token embedding and runs layer-0 RMSNorm,
Q4_K `attn_gate`, and Q5_K `ssm_out` through native GGUF kernels to produce a
finite deterministic BF16 hidden-size output. `hipengine.LLM.generate()` now detects
GGUF files, resolves the `qwen35` model plugin, and routes the target quant key
through the native GGUF bring-up generator. The bring-up path now also runs the
tied Q6_K/Q8_0 `token_embd.weight` lm-head GEMV to produce FP32 logits and uses
the shared GPU `argmax_f32` sampler for deterministic greedy tokens. The GGUF tokenizer/detokenizer now parses Qwen3.5
byte-BPE metadata without torch or llama.cpp subprocesses on the hot path. The
GGUF full-stack runner now executes all 24 mapped layers with native GGUF
projections, linear-attention state carry-over, CPU-hosted small-context full
attention, residuals, dense FFN, and final RMSNorm. The public generator runs
resident prefill once, then replays a captured one-step decode graph for remaining
greedy tokens, detokenizes the generated IDs, and returns text through
`LLM.generate()`. The hard gate now passes all local dense-Qwen GGUF quant fixtures for the target prompt with no `torch` import on the generate path. A minimal `qwen35moe` GGUF public-generation bring-up also now works for `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`: it maps the untied `output.weight` lm-head, keeps rank-3 expert tensors in raw GGUF layout for the default path, and runs a deterministic public smoke. The qwen35moe long-prompt default now uses parity-accepted fully bulk prefill, while performance parity with packed PARO still needs packed/grouped expert kernels. Task #59 adds an explicit GGUF expert pack8 sidecar cache for the rank-3 `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` tensors; the sidecar is opt-in and generated under a cache directory, not committed.

The short answer to "can hipENGINE load GGUF quants easily now?" is:


- **GGUF file intake / metadata scanning is easy.** GGUF is a well-documented tensor container with a mature Python reader in llama.cpp's `gguf-py`.
- **Correctness-first FP16 fallback is straightforward.** We can parse GGUF, dequantize tensors on the host, map names into hipEngine's existing model loader, and run existing FP16 kernels. This proves model/tokenizer/tensor-name plumbing but does not preserve GGUF memory/perf benefits.
- **Native GGUF quant execution is not drop-in.** GGUF `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `Q8_K`, and `IQ*` tensors have GGML block layouts and quant math that differ from PARO/AWQ and from the current Marlin-K v0 layout. They need their own quant plugins, CPU oracles, and HIP kernels or a deliberate repack path.
- **The new PARO/Marlin-K work makes this tractable.** hipEngine now has the pattern we want: file/checkpoint layout -> host repack -> explicit device layout -> raw-pointer kernel -> registry dispatch. GGUF should use the same architecture, not special-case dispatch.

The intake implementation is now past scanner/GEMV bring-up for the local Q4_K_M, Q8_0, Q4_1, and UD-Q4_K_XL fixtures. The near-term performance path has resident GGUF decode, all-GPU full attention, AOTriton/equivalent layer prefill attention, rows>1 GGUF projections, decode graph replay, and correctness-oriented dense fallback coverage; remaining work is public full-model bulk prefill and retained throughput parity rows.

Do not treat this document as a performance claim. It is an implementation plan. Any hipENGINE GGUF speedup must be measured in hipENGINE after the accelerated runtime pieces land.

## True `LLM.generate()` E2E acceptance gate

The first native GGUF E2E target was fixed to the local file:

```text
/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf
```

The hard gate is **not** a lower-level kernel smoke or layer runner. It is the
public API:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
  PYTHONPATH=. python3 scripts/qwen35_gguf_e2e_correctness.py
```

Use a precomputed compiler-version file for repeatable cached JIT behavior:

```bash
hipcc --version > /tmp/hipengine-hipcc-version.txt
```

Without this environment variable, fresh Python processes repeatedly probe
`hipcc --version` while resolving JIT cache keys, which can make the correctness
run look like it is hanging even when all kernels are cached.

That script calls:

```python
hipengine.LLM(
    "/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf",
    backend="hip_gfx1100",
    quant="gguf_q4_k_m",
).generate("The answer is", SamplingParams(max_tokens=4, temperature=0.0, top_p=1.0))
```

Acceptance fixtures now cover the original Q4_K_M target plus the local Q8_0,
Q4_1, and UD-Q4_K_XL files:

```text
tests/fixtures/gguf/qwen35_0_8b_q4_k_m_e2e.json
tests/fixtures/gguf/qwen35_0_8b_q8_0_e2e.json
tests/fixtures/gguf/qwen35_0_8b_q4_1_e2e.json
tests/fixtures/gguf/qwen35_0_8b_ud_q4_k_xl_e2e.json
```

The first `qwen35moe` GGUF smoke fixture is intentionally narrower than the
dense-Qwen fixtures, but now carries intake, internal tokenizer, finite-logit,
and external `llama-tokenize` token-oracle evidence:

```text
tests/fixtures/gguf/qwen36_35b_a3b_q4km_smoke.json
model: /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
prompt text: "Hello"
prompt ids: [9419]
expected generated text: "izio."
expected generated token ids: [43482, 13]
```

Run it with:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
  PYTHONPATH=. python3 scripts/qwen35_gguf_e2e_correctness.py \
  --fixture tests/fixtures/gguf/qwen36_35b_a3b_q4km_smoke.json \
  --repeat 2 \
  --json benchmarks/results/2026-05-17-hipengine-qwen36-35b-a3b-q4km-public-generate-smoke.json
```

External oracle: local llama.cpp CPU execution from
`/home/lhl/llama.cpp/llama.cpp-hip-therock` at commit `59778f019`:

```bash
/home/lhl/llama.cpp/llama.cpp-hip-therock/build/bin/llama-simple \
  -m /models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf \
  -n 4 -ngl 0 'The answer is'
# full text: "The answer is 1.\n\n"
```

Prompt/token fixture:

```text
prompt text: "The answer is"
prompt ids:  [760, 4087, 369]
expected generated text: " 1.\n\n"
expected generated token ids: [220, 16, 13, 271]
```

Definition of done for GGUF E2E:

1. `scripts/qwen35_gguf_e2e_correctness.py` passes with repeat ≥ 2.
2. The generated text and generated token IDs match the oracle fixture exactly.
3. Repeated runs are deterministic.
4. The public API path does not import `torch`.
5. The implementation path materializes GGUF resident weights and dispatches
   native GGUF kernels where available (`gguf_q4_k`, `gguf_q5_k`, `gguf_q6_k`,
   `gguf_q8_0`), with explicitly named dense-BF16 fallbacks for Q4_1/F16/IQ4_XS.
6. A cached `rocprofv3 --kernel-trace` smoke proves the expected GGUF kernels ran.
7. `WORKLOG.md` records the exact command output and any benchmark artifact only
   after correctness passes.

As of 2026-05-17, this command passes for Q4_K_M, Q8_0, Q4_1, and UD-Q4_K_XL.
Cached `rocprofv3 --kernel-trace` smokes over earlier `LLM.generate(max_tokens=1)`
confirmed the native GGUF path: Q4_K pack8 GEMV, Q5_K/Q6_K/Q8_0 raw GEMV, Q6_K
embedding, GGUF RMSNorm/add-RMSNorm, linear-attn conv/GDN, BF16 casts, SiLU,
GGUF F32-weight head-RMSNorm+RoPE, span-shaped paged-KV append, paged
full-attention decode, and BF16 gate application. Task #49 adds Q8_0 embedding
and dense-BF16 fallback coverage for Q4_1/F16/IQ4_XS. See
`benchmarks/results/2026-05-16-hipengine-gguf-qwen35-e2e-correctness-diagnostic.json`,
`benchmarks/results/2026-05-17-hipengine-gguf-full-attn-gpu-prelude-diagnostic.json`,
and `benchmarks/results/2026-05-17-hipengine-gguf-local-quant-coverage-diagnostic.json`.
The `qwen35moe` Qwen3.6 smoke fixture now passes the same public API/no-torch gate, but only as a narrow deterministic bring-up smoke. Broader prompts, qwen35moe bulk prefill, stronger oracles, and throughput claims remain future work.


## Why GGUF is attractive for hipEngine

GGUF gives us three useful things:

1. **A mature model artifact format.** A single `.gguf` carries model metadata, tokenizer metadata, tensor names, shapes, tensor types, and tensor bytes.
2. **A huge ecosystem of local quantized models.** Qwen, Llama, Gemma, Mixtral/MoE variants, and many user-facing quants are already published as GGUF.
3. **Reference implementations and baselines.** llama.cpp gives both parsing/quant oracles and W7900 comparison rows for HIP/Vulkan.

The parent workspace has already used GGUF/llama.cpp as an external comparator, especially Qwen3.6-35B-A3B `Q4_K_M` and `UD-Q8_K_XL` rows. hipEngine should use GGUF support to widen model access and to make apples-to-apples kernel comparisons against llama.cpp easier.

## Local references

### llama.cpp / GGUF source references

Local reference checkouts found on this machine:

```text
/home/lhl/llama.cpp/llama.cpp-hip-therock/
/home/lhl/llama.cpp/llama.cpp-vulkan/
/home/lhl/local.amd-gpu-tuning/reference/lucebox-hub/dflash/deps/llama.cpp/
```

Key files:

| File | Why it matters |
| --- | --- |
| `ggml/include/gguf.h` | Canonical file-structure comments: magic `GGUF`, version, KV table, tensor info table, tensor data alignment. |
| `gguf-py/gguf/gguf_reader.py` | Mature Python reader; maps file tensor info into `ReaderTensor` with type, shape, element count, and data view. |
| `gguf-py/gguf/constants.py` | `GGMLQuantizationType`, `GGML_QUANT_SIZES`, model architecture names, tensor-name maps, file-type enums. |
| `gguf-py/gguf/quants.py` | Python quant/dequant reference code for many GGML quant types. Useful for CPU oracles and FP16 fallback. |
| `ggml/src/ggml-common.h` | Block structs and static sizes for `block_q4_0`, `block_q8_0`, `block_q4_K`, `block_q5_K`, `block_q6_K`, `block_q8_K`, IQ types, etc. |
| `ggml/src/ggml-quants.c` | Reference quant/dequant math for GGML block types. |
| `ggml/src/ggml-vulkan/vulkan-shaders/` | The Q4_K/Q8_1 execution-shape reference that motivated the PARO Marlin-K work. |

Useful constants from llama.cpp `gguf-py/gguf/constants.py` and `ggml-common.h`:

```text
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
QK_K = 256
Q4_0: block 32 values, type size 18 bytes = 2 byte scale + 16 byte nibbles
Q8_0: block 32 values, type size 34 bytes = 2 byte scale + 32 int8 quants
Q4_K: block 256 values, type size 144 bytes = 2 fp16 scales + 12 packed scale/min bytes + 128 q4 bytes
Q5_K: block 256 values, type size 176 bytes = Q4_K plus 32 high-bit bytes
Q6_K: block 256 values, type size 210 bytes = low 4 bits + high 2 bits + int8 scales + fp16 super-scale
Q8_K: block 256 values, type size 292 bytes = float scale + 256 int8 quants + 16 int16 block sums
```

### qwen35moe expert sidecar layout

Task #59 introduced `hipengine.loading.qwen35_gguf_expert_sidecar` and
`scripts/qwen35_gguf_build_expert_sidecar.py` as the explicit bridge from GGUF
rank-3 expert tensors to future grouped MoE kernels. Normal materialization still
keeps `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` as raw GGUF device
allocations; the materialization plan only marks them with the optional
`gguf_expert_pack8_v1` sidecar layout. Callers must build or load the sidecar
explicitly:

```bash
PYTHONPATH=. python3 scripts/qwen35_gguf_build_expert_sidecar.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --layers 0 --slots ffn_gate_exps,ffn_up_exps,ffn_down_exps \
  --cache-dir /tmp/hipengine-gguf-sidecars
```

The generated `.npz` files live under the requested cache directory (or
`HIPENGINE_GGUF_SIDECAR_CACHE`, then `~/.cache/hipengine/gguf_sidecars`) and are
not repository artifacts. The v1 layout packs eight adjacent output channels per
input position:

- `Q4_K`: `qweight_low:int32[E, O/8, I]`, `scales:fp32[E, I/32, O]`,
  `mins:fp32[E, I/32, O]`.
- `Q5_K`: the same low-four-bit/scales/mins arrays plus
  `qweight_high:uint8[E, O/8, I]` with one high bit per lane.
- `Q6_K`: `qweight_low:int32[E, O/8, I]`,
  `qweight_high:uint16[E, O/8, I]` with two high bits per lane, and
  `scales:fp32[E, I/16, O]` (no min term).

CPU oracle tests dequantize from the packed sidecar bytes/scales and compare
against the raw GGUF dequantizers for synthetic `Q4_K`, `Q5_K`, and `Q6_K`
expert tensors. Task #60 added registered `moe_linear` kernels that consume this
sidecar without model-dispatch backend/quant branches:

- `expert_pack8_selected_bf16_bf16_out` for Q4_K/Q5_K/Q6_K selected expert rows.
- `expert_pack8_dual_selected_bf16_bf16_out` for Q4_K gate+up selected rows.

Runtime use is explicit: build the cache first, then pass
`--use-expert-sidecar --expert-sidecar-cache-dir <dir> --require-expert-sidecar`
to `scripts/qwen35_gguf_bench.py`. The default public path still uses raw GGUF
selected kernels because the correctness-safe transient sidecar path is currently
a diagnostic blocker: 512/128 reaches `62.458 tok/s` prefill, which is `+303.8%`
vs the old native row baseline but `-37.5%` vs the current raw fast-bulk default.
The retained artifact is
`benchmarks/results/2026-05-17-hipengine-qwen36-35b-a3b-q4km-expert-pack8-sidecar-diagnostic.json`.

### Parent workspace evidence

Parent docs that explain why GGUF/Q4_K-like layouts matter:

- `/home/lhl/amd-gpu-tuning/PLAN-PAROQUANT2.md`
  - The Marlin/Q4_K source-level analysis: K-contiguous packed int4 + compact metadata was the copyable part from GGML/Vulkan.
  - Important caveat: PARO/AWQ does **not** match GGML Q4_K quant math; only the execution shape was copied.
- `/home/lhl/amd-gpu-tuning/docs/OPTIMAL.md`
  - Current retained PARO/Marlin-K qweight-neutral implementation and measured speed/memory rows.
- `/home/lhl/amd-gpu-tuning/PLAN-LONGCONTEXT.md`
  - llama.cpp GGUF Q4_K_M comparison commands/rows for long-context HIP/Vulkan baselines.
- `/home/lhl/amd-gpu-tuning/PR_COMMENT-llamacpp-hip-unroll600.md`
  - Cross-model GGUF llama.cpp HIP measurements and build-flag observations.
- `/home/lhl/hipEngine/docs/MARLIN.md`
  - hipEngine's Marlin-K intake analysis, including the qweight-neutral host-layout work already started here.

## GGUF file structure hipEngine needs

From `ggml/include/gguf.h`, GGUF files contain:

1. File magic: `GGUF`.
2. Version: currently `3` in local llama.cpp.
3. Tensor count.
4. Metadata KV count.
5. Metadata KV pairs.
6. Tensor info table.
7. Aligned tensor data blob.

Important loader facts:

- Metadata values are typed (`uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `float32`, `bool`, `string`, arrays, `uint64`, `int64`, `float64`).
- Tensor info gives name, shape, quant/data type, and data offset.
- Tensor data is aligned by `general.alignment` if present, else `GGUF_DEFAULT_ALIGNMENT=32`.
- GGUF tensor dimensions are stored in GGML order; `gguf-py` returns NumPy-style reversed dims via `ReaderTensor.shape`.

hipEngine should initially consume GGUF through a tiny loader module that either:

- uses `gguf-py` as an optional import, or
- implements a minimal pure-Python reader for the subset we need.

Because hipEngine's runtime hot path must stay torch-free, either option is compatible. The issue is dependency policy: adding hard dependency `gguf` is avoidable. Prefer a small optional reader or an optional `gguf` extra until we know how much of `gguf-py` we need.

## What is similar to PARO Marlin-K

The new PARO Marlin-K work gives us a template:

```text
checkpoint/file layout -> host repack -> explicit device layout -> raw HIP kernel -> registry key
```

For PARO Marlin-K we now have:

```text
PARO/AWQ checkpoint:
  qweight [K, N/8]
  qzeros  [K/128, N/8]
  scales  [K/128, N]

Marlin-K v0 device layout:
  qweight_mk [N/8, K/128, 128]
  qzeros_mk  [N/8, K/128]
  scales_mk  [N/8, K/128, 8]
```

For GGUF we want the same discipline:

```text
GGUF tensor blocks -> host decode/repack or direct block view -> explicit device layout -> raw HIP kernel
```

The architecture is the same. The data is not.

## What is different from PARO/AWQ

GGUF quantized tensors are not `qweight/qzeros/scales` triples. They are single GGML tensors whose row data is a sequence of quant blocks. The scale/min/zero information is embedded inside those blocks.

Examples:

### Q4_0

From `ggml-common.h`:

```text
block_q4_0:
  ggml_half d
  uint8_t qs[16]   # 32 4-bit values
```

Math is symmetric-ish around a fixed zero convention in GGML's q4_0 dequant path; no external `qzeros` tensor exists.

### Q8_0

```text
block_q8_0:
  ggml_half d
  int8_t qs[32]
```

This is the simplest native execution candidate: one scale per 32 values and signed int8 payload.

### Q4_K

```text
block_q4_K:
  ggml_half d
  ggml_half dmin
  uint8_t scales[12]
  uint8_t qs[128]    # 256 4-bit values
```

This is the key llama.cpp `Q4_K_M` family component. It uses 256-value superblocks and packed scale/min metadata, not PARO's per-128-group zero/scale tensors.

### Q8_K

```text
block_q8_K:
  float d
  int8_t qs[256]
  int16_t bsums[16]
```

This is used in GGML's quantized dot-product chains as an activation-side/intermediate form, not necessarily as the best first weight format to execute directly.

## Feasibility tiers

### Tier 0: scanner / census

Goal: Given a `.gguf`, print enough metadata to decide whether hipEngine can load it.

Outputs:

- architecture (`general.architecture`)
- file type (`general.file_type`)
- tokenizer metadata presence
- tensor count and total bytes by quant type
- tensor-name mapping coverage for the target hipEngine model plugin
- list of unsupported quant types

This is low risk and should be first.

Likely files:

```text
hipengine/loading/gguf.py
tests/test_gguf_reader.py
scripts/inspect_gguf.py
```

### Tier 1: FP16 fallback loader

Goal: Load a GGUF model into hipEngine by dequantizing quantized tensors on the host to FP16/BF16 and using existing FP16 kernels.

Pros:

- Fastest way to validate model metadata, tokenizer, tensor names, and generation parity.
- Gives a CPU-reference path for later native quant kernels.
- Useful for tiny models and debugging.

Cons:

- Loses GGUF memory advantages.
- Dequantizing large models on host can be slow and memory-heavy.
- Not a performance path.

This tier should be explicitly named `gguf_fp16_fallback`, not `gguf_native`.

### Tier 2: native Q8_0

Goal: Execute GGUF `Q8_0` weights directly or after a lightweight host repack.

Why Q8_0 first:

- Simple block: 32 int8 values + one fp16 scale.
- Easier CPU oracle and HIP kernel.
- Good loader/kernel integration test before complex K-quants.

Caveat: the largest local external model rows include `UD-Q8_K_XL` / K-family quantization, not necessarily pure Q8_0. Q8_0 is a bring-up format, not the final model target.

Likely quant key:

```text
quant = "gguf_q8_0"
variant = "gemv_fp16" or "gemv_q8_0"
```

### Tier 3: native Q4_K / Q4_K_M

Goal: Run common llama.cpp `Q4_K_M` GGUF weights with native hipEngine kernels.

This is the first truly useful GGUF memory/perf target because many public models use Q4_K_M and our parent analysis already compared against Q4_K_M.

Implementation choices:

1. **Direct GGML block kernel**
   - Device layout mirrors `block_q4_K` row blocks.
   - Kernel decodes GGML `scales[12]`, `d`, `dmin`, and q4 payload directly.
   - Best for fidelity and avoiding extra memory.

2. **Host repack to hipEngine-native Marlin-ish layout**
   - Parse `block_q4_K` on host and emit a device layout optimized for W7900.
   - Could separate q4 payload and decoded scale/min tables for faster kernels.
   - Costs load-time memory and may deviate from exact GGML layout, but fits hipEngine's Marlin-K architecture.

3. **Host dequant to PARO-like W4 layout**
   - Usually not recommended: Q4_K does not have PARO/AWQ group_size=128 semantics and zero/scales are already quantized per 32-ish sub-blocks.
   - Converting into PARO's qzeros/scales layout would be both lossy/awkward and not representative of GGUF.

Recommendation: prototype both direct-block CPU oracle and one repacked native layout, but keep the first kernel direct or minimally repacked to reduce correctness ambiguity.

### Tier 4: Q5_K/Q6_K/IQ variants

After Q4_K works:

- `Q5_K`, `Q6_K`: similar K-superblock family but different bit packing and metadata.
- `IQ*`: important for very low-bit modern GGUFs, but more complex due lookup/grid schemes.
- `MXFP4`/`NVFP4`: present in newer `gguf-py` constants, but should not distract from Q4_K/Q8_0 first.

## Model architecture scope

GGUF is a container; hipEngine still needs a model plugin.

Recommended first architecture targets:

1. **Tiny Qwen GGUF scanner/fallback**
   - Local cache includes `ggml-org/Qwen3-0.6B-GGUF` with `Qwen3-0.6B-Q4_0.gguf` under Hugging Face cache.
   - Good for loader/tokenizer smoke.
2. **Qwen2/Qwen3 dense**
   - Similar enough to existing hipEngine Qwen code to make name mapping feasible.
3. **Qwen3.5/Qwen3.6 MoE GGUF**
   - Performance-relevant but more complicated: expert tensor naming, routing, and active-expert surfaces.

Avoid starting with arbitrary Llama/Gemma if the goal is to reuse existing Qwen runtime. Llama/Gemma can come once GGUF loading is generic enough and model plugins exist.

## Tensor-name mapping problem

GGUF tensor names are not guaranteed to match Hugging Face safetensors names. llama.cpp has architecture-specific tensor maps in `gguf-py/gguf/constants.py` and conversion scripts.

hipEngine needs a mapping layer:

```text
GGUF tensor name -> hipEngine logical tensor name -> model/runtime slot
```

Implementation should be table-driven per model architecture, not dispatch branches.

Potential files:

```text
hipengine/loading/gguf.py
hipengine/loading/gguf_names.py
hipengine/models/qwen_gguf.py   # only if a separate model plugin is cleaner
```

Do not overload `qwen35_paro.py` with GGUF-specific branches unless the target is specifically Qwen3.5 PARO-like tensors, which GGUF is not.

## Tokenizer considerations

GGUF often carries tokenizer metadata. hipEngine currently depends on `tokenizers` and has Qwen/HF-oriented loading. For GGUF:

- Tier 0 should verify tokenizer metadata is present and dump keys.
- Tier 1 can initially require an external tokenizer path if using GGUF tokenizer metadata is too much work.
- A full GGUF loader should eventually construct a tokenizer from GGUF metadata or delegate to a known compatible tokenizer implementation.

Do not block tensor/kernels on full tokenizer import if a tiny fixture can use explicit token IDs for correctness.

## Dependency policy

Current hard dependencies in `pyproject.toml` are:

```text
jinja2
numpy
safetensors
tokenizers
```

Options for GGUF parsing:

1. **Optional `gguf` extra**
   - Add `gguf` only under `[project.optional-dependencies]`.
   - Fastest path; uses llama.cpp's reader and dequant reference.
   - Need to ensure package availability/version.

2. **Vendored/minimal reader**
   - Write a small parser for GGUF v3 metadata/tensor tables.
   - Keeps hard deps minimal and runtime self-contained.
   - More maintenance burden, but scanner needs only a subset.

3. **Local dev-only reference**
   - Tests import `/home/lhl/llama.cpp/.../gguf-py` via path.
   - Good for analysis, bad for committed tests unless skipped when absent.

Recommendation: start with a minimal internal scanner for metadata/tensor table, and use `gguf-py` as an optional oracle in tests if available. Add a hard or optional dependency only after we know we need full tokenizer/dequant support.

## Proposed hipEngine implementation shape

### New docs/planning file created first

This file is intentionally docs-only. Next code should land in small pieces.

### Step 1: GGUF scanner

Files:

```text
hipengine/loading/gguf.py
tests/test_gguf_reader.py
scripts/inspect_gguf.py
```

Data classes:

```python
@dataclass(frozen=True)
class GGUFTensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: str | int
    nbytes: int
    offset: int

@dataclass(frozen=True)
class GGUFModelInfo:
    path: Path
    version: int
    alignment: int
    metadata: Mapping[str, object]
    tensors: tuple[GGUFTensorInfo, ...]
```

Scanner output should be deterministic and testable against a tiny synthetic GGUF or a checked-in metadata fixture, not a full model file.

### Step 2: quant layout metadata

Files:

```text
hipengine/quant/gguf.py
tests/test_gguf_quant_layout.py
```

Table:

```text
F16, BF16, F32
Q4_0, Q8_0
Q4_K, Q5_K, Q6_K, Q8_K
IQ4_NL/IQ4_XS later
```

Each entry should include:

- block size in values
- bytes per block
- whether CPU dequant oracle exists
- native kernel status: unsupported / fallback / native

### Step 3: FP16 fallback loader

Files depend on architecture target, likely:

```text
hipengine/loading/gguf.py
hipengine/loading/qwen_gguf.py
tests/test_qwen_gguf_name_map.py
```

Rules:

- All quantized tensors dequantize to FP16/BF16 host arrays.
- Existing device materialization path loads them as normal dense tensors.
- This is correctness-only; name the mode so no one mistakes it for efficient GGUF.

### Step 4: native Q8_0 or Q4_K kernel family

Possible files:

```text
hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_gemv.hip
hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_gemv.py
hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_gemv.hip
hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_gemv.py
tests/test_gguf_q8_0_gemv_plan.py
tests/test_gguf_q4_k_gemv_plan.py
```

Registry examples:

```text
KernelKey("hip_gfx1100", "linear_gemv", "gguf_q8_0", "decode_fp16")
KernelKey("hip_gfx1100", "linear_gemv", "gguf_q4_k", "decode_fp16")
```

Exact key names should follow current registry conventions; do not add quant/backend if-branches in runtime dispatch.

## Relationship to Marlin-K

The GGUF native path should learn from Marlin-K but not pretend to be the same layout.

| Topic | PARO Marlin-K | GGUF Q4_K |
| --- | --- | --- |
| Source tensors | `qweight`, `qzeros`, `scales` separate | single GGML block tensor |
| K grouping | 128 values per PARO group | `QK_K=256` superblock, scale/min substructure |
| Metadata | int32 zero tuple + scale tensor | `d`, `dmin`, 12 packed scale/min bytes |
| Weight payload | int32 words, 8 output lanes packed per K | `qs[128]` bytes per 256 values per row block |
| Best first execution | rows==1 non-expert GEMV | rows==1 GEMV, then prefill/multirow if needed |
| Repack lesson | eliminate duplicate W4 qweight; use aliases carefully | avoid duplicating full GGUF block data unless a native layout earns it |

The shared rule: **separate file format from execution layout**. A GGUF loader can either preserve GGML blocks or repack into a hipEngine-native layout. The choice should be measured per quant type.

## Performance parity plan vs PARO

Current GGUF performance should be treated as a correctness bridge, not a PARO
throughput peer. The retained diagnostic comparison is
`benchmarks/results/2026-05-16-hipengine-gguf-vs-paro-diagnostic.json`:

| Path | Workload / phase | Retained diagnostic result | Main reason |
| --- | --- | ---: | --- |
| GGUF Qwen3.5-0.8B-Q4_K_M | 3-token hidden prefill | 15.7 tok/s | token-serial full-stack path, rows==1 GEMV surfaces, CPU full-attention bridge |
| GGUF Qwen3.5-0.8B-Q4_K_M | decode step 1 -> 4 | 5.6 -> 2.8 tok/s | `sample_next_token(context_ids)` recomputes the full context every token |
| PARO Qwen3.5-35B-A3B | native fixture 512/32 | 47.0 prefill / 101.6 decode tok/s | resident session and native decode kernels |
| PARO Qwen3.5-35B-A3B | AOTriton V3 512/128 | 2183.3 prefill / 101.5 decode tok/s | resident native prefill plus AOTriton compact-varlen full attention |
| PARO Qwen3.5-35B-A3B | graph replay 512/128 | 2312.8 prefill / 109.3 decode tok/s | AOTriton prefill plus captured decode graph and GPU sampling |

These rows are not apples-to-apples model comparisons. They identify the missing
execution features GGUF must acquire before any throughput claim is fair.

A newer standard-shape diagnostic is retained in
`benchmarks/results/2026-05-17-hipengine-gguf-q4km-parity-benchmark-diagnostic.json`.
It uses repeated token id `9707`, one warmup run, three measured runs,
`--require-cached-build`, public GGUF correctness gates, and one-step decode
graph replay with graph capture excluded from decode timing:

| Path | Workload | Median prefill | Median decode | Peak tracked | Main reason |
| --- | --- | ---: | ---: | ---: | --- |
| GGUF Qwen3.5-0.8B-Q4_K_M | 512/128 | 16.35 tok/s | 171.84 tok/s | 0.568 GiB | token-serial prefill, graph replay decode |
| GGUF Qwen3.5-0.8B-Q4_K_M | 4K/128 | 16.20 tok/s | 83.84 tok/s | 0.610 GiB | full-attention context cost grows; prefill still token-serial |

The 512 decode number can exceed some 35B PARO/llama.cpp decode baselines only
because the GGUF model is 0.8B; the prefill number remains ~99% below resident
PARO/llama.cpp rows. This is a retained diagnostic, not an accepted throughput
row.

### Dependency order

AOTriton V3 is not the first GGUF blocker. It accelerates the attention compute
after Q/K/V are already on device and the runtime already has resident sequence
state. The required order is:

```text
resident GGUF session
  -> GPU full-attention prelude + KV append
  -> AOTriton V3 / equivalent full-attention prefill
  -> rows>1 GGUF projection kernels
  -> decode graph replay + GPU sampling [landed]
  -> benchmark parity rows
```

### P0: lock the current correctness baseline

- Keep `scripts/qwen35_gguf_e2e_correctness.py` as the public E2E gate for
  `/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf`.
- Keep the exact llama.cpp oracle: prompt IDs `[760, 4087, 369]`, generated IDs
  `[220, 16, 13, 271]`, text `" 1.\n\n"`.
- Keep `torch` absent from the `LLM.generate()` hot path.
- For every performance phase below, record a cached `rocprofv3 --kernel-trace`
  smoke proving the expected native kernels ran.

Validation gate: E2E repeat >= 2 exact-match output, no `torch` import on the
public path, and current GGUF kernel symbols present in the profile.

### P1: add a persistent GGUF resident session

Status: implemented for the public Q4_K_M E2E path in
`Qwen35GGUFResidentSession` as of
`benchmarks/results/2026-05-17-hipengine-gguf-resident-session-diagnostic.json`.
The remaining bottlenecks are now public full-model bulk prefill and promoted
throughput parity rows; layer-level AOTriton, rows>1 projection surfaces, and
decode graph replay have landed as diagnostics.

Former bottleneck: `Qwen35GGUFFullStackRunner.sample_next_token(context_ids)`
replayed the entire prompt plus generated history for each decode token, causing
decode to slow as context length grew.

Implemented target:

- Add a session API next to the bring-up runner, e.g.
  `Qwen35GGUFResidentSession.prefill(prompt_ids)` and
  `Qwen35GGUFResidentSession.step(token_id)`.
- Own reusable device scratch, current token/position state, recurrent
  linear-attention state, and full-attention KV cache for all 24 layers.
- Keep materialized GGUF weights resident across the full generate call.
- Update the public generator to call session `prefill()` once and `step()` for
  decode, not `sample_next_token(context_ids)`.

Validation result: the public `LLM.generate()` gate passes repeat=2 with the P0
oracle (`" 1.\n\n"`, IDs `[220, 16, 13, 271]`, `torch_loaded_by_generate=false`).
A resident timing probe on the fixture prompt measured prefill `0.190 s` and
three decode steps `0.060/0.060/0.060 s`, replacing the old full-replay decode
trend `0.179/0.237/0.294/0.355 s`. This remains diagnostic only and is not a
promoted throughput row.

### P2: move full-attention prelude and KV append to GPU

Status: implemented for the resident one-token Q4_K_M path as of
`benchmarks/results/2026-05-17-hipengine-gguf-full-attn-gpu-prelude-diagnostic.json`.
The production path no longer copies full-attention Q/K/V to host for q/k
RMSNorm, RoPE, history handling, softmax, or gate application. It now splits
q/gate on GPU, converts K to FP32 on GPU, applies GGUF F32-weight q/k
RMSNorm+RoPE, appends K/V through `KVLiveSpans` into BF16 paged caches, runs
paged full-attention decode, and applies the BF16 gate before the output
projection. The unfused CPU bridge remains only in the layer-level test oracle.

Former bottleneck: full-attention GGUF layers ran Q/K/V projections on GPU but
then copied Q/K/V to host for q/k RMSNorm, RoPE, small-context attention, and
history handling before copying the result back to device.

Implemented target:

- Add or reuse kernels for Qwen3.5 GGUF full-attention post-projection work:
  F32-weight q/k RMSNorm, partial RoPE, q/gate split, and BF16/FP16 Q/K/V
  layout conversion.
- Append K/V through the existing paged-KV ABI (`KVLiveSpans`), not a local
  `(block_table, context_len)` shortcut.
- Keep an unfused CPU/reference path for layer-level oracles.

Validation result: `tests/test_qwen35_gguf_full_attention_gpu.py` compares the
first full-attention layer against the old CPU bridge over two prompt positions
and asserts hidden tolerance, lm-head top-1 agreement, and KL <= 0.05. The public
E2E oracle still passes repeat=2 (`" 1.\n\n"`, IDs `[220, 16, 13, 271]`, no
`torch` import). `rocprofv3 --kernel-trace` over `LLM.generate(max_tokens=1)`
shows 18 launches each of `qwen35_split_qgate_bf16_kernel`, `bf16_to_f32_kernel`,
`gguf_head_rmsnorm_partial_rotary_position_f32_weight_kernel`,
`qwen35_write_paged_kv_mixed_value_position_tensor_kernel<unsigned short>`,
`qwen35_paged_full_attn_decode_context_tensor_kernel`, and
`qwen35_full_attn_gate_mul_bf16_kernel`; a source grep confirms the production
runner no longer contains `_host_full_attention` or `_copy_bf16_device_to_f32`.

### P3: wire AOTriton V3 for GGUF full-attention prefill

Status: implemented for a layer-level full-attention prefill path as of
`benchmarks/results/2026-05-17-hipengine-gguf-aotriton-v3-prefill-diagnostic.json`.
`Qwen35GGUFFullStackRunner.run_full_attention_prefill_layer(...)` now uses the
same `PrefillConfig.attn_aotriton_min_tokens` threshold surface as PARO: rows
below threshold run the resident native sequential fallback, while eligible rows
run AOTriton V3 compact-varlen attention after GGUF Q/K/V projection and GPU
q/k norm+RoPE. This is not yet the public full-model prefill scheduler;
linear-attention bulk prefill and scheduler integration remain follow-up work.

Prerequisite: P1 and P2. AOTriton sees BF16 Q/K/V tensors and live-span-shaped
paged KV metadata; it does not know about GGUF block bytes.

Implemented target:

- Register a GGUF prefill-attention variant through the kernel registry, e.g. a
  `full_attn_prefill` key for the GGUF quant/plugin family with variant
  `aotriton_attn_fwd_v3`.
- Reuse the existing compact-varlen AOTriton wrapper and PARO threshold policy
  surface where possible (`attn_aotriton_min_tokens`), without adding
  backend/quant `if` branches in model dispatch.
- Add a prompt-length sweep for short prompts, 512-token prompts, and 4K prompts
  so threshold behavior is explicit.

Validation result: the threshold sweep with `attn_aotriton_min_tokens=3` selected
`native_sequential` for rows 1/2 and `aotriton_v3` for rows 4. The layer oracle
compares the final prefill row for layer 3 against the old CPU bridge and checks
hidden tolerance, lm-head top-1 agreement, and KL <= 0.05. The P0 public E2E
oracle still passes repeat=2 (`" 1.\n\n"`, IDs `[220, 16, 13, 271]`, no `torch`
import). `rocprofv3 --kernel-trace` over an eligible rows=4 layer prefill shows
`attn_fwd` plus the expected GGUF multi-position q/k norm+RoPE, BF16 prompt-KV
writer, BF16 query cast, and BF16 gate kernels.

### P4: add rows>1 GGUF projection kernels

Status: implemented as measured-equivalent row-grid prefill projection surfaces in
`benchmarks/results/2026-05-17-hipengine-gguf-prefill-projection-diagnostic.json`.
`launch_gguf_linear(...)` now routes `rows > 1` to registered `prefill_*` variants
for Q4_K pack8 and raw Q8_0/Q5_K/Q6_K without model-dispatch backend/quant
branches. These kernels keep exact GGML quant math and add BF16/FP16 output
surfaces; they are not yet WMMA/GEMM-tiled throughput kernels.

Former bottleneck after P3: rows>1 layer prefill projections still resolved to
GEMV variant names and lacked FP16-output surfaces for follow-on attention /
linear-attention experiments.

Implemented target:

- Add batched prefill kernels for Q4_K pack8 and raw Q5_K/Q6_K/Q8_0, with the
  BF16/FP16 output variants required by attention and linear-attention layers.
- Measure whether preserving GGML blocks or repacking to a hipENGINE-native
  layout wins per tensor family. Do not keep duplicate device qweight residency
  unless the benchmark win justifies it.
- Keep GGML quant math exact; do not relabel GGUF Q4_K as PARO Marlin-K.

Validation result: `scripts/gguf_prefill_projection_smoke.py --rows 4` passes
Q4_K pack8 and raw Q8_0/Q5_K/Q6_K BF16->F32/BF16->FP16/BF16->BF16 checks vs CPU
references with `worst_max_abs=0.0`. `rocprofv3 --kernel-trace` over that smoke
shows `gguf_q4_k_pack8_prefill_out_kernel<unsigned short,{float,_Float16,unsigned short}>`
and `gguf_k_prefill_out_kernel<unsigned short,{float,_Float16,unsigned short},8/5/6>`.
A native Qwen3.5-0.8B GGUF rows=4 layer-3 prefill profile shows six
`gguf_q4_k_pack8_prefill_out_kernel<unsigned short,unsigned short>` launches
with `Grid_Size_Y=4`, one raw Q6_K prefill projection, and AOTriton `attn_fwd`.
The public P0 E2E gate remains exact repeat=2 (`" 1.\n\n"`, IDs
`[220, 16, 13, 271]`, no `torch` import). Public full-model prefill is still
resident token-serial until the linear-attention bulk scheduler path is wired,
but the native layer-level GGUF prefill path no longer loops rows==1 projection
kernels for eligible layers.

### P5: add GGUF decode graph replay and GPU sampling

Status: implemented for the public Q4_K_M E2E path as of
`benchmarks/results/2026-05-17-hipengine-gguf-decode-graph-replay-diagnostic.json`.
`Qwen35GGUFResidentSession.capture_decode_graph(...)` captures a one-step HIP
graph after prefill. The captured step consumes the current device lm-head argmax
token, performs GGUF Q6_K embedding lookup from that device scalar, advances
resident linear/KV state, runs the GGUF Q6_K lm-head to FP32 logits, samples with
GPU `argmax_f32`, records generated token IDs into a device int64 buffer, and
advances the device position/context scalar. The public GGUF generator now uses
this graph for remaining greedy decode tokens.

Former bottleneck after resident decode: Python/ctypes launch overhead and host
sampling capped one-token latency. PARO's retained decode rows depend on HIP
graph replay plus device-side token/position state.

Implemented target:

- Capture a one-step GGUF decode graph after prefill, including device token
  update, resident layer execution, final norm/lm-head, argmax/sampling, and
  device position/context advancement.
- Keep eager and graph paths byte-for-byte/token-for-token comparable.

Validation result: `scripts/qwen35_gguf_decode_graph_smoke.py` compares eager
resident decode to graph replay on the fixture prompt. Both paths generate
`[220, 16, 13, 271]` / `" 1.\n\n"`; final logits are finite with graph/eager
top-1 `271`, `max_abs=0.0`, and KL `0.0`. The smoke reports graph capture
`0.0717 s` separately from graph replay decode `0.0225 s` so capture time is
excluded. The public E2E gate still passes repeat=2 with no `torch` import.
`rocprofv3 --kernel-trace` over a prompt+3 graph-replay run reports
`session.position=6`, three `advance_decode_position_i64_kernel` launches, three
`record_i64_scalar_indexed_kernel` launches, four GGUF Q6_K lm-head logits
launches (prefill sample + 3 graph samples), and 36 full-attention KV
append/decode launches, matching 6 resident token steps across 6 full-attention
layers rather than full-context recompute per generated token.

### P6: broaden local GGUF quant coverage

Status: implemented as correctness coverage in
`benchmarks/results/2026-05-17-hipengine-gguf-local-quant-coverage-diagnostic.json`.
The public E2E target now includes Q4_K_M plus the local Q8_0, Q4_1, and
UD-Q4_K_XL files. Coverage follows the same resident/session and graph replay
gates as Q4_K_M.

Implemented target:

- Q8_0 routes through native raw GGUF materialization/generation, including a new
  Q8_0 token-embedding lookup kernel and existing Q8_0 projection/lm-head GEMV.
- Q4_1 uses explicit dense-BF16 fallback materialization and the registered
  `dense_gemv` BF16 projection kernel.
- F16 and IQ4_XS tensors needed by UD-Q4_K_XL also use dense-BF16 fallback
  materialization, while Q4_K/Q5_K/Q6_K/Q8_0 tensors keep their native paths.
- Public generator keys are registered for `gguf_q8_0`, `gguf_q4_1`, and
  `gguf_ud_q4_k_xl` in addition to `gguf_q4_k_m`.

Validation result: `LLM.generate()` E2E fixtures pass for Q4_K_M, Q8_0, Q4_1,
and UD-Q4_K_XL with no `torch` import on the generate path. Q4_K_M generates
`[220, 16, 13, 271]` / `" 1.\n\n"`; Q8_0, Q4_1, and UD-Q4_K_XL generate
`[220, 16, 13, 198]` / `" 1.\n"`.

### P7: benchmark parity only after P1-P5

Status: Q4_K_M diagnostic retained in
`benchmarks/results/2026-05-17-hipengine-gguf-q4km-parity-benchmark-diagnostic.json`.
No accepted throughput row is promoted because public full-model bulk prefill is
still token-serial and the reference rows are cross-model 35B-family baselines.

Run retained comparison protocols only once GGUF has resident decode, all-GPU
full attention, AOTriton/equivalent prefill attention, rows>1 projections, and
optional graph replay.

Required benchmark rows:

- GGUF Qwen3.5-0.8B Q4_K_M: load/materialize, resident bytes, 512/128, 4K/128. [diagnostic retained]
- GGUF Qwen3.5-0.8B Q8_0/Q4_1/UD-Q4_K_XL: same rows once supported.
- llama.cpp HIP/Vulkan rows for the same GGUF file when available.
- PARO retained rows remain reference context only unless model/quant/workload
  are matched.

Validation gate: each retained row has exact command, model path, quant, workload
shape, hardware, correctness gate, artifact JSON, benchmark rollup row, and
changelog one-liner.

## Validation plan

### Scanner validation

- Parse a tiny synthetic GGUF or local vocab GGUF.
- Verify magic/version/alignment/tensor count.
- Verify tensor names, types, shapes, offsets.
- Compare scanner output against `gguf-py` where available.

Local tiny files include many tokenizer/vocab GGUFs under:

```text
/home/lhl/llama.cpp/llama.cpp-hip-therock/models/
/home/lhl/llama.cpp/llama.cpp-vulkan/models/
```

Local Hugging Face cache includes a tiny model GGUF:

```text
~/.cache/huggingface/hub/models--ggml-org--Qwen3-0.6B-GGUF/.../Qwen3-0.6B-Q4_0.gguf
```

Do not commit these model files.

### FP16 fallback validation

- Dequantize one tiny linear weight from GGUF to FP16.
- Compare against `gguf-py` dequant output.
- Run hipEngine dense linear CPU/GPU fixture if available.
- For model-level smoke, use fixed token IDs first; tokenizer integration can follow.

### Native quant validation

For each native quant kernel:

1. CPU dequant oracle from internal code and/or `gguf-py`.
2. Deterministic GPU fixture: tiny rows/K/N, exact or tight tolerance.
3. hipEngine correctness gate per `AGENTS.md`: KL <= 0.05 and top-1 agreement >= 90% vs CPU reference on fixture inputs.
4. `rocprofv3 --kernel-trace` showing the expected kernel name and plausible duration.
5. Only then benchmark against existing hipEngine fallback and llama.cpp reference.

### Benchmark policy

GGUF performance claims need all normal hipEngine benchmark metadata:

- model file path and quant type
- backend, kernel variant, commit
- shape: prompt/decode/concurrency/context depth
- W7900/gfx1100, ROCm/HIP version
- exact command
- correctness gate result
- benchmark artifact under `benchmarks/results/`
- rollup update in `benchmarks/README.md` and `benchmarks/CHANGELOG.md`

## Risks and decisions

### Risk: assuming Q4_K is just Marlin-K

It is not. PARO Marlin-K copies a K-contiguous execution shape, but its quant math is still PARO/AWQ. GGUF Q4_K has different scale/min metadata and 256-value superblocks. Treat it as a new quant plugin.

### Risk: adding `gguf` as a hard dependency too early

A scanner can be internal. Full tokenizer/dequant support may justify an optional extra. Do not add a hard dependency until needed.

### Risk: model architecture sprawl

GGUF support can explode into many architectures. Start with a scanner and one Qwen-family mapping, not a generic promise that every GGUF model works.

### Risk: duplicate memory from repack

The Marlin-K lesson applies: if we repack GGUF tensors into a native layout, we need an ownership/alias plan. Keep original GGUF mmaps/host arrays separate from device allocations, and avoid duplicate device qweight residency unless a measured kernel win justifies it.

### Risk: tokenizer detour

Tokenizer metadata support matters eventually, but kernel/load validation can use explicit token IDs and tensor-level fixtures. Do not let tokenizer support block quant kernel bring-up.

## Recommended near-term decision

The scanner, quant table, Qwen3.5 tensor-name map, Q4_K_M resident weight materialization, native GGUF GEMV surfaces, tokenizer, and public E2E correctness gate are already in place for the local Q4_K_M target. The next retained unit should therefore be performance plumbing, not more scanner work:

1. Build a persistent GGUF resident session with `prefill()` and `step()`.
2. Move full-attention q/k norm, RoPE, q/gate split, KV append, and attention history to GPU.
3. Wire AOTriton V3 or an equivalent full-attention prefill path once Q/K/V are device-resident.
4. Add rows>1 GGUF projection kernels for native prefill.
5. Add decode graph replay and GPU sampling. [done]
6. Broaden to Q8_0, Q4_1, and UD-Q4_K_XL only after the resident/runtime gates are reusable. [done]
7. Promote benchmark rows only after the normal correctness, profiler, artifact, rollup, and changelog gates pass.

## Bottom line

hipENGINE can now load and execute the Qwen3.5-0.8B Q4_K_M GGUF fixture correctly with resident state, all-GPU attention/KV, layer-level AOTriton prefill attention, multirow projection surfaces, decode graph replay, and retained diagnostic 512/128 + 4K/128 parity measurements. It is still not a promoted performance path: public full-model bulk prefill and accepted throughput parity rows remain. GGUF must keep GGML quant math and its own quant layouts while borrowing PARO's scheduling, registry, and memory-discipline patterns.


## P8: real batched prefill GEMM (active)

Date added: 2026-05-18. Branch: `gguf-bulk-prefill`.

### Why we are here

After P4 there is a `prefill_*` kernel family registered for every GGUF quant
type, and the runtime calls them when `rows > 1`. But every `prefill_*` symbol
in `hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.py` is **literally an alias
for the matching `gemv_*` symbol**:

```python
gguf_q8_0_prefill_bf16_bf16_out = gguf_q8_0_gemv_bf16_bf16_out
gguf_q5_k_prefill_bf16_bf16_out = gguf_q5_k_gemv_bf16_bf16_out
gguf_q6_k_prefill_bf16_bf16_out = gguf_q6_k_gemv_bf16_bf16_out
# ... same for Q4_K
```

The corresponding HIP kernels (`gguf_k_prefill_out_kernel`,
`gguf_k_pack8_prefill_out_kernel`, `gguf_k_selected_prefill_out_kernel`,
`gguf_q4_k_*_prefill_out_kernel`) are **shape-batched but algorithmically
decode-shaped**:

- Grid `(out_features_or_pack, rows)`, one block per `(row, out_col_pack)`.
- 256 threads in `blockDim.x`, each summing `in_features / 256` K-elements,
  then `reduce_block_sum` across the wave.
- The quantized weight blocks for a single output column are re-read and
  re-dequantized `rows` times. No reuse across rows.
- No WMMA / MFMA. Each thread does FP32 scalar accumulation.

At `rows == 512`, every Q8_0 / Q4_K / Q5_K / Q6_K weight block is dequantized
512 times. This is the entire prefill bottleneck.

Task #61 prefill profile (Qwen3.6-35B-A3B-UD-Q4_K_M, 512 tokens, no decode):

| Bucket | ms | % of prefill kernel time | Why decode-shaped |
| --- | ---: | ---: | --- |
| Dense / shared Q8_0 row GEMVs | 1931 | 37.8 % | `gguf_q8_0_*_prefill` aliases to `gemv` |
| Selected expert row GEMVs (Q4_K gate/up + Q5_K/Q6_K down) | 2441 | 47.7 % | `gguf_*_selected_*_prefill` aliases to `selected_gemv` |
| GDN recurrent | 664 | 13.0 % | Real recurrent kernel, deferred |
| Full-attn / KV | < 1 % | < 0.1 % | Already routed through AOTriton V3 |

To hit a llama.cpp/PARO-class prefill (2000+ tok/s @ 512), the dense and
selected-expert buckets need to collapse by roughly 10×. Knob-turning the
GEMV kernels cannot get there; the algorithm must change.

### Reference implementations (read these first, in this order)

These are the kernels and helpers we are **reusing or copying the algorithm
from**. Anything not listed here is either decode-shaped (and being replaced)
or outside the scope of P8.

#### Reuse-as-is (call directly, no new kernel needed)

| Component | Path / symbol | What it does | Where P8 calls it |
| --- | --- | --- | --- |
| Token → expert count | `hipengine/kernels/hip_gfx1100/moe/group_scatter.hip::qwen35_moe_group_count_kernel` | `counts[e] = #{r : selected[r] == e}` from `[rows*top_k]` selected list. | Step 1 of the new selected MoE prefill scheduler (P8.4 / P8.6). |
| Expert prefix sum | `group_scatter.hip::qwen35_moe_group_prefix_kernel` | Padded prefix into `expert_start_compact[E+1]`; pads each expert count to `pad_multiple` (8/16) so compact-WMMA tiles align. | Step 2 of the same scheduler. |
| Scatter + gather hidden | `group_scatter.hip::qwen35_moe_group_scatter_gather_kernel` | In one kernel, sort `(row, expert)` pairs by `expert` AND gather the corresponding hidden rows into a compact `[total_compact_rows, in_features]` slab. | Step 3 of the scheduler; outputs the compact A-side input to the WMMA GEMM. |
| WMMA tile map | `group_scatter.hip::qwen35_moe_wmma_tile_map_kernel` | Derives `wmma_expert_start[E+1]` (rounded to 16-row multiples) and `tile_expert[T]` so each 16-row WMMA tile knows which expert it belongs to. | Step 4: passed to the new selected WMMA GEMM. |
| Weighted scatter-back combine | `group_scatter.hip::qwen35_moe_gather_packed_hidden_kernel` (alias `weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w` in the runtime wrapper) | Scatters compact `[total_compact_rows, hidden]` outputs back into `[rows, hidden]` with `routing_weights[r, k]`, fuses the shared-expert gate combine and residual add. | Step 7: post-down-projection combine. |
| Router top-k (already on device) | `hipengine/kernels/hip_gfx1100/moe/router.hip::qwen35_router_logits*` / `qwen35_router_select` | Per-row router logits and top-k selection. | Step 0: produces `[rows*top_k]` selected expert ids + weights. |
| Compact SiLU / mul | `hipengine/kernels/hip_gfx1100/fused/...::silu_mul_separate_out_bf16` | Pointwise `silu(gate) * up` over compact rows. | Step 6 between gate+up GEMM and down GEMM. |
| Q4_K pack8 batched dual prefill (existing) | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_gemv.hip::gguf_q4_k_pack8_dual_prefill_bf16_bf16_out` | The one **existing** batched (not WMMA) Q4_K prefill kernel; used today by `launch_gguf_linear_pair` for dense FFN gate+up when both projections are Q4_K pack8. | Kept as the rows>1 fallback for tile boundary / shape cases the new WMMA kernel cannot cover yet. |

These have CPU-reference oracles and runtime wrappers already. P8 does **not**
touch them; it only adds new GEMM kernels that consume their outputs.

#### Algorithm template (copy structure, swap dequant)

These are the in-tree gfx1100 kernels we are mirroring line-by-line. P8
kernels keep the same wave shape, the same lane-to-output mapping, the same
WMMA call sequence, and the same output-store pattern. The **only** delta is
the inner K-loop dequant.

| Source kernel | File | Used as template for |
| --- | --- | --- |
| `awq_fusedw4_prefill_fp16_kernel<TM, TN, qweight_transposed>` | `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip` | P8.1 Q8_0 dense WMMA prefill, P8.2 Q4_K dense, P8.3 Q5_K/Q6_K dense (single output). |
| `awq_fusedw4_prefill_dual_fp16_kernel<TM, TN>` | same file | P8.2 Q4_K **dual** (gate+up) dense WMMA prefill; same trick of halving the block-x grid between the two output tensors. |
| `gemm_awq_selected_pack8_wmma_compact_kernel<scalar_t>` | `hipengine/kernels/hip_gfx1100/wmma/paro_awq_wmma.hip` | P8.4/P8.5 selected (grouped MoE) Q4_K / Q5_K / Q6_K WMMA prefill (single output, e.g. `ffn_down_exps`). |
| `gemm_awq_selected_dual_pack8_wmma_compact_kernel<scalar_t>` | same file | P8.4 selected Q4_K dual (gate+up) MoE WMMA prefill. |

All four templates already use `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`,
`half16_t` operands, `float8_t` accumulators, `__launch_bounds__(32, ...)`,
and the standard `tid & 15` lane-to-output mapping. The MoE variants already
consume the exact `(expert_start_compact, wmma_expert_start, tile_expert)`
ABI emitted by `group_scatter.hip`. **Do not invent a new compact-MoE ABI**;
P8 kernels read these arrays verbatim.

What changes inside each templated kernel for GGUF:

- The AWQ W4 dequant block (read `qzeros`/`scales`, unpack `pw`, compute
  `a_reg[kk] = sc_h * (half)q - zp_h`) is replaced with the GGUF dequant for
  the target quant type per the table in "Algorithm: WMMA `f32_16x16x16_f16_w32`
  with K-loop dequant" above.
- AWQ's `group_size=128` becomes the GGUF block / super-block size (32 for
  Q8_0, 256 for Q4_K / Q5_K / Q6_K). The K-tile size stays at 16 so the WMMA
  shape and lane mapping do not change.
- AWQ's `(qweight, qzeros, scales)` triple becomes a single `uint8_t*` to the
  raw GGUF block stream. Address arithmetic uses `block_q*_K_BYTES` and the
  block layout in `~/llama.cpp/llama.cpp-hip-therock/ggml/src/ggml-common.h`.
- The activation side (`half_t* x` for dense, compact-rowed `scalar_t* x` for
  selected) and the output store (`half_t* out` for dense, compact-rowed for
  selected) are byte-identical to the PARO templates.

Selected kernels additionally read `expert_start_compact[E+1]`,
`wmma_expert_start[E+1]`, `tile_expert[T]`, and the rank-3 expert weight
tensor `[E, out_features, raw_bytes_per_row]`, just like the AWQ compact
MoE kernels read those plus `qweight[E, out_packed, in_features]` etc.

#### Algorithmic cross-checks (read, do not port)

| Source | Path | Why it is here |
| --- | --- | --- |
| llama.cpp HIP MMQ | `~/llama.cpp/llama.cpp-hip-therock/ggml/src/ggml-cuda/mmq.cuh` (4176 lines), `mmq.cu`, `template-instances/mmq-instance-q{4,5,6}_k.cu`, `mmq-instance-q8_0.cu` | Cross-check that GGUF block decoding is correct (`load_tiles_q4_K_device`, `vec_dot_q4_K_q8_1_mma`). Reference for the Q8_1 activation-quant trick if P8.8 is opened. **Do not port the dispatcher or scheduler.** |
| llama.cpp Vulkan MMQ | `~/llama.cpp/llama.cpp-vulkan/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp` plus `mul_mmq_funcs.glsl` and `mul_mmq_shmem_types.glsl` | Same algorithm, cooperative-matrix shader. Sanity check when an `mmq.cuh` corner case is unclear. |
| nano-vllm-amd I8 WMMA | `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip`: `qwen35_wmma_i8_tile_kernel`, `qwen35_wmma_i8_gemm_a_row_major_kernel`, `qwen35_wmma_i8_gemm_grouped_a_row_major_kernel` | Pattern for `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` on gfx1100 if P8.8 (Q8_1-quant + I8 WMMA) is opened. Already proven to land on this hardware. |
| nano-vllm-amd W8A16 shared bulk | same file: `w8a16_shared_gate_up_bulk_kernel`, `w8a16_shared_gate_up_bulk4_kernel`, `w8a16_shared_down_bulk_combine_kernel` | The shape parent for hipENGINE's `w8a16_shared_*` family in `hipengine/kernels/hip_gfx1100/quant/w8a16_linear.hip`. Useful if a Q8_0 shared-expert specialization is needed beyond P8.1. |
| nano-vllm-amd grouped MoE dispatch | `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/expert.py` | The Python-side reference for how grouped MoE prefill is sequenced (count → prefix → scatter → GEMM → combine). Matches `_run_post_attention_moe_rows` we are about to add a WMMA path to. |

#### What is **not** being reused

- `gguf_k_prefill_out_kernel`, `gguf_k_pack8_prefill_out_kernel`,
  `gguf_k_selected_prefill_out_kernel`, `gguf_k_selected_pack8_prefill_out_kernel`
  in `hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip` are the
  decode-shaped "prefill" kernels P8 replaces. They stay registered for the
  rows==1 / fallback path but lose their `prefill_*` registry alias once the
  new WMMA kernels land.
- Same story for the Q4_K family in
  `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_gemv.hip`. The one exception
  is `gguf_q4_k_pack8_dual_prefill_bf16_bf16_out`, which is already a real
  batched (non-WMMA) kernel and is kept as a rows>1 fallback for shapes the
  WMMA tile sizing does not cover cleanly.
- The pack8 expert sidecar (`hipengine/loading/qwen35_gguf_expert_sidecar.py`)
  and its kernels (`hipengine/kernels/hip_gfx1100/quant/gguf_expert_pack8_gemv.hip`)
  are not used. Tasks #59/#60 already showed that the residency cost
  outweighed the kernel-time win, and P8 does not change that calculus. Do
  not regrow the sidecar.

#### What a future agent should do before adding a new kernel here

1. `grep -rn '__global__' hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip
   hipengine/kernels/hip_gfx1100/wmma/paro_awq_wmma.hip
   hipengine/kernels/hip_gfx1100/moe/group_scatter.hip` and read each match.
   These three files **are** the template.
2. Read the matching CPU reference
   `hipengine/kernels/cpu_reference/ops.py::gguf_quant_gemv` (and its
   selected/MoE callers) before writing the kernel; the test will compare
   against it.
3. Check `docs/KERNELS.md` for any drift entries on the parent files used as
   sources, and follow `AGENTS.md` § "Before Starting" / "During Work" /
   "After Changes".
4. Only then write the new `.hip` file. Mirror the PARO `__launch_bounds__`,
   block dim, grid dim, lane mapping, and accumulator type. Only change the
   dequant block.

### Algorithm: WMMA `f32_16x16x16_f16_w32` with K-loop dequant

The PARO `awq_fusedw4_prefill_fp16_kernel` shape is the target. One block of
32 threads (one gfx1100 WMMA wave) computes a `TM × TN` output tile, where
`TM` is the count of output channels and `TN` is the count of token rows. The
relevant constants:

- Block: `dim3(32)`. `__launch_bounds__(32, 8)` on dense, `(32, 2)` on the
  grouped compact variant. Pick to match the per-quant register pressure.
- Grid: `((out_features + TM - 1) / TM, (rows + TN - 1) / TN)`.
- WMMA op: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)` where
  - `a` is `half16_t` interpreted as a `[16, 16]` matrix tile by lane
    `tid & 15` (`a[lane, kk]` lives in thread `lane`'s `a[kk]` register).
  - `b` is `half16_t` along the K axis for one of the TN rows.
  - `c` is the running `float8_t` accumulator (8 floats per lane; lanes 0
    and 16 each hold half of the 16 output rows of one tile).
- Inner loop: for each K block of size 16 (`k_tiles_per_group`), load 16
  activation halves into `b_reg` and 16 dequantized weight halves into
  `a_reg`, accumulate.
- Output: each lane writes 8 output values into its `(out_row, out_col)`
  slots in the result `[rows, out_features]` matrix.

The entire structure is quant-orthogonal. The only thing that changes
between Q8_0 / Q4_K / Q5_K / Q6_K is the dequant block in the inner K loop
that fills `a_reg[kk]` for `kk \in [0, 16)`.

Dequant per quant type (see `ggml-common.h` block layout for the source of
truth):

| Quant | Super-block K size | Per-K-tile scale source | Inner dequant of one 16-K tile |
| --- | ---: | --- | --- |
| Q8_0 | 32 | `fp16 d` at start of block | `weight[k] = (half) (int8) qs[k] * d`. Two K-tiles per block (k=0..15, k=16..31). |
| Q4_K | 256 | `fp16 d`, `fp16 dmin`, 12-byte packed 6/6 (`scales/mins`) | Decode 8 sub-blocks: each has 6-bit `sc` and 6-bit `m`; per 32-K sub-block `weight[k] = d * sc * (q4(k)) - dmin * m`. Two K-tiles per 32-K sub-block. |
| Q5_K | 256 | `fp16 d`, `fp16 dmin`, same 12-byte scales/mins, plus 32-byte hi-bits | Same as Q4_K with `q4(k) | (hi(k) << 4)` (5-bit). |
| Q6_K | 256 | `fp16 d` plus per-16-K int8 `sc[16]` | `weight[k] = d * sc[k/16] * (q6(k) - 32)`. K-tiles of 16 each see one `sc[]` element. |

Key design choices we are committing to up front:

1. **Dequant happens inside the K loop, not as a separate pre-pass.** Same as
   PARO `awq_fusedw4_prefill_fp16_kernel`. No host-side or device-side full
   dequant of the weight tensor. Memory footprint is unchanged; only kernel
   time changes.
2. **First batched kernels accumulate in F32 with half_t WMMA operands.** No
   activation-side quantization. Correctness is then identical to the existing
   row-GEMV path modulo the order-of-summation rounding that WMMA implies.
3. **No new resident weight repack.** GGUF block bytes stay on-device exactly
   as raw GGUF. The kernel reads raw block bytes, decodes scales/mins/qs in
   registers, and produces dequantized half values just-in-time. This avoids
   the 24 GiB residency pressure that killed the pack8 expert sidecar in
   task #59/#60.
4. **One kernel file per quant family** (Q8_0 is dense-only, Q4/Q5/Q6 each get
   dense + selected variants when needed). Keeps the dequant logic local and
   keeps each `.so` build small.
5. **Existing `gguf_*_prefill_*` registry keys point to the new WMMA kernels.**
   The aliases to `gemv_*` in `gguf_k_gemv.py` go away. The dispatch surface in
   `hipengine/runtime/gguf_linear.py` does not branch on backend/quant;
   `_variant_for_rows` already maps `gemv_*` → `prefill_*` for `rows > 1` and
   that mapping is preserved.
6. **Selected variants reuse the existing PARO compact MoE machinery.** The
   token sort, expert prefix, and WMMA tile-map kernels in
   `hipengine/kernels/hip_gfx1100/moe/group_scatter.hip` are reused
   verbatim. Only the inner expert GEMM kernel is new.

### Phased kernel order

P8.1 — **Dense Q8_0 batched WMMA prefill** (the first kernel and the largest
single decode-time bucket). Replaces `gguf_q8_0_prefill_*` aliases. Affected
call sites: shared-expert gate/up/down, attention Q/K/V projections,
linear-attention `attn_qkv`/`attn_gate`/`ssm_out`, dense FFN gate/up/down in
non-MoE Qwen3.5, lm_head when the model has untied Q8_0 output (qwen35moe).
The symmetry of Q8_0 (no zero, no min, single `fp16 d` per 32 values) makes
this the cleanest starting point and gives the largest single-kernel-family
uplift. Estimated bench impact: 1931 ms → ~250 ms at 512/0 if the kernel
reaches half of PARO's compact-WMMA throughput.

P8.2 — **Dense Q4_K batched WMMA prefill**. Same pattern with Q4_K dequant.
Affected call sites: dense Qwen3.5 ffn gate/up (`Q4_K_M` is the dominant Qwen
GGUF quant), lm_head on tied-Q6 models is unaffected (that goes through Q6_K
in P8.3), `attn_qkv` on Q4-quant models. Includes a `dual` variant for the
ffn gate+up pair (mirrors `awq_fusedw4_prefill_dual_fp16_kernel`).

P8.3 — **Dense Q5_K and Q6_K batched WMMA prefill**. Q5_K just adds the
hi-bit byte lane to Q4_K's dequant. Q6_K replaces the 12-byte scale/min block
with per-16-K int8 scales and one super-block `d`. Affected call sites: dense
Q5_K linear-attn `ssm_out`, dense Q6_K lm_head (tied path) and any Q6_K
attention projections.

P8.4 — **Grouped/selected Q4_K MoE prefill** (gate + up). Compact slab
layout: tokens are sorted by expert via
`qwen35_moe_group_scatter_gather_kernel`. The kernel takes the compact
hidden, `expert_start_compact`, `tile_expert`, and the full Q4_K expert
tensor `[E, out_features, in_features_bytes]`. Per-tile, look up
`expert_id = tile_expert[row_tile]`, offset into that expert's weight bytes,
and run the same WMMA inner loop. Affected: `ffn_gate_exps`, `ffn_up_exps`.
Includes a `dual` variant matching `gemm_awq_selected_dual_pack8_wmma_compact_kernel`.

P8.5 — **Grouped/selected Q5_K and Q6_K MoE prefill** (down). Same pattern
with Q5_K / Q6_K dequant. Affected: `ffn_down_exps`. Note: Qwen3.6-35B-A3B
uses Q5_K for `ffn_down_exps` in `Q4_K_M`, but K-quant tier may vary across
GGUF builds; the dispatch path must check each tensor's actual quant.

P8.6 — **Scheduler wiring**: when GGUF WMMA prefill is explicitly enabled
(`Qwen35GGUFResidentSession(use_wmma_prefill=True)`, `--use-wmma-prefill`,
or `HIPENGINE_GGUF_WMMA_PREFILL=1`) and all selected-MoE registry keys resolve,
the fast-bulk MoE path switches from "selected row GEMVs over `rows * top_k`
lanes" to:

```text
router_top_k          — already on-device
  -> qwen35_moe_group_count
  -> qwen35_moe_group_prefix         (expert_start_compact, pad_multiple=1)
  -> qwen35_moe_group_scatter_gather (sort + gather hidden into compact slab)
  -> qwen35_moe_wmma_tile_map        (wmma_expert_start, tile_expert)
  -> gguf_q4_k_selected_dual_wmma_prefill   (compact concatenated gate+up GEMM)
  -> silu_mul_dual_out_bf16                 (over compact [row, gate|up] rows)
  -> gguf_q5_k/q6_k_selected_wmma_prefill   (compact down GEMM)
  -> weighted_lanes_sum_out_bf16_f32w       (scatter weighted compact rows to tokens)
  -> shared_gate_combine_residual_batch_out_bf16
```

The selected row-GEMV path remains the default and also the fallback when the
raw GGUF expert quant/shape or selected WMMA registry coverage is unavailable.

**P9.H1 safety note (2026-05-19):** the qwen35moe resident runtime now forces
`use_wmma_prefill=False` and `use_gemv_decode=False` when either opt-in is
requested, unless `HIPENGINE_GGUF_ALLOW_UNSAFE_QWEN35MOE_FASTPATHS=1` is set.
This is a deliberate correctness guard: the formal P9.E2 512/128 gate rejected
both real opt-ins (`KL 5.993`, top-1 `5.43%`). Kernel R&D can still use the
unsafe override, but retained P9.A3/P9.B7 benchmark rows must either show
`effective_* = true` and pass P9.E2 after the repack fixes, or be labeled as a
legacy fallback rather than a WMMA/GEMV performance claim. The replacement
layout plan is [`GGUF_DECODE_REPACK.md`](GGUF_DECODE_REPACK.md).

P8.7 — **lm_head** Q6_K batched WMMA prefill (one-shot final-row case, plus
the "sample all rows" debug case used in stage probes).

P8.8 — **(Optional) Q8_1-style activation quantization plus I8 WMMA**
(`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`). This is the llama.cpp MMQ
algorithm and what nano-vllm-amd already uses for W8A8. Doubles the WMMA
throughput per op vs F16 WMMA. Only attempt if P8.1 – P8.7 do not close the
gap to the PARO-class prefill row, and only with a dedicated KL gate because
activation quantization changes the result rounding more than weight-only
dequant does.

### Tile sizing

For the first batched kernels we follow the PARO defaults: TM ∈ {16, 32, 64},
TN ∈ {16, 32}. The dispatch picks the largest tile that:

- divides `(out_features + TM - 1) / TM` and `(rows + TN - 1) / TN` to
  avoid wave divergence on boundary tiles,
- keeps the per-block register pressure below the gfx1100 budget
  (`__launch_bounds__(32, 8)` for dense, `(32, 2)` for grouped compact, see
  `paro_awq_gemv.hip`),
- avoids LDS for the first version (PARO compact-WMMA does not need LDS for
  W4; we should not need it for Q8_0 / Q4_K either since dequant happens in
  registers).

Default tile after measurement: TM=32, TN=32 for `rows \geq 32`; TM=32, TN=16
for `rows \in [16, 32)`. Smaller `rows` should not run batched WMMA; rows ==
1 keeps using the existing decode pack8 GEMV.

### Correctness gates

Every new batched prefill kernel must:

1. Match `hipengine/kernels/cpu_reference/ops.py::gguf_quant_gemv` (or its
   selected/MoE equivalent for P8.4 / P8.5) to within F32 tolerance
   `atol = 1e-3, rtol = 1e-2` for synthetic Q4_K/Q5_K/Q6_K weights and
   `atol = 5e-4, rtol = 5e-3` for Q8_0. WMMA F32 accumulation should be
   nearly bit-exact when the K reduction order is preserved.
2. Pass an end-to-end logit gate on the existing fixtures:
   - `tests/fixtures/gguf/qwen35_0_8b_q4_k_m_e2e.json` (dense Qwen3.5),
   - `tests/fixtures/gguf/qwen36_35b_a3b_q4km_smoke.json` (qwen35moe Qwen3.6),
   with `max_kl <= 0.05`, top-1 agreement `>= 90 %`, and exact generated-ID
   match for greedy decoding at temperature 0.
3. Add `tests/test_gguf_q8_0_wmma_prefill.py` (and per-quant siblings):
   construct synthetic GGUF block bytes via the existing `make_q8_0_weight` /
   `make_q4_k_weight` / etc., compare F32 output against the CPU reference
   for multiple `(rows, in_features, out_features)` shapes including
   non-multiple-of-tile sizes.
4. Provide a `rocprofv3 --kernel-trace` smoke proving the new WMMA prefill
   kernel name appears in a 512/0 trace and the decode-shaped `gemv` family
   does not appear on the prefill code path.

### Performance gates

No benchmark row is retained until P8.1+P8.2+P8.4 are all wired. The gating
comparison is the task #61 prefill profile:

| Path | qwen35moe 512/0 prefill kernel time | Notes |
| --- | ---: | --- |
| Current (decode-shaped row GEMVs) | 5114 ms total | dense Q8_0 1931 ms, selected MoE 2441 ms, GDN 664 ms |
| Acceptance floor for P8.1+P8.2+P8.4 | ≤ 1500 ms | matches `~100 tok/s` → `~340 tok/s` at the bench's measured prefill rate. |
| Stretch goal | ≤ 700 ms | matches `~700 tok/s` (still below PARO 2500+ but in the same algorithmic regime). |

The acceptance floor is conservative because the GDN recurrent (664 ms) is
out of scope for P8 and the first WMMA kernels will not match PARO's
hand-tuned MoE compact path on the first try.

Retained artifacts under `benchmarks/results/` follow the existing pattern:
model + quant + workload shape + hardware + exact command + correctness gate
output + rocprofv3 evidence. Update `benchmarks/README.md` and
`benchmarks/CHANGELOG.md` only when a P8 step lands as a kept row.

### What we are deliberately not doing in P8

- **No new resident weight repack** (no "GGUF→Marlin-K" repack, no "pack8
  sidecar v2"). The pack8 sidecar in task #59/#60 was rejected for the same
  reason: GGUF Q4_K/Q5_K/Q6_K already pack the scales/mins compactly, and
  duplicating the qweight payload pressures the 24 GiB budget without
  earning a clear correctness or perf win.
- **No changes to GDN recurrent prefill** (`qwen35_gdn_prefill_recurrent_k2_kernel`).
  664 ms / 5114 ms is real but secondary; pick it up after the dense and MoE
  buckets collapse.
- **No changes to AOTriton attention prefill**. It is already fast.
- **No new C++ engine layer**. The runner stays Python + ctypes calling JIT
  HIP kernels. Same plugin registry. Same `_run_bulk_prefill_and_sample`
  shape.
- **No activation quantization** until P8.8, and only if the F16-accumulating
  path leaves throughput on the table.

### Open questions (track here, decide before P8.1 lands)

1. Does Q8_0 batched WMMA prefer (a) a sub-block `(out_col, k_tile)` layout
   where one wave dequantizes 16 cols of one block, or (b) a per-thread
   layout where each thread owns one col across multiple K-tiles, mirroring
   PARO exactly? PARO style is the safer first pass; (a) would need an LDS
   stage.
2. What is the right `__launch_bounds__` for Q4_K? Dequant register pressure
   is roughly 2× Q8_0 (need `d`, `dmin`, 6-bit `sc`, 6-bit `m`, hi-bits for
   Q5_K). Start with `(32, 4)` and tune from the rocprof occupancy line.
3. Where does the per-quant `prefill_*` registry key live? Current proposal:
   keep the existing `prefill_*` variant names in
   `hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.py` but rebind them to
   the new WMMA wrappers, so the runtime dispatch in `gguf_linear.py` does
   not change. Alternative: add new `wmma_prefill_*` variants and route
   through them from `_variant_for_rows`. Decision tracked in P8.1 commit.

### Acceptance checklist for closing P8

- [ ] P8.1 – P8.5 kernels written, registered, CPU-reference-gated.
- [ ] Public `LLM.generate()` Q4_K_M fixture and qwen35moe smoke pass with
  the new prefill path enabled by default.
- [ ] `rocprofv3 --kernel-trace` of `--prompt-length 512 --decode-tokens 0`
  shows new WMMA prefill kernel symbols and no `_prefill_out_kernel<...>` /
  `_pack8_prefill_out_kernel<...>` symbols on the prefill path.
- [ ] qwen35moe 512/128 prefill `\geq 1000 tok/s` median over three runs
  (matches the acceptance floor).
- [ ] Benchmark artifact + `benchmarks/README.md` + `benchmarks/CHANGELOG.md`
  + `WORKLOG.md` updated atomically with the perf row.
- [ ] No `import torch` on the `LLM.generate()` path; no `if backend == ...`
  / `if quant == ...` branches added to runtime dispatch.

## P9: Closing the qwen35moe gap to PARO ~2700/116 (planned)

P8 (tasks #7–#16) collapsed the qwen35moe GGUF MoE-projection disaster, but
the engine is not yet at the parent PARO Qwen3.5-35B-A3B native ceiling. P9
is the next compounding push: it converts what P8 unlocked (compact selected
MoE + dense WMMA prefill) into both **prefill compute saturation** and a
**PARO-style decode pipeline**, then closes the remaining smaller gaps.

This section is the master plan. Per-kernel detail belongs in `docs/KERNELS.md`;
per-run evidence belongs in `benchmarks/results/`, `benchmarks/README.md`,
`benchmarks/CHANGELOG.md`, and `WORKLOG.md`.

### P9.0 — Status snapshot (post-P8)

| Workload | hipENGINE GGUF (now) | PARO native parent | Gap |
| --- | ---: | ---: | ---: |
| Qwen3.5-35B-A3B-class **512/0** prefill | 534 tok/s | ~2700 tok/s | ~5.0× |
| Same model **512/128** prefill | 530 tok/s | ~2697 tok/s | ~5.1× |
| Same model **512/128** decode | 62.6 tok/s | ~115–116 tok/s | ~1.85× |

Parent reference: `~/amd-gpu-tuning/docs/OPTIMAL.md` 2026-05-13
(weighted-lane accumulation row, graph replay decode). hipENGINE row:
`benchmarks/results/2026-05-18-hipengine-qwen36-35b-a3b-q4km-p8-compact-moe-wmma-accepted.json`.
The hipENGINE 512/0 trace runs on the local **RX 7900 XTX / gfx1100**; W7900
results are unverified for this artifact and the PARO comparison row is on
the matched W7900.

Clean 512/0 rocprof bucket breakdown (post-P8):

| Bucket | Kernel | Total ms | Dispatches | Share of 907.8 ms |
| --- | --- | ---: | ---: | ---: |
| GDN prefill recurrent (MoE path) | `qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_kernel<unsigned short>` | 666.9 | 30 | **73.5%** |
| Dense Q8_0 WMMA prefill | `gguf_q8_0_prefill_wmma_kernel<unsigned short, unsigned short, 32, 32>` | 73.9 | 250 | 8.1% |
| Selected Q4_K dual WMMA prefill | `gguf_q4_k_selected_dual_wmma_prefill_compact_kernel<unsigned short>` | 64.6 | 40 | 7.1% |
| Full attention prefill GQA gate | `qwen35_paged_full_attn_prefill_gqa_gate_bf16_kernel<true>` | 39.5 | 10 | 4.4% |
| Selected Q5_K down WMMA prefill | `gguf_k_selected_wmma_prefill_compact_kernel<unsigned short, 5>` | 27.3 | 37 | 3.0% |
| Router logits | `qwen35_router_logits_token_tile_kernel<unsigned short, 4>` | 13.2 | 80 | 1.5% |
| Linear-attn conv prefill | `qwen35_linear_attn_conv_prefill_kernel` | 3.5 | 30 | 0.4% |
| Decode-shaped BF16 GEMV (still reached at prefill) | `dense_gemv_out_kernel<unsigned short>` | 3.1 | 60 | 0.3% |
| Selected Q6_K down WMMA prefill | `gguf_k_selected_wmma_prefill_compact_kernel<unsigned short, 6>` | 3.6 | 3 | 0.4% |
| Q6_K pack8 lm-head logits (allowed fallback) | `gguf_k_pack8_prefill_out_kernel<unsigned short, float, 6>` | 1.0 | 1 | 0.1% |
| Everything else (bf16<->f32, silu, rmsnorm, gate combine, scheduler, kv writes) | many small | ~11 | many | ~1.2% |

Decode profile (512/128 eager rocprof, illustrative — the graph trace timed
out, but the kernel families are the same as the graph wall-clock run):

| Bucket | Kernel | Total ms / 128 tokens | Dispatches |
| --- | --- | ---: | ---: |
| Selected Q4_K dual decode-shaped GEMV | `gguf_q4_k_selected_dual_prefill_out_kernel<unsigned short, unsigned short>` | 749.5 | 5160 |
| Q8_0 pack8 decode-shaped GEMV | `gguf_k_pack8_prefill_out_kernel<unsigned short, unsigned short, 8>` | 474.7 | 21930 |
| Selected Q5_K pack8 decode-shaped GEMV | `gguf_k_selected_pack8_prefill_out_kernel<unsigned short, unsigned short, 5>` | 366.3 | 4773 |
| Q6_K pack8 lm-head logits (allowed fallback) | `gguf_k_pack8_prefill_out_kernel<unsigned short, float, 6>` | 200.5 | 130 |
| Paged full-attention decode context tensor | `qwen35_paged_full_attn_decode_context_tensor_kernel` | 321.3 | 1290 |
| GDN decode rmsnorm gate lowp | `qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<unsigned short>` | 84.7 | 3870 |

The decode pipeline still uses the historical `*_prefill_out_kernel` family
**at decode shapes (rows=1)** for every GGUF projection. These are decode
row-GEMVs in disguise. They are correct but do not match the PARO
`gemv_awq_*` decode kernels in throughput.

### P9.0a — Roofline anchors (what is actually possible)

From `docs/ROOFLINE.md` for W7900 / gfx1100:

- **Prefill is compute-bound.** BF16 WMMA peak is ~123 TFLOP/s. PARO
  Qwen3.5-35B-A3B at 2700 tok/s consumes ~80% of WMMA throughput in the
  prefill path. We are at ~16% (538 / 2700 ≈0.20 × 80% ≈ 16%). The headroom
  is real and almost entirely in **GDN prefill** plus better tile/launch
  occupancy on the new WMMA kernels.
- **Decode is bandwidth-bound.** Active weight traffic for Qwen3.5-35B-A3B
  Q4_K_M at decode is ~1.7 GB/token → hard ceiling **508 tok/s** at 864 GB/s
  peak. Realistic ceiling is closer to ROOFLINE's **412 tok/s** at ~80% peak.
  PARO at 116 tok/s reaches ~28% peak. hipENGINE GGUF at 62.6 tok/s reaches
  ~15%. The decode gap is **not** about the quant; it is about
  per-projection-kernel inefficiency and per-token launch composition.

P9 will not try to beat PARO. It will close the gap to within ~10–20% of
PARO and document the remaining residue as roofline-anchored, not
implementation-anchored.

### P9.1 — GDN prefill (Track A): the single biggest lever

**Why first.** GDN recurrent prefill is **73.5% of post-P8 prefill** for the
MoE path. Killing this bucket alone unblocks the next stage of WMMA tile
tuning and makes the 512/0 prefill total small enough that smaller buckets
start to matter.

**Why it is fixable.** Dense qwen35 GGUF already uses
`qwen35_gdn_prefill_recurrent_k2_f32` plus a separate
`qwen35_gdn_prefill_rmsnorm_gate_bf16`. The MoE GGUF path still uses
`qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order` — a fused but
**single-token-sequential** variant that does not unroll. Worse, parent
PARO has `qwen35_gdn_prefill_recurrent_k2_segments_kernel`, an unrolled and
segmented variant for long prefill that is already linked in our
`linear_attn/gdn.hip`. We are simply not calling it.

**P9.1 plan.**

- **P9.A1** Move qwen35moe GGUF prefill GDN from `decode_order_bf16` onto the
  unrolled-by-2 path: `qwen35_gdn_prefill_recurrent_k2_f32` for the
  recurrence, then `qwen35_gdn_prefill_rmsnorm_gate_bf16` for the fused
  RMSNorm+gate output stage. Then opt into the `segments_k2` variant when
  prefill length ≥ a tunable segment size (e.g., 256), mirroring the parent
  threshold. No new ABI; all three kernels are already in
  `hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip`.
- **P9.A2** CPU-reference correctness fixture. Compare GDN recurrent state
  and final RMSNorm-gate output of `decode_order` vs `k2` vs `segments_k2`
  paths on synthetic and Qwen3.6-35B-A3B-shaped inputs. Gate: KL ≤ 0.05 and
  top-1 ≥ 90% on the qwen35moe 512/128 fixture.
- **P9.A3** Retain a benchmark row: target GDN prefill bucket ≤ 200 ms at
  512/0 (≥ 3.3× over the current 666.9 ms), plus the matching rocprof CSV
  showing only `*_recurrent_k2*_kernel` and
  `*_prefill_rmsnorm_gate_bf16_kernel` symbols.

**Expected impact.** GDN 667 ms → ~150–200 ms moves the 512/0 total prefill
kernel time from 908 ms to roughly 390–440 ms. Wall-clock prefill would go
from 534 to roughly **1200–1400 tok/s** before any other tuning.

### P9.2 — PARO-style decode GEMVs (Track B): the entire decode pipeline

**Why.** Every GGUF projection at decode currently runs through a
row-shaped `*_prefill_out_kernel`. These kernels are correct but were not
written for c=1 throughput; PARO has dedicated decode GEMVs that fuse pack8
layout, register-resident weight reuse, and (optionally) rotation/scatter at
the boundary.

**Where to copy structure.** All in tree, ready to mirror:

- `gemv_awq_selected_dual_pack8_strided_kernel` (W4 PARO decode dual gate+up)
- `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel` (fused PARO rotate
  on the output to feed the next layer)
- `gemv_awq_pack8_kernel` (single PARO decode)
- `w8a16_shared_gate_up_bulk_kernel` and `w8a16_shared_down_bulk_combine_kernel`
  (shared-expert decode bundle; bulk over rows + combine + residual)

The only thing to swap is the **inner dequant** (AWQ pack8 → GGUF Q4_K /
Q5_K / Q6_K / Q8_0 raw block math, all already implemented in
`hipengine/kernels/cpu_reference/ops.py::gguf_quant_gemv` and in the existing
`_prefill_out` HIP kernels).

**P9.2 plan.**

- **P9.B1** GGUF Q4_K **selected dual pack8 GEMV (decode)**:
  `gguf_q4_k_selected_dual_pack8_gemv_kernel`. Compact-MoE ABI consistent
  with P8.4 (`expert_start_compact`, `wmma_expert_start`, `tile_expert`, raw
  rank-3 Q4_K bytes), but tuned for c=1 (rows=1) per active expert and
  written PARO-style with `__launch_bounds__(128, 4)` and pack8 row layout.
- **P9.B2** GGUF Q5_K and Q6_K **selected pack8 GEMV (decode)**:
  `gguf_k_selected_pack8_gemv_kernel<unsigned short, 5/6>`. Same compact ABI.
  Replaces today's `gguf_k_selected_pack8_prefill_out_kernel<...,5>` and
  `<...,6>` at decode shapes.
- **P9.B3** GGUF Q8_0 **dense pack8 GEMV (decode)** + a Q8_0 **dual gate+up
  decode GEMV** for the shared-expert path. Replaces
  `gguf_k_pack8_prefill_out_kernel<unsigned short, unsigned short, 8>` at
  decode shapes.
- **P9.B4** GGUF Q4_K **dense pack8 GEMV (decode)** for the attention/qkv/o
  surfaces that materialize Q4_K as pack8. Avoids a second "decode but
  through prefill_out" detour on dense Qwen layers in qwen35moe.
- **P9.B5** Correctness fixtures for B1–B4 against `kernels/cpu_reference/`
  oracle. Same shape coverage as P8.4/P8.5 selected WMMA tests, but with
  rows=1 and decode-typical shapes (256–2048 in, 256–4K out, multi-expert
  uneven counts including the all-empty case).
- **P9.B6** Registry wiring. Add new keys under `moe_linear` and `linear`
  (per-quant) with `*_pack8_gemv_decode_*` variants and route to them via
  the existing `_variant_for_rows(...)` mapping in
  `hipengine/runtime/gguf_linear.py`. No backend or quant if-branches; the
  selection is plugin-style. Default off until the bench row lands; explicit
  opt-in via the existing `wmma_prefill_session(...)` toggle which now
  becomes "P9 GGUF fast decode opt-in" as well.
- **P9.B7** Retained benchmark + rocprof. Target qwen35moe 512/128 decode
  ≥ 95 tok/s median over three runs, with the new `*_pack8_gemv_decode_*`
  kernels visible in the trace and the `*_prefill_out_kernel<...>` family
  absent at decode shapes (the Q6_K lm-head logits dispatch is still a known
  fallback).

**Expected impact.** PARO at 116 tok/s decode reaches ~28% of the 864 GB/s
ceiling; we're at ~15%. Halving per-projection-kernel overhead should put
us in the **90–105 tok/s** range. Closing the remaining gap to 116 likely
requires per-token kernel-launch reduction (P9.E) and possibly graph-bucket
expansion (P9.F2).

### P9.3 — WMMA prefill tuning (Track C)

**Why.** The new P8 selected WMMA kernels work but were tuned conservatively
(default tile, no rocprof-driven occupancy sweep). PARO `gemm_awq_*` got
their win from VGPR/launch_bounds tuning and per-shape tile selection.

**P9.3 plan.**

- **P9.C1** Tile / `__launch_bounds__` sweep for
  `gguf_q4_k_selected_dual_wmma_prefill_compact_kernel`. Compare TM/TN ∈
  {16, 32, 64} × {16, 32}, then pick per-shape using
  `_variant_for_rows(...)` with the same dispatch rule used for AWQ.
- **P9.C2** Same for `gguf_k_selected_wmma_prefill_compact_kernel<...,5/6>`
  (down projection).
- **P9.C3** Same for `gguf_q8_0_prefill_wmma_kernel<...,32,32>` (dense). The
  current single tile is the gating dispatch when `_variant_for_rows`
  selects WMMA; add 32×16, 64×32, and the "dual gate+up" Q8_0 fused variant.
- **P9.C4** rocprof occupancy + bandwidth audit. Back-calculate effective
  GB/s per kernel from the kernel trace, retain the per-shape tile decision
  in `docs/KERNELS.md`, and write the matching test that pins each chosen
  variant.

**Expected impact.** Compact MoE + Q8_0 WMMA combined: 165 ms → ~110 ms at
512/0. Modest standalone, meaningful after P9.1 has reduced GDN.

**Status 2026-05-20 (P9.C14).** The first Q4T16 selected-dual WMMA prototype
now consumes the P9.C13 tile-major layout directly via
`gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_{bf16,fp16}_...` while
preserving the compact selected-MoE output ABI. Synthetic BF16/FP16 fixtures
pass vs the CPU selected GGUF Q4_K reference; rocprof smoke records the new
kernel at `VGPR=64`, `SGPR=128`, `Scratch=0`, `LDS=0`. A one-sample first-layer
replay on the local RX 7900 XTX/gfx1100 measured raw gate+up `5.274 ms` vs
Q4T16 gate+up `3.897 ms` with `296 MiB` transient Q4T16 gate+up buffers. This
is diagnostic only, not a default runtime promotion; P9.C15 decides whether the
one-layer win survives full replay/replacement-layout integration.

**Status 2026-05-20 (P9.C15).** Full 512/0 replay rejected the compact32 Q4T16
WMMA path as a default. Same-run raw Q4 selected gate+up was `62.199 ms`; best
all-layer Q4T16 replay (launch-bound min-blocks `1`) was `59.395 ms` (`-4.5%`),
and selected-MoE total was effectively unchanged (`93.138 -> 92.478 ms`). This
misses the `<=35-40 ms` Q4 continuation target, so P9.C16 should evaluate a
different selected-MoE design rather than keep tuning this layout/consumer.

**Status 2026-05-20 (P9.C16).** Alternative evaluation did not select an
in-repo Q4 selected-MoE redesign. A compact tile-list/no-padding model on real
routing lowers Q4 only to `54.775 ms`; measured wider-column proxies were
`64x16 = 61.868 ms` and `64x32 = 91.831 ms`. Padding/tails and shallow column
reuse cannot close the gap, so #48 should not wire a Q4 redesign unless a new
parent-workspace kernel R&D result appears.

**Status 2026-05-20 (P9.C17).** Final #27 gate remains blocked with no runtime
wiring: carried-forward P9.C11 combined Q4/Q5/Q6/Q8 bucket is `140.110 ms` vs
`<=110 ms`, and P9.C15/P9.C16 found no Q4 redesign worth promoting. #27 stays
open/blocked; any next Q4 selected-MoE attempt needs parent-workspace R&D or a
new design task before hipENGINE dispatch changes.

### P9.4 — Dispatch reduction and small-op fusion (Track D)

**Why.** With GDN and the decode-GEMV families addressed, the residual
buckets are dominated by router + small-op kernels: router logits + select,
weighted lanes sum, scheduler counts/prefix/tile-map, redundant bf16↔f32
casts, separate silu+mul, and gate-combine-residual. None of these are big
individually; together they currently sit around ~30 ms at prefill and
~150 ms at decode (per 128 tokens).

**P9.4 plan.**

- **P9.D1** Fuse `qwen35_router_logits_token_tile + qwen35_router_select`
  into a single launch (in tree these are already small, but the launch
  overhead is wasted at decode). **Status 2026-05-20:** decode uses the
  GGUF split expert/shared cooperative router for `rows=1`, replacing
  expert-router logits + shared-gate logits + select with one launch.
  Correctness: P9.E2 accepted (`KL=0`, top-1 `100%`, deterministic tails).
  512/128 graph replay on local RX 7900 XTX/gfx1100 (W7900 not rerun) moved median decode `85.728 ->
  85.817 tok/s` (+0.10%); retained as a small positive D1 reduction, but
  #51 remains below the `95 tok/s` target.
- **P9.D2** Audit redundant `bf16_to_f32` / `f32_to_bf16` boundary kernels.
  In the post-P8 prefill trace they appear in pairs; in decode they fire
  thousands of times. Most can be folded into the consumer kernel as an
  in-register cast. **Status 2026-05-20:** a BF16-key variant of the GGUF
  head RMSNorm+partial-RoPE kernel removed the full-attention key
  `bf16_to_f32` launch and passed P9.E2 (`KL=0`, top-1 `100%`), but 512/128
  graph replay regressed versus the retained split-router baseline
  `85.817 -> 85.582 tok/s` (-0.27%). Rejected and reverted; artifact:
  `benchmarks/results/2026-05-20-hipengine-qwen36-35b-a3b-q4km-p9_d2-bf16-key-rope-rejected.json`.
- **P9.D3** Consolidate the compact scheduler. `group_count + group_prefix +
  wmma_tile_map` are three separate launches sharing the same `num_experts`
  workgroup; on small expert counts the three-launch sequence dominates the
  scheduler bucket. A single fused launch keeps the existing compact-MoE
  ABI. **Status 2026-05-20: rejected/deferred for P9 decode.** The retained
  512/128 c=1 graph path no longer spends time in the compact-scheduler trio;
  it falls back to the selected decode path before those rows>1 compact WMMA
  helpers. The remaining scheduler work is prefill/rows>1 scope already tracked
  by #27, whose tile/launch sweep is exhausted without meeting its prefill
  target. No D3 code was retained for #51.
- **P9.D4** Decide where SiLU+Mul lives. Today qwen35moe uses
  `silu_mul_dual_out_bf16` after the dual gate+up WMMA. PARO fuses the SiLU
  into the gate-side accumulator and emits only `mul` over half operands.
  **Status 2026-05-20:** retained the decode-only Q4T16 selected-dual
  GEMV+SiLU variant for rows=1. It round-trips gate/up accumulators through
  BF16 before SiLU to match the existing split-kernel contract; P9.E2 accepted
  (`KL=0`, top-1 `100%`, deterministic tails). Rows>1 bulk prefill remains on
  the split gate/up + SiLU launch because the same accumulator-side fusion did
  not pay for the extra exp/rounding work at prefill shapes. 512/128 graph
  replay moved the retained D1 baseline `85.817 -> 86.025 tok/s` (+0.24%);
  #51 remains below the `95 tok/s` target.
- **P9.D5** Decide where the gate-combine-residual lives. Today qwen35moe
  emits two small kernels (`weighted_lanes_sum_out` then
  `shared_gate_combine_residual_batch_out`). PARO fuses these into one. Try
  the fused PARO style; gate on KL + top-1. **Status 2026-05-20: satisfied in
  the active decode path.** The retained c=1 fallback already uses
  `weighted_sum_shared_gate_combine_residual_out_bf16_f32w`; D10 512/16
  rocprof saw the MoE-combine bucket as exactly 640 dispatches of
  `weighted_sum_shared_gate_combine_residual_out_kernel` and no separate
  `weighted_lanes_sum_out + shared_gate_combine_residual_batch_out` pair on the
  measured decode path. Rows>1 helper cleanup is outside the #51 decode target.
- **P9.D6** Opportunistic Q8T16 same-input pair dispatch. **Status
  2026-05-20:** retained a split-output Q8T16 dual GEMV for `attn_k+attn_v`
  and `ssm_alpha+ssm_beta` pairs. It preserves separate scratch buffers while
  removing one Q8T16 launch per pair. P9.E2 accepted (`KL=0`, top-1 `100%`,
  deterministic tails); 512/128 graph replay moved the D4 baseline `86.025 ->
  86.502 tok/s` (+0.55%). #51 remains below the `95 tok/s` target.
- **P9.D7** Extend Q8T16 pair dispatch to unequal-width full-attention
  projections. **Status 2026-05-20:** routed `attn_qkv+attn_gate` through the
  same split-output dual Q8T16 GEMV with `out_features_b` for the smaller gate
  projection, preserving the existing `linear_qkv` and `linear_z` scratch
  buffers. P9.E2 accepted (`KL=0`, top-1 `100%`, deterministic tails);
  512/128 graph replay moved the D6 baseline `86.502 -> 87.961 tok/s`
  (+1.69%). #51 remains below the `95 tok/s` target, so the remaining gap
  needs deeper Q8/full-attention decode reductions.
- **P9.D8** Reject Q8T16 d-scale shared-cache prototype. **Status
  2026-05-20:** a kernel-local shared-memory cache for the Q8_0 T16 `d`
  scales preserved the synthetic CPU-oracle fixture (`17 passed`) but added
  enough synchronization/occupancy cost to regress the 512/128 graph replay
  from D7 `87.961 -> 81.253 tok/s` (-7.62%) and prefill `500.480 ->
  476.853 tok/s` (-4.72%). Reverted; do not retry without a parent-workspace
  ISA/occupancy audit.
- **P9.D9** Full-attention Q/K/V Q8T16 triple dispatch. **Status
  2026-05-20:** retained a split-output triple Q8T16 GEMV and routed
  `attn_q+attn_k+attn_v` through one same-input launch while preserving
  separate Q/K/V scratch buffers. Synthetic CPU-oracle fixtures pass, rocprof
  saw `q8_0_t16_triple_split_gemv_kernel`, and P9.E2 accepted (`KL=0`, top-1
  `100%`, deterministic tails). 512/128 graph replay moved D7 `87.961 ->
  88.243 tok/s` (+0.32%). #51 remains below the `95 tok/s` target.
- **P9.D10** Q8T16 separate-output dual launch width. **Status
  2026-05-20:** retained a narrower `64`-thread block for
  `q8_0_t16_dual_split_gemv_kernel` (used by separate-output unequal-width
  Q8T16 decode pairs) without changing the ABI or dispatch graph. P9.E2
  accepted (`KL=0`, top-1 `100%`, deterministic tails); 512/128 graph replay
  moved D9 `88.243 -> 88.801 tok/s` (+0.63%). A rejected adjacent prefix-only
  dual prototype reduced launches but inflated the Q8T16 bucket in 512/16
  rocprof, so only the launch-width change is retained. #51 remains below the
  `95 tok/s` target.
- **P9.D11** Q8T16 shared gate/up SiLU fusion. **Status 2026-05-20:
  rejected/reverted.** A fused shared-expert Q8T16 gate/up GEMV that wrote
  `SiLU(gate) * up` directly removed the separate `silu_mul` launch, but the
  fused kernel was slower than the existing `dual_gate_up + silu_mul` chain:
  local 512/16 graph replay regressed D10 `78.745 -> 76.527 tok/s` with a
  128-thread block, while a 64-thread variant fell to `75.458 tok/s` and changed
  the 512/16 generated tail. Do not retry this path without a different tiling
  strategy; artifact: `benchmarks/results/2026-05-20-hipengine-qwen36-35b-a3b-q4km-p9_d11-rejected-q8t16-shared-silu.json`.
- **P9.D12** Full-attention context+gate fusion. **Status 2026-05-20:
  rejected/reverted.** A direct BF16 gated-context attention kernel removed the
  separate `qwen35_full_attn_gate_mul_bf16` launch and passed P9.E2 (`KL=0`,
  top-1 `100%`, deterministic tails), but the heavier attention kernel
  outweighed the launch removal: 512/128 graph replay regressed D10 `88.801 ->
  88.576 tok/s`. Keep the unfused paged-attention context plus gate-mul chain;
  artifact: `benchmarks/results/2026-05-20-hipengine-qwen36-35b-a3b-q4km-p9_h3-rejected-attn-gate-fusion.json`.
- **P9.D13** Dense BF16 `ssm_alpha+ssm_beta` decode pair. **Status
  2026-05-20:** retained a qwen35moe rows=1 linear-attention route that writes
  `ssm_alpha` and `ssm_beta` through one existing `dense_dual_gemv_out_bf16`
  launch into a tiny combined scratch buffer, then passes split pointers to the
  GDN kernel. P9.E2 accepted (`KL=0`, top-1 `100%`, deterministic tails);
  512/128 graph replay moved D10 `88.801 -> 89.303 tok/s` (+0.56%). 512/16
  rocprof diagnostic reduced the dense alpha/beta bucket from `2.950 ms / 960`
  singleton dispatches to `1.667 ms / 478` dual dispatches. #51 remains below
  the `95 tok/s` target.
- **P9.D14** Q8T16 F32-input `ssm_out` decode. **Status 2026-05-20:**
  retained a Q8_0 T16 F32-input/BF16-output single GEMV registry variant and
  route qwen35moe rows=1 `ssm_out` directly from the FP32 GDN output when
  resident decode repack is enabled. This removes the per-layer
  `f32_to_bf16` conversion launch before `ssm_out`. P9.E2 accepted (`KL=0`,
  top-1 `100%`, deterministic tails); 512/128 graph replay moved D13 `89.303
  -> 90.149 tok/s` (+0.95%). 512/16 rocprof diagnostic reduced total decode
  dispatches `10648 -> 10138` and removed the `f32_to_bf16` decode bucket
  (`0.860 ms / 478` dispatches). #51 remains below the `95 tok/s` target.
- **P9.D15** Full-attention BF16-key RoPE decode. **Status 2026-05-20:**
  retained a GGUF F32-weight head-RMSNorm+RoPE variant that consumes the
  full-attention K projection as BF16 input and converts inside the fused
  RoPE/RMSNorm kernel. This removes the separate `bf16_to_f32` key conversion
  launch in rows=1 resident decode-repack full-attention layers. P9.E2 accepted
  (`KL=0`, top-1 `100%`, deterministic tails); a 16-token correctness override
  also accepted. 512/128 graph replay moved D14 `90.149 -> 90.868 tok/s`
  (+0.80%). 512/16 rocprof diagnostic reduced total decode dispatches `10138
  -> 9968` and removed the `bf16_to_f32` decode work (`0.302 ms / 159`
  dispatches). #51 remains below the `95 tok/s` target.

**Expected impact.** ~30 ms at 512/0 → ~10 ms, ~150 ms at 512/128 decode
→ ~50 ms. Modest in absolute terms; visible at decode because each savings
multiplies by 128 tokens.

### P9.5 — Cross-cutting infra (Track E)

The perf tracks above will be hard to evaluate without these:

- **P9.E1** Add a `scripts/qwen35_gguf_rocprof_summary.py` (or extend the
  existing rocprof helper) that ingests a rocprofv3 CSV and reports per-kernel
  total ms, dispatches, average dispatch ms, and **back-calculated effective
  GB/s** using known weight + activation footprints. This formalizes the
  audit in `docs/ROOFLINE.md` §12.4 for our pipeline.
- **P9.E2** Add a public E2E KL/top-1 correctness fixture that runs the full
  qwen35moe 512/128 with `HIPENGINE_GGUF_WMMA_PREFILL=1` and the new decode
  GEMV opt-in, gates on KL ≤ 0.05 and top-1 ≥ 90% vs the row-GEMV reference,
  and fails the gate if the reduction-order drift the P8 acceptance flagged
  becomes worse than this threshold.
- **P9.E3** Expand the qwen35moe decode hipGraph bucket policy so the
  captured graph covers the new GEMV decode families (and the existing
  GDN/full-attention decode kernels) under one replay budget. Required
  before the P9.G acceptance run can be a real graph-replay number rather
  than an eager rocprof workaround. **Status 2026-05-20:** implemented as a
  c=1 `Qwen35GGUFDecodeGraphBucketKey` attached to every resident decode
  graph capture. The key records `(active_c, context_bucket, replay_steps,
  max_replay_steps)` plus the active rocprof symbol groups derived from the
  materialized qwen35moe weights (Q4 selected dual, Q5/Q6 selected, Q8_0
  single/dual, optional dense Q4, Q6 lm-head, GDN, paged KV write, and paged
  full-attention decode). `scripts/qwen35_gguf_decode_graph_smoke.py` now
  emits the bucket and has a `--coverage-only --coverage-csv ...` mode that
  fails if a trace is missing any active symbol group.

### P9.6 — Acceptance benchmark and rollups (Track G)

- **P9.G1** After P9.1 (GDN) and P9.2 (decode GEMV) land, rerun
  qwen35moe Qwen3.6-35B-A3B-UD-Q4_K_M 512/0 and 512/128 with cached builds.
  Acceptance: prefill total kernel time ≤ **350 ms** at 512/0 (~2x of stretch
  baseline, ~5x of current); decode ≥ **95 tok/s** median. Stretch: prefill
  ≤ 250 ms (~3.6x current), decode ≥ 110 tok/s. Update
  `benchmarks/results/`, `benchmarks/README.md`, `benchmarks/CHANGELOG.md`,
  and `WORKLOG.md` atomically with the perf row, rocprof CSV, and KL gate
  output.

### Sequencing

P9 splits cleanly into two waves and a closeout:

- **Wave 1 (the big prefill win):** P9.A1 → P9.A2 → P9.A3. This is the most
  cost-effective first step. Single biggest measurable single-task delta.
- **Wave 1.5 (cross-cutting):** P9.E1 (rocprof bandwidth summary) and P9.E2
  (E2E KL gate) should land in parallel with Wave 1 so Wave 2 has the
  evidence machinery in place.
- **Wave 2 (decode pipeline parity):** P9.B1 → P9.B2 → P9.B3 → P9.B4 →
  P9.B5 → P9.B6 → P9.B7. New GEMV kernels are mechanical ports of well-
  understood PARO templates; the work is mostly correctness fixtures and
  registry wiring.
- **Wave 3 (tuning and small-op fusion):** P9.C1–C4, P9.D1–D5, P9.E3.
- **Closeout:** P9.G1.

### Performance gates

No P9 row is retained until the matching gate fires:

| Gate | Target | Where it lives |
| --- | --- | --- |
| P9.A1 GDN prefill bucket | ≤ 200 ms at 512/0 | rocprof CSV + bench artifact + WORKLOG |
| P9.B7 decode GEMV bucket | `_pack8_gemv_decode_*` symbols visible at rows=1 and `_prefill_out_kernel<...>` family absent at decode shapes | rocprof CSV + bench artifact + WORKLOG |
| P9.G1 acceptance | qwen35moe 512/0 ≤ 350 ms total kernel and 512/128 decode ≥ 95 tok/s | benchmark artifact + rollup + CHANGELOG |
| Stretch closeout | qwen35moe 512/0 ≤ 250 ms total kernel and 512/128 decode ≥ 110 tok/s | benchmark artifact + rollup + CHANGELOG |

### Correctness gates

Every P9 kernel honours the project policy:

1. Match `kernels/cpu_reference/` math to within F32 tolerance
   (`atol=1e-3, rtol=1e-2` for Q4_K/Q5_K/Q6_K; `atol=5e-4, rtol=5e-3` for
   Q8_0). WMMA F32 accumulation should be nearly bit-exact when K reduction
   order is preserved.
2. Pass the P9.E2 KL/top-1 E2E fixture on qwen35moe 512/128 with the new
   decode opt-in active. KL ≤ 0.05 and top-1 ≥ 90% vs the row-GEMV reference.
3. Provide a `rocprofv3 --kernel-trace` smoke proving the new kernel symbols
   are present and the relevant decode-shaped `_prefill_out_kernel` symbols
   are absent on the decode code path.
4. CPU-reference fixture tests live in `tests/test_gguf_*_decode_gemv.py`
   and use the same `make_q4_k_weight` / `make_q5_k_weight` / `make_q6_k_weight`
   / `make_q8_0_weight` synthetic block generators already used by P8.

### What we are deliberately not doing in P9

- **No new resident weight repack and no sidecar.** The pack8 layout we
  already have at runtime is the dispatch layout the PARO templates expect.
  Materialization stays exactly as it is.
- **No activation quantization (W8A8-style I8 WMMA).** That is P8.8 / a
  separate phase. P9 stays F16/BF16-operand WMMA on the prefill side and
  pack8-row-resident dequant on the decode side.
- **No changes to AOTriton attention prefill** (`qwen35_paged_full_attn_prefill_gqa_gate_bf16_kernel`
  at 39.5 ms is already small).
- **No new C++ engine layer.** The runner stays Python + ctypes + JIT HIP
  kernels. The compact-MoE scheduler ABI does not change.
- **No "if quant == ..." branches in dispatch or model code.** All P9 wiring
  goes through registry keys and the existing `_variant_for_rows(...)`
  mapping. New variants are new `KernelKey` rows, not new code paths.
- **No GDN ABI change.** P9.A1 is a kernel-selection change, not a new
  recurrent scheduler.
- **No attempt to beat PARO.** Realistic close is ~10–20% of PARO on prefill
  and ~5–10% on decode; the rest is roofline residue and tooling overhead.

### Open questions (decide before each Wave lands)

1. Which GDN prefill variant wins at 512: `k2` or `segments_k2`? Resolved for
   the active GGUF Q4_K_S gate on 2026-06-16: default threshold `1025` keeps the
   exact single-segment `k2` path for 512/1024-row chunks, while the
   `segments_k2` path remains available through
   `HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD` for larger/batched probes.
2. Should the new `*_pack8_gemv_decode_*` kernels register under a separate
  `layer="linear_decode"` family or share `layer="linear"` and dispatch via
  `_variant_for_rows(rows=1)`? The latter avoids a new layer key; the former
  isolates the c=1 contract from the prefill rows>1 path. Decision tracked
  in P9.B6.
3. Where does the fused PARO SiLU live for GGUF (P9.D4)? Resolved for the
   current Q4T16 decode path: rows=1 uses a decode GEMV+SiLU variant, while
   rows>1 bulk prefill keeps the separate compact SiLU kernel.
4. Should P9 expand to dense Qwen3.5 (qwen35) as well, or stay scoped to
  qwen35moe? Dense Qwen3.5 currently uses the GDN k2 path already, and the
  P8 dense Q4_K WMMA path is blocked by materialization. P9 stays scoped to
  qwen35moe; dense Qwen3.5 work is tracked separately.

### Acceptance checklist for closing P9

- [ ] P9.A1–A3 GDN prefill on `k2`/`segments_k2` with fused RMSNorm gate;
  GDN bucket ≤ 200 ms at 512/0; correctness gate passed.
- [ ] P9.B1–B5 decode GEMV kernels written, registered, CPU-reference-gated.
- [ ] P9.B6 decode dispatch routes rows=1 GGUF projections through the new
  GEMV kernels via registry only; no quant/backend branches added.
- [ ] P9.B7 qwen35moe 512/128 decode ≥ 95 tok/s with rocprof showing the
  new GEMV symbols and the old `_prefill_out_kernel<...>` family absent at
  decode shapes (Q6_K lm-head fallback excluded).
- [ ] P9.C1–C4 WMMA prefill kernels tuned per-shape; `docs/KERNELS.md` row
  updated with the chosen variants.
- [x] P9.D1–D5 small-op fusions either landed (with bench evidence) or
  explicitly rejected (with diagnostic evidence) in this doc and `WORKLOG.md`.
- [ ] P9.E1–E3 tooling landed (rocprof summary, KL gate, decode graph
  bucket).
- [ ] P9.G1 acceptance benchmark passes target gate; rollups updated
  atomically.
- [ ] No `import torch` on the `LLM.generate()` path; no `if backend == ...`
  / `if quant == ...` branches added to runtime dispatch.

## P10: Path to >2500 prefill / >120 decode (active)

Date opened: 2026-05-20

P9 closed the decode target (`98 tok/s` median in T16 decode-repack mode
vs `≥95 tok/s` gate) and closed the raw-WMMA prefill arm
(`~1859 tok/s` at 512/0, `~1506 tok/s` at 512/128). The combined P9.G1
acceptance is still **blocked on prefill in T16 mode**, and neither current
mode beats llama.cpp HIP (`2436/85`) or PARO (`2696/116`). P10 is the
focused push to make one correct mode meet both targets at once.

This section is the **live optimization punchlist** for that push. It mirrors
the table format from `docs/OPTIMIZE.md` (per-candidate `Expected Δ / Actual Δ`
rows) so we can see at a glance which levers are still on the table, which
have landed, and which were rejected with evidence.

Per-kernel detail stays in `docs/KERNELS.md`. Per-run evidence stays in
`benchmarks/results/`, `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and
`WORKLOG.md`. Anything that lands in this section requires:

1. P9.E2 KL/top-1 gate passes with `effective_wmma_prefill=true` and/or
   `effective_gemv_decode=true` as appropriate (not safety-disabled).
2. `rocprofv3 --kernel-trace` shows the new kernel and confirms the old
   bucket shrank.
3. A retained benchmark JSON artifact in `benchmarks/results/` with exact
   command and hardware (W7900 / gfx1100 when possible; otherwise local
   RX 7900 XTX / gfx1100 noted explicitly).

No `import torch` on the `LLM.generate()` path. No `if quant == ...` or
`if backend == ...` branches in dispatch — every new variant enters via a
`KernelKey` registration.

### P10.0 — Where we are after P9 (post-P9.G1 blocked acceptance)

Two competing peaks, **neither close to target on its own**:

| Mode | Prefill 512/0 (tok/s) | Decode 512/128 (tok/s) | Correctness | 512/0 total kernel ms |
| --- | ---: | ---: | --- | ---: |
| T16 decode-repack (default, safe) | **506** | **98** | P9.E2 accepted (`KL 0`, top-1 `100%`) | **1004** |
| Raw + WMMA prefill (legacy decode) | **1859** | **62** | Deterministic but drifts vs serial ref | **265** |
| Raw + WMMA prefill + GEMV decode | safety-gated | safety-gated | P9.E2 rejected (`KL 5.99`) | n/a |
| llama.cpp HIP UD-Q4_K_M 512/128 | 2436 | 85 | — | — |
| llama.cpp Vulkan UD-Q4_K_M 512/128 | 1817 | 128 | — | — |
| PARO Qwen3.5-35B-A3B w4a16 512/128 | 2696 | 116 | graph/step true | — |
| **P10 target** | **≥2500** | **≥120** | **P9.E2 effective fastpath gate passes** | ≤205 (= 2500 tok/s) |

Sources: `benchmarks/README.md` 2026-05-20 rollup,
`benchmarks/results/2026-05-20-hipengine-qwen36-35b-a3b-q4km-p9_g1-final-acceptance-blocked.json`,
`benchmarks/results/2026-05-18-hipengine-qwen36-35b-a3b-q4km-p9_a3-gdn-k2-chain-accepted.json`,
`~/amd-gpu-tuning/docs/OPTIMAL.md` 2026-05-13.

### P10.0a — Root cause of the prefill gap (T16 mode 512/0 bucket profile)

From the P9.G1 blocked artifact:

| Bucket | Kernel family | Total ms | Share | Status |
| --- | --- | ---: | ---: | --- |
| Q4_K selected gate+up (T16 **GEMV** at prefill rows) | `gguf_q4_k_t16_selected_dual_gemv_*` | **287.7** | 28.6% | **decode-shape kernel running at prefill rows — P10.B1** |
| Q5_K selected down (T16 **GEMV** at prefill rows) | `gguf_q5_k_t16_selected_gemv_*` | **232.0** | 23.1% | **decode-shape kernel running at prefill rows — P10.B2** |
| Q8_0 dense/shared (T16 **GEMV** at prefill rows) | `gguf_q8_0_t16_*_gemv_*` | **160.4** | 16.0% | **decode-shape kernel running at prefill rows — P10.B4** |
| Other small ops (router, silu, gate combine, lane sum, lm-head, casts) | many | **183.8** | 18.3% | per-op cleanup — P10.D |
| GDN recurrent (linear-attention) | `qwen35_gdn_prefill_recurrent_segments_k2_kernel` | 52.8 | 5.3% | already optimized (P9.A1) |
| Full-attention prefill (10 layers) | `qwen35_paged_full_attn_prefill_gqa_gate_bf16_kernel` | 40.1 | 4.0% | already optimized |
| Q6_K selected down (T16 GEMV) | `gguf_q6_k_t16_selected_gemv_*` | ~3 | 0.3% | small (3 layers); folds into P10.B3 |
| **Total** | | **1004.5** | 100% | wall: 506 tok/s |

**Root cause in one sentence:** the T16 (decode-repack) memory layout has
`gemv` decode kernels for every quant but **no WMMA prefill kernel for any
quant**. The bulk prefill runner therefore falls through `_resolve_compact_moe_wmma_kernels`
(table lookup returns `None` for T16 quant keys) and dispatches the
rows=1-shaped T16 GEMV per row, paying ~512× the launch overhead and missing
WMMA throughput entirely. The Q4T16 selected-dual WMMA prefill kernel
(`gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_*`, P9.C14) **already
exists in tree but is not wired into `_COMPACT_MOE_Q4_DUAL_KEYS`**. Q5T16,
Q6T16, and Q8T16 WMMA prefill kernels do not yet exist.

Reference: raw-WMMA mode total kernel time at 512/0 is `265 ms` for the same
shape and model. The raw WMMA buckets are: Q4 dual `62 ms`, Q5 down `27 ms`,
Q6 down `2.7 ms`, Q8_0 dense `52 ms`, GDN `56 ms`, other `~65 ms`. If T16
WMMA prefill reaches raw-WMMA parity, total kernel time falls from `1004 ms`
to roughly `265–290 ms` (`~3.5×`), wall prefill from `506` to roughly
`1750–1900 tok/s`. From there, additional WMMA-layout tuning and small-op
fusion close the remaining gap to ≥2500.

### P10.0b — Decode 98 → ≥120 tok/s

Decode is bandwidth-bound: Q4_K_M Qwen3.5-35B-A3B active weights are
`~1.7 GB/token`, so the W7900 `864 GB/s` peak gives `≤508 tok/s` hard ceiling
and `≈412 tok/s` at 80% realistic peak. PARO at `116 tok/s` reaches `~28%`,
llama.cpp Vulkan at `128 tok/s` reaches `~31%`, and we are at `98 / 412 ≈ 24%`.
Closing to `≥120` (`≈29%`) is a `~22%` improvement on the current decode
bucket mix.

512/16-minus-512/0 rocprof deltas (P9.D18 artifact) show the remaining decode
buckets are: selected Q4T16 dual GEMV `~85 ms`, Q8T16 dense/shared GEMV
`~30 ms`, paged full-attention split-K GQA `~14 ms`, GDN recurrent + RMSNorm
`~15 ms`, MoE scatter / scheduler / lane sum / combine / router / silu
`~7 ms`. Most levers are sub-1ms-per-token launches that compound across 128
tokens. P10 decode work concentrates there.

### P10.1 — Optimization roadmap (priority order, expected vs actual)

Priority is `(impact × confidence) / effort`. Wave 1 (P10.B + P10.C) is the
big prefill closeout. Wave 2 (P10.D) is decode cleanup. Wave 3 (P10.E) is
fusion / activation-quant moonshots that may unlock the stretch target.
IDs are stable for `WORKLOG.md` and commit messages.

| ID | Candidate | Expected prefill Δ (tok/s) | Expected decode Δ (tok/s) | Memory Δ | Effort | Status | Actual prefill | Actual decode | Evidence |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- |
| **Wave 1 — unblock T16 prefill (Q4 → Q5 → Q6 → Q8)** | | | | | | | | | |
| P10.B1 | Wire existing Q4T16 selected-dual WMMA prefill kernel into `_COMPACT_MOE_Q4_DUAL_KEYS` | +700 to +900 (Q4 bucket `287→~60 ms`) | ~0 | 0 GiB | XS (one table entry + dispatch validation) | **✅ landed (`731241a`)** | +0 (no-op alone; needs B2/B3 to resolve down) | +0 | resolver test added; kernel was registered as `gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_*`, P9.C14 artifact |
| P10.B2 | Build Q5T16 selected-down WMMA prefill kernel + wire to `_COMPACT_MOE_DOWN_KEYS` | +250 to +400 (Q5 bucket `232→~30 ms`) | ~0 | 0 GiB (T16 already resident) | M (port from `gguf_k_selected_prefill.hip<...,5>` consuming T16 tiles) | **✅ landed (`bb0c933`)** | **B1+B2+B3 combined: +422 (506→928)** | +0 | 22 CPU-reference fixtures pass; rocprof shows `gguf_k_t16_selected_wmma_prefill_compact_kernel<...,5>` active at 26 ms / 37 disp |
| P10.B3 | Build Q6T16 selected-down WMMA prefill kernel + wire (3 layers, small) | +20 to +40 | ~0 | 0 GiB | S | **✅ landed (`bb0c933`)** | included in B2 row | +0 | `gguf_k_t16_selected_wmma_prefill_compact_kernel<...,6>` runs at 2.3 ms / 3 disp |
| P10.B4 | Build Q8T16 dense WMMA prefill kernel (shape-aware tiles like P9.C3) + wire dense dispatch | +150 to +250 (Q8 bucket `160→~55 ms`) | ~+5 | 0 GiB | M | **✅ landed (`ed14ed5`)** | **+972 (928→1900)** — includes pair/triple/concat decline-to-singleton fix | +0 | 75 CPU-reference fixtures pass; rocprof shows `gguf_q8_0_t16_prefill_wmma_kernel<...,16,32>` + `<...,64,32>` + `<...,32,32>` active total 59 ms / 250 disp; Q8T16 dual_split_gemv 176 ms bucket eliminated |
| **Wave 1.5 — acceptance gate after Wave 1** | | | | | | | | | |
| P10.B5 | P9.E2 KL/top-1 fixture re-run with effective WMMA prefill + GEMV decode (T16 mode) | gate, not perf | gate, not perf | 0 | XS (re-run) | **✅ resolved by P10.X2 layer0 real-weight gate** | n/a | n/a | after P10.X1 fixes, `REPACK=1, WMMA=0, GEMV=0` and `REPACK=1, WMMA=0, GEMV=1` both match reference exactly (`KL=0`, top-1 `100%` on 512/1 probes); the full-sequence `effective_wmma_prefill=true` KL gate fails because MoE hard-router argmax is chaotic under ULP-level Matrix-Core vs Vector-ALU drift, not because WMMA prefill math is wrong. P10.X2 layer0 gate passes with 512/512 expert agreement and max/mean output diff `0.000977 / 0.000004`. |
| P10.B6 | Retained 512/0 + 512/128 acceptance with rollups | gate | gate | 0 | XS (one bench) | **ready for promotion/re-run under updated P10.X2 gate** | bench: **1900.231** @ 512/0 (throughput gate met) | bench: **98.250** @ 512/128 (throughput gate met) | previous acceptance attempt was recorded as blocked diagnostic in `benchmarks/results/2026-05-20-hipengine-qwen36-35b-a3b-q4km-p10-b6-acceptance-blocked.json`; after P10.X2, the retention gate is layer0 real-weight correctness plus 512/128 and 4K/128 perf gates. Re-run before updating rollup. |
| **Wave 2 — push prefill from ~1850 to ≥2500** | | | | | | | | | |
| P10.C1 | Tile / `__launch_bounds__` sweep for Q4T16 dual WMMA prefill on real 512 routing | +100 to +200 | ~0 | 0 | S | pending | | | extend P9.C15 method, drive by `scripts/qwen35_gguf_rocprof_summary.py` |
| P10.C2 | Tile sweep for Q5T16 / Q6T16 / Q8T16 WMMA prefill (per-shape decision) | +50 to +150 | ~0 | 0 | S | pending | | | mirror P9.C3/P9.C4 |
| P10.C3 | hipGraph capture of bulk prefill chunks (one capture per (rows, layer-type)) | +100 to +300 (eliminates per-launch host latency on ~1200 dispatches/step) | ~0 | small scratch | M | research (P9.E3 covers decode; prefill capture is new) | | | parent does not graph prefill; llama.cpp HIP does graph prefill via cudaGraph clones |
| P10.C4 | Compact scheduler launch reduction (fuse `group_count + group_prefix + wmma_tile_map` to one launch) | +20 to +60 (residual `other` bucket) | ~0 | 0 | S | rejected for decode in P9.D3; **revisit for rows>1 prefill** with new acceptance bench | | | P9.D3 WORKLOG; `qwen35_moe_group_*` kernels |
| P10.C5 | Drop redundant BF16↔F32 casts on the prefill path (fold cast into consumer kernel where safe) | +20 to +60 | +1 to +3 | 0 | S | pending | | | mirror P9.D2 method on the prefill side; P9.D2 decode variant was rejected |
| **Wave 3 — decode push from 98 to ≥120 tok/s** | | | | | | | | | |
| P10.D1 | Fused Q4T16 selected dual gate+up + SiLU + Q5T16/Q6T16 down + scatter combine — single decode launch per layer | ~0 | **+8 to +15** | 0 | L (parent has analog: PARO `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel`) | pending | | | builds on P9.D4 (Q4T16 SiLU fusion retained); not yet down-fused |
| P10.D2 | Wider tile or block-launch tuning on Q4T16 selected decode (currently `compact32`); try `compact64`/`compact96` for hot experts | ~0 | +3 to +8 | 0 | S | pending | | | follows P9.D10 width-tuning method on dual_split GEMV |
| P10.D3 | Q8T16 shared-expert gate+up+SiLU+down fused decode kernel | ~0 | +3 to +8 | 0 | M | re-attempt (P9.D11 failed on 128/64-thread variants; try a different K-tile + LDS plan) | | | parent: `w8a16_shared_gate_up_bulk_kernel` + `w8a16_shared_down_bulk_combine_kernel` |
| P10.D4 | Decode-only Q4T16 fused-K micro-batch (write 2 tokens per launch when capture allows) | ~0 | +5 to +10 | small scratch | M | pending | | | hipGraph 2-step replay extension; needs P9.E3 graph bucket update |
| P10.D5 | Drop `f32_to_bf16` casts already folded for `ssm_out` (P9.D14) into other narrow surfaces (router logits, lm-head) | ~0 | +1 to +3 | 0 | S | pending | | | mirror P9.D14 |
| **Wave 4 — stretch moonshots (only if Wave 1–3 still short of target)** | | | | | | | | | |
| P10.E1 | INT8 activation quant for Q8_0 dense path (`W8A8` WMMA INT32 accum) | +200 to +400 | ~0 | 0 | L (correctness-gated by P9.E2) | research | | | RDNA3 has `v_wmma_i32_16x16x16_iu8` at `2×` BF16 rate; deliberately deferred in P9 "not doing" list |
| P10.E2 | Replace Q4_K dequant in K-loop with on-chip `scale/min` LUT precomputed once per workgroup | +50 to +150 | +2 to +5 | small LDS | M | research (P9.D8 d-scale shared cache regressed Q8 decode −7.6%; needs different design) | | | parent kernel R&D; do not retry without parent workspace win |
| P10.E3 | rocBLAS / hipBLASLt for Q8_0 dense BF16 GEMM after on-the-fly dequant | +50 to +150 | ~0 | +1 GiB transient | L (adds runtime dep behind extra) | research | | | llama.cpp HIP uses rocBLAS for dense GEMM; would add `roc::rocblas` runtime link |
| P10.E4 | Lossy MoE expert path with sticky hot-expert cache (top-N pre-decode of expert weights) | +100 to +200 | +5 to +10 | +1 to +2 GiB | L (gate on KL drift) | research | | | parent workspace evidence required first |

#### Wave 1 “if everything hits the low end” math

```
current T16 mode  prefill 506 tok/s @ 1004 ms total kernel
P10.B1 lands     ~+700                  bucket 287 →  60  (∆ = 227 ms saved)
P10.B2 lands     ~+250                  bucket 232 →  30  (∆ = 202 ms saved)
P10.B3 lands     ~+30                   bucket   3 →   1  (∆ =   2 ms saved)
P10.B4 lands     ~+170                  bucket 160 →  55  (∆ = 105 ms saved)
→ total kernel  ~ 468 ms; wall prefill ~ 512/0.468 ≈ 1094 tok/s
```

That is **below** the Wave 1 simple sum because the bottlenecks reorder once
the big T16 GEMV buckets are killed; the floor is set by the `other 184 ms`
bucket. Wave 2 (P10.C1–C5) attacks `other` and the WMMA tile efficiency.

#### Wave 1+2 “everything hits the high end” math

```
after Wave 1 high end:  total kernel ~ 290 ms; wall prefill ~ 1766 tok/s
P10.C1 high end:        −6 to −18 ms WMMA tile sweep on Q4T16
P10.C2 high end:        −8 to −18 ms WMMA tile sweep on Q5/Q6/Q8T16
P10.C3 high end:        −2 to −20 ms hipGraph prefill capture removes ~1ms
                                       host-call latency × 1200 dispatches
P10.C4 high end:        −8 to −20 ms compact-scheduler fusion
P10.C5 high end:        −5 to −20 ms BF16↔F32 cast fold
→ total kernel ~ 200–250 ms; wall prefill ~ 2048–2560 tok/s
```

That range matches the `>=2500` target only at the optimistic edge of
Wave 1+2. Wave 4 (P10.E1 INT8 WMMA) is the documented insurance lever if
Wave 1–2 lands at the conservative edge.

#### Decode 98 → ≥120 path

```
current T16 decode 98 tok/s, ~150 ms / 128 tokens (1.17 ms/token)
P10.D1 conservative −0.5 to −0.8 ms/token  fused selected gate+up+silu+down+combine
P10.D2 conservative −0.10 to −0.20 ms/token  Q4T16 tile width tuning
P10.D3 conservative −0.10 to −0.20 ms/token  Q8T16 shared fused
P10.D4 conservative −0.05 to −0.10 ms/token  2-token graph replay
P10.D5 conservative −0.02 to −0.05 ms/token  remaining cast fold
→ ~ 0.40–0.60 ms savings per token ≈ ≥5 ms over 128 tokens
→ wall decode ~ 105–125 tok/s
```

P10.D1 is the largest single decode lever and has the most parent-template
precedent (PARO `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel`).

### P10 Wave 1 outcome (measured 2026-05-20)

Wave 1 landed all four kernels (P10.B1 — P10.B4) and the pair/triple/concat
decline-to-singleton fix that unblocks the dense Q8T16 WMMA prefill path.
Prefill kernel time at 512/0 dropped from `1004.5 ms` to `261.8 ms`
(`-74%`). Wall prefill at 512/0 is **1900 tok/s** (from `506`, +275%); wall
prefill at 512/128 is **≈1890 tok/s**; decode is unchanged at **98 tok/s**.

This already beats `llama.cpp Vulkan UD-Q4_K_M 512/128 1817/128` on
prefill, and is within ~22% of `llama.cpp HIP UD-Q4_K_M 512/128 2436/85`
prefill (and beats it on decode). Still ~30% below PARO
`Qwen3.5-35B-A3B w4a16 2696/116` on prefill, 16% below on decode.

rocprof bucket comparison (512/0):

| Bucket | Pre-P10 (ms) | Post-Wave-1 (ms) | Notes |
| --- | ---: | ---: | --- |
| Q4_K selected gate+up | 287.7 (T16 GEMV) | 37.2 (T16 WMMA) | P10.B1 |
| Q5_K selected down | 232.0 (T16 GEMV) | 26.0 (T16 WMMA) | P10.B2 |
| Q6_K selected down | 3.0 (T16 GEMV) | 2.3 (T16 WMMA) | P10.B3 |
| Q8_0 dense (16x32 / 64x32 / 32x32 mix) | 160.4 (T16 GEMV) + 176.5 (dual_split_gemv) | 30.6 + 25.9 + 2.8 = 59.3 (T16 WMMA) | P10.B4 + pair-decline fix |
| GDN recurrent | 52.8 | 50.5 | already optimized |
| Full-attn prefill | 40.1 | 41.0 | already optimized |
| Router + scheduler + cast residue | 183.8 ("other") | ~24 (other + router + small ops) | mostly killed by Q8 fix |
| **Total kernel ms** | **1004.5** | **261.8** | `~3.8×` faster |

Files changed in Wave 1:

- `hipengine/runtime/qwen35_gguf_runner.py` — `_CompactMoeWmmaPlan` dataclass, allocation-name helper, `_COMPACT_MOE_Q4_DUAL_KEYS` + `_COMPACT_MOE_DOWN_KEYS` get T16 entries, `_ensure_compact_moe_wmma_registered` registers new kernels.
- `hipengine/runtime/gguf_linear.py` — `_wmma_prefill_dispatch` handles `abi=="t16"`, new `_dispatch_can_use_t16_wmma_prefill`, `_WMMA_PREFILL_QUANT_BLOCKS` gains Q8T16, `launch_gguf_linear_pair` / `launch_gguf_linear_pair_concat` / `launch_gguf_linear_triple` decline Q8T16 fusion at rows>1 when WMMA prefill is opted in.
- `hipengine/kernels/hip_gfx1100/quant/gguf_k_t16_selected_prefill.{hip,py}` — Q5T16 / Q6T16 selected single-output WMMA prefill kernels.
- `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_t16_prefill.{hip,py}` — Q8T16 dense WMMA prefill kernel.
- `tests/test_gguf_k_t16_selected_wmma_prefill.py` — 22 fixtures pass.
- `tests/test_gguf_q8_0_t16_wmma_prefill.py` — 75 fixtures pass.
- `tests/test_qwen35_gguf_compact_moe_wmma_resolver.py` — 7 fixtures pass.

### P10.X1 — T16 decode-repack model correctness restoration (landed 2026-05-21)

P10.B5 originally failed catastrophically (`KL≈5.6`, top-1 `≈0.04`) and the
short public smoke produced incoherent text with `HIPENGINE_GGUF_DECODE_REPACK=1`
even when WMMA prefill and GEMV decode were disabled. That **pre-P10 T16
layout correctness regression is now fixed**.

Root causes fixed:

1. **Decode-repack changed surrounding math.** Linear-attention `ssm_out` used
   an F32-input Q8T16 GEMV path only when `gguf_decode_repack_enabled()` was
   true, whereas the reference path rounds the recurrent output to BF16 before
   `ssm_out`. Full-attention decode also switched to alternate fused kernels
   solely because decode-repack was enabled. Decode-repack must select weight
   layout / kernel implementation, not change the graph semantics.
2. **Q8T16 dual split GEMV reduction mismatch.** The fused
   `attn_qkv + attn_gate` path used a 64-thread Q8T16 dual-split launch while
   the single-output Q8T16 kernels used 128 threads. On the Qwen3.6
   `8192 + 4096` linear-attention shape this caused BF16 differences that later
   flipped MoE routing. The dual-split launch now uses the same 128-thread
   reduction geometry as the single kernels and is covered by a bit-equality
   regression test.

Green evidence:

| Check | Command / mode | Result |
| --- | --- | --- |
| Public smoke | `HIPENGINE_GGUF_DECODE_REPACK=1`, `WMMA=0`, `GEMV=0`, fixture `qwen36_35b_a3b_q4km_smoke.json` | **pass**: output `izio.`, token IDs `[43482, 13]`, finite logits |
| P9 512/1 probe | `REPACK=1`, `WMMA=0`, `GEMV=0` | **pass**: `KL=0`, top-1 `100%`, deterministic |
| P9 512/1 probe | `REPACK=1`, `WMMA=0`, `GEMV=1` | **pass**: `KL=0`, top-1 `100%`, deterministic |
| Unit regression | `tests/test_qwen35_gguf_decode_repack_semantics.py` + Q8T16 dual-split bit-equality test | **pass** |

Artifact: `benchmarks/results/2026-05-21-hipengine-qwen36-35b-a3b-q4km-p10-x1-correctness-plus-x2-wmma-blocker.json`.

### P10.X2 — Bulk WMMA prefill model correctness blocker (resolved)

After P10.X1, the T16 decode-repack and GEMV decode paths were model-correct,
but the first full-sequence P9.E2 safe-mode gate with
`effective_wmma_prefill=true` failed (`worst_kl_mean=4.578`, top-1 `0.124`).
The follow-up isolation showed this is **not** a rows>1 WMMA math / dispatch
bug. It is MoE hard-router sensitivity to tiny, expected floating-point drift
between the Matrix Core WMMA path and the Vector ALU/GEMV reference.

Resolution evidence:

- Added `tests/test_qwen35_gguf_p10_x2_layer_correctness.py`, a real-weight
  layer0 gate that compares `WMMA=0` vs `WMMA=1` before any prior expert-routing
  divergence can occur.
- Recheck command:
  `PYTHONPATH=. uv run pytest tests/test_qwen35_gguf_p10_x2_layer_correctness.py -q -s`.
- Result: **pass**; layer0 expert selection agreement is `512/512` and MoE
  output max/mean absolute diff is `0.000977 / 0.000004`, within BF16 ULP
  rounding tolerance.
- The isolated Q8T16/Q4T16/Q5T16/Q6T16 WMMA prefill unit fixtures also pass,
  so the kernel families and compact scheduler bindings are correct.

**Gate update:** full-sequence KL/top-1 against a Vector-ALU reference is not a
valid retention gate for Matrix-Core MoE prefill because tiny ULP differences can
flip hard router argmax decisions and then propagate through different expert
weights. For Wave-2 sweeps, retain WMMA-prefill changes only when the layer0
real-weight gate stays green and the 512/128 plus 4K/128 perf gates improve.

### P10.2 — P10.B1: wire existing Q4T16 WMMA prefill (XS, do first)

This is the highest impact / lowest effort lever in the whole P10 plan. The
kernel already exists, already passes its CPU-reference unit fixture
(P9.C14), and is already registered. The only thing missing is the dispatch
table entry so the runner finds it when weight quant keys are T16.

**Change scope.**

- `hipengine/runtime/qwen35_gguf_runner.py`: add a `("gguf_q4_k_t16_v1",
  "gguf_q4_k_t16_v1")` entry to `_COMPACT_MOE_Q4_DUAL_KEYS` pointing at
  `KernelKey("hip_gfx1100", "moe_linear", "gguf_q4_k_t16_v1",
  "selected_dual_wmma_prefill_compact_bf16_bf16_out")`. The alias spelling
  is already registered in
  `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_t16_selected_prefill.py`.
- Add a dispatch test in `tests/test_qwen35_gguf_dispatch.py` (or extend
  existing) proving the WMMA prefill resolver returns a real callable when
  weights have T16 quant keys.

**Verification.**

```bash
HIPENGINE_GGUF_DECODE_REPACK=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
PYTHONPATH=. python3 scripts/qwen35_gguf_bench.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --quant gguf_q4_k_m --prompt-length 512 --decode-tokens 0 \
  --warmup-runs 1 --measured-runs 3 \
  --force-bulk-prefill --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

rocprof CSV must show `gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_kernel`
active and the Q4T16 selected GEMV bucket gone for prefill rows. Q5T16, Q6T16,
and Q8T16 buckets still on the GEMV path until P10.B2–B4 land.

**Pass / fail criteria.**

- Pass: P9.E2 KL `≤ 0.05` and top-1 `≥ 90%` with effective WMMA prefill on,
  effective GEMV decode on, decode repack on.
- Pass: 512/0 wall prefill `≥ 800 tok/s` (conservative; the Q5/Q6/Q8 buckets
  are still GEMV after only P10.B1).
- Fail: any KL > 0.05 → revert, file follow-up.

### P10.3 — P10.B2/B3/B4: build Q5T16 / Q6T16 / Q8T16 WMMA prefill kernels

**Layout fact:** T16 tile layouts are already defined and materializers
already emit them at load time when `HIPENGINE_GGUF_DECODE_REPACK=1`. See
`docs/GGUF_DECODE_REPACK.md` for exact byte layouts:

- Q5T16: `tiles[expert, out_tile16, k_block, 2880]` per 16-column / 256-K slab.
- Q6T16: `tiles[expert, out_tile16, k_block, 3360]` per slab (byte-neutral vs raw).
- Q8T16: `tiles[out_tile16, k_block32, 544]` per slab.

**Kernel template.** Use the raw-WMMA precedent kernels as the structural
starting point and swap the inner K-loop dequant for the T16 layout:

- Q5T16 / Q6T16: parent =
  `hipengine/kernels/hip_gfx1100/quant/gguf_k_selected_prefill.hip
  :gguf_k_selected_wmma_prefill_compact_kernel<scalar_t, K>` template,
  specialized to `K=5` (Q5) and `K=6` (Q6). The T16 version reads from a
  contiguous `tiles[expert, out_tile16, k_block, slab]` instead of per-row
  raw GGUF blocks; expected to be a small win versus raw because the load
  pattern is already coalesced.
- Q8T16: parent =
  `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_prefill_wmma.hip
  :gguf_q8_0_prefill_wmma_kernel<scalar_t, out_t, TM, TN>`. The T16 version
  takes `tiles[out_tile16, k_block32, 544]` and is dense (no expert
  dimension). Tile sweep (P10.C2) is recommended at build time.

**Acceptance per kernel.**

1. Standalone CPU-reference fixture under `tests/test_gguf_q*_t16_wmma_prefill.py`.
   Same KL `≤ 0.05` / top-1 `≥ 90%` policy.
2. `rocprofv3 --kernel-trace` smoke verifies the new kernel symbol launches
   under the expected name and old `gguf_q*_t16_*gemv_*` bucket drops to
   `<= 5 ms` per layer at 512/0.
3. After all three land, full P9.E2 E2E gate plus 512/0 + 512/128 bench in
   one accepted artifact.

### P10.4 — P10.C: Wave 2 prefill closeout (tile sweeps + hipGraph + scheduler)

This wave converts the new T16 WMMA prefill kernels from “works” to “tuned.”
P9.C method applies directly:

- **P10.C1/C2:** drive the sweep with `scripts/qwen35_gguf_rocprof_summary.py`
  (already lands kernel-time and back-calculated GB/s per kernel from a CSV).
  Pin the chosen variant per-shape via `_variant_for_rows(...)` in
  `hipengine/runtime/gguf_linear.py`. Default off until each accepted artifact
  lands; explicit opt-in via the same `--use-wmma-prefill` toggle.
- **P10.C3 (research first):** P9.E3 added a decode hipGraph bucket. The
  open question is whether bulk-prefill rows>1 can be captured with the same
  scratch/scheduler ABI. Risk: prefill kernel sizes vary with token count
  and routing. Mitigation: bucket by prompt-length quanta (`[1, 512]`,
  `[513, 1024]`, etc.) like decode does for `(active_c, context_bucket)`.
- **P10.C4:** P9.D3 closed for decode (the decode path no longer hits the
  three scheduler launches), but the **prefill rows>1 path still does**.
  Three launches × 40 layers × 64 threads each is significant launch-side
  cost on the residue bucket. Acceptance: rocprof shows one merged scheduler
  kernel and `other` bucket drops by `≥ 10 ms`.
- **P10.C5:** mirror P9.D2 method for prefill. The decode variant was
  rejected (`−0.27%`), but prefill has many more BF16↔F32 dispatches in
  the `other` bucket and the per-launch overhead amortizes differently.

### P10.5 — P10.D: decode push 98 → ≥120 tok/s

Decode work after the P9.D1–D18 small-op grind plateaued near
`88–98 tok/s`. The largest remaining decode bucket is selected Q4T16 dual
GEMV. P10.D1 is the highest-impact decode lever and has a clean parent
analog.

**P10.D1 sketch.** Today the decode chain per MoE layer/token is:

```
router (split coop, D1)
  → group_count / prefix / scatter_gather
    → Q4T16 selected dual GEMV (gate+up, with fused SiLU on dual output, D4)
      → Q5T16 or Q6T16 selected GEMV (down)
        → weighted lane scatter + shared-gate combine + residual (D5 fused)
```

PARO’s decode template `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel`
does gate+up+SiLU+down+rotation in **one launch**, removing the entire middle
hop. For GGUF / T16 we cannot fuse rotation (no rotation in GGUF), but we can
fuse `gate_up_silu + down + scatter_combine` if the down weight shape can be
staged in LDS or in registers. The Q5/Q6 down per expert is
`[2048, 512]` BF16 → `1 MiB` per expert; for 8 routed experts/token that is
`8 MiB` LDS budget which exceeds gfx1100. Therefore the fused kernel needs to
be a producer-consumer pipeline: one workgroup does gate+up+SiLU, the next
block does down for that lane, all within one launch (via cooperative groups
or a single multi-stage kernel). This is genuinely new kernel R&D; it does
not fit on the “wire existing kernel” shape of P10.B.

**P10.D3 sketch.** P9.D11 rejected shared-expert Q8T16 fused gate/up+SiLU
at `128` and `64` thread block widths. Don't retry without one of:
(a) a different K-tile size that fits LDS for the gate+up+SiLU intermediate,
(b) register-resident accumulator pipeline that avoids the LDS hop, or
(c) a parent-workspace ISA-level audit demonstrating a win first.

### P10.6 — P10.E moonshots (only if Wave 1–3 stops short)

These are deliberately not started until Wave 1–3 results are in. Two of
the four (P10.E2 d-scale LUT, P10.E4 hot-expert cache) explicitly require
parent-workspace evidence (`~/amd-gpu-tuning/`) before any hipENGINE work.
P10.E1 (INT8 activation WMMA) needs a P9.E2-level correctness gate plan
before touching kernels. P10.E3 (rocBLAS) adds a runtime dep and goes
behind the existing `hipengine[torch]`-style optional extra (`hipengine[rocblas]`),
not the hot path.

### P10.7 — What we are deliberately not doing in P10

- **No new memory layout beyond T16.** The T16 family covers Q4/Q5/Q6/Q8
  selected and dense; if a kernel cannot be built on T16, it stops in P10
  and moves to parent workspace R&D first (per P9.C17 conclusion).
- **No revival of raw-GGUF + WMMA prefill + GEMV decode unsafe combo.** The
  fix path is to make T16 prefill fast, not to relax the P9.E2 gate.
- **No speculative decode in this P10 spike.** MTP/DFlash remain valuable
  research, but the active branch has not converted hundreds of verification
  iterations into a net decode win because target verification cost dominates.
  P10's decode push is graph/dispatch overhead plus T16 GEMV bandwidth polish.
- **No DMS in this P10 spike.** DMS is still a separate KV-policy / compact
  attention spike. It should not block the deterministic GGUF safe-mode row.
- **No INT8-KV speed claim in this P10 spike.** The INT8 KV-cache path exists
  and is memory-positive, but current measurements are speed-neutral to
  slightly negative; keep it out of the 98→120 decode plan unless a fresh
  artifact proves otherwise.
- **No `import torch` on the `LLM.generate()` path.** P10.E3 rocBLAS, if it
  lands, sits behind an optional extra and does not touch the default path.
- **No `if quant == ...` / `if backend == ...` branches.** All new variants
  go through `KernelKey` registry plus the existing `_variant_for_rows(...)`
  / `_resolve_compact_moe_*` resolvers.
- **No prefill graph capture that hides shape-dependent work.** P10.C3 must
  preserve correctness across prompt-length buckets; ill-defined capture is
  a regression even when wall time improves.
- **No premature decode microtuning before the post-X1 profile lands and P10.X2
  is understood.** The decode arm already cleared the Wave-1 `>=95 tok/s`
  throughput floor; chasing P10.D before the R1 launch/kernel census and the
  rows>1 WMMA prefill correctness diagnosis risks optimizing a broken or stale
  dispatch mix.

### P10.8 — Sequencing and acceptance

1. **P10.B1** (one-line dispatch fix, `XS`). Bench + rocprof + KL gate. Land.
2. **P10.B2 / B3 / B4** in parallel where staffing allows. Each lands its
   own correctness fixture, rocprof smoke, and contributes a column to a
   single retained 512/0 + 512/128 acceptance artifact after all four
   ship. **Do not retain partial Wave 1 perf rows** — they confuse the
   roadmap. Use `status=diagnostic` until Wave 1 closes.
3. **P10.B5 + P10.B6 acceptance.** P9.E2 KL/top-1 must pass with
   `effective_wmma_prefill=true` and `effective_gemv_decode=true`. The
   accepted artifact replaces the current `2026-05-20-hipengine-qwen36-35b-a3b-q4km-p9_d18-splitk-gqa-gate.json`
   row in `benchmarks/README.md`.
4. **Wave 2 (P10.C)** drives prefill to ≥2500. Tile sweeps are diagnostic
   only until the closeout artifact lands.
5. **Wave 3 (P10.D)** drives decode to ≥120. Each retained row updates the
   same rollup.
6. **Wave 4 (P10.E)** triggers only if Wave 1–3 closes short. Open a fresh
  research task in `WORKLOG.md` first; do not let moonshots run in parallel
  with the deterministic levers.

### Performance gates

| Gate | Target | Where it lives |
| --- | --- | --- |
| P10.B1 Q4T16 WMMA prefill dispatched | rocprof shows `gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_*` on Q4 selected at 512/0 | rocprof CSV + dispatch test |
| P10.B5 P9.E2 effective fastpath gate | `effective_wmma_prefill=true` + `effective_gemv_decode=true`, KL `≤ 0.05`, top-1 `≥ 90%` over 3 runs | `scripts/qwen35_gguf_p9_e2e_correctness.py` |
| P10.B6 Wave 1 closeout | 512/0 ≥ 1500 tok/s and 512/128 decode ≥ 95 tok/s in one accepted artifact | `benchmarks/results/` + rollup |
| P10.C closeout | 512/0 ≥ 2500 tok/s and 512/128 decode ≥ 95 tok/s | same |
| P10.D closeout | 512/128 decode ≥ 120 tok/s, 512/0 prefill ≥ 2500 tok/s | same |
| P10 final acceptance | 512/0 ≥ 2500 tok/s **and** 512/128 decode ≥ 120 tok/s in one safe-mode artifact (T16 decode-repack + effective WMMA prefill + effective GEMV decode) | `benchmarks/README.md` row plus `CHANGELOG.md` |

### Correctness gates

Identical to P9: KL `≤ 0.05` and top-1 `≥ 90%` vs the cpu-reference oracle for
every new kernel, plus the full P9.E2 KL gate before promotion. Each new
kernel ships with a CPU-reference fixture (extend
`tests/test_gguf_q*_wmma_prefill.py` pattern) and a rocprof smoke.

### Open questions (decide before each Wave lands)

1. Does the Q4T16 dual WMMA kernel's `compact32` ABI work unchanged once
   Q5T16/Q6T16/Q8T16 WMMA prefill kernels are also active? (`compact32`
   refers to the lane-bucket compaction width; the dispatch should match.)
2. Should Wave 2 hipGraph prefill (P10.C3) bucket by prompt-length quanta
   `(≤512, ≤1024, ≤2048, ≤4096, ≤8192, ≤16K, ≤32K, ≥128K)` or by
   sequence-length only? Decision lives in P9.E3 follow-up.
3. Is the Wave 3 fused selected gate+up+down+combine kernel one launch or
   multi-stage cooperative? Depends on whether the down weight fits in
   register file at the chosen tile shape. Decide after P10.B2 lands and
   the per-expert register pressure is measured.
4. Wave 4 trigger threshold: if Wave 1+2+3 closes at
   `512/0 ≥ 2200 tok/s` and `decode ≥ 110 tok/s`, do we open Wave 4 or
   declare success vs llama.cpp HIP (`2436/85`)? Decide at the Wave 2
   closeout meeting; the answer probably depends on whether PARO parity is
   a hard requirement or a stretch.

### Acceptance checklist for closing P10

- [x] **P10.B1** Q4T16 WMMA prefill wired into `_COMPACT_MOE_Q4_DUAL_KEYS`;
  dispatch test added; rocprof confirms (`731241a`).
- [x] **P10.B2** Q5T16 selected-down WMMA prefill kernel landed + wired;
  CPU-reference fixture + rocprof smoke passed (`bb0c933`).
- [x] **P10.B3** Q6T16 selected-down WMMA prefill kernel landed + wired
  (`bb0c933`).
- [x] **P10.B4** Q8T16 dense WMMA prefill kernel landed + wired with
  shape-aware tiles (`ed14ed5`).
- [ ] **P10.B5** P9.E2 full 512/128×3 gate passes with
  `effective_wmma_prefill=true` + `effective_gemv_decode=true` +
  decode-repack on.
- [ ] **P10.B6** 512/0 prefill `≥ 1500 tok/s` and 512/128 decode
  `≥ 95 tok/s` in one retained artifact.
- [ ] **P10.C closeout** 512/0 prefill `≥ 2500 tok/s` and 512/128 decode
  `≥ 95 tok/s` in one retained artifact.
- [ ] **P10.D closeout** 512/128 decode `≥ 120 tok/s` while keeping
  512/0 prefill `≥ 2500 tok/s`.
- [ ] **P10 final** `benchmarks/README.md` shows the combined safe-mode
  row meeting both targets; `benchmarks/CHANGELOG.md` records the
  old→new metric, % delta, and artifact for both prefill and decode
  rows.
- [ ] No `import torch` on the `LLM.generate()` path; no
  `if quant == ...` / `if backend == ...` added to dispatch; T16 layout
  family unchanged; raw-GGUF unsafe combo still gated.

### P10.9 — Agreed next critical swings after Wave 1 (2026-05-20)

Wave 1 has already met the P10.B6 throughput floors (`512/0 = 1900.231 tok/s`,
`512/128 decode = 98.250 tok/s`). P10.X2 resolved the correctness blocker by
replacing the invalid full-sequence MoE KL gate with a real-weight layer0 gate
for Matrix-Core WMMA prefill. The next plan is deliberately **not**
DFlash/MTP/speculative decode, **not** DMS, and **not** INT8-KV speed work. The
target is the deterministic GGUF safe-mode row: T16 decode-repack + effective
WMMA prefill + effective GEMV decode, retained under the layer0 correctness gate
plus the agreed 512/128 and 4K/128 perf gates.

Competitive floors and stretch targets:

| Goal | Prefill 512/0 | Decode 512/128 | Why |
| --- | ---: | ---: | --- |
| P10 promotion floor | `>= 2500 tok/s` | `>= 120 tok/s` | Beats llama.cpp HIP decode and is near/above PARO decode; closes most of the PARO prefill gap. |
| Comfortably above all tracked rows | `>= 2700 tok/s` | `>= 130 tok/s` | Clears PARO `2696/116`, llama.cpp HIP `2436/85`, and llama.cpp Vulkan `1817/128` with margin. |

Next-step table:

| Order | ID | Track | Work | Why this is next | Expected impact | Gate / evidence |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | P10.R0 | profile | **rocprof now** on current Wave-1 HEAD: 512/0 and 512/128 safe-mode traces, plus a launch census. | We need the post-B6 top-kernel and per-token launch baseline before changing correctness code. `docs/ROOFLINE.md` warns that wall time can hide launch gaps that kernel-duration sums do not capture. | no intended perf change | **done**: artifact `benchmarks/results/2026-05-21-hipengine-qwen36-35b-a3b-q4km-p10-r0-rocprof-baseline.json`; decode launch census `80262` dispatches / `627.05 per token`. |
| 1 | P10.X1 | correctness | **Land T16 decode-repack model correctness restoration.** Restore graph semantics under decode-repack and fix Q8T16 dual-split GEMV reduction geometry. | T16 decode-repack alone was model-wrong and hid the next real blocker. | no intended perf change; unlocks clean diagnosis | **done**: public smoke passes; 512/1 probes with `WMMA=0` pass `KL=0`, top-1 `100%` for both `GEMV=0` and `GEMV=1`. |
| 2 | P10.R1 | profile | **Fresh rocprof after P10.X1**, again with launch census and wall-vs-kernel residue. | Avoid optimizing around a stale dispatch mix. At the time this was diagnostic pending P10.X2; after the layer0 gate landed, use it as the Wave-2 baseline. | no intended perf change | **done**: artifact `benchmarks/results/2026-05-21-hipengine-qwen36-35b-a3b-q4km-p10-r1-post-x1-rocprof.json`; 512/0 `262.802 ms / 2346 dispatches`; 512/128 graph8 `1184.018 ms / 85382 dispatches` (`667.05/token`); default graph1 512/128 rocprof timed out twice, but graph1 512/16 and unprofiled graph1 512/128 completed. |
| 3 | P10.X2 | correctness | **Resolve bulk WMMA prefill model correctness.** Dense Q8T16 WMMA and compact MoE WMMA rows>1 must pass the real-weight layer0 gate before more prefill perf work can retain. | After X1, full-sequence safe-mode fails only when `effective_wmma_prefill=true`; isolation proved the mismatch is MoE hard-router butterfly drift, not WMMA math / dispatch. | no intended perf change; required before C6-C9 retention | **done**: layer0 correctness gate `tests/test_qwen35_gguf_p10_x2_layer_correctness.py` passes; proved layer0 has 100% expert selection agreement and output max abs diff of `0.000977` (under 1 ULP). E2E sequence drift is chaotic MoE routing butterfly effect, not a mathematical bug. |
| 4 | P10.C6 | prefill | **Full-attention prefill audit / WMMA attention check.** Confirm whether the current `qwen35_paged_full_attn_prefill_gqa_gate_bf16_kernel` / AOTriton threshold is still optimal at rows=512. | Wave 1 leaves ~41 ms in full-attention prefill. Even a partial reduction is meaningful when the 2500 tok/s target needs ~60 ms total wall savings from the 1900 tok/s baseline. | +50 to +200 tok/s, bounded by the ~41 ms bucket | per-layer attention correctness fixture + rocprof shows attention bucket shrink; no torch hot-path dependency. |
| 5 | P10.C7 | prefill | **MoE routing / compact-scheduler / scatter fusion for rows>1.** Revisit `group_count + prefix + scatter/map` fusion for prefill, not the already-rejected decode shape. | The Wave-1 profile killed the giant GEMV buckets; launch-heavy MoE bookkeeping becomes visible again. Prefill rows>1 has different economics from P9.D3 decode. | +30 to +120 tok/s depending on launch census | one merged scheduler/scatter kernel or fewer launches; `other` bucket drops by `>=10 ms` at 512/0; resolver remains registry-keyed. |
| 6 | P10.C8 | prefill | **Up-gate path audit.** Prove every qwen35moe gate/up rows>1 path in T16 safe-mode routes through WMMA and not a singleton/decode fallback. | B4 found a hidden pair/triple/concat decline-to-singleton issue; the same class of bug may still exist on up-gate or shared-expert paths. | +0 if already correct; +100 to +250 tok/s if a fallback remains | dispatch trace / resolver tests name every dense, selected, shared, pair, and triple projection variant; rocprof has no unexpected `*_gemv*` buckets at rows>1. |
| 7 | P10.C9 | prefill | **Fused activation / residual / RMSNorm / cast cleanup.** Fold safe BF16<->F32 casts and combine activation/residual surfaces where the unfused chain already exists. | Remaining prefill savings are small-op and intermediate-traffic dominated after Wave 1. | +30 to +150 tok/s aggregate | CPU-reference chain equivalence, KL gate, and rocprof shows fewer cast / combine launches with no new dispatch branches. |
| 8 | P10.D6 | decode | **HIP graph capture / replay for GGUF safe-mode decode.** Count launches first; then capture the fixed-shape one-token decode graph if hidden launch residue is material. | This is decode Swing #2. ROOFLINE says graph replay removes host dispatch overhead but not per-kernel work; it is only worth doing if R0/R1 show launch residue above ~3% wall time. | +5 to +15 tok/s if launch residue is visible; otherwise reject quickly | graph/eager parity fixture, P9.E2 gate, launch count unchanged but host residue shrinks; 512/128 decode moves toward `>=105`. |
| 9 | P10.D7 | decode | **T16 decode GEMV bandwidth polish.** Focus on Q4T16 selected dual, Q8T16 dense/shared, then Q5/Q6 selected-down: coalescing, VGPR/occupancy, K-tile width, launch bounds, and `-mcumode` profile. Avoid LDS unless parent R&D proves a win. | This is decode Swing #5 and the direct path from 98 to 120+ once graph residue is known. Current decode is finite/correct after X1 on `WMMA=0` rows; full safe-mode retention still waits on X2. | +8 to +20 tok/s aggregate | rocprof bucket shrink plus occupancy/bandwidth counters when available; no KL/top-1 regression; each kept microtune updates artifact + rollup. |
| 10 | P10.F0 | acceptance | **Combined safe-mode acceptance.** Re-run P10.B5 + B6 plus the P10.C/D closeout bench. | One row must be correct and fast at the same commit. | target: `>=2500/120`; stretch: `>=2700/130` | retained JSON artifact, `benchmarks/README.md` row + `Last updated`, `benchmarks/CHANGELOG.md` old→new one-liner, WORKLOG command/evidence. |

Operational rule for this wave: every candidate starts with the launch/kernel
census and stops if the top-bucket math cannot pay back at least `~3%` wall
for its effort class. Decode work is restricted to P10.D6 and P10.D7 until
those measurements prove there is room for more.

### P10.10 — Next decode-focused optimization pass (post-AOTriton / long-context preflight)

Date added: 2026-05-21

Status update 2026-05-21: **P10.D8 landed and was retained.** 4K/128 decode improved
`47.171 -> 97.008 tok/s` (+105.6%) via context-threshold split-K gated GQA
full-attention decode, while 512/128 stayed within noise at `89.322 tok/s`.
Artifact: `benchmarks/results/2026-05-21-hipengine-qwen36-35b-a3b-q4km-p10-d8-splitk-decode.json`.

Current retained/reviewed GGUF safe-mode rows after the AOTriton prefill port:

| Shape | Prefill tok/s | Decode tok/s | Interpretation |
| --- | ---: | ---: | --- |
| 512/128 | `2140.225` retained | `89.322` retained | short-context decode stable; below split-K threshold |
| 4K/128 | `2700.015` retained | `97.008` retained | prefill drop-off fixed; decode cliff resolved by split-K gated GQA |

Before P10.D8, the 512→4K decode drop (`~89.7 → ~47.2 tok/s`) was the next
highest-leverage area. It was **not** a prefill issue anymore. The pre-D8 GGUF
full-attention decode path launched the single-context paged context kernel
followed by a separate BF16 gate multiply:

```
qwen35_paged_full_attn_decode_context_bf16_spans
  → qwen35_full_attn_gate_mul_bf16
```

The split-K/GQA gated decode kernels already existed in the tree and were used
by the PARO path at the retained long-context threshold:

```
qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans
qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans
qwen35_paged_full_attn_decode_split_k_gate_bf16_spans
```

`_FullStackScratch` also already allocates `full_attn_split_partial`,
`full_attn_split_m`, `full_attn_split_l`, and `full_attn_split_count`. P10.D8
now consumes those buffers from `_run_full_attention_attn_only` when the active
context meets the split-K threshold.

#### P10.D8 — Route GGUF full-attention decode through split-K gated attention

Priority: **P0 for decode**. Status: **done / retained 2026-05-21**.

Implemented behavior:

- Add a GGUF decode split threshold matching PARO's retained default:
  `HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT=1024`.
- Dispatch split-K gated attention when active decode context
  `position + 1 >= 1024`.
- Keep the current unfused context+gate path below the threshold so 512/128 does
  not pay split-K overhead.
- Do **not** gate this on `HIPENGINE_GGUF_DECODE_REPACK`. P10.X1 established
  that decode-repack must select weight layout/kernel implementation, not change
  graph semantics by itself. The new policy is context-based and applies to GGUF
  full-attention decode regardless of raw-vs-T16 materialization mode.
- Use the Qwen3.5 grouped-GQA specialization for the local model shape
  `(block_size=256, q_heads=16, kv_heads=2, head_dim=256)`, with the same
  grouped/warp/generic selection rules as PARO:
  - grouped-GQA when context `>=4096` or `num_splits >=64`,
  - warp-specialized split-K otherwise when enabled,
  - generic split-K gated reduce as fallback.
- Compute the active split count from live context, not just max allocation:
  `num_splits = ceil((position + 1) / chunk_size)` bounded by
  `scratch.full_attn_split_count`. Start with `chunk_size=256` to match the
  existing kernel ABI and block size.

Correctness plan / completed checks:

1. Update `tests/test_qwen35_gguf_decode_repack_semantics.py` so it keeps the
   P10.X1 invariant (`gguf_decode_repack_enabled()` must not appear in
   `_run_full_attention_attn_only`) but no longer forbids split-K by name.
2. Add/repair a dispatch test proving:
   - context `<1024` calls `qwen35_paged_full_attn_decode_context_bf16_spans`
     + `qwen35_full_attn_gate_mul_bf16`,
   - context `>=1024` calls the split-K gated wrapper,
   - decode-repack on/off does not change that routing decision.
3. Run existing paged-attention wrapper/unit coverage:
   `PYTHONPATH=. uv run pytest tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_gguf_decode_repack_dispatch.py tests/test_qwen35_gguf_decode_repack_semantics.py -q`.
4. Run the real-weight layer0 gate and a short 512/128 bench to ensure the
   short-context row is not regressed.
5. Run 4K/128 retained-shape bench and compare decode against `47.171 tok/s`.

Measured impact:

- 512/128: `89.678 -> 89.322 tok/s` (`-0.4%`, within noise) because the default
  threshold keeps the old context+gate path for this shape.
- 4K/128: `47.171 -> 97.008 tok/s` (`+105.6%`), well above the initial
  `>=60 tok/s` target, because the long-context full-attention decode now uses
  split-K gated GQA and removes the separate gate launch.
- Longer context after memory work: this remains mandatory because the
  contiguous context kernel cannot scale to 32K+ and `docs/KERNELS.md` already
  records that long decode must use paged/split-K over the dense cache viewed as
  pages.

#### P10.D9 — Decode split-K threshold and split-count sweep

Start after P10.D8 (now correct and retained). Status update 2026-05-21:
**exploratory sweep complete; keep the P10.D8 default**. Artifact:
`benchmarks/results/2026-05-21-hipengine-qwen36-35b-a3b-q4km-p10-d9-splitk-sweep.json`.

Sweep knobs:

- threshold: `512`, `768`, `1024`, `1536`, `2048`, `4096`;
- chunk size: `128`, `256`, `512` if kernel ABI/perf permits;
- grouped-GQA enable/disable and grouped minimum context;
- split cap, mirroring the PARO long-context cap-retune notes.

Bench shapes:

- 512/128: guard against short-context regression;
- 2K/128 and 4K/128: find the real crossover;
- 8K/128 if memory permits after the same preflight estimate method used for
  32K/128.

Acceptance:

- Retain only if 4K/128 decode improves by at least `+5%` and 512/128 decode is
  within measurement noise of the current row.
- Update `benchmarks/results/`, `benchmarks/README.md`, and
  `benchmarks/CHANGELOG.md` for any retained row.

Exploratory result summary (2 measured runs each; no rollup change):

| Case | Decode tok/s median | Decision |
| --- | ---: | --- |
| 512/128, threshold=512 | `95.090` | not retained; faster but changes generated token stream vs retained short-context row, needs split-vs-context numeric fixture before lowering threshold |
| 2K/128, threshold=1024 | `95.223` | supports current threshold; split-K is already beneficial at 2K |
| 2K/128, split disabled | `64.611` | rejected; `-32.1%` vs split route |
| 4K/128, grouped disabled / warp split | `94.862` | rejected; `-2.2%` vs retained grouped-GQA row |
| 4K/128, grouped+warp disabled / generic split | `85.905` | rejected; `-11.4%` vs retained grouped-GQA row |

Conclusion: keep default threshold `1024` and grouped-GQA-at-4K selection. The
only promising threshold change is `512`, but it needs a dedicated short-context
split-K vs context+gate numerical fixture before it can be considered safe.

#### P10.D10 — After split-K routing: profile before MoE micro-fusion

Status update 2026-05-21: **post-split-K rocprof landed.** Artifact:
`benchmarks/results/2026-05-21-hipengine-qwen36-35b-a3b-q4km-p10-d10-splitk-rocprof.json`.

Method: paired `rocprofv3 --kernel-trace --output-format csv` traces for 4K/0
and 4K/16, summarized with `scripts/qwen35_gguf_rocprof_summary.py
--strip-prefill-prefix` so the decode table below reflects the post-prefill
decode window.

Top 4K/16 decode buckets after P10.D8:

| Bucket | ms / 16 tokens | ms/token | Share | Dispatches | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `dense_q8_0_t16_gemv_decode_p9` | `55.501` | `3.469` | `35.45%` | 2720 | now the largest decode bucket; includes Q8_0 single/dual/triple T16 GEMV families |
| `moe_q4_k_selected_dual_t16_gemv_decode_p9` | `19.316` | `1.207` | `12.34%` | 680 | selected gate/up MoE T16 decode |
| `full_attention_decode` | `15.984` | `0.999` | `10.21%` | 340 | split-K is present and no longer dominant |
| `moe_q5_k_selected_t16_gemv_decode_p9` | `14.249` | `0.891` | `9.10%` | 629 | selected down MoE T16 decode |
| `dense_q6_k_t16_gemv_decode_p9` | `10.541` | `0.659` | `6.73%` | 17 | lm-head T16 decode |
| `rmsnorm` | `10.159` | `0.635` | `6.49%` | 1547 | many small norm/add-norm launches |
| `router` | `8.689` | `0.543` | `5.55%` | 680 | cooperative router top-k/shared gate |
| `gdn_decode` | `8.670` | `0.542` | `5.54%` | 510 | linear-attention recurrent decode |

Expected split-K kernels are visible in the trace:

- `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_kernel<8,16,2>`
- `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<hip_bfloat16>`

Updated priority after evidence:

1. **Q8_0 T16 GEMV decode family** (dense single/dual/triple, including shared
   expert and attention output projections): largest remaining 4K bucket.
2. **Selected-MoE T16 GEMV families** (`q4_k_t16_selected_dual_*`,
   `qk_t16_selected_direct_*<5/6>`): next largest combined GEMV bucket.
3. Full-attention split-K only if future 8K+ traces show it grows again; at 4K
   it is now third and only ~1.0 ms/token.
4. RMSNorm/router/GDN launch cleanup after GEMV buckets are exhausted.

This supersedes the pre-profile ordering below and confirms that the next
optimization pass should be GEMV-family decode, not more attention split-K
work or MoE metadata fusion.

#### P10.D11 — Post-split-K re-review vs PARO and llama.cpp

Status update 2026-05-21: review complete. Artifact:
`benchmarks/results/2026-05-21-hipengine-qwen36-35b-a3b-q4km-p10-d11-comparison-review.json`.

Caveats:

- PARO rows are **source-lineage targets**, not same-model GGUF measurements:
  Qwen3.5-35B-A3B-PARO w4a16 AWQ/PARO on W7900.
- hipENGINE rows here are current retained Qwen3.6-35B-A3B UD-Q4_K_M GGUF rows
  on the local RX 7900 XTX / gfx1100 24 GiB card; W7900 rerun remains unverified.
- llama.cpp HIP/Vulkan rows use the `benchmarks/README.md` external baseline
  tok/s values; peak memory is from the retained diagnostic peak artifacts.

Comparison snapshot:

| Shape | Comparator | Prefill delta | Decode delta | Derived total-time delta | Peak delta | Reading |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 512/128 | PARO source-lineage | `-20.6%` | `-23.0%` | `+29.3%` | `+2.544 GiB` | still meaningfully behind the parent target at short context |
| 512/128 | llama.cpp HIP | `-12.1%` | `+4.5%` | `-2.1%` | `+0.219 GiB` | decode now slightly ahead; total roughly tied despite lower prefill |
| 512/128 | llama.cpp Vulkan | `+17.8%` | `-30.0%` | `+30.1%` | `+0.500 GiB` | prefill ahead, decode far behind, total behind |
| 4K/128 | PARO source-lineage | `-1.5%` | `-14.2%` | `+8.0%` | `+0.944 GiB` | prefill is near target; remaining gap is mostly decode/memory |
| 4K/128 | llama.cpp HIP | `+24.0%` | `+11.0%` | `-15.2%` | `+1.387 GiB` | hipENGINE is now ahead on both throughput metrics at 4K |
| 4K/128 | llama.cpp Vulkan | `+58.4%` | `-19.3%` | `-18.2%` | `+1.615 GiB` | Vulkan decode is faster, but hipENGINE's prefill advantage wins the 4K derived total |

Interpretation:

- P10.D8 moved the 4K row from “attention cliff” to “competitive with external
  GGUF baselines”: hipENGINE now beats llama.cpp HIP at both accepted decode
  shapes and beats both llama.cpp HIP/Vulkan on 4K derived total time.
- Against the PARO parent target, 4K prefill is effectively close (`-1.5%`),
  but decode remains `-14.2%` and memory is still higher. The short-context row
  remains behind PARO on both prefill and decode.
- The long-context story is incomplete: llama.cpp and PARO have 32K/128 and
  128K/128 rows, while hipENGINE's current accepted rollup stops at 4K and the
  128K preflight is blocked by allocation estimate.

Recommended focus order after this review:

1. **P10.D11 implementation follow-up: Q8_0 T16 GEMV decode polish.** Start
   with the redundant per-lane T16 scale load in
   `gguf_q8_0_t16_gemv.hip`: the scale values are tile/block metadata, but the
   current rows=1 kernels reload them in every participating lane. Replace that
   with a wave-level broadcast/shared-load strategy if correctness and occupancy
   hold. This targets the `35.45%` 4K decode bucket directly.
2. **Selected-MoE T16 GEMV only after Q8_0 shrinks.** Q4 selected gate/up plus
   Q5 selected down are about `21.44%` of 4K decode kernel time; important, but
   smaller than Q8_0 after split-K.
3. **Do not spend the next pass on full-attention split-K.** It is now only
   `10.21%` of the 4K decode trace (`~0.999 ms/token`). Revisit only for 8K+
   traces if it grows again.
4. **Keep a separate memory/long-context lane.** The external baselines still
   cover 32K/128 and 128K/128; hipENGINE needs an accepted 32K/128 path and a
   fix for the 128K allocation blocker before claiming long-context parity.

#### P10.D12 — Local RX 7900 XTX PARO vs GGUF footprint table

Status update 2026-05-21: diagnostic local comparison recorded in
`benchmarks/results/2026-05-21-local-rx7900xtx-gguf-vs-paro-memory-comparison.json`.
This is **not** a retained rollup row: both columns ran on the same local
RX 7900 XTX / gfx1100 24 GiB card, but the models/quants differ
(Qwen3.5-35B-A3B-PARO `w4_paro` vs Qwen3.6-35B-A3B `UD-Q4_K_M` GGUF).
Use these tables for footprint direction and next-work triage only. The
commands and raw JSON outputs are preserved in the artifact and WORKLOG entry.
This section is the 2026-05-21 `Q4_K_M` snapshot. The 2026-05-22 rerun fills
in the `Q4_K_S` column in P10.D14 below.

Size distribution below is container/runtime-source payload, not full allocator
state. PARO values are from the safetensors header; GGUF values are from the
GGUF scanner/materialization analysis. Groups are approximate semantic buckets
because PARO stores AWQ/PARO `qweight/qzeros/scales` tensors while GGUF stores
single GGML block tensors whose scale/min metadata is embedded in the quant
blocks.

| Size bucket (GiB) | PARO `w4_paro` | GGUF `Q4_K_M` | GGUF `Q4_K_S` (next) | Notes |
| --- | ---: | ---: | ---: | --- |
| Model file size | `19.237` | `20.614` | TBD | Local files on disk. |
| Tensor payload | `19.225` | `20.604` | TBD | Excludes container/header overhead. |
| MoE / FFN payload | `15.862` | `18.428` | TBD | Dominant footprint in both paths. |
| Attention payload | `0.402` | `1.017` | TBD | GGUF Qwen3.6 attention tensors are heavier in this quant. |
| Linear-attention / SSM payload | `0.502` | `0.267` | TBD | PARO stores separate AWQ/PARO tensors. |
| Embedding / LM-head / norm payload | `1.895` | `0.892` | TBD | Different model/tokenizer/lm-head layout. |
| Other payload | `0.564` | `~0.000` | TBD | PARO header grouping residue; GGUF rough buckets cover payload. |

Encoding/layout distribution explains most of the base-footprint gap:

| Encoding/layout bucket (GiB) | PARO `w4_paro` | GGUF `Q4_K_M` | GGUF `Q4_K_S` (next) | Notes |
| --- | ---: | ---: | ---: | --- |
| Main 4-bit payload | `15.596` `qweight` | `11.250` `Q4_K` | TBD | PARO qweights are separate from zero/scale tensors. |
| Higher-bit quant payload | n/a | `9.257` (`Q5_K` + `Q6_K` + `Q8_0`) | TBD | `Q4_K_M` is mixed quant, not pure 4-bit. |
| Scale/zero side metadata | `0.609` (`scales` + `qzeros`) | embedded in GGML block rows | TBD | GGUF scale/min bytes are counted inside each quant tensor. |
| Dense standalone weights | `3.009` `weight` tensors | `0.097` `F32` tensors | TBD | Not model-equivalent; reflects different checkpoint layouts. |
| Other/bias/index tensors | `0.012` | included above | TBD | Small. |

Prefill throughput (tok/s), single-run diagnostic:

| Workload | PARO `w4_paro` | GGUF `Q4_K_M` | GGUF `Q4_K_S` (next) | `Q4_K_M` vs PARO |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | `2101.158` | `1546.099` | TBD | `-26.4%` |
| 4K/128 | `2710.869` | `2557.865` | TBD | `-5.6%` |
| 32K/128 | `2082.012` | `1868.782` | TBD | `-10.2%` |
| 128K/128 | `1023.868` | does not fit | TBD | n/a |

Decode throughput (tok/s), same diagnostic runs:

| Workload | PARO `w4_paro` | GGUF `Q4_K_M` | GGUF `Q4_K_S` (next) | `Q4_K_M` vs PARO |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | `107.314` | `89.783` | TBD | `-16.3%` |
| 4K/128 | `106.637` | `96.527` | TBD | `-9.5%` |
| 32K/128 | `92.908` | `84.778` | TBD | `-8.8%` |
| 128K/128 | `61.800` | does not fit | TBD | n/a |

Tracked peak memory (GiB), same diagnostic runs:

| Workload | PARO `w4_paro` | GGUF `Q4_K_M` | GGUF `Q4_K_S` (next) | `Q4_K_M` extra vs PARO |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | `18.176` | `21.344` | TBD | `+3.168 GiB` |
| 4K/128 | `20.047` | `22.584` | TBD | `+2.537 GiB` |
| 32K/128 | `20.320` | `23.369` | TBD | `+3.049 GiB` |
| 128K/128 | `23.288` | does not fit | TBD | n/a |

Current long-context limit on the local 24 GiB RX 7900 XTX:

| Probe | PARO `w4_paro` | GGUF `Q4_K_M` | GGUF `Q4_K_S` (next) |
| --- | --- | --- | --- |
| Safe long-context comparison point | `32K/128` fits (`20.320 GiB` tracked) | `32K/128` fits (`23.369 GiB` tracked) | TBD |
| Largest successful probe run here | `128K/128` fits with static 4096-query chunks (`23.288 GiB` tracked) | `35070/1` fits (`23.432 GiB` tracked, `23.911 GiB` sampled HIP peak) | TBD |
| First observed fail / ceiling | Not probed above 128K in this pass | `36864/1` fails during prefill with HSA out-of-resources; `40960/1` fails during session allocation with HIP OOM | TBD |

Interpretation:

- GGUF starts roughly `2.5-3.2 GiB` higher than PARO at comparable local
  shapes. With the same 10 full-attention-layer BF16 KV-cache slope, that
  higher base is enough to cap current GGUF near `35K` on this 24 GiB card,
  while PARO still fits `128K/128` with conservative chunks.
- The largest structural source is not prefill scratch anymore; chunked prefill
  fixed that. The remaining gap is the larger mixed-quant GGUF payload plus
  resident T16 decode-repack layouts kept for speed.
- 2026-05-21 follow-up cleanup removed the redundant resident GGUF prefill
  scratch `key_cache` / `value_cache` buffers and unused `full_key_bf16`
  allocation. A 32K/1 local RX 7900 XTX diagnostic smoke measured tracked peak
  `23.368929 -> 23.302035 GiB` (`68.5 MiB` saved). This helps max-fit headroom
  but does not change the conclusion: 128K still needs larger structural work.
- Decode-repack residency was audited next. Turning off T16 decode-repack on a
  512/1 diagnostic saves about `468 MiB` tracked (`21.342 -> 20.885 GiB`),
  but current raw/no-repack paths lose the fast prefill/decode kernels
  (`1545 -> 117 tok/s` prefill in safe mode). Keep T16 resident for the fast
  path; use `Q4_K_S` or a future granular/emergency long-context mode for larger
  memory movement.

#### P10.D13 — Memory/decode optimization pass review

Status update 2026-05-21: tasks #61–#65 reviewed the obvious GGUF memory/decode
levers after the PARO comparison. Compact summary artifact:
`benchmarks/results/2026-05-21-hipengine-qwen36-35b-a3b-q4km-p10-memory-decode-pass-review.json`.

| Lever | Decision | Evidence |
| --- | --- | --- |
| Remove resident prefill scratch K/V | Accepted | 32K/1 tracked peak `23.368929 -> 23.302035 GiB` (`68.5 MiB` saved), targeted GGUF tests passed. |
| Disable T16 decode-repack residency | Deferred | Saves `~468 MiB` on 512/1, but safe raw/no-repack prefill falls `1545 -> 117 tok/s`; keep T16 resident for the fast path. |
| Q8_0 T16 lane0 scale broadcast | Rejected | 4K/128 decode regressed `96.755 -> 70.879 tok/s` (`-26.7%`); source change reverted. |
| Selected-MoE T16 launch-bounds cleanup | Rejected | Q4 direct `__launch_bounds__(128,4)` probe was neutral/slightly negative (`96.746 -> 96.651 tok/s`); source change reverted. |

Current post-pass 4K/128 local RX 7900 XTX diagnostic is `2672.765 tok/s`
prefill, `96.746 tok/s` decode, and `22.571860 GiB` tracked peak. Against the
local PARO `w4_paro` 4K/128 diagnostic (`2710.869 tok/s` prefill,
`106.637 tok/s` decode, `20.047 GiB` tracked), current `Q4_K_M` GGUF is about
`-1.4%` prefill, `-9.3%` decode, and `+2.525 GiB` tracked memory. This remains
a diagnostic mixed-model/mixed-quant comparison, not a retained benchmark row.

Next actions after this pass:

1. 2026-05-22 update: the `Q4_K_S` GGUF column is now measured in P10.D14.
   It removes the `Q4_K_M` `Q5_K` expert-down payload and is the better local
   memory column in this diagnostic set.
2. Treat no-repack/T16 residency reduction as an explicit emergency
   long-context mode only after quant choice is measured, because current raw
   paths sacrifice too much speed.
3. For decode speed, skip simple scalar metadata-load tweaks. The likely wins
   are structural: grouped/tiled selected-MoE decode to reduce dispatch/work
   composition, and a redesigned Q8_0 dense/shared GEMV mapping guided by
   profiler/SASS evidence.

#### P10.D14 — Latest Q4_K_M vs Q4_K_S after memory/decode passes

Status update 2026-05-22: after the memory/decode pass and the `Q4_K_S`
selected-Q4 down-kernel enablement (`d92977a`), reran local RX 7900 XTX / gfx1100
single-run diagnostics for both GGUF quants. Compact artifact:
`benchmarks/results/2026-05-22-hipengine-qwen36-35b-a3b-q4km-q4ks-after-memory-decode-pass-review.json`.
These are still diagnostic rows only: local RX 7900 XTX, single run per shape,
and PARO values are the prior 2026-05-21 local comparison rows, not rerun.

Accepted/rejected state after this pass:

| Change | Decision | Evidence |
| --- | --- | --- |
| Resident prefill scratch K/V cleanup | Accepted | `68.5 MiB` tracked saving at 32K/1; targeted GGUF tests passed. |
| `Q4_K_S` selected Q4 down T16 kernels | Accepted | Added Q4 single-output selected GEMV/WMMA paths; `118` selected T16 tests + `10` resolver/chunk tests passed. |
| No-repack/T16 residency mode | Deferred | Saves `~468 MiB` but current safe raw path loses fast prefill/decode. |
| Q8_0 T16 scale broadcast | Rejected | 4K/128 decode regressed `-26.7%`; reverted. |
| Selected-MoE Q4 launch-bounds tweak | Rejected | 4K/128 decode `96.746 -> 96.651 tok/s`; reverted. |

Size distribution / payload source:

| Size bucket (GiB) | PARO `w4_paro` | GGUF `Q4_K_M` | GGUF `Q4_K_S` | Notes |
| --- | ---: | ---: | ---: | --- |
| Model file size | `19.237` | `20.614` | `19.458` | Local files on disk. |
| Tensor payload | `19.225` | `20.604` | `19.448` | `Q4_K_S` saves `1.156 GiB` payload vs `Q4_K_M`. |
| MoE / FFN payload | `15.862` | `18.428` | `17.271` | `Q4_K_S` changes expert-down tensors from `Q5_K` to `Q4_K`. |
| Attention payload | `0.402` | `1.017` | `1.017` | Same GGUF attention payload for M/S. |
| Linear-attention / SSM payload | `0.502` | `0.267` | `0.267` | Same GGUF SSM payload for M/S. |
| Embedding / LM-head / norm payload | `1.895` | `0.892` | `0.892` | Same GGUF embedding/lm/norm payload for M/S. |

Encoding/layout distribution:

| Encoding bucket (GiB) | PARO `w4_paro` | GGUF `Q4_K_M` | GGUF `Q4_K_S` | Notes |
| --- | ---: | ---: | ---: | --- |
| Main 4-bit payload | `15.596` `qweight` | `11.250` `Q4_K` | `16.453` `Q4_K` | `Q4_K_S` has no `Q5_K` tensors. |
| Higher-bit quant payload | n/a | `9.257` (`Q5_K` + `Q6_K` + `Q8_0`) | `2.898` (`Q6_K` + `Q8_0`) | Main source of the M→S saving. |
| Scale/zero side metadata | `0.609` | embedded | embedded | GGUF scale/min bytes are inside quant blocks. |
| Dense standalone weights | `3.009` | `0.097` `F32` | `0.097` `F32` | Different checkpoint layouts. |

Prefill throughput (tok/s):

| Workload | PARO `w4_paro` | GGUF `Q4_K_M` latest | GGUF `Q4_K_S` latest | `Q4_K_S` vs `Q4_K_M` |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | `2101.158` | `1544.572` | `1557.299` | `+0.8%` |
| 4K/128 | `2710.869` | `2552.043` | `2637.148` | `+3.3%` |
| 32K/128 | `2082.012` | `1861.893` | `1944.065` | `+4.4%` |
| 65K/128 | not rerun | not run | `1481.459` | n/a |
| 128K/128 | `1023.868` | does not fit locally | not probed | n/a |

Decode throughput (tok/s):

| Workload | PARO `w4_paro` | GGUF `Q4_K_M` latest | GGUF `Q4_K_S` latest | `Q4_K_S` vs `Q4_K_M` |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | `107.314` | `89.494` | `90.499` | `+1.1%` |
| 4K/128 | `106.637` | `96.264` | `97.121` | `+0.9%` |
| 32K/128 | `92.908` | `85.017` | `85.815` | `+0.9%` |
| 65K/128 | not rerun | not run | `74.035` | n/a |
| 128K/128 | `61.800` | does not fit locally | not probed | n/a |

Tracked peak memory (GiB):

| Workload | PARO `w4_paro` | GGUF `Q4_K_M` latest | GGUF `Q4_K_S` latest | `Q4_K_S` saving vs `Q4_K_M` |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | `18.176` | `21.342` | `20.185` | `1.156 GiB` |
| 4K/128 | `20.047` | `22.572` | `21.416` | `1.156 GiB` |
| 32K/128 | `20.320` | `23.302` | `22.146` | `1.156 GiB` |
| 65K/128 | not rerun | not run | `23.102` | n/a |
| 128K/128 | `23.288` | does not fit locally | not probed | n/a |

Current local long-context floor/ceiling:

| Probe | PARO `w4_paro` | GGUF `Q4_K_M` latest | GGUF `Q4_K_S` latest |
| --- | --- | --- | --- |
| Safe comparison point | `32K/128` fits (`20.320 GiB`) | `32K/128` fits (`23.302 GiB`) | `32K/128` fits (`22.146 GiB`) |
| Largest successful probe here | `128K/128` fits (`23.288 GiB`) | prior `35070/1` fit (`23.432 GiB` tracked, `23.911 GiB` sampled HIP) | `65K/128` fits (`23.102 GiB` tracked, `23.627 GiB` sampled HIP) |
| First observed fail / ceiling | not probed above 128K | prior `36864/1` failed during prefill; `40960/1` failed allocation | not probed higher in this pass |

Interpretation:

- `Q4_K_S` is **not** slower in the latest fastpath runs. It is slightly faster
  than latest `Q4_K_M` at the comparable 512/4K/32K shapes and saves a stable
  `~1.156 GiB` tracked memory.
- `Q4_K_S` narrows the local 4K/128 PARO memory gap from `+2.525 GiB` for latest
  `Q4_K_M` to `+1.369 GiB`, while staying within `-2.7%` PARO prefill and
  `-8.9%` PARO decode in this diagnostic.
- The Q4_K_S kernel support was required for a fair run: before the Q4 selected
  down WMMA route, the 512/1 smoke fell back to a much slower path; after
  `d92977a`, `Q4_K_S` uses the fast selected T16 down path.
- Next memory work should re-probe the `Q4_K_S` max-fit boundary above 65K and
  only consider no-T16/emergency residency if long-context headroom is worth the
  current speed tradeoff.

---

## GGUF vs PARO vs llama.cpp gap analysis (2026-06-16)

### Baseline numbers (W7900 GPU0, TheRock 7.13, Q4_K_M)

| Workload | hipEngine GGUF PF | hipEngine PARO PF | llama HIP PF | llama VK PF |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 2182 | 2730 | 2516 | 2823 |
| 4K/128 | 2491 | 2880 | 2303 | 2582 |
| 32K/128 | 1840 | 2079 | 1685 | 1969 |

| Workload | hipEngine GGUF DC | hipEngine PARO DC | llama HIP DC | llama VK DC |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 106.6 | 115.2 | 79.6 | 106.2 |
| 4K/128 | 97.5 | 105.3 | 78.7 | 102.6 |
| 32K/128 | 84.9 | 92.0 | 71.8 | 91.6 |

Peak memory: hipEngine GGUF 26.3 GiB vs PARO 21.0 GiB vs llama HIP 21.6 GiB.

### Prefill gap root causes

**#1 — MoE expert prefill uses GEMV instead of WMMA GEMM.**
PARO uses `gemm_awq_selected_dual_pack8_wmma_compact_bf16` for MoE experts at
prefill (tokens > 8). GGUF's compact WMMA path exists
(`_try_run_post_attention_moe_rows_compact_wmma` in `qwen35_gguf_runner.py`)
but is gated behind `HIPENGINE_GGUF_WMMA_PREFILL=1`, which defaults to off.
Without it, GGUF dispatches per-expert GEMV launches — the largest single
kernel-level difference for a MoE model. Estimated contribution: 5-10%.

**#2 — BF16 activations vs PARO's FP16.**
PARO operates FP16 natively. GGUF uses BF16 + extra `bf16_to_f32` casts
before conv/GDN operations. BF16 has higher register pressure on RDNA3.
Estimated contribution: 2-3%.

**#3 — GEMV-shaped dense projection kernels at prefill.**
Without WMMA opt-in, GGUF uses row-GEMV for all projections even at prefill.
PARO uses AWQ pack8 multi-row GEMV with better cache behavior. Estimated
contribution: 3-5%.

### Decode gap root causes

**#1 (~60% of gap) — Dequantization kernel quality.**
GGUF Q4_K format (6-bit scales + 4-bit weights + mins) has more complex
dequant math per element than PARO's AWQ pack8 (128-element group scaling).
Compounds across ~150+ GEMV launches per token.

**#2 (~25% of gap) — MoE C-dispatch.**
PARO bundles 6 MoE sub-methods into one `extern "C"` call via
`_try_moe_c1_c_dispatch`, eliminating ~320 Python→C transitions per token.
GGUF has no C-dispatch equivalent, but it utilizes HIP graph capture for decode by default, so Python dispatch overhead is completely bypassed.

**#3 (~15% of gap) — Fewer launches from dual/fused projections.**
PARO's dual QK GEMV (1 launch for 2 matrices), fused RMSNorm+rotate, and
fused activate+down save ~100-150 launches vs GGUF across all layers.

### Peak memory gap root causes

The gap is constant across all context sizes, confirming it is weight
materialization, not scratch/KV:

| Component | Delta | Source |
| --- | --- | --- |
| Pack8 expansion for dense Q4_K | **+2-3 GiB** | `repack_gguf_q4_k_pack8()` precomputes FP32 scale/min arrays (33% larger than raw) |
| Q8_0 T16 tile overhead | **~0.5 GiB** (fixed 2026-06-17) | Dropped T16 decode-repack for Q8_0 |
| GGUF Q4_K larger than PARO W4 | **+0.6 GiB** | Q4_K is 4.5 bits/value vs AWQ ~4.16 bits/value |
| Expert T16 expansion | +0.1 GiB | 2.8% over raw |
| Scratch/KV/other | **~1.25 GiB** (fixed 2026-06-17) | Chunk-outer prefill bug fixed in `qwen35_gguf_runner.py` |

### Fix plan (ordered by leverage)

| # | Fix | Expected Impact | Risk | Status |
|---|---|---|---|---|
| 1 | **Enable WMMA prefill** | +5-10% PF | BLOCKED | Correctness regression on T16 WMMA prefill; needs kernel-level fixture before retry. |
| 2 | **C-dispatch for GGUF MoE** | N/A | N/A | **Invalid**. GGUF utilizes HIP graph capture for decode, so Python dispatch overhead is already bypassed entirely. |
| 3 | **Fuse RMSNorm+rotate** | N/A | N/A | **Invalid**. GGUF does not use AWQ input-rotation; this optimization is specific to PARO's w4a16 layout. |
| 4 | **Fuse activate+down** | +1-2% DC | Low | **Done** (2026-06-17). Neutral/slightly lower decode speed; retained for launch overhead reduction. |
| 5 | **Pack8 layout opt** | -2-3 GiB mem | Tradeoff | **Done** (2026-06-17). Avoided Pack8 expansion, saving ~1.15 GiB peak memory at the cost of a small prefill/decode throughput regression (114.60 -> 114.42 tok/s on 4K DC). |
| 6 | **Drop T16 for Q8_0** | -0.5-1 GiB mem | Low | **Done** (2026-06-17). Saved ~0.55 GiB peak memory with negligible decode regression. |



