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
`LLM.generate()`. The hard gate now passes all local quant fixtures for the target prompt with no `torch` import on the generate path.

The short answer to "can hipENGINE load GGUF quants easily now?" is:

- **GGUF file intake / metadata scanning is easy.** GGUF is a well-documented tensor container with a mature Python reader in llama.cpp's `gguf-py`.
- **Correctness-first FP16 fallback is straightforward.** We can parse GGUF, dequantize tensors on the host, map names into hipENGINE's existing model loader, and run existing FP16 kernels. This proves model/tokenizer/tensor-name plumbing but does not preserve GGUF memory/perf benefits.
- **Native GGUF quant execution is not drop-in.** GGUF `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `Q8_K`, and `IQ*` tensors have GGML block layouts and quant math that differ from PARO/AWQ and from the current Marlin-K v0 layout. They need their own quant plugins, CPU oracles, and HIP kernels or a deliberate repack path.
- **The new PARO/Marlin-K work makes this tractable.** hipENGINE now has the pattern we want: file/checkpoint layout -> host repack -> explicit device layout -> raw-pointer kernel -> registry dispatch. GGUF should use the same architecture, not special-case dispatch.

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
Broader prompts, public full-model bulk prefill, and throughput claims remain
future work.

## Why GGUF is attractive for hipENGINE

GGUF gives us three useful things:

1. **A mature model artifact format.** A single `.gguf` carries model metadata, tokenizer metadata, tensor names, shapes, tensor types, and tensor bytes.
2. **A huge ecosystem of local quantized models.** Qwen, Llama, Gemma, Mixtral/MoE variants, and many user-facing quants are already published as GGUF.
3. **Reference implementations and baselines.** llama.cpp gives both parsing/quant oracles and W7900 comparison rows for HIP/Vulkan.

The parent workspace has already used GGUF/llama.cpp as an external comparator, especially Qwen3.6-35B-A3B `Q4_K_M` and `UD-Q8_K_XL` rows. hipENGINE should use GGUF support to widen model access and to make apples-to-apples kernel comparisons against llama.cpp easier.

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
- `/home/lhl/hipENGINE/docs/MARLIN.md`
  - hipENGINE's Marlin-K intake analysis, including the qweight-neutral host-layout work already started here.

## GGUF file structure hipENGINE needs

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

hipENGINE should initially consume GGUF through a tiny loader module that either:

- uses `gguf-py` as an optional import, or
- implements a minimal pure-Python reader for the subset we need.

Because hipENGINE's runtime hot path must stay torch-free, either option is compatible. The issue is dependency policy: adding hard dependency `gguf` is avoidable. Prefer a small optional reader or an optional `gguf` extra until we know how much of `gguf-py` we need.

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

Goal: Given a `.gguf`, print enough metadata to decide whether hipENGINE can load it.

Outputs:

- architecture (`general.architecture`)
- file type (`general.file_type`)
- tokenizer metadata presence
- tensor count and total bytes by quant type
- tensor-name mapping coverage for the target hipENGINE model plugin
- list of unsupported quant types

This is low risk and should be first.

Likely files:

```text
hipengine/loading/gguf.py
tests/test_gguf_reader.py
scripts/inspect_gguf.py
```

### Tier 1: FP16 fallback loader

Goal: Load a GGUF model into hipENGINE by dequantizing quantized tensors on the host to FP16/BF16 and using existing FP16 kernels.

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

Goal: Run common llama.cpp `Q4_K_M` GGUF weights with native hipENGINE kernels.

This is the first truly useful GGUF memory/perf target because many public models use Q4_K_M and our parent analysis already compared against Q4_K_M.

Implementation choices:

1. **Direct GGML block kernel**
   - Device layout mirrors `block_q4_K` row blocks.
   - Kernel decodes GGML `scales[12]`, `d`, `dmin`, and q4 payload directly.
   - Best for fidelity and avoiding extra memory.

2. **Host repack to hipENGINE-native Marlin-ish layout**
   - Parse `block_q4_K` on host and emit a device layout optimized for W7900.
   - Could separate q4 payload and decoded scale/min tables for faster kernels.
   - Costs load-time memory and may deviate from exact GGML layout, but fits hipENGINE's Marlin-K architecture.

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

GGUF is a container; hipENGINE still needs a model plugin.

Recommended first architecture targets:

1. **Tiny Qwen GGUF scanner/fallback**
   - Local cache includes `ggml-org/Qwen3-0.6B-GGUF` with `Qwen3-0.6B-Q4_0.gguf` under Hugging Face cache.
   - Good for loader/tokenizer smoke.
2. **Qwen2/Qwen3 dense**
   - Similar enough to existing hipENGINE Qwen code to make name mapping feasible.
3. **Qwen3.5/Qwen3.6 MoE GGUF**
   - Performance-relevant but more complicated: expert tensor naming, routing, and active-expert surfaces.

Avoid starting with arbitrary Llama/Gemma if the goal is to reuse existing Qwen runtime. Llama/Gemma can come once GGUF loading is generic enough and model plugins exist.

## Tensor-name mapping problem

GGUF tensor names are not guaranteed to match Hugging Face safetensors names. llama.cpp has architecture-specific tensor maps in `gguf-py/gguf/constants.py` and conversion scripts.

hipENGINE needs a mapping layer:

```text
GGUF tensor name -> hipENGINE logical tensor name -> model/runtime slot
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

GGUF often carries tokenizer metadata. hipENGINE currently depends on `tokenizers` and has Qwen/HF-oriented loading. For GGUF:

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

## Proposed hipENGINE implementation shape

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

The shared rule: **separate file format from execution layout**. A GGUF loader can either preserve GGML blocks or repack into a hipENGINE-native layout. The choice should be measured per quant type.

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
- Run hipENGINE dense linear CPU/GPU fixture if available.
- For model-level smoke, use fixed token IDs first; tokenizer integration can follow.

### Native quant validation

For each native quant kernel:

1. CPU dequant oracle from internal code and/or `gguf-py`.
2. Deterministic GPU fixture: tiny rows/K/N, exact or tight tolerance.
3. hipENGINE correctness gate per `AGENTS.md`: KL <= 0.05 and top-1 agreement >= 90% vs CPU reference on fixture inputs.
4. `rocprofv3 --kernel-trace` showing the expected kernel name and plausible duration.
5. Only then benchmark against existing hipENGINE fallback and llama.cpp reference.

### Benchmark policy

GGUF performance claims need all normal hipENGINE benchmark metadata:

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
