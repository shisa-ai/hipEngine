# GGUF Intake and Native-Quant Plan

Date: 2026-05-16  
Target repo: `~/hipEngine`
Primary references: local llama.cpp checkouts under `~/llama.cpp/` and parent evidence in `~/amd-gpu-tuning/`

## Executive summary

The short answer to "can hipEngine load GGUF quants easily now?" is:

- **GGUF file intake / metadata scanning is easy.** GGUF is a well-documented tensor container with a mature Python reader in llama.cpp's `gguf-py`.
- **Correctness-first FP16 fallback is straightforward.** We can parse GGUF, dequantize tensors on the host, map names into hipEngine's existing model loader, and run existing FP16 kernels. This proves model/tokenizer/tensor-name plumbing but does not preserve GGUF memory/perf benefits.
- **Native GGUF quant execution is not drop-in.** GGUF `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `Q8_K`, and `IQ*` tensors have GGML block layouts and quant math that differ from PARO/AWQ and from the current Marlin-K v0 layout. They need their own quant plugins, CPU oracles, and HIP kernels or a deliberate repack path.
- **The new PARO/Marlin-K work makes this tractable.** hipEngine now has the pattern we want: file/checkpoint layout -> host repack -> explicit device layout -> raw-pointer kernel -> registry dispatch. GGUF should use the same architecture, not special-case dispatch.

Recommended first implementation is a **GGUF scanner + FP16 fallback loader** and then one narrow native quant path, probably `Q8_0` first or `Q4_K` if the goal is direct llama.cpp Q4_K_M parity.

Do not treat this document as a performance claim. It is an implementation plan. Any hipEngine GGUF speedup must be measured in hipEngine after the loader/kernels land.

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

Implement in this order:

1. `docs/GGUF.md` (this doc).
2. `hipengine/loading/gguf.py` scanner with no new hard dependency.
3. `scripts/inspect_gguf.py` for tensor census.
4. `hipengine/quant/gguf.py` quant layout table and CPU dequant wrappers for `F16/F32/BF16/Q4_0/Q8_0/Q4_K`.
5. Qwen dense GGUF name-map smoke using local `Qwen3-0.6B-Q4_0.gguf` if available.
6. FP16 fallback loader.
7. Native `Q8_0` or `Q4_K` GEMV kernel, chosen by which local model we want to make fast first.

If the goal is user-visible model availability quickly, choose FP16 fallback first. If the goal is llama.cpp Q4_K_M performance comparison, choose Q4_K native first after scanner/oracle work.

## Bottom line

hipEngine is now structurally ready for GGUF intake, but GGUF quants are not automatically supported by the PARO Marlin-K layout. The right plan is to add GGUF as a first-class loader/quant family with staged fallbacks and native kernels:

```text
GGUF scanner -> FP16 fallback -> Q8_0/Q4_K native kernels -> broader K/IQ quant coverage
```

The Marlin-K/qweight-neutral work gives us the porting pattern and the memory discipline. GGUF will need its own quant layouts and correctness gates.
