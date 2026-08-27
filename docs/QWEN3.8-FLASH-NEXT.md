# Qwen3.8-Flash-Next Implementation Campaign

Status: **active — strict gfx1151 text c1/greedy works through 2,051 tokens; official BF16 source completion, real long-context QSA, serving, MTP, and vision qualification remain open**

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
a reproducibility/quant-quality follow-up once the official BF16 snapshot is
complete; it is not required to relabel the verified working artifact. The
51.2B-parameter n-gram table remains one IQ4_NL sparse mmap/host owner rather
than consuming accelerator-resident capacity.

This document is the campaign authority. Architecture-wide decisions also stay
consistent with [`PLAN.md`](PLAN.md); numerical and evidence rules remain
normative in [`TESTING.md`](TESTING.md),
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md),
[`KERNELS.md`](KERNELS.md), and [`BENCHMARK.md`](BENCHMARK.md).

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
| Native context | 262,144 tokens |
| Extended context | up to 1,000,000 tokens; not an initial support claim |

The download is resumable and must resolve the exact revision above. Model
weights and converted GGUF files are local artifacts and are never committed.
The F0 worklog and artifact record the completed file size and SHA-256 after
conversion; placeholders in this document do not constitute evidence.

### 1.2 Read-only architecture and comparator sources

| Source | Frozen identity / use |
| --- | --- |
| Qwen technical report | `QwenLM/Qwen3.8-Flash-Next/tech_report.pdf`, local SHA-256 `04f263446d74a35cb7cea368574e0c561f3b05c133be2c777ac884404063655d`; architecture/formula authority |
| Transformers | `transformers/models/qwen4_exp/modular_qwen4_exp.py`; executable framework oracle outside the hot path |
| llama.cpp PR #27742 | `bea3b12daee45876b0129a3602dc8f534ce30bf0`; primary converter, quantizer fixes, target-text comparator, QSA/PLE/GR reference |
| llama.cpp PR #27739 | `dfa0c0fee2b704fd2ac228d365d40502c3006c40`; alternate MTP/PLE reference only, not the primary quantizer |
| SGLang day-0 path | QSA compressed-cache, sparse PLE offload, HyperConnection and IndexShare MTP implementation reference |
| Unsloth guide/repository | Memory-size expectation and independent GGUF comparator; no unverified quality claim transfers into hipEngine |

The llama.cpp worktree is local and external to hipEngine:
`/home/lhl/llama.cpp/llama.cpp-qwen4exp` at PR #27742 commit `bea3b12da`.
Ported ideas must cite this commit and exact source path. Kernel development
still occurs only in this tree.

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
with zero tracked bytes after close. The measured 381.725→245.855 s prefill
wall is diagnostic only. A repeated-token structural 4,096-token checkpoint
also passes: teacher→serial/chunk KL `4.40e-5/4.78e-5`, serial→chunk KL
`3.19e-5`, top-1 264 exact, and zero tracked bytes after close; diagnostic
serial/chunk walls are `854.982/574.759` seconds. A practical chunk-only 16K
checkpoint also passes teacher KL `7.55e-5`, top-1 264 exact, and zero teardown
bytes in 2,434.172 seconds; strict serial remains measured through 4K. This does
not close natural retrieval, selected-index, strict-above-4K, 64K/262K
inference, or lifecycle/isolation gates. Separately, the real complete
262,144-token owner allocates successfully at 91,126,119,496 tracked bytes,
leaves 38,915,162,112 physical bytes free, and returns to zero tracked bytes on
close; this is capacity/lifecycle evidence only. Exact evidence is in
[`2026-08-27-gfx1151-qwen38-flash-next-qsa-2052-transition.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-2052-transition.json)
and
[`2026-08-27-gfx1151-qwen38-flash-next-qsa-4k.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-4k.json), and
[`2026-08-27-gfx1151-qwen38-flash-next-qsa-16k.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-16k.json), and
[`2026-08-27-gfx1151-qwen38-flash-next-262k-capacity.json`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-262k-capacity.json).

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
change. Physical host identity is mandatory.

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

### F10 — Public serving and closure

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

- [ ] F0 exact source downloaded and Q4_K_M produced, scanned, hashed, and
      llama-smoked.
- [x] F1 qwen4exp plugin and real GGUF tensor map pass.
- [x] F2 CPU GR/PLE/QSA/GDN reduced-model oracles pass.
- [x] F3 one-layout residency fits; PLE is sparse mmap-owned; teardown is zero.
- [x] F4 missing native primitives pass RED/GREEN and kernel traces.
- [x] F5 strict text AR works below the QSA budget through public APIs.
- [ ] F6 QSA passes long-context correctness and capacity through 262K.
- [ ] F7 retained prefill/decode/batching paths are measured and documented.
- [ ] F8 official MTP is correct and either promoted in scope or honestly
      rejected on economics.
- [ ] F9 multimodal image/video support passes or remains explicitly
      unsupported without affecting text claims.
- [ ] F10 serving, reasoning/tool template behavior, lifecycle, rollups, and
      public documentation close.

The campaign is complete only when the checked items match committed code and
durable evidence. A pending background download, a quant file, a campaign doc,
or a single generated answer is not completion.
