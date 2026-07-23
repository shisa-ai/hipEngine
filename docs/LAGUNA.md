# Laguna S 2.1 Q4_K_M and DFlash on gfx1151

Last updated: 2026-07-24

Status: the declared gfx1151 c=1/4K support slice is complete. The pinned
Q4_K_M target has all-resident torch-free loading, exact chunked prefill/B+1
rows, public blocking/streaming generation, Poolside-v1 reasoning/XML tools, and
canonical target-AR evidence. The pinned BF16 DFlash drafter is supported through
B4 only as an explicit library/OpenAI opt-in; its ten-prompt public gate is exact
against true AR. DFlash remains off by default because the current merged-main
full-suite ratio is **0.9477x**, with heldout and all non-code categories
regressive.

The support boundary does not claim exact Poolside free-running greedy-32
identity after the documented low-margin token-30 split, contexts above 4K,
c>1, graph replay, sampled/processed-logit DFlash, a BF16 DFlash GGUF container,
or B7/B15 product admission. Cached loading is retained at **48.20 s median**
versus the original 227.51 s, but remains slower than the qualified 29.85-29.91 s
Poolside readiness reference; sequential reads are the named follow-up rather
than a support blocker.

This document defines the correctness-first plan for running
[`poolside/Laguna-S-2.1-GGUF`](https://huggingface.co/poolside/Laguna-S-2.1-GGUF),
specifically `laguna-s-2.1-Q4_K_M.gguf`, through hipEngine on the local
Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`) host. It records the model contract,
what hipEngine can reuse, the remaining architecture work, the unified-memory
capacity model, the matched Poolside DFlash follow-on, staged implementation
gates, and the evidence required before any result can be called supported or
fast.

Do not treat this document as a benchmark claim. Capacity and performance
projections below are explicitly marked as measured, calculated, or inferred.
Any retained result must follow [`BENCHMARK.md`](BENCHMARK.md), update the
benchmark rollup, and include a compact artifact.

## Executive Summary

Laguna S 2.1 Q4_K_M is a realistic native-resident target on the local gfx1151
system. The GGUF contains about **70.01 GiB** of tensor data, while the host
currently exposes a **120.0 GiB** HIP/GTT allocation domain backed by 128 GiB of
unified LPDDR5X memory. The 512 MiB amdgpu `vram` aperture is not the model
capacity limit. A single short- or medium-context session should fit without
CPU weight offload, expert paging, tensor parallelism, or pipeline parallelism.
A 256K single-request context also appears capacity-feasible if sliding-window
layers retain only 512 tokens and materialization/scratch stay within the budget;
this is not yet measured. Exact 1M BF16-KV operation does not fit the current
120 GiB allocation domain.

The main work is a model-family port:

1. Parse and validate Laguna metadata and tensor names.
2. Add a CPU/reference Laguna layer and tiny deterministic fixture.
3. Preserve or explicitly validate the F16 attention projections.
4. Implement mixed global/sliding attention with two RoPE contracts.
5. Implement per-head softplus attention output gating.
6. Implement sigmoid MoE routing with selection-only correction bias, top-10
   normalization, routed scaling, and one always-on shared expert.
7. Add an eager full-model resident runner, tokenizer, public generator, and
   chat/tool behavior.
8. Add bulk prefill, then independently tune and admit gfx1151 graph/context
   paths.
9. Integrate Poolside's matched Laguna S 2.1 DFlash drafter as a follow-on exact
   speculative path after target AR correctness and B+1 verification are green.

The matched DFlash artifact materially improves the opportunity: it is a
six-layer, 1.115B-parameter BF16 Laguna drafter trained for this target, shares
the target embedding and LM head, and adds only about 2.08 GiB of resident
weights. It does not reduce the work required for base AR support, and no speedup
is assumed. It does mean hidden-tap ownership and verifier-shaped target APIs
should be designed correctly during the base port rather than retrofitted later.

hipEngine already has the difficult GGUF quant foundation: Q4_K/Q6_K intake,
CPU dequantization, lossless replacement layouts, dense and rank-3 selected
expert kernels, F32 router projection, paged attention/KV, BF16 KV, bulk
prefill machinery, sampling, and a native gfx1151 backend. Laguna should extend
those plugin surfaces; it must not be disguised as Qwen or introduced through
backend/quant branches in engine code.

### End-to-end completion audit (2026-07-23)

The thread objective is closed for one precise product boundary: the pinned
Laguna S 2.1 Q4_K_M artifact on gfx1151, exact physical c=1 model ticks with
at most two scheduler-resident rows, at most 4K BF16 KV, raw greedy generation,
public blocking/streaming chat and tools, plus the pinned B4 DFlash
owner as an explicit opt-in. “Complete” below does not broaden that boundary.

| Deliverable | Implementation and focused gates | Retained evidence | Audit verdict |
| --- | --- | --- | --- |
| Source-bound resident load | `hipengine/loading/laguna_gguf*.py`, `hipengine/runtime/laguna_gguf_runner.py`; config/map/materialization/device/lifecycle suites | `2026-07-22-gfx1151-laguna-s21-repacked-cache-startup-retained.json` | Complete; cache/source trajectories agree and tracked ownership returns to zero. |
| Target AR and public serving | `hipengine/generation/laguna_gguf.py`; direct, `LLM`, OpenAI blocking/streaming, resident admission/reclaim, EOT/cancel/capability tests | target-AR, LPF-1/4/5, bulk-correctness, `2026-07-23-gfx1151-laguna-native-scheduler.json`, and qualified Poolside artifacts under `benchmarks/results/` | Complete for exact physical c=1/4K and logical two-slot serving. Current D4 true-AR control is 16.384 decode tok/s; 512/1K/4K prefill is exact at 47.395/44.855/38.552 tok/s. Native scheduling adds bounded c2 ownership, not c>1 model math or a speedup claim. |
| Poolside-v1 reasoning and tools | `hipengine/chat/poolside_v1.py`; frozen renderer/reasoning/tool fixtures plus generic server conformance | `2026-07-22-gfx1151-laguna-poolside-v1-e2e-correctness.json` | Complete: 5/5 live blocking/streaming cases and 7/7 deterministic tool fixtures, including multiple calls and escaped UTF-8. |
| B4 DFlash correctness and public route | `hipengine/speculative/laguna_dflash.py`, `hipengine/generation/laguna_dflash.py`, provider registry/server route; drafter, B+1, rollback, API, and public-gate suites | drafter-B4, verify-commit, post-prefill economics, and `2026-07-23-gfx1151-laguna-dflash-public-e2e.json` | Complete as explicit-only: 10/10 AR, 10/10 blocking, and 10/10 streaming public rows are exact. AR remains default. |
| Ownership and truthful capability surface | shared target weights with isolated target/drafter/cycle request state; finish/cancel/close and fail-before-load gates | parser peak 77,022,439,484 bytes and public DFlash peak 79,817,890,405 bytes both recover to zero | Complete for the supported boundary; identity, revision, budget, exactness, fallback, and no-performance-claim metadata pass. |
| Benchmark and handoff record | reproducible scripts, compact JSON, `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and chronological `WORKLOG.md` | canonical train+heldout four-category AR/DFlash packets and cached `rocprofv3` traces | Complete; no single-prompt speed or automatic DFlash claim is retained. |

The remaining items are explicit extensions or compatibility weaknesses, not
hidden completion claims: strict Poolside free-running equality is 29/32 before
one low-margin branch (31/32 teacher-forced top-1; first-token KL
`6.6214e-6`), the DFlash all-15-row diagnostic is 12/15 while admitted B4 is
12/12 top-k, DFlash safetensors rather than BF16 GGUF is canonical, contexts
above 4K/concurrency/graph are unadmitted, and DFlash is slower overall. Loader
read/upload overlap is the highest-value startup follow-up. None of these are
silently promoted by this audit.

## Scope and Non-goals

Initial scope:

- one local `laguna-s-2.1-Q4_K_M.gguf`;
- backend `hip_gfx1151` compiled with `--offload-arch=gfx1151`;
- torch-free runtime;
- BF16 activations and BF16 KV unless a narrower source-preserving path requires
  FP16 at an explicit boundary;
- deterministic eager greedy generation at 4K context;
- first-token logits and multi-token greedy parity against an independent
  Poolside/llama.cpp oracle;
- then bulk prefill, public `LLM.generate()`, streaming, and chat/tool behavior;
- context progression `4K -> 32K -> 64K -> 128K -> 256K`.

Follow-on DFlash scope, after exact target AR is supported:

- `poolside/Laguna-S-2.1-DFlash` revision
  `b0486d1586daa0d56435c508108171fc1c8daff9`;
- native BF16 drafter weights or Poolside's equivalent BF16 GGUF;
- exact chain speculation using the target's own verifier, embedding, LM head,
  hidden taps, and sampler contract;
- fixed draft-budget gates before any adaptive policy or tree topology;
- same-protocol true AR control over the full multi-category prompt suite.

Non-goals for the first supported slice:

- 1M context;
- INT8 KV;
- making speculative decoding part of the initial AR-support milestone;
- enabling DFlash by default before exactness and economics pass;
- multi-GPU execution;
- CPU weight offload or demand-paged experts;
- concurrency tuning before c=1 correctness;
- claiming speed from a single prompt or unvalidated graph replay;
- editing Poolside's or llama.cpp's repositories in place.

## Evidence and Reference Sources

Authoritative model/runtime sources reviewed for this plan:

- [Laguna S 2.1 source model](https://huggingface.co/poolside/Laguna-S-2.1)
- [Laguna S 2.1 GGUF model card](https://huggingface.co/poolside/Laguna-S-2.1-GGUF)
- [Laguna S 2.1 DFlash](https://huggingface.co/poolside/Laguna-S-2.1-DFlash),
  revision `b0486d1586daa0d56435c508108171fc1c8daff9`
- [Poolside llama.cpp fork](https://github.com/poolsideai/llama.cpp), branch
  `laguna`, which supports the matched target plus DFlash GGUF pair
- [upstream llama.cpp PR #25165](https://github.com/ggml-org/llama.cpp/pull/25165)
  (`Add support for Laguna XS.2 & M.1`; reviewed at PR head
  `54f214a09b8c4e709357ae661a77925edb154f13`, still open on 2026-07-22)
- Hugging Face `config.json`, `modeling_laguna.py`, tokenizer files, and resolved
  `chat_template.jinja`

Relevant hipEngine contracts:

- [`PLAN.md`](PLAN.md) — four-axis plugins and architectural invariants
- [`GGUF.md`](GGUF.md) — GGUF intake, quant layouts, and current runner state
- [`KERNELS.md`](KERNELS.md) — kernel catalog, port gates, and profiler evidence
- [`TESTING.md`](TESTING.md) — RED/GREEN and correctness-oracle policy
- [`TUNING-gfx1151.md`](TUNING-gfx1151.md) — same-device tuning rules
- [`ROOFLINE-gfx1151.md`](ROOFLINE-gfx1151.md) — UMA capacity and bandwidth model
- [`BENCHMARK.md`](BENCHMARK.md) — benchmark/evidence protocol
- [`DFLASH.md`](DFLASH.md) — existing native DFlash provider, verifier, accept,
  commit, and prior gfx1151/gfx1100 lessons

The local target GGUF download completed on 2026-07-22. The final local object
is exactly 75,173,103,200 bytes and its measured SHA-256 is
`7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f`, matching
the Hugging Face LFS object. The repository scanner validated all tensor spans
and reported 814 tensors / 75,169,369,088 tensor bytes with no unsupported
dequant types. One-row Q4_K, Q6_K, F16, and F32 dequantization smokes were all
finite. The strict Laguna map consumed all 814 tensors, resolved 12 full + 36
SWA layers, and detected per-head attention gates throughout. This freezes the
artifact for implementation; it is not an independent model-output oracle or a
performance result.

The DFlash safetensors artifact completed during this review. Its local file is
2,229,962,896 bytes and has SHA-256
`f24f08781c697c19952c02fb2e7e9bdf2071b79a711c2a44b836a74b9b62a1f4`, matching
the Hugging Face LFS object. Poolside's GGUF repository changed its DFlash
object on 2026-07-22 under `Correct DFlash config`: the prior 2,233,764,000-byte
object was `24614292...a67ee`, while the current 2,233,764,224-byte object is
`2ee8aa30...1bfd4`. Neither tensor payload is bit-identical to the pinned
published safetensors. The independent matched oracle therefore uses a local
Poolside conversion of revision `b0486d1`, SHA-256
`ad3d1efffa8763e11e55baf6fedddcbf9138b3077928b55e5da6625745808bd2`;
this local conversion is evidence only and is never committed.

## Measured gfx1151 Host Snapshot

Measured on 2026-07-22:

| Property | Value | Measurement scope |
| --- | ---: | --- |
| CPU/APU | AMD Ryzen AI MAX+ 395 | `rocminfo` |
| GPU | AMD Radeon 8060S | `rocminfo` |
| target | `gfx1151` | `rocminfo` |
| host RAM | about 125 GiB | `free -h` |
| host available at inspection | about 105 GiB | `free -h`; transient |
| HIP allocation total/free | 120.000 / 119.996 GiB | `hipMemGetInfo`; transient free value |
| amdgpu GTT total | 120.000 GiB | `mem_info_gtt_total` |
| amdgpu visible-VRAM aperture | 512 MiB | `mem_info_vram_total`; not capacity |
| TTM pages limit | 120 GiB | `/sys/module/ttm/parameters/pages_limit` |
| boot IOMMU | disabled | `/proc/cmdline`: `amd_iommu=off` |
| HIP runtime | available | `ctypes.CDLL("libamdhip64.so")` |

On this APU, CPU and GPU allocations use the same physical LPDDR5X. GTT is an
allocation/mapping domain, not a discrete slow-host tier behind a PCIe link.
Use `mem_info_gtt_used` for whole-device sampling and hipEngine's tracked
allocator for owned allocations. Never report the 512 MiB aperture as model
memory.

Operational rules inherited from the gfx1151 backend:

- keep `GPU_MAX_HW_QUEUES=1` unless an explicit experiment overrides it;
- compile JIT artifacts natively for `gfx1151` and include the target in the
  cache key;
- prebuild `.so` files before `rocprofv3` runs and require cached builds;
- admit graph replay only after a Laguna-specific eager/graph state gate;
- do not transfer W7900 kernel defaults without same-device evidence.

## GGUF Inventory

The header describes GGUF v3, architecture `laguna`, quantization version 2,
file type `MOSTLY_Q4_K_M`, and 814 tensors.

| GGML type | Tensor count | Tensor bytes | Approx. GiB |
| --- | ---: | ---: | ---: |
| F16 | 240 | 5,606,277,120 | 5.22 |
| F32 | 287 | 149,138,432 | 0.14 |
| Q4_K | 239 | 53,876,883,456 | 50.18 |
| Q6_K | 48 | 15,537,070,080 | 14.47 |
| **Total** | **814** | **75,169,369,088** | **70.01** |

Expected final file size from the tensor directory is 75,173,103,200 bytes;
the small difference from tensor bytes is GGUF metadata/alignment.

Important root tensors:

| Tensor | Type | Shape |
| --- | --- | --- |
| `token_embd.weight` | Q4_K | `(100352, 3072)` |
| `output_norm.weight` | F32 | `(3072,)` |
| `output.weight` | Q6_K | `(100352, 3072)` |

The output projection is untied. The Q6_K LM head must not alias the Q4_K token
embedding.

Representative layer-0 tensors confirm source-preserving F16 attention:

| Tensor | Type | Shape |
| --- | --- | --- |
| `blk.0.attn_gate.weight` | F16 | `(48, 3072)` |
| `blk.0.attn_q.weight` | F16 | `(6144, 3072)` |
| `blk.0.attn_k.weight` | F16 | `(1024, 3072)` |
| `blk.0.attn_v.weight` | F16 | `(1024, 3072)` |
| `blk.0.attn_output.weight` | F16 | `(3072, 6144)` |
| `blk.0.attn_q_norm.weight` | F32 | `(128,)` |
| `blk.0.attn_k_norm.weight` | F32 | `(128,)` |
| `blk.0.ffn_gate.weight` | Q4_K | `(12288, 3072)` |
| `blk.0.ffn_up.weight` | Q4_K | `(12288, 3072)` |
| `blk.0.ffn_down.weight` | Q6_K | `(3072, 12288)` |

Layer 1 switches to 72 query heads, for example
`blk.1.attn_gate.weight` has shape `(72, 3072)`.

Expected MoE tensor families, following the GGUF and reviewed llama.cpp map:

- `blk.{layer}.ffn_gate_inp.weight` — F32 router projection;
- `blk.{layer}.exp_probs_b.bias` — F32 selection correction bias;
- `blk.{layer}.ffn_gate_exps.weight` — rank-3 routed expert gate weights;
- `blk.{layer}.ffn_up_exps.weight` — rank-3 routed expert up weights;
- `blk.{layer}.ffn_down_exps.weight` — rank-3 routed expert down weights;
- `blk.{layer}.ffn_gate_shexp.weight` — shared expert gate;
- `blk.{layer}.ffn_up_shexp.weight` — shared expert up;
- `blk.{layer}.ffn_down_shexp.weight` — shared expert down.

The completed-file scanner must validate every expected tensor, shape, type,
byte span, and layer count before device allocation.

## Model Configuration Contract

### Global dimensions

| Field | Value |
| --- | ---: |
| vocabulary | 100,352 |
| hidden size | 3,072 |
| layers | 48 |
| dense FFN size | 12,288 |
| KV heads | 8 |
| key head dimension | 128 |
| value head dimension | 128 |
| RMS epsilon | `1e-6` |
| sliding window | 512 |
| routed experts | 256 |
| selected experts/token | 10 |
| routed expert FFN size | 1,024 |
| shared expert FFN size | 1,024 |
| leading dense layers | 1 |
| routed scaling factor | 2.5 |

### Layer sequence and head counts

The 48-layer sequence repeats with period four:

```text
FULL, SWA, SWA, SWA
```

Therefore:

- full/global layers are `0, 4, 8, ..., 44` (12 layers);
- SWA layers are the other 36 layers;
- full layers use 48 query heads (`48 * 128 = 6144` Q channels);
- SWA layers use 72 query heads (`72 * 128 = 9216` Q channels);
- all layers use eight KV heads (`8 * 128 = 1024` K and V channels).

The runner and scratch planner must accept per-layer Q width. It must not assume
a model-wide constant `num_q_heads` or attention output width.

### RoPE and context

Full/global layers:

- partial rotary dimension: 64 of 128;
- base theta: 500,000;
- YaRN scaling;
- GGUF factor: 32;
- original context: 8,192;
- beta fast: 32;
- beta slow: 1;
- attention factor: 1.

SWA layers:

- full rotary dimension: 128;
- base theta: 10,000;
- plain RoPE;
- no inherited YaRN extension or magnitude scaling.

The GGUF ships with a 262,144-token context and the model card recommends that
configuration for output quality. The source checkpoint records a 1,048,576
maximum and Poolside documents a llama.cpp override using YaRN factor 128. This
is not the hipEngine default. Any future 1M mode must be explicit and separately
validated for quality and capacity.

### Tokenizer and stopping

GGUF tokenizer metadata:

| Field | Value |
| --- | --- |
| model | GPT-2 byte BPE |
| pre-tokenizer | `laguna` |
| BOS | 2 |
| EOS | 2 |
| EOT | 24 (`</assistant>`) |
| padding | 9 |

The tokenizer must preserve byte-BPE ID parity with the oracle. The production
encoder reconstructs Hugging Face `tokenizers` in memory from the GGUF
vocabulary, merges, token types, and `laguna` pre-tokenizer recipe; it requires
neither a `tokenizer.json` sidecar nor network access. Exact token-ID requests
bypass encoding and report `timing.tokenize_ms = 0`; text generation and the
token diagnostics endpoints expose measured `tokenize_ms` separately from
model prefill/decode timing.

EOT 24 must stop generation and its textual marker must not leak into returned
content. The GGUF contains a resolved chat template; use it rather than an
unresolved Jinja `include`. Thinking/no-thinking and tool-call behavior are
later public-surface gates, not assumptions made by the base completion path.

## Matched Laguna DFlash Artifact

Poolside publishes a target-specific DFlash drafter in both safetensors and BF16
GGUF form. The model card's reference invocation pairs it directly with this
Q4_K_M target:

```bash
./build/bin/llama-server \
  -m laguna-s-2.1-Q4_K_M.gguf \
  -md laguna-s-2.1-DFlash-BF16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 15 \
  -fa on --jinja --port 8000
```

This requires Poolside's `laguna` llama.cpp branch. Upstream PR #25165 covers
the target architecture only, not the Laguna DFlash decoder contract.

### DFlash configuration

| Field | Value |
| --- | ---: |
| architecture | `DFlashLagunaForCausalLM` |
| parameters | 1,114,977,792 BF16 |
| safetensors size | 2,229,962,896 bytes |
| BF16 GGUF repository revision | `e08e1fe855bb2d43f96ad78e24495283f3426c67` |
| BF16 GGUF size | 2,233,764,224 bytes |
| BF16 GGUF SHA-256 | `2ee8aa30338d6599bc7a8ce008cc57c56f2c2b2fdc21f6db9ecda203c751bfd4` |
| draft layers | 6 |
| draft layer type | six sliding-attention layers |
| hidden size | 3,072 |
| dense FFN size | 12,288 |
| Q heads | 72 |
| KV heads | 8 |
| head dimension | 128 |
| sliding window | 512 |
| attention gate | per-head |
| RoPE theta | 500,000 |
| maximum positions | 1,048,576 |
| vocabulary | 100,352, same as target |
| draft vocabulary | 100,352, no d2t/t2d map |
| block size | 16: root plus up to 15 draft rows |
| mask token | 12 |
| target layers | 48 |
| target hidden captures | post-layer depths 2, 11, 20, 30, 39, 48 |
| internal zero-based IDs | 1, 10, 19, 29, 38, 47 |
| causal | true |

Poolside's vLLM example starts with seven speculative tokens even though the
trained block permits 15. hipEngine should likewise begin with small fixed
budgets (`1, 2, 4, 7`) and treat 15 as a later economics point, not an automatic
default.

The source config records all draft layers as SWA and records theta 500,000, but
does not carry the target GGUF's full-versus-SWA dual-RoPE structure. The
drafter must follow its own pinned config/GGUF metadata and Poolside oracle; do
not inherit the target's SWA theta 10,000 by assumption.

### DFlash tensor inventory

The completed safetensors header contains 69 BF16 tensors and 2,229,955,584
payload bytes:

| Family | Count | Shape/role |
| --- | ---: | --- |
| `aux_hidden_norms.{0..5}.weight` | 6 | `(3072,)`; normalize target taps independently |
| `fc.weight` | 1 | `(3072, 18432)`; project six concatenated taps |
| `hidden_norm.weight` | 1 | `(3072,)`; post-projection norm |
| layer input/post-attention norms | 12 | two `(3072,)` vectors per layer |
| `self_attn.qkv_proj.weight` | 6 | `(11264, 3072)` = Q 9216 + K 1024 + V 1024 |
| `self_attn.g_proj.weight` | 6 | `(72, 3072)`; per-head softplus gate |
| `self_attn.o_proj.weight` | 6 | `(3072, 9216)` |
| Q/K norms | 12 | `(128,)` |
| dense gate/up/down | 18 | `(12288,3072)`, `(12288,3072)`, `(3072,12288)` |
| final `norm.weight` | 1 | `(3072,)` |

The Poolside BF16 GGUF download completed on 2026-07-23 and matches the Hugging
Face LFS object exactly at revision
`e08e1fe855bb2d43f96ad78e24495283f3426c67`: 2,233,764,224 file bytes and
SHA-256 `2ee8aa30338d6599bc7a8ce008cc57c56f2c2b2fdc21f6db9ecda203c751bfd4`.
The earlier `19bafe9bde46a3c7f9c0b2ef10ecb8a0ef154a1e` object is 2,233,764,000
bytes with SHA-256
`24614292a4477f3ae5203c3875edcde0bc219f02616a9c9f65791e29b18a67ee`.
All 76 tensor payloads and all 1,114,977,792 values are byte-exact between the
two revisions; e08 only adds the missing sliding-window pattern, rotary
dimension, and RoPE-scaling metadata. e08 remains canonical because the older
header is incomplete, not because its learned weights differ.

The GGUF-v3 header scans cleanly as architecture `dflash`, decoder `laguna`, six
512-window layers, block size 16, target depths `2,11,20,30,39,48`, mask token
12, and a 100,352-entry Laguna tokenizer. Its 76 tensors contain 2,230,081,536
payload bytes: 49 BF16 tensors and 27 F32 norm tensors.

The container has the same architecture schema but is neither storage- nor
weight-identical to the pinned 69-BF16 safetensors contract. It stores the six
auxiliary norms as one F32 `enc.aux_norm.weight (6,3072)` tensor, converts the
hidden/final and per-layer norm families to F32, and expands each fused QKV
allocation into separate BF16 `attn_q`, `attn_k`, and `attn_v` tensors. Six
fused tensors becoming eighteen adds 12 entries while six auxiliary tensors
becoming one removes five, explaining the net **69 -> 76** count.

A complete torch-free payload comparison against both the b048 safetensors and
the prior local direct conversion covers all 69 logical tensors and all
1,114,977,792 parameter values. Every F32 GGUF norm down-converts exactly to its
BF16 source: **32/69 logical tensors and 62,976 values are exact**. All 37 linear
families differ, however: **564,101,261 / 1,114,914,816 linear values
(50.5930%)** have different BF16 bits. This confirms the earlier Poolside
intermediate finding that the published GGUF is a distinct weight artifact, not
just another encoding of `poolside/Laguna-S-2.1-DFlash@b0486d1`.

The current hipEngine runtime deliberately does not consume this alternate
layout or identity. D0 must normalize names/shapes and register the published
GGUF as a separately source-bound model variant; then its standalone candidates,
target-corrected cycles, full category economics, public exactness, and lifecycle
must pass independently. It cannot silently share the supported b048 B4 claim.
Availability alone is not runtime support.

Both containers intentionally omit token embeddings and the LM head. The
drafter calls target-owned Q4_K embedding and Q6_K LM-head/sampler primitives
rather than duplicate or dequantize those complete tables; root/mask embedding
rows are materialized directly to BF16.

### DFlash data flow

The intended chain path is:

```text
1. Target AR/verify forward captures six post-layer hidden states.
2. Each target tap is normalized by its matching aux_hidden_norm.
3. Concatenate six 3072-wide taps -> 18432; fc projects -> 3072; hidden_norm.
4. Build a 16-row root/mask block using target embedding rows and mask token 12.
5. Run six BF16 Laguna SWA+dense draft layers with per-head softplus gates.
6. Final norm; target Q6_K LM head produces full-vocabulary draft logits.
7. Compile at most B candidate rows (initially B <= 7).
8. Target executes one exact B+1 verifier-shaped forward.
9. GPU accept walks the chain until first mismatch; commit accepted target
   hidden/KV/state rows transactionally and discard the rejected tail.
10. Reseed the drafter from committed target taps and repeat.
```

The six capture indices use two representations in upstream integrations:
`target_layer_ids=[1,10,19,29,38,47]` are zero-based layer IDs, while
`eagle_aux_hidden_state_layer_ids=[2,11,20,30,39,48]` identify post-layer
capture depths. hipEngine must normalize this once at load time and test the
boundary explicitly; an off-by-one capture silently destroys acceptance.

### Existing hipEngine DFlash reuse and gaps

hipEngine already has:

- provider-neutral chain `DraftBatch` compilation;
- torch-free DFlash metadata and safetensors loading;
- root/mask input preparation;
- target-hidden `fc + hidden_norm` projection;
- append-only projected-context and draft K/V caches;
- BF16 draft decoder and QKV fusion experiments;
- target B+1 chain/tree verification;
- GPU accept summaries and transactional selected-state commit;
- verifier graph/address contracts and gfx1151 registrations.

That code targets the older z-lab Qwen DFlash artifact. The Poolside checkpoint
requires deliberate generalization:

- accept `DFlashLagunaForCausalLM`, with block size and target-layer count nested
  under `dflash_config` rather than the old top-level schema;
- load six `aux_hidden_norms`;
- consume fused `qkv_proj.weight` via row views or a matching fused kernel rather
  than requiring separate Q/K/V tensors;
- load and execute `g_proj` with Laguna softplus head gating;
- execute six 3072-wide, 72-head SWA Laguna layers rather than the old Qwen draft
  layer contract;
- use the GGUF target's quantized embedding and LM head through target-owned
  primitives; the current DFlash target validator assumes F16/BF16 safetensors;
- capture Laguna target hidden states at the exact six configured boundaries;
- verify B+1 rows through the Laguna target runner, not the Qwen/PARO runner;
- include the extra 2.08 GiB weights and draft rings in memory admission.

DFlash is therefore not free, but it reuses both sides of the planned port: the
base Laguna SWA/gate primitives and hipEngine's existing speculative
verify/accept/commit framework.

## Forward-Pass Contract

The CPU oracle and GPU runner must implement this order. `x` is the layer input
and `n` is the attention pre-norm result.

```text
n = rmsnorm(x, attn_norm)

q = q_proj(n)
k = k_proj(n)
v = v_proj(n)
g = gate_proj(n)

q = head_rmsnorm(q, q_norm)
k = head_rmsnorm(k, k_norm)
q, k = rope_for_layer_type(q, k, absolute_positions)

a = causal_attention(q, k, v, live_spans_for_layer)
a = a * softplus(g)                 # one scalar per Q head, broadcast on dim
attn = o_proj(a)

r = x + attn
m = rmsnorm(r, ffn_norm)

if layer == 0:
    f = down_proj(silu(gate_proj(m)) * up_proj(m))
else:
    logits = router(m)
    probs = sigmoid(logits)
    selected = topk(probs + correction_bias, k=10)
    route_w = probs[selected]
    route_w = route_w / sum(route_w)

    routed = sum(route_w[e] * expert_e(m) for e in selected)
    routed = 2.5 * routed
    shared = shared_expert(m)
    f = routed + shared

x_next = r + f
```

Critical details:

- the attention gate is projected from the same pre-attention normalized hidden
  state as Q/K/V, not from the attention result;
- softplus is applied in FP32 before casting/multiplying at the activation ABI;
- per-head gate width is exactly 48 or 72 and broadcasts over 128 channels;
- correction bias changes top-k selection only;
- final routing weights come from the uncorrected sigmoid probabilities;
- selected probabilities are sum-normalized before applying the 2.5 scale;
- the shared expert is always evaluated and added independently;
- there are no QKV biases and no post-attention/post-FFN norms beyond the two
  pre-norms shown;
- RoPE uses absolute token positions even when SWA physical slots wrap.

## hipEngine Compatibility Map

| Requirement | Current reusable support | Gap / action |
| --- | --- | --- |
| GGUF v3 scan and lazy spans | `hipengine/loading/gguf.py` | Add Laguna metadata/config validation. |
| Q4_K/Q6_K dequant oracle | `hipengine/quant/gguf.py` and plugins | Reuse; add Laguna tensor fixtures. |
| Q4_K dense projections | native pack8/T16 GGUF kernels | Reuse after shape gates. |
| Q4_K token embedding | raw Q4_K/Q6_K/Q8_0 lookup is registered for gfx1100/gfx1151 | Reuse the BF16 row-dequant path for target tokens and DFlash root/mask rows. |
| rank-3 selected experts | Q4/Q5/Q6 T16/raw selected kernels | Reuse for 256 experts/top-10; validate exact rank-3 strides. |
| Q6_K LM head | native Q6 T16/GEMV path | Reuse untied output map. |
| F16 projections | source-preserving mixed BF16/F32-activation, F16-weight single/dual/triple GEMV is registered | Keep the exact GEMV for eager/decode and as the unfused fallback; the now-green bulk oracle activates the true rows>1 GEMM/WMMA work in L10/LPF-1. |
| F32 norms/router weights | dense F32 and RMSNorm support | Reuse; 3072-wide router needs its own exact launch gate. |
| head Q/K RMSNorm | existing Qwen full-attention primitives | Reuse with head dim 128 and variable Q-head counts. |
| paged attention/KV | `KVLiveSpans`, BF16 block-256 global attention/write, and token-granular SWA ring kernels | Eager c=1 ownership is closed: global layers use admitted context, SWA layers use 512 physical slots with absolute positions and eviction masks; bulk prefill remains L8. |
| dual RoPE | exact host YaRN/plain tables plus F32-weight head RMSNorm+partial/full rotary are registered | Reuse absolute-position tables per layer family; bound table residency to admitted context. |
| softplus attention gate | unfused FP32 per-head broadcast emits FP32 or BF16 | Reuse before F16 O projection; fuse only after whole-layer exact parity. |
| softmax router | existing Qwen route selection | Add Laguna sigmoid + correction-bias semantics. |
| shared expert | Qwen GGUF MoE machinery | Wire an always-on, ungated Laguna shared expert. |
| byte-BPE | Qwen GGUF tokenizer is close | Generalize BOS/EOT/control/template handling. |
| public generation/serving | GGUF resident model loop | Add a Laguna model/generator plugin, no engine branches. |
| gfx1151 backend | native target and measured GGUF defaults | Register/resolve Laguna variants and revalidate locally. |
| DFlash chain/verify/commit | provider-neutral DFlash scaffolding and gfx1151 kernels | Generalize old Qwen artifact/target assumptions to Poolside Laguna. |
| DFlash target roots | target owns Q4 embedding and Q6 LM head | Route draft embedding/logits through target primitives; do not duplicate tables. |
| DFlash hidden taps | target runner supports diagnostic hidden capture patterns | Make six configured post-layer taps stable and verifier/graph aware. |

The existing `qwen35_gguf_runner.py` is architecture-specific. Reuse its
primitive plans and extract genuinely model-neutral helpers where practical;
do not copy thousands of lines and rename Qwen to Laguna. Any temporary duplicate
runner or flag must be entered in [`REFACTOR.md`](REFACTOR.md) with a removal
trigger.

## Architectural Requirements

All implementation work must preserve these repository invariants:

1. **Torch-free hot path.** No `import torch` in anything reached by
   `hipengine.LLM.generate()`.
2. **Four-axis registry.** Laguna is a model plugin; Q4_K_M is a quant plugin;
   gfx1151 is a backend. Do not add `if backend == ...` or `if quant == ...` to
   engine/model dispatch.
3. **Raw-pointer kernels.** New HIP bodies and wrappers use device pointers and
   scalar shape metadata, not `torch::Tensor`.
4. **Fused plus unfused.** YaRN/rotate, softplus gate, and router composites need
   numerically equivalent unfused reference chains.
5. **`KVLiveSpans` ABI.** Both global and SWA attention consume live spans,
   token positions, and eviction masks. Do not regress to
   `(block_table, context_len)`.
6. **Backend peer discipline.** Register shared gfx11 bodies for
   `hip_gfx1151`; do not silently treat gfx1151 measurements as gfx1100 evidence.
7. **No benchmark gaming.** Correctness/performance prompts must be diverse and
   predeclared; no prompt-conditioned behavior or token-specific branches.

## Capacity Model

### Resident weights

Raw tensor data is 70.01 GiB. Device replacement layouts add bounded overhead:

- rank-3 Q4_K T16 stores 16 source blocks (2,304 bytes) in a 2,368-byte tile,
  about 2.78% overhead;
- Q6_K T16 is effectively size-neutral for 16 source blocks;
- rank-2 Q4 pack8 expands scale/min metadata modestly;
- F16/F32 source-preserving residents retain their byte widths;
- optional sidecars must remain off until separately budgeted and justified.

Planning estimate, not measured: **72-76 GiB resident weights**. Add roughly
1-3+ GiB for reusable scratch, allocator alignment, prefill workspaces, and
optional graph state. A materialization-plan estimator must calculate exact
bytes before the first full allocation and fail with an actionable error if the
requested session does not fit.

Because source file pages and HIP allocations share physical RAM, loader
transients matter even though page cache is reclaimable. Materialize one tensor
at a time, free host repack arrays promptly, and report both tracked allocation
and whole-device GTT during the first full load.

### BF16 KV

Per cached token per attention layer:

```text
8 KV heads * 128 dims * 2 bytes * (K + V) = 4096 bytes
```

With 12 global layers and 36 SWA layers capped at 512 tokens:

| Context | Global KV | SWA KV | Total BF16 KV |
| ---: | ---: | ---: | ---: |
| 4K | 192 MiB | 72 MiB | 264 MiB |
| 32K | 1.50 GiB | 72 MiB | 1.57 GiB |
| 64K | 3.00 GiB | 72 MiB | 3.07 GiB |
| 128K | 6.00 GiB | 72 MiB | 6.07 GiB |
| 256K | 12.00 GiB | 72 MiB | 12.07 GiB |
| 1M | 48.00 GiB | 72 MiB | 48.07 GiB |

Approximate single-request total planning ranges:

| Context | Weights + KV + initial scratch | Status |
| ---: | ---: | --- |
| 4K | 74-80 GiB | capacity-feasible; first target |
| 64K | 77-83 GiB | likely feasible after lifecycle gates |
| 128K | 80-86 GiB | likely feasible; repeated-run gate required |
| 256K | 86-94 GiB | plausible, not measured |
| 1M | above 120 GiB | rejected for BF16 KV on this host |

These ranges are not allocator measurements. They assume SWA physical capacity
is 512. Allocating 256K slots for all 48 layers would consume about 48 GiB of
KV instead of 12.07 GiB and would likely make the model fail admission. The
runtime must represent physical slot capacity separately from absolute position
and logical context.

Weights are shared across requests, while KV scales with live requests. Long-
context concurrency therefore needs admission based on available GTT and the
sum of per-request global/SWA KV. Do not infer c=2/c=4 256K capacity from c=1.

### INT8 KV and 1M

INT8 KV could make 1M capacity plausible, but existing Qwen/PARO KV8 quality
evidence does not transfer to Laguna. It remains out of initial scope. Any
Laguna KV8 mode needs a model-specific BF16 comparison over multiple prompts,
explicit relaxed/exact labeling, and the normal KL/top-1 gates.

### DFlash memory increment

The matched DFlash drafter adds about 2.08 GiB of BF16 weights. It shares target
embedding and LM-head allocations. Its bounded state is small relative to the
target:

- six target hidden taps at 3072 BF16 values are 36,864 bytes per context row;
  retaining a 512-row raw ring would cost about 18 MiB, while projection-on-
  commit can discard those raw rows sooner;
- six draft-layer BF16 K/V rings at eight heads, 128 dimensions, and 512 tokens
  total about 12 MiB per request;
- root/query, logits, top-k, verifier, and commit scratch still require an exact
  plan and ownership audit.

The measured 4K B4 category session owns 79,349,505,533 resident bytes:
77,099,132,853 target bytes (including lazy verifier resources) plus
2,250,372,680 drafter bytes. Tracked peak is 79,349,726,717 bytes and teardown
returns to zero, closing short-context capacity/lifecycle. A single-request 256K
target+DFlash session remains capacity-plausible on the 120 GiB GTT host but is
not measured. DFlash admission must reserve verifier-shaped target scratch and
must not rely on target-only headroom.

## Inferred Performance Model

This section is a roofline sketch, not a performance result.

The local gfx1151 practical read ceiling is about 221 GB/s. Laguna c=1 decode
appears to read approximately:

- 5.61 GB of F16 attention projections across all layers;
- about 2.9 GB of Q4/Q6 routed weights for ten selected experts across 47 layers;
- roughly 0.7 GB for routers, shared experts, layer-0 dense FFN, LM head, and
  remaining small families.

That is roughly **9-10 GB of active weight traffic per generated token**, even
though all 70 GiB of weights must remain resident. The short-context bandwidth
ceiling is therefore around **22-24 tok/s** before launch, dequant, cache,
reduction, and utilization losses. Actual throughput should be lower until
measured.

Long-context decode additionally reads K/V history. At 256K, the 12 global
layers contain about 12 GiB of BF16 K/V, reducing the optimistic combined
weight+KV bandwidth ceiling toward 10 tok/s before kernel inefficiency. This is
why context gates and long-context attention profiling remain separate from the
first 4K support milestone.

## Implementation Plan

### Dependency sequence

```text
L0 oracle/intake
  -> L1 CPU reference + fixture
  -> L2 model/config/tensor map + tokenizer
  -> L3 materialization + memory admission
  -> L4 attention/RoPE/gate kernels
  -> L5 MoE router/shared/routed experts
  -> L6 eager full-model session + stable hidden taps
  -> L7 public generation/chat/streaming
  -> L8 bulk prefill and graph replay
  -> L9 context/concurrency scaling
  -> L10 performance rollup

D0 DFlash artifact/schema support starts after L2/L4
  -> D1 standalone Laguna drafter parity
  -> D2 target hidden/context ownership
  -> D3 B+1 verify/accept/commit exactness (requires L6/L8 verifier rows)
     -> D4 full-suite economics and default-eligibility gate
     -> D5 explicit opt-in public/server integration
```

Do not begin target performance tuning before L6 eager correctness is green.
Do not begin DFlash economics before exact target AR and D3 state commit are
green.

### L0 — Freeze artifact and independent oracle

Work:

1. Finish the GGUF download.
2. Record final bytes and SHA-256.
3. Scan metadata and all tensor spans:

   ```bash
   python3 scripts/inspect_gguf.py \
     /home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf \
     --json --check-dequant --smoke-rows 1
   ```

4. Record and inspect the completed DFlash safetensors and/or BF16 GGUF:
   revision, hash, config, 69-tensor inventory, and target pairing metadata.
5. Build Poolside's read-only llama.cpp `laguna` branch natively for gfx1151;
   the upstream target-only PR is insufficient for the DFlash oracle.
6. Run a deterministic 4K, temperature-zero target AR completion with all
   layers GPU-visible and save exact prompt IDs, generated IDs, and runtime
   metadata.
7. Run Poolside's target+DFlash command first with seven draft tokens and then
   the documented maximum 15; record accepted lengths, generated IDs, and
   memory as oracle diagnostics, not hipEngine performance evidence.
8. Produce first-token/full-vocabulary logit or perplexity fixtures using the
   fork's supported diagnostic surface. Record the exact command and binary
   commit/hash.
9. Save resolved chat-template renderings for no-thinking, thinking, and a
   minimal tool-call conversation.

Acceptance:

- completed file has no out-of-range tensor spans;
- architecture and every shape/type match the documented contract;
- all Q4_K/Q6_K/F16/F32 dequant smokes are finite;
- llama.cpp target AR completes at 4K on gfx1151 with deterministic IDs;
- Poolside target+DFlash output is exact against its same-build target AR over
  the chosen oracle prompts, or any mismatch is recorded as a blocker;
- DFlash tensor inventory and capture-index convention are frozen;
- tokenizer and template fixtures are committed or reproducibly generated;
- no hipEngine performance claim is made.

#### Target oracle frozen on gfx1151 (2026-07-22)

The target-only part of L0 is frozen in
`tests/fixtures/laguna_poolside_v1_oracle.json` and
`tests/fixtures/laguna_poolside_v1_first_token_logprobs.npy`. The read-only
Poolside checkout is pinned at `04b2b72cb54048ead292884adbe11f284e3ec950`;
the native `llama-server` binary SHA-256 is
`1a3b09cfb9a8034d44239224ac362afce4555b85da376a3a7e1f4ecaffee0419`.
The accepted launch is:

```bash
GPU_MAX_HW_QUEUES=1 \
  build-hip-gfx1151/bin/llama-server \
  -m /home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf \
  --host 127.0.0.1 --port 18081 -c 4096 -ngl 999 \
  -fa off --jinja --parallel 1 --no-warmup --no-repack --no-mmap \
  --cache-ram 0 --metrics -lv 4
```

Exact prompt token IDs are sent to `/completion`; sending the rendered string
would add a second BOS. Two fresh-process captures agreed exactly on all 32
greedy IDs, beginning with token `94557` (`Deterministic`). Two fresh-process
full-vocabulary captures also agreed bit-for-bit over all 100,352 float32
pre-sampler log-probabilities. Those values are equivalent to raw logits up to
the shared additive log-normalizer and are the L6 KL/top-1 oracle.

The Poolside reference has three gfx1151 constraints that must remain visible:

- `--no-mmap` is required to select its pinned asynchronous ROCm upload path;
  fresh model readiness was 29.851-29.907 seconds, while the mmap path remained
  unready after 31 minutes;
- Poolside's `flash_attn_ext_f16` build faults on this device after reporting no
  compatible device code for HIP arch 1300, so correctness uses the unfused
  attention fallback with `-fa off`;
- a second sequential completion in the same Poolside server process diverged
  after a shared greedy prefix, even with prompt cache disabled. Oracle captures
  therefore use one request per fresh process. hipEngine's own repeated-state
  acceptance remains strict and may not inherit this limitation.

These are oracle diagnostics, not a hipEngine or llama.cpp performance claim.
The target+DFlash L0 diagnostic remains behind the later DFlash integration
milestone.

### L1 — CPU reference and tiny Laguna fixture

Likely paths:

- `hipengine/kernels/cpu_reference/` — Laguna primitives/unfused layer;
- `tests/fixtures/` — tiny deterministic Laguna GGUF or compact raw fixture;
- `tests/test_laguna_reference.py`.

The fixture should include at least two layers:

- one full layer with 48 Q heads, partial-64 YaRN, and a dense FFN;
- one SWA layer with 72 Q heads, full-128 plain RoPE, 256 tiny routed experts,
  top-10 routing, correction bias, and one shared expert.

Use smaller hidden/expert dimensions where the math permits, plus production-
shape metadata tests for dimensions that kernels constrain.

Required RED/GREEN units:

- YaRN tables and partial rotation;
- SWA plain RoPE;
- absolute position versus wrapped physical slot;
- per-head softplus broadcast;
- sigmoid top-k with correction-bias selection;
- proof that correction bias does not alter final route weights;
- route sum normalization and 2.5 scale;
- shared expert addition;
- residual/pre-norm order;
- tied-versus-untied output rejection;
- DFlash auxiliary-tap norm/concat/fc order;
- DFlash fused-QKV slicing, softplus gating, block mask, and six SWA layers;
- speculative chain accept/commit for reject/partial/full acceptance.

Acceptance:

- deterministic CPU outputs with checked-in expected values;
- correction-bias fixture selects a different expert than unbiased top-k while
  retaining the unbiased route value;
- global and SWA attention masks pass boundary cases at 511/512/513 tokens;
- no GPU dependency in the reference tests.

Implemented target foundation (2026-07-22):

- `hipengine/kernels/cpu_reference/laguna.py` now provides FP32 full-YaRN and
  SWA-plain split-half RoPE tables, partial rotation, absolute-position causal
  and sliding masks, per-head RMSNorm, gated GQA, dense SwiGLU, sigmoid/corrected
  top-10 routed experts, the independently added shared expert, exact residual
  order, and explicit untied LM-head logits;
- production RoPE fixtures cover full-layer partial-64 YaRN at positions
  `0/1/8191/8192/8193/262143` and SWA full-128 plain RoPE at the same positions.
  The GGUF `yarn_attn_factor=1` is treated as the multiplier before ggml's
  default `1 + 0.1*ln(factor)` magnitude correction, matching the independent
  Transformers output for factor 32;
- the compact two-layer fixture preserves the production 48/72 Q-head counts,
  eight KV heads, 256 experts, top-10 selection, factor 2.5, dense layer 0, and
  sparse SWA layer 1 while reducing hidden/head/expert widths where legal;
- checked-in expected intermediates and logits were captured from Hugging Face
  Transformers 5.12 against Poolside model revision
  `179ee67cf0fff5391c67fe1a392ea849fa6d643f`. The optional capture script uses
  torch only outside the runtime, and repeated capture is byte-exact;
- 14 target CPU-reference tests pass without torch or GPU, including independent
  Transformers parity, 511/512/513 mask boundaries, wrapped-slot absolute
  positions, correction-bias semantics, shared addition, residual order, and
  rejection of a missing untied output tensor.

Target-side L1 is closed. DFlash auxiliary-tap, fused-QKV, six-layer draft, and
accept/commit fixtures remain in the DFlash milestones rather than blocking the
target embedding/projection/MoE kernel path.

### L2 — Model, metadata, tensor map, and tokenizer plugins

Likely new paths (names may change during design review):

- `hipengine/models/laguna.py`;
- `hipengine/loading/laguna_gguf.py`;
- `hipengine/tokenization/laguna_gguf.py`, or a model-neutral generalization of
  the current GGUF byte-BPE tokenizer;
- `tests/test_laguna_gguf_config.py`;
- `tests/test_laguna_gguf_tensor_map.py`;
- `tests/test_laguna_gguf_tokenizer.py`.

Requirements:

- model plugin name separate from Qwen;
- `arch_names`/GGUF architecture includes `laguna`;
- layer specs encode `full_attention`, `sliding_attention`, `dense_mlp`, and
  Laguna MoE without backend/quant knowledge;
- per-layer Q heads and gate widths are validated;
- KV heads/head dimensions are uniform and validated;
- dual RoPE fields are mandatory when SWA is present;
- tensor map distinguishes root, dense layer, routed experts, shared expert,
  router, and correction bias;
- output falls back to tied embeddings only if `output.weight` is actually
  absent; this GGUF must use the untied Q6_K tensor;
- tokenizer loads BOS/EOS/EOT/PAD and suppresses EOT text;
- chat template is read from GGUF metadata and rendered through the existing
  safe template surface.

Implemented foundation (2026-07-22):

- the model/config and strict 814-tensor mapping plugins are registered and
  covered by synthetic plus completed-artifact tests;
- `LagunaGGUFTokenizer` reconstructs the HF Rust encoder directly from the
  `gpt2`/`laguna` GGUF metadata, with BOS 2, EOS 2, EOT 24, PAD 9, SEP 8, MASK
  12, UNK 0, BOS insertion, stop IDs, and the raw chat template while
  suppressing EOT text under special-token skipping; the superseded Python BPE,
  Python pre-tokenizer, and unbounded cache have been removed;
- five checked-in prompt fixtures plus CRLF, newline/punctuation, and combining-
  mark boundaries match Poolside's HF fast tokenizer at revision
  `179ee67cf0fff5391c67fe1a392ea849fa6d643f`; an expanded 23-case local
  comparison also matched, including atomic chat/control tokens;
- five rendered no-thinking/thinking/tool/history fixtures match both the HF
  tokenizer and the pinned Poolside llama.cpp tokenization; the independent
  target oracle now fixes the exact 55-token no-thinking prompt, full first-step
  distribution, and 32 canonical greedy IDs.

Acceptance:

- scanner resolves `laguna` to the Laguna model plugin;
- all 814 tensors are consumed exactly once or explicitly documented metadata;
- malformed head arrays, gate widths, missing correction bias, wrong expert
  rank, and wrong root tying fail before allocation;
- tokenizer IDs and rendered prompts match L0 oracle fixtures exactly;
- importing/configuring the plugin does not import torch.

### L3 — Materialization and memory admission

Likely paths:

- `hipengine/loading/laguna_gguf_materialize.py`, or a model-neutral extraction
  from `qwen35_gguf_materialize.py`;
- targeted materialization tests and a dry-run byte estimator.

Implemented foundation (2026-07-22):

- the dry planner covers all 814 tensors exactly once and preserves every F16
  attention/F32 metadata tensor at source precision;
- rank-2 Q4_K projections use existing pack8 allocation contracts, rank-3
  selected experts use Q4_K/Q6_K T16 replacement layouts, Q6_K rank-2 down
  projections remain raw, and the Q4_K embedding remains raw pending its native
  lookup kernel;
- 4K BF16 KV uses 12 full-context layers plus 36 512-token SWA rings;
- with measured 119.996 GiB HIP-free memory, the current dry estimate is
  71.468 GiB resident weights + 0.258 GiB KV + 2 GiB scratch + 1.230 GiB maximum
  loader transient + 8 GiB reserve = 82.956 GiB peak and 37.040 GiB headroom;
- the streaming device loader now implements owned dense FP16/F32, raw GGUF,
  Q4_K pack8, and Q4_K/Q6_K T16 allocations with exact planned-byte checks and
  failure cleanup; fake-runtime payload/teardown tests are green;
- live gfx1151 selected-slot materialization now passes for FP32, FP16, Q4_K
  pack8, and tiny synthetic Q4T16/Q6T16 payloads with exact D2H readback; five
  tracked allocations / 2,666,496 bytes return to zero after teardown;
- the first full 814-tensor gfx1151 load/free smoke completed in 239.427 s:
  1,054 tracked allocations / 76,737,907,712 bytes (71.468 GiB), 77,024,133,120
  bytes sampled GTT use, and 329,875,456 bytes sampled peak host RSS; teardown
  took 0.373 s and returned tracked allocations/bytes to zero;
- post-free GTT retained 173,998,080 bytes versus 22,913,024 before load and HIP
  free retained the matching one-time context/allocator footprint; no tracked
  resident-weight allocation leaked. L3 is closed for target weight residency;
  the L4 KV owner is now closed, while scratch ownership remains in the eager-
  session gate.

Structured natural-path loading telemetry now records each tensor's source-map,
CPU-repack, HIP-allocation, and H2D wall; faults and physical read bytes are
attributed to the phase where lazy GGUF pages are actually touched. The load
smoke also records `/proc/self/io`, minor/major faults, RSS/high-water, HIP/GTT,
layout aggregates, and the 20 slowest tensors. Its cache-state argument is a
label only; the result independently classifies observed physical reads as
warm, partial, or cold-streamed.

The first complete profiled run used:

```bash
HIPENGINE_HIP_ARCH=gfx1151 GPU_MAX_HW_QUEUES=1 \
uv run python -u scripts/laguna_gguf_load_smoke.py \
  /home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf \
  --backend hip_gfx1151 --context-length 4096 --progress-every 25 \
  --profile-tensors --cache-state warm \
  --model-sha256 7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f \
  --output /tmp/laguna-load-profile-warm.json
```

Despite the declared label, `/proc/self/io` measured 73,051,406,336 physical
read bytes for 75,169,369,088 source bytes, so this was observably a
**cold-streamed**, not warm-cached, load. Total load was 227.510 s. The 814
per-tensor intervals account for 225.536 s: CPU replacement-layout repack is
217.278 s (96.3% of total wall), HIP allocation is 3.641 s, H2D is 4.279 s,
source-map setup is 0.107 s, and classified other work is 0.231 s. Q4T16 alone
uses 160.603 s, Q6T16 52.259 s, and pack8 4.416 s. Process counters record
21,658 major and 3,989,376 minor faults; max process RSS reached 1,588,633,600
bytes and sampled GTT reached 77,024,133,120 bytes. Teardown took 0.226 s and
returned all 1,054 tracked allocations / 76,737,907,712 bytes exactly.

This proves the startup gap is not primarily `hipMalloc` or H2D. It is the
current Python/NumPy whole-model Q4/Q6 T16 transform while faulting 68 GiB of
source pages. Poolside's accepted `--no-repack --no-mmap` path avoids that
persistent transform and reaches readiness in 29.851-29.907 s. The ordered
optimization targets are therefore a versioned repacked artifact cache or
compiled/GPU T16 transform first, followed by arena/pinned/overlapped upload
only after new telemetry shows those phases have become material.

The first retained startup optimization is now a versioned, source-bound
replacement-layout cache. `scripts/laguna_repacked_cache.py` atomically builds
and validates `laguna-repacked-v1`: 262 Q4T16/Q6T16/pack8 entries totaling
70,718,767,104 bytes, keyed by the full materialization-plan fingerprint,
source size/mtime, and optional precomputed GGUF SHA-256. Direct F16/F32/raw
weights continue to stream from the original GGUF, so the cache does not add a
second copy of those 552 tensors. A failed/incomplete build never replaces an
existing artifact.

Reading each cached tensor into one bounded host array before H2D is critical.
The rejected direct-mmap upload took 82.811 s because synchronous HIP copies
faulted mapped pages one at a time. Buffered sequential reads reduce the three
steady cold-streamed loads to 48.812/47.951/48.202 s (median 48.202 s), with
72.1-72.3 GB of measured physical reads, zero repack time, and exact tracked
teardown. This is **78.81% less startup wall / 4.72x faster** than the 227.510 s
natural source-repack baseline. A partially cached diagnostic reached 28.655 s,
but is not the cold-start claim. The cold median remains 1.61x slower than
Poolside's 29.851-29.907 s native-GGUF startup, with ~40.4-41.6 s now in
sequential source reads and ~6.5-6.8 s in allocation/upload; overlap/pinned
staging is therefore the next loader target rather than more repack work.

The cache is wired through the full resident session and frozen-oracle harness.
A cache-backed run reproduces the prior result exactly: first token `94557`, KL
`6.6214e-6`, the same 29-token strict Poolside prefix, repeat logits max-abs
`0`, teacher-forced top-1 31/32, finite taps, and exact lifecycle recovery. Cache
artifacts remain local and must never be committed.

Plan:

- keep F32 norms/router/correction tensors dense F32;
- keep F16 attention weights source-preserving F16 for the initial exact path;
- reuse Q4 pack8/T16 and Q6 T16/raw layouts where their shape contracts pass;
- retain rank-3 expert tensor semantics and verify expert/out/K strides;
- keep optional sidecars off;
- estimate resident bytes, KV bytes, scratch, and loader transient headroom
  before calling `hipMalloc`;
- allocate global-layer KV at requested context and SWA KV at 512 physical slots;
- report tracked resident bytes and whole-device GTT before/after load;
- materialize/repack one large tensor at a time and release host temporaries.

Acceptance:

- dry plan covers every mapped tensor and reports exact allocation names/layouts;
- no F16 attention tensor is silently contracted to BF16;
- rank-3 repack round-trips raw bytes in tests;
- 4K admission succeeds with measured headroom on gfx1151;
- forced over-budget and naïve-all-layers-full-KV plans fail before allocation;
- all allocations free cleanly after a load-only smoke.

### Post-L3 runtime gap audit (2026-07-22)

The full resident load proves capacity and ownership, not executability. A fresh
registry/runtime audit at hipEngine `d4d7b3ae8` used the completed S 2.1 artifact
and concrete `hip_gfx1151` package. The authoritative target reference is
Poolside llama.cpp
[`04b2b72c`](https://github.com/poolsideai/llama.cpp/blob/04b2b72cb54048ead292884adbe11f284e3ec950/src/models/laguna.cpp#L156-L350):
it selects interleaved SWA ownership, separate full/SWA RoPE inputs, FP32
softplus gating, selection-biased sigmoid MoE, an independent shared expert, and
the pre-final-norm DFlash capture. Its exact S 2.1 template is
[`poolside-Laguna-S-2.1.jinja`](https://github.com/poolsideai/llama.cpp/blob/04b2b72cb54048ead292884adbe11f284e3ec950/models/templates/poolside-Laguna-S-2.1.jinja#L1-L92).

| Forward stage | Existing concrete capability | Blocking gap / required gate |
| --- | --- | --- |
| Q4_K token embedding | Raw `embedding/gguf_q4_k/lookup_bf16_out` is registered for gfx1100/gfx1151 and the Laguna table stays source-native | Closed: synthetic and real rows are BF16-exact vs CPU, invalid IDs preserve caller rows, model-neutral resident dispatch resolves, and gfx1151 profiling shows `gguf_q4_k_embedding_bf16_out_kernel`. |
| F32 RMSNorm / residual | GGUF BF16-input/F32-weight RMSNorm and add-RMSNorm are reusable | Wire under Laguna keys and prove layer-0 residual order; no new math is implied. |
| F16 Q/K/V/gate/O projections | Source precision and pointers remain F16; registry-driven single/dual/triple kernels accept BF16/F32 activations and emit FP32/BF16 | Closed and promoted on gfx1151: eager/decode keeps exact GEMV; rows>=2 resolve the reduction-order-preserving 8x4/16x4 tile. CPU/primitive parity plus same-session rows 2..128 and the full category gate are exact; canonical prefill improves 23.333->48.560 tok/s. Unsupported backends retain GEMV. |
| Q/K head norm and RoPE | Exact Laguna YaRN/plain host tables feed the registered F32-input/F32-weight head-norm+rotate body | Closed for eager/bulk math: absolute positions, partial 64/full 128, 48/72 Q heads, eight KV heads, and dim 128 pass CPU parity on gfx1151. |
| Global BF16 KV/attention | Complete dense `KVLiveSpans` plus block-256 paged BF16 writer/context attention accepts GQA ratios 6 and 9 and head dim 128 | Closed for eager c=1: the Laguna body preserves the proven block-256 page-table structure while consuming absolute positions and eviction metadata, 48/8 GQA matches direct CPU attention, and softplus remains the separate exact next stage. |
| 512-token SWA | `KVLiveSpans.sliding_ring` plus native BF16 writer/context attention | Closed for eager c=1: 36 physical rings carry slot offsets, live counts, absolute token positions, eviction masks, and absolute query positions; 72/8 GQA passes 510/511/512/513 and repeated 1024/1025 wraps plus explicit eviction. Bulk prefill remains L8. |
| Per-head attention gate | Unfused `attention_gate/f32/softplus_broadcast_{f32,bf16}_out` is registered on gfx1100/gfx1151 | Closed: 72-head × 128-channel broadcast, extreme gate logits, and FP32/BF16 outputs match CPU; no fused path is added yet. |
| Dense layer-0 MLP | Q4_K pack8 gate/up, raw Q6_K down, SiLU, and residual primitives are reusable | Wired into the complete 48-layer eager step; the frozen first-token distribution passes at KL `6.62e-6` with exact top-1. The only greedy-32 mismatch is the documented token-30 low-margin arithmetic split. |
| Router projection | BF16 hidden × F32 router weight → FP32 logits plus a separate `laguna_sigmoid_router_topk/f32/correction_bias` stage | Closed for eager c=1: stable sigmoid, separate uncorrected/corrected score buffers, lower-ID tie stability, top-10/256, unbiased gathered normalization, and a distinct 2.5-scaled weight buffer pass adversarial CPU parity. No Qwen softmax/shared-gate route is reused. |
| Routed experts | Direct Laguna plan resolves rank-3 Q4T16 dual gate/up plus layer-specific Q4T16 or Q6T16 down under exact gfx1151 registry keys | Closed for eager c=1: the real artifact's 24 Q4/23 Q6 down split and production 3072/1024 top-10 execution validate source byte strides, T16 allocation strides, nontrivial selected IDs, separate SiLU, scaled weighted sum, CPU hidden tolerance, and intended selected kernels. Bulk rows remain L8. |
| Shared expert | Rank-2 Q4_K pack8 gate/up plus layer-specific Q4_K pack8 or raw Q6_K down run in an independent always-on branch | Closed for eager c=1: the real 24 Q4/23 Q6 split executes independently; separate gate/up → SiLU → down is added without a shared sigmoid gate or 2.5 routed scale. The unfused staged chain remains the model path. |
| Final norm / Q6_K LM head | F32-weight RMSNorm, resident Q6T16 BF16→F32 linear sourced losslessly from raw Q6_K, GPU argmax, and sampler primitives are registered | Closed at root-probe scope: full 100,352-way logits are finite, KL is `6.87e-13` vs raw-Q6 CPU math, and top-1 is exactly `81364`; preserve full logits for later whole-model oracle gates. |
| Session and hidden taps | `LagunaGGUFResidentSession` owns all 814 weights, exact c=1 scratch, dual RoPE, global/SWA KV, logits/argmax, and optional caller-owned BF16 taps at depths 2/11/20/30/39/48 | The 55-token oracle run is finite, repeat-exact, teardown-exact, and matches 29/32 autoregressive IDs; oracle teacher forcing is 31/32, isolating one low-margin branch rather than a cascading state bug. |
| Public generator | Generic engine loop and server lifecycle exist | Closed for the initial c=1 boundary: concrete Laguna registration owns resident weights plus isolated sessions, supports blocking/streaming preformatted completion, suppresses EOT/stops, reports truthful routing metadata, and frees through public `LLM.close()`. Bulk rows and performance promotion remain L8/L9. |
| Reasoning | Model-owned Poolside renderer plus assistant-scoped `poolside_v1` prompt state feed the generic blocking/live/buffered splitter | Closed for c=1 server parsing: frozen deterministic scope/control fixtures pass, and the live gfx1151 gate proves thinking-disabled EOT plus prompt-open non-empty reasoning with exact blocking/stream IDs and no marker leakage. |
| Tools | Model-owned `PoolsideV1ToolParser` feeds the generic strict OpenAI tool-result surface | Closed for c=1 server parsing: deterministic XML/newline-less/typed/malformed fixtures pass, and live gfx1151 blocking/streaming emits exact mixed text+single, adjacent-multiple, and escaped UTF-8 OpenAI calls with stable IDs and truthful `poolside_v1_xml` capability. |
| Independent oracle | Closed for target AR: clean local Poolside checkout/build at `04b2b72c`, exact template/token fixtures, a complete 100,352-way first-token distribution, and 32 fresh-process-stable greedy IDs are checked in | Use the frozen fixture for L6 KL/top-1/greedy gates. Keep `-fa off`, `--no-mmap`, exact token-ID input, and fresh-process oracle constraints visible; target+DFlash diagnostics remain later work. |

Implemented root primitive slice (2026-07-22):

- the shared gfx11 embedding family now dequantizes raw GGUF Q4_K rows directly
  to BF16 under `embedding/gguf_q4_k/lookup_bf16_out`; gfx1151 receives the
  normal backend alias rather than a model/backend branch;
- `GGUFDeviceWeight` is a structural runtime ABI, so Laguna and Qwen resident
  owners use the same embedding/linear dispatch without importing a model-owned
  weight type into those dispatch functions;
- CPU and HIP tests cover exact Q4_K row selection, negative/out-of-vocabulary
  IDs (caller output rows remain untouched), registry resolution on gfx1100 and
  gfx1151, and the completed S 2.1 artifact;
- `scripts/laguna_root_probe.py` materializes only the Q4_K embedding, F32 final
  norm, and losslessly repacked Q6T16 LM head. For BOS `100257`, gfx1151 matches
  raw-GGUF CPU math with embedding/norm max-abs `0`, logits max/mean abs
  `5.72e-6`/`6.25e-7`, KL `6.87e-13`, and top-1 `81364 == 81364` over all
  100,352 logits;
- cached `rocprofv3 --kernel-trace` records Q4 lookup `9.818 us` (16 VGPR,
  zero scratch/LDS), RMSNorm `7.614 us`, Q6T16 LM head `1.213 ms` (72 VGPR,
  zero scratch, 512 B LDS), and argmax stage 1/2 `4.769/1.563 us`.

The broad source-lineage scan is currently blocked before any report because the
manifest's read-only `/home/lhl/amd-gpu-tuning/reference/atlas` and
`/home/lhl/amd-gpu-tuning/nano-vllm-amd` checkouts are absent. New Laguna work
must use Poolside's pinned commit above as its architecture source, record
in-tree extensions as new kernels rather than pretending to refresh unavailable
Qwen parents, and rerun the lineage checker if those checkouts are restored.

Ordered critical path:

```text
Poolside llama.cpp target oracle + template/token fixtures
  -> CPU YaRN/SWA/full-layer fixtures
  -> Q4 embedding + mixed-F16 projection/head-gate primitives
  -> global/SWA KVLiveSpans attention and Laguna MoE
  -> eager layer/full-model session + stable hidden taps
  -> first-logit/32-token oracle gate
  -> public blocking/streaming generation
  -> poolside_v1 parsing + server conformance
  -> bulk prefill/verifier rows
  -> exact target AR benchmark and bulk-default promotion (closed at c=1/4K)
  -> Laguna DFlash drafter/target verify/economics
```

Do not use the existing Qwen generator as a renamed shortcut: its linear
attention state, sigmoid attention gate, softmax router/shared-gate semantics,
and uniform full-attention KV owner are all wrong for Laguna.

### L4 — Attention, RoPE, SWA, and output gate

Likely kernel work:

- CPU/reference tables first;
- shared gfx11 YaRN/dual-RoPE primitive registered for `hip_gfx1151`;
- per-head softplus gate primitive;
- optional fused `attention_output + softplus_head_gate` only after unfused
  parity;
- paged-attention wrapper/planner support for 48/72 Q heads and eight KV heads;
- SWA ring/live-span management with absolute token positions.

Correctness cases:

- positions around original YaRN context and SWA wrap;
- Q-head/KV-head ratios 6 and 9;
- partial 64 versus full 128 rotation;
- global causality versus SWA 512 visibility;
- per-head gate broadcast over 128 channels;
- F16 projection input with BF16 activation/output contracts;
- prefill and decode produce equivalent final state for the same tokens.

Acceptance for every new/ported kernel:

- bit-exact or tolerance-scoped primitive test versus CPU reference;
- model-level KL <= 0.05 and top-1 agreement >= 90%;
- `rocprofv3 --kernel-trace` shows the expected kernel name and plausible
  duration on gfx1151;
- fused output matches the unfused chain;
- `KVLiveSpans` remains the only attention/KV ABI.

Implemented L4 primitive foundation (2026-07-22):

- source-F16 Q/K/V/gate/O projections now accept BF16 or FP32 activations and
  emit FP32 or BF16; single/dual/triple registry paths preserve all resident F16
  bytes and cover 48/72 Q heads plus eight KV heads at dimension 128;
- full-layer partial-64 YaRN and SWA full-128 plain tables are generated by the
  independently validated CPU equations, materialized as bounded FP32 tables,
  and indexed by absolute positions rather than wrapped slots;
- the existing exact F32-input/F32-weight head RMSNorm+rotate body is exposed
  under Laguna registry keys and passes production-head CPU parity for both RoPE
  contracts;
- an unfused FP32 per-head softplus broadcast emits FP32 or BF16 and passes
  extreme-logit/broadcast parity. No speculative fusion is present, so this is
  also the required unfused fallback;
- cached gfx1151 traces show mixed projection, head-normalize/rotate, and
  softplus kernels with expected names, zero scratch, and plausible durations;
- `KVLiveSpans.sliding_ring` now makes physical slot mapping, live count,
  absolute per-slot positions, eviction state, and absolute query position
  mandatory for SWA. The in-tree writer overwrites `position % 512`, updates
  metadata on device, and the reader applies `query - 512 < key <= query`;
- `LagunaKVCache` allocates the production 12 global + 36 SWA split. At 4K its
  exact BF16 K/V payload is 264 MiB; 243 tracked payload/metadata allocations
  free back to the pre-allocation counter baseline. Dense global spans fill the
  same absolute-position and eviction fields as SWA instead of using a legacy
  naked `(block_table, context_len)` bridge;
- token-serial gfx1151 tests match direct CPU attention for 48/8 and 72/8 GQA
  at positions 510/511/512/513 and after repeated wraps at 1024/1025, including
  one explicitly evicted live slot;
- cached profiling records native global/SWA writers at `1.603/1.523 us`
  median over 1026 launches each (16 VGPR, zero scratch/LDS), the complete-span
  global reader at `730.650 us`, and the correctness-first SWA reader at
  `1123.828 us` median over six boundary launches (16 VGPR, zero scratch,
  1024 B LDS). These are dispatch diagnostics, not a throughput claim.

L4 is closed for eager c=1 attention, dual RoPE, and unfused output gating.
Bulk/prefill SWA execution remains deliberately in L8 after the full-model eager
oracle is green.

### L5 — Laguna MoE

Start with a clear unfused chain:

1. F32 router projection from normalized BF16 hidden.
2. FP32 sigmoid probabilities.
3. Add FP32 correction bias into a separate selection score buffer.
4. Top-10 expert selection.
5. Gather uncorrected probabilities for selected IDs.
6. Sum-normalize selected route weights.
7. Execute selected Q4/Q6 expert gate/up/down.
8. Weight and sum routed outputs, then multiply by 2.5.
9. Execute the shared expert independently.
10. Add routed and shared outputs.

Reuse existing selected-expert kernels only after validating:

- 256 experts;
- top-k 10;
- expert-major source/T16 strides;
- Q4 gate/up plus mixed Q4/Q6 down layouts;
- production hidden 3072 and expert width 1024;
- duplicate expert IDs across batch rows;
- no dependency on Qwen shared-gate semantics.

An optimized fused sigmoid/correction/top-k router may follow, but it must retain
separate buffers or equivalent semantics proving the correction does not leak
into route weights.

Acceptance:

- selected IDs and route weights match CPU reference over adversarial logits,
  ties, large magnitudes, and correction-bias flips;
- route weights sum to one before the 2.5 output scale;
- routed/shared/combined hidden outputs pass fixture tolerances;
- rank-3 selected kernels pass Q4/Q6 raw-byte oracle tests;
- production-shape gfx1151 trace confirms intended selected kernels run.

Implemented eager c=1 foundation (2026-07-22):

- `laguna_sigmoid_router_topk/f32/correction_bias` is a separate native key
  after the reused BF16-hidden/F32-weight router projection. It emits full
  unbiased sigmoid probabilities and separate corrected selection scores,
  uses stable lower-expert-ID ties, gathers route values only from the unbiased
  buffer, sum-normalizes, and writes both unscaled and 2.5-scaled weights;
- `resolve_laguna_moe_plan(...)` requires the sigmoid/normalized model contract
  and exact gfx1151 registrations for router, Q4T16 selected dual gate/up,
  separate BF16 SiLU, both Q4T16/Q6T16 selected-down keys, weighted sum,
  shared SiLU, and add;
- `validate_laguna_moe_layer(...)` checks all eight sparse weight families,
  rank-3 source shapes/byte strides, replacement layout keys, and complete T16
  allocation strides before launch. The staged owner allocates and frees every
  c=1 intermediate explicitly;
- the production-width test uses hidden 3072, routed/shared width 1024, top-10,
  nontrivial selected IDs, Q4T16 gate/up, parametrized Q4T16/Q6T16 down,
  Q4-pack8 shared gate/up, and matching Q4-pack8/raw-Q6 shared down. The real
  artifact contains 24 Q4 and 23 Q6 routed/shared down layers. The reused direct
  T16 bodies retain their existing
  byte-exact duplicate-expert-ID multi-row coverage; Laguna bulk orchestration
  remains L8. Routed, shared, and combined hidden relative L2 each
  stay at or below `0.02` versus a BF16-staged raw-GGUF CPU oracle; the shared
  result is added independently and is not multiplied by 2.5;
- adversarial 256-expert tests cover logits from -100 to 100, equal maxima
  crossing wave32 boundaries, correction-bias selection flips, exact selected
  IDs, finite sigmoid values, normalized weights, and proof that correction
  never leaks into the gathered route values;
- cache-only gfx1151 profiling records the 3072×256 router projection at
  `10.820 us` (24 VGPR, zero scratch) and three-row 256-expert selection at
  `33.623 us` (32 VGPR, 512 B LDS, zero scratch). The production-width top-10
  chain records Q4T16 dual gate/up `250.550 us`, Q6T16 down `121.628 us`,
  shared Q4 gate/up `20.799/34.264 us`, shared Q6 down `76.183 us`, and zero
  scratch throughout. These are correctness/dispatch diagnostics, not a model
  throughput claim.

L5 is closed for eager c=1 sparse MoE, including the real artifact's mixed
Q4/Q6 down layouts. The complete layer/session owner is now implemented in L6;
batched routing and expert execution remain L8 after the frozen whole-model
oracle passes.

### L6 — Eager full-model resident session

Likely paths:

- `hipengine/runtime/laguna_gguf_runner.py`;
- `scripts/laguna_gguf_smoke.py`;
- `scripts/laguna_gguf_correctness.py`;
- focused end-to-end tests guarded on HIP availability.

First runner constraints:

- c=1;
- 4K max context;
- eager only;
- greedy/argmax only;
- token-serial prefill is allowed as a diagnostic fallback;
- BF16 KV;
- all weights resident;
- explicit layer-boundary diagnostic taps available outside the hot path;
- stable optional post-layer captures at depths 2, 11, 20, 30, 39, and 48,
  written directly into caller-owned target-hidden storage when requested.

Hidden taps are part of the base design because the matched DFlash artifact
requires them. They remain inactive in normal AR unless requested and must not
add steady-state copies or allocations to target-only generation.

Bring-up order:

1. embedding and final LM-head probes;
2. layer 0 dense forward;
3. first SWA MoE layer;
4. one full four-layer period;
5. all 48 layers, one token;
6. prompt prefill and first generated token;
7. 32+ greedy output tokens.

If the full model diverges, bisect saved hidden states at:

- attention pre-norm;
- Q/K after norm and RoPE;
- attention context before/after softplus gate;
- post-`o_proj` residual;
- router logits/probabilities/selected IDs/weights;
- routed, shared, and combined expert outputs;
- final layer and logits.

Acceptance:

- finite hidden/logits at every layer;
- tokenizer IDs match oracle exactly;
- first-token logit KL <= 0.05 and top-1 agreement >= 90%;
- deterministic greedy IDs match the oracle over the agreed output horizon;
- repeated eager state transitions are stable;
- closing the session returns tracked/GTT memory to the expected baseline;
- no torch import appears on the generation path.

Implemented eager resident foundation (2026-07-22):

- `hipengine/runtime/laguna_gguf_runner.py` resolves an exact gfx1151 plan and
  keeps all 814 replacement weights, 4K BF16 global/SWA KV, bounded c=1
  scratch, both RoPE tables, full-vocabulary FP32 logits, and top-1 state under
  one idempotent owner;
- layer execution is the unfused reference order: BF16/F32-weight RMSNorm,
  source-F16 Q/K/V/gate, per-family head norm+RoPE, complete-span KV append and
  context, FP32 softplus gate, source-F16 output, add+FFN norm, then dense or
  sigmoid-routed+shared FFN and residual add;
- optional DFlash taps are caller-owned one-row BF16 destinations at exactly
  depths 2/11/20/30/39/48. Normal AR allocates no tap storage and issues no
  tap copy; requested taps are direct stream-ordered D2D writes;
- all JIT libraries are held once by the session; constructor failure and
  execution failure clean every owned buffer, shared weights are not freed by a
  borrowing session, and owned close returns tracked allocations exactly;
- a cache-only full-artifact smoke loaded 77,022,439,484 owned bytes in
  206.291 s, executed all 48 layers for BOS in 77.221 ms, produced finite taps
  and greedy token 72, and returned a 77,022,476,348-byte/1,347-allocation
  tracked high-water exactly to zero.

This closes construction, one-token execution, hidden-tap ABI, and lifecycle.
It is not a throughput claim.

Frozen-oracle validation used:

```bash
HIPENGINE_HIP_ARCH=gfx1151 GPU_MAX_HW_QUEUES=1 \
uv run python -u scripts/laguna_gguf_correctness.py \
  --backend hip_gfx1151 --greedy-tokens 32 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build \
  --output /tmp/laguna-eager-correctness-final.json
```

The 55-token prompt loaded in 221.855 s and the first autoregressive pass took
4.946 s. First-token top-1 is exactly `94557`; full-vocabulary KL against the
Poolside log-probability oracle is `6.6214e-6`, well inside the `0.05` gate.
All six hidden taps are finite, a second independently allocated runtime state
repeats the complete hipEngine 32-token sequence exactly with first-logit
max-abs `0`, and tracked allocations/bytes return exactly to zero.

The strict 32-ID oracle clause remains blocked at one arithmetic decision
boundary. hipEngine and Poolside agree on the first 29 generated IDs. For token
30, Poolside chooses `604` over `372` by only `0.0342368` log-probability, while
hipEngine chooses `372` over `604` by `0.109314` raw-logit. Replaying all prior
Poolside IDs under teacher forcing gives 31/32 top-1 agreement; tokens 31 and 32
return to exact local top-1 after forcing token `604`. Thus the later natural
mismatch is a consequence of one low-margin branch, not corrupted KV/state
progression. An explicit FP16-KV bisection retained the same 29-token prefix and
branch while worsening first-token KL to `7.4248e-5`; that experimental path
was removed and BF16 KV remains the source-of-truth default.

The script deliberately reports `pass=false` while exact greedy-32 parity is
not met. This is the documented arithmetic-bisection outcome allowed by the L6
task, not an exact-output or throughput claim. Public direct/eager integration
may proceed, but performance promotion and target+DFlash economics remain
subject to their independent correctness gates.

### L7 — Public generation, streaming, chat, and tools

Likely path:

- `hipengine/generation/laguna_gguf.py`;
- model/backend/quant generation registry entries;
- server/public API tests.

Sequence:

1. preformatted prompt through `LLM.generate()`;
2. blocking chat completion using the GGUF template;
3. streaming with correct ownership and EOT suppression;
4. thinking disabled/enabled prompt rendering;
5. minimal tool declaration and parsed tool call;
6. cancellation, max-token finish, stop sequences, and cleanup.

Keep tool parser behavior independent from core model math. Poolside's llama.cpp
changes around optional whitespace and additional stops are compatibility
references; port only behavior proven necessary by fixtures.

Acceptance:

- public blocking and streaming IDs/text match direct eager generation;
- EOT 24 stops without leaking `</assistant>`;
- usage and finish reasons are correct;
- no active request/KV/allocation remains after finish or cancellation;
- `/ready`, capabilities, and model metadata report `hip_gfx1151`, Laguna, and
  Q4_K_M truthfully;
- unsupported modes fail closed with actionable errors.

Implemented initial public c=1 boundary (2026-07-22):

- `hipengine/generation/laguna_gguf.py` is registered only for the concrete
  `(laguna_gguf, hip_gfx1151, gguf_q4_k_m)` key. `backend="auto"` and
  `quant="auto"` resolve to those concrete values on this host; gfx1100,
  c>1, speculative, non-greedy sampling, logprobs, non-BF16 KV, and unsupported
  processors fail closed instead of silently taking another model path;
- the generator discovers a validated sibling `laguna-repacked-v1` artifact,
  owns one immutable resident weight set, and allocates isolated 4K BF16
  KV/scratch state per blocking or streaming request. `LLM.close()` now reaches
  compatibility-generator ownership and frees the shared 71.47 GiB weight set;
- both text and exact-token prompts use the Laguna GGUF tokenizer without
  implicit BOS insertion. Default EOS/EOT, caller stop IDs/sequences,
  `min_tokens`, `ignore_eos`, max-token finish, deadlines, and cooperative
  cancellation are explicit. Stop suffixes are withheld from streaming text;
  EOT 24 never leaks `</assistant>`, and incremental UTF-8 decoding does not
  emit transient replacement glyphs;
- the first public support boundary is **preformatted completion**. Poolside-v1
  deterministic reasoning/tool parser gates are now green, but the full live-
  model server transcript gate remains separate; parser fixtures must not be
  advertised as model-quality validation merely because raw completion works.

The committed `scripts/laguna_public_correctness.py` gate compares public
blocking, public streaming, and direct eager execution while sharing only the
immutable weights. On the frozen 55-token prompt and 32-token horizon all three
produce the exact same 32 IDs, blocking/streaming text is identical, frozen
rendered-text tokenization is exact, EOT markup is absent, finish metadata is
`length/32`, and tracked bytes/allocations return exactly to zero after
`LLM.close()`. The measured run used the cache-backed public path, reported
`hip_gfx1151` / `gguf_q4_k_m`, and took 54.846 s blocking including 49.773 s
model load, 4.966 s resident streaming, and 4.961 s direct eager. These timings
are correctness diagnostics, not a throughput promotion.

Implemented Poolside-v1 reasoning compatibility (2026-07-22):

- `hipengine/chat/poolside_v1.py` is a torch-free transcription of the frozen
  S 2.1 GGUF template. It reproduces all five checked-in no-thinking, thinking,
  tool-declaration, and tool-history renderings byte-for-byte; the existing
  Laguna tokenizer fixture already proves those strings lower to the frozen
  token IDs exactly;
- `PoolsideV1ReasoningParser` follows vLLM's
  [assistant-scoped backward scan](https://github.com/vllm-project/vllm/blob/61c9ef986a807aa3b9c6ccd25bb223b8f4116ac7/vllm/reasoning/poolside_v1_reasoning_parser.py#L33-L69):
  it stops at the current atomic `<assistant>` token, so a prior turn's
  `</think>` cannot make a new prompt-opened reasoning span look closed;
- the generation plugin exposes renderer/parser capabilities rather than
  adding a Laguna branch to the server. The generic server carries the prompt's
  initial reasoning state through blocking choices, live one/many streams,
  buffered scheduler chunks, visible logprobs, tool buffering, finish details,
  and token accounting;
- the splitter accepts both prompt-opened output (`reasoning</think>answer`) and
  output carrying an explicit/duplicate `<think>`, removes complete control
  markers across arbitrary chunk boundaries, and preserves Qwen's prior
  default-closed behavior when no model parser is registered;
- `tests/fixtures/laguna_poolside_v1_reasoning.json` freezes no-thinking,
  thinking, duplicate-marker, stop-at-close, and prior-turn scope cases.
  Focused server tests prove blocking and fragmented streaming emit only
  `reasoning_content`/`content`; capabilities report `poolside_v1` once the
  Laguna generator is loaded.

This closes deterministic reasoning-parser task #22, not a live Laguna chat-
quality claim. Tool parsing is closed separately below; task #24 must run real
blocking/streaming reasoning and tool transcripts before L7 is complete.

Implemented Poolside-v1 tool compatibility (2026-07-22):

- `PoolsideV1ToolParser` follows vLLM's pinned
  [`poolside_v1` envelope and typed-value extraction](https://github.com/vllm-project/vllm/blob/61c9ef986a807aa3b9c6ccd25bb223b8f4116ac7/vllm/tool_parsers/poolside_v1_tool_parser.py#L48-L220):
  complete `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value>`
  blocks parse with or without a newline after the function name;
- schema-declared string values preserve leading indentation, interior Unicode,
  and trailing newlines verbatim. Other values use JSON, then safe Python
  literal decoding, then a string fallback, before arguments are serialized as
  valid JSON;
- the parser returns all complete calls and the content before the first call,
  matching the pinned vLLM non-streaming contract. hipEngine deliberately adds
  malformed/incomplete-block reporting so its existing strict validation can
  suppress unsafe output with `invalid_tool_call` rather than leaking XML;
- the generation plugin owns the parser and advertises `poolside_v1_xml`; the
  generic server selects it through the model protocol without a Laguna/backend
  branch. Models without a parser retain the Qwen JSON parser;
- blocking choices emit OpenAI function calls and `finish_reason=tool_calls`.
  Strict streaming buffers until result validation, then emits 128-character
  JSON argument deltas with one stable ID per call; fragmented source XML and
  long escaped string values reconstruct exactly. Multiple calls require the
  existing `parallel_tool_calls=true` opt-in and receive distinct IDs;
- `tests/fixtures/laguna_poolside_v1_tools.json` freezes single, newline-less,
  typed/verbatim, adjacent-multiple, ordinary, partial, malformed, and empty-name
  outputs. Focused server tests cover blocking, fragmented SSE, stable IDs,
  parallel opt-in, capability truthfulness, and fail-closed behavior.

This closes deterministic tool-parser task #23. Live transcript behavior is
closed separately below.

Live Poolside-v1 parser/API gate (2026-07-22):

```bash
HIPENGINE_HIP_ARCH=gfx1151 GPU_MAX_HW_QUEUES=1 \
uv run python -u scripts/laguna_poolside_v1_e2e.py \
  /home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf \
  --backend hip_gfx1151 \
  --model-sha256 7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f \
  --output benchmarks/results/2026-07-22-gfx1151-laguna-poolside-v1-e2e-correctness.json
```

The source-`9805df7f7` run completed in 159.651 s with `pass=true` and
`performance_claim=false`. All five live cases produced identical blocking and
streaming generated IDs and normalized messages:

- thinking-disabled `Reply with exactly: OK` emitted `[5887,24]`, visible `OK`,
  and EOT stop without leaking `</assistant>`;
- thinking-enabled multiplication emitted 64 non-empty prompt-open reasoning
  tokens and correctly finished at the declared length with no answer leakage;
- mixed output emitted visible `Checking.` plus one `get_weather` call with
  `{"city":"Paris","days":2}`;
- parallel output emitted two distinct calls for Paris/2 and Tokyo/3 in order;
- `write_file` preserved `café 東京`, quotes, a backslash, and an interior newline
  exactly through XML extraction, JSON serialization, and SSE reconstruction.

Every streamed call kept one stable ID and every public response omitted all
reasoning/tool/EOT control markup. The same run replayed all seven deterministic
ordinary/newline-less/typed/partial/malformed/empty-name tool fixtures; unsafe
cases failed closed. Capabilities reported `poolside_v1` reasoning and
`poolside_v1_xml` tools. Closing the shared model plus ten isolated sessions
returned tracked ownership from a 77,022,439,484-byte peak exactly to zero.
These timings are lifecycle diagnostics, not AR throughput measurements.

Focused validation then passed 29 Laguna parser/generator tests, 174 server
reasoning/tool/capability tests, and 13 agentic conformance tests. This closes
L7 parser/transcript task #24 for the declared c=1 greedy server surface; it is
not a broad chat-quality or benchmark claim.

### L8 — Bulk prefill and graph replay

Bulk prefill comes after eager parity:

- F16 Q/K/V/gate/o rows>1 projections;
- full-attention prefill at 48 Q heads;
- SWA prefill at 72 Q heads with bounded history;
- batched sigmoid/correction/top-10 routing;
- selected routed experts and shared expert over prompt rows;
- chunked prefill sized from gfx1151 profiles, initially testing 64 rows before
  any larger-chunk promotion;
- exact final prompt hidden/KV comparison against token-serial eager.

The correctness-first rows path is implemented at a bounded default chunk size
of 64. Normal prompts longer than one token now execute row-major embedding,
F32-weight norms, source-F16 projections, dual RoPE, global/SWA causal
attention, dense/sigmoid MoE, and final hidden state. Only the final prompt row
runs LM-head/argmax; c=1 decode is unchanged. The retained serial prefill
selector is documented in `docs/REFACTOR.md` and remains the independent oracle.

`verify_rows(root, drafts)` remains the committed B+1 diagnostic: it returns
full FP32 logits for every target row and copies BF16 DFlash taps at depths
2/11/20/30/39/48 into caller-owned row buffers. D3's
`verify_dflash_chain(root, drafts)` is the production transaction: it stages all
per-layer K/V, runs GPU accept, and appends only the accepted prefix, so rejected
target rows are now safe across global and SWA ownership.

The accepted gfx1151 artifact
`benchmarks/results/2026-07-22-gfx1151-laguna-bulk-prefill-verifier-correctness.json`
compares lengths 1/2/7/55/65, including the 64-row chunk boundary. Every case is
bit-exact to token-serial execution for complete FP32 logits, final and
pre-final BF16 hidden, all six taps, next token, and a SHA-256 over complete
live-span metadata plus every live BF16 K/V row. The five-row B+1 gate after a
seven-token prefix is exact on the same surfaces. Tracked allocations return to
zero. Timings are diagnostics only: length-55 is `3.1030 -> 2.3552 s` (1.317x),
length-65 is `3.6872 -> 2.8019 s` (1.316x), and B+1 is
`0.2776 -> 0.2286 s` (1.214x).

A cached-build full-model `rocprofv3` gate records the intended rows families:
36 global and 108 SWA prefill-attention/write launches over three 55-row passes,
141 row routers/reducers, 144 source-F16 triple/single projections, rows-form
Q4 pack8 shared projections, and selected Q4/Q6 T16 experts. It also identifies
the next optimization target for L10/task #35: source-F16 triple and BF16-output
single rows dominate the trace at 2.734/2.126 s total, while global/SWA attention
uses only 0.007/0.060 s total. This dispatch gate predates and is separate from
the retained target-AR timing packet below; graph replay remains unclaimed.

Graph replay comes last. The static capture key plus the per-replay state ticket
must jointly cover every semantic axis: context capacity, layer attention
pattern, live-span policy, absolute position, KV addresses/capacities, scratch
addresses, active width, and sampler mode. Existing Qwen/PARO graph admission
does not automatically certify Laguna. The exact c=1 ABI, lifecycle, RED gates,
and bounded performance model are frozen in the D8 design below; graph replay
remains unclaimed until those gates pass.

Acceptance:

- bulk versus serial final hidden, all 12 global KV families, and all 36 SWA
  ring states pass the correctness gate;
- 512/1K/4K prefill is exact before performance tuning;
- graph/eager generated IDs and every hidden/KV transition match over the full
  measured horizon;
- graph capture/instantiate time is reported separately and excluded only from
  the steady-state replay number; cold capture remains included in request/E2E
  promotion accounting; and
- retain graph only if same-device end-to-end wall improves without regression.

### L9 — Context and concurrency progression

Run contexts in order:

```text
4K -> 32K -> 64K -> 128K -> 256K
```

At each context record:

- exact command, model hash, commit, ROCm stack, and gfx1151 settings;
- prefill/decode/wall samples;
- tracked allocator peak and whole-device GTT peak, explicitly labeled;
- global and SWA physical/logical KV capacities;
- logits/IDs and correctness gate;
- load, prefill, decode, close, and repeated-session lifecycle;
- profiler kernel-family breakdown where the regime changes.

Do not publish a context row from one successful pass. The existing gfx1151
GGUF 128K lifecycle history shows that later measured passes can fail after a
successful warmup/first pass. Repeated load/run/close and same-process reuse are
part of the gate.

Concurrency begins only after c=1 is accepted. Admission must account for
shared weights plus per-request global/SWA KV. Validate c=2, then c=4, with
independent-c1 IDs, delayed admission, cancellation, and final ownership. Do not
assume 256K c=4 fits merely because 256K c=1 fits.

### L10 — Performance evidence and default promotion

For any retained speed path:

1. Define the exact Poolside/llama.cpp HIP baseline on the same gfx1151 host.
2. Use the same GGUF hash, context, prompt set, output horizon, and KV dtype
   where the runtimes permit it; disclose unavoidable differences.
3. Run warmups and repeated measurements per [`BENCHMARK.md`](BENCHMARK.md).
4. Run correctness before accepting the performance result.
5. Capture `rocprofv3` kernel-family evidence.
6. Promote exact, non-regressive wins to the Laguna gfx1151 registry defaults.
7. Add rollback flags only when bisection value justifies them and record a
   removal trigger in [`REFACTOR.md`](REFACTOR.md).
8. Update `WORKLOG.md`, `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and a
   compact `benchmarks/results/*.json` artifact.

The first retained c=1/4K target packet closes Task #35 at hipEngine revision
`ee1649e3f`. It runs all ten canonical prompts (68-122 tokens) across four
categories, greedy horizons 16/32, two repetitions, and balanced serial/bulk
order. The 64-row default is exact against serial and repeat-deterministic on
every prompt/horizon. It improves weighted prefill `17.418 -> 23.333 tok/s`
(+33.95%), median TTFT `4.628 -> 3.481 s` (-24.79%), and E2E h16/h32
`2.727/4.670 -> 3.470/5.719 tok/s` (+27.27%/+22.47%); the unchanged eager c=1
decode remains neutral at 16.381 tok/s. Every category passes its predeclared
non-regression gate, KL is `6.6214e-6` with exact Poolside first token, and all
tracked allocations recover.

A separate exact Greedy-4 trace records 12,789 launches / 33 families over
three bulk prefills plus nine decode rows. Source-F16 QKV/O families account for
2.877/2.244 s, selected Q4 dual/down for 1.280/0.717 s, while global/SWA
prefill attention totals 0.007/0.060 s and decode attention 0.003/0.029 s.
F16 projections and selected experts—not attention metadata—remain the next
short-context optimization targets.

The clean matched Poolside llama.cpp `04b2b72c` raw-token baseline is retained
only as a qualified external control: native prompt is 70.45 tok/s and native
predicted output is 19.06/18.88 tok/s at h16/h32. No cross-engine ratio is
claimed because its `predicted_ms` owns all generated tokens and HTTP wall,
whereas hipEngine decode owns `horizon-1` post-TTFT forwards in process.
Poolside same-server output is also not repeat-stable on one mixed prompt, so
the fresh-process frozen distribution remains the independent correctness
oracle. Evidence:
`benchmarks/results/2026-07-22-gfx1151-laguna-s21-target-ar-retained.json` and
`benchmarks/results/2026-07-22-gfx1151-poolside-laguna-s21-target-ar-baseline.json`.

#### L10 prefill bottleneck review and transfer plan (2026-07-23)

The first bulk promotion proves exact row execution, not an optimized prefill
architecture. Segmenting the retained Greedy-4 trace at its embedding launches
isolates three complete 55-row prefills from the nine c=1 decode rows. Their
kernel spans are 2.345/2.346/2.349 s. The median prefill breaks down as follows;
percentages are of the 2.346 s kernel span, not the unprofiled category wall:

| 55-row prefill family | Calls | Median GPU time | Share |
| --- | ---: | ---: | ---: |
| source-F16 Q/K/V triple projection | 48 | 907.4 ms | 38.7% |
| source-F16 attention output projection | 48 | 706.9 ms | 30.1% |
| selected Q4T16 dual gate/up GEMV | 47 | 400.9 ms | 17.1% |
| selected Q4T16/Q6T16 down GEMV | 47 | 223.3 ms | 9.5% |
| global plus SWA prefill attention | 12 + 36 | 22.4 ms | 0.95% |
| all remaining kernels and queue gaps | — | about 85 ms | about 3.6% |

This resolves the earlier questions. Source-F16 QKV/O owns **68.8%** of the
short-prompt kernel span and direct selected-expert GEMV owns another **26.6%**;
attention is below 1%. Eliminating all source-F16 time would have only a
steering/Amdahl ceiling of about 3.2x, while eliminating all attention time
would be below 1.01x. These are bounds, not predicted gains. The qualified
Poolside prompt rate of 70.45 tok/s remains directionally consistent with large
headroom but is still not an apples-to-apples speed ratio.

The source explains the profile. `laguna_f16w_{,triple_}gemv_kernel` launches
one 256-thread block per `(row, output column)`. Every prompt row independently
streams the same F16 weight row, uses no WMMA, and gets the library's `decode`
build profile. The triple QKV wrapper reduces launch fanout but does not create
a matrix tile or reuse weights across rows. `run_laguna_moe_rows(...)` likewise
keeps `rows * top_k` lanes in row-major order and invokes the direct T16 decode
GEMVs; it does not group lanes by expert or call the compact selected-MoE WMMA
families. Dense layer 0 and every shared expert also explicitly disable GGUF
WMMA prefill. The path is therefore row-parallel, but its dominant math remains
decode-shaped.

The transferable Qwen/PARO lessons are narrower than "turn on bulk":

- use a true tiled GEMM/WMMA for rows>1 and keep GEMV for rows=1;
- group routed rows by expert before selected WMMA, but measure real expert
  occupancy and padding rather than assuming a 16-row tile wins;
- share activation tiles across independent output waves where arithmetic order
  permits it (`GPF-3A`, `GPF-5A`, and `LCP-3` precedents);
- remove prompt-sized scratch, spills, and synchronous metadata readbacks only
  after a profile names them (`GPF-2E`, Conv no-scratch, LCP-5A, and GPF-9C);
- choose chunk and kernel policies per architecture and shape, with an exact
  fallback, instead of copying gfx1100/gfx1151 thresholds blindly; and
- do not start with AOTriton, graph replay, or host launch fusion when 95.4% of
  the measured short-prompt span is two projection/expert families.

Laguna prefill work proceeds in this order:

1. **LPF-0 — dedicated profile and replay fixtures.** Add a cached-build,
   prefill-only family trace for rows 16/32/55/64/122/128, plus a real-routing
   replay artifact that records per-expert counts for top-10 lanes. Preserve the
   current trace as the baseline; do not mix candidate selection with nine
   decode rows again.
2. **LPF-1 — source-F16 true bulk projection.** Register separate rows>1
   `fp16_weight` variants for mixed BF16 activation/F16 weight with FP32
   accumulation and FP32 or BF16 output. Start with a 16x16 tiled WMMA or a
   torch-free rocBLAS/hipBLASLt control, compile the retained candidate with the
   `prefill` profile, and route Q/K/V and O through it above a measured row
   threshold. It is acceptable for three real GEMMs to replace the current
   triple GEMV initially; fuse/group QKV only if a later trace shows launch or
   activation rereads matter. Rows=1 must remain on the current exact GEMV.
3. **LPF-2 — selected-expert weight reuse.** Feed Laguna's unchanged sigmoid,
   correction-bias, top-10 IDs, and weights into the model-neutral Qwen group
   count/prefix/scatter/tile-map ABI. Replay the existing Q4T16 dual and
   Q4T16/Q6T16 down compact WMMA kernels against the direct-GEMV control. At 55
   rows there are only 550 lanes across 256 experts, so 16-row padding may erase
   the gain. If so, implement a measured small-M expert-grouped rowtile/pair-
   reuse path for the observed 2-8-row experts rather than forcing compact
   WMMA. Preserve Laguna's normalized uncorrected weights and 2.5 routed scale.
4. **LPF-3 — dense and shared experts.** Pair dense/shared gate+up where the
   existing GGUF ABI permits it, then select real raw/T16 WMMA prefill for Q4
   and Q6 rows instead of merely setting a flag on a pack8 GEMV layout. Profile
   this only after LPF-1/2; the current combined dense/shared family is about
   71 ms of the 55-row span and is not the first bottleneck.
5. **LPF-4 — chunk policy.** After weight-reusing kernels land, compare 64 with
   128 first: all canonical 68-122-token prompts then fit in one chunk instead
   of two complete 48-layer passes. Continue to 256/512 only with bounded
   scratch, exact 511/512/513 SWA transitions, and measured wins. The PARO
   256-row choice is precedent, not a Laguna default.
6. **LPF-5 — long-context attention.** Reprofile at 512/1K/4K before changing
   attention. The current global kernel has capacity-sized score scratch and a
   serial V loop, and the SWA kernel repeatedly reduces over its 512-token
   window, so a Flash/AOTriton-style route may become necessary later. Any such
   route must preserve complete `KVLiveSpans`, global/SWA visibility, BF16 K/V
   rounding, and the separate softplus gate. It is explicitly not LPF-1.
7. **LPF-6 — submission, graph, and packed serving.** Reuse chunk metadata once,
   remove proven D2H/synchronization boundaries, and consider graph capture or
   multi-request packed prefill only after the dominant kernels are no longer
   decode-shaped. Variable prompt/chunk state remains part of every graph key.

LPF-0 is closed on gfx1151. The dedicated prefill-only harness executes one
physical chunk at rows `16/32/55/64/122/128`, with three timed passes in rotating
order and no decode rows. Median rates are
`23.141/23.421/23.450/23.453/23.368/23.377 tok/s`; all repeated next tokens and
the separate routing replay agree, and tracked ownership returns exactly to its
baseline. A cached-build trace contains exactly 12 complete passes and 1,006
embedding-to-argmax dispatches per pass. At 55 rows, source-F16 QKV/O consumes
`907.232 + 706.832 ms` (**68.99%**) of the `2.340 s` median kernel sum,
selected Q4/Q6 direct GEMV consumes `618.885 ms` (**26.45%**), and attention is
`22.365 ms` (**0.96%**), confirming LPF-1 was the correct first candidate.

The replay also makes LPF-2's padding risk concrete. At 55 rows, 25,850 top-10
lanes occupy 6,892 `(layer, expert)` groups; **76.25% of groups contain at most
four rows**, and padding every nonempty group to 16 rows would execute **4.396x**
the useful lanes. At 128 rows that ratio is still **2.704x**. The existing
compact-WMMA route therefore remains a control to measure, not a presumed win;
a small-M grouped rowtile/pair-reuse route is the stronger candidate if compact
replay loses. Evidence:
`benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf0-profile.json`.

LPF-1 is closed and promoted on gfx1151. A reassociated 16x16 F16-WMMA control
reached 60.65 tok/s but changed three of ten free-running trajectories and was
removed. The retained 8x4/16x4 row/column tile instead preserves the original
thread-local K order and wave/block reduction tree while reusing activations
across four output columns and weights across rows. Synthetic F32/BF16 outputs
are bit-exact to GEMV, and cached profiling names
`laguna_f16w_tiled_exact_kernel<unsigned short, 16>` at **3.798 ms** for the
55x9216x3072 O projection (256 threads, 96 VGPR, 128 SGPR, 512 B LDS, zero
scratch).

The clean same-session A/B alternates GEMV/tiled order over three passes at rows
2/3/4/5/7/8/15/16/17/32/55/64/65/122/128. All 90 outputs agree. Every shape
wins: two rows move **20.568 -> 21.327 tok/s (1.0369x)**, 55 rows
**23.460 -> 48.760 (2.0784x)**, and 128 rows **23.374 -> 50.240 (2.1494x)**;
the weighted profile is **2.0538x**. The measured gfx1151 threshold is therefore
two rows, while rows=1 and unsupported backends retain the registered GEMV.

The clean two-repeat canonical category gate then moves the previous bulk-GEMV
prefill **23.333 -> 48.560 tok/s (+108.12%; 2.081x)**, median TTFT
**3.481 -> 1.692 s (-51.39%)**, and h16/h32 E2E
**3.470/5.719 -> 5.955/8.717 tok/s (+71.61%/+52.42%)**. Decode is neutral at
16.386 tok/s. All 20 serial/tiled pairs and same-route repeats have complete-ID
equality at h16/h32; all four categories pass, the Poolside first-token gate
remains KL `6.6214e-6` with exact top-1, lifecycle recovery is exact, and both
artifacts have clean provenance. The gfx1151 backend capability now makes tiled
the default from row two; `HIPENGINE_LAGUNA_F16_PREFILL=gemv` remains a one-
release rollback. Evidence:
`benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf1-{ab,tiled}.json`.
LPF-2 is closed as a measured rejection. The no-padding compact-pair candidate
grouped exact top-10 lanes and covered 84.19%/91.71% of them in pairs at 55/128
rows while preserving direct Q4/Q5/Q6 T16 reduction and BF16 bits. That compute-
block bound did not translate to wall time: balanced same-session full-model
prefill regressed every measured shape, from **46.261 -> 38.362 tok/s (-17.07%)**
at 16 rows to **50.187 -> 45.064 tok/s (-10.21%)** at 128 rows; 55 rows moved
**48.689 -> 42.515 tok/s (-12.68%)**, and the weighted profile was **0.8843x /
-11.57%**. All 36 direct/candidate next-token results agreed and lifecycle
recovery was exact, so this is a performance rejection rather than a correctness
failure. The Laguna selector, grouping library, compact scratch, runtime route,
and temporary A/B harness were removed; direct selected GEMV remains the only
runtime path. The exact generally useful compact-pair kernel primitives and
fixtures remain registered. A 16-row compact WMMA route is not viable here: the
replay requires 4.396x/2.704x useful-row work at 55/128 rows, reassociates the
direct reduction, and has a weaker bound than the already rejected no-padding
candidate. LPF-3 is next. Evidence:
`benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf2-compact-pair-rejected.json`.

LPF-3 is closed as a measured rejection under the current resident contract.
The existing exact Q4 pack8 dual-prefill ABI paired gate/up for dense layer 0
and all 47 shared experts, but balanced full-model timing regressed every shape:
16 rows moved **46.274 -> 45.985 tok/s (-0.63%)**, 55 rows moved **48.672 ->
48.352 (-0.66%)**, and 128 rows moved **50.377 -> 49.977 (-0.79%)**. The
weighted profile was **0.9929x / -0.71%**; all 36 next-token results agreed and
lifecycle recovery was exact. The selector, route, generic held-library change,
and temporary harness were removed. A real Q4/Q6 WMMA route is not a viable
incremental follow-up here: Laguna materializes dense/shared Q4 as pack8 and Q6
down as raw GGUF, while the in-tree Matrix-Core families require raw/T16
residency and use reassociated arithmetic. Changing all 48 layers' replacement
contract plus cache for a family that owned only about 71 ms of the pre-LPF-1
55-row span has a best-case post-LPF-1 ceiling near 6%; it is deferred until a
future resident-layout project or a new profile makes it dominant. Evidence:
`benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf3-dense-shared-rejected.json`.

LPF-4 is closed and promoted at **128 rows**. The clean same-session gate keeps
one 128-row resident allocation and alternates 64/128 scheduling over two
repetitions of all ten canonical prompts. Every prompt crosses 64 and fits 128,
so the candidate removes a second complete 48-layer pass without changing
kernel math. Prefill moves **48.541 -> 49.641 tok/s (+2.27%)**, median TTFT
**1.692 -> 1.639 s (-3.15%)**, and h16/h32 E2E **5.954/8.717 -> 6.042/8.811
(+1.49%/+1.08%)**; decode is neutral within **0.014%**. Every category improves
prefill by **1.09-2.84%** and E2E by **0.48-1.79%**. All 20 chunk pairs are
complete-ID exact at both horizons, same-route repeats are deterministic, the
Poolside gate remains KL `6.6214e-6` with exact top-1, and lifecycle recovery is
exact. The larger bounded scratch adds **49.1 MiB** to resident ownership; the
public session default is now 128 while an explicit 64-row constructor override
remains available. Evidence:
`benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf4-chunk128.json`.

LPF-5 profiling now establishes that attention is the first viable remaining
prefill target. One clean cached-only `rocprofv3` pass with the retained 128-row
chunks reaches **43.732/39.697/33.745 tok/s** at 512/1K/4K. Attention grows from
**1.896 s / 16.25%** of kernel sum at 512 to **6.115 s / 23.78%** at 1K and
**42.609 s / 35.19%** at 4K. At 4K, 12 global layers consume **16.908 s
(13.96%)** and 36 SWA layers consume **25.701 s (21.23%)** even though the SWA
window is bounded at 512. The current 128-thread SWA body therefore has the
strongest exact incremental target: it performs two serial token scans and a
seven-barrier block reduction for every score. Global attention becomes a
separate second target only after SWA is resolved. Final cursors, next IDs,
511/512/513 boundary fixtures, tracked lifecycle, and trace segmentation pass;
the raw trace has SHA-256 `7ca2217d...313ea5`. This is a one-pass attribution
baseline, not a speedup or supported long-context throughput claim. Evidence:
`benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf5-long-context-profile.json`.

The first LPF-5 candidate is closed and promoted on gfx1151. The wave32-exact
SWA body maps four dimensions per lane and reconstructs the baseline 128-thread
stride-64/32/16..1 dot-product tree before preserving the same sequential max,
denominator, and value accumulation. It consumes complete ring `KVLiveSpans`
and removes all per-token block barriers without changing arithmetic. The
508..515 wrap/eviction fixture is F32 byte-exact to both baseline bulk and
scalar attention. A ten-pair leaf screen moves **20.434 -> 9.229 ms (2.214x)**;
cached tracing names the candidate at **9.123 ms**, 32 threads, 32 VGPR, zero
LDS/scratch versus baseline **20.355 ms**, 128 threads, 16 VGPR, 1,024 B LDS,
zero scratch.

The clean shared-weight full-model gate then moves 512/1K/4K prefill
**43.760/39.748/33.800 -> 47.395/44.855/38.552 tok/s
(+8.31%/+12.85%/+14.06%)**, saving **0.898/2.933/14.939 s**. Every complete
100,352-way FP32 logit vector, final/pre-final BF16 hidden vector, next-logit
bit pattern, token ID, and cursor is exact; tracked ownership returns to zero.
A prior complete timing pass independently reproduced **1.082/1.128/1.140x**
before a post-timing artifact bug, so the result is not a single observation.
The gfx1151 backend capability now selects wave32 exact automatically; explicit
baseline selection remains a one-release rollback, and unmeasured backends keep
the 128-thread route. Evidence:
`benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf5-swa-wave32.json`.

LPF-5 global attention is closed for the current 4K scope after an exact
incremental rejection. A paired-query candidate kept each head's baseline
reduction order while sharing K/V loads across adjacent heads in the six-head
GQA group. The 3,968-prior + 128-current / 4K-capacity production leaf was
byte-exact, but moved **86.429 -> 125.319 ms (0.6897x; +45.00%)** because
halving per-head token parallelism outweighed K/V reuse. The kernel, export,
wrapper, registry key, selector, and fixture extension were removed. A
Flash/AOTriton route would reassociate softmax/value arithmetic and add a new
cache adapter; defer it until admitted contexts beyond 4K or a new profile makes
global attention dominant enough to justify that correctness surface. Evidence:
`benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf5-global-pair-rejected.json`.

LPF-6 is also closed as a profile-based defer for c=1. In the clean 4K trace,
kernel span exceeds kernel sum by only **0.302 s / 0.25%** across 35,233
dispatches; the 128-row LPF-0 trace similarly has about **5.6 ms / 0.10%**
residual per complete pass. Graph capture, host submission fusion, or metadata
reuse therefore has a sub-percent ceiling after LPF-1/4/5, while packed
multi-request prefill changes the c=1 product scope and belongs to a separate
serving milestone. No LPF-6 runtime path or temporary flag is retained.

Every LPF candidate is a registered variant with the current exact chain as its
unfused rollback. Exact candidates pass byte comparison on lengths
1/2/7/55/65 plus the affected tile/chunk boundaries. Reassociated GEMM/WMMA
candidates additionally pass their production-shape CPU oracle, the frozen
Poolside first-token gate (KL <= 0.05 and exact top-1), and the full ten-prompt
four-category teacher-forced and free-running h16/h32 suite with deterministic
outputs; report complete-ID equality even when the standard KL/top-1 gate is
the admission rule. Performance admission uses balanced same-session
current/candidate order, requires every category to remain non-regressive,
keeps decode within 2%, excludes model load, and records the intended kernel
name, duration, workgroup, VGPR/SGPR, LDS, and scratch from a prebuilt cached
`rocprofv3` run. A kernel-only micro win does not promote the default.

## Laguna AR Optimization Campaign — 4-10x Prefill

This is the active Laguna performance campaign after LPF-0 through LPF-6. The
previous work established a correct, resident, chunked baseline and found useful
exact improvements; it did **not** establish a competitive matrix prefill
architecture. DFlash is frozen as an explicit correctness-supported provider and
is not an optimization target. AR prefill is the only headline metric in this
campaign. DFlash should be rerun only if shared target-state or public-provider
behavior changes.

### Why the 4-10x target is credible but not yet a claim

The current merged-main canonical result is **50.389 prefill tok/s** and
**16.384 decode tok/s** on the Radeon 8060S/gfx1151. The retained long-context
AR results are **47.395/44.855/38.552 tok/s** at 512/1K/4K. Model loading is
excluded from all of these values.

The following llama.cpp numbers were reported during this planning session and
are directional controls, not retained hipEngine evidence. Their exact prompt,
batch/chunk, build, power, and timing scopes still need to be captured:

| Directional control | Approx. active parameters | Prefill | Decode | Evidence status |
| --- | ---: | ---: | ---: | --- |
| gpt-oss 120B MXFP4 | about 6B | 720 tok/s | 56 tok/s | user-reported; protocol capture pending |
| Nemotron Super 3 120B-A12B | about 12B | 276 tok/s | 14.86 tok/s | user-reported; protocol/quant capture pending |
| Laguna S 2.1 Q4_K_M in hipEngine | about 7.83B prefill linear work | 50.389 tok/s | 16.384 tok/s | retained current-main AR evidence |

Laguna's active linear work can be estimated directly from the published
shapes. Per prefill token it evaluates approximately 2.803B attention-projection
parameters, 4.435B selected routed-expert parameters, 0.444B shared-expert
parameters, 0.113B dense-layer-0 parameters, and 0.037B router parameters: about
**7.83B active linear parameters** before norms, elementwise work, and context
attention. Decode also evaluates the 0.308B-parameter LM head. This is not an
“8B model” capacity statement; all roughly 70 GiB of quantized weights remain
resident.

At 50.389 tok/s, the prefill linear work rate is only about **0.79 TFLOP/s** if
a multiply-add is counted as two operations. A 200-500 tok/s Laguna prefill
would be about **3.1-7.8 TFLOP/s** before attention, comparable in scale to the
rough active-work rates implied by the two directional controls. Differences in
quant format, active-parameter definitions, architecture, and benchmark scope
prevent a direct ratio, but they do show that 50 tok/s is not a plausible
compute roofline.

Decode tells a different story. The existing model estimate is 9-10 GB of
active weight traffic per generated token. At 16.384 tok/s that implies roughly
147-164 GB/s, already 67-74% of the local 221 GB/s practical read ceiling before
other traffic. Decode can still improve, but its result is consistent with
Laguna's mixed F16/Q4/Q6 active bytes and is not the current priority. Prefill
should reuse weights across rows and move onto matrix instructions instead of
paying decode-shaped costs per row.

The earlier same-model Poolside llama.cpp raw-token control was 70.45 prompt
tok/s over the short canonical suite. AR-O0 now adds a matched long-shape
control with the identical Q4_K_M hash, deterministic token stream, BF16 KV,
and 128-row microbatch. Poolside native `prompt_ms` measures
**80.235/103.868/105.435/120.530 tok/s** at 128/512/1K/4K. At the three shapes
with balanced hipEngine timing, that is a diagnostic **2.189/2.351/3.127x**.
The ratio remains qualified because Poolside excludes HTTP/sampling and
hipEngine includes final argmax bookkeeping; Poolside also rounds the requested
4,097-slot endpoint context to 4,352. Nevertheless, the same-model control
proves that the current 38-47 tok/s long-prefill path is not a model-imposed
ceiling. Poolside itself is still far below the cross-model gpt-oss directional
number, so the 200-500 tok/s target remains an optimization objective rather
than a comparator-derived promise.

### What the retained profiles actually say

AR-O0 now replaces the inferred post-LPF split with a clean cached all-family
trace at current runtime revision `7ded0d5f`. The profiler covers a 128-row
warmup plus 512/1K/4K passes, assigns every dispatch to a stable family, and
leaves less than 0.001% in `other`. A separate alternating three-repetition run
measures **47.453/44.848/38.541 tok/s** at 512/1K/4K; every next ID and final
cursor repeats exactly and tracked ownership returns to zero.

| Current family share | 128 rows | 512 | 1K | 4K |
| --- | ---: | ---: | ---: | ---: |
| selected Q4 gate/up | 36.60% | 34.25% | 32.47% | 27.95% |
| selected Q4/Q6 down | 20.17% | 18.98% | 18.01% | 15.50% |
| **all selected experts** | **56.78%** | **53.23%** | **50.48%** | **43.45%** |
| source-F16 projections | 33.40% | 30.82% | 29.12% | 25.02% |
| dense/shared quant projections | 6.55% | 6.06% | 5.71% | 4.94% |
| global + SWA attention | 2.46% | 9.17% | 14.01% | 26.01% |
| all remaining kernels | 0.81% | 0.71% | 0.68% | 0.58% |

This confirms AR-O1 before AR-O2. Selected experts are the majority of kernel
sum through 1K, and Q4 gate/up alone is larger than the complete source-F16
family at every measured shape. Perfectly eliminating selected experts has only
a 2.31x 4K ceiling and perfectly eliminating source-F16 only 1.33x; neither can
deliver the campaign target alone. At 128 rows, making both families 10x faster
would leave 18.84% of current time and yield about 5.3x, after which dense/shared
projections become the next linear residual.

The current wave32 SWA path reduces total attention from the pre-promotion
16.25/23.78/35.19% to **9.17/14.01/26.01%** at 512/1K/4K. At 4K, global/SWA
are now **15.92/10.09%**, so global—not SWA—is the later context target once
matrix work lands. Kernel span exceeds kernel sum by only
**0.28-0.34%** across all four shapes, keeping graph capture and host launch
work last. Evidence:
`benchmarks/results/2026-07-23-gfx1151-laguna-prefill-current-main-all-family-profile.json`.

### Target ladder

Targets are defined against the retained current-main AR route, not against
DFlash and not against a single repeated-token prompt:

| Milestone | Canonical short prefill | 512 / 1K / 4K intent | Meaning |
| --- | ---: | --- | --- |
| O1 | >=100 tok/s | report all three shapes | matrix substrate is working; not campaign success |
| O2 | >=200 tok/s | seek >=190/180/154 tok/s | minimum 4x short/long campaign goal |
| O3 | >=300 tok/s | no long-context regression | comparator-class checkpoint, still protocol-qualified |
| Stretch | 400-500 tok/s | continue context-specific scaling | 8-10x short-prefill objective |

These are outcome gates, not promises. Every retained sub-window or exact
end-to-end gain is still promoted under the repository performance policy even
if it does not cross the next ladder rung. Long-context ratios are reported
separately because causal attention work grows with context while projection
work does not.

### Optimization sequence and task list

Dependencies are intentional. Reprofile after every promoted phase; do not keep
implementing against the pre-LPF attribution.

#### AR-O0 — homologate controls and capture the current bottleneck

- [x] Run cached-build, prefill-only timing at rows/lengths 128, 512, 1K, and 4K
  on the current merged revision. Use at least three balanced timing samples for
  candidate admission; a single profiler pass is sufficient for attribution.
- [x] Extend the trace summary to account for **all** kernel families, not only
  global/SWA attention. Record kernel sum/span, calls, median/total duration,
  VGPR/SGPR, LDS, scratch, and row/chunk shape for selected Q4/Q6, source-F16,
  dense/shared, router, attention, norms, and metadata kernels.
- [x] Preserve the 128-row real-routing histogram and add 256/512-row replays.
  Natural M2/M4/M8/M16/M32 padding factors are
  **1.043/1.134/1.334/1.803/2.924x** at 256 rows and
  **1.022/1.068/1.165/1.379/1.867x** at 512. The deterministic Zipf control is
  **1.050/1.157/1.420/2.049/3.421x** and
  **1.025/1.075/1.182/1.465/2.134x**; the top-10 hot control is exactly 1.0x
  because 256/512 divide every tested tile. All natural useful-row counts are
  retained per `(layer, expert)` and every lane/lifecycle gate passes. This
  admits an M16 crossover screen at 256/512 but not blanket M32 on natural/Zipf
  routing. Evidence:
  `benchmarks/results/2026-07-23-gfx1151-laguna-routing-256-512.json`.
- [x] Capture matched llama.cpp Laguna Q4_K_M 128/512/1K/4K controls with the
  same token streams and native prompt timing. Poolside reaches
  80.235/103.868/105.435/120.530 tok/s; model/token/KV/microbatch hashes match,
  and build, clocks, memory, and timing qualifications are retained in
  `benchmarks/results/2026-07-23-gfx1151-poolside-laguna-prefill-matched-control.json`.
  Reproducible metadata for the gpt-oss and Nemotron directional rows remains
  unavailable, so those rows stay explicitly directional.
- [x] Audit the merged kernel catalog and run `scripts/check_lineage.py` before
  new kernel work. In particular inspect the existing exact Q4 dual-SiLU,
  Q4/Q5 T16 Q8_1/dp4a, grouped compact-MoE, IQ MMQ32, Qwen GGUF Q8T16/MMQ,
  F32-router, device-metadata, and AOTriton paths before writing duplicates.

##### Merged-main transfer audit (2026-07-23)

The audit imported each lazy candidate family, refreshed the gfx1151 backend
aliases, and resolved all 17 inspected four-axis keys. Fifteen focused
registry/build/argument-policy tests pass. The result is not “write a new MoE
stack”: most leaf kernels already exist, but the complete device-resident
Laguna scheduler and a useful small-M selected layout do not.

| Existing family | Laguna compatibility | Decision |
| --- | --- | --- |
| direct Q4T16 dual + fused SiLU | Exact resident `tiles`, selected-ID, K3072/N1024, and BF16-output ABI match. The fused output is bit-identical to dual projection followed by the registered separate SiLU chain. | **Screened and rejected as a runtime default.** Same-session full-model timing is exact but only +0.129% in aggregate and regresses rows 16/64. Candidate runtime wiring was removed; the registered leaf and its bit gate remain. |
| direct Q4T16 Q8_1/dp4a + fused SiLU | Resident T16 and selected-ID ABI match. It needs one caller-owned GGML Q8_1 buffer of `rows * (K/32) * 36` bytes. This is quality-gated, not byte-exact. | **Second screen only.** Quantize each token row once before top-10 expansion and include quantization in timing. There is no qualified Q4/Q6 single-down peer; do not build one unless gate/up wins inclusively. |
| group count/prefix/scatter-gather/tile-map + weighted-lane sum | Raw-pointer metadata and BF16 packed-row ABI are model-neutral; passing Laguna's already-scaled routing weights preserves normalized uncorrected sigmoid probabilities and the 2.5 scale. Registrations still carry Qwen/PARO names. | Reuse bodies through new generic registry aliases and Laguna-owned bounded scratch; do not call Qwen runner helpers or add model branches. |
| Q4T16 dual and Q4T16/Q6T16 compact WMMA | Existing Laguna allocations match the kernel `tiles` layouts exactly. The output can return through `sorted_lanes`, a static lane-to-token map, and weighted-lane sum. | Replay control, not the default design. M16 padding is already 4.396x at 55 rows and 2.704x at 128. gfx1151 also has no admitted no-read compact-WMMA launch policy, so the current complete Qwen orchestration may read one device scalar. |
| compact exact pair-reuse | Layout and arithmetic match. | Do not rewire: LPF-2 already rejected its stronger no-padding bound at 0.8843x weighted full-model throughput. |
| IQ2_XS MMQ32 | Exact K3072/N1024 Laguna Q2 XL gate/up shapes, raw-IQ weights, D4-Q8_1 input, and compact metadata match. It pads populated experts to M32. | Q2-XL-specific later lane. It is a scheduling reference for Q4_K_M, not a quant-format shortcut; prior synthetic evidence wins at 256/512 rows and loses at short shapes. |
| Qwen raw-Q8/Q8T16 MMQ and WMMA | Guarded correction, activation reuse, and tile schedules are useful references, but weight formats and admitted K/N shapes do not match Laguna Q4/Q6 selected experts. | Do not route or duplicate residency. Reuse only scheduler/correctness ideas in a Q4/Q6-specific kernel. |
| 256-thread BF16-hidden/F32 router | Laguna already resolves `router_logits/f32/bf16_hidden` to this exact gfx1151 wrapper. | No AR-O1 work. Revisit only if a post-matrix profile makes router material. |
| contiguous prefill metadata | The helper writes Qwen attention/GDN metadata, not Laguna's span/ring contract. Kernel-span residual is only 0.28-0.34%. | Do not port. Only the grouped-expert count/prefix/scatter metadata is relevant now. |
| AOTriton | Torch-free adapter exists, but it does not directly consume Laguna's global/SWA `KVLiveSpans`, physical ring, eviction, and separate gate ABI. | Keep as an AR-O5 global-attention ceiling after matrix work; not an AR-O1 dependency. |

The first two screens are closed: exact fused Q4 dual-SiLU failed its strict
same-session full-model screen, and inclusive direct Q8_1/dp4a failed the full
quality lane. The exact no-D2H grouped-small-M Q4/Q6 down route is now promoted
on gfx1151 after passing shape and category gates. The remaining AR-O1 order is
therefore: (1) screen M16 at 256/512 using the retained routing replay while
keeping M32 hot-only unless timing overturns its padding bound; (2) admit an
integer-MMQ/WMMA route only at a measured crossover; and (3) evaluate
exact intermediate fusion after gate/up. This deliberately excludes the
rejected compact-pair route, raw-Q4 duplication, Q8T16 substitution, router
retuning, unrelated metadata work, and attention work.

Lineage status is bounded and explicit. Poolside Laguna source/layout is clean
at `04b2b72c`; llama.cpp HIP `mmq.cuh`, `mma.cuh`, and `quantize.cu` are clean
at `1ebf790c`. The broad kernel scan is blocked by the absent read-only Atlas
checkout, and the Qwen/PARO filters are blocked by the absent
`/home/lhl/amd-gpu-tuning/nano-vllm-amd` checkout. No external source is being
copied in this phase; restore and inspect those checkouts before any future
external port.

Exit: one compact current-main artifact with a complete Amdahl table and a
ranked first candidate. If the inferred 55/35 split is wrong, reorder AR-O1 and
AR-O2 from the measured table.

#### AR-O1 — selected Q4/Q6 expert matrix engine

This is the expected first bottleneck. Laguna currently runs direct T16 decode
GEMVs over `rows * top_k`; the prior exact pair-reuse candidate lost 11.57%, and
blanket M16 compact WMMA would execute 4.396x useful lanes at 55 rows. Do not
repeat either experiment unchanged.

- [x] First screen already-landed primitives. **Exact Q4 dual+SiLU is complete
  and rejected:** one-load, same-session, counterbalanced rows
  16/32/55/64/122/128 are exact in all 36 timed IDs, but split -> fused median
  rates are `46.380->46.300`, `48.917->49.000`, `49.088->49.137`,
  `50.558->50.527`, `51.081->51.194`, and `51.412->51.549 tok/s`. Aggregate
  wall improves only 0.129%, while rows 16/64 regress 0.172%/0.060%; the strict
  all-shape gate fails, candidate runtime code is removed, and the original
  split chain stays default. Evidence:
  `benchmarks/results/2026-07-23-gfx1151-laguna-prefill-ar-o1-fused-silu-rejected.json`.
  **Direct Q8_1/dp4a is also complete and rejected.** Its inclusive screen is
  mechanically strong: quantizing each producer row once before top-10
  expansion improves every 16/32/55/64/122/128 shape by 2.51-4.17% and
  aggregate wall by 3.773%, with all 36 next IDs agreeing. The full three-repeat
  ten-prompt category run likewise improves weighted prefill **4.070%**, h16/h32
  E2E **2.650%/1.916%**, and every category's prefill/E2E while decode stays
  within 0.07%. It nevertheless fails the predeclared quality lane: 315/320
  teacher-forced top-1 comparisons agree, but maximum split-vs-Q8 KL is
  **0.17156** (>0.05) on `mixed_ja_en_review`; `mixed_ja_en_translate` reaches
  **0.11889**, and four prompts have deterministic free-running ID differences.
  The frozen Poolside first-token gate still passes at KL `1.2837e-4`, proving
  why the complete category gate was necessary. The env/session selector,
  Q8_1 scratch, production route, and dedicated harnesses are removed; the
  independently tested registered fused leaf remains. Do not develop the
  single-output Q4/Q6 down sibling from this rejected quality trade. Continue
  with device-resident grouping and a quality-preserving small-M Q4/Q6 engine.
  Evidence:
  `benchmarks/results/2026-07-23-gfx1151-laguna-prefill-ar-o1-q8-dp4a-screen.json`
  and
  `benchmarks/results/2026-07-23-gfx1151-laguna-prefill-ar-o1-q8-dp4a-category-rejected.json`.
- [x] Build one device-resident expert grouping/scatter pass with no scalar D2H
  boundary. The exact BF16 route needs no activation quantization; a deterministic
  one-pass compact-active kernel emits starts, active experts, lane order, and
  routing weights, with staged count/prefix/scatter kept as the unfused fallback.
- [x] Implement and promote adaptive small-M grouped Q4/Q6 schedules for the
  measured 1/2/4/8-row populations. C16xR4 reuses each decoded T16 tile across
  up to four packed rows and falls back to direct below 32 token rows. Clean
  rows 32/55/64/122/128 improve 2.63-6.92% and aggregate shape wall improves
  5.461%. The full three-repeat ten-prompt gate moves weighted prefill
  **50.193->53.178 tok/s (+5.948%)**, median TTFT **1.627->1.535 s
  (-5.682%)**, and h16/h32 E2E **+3.835%/+2.762%**. Every category improves;
  decode is neutral within 0.062%; all 320 teacher-forced logits are identical
  (`KL=0`, top-1 100%); all free-running pairs/repeats, the Poolside oracle,
  and lifecycle pass. gfx1151 now selects adaptive grouped down by backend
  capability; gfx1100 and rows below 32 retain direct selected GEMV. Evidence:
  `benchmarks/results/2026-07-23-gfx1151-laguna-prefill-grouped-down-ab.json`
  and
  `benchmarks/results/2026-07-23-gfx1151-laguna-prefill-grouped-down-category.json`.
- [x] Screen M16/M32 integer-MMQ/WMMA only where measured occupancy pays for
  padding. The first M16 control over resident Q4T16/Q6T16 passed the narrow
  screen but **failed the complete quality lane**. Across three repetitions of
  all ten prompts at both 256/512 rows, retained grouped-small-M -> M16 improves
  weighted prefill **52.486->57.421 tok/s (+9.404%)**, h16/h32 E2E
  **+7.994%/+6.878%**, every shape/category wall row, and neutral decode.
  However, maximum final-logit KL is **1.10017** (>0.05), suite top-1 is only
  90%, category top-1 falls to **87.5% code / 75% mixed**, and 23 unique
  shape/prompt/horizon trajectory mismatches repeat in all three runs. Exact
  grouped-small-M therefore remains the gfx1151 default. M32 uses the same
  reassociated WMMA arithmetic while natural/Zipf padding is still
  2.924/3.421x at 256 and 1.867/2.134x at 512, so no M32 category run is
  warranted. The natural M16/M32 lane is closed. After the following exact
  fusion passed, the M16 runtime selectors, route scratch, and benchmark
  harnesses were removed; the separately registered kernel leaf/oracle remains
  as diagnostic evidence. Evidence:
  `benchmarks/results/2026-07-23-gfx1151-laguna-prefill-wmma16-down-screen.json`
  and
  `benchmarks/results/2026-07-23-gfx1151-laguna-prefill-wmma16-down-category-rejected.json`.
- [x] After gate/up, evaluate fused SiLU and routing-weighted down/combine to
  remove the largest expert intermediates. The exact grouped-combine default
  preserves all ten slot-order FMAs, rounds selected output to BF16, adds the
  BF16 shared output, and rounds again while removing one launch and the
  routed-output round trip. The registered unfused grouped chain remains the
  rollback. Clean production-shape GPU span improves **1.249-1.313x** at every
  32-128-row shape (**1.265x** aggregate); five-repeat complete-model wall is
  non-regressive at **0.99972x** with all 60 IDs exact.

  The clean three-repeat category gate is also exact and non-regressive:
  aggregate prefill **53.1880->53.1840 tok/s (0.999924x)**, h16/h32 E2E
  **0.999769/0.999960x**, and per-category prefill **0.998323-1.001962x**.
  All 320 teacher-forced logits are identical (`KL=0`, top-1 100%), all 30
  h16/h32 pairs and repeats are exact, Poolside KL/top-1 is
  `6.6214e-6/1.0`, and tracked ownership returns to zero. gfx1151 therefore
  defaults to adaptive fused combine from 32 rows; gfx1100/short rows retain
  direct and explicit unfused grouped selection remains the rollback. This is
  a launch/traffic promotion, not a standalone model-wall headline. Evidence:
  `benchmarks/results/2026-07-23-gfx1151-laguna-prefill-grouped-combine-screen.json`
  and
  `benchmarks/results/2026-07-23-gfx1151-laguna-prefill-grouped-combine-retained.json`.
  Do not introduce order-dependent atomics across ten experts.

Stop rule: remove a candidate that is slower inclusively at every natural
shape, or whose best applicable family speedup is below 2x with no material
scratch/dispatch benefit. Reprofile the full model after each retained leaf.

#### AR-O2 — true source-F16 matrix projection

The current exact 8x4/16x4 tile preserves GEMV's reduction order. It reuses some
loads but is not a matrix-core GEMM. The removed 16x16 WMMA control reached only
60.65 tok/s and changed three trajectories; that rejects that implementation,
not matrix prefill as a class.

- [x] Establish a torch-free rocBLAS/hipBLASLt FP16-input/F16-weight/FP32-
  accumulate ceiling at Laguna's real M/K/N shapes before tuning a custom body.
  Clean gfx1151 M16/32/64/128/256/512 timing screens all seven returned
  hipBLASLt algorithms per shape, then counterbalances exact, rocBLAS, and
  selected hipBLASLt full/SWA family sequences. At M128, conservative inclusive
  hipBLASLt is **11.497x/14.431x** faster than the retained exact full/SWA
  projection families; the synthetic 12-full/36-SWA sum is **827.901 -> 60.129
  ms (13.769x)**. The inclusive control pays BF16->F32->FP16 before QKV/gate
  and O plus the O FP32->BF16 boundary; every selected algorithm uses zero
  workspace. The nonzero math smoke and lifecycle pass. This is a library
  ceiling, not a runtime/model-throughput claim. Evidence:
  `benchmarks/results/2026-07-23-gfx1151-laguna-f16-library-ceiling.json`.
  BF16 hidden values may be converted once to FP16 only through the quality
  lane and only if range/finite checks pass.
- [ ] Develop a tiled matrix-core path for Q/K/V/gate and O at M=16..512.
  Compare separate GEMMs with a resident composite QKV+gate layout; packing
  should happen once at materialization/cache build, never during inference.
  The first explicit leaf is registered: direct row-major source-F16 reads,
  BF16->F16 register conversion, F16 16x16x16 WMMA, FP32 accumulation, and
  FP32/BF16 output with no sidecar. Seeded M16/M17 CPU-quality and cached
  gfx1151 execution gates pass; production-shape timing and full-model quality
  remain open, so no runtime capability/default has changed.
- [ ] Account for residency explicitly. Duplicating every source-F16 projection
  would cost about 5.61 GB; prefer a replacement/composite device layout with
  offsets usable by the exact rows=1 fallback, or justify the sidecar against
  the 120 GiB admission budget.
- [ ] Select the row threshold from measured shapes. Rows=1 must remain on the
  current exact GEMV, so decode performance and arithmetic stay unchanged.

A reassociated matrix path may be admitted without byte identity only through
the quality lane below. A library control is a ceiling/diagnostic and must not
become a hard runtime dependency without an explicit package decision.

#### AR-O3 — larger row substrate and independent chunk policies

The current owner allocates one global 128-row scratch shape. That is adequate
for exact LPF but can starve grouped experts and matrix tiles. Do not simply set
the global chunk to 512.

- [ ] Add bounded 256/512-row scratch and admission accounting after AR-O1/O2
  establish the layouts they actually need.
- [ ] Decouple projection/MoE row tiles from attention query tiles, following
  the proven Qwen prefill configuration pattern. Matrix work may use M256/512
  while SWA/global attention remains at its independently measured query tile.
- [ ] Compare 128/256/512 on 512/1K/4K with exact final cursor, KV state, and
  511/512/513 wrap behavior. Canonical 68-122-token prompts will not benefit
  from a larger maximum by themselves; this phase targets matrix occupancy and
  long-context passes.
- [ ] Keep request/chunk metadata resident and reusable, but do not add graph
  capture unless kernel-span residual has become material.

#### AR-O4 — dense/shared/router residual

After AR-O1/O2, reprofile before deciding what remains. Likely candidates are
shared-expert Q4/Q6 projections, layer-0 dense Q4/Q6, and the F32 router.
Transfer the promoted Qwen GGUF Q8T16/MMQ and token-tiled F32-router schedules
through registry keys where their shape/quant contracts match. Pair gate/up,
fuse SiLU, or fuse the following norm/residual only when the new profile gives
the family at least a 5% full-model ceiling. The rejected LPF-3 pack8 dual launch
must not be revived without a different resident layout or execution schedule.

#### AR-O5 — context attention after matrix work

Attention is not the short-prompt first move, but it will become dominant as the
linear families accelerate and already matters at 4K.

- [ ] Reprofile at 512/1K/4K after AR-O1 through AR-O4.
- [ ] For SWA, process multiple query rows per tile, reuse the 512-token K/V
  window, and use online softmax instead of one serial scan per score/value.
- [ ] For global layers, screen the existing torch-free AOTriton adapter as a
  ceiling, then implement/adapt a tiled causal GQA route only if the measured
  threshold warrants it. The rejected paired-head exact kernel is not a Flash
  attention test.
- [ ] Preserve complete `KVLiveSpans`, physical SWA rings, absolute positions,
  eviction masks, BF16 K/V rounding, and the separate softplus output gate.
  Keep the exact global/SWA kernels as fallbacks below the selected threshold.

#### AR-O6 — submission and serving only after a new profile asks for it

Graph replay, cross-layer launch fusion, and packed multi-request prefill remain
deferred while kernel sum explains wall time. Re-open this phase only when
kernel-span minus kernel-sum exceeds 5% or HIP API tracing names repeated
synchronization/copies. Packed c>1 prefill is a separate serving throughput
milestone and must not be used to claim a c=1 latency win.

### Correctness, quality, and performance admission

Two candidate lanes are allowed:

1. **Exact lane:** primitive bytes, full logits/hidden/state, token IDs, cursors,
   and lifecycle match the current route.
2. **Quality-gated throughput lane:** reassociated WMMA/MMQ, activation
   quantization, or online softmax may differ numerically. It must pass the
   repository kernel gate (KL <= 0.05 and top-1 >= 90% versus the CPU/source
   oracle), the frozen Poolside first-token gate, and the complete ten-prompt
   `code/general_en/general_ja/mixed_ja_en` train+heldout teacher-forced and
   free-running suite. Report complete-ID agreement but do not require it as a
   substitute for the declared gate. No prompt/token-conditioned tuning is
   admissible.

Every candidate also requires:

- balanced same-session baseline/candidate ordering and at least three timing
  samples for a retained performance claim;
- 128/512/1K/4K reporting at milestone boundaries, with all four categories
  non-regressive and decode within 2%;
- exact 511/512/513 SWA, global cursor, KV, teardown, repeated-session, and
  bounded-memory checks;
- a prebuilt cached `rocprofv3` trace proving the intended symbol, plausible
  duration, workgroup, VGPR/SGPR, LDS, and zero or justified scratch;
- an unfused/exact registry fallback, an explicit removal trigger for temporary
  selectors in `docs/REFACTOR.md`, and the normal artifact/README/changelog/
  WORKLOG update before promotion.

Model load remains outside the prefill metric. Loader optimization is useful for
cold start but cannot be credited toward the resident TTFT or prefill-throughput
metric in this campaign. Likewise, DFlash proposal/verification time and
acceptance are out of scope until AR itself is substantially faster.

## Laguna Q2 XL Decode Optimization Campaign

The W7900 UD-Q2_K_XL route changes the decode priority established by the
mixed-F16 Q4 model on gfx1151. D0 measured **19.596 decode tok/s (51.032
ms/token)** on the retained full category suite. Exact dense decode at D1
revision `fc08ca0e` first reached **35.419 tok/s**. D2 revision `ae20392bb`
right-sized exact IQ3 selected-down K1024 to local128 and reached **38.301
tok/s**. D3 revision `fe89c210c` preserves each D2 projection's BF16 boundary
while contracting scaled routing into the IQ3 down leaf and reaches **38.840
tok/s**. D4 revision `73a2583b` makes four wave32 units compute four exact SWA
slot dots concurrently before baseline-order softmax/value consumption and
reaches **43.081 tok/s**. D5 revision `35b1602e` combines each same-input
raw-Q5 shared gate/up pair while preserving both singleton reduction trees and
BF16 stores, reaching **44.501 tok/s**. D6 revision `22e6144ce` combines each
unequal-width raw-Q5 attention query/per-head-gate pair and reaches **45.433
tok/s**. D7 implementation `973382e68`, measured clean at `51a437bc7`, is now
the default: one raw-Q6 F32 `linear_pair` launch owns each of 47 attention K/V
pairs plus layer 47's Q6 query/per-head-gate pair while preserving every
singleton output. It measures **46.409 tok/s (21.548 ms/token)** at h32,
**+2.147%** versus D6 and **+137.21%** versus D0. D6 -> D7 h32 E2E improves
**11.921 -> 11.972 output tok/s (+0.423%)** while bulk prefill remains within
guard at **43.159 -> 43.093 tok/s (-0.152%)** and median TTFT is **1.870 ->
1.873 s (+0.172% wall)**. Every category's decode/E2E row, exactness gate, and
lifecycle check passes. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-q6-attention-pair-retained.json`.
D9 then contracts the exact sparse MoE tail plus next RMS and reaches **47.132
tok/s**. D12 implementation `338d3afca` now defaults exact local32 raw-Q5
wave32x2 attention-output and unequal query/gate projections on gfx1100. Its
counterbalanced four-effective-repetition gate reaches **48.987 tok/s (20.414
ms/token)**, **+4.124%** over the paired D9 control, with every category's
decode/E2E positive and unaffected prefill neutral. Evidence:
`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d12-q5-wave32x2-retained.json`.

The frozen clean D0 at `e6120872` profiles 16 c=1 rows after the canonical
69-token `code_merge_intervals` bulk prefill; the stable 14 rows contain exactly
**1,055 dispatches/token**, **44.572 ms/token** mean summed kernels, and **49.929
ms** median embedding-to-argmax span. The profiled child wall is 18.974 tok/s;
this is attribution under `rocprofv3`, not a replacement performance claim.

The source quant recipe has a **4.144 GB active encoded-weight proxy/token**:
each dense/router tensor once, ten of 256 rows from each rank-3 expert tensor,
the complete Q4 lm-head, and one embedding row. This is not a DRAM counter and
excludes K/V, activations, cache behavior, and dequant compute, but it corrects
the older 9-10 GB mixed-F16 estimate for this Q2 model. The measured family
order is:

| D0 family | Mean ms/token | Kernel share | Calls/token | Encoded-weight proxy |
| --- | ---: | ---: | ---: | ---: |
| dense Q5 BF16/F32 outputs | **27.303** | **61.26%** | 235 | 1.931 GB / 70.7 GB/s |
| SWA decode attention | **4.237** | **9.51%** | 36 | n/a |
| selected IQ3 down | **4.021** | **9.02%** | 45 | 0.542 GB / 134.8 GB/s |
| selected IQ2 dual+SiLU | **2.318** | **5.20%** | 46 | 0.837 GB / 360.9 GB/s |
| dense Q6 BF16/F32 outputs | **2.006** | **4.50%** | 146 | 0.444 GB / 221.4 GB/s |
| Q4 lm-head | **1.618** | **3.63%** | 1 | 0.173 GB / 107.2 GB/s |
| global decode attention | **0.509** | **1.14%** | 12 | n/a |
| all other kernels | **2.560** | **5.74%** | 534 | mixed |

All 1,055 decode dispatches are classified, all 26 symbols report zero scratch,
final logits are finite, and tracked ownership returns to zero. The complete
resource sets, all per-row sums/spans, generated IDs, exact command, hashes, and
traffic caveats are in
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-decode-d0-profile.json`.

D1 keeps the same 1,055 dispatches but replaces generic dense leaves. Its clean
profile reduces kernel sum **44.572 -> 23.142 ms (-48.08%)**, median dispatch
span **49.929 -> 27.554 ms**, and profiled child wall **52.703 -> 28.820 ms
(-45.32%)**. Exact Q5 falls **27.303 -> 7.133 ms (-73.87%)** and the Q4 lm-head
falls **1.618 -> 0.376 ms (-76.75%)**. The new short-context order is:

| D1 family | Mean ms/token | Kernel share | Calls/token |
| --- | ---: | ---: | ---: |
| dense Q5 BF16/F32 outputs | **7.133** | **30.82%** | 235 |
| SWA decode attention | **4.212** | **18.20%** | 36 |
| selected IQ3 down | **4.040** | **17.46%** | 45 |
| selected IQ2 dual+SiLU | **2.295** | **9.92%** | 46 |
| dense Q6 BF16/F32 outputs | **2.050** | **8.86%** | 146 |
| global decode attention | **0.504** | **2.18%** | 12 |
| Q4 lm-head | **0.376** | **1.63%** | 1 |
| all other kernels | **2.532** | **10.94%** | 534 |

The Q5 symbols are local128, VGPR48/72, LDS1024, and scratch0. Trace SHA-256 is
`18d02d7896c43d9a6986243e562e741ff520d279d2dcc2995f207c227c61515a`.
SWA and IQ3 down are now the two largest individual short-context families.

Direct reduction-order-exact transfers of LPF-5's SWA prefill schedule do not
improve c=1. One wave32 moves SWA **4.212 -> 4.274 ms (+1.49%)** short and
**27.823 -> 29.016 ms (+4.29%)** at the 512-token physical window. A two-wave
64-thread reconstruction is neutral short (**4.210 ms, -0.034%**) and regresses
512 by **2.93%**. Both are bit-exact through positions 508-515 with an explicit
live eviction; both are removed. SWA remains a long-context target, but the
next design must add score/token parallelism or online/split reduction rather
than only remap the 128 head dimensions. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-swa-decode-rejected.json`.

The first IQ3 follow-up is retained. Applying the sibling selected-dual kernel's
wave-uniform super-block base to selected down is address-only and bit-exact.
Two actual `E256/K1024/N3072/top-10` paired medians improve **0.69%/0.75%**;
clean family time falls **4.040 -> 4.002 ms (-0.94%)** and total kernel sum falls
**0.20%**. The clean category suite is non-regressive with every route/category
positive, but its 0.51% decode delta is not promoted as a new headline because
it exceeds the physically attributable leaf gain. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-iq3-wave-base-retained.json`.

D2 right-sizes the same exact IQ3 selected-down schedule. At K1024 there are
only 128 eight-value work units, so local256's final four waves do no arithmetic
and contribute only +0 to the established reduction. The wrapper defaults only
K1024 to local128; all other shapes and the explicit rollback retain local256.
Actual `E256/K1024/N3072/top-10` weights are BF16-bit exact and improve the
clean paired median **43.61%**. Local64 is rejected because its per-thread
second group changes one of 30,720 BF16 outputs. Clean cached profiling confirms
local128/VGPR32/LDS512/scratch0, moves the 45-call family **4.002 -> 2.258
ms/token (-43.57%)**, and reduces total kernel sum **23.097 -> 21.302 ms/token
(-7.77%)** without changing 1,055 dispatches/token. The full category suite
replaces the D1 headline as recorded above. The earlier 29.452 tok/s W7900
DFlash row is D0-relative and no longer a current speedup claim; target/DFlash
economics require a fresh matched rerun after AR optimization closes. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-iq3-local128-retained.json`.

D3 fuses IQ3 selected-down with its scaled routing reduction only for c=1.
The registered unfused selected-single plus weighted-sum chain remains the bulk
and registry fallback. Each route keeps D2's local128 reduction and BF16
rounding before a slot-ordered FMA. Synthetic tokens=1/2 and actual
`E256/K1024/N3072/top-10` outputs are BF16-bit exact; the clean paired micro
median improves **18.13%**. Cached profiling confirms
local128/VGPR32/LDS512/scratch0, removes 45 weighted-reduce launches/token
(**1,055 -> 1,010**), moves IQ3 down plus selected reduction **2.392 -> 2.115
ms/token (-11.61%)**, total kernel sum **21.302 -> 20.997 ms (-1.43%)**, and
median dispatch span **25.524 -> 25.037 ms (-1.91%)**. The full category suite
promotes the D3 headline above. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-iq3-weighted-down-retained.json`.

A second clean D0 at `b4973769` extends the same synthetic canonical token
stream to 512/1K/3,968 prompt tokens and profiles eight c=1 steps at each shape
(six stable rows after two disclosed warmups):

| Admitted regime | Kernel sum | Dispatch span | Profiled child wall | Dense Q5 | SWA | Global |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| positions 69-84 | 44.572 ms | 49.929 ms | 52.703 ms / 18.974 tok/s | 27.303 ms | 4.237 ms | 0.509 ms |
| positions 512-519 | 70.686 ms | 76.434 ms | 79.432 ms / 12.589 tok/s | 27.318 ms | 27.823 ms | 2.988 ms |
| positions 1,024-1,031 | 73.947 ms | 80.125 ms | 82.839 ms / 12.072 tok/s | 27.424 ms | 27.903 ms | 5.922 ms |
| positions 3,968-3,975 | 90.605 ms | 96.493 ms | 99.248 ms / 10.076 tok/s | 27.401 ms | 27.927 ms | 22.713 ms |

Dense Q5 is context-invariant. SWA reaches its physical 512-token window and
plateaus near 27.9 ms, while global attention grows nearly linearly and becomes
the third combined family/second individual symbol near 4K. The fixed 1,055-
launch span gap remains about 5.4-6.2 ms. Thus Q5 is the first canonical-short
route, but **512+ decode cannot approach 50 tok/s without SWA**, and near-4K also
requires global attention work. All context rows are finite, lifecycle-exact,
fully classified, and scratch-free. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-decode-context-profile.json`.

The clean D3 context rerun confirms what the retained dense/IQ3 work changed and
what it did not. At 512/1K/near-4K, stable kernel sum is
**47.088/50.102/66.900 ms/token**, median span is **51.450/54.520/71.395 ms**,
and profiled throughput is **18.476/17.418/13.485 tok/s**. This is
**+46.76%/+44.29%/+33.84%** versus D0 at the same regimes. Dense Q5 is now
**7.12 ms/token**, but SWA remains **27.776/27.846/27.901 ms** and consumes
**58.99%/55.58%/41.71%** of kernel sum; global attention remains
**2.976/5.885/22.638 ms**. Thus token/score-parallel or split/online SWA was the
mandatory next 512+ target, with global attention additionally mandatory near
4K. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-decode-context-d3-profile.json`.

D4 closes that SWA item with exact four-slot score parallelism. Focused tracing
moves six calls **792.747 -> 237.722 us median (-70.01%; 3.335x)**. Clean short,
512, 1K, and near-4K traces move SWA **4.202/27.776/27.846/27.901 ->
2.118/13.111/13.096/13.104 ms/token (-49.60%/-52.80%/-52.97%/-53.03%)**;
kernel sum falls **9.59%/31.09%/29.47%/22.17%**, span falls
**8.62%/28.87%/27.45%/21.00%**, and profiled child throughput rises
**8.56%/38.56%/37.52%/25.51%**. The local128 leaf uses VGPR24, **4,120 B
dynamic LDS**, static-LDS0, and scratch0 while dispatches remain 1,010/token.
The baseline stays registered and explicitly selectable; gfx1151 and unmeasured
backends retain it. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-swa-token4-retained.json`.

The retained D4 trace plus exact GGUF inventory makes dense Q5 the next bounded
short-context family: **235 calls/token, 7.123 ms, and 37.52% of kernel sum**
for a 1.931-GB encoded-weight proxy (271.1 GB/s). The most underfilled subset is
46 same-input K3072/N1024 shared gate/up pairs: **92 launches and 1.561
ms/token** at 127.4 GB/s proxy. The next RED candidate is one registered Q5
`linear_pair` launch per pair using independent block sets and the current
single-projection K order, coefficient hoist, reduction tree, and BF16 stores.
The two singletons remain the unfused fallback; actual-weight family timing and
the full exact category gate decide retention. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-d4-q5-profile.json`.

D5 cleanly retains that RED boundary. One four-axis
`linear_pair/gguf_q5_k/pack8_gemv_decode_bf16_bf16_out` launch uses independent
projection workgroups around the exact singleton block body; Laguna asks only
for registered decode pairs, so every registry/shape miss still executes the
two primitives and rows>1 is unchanged. Synthetic K3072/N1024 outputs are
BF16-bit exact to two singleton launches, actual layer-1/layer-47 CPU-quant
oracles pass, and actual `blk.1` gate/up wall improves **28.148 -> 16.373
us/pair (-41.83%)**. Clean short tracing confirms local128/VGPR72/LDS1024/
scratch0, removes 46 dispatches/token, moves the pair family **1.561 -> 0.890
ms/token (-42.99%)**, complete Q5 **7.123 -> 6.366 ms (-10.62%)**, kernel sum
**18.983 -> 18.260 ms (-3.81%)**, and span **22.878 -> 21.981 ms (-3.92%)**.
At 512/1K/near-4K, kernel sum improves **2.13%/1.68%/1.18%**, span improves
**2.25%/2.02%/1.55%**, and profiled child throughput improves
**1.78%/1.14%/1.50%**. All trace IDs/lifecycle gates and the complete category
suite pass, promoting the D5 headline above. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-q5-shared-pair-retained.json`.

D5's clean short trace now ranks dense-Q5 BF16/F32, selected IQ2, SWA, and
weighted IQ3 at **2.744/2.732/2.300/2.120/2.093 ms/token**. The F32 Q5 family
is exactly 47 same-input attention query/gate pairs: 35 K3072 N9216+72 SWA
layers and 12 K3072 N6144+48 global layers. Query weights consume 2.134 ms at
a 392.3 GB/s encoded-weight proxy, but the tiny gates consume **0.598 ms and
47 launches at only 10.93 GB/s**. The next RED candidate is therefore one
registered unequal-width F32 `linear_pair` launch whose flattened pack grid
maps each independent workgroup to query or gate and invokes the current exact
singleton block body. Rows>1, registry misses, unsupported shapes, and
unmeasured backends retain two singleton launches. Completely hiding the gate
side is only a **0.598 ms / 3.28% kernel-sum / 2.72% span ceiling**, approximately
**45.747 tok/s**, so this candidate cannot close 50 tok/s alone. Actual-weight
SWA and global pairs, clean profile span, full category correctness, and
lifecycle decide retention. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-d5-residual-profile.json`.

D6 cleanly retains that boundary. The registered
`linear_pair/gguf_q5_k/pack8_gemv_decode_bf16_f32_out` flattened pack grid
preserves both singleton F32 outputs byte-for-byte at K3072 N6144+48 and
N9216+72. Actual global/SWA pair medians improve **23.04%/25.83%**. Rows>1,
registry/shape misses, mixed quants, F16 residency, Q6 layer 47, and unmeasured
backends retain the established unfused path. Clean tracing removes 47
dispatches/token (**964 -> 917**) at local128/VGPR48/SGPR128/LDS1024/scratch0.
Short/512/1K/near-4K kernel sum improves **3.13%/1.91%/1.70%/1.21%**, span
improves **3.21%/2.15%/1.95%/1.38%**, and profiled child throughput improves
**3.89%/2.87%/1.81%/1.48%**. All trace and complete category gates pass,
promoting the D6 headline above. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-q5-query-gate-pair-retained.json`.

D7 cleanly retains the Q6 pair boundary. The registered
`linear_pair/gguf_q6_k/pack8_gemv_decode_bf16_f32_out` flattened pack grid
preserves singleton F32 output bytes for K3072 N1024+1024 and N9216+72. Actual
global K/V, SWA K/V, and layer-47 query/gate medians improve
**37.36%/36.65%/8.80%**. Rows>1, registry/shape misses, mixed pairs, and
unmeasured backends retain the established unfused path. Clean tracing removes
48 dispatches/token (**917 -> 869**) at local128/VGPR56/SGPR128/LDS512/scratch0.
Short/512/1K/near-4K kernel sum improves **2.38%/1.60%/1.46%/0.86%**, span
improves **2.63%/2.00%/1.82%/1.17%**, and profiled child throughput improves
**3.33%/1.80%/2.31%/1.53%**. All trace and complete category gates pass,
promoting the D7 headline above. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-q6-attention-pair-retained.json`.

The code-identical D7 residual analysis ranks Q5 attention output, selected IQ2
dual+SiLU, retained Q5 query/gate, weighted IQ3 down, and token4 SWA at
**2.646/2.317/2.171/2.134/2.119 ms/token**. The first family contains 47
raw-Q5 BF16-output N3072 projections: K6144 in 12 global layers and K9216 in 35
SWA layers. Existing local128/VGPR72/LDS1024/scratch0 pack8 blocks reread a
304.35-MB/token BF16 activation proxy around 836.96 MB of encoded weights. D8
screened one exact 16-output workgroup that preserved each output's
K/reduction/BF16 order while halving nominal duplicate activation reads. Both
synthetic and actual K6144/K9216 N3072 outputs were BF16-bit exact, but the best
scratch-free local256/VGPR88/LDS2048 schedule regressed production HIP-event
time **16.69%/19.04%**. The proxy therefore described nominal traffic, not a
cache-visible win. Tile16/tile32 were removed before full-model measurement;
one-step graph replay is now the bounded route against the 3.385-ms short
span-minus-kernel residual. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-d7-residual-profile.json`
and
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-q5-output-tile16-rejected.json`.

The preceding code-identical D6 reanalysis ranked short attention-output Q5,
selected IQ2, retained query/gate Q5, SWA, weighted IQ3, and attention K/V Q6
at **2.628/2.285/2.151/2.118/2.110/1.407 ms/token**. Q6 K/V was the next
bounded exact candidate: 47 same-input K3072/N1024 pairs account for **94
launches/token** at **14.920 us** median per singleton and
local128/VGPR48/SGPR128/LDS1024/scratch0. One registered equal-width
`linear_pair/gguf_q6_k` dispatch can preserve the existing arithmetic while
keeping both singleton fallbacks. Perfect one-side overlap is only **0.703 ms /
3.98% kernel sum / 3.31% span**, approximately **46.933 tok/s**, so later work
must still compound. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-d6-residual-profile.json`.

The Qwen3.6 UD-Q3_K_M final D0 is a useful tactics comparison, not a model ratio:
it uses 671 dispatches, 8.825 ms summed kernels, and 11.347 ms profiled wall per
token on an RX 7900 XTX. D1 has now transferred its dedicated dense pack8
strategy to Laguna: raw Q4/Q5/Q6/Q8 rows=1 use exact decode leaves, rows>1 keep
their existing prefill route, and registry misses preserve the generic
fallback. Laguna already benefits from IQ2 dual+SiLU, and D3 now contracts the
IQ3 projection/reduction tail by 45 launches/token; one-step graph replay is
still absent.

Proceed in measured Amdahl order:

1. **DONE:** exact Q5 plus existing exact Q4/Q6/Q8 dense decode leaves reduce
   canonical D0 wall by 45.3% under profiling and full-suite wall by 44.8%;
2. **REJECTED:** direct one-wave and two-wave exact SWA prefill transfers are
   neutral/regressive; retain local128 and revisit only with token-parallel or
   online/split softmax while preserving `KVLiveSpans` and wrap fixtures;
3. **DONE:** retain IQ3 wave-uniform addressing, K1024 local128, and the exact
   routing-weighted down composite; D3 removes 45 launches and improves h32
   decode 1.407% over D2;
4. **DONE:** reprofile at 512/1K/near-4K after D3; SWA is now 41.71-58.99%
   of kernel sum and remains the mandatory next target, while near-4K global
   attention remains 22.638 ms/token;
5. **DONE:** token4 score-parallel SWA preserves `KVLiveSpans`, wrap,
   eviction, and exact reduction/softmax boundaries; clean full-model traces
   cut SWA 49.60% short and 52.80-53.03% at 512/1K/near-4K, and the complete
   category suite promotes D4 to 43.081 tok/s;
6. **DONE:** exact Q5 shared gate/up pairing removes 46 launches/token, reduces
   clean short Q5 10.62%, improves every context profile, and promotes D5 to
   44.501 tok/s;
7. **DONE:** exact unequal-width Q5 attention query/gate pairing removes 47
   launches/token, improves every clean context and category decode/E2E row,
   and promotes D6 to 45.433 tok/s;
8. **DONE:** reprofile D6; short attention-output Q5, IQ2, retained Q5 pair,
   SWA, IQ3, and Q6 K/V rank at 2.628/2.285/2.151/2.118/2.110/1.407 ms/token,
   while near-4K remains global-attention dominated;
9. **DONE:** exact raw-Q6 attention pairing removes 48 launches/token, improves
   every clean context and category decode/E2E row, and promotes D7 to 46.409
   tok/s; and
10. **DONE:** reprofile D7; Q5 attention output is the largest named short leaf
   at 2.646 ms/token and exact tile16 activation reuse is the first screen;
11. **REJECTED:** tile16 is exact and scratch-free, but the best production
   schedule regresses global/SWA actual weights **16.69%/19.04%**; tile16 and
   tile32 are removed without spending clean full-model/category runs; and
12. **DONE:** freeze the Laguna one-step graph ABI, fail-closed eligibility,
   lifecycle, RED gates, and 3.385-ms residual model below; and
13. **REJECTED:** the pointer-bound graph passes a 956-step exact state gate but
   regresses unprofiled short/512/1K/near-4K throughput by **2.247%, 1.995%,
   1.502%, and 1.146%** and capture-inclusive canonical h16/h32 decode by
   **7.150%/4.774%**.
   The graph owner, selector, capability, paired tail, and tests are removed;
   eager D7 resumed as the only route at that checkpoint.
14. **DONE:** exact aggregate sparse MoE-tail plus next-RMS fusion removes
   **94 dispatches/token (869 -> 775)** while preserving both BF16 add
   boundaries. Every clean context profile improves, every category's decode
   and E2E rows improve, and D9 promotes h32 decode to **47.132 tok/s**; and
15. **DONE:** reprofile retained D9; short Q5 output, selected IQ2, retained Q5
   query/gate, token4 SWA, and weighted IQ3 rank at
   **2.659/2.358/2.189/2.153/2.131 ms/token**, while SWA dominates from 512
   tokens and near-4K remains global-attention led; and
16. **REJECTED AND REMOVED (D10):** exact local256 token8 SWA improves every
   clean short/512/1K/near-4K mechanical row, but the canonical suite fails
   aggregate and every-category h16 non-regression. The token8 kernel, wrapper,
   registry entry, tests, and selector are removed; token4 remains the gfx1100
   default with baseline fallback; and
17. **REJECTED AND REMOVED (D11):** the exact persistent-counter router/top-k
   composite preserves every output/state bit and removes **47
   dispatches/token**, but the predeclared clean mechanical gate fails. Three
   short matched pairs put pooled kernel sum at **17.269 -> 17.277 ms/token
   (+0.046%)** despite positive span/child rows. The composite, counter,
   selector, and tests are removed before the category gate; split D9 remains;
   and
18. **DONE (D12):** exact local32 raw-Q5 wave32x2 attention-output and unequal
   query/gate leaves improve every formal actual-weight leaf **13.63-24.80%**,
   every clean context kernel/span/child row, and every counterbalanced category
   decode/E2E row. gfx1100 defaults both roles and retains pack8 fallback;
   canonical h32 decode is **48.987 tok/s**.

**50 tok/s is a credible W7900 target, not a current claim.** Retained D12 must
reduce the canonical **20.414 ms to 20 ms**, another **0.414 ms / 2.03% wall**
or **2.07% throughput**. Its clean short kernel sum is **16.486 ms** and median
span is **19.567 ms**, so the measured device window does not impose a 20-ms
floor. The rejected tile16 traffic model was not cache-visible, ROCm graph
replay regressed, D10's positive mechanical/h32 diagnostics were not retainable
after h16 category regressions, and D11's launch contraction failed the clean
short kernel-sum gate. The near-4K profile remains led by global attention.
Every retained candidate uses the full category/heldout suite and the same
exact/quality lanes above. Laguna DFlash/MTP economics must use D12 or a later
true-AR baseline rather than the historical D0 row.

### D8 one-step graph replay (exact, performance-rejected, removed)

This design is Laguna-specific. It reuses HIP graph mechanics and the small
runtime-state kernel family, but it does not inherit Qwen/PARO graph correctness.
The current eager ownership audit is:

| Semantic state | Current owner | Current per-token action |
| --- | --- | --- |
| input token | `LagunaEagerScratch.token_id` | host writes the previous top-1 ID |
| RoPE position | `LagunaEagerScratch.position` | host writes `session.position + 1` |
| KV query/write position | `LagunaKVCache._row_position` through every `KVLiveSpans.row_positions` | `prepare_position()` writes the same absolute position |
| dense/SWA visibility | per-layer `base_offsets`, `live_counts`, `token_positions`, `evict_mask` | KV append kernels update metadata; attention reads it |
| next token/value | `argmax_id` / `argmax_value` | argmax writes device scalars, then eager synchronizes and reads both |
| committed cursor | `session.position` and `kv_cache.position` | host advances both only after the complete step succeeds |

The graph must remove Python/ctypes submission of the **869** D7 launches, not
change any model arithmetic. Its fixed one-step body is:

1. read the input ID directly from `scratch.argmax_id` and launch the existing
   quantized embedding into `scratch.hidden`;
2. run the unchanged 48-layer c=1 chain, with RoPE reading
   `scratch.position` and all KV write/attention leaves reading their complete
   device-resident `KVLiveSpans`;
3. run the unchanged final norm, Q4 LM head, and exact F32 top-1 reduction back
   into `scratch.argmax_id` / `scratch.argmax_value`; and
4. run one registered `decode_position/laguna/advance_pair_i64` tail that sets
   both position scalars to the same next append position.

Before the first replay after eager prefill, reset, or exact prefix reuse, the
host primes both device position scalars to `p + 1`, where
`p == session.position == kv_cache.position`. The top-1 scalar already contains
the token produced by the final prefill/suffix row. After one successful replay,
the model has appended that input at `p + 1`, argmax contains the following
input, both device positions contain `p + 2`, and the host commits both cursors
to `p + 1`. Synchronize/readback failure is state-ambiguous and retires the
whole session, matching eager failure ownership. `scratch.token_id` remains the
unfused eager staging buffer and exact fallback.

The graph executable is session-local and pointer-bound but position-independent.
Global/SWA launch geometry is fixed by the admitted 4K capacity; the kernels
already read live counts, absolute token positions, eviction masks, and the SWA
ring slot from device memory. One captured step can therefore survive ordinary
session reset and exact continuation reuse while allocations and registered
variants remain unchanged. Reset/prefill marks the handle unprimed; it does not
make a second graph. The next replay re-primes dynamic state. The retained
continuation invariant is unchanged: physical KV contains
`prompt + generated[:-1]`, and the final generated token remains the device
input waiting in `argmax_id`.

Capture identity and replay state are deliberately separate:

- `LagunaDecodeGraphKey` fingerprints schema, backend/target arch, model/config
  identity, all quant/layout roles, layer/head pattern, 4K/global/SWA capacities,
  BF16 KV, selected-down and attention variants, every weight/RoPE/scratch/MoE/
  KV/span/control pointer, and the one-step raw-greedy sampler contract;
- `LagunaDecodeReplayTicket` carries the session lifecycle epoch, committed host
  cursor, expected next absolute position, dense-span policy, pending-transaction
  state, remaining capacity, and whether device controls have been primed for
  this epoch; and
- reset, eager prefill/decode, and accepted continuation update the ticket.
  Close/address change destroys graph exec, graph, then stream. Manual eviction,
  verifier staging/rollback, or any pointer/variant change invalidates replay
  until a reset or a new exact capture.

This split avoids hashing or reading hundreds of megabytes of live KV on every
step while still covering the old L8 semantic-key requirement. For admitted raw
AR, global live count is `p + 1`, SWA live count is `min(p + 1, 512)`, and slot
positions/eviction state are deterministic from the cursor because explicit
manual eviction is not eligible. The RED state gate fingerprints all payload and
metadata at checkpoints rather than trusting that inference.

#### Fail-closed eligibility

The first implementation is eligible only when every condition holds:

- backend capability explicitly certifies gfx1100; target arch is gfx1100,
  physical width is c=1, context is at most 4K, KV is BF16, and all current
  Laguna D7 registry variants resolve before capture;
- the request is exact raw greedy: temperature 0, top-p 1, top-k 0, min-p 0,
  no penalties/bias/suppression/forced tokens/logprobs, and no external logits
  processor. Host EOS/EOT/stop-sequence checks remain supported because one
  replay exposes exactly one token before the next model transition;
- no hidden-tap capture, DFlash/MTP verifier, staged/pending KV transaction,
  explicit eviction, sampling, multi-step replay, user stream, c>1 batch, or
  diagnostic full-logit readback is active;
- session and KV cursors agree, the next position fits capacity, all static
  addresses match the key, and the dynamic ticket is primed for the current
  lifecycle epoch; and
- any miss runs the existing eager `forward_token()` path. Graph support never
  removes or numerically changes that fallback.

A one-step graph is intentional. Multi-step replay cannot observe host stop,
cancellation, deadline, or stream backpressure between tokens and is outside D8.
Likewise, gfx1151 remains eager until it passes its own device-specific gate; the
Qwen gfx1151 graph rejection is an additional warning, not evidence about Laguna.

#### RED/GREEN and lifecycle gates

Task #277 must add the tests before default wiring:

1. host-only key/ticket tests cover every axis above, pointer and variant drift,
   reset/re-prime, exact continuation, double-close, partial-capture cleanup, and
   every fail-closed reason;
2. a GPU eager-versus-graph trajectory gate runs identical real Q2 XL sessions
   and compares every generated ID, top-1 value bits, complete FP32 logits,
   post-layer/final BF16 hidden, both position scalars, and a digest over all 48
   K/V payloads plus every `KVLiveSpans` field after each selected step;
3. checkpoints include short state, global position 255/256, SWA
   510/511/512/513 wrap, repeated-ring 1023/1024, reset then a third-or-later
   replay, and exact retained-prefix + unmatched-suffix continuation. This
   specifically guards the prior Qwen third-launch/state-corruption class;
4. capture must not execute or mutate state. Replays allocate zero bytes, use one
   `hipGraphLaunch` per token, retain the expected D7 kernel symbols plus only
   the paired position tail, and preserve the unfused eager result after graph
   use; and
5. the complete train+heldout four-category h16/h32 suite remains deterministic,
   serial/bulk/graph IDs are exact, Poolside KL/top-1 and accepted bulk-state
   gates pass, and every graph/session/reset/continuation/error lifecycle returns
   tracked ownership to zero.

The task #277 implementation passed the frozen state/lifecycle gate on the real
UD-Q2_K_XL model. Two resident sessions remained byte-exact for every ID,
argmax-value bit pattern, complete FP32 logits, BF16 final/post-layer hidden,
all 48 complete K/V allocations, and every live-span metadata allocation across
**956** graph transitions. Checkpoints covered positions 69-72, 255/256,
510-513, and 1023/1024, then exact retained-prefix suffix prefill, a forced eager
fallback, graph resumption, reset, and third replay. Capture was non-executing,
replay performed no tracked device allocation, one cached trace contained one
`hipGraphLaunch` and one paired-position tail, and teardown returned
**40,455,911,848 bytes / 1,496 allocations** to zero.

Task #278 nevertheless rejects and removes the route. Counterbalanced
unprofiled graph throughput regresses eager by **2.247%/1.995%/1.502%/1.146%**
at short/512/1K/near-4K. Cached traces also increase kernel sum at every context
by **3.018%/1.076%/0.524%/0.151%**; short/512 dispatch span regresses. The full
capture-inclusive ten-prompt gate is exact and returns ownership to zero, but
h16/h32 decode falls **46.827/46.409 -> 43.480/44.193 tok/s
(-7.150%/-4.774%)** and E2E falls **6.881/11.972 -> 6.819/11.839
(-0.902%/-1.110%)**. Every category regresses both decode and E2E horizons.
ROCm graph-node scheduling costs more than the host submission it replaces for
this chain, so there is no minimum replay horizon to admit. Canonical eager D7
remains the only route; no selector, graph owner, capability, paired tail, or
runtime-state debt remains. Evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-decode-graph-correctness.json`
and
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-decode-graph-rejected.json`.

Graph capture/instantiate latency, first replay, and warm replay are separate
measurements. The canonical fresh-session run includes lazy capture in its
request/E2E wall; steady-state replay may report capture separately but cannot
hide it in model/session construction. Pooled-session and exact-prefix-reuse
rows additionally show amortized behavior. Promotion requires a clean cached
`rocprofv3` trace at short/512/1K/near-4K, non-regressive kernel sum, lower span
and host wall at every context, positive complete-suite and every-category h16/
h32 decode **and E2E**, no prefill/TTFT regression outside the existing guard,
and an eager rollback switch. A graph that only improves a single prompt or a
capture-excluded number is rejected.

#### Bounded performance model

Let `r` be the fraction of D7's measured **3.385 ms/token** short submission
residual removed, `g` the added per-token graph launch/sync/read/state-tail cost,
and `C` the one-time capture/instantiate cost. The canonical h32 row has 31
model-forward calls, so the cold bound is:

```text
T_h32 = 21.548 ms - (3.385 ms * r) + g + C / 31
```

Ignoring `g` and `C` only to show the ceiling:

| residual removed | modeled h32 ms/token | modeled tok/s |
| ---: | ---: | ---: |
| 25% | 20.702 | 48.31 |
| 50% | 19.856 | 50.36 |
| 75% | 19.009 | 52.61 |
| 100% | 18.163 | 55.06 |

Reaching 50 tok/s requires
`r > (1.548 + g + C/31) / 3.385`; even with zero overhead the graph must remove
**45.73%** of the residual. At 50% removal only **0.145 ms/token** remains for
all graph and amortized capture overhead. The clean result resolves the model
negatively: graph h32 is **44.193 tok/s**, not 50, and even steady-state replay
regresses every unprofiled context. The
capture/state-tail and graph scheduling overhead exceed any host-submission
saving. Task #278 therefore removes the route and kernel optimization resumes
from eager D7.

### D9 aggregate MoE tail plus next RMSNorm (retained exact default)

The next candidate is a measured launch-contraction boundary, not another graph
or quant-math rewrite. Splitting every retained D7 token from embedding through
argmax finds exactly **47** adjacent sequences of:

```text
BF16 routed + shared add
BF16 post-attention + MoE add
F32-weight RMSNorm for the next layer (or final output_norm)
```

All 45 IQ3-down layers already emit one weighted BF16 `routed_output`. The other
two sparse layers run their exact selected-down plus weighted-sum fallback before
the tail, so the candidate never serializes selected slots. Short/512/1K/
near-4K control windows are **0.517/0.516/0.518/0.519 ms/token**. Replacing each
three-kernel boundary with one call removes **94 launches/token**, or 10.82% of
D7's 869 dispatches, and leaves dense layer 0 plus all rows>1 paths unchanged.

The proposed four-axis key is
`(hip_gfx1100, moe_tail+next_rmsnorm, bf16,
laguna_aggregate_gguf_f32_weight_out)`. One local256 workgroup emits both the
post-MoE BF16 hidden row and the BF16 next-normalized row. The arithmetic
contract is stricter than qwen-kernel's F32 `add_rms3.comp` and the existing Q3
composite:

1. round `routed + shared` to BF16;
2. reread that value, add BF16 `post_attention`, and round the hidden row to
   BF16;
3. store that hidden row before accumulating squares in the exact current
   `idx=tid,tid+256,...` order;
4. use the same local256 stride-128..1 reduction and `rsqrtf(sum/3072 + eps)`;
5. multiply the stored BF16 hidden value by the F32 next norm weight and round
   the normalized output to BF16.

The host supplies either layer `L+1`'s `attn_norm` pointer or the final
`output_norm` pointer. This changes neither dispatch axes nor model arithmetic.
The current two BF16 adds plus standalone F32-weight RMSNorm remain the required
unfused fallback for registry misses, rows>1, unsupported backends, explicit
rollback, and any failed gate. Source scheduling references are qwen-kernel
`52e240f9` `shaders/add_rms3.comp` and the in-tree Q3
`moe_tail_next_rmsnorm_out_kernel`; neither reference's one-round/shared-sigmoid
math may be copied into Laguna.

The short measured window is 2.995% of kernel sum and 2.497% of span. Its
zero-cost kernel-only ceiling is **47.55 tok/s**. A planning-only transfer that
scales the measured Q3 fused body from hidden 2048 to 3072 and applies the
consistent D6/D7 span-minus-kernel reduction per removed launch models
**0.347 ms/token** saved and **47.17 tok/s**. Even a zero-cost body plus that
launch-gap transfer reaches only **48.17 tok/s**. These are Amdahl bounds, not a
performance claim.

RED requires BF16-bit-exact residual and normalized outputs at hidden 17/3072,
including values where omitting the first BF16 boundary changes bits, plus
explicit rows>1/registry/backend fallback. GREEN requires all 47 real-model
boundaries exact, local256/LDS1024/scratch0 tracing, and exactly
**869 -> 775** dispatches/token. Promotion additionally requires clean short/
512/1K/near-4K kernel-sum and span wins plus the complete ten-prompt,
four-category h16/h32 exact state/KV/lifecycle gate. Evidence and full gate:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-d9-moe-tail-next-rms-design.json`.

Task #280 correctness-admitted the gfx1100 c=1 implementation. The registered
local256 leaf preserves both BF16 boundaries byte-for-byte at hidden 17/3072;
a shared-weight Q2 XL gate matches all 47 actual sparse boundaries, full logits,
argmax bits, final norm, complete K/V/live spans, reset, and lifecycle through
16 decode steps. Cached matched dirty traces show exactly 47 calls and
**869 -> 775 dispatches/token** at VGPR16/SGPR128/LDS1024/scratch0. Kernel sum
is effectively flat within dirty-run noise (**17.296 -> 17.288 ms/token**), while
span improves **20.702 -> 20.383 ms (-1.545%)** and profiled child throughput
improves **43.890 -> 45.003 tok/s (+2.536%)**. The exact three-kernel route
remains available through registry miss, rows>1, gfx1151, and the temporary
explicit constructor rollback. Correctness evidence:
`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d9-moe-tail-next-rms-correctness.json`.

Task #281 promotes D9 on clean evidence. Short/512/1K/near-4K candidate versus
explicit D7 fallback improves kernel sum **0.320%/0.515%/0.462%/0.117%**, median
embedding-to-argmax span **2.667%/1.551%/1.431%/0.751%**, and profiled child
throughput **3.104%/2.387%/2.297%/1.015%**. Every stable candidate row has 775
dispatches and 47 fused calls. The fused body costs **0.389/0.385/0.386/0.385
ms/token**, with **8.12-8.24 us** medians at local256/VGPR16/SGPR128/LDS1024/
scratch0. All context IDs, finite outputs, and teardown are exact.

The clean ten-prompt/four-category h16/h32 gate moves D7 -> D9 bulk prefill
**43.093 -> 43.190 tok/s (+0.224%)**, TTFT **1.873 -> 1.871 s (-0.117% wall)**,
h16/h32 decode **46.827/46.409 -> 47.576/47.132 tok/s
(+1.599%/+1.560%)**, and h16/h32 E2E **6.881/11.972 -> 6.909/12.038
output tok/s (+0.411%/+0.555%)**. Every category improves decode
**0.956-1.855%** and E2E **0.261-0.759%** while prefill stays within
**+0.114% to +0.384%**. All 20 serial/bulk pairs and repeats are exact, the
Poolside gate remains KL `0.000156823`/top-1 `1.0`, accepted bulk state passes,
and tracked ownership returns to zero. Retention evidence:
`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d9-moe-tail-next-rms-retained.json`.

Canonical D9 is **47.132 tok/s / 21.217 ms/token**, still **1.217 ms** above
20 ms and requiring another **6.084% throughput** to reach 50 tok/s. The D10
residual analysis below selects local256 token8 SWA rather than assuming it is
2x faster. Fusing attention with its softplus gate remains deferred: the gate
family is only **0.121 ms/token** and 48 launches. Near-4K global attention
remains a separate requirement.

### D10 token8 exact SWA decode (performance-rejected, removed)

The retained clean D9 traces are runtime-code-identical to current `216bb0c4a`.
Short Q5 attention output, selected IQ2 dual+SiLU, retained Q5 query/gate pair,
token4 SWA, and weighted IQ3 down rank at
**2.659/2.358/2.189/2.153/2.131 ms/token**. The Q5 tile16 traffic premise was
already rejected on both production shapes; IQ2 already carries retained
branchless/pair16/local64/tile2 work and rejected inclusive dp4a; both Q5 pairs,
the Q6 pair, weighted IQ3, and D9 are already exact specialized schedules. SWA
is therefore the largest fresh context-sensitive family, and it dominates at
512+ tokens: token4 costs **2.153/13.099/13.118/13.132 ms/token** at
short/512/1K/near-4K. Global attention grows
**0.518/2.987/5.900/22.637 ms/token**, remaining the separate near-4K owner.
The complete candidate kernel sums are **17.289/30.534/33.452/50.236 ms**,
spans are **20.389/33.798/36.712/53.618 ms**, and span-minus-sum residuals are
**3.100/3.265/3.260/3.382 ms/token** across those contexts.

D10 tested a separately registered
`(hip_gfx1100, laguna_attention_decode, bf16,
swa_context_token8_exact_spans)` sibling. One local256 block still owns one
query head. Eight wave32 units compute eight independent 128-D dots using the
retained exact `((p0+p64)+(p32+p96))` then offsets 16..1 tree. Thread 0 updates
max and denominator for slots 0..7 in increasing logical order. Threads 0..127
then load and FMA the eight value rows in that same order while threads
128..255 stay arithmetic-idle but participate in every block barrier. Scores
remain unscaled in LDS so max and exponential paths retain their separate
`dot*scale` rounding; final denominator clamp and F32 context output are
unchanged. Complete `KVLiveSpans`, BF16 K/V, wrap, and eviction semantics remain
the ABI. The current token4 key remains the registry/backend/shape fallback.
Dynamic LDS rises only **4,120 -> 4,136 B** for eight batch weights; current
token4 is local128/VGPR24/scratch0.

For the stable short slots 72..85, token4 executes mean **20.000 batches / 81
block barriers per layer**; token8 models **10.286 / 42.143**, a **47.97%**
barrier reduction. At the full 512-token window, batches/barriers move
**128/513 -> 64/257 (-49.90% barriers)**. This is not a 2x throughput claim:
all BF16 K/V loads, dot arithmetic, exponentials, denominator additions, and
value FMAs remain; token8 doubles duplicated query-register loads and leaves
half the block idle during value accumulation. Actual cached leaves decide.

The Amdahl bound is explicit. D9 needs **1.217 ms** to reach 20 ms, or
**56.52%** of the measured short SWA family. Removing SWA entirely reaches
**52.46 tok/s**, while the planning-only half-SWA case saves **1.076 ms** and
reaches just **49.65 tok/s**, still **0.140 ms** above 20 ms. D10 cannot be
assumed to close 50 alone and must compound with later exact work.

RED requires token4/token8 output-bit identity at empty/short/full, 510..513
wrap, reversed physical offsets, explicit eviction, and adversarial scores.
GREEN requires the expected local256/token8 symbol, 4,136 B dynamic LDS, no
scratch, and faster cached actual short plus full-window leaves. Promotion then
requires clean short/512/1K/near-4K kernel-sum/span wins and the complete
category/state/KV/lifecycle gate. Do not widen to local512 token16 unless
token8 is exact and positive but leaves a measured synchronization share;
barrier arithmetic alone does not admit a larger block. Design evidence:
`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d9-residual-profile.json`.

Task #283 implements D10 as the gfx1100 candidate default while retaining
explicit token4 and baseline registry fallbacks. The focused 69-test bundle is
GREEN. Token4/token8 F32 output bits match at empty, short, adversarial,
full-window, 510..513 wrap, repeated 1024/1025 wrap, reversed physical offsets,
and explicit eviction; gfx1151 remains baseline. A 72-query-head production
shape micro with reversed offsets, 20 warmups, 200 event-timed launches, and 15
counterbalanced repetitions improves token4 -> token8 median
**21.396 -> 19.405 us (-9.31%)** at 80 tokens and
**169.465 -> 159.121 us (-6.10%)** at 512, with exact output and lifecycle.

The shared-weight Q2 XL gate compares token4 and token8 through bulk prefill,
all 48 post-layer hidden rows, 16 decode steps, reset/re-prefill, full logits,
argmax ID/value bits, final/post-layer hidden, complete K/V plus every live-span
field, and host/device cursors. Everything is exact and tracked ownership
returns **40,455,911,848 bytes / 1,496 allocations** to zero. The dirty
counterbalanced wall median improves **19.670 -> 19.381 ms/token (-1.465%)**.

Matched cached dirty tracing names exactly 36 token8 calls at
local256/VGPR24/SGPR128/static-LDS0/scratch0 (4,136 B launch-time dynamic LDS).
Token4 -> token8 SWA falls **2.144 -> 1.863 ms/token (-13.10%)**, complete kernel
sum **17.229 -> 16.969 ms (-1.513%)**, median span
**20.323 -> 19.988 ms (-1.651%)**, and profiled child throughput rises
**2.026%**. IDs, finite output, and lifecycle match.

Task #284's clean cached profiles confirm a real mechanical improvement at every
context. Short/512/1K/near-4K SWA changes
**2.143/13.114/13.128/13.140 -> 1.876/11.171/11.193/11.187 ms/token
(-12.47%/-14.81%/-14.74%/-14.86%)**. Complete kernel sum improves
**1.05%/6.38%/5.66%/3.84%**, span improves
**0.70%/5.81%/5.32%/3.70%**, and profiled child throughput improves
**0.91%/5.05%/5.80%/3.93%**. Every row preserves IDs, finite output, 775
dispatches/token, and teardown.

The predeclared canonical gate nevertheless rejects D10. Versus retained D9,
token8 changes aggregate h16/h32 decode **47.576/47.132 -> 48.209/47.872
tok/s (+1.331%/+1.569%)**, but h16/h32 E2E changes
**6.909/12.038 -> 6.905/12.060 (-0.055%/+0.178%)**. General-English h16
decode/E2E regress **0.535%/0.254%**; code and mixed h16 E2E regress
**0.128%/0.017%**. Prefill and TTFT remain inside the 0.5% guard, h32 rows are
positive, all 20 serial/bulk pairs and repeats are exact, Poolside KL/top-1 and
accepted bulk state pass, and ownership returns to zero, but the required
aggregate and every-category h16/h32 decode/E2E predicate is false.

Accordingly the token8 kernel, wrapper, registry entry, tests, and backend
selector are removed; retained token4 is again the gfx1100 default and the
baseline span reader remains available. The post-removal focused bundle reports
**69 passed**. Diagnostic D10 h32 was **47.872 tok/s / 20.889 ms/token**, but it
is not a retained headline. D9 remained **47.132 tok/s / 21.217 ms/token** at
that decision; retained D12 later superseded it without reviving token8.
Token16 is not admitted from a candidate that failed the complete suite. Evidence:
`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d10-swa-token8-{correctness,rejected}.json`.

### D11 persistent cooperative Laguna router (exact, performance-rejected, removed)

Retained D9 contains exactly **47** adjacent sparse-layer pairs of the current
registered BF16-hidden/F32-weight router projection and Laguna's correction-only
sigmoid top-k selector. Re-reading the code-identical clean D9 traces measures
projection / selection / complete adjacent window / inter-kernel gap at:

| Context | Projection | Selection | Pair window | Gap |
| --- | ---: | ---: | ---: | ---: |
| short | 0.323 ms | 0.394 ms | **0.896 ms** | **0.179 ms** |
| 512 | 0.317 ms | 0.390 ms | **0.884 ms** | **0.177 ms** |
| 1K | 0.316 ms | 0.387 ms | **0.880 ms** | **0.178 ms** |
| near-4K | 0.320 ms | 0.388 ms | **0.885 ms** | **0.177 ms** |

The short kernel bodies are **0.716 ms/token**, or **4.14%** of D9 kernel sum;
the complete **0.896-ms** window is **4.39%** of the embedding-to-argmax span.
Every pair is immediately adjacent and has a **18.92 us** median complete
window. This is the largest fresh exact launch-contraction boundary after D10:
Q5 tile16 failed both production shapes, IQ2/IQ3 output-tiling lanes have
cold-grid rejection evidence, token8 SWA failed the complete suite, and graph/
stream submission tactics already regressed. The standalone attention-gate and
KV-write leaves are only **0.121/0.165 ms/token**.

D11 therefore selected a separately registered
`(hip_gfx1100, laguna_router_topk, f32,
bf16_hidden_correction_bias_persistent)` composite. One local256 block computes
each of 256 logits with the exact current projection contract: each lane visits
the same eight-value K groups, performs the same scalar FP32 additions, and uses
the same shared stride-128..1 reduction. Thread 0 stores the logit, executes a
device fence, and increments one session-local `int32` completion counter. The
last block then reproduces the current selector unchanged:

1. stable positive/negative FP32 sigmoid branches;
2. separate unbiased routing scores and correction-biased selection scores;
3. ten serial maxima with lower expert ID winning exact ties;
4. sum-normalization of the **unbiased** selected probabilities; and
5. separate multiplication by the model's 2.5 routed scale.

After every output is complete, last-block thread 0 resets the dedicated counter
for the next same-stream launch. There is no selected-ID alias and no per-layer
host memset. The counter is initialized once with session scratch and may be
reused across all serial sparse layers. The existing
`router_logits/f32/bf16_hidden` plus
`laguna_sigmoid_router_topk/f32/correction_bias` primitives remain the required
rows>1, gfx1151, registry/shape-miss, rollback, and unsupported-backend
fallback. D11 removes one launch at each sparse layer, so the intended trace is
**775 -> 728 dispatches/token**.

This mechanism is not speculative. The retained Qwen F32-weight cooperative
router uses the same exact projection reduction plus last-block election; its
persistent form is local256/VGPR40/static-LDS512/scratch0 and measures **10.444
us** at hidden 2048/top-8. Laguna's hidden 3072/top-10 correction ABI is
strictly different, so that latency is transfer evidence only—not a Laguna
claim. D11 must remain scratch-free, keep VGPR at or below the declared bounded
screen, and beat the complete current projection+selection window on actual
Laguna weights before clean model runs.

The Amdahl bound forbids overclaiming. Removing only the measured **0.179-ms**
gap models **47.53 tok/s**. A planning-only fused body of 12-15 us/layer models
**47.88-47.56 tok/s**. Even making the full **0.896-ms** window free yields only
**49.21 tok/s / 20.321 ms**, still **0.321 ms** above 20 ms. The design
could only move D9 toward 50; it could not close the target alone.

RED compares all FP32 logit/routing/selection score bits, every selected ID, and
all normalized/scaled routing-weight bits against the split chain at hidden
17/3072. It covers equal-logit ties, both stable-sigmoid branches,
correction-driven flips, adversarial normalization, poisoned outputs, and two
consecutive launches with no host reset and a zero counter after each. GREEN
then requires all 47 actual layer routers, full logits/argmax bits, hidden,
complete K/V and `KVLiveSpans`, reset, and lifecycle exactness. Cached tracing
must name 47 candidate calls, 728 total dispatches, local256, bounded LDS/VGPR,
scratch0, and no per-layer fill. Promotion still requires every clean context's
kernel sum/span/profiled child plus aggregate and every-category h16/h32 decode
and E2E to improve with prefill/TTFT inside 0.5%. Evidence:
`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d11-persistent-router-design.json`.

D11 reached implementation and correctness admission before the performance
decision. Synthetic hidden-17/3072 replay is bit-exact for full FP32 logits,
unbiased/corrected scores, selected IDs, normalized/scaled weights, and two
consecutive launches; the dedicated counter reads zero after each. The
shared-weight real-model gate compares all **47** sparse-layer routers plus all
48 post-layer hidden rows, full logits/argmax bits, complete K/V and every
`KVLiveSpans` field, reset/re-prefill, and lifecycle through 16 decode steps.
Everything is exact, and **40,455,911,864 bytes / 1,500 tracked allocations**
return to zero.

A 15x100 actual-weight all-layer HIP-event screen moves the isolated split
window **0.820 -> 0.661 ms (-19.37%)**, or **17.452 -> 14.072 us/layer**.
Matched dirty tracing records 47 candidate calls at **13.76 us median / 0.652
ms/token**, local256/VGPR32/SGPR128/LDS512/scratch0, and **775 -> 728
dispatches/token**. Those diagnostics are positive, but the clean gate is the
retention authority.

Clean short/512/1K/near-4K candidate-versus-split profiles move the isolated
router body **-9.69%/-11.05%/-9.66%/-9.87%**, dispatch span
**-0.142%/-1.012%/-0.606%/-0.488%**, and profiled-child throughput
**+2.184%/+0.092%/+0.896%/+0.749%**. Complete kernel sum, however, changes
**+0.169%/-0.269%/-0.135%/-0.160%**: the short row violates the predeclared
requirement that every context improve. Two extra counterbalanced short pairs
confirm rather than clear the failure. Across three pairs, median router/span/
child changes are **-9.611%/-1.075%/+1.699%**, while median pair kernel sum is
**+0.039%** and the pooled 42-step kernel sum is **17.269472 -> 17.277499
ms/token (+0.046%)**.

The canonical category run is therefore skipped rather than used to rescue a
failed mechanical screen. The composite HIP body/export, wrapper, registry
entry, runtime selector, session counter, backend exclusion, and candidate
tests are removed; the registered BF16-hidden/F32-weight projection plus
correction-only top-k selector is again the only D9 route. The **47.132 tok/s /
21.217 ms/token** headline and 50-tok/s gap do not change. Evidence:
`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d11-persistent-router-{design,correctness,rejected}.json`.

### llama.cpp Vulkan c=1 transfer review (diagnostic)

The user's same-W7900 Vulkan `tg128` result is real, but not directly the same
product metric as canonical D9. At read-only llama.cpp revision `c0bc8591e`, an
independent same-file run reproduces **94.513 +/- 0.141 tok/s** across three
128-token repetitions versus the reported **93.67 +/- 0.16 tok/s**. Source
inspection of `tools/llama-bench/llama-bench.cpp` shows that `tg128` clears the
context, then times 128 one-row `llama_decode` calls at positions 0..127. Every
call synchronizes, but the next token is random: there is no sampler, device
argmax, or token/logit readback. Canonical hipEngine D9 instead times 31
post-TTFT calls per natural trajectory, includes GPU argmax plus two scalar
reads, uses BF16 rather than F16 KV, and hard-gates ten prompts/four categories,
full generated IDs, oracle KL/top-1, state, and lifecycle.

Context depth does not explain the gap. The ten D9 prompts are 68..122 tokens;
its h32 calls cover positions 68..152 with mean **101.4**. A matched diagnostic
Vulkan `-d 86 -n 31` run covers positions 86..116 with mean 101 and still
measures **94.152 +/- 0.331 tok/s**. Thus the roughly **2.0x** model-step gap is
genuine enough to guide engineering, but it is not a retained cross-engine
throughput ratio until a natural-token harness matches sampling/readback, KV
arithmetic, prompt trajectories, and correctness gates.

The source and same-build ablations materially change the optimization ranking:

- Vulkan graph fusion is the largest isolated control. `GGML_VK_DISABLE_FUSION`
  moves **94.513 -> 74.865 tok/s**, or **10.581 -> 13.357 ms/token (+2.777
  ms, -20.79% throughput)**. The backend fuses Laguna-relevant top-k sigmoid/
  correction/normalization, multi-add, matvec post-add/scale, selected-down
  weighting, and RMS/mul/RoPE/KV-write subgraphs. D11 tested that transfer but
  failed its clean short kernel-sum gate and is removed; D12 then retained the
  raw-Q5 geometry transfer.
- Graph sorting plus dependency-scoped barriers are source-confirmed and place
  independent Q/K/V/gate and routed/shared MoE work in concurrent groups, but
  disabling graph optimization alone costs only **0.116 ms/token / 1.08%**.
  hipEngine's graph replay and prior stream tactics regressed, so generic replay
  does not reopen.
- The default AMD heuristic's Q8_1 integer-dot MMVQ is not optimal here:
  `GGML_VK_DISABLE_MMVQ=1` improves **94.513 -> 98.568 tok/s (+4.29%)**. Do not
  port activation quantization blindly. The useful next source transfer is the
  raw one-wave/subgroup Q5 geometry for D9's **2.659-ms attention-output** and
  **2.189-ms query/gate** families, with current HIP reduction/BF16 bits and
  actual K6144/K9216/K3072 weights as hard gates. This is distinct from the
  rejected tile16 traffic-sharing design.
- After retained D12 raw-Q5 geometry, rank exact post-op fusion around Q5
  output, IQ3 selected-down weighting, and shared/routed combines ahead of a
  new one-wave attention algorithm. Token16 extrapolation remains closed by
  D10. Generic graph replay, stream overlap, and Q8_1 MMVQ stay deferred.

The Vulkan performance logger can serialize dependency groups and heavily
perturbs wall time, so its operator totals are attribution only. Complete
protocol, commands, hashes, source references, ablations, and transfer ranking:
`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-llamacpp-vulkan-review.json`.

### D12 raw-Q5 wave32x2 decode (exact, retained gfx1100 default)

The source audit selects one precise transfer rather than copying Vulkan's
arithmetic. hipEngine's retained Q5 leaf uses one local128 block for eight
output columns. For every 256-value Q5 superblock, logical thread `t` visits
`t` then `t+128`; the block hoists 8x8 `d*scale`/`dmin*min` coefficient pairs
through 1,024 B LDS, executes two barriers, reduces four physical wave32
partials independently, and lets thread 0 add logical waves 0,1,2,3. The D9
attention-output and unequal query/gate uses of this body cost **2.659** and
**2.189 ms/token**, respectively, or **28.04%** of the short kernel sum
combined.

The read-only llama.cpp Vulkan source uses a materially different raw-Q5
geometry. On the W7900's reported subgroup size 64, AMD selects the subgroup-
sized DMMV workgroup; `rm_kq=2` makes one subgroup compute two output rows,
`K_PER_ITER=8` keeps a wider serial K slice in each lane, and `subgroupAdd`
requires no cross-subgroup shared-memory exchange. That source uses F16 inputs,
F32 output, and a different dot/FMA/reduction association, so its bits are not
portable. The useful premise is only **one physical subgroup owns two output
rows**. The same-build no-MMVQ result supports staying on raw weights, but it is
a full-model ablation rather than a Q5-family speed claim.

D12 therefore freezes a native-wave32, two-output schedule that reconstructs
the current HIP arithmetic exactly:

1. physical lane `l` maintains four logical partials for threads
   `l,l+32,l+64,l+96` and two output columns;
2. within each Q5 superblock, those partials visit
   `[l,l+128]`, `[l+32,l+160]`, `[l+64,l+192]`, and
   `[l+96,l+224]`, preserving every baseline logical thread's K sequence;
3. lanes 0..7 produce the first output's eight coefficient pairs and lanes
   8..15 the second's, then wave shuffles broadcast them without arithmetic
   reassociation;
4. each logical group runs the same offsets 16,8,4,2,1 shuffle tree, and lane 0
   starts from `+0.0` and adds groups 0,1,2,3 before the unchanged F32 or
   RNE-BF16 store; and
5. the unequal pair maps query and gate as separate even tile ranges so no
   two-output tile crosses buffers.

The first implementation keys are
`linear/gguf_q5_k/wave32x2_gemv_decode_bf16_bf16_out` and
`linear_pair/gguf_q5_k/wave32x2_gemv_decode_bf16_f32_out`. Both are gfx1100,
rows=1, role/shape-gated siblings. Current pack8 singleton/pair primitives stay
registered for rows>1, gfx1151, registry or shape misses, shared-expert Q5,
explicit rollback, and any failed family gate. D12 introduces no repack,
sidecar, Q8_1 activation, post-op fusion, or tile16-style inter-output traffic
sharing.

This is not free traffic reduction. Pack8's `N/8` local128 workgroups and
D12's `N/2` local32 workgroups execute the same total physical-wave count and
the same encoded-weight/arithmetic work, but D12 rereads the small activation
row four times as often. The nominal weight-plus-activation source proxy grows
**80%** at every production shape. K6144/K9216 activations are only 12/18 KiB,
so cache may hide those reads while D12 removes 1,024 B LDS, every superblock
barrier, and the final cross-wave exchange; this is a hypothesis, not a waiver.
Both actual-weight shapes must win.

Admission is independent by family. The BF16-output key must be bit-exact and
faster for both `blk.0` K6144/N3072 and `blk.1` K9216/N3072. The F32 unequal-
pair key must be bit-exact and faster for both K3072 N6144+48 and N9216+72.
Matched current controls use 50 warmups, 15 counterbalanced repetitions, and at
least 200 launches/sample; both HIP-event and synchronized-wall medians must
improve. Cached tracing must show local32, zero LDS/scratch, expected `N/2`
workgroups, and preferably at most 96 VGPR. A failing family remains D9 even if
the other passes. Synthetic/adversarial reduction fixtures, production weights,
all 48 hidden rows, logits/argmax bits, complete K/V plus `KVLiveSpans`, reset,
and lifecycle precede any clean model run.

The implementation now clears that correctness admission. Two separately
registered gfx1100 role siblings keep raw 176-byte blocks and default-off
selectors; the standalone F32 singleton wrapper remains an unregistered oracle; rows>1, gfx1151, registry miss, unsupported layout/shape, and
selector disable retain pack8. Synthetic K256/K512 and all four production
shapes are finite-bit exact; non-finite fixtures match every defined bit and
NaN class. A shared-weight 69-token prompt plus 16 decode-step gate matches all
48 post-layer hidden rows, full logits/argmax, final/post-layer hidden, complete
K/V and `KVLiveSpans`, reset, and lifecycle. Cached tracing names the BF16
singleton and F32 unequal-pair candidates at local32, VGPR96, LDS0, and
scratch0 with exactly 1,536/4,644 workgroups. A deliberately sub-formal
5x100 actual-weight screen was positive for every leaf (**16.70-24.19%** HIP-
event contraction for attention output and **11.46-17.04%** for query/gate).
The subsequent formal 50-warmup/15x200 gate is bit-exact and improves all four
required leaves **13.63-24.80%** in HIP-event time and **10.39-23.73%** in
synchronized wall. Clean short/512/1K/near-4K profiles improve output/query-
gate **15.06-17.91%**, kernel sum **1.73-4.49%**, span **1.63-4.01%**, and
profiled-child throughput **1.44-5.25%**. Candidate resources are local32/
VGPR96/LDS0/scratch0 at unchanged **775 dispatches/token**.

Two complete canonical-suite process-order pairs remove the unaffected-prefill
order bias. Pooled h32 decode moves **47.046 -> 48.987 tok/s (+4.124%)** and
h32 E2E **11.997 -> 12.117 (+1.001%)**; every category improves both horizons,
prefill is **+0.016% aggregate** and within **-0.020% to +0.067%** by category,
and all ID/quality/state/KV/lifecycle gates pass. gfx1100 therefore defaults
both D12 roles while pack8 remains the explicit and unsupported-path fallback.
D12 is **20.414 ms/token**, leaving **0.414 ms / 2.068% throughput** to 50.
Evidence:
`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d12-q5-wave32x2-{correctness,retained}.json`.
Frozen design and source/traffic accounting:
`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d12-q5-wave32x2-design.json`.

### D13 raw-Q5 shared pair + SiLU (exact, selected)

The retained D12 traces leave one stable block-local post-op boundary that does
not require another global producer counter. In every sparse token, 46 Q5
shared-expert gate/up pairs are immediately followed by the separate BF16 SiLU
kernel. At short/512/1K/near-4K, the pair costs
**0.916/0.915/0.925/0.918 ms/token**, the targeted SiLU costs
**0.084/0.078/0.082/0.082 ms**, and the pair-to-SiLU submission gap costs
**0.175/0.175/0.174/0.175 ms**. The complete adjacent window is therefore
**1.175-1.182 ms/token** and is context-independent. Layer 47 uses a different
shared quant route and remains fallback.

D12's wave32x2 geometry does **not** transfer to this smaller K3072/N1024
shape. A 50-warmup, 15x300 counterbalanced actual-weight probe on layers 1 and
46 found two exact wave32x2 singleton launches **39.73%/39.79% slower in HIP
events** and **39.82%/39.87% slower in synchronized wall** than the retained
one-launch pack8 pair. D13 therefore preserves the pack8 arithmetic and rejects
another one-wave rewrite.

The selected leaf is
`linear_pair+activation/gguf_q5_k/pack8_gemv_decode_bf16_silu_bf16_out` on
gfx1100 c=1. One local256 block owns an eight-column output pack: threads
0..127 execute the current gate local128 schedule and threads 128..255 execute
the current up schedule. Each half retains the exact coefficient hoist,
`[t,t+128]` K order, four wave32 reductions, and serial wave 0..3 sum. Gate and
up round independently to BF16 before the existing
`g * sigmoid(g) * u` expression and final BF16 round. The leaf writes the
shared intermediate directly, removes **46 launches/token** (**775 -> 729**),
and avoids **376,832 bytes/token** of gate/up BF16 write-read traffic. The
registered Q5 pair plus separate SiLU remains the mandatory rows>1, gfx1151,
registry/shape-miss, explicit-rollback, and Q6 fallback.

The measured opportunity is material but is not yet a performance claim. If
the fused body pays the same SiLU arithmetic cost, removing only the measured
submission gaps saves **0.175 ms/token**, closes **42.30%** of D12's 0.414-ms
gap, and models **49.41 tok/s**. The strict zero-increment post-op ceiling is
**0.259 ms/token**, **62.54%** of the gap and **49.62 tok/s**. D13 cannot claim
50 tok/s by itself. Implementation requires BF16-bit synthetic and actual
layer-1/layer-46 parity, local256/VGPR<=96/LDS<=2048/scratch0, positive
inclusive event and wall micros for both layers, all-layer hidden/logit/KV/state
parity, improved clean traces at all four contexts, and the complete category
non-regression gate. Any failure removes the candidate rather than leaving a
default-off route. Design evidence:
`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d13-q5-shared-silu-design.json`.

## Laguna DFlash Follow-on Plan

DFlash work begins as architecture support during the target port but remains a
separate product/performance milestone. It must preserve exact target AR output;
acceptance affects economics, not output quality.

### D0 — Poolside schema and artifact support

Generalize `hipengine/loading/dflash.py` without breaking the existing z-lab
artifact:

- architecture-dispatched draft config parsing through a registry/plugin, not a
  hardcoded model-name branch;
- accept Poolside's nested `dflash_config.block_size=16` and
  `num_target_layers=48`;
- normalize zero-based target IDs and one-based post-layer depths;
- validate 69 BF16 tensors, including six auxiliary norms, fused QKV, and gates;
- expose zero-copy row views for Q/K/V within `(11264,3072)` when the dense
  kernels permit it;
- validate exact target pairing: vocab 100352, hidden 3072, layers 48, target
  GGUF hash, embedding/LM-head availability;
- support Poolside's BF16 GGUF metadata (`dflash.decoder_arch=laguna`, block
  size, capture layers) as a second container for the same logical plugin.

Acceptance: both safetensors and GGUF resolve to one normalized logical config;
wrong target, off-by-one captures, missing gate, duplicate root tables, and
unsupported layout fail before allocation.

The supported safetensors half of D0 is complete. The architecture registry
preserves the original `DFlashDraftModel` schema while normalizing
`DFlashLagunaForCausalLM` into one config with nested block/target fields,
zero-based target IDs and checked one-based capture depths, causal/SWA/gate/QKV
contracts, and the draft vocabulary. Validation consumes the exact local
69-tensor/2,229,955,584-byte payload, requires all six auxiliary norms and six
attention gates, and exposes non-owning Q/K/V row views over each fused BF16 QKV
allocation. Missing gates and off-by-one captures fail before allocation. The
public provider binds the exact target GGUF/source-cache hash, drafter blob hash,
and pinned revision before allocation. A BF16 DFlash GGUF container remains a
deferred, unsupported alternate input rather than an open B4 support blocker.

### D1 — Standalone Laguna drafter parity

Implement the draft forward independently of speculative control:

1. six target tap vectors -> auxiliary norms -> concat -> FC -> hidden norm;
2. new target-owned Q4_K embedding lookup for root and mask token rows;
3. six BF16 Laguna SWA layers with fused-QKV views and per-head softplus gate;
4. final norm;
5. target Q6_K LM head and compact top-k/argmax;
6. candidate IDs compared with Poolside llama.cpp for fixed target-tap fixtures.

Start with one draft layer and a short fixed block, then all six layers and block
16. Reuse the base Laguna attention/gate code rather than maintain a second
implementation. Keep the unfused CPU/reference chain.

Acceptance: layer hidden states, final hidden, logits/top-k, and all 15 possible
candidate positions meet the agreed oracle tolerances; gfx1151 trace shows the
intended BF16 dense/SWA/gate and Q6 LM-head kernels.

The correctness-first D1 path is resident and admitted through `B=4` as of
2026-07-23. `LagunaDFlashResidentDrafter` owns the 69-tensor BF16 drafter, six
projected-context K/V rings, bounded capture destinations, root/mask scratch,
and target-owned Q4 embedding/Q6 head. Target rows append transactionally;
root/mask K/V is visible to causal prefill attention but discarded rather than
committed. The query residual stays F32 around BF16 projections, while the exact
unfused per-tap normalization/concat and F32-context/F32-gate-to-BF16 softplus
paths preserve the existing fallback and `KVLiveSpans` ABI.

A Poolside intermediate callback exposed and fixed the decisive parity bug: the
first resident draft incorrectly passed BF16 Q/K norm vectors to Laguna's target
F32-weight norm+RoPE kernel. The drafter now uses its BF16-weight norm+RoPE ABI
and int32 query positions. Against a GGUF converted directly from pinned
`poolside/Laguna-S-2.1-DFlash@b0486d1` (the published GGUF payload is not
bit-identical to those published safetensors), `B=4` matches the Poolside root
and all 12 top-k IDs exactly. First-row Poolside-to-hipEngine KL is
`2.3420e-5`, top-1 is exact, teardown returns all `79,324,054,196` tracked bytes,
and a cached gfx1151 trace shows the intended six BF16 norm+RoPE/SWA/gate
layers plus target Q6 head. The proposal kernel span is `31.729 ms` (host wall
`32.030 ms`); this is correctness evidence, not yet an end-to-end speed claim.

`B=7` and `B=15` remain unadmitted rather than being hidden by the B4 result.
The full 15-row Poolside-tap diagnostic reaches only `12/15` top-1 with maximum
KL `0.05525`; live hipEngine target taps reach `11/15`. Two attempted precision
changes did not solve this: F32 projected context left candidate IDs unchanged,
and a full duplicated-F32 query path regressed first-row parity while adding
about 4.24 GiB and raising proposal wall to 311 ms, so both were removed. The
resident implementation is complete for the B1/B2/B4 product boundary, but
D1's stated all-15-row Poolside candidate-parity acceptance remains open. D3 now
proves target-cycle output/state exactness at higher budgets despite those weaker
proposals; B7/B15 remain diagnostic and blocked from D4 promotion. Evidence:
`benchmarks/results/2026-07-23-gfx1151-laguna-dflash-drafter-b4.json`.

### D2 — Target hidden and draft-context ownership

Wire the target runner to capture post-layer depths `2,11,20,30,39,48` during
prefill, AR, and verifier-shaped forwards. Hidden capture should be graph-
embedded or written during the producing layer, not implemented as six later
D2D copies.

Maintain bounded per-request rings:

- six normalized/raw target hidden streams as required by the projection ABI;
- append-only projected context;
- six draft-layer SWA K/V caches capped at 512;
- absolute positions independent of wrapped physical slots;
- rollback/reseed after accepted-prefix commit.

Acceptance: append-only cache equals full rebuild for crafted windows and 32+
cycles; 511/512/513 wrap is exact; rejected target rows never enter committed
draft context; target-only AR has no hidden-capture overhead when DFlash is off.

D2's online ownership boundary is implemented as of 2026-07-23. The target
writes six BF16 post-layer tap planes while producing each verifier row; the
cycle exposes only the accepted leading views to the drafter's projected-context
append. Draft and target cursors are checked after every cycle, fixed capture and
verifier addresses remain stable, and failures after either owner advances close
both owners rather than permit reuse. The verifier K/V transaction described
below is the rollback mechanism: rejected target taps and K/V never become
canonical. Target-only AR allocates neither verifier scratch nor capture planes;
those resources are lazy DFlash-cycle ownership.

### D3 — Laguna B+1 verifier, accept, and commit

Adapt the provider-neutral speculative cycle to Laguna target rows:

- root + B draft tokens execute as one exact verifier-shaped target forward;
- target attention and MoE use rows>1 kernels only after equivalence to serial
  c1 is proven;
- GPU argmax/accept summary avoids full-vocabulary D2H copies;
- accept walks until first mismatch and handles bonus token semantics exactly;
- commit only accepted hidden taps, 12 global KV families, 36 SWA rings, output
  IDs, and positions;
- rejected suffix rows leave no state/KV/ownership residue;
- cancellation and allocation cleanup are transactional.

Gate budgets in order `B=1,2,4,7`; test `B=15` only after B7 exactness and
memory pass. Compare every speculative output to a true no-DFlash AR run from
the same target session protocol.

Acceptance: reject/partial/full, EOS inside draft, max-token boundary, SWA wrap,
and multi-cycle sequences all match true AR IDs and committed state exactly.

D3 is implemented and passes its correctness ladder as of 2026-07-23. Laguna
stages each layer's verifier K/V in fixed F32 row planes (about 24 MiB at the
64-row target bucket) while attention consumes canonical prior K/V plus current
causal rows. The target does not append verifier rows during the 48-layer pass.
After row-wise GPU argmax, the shared DFlash accept kernel writes one seven-int
summary; only `root + accepted` staged rows are converted to BF16 and appended
to the 12 global and 36 SWA caches. This avoids snapshot/restore entirely and is
safe when a rejected suffix crosses ring slots 511/512/513. Payloads are checked
against the provider-neutral CPU chain oracle before commit. Stop-containing
proposals are truncated at the first stop and suppress the bonus when accepted;
remaining decode limits cover the max-token boundary. Stable bucket pointers are
asserted after every cycle, and request state can reset without reallocating
weights or scratch.

The same resident target was reset and compared against a true serial AR run for
12 generated IDs at `B=1,2,4,7,15`. Every budget produced exactly
`[94557,3505,3011,515,2407,365,2291,10723,1687,948,1482,4217]`; all committed
prefixes and target/drafter cursors matched. The ladder covered full, partial,
and zero acceptance: accepted/drafted totals were `5/6`, `7/8`, `8/12`, `8/21`,
and `8/45`, with 1/1/4/13/37 rejected target rows. Peak tracked allocation was
`79,358,606,181` bytes and teardown returned to zero bytes/allocations. B7/B15
here prove target correction/rollback correctness only; they do not override D1's
B4 Poolside candidate-parity admission or qualify those budgets for D4.

A cached B4 `rocprofv3 --kernel-trace` captured the intended zero-scratch path:
one 6.492-us `dflash_accept_chain_i32_kernel`, two row-argmax launches totaling
17.833 us, then exactly 12 global plus 36 SWA accepted-prefix writes totaling
91.212 us. The Q6 LM-head-to-final-KV-commit window was 9.818 ms device timeline
(5.658 ms kernel sum). Full commands, per-budget cycle shapes, address digests,
and lifecycle evidence are in
`benchmarks/results/2026-07-23-gfx1151-laguna-dflash-verify-commit.json`.
These timings are diagnostic, not D4 economics or a speedup claim.

### D4 — Full-suite economics

Only after D3:

- use all categories in
  `benchmarks/prompts/mtpbench-code-general-ja.jsonl` plus heldouts;
- use true no-DFlash AR from the same build/protocol as denominator;
- sweep fixed B values without prompt-conditioned choices;
- record proposals, accepted draft tokens, visible tokens, cycles, acceptance by
  depth/category, drafter wall, target verifier wall, commit wall, output tok/s,
  tracked allocation, and whole-device GTT;
- require exact generated IDs and state/KV on every row;
- retain a DFlash default only if aggregate decode is greater than 1.10x the
  same-protocol true-AR baseline and no category/heldout regression violates the
  predeclared policy.

Reported Poolside/vLLM or other-hardware DFlash speedups are context only. They
are not a gfx1151 or hipEngine baseline. A fixed-budget loss remains a valid
exact diagnostic and leaves AR as default.

D4 completed one admitted B4 decision on 2026-07-23 before LPF-1. One resident
target and pinned `b0486d1` BF16 drafter alternated true AR/DFlash over all ten
canonical prompts, two repetitions, and a fixed 32-output horizon. All 20 pairs
are exact and finite, both routes repeat deterministically, every
target/drafter cursor satisfies a valid fixed-horizon commit boundary, the
frozen Poolside first-token gate passes at KL `6.6214e-6` with exact top-1, and
tracked ownership returns to zero.

Those pre-LPF-1 economics rejected promotion:

| Scope | AR decode tok/s | DFlash B4 tok/s | Ratio | Draft acceptance | Target rows/output |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 16.388 | 10.715 | 0.6538x | 50.48% | 1.6935 |
| train | 16.445 | 11.855 | 0.7209x | 58.33% | 1.5323 |
| heldout | 16.303 | 9.365 | 0.5744x | 41.15% | 1.9355 |
| code | 16.564 | 14.522 | 0.8768x | 78.23% | 1.2500 |
| general English | 16.714 | 8.751 | 0.5236x | 36.54% | 2.0968 |
| general Japanese | 16.248 | 9.405 | 0.5788x | 40.63% | 1.9355 |
| mixed Japanese/English | 15.879 | 9.234 | 0.5815x | 39.58% | 1.9355 |

Across 210 cycles, proposal takes 6.721 s, target verification takes
**50.493/57.861 s (87.27%)**, and post-verify/commit residual takes 0.645 s.
Median TTFT also regresses `3.478 -> 4.764 s` because target AR uses bulk prefill
while DFlash's hidden-capture seed is still serial. The primary D4 decode blocker
is therefore excess verifier work plus insufficient non-code acceptance, not
accept/commit overhead. LPF-1 now routes every B+1 verifier with at least two
rows through the retained tile, so this table is historical evidence rather
than a current promotion decision. Refresh the complete D4 protocol only after
the remaining prefill candidates stabilize; do not extrapolate a new ratio from
projection microbenchmarks. AR remains default and D5 stays deferred. Artifact:
`benchmarks/results/2026-07-23-gfx1151-laguna-dflash-category-economics.json`.
The artifact explicitly records an offline repair to the derived fixed-horizon
state predicate using its exact raw cursors; no measurement value changed and
the complete >5-minute GPU run was not repeated under the focused-repair rule.

D4 is now confirmed on merged main `8f8baf9a1` after LPF-1/4/5. The clean
current-default run again uses all ten prompts, two repetitions, B4, and 32
visible outputs, with exact paired IDs, finite logits, deterministic repeats,
valid target/drafter cursors, the frozen Poolside gate, and exact lifecycle.
Current economics are:

| Scope | AR decode tok/s | DFlash B4 tok/s | Ratio | Draft acceptance | Target rows/output |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 16.384 | 15.527 | **0.9477x** | 50.48% | 1.6935 |
| train | 16.441 | 17.180 | 1.0450x | 58.33% | 1.5323 |
| heldout | 16.299 | 13.569 | 0.8325x | 41.15% | 1.9355 |
| code | 16.562 | 21.020 | **1.2692x** | 78.23% | 1.2500 |
| general English | 16.707 | 12.690 | 0.7595x | 36.54% | 2.0968 |
| general Japanese | 16.245 | 13.647 | 0.8401x | 40.63% | 1.9355 |
| mixed Japanese/English | 15.873 | 13.372 | 0.8424x | 39.58% | 1.9355 |

Weighted prefill is **50.389 tok/s AR** versus **16.906 tok/s DFlash** because
DFlash still seeds hidden captures through the serial path. LPF-1/5 reduce target
verification **50.493 -> 32.644 s (-35.35%)** and move full DFlash decode
**10.715 -> 15.527 tok/s (+44.91%)** without changing the 424/840 accepted
drafts. That is a major verifier win, but the same-session true AR denominator
remains **16.384 tok/s** and the required >1.10x full-suite gate still fails.
Median TTFT is **1.620 -> 4.767 s** and fixed-horizon E2E is **8.872 -> 4.503
output tok/s (0.5075x)**. Code benefits materially; heldout and every non-code
category regress. Keep AR default; D5 may expose this exact path only as an
explicit opt-in with no performance claim. Evidence:
`benchmarks/results/2026-07-23-gfx1151-laguna-dflash-current-main-confirmation.json`.

### D5 — Opt-in public/server integration

Expose DFlash through the speculative-provider registry after D3 exactness,
initially explicit-only. Capabilities must report target/drafter hashes, draft
budget, exactness mode, and fallback reason. Sampling/processors must share the
same processed target distribution; unsupported combinations fail closed. D4
is still mandatory before any automatic/default promotion.

The concrete D5 contract after the D4 rejection is:

- register the adapter by `(provider=dflash, target_model=laguna_gguf,
  backend=hip_gfx1151, quant=gguf_q4_k_m)` rather than adding a DFlash branch to
  `LLM`, server dispatch, or the base Laguna generator;
- configure the public owner explicitly with the pinned drafter path and B4.
  `LLM.generate()` and ordinary OpenAI requests remain target-only AR; a generic
  speculative method/request extension selects the provider and there is no
  automatic route;
- retain one target session, one 69-tensor drafter, and one fixed B+1 cycle under
  the model lock, resetting request state without reloading either weight owner.
  Close the cycle and drafter before the target weights and fail closed after a
  partial cross-owner error;
- bind the target to Q4_K_M SHA-256 `7da520c5...c5753f` through the validated
  repacked-cache manifest and bind the drafter to revision `b0486d1` plus
  safetensors SHA-256 `f24f0878...b62a1f4` before allocation;
- expose blocking and streaming generation with the same stop/EOS/max-token
  semantics and cumulative generated IDs as AR. Multi-token stops may discard a
  now-dead staged suffix because the request owner closes/resets immediately,
  but no suffix may be emitted or reused;
- admit only c=1 BF16-KV raw greedy target top-1 with no processed-logit or
  sampling modifiers. Explicit requests with temperature/top-p/top-k/min-p,
  penalties, logit bias/suppression, forced tokens, thinking/structured-output
  processors, custom EOS/ignore-EOS, logprobs, non-BF16 KV, another provider, or
  a budget other than B4 fail before model/drafter allocation;
- capability metadata reports provider, explicit-only policy, target and drafter
  identities, `candidate_budget=4`, `exactness_mode=target_corrected_greedy`,
  `processed_target_verification=false`, and the D4 fallback reason/evidence
  (`0.9469x` full-suite true AR); response telemetry reports cycle/accept/verify
  counts without claiming a throughput win;
- RED coverage must prove registry resolution, default-AR isolation, config/hash
  rejection, blocking/streaming equality, stop and output-limit boundaries,
  fail-closed sampling/provider/budget behavior, truthful capabilities, request
  route separation, cancellation/reset, and close ordering before the live
  ten-prompt gate.

D5 implementation status (2026-07-23): the library and OpenAI boundaries are
implemented. `LLM` resolves the four-axis provider, retains the source-bound
Laguna target plus pinned B4 drafter/cycle, and exposes provider-neutral
blocking/streaming detailed methods without changing `LLM.generate()`. The
server accepts `--speculative-provider`, `--draft-model`, and the fixed candidate
budget, advertises `sampling.speculative`, and routes only requests carrying the
generic `speculative` extension; the older `speculative_mtp` route remains
separate and mutually exclusive. Synthetic blocking/streaming, capability,
identity, reset/close, default-AR isolation, CLI/env, and fail-before-prepare
gates pass.

The live D5 gate now closes the final opt-in support criterion on gfx1151. Every
canonical prompt runs through true public AR, explicit OpenAI blocking DFlash,
and explicit OpenAI streaming DFlash at h32: all **10/10 + 10/10 + 10/10**
routes have exact cumulative IDs, blocking/streaming text agrees, and train,
heldout, all four categories, EOT-24 suppression, finish metadata, and fixed
oracle stop-policy checks pass. The same target/drafter/cycle owners survive all
requests while target position and drafter committed context reset to `-1/0`;
closing a public library stream after its first emitted chunk also resets both.
Final close releases the provider owners and returns **79,817,890,405 peak
tracked bytes / 1,883 peak allocations** to zero. All identity, revision, B4,
policy, streaming, exactness, and D4 fallback/no-performance-claim capabilities
are truthful. The complete correctness wall is **233.401 s** and is not a speed
comparison. Artifact:
`benchmarks/results/2026-07-23-gfx1151-laguna-dflash-public-e2e.json`.

D5 is therefore **supported as an explicit opt-in** for the pinned target and B4
drafter on gfx1151. AR remains default, and DFlash remains ineligible for
automatic or performance promotion because the current merged-main D4
full-suite ratio is `0.9477x` with heldout/non-code regressions.

Automatic routing is a later model-general policy. Never key route/budget to
known prompt IDs, benchmark categories, candidate token IDs, or fixed-suite
reranks. Any adaptive controller must use online, model-general economics and
pass the full train/heldout category gate.

## Test Matrix

| Layer | CPU deterministic | HIP primitive | Full-model eager | Bulk/graph | Long-context |
| --- | --- | --- | --- | --- | --- |
| GGUF metadata/tensor map | required | n/a | load audit | n/a | repeat load |
| tokenizer/template | required | n/a | direct/public parity | streaming parity | n/a |
| F16 projections | oracle | gfx1151 | layer taps | rows>1 | profile |
| dual RoPE | required | gfx1151 | full/SWA taps | prefill parity | 8K+/256K |
| paged global attention | required | gfx1151 | KV transitions | bulk/graph | 32K-256K |
| SWA attention/ring | required | gfx1151 | 511/512/513 | bulk/graph | wrap/reuse |
| softplus head gate | required | gfx1151 | layer taps | fused/unfused | n/a |
| sigmoid/correction router | required | gfx1151 | IDs/weights | batched | n/a |
| routed/shared experts | required | gfx1151 | hidden taps | batched | profile |
| LM head/sampler | oracle | gfx1151 | logits/IDs | graph/public | n/a |
| lifecycle/memory | plan | allocation smoke | load/run/close | graph close | repeated sessions |
| DFlash schema/tensors | required | load smoke | target-pair audit | n/a | repeat load |
| DFlash draft layers | required | gfx1151 | standalone candidates | block16 | SWA wrap |
| hidden taps/context cache | required | append/cache smoke | cycle parity | captured/bulk | 511/512/513 |
| verify/accept/commit | required | GPU summary/copy | exact AR IDs/state | B1/B2/B4/B7/B15 | repeated cycles |
| DFlash economics | n/a | n/a | same-session AR | full prompt suite | category heldouts |

GPU tests must explicitly skip when `libamdhip64.so` is unavailable.

## Risks and Mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| F16 weights silently converted to BF16 | cumulative logit drift | Preserve F16 first; compare any contraction explicitly. |
| YaRN implementation differs from Poolside/llama.cpp | early token divergence | CPU oracle from reviewed equations; first-layer Q/K fixtures. |
| SWA inherits global YaRN | incorrect local attention | Separate per-layer RoPE configs and tests. |
| correction bias leaks into routing weights | wrong MoE output | Separate score/probability buffers and adversarial fixture. |
| Q-head count assumed constant | bad shapes or memory corruption | Validate per-layer arrays and scratch widths. |
| SWA allocates full-context KV | 256K OOM | 512-slot physical ring plus absolute positions/live spans. |
| T16 layout assumes Qwen expert strides | wrong selected weights | Raw-byte round-trip and production-shape expert tests. |
| host repack transient exhausts UMA | load failure/stall | Tensor-at-a-time repack, prompt frees, GTT/host telemetry. |
| gfx1151 queue/graph stall | hung generation | One queue, eager first, cached builds, dedicated stall diagnostics. |
| llama.cpp PR evolves | moving oracle | Pin commit/binary hash and GGUF hash in artifacts. |
| 256K fits once but not repeatedly | false capacity claim | repeated load/run/close lifecycle gate. |
| chat parser passes simple text only | tool failures | separate frozen chat/thinking/tool fixtures. |
| DFlash capture index is off by one | low acceptance or wrong candidates | Normalize `[1,10,...]` to post-depth `[2,11,...]` once and fixture it. |
| old Qwen DFlash loader assumptions leak | missing QKV/gate/aux norms or wrong target roots | Architecture-specific draft plugin over shared provider ABI. |
| verifier B+1 rows cost more than accepted work | exact but slower than AR | Fixed-budget full-suite economics; AR remains default on loss. |
| DFlash increases load/lifecycle pressure | UMA exhaustion or later-run stall | Reserve 2.2-3.0 GiB plus verifier scratch and repeat load/run/close. |
| prompt-specific route tuning games the suite | invalid speed claim | No prompt IDs/categories/token reranks; full train+heldout gate. |

## Definition of Initial Support

Laguna S 2.1 Q4_K_M is initially **supported** only when all of the following
are true on gfx1151:

- completed GGUF hash and tensor audit are recorded;
- native resident 4K load succeeds without torch or CPU weight offload;
- tokenizer/template fixtures match the pinned oracle;
- CPU/reference primitive tests pass;
- new HIP kernels satisfy the repository correctness gate;
- full-model first-token logits meet KL <= 0.05 and top-1 >= 90%;
- deterministic multi-token output meets the declared oracle agreement gate,
  repeats exactly within hipEngine, and any free-running trajectory split is
  disclosed rather than represented as exact cross-runtime compatibility;
- direct, public blocking, and streaming paths agree;
- EOT, finish reasons, cancellation, KV ownership, and allocation cleanup pass;
- `rocprofv3` confirms intended gfx1151 kernel dispatch;
- limitations are reported explicitly in capabilities/docs.

This definition does not require a performance win, graph replay, 256K context,
INT8 KV, DFlash, or tool-call parity. Those are later milestones with their own
gates.

Matched DFlash is **supported as an opt-in** only when D0-D3 and D5 pass and
public blocking/streaming output is exact against true AR across the full suite.
DFlash becomes performance-eligible or default-eligible only after D4 exceeds
the repository's greater-than-1.10x same-protocol true-AR promotion gate on the
full suite, with all correctness/state/memory and category-heldout gates green.

### UD-Q2_K_XL direct resident path (gfx1100, 2026-07-23)

The pinned Unsloth `Laguna-S-2.1-UD-Q2_K_XL.gguf` path is now admitted for
direct `LagunaGGUFResidentSession` use on the W7900. Its 814 tensors remain in
source layouts: Q5/Q6/Q8 dense and shared projections, IQ2/IQ3 gate/up, IQ3/IQ4
down, Q5 embedding, and Q4 LM head all dispatch through four-axis keys. The
existing Q4_K_M F16/T16/pack8 path and unfused fallbacks remain intact. A
4-GiB explicit safety reserve is required on the 48-GB card; the generic
8-GiB default intentionally still rejects this model.

The independent Poolside llama.cpp `04b2b72c` gfx1100 oracle is frozen in
`tests/fixtures/laguna_poolside_q2_xl_v1_oracle.json` and its complete
100,352-way distribution. hipEngine matches first-token ID `94557` with KL
`0.000156823`, repeats its logits and 32-token trajectory exactly, and reaches
31/32 teacher-forced top-1 agreement. The free-running prefix is 24 tokens:
Poolside serial AR selects `4019` at step 25, while both hipEngine and a fresh
79-token Poolside teacher-forced prefill select `3062`; Poolside's own
teacher-forced margin is `0.1139603` log-probability. This split is disclosed
and is not represented as complete cross-runtime ID equality. All taps are
finite and tracked ownership returns to zero. Compact evidence:
`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-correctness.json`.

This is a direct resident correctness qualification, not yet a public API,
throughput, long-context, or DFlash promotion. Those require their own complete
category-suite and lifecycle artifacts below.

## Resolved Decisions and Deferred Extensions

1. A minimal concrete Laguna resident runner landed without first extracting a
   model-neutral transformer substrate; no backend/quant branch was added to
   engine dispatch.
2. Source F16 weights use FP32 accumulation with boundary-specific F32/BF16
   outputs. BF16 KV remains canonical; the FP16-KV bisection was worse and was
   removed.
3. Exact plain/YaRN tables are generated on the host and uploaded under a
   reusable Laguna RoPE owner.
4. `LagunaKVCache` owns admitted block-256 global pages and 512-slot SWA rings
   per layer; both expose complete `KVLiveSpans` with live counts, absolute
   positions, eviction masks, and current query position.
5. Both preformatted completion and blocking/streaming Poolside chat shipped;
   preformatted completion remains the independent debugging baseline.
6. The pinned safetensors snapshot is the canonical supported DFlash input.
   Poolside BF16 GGUF normalization is deferred.
7. Fused draft QKV uses one owning allocation with zero-copy Q/K/V row views;
   the loader does not split the 69-tensor artifact to mimic Qwen names.
8. Hidden taps are an optional Laguna resident-session capability activated by
   DFlash capture targets; ordinary AR does not allocate or copy them.
9. The target-cycle ladder is exact through B1/B2/B4/B7/B15, but B4 remains the
   highest candidate-parity-admitted product budget. B7/B15 target correction
   does not override the unresolved all-15-row Poolside candidate mismatch.
10. Contexts above 4K, c>1 admission, graph replay, BF16 DFlash GGUF, broader
    sampling/processors, and loader read/upload overlap are separate follow-on
    milestones. They do not inherit support from the c=1/4K evidence.

Architecture or phase changes that outgrow this document must also update
[`PLAN.md`](PLAN.md).
