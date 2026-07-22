# Laguna S 2.1 Q4_K_M and DFlash on gfx1151

Last updated: 2026-07-22

Status: foundation implementation in progress; no end-to-end Laguna runtime or
performance claim yet.

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
the Hugging Face LFS object. Poolside also publishes
`laguna-s-2.1-DFlash-BF16.gguf` at 2,233,764,000 bytes with LFS SHA-256
`24614292a4477f3ae5203c3875edcde0bc219f02616a9c9f65791e29b18a67ee`.

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

The tokenizer must preserve byte-BPE ID parity with the oracle. EOT 24 must stop
generation and its textual marker must not leak into returned content. The GGUF
contains a resolved chat template; use it rather than an unresolved Jinja
`include`. Thinking/no-thinking and tool-call behavior are later public-surface
gates, not assumptions made by the base completion path.

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

The checkpoint intentionally omits token embeddings and LM head. The drafter
must call target-owned Q4_K embedding and Q6_K LM-head/sampler primitives rather
than duplicate or dequantize those complete tables. hipEngine has Q6_K/Q8_0
embedding lookup today, not a Q4_K lookup; the base Laguna target port therefore
needs a registered Q4_K row-dequant/embedding kernel. Root/mask embedding rows
can then be materialized directly to BF16.

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
| Q4_K token embedding | only Q6_K/Q8_0 lookup is currently registered | Add Q4_K row-dequant lookup to BF16 for target and DFlash roots. |
| rank-3 selected experts | Q4/Q5/Q6 T16/raw selected kernels | Reuse for 256 experts/top-10; validate exact rank-3 strides. |
| Q6_K LM head | native Q6 T16/GEMV path | Reuse untied output map. |
| F16 projections | dense FP16 GEMV/WMMA kernels exist | Preserve F16 resident bytes; add Laguna projection plan. |
| F32 norms/router weights | dense F32 and RMSNorm support | Reuse; 3072-wide router needs its own exact launch gate. |
| head Q/K RMSNorm | existing Qwen full-attention primitives | Reuse with head dim 128 and variable Q-head counts. |
| paged attention/KV | `KVLiveSpans`, BF16 paged attention/write | Add per-layer global/SWA capacity and visibility. |
| standard RoPE | existing rotary kernels/tables | Add YaRN and dual per-layer config. |
| softplus attention gate | helper math exists in HIP source | Expose unfused primitive and optional fused multiply. |
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

Planning estimate: add roughly 2.2-3.0 GiB to the target-only resident session.
A short-context target+DFlash session and a single-request 256K target+DFlash
session remain capacity-plausible on the 120 GiB GTT host, but neither is
measured. DFlash admission must reserve verifier-shaped target scratch and must
not rely on target-only headroom.

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
- `LagunaGGUFTokenizer` loads the `gpt2`/`laguna` GGUF vocabulary, BOS 2, EOS 2,
  EOT 24, PAD 9, SEP 8, MASK 12, UNK 0, BOS insertion, stop IDs, and the raw chat
  template while suppressing EOT text under special-token skipping;
- five checked-in prompt fixtures plus CRLF, newline/punctuation, and combining-
  mark boundaries match Poolside's HF fast tokenizer at revision
  `179ee67cf0fff5391c67fe1a392ea849fa6d643f`; an expanded 23-case local
  comparison also matched, including atomic chat/control tokens;
- rendered chat-template fixtures and the authoritative Poolside llama.cpp token
  oracle remain pending, so HF parity is not yet the full L0 acceptance gate.

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
  resident-weight allocation leaked. L3 is closed for target weight residency,
  while KV/scratch ownership moves to the eager-session gate.

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
| Q4_K token embedding | Q6_K/Q8_0 raw lookup is registered and the Laguna Q4_K table remains raw | No `embedding/gguf_q4_k/lookup_bf16_out` key/body. Add row-dequant, invalid-ID checks, exact D2H comparison, registry resolve, and a profiler-visible gfx1151 launch. |
| F32 RMSNorm / residual | GGUF BF16-input/F32-weight RMSNorm and add-RMSNorm are reusable | Wire under Laguna keys and prove layer-0 residual order; no new math is implied. |
| F16 Q/K/V/gate/O projections | Source precision and pointers are resident; dense FP16 kernels handle FP16 activation+weight | Laguna needs BF16/F32 activation with F16 weight and FP32/lowp output. Neither mixed variant is registered. Add single/dual/triple projection primitives and retain an unfused fallback. |
| Q/K head norm and RoPE | The existing FP32-input/F32-weight head-norm+partial-rotate body accepts variable head counts/dimensions | Current table helper implements plain RoPE only. Add exact YaRN tables for full layers (partial 64) and plain SWA tables (full 128), absolute-position tests, and 48/72 Q-head gfx1151 coverage. |
| Global BF16 KV/attention | Uniform block-256 `KVLiveSpans` write/context attention accepts GQA ratios 6 and 9 and head dim 128 | Revalidate at Laguna shapes and ensure the ungated context path feeds softplus rather than a Qwen sigmoid gate. |
| 512-token SWA | Capacity is planned as 36 bounded rings | Current wrappers require `spans_mode="uniform"` and parent attention consumes only page table + live count; it cannot represent a token-granular wrapped window with absolute positions. Add a real `KVLiveSpans` SWA writer/reader using token positions/eviction and 511/512/513 boundary tests. |
| Per-head attention gate | CPU FP32 softplus oracle exists | No gfx1151 softplus-broadcast key exists. Add unfused FP32 softplus+head broadcast before O projection, then consider fusion only after exact parity. |
| Dense layer-0 MLP | Q4_K pack8 gate/up, raw Q6_K down, SiLU, and residual primitives are reusable | Add production-shape layer-0 vertical fixture and trace. |
| Router projection | BF16 hidden × F32 router weight → FP32 logits is registered (gfx1151 256-thread override) | Existing selection is raw-logit top-k followed by softmax. Laguna requires `sigmoid(logits)`, correction bias for selection only, stable top-10, gather of unbiased probabilities, sum normalization, and 2.5 scaling. A separate kernel/key is mandatory. |
| Routed experts | Q4_K/Q6_K rank-3 T16 selected GEMVs support arbitrary positive expert count and top-k-shaped rows | Registration is lazy through the Qwen runner and no 256-expert/top-10/3072×1024 Laguna gate exists. Add direct Laguna plan resolution, exact production-shape raw-byte tests, and rocprof evidence. |
| Shared expert | Rank-2 Q4_K pack8 gate/up plus raw Q6_K down are reusable | Wire always-on SiLU shared output and add it independently of routed scale; do not reuse Qwen's sigmoid shared-gate combine semantics. |
| Final norm / Q6_K LM head | F32-weight RMSNorm, raw Q6_K BF16→F32 linear, GPU argmax, and sampler primitives are registered | Add exact root probes, then preserve full logits for KL/top-1 oracle gates before using direct top-1 shortcuts. |
| Session and hidden taps | Model map/resident weights expose every layer and metadata | No `laguna_gguf_runner.py`, KV/state owner, scratch plan, post-layer capture ABI, or eager step exists. Build c=1/token-serial first and capture depths 2/11/20/30/39/48 only on request. |
| Public generator | Generic engine loop and server lifecycle exist | Built-ins register only Qwen paths; there is no Laguna generation key, tokenizer/template renderer, streaming owner, or model metadata route. |
| Reasoning/tools | Generic server understands Qwen `<think>` plus JSON-in-`<tool_call>` | S 2.1 uses Poolside XML arguments: `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`, and reasoning history must stop its backward scan at the current `<assistant>` token. Implement the `poolside_v1` contracts from vLLM [`61c9ef98`](https://github.com/vllm-project/vllm/blob/61c9ef986a807aa3b9c6ccd25bb223b8f4116ac7/vllm/tool_parsers/poolside_v1_tool_parser.py#L48-L220) and its [assistant-scoped reasoning parser](https://github.com/vllm-project/vllm/blob/61c9ef986a807aa3b9c6ccd25bb223b8f4116ac7/vllm/reasoning/poolside_v1_reasoning_parser.py#L33-L69), including newline-less calls and incremental string values. |
| Independent oracle | Poolside branch and exact source commit are identified remotely | No local Poolside llama.cpp build, rendered template fixtures, first-token logits, or deterministic target IDs exist yet. This remains the first execution dependency, not a post-hoc check. |

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
  -> exact target AR benchmark and default promotion
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
- mixed Q4 gate/up and Q6 down layouts;
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

### L8 — Bulk prefill and graph replay

Bulk prefill comes after eager parity:

- F16 Q/K/V/gate/o rows>1 projections;
- full-attention prefill at 48 Q heads;
- SWA prefill at 72 Q heads with bounded history;
- batched sigmoid/correction/top-10 routing;
- selected routed experts and shared expert over prompt rows;
- chunked prefill sized from gfx1151 profiles, initially testing 256 rows;
- exact final prompt hidden/KV comparison against token-serial eager.

Graph replay comes last. Capture keys must include all state that can change
semantics: context bucket, layer attention pattern, live spans, absolute
position, KV addresses/capacities, scratch addresses, active width, and sampler
mode. Existing Qwen/PARO graph admission does not automatically certify Laguna.

Acceptance:

- bulk versus serial final hidden, all 12 global KV families, and all 36 SWA
  ring states pass the correctness gate;
- 512/1K/4K prefill is exact before performance tuning;
- graph/eager generated IDs and every hidden/KV transition match over the full
  measured horizon;
- graph capture/instantiate time is excluded from decode throughput;
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

Initial performance questions, in order:

- Are F16 attention projections or selected experts the short-context c=1
  bandwidth leader?
- Does source-preserving F16 dense GEMV reach the expected gfx1151 bandwidth?
- Is top-10 selected Q4/Q6 execution coalesced at expert width 1024?
- Does a 256-row prefill chunk remain best for Laguna's all-attention model?
- At what context does global paged attention overtake weight traffic?
- Does SWA ring management add host synchronization or excess copies?
- Does graph replay amortize at the actual Laguna output horizon?

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

### D5 — Opt-in public/server integration

Expose DFlash through the speculative-provider registry after D3 exactness,
initially explicit-only. Capabilities must report target/drafter hashes, draft
budget, exactness mode, and fallback reason. Sampling/processors must share the
same processed target distribution; unsupported combinations fail closed. D4
is still mandatory before any automatic/default promotion.

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
- deterministic multi-token greedy output matches the pinned oracle;
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

## Open Decisions Before Coding

1. Whether to extract a model-neutral GGUF resident transformer substrate before
   adding Laguna or to land a minimal Laguna runner and immediately schedule the
   extraction in `REFACTOR.md`.
2. Whether source-preserving F16 projections should produce BF16 or FP16
   intermediates at each boundary; decide from a tiny exact layer gate, not
   convenience.
3. Whether YaRN tables are generated on host and uploaded or computed in a
   reusable rotary kernel. Prefer the simplest exact unfused path first.
4. How the KV owner represents different physical capacities for global and SWA
   layers while preserving `KVLiveSpans`.
5. Which pinned Poolside/llama.cpp commit and command will be the external
   oracle after the download completes.
6. The first public support boundary: preformatted completion only, or blocking
   chat in the same milestone. Preformatted completion should remain the
   debugging baseline even if chat ships simultaneously.
7. Whether the safetensors or Poolside BF16 GGUF is the canonical DFlash input.
   Normalize both logically, but choose one first implementation artifact.
8. Whether fused QKV is represented as zero-copy row views or a dedicated dense
   kernel; do not repack/split 69 tensors merely to satisfy old Qwen names.
9. Whether target hidden taps become an optional general resident-session ABI or
   remain a Laguna+DFlash capability. They must be inactive at zero cost in AR.
10. The first DFlash budget. Poolside's serving example uses seven; correctness
    should still progress through B1/B2/B4 before B7 and B15.

Resolve these decisions in `WORKLOG.md` as implementation begins. Architecture
or phase changes that outgrow this document must also update [`PLAN.md`](PLAN.md).
