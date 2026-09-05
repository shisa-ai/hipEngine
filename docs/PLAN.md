# hipEngine — Purpose-Built Inference for AMD RDNA3

> **Status:** Design document — architecture and roadmap for a clean-host inference engine built around proven gfx1100 kernels.

## What hipEngine Is

hipEngine is a local LLM inference engine designed from the ground up for AMD RDNA3 GPUs (gfx1100, W7900-class). It pairs a minimal, purpose-built Python host with a complete suite of hand-tuned HIP kernels developed through 100+ iterations of profiling and optimization on real W7900 hardware.

The name signals exactly what we optimize for: **HIP** (AMD's GPU compute platform) as a first-class target, not a CUDA port or afterthought.

## Why hipEngine Exists

Existing inference engines fall into two categories that both fail the W7900 user:

1. **CUDA-first engines** (vLLM, ExLlamaV3, TensorRT-LLM) — treat AMD as a second-class port, disable their best features on ROCm, or don't support it at all.
2. **Generic PyTorch engines** (nano-vllm, HF Transformers) — run on ROCm but leave massive performance on the table because they never replace PyTorch's generic kernels with architecture-specific ones.

hipEngine occupies the gap: **a ROCm-native engine where every hot path has been profiled and replaced with a gfx1100-optimized kernel**, while maintaining the API compatibility and server features users expect.

## References & Lineage

hipEngine is informed by a lineage of inference engines with different strengths. We characterize them by **lines of code** as a proxy for complexity — our goal is a host layer orders of magnitude smaller than production engines, paired with a kernel layer that rivals their performance on AMD hardware.

All numbers marked ✓ were measured directly against the checked-out source in this workspace (`wc -l`, with embedded HIP source strings in Python files counted separately). Numbers marked (unverified) are from upstream reports; we have not audited them.

| Engine | Host LoC | Kernel LoC | Total | Language | What We Learned |
|--------|----------|------------|-------|----------|-----------------|
| **nano-vllm** (`rocm` branch) ✓ | 1,629 | ~20 (1 Triton: `store_kvcache_kernel`; paged attn uses torch SDPA) | ~1,650 | Python | Clean scheduler/engine separation; `torch.compile` discipline; pure-PyTorch ROCm compatibility |
| **mini-sglang** (`rocm` branch) ✓ | 9,908 Python (incl. ~1,100 kernel wrappers) | 520 HIP (`hip_expert_smoke.hip`) + 193 Triton (`fused_moe.py`) + ~1,800 C++/CUDA infra (nccl227.h, tensor.h, utils.cuh) | ~12,400 | Python + HIP/C++ | Production server (FastAPI/ZMQ); RadixCache prefix caching; W8A8 dynamic quant; MoE model definitions; overlap scheduling |
| **ds4** (antirez) | ~18,000 (unverified) | ~3,000 Metal + CUDA (unverified) | ~21,000 | C | Single-file C engine; GGUF mmap loading; session-based KV cache with save/restore; MTP speculative decode; "thinking modes" for reasoning; Metal graph capture; minimal host complexity with maximal kernel density |
| **hipfire** | ~6,000 (unverified) | ~4,000 HIP (unverified) | ~10,000 | C++ / HIP | gfx1100 HFQ4 GEMV with `__launch_bounds__(32,16)`; 32-thread workgroups; packed uint32_t nibble loads; fused RMSNorm+MQ+RoPE; attention_flash_asym3_tile; kv_fold_asym3; boundary fusion thinking |
| **llama.cpp** Qwen slice ✓ | ~60k (25k `src/` + 34k `common/`) | ~62k (25k ggml core + 37k ggml-cuda, reused by ggml-hip via hipify) | ~122k | C/C++ | Vulkan beats HIP on W7900 due to workgroup shape (64-thread wave64 vs 256-thread 8-wave32); `sudot4` mixed-signed dot4; coalesced Q8_1_x4 activation loads; graph-level fusion; `-amdgpu-unroll-threshold-local=600` compiler flag |
| **llama.cpp** total ✓ | ~104k (70k `src/` + 34k `common/`) | ~215k (cpu 75k, cuda 37k, vulkan 31k, sycl 31k, metal 20k, hexagon 21k, opencl 15k, others ~25k) | ~320k | C/C++ | (same) |
| **vLLM** | ~50,000 (unverified) | ~15,000 CUDA + Triton (unverified) | ~65,000 | Python/C++ | Continuous batching; PagedAttention; FlashAttention integration; production serving features; **CUDA-only kernel layer** |
| **ExLlamaV3** | ~8,000 (unverified) | ~25,000 CUDA PTX-heavy (unverified) | ~33,000 | Python/C++/CUDA | EXL3 quantization (QTIP-based); Marlin-inspired GEMM; persistent cooperative-group kernels; **PTX intrinsics make ROCm port ~2-4 weeks for GEMM alone** |

### What Each Reference Taught Us

**nano-vllm** — The ~1,600-line host is the right order of magnitude. Its scheduler (prefill/decode alternation), block manager (paged KV), and model runner separation are sound architecture. We keep the *shape* of these components, not the code, because our kernel dispatch model is different.

**mini-sglang** — The FastAPI/ZMQ server is production-ready and directly portable. RadixCache is algorithmically valuable for prefix sharing. W8A8 quantization math is correct. MoE model definitions (Qwen3 MoE, Qwen3.5 MoE, at 530 + 805 lines respectively) save us from writing them. But the ~9,900-line host carries overlap scheduling, pynccl distributed, and CUDA graph assumptions we don't share.

**ds4** (antirez) — The most instructive reference for host design. ~18,000 lines of C for a complete DeepSeek V4 Flash engine with Metal and CUDA backends. Key lessons:
- **Session-based KV cache**: `ds4_session_sync()` reuses prefix state; `ds4_session_save_payload()` / `load_payload()` for disk persistence. This is a richer KV lifecycle than our current paged-only model.
- **MTP speculative decode**: Built-in multi-token prediction with `mtp_draft_tokens` and `mtp_margin`.
- **Thinking modes**: `DS4_THINK_NONE` / `HIGH` / `MAX` with reasoning effort prefixes injected at the prompt level.
- **GGUF mmap loading**: Zero-copy weight loading with kernel page cache.
- **Metal graph capture**: Full model graph capture for zero-launch-overhead inference.
- **Single-file vertical design**: `ds4.c` owns everything — loader, CPU kernels, Metal driver, tokenizer. This density is the opposite of our layered approach; both are valid depending on goals.

**hipfire** — Another gfx1100-focused engine. Corroborates our kernel principles independently:
- 32-thread workgroups with `__launch_bounds__(32, 16)` for GEMV
- Packed `uint32_t` nibble loads for Q4
- Four independent FP32 accumulators for ILP
- Fused `rmsnorm_mq_rotate` at layer boundaries
- `attention_flash_asym3_tile` and `kv_fold_asym3` as future templates for streaming attention / KV quant

**llama.cpp** — The Vulkan vs HIP comparison on W7900 was our most valuable reference analysis:
- Vulkan uses 64-thread wave64 single-row kernels with subgroup reduction
- HIP uses 256-thread 8-wave32 blocks with LDS/barrier reduction
- For small-K expert-down matvecs (ncols=512), the HIP shape wastes most threads
- RADV/ACO schedules shaders better than ROCm LLVM-AMDGPU for this shape
- The `-amdgpu-unroll-threshold-local=600` flag makes HIP prefill much faster
- Graph-level fusion matters: llama.cpp HIP had ~1600 dispatches/token vs Vulkan's fewer

**vLLM** — The feature set is the target (continuous batching, PagedAttention, production serving) but the implementation is CUDA-only and ~65,000 lines. Not directly portable; we match features selectively.

**ExLlamaV3** — The EXL3 quantization format is interesting but the ~25,000-line CUDA kernel layer with heavy PTX (`mma.sync.aligned`, `cp.async`, `ldmatrix`) makes ROCm support a large project (4-8 weeks estimated). We defer EXL3 support.

## Multi-GPU Strategy

Our kernel layer is single-GPU by design. Multi-GPU support is a **host concern** that does not require kernel rewrites. Here's the strategy:

### Tensor Parallelism (TP) — Default Path

| Aspect | Approach | Rationale |
|--------|----------|-----------|
| Sharding | Column-parallel for QKV/gate_up, row-parallel for o_proj/down | Standard TP, minimizes communication |
| Communication | `rccl` (ROCm NCCL) via `ctypes` on `librccl.so`, or MPI via `mpi4py` | Torch-free. `[distributed]` extra wires in `torch.distributed` for users who want it |
| KV cache | Replicated per GPU | Simpler than sharded KV; memory scales with GPUs |
| All-reduce points | After o_proj, after down_proj, after shared expert | Minimal: 2-3 all-reduces per layer |
| Process model | Single-process multi-GPU preferred; multiprocessing fallback | PyTorch `cuda:0`, `cuda:1` in one process if possible |

**Kernel impact: None.** Kernels see their local shard. The host stitches results.

### Pipeline Parallelism (PP) — For Very Large Models

| Aspect | Approach | Rationale |
|--------|----------|-----------|
| Layer sharding | Assign contiguous layer ranges to GPUs | Simpler than interleaved |
| Communication | P2P tensor transfer between stages | `hipIpcMemHandle` + `rccl` point-to-point, or MPI sendrecv |
| Bubble | Micro-batching to hide pipeline bubbles | Standard GPipe/PipeDream approach |
| KV cache | Each GPU holds its layer range's KV | Natural with layer sharding |

**Kernel impact: None.** Kernels run on their assigned layers.

### Expert Parallelism (EP) — For MoE Models

| Aspect | Approach | Rationale |
|--------|----------|-----------|
| Expert sharding | Distribute experts across GPUs | Each GPU holds subset of experts |
| All-to-all | `all_gather` for expert outputs | Needed when experts span GPUs |
| Router | Replicated on all GPUs | Small, router decision is local |
| Shared expert | Replicated or assigned to one GPU | Depends on size |

**Kernel impact: Minimal.** The `w8a16_gate_up_shared_t_decode_v2_kernel` already handles shared+selected experts. EP adds an all-to-all after expert dispatch.

### What We Don't Do (Yet)

| Approach | Why Deferred |
|----------|-------------|
| **ZeRO-style parameter sharding** | Adds complexity for marginal gain on 2-4 GPU consumer setups |
| **Sequence parallelism (SP)** | Not needed until context lengths exceed single-GPU KV capacity |
| **NVLink-optimized collectives** | No NVLink on consumer AMD; PCIe is the bottleneck |
| **pynccl custom communicators** | mini-sglang uses this; hipEngine uses `rccl` via ctypes (torch-free). Adding pynccl would require torch as a hard dep |

### Minimal Viable Multi-GPU

The smallest useful multi-GPU path for hipEngine:

```python
# hipengine/distributed/tp.py
class TensorParallelConfig:
    world_size: int = 2
    rank: int = 0
    # Column-parallel shards
    qkv_shard: int   # total_heads // world_size
    gate_up_shard: int  # intermediate // world_size
    # Row-parallel input
    o_proj_shard: int   # hidden // world_size
    down_shard: int     # intermediate // world_size

class TensorParallelEngine:
    def __init__(self, model_spec, tp_config):
        self.models = []
        for rank in range(tp_config.world_size):
            core.device.set_device(rank)
            model = build_sharded_model(model_spec, rank, tp_config)
            self.models.append(model)
        
    def forward(self, batch):
        # Run each shard
        outputs = []
        for rank, model in enumerate(self.models):
            core.device.set_device(rank)
            out = model.forward(batch)
            outputs.append(out)
        
        # All-reduce at row-parallel boundaries
        for reduce_point in ["o_proj", "down_proj", "shared_expert"]:
            tensor = gather_outputs(outputs, reduce_point)
            _rccl.all_reduce(tensor)  # via librccl.so ctypes binding
            scatter_outputs(outputs, tensor)
        
        return outputs[0]  # rank 0 has final result
```

**Implementation effort: ~200 lines of host code.** No kernel changes. No new communication library. Just PyTorch `distributed`.

### Roadmap

| Phase | Multi-GPU Feature | Effort |
|-------|-------------------|--------|
| Phase 3 (Week 4) | Basic TP-2 for dense models | ~2 days |
| Phase 5 (Ongoing) | TP-2/4 for MoE models | ~3 days |
| Phase 5 (Ongoing) | PP for models exceeding single-GPU memory | ~1 week |
| Phase 5 (Ongoing) | EP for MoE models with many experts | ~1 week |
| Future | Sequence parallelism for 256K+ contexts | Research |

### Key Insight

**Multi-GPU is a host scheduling problem, not a kernel problem.** Our kernels are already efficient on single GPU. The host just needs to:
1. Shard weights at load time
2. Launch kernels on the right GPU
3. Insert `all_reduce` at the right boundaries
4. Replicate or partition KV cache

This is why we can defer multi-GPU without architectural risk. The kernel layer doesn't need to know about it.

### Multi-GPU Roadmap (LoC)

| Feature | New LoC | What It Does |
|---------|---------|--------------|
| **TP-2 dense** | ~150 | Single-process 2-GPU, `rccl` all-reduce, weight sharding |
| **TP-2/4 MoE** | +150 | Expert sharding awareness, replicated router |
| **Pipeline Parallelism** | ~200 | Layer-range assignment, P2P tensor transfer |
| **Expert Parallelism** | ~250 | All-to-all for expert outputs across GPUs |
| **Sequence Parallelism** | ~400 | Context sharding for 256K+ (research) |

**Key invariant:** Zero kernel changes. All multi-GPU is host weight sharding + communication.

## Tiered Memory & Offloading

hipEngine treats memory as a hierarchy of tiers with async migration, not a single GPU buffer. This can run models and contexts beyond single-GPU memory without changing request lifecycle or scheduling. Hot/cold codecs may still require registered transform, restore, or attention kernels; “tiering” is not permission to hide format work.

### TieredTensor Abstraction

```python
# hipengine/memory/tiers.py
class MemoryTier(Enum):
    DEVICE = auto()      # GPU HBM (24 GiB on W7900)
    HOST_PINNED = auto() # CPU pinned memory (fast DMA)
    HOST = auto()        # CPU regular memory
    DISK = auto()       # NVMe/SATA SSD

@dataclass
class TieredTensor:
    """Tensor that may live on any tier, with async migration."""
    shape: tuple
    dtype: DType                                  # hipengine.core.dtype
    tier: MemoryTier
    data: hipengine.Tensor | mmap.mmap | None
    
    def to(self, tier: MemoryTier, stream=None) -> TieredTensor:
        """Async migrate. Returns immediately, copy in background."""
        ...
    
    def ensure_ready(self) -> hipengine.Tensor:
        """Block until data is on DEVICE and ready for kernels."""
        ...
```

### KV Cache: 3-Layer GPU-CPU-Disk (ktransformers-style)

```python
# hipengine/kvcache/tiering.py
class KVColdTier(Protocol):
    """Optional component of one resolved KVCacheBackend."""

    def plan_pools(self, hot_spec: KVBackendSpec) -> KVPoolPlan: ...
    def estimate_store(self, snapshot: KVSnapshotHandle) -> ResourceClaimSet: ...
    def store(self, snapshot: KVSnapshotHandle) -> MaintenanceWork: ...
    def estimate_restore(self, object_id: str) -> ResourceClaimSet: ...
    def restore(self, object_id: str, hot_backend: KVCacheBackend) -> MaintenanceWork: ...
    def evict(self, pressure: ResourceClaimSet) -> list[str]: ...
```

| Tier | Latency | Bandwidth | Use Case |
|------|---------|-----------|----------|
| **Device (HBM)** | ~1 μs | ~1 TB/s | Active decode, prefill, hot KV |
| **Host Pinned** | ~10 μs | ~16 GB/s (PCIe4) | Warm KV prefix, prefetch target |
| **Host Regular** | ~100 μs | ~50 GB/s (DRAM) | Cold weights, CPU fallback |
| **Disk (NVMe)** | ~100 μs | ~7 GB/s | Cold KV, session persistence |
| **Disk (SATA)** | ~1 ms | ~500 MB/s | Archive, very cold sessions |

### Weight Offloading: Hot/Cold Layer Assignment

```python
# hipengine/models/tiered_model.py
class TieredModel:
    def __init__(self, spec: ModelSpec, tier_config: TierConfig):
        # Hot layers (early, frequently used) on device
        # Cold layers (late, rarely used) on host
        for i in range(spec.num_layers):
            tier = tier_config.layer_tier(i)
            self.layers.append(TieredLayer(spec, i, tier))
        
        # MoE: hot experts on device, cold on host
        if spec.num_experts:
            self.expert_tiers = ExpertTierManager(
                hot_experts=tier_config.hot_expert_count,
                device=device, host=host,
            )
```

### Session Persistence (ds4-style)

```python
# hipengine/session/persistence.py
class SessionPersistence:
    """Save/restore full inference state including KV cache.
    Enables: resume conversations, server restart recovery,
    multi-session switching without re-computing prefixes."""
    
    def save(self, session, path) -> SessionSnapshot:
        # Serialize: prefix tokens, KV payload, sampling state
        # Compress KV, write atomically
        ...
    
    def load(self, snapshot, engine) -> Session:
        # Restore KV from disk to appropriate tier
        # Fast path: if prefix in cache, skip recompute
        ...
    
    def sync(self, session, prompt_tokens):
        # ds4_session_sync equivalent:
        # Reuse common prefix from checkpoint, only evaluate suffix
        common = longest_common_prefix(session.tokens, prompt_tokens)
        if common > 0:
            session.rewind(common)
            session.extend(prompt_tokens[common:])
        else:
            session.rebuild(prompt_tokens)
```

### MoE Expert Tiering (ktransformers-style)

For 256-expert models where only 6-8 are active per token:

```python
# hipengine/moe/tiered_experts.py
class TieredExpertManager:
    """Hot experts (frequently activated) stay on device.
    Cold experts live on host, fetched on-demand."""
    
    def forward(self, hidden_states, selected_experts):
        device_experts = [e for e in selected_experts 
                         if self.experts[e].tier == DEVICE]
        host_experts = [e for e in selected_experts 
                       if self.experts[e].tier == HOST]
        
        # Device path: native fused kernel
        out_device = native_fused_moe(hidden_states, device_experts)
        
        # Host path: CPU kernel or async prefetch+GPU
        if host_experts:
            out_host = cpu_moe_kernel(hidden_states, host_experts)
            # Or: prefetch then native_fused_moe
        
        return combine(out_device, out_host)
```

### Integration with `KVCacheBackend`

```python
# Registry factories validate complete compositions before engine startup.
backend = KVBackendRegistry.resolve(
    topology="paged_dense", hot_codec="bf16", tier="device_only")
backend = KVBackendRegistry.resolve(
    topology="paged_dense", hot_codec="int8_per_token_head", tier="host_lru")
backend = KVBackendRegistry.resolve(
    topology="dms_compact", hot_codec="fp8", tier="kvtc_cold")
```

One resolved composition owns one global hot pool set plus optional cold pools.
DMS is a retention topology; KVTC is a cold codec. Neither owns a second
scheduler.

### Why No Scheduler Changes

The kernel layer sees raw device pointers through the hot backend's
`KVStorageView`. The common scheduler reserves transfer/restore claims and
schedules maintenance work before a request becomes ready. Lossless byte
migration may reuse the same attention kernels; compression, reconstruction,
or mixed-tier attention requires separately registered codec kernels. Async
prefetch can hide latency, but promotion requires measured TTFT/ITL behavior.

### Offloading Roadmap (LoC)

| Feature | New LoC | What It Does |
|---------|---------|--------------|
| `device_only` default | 0 | Current behavior, everything on GPU |
| Host pinning + prefetch | ~200 | `TieredTensor.to(HOST_PINNED)`, async streams |
| Disk spillover | ~200 | NVMe/SATA KV block storage, mmap |
| DMS per-head/layer | ~300 | Importance scoring, selective eviction |
| Expert CPU offload | ~300 | ktransformers-style hot/cold expert tiers |
| Session save/restore | ~150 | ds4-style full state serialization |
| NVMe direct storage | ~400 | Research: bypass page cache for KV |

## Core Principles

| Principle | Meaning |
|-----------|---------|
| **HIP-first, not CUDA-ported** | Every kernel is written for gfx1100/RDNA3 wave32 defaults, vec8 FMA patterns, and cache hierarchy. No PTX, no `cp.async`, no tensor-core assumptions. |
| **Multi-backend from day one** | The kernel tree is parameterized by target (`hip_gfx1100`, `hip_gfx1151`, `cuda_sm86`, `cpu_reference`). Adding a backend adds a sibling directory and registry entries — no engine rewrites. CUDA, Strix Halo, and future hardware are peers of gfx1100, not ports. |
| **Clean host, proven kernels** | The Python host is ~700 lines of purpose-built scheduling and dispatch. The kernel layer is ~18,600 lines of proven, profiled HIP + C++ bindings (120 `__global__` kernels) from the nano-vllm-amd research lineage. Kernel bodies take raw device pointers — torch-independent — so only the host-side launch wrappers change when retargeting to a new backend. |
| **Torch-free at runtime** | hipEngine does not import `torch` at inference time. We own a thin `hipengine.Tensor` over HIP/CUDA device pointers, call `hipblasLt` / `hipGraph` / loading libs via `ctypes`, and JIT kernels with `hipcc` + `ctypes.CDLL` (no `torch.utils.cpp_extension`). This removes a 1.7 GiB dependency. Optional `hipengine[torch]` extra exposes dlpack interop for users who want to hand in torch tensors. |
| **Fast dispatch, no Python in the hot path** | Decode forward is captured at warmup and replayed with zero Python kernel-dispatch overhead per subsequent step. Native `hipGraph` remains the default; an explicit, architecture-gated in-tree retained-PM4 transport may lower admitted kernel-only graphs to one ROCr submission. Python runs only once per token for sampling. |
| **Fused + unfused kernels coexist** | Every fused composite (`rmsnorm_rotate`, `gate_combine_residual`, etc.) has an unfused chain equivalent. The dispatcher prefers fused when a registered composite matches the upcoming op chain and falls back to unfused primitives when not. Unfused kernels also serve as the correctness baseline. |
| **Explicit execution profiles** | `strict`, `production`, and `batch_invariant` separate reference arithmetic, bounded production implementation drift, and cross-composition reproducibility. Every profile preserves exact request/control ownership; profile selection resolves once to registered variants rather than adding hot-path branches. |
| **Library-first, server-included** | `pip install hipengine` gives you `from hipengine import LLM` plus the `hipengine serve` OpenAI-compatible server CLI. The torch-free inference hot path still does not import FastAPI/Uvicorn. |
| **Extensible by design** | Four orthogonal plugin axes — **backend**, **model**, **quant**, **layer** — not hardcoded branches. See Extensibility Design. |
| **Evidence-backed performance** | Every performance claim comes with a reproducible benchmark command, hardware context, and workload shape. No marketing numbers. |

### Execution-profile architecture

hipEngine exposes exactly three execution profiles:

- **`strict`** is the implementation oracle for the selected model, quant, KV
  policy, and backend.
- **`production`** preserves exact request/slot/token/position/mask/KV/state
  ownership while allowing calibrated same-quant T1/T2 arithmetic drift,
  deterministic under an identical execution schedule.
- **`batch_invariant`** adds a fixed-seed result guarantee across supported
  slots, neighbors, batch widths, admission order, cancellation, and
  compaction. Its first implementation may reuse strict variants.

Execution profile is orthogonal to model weights, quant, KV storage, sampling,
and speculative policy. Weight/KV representation changes and approximate
routing, acceptance, or sampling remain explicit product/experiment choices.
Profile resolution produces an immutable variant manifest over the existing
`(backend, layer, quant, variant)` registry; it is not a fifth plugin axis and
must not add `if profile` branches to engine/model hot paths. Missing or
uncertified production variants fall back to registered strict variants.

The exact control-plane, determinism, numerical calibration, evaluator, and
migration/default rules are normative in
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md). The approved implementation
and performance sequence is
[`PRODUCTION-NUMERICS-CAMPAIGN.md`](PRODUCTION-NUMERICS-CAMPAIGN.md). Current
non-exact defaults are not grandfathered, and public behavior does not change
until the evaluator, manifests, and serving gates are retained. The first
ZBook Qwen3.6 c1/cN package decision retains small implementation-route wins
but does not switch the public default: its canonical server packet fails soak
completion (87 completed, 33 overloaded of 120), and named-profile manifest,
task, and BF16-relative evidence remains open. See the
[`bundle decision`](../benchmarks/results/2026-08-16-zbook-qwen36-production-profile-cn-blocked.json).
The next same-host tuning cycle is governed by the frozen
[`ZBook production-numerics PLAN/PUNCHLIST`](QWEN36-35B-ZBOOK-PRODUCTION-NUMERICS.md);
its PN1 named-profile/control foundation must land before new candidate
arithmetic or kernel tuning begins.

## Architecture

### Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  USER INTERFACE                                                  │
│  • hipengine.LLM.generate()           (library API)              │
│  • hipengine serve                    (OpenAI-compatible server) │
│  • hipengine bench                    (benchmark launcher)       │
├─────────────────────────────────────────────────────────────────┤
│  LOADING (~900 lines, torch-free)                                │
│  • safetensors mmap + hipMemcpyAsync to device                   │
│  • HF config + chat template (json + jinja2)                     │
│  • HF tokenizers (Rust via pyo3, no torch)                       │
├─────────────────────────────────────────────────────────────────┤
│  DISPATCH (~700 new + ~10,900 adapted, Python)                   │
│  • Scheduler        — chunked prefill, decode batching           │
│  • KV Pool/Ledger   — backend-declared global pools + claims     │
│  • Prefix Cache     — RadixCache trie (default) or prefix_lru    │
│  • Fusion Planner   — op chain → kernel plan, prefers fused      │
│  • Model Plugin     — Qwen3.5, Gemma 4, sansho, Llama            │
│  • Quant Plugin     — fp16, w8a8, w8a16, w4_paro, gguf           │
│  • Engine Loop      — hipGraph replay after warmup               │
├─────────────────────────────────────────────────────────────────┤
│  CORE (~1,900 lines, torch-free primitives)                      │
│  • hipengine.Tensor  — device ptr + shape/stride/dtype + dlpack  │
│  • device.py         — HIP/CUDA enumeration, multi-GPU context   │
│  • memory.py         — mmap + hipMemcpyAsync, pinned host mem    │
│  • hip.py            — hipGraph capture + replay via ctypes      │
│  • pm4/ (planned)    — exact graph inspection + retained ROCr IB │
│  • blas.py           — hipblasLt / cublasLt bindings (ctypes)    │
│  • build.py          — hipcc/nvcc subprocess + .so cache         │
├─────────────────────────────────────────────────────────────────┤
│  KERNELS (~18,600 HIP + bindings; 120 __global__; backend-keyed) │
│  • kernels/hip_gfx1100/   — W7900/RDNA3, proven kernels          │
│    ├─ attention/    — full_attn_decode, paged_attn_decode        │
│    ├─ linear_attn/  — conv prefill/decode, GDN recurrent         │
│    ├─ moe/          — router, group/scatter, w8a8_grouped, swiglu│
│    ├─ quant/        — w8a8_act, w8a16_linear, w8a16_moe, paro_awq│
│    ├─ wmma/         — i8 tile/GEMM                               │
│    ├─ norm/ rotary/ — rmsnorm, rotary                            │
│    ├─ fused/        — silu_mul, gate_combine, weighted_sum       │
│    └─ common/       — helpers.cuh + extension.cpp aggregator     │
│  • kernels/hip_gfx1151/   — Strix Halo / gfx1151 initial port    │
│  • kernels/cuda_sm86/     — NVIDIA (future)                      │
│  • kernels/cpu_reference/ — torch-free numpy, correctness        │
│  • kernels/registry.py    — (backend, layer, quant, variant)     │
└─────────────────────────────────────────────────────────────────┘
```


### Host Design: Why Clean Instead of Forked

The host is purpose-built because the existing options carry assumptions we don't share:

| Existing | Assumption | Why We Break It |
|----------|-----------|-----------------|
| nano-vllm | Dense models only, FP16/BF16 tensors | We need MoE-first, quantization-native tensors |
| nano-vllm | CUDA graphs, multiprocessing TP | ROCm graphs are weaker; we want single-process with optional gloo/nccl |
| mini-sglang | Overlap scheduling, ZMQ frontend | Adds complexity for throughput we can get from kernel efficiency |
| vLLM | FlashAttention, CUDA-only kernels, torch-bound | FlashAttn doesn't exist on ROCm; we ship our own FA2 prefill kernel; torch is not a runtime dep |
| all of the above | `torch.Tensor` as the universal value type | Our kernels take raw device pointers; torch is optional dlpack interop at the user boundary |

Our host is simpler because **the kernels do the heavy lifting**. The scheduler just needs to:
1. Continuously batch request work into efficient prefill/decode/verify steps
2. Route each step to the right kernel dispatch path
3. Manage KV cache pages with a pluggable policy
4. Commit sampler outputs and completed requests without stalling the active batch

### Concurrent Decode, Continuous Batching, and SpecDec Readiness

The active Generation-2 request-lifecycle, scheduler, global device-KV pool,
prefix-cache, c1-c32, and FastDMS integration design is
[`CONCURRENCY2.md`](CONCURRENCY2.md). The approved continuous speculative
execution campaign is [`SPECDEC2.md`](SPECDEC2.md); S1-S6 are functionally
closed with automatic K0. The stable-gfx1151 activation/hot-cycle follow-up
[`SPECDEC2-PERF.md`](SPECDEC2-PERF.md) is closed through P10: retained explicit
production-FP16/strict-fallback mechanics and exact fixed cells remain, but no
automatic product cell promotes because the capacity-1 C1 premise does not
engage on the normal capacity-4 server owner; automatic remains K0. The
independent gfx1100 campaign is closed with retained exact C1 device chains and
automatic K0 under
[`SPECDEC2-PERF-GFX1100.md`](SPECDEC2-PERF-GFX1100.md). The active W7900
promotion campaign independently targets real Generation-2 MTP for Qwen3.6
35B MoE and 27B Dense under the production numerical/task/serving gates in
[`MTP-CONCURRENCY2-DUAL-PROMOTION.md`](MTP-CONCURRENCY2-DUAL-PROMOTION.md);
the prior measured queue remains historical context in
[`MTP-CONCURRENCY2-RECOVERY.md`](MTP-CONCURRENCY2-RECOVERY.md).
Source audit and rejected alternatives remain in
[`SPECDEC2-RESEARCH.md`](SPECDEC2-RESEARCH.md).
[`CONCURRENCY.md`](CONCURRENCY.md) is the legacy retained c=N kernel/resident-
runner roadmap and evidence history. The batch-shaped, `KVLiveSpans`,
transactional-KV, and plugin invariants below remain binding while Generation-2
host ownership replaces the older implementation sequence.

hipEngine is a better foundation for c>1 than the current `nano-vllm-amd`
native PARO path. The runnable tree now has retained direct c=2/c=4/c=8 PARO
decode where the backend-specific gates pass, an opt-in gfx1151 PARO resident
OpenAI owner, and a guarded GGUF MTP server route, but retained production c>N
throughput still requires the gates below. Treat c>N numbers as diagnostic
unless the benchmark explicitly says retained-ready.

Design rule: **every new runtime, scheduler, KV, and kernel ABI must stay batch-shaped and speculative-verification-safe even when the first implementation only runs `C=1`.** Scalar c=1 entrypoints are allowed as smoke wrappers, not as the canonical internal interface.

#### Terminology

| Term | Meaning |
|---|---|
| Batched prefill | Multiple prompt tokens, usually for one request chunk; shape is token rows, not necessarily concurrent users. |
| c>1 / c=N decode | `C` independent live requests advance one target token each in the same decode step. |
| Continuous batching | Requests are admitted, compacted, finished, and reclaimed while other requests keep decoding. |
| Speculative verify | Draft candidates are flattened into verification rows (`V`) that may share prefixes, form chains, or form trees; `V` is related to but not identical to `C`. |

#### Day-1 invariants

- **Batch-shaped runtime ABI.** Hidden/logit buffers are `[C, hidden]` or `[rows, hidden]`; token ids, positions, context lengths, finish flags, and active masks are `[C]`; per-layer state is indexed by physical batch row plus stable request id. New scalar-only host state is a design bug unless it is explicitly a test wrapper.
- **Stable request identity is separate from physical slots.** The scheduler owns `request_id -> slot` and `slot -> request_id` maps, can compact/reorder slots between graph launches, and passes row maps to kernels whose routed lanes are not simply `row == request`.
- **Continuous batching is the scheduler contract.** Prefill chunks, decode steps, and speculative verification steps are separate work classes sharing the same active-request table, resolved KV-cache backend, sampler, and completion/reclaim path.
- **KV formats do not own concurrency.** Retention topology, hot codec/layout, and cold tiering resolve to a `KVCacheBackend` that supplies pool plans, atomic resource claims/deltas, storage views, and kernel capabilities. BF16, INT8, DMS, and future formats never add request queues or scheduler branches.
- **`KVLiveSpans` is the mandatory attention/KV-write liveness ABI.** Dense paged KV, DMS/H2O/SnapKV, c>1 decode, and speculative verification all pass per-sequence spans rather than scalar `(block_table, context_len)` tuples; a registered `KVStorageView` supplies format planes without leaking codec metadata into the scheduler.
- **KV mutation is transactional.** Canonical KV is changed only through scheduler-owned commit points. Speculative draft/verify writes go to scratch pages or an append journal and are committed by accepted-token count, then rolled back/discarded for rejected candidates.
- **Draft/verify rows are first-class.** MTP, EAGLE3, DFlash, Medusa, and Lookahead all produce `DraftBatch` metadata: `request_id`, candidate token(s), parent position, draft depth, optional tree parent, and active mask. Verification kernels consume that metadata instead of assuming a linear c=1 chain.
- **Graph capture buckets include shape, not just batch size.** Buckets are keyed by active `C`, context/page bucket, prefill/decode/verify mode, draft length or tree shape, active-mask density, top-k/experts, and graph-steps-per-replay.
- **Dispatch remains plugin-based.** c-aware, KV-format-aware, or specdec-aware behavior registers new backend/model/speculative/layer/kernel variants; engine code must not grow `if backend == ...`, `if quant == ...`, `if kv_dtype == ...`, or one-off `if spec_method == ...` hot-path branches.

#### Next campaign: artifact-scoped compact INT8 KV continuous batching

The approved next INT8 KV campaign is
[`QWEN38-INT8-KV-CONTINUOUS.md`](QWEN38-INT8-KV-CONTINUOUS.md). It starts by
integrating the divergent gfx1100/gfx1151 Qwen3.8 evidence and locking runtime
admission to immutable artifact identity, backend, weight quant, KV layout, and
scale policy. It then adds a temporary no-mirror serial c>N correctness route,
a true row-batched INT8 split-K producer/reducer, shared prefill ownership,
complete admission accounting, and cancellation/grow/shrink/overload gates.

This is a kernel/runtime and resource-accounting campaign, not a scheduler
rewrite. Short mirrored gfx1100 c1->c4 controlled SSE already proves live
admission/reclaim, but it is memory-negative. `IKV-C0` and `IKV-C1` are
complete: runtime admission is artifact/backend/contract scoped, and the exact
gfx1100 artifact now keeps compact no-mirror c2/c4 requests resident while
executing every model transition through an explicit physical-c1 serial
fallback. Shifted token/logit/state/KV exactness, varied-prompt server parity,
zero persistent BF16 bytes, staggered cancellation/survivor continuation, and
zero final ownership pass on RX 7900 XTX. This is not native c>N and carries no
throughput claim. `IKV-C2` row-batched direct INT8 attention is next; BF16 stays
supported/default.

#### Current status

| Question | Answer |
|---|---|
| Can current hipEngine run real c=8 PARO decode? | Yes on gfx1151 for W4/BF16-KV greedy contexts covered by the retained profile. Direct physical c2/c4/c8 are independent-c1 exact at p512/d128, use 40/40 selected-batch layers, and never stack c2 groups. G5 attaches those widths to the shared resident OpenAI loop and makes them the gfx1151 package default: blocking F1 c1/c2/c4/c8 is 47.124/51.962/60.323/61.253 aggregate tok/s, all 68 rows exact, and the complementary SSE/native-plus-serial packet keeps all 100 rows exact. gfx1100 remains retained only at direct c2; sampled-native, context >=1024, other-KV, capture/replay, and gfx1100 owner c4/c8 remain open. |
| Can current hipEngine run native GGUF c>N AR? | Yes through one true physical c8 group on both gfx1100 and gfx1151. Direct eager/graph, ragged, sparse-retirement, cancellation, all-layer hidden, Conv/GDN/live-KV, profiler-family, and repeated same-session scaling gates are retained; F3/F3B's clean gfx1151 direct c1/c2/c4/c8 is 50.335/78.552/108.050/133.852 aggregate tok/s, with c8 at 2.659x c1 and 748 packed-native / zero row-local/copy dispatches. The exact singleton-indexed GDN default improves c2/c4/c8 by 8.71%/5.25%/4.04% while leaving c1 structurally unchanged; F3B then adds an exact physical-C8-only 128-thread qkv+gate pair rowtile for another +0.452%, with 30 expected pair-rowtile launches and lower widths/gfx1100 unchanged. gfx1100 keeps segmented GDN pending independent transfer. Both targets retain honest arbitrary-C/C>8 lowering as multiple declared groups. The shared owner uses dense ephemeral execution rows so live occupancy selects c1/c2/c4/c8 without moving stable scheduler slots, state, or KV; gfx1151 clean F2 server retention preserves all p512/d128 and live-transition outputs with occupancy-one at 95.625% of same-process direct c1. The current optimized corrected-window server path adds true physical C8, resident packed graphs, bounded fair-prefill bursts, resident telemetry reuse, and terminal-state discard: blocking C1/C2/C4/C8 is 44.321/59.783/75.580/86.185 tok/s, exact SSE is 42.147/59.102/73.971/84.196, delayed C8 is 67.788, and all 117 rows are exact; F3B's separate clean C1/C8 packet remains mixed within server noise and makes no additional server-speed claim. gfx1100 transfer remains separate. Neither target claims native c9/c13. gfx1151 additionally retains explicit uniform `int8_per_token_head` c1/c2/c4/c8 continuous serving through rounded context 8192 with bounded BF16 attention mirrors: corrected-window exact SSE is 42.759/55.128/71.284/81.140 tok/s, blocking is 44.225/60.598/74.631/83.408, delayed C8 is 65.034, and all 117 server rows plus the 11-prompt/99-position KL/top-1 gate pass. gfx1100 now independently qualifies the same short mirrored lifecycle at 512/24 through staggered c1->c4 SSE (4/4 exact, occupancy `0->1->4->3->2->1->0`, admitted/reclaimed `4/4`, zero final ownership). The exact gfx1100 Qwen3.8 artifact additionally qualifies compact no-mirror logical c2/c4 through a physical-c1 serial fallback: p512/d24 is independent-c1 exact, shifted logits/state/KV are byte-exact, persistent BF16 bytes are zero, and staggered cancellation drains cleanly. Neither mirrored route is default or memory-saving; row-batched direct INT8 attention, longer compact c>N INT8, and broader quant/sampling remain open. |
| What does the merged UD-Q3_K_M branch add? | A separately gated gfx1100 GPU1 direct path: exact fully-bulk Q3 prefill, native C=2/4/8 decode with exact IDs/full logits and no c>N serial fallback, and a transactional blk.40 NextN diagnostic. The direct C8 rows reach 207.780/211.177 aggregate tok/s at 512/4K; the exact NextN route is economically rejected and remains disabled. |
| Does current hipEngine implement continuous batching? | Partially project-wide; correctness and real server scaling are retained for both gfx11 GGUF OpenAI paths and for gfx1151 PARO W4/BF16-KV greedy c2/c4/c8. Blocking calls and SSE share one model-owning loop that admits during decode, executes bounded prompt chunks, streams row-owned tokens through bounded queues, cancels or retires rows, and drains through runner close. The GGUF owner densifies only execution rows and selects c1/c2/c4/c8 from occupancy while request/session/KV identity stays stable; both gfx11 owners are retained for BF16, and gfx1100 additionally has a measured short mirrored-INT8 staggered c1->c4 lifecycle. PARO uses a fixed-capacity stable-slot session, profile-partitions c3/c5/c6/c7 into certified widths, and defaults native c2/c4/c8 on gfx1151. gfx1100 PARO owner symmetry and broader sampling/KV/context remain open. Explicit short mirrored-INT8 requests preserve policy identity, exact outputs, reclaim, and fail-closed unsupported layouts, but do not broaden the default or prove compact INT8. Direct no-mirror c2/c4 residency now has the `IKV-C1` physical-c1 serial correctness anchor; native row-batched attention and full promotion remain the separate `IKV-C2`-`IKV-C7` work. |
| Is exact Qwen3.8 Q4_K_M public MTP wired into generation? | Yes. [`QWEN38-Q4KM-MTP-SERVING.md`](QWEN38-Q4KM-MTP-SERVING.md) and the completed [`Dynamic Admission campaign`](CONCURRENCY2-GFX1151-MTP-DYNAMIC-ADMISSION.md) remain the serving foundation. Strict/BF16 C1/K3/context1-67 natural25 remains automatic at **18.191 vs 11.062 tok/s (1.6445x)**. The reviewed current-head all-ten explicit K3 matrix is C1-C8 **15.753/28.441/30.541/35.474/27.980/32.807/33.106/35.423 tok/s** versus own AR **11.112/18.090/23.879/30.150/35.778/40.343/43.974/47.194**; all 80 generated-ID/route/budget cells pass. AR leads every external engine C3-C8 and explicit K3 leads MTP C3-C4. NextN draft depth is serial but batch-shaped; C2-C4 and qualified explicit C8 target verification use one flattened packed forward, active C1 uses the request-local transactional verifier, production C5-C7 stays on AR, and qualified C8 uses one physical proposal group with one R32 target group. A reviewed full-width K1 diagnostic finds real row-bucket candidates at C6/R12 (**35.383, +7.85% vs split K3**) and C8/R16 (**39.260, +10.83%**), but both remain below own AR and are not admitted by the profile policy. Exact prompt streaming, proposal-head reuse, and Q4/Q5/Q6 owners remain scoped; width-4 prompt streaming changes acceptance and therefore remains explicit T3 rather than automatic. The retained active-C1 target at `b58a70c82` keeps the wide physical provider but selects the request-local transactional verifier when exactly one request is active: capacity-3 C1 is **19.428 tok/s versus 11.518 AR (1.6868x)** with 10/10 AR equality, while C3 stays packed at **32.919 tok/s** with unchanged acceptance and equality. The profile-owned production policy preserves C1-C4/K1-K3 and admits explicit C8/K3 while C5-C7 stay K0; the current ten-prompt C8 rerun is 52.103 versus 52.025 true-AR tok/s (1.0015x), and its cancellation/refill lifecycle is exact and fully drained. Unqualified context, horizon, and sampling axes remain K0, and strict fallback remains registered. The closed [`scaling campaign`](QWEN38-GFX1151-SCALING-CAMPAIGN.md), [C1 retention](../benchmarks/results/2026-09-03-gfx1151-qwen38-c1-singleton-target-retained.json), and [external survey](QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md) own the current evidence and remaining C=N/prefill work. |
| Is current SpecDec wired into generation? | Yes. Generation-2 owns proposal, target frontier, transaction, accept/commit, output, cancellation, and K0 policy. On gfx1100, dense C1 K1-K3 remains 1.272x/1.407x/1.439x AR and packed PARO C1 is exact; physical C2 target repair plus the exact R6 projection route reaches 22.393 tok/s (0.7156x AR) at 74.28% acceptance, so physical capability remains false. On gfx1151 Q4_K_S, the fixed capacity-1 production C1/K2 path reaches 1.4087x AR, but normal capacity-4 serving executes zero speculative cycles; best physical C2/C4 remains 0.6975x/0.5843x AR. The generic gfx1100/gfx1151 campaign automatic policies stay K0. The exact Q4_K_M public-serving scope is tracked separately above. The next premises are a materially cheaper physical target dataflow and true singleton staged engagement under a normal wider server owner, not more acceptance tuning. |
| Is the design cleaner for adding c>1 than `nano-vllm-amd`? | Yes. |
| Would just setting `tokens=8` work? | No. |
| Is hipEngine the better place to build c=8+ PARO and SpecDec? | Probably yes. |

Why the design is better positioned:

- `GenerationOutput` carries exact generated token IDs, and non-streaming OpenAI
  completion/chat responses expose validated per-choice and all-choice totals;
  decoded-text re-tokenization is diagnostic only.
- Generation timing is explicitly scoped and owned: choice timing normalizes to
  one row/one owner, packed PARO/GGUF groups share a stable batch ID, and server
  benchmark aggregation counts one timing owner per batch.
- The hot path owns raw HIP pointers and `hipGraph` replay directly instead of depending on torch tensors or PyTorch graph wrappers.
- Many wrappers already expose `tokens`, `rows`, or row-shaped grids, so partial batching can be tested without changing the public API.
- `KVLiveSpans` plus a registered `KVStorageView` represent per-sequence K/V liveness and format planes rather than a single scalar `(block_table, context_len)` pair; `KVCacheBackend.prepare(...)` builds the batch view.
- The kernel registry can add c-specific variants such as `(layer="selected_pack8_gemv", variant="batch8")` or `(layer="paged_attn_decode", variant="gqa_batch")` without engine-wide backend/quant branches.
- Decode graph capture is already framed as shape buckets rather than one global graph.
- Model plugins can advertise optional speculative heads, while speculative methods live under their own plugin boundary instead of forking the engine.

Current blockers that keep project-wide c>N incomplete:

- The gfx1100 and gfx1151 GGUF adapters now use the same persistent real
  model-loop contract, reusable resident-session identities, scheduler-owned
  policy-shaped device KV, and bounded request-owned token streams rather than wrapping
  complete inner generation calls. gfx1100 D4/D5 and gfx1151 E1 pass mid-generation admission,
  bounded mixed prefill/decode, packed-group membership changes, independent-c1
  survivor state/KV, disconnect/reclaim, real SSE, metrics, and final ownership.
  Both are correctness-retained at `continuous_eq_ok`, have retained real server
  throughput/latency, and retain arbitrary-C lowering plus explicit optional-
  compaction correctness. The backend-neutral owner now maps stable scheduler
  slots into occupancy-adaptive dense c1/c2/c4/c8 execution rows without moving
  state or KV; clean gfx1151 hardware is exact and non-regressive, with only the gfx1100
  transfer still open. Automatic ownership compaction,
  broader sampling, and gfx1100 PARO owner c4/c8 coverage remain open.
- PARO has independently retained explicit selected-batch c2 steps on gfx1100
  and gfx1151. gfx1151 additionally retains true physical c4/c8 direct steps,
  all-layer state/KV/NumPy-context, sparse c8->c1 immutability, category/heldout,
  primitive/profiler gates, repeated blocking/SSE p512/d128 server scaling, and
  package-default shared-loop attachment with a no-flag OpenAI confirmation.
  gfx1100 owner c4/c8, longer contexts, sampled native groups, graph replay, and
  non-BF16 KV remain open. GGUF now has
  retained direct native c2/c4/c8 correctness, family profiling, repeated
  scaling, live membership, and arbitrary-C lowering on both gfx11 targets.
  gfx1151 additionally retains exact BF16-KV real-Uvicorn c2 through 64K,
  mixed 1K/4K/32K membership, bounded grow/shrink, retryable pressure rejection,
  and stale-pointer-safe graph regrow. gfx1151 carries short uniform
  `int8_per_token_head` payloads, FP16 scales, and bounded BF16 mirrors through
  continuous ownership at c1/c2/c4/c8 with exact API/quality/reclaim evidence.
  gfx1100 independently passes staggered short mirrored c1->c4 admission/reclaim
  with exact outputs and zero final ownership. Both remain memory-negative;
  longer c>N INT8, tail4, direct/no-mirror INT8 attention, and artifact/backend
  capability admission are the independent `IKV-C0`-`IKV-C7` gates.
- Several decode kernels are row-parallel GEMV rather than true grouped/MMQ/WMMA
  batch kernels. They increase grid size but do not reliably reuse streamed
  weights across requests, which is visible in the weak gfx1151 c=1->c=8 scale
  versus llama.cpp Vulkan.
- GQA split-K and full-attention now have primitive trace/parity plus exact
  per-sequence `KVLiveSpans` server coverage for gfx1151 BF16-KV c2 through 64K.
  Short mirrored uniform INT8 has payload/scale-backed gfx1151 c2/c4/c8 evidence
  and independent gfx1100 staggered c1->c4 admission/reclaim evidence. Longer
  direct/no-mirror INT8 and tail4 still need row-count-specific kernels and
  artifact/backend quality gates. See the
  [gfx1151 mirrored packet](../benchmarks/results/2026-07-19-gfx1151-gguf-mirrored-int8-continuous-concurrency.json),
  [gfx1100 lifecycle/frontier artifact](../benchmarks/results/2026-08-16-qwen38-27b-actual-context-quality-w7900.json),
  and [`IKV-C0`-`IKV-C7` campaign](QWEN38-INT8-KV-CONTINUOUS.md).
- Selected MoE decode has row-aware/grouped diagnostic coverage for c<=8, but
  retained performance still needs routed-lane profiling and c-aware thresholds
  for grouped GEMV versus compact/WMMA execution.
- GGUF MTP serving is phase-serial at the slot level: draft, target verify, then
  commit. Target verify is packed up to four slots. The canonical milestone
  glossary, ownership distinctions, and qualified scorecard are in
  [`NATIVE_SPEC_CYCLE.md`](NATIVE_SPEC_CYCLE.md). The provider-neutral
  `NativeSpecCycleLauncher` N0 ABI plus gfx1100 reusable B1/B2 N1 target graphs
  are landed. The exact dense Qwen3.6 native verifier extends N1 VERIFY to an
  independent B3 bucket with dynamic row positions/`KVLiveSpans` and exact
  pre-output-norm trunk-row capture. N2 device accept/commit originally remained
  B1/B2; the 2026-08-06 dense extension now admits B3 and the exact transactional
  verifier selects N2 for eligible full-room, no-logit, session-stream B1/B2/B3
  cycles. Diagnostic logits, caller streams, output-cap tails, and unsupported
  captures retain N1/eager execution. The initial clean exact dense natural25
  graph gate selected B3 at 25.193 tok/s / 1.2362x own AR, with every prompt/
  category/heldout and transaction gate exact. Later arithmetic and submission
  work raises the canonical exact B3 packet to **61.020 tok/s**. N1 remains
  byte-exact across dynamic positions and cached-session resets; the retained
  accuracy-traded llama-compat suite reaches 122.667 tok/s versus llama.cpp's
  115.444 tok/s W7900 floor. N2 device acceptance, selected hidden/Conv/GDN
  commit, cursor update, and bounded summary readback are landed both behind that
  explicit compatibility route and in the exact dense transactional path. N2
  keeps verifier hidden rows graph-owned so prompt-prefill scratch growth cannot
  invalidate a captured pointer; the B3 extension also commits the selected BF16
  trunk row in stable session storage and returns all target top-1 rows through
  the same bounded payload. Its current exact-route screen improves target+
  policy median **42.441009 -> 41.489807 ms/cycle (1.022926x, 17/17 pairs)**;
  capture-normalized complete wall improves **366.417 -> 364.004 ms** and removes
  110 dispatches plus 21 copies. The immediate natural B1/B2/B3 packet is mixed
  at **+0.428%/+0.040%/-0.193%**, so canonical B3 remains 61.020 tok/s while the
  exact physical default is retained. N3 historically
  joins strict device-chained proposal, the N2 target transaction, MTP-KV
  rollback/accepted-row repair, reseed, and cursor/result accounting behind one
  GGUF scheduler-facing call. The public single-request GGUF MTP loop uses that
  adapter when the registered B1/B2 graph admits the shape and preserves the
  exact prior loop otherwise. The clean committed N3 gate matches all 240 IDs /
  96 cycle semantics at 118.592 tok/s / 1.2858x true AR versus clean N2's
  117.557 tok/s (+0.88%, aggregate-neutral); the faster N1 topline remains
  unchanged. The independent gfx1151 transfer registers the same backend-neutral
  B1/B2 target launcher under its peer backend key: N1 reaches **80.132 tok/s**
  and public N3 reaches **80.099 tok/s**, versus a clean same-commit direct-commit
  control at **70.020 tok/s (+14.39%)**. All 240 IDs / 97 cycle semantics match,
  every train/heldout/category row improves, and the real-model hidden/Conv/GDN/
  KV/cursor plus cached-profiler gates pass. N3P additionally replays strict
  B1/B2 NextN proposal through one
  native graph launch and is profiler-proven to replace 542 `hipLaunchKernel`
  plus 80 synchronous `hipMemcpy` host calls over eight matched cycles, while
  remaining aggregate-neutral and diagnostic. N4 has one shared gfx1100
  `w4_paro` target+accept graph adapter used by both PARO MTP and DFlash; its
  base path declares only `VERIFY|ACCEPT`, preserves provider commit and exact
  fallback, and remains default-off globally. Capture-width-zero FP16 PARO
  replays now own selected linear-state `COMMIT|UPDATE_CURSORS` by default inside
  explicit N4 through graph-owned pointer tables; `TARGET_COMMIT=0` is the
  temporary rollback. BF16 DFlash hidden/KV repair remains outside.
  Strict B1/B2/B3 is exact for all 720 canonical IDs,
  but pooled MTP/AR is only 0.5767x/0.4242x/0.3568x. The first B1 on/off/on
  bracket localized a reproducible 0.216-0.447 ms/cycle regression to repeated
  ABI control/marshalling plus one extra stream synchronization, with target and
  proposer kernels unchanged. N4+ now reuses one validated state-bound ctypes
  slab on stable pointer/shape/stream/all-active replays and removes only the
  duplicate post-native sync. The clean merged gate is exact for all 240 B1 IDs,
  every train/heldout/category split, 150/150 native records, and the B2 resident
  Conv/GDN/KV/selected-state/hidden/cursor oracles. Its matched residual is only
  +0.028 to +0.105 ms/cycle (+0.17% to +0.64%), while cached final-child tracing
  is non-regressive at 16.418 ms/pass on versus 16.490 off with identical
  81.6875 API calls, 2 synchronizations, 1 graph launch, and 1248.5 kernels/pass.
  Old N4 was 16.744 ms, 82.6875 calls, and 3 synchronizations. This retains the
  wrapper-overhead improvement but does not promote N4: strict B1 remains far
  below AR and has no advantage over the direct graph. The selected-commit gate
  is cleanly exact for three arms x 240 IDs/214 cycles/16 accepts, every
  train/heldout/category split, 150/150 expanded native records, accepted-row-1
  state plus following-cycle continuity, and B2 state/KV/cursors. Both candidate
  arms improve capture-adjusted wall **14.051 -> 13.983/13.992 ms/cycle** across
  every category. Cached profile wall brackets **16.518/16.322 ms** around the
  **16.413 ms** control (mean +0.007 ms, neutral) while mechanically reducing
  HIP APIs **80.6875 -> 75.6875**, synchronizations **2 -> 1**, host launches
  **36.1875 -> 34.1875**, and kernels **1248.5 -> 1247.5**. This admits selected
  commit inside explicit N4 without a global N4/AR speed claim. The next exact
  kernel gate replaces the one-thread 256-expert router top8 with deterministic
  256-thread reduction: matched micro-rocprof improves **94.516 -> 5.395
  us/call**, while clean full-suite complete wall improves **16.202 ->
  15.919/15.951 ms/cycle**, proposer update **1.222 -> 1.107/1.106**, and MTP
  throughput **65.188 -> 66.303/66.259 tok/s** with three x 240 IDs/214 cycles/16
  accepts unchanged and every category capture-adjusted-positive. Final-child
  tracing confirms router **115.948 -> 10.741 us/call** and proposer host
  **1.465 -> 1.328 ms**. Dynamic context/KV slot metadata and the bounded
  next-token D2H still block a stable reusable graph. The initial model-
  incompatibility diagnosis was also wrong; a wider verifier t-loop had failed
  to forward the exact shared-expert control, and its repair did not change model
  bytes. Complete PARO proposal ownership, DFlash proposal/hidden/KV commit
  ownership, independent gfx1151 N4
  admission, gfx1151 N3P proposal-graph admission, draft-side batching, rows>=16
  verifier tuning, streaming, and exact/default MTP serving remain open.
- The 2026-08-22 [`MTP-FIX`](MTP-FIX.md) campaign supersedes the dense-gfx1151
  open-status sentence above without rewriting its historical measurements.
  RF0–RF7 now qualify context containment, eager correctness through 64K, exact
  steady long target graphs, lifecycle/fault ownership, OpenAI API semantics,
  an honest one-target-slot load policy, and zero-scope rollback/restart.
  Canonical short server MTP reaches median 1.925x true AR with deterministic
  repeats, but differs on two strict heldouts; RF1 long-task quality is 4/6 and
  RF2 long graph MTP is 0.7164x AR. No automatic scope is promoted, `auto`
  selects AR, and explicit non-streaming greedy MTP remains diagnostic only.
- Exact all-choice generated-token accounting and batch timing ownership are
  available for non-streaming responses. Benchmark coverage still needs
  aggregate tok/s, per-request tok/s, latency, memory, active occupancy,
  generated-token equality, graph/profiler provenance, and same-quant external
  baselines before promoting c>N claims.

#### Expected c=8 behavior

| Path | Expected aggregate c=8 behavior |
|---|---|
| Current retained hipEngine default | No retained c=8 production claim. |
| Eight serial c=1 sessions sharing weights | About 1× c1 aggregate, worse latency. |
| Current diagnostic row-parallel c=8 | Modest gain from larger grids and lower relative launch cost; weights are still mostly reloaded per row. |
| Proper c=8 batch path | Plausibly 2–4× c1 aggregate for Qwen3.5/PARO decode; not 8×. |
| c>16 | Prefer GEMM/MMQ/WMMA and grouped MoE designs over extending c1 GEMV. |

The key distinction is that many current "batched" kernels are row-parallel GEMV. They increase grid size, but they do not automatically reuse streamed weights across requests the way a true GEMM/MMQ/WMMA or grouped-MoE kernel can.

#### Implementation plan

1. **Request and batch-state containers.** Add `RequestState` plus `ResidentBatchSession` (or equivalent) with `[C, hidden]` buffers, device token ids, per-request positions/context lengths, active masks, finish flags, per-layer linear-attention recurrent/conv state, and per-request/per-layer full-attention KV spans.
2. **Continuous-batching scheduler.** Add admission, chunked prefill, decode-step batching, slot compaction, sampler/output routing, and reclaim around `KVCacheBackend` claims/leases/batch views. The scheduler owns physical slots and stable request IDs; kernels see row and registered storage metadata.
3. **Correctness harness first.** Exact request/slot/token/position/mask/`KVLiveSpans`/state-ownership checks bind in every profile. `strict` and `batch_invariant` retain their declared generated-ID equality gates. `production` compares c=2/4/8 against strict at identical teacher-forced contexts with calibrated mean/tail/max KL, top-1, determinism, isolation, state/KV, and task-quality gates; free-running cross-width generated-ID equality is diagnostic rather than a universal promotion requirement. For fixed prompts and greedy sampling, compare c=2/4/8 batch output against independent c=1 runs and require finite logits plus per-layer state/KV bounds checks before any perf claim.
4. **Transactional KV hooks.** Extend the KV policy contract with scratch/journal allocation and `commit(request_id, accepted_tokens)` / `rollback(request_id)` semantics before speculative verification writes can touch canonical KV.
5. **Attention batch kernels.** Add batched paged GQA decode and KV append variants with a batch grid dimension and per-sequence span metadata. Uniform paged KV is first; DMS/variable spans reuse the same public ABI later.
6. **Linear-attention state kernels.** Make conv/GDN recurrent decode consume `[C, ...]` state and update each sequence independently.
7. **MoE batch kernels.** Replace c1 selected-lane assumptions with token→lane mapping, then add grouped-by-expert and compact/WMMA routes once routed-lane counts justify them. Use routed lanes, not token count alone, for the GEMV-vs-WMMA threshold.
8. **Quantized projection dispatch.** Use c-aware rules: c=1 stays GEMV; c=2/4/8 uses multi-column/MMQ-style kernels where they beat row-GEMV; c>16 moves toward GEMM/WMMA.
9. **SpecDec plugin boundary.** Add `DraftModel`, `DraftBatch`, `Verifier`, and `AcceptResult` interfaces. MTP heads are model-attached draft providers; EAGLE3 and DFlash are draft-model plugins; Lookahead/Medusa are lightweight draft providers. All verify through the same target-model batch runner and transactional KV path.
10. **Graph bucket policy.** Capture/replay by active `C`, context bucket, mode (`prefill`, `decode`, `verify_chain`, `verify_tree`), draft depth/tree shape, top-k/experts, and replay length. Fall back to uncaptured launches for rare shapes.
11. **Benchmark protocol.** Add c=N concurrent rows and SpecDec rows only after the corresponding profile-aware correctness harness is green. Report execution profile and manifest hash, aggregate tok/s, per-request tok/s, p50/p95 latency, SLO goodput, memory, active batch occupancy, graph bucket, acceptance rate, accepted tokens per target pass, and whether generated-token equality versus non-spec c1 is binding or diagnostic for the declared profile.

### Hot-Path Dispatch Strategy

At steady-state decode, a 35B-A3B MoE model launches roughly 1,600 kernels per token. Naive Python dispatch through PyTorch adds ~50–200 µs/token of pure overhead. hipEngine has five compounding levers to move dispatch out of the hot path; we pick the cheapest first and add more only when profiling demands it.

| # | Lever | Removes | Status in hipEngine |
|---|-------|---------|---------------------|
| 1 | **hipGraph capture per shape bucket** | ~100% of Python overhead during decode replay. Python runs once per token (sampling trigger). | **Phase 0 starts with batch-size buckets** patterned on `nano-vllm-amd/nanovllm/engine/model_runner.py:250`; c>1/SpecDec expands the key to `(C, context bucket, mode, draft/tree shape, active mask, experts, replay length)`. |
| 2 | **C++ engine-step extension (pybind11 / nanobind)** | Remaining Python scheduler-loop overhead. Python calls one C++ function per batch step. | Phase 3, conditional on profiling evidence. Natural extraction point for a future standalone binary. |
| 3 | **Per-layer kernel batching inside the graph** | Kernel-launch latency (~3–5 µs each on ROCm) in addition to dispatch cost. | Phase 3+. |
| 4 | **Cython / `mypyc` for non-capturable paths** (prefill, variable-length, prefix lookup) | ~5–10× speedup of pure-Python scheduler loops. | Phase 4+, only if capture doesn't cover. |
| 5 | **GIL-release on kernel submit + overlap scheduling** | Hides remaining Python overhead behind GPU work. | Phase 5, research. |

**Phase-0 commitment:** lever #1 only. The nano-vllm-amd code already demonstrates it works on ROCm via PyTorch's `torch.cuda.CUDAGraph` wrapper; hipEngine's torch-free port calls `hipGraphCreate` / `hipGraphInstantiate` / `hipGraphLaunch` directly through `ctypes` on `libamdhip64.so` (~300 lines).

**In-tree retained-PM4 program:** native HIP graph replay remains the portable
baseline, correctness oracle, and fallback. The gfx1100 path inspects an already
captured kernel-only graph through public HIP APIs, extracts the exact
JIT-embedded HSACO, and lowers it to one retained PM4 indirect buffer submitted
through a session-persistent public-ROCr queue. The canonical
stateful/local-cache encoder is exact across natural, heldout, 4K, rebuild,
cancellation, sparse retirement, shutdown, and memory gates. After removing
full 747/748-record provenance serialization from each packed replay, clean
p512/d128 PM4 beats HIP graph by **7.104%/6.626%/4.126%/2.466%** at physical
c1/c2/c4/c8, with 5/5 wins and non-regressive capture/request wall at every
packed width. The gfx1100 backend package therefore owns a PM4 policy only for
the measured Qwen35-MoE H2048/E256 architecture geometry plus
`MOSTLY_Q4_K_M`, physical c1/c2/c4/c8, one-step graph tapes, and replay windows
of at least **160/64/96/80** steps.
The c1 floor preserves >10% margin above the clean p4096 143-step capture
break-even; packed-width floors retain their independent margins. Shorter
windows, unknown model/quants or widths, unrelated graph families, and
peer backends stay on HIP graph. Logical c3/c5/c6/c7 remain packed
eager and transport-unaffected. The implementation removes the Redline
runtime/interposer dependency and supplies a smaller #6529 isolation surface,
but production retains one queue and does not exercise risky recreate.
Architecture admission, exact ABI checks, conservative ordering, no post-submit
fallback, and promotion gates are specified in [`PM4.md`](PM4.md).

**Rule:** we do not add levers #2–5 without `rocprofv3` evidence that dispatch is above ~3% of decode wall time.

#### Fusion Planner

Dispatch converts a layer's op chain into a kernel plan. Fused composites are preferred when a registered kernel matches a contiguous sub-chain; otherwise the planner falls back to unfused primitives. Every fused kernel must have a registered strict unfused chain. Strict composites satisfy their declared exact/parent-parity contract; production composites may reassociate arithmetic only after the profile-wide semantic gate and still fall back to that strict chain.

```python
# hipengine/dispatch/fusion.py
class OpChain:
    ops: list[str]  # e.g. ["rmsnorm", "rotate", "qkv_proj"]

def plan(chain: OpChain, backend: str, quant: str) -> list[Kernel]:
    """Longest-match against the kernel registry. Fused > unfused."""
    plan, i = [], 0
    while i < len(chain.ops):
        for j in range(len(chain.ops), i, -1):
            candidate = "+".join(chain.ops[i:j])
            k = registry.resolve(backend=backend, layer=candidate, quant=quant,
                                 missing="skip")
            if k:
                plan.append(k); i = j; break
        else:
            plan.append(registry.resolve(backend=backend, layer=chain.ops[i], quant=quant))
            i += 1
    return plan
```

Registry keys are `(backend, layer, quant, variant)`. `layer` can be a primitive (`"rmsnorm"`, `"qkv_proj"`) or a fused composite spelled as `"a+b+c"` (`"rmsnorm+rotate+qkv_proj"`). No hardcoded branches; the planner discovers what's available.

### Runtime Without PyTorch

hipEngine does not import `torch` at inference time. This is an architectural commitment, not a Phase-5 cleanup.

#### Why drop torch

Measured from this workspace (`du -sh`):

| Dependency | Disk | Purpose |
|------------|------|---------|
| `torch` (ROCm wheel) | **1.7 GiB** | Tensor library, autograd, dispatcher, compile, SDPA, CUDAGraph, cpp_extension, nn.Module |
| `safetensors` | ~5 MiB | Weight file format |
| `tokenizers` (HF, Rust via pyo3) | ~10 MiB | BPE tokenization |
| `jinja2` | ~1 MiB | Chat templates |
| `numpy` (optional) | ~30 MiB | Convenience, fallback math |
| AOTriton 0.11.2b gfx11xx subset (Git LFS) | ~24 MiB on disk / ~42 MB logical bytes | Baseline full-attention prefill runtime for Qwen3.5/PARO gfx1100 |

A torch-free hipEngine ships as **~125 MiB** including the vendored AOTriton subset vs **~2 GiB** with torch. Faster cold start, cleaner Docker images, no torch GPU-detection surprises, runs in environments where torch is broken (Strix Halo, edge ROCm builds, CUDA-forked environments). AOTriton is a pinned, vendored runtime dependency for the gfx1100 Qwen3.5/PARO path, tracked with Git LFS rather than pulled from PyTorch.

#### Kernel bodies are already torch-free

Auditing `nano-vllm-amd/csrc/amd/qwen35_expert.hip`: `__global__` kernel signatures take **raw device pointers and scalars** (`const uint16_t* __restrict__ key_cache`, `const int32_t* __restrict__ block_table`, …). Zero `torch::Tensor` references inside kernel bodies. The 3,403 `torch::Tensor` references in that file and the 602 in `extension.cpp` are entirely in **host-side launch wrappers** — the surface where we convert torch tensors into raw pointers + shapes. That surface is mechanical to rewrite (~1 day scripted) and gives us native HIP signatures like:

```cpp
void qwen35_paged_full_attn_decode_split_k_warp_launch(
    void* query,          // device ptr, bf16
    void* key_cache,      // device ptr, bf16 or int8
    void* value_cache,    // device ptr, bf16 or int8
    int32_t* block_table, // device ptr
    int64_t* context_len, // device ptr
    int num_heads, int head_dim,
    int num_blocks, int block_size,
    hipStream_t stream);
```

#### Replacement matrix (Python side)

| What torch gives us | Measured usage on native path | Replacement | New LoC |
|---|---|---|---|
| `torch.Tensor` metadata (strides, dtype, device, contig, views) | 1,373 refs in `native/qwen35/*.py` | `hipengine.Tensor` with dlpack export | ~500 |
| `torch.cat` / `torch.stack` / `torch.split` | 67 refs, mostly weight-load time | numpy + `hipMemcpyAsync` for big stacks; pure-Python for small | ~150 |
| `F.scaled_dot_product_attention` (prefill + short decode) | 8 call sites | Pinned AOTriton C++ ABI shim for the default Qwen3.5/PARO path, plus native HIP attention fallback for diagnostics/short prompts | ~200 C++/Python shim + vendored Git-LFS runtime |
| `torch.cuda.CUDAGraph` | 1 site (`model_runner.py:250`) | `hipGraphCreate` / `hipGraphLaunch` via ctypes on `libamdhip64.so` | ~300 |
| `torch.matmul` / `torch.mm` (prefill fallbacks, M≤4) | 10 sites | `hipblasLt` / `rocBLAS` bindings via ctypes on `libhipblaslt.so` | ~400 |
| `torch.utils.cpp_extension.load[_inline]` (JIT dev loop) | 1 loader + 3 `load_inline` | `subprocess.run(['hipcc', …])` + `ctypes.CDLL` + hash cache | ~400 |
| `nn.Module` (state_dict, parameter registration) | Throughout `nanovllm/layers/*` | Plain dataclasses + explicit weight dicts | ~200 |
| `torch.compile` | 5 sites, all on non-native fallback layers | **Drop** — dead weight on our hot path | 0 |
| Triton kernels | 0 call sites in `nano-vllm-amd` native path | **Drop** from runtime deps; keep Triton-as-reference optional | 0 |
| HF `safetensors`, `tokenizers`, `jinja2`, config JSON | Loading glue (~200 LoC) | **Keep** — all already torch-free | ~200 (glue) |

**Total replacement budget:** ~1,950 new Python LoC + the AOTriton C++ shim/vendored runtime + ~200 LoC of loading glue. Against a 1.7 GiB dependency drop and a clear multi-backend story, this is cheap. A native FA2 HIP kernel remains future work only if AOTriton headroom or packaging constraints justify it.

#### Optional torch interop

Users who have torch tensors can still feed them in via dlpack (~50 lines in `hipengine.Tensor.from_dlpack` / `to_dlpack`). Installed as `pip install hipengine[torch]` if the user wants the extra safety of torch-compatible ergonomics; never a runtime dep of hipEngine itself.

### Kernel Port Strategy

All kernels come from the `nano-vllm-amd` research lineage (`gfx1100-qwen3.5` branch). They are **copied and partitioned**, not rewritten: the source today is two monolithic files which we split into family-grouped `.hip` files during the port. The target tree lives under `hipengine/kernels/hip_gfx1100/` — the `hip_gfx1100` prefix makes it a peer of future `hip_gfx1151/` (Strix Halo) and `cuda_sm86/` (NVIDIA) backend trees, not a hardcoded "AMD" directory.

#### Actual Source Inventory (measured ✓)

| Source file | Lines | `__global__` kernels | PyBind exports | Notes |
|---|---|---|---|---|
| `nano-vllm-amd/csrc/amd/qwen35_expert.hip` | 13,769 | **95** | — | All Qwen3.5 attention, paged KV, MoE routing/group/scatter, W8A8 grouped MoE, W8A16 linear + MoE, WMMA i8 GEMM, linear-attn conv, GDN, RMSNorm, rotary |
| `nano-vllm-amd/csrc/amd/extension.cpp` | 1,040 | — | ~94 | `TORCH_LIBRARY` / `PYBIND11_MODULE` bindings for all of the above |
| `nano-vllm-amd/csrc/amd/smoke.hip` | 51 | 1 | — | `smoke_add` |
| `nano-vllm-amd/csrc/amd/qwen35_expert_hip.hip` | 13,769 | — | — | **Near-duplicate** of `qwen35_expert.hip` (only `ATen/cuda/CUDAContext.h` → `ATen/hip/HIPContext.h`). **Dropped on port.** |
| `nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py` | 4,394 Python | **25** | — | Contains one `r'''...'''` block of **3,766 lines** of embedded HIP source compiled via `torch.utils.cpp_extension.load_inline`. Python wrapper ≈ 628 lines. |
| **Total Qwen/PARO HIP source to port** | **~17,535** lines | **120** kernels | | 13,769 + 3,766, excluding the separate `smoke_add` build smoke |
| **C++ bindings to port** | **~1,040** lines | | ~94 exports | |

Pure-Python dispatch under `nano-vllm-amd/nanovllm/native/qwen35/` totals **~10,886 lines** (14,652 total − 3,766 embedded HIP) across `paroquant.py` (4,753), `expert.py` (1,085), `paroquant_weights.py` (854), `wmma.py` (774), `mtp.py` (676), `full_attention.py` (511), `weights.py` (454), `linear_attention.py` (387), `__init__.py` (306), `rmsnorm.py` (155), `linear.py` (138), `spec.py` (115), `router.py` (101), `paroquant_kernels.py` wrapper (628). This is the dispatch layer hipEngine adapts.

#### Split Plan

The monolithic `qwen35_expert.hip` + `paroquant_kernels.py` embedded string are partitioned by family into the target tree below. Kernels are preserved byte-for-byte (modulo `#include` headers); the split is mechanical and must preserve `__launch_bounds__`, template specializations, and compiler flags (`-mllvm -amdgpu-unroll-threshold-local=600` for decode/prefill, plus `-mcumode` for decode).

| Target file (`hipengine/kernels/hip_gfx1100/...`) | Kernels (count) | Source origin | Proven win |
|---|---|---|---|
| `common/helpers.cuh` | — | new, extracted | vec8, warp-reduce, packing helpers shared across families |
| `common/extension.cpp` | — | from `csrc/amd/extension.cpp` | Aggregated PyBind registrations (one entry point) |
| `attention/full_attn_decode.hip` | 2 | `qwen35_expert.hip` | `qwen35_full_attn_decode_kernel`, `_context_tensor_kernel` |
| `attention/paged_attn_decode.hip` | 13 | `qwen35_expert.hip` | `qwen35_paged_full_attn_decode_*` family incl. 4K/8K variants, split-K, context-tensor, warp-cooperative, GQA, int8, and split-K reduce/gate. +12–62% over SDPA at long context; +33% 32K (warp); +20% 128K (V-loop); +11% long-ctx (GQA) |
| `attention/paged_kv_write.hip` | 6 | `qwen35_expert.hip` | `qwen35_write_paged_kv_*` incl. mixed-value, position-tensor, int8 |
| `linear_attn/conv.hip` | 4 | `qwen35_expert.hip` | `qwen35_linear_attn_conv_{prefill,decode}[_lowp,_state]` |
| `linear_attn/gdn.hip` | 6 | `qwen35_expert.hip` | `qwen35_gdn_*` (prefill recurrent k/k2, decode, rmsnorm gate lowp/normal) |
| `moe/router.hip` | 6 | `qwen35_expert.hip` | `qwen35_router_logits`, `_select`, `qwen35_token_rank_count_{partial,finalize}`, `qwen35_token_top2_{partial,finalize}`. 5.7× kernel speedup vs reference topk |
| `moe/group_scatter.hip` | 11 | `qwen35_expert.hip` | `qwen35_moe_group_{count,prefix,scatter,scatter_gather}`, `qwen35_moe_c1_group_metadata*`, `qwen35_moe_gather_*`, `qwen35_moe_combine`, `qwen35_build_lane_to_sorted` |
| `moe/w8a8_grouped.hip` | 10 | `qwen35_expert.hip` | `qwen35_dequantize_w8a8_*` (5) + `qwen35_moe_grouped_*` (5, gate_up / down_flat / accumulate variants) |
| `moe/swiglu.hip` | 2 | `qwen35_expert.hip` | `qwen35_swiglu_packed_gate_up`, `qwen35_dequantize_swiglu_quantize_grouped` |
| `quant/w8a8_activation.hip` | 2 | `qwen35_expert.hip` | `qwen35_quantize_activation_{i8,f32_i8}_per_row` (per-token dynamic int8) |
| `quant/w8a16_linear.hip` | 5 | `qwen35_expert.hip` | `w8a16_linear`, `_batched`, `_f32`, `_batched_f32`, `_lowp_out` |
| `quant/w8a16_moe.hip` | 17 | `qwen35_expert.hip` | `w8a16_gate_up*`, `_down*`, `_shared_*`, `_selected_experts`, `_single_*`, `_shared_gate_up_bulk*`, `_shared_down_bulk_combine*`. +54% decode family |
| `quant/paro_awq_gemv.hip` | 7 | `paroquant_kernels.py` | `gemv_awq_v8`, `_pack8`, `dual_pack8`, `selected_dual_pack8_strided[_rotate_out]`, `selected_pack8`, `dense_gemv_out`. +19% decode, coalesced pack8 layout |
| `quant/paro_awq_dequant.hip` | 2 | `paroquant_kernels.py` | `dequant_awq_pack8`, `_dual` |
| `wmma/wmma_i8_gemm.hip` | 4 | `qwen35_expert.hip` | `qwen35_wmma_i8_{tile,gemm,gemm_a_row_major,gemm_grouped_a_row_major}` |
| `norm/rmsnorm.hip` | 6 | mixed | `qwen35_rmsnorm`, `_add_rmsnorm`, `_add_rmsnorm_f32`, `_head_rmsnorm`, `paro_rmsnorm_out`, `paro_add_rmsnorm_out` |
| `rotary/rotary.hip` | 5 | mixed | `qwen35_partial_rotary`, `qwen35_head_rmsnorm_partial_rotary[_position]`, `paro_rotate2`, `paro_rotate3` |
| `fused/fused_ops.hip` | 12 | `paroquant_kernels.py` | `silu_mul_dual_out`, `_dual_rotate_out`, `_pair_rotate_out`, `full_attn_gate_mul_out`, `shared_gate_combine_{,residual_}out`, `weighted_{index_add_[atomic_float_]out, lanes_{sum,inverse}, sum_out}`, `weighted_sum_shared_gate_combine_residual_out` |
| `smoke/smoke.hip` | 1 | `csrc/amd/smoke.hip` | `smoke_add` (JIT-build smoke) |

**Split totals:** ~14 `.hip` files + 1 shared header + 1 aggregator `.cpp`, preserving all **120 Qwen/PARO kernels** plus the separate `smoke_add` build smoke and ~94 bindings with **no kernel rewrites**. Per-file boilerplate (includes, anonymous namespaces, per-family binding sections) adds **~300 new LoC**; dropping the near-duplicate `qwen35_expert_hip.hip` removes **13,769 LoC** from the tree. Host-side launch wrappers are retyped from `torch::Tensor` to raw pointer + shape/stride/dtype signatures during the same pass (~1 day scripted).

**Correctness gate for the split:** after partitioning, verify (a) every kernel name still resolves via the Python extension module, (b) `rocprofv3 --kernel-trace` reports the same kernel set with matching `DurationNs` distribution on the Qwen3.6-35B-A3B decode smoke, (c) KL ≤ 0.05 and top-1 ≥ 90% vs the monolithic build on the correctness fixtures.

Build system: **no `torch.utils.cpp_extension`**. hipEngine's own build layer (`hipengine.core.build`) calls `hipcc` (or `nvcc` for CUDA backends) via `subprocess.run`, links with `ctypes.CDLL`, and caches by source+flags hash. Three HIP profiles adopted from `nano-vllm-amd/nanovllm/native/amd/extension.py` — `decode` (`-mllvm -amdgpu-unroll-threshold-local=600` + `-mcumode`, wave32; CU mode is not wave64), `prefill` (`-mllvm -amdgpu-unroll-threshold-local=600`, WGP/wave32), and `baseline` (no flags, wave32). Native HIP target arch (`--offload-arch=gfx1100` / `gfx1151`) is explicit through `target_arch` or `HIPENGINE_HIP_ARCH` and participates in the cache key. The edit→bench loop stays at ~5–10 s per kernel change.

### Reference backend for correctness

`hipengine/kernels/cpu_reference/` holds a torch-free numpy implementation of every `layer` key registered by any hardware backend. This is the correctness oracle: when a new gfx1100 kernel is ported, the test suite runs the same inputs through the CPU reference and asserts KL ≤ 0.05 / top-1 ≥ 90%. The reference backend also lets hipEngine run on machines without a GPU for CI and for architecture bring-up (develop a new model plugin on CPU first, then port its kernels to gfx1100).

## Extensibility Design

hipEngine has **four orthogonal plugin axes**. Each axis is a registry of implementations; the engine composes concrete instances at load time from the user's choice.

| Axis | Purpose | Examples |
|------|---------|----------|
| **Backend** | Hardware target (kernel set + primitives) | `hip_gfx1100`, `hip_gfx1151`, `cuda_sm120a`, `cuda_sm86`, `cuda_sm89`, `cpu_reference` |
| **Model** | Architecture-level layer sequence + weight name map + chat template | `qwen3_dense`, `qwen3_5_hybrid` (full+linear+GDN+MoE), `gemma4`, `llama3`, `sansho` |
| **Quant** | Weight layout + packing + activation quant | `fp16`, `bf16`, `w8a8_dyn`, `w8a16`, `w4_paro`, `w4_gguf`, `int4_awq_orig` |
| **Layer** | Per-layer-type compute structure (primitive + fused variants) | `full_attention`, `linear_attention`, `gdn`, `sliding_attention`, `moe_top2`, `dense_mlp` |

Kernels are registered with the tuple `(backend, layer, quant, variant)`. The dispatcher resolves kernels at layer-build time; the fusion planner resolves at op-chain-build time.

Execution profile is a policy selector over the existing `variant` key, not a
fifth axis. At model/session construction, it resolves to an immutable manifest
of selected and strict-fallback variants plus evidence identifiers. Dispatch
and graph capture consume that manifest without backend-, quant-, or profile-
specific branches in engine/model code. See
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md).

Public APIs and server entry points default to `backend="auto"`. Auto is a selector
resolved before registry lookup, not a registry key: exact `gfx1100`/`gfx1151`
detections map to the matching HIP backend, `HIPENGINE_BACKEND` can force a
backend for nearby targets such as `gfx1101`/`gfx1102`, and unknown/no HIP
detections warn before selecting `cpu_reference` where a CPU implementation exists.
Public APIs and server entry points also default to `quant="auto"`. Model plugins
must declare a concrete `default_quant`; registry lookup receives that resolved
key. Explicit quant strings bypass the plugin default. GGUF text-generator
factories are registered separately for `hip_gfx1100` and `hip_gfx1151`, and the
resolved backend supplies the JIT target architecture.

### Backend Plugin

```python
# hipengine/kernels/registry.py
@dataclass(frozen=True)
class KernelKey:
    backend: str       # "hip_gfx1100", "cuda_sm86", "cpu_reference"
    layer: str         # primitive ("rmsnorm") or fused ("rmsnorm+rotate")
    quant: str         # "fp16", "w8a16", "w4_paro"
    variant: str = ""  # "split_k_warp", "pack8_strided", ""

_KERNELS: dict[KernelKey, Kernel] = {}

def register(key: KernelKey, kernel: Kernel): _KERNELS[key] = kernel

def resolve(*, backend, layer, quant, variant="", missing="error") -> Kernel | None:
    """Narrowest-to-broadest match: variant -> no-variant -> quant:fp16 -> cpu_reference."""
    ...
```

Kernels self-register at module import:

```python
# hipengine/kernels/hip_gfx1100/attention/paged_decode.py
register(
    KernelKey("hip_gfx1100", "paged_attn_decode", "fp16", "split_k_warp"),
    _native.qwen35_paged_full_attn_decode_split_k_ctx_tensor_warp_launch,
)
```

Adding a CUDA backend = a peer `hipengine/kernels/cuda_<arch>/...` tree with the same `layer` / `quant` / `variant` key space. The architecture-qualified `cuda_sm120a` scaffold and first Moonshine FP16 glue family use this boundary; adding Strix Halo similarly uses `hipengine/kernels/hip_gfx1151/...`. The engine, dispatch, model, and quant layers don't change.

### Model Plugin

```python
# hipengine/models/base.py
class ModelPlugin(Protocol):
    arch_names: list[str]               # ["qwen3", "qwen3_moe", "qwen3_5"]
    @classmethod
    def from_hf_config(cls, config) -> "ModelPlugin": ...
    def build_layers(self) -> list[LayerSpec]: ...  # per-layer: type, dims, quant hint
    def weight_name_map(self, hf_name: str) -> str: ...
    def chat_template(self) -> str: ...
    def rope(self) -> RoPEConfig: ...
```

Phase-0 targets (driven by the current research focus):

| Model | Layer mix | Status |
|-------|-----------|--------|
| **Qwen3-0.6B** dense | full_attention + dense_mlp | Phase 0 smoke |
| **Qwen3.5 0.8B** dense | full_attention + dense_mlp | Phase 0 correctness |
| **Qwen3.5 27B** dense | full_attention + dense_mlp | Phase 1 perf target |
| **Qwen3.6 35B-A3B** MoE hybrid | full_attention + linear_attention + gdn + moe_top2 | Phase 2 perf target; the ZBook quant/runtime campaign and quality-only automatic-tool/task [`AGENTIC-QUALITY2`](AGENTIC-QUALITY2.md) follow-up are closed, with no quality runtime mechanism retained |
| **Moonshine ASR** encoder-decoder | conv encoder + self/cross attention + gated decoder MLP | HIP FP16 graph decoder and selected encoder hybrids promoted internally; `cuda_sm120a` C0-C8 includes a torch-free encoder, static/continuous batching, and device-owned decode but remains outside public model admission; gfx1151 transfer campaign: [`MOONSHINE.md`](MOONSHINE.md) |
| **Maple-Preview 20B-A1B** ternary MoE | GQA sliding/global attention + top-8/256 MoE + packed ternary/affine4 | gfx11 public c1/c2/c4/c8 path promoted; `cuda_sm120a` c1 generation, native prefill through p512 performance / 770 state, exact wave32 direct decode, and exact split-K global decode through a full p512 suite are retained on GPU0, while CUDA resident batching/serving remain pending |
| **Gemma 4** | sliding_attention + global_attention + dense_mlp | Phase 3 |
| **Llama 3** | full_attention + dense_mlp | Phase 3 |
| **sansho** (custom) | (your arch; see `/home/lhl/amd-gpu-tuning/reference/sansho/`) | Phase 3+ |

Each model plugin owns:
- **Layer sequence**: Qwen3.5 35B-A3B alternates `full_attn` with `linear_attn` and `gdn`; Gemma 4 alternates `sliding` with `global`; dense models are uniform.
- **Weight name map**: HF `model.layers.0.self_attn.q_proj.weight` → our `layers.0.attn.q_proj`.
- **RoPE variant**: standard, partial (Qwen3), YaRN, NTK, Gemma's 10k+ base, sliding-window rotations.
- **Chat template**: jinja2 source loaded from `tokenizer_config.json`.
- **Special tokens**: BOS/EOS/PAD/thinking markers.
- **Optional speculative capability**: MTP layer spec (Qwen3.5 MTP), Medusa heads, EAGLE3 features, and draft-model hookup (sansho's DFlash). The model plugin advertises capabilities; the speculative plugin owns proposal/verification policy.

The model plugin does **not** know about backends or quant. Those are dispatched at layer granularity.
Backend tuning policies identify validated models by immutable architecture
geometry plus quant metadata, never by `general.name` or a model-path string;
renamed exports and finetunes with unchanged execution geometry inherit the
same policy, while geometry drift fails closed to registered fallbacks.

### Quant Plugin

Quantization is **six orthogonal axes**, not one format label. A real quant preset bundles choices across all six. hipEngine exposes them explicitly so new formats slot in by registering new kernels, not by editing dispatch.

| Axis | Examples | Why orthogonal |
|------|----------|----------------|
| **Weight storage** | `fp16`, `bf16`, `int8`, `int4_packed_8`, `int4_packed_paro`, `codebook_exl3_3bit`, `kron_factors` | How weights sit in device memory |
| **Activation preprocessing** | `passthrough`, `per_token_int8`, `per_tensor_fp8`, `hadamard_rotate_paro`, `hadamard_rotate_qtip` | Some quants need input rotation (PARO, QTIP); W8A8/FP8 need activation quant |
| **Compute dtype / accumulator** | `bf16`, `fp16`, `fp32`, `int32` | What the MAC accumulates in |
| **Scale granularity** | `per_tensor`, `per_channel`, `per_group_{32,64,128}` | Affects kernel's scale-load pattern |
| **Calibration artifact** | `none`, `awq_scales`, `gptq_hessian_inverse`, `paro_rotation_matrix`, `qtip_codebook`, `kron_factors` | Loaded alongside weights; may need preprocessing kernel at load time |
| **Kernel family** | `gemm_dequant`, `gemm_intN_actN`, `codebook_lut`, `kronecker`, `fastkron_fused` | Determines kernel shape and dispatch path — different families can't share launch signatures |

```python
# hipengine/quant/base.py
class QuantPlugin(Protocol):
    name: str                       # "w4_paro", "w4_gptq_g128", "w8a8_dyn", "w4_exl3", ...
    # Orthogonal axes (queried by kernel registry, fusion planner, scheduler)
    weight_storage: str
    activation_preprocess: str
    compute_dtype: DType
    scale_granularity: str
    calibration_artifacts: list[str]
    kernel_family: str              # picks which kernel tree handles this preset
    backends_supported: set[str]    # e.g. FP8 weight excludes "hip_gfx1100"

    def prepare_weights(self, raw: dict[str, Tensor]) -> QuantWeights: ...
    def preprocess_activation(self, x: Tensor) -> tuple[Tensor, ActMetadata]: ...
    def layer_key(self) -> str:      # matches KernelKey.quant
        return self.name
```

Quant plugins own layout gymnastics. The ~4,753-line `paroquant.py` collapses into one `W4ParoQuant` class; the dispatch layer doesn't see pack8 nibble math.

#### Quant format roadmap

| Preset | Phase | Kernel family | Backends | Notes |
|--------|-------|---------------|----------|-------|
| `fp16` / `bf16` | 0 | native GEMM / `hipblasLt` | all | Pass-through, correctness baseline |
| `w8a16` | 2 | `gemm_dequant` | hip_gfx1100 | Ported from nano-vllm-amd (+54% decode family) |
| `w8a8_dyn` | 2 | `gemm_intN_actN` | hip_gfx1100 | Per-token dynamic int8 (Quark / SmoothQuant compat) |
| `w4_paro` | 2 | `gemm_dequant` | hip_gfx1100 | PARO pack8 + Hadamard rotation (+19% decode) |
| `w4_gptq` / `w4_gptq_g128` | 3 | `gemm_dequant` | hip_gfx1100 | Reuses PARO kernel tree with different packing; load-time Hessian-based recon artifact |
| `w4_gptaq` / GPT-AQ variants | 3 | `gemm_dequant` | hip_gfx1100 | Adaptive-granularity GPTQ; same kernel family |
| `w4_awq` | 3 | `gemm_dequant` | hip_gfx1100 | AWQ scales; paroquant already lineage-compatible |
| `fp8_e4m3` weight | 5 | `gemm_intN_actN` | `hip_gfx1200`+, `cuda_sm90`+ | **Not on gfx1100** (no HW FP8 matmul); software fallback would be slower than BF16 |
| `w4_exl3` / QTIP trellis | 5+ | `codebook_lut` | all (HIP + CUDA) | ~8k LoC of new HIP kernels; ExLlamaV3's CUDA lineage is PTX-heavy. Research 2–4 weeks |
| `w4_qtip_yaqa` | 5+ | `codebook_lut` | all | QTIP with YAQA (Yet Another Quantization Algorithm) refinement; same codebook kernel family |
| `fastkron` | Research | `kronecker` | all | Compute is reformulated: `W x = vec(Aᵀ vec(x) B)` — two small matmuls per linear |
| `gguf_q4_k_m` | 5 | `gemm_dequant` | all | llama.cpp-compatible dequant; loader is the hard part |
| `higgs_4bit` | Research | `gemm_dequant` + Hadamard | all | Referenced in `reference/sansho/docs/kvcache-quant.md`; ~50% BF16 speed so deferred |
| `aqua_kv` (KV-side, not weight) | Research | — | — | Additive scalar quantization; see KV Cache Plugin section |

**Kernel family implication:** `gemm_dequant` (the weight-dequant-then-multiply family that covers GPTQ/AWQ/PARO/W8A16) already has a mature tree in hipEngine (6 W8A16 linear kernels + 18 W8A16 MoE kernels + 10 PARO AWQ kernels). Adding GPTQ/AWQ is mostly **weight-preprocessing glue**, not new kernels. Adding EXL3/QTIP adds a **new kernel family** (`codebook_lut`) with its own ~14 kernels to port. FastKron is **a new kernel family with a different compute pattern** (two matmuls instead of one).

### Layer Plugin

Layer plugins describe the *shape* of a layer's compute, not the implementation:

```python
# hipengine/layers/base.py
class LayerPlugin(Protocol):
    layer_type: str  # "full_attention", "linear_attention", "gdn", "moe_top2", ...
    def op_chain(self, spec: LayerSpec, quant: QuantPlugin) -> OpChain: ...
    def forward(self, x: Tensor, weights: QuantWeights, kv: KVState, ctx) -> Tensor: ...
```

Each `forward` is a thin shim that:
1. Calls the fusion planner to turn its op chain into a kernel plan.
2. Launches kernels in order, passing device pointers.
3. Updates KV state.

Because `layer_type` is a first-class key, adding Gemma 4 sliding attention is:
- register `SlidingAttention(LayerPlugin)` with op chain `["rmsnorm", "rotate", "sliding_qk", "sliding_attn_decode", "o_proj", "residual"]`
- register the kernel implementations under `hipengine/kernels/hip_gfx1100/attention/sliding_*.hip`

No engine, dispatch, or quant changes.

### KV Cache Plugin (sub-plugin of engine)

Detailed INT8-KV and FastDMS-derived compact-DMS delivery plan: [docs/KVCACHE.md](KVCACHE.md).

KV cache has **three independently modelled concerns**, plus the standard
block-manager concerns. Designing for all three from day 0 is the specific
lesson from `~/FastDMS` and `~/kvcache-quantization-research/`: integrating DMS
into a fixed-page scheduler is major surgery, while treating every low-bit
layout as one `dtype` toggle fails for multi-plane and cross-layer formats. The
normative scheduler/backend boundary is
[`CONCURRENCY2.md`](CONCURRENCY2.md#swappable-kv-cache-backend-contract).

| Concern | What varies | Examples |
|------|-------------|----------|
| **Retention topology** | How live spans change over time | fixed-page (standard paged KV); sliding-window; attention-sink + sliding (StreamingLLM); DMS per-head learned eviction; H2O heavy-hitter; SnapKV prompt-time pruning |
| **Hot codec/layout pipeline** | How active K/V and its metadata are represented and reconstructed | `bf16`, `fp8_e4m3`, `int8_per_token_head`, `int4_packed`, TurboQuant/HIGGS codebooks, AQUA cross-layer predicted residuals, OSCAR-like BF16+INT2 regions |
| **Tier/cold codec** | How inactive/reusable K/V is offloaded and restored | device-only, host/NVMe tiers, KVTC-style transform/entropy coding |

#### `KVLiveSpans` — the fundamental kernel interface

Every attention / paged-KV-write kernel takes a `KVLiveSpans` instead of the classic `(block_table, context_len)` tuple. Uniform policies fill it the same for every head; DMS varies it. `num_seqs` is intentionally a row count: it can mean active decode requests (`C`), prefill chunks, or speculative verification rows (`V`). Stable request identity remains scheduler metadata, not an implicit row index. `KVBatchView` pairs liveness with a registered `KVStorageView`; quantizer planes and reconstruction rules do not become scheduler fields.

```python
# hipengine/kvcache/spans.py
@dataclass(slots=True)
class KVLiveSpans:
    """Per-(row, layer, head) live K/V token spans.
    The contract between KV storage and every attention / KV-write kernel.
    Dense policies fill this uniformly across heads; DMS and DMS-like
    compaction vary spans per head. Rows can be active requests or
    speculative verification candidates.
    """
    base_offsets:    Tensor          # [num_seqs, num_layers, num_kv_heads] int32
    live_counts:     Tensor          # [num_seqs, num_layers, num_kv_heads] int32
    max_live_count:  int             # max across all (row, layer, head) for grid sizing
    token_positions: Tensor | None   # [num_seqs, total_live] int32 — surviving tok positions
    evict_mask:      Tensor | None   # [num_seqs, max_ctx, num_kv_heads] bool (optional)
    request_ids:     Tensor | None   # [num_seqs] int64 — stable scheduler ids for row ownership
    row_positions:   Tensor | None   # [num_seqs] int32 — decode/verify query or write positions
    span_role:       str             # "prefill", "decode", "verify_chain", "verify_tree"
```

`KVBatchView` pairs this liveness object with the registered `KVStorageView` and
kernel bundle; storage is not embedded into `KVLiveSpans`.

#### `KVCacheBackend` protocol

Generation 2 resolves one validated topology+hot-codec+tier composition per
loaded model replica. The scheduler receives resource claims and storage views;
it never branches on dtype or retention mode.

```python
class KVCacheBackend(Protocol):
    spec: KVBackendSpec

    def plan_pools(self, load_plan: DeviceLoadPlan) -> KVPoolPlan: ...
    def estimate(self, request, prefix, stage) -> ResourceClaimSet: ...
    def reserve(self, claims: ResourceClaimSet) -> KVLease: ...
    def prepare(self, work_item) -> KVBatchView: ...
    def begin_transaction(self, rows, draft) -> KVTransaction: ...
    def commit(self, operation, result) -> ResourceDelta: ...
    def rollback(self, operation) -> ResourceDelta: ...
    def reclaim(self, lease) -> ResourceDelta: ...
    def prefix_lookup(self, tokens) -> PrefixMatch: ...
    def maintenance(self, budget) -> list[MaintenanceWork]: ...
```

`ResourceClaimSet` atomically accounts named persistent planes, resident
metadata, prefill/attention/maintenance workspace, transactions, graph slabs,
and whole-device reserve. `KVBatchView` combines `KVLiveSpans`, a registered
`KVStorageView`, and the matching kernel bundle. `GlobalKVPoolSet`, dense
BF16/qualified INT8, compact-DMS host composition, and optional cold tiering are
implemented; `KVPolicy`/`FixedPagedKVPolicy` remains only a compatibility
adapter for unported packages. Scalar `admission_cap()` is not the scheduler
contract. Product blockers and exact evidence are tracked by the executable
[`CONCURRENCY2` audit](../benchmarks/results/2026-08-17-concurrency2-completion-audit.json).

Example resolved compositions include `paged_dense+bf16`,
`paged_dense+int8_per_token_head`, `sliding_sink+bf16`, `dms_compact+bf16`,
`dms_compact+fp8`, and later `dms_compact+aqua_higgs`. KVTC is a cold tier codec
that restores into one of those hot backends, not an attention dtype. Registry
factories reject unsupported combinations before engine startup.

**Scheduler admission** atomically reserves the backend-produced claim vector.
It does not calculate page bytes, scales, codebooks, protected BF16 windows,
predictor state, compact live-token budgets, or tier-transfer scratch. Commit,
rollback, DMS expiry, mixed-tier demotion, and reclaim return mechanically
conserved resource deltas.

**Attention kernels** are registered under the exact topology/layout/kernel
bundle and consume `KVLiveSpans` plus stable storage-plane views. Execution rows
batch only when their complete compatibility keys match. The kernel registry
naturally routes without engine-wide format branches.

#### Why this shape avoids the vLLM-DMS and v1-INT8 pain

FastDMS identifies memory pool, prefill, decode, attention, admission, prefix,
and continuous-batching changes for a DMS port. The local quantization research
adds multi-plane HIGGS/TurboQuant, heterogeneous cross-layer AQUA, mixed
BF16+INT2 demotion, and cold KVTC requirements. hipEngine pays the host design
cost once through `KVCacheBackend` claims/pools plus `KVLiveSpans` storage views.
Adding a format still requires real allocator, codec, and kernel work, but not a
new queue, scheduler, output path, cancellation path, or resident lifecycle.


## Advanced Features Roadmap

### Speculative Decoding (SpecDec)

SpecDec is planned as a scheduler + plugin feature that reuses the same target-model batch runner, KV-cache backend, and kernel registry described in the c>1 readiness section. Drafting changes the work shape; it must not fork the engine.

| Draft Type | Status | Integration shape |
|------------|--------|-------------------|
| Medusa-style heads | Planned | Model-advertised heads produce shallow candidate rows. |
| Lookahead decoding | Partial | Dense GGUF MTP2 has an opt-in request-local exact `ngram-mod` first-refusal composer for qualified K<=3 chains. Repetition-heavy strict C2 D80 is +2.425% vs MTP-only but 0.9875x true AR; canonical production D24 has zero hits. Ordinary MTP/AR remain fallbacks and promotion requires a correct K>3/long-horizon product cell. |
| MTP (multi-token pred) | Research | Qwen3.5 MTP layers provide `DraftBatch` chains attached to the target model; detailed native plan: [`docs/MTP.md`](MTP.md). |
| EAGLE3 | Research | Draft-model plugin emits feature-conditioned candidate chains/trees. |
| DFlash (draft model) | Partial | Generic four-axis public-provider registry plus an explicit-only Poolside Laguna B4 library owner; OpenAI routing and broader DFlash/DDTree serving remain. Detailed native plan: [`docs/DFLASH.md`](DFLASH.md). |

Method-specific details live in `docs/MTP.md` and `docs/DFLASH.md`; the shared
contract below remains authoritative for plugin boundaries and scheduler/KV
integration.

Required contract:

- `DraftModel.propose(batch_state) -> DraftBatch` emits candidate tokens plus `request_id`, `draft_depth`, parent position, optional tree parent, and active mask metadata. `DraftBatch` carries candidate rows only; verifier implementations may insert a root/current-token row into an internal verify batch.
- `Verifier.verify(target_state, draft_batch) -> AcceptResult` runs target verification over flattened rows using `KVLiveSpans` in verify mode.
- `AcceptResult` records accepted counts/tokens per request and the replacement token for the first rejection.
- Canonical KV is updated only through transactional commit/rollback hooks; rejected draft writes never leak into committed request state.
- Disabling SpecDec must produce the same deterministic greedy outputs as the non-spec c=1/c=N path on the correctness fixtures.

### KVTC-Style Tiered Offloading

```
Device (24 GiB) → Host (64 GiB) → NVMe/SATA
     ↑                    ↑
   Hot tokens          Warm tokens
   (current context)   (prefix + recent history)
        ↑
     Cold tokens
     (evicted to disk)
```

A KV-cache backend's optional cold-tier capability manages:
- which backend snapshot handles stay device-resident;
- which objects are pinned host-resident for restore;
- which objects are encoded by KVTC or another cold codec;
- restore/prefetch maintenance work and its resource claims.

The common EngineService still schedules requests. Tier code may not own a
parallel request lifecycle or block due decode work.

### RadixCache vs. vLLM Prefix Caching

| Feature | vLLM Prefix Caching | hipEngine RadixCache (mini-sglang) |
|---------|---------------------|-----------------------------------|
| Structure | Hash-based block matching | Trie-based prefix tree |
| Granularity | Block-level (256 tokens) | Token-level exact prefix |
| Sharing | Copy-on-write blocks | Reference-counted trie nodes |
| Eviction | LRU on blocks | LRU on trie nodes (finer-grained) |
| Overhead | Lower | Slightly higher CPU, better hit rate |

hipEngine defaults to **RadixCache** for better prefix sharing in multi-turn chat and API serving. Each resolved KV backend declares whether radix entries hold immutable pages, snapshot overlays, or are unsupported; incompatible backend/artifact fingerprints never share physical state.

### DMS Support Plan (and why it shapes Phase-0 design)

See [docs/KVCACHE.md](KVCACHE.md) for the staged delivery order: finish the artifact-scoped compact c>N dense-INT8 campaign in [`QWEN38-INT8-KV-CONTINUOUS.md`](QWEN38-INT8-KV-CONTINUOUS.md), then add FastDMS-derived compact DMS over the same `KVLiveSpans` ABI.

Dynamic Memory Sparsification (DMS) trains per-head learned KV token eviction via logit distillation. Compact DMS saves real allocator memory (5–8× vs BF16 KV at 8K context, up to 49× at max context per `~/FastDMS` benchmarks) while maintaining or improving decode speed. The reference open implementation is `~/FastDMS` (shisa-ai). Validated borrowed-channel checkpoints: `shisa-ai/Llama-3.2-1B-DMS-8x`, `nvidia/Qwen3-8B-DMS-8x`. hipEngine's local schema-v2 exact-Qwen3.8 Q4_K_M route now executes no-shadow c1 decode at 128K/256K, but its 768-token-trained CR2 candidate is rejected at integrated 32K quality (max KL 6.0177, 62.5% top-1; no-evict max KL 7.76e-7/100%). The implementation remains default-off while a firewall-safe long-context sidecar is trained and all product gates close.

#### Why DMS is "major surgery" inside vLLM

From `~/FastDMS/README.md`, a DMS port touches seven vLLM subsystems:

| vLLM subsystem | What DMS needs |
|---|---|
| PagedAttention / KV memory pool | Per-layer, per-head variable token counts with partial block deallocation — not fixed pages |
| Prefill kernel | Stream surviving K/V into compact per-layer storage after DMS extraction, not dense KV pages |
| Decode kernel | Per-head keep/evict + sliding retention window + append to compact storage |
| Attention scoring | Replaced entirely: split-K grouped compact decode over variable-length per-head live spans |
| Scheduler / admission | **Admit on compact KV capacity, not dense full-sequence page count.** The hardest boundary |
| Prefix caching | Per-sequence per-head eviction overlays, or disabled |
| Continuous batching | Memory accounting by actual surviving tokens, not logical sequence length |

#### What hipEngine commits in Phase 0 to make DMS cheap later

| hipEngine design choice | Why it helps DMS |
|---|---|
| `KVLiveSpans` = `(base_offsets, live_counts, token_positions, evict_mask)` as the kernel contract | DMS needs per-(seq, layer, head) variable spans. Dense policies fill uniformly; DMS fills variably. Same kernel ABI. |
| `KVCacheBackend.estimate()` returns atomic `ResourceClaimSet` vectors | Dense, DMS, mixed-tier, and quantized formats claim every payload/metadata/workspace plane without scheduler formulas. |
| Fusion planner with chain-matching (not hardcoded ops) | DMS needs fused `rotate + dms_decide + compact_store + decode` kernels. These register as fused composites for `(quant, layer="rotate+dms+store+attn"`. |
| Retention topology and hot codec/layout are resolved composition keys | DMS + BF16, DMS + FP8, DMS + int4/HIGGS-like, and DMS + AQUA can share topology/lifecycle code while owning different planes and kernels. (`~/kvcache-quantization-research/` showed DMS + AQUA + HIGGS hitting 25.6× at +0.09% PPL.) |
| Model plugin accepts a qualified DMS decision source | Schema-v1 checkpoints carry corrected borrowed-query-channel decisions. Schema-v2 hybrid models may carry an exact-hash external linear sidecar over a declared normalized hidden stage and physical compact-layer map; this source preserves ordinary Q channels. |
| `KVCacheBackend` storage views + registered compact attention/store bundles | A resolved `dms_compact+fp8` or later compressed composition routes without engine-wide branches. |

#### Phase 4 DMS delivery

With the Phase-0/C2 groundwork, adding DMS is:

1. **One DMS topology component and backend composition** — compact
   per-layer/head allocation, live-span bookkeeping, pool plans, claims/deltas,
   storage views, and prefix/transaction capabilities. BF16 is qualified first;
   compressed payload codecs replace only this composition's codec/pools/kernels.
2. **Three new HIP kernel families** ported from `~/FastDMS` Triton reference:
   - `dms_rope_store_compact_decode` (fuses RoPE + eviction decision + compact store at decode)
   - `compact_decode_grouped_splitk` (attention over variable per-head live spans)
   - `streaming_pack_scatter` (prefill surviving-K/V pack)
3. **Model-plugin extension**: `DMSRetrofitConfig` loads either strict schema-v1
   borrowed-channel metadata or strict schema-v2 external-sidecar metadata. The
   latter binds model/sidecar hashes, physical layer IDs, hidden input stage,
   tensor shapes, and training provenance before a plugin can resolve it.
4. **No scheduler subclass or glue branch**: the common scheduler consumes the
   DMS backend's claims, maintenance work, transactions, and resource deltas.

The exact LoC depends on allocator and kernel evidence; the architectural win is
not a small estimate but the absence of a second queue, admission algorithm,
continuous batch, output path, or cancellation lifecycle.

#### What's deferred beyond DMS

| Technique | Blocker |
|---|---|
| AQUA-KV cross-layer residual predictor | Needs per-layer scalar quant codec. Research, ~800 LoC if pursued |
| HIGGS 4-bit KV | ~50% BF16 speed in `kvcache-quantization-research/`; defer until kernel faster |
| H2O / SnapKV heavy-hitter | Research; same `KVLiveSpans` fits; ~300 LoC policy |
| StreamingLLM + attention sinks | Phase 3, ~200 LoC policy; no new kernels |
| TurboQuant 4-bit KV | Add a registered dense topology + TurboQuant codec/layout backend composition and kernel bundle if users need it; no scheduler branch |

## Project Structure

```
hipengine/
├── hipengine/
│   ├── __init__.py              # LLM, SamplingParams exports (no torch import)
│   ├── llm.py                   # Main API: LLM.generate()
│   ├── core/                    # Torch-free primitives
│   │   ├── __init__.py
│   │   ├── tensor.py            # hipengine.Tensor + dlpack import/export
│   │   ├── dtype.py             # DType enum (fp16, bf16, fp32, int8, int4_paro, ...)
│   │   ├── device.py            # HIP/CUDA enumeration, context management
│   │   ├── memory.py            # mmap + hipMemcpyAsync, pinned host mem
│   │   ├── stream.py            # hipStream wrapper via ctypes
│   │   ├── graph.py             # hipGraph capture + replay via ctypes
│   │   ├── blas.py              # hipblasLt / cublasLt bindings (ctypes)
│   │   └── build.py             # hipcc/nvcc subprocess JIT, .so hash cache
│   ├── loading/                 # Torch-free loaders (safetensors + HF glue)
│   │   ├── __init__.py
│   │   ├── safetensors_loader.py
│   │   ├── hf_config.py         # JSON + dataclass translation
│   │   ├── chat_template.py     # jinja2 rendering
│   │   └── tokenizer.py         # thin wrapper over `tokenizers` (Rust via pyo3)
│   ├── dispatch/
│   │   ├── __init__.py
│   │   ├── engine.py            # Forward loop, hipGraph capture+replay
│   │   ├── scheduler.py         # Chunked prefill + decode scheduling
│   │   ├── block_manager.py     # Paged backend compatibility adapter
│   │   ├── prefix_cache.py      # RadixCache or prefix_lru
│   │   └── fusion.py            # Op chain -> kernel plan (longest-match)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # ModelPlugin Protocol
│   │   ├── registry.py          # @register_model, HF arch string -> plugin
│   │   ├── qwen3.py             # Qwen3 dense 0.6B / 0.8B / 27B
│   │   ├── qwen3_5.py           # Qwen3.5 hybrid (full + linear_attn + gdn + MoE)
│   │   ├── gemma4.py            # Gemma 4 (sliding + global)
│   │   ├── llama.py             # Llama 3 family
│   │   ├── mistral.py
│   │   └── sansho.py            # Custom arch + DFlash speculative
│   ├── quant/
│   │   ├── __init__.py
│   │   ├── base.py              # QuantPlugin Protocol
│   │   ├── registry.py
│   │   ├── fp16.py / bf16.py
│   │   ├── w8a8.py              # Per-token dynamic int8
│   │   ├── w8a16.py             # Static weight int8
│   │   ├── w4_paro.py           # PARO pack8 + rotation
│   │   └── w4_gguf.py           # Q4_K_M, Q8_0 (future)
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── base.py              # LayerPlugin Protocol
│   │   ├── full_attention.py    # SDPA prefill + paged decode (split-K, warp, int8)
│   │   ├── linear_attention.py  # conv prefill/decode, L2-norm
│   │   ├── gdn.py               # Gated Delta Net (prefill + decode)
│   │   ├── sliding_attention.py # Gemma 4
│   │   ├── moe.py               # Top-K routing + grouped dispatch + experts
│   │   ├── mlp_dense.py         # gate_up + down
│   │   ├── embed_head.py        # embedding + lm_head + sampler
│   │   └── fused_boundary.py    # rmsnorm+rotate, silu_mul_rotate, gate_combine
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── registry.py          # KernelKey + resolve() + self-register imports
│   │   ├── hip_gfx1100/         # W7900/RDNA3 (Phase 0 primary backend)
│   │   │   ├── common/          # helpers.cuh, extension.cpp aggregator
│   │   │   ├── attention/       # full_attn_decode, paged_attn_decode, paged_kv_write
│   │   │   ├── linear_attn/     # conv, gdn
│   │   │   ├── moe/             # router, group_scatter, w8a8_grouped, swiglu
│   │   │   ├── quant/           # w8a8_activation, w8a16_linear, w8a16_moe, paro_awq_*
│   │   │   ├── wmma/            # wmma_i8_gemm
│   │   │   ├── norm/            # rmsnorm
│   │   │   ├── rotary/          # rotary
│   │   │   ├── fused/           # rmsnorm+rotate, silu_mul_rotate, gate_combine_residual
│   │   │   └── smoke/           # smoke.hip
│   │   ├── hip_gfx1151/         # Strix Halo (future)
│   │   ├── cuda_sm86/           # NVIDIA (future)
│   │   └── cpu_reference/       # torch-free numpy baseline (correctness oracle)
│   ├── kvcache/
│   │   ├── __init__.py
│   │   ├── base.py              # KVCache, BlockRange
│   │   ├── policy.py            # Legacy KVPolicy adapter + backend contracts
│   │   ├── radix.py             # RadixCache implementation
│   │   └── offload.py           # KVTC tiered offloading (device -> host -> disk)
│   ├── distributed/             # Multi-GPU (Phase 3+)
│   │   ├── __init__.py
│   │   ├── tp.py                # Tensor parallelism
│   │   ├── pp.py                # Pipeline parallelism
│   │   └── ep.py                # Expert parallelism
│   ├── speculative/
│   │   ├── __init__.py
│   │   ├── base.py              # DraftModel, DraftBatch, Verifier, AcceptResult
│   │   ├── medusa.py
│   │   ├── lookahead.py
│   │   ├── mtp.py               # Qwen3.5 MTP layers
│   │   ├── eagle3.py            # feature-conditioned draft model
│   │   └── dflash.py            # sansho / FastKMS draft acceptance
│   ├── server/                  # OpenAI-compatible API used by `hipengine serve`
│   │   ├── __init__.py
│   │   ├── api.py               # FastAPI app
│   │   ├── chat.py              # /v1/chat/completions
│   │   └── models.py            # /v1/models
│   └── benchmark/
│       ├── __init__.py
│       ├── suite.py             # Unified harness
│       ├── prefill.py
│       ├── decode.py
│       ├── memory.py
│       └── correctness.py       # KL, top-1, PPL fixtures + cpu_reference oracle
├── tests/
│   ├── test_tensor.py           # hipengine.Tensor round-trip + dlpack
│   ├── test_graph_capture.py    # hipGraph via ctypes
│   ├── test_attention_exactness.py
│   ├── test_moe_correctness.py
│   ├── test_quantization.py
│   ├── test_prefix_cache.py
│   └── test_kernel_registry.py  # All (backend, layer, quant) keys resolve
├── scripts/
│   ├── install_rocm.sh
│   ├── audit_kernels.sh         # rocprofv3 wrapper
│   └── smoke.py
├── docs/
│   ├── PLAN.md                  # This file
│   ├── OPTIMIZE.md              # Current Qwen3.5/PARO perf grind plan
│   ├── BENCHMARK.md             # Evidence policy and benchmark procedures
│   ├── KERNELS.md               # Kernel catalog and port playbook
│   ├── PREFILL.md               # Native bulk prefill plan/evidence
│   ├── LAGUNA-prefill.md        # Active Laguna arithmetic/MMQ prefill attack plan
│   ├── LAGUNA-decode.md         # Active Laguna short/long decode attack plan
│   ├── SAMPLING.md              # Normal sampling parameter support plan
│   ├── ROOFLINE.md
│   ├── LESSONS-LEARNED.md
│   └── API.md
├── benchmarks/
│   └── vllm_bench_adapter.py
├── pyproject.toml               # Deps: safetensors, tokenizers, jinja2, numpy, FastAPI/Uvicorn
│                                # Extras: [torch]=torch (dlpack bridge)
└── README.md
```

## Development Roadmap (LoC Estimates)

The current focused Laguna S 2.1 performance campaign is owned by
[`LAGUNA-prefill.md`](LAGUNA-prefill.md). It succeeds the exhausted LPF/AR-O
work in `LAGUNA.md` and keeps the architecture invariants in this file: new
packed-dot MMQ, repair, layout, and attention routes remain four-axis plugins
with exact fallbacks and no backend/quant branches in model or engine code.

The active W7900 / gfx1100 UD-Q2_K_XL short-prefill sequence has retained the
exact WPF-2b expert-major IQ2 gate/up owner. Its local64/pair16 rowbatch8 body
is BF16-bit exact on all 46 actual M512 layers; clean package-resolved 512/1K
moves **99.230/91.559 -> 118.705/107.804 tok/s (+19.626%/+17.743%)**, and
cached tracing cuts gate/up **62.549%/62.850%** without changing dispatch
count. Cleanup has removed the rejected raw-Q5/Q6 MMQ owner plus the unowned
Laguna rowbatch8/fused-SiLU and losing pair16-rowbatch4 diagnostics. The
base/rowbatch4/adaptive/auto grouped-dual keys remain because they independently
own Qwen3.5 GGUF's exact default-on grouped-prefill route. WPF-3 has now
candidate-admitted an exact local32 qrow4 SWA body and C256-qualified policy,
then promoted that policy as the gfx1100 package default. It traces at
VGPR72/LDS0/scratch0; a no-override M512 gate preserves all state at KL0 and a
paired 512/1K gate improves **117.813/106.486 -> 131.044/124.348 tok/s**.
Clean selector-unset publication reaches **131.919/125.960 tok/s**, improving
the preceding exact packet **11.131%/16.842%**; cached tracing cuts SWA
**55.411%/59.449%** and kernel span **9.643%/14.228%**. The separate online-SWA
lane improves complete-suite natural-prompt prefill **117.170 -> 118.335 tok/s
(+0.995%)** but is rejected at maximum KL **0.394600 > 0.05** despite
**564/576** top-1, deterministic repeats, and positive h16/h32 E2E. Exact
qrow4-C256 remains the gfx1100 default; the independently owned gfx1151 online
registrations remain. WPF-1T now promotes an exact Q5/Q6 `coltile` policy as
the gfx1100 package default after all 15 actual M512 configurations are
byte-exact/faster and the 381-invocation-weighted `(4,8)` kernel sum falls
**2699.147 -> 1828.710 ms (1.476x, -32.249%)** versus RB32. It traces at
local128, VGPR72, SGPR50, LDS512, and private0 with zero spills. Four measured
`(quant, output, K, N)` keys use `(2,16)` instead, saving another **36.773 ms
(2.011%)** from that Q5/Q6 family; every other eligible key keeps `(4,8)`. The
frozen seven-pair ownership gate improves **+0.545%/+0.459%** at 512/1K. A
package-path repeat remains exact and positive at **+0.382%/+0.242%** but misses
the repeated 1K `>0.3%` magnitude threshold, so the policy is limited to those
four keys and does not replace the canonical clean headline. The no-override
M512 gate is KL0/bit-exact through all 48 boundaries and full K/V spans; the
original same-weight promotion improves **131.491/124.949 -> 169.046/157.420
tok/s (+28.561%/+25.987%)**. Clean selector-unset publication remains
**169.253/159.229 tok/s (+28.301%/+26.412%)** versus the preceding exact packet;
cached dense/shared projection and kernel span fall **38.546%/38.875%** and
**21.893%/20.852%**. Both short rows clear 150 tok/s, and the restored clean 4K
gate reaches **123.084 tok/s** with deterministic IDs/positions/lifecycle and
full allocation recovery. H5E now supersedes that canonical row with exact
transient-F32 ordered Q5 at **184.997/172.104/131.496 tok/s** through 4K
(**+3.166%/+2.941%/+1.944%** over H5D). The final-source 235-call Q5 stack falls
**12.320%/7.515%** by event/wall clocks. H5F adds exact 12x4 only for F32 N48.
H5G retains exact 8x10/16x5/8x12/12x8 on five roles, cuts the strong H5F subset
**8.639%/7.479%** by event/wall, and publishes
**188.393/175.042/132.743 tok/s (+2.192%/+2.055%/+1.329%)** over H5F. H5I
reuses the same plane for four exact-Q6 roles, cuts traced Q6 **177.047 ->
110.170 ms (-37.774%)**, and publishes **191.713/178.080/134.411 tok/s
(+1.762%/+1.736%/+1.256%)** over H5G with KL0/byte-exact complete state and no
new allocation. Explicit RB32/raw-coltile and unsupported widths remain exact
fallbacks; gfx1151 excludes the W7900 keys.

A same-host direct-M512 refresh now fixes the next external target. Identical
512 token IDs, context4096 admission, direct M512, FlashAttention, BF16 K/V,
and one last-row projection measure hipEngine **169.228 tok/s** versus
same-revision llama.cpp HIP **694.184 tok/s (4.102x)**; both select first token
2930. Same-revision Vulkan is only **56.274 tok/s** with native F16 K/V, so HIP,
not Vulkan, is the active prefill comparator. An exhaustive disjoint module
ledger reconciles the **3,009.837 vs 724.299 ms** kernel sums and isolates Q5_K
**1,215.391 ms (53.18%, 21.617x body ratio)**, attention **469.194 ms (20.53%,
22.597x)**, IQ3_XXS selected down **379.034 ms (16.58%, 3.487x)**, and Q6_K
**141.195 ms (6.18%, 10.466x)**. Those four rows account for **2,204.814 ms /
96.47%** of the complete gap. llama.cpp launches **2,824 vs 1,477** kernels,
excluding launch count as the primary cause. Source audit at `c0bc8591e` shows
F32-to-Q8_1 256-thread 128x128/K256 WMMA MMQ for Q5/IQ families, bounded Q6
dequantization/casts plus F16 rocBLAS at M512, device expert compaction, and
`flash_attn_ext_f16<128,128,8,8>` plus stream-K fixup.

WPF-H1's primitive is admitted but its runtime route is rejected. The strict
byte-exact DS4 producer plus isolated fast-math I128/J128/K256 consumer moves
the actual eight-role/235-call M512 leaf **1,562.932 -> 97.110 ms (16.094x)**
and complete-suite natural-prompt prefill **151.252 -> 203.862 tok/s (1.348x)**.
The mandatory 18-prompt/576-step lane nevertheless reaches maximum KL
**4.162014 > 0.05** at **561/576 (97.396%)** top-1. Poolside, deterministic
repeats, every performance clause, and lifecycle pass, but cannot waive KL.
The temporary runtime owner/workspace/switch is removed; production remains the
exact role-qualified coltile path and the source-Q5 primitive stays explicit
ceiling evidence only.

WPF-H2's standalone source-faithful F16-WMMA body reaches leaf parity but is
rejected for runtime use. It keeps BF16 resident K/V and complete
`KVLiveSpans`, and the 12-global/36-SWA M512 family moves **490.919 -> 21.719 ms
(22.603x)** versus llama.cpp's matched **21.725-ms** main+fixup trace. The
binding 18-prompt/576-step lane nevertheless reaches maximum KL **1.804860 >
0.05** at **564/576** top-1 despite deterministic repeats, lifecycle recovery,
and diagnostic natural-prompt prefill **152.087 -> 156.219 tok/s (1.027x)**.
F32 PV, full-attention-only, and SWA-only followups also fail. Remove the runtime
owner/selector and retain only the separately registered corrected primitive as
ceiling evidence; exact qrow4/M128 remains production.

WPF-H3's standalone gfx1100 primitive remains admitted, but runtime promotion
is rejected. The strict DS4 producer plus raw-IQ I128/J128/K256 consumer moves
all 45 IQ3_XXS and two IQ4_XS actual M512 selected-down layers **565.437 ->
115.951 ms (4.877x)**; IQ3 alone is **27.145% below** llama.cpp's matched
**152.380-ms** family trace. Complete-suite natural-prompt prefill improves
**152.276 -> 181.556 tok/s (1.192x)**, but the binding 18-prompt/576-step lane
reaches maximum KL **0.373028 > 0.05** at **567/576** top-1. An IQ3-source,
IQ4-exact structural followup reaches maximum KL **0.372917**, isolating source
IQ3 arithmetic rather than the two IQ4 layers. Poolside, deterministic repeats,
complete M512 state, and lifecycle pass but cannot waive KL. Remove the runtime
owner/selector/tile128 metadata route and retain exact grouped production plus
the separately registered spill-free VGPR152/248 leaf; gfx1151 stays excluded.

WPF-H4's standalone source-faithful Q6_K F16/rocBLAS leaf beats its matched
family comparator, but runtime promotion is rejected. A fused local64
raw-Q6/BF16 producer plus F16-compute `rocblas_gemm_ex` and one output cast
moves the actual six-shape/144-call M512 family **174.351 -> 14.349 ms
(12.151x)**, **3.825% below** llama.cpp's **14.919865-ms** stack. Complete-suite
natural-prompt prefill improves **151.784 -> 158.205 tok/s (1.042x)** with every
category positive, but the binding changed-arithmetic lane reaches maximum KL
**0.338657 > 0.05** at **567/576** top-1. Poolside, deterministic repeats,
complete M512 state, and lifecycle pass but cannot waive KL. Remove the runtime
owner/selector/rocBLAS handle/**97,517,568-byte** workspace/package capabilities
and retain exact coltile production plus the separately registered leaf.

These audited llama.cpp routes remain measured ceiling evidence, not production
fallbacks that can bypass quality. Keep source commit/path attribution, the
four-axis registry, raw-pointer kernel ABI, `KVLiveSpans`, and registered exact
fallbacks. H1 source-Q5, H2 source attention/stream-K, H3 source-IQ, H4
source-Q6, P6, WPF-1R, D4/D8/D8R8, and online-SWA rejections remain closed.
WPF-H5's clean exact-production M512 reprofile is complete at **169.516 tok/s**
versus the matched llama.cpp HIP **694.184 tok/s (4.095x)**. Cached tracing
records **3,001.692-ms** kernel sum in a **3,016.780-ms** span across **1,477**
dispatches; only **15.087 ms / 0.500%** lies outside kernels. Exact Q5 coltile
remains first at **1,270.458 ms / 42.325%**, ahead of selected IQ3/IQ4 down
**557.091 ms**, attention **488.304 ms**, gate/up **460.143 ms**, and Q6
**157.073 ms**. WPF-H5A's standalone bounded raw-Q5-to-F32/BF16-to-F32/SGEMM
leaf is now admitted: exact fallback for the regressive N48 role plus the F32
candidate elsewhere moves the actual 235-call family **1,256.936 -> 221.137 ms
(5.684x)** by events and **1,223.263 -> 231.966 ms (5.273x)** by synchronized
wall. Raw operand values are exact and candidate output passes at max mean KL
**1.59e-9**, max-row KL **5.79e-8**, and top-1 **100%**. The stack still costs
**3.751x** llama.cpp's matched Q5 trace. Its default-off owner passes natural
M512 at KL **0.0003742**, top-1 **100%**, deterministic complete state, and exact
teardown, but the binding 18-prompt/576-step lane rejects SGEMM reassociation at
maximum KL **1.143627 > 0.05** despite **564/576 (97.917%)** top-1 and diagnostic
prefill **152.359 -> 202.707 tok/s (1.330x)**. Remove the owner, workspace,
capabilities, and tests; retain exact production plus the standalone leaf. H5B's
existing complete-`KVLiveSpans` F32 dense-initial hipBLASLt route passes the
W7900 transfer screen: tuned packed/wave leaf **109.897 -> 62.655 ms (1.754x)**,
natural M512 KL **0.000429** / top-1 **100%** with deterministic complete state,
and cached attention **488.304 -> 60.669 ms (8.049x)** while full kernel sum
falls **3,001.692 -> 2,603.520 ms (-13.265%)**. The binding extended-prompt lane
observes all **10,512** expected package-mapped launches, but rejects QK/PV
reassociation at maximum KL **0.444675 > 0.05** despite **564/576 (97.917%)**
top-1 and diagnostic prefill **165.555 -> 190.103 tok/s (1.148x)** with every
category positive. Remove the gfx1100 capability/component policy, heuristic
map, generic map seam, owner propagation, and tests; retain exact qrow4/M128
production plus standalone leaf evidence. H5C/H5D returns to exact Q5
arithmetic with transient exact-value expansion plus local128 ordered 8x4/4x8
consumers. H5E extends the identical per-output K/FMA/wave/store sequence to
4x16/8x8/16x4 and removes regressive 1x64/2x32. The final-source 235-call policy
moves H5D **1,085.630 -> 951.876 ms (1.141x)** by events and **1,040.166 ->
961.993 ms (1.081x)** by wall. One bounded **150,994,944-byte** projection-local
plane adds no sidecar. Package-default M512 is KL0/byte-exact across all 48
boundaries/logits/KV/repeat/lifecycle, and selector-unset production publishes
**184.997/172.104/131.496 tok/s** through 4K. H5F's constant-48 screen retains
only 12x4 for F32 N48. H5G's exact constant-80/96 tiles own five roles, trace at
VGPR168/200 with zero scratch, and publish **188.393/175.042/132.743 tok/s**.
H5H closes larger exact tiles: constant-112 is scratch-free at VGPR232 but loses
every role; constant-128 reaches VGPR256/28–52 B scratch and also loses every
role. All candidates are removed. The retained H5G request segment now
reconciles **2,667.034 ms / 1,720 dispatches** in a **2,702.091-ms** kernel
span: Q5 is **920.633 ms**, IQ3/IQ4 down **560.642 ms**, gate/up **470.116
ms**, attention **468.533 ms**, Q6 **177.047 ms**, and all remaining kernels
**70.063 ms**. Q5 geometry and prior attention lanes are closed. WPF-H5I's
exact raw-Q6-to-F32 producer plus ordered consumer clears the all-role leaf:
strong 146-call event/wall sums move **194.758/189.722 -> 119.751/121.353 ms
(-38.513%/-36.037%)**. Four roles select `16x5`/`16x4`/`8x4`; both long-K
roles and the wide-N F32 role remain exact raw coltile. Complete M512 state is
KL0/byte-exact with the unchanged serial plane. Cached tracing records **143+143**
candidate launches and three fallbacks, cuts Q6 **177.047 -> 110.170 ms**, and
cuts request kernel sum **2,667.034 -> 2,600.260 ms**. Clean selector-unset
512/1K/4K promotes **191.713/178.080/134.411 tok/s
(+1.762%/+1.736%/+1.256%)** over H5G. The reconciled H5I request is Q5
**922.619 ms**, exact IQ3/IQ4 down **556.749 ms**, attention **471.150 ms**,
gate/up **469.311 ms**, Q6 **110.170 ms**, and remaining **70.261 ms**. Against
matched llama.cpp, IQ down retains a **401.254-ms** gap. H5J admits exact
resident-segment IQ3 plus an IQ4 local32 launch of the retained physical body.
A generated one-BF16-ULP RED removes the first separately compiled IQ4 body.
The final actual-weight/routing gate is byte-exact and both-clock positive on
all **45+2** layers: IQ3 moves **541.137 -> 491.481 ms (-9.176%)**, IQ4
**26.137 -> 8.696 ms (-66.730%)**, and combined event/wall sums move
**567.274/567.056 -> 500.176/500.448 ms (-11.828%/-11.746%)**. Complete M512
state is KL0/byte-exact through all 48 boundaries, logits, K/V/live spans,
repeat, and teardown. Cached integrated tracing selects exactly **45+2** calls,
moves selected down **556.749 -> 497.145 ms (-10.706%)**, and cuts request
kernel sum **2,600.260 -> 2,532.020 ms (-2.624%)** at unchanged **1,862**
dispatches. Clean selector-unset 512/1K/4K promotes
**196.103/181.859/137.169 tok/s (+2.290%/+2.122%/+2.052%)** over H5I with no
new allocation/workspace/sidecar; all misses and gfx1151 retain exact fallback.
The matched M512 gap is now **3.540x**. H5K closes larger resident IQ3 row
ownership: scratch-free rowbatch12 loses all 45 layers at **+6.893%/+5.771%**
event/wall, while rowbatch16 worsens to **+10.770%/+9.870%**; every byte and
lifecycle matches and all temporary surfaces are removed. The unchanged H5J
request reconciles Q5 **919.697 ms**, IQ down **497.145**, attention **468.007**,
gate/up **466.826**, Q6 **110.293**, and remaining **70.051 ms**. Q5's 235
ordered consumers own **904.399 ms**; BF16 K9216/N3072 plus F32 K3072/N9216
alone contribute **741.721 ms (82.0%)**. WPF-H5L admits an exact weight-tile-
major mapping: linear row-group-inside-output-tile ownership changes no scalar
FMA, wave reduction, store, F32 plane, or fallback. Six material roles qualify;
F32 N48/N72 retain H5G after N48 loses wall and N72's marginal first result
turns mixed-clock on the final-source rerun. Across all **235** calls, final-
source event/wall sums move **882.963/887.364 -> 486.892/474.348 ms
(-44.857%/-46.544%)** with exact bytes, lifecycle recovery, and unchanged
local128/VGPR72-200/LDS512-1536/scratch0 classes. Complete M512 state is
KL0/byte-exact through all 48 boundaries, logits, K/V/live spans, repeat, and
teardown with the unchanged **150,994,944-byte** plane. Cached integrated
tracing physically selects **235** producers, **188** candidates, and **47** H5G
fallbacks: Q5 falls **919.697 -> 466.986 ms (-49.224%)** and request kernel sum
**2,532.020 -> 2,074.261 ms (-18.079%)** at unchanged **1,862** dispatches.
Clean package-default 512/1K/4K promotes **237.956/217.888/157.366 tok/s
(+21.342%/+19.812%/+14.725% over H5J)** and narrows matched M512 to **2.917x**.
Every miss and gfx1151 retain the preceding exact route. The post-H5L request
reconciles **2,074.261 ms / 1,862 dispatches** in a **2,100.389-ms** span.
Matched gaps rank attention **437.720 ms**, Q5 **408.035**, and IQ down
**338.619**; exact SWA qrow4 owns **268.720 ms / 58.49%** of attention. WPF-H5M
promotes source-qualified qrow4 loads while preserving logical-slot/four-row
order, BF16 source rounding, the reconstructed dot tree, two-pass maximum/
denominator/PV order, stores, KV schedule, and fallback. Dense and wrapped/
evicted/ragged outputs are bit-exact, and starts 256/384 improve event/wall
**4.324%/4.354%**. Complete M512 state is KL0/byte-exact across all 48
boundaries, logits, K/V/live spans, repeat, and teardown. Cached tracing selects
exactly **48 global + 72 wave32 + 72 H5M** calls: qrow4 falls **268.720 ->
260.500 ms (-3.059%)**, attention **459.445 -> 450.790 (-1.884%)**, and request
sum **2,074.261 -> 2,060.485 (-0.664%)** at unchanged **1,862** dispatches.
Clean package-default 512/1K/4K promotes **238.565/218.182/158.138 tok/s
(+0.256%/+0.135%/+0.490% over H5L)** and narrows matched M512 to **2.90983x**
with no allocation or sidecar. The production-identical post-H5M request
reconciles **2,060.485 ms / 1,862 dispatches** in a **2,086.586-ms** span.
Matched gaps rank attention **429.065 ms**, Q5 **406.709**, and IQ down
**336.162**. Attention splits into global local256 **80.707 ms**, exact SWA
wave32 **109.583**, and exact source-qualified qrow4 **260.500**; qrow4 remains
**57.79%** of the family at starts 256/384. WPF-H5N's separately registered
exact dense-first-fill qrow4 leaf derives position/visibility from the proven
identity, no-wrap initial ring while preserving cached `base_offsets`, complete
`KVLiveSpans`, attend-before-append scheduling, logical-slot/four-row order,
BF16 rounding, dot tree, two-pass maximum/denominator/PV order, and stores. It
matches H5M and wave32 bytes, retains local32/VGPR72/SGPR128/LDS0/scratch0, and
wins event/wall at start 256 **1.147x/1.144x** and start 384 **1.166x/1.163x**;
combined sums improve **6.653/6.660 -> 5.744/5.762 ms (1.158x/1.156x)**.
Complete M512 state is KL0 and integrated tracing selects all 72 H5N calls,
cutting qrow4/attention/request sum **13.918%/8.087%/1.687%**. Reject runtime
ownership despite those exact wins: clean 4K is **-0.217%**, and a seven-repeat
adjudication confirms **7/7** H5N samples below H5M (**158.152 -> 157.832 tok/s,
-0.202%**). Remove the temporary policy extension, retain the standalone leaf,
and keep H5M production. WPF-H5O then targets Q5's retained **465.660-ms**
family/**406.709-ms** matched gap with a 320-byte exact-factor block instead of
1,024 F32 bytes. Every reconstructed F32 weight bit and rows17/33 role output
matches H5L/H5G, and cached tracing is scratch-free at producer/expand VGPR16
plus consumer VGPR80-200. The logical-byte model does not survive measurement:
**0/8** actual roles win both clocks and producer-inclusive event/wall sums
regress **477.022/473.054 -> 606.780/614.512 ms (+27.202%/+29.903%)**.
Coefficient loads and reconstruction ALU dominate. Remove all H5O symbols, keys,
and tests; keep H5L/H5G and gfx1151 unchanged, and do not retry this
representation without a distinct operation-count premise. WPF-H5P then
cross-screens H5F's exact 64-accumulator 4x16/16x4/8x8 geometries under H5L's
later weight-major traversal. Four of five roles lose at least one clock and all
of their candidate surfaces are removed. BF16 K6144/N3072 `16x4` is the sole
final-source winner: cached tracing confirms local128/VGPR136/SGPR128/LDS1024/
scratch0 versus H5L `16x5` VGPR168/LDS1536, and its unchanged-producer 12-call
event/wall sums fall **31.306/30.890 -> 29.329/29.898 ms
(-6.315%/-3.211%)** with byte-exact output and recovered allocations. The
bounded default-off owner passes complete M512 state at KL0/byte identity and
tracing selects exactly **12** calls, cutting role/Q5/request sum
**5.800%/0.572%/0.187%**. The first clean 512 result is **-0.189%**, but a
predeclared seven-repeat adjudication resolves it at **+0.176%**. The
source-default publication is **+0.093%/-0.019%/-0.054%** at 512/1K/4K and its
final frozen 1K/4K adjudication remains **-0.030%/+0.014%**. Reject runtime
ownership under the all-length rule; remove the eager owner/package change,
retain only the exact leaf, and keep H5M/H5L production. WPF-H5Q addresses
the third-largest matched residual, IQ3/IQ4 down at **491.658 ms** versus
llama.cpp HIP **155.495 ms**. P64/P128 alone win all **45/45** actual IQ3
layers on both clocks; the predeclared max-min rule retains P64. Final-source
event/wall sums fall **492.847/491.518 -> 481.081/483.823 ms
(-2.387%/-1.565%)**, with H5J byte identity, sampled CPU-oracle agreement,
local128/VGPR48/SGPR128/LDS512/scratch0 resources, unchanged metadata/
allocation, and gfx1151 fail-closed. Complete M512 state is KL0/byte-exact;
integrated tracing selects all **45** P64 IQ3 calls and cuts IQ-down/request sum
**3.255%/0.491%**. Default-off clean 512/1K/4K improves
**+0.702%/+0.278%/+0.370%**, with 3/3 paired wins at every length. Selector-unset
publication confirms **+0.663%/+0.355%/+0.267%**, again 3/3 paired wins each,
and promotes **239.981/219.494/158.693 tok/s**. One bounded gfx1100 IQ3
variant+ABI entry changes; H5J remains fallback, IQ4 is unchanged, and gfx1151
stays fail-closed. The production-identical post-H5Q trace reconciles
**2,050.376 ms / 1,862 dispatches** against matched llama.cpp HIP **724.299
ms**, leaving gaps attention **431.450 ms**, Q5 **409.559 ms**, IQ down
**320.157 ms**, and gate/up **59.253 ms**. WPF-H5R screens the largest lane
through a distinct exact schedule/body pair: reuse the existing safe
append-before-attend launch order for complete M128 tiles, then consume only the
preappended BF16 cache in separately registered two-pass global/SWA qrow4
kernels. Both preserve production bytes and full `KVLiveSpans`. Global must
reconstruct local256's dot/denominator/normalized-PV association; it reaches
local32/VGPR248/LDS8192/scratch0 and loses every start at **0.636–0.926x** on
both clocks, so remove its export/key/exclusion/test case. The retained SWA body
is local32/VGPR64/LDS0/scratch0 and wins starts 0/128/256/384 independently.
Including equal append cost, its actual 144-call event/wall sums fall
**337.277/334.031 -> 126.687/125.764 ms (-62.438%/-62.350%, 2.662x/2.656x)**.
It adds no launch, allocation, workspace, or sidecar. Retain attend-before-
append H5M/wave32 routes for every partial, wrapped, staged-verifier, explicit,
missing, or unsupported case. Complete M512 state is KL0/byte-exact; integrated
corrected one-queue tracing records all **144** write->H5R pairs at unchanged
**1,862** dispatches and cuts the SWA schedule/request sum **63.767%/9.690%**.
Selector-unset one-queue 512/1K/4K improves **+11.340%/+4.848%/+0.746%**, with
3/3 paired wins each and unchanged ownership, promoting
**267.205/230.441/160.221 tok/s**. Earlier uncapped speed rows are superseded
([H5R production](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-swa-preappend-cached-exact-production.json) ·
[H5R SWA leaf](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-swa-preappend-cached-exact-candidate.json) ·
[post-H5Q residual / H5R target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5q-residual.json)).
The production-identical post-H5R one-queue request reconciles **1,851.695 ms /
1,862 dispatches** in a **1,877.998-ms** span against matched llama.cpp HIP
**724.299 ms**. Exact gaps now rank Q5 **423.388 ms**, IQ down **332.278 ms**,
attention **195.796 ms**, Q6 **106.386 ms**, and gate/up **65.602 ms**; launch/
submission residue stays below trigger at **26.303 ms / 1.401%**. Select
**WPF-H5S exact persistent row-group Q5 traversal**. Separately registered
fixed partitions **1/2/4/8/16/32** preserve H5L's F32 plane, role geometry,
per-thread K/`fmaf`/wave/serial-sum/store order, launch count, workspace, and
fallbacks. Rows17/33/M512 and actual-role outputs are byte-exact; all 36 cached
symbols are local128/SGPR128/scratch0 with only +8 VGPR. Performance rejects
every partition on every role. Best aggregate P32 moves producer-inclusive
event/wall **459.018/473.034 -> 565.864/566.290 ms
(+23.277%/+19.714%)**; **0/6** roles wins both clocks. Remove all candidate
surfaces and do not infer speed from the P1 **70.68x** workgroup-prologue model
([H5S rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-persistent-row-group-rejected.json) ·
[post-H5R residual / H5S target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5r-residual.json)).
**WPF-H5T exact IQ3 one-wave K-partition collapse** maps H5Q logical lanes
`i/i+32/i+64/i+96` onto physical lane `i`, preserving P64, rowbatch8, decode,
four independent FMA/shuffle trees, serial 0..3 sum, and BF16 store while
removing LDS/barriers. The final named-register body is byte-exact and
local32/VGPR96/LDS0/scratch0. Actual-weight timing rejects it: event/wall move
**474.107/485.298 -> 475.945/469.677 ms (+0.388%/-3.219%)**, with only
**12/45** both-clock-positive layers. Remove all surfaces and retain H5Q; do
not promote a wall-only result
([H5T rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-one-wave-k-partitions-rejected.json) ·
[H5T target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-one-wave-k-partitions-target.json)).
**WPF-H5U exact global preappend cached-source local256** retains only its
standalone leaf. All four starts are byte/CPU exact and weighted event/wall moves
**101.535/101.899 -> 84.124/84.622 ms (-17.148%/-16.955%)**. Default-off runtime
qualification is exact: M512/C4096 is KL0, physical tracing records **48 H5U +
144 H5R** pairs at unchanged **1,862** dispatches, global schedule falls
**15.494%**, and matched M512 improves **268.331 -> 270.610 tok/s (+0.849%, 5/5
wins)**. Source-default ownership is rejected because the binding balanced
role-ineligible 1K adjudication is **230.181 -> 230.175 tok/s (-0.00257%, 2/8
wins)**. Remove the global map/resolver/runner/test seam, retain the leaf, and
keep production **267.205/230.441/160.221 tok/s**
([H5U runtime rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-global-preappend-cached-source-runtime-rejected.json) ·
[H5U leaf](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-global-preappend-cached-source-candidate.json) ·
[H5U target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-global-preappend-cached-source-target.json)).
**WPF-H5V exact Q5 one-wave sequential K-partition replay** preserves all six
H5L role outputs byte-for-byte at rows17/33/M512. Cached symbols are
local32/SGPR128/scratch0 with unchanged LDS and only +8 VGPR. The binding
producer-inclusive 188-call screen nevertheless rejects every role on both
clocks: weighted event/wall regresses **464.968/466.267 -> 492.423/493.754 ms
(+5.905%/+5.895%, 0.944x/0.944x)**. Sequential replay removes a block barrier
but loses four-wave K parallelism without reducing useful dot work. Remove the
body, exports, wrappers, registry keys, gfx1151 exclusions, and focused test;
retain H5L/H5G and do not retry this schedule without a new operation or
cross-tile reuse premise
([H5V rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-one-wave-k-partitions-rejected.json) ·
[H5V target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-one-wave-k-partitions-target.json)).
**WPF-H5W exact Q6 weight-major composite reuse** admits exactly three gfx1100
Q6 wrappers/keys over already-retained local128 16x5-BF16, 16x4-BF16, and
16x5-F32 physical primitives. It adds no HIP body/symbol, launch, allocation,
workspace, sidecar, or package policy. Rows17/33 and actual M512 outputs remain
byte-exact. Cached tracing records each exact Q6 producer immediately before the
expected VGPR136-168/LDS1024-1536/scratch0 consumer and exact grid. All three
final-source roles win both clocks; producer-inclusive weighted event/wall falls
**87.859/81.559 -> 70.756/67.795 ms (-19.466%/-16.876%)** across **142/143**
H5I-selected calls. Default-off runtime qualification is KL0/byte-exact across
all 48 boundaries and complete state. Cached integration records exact
**142 H5W + one H5I + three raw** consumers at unchanged **1,862** request /
**289** Q6 dispatches and moves Q6/request sum **121.306/1,851.695 ->
92.636/1,803.036 ms (-23.635%/-2.628%)**. Default-off clean 512/1K/4K improves
**266.814/230.134/159.970 -> 271.697/233.568/161.668 tok/s
(+1.830%/+1.492%/+1.061%)**, 3/3 wins each. Selector-unset publication confirms
**266.763/230.491/160.091 -> 271.526/234.020/161.853 tok/s
(+1.785%/+1.532%/+1.100%)**, again 3/3 each. Promote H5W at canonical
**271.526/234.020/161.853 tok/s (+1.617%/+1.553%/+1.018% over H5R)** and narrow
matched M512 **2.59795x -> 2.55661x**. Preserve H5I F32-N72 and raw long-K/wide-N
fallbacks
([H5W production](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-weight-major-production.json) ·
[H5W candidate](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-weight-major-candidate.json) ·
[H5W target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-weight-major-target.json)).
The production-identical H5W trace now reconciles **1,803.036 ms / 1,862
dispatches** versus matched llama.cpp HIP **724.299 ms** and ranks exact gaps
Q5/IQ-down/attention/Q6/gate-up at **417.482/327.846/192.029/77.716/60.898
ms**. **WPF-H5X exact tile-K-col F32 AoSoA Q5** now admits a standalone leaf
for Q5's unchanged **476.433-ms** family without reopening H5O compression or
H5P/H5S/H5V ownership. Rows17/33 plane bits and all six actual M512 outputs are
exact. The retained linear local256 producer writes full-F32
`[tile][k][col]`; matching local128 consumers preserve every H5L
FMA/reduction/store and physical VGPR/LDS/scratch0 while ISA realizes
**8/12/16 `global_load_b32` -> 2/3/4 `global_load_b128`**. Four roles / **151
calls** win both clocks. Remove the two losing BF16 surfaces and retain H5L for
**37** calls. The six-role selected event/wall model falls
**465.863/467.511 -> 458.615/459.712 ms (-1.556%/-1.668%)**; final-source
winners fall **265.784/266.992 -> 258.653/258.959 (-2.683%/-3.009%)** with 4/4
wins. Value bytes, useful work, the **150,994,944-byte** workspace, launches,
allocation, package policy, and gfx1151 remain unchanged. Default-off
natural-M512 state is KL0 and byte-exact across all 48 boundaries, complete
logits/KV/`KVLiveSpans`, and repeat. Four counter-rotated cached request
segments record exact **151 H5X + 37 H5L + 47 H5G** ownership at unchanged
**1,862/470** request/Q5 dispatches and move median Q5/request/span
**479.776/1,826.542/1,850.682 -> 470.606/1,814.537/1,834.282 ms
(-1.911%/-0.657%/-0.886%)**. Clean one-queue 512/1K/4K improves
**271.744/233.742/161.579 -> 272.936/234.834/162.416 tok/s
(+0.439%/+0.468%/+0.518%)**, 3/3 wins each. Selector-unset publication confirms
**271.922/234.334/162.004 -> 273.366/235.061/162.533 tok/s
(+0.531%/+0.310%/+0.327%)**, again 3/3 each. Promote H5X at canonical
**273.366/235.061/162.533 tok/s (+0.678%/+0.445%/+0.421% over H5W)** and narrow
matched M512 **2.55661x -> 2.53940x**. Retain four eager aliases and promoted
role entries; preserve two H5L roles, N48/N72 H5G, all Q6 routes, and every
miss. The corrected production H5X external-comparator trace uses C4096/direct
M512 rather than the earlier C512 integration allocation. Five exact wall
samples yield **278.062 tok/s (+64.03% over campaign-start 169.516)** and narrow
llama.cpp HIP **694.184** to **2.49651x**. Five request traces reconcile
**1,831.568 ms / 1,862 dispatches** versus llama.cpp **724.299 ms**; gaps rank
Q5/IQ-down/attention/Q6/gate-up at **407.137/326.998/234.055/77.436/59.236
ms**. Q5's rowbatch8/10 roles own **346.501 ms**, and current ISA still issues
one scalar BF16 activation load per logical row. Select **WPF-H5Y exact
tile-K-row BF16 activation AoSoA**: a bounded projection-local bit-copy plane
feeds width-matched aligned records while every H5X/H5L weight layout,
geometry, four-wave K/FMA/reduction/store boundary, and fallback remains fixed.
The static six-role load model is **4.521B -> 0.920B (-79.65%)**. H5Y now
admits all six standalone leaves: rows17/33/512 planes and outputs are exact,
physical loads are width-matched with unchanged consumer resources/scratch0,
and the **188-call** pack-inclusive event/wall aggregate falls
**462.608/455.971 -> 263.014/274.237 ms (-43.145%/-39.856%)** with 6/6
both-clock wins. The bounded **161,120,256-byte** default-off owner now passes
complete M512 at KL0/byte-exact across all state and repeat. Paired tracing
records **188 packs + 235 weight producers + 188 H5Y + 47 H5G** and cuts
Q5/request/span **47.204%/9.685%/9.770%**. Default-off 512/1K/4K improves
**10.939%/9.051%/5.920%**, 3/3 wins each. Selector-unset confirms
**10.862%/8.969%/5.829%**, again 3/3 each. Promote H5Y at canonical
**303.140/256.139/171.830 tok/s (+10.892%/+8.967%/+5.720% over H5X)**.
Matched C4096/direct-M512 reaches **306.305 tok/s / 1,658.386-ms** kernel sum
and narrows llama.cpp HIP to **2.26632x**. Residual gaps rank IQ-down/attention/
Q5 at **339.558/239.624/188.153 ms**; exact IQ3 alone is **486.381 ms / 45
calls**. **WPF-H5Z exact IQ3 activation-resident output-column sweep** is
admitted as a standalone P256 leaf. It preserves H5Q P64, local128/four-wave K, rowbatch8
rows/tails, scalar FMA/reduction/store order, metadata, allocation, and fallback
while retaining one K8 activation tile across sequential outputs. All five
P32/P64/P128/P256/P512 candidates are byte-exact; only P256/P512 win all
**45/45** actual IQ3 layers on both clocks, and the frozen max-min rule keeps
P256. Selection event/wall falls **481.013/487.809 -> 454.128/455.001 ms
(-5.589%/-6.725%)**. Final-source P256 confirms **478.606/486.167 ->
459.818/451.737 ms (-3.926%/-7.082%)**, token 2930, and lifecycle recovery.
Cached P256 is local128/VGPR112/SGPR128/LDS512/scratch0 with eight b128
activation records before the output loop and unchanged 2-d16/3-b32 IQ3
records. Remove the other four instantiations. A bounded default-off owner now
reuses H5Q's active-expert ABI with no allocation/workspace change. Natural
M512 is KL0/byte-exact across all state and repeat. Four paired cached requests
preserve **2,050** dispatches and exact **45 H5Q or 45 H5Z + two H5J IQ4**
topology, moving IQ3/request/span **488.610/1,625.126/1,650.283 ->
477.168/1,603.812/1,624.882 ms (-2.342%/-1.312%/-1.539%)**. Default-off
512/1K/4K improves **302.425/256.139/171.930 -> 307.870/259.556/173.477 tok/s
(+1.801%/+1.334%/+0.900%)**, 3/3 wins each. Selector-unset confirms
**302.160/256.226/172.061 -> 307.658/259.947/173.562 tok/s
(+1.819%/+1.452%/+0.872%)**, again 3/3. Promote H5Y/H5Z at canonical
**307.658/259.947/173.562 tok/s (+1.490%/+1.486%/+1.008% over H5Y/H5Q)**,
narrowing the canonical M512 gap **2.28998x -> 2.25635x**. The binding H5Z
C4096/direct-M512 reprofile reaches **311.622 tok/s** from five exact token-2930,
lifecycle-clean samples, **+83.83%** over campaign start and **2.22765x** behind
llama.cpp HIP. Five production traces reconcile **1,628.336 ms / 2,050
dispatches** in a **1,651.364-ms** median span. Gaps rank IQ-down/attention/Q5/
gate-up/Q6 at **325.570/235.310/182.882/78.514/77.504 ms**. IQ3 remains first
at **472.416 ms / 45 calls**, but H5J/H5K/H5Q/H5T/H5Z, output tiling, and source
MMQ already close the immediate exact ownership/geometry premises. **WPF-H6A
exact dense-initial cached-only attention metadata elision** now qualifies a
bounded default-off owner for its H5R-derived SWA and H5U-derived global leaves.
Natural M512 is KL0 and byte-exact across all 48 boundaries, complete logits/KV/
`KVLiveSpans`, and repeat at unchanged **161,120,256-byte** workspace. Four
paired cached requests preserve **2,050** dispatches and exact **48 H6A global +
144 H6A SWA** write-before-attention topology, moving attention schedule/request-
sum/span **254.976/1,627.696/1,653.806 -> 170.086/1,560.817/1,581.621 ms
(-33.294%/-4.109%/-4.365%)**. Resources stay global local256/VGPR40/scratch0
and SWA local32/VGPR64/scratch0. Default-off 512/1K/4K improves
**307.071/259.710/173.388 -> 312.331/261.467/173.954 tok/s
(+1.713%/+0.677%/+0.326%)**, 3/3 wins each. Selector-unset confirms
**307.158/260.161/173.375 -> 312.781/261.591/173.997 tok/s
(+1.831%/+0.550%/+0.359%)**, again 3/3. Promote H6A at canonical
**312.781/261.591/173.997 tok/s (+1.665%/+0.633%/+0.251% over H5R/H5Y/H5Z)**.
The binding post-H6A C4096/direct-M512 row is **326.174 tok/s** from five exact
samples, **+92.414%** over campaign start and **+4.670%** over matched H5Z. The
llama.cpp comparator audit supersedes the old launcher-only-bound row as
synthetic: a clean c0bc8591 patched rebuild, implementation hash, and **5/5
2930** markers measure exact natural/C4096/BF16 llama.cpp HIP at **696.342
tok/s**; synthetic pp512 is **711.410 tok/s**, consistent with the user's
**714.07**. The exact matched gap is **2.13488x**. Current/llama kernel sums are
**1,568.190/718.241 ms**; current residuals rank IQ-down/Q5/attention/gate-up/Q6
at **336.609/187.223/147.249/93.203/79.112 ms**.

**WPF-H6B exact active-IQ3 signed-magnitude segment plane** implements the
materially different data-layout screen selected after H6A. Complete reordered
16-byte records match the pinned IQ3 decode bytes; P64/P65/tail/empty outputs
match H5Z and the independent CPU oracle; all **45/45** actual-layer outputs are
byte-exact. The producer-inclusive binding screen rejects the path on every
layer and both clocks: H5Z -> H6B event/wall moves
**462.301/450.204 -> 575.804/587.342 ms (+24.552%/+30.461%)**, with **0/45**
wins. The producer is local256/VGPR24/LDS0/scratch0; the local128/VGPR104/
LDS512/scratch0 consumer compiles one b96 load because dead padding is elided,
missing the frozen b128 physical contract. Remove every candidate source/key/
test surface without a runtime gate and retain H6A/H5Y/H5Z
([H6B rejection](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-signed-magnitude-segment-plane-rejected.json) ·
[post-H6A matched residual / H6B target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6a-matched-residual.json) ·
[H6A production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-dense-initial-cached-exact-attention-production.json) ·
[H6A candidate](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-dense-initial-cached-exact-attention-candidate.json) ·
[post-H5Z matched residual / H6A target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h5z-matched-residual.json) ·
[H5Z production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-activation-resident-output-sweep-production.json) ·
[H5Z candidate](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-activation-resident-output-sweep-candidate.json) ·
[post-H5Y residual / H5Z target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h5y-matched-residual.json) ·
[H5Y production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q5-k-activation-tile-k-row-production.json) ·
[H5Y candidate](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q5-k-activation-tile-k-row-candidate.json) ·
[post-H5X matched residual / H5Y target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h5x-matched-residual.json) ·
[H5X production](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-tile-k-col-production.json) ·
[H5X candidate](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-tile-k-col-candidate.json) ·
[post-H5W residual / H5X target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5w-residual.json)).

Post-H6B decomposition does not justify another Q5 producer or geometry pass:
the reconciled **245.850-ms** Q5 stack is **214.346 ms / 87.2%** exact H5Y
consumers, **25.385 / 10.3%** weight producers, **4.567 / 1.9%** activation
packs, and **1.552 / 0.6%** fallback. Its two dominant consumers already emit
**188/156** static VOPD sites with aligned scratch-free loads. Instead admit one
standalone **WPF-H6C exact special-IQ3 expert-major fused-SiLU rowbatch4** leaf
for K3072/N1024/E256. It instantiates the existing RT1-compatible template,
reuses raw gate/up segments across four sorted rows, and preserves every per-row
decode, scalar FMA, wave32 tree, serial wave-0..7 sum, gate/up BF16 boundary,
SiLU expression, and BF16 output. On actual layer-47 weights and natural M512
routing, complete bytes match and fair control-post-gather versus
candidate-pre-gather event/wall moves **32.691/32.724 -> 15.458/15.438 ms
(-52.716%/-52.825%, 2.115x/2.120x)** with scratch0. The bounded owner resolves
only layer-47 IQ3 at exact model shape, adds no allocation, and passes complete
natural M512 at KL0/byte identity across all state/repeat with unchanged
**600,141,856-byte** total scratch. Four cached requests preserve **2,050**
dispatches and exact 46-IQ2/one-H6C plus 45-H5Z/two-H5J topology;
gather-inclusive special time falls **32.127 -> 15.030 ms (-53.215%)**.
Default-off 512/1K/4K improves **+1.148%/+0.796%/+0.560%**; selector-unset
publication confirms **+1.326%/+0.897%/+0.490%**, 3/3 wins each, promoting
**316.106/263.864/174.840 tok/s**. Fixed natural M512 admitted at C4096 moves
**325.211 -> 328.863 tok/s (+1.123%, 5/5 wins)**, narrowing exact llama.cpp HIP
**696.342** to **2.11742x**
([H6C production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-gate-up-expert-major-production.json) ·
[H6C runtime candidate](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-gate-up-expert-major-runtime-candidate.json) ·
[H6C leaf](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-gate-up-expert-major-candidate.json) ·
[H6C target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-gate-up-expert-major-target.json)).

The clean post-H6C source-default refresh measures **329.563 tok/s** from five
exact token-2930/lifecycle-clean samples, **+94.413%** over campaign start and
**2.11293x** behind exact llama.cpp HIP **696.342 tok/s**. Its representative
cached request reconciles **1,546.351 ms / 2,050 dispatches** in a
**1,567.000-ms** span. Current/llama component sums are IQ down
**488.916/154.434 ms**, Q5 **245.503/58.737**, attention **168.520/21.624**,
Q6 **93.490/14.455**, gate/up **475.796/401.393**, and remaining
**74.124/67.598**. Gaps rank **334.482/186.766/146.896/79.035/74.403 ms**.

**WPF-H6D exact row-interleaved IQ3 VOPD** is now the retained gfx1100 IQ3
source default through the unchanged `grouped_raw_iq_active_experts` ABI. Strict
K1024/N3072/E256, registration, and backend misses fail closed; H5Z/H5Q remain
registered rollback and gfx1151 stays absent. Complete natural-M512 state is
KL0/byte-exact across all **48** hidden boundaries, logits, K/V/`KVLiveSpans`,
and repeat with unchanged **161,120,256-byte** workspace and **600,141,856-byte**
total scratch. Four cached requests retain **2,050** dispatches and exact **45
H5Z or 45 H6D + two H5J** topology; IQ3/request/span moves
**475.549/1,552.920/1,583.786 -> 463.354/1,549.015/1,570.143 ms
(-2.564%/-0.251%/-0.861%)** with H6D local128/VGPR104/LDS512/scratch0.
Selector-unset 512/1K/4K improves H5Z rollback **315.267/264.136/175.276 ->
319.072/265.872/176.138 tok/s (+1.207%/+0.657%/+0.492%)**, 3/3 wins each.
Fixed natural-M512/C4096 improves **329.327 -> 332.308 tok/s (+0.905%, 5/5
wins)**, reaches **+96.033%** over campaign start, and is **2.09547x** behind
exact llama.cpp HIP **696.342**. All state/output/scratch/lifecycle checks and
**92/92** retained guards pass. The clean promoted-source C4096/M512 reprofile
and reranked component gaps follow
([H6D production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-row-interleaved-vopd-production.json) ·
[H6D candidate/runtime](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-row-interleaved-vopd-candidate.json) ·
[post-H6C residual / H6D target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6c-matched-residual.json)).

The clean post-H6D source-default refresh measures **332.992 tok/s** from five
exact token-2930/lifecycle-clean samples, **+96.436%** over campaign start and
**2.09117x** behind exact llama.cpp HIP **696.342 tok/s**. Its representative
cached request reconciles **1,530.211 ms / 2,050 dispatches** in a
**1,551.216-ms** median span. Current/llama component sums are IQ down
**475.308/154.434 ms**, Q5 **245.351/58.737**, attention **168.506/21.624**,
Q6 **93.157/14.455**, gate/up **474.056/401.393**, and remaining
**73.833/67.598**. Gaps rank **320.874/186.614/146.882/78.701/72.663 ms** and
explain **99.232%** of the **811.970-ms** kernel gap.

**WPF-H6E exact Q6 activation-tile-K-row transfer** is admitted as a standalone
gfx1100 leaf for all three H5W roles. The rows17/33/M512 cross-product preserves
complete H5W output bytes, complete H5Y activation-plane bytes, sampled
independent Q6 CPU values, strict role preflight, package maps, gfx1151 absence,
and the existing **161,120,256-byte** owner. On actual Q6 weights with five
warmups, 15 counter-rotated samples, and five launches/sample, pack+exact-
producer+consumer-inclusive weighted event/wall moves **65.969/66.187 ->
58.085/58.217 ms (-11.952%/-12.042%, 1.136x/1.137x)**; every role wins both
clocks. Cached ISA emits b64 for rowbatch4 and b64+u16 for rowbatch5 while
candidate resources exactly match H5W at **VGPR136/168, LDS1024/1536,
scratch0**, matching grids, and no compiler under profile. It reuses the live
H5Y plane with zero workspace growth and passes complete natural-M512 at
KL0/byte identity across all **48/48** hidden boundaries,
logits, K/V/`KVLiveSpans`, repeat, and lifecycle. Four cached requests record
exact **142 H5W** versus **142 packs + 142 H6E** while all other topology is
unchanged; Q6/request-sum/span moves **92.867/1,545.837/1,572.498 -> 84.000/
1,541.912/1,563.696 ms (-9.549%/-0.254%/-0.560%)**. Selector-unset H5W
rollback -> H6E source 512/1K/4K improves **318.215/266.225/176.015 ->
319.854/267.357/176.470 tok/s (+0.515%/+0.425%/+0.259%)**, 3/3 exact wins at
each length, with unchanged **161,120,256-byte** workspace and
**600,141,856-byte** scratch. Fixed C4096/direct-M512 improves **332.443 ->
333.329 tok/s (+0.266%, 5/5 wins)**, reaches **+96.635%** over campaign start,
and is **2.08905x** behind exact llama.cpp HIP **696.342**. Promote H6E as Q6
source production; retain H5W/H5I exact rollback and add no kernel, allocation,
workspace, sidecar, or public selector. The clean promoted-source refresh reaches
**334.512 tok/s** from five exact token-2930/lifecycle-clean samples, **+97.333%**
over campaign start and **2.08166x** behind llama.cpp HIP. Its representative
request is **1,519.289 ms / 2,192 dispatches** in a **1,541.013-ms** median
span; IQ-down/Q5/attention/gate-up/Q6 gaps rank **320.074/186.357/146.489/
71.686/70.012 ms** and explain **99.197%** of the **801.048-ms** kernel gap.

Promote **WPF-H6F exact IQ3 paired-output reduction amortization** as the
retained gfx1100 IQ3 source default; H6D remains the immediate registered
rollback. H6F carries two independent P256/P64/local128 rowbatch8 outputs
through one exact wave0..3 reduction epoch while preserving every per-output
IQ3 decode/FMA/reduction/store operation, load, address, grid, and active
traversal. ISA changes output stride **0x100 -> 0x200** with two barriers in
each body, physically proving **24 -> 12 dynamic barriers per rowbatch (-50%)**.
Metadata/runtime is private0/spill0/scratch0 at **VGPR146/152, LDS256/512**,
within the frozen bounds and unchanged local128/grid32768x64. Rows1/7/8/9/M512,
P64/P65, pair-boundary, complete H6D, and CPU bytes pass. Every **45/45** actual
layer is exact and wins both clocks: event/wall moves **445.316/436.801 ->
352.255/360.918 ms (-20.898%/-17.372%, 1.264x/1.210x)**; minimum layer speedup
is **1.253x/1.202x**.

The source owner reuses the existing raw allocation, grouped-IQ library, and
`grouped_raw_iq_active_experts` ABI with zero workspace, sidecar, or dispatch
growth. Complete natural-M512 control/candidate/repeat is KL0 and byte-exact
across logits, all **48/48** hidden boundaries, K/V/`KVLiveSpans`, and teardown
at unchanged **161,120,256-byte** workspace / **600,141,856-byte** scratch.
Four cached requests preserve **2,192 dispatches** and substitute exactly **45
H6D -> 45 H6F**, moving IQ3/request-sum/span **464.484/1,540.306/1,567.420 ->
366.610/1,458.072/1,479.670 ms (-21.072%/-5.339%/-5.598%)**. Selector-unset
512/1K/4K improves **320.079/267.093/176.521 -> 336.830/278.753/181.563 tok/s
(+5.234%/+4.365%/+2.856%)**, with 3/3 exact wins at each length and **156/156**
retained guards passing. Fixed C4096/direct-M512 improves **333.248 -> 352.761
tok/s (+5.856%, 5/5 wins)**, reaches **+108.099%** over campaign start, and
narrows exact llama.cpp HIP **696.342 tok/s** to **1.97397x**. The clean promoted
H6F refresh reaches **353.798 tok/s** from five exact token-2930/lifecycle-clean
samples, **+108.710%** over campaign start and **1.96819x** behind llama.cpp HIP.
Its representative request reconciles **1,435.431 ms / 2,192 dispatches** in a
**1,460.237-ms** median span; IQ-down/Q5/attention/gate-up/Q6 gaps rank
**221.737/191.928/149.544/75.429/71.249 ms** and explain **98.982%** of the
**717.190-ms** kernel gap. H6F is local128/VGPR152/LDS512/scratch0 and the exact
production topology has no H6D/H5Q escape.

**WPF-H6G exact Q5 one-step K-record prefetch is rejected and every candidate
surface is removed.** It screened BF16 K9216/N3072 row-major 12x8 and F32
K3072/N9216 tile-K-col 8x10, the **156.790-ms / 70-call** dominant H5Y subset.
Rows17/33/M512 preserve complete outputs, activation/weight planes, and sampled
CPU values; candidate/control metadata remains private0 at VGPR **194/162**.
The physical premise fails: AMD clang places `s_waitcnt vmcnt(0)` immediately
after each **13/4-instruction** next-record load group with **zero current FMAs
overlapped**. Actual-weight direct weighted event/wall regresses
**194.591/194.547 -> 203.237/204.091 ms (+4.443%/+4.906%)**, and producer/pack-
inclusive regresses **217.265/217.342 -> 225.464/226.243
(+3.774%/+4.095%)**; both roles lose both clocks. Skip runtime ownership, retain
H5Y/H6F production, close compiler-scheduled Q5 K-record prefetch, and rerank
the unchanged post-H6F residual for a materially different exact operation
([H6G rejection](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q5-k-record-prefetch-rejected.json) ·
[post-H6F residual / H6G target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6f-matched-residual.json) ·
[H6F production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-paired-output-reduction-production.json) ·
[H6F candidate](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-paired-output-reduction-candidate.json)).

The post-H6G residency audit rejects persistent exact-Q6 F32 weights before
implementation, then rejects **WPF-H6H bounded source-F16 raw fallbacks** at the
mandatory quality gate. H6H borrows **97,517,568 bytes** inside the existing
**161,120,256-byte** serial plane with no allocation and exposes only three
raw-M512 roles. Natural M512 passes at KL **0.000685**, top-1 **100%**, token
**2930**, deterministic repeat, all **48/48** hidden boundaries changed, and
clean lifecycle. The quality-only 18-prompt/**576-step** lane reaches max KL
**0.411789 > 0.05** despite **565/576 (98.09%)** top-1; all comparisons exercise
changed arithmetic and Poolside passes at KL **0.000157**. Run no promotion
performance timing. Remove every runtime package/context/library/handle/test
surface, retain the unchanged H4 leaf, and preserve all **143** ordered Q6 calls
plus the three exact raw fallbacks in H6F/H6E production at **353.798 tok/s**
([H6H rejection](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q6-f16-raw-fallback-rejected.json) ·
[target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q6-f16-raw-fallback-target.json)).

Promote **WPF-H6I exact IQ3 triple-output reduction amortization** as the
retained gfx1100 IQ3 source default after its standalone and bounded runtime
gates. Complete natural M512 is KL0 and byte-exact across logits, final/post
hidden, all **48/48** boundaries, K/V/`KVLiveSpans`, repeat, and teardown at
unchanged **161,120,256-byte** workspace / **600,141,856-byte** scratch. Four
cached requests preserve **2,192 dispatches**, Q5/attention/Q6/gate-up counts,
and Q6 sequence while substituting exact **45 H6F -> 45 H6I**. IQ3/request-
kernel-sum/span moves **367.025/1,458.371/1,484.313 ->
331.939/1,430.581/1,451.659 ms (-9.559%/-1.906%/-2.200%)**, with H6I
local128/VGPR168/LDS512/scratch0. Selector-unset H6F rollback -> H6I source at
512/1K/4K improves **337.060/279.095/181.676 -> 344.826/283.701/182.982 tok/s
(+2.304%/+1.650%/+0.719%)**, 3/3 exact wins each. Fixed C4096/direct-M512
improves **352.966 -> 360.154 tok/s (+2.036%, 5/5 wins)**, reaches **+112.459%**
over campaign start, and narrows exact llama.cpp HIP **696.342** to
**1.93346x**; **192/192** retained guards pass. Reuse the existing active-
expert ABI and raw owner; add no adapter, allocation, workspace, sidecar,
public selector, or kernel body. Keep H6F/H6D/H5Z/H5Q registered rollback and
reprofile clean promoted production before choosing the next matched residual
target
([H6I production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-triple-output-reduction-production.json) ·
[H6I candidate/runtime](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-triple-output-reduction-candidate.json) ·
[target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-triple-output-reduction-target.json) ·
[post-H6I residual / H6J target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6i-matched-residual.json)).

The clean committed-source H6I refresh is **359.963 tok/s** from
**360.307/360.619/359.963/359.573/359.501**, all exact token 2930 and lifecycle-
clean: **+112.347%** over campaign start and **1.93448x** behind exact llama.cpp
HIP **696.342 tok/s**. Five cached requests preserve **2,192 dispatches**; the
representative request reconciles **1,409.540 ms** in a **1,433.072-ms** median
span. Current/llama components are IQ down **342.209/154.434 ms**, Q5
**253.606/58.737**, attention **172.347/21.624**, gate/up **479.738/401.393**,
Q6 **86.361/14.455**, and remaining **75.280/67.598**. Gaps rank Q5/IQ-down/
attention/gate-up/Q6 at **194.868/187.775/150.723/78.345/71.906 ms** and explain
**98.889%** of the **691.299-ms** kernel gap.

Do not reopen Q5's compiler-prefetch route after H6G's physical failure or
immediately iterate H6I's just-promoted IQ-down operation. **WPF-H6J exact
dense-initial SWA qrow4 unscaled-dot replay is rejected** on the largest
remaining distinct exact family. The local32 leaf preserves complete H6A bytes,
sampled CPU rows, every `KVLiveSpans` field, and lifecycle at starts
0/128/256/384. Code-object ISA physically removes four second-pass K-load and
20 wave-reduction sites, emits four LDS stores plus four loads, and remains
metadata VGPR54/LDS8192/private0/spill0; rocprof confirms scratch0 and unchanged
grid2304x32 but reports runtime **VGPR248**. Every start fails both timing gates:
event regresses **+28.18%/+34.65%/+42.91%/+41.10%** and wall
**+29.76%/+34.52%/+48.44%/+46.09%**. The weighted 144-call stack moves H6A ->
H6J **95.924 -> 133.542 ms event (0.718x)** and **97.607 -> 139.600 ms wall
(0.699x)**. Skip runtime ownership, remove every HIP/Python/exclusion/test
surface, retain H6A/H6I production, and do not retry full 4x512 LDS score replay
without a materially different occupancy-preserving mechanism
([rejection](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-swa-dot-replay-rejected.json) ·
[target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6i-matched-residual.json)).

**WPF-H6K exact IQ3 quadruple-output reduction amortization is rejected.** The
separate H6I sibling passes the frozen **9/9** rows1/7/8/9/M512/P64/P65 and CPU
matrix, and every **45/45** actual-layer output is byte-exact. Cached ISA proves
stride **0x400**, **288** useful FMAs, and fixed-N3072 **4 -> 3 epochs / 8 -> 6
dynamic barriers (-25%)**. Metadata/runtime remains private0/spill0/scratch0 at
VGPR **193/200**, LDS **512/512**, local128, and grid32768x64; the single M512
smoke improves **650.724 -> 629.001 us**. The binding all-layer screen nevertheless
fails every timing rule: **0/45** layers wins both clocks, event moves **329.061
-> 339.509 ms (+3.175%, 0.969x)**, and synchronized wall moves **332.027 ->
337.538 ms (+1.660%, 0.984x)**. Added register pressure crosses the occupancy/
latency knee and erases the barrier saving. Remove every HIP/Python/key/
exclusion/test surface, skip runtime ownership, retain H6I production at
**359.963 tok/s**, and do not retry wider IQ3 output grouping without a
materially occupancy-preserving mechanism
([rejection](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-quadruple-output-reduction-rejected.json) ·
[target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-quadruple-output-reduction-target.json)).

Promote **WPF-H6L exact IQ2 pair16 grouped rowbatch16 decode amortization** as
the retained gfx1100 IQ2 source default after its standalone and bounded
runtime gates. It instantiates the existing WPF-2b local64/pair16 fused-SiLU
template only at K3072/N1024/E256 while keeping one output/expert block, every
row's FMA/two-wave sum/BF16 gate-up boundary/SiLU/store, grid, activation/useful
work, allocation, and workspace. The frozen boundary/CPU matrix passes
**10/10**; all **46/46** actual layers are byte-exact and both-clock positive.
Cached code-object and rocprof evidence remains private0/spill0/scratch0 at
metadata/runtime **VGPR112/112, LDS256/512**, local64, and grid65536x256.

Natural-M512 rowbatch8 control/H6L candidate/repeat is KL0 and byte-exact across
complete logits, all **48/48** hidden boundaries, K/V/`KVLiveSpans`, and
teardown at unchanged **161,120,256-byte** workspace / **600,141,856-byte**
scratch. Four cached requests preserve **2,192 dispatches** and substitute exact
**46 rowbatch8 -> 46 H6L** while leaving H6C/H6I/Q5/Q6/attention topology
unchanged. IQ2/request-sum/span moves **460.772/1,424.447/1,452.975 ->
377.540/1,351.047/1,372.593 ms (-18.064%/-5.153%/-5.532%)**. Selector-unset
rowbatch8 rollback -> H6L source at 512/1K/4K improves
**343.370/282.905/182.706 -> 362.826/295.544/188.636 tok/s
(+5.666%/+4.468%/+3.246%)**, with 3/3 exact wins each. Fixed natural
C4096/direct-M512 improves **360.451 -> 381.893 tok/s (+5.949%, 5/5 wins)**,
reaches **+125.284%** over campaign start, and narrows exact llama.cpp HIP
**696.342** to **1.82340x**; **212/212** guards pass. Reuse the existing pair16
ABI and raw owner; add no adapter, allocation, workspace, sidecar, public
selector, or kernel body. Keep rowbatch8 as same-ABI rollback
([production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq2-pair16-rowbatch16-production.json) ·
[candidate/runtime](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq2-pair16-rowbatch16-candidate.json) ·
[target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq2-pair16-rowbatch16-target.json)).

The clean H6L refresh reaches **381.977 tok/s**, **+125.334%** over campaign
start, **+6.116%** over clean H6I, and is **1.82299x** behind exact llama.cpp
HIP **696.342**. The representative exact trace is **1,326.062 ms / 2,192
dispatches**. Q5/IQ-down/attention/Q6 gaps are
**194.004/189.827/151.442/72.392 ms**; gate/up now measures **393.895 vs
401.393 ms**, already **7.498 ms faster** than llama.cpp.

**WPF-H6M exact explicit wait-split Q5 K-record pipelining is rejected.** The
rows17/33/M512 matrix and both actual roles preserve complete H5Y/CPU/plane
bytes. Cached ISA physically realizes exact **13/4 next-record loads -> 32
current-record `v_fmac_f32` sites -> one wait** with no intermediate wait or
loaded-value use; metadata/runtime remains private0/scratch0 at VGPR **194/200
and 162/168** within the frozen ceilings. The new premise therefore succeeds
physically, unlike H6G.

It fails the binding timing gate. Actual-weight 70-call direct event/wall moves
**194.618/195.249 -> 205.367/205.331 ms (+5.523%/+5.164%)** and
producer/pack-inclusive moves **215.590/216.860 -> 227.873/227.347
(+5.697%/+4.836%)**; both roles lose both clocks. Remove every source/key/
exclusion/test surface, skip runtime ownership, retain H5Y/H6L production
**381.977 tok/s**, and close exact Q5 geometry, plane, ownership,
compiler-managed prefetch, and explicit wait-split premises. Return to the
unchanged post-H6L IQ-down/attention residual with a distinct operation
([H6M rejection](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q5-k-record-wait-split-rejected.json) ·
[post-H6L residual / target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6l-matched-residual.json)).

Admit standalone **WPF-H6N exact global dense-initial fixed-512 score arena**
on the distinct attention family. The separate gfx1100 sibling retains H6A's
local256/grid48x128, token-strided QK owners, wave trees, materialized-score/
max/exp/denominator order, normalized-weight-before-PV association, F32 store,
complete `KVLiveSpans` ABI, allocation, workspace, dispatch, and source policy.
The frozen **6/6** matrix is complete-byte exact against H6A, matches sampled
CPU rows, preserves all span bytes, rejects invalid shapes before HIP loading,
and recovers allocations at starts **0/128/256/384**.

H6N reduces exact dynamic launch storage **`(4096 + 8 + 128) * 4 = 16,928` ->
`(512 + 8 + 128) * 4 = 2,592` bytes (-84.688%)** and is now the retained
gfx1100 global dense-initial source default. The generic role parser keeps H6A
and H6N as bounded candidates; no backend/quant branch, ABI, allocation,
workspace, sidecar, or public selector is added. Complete natural M512 is KL0/
byte-exact across logits, all **48/48** hidden boundaries, K/V/spans, repeat,
and teardown. Four cached requests keep **2,192 dispatches** and substitute
exact **48 H6A global -> 48 H6N**, moving global/attention/kernel-sum/span
**57.126/169.556/1,320.178/1,346.667 -> 31.969/148.140/1,305.325/1,327.300 ms
(-44.038%/-12.631%/-1.125%/-1.438%)**. Fresh selector-unset fixed C4096/M512
improves **381.772 -> 387.571 tok/s (+1.519%, 5/5 wins)**, **+128.633%** over
campaign start and **1.79668x** behind llama.cpp HIP **696.342**. Selector-
unset 512/1K/4K moves **363.520/295.622/188.755 -> 363.324/296.211/188.858
tok/s (-0.054%/+0.199%/+0.054%)** with exact outputs and clean teardown; 4K
wins **3/3**. H6A global remains registered rollback, H6A SWA is unchanged,
gfx1151 stays excluded, and **81/81** guards pass
([H6N production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-global-dense-initial-score-arena512-production.json) ·
[candidate/runtime](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-global-dense-initial-score-arena512-candidate.json) ·
[target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-global-dense-initial-score-arena512-target.json)).

The clean promoted-source reprofile reaches **386.959 tok/s** from five exact
token-2930/lifecycle-clean samples, **+128.272%** over campaign start and
**1.79952x** behind llama.cpp HIP. Its representative request is **1,309.339
ms / 2,192 dispatches** in a **1,333.225-ms** median span. Current/llama Q5,
IQ-down, attention, Q6, gate/up, and remaining are **254.689/58.737**,
**346.108/154.434**, **148.104/21.624**, **86.987/14.455**,
**397.542/401.393**, and **75.908/67.598 ms**. Gaps rank **195.952/191.674/
126.480/72.532 ms**; kernel sum is **1,309.339 vs 718.241 ms**.

Do not reopen Q5: H6M physically realized the final explicit wait-split premise
and still lost both clocks. H6N provides the distinct attention interval.
**WPF-H6P exact staged-wave-publication triple-output IQ3 is now a qualified
bounded default-off owner.** It preserves H6I P256/P64/rowbatch8 bytes and
arithmetic while reducing runtime VGPR **168 -> 112**. Complete natural M512 is
KL0/byte-exact across logits, all **48/48** hidden boundaries, K/V/
`KVLiveSpans`, repeat, and teardown. Four cached requests retain **2,192
dispatches**, exact all-family topology, and substitute **45 H6I -> 45 H6P**;
IQ3/request-sum/span moves **335.561/1,350.501/1,377.064 -> 326.309/1,346.568/
1,368.182 ms (-2.757%/-0.291%/-0.645%)**.

Default-off 512/1K/4K improves **363.446/296.015/188.932 -> 366.223/297.245/
189.389 tok/s (+0.764%/+0.416%/+0.242%)**, 3/3 exact wins each. Fixed natural
C4096/M512 improves **387.746 -> 388.293 tok/s (+0.141%, 4/5 wins)** and remains
**1.79334x** behind llama.cpp HIP. Workspace/scratch stay **161,120,256/
600,141,856 bytes**; generic grouped-IQ role resolution is now exact at
K1024/N3072/E256, all misses fail closed, and **246/246** guards pass. Keep H6I
as source production and freeze an independent source-default contract before
changing the one selected-map value
([H6P candidate/runtime](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-staged-wave-publication-candidate.json) ·
[post-H6N residual / target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6n-matched-residual.json)).

H6P is subsequently retained as source; clean fixed natural C4096/M512 reaches
**389.145 tok/s / 1,302.492 ms**, **1.77515x** behind fresh matched llama.cpp HIP
**690.791 tok/s / 714.008 ms**. **WPF-H6Q exact compact-shuffle-loop staged-wave
IQ3 now qualifies as a standalone gfx1100 leaf.** It uses a separate no-unroll
helper to preserve H6P's 216 FMAs, 120 dynamic shuffles/order, three staged
accumulator scopes, stride `0x300`, two/eight barriers, local128/grid32768x64/
LDS512, and complete bytes. Static bpermutes fall **120 -> 24**, code
**8,360 -> 6,620 bytes**, and metadata/runtime VGPR **107/112 -> 95/96**, with
unchanged 24 LDS loads/12 stores and private0/spill0/scratch0. Frozen **9/9**
and all **45/45** actual layers are exact and both-clock positive: event
**329.124 -> 313.405 ms (-4.776%, 1.050x)** and wall **326.037 -> 317.946 ms
(-2.481%, 1.025x)**; minimum layer wins are **1.038x/1.016x**. H6Q is
subsequently retained as the source default with H6P as explicit
same-ABI rollback. Complete natural M512 is KL0/byte-exact across all 48
boundaries, K/V/spans, repeat, and teardown. Four cached requests preserve
**2,192 dispatches** and substitute exact **45 H6P -> 45 H6Q**, cutting IQ3/
request-sum/span **4.725%/0.487%/1.076%**. Fresh selector-unset 512/1K/4K gains
**+0.730%/+0.571%/+0.359%**, 3/3 wins each; fixed C4096/M512 gains **+0.467%
(5/5 wins)** at **390.887 tok/s**, **1.76724x** behind fresh matched llama.cpp
HIP **690.791**. Workspace/scratch remain unchanged, gfx1151 remains excluded,
and **156/156** guards pass.

The previous clean H6Q baseline remains **390.947 tok/s / 1,301.236 ms /
2,192 dispatches**. **WPF-H6R exact DPP peer-exchange staged-wave IQ3 is now the
retained source default, with H6Q as explicit same-ABI rollback.** Its admitted
leaf remains exact and both-clock positive on all **45/45** actual layers, with
zero bpermutes, exact **24 permlanex16 + 96 DPP**, unchanged math/topology, and
metadata/runtime VGPR **101/104** at private0/spill0/scratch0. Complete natural
M512 is KL0/byte-exact across logits, all **48/48** hidden boundaries, K/V/
`KVLiveSpans`, repeat, and teardown. Four production-identical cached requests
preserve **2,192 dispatches** and replace only **45 H6Q -> 45 H6R**, moving
IQ3/request-sum/span **310.159/1,332.893/1,362.094 ->
267.241/1,285.199/1,307.416 ms (-13.837%/-3.578%/-4.014%)**. Fresh
selector-unset 512/1K/4K gains **+3.793%/+3.274%/+1.992%**, all 3/3 exact wins;
fixed C4096/M512 improves **391.307 -> 407.780 tok/s (+4.210%, 5/5)** and is
**1.69403x** behind matched llama.cpp HIP **690.791**. Allocation, ABI,
workspace/scratch, sidecar, and dispatch count remain unchanged; gfx1151 fails
closed and **219/219** guards pass. Clean committed H6R reprofiling reaches
**407.091 tok/s / 1,247.252 ms / 2,192 dispatches**, versus campaign-start
**169.516 tok/s / 3,001.692 ms** and matched llama.cpp HIP **690.791 tok/s /
714.008 ms**. Q5/attention/IQ-down/Q6 gaps are
**197.358/127.879/125.185/72.769 ms**; Q5 remains mechanism-closed. H6A SWA now
owns the largest actionable exact leaf at **117.506 ms / 144 calls**. Select
one-shot target-only **WPF-H6S exact DPP peer-exchange dense-initial SWA qrow4**:
transfer only H6R's permlanex16+DPP 8/4/2/1 peer primitive into H6A's exact
reduction while preserving one-wave/one-head/qrow4 ownership, BF16 K/V traffic,
two-pass QK, online arithmetic, softmax/PV order, complete output bytes, and all
`KVLiveSpans` fields. Freeze RED first; require exact **12 remaining bpermutes +
8 permlanex16 + 32 DPP**, unchanged 12 global u16 loads/4 exp/no barriers, code
<=8,000 bytes, VGPR <=80, private/spill/scratch0, and every starts0/128/256/384
plus weighted 144-call schedule to win both clocks. Remove all H6S surfaces on
any miss without follow-up tuning
([post-H6R residual / H6S target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6r-matched-residual.json) ·
[H6R production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-dpp-peer-exchange-production.json) ·
[H6R candidate/runtime](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-dpp-peer-exchange-candidate.json) ·
[post-H6Q target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6q-matched-residual.json) ·
[H6Q production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-compact-shuffle-loop-production.json)).

H6S subsequently fails the binding one-shot gate and every candidate
implementation/test/key/exclusion surface is removed. Complete outputs remain
byte-exact and finite at starts0/128/256/384 with immutable spans and recovered
lifecycle. ISA realizes **12 bpermutes + 8 permlanex16 + 32 DPP**, unchanged
loads/exp/FMA/stores/no barriers, code **7,044 -> 6,676 bytes**, metadata VGPR
**64 -> 59**, and runtime VGPR64/LDS0/scratch0. Every start nevertheless loses
both clocks; weighted 144-call H6A -> H6S event regresses
**94.696 -> 108.850 ms (+14.946%, 0.870x)** and wall
**96.707 -> 112.761 ms (+16.601%, 0.858x)**. Runtime qualification is forbidden;
H6A SWA, H6N global, and clean H6R **407.091 tok/s** remain unchanged. Close DPP
attention peer exchange unless a materially new premise appears
([H6S rejection](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-swa-dpp-peer-rejected.json)).

Promote **WPF-H6T exact fused-DPP-add staged-wave IQ3** as the retained gfx1100
IQ3 source default, with H6R as explicit same-ABI rollback. The leaf remains
exact through **9/9** and **45/45** both-clock wins, realizing **24 permlanex16 +
96 DPP adds + zero moves**, **1,384 slots / 7,920 bytes**, and unchanged runtime
VGPR104/LDS512/scratch0. Complete natural M512 is KL0 and byte-exact across all
**48/48** hidden boundaries, complete K/V/`KVLiveSpans`, repeat, and teardown.
Four cached requests preserve **2,192** dispatches and substitute exact **45 H6R
-> 45 H6T**; IQ3/request-sum/span move **267.433/1,284.605/1,313.165 ->
261.844/1,283.120/1,304.737 ms (-2.090%/-0.116%/-0.642%)**. Fresh selector-
unset fixed C4096/M512 improves **407.600 -> 408.900 tok/s (+0.319%, 5/5)** and
is **1.68939x** behind matched llama.cpp HIP **690.791**. Fresh 512/1K/4K
improves **381.821/307.478/193.289 -> 383.162/308.780/193.629 tok/s
(+0.351%/+0.423%/+0.176%)**, every **3/3** pair exact/finite/lifecycle-clean.
Change only the selected-map value: the nine-entry active-expert ABI, raw
allocation, grouped-IQ library, workspace, total scratch, dispatch count, and
gfx1151 fail-closed behavior remain unchanged; **144/144** source guards pass
([H6T production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-fused-dpp-add-production.json) ·
[H6T candidate/runtime](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-fused-dpp-add-candidate.json) ·
[H6T target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-fused-dpp-add-target.json)).

Promote **WPF-H6U exact DPP-add wave reduction for Q6 activation-row
consumers** as the retained gfx1100 Q6 source default; H6E remains explicit
rollback. The **11/11** exact leaf replaces **320/400/400** generic bpermutes
with **64+256 / 80+320 / 80+320 permlanex16+DPP-add** and cuts runtime VGPR
**136/168/168 -> 112/144/144** at unchanged LDS/private0/spill0/scratch0.
Complete natural M512 is KL0/byte-exact across **48/48** hidden boundaries,
complete logits, K/V/`KVLiveSpans`, repeat, and teardown. Four cached requests
preserve **2,192** dispatches and substitute exact **2/46/94 H6E -> H6U**
consumers; consumer/Q6/request-sum/span move
**54.144/86.958/1,276.589/1,305.317 -> 48.443/81.029/1,274.060/1,295.123 ms
(-10.529%/-6.817%/-0.198%/-0.781%)**. Fresh selector-unset fixed C4096/M512
improves **409.485 -> 411.704 tok/s (+0.542%, 5/5)** and is **1.67788x** behind
matched llama.cpp HIP **690.791**; fresh 512/1K/4K improves
**382.632/308.496/193.767 -> 384.637/309.813/194.321 tok/s
(+0.524%/+0.427%/+0.286%)**, all **3/3** exact wins. Change only three selected-
map values: F32 N72 fallback, allocation, workspace/scratch, dispatches, and
gfx1151 fail-closed behavior remain unchanged; **153/153** source/kernel/
backend/runner guards pass
([H6U production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-q6-dpp-wave-reduction-production.json) ·
[H6U candidate/runtime](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-q6-dpp-wave-reduction-candidate.json) ·
[post-H6T residual / H6U target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6t-matched-residual.json)).

Clean committed H6U production reaches **410.220 tok/s / 1,232.836 ms / 2,192
dispatches**, **+141.994%** over campaign start and **1.68395x** behind matched
llama.cpp HIP **690.791 tok/s / 714.008 ms**. **WPF-H6V exact DPP-add Q5 wave
reduction is rejected and fully removed.** All six roles are byte-exact and ISA
realizes exact **32/96/80/96/80/80 permlanex16 +
128/384/320/384/320/320 DPP adds**, zero bpermutes/moves, fewer slots/VGPR,
unchanged FMA/load/LDS/barrier/store counts, and scratch0. Weighted 188-call
event/wall improves **269.681/271.908 -> 267.729/267.342 ms
(-0.724%/-1.679%)**, but only **3/6** roles pass both clocks. BF16 K3072/N1024
regresses **+12.795%/+13.346%**, BF16 K6144/N3072 misses event by **0.560%**,
and F32 K3072/N6144 regresses **+4.137%/+1.757%**. The predeclared universal
all-role gate fails. Skip runtime/source work, remove all implementation/test/
key/export/gfx1151-exclusion surfaces without tuning, retain H5Y/H6U, and do
not subset or reopen H6V
([H6V rejection](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-q5-dpp-wave-reduction-rejected.json) ·
[target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6u-matched-residual.json)).

Retain **WPF-H6W exact late-start dense-initial SWA qrow4 aligned
global-score-record replay** as the gfx1100 SWA source default with explicit H6A
rollback. Source promotion changes only one selected SWA map value and adds the
rollback map; H6N global, starts0/128 H6A fallback, runner/KV ABI, borrowed
**18,874,368-byte** Q5-plane prefix, workspace, and kernel bodies remain
unchanged. Complete natural M512 is KL0/exact across all **48/48** boundaries,
logits, K/V/spans, repeat, and teardown. Four production-identical cached
requests preserve **2,192** dispatches and exact
**48 H6N + 72 H6A + 72 H6W**, cutting selected late SWA/attention/kernel-sum/
span **23.808%/12.344%/1.319%/2.018%**. Fresh selector-unset fixed natural
C4096/M512 improves **411.192→417.421 tok/s (+1.515%, 5/5)** and fresh
512/1K/4K improves **385.356/309.745/194.411→390.382/312.026/194.709 tok/s
(+1.304%/+0.736%/+0.153%)**, all **3/3**, at unchanged
**161,120,256/600,141,856-byte** workspace/scratch. Keep H6A rollback, gfx1151
fail-closed behavior, and the **115/115** source guard boundary
([production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-swa-global-score-replay-production.json) ·
[candidate/runtime](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-swa-global-score-replay-candidate.json)).

Clean committed H6W reaches **416.891 tok/s / 1,214.475 ms / 2,192
dispatches**, **+145.930%** over campaign start and **1.65700x** behind matched
llama.cpp HIP **690.791 tok/s / 714.008 ms**. Current Q5/IQ-down/attention/Q6
gaps are **198.017/119.429/105.690/66.788 ms**. The representative Q5 request
reconciles exactly to **223.393 ms H5Y consumers, 26.241 ms producers, 4.781
ms packs, and 1.916 ms fallback**. Static-shape-only source-MMQ screening
rejects all six material shapes at max KL **0.585291–4.622387**; the three dominant exact-value F32/
SGEMM shapes also fail at **0.402533–0.846753** over the full 18-prompt/
576-step category-heldout gate. Do not reopen changed-association Q5 or H6V.

Select target-only **WPF-H6X exact workgroup-resident IQ3_XXS grid table** on
H6T's **264.602 ms / 45 calls**. Its current staged local128 body still performs
two divergent constant/global `IQ3_XXS_GRID[256]` loads per segment. Natural
routing records **33,547 rowbatch8 epochs / 103,056,384 segment decodes** and
models **824,451,072** table-load wave instructions / **105.530 GB** logical
bytes. A separate H6T sibling must cooperatively publish the exact **1,024-byte**
uint32 table to LDS once/workgroup, barrier, and change only those two lookup
sources. Preserve 216 FMAs, 24 permlanex16, 96 direct DPP adds, staged scopes,
wave sum/store, P256/P64 traversal, rowbatch8, grid, ABI, allocation, and
workspace. Require exact **19 global loads, two table preloads, six LDS reads,
three barriers, 1,408-byte metadata LDS / <=1,536 runtime LDS, VGPR <=101/104,
private/spill/scratch0**, rows1/7/8/9/M512 plus P64/P65/tails/CPU bytes, and
**45/45** actual-layer event+wall wins. Remove all H6X surfaces on any miss;
runtime and source promotion are separate
([post-H6W residual / H6X target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6w-matched-residual.json)).

H6X is **rejected at the binding physical gate**. Its cached exact matrix passes
**10/10**, and ISA realizes the intended global-load **23→19**, six LDS-read,
one coalesced-preload-store, and barrier **2→3** changes at metadata LDS
**1,408 bytes**, unchanged 216 FMAs/24 permlanex16/96 DPP, and zero private/
spill/scratch. Metadata VGPR nevertheless rises **101→103**, exceeding the
frozen **≤101** maximum. Do not profile, time, tune, or rerun after this miss;
remove every H6X implementation/test/key/export/gfx1151-exclusion surface,
retain H6T/H6W production **416.891 tok/s**, and rerank a materially distinct
operation
([H6X rejection](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-grid-lds-rejected.json)).

Select target-only **WPF-H6Y exact IQ3 packed-prefix b32 load** as a materially
distinct operation; do not tune or reopen H6X. H6T emits exact **8 b128 + 9
b32 + 6 d16 = 23** global loads. The six d16 loads read adjacent FP16-scale and
selector-pair bytes across three scopes. H6Y must load bytes0..3 once/scope as
little-endian b32, recover identical bits, and preserve aux/table/sign/
magnitude/FMA/reduction/store order. Require exact **8 b128 + 12 b32 + zero
d16 = 20** loads, unchanged DS/barriers/LDS384/runtime512/216 FMAs/24
permlanex16/96 DPP/stride, VGPR **≤101/104**, and private/spill/scratch0.
Natural routing models **412,225,536 fewer global-load wave instructions** at
unchanged **52.765 GB** prefix bytes; this is not speed evidence. Freeze RED,
then require complete FP16-bit/row/P64/P65/CPU/lifecycle bytes, cached named
trace with no compiler, and all **45/45** layers plus aggregate to win both
clocks under 5/15/5. Remove H6Y on any miss without tuning/rerun; runtime/source
work remains separate
([post-H6X residual / H6Y target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6x-rejection-matched-residual.json)).

H6Y is **rejected at the binding physical gate**. The correctness-preserving
rolling-window/wave-lane0-scale implementation passes cached **11/11** and
realizes global loads **23→20**, unchanged barriers/LDS/arithmetic/DPP, and
smaller code/slots **7,920/1,384→7,872/1,357**. It nevertheless adds three
`ds_bpermute_b32` scale broadcasts to the frozen unchanged-DS set and raises
metadata VGPR **101→106**, exceeding the frozen **≤101** maximum. Do not
profile, time, tune, or rerun after these misses; remove every H6Y
implementation/test/key/export/gfx1151-exclusion surface and retain H6T/H6W
production **416.891 tok/s**
([H6Y rejection](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-packed-prefix-b32-rejected.json)).

Promote **WPF-H6Z exact late-start global qrow4 aligned score/weight replay** to
the retained gfx1100 global source default; keep H6W as the explicit H6N-global
rollback and H6A as complete rollback. Change only the selected global role;
SWA remains H6A early/H6W late. H6Z remains local32/grid1536x32 at code/slots
**4,024 B / 690**, metadata/runtime VGPR **47/48**, LDS0/private0/spill0/
runtime-scratch0, and borrows H6W's existing **18,874,368-byte** Q5-plane prefix
with a strict **12,582,912-byte** extent. No allocation, workspace, sidecar, or
dispatch changes.

Fresh source-selected natural M512 is KL0/top-1 100% and byte-exact across
complete logits, final/post hidden, all **48/48** hidden boundaries, K/V/
`KVLiveSpans`, repeat, and teardown. Four source-selected cache-only requests
preserve **2,192** dispatches and exact production topology **24 H6N + 24 H6Z +
72 H6A + 72 H6W**. Late-global/attention/kernel-sum/span moves **23.894/125.254/
1,214.563/1,241.814→12.231/116.041/1,205.023/1,227.056 ms** with zero compiler
process.

Fresh selector-unset fixed natural C4096/M512 improves **417.180→420.785 tok/s
(+0.864%, 5/5)**. H6Z is C4096-only, so C512/C1024 are unchanged-path controls
at **390.831/311.543 tok/s**; binding 4K improves **194.478→194.694 (+0.111%,
2/3)**. The exact fixed/event/span wins satisfy the cycle-wall retention policy
despite aggregate 4K noise. Keep gfx1151 excluded, pass **126/126** guards,
commit source production, then cleanly reprofile and rerank toward llama.cpp
**690.791 tok/s**
([H6Z production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-global-score-weight-replay-production.json) ·
[candidate/runtime](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-global-score-weight-replay-candidate.json)).

The clean committed H6Z checkpoint reaches **423.233 tok/s** from
**422.351/424.811/424.219/423.140/423.233**, all exact token 2930, finite, and
lifecycle-clean. This is **+149.671%** over campaign start and **1.63218x**
behind matched llama.cpp HIP **690.791 tok/s**. Cache-only profiling preserves
**2,192 dispatches**, exact **24 H6N + 24 H6Z + 72 H6A + 72 H6W** topology,
**1,195.702-ms** kernel sum, **1,217.373-ms** span, and zero compiler process.
The matched Q5/IQ-down/attention/Q6 gaps are **198.740/116.810/93.654/66.495
ms**, explaining **98.756%** of the **481.694-ms** kernel residual; gate/up is
already **1.929 ms faster** than llama.cpp.

Select target-only **WPF-H7A exact late-start SWA scaled-score replay**. H6W's
**62.562 ms / 72 calls** stores unscaled dots, uses `dot * scale` for max, then
repeats the identical multiplication after loading each aligned record. H7A
must compute the scaled score once in pass one, use that same F32 bit pattern
for max and the `float4` record, and replay `exp(score - max)` in pass two. The
natural schedule removes **255,135,744** duplicate scale multiplications with
no byte, workgroup, plane, allocation, workspace, or result change. Q5
changed-association/H6V, H6X/H6Y, wider-qrow, cross-head, and changed-association
attention controls stay closed.

Commit a RED-only leaf contract before executable changes. Require complete
H6W/CPU/scaled-record/five-span/poison/lifecycle equality at starts256/384;
remove exactly four second-pass scale-subtract FMA sites (**total
`v_fma_f32` <=52 versus 56**) while preserving eight u16 loads, one b128 record
load/store, 32 bpermutes, four exp sites, code **<=4,984 B / <=871 slots**,
metadata/runtime VGPR **<=54/56**, and LDS0/private0/spill0/scratch0; prove a
named cache-only launch with zero compiler; and consume one immutable 5/15/5
screen where both starts and the weighted **72-call** aggregate win event and
wall. Any miss removes all H7A surfaces without tuning/rerun. Leaf, bounded
runtime, and source promotion remain separate
([post-H6Z residual / H7A target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6z-matched-residual.json)).

Reject H7A at the first binding complete-byte gate. The separately named
implementation passes structure, registry/backend exclusion, strict preflight,
and one cached build, and both outputs are finite. It nevertheless differs from
H6W at **80,469/1,179,648** elements for start256 and **100,075/1,179,648** for
start384; maxima are **4.656613e-9 / 3.7252903e-9**. The target analysis missed
that H6W's replay `dot * scale - max` is compiled as a fused `v_fma_f32`.
Storing the first-pass scaled score introduces an intermediate F32 rounding
before subtracting max and therefore changes exp/PV output bits. The exact
premise is false even though the numerical delta is tiny. Per the frozen
contract, do not inspect candidate code-object resources, profile, time, tune,
rerun, or apply a quality waiver. Remove every H7A implementation/test/key/
export/gfx1151-exclusion surface, retain byte-identical H6W/H6Z production
**423.233 tok/s / 1,195.702 ms**, and rerank a materially distinct operation
([H7A rejection](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-swa-scaled-score-replay-rejected.json)).

The clean post-rejection reprofile is **422.602 tok/s** and representative
compiler-free trace is **1,200.759 ms / 2,192 dispatches**. Q5/IQ-down/attention/
Q6 remain the top gaps at **198.174/118.581/93.991/67.187 ms**. Select target-
only **WPF-H7B exact lane-parallel IQ3 final-row publication** on H6T's
**263.748 ms / 45 calls**. Do not reopen H6X/H6Y: H7B changes only ownership
after the first barrier. Lanes0..7 each publish one row while preserving that
row's exact serial wave0→1→2→3 sums and three BF16 stores. This models H6T's
static/dynamic DS-load and global-store issue sites **24/824,451,072→3/
103,056,384 each (-87.5%)** across **34,352,128** phases without changing
logical bytes or arithmetic.

Freeze RED first. Bind complete H6T/CPU bytes, all actual layers, poison,
finite/lifecycle behavior, exact **3 b128 LDS loads + 3 d16 stores**, unchanged
23 global loads/12 LDS stores/two barriers/216 FMAs/24 permlanex16/96 DPP, code
**<=7,920 B / <=1,384 slots**, VGPR **<=101/104**, LDS **384/<=512 B**, and
private/spill/scratch0. Require named cached execution with no compiler and all
**45/45** layers plus aggregate to win event and wall under the immutable
5/15/5 screen. Any miss removes every H7B surface without tuning/rerun; leaf,
runtime, and source decisions remain separate
([post-H7A residual / H7B target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h7a-rejection-matched-residual.json)).

Reject H7B at its first compiled physical-resource gate. Complete H6T/CPU/
poison/lifecycle checks pass **10/10** and codegen realizes the requested **3
b128 LDS loads + 3 d16 stores** with unchanged 23 global loads/12 LDS stores/
two barriers/216 FMAs/24 permlanex16/96 DPP; code/slots fall **7,920/1,384→
5,916/994**. Metadata VGPR nevertheless rises **101→108**, exceeding the frozen
**≤101** ceiling. Apply the one-shot rule: skip rocprof, runtime-resource
adjudication, and all-layer timing; do not tune/recompile/rerun; remove every
H7B implementation/test/key/export/gfx1151-exclusion surface; retain H6T/H6Z
production **422.602 tok/s / 1,200.759 ms**; and rerank a materially different
exact operation
([H7B rejection](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-lane-parallel-final-rows-rejected.json)).

The clean post-H7B checkpoint is **422.947 tok/s** and compiler-free
**1,199.578 ms / 2,192 dispatches**. Q5/IQ-down/attention/Q6 gaps are
**197.783/118.305/93.890/66.748 ms**. The three exact raw-Q6 fallbacks own median
**28.474 ms (34.904% of Q6)**. Standalone **WPF-H7C exact raw-Q6 DPP-add wave
reduction is admitted** for K12288/N3072 BF16, K3072/N9216 F32, and
K9216/N3072 BF16; H6H's source-F16 route remains quality-closed at max KL
0.411789.

The frozen matrix passes **22/22** across all roles and rows1/7/8/9/M512. The
first BF16/F32 object has exact zero bpermutes, **32 permlanex16 + 128 DPP
adds**, unchanged 24 global loads/one store/eight b128 LDS stores/two LDS loads/
one barrier/32 ordered FMAs, and cuts code/slots **4,840/843→4,228/681** and
**5,040/909→4,452/749**. Metadata/runtime VGPR is **60/64** and **55/56**,
LDS512, private/spill/scratch0. Cached named execution covers exact grids
**98,304x64 / 589,824x32** with zero compiler.

The immutable actual-weight 5/15/5 screen improves every role on both clocks.
Layer-0 down is **14.866/14.868→14.741/14.750 ms**, layer-47 Q is
**10.752/10.795→10.705/10.700 ms**, and layer-47 output is
**11.630/11.639→11.537/11.547 ms** event/wall. Aggregate improves
**37.248/37.303→36.983/36.998 ms (-0.712%/-0.817%)** with exact bytes and
lifecycle.

H7C's bounded runtime owner and source publication are now separately
qualified. Generic `gguf_linear` consumes only the three exact M512 `(quant,
output ABI, rows, K, N)` roles; the named generic rollback remains empty, and
every wrong-shape/registration/backend case fails closed. Default-off complete
M512 state is KL0/byte-exact across all **48/48** hidden boundaries and full
KV/spans at unchanged scratch. Four cached requests preserve **2,192
dispatches** and replace exactly **2 BF16 + 1 F32** generic calls with H7C,
moving selected raw-Q6/Q6/span
**28.543/81.457/1,280.898→28.220/81.105/1,279.005 ms** with zero compiler.
Fixed C4096/M512 improves **420.701→420.914 tok/s (+0.0505%, 4/5)**; 512/1K/4K
medians improve **+0.0552%/+0.0274%/+0.0179%**, all exact and lifecycle-clean.

Source promotion changes only the live package map. Fresh selector-unset M512
state remains KL0 and byte-exact across complete state, all **48/48**
boundaries, KV/spans, and repeat. A fresh four-request trace again substitutes
exactly **2 BF16 + 1 F32** calls at unchanged offsets/resources and improves
selected raw-Q6/Q6/span
**28.583/81.639/1,283.417→28.376/81.470/1,280.788 ms** with zero compiler. Fresh
aggregate timing is mixed: fixed C4096/M512 is
**419.433→418.487 tok/s (-0.225%, 2/5)**, while 512/1K/4K is
**+0.0925%/+0.0372%/-0.0488%**. Retain source under the cycle-wall policy based
on two independent selected-subwindow/span wins and the immutable all-role leaf
screen, while recording the fixed/4K rows as noise rather than wins. The last
clean committed checkpoint remains **422.947 tok/s / 1,199.578 ms** until the
required post-commit reprofile
([H7C production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-raw-q6-dpp-wave-reduction-production.json) ·
[candidate/runtime](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-raw-q6-dpp-wave-reduction-candidate.json) ·
[target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h7b-rejection-matched-residual.json)).

The required post-H7C reprofile records **422.786 tok/s** with representative
cache-only kernel sum/span **1,197.499/1,219.043 ms** across **2,192
dispatches**, remaining **1.63390x** behind matched llama.cpp HIP. Current
Q5/IQ-down/attention/Q6 gaps are **196.915/117.620/93.693/66.653 ms** and own
**98.219%** of the complete kernel gap. Close target-only **WPF-H7D exact Q5
row-interleaved VOPD scheduling**: the naive control/candidate both emit **52
`v_dual_fmac_f32`** sites at metadata VGPR122, while explicit pairing fails
compilation 16 times at gfx1100's `src0 operands must use different VGPR banks`
constraint. No production H7D surface exists.

Standalone **WPF-H7E IQ3-only two-plane residual-D4 source-MMQ is admitted**.
The RED-first matrix turns GREEN **9/9** across rows1/7/8/9/M512 plus
empty/uneven/127/128/129 expert tails, complete overwrite, independent CPU
quality, immutable metadata, finiteness, lifecycle, strict IQ3 registry/backend
scope, exact H6T/IQ4 fallback, and gfx1151 exclusion. The first candidate
object independently reproduces local `(32,8)`, grid
`(24,mmq_total_rows/128)`, dynamic LDS57,856, code **31,564 B**, metadata
VGPR/SGPR **148/44**, runtime VGPR **152**, private/spill/dynamic-stack/scratch0,
and exact **128 integer WMMAs / five barriers / 64 BF16 stores**. Cached rocprof
names H7E and records zero compiler activity.

The immutable producer-inclusive 5/15/5 all-layer screen wins event and
synchronized wall for every **45/45** actual IQ3 layer. Aggregate event moves
**247.297→186.732 ms (-24.491%, 1.324x)** and wall
**260.672→180.752 ms (-30.659%, 1.442x)**; max leaf KL is **0.000487** and
minimum top-1 **99.941%**, with finite output and recovered lifecycle. A
separate bounded default-off owner then reused `expert_gate_up` with zero growth.
Natural-M512 state passed at KL **0.000224** / top-1 **100%**, and cached tracing
proved **45 H6T → 45 tile128 + 45 producer + 45 H7E** while diagnostic IQ-down
fell **269.921→208.298 ms**.

Reject and remove that owner on the binding complete gate. Every one of the 18
committed prompts is independently extended to M512; all **576/576** steps
exercise changed arithmetic. Max KL is **5.630805 > 0.05**, general-Japanese
top-1 is **115/128 = 89.844% < 90%**, and suite top-1 is **531/576 = 92.188%**.
Same-mode repeats are deterministic, but free-running pair equality is only
**21/54 h16** and **6/54 h32**. Poolside off-shape fallback and lifecycle pass.
Skip promotion timing; retain only the standalone leaf/evidence. H6T/IQ4 remain
production source at **422.786 tok/s**. Do not reopen residual-D4x2 without a
materially different repair/representation and a fresh complete gate; no prompt
or layer subset is admissible
([H7E rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-d4x2-complete-quality-rejected.json) ·
[candidate](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-d4x2-source-mmq-candidate.json) ·
[target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7c-matched-residual-iq3-d4x2-target.json)).

Reject exact Q6 padded-compute **WPF-H7F** under its universal all-role gate:
rowbatch5 wins, but the rowbatch4 role regresses **0.992574x event / 0.977688x
wall**, so no post-timing subset is admissible. Before Q5 timing, select only
four M512 geometries with a real padded tail (`r12/r5/r5/r10`, **61 calls**) and
exclude divisible `r4/r8` roles. Standalone **WPF-H7G exact padded-row Q5
compute is admitted** after RED-first rows1/7/8/9/M512 correctness reaches
**23/23**, first-object codegen converts control dual/scalar FMA sites to
**91/5, 66/14, 66/14, 73/7** at metadata VGPR **194/162/162/162** and
private/spill/scratch0, and cached rocprof names the intended local128 body at
**434.801 us** with zero compiler.

The immutable selection improves weighted event/wall
**136.918/137.009 -> 128.598/129.496 ms (-6.077%/-5.483%)** with all four roles
both-clock positive and byte-exact. A non-adjudicative integrated replay
confirms every role and improves **136.701/136.993 -> 128.691/129.092 ms
(-5.860%/-5.767%)**.

The separately frozen bounded owner qualifies complete natural-M512 state at
KL0/byte identity across logits, all **48/48** hidden boundaries, K/V/
`KVLiveSpans`, repeat, scratch, and teardown. Its cache-only trace records exact
**2/12/12/35 = 61** H7G calls with zero compiler. The independent source RED
then requires one atomic publication: preserve the complete eight-role H5Y map
as named rollback and change only the four padded-tail live values to H7G.
Divisible `r4/r8`, wrong shapes, absent registration/activation, and gfx1151
remain fail-closed.

Fresh selector-unset source qualification passes complete state and improves
H5Y -> H7G fixed C4096/M512 **420.569 -> 423.981 tok/s (+0.811%, 5/5)** and
512/1K/4K **390.598/312.509/195.078 -> 394.355/313.789/195.471 tok/s
(+0.962%/+0.410%/+0.201%)**, all **3/3** exact wins. Clean production reaches
**424.845 tok/s** (**+0.487%** over pre-H7G), with representative cache-only
kernel sum/span **1,192.424/1,213.450 ms** and exact **2,192** dispatches. Q5
falls **255.229 -> 248.888 ms**, narrowing its llama.cpp gap
**196.915 -> 190.574 ms**. Remaining Q5/IQ-down/attention/Q6 gaps are
**190.574/118.366/93.960/66.873 ms** and own **98.194%** of the matched kernel
gap. Continue parity work only with a materially new exact operation; retain
H5Y rollback, unchanged workspace/allocation, and gfx1151 isolation
([H7G production](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q5-padded-compute-production.json) ·
[candidate/runtime](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q5-padded-compute-candidate.json)).

Promote **WPF-H7H exact full-group Q5 compute** after its separately frozen
source RED proves the atomic live-map switch and complete named H7G rollback.
H7H covers both and only the divisible natural-M512 roles: BF16 K3072/N1024
`c8r4` (**92 calls / 24.093 ms**) and BF16 K9216/N3072 `c12r8` (**35 / 80.144
ms**), totaling **104.237 ms / 41.881%** of pre-H7H Q5. It reuses the qualified
unconditional body through separately named gfx1100 exports/keys; H5Y remains
fallback, H7G remains complete-map rollback, and gfx1151 is excluded.

RED-first leaf correctness passes **13/13** and the production object retains
metadata VGPR **72/194**, LDS **512/1,536**, private/spill/scratch0. Fresh
selector-unset H7G -> H7H source is KL0/byte-exact across **48/48** hidden
boundaries, full state, and repeat at unchanged **161,120,256-byte** workspace /
**600,141,856-byte** scratch. Fixed C4096/M512 improves **423.045 -> 426.745
tok/s (+0.874%, 5/5)**; clean 512/1K/4K improves
**+1.042%/+0.896%/+0.477%**, all 3/3. Source tracing records exact **61 H7G +
127 H7H** calls among **2,925** dispatches; all five production-profile requests
preserve that topology at **2,192 dispatches** and zero compiler. Clean
production reaches **427.407 tok/s / 1,185.096-ms** representative kernel sum,
**+0.603%** over H7G and **1.61624x** behind llama.cpp; Q5 falls **248.888 ->
237.185 ms**. Continue parity work from Q5/IQ-down/attention/Q6 gaps
**178.871/119.717/94.715/67.233 ms**, requiring a materially new exact
mechanism
([production](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q5-full-group-compute-production.json) ·
[candidate/runtime](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q5-full-group-compute-candidate.json) ·
[target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7g-matched-full-group-q5-target.json)).

Select target-only **WPF-H7I exact raw-Q6 full-group compute** after reranking
the clean H7H residual. Do not reopen Q5 compact reconstruction/source-MMQ,
IQ3 source-MMQ/load geometry, FlashAttention/wider-qrow attention, or Q6 F16/
ordered-role H7F arithmetic. H7C's three raw-Q6 roles instead expose one
untried exact boundary: natural M512 is divisible by rowbatch8/16, but the
kernel still checks `row < rows` in every unrolled K/row FMA group. Those roles
own **28.482 ms / 34.776%** of current Q6.

Freeze all three roles before timing with no subset salvage. The first and only
actual-weight 5/15/5 screen is byte-exact, finite, lifecycle-clean, and improves
BF16 K12288 event/wall **14.052/13.588 -> 8.191/8.944 ms**, F32 K3072/N9216
**11.047/10.718 -> 5.750/6.471 ms**, and BF16 K9216
**10.741/10.548 -> 6.381/6.559 ms**. Weighted event/wall improves
**35.840/34.854 -> 20.323/21.974 ms (-43.295%/-36.954%)**.

The first same-flags object removes only the inner compute predicate. BF16/F32
code/slots fall **4,228/681 -> 4,060/623** and **4,452/749 -> 4,032/631**;
row comparisons fall **9/17 -> 2/2** and dual-FMAC sites rise **1/1 -> 10/11**.
Memory/reduction operations remain unchanged, metadata VGPR **69/64 <=72**, and
private/spill/scratch are zero. Production remains H7H/H7C **427.407 tok/s /
1,185.096 ms**. Next freeze a three-role RED, keep strict M512/full-group
selection and H7C fallback, add no runtime/source owner, and require complete
H7C/CPU bytes, first-object bounds, named cache-only execution, and immutable-
screen replay before standalone admission
([post-H7H residual / H7I target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7h-matched-raw-q6-full-group-target.json)).

Admit standalone H7I after the RED-first **22/22** matrix, exact first-object
physical gate, immutable actual-weight replay, and named cache-only trace all
pass without tuning. Add only one sibling HIP body, exact-M512 launch path, two
Python wrappers/registry variants, and two gfx1151 exclusions. H7C kernel and
Python source hashes, complete H7C package/live map, runtime dispatch, and
workspace remain unchanged; no H7I capability exists.

The repository object exactly reproduces BF16/F32 code/slots **4,060/623** and
**4,032/631**, metadata VGPR **69/64**, LDS512, and spill/scratch0. M512 H7I is
byte-exact to H7C and sampled CPU values; rows1/7/8/9 retain complete H7C
fallback and invalid rows/shapes fail before HIP loading. The non-adjudicative
replay improves weighted event/wall **35.432/34.617 -> 20.089/21.762 ms
(-43.302%/-37.135%)**, every role positive. Rocprof names exact **2 BF16 + 1
F32** H7I calls at runtime VGPR72/64 with zero compiler. Production remains
H7H/H7C **427.407 tok/s**.

The separately frozen bounded-runtime contract then adds only a complete named
three-role gfx1100 capability; live source remains H7C. Complete M512
H7C/H7I/repeat state is KL0/byte-exact across logits, all **48/48** hidden
boundaries, full KV/`KVLiveSpans`, unchanged scratch, and teardown. Fixed
C4096/M512 improves **426.583 -> 429.000 tok/s (+0.567%, 5/5)**; clean
512/1K/4K gains **+0.763%/+0.441%/+0.194%**, all 3/3. Cache-only integration
records exact **2 BF16 + 1 F32 H7I**, zero H7C, **2,925** total dispatches, and
zero compiler. Qualify bounded ownership and next freeze a separate
source-default RED retaining the complete named H7C rollback; production
remains H7H/H7C **427.407 tok/s** until that source gate
([H7I candidate/runtime](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-full-group-compute-candidate.json)).

Promote **WPF-H7I exact raw-Q6 full-group compute** after the separate source
RED proves the atomic live-map switch, complete named H7C rollback, empty
generic fallback, and unchanged selector/workspace/gfx1151 behavior. Fresh
selector-unset source qualification is KL0/byte-exact across **48/48** hidden
boundaries, full state, and repeat at unchanged **161,120,256-byte** workspace /
**600,141,856-byte** scratch. Fixed C4096/M512 improves **427.903 -> 429.434
tok/s (+0.358%, 5/5)**; clean 512/1K/4K gains
**+0.455%/+0.309%/+0.322%** with positive medians and exact state throughout.

Source tracing records exact **2 BF16 + 1 F32 H7I**, zero H7C, and **2,925**
dispatches on one queue/stream at local128/LDS512/runtime-VGPR72/64/scratch0.
All five clean profile requests preserve **2,192 dispatches** and the same H7I
topology. Clean production reaches **431.310 tok/s / 1,172.241-ms**
representative kernel sum, **+0.913%** over H7H/H7C and **1.60161x** behind
matched llama.cpp HIP; raw-Q6 falls **81.900 -> 74.409 ms**. Continue parity
from Q5/IQ-down/attention/Q6 gaps **176.885/118.449/93.805/59.742 ms** with a
materially new exact mechanism
([production](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-full-group-compute-production.json) ·
[candidate/runtime](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-full-group-compute-candidate.json) ·
[target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7h-matched-raw-q6-full-group-target.json)).

Reject target-screen **WPF-H7J exact Q5 full-grid bounds specialization**. The
single predeclared two-role actual-weight 5/15/5 screen is byte-exact, finite,
allocation-clean, and compiler-free, but the dominant 92-call `c8r4` role
regresses to **0.99954x event / 0.99127x wall**. The 35-call `c12r8` role and
127-call weighted aggregate improve, but the all-role rule is binding and
post-timing subset salvage is forbidden. H7J added no repository source and
changes no production metric. Keep H7H/H7I at **431.310 tok/s** and rerank a
materially different exact operation
([rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q5-full-grid-bounds-rejected.json)).

Select target-only **WPF-H7K exact late-start SWA score-to-weight publication**
after the post-H7J closure audit. Q5 source-MMQ/SGEMM, plane/geometry,
persistence, replay, prefetch, reduction, full-group compute, and bounds routes
are measured closed; H7F also forbids favorable Q6-rowbatch5 salvage. The
largest materially new exact boundary is H6W's starts256/384 SWA score-replay
owner at **72 calls / 62.627 ms**, **54.309%** of current attention.

H7K must retain H6W's first-pass unscaled dots/maxima, fused
`dot*scale-max`, token-order lane-0 denominator, token-order unnormalized PV,
and final divide. Split only score-to-weight publication: lane 0 overwrites each
aligned `float4` record with four weights in denominator order, then all lanes
consume records in a separate PV pass. The model removes **255,135,744** dynamic
weight broadcasts while adding **128,065,536** aligned record operations /
**2.049 GB** logical record traffic; treat that as instruction-form rationale,
not physical traffic or a speed result.

Freeze starts256/384 as inseparable before implementation. RED must bind exact
M128/C512/window512/H72/KV8/D128 preflight, the existing **18,874,368-byte**
aligned plane, H6W/H6A fallback, unchanged maps/workspace/gfx1151, complete
H6W and sampled CPU output, complete finite/nonnegative causal record bytes,
all `KVLiveSpans` fields, poison, and lifecycle. The first repository object
must pass the frozen local32/grid2304x32 opcode/code/slot/VGPR/scratch bounds
before cache-only named tracing or one 5/15/5 screen. Both starts and their
72-call aggregate must win HIP event and synchronized wall; any miss removes all
H7K surfaces without subset salvage, tuning, recompile, or favorable rerun.
Runtime and source qualification remain separate. Production stays H7H/H7I
**431.310 tok/s / 1,172.241 ms**, **1.60161x** behind matched llama.cpp HIP
([post-H7J residual / H7K target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7j-matched-swa-weight-publication-target.json)).

Reject **WPF-H7K** at its first-object physical gate. The immutable object keeps
H6W exact and meets H7K code/slots/VGPR **5,048 B / 875 / 54**, emits the frozen
8 u16 K/V loads, 16 output stores, 28 bpermutes, four exponentials, 56 FMAs,
two b128 record stores, and stays LDS/private/spill/scratch/barrier0. It fails
the binding aligned-record premise: both score/weight reads scalarize to
**0 b128 + 2 b32 load sites** instead of the required two b128 loads.

The frozen no-rerun rule is binding. Do not alter source spelling, recompile,
time a favorable start, or salvage a subset. Skip candidate correctness, named
trace, and the 5/15/5 screen; remove all H7K source/test/key/export/gfx1151
surfaces. Production remains H7H/H7I **431.310 tok/s / 1,172.241 ms** with no
metric change. Rerank a materially different exact operation
([H7K physical rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-swa-weight-publication-physical-rejected.json)).

Select target-only **WPF-H7L exact IQ3 full-batch/live-tail split** after the
post-H7K rerank. IQ-down is the largest distinct actionable family at
**272.309 ms / 118.449-ms matched gap** after Q5's measured closure and H7K's
attention physical miss. Actual natural-M512 routing across all 45 IQ3 layers
contains **230,400** live rows in **33,547** rowbatch8 iterations. **24,650
(73.479%)** iterations are full and cover **197,200 rows (85.590%)**; **8,897**
tails contain **33,200** live rows and **37,976** inactive slots, so H6T spends
**14.150%** of its row-slot compute on values that cannot be stored.

H7L changes only rowbatch ownership inside one separately named H6T sibling.
Split every expert into unconditional complete batches plus at most one bounded
tail. Keep the complete path's interleaved VOPD/FMA order, fused-DPP wave
publication, serial wave sum, and BF16 stores exact. The tail processes only
1..7 live rows but preserves each row's eight ordered magnitude FMAs, scale
multiply, permlanex16+DPP 8/4/2/1 sequence, serial wave0..3 sum, and store.
Keep compaction, row layout, metadata, ABI, allocation/workspace, package/
runtime/source maps, H6T, and gfx1151 unchanged. The **4.200B FMA / 2.333B
exchange** inactive-wave-operation model is rationale only, not timing.

Freeze RED first for every tail size plus rows1/7/8/9/M512, reversed P64/P65,
complete H6T/CPU/poison/finite/lifecycle behavior, strict K1024/N3072/E256
scope, and source-policy isolation. Before timing require local128/grid32768x64,
metadata/runtime VGPR <=101/104, LDS384/512, code <=14,000 B, slots <=2,400,
private/spill/scratch0, a bounded live-tail ISA loop, and named cache-only
execution. Then consume one 5/15/5 actual-weight screen over all **45/45**
layers. Every layer and aggregate must win both clocks; any miss removes all
H7L surfaces without subset salvage, tuning, recompile, or favorable rerun.
Runtime/source qualification remains separate. Production stays **431.310
tok/s / 1,172.241 ms**
([post-H7K residual / H7L target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7k-matched-iq3-live-tail-target.json)).

Reject **WPF-H7L** at the frozen first-object gate. The first and only object
keeps H6T exact at **7,920 B / 1,384 slots / metadata VGPR101 / SGPR78 /
spill0**. H7L is **49,592 B / 9,082 slots / metadata VGPR133 / SGPR107 / 270
SGPR spills**, failing code <=14,000 B, slots <=2,400, VGPR <=101, and spill0.
LDS384, private0, VGPR-spill0, dynamic-stack0, and scratch-instruction0 pass,
but the target is inseparable and any physical miss is binding.

Do not rewrite or recompile H7L, retain a tail/layer/prompt subset, or consume
candidate correctness, named tracing, or the all-45 timing screen. Remove the
H7L body/export/wrapper/key/RED/gfx1151 exclusion, retain H6T production, and
rerank a materially different exact operation. Production remains H7H/H7I
**431.310 tok/s / 1,172.241 ms**
([H7L physical rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-live-tail-physical-rejected.json)).

Reject out-of-tree **WPF-H7M exact IQ3 two-wave/two-K256-partition replay**.
This is distinct from H5T's one-wave/four-partition collapse and H7L's tail
split: two physical waves preserve H6T's four independent K256 trees and serial
0..3 sum. Before timing, a frozen no-spill/minimum-VGPR rule selects LDS
activation at **12,744 B / 2,171 slots / VGPR113 / LDS16,768** over register
activation at **13,132 B / 2,172 / VGPR166 / LDS384**.

All **45/45** actual IQ3 layers remain byte-exact, but every layer loses both
clocks. Aggregate event regresses **246.763 -> 392.180 ms (+58.929%, 0.629x)**
and wall regresses **261.551 -> 377.358 ms (+44.277%, 0.693x)**. Add no
repository/RED/runtime/source surface, retain H6T and **431.310 tok/s**
production, and do not retry one-/two-wave K-partition collapse without a new
cross-tile reuse premise
([H7M rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-two-wave-k-partition-rejected.json)).

Reject out-of-tree **WPF-H7N exact raw-Q6 c16r4 direct ordered
consumption**. Unlike H7F's existing ordered-F32 rowbatch edit, H7N removes the
entire activation-pack/Q6-to-F32 producer leg for all three H6U roles. Its
immutable BF16/F32 object is physically bounded at **8,900/8,872 B /
1,393/1,390 slots / VGPR112 / LDS1,024 / spill0** and keeps the exact
64-FMA/64-permlanex16/256-DPP structure. Reconcile the analyzer's pre-timing
load-site formula from 96 to the emitted **68** against this same object; no
source change, recompile, or candidate timing precedes the correction.

All three actual roles remain byte-exact and lifecycle-clean, but each loses
both clocks. Inclusive H6U -> H7N weighted event regresses **48.267 -> 233.861
ms (+384.516%, 0.206x)** and wall regresses **48.520 -> 231.238 ms (+376.583%,
0.210x)** across **142 calls**. Add no repository/RED/runtime/source surface,
retain H6U/H7I and **431.310 tok/s**, and close direct raw-Q6 ordered-consumer
replacement absent a new decode/reuse premise
([H7N rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-c16r4-direct-rejected.json)).

Reject out-of-tree **WPF-H7O exact raw-Q6 full-group geometry crossover**. H7O
crosses only H7I's constant-32 geometries: both BF16 roles move c4r8 -> c2r16
and the F32 role moves c2r16 -> c4r8, preserving raw-Q6 decode, 32 ordered
FMAs, full-group reduction, output order, ABI, and production topology. The
immutable first object passes every frozen physical bound: BF16 is **4,060 B /
634 slots / VGPR64**, F32 is **4,032 B / 620 / VGPR69**, and both are
LDS512/private/spill/scratch0 with exact 32-permlanex16/128-DPP structure.

All three actual roles are byte-exact and lifecycle-clean. BF16 improves
**1.077x/1.063x** and **1.093x/1.080x** event/wall, but F32 regresses
**0.912x/0.913x**. Although the three-call aggregate improves **21.909 ->
21.314 ms event (1.028x)** and **21.905 -> 21.488 ms wall (1.019x)**, the
predeclared all-role gate is binding; do not salvage only BF16 after timing.
Add no repository/RED/runtime/source surface, retain H7I and **431.310 tok/s**,
and rerank a materially different exact operation
([H7O rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-full-group-geometry-crossover-rejected.json)).

Reject no-code/no-timing **WPF-H7P H7E candidate-distance-only BF16 boundary
repair**. The prompt-independent natural-M512 audit exposes H7E's pre-BF16 FP32
accumulators and compares them with exact H6T across all **45** IQ3 layers.
Exactly **16,306,295 / 707,788,800 outputs (2.30384%)** differ after BF16
publication. Distance to the candidate's nearest BF16 boundary is not a sparse
complete certificate: risk/recall is **6.234%/43.799%** at 1/16 cell and
**24.931%/68.070%** at 1/4 cell. The latter leaves **5,206,620** mismatches.
At 1.0 cell the guard selects **99.719%** of outputs, still misses **14,702**
values, and its ideal zero-overhead linear model is **0.592x** exact. No tested
threshold captures all mismatches. Add no guard/queue/kernel/runtime/source
surface and consume no candidate timing. This closes repair based only on the
fast accumulator's BF16-boundary distance, not a materially different
prompt-independent error-size certificate or activation representation.
Production remains **431.310 tok/s**
([H7P rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-d4x2-boundary-repair-rejected.json)).

Reject no-code/no-timing **WPF-H7Q/H7R third-plane residual certificates**.
H7Q runs an actual D4x3 FP32 probe: D4x2/D4x3 BF16 disagreement selects
**16,306,421 / 707,788,800 (2.30385%)** outputs and recalls **99.7364%** of D4x2
mismatches, but leaves **42,981** wrong. Union with D4x3 boundary distance is
complete only at **705,810,744 / 707,788,800 (99.7205%)** risk; moreover the
prior producer-inclusive D4x3 path was only **1.0063x** exact with **27/45**
layer wins, so even incomplete disagreement repair models **0.984x** exact
before comparison/queue overhead.

H7R is the materially distinct conservative certificate: outward-rounded
post-D4x2 K32 residual maxima times exact-IQ3 weight L1 norms, grouped at
K64/K128/K256/K1024. Every zero-margin form captures all **16,306,295** observed
mismatches, but risk density is **74.5071%/81.1992%/86.4963%/92.8985%**. Only
**30.6591%** repair could break even before any guard cost. The best model is
already **0.695x** exact with every guard/sidecar/queue/locality/launch cost
deleted and **0.610x** under the declared read ceiling. Add no repair surface,
retain **431.310 tok/s**, and rerank outside IQ3 residual repair
([H7Q/H7R rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-residual-certificates-rejected.json)).

Select target-only **WPF-H7S exact raw-Q6 c2r32 packed-activation cross-row
reuse** after reranking outside IQ3 repair. Production remains **431.310 tok/s /
1,172.241 ms**, with Q5/IQ-down/attention/Q6 gaps **176.885/118.449/93.805/
59.742 ms**. The H6U portion of Q6 consists of **142** activation-pack + exact
Q6-to-F32 producer + ordered-consumer triples at weighted event/wall
**48.267/48.520 ms**. H7N proved one-launch row-major c16r4 direct decode is
byte-exact but 4–5x slower; H7O adjudicated only the separate H7I constant-32
roles. H7S therefore introduces cross-row decode reuse rather than reopening
either target.

Keep H6U's tile-K-row activation ABI and exact fallback. Pack strict M512 into
rowbatch32, assign two raw-Q6 columns per local128 workgroup, and apply each
decoded pair to 32 packed BF16 rows. Preserve every output's thread-local
`k=tid+128n` FMA order, signed `scale*quant` conversion, H6U
permlanex16+DPP tree, serial wave sum, and output store. At fixed 64
accumulators, the source model reduces H7N from **16 column decodes / 68 load
sites** to **2 decodes / 12 sites** (eight Q6 fields plus four b128 activation
records), while removing only the F32 producer launch. This models **142 fewer
request dispatches (2,192 -> 2,050)** and **0.937x** current logical input bytes;
it is not physical ISA, traffic, or timing evidence.

Freeze all three 2/46/94-call M512 roles together. RED must establish complete
primitive/composite H6U and sampled CPU bytes, exact pack layout, poison/
finite/lifecycle, strict fail-closed shape handling, unchanged workspace/maps,
and gfx1151 absence. Before timing require the first object at local128/wave32,
LDS1,024, VGPR<=136, SGPR<=96, code<=14,000 B, slots<=2,400,
private/spill/scratch0, exact 64-FMA/64-permlanex16/256-DPP structure, and four
b128 activation plus eight Q6-field load sites. Require named cache-only pack +
consumer execution with no producer and zero compiler. Then one immutable
actual-weight 5/15/5 screen must win every role and weighted aggregate on both
clocks. Any miss removes H7S without role/geometry/prompt subset, tuning,
recompile, or favorable rerun; runtime/source qualification is separate
([post-H7R residual / H7S target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7r-matched-raw-q6-cross-row-reuse-target.json)).

Reject **WPF-H7S** under that immutable rule. The sole object passes every
physical bound at **5,912/5,884 B**, **864/860 slots**, and **VGPR112 / SGPR24 /
LDS1,024 / spill/scratch0**, with the exact four-b128/eight-Q6-load/64-FMA/
64-permlanex/256-DPP instruction form. Complete all-role correctness is **8/8**
and the compiler-free named trace has only pack→consumer pairs. Yet every role
loses both clocks: **0.290/0.320**, **0.402/0.411**, and **0.293/0.304**
event/wall. The 142-call aggregate moves H6U→H7S **49.193→149.544 ms event
(0.329x)** and **49.721→146.161 ms wall (0.340x)**. Delete all candidate and RED
surfaces, skip runtime/source qualification, retain H6U and **431.310 tok/s**,
and forbid role/geometry/recompile/rerun salvage. Rerank outside direct raw-Q6
ordered-consumer shapes and IQ residual repair
([H7S rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-cross-row-reuse-rejected.json)).

Reject **WPF-H7T quality-gated late-start QK-only tensorized score replay**.
The immutable source passes RED→GREEN **10/10**, first-object resource limits,
and a compiler-free four-role key-widen→query-pack→one-QK→consumer trace with
no PV/value-widen/standalone-softmax. The complete **18-prompt / 576-step**
code/general-English/general-Japanese/mixed heldout lane observes all
**7,008/7,008** expected global/SWA starts256/384 calls. It is finite and keeps
**562/576 (97.569%)** top-1 with every category >=90%, deterministic repeats,
oracle, and exact lifecycle recovery, but maximum KL reaches **0.393845**, above
the mandatory **0.05** ceiling in every category.

Consume no H7T 5/15/5 admission timing. Remove every candidate body/export/
wrapper/key/owner/RED/gfx1151 exclusion and restore the seven target-state
implementation hashes plus H6W/H6Z **10/10** cache-only controls. Production
remains **431.310 tok/s / 1,172.241 ms**, **1.60161x** behind matched llama.cpp
HIP. Do not salvage a family/start/head/layer/prompt subset, retune BLAS,
rewrite/recompile, or favorably rerun H7T; rerank clean production outside
changed-association attention and the already closed exact Q5/IQ-down/raw-Q6
routes
([H7T rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-qk-only-score-replay-quality-rejected.json) ·
[H7T target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7s-qk-only-score-replay-target.json)).

Select target-only **WPF-H7U exact stable parallel MoE active compaction** after
reranking outside those closed arithmetic routes. The current one-workgroup
active compactor owns **25.187 ms / 47 calls**, or **32.960%** of the
**76.417-ms** remaining bucket and **5.497%** of the complete matched kernel
gap. The following **7.717-ms / 47-call** packed-hidden gather remains a
separate unchanged operation. hipEngine already registers a materially
distinct stable three-stage sibling—per-expert count, fixed-256 Blelloch prefix
plus active-ID ballots, and per-expert ballot-ordered scatter—but gfx1100 still
explicitly defaults to serial while gfx1151 owns the qualified parallel mode.

Transfer only that existing scheduler to gfx1100: preserve exact starts, active
IDs/count, sorted lanes/source rows/weights, MMQ tile map, packed hidden,
router, gate/up/down arithmetic, allocation, and workspace. The topology changes
**47 serial launches to 141 parallel stages** (net **+94**, expected request
**2,286**); this is acceptable because measured work, not launch count, is the
premise. Prior gfx1151 exact **7/7** wall wins and **2.564-ms** parallel trace
are rationale only. Freeze RED before a gfx1100 package-capability change, then
require all-47 natural metadata/full-state identity, named cache-only
47+47+47/zero-serial tracing, and one all-layer plus aggregate both-clock 5/15/5
gate. Forbid layer/expert/routing-pattern/length subsets, retuning, rewrite,
recompile, or favorable reruns. Production remains **431.310 tok/s**; no W7900
candidate or speed claim exists
([H7U target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7t-parallel-moe-compaction-target.json)).

H7U now passes standalone admission through one bounded default-off gfx1100
package constant; production and the source-default resolver remain serial.
GREEN is **9/9**. One natural M512 serial/candidate gate proves exact metadata
and packed hidden on all **47** MoE layers, exact **48/48** hidden boundaries,
logits, KV/`KVLiveSpans`, token **2930**, deterministic state, and lifecycle.
The unchanged cached object exposes all three local256/wave32 bodies at metadata
VGPR **10/17/31** with private/spill/scratch0. Selected-region tracing records
exact **47 count + 47 prefix + 47 scatter**, zero serial, unchanged 47 gather,
and **2,286 application dispatches** on one queue/stream with zero compiler.

The immutable first actual-routing 5/15/5 screen is exact and wins **47/47**
layers on event and synchronized wall. Aggregate serial→parallel moves
**20.508→1.297 ms event (15.813x)** and **20.701→1.445 ms wall (14.331x)**;
minimum layer speedups are **14.012x/12.690x**. Retain the bounded capability
and proceed to a separate runtime/source RED. Require fixed C4096/M512 and clean
512/1K/4K non-regression before changing the live source owner; until then
production remains **431.310 tok/s / 1,172.241 ms / 2,192 dispatches**
([H7U candidate](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-parallel-moe-compaction-candidate.json)).

Promote H7U after the separate source RED and complete request-level gates.
Atomically replace the bounded H7U capability with
`LAGUNA_MOE_GROUP_COMPACT_MODE="parallel"`; retain explicit serial rollback and
leave gfx1151 local. The bounded fixed C4096/M512 gate is exact and improves
**430.412→436.602 tok/s (+1.438%, 5/5 wins)**. Clean source-default 512/1K/4K
improves **+1.371%/+1.245%/+0.626%**, with exact state, lifecycle recovery, and
**3/3** wins at every length.

Clean matched production is **437.189 tok/s / 1,171.117-ms wall**, **+1.363%**
over H7I and **1.58007x** behind matched llama.cpp HIP **690.791 tok/s**. The
five-request selected-region trace proves exact **47+47+47**, zero serial,
unchanged 47 gather, **2,286 dispatches**, one queue, and zero compiler.
Compaction falls **25.187→1.155 ms**; representative kernel sum falls
**1,172.241→1,160.833 ms**. Remaining now beats the matched comparator, while
Q5/IQ-down/attention/Q6 gaps remain **178.225/121.564/94.784/60.314 ms**.
Continue only with a materially distinct exact mechanism; do not reopen the
closed Q5 geometry/representation, IQ residual, changed-association attention,
or raw-Q6 direct-consumer routes
([H7U production](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-parallel-moe-compaction-production.json)).

Select target-only **WPF-H7V exact dequantized-Q6 full-batch/live-tail
predicate elimination**. H6U's 142 exact F32-weight consumers own **49.191 ms /
65.604% of Q6**. At M512, **1,757,184 / 1,763,328 workgroups (99.652%)** are
complete, but the shared H6U body retains dynamic compute/store row predicates.
Keep the activation pack, Q6-to-F32 producer, H6U FMA/DPP/LDS/store sequence,
and allocation/workspace exact. Run one predicate-free full-prefix launch per
role call; for the 96 rowbatch5 calls, follow it with one unchanged H6U
remainder-2 tail. The modeled topology is **142→238 consumers** and
**2,286→2,382 request dispatches**.

This is a separate dequantized-Q6 operation from H7I/H7N/H7S raw decode, H7J Q5
full-grid, and physically rejected H7L IQ3 live-tail. Freeze RED before code.
Require all-role complete bytes/CPU/full+tail recomposition, strict fallbacks,
first-object no-row-compare and H6U resource/opcode bounds, cache-only
142-pack/143-producer/142-full/96-tail topology, and one all-three-role plus
weighted-aggregate both-clock 5/15/5 screen. Any miss removes H7V without role,
output-type, layer, prompt, length, retune, recompile, or favorable-rerun
salvage; no candidate or speed result exists
([H7V target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7u-q6-full-batch-live-tail-target.json)).

Reject **WPF-H7V** after the first immutable all-role timing screen. The one
object passes all frozen physical gates at **5,808/6,960/6,928 B**,
**873/1,001/996 slots**, and metadata **VGPR108/139/139** for BF16-r4,
BF16-r5, and F32-r5, respectively; FMA, permlanex16, DPP, barrier, LDS,
private/spill/scratch, and complete outputs remain exact. GREEN passes **9/9**
and tracing records exact **142 full + 96 H6U tail / 2,382 dispatches** with
zero compiler. Rowbatch4 improves **1.00255x/1.00495x**, but both rowbatch5
roles lose both clocks and cannot be salvaged. Weighted H6U→H7V regresses
**47.949→48.680 ms event (0.985x)** and **48.522→49.162 ms wall (0.987x)**.
Remove every H7V kernel/wrapper/key/RED surface, skip runtime/source work,
retain H6U plus **437.189 tok/s**, and rerank a materially different operation
([H7V rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q6-full-batch-live-tail-rejected.json)).

Reprofile clean committed production after H7V before selecting another target.
Five exact requests report **437.836 tok/s** in the established fixed harness
and **434.611 tok/s** in the generic comparator harness. A fresh cache-only
selected-region trace retains **2,286 dispatches** and records **1,153.347-ms**
kernel sum. Current Q5/IQ-down/attention/Q6 gaps are
**177.673/119.203/94.094/60.051 ms**; gate/up is effectively at parity and
remaining remains ahead.

Reject **WPF-H7W exact H6T output-partition P128 crossover** after its sole
immutable all-45-layer screen. The one **469,056-byte** object passes every
physical gate: P128 and H6T P256 are both **7,920 B / 1,384 slots / VGPR101 /
SGPR78 / LDS384 / spill0**, with exact **216 FMA, 24 permlanex16, 96 DPP adds,
24 LDS b128 loads, 12 LDS stores, and two barriers**. GREEN is **12/12** and
cache-only tracing records exact **45 H7W + two unchanged IQ4 / 2,286
dispatches**, local128/grid16,384×64/runtime-VGPR104/LDS512/scratch0, one
queue/stream, positive durations, and zero compiler.

Every complete output is byte-exact and lifecycle-clean, but only **16/45**
layers improve event and wall. H6T P256→H7W P128 moves the all-layer event sum
**260.663→261.392 ms (+0.280%, 0.99721x)** and synchronized wall
**260.731→262.135 ms (+0.538%, 0.99464x)**. Both the per-layer and aggregate
gates fail. Remove every H7W export/wrapper/key/RED/gfx1151-exclusion surface,
run no runtime/source qualification, retain H6T P256 plus production **437.189
tok/s**, and forbid layer/expert/routing/prompt/length subset, partition
retune, body rewrite, recompile, or favorable rerun. The modeled 50%
workgroup/activation-record reduction is not a speed result
([H7W rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-output-p128-rejected.json) ·
[target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7v-iq3-output-p128-target.json)).

Select target-only **WPF-H7X exact H6W one-slot BF16 K/V software pipeline**
after the clean H7W rejection. Production remains **437.189 tok/s / 1,153.347
ms / 2,286 dispatches**, **1.58007×** behind matched llama.cpp HIP **690.791
tok/s / 714.008 ms**. Current campaign-start→production→llama component times
are Q5 **1,270.458→235.987→58.314 ms**, IQ-down
**557.091→273.063→153.860**, attention **488.304→115.607→21.512**, Q6
**157.073→74.719→14.668**, gate/up **460.143→400.672→397.805**, and
remaining **68.623→53.299→67.849**.

H6W owns **72 calls / 62.656 ms**, **54.198%** of attention and **14.262%** of
the total matched kernel gap. Its local32/wave32 body is **4,984 B / 871 slots
/ metadata VGPR54 / runtime VGPR56 / LDS0 / spill0**. Both steady-state K and V
paths issue four BF16 loads then immediately drain `vmcnt(3→0)` before useful
current-slot work. Across both natural starts and all 36 SWA layers,
**63,866,880 / 64,032,768 slots (99.7409%)** per K pass and per V pass have a
next slot.

Freeze RED before adding a separately named gfx1100 H6W-equivalent sibling.
Preload slot0, issue next-slot K before complete current-slot ordered
QK/reduction/max/store work, and independently issue next-slot V before current
exp/denominator/broadcast/PV work. Preserve every consumed value and arithmetic
order, score records, all `KVLiveSpans` fields, allocation/workspace, request
dispatches, H6W source/default, and fallback. Require sole-object local32,
VGPR≤64, LDS/private/spill/scratch0 and physical next-slot overlap; complete
starts256/384 H6W+CPU identity; exact **72 H7X + 72 H6A + 24 H6N + 24 H6Z /
2,286-dispatch** tracing; then one starts256/384 plus weighted-aggregate
both-clock 5/15/5 screen. Any miss removes H7X without start/layer/head/prompt/
length subset, prefetch-distance retune, rewrite, recompile, or favorable rerun.
No candidate has been built or executed and no speed result exists
([H7X target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7w-swa-kv-prefetch-target.json)).

Reject **WPF-H7X** at the sole first-object physical-overlap gate before any
candidate execution. H7X passes all resource/opcode ceilings at **5,320 B / 931
slots / metadata VGPR54 / SGPR44 / LDS0 / private0 / spill0**, while H6W stays
exactly **4,984 B / 871 / VGPR54 / SGPR40**. The candidate adds the expected
second static K and V four-u16 clauses while preserving 32 bpermutes, one b128
record load+store, four exp, 56 FMA, 41 FMAC, 16 output stores, and no barrier/
scratch.

The binding overlap requirement fails twice: in both steady next-slot clauses,
the final `global_load_u16` is immediately followed by `s_waitcnt vmcnt(3)` and
the remaining `vmcnt(2→0)` drains, with **zero current-slot instructions**
before the first wait. Therefore no next-K/current-QK or next-V/current-PV
overlap survived codegen. Consume no correctness/GREEN, trace, timing, runtime,
or source gate. Remove every H7X body/export/wrapper/key/gfx1151-exclusion and
RED, forbid subset/distance-retune/rewrite/recompile/rerun salvage, and retain
H6W plus **437.189 tok/s / 1,153.347 ms / 2,286 dispatches**
([H7X rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-swa-kv-prefetch-physical-rejected.json) ·
[target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7w-swa-kv-prefetch-target.json)).

Select target-only **WPF-H7Y exact H6W lane-major BF16 K/V mirror loads** from
the byte-identical post-H7X production boundary. Retain **437.189 tok/s /
1,153.347 ms / 2,286 dispatches** and the matched **690.791 tok/s / 714.008
ms** comparator. H6W owns **72 calls / 62.656 ms** and currently reads each
head as `[part4][lane32]`, issuing four `u16` loads plus four staged waits in
each K or V pass so each lane receives dimensions `lane + 32*part`.

H7Y changes only caller-provided cache layout to `[lane32][part4]`, mapping
`mirror[base + lane*4 + part] = natural[base + part*32 + lane]`. Each lane's
same four BF16 values form one aligned 8-byte record, enabling one required b64
load per pass before the unchanged H6W QK/score/max/exp/denominator/PV/divide
order. Across the complete route, load-issue slots model
**512,262,144→128,065,536 (-75%)**, removing **384,196,608** load and
**384,196,608** wait issue slots at unchanged **32,784,777,216-byte** K/V
payload. This is target arithmetic, not speed evidence.

Freeze RED and the sole-object physical contract before implementation:
exactly **2 b64 / 0 u16 / ≤2 vmcnt waits**, unchanged b128 record, bpermute,
exp/FMA/FMAC/store counts, local32/wave32, code≤5,200 B, slots≤900,
metadata/runtime VGPR≤56/56, SGPR≤48, and LDS/private/spill/scratch0. Require
complete transpose round-trip plus starts256/384 H6W/CPU/output/record/span/
poison/repeat/lifecycle identity, then one immutable all-72 actual-state 5/15/5
screen where both starts and aggregate win event and wall. Forbid subset,
packing-width retune, rewrite, recompile, or favorable rerun salvage.

H7Y is now admitted as an explicit standalone leaf. Its sole object is **4,900
B / 855 slots / metadata VGPR54 / SGPR40 / spill0**, with exactly **2 b64 / 0
u16 / 2 waits** and unchanged record/reduction/math/store opcodes. GREEN passes
**6/6**; actual-cache tracing records exact **72 H7Y** at runtime VGPR56/LDS0/
scratch0 on one queue with zero compiler. All 72 actual-layer outputs and score
planes are byte-exact.

The first immutable all-72 screen improves H6W→H7Y aggregate **56.607→56.259
ms event (-0.616%) / 56.559→56.317 wall (-0.428%)**, with starts256 and384 each
positive on both clocks.

The separately RED-gated bounded owner now qualifies default-off. It adds exact
**72 MiB / 72 allocations** of SWA mirrors and one fused natural+lane-major
writer without increasing dispatches. The first writer object is **1,724 B /
357 slots / metadata VGPR23 / SGPR53**, two F32 loads, four BF16 stores, and
LDS/private/spill/scratch0. Complete M512 is KL0/byte-exact across **48/48**
boundaries, logits, K/V/spans, repeat, and lifecycle. Named tracing is exact
**144 fused writers + 72 H7Y + 72 H6A + 24 H6N + 24 H6Z / 2,286 dispatches**,
one queue/stream and zero compiler. Writer-inclusive fixed C4096/M512 improves
**436.120→436.785 tok/s (+0.152%)**; 512/1K/4K medians improve
**+0.0530%/+0.1217%/+0.0043%**.

The separate source-default gate rejects promotion at its first binding fixed
median. Selector-unset M512 remains KL0/byte-exact, but H6Z/H6W rollback→H7Y
moves **436.403→436.275 tok/s (-0.0294%, 0.99971×; 2/5)**. Skip all later source
gates, restore H6Z/H6W production **437.189 tok/s**, and retain H7Y default-off
without rerun or subset salvage
([source rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-swa-lane-major-cache-source-rejected.json) ·
[runtime](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-swa-lane-major-cache-runtime-candidate.json) ·
[standalone](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-swa-lane-major-cache-candidate.json) ·
[target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7x-swa-lane-major-cache-target.json)).

Reject the next two changed-arithmetic repair premises before adding code. On
the exact natural-M512 trajectory, H5A Q5 SGEMM changes only **123,111 /
134,742,016 BF16 outputs (0.0914%)**, but candidate-boundary mismatches reach
cell center and complete recall selects **all BF16 outputs**; its F32 roles
change **197,527,937 / 204,189,696 outputs (96.737%)**. H2 source-F16-WMMA
attention changes **44,171,810 / 207,618,048 (21.276%)** post-softplus-gate
BF16 values and touches **405,132 / 405,504 (99.908%)** exact qrow4/head groups.
An omniscient zero-overhead group-repair lower bound is **136.255 ms** versus
current exact attention **115.385 ms**, so no prompt-independent certificate
can restore positive economics. Exact live state remains unchanged after every
candidate call, token/lifecycle pass, and no compiler runs. Add no guard, queue,
repair, runtime, or source surface; rerank outside both premises
([repair-audit rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q5-attention-repair-audits-rejected.json)).

Select **WPF-H8A exact resident global-Q5 tile-K-col F32 cache** as the next
materially distinct target. Scope is the complete architecture-defined class:
all 12 full-attention layers' `attn_q` and `attn_output`, never a prompt,
route, token, or arbitrary layer subset. Reuse the retained exact coltile16
producer once at setup to populate **24 × 75,497,472 = 1,811,939,328 bytes
(1.6875 GiB)**, then keep the unchanged activation pack and H7G padded-compute
consumer while deleting **24 request producers / 5.596 ms** and modeling
**2,286→2,262 dispatches**. A stricter owner+child+24-buffer audit passes exact
M512/token2930 with **4.167 GB** free, zero compiler, and lifecycle recovery.
No HIP body/object or candidate speed result exists. Commit the target, then
freeze all-or-nothing allocation/sharing, complete 24-plane bytes, complete
state, setup/request topology, fixed and 512/1K/4K both-clock RED gates; source
promotion remains separate and the full-family sidecar remains forbidden
([H8A target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h7y-resident-q5-global-f32-cache-target.json)).

H8A's separately committed Python-only owner now qualifies bounded default-off.
The HIP source/object remain byte-identical. One immutable all-or-nothing map
owns exact **24 × 75,497,472 = 1,811,939,328 bytes**; every actual plane matches
a fresh retained producer over all bytes. Complete M512 is KL0/top-1 100% and
byte-exact across **48/48** boundaries, logits, hidden state, KV/spans, repeat,
and lifecycle. Named tracing records exact **24 setup / zero request producers +
24 target packs + 24 H7G consumers / 2,262 dispatches**, one queue/stream and
zero compiler. Fixed C4096/M512 improves **436.765→438.368 tok/s (+0.367%,
5/5)**; clean 512/1K/4K medians improve **+0.748%/+0.332%/+0.257%**, all 3/3
paired wins. At this bounded checkpoint retain H6Z/H6W source **437.189 tok/s**
and freeze source promotion separately
([H8A runtime](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-resident-q5-global-f32-cache-runtime-candidate.json) ·
[H8A target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h7y-resident-q5-global-f32-cache-target.json)).

H8A source promotion now passes the complete frozen source and post-commit gate.
Selector-unset complete planes/state remain exact; fixed transient H7G→source
improves **435.272→437.286 tok/s (+0.463%, 5/5)** and clean 512/1K/4K improves
**+0.290%/+0.142%/+0.215%**, all 3/3. Clean commit `c4ea62347` reaches
**440.353 tok/s (+0.724% over 437.189)**. Five exact profiled requests record
**2,262 dispatches**, **1,151.215-ms** representative sum / **1,174.598-ms**
median span, exact **24 setup / zero request coltile16 producers**, one
queue/stream, and zero compiler. Retain the transient H7G/allocation-failure
fallback and prohibit every role/layer/prompt/length subset. Current matched
kernel gaps rank Q5 **173.395 ms**, IQ-down **120.186**, attention **94.231**,
and Q6 **59.985**; rerank those families for the next materially distinct exact
target
([H8A production](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-resident-q5-global-f32-cache-production.json)).

Select **WPF-H8B exact scoped tile-K-row activation-pack reuse** after auditing
the complete post-H8A request without changing execution. H5Y/H6U issue **330**
packs; **95** immutable recurrence runs contain **107** redundant calls: 12
full-attention Q/K/V triples remove 24, 35 SWA K/V pairs remove 35, 46 shared-
Q5 gate/up pairs remove 46, and the dense-Q5 plus layer-47 shared-Q6 pairs remove
two. Reuse is valid only inside an explicit host scope and only for an identical
`(input pointer, activation pointer, rows, K, row batch, stream)` key after a
successful producer. Different keys, streams, scopes, failed producers, c=1,
non-M512, registry misses, and unmeasured backends retain the producer.

The complete target models **330→223 packs / 2,262→2,155 dispatches**. Current
profile medians assign **2.342313 ms** to the redundant calls, yielding a
zero-replacement-cost wall ceiling of **441.242 tok/s (+0.202%)**; this is not a
candidate performance claim. Freeze RED before implementation, keep the HIP
source/object byte-identical, require all 95 runs together, complete state and
scope/failure isolation, exact **223-pack / 2,155-dispatch** tracing, then fixed
and 512/1K/4K positive medians before a separate source-default gate. Forbid
attention-only, shared-only, layer, role, prompt, token, length, or favorable-
rerun salvage
([H8B target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8a-activation-pack-reuse-target.json)).

H8B now passes bounded qualification without changing a device body, object,
allocation, workspace, or output byte. Scope/runtime and retained coverage is
**103/103**; complete M512 remains exact at token2930/position511 and executes
**223 packs (24 resident + 199 transient)**. The committed named trace proves
**330→223 packs / 2,262→2,155 application dispatches**, unchanged non-pack
kernel names/counts, one queue/stream, expected resources, and zero compiler.
Fixed C4096/M512 improves **438.412→438.919 tok/s (+0.116%, 4/5)**, while
clean 512/1K/4K improves **+0.148%/+0.175%/+0.152%** with exact state and
lifecycle. Retain the complete owner default-off and freeze source promotion
separately; H8A remains production **440.353 tok/s**
([H8B runtime](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-scoped-activation-pack-reuse-runtime-candidate.json) ·
[H8B target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8a-activation-pack-reuse-target.json)).

H8B source promotion now passes the frozen selector-unset and post-commit
production gates. Fixed rollback→source improves **+0.258% (5/5)** and clean
512/1K/4K improves **+0.109%/+0.0097%/+0.055%**. Clean commit `6b9411b15`
reaches **440.893 tok/s (+0.122% over H8A)**. Five exact profiled requests have
**2,155 dispatches**, **1,146.420-ms** median sum / **1,166.621-ms** span,
one queue/stream, and zero compiler. Retain disabled rollback and prohibit every
class/layer/prompt/length subset. Current matched kernel gaps rank Q5 **172.115
ms**, IQ-down **119.303**, attention **93.837**, and Q6 **58.652**; rerank those
families after candidate-seam cleanup
([H8B production](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-scoped-activation-pack-reuse-production.json)).

WPF-H8C exact dual-weight shared-Q5 gate/up consumption is rejected before
runtime integration. The first object and leaf checks pass: local128,
metadata/runtime VGPR **134/136**, LDS **1 KiB**, zero private/scratch/spills,
one shared activation load, and byte-exact rows17/33/M512 gate/up outputs versus
H7H and sampled CPU references. A cache-only named trace confirms one H8C call
at **808.921 µs** with zero compiler activity.

The frozen complete-class timing gate does not pass. All **46/46** actual Q5
shared gate/up pairs remain byte-exact and finite, but only **14/46** win both
clocks. Summed H7H→H8C event time is **27.8051→27.8323 ms (0.9990×)** and wall
time is **28.0210→28.0053 ms (1.0006×)**. Honor the no-salvage boundary: remove
all candidate code/capabilities/tests, skip runtime/state/topology/fixed/length
qualification, and retain H8B/H7H production at **440.893 tok/s / 2,155
dispatches**. Do not retry this dual-weight schedule without a materially new
physical operation
([H8C rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-shared-q5-dual-consumer-rejected.json) ·
[H8C target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8b-shared-q5-dual-consumer-target.json)).

Reject **WPF-H8D complete-class Q6 exact-value F32 SGEMM** before target
publication. Screen all six ordinary-Q6 M512 shapes/**144 calls** with the
current H6U/H7I/H5I routes as control and the retained exact Q6-to-F32
producer, exact BF16 widening, rocBLAS SGEMM, and BF16 result cast as candidate.
Five shapes win both clocks; the weighted diagnostic moves **74.099→40.969 ms
event (1.809×)** and **74.469→41.232 ms wall (1.806×)**, materially below the
H8B Q6 **73.320-ms** family. F32 K3072×N72 is binding and regresses
**0.03965→0.09167 ms event (0.4325×)** and **0.04260→0.09559 ms wall
(0.4456×)**. Exact operands, finite outputs, primitive quality, lifecycle, and
zero compiler all pass. Honor the prospectively declared complete-class rule:
do not salvage the other 143 calls after timing, publish no target/RED/runtime,
skip the 576-step quality lane, and retain H8B **440.893 tok/s / 2,155
dispatches**
([H8D rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q6-k-f32-sgemm-complete-class-rejected.json)).

Reject **WPF-H8E synthetic-quality-selected F32 hipBLASLt attention** without
restoring the old H5B runtime map. Enumerate all **128 = 8 shapes × 4 QK × 4
PV** zero-workspace combinations on fixed synthetic operands. Every combination
is finite/primitive-green and each shape exposes four numerical classes. The
prospective closest-output-then-fastest-wall rule changes five of six H5B
indices and models **109.065→85.822 ms event / 86.427-ms wall-proxy**. Matched
C4096/M512 is deterministic at token2930, top-1 100%, and KL **0.000231**.

The binding 18-prompt/**576-step** lane observes exactly **10,512** candidate
stacks and clean lifecycle, but fails every category's KL ceiling: code/general
en/general ja/mixed maxima are **0.101711/0.201151/0.391103/0.178258**. Suite
top-1 is **563/576 (97.743%)**. Diagnostic prefill improves
**328.443→429.801 tok/s (1.3086×)**, but quality failure forbids a target, RED,
algorithm map, owner, or source promotion. Do not try another numerical class
post-result; retain H8B **440.893 tok/s / 2,155 dispatches**
([H8E rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-quality-selected-f32-attention-algorithms-rejected.json)).

Select **WPF-H8F exact resident shared-Q5 tile-K-col F32 cache** as the next
materially distinct target. The complete architecture class is both
`ffn_gate_shexp` and `ffn_up_shexp` across layers1–46: **92** immutable raw-Q5
K3072×N1024 tensors, each using the retained coltile8/rowbatch4 BF16-output
producer and unchanged H7H consumer. The clean H8B trace observes exactly 92
small-grid producers in each of five requests at **3.439745 ms median**; H8B
already shares their activation into 46 packs. H8F changes plane lifetime only,
not consumer arithmetic, and models request dispatches **2,155→2,063**.

A source-unchanged live audit holds all **92 allocations / 1,157,627,904 bytes
(1.078125 GiB)** alongside H8A and exact M512, reaches token2930/position511,
remains finite, recovers all tracked allocation, sees zero compiler, and leaves
**3,009,413,120 bytes (2.802734 GiB)** free. Extend H8A's immutable map
all-or-nothing from **24→116** planes; any H8F allocation/setup failure retains
the admitted 24 global planes and complete transient shared path. Commit this
target before RED, change no HIP body/object, and require all 92 planes,
complete state/topology, fixed C4096/M512, and clean 512/1K/4K both-clock wins.
No partial-plane, layer, role, prompt, token, route, length, threshold, or
favorable-rerun salvage is allowed
([H8F target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8e-resident-shared-q5-f32-cache-target.json)).

Reject H8F at the binding clean-length transfer gate. The Python-only bounded
owner and reused device primitives pass focused GREEN **6/6**, complete
**116/116** plane identity, KL0/exact full M512 state and repeat, lifecycle, and
zero-compiler gates. Named tracing proves exact **24 global + 92 shared** setup
producers, then **0** shared request producers with unchanged **46** shared
packs/**92** H7H consumers and **2,155→2,063** request dispatches. Fixed
C4096/M512 is positive at **439.301→439.811 tok/s (+0.1162%, 5/5)**.

The immutable first three-pair 512/1K/4K gate is nevertheless
**+0.3421%/-0.0710%/-0.00171%**; only 512 wins. Honor the declared every-length
rule without subset or rerun: remove all H8F package/registry/runtime/backend
and RED-test surfaces, retain the unchanged HIP object and H8B production, and
do not retry shared-Q5 residency without a materially different representation
or operation
([H8F rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-resident-shared-q5-f32-cache-rejected.json)).

Reject **WPF-H8G complete existing global-qrow6 transfer** before target
publication. Screen the already-registered sibling-RDNA3 dense-initial qrow6
body on W7900 at the architecture-defined starts128/256/384, retaining start0
on H6N. Against current H6N/H6Z, qrow6 wins both clocks only at start128 but is
not F32-bit identical at any start (**730,971–749,888 mismatches**). Starts256
and 384 lose both clocks, and the 36-call aggregate regresses
**15.869→21.545 ms event (0.7365×)** and **16.078→21.803 ms wall (0.7374×)**.
Finite output, lifecycle, and zero-compiler checks pass but cannot waive the
complete-class identity/timing gate. Add no target, RED, runtime selector, or
source map; prohibit start128-only salvage and retain H6N/H6Z plus H8B
**440.893 tok/s / 2,155 dispatches**
([H8G rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-global-qrow6-transfer-rejected.json)).

Reject **WPF-H8H exact prefill attention+softplus dual publication** at the
first-object physical gate. The four separately named H6N/H6Z/H6A/H6W siblings
all execute with F32-bit-exact context, BF16-byte-exact gated output, unchanged
complete `KVLiveSpans`, clean lifecycle, and zero new compiler processes.
Control bodies remain source-identical. Runtime resources are H6N **VGPR40≤48**,
H6Z **88>56**, H6A **80>72**, and H6W **80>64**, all at their frozen local sizes
with zero LDS/scratch.

Three immutable ceilings fail before timing. Honor the declared no-resource-
rewrite/no-partial-route rule: run no leaf timing, owner, state/topology, fixed,
length, or source gate; remove all candidate bodies/wrappers/keys/backend
exclusions and the RED test. Retain the registered H6N/H6Z/H6A/H6W plus
standalone-gate chain and unchanged H8B **440.893 tok/s / 2,155 dispatches**
([H8H target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8g-prefill-attention-softplus-dual-publication-target.json) ·
[H8H rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-prefill-attention-softplus-dual-publication-physical-rejected.json)).

Select target-only **WPF-H8I exact stream-ordered Q5 partition accumulation**.
Q5 remains the largest matched gap at **172.115 ms**; its complete six-role
class is **188 H7G/H7H calls / 203.861 ms**. Replace each local128 four-wave
consumer with four stream-ordered local32 partition grids. Preserve every
logical lane's FMA sequence, wave32 reduction, and **0→1→2→3** sum, so compute
waves remain **20,085,760**, while removing **5,021,440** workgroup barriers.
The operation explicitly adds **564 dispatches (2,155→2,719)** and **7.546875
GiB** of M512 global traffic; borrow at most **24 MiB** of aligned inactive
request scratch. No candidate or performance result exists. Freeze RED first
and bind all six roles plus the weighted aggregate to exactness, first-object
resources, and both-clock wins without subset/rewrite/recompile/rerun salvage
([H8I target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8h-streamed-q5-partitions-target.json)).

Reject H8I after exactness and physical admission. All **24** local32 stages
are partition/final-bit exact, LDS/private/scratch/spill-free, and within their
VGPR ceilings, but every actual-weight role loses both clocks. The weighted
188-call aggregate regresses **222.555→289.013 ms event (+29.861%)** and
**225.438→286.922 ms wall (+27.273%)**. Apply the no-subset/no-rerun rule,
remove all candidate/RED surfaces, skip owner/state/length/source work, and
retain unchanged H8B **440.893 tok/s / 2,155 dispatches**
([H8I rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-stream-ordered-q5-partitions-rejected.json)).

Select target-only **WPF-H8J exact IQ3 four-workgroup occupancy** after closing
Q5's remaining straightforward partition route. H6T owns **264.377 ms / 45
calls**, or **96.783%** of current IQ-down. Its local128/runtime-VGPR104 object
fits **3 complete workgroups / 12 resident waves per SIMD**; one separately
named, otherwise identical `launch_bounds(128,4)` sibling must reach **≤96
VGPR**, zero scratch, and **4 workgroups / 16 waves (+33.333%)**. This is target
arithmetic, not a performance claim. Freeze RED first and preserve every H6T
operation/byte/ABI/fallback. Bind all 45 layers and the aggregate to one exact,
physical, named-trace, both-clock screen; forbid launch-bound sweeps, subsets,
rewrites, recompiles, and favorable reruns
([H8J target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8i-iq3-four-workgroup-occupancy-target.json)).

Reject **WPF-H8J** at the first-object physical gate. The one
`launch_bounds(128,4)` object keeps H6T's relocation-normalized **7,920-byte /
1,384-slot** instruction stream and **VGPR101 / SGPR78 / LDS384 / spill0**
metadata. Because **101>96**, it still admits only **3 complete four-wave
workgroups/SIMD** instead of four. Honor the no-resource-rewrite/no-rerun rule:
skip correctness, runtime tracing, all-layer timing, and runtime/source work;
remove candidate plus RED and retain H6T/H8B production **440.893 tok/s / 2,155
dispatches**
([H8J rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-iq3-four-workgroup-occupancy-physical-rejected.json)).

Select target-only **WPF-H8K exact IQ3 uniform-rowbatch4 triple-output
ownership** as a body/liveness operation distinct from H8J's launch-bound-only
miss and H7L's branched tail path. Frozen all-45 natural routing keeps
**230,400 useful rows** while changing rowbatch8 **33,547 epochs / 268,376
slots / 37,976 padded** to rowbatch4 **61,546 / 246,184 / 15,784**. That removes
**8.269%** of compute slots and models explicit activation-plus-accumulator
liveness **40→20 dwords**, at the declared cost of **83.462%** more row epochs
and **57,341,952** more barrier epochs. This is target arithmetic, not speed.
Freeze RED, require metadata/runtime VGPR≤96 and complete H6T/CPU/all-45 bytes,
then require every layer and aggregate to win one immutable both-clock screen.
Forbid alternate rowbatch compilation, subsets, rewrites, recompiles, and
favorable reruns
([H8K target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8j-iq3-rowbatch4-triple-output-target.json)).

Reject **WPF-H8K** at the named-trace resource gate. Exact edge/CPU/state gates
and first-object physical bounds pass: metadata **VGPR70 / SGPR58 / LDS192 /
spill0**, **4,916 B / 882 slots**, and the frozen halved instruction counts.
The natural-M512 trace has exact **45 H8K + 2 IQ4 / 2,155 dispatches**, one
queue/stream, runtime-VGPR72, and scratch0, but runtime LDS is **512 > frozen
256 B**. Honor the no-resource-gate-rewrite rule: skip all-layer timing and
runtime/source work, remove candidate plus RED, and retain H6T/H8B production
**440.893 tok/s**
([H8K rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-iq3-rowbatch4-triple-output-runtime-lds-rejected.json)).

Select target-only **WPF-H8L exact IQ3 lossless 12-bit codebook packing** as a
representation-width operation distinct from H6X's rejected LDS table and
H8J/H8K occupancy/ownership misses. All **256** four-coordinate uint32 entries
reconstruct exactly and uniquely from uint16 3-bit codes using
`4 + 8*code + 2*(code == 7)`. Storage falls **1,024→512 bytes**; across frozen
all-45 natural routing, modeled logical codebook bytes fall
**105,529,737,216→52,764,868,608 (−50%)** at unchanged wave-load count. This is
not cache-traffic or speed evidence. Freeze RED and one fixed representation;
preserve H6T's 216 FMAs, reductions, rowbatch8/triple-output ownership, raw ABI,
LDS order, and all bytes. Require exact table/H6T/CPU/all-45 results, bounded
physical resources, then every layer and aggregate to win one immutable both-
clock screen. Forbid alternate widths/formulas, subsets, rewrites, recompiles,
and favorable reruns
([H8L target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8k-iq3-codebook12-target.json)).

Reject **WPF-H8L** after its one immutable all-45 timing screen. All entry/
edge/CPU bytes, first-object bounds, and exact-state **45 H8L + 2 IQ4** trace
pass; candidate metadata/runtime is VGPR **111/112**, LDS **384/512**, and
scratch0. Nevertheless **0/45** layers win both clocks and aggregate H6T→H8L
event/wall regresses **260.044/260.757→290.496/290.437 ms
(+11.710%/+11.382%)**. Forbid table-width/formula/load-source/layer/rerun
salvage, remove candidate plus RED, add no runtime/source owner, and retain
H6T/H8B production **440.893 tok/s**
([H8L rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-iq3-codebook12-all45-timing-rejected.json)).

Select target-only **WPF-H8M exact IQ3 sign-folded BF16 codebook** as the
materially opposite representation operation after H8L: spend read-only table
bytes to remove dynamic sign ALU rather than compressing magnitudes and adding
extracts. All **4,096** `(sign_nibble, grid_index)` records are exact and unique
uint64 packs of four signed BF16 values. The model expands storage
**1,024→32,768 bytes** and all-45 logical table bytes
**105,529,737,216→211,059,474,432 (+100%)** at unchanged wave-load count, while
targeting H6T's **24 compare + 24 select** sign sites. This is not cache-traffic
or speed evidence. Freeze one six-b64 body under RED; preserve all H6T
arithmetic/ownership/output bytes, require code≤8,500 B, slots≤1,500,
metadata/runtime VGPR≤101/104, exact edge/CPU/all-45 results, then every layer
and aggregate on one immutable both-clock screen. Forbid alternate table dtypes,
widths, indexing/layout, cache placement, subsets, rewrites, recompiles, and
favorable reruns
([H8M target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8l-iq3-signed-bf16-codebook-target.json)).

Reject **WPF-H8M** at the frozen first-object metadata gate. Exact codebook and
edge/CPU GREEN pass **4/4**. The one object realizes six b64 loads, removes all
24 compare and 24 select sign sites, preserves 216 FMAs and every reduction/LDS/
barrier/store count, and reduces code/slots **7,920/1,384→7,768/1,321**. It
nevertheless increases metadata VGPR **101→102**, failing the explicit
non-growing ≤101 requirement. Honor the no-rewrite/recompile/resource-gate-
relaxation rule: skip named trace, all-45 timing, runtime/source work, remove
candidate plus RED, and retain H6T/H8B production **440.893 tok/s**
([H8M rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-iq3-signed-bf16-codebook-physical-rejected.json)).

Select target-only **WPF-H8N exact Q5 paired-rowgroup twin-team F32-weight
staging** after H8M. Q5 remains the largest matched gap at **172.115 ms**; its
six H7G/H7H consumers own **188 calls / 203.861 ms**. One fixed local256 body
must contain two independent logical local128 teams and share each exact F32
K128×COL slab from ping-pong LDS across adjacent activation row groups while
preserving every current scalar FMA, wave reduction, serial wave sum, and store.
The six fixed geometries model logical F32-plane bytes
**807,571,292,160→407,862,509,568 (−49.495%)** and workgroups
**5,021,440→2,689,792 (−46.434%)** with **1,433,445,335,040 FMAs** unchanged.
Record the opposing cost: barrier epochs rise **5,021,440→90,764,032
(18.075×)** and fixed LDS is **9,216–18,944 bytes**. This is not cache-traffic
or speed evidence.

Freeze RED before source work. Require exact odd-pair tails and all edge/CPU/
M512 outputs, local256 with per-role VGPR/LDS ceilings and no spill/scratch, a
six-name cache-only trace, then one immutable all-role 5/15/5 screen where every
role and the weighted aggregate win both clocks. Forbid role/dtype/shape/layer/
prompt/token/length/geometry/buffer/K-tile/resource-rewrite/recompile/rerun
salvage
([H8N target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8m-q5-twin-team-weight-staging-target.json)).

Reject **WPF-H8N** after its one immutable six-role timing screen. Complete
control/CPU bytes pass **5/5**, all five object instances and six named traces
meet local256 VGPR/LDS/spill/scratch bounds, and no compiler appears after the
sole build. Yet **0/6** actual-weight roles win both clocks: weighted 188-call
H7G/H7H→H8N event regresses **212.742→370.566 ms (+74.186%)** and synchronized
wall regresses **224.095→365.407 ms (+63.059%)**. The 49.495% modeled logical
weight-byte saving cannot repay LDS staging and 18.075× barrier epochs. Forbid
role/geometry/buffer/K-tile/rewrite/recompile/rerun salvage, remove candidate
plus RED, add no runtime/source owner, and retain H7G/H7H plus H8B production
**440.893 tok/s**
([H8N rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q5-twin-team-weight-staging-rejected.json)).

Reject **WPF-H8O exact gfx1100 after-router least-priority MoE branch
concurrency** at the binding fixed gate. The exact frozen two-queue schedule
passes all-48-boundary complete-state, logits/hidden/KV+`KVLiveSpans`, token,
finiteness, session/owner lifecycle, resident-sidecar, priority, and zero-
compiler checks. It fails both mandatory timing checks: queue-matched serial
control→candidate is **438.604→436.514 tok/s (-0.4765%)** with **0/7 wins**
versus required positive median and ≥5/7.

Apply the no eager/normal-priority/queue-count/layer/length/event-boundary/
schedule/recompile/favorable-rerun rule. Skip the named two-queue trace, clean
512/1K/4K transfer, and source-default RED; remove the temporary descriptor and
RED, leave all three gfx1100 concurrency capabilities false, preserve gfx1151's
independent policy, and retain H8B production **440.893 tok/s / 2,155
dispatches**, **1.566801×** behind matched llama.cpp HIP
([H8O rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-after-router-low-priority-moe-concurrency-rejected.json) ·
[H8O target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8n-moe-shared-after-router-low-priority-target.json)).

Reject no-code **WPF-H8P lossless Q5 signed-int16 power-of-two plane** before a
GPU target. Although 256 int16 values plus eight exponents would reduce each
transient Q5 block **1,024→520 bytes (−49.219%)**, exact representation is
impossible on production weights. In a 16,777,216-value audit, only **53.868%**
of individual values fit even with one exponent each. The decisive F32 bit
pattern is `0x3d72fd00 = 62205 × 2^-20`: 62,205 is odd and exceeds +32,767, so
no signed-int16/power-of-two pair can encode it. Add no source, RED, GPU,
compiler, runtime, or timing path; retain H8B **440.893 tok/s** and rerank
outside fixed16 Q5 planes
([H8P analytical rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q5-fixed16-power2-plane-analytical-rejected.json)).

Reject **WPF-H8Q exact Q6 int16-product plus tiled-F32-scale transient plane**
at the first-object physical gate. Its transient producer/three-consumer family
passes **15/15** exact correctness tests, including **−4,064/+4,096** planes,
all roles at rows17/33/M512, sampled CPU values, lifecycle, fallbacks, and
backend isolation. The producer is **VGPR14/LDS0/spill0**, and the consumers
contain the intended uniform scale load with no private segment/spills/scratch.
Consumer metadata VGPR is nevertheless **169/136/169**, already above all
frozen runtime ceilings **160/128/160**.

Honor the no-resource-rewrite/recompile/rerun rule: skip named trace and timing,
remove candidate plus RED, add no owner, and retain H8B production
**440.893 tok/s / 1,146.420-ms / 2,155 dispatches**, **1.566801×** behind
matched llama.cpp HIP. The prior **450.782 tok/s** traffic-only ceiling remains
a rejected model, not a result
([H8Q rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q6-int16-product-plane-physical-rejected.json) ·
[H8Q target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8p-q6-int16-product-plane-target.json)).

**Table W7900 Laguna parity implementation here; select no H8R target.** Any
resumption starts from the campaign/current/matched-llama.cpp table and
high-leverage admission rules in
[`LAGUNA-PARITY-STATUS.md`](LAGUNA-PARITY-STATUS.md): prioritize Q5, IQ-down,
and attention algorithm/dataflow transfers, require a plausible ≥50-ms or ≥5%
end-to-end target before code, and do not restart an adjacent micro-variant
ladder.

The old wider-qrow, cross-head/key-split, attention-rowbatch16,
attention output-tile/source-MMQ, combined QK+PV changed-association attention, H5O representation, H5P geometry, H5S persistent
ownership, H5T one-wave IQ3 ownership, H6B segment-plane representation, and
P6/repair routes remain closed.
Cross-boundary launch fusion other than the explicitly bounded H8H target remains deferred.
Keep 16K+ closed until direct M512 reaches **696.342 tok/s**, then measure
matched llama.cpp HIP at M4K before setting a long-context parity gate; 800/700
remains stretch. The full ledger, source-port boundaries, and admission gates
are owned by `LAGUNA-prefill.md`.

LAP-0 is complete at the clean gfx1151 control packet. LAP-1 is complete: the
source-arithmetic packed-dot body, live-row schedule, and direct resident-T16
consumer pass every leaf gate. Producer-pack-inclusive direct T16 reaches
**2.502x/3.959x/5.502x** retained at M128/M256/M512, stays within
**4.66%/4.05%/3.02%** of the X8 ceiling, remains positive at all natural
shapes, and adds no resident sidecar or layout transpose. The first LAP-2
three-plane/guarded/exact-repair primitive is implemented and traced.

The campaign order is no longer numeric. A 2026-07-25 source and artifact audit
corrected the actual Vulkan comparator from a 32x32 small tile to a 64x64
medium tile on RADV gfx1151 and showed that the **344.56 tok/s** Vulkan row is
a compatibility floor, not a hardware ceiling. Next establish a same-host
locked-clock bandwidth/active-byte ledger and ablate the shipping
**0.0459275** maximum KL debt. The torch-free hipBLASLt source-F16 candidate is
now integrated with exact power-of-two row scaling: compounded with selected
MMQ it moves a same-session real pp512 diagnostic **127.831 -> 154.321 tok/s**
with the same next token and no added scratch. It remains explicit pending the
complete quality/clean gate. The low-risk dense/shared Q4 candidate is also
integrated without a sidecar: its resident-pack8 64x16 WMMA leaf is 5.275x the
old M512/K3072/N1024 kernel and the compounded real pp512 stack now reaches
**163.881 tok/s** with next token 2930. Selected Q4 gate/up and Q4/Q6 down
first used a range-safe one-plane Q8_1 pack plus direct resident-T16 64x32
integer-dot MMQ. That stack reached **355.273/355.721 tok/s**, but the complete
category gate rejected it at maximum KL **0.0767056**. The repaired gate/up
pack uses one FP32 scale per 16 values in the same 160-byte activation block,
while down remains D4; its Q4 consumer widens to 128 columns x 32 rows and
reconstructs the signed half-block sums required by the min term. The complete
320-step diagnostic now passes at maximum KL **0.040724836**, **317/320**
top-1. Compounded with LAP-5/LAP-6 and the raw-Q6 64x16 dense/shared consumer,
exact reconstructed-sum pp512 repeats at **353.951/356.082/356.473 tok/s**
with next token 2930. The clean complete category gate admits the compounded
route at maximum KL **0.040724836**, **317/320** top-1, **2.615x** aggregate
natural-prompt prefill, flat decode, and exact tracked-lifecycle recovery.
gfx1151 now selects D8 gate/up, D4 down, row-scaled hipBLASLt, and Q4/Q6 WMMA
dense/shared as package defaults. Clean selector-unset production pp512 passes
at **353.421/355.584/354.820 tok/s** (median **354.820**, **4.655x** the old
76.226 row), with every sample above 350, token 2930, deterministic repeated
state, and exact lifecycle recovery. A cached-only production trace
independently reaches **354.763 tok/s** and names the D8 128-column gate/up
MMQ, D4 Q4/Q6 down, Q4/Q6 WMMA dense/shared, scaled hipBLASLt, and online
attention families. The production milestone is complete. Streaming families
still target at least 70% of the measured achievable-read ceiling unless
profiling proves another limiter; the detailed post-350 sequence and
T16-lite/X16 replacement-layout screen remain owned by
`LAGUNA-prefill.md`.

Subsequent exact production work reached **654.249/579.699/468.608 tok/s** at
512/1K/4K. The pre-campaign one-session
512/1K/4K/32K/64K/128K closure measured
**622.009/579.152/470.270/214.698/131.997/72.323 tok/s**. The bounded
LC-0 through LC-6 campaign then closed with a tracked-clean selector-unset
six-shape sweep at
**614.031/666.901/609.879/365.481/247.408/149.308 tok/s**, improving
1K/4K/32K/64K/128K by
**15.151%/29.687%/70.230%/87.435%/106.446%**. The 512 singleton is
**-1.283%** in the 128K-capacity session, while a separate capacity
counterbalance is within **-0.425%/+0.654%/-0.004%** at 512/1K/4K. Every
final position, next token, finite-state check, and allocation teardown
passes. The pp512-to-700 expert lane is paused, not closed; a fresh retained
64K trace must now choose between another fused global-attention owner and
resuming physical-byte expert work. A same-GGUF one-pass llama.cpp Vulkan
control measured
**341.999/333.502/280.349/126.624/65.584 tok/s** at
512/4K/16K/64K/128K. Final hipEngine is
**79.542%/82.871%/95.388%/127.659%** faster at the overlapping
512/4K/64K/128K shapes, so Vulkan remains a floor rather than the target.

The long-context roofline is Laguna-specific, not borrowed from hybrid
Qwen3.x/GDN. The exact production GGUF metadata confirms that every one of
Laguna's 48 decoder blocks uses softmax attention: 12 global layers with 48
query heads and 36 sliding-window layers with 72 query heads and a 512-token
window. There are no GDN/linear-attention blocks. Both attention families use
eight KV heads of dimension 128. The exact mixed-attention QK+PV ledger is
**5.084/50.544/677.685/2,622.181 TFLOP** at 4K/16K/64K/128K. SWA contributes
**105.46%/27.68%/7.00%/3.51%** as much arithmetic as global attention at those
shapes: it matters to the short guard, but the global layers own the quadratic
tail. The coherent LC-0 control measures
**466.482/307.953/132.831/72.139 tok/s** at those shapes with exact positions,
deterministic tokens, and complete allocation recovery.

The completed ordered LC-0 through LC-6 attack, rough one-pass 4K/16K/64K
development screens, mandatory positive 128K gate between major stages, and
full promotion gates are defined in `LAGUNA-prefill.md`. The retained route
uses exact 4K-block online global attention, a fixed tensorized rolling-SWA
union, and six-query-head GQA reuse across each M2,048 query chunk without a
quadratic score matrix. Whole-model M4,096/M8,192, lazy KV, Q8 KV, and
unchanged AOTriton/GroupedGemm routes are closed by measured evidence.

LC-0 trace attribution now measures global/SWA/complete-minus-attention wall
at **22.670/10.462/19.860 seconds** for 16K and
**370.549/43.499/79.483 seconds** for 64K. Their 16K-to-64K ratios are
**16.345x/4.158x/4.002x**, directly proving the expected
quadratic/linear/linear split. At 64K, global attention alone owns **75.08%**
of complete wall and all attention owns **83.90%**; kernel span is within
**19.7 ms** of complete wall. The superseded LC-0 scalar global qrow6
multiplied K/V load requests by about **131.76x** across 22 row groups and six
GQA heads, while the superseded SWA qrow4 multiplied them by **288x** across
32 row groups and nine heads.
These request counts include cache hits and are not DRAM counters. The next
decision starts with a fresh cached 64K trace. If global attention still
dominates, the strongest new premise is an in-tree fused head-dim-128 GQA
FlashAttention owner consuming `KVLiveSpans` directly, or an attention-only
two-M2,048 scheduling window that raises query reuse without the rejected
whole-model M4,096 scratch.

Laguna long-context decode is now a separate active track in
`LAGUNA-decode.md`. LC-D1 removed the false `allocated_capacity == 4096`
qualification from the exact fused global owner while retaining resource-safe
live-work bounds. At real capacity 131,200, clean d1K/d4K improve
**20.637969 -> 23.068316 tok/s (+11.776%)** and
**15.477837 -> 21.666976 tok/s (+39.987%)**, landing within
**1.275%/5.947%** of same-GGUF Vulkan. The mandatory
16K/64K/128K gate is neutral-to-positive with exact recurrent state and full
allocation recovery. LC-D2 was assigned the exact generic score-plane and
reducer route above 6,000 live slots. The matched 16K HIP/Vulkan capture
completes that attribution: hipEngine spends **86.240 ms/token** in global
attention versus a **3.693-ms** Vulkan global-FA-plus-output scheduled group,
and the **82.547-ms** group gap explains **99.991%** of the complete profiled
device gap. The hipEngine reducer/PV alone costs **76.527 ms/token**. LC-D3's
first exact GQA6 milestone is the clean production default: a shared-K
score producer, exact exp32 normalizer, and D32/V64 shared-V PV owner cut
live16,448 global attention to **16.209 ms/token** and improve directional
4K/16K/64K/128K production decode by
**39.96%/117.15%/262.23%/326.73%** with exact recurrent state. This clears the
first **<=20-ms** gate; tracked-clean 16K reproduces **16.756 tok/s
(+116.72%)** within **0.20%** of the directional row. The next exact checkpoint
defers probability normalization into the D32/V64 PV loader, removes one full
score-plane writeback/read pass, and improves complete 4K/16K/64K/128K another
**0.036%/1.078%/1.490%/2.190%** to
**21.670/16.971/9.214/5.725 tok/s**, with identical F32/BF16 leaves, generated
hashes, positions, residency, and lifecycle. The 4,096-token context-parallel
partial/merge path is rejected across all twelve global layers at maximum KL
**0.687034**, but a measured final-four-layer scope passes two independent
16K/127-step gates at maximum KL **0.042569/0.007344** and 100% top-1. gfx1151
therefore applies bounded context-parallel PV only to layers 32/36/40/44,
improving exact 16K/64K/128K another **2.321%/6.585%/8.600%** to
**17.364/9.821/6.218 tok/s**; 4K is route-inactive and flat. All generated
hashes/positions and lifecycle checks pass. LC-D3 remains active because the
exact first eight global layers still carry the full score/physical plane.
The broad compensated layer-28-plus screen is not promotable despite passing
teacher forcing because it changes the mandatory 128K recurrent final
token/hash. Isolating compensation to newly admitted layer 28 while leaving
layers 32/36/40/44 on their retained scalar-F32 partial/merge fixes that
failure: d16K/127 passes at maximum KL **0.007761** and d16K/d64K/d128K improve
another **0.393%/0.479%/1.667%** with every established trajectory and
lifecycle check intact. The fifth checkpoint widens only those five admitted
PV owners from D32 to D64. It preserves every F32/BF16 output bit, halves the
PV grid, cuts the ordinary active leaf **10.665-17.258%** and the compensated
leaf **12.989-19.729%**, and improves complete d16K/d64K/d128K another
**1.705%/2.558%/2.348%** to **17.731/10.120/6.470 tok/s** with the mandatory
128K trajectory and lifecycle intact. LC-D3 now leaves exact global layers
0..24. An isolated compensated layer-24 screen reaches **0.235600 max KL**
on teacher70 and is removed, so precision-boundary widening is no longer the
next lever. The next owner must reduce that seven-layer score/PV path or use
an ordered tiled replay while targeting
**<=5 ms/token** at 16K. Reassociated online-softmax arithmetic is not a
numerics waiver.

| Phase | Scope | New LoC | Adapted LoC | Total |
|-------|-------|---------|-------------|-------|
| **0. Foundation** | Core host (scheduler, block manager, engine loop, model registry, fusion planner) | ~700 | ~0 | **~700** |
| | Torch-free core primitives (`hipengine.core.*`: Tensor, device, memory, graph, blas, build, stream) | ~1,900 | ~0 | **~1,900** |
| | Torch-free loading (safetensors + HF config + chat template + tokenizer glue) | ~900 | ~0 | **~900** |
| | `KVLiveSpans` + `KVStorageView` + `KVCacheBackend` claim/pool contract + per-head-variable-span attention kernel ABI | ~400 | ~0 | **~400** |
| | Port + split nano-vllm-amd HIP kernels into `hipengine/kernels/hip_gfx1100/<family>/` | ~300 (split scaffolding) | **~17,590** (HIP) + **~1,040** (retyped bindings) | **~18,930** |
| | Retype kernel launch wrappers from `torch::Tensor` to raw-pointer signatures | ~200 | ~1,040 | **~1,240** |
| | Port Python dispatch wrappers from `nano-vllm-amd/nanovllm/native/qwen35/` (retyped to `hipengine.Tensor`) | ~500 | **~10,900** | **~11,400** |
| | Own FA2 prefill HIP kernel (replaces `F.scaled_dot_product_attention`) | **~1,500** (HIP) | ~0 | **~1,500** |
| | CPU-reference backend (numpy implementations of all `layer` keys for correctness oracle) | ~800 | ~0 | **~800** |
| | Smoke: Qwen3-0.6B + Qwen3.5 0.8B dense generate text end-to-end | ~20 | ~0 | **~20** |
| **1. Server + Benchmark** | FastAPI server (`hipengine serve`, installed by default) | ~150 | ~200 | **~350** |
| | Benchmark harness (prefill/decode/memory) | ~150 | ~0 | **~150** |
| | Correctness fixtures (KL, top-1, PPL) driven by `cpu_reference` oracle | ~200 | ~0 | **~200** |
| | Qwen3.5 27B dense target benchmark vs `llama.cpp` ROCm baseline | ~50 | ~0 | **~50** |
| **2. Quantization + MoE** | W8A16 native dispatch via `W8A16Quant` plugin | ~150 | ~200 | **~350** |
| | W8A8 dynamic quant via `W8A8Quant` plugin | ~100 | ~100 | **~200** |
| | PARO W4 plugin (collapse `paroquant.py` into `W4ParoQuant`) | ~200 | ~500 | **~700** |
| | GPTQ / GPTAQ / AWQ plugins (all reuse `gemm_dequant` kernel family; new weight-preprocess glue) | ~600 | ~0 | **~600** |
| | Qwen3.5 MoE hybrid model plugin (`full_attention` + `linear_attention` + `gdn` + `moe_top2`) | ~400 | ~100 | **~500** |
| | Qwen3.6 35B-A3B perf target | ~50 | ~0 | **~50** |
| **3. Advanced KV + Prefix + TP + more models** | RadixCache implementation | ~200 | ~0 | **~200** |
| | Sliding-window + attention-sink topology (StreamingLLM) | ~200 | ~0 | **~200** |
| | Paged FP8 codec/backend composition (software FP8 where qualified) | ~250 | ~0 | **~250** |
| | Basic multi-GPU TP (rccl all-reduce via ctypes) | ~150 | ~0 | **~150** |
| | Gemma 4 model plugin + sliding_attention kernels | ~500 | ~0 | **~500** |
| | Llama 3 model plugin | ~200 | ~0 | **~200** |
| | sansho custom arch plugin | ~300 | ~0 | **~300** |
| **4. SpecDec + DMS** | `DraftModel` interface | ~50 | ~0 | **~50** |
| | Medusa / Lookahead / MTP / DFlash paths | ~200 each | ~0 | **~800** |
| | Scheduler speculation awareness | ~100 | ~0 | **~100** |
| | DMS topology/backend component + model-plugin DMS config loader (eviction head weights) | ~500 | ~0 | **~500** |
| | DMS HIP kernels: `dms_rope_store_compact_decode`, `compact_decode_grouped_splitk`, `streaming_pack_scatter` | ~1,500 (HIP) | ~0 | **~1,500** |
| **5. Advanced Features** | C++ engine-step extension (lever #2) if profiling demands | ~1,500 | ~0 | **~1,500** |
| | CUDA backend (`kernels/cuda_sm86/`) — reuse kernel tree shape | ~500 scaffolding | **~18,630** (retyped + recompiled per-kernel porting) | **~19,130** |
| | EXL3 / QTIP codebook kernel family (new `codebook_lut` tree, ~14 kernels) | ~300 | ~8,000 (port from ExLlamaV3) | **~8,300** |
| | FastKron `kronecker` kernel family (compute pattern rewrite) | ~1,500 | ~0 | **~1,500** |
| | FP8 weight quant (only on `hip_gfx1200`+ / `cuda_sm90`+; skipped on gfx1100) | ~400 | ~0 | **~400** |
| | H2O / SnapKV topology plugins | ~600 | ~0 | **~600** |
| | AQUA-KV cross-layer predictor (requires per-layer scalar-quant codec) | ~800 | ~0 | **~800** |
| | Tiered offloading (host pinning, disk spill) | ~400 | ~0 | **~400** |
| | Session save/restore (ds4-style) | ~150 | ~0 | **~150** |
| | Expert CPU offload (ktransformers-style) | ~300 | ~0 | **~300** |
| | GGUF Q4_K_M loader | ~500 | ~0 | **~500** |
| | Pipeline Parallelism | ~200 | ~0 | **~200** |
| | Expert Parallelism | ~250 | ~0 | **~250** |

**Cumulative totals:**
- Phase 0 (MVP): ~36,790 lines (~700 host + ~1,900 core + ~900 loading + ~400 KV backend ABI + ~18,930 HIP+bindings + ~1,240 retype + ~11,400 dispatch + ~1,500 FA2 + ~800 cpu_reference + ~20 smoke)
- Phase 1 (server+bench): +750 lines → **~37,540**
- Phase 2 (quant+MoE): +2,400 lines → **~39,940** (adds GPTQ/GPTAQ/AWQ line)
- Phase 3 (KV+prefix+TP+models): +1,950 lines → **~41,890** (adds StreamingLLM, paged_fp8)
- Phase 4 (specdec+DMS): +2,950 lines → **~44,840** (adds DMS topology/backend + kernels)
- Phase 5 (advanced, incl. CUDA backend + codebook + FastKron + H2O/AQUA): +34,130 lines → **~78,970**

> **Note:** LoC is an imperfect proxy for effort. ~17,590 HIP lines + ~1,040 retyped bindings are **copied and repartitioned kernels** (known working; split + retype are mechanical and gated by `rocprofv3` + KL). ~10,900 Python dispatch lines are **adapted** — real porting work because they encode kernel-selection policy and weight layout. The torch-free core (~1,900) and loading (~900) and CPU reference (~800) are **new engineering** but ~80% straightforward and testable against the existing torch-based workspace as oracle. The FA2 prefill kernel (~1,500 HIP) and the DMS compact-decode kernels (~1,500 HIP) are the two hardest new HIP pieces. Phase-5 CUDA backend is the largest single deferred item because each of the 120 kernels needs a CUDA variant (though most are straightforward: wavefront=32, `cub::WarpReduce` instead of AMD shuffle, `wmma` instead of ROCm WMMA). The Phase-4 DMS estimate remains provisional; its architectural advantage is that Phase-0/C2 `KVLiveSpans`, backend pool plans, and atomic resource claims are designed before the topology, so it does not require a second continuous-batching implementation.

## Comparison to Existing Engines

| Feature | vLLM | ExLlamaV3 | llama.cpp | atlas | FastDMS | hipEngine |
|---------|------|-----------|-----------|-------|---------|-----------|
| AMD ROCm support | Partial (no FA) | Missing | Good (HIP/Vulkan) | No | No (CUDA) | **First-class** |
| Custom gfx1100 kernels | No | No | Some | No | No | **Extensive (120 kernels)** |
| W4 quant families | No | EXL3 (codebook) | Q4_K_M (dequant) | NVFP4 | — | **PARO pack8 day-1; GPTQ/AWQ/GPTAQ share same family Phase 2; EXL3/QTIP+YAQA/FastKron Phase 5** |
| FP8 weight | Yes (sm90+) | No | No | Yes (sm90+) | Yes | **Phase 5, backend-gated (not on gfx1100)** |
| FP8 KV | Yes | No | No | — | Yes | **Phase 3 (software, all backends)** |
| MoE native kernels | No | No | No | Some | Dense-focused | **W8A16 fused** |
| Prefix caching | Prefix | No | No | Yes | Disabled in DMS mode | **RadixCache** |
| OpenAI API | Yes | Via TabbyAPI | No | Yes | Yes | **Built-in (optional)** |
| Library API | No | No | Bindings | No | Yes | **Primary** |
| Benchmark harness | Internal | No | llama-bench | — | Yes | **Built-in, comparable** |
| Speculative decode | Medusa | No | No | Yes | No | **Phase 4 (Medusa, Lookahead, MTP, EAGLE3, DFlash)** |
| KV compression: DMS | Major surgery (per FastDMS README) | No | No | No | Yes (reference impl) | **Phase 4 via DMS topology/backend composition; `KVLiveSpans` and generic claims designed first** |
| KV compression: H2O / SnapKV / sliding | Sliding (via model) | No | No | — | — | **Phase 3 sliding, Phase 5 H2O/SnapKV** |
| KV backend composition | bf16, fp8, TurboQuant-4bit | bf16 | Various | — | bf16, fp8, int4-shadow | **Retention topology + hot codec/layout + cold tier resolve to one backend; no concurrency fork** |
| Torch-free runtime | No | No | Yes | Yes | No | **Yes** (`~100 MiB` vs `~2 GiB`) |
| Multi-backend kernel tree | CUDA-only | CUDA-only | All (per-backend dirs) | CUDA-only | CUDA-only | **HIP + CUDA + CPU reference** |
| Single-binary shipping | No | No | Yes | Yes | No | Via C++ engine-core extract (Phase 3+, optional) |
| Python API | Yes | Yes | Bindings | No | Yes | **Yes, no torch dep** |

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Host from scratch | ~700 lines new | Fits kernel dispatch model; nano-vllm/mini-sglang carry wrong assumptions |
| Runtime language | Python, torch-free | Keeps HF ecosystem + pip/uv install + notebook workflow; drops 1.7 GiB torch dep |
| Tensor type | `hipengine.Tensor` (thin wrapper over HIP/CUDA ptr + dlpack) | Controls backend dispatch at the type level; torch tensors flow in/out via `[torch]` extra |
| Kernels copied + split | `nano-vllm-amd/csrc/amd/` + `paroquant_kernels.py` embedded HIP | ~17,590 lines HIP + ~1,040 lines retyped bindings, 120 `__global__` kernels across 14 files under `kernels/hip_gfx1100/` |
| Kernel launch signatures | Raw pointer + shape/stride/dtype (not `torch::Tensor`) | Kernel bodies already torch-free; wrappers retype in one scripted pass |
| Python dispatch adapted | `nano-vllm-amd/nanovllm/native/qwen35/` | ~10,900 lines of weight-layout / kernel-selection wrappers, retyped to `hipengine.Tensor` |
| Dispatch axes | Backend × Model × Quant × Layer | Orthogonal plugin registries, no hardcoded branches |
| Prefill attention | **Own HIP FA2 prefill kernel** (~1,500 lines) | Replaces `F.scaled_dot_product_attention`; needed anyway for long-context prefill on gfx1100 |
| Graph capture | `hipGraph` via ctypes on `libamdhip64.so` | Phase-0 dispatch lever; zero Python overhead at replay; ROCm-native (no torch) |
| Build | `hipcc` / `nvcc` via `subprocess.run` + `ctypes.CDLL` + hash cache | Drop `torch.utils.cpp_extension`; 3 profiles (decode `-mcumode`, prefill WGP, baseline) |
| Correctness oracle | `kernels/cpu_reference/` torch-free numpy | Every `layer` key has a CPU implementation; KL ≤ 0.05 / top-1 ≥ 90% gate |
| Quantization | Plugin registry with six orthogonal axes (weight storage / activation preprocess / compute dtype / scale granularity / calibration artifact / kernel family) | Lets GPTQ, GPTAQ, AWQ, PARO-W4, W8A16 all share the `gemm_dequant` kernel family. EXL3/QTIP adds `codebook_lut` family (Phase 5). FastKron adds `kronecker` family (research). FP8 weight is backend-gated (not gfx1100). |
| KV cache | `KVCacheBackend` with `KVLiveSpans` + registered `KVStorageView` as the kernel ABI and atomic `ResourceClaimSet`/delta accounting as the scheduler boundary | Makes retention topology, hot codec/layout, and cold tiering replaceable without request-lifecycle forks. Avoids both vLLM-DMS surgery and the current separate-INT8-concurrency trap. RadixCache uses backend snapshot capabilities. |
| DMS support | Phase 4 topology/backend composition + compact HIP kernel families + loader | `KVLiveSpans`, global backend pool sets, and generic resource claims are designed before DMS so BF16/FP8/INT8 payload changes do not rewrite concurrency |
| Server | FastAPI installed by default, launched via `hipengine serve` | Most users want the OpenAI-compatible API; server deps remain outside the torch-free inference hot path |
| Wavefront | Wave32 default for gfx1100 HIP device code | `-mcumode` is orthogonal to wavefront size; wave64 is only an isolated experiment with explicit flags/probes/gates |
| Native binary path | Phase 3+ (conditional on profiling) | Extract C++ engine-step extension once dispatch layer is stable; keeps Shape A as Phase 0 |

## RDNA3 Wavefront and Scheduling Caveat

For `hip_gfx1100` / W7900, hipEngine treats HIP device code as **wave32 by default**.
RDNA3 wave64 is architecturally real and LLVM can emit it with `-mwavefrontsize64`, but
it is not a practical project default for the nano-vllm-amd kernel lineage.
`-mcumode` and wavefront size are orthogonal: the decode profile keeps `-mcumode` for
CU scheduling, while still assuming wave32 collectives unless an isolated experiment
explicitly opts into wave64.

Default optimization focus:

1. **Wave32 + enough ILP** — use multiple independent accumulators, unrolled loops, and
   avoid long dependent FMA/VALU chains where possible.
2. **Expose RDNA3 dual-issue / VOPD opportunities** — keep independent VALU ops near each
   other, avoid unnecessary barriers/shared-memory traffic, and watch VGPR/scratch/LDS so
   occupancy does not collapse.
3. **Use wave32-compatible collectives** — `__shfl_down` within 32 lanes, then LDS/shared
   memory exchange for cross-wave/block reductions. Never assume `64 threads == one wave`.
4. **Verify hot kernels with measurements** — use `rocprofv3` time share first, check
   VGPR/scratch/LDS, and inspect generated ISA only for kernels hot enough to justify it.

Wave64 remains available only for isolated experiments with their own build flags,
`warpSize`/shuffle probes, correctness gates, ISA checks, and end-to-end benchmarks. Treat
wave64 as architecturally possible on gfx1100, not as a retained default. This also applies
to gfx1151 / Strix Halo targets: the dual-issue rules remain RDNA3-family, but lower CU
count and cache/LDS differences make wave32 + explicit ILP/VOPD exposure the safer default
for AWQ GEMV and grouped-GEMM hot paths.

## Open Research Questions

These are deliberately deferred. Each has a `rocprofv3` or benchmarking prerequisite before committing to an answer.

| Question | Blocker / Prerequisite | Decision deadline |
|---|---|---|
| Should the engine step move to C++ (pybind11) for lever #2 from the dispatch strategy? | `rocprofv3` showing dispatch > 3% of decode wall time on Qwen3.6-35B-A3B after hipGraph capture | End of Phase 2 |
| Ship a standalone `hipengine-cli` binary via the same C++ core? | Lever #2 decision first; then evaluate cold-start + deploy story | Phase 4 |
| Do we maintain a Triton fallback for portability to backends we haven't HIP-ported yet? | Usage evidence from users on non-gfx1100 hardware | Phase 3 |
| `tilelang` for fused prefill attention or for the DMS compact-decode kernels? | Write our FA2 prefill kernel and one DMS kernel as HIP first; compare a tilelang prototype | Phase 4 |
| Share the CPU reference backend with test-time inference (true offline mode)? | Measure cpu_reference perf on Qwen3-0.6B; if within 10× of GPU decode, worth it for CI | Phase 1 |
| EXL3 / QTIP codebook kernel port priority | Evidence of user demand for EXL3 models on W7900; port cost is ~8k LoC from ExLlamaV3 CUDA (PTX-heavy) | Phase 5 |
| FastKron for any target layer? | Needs a model / layer where Kronecker decomposition beats W8A16 by enough to justify a new kernel family | Research |
| YAQA refinement on top of QTIP — does it change the codebook kernel shape? | Write QTIP base first; if YAQA only changes calibration, it's free | Phase 5+ |
| AQUA-KV + HIGGS 4-bit KV stack (the 25.6× sansho finding)? | HIGGS is ~50% BF16 speed in `kvcache-quantization-research/`; defer until kernel faster | Research |
| Keep the aspirational GGUF loader, or punt to a llama.cpp FFI shim? | Measure loader complexity for Q4_K_M + Q8_0 | Phase 5 |
| Structural/thinking tokens (ds4-style thinking modes) as first-class sampling options? | User demand; see `docs/STRUCTURED-COT.md` from nano-vllm research | Phase 4 |
| Session save/restore: filesystem layout + compression policy | Decide after RadixCache is stable | Phase 5 |
| NVFP4 / MXFP8 support (atlas-style NVIDIA-only formats) | Only on CUDA backend; not a blocker for gfx1100 | Phase 5+ |
| Multi-tenant server with fair-share scheduling | Only if someone runs hipEngine in production | Research |
| DMS scheduler interaction with RadixCache prefix overlays | FastDMS disables prefix caching entirely; can we do per-sequence eviction overlays on shared prefix blocks? | Phase 4+ |

## Evidence Policy

Every performance claim in hipEngine must include:
- **Model**: exact checkpoint name, immutable revision, and content fingerprint
- **Quantization**: FP16, W8A16, W4, etc.
- **Workload**: prompt/generation length, client concurrency, choices, queue
  grouping, actual backend widths, and verifier rows
- **Hardware**: physical host identity, selected GPU, configured/resolved backend, target arch, ROCm and compiler versions; same-arch results from different hosts are independent and never form an old→new comparison without a declared same-host A/B
- **Source**: hipEngine commit plus separate staged, unstaged, and untracked state
- **Command**: exact benchmark invocation
- **Result**: tok/s prefill, tok/s decode, peak GiB
- **Correctness**: KL divergence ≤ 0.05, top-1 agreement ≥ 90%

This policy is inherited from the `LESSONS-LEARNED.md` discipline: fast rows are invalid until output sanity proves they are real.

Server, retained PARO, GGUF, and microbenchmark harnesses share the stdlib-only,
torch-free `hipengine_artifact_provenance` v2 collector (with backward-readable
v1) and the formal
`benchmarks/schemas/artifact-provenance.schema.json` contract. It resolves
`backend="auto"` to a concrete backend/target/device, fingerprints model
content, records physical `host_name`, and preserves staged, unstaged, and
untracked dirtiness separately.
Legacy artifact-specific provenance fields remain readable but are not a
substitute for this canonical block on new retained rows.

Non-streaming server evidence also uses `hipengine.generation_shape` v1. It
records the route cap explicitly as a queued-request limit, then records queue
request/prompt grouping, actual backend calls/widths, and speculative target
verifier rows independently. Benchmark consumers validate complete queue groups
and deduplicate the repeated shape by ID, preventing client c8 from being
reported as a width-8 backend or verifier result when the route actually ran
two c4 groups.

Direct and OpenAI completion benchmarks share exact token-ID prompt values at
the `GenerationRequest` boundary. Raw rows bypass model tokenizers in both PARO
and GGUF; server admission/usage uses their supplied lengths and returns
`hipengine.prompt_token_accounting` hashes/counts. The
`hipengine_exact_token_oracle` v1 artifact binds the committed fixture plus all
generated IDs, so HTTP evidence cannot be compared with direct evidence until
the 512/128 parity gate passes.

## License

AGPL-3.0-or-later. hipEngine is intended as copyleft software for local/home users, including the optional hosted/server paths; model weights, checkpoints, and external datasets remain under their own licenses.

## Acknowledgements

hipEngine is built on the research lineage of:
- **nano-vllm** (GeeeekExplorer) — clean engine architecture
- **mini-sglang** — production server and model definitions
- **nano-vllm-amd research** — 100+ iterations of gfx1100 kernel tuning
- **llama.cpp** (ggerganov) — Vulkan/HIP reference paths and quantization thinking
- **PARO** — W4 quantization format and pack8 layout

The engine is not a fork of any single project. It is a new integration that treats AMD RDNA3 as a first-class optimization target from day one.
