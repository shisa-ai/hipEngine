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

hipEngine treats memory as a hierarchy of tiers with async migration, not a single GPU buffer. This enables running models and contexts far exceeding single-GPU memory without kernel changes.

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
# hipengine/kvcache/tiered_policy.py
class TieredKVPolicy(KVPolicy):
    def __init__(self, 
                 device_budget: int,      # blocks on GPU
                 host_budget: int,        # blocks in pinned CPU RAM
                 disk_path: Path | None,  # SSD spillover
                 compression: str | None):  # "fp16", "int8", "q4"
        ...
    
    def evict(self, pressure_tokens: int) -> list[BlockRange]:
        # Device → Host (compress) → Disk
        # Uses per-head/layer DMS importance scores
        ...
    
    def prefetch(self, block_ids: list[int], stream):
        # Async device←host←disk for upcoming decode
        ...
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

### Integration with KVPolicy

```python
# Usage: pick your memory/performance tradeoff
policy = KVPolicy.device_only()              # 24 GiB limit, fastest
policy = KVPolicy.tiered(                     # Balanced
    device_budget=4096, host_budget=16384,
    disk_path="/mnt/kvcache", compression="int8")
policy = KVPolicy.kvtc_offload(                # Aggressive offloading
    host_budget=8*1024**3, disk_path="/mnt/kvcache",
    prefetch_depth=2)
policy = KVPolicy.dms_per_head()               # Smart eviction
```

### Why No Kernel Changes

The kernel layer sees `hipengine.Tensor` (raw device ptr + metadata) on the active HIP/CUDA device. The host ensures tensors are on the right tier before calling kernels. Async prefetch hides latency. This is **memory management, not math**.

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
| **Fast dispatch, no Python in the hot path** | Decode forward is captured into a `hipGraph` at warmup and replayed with zero Python overhead per subsequent step. Python runs only once per token for sampling. |
| **Fused + unfused kernels coexist** | Every fused composite (`rmsnorm_rotate`, `gate_combine_residual`, etc.) has an unfused chain equivalent. The dispatcher prefers fused when a registered composite matches the upcoming op chain and falls back to unfused primitives when not. Unfused kernels also serve as the correctness baseline. |
| **Library-first, server-included** | `pip install hipengine` gives you `from hipengine import LLM` plus the `hipengine serve` OpenAI-compatible server CLI. The torch-free inference hot path still does not import FastAPI/Uvicorn. |
| **Extensible by design** | Four orthogonal plugin axes — **backend**, **model**, **quant**, **layer** — not hardcoded branches. See Extensibility Design. |
| **Evidence-backed performance** | Every performance claim comes with a reproducible benchmark command, hardware context, and workload shape. No marketing numbers. |

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
│  • Block Manager    — paged KV with pluggable KVPolicy           │
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
│  • graph.py          — hipGraph capture + replay via ctypes      │
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

hipEngine is a better foundation for c>1 than the current `nano-vllm-amd` native PARO path, but the runnable implementation remains c=1 until the batch-state and c-aware kernels below land. Treat current Qwen3.5/PARO numbers as **single-request decode** unless a benchmark explicitly says otherwise.

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
- **Continuous batching is the scheduler contract.** Prefill chunks, decode steps, and speculative verification steps are separate work classes sharing the same active-request table, KV allocator, sampler, and completion/reclaim path.
- **`KVLiveSpans` is the only attention/KV-write ABI.** Dense paged KV, DMS/H2O/SnapKV, c>1 decode, and speculative verification all pass per-sequence spans rather than scalar `(block_table, context_len)` tuples.
- **KV mutation is transactional.** Canonical KV is changed only through scheduler-owned commit points. Speculative draft/verify writes go to scratch pages or an append journal and are committed by accepted-token count, then rolled back/discarded for rejected candidates.
- **Draft/verify rows are first-class.** MTP, EAGLE3, DFlash, Medusa, and Lookahead all produce `DraftBatch` metadata: `request_id`, candidate token(s), parent position, draft depth, optional tree parent, and active mask. Verification kernels consume that metadata instead of assuming a linear c=1 chain.
- **Graph capture buckets include shape, not just batch size.** Buckets are keyed by active `C`, context/page bucket, prefill/decode/verify mode, draft length or tree shape, active-mask density, top-k/experts, and graph-steps-per-replay.
- **Dispatch remains plugin-based.** c-aware or specdec-aware behavior registers new model/speculative/layer/kernel variants; engine code must not grow `if backend == ...`, `if quant == ...`, or one-off `if spec_method == ...` hot-path branches.

#### Current status

| Question | Answer |
|---|---|
| Can current hipEngine run real c=8 PARO decode? | No. |
| Does current hipEngine implement continuous batching? | No. |
| Is current SpecDec wired into generation? | No; only the design/file-tree placeholder exists. |
| Is the design cleaner for adding c>1 than `nano-vllm-amd`? | Yes. |
| Would just setting `tokens=8` work? | No. |
| Is hipEngine the better place to build c=8+ PARO and SpecDec? | Probably yes. |

Why the design is better positioned:

- The hot path owns raw HIP pointers and `hipGraph` replay directly instead of depending on torch tensors or PyTorch graph wrappers.
- Many wrappers already expose `tokens`, `rows`, or row-shaped grids, so partial batching can be tested without changing the public API.
- `KVLiveSpans` and `KVPolicy.batch_spans(...)` are intended to represent per-sequence KV state rather than a single scalar `(block_table, context_len)` pair.
- The kernel registry can add c-specific variants such as `(layer="selected_pack8_gemv", variant="batch8")` or `(layer="paged_attn_decode", variant="gqa_batch")` without engine-wide backend/quant branches.
- Decode graph capture is already framed as shape buckets rather than one global graph.
- Model plugins can advertise optional speculative heads, while speculative methods live under their own plugin boundary instead of forking the engine.

Current blockers that keep Qwen3.5/PARO effectively c=1:

- `hipengine.generation.qwen35_paro.Qwen35ParoOneTokenGenerator` is a smoke path: it requires `max_tokens == 1` and serializes prompts in Python.
- `Qwen35ParoResidentSession` owns single-request state: `(1, hidden_size)` hidden buffers, scalar token/position/context device buffers, one block table/span object, and one KV cache per full-attention layer.
- There is no active-request table, no request admission loop, no batch compaction/reclaim path, no mixed prefill+decode scheduler, and no per-request sampler/output queue.
- Several decode orchestrators still reject `tokens != 1` or only support c1-style GEMV batching; this is useful for prefill bring-up but not a complete concurrent decode scheduler.
- The current GQA split-K attention kernels consume one query stream with scalar context length. c>1 needs a batch grid dimension plus per-sequence `live_counts`/page tables.
- Selected MoE GEMV needs a real mapping from token rows to routed lanes. For c=8 and top-k=8 the natural shape is `x_rows=8`, `rows=64`; kernels must gather hidden rows by `lane // top_k` or run grouped/compact expert kernels rather than assuming one hidden row or one row per lane.
- Speculative decode needs transactional KV scratch/journaling, candidate-row metadata, target verification passes, and scheduler-side accept/reject accounting before MTP/EAGLE3/DFlash can be enabled.

#### Expected c=8 behavior

| Path | Expected aggregate c=8 behavior |
|---|---|
| Current hipEngine as-is | Unsupported. |
| Eight serial c=1 sessions sharing weights | About 1× c1 aggregate, worse latency. |
| Naive `rows=8` where wrappers allow it | Modest gain from larger grids and lower relative launch cost; weights are still mostly reloaded per row. |
| Proper c=8 batch path | Plausibly 2–4× c1 aggregate for Qwen3.5/PARO decode; not 8×. |
| c>16 | Prefer GEMM/MMQ/WMMA and grouped MoE designs over extending c1 GEMV. |

The key distinction is that many current "batched" kernels are row-parallel GEMV. They increase grid size, but they do not automatically reuse streamed weights across requests the way a true GEMM/MMQ/WMMA or grouped-MoE kernel can.

#### Implementation plan

1. **Request and batch-state containers.** Add `RequestState` plus `ResidentBatchSession` (or equivalent) with `[C, hidden]` buffers, device token ids, per-request positions/context lengths, active masks, finish flags, per-layer linear-attention recurrent/conv state, and per-request/per-layer full-attention KV spans.
2. **Continuous-batching scheduler.** Add admission, chunked prefill, decode-step batching, slot compaction, sampler/output routing, and reclaim around `KVPolicy.batch_spans(...)`. The scheduler owns physical slots and stable request ids; kernels only see row metadata.
3. **Correctness harness first.** For fixed prompts and greedy sampling, compare c=2/4/8 batch output against independent c=1 runs. Require finite logits, matching generated ids for deterministic fixtures, and per-layer state/KV bounds checks before any perf claim.
4. **Transactional KV hooks.** Extend the KV policy contract with scratch/journal allocation and `commit(request_id, accepted_tokens)` / `rollback(request_id)` semantics before speculative verification writes can touch canonical KV.
5. **Attention batch kernels.** Add batched paged GQA decode and KV append variants with a batch grid dimension and per-sequence span metadata. Uniform paged KV is first; DMS/variable spans reuse the same public ABI later.
6. **Linear-attention state kernels.** Make conv/GDN recurrent decode consume `[C, ...]` state and update each sequence independently.
7. **MoE batch kernels.** Replace c1 selected-lane assumptions with token→lane mapping, then add grouped-by-expert and compact/WMMA routes once routed-lane counts justify them. Use routed lanes, not token count alone, for the GEMV-vs-WMMA threshold.
8. **Quantized projection dispatch.** Use c-aware rules: c=1 stays GEMV; c=2/4/8 uses multi-column/MMQ-style kernels where they beat row-GEMV; c>16 moves toward GEMM/WMMA.
9. **SpecDec plugin boundary.** Add `DraftModel`, `DraftBatch`, `Verifier`, and `AcceptResult` interfaces. MTP heads are model-attached draft providers; EAGLE3 and DFlash are draft-model plugins; Lookahead/Medusa are lightweight draft providers. All verify through the same target-model batch runner and transactional KV path.
10. **Graph bucket policy.** Capture/replay by active `C`, context bucket, mode (`prefill`, `decode`, `verify_chain`, `verify_tree`), draft depth/tree shape, top-k/experts, and replay length. Fall back to uncaptured launches for rare shapes.
11. **Benchmark protocol.** Add c=N concurrent rows and SpecDec rows only after the corresponding correctness harness is green. Report aggregate tok/s, per-request tok/s, p50/p95 latency, memory, active batch occupancy, graph bucket, acceptance rate, accepted tokens per target pass, and generated-token equality vs non-spec c1.

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

**Rule:** we do not add levers #2–5 without `rocprofv3` evidence that dispatch is above ~3% of decode wall time.

#### Fusion Planner

Dispatch converts a layer's op chain into a kernel plan. Fused composites are preferred when a registered kernel matches a contiguous sub-chain; otherwise the planner falls back to unfused primitives. Every fused kernel must have an unfused chain that is numerically equivalent (used as correctness baseline and fallback for backends that haven't ported the composite yet).

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
| **Backend** | Hardware target (kernel set + primitives) | `hip_gfx1100`, `hip_gfx1151`, `cuda_sm86`, `cuda_sm89`, `cpu_reference` |
| **Model** | Architecture-level layer sequence + weight name map + chat template | `qwen3_dense`, `qwen3_5_hybrid` (full+linear+GDN+MoE), `gemma4`, `llama3`, `sansho` |
| **Quant** | Weight layout + packing + activation quant | `fp16`, `bf16`, `w8a8_dyn`, `w8a16`, `w4_paro`, `w4_gguf`, `int4_awq_orig` |
| **Layer** | Per-layer-type compute structure (primitive + fused variants) | `full_attention`, `linear_attention`, `gdn`, `sliding_attention`, `moe_top2`, `dense_mlp` |

Kernels are registered with the tuple `(backend, layer, quant, variant)`. The dispatcher resolves kernels at layer-build time; the fusion planner resolves at op-chain-build time.

Public APIs and server entry points default to `backend="auto"`. Auto is a selector
resolved before registry lookup, not a registry key: exact `gfx1100`/`gfx1151`
detections map to the matching HIP backend, `HIPENGINE_BACKEND` can force a
backend for nearby targets such as `gfx1101`/`gfx1102`, and unknown/no HIP
detections warn before selecting `cpu_reference` where a CPU implementation exists.

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

Adding a CUDA backend = new `hipengine/kernels/cuda_sm86/...` tree with the same `layer` / `quant` / `variant` key space. Adding Strix Halo = `hipengine/kernels/hip_gfx1151/...`. The engine, dispatch, model, and quant layers don't change.

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
| **Qwen3.6 35B-A3B** MoE hybrid | full_attention + linear_attention + gdn + moe_top2 | Phase 2 perf target |
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

KV cache has **two orthogonal axes**, plus the standard block-manager concerns. Designing for both from day 0 is the specific lesson from `~/FastDMS` — integrating DMS into vLLM is "major surgery" ([FastDMS README](/home/lhl/FastDMS/README.md)) precisely because vLLM's KV pool assumes fixed-page uniform-per-sequence blocks. hipEngine avoids that trap by designing the interface around per-(seq, layer, head) live spans from the start, even if the default policy has uniform spans.

| Axis | What varies | Examples |
|------|-------------|----------|
| **Eviction / compaction** | How live spans change over time | fixed-page (standard paged KV); sliding-window; attention-sink + sliding (StreamingLLM); DMS per-head learned eviction; H2O heavy-hitter; SnapKV prompt-time pruning |
| **Storage dtype** | KV precision | `bf16`, `fp16`, `fp8_e4m3`, `int8_per_channel`, `int4_packed`, `turboquant_4bit`, `higgs_4bit`, `aqua_kv` (cross-layer predicted residual) |

#### `KVLiveSpans` — the fundamental kernel interface

Every attention / paged-KV-write kernel takes a `KVLiveSpans` instead of the classic `(block_table, context_len)` tuple. Uniform policies fill it the same for every head; DMS varies it. `num_seqs` is intentionally a row count: it can mean active decode requests (`C`), prefill chunks, or speculative verification rows (`V`). Stable request identity remains scheduler metadata, not an implicit row index.

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
    storage_dtype:   DType           # dtype of the K/V arena (bf16, fp8, int4, ...)
```

#### `KVPolicy` protocol

```python
class KVPolicy(Protocol):
    spans_mode: str                  # "uniform", "per_head_variable"
    storage_dtype: DType

    def allocate(self, seq: Sequence, prefill_len: int, decode_budget: int) -> KVReservation: ...
    def admission_cap(self, seq: Sequence) -> int:
        """Token budget used by the scheduler — compact tokens for DMS,
        dense page-equivalent for fixed-page."""
    def prefill_spans(self, seq: Sequence) -> KVLiveSpans: ...
    def decode_step(self, seqs: list[Sequence],
                    new_k: Tensor, new_v: Tensor, q: Tensor | None) -> None:
        """Store committed decode K/V. q is passed for policies that need
        query-conditional eviction (DMS uses the last query channel as the
        eviction signal)."""
    def batch_spans(self, batch: list[Sequence], *, role: str = "decode") -> KVLiveSpans: ...
    def begin_transaction(self, seqs: list[Sequence], draft: DraftBatch) -> KVTransaction: ...
    def commit(self, txn: KVTransaction, accepted_counts: Tensor) -> None: ...
    def rollback(self, txn: KVTransaction) -> None: ...
    def reclaim(self, seq: Sequence) -> None: ...

# Built-in policies (Phase 0/2)
policy = KVPolicy.paged_bf16()        # fixed pages, BF16, the nano-vllm default
policy = KVPolicy.paged_fp8()         # fixed pages, FP8 KV (works on any GPU via software)
policy = KVPolicy.radix_cache()       # prefix-sharing trie, BF16
policy = KVPolicy.sliding_sink(sink=4, window=1024)  # StreamingLLM

# Phase 4 (DMS support)
policy = KVPolicy.dms_fp8(            # FastDMS compact default
    retention_mode="dms",
    storage_dtype="fp8_e4m3")
policy = KVPolicy.dms_int4_shadow()   # FastDMS B46/B25 storage-for-speed profile

# Phase 5+ (research)
policy = KVPolicy.h2o(heavy_budget=256)
policy = KVPolicy.snapkv(compression=8)
policy = KVPolicy.aqua_kv(higgs_bits=4)  # DMS + AQUA + HIGGS (sansho's 25.6x stack)
```

**Scheduler admission** queries `KVPolicy.admission_cap()` per sequence. Fixed-page policies return `num_pages * block_size - current_usage`. DMS returns the per-(layer,head) `range_capacity - live_counts` minimum across all layers/heads. The scheduler doesn't know which policy it's talking to.

**Attention kernels** are registered under a `layer` key that matches the span mode: `paged_attn_decode` for uniform, `compact_attn_decode` for per-head-variable (which DMS uses). The kernel registry naturally routes.

#### Why this shape avoids the vLLM-DMS pain

The FastDMS README lists seven subsystems that a DMS port to vLLM has to change (PagedAttention memory pool, prefill kernel, decode kernel, attention scoring, scheduler/admission, prefix caching, continuous batching). hipEngine pays that design cost once, up front, by making `KVLiveSpans` + `KVPolicy.admission_cap()` the fundamental contract. Adding DMS later is **one new KVPolicy subclass** (`DMSKVPolicy`) plus **three new HIP kernels** (`dms_rope_store_compact_decode`, `compact_decode_grouped_splitk`, `streaming_pack_scatter`) ported from the `~/FastDMS` Triton reference. No engine rewrite.


## Advanced Features Roadmap

### Speculative Decoding (SpecDec)

SpecDec is planned as a scheduler + plugin feature that reuses the same target-model batch runner, KV policy, and kernel registry described in the c>1 readiness section. Drafting changes the work shape; it must not fork the engine.

| Draft Type | Status | Integration shape |
|------------|--------|-------------------|
| Medusa-style heads | Planned | Model-advertised heads produce shallow candidate rows. |
| Lookahead decoding | Planned | Scheduler-side n-gram/cache provider emits candidate chains. |
| MTP (multi-token pred) | Research | Qwen3.5 MTP layers provide `DraftBatch` chains attached to the target model; detailed native plan: [`docs/MTP.md`](MTP.md). |
| EAGLE3 | Research | Draft-model plugin emits feature-conditioned candidate chains/trees. |
| DFlash (draft model) | Research | z-lab/FastKMS-lineage draft-model plugin plus DDTree/tree-verify support; detailed native plan: [`docs/DFLASH.md`](DFLASH.md). |

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

The `KVPolicy.kvtc_offload()` plugin manages:
- Which blocks stay device-resident
- Which blocks are pinned host-resident (fast prefetch)
- Which blocks are compressed before offloading
- Prefetch scheduling for decode-time block retrieval

### RadixCache vs. vLLM Prefix Caching

| Feature | vLLM Prefix Caching | hipEngine RadixCache (mini-sglang) |
|---------|---------------------|-----------------------------------|
| Structure | Hash-based block matching | Trie-based prefix tree |
| Granularity | Block-level (256 tokens) | Token-level exact prefix |
| Sharing | Copy-on-write blocks | Reference-counted trie nodes |
| Eviction | LRU on blocks | LRU on trie nodes (finer-grained) |
| Overhead | Lower | Slightly higher CPU, better hit rate |

hipEngine defaults to **RadixCache** for better prefix sharing in multi-turn chat and API serving. vLLM-style is available as `KVPolicy.prefix_lru()`.

### DMS Support Plan (and why it shapes Phase-0 design)

See [docs/KVCACHE.md](KVCACHE.md) for the staged delivery order: dense paged INT8 KV with no BF16 shadowing first, then FastDMS-derived compact DMS over the same `KVLiveSpans` ABI.

Dynamic Memory Sparsification (DMS) trains per-head learned KV token eviction via logit distillation. Compact DMS saves real allocator memory (5–8× vs BF16 KV at 8K context, up to 49× at max context per `~/FastDMS` benchmarks) while maintaining or improving decode speed. The reference open implementation is `~/FastDMS` (shisa-ai). Validated checkpoints: `shisa-ai/Llama-3.2-1B-DMS-8x`, `nvidia/Qwen3-8B-DMS-8x`.

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
| `KVPolicy.admission_cap(seq)` as the scheduler's unit | Fixed-page returns page-equivalent; DMS returns compact-token budget. Scheduler doesn't care which. |
| Fusion planner with chain-matching (not hardcoded ops) | DMS needs fused `rotate + dms_decide + compact_store + decode` kernels. These register as fused composites for `(quant, layer="rotate+dms+store+attn"`. |
| `storage_dtype` as a `KVPolicy` property, separate from eviction | DMS + BF16, DMS + FP8, DMS + int4-shadow, DMS + AQUA all compose. (`~/kvcache-quantization-research/` showed DMS + AQUA + HIGGS hitting 25.6× at +0.09% PPL.) |
| Model plugin accepts "DMS-trained" as a model subtype | DMS-trained checkpoints carry per-head eviction head weights (borrowed query channel, alpha scale/offset). Loader gets a `dms_config` sub-block. |
| `KVPolicy` + Attention kernel registration under `layer="compact_attn_decode"` key | When a user picks `KVPolicy.dms_fp8()`, the dispatcher routes to compact-decode kernels. No engine-wide branches. |

#### Phase 4 DMS delivery

With the Phase-0 groundwork, adding DMS is:

1. **One KVPolicy subclass** — `DMSKVPolicy` (~400 Python, most of it the compaction bookkeeping from `~/FastDMS/fastdms/engine/compact_kv.py` 1,850 lines → our ~400 because the `KVLiveSpans` plumbing is already there)
2. **Three new HIP kernels** ported from `~/FastDMS` Triton reference:
   - `dms_rope_store_compact_decode` (fuses RoPE + eviction decision + compact store at decode)
   - `compact_decode_grouped_splitk` (attention over variable per-head live spans)
   - `streaming_pack_scatter` (prefill surviving-K/V pack)
   - ~1,500 HIP total
3. **Model-plugin extension**: `DMSRetrofitConfig` dataclass loaded from the checkpoint, wires per-head eviction heads into the attention layer
4. **Scheduler glue**: `admission_cap()` already exists; just needs a DMS-specific calculator (~50 LoC)

Total DMS support: **~2,000 LoC** in Phase 4, vs a "multi-week major surgery" port inside vLLM. The Phase-0 `KVLiveSpans` design is the reason the port is this small.

#### What's deferred beyond DMS

| Technique | Blocker |
|---|---|
| AQUA-KV cross-layer residual predictor | Needs per-layer scalar quant codec. Research, ~800 LoC if pursued |
| HIGGS 4-bit KV | ~50% BF16 speed in `kvcache-quantization-research/`; defer until kernel faster |
| H2O / SnapKV heavy-hitter | Research; same `KVLiveSpans` fits; ~300 LoC policy |
| StreamingLLM + attention sinks | Phase 3, ~200 LoC policy; no new kernels |
| TurboQuant 4-bit KV | vLLM-compatible format; implement as `KVPolicy.turboquant_4bit()` if users need it |

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
│   │   ├── block_manager.py     # Paged allocation with KVPolicy
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
│   │   ├── policy.py            # KVPolicy interface + built-ins
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

| Phase | Scope | New LoC | Adapted LoC | Total |
|-------|-------|---------|-------------|-------|
| **0. Foundation** | Core host (scheduler, block manager, engine loop, model registry, fusion planner) | ~700 | ~0 | **~700** |
| | Torch-free core primitives (`hipengine.core.*`: Tensor, device, memory, graph, blas, build, stream) | ~1,900 | ~0 | **~1,900** |
| | Torch-free loading (safetensors + HF config + chat template + tokenizer glue) | ~900 | ~0 | **~900** |
| | `KVLiveSpans` + `KVPolicy.admission_cap()` + per-head-variable-span attention kernel ABI | ~250 | ~0 | **~250** |
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
| | Sliding-window + attention-sink `KVPolicy` (StreamingLLM) | ~200 | ~0 | **~200** |
| | `KVPolicy.paged_fp8()` (software FP8 KV, works on any backend) | ~250 | ~0 | **~250** |
| | Basic multi-GPU TP (rccl all-reduce via ctypes) | ~150 | ~0 | **~150** |
| | Gemma 4 model plugin + sliding_attention kernels | ~500 | ~0 | **~500** |
| | Llama 3 model plugin | ~200 | ~0 | **~200** |
| | sansho custom arch plugin | ~300 | ~0 | **~300** |
| **4. SpecDec + DMS** | `DraftModel` interface | ~50 | ~0 | **~50** |
| | Medusa / Lookahead / MTP / DFlash paths | ~200 each | ~0 | **~800** |
| | Scheduler speculation awareness | ~100 | ~0 | **~100** |
| | `DMSKVPolicy` + model-plugin DMS config loader (eviction head weights) | ~500 | ~0 | **~500** |
| | DMS HIP kernels: `dms_rope_store_compact_decode`, `compact_decode_grouped_splitk`, `streaming_pack_scatter` | ~1,500 (HIP) | ~0 | **~1,500** |
| **5. Advanced Features** | C++ engine-step extension (lever #2) if profiling demands | ~1,500 | ~0 | **~1,500** |
| | CUDA backend (`kernels/cuda_sm86/`) — reuse kernel tree shape | ~500 scaffolding | **~18,630** (retyped + recompiled per-kernel porting) | **~19,130** |
| | EXL3 / QTIP codebook kernel family (new `codebook_lut` tree, ~14 kernels) | ~300 | ~8,000 (port from ExLlamaV3) | **~8,300** |
| | FastKron `kronecker` kernel family (compute pattern rewrite) | ~1,500 | ~0 | **~1,500** |
| | FP8 weight quant (only on `hip_gfx1200`+ / `cuda_sm90`+; skipped on gfx1100) | ~400 | ~0 | **~400** |
| | H2O / SnapKV `KVPolicy` plugins | ~600 | ~0 | **~600** |
| | AQUA-KV cross-layer predictor (requires per-layer scalar-quant codec) | ~800 | ~0 | **~800** |
| | Tiered offloading (host pinning, disk spill) | ~400 | ~0 | **~400** |
| | Session save/restore (ds4-style) | ~150 | ~0 | **~150** |
| | Expert CPU offload (ktransformers-style) | ~300 | ~0 | **~300** |
| | GGUF Q4_K_M loader | ~500 | ~0 | **~500** |
| | Pipeline Parallelism | ~200 | ~0 | **~200** |
| | Expert Parallelism | ~250 | ~0 | **~250** |

**Cumulative totals:**
- Phase 0 (MVP): ~36,640 lines (~700 host + ~1,900 core + ~900 loading + ~250 KVLiveSpans + ~18,930 HIP+bindings + ~1,240 retype + ~11,400 dispatch + ~1,500 FA2 + ~800 cpu_reference + ~20 smoke)
- Phase 1 (server+bench): +750 lines → **~37,390**
- Phase 2 (quant+MoE): +2,400 lines → **~39,790** (adds GPTQ/GPTAQ/AWQ line)
- Phase 3 (KV+prefix+TP+models): +1,950 lines → **~41,740** (adds StreamingLLM, paged_fp8)
- Phase 4 (specdec+DMS): +2,950 lines → **~44,690** (adds DMS policy + kernels)
- Phase 5 (advanced, incl. CUDA backend + codebook + FastKron + H2O/AQUA): +34,130 lines → **~78,820**

> **Note:** LoC is an imperfect proxy for effort. ~17,590 HIP lines + ~1,040 retyped bindings are **copied and repartitioned kernels** (known working; split + retype are mechanical and gated by `rocprofv3` + KL). ~10,900 Python dispatch lines are **adapted** — real porting work because they encode kernel-selection policy and weight layout. The torch-free core (~1,900) and loading (~900) and CPU reference (~800) are **new engineering** but ~80% straightforward and testable against the existing torch-based workspace as oracle. The FA2 prefill kernel (~1,500 HIP) and the DMS compact-decode kernels (~1,500 HIP) are the two hardest new HIP pieces. Phase-5 CUDA backend is the largest single deferred item because each of the 120 kernels needs a CUDA variant (though most are straightforward: wavefront=32, `cub::WarpReduce` instead of AMD shuffle, `wmma` instead of ROCm WMMA). **The Phase-4 DMS delivery is ~2,500 LoC total, not a multi-week surgical port**, because the Phase-0 `KVLiveSpans` + `KVPolicy.admission_cap()` interface was designed for it from day 0.

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
| KV compression: DMS | Major surgery (per FastDMS README) | No | No | No | Yes (reference impl) | **Phase 4 via `DMSKVPolicy`; `KVLiveSpans` interface designed day-1** |
| KV compression: H2O / SnapKV / sliding | Sliding (via model) | No | No | — | — | **Phase 3 sliding, Phase 5 H2O/SnapKV** |
| KV storage dtype (orthogonal to eviction) | bf16, fp8, TurboQuant-4bit | bf16 | Various | — | bf16, fp8, int4-shadow | **Orthogonal `storage_dtype` axis on every `KVPolicy`** |
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
| KV cache | `KVPolicy` with `KVLiveSpans` as the kernel ABI and `admission_cap()` as the scheduler unit | Makes DMS, H2O, SnapKV, StreamingLLM all drop-in policy plugins. Avoids the vLLM-DMS "major surgery" (per `~/FastDMS/README.md`). RadixCache default; others plug in. |
| DMS support | Phase 4, ~2,500 LoC total (`DMSKVPolicy` + 3 HIP kernels + loader) | `KVLiveSpans` + `admission_cap()` designed day-1 so DMS is a policy drop, not a rewrite |
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
- **Model**: exact checkpoint name
- **Quantization**: FP16, W8A16, W4, etc.
- **Workload**: prompt length, generation length, batch size
- **Hardware**: W7900/gfx1100, ROCm version, PyTorch version
- **Command**: exact benchmark invocation
- **Result**: tok/s prefill, tok/s decode, peak GiB
- **Correctness**: KL divergence ≤ 0.05, top-1 agreement ≥ 90%

This policy is inherited from the `LESSONS-LEARNED.md` discipline: fast rows are invalid until output sanity proves they are real.

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
