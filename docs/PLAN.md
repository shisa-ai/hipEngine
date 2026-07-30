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
- **Continuous batching is the scheduler contract.** Prefill chunks, decode steps, and speculative verification steps are separate work classes sharing the same active-request table, KV allocator, sampler, and completion/reclaim path.
- **`KVLiveSpans` is the only attention/KV-write ABI.** Dense paged KV, DMS/H2O/SnapKV, c>1 decode, and speculative verification all pass per-sequence spans rather than scalar `(block_table, context_len)` tuples.
- **KV mutation is transactional.** Canonical KV is changed only through scheduler-owned commit points. Speculative draft/verify writes go to scratch pages or an append journal and are committed by accepted-token count, then rolled back/discarded for rejected candidates.
- **Draft/verify rows are first-class.** MTP, EAGLE3, DFlash, Medusa, and Lookahead all produce `DraftBatch` metadata: `request_id`, candidate token(s), parent position, draft depth, optional tree parent, and active mask. Verification kernels consume that metadata instead of assuming a linear c=1 chain.
- **Graph capture buckets include shape, not just batch size.** Buckets are keyed by active `C`, context/page bucket, prefill/decode/verify mode, draft length or tree shape, active-mask density, top-k/experts, and graph-steps-per-replay.
- **Dispatch remains plugin-based.** c-aware or specdec-aware behavior registers new model/speculative/layer/kernel variants; engine code must not grow `if backend == ...`, `if quant == ...`, or one-off `if spec_method == ...` hot-path branches.

#### Current status

| Question | Answer |
|---|---|
| Can current hipEngine run real c=8 PARO decode? | Yes on gfx1151 for W4/BF16-KV greedy contexts covered by the retained profile. Direct physical c2/c4/c8 are independent-c1 exact at p512/d128, use 40/40 selected-batch layers, and never stack c2 groups. G5 attaches those widths to the shared resident OpenAI loop and makes them the gfx1151 package default: blocking F1 c1/c2/c4/c8 is 47.124/51.962/60.323/61.253 aggregate tok/s, all 68 rows exact, and the complementary SSE/native-plus-serial packet keeps all 100 rows exact. gfx1100 remains retained only at direct c2; sampled-native, context >=1024, other-KV, capture/replay, and gfx1100 owner c4/c8 remain open. |
| Can current hipEngine run native GGUF c>N AR? | Yes through one true physical c8 group on both gfx1100 and gfx1151. Direct eager/graph, ragged, sparse-retirement, cancellation, all-layer hidden, Conv/GDN/live-KV, profiler-family, and repeated same-session scaling gates are retained; F3/F3B's clean gfx1151 direct c1/c2/c4/c8 is 50.335/78.552/108.050/133.852 aggregate tok/s, with c8 at 2.659x c1 and 748 packed-native / zero row-local/copy dispatches. The exact singleton-indexed GDN default improves c2/c4/c8 by 8.71%/5.25%/4.04% while leaving c1 structurally unchanged; F3B then adds an exact physical-C8-only 128-thread qkv+gate pair rowtile for another +0.452%, with 30 expected pair-rowtile launches and lower widths/gfx1100 unchanged. gfx1100 keeps segmented GDN pending independent transfer. Both targets retain honest arbitrary-C/C>8 lowering as multiple declared groups. The shared owner uses dense ephemeral execution rows so live occupancy selects c1/c2/c4/c8 without moving stable scheduler slots, state, or KV; gfx1151 clean F2 server retention preserves all p512/d128 and live-transition outputs with occupancy-one at 95.625% of same-process direct c1. The current optimized corrected-window server path adds true physical C8, resident packed graphs, bounded fair-prefill bursts, resident telemetry reuse, and terminal-state discard: blocking C1/C2/C4/C8 is 44.321/59.783/75.580/86.185 tok/s, exact SSE is 42.147/59.102/73.971/84.196, delayed C8 is 67.788, and all 117 rows are exact; F3B's separate clean C1/C8 packet remains mixed within server noise and makes no additional server-speed claim. gfx1100 transfer remains separate. Neither target claims native c9/c13. gfx1151 additionally retains explicit uniform `int8_per_token_head` c1/c2/c4/c8 continuous serving through rounded context 8192 with bounded BF16 attention mirrors: corrected-window exact SSE is 42.759/55.128/71.284/81.140 tok/s, blocking is 44.225/60.598/74.631/83.408, delayed C8 is 65.034, and all 117 server rows plus the 11-prompt/99-position KL/top-1 gate pass. This is not default or memory-saving; tail4, direct/no-mirror INT8 attention, longer c>N INT8, gfx1100 transfer, and broader quant/sampling remain open. |
| What does the merged UD-Q3_K_M branch add? | A separately gated gfx1100 GPU1 direct path: exact fully-bulk Q3 prefill, native C=2/4/8 decode with exact IDs/full logits and no c>N serial fallback, and a transactional blk.40 NextN diagnostic. The direct C8 rows reach 207.780/211.177 aggregate tok/s at 512/4K; the exact NextN route is economically rejected and remains disabled. |
| Does current hipEngine implement continuous batching? | Partially project-wide; correctness and real server scaling are retained for both gfx11 GGUF OpenAI paths and for gfx1151 PARO W4/BF16-KV greedy c2/c4/c8. Blocking calls and SSE share one model-owning loop that admits during decode, executes bounded prompt chunks, streams row-owned tokens through bounded queues, cancels or retires rows, and drains through runner close. The GGUF owner densifies only execution rows and selects c1/c2/c4/c8 from occupancy while request/session/KV identity stays stable; gfx1151 F2 is retained and gfx1100 transfer is pending. PARO uses a fixed-capacity stable-slot session, profile-partitions c3/c5/c6/c7 into certified widths, and defaults native c2/c4/c8 on gfx1151. gfx1100 PARO owner symmetry and broader sampling/KV/context remain open. The gfx1151 GGUF owner also supports explicit short mirrored-INT8 continuous requests through C8 while preserving policy identity, exact outputs, reclaim, and fail-closed unsupported layouts; it does not broaden the project-wide default. |
| Is current SpecDec wired into generation? | Partially. GGUF llama-compat MTP has a guarded non-streaming greedy server route with resident slots and packed target verify; exact/default MTP serving, streaming, and broad SpecDec pluginization remain future work. |
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
- `KVLiveSpans` and `KVPolicy.batch_spans(...)` are intended to represent per-sequence KV state rather than a single scalar `(block_table, context_len)` pair.
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
  and stale-pointer-safe graph regrow. gfx1151 now also carries short uniform
  `int8_per_token_head` payloads, FP16 scales, and bounded BF16 mirrors through
  continuous ownership at c1/c2/c4/c8 with exact API/quality/reclaim evidence.
  gfx1100 transfer, longer c>N INT8, tail4, and direct/no-mirror INT8 attention
  remain independent gates.
- Several decode kernels are row-parallel GEMV rather than true grouped/MMQ/WMMA
  batch kernels. They increase grid size but do not reliably reuse streamed
  weights across requests, which is visible in the weak gfx1151 c=1->c=8 scale
  versus llama.cpp Vulkan.
- GQA split-K and full-attention now have primitive trace/parity plus exact
  per-sequence `KVLiveSpans` server coverage for gfx1151 BF16-KV c2 through 64K.
  Short mirrored uniform INT8 now has payload/scale-backed gfx1151 c2/c4/c8
  evidence through 1K allocated context; equivalent longer/direct-INT8/tail4 and
  gfx1100 spans still need independent row-count-specific evidence. See the
  [clean mirrored-INT8 continuous packet](../benchmarks/results/2026-07-19-gfx1151-gguf-mirrored-int8-continuous-concurrency.json).
- Selected MoE decode has row-aware/grouped diagnostic coverage for c<=8, but
  retained performance still needs routed-lane profiling and c-aware thresholds
  for grouped GEMV versus compact/WMMA execution.
- GGUF MTP serving is phase-serial at the slot level: draft, target verify, then
  commit. Target verify is packed up to four slots. The canonical milestone
  glossary, ownership distinctions, and qualified scorecard are in
  [`NATIVE_SPEC_CYCLE.md`](NATIVE_SPEC_CYCLE.md). The provider-neutral
  `NativeSpecCycleLauncher` N0 ABI plus gfx1100 reusable B1/B2 N1 target graphs
  are landed. N1 is byte-exact across dynamic positions and cached-session
  resets; the retained accuracy-traded llama-compat suite reaches 122.667 tok/s
  versus llama.cpp's 115.444 tok/s W7900 floor. N2 device acceptance, selected
  hidden/Conv/GDN commit, cursor update, and bounded summary readback are also
  landed behind the explicit llama-compat native-cycle route. N2 keeps verifier
  hidden rows graph-owned so prompt-prefill scratch growth cannot invalidate a
  captured pointer, and matches all 240 IDs / 96 cycle semantics in the full
  category+heldout suite. Its first same-tree aggregate screen is neutral within
  run variance, while state/KV commit and host-seed sub-windows shrink. N3 now
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
│   ├── LAGUNA-prefill.md        # Active Laguna arithmetic/MMQ prefill attack plan
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

Select **WPF-H6B exact active-IQ3 signed-magnitude segment plane**. A bounded
producer writes one aligned 16-byte record per active-expert/output/group8:
the exact current F32 scale, eight exact signed int8 magnitudes, and padding. A
matching H5Z-derived consumer performs exact int8-to-F32 conversion and retains
the scalar BF16 dot, scale multiply, wave tree, serial wave sum, rowbatch8, P64
expert, P256 output, store, metadata, and fallback order. Natural M512 has
**9,844** active layer-expert instances and **33,547** rowbatch8 iterations, so
the static repeated-decode ratio is **3.40786x**. The fixed 256-expert plane is
**1,610,612,736 bytes** and combined workspace is **1,771,732,992 bytes**; this
is an admission bound/static rationale, not a speed claim. Require exact record/
output bytes, producer-inclusive all-45-layer both-clock wins, bounded resources/
memory, complete state, and clean 512/1K/4K before ownership
([post-H6A matched residual / comparator correction / H6B target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6a-matched-residual.json) ·
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
The old wider-qrow, cross-head/key-split, rowbatch16, output-tile/source-MMQ,
changed-association attention, H5O representation, H5P geometry, H5S persistent
ownership, H5T one-wave IQ3 ownership, and P6/repair routes remain closed.
Launch fusion remains deferred.
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
- **Model**: exact checkpoint name, immutable revision, and content fingerprint
- **Quantization**: FP16, W8A16, W4, etc.
- **Workload**: prompt/generation length, client concurrency, choices, queue
  grouping, actual backend widths, and verifier rows
- **Hardware**: selected GPU, configured/resolved backend, target arch, ROCm and compiler versions
- **Source**: hipEngine commit plus separate staged, unstaged, and untracked state
- **Command**: exact benchmark invocation
- **Result**: tok/s prefill, tok/s decode, peak GiB
- **Correctness**: KL divergence ≤ 0.05, top-1 agreement ≥ 90%

This policy is inherited from the `LESSONS-LEARNED.md` discipline: fast rows are invalid until output sanity proves they are real.

Server, retained PARO, GGUF, and microbenchmark harnesses share the stdlib-only,
torch-free `hipengine_artifact_provenance` v1 collector and the formal
`benchmarks/schemas/artifact-provenance.schema.json` contract. It resolves
`backend="auto"` to a concrete backend/target/device, fingerprints model
content, and preserves staged, unstaged, and untracked dirtiness separately.
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
