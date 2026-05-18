# Speculative Decoding: Analysis & Architecture Constraints

## Executive Summary

Speculative decoding (DFlash, MTP, Medusa, EAGLE, etc.) fundamentally assumes
that **verification of B draft tokens costs approximately the same as one AR
step** — because during decode, transformer models are memory-bandwidth-bound
and the extra arithmetic for B tokens is cheap compared with loading the same
weights repeatedly.

**This assumption breaks on 256-expert MoE models with sequential per-token
dispatch at c=1.** On the Qwen3.6-35B-A3B architecture with our current verifier,
verification costs scale nearly linearly with B rather than being close to constant.
That makes our measured DFlash/MTP loops lose to AR on aggregate prompts; even
very high acceptance has only a narrow ceiling unless verifier cost drops sharply.

This is not a universal "MoE can't speculate" claim — it is specific to this
combination of high expert count (256), high top-K (8), sequential dispatch,
recurrent state layers, and c=1 workload. At higher concurrency or with
grouped/budgeted expert dispatch, the economics could change.

For dense or dense-like 27B-31B targets, the same verification infrastructure
should be much more favorable. The projected range is roughly 2-3.5x, depending
on the target's actual AR speed, verifier efficiency, and draft cost.

---

## Hardware: AMD Radeon Pro W7900 (gfx1100/RDNA3)

### Specifications

```
GPU:                  AMD Radeon Pro W7900
Architecture:         RDNA 3 (gfx1100)
Chiplet Design:       GCD (5nm) + MCD (6nm)
Compute Units:        96 CUs
SIMDs per CU:         2 (dual-issue SIMD32 + SIMD32)
Shader Engines:       6
Shader Arrays/Engine:  2
Max Clock:            2495 MHz (boost), 1760 MHz (reported by rocminfo)
Wavefront Size:       32 (native), 64 (CU mode for compute)
Max Waves per CU:     32 (wave32) or 16 (wave64)
Max Workgroup Size:   1024 threads
TDP:                  295W
```

### Memory Hierarchy

```
VRAM:                 48 GB GDDR6 (ECC)
Bus Width:            384-bit
Memory Speed:         18 Gbps (effective)
Peak Bandwidth:       864 GB/s (theoretical)
Measured Bandwidth:   ~700-750 GB/s (sustained inference workloads)
Infinity Cache (L3):  96 MB (98304 KB per rocminfo)
L2 Cache:             6 MB (6144 KB)
L1 (per CU):          32 KB
L0 Vector Cache:      32 KB
LDS (per WGP):        128 KB (64 KB per CU)
Cacheline Size:       128 bytes
```

### Compute Throughput

```
Vector (SIMD) Compute:
  FP64:               1.92 TFLOPS
  FP32:               61.3 TFLOPS
  FP16/BF16:          122.6 TFLOPS (2× rate via packed math)
  INT8:               122.6 TOPS (same rate as FP16 on vector units)

Matrix (WMMA) Compute:
  FP16/BF16:          512 FLOPS/clock/CU → 96 CU × 2495 MHz × 512 = 122.6 TFLOPS
  INT8:               512 OPS/clock/CU (same as FP16 on RDNA3!)
  INT4:               1024 OPS/clock/CU → 245 TOPS
  WMMA tile size:     16×16×16 (wave32)
  WMMA cycles:        32 clocks per instruction

Key insight: On RDNA3, INT8 WMMA throughput equals FP16 WMMA throughput
per clock per CU. INT8 inference benefits come ONLY from halving memory
traffic (bandwidth-bound), not from higher compute throughput.
This is unlike NVIDIA (where INT8 tensor cores are 2× FP16).
```

### Roofline Model

```
Peak Compute:         122.6 TFLOPS (FP16/BF16/INT8)
Peak Bandwidth:       864 GB/s (theoretical), ~750 GB/s (practical)

Arithmetic Intensity crossover:
  Theoretical:  122.6 TFLOPS / 864 GB/s = 142 FLOPS/byte
  Practical:    122.6 TFLOPS / 750 GB/s = 163 FLOPS/byte

For LLM decode (c=1):
  Model weights dominate memory traffic.
  Arithmetic intensity ≈ 2 × batch_size FLOPS per byte loaded.
  At c=1: AI = 2 FLOPS/byte → deeply memory-bound (0.014× compute ceiling)
  At c=8: AI = 16 FLOPS/byte → still memory-bound (0.11× compute ceiling)
  At c=64: AI = 128 FLOPS/byte → approaching crossover
  At c=142+: compute-bound (theoretical)

Implication for spec decode:
  Dense model verification (B=8 tokens, same weights loaded once):
    AI ≈ 2×8 = 16 FLOPS/byte → still memory-bound
    → verification amortizes toward one AR step, not B independent steps
  MoE verification (B=8 tokens, different experts per token):
    Each expert activated for 1-2 tokens → AI ≈ 2-4 FLOPS/byte per expert
    → Each expert call is memory-bound independently
    → Verification costs scale with total expert activations, not constant
```

### RDNA3 Architectural Constraints for Inference

**vs NVIDIA (where most spec decode literature comes from):**

| Property | RDNA3 (W7900) | NVIDIA A100/H100 | Impact on Inference |
|---|---|---|---|
| INT8 vs FP16 speedup | 1× (same throughput) | 2× (tensor core) | INT8 helps only via bandwidth, not compute |
| Memory type | GDDR6 (864 GB/s) | HBM2e/3 (2-3.3 TB/s) | 3-4× less bandwidth → more bandwidth-bound |
| L2 cache | 6 MB | 40-50 MB | Less L2 reuse for expert weights |
| Infinity Cache | 96 MB | N/A (HBM is fast enough) | Helps if working set fits; experts don't |
| Tensor core equivalent | WMMA 16×16×16 | MMA (various shapes) | Less flexible shapes for small GEMMs |
| Warp size | 32 (native) or 64 (CU mode) | 32 (fixed) | CU mode needed for some HIP kernels |
| Kernel launch overhead | Higher (ROCm dispatch) | Lower (CUDA optimized) | Matters more for many-small-kernel MoE |

**Key RDNA3 limitations for this workload:**

1. **No INT8 compute advantage**: Unlike NVIDIA where INT8 tensor cores are 2× FP16,
   RDNA3 WMMA INT8 = FP16 throughput. W8A8 quantization helps only by halving weight
   bytes (memory-bound benefit), not by doubling compute.

2. **WMMA needs enough rows to amortize tile overhead**: The 16×16×16 WMMA tile
   naturally wants M≥16. For c=1 decode (M=1), vector GEMV is usually the right
   primitive. For verification of B=8 tokens through MoE where each expert sees
   0-2 tokens, WMMA is unlikely to win without padding/grouping that proves out
   end-to-end. Prefill or high-batch serving is the natural WMMA regime.

3. **Kernel launch overhead penalizes MoE**: With 256 experts, top-8 routing, and
   40 layers: a naive implementation would need ~320 expert kernel launches per
   verification step. Even with fused kernels, the dispatch overhead on ROCm is
   non-trivial. Our fused `hip_w8a16_selected_experts_shared_gate_up_down_t_out`
   collapses this to 1 launch per layer per token, but that's still 40×8 = 320
   fused kernel launches for B=8 verification.

4. **L2 cache too small for expert weight reuse**: Each expert has gate_up
   (2048×512×2 = 2MB) + down (512×2048 = 1MB) = 3MB per expert at INT8. L2 is 6MB.
   So at most 2 experts' weights can be L2-resident simultaneously. With top-8
   routing and 256 experts, there is essentially zero L2 reuse across tokens
   unless they happen to route to the same expert.

5. **Infinity Cache (96MB) could help but mostly doesn't for MoE**: 96MB fits
   ~32 experts' weights. With 256 experts, cache hit rate for random routing is
   ~12.5%. The shared expert (always active) benefits, but selected experts mostly
   miss. Dense targets get better temporal/tile locality than sparse expert
   dispatch, but entire 27B-31B dense layers are hundreds of MB and do not fit in
   Infinity Cache.

### Measured Performance Context

```
From this project's measurements (PROFILE_NOTES.md, WORKLOG.md):

Qwen3.6-35B-A3B W8A8 c=1 decode attribution:
  MoE total:              67.3% of decode time
  Linear attention:       19.8%
  Full attention:          6.8%
  Other (norms, embed):    6.1%

Retained decode speeds on this hardware:
  mini-sglang (early):    8.46 tok/s (118 ms/token, unoptimized)
  nano-vllm native W8A8:  115-120 tok/s (8.3-8.7 ms/token, optimized)
  nano-vllm PARO W4A16:   ~140 tok/s (proof-of-life, packed-8 GEMV)
  llama.cpp Q8_K_XL:      ~105-110 tok/s (reference)

Peak measured memory bandwidth (inference workloads):
  Effective:  ~700-750 GB/s (81-87% of 864 GB/s theoretical)
  
This puts the W7900 in a regime where:
  - c=1 decode is deeply memory-bandwidth-bound for dense models
  - MoE decode is a mix of bandwidth-bound (expert weights) and
    launch-overhead-bound (many small kernel dispatches)
  - Prefill is compute-bound for large batches (WMMA helps)
  - Speculation verification on dense models should amortize well
    (same weights, 8× more compute, still bandwidth-bound)
  - Speculation verification on MoE is expensive (different experts
    per token, no weight reuse, launch overhead)
```

---

## Model Architecture: Qwen3.6-35B-A3B

```
Total parameters:     ~35B
Active parameters:    ~3B per token
Layers:               40 (30 linear attention + 10 full attention, every 4th is full)
Hidden size:          2048
Experts:              256 per MoE layer
Top-K routing:        8 experts selected per token
Expert intermediate:  512 (tiny per expert)
Shared expert:        512 intermediate (always active)
Attention heads:      16 (head_dim=256)
KV heads:             2 (GQA)
Vocab size:           248,320
Linear attention:     conv kernel dim 4, 16 key heads (128 dim), 32 value heads (128 dim)
Quantization:         W8A8 INT8 (Quark)
MTP module:           1 layer (full attention + 256-expert MoE + shared lm_head)
```

**Key architectural properties for speculation:**
- Linear attention layers have **recurrent state** — must advance sequentially per token
- Full attention layers need **KV cache writes** — position-dependent, sequential
- MoE with 256 experts and top-8: for B=8 tokens, you get 64 expert activations
  spread across 256 experts = **average 0.25 tokens per expert = nothing to batch**

---

## Measured Performance (W7900/gfx1100, ROCm 7.13, PyTorch 2.11)

### AR Baseline
```
Qwen3.6-35B-A3B W8A8:  115-120 tok/s decode (8.3-8.7 ms/token)
```

### DFlash State-Block (B=8, 6 prompts, decode-128)
```
DFlash:                 81.75 tok/s (0.71× AR)
AR same-session:        114.97 tok/s
Verification time:      7.803s for 1219 token-positions (155 forward passes)
Per-row cost:           6.40 ms/row
Per-forward-pass cost:  50.3 ms (8 positions per pass)
Draft time:             1.370s total
Accept/host overhead:   0.051s
Acceptance histogram:   avg 4.0-6.7 tokens/step across prompts
```

### MTP B5 Interleaved Native-Loop
```
MTP:                    82.2 tok/s (0.685× AR)
AR same-session:        120.0 tok/s
Target verify calls:    12 iterations for 32 output tokens
Target verify tokens:   31 for 32 committed (nearly 1:1!)
Acceptance:             depth-1: 64%, depth-2: 71%, depth-3: 60%, depth-4: 100%, depth-5: 67%
Avg output/iteration:   2.91 tokens
```

### MTP with Graph Replay
```
MTP graph-replay:       83.9 tok/s (0.70× AR)
AR same-session:        120.0 tok/s
```

### Profile Breakdown (DFlash State-Block Verifier)
```
Total verify:           7.903s
  model_block:          6.996s (88.5%)
    MoE selected/shared:  2297ms (32.8% of model block, per truncated aggregate)
    router/top-k:          994ms (14.2%)
    visible MoE buckets:  3291ms (47.0% of model block)
  lm_head:              0.851s (10.8%)
  state_install:        0.071s (0.9%)
```

The verifier profile above is an orientation profile, not a complete MoE
accounting table. The `selected/shared` and `router/top-k` buckets are the
visible named buckets from a truncated aggregate; do not derive an exact MoE
floor from only those two lines.

---

## Why Speculation Fails on This Architecture

### The Core Assumption of Speculative Decoding

Standard spec decode economics for a **dense** model:
```
AR decode: Load all weights → apply to 1 token → memory-bandwidth bound
Verify B:  Load all weights → apply to B tokens → SAME bandwidth, ~same time!

Result: Verify(B) ≈ 1× AR step regardless of B
```

### What Actually Happens on 256-Expert MoE

The verifier code (`_run_layer_batched`) processes MoE **sequentially per token**:

```python
# For each of B=8 positions:
for row in range(tokens):
    moe_in, residual_out = hip_qwen35_add_rmsnorm(...)
    moe_rows.append(_moe_native(moe_in, layer, args, profile))
```

Each `_moe_native()` call:
1. Router: 2048 → 256 logits → top-8 selection
2. 8 selected experts: gate_up (2048→512) + silu + down (512→2048) each
3. Shared expert: same structure
4. Weighted combine

For B=8 across 40 layers: **320 sequential MoE kernel launches per verification step.**

Only QKV and output projections are batched (the cheap parts). MoE — which is
2/3 of the cost — runs as B separate single-token dispatches.

### Why You Can't Batch MoE Over Verification Tokens (Current Dispatch)

With 256 experts and B=8 tokens × top-8 routing:
- 64 total expert activations spread across 256 experts
- Under uniform routing: expected ~57 unique experts (average 0.25 tokens per expert)
- **With current sequential dispatch, there is nothing to batch within an expert!**

However, if routing is heavy-tailed or temporally correlated (adjacent tokens
route to similar experts), the actual unique expert union could be smaller.
This would make **grouped expert dispatch** viable: batch all tokens routed to
the same expert into one GEMM call, improving L2/Infinity Cache reuse.

The previous batched MoE attempt was rejected for 1e-8 to 1e-7 hidden-state
drift, but this strictness may have been excessive — see "Next Steps: Diagnostic 3"
for the relaxed correctness criterion.

Key measurement needed: actual unique expert union per layer for B=2,4,8.

### Measured Verification Scaling

```
Dense model (theoretical): Verify(8) ≈ 1.0-1.2× AR_step    → O(1) in B
This MoE (measured):       Verify(8) = 50.3ms = 5.8× AR_step → O(B) in B

Efficiency ratio: 50.3ms / (8 × 8.7ms) = 0.72
  → Verification costs 72% of B separate AR steps
  → Only 28% savings from projection batching
  → vs dense where you'd expect ~80-87% savings
```

### η Decomposition: What the 0.736 Means

The measured η = 0.736 combines multiple effects:

```
From profile (model_block = 6.996s for 1219 token-positions):
  Visible MoE selected/shared + router buckets: 3.291s (47.0% of model_block)
  Other model-block work: norms, projections, attention, GDN, state updates,
                          residuals, and smaller unbucketed helpers

Measured total:
  η_total = Verify(8) / (8 × AR_cost) = 50.3ms / (8 × 8.7ms) ≈ 0.72-0.74

Interpretation:
  - The verifier is far from the dense regime (η≈0.13-0.20).
  - MoE/router is the largest visible bucket and runs per token in the current path.
  - Linear attention recurrence is a second real source of B-scaling.
  - The exact split needs a non-truncated verifier profile before we assign
    hard η floors to MoE vs attention.
```

This still clarifies where effort should go:
- Grouped/budgeted MoE dispatch is the main possible way to reduce the largest
  visible bucket.
- Parallel-scan linear attention could reduce recurrence span, but it will not
  fix the sparse expert dispatch by itself.
- Claims like "η_MoE has a hard floor of 0.49" require a complete verifier
  profile and should not be treated as measured fact yet.

### Why Linear Attention Makes It Worse

Even if MoE could somehow be batched, the 30 linear attention layers have
**sequential state dependencies** in our current implementation:

```
state[t] = decay * state[t-1] + input[t] ⊗ value[t]
```

Our verifier advances this recurrence token-by-token. In principle, this is a
linear recurrence that could be parallelized via Blelloch parallel scan (as
Mamba prefill does), reducing span from O(B) to O(log B). However:
- Our implementation processes sequentially for bit-exactness with c=1 AR
- A parallel-scan verifier would need to prove equivalent numerics
- Even with parallel scan, MoE/router still remains the largest visible blocker
  unless grouped dispatch or an approximate prefilter also helps

The linear-attention contribution is meaningful, but current evidence says it is
not the first blocker to chase before MoE/router dispatch.

---

## The Mathematical Framework

### Speculative Decode Break-Even Condition

For any speculation scheme to beat AR:

```
committed_tokens_per_step / step_cost > 1 / AR_cost

Where:
  step_cost = draft_cost + verify_cost(B)
  committed_tokens_per_step = f(acceptance_rate, B)
```

### Key Ratio: Verification Efficiency

Define **verification efficiency** `η`:
```
η = Verify(B) / (B × AR_cost)

Dense model:  η ≈ 1/B  (ideal: verify B tokens for ~1 AR step)
This MoE:     η ≈ 0.72  (verify B tokens costs ~0.72B AR steps)
```

### Break-Even Formula

```
Speedup = committed_per_step × AR_cost / (draft_cost + B × η × AR_cost)

For speedup > 1.0:
  committed_per_step > (draft_cost / AR_cost) + B × η
```

### Applied to This MoE (Qwen3.6-35B-A3B, W8A8, W7900)

```
AR_cost = 8.7 ms
draft_cost (DFlash) = 10 ms ÷ 1 step = 10 ms
draft_cost (MTP) = 2 ms
B = 8
η = 0.72

DFlash break-even:
  committed > (10/8.7) + 8 × 0.72 = 1.15 + 5.76 = 6.91 tokens/step
  → Need avg 6.91 accepted tokens out of 8 → ~86% acceptance across all positions
  → Best measured prompt: avg 6.7 → STILL BELOW BREAK-EVEN

MTP break-even (B=5):
  committed > (2/8.7) + 5 × 0.72 = 0.23 + 3.60 = 3.83 tokens/step
  → Need avg 3.83 accepted out of 5 → ~77% acceptance across all positions
  → Measured avg: 2.91 → BELOW BREAK-EVEN
```

**The break-even threshold is essentially unreachable** because acceptance rate
naturally decays with position depth, and η ≈ 0.72 means verification costs
grow almost linearly with B.

### What η Would Need to Be

For DFlash B=8 with avg 4.0 committed tokens to break even:
```
4.0 > (10/8.7) + 8 × η
4.0 > 1.15 + 8η
η < 0.356

Need: Verify(8) < 0.356 × 8 × 8.7ms = 24.8ms
Currently: Verify(8) = 50.3ms
Required speedup: 2.0× on the verification path
```

For η ≈ 0.36, you'd need per-row verification cost of ~3.1 ms (vs current 6.4 ms).
This matches the earlier back-of-envelope estimate that sub-4ms/row is required.

---

## Local Target Comparison: Qwen3.6-27B vs Gemma4

The most important split is not "Qwen vs Gemma"; it is dense/dense-like
verification vs high-cardinality MoE verification.

### Local Artifacts and Architecture

| Target | Local artifact | Architecture | Spec-dec implication |
|---|---:|---|---|
| Qwen3.6-27B W8A8 | 34G | 64 layers, hidden 5120, dense MLP, 48 linear-attention + 16 full-attention layers | No MoE dispatch tax, but linear-attention state may keep η above pure dense |
| Gemma4-31B W8A8 | 32G | 60 layers, hidden 5376, dense MLP, 50 sliding-attention + 10 full-attention layers | Best dense target shape; no MoE and no linear-attention recurrence |
| Gemma4-E4B W8A8 | 12G | 42 layers, hidden 2560, dense + per-layer embedding state, 35 sliding + 7 full | Promising once PLE/runtime support is solid; not first math target |
| Gemma4-26B-A4B W8A8 | 26G | 30 layers, hidden 2816, 128 experts, top-8, 25 sliding + 5 full | Better than Qwen's 256 experts, but still sparse MoE with weak B=8 batching |
| Qwen3.6-35B-A3B W8A8 | 35B total / 3B active | 40 layers, 256 experts, top-8, 30 linear + 10 full | Measured losing case: η≈0.736 |

### Same Math, Local Targets

Assumptions for projections:
- `B=8`
- draft cost = `4ms/step` for dense-target projections
- Qwen3.6-35B-A3B measured row uses its measured DFlash draft cost (~10ms/step)
- low acceptance `A_out=4.0`, aggregate acceptance `A_out=5.2`
- projected rows need direct measurement before being promoted to retained numbers

```
Target / scenario                  AR tok/s  η(B=8)  Verify8  Step    A=4 tok/s   A=5.2 tok/s  Break-even A
────────────────────────────────────────────────────────────────────────────────────────────────────────────
Qwen3.6-27B dense-like projection      30      0.17    45ms    49ms     81  2.70×   105  3.51×      1.48
Qwen3.6-27B hybrid-mid projection      30      0.25    67ms    71ms     57  1.89×    74  2.45×      2.12
Qwen3.6-27B hybrid-high projection     30      0.35    93ms    97ms     41  1.37×    53  1.78×      2.92

Gemma4-31B dense, GGUF-ref AR          20.8    0.17    65ms    69ms     58  2.77×    75  3.60×      1.44
Gemma4-31B dense, W8A8 AR estimate     30      0.17    45ms    49ms     81  2.70×   105  3.51×      1.48

Gemma4-26B-A4B MoE mid projection      70      0.55    63ms    67ms     60  0.85×    78  1.11×      4.68
Gemma4-26B-A4B MoE fast projection     90      0.55    49ms    53ms     76  0.84×    98  1.09×      4.76

Qwen3.6-35B-A3B measured DFlash       115      0.736   51ms    61ms     65  0.57×    85  0.74×      7.04
```

Interpretation:
- Gemma4-31B dense is the cleanest speculation target if the W8A8 runtime path
  loads and verifies efficiently.
- Qwen3.6-27B should still be far better than Qwen3.6-35B-A3B, but it is a
  hybrid linear-attention model, so its actual η could land around 0.25-0.35
  rather than the pure dense 0.17 planning value.
- Gemma4-26B-A4B is a useful MoE comparison: 128 experts/top-8 is less hostile
  than 256 experts/top-8, but B=8 still spreads 64 expert activations across
  many experts. It likely needs high acceptance or grouped expert dispatch to be
  more than marginal at c=1.
- The first measurement to run on any new target is still `η = Verify(8) /
  (8 × AR_cost)`. One direct η measurement is more valuable than another
  projected speed table.

---

## Applied to Dense Models: Where Speculation Wins

### Dense 27B-31B Model Characteristics (Projected, W7900)

```
Parameters:         27-31B (all dense, no MoE)
Weight size (BF16): ~54-62 GB (doesn't fit VRAM, needs quantization)
Weight size (W4):   ~14-16 GB (fits in 48GB VRAM)
Weight size (W8):   ~27-31 GB (fits in 48GB VRAM)

W7900 memory bandwidth: ~864 GB/s (theoretical), ~700-750 GB/s (effective)

AR decode speed (W8, 31B):
  Weight load: 31GB / 750 GB/s ≈ 41 ms/token → ~24 tok/s
  (Real-world with overhead: 25-35 tok/s)

AR decode speed (W4, 27B):
  Weight load: 14GB / 750 GB/s ≈ 19 ms/token → ~53 tok/s  
  (Real-world with overhead: 35-45 tok/s)
```

### Why Dense Models Have η ≈ 1/B

During decode on a dense model:
- **All computation is a single large GEMM/GEMV per layer**
- Weight loading dominates (memory-bound for c=1)
- Processing 1 token vs 8 tokens: same weights loaded, 8× more compute
- But compute is cheap (model is bandwidth-bound, not compute-bound)
- Net: Verify(8) ≈ 1.0-1.6× AR_step (literature: EAGLE, SpecInfer, Medusa)

```
Dense η ≈ (1.0 to 1.6) / B ≈ 0.13 to 0.20 for B=8

Note: On W7900 with ROCm 7.13 and W8A8 quantization, expect toward the
higher end (0.15-0.20) due to INT8 throughput being bursty and not fully
hiding compute at B=8 for large models. Plan for η ≈ 0.17 as a conservative
practical estimate.

Qwen3.6-27B is dense in the MLP sense but not a pure dense transformer: 48/64
layers are linear-attention layers. Its MoE tax disappears, but exact verifier
state recurrence can push η above this dense-transformer planning range.
```

### Dense Model Break-Even (Much Easier!)

For a dense 31B W8 model at 30 tok/s (AR_cost = 33ms), using conservative η=0.17:
```
draft_cost ≈ 3-5ms (small draft model or MTP)
B = 8
η ≈ 0.17 (conservative for W7900 ROCm)

Break-even:
  committed > (5/33) + 8 × 0.17 = 0.15 + 1.36 = 1.51 tokens/step
  → Only need 1.5 committed tokens per step!
  → Even 55% position-1 acceptance is enough!
```

### Expected Dense Model Speedup

With avg 4.0 committed tokens/step (conservative, our "low" acceptance):
```
Conservative (η=0.17):
  Speedup = 4.0 × 33ms / (5ms + 8 × 0.17 × 33ms) = 132 / 50 = 2.64×

Optimistic (η=0.14):
  Speedup = 4.0 × 33ms / (5ms + 8 × 0.14 × 33ms) = 132 / 42 = 3.14×

Dense 31B W8:  30 tok/s AR → 79-94 tok/s with low acceptance (2.6-3.1×)
```

With avg 5.2 committed tokens (our aggregate acceptance):
```
Conservative: 5.2 × 33 / 50 = 3.43× → 103 tok/s
Optimistic:   5.2 × 33 / 42 = 4.09× → 123 tok/s

Expect: 30 tok/s AR → 100-130 tok/s (≈3-4× speedup, plan for 3×)
```

These would be **transformative** if direct η measurement lands near the dense
planning range. The same verification infrastructure that fails on high-expert
MoE should produce much better economics on dense targets. Even a 2.5× speedup
on a 30 tok/s model gives 75 tok/s, which is more impactful than marginal
Qwen3.6-35B-A3B spec-dec wins.

---

## Decision Framework: When Does Speculation Help?

### The η Threshold

```
Speculation is viable when:  η < committed_avg / B - draft_cost / (B × AR_cost)

Simplified (assuming cheap drafts):
  η < committed_avg / B

With typical acceptance (avg 4 of 8):
  η < 4/8 = 0.50 → need verification at most 50% of B×AR cost

With good acceptance (avg 6 of 8):
  η < 6/8 = 0.75 → need verification at most 75% of B×AR cost
  → This MoE at η=0.72 is RIGHT AT the boundary with perfect acceptance
```

### η By Architecture Type

| Architecture | η (B=8) | Speculation Viable? | Expected Speedup | Notes |
|---|---|---|---|---|
| Dense 7B (W4) | 0.13-0.15 | ✅ Strongly | 3-5× | Fully memory-bound |
| Dense 27B (W8) | 0.15-0.20 | ✅ Strongly | 2.5-4× | Approaching compute-bound at B=8 |
| Dense + linear-attention hybrid 27B | 0.20-0.35 | ✅ Likely | 1.5-3× | Qwen3.6-27B planning range; measure η |
| Dense 70B (W4) | 0.13-0.15 | ✅ Strongly | 3-5× | Large model, very bandwidth-bound |
| MoE 8×7B (8 experts, top-2) | 0.25-0.35 | ✅ Marginal to good | 1.5-2.5× | Few experts, some batching possible |
| Gemma4-26B-A4B MoE (128 experts top-8) | 0.45-0.60 | ⚠️ Marginal | 0.8-1.3× | Better than 256 experts, still sparse |
| MoE 47B-A14B (DeepSeek-like, 64 experts top-6) | 0.45-0.60 | ⚠️ Marginal | 1.0-1.5× | Many experts, limited batching |
| **Qwen3.6-35B-A3B (256 experts top-8 + linear attn)** | **0.72** | **❌ Not viable** | **0.7-0.85×** | Sequential MoE + recurrence |
| Pure SSM/Mamba (current impl) | 0.7-0.9 | ❌ Not viable (impl-bound) | N/A | Parallel scan could reduce to 0.3-0.5 |

**Important caveat:** η values above reflect current implementations, not theoretical
floors. Parallel scan (for SSM/linear attn) and grouped-expert kernels (for large MoE)
could reduce η in future work. Do not assign a hard MoE η floor without a complete
non-truncated verifier profile.

References for dense η ≈ 0.13-0.20: EAGLE (Li et al., 2024), SpecInfer (Miao et al., 2024),
Medusa (Cai et al., 2024) all report verification cost approximately equal to one AR step
on dense models. Exact values depend on hardware, model size, and quantization.

### Key Factors That Determine η

1. **Expert count / routing diversity**: More experts → less batching benefit
2. **Top-K**: Higher K → more expert activations per token → can't share across batch
3. **Sequential state layers**: Linear attention, Mamba, RWKV → forces O(B) processing
4. **Model size vs bandwidth**: Larger dense models are MORE bandwidth-bound → better η
5. **Quantization level**: More compressed = faster weight load = less compute-bound = better η

### The MoE Paradox

MoE models are **already fast at AR** because they only activate 3B of 35B parameters
per token. This means:
- AR_cost is already low (8.7ms vs ~33ms for a dense 31B)
- The "room to improve" via speculation is small in absolute terms
- AND verification is expensive (sequential dispatch, can't batch over sparse routing)

**MoE gives you speed through sparsity. Speculation gives you speed through parallelism.
These two approaches are fundamentally in tension when expert count is high.**

### Tree-Structured Drafts (EAGLE-2/3, DDTree)

Tree-structured verification (EAGLE-2, EAGLE-3, our DDTree implementation) verifies
tree-shaped candidate sets (effectively B=32-64 candidates) with higher per-node
acceptance than linear chains. On dense models, this pushes speedups significantly
higher (5-7× reported in EAGLE-3).

However, the same MoE dispatch tax applies: each tree node still requires sequential
MoE evaluation. With B_effective=32 at η=0.74, the verification step would cost
32×6.4ms = 205ms — far worse than the linear-chain B=8 case. Tree drafts amplify
the dense-model advantage but equally amplify the MoE penalty.

For our dense model projections, tree-structured drafts could push the 3-4× speedups
even higher, but we have not measured this.

---

## Our Verification Infrastructure Applied to Dense Models

The verification code we built (state-block verifier, acceptance tracking,
exactness proofs, adaptive policies) would work dramatically better on dense
models because:

1. **`_run_layer_batched` projections already batch** — for dense models, this IS
   the whole computation. MoE sequential loop disappears entirely.

2. **No recurrent state in pure dense transformers** — pure full/sliding-attention
   transformers have no sequential state dependency beyond KV cache. Dense hybrids
   like Qwen3.6-27B still need a real η measurement because linear-attention state
   can reintroduce sequential verifier work.

3. **Verification = mini-prefill** — standard spec decode verification on a
   dense transformer is literally a short prefill of B tokens. On W7900 with
   a 31B model, prefilling 8 tokens costs about the same as decoding 1 token.

4. **DFlash draft model is already dense** — the `z-lab/Qwen3.6-35B-A3B-DFlash`
   draft model (8-layer dense Qwen3, 948MB) was designed for this. On a dense
   target, both draft and verify are efficient.

### What Dense Model Targets Make Sense on W7900?

| Model | AR (est.) | With Spec (conservative) | Speedup | Notes |
|---|---|---|---|---|
| Qwen3-32B W8A8 | ~28-32 tok/s | ~80-100 tok/s | 2.8-3.2× | Fits VRAM, best target |
| Qwen3-30B-A3B W8 | ~115 tok/s | N/A (η=0.74) | N/A | MoE, spec doesn't help |
| Mistral-Small-3.2-24B W8 | ~35-40 tok/s | ~95-115 tok/s | 2.7-3.0× | Dense, fits comfortably |
| Gemma-3-27B W8 | ~30-35 tok/s | ~85-105 tok/s | 2.8-3.0× | Dense, good candidate |
| Qwen3-32B W4 (GPTQ/AWQ) | ~40-50 tok/s | ~100-130 tok/s | 2.5-2.8× | Compressed, fast with spec |
| Llama-3-70B W4 | ~20-25 tok/s | ~60-80 tok/s | 3.0-3.3× | Tight on VRAM, high leverage |

**Planning estimate: expect 2.5-3.5× speedup on dense models. Plan for 3×.**
The exact value needs measurement — W7900 INT8 throughput is bursty under
ROCm 7.13, and η for dense W8A8 may land at 0.15-0.20 rather than the
literature-ideal 0.13.

---

## The Mathematical Cutoff: When to Use vs Skip Speculation

### Quick Decision Rule

```
Measure one number:  η = Verify(B) / (B × AR_cost)

  η < 0.20  →  Always speculate. Dense-model regime. Expect 2.5-4× speedup.
  η 0.20-0.40  →  Probably speculate. Expect 1.5-2.5× speedup.
  η 0.40-0.60  →  Maybe speculate. Needs high acceptance. Marginal gains.
  η 0.60-0.80  →  Probably don't. Only wins with exceptional acceptance (>85% all positions).
  η > 0.80  →  Don't speculate. Verification is more expensive than just running AR.

Our MoE: η = 0.736 → "Probably don't" ← THIS IS WHERE WE ARE

NOTE: η values reflect current implementation, not structural floors.
Parallel scan, grouped-expert kernels, or fused dispatch could reduce η.
But for 256-expert MoE, the current visible MoE/router buckets already keep
the path far from the dense "η < 0.20" regime unless grouped/budgeted expert
dispatch changes the economics.
```

### How to Measure η on Any New Model

```python
# Step 1: Measure AR baseline
AR_cost = time_one_decode_token(model)  # e.g. 33ms for dense 32B

# Step 2: Measure batched verification of B tokens
Verify_cost = time_verify_B_tokens(model, B=8)  # e.g. 36ms for dense, 51ms for MoE

# Step 3: Compute η
η = Verify_cost / (B × AR_cost)  # e.g. 36/(8×33) = 0.136 for dense
```

### η-Speedup Curves (with measured acceptance A_out=5.2, MTP drafts)

```
η      Step Cost (B=8)   tok/s    vs AR    Category
0.10     9.0 ms          578      5.0×     Dense ideal (impossible on GPU, theoretical)
0.13    11.1 ms          468      4.1×     Dense 14B (measured in literature)
0.14    11.7 ms          444      3.9×     Dense 27-32B on W7900 (projected)
0.20    16.1 ms          323      2.8×     Small MoE (8 experts, top-2)
0.30    22.9 ms          227      2.0×     Medium MoE (16-64 experts)
0.40    29.8 ms          174      1.5×     Large MoE (64 experts, top-6)
0.50    36.6 ms          142      1.2×     Very large MoE (marginal zone)
0.60    43.4 ms          120      1.04×    Break-even boundary
0.70    50.2 ms          104      0.90×    Sub-AR territory (losing money)
0.736   53.2 ms           98      0.85×    ← OUR MEASURED POSITION
0.80    57.0 ms           91      0.79×    Deep loss
1.00    71.6 ms           73      0.63×    Sequential verification (no batching)
```

### The Dense Model Opportunity

For a dense 27B or 32B model on W7900:
- AR: ~30-40 tok/s (limited by memory bandwidth loading 27-32GB of weights per token)
- Verification: loads same weights once, applies to 8 tokens → ~1.2-1.6× AR cost
- **The slower the AR, the MORE speculation helps** (because draft overhead is fixed)
- Conservative estimate with our acceptance: 30 tok/s → 90-105 tok/s (3-3.5×)
- **This would be a flagship result for W7900 inference**
- Tree-structured drafts (EAGLE-2/3) could push even higher on dense targets

---

## Options for the MoE Model (If We Still Want to Try)

### Option 1: Make AR Faster (Already Working — PARO)
- Packed-8 AWQ GEMV: halved the dominant kernel bucket
- Expert kernel fusion: gate_up + silu + down in one launch
- Directly speeds up AR without speculation overhead
- **Best ROI for this architecture**

### Option 2: Reduce Per-Row Verification Cost Below 4ms
- Need 2× improvement on MoE dispatch per token
- Possible paths: fused expert kernel, WMMA-based expert matmul, expert weight prefetching
- Would bring η from 0.72 to ~0.36, making B=4-5 marginally viable
- Very high engineering effort for marginal gain

### Option 3: Reduce B with Adaptive Policy  
- Current B=8 always pays 8 rows. With η=0.72, even full acceptance barely wins.
- B=2-3 with high acceptance might marginally beat AR:
  ```
  B=2, η=0.72, committed=1.8 (90% pos-1 acceptance):
  Speedup = 1.8 × 8.7ms / (2ms + 2 × 0.72 × 8.7ms) = 15.7 / 14.5 = 1.08×
  ```
- Tiny win (~8%), high complexity, fragile to acceptance variance

### Option 4: Approximate/Lossy Verification
- Skip some layers for verification (run 20/40 layers)
- Use "mismatch = reject" safety rule
- Could halve verification cost → η ≈ 0.36
- But: introduces false rejections, reduces effective acceptance rate
- Net benefit unclear without measurement

### Option 5: Accept This Is Not a Speculation Model
- Focus engineering effort on AR optimizations (PARO kernels, fused MoE, memory)
- Use speculation infrastructure for correctness/acceptance research only
- Port speculation to dense model targets where it provides 3× wins
- **Recommended path**

---

## Next Steps: Kill Criteria Before Fully Archiving

Before fully pausing spec-dec on this model, three cheap diagnostics (~1 day total)
would either confirm the kill or reveal a narrow viable path:

### Diagnostic 1: Measure Verify(B) Directly for B=1,2,3,4

**Why**: Our η=0.736 is derived from B=8 batches where projection amortization
is maximal. Extrapolating to B=2 is unfair in both directions — it credits small
B with savings it doesn't get, but it also doesn't account for reduced MoE calls.

**Method**: Run state-block verifier with `--dflash-block-size` set to 2, 3, 4
(and 1 as degenerate baseline) on the same 6-prompt suite. Record actual ms/step
and compute real η(B) for each.

**Kill criterion**: If η(B=2) > 0.80 (worse than B=8 extrapolation), the B=2
path from Table 3 is dead. If η(B=2) < 0.60, there may be a real 10-15% win.

### Diagnostic 2: Log Unique Expert Union per Layer

**Why**: Our "nothing to batch" argument assumes near-uniform routing. If routing
is heavy-tailed or temporally correlated, the actual expert union for B=8 could
be much smaller than the theoretical ~57 unique experts. MoE-Spec (arXiv 2602.16052)
and Cohere (2026) both point to routing concentration as a real effect.

**Method**: During one DFlash B=8 verification run, log per layer:
```python
unique_experts = len(set(selected_experts.flatten().tolist()))
ratio = unique_experts / 8  # vs single-token baseline
```

**Kill criterion**: If ratio ≈ 6-7× (near theoretical maximum), grouped MoE
has no headroom. If ratio ≈ 2-4×, there's real weight-reuse opportunity.

### Diagnostic 3: Relax Hidden-State Bit-Exactness, Not Accepted-Token Correctness

**Why**: We previously built and rejected batched MoE because it introduced
1e-8 to 1e-7 hidden-state drift vs c=1 AR. But the correct test for a
spec-dec verifier is accepted-token behavior, not hidden-state byte equality.
Production inference routinely has batch-shape-dependent numeric differences,
but speculative decoding still must not commit tokens that exact AR would reject.

Important correctness distinction:
- **Safe prefilter mode**: grouped/approx MoE may reject early. If it says
  "match", run exact verification before committing. False rejects only lose
  speed; false accepts are caught by the exact verifier.
- **Final verifier mode**: grouped/approx MoE directly commits matching drafts.
  This is only exact-AR-safe if every would-be accept also matches exact AR.
  A generic top-1 mismatch rate is not enough, because the dangerous case is
  `approx_top1 == draft_token` while `exact_top1 != draft_token`.

**Method**: Re-enable the batched/grouped MoE path. Instead of asserting
hidden-state equality, measure against exact c=1 AR:
```
max |Δ logit|
top-1 mismatch rate
false accept rate: approx_top1 == draft and exact_top1 != draft
false reject rate: approx_top1 != draft and exact_top1 == draft
accepted-token mismatch rate after the intended commit policy
```

**Kill criterion**:
- If final verifier mode has any false accepts on the evaluation suite, do not
  use it as the exact verifier.
- If top-1 mismatch rate is high (>1%), the grouped path is probably too lossy
  even as an aggressive approximation.
- If false accepts are 0 and top-1 mismatch rate is <0.1% on a large sample,
  grouped MoE may be usable as an exact-enough verifier candidate; still keep a
  final exactness guard until the sample is large enough to trust.
- If false accepts occur but false rejects dominate, keep grouped MoE only as a
  prefilter and measure whether it actually saves exact rows.

**Impact if grouped MoE works**: Grouped dispatch could process all 8
tokens' experts in fewer kernel launches with better L2/Infinity Cache reuse.
If it is safe as a final verifier, even a 30-40% reduction in per-row MoE cost
could bring η down materially. If it is only safe as a prefilter, the win depends
on how many rejected positions it avoids before exact verification.

### Decision Gate

```
IF η(B=2) < 0.60 AND (
    expert_union_ratio < 4
    OR grouped_moe_false_accept_rate == 0 on a large exact-AR comparison
    OR grouped_moe_prefilter demonstrably saves >=25% exact rows with 0 accepted-token mismatches
):
  → Pursue B=2-3 MTP with grouped/relaxed MoE verifier
  → Target: 1.15-1.25× AR (130-145 tok/s)

ELSE:
  → Archive DFlash/MTP speculation on this model
  → Focus PARO/AR kernel work (already delivering gains)
  → Port spec-dec infrastructure to dense 27-32B targets
```

### References

- MoE-Spec: Expert Budgeting for Efficient Speculative Decoding (arXiv 2602.16052, 2026)
- Cohere: "Why MoE models get more from speculative decoding" (2026 blog)
- Qwen3.6-35B-A3B model card recommends `num_speculative_tokens: 2` for MTP in vLLM
- Official SGLang Qwen3.6 guidance also uses small NEXTN-style setup (B≤2)

---

## Appendix: Architecture Comparison

### Why Qwen3.6-35B-A3B Is the Worst Case for Speculation

| Property | Impact on Speculation |
|---|---|
| 256 experts | Maximizes routing diversity → minimizes batching benefit |
| Top-8 routing | 8 experts per token → 64 activations for B=8, but spread thinly |
| 512 intermediate size | Tiny expert GEMVs → launch-overhead-dominated, not compute-bound |
| Linear attention (30/40 layers) | Sequential state dependency → forces O(B) processing for 75% of layers |
| Already 3B active / 35B total | AR is already fast → small absolute room for improvement |
| W8A8 quantization | Fast weight loads → AR already optimized → harder to beat |

### The DFlash Draft Model (z-lab/Qwen3.6-35B-A3B-DFlash)

```
Type:               Dense transformer (Qwen3 architecture)
Layers:             8 (all full attention, no linear attention, no MoE)
Hidden size:        2048
Attention heads:    32 (head_dim=128)
KV heads:           4
Parameters:         ~948MB
Purpose:            Cheap draft generation from target hidden states
```

### The MTP Module

```
Input:              FC([RMSNorm(hidden), RMSNorm(embedding)]) → 2048
Layer:              1× full attention + 1× 256-expert MoE (same as main model)
Output:             RMSNorm → shared lm_head (2048 → 248,320)
Cost per prediction: ~1-2ms (1/40th of target model + lm_head)
```

---

## Detailed Acceptance Rate Analysis

### DFlash Per-Prompt Results (State-Block, B=8, decode-128, 6 prompts)

| Prompt | Verify Calls | Verify Tokens | Avg Accept | Best Row | Speed tok/s | Speed vs AR |
|---|---|---|---|---|---|---|
| 1 | 32 | 251 | 4.0 | 8 | 65.4 | 0.57× |
| 2 | 32 | 248 | 4.0 | 8 | 67.8 | 0.59× |
| 3 | 19 | 152 | 6.7 | 8 | 109.8 | 0.95× |
| 4 | 28 | 222 | 4.6 | 8 | 75.0 | 0.65× |
| 5 | 19 | 152 | 6.7 | 8 | 108.8 | 0.95× |
| 6 | 25 | 194 | 5.1 | 8 | 85.3 | 0.74× |
| **Total** | **155** | **1219** | **5.0** | — | **81.8** | **0.71×** |

### Per-Position Acceptance Rates (All 6 Prompts)

```
Position   Low (P1-2)   Medium (P4,6)   High (P3,5)   Aggregate
   1         100.0%        100.0%         100.0%        100.0%
   2          81.3%         92.9%          97.4%         90.5%
   3          64.1%         75.9%          92.1%         77.4%
   4          48.4%         60.6%          92.1%         67.0%
   5          35.9%         51.0%          81.6%         56.2%
   6          26.6%         45.2%          76.3%         49.4%
   7          25.0%         36.1%          68.4%         43.2%
   8          18.8%         22.9%          65.8%         35.8%
───────────────────────────────────────────────────────────────
A_out:        4.00          4.85           6.74          5.20
```

### MTP Proposal Depth Acceptance (B5, Interleaved Native-Loop)

```
Depth   Proposed   Accepted   Rate     Notes
  1        11         7       63.6%    First draft position
  2         7         5       71.4%    Higher than depth 1 (selection bias)
  3         5         3       60.0%    Only reached when 1-2 accepted
  4         3         3      100.0%    Small sample, very high
  5         3         2       66.7%    Small sample
```

MTP overall: avg 2.91 committed tokens/iteration, 83.9 tok/s vs AR 120.0 tok/s (0.70×)

### Accept-Length Histogram (DFlash B=8, All Prompts)

```
Accepted   Count   Cumulative   Notes
   1         22      14.2%      All drafts rejected (only root committed)
   2         22      28.4%      
   3         18      39.9%      
   4         15      49.7%      Median falls here
   5         11      56.8%      
   6          9      62.6%      
   7         12      70.3%      
   8         46      100.0%     Full acceptance (all 7 drafts + bonus)
```

Full acceptance (8/8) occurs in 29.7% of steps — driven primarily by the
high-acceptance prompts. On low-acceptance prompts, full acceptance is only ~19%.

---

## Speed Projection Tables

### Table 1: Acceptance Rate → Output Speed (This MoE Model)

```
AR baseline: 114.9 tok/s (8.7 ms/token)
Verify cost: 6.4 ms/row (η = 0.736)
Draft cost:  DFlash=10ms/step, MTP=2ms/step

Profile                   A_out  Per-Position Rates (1→8)                       DFlash   MTP    vs AR
─────────────────────────────────────────────────────────────────────────────────────────────────────
low (prompt 1-2)           4.00  1.00 0.81 0.64 0.48 0.36 0.27 0.25 0.19        65     75    0.65× ✗
medium (prompt 4,6)        4.85  1.00 0.93 0.76 0.61 0.51 0.45 0.36 0.23        79     91    0.79× ✗
high (prompt 3,5)          6.74  1.00 0.97 0.92 0.92 0.82 0.76 0.68 0.66       110    127    1.10× ✓
aggregate (all 6)          5.20  1.00 0.91 0.77 0.67 0.56 0.49 0.43 0.36        85     98    0.85× ✗
·····················································································
geometric α=0.9            5.70  1.00 0.90 0.81 0.73 0.66 0.59 0.53 0.48        93    107    0.93× ✗
geometric α=0.8            4.16  1.00 0.80 0.64 0.51 0.41 0.33 0.26 0.21        68     78    0.68× ✗
geometric α=0.7            3.14  1.00 0.70 0.49 0.34 0.24 0.17 0.12 0.08        51     59    0.51× ✗
─────────────────────────────────────────────────────────────────────────────────────────────────────
AR baseline               1.00  (one token per step)                           115    115    1.00×
```

**Only the "high" acceptance prompts marginally beat AR (1.10×) — and only with MTP drafts.**

### Table 2: Verification Cost Sensitivity

```
"What per-row verification cost is needed for speculation to work?"

With MTP drafts (2ms), B=8, using aggregate measured acceptance (A_out=5.20):

Verify/row    η       Agg tok/s    vs AR    Verdict
─────────────────────────────────────────────────────
  8.7 ms    1.000       73         0.63×    ✗ No batching benefit at all
  6.4 ms    0.736       98         0.85×    ✗ Current measured (28% proj batching)
  5.0 ms    0.575      124         1.08×    ~ Marginal (need 22% MoE speedup)
  4.0 ms    0.460      153         1.33×    ✓ Target threshold (38% MoE speedup)
  3.5 ms    0.402      173         1.51×    ✓ Strong (45% MoE speedup)
  3.0 ms    0.345      200         1.74×    ✓ Very strong (53% MoE speedup)
  2.0 ms    0.230      289         2.51×    ✓ Excellent (69% MoE speedup)
  1.0 ms    0.115      520         4.52×    ✓ Dense-equivalent (impossible on MoE)
─────────────────────────────────────────────────────

Key: To reach 1.3× AR, we need verify ≤ 4.0 ms/row (37% reduction from current).
     To reach 2.0× AR, we need verify ≤ 2.5 ms/row (61% reduction — unrealistic for MoE).
```

### Table 3: Optimal Block Size for This MoE

```
Using aggregate per-position acceptance, MTP drafts (2ms), verify=6.4ms/row:

 B   A_out   Verify    Step    tok/s   vs AR   Marginal B+1
────────────────────────────────────────────────────────────────
 1    1.00    6.4ms    8.4ms    119    1.04×   (baseline)
 2    1.91   12.8ms   14.8ms    129    1.12×   +10 tok/s ✓ (optimal B!)
 3    2.68   19.2ms   21.2ms    126    1.10×   -3 tok/s ✗
 4    3.35   25.6ms   27.6ms    121    1.06×   -5 tok/s ✗
 5    3.91   32.0ms   34.0ms    115    1.00×   -6 tok/s ✗ (break-even)
 6    4.41   38.4ms   40.4ms    109    0.95×   -6 tok/s ✗
 7    4.84   44.8ms   46.8ms    103    0.90×   -6 tok/s ✗
 8    5.20   51.2ms   53.2ms     98    0.85×   -6 tok/s ✗
────────────────────────────────────────────────────────────────
AR   1.00    8.7ms    8.7ms    115    1.00×

Note on B=2: The table shows B=2 at 1.12×, but this is UNRELIABLE. The 6.4ms/row
was measured within B=8 batches where projection amortization is maximal. At B=2,
projection-batching efficiency degrades and you lose most kernel-launch amortization.
Real per-row cost at B=2 is likely ~7.5-8.0ms, giving:
  B=2 realistic: 1.91 × 1000 / (2 + 2×7.5) = 112 tok/s ≈ 0.98× AR

Until B=2 is independently measured, do not treat the 1.12× as a real result.

Bottom line: No B value reliably beats AR on this architecture at current η.
```

### Table 4: Break-Even Acceptance Requirements

```
"What minimum A_out is needed for speedup ≥ 1.0× at various verify costs?"
MTP drafts (2ms), AR=8.7ms:

Verify/row    η     B=2 need  B=4 need  B=6 need  B=8 need   Feasible with measured rates?
─────────────────────────────────────────────────────────────────────────────────────────────
  1.0 ms   0.115    0.46      0.69      0.92      1.15      ✓ All B values
  2.0 ms   0.230    0.69      1.15      1.61      2.07      ✓ All B values  
  3.0 ms   0.345    0.92      1.61      2.30      2.99      ✓ All B values
  4.0 ms   0.460    1.15      2.07      2.99      3.91      ✓ All B values
  5.0 ms   0.575    1.38      2.53      3.68      4.83      ✓ B=2,4,6,8 (barely)
  6.4 ms   0.736    1.70      3.17      4.64      6.11      ✓ B=2,4 only
  8.7 ms   1.000    2.23      4.23      6.23      8.23      ✗ None feasible
─────────────────────────────────────────────────────────────────────────────────────────────
Measured A_out: B=2→1.91, B=4→3.35, B=6→4.71, B=8→5.20

At current 6.4ms/row: Only B=2 (need 1.70, have 1.91) and B=4 (need 3.17, have 3.35) are
technically above break-even, but by tiny margins that evaporate with overhead.
```

### Table 5: Dense Model Projections (Same Draft, Same Acceptance Rates)

```
Using our MEASURED DFlash acceptance rates on dense model targets.
Draft: 8-layer dense model, ~4ms/step on W7900
Shown with conservative η=0.17. Optimistic η=0.13 would be ~20-30% faster.

         Dense Target    AR    Verify(8)   Step  │  Low    Agg    High  │ Speedup  Δ tok/s
                       tok/s        ms      ms   │ tok/s  tok/s  tok/s  │  (agg)    (agg)
────────────────────────────────────────────────────────────────────────────────────────────
       Qwen3-32B W8A8    30     45.3ms   49.3ms  │   81    105    137   │  3.5×    +75
      Qwen3-32B W4A16    42     32.4ms   36.4ms  │  110    143    185   │  3.4×    +101
 Mistral-Small-24B W8    38     35.8ms   39.8ms  │  101    131    169   │  3.4×    +93
       Gemma-3-27B W8    33     41.2ms   45.2ms  │   88    115    149   │  3.5×    +82
       Qwen3-14B W8A8    58     23.4ms   27.4ms  │  146    190    246   │  3.3×    +132
       Llama-3-70B W4    22     61.8ms   65.8ms  │   61     79    102   │  3.6×    +57
───────────────────────────────────────────────────────────────────────────────────────────
  Qwen3.6-35B-A3B MoE   115     51.2ms   53.2ms  │   75     98    127   │  0.8×    -17 ← LOSES
────────────────────────────────────────────────────────────────────────────────────────────

Conservative estimate (η=0.17): dense 32B at 30 tok/s → ~105 tok/s = +75 tok/s gain (3.5×).
Optimistic estimate (η=0.13): dense 32B at 30 tok/s → ~134 tok/s = +104 tok/s gain (4.5×).
Plan for 2.5-3× until direct W7900 dense-verifier measurements exist.
The MoE at 115 tok/s → 98 tok/s = -17 tok/s loss. Same infrastructure, opposite outcomes.

Note: η=0.13 matches literature on NVIDIA (EAGLE, SpecInfer, Medusa papers).
η=0.17 conservatively accounts for W7900 ROCm INT8 burst behavior at B=8.
Actual η on W7900 needs direct measurement before committing to projections.
```

---

## Lessons from reference implementations (2026-05-15)

We spent several iteration cycles micro-optimizing our DFlash verifier into
increasingly intricate path / hybrid / adaptive policies while sitting at
`<= 0.92× AR` on Qwen3.5-27B. A side-by-side audit of every working DFlash
+ DDTree implementation we have a checkout of (`reference/ddtree-mlx`,
`reference/lucebox-hub/dflash`, `reference/hipfire`, `reference/ddtree`, plus
our reading of `reference/atlas/.../dflash_head`) showed that the policy
layer is **not** where the gain comes from. The structural pieces below are
shared by every reference that beats AR and absent from ours. Full audit
lives in `docs/DFLASH-FRESH-EYES.md`.

### Reference results on Qwen3.5-27B target + `z-lab/Qwen3.5-27B-DFlash`

| Impl | HW (BW) | Weights | AR tok/s | DFlash tok/s | vs AR |
| --- | --- | --- | ---: | ---: | ---: |
| **us — `ddtree_path_b8_budget22`** | W7900 (864 GB/s) | PARO W4A16 | 28.8 | 26.5 | **0.92×** |
| ddtree-mlx | M3 Ultra (819 GB/s) | MLX 4-bit | 27.9 | 38.6 chain / 42.3 +DDTree | **1.38× / 1.52×** |
| lucebox-hub `test_dflash` | RTX 3090 (936 GB/s) | GGUF Q4_K_M | 37.8 | 129.5 HumanEval mean | **3.43×** |
| hipfire `spec_step_ddtree_batched` | 7900 XTX (960 GB/s, gfx1100) | MQ4 asym3 | 44.1 | 196.0 code | **4.45×** |

W7900 has bandwidth comparable to or higher than M3 Ultra and RTX 3090, and
is the same gfx1100 family as the 7900 XTX. The gap is not a hardware
ceiling.

### The five structural patterns shared by every reference

1. **Tree verify is one batched forward over `N = 1 + budget` tree nodes**,
   not a per-row or depth-batched loop. Inputs are `tokens[N]`,
   `position_ids[N]`, an ancestor-only attention mask
   `[N, prefix_len + N]`, and `parent_ids[N]` (int32) that gets passed
   straight into the linear-attention kernels. No Python in the per-layer
   loop. The recurrent layers (Conv1D, GatedDelta) are themselves
   tree-aware, not sequenced from the host.

2. **Tree-aware Conv1D and GatedDelta kernels are *single launches* over all
   `N` nodes.** The grid shape that makes this work is the same across MLX,
   CUDA, and HIP variants:

   ```
   grid  = (head_v_dim, batch * num_v_heads)       // or similar; heads on the block axis
   block = (32, num_dk_warps, 1)                    // K_dim chunked across a warp/wavefront
   ```

   Inside each thread, the recurrence walks tree rows with a `t = 0..N` loop:

   ```
   for (int t = 0; t < N; ++t) {
     int parent_idx = parents[t];
     state[t] = (parent_idx < 0) ? base_state
                                 : state[parent_idx];   // same thread, earlier iter
     // recurrent update; write state[t]
   }
   ```

   Because the tree is topologically ordered (`parent_idx < t`), every
   parent-state read inside thread `T` was written by the same thread at an
   earlier iteration of the same kernel — no cross-thread sync needed.
   **One launch handles the whole tree.** No depth-batched Python.

3. **Per-node intermediate state goes straight into a persistent cache buffer
   during verify.** The verify kernel itself writes
   `state[node]` and `conv_state[node]` into a preallocated
   `[max_budget + 1, ...]` ring. Commit selects a slot index and copies it
   into the live cache with one `cudaMemcpyAsync` per layer. No re-forward,
   no per-cycle `torch.empty`. The lucebox `_persist` variant takes this
   further by writing intermediate states straight into the *target's*
   persistent state buffer, saving `~5-10 ms` per verify on Qwen 27B.
   DDTree-MLX reports that wiring this up dropped commit cost from
   `8511 ms` to `275 ms` (97 % reduction).

4. **Chain DFlash is the speed lane; DDTree is a `+10-15 %` topping.** All
   four references publish chain DFlash numbers separately and they are the
   ones that produce the headline speedup over AR. DDTree refines
   acceptance density on top of an already-winning chain. If chain DFlash
   is at `<= 1.0× AR`, no DDTree variant will recover it.

5. **Tree budget is small.** DDTree-MLX defaults to `budget=4` (5 verified
   rows) and explicitly calls higher budgets a loss on hybrid models because
   the recurrent layers can't parallelize across more tree nodes. Hipfire
   prunes by log-weight cutoff. Lucebox uses `budget=22` only on code
   prompts where AL `~ 8`; their non-code defaults are smaller. Our
   reflex of running `budget=22` everywhere is unsupported by the
   references.

### Implied design constraints for our stack

These are the hard rules we will treat as part of the speculative-decode
contract going forward (analogous to the η-threshold rules above):

- **Kernel grid shape is *the* structural choice.** Putting one tree node
  per threadblock forces depth-batched Python and is the dominant reason
  our tree verify is slow. The first kernel reshape work is to move
  `tree_node` from the block axis to a per-thread `t`-loop axis. This is
  not an optimization, it is a precondition.
- **Tree verify is a property of the kernel launch, not of the Python
  control flow.** If a tree verify path requires per-depth host launches,
  it is not the right path. The shape of the public DFlash verify API
  inside the engine should be: `forward(tokens[N], positions[N], mask, parents[N])`,
  and every recurrent layer kernel must accept `parents`.
- **Commit is a tensor copy, not a forward.** Once verify writes node
  states into the live cache ring, commit is `cudaMemcpyAsync` of one slot
  per layer plus a small KV compaction. Path-replay and accepted-prefix
  re-forwards are correctness debug paths, never the production path.
- **Default budget is small and prompt-conditioned.** Default `budget=4`,
  with `budget=8` opt-in for code-class prompts when the chain DFlash
  baseline shows AL `>= 6`. We will not promote `budget >= 16` until those
  smaller budgets are saturated.
- **Headline benchmark is a code prompt.** All references publish
  HumanEval-class numbers because code is where the acceptance multiplier
  shows up. Our `code / instruct / prose` mix is fine for correctness;
  the speed gate is HumanEval-class.
- **DFlash is a kernel-and-cache engineering problem.** Policy work
  (path-vs-bulk-vs-tree, adaptive selectors, etc.) is downstream of
  having a single-launch tree-aware kernel pair and a persistent-cache
  commit. Skip policy work until the kernel and cache pieces land.

### What this rules in / rules out

- *Rules in:* re-shaping our existing HIP
  `qwen35_linear_attn_tree_conv_decode_lowp_kernel` and
  `qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_kernel` to be one-launch
  parent-indexed kernels; allocating per-layer node-state rings;
  replacing the depth-loop in `_decode_linear_layer_tree_native_depths`
  with a single launch per layer; benching chain DFlash on HumanEval-class
  prompts.
- *Rules out:* further adaptive selectors, path-vs-hybrid policy variants,
  larger DDTree budgets, more profiling of the current per-node-block
  kernel shape, micro-tuning W4 projection inside the tree verify. The
  current verify shape is the wrong baseline to micro-tune.

### What landed (2026-05-15)

- **R1 — single-launch parent-indexed tree Conv1D + GDN kernels.**
  Shipped in `nano-vllm-amd` `b95eaa5` as
  `qwen35_linear_attn_tree_conv_decode_lowp_tloop_kernel` and
  `qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_kernel` (plus a
  separate `qwen35_gdn_tree_rmsnorm_gate_finalize_kernel`). Bit-exact
  vs the existing depth-batched v1 kernels and vs a python reference
  at real Qwen3.5-27B GDN shape (hk=128, hv=128, nv=32, nk=16). Conv
  tloop is 22.81× faster per call at chain=22; GDN tloop is at parity
  or up to 1.35× faster per call depending on chain length. While
  validating R1 we also retracted a long-standing `warpSize == 64`
  myth: under our wave32 production build flags `__shfl_down(x, 32)`
  is a silent no-op, so any block-wide reduction past 32 lanes must go
  through shared memory or per-32-lane shuffle + LDS exchange. The
  `shfl_probe_kernel` in `csrc/amd/smoke.hip` is kept as a future
  build-flag verification artifact.
- **R2 — bulk verifier driven by one launch per layer.** Shipped in
  `nano-vllm-amd` `69eb9d8` (Python wrappers) and workspace `1d1b6f0`
  (`_decode_linear_layer_tree_native_depths` learns the tloop branch).
  Gated on `DFLASH_USE_TLOOP=1` (default on); when enabled,
  `project_all_rows` is forced to `True` so the tree-aware verifier
  also exercises the tloop path. All correctness gates pass. E2E on
  the code prompt is within run-to-run variance because the remaining
  cycle cost (MoE, full-attn, W4 projection, allocator churn)
  dominates the saved kernel time at this point.
- **R7 — HumanEval-class prompts.** Added `humaneval` (HumanEval/53
  `add(a, b)`) and `humaneval_medium` (`sort_third`) to
  `STABLE_PROMPTS` in `scripts/bench_dense27_dflash_smoke.py`. Every
  reference DFlash implementation publishes its headline number on a
  HumanEval-class prompt; our `code / instruct / prose` mix was good
  for correctness but missed the acceptance regime where chain DFlash
  wins.
- **R5 — chain DFlash baseline.** The R7 sweep at
  `--proposal-mode ddtree-hybrid-bulk-direct --ddtree-topk 1
  --ddtree-budget bs --block-size bs` is structurally a chain DFlash
  baseline. It loses to path on the code prompt but wins on
  HumanEval; see the headline finding below.

R3 (per-layer node-state rings) and R4 (tree-aware FA KV ring commit)
remain open. R3 is currently deferred pending a profile that shows
allocator overhead is a measurable share of `verify_secs`.

### Headline finding (2026-05-15)

First time the bulk verifier beats `path` mode on this stack:

| Config (Qwen3.5-27B, decode=64, DFLASH_USE_TLOOP=1) | path vs_AR | hybrid-direct vs_AR | acc/cycle | rows/out (hybrid) |
| --- | ---: | ---: | ---: | ---: |
| code prompt, bs=8 | 0.740 | 0.578 | 1.91 | 1.91 |
| HumanEval, bs=4 | 0.806 | 0.848 | 2.56 | 1.03 |
| **HumanEval, bs=8** | **0.841** | **0.954** | **4.82** | **1.20** |
| HumanEval, bs=12 | 0.854 | 0.492 | 6.11 | 1.52 |
| HumanEval, bs=16 | 0.859 | 0.597 | 7.00 | 1.48 |

Key reads:

- HumanEval bs=8 hybrid-direct = 0.954× AR is within 5 % of break-even
  and within reach of >1.0× with one more pass. The same configuration
  on the code prompt was 0.578× AR — acceptance per cycle 2.5× higher
  on HumanEval is what flips the verdict, confirming the references'
  finding that HumanEval-class prompts are where chain DFlash wins.
- Path-mode AR ratio is monotonic and stable across the sweep
  (`0.74 → 0.86`). The bulk verifier is the lane that flips.
- The bs=12 and bs=16 rows are *not* monotonic in either direction.
  `verify_secs` cliff-jumps `1.77 → 3.92 → 3.19` despite acceptance
  climbing monotonically. Profile required (see below).

**Quality-gate violation at bs=16.** `peak_allocated_gib` trajectory
over the sweep is `22.71 (bs=8) → 23.96 (bs=12) → 25.15 (bs=16)`. The
bs=16 row breaches the 24 GiB ceiling documented in
`AGENTS.md “Post-Run Quality Gates”`. bs=16 should not be retained as
a production config without first cutting the runtime working-set;
bs=8 remains within budget.

### What this implies for the next pass

The priority queue shifts:

1. **Profile the bulk verifier on HumanEval bs=8.** Run
   `rocprofv3 --kernel-trace true` per `AGENTS.md “Pre-optimization
   grid/occupancy audit”`. Sum `DurationNs` per `KernelName` to find
   the top time-share kernel before any further structural work. Prior
   project history makes the MoE pack8 family the strong prior
   (~70 % of decode time on similar shapes), but it is *not* measured
   on the bulk verifier yet.
2. **Diagnose the bs≥ 12 cliff** as a side-track from the main
   profile. Goal: rule out a single fixable cause (allocator threshold,
   MoE expert-grouping degradation past some token count, full-attn
   ancestor-mask cost rising as N²) before committing to any
   structural MoE batching work.
3. **MoE bulk-batching by expert** only if profile #1 confirms MoE
   pack8 is the dominant bucket on the bulk verifier path. Sort-
   permute rows by selected expert, run grouped GEMM, gather back.
   References almost certainly do this; we have not measured ours.
4. **R3 / R4** remain queued, but only as follow-on once the dominant
   bulk-verifier bottleneck is identified.

Policy work (adaptive selector, larger DDTree budgets) stays
rules-out until step 1 finds a real lever.

---

## Appendix: Raw Measurement Data

### Per-Position Acceptance Rates (All 6 Prompts, Individual)

```
Prompt 1 (low):  1.00  0.78  0.66  0.50  0.34  0.28  0.25  0.19  │ A_out=4.00
Prompt 2 (low):  1.00  0.84  0.63  0.47  0.38  0.25  0.25  0.19  │ A_out=4.00
Prompt 3 (high): 1.00  1.00  0.95  0.95  0.84  0.74  0.63  0.63  │ A_out=6.74
Prompt 4 (med):  1.00  0.86  0.68  0.57  0.50  0.46  0.32  0.18  │ A_out=4.57
Prompt 5 (high): 1.00  0.95  0.89  0.89  0.79  0.79  0.74  0.68  │ A_out=6.74
Prompt 6 (med):  1.00  1.00  0.84  0.64  0.52  0.44  0.40  0.28  │ A_out=5.12
```

### MTP Proposal Depth Acceptance (B5, Interleaved)

```
Depth 1: 63.6% (11 proposed, 7 accepted)
Depth 2: 71.4% (7 proposed, 5 accepted)
Depth 3: 60.0% (5 proposed, 3 accepted)
Depth 4: 100%  (3 proposed, 3 accepted)
Depth 5: 66.7% (3 proposed, 2 accepted)
```

### Timing Breakdown (DFlash State-Block, per output token)

```
target_verify_state_block: 12.43 ms/output_token (dominant)
draft_forward_lmhead:       2.20 ms/output_token
reference_ar_compare:       8.64 ms/output_token (sanity check, not production)
accept_follow:              0.08 ms/output_token
context_hidden_append:      0.02 ms/output_token
```

---

## Appendix: Formulas & Quick Reference

### Speedup Formula
```
Speedup = A_out / (D_norm + B × η)

Where:
  A_out    = average committed output tokens per step
  D_norm   = draft_cost / AR_cost (normalized draft overhead)
  B        = block/draft length
  η        = Verify(B) / (B × AR_cost) (verification efficiency)
```

### Break-Even Condition
```
Speculation beats AR when:  A_out > D_norm + B × η
```

### Quick Calculator

For any model/architecture, measure:
1. `AR_cost` — time for one AR decode step
2. `Verify_B_cost` — time for one batched verification of B positions
3. Compute `η = Verify_B_cost / (B × AR_cost)`
4. If `η > 0.5` → speculation is unlikely to help (need >50% acceptance at all positions)
5. If `η < 0.2` → speculation almost certainly helps (need only >20% position-1 acceptance)
6. Between 0.2-0.5 → depends on acceptance rate and draft cost

### Rules of Thumb

- **Dense models**: η ≈ 0.13-0.20 → always speculate, expect 2.5-4× speedup (plan for 3×)
- **Small MoE** (8-16 experts, top-2): η ≈ 0.25-0.35 → usually viable, expect 1.5-2.5×
- **Large MoE** (64+ experts, top-6+): η ≈ 0.5-0.7 → marginal at best
- **Hybrid MoE + recurrent** (this model): η ≈ 0.7-0.8 → not viable with current dispatch
- **Pure recurrent** (Mamba, RWKV, current impls): η ≈ 0.7-0.9 → not viable (parallel scan could help)

All values are implementation-bound, not structural limits. For 256-expert MoE,
the current visible MoE/router buckets are already enough to keep the path out
of the dense regime unless grouped/budgeted expert dispatch changes the verifier
cost curve.
