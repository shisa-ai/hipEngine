# Relaxed Precision Mode Plan

_Status: planning/catalog document. Strict/exact remains the default policy. This
file inventories what an opt-in relaxed mode could unlock and what evidence it
would need before it can become a retained path._

## Purpose

`docs/KERNELS.md` catalogs the kernels we have landed and the gates that made
them acceptable. Today those gates strongly prefer **bit-exact or parent-parity
matches** for fused kernels: the fused output should match the unfused/reference
chain at the same dtype and operation order unless the kernel is explicitly a
quantized feature such as INT8 KV cache.

That policy is useful while the runtime is still small: it localizes bugs, keeps
`LLM.generate()` reproducible, and prevents a benchmark win from hiding math
regressions. It also blocks a class of optimizations where the only practical
implementation changes reduction order, uses lower-precision intermediates,
relies on approximate math intrinsics, or makes token/expert ordering unstable.

This document defines two retained modes:

- **Strict / exact mode**: the default. Preserve the current behavior, current
  correctness oracles, and exact/parent-parity expectations.
- **Relaxed mode**: explicit opt-in only. May use non-bit-perfect kernels when
  the drift is bounded by model-level quality gates and every relaxed variant has
  a strict fallback.

No speedup is claimed here. "Savings" below means a plausible source of saved
launches, memory traffic, scratch, or occupancy pressure that still needs a
measured artifact before it can be reported as performance.

## Current strict inference path

The current Qwen3.5/PARO path is intentionally narrow and auditable:

1. `hipengine.LLM.generate(...)`
2. generation registry lookup
3. `Qwen35ParoOneTokenGenerator`
4. `Qwen35ParoNextTokenRunner` / `Qwen35ParoResidentSession`
5. native prefill via `prefill_native(...)` and `_run_native_prefill_layers(...)`
6. decode via the resident `_run_layers(...)` path

The active kernel catalog spans HIP gfx1100 attention/KV, rotary, RMSNorm,
linear/W8A16/AWQ/Marlin-style packed paths, MoE routing/scatter/expert/combine,
linear-attention conv/GDN pieces, casts, and runtime utility kernels. Existing
experimental toggles are already opt-in, for example:

- `HIPENGINE_PARO_ROTATE_DUAL_PACK8_FUSED`
- `HIPENGINE_PARO_FULL_ATTN_KV_PACK8_FUSED`
- `HIPENGINE_PARO_ROUTER_TOPK_COOP`

Those toggles should be treated as prototypes for a cleaner precision-policy
surface, not as permission to silently change the default path.

## Mode contract

### Strict / exact mode

Strict mode remains the retained default for all public APIs, benchmarks, and
fixtures unless a command explicitly names a relaxed profile.

Requirements:

- Resolve kernels through the existing `(backend, layer, quant, variant)` plugin
  registry. Do not add ad-hoc `if relaxed` branches in engine/model code.
- Keep strict variants available when adding relaxed variants.
- Preserve bit-exact or parent-parity comparisons for fused kernels where those
  comparisons exist.
- Preserve exact dense-KV semantics unless the user explicitly selects a
  quantized KV storage policy. INT8 KV remains a named capacity/diagnostic path,
  not a silent replacement for BF16 KV.
- Keep deterministic fixture behavior. If a kernel previously matched generated
  token IDs, strict mode must continue to do so.

### Relaxed mode

Relaxed mode is allowed to trade exact matching for bounded model-level drift.
It should be configured once near the public/runtime boundary and passed down as
a precision policy or registry variant selector. Environment variables can remain
for experiments, but retained relaxed behavior should be visible in command-line
arguments, benchmark artifacts, and logs.

Minimum requirements for a retained relaxed kernel:

- Explicit opt-in profile, for example `precision_mode="relaxed"` or a named
  profile such as `relaxed_kv_int8` / `relaxed_fast_math`.
- Strict fallback registered for the same layer/quant combination.
- Variant name carries the policy, for example `variant="relaxed_fast_math"`,
  instead of replacing `default`.
- Per-kernel oracle coverage plus an end-to-end fixture gate. The default floor
  should be no weaker than the project gate for new/ported kernels: KL <= 0.05
  and top-1 agreement >= 90% vs the CPU/reference fixture, unless a stricter
  task-specific gate is documented.
- Repeated fixed-seed runs to catch nondeterminism. Relaxed mode may be
  non-bit-perfect; it must not be flaky.
- Benchmark artifacts must record the relaxed profile, kernel variants, model,
  quant, workload shape, hardware, command, result, and correctness gate.

## Savings vocabulary

| Term | Meaning |
| --- | --- |
| Launch | Remove or fuse a HIP launch, or make graph replay cover a larger unit. |
| HBM | Avoid a global-memory read/write of activations, KV, logits, or scratch. |
| Scratch | Shrink temporary buffers or avoid materializing intermediates. |
| Occupancy | Improve waves/VGPR/LDS balance or choose a faster compiler schedule. |
| Capacity | Reduce retained KV/weight footprint enough to unlock longer contexts. |

## Per-kernel relaxed opportunities

| Kernel family | Strict constraint today | Relaxed candidates | Possible savings | Required evidence / risks |
| --- | --- | --- | --- | --- |
| RMSNorm / add-RMSNorm | Match reference reduction and dtype behavior; fused forms must match the unfused chain. | Approximate `rsqrt`, reordered reductions, BF16/FP16 intermediate accumulation where safe, fusing residual add + norm + following elementwise ops. | Launch, HBM, occupancy. | Hidden-state drift accumulates across every layer; require per-layer max error, fixture KL/top-1, and repeated-run stability. |
| Rotary / RoPE | Preserve parent-parity for sin/cos lookup, pair ordering, and output dtype. | Fuse Q/K rotate, rotate after/with RMSNorm, use lower-precision trig tables, reorder vector packs, keep dual-pack8 fused path as a policy variant. | Launch, HBM, scratch. | Small phase errors can move attention logits; validate full-attention and linear-attention layers separately. |
| Full-attention prefill | Softmax/reduction order and KV append behavior are correctness-sensitive; previous nondeterministic softmax failures were real bugs, not acceptable relaxation. | Flash/AOTriton-style block reductions, lower-precision logits/PV intermediates, query chunking choices that alter associativity, fused QKV/KV pack path. | HBM, scratch, occupancy, launch. | Must prove deterministic enough under fixed seeds; compare per-layer attention output plus end-to-end fixtures. |
| Paged-attention decode / `KVLiveSpans` | Dense BF16 KV is the exact baseline; live-span ABI must be honored for dense and eviction policies. | INT8/FP8 KV profiles, approximate or coarser scales, reordered page traversal, fused dequant + attention, larger split/merge choices. | HBM, capacity, launch. | Long-context quality risk; report KV policy explicitly and gate both short fixture KL/top-1 and long-context smoke. |
| KV write / append / repack | Writes must preserve exact layout consumed by strict attention. | Quantize-on-write, fuse append with pack/dequant metadata, use relaxed scale dtype, avoid temporary BF16 materialization. | HBM, scratch, capacity. | A write-side bug contaminates all later tokens; require readback/oracle tests for live spans and eviction masks. |
| Dense W8A16 / AWQ / Marlin-style linear | Packed layout, scale application, and accumulation order are held to parent/reference parity. | Different MMA tiling, reordered split-K reductions, lower-precision accumulators for selected layers, pre-swizzled relaxed layouts, per-kernel waves/VGPR tuning. | Occupancy, HBM, scratch. | Logit drift can be layer-local but token-critical; validate layer outputs and full fixture logits before accepting throughput. |
| MoE router / top-k | Expert choice, tie handling, and token/expert order must match strict behavior. | Cooperative top-k, unstable tie order, approximate gating math, early pruning of tiny probabilities, token grouping that reorders equal-priority work. | Launch, scratch, occupancy. | Highest semantic risk: expert choice changes are discrete. Require route agreement metrics in addition to KL/top-1. |
| MoE scatter / gather / group GEMM | Strict mode preserves token ordering and exact combine inputs. | Out-of-order token groups, atomic or unordered combine, larger grouped GEMM batches that alter accumulation order. | Launch, HBM, occupancy. | OOO/atomic behavior must be repeatable enough; require fixed-seed repeat checks and per-expert accounting. |
| MoE combine / SwiLU / residual | Fused combine must match unfused gate/up/down + SiLU + residual chain. | Approximate sigmoid/SiLU, lower-precision gate/up product, fuse residual/writeback, relaxed expert accumulation order. | Launch, HBM, scratch. | Nonlinear error can amplify; gate with layer-level deltas and generated-token/quality fixtures. |
| Linear-attention conv / GDN / recurrence | Recurrent state updates are order-sensitive; chunking must preserve strict semantics. | Reassociate chunk scans, fuse GDN RMSNorm + SiLU + rotate, fuse shared-gate sigmoid in prefill, lower-precision recurrent intermediates. | Launch, HBM, scratch, occupancy. | Recurrence drift can grow with context length; validate at multiple prompt lengths and after decode continuation. |
| PARO-specific fused attention helpers | Existing fused pack/rotate helpers are opt-in until parity and fixture gates are strong. | Promote rotate dual-pack8 and full-attn KV pack8 fusions under a named relaxed profile if exact parity is too expensive. | Launch, HBM. | Must be labeled relaxed if it cannot preserve strict parity. Do not make env-only behavior the retained path. |
| LM head / argmax / sampling | Greedy path expects stable logits and token choice; ties should be deterministic. | Quantized LM head, chunked top-k/argmax with relaxed tie order, avoid full-logit materialization when only top-1/top-k is needed. | HBM, scratch, launch. | Token ID changes are user-visible. Gate with logit KL/top-1 and sequence-level prompt suites, not just per-token error. |
| Casts / activation utilities | Cast points and rounding match strict dtype expectations. | Lazy casts, lower-precision scratch buffers, in-place updates when aliasing is proven safe, vectorized unaligned paths. | Launch, HBM, scratch. | Easy to hide aliasing bugs; require sanitizer-style shape tests and strict fallback. |
| Compiler/build-profile variants | Current retained flags are conservative; `-amdgpu-unroll-threshold-local=600` is neutral/default, not a broad speed lever. | Per-kernel fast-math/denormal-flush experiments, waves-per-EU/VGPR retuning, local unroll changes only where measured. | Occupancy, launch latency side effects. | Treat as kernel variants with artifacts; no blanket flag sweeps without per-kernel correctness and benchmark evidence. |

## Backlog unlocked by relaxed mode

These are candidates that were previously blocked, parked, or risky because they
changed strict ordering, associativity, or bit-perfect behavior. They are not
approved by this document; they become legal to prototype only inside a relaxed
profile with the evidence above.

1. **Central precision-policy plumbing**
   - Add a runtime-visible policy object/profile and map it to registry variants.
   - Migrate env-only experiments into named variants once retained.
   - Ensure benchmark artifacts print the selected policy.

2. **Compound decode launch reduction**
   - Fuse RMSNorm/add-RMSNorm with rotate where exact parity is too expensive.
   - Combine small elementwise casts/scale/writeback kernels in decode.
   - Keep strict unfused chain registered for bisecting.

3. **Relaxed full-attention prefill path**
   - Revisit AOTriton/flash-style reductions and query chunking that change
     softmax/PV associativity.
   - Prototype fused QKV/KV-pack append under a named relaxed variant.
   - Include deterministic repeat tests because earlier prefill softmax issues
     showed that nondeterminism can masquerade as precision drift.

4. **Relaxed KV cache profiles**
   - Promote INT8 KV from explicit capacity/diagnostic mode only after fixture
     and long-context gates are documented for the selected model.
   - Evaluate FP8 or mixed-scale variants if the scale metadata and live-span ABI
     remain compatible with strict fallback.

5. **MoE router and combine experiments**
   - Revisit cooperative top-k, unstable tie ordering, route pruning, and
     OOO token grouping.
   - Track route agreement and expert-load deltas, not just final logits.

6. **Linear-attention prefill fusion**
   - Prototype GDN RMSNorm + SiLU + rotate and shared-gate sigmoid fusions.
   - Validate recurrence drift across short, medium, and long prompt lengths.

7. **Packed linear / Marlin / WMMA retuning**
   - Allow split-K and WMMA accumulation orders that are not bit-identical.
   - Try per-kernel waves/VGPR/unroll settings only with retained artifacts.

8. **Approximate nonlinear/math intrinsics**
   - Evaluate approximate sigmoid/SiLU/exp/rsqrt in isolated kernels first.
   - Accept only if layer-local error and end-to-end quality remain inside the
     relaxed profile budget.

9. **Argmax/top-k without full logits**
   - Add a relaxed LM-head top-1/top-k path that may change tie behavior but
     avoids materializing full logits when the API does not need them.

10. **Out-of-order and atomic reductions**
    - Prototype OOO expert grouping, split-K atomics, or unordered combine only
      in relaxed mode.
    - Require repeat-run variance checks and a strict deterministic fallback.

## Acceptance checklist

Before any relaxed variant is called retained:

- [ ] Strict mode still passes the same tests and uses the same default variants.
- [ ] Relaxed variant is selected by an explicit profile and appears in logs.
- [ ] Registry entries keep strict and relaxed variants side by side.
- [ ] Per-kernel oracle output is saved or reproducible.
- [ ] End-to-end fixture reports KL, top-1 agreement, generated-token behavior,
      and repeated-run stability.
- [ ] Benchmark artifact records exact command, hardware, model, quant, workload,
      relaxed profile, correctness gate, and measured result.
- [ ] `WORKLOG.md`, `benchmarks/README.md`, and `benchmarks/CHANGELOG.md` are
      updated for any retained performance claim.

## Non-goals

- Relaxed mode is not a reason to weaken strict mode.
- Relaxed mode does not excuse nondeterministic bugs, illegal memory behavior,
  stale JIT cache issues, or untracked benchmark conditions.
- Relaxed variants do not bypass the plugin registry architecture.
- Relaxed precision does not move kernel R&D into this repository; exploratory
  micro-tuning still belongs in `~/amd-gpu-tuning/` until a stable kernel is
  ready to port.
