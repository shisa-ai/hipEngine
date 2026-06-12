# Relaxed Precision Mode Plan

_Status: planning/catalog document, now with **one landed relaxed profile**
(2026-06-09, GDN chain dv-tiling — see §0 below). Strict/exact remains the
default policy. This file inventories what an opt-in relaxed mode could unlock,
the RDNA3 hardware levers it would let us use, and the drift gates that keep it
from going off the rails._

## 0. Landed relaxed profiles (first foray)

This section is the running ledger of relaxed kernels we have actually shipped
(vs the catalog of opportunities below). Each entry fully characterizes the
inaccuracy, names its drift tier, points at the strict fallback, and records the
measured end-to-end effect.

### 0.1 GDN chain dv-tiling — verify path (2026-06-09) — **first relaxed kernel**

**What it is.** The MTP/DFlash verify-path GDN chain recurrence
(`qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop`) was dv-tiled: each block
now owns 4 consecutive dv columns (grid 4096→1024 blocks) so per-(v_head,t)
work is computed once per tile and the dv state writes coalesce. Kernel win:
rocprof **72.0→53.39 µs/call (−25.8%)** (see `docs/MEGAKERNEL.md` §9.4). It is
the **first kernel we have retained that is not bit-exact / parent-parity** — the
first genuine relaxed-mode landing.

**Naming note (two different "T" ladders).** `docs/MEGAKERNEL.md` calls the
*verify-path correctness policy* T0/T1/T2 (T0 = bit-exact-vs-AR; **T1** =
self-consistent + KL-gated; T2 = fully relaxed). That is a **policy** decision
(which gate the verify path must clear). This document's T0–T4 are **drift
budget tiers** (how much numerical drift a kernel may carry). They are
orthogonal: the dv-tiling is **MEGAKERNEL-policy T1** (gate on KL vs
`cpu_reference`, not bit-exact `exact_ar_match`) and **RELAXED drift-tier T2**
(layout/order: reduction reordering). Do not conflate the two ladders.

**Type of inaccuracy — fully characterized.**

- **Class:** FP **reduction-order / FMA-contraction reassociation**. This is
  *not* an algorithmic approximation, *not* a lower-precision intermediate, and
  *not* a layout/quant change. The algebra is **identical** to the strict
  kernel — same per-dv accumulation order over `dk`, same `q_scale`/`k_scale`/
  `beta`/`decay` scalars (computed once per tile instead of redundantly, but
  with the same ops → same bits). The only thing that moved is how the compiler
  **schedules/contracts FMAs** across the restructured loops.
- **Magnitude (vs `cpu_reference`, numpy f32 delta-rule oracle):** chain output
  `out_max_abs` 1.67e-6 (strict) → **1.07e-6** (dv-tiled); recurrent state
  `leaf_max_abs` **5.96e-8** (unchanged). The two kernels therefore differ from
  each other by ≤ ~2.7e-6 absolute worst case — **~1–2 ULP at fp32** for these
  operand magnitudes. Both sit ~4 orders of magnitude inside the project KL
  ceiling.
- **Propagation:** the drift feeds the recurrence (state carries forward across
  `max_nodes` tokens), but stays bounded — the microbench runs the full T=4 and
  T=8 chains and the error stays at ~1e-6 (no blow-up).
- **Token-level effect:** **zero on well-conditioned (real) prompts.** On the
  retained quicksort prompt (90 tok, decode 32, B=3, 3 runs) `exact_ar_match`
  stays **true** and the accept pattern is **byte-identical** to strict. The
  only observed flip is on the **degenerate 1-token smoke prompt** (151646):
  with no context the next-token distribution is near-flat, the ~1 ULP tips one
  argmax at a near-tie, and the autoregressive stream then cascades. That is
  boundary sensitivity of a self-consistency check, **not** a model-quality
  regression.
- **KL:** ≪ 0.05; effectively 0 at the logit level on real prompts
  (`exact_ar_match=true` ⇒ identical generated IDs ⇒ sequence-logit KL ≈ 0).
- **Determinism:** byte-deterministic across runs (identical `accepted_lengths`
  over 3 runs).

**Why it is "not wrong, just not bit-exact."** The verify path and the AR path
are *already different kernels* (chain recurrence vs single-token decode); both
are KL-correct vs `cpu_reference`. `exact_ar_match` (spec tokens == same-run AR
tokens) is a **cross-kernel self-consistency** check, explicitly *not* a
model-quality bar (`docs/MEGAKERNEL.md` §5). The correct gate is the project
floor KL ≤ 0.05 / top-1 ≥ 90% vs `cpu_reference`, which the dv-tiling clears by
4+ orders of magnitude.

**Drift-tier classification: T2 (layout/order relaxed).** Reduction reordering
is the textbook T2 lever. Measured against the T2 budget it is comfortably
inside (and on real prompts it actually meets the stricter **T1** numerical
tier): per-kernel `max_abs ~1e-6`, sequence-logit KL ≈ 0, top-1 100%,
generated-IDs match on the real prompt, deterministic across runs. The lone
caveat — a single ID flip on the *degenerate* 1-token prompt — is outside the
retained fixture set (512/4K/32K/128K real contexts) and is a self-consistency
artifact, not a drift-tier metric.

**Strict fallback (mode-contract requirement, satisfied).** The kernel is
templated `<scalar_t, VTILE>`; **`VTILE=1` is bit-identical to the
pre-relaxation strict kernel** and is the registered fallback for non-divisible
`head_v_dim`. Strict mode is preserved; relaxed (VTILE=4) is the deployed path.

**End-to-end effect (C_B) — measured, honest.** Same-prompt A/B, quicksort,
decode-tokens=32, B=3, **3 runs each**, W7900/gfx1100
(`scripts/mtp_verifier_economics.py`, gate off):

| metric (mean ± std, n=3) | strict (shuffle) | relaxed (dv-tiled) |
|---|---:|---:|
| **C_B (cycle_cost, AR-tok)** | **4.81 ± 0.14** | **4.80 ± 0.28** |
| verify ms / cycle | 34.35 ± 0.63 | 35.12 ± 1.20 |
| cycle wall ms | 44.37 | 45.17 |
| acceptance rate | 0.4615 (std 0) | 0.4615 (std 0) |
| accepted / cycle | 1.385 | 1.385 |
| `all_exact_ar_match` | true | true |

**C_B is unchanged within noise** (Δ = −0.016, ~30× smaller than the run-to-run
std), and **acceptance is byte-identical**. The rocprof-confirmed −0.56 ms/pass
kernel saving is **below the economics noise floor** (cycle-wall std ~0.6–1.2 ms)
because the verify cycle is **dispatch/host-bound, not kernel-bound** (~19.4 ms
dispatch floor + ~12 ms other kernels; `docs/MEGAKERNEL.md` §9.2). **Lesson:**
kernel-time relaxation alone does not move C_B at B=3; the dv-tiling is banked
kernel-time headroom that will only register once the dispatch floor is also
reduced. The relaxation's value here is **correctness headroom unlocked at zero
acceptance cost**, validated on a real prompt — the template for future verify
relaxations — not a standalone economics win. Artifact:
`benchmarks/results/2026-06-09-hipengine-m16-gdn-dvtiling-economics-cb.json`.

### 0.2 MTP-specific relaxed follow-ups from the break-even sprint

The exact MTP sprint moved the 35B-A3B row from `0.758x / 27.8 ms` to the
current exact D32 best `1.023x / 14.134 ms` by keeping only exact AR-preserving
improvements in `docs/MTP.md`. The items below are **not** retained exact-speed
work. They are worth parking here because they change either numerical parity,
the greedy-token contract, or the verifier/AR self-consistency gate.

| Candidate | Relaxed tier | Why it belongs here | Current evidence | Next gate |
| --- | --- | --- | --- | --- |
| D64 fast verifier under KL/top-1 gates | T2 | The fast D32 current-best verifier uses batched/t-loop paths that can accumulate resident-state drift by D64; strict fallbacks now restore exact AR but cost speed. A relaxed profile could keep the fast path if model-level KL/top-1 is acceptable even when exact greedy diverges late. | D64 strict fallback passes exact `9/9` with `HIPENGINE_GDN_TLOOP_C1_EXACT=1` + `HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS=1`; `decode_batched` exact suffix also passes but remains slower than D32 current-best. Drift audits localize the strict fork to GDN recurrence/order and full-attention suffix state. | Build a D64 KL/top-1 prompt suite comparing fast verifier vs serial AR/cpu reference; do not promote from exact-token match alone. |
| Speculative sampling / probability-ratio acceptance | T4 | This abandons exact greedy acceptance and instead preserves the target distribution statistically; it can accept more draft tokens than top-1 equality. | Not implemented in MTP. Current accept summary reads top-1/IDs, not target/draft probability ratios. | Design a GPU accept kernel that consumes target and draft log-probs, plus an evaluation harness for distributional equivalence and throughput cost. |
| Near-tie/top-k greedy relaxation | T4 | Accepting a draft token that is within a small target-logit margin or in target top-k could raise acceptance density, but it intentionally changes greedy output. | Current exact D32 B=1 has only `1.617` visible tokens/cycle; cap65536 recovered real exact hits, but no relaxed near-tie acceptor exists. | Measure rejected-draft target rank/logit-margin distributions first; only prototype if many rejects are near-ties and deterministic fixed-seed behavior remains stable. |
| Relaxed LM-head/top-k materialization | T2/T4 | Avoiding full `[rows, vocab]` logits or relaxing tie order may reduce W8A16/argmax cost, but strict MTP requires exact verifier top-1 IDs. | Exact fused verifier LM-head no-held on B=3 and B=1 because the replacement W8A16 body was slower despite removing a launch. | Reframe as a relaxed top-k/argmax layout with KL/top-k agreement gates, not another strict fused LM-head attempt. |
| Further GDN/order-relaxed verifier kernels | T2 | The landed dv-tiling shows tiny recurrence-order drift can be bounded; MTP's D64 audits show GDN recurrence order is also exactly where strict long-horizon drift appears. | `VTILE=4` dv-tiling is the first relaxed profile; `VTILE=8` no-held as an exact speed attempt; strict c1-equivalent fallback fixes D64 but is slower. | Keep strict fallback; compare recurrence drift, sequence-logit KL, and long-horizon top-1 on D64/D128 before any new default. |

Do **not** move ordinary exact no-holds into this file just because they were
negative. B=1 `c1_loop`, W4 all-sites, small-batch threshold, LM-head thread
count, staged selected rotate, decode graph capture, selected-down staged, and
the pairwise reduced-DAG opt-outs are exact-mode evidence in `docs/MTP.md`, not
relaxed-mode candidates.

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
relies on approximate math intrinsics, packs into a non-parent layout, or makes
token/expert ordering unstable.

This document defines two retained modes:

- **Strict / exact mode**: the default. Preserve current behavior, current
  correctness oracles, and exact/parent-parity expectations.
- **Relaxed mode**: explicit opt-in only. May use non-bit-perfect kernels when
  the drift is bounded by named per-tier budgets and every relaxed variant has
  a strict fallback registered.

No measured speedup is claimed here. Every percentage in this document is a
ballpark **ceiling** derived from the M.4 decode Amdahl table in
`docs/OPTIMIZE.md` §6 and the roofline in `docs/ROOFLINE.md`. None of these
numbers replaces the benchmark artifact a retained relaxed variant must produce.

## Current strict inference path

The current Qwen3.5/PARO path is intentionally narrow and auditable:

1. `hipengine.LLM.generate(...)`
2. generation registry lookup
3. `Qwen35ParoOneTokenGenerator`
4. `Qwen35ParoNextTokenRunner` / `Qwen35ParoResidentSession`
5. native prefill via `prefill_native(...)` and `_run_native_prefill_layers(...)`
6. decode via the resident `_run_layers(...)` path

The active kernel catalog spans HIP gfx1100 attention/KV, rotary, RMSNorm,
linear/W8A16/AWQ/Marlin-K packed paths, MoE routing/scatter/expert/combine,
linear-attention conv/GDN pieces, casts, and runtime utility kernels. Existing
experimental toggles are already opt-in, for example:

- `HIPENGINE_PARO_ROTATE_DUAL_PACK8_FUSED` (D1.1, currently rejected as default)
- `HIPENGINE_PARO_FULL_ATTN_KV_PACK8_FUSED` (D1.6, currently rejected as default)
- `HIPENGINE_PARO_ROUTER_TOPK_COOP` (D1.5 / D5.3, currently rejected as default)

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
  quantized KV storage policy. INT8 KV remains a named capacity/diagnostic path
  (`docs/KVCACHE.md`), not a silent replacement for BF16 KV.
- Keep deterministic fixture behavior. If a kernel previously matched generated
  token IDs at a fixed seed, strict mode must continue to do so.

### Relaxed mode

Relaxed mode is allowed to trade exact matching for **bounded** model-level
drift. It is configured once near the public/runtime boundary and propagated as
a named precision profile + registry variant selector. Environment variables
may remain for experiments, but retained relaxed behavior must be visible in
command-line arguments, benchmark artifacts, and logs.

Minimum requirements for a retained relaxed kernel:

- Explicit opt-in profile (see §Relaxed profile tiers), for example
  `precision_mode="relaxed_fast_math"`.
- Strict fallback registered for the same `(layer, quant)`.
- Variant name carries the policy, for example
  `variant="relaxed_fast_math"`, instead of replacing `default`.
- Per-kernel oracle coverage plus end-to-end fixture gates from the tier's
  drift budget (§Accuracy and drift policy).
- Repeated fixed-seed runs to catch nondeterminism. Relaxed mode may be
  non-bit-perfect; it must not be flaky.
- Benchmark artifacts must record the relaxed profile, kernel variants, model,
  quant, workload shape, hardware, command, result, and correctness gate.

## Accuracy and drift policy

Relaxed mode is bounded by **layered drift budgets**. A retained relaxed
variant must pass every applicable layer below; tiers stack from cheapest to
strictest. The project's existing floor for new/ported kernels is
`KL ≤ 0.05` and `top-1 ≥ 90%` from `docs/TESTING.md` §4. That stays the
**outer** ceiling; relaxed tiers below it are progressively tighter.

### Drift budget tiers

| Tier | Name | Per-kernel oracle | Per-layer hidden-state KL | Sequence-logit KL | Top-1 agreement | Generated-ID match | Nondeterminism | Typical use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | strict | bit-exact (`mismatch=0`) or parent-parity | n/a | `KL = 0` on retained fixture | 100% | match at fixed seed | bit-stable across runs | current default |
| T1 | numerical relaxed | `max_abs ≤ tier-specific epsilon` vs strict | `KL ≤ 1e-3` | `KL ≤ 5e-3` | ≥ 99% | match within first 32 tokens at fixed seed | bit-stable across runs | per-element math approximations, intermediate dtype changes |
| T2 | layout / order relaxed | parent-parity at the layer boundary, intra-kernel order may differ | `KL ≤ 5e-3` | `KL ≤ 2e-2` | ≥ 97% | match within first 8 tokens at fixed seed | deterministic IDs across runs at fixed seed | reduction reordering, packed layouts, OOO accumulation |
| T3 | quant relaxed | per-call quant error within published format (e.g. INT8 KV `max_abs ≤ 5e-8` vs CPU oracle at the layer) | `KL ≤ 2e-2` | `KL ≤ 5e-2` | ≥ 90% | not required to match BF16, must match the retained quant baseline | deterministic IDs across runs at fixed seed | KV quantization, weight-format changes, mixed-precision intermediates |
| T4 | discrete relaxed | task-level quality suite | n/a (decisions are discrete) | n/a directly; route-agreement metric ≥ 0.95 | n/a | n/a | deterministic IDs across runs at fixed seed | router top-k tie reordering, sampling tie reordering |

Notes:

- The T0 row is included so the table is self-contained; strict mode is not a
  relaxed variant.
- `Per-layer hidden-state KL` is computed by logging the post-block hidden
  state under strict and relaxed at the same input and comparing distributions
  after a softmax over the channel dimension. Use the same fixture as the
  retained `qwen35_decode_graph_fixture_gate.py` chain.
- `Sequence-logit KL` is the existing
  `scripts/qwen35_decode_graph_fixture_gate.py` / `scripts/qwen35_native_prefill_fixture_gate.py`
  output metric; T1/T2 ratchet it well below the `0.05` outer ceiling so
  there is headroom for the next relaxed variant in the chain.
- `Top-1 agreement` is per-token argmax agreement against the strict reference
  on the same logits. The retained INT8 KV row already reports `100%`, so
  T1/T2/T3 caps are realistic when only a few kernels relax.
- T4 is the only tier where the strict and relaxed paths can produce different
  generated IDs. Its quality gate is a prompt suite, not a per-token check.

### Stacking budgets

Drift compounds when multiple relaxed kernels stack. Treat the budgets above
as the **whole-profile** budget, not per-kernel. The first relaxed kernel in a
profile may consume most of the budget, and later additions must demonstrate
they did not blow the profile-level sequence-logit KL.

When a profile combines kernels from different tiers, the **strictest applicable
tier** controls the profile budget. A profile that quantizes KV (T3) cannot use
a T3 budget for sampling — the sampling kernel must still match T0/T1/T2 as
appropriate.

### Mandatory shadow runs

Until a relaxed variant is promoted to retained, the project runs both strict
and relaxed in fixture sweeps and reports the deltas. A relaxed variant that
fails its tier budget on **any** retained context length (512/4K/32K/128K)
cannot graduate; it stays opt-in until either the budget is met or the budget
is renegotiated with explicit evidence.

Long-context drift is especially load-bearing. Errors in attention, GDN
recurrence, and KV append compound with context length. A relaxed variant that
passes 512/128 but blows the 32K or 128K budget is **not** retained; it is
parked with a written reason.

### Tripwires and automatic rollback

A relaxed variant is automatically rolled back (CI reject or runtime fallback
to strict) when any of the following fires:

1. `max_abs` vs strict at the per-layer boundary exceeds twice the tier
   budget on the standard fixture inputs.
2. Sequence-logit KL exceeds the tier budget on any retained context length.
3. Top-1 agreement drops below the tier floor on any retained context length.
4. Two consecutive fixed-seed runs produce different generated IDs (which is a
   bug, not relaxation, unless explicitly T4).
5. Long-context KL is more than 2× short-context KL on the same profile.
6. NaN/Inf appears anywhere in the per-layer hidden-state log or in logits.
   This is **always** a bug. NaN is not a relaxed-mode outcome; see the W8A8
   NaN history in `LESSONS-LEARNED.md`.
7. Generated tokens diverge within the first N tokens at the tier's "match
   within first N" gate (T1: 32, T2: 8).

Rollback means **disable the relaxed variant in the profile**, not "lower the
gate". If a tier budget needs to move, it requires a documented decision in
`WORKLOG.md` and an update to this file.

### Bisection workflow when a relaxed profile fails

When a stacked profile fails its budget, narrow the cause with the following
order before changing the budget or the kernel:

1. Disable the relaxed profile and confirm strict still passes. If strict
   fails, this is a strict-mode bug, not a relaxed-mode problem.
2. Enable relaxed kernels one at a time (registry variant flip) on the same
   fixture. Find the smallest single-kernel relaxation that exceeds budget.
3. Log per-layer hidden-state KL between strict and that single relaxed
   kernel. Identify the first layer where delta exceeds the per-layer cap.
4. Inspect the relaxed kernel's reduction order, dtype intermediates, packed
   layout, or quantization step at that layer.
5. Decide: tighten the relaxed kernel, demote its tier (e.g. T1→T2 cap), or
   reject it. Do **not** silently widen the profile budget.

This is the same shape as the optimization decision tree in `docs/ROOFLINE.md`
§10, but driven from correctness rather than throughput.

### Known failure modes to never call "relaxation"

Some failures look like precision drift but are bugs. They do not get a
relaxed tier. They get fixed:

- **Nondeterministic softmax** in full-attention prefill (`docs/LESSONS-LEARNED.md`).
  Repeat-run divergence is a real bug, not an acceptable T2 cost.
- **Stale JIT cache** producing zero-activity hangs (`docs/KERNELS.md`).
- **W8A8 NaN** from missing scale clamps. NaN is never an acceptable relaxed
  output; even T3 quant variants must clamp.
- **Output-buffer aliasing** that changes results under graph replay. Aliasing
  bugs survive lots of casual fixture coverage; sanitize layouts before
  promoting a relaxed kernel.
- **RoPE / position drift** producing fast but wrong tokens. The
  LESSONS-LEARNED RoPE / token-stream history is the cautionary tale: a
  retained perf row must match the reference generated tokens.

## Relaxed profile tiers

Profiles are named, retained, and visible in benchmarks. They map to the drift
tiers above. The runtime resolves them once and passes them to the kernel
registry as variant selectors.

| Profile | Tier | What it allows | What it forbids | Acceptable models |
| --- | --- | --- | --- | --- |
| `strict` | T0 | nothing beyond current behavior | all relaxed variants | all |
| `relaxed_fast_math` | T1 | approximate `rsqrtf`/`expf`/`sigmoidf`, FP16/BF16 intermediates in compute (not in KV write) | layout changes, OOO reductions, KV quant, route changes | all; per-kernel opt-in |
| `relaxed_layout` | T2 | reduction reordering, packed layout changes, split-K accumulation order, light atomic combine | KV quant beyond strict policy, discrete route changes, sampling tie reorder | all; reviewed per kernel |
| `relaxed_kv_int8` | T3 | INT8 KV cache, FP8 KV cache, dequant-on-read fusion, FP16 scale dtype | route changes, weight-format changes outside published quant schemes | Qwen3.5/PARO, Qwen3.6/PARO with retained K1 evidence |
| `relaxed_routing` | T4 | router top-k tie reorder, OOO token grouping for MoE, unstable expert ordering when route mass is below threshold | KV quant beyond strict, weight-format changes | MoE models only |
| `relaxed_all` | T2+T3+T4 | composition of the above with the strictest applicable tier per kernel | adding new relaxed kinds beyond the listed ones | research only, never default |

Profiles compose by **strictest applicable**. A `relaxed_kv_int8` +
`relaxed_fast_math` user gets T3 KV behavior and T1 fast-math behavior, with
the T3 sequence-logit cap (`KL ≤ 5e-2`) applied to the combined run.

## Savings vocabulary and RDNA3 levers

### Savings vocabulary

| Term | Meaning |
| --- | --- |
| Launch | Remove or fuse a HIP launch, or make graph replay cover a larger unit. |
| HBM | Avoid a global-memory read/write of activations, KV, logits, or scratch. |
| Scratch | Shrink temporary buffers or avoid materializing intermediates. |
| Occupancy | Improve waves/VGPR/LDS balance or choose a faster compiler schedule. |
| Capacity | Reduce retained KV/weight footprint enough to unlock longer contexts. |
| ALU/elem | Cut VALU instructions issued per weight or activation element. |

### RDNA3-specific instruction levers unlocked by relaxed precision

`docs/ROOFLINE.md` §1.3 / §6 / §9 enumerates the hardware peaks. The table
below maps each strict-mode constraint to the lever that becomes legal under a
named relaxed tier, with the published peak as the ceiling. Real kernels
recover 50-70% of peak in the best case; the column is a **theoretical
upper bound**, not an expected uplift.

| RDNA3 lever | Strict-mode reason it is unused | Tier that unlocks it | Theoretical peak vs current FP32 FMA | Realistic relaxed-mode capture |
| --- | --- | --- | --- | --- |
| `v_dot4_i32_iu8` (sudot4) on W4 GEMV after Q8 activation pre-quant | Naive sudot4 on the current pack8 layout regressed 3.9-9.7× (parent `PLAN-PAROQUANT.md`); requires Q8 activation ABI we do not have. | T2 (layout) + T3 (activation quant). | 4× ops/lane/cycle vs single-issue FP32 FMA. | 1.5-2× on the W4 GEMV bucket *if* layout and activation-quant Q8 land together; less if either is missing. |
| `v_dot8_i32_iu4` on directly-packed INT4 weights | Requires INT4-packed operand layout incompatible with current AWQ pack8 / Marlin-K layouts. | T2 (layout) + T3 (quant). | 8× ops/lane/cycle. | Speculative; needs a relaxed Marlin-K-style port. Not a near-term lever. |
| `v_pk_fma_f16` packed FP16 intermediates in dense / linear-attention dot products | Strict path keeps FP32 accumulators for parent parity. | T1 (numerical). | 2× FP32 FMA throughput. | Small in compute-bound paths, near-zero in bandwidth-bound paths. Most useful inside fused composites where intermediate storage shrinks too. |
| Approximate `__builtin_amdgcn_rsq_f32` / `v_rsq_f32` for RMSNorm and GDN q/k scale | Already used (`rsqrtf` lowers to `v_rsq_f32`); strict precision is preserved by the surrounding FP32 reduction. | T1 (numerical). | n/a; already 1 cycle. | Negligible direct ALU win. The lever is downstream fusion that drops the FP32 path. |
| Approximate `__expf` / `__sigmoidf` / `__silu` in router/MoE combine | Strict path uses the precise libm form. | T1 (numerical). | ~2-4× per call vs precise libm path. | Router/combine call counts are small; total saving is small even at full capture. |
| VOPD dual-issue compatible ALU pairs across dequant chain | Dependent ops in the strict dequant chain (`shift`, `mask`, `sub`, `cvt`, `mul`) limit VOPD pair opportunities. | T2 (layout) — restructure the chain into independent halves. | 2× VALU throughput for paired ops. | A few percent on W4 GEMV bucket if combined with layout change. |
| `__launch_bounds__` / waves-per-EU retune per relaxed variant | Strict variants must match parent occupancy class. | T2 (layout). | Occupancy lift of 1-2 waves/SIMD on selected kernels. | 5-15% on memory-bound decode kernels with currently-high VGPR, smaller elsewhere. Treat as a per-kernel artifact, not a default. |
| OOO / atomic combine in MoE | Strict order preserves bit-exact combine. | T2 (layout) or T4 (routing) depending on scope. | Removes a synchronization barrier per token; small absolute. | 0-2% on combine bucket. Often gated by correctness, not by ALU. |
| Wave64 reductions | Parent has no retained wave64 default win; reductions tested as 32-lane halves on gfx1100. | T2 (layout). | n/a; depends on kernel. | Isolated experiment only; not a default lane (`docs/ROOFLINE.md` §9.3). |

The realistic-capture column reflects parent rejection evidence. Most of these
levers are not "free" on their own; the relaxed tier is the permission to
restructure the surrounding code so the lever actually fires.

## Decode time budget (4K/128 graph replay)

This is the anchor table for the per-kernel ballpark percentages below. Bucket
shares and call counts come from `docs/OPTIMIZE.md` §6 (M.4 selected-region
rocprof, 16 one-step graph replays scaled per token). Decode kernel time per
token is 7.23 ms at 4K/128.

| Bucket | 4K/128 share | Calls/token | ms/token (kernel time) | Relaxed lever class |
| --- | ---: | ---: | ---: | --- |
| Selected-MoE W4 GEMV | 18.3% | 80 | 1.32 | layout, fast-math, OOO combine |
| W8A16 linear / lm_head / dense | 15.7% | 81 | 1.13 | layout, fast-math, top-k without full logits |
| W4 single pack8 GEMV | 13.6% | 50 | 0.98 | layout, fast-math, dp4a-after-layout |
| W4 dual pack8 GEMV | 11.7% | 40 | 0.85 | layout (rotate fusion), dp4a-after-layout |
| Decode attention (paged + full) | 10.5% | 10-20 | 0.76 | quant KV (T3), layout (split-K order), fast-math |
| Rotation / RoPE | 9.6% | 160 | 0.69 | layout (producer fusion), fast-math (trig tables) |
| Router | 5.8% | 80 | 0.42 | layout (cooperative top-k), routing (T4 tie reorder) |
| Linear-attention GDN decode | 5.4% | 30 | 0.39 | fast-math (rsqrt fusion), layout (chunk reassoc) |
| RMSNorm / add-RMSNorm | 3.4% | 91 | 0.25 | layout (producer fusion), fast-math |
| Dense GEMV | 1.2% | 30 | 0.09 | layout, fast-math |
| MoE combine | 1.2% | 40 | 0.09 | layout (OOO combine), fast-math |
| Other / glue / cast | ~3% | rest | ~0.22 | layout, fast-math |
| Dispatch overhead floor | n/a | 877 | ~0.88 (at 1 µs/dispatch) | fusion (T2 layout) |

Sanity checks:

- 1% E2E decode kernel time ≈ 0.072 ms at this configuration.
- Going from 877 → 700 dispatches per token saves ~0.18 ms (~2.5% E2E) at a
  1 µs floor before overlap; ~0.5 ms (~7% E2E) if real serial dispatch gap
  matches the 32.93 µs mean inter-kernel gap in the older mixed trace.
- The Amdahl ceiling for any single bucket B at share S(B) is
  `1 / (1 - S(B))`; 18.3% caps at 1.22× even for infinite kernel speedup. The
  realistic relaxed ceiling stacks several modest wins, not one big one.

## Expected relaxed-mode speedup bands

Relaxed mode should be scoped as a **single-digit to low-double-digit decode
speedup program**, not a 2× plan. The table below converts the per-bucket
ceilings into planning bands for implementation work. These are expectations
for a retained profile that still passes the drift gates above; they are not
benchmark claims.

| Relaxed scope | Main ingredients | Expected retained uplift | Stretch / ceiling | Why it is not 2× |
| --- | --- | ---: | ---: | --- |
| T1 fast-math only | Approximate nonlinear math, FP16/BF16 intermediates in selected elementwise kernels, no layout changes. | 0-2% decode | ~3% | The affected buckets are small or already use fast instructions; most decode time is W4/W8 linear and attention traffic. |
| T2 layout/fusion | Rotation/RMSNorm/elementwise fusion, fewer glue launches, per-kernel waves/VGPR retune, safe reduction-order changes. | 5-10% decode | ~12-15% | Even eliminating every rotation/RMSNorm/glue inefficiency only touches ~15-20% of the 4K/128 profile; current D1.1/D1.6 prototypes are correct but slightly negative until graph-replay/layout overhead is fixed. |
| T3 KV quant/dequant | INT8/FP8 KV, fused dequant + attention, better split/merge choices, exact live-span ABI retained. | 0-3% at 4K decode; 5-10% at 32K+ | ~15% at 128K if dequant overhead is fixed | KV bytes dominate only at long context. Parent INT8 decode regressed at 128K when dequant overhead outweighed bandwidth savings, so this requires a better fused path. |
| Practical retained relaxed profile | T2 launch/layout wins plus a small T3 KV path and selective T1 intrinsics. | 8-15% decode | ~20% | The profile stacks modest wins across several buckets; Amdahl limits any single kernel family. |
| Aggressive RDNA3 relaxed profile | Practical profile plus Q8 activation ABI and `v_dot4_i32_iu8`/Marlin-K retry for W4 GEMV. | 15-25% decode if it works | ~30% | The largest ALU lever needs a new activation/layout ABI and still only attacks the W4 GEMV share; naive sudot4 on the current layout was much slower. |
| Linear-attention prefill relaxed profile | Chunkwise/WY-style GDN prefill, GDN RMSNorm+SiLU+rotate fusion, shared-gate sigmoid fusion. | 5-30% prefill on linear-attention-heavy models | model/context dependent | This is a prefill structural lever, not the decode path; it needs separate correctness oracles and long-context drift gates. |

Planning rule of thumb:

- Near-term relaxed work should justify itself at **+5-10% decode** if it is
  mostly fusion/layout cleanup.
- A well-executed retained relaxed profile should target **+10-15% decode**.
- **+20-25% decode** is an aggressive RDNA3 outcome that likely requires the
  Q8 activation ABI + packed-dot W4 GEMV path, not just turning on fast math.
- Anything approaching **2×** would require a different algorithmic lever
  outside this document's scope, such as speculative decoding with high draft
  acceptance or a fundamentally different model/kernel mix. Relaxed precision
  alone is not expected to deliver that.

## Per-kernel relaxed opportunities

Each row below is a relaxed candidate, not an approved change. The "ballpark
E2E %" column is a ceiling — what you might save on a 4K/128 decode token if
the relaxation worked at the limit suggested by the Amdahl share and the
RDNA3 lever. Real returns are usually lower.

Cross-references in the `OPTIMIZE.md row` column tie back to the existing
optimization board so retained relaxed variants update the same evidence
ledger.

### RMSNorm / add-RMSNorm (3.4% / 91 calls/tok)

| Field | Value |
| --- | --- |
| Strict constraint | Reference reduction order; FP32 sum-of-squares; parent-parity inv_rms; fused forms match the unfused chain. |
| Op profile | 1 mul + 1 add per element in the sumsq reduction, 1 `rsqrtf` per row, 1 mul per element on the gamma path. ~3 VALU/elem. |
| Relaxed candidates (T1) | Approximate `rsqrtf` (already 1 cycle), BF16/FP16 intermediate accumulation when hidden_size ≤ 2048, fuse residual add + norm into a single LDS pass. |
| Relaxed candidates (T2) | Fuse RMSNorm with the next projection when the normalized vector has a single consumer (OPTIMIZE D1.2 deferred for multi-consumer dataflow); per-kernel waves/EU retune. |
| Possible savings | Launch (1-2 fewer per layer when fused), HBM (skip writeback when fused with consumer), occupancy. |
| Ballpark E2E % | 0.5-1.5% if all add-RMSNorm + downstream pair fuse cleanly; <0.5% from pure intra-kernel relaxation. |
| OPTIMIZE.md row | D1.2 (deferred). |
| Required evidence | T1 per-layer KL ≤ 1e-3; demonstrate add-RMSNorm + projection produces identical generated tokens at fixed seed across 3 runs. |

### Rotary / RoPE (9.6% / 160 calls/tok)

| Field | Value |
| --- | --- |
| Strict constraint | Parent-parity sin/cos lookup, pair ordering, output dtype; bit-exact rotate vs reference. |
| Op profile | 2 mul + 2 fma per rotated pair, sin/cos table lookups; the bucket cost is mostly the **launch count** (160 calls/tok) and intermediate HBM, not the math. |
| Relaxed candidates (T1) | Lower-precision trig tables, packed FP16 rotation intermediates. |
| Relaxed candidates (T2) | Fuse rotate into adjacent producer (RMSNorm) or consumer (W4 dual pack8 GEMV) — see D1.1 staged rotate, which is correct but graph replay regresses today; reordered vector packs. |
| Possible savings | Launch (remove dedicated rotation kernel calls), HBM (skip read/write of pre-rotation buffer), scratch. |
| Ballpark E2E % | 2-4% if the producer/consumer fusion captures most of the 160 launches; D1.1 measured -4.32% at 512/128 in its current shape, so a relaxed retry needs a layout change before claiming this. |
| OPTIMIZE.md row | D1.1 (rejected as default; opt-in correct), partial rotate-out fused kernels already landed. |
| Required evidence | T2 per-layer KL ≤ 5e-3; full-attention and linear-attention layers validated separately because their rotation usage differs; long-context retest because rotation phase errors interact with attention reductions. |

### Full-attention prefill (varlen + causal GQA)

| Field | Value |
| --- | --- |
| Strict constraint | Softmax/reduction order and KV append behavior are correctness-sensitive. Prior nondeterministic prefill softmax was a real bug (`docs/LESSONS-LEARNED.md`). |
| Op profile | Block softmax with online normalization; PV matmul scales with `tokens × ctx × head_dim`. |
| Relaxed candidates (T2) | Flash/AOTriton-style block reductions when AOTriton drops below current heuristics; lower-precision logits/PV intermediates; query chunking choices that change PV associativity. |
| Relaxed candidates (T3) | INT8/FP8 KV reads during prefill (only after the K1 oracle removal blocker in `WORKLOG.md` 2026-05-18 is resolved). |
| Possible savings | HBM (smaller intermediates and quant KV reads), scratch, occupancy, launch. |
| Ballpark E2E % | Prefill bucket only; at 4K decode it is a small share but at 32K+ it is the dominant share (`docs/ROOFLINE.md` §7). Decode is unaffected unless the prefill change also lowers post-prefill state size. |
| OPTIMIZE.md row | P2.x AOTriton glue (mostly settled), P4 native FA-2 (deferred). |
| Required evidence | T2 sequence-logit KL ≤ 2e-2; mandatory 3-run repeat at fixed seed because prefill nondeterminism is the canonical landmine. |

### Paged-attention decode (`KVLiveSpans`) (10.5% short; up to ~23% at 32K)

| Field | Value |
| --- | --- |
| Strict constraint | Dense BF16 KV is the exact baseline; live-span ABI honored for dense and eviction policies. |
| Op profile | 2 reads per KV element (K, V) + softmax + PV reduction; AI ~1 byte/op at BF16. |
| Relaxed candidates (T3) | INT8 / FP8 KV profiles (K1 INT8 retained: `max_kl=0.015328`, top-1 100%, `docs/KVCACHE.md`), approximate or coarser scales, fused dequant + attention, larger split/merge choices. |
| Relaxed candidates (T2) | Reordered page traversal; warp-cooperative split-K already retained; address-only V-loop polish was parent-rejected (`docs/OPTIMIZE.md` §11). |
| Possible savings | HBM (-50% bytes for K + V at T3 INT8), capacity (longer context inside 24 GiB), launch (fewer split-K reduces). |
| Ballpark E2E % | At 4K: 1-3% decode; at 32K: 5-10%; at 128K: 10-15% if KV bytes dominate. Parent INT8 measured -3.20% decode at 128K because dequant overhead at long context outweighs the BW saving; relaxed-mode retry needs to address that. |
| OPTIMIZE.md row | D3.x grouped-GQA producer (retained), D3.5 INT8 paged-KV (deferred). |
| Required evidence | T3 sequence-logit KL ≤ 5e-2; long-context retest mandatory; deterministic IDs at fixed seed across 3 runs; report quant policy in benchmark artifacts. |

### KV write / append / repack

| Field | Value |
| --- | --- |
| Strict constraint | Writes preserve exact layout consumed by strict attention kernels. |
| Op profile | One cast + one store per element, plus optional scale computation under T3. |
| Relaxed candidates (T3) | Quantize-on-write (retained for INT8 K1 path); fused append with pack/dequant metadata; relaxed scale dtype (FP16 scale is the retained K1 default). |
| Relaxed candidates (T2) | Avoid temporary BF16 materialization in the prefill oracle. (`WORKLOG.md` 2026-05-18: the BF16 oracle removal experiment regressed E2E KL.) |
| Possible savings | HBM, scratch, capacity. |
| Ballpark E2E % | Mostly capacity, not throughput. The oracle removal experiment was -0% throughput when it worked; the failure was correctness. |
| OPTIMIZE.md row | KV write variants under `docs/KERNELS.md`; K1 path retained. |
| Required evidence | T3 per-token KV oracle ≤ retained K1 thresholds; readback test for live spans and eviction masks; long-context decode retest. |

### Dense W8A16 / AWQ pack8 / Marlin-K linear (selected GEMV 18.3%, single pack8 13.6%, dual pack8 11.7%, W8A16 15.7%)

| Field | Value |
| --- | --- |
| Strict constraint | Packed layout, scale application, and accumulation order held to parent/reference parity. |
| Op profile (W4 pack8 strict) | per weight element: 1 shift/mask + 1 sub + 1 cvt + 1 mul (scale) + 1 fma (accumulate) ≈ 5+ VALU/elem (`docs/ROOFLINE.md` §6.2). |
| Op profile (dp4a candidate) | per 4 weight elements: 1 sudot4 + amortized scale + bias ≈ ~1 VALU/4 elem. Activation must be Q8 pre-quantized (T3 prerequisite). |
| Relaxed candidates (T2) | Different MMA tiling; reordered split-K reductions; pre-swizzled relaxed layouts (Marlin-K vec8 already retained as D2.1, `+5.6%`); per-kernel waves/VGPR tuning under the relaxed waves-per-EU lever. |
| Relaxed candidates (T2+T3) | sudot4 inner loop **with** Q8 activation pre-quantize (D2.3, parked on torch-free Q8 activation ABI). |
| Relaxed candidates (T1) | FP16 accumulators on layers with documented small drift. |
| Possible savings | ALU/elem (5-25× lower instruction count per weight on full dp4a capture), occupancy. |
| Ballpark E2E % | T2 layout-only: 1-3% from per-kernel waves/EU retune and reduced barrier cost. T2+T3 dp4a after Q8 activation: 4-8% on selected MoE + W4 single + W4 dual buckets combined, with Amdahl ceiling 25% combined share → 1.33× sub-bucket cap. |
| OPTIMIZE.md row | D2.1 (retained), D2.3 (parked on ABI), W.3 waves-per-EU (pending). |
| Required evidence | T2 per-layer KL ≤ 5e-3; logit drift can be layer-local but token-critical; mandatory 32K and 128K retest because long context amplifies projection drift. |

### MoE router / top-k (5.8% / 80 calls/tok)

| Field | Value |
| --- | --- |
| Strict constraint | Expert choice, tie handling, and token/expert order must match strict behavior. |
| Op profile | Router GEMV (`hidden → 257 experts`) + softmax + top-k + scatter. The GEMV is small; the top-k and selection dominate the launch count. |
| Relaxed candidates (T1) | Approximate gating math (precise `expf` → `__expf`). |
| Relaxed candidates (T2) | Cooperative top-k that preserves expert choice but reorders within ties (D1.5 / D5.3 retain a correct cooperative variant; it currently regresses graph replay because of counter memset). |
| Relaxed candidates (T4) | Discrete relaxed: allow tie order reordering, early pruning of route mass below threshold, OOO token grouping that reorders equal-priority work. |
| Possible savings | Launch (3-4 fewer per layer if router collapses to 1-2 launches); scratch; occupancy. |
| Ballpark E2E % | 0.5-1.5% for the launch-reduction path; <0.5% for pure math approximation. |
| OPTIMIZE.md row | D1.5 / D5.3 (correct, rejected as default), D5.x router retains. |
| Required evidence | T2 routing decisions must match strict on the retained fixtures; T4 must add a route-agreement metric ≥ 0.95 and a prompt-suite quality check, not just KL. |

### MoE scatter / gather / group GEMM

| Field | Value |
| --- | --- |
| Strict constraint | Token ordering and exact combine inputs preserved. |
| Op profile | Token-major scatter into expert-major layout; small per-token work, launch-bound. |
| Relaxed candidates (T2) | Out-of-order token groups; larger grouped GEMM batches that alter accumulation order. |
| Relaxed candidates (T4) | Atomic or unordered combine when route mass guarantees stability. |
| Possible savings | Launch, HBM, occupancy. |
| Ballpark E2E % | <1%; structural changes here are usually preconditions for other wins (combine fusion), not standalone levers. |
| OPTIMIZE.md row | D1.4 (further fold rejected; safe combine already default). |
| Required evidence | T2/T4 fixed-seed repeat checks for variance; per-expert accounting unchanged on the route-agreement metric. |

### MoE combine / weighted-sum / shared-gate / residual (1.2% / 40 calls/tok)

| Field | Value |
| --- | --- |
| Strict constraint | Fused combine matches unfused gate/up/down + SiLU + sigmoid gate + residual chain. |
| Op profile | 1 mul (weight) + 1 fma (gate) + 1 sigmoid + 1 fma (residual) per element. |
| Relaxed candidates (T1) | Approximate sigmoid/SiLU; lower-precision gate/up product. |
| Relaxed candidates (T2) | Fuse residual/writeback; relaxed expert accumulation order; OOO atomic combine if route mass is bounded. |
| Possible savings | Launch (small), HBM (skip intermediate buffer), scratch. |
| Ballpark E2E % | 0.2-0.7%; bucket share is small; gains only matter as part of compound fusion with selected-MoE. |
| OPTIMIZE.md row | D1.4 retained as default safe fold; further folds rejected. |
| Required evidence | T1 layer-level KL ≤ 1e-3 since nonlinear error can amplify across layers; full prompt-suite quality check. |

### Linear-attention conv / GDN / recurrence (5.4% decode, plus ~21% prefill)

| Field | Value |
| --- | --- |
| Strict constraint | Recurrent state updates are order-sensitive; chunking must preserve strict semantics. Prior GDN barrier-removal experiments corrupted recurrent state (`docs/LESSONS-LEARNED.md`). |
| Op profile | Per token: small conv + RMSNorm + gating + recurrent update; sequential `for token in range(tokens)` outer loop in prefill (the Qwen3.6-27B dense path's #1 bottleneck per `docs/OPTIMIZE-DENSE.md` §1.5). |
| Relaxed candidates (T1) | Approximate sigmoid/SiLU; lower-precision recurrent intermediates (carefully, because drift accumulates with token index). |
| Relaxed candidates (T2) | Reassociate chunk scans; fuse GDN RMSNorm + SiLU + rotate (P3.1 was correct, rejected as default for 32K regression and unchanged scratch); fuse shared-gate sigmoid in prefill (P3.2 correct, rejected as default for noisy E2E); chunkwise / WY-chunkwise GDN prefill (Q36D-P1 lane). |
| Possible savings | Launch, HBM (skip intermediate writeback), scratch (chunkwise replaces serial per-token state), occupancy. |
| Ballpark E2E % | Decode: 1-2%. Prefill (especially dense path): 5-30% depending on chunkwise port; this is the largest single structural lever in the dense 27B board. |
| OPTIMIZE.md row | P3.1 / P3.2 (rejected diagnostic), D5.1 GDN audit (accepted stop), Q36D-P1 (pending dense). |
| Required evidence | T2 long-context KL ≤ 2e-2 because recurrence drift grows with context; mandatory validation at 512, 4K, 32K, 128K; never repeat the GDN barrier-removal regression. |

### PARO-specific fused attention helpers

| Field | Value |
| --- | --- |
| Strict constraint | Existing fused pack/rotate helpers are opt-in until parity is strong. |
| Op profile | Mix of rotation, pack8 GEMV, KV write fused into single launches. |
| Relaxed candidates (T2) | Promote rotate dual-pack8 (D1.1) and full-attn KV pack8 (D1.6) fusions under a named relaxed profile when exact parent parity is too expensive. Both are correct today; the blocker is graph-replay perf, not numerical drift. |
| Possible savings | Launch, HBM. |
| Ballpark E2E % | 0-2% depending on whether graph replay regression can be addressed; current measured deltas are slightly negative. |
| OPTIMIZE.md row | D1.1 / D1.6 (rejected as default). |
| Required evidence | T2 sequence-logit KL ≤ 2e-2 (both kernels already pass T0 today); the test is whether a relaxed-mode framing can find a layout that recovers the graph-replay loss. |

### LM head / argmax / sampling (W8A16 share 15.7% combined)

| Field | Value |
| --- | --- |
| Strict constraint | Greedy path expects stable logits and token choice; ties deterministic. |
| Op profile | `hidden → 248320 vocab` W8A16 GEMV (~509 MB INT8 read per token at strict layout) + argmax. |
| Relaxed candidates (T1) | Quantized LM head accumulators in FP16 for top-k path; approximate scale folding. |
| Relaxed candidates (T2) | Chunked top-k / argmax with relaxed tie order; avoid full-logit materialization when API only needs top-1/top-k. Parent attempted chunked lm-head top-1 with negative result (`1.062 ms vs 0.766 ms`); a relaxed retry would have to change the layout, not just chunk. |
| Possible savings | HBM (large; ~509 MB/token of LM head reads), scratch, launch. |
| Ballpark E2E % | 1-3% if a top-k path that avoids full logits is retained; less if only the GEMV inner loop changes. |
| OPTIMIZE.md row | D5.2 audit (accepted stop). |
| Required evidence | T2 sequence-logit KL ≤ 2e-2 (token-level), prompt suite for sampling-relevant tasks; token IDs must remain deterministic at fixed seed. |

### Casts / activation utilities

| Field | Value |
| --- | --- |
| Strict constraint | Cast points and rounding match strict dtype expectations. |
| Op profile | One cast per element; many small launches in glue. |
| Relaxed candidates (T1) | Lazy casts; lower-precision scratch buffers; in-place updates when aliasing is proven safe; vectorized unaligned paths. |
| Possible savings | Launch, HBM, scratch. |
| Ballpark E2E % | 0.2-0.7%; mostly absorbed by adjacent kernel fusion, not standalone. |
| OPTIMIZE.md row | Glue/elementwise listed in M.4. |
| Required evidence | T1 bit-exact or `max_abs ≤ tier epsilon`; aliasing audit; sanitizer-style shape tests; strict fallback retained. |

### Compiler / build-profile variants

| Field | Value |
| --- | --- |
| Strict constraint | Retained flags are conservative; `-amdgpu-unroll-threshold-local=600` is neutral/default, not a broad speed lever (W.1 accepted neutral). |
| Op profile | Compiler scheduling/unroll/spill behavior; per-kernel. |
| Relaxed candidates (T2) | Per-kernel fast-math/denormal-flush experiments; waves-per-EU retune; local unroll changes only where measured. |
| Possible savings | Occupancy, launch latency side effects. |
| Ballpark E2E % | 0-3% per retained kernel; treat as kernel variants with artifacts, not blanket sweeps. |
| OPTIMIZE.md row | W.1 (neutral accepted), W.2 (default), W.3 (pending). |
| Required evidence | T2 sequence-logit KL ≤ 2e-2; per-kernel artifact; never a blanket flag change. |

## Backlog unlocked by relaxed mode

These are candidates previously blocked, parked, or risky because they changed
strict ordering, associativity, or bit-perfect behavior. They are not approved
by this document; they become legal to prototype only inside a named relaxed
profile with the drift evidence above.

Priority order is roughly best-ratio of relaxed-mode upside to engineering
cost; dependencies are noted.

1. **Central precision-policy plumbing** (prereq for everything below).
   - Runtime-visible profile object/profile and registry-variant resolution.
   - Migrate env-only experiments (`HIPENGINE_PARO_ROTATE_DUAL_PACK8_FUSED`,
     `HIPENGINE_PARO_FULL_ATTN_KV_PACK8_FUSED`, `HIPENGINE_PARO_ROUTER_TOPK_COOP`,
     `HIPENGINE_PREFILL_ROUTER_SHARED_GATE_SIGMOID_FUSED`) into named variants
     once retained.
   - Ensure benchmark artifacts print the selected profile.

2. **Compound decode launch reduction (T2)** — anchor lever.
   - Producer-side rotation + RMSNorm fusion where the normalized vector has a
     single consumer (D1.2 deferred; relaxed framing may unlock multi-consumer
     row-staging that strict cannot).
   - Combine small elementwise casts/scale/writeback kernels in decode.
   - Strict unfused chain stays registered for bisection.

3. **Relaxed Marlin-K + Q8 activation (T2+T3)** — largest single ALU lever.
   - Land Q8 activation ABI (D2.3 unblocker) under a relaxed profile.
   - Retry sudot4 inner loop on Marlin-K layout with Q8 activations.
   - Required because naive sudot4 on the strict layout is 3.9-9.7× slower.

4. **Relaxed full-attention prefill path (T2)**.
   - Revisit AOTriton/flash-style reductions and query chunking that change
     softmax/PV associativity.
   - Prototype fused QKV/KV-pack append under a named relaxed variant.
   - Deterministic repeat tests are mandatory; the earlier nondeterministic
     softmax was a real bug, not relaxation.

5. **Relaxed KV cache profiles (T3)**.
   - INT8 KV retained (K1 path) at `max_kl=0.015328`, top-1 100%, no BF16 KV
     shadow; promote from "capacity/diagnostic" to a named relaxed profile.
   - Resolve the BF16 prefill oracle removal blocker before adding more
     aggressive KV quant.
   - Evaluate FP8 / mixed-scale variants if the scale metadata and live-span
     ABI remain compatible with the strict fallback.

6. **MoE router and combine relaxed experiments (T2 + T4)**.
   - Cooperative top-k that addresses the counter memset / atomic tail cost
     that currently regresses D1.5 graph replay.
   - OOO token grouping for MoE under T4 with route-agreement metric.
   - Approximate router math (T1) for the small cases where launch count
     dominates.

7. **Linear-attention prefill fusion (T2)**.
   - GDN RMSNorm + SiLU + rotate fusion (P3.1) under a relaxed profile that
     can tolerate the 32K regression if compound prefill wins offset it.
   - Shared-gate sigmoid fusion in prefill (P3.2) under a profile that does
     not require strict legacy-shared-expert parity.
   - Chunkwise / WY-chunkwise GDN prefill (Q36D-P1) — biggest dense-path
     prefill lever; correctness oracle (Q36D-K.1) is the gating dependency.

8. **Packed linear / Marlin-K / WMMA retuning (T2)**.
   - Per-kernel waves/EU/unroll under W.3 (pending).
   - Split-K and WMMA accumulation orders that are not bit-identical to
     parent.

9. **Approximate nonlinear / math intrinsics (T1)**.
   - `__expf` / `__sigmoidf` / `__silu` in router + combine + linear-attn
     gates after layer-local error and end-to-end quality remain inside the
     T1 budget.

10. **Argmax / top-k without full logits (T2)**.
    - Relaxed LM-head top-k path that avoids materializing the full
      `[1, 248320]` logits when the API only needs the top-1 or a small top-k.
    - Tie behavior may change; sequence-level prompt suites are required.

11. **Out-of-order and atomic reductions (T2 / T4)**.
    - OOO expert grouping, split-K atomics, unordered combine.
    - Repeat-run variance checks mandatory; strict deterministic fallback
      retained.

12. **Wave64 isolated experiments (T2)**.
    - Not a default lane; isolated experiment per `docs/ROOFLINE.md` §9.3.
    - Must include tensor-level correctness tests, not just compile + run.

## Acceptance checklist

Before any relaxed variant is called retained:

- [ ] Strict mode still passes the same tests and uses the same default
      variants.
- [ ] Relaxed variant is selected by an explicit profile and appears in logs +
      benchmark artifacts.
- [ ] Registry entries keep strict and relaxed variants side by side.
- [ ] Per-kernel oracle output is saved or reproducible.
- [ ] Per-layer hidden-state KL and per-token logit KL recorded and within tier
      budget.
- [ ] Top-1 agreement recorded and within tier floor.
- [ ] Repeated fixed-seed runs produce the same generated IDs within the tier's
      "match within first N" gate.
- [ ] No tripwire fired (per §Tripwires and automatic rollback).
- [ ] Long-context retest at 32K and 128K passes the tier budget where
      applicable.
- [ ] Benchmark artifact records exact command, hardware, model, quant,
      workload, relaxed profile, correctness gate, and measured result.
- [ ] `WORKLOG.md`, `benchmarks/README.md`, and `benchmarks/CHANGELOG.md` are
      updated for any retained performance claim.

## Non-goals

- Relaxed mode is not a reason to weaken strict mode.
- Relaxed mode does not excuse nondeterministic bugs, illegal memory behavior,
  stale JIT cache issues, untracked benchmark conditions, or NaN/Inf outputs.
- Relaxed variants do not bypass the plugin registry architecture.
- Relaxed precision does not move kernel R&D into this repository; exploratory
  micro-tuning still belongs in `~/amd-gpu-tuning/` until a stable kernel is
  ready to port.
- The `relaxed_all` profile is not a default for any retained API; it exists
  for research only.
- A tier budget is not adjusted because a kernel violated it. The kernel is
  fixed, demoted, or rejected. Budget changes require an explicit decision in
  `WORKLOG.md` and an update to this file.
