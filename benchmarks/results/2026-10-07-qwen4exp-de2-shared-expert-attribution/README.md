# DE-2 shared expert: decode attribution by dispatch signature

Splits the replayed MoE-graph family of the Qwen4Exp decode capture into tensor owners and
measures the shared expert's own decode cost. Correctness/attribution evidence;
`performance_claim: false`.

- **Script:** `scripts/qwen4exp_de2_shared_expert_attribution.py`
- **Renamed capture:** `benchmarks/results/2026-09-29-qwen4exp-x4-decode-split/rocprof/x4full_*`
  (`--kernel-rename`; every in-window kernel carries a `qwen4exp_role:` name)
- **Real-name twin:** `benchmarks/results/2026-09-29-qwen4exp-de1-dp4a-e2e/rocprof/e2ebase_*`
  (`renamed = 0`; same decode protocol, same 5,313-kernel decode selection)
- **Model:** unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL`, 48 layers
- **Source:** committed traces + read-only GGUF header parse; no GPU, no model load

## What changed in the DE-2 picture

The DE-2 row ended: *"a scoped marker or a name-to-kernel join that this capture cannot
supply"*, and a later note concluded the join is *"structurally impossible"*. That is true of
the renamed X-4 export only. A **real-kernel-name twin of the same decode protocol is
committed** (`e2ebase_kernel_trace.csv`, `renamed = 0`), so the join is available.

What name alone cannot do is separate the shared expert, because the whole per-layer MoE runs
as **one replayed HIP graph**: the harness snapshot shows `graph_before.capture = 48,
replay = 0` and `graph_after.replay = 144`, so the measured decode reps replay 48 graphs per
step and the nested Python `launch_gguf_linear` role markers never execute. Every node in the
layer graph therefore inherits the enclosing `moe:layers.N.expert_gate` role. This script
splits the nodes by **dispatch signature** `(Grid_Size_X, Workgroup_Size_X, VGPR, LDS)`,
attaches the real kernel name from the twin (restricted to graph-replayed rows - those with no
HIP launch API row), and classifies each signature against the exact GGUF payload shapes.

## The split

Decode kernel sum **54.2181 ms/step**; replayed MoE-graph family **24.5504 ms/step** over
583 kernels/step. The family decomposes exactly (sum = 24.5504):

| role | ms/step | kernels/step | % of moe | % of decode |
|---|---:|---:|---:|---:|
| routed gate+up (Q4_K) | 14.0126 | 47.0 | 57.08% | 25.84% |
| routed down (Q5_1) | 5.4436 | 43.0 | 22.17% | 10.04% |
| routed selected misc | 1.3444 | 7.0 | 5.48% | 2.48% |
| **shared gate+up (Q8_0, out=640)** | **1.2939** | 96.0 | 5.27% | 2.39% |
| router GEMV (F32, out=512) | 1.2389 | 48.0 | 5.05% | 2.29% |
| **shared down (Q8_0, out=2560)** | **0.5185** | 48.0 | 2.11% | 0.96% |
| router select | 0.3310 | 48.0 | 1.35% | 0.61% |
| **shared gate weight (F32, out=1)** | **0.1239** | 48.0 | 0.50% | 0.23% |
| convert / weighted-sum (mixed) | 0.1164 | 101.0 | 0.47% | 0.21% |
| **shared combine** | **0.0671** | 48.0 | 0.27% | 0.12% |
| **shared silu_mul** | **0.0587** | 48.0 | 0.24% | 0.11% |
| misc | 0.0014 | 1.0 | 0.01% | 0.00% |

**Shared expert total: 2.0621 ms/step = 3.80% of the decode kernel sum.** The weight-bearing
part (gate+up+down+gate weight) is **1.9363 ms/step**; the elementwise remainder (silu_mul +
combine) is 0.1258. The `moe_convert_mixed` bucket (f32_to_bf16 + weighted_sum_out) is shared
between the shared down conversion and the routed weighted sum and is reported separately.

## Why the split is sound

- **The shared expert's three Q8_0 GEMMs are the only `out=640` Q8_0 tensors in the model.**
  The GGUF has exactly 96 `[2560, 640]` Q8_0 tensors (`ffn_gate_shexp`, `ffn_up_shexp`), and
  the replayed graph contains exactly 2 pack8 `out=640` nodes per layer = 96/step. The
  `out=2560` graph node (1/layer = 48/step) is `ffn_down_shexp` (`[640, 2560]` Q8_0).
- **The gate weight is the only `out=1` GEMV** (`ffn_gate_inp_shexp`, `[2560]` F32), and the
  router is the only `out=512` GEMV (`ffn_gate_inp`, `[2560, 512]` F32; 251.66 MB/step).
- **Three independently published numbers reproduce as self-checks:** routed gate+up 14.0126
  vs the row's 14.013, routed down 5.4436 vs 5.444, and shared combine 0.0671 vs the row's
  0.067 ms/step.
- **The measured shared cost sits inside the row's own bracket** `[1.36, 2.544] ms/step`
  (2.0621), near its upper half.

## What this does to the row

- The attribution gap is closed: the shared expert's decode cost is measured at
  **2.0621 ms/step (3.80%)**, not merely bracketed.
- The fused gate+up action targets `shared_gate_up` **1.2939 ms/step** for 167.44 MB
  (129.4 GB/s) while `shared_down` runs 0.5185 ms/step for 83.72 MB (161.5 GB/s); gate+up is
  the slower of the two, consistent with the row's proposed fusion target.
- The hard ceiling for a perfect shared expert is **2.0621 ms/step = 3.80%** of decode, so the
  realistic fused gate+up recovery remains well under 1%. DE-2 still ranks roughly 4x below
  DE-1's routed gate+up owner (14.0126 ms/step).

## Limits

- Profiled capture: per-kernel times are diagnostic, not a performance measurement. One case
  (`code-p4096`), one live count (4097), one model, one host, one execution profile.
- The split is by dispatch signature cross-checked against exact GGUF shapes, not a per-node
  role range; `f32_to_bf16` / `weighted_sum_out` share one signature and stay grouped.
- No speedup is claimed and no kernel was written. This artifact does not by itself justify
  promoting a fusion kernel.
