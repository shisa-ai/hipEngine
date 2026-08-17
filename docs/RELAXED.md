# Historical Relaxed-Mode Inventory

Status: **historical inventory; superseded as normative policy on 2026-08-16**

The canonical public contract is now
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md). The active implementation,
calibration, historical-recovery, and c1/c>N plan is
[`PRODUCTION-NUMERICS-CAMPAIGN.md`](PRODUCTION-NUMERICS-CAMPAIGN.md).

This path remains to preserve references to the first retained changed-
arithmetic kernel and to explain how the old strict-versus-relaxed proposal
maps to the approved three-profile architecture. The former six-profile
`relaxed_fast_math` / `relaxed_layout` / `relaxed_kv_int8` /
`relaxed_routing` / `relaxed_all` design and its old per-tier generated-ID
requirements are **not active promotion policy**. The complete pre-supersession
catalog remains available in Git at
`dc6b603a2:docs/RELAXED.md`.

## 0. Landed historical evidence

### 0.1 GDN chain dv-tiling — verifier path (2026-06-09)

The MTP/DFlash verify-path GDN chain recurrence
`qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop` was tiled so each block
owned four consecutive dv columns. Grid size fell from 4096 to 1024 blocks and
rocprof measured `72.0 -> 53.39 us/call` (`-25.8%`).

The operation is algebraically the same but compiler FMA contraction and
scheduling differ. Versus the NumPy FP32 delta-rule oracle, chain output error
was about `1e-6` and recurrent-state error about `6e-8`; the difference is
roughly one to two FP32 ULP at the measured magnitudes. Three fixed-seed
quicksort runs retained byte-identical acceptance and exact AR IDs. A degenerate
one-token near-flat prompt could flip one argmax and cascade.

The strict `VTILE=1` implementation remained available. End-to-end verifier
cycle cost was unchanged within noise (`4.81 +/- 0.14 -> 4.80 +/- 0.28` AR-token
cost), because the kernel saving was below the dispatch/host-bound cycle noise
floor. The route is evidence that bounded recurrence reassociation can be
semantically valid; it is not a standalone serving-speed claim.

Artifact:
`benchmarks/results/2026-06-09-hipengine-m16-gdn-dvtiling-economics-cb.json`.
Detailed contemporary context remains in [`MEGAKERNEL.md`](MEGAKERNEL.md).

Under the approved contract this is a **T2 arithmetic-source candidate**. It
must still pass the new whole-profile evaluator before entering a certified
`production` manifest; historical retention does not grandfather it.

## 1. Why the old policy was replaced

The original document made strict the universal public default and described a
ladder of opt-in relaxed profiles. Later repository evidence showed that:

- several backend defaults already use quality-gated reassociated GDN,
  attention, and WMMA arithmetic;
- exact generated-ID equality conflates model near-ties with request/state bugs;
- production serving, debugging/reference parity, and batch-composition
  reproducibility need different contracts;
- the broad KL `0.05` / top-1 `90%` kernel floor is too loose by itself for a
  default same-quant implementation-drift policy; and
- weight/KV representation and routing/sampling-policy changes should not share
  the same authorization as reduction or fusion reassociation.

The approved replacement therefore has exactly three profiles:

| Profile | Contract |
| --- | --- |
| `strict` | hipEngine reference/oracle arithmetic for the selected model, quant, KV policy, and backend |
| `production` | exact control and ownership with calibrated, tightly bounded same-quant implementation drift |
| `batch_invariant` | fixed-seed request result invariant across supported slots, neighbors, widths, admission order, and compaction |

## 2. Historical candidates retained for re-evaluation

The old catalog nominated several useful hypotheses. Their current audited
status is carried by the active campaign rather than by old tier labels.

| Candidate | Current disposition |
| --- | --- |
| D64 fast verifier with reassociated GDN/full-attention state | Re-gate at D64/D128 with strict-teacher logits, state/KV ownership, and complete MTP economics. |
| GDN recurrence/order variants | Re-gate only where current profiles show a first-order cost; known K2/wave32 quality failures remain closed without a new numerical mechanism. |
| Attention online/split-K merge order | Valid T2 family; use full category, long-context, page/ring, and transition gates. |
| Packed projection/WMMA accumulation changes | Valid T1/T2 family when model/quant bytes are unchanged; require complete model and task gates. |
| Compound fusion and launch reduction | Valid T1/T2 family with a strict unfused fallback and profile-visible variant manifest. |
| INT8/FP8 KV, weight-format changes | T3 representation work; explicit product configuration, outside the initial production-numerics campaign. |
| Approximate routing, speculative probability-ratio acceptance, near-tie/top-k acceptance | T3 decision-policy work; explicit research/product campaign, never silently enabled by `production`. |
| Approximate sampler/tie semantics | Separate sampling contract; not admitted by implementation-drift evidence. |

The audited candidate order and artifacts are in
[`PRODUCTION-NUMERICS-CAMPAIGN.md`](PRODUCTION-NUMERICS-CAMPAIGN.md) section 4.

## 3. Rules that remain valid

Several original cautions remain binding:

- NaN/Inf, nondeterminism under an identical schedule, illegal memory access,
  output-buffer aliasing, stale graph pointers, wrong RoPE positions, lost KV,
  and cross-request state contamination are bugs, not relaxation.
- Stateful attention/GDN/Conv changes require long-context and transition
  coverage because drift and ownership failures accumulate.
- Every changed-arithmetic route retains a registered strict fallback.
- Profile and selected/fallback variants appear in benchmark artifacts and
  logs.
- A stacked profile is judged as a whole; per-kernel small errors are not summed
  as independent permissions.
- Rejected code does not remain as an undocumented fallback chain.
- Kernel development and micro-tuning happen in this repository. External
  repositories are read-only lineage and evidence references.

## 4. Promotion checklist

Use the canonical checklists in
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md), [`TESTING.md`](TESTING.md),
[`BENCHMARK.md`](BENCHMARK.md), and [`KERNELS.md`](KERNELS.md). At minimum:

- exact control/ownership and lifecycle pass;
- same-schedule repeat determinism pass;
- strict-teacher mean/tail/max KL and top-1 pass by category/shape/transition;
- BF16-relative non-inferiority where available;
- task-quality pass;
- strict fallback and variant-manifest provenance recorded;
- complete performance A/B on the declared workload; and
- worklog, compact artifact, benchmark rollup, and changelog updated for a
  retained performance claim.
