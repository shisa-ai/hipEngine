# BridgeSpec applicability audit

- Status: **complete; sliced head rejected, graph deprioritized, accepted-tail K/V-only transition retained**
- Reviewed upstream: [`kdheeraj-p/bridgespec@2b846f2`](https://github.com/kdheeraj-p/bridgespec/tree/2b846f2ff1eb95ac84e4b0488882b7e4066bff14) (MIT)
- Source review: 2026-08-27; RX 7900 XTX feasibility: 2026-09-02
- Evidence: [`RX 7900 XTX feasibility artifact`](../benchmarks/results/2026-09-02-rx7900xtx-bridgespec-feasibility.json); [`W7900 accepted-tail retention`](../benchmarks/results/2026-09-02-w7900-q4km-k3-c5c8-nextn-accepted-tail-kv-only-retained.json)

## Scope and evidence caveat

BridgeSpec is a useful gfx1100 research reference, not a comparable benchmark
baseline. Its reported rows use an RX 7900 XTX on Windows, ROCm 7.2, Qwen3.8-27B
Q4_0, F16/Q4_0 KV, one slot, and decode-only timing. hipEngine's retained W7900
rows use Linux, Q4_K_M, BF16 KV, Generation-2 complete-request timing, and a
binding category/heldout/profile/serving packet. Do not report BridgeSpec's
absolute rates as old/new hipEngine results.

The public repository explicitly omits raw request logs and a public benchmark
harness. Its strongest evidence is a controlled local A/B; the 146 tok/s
DFlash row is correctly labeled a fixed predictable-edit high-water, with a
56.8 tok/s prose counterexample and lower cache-correct release-candidate rows.

## Mechanism review

| Mechanism | Source | hipEngine disposition |
| --- | --- | --- |
| RDNA3 width-2–8 MMVQ: two waves, type-specific 2/4 output rows per block | `integrations/llama.cpp/patches/0002-gfx1100-wide-mmvq-tuning.patch` | **Profiler-driven screen only.** hipEngine already has shared-weight Q4/Q5/Q6 rowtile owners for rows 2–8 and a rows-2–8 Q6 head owner (`docs/KERNELS.md`). First prove an actual production projection misses those owners or is underoccupied; do not transplant ggml launch constants across Q4_0→Q4_K_M layouts. |
| Fold broadcast-weight channel batches into verifier columns | patch `0002`, `ggml_cuda_mul_mat_vec_q` | **Concept already owned.** hipEngine's row-shaped `launch_gguf_linear(..., rows=N)` and rowtile kernels traverse weights across verifier rows. Audit rocprof dispatch names/counts; retain only if a role still walks one weight per channel. |
| Disable fused wide gate/matvec when doubled arithmetic outweighs dispatch savings | patch `0002`; `docs/negative-results.md` | **Low-priority A/B.** BridgeSpec found both wide fusion and forcing K-quants to GEMM negative. hipEngine's fused pairs have independent exact evidence; test separate primitives only when the production kernel breakdown identifies that family as the cycle wall. |
| 40,960-row sliced draft head plus deterministic full-vocabulary remap | `tools/prepare_assets.py`, `src/mtp/*`, `src/dflash/*` | **Rejected selection; mechanism remains model-artifact research.** On RX 7900 XTX the head leaf improved 2.676→1.658 ms/call and four-prompt throughput 1.108x, but recall@16 fell 0.883→0.720 and Japanese mean acceptance 1.846→1.091. This already fails the four-category feasibility floor, so the final ten-prompt+heldout promotion gate was not spent. A materially different model-bound selection may reopen; do not reuse or tune this fixed list to benchmark prompts. |
| Graph-captured five-layer DFlash block drafter | `src/dflash/dflash_sidecar.hip`, `dflash_kernels.hip` | **Deprioritized after profiling.** Real B8 forward+top16+selector measured 17.225 ms/cycle unprofiled and 16.452 ms/cycle of marker-scoped kernels under rocprof (~93% kernel work), bounding ideal launch-only upside near 4.5% of unprofiled wall and ruling out the claimed 5.7x compression. Reopen only if DFlash becomes competitive or kernel reductions expose a larger gap; lifecycle requirements still apply. |
| External drafter KV and host-mediated target/drafter boundary | `docs/architecture.md`, `docs/correctness.md` | **No port.** hipEngine already has device-owned provider KV, stable graph pointers, request reset/reclaim, fault recovery, overload, and soak. BridgeSpec explicitly lacks reset/free/save/restore/fork/shift and is single-slot/non-thread-safe. |
| KV-only catch-up deletes dead Q/attention/FFN work | `docs/findings.md` §5 | **Retained.** RX evidence first proved the existing `kv_write_only` branch byte-exact for consumed cache, follow-on full logits/top16/token, and final state at 1.261→0.558 ms/token (2.261x). The in-tree W7900 follow-up passes the same future-state contract at 0.910→0.329 ms/token (2.763x), routes singleton/device and physical-batch accepted tails through that owner, and improves complete C5/C6/C7/C8 by +1.18%/+1.30%/+0.65%/+1.23% with all category/heldout slices positive and 520/520 IDs plus acceptance rows exact. Explicit zero keeps the complete block as rollback. |
| Sliced-head GPU top-k and ID remap guards | `src/mtp/kernels.hip`, `src/dflash/dflash_kernels.hip`, `argmax_guard_test.hip` | **Reuse the invariants, not hard-coded kernels.** Require finite logits, valid mapped IDs, deterministic tie behavior, manifest dimensions, and target-vocabulary bounds in hipEngine's four-axis registry path. |
| GQA stage all six query heads per KV head | `src/mtp/kernels_b.hip`; negative-results | **Reject by default.** BridgeSpec reports LDS occupancy regression. Reopen only for deep-context profiler evidence on the exact hipEngine attention owner. |
| Workload/category routing to wide DFlash | headline/findings §8 | **No prompt-conditioned routing.** This violates hipEngine anti-gaming rules if selected by prompt/category/token identity. Existing generic acceptance/circuit-breaker policy is the admissible mechanism; qualify it on every category and heldout. |
| Confidence threshold (`p_min`) removal | findings §2 | **Already reflected in policy discipline.** Optimize accepted progress per complete cycle. Any threshold change must be fixed before the full suite and cannot be tuned to a prompt subset. |

## Concrete test queue

### B1 — Verify current width-2–8 ownership before tuning

1. Run the in-tree cached `rocprofv3` verifier harness on the exact 27B C1/K3
   and 35B C1/K2 production manifests.
2. Classify every Q4/Q5/Q6 projection and lm-head launch by rows, duration,
   launch count, VGPR/LDS where available, and registry variant.
3. Compare current rowtile versus direct/fused fallbacks on the **actual operation
   shapes**. Candidate axes inspired by BridgeSpec are waves `{1,2}` and output
   rows/block `{2,4}`, but only inside the existing in-tree kernel family.
4. Stop if rowtile already owns the family or complete target wall does not
   improve. Run strict/profile quality and complete-request economics for any
   retained change.

### B2 — DFlash sliced-head artifact (closed/rejected)

The tested 40,960-row list fails the four-category feasibility quality floor;
do not spend or report it as a full ten-prompt+heldout promotion gate. Reopen
only for a materially different, model-bound selection with immutable manifest,
full-head fallback, CPU/GPU remap guards, and the complete required suite.

### B3 — Complete DFlash drafter graph (closed/deprioritized)

The measured route is kernel-bound. Do not open graph lifecycle complexity for
the current DFlash product route. Reopen only if kernel/top16/selector work first
falls enough to make the residual launch gap material.

### B4 — Provider catch-up dead-work proof (closed/retained)

RED route contracts now cover host, device-token, and physical-batch accepted-tail
propagation. The independent RX proof and in-tree W7900 future-state probe both
pass consumed-cache/follow-on-output/final-state exactness. The tracked-clean,
counterbalanced C5-C8 complete-request gate is positive at every width and every
category/heldout slice with 520/520 generated-ID and acceptance rows exact, so
K/V-only repair is the default. `HIPENGINE_GGUF_NEXTN_ACCEPT_KV_WRITE_ONLY=0`
keeps the complete-block correctness parent for rollback and diagnostics.

## Rejected shortcuts

- No cross-host or cross-protocol use of BridgeSpec's tok/s numbers.
- No hard-coded Qwen3.8 dimensions in generic dispatch or model code.
- No prompt, category, token-ID, or candidate-ID policy branches.
- No raw source copy before `docs/KERNELS.md`, lineage, license, and in-tree RED
  review. Any later port cites BridgeSpec commit and exact source path.
- No assumption that target verification makes state/lifecycle bugs harmless.
