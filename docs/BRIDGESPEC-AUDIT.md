# BridgeSpec applicability audit

- Status: **source review complete; candidates queued, no source ported**
- Reviewed upstream: [`kdheeraj-p/bridgespec@2b846f2`](https://github.com/kdheeraj-p/bridgespec/tree/2b846f2ff1eb95ac84e4b0488882b7e4066bff14) (MIT)
- Reviewed: 2026-08-27

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
| 40,960-row sliced draft head plus deterministic full-vocabulary remap | `tools/prepare_assets.py`, `src/mtp/*`, `src/dflash/*` | **High-value DFlash candidate.** hipEngine DFlash2 currently allocates/scores full vocabulary rows. Implement as a model artifact/plugin with exact manifest+ID map, full-vocab target authority, registered full-head fallback, and full multi-category acceptance/economics gate. It is a draft-policy/artifact change, not a free kernel optimization. |
| Graph-captured five-layer DFlash block drafter | `src/dflash/dflash_sidecar.hip`, `dflash_kernels.hip` | **High-value after profiling.** hipEngine has native DFlash2 kernels and explicit cache specs but no equivalent complete drafter graph. Measure wall/GPU/launch gap first; capture under per-request lifecycle/transaction identities, never BridgeSpec's process singleton assumptions. |
| External drafter KV and host-mediated target/drafter boundary | `docs/architecture.md`, `docs/correctness.md` | **No port.** hipEngine already has device-owned provider KV, stable graph pointers, request reset/reclaim, fault recovery, overload, and soak. BridgeSpec explicitly lacks reset/free/save/restore/fork/shift and is single-slot/non-thread-safe. |
| KV-only catch-up deletes dead Q/attention/FFN work | `docs/findings.md` §5 | **Medium-priority state audit.** hipEngine `advance_state_only` already skips logits/head but executes the provider block. Before removing more, add a RED request-state fingerprint proving future proposals need only the retained K/V/state side effects. Benchmark complete cycle and following-token state, not the leaf alone. |
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

### B2 — DFlash sliced-head artifact

1. Add an immutable draft-vocabulary manifest containing target/drafter/model
   hashes, ordered target token IDs, sliced rows, and deterministic remap hash.
2. Generate the sliced head from user-supplied weights; do not commit weights or
   external ID lists.
3. Add CPU/GPU remap and invalid/NaN/tie RED fixtures plus full-head fallback.
4. Measure VRAM, head wall, draft acceptance, accepted tokens/output, and true-AR
   complete-wall speed on all `code`, `general_en`, `general_ja`, `mixed_ja_en`
   prompts and category heldouts. A prose/category floor is binding.

### B3 — Complete DFlash drafter graph

1. Profile native DFlash proposal launch/H2D/D2H and wall-GPU gap before capture.
2. Capture only stable per-request pointers and dynamic metadata; include
   request/slot/profile/manifest/cache generations in compatibility identity.
3. Prove reset, cancellation, fault rollback, pressure, and alternating prompt
   isolation. BridgeSpec's singleton lifecycle is not acceptable here.
4. Retain only when graph/eager outputs and state gates pass and complete cycle
   wall improves.

### B4 — Provider catch-up dead-work proof

1. Fingerprint provider Conv/recurrent/KV/pending hidden before and after current
   `advance_state_only` and the proposed reduced path.
2. Compare the next two proposal cycles, not only the catch-up output.
3. If exact, profile the current block by Q/attention/FFN/KV/head and remove only
   components proven dead. Keep the full-block registered fallback.

## Rejected shortcuts

- No cross-host or cross-protocol use of BridgeSpec's tok/s numbers.
- No hard-coded Qwen3.8 dimensions in generic dispatch or model code.
- No prompt, category, token-ID, or candidate-ID policy branches.
- No raw source copy before `docs/KERNELS.md`, lineage, license, and in-tree RED
  review. Any later port cites BridgeSpec commit and exact source path.
- No assumption that target verification makes state/lifecycle bugs harmless.
