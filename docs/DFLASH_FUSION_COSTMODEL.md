# DFlash R3.2 fusion L1 cost model

Date: 2026-05-23
Scope: paper-only cost model for DFlash Round-3 / R3.6. No GPU work was run for this document.

## Goal

Before any new `.hip` fusion work, classify each plausible verifier/drafter fuse as **PASS**, **FAIL**, or **INCONCLUSIVE** using the Round-2 lessons:

- L1: a fuse that saves N launches but multiplies per-block work by M loses when `M × block_count > N × launch_overhead`.
- L2: staged fuses that require host-side scratch/barrier reset can consume the entire dispatch saving.
- L9/L13: host-launch overhead is real but small for current DFlash (`~2%` of cycle); only fuses that also reduce kernel work are promotable as major levers.

Output of this doc: the short list for R3.6, plus a do-not-implement list so future agents do not repeat M13.B.1/B.2 mistakes.

## Constants and source measurements

| Quantity | Value | Source / note |
| --- | ---: | --- |
| W7900 memory bandwidth | `864 GB/s` | `docs/ROOFLINE.md` hardware model |
| W7900 BF16 WMMA peak | `~120 TFLOP/s` | `docs/ROOFLINE.md` / R3.3 planning constant |
| Direct launch overhead | `~3.6 µs/launch` | R2.3 replay math: `~0.45 ms / ~124 drafter launches` |
| DFlash drafter launches/cycle | `~124` | R2.3 graph replay analysis (`8 layers × ~15` + final stages) |
| Verifier launches/pass | `~1011` | M13.B.0/M13.D W7900 rocprof after write-through (`1052 → 1011.55`) |
| DFlash B=4 verifier rows | `B+1 = 5` | R2.2/R2.3 DFlash benchmark shape |
| DFlash drafter rows | `block_size = 16`; lm-head rows `B=4` | `NativeDFlashChainDrafter`, z-lab drafter |
| DFlash drafter hidden | `2048` | z-lab drafter config / R3.3 |
| DFlash drafter intermediate | `~8192` | Qwen-style MLP, exact value should be read from artifact before kernel work |
| DFlash vocab | `~262k` | target tokenizer/head shape; exact value in model config |

Useful conversion:

```text
saved_ms_from_launches = saved_launches * 0.0036 ms
```

A launch-only fuse must save roughly `556` launches/cycle to move a 62 ms DFlash cycle by 2 ms. No single local fuse in this document is in that class. Therefore R3.6 is a **polish / enabling track**, not the main wall lever; R3.4 drafter dense WMMA remains the primary wall-reduction track.

## Verdict summary

| # | Candidate | Phase | L1 verdict | Est. launches saved | Est. saved wall | R3.6 action |
| --- | --- | --- | --- | ---: | ---: | --- |
| C1 | DFlash post-attention `add + RMSNorm` | drafter | **PASS-small** | `8/cycle` | `~0.029 ms/cycle` + one hidden reread/layer | Implement only if touching drafter kernels anyway; safe first fuse. |
| C2 | DFlash final-MLP residual `add + next-layer RMSNorm` | drafter | **PASS-small / API-risk** | `7/cycle` internal layer boundaries | `~0.025 ms/cycle` + one hidden reread/boundary | Defer unless R3.4 refactors layer pipeline. |
| C3 | Verifier post-attention `add + RMSNorm` | verifier | **Already fused** | `0` | `0` | Do not duplicate; current `post_attention_add_rmsnorm_fp16` is one launch. |
| C4 | Verifier MoE-combine residual + next-layer input RMSNorm | verifier | **PASS-small / API-risk** | `~39/pass` | `~0.14 ms/pass` | Shortlist only if API can carry `next_norm` without disturbing exact AR. |
| C5 | Full-attn verifier QKV prepare fusion (`split + cast + q/k RMSNorm+RoPE`) | verifier | **PASS-small** | `20/pass` | `~0.072 ms/pass` | Good R3.6 verifier starter; avoids GEMV fusion trap. |
| C6 | Q/K projection GEMV + RoPE/RMSNorm (direct) | verifier/drafter | **FAIL** | superficially `8–40` | negative | Do not implement direct per-output GEMV fusion; it cannot reduce across head_dim without duplicating work or adding a barrier. |
| C7 | DFlash query-side `RoPE + QKV-noise` direct fuse | drafter | **FAIL** | `8/cycle` | negative | Same head-dim reduction issue as C6; revisit only inside a staged WMMA dense rewrite. |
| C8 | DFlash `SiLU + mul + down-proj` direct fuse | drafter | **FAIL** | `8/cycle` | negative by orders of magnitude | Do not implement; duplicates activation by `out_features` (`~2048×`). |
| C9 | DFlash `final_norm + lm_head` direct fuse | drafter | **FAIL** | `1/cycle` | negative by orders of magnitude | Do not implement; duplicates norm by vocab (`~262k×`) unless redesigned as separate norm. |
| C10 | DFlash final-norm row trimming (`16 rows → B rows`) | drafter | **PASS-tiny** | `0` | `≤0.03 ms/cycle` | Fold into R3.4 cleanup; not a standalone task. |
| C11 | DFlash K rotate + KV write | drafter | **Mostly already done / no-op** | `0–8/cycle` | tiny | Context K rotate already writes cache; query concat remains an attention-layout issue. |
| C12 | Verifier write-through extensions | verifier | **Already done for hot path** | `0 new` | `0 new` | M13.B.0 is wired in chain/tree hot paths; only prefill/BF16 consistency remains, not R3.6. |
| C13 | Staged rotate+GEMV with keyed persistent barrier | verifier | **INCONCLUSIVE** | `40–190/pass` | `0.14–0.68 ms/pass` before kernel cost | Revisit only with keyed barrier design; M13.B.1/B.2 default-off attempts failed. |

Shortlist for R3.6:

1. **Verifier starter:** C5 (`split + cast + q/k RMSNorm+RoPE` prepare fusion) because it avoids the GEMV/reduction trap and has low numerical risk.
2. **Drafter starter:** C1 (`dflash_add_rmsnorm_bf16`) because it is structurally safe, has an obvious CPU-reference oracle, and exercises the fused-kernel registration/fallback path before harder R3.4 work.
3. **Optional verifier API fuse:** C4 (MoE-combine residual + next input RMSNorm) only if the Python/runtime API can carry both raw `next_hidden` and normalized `next_norm` cleanly.

Do **not** expect these to make DFlash profitable alone. Combined C1+C5+C4 launch savings are only `~0.24 ms` against a `~62 ms` cycle. R3.6's value is to remove obvious fragmentation and prepare fallback/registration patterns; R3.4 dense WMMA and R3.1 routing remain the real Round-3 wall/guard levers.

## Candidate details

### C1 — DFlash post-attention `add + RMSNorm`

Current drafter layer path (`scripts/dflash_chain_e2e_bench.py:_run_layer`):

```text
O dense -> dflash_add_bf16(query_in, attn_proj -> hidden_attn)
        -> dflash_rmsnorm_bf16(hidden_attn -> post)
        -> gate/up dense
```

L1 math:

| Term | Estimate |
| --- | ---: |
| Saved launches | `1/layer × 8 layers = 8/cycle` |
| Saved dispatch | `8 × 3.6 µs = 28.8 µs/cycle` |
| Added redundant compute | none; RMSNorm already does the hidden reduction |
| HBM effect | avoids rereading `hidden_attn` for RMSNorm (`16 × 2048 × 2 B = 64 KiB/layer`) but must still write `hidden_attn` because MLP residual needs it |
| Added kernel complexity | low; one block/row RMSNorm with add folded into the load path |

Verdict: **PASS-small**. Implement as `dflash_add_rmsnorm_bf16(input, residual, weight, hidden_out, norm_out, rows, hidden, ...)`. Keep existing `dflash_add_bf16 + dflash_rmsnorm_bf16` fallback registered.

### C2 — DFlash final-MLP residual `add + next-layer RMSNorm`

Current path ends each layer with:

```text
down dense -> dflash_add_bf16(hidden_attn, mlp -> query_out)
next layer -> dflash_rmsnorm_bf16(query_out -> norm)
```

L1 math:

| Term | Estimate |
| --- | ---: |
| Saved launches | `1` per internal layer boundary = `7/cycle` (last layer feeds final norm, not next input norm) |
| Saved dispatch | `25.2 µs/cycle` |
| Added redundant compute | none if fused kernel writes both `query_out` and `next_norm` |
| API cost | layer loop must carry an additional normalized buffer or alternate input contract |

Verdict: **PASS-small / API-risk**. Do not start here unless R3.4 already refactors the drafter layer loop.

### C3 — Verifier post-attention `add + RMSNorm`

The target verifier already uses a fused post-attention add+rmsnorm primitive:

- linear-attn audit row: `post_attention_add_rmsnorm_fp16` → `paro_add_rmsnorm_out_fp16` (1 launch)
- full-attn audit row: `post_attention_add_rmsnorm_fp16` → `paro_add_rmsnorm_out_fp16` (1 launch)

Verdict: **Already fused**. No R3.6 work.

### C4 — Verifier MoE-combine residual + next-layer input RMSNorm

Current verifier state after M13.B.0:

- `run_moe_c1_fp16(..., out=next_hidden)` writes raw next hidden directly into the trunk buffer.
- The next layer still starts with `input_rmsnorm_fp16` as a separate launch.

A fused combine+nextnorm kernel would write both:

```text
next_hidden[row, hidden] = residual + selected_moe + shared_moe
next_norm[row, hidden]   = RMSNorm(next_hidden[row])
```

L1 math:

| Term | Estimate |
| --- | ---: |
| Saved launches | one next-layer input RMSNorm for layers 1..39 = `~39/pass` |
| Saved dispatch | `39 × 3.6 µs = 140 µs/pass` |
| Added redundant compute | none if combine kernel already has row-wise hidden output and adds an RMS reduction before store |
| HBM effect | avoids one read of `next_hidden` per layer (`rows × hidden × 2 B`; small but positive) |
| API cost | high: next layer must consume `next_norm` for projections while raw `next_hidden` remains available for residual/capture |

Verdict: **PASS-small / API-risk**. This is the most plausible verifier-side add+rmsnorm fuse, but the API work may exceed the value. Shortlist only after C5 if a clean `normalized_hidden` side-buffer already exists.

### C5 — Full-attention verifier QKV prepare fusion

Current full-attn verifier audit row:

```text
prepare_full_attention_qkv_fp16:
  qwen35_split_qgate_fp16
  fp16_to_f32
  qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16
```

This is not a GEMV fuse. It is a prepare-kernel fuse after Q/K/V projection has materialized. A single kernel can split Q/K/V, cast Q/K to FP32, perform per-head RMSNorm+RoPE, and write the same prepared buffers.

L1 math:

| Term | Estimate |
| --- | ---: |
| Saved launches | `2/full-attn layer × 10 layers = 20/pass` (3 prepare launches → 1) |
| Saved dispatch | `20 × 3.6 µs = 72 µs/pass` |
| Added redundant compute | none; same per-head reductions as existing `qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16` |
| HBM effect | avoids intermediate qgate/split and fp16→f32 temporary traffic |
| Numerical risk | medium: must match current BF16/FP32 cast order and RoPE exactly |

Verdict: **PASS-small**. Best verifier starter for R3.6 because it avoids the M13.B.1 GEMV-fusion failure mode and has bounded per-head work.

### C6/C7 — Q/K projection GEMV + RoPE/RMSNorm direct fuse

Tempting but wrong design:

```text
Q/K GEMV output element block also computes head RMSNorm + RoPE
```

Why it fails:

- The GEMV block owns one output element or output pack; head RMSNorm needs the full `head_dim` vector.
- Without a cross-block barrier, each output block cannot know the head RMS.
- Recomputing the head reduction inside every output block multiplies the reduction work by `head_dim` / block ownership. This is the same structural bug as M13.B.1, where in-LDS rotation was re-done by every `(out_pack, row)` block and turned `-40 launches` into `+12.4 ms/pass`.

Verdict: **FAIL** for direct fusion. Revisit only inside R3.4's staged WMMA dense design, where a tile/block can own a full vector tile and write a prepared side buffer without duplicating reductions.

### C8 — DFlash `SiLU + mul + down-proj` direct fuse

Current path:

```text
gate dense -> up dense -> dflash_silu_mul_bf16(gate, up -> act)
                        -> dflash_dense_bf16_to_bf16(act, down_weight -> mlp)
```

Naive direct fuse computes `silu(gate[j]) * up[j]` inside every down-proj output block.

L1 math for one drafter layer, approximate:

| Term | Estimate |
| --- | ---: |
| Saved launches | `1/layer`, `8/cycle` → `28.8 µs/cycle` |
| Current activation work | `rows × intermediate ≈ 16 × 8192 = 131k` SiLU/mul ops/layer |
| Naive fused activation work | `rows × out_features × intermediate ≈ 16 × 2048 × 8192 = 268M` SiLU/mul ops/layer |
| Work multiplier | `out_features ≈ 2048×` |

Verdict: **FAIL**. The activation must be materialized once, then down-projected. If R3.4 rewrites dense WMMA, keep `silu_mul` as a separate cheap kernel or use a two-stage epilogue that materializes `act` once.

### C9 — DFlash `final_norm + lm_head` direct fuse

Current path:

```text
dflash_rmsnorm_bf16(query_in[16 rows] -> final_norm)
w8a16_linear_bf16_f32_out(final_norm[row 1..B] -> logits[B, vocab])
topk_f32_rows_i32(logits -> ids/values)
```

Directly fusing norm into the lm-head GEMV would require each vocab-output block to compute or reload the RMS for its row.

L1 math:

| Term | Estimate |
| --- | ---: |
| Saved launches | `1/cycle` → `3.6 µs/cycle` |
| Naive work multiplier | `vocab_size ≈ 262k×` if RMS is recomputed per vocab output |
| Hidden traffic avoided | at most `B × hidden × 2 B` (`~16 KiB` at B=4, hidden=2048) |

Verdict: **FAIL**. Do not fuse final norm into the vocab projection. If lm-head becomes a bottleneck, the correct direction is reduced logits / token-subset scoring, not norm fusion.

### C10 — DFlash final-norm row trimming

Observation: current `dflash_rmsnorm_bf16` final norm is launched with `rows=block_size=16`, while the lm-head consumes `candidate_budget` rows starting at row 1. At B=4, rows 5..15 are not used by the lm-head/top-k.

L1 math:

| Term | Estimate |
| --- | ---: |
| Saved launches | `0` |
| Saved kernel work | up to `12/16 = 75%` of final_norm work |
| Absolute phase time | R2.2 sync split: final_norm is included in `drafter other`; measured around `~0.04 ms/cycle` |

Verdict: **PASS-tiny**. Fold into R3.4 cleanup if easy (e.g., normalize only rows `[1, B]` into a compact lm-head input). Not a standalone R3.6 item.

### C11 — DFlash K rotate + KV write

There are two similar-looking sites:

1. **Committed context cache path** (`commit_context_rows`): K projection -> `dflash_key_rmsnorm_rotary_f32` writes directly into `kv_cache_keys`; V projection writes directly into `kv_cache_values`. There is no separate K-rotate + KV-write pair left to fuse.
2. **Per-cycle query path** (`_run_layer`): Q/K/V query rows are projected, Q/K query rows are RMSNorm+RoPE'd, then context K/V and query K/V are concatenated for attention. This is not a KV-write problem; it is an attention input-layout problem.

Verdict: **Mostly already done / no-op**. The remaining query concat launches are better addressed by changing attention to consume `(context_cache, query_k, query_v)` as two spans rather than by a rotate+write fuse.

### C12 — Verifier write-through extensions

M13.B.0 is already wired in the hot verifier paths:

- `_iterate_verify_chain_layers`: linear-attention `run_*(..., out=next_hidden)`.
- `_run_full_attention_chain_c1_loop`: `out=row_out`.
- `_run_full_attention_chain_batched`: `out=next_hidden`.
- `_run_full_attention_tree_batched`: `out=next_hidden`.

The remaining write-through notes in `docs/MTP.md` are prefill/BF16 consistency items, not DFlash R3.6 hot-path work.

Verdict: **Already done for hot path**. No R3.6 action unless a future profile shows a new D2D copy family inside the DFlash verifier window.

### C13 — Staged rotate+GEMV with keyed persistent barrier

This is the corrected version of the M13.B.1/B.2 attempts:

- Rotate once per source row into HBM scratch.
- GEMV blocks read the staged rotated vector.
- Use a keyed/persistent barrier so the launcher does **not** need a `hipMemsetAsync(barrier, 0, 8)` per call.

Prior evidence:

- M13.B.1 selected-MoE rotate+GEMV saved `40` rotate launches/pass but was rejected: kernel time exploded because rotation was re-done in every `(out_pack,row)` block (`+12.4 ms/pass`).
- M13.B.2 shared-expert staged rotate saved `10` rotate launches/pass but added `10` barrier memset launches/pass, net launch delta `0`, kernel time `+0.5%`.

L1 math if keyed barrier works:

| Scope | Launches saved | Dispatch saved |
| --- | ---: | ---: |
| selected gate/up rotate-out | `40/pass` | `0.144 ms/pass` |
| shared-expert rotate-in/out subset | `10–30/pass` | `0.036–0.108 ms/pass` |
| all rotate family upper bound | `190/pass` | `0.684 ms/pass` |

Verdict: **INCONCLUSIVE**. The paper case passes only if the keyed barrier removes the hidden host memset and the staged kernel time stays ≤ unfused. Do not schedule as R3.6 until the keyed-barrier design is written; keep it as an M14.fuse.barrier follow-up.

## R3.6 implementation checklist

For every R3.6 fuse that graduates from this document:

1. Register via `KernelKey(...)`; do not add backend/quant conditionals in dispatch/model code.
2. Keep the unfused fallback registered and env-gate the fuse during validation.
3. Add CPU-reference or unfused-chain parity tests before perf measurement.
4. Validate exact greedy AR equality on the DFlash same-session suite.
5. Run `rocprofv3 --kernel-trace` only after the GPU is free; expected evidence is:
   - the old kernel sequence disappears,
   - the new kernel name appears,
   - launch count drops by the predicted amount,
   - kernel duration does not grow enough to eat the dispatch saving.

## Final R3.2 decision

R3.2 finds **no local fusion that can materially close the 62 ms DFlash cycle by itself**. The viable fuses are sub-millisecond polish:

- C1 (drafter add+rmsnorm): safe and useful as a first fused-kernel exercise.
- C5 (verifier QKV prepare): safest verifier-side launch reduction.
- C4 (combine+nextnorm): plausible but API-heavy; only do it if the runtime already exposes the right buffers.

Therefore R3.6 should be kept narrow (1 drafter fuse + 1 verifier fuse) and should not distract from R3.4 dense WMMA. If R3.6 fails to show at least a measured `≥3%` cycle-cost reduction, record that as a negative result and move back to R3.4/R3.5.
