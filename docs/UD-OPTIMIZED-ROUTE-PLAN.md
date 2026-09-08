# Getting UD files onto the optimized kernels

_Status: in progress. Written 2026-09-08 from static analysis plus the
prefill route-gap measurement; updated as items land._

| Item | State | Result |
| --- | --- | --- |
| 0. Reconstruct the published protocol | **done** (`d8f5e951c`) | the published row uses `qwen35_readme_sweep.py --force-bulk-prefill --bulk-prefill-attention-mode bulk --use-wmma-prefill --use-gemv-decode --graph-replay-decode`; zbook floor for the plain file is 294.8 tok/s against the desktop-published 396.1 |
| 1. Per-tensor repack eligibility (UD-U3) | **done** (`d8f5e951c`) | optimized bytes 0.0% -> 44.2% (K_M); prefill 22.1 -> 29.9 tok/s, decode 6.21 -> 7.24, resident -0.92 GB |
| 2. Close the remaining non-IQ fallbacks | **partly done** | Q5/Q6/Q4 dense role coverage landed: optimized bytes 44.2% -> 61.3% (K_M), 34.4% -> 44.4% (K_S); prefill 30.4 -> 41.5 tok/s. Q3_K still has no T16 layout. |
| 3. IQ4_XS dense GEMM entry | **done, gfx1151** | dense = the degenerate single-expert case of the existing MMQ kernel; no repack. Prefill 41.5 -> **127.2 tok/s**; IQ4_XS kernel 9,280 -> 791 ms. gfx1100 is a follow-up. |
| 4. Extend the MMQ route to every expressible quant | **done, gfx1151** | IQ3_S and IQ4_NL added; prefill 127.2 -> **171.9 tok/s**. Q3_K/IQ2_S/IQ2_XS need a per-16-scale contract. |

**Where this leaves it.** On the published protocol, UD `Q4_K_M` prefill has gone **22.1 -> 171.9 tok/s (7.78x)** and the gap to the plain `Q4_K_S` file on the same host **13.3x -> 1.72x**. The residual is Q3_K, IQ2_S and IQ2_XS, which carry a scale per 16 elements and need a per-16-scale MMQ contract, plus Q3_K's missing T16 layout.

The kernel work is largely done. The published UD files do not reach it. This
document says exactly why, what it is worth, and in what order to fix it.

Companion to [`OPTIMIZE-KERNEL-IQ-DENSE.md`](OPTIMIZE-KERNEL-IQ-DENSE.md)
(kernel-level campaign) and
[`../benchmarks/results/2026-09-08-zbook-ud-prefill-route-gap.json`](../benchmarks/results/2026-09-08-zbook-ud-prefill-route-gap.json)
(the measurement that opened this).

## The number

Same model, same GPU, same harness, same 512-token prompt:

| File | Prefill | Prefill GPU time | Weight bytes in an optimized layout |
| --- | ---: | ---: | ---: |
| `Qwen3.8-27B-Q4_K_S` (plain) | 160.5 tok/s | 3,320 ms | 15.03 GB (**94.8%**) |
| `Qwen3.8-27B-UD-Q4_K_M` | 13.77 tok/s | 37,299 ms | 0.00 GB (**0.0%**) |
| `Qwen3.8-27B-UD-Q4_K_S` | — | — | 0.00 GB (**0.0%**) |

Not "mostly raw". **Zero.** Every one of the 554 rank-2 tensors in both UD files
plans to a GEMV-shaped layout, including the Q4_K, Q5_K and Q6_K tensors that
have qualified T16 kernels sitting right there.

## Root cause: one model-wide boolean

`plan_qwen35_gguf_materialization` (`qwen35_gguf_materialize.py`) computes

```python
ar_repack_veto  = gguf_ar_decode_repack_veto(ar_layer_types)   # ANY layer tensor
use_decode_repack = requested_decode_repack and not ar_repack_veto
```

and threads that **single boolean** into every `_spec_for_tensor` call.
`gguf_ar_decode_repack_veto` is
`any(type in {IQ2_XS, IQ3_XXS, IQ4_XS} for type in ...)`.

So the 117 IQ4_XS tensors in `UD-Q4_K_M` (172 IQ4_XS + 5 IQ3_XXS + 1 IQ2_XS in
`UD-Q4_K_S`) strip the T16/x8/planar layouts from **all** the others. With
repack off, rank-2 Q4_K falls to `q4_k_pack8` and Q5_K/Q6_K/Q3_K/IQ fall to
`raw_gguf`, and those layouts have only GEMV-shaped prefill kernels.

**The veto's stated rationale does not apply to these files.** Its comment reads
"Raw-IQ models' selected kernels consume compressed rank-3 GGUF layouts. Keep
one compatible resident plan…". Measured: both UD files contain **no rank-3
tensors at all** (554 rank-2, 312 rank-1), and every veto-triggering IQ tensor
is rank-2 dense. The veto is a MoE/selected-path guard firing on a dense model.

The code already anticipates this fix by name: `gguf_ar_decode_repack_veto`'s
docstring calls it "a future per-tensor repack-eligibility change (UD-U3 layout
selection)", and UD-U1 already split this knob from the model-wide F32 linear
contraction precisely so it could move independently.

## What lifting the veto is worth

Planned statically through the real planner with `repack_veto=False`, nothing
else changed:

| File | Optimized bytes, today | With per-tensor veto | Resident weight bytes |
| --- | ---: | ---: | ---: |
| `UD-Q4_K_M` | 0.00 GB (0.0%) | **7.11 GB (44.2%)** | 17.265 → **16.284 GB (−5.7%)** |
| `UD-Q4_K_S` | 0.00 GB (0.0%) | **5.15 GB (34.4%)** | 15.972 → **15.143 GB (−5.2%)** |

It is faster **and smaller**. There is no memory/speed trade to argue about:
`q4_k_pack8`, the fallback for rank-2 Q4_K, stores `qweight + scales + mins` at
`n*k*0.75` bytes (~6 bpw) where the source Q4_K is ~4.5 bpw and T16 is compact.
The veto is currently costing both throughput and VRAM.

## Work, in order

### 0. Reconstruct the published prefill protocol — do this first

The same harness measures the plain file at 160.5 tok/s where the scoreboard
publishes **396.1** for that model/quant/GPU, while decode agrees (12.56 vs
13.1). Until that is explained, no before/after on the items below can be
quoted honestly. Two concrete leads:

- `use_wmma_prefill` defaults **off**: `_env_wmma_prefill_enabled()` returns
  False when `HIPENGINE_GGUF_WMMA_PREFILL` is unset, and
  `Qwen35GGUFResidentSession.prefill()` exposes no per-call override, so it
  inherits the session value resolved at construction.
- `resolve_qwen35moe_fastpath_safety()` can force it off again, and its
  docstring is **stale**: it says "T16 has no WMMA prefill kernels yet, so
  prefill stays on the slow GEMV fallback", but
  `hipengine_gguf_q4_k_t16_wmma_prefill_shared_b_*` exist in
  `gguf_k_t16_selected_prefill.py` and the plain-file kernel trace shows
  `gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel` running with 212 calls.
  Determine which of those routes the published row used.

Cost: measurement only. Blocks honest reporting on everything else.

### 1. Per-tensor repack eligibility (UD-U3)

Replace the model-wide `use_decode_repack` boolean with a per-tensor predicate.
Narrowest version that preserves the stated intent: veto repack only for a
tensor whose **own** ggml type is in `_AR_RAW_IQ_GGML_TYPE_IDS`, or only for
rank-3 tensors, which is what the rationale actually describes.

- `qwen35_gguf_policy.py`: add the per-tensor form beside
  `gguf_ar_decode_repack_veto`; leave `gguf_ar_f32_linear_contraction`
  model-wide and untouched.
- `qwen35_gguf_materialize.py`: thread it through
  `plan_qwen35_gguf_materialization` → `_plan_layer` → `_spec_for_tensor`.
- Keep the model-wide form as the fallback behind an explicit switch so the
  change is bisectable.

Prize: the table above. This is the single biggest and cheapest item, and it
adds no kernel code.

**Gate.** This changes which kernels serve most of the model, so it is an
arithmetic change, not a scheduling one: it needs the full production-profile
numerics gate from `EXECUTION-PROFILES.md` (calibrated mean/tail/max KL, top-1
by category and shape, deterministic repeats, BF16-relative and task gates),
plus the admission/consumer-surface suites. Existing tests that pin the current
behaviour: `tests/test_loading_qwen35_gguf_policy.py`,
`tests/test_qwen35_gguf_materialize_helpers.py`,
`tests/test_gguf_ud_admission.py`, `tests/test_gguf_selected_call_intents.py`.

**Risk to check first.** `resolve_qwen35moe_fastpath_safety` records that the
raw-GGUF path failed the KL/top-1 contract (P9.E2) with WMMA prefill and GEMV
decode both on, and names "enable resident T16 decode repack" as the remedy.
So this change moves toward the configuration that docstring calls safe, not
away from it — but confirm that on the UD files rather than assuming it.

### 2. Close the remaining non-IQ fallbacks — partly done

After item 1, `UD-Q4_K_M` still left 8.99 GB on GEMV. The Q5/Q6/Q4 role gaps
are now closed; Q3_K remains.

**Landed.** Every one of these was a *measurement-scoped* allowlist, not a
kernel limit: each predicate reads "select only the measured …" and was written
against the plain `Q4_K_S` file, which carries no tensor at the missing roles.
Every remaining tensor was verified T16-alignable first, and every missing
(role, shape) pair was already admitted for Q4_K on the identical geometry.

- Q5_K: `_is_dense_h5120_q5_t16_tensor` grew from 3 roles to the full 9-role
  dense set, matching `_DENSE_Q4_T16_SIDECAR_POLICY`. 2.32 GB (K_M) / 1.19 GB
  (K_S) off the raw path.
- Q6_K: the wide predicate gained `ffn_up`, `attn_output`, `ssm_out` (all
  >= 5120 wide, so they keep the wide-path measurement); narrow `attn_k`
  follows `attn_v` at the identical (1024, 5120) geometry.
- Q4_K: `ssm_out` (5120, 6144) joined the sidecar policy — the same geometry as
  `attn_output`.

**Still open.** Q3_K (0.27 GB K_M / 1.01 GB K_S) has no T16 layout constant at
all, so it needs a new layout plus a kernel rather than an allowlist entry.

**Not a routing gap after all.** `GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES`
excludes `MOSTLY_Q4_K_M`, but K_M's 28 Q4_K `ffn_gate`/`ffn_up` tensors already
resolve to `gguf_q4_k_t16_v1`. Adding the stamp would swap one optimized
variant for another, so it is a tuning question with its own measurement, not
part of this item.

The original table, for reference:

| Bytes (K_M / K_S) | Group | Why it stays raw |
| ---: | --- | --- |
| 2.32 / 1.19 GB | Q5_K `raw_gguf` | roles `attn_gate, attn_k, attn_output, ffn_gate, ffn_up` are outside the role-scoped Q5 T16 policy (`dense_q5_t16_qkv`, `_h5120`, `_ssm_out`), while `attn_v, ffn_down, ssm_out` do get T16. Compounded by our own `ca4e01581` rank-2 Q5/Q6 raw flip. |
| 0.27 / 1.01 GB | Q3_K `raw_gguf` | no T16 layout exists for Q3_K at all |
| 0.34 / 0.12 GB | Q6_K `raw_gguf` | role gaps in the planar policy |
| — / 0.19 GB | Q4_K `q4_k_pack8` (`ssm_out`) | shape absent from `_DENSE_Q4_T16_SIDECAR_POLICY` |
| 0.72 / — GB | Q4_K `raw_gguf` (`token_embedding`) | correct: the embedding has its own path, not a target |

Also check `GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES = ("MOSTLY_Q4_K_S",)`,
which excludes `UD-Q4_K_M` (file type `MOSTLY_Q4_K_M`) from the qmicro gate/up
route that `UD-Q4_K_S` does get. Adding the K_M stamp is a one-line policy
change if the shapes qualify.

### 3. IQ4_XS: the largest single remaining block

4.76 GB (K_M) and 6.29 GB (K_S), all `raw_gguf`, with no optimized dense layout
in the tree. Three options, cheapest first:

- **A. Dense entry point over the existing IQ integer-MMQ prefill kernels.**
  `gguf_iq_source_mmq_prefill.hip` already implements IQ4_XS/IQ3_XXS on the
  I128×J128×K256 RDNA3 integer-WMMA dataflow and is correctness-tested — but
  only in selected/MoE shape. A dense wrapper is far less work than a new
  kernel. Approximate (Q8_1 activations), so production-profile gate.
- **B. An IQ4_XS T16-style repack plus a dense T16 kernel**, mirroring what
  Q4/Q5/Q6 already have. Most work, best long-term fit.
- **C. Prefill-only on-the-fly dequant** into a bounded BF16 workspace, then the
  existing dense BF16 WMMA/rocBLAS path. No residency cost, prefill only;
  useful as a floor if A's ABI mismatch is larger than it looks.

Recommend A, with C as the fallback.

## Queued GPU work

Nothing below has been run. In order:

1. Reconstruct the published 396.1 protocol; re-measure both UD files under it.
2. Prototype the per-tensor veto behind a switch; measure prefill, decode and
   resident bytes on both UD files; run the production-profile numerics gate.
3. Re-run the layout audit to confirm the planned 44.2% / 34.4% is what the
   loader actually materializes, and take a kernel census to confirm the T16
   WMMA prefill kernels are the ones running.
4. Only then re-tune the IQ kernel items (`B4`, `T`×`R`, the R=8-vs-R=4
   crossover) on the desktop gfx1151, since their share of prefill will have
   changed.

## Reproducing the audit

`scratchpad/layout_audit2.py` plans both files through the real planner and
reports optimized-versus-fallback bytes by (quant, layout, role); set
`AUDIT_NO_VETO=1` to plan with `repack_veto=False`. It is CPU-only — it reads
GGUF headers and runs the planner, and never touches the GPU.
