# Getting UD files onto the optimized kernels

_Status: in progress. Written 2026-09-08 from static analysis plus the
prefill route-gap measurement; updated 2026-09-09 with the W4A16 route, the
production-referenced accuracy ruling, and the post-MMQ next steps._

**Accuracy basis (human-lead ruling, 2026-09-09).** This campaign scores
accuracy in the **production profile**: candidate production against the
incumbent production default on the same model artifact, config, prompts and
forced positions, under the calibrated envelope of
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) §6.1 (mean/p95/p99/max KL ≤
`1e-3`/`5e-3`/`2e-2`/`5e-2`, top-1 ≥ 99% overall / 97% per scope). The
reference is the previous production default — **not** a full strict profile,
and **not** an external engine. `EXECUTION-PROFILES.md` §6 currently names
strict as the initial-gate reference; reconciling that wording with this
ruling is a follow-up owed to that document (item 15), not something this
campaign decides.

Companion to [`OPTIMIZE-KERNEL-IQ-DENSE.md`](OPTIMIZE-KERNEL-IQ-DENSE.md)
(kernel-level campaign) and
[`../benchmarks/results/2026-09-08-zbook-ud-prefill-route-gap.json`](../benchmarks/results/2026-09-08-zbook-ud-prefill-route-gap.json)
(the measurement that opened this).

## Status

| # | Item | State | Result |
| --- | --- | --- | --- |
| 0 | Reconstruct the published protocol | **done** (`607b10a3e`) | the published row uses `qwen35_readme_sweep.py --force-bulk-prefill --bulk-prefill-attention-mode bulk --use-wmma-prefill --use-gemv-decode --graph-replay-decode`; zbook floor for the plain file is 294.8 tok/s against the desktop-published 396.1 |
| 1 | Per-tensor repack eligibility (UD-U3) | **done** (`607b10a3e`) | optimized bytes 0.0% → 44.2% (K_M); prefill 22.1 → 29.9 tok/s, decode 6.21 → 7.24, resident −0.92 GB |
| 2 | Close the remaining non-IQ fallbacks | **mostly done** (`e35a032bd`) | Q5/Q6/Q4 dense role coverage: optimized bytes 44.2% → 61.3% (K_M), 34.4% → 44.4% (K_S); prefill 30.4 → 41.5 tok/s. Q3_K has no T16 layout — moot for prefill under item 6 |
| 3 | IQ4_XS dense GEMM entry | **done, gfx1151** (`f4a6c8cd4`) | dense = the degenerate single-expert case of the integer-MMQ kernel, no repack; prefill 41.5 → 127.2 tok/s; IQ4_XS kernel 9,280 → 791 ms |
| 4 | Extend the integer MMQ to every expressible quant | **done, later failed admission** (`ef8293dff`) | IQ3_S + IQ4_NL added; prefill 127.2 → 171.9 tok/s. Accepted on teacher-referenced scoring; item 7 reversed that basis |
| 5 | IQ4_NL per-tensor localization | **done, negative** (`fb7592949`) | the cost is not attributable to particular tensors; the bisect measured divergence-from-default instead of gate agreement — the first proxy error below |
| 6 | W4A16 dense IQ prefill kernel | **done, both HIP backends** (`e37888dc4`) | one bf16-WMMA route covers all seven dense IQ quants — no int8 expressibility constraint; registered unrouted at landing |
| 7 | Re-score on the production reference | **done** (`17ba7822a`, `85aac60c7`) | the ranking inverts: the four-quant integer MMQ breaches the 5e-2 max-row ceiling at 0.170390; W4A16 passes every threshold and is the default route (`563cece26`) |

## Where this leaves it

On the published protocol, UD `Q4_K_M` prefill has gone **22.1 → 171.9 tok/s**
with the integer-MMQ route, or **22.1 → 150.9 tok/s** with the admissible
W4A16 route. The prefill gap to the plain `Q4_K_S` file on the same host
narrowed from 13.3× to **1.72×** (294.8 tok/s). **Decode is a separate,
untouched gap: 7.71 against the plain file's 12.56 tok/s (1.63×).** Every
rate is gross, from a power/thermal-limited laptop; absolute numbers are
provisional pending the desktop gfx1151 (section D).

Accuracy, scored per the ruling above over 162 teacher-forced rows
(18 prompts × 9 forced steps, production config, reference = the pre-MMQ
production default):

| Route | Prefill | Mean | p95 | p99 | Max | Rows > 5e-2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Two-quant integer MMQ | 127.2 | 0.001038 | 0.003479 | 0.014385 | 0.049730 | 0 |
| Four-quant integer MMQ | 171.9 | 0.002110 | 0.006797 | 0.024044 | **0.170390** | **1** |
| Four-quant W4A16 | 150.9 | **0.000827** | 0.004547 | 0.012475 | 0.023513 | 0 |

The four-quant integer MMQ breaches the binding 5e-2 absolute maximum-row
ceiling by 3.4× (0.170390 at prompt 10 step 1, where W4A16 scores 0.0003 — a
real distribution failure of that route at that position, not noise; top-1
still agrees there). The two-quant integer MMQ stays inside the ceiling but
marginally fails the mean (1.038e-3 against 1e-3). W4A16 passes every
threshold, so it is the default at a 12% throughput cost against the
four-quant integer route; the integer-MMQ variant name is kept beside the
policy so a route swap is a one-word edit.

W4A16's accuracy is established by this gate measurement, not by arguments
about the format. bf16 has an 8-bit mantissa — exact for the integer codebook
values before scaling, not after — and both candidate kernels write bf16, so
leaf-level comparisons sit on a 1.66e-03 output-rounding floor and cannot
resolve route differences. Only end-to-end gate scoring above that floor is
evidence. See
[`../benchmarks/results/2026-09-09-zbook-ud-production-gate-reference.json`](../benchmarks/results/2026-09-09-zbook-ud-production-gate-reference.json).

### The reference error, recorded so it is not repeated

Three decisions in this campaign were scored against cheap proxies instead of
the gate objective, all the same shape:

1. **Divergence from the incumbent.** The IQ4_NL localization bisected on
   distance from our own default rather than agreement with the reference
   (`fb7592949`). Moving away from the incumbent says nothing about the gate;
   the "promising subset" it produced measured no better against the
   reference.
2. **The output-format floor.** W4A16's first accuracy claim compared leaf
   outputs that both write bf16; the 1.66e-03 rounding floor made the two
   routes indistinguishable (`17ba7822a`).
3. **The external engine.** Every arm was admission-scored against a
   llama.cpp teacher. hipEngine's own production default sits at mean
   0.001271 against that teacher, and that common codec gap dominates the
   teacher-referenced numbers: it compresses arm differences, reorders the
   maxima (against the teacher the four-quant MMQ had the *best* max and
   W4A16 the *worst*; against the production reference exactly reversed), and
   it made every arm appear to fail a 1e-3 mean limit that was never meant
   for that comparison (`85aac60c7`).

Rule: score admission and retention candidate-production vs
incumbent-production, and validate kernel accuracy claims above the
output-format floor.

## Next steps, in order

### A. W4A16 is the default — close its throughput gap

1. The default is W4A16 at 150.9 tok/s, the only measured arm that passes
   the envelope. The integer-MMQ variant name stays beside the policy as a
   one-word swap for bisection. (Policy table and variant names:
   `hipengine/kernels/hip_gfx1151/__init__.py`; the W4A16 promotion renames
   the table to `GGUF_IQ_DENSE_PREFILL_POLICY`.)
2. Tune the W4A16 kernel (`hipengine/kernels/hip_gfx1100/quant/gguf_iq_wmma_prefill.hip`,
   294 lines) toward the integer route's 171.9 tok/s: tile config, unrolls,
   launch bounds. The 12% gap is the target, not a floor.
3. Measure the W4A16 crossover against the strict GEMV. W4A16 has no 128-row
   padding, so its `min_rows` may be lower than the shared 8 — that floor is
   the integer route's measured crossover and is kept shared only so a route
   swap carries no second variable.

### B. Route the remaining quants over W4A16 (kernel work is done)

4. **Q3_K — the largest remaining prefill pot.** 624 ms of the 512-token
   K_M kernel census (worth roughly +15–20% prefill — estimate, to be
   measured), and a 1.01 GB share in K_S. Needs a policy entry, a crossover measurement, and the
   production-referenced gate. No T16 layout and no per-16 integer contract
   is required; the W4A16 expansion handles the per-16 scales directly.
5. **IQ2_S / IQ2_XS:** same treatment after Q3_K. Small in these files (one
   IQ2_XS tensor in K_S), so this is completeness, not throughput.
6. Optional, only if the integer route's speed is wanted back: a hybrid
   policy (integer MMQ where admissible, W4A16 elsewhere) needs its own gate
   run. Expectations are low — even the two-quant integer arm marginally
   fails the mean — so this comes after B, not before.

### C. Decode — the untouched 1.63× gap

7. Decode is 7.71 tok/s against the plain file's 12.56 on the same host.
   rows=1 is below `min_rows` everywhere, so every IQ quant decodes on the
   strict GEMV by design. Start with a decode kernel census to find what
   dominates, then assign decode-side owners;
   [`GFX1100-SHAPE-AWARE-GEMV-CAMPAIGN.md`](GFX1100-SHAPE-AWARE-GEMV-CAMPAIGN.md)
   may hold transferable GEMV tuning.

### D. Desktop gfx1151 re-measure (the move)

8. Re-run the published-protocol sweep on the desktop for every candidate
   arm (two-quant, four-quant, W4A16, +Q3_K once routed). KL scores are
   deterministic and expected to transfer bitwise; confirm with one gate
   run.
9. Re-measure crossovers on the desktop: `min_rows=8` is laptop-derived;
   **IQ3_XXS has never had its own crossover measurement** (it shares the
   IQ4_XS policy and kernel); W4A16's is unmeasured (A3).
10. Re-tune the queued kernel items from the IQ dense campaign (B4, T×R,
    R8-vs-R4) under the W4A16 route — kernel shares have shifted twice since
    that queue was written.
11. Run UD `Q4_K_S` end to end: every perf number in this doc is K_M; K_S
    has a different quant mix and the larger Q3_K share (1.01 GB).
12. Land the desktop numbers in `benchmarks/README.md` (the table is
    structured for them) and drop the "provisional laptop" caveats.

### E. Numerics governance

13. Re-score the earlier teacher-referenced decisions (item 1 repack
    eligibility, item 2 T16 role coverage) on the production reference; both
    predate the ruling.
14. Run the task/category-heldout gate for the promoted W4A16 arm; the
    campaign has no task gate yet.
15. Reconcile `EXECUTION-PROFILES.md` §6's strict-reference wording with the
    2026-09-09 ruling at the top of this doc. This is a human-lead decision;
    this doc does not amend the normative one.
16. Where "top-1 100%" appears in campaign records, state the reference:
    the four-quant integer MMQ is 161/162 against the llama.cpp teacher and
    162/162 against the production reference.

### F. gfx1100 lane (W7900)

17. W4A16 is registered on `hip_gfx1100` but no dense IQ prefill policy
    exists there at all; the integer-MMQ route is gfx1151-only. Port
    decision and measurement when the W7900 lane has time.

## How we got here (historical, preserved)

### The number

Same model, same GPU, same harness, same 512-token prompt, measured
2026-09-08:

| File | Prefill | Prefill GPU time | Weight bytes in an optimized layout |
| --- | ---: | ---: | ---: |
| `Qwen3.8-27B-Q4_K_S` (plain) | 160.5 tok/s | 3,320 ms | 15.03 GB (**94.8%**) |
| `Qwen3.8-27B-UD-Q4_K_M` | 13.77 tok/s | 37,299 ms | 0.00 GB (**0.0%**) |
| `Qwen3.8-27B-UD-Q4_K_S` | — | — | 0.00 GB (**0.0%**) |

Not "mostly raw". **Zero.** Every one of the 554 rank-2 tensors in both UD files
plans to a GEMV-shaped layout, including the Q4_K, Q5_K and Q6_K tensors that
have qualified T16 kernels sitting right there.

### Root cause: one model-wide boolean

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

### What lifting the veto was worth

Planned statically through the real planner with `repack_veto=False`, nothing
else changed:

| File | Optimized bytes, before | With per-tensor veto | Resident weight bytes |
| --- | ---: | ---: | ---: |
| `UD-Q4_K_M` | 0.00 GB (0.0%) | **7.11 GB (44.2%)** | 17.265 → **16.284 GB (−5.7%)** |
| `UD-Q4_K_S` | 0.00 GB (0.0%) | **5.15 GB (34.4%)** | 15.972 → **15.143 GB (−5.2%)** |

Faster **and smaller**: `q4_k_pack8`, the fallback for rank-2 Q4_K, stores
`qweight + scales + mins` at `n*k*0.75` bytes (~6 bpw) where the source Q4_K is
~4.5 bpw and T16 is compact.

### Item 2 notes (role coverage)

Every closed gap was a *measurement-scoped* allowlist, not a kernel limit: each
predicate read "select only the measured …" and was written against the plain
`Q4_K_S` file, which carries no tensor at the missing roles. Every remaining
tensor was verified T16-alignable first, and every missing (role, shape) pair
was already admitted for Q4_K on the identical geometry.

- Q5_K: `_is_dense_h5120_q5_t16_tensor` grew from 3 roles to the full 9-role
  dense set, matching `_DENSE_Q4_T16_SIDECAR_POLICY`. 2.32 GB (K_M) / 1.19 GB
  (K_S) off the raw path.
- Q6_K: the wide predicate gained `ffn_up`, `attn_output`, `ssm_out`; narrow
  `attn_k` follows `attn_v` at the identical (1024, 5120) geometry.
- Q4_K: `ssm_out` (5120, 6144) joined the sidecar policy — the same geometry
  as `attn_output`.

The original residual table, for reference:

| Bytes (K_M / K_S) | Group | Why it stayed raw |
| ---: | --- | --- |
| 2.32 / 1.19 GB | Q5_K `raw_gguf` | role-scoped Q5 T16 policy gaps |
| 0.27 / 1.01 GB | Q3_K `raw_gguf` | no T16 layout exists for Q3_K at all |
| 0.34 / 0.12 GB | Q6_K `raw_gguf` | role gaps in the planar policy |
| — / 0.19 GB | Q4_K `q4_k_pack8` (`ssm_out`) | shape absent from `_DENSE_Q4_T16_SIDECAR_POLICY` |
| 0.72 / — GB | Q4_K `raw_gguf` (`token_embedding`) | correct: the embedding has its own path, not a target |

Q3_K's missing T16 layout is **superseded for prefill** by the W4A16 route
(item 6 / next step B4). `GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES`
excluding `MOSTLY_Q4_K_M` was checked and is not a routing gap: K_M's 28 Q4_K
`ffn_gate`/`ffn_up` tensors already resolve to `gguf_q4_k_t16_v1`, so adding
the stamp would swap one optimized variant for another — a tuning question
with its own measurement, not part of this item.

### Item 3 notes (the IQ4_XS block)

The block was 4.76 GB (K_M) / 6.29 GB (K_S), all `raw_gguf`. Three options
were weighed:

- **A. Dense entry point over the existing IQ integer-MMQ prefill kernels** —
  landed as items 3–4. Fastest measured route (171.9 tok/s four-quant) but
  not admissible on the production reference (item 7).
- **B. An IQ4_XS T16-style repack plus a dense T16 kernel** — not pursued.
- **C. Prefill-only on-the-fly dequant toward a BF16 matmul** — realized in a
  different shape: the W4A16 kernel (item 6) expands weights in-register and
  multiplies bf16 WMMA against the resident bf16 activations, a dedicated
  kernel rather than a workspace plus rocBLAS. It is the default.

## Reproducing the audit

`scratchpad/layout_audit2.py` plans both files through the real planner and
reports optimized-versus-fallback bytes by (quant, layout, role); set
`AUDIT_NO_VETO=1` to plan with `repack_veto=False`. It is CPU-only — it reads
GGUF headers and runs the planner, and never touches the GPU.
