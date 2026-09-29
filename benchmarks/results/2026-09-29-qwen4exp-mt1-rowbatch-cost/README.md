# MT-1: the width-invariant body pass already exists, and it is sub-linear

MT-1's stated blocker: the deferred-head verifier is exact but "the missing piece
is a **body pass whose cost grows sub-linearly with rows** while keeping c1
arithmetic". Both halves of that requirement are **already satisfied** by an
existing kernel in the q8_0 linear family:

* **bit-exact with serial c1 decode, per row, at every width 2-8** (RED first);
* **sub-linear in rows** - 8x the rows for 2.42x the cost, **1.85x** faster than
  serial c1 at 8 rows.

No new kernel is needed for this half of MT-1. What remains is wiring, not
arithmetic.

## The arithmetic half (RED, done first)

Two links were already tested independently, but never composed:

| link | test | asserts |
|---|---|---|
| c1 decode == scalar GEMV | `test_q8_0_pack8_f32_matches_scalar_gemv_bits` | bit-exact, rows=1 |
| rowbatch == scalar GEMV | `test_q8_0_f32_rowbatch32_matches_scalar_prefill_bits` | bit-exact, rows=37 |

Link 2 is checked at a **single** batch width, and neither test asserts
**batch-composition invariance** - that row `i` of a batched launch equals what
serial c1 decode produced for that row, at every width. That is the property a
width-invariant verifier actually needs, and it was the real gap.

New test, RED-first, against the serial c1 kernel directly with no scalar-GEMV
intermediary:

`test_q8_0_coltile8_rowbatch4_is_width_invariant_to_serial_c1[2..8]`

It runs `gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out` at width `rows` and
compares against `rows` separate single-row `gguf_q8_0_pack8_gemv_f32_f32_out`
launches, `assert_array_equal`. **7/7 widths pass.**

## The cost half

`q8_0` GEMV pair at the `attn_qkv` shape (2560 -> 10240, DE-3's largest owner),
median of 30 reps after 5 warmups, device-synchronized both sides:

| rows | batched | serial c1 | speedup | per-row batched | per-row serial |
|---:|---:|---:|---:|---:|---:|
| 1 | 82.1 us | 65.9 us | **0.80x** | 82.1 | 65.9 |
| 2 | 94.7 us | 108.2 us | 1.14x | 47.3 | 54.1 |
| 4 | 116.1 us | 193.6 us | 1.67x | 29.0 | 48.4 |
| 8 | 198.8 us | 367.8 us | **1.85x** | **24.8** | 46.0 |

Per-row cost falls **82.1 -> 24.8 us (3.3x)**. The growth is clearly sub-linear:
8x the rows costs 2.42x.

**rows=1 is slower (0.80x).** Batching only wins from rows>=2, which is the right
regime for a verifier that consumes 2-8 draft rows - but it means the pass must
not be used at width 1.

## What this does not establish

- **Not an end-to-end verifier speedup.** This measures the q8_0 GEMV kernel pair
  only. MT-1 needs the *whole* target pass to be sub-linear; this is one
  component, and it is the component DE-3 owns.
- **Does not cover DE-1** (the MoE expert GEMV pair, 11.3 ms/token). Only the
  q8_0 dense linear path was tested. DE-1 is a separate link in the same chain.
- **Part of the 1.85x is launch-count reduction, not kernel throughput.** The
  serial arm issues `rows` launches and the batched arm one, so the comparison
  includes launch overhead - which is the cost a serial verifier genuinely pays,
  but it should not be read as a pure throughput ratio.
- No MTP or speculative-decode end-to-end run was performed.

## Why this matters for MT-1

The row was scoped as "pick the two largest decode owners and **build** rows<=8
variants that reproduce the c1 kernels' per-output reduction bit for bit". For the
q8_0 linear family that build is **already done and already registered**: the
kernel is in the prefill path today (`census.json` shows it at rows=1024 in the
p4096 prefill). The remaining work is composing it into a verifier body pass and
measuring the complete pass, which is a different and smaller unit than the row
assumed.

## Limits

- One host (gfx1151), one quant (q8_0), one shape (2560 -> 10240).
- Kernel-level timing only; no model, no end-to-end request.
- The 30-rep median is of the launch sequence, so it includes host-side launch
  cost in both arms by construction.
