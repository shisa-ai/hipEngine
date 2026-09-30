# GD-4's state-drift screen: PASSED — the chunked fp16 arithmetic is admissible in principle

**Status: SCREEN PASSED** (`20260930T`). `GD-4`'s prescribed first experiment — "screen
state drift over 4K prompts and chunk/tail boundaries before a full production gate" —
is answered. The chunked fp16 reassociation of the GDN recurrence drifts at
**1.1e-07 NMSE**, flat in chunk size, flat in prompt length, with no tail penalty.

**This is not a production admissibility result.** It is a screen on the arithmetic,
and the production gate remains the real test.

## Question

`GD-4` proposes a chunked fp16 WMMA GDN scan (halo-box PR #91) against the sequential
f32 chain the production kernel runs. Its cost gate is measured: the GDN family is
**4.97%** of p512 prefill kernel time, so the fused chunk scan's ceiling is roughly
**5.5%** — and the row notes it is the same lever as `GD-3`, not a separate one.

Its risk is that it **changes arithmetic in the recurrence, which carries state across
the whole prompt**, so the row requires a state-drift screen before any build. No
chunked/scan GDN kernel exists in this tree, so the screen was run on the arithmetic
itself: the same gated recurrence computed in chunks with fp16 operands, fp16
intra-chunk accumulation and an f32 chunk carry, against the sequential f32 chain.

Shape is the model's real GDN geometry: **Hk16/Hv48/Dk=Dv=128**, 3x kv-repeat,
4096 tokens, decay gates log-uniform in (1e-3, 1).

## Result

| variant | state NMSE | out NMSE | max \|dS\| |
| --- | ---: | ---: | ---: |
| chunk 64, fp16 operands + fp16 intra | 1.10599e-07 | 1.57663e-07 | 6.315e-05 |
| chunk 128 | 1.10582e-07 | 1.56688e-07 | 6.316e-05 |
| chunk 256 | 1.10516e-07 | 1.56198e-07 | 6.316e-05 |
| chunk 512 | 1.10602e-07 | 1.55947e-07 | 6.315e-05 |
| chunk 64, fp16 operands, **f32 intra** | 8.46195e-08 | 1.31548e-07 | 6.259e-05 |
| **f32 operands (reassociation only)** — control | 1.9e-13 … 1.5e-11 | 8.1e-14 … 5.3e-12 | ≤ 8.4e-07 |

**Non-divisible tail** (chunk 384, 4096 % 384 = 256): state NMSE **1.105e-07** — the
same as every divisible chunk size, so **no tail penalty**.

**Drift growth over the prompt** (chunk 64): 1.138e-07 at 512 tokens, 1.147e-07 at 1024,
1.099e-07 at 2048, 1.106e-07 at 4096 — **flat, so the drift does not compound**.

## What this establishes

1. **The reference's claim is reproduced in this model's regime.** halo-box PR #91
   reports NMSE ~2e-7 against the sequential kernel; measured here at this shape it is
   **1.1e-07**, slightly better.
2. **The drift is chunk-size-independent** (1.106e-07 at chunk 64 and at chunk 512), so
   a larger chunk — the efficiency-relevant direction — costs no accuracy.
3. **There is no tail penalty and no compounding over the prompt**, which were the two
   specific hazards the row named.
4. **The dominant error is operand rounding, not reassociation order.** Pure f32
   reassociation sits at fp32 round-off (1.9e-13 … 1.5e-11); switching the operands to
   fp16 moves it six orders to 1.1e-07; and fp16 *intra-chunk accumulation* on top of
   that costs only a further 1.3x (1.106e-07 vs 8.462e-08). So the fused fp16
   accumulation the kernel needs is nearly free relative to the operand rounding it
   already implies.
5. **The screen passes, so the row proceeds to the build** rather than closing.

## A bug this screen's own control caught

The first version of the screen omitted the chunk's **first** gate factor on the
cross-chunk and carry terms. The symptom was that `out NMSE` was **identical across all
three variants at each chunk size** (3.265e-03 for all three at chunk 64) while the
f32-reassociation control showed state NMSE 8.7e-13 — an exact state with a 0.16%
output error. That combination is only possible if the discrepancy is structural rather
than numerical, which is what it was. Fixed by carrying the unshifted cumulative
log-gate (`A[t] = exp(sum_{u<=t} log g[u])`) and forming the pairwise ratio in log space
so it stays ≤ 1 and cannot hit 0/0 when `A` underflows.

After the fix the control returns fp32 round-off, which is what makes the fp16 numbers
above signal rather than artifact. **The control variant is the reason to trust the
result**; without it the screen would have reported a plausible 3.3e-03 that was
entirely its own bug.

## Scope and limits

- **Synthetic unit-scale inputs and a NumPy model of the fp16 arithmetic, not the real
  kernel.** Real activations would confirm the regime.
- **State/output NMSE is not a KL measurement and does not establish production
  admissibility.** The binding production bar is the per-scope envelope, and this
  session's own `MX-1` result is the cautionary case: a route whose output error sits
  8.0-9.5x below the BF16 floor already in the path still failed the 18-row
  `prefill_last` scope. The screen says the arithmetic is admissible *in principle*;
  the gate decides production.
- The ~5.5% ceiling is unchanged and is the same lever as `GD-3`.

## Artifacts

- `screen.json` — full payload (variants, tail, growth, shape, host, caveat).
- Harness: `scripts/qwen4exp_gdn_chunk_scan_drift_screen.py` (in-tree, reproducible,
  `--json`).
- Worklog: `worklog/entries/20260930T160236*-qwen4exp-gd4-chunk-scan-drift*.md`.
