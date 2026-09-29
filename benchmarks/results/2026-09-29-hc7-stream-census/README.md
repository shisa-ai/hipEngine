# HC-7 stream-boundary census: the class ceiling is 2% of prefill

**Date:** 2026-09-29
**Source:** `benchmarks/results/2026-09-28-qwen4exp-current-attribution/artifact.json`
(case `code-p4096`, chunk 1024, Qwen3.8-Flash-Next UD-Q4_K_XL, BF16 KV, Radeon
8060S gfx1151). No new measurement was taken; this is a census over an existing
capture.
**Artifact:** `census.json` (this directory)

## Why this census exists

Ledger HC-7 asks for a last-reader census of the remaining stream intermediates
before any of them is removed or fused. HC-3 is the reason to do it: that item
was recorded as "no arithmetic risk → remove the write" and turned out to cost
**2% more** when the write was removed. A census that only classifies writes as
dead or live would repeat that mistake, so this one prices the owners.

## Finding 1 — the prefill is kernel-bound, and the copies are not the story

In the memory capture window, 15600.44 ms of kernels sit inside a 16159.40 ms
window, leaving 558.97 ms outside. Adding device copies to the union moves that
outside figure by 2.71 ms (558.97 → 556.26), so copies are not a hidden cost.

The raw host-API table looks alarming and is a trap:

```
hipMemcpy      246 calls   14086.59 ms   <- host interval, NOT transfer
hipLaunchKernel 9622 calls    338.08 ms
hipMemcpyAsync   48 calls       1.86 ms
```

**Only 5 device copy rows exist in the window, totalling 2.711 ms.** The
14086 ms is host time blocked in `hipMemcpy` waiting for the stream to drain,
not bytes moved. Do not cite it as reclaimable.

## Finding 2 — the class ceiling is 2.03% of prefill

The family taxonomy in the audit window sums to exactly its kernel union
(15741.963 ms), so attribution is complete:

| family | ms | share |
|---|---:|---:|
| dense_projection | 4553.007 | 28.9% |
| expert_gate_up | 3375.634 | 21.4% |
| hyper_connection | 2587.665 | 16.4% |
| expert_down | 2230.120 | 14.2% |
| qsa_attention | 1040.374 | 6.6% |
| gdn | 844.295 | 5.4% |
| moe_reduce | 763.833 | 4.9% |
| **elementwise_norm** | **319.494** | **2.03%** |
| indexer | 17.520 | 0.11% |
| ple | 9.834 | 0.06% |
| other | 0.188 | 0.00% |

Every stream elementwise op, conversion, and norm lives in `elementwise_norm`.
That family is **319.494 ms of a 15741.963 ms prefill**. So no amount of
stream-boundary work inside HC-7 can reach beyond ~2% of prefill, and any
individual intermediate is a fraction of that.

## Finding 3 — one redundant intermediate, priced at 0.10%

`run_qwen4_exp_qsa_prefill_layer` (and its GDN twin) runs two kernels in
sequence:

```python
bf16_to_f32(moe.output.ptr, scratch.moe_f32.ptr, rows * hidden, ...)
qwen4_exp_gr_write_bf16_f32(
    scratch.after_attention.ptr, scratch.moe_f32.ptr,
    ffn_read.inject_logits.ptr, scratch.output.ptr, rows, branches, hidden, ...)
```

`scratch.moe_f32` is written by the first and read only by the second. The
combine kernel is:

```c
const float value = bf16_bits_to_float(residual[offset + column])
    + scatter * block_output[block_offset + column];
```

`block_output` enters **only as a read**, multiplied by `scatter`. So the pair
fuses into one kernel that reads `moe.output` as bf16 and converts inline, and
the result is **bit-identical** — no arithmetic change, no rounding boundary
crossed. This is a strict fusion, not a T1/T2 one.

Priced at rows=1024, hidden=2560:

- round trip removed = `1024 × 2560 × 8 B` = 21.0 MB per layer-step
- 48 layers = 1.0 GB per chunk, ~4.0 GB per `code-p4096` prefill
- ≈ 16 ms at 250 GB/s = **0.10% of the prefill**

That is below this protocol's measurement floor.

## Verdict

**HC-7 has a low ceiling.** The class ceiling is 2.03% of prefill and the one
redundant intermediate found is priced at 0.10%. The `moe_f32` fusion is worth
doing as a free bit-identical cleanup — it removes a real allocation and a real
round trip — but it must never be reported as a performance win, because it
cannot be measured above noise on this host.

The productive place to spend effort is not here: `dense_projection` +
`expert_gate_up` + `hyper_connection` + `expert_down` are 80.9% of the prefill
between them.

## Provenance caveat

This census reads an existing artifact rather than taking new measurements, so
its host identity is that artifact's: Radeon 8060S (gfx1151), machine-id
`55ea6c509d0b49eea8de7094a1023668`, kernel 7.1.6-1-cachyos. The underlying
capture is a single `code-p4096` prefill with synthetic-free natural prompt
tokens, not category/heldout qualification and not a performance A/B. The two
windows in that artifact are separate runs; every ratio above is computed within
one window and names which.
