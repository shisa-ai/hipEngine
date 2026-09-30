# DP-5 consumer census: the HC down projections are 64% of the cast, so the row's order is right

Completes DP-5's named prerequisite - *"it does need the consumer census to confirm the HC
mix epilogue feeds the most dense consumers before it is built first"* - from the committed
X-3 capture. Read-only re-read; no GPU, no capture, no kernel. `performance_claim: false`.

- **Source:** `benchmarks/results/2026-09-28-qwen4exp-current-attribution/artifact.json`
  (`2789c19d6`), pointer `audit.prefill.windows[0].engagement_by_role_kernel` joined to
  `kernel_names` by `kernel_id`. Case `code-p4096`, `kernel_union_ms` = 15,741.963266.

## DP-5's 131.001 ms is ONE kernel

`q8_0_f32_to_f16_kernel`, launched under **12 distinct roles**, all on **layers 16-47** -
the promoted 16-47 scope, 32 layers.

| role | ms | launches | layers |
| --- | ---: | ---: | ---: |
| `linear:layers.*.hc_ffn_down` | **42.324** | 128 | 32 |
| `linear:layers.*.hc_attn_down` | **41.536** | 128 | 32 |
| `linear:layers.*.ssm_out` | 14.921 | 96 | 24 |
| `linear:layers.*.attn_gate` | 6.789 | 96 | 24 |
| `linear:layers.*.shared_gate` | 6.743 | 128 | 32 |
| `linear:layers.*.attn_output` | 5.118 | 32 | 8 |
| `linear:layers.*.attn_qkv` | 5.102 | 96 | 24 |
| `linear:layers.*.shared_up` | 3.096 | 128 | 32 |
| `linear:layers.*.attn_k` | 2.143 | 32 | 8 |
| `linear:layers.*.attn_q` | 1.589 | 32 | 8 |
| `linear:layers.*.shared_down` | 0.845 | 128 | 32 |
| `linear:layers.*.attn_v` | 0.793 | 32 | 8 |
| **total** | **131.001** | | |

## The verdict on the row's proposed order

**CONFIRMED, and it is not close.** The two HC down projections are
**83.861 ms = 64.0% of the cast total** - the largest single block by a wide margin, more
than the other ten roles combined. Moving the F16 write into the HC mix epilogue first is
the right order.

## Each HC down role is exactly the three-stage pattern DP-5 describes

| stage | `hc_ffn_down` | `hc_attn_down` |
| --- | ---: | ---: |
| producer `gguf_k_prefill_out_coltile_rowbatch_kernel<float, float, 8, 8, 4, true>` | 183.643 ms / 64 launches | 182.713 ms / 64 launches |
| **cast `q8_0_f32_to_f16_kernel`** | **42.324 ms / 128 launches** | **41.536 ms / 128 launches** |
| consumer `q8_0_dense_wide_kernel<_Float16, 128, 256, 64, 64, true>` | 128.378 ms / 128 launches | 127.538 ms / 128 launches |

The cast exists solely to let the F16 `dense_wide` consumer read the producer's F32 output.
The producer is already the model's largest kernel (3,367.698 ms, 21.39% of the union), so
the epilogue is where the F16 store belongs - which is also gufo's shape (*"the HC mix
epilogue writes the mixed rows as F32 + F16 + tiled Q8"*, i.e. the F32 output is retained).

## What the change would recover, as an estimate

Per element the cast reads F32 (4 B) and writes F16 (2 B) = **6 B**; the producer would add a
**2 B** F16 store. Net traffic change **-4 B per element**, so roughly **4/6 of the cast
time** *if both kernels run at the same effective rate*:

- whole row: 4/6 x 131.001 = **87.3 ms = 0.55% of the union**
- HC down pair: 4/6 x 83.861 = **55.9 ms = 0.36%**

**This is an estimate, not a measurement.** The cast may be launch-bound rather than
bandwidth-bound (0.331 ms per launch on `hc_ffn_down`), in which case deleting it recovers
more than 4/6; conversely the producer's epilogue is not free. The change is exact - the F16
values are identical to the staged ones - so it needs no production gate, only an
end-to-end A/B.

## Limits

One model, one quant, one host (gfx1151, not the gfx1100 default), one case
(`code-p4096`), one window, one execution profile, prefill only. Attribution is a
per-role-per-kernel join inside a marked window and is diagnostic, not a wall measurement.
No speedup claimed and no kernel built.
