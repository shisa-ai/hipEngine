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

---

# Addendum: the cast's bandwidth, measured — and the whole-row estimate refuted

The estimate above assumed both kernels run at the same effective rate. That assumption was
then tested against the raw trace, and it does not hold uniformly.

**Method.** Raw `rocprofv3` kernel trace filtered to the `qwen4exp_prefill_p4096_` marker
window. Elements per launch = `Grid_Size_X * Grid_Size_Y * Grid_Size_Z` **as work-items**,
per §8 rule 2 — *not* multiplied by `Workgroup_Size_X`. This reproduces the artifact's
1,056 launches / 131.001 ms exactly.

| elements/launch | shape | launches | ms | GB | GB/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| **10,485,760** | **1024x10240 (the hc_*_down pair)** | 256 | **83.861** | **16.106** | **192.1** |
| 2,621,440 | 1024x2560 | 544 | 26.255 | 8.556 | 325.9 |
| 6,291,456 | 1024x6144 | 128 | 20.039 | 4.832 | 241.1 |
| 655,360 | 1024x640 | 128 | 0.845 | 0.503 | 595.4 |
| | | 1,056 | **131.001** | **29.998** | **229.0** |

## What the table says

**Apparent bandwidth rises as the transfer shrinks** — 192.1 GB/s at 10.5M elements per
launch, 595.4 GB/s at 655K — and the total, 229.0 GB/s, is **103.6% of the 221 GB/s DRAM
roof**. That is impossible for pure DRAM traffic, so the small roles are partly **L2-served**
and the 6 B/element model is not a DRAM model for them.

## Consequences for DP-5

- **The defensible target is the HC down pair alone.** 16.106 GB at **192.1 GB/s = 86.9% of
  roof**, the largest transfers and the only ones plausibly DRAM-bound. Net saving
  **4/6 x 83.861 = 55.9 ms = 0.36% of the union.**
- **The whole-row figure of 87.3 ms is REFUTED as an over-estimate.** It applies a DRAM byte
  model to roles whose apparent bandwidth exceeds the DRAM roof, i.e. whose traffic is partly
  served by cache and would *not* be saved by eliminating the kernel.
- **This strengthens the row's "epilogue first" order for a second, independent reason.** The
  HC pair is not only 64.0% of the cast by time; it is also the only part whose traffic is
  actually DRAM-bound, so it is where the saving genuinely is.

## A join that had to be discarded on the way

The first attempt joined `captures.child.launch_census` shapes to `audit.prefill` times —
two different captures. It produced plausible per-role numbers but a total of **229.0 GB/s =
103.6% of roof**, which is how the join was caught. **Cross-capture shape/time joins must be
checked against a physical bound before they are used**, and the raw trace is what resolved
it. The shape join happened to be right for the HC pair (10,485,760 = 1024x10240); the
`launches` column was what could not be trusted.
