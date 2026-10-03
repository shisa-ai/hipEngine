# GR up row-batch widening beyond 4 is dead, and the mechanism is occupancy

**Result: `ROW_BATCH` 8 with `HIDDEN_TILE` 2 loses 0.833-0.844x against the
production `ROW_BATCH` 4 parent on the real weight banks, and `ROW_BATCH` 16
loses 0.43-0.44x. Every arm is bit-identical to the parent. The loss is a
measured register/occupancy effect, not a modelling one, and it also refutes the
cost attribution this lever was derived from: the kernel's 256x weight re-read is
**not** what its time consists of.**

Date 2026-10-04. Host gfx1151 (Strix Halo, Radeon 8060S, machine
`55ea6c509d0b49eea8de7094a1023668`). Model `unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL`,
UD-Q4_K_XL, Q8_0 GR up tensors. Base commit `cc364a6d1` plus the uncommitted
candidate diff (the candidate instantiations; the screen records
`tracked_clean: false` for that reason).

## What was tried and why

`q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32` is the largest single
kernel in a `code-p4096` prefill: 2507 ms, 15.5% of the kernel union
(`../2026-09-27-gr-up-cost-attribution/`). That attribution models 90.3% of its
987 MB per launch as a **256x re-read** of the 3.48 MB packed Q8_0 weight
(`row_base = blockIdx.y * ROW_BATCH`, 256 row-batches each sweeping all 1280
hidden tiles) and reads the kernel as weight-bandwidth-bound at 59% of DRAM peak.
Halving that re-read is the obvious lever, and it was already tried twice in the
constant-accumulator form the ledger records as dead: RB8/H1 (0.848x) and RB2/H4
(1.005x), both on 2026-09-06.

Neither of those tested the configuration this packet tests. `ROW_BATCH` 8 with
`HIDDEN_TILE` 2 keeps every operand load count per FMA the same as the parent
while doubling the FMAs amortized over one weight-decode set (the parent decodes
8 weights per k-step per thread for 32 FMAs; this decodes the same 8 for 64), so
it separates "fewer weight bytes" from "fewer weight decodes per output". The
2026-09-06 RB8 arm traded one for the other (4 decodes, 8 activation loads).

`ROW_BATCH` was promoted from a `constexpr` to a template parameter of the same
kernel body for the screen. The per-output K order, the shuffle tree, the
four-wave LDS publication, the serial wave sum and the epilogue are unchanged, so
every instantiation must be bit-identical; the screen asserts that on real
weights rather than assuming it.

## Measured

Same-host, real Q8_0 weights from two layers, `rows` 512 and 1024, `K=320`,
`N=10240`, 128 threads, 20 order-balanced pairs per case, medians. The
`wave-scale` parent is the production decode path, so both arms match production
except for the width.

| arm | tensor | rows | parent ms | candidate ms | speedup | parent-first | candidate-first |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RB8/HT2 | `hc_attn_up` | 512 | 3.2491 | 3.8478 | **0.8444** | 0.8607 | 0.8233 |
| RB8/HT2 | `hc_attn_up` | 1024 | 6.3736 | 7.6285 | **0.8355** | 0.8531 | 0.8182 |
| RB8/HT2 | `hc_ffn_up` | 512 | 3.2304 | 3.8383 | **0.8416** | 0.8656 | 0.8238 |
| RB8/HT2 | `hc_ffn_up` | 1024 | 6.3527 | 7.6258 | **0.8330** | 0.8304 | 0.8292 |
| RB16/HT2 | `hc_attn_up` | 512 | 3.2497 | 7.4265 | **0.4376** | 0.4492 | 0.4343 |
| RB16/HT2 | `hc_attn_up` | 1024 | 6.3380 | 14.7458 | **0.4298** | 0.4335 | 0.4290 |
| RB16/HT2 | `hc_ffn_up` | 512 | 3.2279 | 7.3921 | **0.4367** | 0.4469 | 0.4323 |
| RB16/HT2 | `hc_ffn_up` | 1024 | 6.3921 | 14.7233 | **0.4341** | 0.4339 | 0.4328 |

`exact: true` on all 32 case rows in both runs: `gate` and `mixed` are
`assert_array_equal` on the raw `uint32` bits against the parent, twice per case.
The parent medians reproduce the recorded production shape (6.53-6.66 ms at
rows=1024, against the attribution's 6.529 ms), and the wave-scale parent is
1.9-4.4% faster than the plain parent, consistent with the 2026-09-06 wave-scale
promotion.

A second pair of runs used the plain (non-wave-scale) RB4 parent and is recorded
in the same directory: RB8 0.866-0.873x, RB16 0.446-0.455x. Both comparisons
agree, so the negative does not depend on which parent the candidate is given.

A confirmation run after the candidate instantiations were finalized (rows=1024
only, same protocol) reproduces it: RB8 **0.8462x** on `hc_attn_up` (parent
6.4692 ms, candidate 7.6453 ms) and **0.8437x** on `hc_ffn_up` (6.4150 / 7.6038),
both exact. In the `hc_attn_up` case of that run the parent's own samples split by
order - ~6.3 ms when it ran second, ~8.0 ms when it ran first - which makes the
median-based headline *generous to the parent* (it takes the parent's fast
samples) and therefore conservative for this negative; the `candidate_first`
stratum agrees (0.8244x) while the `parent_first` stratum reports 1.0686x for that
one case. The four-case run above has consistent strata on every case, so the
order effect is not what produces the result. It is recorded rather than smoothed
over because a first-slot parent slowdown is a harness behaviour that any screen
using this script should know about.

## Mechanism: the width costs occupancy, and this kernel is latency-bound

Static census of the compiled profile (`scripts/qwen4exp_repair_register_census.py`,
gfx11 model: 65536 VGPR/CU, 32 waves/CU ceiling, 8-VGPR granules), full artifact
in `register-census-gr.json`:

| instantiation | VGPR | SGPR | static instructions | waves/CU | occupancy | blocks/CU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `<false, true, 4>` (parent) | 72 | 44 | 320 | 28 | 87.5% | 7 |
| `<true, true, 4>` (wave-scale parent) | 72 | 62 | 311 | 28 | 87.5% | 7 |
| `<false, false, 4>` (no-gate) | 72 | 42 | 301 | 28 | 87.5% | 7 |
| `<false, true, 8>` | **130** | 56 | 453 | **15** | **46.9%** | 3 |
| `<true, true, 8>` | **130** | 74 | 430 | **15** | **46.9%** | 3 |
| `<true, true, 16>` | **256** | 98 | 829 | **8** | **25.0%** | 2 |

The accumulator array is `acc[ROW_BATCH][4][2]`, so the width is a direct register
cost: 32 → 64 → 128 float accumulators per thread. 130 VGPR crosses the 128-VGPR
granule and halves resident waves; 256 VGPR is the whole register file for one
thread and leaves 8 waves. No arm spills (scratch is 0 in the screen's own
resource record), so this is pure occupancy loss, not spill traffic.

**The two effects therefore trade against each other and occupancy wins.** RB8
removes half of the weight-decode instructions per output and gains nothing
measurable from it, while losing 40.6 percentage points of occupancy and 16% of
time. RB16 removes three quarters of them and loses 2.3x.

## What this refutes

The cost attribution that motivated the lever reads the kernel as
weight-bandwidth-bound with 90.3% of its traffic in the re-read. **That model does
not survive a direct test.** Halving the re-read (RB8) is 16% *slower*; quartering
it (RB16) is 2.3x slower. A DRAM-bandwidth-bound kernel cannot behave that way,
and the earlier `HC-3` result points the same direction from the other side: a
dead 42 MB store that is 4.2% of the modelled traffic is a 2% *regression* when
removed.

The consistent reading is that the re-read is served from cache (the whole weight
tensor is 3.48 MB, and per-block reuse is within a 1280-tile sweep), that the
kernel is latency-bound at 87.5% occupancy, and that its real cost is the
per-output work that the tiling imposes - the 128-way reduction tree per output
(4 waves x 5 shuffle levels), the per-row-batch weight decode, and the short
2.5-iteration K loop. That is exactly what the `HC-1` LDS-staged tile removes:
BM=128 rows per block with no per-output tree and one weight decode per 128 rows.
`HC-1` measured 2.388x and is rejected on numerics, so the direction is not
reopened by this packet - but its justification should not be restated as a
weight-traffic saving, because the weight traffic is not what costs the time.

## Limits

Microbenchmark on real weights with synthetic activations and synthetic
normalized/gate operands; main kernel only; two tensors of one model; `rows` 512
and 1024 only (the chunk-1024 production shape); one host, one arch, one compiled
profile. The census is a static register model, not a counter measurement - it
explains the direction and the size of the loss but no hardware counter was read.
No default, flag or dispatch changed, and nothing was promoted.

## Reproduce

```bash
# candidate instantiations live on the working tree at cc364a6d1 + the diff
python3 scripts/qwen4exp_q8_wave_scale_screen.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --gr-composite \
  --gr-parent gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_wave_scale_f32 \
  --gr-candidate gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch8_wave_scale_f32 \
  --rows 512 1024 --pairs 20 --output /tmp/gr-ws-rowbatch8.json

python3 scripts/qwen4exp_repair_register_census.py \
  --source hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip \
  --threads-per-block 128 --output /tmp/gr-rowbatch-census.json
```
