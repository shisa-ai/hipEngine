# Flash-Next Prefill: Same-File Engine Comparison

> **Correction, 2026-09-16.** Two claims in this record are withdrawn, and one
> number is corrected. The rate table and the per-prefill kernel totals stand;
> see the notes below before quoting anything else.
>
> 1. **The 1.65x x 3.38x gap decomposition is withdrawn.** Telescoping two
>    ratios is algebra, not a causal decomposition. "Kernel quality at equal
>    arithmetic" was never demonstrated: accumulation, operand precision,
>    repair strategy, KV type and execution shape all differ. "1.65x is a lower
>    bound" also does not follow, because selector count does not bound
>    performance and the effects can interact non-monotonically. The historical
>    census is not a measurement of today's code with every selector restored.
> 2. **"Repair is only 6.3%, so correctness is not the bottleneck" is
>    withdrawn.** Explicit repair kernels are only part of the cost of
>    preserving arithmetic. Multi-plane computation, staging and choosing slower
>    primary kernels also contribute, and the dominant projection kernel may
>    itself be a correctness-driven fallback. Slow implementation and
>    arithmetic-policy cost are not mutually exclusive.
> 3. **The dense comparison figure was wrong.** This record said pwilkin's dense
>    and fused-split variants total "about 2770 ms". That was a whole-window sum
>    used as a per-prefill figure. The per-prefill total is **1387.5 ms**
>    (`mmb_dense_kernel<128,128,32,64,1>` 508.7 + `<128,256,64,64,1>` 446.1 +
>    `mmb_f32split_kernel` 429.3 + 3.4), so the dense ratio is **7.3x**, not 3.6x.
> 4. **The `gr_read` row is not comparable across engines.** Upstream reports
>    0.0 ms because it has no hyper-connection kernels at all; that work runs
>    through its generic matmuls and elementwise ops. A zero here means
>    "classified elsewhere", not "not computed".
> 5. **The per-prefill attribution divided a whole-server trace by a supplied
>    prefill count.** It did not delimit measured requests, so startup, warmup
>    and initialization were included. The bucketer also mis-assigned
>    `quantize_mmq_q8_1` to `dense_matmul`, dropped `qsa3_attn_kernel` and the
>    rocBLAS `Cijk_*` GEMM to `unattributed`, and that row's totals are
>    superseded by the request-delimited per-role table in
>    [the 2026-09-16 artifact](../2026-09-16-flashnext-per-role-cost/README.md),
>    which attributes 100% of kernels by tensor role.
>
> The defensible headline is: the profiles identify projection kernels as the
> dominant optimization target, and the historical faster hipEngine profile
> still substantially trails pwilkin, but neither establishes an equal-arithmetic
> comparison.

Measured September 15, 2026 on Framework `gfx1151`, machine
`55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S/40CU. One file throughout:
unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL` (four shards). Every engine below can
open that file; the ones that cannot are named at the end rather than dropped
silently.

Protocol: prefill only (`n_predict=1`), exact token ids from the committed
canonical fixture, twelve cases (four categories at 512/1024/4096), one
unmeasured warmup per case, three measured repetitions, median per case then
equal-weight mean across cases. llama.cpp-family servers ran with
`-ngl 999 -fa on -c 4352 -b 8192 -ub 2048 -t 4 --parallel 1`. hipEngine ran
its own canonical bench at chunk1024 with warm PLE and BF16 KV, its production
default.

## Prefill tok/s

| Engine | Source | KV | 512 | 1K | 4K | 4K vs hipEngine |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| pwilkin `strix-halo` | `40a9f4d01` 2026-09-15 | f16 | 339.2 | 858.9 | **1061.9** | **5.78x** |
| halo-box `strix-llama.cpp` | `69946438a` 2026-09-13 | f16 | 451.6 | 620.7 | 660.6 | 3.60x |
| halo-box `strix-llama.cpp` | `69946438a` | bf16 | 410.0 | 553.3 | 653.2 | 3.56x |
| halo-box `strix-llama.cpp` | `5f85164` (prior comparator) | bf16 | 468.1 | 606.9 | 712.2 | 3.88x |
| upstream llama.cpp | `6011c34ce` 2026-09-15 | f16 | 321.3 | 415.1 | 459.0 | 2.50x |
| upstream llama.cpp | `6011c34ce` | bf16 | 321.0 | 420.3 | 452.9 | 2.47x |
| **hipEngine current default** | `ddfc2a746` | bf16 | **185.2** | **190.6** | **183.6** | 1.00x |

hipEngine's prefill coefficient of variation is 0.55%/0.43%/0.32% and its
per-case medians span 183.1-184.2 at 4K, so its row is flat across categories
and shapes. The engines ahead of it are not flat, which is the first reason a
single headline number for them is misleading.

## How the 4K gap was previously split (withdrawn)

This section previously multiplied a conservative-arithmetic factor by a
kernel-quality factor and presented the product as the gap. That is withdrawn;
see the correction at the top of this file. The two device-time measurements it
used are kept below only because they are still real measurements of their own
configurations.

| Step | Device ms | Ratio |
| --- | ---: | ---: |
| hipEngine current default | 22206.4 | 1.00x |
| Arithmetic-restored census, `53512b509` | 13440.9 | 1.65x |
| pwilkin `40a9f4d01` | 3972.8 | 5.59x |

The census is a profiling capture with `performance_claim: false` and it enabled
six of the fifteen recovery selectors. It is a same-case device total and
nothing more: it does not measure today's code, it does not bound the effect of
the selectors it left off, and the ratio between it and the current default is
not a causal factor in the difference against pwilkin.

## Per-prefill kernel attribution

`code-p4096`, one 4096-token prefill, milliseconds of device time, bucketed by
kernel name. Each llama.cpp window contains a warmup and a measured request, so
its total is divided by two; hipEngine's divisor of two is confirmed
independently by its marker-delimited owner total (22201.9 ms against 22206.5 ms
from the raw trace).

| Work | hipEngine | pwilkin | halo-box | upstream |
| --- | ---: | ---: | ---: | ---: |
| dense quantized matmul | **10749.6** | 1485.2 | 2536.7 | 3133.5 |
| MoE expert gate/up | 3291.2 | 749.8 | 800.2 | 1451.3 |
| MoE expert down | 2301.1 | 509.9 | 199.5 | 200.7 |
| hyper-connection / GR | 2579.8 | 321.3 | 421.5 | 0.0 |
| attention (QSA + paged) | 1420.3 | 82.1 | 140.2 | 169.8 |
| Gated DeltaNet | 842.0 | 269.9 | 294.2 | 1835.6 |
| expert routing / index | 619.6 | 171.5 | 8.6 | 230.4 |
| norm and elementwise | 227.3 | 179.1 | 397.8 | 1505.1 |
| activation packing | 0.0 | 3.7 | 3.4 | 445.0 |
| unattributed | 175.6 | 200.2 | 625.9 | 662.0 |
| **Total** | **22206.4** | **3972.8** | **5428.1** | **9633.5** |

Four rows carry 94% of the 18234 ms difference: dense matmul (+9264 ms),
MoE gate/up and down together (+4332 ms), hyper-connection (+2258 ms) and
attention (+1338 ms).

The dense row is the single largest item and it is **not** conservative
arithmetic. hipEngine's `gguf_k_prefill_out_coltile_rowbatch_kernel` accounts
for 10095.6 ms of it in 3184 launches - the ordinary K-quant dense prefill
kernel. pwilkin's equivalent work runs in three `mmb_dense_kernel` /
`mmb_f32split_kernel` variants totalling **1387.5 ms** per prefill. That is a
**7.3x** difference in the default dense path. (An earlier revision of this
file quoted 2770 ms here, which was a whole-window sum used as a per-prefill
figure.)

The sparse exact repair passes - `gguf_q4_k_selected_dual_sparse_exact_repair_bf16`,
`q5_1_selected_sparse_exact_repair_row_publish`, `q8_0_selected_sparse_repair` -
cost 1396.6 ms, or 6.3% of hipEngine's prefill and 7.7% of the gap. The
risk-collecting iu8 kernels they repair into cost a further 4195.7 ms against
the fork's 1260 ms of fused routed GLU. An earlier revision described that as
"kernel quality, not a correctness tax"; that is withdrawn, because the
primary kernels may themselves be slower in order to preserve arithmetic.

### Why this table is not hipEngine's owner table

Both sides are bucketed by kernel name, so the comparison is like-for-like:
hipEngine's generic K-quant kernel is compared against the fork's dense and
routed kernels regardless of which owner invoked them. hipEngine's own owner
taxonomy attributes by marker span instead, and the two disagree on the rows
where one kernel serves several owners:

| Work | owner ms | bucket ms | delta |
| --- | ---: | ---: | ---: |
| MoE (gate/up + down) | 7413.0 | 5592.3 | -1820.7 |
| hyper-connection / GR | 3788.6 | 2579.8 | -1208.8 |
| attention | 1422.1 | 1420.3 | -1.8 |
| Gated DeltaNet | 837.6 | 842.0 | +4.4 |
| dense matmul | 8630.4 | 10749.6 | +2119.2 |
| **Total** | **22201.9** | **22206.5** | **+4.6** |

The totals agree to 0.02%. `gguf_k_prefill_out` runs under the linear, MoE-down
and GR-down owners, so name-based bucketing puts all of it in one row. Read the
table above as a comparison of kernels doing the same arithmetic, and the owner
table as the account of which part of the model spends the time; do not read
either row as the other.

## What this does not say

- **Not a numerics result.** No engine's output was compared against another's.
  A faster row here is not a qualified row, and hipEngine's conservative
  arithmetic was chosen because the fast composition failed its numerical and
  task gates.
- **Not an MTP or long-context result.** These are single text-AR prefills at
  up to 4096 tokens. pwilkin's lead grows with context (1.83x at 512, 4.51x at
  1K, 5.78x at 4K), which is consistent with its recent sparse-QSA and indexer
  work, but this measurement does not isolate that mechanism.
- **Not a decode result.** Decode is a separate campaign.
- **KV dtype is not matched.** pwilkin `40a9f4d01` asserts at model load with
  BF16 KV (`src/models/qwen4exp.cpp:1360` requires k/v F16 on the sparse-QSA
  direct-indices path), so its arm runs F16. Re-running the other engines in
  both dtypes bounds the effect at about 1% (halo-box 660.6 f16 vs 653.2 bf16;
  upstream 459.0 f16 vs 452.9 bf16). hipEngine's QSA kernels are registered for
  `bf16_kv` only, so it has no matched arm.

## Engines that could not run this file

- **pwilkin `master`** (`e8e6c7af2`, 2026-07-22) has no `qwen4exp` architecture
  at all and cannot load the model. The live Flash-Next work is on branch
  `strix-halo`. A "latest pwilkin" comparison that checks out the default branch
  compares against a tree that cannot open the file.
- **halogen** refuses this quant family by name at startup: its README lists
  `Q4_K`, `Q5_K`, `Q5_1` and `Q4_1` - "unsloth's `UD-Q4_K_XL`" - as refused
  before anything is loaded. Its published 1246-1424 tok/s GGUF figures are on
  unsloth `UD-IQ4_XS`, a different file. Comparing them to this table would be a
  quant and configuration comparison, not an engine one.

## Reproducing

```bash
.venv/bin/python benchmarks/results/2026-09-15-flashnext-engine-comparison/assemble.py \
  --raw-root /tmp/comparators-20260915/r2 \
  --attribution /tmp/comparators-20260915/per-prefill-attribution.json \
  --hipengine /tmp/comparators-20260915/r2/hipengine-current.json \
  --hipengine-profile /tmp/hipengine-gap-20260915/current-4k-profiled.json \
  --census benchmarks/results/2026-09-14-journey-progress/artifact.json \
  --output benchmarks/results/2026-09-15-flashnext-engine-comparison/artifact.json
```

Raw per-engine outputs are under `/tmp/comparators-20260915/r2/`; the
attribution table is regenerated by
`scripts/qwen4exp_per_prefill_attribution.py`. Engine sources are fresh clones
under `/home/lhl/comparators-20260915/` with builds in `build-upstream`,
`build-pwilkin` and `build-halobox`, each built by
`/home/lhl/llama.cpp/build-hip.sh` for `gfx1151`. Every row records its server
binary sha256 and source commit in `artifact.json`.
