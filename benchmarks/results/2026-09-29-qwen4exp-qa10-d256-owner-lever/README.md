# QA-10: the D256 sparse-attention owner has no K-tile or occupancy lever to configure

**Date:** 2026-09-29
**Host:** Radeon 8060S (gfx1151), hostname `gfx1151`, machine-id
`55ea6c509d0b49eea8de7094a1023668`
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL
**Command:** `python3 scripts/qwen4exp_repair_register_census.py --source
hipengine/kernels/hip_gfx1100/attention/qwen4_exp_qsa.hip --threads-per-block 256
--output benchmarks/results/2026-09-29-qwen4exp-qa10-d256-owner-lever/census.json`
**Artifact:** `census.json` (this directory) - static, no GPU execution
**Tree:** `7ebd2f68b`

## The question

`QA-10` was parked "historically blocked, not tested", with the action:

> After `X-3`, inspect the actual owner for an equivalent configurable K
> tile/occupancy lever; if present, screen a bounded pair at real 24/2/256
> geometry and tails, without changing arithmetic. Charge scratch and
> complete-owner time; do not blindly copy the external `nbatch_K` constant.

`X-3` is **CAPTURED** at `2789c19d6`, so the dependency is satisfied. The
external reference (nasone32 `ed11a0d2f`) had an `nbatch_K` constant; the
question is whether our owner has an equivalent lever.

## Verdict

**No applicable lever. Close `QA-10`.**

## The owner, and what it actually does

The D256 / 24-query-head / 2-KV-head geometry is owned by:

| role | variant | implementation |
|---|---|---|
| prefill (`prefill_h256_page256_sparse_rows_ge16`) | `strict_rows_spans` | `qsa_sparse_attention_paged_bf16_rows_f32_kernel` |
| decode c1 (`qwen4exp_c1_h256_indexed_sparse_decode`) | `strict_spans` | `qsa_sparse_attention_paged_bf16_f32_kernel` |

Both live in `hipengine/kernels/hip_gfx1100/attention/qwen4_exp_qsa.{hip,py}`.

The kernel's parallel decomposition is `grid = (query_heads, rows)`,
`block = (head_dim, 1, 1)`, and its inner loop is:

```
for (selected_index = 0; selected_index < selected_count; ++selected_index) {
    ... product = query_row[dimension] * key[...]     // one element per thread
    partial[dimension] = product;
    __syncthreads();
    for (stride = blockDim.x/2; stride > 0; stride >>= 1) {   // full block-wide
        if (dimension < stride) partial[dimension] += ...;     // tree reduction
        __syncthreads();
    }
    ... online-softmax update on thread 0
}
```

**`head_dim` is consumed in a single block-wide reduction per selected position.
There is no K tiling to configure**, and `grep -niE 'nbatch|k_tile|ktile|k_batch|TILE_K'`
over the whole `.hip` source returns **nothing** - the external `nbatch_K` concept
has no analogue here. The row's instruction not to copy that constant is
therefore not merely a caution; there is no corresponding site.

## The occupancy half, measured

From `census.json`, gfx11 model (65536 VGPR/CU, 8-VGPR granules, 32-wave/CU
ceiling):

| kernel | VGPR | SGPR | static ins | waves/CU | occupancy |
|---|---:|---:|---:|---:|---:|
| `qsa_sparse_attention_paged_bf16_rows_f32_kernel` (**prefill owner**) | **16** | 49 | 280 | **32** | **100.0%** |
| `qsa_sparse_attention_paged_bf16_f32_kernel` (**decode owner**) | **18** | 50 | 275 | **32** | **100.0%** |
| `..._rows_wave32_f32_kernel` (sibling) | 26 | 49 | 274 | 32 | 100.0% |
| `..._h256_wave_rows_f32_kernel<true,4>` (sibling) | 66 | 44 | 331 | 28 | 87.5% |
| `..._h256_wave_rows_f32_kernel<true,8>` (sibling) | 108 | 44 | 442 | 18 | 56.2% |
| `..._ordered_values_v2_f32_kernel` | 95 | 62 | 512 | 21 | 65.6% |

**Both owners are at the architectural ceiling.** VGPR=16 and 18 sit far below
any granule threshold, so there is no register pressure to relieve and no
occupancy headroom to buy. Adding `__launch_bounds__` cannot raise a kernel that
is already at 32 waves/CU.

**The census discriminates**, which is what makes the 100% credible rather than a
default: three sibling kernels in the same file land at 87.5%, 65.6% and 56.2%,
so the measurement is sensitive to register pressure and is not reporting a
constant.

## What this does and does not close

**Closed:** the two axes the row names - a configurable **K tile** (does not
exist) and **occupancy** (already maximal). A bounded pair at real 24/2/256
geometry cannot be constructed because there is no knob to vary. The row's own
status anticipated this outcome: "close if no applicable lever".

**Not closed, and explicitly out of this row's scope:** the kernel's actual
inefficiency is visible in the loop above and is **not** a tiling or occupancy
problem - it is a full block-wide tree reduction with `2 * log2(blockDim.x)`
`__syncthreads()` calls **per selected position**. That is an issue-limited
structure, and it is `QA-2`'s subject ("The kernel is issue-limited, not
traffic-limited"). `QA-10` itself routes it away: "A changed reduction belongs
under §3.1." Nothing here measures or proposes a reduction change.

**Also not closed:** whether a different *decomposition* (e.g. splitting head_dim
across warps rather than one block-wide reduction) would help. That is a changed
reduction, i.e. `QA-2`/§3.1 territory, not a configuration lever.

## Limits

Static census only - **no GPU execution, no timing, no performance claim**. The
census is one arch (`gfx1151`), one compile flag set, one `--threads-per-block`
value (256, matching the D256 geometry). Register counts are the assembler's
`next_free_vgpr`/`next_free_sgpr` directives; occupancy is the gfx11 model's
arithmetic on those counts, not a measured `occupancy` counter. The K-tile
conclusion is a source property of this tree at `7ebd2f68b` and would need
re-reading if the kernel's loop structure changes.

## Reproduce

```bash
TH=/home/lhl/miniforge3/envs/therock10-staging-20260828
SP=$TH/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$SP/_rocm_sdk_core/lib:$SP/_rocm_sdk_devel/lib:$SP/_rocm_sdk_libraries/lib:$SP/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SP/_rocm_sdk_devel/lib/rocm_sysdeps/lib:$LD_LIBRARY_PATH"
export PATH="$TH/bin:$PATH"
$TH/bin/python scripts/qwen4exp_repair_register_census.py \
  --source hipengine/kernels/hip_gfx1100/attention/qwen4_exp_qsa.hip \
  --threads-per-block 256 --output /tmp/qa10-census.json
```

`scripts/qwen4exp_repair_register_census.py` was written for the IU8 repair
family (`EG-3`) but takes `--source` and `--threads-per-block`, so it applies
unchanged to any single kernel source in this tree.
