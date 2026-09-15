# Flash-Next Per-Role Prefill Cost

One 4096-token prefill of `code-p4096`, current production default, measured
2026-09-16 on Framework `gfx1151` (machine `55ea6c509d0b49eea8de7094a1023668`,
Radeon 8060S), chunk1024, BF16 KV, warm PLE.

This replaces a kernel-name attribution that could not answer the question it
was being used for. A profile showing
`gguf_k_prefill_out_coltile_rowbatch_kernel` does not say whether that time went
to an attention projection, a hyper-connection projection or a shared-expert
FFN, because one variant serves all of them.

## How the numbers were obtained

Two records from the same process, joined on the tensor slot path:

1. A role-marked `rocprofv3` capture (`--profile --role-markers`). The owner
   entry points push a ROCTX range naming the tensor
   (`qwen4exp_role:linear:layers.15.attn_q`), and every kernel dispatch is
   attributed to the innermost enclosing range. **Attribution is 100%:** 9652
   kernels, 22368.1 ms attributed over a 22932.5 ms window, 0 ms unattributed.
2. A launch census (`--launch-census`) that records the quant type, `K`, `N`,
   row count and launch count behind each owner call.

Joining them turns a kernel symbol into a tensor, a shape and a launch count.
Roles are normalized over layer index, so the table has one row per kind of
projection rather than one row per layer.

The census is reset after the warmup prefill, so graph instantiation, allocator
growth and cache population are not attributed to the measured pass.

## Per-role cost

| Role | ms | % | K | N | launches | GFLOP | GFLOP/s | % of FP32 peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `moe:expert_gate` | **6667.6** | **29.8** | — | — | — | — | — | — |
| `linear:attn_qkv` | 2724.2 | 12.2 | 2560 | 10240 | 144 | 7730.9 | 2837.9 | 19.2 |
| `linear:ssm_out` | 1967.1 | 8.8 | 6144 | 2560 | 144 | 4638.6 | 2358.1 | 15.9 |
| `linear:attn_gate` | 1584.2 | 7.1 | 2560 | 6144 | 144 | 4638.6 | 2928.0 | 19.8 |
| `qsa_prefill:attn_q` | 1423.6 | 6.4 | — | — | — | — | — | — |
| `gr_read:hc_attn_down` | 1304.6 | 5.8 | — | — | — | — | — | — |
| `gr_read:hc_ffn_down` | 1291.7 | 5.8 | — | — | — | — | — | — |
| `linear:attn_q` | 1282.3 | 5.7 | 2560 | 12288 | 48 | 3092.4 | 2411.6 | 16.3 |
| `gdn:attn_qkv` | 831.3 | 3.7 | — | — | — | — | — | — |
| `linear:attn_output` | 659.6 | 3.0 | 6144 | 2560 | 48 | 1546.2 | 2344.0 | 15.8 |
| `linear:hc_attn_down` | 551.6 | 2.5 | 10240 | 320 | 192 | 1288.5 | 2335.9 | 15.8 |
| `linear:hc_ffn_down` | 547.0 | 2.5 | 10240 | 320 | 192 | 1288.5 | 2355.5 | 15.9 |
| `linear:shared_down` | 301.0 | 1.4 | 640 | 2560 | 192 | 644.2 | 2140.5 | 14.5 |
| `linear:shared_gate` | 271.5 | 1.2 | 2560 | 640 | 192 | 644.2 | 2373.2 | 16.0 |
| `linear:index_q` | 226.5 | 1.0 | — | — | — | — | — | — |
| `linear:shared_up` | 185.9 | 0.8 | 2560 | 640 | 192 | 644.2 | 3465.5 | 23.4 |
| remaining 16 roles | 1444.6 | 6.4 | | | | | | |
| **Total** | **22368.1** | **100.0** | | | | | | |

`—` marks a role that issues no matmul through the census hook, so its shape is
not recoverable from these two files. The MoE, QSA, GR, GDN and indexer paths
launch through their own helpers and need the same census applied there before
their shapes can be reported. This is a real gap in the table, not a zero.

## What the arithmetic says

Every Q8_0 dense projection lands between **2140 and 3466 GFLOP/s**, or
**14.5–23.4% of the 14.8 TFLOP/s FP32 peak** of this part. That range is narrow
across wildly different shapes — a 10240×320 skinny projection and a 2560×12288
wide one both sit near 16%.

The weight-traffic column runs at 1.1–1.8 TB/s, above this part's DRAM
bandwidth. That is expected rather than alarming: a layer's weights are re-read
once per prefill chunk, and at 27.9 MB for `attn_qkv` they stay cache-resident
across the four chunks. It is a traffic figure, not a bandwidth measurement, and
no bandwidth-bound conclusion is drawn from it.

**The GFLOP/s figures exclude dequantization work**, so they are not an
efficiency verdict on their own. A Q8_0 kernel unpacks a scale and an int8 per
weight before the multiply, and that work does not appear in the FLOP count. The
number that does bear on the mechanism is the reuse per decoded weight.

## The concrete implementation difference

hipEngine's dominant variant is
`gguf_k_prefill_out_coltile_rowbatch_kernel<scalar_t, out_t, qtype, COL_TILE,
ROW_BATCH, WAVE_SCALE>` launched with `<float, float, 8, 8, 4>` — that is
`COL_TILE=8, ROW_BATCH=4`. Each thread holds 32 accumulators. For each `k` the
kernel decodes `COL_TILE=8` weights and performs `8 × 4 = 32` FMAs, so **each
decoded weight feeds 4 multiply-accumulates**. The launch geometry follows:
`grid.x = N / 8`, `grid.y = ceil(rows / 4)`.

pwilkin's MMB dense kernels are tiled far wider. `mmb_dense_kernel<128, 128, 32,
64, WT>` decodes 128 weights per `k` and does `128 × 128` FMAs, and the big-tile
form `mmb_dense_kernel<128, 256, 64, 64, WT>` reaches `128 × 256`. That is **128
to 256 FMAs per decoded weight, 32× to 64× hipEngine's reuse**, with a
256-thread block against hipEngine's 128.

MMB also changes what is being decoded and when:

- it converts the **activation** to BF16 once per matmul
  (`mmb_bf16_activation`) rather than per row;
- it can keep the **output** in BF16 (`store_f32 = false`) so the next op reads
  a narrower tensor;
- for IQ4_NL and Q6_K it uses a preprocessed **shadow** weight buffer
  (`mmb_shadow_lookup`) instead of decoding the on-disk layout;
- it picks the tile from the shape (`big` when `M >= 6144 && K >= 2560`) and has
  a separate tall-M tile for `IQ4_NL` with `M <= 384, K >= 4096`.

`mmb_min_t()` is 512, so this path does not engage below a 512-token batch: it
is a prefill-only mechanism and says nothing about decode.

## What this does not establish

- **Not an equal-arithmetic comparison.** hipEngine's Q8_0 projections carry
  guarded block-scale selection and its MoE carries iu8 risk plus sparse exact
  repair. Whether a wider tile is reachable without moving those numerics is an
  open question, not an assumption. The tile geometry above is a mechanism, and
  the two explanations — a slow kernel and a cost of preserving arithmetic — are
  not mutually exclusive.
- **No causal factorization.** An earlier version of this work multiplied a
  conservative-arithmetic factor by a kernel-quality factor and presented the
  product as the gap. Telescoping two ratios is algebra, not decomposition. That
  claim is withdrawn in
  [the 2026-09-15 artifact](../2026-09-15-flashnext-engine-comparison/README.md).
- **No comparator column yet.** The per-role comparator side needs the same
  tensor-role mapping applied to the llama.cpp-family engines. Their `hc_*` and
  `mmb_*` kernels are role-specific, but upstream folds its hyper-connection
  work into generic matmuls, so a name-based comparison would report upstream
  GR as zero work.
- **Not a microbenchmark.** These are in-context costs inside a real prefill,
  including launch overhead and cache state. An isolated kernel microbenchmark
  on the same operands is the next step and would separate the two.

## Reproducing

```bash
# 1. role-marked capture, with the launch census in the same process
HIPENGINE_KERNEL_CENSUS=/tmp/census/roles-census.json \
  rocprofv3 --kernel-trace --marker-trace --hip-trace --output-format csv \
  -d /tmp/census/roles.raw -- .venv/bin/python scripts/qwen4exp_profile_gap.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --case-id code-p4096 --mode prefill --repetitions 1 --prefill-chunk-size 1024 \
  --profile --role-markers --launch-census /tmp/census/roles-census.json \
  --compiler-version-file /tmp/hipengine-journey-hipcc-version-20260913.txt \
  --require-cached-build --output /tmp/census/roles-profile.json

# 2. attribute kernels to roles
.venv/bin/python scripts/qwen4exp_role_analyze.py \
  --trace-dir /tmp/census/roles.raw/gfx1151 \
  --output /tmp/census/role-analysis.json --measure-prefix qwen4exp_prefill_p4096_

# 3. join to shapes and compute the table
.venv/bin/python scripts/qwen4exp_role_gap_table.py \
  --role-analysis /tmp/census/role-analysis.json \
  --census /tmp/census/roles-census.json \
  --output benchmarks/results/2026-09-16-flashnext-per-role-cost/artifact.json
```

`--profile` is required for `--role-markers`: without it the ROCTX shim is never
loaded and no marker trace is written.
