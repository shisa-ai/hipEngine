# Cross-engine identical-operand replay: Q8_0 `attn_qkv`

On the fused Q8_0 attention-QKV projection of a GDN layer in Qwen3.8-Flash-Next
`UD-Q4_K_XL`, hipEngine's production kernel takes **11.70x** the pinned
comparator's time on byte-identical operands: **17.361 ms** against **1.492 ms**
for the same 53.69 GFLOP at the same float32 output boundary. The exact
(strict-arithmetic) kernel takes **13.29x**, at 19.716 ms.

Both engines were handed the same weight bytes, the same activation matrix, and
were read back from the same float32 output. hipEngine's replay reproduces the
output captured during the model run **bit-exactly** (maximum absolute
difference 0.0), so the comparison is between two runs of the same operation on
the same data and not between two reconstructions of it.

## Result

Operation: `layers.8.attn_qkv`, rows = 1024, K = 2560, M = 10240, 53.69 GFLOP.
Host: Radeon 8060S / `gfx1151`, 120 GB. Times are GPU events on the stream the
kernel actually launched on; three counterbalanced rounds, ten repetitions each.

| Engine | Kernel | Complete ms | TFLOP/s | Activation conversion ms |
| --- | --- | ---: | ---: | ---: |
| hipEngine production | `coltile8_rowbatch4_wave_scale_f32_f32_out` | 17.361 | 3.09 | fused |
| hipEngine strict | `coltile8_rowbatch4_f32_f32_out` | 19.716 | 2.72 | fused |
| comparator | `mmb_dense_kernel<128, 256, 64, 64, 1>` | **1.492** | **35.99** | 0.149 |
| comparator, conversion cache hit | same | 1.342 | 40.01 | 0 |

The comparator's conversion cost is measured, not assumed: it is the difference
between a rotating set of activation buffers that misses its conversion cache on
every call and a single buffer that hits it. The comparator dequantizes the Q8_0
weight to bf16 inside its kernel and converts the activation to bf16 once per
graph, so its complete operation is 1.492 ms and its matmul alone is 1.342 ms.

`hipEngine / comparator` is 11.70 for the production path and 13.29 for the
strict path.

### What one operation costs the prefill

The same capture observed four `attn_qkv` launches per layer, one per prefill
chunk of 1024 rows, and the model has 36 GDN layers carrying this tensor. A
4096-token prefill therefore runs this operation 144 times:

| | Per call | 144 calls |
| --- | ---: | ---: |
| hipEngine production | 17.361 ms | **2500.0 ms** |
| comparator | 1.492 ms | 214.8 ms |

One projection geometry accounts for 2500 ms of a single prefill. Against the
most recent published hipEngine 4K prefill rate of 183.6 tokens/s
(4096 tokens in 22.31 s), that is 11.2% of the prefill, and closing it entirely
would put the 4K rate near 204.6 tokens/s. That fraction is a cross-run estimate:
the per-call time and the call count come from one capture, the 22.31 s total
comes from a different commit's published row.

## Dispatch

Both engines' kernel names, launch geometry and register usage come from one
`rocprofv3 --kernel-trace` capture of the replay. `Grid_Size_X` as reported by
`rocprofv3` is `gridDim.x * blockDim.x`.

| Engine | Kernel | `gridDim` | block | LDS | VGPR | mean ms |
| --- | --- | --- | --- | ---: | ---: | ---: |
| hipEngine production | `gguf_k_prefill_out_coltile_rowbatch_kernel<float, float, 8, 8, 4, true>` | (1280, 256, 1) | 128 | 512 B | 72 | 17.395 |
| hipEngine strict | `...<float, float, 8, 8, 4, false>` | (1280, 256, 1) | 128 | 512 B | 72 | 19.677 |
| comparator | `mmb_dense_kernel<128, 256, 64, 64, 1>` | (80, 4, 1) | 256 | 55,296 B | 256 | 1.335 |
| comparator conversion | `mmb_cvt_f32_bf16` | (1280, 1, 1) | 256 | 0 B | 16 | 0.084 |

Neither adapter may fall back silently, and neither did. The hipEngine adapter
requires an exact-key registration through `is_registered`, which performs no
backend/quant/variant fallback; `resolve` alone would have accepted a broader
kernel. The comparator shim requires `ggml_cuda_mmb_supported_mm` to select MMB
for the operand shapes and returns an error rather than a timing when it does
not, so a fallback kernel cannot be reported as the comparator.

The profile that selects the hipEngine variant is part of the record. Under the
strict execution profile the dispatch chose `coltile8_rowbatch4_f32_f32_out`.
Under the production profile it chose `coltile8_rowbatch4_wave_scale_f32_f32_out`,
because this projection has M = 10240 and the production selection keyed on
`n12288` does not match it. Enabling `HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL=1` did
not change the choice for this geometry; the MMQ production variant serves the
M = 12288 `attn_q` of the full-attention layers, not this tensor.

## Numerics

The comparison is against float64 references built from the packet's own bytes,
so the rounding sources separate instead of collapsing into one difference. The
mean absolute difference from each reference, over all 10,485,760 outputs:

| Reference | hipEngine | comparator |
| --- | ---: | ---: |
| exact (`f64(x) @ f64(w)`) | **4.33e-08** | 7.87e-04 |
| weight rounded to bf16 | 5.01e-04 | 6.04e-04 |
| activation rounded to bf16 | 6.04e-04 | 5.01e-04 |
| both rounded to bf16 | 7.87e-04 | **2.17e-06** |

Each engine tracks the reference that describes its arithmetic. hipEngine's
selected kernel accumulates in float32 on float32 activations and exactly
dequantized Q8_0 weights, and its maximum absolute difference from the exact
reference is 2.06e-06. The comparator converts the activation to bf16 and
dequantizes the weight to bf16 in LDS, and its maximum absolute difference from
the both-rounded reference is 1.10e-04. Neither engine's difference from its own
reference is a defect; they are different arithmetics.

hipEngine against the comparator directly: maximum absolute difference 1.47e-02,
mean absolute difference 7.87e-04, on an output whose RMS is 0.834. The two
kernels are not interchangeable outputs, which is the point — the 11.70x gap and
the bf16 rounding are the same design decision seen twice.

## Why the gap is this large

The two kernels differ in how many rows share one read of the weight, and the
measured launch geometry states it directly.

hipEngine's kernel covers 8 output columns and 4 rows per block (`COL_TILE=8`,
`ROW_BATCH=4`) and iterates all K = 2560 inside the block. Its weight traffic
follows from that geometry: 1280 column tiles x 256 row batches x 8 columns x
2720 bytes per column = **7.13 GB**, or 256 re-reads of the 27.85 MB weight.

The comparator's kernel covers 128 output columns and 256 rows per block and
stages the dequantized weight tile in 55,296 bytes of LDS, which it reuses across
all 256 rows. Its weight traffic is 80 column tiles x 4 row batches x 128 columns
x 2720 bytes = **0.111 GB**, 64 times less.

Those traffic figures are derived from the measured `gridDim`, block size and
template arguments, not measured directly. The LDS allocation, which is the
mechanism, is measured: 512 bytes against 55,296 bytes. The derived ratio, 64,
is close to the measured 11.70x time ratio, and the direction is the one the
roofline predicts for a kernel that re-reads its weight 256 times.

### The complete byte accounting

The same geometry closes an open question in the operation's cost model. `attn_qkv`
was recorded as reading and writing about 80.3 MB per 1024-row launch, or
4.6 TB/s, which is above both the measured 211 GB/s stream rate and the 32 MB
MALL — so the accounting did not close. It does not close because 80.3 MB counts
each operand once:

| Operand | Reads per launch | Bytes |
| --- | ---: | ---: |
| weight | 256, once per row batch | 7.13 GB |
| activation | 1280, once per column tile | 13.42 GB |
| output | 1 | 41.9 MB |
| **total** | | **20.6 GB** |

20.6 GB in 17.361 ms is 1186 GB/s aggregate. Serving that needs L2 rather than
DRAM, and it can: the weight is 27.85 MB and the activation 10.49 MB, both inside
the 32 MB MALL. This is derived from launch geometry, not from hardware
counters, so a memory-counter capture is still the confirming measurement.

## Protocol

Packet capture, one real prefill, no output substitution:

```
HIPENGINE_HIP_ARCH=gfx1151 HIPENGINE_COMPILER_VERSION_FILE=<file> \
.venv/bin/python tools/replay_bridge/capture_packet.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --output /tmp/replay-bridge/packets/q8-attnqkv-L8-c0-mmq \
  --profile production --layer 8 --chunk 0
```

Comparison, both engines on that packet:

```
.venv/bin/python tools/replay_bridge/replay_ab.py \
  --packet /tmp/replay-bridge/packets/q8-attnqkv-L8-c0-mmq \
  --shim /tmp/replay-bridge/libmmb_replay.so \
  --output /tmp/replay-bridge/q8-attnqkv-L8-c0-mmq-prod-ab.json
```

The comparator shim is built from the pinned tree by
`tools/replay_bridge/build_shim.sh <comparator-build-dir> <output.so>`; it
compiles with that build's own `compile_commands.json` flags and includes that
revision's `common.cuh`, because ggml creates its streams with
`cudaStreamNonBlocking` and events recorded on the legacy default stream do not
order against its kernels.

Packet identity, all three captures byte-identical:

| Field | Value |
| --- | --- |
| prompt | `code-p4096`, category `code`, 4096 exact token IDs from the canonical fixture |
| fixture | `benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json` |
| chunk | index 0, size 1024 |
| weight | 27,852,800 bytes, sha256 `7734d93dfe282c2966fb08ebf907bd7f8e20447520e499152f11fed9f4343f0a` |
| activation | 1024 x 2560 float32, sha256 `c064619ac52949881b109f68aaf7ce4a1295c5ab42168c13aba0b191b7a2598f` |
| output | 1024 x 10240 float32, sha256 `0470c1061359edea7d8ef72f0d91929ad4f17db140fce9f32253189192bce402` |

The strict, production and MMQ-enabled captures produced identical weight,
activation and output hashes. The strict and production kernels are therefore
bit-identical on this operation, and a packet is reproducible across separate
full-model prefill runs.

## Limitations

- One operation geometry, one layer, one chunk index, one prompt, one host. The
  per-operation ratio is not a model-level rate and does not generalize to other
  shapes without measuring them.
- The comparator's 1.492 ms is a per-operation cost with a cold activation
  conversion. In a full graph the converted activation is shared by the
  projections that read the same tensor, so the amortized cost is nearer the
  1.342 ms figure. Both are reported.
- The comparator is the pinned halo-box PR #63 build (`c4aa30229`). A different
  comparator revision has a different MMB kernel.
- The 7.13 GB and 0.111 GB weight-traffic figures, and the 20.6 GB aggregate, are
  derived from launch geometry, not measured with hardware counters. The LDS and
  register figures beside them are measured.
- The 11.2%-of-prefill figure combines a same-capture per-call time and call
  count with a published prefill total from a different commit. It bounds the
  operation's share; it is not a measured end-to-end delta.
- Timing is three counterbalanced rounds of ten repetitions on one host. Per-round
  values for the production path spanned 17.355-17.390 ms and for the comparator
  1.488-1.496 ms, so the ratio is stable to about 0.2% within this session. No
  cross-session repeatability claim is made.
