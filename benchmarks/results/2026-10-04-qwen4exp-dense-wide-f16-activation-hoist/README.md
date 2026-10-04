# The dense-wide f16 activation hoist is live and worth +4.09% prefill end to end

The dense-wide Q8 prefill route has two activation ABIs: the `f32_f32_out`
variants convert activations to f16 **inside** the kernel once per K-stage, and
the `f16in` variants consume activations **already converted** once. The two are
**bit-identical by construction** - the same f32-to-f16 rounding, done once
instead of repeatedly - so no numerical gate can tell them apart. Only a symbol
census or a speed measurement can.

## The tile-family sweep, which had never been done for this family

`scripts/qwen4exp_dense_wide_tile_sweep.py`, packet shape `rows 1024, K 2560,
N 10240` (53.69 GFLOP, 27.9 MB of Q8_0 weights), 3 warmup + 7 repeats:

| variant | BM | BN | best ms | TFLOP/s | vs default | max abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`dense_wide256_f16in`** | 128 | 256 | **1.3207** | **40.65** | **-48.25%** | **0.000e+00** |
| `dense_wide128x128_f16in` | 128 | 128 | 1.4311 | 37.52 | -43.92% | 0.000e+00 |
| `dense_wide256` (f32 act) | 128 | 256 | 2.5518 | 21.04 | baseline | - |
| `dense_wide128x128` | 128 | 128 | 2.6543 | 20.23 | +4.01% | 0 |
| `dense_wide64x256` | 64 | 256 | 2.9910 | 17.95 | +17.21% | 0 |
| `dense_wide64x128` | 64 | 128 | 3.3044 | 16.25 | +29.49% | 0 |
| `dense_wide256_split3` | 128 | 256 | 4.5115 | 11.90 | +76.80% | timing only |

**The tile ordering is a small result in itself:** 128x256 beats 128x128 beats
64x256 beats 64x128, so the widest `BN` wins and the narrower layouts lose, exactly
as the kernel's own comment predicted ("the narrower layouts give up some weight
reuse for headroom; which one wins is a measurement, not an argument").

**The sweep was genuinely missing.** The ledger's tile row (`128x64`, `64x64`,
`64x128`, "128x128 optimal in family") is for the **MMQ** family, and every other
reference to these `dense_wide` tile variants in the tree is a correctness test.

## Production already runs the hoisted variant

`scripts/qwen4exp_q8_dense_wide_route_probe.py --case code-p4096` reports the
launched symbols for a real 4096-token prefill (15.35 s):

```
hipengine_gguf_q8_0_dense_wide256_f16in_f32_f32_out                             1156
hipengine_gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out  436
hipengine_gguf_q8_0_pack8_gemv_f32_f32_out                                         3
```

## End-to-end value: +4.09%, at ~40x signal-to-drift

Interleaved A/B/A/B/A at `code-p4096`, chunk 1024, one process per arm. A is the
default (hoist on), B sets `HIPENGINE_GGUF_Q8_DENSE_WIDE_F16_ACT=0` (f32
activations, hoist off):

| arm | prefill tok/s | mean | within-arm spread |
| --- | --- | ---: | ---: |
| A (hoist on) | 280.83 / 280.71 / 280.77 | **280.77** | 0.04% |
| B (hoist off) | 269.12 / 269.47 | 269.29 | 0.13% |

**Turning the hoist off costs 4.09% prefill.** The arms are fully separated - A's
floor (280.71) sits 11.2 tok/s above B's ceiling (269.47) - so the signal-to-drift
ratio is roughly **40x**, an order of magnitude cleaner than the 1.7x that the
13-47 scope promotion had to argue from.

The GEMM-level 48% becomes 4.09% end to end because the dense-wide route is only
part of prefill and the conversion is paid separately.

## A correction this artifact carries

An earlier conclusion in this campaign called
`HIPENGINE_GGUF_Q8_DENSE_WIDE_F16_ACT` "a no-op" because setting it to 0
reproduced the default's KL figures exactly. **The observation was right and the
inference was wrong.** The two paths are bit-identical by construction, so a
numerical gate *must* reproduce the default's numbers whether or not the path
changed. The knob moves speed by about 4% end to end (48% on the GEMM) and
numerics by nothing.

**Transferable rule:** when two paths are bit-identical by construction, a
numerical gate is not a test of whether the path changed. Which one is engaged has
to be answered by a symbol census, and its value by a speed A/B.

## The ledger's dense figure is stale

§4A of `QWEN4EXP-OPTIMIZATION-LEDGER.md` said `dense_wide256` "runs the packet
shape in 2.573 ms, **20.9 TFLOP/s**". The sweep reproduces that number exactly for
the **f32-activation** variant (2.5518 ms), so the figure has the right provenance
and describes the variant production does not run.

**The production figure is 40.65 TFLOP/s**, against gufo's reported **30-34** for
its F16 dense kernels and hipBLASLt's **19-26**. On this operation the promoted
route is ahead of both, so the dense block's remaining opportunity is not the wide
route's throughput - it is the layers still on the exact coltile chain, which is
the 8-47 scope lever and is closed by accumulation-order drift.

## Limits

One host (`gfx1151` Strix Halo, Radeon 8060S - not the gfx1100 default hardware),
one model, one quant, one profile. The sweep is one GEMM shape and ranks tilings
only; it promotes nothing. The end-to-end A/B is one case (`code-p4096`) at one
chunk size, with 3 A samples and 2 B samples. No new engine change is claimed
here: the hoist was already live, so this artifact prices an existing default
rather than a promotion.
