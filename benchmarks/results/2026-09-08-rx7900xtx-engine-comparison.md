# hipEngine, nasone32 and strix-llama.cpp on RX 7900 XTX

Measured September 8, 2026 on `epyc`, physical GPU1: RX 7900 XTX,
`gfx1100`, PCI `0000:10:00.0`, 23.984 GiB driver-visible VRAM.
Every engine uses the identical local **Qwen3.8-27B Q4_K_M** GGUF:
17,106,773,984 bytes, SHA-256
`7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b`.

| Engine | Measured source |
| --- | --- |
| hipEngine | `6c01f1f1c`, clean detached snapshot after the C1 MTP repair |
| nasone32 | `nasone32/llama.cpp-RDNA3-7900xtx-opt`, `7dc2f0cb28326816f67f6b979008383344e2038b` |
| strix-llama.cpp | `halo-box/strix-llama.cpp` HEAD fetched immediately before its lane, `5f851647fe5ed795dfd6c0a3fba543114879e874` |

All three use HIP on the XTX. The external builds use Release, HIP graphs,
`gfx1100`, and the same local-unroll threshold of 600 as hipEngine's prefill
profile. The installed toolchain is HIP 7.2.53211 / AMD Clang 22.
Final measurements use idle admission and a GPU1 process-group ownership
guard. Compilation is outside timing.

## Prompt Processing And Ordinary Generation

Tokens/second, medians of three measured runs. Inputs repeat token 9707;
generation has 128 timed transitions. KV storage is BF16 and decode graphs
are enabled. hipEngine uses the direct resident API; the other engines use
server-owned phase timers. These are model execution rates, not HTTP throughput.

| Engine | PP512 | PP8192 | TG128 after 512 | TG128 after 8192 |
| --- | ---: | ---: | ---: | ---: |
| hipEngine | **985.36** | 777.23 | 34.31 | 29.82 |
| nasone32 | 911.61 | **943.14** | **36.67** | **35.70** |
| strix-llama.cpp | 847.23 | 901.73 | 32.62 | 29.55 |

hipEngine leads short prefill; nasone32 leads the 8K prefill and decode
measurements. The external configuration is one slot, batch 4096,
ubatch 1024, context 8704, prompt caching off and automatic fit off.
hipEngine uses one full warmup plus a decode warmup token; external runs use
a short warmup request. Exact commands and samples are in the companion JSON.

## C1 MTP

All ten committed prompts, all four categories, and the six-train/four-heldout
split are included. Both sides consume the same raw prompt IDs. Sampling is
greedy, context capacity is 1024, and MTP is explicitly requested at budget 3.
The denominator is a separate true-AR generation path, not a verifier estimate.

### 25 Visible Outputs / 24 Timed Transitions

Three repetitions. Output agreement compares complete MTP output IDs with
the same engine's AR IDs; it is not a task-quality score.

| Engine / setting | AR tok/s | MTP tok/s | MTP / AR | Exact AR matches |
| --- | ---: | ---: | ---: | ---: |
| hipEngine | 35.88 | **63.48** | **1.769x** | 30/30 |
| nasone32 default | 35.00 | 58.56 | 1.673x | 27/30 |
| nasone32, sequential GDN | 35.59 | 58.65 | 1.648x | 30/30 |
| strix-llama.cpp | 32.21 | 47.69 | 1.481x | 30/30 |

The nasone32 sequential setting is `GGML_CUDA_GDN_CHUNKED=0`.
Its default AR output changes between repetitions on `general_ja_explain`;
the sequential control is repeat-stable and restores short-suite AR/MTP
equality. The default MTP output is repeat-stable but differs from its AR
counterpart on that prompt.

### 129 Visible Outputs / 128 Timed Transitions

One full-suite repetition:

| Engine / setting | AR tok/s | MTP tok/s | MTP / AR | Exact AR matches |
| --- | ---: | ---: | ---: | ---: |
| hipEngine | 35.51 | 30.40 | 0.856x | 10/10 |
| nasone32 default | 31.65 | 44.47 | 1.405x | 6/10 |
| strix-llama.cpp | 32.48 | **50.16** | **1.544x** | 7/10 |

hipEngine currently switches to serial verification when a target block
extends past the qualified native context of 95 positions. The fallback is
correct, but dominates longer generations. Extending qualified native
verification to larger context buckets is the next C1 performance target.
The two external long-generation MTP rows are not exact-AR equivalents.

## Context Capacity

One request, no weight offload and no KV eviction. A pass includes the entire
synthetic prompt and eight timed decode transitions, not merely allocation.
The hipEngine BF16 point additionally has one decode warmup transition.
These are observed bounds for the recorded settings, not operational reserves
or exhaustive maxima.

| Engine / KV | Largest completed prompt | Peak whole-card GiB | Higher tested OOM |
| --- | ---: | ---: | ---: |
| hipEngine / BF16 | 114,688 (112K) | 23.899 | 131,072 |
| nasone32 / BF16 | 114,688 (112K) | 23.865 | 131,072 |
| strix-llama.cpp / BF16 | 114,688 (112K) | 23.669 | 131,072 |
| hipEngine / pure INT8, FP32 scales | 131,072 (128K) | 23.898 | 139,264 |
| nasone32 / Q8_0 | **196,608 (192K)** | 23.727 | 229,376 |
| strix-llama.cpp / Q8_0 | **196,608 (192K)** | 23.531 | 229,376 |

The hipEngine INT8 row uses explicit pure INT8 storage:
`HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1`,
`HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS=none`, and FP32 scales.
Its audit confirms no persistent BF16 KV mirror. The default mixed-layout
INT8 option is a different configuration.

Full-sequence prefill hidden buffers occupy about 2.69 GB at 128K on this
hipEngine route. Reducing that lifetime/footprint is a concrete capacity
optimization target. This direct-resident study does not change public-server
pool limits. DMS is excluded because it evicts history. Synthetic capacity
execution is separate from long-context task-quality evaluation.

## C1 Repair

The comparison uncovered a real hipEngine scratch-lifetime error. Large
prefill arenas alias QKV, convolution output and BF16 recurrent output.
Native verification reads the BF16 QKV directly while producing larger FP32
convolution rows, so that aliasing overwrote unread inputs and produced NaNs.
A small separately owned convolution buffer corrects the native path while
preserving bulk-prefill memory optimization.

The repair also aligns graph extents and proposal admission with backend
qualification, preserves separate native/serial journals, handles short output
tails on device, and removes a mutable asynchronous metadata-upload source.
Validation covers 206 distinct focused CPU nodes and 100 actual-model GPU
AR/MTP comparisons, including budget switching and longer continuations;
all GPU comparisons are exact and all allocations are freed.
See the [repair worklog](../../worklog/entries/20260908T120624.010092Z-lhl-repair-c1-mtp-native-scratch-and-fallback-owners-716c2b.md).

## Optimization Review

- **nasone32 K-quant load reuse:** `efa4e8641`, `ggml/src/ggml-cuda/vecdotq.cuh`
  and `mmvq.cu` increase Q4/Q5 VDR to 4 and Q6 VDR to 2, amortizing adjacent
  data/scale loads. This is worth a shaped kernel comparison. The source uses
  Q8_1 activations, unlike hipEngine's floating pack8 routes, so it is not a
  drop-in exact replacement.
- **nasone32 chunked GDN:** `4169fbbf5`, `gated_delta_net_chunked*.cu` contain
  chunk-64 Gram/triangular work and a state scan, including gfx11 BF16 WMMA.
  The observed repeat/equivalence failures prevent blindly adopting its
  default. Any arithmetic variant needs the complete profile gate and strict
  fallback. hipEngine already has register-resident recurrence variants.
- **Adaptive-depth configuration:** nasone32's `common/common.h:329` defaults
  the adaptive minimum to 3. Its README example also sets maximum 3, so that
  example does not adapt. The explicit minimum-1 screen was slower than fixed
  B3 on both tested horizons despite higher acceptance; it is not a transfer win.
- **Already represented:** channels-contiguous convolution input and
  device-resident speculative state checkpoints are already present in
  hipEngine. There is no corresponding mandatory input transpose or full
  state CPU round trip to eliminate.
- **Check architecture guards:** nasone32's dequant-float matvec and D=256
  tile override default specifically to RDNA3.5. strix-llama.cpp also carries
  many gfx1151-specific HIP optimizations. Availability in source does not
  establish selection on gfx1100. Its latest merged stack is predominantly
  Vulkan; this comparison measures HIP.
- **Future multi-GPU reference:** nasone32's internal two-device reduction,
  optional Q8 wire format and residual fusion are relevant to planned TP work,
  not this single-card result. Peer access is capability-checked, with host
  staging fallback; it cannot create unsupported connectivity.
- **Other models:** MoE MMQ/TOP_K/expert-cache and lazy PLE/QSA changes do not
  explain this dense-model result.

No donor kernel was promoted. The immediate priorities are larger native
verifier context coverage, long-prefill memory lifetime, then measured
projection/GDN kernel comparisons with the required correctness gates.

## Evidence

The [machine-readable comparison](2026-09-08-rx7900xtx-engine-comparison.json)
contains commands, source identities, category/heldout metrics, complete output
ID rows, memory audits, samples and process-group ownership records.
All timing cells and the final hipEngine 128K row use clean measurements;
setup failures and superseded attempts are not used in these tables.
