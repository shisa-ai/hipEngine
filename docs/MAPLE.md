# MAPLE — Ternary MoE inference on hipEngine

Last updated: **2026-08-07** (branch `maple`)

## Summary

Maple-Preview is DeepGrove's approximately **20B-total / 1B-active**
Mixture-of-Experts language model. It uses 24 all-MoE transformer layers,
selects 8 of 256 small experts per token, alternates three sliding-window
attention layers with one global layer, and was trained for ternary projection
weights. From the pinned geometry, the model contains about **20.21B logical
weight coefficients**; about **1.18B coefficients participate in one token
forward** when the selected experts and exact full-vocabulary head are counted.

Two related checkpoints matter:

| Checkpoint | Role | Size / format | hipEngine use |
| --- | --- | --- | --- |
| [`deepgrove/maple-preview`](https://huggingface.co/deepgrove/maple-preview) | BF16 source model and independent Transformers oracle | ~40.4 GB, custom `MapleForCausalLM` | Correctness oracle only; not the deployed runtime payload |
| [`deepgrove/maple-preview-2bit-mlx`](https://huggingface.co/deepgrove/maple-preview-2bit-mlx) | Official quantized deployment checkpoint | **5,308,186,624 bytes** of exact weights (5.31 GB / 4.944 GiB), 463 required tensors | The checkpoint hipEngine loads directly |

The deployment checkpoint stores the model's trained ternary projections in a
2-bit packed representation and keeps embeddings plus the exact LM head in
4-bit affine form. hipEngine consumes those layouts directly—there is no Torch
hot path, no full model dequantization, and no host-side expert repack. The
optional approximate FlashHead sidecar is present in the repository but is **not
used** by the exact production path.

### Current status at a glance

- Public model-ID loading and greedy text generation work on
  `hip_gfx1151`/Radeon 8060S and the peer `hip_gfx1100` registry key.
- Public prompts of **up to 512 tokens** use exact batched native prefill;
  longer prompts use the token-serial correctness fallback.
- c1 decode is resident and exact. Sampling remains greedy-only
  (`temperature=0`).
- c=2/4/8 batch decode and sparse slot reclaim are validated as a
  **fixed-capacity runtime helper**, not yet as public server throughput.
- All tracked allocations return to zero on close.

## Model architecture

| Field | Pinned value |
| --- | --- |
| Layers | 24, all MoE (`first_k_dense_replace=0`) |
| Hidden / head dimension | 2048 / 128 |
| Attention | GQA: 16 query heads / 4 KV heads; per-head QK-RMSNorm |
| Layer pattern | 18 sliding-attention + 6 full-attention layers, repeated 3:1 |
| Sliding window | 512 tokens |
| Position encoding | partial RoPE on sliding layers only; rotary dimension 64, theta 10,000; global layers are NoPE |
| Advertised model context | 128K positions; hipEngine public default is 4K and current native-prefill qualification ends at 512 |
| MoE | 256 experts, stable top-8, FP32 router logits, all-expert softmax, selected-weight renormalization |
| Expert MLP | 2048 → 512 gate/up, trained clamp-7 SwiGLU, 512 → 2048 down; no shared expert |
| Norms | RMSNorm, epsilon 1e-6, FP32 internal arithmetic and BF16 boundary |
| Vocabulary | 151,936 Qwen2 BPE tokens; untied exact LM head |
| EOS | 151645 (`<|im_end|>`) |

## Deployment checkpoint format

### Ternary projections

Self-attention Q/K/V/O and all expert gate/up/down projections use U32 words
with 16 LSB-first 2-bit codes per word. Each output row has a BF16 `row_alpha`:

```text
weight = row_alpha * (code - 1),  code in {0, 1, 2}
```

The logical values are therefore `{-alpha, 0, +alpha}`. Expert tensors remain
stacked as `[256, out, in/16]`, and selected-expert kernels read only the routed
experts.

### Embedding and LM head

The embedding and untied LM head use LSB-first affine 4-bit/group-64 storage:

```text
weight = q4_code * bf16_scale + bf16_bias
```

The exact LM head covers all 151,936 vocabulary rows. The separate
`model-flashhead.safetensors` file is optional and intentionally excluded from
the exact 463-tensor / 5,308,186,624-byte manifest.

## hipEngine execution paths

### Public c1 generation

`LLM("deepgrove/maple-preview-2bit-mlx", backend="auto", quant="auto")`
resolves to `MapleGenerator` and a resident `MapleRunner` without importing
Torch. The generator supports one prompt at a time and greedy decoding. It
resets reusable state between requests and keeps immutable packed weights on the
device until `close()`.

- **Prompt length <=512:** `MapleRunner.prefill_native()` processes rows in
  chunks of at most 256, reads the complete causal prefix across chunks, and
  returns the final row's token, position, and top logit.
- **Prompt length >512:** public generation uses `MapleRunner.prefill()`, the
  exact token-serial fallback. Native append-all ring prefill is deliberately
  not used beyond one SWA capacity because later rows could overwrite prefix
  slots still needed by earlier rows in the chunk.
- **Decode:** resident c1 kernels append to separate SWA/global
  `KVLiveSpans` owners and run exact full-vocabulary argmax.

### Batch helper

`MapleBatchRunner` runs fixed c=2/4/8 rows through the batched embedding,
attention, MoE, LM-head, and argmax chain. Each request has disjoint SWA and
global ring regions; positions wrap inside that request rather than into an
adjacent slot. `MapleContinuousBatcher` adds validated admission, sparse active
masks, and offset-correct reclaim.

This helper is useful for kernel and throughput work, but it is **not connected
to hipEngine's public generation scheduler or a production server endpoint**.
It also does not perform batched prompt prefill.

## Current Radeon 8060S / gfx1151 performance

All retained rows use the pinned 2-bit checkpoint, `GPU_MAX_HW_QUEUES=1`, a
clean tracked revision, repeated measurements, actual `rocminfo`/`rocm-smi`
capture, and exact teardown accounting. Full commands and samples are in the
linked artifacts.

### Public native prefill

The M5 protocol measures eight fixed-shape inputs per length—one natural and one
heldout prompt from each of code, general English, general Japanese, and mixed
Japanese/English—after one warmup, with three repetitions.

| Prompt tokens | Native prefill tok/s | Serial reference tok/s | Speedup |
| ---: | ---: | ---: | ---: |
| 128 | **339.890** | 148.180 | **2.294x** |
| 320 | **326.573** | 106.639 | **3.062x** |
| 512 | **317.488** | 83.257 | **3.813x** |

Native sample ranges are 337.300-341.801, 325.396-328.104, and
315.630-319.462 tok/s respectively. The earlier ~347.5 tok/s 320-token number
is superseded: it predated the cross-chunk causal-prefix correction and did not
carry the current multi-prompt gate.

Evidence:
[`2026-08-07-gfx1151-maple-m5-native-prefill-recertified.json`](../benchmarks/results/2026-08-07-gfx1151-maple-m5-native-prefill-recertified.json).

### Decode

| Path / workload | Current rate | Scope |
| --- | ---: | --- |
| c1 autoregressive decode | **163.459 tok/s** (6.118 ms/token) | clean cached-profile diagnostic, 4 warmup + 32 measured steps |
| c=2, 64 tokens/request | **218.818 aggregate tok/s** | fixed-capacity helper median, 3 repeats |
| c=4, 64 tokens/request | **261.099 aggregate tok/s** | fixed-capacity helper median, 3 repeats |
| c=8, 64 tokens/request | **299.181 aggregate tok/s** | fixed-capacity helper median, 3 repeats |

Every measured c=2/4/8 trajectory matches an independent c1 trajectory. The
18-prompt category/heldout seed gate also passes, including a sparse final c=8
group. These batch rows exclude model load and prompt prefill and must not be
reported as public server throughput.

Evidence:
[`2026-08-07-gfx1151-maple-m6-batch-decode-recertified.json`](../benchmarks/results/2026-08-07-gfx1151-maple-m6-batch-decode-recertified.json).

### Tracked memory

| Resident configuration | hipEngine-owned device memory |
| --- | ---: |
| Exact checkpoint payload alone | **4.944 GiB** |
| Public c1 runner, max context 512 (M5 protocol) | **5.133 GiB** |
| Public c1 runner, default context 4096 (diagnostic smoke) | **5.174 GiB** |
| Batch helper c=2 / c=4 / c=8, capacity 66/request | **4.951 / 4.958 / 4.973 GiB** |
| After `close()` | **0 bytes / 0 active allocations** |

These are exact process-local hipEngine allocation counters, not sampled
whole-device GTT. They exclude allocations internal to the HIP runtime and
other processes. The batch helper uses much smaller row scratch than the
256-row public prefill runner, which is why its resident number can be lower.

## Correctness evidence

The implementation uses several independent gates rather than treating coherent
text as proof of numerical correctness.

| Gate | Result |
| --- | --- |
| Packed NumPy/Torch formula vs hipEngine, 18 positions | max KL **0.013508**, mean KL 0.001679, top-1 **18/18** |
| Pinned HF `trust_remote_code` oracle with matched affine4 endpoints | max KL **0.004719**, mean KL 0.000723, top-1 **18/18** |
| M5 native vs serial, 18 natural+heldout prompts / 90 seed+continuation positions | max/mean KL **0/0**, top-1 **90/90**, token equality **90/90** |
| M5 260-token cross-chunk continuation | seed plus three subsequent decode tokens exact |
| M6 c=2/4/8 and 514-step SWA-wrap tests | all generated trajectories exact |
| Public canonical prompt | coherent 37-token answer, real EOS 151645, deterministic resident repeat |
| Lifecycle | tracked ownership returns to zero |

The untouched dense-BF16 endpoint comparison remains an intentionally failed
quantization-quality diagnostic (max KL 0.149840, top-1 16/18). Matching the
packed checkpoint's affine4 embedding and head reduces the implementation gate
to the accepted values above; thresholds were not weakened.

Primary evidence:

- [`maple-ternary2-correctness.json`](../benchmarks/results/2026-08-05-gfx1151-maple-ternary2-correctness.json)
- [`maple-public-e2e-smoke.json`](../benchmarks/results/2026-08-05-gfx1151-maple-public-e2e-smoke.json)
- [M5 recertification](../benchmarks/results/2026-08-07-gfx1151-maple-m5-native-prefill-recertified.json)
- [M6 recertification](../benchmarks/results/2026-08-07-gfx1151-maple-m6-batch-decode-recertified.json)

## Current profile and next optimization

The corrected cached-only phase profile shows that the paths are kernel-bound:

| Phase | Wall | Kernel | Host gap | Exact LM-head share |
| --- | ---: | ---: | ---: | ---: |
| native prefill320 | 982.015 ms/request | 975.347 ms | 0.68% | **49.90%** |
| c1 decode | 6.118 ms/token | 5.035 ms | 17.69% | **28.75%** |
| c8 helper decode | 27.256 ms/batch | 25.337 ms | 7.04% | **46.52%** |

The current batched affine4 LM-head kernel launches grid `(vocab, rows)` and
rereads the complete **166.922-MiB** packed weight + scale/bias payload for each
row, reaching only about 115-121 GB/s effective traffic. The next exact
performance owner is therefore a rows>1 affine4 consumer that reuses each
weight row across multiple prompt/request rows. The c1 kernel stays unchanged:
the prior c1 tile was measured at 0.96x and rejected. FlashHead and
prompt-conditioned shortcuts are outside this exact target.

Evidence:
[`2026-08-07-gfx1151-maple-corrected-phase-profile.json`](../benchmarks/results/2026-08-07-gfx1151-maple-corrected-phase-profile.json).
The detailed optimization history and punchlist remain in
[`MAPLE-PERF.md`](MAPLE-PERF.md); kernel names and trace resources are in
[`KERNELS.md`](KERNELS.md).

## Reproduction

Download the exact deployment revision into the normal Hugging Face cache:

```bash
hf download deepgrove/maple-preview-2bit-mlx \
  --revision 361db5da5e74ff6fcdd852d478e1f266ce11013a
```

Then run the qualified paths from the repository root:

```bash
# Focused runtime/correctness gates
python3 -m pytest tests/test_maple_runtime.py tests/test_maple_generation.py -q

# M5 public-native prefill recertification
GPU_MAX_HW_QUEUES=1 HIPENGINE_HIP_ARCH=gfx1151 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
python3 scripts/maple_prefill_bench.py \
  --model deepgrove/maple-preview-2bit-mlx --backend hip_gfx1151 \
  --suite benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --heldout benchmarks/prompts/gdn-prefill-category-heldouts.jsonl \
  --lengths 128,320,512 --repetitions 3 --warmups 1 \
  --continuation-steps 4 --out /tmp/maple-m5.json

# M6 fixed-capacity helper recertification
GPU_MAX_HW_QUEUES=1 HIPENGINE_HIP_ARCH=gfx1151 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
python3 scripts/maple_batch_decode_bench.py \
  --model deepgrove/maple-preview-2bit-mlx --backend hip_gfx1151 \
  --suite benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --heldout benchmarks/prompts/gdn-prefill-category-heldouts.jsonl \
  --steps 64 --repetitions 3 --warmup-steps 8 \
  --natural-gate-steps 8 --out /tmp/maple-m6.json
```

## Known limits

- Public sampling is greedy-only; temperature/top-p sampling is not implemented.
- Native prefill is qualified only through 512 tokens; longer prompts are exact
  but token-serial.
- M6 does not provide public batched prompt prefill, scheduler integration, HTTP
  serving, or a public concurrency API.
- FlashHead is approximate and excluded from the exact default.
- CUDA-peer support, speculative decoding, tensor parallelism, and 128K runtime
  qualification are separate future tracks.
