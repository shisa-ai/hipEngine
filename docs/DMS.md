# Dynamic Memory Sparsification (DMS)

Last updated: 2026-08-23

> **Current status:** explicit c1 integration and 128K/256K capacity execution
> pass, but DMS remains **rejected for production**. Exact 32K retraining raises
> disjoint-row validation accuracy to 83.60%, yet thresholded selection still
> drifts to 2.305–3.495x live compression on unseen development sequences and
> fails one Japanese step (max KL 0.14007, 87.5% category top-1). No-evict is
> essentially exact, so policy/budget selection—not compact kernel math—is the
> blocker. Dense remains default while the new exact-budget ranking mode is
> quality-qualified and moved fully onto device.

This document is the end-to-end record and continuation plan for hipEngine's
external Dynamic Memory Sparsification campaign. It covers the design, exact
model scope, data and training process, reproducible commands, retained artifact,
quality results, implementation map, known limitations, and the production
punchlist.

The trained-candidate evidence is
[`2026-08-23-qwen38-external-dms-cr2-trained-candidate.json`](../benchmarks/results/2026-08-23-qwen38-external-dms-cr2-trained-candidate.json);
the integrated long-capacity evidence is
[`2026-08-23-gfx1151-qwen38-external-dms-integrated-128k-256k-capacity.json`](../benchmarks/results/2026-08-23-gfx1151-qwen38-external-dms-integrated-128k-256k-capacity.json),
the original binding 8K/32K quality rejection is
[`2026-08-23-gfx1151-qwen38-external-dms-integrated-8k-32k-quality.json`](../benchmarks/results/2026-08-23-gfx1151-qwen38-external-dms-integrated-8k-32k-quality.json),
and the exact-label long-retraining rejection is
[`2026-08-23-gfx1151-qwen38-dms-long-trained-linear-rejected.json`](../benchmarks/results/2026-08-23-gfx1151-qwen38-dms-long-trained-linear-rejected.json).
The normative KV ABI and broader storage roadmap remain in
[`KVCACHE.md`](KVCACHE.md); lifecycle integration is tracked in
[`CONCURRENCY2.md`](CONCURRENCY2.md).

## Executive summary

| Question | Answer |
| --- | --- |
| Can we train a DMS predictor without modifying the model? | **Yes.** The base GGUF is never optimized, modified, or requantized. |
| Can we test predictor correctness and exact-Q4 model quality? | **Yes.** The integrated dense-teacher route localizes kernel versus policy error. Short and exact-32K-trained threshold policies are both rejected at 32K; no-evict remains essentially exact. |
| How large is the predictor? | **655,640 bytes** (`640.27 KiB`, `0.6253 MiB`) plus required metadata. Retraining changes values, not tensor geometry. |
| How long did training take? | The retained short candidate took **99.42 s** internally. Exact 32K fine-tuning to epoch 20 took **583.07 s cumulative trainer time**, plus **637.08 s** for train-only per-head calibration. |
| How long did the measured retained pipeline take? | **774 seconds (12m54s)** for capture, labels, final training, and both quality gates; excludes data curation, replay, code development, and cache warmup. |
| What candidate passed? | **CR2/window256:** max KL `0.009691`, 100% top-1, 1.54293x observed total live-cell compression on 768-token heldouts. |
| Did CR4 or CR8 pass? | **No.** Their max KL values were `0.08908` and `0.24993`, above the `0.05` outer floor. |
| Does this already save serving memory? | **Yes for the explicit c1 post-prefill owner:** tracked residency falls 4.592 GiB at 128K and 7.813 GiB at the 256K limit after dense release. Full production capacity qualification remains open because prefill still has a dense peak, physical compact buffers retain max-head slack, and same-host controls/sampled peaks are missing. |
| Can it become faster than dense decode? | **Plausibly at long context, but not yet measured.** Compact attention scans fewer rows; the production GPU predictor/pack/attention path must make its overhead smaller than that saving. |
| Is it quantization-independent? | The BF16 sidecar representation is not tied to GGUF storage, but the current metadata and evidence are intentionally bound to one exact Q4_K_M file. Every additional quant requires calibration and qualification. |
| Is it on `origin/main`? | Not as of this campaign snapshot. The implementation is committed on branch `fastdms`. |

## Implementation status

| Area | Status |
| --- | --- |
| Exact Q/K capture | Complete |
| Deterministic future-attention labels | Complete |
| Sidecar-only training and BF16 export | Complete |
| Strict sidecar/model/provenance binding | Complete |
| Torch-free replay and compression screening | Complete |
| Exact-Q4 logit quality evaluation | Short/dense-shadow passed; integrated 8K passed; integrated 32K rejected |
| Compact allocator/backend primitives | Implemented and fixture-tested |
| Host/device compact transaction rollback | Implemented and fixture-tested |
| GPU schema-v2 external-linear prediction | Implemented, exact-decision validated, and wired into explicit c1 prefill/decode |
| Bounded split-K compact attention | Implemented, primitive-qualified, and executed by explicit c1 at 128K/256K |
| Normal resident-session DMS selection | Integrated for explicit c1 exact-artifact use; public `LLM.generate()` selection open |
| Sole-owner no-dense-shadow GGUF decode | Integrated; dense KV is temporary during correctness-first prefill and released after pack |
| Allocator-visible production savings | Partial: c1 tracked residency drops 4.592/7.813 GiB at 128K/256K; full P7 controls open |
| Serving throughput and profiler evidence | Integrated diagnostic timings measured; no comparator or performance claim |
| Integrated c1-c32 lifecycle and long soak | Open |
| Long-context-stable sidecar | Open: bias-only CR2, conservative CR1.5, and epoch-20 exact-label linear weights all reject Japanese at 32K; exact-budget prefill ranking is next |
| Portable cross-host sidecar package | Open |
| End-to-end campaign and production guide | Complete in this document |
| Merge into `origin/main` | Open |

## Candidate size

The retained sidecar contains one independent linear eviction score per compact
full-attention layer and KV head:

```text
weight: [16 compact layers, 4 KV heads, 5120 hidden] BF16
bias:   [16 compact layers, 4 KV heads] BF16
```

That is:

- `327,680` weight parameters;
- `64` bias parameters;
- `327,744` total parameters;
- `655,488` raw BF16 parameter bytes;
- `655,640` bytes after the safetensors header.

| File | Required | Bytes | Binary size | SHA-256 |
| --- | --- | ---: | ---: | --- |
| `qwen38-27b-q4km-dms-sidecar.safetensors` | Yes | 655,640 | 640.27 KiB / 0.6253 MiB | `1960ee834c0bd4572249e71b5cba16668e139a2e93a0113af60374d1a517f9f2` |
| `dms_metadata.json` | Yes | 2,620 | 2.56 KiB | `7e1ea739617047da7d687827eca121c2c969fbaf899a76ecc871b72e39b8d225` |
| `candidate_summary.json` | Optional evidence index | 1,710 | 1.67 KiB | `d5e097f35382e4a351a369cdaa744c75d4cfd5a942167abee0113e18dfee166a` |

The minimal sidecar plus metadata distribution is **658,260 bytes** (`642.83
KiB`, `0.6278 MiB`). The model GGUF is 17,106,775,008 bytes, so the learned
safetensors payload is about **0.00383%** of the model file, or approximately
**26,092x smaller**.

The training workspace is much larger than the distributable result:

| Artifact | Disk bytes | Approximate binary size | Retained in Git? |
| --- | ---: | ---: | --- |
| FP32 capture tree | 19,131,333,070 | 17.817 GiB | No |
| Label tree | 5,049,874,413 | 4.703 GiB | No |
| Final training directory, including checkpoints | 4,627,589 | 4.41 MiB | No |
| Qualified candidate directory | 660,100 | 644.63 KiB | No; hashes and compact evidence are committed |

Large captures, label shards, model weights, checkpoints, and raw logs remain
outside Git by project policy.

## Exact model scope

The retained candidate is bound to these exact bytes:

| Field | Value |
| --- | --- |
| Model | Qwen3.8-27B `Q4_K_M` GGUF |
| Path used for qualification | `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` |
| Model SHA-256 | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |
| File size | 17,106,775,008 bytes |
| Model family metadata | `qwen35_dense_hybrid` |
| Autoregressive layers | 64 |
| Full-attention physical layers | `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63` |
| Hidden size | 5120 |
| Query heads | 24 |
| KV heads | 4 |
| Head dimension | 256 |
| Predictor input | `post_attn_rmsnorm_pre_q_projection` |

The exact-file binding is deliberate. Geometry equality is insufficient: a
finetune, revision, tokenizer change, or quantization can change hidden states,
Q/K values, attention mass, and the correct decision threshold. The current
schema-v2 loader hashes the complete model and fails closed on mismatch.

The learned weights may eventually be reusable across quantizations of the same
underlying checkpoint because the sidecar consumes normalized hidden state and
is stored independently in BF16. That is a hypothesis, not a current
compatibility claim. See [Cross-quant and family distribution](#cross-quant-and-family-distribution).

## What hipEngine DMS predicts

For compact full-attention layer `l`, KV head `h`, and normalized hidden row
`x`, the external predictor computes:

```text
score[l,h] = dot(weight[l,h], x) + bias[l,h]
evict[l,h] = score[l,h] * alpha_scale - alpha_offset > 0
```

For the retained CR2 package:

```text
alpha_scale  = 1.0
alpha_offset = 3.5251435041427612
window_size  = 256
target eligible-history CR = 2
```

The predictor does not replace or borrow a query channel. Ordinary Q remains
unchanged. A token cannot be physically discarded while it is in the protected
window; the compact backend suppresses or defers such decisions and tracks
positions and live counts through `KVLiveSpans`.

`CR2` applies to **eligible history**, not the protected window and not total
model residency. If context length is `T`, the ideal per-head CR2 target is:

```text
protected = min(T, 256)
eligible  = max(0, T - 256)
target_live = protected + ceil(eligible / 2)
```

Consequently, total compression is below 2x near the window and approaches 2x
at long context. DMS affects the 16 full-attention layers; the other hybrid
layers' recurrent state is outside this live-cell ratio.

## Design and implementation

### Runtime contract

The sidecar is external to the model and torch-free at inference time. The
runtime requires:

- explicit schema-v2 metadata;
- exact model and sidecar SHA-256 values;
- exact physical full-attention layer mapping;
- hidden-stage identity and geometry;
- canonical BF16 tensor names, shapes, and safetensors bounds;
- immutable data, trainer, and FastDMS provenance;
- a matching compact backend fingerprint;
- fail-closed handling for prefix and speculative span roles that are not yet
  qualified.

The base GGUF is not modified, trained, or included in the sidecar checkpoint.
Offline GPU labeling and training may import PyTorch. Loading, projection,
replay, policy bookkeeping, and normal hipEngine runtime modules remain
torch-free.

### Source map

| Path | Responsibility |
| --- | --- |
| `hipengine/kvcache/dms.py` | Strict metadata, compact allocator/backend, transactions, spans, codecs, and storage views |
| `hipengine/kvcache/dms_capture.py` | Checksummed bounded capture writer/loader |
| `hipengine/kvcache/dms_labels.py` | Future-attention oracle and deterministic budget labels |
| `hipengine/kvcache/dms_sidecar.py` | Torch-free sidecar loader, projection, runtime decisions, replay, and screening |
| `hipengine/kvcache/dms_device.py` | Device compact transaction bridge and byte rollback |
| `hipengine/kernels/cpu_reference/dms.py` | Registered CPU-reference predictor and compact primitives |
| `hipengine/models/qwen35_dms.py` | Qwen family capability and physical-to-compact layer mapping |
| `hipengine/runtime/qwen35_gguf_runner.py` | Exact GGUF capture tap and quality substitution hooks |
| `scripts/qwen38_dms_capture.py` | Exact-Q4 corpus capture CLI |
| `scripts/qwen38_dms_build_labels.py` | Deterministic oracle-label CLI; full-prefix query-stride subsampling for feasible 32K retraining |
| `scripts/qwen38_dms_train_sidecar.py` | Sidecar-only trainer/exporter; optional strict initialization and deterministic calibration-row validation split |
| `scripts/qwen38_dms_replay.py` | Train-only threshold calibration and capture replay |
| `scripts/qwen38_dms_quality.py` | Dense/no-evict/CR exact-Q4 KL and top-1 runner |
| `scripts/qwen38_dms_integrated_long.py` | Explicit c1 no-shadow 128K/256K capacity/decode/teardown runner |
| `scripts/qwen38_dms_integrated_quality.py` | Single-prompt dense-teacher versus compact no-evict/sidecar full-logit quality gate |
| `scripts/qwen38_dms_integrated_quality_suite.py` | Shared-load four-category long-heldout integrated quality suite |
| `scripts/qwen38_dms_build_long_manifest.py` | Builds source-disjoint 32K calibration/heldout corpora under the benchmark firewall |
| `scripts/qwen38_dms_calibrate_long_bias.py` | Captures score-only long streams and folds per-layer/head CR2 quantiles into 64 BF16 biases |

The implementation derives its DMS semantics from read-only FastDMS reference
commit `c602b0ec3266da7f74d6a658b3dafcddb443fddd`. All hipEngine development and
evidence were produced in this repository; `~/FastDMS` was not modified.

### Integrated-serving audit

The first production-route audit on gfx1151 identified the low-level seams;
the subsequent explicit c1 integration now has this status:

- `dms_streaming_pack`, `dms_append_decode`, and `dms_compact_attn_decode` pass
  focused host/device fixtures after repairing stale post-shrink raw-buffer
  sizing in the tests;
- the older GPU `dms_extract_decision/corrected_mask` primitive implements
  schema-v1 borrowed-query-channel extraction;
- schema-v2 has a registered GPU external-linear projector with resident BF16
  sidecar weights. It matches all 393,216 retained validation decisions, has a
  scratch-free profiler identity, and is called by explicit c1 GGUF prefill and
  decode;
- `DMSExternalDecisionRuntime` remains the host/reference composition used by
  replay and quality tooling;
- `DMSDevicePayloadStore` owns no host K/V mirror and accepts direct
  device-resident K/V and decisions with persistent base/capacity/live metadata;
  strict host composition remains the fallback, while explicit c1 GGUF sessions
  select the direct seam;
- explicit `Qwen35GGUFResidentSession(..., dms_metadata_path=...)` now selects
  the integrated c1 route. Paged dense KV is a temporary correctness-first
  prefill owner, then is released; decode scans only compact extents. Public
  `LLM.generate()`/c>N selection and streaming prefill without the temporary
  dense peak remain open;
- the original fixture compact-attention body still uses dynamic shared score
  storage and remains the small-live fallback. The 256-row split-K producer/
  reducer holds LDS at 2,048/1,024 bytes, passes direct 65,664/131,200-row
  numerical probes, and is selected by explicit c1 when live capacity exceeds
  the bounded small-live route.

The old quality harness remains dense-shadow evidence only. The explicit
integrated resident owner passes 128K and exact-limit 256K execution: after pack
it releases dense KV, produces finite compact decode, lowers tracked residency
by 4.592/7.813 GiB, and drains to zero. This is retained capacity evidence, not
quality or speed evidence. Its remaining questions are temporary dense-prefill
peak, severe long-context calibration drift, physical max-head slack,
same-host controls, performance, c>N, and public API promotion. See
[`2026-08-23-gfx1151-qwen38-external-dms-integrated-128k-256k-capacity.json`](../benchmarks/results/2026-08-23-gfx1151-qwen38-external-dms-integrated-128k-256k-capacity.json).

## Campaign host

| Field | Value |
| --- | --- |
| Host | `zbook` |
| Device | AMD Radeon 8060S Graphics (`gfx1151`) |
| Visible unified memory | 124.0 GiB |
| Free memory at intake | 121.603 GiB |
| PyTorch | `2.10.0+rocm7.15.0a20260727` |
| HIP reported by PyTorch | `7.15.0` |
| Python | 3.13.13 conda-forge |

This evidence is an independent gfx1151 lane. It is not an absolute-rate
comparison with a W7900 or another host.

## Data campaign

### Data firewall and split

Training used 40 deterministic 768-token sequences: 30,720 tokens total.
Benchmark prompts did not enter training.

| Category | Sequences | Tokens | Train | Validation |
| --- | ---: | ---: | ---: | ---: |
| Code | 10 | 7,680 | 8 | 2 |
| General English | 10 | 7,680 | 8 | 2 |
| General Japanese | 10 | 7,680 | 8 | 2 |
| Mixed Japanese/English | 10 | 7,680 | 8 | 2 |
| **Total** | **40** | **30,720** | **32** | **8** |

Sources and licenses:

- CPython 3.13.13 standard-library source, PSF-2.0;
- pinned `wikimedia/wikipedia` English and Japanese snapshots,
  CC-BY-SA-3.0 and GFDL.

Construction used seed `20260823`. Per-category indices `0-7` are train and
`8-9` are validation. The tokenizer identity is `qwen35:gpt2:qwen35`, SHA-256
`5682706dd39a71eeda15983347cca4fb2ed9edce3e6f2db34ceccee41b665aba`.

Retained data manifest:

```text
/home/lhl/dms-artifacts/qwen38-external-v1/data_manifest.json
SHA-256 e062a7a722bfc4deed0cbc001dc4de12a6263db6c8e6ab314e5708f39f594934
```

The source manifest SHA-256 is
`a9c89dfff3e9e3babfe6b2d764e7300c586d8a9b761a801c47707b702d4712eb`.
The local data manifests are evidence inputs, not packaged inference files.

### Stage 1: exact-Q4 capture

The capture route runs the exact Q4_K_M GGUF and records each bounded
full-attention chunk:

- BF16-bit normalized hidden input at the declared predictor stage;
- post-head-norm/RoPE Q and K, stored in FP32 for this campaign;
- positions and token IDs;
- final-row teacher top-64 logits/logsumexp;
- model, tokenizer, data, source-commit, and shard hashes.

There are 40 sequences x 16 full-attention layers = 640 shards. Capture wrote
19,130,762,240 shard bytes and took **192 seconds**.

```bash
MODEL=/models/gguf/Qwen3.8-27B-Q4_K_M.gguf
MODEL_SHA=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
ROOT=/home/lhl/dms-artifacts/qwen38-external-v1

python3 scripts/qwen38_dms_capture.py \
  --model "$MODEL" \
  --expected-artifact "$MODEL_SHA" \
  --data-manifest "$ROOT/data_manifest.json" \
  --output-dir "$ROOT/captures-fp32-768" \
  --backend hip_gfx1151 \
  --max-sequence-tokens 768 \
  --max-sequences 40 \
  --qk-storage-dtype float32 \
  --teacher-topk 64
```

Capture source commit: `86e9f2ae9f662c5a65272d9ff6279746e41cc9e9`.
Manifest SHA-256:
`eb3beadf1e5b5b068db9be7ea72d224835cae10bc5aa5be4e31b6b86520f60d6`.
The writer was clean when capture began.

### Stage 2: deterministic future-attention labels

For each full-attention layer and KV head, the oracle reconstructs dense causal
GQA attention probabilities from captured Q/K. It sums the future attention
received by each key only after that key leaves the grace window. Eligible keys
are ranked by `(future_attention_mass, position)`; lower-mass history is labeled
for eviction under an exact per-head budget.

The campaign built CR4/window256 labels. CR2/CR4/CR8 replay thresholds are then
selected from the same learned score using **train rows only**. Protected-window
labels must always be false.

The practical corpus builder used tiled ROCm/PyTorch FP32 score computation with
FP64 host accumulation. It wrote 640 shards in **30 seconds**.

```bash
python3 scripts/qwen38_dms_build_labels.py \
  --captures "$ROOT/captures-fp32-768" \
  --target-cr 4 \
  --window-size 256 \
  --output-dir "$ROOT/labels-cr4-w256-fp32" \
  --device cuda \
  --query-tile 128
```

Label source commit: `d5578ed052b4e8e686e2846991e88237ba152f9b`.
Manifest SHA-256:
`1de9f739769392a97389d9fe4624a476a1255c15ea2dc960e328a0be9f458262`.
The writer was clean when labeling began.

### Stage 3: external sidecar training

Only the external weight and bias tensors enter the optimizer. The loss is
binary cross-entropy plus a per-head predicted-versus-target eviction-budget
penalty. Epoch order, shard order, row order, initialization, and export are
seeded. AdamW checkpoints contain sidecar state only.

Retained configuration:

| Parameter | Value |
| --- | ---: |
| Epochs | 20 |
| Batch size | 512 |
| Learning rate | 0.001 |
| Budget loss weight | 0.1 |
| Weight decay | 0.0 |
| Gradient norm cap | 1.0 |
| Seed | 0 |
| Trainable parameters | 327,744 |
| Optimizer parameter count | 327,744 |
| Optimizer-state elements | 655,490 |
| Peak PyTorch device bytes | 185,618,944 (177.02 MiB) |
| Internal duration | 99.418 seconds |
| Process wall duration | 101 seconds |

```bash
python3 scripts/qwen38_dms_train_sidecar.py \
  --labels "$ROOT/labels-cr4-w256-fp32" \
  --output-dir "$ROOT/sidecar-cr4-w256-run4-bf16cal" \
  --device cuda \
  --epochs 20 \
  --batch-size 512 \
  --learning-rate 0.001 \
  --budget-weight 0.1 \
  --weight-decay 0.0 \
  --max-grad-norm 1.0 \
  --seed 0
```

Trainer source commit: `8f2a4aad5d3249d5304f087be5cdbbf61985ea34`.
Training-summary SHA-256:
`f49b26371f24335f213b80bbe594e2be4da81ea33ce1f246d8c014373232be8f`.
Same-seed exports are byte-identical.

#### Training candidate ladder

| Run | Epochs | LR | Internal time | Validation accuracy | Validation BCE | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Run 1 | 20 | 0.003 | 99.38 s | 78.42% | 1.4951 | Superseded; overfit/high BCE |
| Run 2 | 3 | 0.003 | 20.24 s | 76.51% | 1.1492 | Superseded early-stop diagnostic |
| Run 3 | 20 | 0.001 | 99.83 s | 79.479% | 0.71741 | Selected learned weights |
| Run 4 | 20 | 0.001 | 99.42 s | 79.485% | 0.71742 | Retained exact-BF16 calibration run |

Runs 3 and 4 export the same final BF16 safetensors SHA-256. Run 4 is retained
because its calibration is performed on the exact exported BF16 arithmetic.

Validation metrics for the retained run:

| Slice | Accuracy | Precision | Recall | BCE |
| --- | ---: | ---: | ---: | ---: |
| Code | 80.33% | 87.63% | 85.88% | 0.62903 |
| General English | 76.93% | 83.52% | 86.25% | 0.88089 |
| General Japanese | 79.18% | 84.97% | 87.74% | 0.71875 |
| Mixed Japanese/English | 81.50% | 83.79% | 93.38% | 0.64103 |
| **Global** | **79.48%** | **84.92%** | **88.31%** | **0.71742** |

Classification quality is a diagnostic, not the model-quality acceptance gate.
The binding gate is the exact-Q4 logit distribution and task/control suite.

### Stage 4: train-only threshold calibration and replay

The BF16 sidecar is replayed over checksummed label shards without PyTorch.
Thresholds are selected only from the training split, then frozen before heldout
and repository evaluation.

```bash
uv run python scripts/qwen38_dms_replay.py \
  --model "$MODEL" \
  --metadata "$ROOT/sidecar-cr4-w256-run4-bf16cal/dms_metadata.json" \
  --labels "$ROOT/labels-cr4-w256-fp32" \
  --expected-artifact "$MODEL_SHA" \
  --compression-ratios 2,4,8 \
  --output "$ROOT/replay-run4-cr2-cr4-cr8.json"
```

| Scenario | Train-only threshold | Validation precision | Validation recall | Replay live-cell CR | Protected-window violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| CR2 | 3.5251435 | 70.27% | 74.12% | 1.53898x | 0 |
| CR4 | -0.0638902 | 84.92% | 88.31% | 2.07733x | 0 |
| CR8 | -2.9596256 | 91.65% | 94.27% | 2.49166x | 0 |

Replay SHA-256:
`a13b196b2481cc2f45b0f3f749dd3d5d1250580097e63437f947b435774260df`.
Repeats were deterministic.

CR4 labels can support a CR2 operating point because the learned linear score
is thresholded after training. The final distributable metadata freezes the CR2
threshold; it does not expose CR4 or CR8 as qualified modes.

### Stage 5: broad long exact-Q4 quality gate

The heldout gate uses all eight 768-token validation sequences, four decode
steps, two deterministic repeats, and every category. Dense and compact logits
are compared over the exact existing Q4_K_M behavior.

```bash
uv run python scripts/qwen38_dms_quality.py \
  --model "$MODEL" \
  --metadata "$ROOT/sidecar-cr4-w256-run4-bf16cal/dms_metadata.json" \
  --replay "$ROOT/replay-run4-cr2-cr4-cr8.json" \
  --data-manifest "$ROOT/data_manifest.json" \
  --expected-artifact "$MODEL_SHA" \
  --output "$ROOT/quality-run4-heldout-d4-r2.json" \
  --backend hip_gfx1151 \
  --scenarios no_evict,cr2,cr4,cr8 \
  --decode-steps 4 \
  --repeats 2
```

Measured wall time: **300 seconds**.

| Scenario | Mean KL | p95 KL | Max KL | Top-1 | Live cells | Logical cells | Total CR | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| No-evict compact control | 0.00014791 | 0.00062252 | 0.00152388 | 100% | 395,264 | 395,264 | 1.000x | Pass |
| **CR2** | **0.00140655** | **0.00750444** | **0.00969100** | **100%** | **256,178** | **395,264** | **1.54293x** | **Pass** |
| CR4 | 0.00654767 | 0.01594721 | 0.08908000 | 100% | 189,528 | 395,264 | 2.08552x | Reject |
| CR8 | 0.01610897 | 0.04496638 | 0.24993262 | 100% | 157,904 | 395,264 | 2.50319x | Reject |

CR4 and CR8 are rejected despite 100% top-1 because maximum KL exceeds the
project's `0.05` outer safety floor. The first retained failure is
`mixed_ja_en-08`, decode step 3.

CR2 category slices:

| Category | Mean KL | Max KL | Top-1 | Rows | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Code | 0.00172023 | 0.00753754 | 100% | 16 | Pass |
| General English | 0.00076740 | 0.00314858 | 100% | 16 | Pass |
| General Japanese | 0.00141119 | 0.00731682 | 100% | 16 | Pass |
| Mixed Japanese/English | 0.00172737 | 0.00969100 | 100% | 16 | Pass |

Quality artifact SHA-256:
`a83e864f433fb8bebbcebdebe80cd9cc2b6e094a6928600803f892d538708229`.

### Stage 6: repository category and heldout gate

The evaluation firewall then runs the complete 10-prompt mtpbench category suite
plus eight category-heldouts. None entered training.

```bash
uv run python scripts/qwen38_dms_quality.py \
  --model "$MODEL" \
  --metadata "$ROOT/sidecar-cr4-w256-run4-bf16cal/dms_metadata.json" \
  --replay "$ROOT/replay-run4-cr2-cr4-cr8.json" \
  --data-manifest "$ROOT/eval-mtpbench-heldouts-manifest.json" \
  --expected-artifact "$MODEL_SHA" \
  --output "$ROOT/quality-cr2-mtpbench-heldouts-d4-r2.json" \
  --backend hip_gfx1151 \
  --scenarios no_evict,cr2 \
  --decode-steps 4 \
  --repeats 2
```

Measured wall time: **151 seconds**. Across 144 scored rows, no-evict and CR2
both recorded:

- mean KL `0.0000575037`;
- p95 KL `0.000327470`;
- max KL `0.000668555`;
- 100% top-1 agreement;
- deterministic repeats.

All repository prompts are shorter than the 256-token protected window, so CR2
correctly performs no eviction and reports 1.0x compression. This gate proves
task/control coverage and protected-window behavior; it is **not** compression
evidence. Artifact SHA-256:
`718d40208c14a9eeba878ebce123a368349d2ca041cdf7db314a8436a39bbb6b`.

## Exact 32K retraining outcome (rejected)

Four source-disjoint 32K calibration sequences produced 64 full-prefix,
query-stride-one layer shards: 131,072 sequence tokens under the exact
CR2/window256 future-attention oracle. Fine-tuning initialized from the short
sidecar and reserved `row_index % 16 == 0` in every shard for disjoint internal
validation. After 20 epochs, 520,192 heldout decisions reached 83.60% accuracy,
83.48% precision, 83.74% recall, and BCE 0.3660. The 655,640-byte epoch-20
sidecar SHA is `e52fc60a...d764`.

A train-only per-layer/head quantile pass then changed 64 biases while preserving
all 327,680 weights. Calibration-source compression was 1.878–2.077x. On the
reused v2 four-category 32K development suite, however, threshold selection
shifted to 2.305–3.495x. Code, English, and mixed Japanese/English passed, but
Japanese recorded max KL 0.14007 and 87.5% top-1 over eight steps, so the
candidate is rejected. Aggregate top-1 was 96.875%; the binding rule is per
category and maximum KL, not the aggregate average.

This isolates two distinct facts: the linear classifier does learn useful long
rankings, but one static threshold cannot enforce CR2 under unseen score
marginals, and the remaining Japanese critical-token error cannot be waived.
The fresh v3 final corpus remains unconsumed. The next candidate must use the
same learned logits as ranks while enforcing the exact historical live budget
per layer/head and preserving the protected window; it must pass v2 before v3
is opened.

## Timing summary

### Retained path

| Stage | Measured wall time |
| --- | ---: |
| Exact-Q4 FP32 capture | 192 s |
| Oracle labels | 30 s |
| Final training process | 101 s |
| Broad long quality gate | 300 s |
| Repository category+heldout gate | 151 s |
| **Measured retained total** | **774 s (12m54s)** |

The trainer's internal timer reports 99.418 seconds within its 101-second
process wall. The table excludes deterministic source-data curation, replay
threshold calibration, JIT/cache warmup done outside a stage, code development,
and documentation.

### Exploratory training cost

The four training processes took 101 + 21 + 101 + 101 = **324 seconds**. If all
four exploratory runs are added to capture, labels, and both quality gates, the
known GPU-job wall is **997 seconds (16m37s)**, still excluding the items above.
This is historical campaign cost, not the cost required to reproduce only the
retained candidate.

## Current test coverage

The final focused torch-free bundle reported 50 passes and one expected
training-only/PyTorch skip. The system-PyTorch trainer bundle reported two
passes, including byte-identical same-seed exports. Focused HIP tests exercise
real GGUF capture and byte-restoring compact device rollback.

Representative commands:

```bash
uv run pytest -q \
  tests/test_dms_sidecar_metadata.py \
  tests/test_dms_capture.py \
  tests/test_qwen38_dms_capture_script.py \
  tests/test_dms_labels.py \
  tests/test_dms_sidecar_replay.py \
  tests/test_dms_external_runtime.py \
  tests/test_qwen38_dms_quality.py \
  tests/test_kvcache_dms.py

python3 -m pytest -q tests/test_dms_sidecar_training.py -s

uv run pytest -q \
  tests/test_dms_capture_gguf_hip.py \
  tests/test_kvcache_dms_device_hip.py
```

Coverage includes malformed metadata, model/sidecar/data/tensor hash mismatch,
physical-layer mismatch, tensor bounds and dtype/shape errors, protected-window
budgets, causal future-use labels, deterministic ties, training split isolation,
same-seed export, replay determinism, runtime role rejection, compact
transactions, device payload rollback, no-evict control, and exact-Q4 quality
aggregation.

A fresh full `uv run pytest -v` was not run at this trained-candidate handoff.
That broad gate is required at production milestone closure, following the
focused-repair policy in [`TESTING.md`](TESTING.md).

## What is proven and what is not

### Proven

- A real sidecar can be trained against the exact current Q4_K_M behavior without
  altering or requantizing the base model.
- Model, sidecar, geometry, input stage, physical layers, training provenance,
  and tensor identities fail closed.
- Capture, labels, training, BF16 export, and replay are deterministic under the
  recorded protocol.
- The retained CR2 candidate passes the broad 768-token exact-Q4 outer floor in
  every category with 100% top-1.
- CR4 and CR8 do not pass and are not qualified.
- Host compact pack/decode/reclaim is transactionally exercised by the quality
  path.
- Focused device gates restore request-owned compact K/V, position, eviction,
  metadata, and counters byte-for-byte after post-mutation failure.

### Not proven

- The short-context quality runner uses
  `dense_prefill_then_host_compact_decode_override` and retains dense KV; its
  1.54293x result remains logical-only.
- The separate integrated c1 route proves post-pack tracked savings and a
  near-exact no-evict control, but the trained sidecar fails the binding 32K
  dense-teacher gate; sampled process/device peak, same-host memory controls,
  and bounded streaming-prefill peak remain open.
- No DMS throughput, latency, TTFT, ITL, or E2E performance claim exists.
- No long c1/c8 soak or integrated c1-c32 request lifecycle is qualified.
- No production serving `rocprofv3` trace proves the final kernel route.
- Prefix sharing and speculative spans intentionally fail closed.
- The sidecar is qualified only for one exact Q4_K_M artifact.
- Metadata contains local evidence paths and is not yet a polished portable
  package.
- Dense paging remains the product default.

## Capacity goals

For this model's 16 full-attention layers, dense BF16 K/V payload costs:

```text
16 layers * 2 (K,V) * 4 KV heads * 256 head_dim * 2 bytes
= 65,536 bytes/token
= 64 KiB/token
```

Ignoring bounded metadata and allocator alignment, the CR2/window256 target is:

| Context | Dense BF16 full-attention KV | Ideal CR2/window256 live tokens | Ideal compact BF16 payload | Payload saved | Ideal payload CR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 0.500 GiB | 4,224 | 0.258 GiB | 0.242 GiB | 1.939x |
| 32K | 2.000 GiB | 16,512 | 1.008 GiB | 0.992 GiB | 1.985x |
| 128K | 8.000 GiB | 65,664 | 4.008 GiB | 3.992 GiB | 1.996x |
| 256K | 16.000 GiB | 131,200 | 8.008 GiB | 7.992 GiB | 1.998x |

Those ideal CR2 rows are planning calculations. The first integrated c1
measurements differ materially:

| Integrated row | Logical live CR | Logical live BF16 K/V | Physical compact slot planes | Net tracked post-pack residency drop | Tracked peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128K + 2 decode | 6.585x | 1.215 GiB | 3.408 GiB | 4.592 GiB | 30.294 GiB |
| 262,142 + 2 decode (256K limit) | 7.220x | 2.216 GiB | 8.171 GiB | 7.813 GiB | 43.322 GiB |

Both rows release dense KV before decode, produce finite logits, and drain
tracked allocation to zero. However, this is a repeated-corpus capacity stress,
not a semantic-quality workload. The excessive live compression versus CR2 is
now confirmed as a quality blocker: at 32K, sidecar max KL is 6.0177 and top-1
is 62.5%, while compact no-evict passes at max KL 7.76e-7 and 100% top-1.
Uniform max-head allocation also leaves 2.79x/3.67x capacity-to-live slot slack.
Production acceptance requires a newly long-qualified candidate plus sampled
peaks, same-host memory controls, and full scratch/metadata reconciliation.

Primary capacity goal:

- retain the CR2 quality gate;
- approach at least 1.9x allocator-visible full-attention KV payload reduction
  at 32K and longer contexts;
- account every compact plane and all temporary storage;
- show the expected approximately 4 GiB and 8 GiB BF16 payload savings at 128K
  and 256K, respectively, within explicit metadata/alignment overhead;
- never describe logical masks over a dense arena as memory savings.

Dense INT8 remains a separate comparator. At long context, CR2 BF16 payload is
roughly comparable to dense INT8 payload before metadata; combining DMS with a
compressed codec requires its own artifact-specific quality campaign.

## Can DMS make decode faster?

**Potentially yes at long context, but it is not automatic.** Dense attention
must read and score every live K/V row. CR2 asymptotically halves those rows, so
a memory-bound compact attention kernel can reduce full-attention scan time.
The sidecar adds only 327,680 weight MACs plus 64 biases per generated token,
but a host projection, device round trip, serial compaction, poor layout, or
extra launch can easily erase the saving.

For total step time, the upper bound follows Amdahl's law:

```text
whole_step_speedup = 1 / ((1 - f_attention)
                          + f_attention / attention_speedup
                          + normalized_DMS_overhead)
```

`f_attention` is the measured dense full-attention share. Because only one in
four layers is full attention in this hybrid model, whole-model speedup will be
smaller than the compact-attention speedup. At contexts at or below window256,
there is no scan reduction and the predictor is pure overhead; production
policy should remain dense below a measured break-even unless capacity forces
otherwise.

Performance goals:

1. Move sidecar projection and decision generation to a registered gfx1151/
   gfx1100 GPU path; eliminate host decode projection and copies.
2. Fuse or co-schedule normalized-hidden projection, thresholding, append, and
   metadata updates where correctness permits.
3. Make grouped-GQA compact attention consume `KVLiveSpans` directly and scan
   only true live rows.
4. Demonstrate kernel time scaling with measured live counts under
   `rocprofv3`, with expected kernel names and no unexpected dense attention.
5. Measure c1 and serving-shaped c8/c32 at 8K/32K/128K/256K against true dense
   BF16 and applicable dense INT8 baselines on the same host.
6. Require non-regressive E2E/ITL at the selected activation threshold. If short
   contexts regress, use a measured context/admission policy rather than
   defaulting DMS globally.
7. Retain even a capacity-only mode only when its slower/faster tradeoff and SLO
   are explicit; do not claim a speedup from logical compression.

No expected speedup is promoted until the exact same-host benchmark and full
quality/task gates pass.

## Cross-quant and family distribution

The present distribution unit is an **experimental exact-artifact sidecar**:

```text
qwen38-27b-q4km-dms-sidecar.safetensors
dms_metadata.json
```

The sidecar path inside metadata is relative, but qualification evidence paths
are currently local. The loader verifies the exact model file SHA-256.

A future quant-independent package for the same base checkpoint should separate:

```text
DMS package
├── dms.safetensors                 # shared learned predictor, if qualified
├── dms_config.json                 # canonical source-checkpoint identity
└── variants/
    ├── q4_k_m.json                 # artifact hash, threshold, evidence
    ├── q4_k_s.json
    ├── q8_0.json
    └── bf16.json
```

Each variant must receive:

- an exact artifact hash and tokenizer/config identity;
- a no-evict compact control;
- train-only threshold calibration, never benchmark calibration;
- broad long category+heldout exact-quant KL/top-1 evaluation;
- deterministic repeats;
- integrated no-shadow capacity and serving gates.

If one weight set does not transfer cleanly, train across quantized teachers or
ship quant-specific weights. Different model revisions, sizes, finetunes, or
physical layer maps are not members of the same compatibility set without
separate training and qualification. DFlash and MTP heads are likewise usually
checkpoint-specific even when their representation can be reused across
quantizations.

## Production punchlist

The following is the ordered continuation campaign. A checked implementation
primitive is not equivalent to a promoted product path; each phase closes only
when its listed evidence exists.

### P0 — Branch integration and durable package contract

- [ ] Reconcile branch `fastdms` with current `origin/main` without dropping the
      16-commit campaign lineage or unrelated work.
- [ ] Run focused tests after reconciliation; run the full suite at milestone
      closure.
- [ ] Define a portable sidecar package schema with relative files and durable
      evidence URIs/hashes instead of host-only absolute paths.
- [ ] Add a package validator and deterministic packaging command.
- [ ] Preserve exact-file fail-closed behavior until an explicit compatibility
      allowlist is qualified.
- [ ] Keep model weights, captures, labels, raw logs, and JIT products out of Git.

**Exit:** the candidate can be copied to another host, validated offline, and
rejected on every model/sidecar/config mismatch without importing PyTorch.

### P1 — Public selection and strict fallback

- [ ] Add a model/plugin registration for the exact artifact and sidecar package;
      do not add backend/quant branches to dispatch or engine code.
- [ ] Add explicit API/CLI/config selection for `dense` versus external DMS.
- [ ] Keep dense as default and register a strict dense fallback.
- [ ] Reject absent, malformed, untrained, unqualified, or mismatched metadata.
- [ ] Report selected model, sidecar, profile, target CR, window, and variant
      manifest in runtime diagnostics.
- [ ] Keep prefix and speculative roles off until their ownership contracts pass.

**Exit:** a user can deliberately request the exact candidate through the normal
construction path, observe the selected manifest, and receive deterministic
fail-closed behavior.

### P2 — Sole-owner no-shadow GGUF prefill

- [x] Provision compact capacity from exact measured survivors and pack directly
      from the temporary dense prefill owner.
- [x] Remove the dense full-attention K/V arena after pack; no decode-time raw+compact mirror.
- [ ] Keep protected-window rows and required sinks exact.
- [ ] Account K, V, positions, eviction mask, live counts, extent slack,
      descriptors, transaction journal, and phase scratch.
- [ ] Prove rollback restores every byte and ownership counter after failures at
      pre-allocation, mid-pack, post-pack, and shrink boundaries.
- [ ] Add no-evict compact control through the same sole-owner route.
- [ ] Replace the temporary dense prefill owner with bounded streaming compact
      prefill so peak capacity also scales with live rows.

**Exit:** tracked and sampled memory show one compact owner, no dense shadow, and
byte-exact reclaim after success, cancellation, and injected failure.

### P3 — GPU sidecar decision and transactional append

- [x] Register gfx1151 and gfx1100 external-linear BF16 decision kernels.
- [x] Consume the declared normalized hidden stage on device.
- [x] Preserve all ordinary query channels.
- [x] Replace distribution-sensitive threshold-only prefill selection with a
      metadata-bound exact historical budget over learned per-layer/head ranks;
      preserve window protection and deterministic tie-breaking. Correctness-
      first host ranking/device-mask update is implemented; quality is open.
- [ ] Fuse or co-schedule ranking/thresholding, protected-window handling,
      append, and compact metadata update where the numerical contract permits.
- [x] Eliminate host projection and per-token K/V copies from c1 decode serving.
- [ ] Journal request-owned extents only; rollback must not disturb neighbors.
- [ ] Validate graph/eager repeats and changed-page updates.

**Exit:** decode decisions and appends stay on device, are deterministic under a
fixed schedule, and pass mutation/rollback/isolation fixtures.

### P4 — Production compact attention

- [x] Port bounded-LDS GQA compact split-K attention over persistent compact extents.
- [ ] Scan each KV stream once for the query heads that share it when profitable.
- [x] Support ragged per-layer/per-head `live_counts` and monotonic positions.
- [x] Remove any dense-context fallback from the selected c1 DMS decode route.
- [x] Add integrated no-evict control plus forced-pattern strict primitive fixtures against CPU reference.
- [x] Record primitive `rocprofv3 --kernel-trace` identities and plausible durations.
- [x] Audit primitive VGPR, scratch, LDS, launch count, and reduction ownership.

**Exit:** exact no-evict and profile-qualified CR2 paths execute the intended
compact kernel family without hidden dense scans.

### P5 — Scheduler and c1-c32 lifecycle

- [ ] Use the common scheduler and allocator protocol; do not create a DMS-only
      serving scheduler.
- [ ] Qualify c1/c2/c4/c8/c16/c32 admission and physical grouping.
- [ ] Cover ragged prompts, delayed admission, sparse retirement, and row
      permutation.
- [ ] Cover cancellation before pack, during pack, during decode, and after
      partial output.
- [ ] Cover pressure denial, fragmentation, growth, recovery, and final drain.
- [ ] Verify neighbor substitution/isolation and c1<->cN transitions.
- [ ] Reconcile resource-ledger deltas with backend-owned bytes.

**Exit:** all lifecycle matrices return to the exact pre-run allocator baseline,
with no stale spans, generations, masks, payloads, or transaction ownership.

### P6 — Integrated quality and task qualification

- [x] Re-run no-evict and CR2 through the sole-owner device route at the 768-token smoke scope.
- [ ] Use the strict dense teacher trajectory and full logits.
- [ ] Record mean/p95/p99/max KL and top-1 by category, shape, layer/head where
      useful, and transition.
- [ ] Run deterministic graph/eager and same-schedule repeats.
- [ ] Run the complete mtpbench category suite plus category-heldouts and broad
      long heldouts; no single-prompt tuning.
- [ ] Add 8K/32K/128K contexts that materially exercise eviction (8K passes; 32K rejects the current candidate; 128K correctly skipped).
- [ ] Add production task/SLO and BF16-relative gates from
      [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md).
- [ ] Reject unexplained first failures; never waive max KL because top-1 passes.

**Exit:** CR2 passes every binding production numerical, deterministic,
isolation, and task threshold through the actual serving route.

### P7 — Allocator-visible capacity qualification

- [ ] Measure dense BF16, dense INT8 where qualified, compact no-evict, and CR2
      on identical model/workload/host runs.
- [ ] Report logical context, mean/max live counts, payload bytes by plane,
      metadata, scratch, tracked peak, and sampled process/device peak.
- [ ] Validate planning calculations at 8K/32K/128K/256K.
- [ ] Require no dense mirror and bounded transaction scratch.
- [ ] Run long c1 and serving-shaped c8 soak under memory pressure.
- [ ] Test OOM/denial without corruption and exact final drain.

**Exit:** CR2 demonstrates real allocator-visible savings consistent with live
counts and the capacity target, not merely an attention mask.

### P8 — Performance and profiler qualification

- [ ] Establish true no-DMS baselines on the same physical host.
- [ ] Measure prefill, decode, TTFT, ITL, E2E, throughput, and peak memory at
      c1/c8/c32 and 8K/32K/128K/256K where feasible.
- [ ] Use exact same prompts/token IDs, quant, KV dtype, scheduler, and timing
      owner.
- [ ] Profile predictor, pack, append, compact attention, split reduce,
      scheduler, and host gaps.
- [ ] Verify kernel time scales with actual live counts.
- [ ] Determine the context/concurrency break-even threshold.
- [ ] Promote a context-aware selection policy only if the full quality suite
      remains green and the complete wall is non-regressive for its declared
      scope.
- [ ] Update benchmark rollup, changelog, artifact, and worklog for every retained
      performance result.

**Exit:** DMS has either a measured same-host speedup in its declared long-context
scope or an explicitly accepted capacity-only mode with quantified cost. There
is no performance claim until this phase closes.

### P9 — Prefix, speculative decode, and codec composition

These are post-CR2-core gates and must not block initial dense-versus-DMS serving
qualification.

- [ ] Design per-sequence eviction overlays before enabling prefix sharing; never
      share an already-evicted prefix blindly.
- [ ] Apply speculative writes through scratch/journal spans and commit accepted
      rows only.
- [ ] Qualify chain/tree verification, rejection rollback, and RNG/control
      ownership before enabling DMS+MTP/DFlash.
- [ ] Evaluate DMS plus INT8/FP8 storage only with an artifact-specific quality
      gate and no dense/BF16 shadow.
- [ ] Keep a registered strict BF16 compact fallback.

**Exit:** each composition has independent correctness, quality, capacity,
lifecycle, and profiler evidence.

### P10 — Cross-quant qualification and release

- [ ] Identify canonical source-checkpoint identity independently of serialized
      quant artifact hashes.
- [ ] Test shared weights on Q4_K_S, Q5/Q6, Q8, and BF16 candidates that truly
      derive from the same checkpoint.
- [ ] Calibrate per-quant thresholds on non-benchmark train data.
- [ ] Add only exact qualified artifacts to the compatibility allowlist.
- [ ] Publish model card, licenses/provenance, checksums, supported engine/schema
      version, limitations, and removal/rollback instructions.
- [ ] Add package download/cache verification without adding model weights as a
      hard dependency.
- [ ] Make DMS default only for scopes where all product gates pass; preserve a
      user-visible dense opt-out while rollback value remains.

**Exit:** the sidecar is portable, reproducible, narrowly compatible, and safe to
select in supported production scopes.

### Optional research after CR2 production

- [ ] Improve the predictor or training objective for CR4 without weakening the
      quality gate.
- [ ] Expand licensed training contexts beyond 768 tokens and include transition
      shapes while preserving an evaluation firewall.
- [ ] Explore nonlinear or low-rank predictors if exact-budget use of the
      measured epoch-20 linear rankings still fails CR2 quality; thresholded
      linear selection is already a measured 32K blocker.
- [ ] Explore DMS plus compressed KV after standalone CR2 BF16 serving closes.
- [ ] Do not promote CR4/CR8 from the current weights; both are rejected.

## Promotion policy

DMS remains default-off until all of the following are true:

1. exact sidecar/model/package identity passes;
2. normal serving uses compact K/V as sole owner;
3. no-evict and CR2 production quality/task gates pass;
4. c1-c32 lifecycle, cancellation, pressure, recovery, and drain pass;
5. long-context c1/c8 soak passes;
6. profiler evidence proves the intended kernels and timing owner;
7. allocator evidence proves real memory savings;
8. the declared performance or capacity SLO passes;
9. a strict dense fallback remains registered;
10. portable packaging and compatibility documentation are complete.

An exact and non-regressive performance win should become the default for its
qualified scope. A quality-passing capacity mode may remain explicit/default-off
if it is slower, but its tradeoff must be measured and documented. CR4/CR8 cannot
be promoted by relaxing the existing max-KL threshold.

## Evidence and lineage

### Retained artifacts

| Evidence | SHA-256 |
| --- | --- |
| Data manifest | `e062a7a722bfc4deed0cbc001dc4de12a6263db6c8e6ab314e5708f39f594934` |
| Capture manifest | `eb3beadf1e5b5b068db9be7ea72d224835cae10bc5aa5be4e31b6b86520f60d6` |
| Label manifest | `1de9f739769392a97389d9fe4624a476a1255c15ea2dc960e328a0be9f458262` |
| Training summary | `f49b26371f24335f213b80bbe594e2be4da81ea33ce1f246d8c014373232be8f` |
| Sidecar replay | `a13b196b2481cc2f45b0f3f749dd3d5d1250580097e63437f947b435774260df` |
| Long exact-Q4 quality | `a83e864f433fb8bebbcebdebe80cd9cc2b6e094a6928600803f892d538708229` |
| Repository evaluation | `718d40208c14a9eeba878ebce123a368349d2ca041cdf7db314a8436a39bbb6b` |
| Qualified sidecar | `1960ee834c0bd4572249e71b5cba16668e139a2e93a0113af60374d1a517f9f2` |
| Qualified metadata | `7e1ea739617047da7d687827eca121c2c969fbaf899a76ecc871b72e39b8d225` |

### Major implementation commits

| Commit | Unit |
| --- | --- |
| `e7c3e00eb` | External DMS sidecar schema v2 |
| `2bbb4616b` | Exact GGUF DMS capture pipeline |
| `a205029dd` | Deterministic DMS oracle labels |
| `e49849a96` | External sidecar trainer/exporter |
| `58b5786d7` | Replay and compression screening |
| `978d1369e` | External decision runtime integration |
| `42dbb0d40` | Byte-restoring compact device rollback |
| `d5578ed05` | Eligible-history compression budgets |
| `4ff40d1f7` | Exported threshold calibration |
| `8f2a4aad5` | Exact BF16 sidecar arithmetic calibration |
| `e520d67a9` | Provision-then-shrink compact prompts |
| `1024fbe3f` | Exact-Q4 compact quality gate |
| `c12d807f2` | Qualified CR2 candidate evidence and rollup |
| `60ea51549` | Branch handoff |

### Durable worklogs

- [`20260823T102244.480574Z-lhl-fastdms-exact-cr2-trained-candidate-b64a81.md`](../worklog/entries/20260823T102244.480574Z-lhl-fastdms-exact-cr2-trained-candidate-b64a81.md)
- [`20260823T102851.497538Z-lhl-fastdms-branch-handoff-95cb3b.md`](../worklog/entries/20260823T102851.497538Z-lhl-fastdms-branch-handoff-95cb3b.md)

Use `python3 scripts/worklog.py render` for the complete immutable campaign
sequence. Earlier superseded attempts are historical diagnostics, not qualified
sidecars.
