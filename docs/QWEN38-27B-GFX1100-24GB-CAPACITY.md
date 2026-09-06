# Qwen3.8-27B gfx1100 24 GB Capacity Campaign

Status: **open, baseline measured 2026-09-06.** Qwen3.8-27B `Q4_K_M` reaches
only about 3K context at one request on a 24 GB Radeon RX 7900 XTX, against a
previously published 32K on BF16 and 112K/126K on pure INT8. INT8 KV currently
saves no memory at all. No cause is diagnosed yet; this document is the
starting point.

Authoritative evidence:

- [`XTX c1 context ceiling`](../benchmarks/results/2026-09-06-rx7900xtx-qwen38-c1-context-ceiling.json)
- [`W7900 direct c1-c8 sweep`](../benchmarks/results/2026-09-06-gfx1100-qwen38-q4km-direct-c1c8-sweep.json)
- [`INT8 KV continuous batching`](QWEN38-INT8-KV-CONTINUOUS.md) — the existing
  INT8 KV campaign; the inert-INT8 finding in section 4 bears directly on it
- [`benchmark policy`](BENCHMARK.md), [`testing`](TESTING.md),
  [`roofline`](ROOFLINE.md)

---

## 1. Objective

Make Qwen3.8-27B `Q4_K_M` usable at a useful context on a 24 GB consumer card.
Three things must be true before this campaign closes:

1. A single request reaches a context worth advertising. 3K is not.
2. `--kv-storage int8_per_token_head` actually reduces footprint, or the option
   is removed rather than left inert.
3. The published capacity numbers match a default server invocation, so a user
   who follows the README gets what it says.

Out of scope: throughput. Nothing here should trade decode or prefill rate for
capacity without a separate measured decision.

## 2. Lane identity

| Field | Value |
| --- | --- |
| Card | AMD Radeon RX 7900 XTX, `gfx1100`, GPU index 1 on host `epyc` |
| VRAM | 25,753,026,560 B (23.984 GiB) |
| Model | `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf`, 17,106,773,984 B (15.932 GiB) |
| Sampled fingerprint | `2512f262273074db82860f1f3d6c15b4d9054b29b3c4babb0e2c770d6474c850` |
| ROCm | HIP 7.2.53211-3d9ef42 |

The W7900 in the same host is GPU 0 and has 44.98 GiB. It is a different lane:
its numbers do not establish a 24 GB limit and must not be relabelled as one.
The benchmark harnesses override `HIP_VISIBLE_DEVICES`, so device selection has
to be passed as an explicit flag and then verified in the artifact.

## 3. Measured baseline

Default single-request server, one completion per point, VRAM sampled from
`rocm-smi` on GPU 1.

| KV storage | Context | Result | Peak |
| --- | ---: | --- | ---: |
| `bf16` | 2,048 | OK | 21.869 GiB |
| `bf16` | 3,072 | **OK** | 23.328 GiB |
| `bf16` | 4,096 | **HIP OOM during warmup** | — |
| `bf16` | 8,192 | HIP OOM during warmup | — |
| `int8_per_token_head` | 2,048 | OK | **21.869 GiB** |
| `int8_per_token_head` | 3,072 | OK | **23.328 GiB** |
| `int8_per_token_head` | 4,096 | HIP OOM during warmup | — |

Every failure is `HipError: HIP error 2: out of memory` raised during eager
server warmup, before any request is issued. The failures are therefore
independent of the prompt and reproduce from a cold start.

### Derived footprint model

Fitting the two BF16 successes:

| Term | Value |
| --- | ---: |
| Fixed overhead | 18.951 GiB |
| — of which weights | 15.932 GiB |
| — of which runtime state | 3.019 GiB |
| Per-token KV | 1.459 MiB |
| Predicted ceiling on 23.984 GiB | ~3,532 tokens |

The prediction matches the observed 3,072 OK / 4,096 fail boundary, so the
model is good enough to reason with.

### Concurrency, for reference

From the W7900 direct sweep at 512-token prompts, HIP used peak rises from
19.609 GiB at one request by ~0.9 GiB per added request. That is consistent
with the fixed-overhead model above (19.7 GiB predicted at c1/512), so the
same budget allows four to five concurrent short requests on a 24 GB card.
Concurrency and context draw on the same ~5 GiB.

## 4. What the baseline means

**Two independent problems.**

1. **INT8 KV is inert.** `int8_per_token_head` and `bf16` produce byte-identical
   peaks at 2,048 and at 3,072 and fail at the same 4,096. Whatever dominates
   the footprint is not KV element width. Until this is fixed, no INT8 capacity
   claim can be made for this model on this backend.
2. **1.459 MiB per token is high** for a 27B model. This is what limits context
   once weights and fixed state are paid for. Attribution is the first
   engineering task.

**One measurement discrepancy that is not yet explained.** The withdrawn
scoreboard row read "32K, 21.869 / 2.115 GiB". That memory figure reproduces
*exactly* here — at 2,048 context, not 32K. The footprint number survived while
the context label did not, which points at the original row's configuration or
labelling rather than a 16x growth in allocation. Reconstructing that
configuration is a prerequisite for calling this a regression.

## 5. Candidate directions

Unranked; each needs its own measured decision.

1. **Attribute the per-token cost.** Break 1.459 MiB/token down by layer class
   (full attention versus GDN), by K/V, and by any padding or pooling. A 27B
   model with grouped-query attention should be well under this.
2. **Diagnose the INT8 path.** Determine whether the request never reaches the
   INT8 kernels or whether allocation is sized before the KV dtype is applied.
   The second is the more likely shape of the bug given identical peaks.
3. **Check whether KV is sized to `--max-context-tokens` rather than to live
   tokens.** A pool preallocated to the declared context would explain both the
   warmup-time OOM and the steep slope.
4. **Reconstruct the original 32K configuration.** Chunked prefill, prefix
   cache, and a growing KV pool were not exercised by the baseline runs and any
   of them may be what the published row depended on.
5. **Reduce fixed runtime state.** 3.019 GiB above weights is worth an
   inventory; some of it may be prefill scratch sized for a shape a 24 GB card
   will never run.

## 6. Evidence rules

- Measure on the physical XTX and verify the device in the artifact. The
  harnesses override the environment; a run that silently lands on the W7900
  invalidates the result.
- A capacity claim is the largest context that **starts and completes one
  request from a cold server**, not the largest that allocates.
- Report peak from `rocm-smi` on the target card. Tracked allocator high-water
  understates what the card must supply by 0.2-0.9 GiB and must not be used for
  a fits-or-not claim.
- State the KV storage mode explicitly in every row. BF16 and INT8 are
  currently indistinguishable by footprint, so an unlabelled number is
  ambiguous.
- Any published capacity number must come from a default server invocation, or
  must name the extra flags it requires.

## 7. Canonical commands

Single-request ceiling point:

```bash
env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 HIPENGINE_HIP_ARCH=gfx1100 \
    GPU_MAX_HW_QUEUES=1 \
    HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-gfx1100-hipcc-version.txt \
  .venv/bin/python3 -m hipengine.server \
    --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
    --backend hip_gfx1100 --quant gguf_q4_k_m --served-model-name q38 \
    --kv-storage <bf16|int8_per_token_head> \
    --max-context-tokens <N> --max-active-requests 1 --port 8077
```

Then issue one completion and sample the card:

```bash
rocm-smi --showmeminfo vram | awk '/GPU\[1\].*Used/{print $NF}'
```

Concurrency reference on the W7900 lane:

```bash
.venv/bin/python3 scripts/gguf_packed_ar_bench.py \
  --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf --backend hip_gfx1100 \
  --configurations c1,c2,c3,c4,c5,c6,c7,native_c8 \
  --prompt-length 512 --decode-steps 128 --warmup-runs 1 --measured-runs 3 \
  --compiler-version-file /tmp/hipengine-gfx1100-hipcc-version.txt \
  --require-cached-build --json <artifact>
```

## 8. Stop rules

- Stop and record if a change buys context by trading exactness. Capacity is
  not worth a correctness contract.
- Stop if a candidate improves the XTX lane but regresses the W7900 direct
  c1-c8 sweep; both lanes must hold.
- Do not publish a capacity number that a default server cannot reproduce.
