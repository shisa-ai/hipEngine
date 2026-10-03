# Qwen3.6-35B-A3B INT8 decode KV on gfx1151 — blocked on quality

**Date:** 2026-10-03 UTC
**Host:** Framework Desktop, AMD Radeon 8060S (`gfx1151`, 40 CU), 125 GiB unified LPDDR5X, HIP 7.15.26333
**Model:** `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (sha256 `0b21525e…dac58b`, 22,663,387,424 B, `gguf_ud_q4_k_m`)
**Backend:** `hip_gfx1151`, commit `4fb6a04d2`

## Question

A competing engine retains long context with an INT8 KV copy written at decode and read by decode attention,
keeping F16 for prefill, using per-32-value F16 scales. hipEngine has `--kv-storage int8_per_token_head` for
this model, but `auto` resolves to BF16 because no capability evidence is registered for the artifact.

The lane's premise was that KV width drives the decode gap that widens with depth. This run tests that premise
and screens the representation under the production numerical gate.

## Protocol

`scripts/qwen35_native_mixed_kv_suite.py`, teacher forcing: the candidate consumes the BF16 reference token IDs
and every logit comparison has matched token history.

- **Suite:** `benchmarks/prompts/mtpbench-code-general-ja.jsonl`, sha256 `fac920be…86084a` — 10 natural rows
  (code 4, general_en 2, general_ja 2, mixed_ja_en 2) plus the harness control `mixed_v1` (a repeated 36-token
  unit), 11 cases per invocation.
- **Gate:** mean KL ≤ 0.05 per prompt **and** top-1 agreement ≥ 0.90 per prompt **and** finite logits, with
  **every** prompt required to pass.
- **Shapes:** 512/8 → 1024/8 → 4096/16. A representation stops at its first failed shape.
- **Flags:** `--diagnostic-kv-capability --require-no-bf16-mirror --max-sequence-length 32786
  --compiler-version-file … --require-cached-build`.

## Result — stop-rule trace

| Candidate | 512/8 | 1024/8 | 4096/16 | First failure |
| --- | --- | --- | --- | --- |
| `int8_per_token_head`, fp16 scales | pass | pass | **FAIL** | 4096/16 |
| `int8_per_token_head`, fp32 scales | pass | **FAIL** | not run | 1024/8 |
| `tail4_hadamard_group32`, fp16 scales | **FAIL** | not run | not run | 512/8 |

The fp16-scale layout is the gufo-matching representation, so it is the primary candidate. It carries zero
persistent BF16 mirror bytes.

### Binding failure

`mixed_v1` at 4096/16: **mean KL 0.1597** (> 0.05), max KL 2.6939, top-1 agreement 0.9412 (1 flip in 17
positions).

This is a single-position event, not diffuse drift. The 17 positions sum to KL 2.7142 against a maximum of
2.6939, so **99.25% of the per-prompt KL mass sits at one position**, where the reference top-1 is `271` and the
candidate top-1 is `30523`; the other 16 positions average 1.27e-3. The control is a repeated 36-token unit, so
its reference distributions are near-one-hot and a single rank flip costs KL ≈ 2.7.

The same control is what closed the gfx1151 27B INT8 lane, where it was joined by a failing natural prompt
(`mixed_ja_en_review`, minimum top-1 0.7778 at 1024/8). Here the natural prompts are clean:

| Shape | Natural prompts | Worst natural prompt mean KL | Min natural top-1 |
| --- | --- | --- | --- |
| 512/8 | 10/10 pass | 1.223e-3 | 1.0 |
| 1024/8 | 10/10 pass | ~1.3e-3 | 1.0 |
| 4096/16 | 10/10 pass | 1.09e-3 (code 3.1e-4, general_en 2.4e-4, general_ja 7.1e-4, mixed_ja_en 1.0e-3) | 1.0 |

So the 35B route is better behaved on natural prompts than the 27B route was, and its blocker is confined to
the synthetic control. It is still a blocker: the declared gate requires every prompt in the suite to pass, and
waiving the control for a second artifact would apply a weaker gate here than the one that closed the 27B lane.

### Determinism

Repeating the 1024/8 and 4096/16 runs gives bit-identical candidate **and** reference logits for all 11 prompts,
identical per-prompt metrics, and an exactly reproducing `mixed_v1` failure.

## Runtime — the KV-width premise is not supported

Decode throughput under the comparator's cached-prefix protocol (2048-token timed turn, 128 decode tokens,
untimed prefix at depth `d`, one resident session, 2 measured repetitions):

| KV route | d=0 | d=4096 | d=16384 | Drop 0→16384 |
| --- | ---: | ---: | ---: | ---: |
| BF16, eager | 56.87 | 56.86 | 53.58 | −5.78% |
| BF16, graph | 56.81 | 56.80 | 56.73 | −0.14% |
| INT8, eager | 57.67 | 57.63 | 56.89 | −1.35% |
| INT8, graph | 57.63 | 57.65 | 57.63 | 0.00% |

INT8 KV is worth **+1.44% / +1.50% / +1.59%** against the BF16 graph route at depths 0 / 4096 / 16384. The gain
is real but flat: it does not widen with depth. **An INT8 KV copy is not the depth-sensitive lever the lane
assumed it was.**

The comparator's published headline is a single value (`gufo Qwen3.6-35B-A3B-Q6dense`, 68.43 tok/s decode,
3095 tok/s prefill); its per-depth values were not reproduced, so a ratio computed as 68.43 divided by a
depth-varying hipEngine number is a statement about hipEngine's own depth curve:

| Route | d=0 | d=4096 | d=16384 | Trend |
| --- | ---: | ---: | ---: | --- |
| BF16, eager | 1.203x | 1.204x | 1.277x | widens |
| BF16, graph | 1.205x | 1.207x | 1.206x | flat |
| INT8, graph | 1.187x | 1.187x | 1.187x | flat |

Only the **eager** route widens, and graph admission removes the widening entirely — the depth loss is a property
of the eager decode route, not of KV width or context depth. Settling the gap itself needs a same-protocol
measurement of the comparator build rather than a ratio against a headline number.

## Memory — bandwidth, not capacity

At 18,688 retained rows:

| Quantity | BF16 | INT8 | Delta |
| --- | ---: | ---: | ---: |
| Retained KV bytes | 382,730,240 | 192,860,160 | −49.61% |
| Decode scratch bytes | 462,951,952 | 273,081,872 | −41.0% |
| Owned session bytes | 22,827,360,664 | 22,680,598,936 | −146,761,728 (−0.64%) |
| Graph run peak | 23,367,704,984 | 23,259,216,280 | −108,488,704 (−0.101 GiB) |

The KV payload halves, but KV is a small share of the owned session at this depth, so the net session reduction
is 0.64%. The no-mirror prefill hidden buffer grows 42,991,616 B on the INT8 route. On a host with 125 GiB of
unified memory this is a bandwidth play, and the bandwidth it buys is 1.5%.

## Decision

**Blocked on quality.** No registry entry was added, `auto` still resolves to BF16, and no production default
changed. `hipengine/models/qwen35.py` continues to record no INT8-KV capability for this artifact.

The decision does not rest on the quality gate alone: even a clean pass would buy +1.5% decode and 0.64% session
memory at 18,688 rows, which does not justify re-cutting a gate that closed the 27B lane.

## Evidence

| File | Contents |
| --- | --- |
| [`screening.json`](screening.json) | Per-candidate, per-shape, per-prompt KL / max-KL / top-1 / logit hashes for all 11 prompts, plus the aggregate and per-category view and the stop-rule trace. |
| [`runtime.json`](runtime.json) | The four-way decode comparison and depth deltas. |
| [`memory.json`](memory.json) | Full byte accounting at 18,688 rows. |
| [`determinism.json`](determinism.json) | Repeat-run logit-hash and metric pairing. |
| [`rt_protocol.py`](rt_protocol.py) | The cached-prefix runtime protocol as run. |
| [`assemble.py`](assemble.py) | Builds the four artifacts above from the raw runs. |

Exact commands (one per shape and candidate; `/tmp/reprof/compiler.txt` is a precomputed compiler-version file
so no profiled process spawns `hipcc`):

```bash
python3 scripts/qwen35_native_mixed_kv_suite.py --engine gguf --backend hip_gfx1151 \
  --gguf-model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --paro-model /home/lhl/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1 \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --prompt-length 4096 --decode-steps 16 \
  --candidate-kv-storage int8_per_token_head --kv-scale-dtype fp16 \
  --kl-threshold 0.05 --top1-threshold 0.9 --max-sequence-length 32786 \
  --require-no-bf16-mirror --compiler-version-file /tmp/reprof/compiler.txt --require-cached-build \
  --json /tmp/reprof/int8kv/A-int8-fp16-4096x16.json
```

Runtime rows come from `/tmp/reprof/int8kv/rt_protocol.py` over `rt-{bf16,bf16graph,int8,int8graph}.json` with
`HIPENGINE_GGUF_INT8_KV_DECODE_GRAPH=1` for the graph INT8 arm.
