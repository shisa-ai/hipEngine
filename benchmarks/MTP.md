# llama.cpp MTP External Comparison

Last updated: 2026-05-19

This page tracks the local llama.cpp MTP comparison used while evaluating
Qwen3.6 27B behavior against hipEngine PARO diagnostics. These rows are
external comparison diagnostics, not accepted hipEngine performance claims.

Durable artifact:
[`2026-05-19-llamacpp-mtp-qwen36-27b-diagnostic.json`](results/2026-05-19-llamacpp-mtp-qwen36-27b-diagnostic.json)

## How To Run

```bash
python3 scripts/llamacpp_mtp_bench.py \
  --server-bin /home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-server \
  --model /models/gguf/Qwen3.6-27B-Q4_K_M.gguf \
  --ctx-size 8192 \
  --draft-max 2 \
  --protocol both \
  --mode both \
  --output /tmp/llamacpp-mtp-qwen36-27b-diagnostic.json
```

The default prompt suite lives at
[`benchmarks/prompts/mtpbench-code-general-ja.jsonl`](prompts/mtpbench-code-general-ja.jsonl).
The default config is captured in
[`benchmarks/configs/llamacpp-mtp-qwen36-27b.json`](configs/llamacpp-mtp-qwen36-27b.json).

The runner supports two protocols:

- `natural`: 10 deterministic chat prompts covering code, general English,
  general Japanese, and mixed JA/EN.
- `token-repeat`: explicit `/completion` requests with token id `9707`
  repeated to 512 or 4096 prompt tokens, matching the repeated-token hipEngine
  diagnostic shape.

## Natural Prompt Draft Sweep

Settings: `/models/gguf/Qwen3.6-27B-Q4_K_M.gguf`, W7900/gfx1100,
llama.cpp build `232f46658` / `9214`, f16 KV cache, flash attention on,
`max_tokens=512`, `temperature=0`, `top_k=1`, `seed=12345`.

| draft max | base weighted pred t/s | MTP weighted pred t/s | speedup | MTP draft acc | source |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 25.21 | 40.17 | 1.593x | 0.745 | `/home/lhl/mtpbench/runs/20260519-023133` |
| 4 | 25.01 | 36.49 | 1.459x | 0.567 | `/home/lhl/mtpbench/runs/20260519-023738` |
| 6 | 24.92 | 31.03 | 1.245x | 0.436 | `/home/lhl/mtpbench/runs/20260519-024359` |

Category speedups:

| draft max | code | general EN | general JA | mixed JA/EN |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1.647x | 1.520x | 1.511x | 1.656x |
| 4 | 1.522x | 1.328x | 1.320x | 1.659x |
| 6 | 1.338x | 1.102x | 1.081x | 1.452x |

Conclusion: use `--spec-draft-n-max 2` as the default for this GGUF. Larger
draft windows lose enough acceptance that throughput falls.

## KV Cache Type Sweep

Settings: same natural prompt suite, `DRAFT_MAX=2`.

| cache K/V | base weighted pred t/s | MTP weighted pred t/s | speedup | MTP draft acc | source |
| --- | ---: | ---: | ---: | ---: | --- |
| f16/f16 | 25.19 | 40.04 | 1.589x | 0.745 | `/home/lhl/mtpbench/runs/cache-sweep-20260519-043042/cache-f16` |
| q8_0/q8_0 | 24.16 | 39.55 | 1.637x | 0.751 | `/home/lhl/mtpbench/runs/cache-sweep-20260519-043042/cache-q8_0` |
| q4_0/q4_0 | 23.95 | 39.54 | 1.651x | 0.754 | `/home/lhl/mtpbench/runs/cache-sweep-20260519-043042/cache-q4_0` |

Conclusion: KV cache quantization did not improve absolute single-stream
throughput in this setup. It raises relative speedup only because the base run
slows down more. Keep f16 KV for peak short-context single-request throughput.

## Qwen3.6 27B Token-Repeat Comparison

This uses two different quantizations, so it is a practical comparison rather
than a kernel-equivalent comparison:

- llama.cpp: `/models/gguf/Qwen3.6-27B-Q4_K_M.gguf`
- hipEngine: `z-lab/Qwen3.6-27B-PARO`, snapshot
  `84f86409151d4f2ec86dc0b6a096d5f6daa7f207`, `w4_paro`

Prompt for hipEngine and llama-server rows: token id `9707` repeated to the
target prompt length. Decode length: 128 tokens.

| engine / mode | shape | prefill or prompt t/s | decode pred t/s | MTP draft acc | peak GiB | notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| hipEngine PARO | 512/128 | 575.80 | 32.10 | - | 24.81 sampled / 24.02 tracked | eager decode, no graph replay |
| llama.cpp server base | 512/128 | 721.09 | 25.61 | - | - | exact token prompt |
| llama.cpp server MTP | 512/128 | 685.41 | 48.96 | 1.000 | - | exact token prompt, artificial easy case |
| llama-bench base | pp512/tg128 | 851.55 | 25.25 | - | - | `-p 512,4096 -n 128 -r 3` |
| hipEngine PARO | 4096/128 | 626.01 | 28.69 | - | 25.09 sampled / 25.70 tracked | auto chunks `1024/1024/4096/1024/1024` |
| llama.cpp server base | 4096/128 | 800.06 | 25.12 | - | - | exact token prompt |
| llama.cpp server MTP | 4096/128 | 749.16 | 47.95 | 1.000 | - | exact token prompt, artificial easy case |
| llama-bench base | pp4096/tg128 | 812.50 | 25.25 | - | - | tg128 is independent of prompt shape |

Interpretation:

- hipEngine PARO eager decode is faster than llama.cpp server base on this
  repeated-token shape: `1.25x` at 512 and `1.14x` at 4K.
- llama.cpp MTP is about `1.91x` faster than llama.cpp server base on the same
  repeated-token shape because draft acceptance is perfect.
- The repeated-token result overstates real MTP behavior. The natural prompt
  suite above is the better practical estimate.

## Takeaways

- Default MTP knob for this model: `--spec-draft-n-max 2`.
- Default KV cache for throughput: f16/f16.
- Natural prompts measured about `1.59x` weighted decode speedup at draft max 2.
- Repeated-token prompts measured about `1.91x` server decode speedup, but only
  because the prompt is an artificial perfect-acceptance case.
- These rows are external diagnostics and should not be promoted to accepted
  hipEngine performance claims without a shared correctness protocol.
