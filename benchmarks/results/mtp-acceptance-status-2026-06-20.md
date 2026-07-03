# MTP acceptance status — 2026-06-20

hipEngine MTP acceptance is recorded but still behind llama.cpp: the broad gfx1151 B1 row is exact yet slower than AR and far below llama.cpp B4 accepted/output; native GGUF B2 remains parity-blocked.

## Current broad comparison

| Engine | Budget | accepted/output | decode tok/s | speedup vs base/AR | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| hipEngine PARO MTP | B1 | 0.360 | 59.56 | 0.912× | `benchmarks/results/2026-06-15-gfx1151-mtp-compare-20260615-060801-summary.json` |
| llama.cpp HIP UD-Q4_K_M | B4 | 0.743 | 91.11 | 1.790× | `/tmp/hipengine-mtp-gfx1151-runs/20260615-060801/llamacpp-hip-ud-q4km-mtp-d32/summary.json` |
| llama.cpp Vulkan UD-Q4_K_M | B4 | 0.747 | 108.96 | 1.733× | `/tmp/hipengine-mtp-gfx1151-runs/20260615-060801/llamacpp-vulkan-ud-q4km-mtp-d32/summary.json` |

Gaps: hipEngine accepted/output is 0.485× llama.cpp HIP B4 and 0.482× llama.cpp Vulkan B4; decode tok/s is 0.654× HIP and 0.547× Vulkan.

## Native GGUF diagnostics

| Row | accept/draft | accepted/output | warm speedup | cold speedup | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| B2 3-prompt sweep | 0.100 | 0.167 | 1.019× | 0.775× | `benchmarks/results/mtp-bench-1781845000-b2-sweep-summary.json` |
| B2 count prompt | 0.200 | 0.286 | 1.240× | 0.909× | `benchmarks/results/mtp-bench-1781844300-b2-count-prompt-visible-output.json` |
| B2 minimal prompt | 0.100 | 0.167 | 1.030× | 0.785× | `benchmarks/results/mtp-bench-1781843600-b2-minimal-visible-output.json` |
| B1 visible-output | 0.150 | 0.130 | 1.021× | 0.937× | `benchmarks/results/mtp-bench-1781842600-b1-cycles20-visible-output.json` |

Greeting blocker: llama.cpp accepted 3/4 traced drafts on the greeting prompt, while native depth-1 top-k omitted the target token; comparison artifact: `benchmarks/results/mtp-bench-1781845600-b2-greeting-native-vs-llamacpp-topk-comparison.json`.

## Interpretation

- The broad hipEngine row is exact but below AR (`0.912x` prompt-mean) and behind llama.cpp B4 on both accepted/output and decode tok/s.
- The native GGUF B2 path shows isolated warm wins, but aggregate acceptance is too low and greeting remains candidate-set/parity blocked.
- Use `accepted_per_output` for engine-to-engine status; `accept_per_draft` is useful within a fixed budget but not comparable across B1/B2/B4 alone.
