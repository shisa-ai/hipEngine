# Prefill chunk capacity guard: production takes the largest chunk its admission can afford

`code-p4096`, production profile, gfx1151. The chunk ladder above 1024 had never
been measured (iteration 16 compared 1024 against 512 only).

## Speed

6 samples per arm (2 repetitions x 3), **interleaved A/B/A/B** so session drift
cannot masquerade as an arm difference:

| chunk | prefill tok/s | spread | decode tok/s | spread |
| ---: | ---: | ---: | ---: | ---: |
| 1024 (was production) | 279.28 | 1.03% | 18.127 | 1.30% |
| 2048 | 292.59 (+4.77%) | 0.72% | 18.152 (+0.14%) | 0.71% |
| **4096 (guard's top rung)** | **302.81 (+8.43%)** | 1.13% | **18.431 (+1.68%)** | 1.24% |

In the **strict** profile chunk 4096 is *slower* (70.56 vs 72.35 tok/s, -2.48%),
so the gain belongs to the production tile path, not to chunking as such.

## Numerics

Token ids are **bit-identical** across all four configurations
(`c18d529b86947504e12e0d06aaa26f2ce7942387b98b95ec3bacdd65e6278eb5`):

| profile | chunk 1024 | chunk 4096 |
| --- | --- | --- |
| strict | identical | identical |
| production | identical | identical |

Layer2 gate at chunk 4096 (`gate-chunk4096.json`) reproduces the chunk-1024 gate
numbers digit for digit: mean KL 3.039e-4 (gate 1e-3), p95 1.785e-3, p99
4.271e-3 (gate 2e-2), max 7.771e-3 (gate 5e-2), top-1 0.99556 (448/450), zero
top-1 mismatches outside the flip-eligible set, `hard_gates_passed: true`,
`scope_failures: []`.

## Memory

| chunk | staging (host) | scratch (reserved) | required | context tokens |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 20.97 MB | 8.59 GB | 112.62 GB | 262144 |
| 2048 | 41.94 MB | 8.59 GB | 112.65 GB | 262144 |
| 4096 | 83.89 MB | 15.39 GB | 119.49 GB | 262144 |

The scratch is `max(4 GiB per-runner floor, mandatory + wide-f16 staging)`, so
2048 is free (under the floor) and 4096 costs +6.80 GB of reservation. Against
128.84 GB available that leaves 9.35 GB headroom and **256K context is still
admitted**.

## Files

- `c{1024,2048,4096}-r{1,2}.json` - production arms.
- `s{1024,4096}-r{1,2}.json` - strict arms.
- `gate-chunk4096.json` - layer2 numerical gate at chunk 4096.
- `sweep.json` - reduced summary.

Command:
`uv run python scripts/qwen4exp_canonical_ar_bench.py hipengine --model-root
/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL --case-id
code-p4096 --prefill-chunk-size <chunk> --execution-profile production`

## What is promoted

The production default is a **capacity guard**, not a fixed chunk:
`_select_prefill_chunk` in `hipengine/generation/qwen4_exp_gguf.py` takes the
largest rung of `(4096, 2048, 1024)` whose admission still yields the requested
context, falling back a step at a time. It runs on the residency plan before any
weight is materialized, so a fallback costs no device memory and no time.

The workspace is context-aware rather than prompt-aware — rows are
`min(chunk, max_sequence_length)` and the runner's `max_sequence_length` comes
from the model's native context — so a blanket larger default would be paid at
session construction for every session regardless of traffic. That is why the
guard is the right form.

Verified end to end through `resolve_runtime_profile(...).construct_generator(...)`
with the real model: `admitted=4096`, `runner_chunk=4096`, `context=262144`,
`profile=production`.
