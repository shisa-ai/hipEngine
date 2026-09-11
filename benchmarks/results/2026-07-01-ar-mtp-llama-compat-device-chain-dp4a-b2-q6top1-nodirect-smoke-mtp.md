# hipEngine GGUF-MTP category matrix

Raw root: `/tmp/hipengine-mtp-q6top1-nodirect-smoke`

> **Diagnostic only:** the `off` row is derived from B1 target-verifier timing, not a true no-MTP autoregressive run. Do not use `vs verifier off` as a retained MTP speedup claim until a true AR baseline is measured by this harness.

## Train / heldout split
Strategy: `fixed_category_heldout_v1`
Train prompts: `code_merge_intervals`
Heldout prompts: ``
Heldout covers all present categories: `False`

| split | budget | decode tok/s | vs verifier off | draft accept | accepted/output | prompts |
|---|---|---:|---:|---:|---:|---:|
| full | b2 | 61.97 | 0.805 | 1.0000 | 0.6667 | 1 |
| train | b2 | 61.97 | 0.805 | 1.0000 | 0.6667 | 1 |
| heldout | b2 | 0.00 | — | 0.0000 | — | 0 |

## Total
| budget | decode tok/s | vs verifier off | draft accept | accepted/output | output tokens |
|---|---:|---:|---:|---:|---:|
| off | 76.99 | 1.000 | — | — | 9 |
| b2 | 61.97 | 0.805 | 1.0000 | 0.6667 | 9 |

## code
| budget | decode tok/s | vs verifier off | draft accept | accepted/output | output tokens |
|---|---:|---:|---:|---:|---:|
| off | 76.99 | 1.000 | — | — | 9 |
| b2 | 61.97 | 0.805 | 1.0000 | 0.6667 | 9 |
