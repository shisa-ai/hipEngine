# hipEngine GGUF-MTP category matrix

Raw root: `/tmp/hipengine-gguf-mtp-parity-workbench/2026-06-28-resident-serial-fallback-category-b3-c5/category/resident-serial-fallback`

> **Diagnostic only:** true no-MTP AR baseline attached, so `vs true AR` is available for same-protocol diagnostics. This artifact is still not a retained speed claim unless `speed_claim_eligible=true` and `performance_claim=true`; `vs verifier off` remains diagnostic telemetry.

## Train / heldout split
Strategy: `fixed_category_heldout_v1`
Train prompts: `code_merge_intervals, code_topological_sort, code_lru_cache, general_en_plan, general_ja_plan, mixed_ja_en_translate`
Heldout prompts: `code_markdown_table, general_en_explain, general_ja_explain, mixed_ja_en_review`
Heldout covers all present categories: `True`

| split | budget | decode tok/s | vs verifier off | vs true AR | draft accept | accepted/output | prompts |
|---|---|---:|---: |---:|---:|---:|---:|
| full | b3 | 47.62 | 0.870 | 2.421 | 0.5417 | 0.4382 | 10 |
| train | b3 | 47.18 | 0.862 | 2.392 | 0.6491 | 0.5522 | 6 |
| heldout | b3 | 49.00 | 0.894 | 2.501 | 0.1333 | 0.0909 | 4 |

## Total
| budget | decode tok/s | vs verifier off | vs true AR | draft accept | accepted/output | output tokens |
|---|---:|---: |---:|---:|---:|---:|
| off | 54.76 | 1.000 | — | — | — | 89 |
| b3 | 47.62 | 0.870 | 2.421 | 0.5417 | 0.4382 | 89 |

## code
| budget | decode tok/s | vs verifier off | vs true AR | draft accept | accepted/output | output tokens |
|---|---:|---: |---:|---:|---:|---:|
| off | 54.83 | 1.000 | — | — | — | 48 |
| b3 | 47.19 | 0.861 | 2.401 | 0.6667 | 0.5833 | 48 |

## general_en
| budget | decode tok/s | vs verifier off | vs true AR | draft accept | accepted/output | output tokens |
|---|---:|---: |---:|---:|---:|---:|
| off | 54.86 | 1.000 | — | — | — | 10 |
| b3 | 49.86 | 0.909 | 2.543 | 0.0000 | 0.0000 | 10 |

## general_ja
| budget | decode tok/s | vs verifier off | vs true AR | draft accept | accepted/output | output tokens |
|---|---:|---: |---:|---:|---:|---:|
| off | 54.89 | 1.000 | — | — | — | 10 |
| b3 | 49.90 | 0.909 | 2.513 | 0.0000 | 0.0000 | 10 |

## mixed_ja_en
| budget | decode tok/s | vs verifier off | vs true AR | draft accept | accepted/output | output tokens |
|---|---:|---: |---:|---:|---:|---:|
| off | 54.49 | 1.000 | — | — | — | 21 |
| b3 | 46.59 | 0.855 | 2.378 | 0.6111 | 0.5238 | 21 |
