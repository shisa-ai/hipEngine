# hipEngine GGUF-MTP category matrix

Raw root: `/tmp/hipengine-gguf-mtp-parity-workbench/2026-06-28-resident-production-category-b3-c5/category/resident-draft`

> **Diagnostic only:** true no-MTP AR baseline attached, so `vs true AR` is available for same-protocol diagnostics. This artifact is still not a retained speed claim unless `speed_claim_eligible=true` and `performance_claim=true`; `vs verifier off` remains diagnostic telemetry.

## Train / heldout split
Strategy: `fixed_category_heldout_v1`
Train prompts: `code_merge_intervals, code_topological_sort, code_lru_cache, general_en_plan, general_ja_plan, mixed_ja_en_translate`
Heldout prompts: `code_markdown_table, general_en_explain, general_ja_explain, mixed_ja_en_review`
Heldout covers all present categories: `True`

| split | budget | decode tok/s | vs verifier off | vs true AR | draft accept | accepted/output | prompts |
|---|---|---:|---: |---:|---:|---:|---:|
| full | b3 | 16.60 | 0.933 | 0.844 | 0.4933 | 0.5968 | 10 |
| train | b3 | 19.48 | 0.930 | 0.987 | 0.6000 | 0.6429 | 6 |
| heldout | b3 | 12.66 | 0.937 | 0.646 | 0.3333 | 0.5000 | 4 |

## Total
| budget | decode tok/s | vs verifier off | vs true AR | draft accept | accepted/output | output tokens |
|---|---:|---: |---:|---:|---:|---:|
| off | 17.79 | 1.000 | — | — | — | 124 |
| b3 | 16.60 | 0.933 | 0.844 | 0.4933 | 0.5968 | 124 |

## code
| budget | decode tok/s | vs verifier off | vs true AR | draft accept | accepted/output | output tokens |
|---|---:|---: |---:|---:|---:|---:|
| off | 21.56 | 1.000 | — | — | — | 59 |
| b3 | 20.07 | 0.931 | 1.021 | 0.6500 | 0.6610 | 59 |

## general_en
| budget | decode tok/s | vs verifier off | vs true AR | draft accept | accepted/output | output tokens |
|---|---:|---: |---:|---:|---:|---:|
| off | 16.52 | 1.000 | — | — | — | 25 |
| b3 | 15.51 | 0.939 | 0.791 | 0.5000 | 0.6000 | 25 |

## general_ja
| budget | decode tok/s | vs verifier off | vs true AR | draft accept | accepted/output | output tokens |
|---|---:|---: |---:|---:|---:|---:|
| off | 10.44 | 1.000 | — | — | — | 14 |
| b3 | 9.76 | 0.935 | 0.492 | 0.1333 | 0.2857 | 14 |

## mixed_ja_en
| budget | decode tok/s | vs verifier off | vs true AR | draft accept | accepted/output | output tokens |
|---|---:|---: |---:|---:|---:|---:|
| off | 18.84 | 1.000 | — | — | — | 26 |
| b3 | 17.49 | 0.929 | 0.893 | 0.5333 | 0.6154 | 26 |
