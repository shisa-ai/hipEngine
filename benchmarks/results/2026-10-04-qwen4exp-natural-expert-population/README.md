# Natural-prompt MoE expert population — 2026-10-04 UTC

Ledger row **ED-3**. The row states: *"The historical population probe measured 203 of
512 active experts on average using **synthetic token IDs `4000..5023`**, not a natural
code-p4096 prompt."* This measures the natural case.

## Result

**The synthetic model understates the natural population by 1.9x at the same token count.**

| input | tokens | assignments | active experts of 512 | share |
| --- | ---: | ---: | ---: | ---: |
| synthetic IDs `4000..5023` (historical) | 1,024 | 10,240 | 202.96 | 39.6% |
| **natural `code-p4096`, first 2 chunks** | 1,024 | 10,240 | **385.48** | **75.3%** |
| **natural `code-p4096`, full prompt** | 4,096 | 40,960 | **446.48** | **87.2%** |

The first two rows are exactly comparable — same token count, same top-10 routing, same
model, same per-layer structure. The delta is **+182.52 experts (+89.93%)**.

Range across the 48 MoE layers: 304–444 active at 1,024 tokens, **354–493 at 4,096**.

By down-projection quant, which the row also asked for:

| quant | layers | active @1,024 tok | active @4,096 tok |
| --- | ---: | ---: | ---: |
| `gguf_q5_1` | 43 | 388.4 | 451.7 |
| `gguf_q8_0` | 5 | 360.6 | 402.0 |

The Q8_0 layers are layers **2, 4, 30, 46, 47** — exactly the split ED-3 states (43 Q5_1,
five Q8_0). The Q8_0 population is lower but still ~1.8x the synthetic figure, so a pooled
number would not have hidden anything here.

## What this means, and what it does not

**Measured:** the population. **Inferred, not measured:** the MoE is 25.3% of the prefill
(gate/up 16.38% + down 8.92% from the stall census). If 75–87% of experts are active rather
than ~40%, then the expert-weight traffic behind the selected MoE kernels is close to the
**full** matrix, so the headroom a selected/sparse route can have over a dense one is much
smaller than the 40% model implies. The census *times* are measured and unaffected by this.

The synthetic input was not merely different, it was **pathological for this purpose**: with
10,240 assignments spread over only 203 experts, routing was strongly concentrated, which is
what a repetitive degenerate token sequence produces. A natural prompt spreads the same
number of assignments over 385.

## Category coverage — the population IS category-dependent, but mildly

All four natural categories measured at the **same shape** (1,024 tokens x top-10 = 10,240
assignments per layer), so every arm below is directly comparable to the synthetic probe:

| arm | active of 512 | share | range across layers | vs synthetic |
| --- | ---: | ---: | ---: | ---: |
| synthetic `4000..5023` | 202.96 | 39.6% | — | — |
| `code-p1024` | 385.48 | 75.3% | 304–444 | +89.9% |
| `general_en-p1024` | 359.10 | 70.1% | 268–412 | +76.9% |
| **`general_ja-p1024`** | **341.54** | **66.7%** | 210–402 | +68.3% |
| `mixed_ja_en-p1024` | 387.92 | 75.8% | 279–444 | +91.1% |
| **natural mean** | **368.51** | **72.0%** | | |

**Category spread: 46.38 experts = 12.6% of the natural mean.** The synthetic figure is
**44.9% below** the natural mean.

Two conclusions, and the second is the one that keeps the row open:

1. **The synthetic input is an outlier against every category**, not just against code. The
   39.6% regime does not appear in any natural prompt measured.
2. **The population is category-dependent.** `general_ja` — the category this campaign's task
   gates treat as the sensitive one — is the **lowest** at 341.54 (66.7%), 46.4 experts below
   `mixed_ja_en`. So the correct model is a per-category band, not a single number:
   **use ~368 of 512 (72%) rather than 203, with `general_ja` running ~7% below that mean.**

The spread is small relative to the synthetic error (12.6% against 44.9%), so a single
corrected figure is far better than the historical one — but it is not category-invariant, and
the per-layer ranges are wide (210–444), so any tuning that depends on the *per-layer*
population rather than the mean should read `populations.json`'s `category_arms.per_layer`.

## Protocol

| | |
| --- | --- |
| Host | `gfx1151`, AMD Radeon 8060S / Strix Halo |
| Model / quant | Qwen3.8 Flash Next, UD-Q4_K_XL, BF16 KV |
| Case | `code-p4096` — natural: four real code prompts concatenated (`code_merge_intervals`, `code_topological_sort`, `code_lru_cache`, `code_markdown_table`), 4,096 tokens, `prompt_token_ids_sha256` `72a815ea...` |
| Harness | `uv run python scripts/qwen4exp_profile_gap.py --mode prefill --case-id code-p4096 --moe-telemetry` |
| Profile / chunk | production, chunk 512 (8 chunks captured) |
| Capture | 384 telemetry rows = 8 chunks x 512 rows x 48 layers |

**Interpreter:** `uv run`, not `python3` — `python3 scripts/X.py` resolves `import hipengine`
through the conda editable `.pth` to a different clone. See worklog `20261004T` (iteration
149 of the ledger run-through).

## Limits

- Counts, not throughput measurements.
- One case, one model, one quant, one host (gfx1151, not the gfx1100 default hardware).
- The layer label is the telemetry's own; the `quant_triplet` is captured per row and is what
  the split above is derived from.
- The synthetic arm is the recovered fixture (`2026-09-29-qwen4exp-population-recovery`), not
  a re-run; its own artifact records its provenance limits.
- Natural-task routing is qualified for **this prompt family only**. A different category
  (`general_en`, `general_ja`, `mixed_ja_en`) may have a different population, and the
  category-heldout requirement in `docs/BENCHMARK.md` applies before any tuning is based on it.

## Files

- `populations.json` — per-layer populations at both token counts, assignments, per-expert
  extrema, and the captured quant triplet.
