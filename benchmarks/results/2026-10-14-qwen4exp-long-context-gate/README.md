# Qwen4Exp long-prompt gate against a round-off floor (2026-10-14)

`qwen4_exp_gguf` / `hip_gfx1151` / `gguf_ud_q4_k_xl`, `production` against
`strict`, Radeon 8060S / gfx1151, host `gfx1151` (machine
`55ea6c509d0b49eea8de7094a1023668`), BF16 KV, chunk 4096, tracked-clean
`d093d40b6`, measured 2026-10-14 UTC. Wall time 24.5 minutes.

```bash
python scripts/qwen4exp_long_context_gate.py --model-root <UD-Q4_K_XL> \
  --output gate-production.json
```

## Why this gate

The production envelope (mean KL 1e-3, p95 5e-3, ...) was calibrated on
prompts of tens to hundreds of tokens. At 2,000 tokens and more, any change
that is not bit-identical to `strict`, even an FP32 summation-order change in
one projection, measures about 2e-3 mean KL, so fixed limits there reject
every plan and separate nothing. This gate measures what one rounding
flip does on the same prompts and requires the candidate to do no more.

## Numerical axis: candidate against the round-off floor

Four prompts of 1,981 to 2,572 tokens
(`benchmarks/prompts/qwen4exp-long-context-gate.jsonl`), 24 teacher-forced
decode steps each, 100 rows. Each control is `strict` with one BF16 ulp
flipped in one element of one prompt embedding row (seeds 1-3).

| Arm against `strict` | Mean KL | p95 | p99 | Max | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control, seed 1 | 4.48e-3 | 2.14e-2 | 4.82e-2 | 5.32e-2 | 98/100 |
| Control, seed 2 | 5.03e-3 | 2.24e-2 | 4.37e-2 | 1.09e-1 | 99/100 |
| Control, seed 3 | 7.69e-3 | 4.21e-2 | 6.63e-2 | 1.16e-1 | 97/100 |
| **Floor** (median mean/p95, largest max, lowest top-1) | 5.03e-3 | 2.24e-2 | — | 1.16e-1 | 97 |
| `production` | **2.13e-3** | **8.28e-3** | 3.11e-2 | **3.46e-2** | **98/100** |

A single rounding flip moves these prompts more than the complete
`production` route composition does. The two candidate repeats are
bit-identical.

## Task axis: retrieval at 8K and 16K

`benchmarks/prompts/qwen4exp-long-retrieval-v1.jsonl`
(built by `scripts/build_qwen4exp_long_retrieval_suite.py`): seeded English
and Japanese shipping logs of 8,165 to 16,357 tokens, four questions each
whose answers sit at 10%, 40%, 70% and 95% depth. Each arm is
teacher-forced through the gold answer text; an answer counts when the
argmax reproduces every token of its number.

| Prompt | Tokens | `strict` | `production` | Gold NLL `strict` / `production` |
| --- | ---: | ---: | ---: | ---: |
| English, 8K | 8,165 | 4/4 | 4/4 | 1.56e-4 / 1.85e-4 |
| Japanese, 8K | 8,176 | 4/4 | 4/4 | 1.45e-4 / 1.39e-4 |
| English, 16K | 16,337 | 4/4 | 4/4 | 4.37e-4 / 4.69e-4 |
| Japanese, 16K | 16,357 | 4/4 | 4/4 | 1.13e-4 / 1.15e-4 |

On the 123 gold-forced rows, `production` against `strict` measures mean KL
6.0e-6, max 2.2e-4, top-1 123/123.

**Status: passed** (every check in `gate-production.json` `decision.checks`).

Limits of this evidence: four numerical contexts of 25 correlated rows each,
three controls, and retrieval questions that both arms answer with very low
NLL, so the retrieval axis catches a broken long-context path rather than a
small quality loss.
