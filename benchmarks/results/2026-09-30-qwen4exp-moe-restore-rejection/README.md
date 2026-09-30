# `production_moe_restore` rejected on the task gate

**Verdict: reject the candidate, keep the strict fallback.** The restored
WMMA-MoE27 prefill route passes the numerical envelope in every category and
fails the task gate on three of eighteen prompts. No promotion, no default
changed.

## What was tested

`CANDIDATES["production_moe_restore"]`, classification T2, base profile
`production`, bound environment
`{PRODUCTION_MOE_PREFILL: "1", Q4_IU8_PREFILL: "1"}`.

The candidate restores the WMMA-MoE27 prefill route on layers 27-47 (iu8-WMMA
gate/up plus the compact f16-WMMA down), replacing the exact grouped
iu8-risk+repair route production runs today. Its mechanism text argues the
2026-09-14 arithmetic recovery zeroed the flag as part of a blanket sweep, and
that the sweep's own triggering candidate (`production_q8_fallback`) declared
"other production paths unchanged" and failed on task quality rather than on
this route. This run gives the route its own verdict instead of inheriting one
from the sweep.

## Result

| Gate | Outcome |
| --- | --- |
| lifecycle | **pass** — both arms return to 0 allocated bytes |
| numerical envelope | **pass** — all four categories |
| state / determinism | **pass** — no repeat mismatches, 3 repeats per prompt |
| task gate | **fail** — 15 of 18 strict-exact, `requires_review` |
| decision | **`reject candidate and keep strict fallback`** |

Numerical envelope by category (thresholds mean ≤ 1e-3, p95 ≤ 5e-3,
p99 ≤ 2e-2, max ≤ 5e-2, top-1 ≥ 0.99):

| Category | mean KL | p95 | p99 | max | top-1 |
| --- | --- | --- | --- | --- | --- |
| `code` | 1.177e-04 | 5.192e-04 | 2.084e-03 | 2.820e-03 | 0.9933 |
| `general_en` | 3.376e-04 | 1.708e-03 | 3.812e-03 | 4.312e-03 | 1.0000 |
| `general_ja` | 5.146e-04 | 2.236e-03 | 8.027e-03 | 1.144e-02 | 0.9900 |
| `mixed_ja_en` | 5.286e-04 | 2.669e-03 | 4.746e-03 | 5.343e-03 | 1.0000 |

The three divergences, all heldout prompts:
`heldout_code_rate_limiter`, `heldout_general_ja_migration`,
`heldout_general_ja_speculative`.

The envelope passes with margin on every axis while the task gate fails, and
there are no repeat mismatches, so the divergence is deterministic arithmetic
rather than flakiness. This is the second route in this campaign to be decided
by the task gate rather than by the envelope: `production_q8_fallback`, the
candidate that triggered the original sweep, also failed on task quality.

## Why this changes the ledger

`docs/QWEN4EXP-OPTIMIZATION-LEDGER.md` row 555 withdraws this flag pair on
*performance* grounds — "Slower on `code-p4096` in the disabled-family sweep" —
with reopen condition "New tiling, packing, fusion or repair work". A faster
tiling would not change this result, because the binding failure is task
quality. The reopen condition is corrected to require a mechanism that fixes
the three divergent prompts.

## Protocol

- Model `qwen4_exp_gguf`, quant `gguf_ud_q4_k_xl`, architecture `qwen4exp`,
  model root `/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL`.
- Prompt suites: `benchmarks/prompts/mtpbench-code-general-ja.jsonl` plus
  `benchmarks/prompts/gdn-prefill-category-heldouts.jsonl`, 18 prompts, the
  complete suite rather than a fixed subset.
- 24 decode steps, 450 teacher-forced rows, 3 candidate repeats, 32 free
  tokens, 2 free runs, prefill chunk 512, BF16 KV.
- Host: `gfx1151`, machine id `55ea6c509d0b49eea8de7094a1023668`, kernel
  `7.1.6-1-cachyos`.
- Source: `head 3f5abbd00430021a091a43d6138dba029b58be78`, `tracked_clean: true`.
- `measurement_valid: false` and `performance_claim: false` are expected: this
  is a correctness gate, not a timing harness, and it makes no speed claim.

Exact command:

```
python3.12 scripts/qwen4exp_layer2_profile_gate.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --candidate production_moe_restore \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
  --output /tmp/moe-restore-gate-clean/result.json
```

## Reproducibility

An earlier run on a dirty tree (`bg-138`, before the Q8 iu8 route removal
landed) reproduced this packet exactly — the envelope matches to every printed
digit and the same three prompts diverge. The route does not touch the removed
dense Q8 code, so the removal is not a confound; the clean run is the citable
one and the earlier packet is recorded in `artifact.json` under
`source_packet.supersedes`.

## Files

- `artifact.json` — decision-relevant fields, thresholds, protocol, and the
  provenance of both runs.
- `gate-packet-full.json` — the complete gate packet, including per-prompt
  state hashes and per-row task-gate detail.
