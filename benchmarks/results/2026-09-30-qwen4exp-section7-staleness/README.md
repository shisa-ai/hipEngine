# Six section-7 rows record as "not retained" flags that are production defaults

**Section 7 is the ledger's "tried and not retained" table — its only purpose is to tell a
reader what *not* to retry. Six of its 78 rows name flags that are currently bound to
`"1" if production`.** Three of the six were withdrawn *by association*, not on their own
measurement.

Verified by **reading the binder**, not by inference from names or comments:
`hipengine/generation/qwen4_exp_profiles.py`, production branch.

| row | flag | recorded reason | current |
| --- | --- | --- | --- |
| Dense Q8 MMQ stack | `Q8_MMQ_PREFILL` | failed canonical depth | **ON** |
| `PRODUCTION_MOE_PREFILL` / `Q4_IU8_PREFILL` | both | rejected, keep strict fallback (2026-09-30) | **ON** |
| iu8 WMMA GR up/down | `GR_IU8`, `GR_IU8_DOWN` | mean KL 1.40e-3 / 1.31e-3 | **ON** |
| QSA flash prefill | `QSA_FLASH_PREFILL` | part of the failed recovery composition | **ON** |
| Column warps, peer | `GDN_COLWARPS_PREFILL`, `GDN_PEER_PREFILL` | inside run-to-run spread | **ON** |
| DP4A safe-43 decode | `Q4_DP4A64` | part of the failed composition | **ON** |

## Withdrawn by association

Three rows give *"Part of the failed recovery composition"* as their whole reason — no
measurement of the route itself. **This is the defect that hid the multi-column GDN win**,
where `R02` was condemned by a row citing only `R01`'s evidence.

`GDN_COLWARPS_PREFILL` is worth singling out: it is not merely live, it is the **gate the
current GDN prefill default runs inside** — `qwen4_exp_gdn_tile16_prefill_selected` is only
consulted when `colwarps_prefill` is true. The recorded *"+0.228 / +0.126 s, inside
run-to-run spread"* therefore describes the wall of what is now the default path.

## One settled case: `GR_IU8` / `GR_IU8_DOWN`

This row is **simply stale**, and the size of the error is the point. It records
**mean KL 1.40e-3 / 1.31e-3**. The binder records what actually shipped:

> GR up: *"teacher-forced logits KL <= 7.4e-8 versus the incumbent with 100% top-1 and
> 10-100x argmax margins, bit-deterministic; full-suite A/B +9.2% prefill geo-mean with
> all 12 cases winning"*
>
> GR down: *"KL <= 5.5e-8 vs incumbent, top-1 100%, deterministic; full-suite A/B +2.0%
> prefill geo-mean on top of the promoted up leg"*

**Five orders of magnitude in KL.** The reason is that the promoted mechanism is a
different one — three-plane fp32 staging — and **the row's own reopen condition, "A
structurally different mechanism", was met.** Nothing tracked it.

So there is **no win left here** — the +9.2%/+2.0% is already in production. Only a wrong
record.

## The systemic defect

**Every section-7 row carries a "Reopen only with" condition, and nothing in the
repository evaluates whether one has been satisfied.** A row can be correct when written,
superseded by later work, and still read as a rejection by anyone consulting the table —
which is the table's only purpose.

**Consequence for the census: section 7 is a candidate *source*, not only a graveyard.**
The retained multi-column GDN win (1.34x-1.68x) was found by asking whether a recorded
rejection still held.

## Not established

- Whether the Q8 MMQ depth failure was later resolved, or the flag was restored without a
  re-gate.
- Whether the 2026-09-30 `PRODUCTION_MOE_PREFILL`/`Q4_IU8_PREFILL` task-gate rejection
  coexists with both flags enabled by design or by drift.
- **Whether any of the three association-withdrawn rows holds a further win.**

## Limits

One profile (production, `gguf_ud_q4_k_xl`), one host (gfx1151), static reading of the
binder. **No timing was run for this artifact** — `performance_claim: false`.
