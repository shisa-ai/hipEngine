# Disabled-selector ranking for `gguf_ud_q4_k_xl` on gfx1151

Date: 2026-10-04
Host: AMD Radeon 8060S (gfx1151), AMD RYZEN AI MAX+ 395, kernel `7.1.6-1-cachyos`,
machine-id `55ea6c509d0b49eea8de7094a1023668`. `tuned-adm active` =
`accelerator-performance` for every arm. Source `aa49c0850`, `tracked_clean: True`.
Model: unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL`, case `code-p4096` from
`benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json`
(sha256 `42b562bd…`), chunk 1024, 2 measured repetitions per arm, median reported.

This is a **performance-only** sweep. `performance_claim: false`,
`numerics_evaluated: false` in the report. A faster arm is not a qualified arm.

## The result: only the GR family wins, and the full set is worth +1.7%

Baseline median prefill **15.544 s** (263.5 tok/s).

| Rank | Arm | Median s | tok/s | Saved s | Ratio | Selector |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `gr_wide` | 13.703 | 298.9 | **+1.841** | **1.1344x** | `GR_WIDE` |
| 2 | `gr_both` | 13.828 | 296.2 | **+1.717** | **1.1241x** | `GR_IU8` + `GR_IU8_DOWN` |
| 3 | `gr_iu8` | 13.954 | 293.5 | **+1.590** | **1.1140x** | `GR_IU8` |
| 4 | `all_prefill` | 15.281 | 268.1 | +0.264 | 1.0172x | **all 13 overrides** |
| 5 | `gr_iu8_down` | 15.398 | 266.0 | +0.146 | 1.0095x | `GR_IU8_DOWN` |
| 6 | `gdn_colwarps` | 15.563 | 263.2 | −0.019 | 0.9988x | `GDN_COLWARPS_PREFILL` |
| 7 | `qsa_flash` | 15.583 | 262.8 | −0.039 | 0.9975x | `QSA_FLASH_PREFILL` |
| 8 | `gdn_peer` | 15.606 | 262.5 | −0.062 | 0.9960x | `GDN_PEER_PREFILL` |
| 9 | `q8_mmq` | 16.083 | 254.7 | −0.539 | 0.9665x | `Q8_MMQ_PREFILL` |
| 10 | `moe` | 16.094 | 254.5 | −0.550 | 0.9658x | `PRODUCTION_MOE_PREFILL` |
| 11 | `q4_iu8_fast` | 18.366 | 223.0 | −2.822 | 0.8463x | `Q4_IU8_PREFILL` |

**The headline is rank 4.** Enabling every disabled prefill selector at once —
`PRODUCTION_MOE_PREFILL`, `Q4_IU8_PREFILL`, `Q8_MMQ_PREFILL`, `GR_IU8`,
`GR_IU8_DOWN`, `GDN_PEER_PREFILL`, `GDN_COLWARPS_PREFILL`, `QSA_FLASH_PREFILL`
(13 overrides) — is worth **+1.7%**. The `GR_IU8` + `GR_IU8_DOWN` pair alone is
worth **+12.4%**, which is **6.5x what the whole set delivers**. Six of the nine
selectors are individually neutral-to-negative on this case.

`gr_wide` is not in `all_prefill`; that arm's table excludes it, so rank 4
understates the full set by exactly the one selector that wins most.

## Engagement: every override is verified, not assumed

Each arm's `route_env` in its `child.json` records the selector actually bound,
and `effective_route_missing_keys` is empty for all twelve arms. Spot checks:

```
baseline         PROD=0 Q8=0 Q4=0 GDN=0 QSA=0  || GR_IU8=- GR_IU8_DOWN=- GDN_PEER=- GR_WIDE=-
q8_mmq           PROD=0 Q8=1 Q4=0 GDN=0 QSA=0  || GR_IU8=- GR_IU8_DOWN=- GDN_PEER=- GR_WIDE=-
gr_both          PROD=0 Q8=0 Q4=0 GDN=0 QSA=0  || GR_IU8=1 GR_IU8_DOWN=1 GDN_PEER=- GR_WIDE=-
all_prefill     PROD=1 Q8=1 Q4=1 GDN=1 QSA=1  || GR_IU8=1 GR_IU8_DOWN=1 GDN_PEER=1 GR_WIDE=-
```

`named_profile_intact` is `false` for every overridden arm and `true` for the
baseline: these are post-binder diagnostic overrides, not profile states, which
is the expected and correct result. Every arm produces the same generated
`token_id` (248068) and all twelve logits hashes differ, as they must.

This check is not ceremonial here: `docs/REFACTOR.md` records that a GR or dense
route screened against the *default* prompt suite never executes at all (39-71
tokens against a `rows > 256` admission) and "scores a perfect zero, which reads
as a pass". The canonical `code-p4096` fixture is above that admission, and the
route dump confirms it.

## The 2026-09-16 ranking is stale in ORDER, not just in level

`benchmarks/results/2026-09-16-disabled-family-recoverable-time/` measured the
same question with a different driver
(`scripts/qwen4exp_disabled_family_sweep.py`) and its fallback median was
**23.648 s** against today's **15.544 s** — the recovery baseline is **1.52x
faster** than it was. The ordering moved with it:

| Selector | 2026-09-16 | today | |
| --- | ---: | ---: | --- |
| `Q8_MMQ_PREFILL` | 1.203x | **0.9665x** | **inverted** |
| `Q4_IU8_PREFILL` | 0.997x | **0.8463x** | **inverted** |
| `GDN_COLWARPS_PREFILL` | 1.010x | 0.9988x | inverted (noise) |
| `GDN_PEER_PREFILL` | 1.005x | 0.9960x | inverted (noise) |
| `GR_IU8` | 1.085x | 1.1140x | still a win |
| `GR_IU8_DOWN` | 1.028x | 1.0095x | still a small win |
| `PRODUCTION_MOE_PREFILL` | 0.976x | 0.9658x | still a loss |
| `GR_WIDE` | not measured | **1.1344x** | **new, and the largest** |

The old sweep's top two were `Q8_IU8_WMM` (1.388x, since closed — 14 of 17
claiming roles fail the canonical envelope alone) and `Q8_MMQ_PREFILL`
(1.203x, now a **loss**). **Neither is available**, and the largest current
winner was not in that sweep at all. Reading that artifact as an opportunity
list would have pointed at the wrong selector.

## Numerical history per selector, from the in-tree evidence

Performance is one axis; admissibility is another and is **not** re-derived here.

The binding scope for this profile is **`prefill_last` / `prefill_to_c1`**,
because that is exactly the scope these selectors move: the recovery profile's
`prefill_last` is **bit-identical at mean KL 0.0**, while the unrecovered
composition measures **mean 1.333e-3 against a 1e-3 limit** and max KL
`0.05958` against `0.05` (`docs/EXECUTION-PROFILES.md` §2.9).

The dispositions below come from `docs/REFACTOR.md`, which is **newer and more
specific** than the ledger's section-7 rows. One distinction matters and is easy
to get wrong: the ledger marks several of these selectors "SUPERSEDED — this
route is a current production default", but that is a **profile-scope** fact
(the binder binds them ON for `gguf_q4_k_m` and other non-UD quants), **not a
numerical re-qualification for this profile**.

| Selector | Saved s | Standing numerical disposition | Reachable? |
| --- | ---: | --- | --- |
| `GR_WIDE` | **+1.841** | Gate rejected: mean KL 1.554e-3, p95 7.081e-3, max **7.349e-2**, top-1 768/780, all 12 mismatches flip-eligible. Determinism/state/lifecycle pass, so the failure is purely numerical. REFACTOR: *"The GR family is closed to arithmetic change: every measured route, including corrections more accurate than the incumbent, saturates the gate at 1.3e-3 – 1.6e-3."* | **only via a bit-exact change** |
| `GR_IU8` | **+1.590** | Prefill-boundary KL **1.48e-8 against a ~5e-9 cliff**; all 8 top-1 flips are decode rows. REFACTOR: *"Keep this arithmetic off. Revisit only after actual-input projection/sigmoid/mean boundary localization and a new correction, not an unchanged flag retry."* | no |
| `GR_IU8_DOWN` | +0.146 | Prefill-boundary KL **4.32e-8** against the same cliff; its compensated correction reached **8.48e-9**, still **1.9x above** the passing bracket. | no |
| `Q8_MMQ_PREFILL` | −0.539 | "Passed short prompts, failed canonical depth after the recovery"; ledger notes the section-7 rejection was never given its own operand localization. | moot — now a loss |
| `GDN_COLWARPS_PREFILL` | −0.019 | "Inside run-to-run spread". REFACTOR: the flag is the **gate** that admits the tiled16 owner; the columnwarp kernel it is named after is a measured **1.53x slower** than that owner. | moot — noise |
| `GDN_PEER_PREFILL` | −0.062 | "Inside run-to-run spread". | moot — noise |
| `QSA_FLASH_PREFILL` | −0.039 | *"Part of the failed recovery composition"* — a **withdrawn-by-association** reason with no measurement of the route itself. | moot — noise |
| `PRODUCTION_MOE_PREFILL` | −0.550 | Envelope **PASSES** all four categories (mean 1.18e-4–5.29e-4); **task gate FAILS 15/18** on three heldout prompts, deterministically. | moot — loss *and* task-gate failure |
| `Q4_IU8_PREFILL` | −2.822 | Envelope passes alongside MOE; rejected with it. | moot — large loss |

Two structural findings fall out of this table:

- **`GR_WIDE`'s prefill-boundary KL screen is not a predictor.** It screened at
  `5.17e-11` — 285x *better* than the rejected iu8 route's `1.48e-8` — while
  carrying 300x more projection drift (4.7e-5 against 1.1e-7 relative L2), and
  then failed at the same plateau. The publication row's ~22-logit argmax gap
  makes the softmax one-hot, so KL is blind to logits that genuinely moved
  (median absolute drift 0.095, max 1.20). **Do not admit or reject a GR
  candidate on that screen**; go to the canonical-fixture depth gate.
- **The GR family's failure mode is a discrete router flip, not diffuse drift.**
  That is why every arithmetic route saturates at the same plateau regardless of
  accuracy, and why "a more accurate correction" has repeatedly failed to land.

## Minimum re-admission set: **none of the nine**

This is the task's headline answer and it is negative. **No selector in this
ranking can be re-admitted on this evidence**, and the reason is different for
each group:

- **`Q8_MMQ_PREFILL`, `PRODUCTION_MOE_PREFILL`, `Q4_IU8_PREFILL`** are **losses
  today** (−0.539, −0.550, −2.822 s). There is no value to requalify, so the
  requalification cost is not worth paying at any risk. `PRODUCTION_MOE_PREFILL`
  additionally fails the task gate 15/18, and REFACTOR records the correct
  consequence: *"a faster tiling cannot change a task-gate failure."*
- **`GDN_COLWARPS_PREFILL`, `GDN_PEER_PREFILL`, `QSA_FLASH_PREFILL`** are all
  within ±0.4% — **no material effect** at two repetitions. `QSA_FLASH_PREFILL`
  has no own measurement at all.
- **`GR_IU8`, `GR_IU8_DOWN`, `GR_WIDE`** are the only material wins (+1.590,
  +0.146, +1.841 s) and all three are blocked on numerics for this profile,
  which needs `prefill_last` bit-identical. The two iu8 routes sit 3x and 8.6x
  above a ~5e-9 prefill-boundary cliff, and the wide tile fails the 780-row gate
  outright.

**The reachable value is a kernel task, not a re-admission.** REFACTOR names the
GR-up wide tile's bit-exact levers as *"the only actionable ones"*:

1. **`ROW_BATCH` 4 → 8 at `HIDDEN_TILE=2`**, which halves the 4x2x4 block tile's
   weight re-read without touching the per-output k-order, the shuffle tree, or
   the 4-wave LDS sum — **estimated ~900 ms** of the code-p4096 prefill.
2. **Drop the dead `gate` store**, ~107 ms: in the `fused_up` path
   `scratch.gate` is written and read by nothing. Use a template parameter
   rather than deletion, since `gr_wide` and `gr_iu8` do read it back.

Together those are ~1.0 s of the 1.841 s that `GR_WIDE` measures — and they are
**bit-exact**, which is the one property this profile requires. The previously
measured `RB8/HT1` retile (0.80x) is **not** evidence against lever 1: it moved
`ROW_BATCH` and `HIDDEN_TILE` together, and `HT1` halves the FMAs per x load
while doubling the block count, so it is confounded. No clean `RB8/HT2` variant
exists in-tree.

**So the ranked recommendation is:** build the `RB8/HT2` wide-tile variant plus
the dead-store removal, then gate it — expected ~1.0 s at bit-exactness, which
would be the largest single prefill win available in this family. Everything
else in this sweep is deferred because it is a loss, noise, or blocked on a
mechanism nobody has yet produced.

## What this does not establish

- **One case, one shape.** `code-p4096` only. The family shares differ at 512
  and 1024 and this sweep says nothing about them.
- **Two repetitions per arm.** Ranks 6-8 span 0.019-0.062 s and should be read
  as "no material effect" rather than as an ordering.
- **No numerics.** The `token_id` agreement is a single-token check, not a gate.
  Nothing here re-admits anything, and the dispositions above are *read from*
  the in-tree evidence rather than re-measured.
- **The §2.9 recovery cost does not reconcile with this sweep, and that is
  unresolved.** §2.9 records the unrecovered composition as
  `-40.94/-41.89/-54.53%` prefill against the recovery profile. On today's tree
  at `code-p4096`, enabling all eight disabled prefill selectors recovers
  **+1.7%**. The likely cause is visible in the same data — the recovery
  baseline improved 1.52x since the 2026-09-16 sweep, so the recovery profile's
  own optimization appears to have closed most of the gap — but §2.9's figure is
  a *weighted* p512/p1024/p4096 measurement of the `gguf_q4_k_m` **profile**,
  while this is one case under post-binder overrides on the UD profile. The two
  are not the same measurement and the discrepancy is not settled here. **A
  re-derivation of the §2.9 recovery cost at today's HEAD is the follow-up this
  result creates**, and it is now the more valuable question: if the cost really
  has collapsed from 41-55% to ~2%, §2.9's product-policy framing is obsolete.
- **`all_prefill` omits `GR_WIDE`**, so rank 4 is a lower bound on the full set.

## Reproduce

```bash
source /tmp/reprof/env.sh && export PYTHONPATH=/home/lhl/hipEngine
python3 scripts/qwen4exp_disabled_selector_ranking.py \
  --work-root /tmp/qwen4exp-selector-ranking \
  --output benchmarks/results/2026-10-04-qwen4exp-disabled-selector-ranking/ranking.json \
  --case-id code-p4096 --repetitions 2 --prefill-chunk-size 1024
```

Per-arm raw records remain in `/tmp/qwen4exp-selector-ranking/<arm>/child.json`
(`route_env`, `manifest_sha256`, `wall_summary`, `logits_sha256`); the committed
`ranking.json` carries the per-arm ratios, effective routes and commands.

## Artifacts

- `ranking.json` — the harness report: 12 arms with `tok_s`,
  `ratio_vs_baseline`, `delta_pct_vs_baseline`, `effective_route`,
  `overrides_recorded`, `named_profile_intact`, `token_id`, command.
