# Recoverable prefill time in the disabled Qwen4Exp families

**What this is.** A measurement of how much time each disabled prefill family
would save on a real production prefill, ranked by seconds recoverable rather
than by any historical headline gain. It measures the performance half of the
decision. Admissibility is a separate axis and is not re-derived here.

**What this is not.** These runs are diagnostics, not rate rows. Enabling any of
these families through a post-binder override breaks the named production
profile (`named_profile_intact` is false for every candidate), which is expected:
they are diagnostic overrides, not profile states.

## Protocol

| | |
| --- | --- |
| host | Framework `gfx1151`, Radeon 8060S |
| model | unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL` |
| case | `code-p4096` from the canonical fixture (4096 exact token ids) |
| harness | `scripts/qwen4exp_profile_gap.py --mode prefill --risk-diagnostics` |
| repetitions | 2 per arm, median reported |
| driver | `scripts/qwen4exp_disabled_family_sweep.py` |
| fallback | the current default with no overrides |

Each family is enabled by a post-binder `--override`, so it is applied after the
named profile binder and recorded in the run's `route_env`.

## Result

Fallback median prefill: **23.648 s**.

| Rank | Family | Median s | Saved s | Speedup | Repair rate | Same logits |
| ---: | --- | ---: | ---: | ---: | ---: | :---: |
| 1 | `Q8_IU8_WMM` | 17.034 | **+6.614** | **1.388x** | 0.7115% | no |
| 2 | `Q8_MMQ_PREFILL` | 19.662 | **+3.986** | 1.203x | 0.7117% | no |
| 3 | `GR_IU8` | 21.800 | +1.849 | 1.085x | 0.7119% | no |
| 4 | `GR_IU8_DOWN` | 23.004 | +0.645 | 1.028x | 0.7114% | no |
| 5 | `GDN_COLWARPS_PREFILL` | 23.420 | +0.228 | 1.010x | 0.7120% | no |
| 6 | `GDN_PEER_PREFILL` | 23.522 | +0.126 | 1.005x | 0.7122% | no |
| 7 | `Q4_DP4A64` | 23.582 | +0.066 | 1.003x | 0.7121% | **yes** |
| 8 | `Q4_IU8_PREFILL` | 23.725 | -0.077 | 0.997x | 0.7121% | **yes** |
| 9 | `PRODUCTION_MOE_PREFILL` | 24.243 | -0.595 | 0.976x | 0.7162% | no |

**These rows overlap and must not be summed.** `total_seconds_recoverable` in the
artifact is the sum of the individual savings and is reported only to show the
upper bound if every family were independent; it is not a reachable number.

### Combined measurement of the top four

The four largest families were enabled together, because summing overlapping
rows would overstate the opportunity. The measured combined result is well below
the sum:

| | median prefill s | saved s | speedup |
| --- | ---: | ---: | ---: |
| fallback | 23.648 | — | — |
| sum of the four individually | — | +13.094 | (not reachable) |
| **`Q8_IU8_WMM` + `Q8_MMQ_PREFILL` + `GR_IU8` + `GR_IU8_DOWN`** | **15.840** | **+7.808** | **1.493x** |

So the measured combined opportunity is **+33% on this prefill**, against a sum
that would have claimed +55%. The repair rate for the combined run is 0.7118%
and `over_capacity_calls` is zero, both unchanged from the individual runs.

## What the repair rates say

The repair rate is essentially constant at **0.711-0.716%** across all nine
candidates, including the two that are bit-identical to the fallback and the one
that is slower than it.

That is the expected result once the mechanism is visible: the risk queue comes
from the **always-on** exact `Q4_IU8_EXACT` route, which the fallback already
runs. Enabling another family does not add risk to the iu8 path; it changes which
other kernel runs alongside it. So the risk-criterion trigger rate is a property
of the instrument and the input, not of the family under test.

The practical consequence: **an exact risk-plus-repair variant is not what makes
these families expensive.** The repair rate is already under 1%, and it does not
move when the family is enabled. A corrected arithmetic for any of these families
would not have to pay a materially larger repair cost than the default already
pays.

`over_capacity_calls` is zero for every candidate, so the bounded risk queue
never overflowed on this input.

## Admissibility, from the project's own evidence

> **Correction (2026-09-30, added after the sweep was cited as an opportunity list).** Two
> things in this section are stale and one is structural. Read the flags first, because they
> change what the sweep is a measurement *of*.
>
> **Every family in the top six is bound to `"1" if production` in the profile binder**
> (`hipengine/generation/qwen4_exp_profiles.py`): `Q8_MMQ_PREFILL`, `GR_IU8`,
> `GR_IU8_DOWN`, `GDN_COLWARPS_PREFILL`, `GDN_PEER_PREFILL` and `Q4_DP4A64` are all in
> `PRODUCTION_ARITHMETIC_RECOVERY_FLAGS` and all bound ON for the non-recovery production
> profile. So the savings in the table above are **not unexploited headroom**: they are the
> difference between the **recovery profile** (this artifact's fallback) and the
> **production profile**, i.e. a re-measurement of the recovery cost already recorded as
> `-40.94/-41.89/-54.53%` prefill and `-12.60/-12.43/-36.88%` decode in
> `2026-09-14-q8-prefill-numerics`. The `+1.493x` combined figure is what the recovery
> profile gives back, not a prize sitting on the table.
>
> **The `Q8_IU8_WMM` disposition below is stale and the family is now closed.** It read "no
> current rejection recorded at this scope; largest measured opportunity in this sweep".
> The role-isolated evidence that rejection cell asked for has since been produced: **14 of
> 17 roles claim the route and all 14 fail the canonical envelope alone, zero pass**
> (`docs/QWEN4EXP-OPTIMIZATION-LEDGER.md`, row for `Q8_IU8_WMM`, 2026-09-30). The 1.388x is
> real as a rate and unavailable as a route.
>
> **The `GR_IU8` / `GR_IU8_DOWN` dispositions below are also stale.** They were rejected at
> this scope, and the binder now binds both to `"1" if production`; the ledger records them
> as superseded production defaults. A rejection at one scope does not survive a
> re-qualification at another, and this table did not say so.
>
> **What is still true and still useful here:** the protocol, the measured rates, the
> overlap warning, the combined measurement, and the repair-rate finding (that an exact
> risk-plus-repair variant is not what makes these families expensive). Those stand.

The performance numbers above are not a promotion argument. Two of the three
largest entries already have recorded numerical rejections:

| Family | Recorded disposition |
| --- | --- |
| `GR_IU8` | Fails the 780-row production gate unchanged: mean KL `0.0014024`, p95 `0.0074015`, top-1 772/780. `docs/REFACTOR.md` |
| `GR_IU8_DOWN` | Fails independently: mean KL `0.0013128`, p95 `0.0068352`, top-1 769/780. `docs/REFACTOR.md` |
| `Q8_IU8_WMM` | No current rejection recorded at this scope; largest measured opportunity in this sweep |
| `Q8_MMQ_PREFILL` | Owned by the named production manifest for layers 32-47 in the MMQ stack; enabling it here is a diagnostic override |

`Q8_IU8_WMM` at 1.388x is the largest single measured rate in this sweep. **It is not a
target:** the role-isolated screen has since shown all 14 claiming roles fail the canonical
envelope alone, so the route is closed rather than awaiting a corrected arithmetic. See the
correction above.

## What this does not establish

- One case at one shape. `code-p4096` only. The family shares differ at 512 and
  1024 and this sweep says nothing about them.
- Two repetitions per arm. The differences at ranks 5-9 (0.066-0.228 s) are
  within the run-to-run spread this project has already measured, and should be
  read as "no material effect" rather than as a ranking.
- No numerical evaluation. `same logits` records only whether the final prefill
  token's logits hash matches the fallback; it is not a quality gate. Six of the
  nine candidates already differ at that single token, so a full evaluation is a
  precondition for any of them, not a formality.
- Enabling these families through overrides breaks the named production profile,
  so none of these configurations is a supported profile state.
- The combined run is one measurement of one combination. Other combinations
  were not measured and are not implied by it.
