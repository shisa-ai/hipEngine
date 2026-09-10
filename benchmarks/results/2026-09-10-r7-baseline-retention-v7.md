# R7 post-promotion baseline retention - v7b packet (batched QSA position promote)

First profile change since the R2e seven-promotion state: batched QSA position
prepare promoted to default (b7e9f19b9). Three-engine capture at the promoted
default, validated by the retention verifier (all samples validate,
fixture/host/binary hashes match).

**v7b**: the halo-box-hip comparator stage was re-captured (resume from a
b7e9f19b9 worktree; hipengine/vulkan stages hash-verified and carried over)
after the original v7 comparator window ran at 21.03% PP CV with rates
depressed -15/-11/-6% vs v6. The re-captured stage restores v6-family rates;
residual PP CV 11.35% recorded as a comparator-window caveat.

## Rates (v6 -> v7b)

| Engine | p512 PP / TG | p1024 PP / TG | p4096 PP / TG | Max PP / TG CV |
| --- | ---: | ---: | ---: | ---: |
| hipengine | 294.33 -> 296.28 / 19.70 -> 20.02 | 313.05 -> 294.99 / 19.26 -> 19.53 | 288.54 -> 262.21 / 18.48 -> 19.14 | 3.18% / 3.93% |
| halo-box-vulkan | 344.40 / 26.02 | 393.59 / 25.59 | 413.23 / 24.73 | 0.19% / 0.07% |
| halo-box-hip | 316.67 -> 305.44 / 22.17 -> 22.01 | 387.24 -> 385.74 / 21.17 -> 21.18 | 354.19 -> 349.59 / 19.47 -> 19.49 | 11.35% / 2.88% |

## Disposition

- hipEngine TG improved at every shape (+1.6% / +1.4% / +3.6%), consistent
  with the promotion's gated A/B (-0.77 ms/token ~ 1.3%).
- hipEngine PP at p1024/p4096 (packet-level -5.8% / -9.1% vs v6) was
  discriminated as environmental by a direct interleaved same-process PP A/B
  (all deltas within +/-0.85%, median ~0.1%) across all p1024/p4096 cases;
  the original comparator's simultaneous PP depression corroborates a noisy
  host window during that capture.
- halo-box-hip: v7 window anomalous (21.03% PP CV, rates -15/-11/-6%);
  re-captured in v7b to v6-family rates (within ~3.5% PP, TG identical to
  ~0.5%). Residual 11.35% PP CV is a comparator caveat, not a hipEngine
  signal.

## Direct PP A/B evidence (interleaved, same process)

| case | off (ms) | on (ms) | delta |
| --- | ---: | ---: | ---: |
| code-p1024 | 3564.9 | 3564.0 | -0.02% |
| code-p4096 | 14502.2 | 14478.6 | -0.16% |
| general_en-p1024 | 3533.8 | 3531.9 | -0.06% |
| general_en-p4096 | 14362.2 | 14363.9 | +0.01% |
| general_ja-p1024 | 3526.2 | 3538.5 | +0.35% |
| general_ja-p4096 | 14323.1 | 14388.1 | +0.45% |
| mixed_ja_en-p1024 | 3575.3 | 3553.4 | -0.61% |
| mixed_ja_en-p4096 | 14425.0 | 14547.4 | +0.85% |
