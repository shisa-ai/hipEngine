# R7 post-promotion baseline retention - v7 packet (batched QSA position promote)

First profile change since the R2e seven-promotion state: batched QSA position
prepare promoted to default (b7e9f19b9). Fresh three-engine capture at the
promoted default (tree clean at promotion commit), validated by the retention
verifier (all samples validate, fixture/host/binary hashes match).

## Rates (v6 -> v7)

| Engine | p512 PP / TG | p1024 PP / TG | p4096 PP / TG | Max PP / TG CV |
| --- | ---: | ---: | ---: | ---: |
| hipengine | 294.33 -> 296.28 / 19.70 -> 20.02 | 313.05 -> 294.99 / 19.26 -> 19.53 | 288.54 -> 262.21 / 18.48 -> 19.14 | 3.18% / 3.93% |
| halo-box-vulkan | 344.40 / 26.02 | 393.59 / 25.59 | 413.23 / 24.73 | 0.19% / 0.07% |
| halo-box-hip | 268.24 / 21.38 | 345.74 / 20.22 | 333.72 / 18.45 | 21.03% / 4.63% |

## Disposition

- TG improved at every shape (+1.6% / +1.4% / +3.6%), consistent with the
  promotion's gated A/B (-0.77 ms/token ~ 1.3%).
- Packet-level hipEngine PP at p1024/p4096 dropped (-5.8% / -9.1%), outside
  CV. A direct interleaved same-process PP A/B (3 reps, flag off/on) shows
  deltas within +/-0.85% on all p1024/p4096 cases (median ~0.1%): prefill is
  unaffected by the change. The packet PP drop is environmental - the
  halo-box-hip comparator (untouched binary) dropped PP simultaneously
  (-15%/-11%/-6%) with 21% CV, while vulkan was stable.
- halo-box-hip ran a noisy PP window in this capture (21% CV); recorded as a
  packet caveat, not a regression signal.

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
