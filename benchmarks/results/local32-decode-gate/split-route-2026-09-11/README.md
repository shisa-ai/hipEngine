# Natural-prompt probes for the split/scale local32 decode admission

Run with the corrected arm protocol (candidate = shipped policy, incumbent
= shipped minus the four split quants; natural text generated under the
incumbent). These runs predate the UD-Q4_K_M strict-slot pins: on K_M the
unpinned route already passed every seed (KL mean <= 2.6e-5, top-1 100%),
and the pinned state is numerically identical to the NL-round admitted
state, for which the same probe passed. K_S matches the final shipped
state exactly.

- K_M-{7,11,23}: mean 6.9e-6..2.6e-5, max <= 7.8e-4, top-1 100%
- K_S-{7,11,23}: mean 1.8e-5..2.9e-4, max <= 8.9e-3, top-1 100%

sweeps/ holds the decode measurements (GPU1 XTX, 512/128, median of 3):
k_s.json 28.41 tok/s (26.77 before); k_m-all-routed.json 28.04 (27.67
before, gate-failing state); k_m-pinall.json 27.70 (the admitted state -
the pins return K_M to its NL-round level). The w7900-gpu0-*.json runs
document GPU0 variance (different card, not comparable).
