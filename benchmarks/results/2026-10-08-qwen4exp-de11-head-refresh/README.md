# DE-11 paired decode wall/kernel refresh on promoted HEAD — decode depth growth is host-side

Re-measured on HEAD `6243efce279e19c9c4cc9d926170a01c9ea3af03` (both exact-class
promotions live: checked full-rows dense Q8 and K320 GR-up), production
profile, chunk 1024, full capture, thermal-gated counterbalanced session
(same protocol and driver as the 2026-10-07 DE-11 packet).

## Paired wall/kernel split (code-p512 vs code-p4096, 128 decode transitions)

| Metric | p512 | p4096 | Growth |
| --- | ---: | ---: | ---: |
| Wall ms/step | 47.777 | 56.384 | +8.61 (+18.0%) |
| Kernel ms/step (rocprof sum) | 44.279 | 45.025 | +0.75 (+1.7%) |
| **Non-kernel remainder ms/step** | **3.498** | **11.360** | **+7.86 (+225%)** |
| Launches per step | 1740 | 1815 | +75 (+4.3%) |

Device kernel time is essentially flat with depth; the decode cost growth is
non-kernel. Reproduces the 2026-10-07 estimates (3.457/10.896 ms) on the
promoted HEAD.

## The compact-path host window confirms the host is the depth bottleneck

From the cooled canonical verify (compact capture, chunk 4096), the
host-enqueue window per decode step (which contains the device wait at the
per-step argmax readback, so window >= max(host, device)):

| Depth | host window ms/step | device kernel ms/step | excess over device |
| --- | ---: | ---: | ---: |
| p512 | 49.40 | 44.28 | ~5.1 |
| p1024 | 50.93 | — | — |
| p4096 | 61.63 | 45.02 | **~16.6** |

At p4096 the host can no longer keep the device fed: the step is host-bound by
~16.6 ms/step, i.e. the device idles roughly 20% of each step. Removing that
gap is worth moving p4096 decode from ~16.7 toward ~20+ tok/s — the single
biggest decode-depth lever and the reason our decode decays with context while
gufo's stays flat (25.96 tok/s at all depths).

## cProfile of the decode loop names the host cost cluster

`--host-profile` passes (compact capture; absolute times inflated by the
profiler — read ordering, shares, and call counts): per decode step the host
performs **~14,366 `os.environ` lookups** (p512) rising to ~14,846 (p4096) —
the `encode`/`get`/`__getitem__` cluster measures 4.7→6.0 ms/step — plus
**919→958 per-launch `build_hip` library re-resolutions** (the
`_resolution_environment_identity` + `build_hip` pair, 2.2→2.8 ms/step), with
`launch_gguf_linear`'s body at 4.0→5.0 ms/step and `_launch` at 6.4→6.6
ms/step. The depth-growing launch count (+75/step at p4096) multiplies exactly
these per-launch host costs, which is the mechanism behind the remainder
growth.

This is the per-launch `build_X(load=True)` debt already tracked in
`docs/REFACTOR.md` with its designed removal condition: thread the resolved
library through the launch sites so the launch path resolves once per process
rather than once per launch, and decide explicitly that the production launch
path does not need to observe mid-process build-identity environment changes
(no harness does; environment flags are bound before generator construction).
Removing the ~5-9 ms/step of per-launch re-resolution plus its share of the
environment-lookup storm is the concrete next fix.

## Artifacts

- `paired-summary.json` — joined wall/kernel/remainder split (thermal-gated, counterbalanced A/B arms per depth)
- `wall/`, `kernel/`, `wall-raw/`, `kernel-raw/` — per-arm summaries and raw bench outputs
- `host-profile-code-p512.json`, `host-profile-code-p4096.json` — cProfile top-30 entries of the decode loop
