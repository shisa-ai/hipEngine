# Multi-column GDN prefill supersedes the DPP route: **1.34x-1.68x**

The multi-column GDN prefill route (`HIPENGINE_QWEN4_GDN_MULTI_COLUMN`) is **1.34x-1.68x
faster than the DPP route it replaces**, measured on the production GDN prefill scope.

| tokens | incumbent (`tiled16_dpp`) | candidate (DPP+multi-column) | speedup |
| ---: | ---: | ---: | ---: |
| 512 | 1.864 ms | **1.112 ms** | **1.6756x** |
| 1024 | 4.556 ms | **3.118 ms** | **1.4611x** |
| 2048 | 10.583 ms | **7.706 ms** | **1.3733x** |
| 4096 | 22.799 ms | **16.997 ms** | **1.3413x** |

Host **gfx1151** (AMD Radeon 8060S, device `0x1586`), HIP 7.15.26333, quant
`gguf_ud_q4_k_xl`, geometry Hk16/Hv48/Dk=Dv=128.

```bash
python3 scripts/qwen4exp_gdn_multi_column_ab.py --tokens 1024 --reps 20 \
  --out benchmarks/results/2026-09-30-qwen4exp-gdn-multicolumn-ab/ab-1024.json
```

## What changed

The two arms are **the same kernel template with different template arguments**. Both are
built from the same source with `-DHIPENGINE_QWEN4_GDN_DPP=1`, so the only difference is
`-DHIPENGINE_QWEN4_GDN_MULTI_COLUMN=1`, which sets `warps_per_block=8, cols=8` instead of
`warps_per_block=16, cols=1`. **One k-load now feeds eight columns instead of one.** Both
configurations cover all 128 value columns exactly once with the same per-lane k-reduction.

## Correctness

**NMSE ~1.8e-14 on the core output and ~7.2e-15 on the recurrent state**, stable across all
four shapes. That is fp32 round-off — the same order the chunk-drift screen measured for a
pure reassociation (5-7e-14). The route is **not** bit-exact to the incumbent, and is
admitted on the production numerical envelope rather than on exact generated IDs.

The production gates were run against this variant previously, on this hardware class:

| gate | measured | limit |
| --- | ---: | ---: |
| short mean KL | 1.0724e-04 | — |
| short top-1 | 593/594 | — |
| depth mean KL | 8.92215e-05 | ≤1e-3 |
| depth p95 KL | 4.206676e-04 | ≤5e-3 |
| depth p99 KL | 1.2666802e-03 | ≤2e-2 |
| depth max KL | 4.4548289e-03 | ≤5e-2 |
| depth top-1 | 777/780 (99.615%) | ≥99% |

**All declared numerical limits pass**, with deterministic repeats, state-layout/metadata
checks and zero-allocation teardown.

### The task finding is recorded, not dismissed

The variant carries one task finding (`heldout_general_en_quant`). It is **not a regression
from this change**: the artifact records the defects as *"shared instruction-following
defects, not new candidate regressions"* and states *"Strict's own errors and shared
omissions remain disclosed"* — the strict arm has findings of its own. The artifact also
scopes the finding itself as *"not proof of worse expected quality or a GDN kernel defect"*.

## Promotion

- `hipengine/generation/qwen4_exp_profiles.py` — the production GDN prefill selection and
  the `HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT` binding now name
  `qwen4exp_gdn_tiled16_multi_prefill`.
- **The DPP route stays registered** as the strict fallback, so rollback is a one-line
  revert.
- **The arithmetic-recovery profile is unchanged.** Its path deactivates tiled variants
  *by name*; the multi-column route was added to that set so the recovery UD profile keeps
  its exact owners. The first attempt at this promotion missed that and broke
  `test_ud_profile_recovers_arithmetic_but_keeps_exact_owners` — the test caught it.

## Limits

- **One host (gfx1151)**, one quant, one shape family. AGENTS.md's stated default hardware
  is W7900/gfx1100; this host is gfx1151, and these figures are **not** comparable to a
  gfx1100 lane.
- **GDN prefill scope only** — not a whole-request rate.
- Not bit-exact; admitted on the envelope.
