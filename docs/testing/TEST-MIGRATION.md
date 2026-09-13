# Test Tier Migration

Date: 2026-09-12.

The naming pass covers all 1,230 test modules. Pytest discovery uses
`test_<tier>_<subject>.py`, not the audit inventory or a list of exceptions.
No correctness assertions or numerical matrix points were deleted.

| Tier | Modules | Default discovery |
| --- | ---: | --- |
| unit | 794 | Yes |
| integration | 57 | No |
| gpu | 305 | No |
| live | 72 | No |
| benchmark | 2 | No |
| slow | 0 | No |

The counts cover the modules the migration classified plus the one module
re-tiered by the follow-up repairs below. Modules added afterwards are not
counted here; `docs/testing/test-tier-migration-2026-09-12.json` carries the
same figures and is a migration record, not a live inventory.

`uv run pytest` runs the CPU unit tier. Use `--suite integration`, `--suite gpu`,
or another tier explicitly. A file/node argument selects that target directly.
`--suite all` is reserved for an explicitly requested full run or release gate.

## Validation

- Before/after full collection preserved all 13,460 node IDs after applying
  the filename mapping, with zero missing or extra cases. A subsequently
  added repository naming check and six review regressions add seven cases:
  final collection is 13,467. Default unit-only discovery selects 7,324 cases.
- The guarded CPU pass finished in 17.90 seconds: 7,322 passed, 15 skipped,
  22 failed. This was a unit-tier run, not a full-suite run or a speedup claim
  against equivalent coverage.
- Failure analysis identified missed subprocess/socket dependencies and
  incomplete isolation. Six modules moved to integration; one with an
  incomplete mock moved to GPU pending repair.
- Repaired CPU files plus the naming checker: 127 passed in 2.93 seconds.
  Fake-tool integration and discovery checks: 18 passed in 2.04 seconds.
  The socket test passed separately once sandbox permissions allowed sockets.
- Read-only review found nested-file and relative-import enforcement gaps.
  Both were reproduced RED and repaired; naming/discovery checks pass all
  16 cases, including nested absolute and relative imports.
- No GPU/live/benchmark tests were executed during the naming pass. Full
  collection is not execution.
- The follow-up repairs below resolved every recorded CPU failure. The unit tier
  now passes as a whole: `uv run pytest -p no:randomly -q` reports 7,353 passed,
  15 skipped, 0 failed. GPU-tier source-pin repairs were validated by running
  the affected modules on the W7900 (gfx1100) host: the 60 `test_gpu_laguna_*`
  modules report 475 passed, 1 skipped, 0 failed.
- The migration-time measurements above are preserved unchanged, including the
  22 failures they recorded. They describe the run that produced them, not the
  current tree.

## Resolved Isolation Problems

- `scripts/qwen4exp_gdn_owner_probe.py` now imports its GPU test helper only
  inside its GPU entry point, not when CPU geometry helpers are imported.
- Synthetic Q4_K device-weight metadata moved to
  `tests/_gguf_device_weight_fixtures.py`; a CPU resolver test no longer loads
  an entire GPU test module just to construct fake pointer metadata.
- The ownership-trace case-table test reads literal parametrization with
  Python's AST instead of importing the GPU test and probing HIP.
- The packed-state commit unit fixture supplies a fake runtime instead of
  falling through to the real HIP loader.

## Resolved Follow-Ups

### 1. Source-hash assertions that disagreed with current source

Every failing assertion was audited against the intervening commits rather
than refreshed. Most of them pinned a whole file that a later, unrelated
campaign had extended, so the repair pins the symbol the contract actually
owns.

- `_raw_k_prefill_rowbatch_dispatch` (unit `h7c`/`h7i`, GPU `h7c`) gained a
  Q8_0 rowbatch/coltile branch in `665e9a147`, scoped to gfx1151 in `053165acc`.
  Every Q5/Q6 branch is unchanged: the rowbatch capability read, the
  `in_features` multiple, the `prefill_` variant strip, the
  `coltile2_rowbatch16`/`coltile4_rowbatch8` choice, and the role-variant
  lookup. Re-baselined with that audit recorded at the constant.
- The shared gfx1100 registry package (unit `h7u`/`h8a`/`h8b`, GPU `h7u`) is no
  longer pinned by normalized whole-file hash. `tests/_laguna_policy_pin.py`
  parses every `LAGUNA_*` assignment and fails with the added, removed, or
  changed names; each test excludes the one or two flags it exists to flip.
  This freezes the policy a contract depends on while ignoring unrelated
  non-Laguna additions.
- Peer-backend pins on `hipengine/kernels/hip_gfx1151/__init__.py` are dropped.
  Those files are shared by every campaign, and each test already asserts its
  own capability-isolation contract against `hip_gfx1151` directly.
- `hipengine/runtime/gguf_linear.py` (unit `h8a`) gained 502 lines of Q8 and
  Qwen4Exp dispatch. The two symbols `h8a` added there, `Q5F32ResidentPlane`
  and `_launch_raw_k_f32_resident_activation_tile_k_row`, are byte-identical to
  the recorded revision and are now pinned by name.
- GPU `h7c` pins the generic incumbent
  `gguf_k_prefill_out_coltile_rowbatch_kernel` and its launcher. Four post-`h7c`
  campaigns added `if constexpr (qtype == 8)` branches and widened the
  accumulator `static_assert` to admit a 64-wide Q8 tile; the Q5_K and Q6_K
  instantiations keep their previous branches and tile width. Re-baselined with
  that audit recorded.
- GPU `h7e` pinned the whole `gguf_q8_0_mmq_prefill.hip` producer, which later
  gained 512 lines of unrelated f32, prepacked, and tiled MMQ kernels. The d4x2
  fallback exports are byte-identical and are now pinned by declaration.

### 2. Published-command drift gate

Resolved. The gate now reads the parser a script composes, checks every `.py`
target in a command rather than only the first, and resolves a recorded test
path through this migration's rename record. See
`worklog/entries/20260912T132634.029954Z-lhl-published-command-drift-gate-ac99d5.md`.

### 3. Incomplete context-admission fixture

Resolved. The construction-failure case in the Qwen4Exp GGUF generator test
now supplies a synthetic HIP runtime and admission record, so it no longer
reaches real context admission. All eight cases in that module are CPU-only,
so it moved back to `unit` tier as
`tests/test_unit_generation_qwen4_exp_gguf.py`; the rename record and the tier
counts above were updated with it.

### 4. Migrated test paths in frozen artifact pin lists

The GPU `h7u` and `h7y` tests resolve artifact `source_sha256` keys by
pre-migration path. Both now map those recorded paths to the current module
names. Every renamed module is otherwise byte-identical to the hash it
recorded, except `tests/test_gpu_laguna_kv_attention.py`, which `h7y` already
re-baselined for the gfx1151 handoff.

## Issues For Follow-Up

1. **Mixed-module extraction is not complete.**
   Whole modules containing device/live/integration dependencies were moved
   conservatively. Many still contain CPU cases worth extracting.
   The scan found 125 modules importing other test modules, including eight
   whose tier was raised because of those imports. The JSON record identifies
   them. Prefer shared non-test fixture helpers over importing test modules.

2. **Hardware and long-test duration attribution remains unmeasured.**
   Source screening and the short CPU run do not identify which GPU/model
   tests consumed the historical hour-long run. Keep hardware timing and
   deliberate exhaustive coverage opt-in. Do not shrink matrices, share
   mutable fixtures, or remove assertions without evidence.

3. **The GPU tier was not part of the naming pass's validation.**
   Its 305 modules were classified by source screening only. Running the 60
   `test_gpu_laguna_*` modules found five stale source pins, all repaired
   above. Other families were not run, so similar pins may remain; treat a
   first `--suite <tier>` run of any family as the way to find them.

## Audit Record

[`test-tier-migration-2026-09-12.json`](test-tier-migration-2026-09-12.json)
records every old/new path, tier, source evidence, dependency notes, and
validation status. It is a migration record, not an automatic classifier,
runtime alias map, or guarantee that all mixed modules have been split.

Future changes must follow `docs/TESTING.md` and pass
`python3 scripts/check_test_tiers.py`. Immutable worklogs and measured result
artifacts retain their historical names and commands.

The record's `validation` block is the naming pass's own measurement, left
unchanged. Read it as the state that run observed, not as the current tree.
