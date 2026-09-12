# Test Tier Migration

Date: 2026-09-12.

The naming pass covers all 1,230 test modules. Pytest discovery uses
`test_<tier>_<subject>.py`, not the audit inventory or a list of exceptions.
No correctness assertions or numerical matrix points were deleted.

| Tier | Modules | Default discovery |
| --- | ---: | --- |
| unit | 793 | Yes |
| integration | 57 | No |
| gpu | 306 | No |
| live | 72 | No |
| benchmark | 2 | No |
| slow | 0 | No |

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
- No GPU/live/benchmark tests were executed. Full collection is not execution.
- A fresh all-green CPU run is **not** claimed: the source-pin and publication
  issues below remain. Preserve the original CPU result plus focused repairs;
  do not repeat a broad run solely to re-establish already-passing cases.

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

## Issues For Follow-Up

1. **Eight source-hash assertions already disagree with current source.**
   `test_unit_laguna_h7c_source_default.py`,
   `test_unit_laguna_h7i_source_default.py`,
   `test_unit_laguna_h7u_source_default.py`,
   `test_unit_laguna_h8a_source_default.py`, and
   `test_unit_laguna_h8b_source_default.py` contain the failures.
   The pinned gfx1100 package bytes and `_raw_k_prefill_rowbatch_dispatch`
   source match the pre-migration commit. Do not refresh expected hashes
   blindly; audit the intervening changes and the intended contract.
   One historical test-path comment in that package is deliberately unchanged
   to avoid changing its evidence hash during a naming migration.

2. **Published-command drift remains a failing unit gate.**
   `test_unit_scripts_check_published_command_drift.py` reports 29 unknown
   flags in existing published commands, plus two missing-script reports
   caused by renaming the DMS test referenced in
   `2026-09-07-rx7900xtx-dms-int8-postfix-audit.json`.
   Recorded measured commands were not rewritten to pretend the renamed
   files existed at measurement time. Decide how current reproduction
   instructions should accompany historical provenance; do not add a
   filename exception or silently weaken the gate.

3. **An incomplete mock still reaches HIP.**
   `test_gpu_generation_qwen4_exp_gguf.py` was intended to use synthetic
   runners, but its construction-failure case reaches real context admission
   and `get_hip_runtime()`. Repair the mock/admission fixture and move the
   CPU tests back to `unit`; audit no-ROCm availability guards in the meantime.
   It was not run against a real GPU during this migration.

4. **Mixed-module extraction is not complete.**
   Whole modules containing device/live/integration dependencies were moved
   conservatively. Many still contain CPU cases worth extracting.
   The scan found 125 modules importing other test modules, including eight
   whose tier was raised because of those imports. The JSON record identifies
   them. Prefer shared non-test fixture helpers over importing test modules.

5. **Hardware and long-test duration attribution remains unmeasured.**
   Source screening and the short CPU run do not identify which GPU/model
   tests consumed the historical hour-long run. Keep hardware timing and
   deliberate exhaustive coverage opt-in. Do not shrink matrices, share
   mutable fixtures, or remove assertions without evidence.

## Audit Record

[`test-tier-migration-2026-09-12.json`](test-tier-migration-2026-09-12.json)
records every old/new path, tier, source evidence, dependency notes, and
validation status. It is a migration record, not an automatic classifier,
runtime alias map, or guarantee that all mixed modules have been split.

Future changes must follow `docs/TESTING.md` and pass
`python3 scripts/check_test_tiers.py`. Immutable worklogs and measured result
artifacts retain their historical names and commands.
