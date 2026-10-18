# MTP verify-journal repair (2026-09-11) — gfx1100 serial-row route

Native GGUF MTP had been failing on gfx1100 for any request whose decode
crossed target position 95. The repair is commit `4b8575a5d`; this
directory holds the retained re-measurement and the pre-fix evidence.

## What was wrong

`_StateJournal` is allocated once per `Qwen35GGUFTransactionalVerifier`,
but its mode was decided by `_initial_state_only_journal_applies`, which
passed only the row count to `_effective_target_verify_mode` and never the
`backend`/`end_position` clause that `prepare()` applies at verify time.
Past the gfx1100 native target context limit
(`GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT = 95`) the route becomes
`serial_exact` and calls `capture_row`, which an initial-state-only
journal refuses by design:

```
RuntimeError: initial-state-only journal cannot capture serial rows
    hipengine/runtime/qwen35_gguf_mtp.py:733
```

The journal now bounds the whole reachable position range, producer
capture is paired with the initial-state-only journal only (it lends a
single rollback row), and `prepare()` fails closed before
`capture_initial` if the route ever needs rows the journal cannot serve.
See the worklog entry
[`20260911T140433.994834Z-lhl-ud-mtp-journal-repair-daa49c.md`](../../../worklog/entries/20260911T140433.994834Z-lhl-ud-mtp-journal-repair-daa49c.md).

## Why only the tenth prompt failed

The suite's own prompts decide it. Chat-template prompt lengths (reasoning
off) and the last verify cycle's end position (`root_position +
candidate_budget + 1`) at 25 visible outputs:

| prompt | tokens | last root | last end | route |
| --- | ---: | ---: | ---: | --- |
| code_merge_intervals | 40 | 63 | 67 | native |
| code_topological_sort | 50 | 70 | 74 | native |
| code_lru_cache | 47 | 68 | 72 | native |
| code_markdown_table | 52 | 75 | 79 | native |
| general_en_plan | 39 | 59 | 63 | native |
| general_en_explain | 40 | 63 | 67 | native |
| general_ja_plan | 43 | 64 | 68 | native |
| general_ja_explain | 43 | 66 | 70 | native |
| mixed_ja_en_translate | 64 | 86 | 90 | native |
| mixed_ja_en_review | 71 | 94 | **98** | **native then serial_exact** |

`mixed_ja_en_review` is the only prompt that crosses 95, so it is the only
one that reaches `capture_row`. The repaired run records
`target_verify_mode` per cycle: that prompt's cycles at root positions 93
and 94 run `serial_exact`, every other cycle of every prompt runs
`native`. Those two cycles are the first successful end-to-end execution
of the gfx1100 serial row route.

## Retained measurement

`natural25-b3-xtx-runs2.json` — plain `Qwen3.8-27B-Q4_K_M`, XTX
(gfx1100), commit `4b8575a5d`, `--candidate-budgets 3 --runs 2`, 25
visible outputs, true no-MTP AR denominator from the same run:

```
HIP_VISIBLE_DEVICES=1 .venv/bin/python scripts/qwen36_dense_gguf_suite.py \
  --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf --quant gguf_q4_k_m \
  --candidate-budgets 3 --runs 2 --output natural25-b3-xtx-runs2.json
```

| scope | true AR tok/s | MTP B3 tok/s | ratio |
| --- | ---: | ---: | ---: |
| full (10 prompts) | 36.782 | **62.845** | **1.7086x** |
| train (6) | 36.727 | 64.089 | 1.7450x |
| heldout (4) | 36.865 | 61.067 | 1.6565x |

Categories: code 2.1470x, general_en 1.4923x, general_ja 1.4753x,
mixed_ja_en 1.5491x. `accepted_per_output` 0.608. `status:
complete_exact`, `all_exact_greedy: true`, `all_gpu_accept_match_cpu:
true`, `active_allocations: 0` after close, peak 19.6 GB.

Determinism: the two runs produce identical token IDs for all twenty
prompt/arm pairs, and MTP equals true AR in both runs. Per-run means were
AR 36.676 / 36.897 and MTP B3 63.945 / 66.155 tok/s.

### Provenance note

`provenance.dirty` is true because the shared worktree carried a
concurrent unit's unstaged *selected-expert* (MoE) C1 table entry
(`GGUF_T16_SELECTED_C1_VARIANTS_BY_QUANT_SHAPE`, Qwen3.6-35B shapes).
Nothing was staged. The measured model has 866 tensors and zero
expert/selected tensors, so that table is never resolved on this path.
The comparison basis for the retained row is the 2026-08-15 XTX snapshot
of the same protocol (AR 35.287 / MTP 62.440 / 1.7695x); the AR
denominator here is the same-host same-run no-MTP arm, not that snapshot.

## Secondary evidence

- `w7900-correctness-run.json` — the same command on the second gfx1100
  card (W7900), pre-commit tree: `complete_exact`, 10/10 prompts exact,
  GPU/CPU acceptance agreement, ratio 1.5397x. Its absolute rates were
  measured while the full pytest suite ran concurrently on the other card,
  so they are **not** published; it confirms the fix on the second device.
- `pre-fix-repro.txt` — the pre-fix crash: nine prompts complete, the
  tenth dies at `capture_row`.
- `ud-admission-refusal.txt` — UD MTP stays refused by admission (both UD
  presets are pinned `("ar",)`); the repair does not change that.
- `journal-mem-probe.py` — the footprint probe. On gfx1100 the row-capable
  journal costs 5 x 151.50 MiB instead of 1 x 151.50 MiB, i.e. **+606
  MiB** for budget 3; gfx1151 keeps the initial-state-only journal.
  Tracked for recovery in [`docs/REFACTOR.md`](../../../docs/REFACTOR.md).
