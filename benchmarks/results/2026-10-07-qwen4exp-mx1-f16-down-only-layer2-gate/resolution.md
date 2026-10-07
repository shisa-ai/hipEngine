# Resolution: the F16 Q5_1 down tile-map prerequisite is repaired

This appends to the blocker packet. The blocker recorded in `blocker.json` and
`README.md` (the F16 Q5_1 down owner receiving a stale `wmma_total_rows = 0` on
the production exact gate/up path) is repaired.

## Fix

Commit `9b1ef9090` (`fix: establish F16 Q5_1 down 16-row tile map before the
owner call`) adds `_run_qwen4_exp_q51_wmma_down` in
`hipengine/runtime/qwen4_exp_runner.py`. The `elif q5_wmma:` down branch now
rebuilds the 16-row WMMA tile map from the current invocation's compact expert
starts, synchronizes the stream, reads the row count back, validates it
(`0 < rows <= tile_capacity * 16` and `rows % 16 == 0`), and only then calls the
F16 down owner. This is a control/ownership fix: no kernel body, no quant
branch, no threshold, no new flag, and no default-owner change. The registered
strict fallback (`selected_wmma_iu8_risk_prefill_bf16_bf16_out`) is unchanged.

Because the branch rebuilds unconditionally, it covers the production exact
gate/up route with `Q4_IU8_EXACT` enabled and disabled, the full F16
composition, and a preceding `q4_k_mmq_prefill` 32-row map. Only the F16 down
route pays the extra map launch and D2H readback; no default production or
strict path reaches the branch.

## Evidence

- `tests/test_unit_qwen4exp_q51_wmma_down_tile_map.py`: 5 mocked CPU recording
  tests (map -> sync -> readback -> owner ordering, owner count argument,
  invalid-count rejection without an owner call, no sync on the default stream).
- `tests/test_gpu_qwen4exp_q51_wmma_down_tile_map_smoke.py`: HIP-guarded live
  smoke on the real UD-Q4_K_XL model, 1 passed (52.7 s). The former
  `ValueError` did not occur; the F16 down owner engaged with positive
  multiple-of-16 counts, and the production exact iu8-risk gate/up owner plus
  its sparse repair both engaged.
- Worklog entry `20261007T033022.882747Z-lhl-qwen4exp-q51-wmma-down-tile-map-prerequisite-894d2f`.

## Not yet done

The complete 18-prompt layer2 qualification of `f16_compact_down` has not been
run; the parent verifies this prerequisite first. The added per-invocation map
launch and D2H readback must be charged in any future full-operation performance
comparison for this candidate. No promotion, no default change.
