# MX-1 F16 down-only layer2 gate: blocked by a runtime dispatch precondition

This packet records the first qualification attempt for the MX-1 **down-only**
candidate against the canonical layer2 numerical/state/task gate. The attempt
did **not** produce a numerical, state, or task verdict: the candidate leg
aborts on its first call, before any arithmetic, because the F16 down owner is
handed a zero host tile-row count on the production gate/up path. The result is
a concrete dispatch blocker, not a pass and not a rejection.

## What was being qualified

The down-only MX-1 configuration keeps the Q4_K gate/up leg on the production
exact iu8-risk+repair route and swaps only the Q5_1 down leg to the F16-WMMA
compact grouped owner with no repair pass:

```
HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN=0
HIPENGINE_QWEN4_EXP_Q5_1_WMMA=1
```

This is the only MX-1 configuration with a favorable main-kernel measurement
whose numerical status is unmeasured: the F16 down owner is the fastest
registered down owner, -42.29% median against the production iu8-risk down owner
(`benchmarks/results/2026-10-07-qwen4exp-mx1-f16-owner-recheck/`). The full F16
route and its gate/up half are already refuted on numerics, and the gate/up F16
owner is +138% slower, so the down-only variant is the remaining open question.

The named T1 candidate `f16_compact_down` was added to
`scripts/qwen4exp_layer2_profile_gate.py` and pinned by RED-first unit tests
before this run.

## Dispatch path, verified from code

`hipengine/runtime/qwen4_exp_runner.py`:

- `exact_grouped_down` (line 4153) reads `HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN`
  -> **false** with the override.
- `exact_grouped_q4_gate` (line 4160) stays **true**: its trailing clause
  `(exact_grouped_down or HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4_ALL)` keeps the
  production default `EXACT_GROUPED_Q4_ALL=1`. The gate/up leg therefore stays
  in the `elif exact_grouped_q4_gate:` production iu8-risk+repair branch.
- `q5_wmma` (line 4559) reads `HIPENGINE_QWEN4_EXP_Q5_1_WMMA` -> **true**.
- The down chain is `if q5_mmq / elif exact_grouped_down / elif q5_wmma / else`
  (lines 4560-4874); with `exact_grouped_down` false and `q5_wmma` true the
  `elif q5_wmma:` branch at line 4874 calls
  `qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out` directly.

## The blocker

The candidate engages, then the kernel rejects its arguments:

```
ValueError: compact_rows, num_experts, and wmma_total_rows must be positive
  at hipengine/kernels/hip_gfx1100/quant/qwen4_exp_q5_1.py:158
  called from hipengine/runtime/qwen4_exp_runner.py:4875
```

The traceback passes through `scripts/qwen4exp_candidate_dispatch.py:43
counted()`, so the counted direct alias is the frame that raises: the route
**did** engage. The problem is the argument it received.

The host-side `wmma_total_rows` starts at `0`
(`qwen4_exp_runner.py:4244`) and is only read back from the device when

```
q4_k_mmq_prefill or (not exact_grouped_down and not exact_grouped_q4_gate)
```

(`qwen4_exp_runner.py:4254-4284`). With gate/up left on the production exact
route, `exact_grouped_q4_gate` is true and neither condition holds, so the
read-back is skipped and the F16 down kernel receives `wmma_total_rows = 0`.
The kernel requires it positive and divisible by 16, and rejects the call before
any arithmetic.

The full MX-1 route works only because it also clears
`HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4=0`, which makes `exact_grouped_q4_gate`
false and lets the tile map and read-back run. The down-only route deliberately
leaves gate/up at production, so it cannot take that path. The failure is
structural: it is independent of prompt count, decode steps, repeats, and free
tokens.

A targeted fix would populate the host `wmma_total_rows` for the F16 down owner
on the production exact gate/up path (that branch already builds the device tile
map at `qwen4_exp_runner.py:4340`), or have the `q5_wmma` branch build/read its
own tile map before the call. Editing the runtime is outside this unit's scope,
so the gap is recorded as a blocker for parent coordination rather than fixed
here.

## Command and evidence

Reproduction (raw log kept outside the tracked tree at
`/tmp/mx1-down-gate/smoke.log`):

```
PY=/home/lhl/miniforge3/envs/therock/bin/python
ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib" \
PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH" \
PYTHONPATH=/home/lhl/hipEngine \
HIPENGINE_HIP_ARCH=gfx1151 \
$PY scripts/qwen4exp_layer2_profile_gate.py \
  --candidate f16_compact_down \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --limit 1 --decode-steps 2 --free-tokens 4 --free-runs 2 \
  --output /tmp/mx1-down-gate/smoke.json
```

The strict trajectory for the single prompt completed; the candidate trajectory
aborted on its first call. The complete default 18-prompt protocol was not run
because it would reproduce the same deterministic precondition failure after the
strict pass, adding cost without new information. The exact canonical command
that remains to be run once the blocker is resolved is recorded in
`blocker.json`.

- Tree: `0c0bcb6e8` (`test: add named layer2 gate candidate for MX-1 F16
  down-only route`), tracked-clean at run time.
- Host: `gfx1151`, machine id `55ea6c509d0b49eea8de7094a1023668`, AMD RYZEN AI
  MAX+ 395 / Radeon 8060S (Strix Halo).
- ROCm: TheRock `10.0.0`, Python 3.12.
- Model: unsloth `Qwen3.8-Flash-Next` `UD-Q4_K_XL`, BF16 KV, chunk 512.

## Limits

- No numerical, state, task, or lifecycle verdict exists for this candidate; the
  run aborted before the first candidate trajectory completed.
- `performance_claim: false`.
- One host, one model, one quant, one profile.
