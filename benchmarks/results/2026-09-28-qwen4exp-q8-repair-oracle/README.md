# Focused Q8_0 repair coverage

The existing Q8_0 blockscale tests cover the sparse repair kernel changed by
`7804517a2`. On September 28, 2026, **17 tests passed** on Framework machine
`55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S/gfx1151, TheRock 10 /
HIP 7.15.26333. This supplies the focused evidence missing from the earlier
Q4_K/Q5_1/Q5_K bundle; it is not full-profile or task qualification.

## Coverage

- `tests/test_gpu_qwen4exp_q8_blockscale.py`: 16 cases, guarded/unguarded,
  positive basis, signed basis, random and cancellation, K64/N33 and K640/N320.
  Compact expert counts `[1, 0, 17, 33]` cover an empty expert and ragged tiles.
  Each case repeats three times. Guarded results match the strict grouped GEMV
  parent exactly; non-cancellation cases also use an independent raw-weight CPU
  oracle and the outer KL/top-1 floor. Cancellation compares against the
  independent sum reference and asserts full queue membership without duplicates.
- `tests/test_gpu_qwen4exp_q8_blockscale_edges.py`: one nonfinite-risk estimate
  case requires every output queued and verifies the finite expected output.
- The profiled K640/N320 guarded cancellation case asserts all 51×320=16,320
  outputs queued on each repetition. The trace shows three
  `q8_0_selected_sparse_repair_kernel` dispatches, each 4096 blocks ×128 threads,
  with durations 54,903 / 54,863 / 54,662 ns. These timings prove engagement;
  they are not a performance comparison.

The all-queued cancellation fixture produces zero outputs, so it alone is not
proof of nonzero-value repair arithmetic. It is combined with the random,
basis and finite-result edge checks, not offered as a universal kernel proof.
These synthetic tests do not certify long natural tasks, heldouts, new chunk
sizes or the final composed stack. No numerical variant or default changed.

[Artifact](artifact.json) contains source hashes, host identity, profiler rows,
raw evidence hashes and explicit scope limits.

## Commands

Use the environment exports from the
[scratch-growth packet](../2026-09-28-qwen4exp-scratch-growth/README.md), including
`HIPENGINE_REQUIRE_CACHED_BUILD=1` and the precomputed compiler-version file.

```bash
"$ENV_PREFIX/bin/python" -m pytest -q \
  tests/test_gpu_qwen4exp_q8_blockscale.py \
  tests/test_gpu_qwen4exp_q8_blockscale_edges.py
rocprofv3 --kernel-trace --output-format csv \
  --output-directory /tmp/qwen4exp-q8-repair-oracle-trace -- \
  "$ENV_PREFIX/bin/python" -m pytest -q \
  'tests/test_gpu_qwen4exp_q8_blockscale.py::test_blockscale_raw_oracle_and_ragged_tiles[640-320-cancellation-True]'
```
