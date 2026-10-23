# gfx1151 PARO staged-adapter repair

This record covers the public-generation checks for enabling the existing
single-request, single-candidate PARO MTP adapter on Radeon 8060S. It is not a
throughput result or an arithmetic-profile promotion.

`validation.json` records the model fingerprint, ROCm/compiler environment,
commands, generated IDs, speculative-cycle counters, allocation cleanup, and
same-state numerical diagnostics. The worktree was dirty during validation;
its provenance lists those paths. No kernel arithmetic was changed.

## Results

- All 22 prompts produced 64 tokens. Both MTP repetitions used the staged
  adapter and produced identical IDs. `LLM.generate()` also matched the
  detailed-output API. All tracked allocations were freed after close.
- Eighteen MTP trajectories matched ordinary decoding; four did not. The
  original diagnostic's failed token-identity result remains in the artifact.
- On the four differing prompts, comparing two serial target steps with the
  bulk verifier from the same saved state gave 632 matching top-1 decisions
  across 634 rows. Mean KL was 0.00009145, p99 was 0.002298, and maximum was
  0.006950. GPU acceptance decisions matched the CPU checks. Instrumentation
  did not change the generated IDs. Rows include repeated requests and are
  not independent statistical samples.
- Switching only attention to its scalar route did not change any MTP
  trajectory. That experimental code was removed. Exact-route environment
  controls were also diagnostic only and are not part of the repair.

The numerical comparison localizes a difference in existing bulk arithmetic;
it does not establish a full strict-teacher quality gate. Cross-route token
identity and same-schedule repeatability are different contracts; see
`docs/OPTIMIZATION.md` §4.1. No MTP rate is published from this repair.

## As-run drivers

The two Python files preserve the executed drivers, including their temporary
output paths and original token-identity assertion. They therefore exit 1 when
that diagnostic differs; this is not hidden by the functional-check summary.
They require the model at the recorded path and the local ROCm Python environment.
Run from the repository root, with the GPU idle:

```bash
ART=benchmarks/results/2026-10-23-gfx1151-paro-mtp-adapter
bash "$ART/environment-as-run.sh" env PYTHONPATH="$PWD" \
  "$HOME/miniforge3/envs/therock/bin/python" "$ART/public-generation-as-run.py"
bash "$ART/environment-as-run.sh" env PYTHONPATH="$PWD" \
  HIPENGINE_VERIFY_GPU_ACCEPT=validate \
  "$HOME/miniforge3/envs/therock/bin/python" "$ART/same-state-logits-as-run.py"
```

The environment wrapper creates the compiler-version input used by the drivers.
The `PYTHONPATH` override selects the current checkout rather than the earlier
benchmark checkout named in that preserved wrapper.
