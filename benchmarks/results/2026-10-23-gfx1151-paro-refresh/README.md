# gfx1151 packed PARO refresh — 2026-10-23

The published row is the first eager sweep in `summary.json`. Its
`confirmation_components` record a second invocation of each command, not a
selection of the faster run. Both sweeps use two discarded warmups and five
samples at each shape. The confirmation medians differ by at most 0.98% for
prefill and 0.19% for decode.

## Evidence

- `paro-eager-*.json`: all per-run timing samples, output sanity, memory
  summaries, model fingerprints, compiler versions and commands. Files ending
  in `-repeat.json` are the confirmation sweep. Verbose allocation snapshots
  are omitted; no timing samples were removed.
- `validation.json`: serial/native numerical checks and the rejected graph
  probe. Its overall `passed: false` includes the graph failure; the ten
  serial/native cases all pass. No graph throughput is published.
- `public.json`: repeated `LLM.generate_detailed()` requests, agreement with
  `LLM.generate()`, route telemetry and the explicit MTP failure.
- `validate.py`, `environment.sh`, `assemble.py`: exact diagnostic and
  assembly sources used for this packet. They are historical drivers, not
  installed commands. The assembler checks the shipping eager path separately
  from the rejected graph probe.
- `hipcc-version.txt`: compiler identity captured before timed execution.

No kernel or runtime change accompanies the measurements. No gfx1151 PARO
execution profile resolves; the run exercises the shipped default kernels,
not a newly promoted arithmetic profile. Numerical agreement is against the
same checkpoint's serial runtime, not a fresh BF16 quality evaluation. Public
MTP fails with `PARO MTP2 prompt capture has no staged adapter`; implementing
and validating that adapter is necessary before a public MTP benchmark can run.

## Reproduction

Use the ROCm 10 environment in `docs/reference/THEROCK.md`, with the packed
model snapshot `437eba06df05aad71a4dacdcaf3fff70ae1ee8a1`. The measurement
checkout is a clean detached worktree at
`3d246c549b303a74ab659e3a7076a360916a6c85` under
`/tmp/hipengine-paro-gfx1151-20261023`. The archived environment script binds
that checkout through `PYTHONPATH` and sets `GPU_MAX_HW_QUEUES=2`.

Each component's `provenance.command` is the complete measured argument list.
Run it from that checkout through `bash environment.sh` and the TheRock
Python interpreter. Build the kernels once without `--require-cached-build`
before repeating the recorded cached-only command. Discard the build run.
The exact local paths are recorded as evidence; substitute the local model
snapshot and interpreter when reproducing on another machine.

The diagnostic driver expects the compiler identity at
`/tmp/hipengine-paro-refresh-20261023/hipcc-version.txt`. Its `--mode numerical`
run returns nonzero because of the graph probe. Its `--mode public` run checks
plain generation and records, rather than hides, the explicit MTP error.
Both exact invocations are in `summary.json` under `reproduction`.

The assembler reads the original run directory and writes its `packet/`
subdirectory. To reassemble from archived files, copy `validation.json` back
as `numerical.json`; other input basenames are unchanged. The original
component hashes identify the untrimmed local files, not the compact copies.
