# Qwen3.8 dense mixed-pair batch diagnostic

## Change and contract

The native verifier executes mixed IQ4_XS/Q4_K-T16 gate/up plus SiLU in one
launch instead of one launch per row. The second grid dimension selects the
activation/output row; weights, per-row reductions, projection BF16 rounding
and the SiLU expression are unchanged. Both operand orders use this path.
No new runtime flag or model-identity gate is added.

## Provenance

- Physical host: `gfx1151`, AMD Ryzen AI MAX+ 395 / Radeon 8060S, Linux
  `7.3.1-1-cachyos`; this is not the separate Desktop or W7900 lane.
- Model: `/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf`; sampled fingerprint and
  byte size are in `summary.json`. The kernel microbenchmark uses fixture-derived
  IQ4_XS weights and synthetic Q4_K weights, not a full-model measurement.
- HIP `7.15.26333`, AMD clang `23.0.0git`, LLVM revision
  `8f497e0992fb7513f7f78a6f6b6f1056c375e961`.
- Public generation: default `production` profile, BF16 KV, greedy,
  prefix cache off, MTP candidate budget 3, capacity 2048, one request.
- Before-change source: `ff1c5ed41`. Public profiler and microbenchmark
  candidate source hashes are captured in `summary.json`.

## Commands and archived drivers

Run from the repository root. `run.sh` selects the local TheRock environment
at `~/miniforge3/envs/therock`; adjust that environment location for another host.
The files here preserve the diagnostic drivers executed from `/tmp`.

```bash
D=benchmarks/results/2026-11-12-gfx1151-qwen38-mixed-pair-batch
bash "$D/run.sh" "$D/microbench.py" --batch --out /tmp/qwen38-pair-batch-ab.json
bash "$D/run.sh" "$D/public_ab.py" --out /tmp/qwen38-mixed-pair-public-ab.json
bash "$D/run.sh" "$D/public_profile.py" --out /tmp/qwen38-public-profile-candidate-warm.json
bash "$D/run.sh" -c 'import subprocess; from pathlib import Path; Path("/tmp/qwen38-hipcc-version.txt").write_text(subprocess.check_output(["hipcc", "--version"], text=True))'
bash "$D/run.sh" "$D/profile.py" /tmp/qwen38-public-candidate-profile "$D/public_profile.py" --profile --out /tmp/qwen38-public-profile-candidate.json
```

The microbenchmark alternates loop/batch ordering across ten samples, each
containing ten operations after three warmups. Timing uses HIP events, with
synchronized host wall also saved. Every sample checks output equality.

The public-library diagnostic uses fresh model sessions in AR, loop, batch,
batch, loop order over the canonical ten prompts plus eight category-heldouts.
Each session discards a separate warmup request. Only the loop arm replaces
the pair executor, reconstructing the previous row loop; the batch arm calls
the shipping implementation unchanged. Fresh sessions prevent reuse of a graph
captured under the other arm. Public elapsed time includes prompt processing;
no result here is a decode-only rate.

The profiler warms outside `rocprofv3`, then requires cached builds and records
only a warmed public request (15 input tokens, 32 outputs). The before/after
request uses identical output IDs and speculative control. Profile durations
are attribution diagnostics, not standalone throughput evidence. Raw profiler
databases and logs stay outside Git; the extracted kernel/grid tables are
included here.

## Validation

```bash
bash "$D/run.sh" -m pytest tests/test_gpu_gguf_mixed_pair_batch.py tests/test_gpu_gguf_iq4_q4_pair.py -q
bash "$D/run.sh" -m pytest tests/test_live_qwen35_gguf_mtp_e2e.py::test_dense_q4_k_m_mtp_crosses_1024_via_prelaunch_eager_fallback -q
uv run pytest tests/test_unit_gguf_linear_pair_silu_plan.py tests/test_unit_gguf_linear_dispatch.py tests/test_unit_qwen35_gguf_mtp_session.py tests/test_unit_refactor_ledger_gate.py -q
python3 scripts/check_fixtures.py
python3 scripts/docs/check_all.py
python3 audit/audit.py check
```

The GPU matrix covers both orders, rows 1/2/3/4/7/8/13 and split-K wave counts
1/2/4. Repeated launches match independent scalar rows bit-for-bit. A separate
CPU-reference safety check requires maximum KL <= 0.05 and top-1 agreement
>= 90%; it is not substituted for scalar-parent equality. Existing unfused
projection/SiLU tests also pass. The live crossing includes reset and
fresh-process health. The completed default CPU tier had only a resolved-ledger
format failure, fixed separately in `0e08813cb`; its focused rerun passes.

## Results

- Mean public suite wall: row loop 63.339 s, batch 62.701 s (1.01% lower).
  Pooled whole-call throughput: 18.188 versus 18.373 tok/s (1.02% higher).
  All four categories improve; every loop/batch output and selected control
  comparison passes, including repetitions. True AR agrees on 17 of 18 prompts;
  the exception is the pre-existing `general_en_explain` difference.
- Fixture microbenchmark at K5120/N17408: batch latency is 2.68–8.27% lower
  across rows 2/4/8 and both operand orders. This is kernel-only timing.
- Public profile: mixed-pair launches fall from 637 to 169, with four-row
  grids present in the candidate; summed mixed-pair duration falls from
  305.487 to 292.885 ms. Output IDs are identical and no private scratch is
  reported for these kernels.
- Tracked allocations after close are stable across all fresh sessions but
  not zero: 3,716,169,728 bytes / 100 allocations after both AR and MTP arms.
  This diagnostic does not establish complete process-wide teardown. Batching
  allocates no additional device memory and does not grow that residual.

## Limitations

The unchanged canonical server baseline already fails AR/MTP exactness on
`general_en_explain`, first at output index 51. Do not attribute that difference
to batching, call it a quality failure without its numerical evaluation, or
claim all production MTP outputs equal AR. The optimization's acceptance
comparison is batch versus the previous loop, including control and repeated
output equality. The baseline result is preserved rather than relabeled as a
passing AR-exact gate.

A precommit attempt to rerun the canonical server command was refused by its
tracked-clean-source guard. It performed no generation. The public-library A/B
is separately labeled; a clean-source server rerun belongs to the end-to-end
follow-up. No W7900 or other-host performance transfer is claimed.
