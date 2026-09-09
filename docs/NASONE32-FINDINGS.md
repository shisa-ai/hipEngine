# nasone32 llama.cpp fork findings for upstream reporting

**Reviewed:** 2026-09-08 (comparison), 2026-09-09 (follow-up screens); doc
opened 2026-09-09.

**Scope:** `nasone32/llama.cpp-RDNA3-7900xtx-opt` at
`7dc2f0cb28326816f67f6b979008383344e2038b`, measured on RX 7900 XTX
(`gfx1100`), HIP 7.2.53211 / AMD Clang 22, Release build, HIP graphs on,
unroll threshold 600 — full build and server commands plus per-run output ID
rows are recorded in
[`benchmarks/results/2026-09-08-rx7900xtx-engine-comparison.json`](../benchmarks/results/2026-09-08-rx7900xtx-engine-comparison.json)
and the [report](../benchmarks/results/2026-09-08-rx7900xtx-engine-comparison.md).
Model: Qwen3.8-27B Q4_K_M GGUF (sha256 `7b2aec3b…`), greedy sampling, one
slot, prompt caching off.

Two issues below are reportable upstream. Each draft is ready to file;
evidence fields cite our committed artifacts so the claim is reproducible
without trusting this doc.

## Issue 1 (draft): chunked GDN default is non-deterministic under greedy sampling and breaks draft/target agreement

**Summary.** With the fork's default configuration (chunked GDN enabled),
greedy autoregressive generation is not repeat-stable on one prompt of our
ten-prompt suite, and the MTP speculative output disagrees with the same
server's own AR output on that prompt. Setting `GGML_CUDA_GDN_CHUNKED=0`
(sequential GDN) restores repeat stability and full AR/MTP agreement, so the
instability is specific to the chunked GDN path (donor commit `4169fbbf5`,
`gated_delta_net_chunked*.cu`).

**Reproduction.** Build as recorded above; run `llama-server`:

```
llama-server -m Qwen3.8-27B-Q4_K_M.gguf -ngl 99 -fa on \
  -ctk bf16 -ctv bf16 -c 1024 -np 1 -b 4096 -ub 1024 \
  --no-cache-prompt --fit off \
  --spec-type draft-mtp --spec-draft-n-max 3
```

Send the same greedy request three times for the `general_ja_explain`
prompt (43 prompt tokens, 25 outputs; fixture
`benchmarks/prompts/mtpbench-code-general-ja.jsonl`, sha256 `fac920be…`,
committed in the hipEngine repository).

**Observed (default, 3 repetitions).**

- AR output IDs for `general_ja_explain` differ across repetitions: the
  run-0 ID row hash differs from runs 1 and 2 (`repeat_stable: false` in
  the artifact's `mtp_summary.nasone32-short.ar`).
- MTP output is repeat-stable but disagrees with the same server's AR
  output on that prompt in all three repetitions; the suite total is
  27/30 exact AR matches (`mtp_summary.nasone32-short.exact_matches`).
- Every other prompt in the suite is exact and repeat-stable.

**Observed (control, same server command, only the env var changed).**
`GGML_CUDA_GDN_CHUNKED=0`: `repeat_stable: true` for AR and MTP, 30/30
exact AR matches, no non-exact checks
(`mtp_summary.nasone32-sequential-short`).

**Suggested investigation.** Audit the chunk-64 Gram/triangular work and
state scan in `gated_delta_net_chunked*.cu` for a non-deterministic
reduction order or race (the gfx11 BF16 WMMA path is the likely suspect on
this target). Either fix the determinism or document that greedy results
may vary run to run with chunking enabled.

## Issue 2 (draft): adaptive speculative depth is silently disabled by default and the README example cannot adapt

**Summary.** `common/common.h:329` initializes `n_min_adaptive` to 3, and
the README's adaptive example also sets the maximum to 3. The documented
example therefore runs a fixed depth of 3 and cannot demonstrate
adaptation; any user who copies it is silently running non-adaptive
speculation. We confirmed the effect externally: the fork's "adaptive"
survey rows were aggregate-identical to fixed B3 in proposal and acceptance
counts (equal `accepted`/`drafted` totals), consistent with the floor
pinning the depth.

**Suggested fix.** Either lower the default floor (for example to 1) after
measuring, or change the README example to set
`--spec-draft-n-min-adaptive 1` (the explicit flag exists) so the example
actually adapts, and state the default floor in the flag's help text.

**Measured feedback for that decision (not a bug).** With an explicit
floor of 1, adaptation engages, but it was slower than fixed B3 on both of
our tested horizons (25-output short suite and 129-output long suite)
despite higher acceptance — see the `nasone32_adaptive_screen` block of
the comparison artifact. Worth knowing before making adaptive the default.

## Not reported (checked and excluded)

- **VRAM growth across a survey run** (1.64 → 4.48 GiB in one recorded
  attempt) was our own measurement contamination: the card started busy
  from an earlier attempt, which is why the external benchmark driver now
  has idle admission. Not a fork defect.
- **A mid-context crash at 8K/16K prompts** that we fixed in the same
  window was in our own runner (1024-row dense scratch cap vs 4096-row
  chunks), not in the fork.
- The fork's projection-kernel ideas (k-quant load reuse, raw-Q4_K/Q8_1
  integer MMQ) are performance-rejected on our stack by the 2026-09-09
  screens, but they are design choices, not defects, and out of scope for
  an issue report.
