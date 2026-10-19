# Qwen3.8 Flash-Next gfx1151 topline refresh — October 19, 2026

Qwen3.8-Flash-Next `UD-Q4_K_XL`, BF16 KV, production profile, Framework
Desktop / Radeon 8060S (`gfx1151`), machine
`55ea6c509d0b49eea8de7094a1023668`, ROCm 10.0.0. Source:
`e29d34459c2570217576909552fa95ad1f7cc227`, clean detached worktree. No code,
arithmetic, profile, or default changed for this refresh.

## Results

The canonical exact-token fixture covers four categories at three lengths,
with one discarded warmup and three measured repetitions per case. Rates are
weighted by phase wall time. Prefill includes the first greedy output;
decode counts 128 subsequent transitions. Kernels were prebuilt across all
12 cases before the cached-only run; PLE was warm, chunk size 4096, compact
capture selected the production greedy path.

| Prompt tokens | Prefill tok/s | Decode tok/s |
| ---: | ---: | ---: |
| 512 | 1036.217 | 23.677 |
| 1024 | 1323.865 | 22.716 |
| 4096 | 1650.651 | 23.244 |

MTP uses the Q8_0 sidecar, budget three, 64 greedy output tokens, and a true
no-MTP AR arm in the same process. Timing is whole-request wall, not the
canonical decode-only window. Each prompt has one warmup per arm.

| Prompt set | AR tok/s | MTP tok/s | MTP / AR | Draft acceptance |
| --- | ---: | ---: | ---: | ---: |
| Full ten-prompt suite, run 1 | 22.670 | 28.930 | 1.2762x | 56.06% |
| Full ten-prompt suite, run 2 | 22.671 | 28.928 | 1.2760x | 56.06% |
| Eight category heldouts | 22.569 | 28.164 | 1.2479x | 53.46% |

The README uses the 512-token canonical plain rates and the median of the two
full-suite MTP rates, rounded to one decimal. Its plain and MTP cells have
different workload/timing windows; the MTP speed ratio is against the matched
AR arm here, not against the README's plain cell. These are single-request
rates, not HTTP throughput or concurrent scaling.

The canonical peak owned allocation was 82.492 GiB; teardown left zero active
allocations. Compared with the October 14 canonical artifact on this same
physical host, prefill changed −0.03% / +0.00% / +0.27% and decode +0.09% /
+0.09% / +0.08% at 512 / 1024 / 4096. Every generated-ID hash is unchanged.
The October 18 MTP artifacts ran on zbook machine
`87c566d30a5645cf8d12ed7ef6b6e1e8`, not this Framework machine. Their rates
are a separate host lane, not an old-to-new comparison.

## Correctness

- All 12 canonical cases are deterministic across the three measured runs.
- Fresh production-versus-strict gate: 18 prompts, 450 teacher-forced rows,
  three candidate repeats. Mean/p95/p99/max KL:
  `6.899e-4 / 4.361e-3 / 8.414e-3 / 1.463e-2`; top-1 `446/450`. Numerical,
  state-repeat, and teardown checks pass.
- The raw gate exits 2 (`requires_task_review`), not 0: 13 of 18 free-run
  outputs are strict-exact. All 18 strict hashes and all 18 production hashes
  exactly match the [October 14 gate](../2026-10-14-qwen4exp-decode-batch/gate-c.json).
  Its semantic review was resolved in
  [production-serving PS12](../../../docs/QWEN4EXP-PRODUCTION-SERVING.md#implementation-and-qualification-checklist),
  with 18 task passes. This refresh carries that review over by exact output
  identity; it does not relabel the raw gate as passed or require production
  to generate strict-identical text.
- Public `LLM.generate()` and `generate_detailed()` succeed on plain and
  explicit MTP routes at 4096 context. Each returns 64 tokens; repeated text
  agrees; plain IDs match the canonical prefix. Backend/quant resolve to
  `hip_gfx1151` / `gguf_ud_q4_k_xl`; the manifest matches the benchmark. MTP
  reports `qwen4exp_mtp_batched_state_rows_verify`, not an AR fallback.
- MTP acceptance and cycle counts match prompt-for-prompt across both suite
  runs. Eight of ten MTP outputs are AR-identical; production generated-ID
  equality across those different execution shapes is diagnostic.

## Reproduction

Use the clean-process environment in `docs/reference/THEROCK.md`. Resolve
`PY`, SDK paths and library paths through `~/miniforge3/envs/therock`; do not
use the base Python. Set `HIPENGINE_HIP_ARCH=gfx1151`,
`GPU_MAX_HW_QUEUES=2`, and `HIPENGINE_COMPILER_VERSION_FILE` to a file captured
from that SDK's `hipcc --version` before launching. Run at the pinned source
commit in a clean worktree. The JSON artifacts record exact paths, commands,
model/sidecar content fingerprints, compiler identity and environment.

```bash
MODEL=~/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
SIDECAR=~/models/gguf/Qwen3.8-Flash-Next-MTP-Q8_0/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
OUT=/tmp/flashnext-topline
mkdir -p "$OUT"

# Warm every shape before requiring cached builds.
"$PY" scripts/qwen4exp_canonical_ar_bench.py hipengine \
  --model-root "$MODEL" --output "$OUT/prebuild.json" \
  --prefill-chunk-size 4096 --capture-mode compact --ple-cache-mode warm \
  --execution-profile production --warmups 0 --repetitions 1 \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE"

"$PY" scripts/qwen4exp_canonical_ar_bench.py hipengine \
  --model-root "$MODEL" --output "$OUT/canonical-ar.json" \
  --prefill-chunk-size 4096 --capture-mode compact --ple-cache-mode warm \
  --execution-profile production --warmups 1 --repetitions 3 \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" --require-cached-build

for REP in 1 2; do
  "$PY" scripts/qwen4exp_mtp_inprocess_bench.py \
    --model-root "$MODEL" --sidecar "$SIDECAR" \
    --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
    --max-tokens 64 --budget 3 --execution-profile production \
    --output "$OUT/mtp-suite-$REP.json"
done
"$PY" scripts/qwen4exp_mtp_inprocess_bench.py \
  --model-root "$MODEL" --sidecar "$SIDECAR" \
  --prompts benchmarks/prompts/gdn-prefill-category-heldouts.jsonl \
  --max-tokens 64 --budget 3 --execution-profile production \
  --output "$OUT/mtp-heldouts.json"

"$PY" scripts/qwen4exp_layer2_profile_gate.py \
  --model-root "$MODEL" --candidate production_baseline \
  --prefill-chunk-size 4096 \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
  --output "$OUT/production-gate.json"
```

## Artifacts

- [Summary, provenance, commands and public API probe](summary.json)
- [Canonical plain samples](canonical-ar.json)
- [MTP suite run 1](mtp-suite-1.json) · [run 2](mtp-suite-2.json) · [heldouts](mtp-heldouts.json)
- [Raw production gate](production-gate.json) · [public API confirmation](public-api.json)

The first cached-only attempt failed before measurement because a short-context
attention kernel was not prebuilt. The first public API probe at 1024 context
failed admission because the shared index requires at least 2048 positions;
the accepted probe used 4096. Neither failed attempt contributes a timing row.
No new profiler trace, long-context capacity test, or full release-suite run
was performed for this documentation-only refresh.
