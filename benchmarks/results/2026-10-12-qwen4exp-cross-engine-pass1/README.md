# Cross-engine pass 1: hipEngine vs halo-box vs gufo (canonical protocol)

Date: 2026-10-12. Host: AMD Radeon 8060S / gfx1151, AMD Ryzen AI Max+ 395,
128 GB unified LPDDR5X, machine `55ea6c509d0b49eea8de7094a1023668`, kernel
`7.2.9-2-cachyos` (moved from the 2026-10-04 record's `7.1.6`). Power state:
`tuned-adm` profile `accelerator-performance` with
`scaling_governor=performance`, restored at run start (the host had rebooted
out of its profile) and verified after every block. One engine resident at a
time; idle-VRAM check (148 MiB baseline) between blocks. Protocol:
[`docs/CROSS-ENGINE-PROTOCOL.md`](../../../docs/CROSS-ENGINE-PROTOCOL.md) —
committed 12-case exact-token fixture, prefill-only wall, 1 warmup + 3
measured repetitions, median per case, equal-weight mean; decode measured
separately over 128 transitions.

This is a same-file, same-host, same-protocol characterization. It is not an
old-to-new comparison against any previously published row, and no ratio here
is a causal decomposition.

## Engine identities

| Engine | Source | Binary |
| --- | --- | --- |
| hipEngine production | this tree at `c4b160668` (`hipengine/`, `kernels/` clean; only harness scripts uncommitted at run time) | in-tree JIT, cached build required |
| halo-box `strix-llama.cpp` | master `6aa81b88e` (2026-10-09) | `llama-server` `0.5.0-dev`, sha256 `ed60d2d222753f81…` |
| gufo | tag `v0.12.0` (2026-10-11), commit `88090afb` | sha256 `8bfd5076dafaec6c…`, Release/-O3, recorded libstdcxx preinclude shim |

Model for all three: unsloth `Qwen3.8-Flash-Next` `UD-Q4_K_XL`, four shards
(103.69 GiB). hipEngine ran its production profile (BF16 KV, chunk 4096,
compact capture, warm PLE); halo-box ran `-ngl 999 -fa on -c 4352 -b 8192
-ub 2048 -t 4 --parallel 1` (F16 KV); gufo ran `serve llm -c 4352
--cache-ram-bytes 1`.

**Protocol findings recorded during this run.**

- gufo v0.12.0's `/v1/completions` has no per-request `cache_prompt`, and its
  RAM continuation cache serves an identical prompt re-request from memory
  (`prefill_tokens=0`, real prefill wall zero). The canonical protocol
  requires cold prompts, so the gufo server ran with `--cache-ram-bytes 1`
  (retention budget 1 byte) — verified: identical repeated requests then show
  `cache=miss`, `cache_n=0`, with genuine prefill walls. The harness's
  `cache_n=0` guard enforces this on every measured row.
- halo-box master requires `-fa on` (the bare `-fa` is a usage error in this
  fork).
- `general_ja-p1024` is disqualified for gufo (text-driven): the fixture's
  tail ids `198,198` re-encode as merged token `271`. The comparison runs on
  the 11-case intersection; each engine's aggregate was recomputed to that
  basis where it measured extra cases.
- The hipEngine leg required one unmeasured prebuild pass first (the
  `qwen4_exp_gr` cache key changed with the K320 promotion); the measured
  block then ran with `--require-cached-build`.

## Prefill tok/s (equal-weight mean over compared cases, median of 3 reps)

| Engine | 512 | 1024 | 4096 | 4K vs hipEngine |
| --- | ---: | ---: | ---: | ---: |
| hipEngine production | 305.1 | 328.9 | 334.4 | 1.00x |
| halo-box master `6aa81b88e` | 759.3 | 1109.1 | 1446.4 | 4.33x |
| gufo `v0.12.0` | 1095.0 | 1482.2 | 1736.4 | 5.19x |

Decode over 128 transitions (ignore_eos on every engine): hipEngine
20.9 / 20.2 / 20.6 tok/s; halo-box 26.7 / 26.7 / 25.1; gufo 28.6 / 28.5 /
28.1 at 512/1K/4K. All engines deterministic across repetitions
(`all_cases_deterministic: true`).

Cross-engine greedy 129-token exact agreement is 0/11 — expected at this
output length across different arithmetic; the agreement signal lives in the
gate table below.

## Quality gates (18-prompt battery: mtpbench 10 + GDN heldouts 8)

Teacher: hipEngine **strict** profile, 32-step free greedy trajectories, EOS
fed back, 2 repeats, top-20 logprobs captured. Score:
[`quality-score.json`](quality-score.json).

| Engine | G1 greedy identity (exact / 18) | G2 aligned top-20 vs teacher | G3 determinism | G4 strict-exact tasks |
| --- | ---: | --- | --- | ---: |
| hipEngine production | **15** | top-1 1.00, mean KL 0.0052, max 0.62 | all repeats exact | 15 / 18 |
| halo-box master | 11 | top-1 1.00, mean KL 0.241, max 3.85 | all repeats exact | 11 / 18 |
| gufo `v0.12.0` | 13 | not applicable (no logprob surface) | all repeats exact | 13 / 18 |

Notes on reading this table:

- hipEngine production reproduces its documented production-envelope task
  axis exactly (15 of 18 strict-exact vs its own strict teacher), which
  validates the comparator against a known reference.
- G1 divergences carry first-divergence positions per prompt in the score
  artifact (halo-box first divergences: positions 0-19 across seven prompts;
  gufo character positions 0-82 on the decoded-text basis across five).
- G2 is measured only on aligned common-prefix positions, over the union of
  two top-20 lists renormalized in float64, with absent tokens at an epsilon
  floor. It is a truncated-distribution proxy, sensitive to tail mass
  outside the lists, and comparable only because every engine is scored
  identically against the same teacher. halo-box's top-1 agreement of 1.0
  with mean KL 0.24 quantifies tail-shape distance at matched argmax.
- gufo's G1/G4 basis is decoded text (its re-encoded ids are merge-unreliable
  at BPE boundaries); halo-box's is exact token ids.
- These are characterizations under one shared protocol. They are not
  hipEngine production-envelope verdicts for the external engines, and the
  internal calibrated KL envelope remains hipEngine-only by design.

## What this run does and does not say

- It establishes, for the first time on this host, same-file current-release
  rates for all three engines under the exact-token canonical protocol: the
  prefill gap at p4096 is 4.33x (halo-box) and 5.19x (gufo) against hipEngine
  production, with all engines' decode within 20.6-28.6 tok/s.
- halo-box master more than doubled its own 2026-09-13 pin at p4096
  (660.6 → 1446.4) — an engine-release change measured under this protocol,
  not a hipEngine change.
- It does not decompose the gap into kernel-quality versus arithmetic-policy
  components; per-operator attribution is the separate microbenchmarking leg
  of the campaign (`rocprofv3` captures, task #7).
- kyojin is absent: its EXL3 pack leg is blocked on the host's disk-space
  decision (needs ~120 GiB; the NVMe is full). Its rows, when they land, are
  engine-plus-format rows and never enter same-file ratios.

## Reproducing

```bash
# Environment: TheRock ROCm (see docs/ENVS.md), HIPENGINE_HIP_ARCH=gfx1151.
MODEL=/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
SHARD=$MODEL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
RES=benchmarks/results/2026-10-12-qwen4exp-cross-engine-pass1

$PY scripts/qwen4exp_canonical_ar_bench.py hipengine \
  --model-root "$MODEL" --output $RES/raw/hipengine-production.json \
  --prefill-chunk-size 4096 --capture-mode compact --ple-cache-mode warm \
  --compiler-version-file /tmp/hipengine-pass1-hipcc-version.txt --require-cached-build

$PY scripts/qwen4exp_canonical_ar_bench.py llamacpp \
  --server-bin /home/lhl/comparators-20261012/build-halobox/bin/llama-server \
  --model "$SHARD" --source-root /home/lhl/comparators-20261012/halobox-strix \
  --engine-label halobox-master-6aa81b88e \
  --output $RES/raw/halobox-master.json --server-log $RES/raw/halobox-master-server.log \
  --port 18082 --server-arg=-ngl --server-arg 999 --server-arg=-fa --server-arg on \
  --server-arg=-c --server-arg 4352 --server-arg=-b --server-arg 8192 \
  --server-arg=-ub --server-arg 2048 --server-arg=-t --server-arg 4

$PY scripts/qwen4exp_canonical_ar_bench.py openai \
  --adapter gufo --server-bin /home/lhl/comparators-20261012/gufo-src/build/release/gufo \
  --source-root /home/lhl/comparators-20261012/gufo-src --model-root "$MODEL" \
  --engine-label gufo-v0.12.0 --output $RES/raw/gufo-v0.12.0.json \
  --server-log $RES/raw/gufo-v0.12.0-server.log --port 18083 \
  --server-arg serve --server-arg llm --server-arg=-m --server-arg "$SHARD" \
  --server-arg=-c --server-arg 4352 --server-arg=--cache-ram-bytes --server-arg 1 \
  --server-arg=--host --server-arg 127.0.0.1 --server-arg=--port --server-arg 18083

$PY scripts/qwen4exp_canonical_ar_bench.py compare \
  $RES/raw/hipengine-production.json $RES/raw/halobox-master.json $RES/raw/gufo-v0.12.0.json \
  --allow-case-intersection --output $RES/artifact.json

# Quality battery (teacher + engines), then the score:
$PY scripts/cross_engine_quality_gate.py hipengine --model-root "$MODEL" \
  --engine-label hipengine-strict-teacher --profile strict \
  --output $RES/raw/quality-hipengine-strict-teacher.json
$PY scripts/cross_engine_quality_gate.py score \
  --teacher $RES/raw/quality-hipengine-strict-teacher.json \
  --engines $RES/raw/quality-hipengine-production.json \
             $RES/raw/quality-halobox-master.json \
             $RES/raw/quality-gufo-v0.12.0.json \
  --model-root "$MODEL" --output $RES/quality-score.json
```

Server logs and the prebuild diagnostic are kept in `raw/` locally but are
not part of the committed evidence. Per-engine run artifacts (sample rows,
identity blocks, disqualified-case records) are committed alongside
`artifact.json` and `quality-score.json`.
