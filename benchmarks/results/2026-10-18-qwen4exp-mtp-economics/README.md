# Qwen4Exp MTP economics, 2026-10-18 (zbook / gfx1151)

Model: unsloth `Qwen3.8-Flash-Next` `UD-Q4_K_XL`, execution profile `production`,
MTP sidecar `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, budget 3, greedy, 64 generated
tokens, host machine `55ea6c509d0b49eea8de7094a1023668`, tracked HEAD
`b021b007c`.

Two harnesses ran on one machine on one day, so every ratio below is a
same-host, same-session comparison:

- `scripts/qwen4exp_mtp_inprocess_bench.py` — one resident target, true AR arm
  (no MTP, plain decode steps) versus the MTP provider, over the full category
  suite and over the category heldouts.
- `scripts/qwen4exp_http_mtp_economics.py` — the served route, requests with
  `speculative_mtp` off versus on, at a fixed request width. The MTP arm is only
  counted when the server reports that MTP executed.

## Single-request economics (in-process, C1)

| run | AR tok/s | MTP tok/s | MTP/AR | acceptance |
| --- | --- | --- | --- | --- |
| full suite, run 1 | 20.48 | 23.83 | 1.163 | 0.561 |
| full suite, run 2 | 20.95 | 24.45 | 1.167 | 0.561 |
| category heldouts | 20.68 | 23.93 | 1.157 | 0.535 |

Per category on the suite: code 1.408 / 1.384, general_en 1.142 / 1.150,
general_ja 1.074 / 1.087, mixed_ja_en 0.946 / 0.960. The two suite prompts in
`mixed_ja_en` are the only pair below 1.0; the two heldout `mixed_ja_en` prompts
measure 1.079, so that category is not uniformly negative.

## The retained 1.273x row is a session artifact, not a HEAD regression

The 2026-10-14 rollup retained 1.273x (AR 22.66, MTP 28.85) at commit
`9a0c408d2`. Running that commit's own code on today's machine
(`inprocess-suite-at-9a0c408d2.json`) measures AR 20.91, MTP 22.70, **1.086**.

So today's machine reproduces neither absolute rate, and HEAD is **faster** than
the commit that produced the retained row under the same session protocol:

| same host, same session | AR tok/s | MTP tok/s | MTP/AR |
| --- | --- | --- | --- |
| `9a0c408d2` (retained row's commit) | 20.91 | 22.70 | 1.086 |
| `b021b007c` (HEAD) | 20.48–20.95 | 23.83–24.45 | 1.163–1.167 |

Draft acceptance is identical on both (0.561) and the cycle counts match prompt
for prompt, so the difference is execution speed, not a different draft path.
Absolute tok/s across days is not an old→new comparison; the same-session pair
above is.

## Served economics by request width

`http-*.json`. AR at width 1 is measured twice: on the shipped default (serial
c1 decode) and with the opt-in AR batch path enabled, which an operator needs
for concurrency.

| width | AR arm | AR tok/s | MTP arm | MTP tok/s | MTP/AR |
| --- | --- | --- | --- | --- | --- |
| 1 | serial c1 (default) | 18.93 | all MTP | 23.80 | **1.257** |
| 1 | AR batch path | 13.39 | all MTP | 23.77 | 1.775 |
| 2 | AR batch path | 18.35 | all MTP | 24.40 | **1.330** |
| 2 | AR batch path | 18.35 | 1 MTP + 1 AR | 18.46 | 1.006 |
| 4 | AR batch path | 28.80 | all MTP | 24.92 | **0.865** |
| 4 | AR batch path | 28.80 | 1 MTP + 3 AR | 26.76 | 0.929 |

MTP throughput is essentially flat across widths (23.8–24.9 tok/s) because
speculative rows keep a request-owned c1 runner and never join a batch step.
AR throughput scales with width (13.4 → 18.4 → 28.8), so the crossover sits
between two and four concurrent requests.

At width 4 a single MTP request is 1.5–3.1x faster than its AR peers in the same
wave (2.94 s and 4.96 s versus 7.75–9.11 s for 64 tokens) while those peers lose
7–17%, which nets out at −7% aggregate.

## Concurrent MTP crashed the server before this run

The first width-4 attempt returned `500` for three of four concurrent MTP
requests:

```
generation failed: '_CompositeGenerationCancellationToken' object has no
attribute 'cancel_requested'
```

The engine service treats a driver-tick exception as a driver failure and tears
the resident engine down, so every later request returned `503 engine service is
closed`. Fixed by giving the composite token a `cancel_requested` property that
mirrors `cancelled`, and by probing the attribute with `getattr` in the serving
MTP provider the way the other call sites already did
(`tests/test_unit_server_composite_cancellation.py`). `http-width4.json` is the
post-fix run: MTP executed on 10/10 requests.

## Policy conclusion

- Single request: MTP is worth 1.26x on the shipped serial AR path, and 1.33x at
  two concurrent requests.
- Four concurrent requests: AR batching wins by 15%; requesting MTP in that
  batch costs the other requests about 7% aggregate.
- The served default therefore stays `--speculative-mtp-serving opt_in`.
  Evidence-gated `auto` needs an artifact-identity path in the Qwen4Exp
  generator, which does not exist yet; the measured scope to encode when it does
  is `realized_group_rows <= 2`, greedy, budget 3, context <= 2048. See
  `docs/REFACTOR.md`.
- The AR batch path is slower than serial c1 decode below four rows (13.39 and
  18.35 against 18.93 tok/s). That is an AR-side opportunity, recorded in
  `docs/REFACTOR.md`, and it is why the width-1 comparison is reported twice.
