# hipEngine vs Atlas on gfx1151: Qwen3.8-27B

Date: 2026-09-17. Both engines served the same model family on the same
physical host and were driven by the same client script with byte-identical
request JSON. This document records the versions, the host, the protocol, the
measured rates, and the setup needed to reproduce either side.

The comparison covers autoregressive decode, prefill, and speculative decode.
It is a throughput comparison, not an output-equality comparison: the two
engines apply different chat templates and serve different quantizations, so
their greedy continuations differ (see [Output agreement](#output-agreement)).

## Versions compared

| | hipEngine | Atlas |
| --- | --- | --- |
| Revision | `680b6a554ecbbc21f0a000a14f5ccba07c3f6ca2` (2026-09-16) for the original run; `1a0a40cedcae4910e49d190d62125159884c2059` (2026-09-17) for the rerun | `95f674951d6ab8f491f7907804a462c170f9c048` plus local edits to `build-amd.sh`, `serve-amd.sh`, `crates/atlas-kernels/build.rs`, `crates/atlas-kernels/build_target.rs` |
| Checkout | `/home/lhl/hipEngine-main` | `/home/lhl/atlas` |
| Server binary | `python -m hipengine.server` from that checkout | `target/release/spark`, sha256 `c2b457015de71d6fcad3bd86a7e8a1755b2c61b1e0e7312933f993bb200098c6` |
| Weights | `Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 bytes, sha256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` | `nvidia/Qwen3.8-27B-NVFP4`, revision `dbb8f445b3145f8a4c18ddc769f032d57d32867c`, ~22 GB |
| Weight quant | Q4_K_M (GGUF) | NVFP4 (mixed, requanted at load) |
| Backend | `hip_gfx1151` | `strix-hip` |
| KV cache | BF16, `--kv-storage bf16` | BF16, `--kv-cache-dtype bf16` |
| Session | `--max-context-tokens 16384` | `--max-seq-len 16384` |
| Speculative decode | off (`--speculative-mtp-serving off`) for the AR arm; `enabled` with `--speculative-candidate-budget 3` for the MTP arm | off (`NUM_DRAFTS=0`) and MTP K=4 (`--num-drafts 3`) |

## Host and hardware

| | |
| --- | --- |
| APU | AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA3.5) |
| Memory | 125 GiB unified |
| ROCm | 10.0.0 (TheRock `therock10-staging-20260828` environment for Atlas; the same SDK root through `python -m rocm_sdk path --root` for hipEngine) |
| Host | single node, no containers, GPU otherwise idle |

## Protocol

Both engines were driven by the same script
([Atlas `scripts/strix/bench_1to1.py`](https://github.com/Atlas-Inf/atlas/blob/main/scripts/strix/bench_1to1.py),
copied to the host as `/tmp/bench_1to1.py`) over their OpenAI-compatible
`/v1/chat/completions` endpoints with `stream: true` and
`stream_options: {include_usage: true}`.

Request JSON, identical for both engines:

```json
{
  "model": "<served model id>",
  "messages": [{"role": "user", "content": "<prompt>"}],
  "max_tokens": 128,
  "temperature": 0.0,
  "enable_thinking": false,
  "chat_template_kwargs": {"enable_thinking": false},
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

- One discarded warmup per shape, then three measured runs. The reported value
  is the median of the three; `CV` is the coefficient of variation across them.
- Prompts are deterministic filler text asking for a long continuation, so the
  decode window is the full 128 tokens at every shape.
- `decode tok/s = (completion_tokens - 1) / (last_chunk_time - first_chunk_time)`.
- `prefill tok/s = prompt_tokens / TTFT`, where TTFT is measured to the first
  streamed token. This is a client-side rate that includes template render and
  first-token latency; it is not hipEngine's phase-timed internal prefill rate.
- Thinking is pinned off on both engines: Atlas defaults to thinking off,
  hipEngine defaults to on and streams `reasoning_content`.

## Results

| Engine | Prompt tokens | TTFT ms | Prefill tok/s | Decode tok/s | CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Atlas AR (NVFP4) | 494 | 1804 | 273.8 | 12.31 | 0.02% |
| Atlas AR (NVFP4) | 922 | 3307 | 278.8 | 12.44 | 0.01% |
| Atlas AR (NVFP4) | 3507 | 10495 | 334.2 | 11.88 | 0.02% |
| Atlas MTP K=4 (NVFP4) | 494 | 1801 | 274.3 | **35.19** | 0.04% |
| Atlas MTP K=4 (NVFP4) | 922 | 3287 | 280.5 | 28.70 | 0.02% |
| Atlas MTP K=4 (NVFP4) | 3507 | 10770 | 325.6 | 21.01 | 0.01% |
| hipEngine AR (Q4_K_M) | 517 | 1441 | 358.8 | 11.87 | 0.06% |
| hipEngine AR (Q4_K_M) | 945 | 2397 | 394.2 | 11.72 | 0.05% |
| hipEngine AR (Q4_K_M) | 3530 | 8660 | 407.6 | 10.96 | 0.03% |
| hipEngine MTP K=3 (Q4_K_M) | 517 | 1639 | 315.4 | 18.20 | 0.01% |
| hipEngine MTP K=3 (Q4_K_M) | 945 | 2751 | 343.5 | 17.20 | 0.16% |
| hipEngine MTP K=3, AR fallback (Q4_K_M) | 3530 | 8670 | 407.2 | 10.94 | 0.03% |

Rows are grouped by target prompt length (~512, ~1024, ~4096 tokens); the
prompt-token column is what each engine reported after applying its own chat
template to the same user text. The hipEngine rows were measured on 2026-09-17
in one session (AR arm first, MTP arm second); the Atlas rows are from the
original run and were not rerun.

- **Autoregressive decode**: Atlas is ahead by 3.7% / 6.1% / 8.4% at the three
  shapes. Both engines' AR rows land between 10.96 and 12.44 tok/s.
- **Prefill**: hipEngine is 1.2–1.4x faster by TTFT-derived rate, and the gap
  widens with prompt length (334 vs 408 tok/s at ~3.5k tokens). MTP raises
  hipEngine's TTFT by 198 ms at ~512 and 354 ms at ~1024, because the request
  pays a prompt-priming stage before the first speculative cycle.
- **Speculative decode**: Atlas MTP K=4 is 2.86x / 2.31x / 1.77x its own AR row,
  at 35.19 / 28.70 / 21.01 tok/s. hipEngine MTP K=3 is 1.53x / 1.47x its own AR
  row at the two shapes where it engages, 18.20 / 17.20 tok/s, and does not
  engage at ~4096 (see [MTP on this host](#mtp-on-this-host)). Atlas's MTP rows
  are therefore 1.93x / 1.67x / 1.92x hipEngine's corresponding rows. The
  hipEngine figures here are 128-output rows at a 16,384-token session; the
  repository's gfx1151 topline MTP number for this model (21.0 tok/s, 1.88x its
  matched 11.15 tok/s AR baseline) uses 24–25 outputs in a 1,024-token session.
  It is a different workload, and the artifacts do not establish physical-host
  identity with this comparison.
- Every timing CV is at or below 0.16%, so the differences above are well
  outside run-to-run variation.

### Output agreement

The engines produce different text from the same user prompt, for two reasons:
each applies its own chat template (494 vs 517 prompt tokens at the same shape)
and each serves a different quantization. Longest common prefix of the first
600 generated characters:

| Shape | Atlas AR vs hipEngine AR | Atlas AR vs Atlas MTP K=4 | hipEngine AR vs hipEngine MTP K=3 |
| --- | ---: | ---: | ---: |
| ~512 | 4 chars | 600 chars | 600 chars |
| ~1024 | 536 chars | 377 chars | 600 chars |
| ~4096 | 257 chars | 257 chars | 600 chars |

Atlas MTP reproduces the Atlas AR continuation exactly at ~512 tokens and
diverges later at longer contexts, which is the expected behavior of a
verify-path arithmetic change.

hipEngine MTP K=3 reproduces the hipEngine AR continuation exactly over the
compared window at all three shapes, including the ~4096 shape where the two
arms are the same execution anyway. The compared window is the first 600
generated characters, which is the whole 128-token generation at these shapes.
That is decoded-text agreement, not an independently recorded token-ID or
intermediate-state equality check.

## Why the original run served AR

`--speculative-mtp-serving` accepts `off`, `opt_in`, `auto` (the default), and
`enabled`. None of those modes route a request through MTP unless the request
matches a qualified evidence row, so the mode alone cannot enable MTP outside a
certified scope. The rows live in `hipengine/models/qwen35.py` and are resolved
by `resolve_speculative_mtp_serving_plan` in `hipengine/speculative/serving.py`.

The comparison ran against hipEngine revision `680b6a554ecbbc21f0a000a14f5ccba07c3f6ca2`,
whose resolver also required four axes that describe the envelope a benchmark
measured rather than the verified path:

| Removed axis | Qualified value | This host's server |
| --- | --- | --- |
| `execution_profile` + manifest | `strict` `393155123c5e0970…` or `production` `534a8bac3ca74428…` / `af20ee3b22921dc9…` | `production`, manifest `c4a4a342e2243c2d…` — no match |
| `max_sequence_length` | 1024 | 16384 (8192 in the first attempt) — no match |
| `context_tokens` | 1–67 or 1–128 | 517 / 945 / 3530 — no match |
| `output_horizon_tokens` | exactly 24 or 25 | 128 — no match |

A 128-token horizon at a 16,384-token session failed all four at once, so every
request fell back to AR, which is why the two hipEngine rows in the results
table are AR rows. Those axes were removed on 2026-09-17 (commit `25121713c`,
`docs/EXECUTION-PROFILES.md` §2.9): session length, prompt context, output
horizon, and the resolved variant-manifest hash change with ordinary serving
traffic and with any kernel or variant selection, so gating on them silently
disabled an already-qualified path.

Admission is now decided only by artifact content, backend, architecture,
weight quant, KV storage and layout, realized group rows, resident capacity,
candidate depth, sampling mode, and memory fit. Every one of those axes still
must match, and this host's server matches all of them:

| Axis | Qualified value | This host's server |
| --- | --- | --- |
| `artifact_sha256` / size | `7e78da5d…` / 17,106,775,008 | matches |
| `backend` / `target_arch` | `hip_gfx1151` / `gfx1151` | matches |
| `weight_quant` | `gguf_q4_k_m` | matches |
| `kv_storage` / `kv_layout` | `bf16` / `uniform` | matches |
| `candidate_budget` | 3 | 4 (server default) — no match; the measured MTP arm passes 3 |
| `resident_capacity` | 1, 4, or 8 | 4 — matches the capacity-4 row |
| `realized_group_rows` | 1, 2, or 8 | 1 — matches |
| `sampling_mode` | `greedy_fast` | `greedy_fast` at `temperature: 0` — matches |
| `memory_fit` | true | true |

`/v1/hipengine/capabilities` reports the server-side values:

```console
$ curl -s http://127.0.0.1:8000/v1/hipengine/capabilities | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["model"]["execution_profile"], d["context"]["configured_max_context_tokens"])'
{'requested': None, 'resolved': 'production', 'manifest_sha256': 'c4a4a342e2243c2dcc430174606dde682393a2bd2e30acc83129027fcf572acc', ...} 8192
```

The gfx1151 one-row cell is automatic-eligible, so `auto` and `enabled` admit a
greedy single request on this artifact at any context, horizon, or session
length, subject only to the physical axes above. Candidate depth is one of those
axes, so the depth has to be pinned to a qualified value first; the next section
has the measured details.

## MTP on this host

Measured 2026-09-17 at revision `1a0a40cedcae4910e49d190d62125159884c2059`, one
session, AR arm first and MTP arm second. The MTP arm passed
`--speculative-mtp-serving enabled --speculative-candidate-budget 3`; everything
else is the command in [Running the hipEngine side](#running-the-hipengine-side).

Two limits decide whether a request actually speculates, and neither is visible
in the response body:

| Limit | Effect |
| --- | --- |
| Candidate depth | The server's default `--speculative-candidate-budget` is 4. Every retained Qwen3.8 evidence row pins 3 (11 rows) or 2 (3 rows), so the default fails admission with `candidate_budget_not_qualified` and `permanent_ar`. Passing 3 admits the cell above. |
| Prompt-priming window | The MTP path refuses a prompt when `prompt_tokens + 1 >= min(1023, max_sequence_length)`, recording `target_context_k0`, and the whole request then runs AR. The ~4096 shape (3,530 tokens) is past it; ~512 and ~1024 are inside it. |

The second limit is why the ~4096 MTP row above is an AR row: admission and
routing both say `speculative_mtp`, and the backend still executes AR. The
response reports this as `effective_route: default` with
`decision_reason: backend_k0_fallback`, which is the only signal a client gets.

Per-shape draft accounting from a non-streaming probe of the same three prompts
(`usage.completion_tokens_details` carries the accepted and rejected counts):

| Shape | Route selected | Effective | Cycles | Draft tokens | Accepted | Accept rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ~512 | `speculative_mtp` | `speculative_mtp` | 55 | 163 | 72 | 0.442 |
| ~1024 | `speculative_mtp` | `speculative_mtp` | 21 | 63 | 56 | 0.889 |
| ~4096 | `speculative_mtp` | `default` | 0 | 0 | 0 | — |

Acceptance is not the whole explanation of the 1.53x/1.47x rates. The shorter
shape has the larger measured gain despite lower draft acceptance. The backend
also caps speculative execution during decode at context 1023, so a request
that starts inside the prompt-priming window can switch to AR before finishing.
The ~512 probe accounts for all 127 post-first-token outputs through 55 cycles
plus 72 accepted drafts. The ~1024 probe accounts for only 77 through 21 cycles
plus 56 accepted drafts, leaving 50 outputs consistent with an AR tail at the
context boundary. This is an inference from the separate non-streaming probe
and runtime policy, not a per-cycle trace of the timed streaming requests.

The prompt predicate above admits at most **1,021** prompt tokens with a large
enough session: 1,022 + 1 already reaches the rejected boundary. The response's
`used: true` means MTP occurred, not that it covered the entire request.
See the [serving improvement review](QWEN38-GFX1151-MTP-SERVING-IMPROVEMENTS.md)
for the source trace, matched-baseline requirements and prioritized fixes.

Artifact: [`2026-09-17-gfx1151-qwen38-atlas-comparison-mtp-k3.json`](../benchmarks/results/2026-09-17-gfx1151-qwen38-atlas-comparison-mtp-k3.json).

## Running the hipEngine side

```bash
git clone https://github.com/shisa-ai/hipEngine.git
cd hipEngine
git lfs install          # required: the vendored AOTriton runtime is LFS-tracked
git lfs pull
pip install -e .
```

`git lfs pull` is not optional. Without it the 13 AOTriton files under
`hipengine/kernels/hip_gfx1100/attention/aotriton_runtime/` are pointer text,
and the first prefill fails with:

```text
generation failed: AOTriton library at .../libaotriton_v2.so.0.11.2 is a Git LFS
pointer, not the binary payload. Run `git lfs pull` to install the vendored runtime.
```

A clone made without Git LFS already has the pointer files in its working tree;
`git lfs install && git lfs pull` replaces them in place.

Then download the GGUF and serve:

```bash
# unsloth/Qwen3.8-27B-GGUF at revision 65ca473 (the standard, non-UD file)
# 17,106,775,008 bytes; sha256 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf \
  --revision 65ca473 --local-dir /models/gguf

ROCM_ROOT="$(python -m rocm_sdk path --root)"
export HIP_PATH="$ROCM_ROOT" ROCM_PATH="$ROCM_ROOT"
export PATH="$ROCM_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib64:$LD_LIBRARY_PATH"

python -m hipengine.server \
  --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  --backend hip_gfx1151 \
  --served-model-name qwen3.8-27b-q4km \
  --max-context-tokens 16384 \
  --kv-storage bf16 \
  --speculative-mtp-serving off \
  --host 127.0.0.1 --port 8000 --log-level info
```

For the MTP arm, change the serving flag and pin the qualified candidate depth:

```bash
  --speculative-mtp-serving enabled \
  --speculative-candidate-budget 3 \
```

`--max-context-tokens 16384` is deliberate. On the original 2026-09-16 run, at
`8192`, a prompt of ~3,500 tokens failed with:

```text
generation failed: packed workspace lease holds 32 pages but the workspace needs 36
```

and at `1024` the server did not finish startup warmup:

```text
WARMUP: failed during eager startup
RuntimeError: packed workspace lease holds 4 pages but the workspace needs 16
```

The packed-execution workspace lease was sized `1 x ceil(session_tokens / 256)`
pages while the workspace is allocated for the union geometry, which multiplies
by the serving slot capacity. The two did not agree on the slot term:

| Session | Leased pages | Workspace need | Realized geometry |
| --- | ---: | ---: | --- |
| 1024 | 4 | 16 | 4 slots x 4 pages/slot (1024 tokens) |
| 8192 | 32 | 36 | 4 slots x 9 pages/slot (2304 tokens) |

At a 16,384-token session the lease is 64 pages, which covers the 36 the
workspace asks for, so the failure did not appear and the AR rows above were
measurable. The lease now uses the same slot ceiling as the union geometry
(`packed_verify_lease_slot_ceiling`), so a 1024-token session warms up and a
3,500-token prompt at 8192 succeeds.

Atlas sizes its KV pool from available GPU memory instead
(`--gpu-memory-utilization 0.88` produced a 60.5 GB pool of 61,914 blocks in
this run) and caps the session separately, so it does not have this failure mode
at the same prompt length.

## Running the Atlas side

Atlas is a separate project (`https://github.com/Atlas-Inf/atlas`, AGPL-3.0). On
this host it runs as a native binary against ROCm with no container.

```bash
git clone https://github.com/Atlas-Inf/atlas.git
cd atlas

# ~22 GB into the Hugging Face cache
hf download nvidia/Qwen3.8-27B-NVFP4

./build-amd.sh                  # strix-hip backend, all targets; needs ROCm and cargo
MAX_SEQ_LEN=16384 ./serve-amd.sh
```

`build-amd.sh` finds ROCm at `/opt/rocm`, at `$ATLAS_ROCM_HOME`, or from the
`hipcc` and `rocminfo` that an activated ROCm environment puts on `PATH` (conda
and pip ROCm SDKs included). `serve-amd.sh` serves
`nvidia/Qwen3.8-27B-NVFP4` on port 8081 by default with the validated
configuration: BF16 KV, NVFP4 lm-head, max batch size 1, GPU memory utilization
0.88, and MTP K=4 (`--speculative --num-drafts 3`).

Environment knobs used by this comparison:

| Variable | Value used | Effect |
| --- | --- | --- |
| `MAX_SEQ_LEN` | 16384 | session length; matches hipEngine's `--max-context-tokens 16384` |
| `NUM_DRAFTS` | `0` for the AR row, unset (3) for the MTP row | `0` disables speculation |
| `GPU_UTIL` | unset (0.88) | KV pool budget fraction |
| `MAX_PREFILL_TOKENS` | unset (2048) | prefill chunk size |
| `SSM_CKPT_INTERVAL` | unset (16) | SSM snapshot granularity; `serve-amd.sh` derives the slot count from `MAX_SEQ_LEN` and this interval |
| `LM_HEAD` | unset (`nvfp4`) | lm-head precision; use `bf16` for the `unsloth` checkpoint |

Startup is ~25 s and the server reports readiness on
`GET http://127.0.0.1:8081/v1/models` with
`{"id": "nvidia/Qwen3.8-27B-NVFP4", "max_model_len": 16384}`.

## Running the harness

```bash
python3 bench_1to1.py \
  --url http://127.0.0.1:8000 \
  --model qwen3.8-27b-q4km \
  --tag hipengine-ar-q4km \
  --shapes 512,1024,4096 --decode 128 --repeats 3 \
  --out /tmp/bench-hipengine-ar.json
```

For the Atlas rows, point `--url` at `http://127.0.0.1:8081` and `--model` at
`nvidia/Qwen3.8-27B-NVFP4`. Per-run JSON is written to `--out`; the reported
medians and CVs in this document come from those files. Extra request fields can
be added with `BENCH_EXTRA_JSON`, for example
`BENCH_EXTRA_JSON='{"speculative_mtp": true}'`.

## Scope and limitations

- One host, one APU, one model family. Both engines ran on `gfx1151`; the
  hipEngine evidence rows that gate MTP are pinned per architecture, and the
  gfx1100 rows do not apply here.
- The prefill column is TTFT-derived, so it includes HTTP, template render, and
  first-token latency. It is comparable between the two engines because the same
  client measured both, but it is not comparable to hipEngine's phase-timed
  internal prefill figures.
- Atlas MTP rows were measured with the engine's default draft count (K=4) and
  no per-shape K sweep, so they are not a tuned Atlas figure. hipEngine MTP was
  measured at K=3 because that is the depth every retained evidence row pins;
  its default depth of 4 admits nothing.
- hipEngine MTP is bounded by both prompt activation and per-cycle context on
  this host: prompts above 1,021 tokens run AR even though admission and routing
  select `speculative_mtp`. Admitted shorter prompts can switch to AR during
  decode. The ~4096 row is therefore an AR row, and no hipEngine MTP figure
  here describes long-prompt speculation.
- Both hipEngine arms generated 128 tokens at a 16,384-token session, which is
  outside the envelope of the repository's qualified MTP rows (24–25 outputs,
  1,024-token sessions). The numbers here measure the path at these shapes; they
  are not a re-qualification of it.
- The two hipEngine arms ran in one session with the AR arm first. The AR rates
  reproduce the original run within 0.01 tok/s at all three shapes, so the
  matched AR baselines are same-host and same-protocol.
- Atlas's numbers were unchanged by session length: an earlier AR run at
  `MAX_SEQ_LEN=8192` measured 12.32 / 12.45 / 11.88 tok/s at 1808 / 3306 /
  10488 ms TTFT, against 12.31 / 12.44 / 11.88 tok/s at 1804 / 3307 / 10495 ms
  at 16384.
