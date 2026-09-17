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
| Revision | `680b6a554ecbbc21f0a000a14f5ccba07c3f6ca2` (2026-09-16) | `95f674951d6ab8f491f7907804a462c170f9c048` plus local edits to `build-amd.sh`, `serve-amd.sh`, `crates/atlas-kernels/build.rs`, `crates/atlas-kernels/build_target.rs` |
| Checkout | `/home/lhl/hipEngine-main` | `/home/lhl/atlas` |
| Server binary | `python -m hipengine.server` from that checkout | `target/release/spark`, sha256 `c2b457015de71d6fcad3bd86a7e8a1755b2c61b1e0e7312933f993bb200098c6` |
| Weights | `Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 bytes, sha256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` | `nvidia/Qwen3.8-27B-NVFP4`, revision `dbb8f445b3145f8a4c18ddc769f032d57d32867c`, ~22 GB |
| Weight quant | Q4_K_M (GGUF) | NVFP4 (mixed, requanted at load) |
| Backend | `hip_gfx1151` | `strix-hip` |
| KV cache | BF16, `--kv-storage bf16` | BF16, `--kv-cache-dtype bf16` |
| Session | `--max-context-tokens 16384` | `--max-seq-len 16384` |
| Speculative decode | off (`--speculative-mtp-serving off`) | off (`NUM_DRAFTS=0`) and MTP K=4 (`--num-drafts 3`) |

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
| hipEngine AR (Q4_K_M) | 517 | 1440 | 358.9 | 11.88 | 0.04% |
| hipEngine AR (Q4_K_M) | 945 | 2395 | 394.5 | 11.72 | 0.08% |
| hipEngine AR (Q4_K_M) | 3530 | 8667 | 407.3 | 10.96 | 0.03% |

Rows are grouped by target prompt length (~512, ~1024, ~4096 tokens); the
prompt-token column is what each engine reported after applying its own chat
template to the same user text.

- **Autoregressive decode**: Atlas is ahead by 3.6% / 6.1% / 8.4% at the three
  shapes. Both engines land between 10.96 and 12.44 tok/s.
- **Prefill**: hipEngine is 1.2–1.4x faster by TTFT-derived rate, and the gap
  widens with prompt length (334 vs 407 tok/s at ~3.5k tokens).
- **Speculative decode**: Atlas MTP K=4 is 2.86x / 2.31x / 1.77x its own AR row
  and 2.96x / 2.45x / 1.92x hipEngine's AR row. hipEngine's qualified MTP
  result in this repository (20.985 tok/s complete wall) uses 1,024-token
  sessions with 25 generated tokens and is not comparable to these 128-token
  rows; see [Why hipEngine served AR](#why-hipengine-served-ar).
- Every timing CV is at or below 0.08%, so the differences above are well
  outside run-to-run variation.

### Output agreement

The engines produce different text from the same user prompt, for two reasons:
each applies its own chat template (494 vs 517 prompt tokens at the same shape)
and each serves a different quantization. Longest common prefix of the first
600 generated characters:

| Shape | Atlas AR vs hipEngine AR | Atlas AR vs Atlas MTP K=4 |
| --- | ---: | ---: |
| ~512 | 4 chars | 600 chars |
| ~1024 | 536 chars | 377 chars |
| ~4096 | 257 chars | 257 chars |

Atlas MTP reproduces the Atlas AR continuation exactly at ~512 tokens and
diverges later at longer contexts, which is the expected behavior of a
verify-path arithmetic change.

## Why hipEngine served AR

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
| `candidate_budget` | 3 | 3 (default) — matches |
| `resident_capacity` | 1, 4, or 8 | 1 — matches |
| `realized_group_rows` | 1, 2, or 8 | 1 — matches |
| `sampling_mode` | `greedy_fast` | `greedy_fast` at `temperature: 0` — matches |
| `memory_fit` | true | true |

`/v1/hipengine/capabilities` reports the server-side values:

```console
$ curl -s http://127.0.0.1:8000/v1/hipengine/capabilities | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["model"]["execution_profile"], d["context"]["configured_max_context_tokens"])'
{'requested': None, 'resolved': 'production', 'manifest_sha256': 'c4a4a342e2243c2dcc430174606dde682393a2bd2e30acc83129027fcf572acc', ...} 8192
```

The gfx1151 one-row, three-draft cell at resident capacity 1 is
automatic-eligible, so on this artifact `auto` and `enabled` route a greedy
single request through MTP at any context, horizon, or session length, subject
only to the physical axes above. The hipEngine MTP column is not measured here;
filling it needs the same protocol as the AR rows, with the server started as
in [Running the hipEngine side](#running-the-hipengine-side) plus
`--speculative-mtp-serving enabled`.

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

`--max-context-tokens 16384` is deliberate. At `8192`, a prompt of ~3,500
tokens failed with:

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
  no per-shape K sweep; hipEngine MTP could not be measured on this host because
  no request shape tried matched an evidence row. The four shape and profile
  axes that caused that were removed on 2026-09-17, so the missing hipEngine MTP
  column needs a rerun rather than a narrower request.
- Atlas's numbers were unchanged by session length: an earlier AR run at
  `MAX_SEQ_LEN=8192` measured 12.32 / 12.45 / 11.88 tok/s at 1808 / 3306 /
  10488 ms TTFT, against 12.31 / 12.44 / 11.88 tok/s at 1804 / 3307 / 10495 ms
  at 16384.
