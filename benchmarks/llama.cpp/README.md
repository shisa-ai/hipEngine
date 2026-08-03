# llama.cpp MTP benchmark patchset

This directory preserves the local llama.cpp instrumentation used by the GGUF
MTP comparison work. The external checkout under `/home/lhl/llama.cpp/` remains
a read-only reference; these patches are the repository-owned reproduction
surface.

## Source and patch order

The captured source lineage is:

1. upstream base `6e9007ae61f4e994c27484759caac6ef2aa32b30`;
2. [`0001-mtp-instrumentation-committed.patch`](0001-mtp-instrumentation-committed.patch),
   the seven local commits through
   `1ebf790cda38d827559548f67b0469189690cc8c`;
3. [`0002-mtp-instrumentation-working-tree.patch`](0002-mtp-instrumentation-working-tree.patch),
   the additional instrumented working-tree state used to build the retained
   local binaries.

The two patches are intentionally separate. The first preserves reviewable
local commit history as one base-to-head diff; the second makes the otherwise
uncommitted instrumentation reproducible. Their hashes, commit list, binary
identity, and validation are recorded in [`manifest.json`](manifest.json).

Apply them to a fresh checkout in order:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout 6e9007ae61f4e994c27484759caac6ef2aa32b30
git apply /path/to/hipEngine/benchmarks/llama.cpp/0001-mtp-instrumentation-committed.patch
git apply /path/to/hipEngine/benchmarks/llama.cpp/0002-mtp-instrumentation-working-tree.patch
```

The capture was validated in a fresh shared clone: both `git apply --check`
steps passed, and the resulting base-to-working-tree binary diff matched the
source checkout byte-for-byte.

## Measurement-only Laguna patches

Two later patches are independent of the MTP patch stack and apply to clean
llama.cpp `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`:

- [`0003-laguna-content-only-raw-measurement.patch`](0003-laguna-content-only-raw-measurement.patch)
  bypasses redundant post-generation PEG parsing only for content-only server
  responses. It changes no timed model work or device code.
- [`0004-laguna-matched-prefill-token-fixture.patch`](0004-laguna-matched-prefill-token-fixture.patch)
  gates llama-bench context4096 admission and the exact 512-token Laguna stream
  behind `LLAMA_BENCH_MATCHED_LAGUNA_M512`, then prints last-row top-1 after the
  timed decode. It changes no backend graph, kernel selection, arithmetic, or
  timed boundary.

Apply the prefill fixture to a clean same-revision HIP or Vulkan tree:

```bash
git checkout c0bc8591e8815c63cb01dd3f051a8b0df02501c9
git apply /path/to/hipEngine/benchmarks/llama.cpp/0004-laguna-matched-prefill-token-fixture.patch
```

The canonical direct-M512 command is:

```bash
LLAMA_BENCH_MATCHED_LAGUNA_M512=1 \
HIP_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1 \
./llama-bench \
  -m /models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf \
  -p 512 -n 0 -fa 1 -ctk bf16 -ctv bf16 -r 5 -o json
```

The token fixture, HIP/Vulkan rates, binary/source hashes, trace families, and
cross-engine caveats are recorded in
[`2026-07-29-gfx1100-laguna-q2-xl-llamacpp-prefill-matched-attribution.json`](../results/2026-07-29-gfx1100-laguna-q2-xl-llamacpp-prefill-matched-attribution.json).

## Instrumentation provided

The committed patch adds MTP stage timings, ROCTX ranges, token/proposal
traces, draft score traces, hidden-state summaries, graph tensor taps, and
server JSON emission. The working-tree patch extends that surface with
selective full tensor values, target-layer/pre-output-norm rows, target sample
traces, and detailed attention Q/K/V/mask/output probes.

Primary environment controls include:

| Variable | Purpose |
| --- | --- |
| `LLAMA_MTP_STAGE_TIMINGS=<jsonl>` | Emit per-cycle stage and cycle-wall JSONL. |
| `LLAMA_MTP_TOKEN_TRACE=1` | Include token, proposal, hidden, and enabled tensor traces. |
| `LLAMA_MTP_ROCTX=1` | Add ROCTX ranges around MTP stages. |
| `LLAMA_MTP_HIDDEN_TRACE_VALUES=1` | Retain selected full hidden rows. |
| `LLAMA_MTP_HIDDEN_TRACE_VALUE_LABELS=<csv>` | Filter retained hidden labels. |
| `LLAMA_MTP_HIDDEN_TRACE_VALUE_ROWS=<csv>` | Filter retained hidden row indices. |
| `LLAMA_MTP_TENSOR_TRACE_VALUES=1` | Retain selected full graph-tensor rows. |
| `LLAMA_MTP_TENSOR_TRACE_VALUE_LABELS=<csv>` | Filter graph-tensor labels. |
| `LLAMA_MTP_TENSOR_TRACE_VALUE_ROWS=<csv>` | Filter graph-tensor row indices. |
| `LLAMA_MTP_ATTENTION_TRACE=1` | Emit detailed MTP attention probes. |
| `LLAMA_MTP_TARGET_TRACE_TOP_K=<n>` | Set target-sample trace top-k. |
| `LLAMA_MTP_TARGET_TRACE_CANDIDATES=<csv>` | Add target token IDs to trace explicitly. |

## Reference HIP build

The retained gfx1151 comparator used a Release build with `GGML_HIP=ON` and
`GGML_NATIVE=ON`, built with GNU 16.1.1:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_NATIVE=ON
cmake --build build --parallel
```

The benchmarked `llama-server` identifies itself as build 9648 / `1ebf790cd`.
Always record the executable SHA-256 as authoritative because llama.cpp's
self-reported commit does not encode a dirty working tree.

## Decode timing boundary

llama.cpp's native `predicted_per_second` is not directly comparable with a
post-prefill decode-loop rate. In the captured source,
`server_slot::t_start_generation` is set **after** the first output token has
been sampled, while `predicted_n` includes that token. Consequently native
`predicted_n / predicted_ms` counts one untimed token per request.

For a comparison covering `N` timed decode transitions per prompt:

1. request `N + 1` output tokens from llama.cpp;
2. sum `predicted_ms` across the requests;
3. report `sum(predicted_n - 1) * 1000 / sum(predicted_ms)`;
4. compare with hipEngine MTP `sum(visible_output_tokens) * 1000 /
   sum(cycle_wall_ms)`, not its narrower legacy stage sum.

Client/request wall includes prompt processing, HTTP, and response handling and
must be reported separately. See the canonical contract in
[`../README.md`](../README.md).
