# MLPerf Edge Agentic local evaluations

This directory records **140-turn engine screens** and the earlier
27-turn integration diagnostics, not a full MLPerf result or BFCL accuracy pass.
The campaign and comparison rules are in
[`docs/campaigns/MLPERF-EDGE-AGENTIC.md`](../../docs/campaigns/MLPERF-EDGE-AGENTIC.md).
The first attempt used the upstream runner unchanged. The repaired run uses
only the coordinator-timestamp patch documented below.

## Speculative 140-turn comparison: 2026-09-25

**hipEngine and llama.cpp complete the replay in less wall time than gufo in
these passes.** All three execute speculation and return valid tool calls.
This compares the frozen Qwen3.6 workload, not each project's headline setup.

| Metric | hipEngine MTP | llama.cpp MTP | Patched gufo DFlash2 |
| --- | ---: | ---: | ---: |
| Completed turns | 140/140 | 140/140 | 140/140 |
| Measured replay phase | 793.15 s | 827.10 s | 1,041.49 s |
| Mean / median turn latency | 5.665 / 4.405 s | 5.908 / 4.462 s | 7.439 / 5.195 s |
| Mean client-visible first output | 2.535 s | 2.872 s | 7.039 s |
| Inline command-name IoU | 0.5586 | 0.6423 | 0.6387 |
| Accepted / proposed draft tokens | 7,000 / 8,121 | 7,565 / 8,178 | 2,584 / 16,369 |
| Draft acceptance | 86.20% | 92.50% | 15.79% |
| Prefix-token reuse | 95.31% | 95.95% | 96.55% |
| Effective cache-inclusive prefill | 4,744 tok/s | 6,089 tok/s | approximately 8,023 tok/s |

Same physical `gfx1151` / Radeon 8060S host, concurrency 1, 32K context,
temperature 0, seed 42, thinking off, maximum 1024 output tokens, and frozen
recorded histories. hipEngine runs production profile at `c22902eda`;
llama.cpp uses the same `1ebf790cd` binary as its AR baseline. No tool is executed.
hipEngine and llama.cpp use BF16 target KV; llama.cpp's draft KV defaults to
FP16. Gufo uses FP16 target KV and the user-selected Qwen3.8 DFlash2 draft.

The original GGUF has no NextN weights. For the MTP arms, a separate container
copies **all 851 original target tensors byte-for-byte** and adds 15 Qwen3.6
NextN tensors from `unsloth/Qwen3.6-27B-MTP-GGUF` revision
`5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace`. Target tensor hashes are verified;
the container hash differs. Gufo uses the original file unchanged. No target
quantization or weights were substituted.

Both MTP arms have zero empty, malformed, truncated or failed responses.
hipEngine executes speculative cycles on all 140 requests with zero recoverable
speculative failures. Compared with each engine's earlier AR pass, 139/140
hipEngine and 132/140 llama.cpp rendered outputs match after removing generated
tool IDs. hipEngine's command IoU is unchanged; llama.cpp's changes from 0.6420
to 0.6423. These are replay diagnostics, **not accuracy parity or a BFCL pass**.
Gufo's inline score is close to llama.cpp's and higher than hipEngine's; that
metric does not establish whether command arguments solve the task. Shared
upstream-scored quality evaluation remains pending.

Effective prefill divides each engine's full prompt-token count by pooled
prefill time, including prefix reuse. Tokenization differs: hipEngine counts
1,400,682 prompt tokens and reuses 1,335,040; llama.cpp counts 1,675,019 and reuses
1,607,127. Prefill totals are 295.278 / 275.096 seconds. hipEngine's streamed
decode window totals 436.152 seconds for 9,847 output tokens; llama.cpp's engine
eval totals 514.129 seconds for 10,263 tokens. These windows are not equivalent
and must not be presented as interchangeable decode throughput. Harness TPOT
is still unsuitable for tool-call comparisons, as explained below.

[Compact artifact](../results/2026-09-25-gfx1151-edge-speculative-140.json)
contains commands, hashes, measurements and validation. Raw root:
`~/gate-runs/mlperf-edge-speculative-20260925`. The primary hipEngine run is
`hipengine-screen-140-accounted/`; the first pass is also preserved (791.28 s,
same accepted/proposed totals) but its extra logger was filtered and cache
counts were not captured. The repeat changes only telemetry capture.
llama.cpp uses verbosity 5 to expose prompt counts; logging overhead is included.
Runs are serial with fresh servers and no concurrent GPU work. Repeat with a
new output directory, the artifact's server command, and:

```bash
cd ~/mlperf-edge-endpoints
.venv/bin/inference-endpoint benchmark from-config --config "$HOME/gate-runs/mlperf-edge-speculative-20260925/hipengine-screen-140-accounted/config.yaml"
.venv/bin/inference-endpoint benchmark from-config --config "$HOME/gate-runs/mlperf-edge-speculative-20260925/llamacpp-screen-140/config.yaml"
```

## Paired 140-turn autoregressive screen: 2026-09-25

**The engines have similar elapsed time in this single pass; hipEngine has a
lower inline command-overlap score.** Both complete all 140 turns without
errors, missing turns, empty outputs, malformed arguments or length-limited
responses. Every response finishes with `tool_calls`. hipEngine produces 140
valid `bash` calls; llama.cpp produces 141 (one response contains two).

Same physical host `gfx1151` (machine ID `55ea6c509d0b49eea8de7094a1023668`),
Ryzen AI Max+ 395 / Radeon 8060S, identical Qwen3.6-27B Q4_K_M bytes, BF16 KV,
32,768-token window, concurrency 1, temperature 0, seed 42, maximum 1,024 output
tokens, reasoning off. All issued prompt data matches across engines; canonical
client prompt lengths span 1,363–22,902 tokens. No generated tool is executed,
and recorded tool delays are not injected.

| Metric | hipEngine | llama.cpp |
| --- | ---: | ---: |
| Measured replay phase | 1,249.47 s | 1,282.06 s |
| Mean turn latency | 8.925 s | 9.157 s |
| Median turn latency | 6.537 s | 6.525 s |
| Mean time to first visible output | 2.589 s | 2.832 s |
| Mean first-output-to-completion time | 6.336 s | 6.325 s |
| Full-message client token estimate | 9,703 | 9,799 |
| Inline executable-call multiset IoU | 0.5586 | 0.6420 |

Medians in this table use the conventional sample median; the artifact also
preserves upstream's percentile estimates, which use a different convention.
The measured phase is 2.54% shorter for hipEngine, but this is one sequential
pass per engine with different outputs, not a stable speedup claim. Only 49/140
paired responses contain identical command lists. IoU measures overlap with
recorded executable names, not command-argument correctness, issue resolution or
BFCL accuracy. No BFCL sample or numerical quality gate ran.

| Trajectory | Turns | Mean latency: hipEngine / llama.cpp | Client output tokens: hipEngine / llama.cpp | IoU: hipEngine / llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| Flask | 27 | 8.407 / 10.875 s | 1,586 / 2,315 | 0.6574 / 0.6975 |
| Xarray | 52 | 10.190 / 10.112 s | 4,280 / 4,119 | 0.5513 / 0.6154 |
| Pytest | 61 | 8.075 / 7.584 s | 3,837 / 3,365 | 0.5210 / 0.6402 |

### Timing boundaries and a TPOT defect

**Do not use this run's harness TPOT as decode-speed evidence.** Its mean values
are 68.885 / 96.541 ms, but `TpotTrigger` removes the first stream chunk before
retokenizing the remaining message. That can remove the tool name. Both runner
logs record a chat-template `UndefinedError` followed by fallback tokenization.
A saved-output replay reproduces both means exactly: the post-first token
estimates total 12,275 / 9,709, versus only 9,703 / 9,799 for the full messages.
Chunk shape changes the denominator, so the apparent TPOT advantage is not a
valid engine decode-rate comparison. Original distributions are preserved, not
silently corrected. Coordinator-arrival elapsed times do not depend on this
tokenization and remain usable.

Server-reported prompt-processing totals are 276.340 / 257.870 seconds. They
include each engine's selected cache reuse, not full-prompt recomputation.
hipEngine reports 885.619 seconds of `stream_decode_ms`, measured since its first
observed output token, and 9,844 generated tokens. llama.cpp reports 1,002.894
seconds of `eval time` and 9,939 generated tokens. These are different timing
windows and token denominators; do not divide them into an isolated kernel-speed
claim or equate them with client-visible tool-argument generation.

### Provenance and repeat commands

hipEngine ran `2fafa58f5`, production profile, radix prefix cache, with route
counters confirming 140 host-sampler AR requests and zero MTP. llama.cpp's HIP
binary reports `1ebf790cda38d827559548f67b0469189690cc8c`; its checkout HEAD is
newer and is not the measured binary identity. It used its prompt cache and
recurrent checkpoints, with no speculative implementation configured. The
[compact artifact](../results/2026-09-25-gfx1151-mlperf-edge-140.json) contains
binary/model hashes, original distributions, exact commands, host/compiler
provenance and hashes of all raw files.

Raw root: `~/gate-runs/mlperf-edge-140-20260925`. Each arm contains `config.yaml`,
`server-command.json`, `runner-command.json`, logs, counters and the upstream
reports. `manifest.json` freezes the 280 source rows / 140 generated turns.
`run-edge-140.py` records orchestration; `environment.sh` preserves the local
ROCm environment. Both measured servers started fresh, hipEngine first; a
separate llama.cpp tool-call preflight ran before either measured process.

With the corresponding server command from the artifact running, the measured
CLI commands were:

```bash
cd ~/mlperf-edge-endpoints
.venv/bin/inference-endpoint benchmark from-config --config "$HOME/gate-runs/mlperf-edge-140-20260925/hipengine/config.yaml"
.venv/bin/inference-endpoint benchmark from-config --config "$HOME/gate-runs/mlperf-edge-140-20260925/llamacpp/config.yaml"
```

For a repeat, copy configurations to a new run directory and change `report_dir`;
do not overwrite the recorded results. Run servers sequentially. Budget about
25–30 minutes per subset arm including startup/drain margin, based on this
21-minute measured phase. Keep the full performance reservation at 1–3 hours
per engine: these three correlated trajectories do not establish a reliable
stratified prediction for the other 17. Full BFCL runtime is unmeasured.

## Patched gufo with cross-version DFlash2: 2026-09-25

**Gufo completes all 140 turns with active DFlash2 speculation.** The target is
unchanged Qwen3.6-27B Q4_K_M on the same physical gfx1151 host as the paired
screen. The user selected a Qwen3.8 DFlash2 Q4_K_M draft; this is a cross-version
pairing, not a Qwen3.8 target run. Gufo uses FP16 attention KV, unlike the BF16
KV of the earlier pair. Context, concurrency, sampling and output limits match
the frozen screen; thinking and history preservation are explicitly off.

| Metric | Patched gufo + DFlash2 |
| --- | ---: |
| Completed turns / valid bash calls | 140 / 141 |
| Measured replay phase | 1,041.49 s |
| Mean / median turn latency | 7.439 / 5.195 s |
| Mean client-visible first output | 7.039 s |
| Inline executable-call multiset IoU | 0.6387 |
| Full prompt / cached tokens | 1,675,019 / 1,617,191 |
| Prefix token reuse | 96.55% |
| Accepted / proposed draft tokens | 2,584 / 16,369 (15.79%) |
| Effective prefill throughput, approximate | 8,023 tok/s |
| Server decode throughput, approximate | 12.86 tok/s |

There are no empty or failed responses. Every response finishes `tool_calls`;
all calls have parseable arguments and a nonempty `command` string. This is
output sanity, not an accuracy pass. The inline score measures executable-name
overlap; BFCL and numerical quality gates have not run. No tool is executed.
The earlier hipEngine/llama.cpp results are **autoregressive baselines**; the
speculative comparison is recorded above. Different outputs and KV settings,
one gufo pass and no gufo AR control prevent attributing its elapsed time to DFlash2.

**Tool output is buffered:** 132/140 first visible chunks arrive within 1 ms of
completion. Internal first-token timestamps average 1.762 s, but they do not
measure when the client receives tool arguments. Harness TPOT has only eight
observations and is not a usable comparison metric for these responses.

Effective prefill is `sum(full prompt tokens) / sum(prefill seconds)`, counting
the benefit of prefix reuse. Gufo logs rates rounded to 0.1 tok/s, not exact
prefill/decode durations. Reconstructing each duration from its token count and
rate yields approximately 208.775 prefill seconds and 792.701 decode seconds.
The resulting rate intervals from rounding alone are 8,021.57–8,024.63 effective
prefill tok/s and 12.812–12.913 decode tok/s; these are not repeat-confidence
intervals. Source inspection identifies prefill as time inside `Prefill()` and
decode as accumulated decode-step time. Cache restoration (1.667 s total),
admission, tokenization and snapshot work are outside those windows. Decode
windows differ across engines; these rates are not isolated kernel comparisons.

The [compatibility patch](gufo-qwen36-template.patch) removes the incorrect
inference that the shared 64-layer/5120-hidden/248320-vocabulary dimensions imply
a Qwen3.8 chat template. It changes no weights or kernels. Existing gufo template
tests and two Qwen3.6 Hugging Face byte-exact tool/history rendering checks pass
with the explicit options above; other template settings are not certified.
The preflight completed a real streamed tool call with actual accepted proposals.

[Compact artifact](../results/2026-09-25-gfx1151-gufo-dflash-edge-140.json)
records the pinned source, patch and binary hashes, target/draft identities,
commands, timing definitions and raw-file hashes. Raw root:
`~/gate-runs/gufo-edge-20260925`, with `screen-140/` for the measured run and
`preflight-dflash/` for the smoke. The draft is
`z-lab/Qwen3.8-27B-DFlash2-GGUF@2d9571f8ce46e151f61c6499c99dee6079e1d610`.
Apply the patch to upstream `98641a6503da2ec5d6dbb1888ddc95f8a3e13b28`, rebuild,
then use the artifact's server command and a copied config with a fresh report
directory. The measured runner command was:

```bash
cd ~/mlperf-edge-endpoints
.venv/bin/inference-endpoint benchmark from-config --config "$HOME/gate-runs/gufo-edge-20260925/screen-140/config.yaml"
```

## Stock gufo failed preflight: 2026-09-25

**Stock gufo does not load the comparison GGUF.** A separate clean checkout of
[`gufo-org/gufo` at `98641a6503da2ec5d6dbb1888ddc95f8a3e13b28`](https://github.com/gufo-org/gufo/tree/98641a6503da2ec5d6dbb1888ddc95f8a3e13b28)
built successfully on the same host. `gufo serve llm` exits with status 1 before
HTTP readiness:

```text
Qwen3.8 GGUF chat template SHA-256 is not a recognized pinned version: 55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c
```

`src/models/qwen/chat_template.cpp` classifies a model with 64 layers, hidden
size 5120 and vocabulary size 248320 as Qwen3.8, even without that name. It then
requires one of two template hashes. Our Qwen3.6 file reaches that rejection.
This is an observed loader refusal, not a finding that its tensor kernels
cannot execute Qwen3.6. In this initial attempt no template, model bytes or gufo
source was changed, and no HTTP generation, replay or BFCL evaluation ran.
The subsequent patched run is recorded in the preceding section.

The Qwen execution policy also declares FP16/FP32 attention KV, with FP16 as the
production default; it does not offer the BF16 KV used by the paired engines.
A future comparison must identify that difference rather than claim identical
KV settings. BF16 recurrent-state storage is a separate setting.

The first build hit a GCC 16 standard-header/HIP `__noinline__` macro conflict.
Adding `-DCMAKE_CXX_FLAGS="-include format"` resolved it without a source patch.
Build logs, CMake cache, executable, exact server command and loader log are in
`~/gate-runs/gufo-edge-20260925`; the
[worklog entry](../../worklog/entries/20260925T131432.066109Z-lhl-gufo-edge-screen-55564b.md)
records the executable hash and reproduction command. This failed stock attempt
has no speed or accuracy result; the compatibility-patched run is a separately
labelled comparison.

## Repaired run: 2026-09-25

**The 27-turn integration check passes.** On the same `gfx1151` host with
Qwen3.6-27B Q4_K_M, production, BF16 KV, 32K context and concurrency 1:

- 27/27 requests completed with one valid structured `bash` call each.
- No empty, missing, errored, dropped or length-limited responses; no raw tool
  markup in content. Every final reason was `tool_calls`.
- No negative first-receive-to-completion windows. Direct saved-request probes
  confirmed incremental content/argument delivery before completion.
- Route-counter deltas show 27 AR host-sampler requests and zero MTP requests.

The runner's measured phase was 225.36 seconds (3m45s); inline executable-call
IoU was 0.6574. These are integration diagnostics, **not** a BFCL score or an
Atlas decode-rate comparison. The output protocol and timestamp basis changed
from the failed attempt, so the shorter run is not an engine-speedup claim.
Client-visible token estimates still differ from engine token counts.

The repair aligns the host constraint with the prompted XML function format,
allows text before an automatic tool call, streams string arguments, and rejects
invalid or truncated live calls with an SSE error. XML close tags are no longer
used as stop sequences: that removed them before final parsing. Generic
outer-tag repair is also disabled for XML; its structural constraint owns safe
closure. Legacy JSON parsing remains available.

The harness patch makes COMPLETE use coordinator arrival time, matching
RECV_FIRST and RECV_NON_FIRST. It does not change workload rows, scoring, token
counts or metric formulas. Apply it to the pinned checkout before a repaired
repeat (set `HIPENGINE_REPO` to your hipEngine checkout):

```bash
cd ~/mlperf-edge-endpoints
git apply --check "$HIPENGINE_REPO/benchmarks/mlperf-edge/coordinator-timestamps.patch"
git apply "$HIPENGINE_REPO/benchmarks/mlperf-edge/coordinator-timestamps.patch"
```

The patch includes its session-test expectation; 57 harness session tests passed.
Do not apply it a second time to an already patched checkout. Use a new `RUN`
directory with the preparation and execution commands below.

[Repair artifact](../results/2026-09-25-gfx1151-mlperf-edge-repair.json) records
source hashes, patch identity, exact command, counters, output checks and raw
artifact hashes. Raw data is under
`~/gate-runs/mlperf-edge-repair-20260925/replay-2`; focused SSE captures are under
its sibling `closure-fix`. The server was stopped after validation. The fresh
140-turn screen is recorded above; the full BFCL gate has not run.

## First attempt: 2026-09-25

Host `gfx1151`, Ryzen AI Max+ 395 / Radeon 8060S, Qwen3.6-27B Q4_K_M,
production profile, BF16 KV, 32K window, one request in flight. The pinned
upstream harness completed all 27 requests in a 551.18-second measured phase
(9m11s), with zero HTTP failures, dropped turns or missing turns. Mean turn
latency was 20.414 seconds and inline executable-call IoU was 0.5278.

**The integration smoke did not pass its output/metrics checks.** One completion
(request turn 45, scored assistant turn 46) contained neither text nor a tool
call. A direct SSE reproduction generated 1,024 tokens and privately reported
`finish_details.reason=invalid_tool_call` after forced tool closure, but publicly
returned `finish_reason=stop` and no output. Two other harness completions reached
the output limit, and some responses emitted tool-like markup as ordinary text.
This is not a BFCL score or accuracy-gate pass.

The harness produced negative TPOT: 23 of the 26 nonempty turns have a completion
timestamp slightly earlier than their first-receive timestamp. In the pinned
`load_generator/session.py`, completion uses the worker's `resp.completed_at`
while first-receive uses coordinator arrival time. A direct SSE check also
showed the first replay response arriving as one content chunk near completion,
not token by token. These timings cannot be used for decode tok/s comparisons.

Before/after server counters show 27 additional host-sampler requests and zero
MTP requests. The tool probe reports `processed_argmax` with tool-call and
stop/repair constraints. Client token estimates also differ from engine counts;
retain both rather than silently substituting denominators.

[Compact artifact](../results/2026-09-25-gfx1151-mlperf-edge-smoke.json) records
commands, provenance, hashes and follow-ups. Raw reports and focused SSE captures
are under `~/gate-runs/mlperf-edge-smoke-20260925`. Fix and recheck these concrete
integration failures before using a larger run for a performance comparison.
No 140-turn screen, full replay or BFCL gate was run.

## Pinned inputs

| Input | Revision or identity |
| --- | --- |
| `mlcommons/endpoints` | `e71b928f8a72fd0c9d850dc5ddd8fd7760356354` |
| `unsloth/Qwen3.6-27B-GGUF` | `82d411acf4a06cfb8d9b073a5211bf410bfc29bf` |
| GGUF | `Qwen3.6-27B-Q4_K_M.gguf`, SHA-256 `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0` |
| `Qwen/Qwen3.6-27B` tokenizer | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Conversation | `pallets__flask-5014`, all 54 original rows / 27 generated turns |
| Filtered JSONL | SHA-256 `f8b7dcb551a6fc15e829bbeb96465f3f7bce7a4a3da9cb8f95a02c50f1eca85f` |

The model and tokenizer are stored under `~/models/mlperf-edge/{weights,tokenizer}`
on the recorded host. `download-manifest.json` alongside them records the exact
revision, size and hash of each file. Do not use moving model revisions for a
repeat comparison.

## Harness environment

Use a dedicated checkout at `~/mlperf-edge-endpoints` and Python 3.12. Keep these
evaluator dependencies separate from the hipEngine runtime. Inspect upstream's
installation requirements before installing. Once the checkout is pinned:

```bash
cd ~/mlperf-edge-endpoints
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python 'torch==2.14.0+cpu' \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python -e '.[bfcl]'
uv pip check --python .venv/bin/python
.venv/bin/python scripts/bfcl_import_smoke.py
uv pip freeze --python .venv/bin/python > installed-smoke.txt
```

The CPU PyTorch package is for the evaluator's transitive dependencies. Installing
`.[bfcl]` without it selected CUDA wheels on this host and exceeded the initial
30-minute setup budget. The CPU index alone lacked the required setuptools
version; the command above allows PyPI to supply dependencies. The final package
freeze is part of the run record. No Torch dependency is added to hipEngine.

## Prepare the diagnostic config

From the pinned Endpoints checkout, run this with its `.venv/bin/python`. It
preserves every selected JSONL line byte-for-byte and derives a performance-only
config from the upstream full config. `RUN` is a new output directory for each
attempt; do not overwrite previous evidence.

```bash
export RUN="$HOME/gate-runs/mlperf-edge-smoke-repeat"
.venv/bin/python - <<'PY'
import collections
import hashlib
import json
import os
from pathlib import Path
import yaml
from inference_endpoint.config.schema import BenchmarkConfig
from inference_endpoint.dataset_manager.factory import DataLoaderFactory

run = Path(os.environ['RUN'])
run.mkdir(parents=True, exist_ok=False)
example = Path('examples/11_Edge_Agentic_Example')
source = (example / 'agentic_coding_2.5h.jsonl').read_bytes()
assert hashlib.sha256(source).hexdigest() == 'b7da8c4ffbe1cabd79c4d8169e5201541e006364b4bfc016d01d893e20ee6f70'
lines = [line for line in source.splitlines(keepends=True)
         if line.strip() and json.loads(line)['conversation_id'] == 'pallets__flask-5014']
assert collections.Counter(json.loads(line)['role'] for line in lines) == {'user': 1, 'assistant': 27, 'tool': 26}
subset = run / 'smoke-27.jsonl'
subset.write_bytes(b''.join(lines))
assert hashlib.sha256(subset.read_bytes()).hexdigest() == 'f8b7dcb551a6fc15e829bbeb96465f3f7bce7a4a3da9cb8f95a02c50f1eca85f'
cfg = yaml.safe_load((example / 'online_edge_full_run.yaml').read_text())
cfg['name'] = 'hipengine-edge-integration-smoke-27'
cfg['model_params']['tokenizer_name'] = str(Path.home() / 'models/mlperf-edge/tokenizer')
cfg['model_params']['chat_template_kwargs'] = {'enable_thinking': False}
cfg['datasets'] = [d for d in cfg['datasets'] if d['type'] == 'performance']
cfg['datasets'][0]['path'] = str(subset.resolve())
cfg['datasets'][0]['agentic_inference']['num_trajectories_to_issue'] = 1
cfg['endpoint_config']['endpoints'] = ['http://127.0.0.1:8098']
cfg['report_dir'] = str((run / 'harness').resolve())
config = run / 'smoke.yaml'
config.write_text(yaml.safe_dump(cfg, sort_keys=False))
parsed = BenchmarkConfig.from_yaml_file(config)
loader = DataLoaderFactory.create_loader(parsed.datasets[0])
loader.load()
assert loader.num_samples() == 27
assert not parsed.datasets[0].agentic_inference.inject_tool_delay
print(config)
PY
```

This is replay data, not a list of commands to execute on the benchmark host.
The harness replays recorded tool results and scores generated calls.

## Serve and run

With an idle gfx1151 GPU, start this in the hipEngine checkout using its configured
ROCm/Python environment (not the evaluator venv):

```bash
python -m hipengine.server \
  --model "$HOME/models/mlperf-edge/weights/Qwen3.6-27B-Q4_K_M.gguf" \
  --backend hip_gfx1151 --served-model-name Qwen3.6-27B-Q4_K_M \
  --max-context-tokens 32768 --kv-storage bf16 --execution-profile production \
  --metrics prometheus --info --host 127.0.0.1 --port 8098 --log-level info
```

Cache and speculative policy use shipped defaults. Before timing, send a real
streamed tool request with `chat_template_kwargs.enable_thinking=false` and
`stream_options.include_usage=true`; confirm a parsed call, valid usage and no
reasoning chunks. A separate diagnostic request may set
`stream_options.include_hipengine=true` to expose sampler/route decisions. The
initial probe used `Use the bash tool to run pwd. Do not explain.`, temperature
0, seed 42 and a 128-token ceiling; it did not execute the generated command.

Then, from the Endpoints checkout, using the same `RUN`:

```bash
curl -fsS http://127.0.0.1:8098/metrics > "$RUN/metrics-before.txt"
.venv/bin/inference-endpoint benchmark from-config --config "$RUN/smoke.yaml"
curl -fsS http://127.0.0.1:8098/metrics > "$RUN/metrics-after.txt"
```

Preserve the runner exit status, full log, resolved config, per-turn outputs,
inline score, token metrics, server logs/capabilities and before/after counters.
A successful process exit alone does not establish 27 completed turns. Check
missing turns and error records explicitly. Do not infer MTP execution from its
startup capability: the initial tool probe used AR `processed_argmax` with
`tool_call_constraint` and stop/repair sequences requiring processed logits.
Do not remove these controls to make an ostensibly equivalent benchmark faster.
