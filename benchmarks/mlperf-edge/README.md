# MLPerf Edge Agentic integration smoke

This directory documents a **27-turn integration diagnostic**, not a full MLPerf
result or BFCL accuracy pass. The campaign and comparison rules are in
[`docs/campaigns/MLPERF-EDGE-AGENTIC.md`](../../docs/campaigns/MLPERF-EDGE-AGENTIC.md).
The first attempt used the upstream runner unchanged. The repaired run uses
only the coordinator-timestamp patch documented below.

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
its sibling `closure-fix`. The server is stopped. The next campaign step is the
fixed 140-turn screen; neither that screen nor the full BFCL gate has run.

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
