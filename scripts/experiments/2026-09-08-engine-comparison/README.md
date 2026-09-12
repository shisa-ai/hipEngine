# September 8 Engine Comparison Scripts

This is a byte-for-byte archive of the 19 session scripts that were left in
`/tmp` after the hipEngine/nasone32/strix-llama.cpp comparison. `SHA256SUMS`
records their original bytes. Keep these copies unchanged; develop a new
campaign separately rather than silently changing the historical procedure.

The reusable external driver is
[`scripts/llamacpp_raw_suite_bench.py`](../../llamacpp_raw_suite_bench.py);
it and its tests were committed during the original session. The hipEngine
drivers are [`qwen36_dense_gguf_suite.py`](../../qwen36_dense_gguf_suite.py)
and [`qwen35_gguf_bench.py`](../../qwen35_gguf_bench.py).
The [report](../../../benchmarks/results/2026-09-08-rx7900xtx-engine-comparison.md)
and [compact evidence](../../../benchmarks/results/2026-09-08-rx7900xtx-engine-comparison.json)
are committed independently of these scripts.

## Inventory

| Files | Role |
| --- | --- |
| `engine_compare_runner.py` | Final clean-source, idle-VRAM and KFD process-group monitor. Owns child process groups and 20 ms memory sampling. |
| `run_engine_compare_final.py` | Final serial orchestration: true AR/MTP, PP/TG, capacity, strix source refresh/build and retries. |
| `publish_engine_comparison.py` | Actual publisher of the committed three-engine JSON: joins output IDs, category/heldout metrics and capacity evidence. |
| `validate_engine_comparison.py` | Final artifact integrity, denominators, output hashes and completeness checks. |
| `nasone_next.py`, `nasone_clean.py`, `nasone_adapt_real.py` | Earlier capacity/GDN screens, clean repeats and explicit adaptive-floor trials. |
| `nasone_hip_ar.py` | AR-only diagnostic used while native MTP was broken; final comparison uses the committed AR/MTP suite instead. |
| `nasone_hip_remaining.py`, `nasone_finish.py`, `nasone_pure.py` | Earlier hipEngine capacity/AR runs and pure-INT8 boundary probes. |
| `nasone_ownership.py` | Earlier standalone KFD ownership logger. The final runner integrates ownership checks. |
| `nasone_summarize.py` | Superseded intermediate publisher, not the producer of the final comparison. |
| `nasone_mtp_debug.py`, `nasone_mtp_eager_debug.py`, `nasone_mtp_layer_debug.py` | Process-local instrumentation for fallback, eager/native and first-NaN localization. Deliberately failing debug runs, not timing tools. |
| `nasone_pipeline.py`, `nasone_last.py`, `nasone_wait_adapt.py` | Historical PID-wait launchers. **Do not execute:** the stored PIDs are obsolete. |

## Important Execution Limits

These are historical scripts, **not a portable one-command benchmark package**.
Several execute immediately on import. Do not import them to inspect constants.
Do not run Python with `-O`: the historical checks use assertions.

The final runner hardcodes the measured detached tree
`/tmp/hipengine-engine-compare-6c01f1f1c`, raw directory
`/tmp/engine-compare-final`, GPU1, PCI `0000:10:00.0` and KFD GPU ID `33912`.
The orchestrator expects configured external clones at
`/tmp/llama-rdna3-nasone32` and `/tmp/strix-llama-head`, model weights under
`/models/gguf`, and `/tmp/hipengine-hipcc-version.txt`.
Required HIP/JIT caches must be prebuilt outside measured regions.

The orchestrator fetches and checks out strix HEAD when its lane starts.
Use only disposable clones, never someone else's checkout. To reproduce the
historical revision, use the pinned revisions/build commands from the artifact,
not today's HEAD.

Resume checks only completed filenames/status, **not full model, command,
source or environment identity**. Always use a new raw directory for a new
revision or configuration. Never reuse the old directory to measure current
main. The monitor checks HIP memory ownership; the original final script
does not fully clean up its process group for every interruption/exception.
Inspect and terminate only owned children before restarting an interrupted run.

## Running Another Comparison

Use the committed drivers directly for a new campaign, or make an explicitly
configured successor of the archived final runner. Required setup:

1. Select a clean hipEngine commit/worktree and record it. Confirm physical
   GPU identity and exclusivity before each expensive run.
2. Clone and pin both external projects in disposable directories. Build
   HIP/gfx1100 using the artifact's exact CMake flags and prebuild hipEngine
   caches with a matching compiler-version file.
3. Choose a fresh raw directory; run full-category true AR and B3 MTP with
   identical model, raw prompt IDs, output horizon and KV policy. Retain the
   server response IDs, not decoded-text token estimates.
4. Run PP/TG and actual prefill-plus-decode capacity separately. Preserve
   failed attempts and require actual allocation-error evidence for OOM.
5. Generate a new dated artifact from that campaign's own inputs and validate
   it before updating the scoreboard. Do not relabel the September 8 publisher
   output as a new campaign.

Example single-engine invocation from the selected clean hipEngine tree,
after the device and idle checks (substitute the new paths):

```bash
HIP_VISIBLE_DEVICES=1 python3 scripts/llamacpp_raw_suite_bench.py \
  --server /tmp/new-nasone32/build/bin/llama-server \
  --source /tmp/new-nasone32 \
  --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  --mode ar --outputs 25 --repetitions 3 --context 1024 \
  --output /tmp/new-comparison/nasone32-ar.json
```

Repeat with `--mode mtp` and a distinct output. For actual nasone32 adaptive
depth use `--mode adaptive --adaptive-min 1`; its historical default minimum
3 plus maximum 3 was fixed depth. The full campaign commands are preserved in
`run_engine_compare_final.py` and the result artifact.

## Analysis Reproduction And Retention

`publish_engine_comparison.py` reads `/tmp/engine-compare-final/*` **and**
earlier `/tmp/nasone-*.json`/logs for capacity and adaptive screens. It also
reads the compiler file and current prompt fixture. It hardcodes September 8
model/hardware/revision assertions and overwrites the historical tracked JSON
on execution. Do not run it against another campaign or overwrite the published
artifact just to inspect results.

`validate_engine_comparison.py` is safe to execute: it only reads the published
artifact at its original absolute repo path. It requires neither GPU nor raw
logs:

```bash
python3 scripts/experiments/2026-09-08-engine-comparison/validate_engine_comparison.py
```

The archive integrity test also checks syntax without executing any script:

```bash
python3 -m pytest tests/test_unit_engine_comparison_script_archive.py -q
```

Model weights, external source clones, compiled libraries, JIT caches, profiler
dumps and raw benchmark logs are intentionally not committed. The raw `/tmp`
inputs are not durable; losing them prevents exact raw-to-summary regeneration
but does not lose the committed compact evidence, complete output ID rows,
commands or this analysis source. Preserve raw inputs outside Git separately
when raw regeneration is required.
