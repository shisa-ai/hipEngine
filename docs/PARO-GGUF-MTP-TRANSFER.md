# PARO Follow-Ups from GGUF/MTP Work

Last updated: 2026-07-07.

This is the active TODO list for applying recent GGUF/MTP server and verifier
lessons to PARO. The key split is:

- GGUF quant kernels mostly do not port directly to PARO (`Q*_K`, Q8/dp4a,
  Q6_K rowtile LM-head, GGUF T16 layouts).
- Server shape handling, warmup, timing attribution, route caps, and verifier
  lifecycle ideas do apply and should be tested against PARO with PARO evidence.

## Immediate PARO Server Queue

| Priority | Item | Status | Evidence / next action |
| --- | --- | --- | --- |
| P0 | Measure PARO OpenAI server c=1/2/4/8 vs the direct retained c>N harness. | Open | Run the same 512/128 protocol used by the README concurrency table, but through `hipengine.server`; capture `scheduler_token_chunks` telemetry and `/ready` startup diagnostics. |
| P0 | Prove whether the PARO server path is using native packed prefill plus native c>N decode or falling back to the serial slot bridge. | Partially wired | `last_batch_generation` now preserves `batch_execution` JSON when available; output telemetry already carries `execution_path`, `native_compact_prefill`, `native_caware_decode`, and `serial_decode_fallback`. |
| P1 | Warm PARO server startup c>N shapes instead of only reserving c=1-style scratch. | Opt-in hook landed | Set `HIPENGINE_QWEN35_SERVER_STARTUP_NATIVE_BATCH_WARMUP=1` to exercise packed prefill widths 2/4/8 up to `max_batch_size`. Native decode warmup also requires `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1`. |
| P1 | Tune PARO server grouping/cap policy separately for gfx1151. | Open | GGUF AR/MTP retained a four-request backend group on gfx1151. PARO currently uses the global `--max-active-requests`; sweep caps 2/4/8 before promoting any route-specific default. |
| P1 | Add PARO server stage buckets comparable to GGUF MTP server buckets. | Open | Minimum buckets: prompt prefill, decode layer wall, projection dispatch, sampler/LM-head, graph replay, host readback, scheduler/queue wall. |
| P2 | Reconcile PARO concurrency docs and rollup wording. | Open | `benchmarks/README.md` has an accepted c=8 retained PARO row; `docs/CONCURRENCY.md` still warns that production retained c>N is not fully claimable. Review the current evidence and update one source of truth. |

## Speculative PARO Queue

| Priority | Item | Status | Evidence / next action |
| --- | --- | --- | --- |
| P2 | Audit PARO MTP/DFlash verifier commit/scatter paths against GGUF deferred packed verifier scatter. | Open | Look for accepted-row-only state/KV/hidden commits and avoided scatter/copy for rejected tails. This is the most likely portable verifier-lifecycle win. |
| P2 | Compare PARO MTP/DFlash target verifier small-B shape policy against GGUF exact and llama-compat. | Open | GGUF exact and llama-compat both became faster than the older PARO speculative path by avoiding bad small-B WMMA shapes and tightening LM-head/sample. PARO needs a fresh bucketed verifier profile before copying any mechanism. |
| P3 | Carefully review the GGUF llama-compat verifier against PARO MTP/DFlash. | Open, last step | Do this after PARO AR server measurement/warmup/cap work. The review should map draft, target verify, LM-head/sample, state capture, commit/rollback, and rejection-tail handling from GGUF exact and llama-compat to PARO MTP and DFlash one by one. |

## Non-Portable GGUF Wins

These should stay as reference evidence, not direct PARO tasks:

- GGUF `Q*_K`/Q8/dp4a/T16 rowtile kernel bodies.
- GGUF Q6_K rowtile verifier LM-head.
- llama.cpp compatibility precision trades.
- GGUF-specific no-copy GDN verifier capture state layout.
- GGUF MTP direct partial commit semantics, except as a speculative lifecycle
  pattern to compare against PARO verifier commit/rollback.
