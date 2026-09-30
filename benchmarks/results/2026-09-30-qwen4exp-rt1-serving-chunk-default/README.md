# RT-1 promotion: the serving prefill chunk cap is not the fairness bound

**Default changed:** `DEFAULT_MAX_PREFILL_CHUNK_TOKENS` 256 -> 1024 in
`hipengine/generation/engine_loop.py`. This is the serving/CLI path's chunk
size, so the engine now runs the chunk size RT-1 measured and the X-6 numerical
gate certifies.

This directory records a **default change and its justification**, not a new
measurement. The measurement it rests on is
[`../2026-09-29-qwen4exp-rt1-256-vs-1024/`](../2026-09-29-qwen4exp-rt1-256-vs-1024/README.md),
which is complete and unchanged.

## Why the recorded blocker does not hold

RT-1's ledger row recorded a concrete blocker: `max_prefill_chunk_tokens` "is a
**C4 scheduler knob bounding per-tick prefill work so decode is not starved** ...
Raising it 4x makes each prefill tick up to 4x longer", and the -29% to -36% is
prefill throughput, so the serving default was blocked on an
interleaving/fairness measurement.

**That premise is wrong, and the code it is about says so.**
`EngineLoop._run_token_budget_round` bounds per-tick prefill work with
`round_prefill_token_budget` and only splits that already-bounded budget into
chunks:

```python
prefill_budget = self.round_prefill_token_budget          # 1024
while prefill_budget > 0 and self.scheduler.has_prefill_work():
    chunk_size = min(self.prefill_chunk_size, prefill_budget)
```

So the per-tick fairness bound is the **round budget** (1024). The chunk cap can
only subdivide it:

| chunk cap | tokens per round | chunks per round |
| ---: | ---: | ---: |
| 256 (old) | 1024 | 4 x 256 |
| 1024 (new) | 1024 | 1 x 1024 |

**Per-tick work is identical, so there is no tick to lengthen.** A cap below the
round budget cannot make the engine fairer - it can only multiply per-chunk
overhead, which is the whole of the measured win.

Independent corroboration: `ResidentEngineLoop.__init__`'s own
`prefill_chunk_size` default was **already 1024**. Only
`EngineLoopConfig.max_prefill_chunk_tokens` was pulling it down to 256, so the
two knobs were inconsistent and this change aligns them.

## The measurement, restated

Prefill harness (`qwen4exp_profile_gap.py --mode prefill`), 12 canonical cases
(4 categories x 3 prompt sizes), 3 reps per arm, one process per arm, **both arms
in one run**. Host: AMD gfx1151, Qwen3.8-Flash-Next UD-Q4_K_XL, production
profile.

| Prompt tokens | Range of deltas across the 4 categories |
| ---: | ---: |
| 512 | -29.0% to -29.4% |
| 1024 | -32.9% to -34.4% |
| 4096 | -32.9% to -35.7% |

- **12/12 cases faster, 12/12 arms separated.**
- **`token_id` identical on 12/12.**
- **`logits_sha256` identical on 0/12** - this is an arithmetic change against
  the old default, not a pure scheduling one. It is admissible because chunk
  1024 **is** the configuration the X-6 numerical gate certifies (it hashes
  bit-identically at 4096), not because it is bit-identical to 256.
- **Peak cost +1.24 GB** at p1024/p4096, +0.77 GB at p512. The workspace is
  `min(chunk, max_seq_len)`, so a 512-token request resolves to the same 768
  rows either way and short requests are unaffected in time and memory.

## What is NOT claimed

- **No serving-path timing.** A live-server A/B was attempted and could not run
  in this environment on either backend:
  - `gguf_live_server_bench.py` requires `--model` to be a single file
    (`model.is_file()`), so the 4-shard Qwen3.8-Flash-Next directory cannot be
    loaded, and its own `--prefill-chunk-size` default (256) means it cannot
    exercise the engine's serving default without an explicit override anyway.
  - With the harness's default single-file model (Qwen3.6-35B-A3B UD-Q4_K_M):
    `hip_gfx1100` -> `HIP error 98: invalid device function` (also without the
    cached-build gate), and `hip_gfx1151` (the backend `QWEN4_EXP_BACKEND`
    names) -> timeout waiting for the engine service.
- **No decode-latency-under-concurrency claim.** The measurement is prefill
  only. The specific protection the old default was thought to provide - a
  shorter prefill tick - is provided by `round_prefill_token_budget`, which is
  unchanged; but no c2/c8 prefill-interleaving measurement was taken.

## Validation

- `tests/test_unit_engine_loop_prefill_chunk.py` (new) pins the relationship
  rather than a bare number: the cap must equal the round budget, the per-round
  token count is unchanged by the cap, and the cap strictly reduces
  fragmentation.
- `test_engine_loop_cli_env_defaults_match_docs` now asserts the config default
  against the constant instead of the literal 256.
- 432 tests green across the affected bundle (engine loop, batch scheduler,
  engine-loop burst, llm_generate, concurrency2 token budget, KV pool, KV
  device pool, KV resource ledger).

[Worklog entry](../../../worklog/entries/20260930T011826.957349Z-lhl-qwen4exp-rt1-serving-chunk-default-3c5235.md)
