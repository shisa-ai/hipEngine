# The decode step's non-kernel remainder is 14,293 environment-variable lookups

`DE-11` in `docs/QWEN4EXP-OPTIMIZATION-LEDGER.md` records that the canonical
bench's decode wall is 47.66 ms/step at p512 and 57.69 at p4096 while the
profiled kernel sum is 46.19 and 47.13, so the non-kernel remainder grows
**1.47 -> 10.56 ms/step**. That number says how much of a step is not device
kernel time but not what it is. This directory is the attribution, measured with
the bench's own loop.

## Instrument

`scripts/qwen4exp_canonical_ar_bench.py --host-profile` runs one **extra**
`runner.reset()` + `prefill()` + 128-step pass after the measured repetitions,
under cProfile, and records the top 30 functions by tottime with call counts,
shares and their top callers into `artifact["host_profiles"]`. The pass is extra
and resets first, so it neither grows the sequence nor perturbs the samples.
cProfile inflates absolute times, so read the ordering and the shares: the
profiled pass is 61.33 ms/step against an unprofiled wall of 56.40.

## Result

**The single largest call group in a decode step is `os.environ` access.**
14,293 lookups per step, every one of them going through four Python frames:

| function | calls/step | share |
| --- | ---: | ---: |
| `os.__getitem__` | 14,293 | 9.5% |
| `os.encode` (the key encode) | 14,293 | 6.7% |
| `collections.abc.get` | 14,293 | 5.7% |
| `str.encode` | 14,293 | 2.9% |
| **wrapper block** | | **24.8%** |

Plus `str.strip` 19,973/step (2.6%), `_resolution_environment_identity` 2.6%,
`build_hip` 1.9%, `_q8_dense_wide_layers` 5.5%, and 5,340/step of `os.decode`.
**About a third of the profiled host time is environment-variable handling.**

### Who reads the environment

`pstats` caller attribution for `collections.abc.get`, per decode step:

| caller | lookups/step | share of all |
| --- | ---: | ---: |
| `_resolution_environment_identity` (`core/build.py:486`) | **7,656** | **53.6%** |
| `launch_gguf_linear` (`runtime/gguf_linear.py`) | 1,090 | 7.6% |
| `_q8_dense_wide_enabled` | 545 | 3.8% |
| `_q8_dense_wide_layers` | 545 | 3.8% |
| `_env_wmma_prefill_enabled` | 545 | 3.8% |
| remainder (below the top-5 cutoff) | ~3,912 | 27.4% |

`build_hip` is called **957 times per decode step** - once per kernel launch -
and each call re-derives its resolution identity from **eight** environment
variables. Micro-benchmarked on this host, an 8-key identity costs **2.281 us**
through `os.environ.get(str)`, so that term alone is **2.18 ms/step**.

`_q8_dense_wide_layers` is called **545 times per step** and re-parsed its
variable on every call: `raw.split(",")` plus a `part.strip()` and an `int(part)`
per entry. The production scope names 32 layers, so that is **~17,000 `str.strip`
calls and ~17,000 `int()` calls per step** - which is where the measured
19,973/step comes from - for a value that cannot have changed.

## Files

- `code-p4096-callers.json` - p4096 with caller attribution (the authoritative
  run; 56.398 ms/step measured, 61.331 profiled).
- `code-p4096.json`, `code-p512.json` - the first run, without callers.

## Limits

- cProfile tottime is not a timing measurement. The **call counts** and the
  caller attribution are exact; the shares are indicative and the profiled pass
  is 8.7% longer than the unprofiled wall.
- One host (`gfx1151`, machine-id `55ea6c509d0b49eea8de7094a1023668`), one
  model, one quant, production profile, chunk 1024, warm PLE, one case per
  process.
- `wall - kernel_sum` uses a kernel sum from a profiled capture whose own note
  records that the profiler inflates absolute times, so the 10.56 ms/step
  remainder is an estimate; the call counts in this directory are not.
