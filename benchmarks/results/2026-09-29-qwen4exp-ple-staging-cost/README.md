# PL-2: the PLE staging/H2D copy cost, measured

**Verdict: PL-2 is rejected by its own cost gate.** The entire addressable cost -
the H2D copy plus the staging memcpy - is **~2.65 ms per complete request**,
against a **9,055.58 ms** production `code-p512` wall: **0.029%**. PL-2's gate
says "if material, test one bounded double-buffer path"; 2.65 ms of 9.06 s is not
material, so no candidate path was built.

## What was measured

Two independent measurements, neither requiring the model:

`scripts/qwen4exp_ple_h2d_cost.py` - the pinned host->device copy at the real
staging shapes (median of 15, 3 warmups, device-synchronized):

| copy | bytes | median | effective |
|---|---:|---:|---:|
| prefill (512 rows x 2560 fp32) | 5.00 MiB | **67.60 us** | 77.6 GB/s |
| decode (1 row x 2560 fp32) | 10.00 KiB | **16.97 us** | 0.6 GB/s (latency-bound) |
| **per request** (128 decode + 1 prefill) | **6.25 MiB** | **2.240 ms** | |

`scripts/qwen4exp_canonical_ar_bench.py ... --ple-telemetry --case-id code-p512`
- the in-engine attribution, one production request (512-token prefill, 128
decode transitions):

| term | value | share of request |
|---|---:|---:|
| gather + dequantize | **0.974 s** | 10.8% |
| staging copy | 0.409 ms | 0.005% |
| H2D (decode only, pre-fix) | 11.444 ms | 0.126% |

## Why the copy is not worth overlapping

**1. The payload is tiny and mostly latency-bound.** 6.25 MiB per request. The
prefill copy is the only bandwidth-shaped one (5 MiB at 77.6 GB/s = 67.60 us);
the 128 decode copies are 10 KiB each at **0.6 GB/s** - pure per-call latency,
16.97 us apiece. Against a 61.75 ms decode step, one decode copy is **0.027%**.

**2. The dominant PLE term is not a copy.** Gather/dequantize is **0.974 s**, ~370x
the entire copy cost, and it is CPU work against the mmap'd PLE table.
Asynchronous H2D staging does not touch it. The 2026-09-02 telemetry artifact
reached the same ranking independently ("ranks gather/dequant above staging copy
and H2D").

**3. Perfect overlap would recover 0.029%.** 2.65 ms of 9,055.58 ms.

## A telemetry gap found and fixed

The engine's **prefill** PLE copy did not call `record_h2d`, so PLE telemetry
counted only the 128 decode copies and **understated the per-request H2D payload
4x**. Measured before and after the fix:

| | h2d_bytes | |
|---|---:|---|
| before | 1,310,720 | 1.25 MiB (decode only) |
| after | **6,553,600** | **6.25 MiB** |

6,553,600 = 5,242,880 (prefill, 512 x 2560 x 4) + 1,310,720 (128 x 10,240) -
matching the independently measured shapes exactly. Without this fix the next
person cost-gating PLE I/O would read a 4x-low number. The added
`record_h2d` is guarded exactly like the decode path's, so it is a no-op with
telemetry disabled and production is unaffected.

## What this does not establish

- **Not a speed claim.** `performance_claim: false`. This measures the existing
  synchronous copy and rejects a candidate; it does not measure an alternative.
- **The instrumented walls are noisy and excluded from performance evidence.**
  `gather_dequant_wall_ns` moved 0.974 s -> 1.635 s between two identical
  telemetry runs, and `h2d_wall_ns` moved 11.444 -> 11.988 ms for an added
  67.60 us copy. The **bytes** are deterministic and are the reliable evidence;
  the instrumented nanosecond counters are not. The copy figures above come from
  the standalone uninstrumented measurement, not from the telemetry counters.
- **One workload shape** (p512 + 128 decode) on one host. A much longer prompt
  would scale the 5 MiB prefill copy linearly - but so does the request, and the
  copy is 77.6 GB/s against a ~256 GB/s machine.
- The boundary excludes the gather/dequantize, which is where the real PLE cost
  is.
