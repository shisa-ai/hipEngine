# The repair rate is 0.7119%: quantified corroboration for EG-3, not a reopening

EG-3 concluded from indirect evidence that the sparse exact repair is *"limited by the
latency of its dependent memory chain"*: more blocks did not help (17,920 -> 262,144
blocks: 3.05 -> 3.18 ms), the bitmap variant moved 3.3x fewer bytes and was 22% slower,
fewer instructions at equal occupancy was slower, and L2 residency gained only 13.8%.

This packet supplies the **direct number EG-3 inferred**: the repair's incidence and its
effective throughput. It **corroborates EG-3 and does not reopen it.**

## The measurement

The engine has a built-in opt-in repair-rate readback
(`HIPENGINE_QWEN4_EXP_RISK_DIAGNOSTICS`, consumed by `qwen4exp_profile_gap.py
--risk-diagnostics`). One production-profile `code-p4096` prefill on the usual host:

| quantity | value |
|---|---:|
| calls | 94 |
| outputs | 4,928,307,200 |
| risky outputs | 35,083,610 |
| **aggregate repair rate** | **0.7119%** |
| max per-call rate | 0.7797% |
| over-capacity calls | **0** (the risk queue never overflowed) |

## What that says about the mechanism

| derived | value |
|---|---:|
| outputs per call | 52,428,800 |
| risky per call | 373,230 |
| compact rows per call | 40,960 |
| weight bytes re-read per repair (K=2560 Q4_K) | 1,440 B |
| bytes per call | 537.5 MB |
| bytes over all calls | **50.5 GB** |
| **effective scattered bandwidth** | **40.9 GB/s** |
| time per repair | 35.2 ns |

**40.9 GB/s is far below this part's memory peak.** Combined with EG-3's finding that
moving 3.3x fewer bytes made the kernel *slower*, that rules out a bandwidth limit and
**confirms the latency-bound dependent chain**. The repair is not slow because it moves
too many bytes; it is slow because each slot's work is a serial chain.

## The prize, quantified - the reason this matters for planning

| route | fast | repair | ideal repair at fast-path efficiency | disproportion | saving |
|---|---:|---:|---:|---:|---:|
| q4_k (**rate measured**) | 2,713.953 ms | 573.211 ms | 19.32 ms | **29.7x** | **553.9 ms = 2.44%** |
| q5_1 (rate *not* observed) | 1,482.958 ms | 661.210 ms | 10.6 ms | - | 650.7 ms = 2.87% |
| **combined ceiling** | | | | | **1,204.5 ms = 5.31%** |

The repair does **0.71% of the work in 22.7% of the MoE GEMM time**.

**Scope discipline**: the 0.7119% is measured for the **q4_k route only**
(`q4_iu8_exact:expert_gate_up`). The census's 1,234.4 ms covers q4_k (573.2) *and* q5_1
(661.2). The q5_1 row above is an **inference**, explicitly labelled as such, and the
5.31% combined figure is a **ceiling that assumes both routes behave identically**. The
measured, defensible claim is the **2.44%** q4_k row.

## Why this does not reopen EG-3

EG-3 closed the whole "reduce per-slot scalar work" class *by measurement*, and its
reopen condition was "a measured mechanism (registers, instruction count, occupancy)".
All three were falsified there. This packet **adds no new mechanism** - it measures
incidence and effective bandwidth, both of which agree with EG-3's conclusion.

What it does add is the **size of the prize**, which EG-3 did not state: 2.44% measured,
5.31% ceiling. That is the number a future attempt should be judged against, and it is
the number MX-1's alternative should be compared with.

**MX-1 comparison**: MX-1 would remove the repair entirely (up to 5.44% of the window)
but changes the arithmetic and therefore needs the full production numerical gate. A
repair-efficiency fix would keep the arithmetic and need no such gate - but EG-3's
evidence says no known mechanism achieves it.

## Limits

- One prompt (`code-p4096`), one model, one host, one execution profile, one repetition.
  The rate is a property of the risk criterion and the data, and is not expected to be
  constant across prompts; this packet does not claim it is.
- The diagnostic observes the **q4_k** route only. The q5_1 repair rate is unmeasured.
- The "ideal repair" figure assumes the repair could reach the fast path's per-output
  efficiency. That is a **bound**, not a proposal, and EG-3's evidence suggests it is not
  reachable by the mechanisms tried so far.
- Per-repair byte count assumes K=2560 Q4_K (1,440 B) for the gate/up half; the down half
  has a different K, so the 50.5 GB is an estimate at that K.
- Diagnostic only. No runtime default changed.
