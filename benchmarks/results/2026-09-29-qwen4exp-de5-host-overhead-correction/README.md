# DE-5 correction: the "~65% host/launch overhead" is a cross-run artifact, and the synchronous copies total 10.2 KiB

**Date:** 2026-09-29
**Host:** Radeon 8060S (gfx1151), hostname `gfx1151`, machine-id
`55ea6c509d0b49eea8de7094a1023668`
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL (unsloth split GGUF)
**Evidence:** re-analysis of two pre-existing artifacts plus a same-host
same-model decode figure; **no new measurement was run**
**Tree:** `2454f4908`

## What is being corrected

`DE-5`'s appended `20260929T` note reads, in part:

> **Decode kernel time is 54.218 ms/step** over 1,771 kernels/step and 740 roles, **100%
> attributed** … **The governing new number for this row is that DECODE IS NOT KERNEL-BOUND**: the
> unprofiled wall for the same transition is median **153.0 ms** (spread 1.30x), so the kernels are
> only **~35%** of a decode step and **~65% is host/launch overhead**, with 48 graph launches
> carrying 1,771 kernels.

**Both halves of that inference are unsound, and the same-host history contradicts it.**

## Finding 1 - the two numbers come from different runs

| figure | value | source file | instrumentation |
|---|---:|---|---|
| step wall | **153.04 ms** | `decode-warmup.json` | `profile: false`, `role_markers: false` |
| kernel sum | **54.218 ms** | `decode-profiled.json` | `profile: true`, `role_markers: true` |
| *that run's own wall* | *225.79 ms* | `decode-profiled.json` | same |

The kernel sum is taken from the **profiled** run; the wall is taken from the **unprofiled** run.
The profiled run's own wall median was **225.79 ms** (min 82.0, max 228.1), not 153.0 ms.

Subtracting a kernel sum measured under one instrumentation state from a wall measured under
another and calling the remainder "host/launch overhead" is not a decomposition. It also has a
visible tell: the profiled run's kernel sum (**54.218 ms**) is *larger than the true step time*
derived below, which is only possible because profiling inflates it.

## Finding 2 - the same host, model and case decode ~3x faster than this harness's step

`benchmarks/results/2026-09-13-framework-qwen4exp-strix-journey-baselines.json` records, on the
**same machine** (`host.machine_id` `55ea6c509d0b49eea8de7094a1023668`, `hostname` `gfx1151`) and
the **same model** (`unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL`, fingerprint
`fb1f2fbf73d588c9ac27f24bade5663bd3da8ac1862f62ee5bf457578a88ec53`):

| arm | metric | value |
|---|---|---:|
| `he-ar` | `cases.code-p4096.decode_tok_s.median` | **19.134 tok/s** |
| `he-ar` | `shapes.4096.decode_tok_s_weighted` | **19.166 tok/s** |

**19.134 tok/s is 52.26 ms/token** - decode-only, full serving request, production defaults.

So for the same host, model and case:

| quantity | value | ratio to the 09-13 decode |
|---|---:|---:|
| 2026-09-13 hipEngine AR decode (measured) | **52.26 ms/token** | 1.00x |
| x4 profiled kernel sum | 54.218 ms/step | 1.04x |
| **x4 unprofiled step wall** | **153.04 ms/step** | **2.93x** |

**The kernel sum agrees with the independently measured step time; the harness's wall does not.**
That is the opposite of the appended note's reading: the wall is the outlier, not the kernel sum.

A 2.93x *regression* between 09-13 and 09-29 would be a headline event, and the rollup shows only
improvements in that window (2026-09-26 F16 activation staging +3.4% category means, 2026-09-27
full-attention decode-context +2.11%, 2026-09-28 four-token QK unroll +2.65%). Nothing in
`benchmarks/CHANGELOG.md` records a decode regression of any size.

## Finding 3 - the row's own target is measured negligible

`DE-5`'s action is *"target the remaining synchronous copies and sampling path, not graph
coverage"*. The harness's runtime census, **identical across all three repetitions**, gives the
per-step copy budget exactly:

| operation | calls | bytes |
|---|---:|---:|
| `memcpy` (synchronous) | 3 | **10,440** |
| `memcpy_async` | 12 | 6,144 |
| `memset` | 1 | 16 |
| `graph_launch` | 48 | - |

**The entire synchronous-copy budget is 10.2 KiB per step.** At this host's documented ~221 GB/s
practical read roof that is ~47 ns of transfer; even charging ~5 us of launch cost per call, all
three synchronous copies together are on the order of **15 us**. Against a true ~52 ms step that is
**~0.03%**. There is no meaningful win in the row's stated target.

The "sampling path" half is not exercised by this harness at all - it runs
`capture_logits=False` with `token_id_resident=True`. The row's original mechanism estimate was
"~3.6 ms/token above the runner loop", which is **~6.9%** of a 52.26 ms step - worth attention, but
two orders of magnitude larger than the copies the action names, and not what the 65% figure
describes.

## Finding 4 - the row already said this

`DE-5`'s **mechanism** cell, written before the appended note, reads:

> **Serving-wrapper host time, ~3.6 ms/token** above the runner loop (logits D2H, sampling,
> bookkeeping) after the R7 one-H2D promotion. **Layer graphs are measured negative: the launch
> host time is already overlapped**

"Layer graphs are measured negative: the launch host time is already overlapped" is the same
conclusion this correction reaches. The appended note's 65% claim contradicts the row's own
mechanism cell, which was based on a direct measurement.

## Verdict

- **The "~65% is host/launch overhead" figure does not hold** and should not drive work. It is a
  cross-run subtraction, and the same-host same-model decode figure shows the wall it rests on is
  inflated ~2.9x.
- **The row's stated target - the remaining synchronous copies - is measured at 10.2 KiB and ~15 us
  per step (~0.03%).** Nothing to win.
- **The row's original scope remains the right one**: the wrapper's ~3.6 ms/token of host time
  (logits D2H, sampling, bookkeeping), which is ~6.9% of a 52.26 ms step. That is a real but modest
  target, and it is *not* what the 65% figure describes.
- **The harness, not the engine, needs the follow-up.** The most likely mechanism is
  `snapshot_restore_between_repeats: true`: the harness restores request state before every timed
  step, and the QSA indexed-sparse path carries rewindable index state
  (`Qwen4ExpQSAIndexDeviceState.append` short-circuits on `if complete <= first: return`), so a
  restore plausibly forces work the steady-state serving path does not repeat. That is a hypothesis,
  not a measurement.

## Limits

**No new measurement was taken.** Every figure is quoted from artifacts already in-tree. The 09-13
decode figure is a different protocol from the x4 harness (full serving request vs one
snapshot-restored step, `capture_logits` on vs off, different dates and possibly different defaults
in between), so the 2.93x gap is an **inconsistency that requires explanation**, not by itself proof
of which number is wrong. The supporting argument is that the x4 run's *own* kernel sum (54.218 ms)
matches the independently measured step (52.26 ms) while its wall does not - which locates the fault
in the wall. The harness-artifact mechanism (snapshot restore vs indexed-sparse re-selection) is
stated as a hypothesis and is untested.

## Reproduce

```bash
# the two runs, and their instrumentation
python3 -c "import json;d=json.load(open('benchmarks/results/2026-09-29-qwen4exp-x4-decode-split/decode-warmup.json'));print(d['protocol']);print(d['contexts'][0]['summary']);print(d['contexts'][0]['repeats'][0]['runtime_census'])"
python3 -c "import json;d=json.load(open('benchmarks/results/2026-09-29-qwen4exp-x4-decode-split/decode-profiled.json'));print(d['protocol']);print(d['contexts'][0]['summary'])"
# the same-host same-model decode figure
python3 -c "import json;d=json.load(open('benchmarks/results/2026-09-13-framework-qwen4exp-strix-journey-baselines.json'));print(d['host']['machine_id']);print(d['arms']['he-ar']['summary']['cases']['code-p4096']['decode_tok_s']['median'])"
```
