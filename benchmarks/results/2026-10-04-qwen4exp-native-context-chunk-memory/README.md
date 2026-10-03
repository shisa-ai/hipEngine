# Chunk 4096 at native context: what the promotion actually costs in memory

RT-1 and RT-3 measured the 4096-row prefill workspace at **+4.96 GB** and framed it
as "paid by the prompt that receives the win", with short requests
"byte-for-byte identical in time and memory". **Those framings are harness
artifacts.**

At the **native context** a serving session actually uses, the workspace is sized
by the *model's* context, not the prompt, so the larger chunk is paid at session
construction by **every** session. Measured with the Tier-1 allocation probe:

| config | ctx | peak GiB | free GiB | free % | scratch allowance | observed scratch | allowance unused |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1, chunk 1024 | 262144 | 86.64 | 33.36 | 27.8% | 4.00 GiB | 1.77 GiB | 2299 MiB |
| c1, chunk 4096 | 262144 | 91.95 | 28.05 | 23.4% | 7.17 GiB | 7.09 GiB | 160 MiB |
| c2, chunk 1024 | 262144 | 96.42 | 23.58 | 19.7% | 8.00 GiB | 3.55 GiB | 4578 MiB |
| c2, chunk 4096 | 262144 | **107.05** | **12.95** | **10.8%** | 14.33 GiB | 14.18 GiB | 240 MiB |
| c2, chunk 4096 | 131072 | 98.14 | 21.86 | 18.2% | 13.32 GiB | 13.16 GiB | 240 MiB |
| c2, chunk 4096 | 65536 | 93.68 | 26.32 | 21.9% | 12.81 GiB | 12.65 GiB | 240 MiB |
| c2, chunk 4096 | 32768 | 91.45 | 28.55 | 23.8% | 12.56 GiB | 12.40 GiB | 240 MiB |

GTT on this host is **120 GiB**. Deltas: **c2 +10.63 GiB (+11.02%)**, **c1 +5.31 GiB
(+6.13%)**.

## The cost, stated accurately

**At c2 and native context the promotion roughly halves the absolute headroom** -
23.58 -> 12.95 GiB free, 19.7% -> 10.8%. It does not exhaust it. That is a
substantial reduction on a global model setting, not an out-of-memory result, and
the earlier claim in this artifact's own first revision ("leaves 84-168 MB") was
**wrong**: that figure is the *allowance* margin, a different quantity (see below).

**Two independent signals, and they mean different things:**

1. **Absolute headroom** (peak vs GTT): 12.95 GiB / 10.8% free at c2 native. This
   is the number that decides whether the host can run the configuration.
2. **Allowance saturation**: `scratch_margin_bytes` is
   `declared_allowance - observed_scratch`. At chunk 4096 it is 240 MiB of
   14.33 GiB (1.7% unused), against 4578 MiB of 8.00 GiB at chunk 1024. This is
   **not** host headroom - it says the model's observed scratch has grown to
   saturate its own declared budget. It is constant at 240 MiB across every
   context, because both the allowance and the workspace depend only on the
   chunk.

Both move the same way, but only (1) is an admission number. Conflating them was
the error in this artifact's first revision.

## Why RT-1/RT-3 saw something different

`Qwen4ExpPrefillWorkspace.allocate_inputs` sizes every prefill input buffer as

```python
rows = min(self.prefill_chunk_size, self.max_sequence_length)
```

`self.max_sequence_length` is the **runner's**, set from the model's native context
in `__init__` — and the base workspace is allocated there, before any
request-level context applies. `qwen4exp_profile_gap.py` set
`max_sequence_length = max(768, prompt_tokens + 1)`, tying context to prompt
length, which is why RT-3's table showed p512 resolving to 768 rows at *both*
chunk sizes and concluded the workspace "is already shape-aware".

It is **context-aware, not prompt-aware**. In the serving path the context is the
model's native 262144, so `rows = min(chunk, 262144) = chunk` for every session.

**Direct evidence from the fairness packet**, where `LLM.prepare()` was passed
`max_sequence_length=4097` and the session still reported 262144:

| arm | session `max_sequence_length` | workspace | `prefill_token_ids` |
|---|---:|---|---:|
| chunk 1024 | 262144 | `[1024]` | 8,192 B |
| chunk 4096 | 262144 | `[4096]` | 32,768 B |

`prefill_token_ids` is `rows * 8`, so 8,192 -> 32,768 is a direct read of rows
1024 -> 4096 at native context, independent of the prompt.

## The capacity guard's operating envelope

The context sweep is the useful half of this packet, and it is the answer to "at
what context is chunk 4096 affordable".

Because the workspace is `min(chunk, context)`, the workspace delta is **constant
at 3072 rows for any context above 4096** - so the lever is not a chunk-vs-context
crossover. The lever is that **a smaller context frees KV memory**, which pays for
the larger workspace:

| c2, chunk 4096 | peak GiB | free GiB | vs chunk 1024 at native |
|---|---:|---:|---:|
| ctx 262144 | 107.05 | 12.95 | **-10.63 GiB** |
| ctx 131072 | 98.14 | 21.86 | -1.72 GiB |
| ctx 65536 | 93.68 | 26.32 | +2.74 GiB |
| ctx 32768 | 91.45 | **28.55** | **+4.97 GiB** |

**At 32K context, chunk 4096 leaves more absolute headroom (28.55 GiB) than chunk
1024 does at native context (23.58 GiB).** So the guard has a real operating
envelope: the larger chunk is comfortably admissible at reduced context, and only
the full 256K configuration is tight.

## What this does and does not decide

- **It does not retract the speed win.** The generator chunk 4096 delivers
  **-5.74% / -5.80%** prefill wall through the engine loop at 183x separation,
  with no peer-fairness cost and a halved peer worst-gap.
- **It does not by itself block the promotion.** 12.95 GiB / 10.8% free is not an
  OOM, and this packet did not measure a failure. Calling it a blocker - as this
  artifact's first revision did - overstated the evidence.
- **It does establish that the cost is universal, not prompt-gated**, which is the
  correction that matters: every session at native context pays it, so a global
  default change is a real reduction in headroom for every deployment rather than
  a cost borne only by long prompts.
- **It gives the guard its budget**: chunk 4096 is affordable at reduced context
  and tight at 256K, so a guard that keys on the admitted context is the right
  shape and now has measured points to interpolate.

## Limits

- One host (gfx1151 APU, unified memory, GTT 120 GiB), one model, one quant, one
  profile, one repetition per arm. The 4x row relationship is structural
  (`min(chunk, context)`), not host-specific, but the absolute headroom is.
- **Tier-1 only**: allocation validity and tracked peak. No full-prefill
  completion or stability claim at depth, and no OOM was produced.
- **The allowance figures are the probe's declared budget, not a hardware limit**,
  and the probe does not test behaviour when that budget is exceeded.
- The fairness packet's `+4.96 GB` from RT-1/RT-3 is not re-derived here; this
  packet measures the same knob at the context that matters and finds a larger and
  more broadly-applied cost.
- Contexts below 32768 were not swept.
