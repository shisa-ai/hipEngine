# Chunk 4096 at native context: a measured promotion blocker

RT-1 and RT-3 measured the 4096-row prefill workspace at **+4.96 GB** and framed
it as "paid by the prompt that receives the win", with short requests
"byte-for-byte identical in time and memory". **Both framings are harness
artifacts, and the real cost is a blocker.**

At the **native context** a serving session actually uses, the workspace is sized
by the *model's* context, not the prompt, so the larger chunk is paid at session
construction by **every** session. Measured at c1 and c2:

| config | chunk 1024 peak | chunk 4096 peak | delta | scratch margin @4096 | device scratch margin @4096 |
|---|---:|---:|---:|---:|---:|
| c1, native 262144 | 93.03 GB | 98.73 GB | **+5.71 GB (+6.13%)** | **168 MB** | **84 MB** |
| c2, native 262144 | 103.53 GB | 114.94 GB | **+11.41 GB (+11.02%)** | **252 MB** | **168 MB** |

**At chunk 4096 the device scratch allowance is consumed to within 84 MB (c1) and
168 MB (c2).** For comparison, chunk 1024 leaves 2.39 GB and 4.78 GB. There is no
headroom left for another owner, a third session, or any future workspace.

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

## Method

`scripts/qwen4exp_chunk_memory_probe.py` (Tier-1 allocation probe, no full
prefill), one process per arm, production profile, cached build,
`HIPENGINE_REQUIRE_CACHED_BUILD=1`, clean tracked revision. Native context
262144; `--capacity` 1 and 2; `--chunk-size` the only difference within each
pair. ~52 s per arm.

The delta is **entirely scratch**: `explicit_components_bytes` moves only
99,739,496,800 -> 99,802,411,360 (c2), while the scratch allowance itself grows
8.59 -> 15.39 GB and observed device scratch grows 3.81 -> 15.22 GB.

## What this decides

**A blanket production chunk-4096 is not promotable.** AGENTS.md requires a
concrete recorded blocker to withhold a measured non-regressive win, and this is
one: at the configuration a real c2 server runs, the change leaves **84-168 MB**
of device scratch margin. That is not a safe default for a global model setting,
and it is a measurement rather than a hypothesis.

**This is not a retraction of the speed win.** The generator chunk 4096 delivers
**-5.74% / -5.80%** prefill wall through the engine loop at 183x separation, with
no peer-fairness cost and a halved peer worst-gap. Both results stand:

- **The win is real, reachable, and free of peer-fairness cost.**
- **The blanket default is blocked on measured admission headroom.**

Together they are the case for the **capacity guard** RT-1/RT-3 named — take the
larger chunk when the allocation is admissible, fall back when it is not — now
with a measured reason and a measured budget (the guard has 84-168 MB to work
with at native context, so it must actually check rather than assume).

## Limits

- One host (gfx1151 APU, unified memory), one model, one quant, one profile,
  native context only, one repetition per arm. The 4x row relationship is
  structural (`min(chunk, context)`), not host-specific, but the absolute
  headroom is.
- **No context sweep.** Only native 262144 was measured. The crossover — the
  context below which chunk 4096 is admissible — is not established, and it is
  what a capacity guard would need.
- Tier-1 only: allocation validity and tracked peak, no full-prefill completion
  claim at depth.
- The fairness packet's `+4.96 GB` from RT-1/RT-3 is not re-derived here; this
  packet measures the same knob at the context that matters and finds a larger
  and more broadly-applied cost.
