# RT-1: chunk 256 -> 1024, both arms in one run

**This is the decisive comparison.** 256 -> 1024 captures almost all of the
available win at a fifth of the memory cost of 256 -> 4096.

Both arms were measured in a **single run** on purpose: `docs/REFACTOR.md`
explicitly forbids claiming a direct chunk comparison from separate runs.

## Protocol

All 12 fixture cases (`qwen4exp_canonical_ar_p512_p1024_p4096.json`, sha256
`42b562bd…`) at chunk 256 and chunk 1024, three repetitions each,
`--mode prefill --execution-profile production --require-cached-build`, one host,
one run. 24 invocations, 24 artifacts, all exit 0.

## Result

| case | category | 256 median (s) | 1024 median (s) | delta | % | separated | token_id | logits | peak delta |
|---|---|---:|---:|---:|---:|:--:|:--:|:--:|---:|
| code-p512 | code | 2.8448 | 2.0138 | -0.8310 | **-29.21%** | yes | same | DIFF | +0.77 GB |
| code-p1024 | code | 5.7568 | 3.7778 | -1.9790 | **-34.38%** | yes | same | DIFF | +1.24 GB |
| code-p4096 | code | 23.8834 | 15.5595 | -8.3240 | **-34.85%** | yes | same | DIFF | +1.24 GB |
| general_en-p512 | general_en | 2.8148 | 1.9922 | -0.8226 | **-29.22%** | yes | same | DIFF | +0.77 GB |
| general_en-p1024 | general_en | 5.6253 | 3.7768 | -1.8486 | **-32.86%** | yes | same | DIFF | +1.24 GB |
| general_en-p4096 | general_en | 23.2397 | 15.5136 | -7.7261 | **-33.25%** | yes | same | DIFF | +1.24 GB |
| general_ja-p512 | general_ja | 2.7851 | 1.9776 | -0.8075 | **-28.99%** | yes | same | DIFF | +0.77 GB |
| general_ja-p1024 | general_ja | 5.5865 | 3.7391 | -1.8475 | **-33.07%** | yes | same | DIFF | +1.24 GB |
| general_ja-p4096 | general_ja | 22.9920 | 15.4371 | -7.5549 | **-32.86%** | yes | same | DIFF | +1.24 GB |
| mixed_ja_en-p512 | mixed_ja_en | 2.8522 | 2.0130 | -0.8392 | **-29.42%** | yes | same | DIFF | +0.77 GB |
| mixed_ja_en-p1024 | mixed_ja_en | 5.7318 | 3.7897 | -1.9421 | **-33.88%** | yes | same | DIFF | +1.24 GB |
| mixed_ja_en-p4096 | mixed_ja_en | 24.1258 | 15.5183 | -8.6075 | **-35.68%** | yes | same | DIFF | +1.24 GB |

**12/12 faster, 12/12 completely separated** (worst-of-1024 beats best-of-256),
`token_id` identical 12/12, `logits_sha256` identical 0/12.

By size: p512 **-29.0 to -29.4%**, p1024 **-32.9 to -34.4%**, p4096 **-32.9 to
-35.7%**. Flat across category.

## The marginal trade is the finding

| promotion | `code-p4096` speedup | `code-p4096` peak cost |
|---|---:|---:|
| 256 -> **1024** | **-34.85%** | **+1.24 GB** |
| 256 -> 4096 | -37.61% | +6.20 GB |

Going on from 1024 to 4096 buys **2.76 further percentage points for 4.96 GB
more**. 256 -> 1024 captures **93%** of the total win at **20%** of the memory.
So 1024 is the promotion candidate, not 4096.

Peak delta is **flat at +1.24 GB for p1024 and p4096**, because the workspace is
`min(chunk, max_sequence_length)` - once the chunk exceeds the sequence length it
stops growing.

## Chunk 1024 is the certified configuration

The X-6 production-profile gate's numerical certification is **at chunk 1024**,
and it hashes **bit-identically** at chunk 4096. So promoting 256 -> 1024 lands on
the configuration the gate actually certifies, rather than on one that merely
matches it.

`logits_sha256` still differs from chunk 256 in all 12 cases, so this is an
**arithmetic change relative to the current serving default** and must be
described that way.

## Promotion blocker: the knob is a scheduler granularity, not a model shape

**No default changed.** The blocker is specific and is *not* the memory cost,
which at +1.24 GB is small.

`max_prefill_chunk_tokens` lives in `EngineLoopConfig` alongside
`fair_prefill_burst_chunks`, `round_prefill_token_budget` and
`round_decode_row_budget`. Its CLI help reads "Maximum prefill chunk tokens per
loop tick", and the dispatch caps it again at the round budget
(`chunk_size=min(self.prefill_chunk_size, prefill_budget)`). It is a **C4
scheduler knob that bounds per-tick prefill work so decode requests are not
starved** - not a model-shape parameter.

Raising it 256 -> 1024 makes each prefill tick up to 4x longer. The measured
-29% to -36% is **prefill throughput**, and it says nothing about **decode
latency under concurrency**, which is what the tick bound exists to protect. That
effect is **unmeasured**, and it is the same prerequisite `docs/REFACTOR.md`
already names for this mechanism ("qualify ... c2 behavior").

So the honest position is:

- **Prefill throughput for 256 -> 1024: measured**, 12/12, -29% to -36%, +0.77 to
  +1.24 GB, completely separated, on the gate's certified configuration.
- **Serving default: blocked** on the interleaving/fairness measurement, because
  the knob's purpose is bounding per-tick work and that effect has not been
  measured.

A prefill-only or offline/batch path is directly supported by the evidence above.
A serving-path default change is not, yet.

## Limits

- **One host** (gfx1151 APU, unified memory, 120 GB GTT).
- **One model** (Qwen3.8-Flash-Next UD-Q4_K_XL).
- **Prefill only.** Decode is unaffected by this parameter.
- No concurrency (c2/c8) measurement; this packet makes no claim about decode
  latency under load.
