# RT-3: the chunk setting is inert when the prompt fits in the chunk

RT-3's rationale is "the September 15 4096 option improved 4K but **short-request
costs kept 1024 default**". Measured: that short-request cost is **not
reproducible today**. The chunk setting has no effect - on time or memory - for
any prompt that fits inside the chunk, and the only real cost of a larger chunk is
memory admitted at long prompts.

That **simplifies** RT-3 rather than complicating it: the shape-based *time*
policy the row asks for is unnecessary, and the lever is a capacity guard.

## The mechanism

Scratch is sized by `min(chunk_size, max_sequence_length)`, and the harness sets
`max_sequence_length = max(768, prompt_tokens + 1)`. So the chunk setting only
changes anything once the prompt exceeds the chunk:

| prompt | `max_seq_len` | rows at chunk 1024 | rows at chunk 4096 | differs? |
|---|---:|---:|---:|---|
| 512 | 768 | 768 | 768 | no |
| 1024 | 1025 | 1024 | 1025 | +1 row |
| 4096 | 4097 | 1024 | 4096 | 4x |

That model predicts all three measurements below, including the two I ran before
stating it.

## Measurements

`code-p1024`, three repetitions per arm, production profile, cached build, one
process per arm, `--prefill-chunk-size` the only difference:

| chunk | wall (3 reps) | median | peak allocation |
|---|---|---:|---:|
| 1024 | 3.6975 / 3.7710 / 3.7671 s | **3.7671 s** | 84,337,158,344 B |
| 4096 | 3.7459 / 3.8207 / 3.8196 s | **3.8196 s** | 84,338,566,196 B |
| **delta** | | **+0.0525 s (+1.39%)** | **+1,407,852 B (+0.00167%)** |

`token_id` (248068) and `logits_sha256` are **identical**, `fell_back_to_strict`
is False on both.

**The +1.39% is not an effect.** The arms' within-arm ranges are 0.0735 s (1024)
and 0.0748 s (4096); the median gap of 0.0525 s is *smaller* than either range, and
the arms overlap. This is indistinguishable, and I am not reporting it as a
slowdown.

## The full picture across the three measured points

| prompt | wall at chunk 4096 vs 1024 | peak allocation at 4096 vs 1024 |
|---|---|---|
| 512 | indistinguishable (arms overlap) | **byte-identical** (83,857,109,608 B both) |
| 1024 | indistinguishable (arms overlap) | **+1.34 MiB (+0.0017%)** |
| 4096 | **-4.94%** (4096 faster) | **+4.96 GB (+5.9%)** |

The transition is sharp and lands exactly where the model says it should: at the
point the prompt stops fitting in the chunk.

## What this means for RT-3

**Falsified:** the "short-request costs" that historically kept 1024 as the
default. At p512 the two settings are byte-identical in allocation and
indistinguishable in time; at p1024 they differ by 1.34 MiB and overlap in time.
A blanket chunk 4096 costs short requests **nothing measurable**.

**The real trade is memory admission at long prompts only.** The cost is
proportional to `min(chunk_size, prompt_tokens)` - a request pays for the scratch
it can actually use, not for the chunk setting.

**So the shape-based policy RT-3 asks for is not the right form.** There is no
time/memory tradeoff to balance below the chunk size because nothing changes
there. The lever is a **capacity guard**: take the larger chunk when the
allocation is admissible, and fall back when it is not. That is a simpler change
than a length-bucketed policy, and it is what the evidence supports.

## What was not done

RT-3 asks for substantially more than this packet delivers, and the gaps are real:

- **No 2051/2052 case was measured.** The canonical fixture has p512/p1024/p4096
  only, so the exact boundary the row names was not probed. The model above
  predicts the behaviour there (2049 vs 1024 rows, roughly 2x scratch) but that is
  a prediction, **not a measurement**.
- **No separate accounting** of repair queues, F16 workspace, or c2 peak. Only
  aggregate `peak_allocated_bytes` was read.
- **No peer-fairness measurement.**
- **No policy was implemented.** Nothing was promoted; the default is unchanged.
- **Ragged final chunks** were not specifically exercised.
- One prompt family (`code-*`), one host, one quant.

## Limits

- The p512 byte-identical result is from the earlier RT-1 packet; p1024 is new
  here. Both are the same harness, profile, and host.
- Peak allocation is the whole-process peak, so it includes everything, not just
  chunk scratch.
