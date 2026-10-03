# gfx1151 H2048-MoE prefill chunk: 512 -> 1024

Model `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, quant `gguf_ud_q4_k_m`, host gfx1151
Strix Halo Radeon 8060S. Date 2026-10-03.

Promotes the linear/MoE prefill chunk for the Qwen3.6-35B-A3B (H2048-MoE)
geometry from 512 to 1024. The published benchmark table was not replaced: its
decode column is measured on the 2026-08-20 host state, so swapping in these
rows would mix protocols. These rows are the reference for the next clean
same-session re-measurement.

## Why the change

The 512 override came from a 2026-08-20 sweep that passed an explicit
`PrefillConfig`. Under the manual-chunk contract an explicit config leaves
`full_attn_query_chunk_size` at 0, so that sweep left full attention unchunked
in every arm. Re-measured with the production config shape, 1024 wins at every
short shape.

## Numerical gate

`scripts/gguf_prefill_chunk_boundary_gate.py --chunks 512,1024,4096 --length
4096 --repeats 1` (`chunk-boundary-gate-4k.json`). Same-config repeats are
bit-identical, so the run-noise floor is exactly zero.

| pair | KL | max abs logit | top-1 |
| --- | --- | --- | --- |
| 512 vs 512 | 0.00000000 | 0.00000000 | identical |
| 1024 vs 1024 | 0.00000000 | 0.00000000 | identical |
| 4096 vs 4096 | 0.00000000 | 0.00000000 | identical |
| **512 vs 1024** | **0.00000000** | **0.00000000** | identical |
| 512 vs 4096 | 0.00000031 | 1.27896905 | identical |
| 1024 vs 4096 | 0.00000031 | 1.27896905 | identical |

**512 vs 1024 is bit-exact.** Row chunking partitions tokens rather than
reductions, so a row-chunk size cannot change any single token's arithmetic.
4096 is not bit-exact, so it selects a different kernel or tile instead of
merely regrouping; its KL is far below the 0.05 floor but it is not promoted.

## Chunk-size sweep

One host state, one session per point, 4 repetitions, 4 s settle, geometry
policy disabled so each size is passed explicitly. Prefill tok/s
(`chunk-size-sweep.json`):

| shape | 512 | 1024 | 4096 |
| --- | --- | --- | --- |
| 1K | 1454.8 | 1486.4 (+2.17%) | 1488.2 (+2.29%) |
| 4K | 1459.3 | 1487.7 (+1.95%) | 1503.9 (+3.06%) |
| 16K | 1358.5 | 1358.5 (0.00%) | 1358.3 (-0.01%) |
| 64K | 969.9 | 969.5 (-0.04%) | 969.2 (-0.07%) |

The lever is confined to short prompts and vanishes at 16K, where all three
values land inside 0.02%. 4096 is faster at 1K/4K but costs more at 64K with
non-overlapping samples, so its long-context penalty is real if tiny and grows
with context. 1024 was chosen because it is bit-exact, keeps that penalty at
noise level, and still takes the short-prompt win.

## Production rows after promotion

`scripts/qwen35_gguf_bench.py --public-ar-profile --measured-runs 3
--warmup-runs 1 --decode-tokens 128`, with `chunks=1024/1024
reason=geometry_prefill_chunk_policy` confirmed in each run's tuning record
(`promoted-rows.json`):

| shape | prefill tok/s | vs 512-chunk | decode tok/s | published prefill / decode |
| --- | --- | --- | --- | --- |
| 512 | 1401.89 | +0.84% | 56.551 | 1369.489 / 54.330 |
| 4K | 1479.75 | +1.39% | 56.163 | 1430.215 / 54.798 |
| 32K | 1197.59 | -0.17% | 45.067 | 1144.713 / 46.405 |
| 64K | 969.08 | -0.02% | 38.241 | 936.218 / 40.180 |

Prefill moves where the sweep predicted and decode is unchanged by the chunk.
The `vs published` prefill gain (+2.4% to +4.6%) combines this change with the
2026-10-03 accelerator-performance host profile, so it is not attributable to
the chunk alone. Long-context decode is 2.9% (32K) and 4.8% (64K) below the
published rows and remains unexplained; it also varies about 1% between
sessions, so it needs its own host-state investigation.

## Reproduce

```bash
python3 scripts/gguf_prefill_chunk_boundary_gate.py \
    --chunks 512,1024,4096 --length 4096 --repeats 1 --json gate.json

python3 scripts/qwen35_gguf_bench.py \
    --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --quant gguf_ud_q4_k_m \
    --prompt-length 4096 --decode-tokens 128 --warmup-runs 1 --measured-runs 3 \
    --public-ar-profile --json row.json
```
