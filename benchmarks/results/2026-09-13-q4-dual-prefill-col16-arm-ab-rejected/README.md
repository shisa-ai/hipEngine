# Rejected: 16-column dual-SiLU Q4T16 prefill arms (gfx1151)

**Status:** rejected and removed. No published number changed.

The gfx1151 prefill profile put 33-36% of prefill kernel time on
`gguf_q4_t16_dense_dual_wmma_prefill_silu_bf16_kernel<false,false,4,4>`
(32 columns x 256 rows per block), and the tile-knob screen that followed ranked
a 16-column block first on an instruction-stream argument. Two 16-column arms
were built on a new `out_tiles_per_block` template parameter, gated bit-exact
against the parent, and measured here. Both lose by 29-59% at every prefill
shape, so both arms and the parameter they were built on were removed in the
same unit.

## Same-host A/B

Real `blk.0.ffn_gate`/`blk.0.ffn_up` Q4_K tiles from `Qwen3.8-27B-Q4_K_M.gguf`,
the same harness and flags as the tile-knob screen, three harness invocations
per shape. Ratios are control/candidate, so above 1.0 means the candidate is
faster.

| Rows | Control (32 cols, 256 rows) | `col16_row256` | ratio | `col16_row512` | ratio |
| --- | --- | --- | --- | --- | --- |
| 256 | 3.312 ms | 4.640 ms | **0.714** | 8.099 ms | **0.409** |
| 512 | 6.622 ms | 9.895 ms | **0.669** | 9.314 ms | **0.711** |
| 1,024 | 13.125 ms | 20.788 ms | **0.631** | 18.984 ms | **0.691** |
| 2,048 | 26.026 ms | 42.444 ms | **0.613** | 38.142 ms | **0.682** |
| 4,096 | 52.146 ms | 85.978 ms | **0.607** | 78.163 ms | **0.667** |

- Worst per-cell spread across the three invocations: **3.9%** against a 29-59%
  effect, so the loss is outside the noise floor.
- Output is **bit-identical to the control in all 30 cells** (3 invocations x 5
  shapes x 2 arms); `assemble.py` fails closed if any cell is not.
- The control column reproduces the in-situ profiled owner within **0.1%**
  (6.622 ms here against 6.629 ms in the tile-knob screen at 512 rows) and the
  prior screen's own profiled check (6.637 ms), which is what makes this a
  same-host comparison.

## Why the arms lose

The weight decode needs `threads/4 >= 2 matrices x (columns/2)` decode pairs,
which is exactly saturated at 32 columns with 128 threads. At 16 columns only 64
of the 128 threads decode, each doing the same 32 k-values per sub-block, so a
block's decode takes as long as the parent's while its WMMA work halves. Blocks
double for the same output, so total decode time doubles and total WMMA time is
unchanged. With the decode at roughly 40% of the parent's issue stream
(source-level count in the tile-knob screen), that predicts about 1.4x the
parent's time; the measurement gives 1.40-1.65x.

The tile-knob screen's first-ranked candidate assumed halving columns halves the
per-block decode for the same compute. It does not, which is why that screen's
own sibling measurement (16 columns 1.7x slower) was right and its
instruction-stream ranking was wrong.

## No resident-session arm was spent

The candidate has no dispatch path, so an end-to-end 512/128, 1K/128, 4K/128 arm
would have required a new temporary selector for a kernel that this owner-level
measurement already shows is 1.40-1.65x slower. With the owner at 33-36% of
prefill kernel time, that is roughly 13-23% of prefill wall by arithmetic, so no
end-to-end result could have been non-regressive. This is recorded as an
inference from the measured owner share, not as a measurement.

## Files

| File | Contents |
| --- | --- |
| `artifact.json` | Assembled result: cells, summary, mechanism, decision, limitations. |
| `assemble.py` | Regenerates `artifact.json` byte-identically from the three runs; fails closed on any non-exact cell. |
| `run1.json`, `run2.json`, `run3.json` | Harness output for the three invocations. |

## Reproduce

```bash
cd /home/lhl/hipEngine
for i in 1 2 3; do
  PYTHONPATH=. .venv/bin/python scripts/qwen38_packet4_q4_dual_silu_row_leaf.py \
    --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
    --rows 256 512 1024 2048 4096 --burst 4 --repetitions 3 --warmups 2 \
    --output /tmp/q4-col16-ab/run$i.json
done
python3 benchmarks/results/2026-09-13-q4-dual-prefill-col16-arm-ab-rejected/assemble.py \
  --out /tmp/artifact-check.json
```

The candidate arms themselves are gone (removed at commit
`88e0e6b38`-equivalent state); re-running the harness as written above measures
the control and the retained row arms only. The rejected arms are reproducible
from commit `4620b9cf5`, which this unit reverts.

## Limits

Single host, single model, one quant, one session per invocation. Arm order
inside an invocation is fixed, so a position or thermal effect is not
counterbalanced; the three-invocation spread bounds it at 3.9%. Prefill owner
only: no resident-session wall, no decode column, and no multi-prompt category
suite, for the reason above.
