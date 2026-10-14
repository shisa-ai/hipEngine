# Qwen4Exp MTP: batched verify with per-row state commit (2026-10-14)

`qwen4_exp_gguf` / `hip_gfx1151` / `gguf_ud_q4_k_xl`, `production`, Q8_0
MTP sidecar, Radeon 8060S / gfx1151, host `gfx1151`. Full 10-prompt
mtp-bench category suite (`benchmarks/prompts/mtpbench-code-general-ja.jsonl`,
heldouts included), chat-formatted, 64 greedy tokens per prompt, draft
budget 3, one unmeasured warmup per prompt and arm, true autoregressive
arm in the same process (`scripts/qwen4exp_mtp_inprocess_bench.py`).
The first rows are history; the bonus-row rows are the current result.

| Verify | MTP / AR full | code | general_en | general_ja | mixed_ja_en | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| serial target steps | 0.825 | 0.839 | 0.832 | 0.818 | 0.798 | 55.3% |
| one batched pass, accepted row's state committed | 0.930 | 1.063 | 0.952 | 0.863 | 0.781 | 55.4% |
| batched pass, eight-column pack8 dense GEMV for 2..8 rows (budget 3) | **1.107** | **1.265** | **1.136** | **1.034** | 0.923 | 55.4% |
| the same, budget 4 | 1.125 | 1.348 | 1.087 | 1.057 | 0.920 | 49.0% |
| verify `[root, c1 .. cn]` with a bonus row (budget 3) | **1.172** | **1.402** | **1.159** | **1.092** | **0.948** | 56.1% |
| the same, budget 4 | 1.160 | 1.421 | 1.132 | 1.052 | 0.939 | 49.1% |

AR measured 22.66 tok/s over the suite in this protocol. The batched pass
runs `[root, c1, c2]` through the prefill-chunk owners with the row-split
GDN recurrence and the exact conv writing the state after every row; a
partial accept copies the accepted row's GDN and conv states back and
rewinds the attention, index, PLE and position cursors (1.5 ms), with no
replay. Equal acceptance across the two arms shows the verify path does not
change what is accepted. 8 of 10 MTP outputs are identical to the AR arm;
the two others diverge at tokens 22 and 51 (`general_en_explain`,
`general_ja_explain`), where the prefill-shaped verify kernels' production
arithmetic picks a different greedy token than serial decode.

Cost per cycle at budget 3: verify pass about 76 ms (3 rows) against a
42 ms decode step, draft proposal about 13.5 ms. The verify pass costs 1.8
decode steps for 3 rows, so MTP only wins where acceptance is high (code
prompts, 75%). Next: profile and cut the small-row verify pass.

## Dense projections at verify widths

`rocprofv3` of a 3-row verify pass (code-p512 context): 72.8 ms of kernels,
43.5 ms of it the one-column row-tile Q8_0/Q5_K/Q6_K kernel over the dense
projections, against 18 ms for the same weights in a decode step. The
row-tile launcher now runs an eight-column pack8 kernel that dequantizes each
weight element once for all rows; each row keeps the row-tile and pack8
kernels' k order and reduction tree, so outputs are unchanged
(`tests/test_gpu_gguf_k_rowtile_gemv.py`). Verify plus commit, 2 / 3 / 4
rows: 72 / 76 / 83 -> 54 / 59.5 / 67 ms (decode step 42 ms). MTP over the
suite (AR 22.64 tok/s): 25.06 tok/s at budget 3, 25.52 at budget 4.

## Bonus row

As in the in-tree Gemma 4 MTP, the verify block is `[root, c1 .. cn]`
(n + 1 rows), so a cycle whose drafts all match commits one more token from
the last row; the draft then takes `cn` with the target's hidden row so its
cursor reaches the committed position. Budget 3: 25.06 -> 26.56 tok/s
(1.107 -> 1.172x AR).
