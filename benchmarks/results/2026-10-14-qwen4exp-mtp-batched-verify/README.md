# Qwen4Exp MTP: batched verify with per-row state commit (2026-10-14)

`qwen4_exp_gguf` / `hip_gfx1151` / `gguf_ud_q4_k_xl`, `production`, Q8_0
MTP sidecar, Radeon 8060S / gfx1151, host `gfx1151`. Full 10-prompt
mtp-bench category suite (`benchmarks/prompts/mtpbench-code-general-ja.jsonl`,
heldouts included), chat-formatted, 64 greedy tokens per prompt, draft
budget 3, one unmeasured warmup per prompt and arm, true autoregressive
arm in the same process (`scripts/qwen4exp_mtp_inprocess_bench.py`).
Diagnostic: MTP is still slower than AR overall, so no speed claim.

| Verify | MTP / AR full | code | general_en | general_ja | mixed_ja_en | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| serial target steps | 0.825 | 0.839 | 0.832 | 0.818 | 0.798 | 55.3% |
| one batched pass, accepted row's state committed | **0.930** | **1.063** | 0.952 | 0.863 | 0.781 | 55.4% |

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
