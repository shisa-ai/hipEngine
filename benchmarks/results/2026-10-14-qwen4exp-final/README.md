# Qwen4Exp end-of-day rates (2026-10-14)

`qwen4_exp_gguf` / `hip_gfx1151` / `gguf_ud_q4_k_xl`, `production`, Radeon
8060S / gfx1151, host `gfx1151` (machine `55ea6c509d0b49eea8de7094a1023668`),
BF16 KV, chunk 4096, canonical 12-case protocol, tracked-clean `ffd1335c7`.

| Prompt tokens | Prefill tok/s | Decode tok/s | gufo prefill / decode (2026-10-12) |
| ---: | ---: | ---: | ---: |
| 512 | **1037.2** | 23.55 | 1095.0 / — |
| 1024 | **1314.2** | 22.60 | 1482.2 / — |
| 4096 | **1615.5** | 23.12 | 1736.4 / 28.1 |
| 8192 | **1570.4** | 23.02 | — |
| 16384 | **1516.7** | 22.91 | — |

Relative to gufo on the same host: 0.95x / 0.89x / 0.93x prefill at 512 /
1024 / 4096 and 0.82x decode at 4096; ahead of halo-box (759.3 / 1109.1 /
1446.4 prefill, 25.1 decode) in prefill at every length, 0.92x in decode.
At the start of the day the same protocol measured 779.4 / 977.9 / 1190.3
prefill and 20.59 decode at p4096.

The last changes measured here are all bit-identical:
`scripts/qwen4exp_rate_regression_check.py --require-identical-output`
against the decode-batch baselines and then against this directory's first
end-of-day run passes on both fixtures with identical output tokens
(`regression-check-*.json`, `regression-check-final.txt`). The small-row
router and GDN routes committed the same day do not run on these cases.
Gates for the plan: `../2026-10-14-qwen4exp-decode-batch/` (`gate.json`,
`gate-c.json`) and the task review there.

