# Qwen4Exp end-of-day rates (2026-10-14)

`qwen4_exp_gguf` / `hip_gfx1151` / `gguf_ud_q4_k_xl`, `production`, Radeon
8060S / gfx1151, host `gfx1151` (machine `55ea6c509d0b49eea8de7094a1023668`),
BF16 KV, chunk 4096, canonical 12-case protocol, tracked-clean `988bbedf9`.

| Prompt tokens | Prefill tok/s | Decode tok/s | gufo prefill / decode (2026-10-12) |
| ---: | ---: | ---: | ---: |
| 512 | **1031.8** | 23.41 | 1095.0 / — |
| 1024 | **1309.3** | 22.47 | 1482.2 / — |
| 4096 | **1596.3** | 22.99 | 1736.4 / 28.1 |
| 8192 | **1553.6** | 22.94 | — |
| 16384 | **1503.1** | 22.82 | — |

Against the decode-batch baselines (`../2026-10-14-qwen4exp-decode-batch/`) the
only change on these cases is the bit-identical one-launch norm and inject
for prefill reads: prefill +0.3% / +2.0% / +2.1% at 512 / 1024 / 4096 and
+1.7% / +1.8% at 8K / 16K, decode unchanged, output tokens identical; both
regression checks pass. The small-row router and GDN routes committed the
same day do not run on these cases. Gates for the plan are in the
decode-batch README (`gate.json`, `gate-c.json`).

Relative to gufo on the same host: 0.94x / 0.88x / 0.92x prefill at
512 / 1024 / 4096 and 0.82x decode at 4096. At the start of the day the
same protocol measured 779.4 / 977.9 / 1190.3 prefill and 20.59 decode.
