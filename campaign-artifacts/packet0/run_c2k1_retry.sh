#!/bin/bash
set -u
cd /home/lhl/hipEngine-gfx1100-concurrency2-better-mtp
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1
export HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS=1
.venv/bin/python scripts/gguf_mtp_c1c8_server_bench.py \
  --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  --backend hip_gfx1100 --quant gguf_q4_k_m --execution-profile production \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --generation2-diagnostic \
  --mtp-request-mode explicit --widths 2 --resident-capacity 2 \
  --expected-mtp-widths 2 --candidate-budget 1 \
  --max-tokens 24 --batch-window-ms 20 --correctness-contract ar_exact \
  --output campaign-artifacts/packet0/p0-c2k1-screen-rep1.json 2>&1
echo "rc=$?"
