#!/bin/bash
# Packet 1 stage 3: retry screening-cell baselines + profiles after the
# eligibility fix (tracked-clean tree required).
set -u
cd /home/lhl/hipEngine-gfx1100-concurrency2-better-mtp
RAW=/tmp/he-bettermtp-raw/packet1
export HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS=1
bash campaign-artifacts/packet1/run_p1_baselines.sh
bash campaign-artifacts/packet1/run_p1_profiles.sh
echo "=== STAGE P1-3 COMPLETE $(date -u +%H:%M:%S)"
