# Padded-fragment dense-wide tail operation screen

On 2026-10-06 UTC, Radeon 8060S host `gfx1151`, machine-id
`55ea6c509d0b49eea8de7094a1023668`, measured the Qwen3.8-Flash-Next
UD-Q4_K_XL HC-down Q8_0 GEMM at K10240/N320. A measurement-only sibling
computes every WMMA row fragment from the existing zero-padded LDS rows,
instead of branching around inactive fragments inside the K loop. It keeps
bounded loads/stores and the parent's F16-operand/F32-accumulator order for
valid outputs. Full tiles still use the original tail-free owner.

| Rows | Real weights, synthetic activation parent/candidate ms | Speedup |
| ---: | ---: | ---: |
| 512 | 0.786610 / 0.598936 | 1.313x |
| 1024 | 0.796243 / 0.601961 | 1.323x |
| 4096 | 2.717172 / 2.007559 | 1.353x |
| 257 | 0.781622 / 0.592872 | 1.318x |
| 300 | 0.783343 / 0.593764 | 1.319x |

These medians use 16 counterbalanced pairs, ten complete GEMMs per event
interval after one warm pair. Activations are deterministic synthetic F16
inputs; weights are the actual `blk.16.hc_attn_down.weight` GGUF bytes.
Preparation and downloads are outside timing. All output hashes match.

Six model-produced activation cases intercept all 70 HC-down calls per case:
code p512/p1024/p4096 and English/Japanese/mixed p4096. One warm pair plus two
counterbalanced event pairs compare complete output and input/weight hashes
at each call. All **420 calls** are exact, deterministic and input-preserving.
Parent/candidate aggregate owner times are 55.228/42.099 ms at code-p512,
56.171/42.510 ms at code-p1024 and 188.502/138.471 ms at code-p4096.
The six speedups span **1.312–1.361x**. Teardown reports zero allocations and
bytes. These are complete-GEMM sub-window results, not model-wall speedups.

The F16 tail code object drops private scratch **92→0 bytes/lane**, VGPR
spills **58→0**, static scratch instructions **38→0**, and static VGPR
**256→253**. LDS is unchanged at 55,296 bytes. A cached-only trace names the
parent and padded sibling, observes scratch92/0 respectively, and records
candidate durations 1.924–2.003 ms. The profiler rounds VGPR allocation to 256;
253 is the code-object static count. Trace timing is diagnostic, not the
unprofiled performance result.

The first synthetic run reversed the row-major GGUF `(N,K)` shape into the
launch ABI incorrectly. Its timings are **INVALID**, excluded, and retained
only under `/tmp/qwen4exp-dense-tail-20261006/micro.json`. A unit regression
now pins both dimension order and source bytes. The corrected run above is
`micro-correct-shape.json`.

## Correctness and decision

The targeted RED run fails seven times because the sibling registry surface
does not exist. After implementation, 12 GPU cases cover both activation
ABIs, full tiles, row/column tails, repeated byte parity, guard publication,
input immutability and the independent CPU-reference KL/top-1 floor. The
focused CPU bundle passes 60 tests; a source-export parser was expanded to
recognize the new macro after its isolated failure. Registry/CPU fixtures
and test-tier checks pass. The broader existing family GPU bundle passes
39 tests, including the 12 padded cases.

The sibling is registered but has **no runtime selector or profile binding**.
Qualify full-model canonical/heldout logits, recurrent state, KV, engagement,
repeatability and lifecycle, then run tracked-clean same-host engine A/B.
The predeclared natural-owner scope is K10240/N320 and 512–4096 rows; do not
infer admission for other tail shapes from this packet.

## Reproduction

```bash
ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH"
export HIPENGINE_HIP_ARCH=gfx1151
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-gr-dpp-20261005/hipcc-version.txt
uv run --no-sync pytest -q tests/test_gpu_qwen4exp_dense_wide_padded.py
export HIPENGINE_REQUIRE_CACHED_BUILD=1
uv run --no-sync python scripts/qwen4exp_dense_wide_padded_micro.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --output /tmp/qwen4exp-dense-tail-20261006/micro-correct-shape.json \
  --rows 512 1024 4096 257 300 --repeats 16 --batch 10
uv run --no-sync python scripts/qwen4exp_dense_wide_padded_owner.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
  --output /tmp/qwen4exp-dense-tail-20261006/natural-owner.json
```

[artifact.json](artifact.json) includes exact commands, physical host,
manifest/build hashes, individual samples and natural captures. The named
production manifest describes the surrounding composition, not admission of
the diagnostic substitution. No precision, quant/KV format or default changes
in this unit. Raw traces/code objects remain outside Git.
