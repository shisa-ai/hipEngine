# Quantization quality

This directory is the canonical index for hipEngine quantization-quality work.
It is deliberately separate from throughput and latency tables: model fidelity
and speed answer different questions, and neither substitutes for the other.

## Current status

The Qwen3.6-35B-A3B exact-artifact gate is complete for the local Q4_K_M,
PARO W4, and ROCmFP4 STRIX_LEAN artifacts. All candidates were scored against
one BF16 teacher fixture over the same 90 full-vocabulary positions. PARO also
passes an independent same-checkpoint ParoQuant/Transformers implementation
gate and a bit-identical repeat.

Q4_K_M is the distribution-fidelity baseline. PARO and ROCmFP4 are both
**quality-traded**, not Q4-equivalent. They remain valid cross-format product
and performance candidates only when that label and their measured tradeoffs
are shown.

Authoritative evidence:
[`2026-08-16-zbook-qwen36-quant-quality.json`](../results/2026-08-16-zbook-qwen36-quant-quality.json).

The separate [ROCmFPX transferable-mechanism report](ROCMFPX-TRANSFER.md)
consolidates the five implementation opportunities tested on the admitted
35B-A3B paths. Its speed/policy decisions do not alter the quality labels in
this file.

## Repository layout

- `benchmarks/quant/` — current protocols, provenance, compact quality tables,
  and transferable-mechanism comparison indices.
- `scripts/quant_quality/` — reusable fixture, capture, comparison, and paired
  bootstrap tooling.
- `benchmarks/results/` — compact committed JSON summaries from completed runs.
- Large full-logit caches (`*.npy`, raw llama.cpp captures, and KLD streams) —
  local output directories only; never commit them.

## Binding exact-artifact protocol

The current gate uses every prompt in
`benchmarks/prompts/mtpbench-code-general-ja.jsonl`: code, English, Japanese,
and mixed Japanese/English, including the fixed train/heldout split. BF16
greedily defines nine teacher tokens per prompt. Every candidate consumes the
same rendered prompt IDs and the same BF16 teacher contexts, producing 90
full-vocabulary rows in total.

This design answers two distinct questions:

1. **Quantization quality:** how far each exact local artifact drifts from the
   original BF16 distribution at identical contexts.
2. **Implementation correctness:** how closely two runtimes executing the same
   artifact agree, separated from quantization loss.

The 90-row suite is a precise, reproducible cross-runtime gate and ranking
signal. It is not held-out-corpus perplexity or downstream task accuracy. Add a
larger exact-artifact corpus or task suite when a deployment decision needs
more coverage; do not mix those metrics into this table.

## Metric definitions

All drift metrics use the original BF16 HF distribution as `P_ref` unless the
row explicitly names a same-artifact runtime reference:

- `KL = mean KL(P_ref || P_candidate)` over the full vocabulary; lower is
  better.
- `Top-1 agreement` is reference/candidate argmax agreement; higher is better.
  It is not downstream benchmark accuracy.
- `Top-5 overlap` compares the two top-five token sets.
- `Teacher ΔNLL` measures candidate minus BF16 NLL on the fixed BF16 teacher
  trajectory; zero is ideal.
- `Teacher PPL/BF16 = exp(Teacher ΔNLL)`; it is a trajectory diagnostic, not
  held-out-corpus PPL.
- `RMS Δp` is the RMS percentage-point change in teacher-token probability;
  lower is better.
- `BPW = active artifact bytes * 8 / 35,000,000,000` for this model family.

## Current exact-artifact results (2026-08-16)

Q4_K_M and ROCmFP4 were captured through ROCmFPX HIP commit `0d313da1849f`,
so their direct format comparison uses one runtime/backend. The additional
Q4_K_M rows quantify backend/runtime drift. PARO uses hipEngine because it is
the native packed-PARO product route.

| Exact local artifact / runtime | BPW ↓ | Mean KL ↓ | P95 KL ↓ | Max KL ↓ | Teacher PPL/BF16 ↓ | Top-1 % ↑ | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Q4_K_M / ROCmFPX HIP | 5.180 | **0.013713** | **0.067523** | **0.269131** | 1.02893 | 92.222 | matched-runtime baseline |
| ROCmFP4 STRIX_LEAN / ROCmFPX HIP | **4.354** | 0.045984 | 0.205053 | 1.272484 | **0.99676** | **97.778** | `quality-traded` |
| Q4_K_M / ROCmFPX CPU | 5.180 | **0.009005** | **0.054460** | **0.139566** | 1.01234 | 94.444 | backend diagnostic |
| Q4_K_M / hipEngine HIP | 5.180 | 0.011807 | **0.041929** | 0.286379 | 1.00740 | 95.556 | runtime control |
| PARO W4 / hipEngine HIP | 4.680 | 0.027038 | 0.143607 | 0.321606 | 1.02733 | 92.222 | `quality-traded`; runtime-correct |

### ROCmFP4 interpretation

ROCmFP4 is 15.96% smaller than the exact local Q4_K_M and preserves BF16
argmax unusually well (88/90 rows), but its full-distribution drift is 3.35x
the matched Q4_K_M row. The paired 10,000-sample prompt-block bootstrap gives a
ROCmFP4-minus-Q4 mean-KL 95% interval of `[+0.00690,+0.07168]`, above the
predeclared `+0.005` margin. Japanese is the clear distribution outlier
(ROCmFP4 mean KL `0.135091` despite 100% top-1). It is therefore
**quality-traded**, not Q4-equivalent; Japanese and task-level validation are
required before deployment.

### PARO interpretation

PARO's packed-runtime blocker is closed. The checkpoint retains Transformers'
grouped V-head order, while hipEngine had applied the tiled order used only
after llama.cpp's GGUF converter explicitly permutes V-head tensors. With the
grouped GDN sibling, hipEngine matches the independent ParoQuant/Transformers
90-row capture at mean/max KL `0.001151/0.023016` and `98.889%` top-1; two
hipEngine captures are bit-identical. Against BF16 it measures mean KL
`0.027038` and top-1 `92.222%`.

PARO is nevertheless **quality-traded** versus hipEngine Q4_K_M. Its paired
prompt-block PARO-minus-Q4 mean-KL 95% interval is
`[+0.00741,+0.02317]`, top-1 delta is `[-6.667,0.000]` percentage points, and
teacher-PPL-ratio delta is `[-0.00799,+0.05153]`; all predeclared noninferiority
gates fail and category point margins veto equivalence. PARO speed may be
measured, but every comparison must retain this quality and cross-format
caveat.

## Reproduction

Create the BF16 teacher fixture and reference cache (about 45 MiB). This CPU
step loads the 72 GB BF16 checkpoint and may take several minutes. Use a Python
environment with the optional Torch/Transformers dependencies (the current
ZBook uses `/home/lhl/miniforge3/bin/python3`):

```bash
PYTHONPATH=. python3 scripts/quant_quality/qwen36_teacher.py capture-bf16 \
  --output-dir /models/eval-results/hipengine-qwen36-teacher-v1
```

Capture hipEngine Q4_K_M and packed PARO full logits:

```bash
uv run python scripts/quant_quality/qwen36_teacher.py capture-hipengine-gguf \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --fixture /models/eval-results/hipengine-qwen36-teacher-v1/fixture.json \
  --output /models/eval-results/hipengine-qwen36-teacher-v1/q4km-hipengine.npy

uv run python scripts/quant_quality/qwen36_teacher.py capture-hipengine-paro \
  --model ~/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1 \
  --fixture /models/eval-results/hipengine-qwen36-teacher-v1/fixture.json \
  --output /models/eval-results/hipengine-qwen36-teacher-v1/paro-hipengine.npy
```

Capture the optional same-checkpoint ParoQuant/Transformers implementation
reference in the Torch environment:

```bash
PYTHONPATH=. /home/lhl/miniforge3/bin/python3 \
  scripts/quant_quality/qwen36_teacher.py capture-transformers-paro \
  --model ~/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1 \
  --model-sha256 a5c9100b17846ff0b2b507dc16dfc3ff1d622adbfc4782f30b4f1b9fac58cc60 \
  --fixture /models/eval-results/hipengine-qwen36-teacher-v1/fixture.json \
  --output /models/eval-results/hipengine-qwen36-teacher-v1/paro-transformers.npy \
  --device cuda:0 --dtype float16 --local-files-only
```

`scripts/quant_quality/llama_teacher_logits.cpp` captures the same rows from a
llama.cpp-compatible runtime. Build it against `~/ROCmFPX`, run it once for
Q4_K_M and once for ROCmFP4, then use `register-raw` to create manifests. This
same-runtime Q4_K_M control separates format drift from runtime drift.

Compare each candidate manifest to `bf16.manifest.json`:

```bash
uv run python scripts/quant_quality/qwen36_teacher.py compare \
  --fixture /models/eval-results/hipengine-qwen36-teacher-v1/fixture.json \
  --reference-manifest /models/eval-results/hipengine-qwen36-teacher-v1/bf16.manifest.json \
  --candidate-manifest CANDIDATE.manifest.json \
  --output benchmarks/results/DATE-zbook-qwen36-CANDIDATE-teacher-quality.json
```

## Acceptance and reporting

- Require exact fixture and row-manifest hashes before comparing logits.
- Require every scored row to be finite.
- Report overall, train/heldout, and all four category metrics.
- Use paired prompt-block bootstrap intervals for Q4-equivalence claims.
- Keep same-artifact runtime controls separate from BF16 quantization drift.
- A quantization-quality regression cannot be hidden by a speed win.
- Record exact model revision/hash, runtime revision, hardware, command, token
  fixture hash, and result scope for every retained row.
- Treat downstream tasks and larger exact-artifact corpora as additive evidence,
  never as permission to relabel a failed exact-artifact gate.
