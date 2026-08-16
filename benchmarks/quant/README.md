# Quantization quality

This directory is the canonical index for hipEngine quantization-quality work.
It is deliberately separate from throughput/latency tables: speed and model
fidelity answer different questions and neither substitutes for the other.

## Repository layout

- `benchmarks/quant/` — protocols, provenance, compact comparison tables, and
  small token fixtures.
- `scripts/quant_quality/` — reusable capture and comparison tooling.
- `benchmarks/results/` — compact committed JSON summaries from completed runs.
- Large full-logit caches (`*.npy`, raw llama.cpp captures, and KLD streams) —
  local output directories only; never commit them.

## Evidence tiers

Do not compare absolute PPL values across tiers.

1. **Canonical rolling-corpus gate** — the primary quantization-quality result.
   It scores a held-out multilingual/code/math corpus with `ctx=2048`, a
   1025-token warmup, 1023 scored positions/window, and stride 1023. Every
   candidate uses the original BF16 HF model and identical token IDs/positions.
2. **Portable BF16-teacher prompt suite** — a fast cross-runtime gate over all
   ten prompts in `benchmarks/prompts/mtpbench-code-general-ja.jsonl`. BF16
   greedily defines nine teacher tokens per prompt; all candidates consume the
   same prompt IDs and teacher contexts. It reports full-distribution KL,
   teacher-trajectory ΔNLL, top-1 agreement, top-k overlap, and RMS Δp. Its
   teacher PPL is **not** held-out-corpus PPL.
3. **Task quality** — execution/human/task metrics for code, multilingual,
   tool-use, or long-context behavior. These are a later complement, not a
   replacement for distribution metrics.

## Metric definitions

All drift metrics use the original BF16 HF distribution as `P_ref`:

- `PPL = exp(mean(-log P_model(true token)))`; lower is better.
- `ΔNLL = mean_NLL_candidate - mean_NLL_BF16`; zero is ideal.
- `KL = mean KL(P_BF16 || P_candidate)` over the full vocabulary; zero is ideal.
- `Top-1 agreement` is BF16/candidate argmax agreement; higher is better. It is
  not downstream benchmark accuracy.
- `RMS Δp` is the RMS percentage-point change in true-token probability; lower
  is better.
- `BPW = active artifact bytes * 8 / 35,000,000,000` for this model family.

These definitions and the rolling layout mirror
`~/paroquant/docs/QUANTIZATION-QUALITY.md` at paroquant commit `7bfbf5e`.

## Existing canonical Qwen3.6-35B-A3B evidence

The downloaded PARO checkpoint contains its original canonical evaluation
payload. The GGUF row below comes from the same ParoQuant tx4/quality3 protocol,
but its historical artifact is not yet hash-matched to the current local GGUF;
treat that row as a recipe-level control until the portable exact-artifact run
lands.

| Model | Artifact / status | BPW ↓ | PPL ↓ | ΔNLL ↓ | KL nats ↓ | RMS Δp % ↓ | Top-1 % ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Original BF16 HF | revision `995ad96e` | 16.435 | 6.5590 | +0.000000 | 0.000000 | 0.000 | 100.000 |
| Local GGUF UD-Q4_K_M | exact 21.107 GiB artifact | 5.180 | pending | pending | pending | pending | pending |
| GGUF UD-Q4_K_M | historical same-protocol control | 5.059 | **6.5643** | **+0.001718** | **0.010849** | pending import | **95.354** |
| PARO full8192 old+fresh rbparams e5 | local snapshot `437eba06`; canonical payload bundled | 4.680 | 6.6090 | +0.007594 | 0.027939 | **4.646** | 92.856 |
| ROCmFP4 STRIX_LEAN | exact local 17.739 GiB artifact | **4.354** | pending | pending | pending | pending | pending |

Measured canonical interpretation: the historical Q4_K_M control preserves the
BF16 distribution better than PARO full8192 (`0.010849` vs `0.027939` mean KL,
`95.354%` vs `92.856%` top-1). For the exact local artifacts, ROCmFP4 is smallest
at 17.739 GiB / 4.354 BPW, then PARO at 19.068 GiB / 4.680 BPW, then Q4_K_M at
21.107 GiB / 5.180 BPW. PARO's latest calibration materially improves earlier
PARO runs. ROCmFP4's canonical corpus row remains pending; the portable exact-
artifact gate below supplies the current diagnostic ranking.

## Portable exact-artifact results (2026-08-16)

These are the 90-row BF16-teacher results, not canonical corpus PPL. Q4_K_M and
ROCmFP4 were both captured through ROCmFPX HIP commit `0d313da1849f` so their
direct format comparison uses one runtime/backend.

| Exact local artifact / runtime | BPW ↓ | Mean KL ↓ | P95 KL ↓ | Max KL ↓ | Teacher PPL/BF16 ↓ | Top-1 % ↑ | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Q4_K_M / ROCmFPX HIP | 5.180 | **0.013713** | **0.067523** | **0.269131** | 1.02893 | 92.222 | matched-runtime baseline |
| ROCmFP4 STRIX_LEAN / ROCmFPX HIP | **4.354** | 0.045984 | 0.205053 | 1.272484 | **0.99676** | **97.778** | `quality-traded` |
| Q4_K_M / ROCmFPX CPU | 5.180 | **0.009005** | **0.054460** | **0.139566** | 1.01234 | 94.444 | backend diagnostic |
| Q4_K_M / hipEngine HIP | 5.180 | 0.011807 | **0.041929** | 0.286379 | 1.00740 | 95.556 | runtime control |
| PARO W4 / hipEngine HIP | 4.680 | 12.072252 | 18.196675 | 21.878865 | 217649.82 | 0.000 | **implementation blocked** |

ROCmFP4 is 15.96% smaller than the exact local Q4_K_M and preserves BF16
argmax unusually well (88/90 rows), but its full-distribution drift is 3.35x
the matched Q4_K_M row. The paired 10,000-sample prompt-block bootstrap gives a
ROCmFP4-minus-Q4 mean-KL 95% interval of `[+0.00690, +0.07168]`, above the
predeclared `+0.005` margin. Japanese is the clear distribution outlier
(ROCmFP4 mean KL `0.135091` despite 100% top-1). It is therefore
**quality-traded**, not Q4-equivalent; downstream Japanese and calibration/task
tests are required before deployment.

The PARO checkpoint itself is not bad: its bundled 129,921-token canonical
result is KL `0.027939`, top-1 `92.856%`. The exact local checkpoint's hipEngine
full logits are self-consistent with hipEngine sampling but disagree with BF16
on all 90 rows. That is a packed-loader/runtime correctness blocker. The native
ParoQuant/Transformers cross-check loaded the checkpoint but its ROCm rotation
extension failed with `HIP error: invalid device function` on gfx1151, so the
bundled canonical payload remains the independent quant-quality authority.

Evidence:
[`2026-08-16-zbook-qwen36-quant-quality.json`](../results/2026-08-16-zbook-qwen36-quant-quality.json).

Canonical PARO source:

```text
~/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/
  snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1/
  canonical_eval_results.json
```

## Portable exact-artifact runbook

Create the BF16 teacher fixture and reference cache (about 45 MiB). This CPU
step loads the 72 GB BF16 checkpoint and may take several minutes:

Use a Python environment with the optional Torch/Transformers dependencies
(the current ZBook uses `/home/lhl/miniforge3/bin/python3`):

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

`scripts/quant_quality/llama_teacher_logits.cpp` captures the same rows from a
llama.cpp-compatible runtime. Build it against `~/ROCmFPX`, run it once for
Q4_K_M and once for ROCmFP4, then use `register-raw` to create manifests. This
same-runtime Q4_K_M control separates format drift from runtime drift.

Finally compare each candidate manifest to `bf16.manifest.json`:

```bash
uv run python scripts/quant_quality/qwen36_teacher.py compare \
  --fixture /models/eval-results/hipengine-qwen36-teacher-v1/fixture.json \
  --reference-manifest /models/eval-results/hipengine-qwen36-teacher-v1/bf16.manifest.json \
  --candidate-manifest CANDIDATE.manifest.json \
  --output benchmarks/results/DATE-zbook-qwen36-CANDIDATE-teacher-quality.json
```

## Acceptance and reporting

- Require exact fixture and row-manifest hashes before comparing logits.
- Report overall and category-level metrics across all four prompt categories.
- Treat the portable suite as a smoke/ranking signal because it has only 90
  scored rows; canonical held-out-corpus evidence remains stronger.
- A quantization-quality regression cannot be hidden by a speed win.
- Every retained result records exact model revision/hash, runtime revision,
  hardware, command, token fixture hash, and whether the result is canonical or
  portable.
