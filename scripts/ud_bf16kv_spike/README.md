# Matched-KV precision spike (K_M, 2026-09-08)

These are the exact diagnostic sources used on zbook / Radeon 8060S, preserved
for the primary coder. They are campaign-specific scripts, not serving APIs.
No production path or inference score is changed. They depend on the original
captures in `/tmp` and the published K_M model at `/models/gguf/`.

Working data: `/tmp/ud-bf16kv-spike-20260908/` (about 2.0 GiB of raw boundary
captures, intentionally uncommitted). The compact result artifact records hashes.
The sources executed originally lived in that directory; these copies preserve
their bytes. `compare.py` supplies the shared binary parser and state-layout map.

The teacher links the read-only llama.cpp checkout at
`17252c769a63c1cb650ce98ae309cf4de0da7778`. `teacher.cpp` extends the previous
`/tmp/ud-llama-layer-capture.cpp` callback harness. CPU recurrence references use
`src/models/qwen35.cpp`, `src/models/delta-net-base.cpp`, and
`ggml/src/ggml-cuda/gated_delta_net.cu` at that pin. HIP state orientation and
convolution history indexing come from the in-tree `linear_attn/gdn.hip` and
`linear_attn/conv.hip`. No external repo or production kernel was edited.

## Reproduce

Run from the repository root. Use a new output directory or preserve existing
captures before intentionally rerunning capture scripts: paths are fixed in the
sources and the scripts overwrite their outputs. Run GPU commands sequentially.
The two-prompt teacher/hipEngine capture pair took several minutes; the separate
intervention pass takes several more. CPU analysis takes seconds.

Create the two-prompt wire from the existing baseline, preserving all nine
output rows (and therefore the original context/batch geometry):

```python
from pathlib import Path
import struct
import numpy as np
root = Path('/tmp/ud-bf16kv-spike-20260908')
root.mkdir(exist_ok=True)
z = np.load('/tmp/ud-q56-context512-M-baseline.npz')
with (root / 'input.bin').open('wb') as f:
    f.write(b'Q36Q' + struct.pack('<II', 1, 2))
    for p in [10, 4]:
        f.write(struct.pack('<II', 512, 9))
        f.write(z['inputs'][p].astype('<i4').tobytes())
        f.write(z['tokens'][9*p:9*(p+1)].astype('<i4').tobytes())
```

```bash
g++ -std=c++17 -O2 scripts/ud_bf16kv_spike/teacher.cpp \
  -I/home/lhl/llama.cpp/llama.cpp-hip/include \
  -I/home/lhl/llama.cpp/llama.cpp-hip/ggml/include \
  -L/home/lhl/llama.cpp/llama.cpp-hip/build-hip/bin \
  -Wl,-rpath,/home/lhl/llama.cpp/llama.cpp-hip/build-hip/bin \
  -lllama -lggml-base -o /tmp/ud-bf16kv-spike-20260908/teacher
/tmp/ud-bf16kv-spike-20260908/teacher \
  /models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf \
  /tmp/ud-bf16kv-spike-20260908/input.bin \
  /tmp/ud-bf16kv-spike-20260908/teacher.f32 16 999
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 HIPENGINE_COMPILER_VERSION_FILE=/tmp/ud-hipcc-version.txt \
  .venv/bin/python -u scripts/ud_bf16kv_spike/capture.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ud_bf16kv_spike/compare.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ud_bf16kv_spike/replay.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ud_bf16kv_spike/norm_replay.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ud_bf16kv_spike/hip_replay.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 HIPENGINE_COMPILER_VERSION_FILE=/tmp/ud-hipcc-version.txt \
  .venv/bin/python -u scripts/ud_bf16kv_spike/intervene.py
```

## Interpretation constraints

- Teacher and hipEngine captures each match their prior 18 full-logit rows
  bitwise. The HIP resident manifest also matches the earlier residual probe.
- Teacher state is `[head, value, key]`; HIP state is `[head, key, value]`.
  Transpose explicitly. The teacher-input F64 recurrence validates this mapping.
- HIP convolution history has four slots; the next update discards slot 0 and
  consumes slots 1–3. Teacher `conv_input` holds those three plus the current QKV.
- `replay.py` substitutes only HIP incoming recurrent state into otherwise
  identical teacher inputs. This is a local sensitivity experiment.
- `hip_replay.py` reconstructs alpha/beta from decoded F32 weights, BF16 inputs
  and BF16 output rounding. Those two projection outputs were not captured;
  differences near BF16 rounding boundaries are not a proven kernel defect.
- `intervene.py` uses the same live HIP prefix KV and replaces only incoming
  recurrent/convolution state at one position. Re-execution overwrites current
  KV slots. Original logits must match before and after the interventions.
  This deliberate private-position rewind is diagnostic code only.
- No raw K/V tensors or attention probabilities were captured. No K_S state
  intervention, full-suite correction, performance result or promotion is claimed.
