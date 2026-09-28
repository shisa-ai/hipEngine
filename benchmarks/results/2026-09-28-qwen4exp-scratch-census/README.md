# Current scratch metadata census

All 70 distinct project kernel symbols in the September 28 prefill/decode
captures match the private-segment size reported by their corresponding
current cached code objects. Two ROCm internal copy/fill symbols report zero
scratch in the trace but remain unresolved by this project-library census.

The only nonzero project entries are:

| Symbol | Private segment bytes |
| --- | ---: |
| `qwen4_exp_gdn_prefill_f32_kernel<true,true>` | 36 |
| `q8_0_dense_wide_kernel<_Float16,128,256,64,64,true>` | 92 |

Every other traced project symbol has zero private segment. The full 72-symbol
mapping and 23 library/code-object hashes are in [artifact.json](artifact.json).
The captures cover the Qwen3.8-Flash-Next `UD-Q4_K_XL` model with the production
profile, Q8_0 dense and mixed MoE routes on Framework Radeon 8060S/gfx1151, physical machine
`55ea6c509d0b49eea8de7094a1023668`. See the
[current-attribution packet](../2026-09-28-qwen4exp-current-attribution/README.md)
for exact workload shapes, manifest, commands and trace correctness checks.
This metadata-only operation does not launch GPU kernels or produce rates.

## Provenance limit

The library inventory was resolved from current source build planners with
`HIPENGINE_HIP_ARCH=gfx1151` and the original cached compiler-version text.
The tool copies each `.so` to a temporary directory, extracts its gfx1151
code object, reads AMDGPU notes and reconciles exact demangled symbols and
private-segment sizes against both traces. No compilation is performed.

The trace did not record loaded-library hashes. Accordingly,
`loaded_library_provenance_verified` is **false**: this is current-build
metadata consistency, not retroactive proof of historical binary identity.
`project_metadata_complete` is true, but `all_metadata_complete` is false
because the two runtime symbols are unresolved. Neither flag closes X-5's
representative decode-victim or pristine first-scratch causality work.

## Commands and checks

Recreate paths in the inventory from its named modules/functions:

```bash
HIPENGINE_HIP_ARCH=gfx1151 .venv/bin/python - <<'PY'
import importlib, json
from pathlib import Path
p = Path('benchmarks/results/2026-09-28-qwen4exp-scratch-census/libraries.json')
rows = json.loads(p.read_text())
version = Path('/tmp/qwen4exp-ledger-20260928-current/hipcc-version.txt').read_text().strip()
for row in rows:
    module = importlib.import_module(row['module'])
    plan = getattr(module, row['function'])(compiler_version=version)
    assert str(plan.output_path) == row['path'] and plan.output_path.exists()
print('all 23 current plans match inventory')
PY
python3 scripts/qwen4exp_scratch_census.py \
  --libraries-json benchmarks/results/2026-09-28-qwen4exp-scratch-census/libraries.json \
  --trace /tmp/qwen4exp-ledger-20260928-current/role-trace/gfx1151/3033290_kernel_trace.csv \
  --trace /tmp/qwen4exp-ledger-20260928-current/decode-trace/gfx1151/3039091_kernel_trace.csv \
  --llvm-bin /home/lhl/miniforge3/envs/therock10-staging-20260828/lib/python3.12/site-packages/_rocm_sdk_core/lib/llvm/bin \
  --output /tmp/qwen4exp-scratch-audit/full-census.json
.venv/bin/python -m pytest -q tests/test_unit_qwen4exp_scratch_census.py
```

All four unit tests pass: metadata parsing, exact size consistency,
missing/ambiguous/mismatched entries, empty coverage and runtime exceptions.
The tool exits nonzero on incomplete project coverage and always preserves
runtime omissions in the report.
