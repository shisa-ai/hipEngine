# UD Quant Review Reproduction

Date: 2026-09-06. Source: hipEngine
`bf46abefc5ad8fbb00608cd5fb274ca1af21f716`.
Companion to [the campaign](UD-QUANTS.md).

This is analysis code, not a loader implementation. It reads metadata from
complete files, resolves actual AR model-map slots, and evaluates existing
allocation formulas without constructing a device/runtime or loading kernels.
All three local files had complete byte ranges by the final review. A complete
byte range is not a content checksum or proof of a successful download.

Run from the repository root in an environment without `HIPENGINE_*` overrides:

```bash
awk '/^```python$/{p=1;next} p && /^```$/{exit} p' \
  docs/UD-QUANTS-REPRO.md > /tmp/ud-review-reproduce.py
PYTHONPATH=. .venv/bin/python /tmp/ud-review-reproduce.py \
  /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  /models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf \
  /models/gguf/Qwen3.8-27B-UD-Q4_K_S.gguf \
  > /tmp/ud-review-reproduce.json
```

The output's `native_refusals_gib` is a hypothetical lower-bound weight plan
that keeps refused weights at their compressed source size. `bf16_refusals_gib`
instead expands those refused weights to BF16. Neither is a working load.
`repack_iq4` additionally assumes a new IQ4_XS resident costs exactly its source
bytes, without claiming such a consumer exists. Every other accepted spec uses
`planned_qwen35_gguf_weight_allocation_nbytes`, including sidecars.
The repack scenarios deliberately override the current file-global veto; they
do not implement or certify the override.

```python
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

from hipengine.loading.gguf import scan_gguf
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_materialize import (
    _spec_for_tensor, planned_qwen35_gguf_weight_allocation_nbytes,
)

assert not any(k.startswith("HIPENGINE_") for k in os.environ)
caps_map = {
    "dense_q4_t16": "GGUF_DENSE_Q4_T16",
    "dense_q4_qmicro_t16_gate_up": "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP",
    "dense_q4_t16_attn_q_08b": "GGUF_DENSE_Q4_T16_ATTN_Q_08B",
    "dense_q5_t16_ssm_out": "GGUF_DENSE_Q5_T16_SSM_OUT",
    "dense_q5_raw_mmq_ssm_out": "GGUF_C8_Q5_RAW_MMQ_SSM_OUT",
    "dense_q5_t16_ssm_out_08b": "GGUF_DENSE_Q5_T16_SSM_OUT_08B",
    "dense_q5_t16_qkv": "GGUF_DENSE_Q5_T16_QKV",
    "dense_q5_t16_h5120": "GGUF_DENSE_Q5_T16_H5120",
    "dense_q6_qmicro_planar": "GGUF_DENSE_Q6_T16_QMICRO_PLANAR",
}
report = []
for path in map(Path, sys.argv[1:]):
    info = scan_gguf(path)
    model = build_qwen35_gguf_tensor_map(info)
    entries = [(f"root.{s}", t) for s, t in model.root_tensors.items()]
    entries += [(f"layers.{l.layer_id}.{s}", t)
                for l in model.layers for s, t in l.tensors.items()]
    source_names = {t.name for _, t in entries}
    with path.open("rb") as handle:
        header_hash = hashlib.sha256(handle.read(info.tensor_data_offset)).hexdigest()
    record = {
        "file": path.name, "header_sha256": header_hash,
        "header_bytes": info.tensor_data_offset,
        "file_bytes_observed": path.stat().st_size,
        "required_file_end": max(t.data_offset + t.nbytes for t in info.tensors),
        "file_type": info.file_type_name,
        "disk_tensors": len(info.tensors), "ar_tensors": len(source_names),
        "ar_layers": model.config.block_count,
        "ignored_blocks": model.config.ignored_block_ids,
        "histogram": dict(Counter(t.ggml_type_name for t in info.tensors)),
        "source_bytes": sum(t.nbytes for t in info.tensors),
        "ar_source_bytes": sum(t.nbytes for t in info.tensors if t.name in source_names),
        "lanes": [],
    }
    raw_iq = any(t.ggml_type_name in {"IQ2_XS", "IQ3_XXS", "IQ4_XS"}
                 for layer in model.layers for t in layer.tensors.values())
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        constants = {}
        source = Path("hipengine/kernels") / backend / "__init__.py"
        for node in ast.parse(source.read_text()).body:
            if not isinstance(node, ast.Assign):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value
        flags = {k: bool(constants.get(v, False)) for k, v in caps_map.items()}
        flags["dense_q4_qmicro_t16_gate_up"] &= info.file_type_name in constants.get(
            "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES", ())
        flags["dense_q6_qmicro_planar_excluded_slots"] = frozenset(constants.get(
            "GGUF_DENSE_Q6_T16_QMICRO_PLANAR_EXCLUDED_SLOTS", ()))
        for scenario, repack, iq4 in (
            ("current", not raw_iq, False),
            ("repack", True, False),
            ("repack_iq4", True, True),
        ):
            totals, routes, rejected, seen = Counter(), Counter(), [], set()
            for slot, tensor in entries:
                try:
                    spec = _spec_for_tensor(
                        slot, tensor, decode_repack=repack,
                        contract_f32_linear=raw_iq, **flags)
                except ValueError as error:
                    key = (tensor.name, "refused")
                    if key in seen:
                        continue
                    seen.add(key)
                    totals["refused_source_bytes"] += tensor.nbytes
                    totals["refused_bf16_bytes"] += 2 * tensor.n_elements
                    rejected.append([slot, tensor.ggml_type_name, list(tensor.shape), str(error)])
                    continue
                key = (tensor.name, spec.layout)
                if key in seen:
                    continue
                seen.add(key)
                nbytes = sum(n for _, n in planned_qwen35_gguf_weight_allocation_nbytes(spec))
                layout = spec.layout
                if iq4 and tensor.ggml_type_name == "IQ4_XS":
                    nbytes, layout = tensor.nbytes, "hypothetical_raw_iq4"
                totals["accepted_bytes"] += nbytes
                totals["bf16_count"] += int(layout == "dense_bf16")
                routes[f"{tensor.ggml_type_name}:{layout}"] += 1
            record["lanes"].append({
                "backend": backend, "scenario": scenario, **totals,
                "native_refusals_gib": (totals["accepted_bytes"] + totals["refused_source_bytes"]) / 2**30,
                "bf16_refusals_gib": (totals["accepted_bytes"] + totals["refused_bf16_bytes"]) / 2**30,
                "routes": dict(routes), "refused_count": len(rejected),
                "refused": rejected if scenario == "current" else [],
            })
    report.append(record)
print(json.dumps(report, indent=2))
```

## Limits

- The AST reader is intentionally limited to the literal capability assignments
  used by these pinned sources. It is not a replacement for package policy
  resolution; a source change requires re-review, not blind reuse.
- Defaults mirror `materialize_qwen35_gguf_weights`: Q5 raw-MMQ sidecars on where
  declared, optional planar-Q5 sidecar off. No request/profile binder is run.
- Unique `(source name, layout)` allocations are counted. AR ignores block 64;
  NextN has its own planner and shared-root aliases, so do not add all block-64
  bytes mechanically to estimate an MTP session.
- These are planned weight allocations, not allocator peaks. No scratch,
  K/V, recurrent state, graph pools, attention mirrors, host staging, padding
  outside the weight formulas, or loader transient coexistence is included.
- This review's snapshot output is [UD-QUANTS-REVIEW.json](UD-QUANTS-REVIEW.json).
  No full tensor-payload hashes were read during this CPU-only analysis.

## Schema-v2 Snapshot (UD-U0 repair, 2026-09-07)

The repaired audit supersedes the inline script above for reporting. Its
schema-v2 output is [UD-QUANTS-REVIEW-v2.json](UD-QUANTS-REVIEW-v2.json)
(`schema_version: 2`: header identity over `[0, data_start)`, per-scope
allocation accounting from `planned_qwen35_gguf_weight_allocation_nbytes` with
sidecar reasons and both hypothetical refusal treatments, shared-policy
capability resolution with per-capability status, and `f32_contracted_slots`).
The artifact is the exact CLI output below, so a byte-identical regeneration
on the same pinned files is the reproduction test:

```bash
cd /home/lhl/hipEngine-ud-quants && git checkout <commit>
env -u HIPENGINE_GGUF_DECODE_REPACK -u HIPENGINE_GGUF_C8_Q5_RAW_MMQ \
  -u HIPENGINE_C8_Q5_PLANAR_DP4A \
  .venv/bin/python scripts/gguf_quant_route_audit.py \
  /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  /models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf \
  /models/gguf/Qwen3.8-27B-UD-Q4_K_S.gguf \
  --json /tmp/ud-u0-v2-full.json
```

Artifact identity (upstream revisions, payload checksum provenance, local
sizes, header hashes) is pinned in
[UD-QUANTS-U0-IDENTITY.json](UD-QUANTS-U0-IDENTITY.json). The allocation
totals of this snapshot reproduce the section-3/4 tables of
[UD-QUANTS.md](UD-QUANTS.md) exactly; per-scope refusal lists, sidecar
counts/bytes, and NextN draft scopes are additional detail the older snapshot
did not carry. Everything remains metadata-only: no payload bytes beyond the
header region are read, and no section is model admission.
