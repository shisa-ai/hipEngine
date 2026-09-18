#!/usr/bin/env python3
"""Prove which long-context verifier route ran, alongside the gate's verdict.

Wraps ``scripts/gguf_mtp_long_context_gate.py`` so the same in-process run that
compares the production verifier against the serial-exact teacher also records,
per layer, whether the staged chain or the scalar attention owner ran, how many
rows the FFN owner received, and what the staged split-K plan resolved to.
Teacher equality alone cannot tell the two routes apart, because both must
produce the same tokens.

Every gate flag is passed through unchanged, so a published packet can be
re-run with this wrapper and compared to its own recorded numbers:

    python3 scripts/gguf_mtp_route_owner_probe.py \
      --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
      --cycle-ends 1024,1025 --candidate-budgets 1,2,3 \
      --acceptance-cycle-ends 1024 --generation-contexts '' \
      --out <out>.json

``HIPENGINE_LC_PROBE_FORCE_SCALAR=1`` keeps the per-row views but forces the
scalar row-wise attention owner, which separates a row-view defect from a
staged-chain defect. ``HIPENGINE_LC_PROBE_OUT=<path>`` also writes the probe
report to a file; the report is always printed to stdout as one
``PATH_PROBE <json>`` line.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gguf_mtp_long_context_gate as gate  # noqa: E402
import hipengine.runtime.qwen35_gguf_runner as qgr  # noqa: E402

Runner = qgr.Qwen35GGUFFullStackRunner

_PROBE_FORCE_SCALAR_ENV = "HIPENGINE_LC_PROBE_FORCE_SCALAR"
_PROBE_OUT_ENV = "HIPENGINE_LC_PROBE_OUT"

observed: dict[str, Any] = {
    "staged_chain_calls": [],
    "scalar_attn_calls": [],
    "ffn_rows_by_layer": {},
    "split_plan_calls": [],
}

_original_leaf = qgr._staged_full_attention_row_leaf


def _leaf_probe(config, *, backend, scratch, row_scratch, position):
    result = _original_leaf(
        config,
        backend=backend,
        scratch=scratch,
        row_scratch=row_scratch,
        position=position,
    )
    observed["split_plan_calls"].append(
        {
            "position": int(position),
            "active_context": int(position) + 1,
            "split_needed": qgr._use_gguf_full_attention_split_decode(int(position) + 1),
            "scratch_split_count": int(getattr(scratch, "full_attn_split_count", -1)),
            "num_splits": None if result is None else int(result[1]),
            "resolved": result is not None,
        }
    )
    return result


qgr._staged_full_attention_row_leaf = _leaf_probe

if os.environ.get(_PROBE_FORCE_SCALAR_ENV) not in (None, "", "0"):
    # Diagnostic arm: keep the per-row views but force the scalar row-wise
    # attention owner, to separate a row-view defect from a staged-chain defect.
    qgr._staged_full_attention_rows_ready = lambda *a, **k: False

_original_staged = Runner._run_full_attention_attn_chain_rows_exact


def _staged_probe(self, layer_id, *args, **kwargs):
    observed["staged_chain_calls"].append(
        {"layer": int(layer_id), "rows": kwargs.get("rows")}
    )
    return _original_staged(self, layer_id, *args, **kwargs)


Runner._run_full_attention_attn_chain_rows_exact = _staged_probe

_original_scalar = Runner._run_full_attention_attn_only
_stacks: list[list[str]] = []


def _scalar_probe(self, layer_id, *args, **kwargs):
    observed["scalar_attn_calls"].append(int(layer_id))
    if len(_stacks) < 2:
        _stacks.append(
            [f"{frame.name}:{frame.lineno}" for frame in traceback.extract_stack()[-9:-1]]
        )
    return _original_scalar(self, layer_id, *args, **kwargs)


Runner._run_full_attention_attn_only = _scalar_probe

_original_ffn = Runner._run_post_attention_ffn_rows


def _ffn_probe(self, layer_id, *args, **kwargs):
    rows = kwargs.get("rows")
    if rows is None and len(args) > 3:
        rows = args[3]
    observed["ffn_rows_by_layer"].setdefault(int(layer_id), []).append(rows)
    return _original_ffn(self, layer_id, *args, **kwargs)


Runner._run_post_attention_ffn_rows = _ffn_probe

status = gate.main(sys.argv[1:])
report = {
    "gate_status": status,
    "forced_scalar": os.environ.get(_PROBE_FORCE_SCALAR_ENV) not in (None, "", "0"),
    "staged_chain_calls": observed["staged_chain_calls"][:8],
    "staged_chain_total": len(observed["staged_chain_calls"]),
    "scalar_attn_total": len(observed["scalar_attn_calls"]),
    "scalar_attn_layers": sorted(set(observed["scalar_attn_calls"])),
    "ffn_rows_by_layer": {
        str(layer): rows for layer, rows in sorted(observed["ffn_rows_by_layer"].items())
    },
    "split_plan_calls": observed["split_plan_calls"][:8],
    "split_plan_total": len(observed["split_plan_calls"]),
    "scalar_attn_stacks": _stacks,
}
out = os.environ.get(_PROBE_OUT_ENV)
if out:
    Path(out).write_text(json.dumps(report, indent=2, sort_keys=True))
print("PATH_PROBE " + json.dumps(report, sort_keys=True))
raise SystemExit(status)
