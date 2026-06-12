#!/usr/bin/env python3
"""Compare cycle-1 MTP verifier layer-0 state production with serial c1.

This is a correctness-only diagnostic for the 35B MTP D64 drift lane.  It
starts two clean resident sessions from the same prompt, runs only layer 0 of a
serial c1 decode step on one session, then runs the normal B+1 verifier on the
other session and compares row-0 linear-attention intermediates plus Conv/GDN
state outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from hipengine.speculative import MTP_CHAIN_CANDIDATE_BUDGETS
from scripts.mtp_prompt_suite_economics import DEFAULT_MODEL, DEFAULT_PROMPTS, PROMPT_RENDER_MODES
from scripts.mtp_state_drift_audit import (
    _compare_arrays,
    _copy_tensor_host,
    _copy_tensor_slice_host,
    _parse_csv_ints,
    _prefill_serial,
    _resolve_prompt_tokens,
    _target_batch,
)


def _copy_scratch_row(
    session: Qwen35ParoResidentSession,
    tensor: Tensor,
    *,
    row: int,
) -> np.ndarray:
    if int(tensor.shape[0]) <= int(row):
        raise ValueError(f"scratch row {row} outside tensor shape {tensor.shape}")
    row_shape = tuple(int(dim) for dim in tensor.shape[1:])
    row_nbytes = int(np.prod(row_shape, dtype=np.int64)) * tensor.dtype.itemsize
    return _copy_tensor_slice_host(
        session,
        ptr=int(tensor.ptr) + int(row) * row_nbytes,
        dtype=tensor.dtype,
        shape=row_shape,
    )


def _compare_named(
    *,
    name: str,
    left: np.ndarray,
    right: np.ndarray,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    record = _compare_arrays(left, right)
    record.update({"name": name, "left": left_label, "right": right_label})
    return record


def _first_failed(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    return next((record for record in records if not bool(record.get("passed", False))), None)


def _compare_layer0_states(
    serial: Qwen35ParoResidentSession,
    verifier: Qwen35ParoResidentSession,
    *,
    serial_conv: Tensor,
    serial_recurrent: Tensor,
    verifier_row: int,
) -> list[dict[str, Any]]:
    verifier_scratch = verifier.linear_scratch[0]
    verifier_conv, verifier_recurrent = verifier._slot_linear_state(0, 0)
    records = [
        _compare_named(
            name="conv_state_serial_vs_verifier_scratch",
            left=_copy_tensor_host(serial, serial_conv),
            right=_copy_scratch_row(verifier, verifier_scratch.tree_conv_state, row=verifier_row),
            left_label="serial_c1_resident_layer0_conv",
            right_label=f"verifier_tree_conv_state_row_{verifier_row}",
        ),
        _compare_named(
            name="recurrent_state_serial_vs_verifier_scratch",
            left=_copy_tensor_host(serial, serial_recurrent),
            right=_copy_scratch_row(verifier, verifier_scratch.tree_recurrent_state, row=verifier_row),
            left_label="serial_c1_resident_layer0_recurrent",
            right_label=f"verifier_tree_recurrent_state_row_{verifier_row}",
        ),
        _compare_named(
            name="conv_state_verifier_scratch_vs_resident",
            left=_copy_scratch_row(verifier, verifier_scratch.tree_conv_state, row=verifier_row),
            right=_copy_tensor_host(verifier, verifier_conv),
            left_label=f"verifier_tree_conv_state_row_{verifier_row}",
            right_label="verifier_resident_layer0_conv",
        ),
        _compare_named(
            name="recurrent_state_verifier_scratch_vs_resident",
            left=_copy_scratch_row(verifier, verifier_scratch.tree_recurrent_state, row=verifier_row),
            right=_copy_tensor_host(verifier, verifier_recurrent),
            left_label=f"verifier_tree_recurrent_state_row_{verifier_row}",
            right_label="verifier_resident_layer0_recurrent",
        ),
    ]
    return records


def _compare_layer0_intermediates(
    serial: Qwen35ParoResidentSession,
    verifier: Qwen35ParoResidentSession,
    *,
    verifier_row: int,
) -> list[dict[str, Any]]:
    serial_scratch = serial.linear_scratch[0]
    verifier_scratch = verifier.linear_scratch[0]
    stage_names = (
        "attn_input",
        "qkv_rot",
        "z_rot",
        "qkv",
        "z",
        "a",
        "b",
        "conv_out",
        "recurrent_out",
        "recurrent_bf16",
        "out_rot",
        "out_proj",
    )
    records: list[dict[str, Any]] = []
    for name in stage_names:
        serial_tensor = getattr(serial_scratch, name)
        verifier_tensor = getattr(verifier_scratch, name)
        records.append(
            _compare_named(
                name=name,
                left=_copy_scratch_row(serial, serial_tensor, row=0),
                right=_copy_scratch_row(verifier, verifier_tensor, row=verifier_row),
                left_label=f"serial_c1_{name}_row_0",
                right_label=f"verifier_{name}_row_{verifier_row}",
            )
        )
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.candidate_budget) not in MTP_CHAIN_CANDIDATE_BUDGETS:
        raise ValueError(f"candidate budget must be one of {sorted(MTP_CHAIN_CANDIDATE_BUDGETS)}")
    if str(args.chain_attn_mode) == "decode_batched" and str(args.graph_mode) != "off":
        raise ValueError("decode_batched currently requires graph-mode=off")
    prompt_name, prompt_tokens, prompt_meta = _resolve_prompt_tokens(args)
    candidate_tokens = _parse_csv_ints(str(args.candidate_tokens))
    if not candidate_tokens:
        candidate_tokens = [int(args.candidate_token)]
    candidate_budget = int(args.candidate_budget)
    if len(candidate_tokens) > candidate_budget:
        raise ValueError("--candidate-tokens cannot exceed --candidate-budget")
    active_count = len(candidate_tokens)
    rows = candidate_budget + 1
    verifier_row = int(args.verifier_row)
    if verifier_row < 0 or verifier_row >= rows:
        raise ValueError("--verifier-row outside verifier rows")

    runner = Qwen35ParoNextTokenRunner(Path(args.model), backend=str(args.backend))
    max_sequence = len(prompt_tokens) + int(args.decode_tokens) + candidate_budget + 4
    started = time.perf_counter()

    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_batch_size=1,
    ) as serial_session, Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_batch_size=rows,
    ) as verifier_session:
        if str(serial_session.config.layer_types[0]) != "linear_attention":
            raise RuntimeError("diagnostic expects layer 0 to be linear_attention")

        serial_root = _prefill_serial(serial_session, prompt_tokens)
        verifier_root = _prefill_serial(verifier_session, prompt_tokens)
        context = len(prompt_tokens)
        pre_serial_conv, pre_serial_recurrent = serial_session._slot_linear_state(0, 0)
        pre_verifier_conv, pre_verifier_recurrent = verifier_session._slot_linear_state(0, 0)
        pre_state_records = [
            _compare_named(
                name="pre_conv_state",
                left=_copy_tensor_host(serial_session, pre_serial_conv),
                right=_copy_tensor_host(verifier_session, pre_verifier_conv),
                left_label="serial_prefill_layer0_conv",
                right_label="verifier_prefill_layer0_conv",
            ),
            _compare_named(
                name="pre_recurrent_state",
                left=_copy_tensor_host(serial_session, pre_serial_recurrent),
                right=_copy_tensor_host(verifier_session, pre_verifier_recurrent),
                left_label="serial_prefill_layer0_recurrent",
                right_label="verifier_prefill_layer0_recurrent",
            ),
        ]

        serial_session._set_token_embedding(serial_root)
        serial_session._set_position(context)
        serial_state = serial_session.states[0]
        serial_conv, serial_recurrent = serial_session._slot_linear_state(0, 0)
        serial_scratch = serial_session._linear_decode_scratch(0, serial_state)
        serial_session.linear_scratch[0] = serial_scratch
        serial_moe_scratch = serial_session._mlp_decode_scratch(0, serial_state)
        serial_session.moe_scratch[0] = serial_moe_scratch
        serial_state.run_linear_attention_moe_c1_layer_fp16(
            serial_session.hidden,
            conv_state=serial_conv,
            recurrent_state=serial_recurrent,
            linear_scratch=serial_scratch,
            moe_scratch=serial_moe_scratch,
            library=serial_session.libraries,
        )
        serial_session.runtime.stream_synchronize(0)

        target_batch = _target_batch(
            verifier_root,
            context,
            candidate_tokens,
            active_count,
            candidate_budget=candidate_budget,
        )
        verifier_no_capture = Tensor.from_handle(0, (rows, 0), DType.BF16, Device("hip", 0))
        verify = verifier_session.verify_chain_bulk_and_commit(
            target_batch,
            base_slot=0,
            capture_layer_ids=(),
            capture_hidden_concat=verifier_no_capture,
            capture_row_start=0,
            chain_attn_mode=str(args.chain_attn_mode),
            graph_mode=str(args.graph_mode),
            canonicalize_after=False,
        )
        verifier_top1, verifier_values = verifier_session._read_verify_top1(int(verify.rows))

        state_records = _compare_layer0_states(
            serial_session,
            verifier_session,
            serial_conv=serial_conv,
            serial_recurrent=serial_recurrent,
            verifier_row=verifier_row,
        )
        intermediate_records = _compare_layer0_intermediates(
            serial_session,
            verifier_session,
            verifier_row=verifier_row,
        )

    seconds = time.perf_counter() - started
    all_records = [*pre_state_records, *intermediate_records, *state_records]
    first = _first_failed(all_records)
    return {
        "status": "matched" if first is None else "mismatch",
        "performance_claim": False,
        "model": str(args.model),
        "backend": str(args.backend),
        "workload": {
            "prompt_name": prompt_name,
            "prompt_tokens": len(prompt_tokens),
            "decode_tokens": int(args.decode_tokens),
            "candidate_budget": candidate_budget,
            "candidate_tokens": [int(token) for token in candidate_tokens],
            "active_count": int(active_count),
            "chain_attn_mode": str(args.chain_attn_mode),
            "graph_mode": str(args.graph_mode),
            "verifier_row": verifier_row,
        },
        "prompt": prompt_meta,
        "seconds": float(seconds),
        "roots": {
            "serial": int(serial_root),
            "verifier": int(verifier_root),
            "match": int(serial_root) == int(verifier_root),
        },
        "verify": {
            "accepted": int(verify.accepted_count),
            "commit_row": int(verify.commit_row),
            "commit_position": int(verify.commit_position),
            "next_token": int(verify.next_token) if verify.next_token is not None else None,
            "target_top1": [int(token) for token in verifier_top1],
            "target_top1_values": [float(value) for value in verifier_values],
            "graph": verify.graph,
        },
        "pre_state_compare": {
            "passed": _first_failed(pre_state_records) is None,
            "records": pre_state_records,
        },
        "intermediate_compare": {
            "passed": _first_failed(intermediate_records) is None,
            "first_mismatch": _first_failed(intermediate_records),
            "records": intermediate_records,
        },
        "state_compare": {
            "passed": _first_failed(state_records) is None,
            "first_mismatch": _first_failed(state_records),
            "records": state_records,
        },
        "first_mismatch": first,
        "note": "Correctness diagnostic only: compares serial c1 layer-0 producer outputs against verifier row-0 outputs after one clean prompt prefill.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-name", default="translation")
    parser.add_argument("--prompt-render", choices=PROMPT_RENDER_MODES, default="raw")
    parser.add_argument("--prompt-tokens", default="", help="comma-separated token ids; bypasses --prompt-name")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--candidate-budget", type=int, default=1)
    parser.add_argument("--candidate-token", type=int, default=760)
    parser.add_argument("--candidate-tokens", default="", help="comma-separated candidate token ids; overrides --candidate-token")
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--chain-attn-mode", choices=("c1_loop", "batched", "decode_batched"), default="c1_loop")
    parser.add_argument("--graph-mode", choices=("off", "auto", "validate"), default="off")
    parser.add_argument("--verifier-row", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    payload = run(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if payload["status"] in {"matched", "mismatch"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
