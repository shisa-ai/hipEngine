#!/usr/bin/env python3
"""Forced-prefix GGUF MTP target verifier probe.

This diagnostic replays the target session to the start of one recorded MTP
cycle, runs the target verifier on ``[prev] + draft_tokens``, and emits
row-level top-k/candidate logits.  It is meant for semantic parity debugging,
not performance measurement.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gguf_mtp_bench import GGUF_PATH, build_chat_prompt, hidden_state_summary


class ProbeError(RuntimeError):
    pass


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _set_env_bool(name: str, value: bool) -> None:
    os.environ[name] = "1" if bool(value) else "0"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ProbeError(f"expected JSON object in {path}")
    return data


def _find_cycle(cycles: list[dict[str, Any]], cycle_number: int) -> tuple[int, dict[str, Any]]:
    for index, row in enumerate(cycles):
        if int(row.get("cycle", index)) == int(cycle_number):
            return index, row
    raise ProbeError(f"cycle {cycle_number} not found in trace")


def _flatten_output_tokens(cycles: Iterable[dict[str, Any]]) -> list[int]:
    tokens: list[int] = []
    for row in cycles:
        output = row.get("output_tokens")
        if not isinstance(output, list):
            raise ProbeError("trace cycle is missing output_tokens[]")
        tokens.extend(int(token) for token in output)
    return tokens


def _parse_token_csv(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        return []
    values: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            token = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--candidate-token must be an int or comma-separated ints") from exc
        if token < 0:
            raise argparse.ArgumentTypeError("--candidate-token values must be non-negative")
        values.append(token)
    return values


def _unique_ints(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        token = int(value)
        if token not in seen:
            result.append(token)
            seen.add(token)
    return result


def _apply_trace_route_env(workload: dict[str, Any], *, decode_repack: bool | None) -> dict[str, str | None]:
    if decode_repack is None:
        decode_repack = bool(workload.get("decode_repack", workload.get("decode_repack_env") == "1"))
    _set_env_bool("HIPENGINE_GGUF_DECODE_REPACK", bool(decode_repack))

    for name in (
        "HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A",
        "HIPENGINE_GGUF_T16_SELECTED_DP4A",
        "HIPENGINE_GGUF_RAW_SELECTED_DP4A",
    ):
        _set_env_bool(name, bool(workload.get("verify_dp4a", False)))

    dense_pair = bool(workload.get("verify_dense_q8_dp4a", False))
    dense_all = bool(workload.get("verify_dense_q8_dp4a_all", False))
    dense_shared = bool(workload.get("verify_dense_q8_dp4a_shared", False))
    dense_f32 = bool(workload.get("verify_dense_q8_dp4a_f32", False))
    _set_env_bool("HIPENGINE_GGUF_Q8_0_RAW_SIDECAR", dense_pair or dense_all or dense_shared or dense_f32)
    _set_env_bool("HIPENGINE_GGUF_DENSE_Q8_DP4A", dense_pair)
    _set_env_bool("HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL", dense_all or dense_shared or dense_f32)
    _set_env_bool("HIPENGINE_GGUF_DENSE_Q8_DP4A_SHARED", dense_shared)
    _set_env_bool("HIPENGINE_GGUF_DENSE_Q8_DP4A_F32", dense_f32)

    selected_down = str(workload.get("selected_down_x8_repack", workload.get("selected_down_x8_repack_env") or "off"))
    os.environ["HIPENGINE_GGUF_SELECTED_X8_REPACK"] = selected_down
    _set_env_bool("HIPENGINE_GGUF_SELECTED_GATE_UP_X8", bool(workload.get("selected_gate_up_x8", False)))
    _set_env_bool("HIPENGINE_GGUF_SELECTED_GATE_UP_RAW", bool(workload.get("selected_gate_up_raw", False)))

    # Full row logits are the point of this probe; force the direct top-1
    # verifier lm-head route off even if the shell inherited it from a prior run.
    _set_env_bool("HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A", False)
    _set_env_bool("HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR", False)

    return {name: os.environ.get(name) for name in sorted(os.environ) if name.startswith("HIPENGINE_GGUF_")}


def _repo_provenance() -> dict[str, Any]:
    def git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).strip()
        except Exception:
            return None

    status = git(["status", "--short"])
    return {
        "root": str(REPO_ROOT),
        "commit": git(["rev-parse", "HEAD"]),
        "branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
    }


def _top_k_rows(logits: np.ndarray, *, top_k: int) -> list[dict[str, Any]]:
    row = np.ascontiguousarray(np.asarray(logits, dtype=np.float32).reshape(-1))
    if row.size == 0:
        raise ProbeError("cannot rank an empty logits row")
    if not np.all(np.isfinite(row)):
        raise FloatingPointError("logits row contains NaN or Inf")
    k = min(max(1, int(top_k)), int(row.size))
    if k == 1:
        top_idx = np.asarray([int(np.argmax(row))], dtype=np.int64)
    else:
        top_idx = np.argpartition(row, -k)[-k:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
    top_logit = float(row[int(top_idx[0])])
    max_logit = float(np.max(row))
    exp_sum = float(np.exp(row.astype(np.float64) - max_logit).sum())
    result: list[dict[str, Any]] = []
    for rank, token in enumerate(top_idx.tolist(), start=1):
        logit = float(row[int(token)])
        result.append(
            {
                "rank": int(rank),
                "token_id": int(token),
                "logit": logit,
                "margin_from_top": float(top_logit - logit),
                "prob": float(np.exp(float(logit) - max_logit) / exp_sum),
            }
        )
    return result


def _candidate_rows(logits: np.ndarray, token_ids: Sequence[int]) -> list[dict[str, Any]]:
    row = np.ascontiguousarray(np.asarray(logits, dtype=np.float32).reshape(-1))
    if row.size == 0:
        raise ProbeError("cannot score candidates from an empty logits row")
    if not np.all(np.isfinite(row)):
        raise FloatingPointError("logits row contains NaN or Inf")
    top_logit = float(np.max(row))
    max_logit = top_logit
    exp_sum = float(np.exp(row.astype(np.float64) - max_logit).sum())
    result: list[dict[str, Any]] = []
    for token in _unique_ints(token_ids):
        if token < 0 or token >= int(row.size):
            result.append({"token_id": int(token), "in_vocab": False})
            continue
        logit = float(row[token])
        rank = int(np.count_nonzero(row > logit) + 1)
        result.append(
            {
                "token_id": int(token),
                "in_vocab": True,
                "rank": rank,
                "logit": logit,
                "margin_from_top": float(top_logit - logit),
                "prob": float(np.exp(float(logit) - max_logit) / exp_sum),
            }
        )
    return result


def _llama_accept_count(draft_tokens: Sequence[int], target_tokens: Sequence[int]) -> int:
    accepted = 0
    for draft, target in zip(draft_tokens, target_tokens, strict=False):
        if int(draft) != int(target):
            break
        accepted += 1
    return accepted


def _copy_verify_logits(session: Any, rows: int) -> np.ndarray:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    if session.runner is None:
        raise ProbeError("session runner is closed")
    logits_buf = getattr(session, "_verify_logits_buf", None)
    if logits_buf is None:
        raise ProbeError(
            "target block did not populate _verify_logits_buf; "
            "disable HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A for this probe"
        )
    vocab_size = int(session.runner.vocab_size)
    logits = np.empty((int(rows), vocab_size), dtype=np.float32)
    runtime = session.runtime
    copy_device_to_host(
        host_array_ptr(logits),
        DeviceBuffer(int(logits_buf.ptr), int(logits.nbytes)),
        int(logits.nbytes),
        runtime=runtime,
    )
    return np.ascontiguousarray(logits, dtype=np.float32)


def _copy_current_hidden_seed(session: Any) -> np.ndarray:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
    from hipengine.runtime.qwen35_gguf_runner import DType

    if session.runner is None or session.scratch is None:
        raise ProbeError("session is closed")
    hidden = np.empty((1, int(session.runner.hidden_size)), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(hidden),
        DeviceBuffer(
            int(session.scratch.hidden_seed_fp32.ptr),
            int(session.runner.hidden_size) * DType.FP32.itemsize,
        ),
        hidden.nbytes,
        runtime=session.runtime,
    )
    return np.ascontiguousarray(hidden, dtype=np.float32)


def _probe_bulk_or_native(
    session: Any,
    verifier_inputs: list[int],
    *,
    mode: str,
    use_wmma_prefill: bool,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    result = session.verify_target_block(
        verifier_inputs,
        bulk_attention_mode=mode,
        use_wmma_prefill=bool(use_wmma_prefill),
        record_stage_timings=False,
    )
    return (
        [int(token) for token in result.token_ids],
        _copy_verify_logits(session, len(verifier_inputs)),
        np.ascontiguousarray(result.hidden_seeds, dtype=np.float32),
    )


def _probe_serial(session: Any, verifier_inputs: list[int]) -> tuple[list[int], np.ndarray, np.ndarray]:
    token_ids: list[int] = []
    logits_rows: list[np.ndarray] = []
    hidden_rows: list[np.ndarray] = []
    for token in verifier_inputs:
        row = session.step(
            int(token),
            return_logits=True,
            capture_hidden_seed_fp32=True,
        )
        token_ids.append(int(row.token_id))
        logits_rows.append(np.ascontiguousarray(row.logits.reshape(-1), dtype=np.float32))
        hidden_rows.append(_copy_current_hidden_seed(session).reshape(-1))
    return (
        token_ids,
        np.ascontiguousarray(np.stack(logits_rows, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.stack(hidden_rows, axis=0), dtype=np.float32),
    )


def _run_trace_block(
    session: Any,
    verifier_inputs: list[int],
    *,
    mode: str,
    use_wmma_prefill: bool,
    capture_linear_state_rows: bool,
    defer_linear_state_commit: bool,
) -> list[int]:
    if mode == "serial-exact":
        result = session.verify_target_block_serial_exact(
            verifier_inputs,
            capture_linear_state_rows=bool(capture_linear_state_rows),
        )
    else:
        result = session.verify_target_block(
            verifier_inputs,
            bulk_attention_mode=mode,
            use_wmma_prefill=bool(use_wmma_prefill),
            capture_linear_state_rows=bool(capture_linear_state_rows),
            defer_linear_state_commit=bool(defer_linear_state_commit),
            record_stage_timings=False,
        )
    if capture_linear_state_rows and not result.linear_state_rows_captured:
        raise ProbeError("trace replay requested linear-state rows, but verifier did not capture them")
    return [int(token) for token in result.token_ids]


def _replay_prior_cycles(
    session: Any,
    cycles: list[dict[str, Any]],
    *,
    current_prev: int,
    mode: str,
    use_wmma_prefill: bool,
) -> tuple[int, list[dict[str, Any]]]:
    replay_rows: list[dict[str, Any]] = []
    for row in cycles:
        cycle_number = int(row.get("cycle", len(replay_rows)))
        start_position = int(row["cycle_start_seq_position"])
        if int(session.position) != start_position:
            raise ProbeError(
                f"cycle {cycle_number} replay position {session.position} does not match trace {start_position}"
            )
        cycle_prev = int(row["cycle_prev_token"])
        if int(current_prev) != cycle_prev:
            raise ProbeError(
                f"cycle {cycle_number} replay prev {current_prev} does not match trace {cycle_prev}"
            )
        draft_tokens = [int(token) for token in row.get("draft_tokens", [])]
        trace_target_tokens = [int(token) for token in row.get("target_tokens", [])]
        trace_output_tokens = [int(token) for token in row.get("output_tokens", [])]
        accepted = int(row.get("accepted_draft_tokens", _llama_accept_count(draft_tokens, trace_target_tokens)))
        consumed_rows = accepted + 1
        verifier_inputs = [cycle_prev] + draft_tokens
        if consumed_rows <= 0 or consumed_rows > len(verifier_inputs):
            raise ProbeError(
                f"cycle {cycle_number} consumed rows {consumed_rows} outside verifier input length {len(verifier_inputs)}"
            )
        sampled_tokens = _run_trace_block(
            session,
            verifier_inputs,
            mode=mode,
            use_wmma_prefill=bool(use_wmma_prefill),
            capture_linear_state_rows=True,
            defer_linear_state_commit=(mode != "serial-exact"),
        )
        if sampled_tokens[: len(trace_target_tokens)] != trace_target_tokens:
            raise ProbeError(
                f"cycle {cycle_number} target replay mismatch: sampled {sampled_tokens}, trace {trace_target_tokens}"
            )
        if trace_output_tokens != sampled_tokens[:consumed_rows]:
            raise ProbeError(
                f"cycle {cycle_number} output replay mismatch: sampled {sampled_tokens[:consumed_rows]}, "
                f"trace {trace_output_tokens}"
            )
        session._commit_verify_linear_state_row(consumed_rows - 1, position=start_position + consumed_rows)
        current_prev = int(trace_output_tokens[-1])
        replay_rows.append(
            {
                "cycle": int(cycle_number),
                "start_position": int(start_position),
                "verifier_input_tokens": verifier_inputs,
                "sampled_tokens": sampled_tokens,
                "accepted_draft_tokens": int(accepted),
                "consumed_rows": int(consumed_rows),
                "committed_position": int(start_position + consumed_rows),
                "next_prev_token": int(current_prev),
            }
        )
    return int(current_prev), replay_rows


def exact_command_payload(argv: Sequence[object]) -> dict[str, Any]:
    argv_strings = [str(item) for item in argv]
    return {"argv": argv_strings, "command": shlex.join(argv_strings)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(GGUF_PATH))
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-token", action="append", type=_parse_token_csv, default=[])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--target-block-verify-mode", choices=("bulk", "native", "serial-exact"), default="bulk")
    parser.add_argument("--replay-target-block-verify-mode", choices=("bulk", "native", "serial-exact"), default=None)
    parser.add_argument("--target-block-wmma-prefill", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decode-repack", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-wmma-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gemv-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefill-attention-mode", choices=("bulk", "native"), default="bulk")
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if not _hip_available():
        raise ProbeError("ROCm/HIP is not available")
    if not args.model.exists():
        raise ProbeError(f"model not found: {args.model}")
    trace = _read_json(args.trace)
    workload = trace.get("workload")
    if not isinstance(workload, dict):
        raise ProbeError("trace is missing workload{}")
    cycles = trace.get("cycles")
    if not isinstance(cycles, list) or not all(isinstance(row, dict) for row in cycles):
        raise ProbeError("trace is missing cycles[]")
    cycle_index, cycle = _find_cycle(cycles, int(args.cycle))
    prompt = str(workload.get("prompt", ""))
    if not prompt:
        raise ProbeError("trace workload is missing prompt")
    prompt_reasoning = str(workload.get("prompt_reasoning", "off"))
    initial_prev_token = int(workload["initial_prev_token"])
    initial_prev_position = int(workload["initial_prev_position"])
    cycle_prev_token = int(cycle["cycle_prev_token"])
    cycle_start_position = int(cycle["cycle_start_seq_position"])
    draft_tokens = [int(token) for token in cycle.get("draft_tokens", [])]
    target_tokens = [int(token) for token in cycle.get("target_tokens", [])]
    output_tokens = [int(token) for token in cycle.get("output_tokens", [])]
    prefix_visible_outputs = _flatten_output_tokens(cycles[:cycle_index])
    if not prefix_visible_outputs:
        raise ProbeError("forced cycle must have at least one prior visible output token")
    if int(prefix_visible_outputs[-1]) != cycle_prev_token:
        raise ProbeError(
            f"prefix final output {prefix_visible_outputs[-1]} does not match cycle_prev_token {cycle_prev_token}"
        )
    consumed_prefix_tokens = [initial_prev_token] + prefix_visible_outputs[:-1]
    expected_start_position = initial_prev_position + len(consumed_prefix_tokens)
    if expected_start_position != cycle_start_position:
        raise ProbeError(
            f"derived start position {expected_start_position} does not match trace {cycle_start_position}"
        )
    verifier_inputs = [cycle_prev_token] + draft_tokens
    extra_candidates = [token for values in args.candidate_token for token in values]
    candidate_tokens = _unique_ints([*draft_tokens, *target_tokens, *output_tokens, *extra_candidates])
    route_env = _apply_trace_route_env(workload, decode_repack=args.decode_repack)
    if _env_truthy("HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A"):
        raise ProbeError("direct verifier top-1 path is enabled; full row logits would be unavailable")

    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8")
    gguf_info = scan_gguf(args.model)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(gguf_info)
    prompt_tokens = build_chat_prompt(tokenizer, prompt, reasoning=prompt_reasoning)
    if int(workload.get("prompt_tokens", len(prompt_tokens))) != len(prompt_tokens):
        raise ProbeError(
            f"rebuilt prompt has {len(prompt_tokens)} tokens; trace has {workload.get('prompt_tokens')}"
        )

    max_sequence_length = max(
        int(cycle_start_position) + len(verifier_inputs) + 4,
        len(prompt_tokens) + len(consumed_prefix_tokens) + len(verifier_inputs) + 4,
    )
    session = Qwen35GGUFResidentSession(
        args.model,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=bool(args.use_wmma_prefill),
        use_gemv_decode=bool(args.use_gemv_decode),
        prefill_config=PrefillConfig(),
    )
    try:
        prefill = session.prefill(
            prompt_tokens,
            use_bulk=True,
            bulk_attention_mode=str(args.prefill_attention_mode),
            return_logits=False,
            capture_hidden_seed_fp32=True,
        )
        current_prev = int(prefill.token_id)
        if current_prev != initial_prev_token:
            raise ProbeError(f"prefill token {current_prev} does not match trace initial_prev_token {initial_prev_token}")
        if int(session.position) != initial_prev_position:
            raise ProbeError(f"prefill position {session.position} does not match trace {initial_prev_position}")
        current_prev, replay_rows = _replay_prior_cycles(
            session,
            cycles[:cycle_index],
            current_prev=current_prev,
            mode=str(args.replay_target_block_verify_mode or args.target_block_verify_mode),
            use_wmma_prefill=bool(args.target_block_wmma_prefill),
        )
        if current_prev != cycle_prev_token:
            raise ProbeError(f"replayed prev token {current_prev} does not match cycle prev {cycle_prev_token}")
        if int(session.position) != cycle_start_position:
            raise ProbeError(f"replayed position {session.position} does not match cycle start {cycle_start_position}")
        cycle_pending_hidden_seed = _copy_current_hidden_seed(session)

        if args.target_block_verify_mode == "serial-exact":
            sampled_tokens, logits, hidden_seeds = _probe_serial(session, verifier_inputs)
        else:
            sampled_tokens, logits, hidden_seeds = _probe_bulk_or_native(
                session,
                verifier_inputs,
                mode=str(args.target_block_verify_mode),
                use_wmma_prefill=bool(args.target_block_wmma_prefill),
            )
        accepted_draft_tokens = _llama_accept_count(draft_tokens, sampled_tokens[: len(draft_tokens)])
        rows: list[dict[str, Any]] = []
        for row_index, (input_token, sampled_token) in enumerate(zip(verifier_inputs, sampled_tokens, strict=True)):
            row_logits = logits[row_index]
            rows.append(
                {
                    "row": int(row_index),
                    "position": int(cycle_start_position + row_index),
                    "input_token": int(input_token),
                    "sampled_token": int(sampled_token),
                    "trace_target_token": int(target_tokens[row_index]) if row_index < len(target_tokens) else None,
                    "trace_output_token": int(output_tokens[row_index]) if row_index < len(output_tokens) else None,
                    "draft_token_at_depth": int(draft_tokens[row_index]) if row_index < len(draft_tokens) else None,
                    "hidden_seed_summary": hidden_state_summary(
                        hidden_seeds[row_index],
                        label="target_verify_hidden_seed",
                        depth=int(row_index),
                        token_id=int(input_token),
                        position=int(cycle_start_position + row_index),
                    ),
                    "top_k": _top_k_rows(row_logits, top_k=int(args.top_k)),
                    "candidate_scores": _candidate_rows(row_logits, candidate_tokens),
                }
            )
    finally:
        session.close()

    artifact = {
        "schema": 1,
        "kind": "hipengine_gguf_mtp_forced_target_probe",
        "status": "complete",
        "performance_claim": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repo": _repo_provenance(),
        "model": str(args.model),
        "source_trace": str(args.trace),
        "command": exact_command_payload(sys.argv),
        "route_env": route_env,
        "probe": {
            "cycle": int(args.cycle),
            "cycle_index": int(cycle_index),
            "prompt": prompt,
            "prompt_reasoning": prompt_reasoning,
            "prompt_tokens": int(len(prompt_tokens)),
            "initial_prev_token": int(initial_prev_token),
            "initial_prev_position": int(initial_prev_position),
            "prefix_visible_output_tokens": prefix_visible_outputs,
            "consumed_prefix_tokens": consumed_prefix_tokens,
            "cycle_prev_token": int(cycle_prev_token),
            "cycle_start_seq_position": int(cycle_start_position),
            "verifier_input_tokens": verifier_inputs,
            "trace_draft_tokens": draft_tokens,
            "trace_target_tokens": target_tokens,
            "trace_output_tokens": output_tokens,
            "candidate_tokens": candidate_tokens,
        },
        "result": {
            "target_block_verify_mode": str(args.target_block_verify_mode),
            "replay_target_block_verify_mode": str(args.replay_target_block_verify_mode or args.target_block_verify_mode),
            "target_block_wmma_prefill": bool(args.target_block_wmma_prefill),
            "prior_cycle_replay": replay_rows,
            "cycle_pending_hidden_seed_summary": hidden_state_summary(
                cycle_pending_hidden_seed,
                label="cycle_pending_hidden_seed",
                depth=-1,
                token_id=int(cycle_prev_token),
                position=int(cycle_start_position),
            ),
            "sampled_tokens": sampled_tokens,
            "accepted_draft_tokens": int(accepted_draft_tokens),
            "rows": rows,
        },
        "notes": [
            "Diagnostic only: forced-prefix target verifier score capture, not a performance run.",
            "Prefix replay consumes initial_prev_token plus prior visible outputs except the final cycle_prev_token, matching scripts/gguf_mtp_bench.py block verifier state.",
            "The verifier direct Q6 top-1 path is forced off so full row logits are available.",
            "cycle_pending_hidden_seed_summary is the seed row that starts the MTP draft for this cycle.",
            "hidden_seed_summary is the FP32 post-output_norm verifier row used as the MTP draft seed.",
        ],
    }
    text = json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
