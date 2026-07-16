#!/usr/bin/env python3
"""Compare GGUF packed-AR token, Conv/GDN, and live-KV lifecycle to c1.

``independent_c1`` prefill isolates packed decode from packed prompt prefill.
``packed`` prefill exercises the complete public c>N state lifecycle.  This is
a byte-exact correctness diagnostic and never a performance benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


_CAPTURE_PREFILL_GDN_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"


@contextmanager
def _temporary_env(updates: dict[str, str]) -> Iterator[None]:
    prior = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _device_hash(session: Any, buffer: Any, *, nbytes: int | None = None) -> str:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    size = int(buffer.nbytes if nbytes is None else nbytes)
    raw = np.empty((size,), dtype=np.uint8)
    if size:
        copy_device_to_host(
            host_array_ptr(raw),
            DeviceBuffer(int(buffer.ptr), size),
            size,
            runtime=session.runtime,
        )
    return hashlib.blake2b(raw.tobytes(), digest_size=16).hexdigest()


def _capture_state(session: Any) -> dict[str, Any]:
    from hipengine.core.dtype import DType

    if session.runner is None or session.runner.weights is None or session.scratch is None:
        raise RuntimeError("GGUF resident session is closed")
    session.runtime.device_synchronize()
    scratch = session.scratch
    cfg = session.runner.weights.config
    linear: list[dict[str, Any]] = []
    for layer_id, (conv, recurrent) in enumerate(
        zip(scratch.layer_conv_states, scratch.layer_recurrent_states, strict=True)
    ):
        if conv is None or recurrent is None:
            continue
        linear.append(
            {
                "layer": int(layer_id),
                "conv": _device_hash(session, conv),
                "recurrent": _device_hash(session, recurrent),
            }
        )
    live_positions = int(session.position)
    kv_row_nbytes = (
        int(cfg.head_count_kv)
        * int(cfg.key_length)
        * DType.BF16.itemsize
    )
    live_nbytes = live_positions * kv_row_nbytes
    kv: list[dict[str, Any]] = []
    for layer_id, (key, value) in enumerate(
        zip(scratch.full_key_caches, scratch.full_value_caches, strict=True)
    ):
        if key is None or value is None:
            continue
        checked_nbytes = min(live_nbytes, int(key.nbytes), int(value.nbytes))
        kv.append(
            {
                "layer": int(layer_id),
                "key": _device_hash(session, key, nbytes=checked_nbytes),
                "value": _device_hash(session, value, nbytes=checked_nbytes),
                "checked_nbytes": checked_nbytes,
            }
        )
    return {
        "position": live_positions,
        "linear": linear,
        "kv": kv,
    }


def _component_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["layer"]): row for row in rows}


def _compare_state_rows(
    packed: Sequence[dict[str, Any]],
    references: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for row_index, (actual, expected) in enumerate(zip(packed, references, strict=True)):
        if int(actual["position"]) != int(expected["position"]):
            mismatches.append(
                {
                    "row": row_index,
                    "component": "position",
                    "layer": None,
                    "part": None,
                    "packed": int(actual["position"]),
                    "c1": int(expected["position"]),
                }
            )
        for component, parts in (("linear", ("conv", "recurrent")), ("kv", ("key", "value"))):
            actual_layers = _component_map(actual[component])
            expected_layers = _component_map(expected[component])
            for layer in sorted(set(actual_layers) | set(expected_layers)):
                actual_layer = actual_layers.get(layer, {})
                expected_layer = expected_layers.get(layer, {})
                for part in parts:
                    if actual_layer.get(part) != expected_layer.get(part):
                        mismatches.append(
                            {
                                "row": row_index,
                                "component": component,
                                "layer": layer,
                                "part": part,
                                "packed": actual_layer.get(part),
                                "c1": expected_layer.get(part),
                            }
                        )
    return mismatches


def _prefill_c1(session: Any, prompt: Sequence[int]) -> int:
    result = session.prefill(
        [int(token) for token in prompt],
        use_bulk=True,
        bulk_attention_mode="bulk",
        return_logits=False,
    )
    return int(result.token_id)


def _session_build_policy(args: argparse.Namespace) -> dict[str, Any]:
    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.expanduser().read_text(
            encoding="utf-8"
        ).strip()
        if not compiler_version:
            raise ValueError(
                f"compiler version file is empty: {args.compiler_version_file}"
            )
    return {
        "compiler_version": compiler_version,
        "require_cached_build": bool(args.require_cached_build),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = int(args.rows)
    if rows < 2 or rows > 4:
        raise ValueError("rows must be between 2 and the production packed-chunk cap (4)")
    lifecycle = str(args.lifecycle)
    if lifecycle == "shrink_sparse" and (rows != 4 or int(args.decode_steps) != 4):
        raise ValueError("shrink_sparse lifecycle requires --rows 4 --decode-steps 4")
    alternate_prompt_length = (
        int(args.prompt_length)
        if args.alternate_prompt_length is None
        else int(args.alternate_prompt_length)
    )
    if min(int(args.prompt_length), alternate_prompt_length) < 4:
        raise ValueError("prompt lengths must be at least the GDN convolution width (4)")
    if int(args.decode_steps) < 1:
        raise ValueError("decode-steps must be positive")
    if max(int(args.prompt_length), alternate_prompt_length) + int(args.decode_steps) >= 1024:
        raise ValueError("packed GGUF AR state oracle currently requires context < 1024")
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    prompts: tuple[list[int], ...] = tuple(
        [int(args.prompt_token_id)] * (
            int(args.prompt_length) if row_index == 0 else alternate_prompt_length
        )
        for row_index in range(rows)
    )
    for row_index in range(1, rows):
        prompts[row_index][-1] = int(args.alternate_token_id) + row_index - 1
    max_sequence_length = max(len(prompt) for prompt in prompts) + int(args.decode_steps) + 2
    build_policy = _session_build_policy(args)

    with ExitStack() as stack:
        owner = stack.enter_context(
            Qwen35GGUFResidentSession(
                model,
                backend=str(args.backend),
                max_sequence_length=max_sequence_length,
                **build_policy,
            )
        )
        shared_runner = owner.runner
        if shared_runner is None:
            raise RuntimeError("GGUF shared runner was not materialized")
        sessions = [owner]
        for _ in range(2 * rows - 1):
            sessions.append(
                stack.enter_context(
                    Qwen35GGUFResidentSession(
                        model,
                        backend=str(args.backend),
                        runtime=owner.runtime,
                        shared_runner=shared_runner,
                        max_sequence_length=max_sequence_length,
                        **build_policy,
                    )
                )
            )
        packed_sessions = tuple(sessions[:rows])
        reference_sessions = tuple(sessions[rows:])

        with _temporary_env({_CAPTURE_PREFILL_GDN_ENV: "1"}):
            if str(args.prefill_mode) == "packed":
                prefill_results = owner.prefill_batch_native(
                    prompts,
                    sessions=packed_sessions,
                    return_logits=False,
                )
                packed_tokens = [int(result.token_id) for result in prefill_results]
            else:
                packed_tokens = [
                    _prefill_c1(session, prompt)
                    for session, prompt in zip(packed_sessions, prompts, strict=True)
                ]
        reference_tokens = [
            _prefill_c1(session, prompt)
            for session, prompt in zip(reference_sessions, prompts, strict=True)
        ]
        initial_packed = [_capture_state(session) for session in packed_sessions]
        initial_reference = [_capture_state(session) for session in reference_sessions]
        initial_mismatches = _compare_state_rows(initial_packed, initial_reference)

        packed_trajectory = [list(packed_tokens)]
        reference_trajectory = [list(reference_tokens)]
        lifecycle_events: list[dict[str, Any]] = []
        dirty_before_flush = False
        flushed = False
        if lifecycle == "steady":
            for _step in range(int(args.decode_steps)):
                with _temporary_env({_CAPTURE_PREFILL_GDN_ENV: "1"}):
                    packed_tokens = [
                        int(result.token_id)
                        for result in owner.step_batch_native(
                            packed_tokens,
                            sessions=packed_sessions,
                            positions=[int(session.position) for session in packed_sessions],
                            return_logits=False,
                            scatter_state=False,
                        )
                    ]
                reference_tokens = [
                    int(session.step(token, return_logits=False).token_id)
                    for session, token in zip(reference_sessions, reference_tokens, strict=True)
                ]
                packed_trajectory.append(list(packed_tokens))
                reference_trajectory.append(list(reference_tokens))
            dirty_before_flush = bool(owner._packed_decode_state_dirty)
            flushed = bool(owner.flush_packed_decode_state())
        else:
            live_groups = ((0, 1, 2, 3), (0, 2, 3), (0, 3), (3,))
            for step_index, live_indices in enumerate(live_groups, start=1):
                group_sessions = tuple(packed_sessions[index] for index in live_indices)
                if len(live_indices) > 1:
                    group_owner = group_sessions[0]
                    with _temporary_env({_CAPTURE_PREFILL_GDN_ENV: "1"}):
                        results = group_owner.step_batch_native(
                            [packed_tokens[index] for index in live_indices],
                            sessions=group_sessions,
                            positions=[int(session.position) for session in group_sessions],
                            return_logits=False,
                            scatter_state=False,
                        )
                    for index, result in zip(live_indices, results, strict=True):
                        packed_tokens[index] = int(result.token_id)
                    dirty_before_flush = dirty_before_flush or bool(
                        group_owner._packed_decode_state_dirty
                    )
                    flushed = bool(group_owner.flush_packed_decode_state()) or flushed
                else:
                    index = live_indices[0]
                    packed_tokens[index] = int(
                        packed_sessions[index].step(
                            packed_tokens[index],
                            return_logits=False,
                        ).token_id
                    )
                for index in live_indices:
                    reference_tokens[index] = int(
                        reference_sessions[index].step(
                            reference_tokens[index],
                            return_logits=False,
                        ).token_id
                    )
                event_packed = [_capture_state(packed_sessions[index]) for index in live_indices]
                event_reference = [_capture_state(reference_sessions[index]) for index in live_indices]
                event_mismatches = _compare_state_rows(event_packed, event_reference)
                lifecycle_events.append(
                    {
                        "step": step_index,
                        "live_indices": list(live_indices),
                        "state_exact": not event_mismatches,
                        "mismatches": event_mismatches,
                    }
                )
                packed_trajectory.append(list(packed_tokens))
                reference_trajectory.append(list(reference_tokens))

        final_packed = [_capture_state(session) for session in packed_sessions]
        final_reference = [_capture_state(session) for session in reference_sessions]
        final_mismatches = _compare_state_rows(final_packed, final_reference)
        target_arch = str(shared_runner.target_arch)
        resolved_backend = str(shared_runner.backend)

    tokens_exact = packed_trajectory == reference_trajectory
    lifecycle_state_exact = all(bool(event["state_exact"]) for event in lifecycle_events)
    passed = bool(
        tokens_exact
        and not initial_mismatches
        and dirty_before_flush
        and flushed
        and lifecycle_state_exact
        and not final_mismatches
    )
    first_divergence = None
    if initial_mismatches:
        first_divergence = {"phase": "prefill", **initial_mismatches[0]}
    elif not tokens_exact:
        first_divergence = {"phase": "decode_tokens"}
    elif not lifecycle_state_exact:
        failed_event = next(event for event in lifecycle_events if not event["state_exact"])
        first_divergence = {
            "phase": "lifecycle",
            "step": int(failed_event["step"]),
            **failed_event["mismatches"][0],
        }
    elif final_mismatches:
        first_divergence = {"phase": "decode_state", **final_mismatches[0]}
    return {
        "schema": 1,
        "kind": "gguf_packed_ar_state_oracle",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "performance_claim": False,
        "model": str(model),
        "backend": resolved_backend,
        "target_arch": target_arch,
        "build": {
            "compiler_version_file": (
                str(args.compiler_version_file)
                if args.compiler_version_file is not None
                else None
            ),
            "compiler_version_supplied": build_policy["compiler_version"] is not None,
            "require_cached_build": build_policy["require_cached_build"],
        },
        "prefill_mode": str(args.prefill_mode),
        "lifecycle": lifecycle,
        "workload": {
            "rows": rows,
            "prompt_token_id": int(args.prompt_token_id),
            "alternate_token_id": int(args.alternate_token_id),
            "prompt_length": int(args.prompt_length),
            "alternate_prompt_length": alternate_prompt_length,
            "prompt_lengths": [len(prompt) for prompt in prompts],
            "prompt_terminal_token_ids": [int(prompt[-1]) for prompt in prompts],
            "decode_steps": int(args.decode_steps),
            "sampling": "greedy_top1",
            "kv_dtype": "bf16",
        },
        "packed_token_trajectory": packed_trajectory,
        "c1_token_trajectory": reference_trajectory,
        "tokens_exact": tokens_exact,
        "initial_state_exact": not initial_mismatches,
        "initial_mismatches": initial_mismatches,
        "dirty_before_flush": dirty_before_flush,
        "flush_executed": flushed,
        "lifecycle_state_exact": lifecycle_state_exact,
        "lifecycle_events": lifecycle_events,
        "final_state_exact": not final_mismatches,
        "final_mismatches": final_mismatches,
        "first_divergence": first_divergence,
        "notes": [
            "All Conv/GDN FP32 state bytes and all live BF16 K/V bytes are compared.",
            "independent_c1 prefill isolates packed decode; packed prefill covers the complete c>N lifecycle.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument(
        "--lifecycle",
        choices=("steady", "shrink_sparse"),
        default="steady",
    )
    parser.add_argument(
        "--prefill-mode",
        choices=("independent_c1", "packed"),
        default="independent_c1",
    )
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--alternate-token-id", type=int, default=9708)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--alternate-prompt-length", type=int)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        help="read precomputed hipcc --version text so profiled runs do not spawn hipcc",
    )
    parser.add_argument(
        "--require-cached-build",
        action="store_true",
        help="fail rather than invoke hipcc when any resident-session library is missing",
    )
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    command_args = list(sys.argv[1:] if argv is None else argv)
    payload["command"] = shlex.join(
        [sys.executable, "scripts/gguf_packed_ar_state_oracle.py", *command_args]
    )
    payload["environment"] = {
        key: os.environ.get(key)
        for key in (
            "HIPENGINE_HIP_ARCH",
            "HIPENGINE_GGUF_DECODE_REPACK",
            "HIPENGINE_GGUF_WMMA_PREFILL",
            "HIPENGINE_GGUF_GEMV_DECODE",
            "HIP_VISIBLE_DEVICES",
        )
    }
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
