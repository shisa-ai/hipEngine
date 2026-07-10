#!/usr/bin/env python3
"""Four-step GGUF eager token/hidden/linear-state/KV correctness oracle.

The production token lane checks the current bulk-prefill + eager-decode route
against llama.cpp on an exact raw-token prompt.  The state lane then
teacher-forces that external token trajectory through eager decode and compares
every checkpoint with a fresh token-serial prefix recomputation.  Hidden rows,
all Conv/GDN state, and every live full-attention K/V prefix must be byte exact.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance


ORACLE_KIND = "hipengine_gguf_eager_teacher_forced_oracle"
ORACLE_SCHEMA_VERSION = 1
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_LLAMA_COMPLETION = Path(
    "/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-completion"
)
DEFAULT_LLAMA_TOKENIZE = Path(
    "/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-tokenize"
)


class OracleError(RuntimeError):
    """Raised when the external or resident oracle contract is incomplete."""


def _parse_token_ids(text: str, *, label: str) -> list[int]:
    try:
        payload = ast.literal_eval(text.strip())
    except (SyntaxError, ValueError) as exc:
        raise OracleError(f"{label} did not emit a token-ID list") from exc
    if (
        not isinstance(payload, list)
        or not payload
        or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in payload)
    ):
        raise OracleError(f"{label} must be a non-empty list of non-negative token IDs")
    return [int(token) for token in payload]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_checked(
    command: Sequence[str],
    *,
    label: str,
    timeout: float = 300.0,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OracleError(f"{label} could not run: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-1000:]
        raise OracleError(f"{label} failed with exit {completed.returncode}: {detail}")
    return completed


def _validate_external_trajectory(
    *,
    expected_prompt_ids: Sequence[int],
    actual_prompt_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    decode_steps: int,
) -> dict[str, Any]:
    expected_prompt = [int(token) for token in expected_prompt_ids]
    actual_prompt = [int(token) for token in actual_prompt_ids]
    if actual_prompt != expected_prompt:
        raise OracleError("llama.cpp prompt token IDs differ from the requested raw-token prompt")
    required = int(decode_steps) + 1
    generated = [int(token) for token in generated_token_ids]
    if len(generated) < required:
        raise OracleError(
            f"llama.cpp produced {len(generated)} tokens; {required} are required for "
            f"{decode_steps} eager transitions"
        )
    return {
        "passed": True,
        "prompt_token_ids": actual_prompt,
        "required_generated_tokens": required,
        "generated_token_ids": generated[:required],
    }


def _llama_reference(
    *,
    model: Path,
    prompt_token_id: int,
    prompt_length: int,
    decode_steps: int,
    llama_completion: Path,
    llama_tokenize: Path,
    gpu_layers: int,
) -> dict[str, Any]:
    for label, path in (
        ("llama-completion", llama_completion),
        ("llama-tokenize", llama_tokenize),
    ):
        if not path.is_file():
            raise OracleError(f"{label} does not exist: {path}")

    from hipengine.loading import load_gguf_index
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(model))
    token_piece = tokenizer.decode([int(prompt_token_id)])
    if not token_piece:
        raise OracleError(f"prompt token {prompt_token_id} decodes to an empty string")
    prompt_text = token_piece * int(prompt_length)
    expected_prompt_ids = [int(prompt_token_id)] * int(prompt_length)
    if tokenizer.encode(prompt_text) != expected_prompt_ids:
        raise OracleError("internal tokenizer cannot round-trip the repeated raw-token prompt")

    tokenize_prompt_command = [
        str(llama_tokenize),
        "-m",
        str(model),
        "-p",
        prompt_text,
        "--ids",
        "--no-bos",
        "--log-disable",
    ]
    prompt_result = _run_checked(tokenize_prompt_command, label="llama.cpp prompt tokenizer")
    actual_prompt_ids = _parse_token_ids(
        prompt_result.stdout,
        label="llama.cpp prompt tokenizer",
    )

    required_generated = int(decode_steps) + 1
    completion_command = [
        str(llama_completion),
        "-m",
        str(model),
        "-ngl",
        str(int(gpu_layers)),
        "-c",
        str(max(256, int(prompt_length) + required_generated + 16)),
        "-n",
        str(required_generated),
        "-p",
        prompt_text,
        "--temp",
        "0",
        "--ignore-eos",
        "--no-warmup",
        "--no-display-prompt",
        "-no-cnv",
    ]
    completion = _run_checked(
        completion_command,
        label="llama.cpp greedy completion",
        timeout=600.0,
    )
    if not completion.stdout:
        raise OracleError("llama.cpp greedy completion emitted no visible token text")
    tokenize_output_command = [
        str(llama_tokenize),
        "-m",
        str(model),
        "-p",
        completion.stdout,
        "--ids",
        "--no-bos",
        "--log-disable",
    ]
    output_result = _run_checked(
        tokenize_output_command,
        label="llama.cpp output tokenizer",
    )
    generated_token_ids = _parse_token_ids(
        output_result.stdout,
        label="llama.cpp output tokenizer",
    )
    validated = _validate_external_trajectory(
        expected_prompt_ids=expected_prompt_ids,
        actual_prompt_ids=actual_prompt_ids,
        generated_token_ids=generated_token_ids,
        decode_steps=int(decode_steps),
    )
    return {
        **validated,
        "prompt_token_id": int(prompt_token_id),
        "prompt_length": int(prompt_length),
        "prompt_piece": token_piece,
        "prompt_text_sha256": _text_sha256(prompt_text),
        "generated_text_sha256": _text_sha256(completion.stdout),
        "generated_text": completion.stdout,
        "completion_stderr_tail": completion.stderr.strip().splitlines()[-12:],
        "commands": {
            "prompt_tokenize": tokenize_prompt_command,
            "completion": completion_command,
            "output_tokenize": tokenize_output_command,
        },
        "tools": {
            "llama_completion": {
                "path": str(llama_completion),
                "sha256": _file_sha256(llama_completion),
            },
            "llama_tokenize": {
                "path": str(llama_tokenize),
                "sha256": _file_sha256(llama_tokenize),
            },
        },
    }


def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    words = np.ascontiguousarray(raw, dtype=np.uint8).view("<u2").astype(np.uint32)
    return np.ascontiguousarray((words << np.uint32(16)).view(np.float32))


def _fingerprint_raw(raw: np.ndarray, *, dtype: str) -> dict[str, Any]:
    raw_u8 = np.ascontiguousarray(raw, dtype=np.uint8).reshape(-1)
    if dtype == "fp32":
        if raw_u8.size % np.dtype(np.float32).itemsize:
            raise OracleError("FP32 state buffer is not element-aligned")
        values = raw_u8.view("<f4")
    elif dtype == "bf16":
        if raw_u8.size % np.dtype(np.uint16).itemsize:
            raise OracleError("BF16 state buffer is not element-aligned")
        values = _bf16_to_f32(raw_u8)
    else:
        raise OracleError(f"unsupported state dtype: {dtype}")
    finite = bool(np.all(np.isfinite(values)))
    values64 = values.astype(np.float64, copy=False)
    rms = float(math.sqrt(float(np.mean(values64 * values64)))) if values.size else 0.0
    max_abs = float(np.max(np.abs(values64))) if values.size else 0.0
    return {
        "nbytes": int(raw_u8.size),
        "blake2b_128": hashlib.blake2b(raw_u8.tobytes(), digest_size=16).hexdigest(),
        "finite": finite,
        "rms": rms,
        "max_abs": max_abs,
    }


def _fingerprint_array(array: np.ndarray) -> dict[str, Any]:
    values = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return _fingerprint_raw(values.view(np.uint8), dtype="fp32")


def _copy_device_fingerprint(
    session: Any,
    *,
    ptr: int,
    nbytes: int,
    dtype: str,
) -> dict[str, Any]:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    size = int(nbytes)
    raw = np.empty((size,), dtype=np.uint8)
    if size:
        copy_device_to_host(
            host_array_ptr(raw),
            DeviceBuffer(int(ptr), size),
            size,
            runtime=session.runtime,
        )
    return _fingerprint_raw(raw, dtype=dtype)


def _capture_checkpoint(
    session: Any,
    *,
    current_token_id: int,
    predicted_token_id: int,
) -> dict[str, Any]:
    from hipengine.core.dtype import DType

    if session.runner is None or session.runner.weights is None or session.scratch is None:
        raise OracleError("GGUF resident session is closed")
    if not session.fp32_hidden_seed_contract().ready_for_mtp:
        raise OracleError("eager checkpoint did not populate the FP32 hidden seed")
    session.runtime.device_synchronize()
    runner = session.runner
    scratch = session.scratch
    hidden_nbytes = int(runner.hidden_size) * DType.FP32.itemsize
    hidden_seed = _copy_device_fingerprint(
        session,
        ptr=int(scratch.hidden_seed_fp32.ptr),
        nbytes=hidden_nbytes,
        dtype="fp32",
    )
    captured_layers = session.last_layer_output_hidden
    expected_layer_count = len(runner.weights.config.layer_types)
    if sorted(captured_layers) != list(range(expected_layer_count)):
        raise OracleError("eager checkpoint did not capture every layer output")
    layer_outputs = [
        {
            "layer": int(layer_id),
            "fingerprint": _fingerprint_array(captured_layers[layer_id]),
        }
        for layer_id in sorted(captured_layers)
    ]

    linear_states: list[dict[str, Any]] = []
    for layer_id, (conv_state, recurrent_state) in enumerate(
        zip(scratch.layer_conv_states, scratch.layer_recurrent_states, strict=True)
    ):
        if conv_state is None or recurrent_state is None:
            continue
        linear_states.append(
            {
                "layer": int(layer_id),
                "conv": _copy_device_fingerprint(
                    session,
                    ptr=int(conv_state.ptr),
                    nbytes=int(conv_state.nbytes),
                    dtype="fp32",
                ),
                "recurrent": _copy_device_fingerprint(
                    session,
                    ptr=int(recurrent_state.ptr),
                    nbytes=int(recurrent_state.nbytes),
                    dtype="fp32",
                ),
            }
        )

    cfg = runner.weights.config
    live_positions = int(session.position)
    kv_row_nbytes = int(cfg.head_count_kv) * int(cfg.key_length) * DType.BF16.itemsize
    live_nbytes = live_positions * kv_row_nbytes
    kv_states: list[dict[str, Any]] = []
    for layer_id, (key_cache, value_cache) in enumerate(
        zip(scratch.full_key_caches, scratch.full_value_caches, strict=True)
    ):
        if key_cache is None or value_cache is None:
            continue
        checked_nbytes = min(live_nbytes, int(key_cache.nbytes), int(value_cache.nbytes))
        kv_states.append(
            {
                "layer": int(layer_id),
                "live_positions": live_positions,
                "key": _copy_device_fingerprint(
                    session,
                    ptr=int(key_cache.ptr),
                    nbytes=checked_nbytes,
                    dtype="bf16",
                ),
                "value": _copy_device_fingerprint(
                    session,
                    ptr=int(value_cache.ptr),
                    nbytes=checked_nbytes,
                    dtype="bf16",
                ),
            }
        )
    fingerprints = [
        hidden_seed,
        *(row["fingerprint"] for row in layer_outputs),
        *(row[part] for row in linear_states for part in ("conv", "recurrent")),
        *(row[part] for row in kv_states for part in ("key", "value")),
    ]
    return {
        "position": live_positions,
        "current_token_id": int(current_token_id),
        "predicted_token_id": int(predicted_token_id),
        "finite": all(bool(row["finite"]) for row in fingerprints),
        "hidden_seed": hidden_seed,
        "layer_outputs": layer_outputs,
        "linear_states": linear_states,
        "kv_states": kv_states,
    }


def _layer_map(rows: Any, *, label: str) -> dict[int, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise OracleError(f"{label} must be a list")
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or isinstance(row.get("layer"), bool):
            raise OracleError(f"{label} contains an invalid layer row")
        layer = int(row["layer"])
        if layer in result:
            raise OracleError(f"{label} contains duplicate layer {layer}")
        result[layer] = row
    return result


def _same_fingerprint(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("nbytes") == right.get("nbytes")
        and left.get("blake2b_128") == right.get("blake2b_128")
    )


def _compare_checkpoint(
    eager: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    expected_predicted_token_id: int,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []

    def add(
        component: str,
        *,
        layer: int | None,
        part: str | None,
        eager_value: Any,
        reference_value: Any,
    ) -> None:
        mismatches.append(
            {
                "component": component,
                "layer": layer,
                "part": part,
                "eager": eager_value,
                "reference": reference_value,
            }
        )

    for key, component in (
        ("position", "position"),
        ("current_token_id", "current_token"),
        ("predicted_token_id", "predicted_token"),
    ):
        if eager.get(key) != reference.get(key):
            add(
                component,
                layer=None,
                part=None,
                eager_value=eager.get(key),
                reference_value=reference.get(key),
            )
    if eager.get("finite", True) is not True or reference.get("finite", True) is not True:
        add(
            "nonfinite",
            layer=None,
            part=None,
            eager_value=eager.get("finite"),
            reference_value=reference.get("finite"),
        )
    if not _same_fingerprint(
        eager["hidden_seed"],
        reference["hidden_seed"],
    ):
        add(
            "hidden_seed",
            layer=None,
            part=None,
            eager_value=eager["hidden_seed"].get("blake2b_128"),
            reference_value=reference["hidden_seed"].get("blake2b_128"),
        )

    eager_layers = _layer_map(eager.get("layer_outputs"), label="eager.layer_outputs")
    reference_layers = _layer_map(
        reference.get("layer_outputs"),
        label="reference.layer_outputs",
    )
    for layer in sorted(set(eager_layers) | set(reference_layers)):
        eager_row = eager_layers.get(layer)
        reference_row = reference_layers.get(layer)
        if eager_row is None or reference_row is None or not _same_fingerprint(
            eager_row.get("fingerprint", {}),
            reference_row.get("fingerprint", {}),
        ):
            add(
                "layer_output",
                layer=layer,
                part=None,
                eager_value=None if eager_row is None else eager_row.get("fingerprint", {}).get("blake2b_128"),
                reference_value=(
                    None
                    if reference_row is None
                    else reference_row.get("fingerprint", {}).get("blake2b_128")
                ),
            )

    eager_linear = _layer_map(eager.get("linear_states"), label="eager.linear_states")
    reference_linear = _layer_map(
        reference.get("linear_states"),
        label="reference.linear_states",
    )
    for layer in sorted(set(eager_linear) | set(reference_linear)):
        eager_row = eager_linear.get(layer)
        reference_row = reference_linear.get(layer)
        for part in ("conv", "recurrent"):
            if eager_row is None or reference_row is None or not _same_fingerprint(
                eager_row.get(part, {}),
                reference_row.get(part, {}),
            ):
                add(
                    "linear_state",
                    layer=layer,
                    part=part,
                    eager_value=None if eager_row is None else eager_row.get(part, {}).get("blake2b_128"),
                    reference_value=(
                        None
                        if reference_row is None
                        else reference_row.get(part, {}).get("blake2b_128")
                    ),
                )

    eager_kv = _layer_map(eager.get("kv_states"), label="eager.kv_states")
    reference_kv = _layer_map(reference.get("kv_states"), label="reference.kv_states")
    for layer in sorted(set(eager_kv) | set(reference_kv)):
        eager_row = eager_kv.get(layer)
        reference_row = reference_kv.get(layer)
        if (
            eager_row is not None
            and reference_row is not None
            and eager_row.get("live_positions") != reference_row.get("live_positions")
        ):
            add(
                "full_attention_kv",
                layer=layer,
                part="live_positions",
                eager_value=eager_row.get("live_positions"),
                reference_value=reference_row.get("live_positions"),
            )
        for part in ("key", "value"):
            if eager_row is None or reference_row is None or not _same_fingerprint(
                eager_row.get(part, {}),
                reference_row.get(part, {}),
            ):
                add(
                    "full_attention_kv",
                    layer=layer,
                    part=part,
                    eager_value=None if eager_row is None else eager_row.get(part, {}).get("blake2b_128"),
                    reference_value=(
                        None
                        if reference_row is None
                        else reference_row.get(part, {}).get("blake2b_128")
                    ),
                )

    component_order = {
        "position": 0,
        "current_token": 1,
        "predicted_token": 2,
        "nonfinite": 3,
        "hidden_seed": 4,
        "full_attention_kv": 5,
        "layer_output": 6,
        "linear_state": 7,
    }
    mismatches.sort(
        key=lambda row: (
            -1 if row["layer"] is None else int(row["layer"]),
            component_order.get(str(row["component"]), 99),
            "" if row["part"] is None else str(row["part"]),
        )
    )
    token_match_external = (
        int(eager["predicted_token_id"]) == int(expected_predicted_token_id)
        and int(reference["predicted_token_id"]) == int(expected_predicted_token_id)
    )
    state_exact = not mismatches
    if mismatches:
        first = {
            "component": mismatches[0]["component"],
            "layer": mismatches[0]["layer"],
            "part": mismatches[0]["part"],
        }
    elif not token_match_external:
        first = {"component": "external_token", "layer": None, "part": None}
    else:
        first = None
    return {
        "passed": bool(state_exact and token_match_external),
        "token_match_external": token_match_external,
        "state_exact": state_exact,
        "mismatches": mismatches,
        "first_divergence": first,
    }


def _production_token_trajectory(
    session: Any,
    *,
    prompt_ids: Sequence[int],
    external_tokens: Sequence[int],
    decode_steps: int,
) -> list[int]:
    first = session.prefill(
        [int(token) for token in prompt_ids],
        use_bulk=True,
        bulk_attention_mode="bulk",
        return_logits=False,
    )
    predicted = [int(first.token_id)]
    for step in range(int(decode_steps)):
        result = session.step(int(external_tokens[step]), return_logits=False)
        predicted.append(int(result.token_id))
    return predicted


def _state_oracle(
    session: Any,
    *,
    prompt_ids: Sequence[int],
    external_tokens: Sequence[int],
    decode_steps: int,
) -> dict[str, Any]:
    if session.runner is None or session.runner.weights is None:
        raise OracleError("GGUF resident session is closed")
    layer_ids = list(range(len(session.runner.weights.config.layer_types)))

    initial = session.prefill(
        [int(token) for token in prompt_ids],
        use_bulk=False,
        return_logits=False,
        capture_hidden_seed_fp32=True,
    )
    eager_initial_token = int(initial.token_id)
    eager_checkpoints: list[dict[str, Any]] = []
    for step in range(int(decode_steps)):
        current_token = int(external_tokens[step])
        result = session.step(
            current_token,
            return_logits=False,
            capture_hidden_seed_fp32=True,
            capture_layer_output_hidden=layer_ids,
        )
        eager_checkpoints.append(
            _capture_checkpoint(
                session,
                current_token_id=current_token,
                predicted_token_id=int(result.token_id),
            )
        )

    rows: list[dict[str, Any]] = []
    for step in range(int(decode_steps)):
        prefix_before_current = [
            *[int(token) for token in prompt_ids],
            *[int(token) for token in external_tokens[:step]],
        ]
        prefix_result = session.prefill(
            prefix_before_current,
            use_bulk=False,
            return_logits=False,
            capture_hidden_seed_fp32=True,
        )
        current_token = int(external_tokens[step])
        result = session.step(
            current_token,
            return_logits=False,
            capture_hidden_seed_fp32=True,
            capture_layer_output_hidden=layer_ids,
        )
        reference = _capture_checkpoint(
            session,
            current_token_id=current_token,
            predicted_token_id=int(result.token_id),
        )
        comparison = _compare_checkpoint(
            eager_checkpoints[step],
            reference,
            expected_predicted_token_id=int(external_tokens[step + 1]),
        )
        prefix_token_match_external = int(prefix_result.token_id) == current_token
        if not prefix_token_match_external:
            comparison["passed"] = False
            comparison["token_match_external"] = False
            if comparison["first_divergence"] is None:
                comparison["first_divergence"] = {
                    "component": "fresh_prefix_token",
                    "layer": None,
                    "part": None,
                }
        rows.append(
            {
                "step": step + 1,
                "prefix_length_before_step": len(prefix_before_current),
                "prefix_predicted_token_id": int(prefix_result.token_id),
                "prefix_token_match_external": prefix_token_match_external,
                "input_token_id": current_token,
                "expected_predicted_token_id": int(external_tokens[step + 1]),
                "eager": eager_checkpoints[step],
                "fresh_prefix_reference": reference,
                "comparison": comparison,
            }
        )
    return {
        "initial_predicted_token_id": eager_initial_token,
        "initial_token_match_external": eager_initial_token == int(external_tokens[0]),
        "decode_steps": int(decode_steps),
        "all_steps_passed": all(bool(row["comparison"]["passed"]) for row in rows),
        "steps": rows,
    }


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if int(args.decode_steps) < 4:
        raise OracleError("SOL-G1 requires at least four eager decode steps")
    if int(args.prompt_length) < 1:
        raise OracleError("prompt_length must be positive")
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise OracleError(f"model does not exist: {model}")
    prompt_ids = [int(args.prompt_token_id)] * int(args.prompt_length)
    external = _llama_reference(
        model=model,
        prompt_token_id=int(args.prompt_token_id),
        prompt_length=int(args.prompt_length),
        decode_steps=int(args.decode_steps),
        llama_completion=args.llama_completion.expanduser().resolve(),
        llama_tokenize=args.llama_tokenize.expanduser().resolve(),
        gpu_layers=int(args.llama_gpu_layers),
    )
    external_tokens = [int(token) for token in external["generated_token_ids"]]
    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8")

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    with Qwen35GGUFResidentSession(
        model,
        backend=str(args.backend),
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=int(args.prompt_length) + int(args.decode_steps) + 2,
    ) as session:
        production_tokens = _production_token_trajectory(
            session,
            prompt_ids=prompt_ids,
            external_tokens=external_tokens,
            decode_steps=int(args.decode_steps),
        )
        session.reset()
        state_oracle = _state_oracle(
            session,
            prompt_ids=prompt_ids,
            external_tokens=external_tokens,
            decode_steps=int(args.decode_steps),
        )
        if session.runner is None:
            raise OracleError("GGUF resident session closed before provenance capture")
        resolved_backend = str(session.runner.backend)
        target_arch = str(session.runner.target_arch)
        fastpath_safety = None if session.fastpath_safety is None else session.fastpath_safety.as_dict()

    production_match = production_tokens == external_tokens
    repeated_stream = all(token == int(args.prompt_token_id) for token in external_tokens)
    passed = bool(
        external["passed"]
        and production_match
        and state_oracle["initial_token_match_external"]
        and state_oracle["all_steps_passed"]
    )
    first_divergence = next(
        (
            row["comparison"]["first_divergence"]
            for row in state_oracle["steps"]
            if row["comparison"]["first_divergence"] is not None
        ),
        None,
    )
    if passed and repeated_stream:
        status = "passed_repeated_stream_valid"
    elif passed:
        status = "passed_nonrepeating_stream_valid"
    else:
        status = "failed_eager_oracle"
    return {
        "kind": ORACLE_KIND,
        "schema_version": ORACLE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "correctness_claim": True,
        "workload": {
            "model": str(model),
            "quant": "gguf_q4_k_m",
            "kv_dtype": "bf16",
            "prompt_source": "repeated_token_id",
            "prompt_token_id": int(args.prompt_token_id),
            "prompt_length": int(args.prompt_length),
            "decode_steps": int(args.decode_steps),
            "sampling": {"temperature": 0.0, "ignore_eos": True},
        },
        "external_oracle": external,
        "production_bulk_eager": {
            "predicted_token_ids": production_tokens,
            "exact_external_match": production_match,
            "fastpath_safety": fastpath_safety,
        },
        "teacher_forced_state_oracle": state_oracle,
        "classification": {
            "passed": passed,
            "status": status,
            "repeated_stream": repeated_stream,
            "first_divergence": first_divergence,
            "conclusion": (
                "The repeated 9707 stream is shared by llama.cpp and hipEngine; "
                "four eager transitions are byte-exact against fresh serial-prefix "
                "hidden, Conv/GDN, and live KV state."
                if passed and repeated_stream
                else "The eager trajectory or state oracle diverged; inspect first_divergence."
            ),
        },
        "provenance": collect_artifact_provenance(
            repo_root=REPO_ROOT,
            configured_backend=str(args.backend),
            resolved_backend=resolved_backend,
            target_arch=target_arch,
            model_path=model,
            quant="gguf_q4_k_m",
            kv_dtype="bf16",
            command=command,
            environment={
                "HIPENGINE_BACKEND": os.environ.get("HIPENGINE_BACKEND"),
                "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
                "HIPENGINE_GGUF_WMMA_PREFILL": os.environ.get("HIPENGINE_GGUF_WMMA_PREFILL"),
                "HIPENGINE_GGUF_GEMV_DECODE": os.environ.get("HIPENGINE_GGUF_GEMV_DECODE"),
                "HIPENGINE_GGUF_DECODE_REPACK": os.environ.get("HIPENGINE_GGUF_DECODE_REPACK"),
            },
            build_profile="gguf_eager_teacher_forced_oracle",
            timing_protocol="correctness_only_no_timing_claim",
            warmups=0,
            repetitions=1,
            profiler={"enabled": False, "kind": None, "command": None},
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--llama-completion", type=Path, default=DEFAULT_LLAMA_COMPLETION)
    parser.add_argument("--llama-tokenize", type=Path, default=DEFAULT_LLAMA_TOKENIZE)
    parser.add_argument("--llama-gpu-layers", type=int, default=999)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    command = [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *raw_argv]
    try:
        artifact = run(args, command=command)
    except (OracleError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.json)
    print(json.dumps(artifact["classification"], indent=2, sort_keys=True))
    return 0 if artifact["classification"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
