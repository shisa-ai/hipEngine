#!/usr/bin/env python3
"""Gate Qwen3.8 direct row-batched INT8 KV decode against independent c1.

This is a correctness/ownership harness, not a performance benchmark. Prompts
are prefetched independently through the exact scalar path so IKV-C2 decode can
be qualified without claiming IKV-C3 shared-prefill ownership.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.dtype import DType  # noqa: E402
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr  # noqa: E402
from hipengine.kernels.backends import hip_target_arch_for_backend  # noqa: E402
from hipengine.kvcache import FixedPagedKVPolicy  # noqa: E402
from hipengine.loading.gguf import scan_gguf  # noqa: E402
from hipengine.models.kv_capabilities import KVCapabilityKey, model_artifact_identity  # noqa: E402
from hipengine.models.qwen35 import Qwen35GGUFModel  # noqa: E402
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession  # noqa: E402
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer  # noqa: E402
from scripts.gguf_mtp_bench import build_chat_prompt  # noqa: E402

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
DEFAULT_LENGTHS = (255, 256, 257, 512)
_REQUIRED_CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")
_CAPTURE_PREFILL_GDN_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"
_GDN_PREFILL_MODE_ENV = "HIPENGINE_GGUF_GDN_PREFILL_MODE"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return _sha256_bytes(contiguous.view(np.uint8).tobytes())


def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel) * int(tensor.dtype.itemsize)


def _device_hash(session: Qwen35GGUFResidentSession, ptr: int, nbytes: int) -> str:
    size = int(nbytes)
    raw = np.empty((size,), dtype=np.uint8)
    if size:
        copy_device_to_host(
            host_array_ptr(raw),
            DeviceBuffer(int(ptr), size),
            size,
            runtime=session.runtime,
        )
    return _sha256_bytes(raw.tobytes())


def _capture_state(session: Qwen35GGUFResidentSession) -> dict[str, Any]:
    if session.runner is None or session.runner.weights is None or session.scratch is None:
        raise RuntimeError("resident session is closed")
    runtime = session.runtime
    assert runtime is not None
    runtime.device_synchronize()
    scratch = session.scratch
    linear = []
    for layer_id, (conv, recurrent) in enumerate(
        zip(scratch.layer_conv_states, scratch.layer_recurrent_states, strict=True)
    ):
        if conv is None or recurrent is None:
            continue
        linear.append(
            {
                "layer": layer_id,
                "conv": _device_hash(session, conv.ptr, conv.nbytes),
                "recurrent": _device_hash(session, recurrent.ptr, recurrent.nbytes),
            }
        )
    kv = []
    for layer_id, (key, value) in enumerate(
        zip(scratch.full_key_caches, scratch.full_value_caches, strict=True)
    ):
        if key is None or value is None:
            continue
        metadata = scratch.full_scale_metadata(layer_id)
        row: dict[str, Any] = {
            "layer": layer_id,
            "key_payload": _device_hash(session, key.ptr, key.nbytes),
            "value_payload": _device_hash(session, value.ptr, value.nbytes),
        }
        if metadata is not None:
            row["key_scale"] = _device_hash(
                session,
                metadata.k_scale.ptr,
                _tensor_nbytes(metadata.k_scale),
            )
            row["value_scale"] = _device_hash(
                session,
                metadata.v_scale.ptr,
                _tensor_nbytes(metadata.v_scale),
            )
        kv.append(row)
    return {
        "position": int(session.position),
        "linear": linear,
        "kv": kv,
    }


def _state_mismatches(
    actual: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for row, (got, want) in enumerate(zip(actual, expected, strict=True)):
        if got != want:
            mismatches.append(
                {
                    "row": row,
                    "actual_sha256": _sha256_bytes(
                        json.dumps(got, sort_keys=True).encode("utf-8")
                    ),
                    "expected_sha256": _sha256_bytes(
                        json.dumps(want, sort_keys=True).encode("utf-8")
                    ),
                }
            )
    return mismatches


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _logit_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    got = np.asarray(actual, dtype=np.float32)
    want = np.asarray(expected, dtype=np.float32)
    if got.shape != want.shape:
        return {
            "shape_match": False,
            "max_abs": None,
            "kl": None,
            "top1_match": False,
        }
    p = _softmax(want)
    q = _softmax(got)
    kl = float(np.sum(p * (np.log(np.maximum(p, 1e-30)) - np.log(np.maximum(q, 1e-30)))))
    return {
        "shape_match": True,
        "max_abs": float(np.max(np.abs(got - want))),
        "kl": kl,
        "top1_match": int(np.argmax(got)) == int(np.argmax(want)),
    }


def _load_prompt_rows(path: Path) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        category = str(row.get("category"))
        if category not in _REQUIRED_CATEGORIES or category in selected:
            continue
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"prompt {row.get('id')!r} has no messages")
        content = str(messages[-1].get("content") or "")
        if not content:
            raise ValueError(f"prompt {row.get('id')!r} has empty content")
        selected[category] = {
            "id": str(row.get("id")),
            "category": category,
            "content": content,
        }
    missing = [category for category in _REQUIRED_CATEGORIES if category not in selected]
    if missing:
        raise ValueError(f"prompt file is missing categories: {missing}")
    return [selected[category] for category in _REQUIRED_CATEGORIES]


def _build_prompts(
    tokenizer: Qwen35GGUFTokenizer,
    rows: Sequence[Mapping[str, Any]],
    lengths: Sequence[int],
) -> tuple[tuple[tuple[int, ...], ...], list[dict[str, Any]]]:
    if len(rows) != len(lengths):
        raise ValueError("prompt rows and lengths must align")
    prompts = []
    manifest = []
    for row, length in zip(rows, lengths, strict=True):
        target = int(length)
        if target <= 0:
            raise ValueError("prompt lengths must be positive")
        expanded = "\n".join([str(row["content"])] * 128)
        tokens = tuple(int(token) for token in build_chat_prompt(tokenizer, expanded))
        if len(tokens) < target:
            raise ValueError(f"expanded prompt {row['id']!r} has only {len(tokens)} tokens")
        selected = tokens[:target]
        prompts.append(selected)
        manifest.append(
            {
                "id": row["id"],
                "category": row["category"],
                "source_text_sha256": _sha256_bytes(str(row["content"]).encode("utf-8")),
                "tokens": target,
                "token_ids_sha256": _sha256_bytes(
                    np.asarray(selected, dtype=np.int64).tobytes()
                ),
            }
        )
    return tuple(prompts), manifest


@contextmanager
def _temporary_env(values: Mapping[str, str]) -> Iterator[None]:
    prior = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _quant_key(info: Any) -> str:
    name = str(getattr(info, "file_type_name", "") or "").strip().lower()
    if name.startswith("mostly_"):
        name = name[len("mostly_") :]
    if not name:
        raise ValueError("GGUF metadata does not expose file_type_name")
    return f"gguf_{name}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    model = args.model.expanduser().resolve()
    prompts_path = args.prompts.expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    if not prompts_path.is_file():
        raise ValueError(f"prompt file does not exist: {prompts_path}")
    lengths = tuple(int(value) for value in args.prompt_lengths.split(",") if value)
    if len(lengths) not in {2, 4}:
        raise ValueError("--prompt-lengths must contain c2 or c4 values")
    rows = len(lengths)
    if args.decode_steps <= 0:
        raise ValueError("--decode-steps must be positive")
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8").strip()
    if not compiler_version:
        raise ValueError("compiler version file is empty")

    info = scan_gguf(model)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(info)
    prompt_rows = _load_prompt_rows(prompts_path)[:rows]
    prompts, prompt_manifest = _build_prompts(tokenizer, prompt_rows, lengths)
    identity = model_artifact_identity(model)
    if not identity.content_verified:
        raise ValueError(f"model identity unavailable: {identity.error}")
    key = KVCapabilityKey(
        artifact_sha256=identity.sha256,
        artifact_size_bytes=identity.size_bytes,
        backend=str(args.backend),
        target_arch=hip_target_arch_for_backend(str(args.backend)),
        weight_quant=_quant_key(info),
        kv_storage="int8_per_token_head",
        storage_layout="uniform",
        scale_dtype="fp32",
        scale_granularity="per_token_head",
    )
    resolution = Qwen35GGUFModel().resolve_kv_capability(key=key, artifact=identity)
    if not resolution.promotion_eligible:
        raise ValueError(f"artifact is not qualified for INT8 KV: {resolution.reason}")
    capability = copy.deepcopy(resolution.as_dict())
    evidence = capability.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("qualified capability has no evidence payload")
    admitted_rows = int(evidence.get("max_direct_rows", 0))
    diagnostic_override = admitted_rows < rows
    if diagnostic_override:
        if int(args.diagnostic_direct_rows) < rows:
            raise ValueError(
                f"artifact admits c{admitted_rows}; pass --diagnostic-direct-rows {rows} for a pre-promotion gate"
            )
        evidence["max_direct_rows"] = rows
        evidence["max_serial_resident_rows"] = max(
            rows,
            int(evidence.get("max_serial_resident_rows", 0)),
        )

    max_sequence_length = max(lengths) + int(args.decode_steps) + 4
    session_kwargs = {
        "backend": str(args.backend),
        "max_sequence_length": max_sequence_length,
        "max_batch_size": rows,
        "kv_scale_dtype": DType.FP32,
        "kv_scale_granularity": "per_token_head",
        "compiler_version": compiler_version,
        "require_cached_build": bool(args.require_cached_build),
    }

    def policy() -> FixedPagedKVPolicy:
        return FixedPagedKVPolicy(
            block_size=256,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            scale_granularity="per_token_head",
        )

    with ExitStack() as stack:
        owner = stack.enter_context(
            Qwen35GGUFResidentSession(
                model,
                kv_policy=policy(),
                kv_capability=copy.deepcopy(capability),
                **session_kwargs,
            )
        )
        sessions = [owner]
        for _ in range(rows - 1):
            sessions.append(
                stack.enter_context(
                    Qwen35GGUFResidentSession(
                        model,
                        runtime=owner.runtime,
                        shared_runner=owner.runner,
                        kv_policy=policy(),
                        kv_capability=copy.deepcopy(capability),
                        **session_kwargs,
                    )
                )
            )
        if owner.runner is None or owner.runner.weights is None:
            raise RuntimeError("resident owner failed to load")
        layer_ids = tuple(range(len(owner.runner.weights.config.layer_types)))
        limits = [int(session.packed_decode_max_rows) for session in sessions]
        kernels = [callable(session._retained_decode_kernel) for session in sessions]
        if min(limits) < rows or not all(kernels):
            raise RuntimeError(f"direct batch route did not resolve: limits={limits}, kernels={kernels}")

        reference_tokens: list[list[int]] = []
        reference_logits: list[list[np.ndarray]] = []
        reference_hidden: list[list[dict[int, str]]] = []
        with _temporary_env(
            {
                _CAPTURE_PREFILL_GDN_ENV: "1",
                _GDN_PREFILL_MODE_ENV: "exact",
            }
        ):
            for session, prompt in zip(sessions, prompts, strict=True):
                first = session.prefill(prompt, return_logits=True)
                current = int(first.token_id)
                tokens = [current]
                logits = [np.asarray(first.logits, dtype=np.float32).copy()]
                hidden_steps: list[dict[int, str]] = []
                for _ in range(int(args.decode_steps)):
                    result = session.step(
                        current,
                        return_logits=True,
                        capture_layer_output_hidden=layer_ids,
                    )
                    current = int(result.token_id)
                    tokens.append(current)
                    logits.append(np.asarray(result.logits, dtype=np.float32).copy())
                    hidden_steps.append(
                        {
                            int(layer): _hash_array(array)
                            for layer, array in session.last_layer_output_hidden.items()
                        }
                    )
                reference_tokens.append(tokens)
                reference_logits.append(logits)
                reference_hidden.append(hidden_steps)
            reference_state = [_capture_state(session) for session in sessions]

            initial_tokens = []
            initial_logits = []
            for session, prompt in zip(sessions, prompts, strict=True):
                session.reset()
                first = session.prefill(prompt, return_logits=True)
                initial_tokens.append(int(first.token_id))
                initial_logits.append(np.asarray(first.logits, dtype=np.float32).copy())

            batch_tokens = [[token] for token in initial_tokens]
            batch_logits = [[logits] for logits in initial_logits]
            hidden_mismatches: list[dict[str, Any]] = []
            manifests = []
            current = list(initial_tokens)
            for step in range(int(args.decode_steps)):
                results = owner.step_batch_native(
                    current,
                    sessions=sessions,
                    return_logits=True,
                    require_logits=True,
                    scatter_state=True,
                    capture_layer_output_hidden=layer_ids,
                    physical_rows=rows,
                    active_slot_indices=tuple(range(rows)),
                )
                current = [int(result.token_id) for result in results]
                for row, (session, result) in enumerate(zip(sessions, results, strict=True)):
                    batch_tokens[row].append(current[row])
                    batch_logits[row].append(np.asarray(result.logits, dtype=np.float32).copy())
                    actual_hidden = {
                        int(layer): _hash_array(array)
                        for layer, array in session.last_layer_output_hidden.items()
                    }
                    expected_hidden = reference_hidden[row][step]
                    if actual_hidden != expected_hidden:
                        first_layer = next(
                            (
                                layer
                                for layer in sorted(set(actual_hidden) | set(expected_hidden))
                                if actual_hidden.get(layer) != expected_hidden.get(layer)
                            ),
                            None,
                        )
                        hidden_mismatches.append(
                            {
                                "row": row,
                                "step": step,
                                "first_layer": first_layer,
                                "actual_sha256": _sha256_bytes(
                                    json.dumps(actual_hidden, sort_keys=True).encode("utf-8")
                                ),
                                "expected_sha256": _sha256_bytes(
                                    json.dumps(expected_hidden, sort_keys=True).encode("utf-8")
                                ),
                            }
                        )
                manifests.append(copy.deepcopy(owner.last_packed_execution_manifest))
            batch_state = [_capture_state(session) for session in sessions]

    logit_metrics = []
    for row in range(rows):
        for step in range(int(args.decode_steps) + 1):
            metric = _logit_metrics(batch_logits[row][step], reference_logits[row][step])
            metric.update({"row": row, "step": step})
            logit_metrics.append(metric)
    max_kl = max(float(metric["kl"]) for metric in logit_metrics if metric["kl"] is not None)
    top1 = sum(bool(metric["top1_match"]) for metric in logit_metrics) / len(logit_metrics)
    state_mismatches = _state_mismatches(batch_state, reference_state)
    route_ok = bool(manifests) and all(
        manifest.get("full_attention_decode_path") == "kv_live_spans_int8_batch"
        and int(manifest.get("physical_rows", 0)) == rows
        and int(manifest.get("model_step", {}).get("host_model_row_iterations", -1)) == 0
        for manifest in manifests
    )
    token_exact = batch_tokens == reference_tokens
    logits_passed = max_kl <= 0.05 and top1 >= 0.90
    passed = bool(
        token_exact
        and logits_passed
        and not hidden_mismatches
        and not state_mismatches
        and route_ok
    )
    command = shlex.join(
        [
            *(f"{key}={os.environ[key]}" for key in ("HIP_VISIBLE_DEVICES",) if key in os.environ),
            sys.executable,
            *sys.argv,
        ]
    )
    return {
        "schema": 1,
        "kind": "qwen38_int8_row_batched_decode_correctness",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "performance_claim": False,
        "model": identity.as_dict(),
        "backend": {
            "backend": str(args.backend),
            "target_arch": hip_target_arch_for_backend(str(args.backend)),
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "expected_device": "AMD Radeon RX 7900 XTX 24GB when HIP_VISIBLE_DEVICES=1",
        },
        "capability": {
            "capability_id": resolution.capability_id,
            "admitted_max_direct_rows": admitted_rows,
            "executed_direct_rows": rows,
            "diagnostic_width_override": diagnostic_override,
            "decode_batch_variant": evidence.get("decode_batch_variant"),
        },
        "workload": {
            "rows": rows,
            "prompt_lengths": list(lengths),
            "decode_steps": int(args.decode_steps),
            "prompts": prompt_manifest,
        },
        "correctness": {
            "passed": passed,
            "token_exact": token_exact,
            "reference_trajectories": reference_tokens,
            "batch_trajectories": batch_tokens,
            "max_kl": max_kl,
            "top1_agreement": top1,
            "max_logit_abs": max(float(metric["max_abs"]) for metric in logit_metrics),
            "logit_metrics": logit_metrics,
            "hidden_mismatches": hidden_mismatches,
            "state_kv_scale_mismatches": state_mismatches,
        },
        "execution": {
            "route_ok": route_ok,
            "full_attention_decode_paths": sorted(
                {str(manifest.get("full_attention_decode_path")) for manifest in manifests}
            ),
            "physical_rows": sorted({int(manifest.get("physical_rows", 0)) for manifest in manifests}),
            "host_model_row_iterations": sorted(
                {
                    int(manifest.get("model_step", {}).get("host_model_row_iterations", -1))
                    for manifest in manifests
                }
            ),
            "decode_steps_observed": len(manifests),
        },
        "command": command,
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--prompt-lengths",
        default=",".join(str(value) for value in DEFAULT_LENGTHS),
    )
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--diagnostic-direct-rows", type=int, default=0)
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=Path("/tmp/hipengine-hipcc-version.txt"),
    )
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
