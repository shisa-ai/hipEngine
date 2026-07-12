#!/usr/bin/env python3
"""Compare fused and split-chain GGUF GDN prefill tokens and resident state.

The production lane runs the real bulk-prefill scheduler under both explicit
GDN modes.  The optional row-bulk lane captures every final-row layer output
to localize the first hidden divergence while using the same GDN dispatcher.
Artifacts contain compact fingerprints and numeric deltas, never raw tensors.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance


KIND = "hipengine_gguf_gdn_prefill_fused_chain_compare"
SCHEMA_VERSION = 1
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
GREETING_PROMPT_IDS = (
    248045,
    846,
    198,
    7734,
    264,
    2716,
    40719,
    13,
    248046,
    198,
    248045,
    74455,
    198,
    248068,
    271,
    248069,
    271,
)


class CompareError(RuntimeError):
    """Raised when the fused/chain comparison cannot be completed safely."""


def _fingerprint(array: np.ndarray) -> dict[str, Any]:
    values = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    raw = values.view(np.uint8).reshape(-1)
    values64 = values.astype(np.float64, copy=False)
    return {
        "shape": [int(dim) for dim in values.shape],
        "dtype": "fp32",
        "nbytes": int(raw.size),
        "blake2b_128": hashlib.blake2b(raw.tobytes(), digest_size=16).hexdigest(),
        "finite": bool(np.all(np.isfinite(values))),
        "rms": (
            float(math.sqrt(float(np.mean(values64 * values64))))
            if values.size
            else 0.0
        ),
        "max_abs": float(np.max(np.abs(values64))) if values.size else 0.0,
    }


def _array_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_f32 = np.ascontiguousarray(np.asarray(left, dtype=np.float32))
    right_f32 = np.ascontiguousarray(np.asarray(right, dtype=np.float32))
    if left_f32.shape != right_f32.shape:
        raise CompareError(
            f"array shape mismatch: {left_f32.shape!r} != {right_f32.shape!r}"
        )
    left_bits = left_f32.view(np.uint32)
    right_bits = right_f32.view(np.uint32)
    mismatch_elements = int(np.count_nonzero(left_bits != right_bits))
    delta = np.abs(left_f32.astype(np.float64) - right_f32.astype(np.float64))
    return {
        "exact": mismatch_elements == 0,
        "mismatch_elements": mismatch_elements,
        "max_abs": float(np.max(delta)) if delta.size else 0.0,
        "rms_diff": (
            float(math.sqrt(float(np.mean(delta * delta)))) if delta.size else 0.0
        ),
        "left": _fingerprint(left_f32),
        "right": _fingerprint(right_f32),
    }


def _build_prompt_ids(*, kind: str, token_id: int, length: int) -> list[int]:
    if kind == "greeting":
        return list(GREETING_PROMPT_IDS)
    if kind != "repeated":
        raise CompareError(f"unsupported prompt kind: {kind!r}")
    if int(length) <= 0:
        raise CompareError("repeated prompt length must be positive")
    if int(token_id) < 0:
        raise CompareError("repeated prompt token ID must be non-negative")
    return [int(token_id)] * int(length)


def _first_layer_part_divergence(
    comparisons: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any] | None:
    for layer in sorted(int(value) for value in comparisons):
        row = comparisons[layer]
        for part in ("conv", "recurrent"):
            comparison = row.get(part)
            if comparison is not None and comparison.get("exact") is not True:
                return {"layer": int(layer), "part": part}
        for part in sorted(set(row) - {"conv", "recurrent"}):
            if row[part].get("exact") is not True:
                return {"layer": int(layer), "part": str(part)}
    return None


def _classify(
    *,
    fused_token: int,
    chain_token: int,
    actual_hidden_exact: bool,
    actual_first_state: Mapping[str, Any] | None,
    bisect_first_hidden: Mapping[str, Any] | None,
    bisect_first_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    visible_token_exact = int(fused_token) == int(chain_token)
    state_exact = actual_first_state is None and bisect_first_state is None
    hidden_exact = bool(actual_hidden_exact and bisect_first_hidden is None)
    passed = bool(visible_token_exact and state_exact and hidden_exact)
    if passed:
        status = "fused_chain_exact"
        conclusion = "Fused and split-chain bulk prefill are byte-exact."
    elif visible_token_exact:
        status = "visible_token_match_state_divergence"
        conclusion = (
            "The sampled token matches, but hidden or resident GDN state diverges; "
            "the split chain is not promotable."
        )
    else:
        status = "visible_token_and_state_divergence"
        conclusion = (
            "The split chain changes the sampled token and internal state; inspect "
            "the first recorded divergence."
        )
    return {
        "passed": passed,
        "status": status,
        "visible_token_exact": visible_token_exact,
        "hidden_exact": hidden_exact,
        "state_exact": state_exact,
        "first_actual_state_divergence": actual_first_state,
        "first_bisect_hidden_divergence": bisect_first_hidden,
        "first_bisect_state_divergence": bisect_first_state,
        "conclusion": conclusion,
    }


@contextlib.contextmanager
def _gdn_mode(mode: str) -> Iterator[None]:
    name = "HIPENGINE_GGUF_GDN_PREFILL_MODE"
    previous = os.environ.get(name)
    os.environ[name] = str(mode)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _copy_device_f32(session: Any, *, ptr: int, nbytes: int) -> np.ndarray:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    size = int(nbytes)
    if size < 0 or size % np.dtype(np.float32).itemsize:
        raise CompareError(f"FP32 device buffer has invalid byte count: {size}")
    values = np.empty((size // np.dtype(np.float32).itemsize,), dtype=np.float32)
    if size:
        copy_device_to_host(
            host_array_ptr(values),
            DeviceBuffer(int(ptr), size),
            size,
            runtime=session.runtime,
        )
    return values


def _capture_resident_state(session: Any) -> dict[int, dict[str, np.ndarray]]:
    if session.scratch is None:
        raise CompareError("GGUF resident session is closed")
    captured: dict[int, dict[str, np.ndarray]] = {}
    for layer, (conv, recurrent) in enumerate(
        zip(
            session.scratch.layer_conv_states,
            session.scratch.layer_recurrent_states,
            strict=True,
        )
    ):
        if conv is None and recurrent is None:
            continue
        if conv is None or recurrent is None:
            raise CompareError(f"layer {layer} has partial Conv/GDN resident state")
        captured[int(layer)] = {
            "conv": _copy_device_f32(
                session,
                ptr=int(conv.ptr),
                nbytes=int(conv.nbytes),
            ),
            "recurrent": _copy_device_f32(
                session,
                ptr=int(recurrent.ptr),
                nbytes=int(recurrent.nbytes),
            ),
        }
    return captured


def _capture_hidden_seed(session: Any) -> np.ndarray:
    if session.runner is None or session.scratch is None:
        raise CompareError("GGUF resident session is closed")
    if not session.fp32_hidden_seed_contract().ready_for_mtp:
        raise CompareError("bulk prefill did not populate the FP32 hidden seed")
    return _copy_device_f32(
        session,
        ptr=int(session.scratch.hidden_seed_fp32.ptr),
        nbytes=int(session.runner.hidden_size) * np.dtype(np.float32).itemsize,
    )


def _run_production_mode(
    session: Any,
    *,
    mode: str,
    prompt_ids: Sequence[int],
    bulk_attention_mode: str,
) -> dict[str, Any]:
    session.reset()
    session.runtime.device_synchronize()
    started = time.perf_counter()
    with _gdn_mode(mode):
        result = session.prefill(
            [int(token) for token in prompt_ids],
            use_bulk=True,
            bulk_attention_mode=str(bulk_attention_mode),
            return_logits=False,
            capture_hidden_seed_fp32=True,
        )
    session.runtime.device_synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "token_id": int(result.token_id),
        "hidden_seed": _capture_hidden_seed(session),
        "linear_states": _capture_resident_state(session),
        "wall_ms_diagnostic": float(wall_ms),
    }


def _run_layer_bisect_mode(
    session: Any,
    *,
    mode: str,
    prompt_ids: Sequence[int],
    bulk_attention_mode: str,
) -> dict[str, Any]:
    if session.runner is None or session.runner.weights is None:
        raise CompareError("GGUF resident session is closed")
    layer_ids = list(range(len(session.runner.weights.config.layer_types)))
    session.reset()
    with _gdn_mode(mode):
        result = session.verify_target_block(
            [int(token) for token in prompt_ids],
            bulk_attention_mode=str(bulk_attention_mode),
            capture_layer_output_hidden=layer_ids,
        )
    if result.layer_output_hidden is None:
        raise CompareError("row-bulk diagnostic did not capture layer outputs")
    layer_final_rows = {
        int(layer): np.ascontiguousarray(rows[-1], dtype=np.float32)
        for layer, rows in result.layer_output_hidden.items()
    }
    return {
        "token_id": int(result.token_ids[-1]),
        "hidden_seed": np.ascontiguousarray(result.hidden_seeds[-1], dtype=np.float32),
        "layer_final_rows": layer_final_rows,
        "linear_states": _capture_resident_state(session),
    }


def _compact_states(
    states: Mapping[int, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    return [
        {
            "layer": int(layer),
            "conv": _fingerprint(states[layer]["conv"]),
            "recurrent": _fingerprint(states[layer]["recurrent"]),
        }
        for layer in sorted(states)
    ]


def _compare_states(
    fused: Mapping[int, Mapping[str, np.ndarray]],
    chain: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[int, dict[str, dict[str, Any]]]:
    if set(fused) != set(chain):
        raise CompareError("fused and chain captured different linear-state layers")
    return {
        int(layer): {
            part: _array_comparison(fused[layer][part], chain[layer][part])
            for part in ("conv", "recurrent")
        }
        for layer in sorted(fused)
    }


def _compact_state_comparisons(
    comparisons: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {
            "layer": int(layer),
            **{part: dict(comparisons[layer][part]) for part in comparisons[layer]},
        }
        for layer in sorted(comparisons)
    ]


def _prompt_record(prompt_ids: Sequence[int], *, kind: str) -> dict[str, Any]:
    values = np.ascontiguousarray(prompt_ids, dtype=np.int64)
    record: dict[str, Any] = {
        "kind": str(kind),
        "length": int(values.size),
        "sha256_i64": hashlib.sha256(values.view(np.uint8).tobytes()).hexdigest(),
        "first_token_ids": [int(token) for token in values[:16].tolist()],
        "last_token_ids": [int(token) for token in values[-16:].tolist()],
    }
    if values.size <= 64:
        record["token_ids"] = [int(token) for token in values.tolist()]
    return record


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    prompt_ids = _build_prompt_ids(
        kind=str(args.prompt_kind),
        token_id=int(args.prompt_token_id),
        length=int(args.prompt_length),
    )
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise CompareError(f"model does not exist: {model}")
    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8")

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    with Qwen35GGUFResidentSession(
        model,
        backend=str(args.backend),
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max(64, len(prompt_ids) + 2),
    ) as session:
        production = {
            mode: _run_production_mode(
                session,
                mode=mode,
                prompt_ids=prompt_ids,
                bulk_attention_mode=str(args.bulk_attention_mode),
            )
            for mode in ("fused", "chain")
        }
        bisect = None
        if not bool(args.skip_layer_bisect):
            bisect = {
                mode: _run_layer_bisect_mode(
                    session,
                    mode=mode,
                    prompt_ids=prompt_ids,
                    bulk_attention_mode=str(args.bulk_attention_mode),
                )
                for mode in ("fused", "chain")
            }
        if session.runner is None:
            raise CompareError("GGUF resident session closed before provenance capture")
        resolved_backend = str(session.runner.backend)
        target_arch = str(session.runner.target_arch)

    actual_hidden = _array_comparison(
        production["fused"]["hidden_seed"],
        production["chain"]["hidden_seed"],
    )
    actual_states = _compare_states(
        production["fused"]["linear_states"],
        production["chain"]["linear_states"],
    )
    actual_first_state = _first_layer_part_divergence(actual_states)

    bisect_record = None
    bisect_first_hidden = None
    bisect_first_state = None
    if bisect is not None:
        fused_layers = bisect["fused"]["layer_final_rows"]
        chain_layers = bisect["chain"]["layer_final_rows"]
        if set(fused_layers) != set(chain_layers):
            raise CompareError("fused and chain captured different hidden-output layers")
        layer_comparisons = {
            int(layer): _array_comparison(fused_layers[layer], chain_layers[layer])
            for layer in sorted(fused_layers)
        }
        bisect_first_hidden = next(
            (
                {"layer": int(layer), "part": "layer_output"}
                for layer in sorted(layer_comparisons)
                if layer_comparisons[layer]["exact"] is not True
            ),
            None,
        )
        bisect_states = _compare_states(
            bisect["fused"]["linear_states"],
            bisect["chain"]["linear_states"],
        )
        bisect_first_state = _first_layer_part_divergence(bisect_states)
        bisect_record = {
            "route": "verify_target_block_all_layer_final_row",
            "tokens": {
                mode: int(bisect[mode]["token_id"]) for mode in ("fused", "chain")
            },
            "hidden_seed": _array_comparison(
                bisect["fused"]["hidden_seed"],
                bisect["chain"]["hidden_seed"],
            ),
            "layer_final_rows": [
                {"layer": int(layer), "comparison": layer_comparisons[layer]}
                for layer in sorted(layer_comparisons)
            ],
            "linear_states": _compact_state_comparisons(bisect_states),
            "first_hidden_divergence": bisect_first_hidden,
            "first_state_divergence": bisect_first_state,
        }

    classification = _classify(
        fused_token=int(production["fused"]["token_id"]),
        chain_token=int(production["chain"]["token_id"]),
        actual_hidden_exact=bool(actual_hidden["exact"]),
        actual_first_state=actual_first_state,
        bisect_first_hidden=bisect_first_hidden,
        bisect_first_state=bisect_first_state,
    )
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "correctness_claim": True,
        "workload": {
            "model": str(model),
            "quant": "gguf_q4_k_m",
            "kv_dtype": "bf16",
            "prompt": _prompt_record(prompt_ids, kind=str(args.prompt_kind)),
            "bulk_attention_mode": str(args.bulk_attention_mode),
            "gdn_modes": ["fused", "chain"],
            "segment_threshold": os.environ.get(
                "HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD", "1025"
            ),
        },
        "production_bulk_prefill": {
            "tokens": {
                mode: int(production[mode]["token_id"])
                for mode in ("fused", "chain")
            },
            "wall_ms_diagnostic": {
                mode: float(production[mode]["wall_ms_diagnostic"])
                for mode in ("fused", "chain")
            },
            "hidden_seed": actual_hidden,
            "mode_state_fingerprints": {
                mode: _compact_states(production[mode]["linear_states"])
                for mode in ("fused", "chain")
            },
            "linear_state_comparisons": _compact_state_comparisons(actual_states),
            "first_state_divergence": actual_first_state,
        },
        "layer_bisect": bisect_record,
        "classification": classification,
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
                "HIPENGINE_GGUF_GDN_PREFILL_MODE": "explicit fused/chain sweep",
                "HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD": os.environ.get(
                    "HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD"
                ),
            },
            build_profile="gguf_gdn_prefill_fused_chain_compare",
            timing_protocol="correctness_only_single_order_diagnostic_wall",
            warmups=0,
            repetitions=1,
            profiler={"enabled": False, "kind": None, "command": None},
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--prompt-kind",
        choices=("greeting", "repeated"),
        default="greeting",
    )
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument(
        "--bulk-attention-mode",
        choices=("bulk", "native"),
        default="bulk",
    )
    parser.add_argument("--skip-layer-bisect", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--allow-mismatch", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    command = [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *raw_argv]
    try:
        artifact = run(args, command=command)
    except (CompareError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.json)
    print(json.dumps(artifact["classification"], indent=2, sort_keys=True))
    if artifact["classification"]["passed"] or bool(args.allow_mismatch):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
