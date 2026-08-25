#!/usr/bin/env python3
"""Exact-Q4 dense/no-evict/CR2/CR4/CR8 compact-DMS logit quality gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
)
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_full_attn_gate_mul_bf16,
)
from hipengine.kvcache import (
    DMSExternalDecisionRuntime,
    ExternalDMSDecisionCollector,
    ExternalDMSLinearSidecar,
    compact_attention_reference,
    create_dms_bf16_backend,
    load_dms_retrofit_config,
    load_external_dms_sidecar,
)
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.gguf_linear import launch_gguf_linear
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

_EXPECTED_PHYSICAL_LAYERS = tuple(range(3, 64, 4))
_REQUIRED_CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_bf16(ptr: int, shape: tuple[int, ...], runtime: Any) -> np.ndarray:
    bits = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(
        host_array_ptr(bits),
        DeviceBuffer(int(ptr), bits.nbytes),
        bits.nbytes,
        runtime=runtime,
    )
    return np.ascontiguousarray(bf16_to_float32(bits), dtype=np.float32)


def _copy_f32(ptr: int, shape: tuple[int, ...], runtime: Any) -> np.ndarray:
    values = np.empty(shape, dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(values),
        DeviceBuffer(int(ptr), values.nbytes),
        values.nbytes,
        runtime=runtime,
    )
    return values


def _row_kl(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref -= np.max(ref)
    cand -= np.max(cand)
    ref_prob = np.exp(ref)
    cand_prob = np.exp(cand)
    ref_prob /= np.sum(ref_prob)
    cand_prob /= np.sum(cand_prob)
    return float(
        np.sum(
            ref_prob
            * (
                np.log(np.maximum(ref_prob, 1.0e-300))
                - np.log(np.maximum(cand_prob, 1.0e-300))
            )
        )
    )


def summarize_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("quality summary requires rows")
    kls = np.asarray([float(row["kl"]) for row in rows], dtype=np.float64)
    top1 = np.asarray([bool(row["top1_match"]) for row in rows], dtype=np.bool_)
    return {
        "rows": len(rows),
        "kl_mean": float(np.mean(kls)),
        "kl_p95": float(np.quantile(kls, 0.95)),
        "kl_max": float(np.max(kls)),
        "top1_agreement": float(np.mean(top1)),
        "outer_floor_passed": bool(float(np.max(kls)) <= 0.05 and float(np.mean(top1)) >= 0.90),
    }


def load_scenario_thresholds(
    replay_path: str | Path,
    *,
    expected_sidecar_sha256: str,
) -> dict[str, float | None]:
    replay = json.loads(Path(replay_path).read_text(encoding="utf-8"))
    if str(replay.get("sidecar_sha256")) != str(expected_sidecar_sha256):
        raise ValueError("DMS replay thresholds belong to a different sidecar")
    calibration = replay.get("calibration", {}).get("scenarios", {})
    thresholds: dict[str, float | None] = {"no_evict": None}
    for scenario in ("cr2", "cr4", "cr8"):
        row = calibration.get(scenario)
        if not isinstance(row, dict) or not np.isfinite(float(row.get("threshold", np.nan))):
            raise ValueError(f"DMS replay lacks finite {scenario} threshold")
        thresholds[scenario] = float(row["threshold"])
    return thresholds


@dataclass(slots=True)
class _DenseTrace:
    initial_token: int
    input_tokens: tuple[int, ...]
    output_tokens: tuple[int, ...]
    logits: tuple[np.ndarray, ...]


class _CompactAttentionOverride:
    def __init__(
        self,
        session: Qwen35GGUFResidentSession,
        backend: Any,
        source: ExternalDMSLinearSidecar,
        *,
        request_id: int,
        force_no_evict: bool,
    ) -> None:
        self.session = session
        self.backend = backend
        self.source = source
        self.request_id = int(request_id)
        self.force_no_evict = bool(force_no_evict)
        self.original = session.runner._run_full_attention_attn_only
        config = source.config
        self.hidden = np.zeros((config.num_layers, int(config.hidden_size)), dtype=np.float32)
        self.k = np.zeros(
            (config.num_layers, config.num_kv_heads, config.head_dim), dtype=np.float32
        )
        self.v = np.zeros_like(self.k)
        self.seen: set[int] = set()

    def install(self) -> None:
        owner = self

        def wrapped(layer_id, hidden_ptr, attn_out_ptr, scratch, **kwargs):
            owner.original(layer_id, hidden_ptr, attn_out_ptr, scratch, **kwargs)
            owner._overwrite_attention(
                layer_id=int(layer_id),
                attn_out_ptr=int(attn_out_ptr),
                scratch=scratch,
                position=int(kwargs["position"]),
                stream=int(kwargs.get("stream", 0)),
            )

        self.session.runner._run_full_attention_attn_only = wrapped

    def uninstall(self) -> None:
        self.session.runner._run_full_attention_attn_only = self.original

    def begin_step(self) -> None:
        self.seen.clear()

    def _overwrite_attention(
        self,
        *,
        layer_id: int,
        attn_out_ptr: int,
        scratch: Any,
        position: int,
        stream: int,
    ) -> None:
        config = self.source.config
        compact_layer = config.physical_layer_ids.index(layer_id)
        runtime = self.session.runtime
        if stream:
            runtime.stream_synchronize(stream)
        else:
            runtime.device_synchronize()
        norm = _copy_bf16(
            scratch.norm.ptr,
            (1, int(config.hidden_size)),
            runtime,
        )[0]
        query = _copy_f32(
            scratch.full_query.ptr,
            (config.num_q_heads, config.head_dim),
            runtime,
        )
        key_current = _copy_f32(
            scratch.full_key.ptr,
            (config.num_kv_heads, config.head_dim),
            runtime,
        )
        value_current = _copy_bf16(
            scratch.full_v.ptr,
            (config.num_kv_heads, config.head_dim),
            runtime,
        )
        self.hidden[compact_layer] = norm
        self.k[compact_layer] = key_current
        self.v[compact_layer] = value_current
        self.seen.add(compact_layer)

        state = self.backend.state_for_request(self.request_id)
        counts: list[int] = []
        keys: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for head in range(config.num_kv_heads):
            live = int(state.live_counts[compact_layer, head])
            positions = state.token_positions[compact_layer, head, :live]
            prior_evict = state.evict_mask[compact_layer, head, :live]
            keep = (~prior_evict) | (
                int(position) - positions <= config.window_size
            )
            key_rows = np.concatenate(
                (state.k_payload[(compact_layer, head)][keep], key_current[head : head + 1]),
                axis=0,
            )
            value_rows = np.concatenate(
                (state.v_payload[(compact_layer, head)][keep], value_current[head : head + 1]),
                axis=0,
            )
            counts.append(len(key_rows))
            keys.append(key_rows)
            values.append(value_rows)
        max_live = max(counts)
        key_batch = np.zeros(
            (1, config.num_kv_heads, max_live, config.head_dim), dtype=np.float32
        )
        value_batch = np.zeros_like(key_batch)
        for head in range(config.num_kv_heads):
            key_batch[0, head, : counts[head]] = keys[head]
            value_batch[0, head, : counts[head]] = values[head]
        context = compact_attention_reference(
            query[None],
            key_batch,
            value_batch,
            np.asarray(counts, dtype=np.int32)[None],
        )[0]
        context = np.ascontiguousarray(context, dtype=np.float32)
        copy_host_to_device(
            DeviceBuffer(scratch.full_attn_context.ptr, context.nbytes),
            host_array_ptr(context),
            context.nbytes,
            runtime=runtime,
        )
        qwen35_full_attn_gate_mul_bf16(
            scratch.full_attn_context.ptr,
            scratch.full_gate.ptr,
            scratch.full_gated.ptr,
            config.num_q_heads * config.head_dim,
            stream=stream,
            runtime=runtime,
        )
        layer = self.session.runner.weights.layer(layer_id)
        launch_gguf_linear(
            layer.weight("attn_output"),
            scratch.full_gated.ptr,
            attn_out_ptr,
            rows=1,
            in_features=self.session.runner.q_width,
            out_features=self.session.runner.hidden_size,
            stream=stream,
            runtime=runtime,
        )

    def commit_step(self, position: int) -> None:
        config = self.source.config
        if self.seen != set(range(config.num_layers)):
            raise RuntimeError("compact quality override did not visit every full-attention layer")
        if self.force_no_evict:
            decisions = np.zeros(
                (config.num_layers, config.num_kv_heads), dtype=np.bool_
            )
            lease = self.backend.lease_for_request(self.request_id)
            operation = self.backend.begin_transaction((lease,), None)
            try:
                self.backend.append_decode(
                    self.request_id,
                    self.k,
                    self.v,
                    decisions,
                    position=int(position),
                )
            except Exception:
                self.backend.rollback(operation)
                raise
            self.backend.commit(operation, None)
        else:
            DMSExternalDecisionRuntime(self.source).append_decode(
                self.backend,
                request_id=self.request_id,
                hidden=self.hidden,
                k=self.k,
                v=self.v,
                position=int(position),
            )


def _dense_trace(
    session: Qwen35GGUFResidentSession,
    prompt: list[int],
    *,
    decode_steps: int,
) -> _DenseTrace:
    first = session.prefill(
        prompt,
        use_bulk=True,
        bulk_attention_mode="bulk",
        return_logits=True,
    )
    current = int(first.token_id)
    inputs: list[int] = []
    outputs: list[int] = []
    logits: list[np.ndarray] = []
    for _ in range(int(decode_steps)):
        inputs.append(current)
        result = session.step(current, return_logits=True)
        outputs.append(int(result.token_id))
        logits.append(np.ascontiguousarray(result.logits, dtype=np.float32))
        current = int(result.token_id)
    return _DenseTrace(
        initial_token=int(first.token_id),
        input_tokens=tuple(inputs),
        output_tokens=tuple(outputs),
        logits=tuple(logits),
    )


def _prompt_kv(
    session: Qwen35GGUFResidentSession,
    *,
    prompt_tokens: int,
    config: Any,
) -> tuple[np.ndarray, np.ndarray]:
    runtime = session.runtime
    key = np.empty(
        (
            prompt_tokens,
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
        ),
        dtype=np.float32,
    )
    value = np.empty_like(key)
    for compact_layer, physical_layer in enumerate(config.physical_layer_ids):
        key_buffer, value_buffer = session.scratch.full_cache(physical_layer)
        shape = (prompt_tokens, config.num_kv_heads, config.head_dim)
        key[:, compact_layer] = _copy_bf16(key_buffer.ptr, shape, runtime)
        value[:, compact_layer] = _copy_bf16(value_buffer.ptr, shape, runtime)
    return key, value


def _compact_trace(
    session: Qwen35GGUFResidentSession,
    prompt: list[int],
    dense: _DenseTrace,
    source: ExternalDMSLinearSidecar,
    *,
    force_no_evict: bool,
) -> tuple[tuple[int, ...], tuple[np.ndarray, ...], dict[str, Any]]:
    config = source.config
    collector = ExternalDMSDecisionCollector(source, token_count=len(prompt))
    first = session.prefill(
        prompt,
        use_bulk=True,
        bulk_attention_mode="bulk",
        return_logits=True,
        dms_capture=collector,
    )
    if int(first.token_id) != dense.initial_token:
        raise RuntimeError("dense and compact prefills produced different initial tokens")
    decisions = collector.finalize()
    key, value = _prompt_kv(session, prompt_tokens=len(prompt), config=config)
    backend_config = (
        replace(config, target_compression_ratio=1)
        if force_no_evict
        else config
    )
    backend = create_dms_bf16_backend(
        retrofit=backend_config,
        slots_per_layer=max(4096, config.num_kv_heads * (len(prompt) + len(dense.input_tokens) + 8)),
        max_request_rows=1,
        max_pack_rows=len(prompt),
        device_payloads=False,
    )
    request_id = 1
    request = SimpleNamespace(
        request_id=request_id,
        prompt_tokens=tuple(prompt),
        max_new_tokens=len(dense.input_tokens) + 2,
    )
    lease = backend.reserve(backend.estimate(request, None, {}))
    backend.streaming_pack(
        request_id,
        key,
        value,
        np.zeros_like(decisions) if force_no_evict else decisions,
    )
    owner = _CompactAttentionOverride(
        session,
        backend,
        source,
        request_id=request_id,
        force_no_evict=force_no_evict,
    )
    outputs: list[int] = []
    logits: list[np.ndarray] = []
    owner.install()
    try:
        for step, token in enumerate(dense.input_tokens):
            owner.begin_step()
            result = session.step(int(token), return_logits=True)
            owner.commit_step(len(prompt) + step)
            outputs.append(int(result.token_id))
            logits.append(np.ascontiguousarray(result.logits, dtype=np.float32))
    finally:
        owner.uninstall()
    snapshot = backend.observability_snapshot()
    backend.reclaim(lease)
    backend.assert_conserved()
    return tuple(outputs), tuple(logits), snapshot


def _git_provenance() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "source_commit": commit,
        "working_tree_clean": not bool(status.strip()),
        "command": [str(value) for value in sys.argv],
        "host": platform.node(),
        "python": sys.version,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = args.model.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    replay_path = args.replay.expanduser().resolve()
    data_manifest = args.data_manifest.expanduser().resolve()
    config = load_dms_retrofit_config(
        model,
        metadata_path=metadata,
        expected_artifact_fingerprint=str(args.expected_artifact),
        expected_physical_layer_ids=_EXPECTED_PHYSICAL_LAYERS,
    )
    base_source = load_external_dms_sidecar(config)
    thresholds = load_scenario_thresholds(
        replay_path,
        expected_sidecar_sha256=config.sidecar.sha256,
    )
    data = json.loads(data_manifest.read_text(encoding="utf-8"))
    validation = [
        row
        for row in data["sequences"]
        if row.get("split") == "validation"
    ]
    requested_ids = None
    if args.sequence_ids:
        requested_ids = {value for value in args.sequence_ids.split(",") if value}
        validation = [row for row in validation if row["sequence_id"] in requested_ids]
    categories = {str(row["category"]) for row in validation}
    if not validation or (requested_ids is None and categories != set(_REQUIRED_CATEGORIES)):
        raise ValueError("quality gate requires validation rows covering all categories")
    scenario_names = tuple(value for value in args.scenarios.split(",") if value)
    if any(name not in thresholds for name in scenario_names):
        raise ValueError("quality scenarios must be no_evict,cr2,cr4,cr8")
    max_sequence_length = max(len(row["token_ids"]) for row in validation) + int(
        args.decode_steps
    ) + 4
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in scenario_names}
    repeat_hashes: dict[str, dict[str, list[str]]] = {
        name: {} for name in scenario_names
    }
    compression: dict[str, dict[str, int]] = {
        name: {"logical_cells": 0, "live_cells": 0} for name in scenario_names
    }
    with Qwen35GGUFResidentSession(
        model,
        backend=str(args.backend),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as dense_session:
        compact_session = Qwen35GGUFResidentSession(
            model,
            backend=str(args.backend),
            runtime=dense_session.runtime,
            shared_runner=dense_session.runner,
            max_sequence_length=max_sequence_length,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        )
        try:
            for sequence in validation:
                prompt = [int(token) for token in sequence["token_ids"]]
                dense = _dense_trace(
                    dense_session,
                    prompt,
                    decode_steps=int(args.decode_steps),
                )
                for scenario in scenario_names:
                    force_no_evict = scenario == "no_evict"
                    scenario_cr = 1 if force_no_evict else int(scenario.removeprefix("cr"))
                    scenario_config = replace(
                        config,
                        target_compression_ratio=scenario_cr,
                        alpha_offset=(
                            config.alpha_offset
                            if force_no_evict
                            else float(thresholds[scenario])
                        ),
                    )
                    source = ExternalDMSLinearSidecar(
                        config=scenario_config,
                        weight=base_source.weight,
                        bias=base_source.bias,
                    )
                    sequence_repeat_hashes: list[str] = []
                    for repeat in range(int(args.repeats)):
                        outputs, candidate_logits, snapshot = _compact_trace(
                            compact_session,
                            prompt,
                            dense,
                            source,
                            force_no_evict=force_no_evict,
                        )
                        digest = hashlib.sha256()
                        for step, (reference_logits, logits) in enumerate(
                            zip(dense.logits, candidate_logits, strict=True)
                        ):
                            digest.update(logits.tobytes(order="C"))
                            rows[scenario].append(
                                {
                                    "sequence_id": sequence["sequence_id"],
                                    "category": sequence["category"],
                                    "repeat": repeat,
                                    "step": step,
                                    "input_token": dense.input_tokens[step],
                                    "dense_token": dense.output_tokens[step],
                                    "candidate_token": outputs[step],
                                    "kl": _row_kl(reference_logits, logits),
                                    "top1_match": bool(
                                        int(np.argmax(reference_logits))
                                        == int(np.argmax(logits))
                                    ),
                                }
                            )
                        sequence_repeat_hashes.append(digest.hexdigest())
                        if repeat == 0:
                            logical = int(snapshot["capacity"]["logical_token_rows"])
                            live = int(snapshot["capacity"]["live_token_rows"])
                            compression[scenario]["logical_cells"] += logical
                            compression[scenario]["live_cells"] += live
                    repeat_hashes[scenario][sequence["sequence_id"]] = sequence_repeat_hashes
        finally:
            compact_session.close()
    scenarios: dict[str, Any] = {}
    for scenario in scenario_names:
        scenario_rows = rows[scenario]
        by_category = {
            category: summarize_quality(
                [row for row in scenario_rows if row["category"] == category]
            )
            for category in sorted({row["category"] for row in scenario_rows})
        }
        deterministic = all(
            len(set(hashes)) == 1 for hashes in repeat_hashes[scenario].values()
        )
        logical = compression[scenario]["logical_cells"]
        live = compression[scenario]["live_cells"]
        scenarios[scenario] = {
            "threshold": thresholds[scenario],
            "quality": summarize_quality(scenario_rows),
            "by_category": by_category,
            "deterministic_repeats": deterministic,
            "repeat_hashes": repeat_hashes[scenario],
            "compression": {
                "logical_cells": logical,
                "live_cells": live,
                "actual_compression_ratio": logical / max(1, live),
            },
            "rows": scenario_rows,
        }
    no_evict = scenarios.get("no_evict")
    if no_evict is None or not no_evict["quality"]["outer_floor_passed"]:
        raise RuntimeError("DMS no-evict compact control failed the outer quality floor")
    result = {
        "schema_version": 1,
        "kind": "hipengine_qwen38_dms_exact_q4_quality",
        "performance_claim": False,
        "model": {"path": str(model), "sha256": str(args.expected_artifact)},
        "metadata": {"path": str(metadata), "sha256": _sha256_file(metadata)},
        "sidecar_sha256": config.sidecar.sha256,
        "sidecar_fingerprint": config.fingerprint,
        "data_manifest": {"path": str(data_manifest), "sha256": _sha256_file(data_manifest)},
        "replay_thresholds": {"path": str(replay_path), "sha256": _sha256_file(replay_path)},
        "protocol": {
            "split": "validation",
            "sequence_ids": [row["sequence_id"] for row in validation],
            "decode_steps": int(args.decode_steps),
            "repeats": int(args.repeats),
            "route": "dense_prefill_then_host_compact_decode_override",
            "dense_shadow": True,
            "purpose": "quality-only; not allocator or performance evidence",
        },
        "provenance": _git_provenance(),
        "scenarios": scenarios,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n",
        encoding="ascii",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--expected-artifact", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--scenarios", default="no_evict,cr2,cr4,cr8")
    parser.add_argument("--sequence-ids", default="")
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    compact = {
        "sidecar_sha256": result["sidecar_sha256"],
        "sidecar_fingerprint": result["sidecar_fingerprint"],
        "scenarios": {
            name: {
                "quality": row["quality"],
                "compression": row["compression"],
                "deterministic_repeats": row["deterministic_repeats"],
            }
            for name, row in result["scenarios"].items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
