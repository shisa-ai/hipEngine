#!/usr/bin/env python3
"""Full-model quality gate for diagnostic Laguna source-F16 Q8 decode."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Iterator, Sequence

import numpy as np

from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_quantize_bf16_q8_1,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dp4a_gemv import (
    build_gguf_q8_0_dp4a_gemv,
    gguf_q8_0_dp4a_gemv_bf16_bf16_out,
    gguf_q8_0_dp4a_gemv_f32_f32_out,
    gguf_q8_0_dp4a_triple_split_rowtile4_gemv_f32_f32_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q8_0_gemv_bf16_f32_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv import (
    build_gguf_q8_0_t16_gemv,
    gguf_q8_0_t16_gemv_decode_f32_bf16_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.laguna_gguf_materialize import LAYOUT_DENSE_F16
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_t16 import repack_gguf_q8_0_tile16
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_f16_decode_q8_screen import (
    _Q8_1_BYTES,
    _layer_id,
    _quantize_f16_q8_0,
    _upload,
)
from scripts.laguna_prefill_profile import _profile_token_stream
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_PROMPTS,
    _compiler_version,
    _load_prompts,
    _progress,
)

_BLOCK = 32
_MODES = ("all", "qkv_gate", "output")
_MODE_CHOICES = ("control", *_MODES)
_ACTIVATION_PATHS = ("q8_1_dp4a", "exact")
_SCOPES = (
    "all",
    "full",
    "swa",
    "even",
    "odd",
    "first24",
    "last24",
    "mod0",
    "mod1",
    "mod2",
    "mod3",
)
_SLOTS = ("attn_q", "attn_k", "attn_v", "attn_gate", "attn_output")


def _parse_list(
    value: str,
    *,
    choices: Sequence[str],
    noun: str,
) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected or len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError(f"{noun} must be distinct and non-empty")
    invalid = sorted(set(selected) - set(choices))
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported {noun}: {invalid}")
    return selected


def _parse_modes(value: str) -> tuple[str, ...]:
    return _parse_list(value, choices=_MODE_CHOICES, noun="modes")


def _parse_scopes(value: str) -> tuple[str, ...]:
    return _parse_list(value, choices=_SCOPES, noun="scopes")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--modes", type=_parse_modes, default=_MODES)
    parser.add_argument("--scopes", type=_parse_scopes, default=("all",))
    parser.add_argument(
        "--activation-path",
        choices=_ACTIVATION_PATHS,
        default="q8_1_dp4a",
        help=(
            "q8_1_dp4a reproduces the original weight+activation screen; "
            "exact keeps BF16/F32 activations unquantized and changes only weights"
        ),
    )
    parser.add_argument("--kl-limit", type=float, default=0.05)
    parser.add_argument("--top1-minimum", type=float, default=0.90)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _classify_source_name(name: str) -> str | None:
    if name.endswith((".attn_q.weight", ".attn_k.weight", ".attn_v.weight")):
        return "qkv"
    if name.endswith(".attn_gate.weight"):
        return "gate"
    if name.endswith(".attn_output.weight"):
        return "output"
    return None


def _scope_layers(
    scope: str,
    attention_types: Sequence[str],
) -> frozenset[int]:
    layers = frozenset(range(len(attention_types)))
    if scope == "all":
        return layers
    if scope == "full":
        return frozenset(
            layer
            for layer, attention_type in enumerate(attention_types)
            if attention_type == "full_attention"
        )
    if scope == "swa":
        return frozenset(
            layer
            for layer, attention_type in enumerate(attention_types)
            if attention_type == "sliding_attention"
        )
    if scope == "even":
        return frozenset(range(0, len(attention_types), 2))
    if scope == "odd":
        return frozenset(range(1, len(attention_types), 2))
    if scope == "first24":
        return frozenset(range(min(24, len(attention_types))))
    if scope == "last24":
        return frozenset(range(max(0, len(attention_types) - 24), len(attention_types)))
    if scope.startswith("mod") and scope[3:].isdigit():
        return frozenset(range(int(scope[3:]), len(attention_types), 4))
    raise ValueError(f"unsupported Q8 layer scope {scope!r}")


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    return shifted - np.log(np.sum(np.exp(shifted)))


def _kl_divergence(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_log = _log_softmax(reference)
    candidate_log = _log_softmax(candidate)
    reference_probability = np.exp(reference_log)
    return float(
        np.sum(reference_probability * (reference_log - candidate_log))
    )


def _summarize_records(
    records: Sequence[dict[str, object]],
    *,
    kl_limit: float,
    top1_minimum: float,
) -> dict[str, object]:
    if not records:
        raise ValueError("quality summary requires at least one record")
    kls = np.asarray([float(record["kl"]) for record in records], dtype=np.float64)
    matches = np.asarray(
        [bool(record["top1_match"]) for record in records],
        dtype=np.bool_,
    )
    maximum_kl = float(np.max(kls))
    top1_agreement = float(np.mean(matches))
    return {
        "passed": bool(
            maximum_kl <= float(kl_limit)
            and top1_agreement >= float(top1_minimum)
        ),
        "mean_kl": float(np.mean(kls)),
        "p95_kl": float(np.percentile(kls, 95)),
        "maximum_kl": maximum_kl,
        "top1_agreement": top1_agreement,
        "kl_limit": float(kl_limit),
        "top1_minimum": float(top1_minimum),
        "first_top1_mismatch": next(
            (
                {
                    "step": int(record["step"]),
                    "reference_token": int(record["reference_token"]),
                    "candidate_token": int(record["candidate_token"]),
                    "kl": float(record["kl"]),
                }
                for record in records
                if not bool(record["top1_match"])
            ),
            None,
        ),
    }


class Q8Sidecar:
    """Diagnostic all-layer Q8_0 weights plus one reusable Q8_1 row."""

    def __init__(
        self,
        owner: LagunaGGUFResidentSession,
        reader: GGUFReader,
    ) -> None:
        assert owner.weights is not None
        self.runtime = owner.runtime
        self.weights: dict[str, DeviceBuffer] = {}
        self.source_nbytes = 0
        self.q8_nbytes = 0
        self.build_seconds = 0.0
        maximum_in_features = 0
        started = time.perf_counter()
        try:
            for layer in owner.weights.layers:
                for slot in _SLOTS:
                    weight = layer.weight(slot)
                    if weight.spec.layout != LAYOUT_DENSE_F16:
                        raise ValueError(
                            f"Q8 sidecar requires source F16, got {weight.spec.layout}"
                        )
                    source_name = weight.spec.source.name
                    if (
                        GGMLQuantizationType(weight.spec.source.ggml_type)
                        != GGMLQuantizationType.F16
                    ):
                        raise ValueError(
                            f"Q8 sidecar source must be GGML F16: {source_name}"
                        )
                    source = np.asarray(reader.tensor_data(source_name))
                    if source.ndim != 2:
                        raise ValueError(
                            f"Q8 sidecar source must be rank two: {source_name}"
                        )
                    maximum_in_features = max(maximum_in_features, source.shape[1])
                    packed = _quantize_f16_q8_0(source)
                    self.weights[source_name] = _upload(self.runtime, packed)
                    self.source_nbytes += int(source.nbytes)
                    self.q8_nbytes += int(packed.nbytes)
                    del packed
                print(
                    f"q8 sidecar layer {layer.layer_id + 1}/48 "
                    f"{self.q8_nbytes / (1024 ** 3):.3f} GiB",
                    flush=True,
                )
            self.activation = malloc(
                maximum_in_features // _BLOCK * _Q8_1_BYTES,
                runtime=self.runtime,
            )
        except BaseException:
            self.close()
            raise
        self.build_seconds = time.perf_counter() - started

    def close(self) -> None:
        activation = getattr(self, "activation", None)
        if activation is not None:
            free(activation, runtime=self.runtime)
            self.activation = None
        for allocation in reversed(tuple(self.weights.values())):
            free(allocation, runtime=self.runtime)
        self.weights.clear()

    def __enter__(self) -> "Q8Sidecar":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class Q8WeightOnlySidecar:
    """Diagnostic Q8_0 weights for the exact-activation quality screen."""

    def __init__(
        self,
        owner: LagunaGGUFResidentSession,
        reader: GGUFReader,
    ) -> None:
        assert owner.weights is not None
        self.runtime = owner.runtime
        self.raw_weights: dict[str, DeviceBuffer] = {}
        self.output_t16_weights: dict[str, DeviceBuffer] = {}
        self.source_nbytes = 0
        self.q8_nbytes = 0
        self.output_t16_nbytes = 0
        self.build_seconds = 0.0
        started = time.perf_counter()
        try:
            for layer in owner.weights.layers:
                for slot in _SLOTS:
                    weight = layer.weight(slot)
                    if weight.spec.layout != LAYOUT_DENSE_F16:
                        raise ValueError(
                            f"Q8 sidecar requires source F16, got {weight.spec.layout}"
                        )
                    source_name = weight.spec.source.name
                    if (
                        GGMLQuantizationType(weight.spec.source.ggml_type)
                        != GGMLQuantizationType.F16
                    ):
                        raise ValueError(
                            f"Q8 sidecar source must be GGML F16: {source_name}"
                        )
                    source = np.asarray(reader.tensor_data(source_name))
                    if source.ndim != 2:
                        raise ValueError(
                            f"Q8 sidecar source must be rank two: {source_name}"
                        )
                    packed = _quantize_f16_q8_0(source)
                    self.raw_weights[source_name] = _upload(self.runtime, packed)
                    self.source_nbytes += int(source.nbytes)
                    self.q8_nbytes += int(packed.nbytes)
                    if slot == "attn_output":
                        output_t16 = repack_gguf_q8_0_tile16(packed).tiles
                        self.output_t16_weights[source_name] = _upload(
                            self.runtime,
                            output_t16,
                        )
                        self.output_t16_nbytes += int(output_t16.nbytes)
                        del output_t16
                    del packed
                print(
                    f"weight-only q8 sidecar layer {layer.layer_id + 1}/48 "
                    f"{self.q8_nbytes / (1024 ** 3):.3f} GiB",
                    flush=True,
                )
        except BaseException:
            self.close()
            raise
        self.build_seconds = time.perf_counter() - started

    def close(self) -> None:
        for allocation in reversed(tuple(self.output_t16_weights.values())):
            free(allocation, runtime=self.runtime)
        self.output_t16_weights.clear()
        for allocation in reversed(tuple(self.raw_weights.values())):
            free(allocation, runtime=self.runtime)
        self.raw_weights.clear()

    def __enter__(self) -> "Q8WeightOnlySidecar":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _mode_owns(mode: str, family: str) -> bool:
    return mode == "all" or mode == family


@contextmanager
def _unfused_source_f16_projection_owner(
    owner: LagunaGGUFResidentSession,
) -> Iterator[None]:
    """Expose source-F16 projection calls without changing their math."""

    names = (
        "use_f16_projection_head_kv_decode",
        "use_f16_attention_quad_decode",
        "use_f16_output_add_rmsnorm_decode",
    )
    original = {name: bool(getattr(owner, name)) for name in names}
    for name in names:
        setattr(owner, name, False)
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(owner, name, value)


@contextmanager
def _q8_decode_owner(
    owner: LagunaGGUFResidentSession,
    sidecar: Q8Sidecar,
    mode: str,
    selected_layers: frozenset[int],
    *,
    q4_library,
    dp4a_library,
) -> Iterator[dict[str, int]]:
    import hipengine.runtime.laguna_gguf_runner as runner

    if mode not in _MODE_CHOICES:
        raise ValueError(f"unsupported Q8 owner mode {mode!r}")
    original_single = runner.launch_f16_weight_linear
    original_triple = runner.launch_f16_weight_linear_triple
    counters = {
        "q8_quantize": 0,
        "q8_triple": 0,
        "q8_gate": 0,
        "q8_output": 0,
        "exact_single": 0,
        "exact_triple": 0,
    }
    reusable_key: tuple[int, int, int] | None = None

    def quantize(x_ptr: int, in_features: int, stream: int) -> None:
        nonlocal reusable_key
        assert sidecar.activation is not None
        gguf_q4_k_quantize_bf16_q8_1(
            x_ptr,
            sidecar.activation.ptr,
            1,
            in_features,
            stream=stream,
            library=q4_library,
            runtime=owner.runtime,
        )
        counters["q8_quantize"] += 1
        reusable_key = (int(x_ptr), int(in_features), int(stream))

    def single(
        weight,
        x_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    ):
        nonlocal reusable_key
        source_name = weight.spec.source.name
        family = _classify_source_name(source_name)
        layer = _layer_id(source_name)
        stream = int(kwargs.get("stream", 0))
        owns_qkv_gate = _mode_owns(mode, "qkv_gate")
        owns_output = _mode_owns(mode, "output")
        if (
            rows == 1
            and layer in selected_layers
            and family == "gate"
            and owns_qkv_gate
        ):
            expected = (int(x_ptr), int(in_features), stream)
            if reusable_key != expected:
                quantize(x_ptr, in_features, stream)
            gguf_q8_0_dp4a_gemv_f32_f32_out(
                sidecar.activation.ptr,
                sidecar.weights[source_name].ptr,
                out_ptr,
                1,
                in_features,
                out_features,
                stream=stream,
                library=dp4a_library,
                runtime=owner.runtime,
            )
            counters["q8_gate"] += 1
            reusable_key = None
            return
        if (
            rows == 1
            and layer in selected_layers
            and family == "output"
            and owns_output
        ):
            quantize(x_ptr, in_features, stream)
            gguf_q8_0_dp4a_gemv_bf16_bf16_out(
                sidecar.activation.ptr,
                sidecar.weights[source_name].ptr,
                out_ptr,
                1,
                in_features,
                out_features,
                stream=stream,
                library=dp4a_library,
                runtime=owner.runtime,
            )
            counters["q8_output"] += 1
            reusable_key = None
            return
        counters["exact_single"] += 1
        reusable_key = None
        original_single(
            weight,
            x_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            **kwargs,
        )

    def triple(
        q_weight,
        k_weight,
        v_weight,
        x_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        rows,
        in_features,
        q_features,
        k_features,
        v_features,
        **kwargs,
    ):
        nonlocal reusable_key
        source_name = q_weight.spec.source.name
        family = _classify_source_name(source_name)
        layer = _layer_id(source_name)
        stream = int(kwargs.get("stream", 0))
        if (
            rows == 1
            and layer in selected_layers
            and family == "qkv"
            and _mode_owns(mode, "qkv_gate")
        ):
            quantize(x_ptr, in_features, stream)
            gguf_q8_0_dp4a_triple_split_rowtile4_gemv_f32_f32_out(
                sidecar.activation.ptr,
                sidecar.weights[q_weight.spec.source.name].ptr,
                sidecar.weights[k_weight.spec.source.name].ptr,
                sidecar.weights[v_weight.spec.source.name].ptr,
                q_ptr,
                k_ptr,
                v_ptr,
                1,
                in_features,
                q_features,
                k_features,
                v_features,
                stream=stream,
                library=dp4a_library,
                runtime=owner.runtime,
            )
            counters["q8_triple"] += 1
            return
        counters["exact_triple"] += 1
        reusable_key = None
        original_triple(
            q_weight,
            k_weight,
            v_weight,
            x_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            rows,
            in_features,
            q_features,
            k_features,
            v_features,
            **kwargs,
        )

    runner.launch_f16_weight_linear = single
    runner.launch_f16_weight_linear_triple = triple
    try:
        with _unfused_source_f16_projection_owner(owner):
            yield counters
    finally:
        runner.launch_f16_weight_linear = original_single
        runner.launch_f16_weight_linear_triple = original_triple


@contextmanager
def _q8_weight_only_decode_owner(
    owner: LagunaGGUFResidentSession,
    sidecar: Q8WeightOnlySidecar,
    mode: str,
    selected_layers: frozenset[int],
    *,
    raw_library,
    t16_library,
) -> Iterator[dict[str, int]]:
    """Replace source-F16 weights while keeping every activation unquantized."""

    import hipengine.runtime.laguna_gguf_runner as runner

    if mode not in _MODE_CHOICES:
        raise ValueError(f"unsupported Q8 owner mode {mode!r}")
    original_single = runner.launch_f16_weight_linear
    original_triple = runner.launch_f16_weight_linear_triple
    counters = {
        "q8_weight_gate": 0,
        "q8_weight_output": 0,
        "q8_weight_qkv": 0,
        "exact_single": 0,
        "exact_triple": 0,
    }

    def single(
        weight,
        x_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    ):
        source_name = weight.spec.source.name
        family = _classify_source_name(source_name)
        layer = _layer_id(source_name)
        stream = int(kwargs.get("stream", 0))
        if (
            rows == 1
            and layer in selected_layers
            and family == "gate"
            and _mode_owns(mode, "qkv_gate")
        ):
            gguf_q8_0_gemv_bf16_f32_out(
                x_ptr,
                sidecar.raw_weights[source_name].ptr,
                out_ptr,
                1,
                in_features,
                out_features,
                stream=stream,
                library=raw_library,
                runtime=owner.runtime,
            )
            counters["q8_weight_gate"] += 1
            return
        if (
            rows == 1
            and layer in selected_layers
            and family == "output"
            and _mode_owns(mode, "output")
        ):
            gguf_q8_0_t16_gemv_decode_f32_bf16_out(
                x_ptr,
                sidecar.output_t16_weights[source_name].ptr,
                out_ptr,
                1,
                in_features,
                out_features,
                stream=stream,
                library=t16_library,
                runtime=owner.runtime,
            )
            counters["q8_weight_output"] += 1
            return
        counters["exact_single"] += 1
        original_single(
            weight,
            x_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            **kwargs,
        )

    def triple(
        q_weight,
        k_weight,
        v_weight,
        x_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        rows,
        in_features,
        q_features,
        k_features,
        v_features,
        **kwargs,
    ):
        source_name = q_weight.spec.source.name
        family = _classify_source_name(source_name)
        layer = _layer_id(source_name)
        stream = int(kwargs.get("stream", 0))
        if (
            rows == 1
            and layer in selected_layers
            and family == "qkv"
            and _mode_owns(mode, "qkv_gate")
        ):
            for weight, target_ptr, features in (
                (q_weight, q_ptr, q_features),
                (k_weight, k_ptr, k_features),
                (v_weight, v_ptr, v_features),
            ):
                gguf_q8_0_gemv_bf16_f32_out(
                    x_ptr,
                    sidecar.raw_weights[weight.spec.source.name].ptr,
                    target_ptr,
                    1,
                    in_features,
                    features,
                    stream=stream,
                    library=raw_library,
                    runtime=owner.runtime,
                )
                counters["q8_weight_qkv"] += 1
            return
        counters["exact_triple"] += 1
        original_triple(
            q_weight,
            k_weight,
            v_weight,
            x_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            rows,
            in_features,
            q_features,
            k_features,
            v_features,
            **kwargs,
        )

    runner.launch_f16_weight_linear = single
    runner.launch_f16_weight_linear_triple = triple
    try:
        with _unfused_source_f16_projection_owner(owner):
            yield counters
    finally:
        runner.launch_f16_weight_linear = original_single
        runner.launch_f16_weight_linear_triple = original_triple


def _copy_logits(
    owner: LagunaGGUFResidentSession,
    result,
) -> np.ndarray:
    assert owner.weights is not None
    logits = np.empty(owner.weights.config.vocab_size, dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(logits),
        result.logits,
        logits.nbytes,
        runtime=owner.runtime,
    )
    return logits


def _exact_reference(
    owner: LagunaGGUFResidentSession,
    prompt: Sequence[int],
    decode_tokens: int,
) -> tuple[list[int], list[int], list[np.ndarray]]:
    owner.reset_state()
    prefill = owner.prefill(prompt)
    inputs: list[int] = []
    predictions: list[int] = []
    logits: list[np.ndarray] = []
    token = int(prefill.next_token_id)
    for _ in range(decode_tokens):
        inputs.append(token)
        result = owner.forward_token(token)
        predictions.append(int(result.next_token_id))
        logits.append(_copy_logits(owner, result))
        token = int(result.next_token_id)
    return inputs, predictions, logits


def _candidate(
    owner: LagunaGGUFResidentSession,
    sidecar: Q8Sidecar | Q8WeightOnlySidecar,
    prompt: Sequence[int],
    teacher_inputs: Sequence[int],
    reference_predictions: Sequence[int],
    reference_logits: Sequence[np.ndarray],
    mode: str,
    scope: str,
    selected_layers: frozenset[int],
    *,
    activation_path: str,
    q4_library=None,
    dp4a_library=None,
    raw_library=None,
    t16_library=None,
    kl_limit: float,
    top1_minimum: float,
) -> dict[str, object]:
    owner.reset_state()
    prefill = owner.prefill(prompt)
    records: list[dict[str, object]] = []
    elapsed_ms: list[float] = []
    if activation_path == "q8_1_dp4a":
        assert isinstance(sidecar, Q8Sidecar)
        decode_owner = _q8_decode_owner(
            owner,
            sidecar,
            mode,
            selected_layers,
            q4_library=q4_library,
            dp4a_library=dp4a_library,
        )
    elif activation_path == "exact":
        assert isinstance(sidecar, Q8WeightOnlySidecar)
        decode_owner = _q8_weight_only_decode_owner(
            owner,
            sidecar,
            mode,
            selected_layers,
            raw_library=raw_library,
            t16_library=t16_library,
        )
    else:
        raise ValueError(f"unsupported activation path {activation_path!r}")
    with decode_owner as counters:
        for step, token in enumerate(teacher_inputs):
            started = time.perf_counter()
            result = owner.forward_token(int(token))
            owner.runtime.device_synchronize()
            elapsed_ms.append((time.perf_counter() - started) * 1000.0)
            candidate_logits = _copy_logits(owner, result)
            candidate_token = int(np.argmax(candidate_logits))
            reference_token = int(reference_predictions[step])
            records.append(
                {
                    "step": step,
                    "input_token": int(token),
                    "reference_token": reference_token,
                    "candidate_token": candidate_token,
                    "top1_match": candidate_token == reference_token,
                    "kl": _kl_divergence(
                        reference_logits[step],
                        candidate_logits,
                    ),
                }
            )
        counts = dict(counters)
    return {
        "mode": mode,
        "scope": scope,
        "activation_path": activation_path,
        "selected_layers": sorted(selected_layers),
        "selected_layer_count": len(selected_layers),
        "prefill_next_token": int(prefill.next_token_id),
        "summary": _summarize_records(
            records,
            kl_limit=kl_limit,
            top1_minimum=top1_minimum,
        ),
        "records": records,
        "decode_wall": {
            "median_ms_per_token": statistics.median(elapsed_ms),
            "mean_ms_per_token": statistics.mean(elapsed_ms),
            "samples_ms": elapsed_ms,
            "performance_claim": False,
        },
        "dispatch_counts": counts,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.decode_tokens <= 0:
        raise ValueError("--decode-tokens must be positive")
    if args.kl_limit < 0.0 or not 0.0 <= args.top1_minimum <= 1.0:
        raise ValueError("quality limits are invalid")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(DEFAULT_PROMPTS, tokenizer)
    prompt, _ = _profile_token_stream(prompts, 512)
    owner = LagunaGGUFResidentSession(
        args.model,
        context_length=512 + args.decode_tokens + 1,
        backend="hip_gfx1151",
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached_build=args.require_cached_build,
        progress=_progress,
        repacked_cache=DEFAULT_CACHE,
        model_sha256=(
            DEFAULT_MODEL_SHA256
            if Path(args.model).resolve() == Path(DEFAULT_MODEL).resolve()
            else None
        ),
        prefill_chunk_size=512,
        prefill_attention_chunk_size=128,
    )
    try:
        teacher_inputs, reference_predictions, reference_logits = _exact_reference(
            owner,
            prompt,
            args.decode_tokens,
        )
        print(
            f"exact reference complete: {args.decode_tokens} transitions, "
            f"last={reference_predictions[-1]}",
            flush=True,
        )
        q4_library = None
        dp4a_library = None
        raw_library = None
        t16_library = None
        if args.activation_path == "q8_1_dp4a":
            q4_library = build_gguf_q4_k_gemv(
                load=True,
                require_cached=args.require_cached_build,
            )
            dp4a_library = build_gguf_q8_0_dp4a_gemv(
                load=True,
                require_cached=args.require_cached_build,
            )
            sidecar_context = Q8Sidecar(owner, reader)
        elif args.activation_path == "exact":
            raw_library = build_gguf_k_gemv(
                load=True,
                require_cached=args.require_cached_build,
            )
            t16_library = build_gguf_q8_0_t16_gemv(
                load=True,
                require_cached=args.require_cached_build,
            )
            sidecar_context = Q8WeightOnlySidecar(owner, reader)
        else:
            raise ValueError(
                f"unsupported activation path {args.activation_path!r}"
            )
        with sidecar_context as sidecar:
            mode_results = []
            assert owner.weights is not None
            attention_types = tuple(
                layer.attention_type for layer in owner.weights.layers
            )
            for scope in args.scopes:
                selected_layers = _scope_layers(scope, attention_types)
                for mode in args.modes:
                    result = _candidate(
                        owner,
                        sidecar,
                        prompt,
                        teacher_inputs,
                        reference_predictions,
                        reference_logits,
                        mode,
                        scope,
                        selected_layers,
                        activation_path=args.activation_path,
                        q4_library=q4_library,
                        dp4a_library=dp4a_library,
                        raw_library=raw_library,
                        t16_library=t16_library,
                        kl_limit=args.kl_limit,
                        top1_minimum=args.top1_minimum,
                    )
                    mode_results.append(result)
                    print(
                        mode,
                        scope,
                        json.dumps(result["summary"], sort_keys=True),
                        flush=True,
                    )
            sidecar_summary = {
                "source_f16_bytes": sidecar.source_nbytes,
                "q8_0_bytes": sidecar.q8_nbytes,
                "byte_change_percent": (
                    sidecar.q8_nbytes / sidecar.source_nbytes - 1.0
                )
                * 100.0,
                "build_seconds": sidecar.build_seconds,
                "weight_count": (
                    len(sidecar.weights)
                    if isinstance(sidecar, Q8Sidecar)
                    else len(sidecar.raw_weights)
                ),
                "activation_path": args.activation_path,
                "output_t16_duplicate_bytes": (
                    sidecar.output_t16_nbytes
                    if isinstance(sidecar, Q8WeightOnlySidecar)
                    else 0
                ),
            }
    finally:
        owner.close()
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_f16_decode_q8_full_model_gate",
        "status": "diagnostic",
        "performance_claim": False,
        "source": {
            "revision": _revision(),
            "model": str(Path(args.model).resolve()),
            "model_sha256": (
                DEFAULT_MODEL_SHA256
                if Path(args.model).resolve() == Path(DEFAULT_MODEL).resolve()
                else None
            ),
        },
        "protocol": {
            "prompt_tokens": 512,
            "decode_tokens": args.decode_tokens,
            "teacher_forced": True,
            "activation_path": args.activation_path,
            "modes": list(args.modes),
            "scopes": list(args.scopes),
            "quality_gate": {
                "maximum_kl": args.kl_limit,
                "top1_agreement": args.top1_minimum,
            },
        },
        "sidecar": sidecar_summary,
        "reference": {
            "teacher_inputs": teacher_inputs,
            "predictions": reference_predictions,
        },
        "modes": mode_results,
    }


def main() -> int:
    args = _parse_args()
    payload = run(args)
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
