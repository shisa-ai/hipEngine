"""Bounded layer-0 comparison: resident TP1 bulk teacher vs TP2 bulk candidate.

Diagnostics only.

Builds the optimized resident TP1 bulk control and the opt-in rank-local bulk
TP2 session in the same process under the same production profile env, feeds
both the identical 64-token prompt, and captures the layer-0 linear-attention
(GDN) primitive chain for both:

  attention norm -> QKV/Z projections -> alpha/beta -> QKV bf16->f32 ->
  convolution -> normalized prepare -> recurrent output -> normalization/gating
  -> output projection (``attn_out``)

Capture method
--------------

The bulk prefill scratch allocates its fields with **liveness aliasing**
(``allocation_mode == "liveness_aliased"``): fields whose
``_GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES`` intervals do not overlap share arena
bytes.  For the layer-0 linear route ``norm``/``linear_qkv``/
``linear_qkv_f32``/``conv_out``/``recurrent_out`` are pairwise stage-disjoint,
so reading them all once after the helper returns yields whichever later
producer last wrote those bytes.  That is why the first comparison could not
localize anything upstream of ``recurrent_bf16``.

This diagnostic therefore:

* derives every capture descriptor from its producer call site -- the live
  ``DeviceBuffer`` pointer/allocation size plus the producer's config-derived
  ``(dtype, width)`` and the helper's ``rows`` argument (never the scratch
  capacity, never a data-derived column mask), validated by
  ``scripts.tp2_layer0_capture``;
* reads each field immediately after its own producer and on the producer's
  stream, before any aliasing writer runs;
* records the allocator's own ``allocation_mode``/``allocation_groups``/
  ``allocation_inplace_aliases``/``allocation_lifetimes`` for the captured
  fields, so aliasing is read off the allocator rather than inferred;
* records the actually-selected kernel/route for each producer (launch log)
  plus the helper arguments, so "same env" is not mistaken for "same route";
* records the exact initial conv/recurrent state hashes and the layer-0
  attention weight hashes for both routes.

Usage::

    HIP_VISIBLE_DEVICES=0,1 HIP_PATH=/opt/rocm python3 \
        scripts/tp2_bulk_vs_resident_layer0.py \
        --prompt mixed_ja_en_translate --json /tmp/tp2_layer0.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tp2_bulk_prefill_diagnostic import MODEL, _build_session  # noqa: E402
from tp2_bulk_prefill_first_divergence import (  # noqa: E402
    _bf16_to_f32,
    _load_prompt,
)
from scripts.tp2_layer0_capture import (  # noqa: E402
    ITEMSIZE,
    aliasing_pairs,
    build_capture_plan,
    describe_layer0_linear,
    descriptor,
    linear_layer0_producers,
)
from scripts.tp2_layer0_mlp_capture import (  # noqa: E402
    BF16,
    PERSISTENT_LIFETIME,
    bf16_round,
    build_mlp_capture_plan,
    resident_mlp_producers,
    resident_mlp_widths,
    shard_mlp_lifetimes,
    shard_mlp_producers_single_rank,
    shard_mlp_widths,
)
from hipengine.core.memory import copy_device_to_host, scoped_current_device  # noqa: E402

_LAYER0_WEIGHTS = (
    "attn_norm",
    "attn_qkv",
    "attn_gate",
    "ssm_alpha",
    "ssm_beta",
    "ssm_conv1d",
    "ssm_dt_bias",
    "ssm_a",
    "ssm_norm",
    "ssm_out",
)

class _RawBuffer:
    """Duck-typed buffer for ``copy_device_to_host`` with a raw pointer."""

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)


def _sha256(raw) -> str:
    if isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    return hashlib.sha256(raw).hexdigest()


def _hex(value) -> str | None:
    if value is None:
        return None
    try:
        return hex(int(value))
    except (TypeError, ValueError):
        return str(value)


def _read_buffer(runtime, device: int, buffer, nbytes: int) -> np.ndarray:
    raw = np.empty(int(nbytes), dtype=np.uint8)
    with scoped_current_device(runtime, device):
        copy_device_to_host(
            raw.ctypes.data,
            _RawBuffer(int(buffer.ptr), int(nbytes)),
            int(nbytes),
            runtime=runtime,
        )
    return raw


class _Layer0Recorder:
    """Per-runner layer-0 capture: producer-derived descriptors + grabs."""

    def __init__(
        self,
        runner,
        scratch,
        decode_scratch,
        rows: int,
        tag: str,
        *,
        mlp_kind: str = "resident",
        per_rank_ffn: int = 0,
    ) -> None:
        self.runner = runner
        self.runtime = runner.runtime
        self.device = int(self.runtime.get_device())
        self.scratch = scratch
        self.decode_scratch = decode_scratch
        self.rows = int(rows)
        self.tag = tag
        self.mlp_kind = str(mlp_kind)
        self.ffn_size = int(getattr(runner, "ffn_size", 0))
        self._per_rank_ffn = int(per_rank_ffn)
        cfg = runner.weights.config
        lifetimes = getattr(scratch, "allocation_lifetimes", None) or None
        if not lifetimes:
            from hipengine.runtime.qwen35_gguf_runner import (
                _GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES,
            )

            lifetimes = _GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES
        self.descriptors = describe_layer0_linear(
            scratch=scratch,
            rows=self.rows,
            hidden_size=int(runner.hidden_size),
            linear_qkv_width=int(runner.linear_qkv_width),
            ssm_inner_size=int(cfg.ssm_inner_size),
            ssm_group_count=int(cfg.ssm_group_count),
            ssm_state_size=int(cfg.ssm_state_size),
            ssm_time_step_rank=int(cfg.ssm_time_step_rank),
            lifetimes=lifetimes,
        )
        self.by_name = {item.name: item for item in self.descriptors}
        self.records: dict[str, np.ndarray] = {}
        self.events: list[dict] = []
        self.launch_log: list[dict] = []
        self.input: np.ndarray | None = None
        self.args: dict = {}
        # Dense-MLP half of the same layer: declared lazily at the producer
        # call site so the width/row count comes from the launch, not from a
        # guess about the scratch layout.
        self.mlp_descriptors: dict[str, object] = {}
        self.mlp: dict[str, np.ndarray] = {}
        self.mlp_events: list[dict] = []

    # -- capture -----------------------------------------------------------

    def _sync(self, stream) -> None:
        with scoped_current_device(self.runtime, self.device):
            if stream is None:
                self.runtime.device_synchronize()
            else:
                self.runtime.stream_synchronize(int(stream))

    def grab(self, names, *, producer: str, kernel: str, stream=None) -> None:
        self.events.append({"producer": producer, "kernel": kernel})
        self._sync(stream)
        for name in names:
            item = self.by_name[name]
            self.records[name] = _read_buffer(
                self.runtime, self.device, item, item.nbytes
            )

    def grab_input(self, hidden_ptr: int) -> None:
        nbytes = self.rows * int(self.runner.hidden_size) * 2
        self.input = _read_buffer(
            self.runtime, self.device, _RawBuffer(int(hidden_ptr), nbytes), nbytes
        )

    def log_launch(self, kernel: str, **fields) -> None:
        entry = {"kernel": kernel}
        entry.update(
            {k: (_hex(v) if k.endswith("_ptr") else v) for k, v in fields.items()}
        )
        self.launch_log.append(entry)

    # -- dense MLP half ----------------------------------------------------

    def declare_mlp(
        self,
        name: str,
        *,
        ptr: int,
        rows: int,
        width: int,
        dtype: str,
        producer: str,
        lifetime: tuple,
        allocated_nbytes: int | None = None,
    ):
        """Register one MLP capture field from its producer call site."""

        item = descriptor(
            name=name,
            producer=producer,
            ptr=int(ptr),
            rows=int(rows),
            width=int(width),
            dtype=dtype,
            allocated_nbytes=(
                int(rows) * int(width) * ITEMSIZE[dtype]
                if allocated_nbytes is None
                else int(allocated_nbytes)
            ),
            lifetime=lifetime,
        )
        self.mlp_descriptors[name] = item
        return item

    def grab_mlp(self, names, *, producer: str, stream=None, host_mapped: bool = False) -> None:
        """Read the named MLP fields immediately after their producer ran."""

        self.mlp_events.append({"producer": producer})
        if not host_mapped:
            self._sync(stream)
        for name in names:
            item = self.mlp_descriptors[name]
            if host_mapped:
                import ctypes

                self.mlp[name] = np.frombuffer(
                    ctypes.string_at(int(item.ptr), item.nbytes), dtype=np.uint8
                ).copy()
            else:
                self.mlp[name] = _read_buffer(
                    self.runtime, self.device, item, item.nbytes
                )

    def declare_norm_residual(self, scratch, *, rows: int) -> None:
        """Declare the shared post-attention norm/residual fields for this rank."""

        widths = (
            resident_mlp_widths(hidden_size=self.hidden_size, ffn_size=self.ffn_size)
            if self.mlp_kind == "resident"
            else shard_mlp_widths(
                hidden_size=self.hidden_size, per_rank_ffn=self.per_rank_ffn
            )
        )
        for name in ("post_norm", "residual"):
            buffer = getattr(scratch, name, None)
            if buffer is None:
                continue
            self.declare_mlp(
                name,
                ptr=int(buffer.ptr),
                rows=int(rows),
                width=widths[name][1],
                dtype=widths[name][0],
                producer="post_norm_residual",
                lifetime=self.mlp_lifetime(name),
                allocated_nbytes=int(getattr(buffer, "nbytes", 0)) or None,
            )

    @property
    def hidden_size(self) -> int:
        return int(self.runner.hidden_size)

    @property
    def per_rank_ffn(self) -> int:
        return int(self._per_rank_ffn)

    def mlp_lifetime(self, name: str) -> tuple:
        table = getattr(self, "mlp_lifetimes", None)
        if table is None:
            table = self.mlp_lifetimes = self._build_mlp_lifetimes()
        return table[name]

    def _build_mlp_lifetimes(self) -> dict:
        if self.mlp_kind == "resident":
            from hipengine.runtime.qwen35_gguf_runner import (
                _GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES,
            )

            table = dict(_GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES)
            return {
                name: table.get(name, PERSISTENT_LIFETIME)
                for name in resident_mlp_widths(
                    hidden_size=self.hidden_size, ffn_size=self.ffn_size
                )
            }
        table = dict(getattr(self.scratch, "allocation_lifetimes", None) or {})
        if not table:
            from hipengine.runtime.qwen35_gguf_runner import (
                _GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES,
            )

            table = dict(_GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES)
        return dict(shard_mlp_lifetimes(scratch_lifetimes=table))

    # -- reporting ---------------------------------------------------------

    def allocation_report(self) -> dict:
        scratch = self.scratch
        groups = getattr(scratch, "allocation_groups", {}) or {}
        offsets = getattr(scratch, "allocation_offsets", {}) or {}
        aliases = getattr(scratch, "allocation_inplace_aliases", {}) or {}
        lifetimes = getattr(scratch, "allocation_lifetimes", {}) or {}
        out = {
            "allocation_mode": getattr(scratch, "allocation_mode", None),
            "scratch_rows": int(getattr(scratch, "rows", -1)),
            "max_positions": int(getattr(scratch, "max_positions", -1)),
            "gdn_effective_mode": getattr(scratch, "gdn_effective_mode", None),
            "fields": {},
        }
        for item in self.descriptors:
            name = item.name
            offset = offsets.get(name)
            out["fields"][name] = {
                "ptr": hex(item.ptr),
                "nbytes": item.nbytes,
                "allocated_nbytes": item.allocated_nbytes,
                "shape": list(item.shape),
                "dtype": item.dtype,
                "arena_group": groups.get(name),
                "arena_offset": None
                if offset is None
                else [int(offset[0]), int(offset[1])],
                "inplace_alias": aliases.get(name),
                "lifetimes": [list(x) for x in lifetimes.get(name, item.lifetime)],
            }
        out["aliasing_pairs"] = [
            list(pair) for pair in aliasing_pairs(self.descriptors, route="linear")
        ]
        out["capture_plan"] = [
            step.to_json()
            for step in build_capture_plan(
                self.descriptors, linear_layer0_producers(), route="linear"
            )
        ]
        return out

    def state_hashes(self) -> dict:
        conv = self.decode_scratch.layer_conv_states[0]
        recurrent = self.decode_scratch.layer_recurrent_states[0]
        out = {}
        for name, buffer in (("conv", conv), ("recurrent", recurrent)):
            if buffer is None:
                out[name] = {"error": "missing layer-0 state"}
                continue
            raw = _read_buffer(self.runtime, self.device, buffer, int(buffer.nbytes))
            out[name] = {
                "sha256": _sha256(raw),
                "nbytes": int(buffer.nbytes),
                "absmax": float(np.abs(raw.view(np.float32)).max()),
            }
        return out

    def to_json(self) -> dict:
        return {
            "tag": self.tag,
            "rows": self.rows,
            "device": self.device,
            "args": self.args,
            "allocation": self.allocation_report(),
            "events": self.events,
            "launch_log": self.launch_log,
            "input_sha256": None if self.input is None else _sha256(self.input),
            "captured_fields": sorted(self.records),
            "mlp_kind": self.mlp_kind,
            "mlp_events": self.mlp_events,
            "mlp_captured_fields": sorted(self.mlp),
            "mlp_capture_plan": [
                step.to_json()
                for step in build_mlp_capture_plan(
                    tuple(self.mlp_descriptors.values()),
                    (
                        resident_mlp_producers()
                        if self.mlp_kind == "resident"
                        else shard_mlp_producers_single_rank()
                    ),
                )
            ],
            "mlp_fields": {
                name: {
                    "ptr": hex(item.ptr),
                    "rows": item.rows,
                    "width": item.width,
                    "dtype": item.dtype,
                    "shape": list(item.shape),
                    "nbytes": item.nbytes,
                    "allocated_nbytes": item.allocated_nbytes,
                    "producer": item.producer,
                    "lifetime": [list(x) for x in item.lifetime],
                    "sha256": None
                    if name not in self.mlp
                    else _sha256(self.mlp[name]),
                }
                for name, item in sorted(self.mlp_descriptors.items())
            },
        }


def _weight_hashes(runner, layer_id: int) -> dict:
    runtime = runner.runtime
    layer = runner.weights.layer(layer_id)
    device = int(runtime.get_device())
    out: dict = {}
    for name in _LAYER0_WEIGHTS:
        try:
            allocation = layer.weight(name).allocation()
        except Exception as exc:  # pragma: no cover - diagnostic only
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        buffer = allocation.buffer
        raw = _read_buffer(runtime, device, buffer, int(buffer.nbytes))
        source = allocation.source
        out[name] = {
            "sha256": _sha256(raw),
            "nbytes": int(buffer.nbytes),
            "source_dtype": str(getattr(source, "dtype", "")),
            "source_shape": list(getattr(source, "shape", ()) or ()),
        }
    return out


def _decode(raw: np.ndarray, item) -> np.ndarray:
    if item.dtype == "f32":
        values = raw.view(np.float32).astype(np.float64)
    else:
        values = _bf16_to_f32(raw)
    return values.reshape(item.shape)


def _stat(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {
            "shape": [list(a.shape), list(b.shape)],
            "rel": float("nan"),
            "max_abs": float("nan"),
        }
    diff = np.abs(a - b)
    scale = max(float(np.abs(b).max()), 1e-9)
    finite = np.isfinite(a) & np.isfinite(b)
    return {
        "shape": list(a.shape),
        "rel": float(diff.max() / scale),
        "max_abs": float(diff.max()),
        "rms": float(np.sqrt(np.mean((a - b) ** 2))),
        "bit_equal": bool(np.array_equal(a, b)),
        "argmax_equal": bool(int(np.argmax(a)) == int(np.argmax(b))),
        "teacher_absmax": float(np.abs(a).max()),
        "candidate_absmax": float(np.abs(b).max()),
        "all_finite": [bool(np.isfinite(a).all()), bool(np.isfinite(b).all())],
        "finite_cells": int(finite.sum()),
        "cells": int(a.size),
    }


def _wrap_out_logger(fn, label, out_index):
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        log = _ACTIVE["log"]
        if log is not None:
            fields: dict = {}
            if isinstance(out_index, int):
                fields["out_ptr"] = (
                    args[out_index] if len(args) > out_index else kwargs.get("out_ptr")
                )
            else:
                for index in out_index:
                    fields[f"out{index}_ptr"] = (
                        args[index] if len(args) > index else None
                    )
            if "rows" in kwargs:
                fields["rows"] = kwargs["rows"]
            log(label, **fields)
        return result

    wrapper.__name__ = getattr(fn, "__name__", label)
    wrapper.__wrapped__ = fn
    return wrapper


_ACTIVE: dict = {"log": None}

# Dense-MLP capture wiring. ``main`` fills the tag tables; the hooks below arm a
# route for exactly one layer (layer 0) so a per-layer launch cannot be
# mistaken for the captured layer.
_MLP_TAGS_BY_DEVICE: dict[int, str] = {}
_MLP_SPECS: dict[str, dict] = {}
_MLP_RECORDERS: dict[str, "_Layer0Recorder"] = {}
_MLP_ARMED: set[str] = set()


def _armed_recorder(kind: str) -> "_Layer0Recorder | None":
    """The armed recorder of one route kind, or None outside the window."""

    for tag in sorted(_MLP_ARMED):
        recorder = _MLP_RECORDERS.get(tag)
        if recorder is not None and recorder.mlp_kind == kind:
            return recorder
    return None


def _armed_recorder_for_device(device: int) -> "_Layer0Recorder | None":
    tag = _MLP_TAGS_BY_DEVICE.get(int(device))
    if tag is None or tag not in _MLP_ARMED:
        return None
    return _MLP_RECORDERS.get(tag)

# Route-scoped GGUF linear dispatch resolution log. ``main`` sets this to a
# list around one route's prefill; ``_install_hooks`` wraps
# ``gguf_linear.resolve`` so every cache-miss resolution records the kernel it
# selected together with the session-scoped context that keyed it. Comparing
# two routes' logs is what shows *which* owner changed *which* leaf, instead
# of inferring it from a shared environment.
_RESOLVE_LOG: list | None = None


def _install_resolve_logger(module) -> None:
    original = getattr(module, "resolve", None)
    if original is None or getattr(original, "__wrapped__", None) is not None:
        return

    def logged_resolve(*args, **kwargs):
        log = _RESOLVE_LOG
        if log is not None:
            entry = dict(kwargs)
            entry["context"] = dict(module.gguf_prefill_dispatch_context())
            log.append(entry)
        return original(*args, **kwargs)

    logged_resolve.__wrapped__ = original
    module.resolve = logged_resolve


def _install_hooks(module, tags: dict, recorders: dict, armed: set) -> None:
    # ``module`` is the runner module; the dispatch resolver lives next door.
    import hipengine.runtime.gguf_linear as gguf_linear

    _install_resolve_logger(gguf_linear)
    """Wrap every layer-0 producer so its output is read on its own stream."""

    runner_cls = module.Qwen35GGUFFullStackRunner

    def _recorder_for(runner):
        if id(runner) not in armed:
            return None
        return recorders.get(id(runner))

    # -- launch log ---------------------------------------------------------

    for name, index in (
        ("launch_gguf_linear_pair", (3, 4)),
        ("launch_gguf_linear_f32_out", 2),
        ("bf16_to_f32", 1),
        ("f32_to_bf16", 1),
        ("_try_launch_dense_q8_pair_dp4a", (3, 4)),
        ("_try_launch_dense_q8_pair_dp4a_f32", (3, 4)),
        ("_try_launch_dense_q8_pair_dp4a_f32_out", (3, 4)),
    ):
        original = getattr(module, name, None)
        if original is not None and not hasattr(original, "__wrapped__"):
            setattr(module, name, _wrap_out_logger(original, name, index))

    # the ssm_out singleton launchers also carry the attn_out grab
    for name in (
        "_try_launch_dense_q8_single_dp4a",
        "_try_launch_dense_q8_single_dp4a_f32",
        "_try_launch_dense_q8_single_dp4a_f32_out",
    ):
        original = getattr(module, name, None)
        if original is None or hasattr(original, "__wrapped__"):
            continue

        def make_single(original, name):
            def wrapper(*args, **kwargs):
                result = original(*args, **kwargs)
                log = _ACTIVE.get("log")
                rec = _ACTIVE.get("recorder")
                if log is not None:
                    log(
                        name,
                        out_ptr=args[2] if len(args) > 2 else None,
                        rows=kwargs.get("rows"),
                    )
                if rec is not None and len(args) > 2:
                    if int(args[2]) == rec.by_name["attn_out"].ptr:
                        rec.grab(
                            ("attn_out",),
                            producer="ssm_out",
                            kernel=name,
                            stream=kwargs.get("stream"),
                        )
                return result

            wrapper.__name__ = name
            wrapper.__wrapped__ = original
            return wrapper

        setattr(module, name, make_single(original, name))

    # -- producer boundaries ------------------------------------------------

    orig_norm = runner_cls._run_attention_norm_rows

    def hooked_norm(self, *, hidden_ptr, weight_ptr, out_ptr, rows, **kwargs):
        result = orig_norm(
            self,
            hidden_ptr=hidden_ptr,
            weight_ptr=weight_ptr,
            out_ptr=out_ptr,
            rows=rows,
            **kwargs,
        )
        rec = _recorder_for(self)
        if rec is not None and int(out_ptr) == rec.by_name["norm"].ptr:
            kernel = getattr(
                module._gguf_norm_residual_decode_kernel(
                    self, layer="rmsnorm", rows=rows, hidden_size=self.hidden_size
                ),
                "__name__",
                "rmsnorm",
            )
            rec.args["hidden_f32_ptr"] = kwargs.get("hidden_f32_ptr")
            rec.args["out_f32_ptr"] = kwargs.get("out_f32_ptr")
            rec.grab(
                ("norm",),
                producer="attn_norm",
                kernel=kernel,
                stream=kwargs.get("stream"),
            )
        return result

    runner_cls._run_attention_norm_rows = hooked_norm

    orig_alpha_beta = runner_cls._run_linear_attention_alpha_beta_rows

    def hooked_alpha_beta(self, layer, norm_ptr, norm_f32_ptr, scratch, *, rows, **kwargs):
        result = orig_alpha_beta(
            self, layer, norm_ptr, norm_f32_ptr, scratch, rows=rows, **kwargs
        )
        rec = _recorder_for(self)
        if rec is not None and int(norm_ptr) == rec.by_name["norm"].ptr:
            rec.args["norm_f32_ptr"] = norm_f32_ptr
            # the QKV/gate projections completed just before this call
            rec.grab(
                ("linear_qkv", "linear_z"),
                producer="qkv_gate",
                kernel="see launch_log",
                stream=kwargs.get("stream"),
            )
            rec.grab(
                ("linear_alpha", "linear_beta"),
                producer="alpha_beta",
                kernel=str(result),
                stream=kwargs.get("stream"),
            )
        return result

    runner_cls._run_linear_attention_alpha_beta_rows = hooked_alpha_beta

    orig_conv_kernel = runner_cls._linear_attn_conv_prefill_kernel

    def hooked_conv_kernel(self):
        kernel = orig_conv_kernel(self)
        label = getattr(kernel, "__name__", repr(kernel))

        def wrapper(*args, **kwargs):
            rec = _recorder_for(self)
            if rec is not None:
                # the bf16->f32 cast of linear_qkv ran immediately before
                rec.grab(
                    ("linear_qkv_f32",),
                    producer="qkv_bf16_to_f32",
                    kernel="bf16_to_f32",
                    stream=kwargs.get("stream"),
                )
            result = kernel(*args, **kwargs)
            if rec is not None:
                rec.grab(
                    ("conv_out",),
                    producer="conv_prefill",
                    kernel=label,
                    stream=kwargs.get("stream"),
                )
            return result

        return wrapper

    runner_cls._linear_attn_conv_prefill_kernel = hooked_conv_kernel

    orig_gdn = runner_cls._run_gdn_prefill

    def hooked_gdn(self, *, layer, scratch, cfg, rows, recurrent_state, stream, runtime):
        rec = _recorder_for(self)
        if rec is None:
            return orig_gdn(
                self,
                layer=layer,
                scratch=scratch,
                cfg=cfg,
                rows=rows,
                recurrent_state=recurrent_state,
                stream=stream,
                runtime=runtime,
            )
        plan = self._gdn_prefill_plan()
        cached = getattr(self, "_gguf_gdn_prefill_plan_cache", None)
        rec.args["gdn_mode"] = self._gdn_prefill_mode_for_report()

        def wrap_prepare(original):
            if original is None:
                return None

            def inner(*args, **kwargs):
                result = original(*args, **kwargs)
                rec.grab(
                    (
                        "prefill_query",
                        "prefill_key",
                        "prefill_value",
                        "prefill_beta",
                        "prefill_decay",
                    ),
                    producer="gdn_prepare",
                    kernel=getattr(original, "__name__", repr(original)),
                    stream=kwargs.get("stream"),
                )
                return result

            return inner

        def wrap_recurrent(original):
            if original is None:
                return None

            def inner(*args, **kwargs):
                result = original(*args, **kwargs)
                rec.log_launch(
                    getattr(original, "__name__", repr(original)),
                    out_ptr=args[6] if len(args) > 6 else None,
                    rows=args[7] if len(args) > 7 else None,
                )
                return result

            return inner

        def wrap_gate(original):
            if original is None:
                return None

            def inner(*args, **kwargs):
                rec.grab(
                    ("recurrent_out",),
                    producer="gdn_recurrent",
                    kernel="see launch_log",
                    stream=kwargs.get("stream"),
                )
                result = original(*args, **kwargs)
                rec.grab(
                    ("recurrent_bf16",),
                    producer="gdn_rmsnorm_gate",
                    kernel=getattr(original, "__name__", repr(original)),
                    stream=kwargs.get("stream"),
                )
                return result

            return inner

        self._gguf_gdn_prefill_plan_cache = dataclasses.replace(
            plan,
            prepare_compact_peer_normalized=wrap_prepare(
                plan.prepare_compact_peer_normalized
            ),
            recurrent_compact_peer_wave32=wrap_recurrent(
                plan.recurrent_compact_peer_wave32
            ),
            rmsnorm_gate=wrap_gate(plan.rmsnorm_gate),
        )
        try:
            return orig_gdn(
                self,
                layer=layer,
                scratch=scratch,
                cfg=cfg,
                rows=rows,
                recurrent_state=recurrent_state,
                stream=stream,
                runtime=runtime,
            )
        finally:
            self._gguf_gdn_prefill_plan_cache = cached

    runner_cls._run_gdn_prefill = hooked_gdn

    # the fallback ssm_out route when the q8 single kernel declines
    orig_launch_linear = module.launch_gguf_linear

    def hooked_launch_linear(*args, **kwargs):
        result = orig_launch_linear(*args, **kwargs)
        rec = _ACTIVE.get("recorder")
        log = _ACTIVE.get("log")
        if log is not None:
            log(
                "launch_gguf_linear",
                out_ptr=args[2] if len(args) > 2 else kwargs.get("out_ptr"),
                rows=kwargs.get("rows"),
            )
        if rec is not None and len(args) > 2:
            if int(args[2]) == rec.by_name["attn_out"].ptr:
                rec.grab(
                    ("attn_out",),
                    producer="ssm_out",
                    kernel="launch_gguf_linear",
                    stream=kwargs.get("stream"),
                )
        return result

    hooked_launch_linear.__wrapped__ = orig_launch_linear
    module.launch_gguf_linear = hooked_launch_linear

    # -- top-level helper ---------------------------------------------------

    orig_helper = runner_cls._run_linear_attention_prefill_attn_rows

    def hooked_helper(self, layer_id, hidden_ptr, scratch, *, rows, decode_scratch, **kwargs):
        tag = tags.get(id(self))
        if tag is None or int(layer_id) != 0:
            return orig_helper(
                self,
                layer_id,
                hidden_ptr,
                scratch,
                rows=rows,
                decode_scratch=decode_scratch,
                **kwargs,
            )
        recorder = _Layer0Recorder(
            self,
            scratch,
            decode_scratch,
            rows,
            tag,
            mlp_kind=str(_MLP_SPECS.get(tag, {}).get("kind", "resident")),
            per_rank_ffn=int(_MLP_SPECS.get(tag, {}).get("per_rank_ffn", 0)),
        )
        recorders[id(self)] = recorder
        _MLP_RECORDERS[tag] = recorder
        _MLP_TAGS_BY_DEVICE[recorder.device] = tag
        armed.add(id(self))
        previous_log = _ACTIVE.get("log")
        previous_rec = _ACTIVE.get("recorder")
        _ACTIVE["log"] = recorder.log_launch
        _ACTIVE["recorder"] = recorder
        try:
            recorder.grab_input(int(hidden_ptr))
            return orig_helper(
                self,
                layer_id,
                hidden_ptr,
                scratch,
                rows=rows,
                decode_scratch=decode_scratch,
                **kwargs,
            )
        finally:
            _ACTIVE["log"] = previous_log
            _ACTIVE["recorder"] = previous_rec
            armed.discard(id(self))

    runner_cls._run_linear_attention_prefill_attn_rows = hooked_helper


    # -- dense-MLP half (layer 0) -------------------------------------------

    import hipengine.distributed.shard_exec as shard_exec
    import hipengine.distributed.shard_group as shard_group
    import hipengine.distributed.tp2_generate as tp2_generate

    orig_norm_residual = runner_cls._run_post_attention_norm_residual_rows

    def hooked_norm_residual(self, layer_id, hidden_ptr, attn_out_ptr, scratch, *, rows, **kwargs):
        result = orig_norm_residual(
            self, layer_id, hidden_ptr, attn_out_ptr, scratch, rows=rows, **kwargs
        )
        device = int(self.runtime.get_device())
        tag = _MLP_TAGS_BY_DEVICE.get(device)
        if tag is None:
            return result
        if int(layer_id) != 0:
            _MLP_ARMED.discard(tag)
            return result
        _MLP_ARMED.add(tag)
        recorder = _MLP_RECORDERS.get(tag)
        if recorder is None:
            return result
        recorder.declare_norm_residual(scratch, rows=rows)
        recorder.grab_mlp(
            ("post_norm", "residual"),
            producer="post_norm_residual",
            stream=kwargs.get("stream"),
        )
        return result

    runner_cls._run_post_attention_norm_residual_rows = hooked_norm_residual

    # Teacher: the fused gate/up+SiLU writes only the activation, and the
    # down+residual fusion fails closed for T16 layouts, so the resident layer
    # materializes ``ffn_intermediate``, then ``ffn_down`` and its own add.
    orig_pair_silu = getattr(module, "launch_gguf_linear_pair_silu", None)
    if orig_pair_silu is not None and not hasattr(orig_pair_silu, "__wrapped__"):

        def hooked_pair_silu(*args, **kwargs):
            result = orig_pair_silu(*args, **kwargs)
            recorder = _armed_recorder("resident")
            if result and recorder is not None:
                rows = int(kwargs.get("rows", args[4] if len(args) > 4 else 0))
                out_ptr = int(kwargs.get("out_ptr", args[3] if len(args) > 3 else 0))
                recorder.declare_mlp(
                    "ffn_intermediate",
                    ptr=out_ptr,
                    rows=rows,
                    width=recorder.ffn_size,
                    dtype=BF16,
                    producer="gate_up_silu",
                    lifetime=recorder.mlp_lifetime("ffn_intermediate"),
                )
                recorder.grab_mlp(
                    ("ffn_intermediate",),
                    producer="gate_up_silu",
                    stream=kwargs.get("stream"),
                )
            return result

        hooked_pair_silu.__wrapped__ = orig_pair_silu
        module.launch_gguf_linear_pair_silu = hooked_pair_silu

    orig_ffn_rows = runner_cls._run_post_attention_ffn_rows

    def hooked_ffn_rows(self, layer_id, hidden_ptr, attn_out_ptr, out_ptr, scratch, *, rows, **kwargs):
        result = orig_ffn_rows(
            self,
            layer_id,
            hidden_ptr,
            attn_out_ptr,
            out_ptr,
            scratch,
            rows=rows,
            **kwargs,
        )
        device = int(self.runtime.get_device())
        tag = _MLP_TAGS_BY_DEVICE.get(device)
        if tag is None or tag not in _MLP_ARMED or int(layer_id) != 0:
            return result
        recorder = _MLP_RECORDERS.get(tag)
        if recorder is None or recorder.mlp_kind != "resident":
            return result
        # The down projection and the residual add both ran inside the helper;
        # nothing else writes either plane until layer 1's norm+residual.
        for name, ptr in (
            ("ffn_down", int(scratch.ffn_down.ptr)),
            ("out", int(out_ptr)),
        ):
            recorder.declare_mlp(
                name,
                ptr=ptr,
                rows=rows,
                width=recorder.hidden_size,
                dtype=BF16,
                producer="down" if name == "ffn_down" else "residual_add",
                lifetime=recorder.mlp_lifetime(name),
                allocated_nbytes=int(getattr(getattr(scratch, name, None), "nbytes", 0))
                or None,
            )
        recorder.grab_mlp(("ffn_down", "out"), producer="down_residual", stream=kwargs.get("stream"))
        return result

    runner_cls._run_post_attention_ffn_rows = hooked_ffn_rows

    # Candidate: the rank's own chain, the exchange's f32 payload, the bf16
    # boundary buffer, and the layer output are all rank-local allocations.
    orig_forward_partial = shard_exec.MlpShardRank.forward_partial

    def hooked_forward_partial(self, *, rows=None, **kwargs):
        result = orig_forward_partial(self, rows=rows, **kwargs)
        recorder = _armed_recorder_for_device(self.device)
        if recorder is None or recorder.mlp_kind != "shard":
            return result
        widths = shard_mlp_widths(
            hidden_size=recorder.hidden_size, per_rank_ffn=recorder.per_rank_ffn
        )
        active = int(rows if rows is not None else self.active_rows)
        for name, ptr in (
            ("gate", int(self.gate_ptr)),
            ("up", int(self.up_ptr)),
            ("act", int(self.act_ptr)),
            ("down_partial", int(self.down_partial_ptr)),
        ):
            recorder.declare_mlp(
                name,
                ptr=ptr,
                rows=active,
                width=widths[name][1],
                dtype=widths[name][0],
                producer={
                    "gate": "shard_gate",
                    "up": "shard_up",
                    "act": "shard_silu",
                    "down_partial": "shard_down",
                }[name],
                lifetime=recorder.mlp_lifetime(name),
            )
        for producer, names in (
            ("shard_gate", ("gate",)),
            ("shard_up", ("up",)),
            ("shard_silu", ("act",)),
            ("shard_down", ("down_partial",)),
        ):
            recorder.grab_mlp(names, producer=producer, stream=self.stream)
        return result

    shard_exec.MlpShardRank.forward_partial = hooked_forward_partial

    orig_cast_reduced = shard_group.MlpShardGroup.cast_reduced

    def hooked_cast_reduced(self, device, reduced_ptr):
        result = orig_cast_reduced(self, device, reduced_ptr)
        recorder = _armed_recorder_for_device(device)
        if recorder is None or recorder.mlp_kind != "shard":
            return result
        widths = shard_mlp_widths(
            hidden_size=recorder.hidden_size, per_rank_ffn=recorder.per_rank_ffn
        )
        active = int(recorder.rows)
        recorder.declare_mlp(
            "reduced",
            ptr=int(reduced_ptr),
            rows=active,
            width=widths["reduced"][1],
            dtype=widths["reduced"][0],
            producer="staged_reduce",
            lifetime=recorder.mlp_lifetime("reduced"),
        )
        recorder.declare_mlp(
            "cast",
            ptr=int(self._out_ptrs[int(device)]),
            rows=active,
            width=widths["cast"][1],
            dtype=widths["cast"][0],
            producer="cast_reduced",
            lifetime=recorder.mlp_lifetime("cast"),
        )
        # The Python driver publishes the reduced row into a device buffer;
        # the compiled driver returns the *mapped pinned* host address of its
        # fixed payload slot, which has to be read as host memory.
        host_mapped = hasattr(self._transport, "payload_ptr")
        recorder.grab_mlp(
            ("reduced",),
            producer="staged_reduce",
            stream=None if host_mapped else self._streams[int(device)],
            host_mapped=host_mapped,
        )
        recorder.grab_mlp(
            ("cast",), producer="cast_reduced", stream=self._streams[int(device)]
        )
        return result

    shard_group.MlpShardGroup.cast_reduced = hooked_cast_reduced

    orig_sharded_mlp_layer = tp2_generate.MlpTP2GenerationSession._bulk_sharded_mlp_layer

    def hooked_sharded_mlp_layer(self, layer_id, src, dst, rows):
        result = orig_sharded_mlp_layer(self, layer_id, src, dst, rows)
        if int(layer_id) != 0:
            return result
        for device in self.devices:
            recorder = _armed_recorder_for_device(device)
            if recorder is None or recorder.mlp_kind != "shard":
                continue
            widths = shard_mlp_widths(
                hidden_size=recorder.hidden_size, per_rank_ffn=recorder.per_rank_ffn
            )
            recorder.declare_mlp(
                "out",
                ptr=int(dst[device]),
                rows=int(rows),
                width=widths["out"][1],
                dtype=widths["out"][0],
                producer="residual_add",
                lifetime=recorder.mlp_lifetime("out"),
            )
            recorder.grab_mlp(
                ("out",), producer="residual_add", stream=self._rank_stream(device)
            )
        return result

    tp2_generate.MlpTP2GenerationSession._bulk_sharded_mlp_layer = hooked_sharded_mlp_layer


def _summarize_resolve_log(entries: list) -> dict:
    """Group one route's dispatch resolutions by the owner that keyed them."""

    out: dict = {"resolutions": len(entries), "variants": {}, "contexts": []}
    seen_contexts: list[dict] = []
    for entry in entries:
        key = f"{entry.get('layer')}:{entry.get('quant')}:{entry.get('variant')}"
        out["variants"][key] = out["variants"].get(key, 0) + 1
        context = entry.get("context")
        if context is not None and context not in seen_contexts:
            seen_contexts.append(context)
    out["contexts"] = seen_contexts
    return out


def _diff_resolve_logs(lhs: list, rhs: list) -> dict:
    """Which leaf variants the two routes selected for the same weight shape."""

    def index(entries: list) -> dict:
        out: dict[str, set] = {}
        for entry in entries:
            key = f"{entry.get('layer')}:{entry.get('quant')}"
            out.setdefault(key, set()).add(str(entry.get("variant")))
        return out

    left = index(lhs)
    right = index(rhs)
    only_left = {k: sorted(v) for k, v in left.items() if k not in right}
    only_right = {k: sorted(v) for k, v in right.items() if k not in left}
    differing = {
        k: {"teacher": sorted(left[k]), "candidate": sorted(right[k])}
        for k in sorted(set(left) & set(right))
        if left[k] != right[k]
    }
    return {
        "only_teacher": only_left,
        "only_candidate": only_right,
        "differing_variants": differing,
    }


def _install_gdn_mode_reporter(module, runner_cls) -> None:
    if hasattr(runner_cls, "_gdn_prefill_mode_for_report"):
        return

    def _mode_for_report(self):
        try:
            return module._gguf_gdn_prefill_session_mode(
                self.backend, weights=self.weights, cfg=self.weights.config
            )
        except Exception as exc:  # pragma: no cover - diagnostic only
            return f"{type(exc).__name__}: {exc}"

    runner_cls._gdn_prefill_mode_for_report = _mode_for_report


def _initial_state_hashes(runtime, device: int, decode_scratch) -> dict:
    out = {}
    for name, holder in (
        ("conv", decode_scratch.layer_conv_states[0]),
        ("recurrent", decode_scratch.layer_recurrent_states[0]),
    ):
        if holder is None:
            out[name] = {"error": "missing layer-0 state"}
            continue
        raw = _read_buffer(runtime, device, holder, int(holder.nbytes))
        out[name] = {
            "sha256": _sha256(raw),
            "nbytes": int(holder.nbytes),
            "absmax": float(np.abs(raw.view(np.float32)).max()),
        }
    return out


def _mlp_values(recorder, name: str) -> np.ndarray:
    item = recorder.mlp_descriptors[name]
    return _decode(recorder.mlp[name], item)


def _mlp_has(recorder, name: str) -> bool:
    return recorder is not None and name in recorder.mlp


def _bf16_ulp_distance(a: np.ndarray, b: np.ndarray) -> dict:
    """How far apart two bf16-valued arrays are, measured in bf16 ULPs."""

    a_bits = np.asarray(a, dtype=np.float32).astype(np.float32).view(np.uint32)
    b_bits = np.asarray(b, dtype=np.float32).astype(np.float32).view(np.uint32)
    # Both sides are already bf16-representable, so the low 16 bits are zero;
    # the high halves are the bf16 bit patterns and their integer distance is
    # the ULP distance for same-sign values.
    a_bf16 = (a_bits >> np.uint32(16)).astype(np.int64)
    b_bf16 = (b_bits >> np.uint32(16)).astype(np.int64)
    distance = np.abs(a_bf16 - b_bf16)
    return {
        "cells": int(distance.size),
        "max_ulp": int(distance.max()) if distance.size else 0,
        "nonzero_cells": int((distance != 0).sum()),
        "gt_one_ulp_cells": int((distance > 1).sum()),
    }


def _compare_mlp(captured: dict) -> dict:
    """Compare the layer-0 dense-MLP half between the resident and bulk routes.

    Everything here is a *schedule-internal* check plus the cross-route
    boundary comparison; the independent FP32/quantized reference lives in
    ``scripts/tp2_layer0_mlp_reference_check.py``, which consumes the saved
    capture instead of re-running the model.
    """

    teacher = captured.get("teacher")
    ranks = sorted(tag for tag in captured if tag.startswith("candidate_"))
    out: dict = {
        "teacher_fields": sorted(getattr(teacher, "mlp", {}) or {}),
        "rank_fields": {tag: sorted(captured[tag].mlp) for tag in ranks},
        "teacher_vs_rank": {},
        "activation_reconstruction": {},
        "exactness": {},
        "earliest_divergence": {},
    }
    if teacher is None or not ranks:
        return out

    per_rank_ffn = int(captured[ranks[0]].per_rank_ffn)
    if not per_rank_ffn:
        raise RuntimeError("the shard recorders were built without a per_rank_ffn")

    for tag in ranks:
        recorder = captured[tag]
        block: dict = {}
        for name in ("post_norm", "residual", "out"):
            if _mlp_has(teacher, name) and _mlp_has(recorder, name):
                block[name] = _stat(
                    _mlp_values(teacher, name), _mlp_values(recorder, name)
                )
        if _mlp_has(teacher, "ffn_down") and _mlp_has(recorder, "reduced"):
            block["reduced_vs_teacher_ffn_down"] = _stat(
                _mlp_values(recorder, "reduced"), _mlp_values(teacher, "ffn_down")
            )
        if _mlp_has(teacher, "ffn_down") and _mlp_has(recorder, "cast"):
            block["cast_vs_teacher_ffn_down"] = _stat(
                _mlp_values(recorder, "cast"), _mlp_values(teacher, "ffn_down")
            )
            block["cast_vs_teacher_ffn_down_ulp"] = _bf16_ulp_distance(
                _mlp_values(recorder, "cast"), _mlp_values(teacher, "ffn_down")
            )
        out["teacher_vs_rank"][tag] = block

    # The activation is split on the output-feature axis, so concatenating the
    # rank activations has to reproduce the full-width activation exactly.
    if _mlp_has(teacher, "ffn_intermediate") and all(
        _mlp_has(captured[tag], "act") for tag in ranks
    ):
        full = _mlp_values(teacher, "ffn_intermediate")
        parts = [_mlp_values(captured[tag], "act") for tag in ranks]
        rebuilt = np.concatenate(parts, axis=1)
        block = _stat(rebuilt, full)
        for index, tag in enumerate(ranks):
            start = index * per_rank_ffn
            block[f"slice_{tag}"] = _stat(
                parts[index], full[:, start : start + per_rank_ffn]
            )
        out["activation_reconstruction"] = block

    # Schedule-internal exactness: the exchange must not round, the cast must be
    # exactly the bf16 rounding of the reduced payload, and the residual add
    # must be exactly the bf16 sum of residual and the cast output.
    exactness: dict = {}
    if len(ranks) == 2 and all(_mlp_has(captured[t], "down_partial") for t in ranks):
        staged_sum = np.zeros_like(_mlp_values(captured[ranks[0]], "down_partial"))
        for tag in ranks:
            staged_sum = staged_sum + _mlp_values(captured[tag], "down_partial")
        for tag in ranks:
            recorder = captured[tag]
            entry: dict = {}
            if _mlp_has(recorder, "reduced"):
                entry["reduced_is_the_f32_sum_of_staged_partials"] = _stat(
                    _mlp_values(recorder, "reduced"), staged_sum
                )
            if _mlp_has(recorder, "cast") and _mlp_has(recorder, "reduced"):
                host_cast = bf16_round(_mlp_values(recorder, "reduced").astype(np.float32))
                entry["cast_is_bf16_round_of_reduced"] = _stat(
                    _mlp_values(recorder, "cast"), host_cast.astype(np.float64)
                )
            if (
                _mlp_has(recorder, "out")
                and _mlp_has(recorder, "residual")
                and _mlp_has(recorder, "cast")
            ):
                host_add = bf16_round(
                    (
                        _mlp_values(recorder, "residual")
                        + _mlp_values(recorder, "cast")
                    ).astype(np.float32)
                )
                entry["out_is_bf16_residual_plus_cast"] = _stat(
                    _mlp_values(recorder, "out"), host_add.astype(np.float64)
                )
            exactness[tag] = entry
    out["exactness"] = exactness

    for name in ("post_norm", "residual", "ffn_intermediate"):
        stat = out["teacher_vs_rank"].get(ranks[0], {}).get(name)
        if stat is not None and not stat.get("bit_equal", False):
            out["earliest_divergence"] = {"field": name, "stat": stat}
            break
    if not out["earliest_divergence"]:
        reconstruction = out["activation_reconstruction"]
        if reconstruction and not reconstruction.get("bit_equal", False):
            out["earliest_divergence"] = {
                "field": "activation_reconstruction",
                "stat": reconstruction,
            }
    if not out["earliest_divergence"]:
        for tag in ranks:
            stat = out["teacher_vs_rank"].get(tag, {}).get("cast_vs_teacher_ffn_down")
            if stat is not None:
                out["earliest_divergence"] = {
                    "field": f"cast_vs_teacher_ffn_down ({tag})",
                    "stat": stat,
                }
                break
    return out


def main(argv: list[str] | None = None) -> int:
    global _RESOLVE_LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="mixed_ja_en_translate")
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--json", required=True)
    parser.add_argument("--save-dir", default="")
    args = parser.parse_args(argv)

    prompt = _load_prompt(args.prompt)
    result: dict = {
        "kind": "tp2_bulk_vs_resident_layer0",
        "model": MODEL,
        "prompt": args.prompt,
        "prompt_tokens": len(prompt),
        "host": platform.node(),
        "command": " ".join(sys.argv),
        "protocol": (
            "resident TP1 bulk vs TP2 bulk, layer-0 GDN primitive chain, "
            "producer-derived descriptors captured immediately after each "
            "producer on its owning stream"
        ),
    }

    from scripts.tp2_resident_control import (
        bind_resident_profile,
        create_resident_control,
    )
    import hipengine.runtime.qwen35_gguf_runner as qr
    import hipengine.runtime.gguf_linear as gguf_linear

    if not hasattr(qr, "Qwen35GGUFFullStackRunner"):
        raise RuntimeError(
            "hipengine.runtime.qwen35_gguf_runner is missing "
            "Qwen35GGUFFullStackRunner; refusing to load two models before failing"
        )
    for attribute in (
        "_run_attention_norm_rows",
        "_run_linear_attention_alpha_beta_rows",
        "_linear_attn_conv_prefill_kernel",
        "_run_gdn_prefill",
        "_run_linear_attention_prefill_attn_rows",
    ):
        if not hasattr(qr.Qwen35GGUFFullStackRunner, attribute):
            raise RuntimeError(f"Qwen35GGUFFullStackRunner is missing {attribute}")
    for attribute in (
        "launch_gguf_linear",
        "_try_launch_dense_q8_single_dp4a",
        "_try_launch_dense_q8_pair_dp4a",
        "bf16_to_f32",
        "f32_to_bf16",
    ):
        if not hasattr(qr, attribute):
            raise RuntimeError(f"qwen35_gguf_runner is missing module-level {attribute}")
    for attribute in (
        "resolve",
        "clear_gguf_linear_dispatch_cache",
        "gguf_prefill_dispatch_context",
    ):
        if not hasattr(gguf_linear, attribute):
            raise RuntimeError(f"gguf_linear is missing module-level {attribute}")

    result["profile"] = bind_resident_profile("production")
    result["env"] = {k: v for k, v in os.environ.items() if k.startswith("HIPENGINE_")}

    resident = None
    tp2 = None
    recorders: dict[int, _Layer0Recorder] = {}
    armed: set[int] = set()
    captured: dict[str, _Layer0Recorder] = {}
    initial_states: dict[str, dict] = {}
    weight_hashes: dict[str, dict] = {}
    resolve_logs: dict[str, list] = {}
    started = time.perf_counter()
    try:
        resident = create_resident_control(MODEL, capacity=args.capacity, capture_rows=False)
        tp2 = _build_session(args.capacity, rows=args.capacity, schedule="graphed", bulk=True)

        tags = {id(resident.session.runner): "teacher"}
        for device, runner in tp2._runners.items():
            tags[id(runner)] = f"candidate_{device}"

        _MLP_SPECS.clear()
        _MLP_SPECS["teacher"] = {"kind": "resident", "per_rank_ffn": 0}
        for device in tp2._runners:
            _MLP_SPECS[f"candidate_{device}"] = {
                "kind": "shard",
                "per_rank_ffn": int(tp2._per_rank_ffn),
            }
        _MLP_RECORDERS.clear()
        _MLP_TAGS_BY_DEVICE.clear()
        _MLP_ARMED.clear()

        _install_gdn_mode_reporter(qr, qr.Qwen35GGUFFullStackRunner)
        _install_hooks(qr, tags, recorders, armed)

        # Teacher: optimized resident TP1 bulk prefill, exactly the capture path.
        resident.session.reset()
        initial_states["teacher"] = _initial_state_hashes(
            resident.session.runner.runtime,
            int(resident.session.runner.runtime.get_device()),
            resident.session.scratch,
        )
        weight_hashes["teacher"] = _weight_hashes(resident.session.runner, 0)
        gguf_linear.clear_gguf_linear_dispatch_cache()
        _RESOLVE_LOG = []
        try:
            resident.session.prefill(
                prompt, use_bulk=None, bulk_attention_mode="bulk", return_logits=False
            )
        finally:
            resolve_logs["teacher"] = _RESOLVE_LOG
            _RESOLVE_LOG = None
        captured["teacher"] = recorders[id(resident.session.runner)]

        # Candidate: opt-in rank-local bulk TP2 prefill on the same prompt.
        tp2.reset()
        for device, runner in tp2._runners.items():
            initial_states[f"candidate_{device}"] = _initial_state_hashes(
                runner.runtime, int(runner.runtime.get_device()), tp2._scratches[device]
            )
            weight_hashes[f"candidate_{device}"] = _weight_hashes(runner, 0)
        gguf_linear.clear_gguf_linear_dispatch_cache()
        _RESOLVE_LOG = []
        try:
            tp2.bulk_prefill(prompt)
        finally:
            resolve_logs["candidate_0"] = _RESOLVE_LOG
            _RESOLVE_LOG = None
        for device, runner in tp2._runners.items():
            captured[f"candidate_{device}"] = recorders[id(runner)]

        result["routes"] = {tag: rec.to_json() for tag, rec in captured.items()}
        result["resolve_log"] = {
            tag: _summarize_resolve_log(entries) for tag, entries in resolve_logs.items()
        }
        result["resolve_log_diff"] = _diff_resolve_logs(
            resolve_logs.get("teacher", []), resolve_logs.get("candidate_0", [])
        )
        result["initial_state_hashes"] = initial_states
        result["weight_hashes"] = weight_hashes

        def compare(lhs: str, rhs: str) -> dict:
            a = captured[lhs]
            b = captured[rhs]
            out: dict = {
                "input": _stat(
                    _bf16_to_f32(a.input).reshape(a.rows, a.by_name["norm"].width),
                    _bf16_to_f32(b.input).reshape(b.rows, b.by_name["norm"].width),
                ),
                "input_sha256": [a.to_json()["input_sha256"], b.to_json()["input_sha256"]],
                "fields": {},
            }
            for name in _PRODUCER_ORDER_BUFFERS:
                if name not in a.records or name not in b.records:
                    continue
                out["fields"][name] = _stat(
                    _decode(a.records[name], a.by_name[name]),
                    _decode(b.records[name], b.by_name[name]),
                )
            return out

        result["comparison"] = {
            "teacher_vs_candidate_0": compare("teacher", "candidate_0"),
            "candidate_0_vs_candidate_1": compare("candidate_0", "candidate_1"),
        }
        result["mlp_comparison"] = _compare_mlp(captured)

        def earliest(block: dict) -> dict:
            for name in _PRODUCER_ORDER_BUFFERS:
                stat = block["fields"].get(name)
                if stat is None:
                    continue
                if not stat.get("bit_equal", False):
                    return {"field": name, "stat": stat}
            return {}

        result["earliest_divergence"] = {
            "teacher_vs_candidate_0": earliest(result["comparison"]["teacher_vs_candidate_0"]),
        }
        result["earliest_mlp_divergence"] = result["mlp_comparison"].get(
            "earliest_divergence", {}
        )

        if args.save_dir:
            out = Path(args.save_dir)
            out.mkdir(parents=True, exist_ok=True)
            payload = {}
            for tag, rec in captured.items():
                for name, raw in rec.records.items():
                    payload[f"{tag}.{name}"] = raw
                if rec.input is not None:
                    payload[f"{tag}.input"] = rec.input
                for name, raw in rec.mlp.items():
                    payload[f"{tag}.mlp.{name}"] = raw
            np.savez(out / f"{args.prompt}.layer0.npz", **payload)
    finally:
        if resident is not None:
            resident.close()
        if tp2 is not None:
            tp2.close()

    result["seconds"] = time.perf_counter() - started
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


_PRODUCER_ORDER_BUFFERS = (
    "norm",
    "linear_qkv",
    "linear_z",
    "linear_alpha",
    "linear_beta",
    "linear_qkv_f32",
    "conv_out",
    "prefill_query",
    "prefill_key",
    "prefill_value",
    "prefill_beta",
    "prefill_decay",
    "recurrent_out",
    "recurrent_bf16",
    "attn_out",
)


if __name__ == "__main__":
    raise SystemExit(main())
