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
    aliasing_pairs,
    build_capture_plan,
    describe_layer0_linear,
    linear_layer0_producers,
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

    def __init__(self, runner, scratch, decode_scratch, rows: int, tag: str) -> None:
        self.runner = runner
        self.runtime = runner.runtime
        self.device = int(self.runtime.get_device())
        self.scratch = scratch
        self.decode_scratch = decode_scratch
        self.rows = int(rows)
        self.tag = tag
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


def _install_hooks(module, tags: dict, recorders: dict, armed: set) -> None:
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
        recorder = _Layer0Recorder(self, scratch, decode_scratch, rows, tag)
        recorders[id(self)] = recorder
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


def main(argv: list[str] | None = None) -> int:
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

    result["profile"] = bind_resident_profile("production")
    result["env"] = {k: v for k, v in os.environ.items() if k.startswith("HIPENGINE_")}

    resident = None
    tp2 = None
    recorders: dict[int, _Layer0Recorder] = {}
    armed: set[int] = set()
    captured: dict[str, _Layer0Recorder] = {}
    initial_states: dict[str, dict] = {}
    weight_hashes: dict[str, dict] = {}
    started = time.perf_counter()
    try:
        resident = create_resident_control(MODEL, capacity=args.capacity, capture_rows=False)
        tp2 = _build_session(args.capacity, rows=args.capacity, schedule="graphed", bulk=True)

        tags = {id(resident.session.runner): "teacher"}
        for device, runner in tp2._runners.items():
            tags[id(runner)] = f"candidate_{device}"

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
        resident.session.prefill(
            prompt, use_bulk=None, bulk_attention_mode="bulk", return_logits=False
        )
        captured["teacher"] = recorders[id(resident.session.runner)]

        # Candidate: opt-in rank-local bulk TP2 prefill on the same prompt.
        tp2.reset()
        for device, runner in tp2._runners.items():
            initial_states[f"candidate_{device}"] = _initial_state_hashes(
                runner.runtime, int(runner.runtime.get_device()), tp2._scratches[device]
            )
            weight_hashes[f"candidate_{device}"] = _weight_hashes(runner, 0)
        tp2.bulk_prefill(prompt)
        for device, runner in tp2._runners.items():
            captured[f"candidate_{device}"] = recorders[id(runner)]

        result["routes"] = {tag: rec.to_json() for tag, rec in captured.items()}
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

        if args.save_dir:
            out = Path(args.save_dir)
            out.mkdir(parents=True, exist_ok=True)
            payload = {}
            for tag, rec in captured.items():
                for name, raw in rec.records.items():
                    payload[f"{tag}.{name}"] = raw
                if rec.input is not None:
                    payload[f"{tag}.input"] = rec.input
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
