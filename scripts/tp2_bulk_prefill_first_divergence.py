"""Bounded first-divergence probe for the opt-in bulk TP2 prefill.

Diagnostics only. Compares the TP2 bulk prefill against the token-serial TP2
prefill on the same prompt at the **end of prefill** (before any decode), so
the decode-amplified association difference cannot mask a prefill arithmetic
difference. Also captures per-layer attention output / post-norm / residual /
MLP reduced output for the bulk run so a later step can localize the first
diverging layer against a resident TP1 bulk capture.

Usage::

    HIP_VISIBLE_DEVICES=0,1 HIP_PATH=/opt/rocm python3 \
        scripts/tp2_bulk_prefill_first_divergence.py \
        --prompt mixed_ja_en_translate --json /tmp/tp2_first_div.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tp2_bulk_prefill_diagnostic import MODEL, _build_session  # noqa: E402
from tp2_teacher_coverage_broad import _kl_rows  # noqa: E402
from hipengine.core.memory import copy_device_to_host  # noqa: E402

REFERENCE = "/home/lhl/.cache/hipengine/tp2-d128-baseline/quality-tp1-d0.json"


def _capture_state(session) -> dict:
    """Copy the GDN conv/recurrent state and the full-attention KV cache.

    The prefill-end *logits* only exercise the attention output for the last
    row; the decode reads the whole KV cache and the committed GDN state, so
    those must be compared directly before any claim that a decode divergence
    is only association amplification.
    """
    from hipengine.core.memory import scoped_current_device

    device = session.devices[0]
    scratch = session._scratches[device]
    layers = list(session._config.layer_types)
    captured: dict = {}
    with scoped_current_device(session.runtime, device):
        for layer_id, layer_type in enumerate(layers):
            if layer_type == "linear_attention":
                conv = scratch.layer_conv_states[layer_id]
                recurrent = scratch.layer_recurrent_states[layer_id]
                if conv is None or recurrent is None:
                    continue
                for name, buffer in (("conv", conv), ("recurrent", recurrent)):
                    host = np.empty(buffer.nbytes, dtype=np.uint8)
                    copy_device_to_host(
                        host.ctypes.data, buffer, buffer.nbytes, runtime=session.runtime
                    )
                    captured[f"L{layer_id}.{name}"] = host
            elif layer_type == "full_attention":
                key, value = scratch.full_cache(layer_id)
                for name, buffer in (("key", key), ("value", value)):
                    host = np.empty(buffer.nbytes, dtype=np.uint8)
                    copy_device_to_host(
                        host.ctypes.data, buffer, buffer.nbytes, runtime=session.runtime
                    )
                    captured[f"F{layer_id}.{name}"] = host
    return captured


def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    u16 = raw.view(np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32).astype(np.float64)


class _RawBuffer:
    """Duck-typed buffer for ``copy_device_to_host`` with a raw pointer."""

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)


def _read_raw(session, ptr: int, nbytes: int, device: int) -> np.ndarray:
    host = np.empty(int(nbytes), dtype=np.uint8)
    copy_device_to_host(
        host.ctypes.data, _RawBuffer(ptr, nbytes), nbytes, runtime=session.runtime
    )
    return host


def _state_report(bulk: dict, serial: dict) -> dict:
    report: dict = {}
    for key in sorted(bulk):
        a = bulk[key].view(np.float32) if "recurrent" in key or "conv" in key else None
        # bf16 KV caches: reinterpret as uint16 then widen, compare bit patterns
        # and float values separately.
        if key.endswith(("key", "value")):
            # bf16 KV cache: widen the bf16 bit pattern to f32, then compare
            # as floats (comparing uint16 patterns directly is meaningless).
            a = _bf16_to_f32(bulk[key])
            b = _bf16_to_f32(serial[key])
        else:
            a = bulk[key].view(np.float32).astype(np.float64)
            b = serial[key].view(np.float32).astype(np.float64)
        diff = np.abs(a - b)
        scale = max(float(np.abs(b).max()), 1e-6)
        report[key] = {
            "max_abs": float(diff.max()),
            "rel_err": float(diff.max() / scale),
            "nonzero_bulk": int(np.count_nonzero(a)),
            "nonzero_serial": int(np.count_nonzero(b)),
        }
    return report


def _load_prompt(name: str):
    reference = json.loads(Path(REFERENCE).read_text())
    ids = list(reference["suite"]["ids"])
    index = ids.index(name)
    return [int(t) for t in reference["suite"]["tokens"][index]]


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    scale = max(float(np.abs(b).max()), 1e-6)
    return float(np.abs(a - b).max() / scale)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="mixed_ja_en_translate")
    parser.add_argument("--max-sequence-length", type=int, default=200)
    parser.add_argument("--json", required=True)
    parser.add_argument("--save-dir", default="")
    parser.add_argument(
        "--mlp-per-row",
        action="store_true",
        help=(
            "run the bulk sharded MLP one row at a time through the same "
            "group instead of the batched (rows>1) exchange, to isolate the "
            "batched GEMM reassociation as the first divergence"
        ),
    )
    parser.add_argument(
        "--capture-mlp",
        action="store_true",
        help=(
            "capture the layer-0 sharded MLP input (post-attention norm) and "
            "reduced output for the bulk and token-serial routes and compare "
            "the last row"
        ),
    )
    args = parser.parse_args(argv)

    mlp_records: list[dict] = []
    attn_records: list[dict] = []
    if args.capture_mlp:
        import hipengine.distributed.shard_group as _sg
        import hipengine.runtime.qwen35_gguf_runner as _qr

        _orig_forward = _sg.MlpShardGroup.forward

        def _capturing_forward(self, layer_id, inputs, *, rows=None):
            out = _orig_forward(self, layer_id, inputs, rows=rows)
            if int(layer_id) == 0:
                mlp_records.append(
                    {
                        "rows": int(self._resolve_rows(rows)),
                        "in": dict(inputs),
                        "out": dict(out),
                        "hidden": int(self.hidden),
                        "device": int(self.devices[0]),
                        "group_id": id(self),
                    }
                )
            return out

        _sg.MlpShardGroup.forward = _capturing_forward

        _orig_prefill_attn = _qr.Qwen35GGUFFullStackRunner._run_linear_attention_prefill_attn_rows
        _orig_decode_attn = _qr.Qwen35GGUFFullStackRunner._run_linear_attention_attn_only

        def _cap_prefill_attn(self, layer_id, hidden_ptr, scratch, *, rows, **kw):
            out = _orig_prefill_attn(self, layer_id, hidden_ptr, scratch, rows=rows, **kw)
            if int(layer_id) == 0:
                attn_records.append(
                    {
                        "route": "bulk",
                        "ptr": int(scratch.attn_out.ptr),
                        "rows": int(rows),
                        "hidden": int(self.hidden_size),
                        "device": int(self.runtime.get_device()),
                    }
                )
            return out

        def _cap_decode_attn(self, layer_id, hidden_ptr, attn_out_ptr, scratch, **kw):
            out = _orig_decode_attn(self, layer_id, hidden_ptr, attn_out_ptr, scratch, **kw)
            if int(layer_id) == 0:
                attn_records.append(
                    {
                        "route": "serial",
                        "ptr": int(attn_out_ptr),
                        "rows": 1,
                        "hidden": int(self.hidden_size),
                        "device": int(self.runtime.get_device()),
                    }
                )
            return out

        _qr.Qwen35GGUFFullStackRunner._run_linear_attention_prefill_attn_rows = _cap_prefill_attn
        _qr.Qwen35GGUFFullStackRunner._run_linear_attention_attn_only = _cap_decode_attn

    if args.mlp_per_row:
        import hipengine.distributed.tp2_generate as _tg

        def _per_row(
            group, *, layer_id, rows, post_norm_ptrs, residual_ptrs, out_ptrs, add_residual
        ):
            hidden = int(group.hidden)
            for row in range(int(rows)):
                offset = row * hidden * 2
                inputs = {
                    device: int(post_norm_ptrs[device]) + offset
                    for device in group.devices
                }
                outputs = group.forward(layer_id, inputs, rows=1)
                for device in group.devices:
                    add_residual(
                        device,
                        int(residual_ptrs[device]) + offset,
                        int(outputs[device]),
                        int(out_ptrs[device]) + offset,
                        1,
                    )

        _tg.run_sharded_mlp_with_residual = _per_row
        print("mlp-per-row: installed per-row bulk sharded MLP", flush=True)

    prompt = _load_prompt(args.prompt)
    result: dict = {
        "kind": "tp2_bulk_prefill_first_divergence",
        "model": MODEL,
        "prompt": args.prompt,
        "prompt_tokens": len(prompt),
        "host": platform.node(),
        "command": " ".join(sys.argv),
        "protocol": "compare bulk vs token-serial TP2 at prefill end",
    }

    session = None
    started = time.perf_counter()
    try:
        session = _build_session(
            args.max_sequence_length,
            rows=args.max_sequence_length,
            schedule="graphed",
            bulk=True,
        )
        session._ensure_graph_schedule()

        # 1. Bulk prefill: logits for every prompt row.
        session.reset()
        bulk_logits = np.asarray(session.bulk_prefill(prompt), dtype=np.float64)
        result["bulk_prefill_logits_shape"] = list(bulk_logits.shape)
        result["bulk_prefill_end_sha256"] = hashlib.sha256(
            np.ascontiguousarray(bulk_logits[-1].astype("<f4")).tobytes()
        ).hexdigest()

        # 2. Token-serial TP2 prefill: last token's logits are the prefill end.
        bulk_state = _capture_state(session)
        session.bulk_prefill_enabled = False
        try:
            session.reset()
            serial_last = None
            for position, token in enumerate(prompt):
                serial_last, _ = session._forward_token(
                    int(token), position, kind="prefill"
                )
        finally:
            session.bulk_prefill_enabled = True
        serial_last = np.asarray(serial_last, dtype=np.float64).reshape(-1)
        serial_state = _capture_state(session)
        result["state"] = _state_report(bulk_state, serial_state)
        result["serial_prefill_end_sha256"] = hashlib.sha256(
            np.ascontiguousarray(serial_last.astype("<f4")).tobytes()
        ).hexdigest()

        bulk_last = bulk_logits[-1]
        kl, top1 = _kl_rows(bulk_last[None, :], serial_last[None, :])
        result["prefill_end"] = {
            "kl": float(kl[0]),
            "top1": float(top1[0]),
            "max_abs": float(np.abs(bulk_last - serial_last).max()),
            "rel_err": _rel_err(bulk_last, serial_last),
            "argmax_bulk": int(bulk_last.argmax()),
            "argmax_serial": int(serial_last.argmax()),
        }

        if args.capture_mlp and mlp_records:
            bulk_group = session._bulk_shard_group
            serial_group = session._shard_group
            bulk_rec = next(
                (r for r in mlp_records if r["group_id"] == id(bulk_group)), None
            )
            serial_recs = [
                r for r in mlp_records if r["group_id"] == id(serial_group)
            ]
            if bulk_rec is not None and serial_recs:
                serial_rec = serial_recs[-1]
                from hipengine.core.memory import scoped_current_device

                def _read_bf16(rec):
                    device = rec["device"]
                    rows = rec["rows"]
                    nbytes = rows * rec["hidden"] * 2
                    with scoped_current_device(session.runtime, device):
                        return {
                            name: _read_raw(session, ptr, nbytes, device)
                            for name, ptr in (("in", rec["in"][device]),
                                              ("out", rec["out"][device]))
                        }

                bulk_mlp = _read_bf16(bulk_rec)
                serial_mlp = _read_bf16(serial_rec)
                bulk_row = _bf16_to_f32(bulk_mlp["in"])[-(1) * bulk_rec["hidden"]:]
                serial_row = _bf16_to_f32(serial_mlp["in"])[-(1) * serial_rec["hidden"]:]
                bulk_out = _bf16_to_f32(bulk_mlp["out"])[-(1) * bulk_rec["hidden"]:]
                serial_out = _bf16_to_f32(serial_mlp["out"])[-(1) * serial_rec["hidden"]:]
                result["mlp_layer0"] = {
                    "bulk_rows": bulk_rec["rows"],
                    "serial_rows": serial_rec["rows"],
                    "post_norm_rel": _rel_err(bulk_row, serial_row),
                    "post_norm_max_abs": float(np.abs(bulk_row - serial_row).max()),
                    "mlp_out_rel": _rel_err(bulk_out, serial_out),
                    "mlp_out_max_abs": float(np.abs(bulk_out - serial_out).max()),
                }

            if attn_records:
                from hipengine.core.memory import scoped_current_device

                def _read_attn(rec):
                    nbytes = rec["rows"] * rec["hidden"] * 2
                    with scoped_current_device(session.runtime, rec.get("device", 0)):
                        return _read_raw(session, rec["ptr"], nbytes, 0)

                bulk_attn = next(
                    (
                        r
                        for r in attn_records
                        if r["route"] == "bulk" and r["device"] == 0
                    ),
                    None,
                )
                serial_attns = [
                    r
                    for r in attn_records
                    if r["route"] == "serial" and r["device"] == 0
                ]
                if bulk_attn is not None and serial_attns:
                    serial_attn = serial_attns[-1]
                    a = _bf16_to_f32(_read_attn(bulk_attn))
                    b = _bf16_to_f32(_read_attn(serial_attn))
                    ar = a[-bulk_attn["hidden"]:]
                    br = b[-serial_attn["hidden"]:]
                    result["attn_layer0"] = {
                        "bulk_rows": bulk_attn["rows"],
                        "serial_rows": serial_attn["rows"],
                        "rel_err": _rel_err(ar, br),
                        "max_abs": float(np.abs(ar - br).max()),
                        "bulk_argmax": int(ar.argmax()),
                        "serial_argmax": int(br.argmax()),
                    }

        if args.save_dir:
            out = Path(args.save_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / f"{args.prompt}.bulk_prefill.npy", bulk_logits.astype("<f4"))
            np.save(out / f"{args.prompt}.serial_prefill_end.npy", serial_last.astype("<f4"))
            np.savez(
                out / f"{args.prompt}.state.npz",
                **{f"bulk.{k}": v for k, v in bulk_state.items()},
                **{f"serial.{k}": v for k, v in serial_state.items()},
            )
    finally:
        if session is not None:
            session.close()

    result["seconds"] = time.perf_counter() - started
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
