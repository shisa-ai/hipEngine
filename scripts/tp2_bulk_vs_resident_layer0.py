"""Bounded layer-0 comparison: resident TP1 bulk teacher vs TP2 bulk candidate.

Diagnostics only. Builds the optimized resident TP1 bulk control and the opt-in
rank-local bulk TP2 session in the same process under the same production
profile env, feeds both the identical 64-token prompt, and captures the layer-0
linear-attention (GDN) primitive chain from the shared helper
``_run_linear_attention_prefill_attn_rows``:

  attention norm -> QKV/Z projections -> convolution -> GDN output before
  normalization/gating -> normalized/gated output -> output projection
  (``attn_out``), plus the committed conv/recurrent state.

The helper arguments (``linear_state_rows``, ``commit_final_linear_state``,
``hidden_f32_ptr``, ``out_f32_ptr``) and the selected routes are recorded so a
missing argument/state operation is distinguishable from arithmetic drift.

Usage::

    HIP_VISIBLE_DEVICES=0 HIP_PATH=/opt/rocm python3 \
        scripts/tp2_bulk_vs_resident_layer0.py \
        --prompt mixed_ja_en_translate --json /tmp/tp2_layer0.json
"""

from __future__ import annotations

import argparse
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
from tp2_bulk_prefill_first_divergence import _bf16_to_f32, _load_prompt, _rel_err  # noqa: E402
from hipengine.core.memory import copy_device_to_host, scoped_current_device  # noqa: E402


class _RawBuffer:
    def __init__(self, ptr: int, nbytes: int) -> None:
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)


def _read(runtime, ptr_or_buf, nbytes: int | None = None) -> np.ndarray:
    if hasattr(ptr_or_buf, "nbytes"):
        buf = ptr_or_buf
    else:
        buf = _RawBuffer(ptr_or_buf, nbytes)
    host = np.empty(buf.nbytes, dtype=np.uint8)
    copy_device_to_host(host.ctypes.data, buf, buf.nbytes, runtime=runtime)
    return host


_BUFFERS = (
    "norm",
    "linear_qkv",
    "linear_z",
    "conv_out",
    "recurrent_out",
    "recurrent_bf16",
    "attn_out",
)


def _snapshot(runner, scratch, decode_scratch, hidden_ptr, rows, kwargs, capacity):
    runtime = runner.runtime
    record: dict = {
        "rows": int(rows),
        "capacity": int(capacity),
        "args": {
            "linear_state_rows": kwargs.get("linear_state_rows"),
            "commit_final_linear_state": kwargs.get("commit_final_linear_state"),
            "hidden_f32_ptr": kwargs.get("hidden_f32_ptr"),
            "out_f32_ptr": kwargs.get("out_f32_ptr"),
        },
        "input": _read(runtime, hidden_ptr, int(rows) * int(runner.hidden_size) * 2),
        "buffers": {},
    }
    for name in _BUFFERS:
        buf = getattr(scratch, name, None)
        if buf is not None:
            record["buffers"][name] = _read(runtime, buf)
    conv = getattr(decode_scratch, "layer_conv_states", None)
    rec = getattr(decode_scratch, "layer_recurrent_states", None)
    if conv is not None and rec is not None and conv[0] is not None:
        record["state"] = {"conv": _read(runtime, conv[0]), "recurrent": _read(runtime, rec[0])}
    return record


def _compare(a: dict, b: dict) -> dict:
    out: dict = {
        "rows": [a["rows"], b["rows"]],
        "args_equal": a["args"] == b["args"],
        "args": {"teacher": {k: str(v) for k, v in a["args"].items()},
                 "candidate": {k: str(v) for k, v in b["args"].items()}},
    }
    capacity = a["capacity"]
    itemsize = 2

    def first_rows(raw: np.ndarray, rows: int) -> np.ndarray:
        width = len(raw) // (capacity * itemsize)
        return raw[: rows * width * itemsize]

    def cmp_bytes(ka: bytes, kb: bytes) -> dict:
        width = len(ka) // (capacity * itemsize)
        a16 = _bf16_to_f32(ka[: a["rows"] * width * itemsize])
        b16 = _bf16_to_f32(kb[: b["rows"] * width * itemsize])
        diff = np.abs(a16 - b16)
        scale = max(float(np.abs(b16).max()), 1e-9)
        return {
            "rel": float(diff.max() / scale),
            "max_abs": float(diff.max()),
            "rms": float(np.sqrt(np.mean((a16 - b16) ** 2))),
            "bit_equal": bool(np.array_equal(ka, kb)),
        }

    out["input"] = cmp_bytes(a["input"], b["input"])
    out["buffers"] = {
        name: cmp_bytes(a["buffers"][name], b["buffers"][name])
        for name in _BUFFERS
        if name in a["buffers"] and name in b["buffers"]
    }
    if "state" in a and "state" in b:
        out["state"] = {}
        for name in ("conv", "recurrent"):
            fa = a["state"][name].view(np.float32).astype(np.float64)
            fb = b["state"][name].view(np.float32).astype(np.float64)
            diff = np.abs(fa - fb)
            scale = max(float(np.abs(fb).max()), 1e-9)
            out["state"][name] = {
                "rel": float(diff.max() / scale),
                "max_abs": float(diff.max()),
                "bit_equal": bool(np.array_equal(a["state"][name], b["state"][name])),
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
        "protocol": "resident TP1 bulk vs TP2 bulk, layer-0 GDN primitive chain",
    }

    from scripts.tp2_resident_control import bind_resident_profile, create_resident_control
    import hipengine.runtime.qwen35_gguf_runner as qr

    result["profile"] = bind_resident_profile("production")
    result["env"] = {k: v for k, v in os.environ.items() if k.startswith("HIPENGINE_")}

    resident = None
    tp2 = None
    records: dict[str, dict] = {}
    per_layer: dict[str, dict] = {}
    started = time.perf_counter()
    try:
        resident = create_resident_control(MODEL, capacity=args.capacity, capture_rows=False)
        tp2 = _build_session(args.capacity, rows=args.capacity, schedule="graphed", bulk=True)

        tags = {id(resident.session.runner): "teacher"}
        for device, runner in tp2._runners.items():
            tags[id(runner)] = f"candidate_{device}"

        chunk_calls: list = []
        _scratch_cls = type(resident.session._bulk_prefill_scratch)
        _orig_chunk = _scratch_cls.for_chunk

        def _rec_chunk(self, start, rows, total_tokens, *, runtime, stream=0):
            chunk_calls.append([int(start), int(rows), int(total_tokens)])
            return _orig_chunk(self, start, rows, total_tokens, runtime=runtime, stream=stream)

        _scratch_cls.for_chunk = _rec_chunk

        ffn_records: dict = {}
        _orig_norm = qr.Qwen35GGUFFullStackRunner._run_post_attention_norm_residual_rows

        def _cap_norm(self, layer_id, hidden_ptr, attn_out_ptr, scratch, *, rows, **kw):
            res = _orig_norm(
                self, layer_id, hidden_ptr, attn_out_ptr, scratch, rows=rows, **kw
            )
            if int(layer_id) == 0 and id(self) in tags:
                with scoped_current_device(self.runtime, self.runtime.get_device()):
                    ffn_records.setdefault(tags[id(self)], {})["post_norm"] = _read(
                        self.runtime, scratch.post_norm
                    )
                    ffn_records[tags[id(self)]]["residual"] = _read(
                        self.runtime, scratch.residual
                    )
            return res

        qr.Qwen35GGUFFullStackRunner._run_post_attention_norm_residual_rows = _cap_norm

        _orig = qr.Qwen35GGUFFullStackRunner._run_linear_attention_prefill_attn_rows

        def hooked(self, layer_id, hidden_ptr, scratch, *, rows, decode_scratch, **kwargs):
            res = _orig(
                self, layer_id, hidden_ptr, scratch, rows=rows,
                decode_scratch=decode_scratch, **kwargs,
            )
            if id(self) in tags:
                with scoped_current_device(self.runtime, self.runtime.get_device()):
                    snap = _snapshot(
                        self, scratch, decode_scratch, hidden_ptr, rows, kwargs,
                        args.capacity,
                    )
                if int(layer_id) == 0:
                    records[tags[id(self)]] = snap
                per_layer.setdefault(tags[id(self)], {})[int(layer_id)] = {
                    "input": snap["input"],
                    "attn_out": snap["buffers"].get("attn_out"),
                    "rows": snap["rows"],
                }
            return res

        qr.Qwen35GGUFFullStackRunner._run_linear_attention_prefill_attn_rows = hooked

        # Teacher: optimized resident TP1 bulk prefill, exactly the capture path.
        resident.session.reset()
        resident.session.prefill(
            prompt, use_bulk=None, bulk_attention_mode="bulk", return_logits=False
        )

        # Candidate: opt-in rank-local bulk TP2 prefill on the same prompt.
        tp2.reset()
        tp2.bulk_prefill(prompt)

        result["records"] = sorted(records)
        result["resident_chunk_calls"] = chunk_calls[:8]
        if "teacher" in ffn_records and "candidate_0" in ffn_records:
            def cmp_norm(name):
                a = ffn_records["teacher"][name]
                b = ffn_records["candidate_0"][name]
                n = 64 * 5120 * 2
                fa = _bf16_to_f32(a[:n]); fb = _bf16_to_f32(b[:n])
                d = np.abs(fa - fb)
                finite = np.isfinite(d)
                scale = max(float(np.nanmax(np.abs(fb[finite]))), 1e-9) if finite.any() else float("nan")
                return {
                    "rel": float(np.nanmax(d[finite]) / scale) if finite.any() else float("nan"),
                    "max_abs": float(np.nanmax(d[finite])) if finite.any() else float("nan"),
                    "bit_equal": bool(np.array_equal(a[:n], b[:n])),
                }
            result["layer0_norm"] = {name: cmp_norm(name) for name in ("post_norm", "residual")}
        result["layer_types"] = list(tp2._config.layer_types)
        layers = sorted(set(per_layer.get("teacher", {})) & set(per_layer.get("candidate_0", {})))

        def stat(x, y):
            if x.shape != y.shape:
                return {"rel": float("nan"), "max_abs": float("nan"),
                        "shapes": [list(x.shape), list(y.shape)]}
            d = np.abs(x - y)
            finite = np.isfinite(d)
            scale = max(float(np.nanmax(np.abs(y[finite]))), 1e-9) if finite.any() else float("nan")
            return {
                "rel": float(np.nanmax(d[finite]) / scale) if finite.any() else float("nan"),
                "max_abs": float(np.nanmax(d[finite])) if finite.any() else float("nan"),
            }

        def as_row(raw, rows):
            # capacity-sized scratch buffers use max_positions rows
            width = len(raw) // (256 * 2)
            return _bf16_to_f32(raw[: rows * width * 2])

        first = None
        layer_report = []
        for layer_id in layers:
            t = per_layer["teacher"][layer_id]
            c = per_layer["candidate_0"][layer_id]
            entry = {
                "layer_id": layer_id,
                "type": tp2._config.layer_types[layer_id],
                "teacher_rows": t["rows"],
                "candidate_rows": c["rows"],
                "input_bytes": [len(t["input"]), len(c["input"])],
                "input": stat(_bf16_to_f32(t["input"]), _bf16_to_f32(c["input"])),
            }
            if t["attn_out"] is not None and c["attn_out"] is not None:
                entry["attn_out"] = stat(
                    as_row(t["attn_out"], t["rows"]), as_row(c["attn_out"], c["rows"])
                )
            layer_report.append(entry)
            if first is None and entry["input"]["rel"] > 1e-3:
                first = layer_id
        result["first_layer_input_rel_gt_1e-3"] = first
        result["layers"] = layer_report
        if "teacher" in records and "candidate_0" in records:
            result["teacher_vs_candidate0"] = _compare(
                records["teacher"], records["candidate_0"]
            )
        if "candidate_0" in records and "candidate_1" in records:
            result["candidate0_vs_candidate1"] = _compare(
                records["candidate_0"], records["candidate_1"]
            )

        if args.save_dir:
            out = Path(args.save_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.savez(
                out / f"{args.prompt}.layer0.npz",
                **{f"{tag}.{name}": rec["buffers"][name]
                   for tag, rec in records.items() for name in rec["buffers"]},
                **{f"{tag}.input": rec["input"] for tag, rec in records.items()},
            )
    finally:
        if resident is not None:
            resident.close()
        if tp2 is not None:
            tp2.close()

    result["seconds"] = time.perf_counter() - started
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
