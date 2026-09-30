#!/usr/bin/env python3
"""BF16 teacher fixture for the Gemma 4 execution-profile packet.

This is the Gemma 4 analogue of ``scripts/quant_quality/qwen36_teacher.py``
for the one input the control-smoke packet lacks: an aligned full-precision
logit cache for ``execution_profile_gate.py --bf16-logits`` (the BF16-relative
non-inferiority section of ``docs/EXECUTION-PROFILES.md`` 6.2).

``prepare``
    Read a completed control-smoke packet (strict capture + smoke-env), take
    each request's padded prompt ids and its scored row chain from the
    packet's own row schedule, and write ``teacher-input.bin`` (magic ``G4TB``)
    plus ``capture_input.json``. The row order of the packet's strict capture
    is the alignment contract: raw row ``i`` in the teacher output must be the
    logits at row ``i`` of that capture.

``finalize``
    Validate the raw float32 logits written by
    ``scripts/quant_quality/gemma4_teacher_logits.cpp``, check them against the
    packet's schedule, and emit the aligned ``bf16-aligned-logits.npy``
    (float16, as the Qwen lane's cache) plus ``bf16.manifest.json``.

Alignment is verified, not assumed: row counts per request, the flat label
sequence, and total shape are recomputed from the packet. Argmax agreement
between the BF16 teacher and the packet's teacher labels is *reported*, never
asserted -- the packet's chain is the frozen campaign corpus both arms
replay, so a BF16 argmax disagreement is a legitimate observation, not a
protocol error.

Large ``.npy`` caches stay outside the repository; the manifest carries the
sha256 of everything it was built from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_ID = "gemma4-bf16-teacher-v1"
BIN_MAGIC = b"G4TB"
BIN_VERSION = 1


def _sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _packet_paths(packet: Path) -> dict[str, Path]:
    paths = {
        "strict_capture": packet / "gemma4-c1-smoke-strict-capture.json",
        "smoke_env": packet / "smoke-env.json",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"packet is missing: {', '.join(missing)}")
    return paths


def _resolve_in(packet: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else packet / path


def _packet_rows(packet: Path) -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[str, Any]]:
    """Return (rows in packet order, prompt ids per request, smoke env).

    Rows are grouped per request in first-seen order and sorted inside a
    request by ``teacher_step``, which must be contiguous from 0; the flat
    label list built from that order is what the raw teacher output aligns to.
    """

    paths = _packet_paths(packet)
    capture = _load_json(paths["strict_capture"])
    env = _load_json(paths["smoke_env"])
    rows = capture.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("strict capture carries no rows")
    prompt_ids = env.get("prompt_tokens_ids")
    if not isinstance(prompt_ids, dict) or not prompt_ids:
        raise SystemExit(
            "smoke-env.json has no prompt_tokens_ids; regenerate the packet "
            "with the current scripts/gemma4_control_smoke.py so the packet "
            "carries its own padded prompt ids"
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["request_id"]), []).append(row)
    ordered_rows: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    for request_id, group in grouped.items():
        steps = sorted(int(r["teacher_step"]) for r in group)
        if steps != list(range(len(steps))):
            raise ValueError(f"{request_id}: teacher_step is not contiguous from 0")
        group.sort(key=lambda r: int(r["teacher_step"]))
        prompt_id = request_id.removeprefix("prompt-")
        if prompt_id not in prompt_ids:
            raise ValueError(f"{request_id}: no padded prompt ids in smoke-env.json")
        prompt = [int(t) for t in prompt_ids[prompt_id]]
        if not prompt:
            raise ValueError(f"{request_id}: empty prompt")
        ordered_rows.extend(group)
        schedule.append(
            {
                "request_id": request_id,
                "prompt_id": prompt_id,
                "prompt_len": len(prompt),
                "rows": len(group),
                "row_start": len(ordered_rows) - len(group),
            }
        )
    # The raw teacher output aligns to the packet's row order, so the grouped
    # concatenation must reproduce it exactly -- interleaved requests would
    # silently shift every row.
    if [id(r) for r in ordered_rows] != [id(r) for r in rows]:
        raise ValueError("packet rows are interleaved across requests; cannot align")
    return ordered_rows, {s["prompt_id"]: [int(t) for t in prompt_ids[s["prompt_id"]]] for s in schedule}, env


def cmd_prepare(args: argparse.Namespace) -> int:
    packet = args.packet.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, prompts_by_id, env = _packet_rows(packet)
    paths = _packet_paths(packet)
    capture = _load_json(paths["strict_capture"])
    # Vocab + dtype come from the packet's own strict logits, memmapped so a
    # 642 MB cache costs only its header here.
    logits_path = _resolve_in(packet, str(capture["logits_path"]))
    logits = np.load(logits_path, mmap_mode="r")
    if logits.ndim != 2 or logits.shape[0] != len(rows):
        raise ValueError(
            f"strict logits {logits.shape} do not match {len(rows)} packet rows"
        )
    vocab = int(logits.shape[1])
    del logits

    bin_path = output_dir / "teacher-input.bin"
    schedule, flat_labels = _write_bin(bin_path, rows, prompts_by_id)

    from hipengine.loading.gguf import scan_gguf

    model_info = scan_gguf(args.model)
    softcap = float(model_info.metadata.get("gemma4.final_logit_softcapping") or 0.0)
    capture_input = {
        "schema": 1,
        "kind": "gemma4_bf16_teacher_capture_input",
        "protocol_id": PROTOCOL_ID,
        "rows": len(rows),
        "vocab_size": vocab,
        "softcap": softcap,
        "teacher_labels": flat_labels,
        "schedule": schedule,
        "packet": {
            "run_id": env.get("run_id"),
            "scenario_id": env.get("scenario_id"),
            "strict_capture": str(paths["strict_capture"]),
            "strict_capture_sha256": _sha256(paths["strict_capture"]),
            "smoke_env_sha256": _sha256(paths["smoke_env"]),
            "logits_path": str(logits_path),
            "logits_sha256": _sha256(logits_path),
        },
        "binary_path": str(bin_path),
        "binary_sha256": _sha256(bin_path),
        "model": str(args.model),
        "rendering": (
            "packet's own padded prompt ids (chat template + per-row text "
            "cycling) and the frozen chain both arms teacher-force, one "
            "label per scored row"
        ),
    }
    capture_input_path = output_dir / "capture_input.json"
    _write_json(capture_input_path, capture_input)
    print(json.dumps({"binary": str(bin_path), "capture_input": str(capture_input_path),
                      "rows": len(rows), "vocab": vocab}))
    return 0


def _write_bin(
    bin_path: Path, rows: list[dict[str, Any]], prompts_by_id: dict[str, list[int]]
) -> tuple[list[dict[str, Any]], list[int]]:
    """Write one G4TB record per request: prompt ids then the full label chain.

    The llama.cpp tool batches ``prompt + teacher[:-1]`` and reads logits at
    positions ``prompt_len - 1 .. prompt_len - 1 + teacher_len - 1`` -- exactly
    the packet's rows (prefill-last, then one row per forced step).
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["request_id"]), []).append(row)
    schedule: list[dict[str, Any]] = []
    flat_labels: list[int] = []
    with bin_path.open("wb") as handle:
        handle.write(BIN_MAGIC)
        handle.write(struct.pack("<II", BIN_VERSION, len(grouped)))
        for request_id, group in grouped.items():
            group.sort(key=lambda r: int(r["teacher_step"]))
            prompt_id = request_id.removeprefix("prompt-")
            prompt = prompts_by_id[prompt_id]
            labels = [int(r["teacher_token_id"]) for r in group]
            handle.write(struct.pack("<II", len(prompt), len(labels)))
            handle.write(np.asarray(prompt, dtype="<i4").tobytes())
            handle.write(np.asarray(labels, dtype="<i4").tobytes())
            schedule.append(
                {
                    "request_id": request_id,
                    "prompt_id": prompt_id,
                    "prompt_len": len(prompt),
                    "teacher_labels": len(labels),
                    "rows": len(group),
                    "row_start": len(flat_labels),
                }
            )
            flat_labels.extend(labels)
    return schedule, flat_labels


def cmd_finalize(args: argparse.Namespace) -> int:
    started = time.time()
    input_path = args.capture_input.resolve()
    capture_input = _load_json(input_path)
    if capture_input.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("capture input protocol differs from gemma4 bf16 teacher tooling")
    rows = int(capture_input["rows"])
    vocab = int(capture_input["vocab_size"])
    labels = [int(x) for x in capture_input["teacher_labels"]]
    if len(labels) != rows:
        raise ValueError(f"capture input labels {len(labels)} != rows {rows}")

    raw_path = args.raw.resolve()
    expected_bytes = rows * vocab * 4
    actual = raw_path.stat().st_size
    if actual != expected_bytes:
        raise ValueError(f"raw logits bytes {actual} != expected {expected_bytes}")

    raw = np.memmap(raw_path, mode="r", dtype=np.float32, shape=(rows, vocab))
    finite = True
    max_abs = 0.0
    for start in range(0, rows, 64):
        chunk = np.asarray(raw[start : start + 64])
        finite = finite and bool(np.isfinite(chunk).all())
        max_abs = max(max_abs, float(np.abs(chunk).max(initial=0.0)))
    if not finite:
        raise ValueError("BF16 teacher raw logits contain non-finite values")

    argmax = np.empty(rows, dtype=np.int64)
    for start in range(0, rows, 64):
        argmax[start : start + 64] = np.argmax(
            np.asarray(raw[start : start + 64]), axis=1
        )
    labels_arr = np.asarray(labels, dtype=np.int64)
    # Row r of the aligned logits is the distribution after consuming
    # labels[r] (the packet capture convention: capture row r predicts
    # labels[r + 1]), so the teacher argmax at row r should equal
    # labels[r + 1]. The final row predicts the first free sample, which has
    # no teacher label; the diagnostic covers the labeled pairs only.
    compared = max(rows - 1, 0)
    matches = int((argmax[:-1] == labels_arr[1:]).sum()) if compared else 0
    softcap = float(capture_input.get("softcap") or 0.0)
    softcap_violation = bool(softcap and max_abs > softcap + 0.05)

    output_dir = input_path.parent
    output_path = args.output.resolve() if args.output else output_dir / "bf16-aligned-logits.npy"
    cache = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float16, shape=(rows, vocab)
    )
    cache[:] = raw
    cache.flush()
    del cache, raw

    model = Path(args.model).resolve()
    manifest = {
        "schema": 1,
        "kind": "gemma4_bf16_teacher_cache",
        "protocol_id": PROTOCOL_ID,
        "name": "Original gemma-4-26B-A4B-it BF16 GGUF / llama.cpp HIP",
        "runtime": f"llama.cpp HIP {args.llama_revision}",
        "model_path": str(model),
        "model_bytes": model.stat().st_size if model.is_file() else None,
        "model_sha256": args.model_sha256,
        "capture_input": str(input_path),
        "capture_input_sha256": _sha256(input_path),
        "logits_path": str(output_path),
        "logits_sha256": _sha256(output_path),
        "shape": [rows, vocab],
        "dtype": "float16",
        "raw_capture": str(raw_path),
        "raw_capture_sha256": _sha256(raw_path),
        "row_alignment": {
            "source": "packet strict-capture row order",
            "rows": rows,
            "requests": len(capture_input["schedule"]),
        },
        "argmax_vs_packet_teacher_labels": {
            "matches": matches,
            "rows": compared,
            "agreement": matches / compared if compared else 0.0,
            "note": (
                "diagnostic only: row r predicts the packet's next label "
                "(labels[r + 1]); the packet's chain is the frozen campaign "
                "corpus both arms replay, so BF16 argmax disagreement is a "
                "legitimate observation, not a protocol failure"
            ),
        },
        "softcap": {
            "config_value": softcap,
            "raw_max_abs": max_abs,
            "violation": softcap_violation,
            "note": "raw_max_abs above the model's final_logit_softcapping means "
                    "the teacher tool did not apply the cap the engine applies",
        },
        "elapsed_seconds_diagnostic": round(time.time() - started, 3),
        "host": platform.node(),
        "teacher_generation": "teacher-forced replay of the packet's chain",
    }
    manifest_path = args.manifest.resolve() if args.manifest else output_dir / "bf16.manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "output": str(output_path),
        "manifest": str(manifest_path),
        "argmax_agreement": manifest["argmax_vs_packet_teacher_labels"]["agreement"],
        "raw_max_abs": max_abs,
        "softcap_violation": softcap_violation,
    }))
    if softcap_violation:
        print(
            "WARNING: teacher logits exceed the model's softcap; the "
            "BF16-relative comparison is suspect until the tool applies it",
            file=sys.stderr,
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="packet -> teacher input bin")
    prep.add_argument("--packet", type=Path, required=True,
                      help="completed control-smoke output directory")
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--model", type=Path, required=True,
                      help="the original BF16/F16 GGUF the teacher runs")
    prep.set_defaults(func=cmd_prepare)

    fin = sub.add_parser("finalize", help="raw f32 logits -> aligned npy + manifest")
    fin.add_argument("--capture-input", type=Path, required=True)
    fin.add_argument("--raw", type=Path, required=True)
    fin.add_argument("--model", type=Path, required=True)
    fin.add_argument("--model-sha256", default=None)
    fin.add_argument("--llama-revision", default="unknown")
    fin.add_argument("--output", type=Path, default=None)
    fin.add_argument("--manifest", type=Path, default=None)
    fin.set_defaults(func=cmd_finalize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())