"""Unit tests for the Gemma 4 BF16 teacher fixture (piece C of the gate packet).

The contract under test is *alignment*: ``prepare`` must reproduce the
packet's row order exactly (that order is what ``execution_profile_gate.py``
indexes ``--bf16-logits`` rows by), and ``finalize`` must reject any raw
capture whose bytes, shape, or schedule do not match the packet it claims to
come from. Argmax disagreement with the packet's teacher labels is a reported
diagnostic, never an error -- the chain is the frozen campaign corpus both
arms replay.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from scripts.gemma4_bf16_teacher import (
    BIN_MAGIC,
    main as teacher_main,
)

VOCAB = 8
REQUESTS = ("a", "b")
ROWS_PER = 3
PROMPT_IDS = {"a": [11, 12, 13], "b": [21, 22, 23, 24]}


def _labels() -> list[int]:
    # In-vocab teacher tokens, unique per row across the packet.
    return [(index % VOCAB) for index in range(len(REQUESTS) * ROWS_PER)]


def _write_packet(tmp_path):
    packet = tmp_path / "packet"
    packet.mkdir()
    labels = _labels()
    rows = []
    for i, req in enumerate(REQUESTS):
        base = i * ROWS_PER
        for t in range(ROWS_PER):
            rows.append(
                {
                    "scenario_id": "sc",
                    "scenario_step": base + t,
                    "request_id": f"prompt-{req}",
                    "teacher_step": t,
                    "teacher_token_id": labels[base + t],
                    "shape": "c1",
                    "transition": "steady" if t else "prefill_to_c1",
                    "category": "smoke",
                    "input_token_id": 5,
                }
            )
    logits = np.full((len(rows), VOCAB), -1.0, dtype=np.float32)
    for r, label in zip(rows, labels):
        logits[r["scenario_step"], label] = 1.0  # argmax == teacher label
    np.save(packet / "strict-logits.npy", logits)
    (packet / "gemma4-c1-smoke-strict-capture.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "scenario_id": "sc",
                "rows": rows,
                "logits_path": "strict-logits.npy",
            }
        )
    )
    (packet / "smoke-env.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "scenario_id": "sc",
                "prompt_tokens_padded": {k: len(v) for k, v in PROMPT_IDS.items()},
                "prompt_tokens_ids": PROMPT_IDS,
            }
        )
    )
    return packet


def _prepare(tmp_path, monkeypatch):
    packet = _write_packet(tmp_path)
    out = tmp_path / "teacher"

    # prepare only reads the softcap from the model header; the artifact
    # itself must not be a test dependency.
    class _Info:
        metadata = {"gemma4.final_logit_softcapping": 30.0}

    monkeypatch.setattr(
        "hipengine.loading.gguf.scan_gguf", lambda path: _Info(), raising=True
    )
    rc = teacher_main(
        [
            "prepare",
            "--packet", str(packet),
            "--output-dir", str(out),
            "--model", str(tmp_path / "fake.gguf"),
        ]
    )
    assert rc == 0
    return packet, out


def _read_bin(path):
    data = path.read_bytes()
    assert data[:4] == BIN_MAGIC
    version, count = struct.unpack_from("<II", data, 4)
    assert version == 1
    offset = 12
    records = []
    for _ in range(count):
        prompt_len, teacher_len = struct.unpack_from("<II", data, offset)
        offset += 8
        prompt = np.frombuffer(data, dtype="<i4", count=prompt_len, offset=offset)
        offset += 4 * prompt_len
        teacher = np.frombuffer(data, dtype="<i4", count=teacher_len, offset=offset)
        offset += 4 * teacher_len
        records.append((prompt.tolist(), teacher.tolist()))
    assert offset == len(data)
    return records


def test_prepare_writes_a_g4tb_bin_aligned_to_packet_rows(tmp_path, monkeypatch):
    packet, out = _prepare(tmp_path, monkeypatch)
    capture_input = json.loads((out / "capture_input.json").read_text())
    assert capture_input["protocol_id"] == "gemma4-bf16-teacher-v1"
    assert capture_input["rows"] == len(REQUESTS) * ROWS_PER
    assert capture_input["vocab_size"] == VOCAB
    assert capture_input["softcap"] == 30.0

    records = _read_bin(out / "teacher-input.bin")
    # Packet row order: request a rows 0..2, request b rows 3..5; the labels
    # in the bin are exactly the packet's teacher_token_id sequence, because
    # raw row i must be the logits at packet row i.
    assert [prompt for prompt, _ in records] == [
        PROMPT_IDS["a"], PROMPT_IDS["b"],
    ]
    flat = [label for _, teacher in records for label in teacher]
    assert flat == capture_input["teacher_labels"]
    assert flat == _labels()
    # Schedule bookkeeping: row_start bands must tile the packet contiguously.
    starts = [s["row_start"] for s in capture_input["schedule"]]
    assert starts == [0, ROWS_PER]
    # The packet files it was built from are pinned for provenance.
    assert capture_input["packet"]["strict_capture_sha256"]
    assert capture_input["packet"]["smoke_env_sha256"]


def test_prepare_refuses_a_packet_without_prompt_ids(tmp_path, monkeypatch):
    packet = _write_packet(tmp_path)
    env = json.loads((packet / "smoke-env.json").read_text())
    del env["prompt_tokens_ids"]
    (packet / "smoke-env.json").write_text(json.dumps(env))
    with pytest.raises(SystemExit, match="prompt_tokens_ids"):
        teacher_main(
            [
                "prepare",
                "--packet", str(packet),
                "--output-dir", str(tmp_path / "out"),
                "--model", str(tmp_path / "fake.gguf"),
            ]
        )


def test_finalize_writes_aligned_fp16_npy_and_manifest(tmp_path, monkeypatch):
    _, out = _prepare(tmp_path, monkeypatch)
    rows = len(REQUESTS) * ROWS_PER
    raw = np.full((rows, VOCAB), -0.5, dtype=np.float32)
    labels = json.loads((out / "capture_input.json").read_text())["teacher_labels"]
    # Row r is the distribution after consuming labels[r]: it predicts
    # labels[r + 1]. The final row predicts the first free sample (no label).
    for i in range(rows - 1):
        raw[i, labels[i + 1]] = 2.0
    raw[rows - 1, labels[0]] = 2.0
    raw_path = tmp_path / "raw.f32"
    raw.tofile(raw_path)

    rc = teacher_main(
        [
            "finalize",
            "--capture-input", str(out / "capture_input.json"),
            "--raw", str(raw_path),
            "--model", str(tmp_path / "fake.gguf"),
            "--llama-revision", "testrev",
        ]
    )
    assert rc == 0
    npy_path = out / "bf16-aligned-logits.npy"
    aligned = np.load(npy_path)
    assert aligned.shape == (rows, VOCAB)
    assert aligned.dtype == np.float16
    # next-label argmax holds on every labeled pair.
    manifest = json.loads((out / "bf16.manifest.json").read_text())
    diag = manifest["argmax_vs_packet_teacher_labels"]
    assert diag["matches"] == rows - 1 and diag["agreement"] == 1.0
    assert diag["rows"] == rows - 1
    assert manifest["softcap"]["violation"] is False
    assert manifest["softcap"]["raw_max_abs"] == pytest.approx(2.0)
    assert manifest["runtime"] == "llama.cpp HIP testrev"
    assert manifest["row_alignment"]["rows"] == rows


def test_finalize_rejects_wrong_raw_bytes_and_non_finite_values(tmp_path, monkeypatch):
    _, out = _prepare(tmp_path, monkeypatch)
    rows = len(REQUESTS) * ROWS_PER

    bad = tmp_path / "short.f32"
    bad.write_bytes(b"\x00" * (rows * VOCAB * 4 - 4))
    with pytest.raises(ValueError, match="raw logits bytes"):
        teacher_main(
            [
                "finalize",
                "--capture-input", str(out / "capture_input.json"),
                "--raw", str(bad),
                "--model", str(tmp_path / "fake.gguf"),
            ]
        )

    nonfinite = np.full((rows, VOCAB), 1.0, dtype=np.float32)
    nonfinite[0, 0] = np.nan
    nf_path = tmp_path / "nan.f32"
    nonfinite.tofile(nf_path)
    with pytest.raises(ValueError, match="non-finite"):
        teacher_main(
            [
                "finalize",
                "--capture-input", str(out / "capture_input.json"),
                "--raw", str(nf_path),
                "--model", str(tmp_path / "fake.gguf"),
            ]
        )


def test_finalize_flags_logits_above_the_model_softcap(tmp_path, monkeypatch):
    _, out = _prepare(tmp_path, monkeypatch)
    rows = len(REQUESTS) * ROWS_PER
    raw = np.full((rows, VOCAB), -1.0, dtype=np.float32)
    raw[0, 0] = 35.0  # above final_logit_softcapping=30 -> uncapped teacher
    raw_path = tmp_path / "uncapped.f32"
    raw.tofile(raw_path)
    rc = teacher_main(
        [
            "finalize",
            "--capture-input", str(out / "capture_input.json"),
            "--raw", str(raw_path),
            "--model", str(tmp_path / "fake.gguf"),
            "--manifest", str(tmp_path / "m.json"),
            "--output", str(tmp_path / "o.npy"),
        ]
    )
    assert rc == 0  # reported, not fatal: the manifest carries the verdict
    manifest = json.loads((tmp_path / "m.json").read_text())
    assert manifest["softcap"]["violation"] is True


def test_gate_command_attaches_bf16_logits_only_when_supplied(tmp_path):
    from scripts.gemma4_control_smoke import _gate_command

    kwargs = dict(
        output_dir=tmp_path,
        strict_capture=tmp_path / "s.json",
        production_capture=tmp_path / "p.json",
        production_fixture=tmp_path / "pf.json",
        strict_fixture=tmp_path / "sf.json",
        isolation_fixture=tmp_path / "iso.json",
        repeat_capture=tmp_path / "rep.json",
        isolation_capture=tmp_path / "isoc.json",
        task_path=tmp_path / "t.json",
        arithmetic_class="T2",
        verdict_path=tmp_path / "v.json",
    )
    plain = _gate_command(**kwargs)
    assert "--bf16-logits" not in plain
    attached = _gate_command(**kwargs, bf16_logits=tmp_path / "bf16.npy")
    index = attached.index("--bf16-logits")
    assert attached[index + 1] == str(tmp_path / "bf16.npy")
    # every other argument is identical
    assert [c for i, c in enumerate(attached) if i not in (index, index + 1)] == plain