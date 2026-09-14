"""Evidence-integrity and merge/export guard regressions (CPU only).

Each test pins a defect that was found by review and fixed:

- WER unit: ``_wer`` returns a fraction, ``_wer_pct`` the percentage; the
  100x mix-up understated every reported VibeVoice WER.
- transcript parsing: malformed structured output is reported, never
  silently scored as ordinary word errors.
- lane selection: an unknown ``--systems`` name is rejected instead of
  silently running the bf16 lane under that label.
- GGUF f16 export: bf16 *bit patterns* must be converted numerically, not
  reinterpreted as integers (0x3F80 -> 1.0, not 16256.0).
- GGUF merge: incomplete, overlapping, or non-BF16 inputs are rejected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import vibevoice_asr_gguf_merge as merge  # noqa: E402
from scripts import vibevoice_asr_to_gguf as to_gguf  # noqa: E402
from scripts.vibevoice_asr_wer import (  # noqa: E402
    _transcription_only,
    _wer,
    _wer_content,
    _wer_pct,
    _wer_pct_content,
    parse_transcript,
)


def _has_jiwer() -> bool:
    try:
        import jiwer  # noqa: F401
    except ImportError:
        return False
    return True


# Scoring needs jiwer, which ships with the benchmark extra rather than the
# package; the parser/merge/cache tests below must run without it.
requires_jiwer = pytest.mark.skipif(not _has_jiwer(), reason="needs jiwer")


# --- WER unit -----------------------------------------------------------

@requires_jiwer
def test_wer_is_a_fraction_and_wer_pct_is_its_percentage():
    refs = ["the quick brown fox jumps"]
    hyps = ['[{"Content": "the quick brown fox jumped"}]']
    fraction = _wer(refs, hyps)
    assert fraction == pytest.approx(0.2)
    assert _wer_pct(refs, hyps) == pytest.approx(20.0)
    # The reported-percentage bug: a fraction printed with a '%' sign.
    assert _wer_pct(refs, hyps) == 100.0 * fraction


@requires_jiwer
def test_wer_perfect_transcript_is_zero_in_both_units():
    refs = ["hello world"]
    hyps = ['[{"Content": "hello world"}]']
    assert _wer(refs, hyps) == 0.0
    assert _wer_pct(refs, hyps) == 0.0


def test_extracted_content_is_not_reparsed():
    """_wer takes raw model output; _wer_content takes extracted text.

    Scoring an already-extracted transcript through _wer() re-parses it as
    JSON and raises, which broke the per-clip records.
    """
    raw = '[{"Content": "hello world"}]'
    content, status = parse_transcript(raw)
    assert status == "ok" and content == "hello world"
    with pytest.raises(ValueError):
        _wer(["hello world"], [content])


@requires_jiwer
def test_extracted_content_scores_without_reparsing():
    raw = '[{"Content": "hello world"}]'
    content, _ = parse_transcript(raw)
    assert _wer(["hello world"], [raw]) == 0.0
    assert _wer_content(["hello world"], [content]) == 0.0
    assert _wer_pct_content(["hello world"], [content]) == 0.0


# --- transcript parsing -------------------------------------------------

def test_parse_transcript_reports_status():
    assert parse_transcript('[{"Content": "a b"}]') == ("a b", "ok")
    assert parse_transcript("plain text")[1] == "no_json"
    assert parse_transcript('[{"Content": "a"') [1] == "bad_json"
    assert parse_transcript('{"Content": "a"}')[1] == "no_json"


def test_content_outside_the_array_is_not_ok():
    """Only the assistant role prefix may sit outside the JSON array."""
    good = '[{"Content": "hello world"}]'
    assert parse_transcript(good)[1] == "ok"
    assert parse_transcript("assistant\n" + good)[1] == "ok"
    assert parse_transcript("<|im_start|>assistant\n" + good)[1] == "ok"
    assert parse_transcript("garbage " + good)[1] == "no_json"
    assert parse_transcript(good + " trailing")[1] == "bad_json"
    assert parse_transcript(good + " trailing")[1] != "ok"


def test_malformed_transcript_raises_in_strict_mode():
    with pytest.raises(ValueError, match="malformed VibeVoice transcript"):
        _transcription_only("not json at all")
    with pytest.raises(ValueError, match="malformed VibeVoice transcript"):
        _transcription_only('[{"Content": broken')
    # Lenient mode stays available for diagnostics only.
    assert _transcription_only("not json at all", strict=False) == "not json at all"


def test_unknown_system_name_is_rejected(monkeypatch):
    """An unknown lane must not silently run the bf16 hip lane."""
    from scripts import vibevoice_asr_wer as wer_script

    monkeypatch.setattr(sys, "argv", ["vibevoice_asr_wer.py", "--systems", "hipq4",
                                      "--num-clips", "1"])
    with pytest.raises(SystemExit):
        wer_script.main()


# --- GGUF export dtype conversion --------------------------------------

def test_bf16_bits_convert_numerically_to_f16():
    one = np.array([0x3F80], dtype=np.uint16)  # bf16 bit pattern for 1.0
    assert to_gguf._bf16_bits_to_f32(one)[0] == 1.0
    assert to_gguf._bf16_bits_to_f32(one).astype(np.float16)[0] == np.float16(1.0)
    # The defect being pinned: integer reinterpretation instead of conversion.
    assert one.astype(np.float16)[0] == np.float16(16256.0)


def test_f32_to_bf16_bits_round_trips():
    host = np.array([1.0, -2.5, 0.0, 3.14159], dtype=np.float32)
    bits = to_gguf._f32_to_bf16_bits(host)
    assert bits.dtype == np.uint16
    back = to_gguf._bf16_bits_to_f32(bits)
    assert np.allclose(back, host, rtol=1e-2)


def test_iter_tensors_yields_target_encoding(tmp_path):
    """BF16 checkpoint -> f16 export must yield float16 values, not integers."""
    bits = np.array([0x3F80, 0x4000], dtype=np.uint16)  # 1.0, 2.0
    infos = [("w", "BF16", (2,))]

    class _Info:
        def __init__(self, dtype, shape):
            self.dtype, self.shape = dtype, shape

    class _Index:
        def names_with_prefix(self, prefix):
            return ["w"]

        def require(self, names):
            return [_Info("BF16", (2,))]

    monkeypatch_target = to_gguf
    orig_index, orig_payload = monkeypatch_target.load_weight_index, monkeypatch_target.read_tensor_storage_bytes
    try:
        monkeypatch_target.load_weight_index = lambda snapshot: _Index()
        monkeypatch_target.read_tensor_storage_bytes = lambda info: bits.tobytes()
        out = dict(monkeypatch_target.iter_tensors(Path("/nonexistent"), "f16"))
        assert out["w"].dtype == np.float16
        assert list(out["w"]) == [np.float16(1.0), np.float16(2.0)]
        raw = dict(monkeypatch_target.iter_tensors(Path("/nonexistent"), "bf16"))
        assert raw["w"].dtype == np.uint16
        assert list(raw["w"]) == [0x3F80, 0x4000]
    finally:
        monkeypatch_target.load_weight_index = orig_index
        monkeypatch_target.read_tensor_storage_bytes = orig_payload


# --- GGUF merge validation ---------------------------------------------

class _Tensor:
    def __init__(self, name, ggml_type=12):
        self.name, self.ggml_type = name, ggml_type


class _ModelInfo:
    def __init__(self, tensors, architecture="vibevoice-asr"):
        self.tensors, self.architecture = tensors, architecture


def _backbone_tensors(layers=(0,)):
    names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    for i in layers:
        names += [f"blk.{i}.{part}.weight" for part in merge.REQUIRED_LAYER_PARTS]
    return [_Tensor(n) for n in names]


def _files(tmp_path):
    a = tmp_path / "backbone.gguf"
    b = tmp_path / "full.gguf"
    a.write_bytes(b"backbone")
    b.write_bytes(b"full")
    return str(a), str(b)


def test_merge_accepts_complete_inputs(tmp_path):
    backbone, full = _files(tmp_path)
    merge.validate_inputs(backbone, full, _ModelInfo(_backbone_tensors()),
                          _ModelInfo([_Tensor("ate.blk.0.ffn.weight", 30)]))


def test_merge_rejects_incomplete_backbone(tmp_path):
    backbone, full = _files(tmp_path)
    incomplete = [t for t in _backbone_tensors() if "ffn_down" not in t.name]
    with pytest.raises(SystemExit, match="incomplete"):
        merge.validate_inputs(backbone, full, _ModelInfo(incomplete),
                              _ModelInfo([_Tensor("ate.blk.0.ffn.weight", 30)]))


def test_merge_rejects_missing_globals(tmp_path):
    backbone, full = _files(tmp_path)
    no_globals = [t for t in _backbone_tensors() if not t.name.startswith(("token_embd", "output"))]
    with pytest.raises(SystemExit, match="incomplete"):
        merge.validate_inputs(backbone, full, _ModelInfo(no_globals),
                              _ModelInfo([_Tensor("ate.blk.0.ffn.weight", 30)]))


def test_merge_rejects_overlapping_inputs(tmp_path):
    backbone, full = _files(tmp_path)
    with pytest.raises(SystemExit, match="share"):
        merge.validate_inputs(backbone, full,
                              _ModelInfo(_backbone_tensors() + [_Tensor("ate.x", 30)]),
                              _ModelInfo([_Tensor("ate.x", 30)]))


def test_merge_rejects_non_bf16_encoder(tmp_path):
    backbone, full = _files(tmp_path)
    with pytest.raises(SystemExit, match="not BF16"):
        merge.validate_inputs(backbone, full, _ModelInfo(_backbone_tensors()),
                              _ModelInfo([_Tensor("ate.blk.0.ffn.weight", 1)]))


def test_merge_accepts_llama_backbone_with_vibevoice_export(tmp_path):
    """The real pipeline pairs a llama-quantize backbone with a BF16 export.

    Requiring equal architectures rejected the pipeline's own inputs.
    """
    backbone, full = _files(tmp_path)
    merge.validate_inputs(backbone, full, _ModelInfo(_backbone_tensors(), "llama"),
                          _ModelInfo([_Tensor("ate.blk.0.ffn.weight", 30)]))
    merge.validate_inputs(backbone, full, _ModelInfo(_backbone_tensors(), "vibevoice-asr"),
                          _ModelInfo([_Tensor("ate.blk.0.ffn.weight", 30)]))


def test_merge_rejects_unexpected_backbone_architecture(tmp_path):
    backbone, full = _files(tmp_path)
    with pytest.raises(SystemExit, match="backbone architecture"):
        merge.validate_inputs(backbone, full, _ModelInfo(_backbone_tensors(), "qwen2"),
                              _ModelInfo([_Tensor("ate.blk.0.ffn.weight", 30)]))


def test_merge_rejects_non_vibevoice_full_export(tmp_path):
    backbone, full = _files(tmp_path)
    with pytest.raises(SystemExit, match="full-bf16 architecture"):
        merge.validate_inputs(backbone, full, _ModelInfo(_backbone_tensors(), "llama"),
                              _ModelInfo([_Tensor("ate.blk.0.ffn.weight", 30)], "llama"))


# --- full driver request cache -----------------------------------------

def test_request_cache_revalidates_against_run_parameters(tmp_path):
    """A stale request must not be reused after seed/model/audio changes."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "wer_full", Path(__file__).resolve().parents[1] / "scripts" / "vibevoice_asr_wer_full.py")
    driver = importlib.util.module_from_spec(spec)
    sys.modules["wer_full"] = driver
    spec.loader.exec_module(driver)

    sidecar = tmp_path / "clip.json"
    key = driver._cache_key(20260914, "/model", 256, "audio-hash")
    sidecar.write_text(json.dumps(key))
    assert driver._cache_is_current(sidecar, key)
    for changed in (driver._cache_key(1, "/model", 256, "audio-hash"),
                    driver._cache_key(20260914, "/other", 256, "audio-hash"),
                    driver._cache_key(20260914, "/model", 256, "other-audio"),
                    driver._cache_key(20260914, "/model", 512, "audio-hash")):
        assert not driver._cache_is_current(sidecar, changed)
    assert not driver._cache_is_current(tmp_path / "absent.json", key)
