"""Accounting tests for the Gemma 4 llama.cpp same-artifact reference bench.

The regression these cover: the comparator's value is its exact accounting —
a response that truncated the prompt, returned fewer tokens than requested, or
reported non-positive timings must be rejected rather than averaged into a
reference number the campaign target is computed from. The decode denominator
must match the hipEngine harness convention: outputs - 1 transitions over the
predicted time.
"""

from __future__ import annotations

import pytest

from scripts.gemma4_llamacpp_reference_bench import reference_row, tokenizer_probe


def _response(**overrides):
    payload = {
        "tokens": list(range(128)),
        "tokens_evaluated": 1024,
        "truncated": False,
        "timings": {
            "cache_n": 0,
            "prompt_n": 1024,
            "prompt_ms": 1600.0,
            "predicted_n": 128,
            "predicted_ms": 1120.0,
        },
    }
    if "timings" in overrides:
        payload["timings"].update(overrides.pop("timings"))
    payload.update(overrides)
    return payload


def test_reference_row_uses_transitions_over_predicted_time():
    row = reference_row(_response(), prompt_ids=list(range(1024)), outputs=128)

    assert row["prompt_tokens"] == 1024
    assert row["generated_tokens"] == 128
    assert row["decode_forwards"] == 127
    assert row["prefill_s"] == pytest.approx(1.6)
    assert row["decode_s"] == pytest.approx(1.12)
    assert row["decode_tps"] == pytest.approx(127 / 1.12)
    assert row["prefill_tps"] == pytest.approx(1024 / 1.6)


def test_reference_row_rejects_truncated_prompt_and_short_output():
    with pytest.raises(ValueError, match="truncated"):
        reference_row(_response(truncated=True), prompt_ids=list(range(1024)), outputs=128)
    with pytest.raises(ValueError, match="tokens_evaluated"):
        reference_row(
            _response(tokens_evaluated=1023), prompt_ids=list(range(1024)), outputs=128
        )
    with pytest.raises(ValueError, match="output"):
        reference_row(
            _response(tokens=list(range(100))), prompt_ids=list(range(1024)), outputs=128
        )
    with pytest.raises(ValueError, match="predicted_n"):
        reference_row(
            _response(**{"timings": {
                "prompt_n": 1024, "prompt_ms": 1600.0,
                "predicted_n": 127, "predicted_ms": 1120.0,
            }}),
            prompt_ids=list(range(1024)),
            outputs=128,
        )


def test_reference_row_rejects_nonpositive_timings():
    with pytest.raises(ValueError, match="timing"):
        reference_row(
            _response(**{"timings": {
                "prompt_n": 1024, "prompt_ms": 0.0,
                "predicted_n": 128, "predicted_ms": 1120.0,
            }}),
            prompt_ids=list(range(1024)),
            outputs=128,
        )


def test_tokenizer_probe_accepts_exact_and_bos_prefixed_spellings():
    ids = [5, 6, 7]
    assert tokenizer_probe([5, 6, 7], ids)["match"] is True
    bos = tokenizer_probe([2, 5, 6, 7], ids)
    assert bos["match"] is True and bos["bos"] == 2


def test_tokenizer_probe_reports_the_first_divergence():
    miss = tokenizer_probe([5, 6, 9], [5, 6, 7])
    assert miss["match"] is False
    assert miss["first_divergence"] == 2
    short = tokenizer_probe([5, 6], [5, 6, 7])
    assert short["match"] is False and short["expected"] == 3
    assert short["first_divergence"] == 2


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_reference_cli_rejects_invalid_timeout_before_loading(value, capsys):
    from scripts.gemma4_llamacpp_reference_bench import main
    with pytest.raises(SystemExit) as raised:
        main(["--request-timeout", value])
    assert raised.value.code == 2
    assert "finite positive seconds" in capsys.readouterr().err


def test_reference_forwards_timeout_to_warmup_and_sample(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from scripts import gemma4_llamacpp_reference_bench as bench
    from hipengine.loading.gguf import GGUFReader
    from hipengine.tokenization.gguf import Gemma4GGUFTokenizer
    from hipengine.util import amdgpu_vram

    ids = [5, 6, 7, 8]
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"fixture")
    used = tmp_path / "used"
    used.write_text("0")
    tokenizer = SimpleNamespace(encode=lambda text: ids, decode=lambda tokens: "text")
    monkeypatch.setattr(GGUFReader, "__init__", lambda self, path: setattr(self, "info", None))
    monkeypatch.setattr(Gemma4GGUFTokenizer, "from_gguf_info", lambda info: tokenizer)
    monkeypatch.setattr(bench, "exact_prompt_ids", lambda tokenize, count: ids)
    monkeypatch.setattr(amdgpu_vram, "select_card", lambda **kw: SimpleNamespace(vram_used_path=used, pci_id="fixture"))
    monkeypatch.setattr(bench.subprocess, "check_output", lambda *a, **kw: "fixture-commit")
    monkeypatch.setattr(bench.subprocess, "Popen", lambda *a, **kw: object())
    monkeypatch.setattr(bench, "binary_provenance", lambda *a: {"version": "fixture"})
    monkeypatch.setattr(bench, "_provenance", lambda *a: {"host": "fixture"})
    monkeypatch.setattr(bench, "_run", lambda *a: "")
    monkeypatch.setattr(bench, "_wait_health", lambda *a: None)
    monkeypatch.setattr(bench, "_stop", lambda *a: output.with_suffix(".log").write_text("n_ctx_slot = 6\n"))
    monkeypatch.setattr(bench, "_tokenize", lambda *a: ids)
    timeouts = []
    def post(base, body, timeout=600.0):
        timeouts.append(timeout)
        return {"tokens_evaluated": 4, "tokens": [1, 2], "timings": {
            "cache_n": 0, "prompt_n": 4,
            "prompt_ms": 100., "predicted_ms": 200., "predicted_n": 2}}
    monkeypatch.setattr(bench, "_post", post)
    output = tmp_path / "result.json"
    assert bench.main(["--artifact", str(artifact), "--prompt", "4", "--output", "2",
                       "--context", "6", "--samples", "1", "--warmup", "1",
                       "--port", "0", "--out", str(output),
                       "--request-timeout", "2400"]) == 0
    assert timeouts == [2400., 2400.]
    recorded = json.loads(output.read_text())
    assert recorded["request_timeout_s"] == 2400.
    assert len(recorded["warmups"]) == len(recorded["samples"]) == 1
    monkeypatch.setattr(bench, "_stop", lambda *a: output.with_suffix(".log").write_text("n_ctx_slot = 256\n"))
    with pytest.raises(ValueError, match="observed context"):
        bench.main(["--artifact", str(artifact), "--prompt", "4", "--output", "2",
                    "--context", "6", "--samples", "1", "--warmup", "0",
                    "--port", "0", "--out", str(output)])


@pytest.mark.parametrize("overrides", [{"prompt_n": 1}, {"cache_n": 1023},
                                        {"prompt_n": None}, {"cache_n": None}])
def test_reference_requires_full_uncached_prefill(overrides):
    response = _response()
    response["timings"].update(overrides)
    with pytest.raises(ValueError, match="prefill"):
        reference_row(response, list(range(1024)), 128)


@pytest.mark.parametrize("field", ["prompt_ms", "predicted_ms"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1600"])
def test_reference_rejects_nonfinite_or_nonnumeric_timings(field, value):
    response = _response()
    response["timings"][field] = value
    with pytest.raises(ValueError, match="timing"):
        reference_row(response, list(range(1024)), 128)


def test_reference_retains_response_output_and_metadata():
    response = _response(content="hello", stop_type="limit",
                         generation_settings={"temperature": 0})
    row = reference_row(response, list(range(1024)), 128)
    assert row["generated_token_ids"] == response["tokens"]
    assert row["content"] == "hello" and row["stop_type"] == "limit"
    assert row["generation_settings"] == {"temperature": 0}
    assert row["timings"] == response["timings"]


@pytest.mark.parametrize("tokens", [[True] * 128, [-1] * 128, ["x"] * 128])
def test_reference_rejects_malformed_ids(tokens):
    with pytest.raises(ValueError, match="token IDs"):
        reference_row(_response(tokens=tokens), list(range(1024)), 128)


def test_reference_records_runtime_capacity_and_graph_fallback():
    from scripts.gemma4_llamacpp_reference_bench import runtime_log_metadata
    log = ("llama_context: n_ctx = 768\nllama_context: n_ctx_per_seq = 768\n"
           "llama_context: flash_attn = enabled\n"
           "llama_kv_cache: HIP0 KV buffer size = 10 MiB (K (bf16): 5 MiB, V (bf16): 5 MiB)\n"
           "ggml_backend_cuda_graph_compute: disabling CUDA graphs due to graph structure\n")
    metadata = runtime_log_metadata(log)
    assert metadata["effective_context"] == 768
    assert metadata["graph_fallback_observed"] is True
    assert metadata["kv_dtype_pairs"] == [["bf16", "bf16"]]
    assert "enabled" in metadata["flash_attention_log"][0]
    assert metadata["graph_enabled"] is None
    observed = runtime_log_metadata("n_ctx_slot = 768\ngraphs reused = 127\ngraphs reused = 253")
    assert observed["effective_context"] == 768
    assert observed["graph_enabled"] is True
    assert observed["graph_reuse_counts"] == [127, 253]


def test_reference_binary_fingerprint_and_embedded_commit(monkeypatch, tmp_path):
    import hashlib
    from scripts import gemma4_llamacpp_reference_bench as bench
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"binary")
    monkeypatch.setattr(bench, "_run", lambda command: "version: build 123, commit 1234567")
    provenance = bench.binary_provenance(binary, "1234567" + "0" * 33)
    assert provenance["sha256"] == hashlib.sha256(b"binary").hexdigest()
    with pytest.raises(ValueError, match="revision mismatch"):
        bench.binary_provenance(binary, "abcdefg")