"""Real HTTP DMS INT8 requests with native NextN speculation and session reuse."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

_MODEL = Path(os.environ.get("HIPENGINE_DMS_MTP_MODEL", "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
_METADATA = Path(os.environ.get("HIPENGINE_DMS_MTP_METADATA", str(Path.home() / "dms-artifacts/qwen38-external-v1/sidecar/dms_metadata.json")))


def _available():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return _MODEL.is_file() and _METADATA.is_file()


pytestmark = pytest.mark.skipif(not _available(), reason="requires ROCm, GGUF model and DMS sidecar")


def test_http_dms_int8_native_mtp_and_following_ar_request():
    from fastapi.testclient import TestClient
    from hipengine.server.api import ServerConfig, create_app

    app = create_app(ServerConfig(
        model=str(_MODEL), backend="hip_gfx1151", execution_profile="production",
        dms_metadata_path=str(_METADATA), kv_storage="int8_per_token_head",
        kv_scale_dtype="fp32", max_context_tokens=2048, speculative_candidate_budget=3,
        eager_load=False, startup_chat_smoke=False, startup_scratch_probe=False,
    ))
    with TestClient(app) as client:
        results = []
        for speculative in (False, True, False):
            response = client.post("/v1/completions", json={
                "model": _MODEL.name, "prompt": "Explain why the sky is blue. " * 96,
                "max_tokens": 16, "temperature": 0, "speculative_mtp": speculative,
            })
            assert response.status_code == 200, response.text
            result = response.json()["choices"][0]["hipengine"]
            dms = result["diagnostics"]["dms"]
            assert dms["backend"]["codec"] == "int8_per_token_head"
            assert dms["backend"]["codec_evaluation_only"] is False
            assert dms["backend"]["device_payloads"] is True
            assert dms["capacity"]["live_token_rows"] < dms["capacity"]["logical_token_rows"]
            accounting = result["diagnostics"].get("specdec2_mtp2", {})
            assert (int(accounting.get("cycles", 0)) > 0) == speculative, result
            assert int(accounting.get("recoverable_failures", 0)) == 0, accounting
            results.append(result["generated_token_ids"])
        assert results[0] == results[1] == results[2]
        ready = client.get("/ready").json()
        policy = ready["dms"]["policy"]
        assert policy["decision_mode"] == "sidecar"
        assert policy["storage"] == "int8_per_token_head"
        assert policy["prefix_cache"] == "off"


def test_library_dms_stream_close_releases_request_for_reuse():
    from hipengine import LLM, SamplingParams

    llm = LLM(
        str(_MODEL), backend="hip_gfx1151", execution_profile="production",
        dms_metadata_path=str(_METADATA), kv_storage="int8_per_token_head",
        kv_scale_dtype="fp32", max_sequence_length=2048,
        speculative_candidate_budget=3,
    )
    prompt = "Explain why the sky is blue. " * 96
    try:
        for speculative in (False, True):
            stream = (
                llm.stream_speculative_mtp_detailed([prompt], SamplingParams(max_tokens=64, temperature=0))
                if speculative else llm.stream_detailed(prompt, SamplingParams(max_tokens=64, temperature=0))
            )
            next(stream)
            stream.close()
            output = llm.generate_detailed([prompt], SamplingParams(max_tokens=4, temperature=0))[0]
            assert len(output.generated_token_ids) == 4
            assert output.telemetry.diagnostics["dms"]["backend"]["codec"] == "int8_per_token_head"
    finally:
        llm.close()
