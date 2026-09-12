"""Profile provenance must fail closed before expensive AR captures."""

from types import SimpleNamespace
from contextlib import contextmanager

import hipengine
import pytest

from scripts.qwen38_production_ar_gate import profile_identity, profile_session


def test_profile_identity_records_actual_default_and_state_dtype():
    llm = SimpleNamespace(execution_profile_manifest={"execution_profile": "production"},
                          execution_profile_manifest_sha256="abc")
    generator = SimpleNamespace(execution_profile="production",
                                execution_profile_manifest_sha256="abc",
                                execution_profile_fell_back_to_strict=False)
    session = SimpleNamespace(kv_storage_dtype="bf16",
                              runner=SimpleNamespace(fp16_recurrent_state=True))
    result = profile_identity(llm, generator, session, requested=None)
    assert result["requested"] is None
    assert result["resolved"] == "production"
    assert result["fp16_recurrent_state"] is True
    generator.execution_profile_manifest_sha256 = "different"
    with pytest.raises(ValueError, match="manifest"):
        profile_identity(llm, generator, session, requested=None)


def test_profile_identity_rejects_silent_strict_fallback():
    llm = SimpleNamespace(execution_profile_manifest={},
                          execution_profile_manifest_sha256="abc")
    generator = SimpleNamespace(execution_profile="strict",
                                execution_profile_manifest_sha256="abc",
                                execution_profile_fell_back_to_strict=True)
    session = SimpleNamespace(kv_storage_dtype="bf16",
                              runner=SimpleNamespace(fp16_recurrent_state=False))
    with pytest.raises(ValueError, match="profile"):
        profile_identity(llm, generator, session, requested=None)


@pytest.mark.parametrize("fail_capture", [False, True])
def test_profile_session_closes_llm_on_success_and_capture_failure(
    monkeypatch, fail_capture
):
    closed = []
    session = SimpleNamespace(kv_storage_dtype="bf16",
                              runner=SimpleNamespace(fp16_recurrent_state=True))

    @contextmanager
    def lease(**kwargs):
        yield session, False

    generator = SimpleNamespace(
        execution_profile="production", execution_profile_manifest_sha256="abc",
        execution_profile_fell_back_to_strict=False,
        _resident_session_scope=lease, _get_shared_runner=lambda: session.runner)
    llm = SimpleNamespace(
        execution_profile_manifest={"execution_profile": "production"},
        execution_profile_manifest_sha256="abc",
        prepare=lambda **kwargs: None, _get_text_generator=lambda: generator,
        close=lambda: closed.append(True))
    monkeypatch.setattr(hipengine, "LLM", lambda *args, **kwargs: llm)
    args = SimpleNamespace(model="model.gguf", max_sequence_length=2048)
    try:
        with profile_session(args, None) as (actual, identity):
            assert actual is session
            assert identity["resolved"] == "production"
            if fail_capture:
                raise RuntimeError("capture failed")
    except RuntimeError:
        assert fail_capture
    assert closed == [True]
