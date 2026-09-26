"""Public DMS configuration and resident-route contracts (no GPU required)."""
from types import SimpleNamespace

import pytest

from hipengine import LLM
from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator
from hipengine.generation.qwen35_gguf_mtp2 import _target_verify_mode_for_context
from hipengine.server.api import ServerConfig


def test_llm_accepts_dms_and_defaults_to_single_resident_request():
    llm = LLM("model.gguf", dms_metadata_path="sidecar/dms_metadata.json")
    assert llm.dms_metadata_path == "sidecar/dms_metadata.json"
    assert llm.max_active_requests == 1
    assert llm.prefix_cache == "off"


@pytest.mark.parametrize("kwargs, message", [
    ({"max_active_requests": 2}, "max_active_requests=1"),
    ({"prefix_cache": "radix"}, "prefix"),
    ({"dms_metadata_path": ""}, "non-empty"),
])
def test_llm_rejects_unimplemented_dms_combinations(kwargs, message):
    options = {"dms_metadata_path": "sidecar/dms_metadata.json", **kwargs}
    with pytest.raises(ValueError, match=message):
        LLM("model.gguf", **options)


def test_cli_forwards_dms_and_leaves_ownership_defaults_unset(monkeypatch):
    from hipengine.server.__main__ import build_parser

    monkeypatch.delenv("HIPENGINE_PREFIX_CACHE", raising=False)
    args = build_parser().parse_args(["--model", "model.gguf", "--dms-metadata-path", "sidecar.json"])
    assert args.dms_metadata_path == "sidecar.json"
    assert args.prefix_cache is None


def test_server_config_retains_dms_selection():
    config = ServerConfig(model="model.gguf", dms_metadata_path="sidecar.json",
                          kv_storage="int8_per_token_head")
    assert config.dms_metadata_path == "sidecar.json"
    assert config.max_active_requests == 1


def test_dms_policy_is_not_routed_through_dense_int8_evidence(monkeypatch):
    gen = Qwen35GGUFBringupGenerator.__new__(Qwen35GGUFBringupGenerator)
    gen.configure_dms("sidecar.json")
    monkeypatch.setattr(gen, "_resolve_int8_kv_capability",
                        lambda *args: pytest.fail("dense KV evidence is not a DMS contract"))
    gen._prepare_kv_policy(SimpleNamespace(kv_storage="int8_per_token_head",
                                        kv_scale_dtype="fp32", kv_scale_granularity="per_token_head"))
    kwargs = gen._prepared_session_kv_kwargs()
    assert kwargs["dms_metadata_path"] == "sidecar.json"
    assert kwargs["dms_backend_factory"].__name__ == "create_dms_int8_backend"
    # DMS prefill owns a transient BF16 source, not the dense INT8 retained pool.
    assert "kv_policy" not in kwargs
    assert gen._prepared_kv_signature[0] == "int8_per_token_head"


def test_dms_verifier_selects_transactional_serial_route():
    target = SimpleNamespace(_dms_backend=object())
    assert _target_verify_mode_for_context("native", backend="hip_gfx1151",
                                          end_position=900, target=target) == "serial_exact"


def test_dms_ar_prefill_does_not_pass_a_hidden_sink_request_id():
    from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner

    calls = []
    session = SimpleNamespace(prefill=lambda tokens, **kwargs: calls.append(kwargs) or SimpleNamespace(token_id=42))
    runner = Qwen35GGUFResidentModelRunner.__new__(Qwen35GGUFResidentModelRunner)
    runner._acquire_lease = lambda: SimpleNamespace(session=session)
    runner._begin_mtp2_prompt_streaming = lambda rows: (None,)
    runner._finish_mtp2_prompt_streaming = lambda *args, **kwargs: None
    finished = []
    runner._finish_native_prefill = lambda row, result, **kwargs: finished.append(result.token_id)
    row = SimpleNamespace(lease=None, request=SimpleNamespace(max_tokens=10, deadline_at=None),
                          prompt_ids=(1, 2, 3, 4), request_id=7, mtp2_candidate_budget=0,
                          native_sampled=False, prefill_ms=0, prefill_chunk_count=0)
    runner._prefill_dms_row(row)
    assert finished == [42]
    assert "target_hidden_request_id" not in calls[0]
    assert "target_hidden_chunk_sink" not in calls[0]
    assert session.dms_max_new_tokens >= 10


def test_dms_explicit_mtp_cannot_silently_prefill_without_provider():
    from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner

    runner = Qwen35GGUFResidentModelRunner.__new__(Qwen35GGUFResidentModelRunner)
    runner._acquire_lease = lambda: SimpleNamespace(session=SimpleNamespace())
    runner._begin_mtp2_prompt_streaming = lambda rows: (None,)
    row = SimpleNamespace(lease=None, request=SimpleNamespace(max_tokens=10, deadline_at=None),
                          request_id=7, mtp2_candidate_budget=3, native_sampled=False)
    with pytest.raises(NotImplementedError, match="NextN prompt streaming"):
        runner._prefill_dms_row(row)


def test_dms_journal_maps_engine_request_to_session_local_store_row():
    from hipengine.runtime.qwen35_gguf_mtp import _DMSStoreJournal

    calls = []
    journal = _DMSStoreJournal.__new__(_DMSStoreJournal)
    journal.target = SimpleNamespace(dms_store_request_id=0)
    journal.backend = SimpleNamespace(state_for_request=lambda rid: calls.append(rid) or SimpleNamespace(lease="lease"))
    journal._request_id = None
    journal.bind_request(73)
    assert journal._lease_row().lease == "lease"
    assert calls == [0]
    with pytest.raises(RuntimeError, match="bound to one request"):
        journal.bind_request(74)
