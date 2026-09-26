"""Explicit KV requests must not silently change storage."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator


def test_explicit_int8_refusal_reports_reason_instead_of_bf16():
    capability = Mock(runtime_action="fallback_bf16", reason="recorded numerical failure")
    runner = SimpleNamespace(
        model_plugin=SimpleNamespace(kv_capability_declarations=()),
        backend="cpu_reference", target_arch="cpu",
        _kv_weight_quant_key=lambda: "gguf_q4_k_m",
        _resolve_int8_kv_capability=lambda requested: capability,
        _int8_kv_diagnostic_override_enabled=lambda: False,
    )
    with pytest.raises(ValueError, match="explicit.*int8_per_token_head.*recorded numerical failure"):
        Qwen35GGUFBringupGenerator._resolve_request_kv_policy(
            runner, SimpleNamespace(kv_storage="int8_per_token_head"))
