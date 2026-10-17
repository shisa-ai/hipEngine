"""Generic speculative-provider admission must key on the resident KV format.

The fixed server KV policy carries INT8 scale metadata (``--kv-scale-dtype``,
default ``fp32`` on the CLI) even when the policy that is actually resident is
BF16.  BF16 KV stores no scales, so the generic ``speculative`` provider route
must not read that unused metadata as a compatibility blocker.  INT8 and
unqualified storage formats keep the existing guards.
"""

from __future__ import annotations

from typing import Any

import pytest

from hipengine import SamplingParams
from hipengine.server import ServerConfig
from hipengine.server.__main__ import build_parser
from hipengine.server.api import (
    ChatCompletionRequest,
    CompletionRequest,
    OpenAIHTTPError,
    _speculative_provider_route_for_request,
)

_CLI_KV_ARGS = (
    "--speculative-provider",
    "qwen4_exp_mtp",
    "--draft-model",
    "/models/mtp.gguf",
    "--speculative-candidate-budget",
    "3",
)


class _AdmissionEngine:
    """Loaded-engine stand-in that only exposes what admission inspects."""

    def __init__(self, effective_kv_storage: str | None) -> None:
        self.effective_kv_storage = effective_kv_storage
        if effective_kv_storage is not None:
            self.kv_capability_provenance = {
                "schema_version": 1,
                "status": "qualified",
                "runtime_action": "admit",
                "promotion_eligible": True,
                "diagnostic_override": False,
                "requested": {"kv_storage": effective_kv_storage},
                "effective_kv_storage": effective_kv_storage,
                "artifact": {"sha256": "a" * 64, "content_verified": True},
                "evidence": None,
                "reason": "admission fixture",
            }

    def generate_speculative_detailed(self, prompts: Any, sampling_params: Any) -> Any:
        raise AssertionError("admission must not execute the provider")


def _config_from_cli(*extra: str, **overrides: Any) -> ServerConfig:
    """Build the server config the CLI produces for these arguments."""

    parsed = build_parser().parse_args(["--model", "target", *_CLI_KV_ARGS, *extra])
    values: dict[str, Any] = {
        "model": parsed.model,
        "served_model_name": "target",
        "speculative_provider": parsed.speculative_provider,
        "draft_model": parsed.draft_model,
        "speculative_candidate_budget": parsed.speculative_candidate_budget,
        "kv_storage": parsed.kv_storage,
        "kv_scale_dtype": parsed.kv_scale_dtype,
        "kv_scale_granularity": parsed.kv_scale_granularity,
    }
    values.update(overrides)
    return ServerConfig(**values)


def _sampling_for(
    config: ServerConfig,
    request: CompletionRequest | ChatCompletionRequest,
    **overrides: Any,
) -> SamplingParams:
    """Build sampling params the way ``create_app``'s closure resolves them."""

    values: dict[str, Any] = {
        "max_tokens": 2,
        "temperature": float(request.temperature or 0.0),
        "top_p": float(request.top_p if request.top_p is not None else 1.0),
        "top_k": int(request.top_k or 0),
        "min_p": float(request.min_p or 0.0),
        "ignore_eos": bool(request.ignore_eos),
        "kv_storage": request.kv_storage or config.kv_storage,
        "kv_scale_dtype": request.kv_scale_dtype or config.kv_scale_dtype,
        "kv_scale_granularity": (
            request.kv_scale_granularity or config.kv_scale_granularity
        ),
    }
    values.update(overrides)
    return SamplingParams(**values)


def _route(
    config: ServerConfig,
    request: CompletionRequest | ChatCompletionRequest,
    *,
    engine: Any,
    sampling: SamplingParams | None = None,
) -> str | None:
    return _speculative_provider_route_for_request(
        config,
        request,
        engine=engine,
        sampling=sampling if sampling is not None else _sampling_for(config, request),
        prompt_count=1,
    )


def _blockers(exc: OpenAIHTTPError) -> list[str]:
    return list(exc.extra["hipengine"]["speculative"]["blockers"])


def _request(**overrides: Any) -> CompletionRequest:
    values: dict[str, Any] = {
        "model": "target",
        "prompt": "one",
        "max_tokens": 2,
        "speculative": True,
    }
    values.update(overrides)
    return CompletionRequest(**values)


def test_bf16_kv_admits_generic_speculation_despite_fp32_scale_metadata() -> None:
    """The reported defect: BF16 KV rejected for unused FP32 scale metadata."""

    config = _config_from_cli("--kv-storage", "bf16")
    # The CLI default really is fp32 even though the dataclass default is fp16.
    assert config.kv_scale_dtype == "fp32"
    assert config.kv_scale_granularity == "per_token_head"

    route = _route(config, _request(), engine=_AdmissionEngine("bf16"))

    assert route == "speculative"


def test_auto_kv_storage_that_resolves_to_bf16_admits_with_cli_scale_defaults() -> None:
    config = _config_from_cli("--kv-storage", "auto")
    assert config.kv_scale_dtype == "fp32"

    route = _route(config, _request(), engine=_AdmissionEngine("bf16"))

    assert route == "speculative"


def test_bf16_admission_does_not_depend_on_the_engine_capability_probe() -> None:
    """No provenance (older/fake engines) must still resolve ``auto`` to BF16."""

    config = _config_from_cli("--kv-storage", "auto")

    route = _route(config, _request(), engine=_AdmissionEngine(None))

    assert route == "speculative"


def test_chat_request_with_bf16_kv_admits_despite_fp32_scale_metadata() -> None:
    config = _config_from_cli("--kv-storage", "bf16")
    request = ChatCompletionRequest(
        model="target", messages=[{"role": "user", "content": "hi"}],
        max_tokens=2, speculative=True,
    )

    route = _route(config, request, engine=_AdmissionEngine("bf16"))

    assert route == "speculative"


def test_int8_resident_kv_keeps_the_fp16_scale_metadata_guard() -> None:
    config = _config_from_cli("--kv-storage", "int8_per_token_head")
    assert config.kv_scale_dtype == "fp32"

    with pytest.raises(OpenAIHTTPError) as excinfo:
        _route(config, _request(), engine=_AdmissionEngine("int8_per_token_head"))

    assert excinfo.value.status_code == 400
    blockers = _blockers(excinfo.value)
    assert "kv_storage" in blockers
    assert "kv_scale_dtype" in blockers


def test_int8_resident_kv_keeps_the_scale_granularity_guard() -> None:
    config = _config_from_cli(
        "--kv-storage",
        "int8_per_token_head",
        "--kv-scale-dtype",
        "fp16",
        "--kv-scale-granularity",
        "hadamard_group32",
    )

    with pytest.raises(OpenAIHTTPError) as excinfo:
        _route(config, _request(), engine=_AdmissionEngine("int8_per_token_head"))

    blockers = _blockers(excinfo.value)
    assert "kv_scale_granularity" in blockers
    assert "kv_scale_dtype" not in blockers


def test_explicit_int8_storage_control_stays_rejected_under_bf16_kv() -> None:
    config = _config_from_cli("--kv-storage", "bf16")

    with pytest.raises(OpenAIHTTPError) as excinfo:
        _route(
            config,
            _request(kv_storage="int8_per_token_head"),
            engine=_AdmissionEngine("bf16"),
        )

    assert "kv_storage" in _blockers(excinfo.value)


def test_unqualified_tail4_storage_control_stays_rejected() -> None:
    config = _config_from_cli("--kv-storage", "bf16")

    with pytest.raises(OpenAIHTTPError) as excinfo:
        _route(
            config,
            _request(kv_storage="tail4_hadamard_group32"),
            engine=_AdmissionEngine("int8_per_token_head"),
        )

    assert "kv_storage" in _blockers(excinfo.value)


@pytest.mark.parametrize(
    ("request_update", "expected_blocker"),
    [
        ({"temperature": 0.2}, "temperature"),
        ({"top_k": 2}, "top_k"),
        ({"top_p": 0.5}, "top_p"),
        ({"min_p": 0.1}, "min_p"),
        ({"n": 2}, "c"),
    ],
)
def test_bf16_admission_still_rejects_unsupported_controls(
    request_update: dict[str, Any],
    expected_blocker: str,
) -> None:
    config = _config_from_cli("--kv-storage", "bf16")

    with pytest.raises(OpenAIHTTPError) as excinfo:
        _route(config, _request(**request_update), engine=_AdmissionEngine("bf16"))

    blockers = _blockers(excinfo.value)
    assert expected_blocker in blockers
    assert "kv_scale_dtype" not in blockers


def test_server_config_dataclass_default_scale_dtype_still_admits_bf16() -> None:
    """Admission must not hinge on which default produced the metadata."""

    config = ServerConfig(
        model="target",
        served_model_name="target",
        speculative_provider="qwen4_exp_mtp",
        draft_model="/models/mtp.gguf",
        speculative_candidate_budget=3,
        kv_storage="bf16",
    )
    assert config.kv_scale_dtype == "fp16"

    assert _route(config, _request(), engine=_AdmissionEngine("bf16")) == "speculative"


@pytest.mark.parametrize("requested_storage", [None, "auto", "bf16"])
def test_effective_int8_resident_kv_rejects_auto_requested_storage(
    requested_storage: str | None,
) -> None:
    """The BF16-only route must reject a resident INT8 engine, not its spelling.

    A qualified INT8 engine reports ``effective_kv_storage=int8_per_token_head``
    while the request may still say ``auto`` (or omit storage entirely) and
    carry valid fp16/per_token_head scales. Those valid scales are not an
    admission argument: this provider route only runs on BF16 KV.
    """

    config = _config_from_cli(
        "--kv-storage",
        "auto",
        "--kv-scale-dtype",
        "fp16",
        "--kv-scale-granularity",
        "per_token_head",
    )
    request = _request(kv_storage=requested_storage)
    sampling = _sampling_for(config, request)
    assert sampling.kv_storage == (requested_storage or "auto")
    assert sampling.kv_scale_dtype == "fp16"
    assert sampling.kv_scale_granularity == "per_token_head"

    with pytest.raises(OpenAIHTTPError) as excinfo:
        _route(
            config,
            request,
            engine=_AdmissionEngine("int8_per_token_head"),
            sampling=sampling,
        )

    assert excinfo.value.status_code == 400
    blockers = _blockers(excinfo.value)
    assert "kv_storage" in blockers
    assert "kv_scale_dtype" not in blockers
    assert "kv_scale_granularity" not in blockers
