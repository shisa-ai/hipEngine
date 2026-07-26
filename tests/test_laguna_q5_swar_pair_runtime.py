from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_gguf_runner as runner_module
from scripts import laguna_target_ar_bench as benchmark
from tests._laguna_synthetic import make_laguna_info

_OUTPUT_CONTROL = "wave32x2_fixed_meta_gemv_decode_bf16_bf16_out"
_OUTPUT_CANDIDATE = "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out"
_SHARED_CONTROL = "wave32x2_fixed_meta_gemv_decode_bf16_bf16_out"
_SHARED_CANDIDATE = "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out"
_MIXED_CONTROL = "mixed_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out"
_MIXED_CANDIDATE = (
    "mixed_local32_q5_swar_pair_fixed_meta_gemv_decode_bf16_f32_out"
)
_MIXED_QUANT = "gguf_q5_k+gguf_q6_k+gguf_q6_k+gguf_q5_k"
_CANDIDATE_KEYS = (
    KernelKey("hip_gfx1100", "linear", "gguf_q5_k", _OUTPUT_CANDIDATE),
    KernelKey("hip_gfx1100", "linear_pair", "gguf_q5_k", _SHARED_CANDIDATE),
    KernelKey(
        "hip_gfx1100",
        "attention_projection_quad",
        _MIXED_QUANT,
        _MIXED_CANDIDATE,
    ),
)


def _mock_session_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resource:
        split_gate_fusion = False
        swa_split_wave_local = False

        def free(self, **kwargs) -> None:
            del kwargs

    monkeypatch.setattr(
        runner_module.LagunaGGUFResidentSession,
        "_validate_resident_weights",
        lambda self: None,
    )
    monkeypatch.setattr(
        runner_module,
        "load_laguna_eager_libraries",
        lambda **kwargs: Resource(),
    )
    monkeypatch.setattr(
        runner_module,
        "materialize_laguna_rope_tables",
        lambda *args, **kwargs: Resource(),
    )
    monkeypatch.setattr(
        runner_module,
        "allocate_laguna_kv_cache",
        lambda *args, **kwargs: Resource(),
    )
    monkeypatch.setattr(
        runner_module.LagunaEagerScratch,
        "allocate",
        lambda *args, **kwargs: Resource(),
    )
    monkeypatch.setattr(
        runner_module.LagunaRowsScratch,
        "allocate",
        lambda *args, **kwargs: Resource(),
    )
    monkeypatch.setattr(
        runner_module,
        "resolve_laguna_moe_plan",
        lambda *args, **kwargs: Resource(),
    )
    monkeypatch.setattr(
        runner_module,
        "allocate_laguna_moe_scratch",
        lambda *args, **kwargs: Resource(),
    )


def _session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_q5_swar_pair: bool | None,
    **kwargs,
):
    _mock_session_dependencies(monkeypatch)
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    weights = SimpleNamespace(config=config, backend="hip_gfx1100")
    return runner_module.LagunaGGUFResidentSession(
        resident_weights=weights,
        backend="hip_gfx1100",
        runtime=SimpleNamespace(),
        use_q5_swar_pair=use_q5_swar_pair,
        **kwargs,
    )


def test_q5_swar_pair_runtime_capability_is_default_off_and_gfx1100_only() -> None:
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_Q5_SWAR_PAIR",
        None,
    ) is False
    assert backend_package_capability(
        "hip_gfx1151",
        "LAGUNA_Q5_SWAR_PAIR",
        None,
    ) is None
    resolver = getattr(runner_module, "resolve_laguna_q5_swar_pair", None)
    assert callable(resolver)
    assert not resolver("hip_gfx1100")
    assert resolver("hip_gfx1100", True)
    assert not resolver("hip_gfx1100", False)
    assert not resolver("hip_gfx1151", True)
    assert "use_q5_swar_pair" in inspect.signature(
        runner_module.LagunaGGUFResidentSession
    ).parameters


def test_q5_swar_pair_runtime_requires_all_three_exact_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = getattr(runner_module, "resolve_laguna_q5_swar_pair", None)
    assert callable(resolver)
    assert all(is_registered(key) for key in _CANDIDATE_KEYS)
    assert resolver("hip_gfx1100", True)

    original_is_registered = runner_module.is_registered
    for missing_key in _CANDIDATE_KEYS:
        monkeypatch.setattr(
            runner_module,
            "is_registered",
            lambda key, missing_key=missing_key: (
                False if key == missing_key else original_is_registered(key)
            ),
        )
        assert not resolver("hip_gfx1100", True)


def test_q5_swar_pair_session_owns_all_three_variants_or_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _session(monkeypatch, use_q5_swar_pair=None)
    try:
        assert not control.use_q5_swar_pair
        assert control._q5_output_variant == _OUTPUT_CONTROL
        assert control._q5_shared_pair_variant == _SHARED_CONTROL
        assert control._q5_mixed_variant is None
    finally:
        control.close()

    candidate = _session(monkeypatch, use_q5_swar_pair=True)
    try:
        assert candidate.use_q5_swar_pair
        assert candidate._q5_output_variant == _OUTPUT_CANDIDATE
        assert candidate._q5_shared_pair_variant == _SHARED_CANDIDATE
        assert candidate._q5_mixed_variant == _MIXED_CANDIDATE
        assert candidate._q5_query_gate_variant == (
            "wave32x2_fixed_meta_gemv_decode_bf16_f32_out"
        )
    finally:
        candidate.close()

    rollback = _session(
        monkeypatch,
        use_q5_swar_pair=True,
        use_q5_shared_fixed_meta=False,
    )
    try:
        assert not rollback.use_q5_swar_pair
        assert rollback._q5_output_variant == _OUTPUT_CONTROL
        assert rollback._q5_shared_pair_variant is None
        assert rollback._q5_mixed_variant is None
    finally:
        rollback.close()

    missing_key = _CANDIDATE_KEYS[1]
    original_is_registered = runner_module.is_registered
    monkeypatch.setattr(
        runner_module,
        "is_registered",
        lambda key: False if key == missing_key else original_is_registered(key),
    )
    missing = _session(monkeypatch, use_q5_swar_pair=True)
    try:
        assert not missing.use_q5_swar_pair
        assert missing._q5_output_variant == _OUTPUT_CONTROL
        assert missing._q5_shared_pair_variant == _SHARED_CONTROL
        assert missing._q5_mixed_variant is None
    finally:
        missing.close()


def test_q5_swar_pair_mixed_fallback_and_cli_opt_in_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants: list[str] = []
    accept_candidate = [True]

    def mixed(*args, **kwargs):
        del args
        variant = kwargs["variant"]
        variants.append(variant)
        return variant == _MIXED_CANDIDATE and accept_candidate[0] or (
            variant == _MIXED_CONTROL and not accept_candidate[0]
        )

    monkeypatch.setattr(runner_module, "launch_laguna_mixed_attention_projections", mixed)
    weight = SimpleNamespace(spec=SimpleNamespace(layout="gguf_raw"))

    def launch() -> bool:
        return runner_module.launch_laguna_attention_projections(
            weight,
            weight,
            weight,
            weight,
            10,
            20,
            30,
            40,
            50,
            1,
            3072,
            6144,
            1024,
            1024,
            48,
            backend="hip_gfx1100",
            stream=0,
            libraries=SimpleNamespace(),
            runtime=None,
            use_mixed_q5_q6_attention=True,
            use_mixed_q6_fixed_meta_attention=True,
            use_mixed_local32_fixed_meta_attention=True,
            q5_swar_pair_variant=_MIXED_CANDIDATE,
        )

    assert launch()
    assert variants == [_MIXED_CANDIDATE]
    variants.clear()
    accept_candidate[0] = False
    assert launch()
    assert variants == [_MIXED_CANDIDATE, _MIXED_CONTROL]

    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert not benchmark._parse_args().enable_q5_swar_pair
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-q5-swar-pair"],
    )
    args = benchmark._parse_args()
    assert args.enable_q5_swar_pair

    captured: dict[str, object] = {}

    def session_factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(benchmark, "LagunaGGUFResidentSession", session_factory)
    owner = SimpleNamespace(weights=object(), runtime=object())
    benchmark._session(owner, args)
    assert captured["use_q5_swar_pair"] is True
